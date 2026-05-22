/*
 * lrm.c - LRM coordinator: dispatch + end-to-end inference.
 *
 * Phase 12 ships the full pipeline:
 *   image -> preprocess -> DINO encoder -> triplane decoder -> upsample
 *   -> marching-cubes density grid (sampled via triplane_sample + NeRF MLP,
 *      chunked to bound memory)
 *   -> Lorensen-Cline MC
 *   -> vertex color re-query through sample + MLP at vertex positions
 *   -> lrm_mesh.
 *
 * Each individual stage was validated against the canonical PyTorch
 * reference in Phases 4-11; this file is the orchestration glue.
 *
 * When a second model kind lands (OpenLRM in Phase 15), this file gains a
 * dispatch table keyed on a `kind` field in struct lrm_model.
 */

#include "lrm.h"
#include "lrm_mesh_export.h"
#include "lrm_marching_cubes.h"
#include "lrm_nerf_mlp.h"
#include "lrm_triplane_decoder.h"
#include "lrm_triplane_sample.h"
#include "lrm_triplane_upsample.h"
#include "lrm_triposr.h"
#include "lrm_vit_dino.h"

#include "iris.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

lrm_model *lrm_load(const char *model_dir) {
    return lrm_triposr_load(model_dir);
}

void lrm_free(lrm_model *m) {
    lrm_triposr_free(m);
}

/* ========================================================================
 * Inference
 * ======================================================================== */

/* Optional progress / timing reporting. Set LRM_TIMING=1 in the
 * environment to get per-stage walltime to stderr; without the env
 * var only the stage banners print. */
static double clock_ms_internal(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int timing_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("LRM_TIMING");
        cached = (v && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached;
}

static double g_last_stage_ms = 0.0;

static void stage(const char *msg) {
    double now = clock_ms_internal();
    if (timing_enabled() && g_last_stage_ms > 0.0) {
        fprintf(stderr, "lrmc:   (prev stage: %7.1f ms)\n",
                now - g_last_stage_ms);
    }
    fprintf(stderr, "lrmc: %s\n", msg);
    fflush(stderr);
    g_last_stage_ms = now;
}

static void stage_done(void) {
    if (timing_enabled() && g_last_stage_ms > 0.0) {
        fprintf(stderr, "lrmc:   (last stage: %7.1f ms)\n",
                clock_ms_internal() - g_last_stage_ms);
    }
    g_last_stage_ms = 0.0;
}

/* Build the density volume by chunked sample+MLP over the MC grid.
 * Density layout: [R, R, R] C-contiguous matching extract_golden.py's
 * meshgrid('ij') view. Returns 0 on success. */
static int build_density_grid(const lrm_model *m, const float *triplane,
                              int R, float *density_out) {
    const int chunk = 8192;
    const float r = m->radius;
    const size_t N = (size_t)R * R * R;

    /* Per-chunk scratch. */
    float *positions = (float *)malloc((size_t)chunk * 3 * sizeof(float));
    float *features  = (float *)malloc((size_t)chunk * 120 * sizeof(float));
    size_t ts_ws = lrm_triplane_sample_workspace_bytes(&m->sample_cfg, chunk);
    size_t nm_ws = lrm_nerf_mlp_workspace_bytes(&m->mlp, chunk);
    float *ts_work = (float *)malloc(ts_ws);
    float *nm_work = (float *)malloc(nm_ws);
    float *color_scratch = (float *)malloc((size_t)chunk * 3 * sizeof(float));
    if (!positions || !features || !ts_work || !nm_work || !color_scratch) {
        free(positions); free(features); free(ts_work); free(nm_work);
        free(color_scratch);
        iris_set_error("lrm_infer: oom in density-grid scratch");
        return -1;
    }

    /* Precompute linspace. */
    float *lin = (float *)malloc((size_t)R * sizeof(float));
    if (!lin) {
        free(positions); free(features); free(ts_work); free(nm_work);
        free(color_scratch);
        iris_set_error("lrm_infer: oom for linspace");
        return -1;
    }
    for (int i = 0; i < R; i++) {
        lin[i] = -r + (2.0f * r) * (float)i / (float)(R - 1);
    }

    for (size_t off = 0; off < N; off += (size_t)chunk) {
        int n_this = (off + chunk <= N) ? chunk : (int)(N - off);

        for (int q = 0; q < n_this; q++) {
            size_t p = off + (size_t)q;
            size_t i = p / ((size_t)R * R);
            size_t j = (p / R) % R;
            size_t k = p % R;
            float *pos = positions + (size_t)q * 3;
            pos[0] = lin[i];
            pos[1] = lin[j];
            pos[2] = lin[k];
        }

        if (lrm_triplane_sample_forward(&m->sample_cfg, triplane, positions,
                                        n_this, features, ts_work) != 0) {
            goto fail;
        }
        if (lrm_nerf_mlp_forward(&m->mlp, features, n_this,
                                 density_out + off, color_scratch,
                                 nm_work) != 0) {
            goto fail;
        }
    }

    free(lin);
    free(positions); free(features); free(ts_work); free(nm_work);
    free(color_scratch);
    return 0;

fail:
    free(lin);
    free(positions); free(features); free(ts_work); free(nm_work);
    free(color_scratch);
    return -1;
}

/* Re-query NeRF MLP at the mesh vertex positions to get vertex colors.
 * Output is RGBA in [0, 1] (alpha set to 1.0 for all vertices). */
static int query_vertex_colors(const lrm_model *m,
                               const float *triplane,
                               const float *vertices, int n_vertices,
                               float *colors_rgba) {
    const int chunk = 8192;
    float *features = (float *)malloc((size_t)chunk * 120 * sizeof(float));
    float *density  = (float *)malloc((size_t)chunk * sizeof(float));
    float *color    = (float *)malloc((size_t)chunk * 3 * sizeof(float));
    size_t ts_ws = lrm_triplane_sample_workspace_bytes(&m->sample_cfg, chunk);
    size_t nm_ws = lrm_nerf_mlp_workspace_bytes(&m->mlp, chunk);
    float *ts_work = (float *)malloc(ts_ws);
    float *nm_work = (float *)malloc(nm_ws);
    if (!features || !density || !color || !ts_work || !nm_work) {
        free(features); free(density); free(color);
        free(ts_work); free(nm_work);
        iris_set_error("lrm_infer: oom in vertex color scratch");
        return -1;
    }

    for (int off = 0; off < n_vertices; off += chunk) {
        int n_this = (off + chunk <= n_vertices)
                   ? chunk : (n_vertices - off);

        if (lrm_triplane_sample_forward(&m->sample_cfg, triplane,
                                        vertices + (size_t)off * 3,
                                        n_this, features, ts_work) != 0) {
            goto fail;
        }
        if (lrm_nerf_mlp_forward(&m->mlp, features, n_this,
                                 density, color, nm_work) != 0) {
            goto fail;
        }
        /* Pack RGB into RGBA with alpha=1. */
        for (int i = 0; i < n_this; i++) {
            float *dst = colors_rgba + ((size_t)off + i) * 4;
            const float *rgb = color + (size_t)i * 3;
            dst[0] = rgb[0];
            dst[1] = rgb[1];
            dst[2] = rgb[2];
            dst[3] = 1.0f;
        }
    }

    free(features); free(density); free(color);
    free(ts_work); free(nm_work);
    return 0;

fail:
    free(features); free(density); free(color);
    free(ts_work); free(nm_work);
    return -1;
}

lrm_mesh *lrm_infer(lrm_model *m, const iris_image *im,
                    const lrm_infer_opts *opts) {
    if (!m || !im) {
        iris_set_error("lrm_infer: NULL argument");
        return NULL;
    }
    if (!m->modules_ready) {
        iris_set_error("lrm_infer: model sub-modules not initialized");
        return NULL;
    }
    lrm_infer_opts defaults = LRM_INFER_OPTS_DEFAULT;
    if (!opts) opts = &defaults;

    const int mc_res    = opts->mc_resolution > 0 ? opts->mc_resolution
                                                  : defaults.mc_resolution;
    const float thresh  = opts->density_threshold > 0.0f
                          ? opts->density_threshold
                          : m->density_threshold;
    const float radius  = opts->radius > 0.0f ? opts->radius : m->radius;
    /* opts->radius is informational; the sampler uses m->sample_cfg.radius
     * which is already set from m->radius at load. */
    (void)radius;

    /* Reset stage timing for a fresh inference. */
    g_last_stage_ms = 0.0;

    /* ----- 1. Preprocess image to CHW float in [0, 1]. */
    stage("preprocess: foreground composite + 512x512 resize");
    const int cond = m->cond_image_size;
    float *img_chw = (float *)malloc((size_t)3 * cond * cond * sizeof(float));
    if (!img_chw) {
        iris_set_error("lrm_infer: oom for preprocessed image");
        return NULL;
    }
    if (lrm_triposr_preprocess(m, im, img_chw) != 0) {
        free(img_chw);
        return NULL;
    }

    /* ----- 2. DINO encoder forward. */
    stage("DINO ViT-B/16 encoder (1025 tokens x 768)");
    const int n_tokens = m->vit.num_tokens;
    const int dino_hidden = m->vit.hidden_dim;
    float *dino_tokens = (float *)malloc(
        (size_t)n_tokens * dino_hidden * sizeof(float));
    size_t vit_ws_bytes = lrm_vit_dino_workspace_bytes(&m->vit);
    float *vit_ws = (float *)malloc(vit_ws_bytes);
    if (!dino_tokens || !vit_ws) {
        free(dino_tokens); free(vit_ws); free(img_chw);
        iris_set_error("lrm_infer: oom for DINO scratch");
        return NULL;
    }
    if (lrm_vit_dino_forward(&m->vit, img_chw, dino_tokens, vit_ws) != 0) {
        free(dino_tokens); free(vit_ws); free(img_chw);
        return NULL;
    }
    free(vit_ws);
    free(img_chw);

    /* ----- 3. Triplane decoder forward. */
    stage("triplane decoder (16 cross-attn blocks)");
    const int dec_planes = m->decoder.triplane_planes;
    const int dec_ct     = m->decoder.hidden_dim;
    const int dec_size   = m->decoder.triplane_size;
    const size_t dec_out_n =
        (size_t)dec_planes * dec_ct * dec_size * dec_size;
    float *triplane_pre = (float *)malloc(dec_out_n * sizeof(float));
    size_t dec_ws_bytes = lrm_triplane_decoder_workspace_bytes(&m->decoder);
    float *dec_ws = (float *)malloc(dec_ws_bytes);
    if (!triplane_pre || !dec_ws) {
        free(triplane_pre); free(dec_ws); free(dino_tokens);
        iris_set_error("lrm_infer: oom for decoder scratch");
        return NULL;
    }
    if (lrm_triplane_decoder_forward(&m->decoder, dino_tokens, n_tokens,
                                     triplane_pre, dec_ws) != 0) {
        free(triplane_pre); free(dec_ws); free(dino_tokens);
        return NULL;
    }
    free(dec_ws);
    free(dino_tokens);

    /* ----- 4. Post-processor: ConvTranspose2d upsample. */
    stage("post-processor upsample (1024 -> 40 ch, 2x)");
    const int up_ch   = m->upsample.out_ch;       /* 40 */
    const int up_size = m->upsample.out_size;     /* 64 */
    const size_t tri_n =
        (size_t)m->upsample.planes * up_ch * up_size * up_size;
    float *triplane = (float *)malloc(tri_n * sizeof(float));
    size_t up_ws_bytes = lrm_triplane_upsample_workspace_bytes(&m->upsample);
    float *up_ws = (float *)malloc(up_ws_bytes);
    if (!triplane || !up_ws) {
        free(triplane); free(up_ws); free(triplane_pre);
        iris_set_error("lrm_infer: oom for upsample scratch");
        return NULL;
    }
    if (lrm_triplane_upsample_forward(&m->upsample, triplane_pre,
                                      triplane, up_ws) != 0) {
        free(triplane); free(up_ws); free(triplane_pre);
        return NULL;
    }
    free(up_ws);
    free(triplane_pre);

    /* ----- 5. Build density grid for marching cubes. */
    char stage_buf[128];
    snprintf(stage_buf, sizeof(stage_buf),
             "density grid %d^3 = %d queries (sample + NeRF MLP, chunked)",
             mc_res, mc_res * mc_res * mc_res);
    stage(stage_buf);
    const size_t grid_n = (size_t)mc_res * mc_res * mc_res;
    float *density = (float *)malloc(grid_n * sizeof(float));
    if (!density) {
        free(triplane);
        iris_set_error("lrm_infer: oom for density grid");
        return NULL;
    }
    if (build_density_grid(m, triplane, mc_res, density) != 0) {
        free(density); free(triplane);
        return NULL;
    }

    /* ----- 6. Marching cubes. */
    stage("marching cubes");
    lrm_mc_mesh mc = {0};
    if (lrm_marching_cubes_extract(density, mc_res, thresh,
                                   -m->radius, +m->radius, &mc) != 0) {
        free(density); free(triplane);
        return NULL;
    }
    free(density);

    if (mc.n_vertices == 0 || mc.n_faces == 0) {
        lrm_mc_mesh_free(&mc);
        free(triplane);
        iris_set_error("lrm_infer: empty mesh "
                       "(threshold too high?)");
        return NULL;
    }

    /* ----- 7. Vertex color re-query. */
    snprintf(stage_buf, sizeof(stage_buf),
             "vertex color re-query (%d vertices)", mc.n_vertices);
    stage(stage_buf);
    float *vertex_colors = (float *)malloc(
        (size_t)mc.n_vertices * 4 * sizeof(float));
    if (!vertex_colors) {
        lrm_mc_mesh_free(&mc);
        free(triplane);
        iris_set_error("lrm_infer: oom for vertex colors");
        return NULL;
    }
    if (query_vertex_colors(m, triplane, mc.vertices, mc.n_vertices,
                            vertex_colors) != 0) {
        free(vertex_colors);
        lrm_mc_mesh_free(&mc);
        free(triplane);
        return NULL;
    }
    free(triplane);

    /* ----- 8. Pack into lrm_mesh (transfers ownership of the buffers). */
    lrm_mesh *mesh = lrm_mesh_from_buffers(mc.n_vertices, mc.vertices,
                                           mc.n_faces, mc.faces,
                                           vertex_colors);
    if (!mesh) {
        free(vertex_colors);
        lrm_mc_mesh_free(&mc);
        return NULL;
    }
    /* lrm_mesh took the buffers; null them in mc so its free() doesn't
     * double-free. */
    mc.vertices = NULL;
    mc.faces    = NULL;
    lrm_mc_mesh_free(&mc);

    stage_done();
    return mesh;
}

/* lrm_mesh_save_glb and lrm_mesh_free live in lrm_mesh_export.c. */
