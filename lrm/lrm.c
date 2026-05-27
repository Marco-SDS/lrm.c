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
#include "lrm_bake_texture.h"
#include "lrm_density.h"
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
    if (lrm_triposr_preprocess(m, im, opts->foreground_ratio, img_chw) != 0) {
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
             "density grid %d^3 (coarse-to-fine sample + NeRF MLP)",
             mc_res);
    stage(stage_buf);
    const size_t grid_n = (size_t)mc_res * mc_res * mc_res;
    float *density = (float *)malloc(grid_n * sizeof(float));
    if (!density) {
        free(triplane);
        iris_set_error("lrm_infer: oom for density grid");
        return NULL;
    }
    if (lrm_density_build_sparse(&m->sample_cfg, &m->mlp, triplane, mc_res,
                                 m->radius, thresh, density) != 0) {
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

    if (mc.n_vertices == 0 || mc.n_faces == 0) {
        lrm_mc_mesh_free(&mc);
        free(density); free(triplane);
        iris_set_error("lrm_infer: empty mesh "
                       "(threshold too high?)");
        return NULL;
    }

    /* ----- 6a. Drop disconnected floater components (specks the density
     * field leaves around the main object). Keep anything >= 1% of the
     * largest component's triangle count. */
    stage("floater removal");
    if (lrm_mc_remove_small_components(&mc, 0.01f) != 0) {
        lrm_mc_mesh_free(&mc);
        free(density); free(triplane);
        return NULL;
    }

    /* ----- 6b. Smooth per-vertex normals from the density gradient. The
     * field is still in memory here, so normals come for free; they replace
     * the writer's face-averaged fallback and remove MC stair-stepping from
     * the shading without altering geometry. */
    stage("gradient normals");
    float *vnormals = (float *)malloc((size_t)mc.n_vertices * 3 * sizeof(float));
    if (!vnormals) {
        lrm_mc_mesh_free(&mc);
        free(density); free(triplane);
        iris_set_error("lrm_infer: oom for gradient normals");
        return NULL;
    }
    if (lrm_density_gradient_normals(density, mc_res, -m->radius, +m->radius,
                                     mc.vertices, mc.n_vertices,
                                     mc.faces, mc.n_faces, vnormals) != 0) {
        free(vnormals);
        lrm_mc_mesh_free(&mc);
        free(density); free(triplane);
        return NULL;
    }
    free(density);

    /* ----- 7a. With --bake-texture: rasterize a UV atlas into a PNG. */
    if (opts->bake_texture) {
        int tex_res = (opts->texture_resolution > 0)
                      ? opts->texture_resolution : 2048;
        snprintf(stage_buf, sizeof(stage_buf),
                 "bake texture (%dx%d, atlas rasterization + NeRF MLP)",
                 tex_res, tex_res);
        stage(stage_buf);

        lrm_bake_cfg bcfg = LRM_BAKE_CFG_DEFAULT;
        bcfg.texture_resolution = tex_res;

        float   *atlas_uvs  = NULL;  /* [Nf*3, 2] */
        uint8_t *tex_rgba   = NULL;  /* [tex_res*tex_res*4] */
        if (lrm_bake_texture(&m->sample_cfg, &m->mlp, triplane,
                             mc.vertices, mc.n_vertices,
                             mc.faces, mc.n_faces, &bcfg,
                             &atlas_uvs, &tex_rgba) != 0) {
            free(vnormals);
            lrm_mc_mesh_free(&mc);
            free(triplane);
            return NULL;
        }

        /* PNG-encode the texture in memory via iris_image_encode_png. */
        iris_image img = {
            .width = tex_res,
            .height = tex_res,
            .channels = 4,
            .data = tex_rgba,
        };
        uint8_t *png_bytes = NULL;
        size_t   png_size  = 0;
        if (iris_image_encode_png(&img, &png_bytes, &png_size) != 0) {
            free(vnormals);
            free(atlas_uvs);
            free(tex_rgba);
            lrm_mc_mesh_free(&mc);
            free(triplane);
            iris_set_error("lrm_infer: PNG encode failed");
            return NULL;
        }
        free(tex_rgba);
        tex_rgba = NULL;
        free(triplane);
        triplane = NULL;

        /* Duplicate vertices: each triangle gets its own 3 vertices so
         * each can carry its per-triangle UV. New layout:
         *   new_vertices[3*i + k] = mc.vertices[mc.faces[i, k]]
         *   new_faces[i] = (3*i, 3*i+1, 3*i+2)
         * vertex count = 3 * Nf. */
        const int Nf_saved = mc.n_faces;  /* snapshot before free */
        const int Nv_new   = 3 * Nf_saved;
        float   *new_verts = (float *)malloc((size_t)Nv_new * 3 * sizeof(float));
        float   *new_norms = (float *)malloc((size_t)Nv_new * 3 * sizeof(float));
        int32_t *new_faces = (int32_t *)malloc((size_t)Nf_saved * 3 * sizeof(int32_t));
        if (!new_verts || !new_norms || !new_faces) {
            free(new_verts); free(new_norms); free(new_faces);
            free(vnormals); free(atlas_uvs); free(png_bytes);
            lrm_mc_mesh_free(&mc);
            iris_set_error("lrm_infer: oom for vertex duplication");
            return NULL;
        }
        for (int i = 0; i < Nf_saved; i++) {
            for (int k = 0; k < 3; k++) {
                int src = mc.faces[i * 3 + k];
                int dst = i * 3 + k;
                new_verts[dst * 3 + 0] = mc.vertices[src * 3 + 0];
                new_verts[dst * 3 + 1] = mc.vertices[src * 3 + 1];
                new_verts[dst * 3 + 2] = mc.vertices[src * 3 + 2];
                new_norms[dst * 3 + 0] = vnormals[src * 3 + 0];
                new_norms[dst * 3 + 1] = vnormals[src * 3 + 1];
                new_norms[dst * 3 + 2] = vnormals[src * 3 + 2];
                new_faces[i * 3 + k]   = dst;
            }
        }
        free(vnormals);
        lrm_mc_mesh_free(&mc);  /* original indexed buffers no longer needed */

        lrm_mesh *mesh = lrm_mesh_from_buffers(Nv_new, new_verts,
                                               Nf_saved, new_faces,
                                               /*vertex_colors=*/NULL);
        if (!mesh) {
            free(new_verts); free(new_norms); free(new_faces);
            free(atlas_uvs); free(png_bytes);
            return NULL;
        }
        lrm_mesh_set_normals(mesh, new_norms);
        lrm_mesh_set_texture(mesh, atlas_uvs, png_bytes, png_size);
        stage_done();
        return mesh;
    }

    /* ----- 7b. Default path: vertex color re-query. */
    snprintf(stage_buf, sizeof(stage_buf),
             "vertex color re-query (%d vertices)", mc.n_vertices);
    stage(stage_buf);
    float *vertex_colors = (float *)malloc(
        (size_t)mc.n_vertices * 4 * sizeof(float));
    if (!vertex_colors) {
        free(vnormals);
        lrm_mc_mesh_free(&mc);
        free(triplane);
        iris_set_error("lrm_infer: oom for vertex colors");
        return NULL;
    }
    if (query_vertex_colors(m, triplane, mc.vertices, mc.n_vertices,
                            vertex_colors) != 0) {
        free(vnormals);
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
        free(vnormals);
        free(vertex_colors);
        lrm_mc_mesh_free(&mc);
        return NULL;
    }
    lrm_mesh_set_normals(mesh, vnormals);
    /* lrm_mesh took the buffers; null them in mc so its free() doesn't
     * double-free. */
    mc.vertices = NULL;
    mc.faces    = NULL;
    lrm_mc_mesh_free(&mc);

    stage_done();
    return mesh;
}

/* lrm_mesh_save_glb and lrm_mesh_free live in lrm_mesh_export.c. */
