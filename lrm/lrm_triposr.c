/*
 * lrm_triposr.c - TripoSR loader, validator, and tensor-tree diagnostic.
 *
 * Phase 5 ships:
 *   - lrm_triposr_load    : mmap the safetensors, fill cached metadata,
 *                           spot-check that the expected layout is intact.
 *   - lrm_triposr_free    : unmap and free.
 *   - lrm_triposr_print_tree : structured dump used by `lrmc info`.
 *
 * Phases 6-10 will plug per-tensor pointers into the forward path; Phase 5
 * intentionally stops at validation so we know the file we just mmap'd is
 * the one we think it is before any compute lands.
 */

#include "lrm_triposr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "iris.h"

/* ========================================================================
 * Path resolution
 * ======================================================================== */

/* Accept either a directory containing model.safetensors, or a direct
 * .safetensors path. Returns a heap-allocated string the caller must free. */
static char *resolve_safetensors_path(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return NULL;
    }

    if (S_ISREG(st.st_mode)) {
        return strdup(path);
    }
    if (S_ISDIR(st.st_mode)) {
        size_t n = strlen(path);
        size_t need = n + strlen("/model.safetensors") + 1;
        char *full = (char *)malloc(need);
        if (!full) return NULL;
        snprintf(full, need, "%s%smodel.safetensors", path,
                 (n > 0 && path[n - 1] == '/') ? "" : "/");
        if (stat(full, &st) != 0) {
            free(full);
            return NULL;
        }
        return full;
    }
    return NULL;
}

/* ========================================================================
 * Validation
 * ======================================================================== */

/* A small set of architectural anchors. If any of these is missing, has the
 * wrong shape, or has the wrong dtype, this is not a TripoSR checkpoint we
 * can handle. Picked to cover every major sub-module exactly once. */
typedef struct {
    const char *name;
    int ndim;
    int64_t shape[4];
} expected_t;

static const expected_t kExpectedAnchors[] = {
    /* DINO ViT-B/16 sentinel: CLS token confirms hidden=768. */
    { "image_tokenizer.model.embeddings.cls_token",                3, {1, 1, 768, 0} },

    /* Triplane learned queries [3 planes x 1024 ch x 32 x 32]. */
    { "tokenizer.embeddings",                                      4, {3, 1024, 32, 32} },

    /* Backbone final norm + projections + block-0 anchors. */
    { "backbone.norm.weight",                                      1, {1024, 0, 0, 0} },
    { "backbone.proj_in.weight",                                   2, {1024, 1024, 0, 0} },
    { "backbone.proj_out.weight",                                  2, {1024, 1024, 0, 0} },

    /* Block-0 self-attn and cross-attn: cross-attn K/V come from image
     * tokens (768), so to_k / to_v shapes must be [1024, 768]. */
    { "backbone.transformer_blocks.0.attn1.to_q.weight",           2, {1024, 1024, 0, 0} },
    { "backbone.transformer_blocks.0.attn2.to_k.weight",           2, {1024, 768, 0, 0} },
    { "backbone.transformer_blocks.0.attn2.to_v.weight",           2, {1024, 768, 0, 0} },

    /* Block-0 GEGLU FFN. proj outputs 2*4096=8192 (gate + value halves);
     * net.2 projects 4096 -> 1024. */
    { "backbone.transformer_blocks.0.ff.net.0.proj.weight",        2, {8192, 1024, 0, 0} },
    { "backbone.transformer_blocks.0.ff.net.2.weight",             2, {1024, 4096, 0, 0} },

    /* Last block exists (catches truncated checkpoints early). */
    { "backbone.transformer_blocks.15.norm3.weight",               1, {1024, 0, 0, 0} },

    /* Post-processor ConvTranspose2d: in=1024, out=40, kernel 2x2. */
    { "post_processor.upsample.weight",                            4, {1024, 40, 2, 2} },
    { "post_processor.upsample.bias",                              1, {40, 0, 0, 0} },

    /* NeRF MLP head: first and last linear layers pin input/output sizes. */
    { "decoder.layers.0.weight",                                   2, {64, 120, 0, 0} },
    { "decoder.layers.18.weight",                                  2, {4, 64, 0, 0} },
};
#define N_EXPECTED ((int)(sizeof(kExpectedAnchors) / sizeof(kExpectedAnchors[0])))

/* TripoSR's full tensor count at HF revision 5b521936. We treat a different
 * count as a fatal mismatch (slack would just hide bugs). */
#define TRIPOSR_EXPECTED_TENSOR_COUNT 549

static int shape_equal(const safetensor_t *t, const expected_t *e) {
    if (t->ndim != e->ndim) return 0;
    for (int i = 0; i < e->ndim; i++) {
        if (t->shape[i] != e->shape[i]) return 0;
    }
    return 1;
}

static int validate_anchors(const safetensors_file_t *sf) {
    char err[512];
    for (int i = 0; i < N_EXPECTED; i++) {
        const expected_t *e = &kExpectedAnchors[i];
        const safetensor_t *t = safetensors_find(sf, e->name);
        if (t == NULL) {
            snprintf(err, sizeof(err),
                     "TripoSR validation: missing tensor '%s'", e->name);
            iris_set_error(err);
            return -1;
        }
        if (t->dtype != DTYPE_F32) {
            snprintf(err, sizeof(err),
                     "TripoSR validation: '%s' has dtype %d, expected f32",
                     e->name, (int)t->dtype);
            iris_set_error(err);
            return -1;
        }
        if (!shape_equal(t, e)) {
            /* Build a readable mismatch message. */
            char got[64] = "[";
            for (int k = 0; k < t->ndim; k++) {
                char one[24];
                snprintf(one, sizeof(one), "%lld%s",
                         (long long)t->shape[k], k + 1 < t->ndim ? "," : "");
                strncat(got, one, sizeof(got) - strlen(got) - 1);
            }
            strncat(got, "]", sizeof(got) - strlen(got) - 1);
            char want[64] = "[";
            for (int k = 0; k < e->ndim; k++) {
                char one[24];
                snprintf(one, sizeof(one), "%lld%s",
                         (long long)e->shape[k], k + 1 < e->ndim ? "," : "");
                strncat(want, one, sizeof(want) - strlen(want) - 1);
            }
            strncat(want, "]", sizeof(want) - strlen(want) - 1);
            snprintf(err, sizeof(err),
                     "TripoSR validation: '%s' shape %s, expected %s",
                     e->name, got, want);
            iris_set_error(err);
            return -1;
        }
    }
    if (sf->num_tensors != TRIPOSR_EXPECTED_TENSOR_COUNT) {
        snprintf(err, sizeof(err),
                 "TripoSR validation: %d tensors, expected %d",
                 sf->num_tensors, TRIPOSR_EXPECTED_TENSOR_COUNT);
        iris_set_error(err);
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Loader
 * ======================================================================== */

struct lrm_model *lrm_triposr_load(const char *model_dir) {
    if (model_dir == NULL || model_dir[0] == '\0') {
        iris_set_error("lrm_triposr_load: model_dir is empty");
        return NULL;
    }

    char *path = resolve_safetensors_path(model_dir);
    if (path == NULL) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "lrm_triposr_load: cannot find 'model.safetensors' at '%s'",
                 model_dir);
        iris_set_error(buf);
        return NULL;
    }

    safetensors_file_t *sf = safetensors_open(path);
    free(path);
    if (sf == NULL) {
        iris_set_error("lrm_triposr_load: safetensors_open failed");
        return NULL;
    }

    if (validate_anchors(sf) != 0) {
        safetensors_close(sf);
        return NULL;
    }

    struct lrm_model *m = (struct lrm_model *)calloc(1, sizeof(struct lrm_model));
    if (m == NULL) {
        safetensors_close(sf);
        iris_set_error("lrm_triposr_load: out of memory");
        return NULL;
    }

    m->st = sf;
    /* Architectural constants matching the pinned config.yaml. */
    m->cond_image_size       = 512;
    m->dino_hidden           = 768;
    m->dino_layers           = 12;
    m->backbone_hidden       = 1024;
    m->backbone_layers       = 16;
    m->backbone_kv_dim       = 768;
    m->triplane_planes       = 3;
    m->triplane_channels     = 40;
    m->triplane_size         = 64;
    m->decoder_linear_layers = 10;
    m->radius                = 0.87f;
    m->density_threshold     = 25.0f;

    /* Initialize all sub-modules (Phase 12). On any failure, free what's
     * been built and return NULL. */
    if (lrm_vit_dino_init(&m->vit, sf, m->cond_image_size) != 0) {
        lrm_triposr_free(m);
        return NULL;
    }
    if (lrm_triplane_decoder_init(&m->decoder, sf) != 0) {
        lrm_vit_dino_release(&m->vit);
        safetensors_close(sf);
        free(m);
        return NULL;
    }
    if (lrm_triplane_upsample_init(&m->upsample, sf) != 0) {
        lrm_triplane_decoder_release(&m->decoder);
        lrm_vit_dino_release(&m->vit);
        safetensors_close(sf);
        free(m);
        return NULL;
    }
    if (lrm_nerf_mlp_init(&m->mlp, sf) != 0) {
        lrm_triplane_upsample_release(&m->upsample);
        lrm_triplane_decoder_release(&m->decoder);
        lrm_vit_dino_release(&m->vit);
        safetensors_close(sf);
        free(m);
        return NULL;
    }
    m->sample_cfg.planes   = m->triplane_planes;
    m->sample_cfg.channels = m->triplane_channels;
    m->sample_cfg.size     = m->triplane_size;
    m->sample_cfg.radius   = m->radius;
    m->modules_ready = 1;
    return m;
}

void lrm_triposr_free(struct lrm_model *m) {
    if (m == NULL) return;
    if (m->modules_ready) {
        /* lrm_nerf_mlp has no owned allocations; nothing to release. */
        lrm_triplane_upsample_release(&m->upsample);
        lrm_triplane_decoder_release(&m->decoder);
        lrm_vit_dino_release(&m->vit);
    }
    if (m->st) safetensors_close(m->st);
    free(m);
}

/* ========================================================================
 * Tensor tree printer
 * ======================================================================== */

static const char *dtype_name(safetensor_dtype_t d) {
    switch (d) {
        case DTYPE_F32:  return "f32";
        case DTYPE_F16:  return "f16";
        case DTYPE_BF16: return "bf16";
        case DTYPE_I32:  return "i32";
        case DTYPE_I64:  return "i64";
        case DTYPE_BOOL: return "bool";
        default:         return "?";
    }
}

static void print_one(FILE *out, const safetensor_t *t) {
    fprintf(out, "  %-72s %-4s  [", t->name, dtype_name(t->dtype));
    for (int i = 0; i < t->ndim; i++) {
        fprintf(out, "%lld%s", (long long)t->shape[i],
                i + 1 < t->ndim ? "," : "");
    }
    fprintf(out, "]  %zu B\n", t->data_size);
}

/* Return 1 if `name` starts with `prefix.` (or equals `prefix`). */
static int has_prefix(const char *name, const char *prefix) {
    size_t pl = strlen(prefix);
    if (strncmp(name, prefix, pl) != 0) return 0;
    return name[pl] == '\0' || name[pl] == '.';
}

void lrm_triposr_print_tree(const struct lrm_model *m, FILE *out) {
    if (m == NULL || m->st == NULL) return;
    const safetensors_file_t *sf = m->st;

    fprintf(out, "model: %s\n", sf->path);
    fprintf(out, "  file size:      %zu bytes (%.2f GB)\n",
            sf->file_size, (double)sf->file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(out, "  header size:    %zu bytes\n", sf->header_size);
    fprintf(out, "  total tensors:  %d\n", sf->num_tensors);
    fprintf(out, "  kind:           TripoSR (single-image -> 3D)\n");
    fprintf(out, "  config:        cond_image=%d, dino_layers=%d hidden=%d,"
                 " backbone_layers=%d hidden=%d,\n"
                 "                 triplane=%dx%dch %dx%d, decoder %d linear,"
                 " radius=%.2f, threshold=%.1f\n\n",
            m->cond_image_size, m->dino_layers, m->dino_hidden,
            m->backbone_layers, m->backbone_hidden,
            m->triplane_planes, m->triplane_channels,
            m->triplane_size, m->triplane_size,
            m->decoder_linear_layers, m->radius, m->density_threshold);

    static const char *kGroups[] = {
        "image_tokenizer",
        "tokenizer",
        "backbone",
        "post_processor",
        "decoder",
        NULL
    };

    for (int g = 0; kGroups[g]; g++) {
        const char *prefix = kGroups[g];
        int count = 0;
        size_t bytes = 0;
        for (int i = 0; i < sf->num_tensors; i++) {
            if (has_prefix(sf->tensors[i].name, prefix)) {
                count++;
                bytes += sf->tensors[i].data_size;
            }
        }
        fprintf(out, "[%s]  %d tensors, %.2f MB\n",
                prefix, count, (double)bytes / (1024.0 * 1024.0));
        for (int i = 0; i < sf->num_tensors; i++) {
            if (has_prefix(sf->tensors[i].name, prefix)) {
                print_one(out, &sf->tensors[i]);
            }
        }
        fprintf(out, "\n");
    }

    /* Catch anything outside the known groups (would indicate a non-TripoSR
     * payload sneaking past validation). */
    int orphans = 0;
    for (int i = 0; i < sf->num_tensors; i++) {
        const char *n = sf->tensors[i].name;
        int known = 0;
        for (int g = 0; kGroups[g] && !known; g++) {
            if (has_prefix(n, kGroups[g])) known = 1;
        }
        if (!known) orphans++;
    }
    if (orphans > 0) {
        fprintf(out, "[unknown prefix]  %d tensors\n", orphans);
        for (int i = 0; i < sf->num_tensors; i++) {
            const char *n = sf->tensors[i].name;
            int known = 0;
            for (int g = 0; kGroups[g] && !known; g++) {
                if (has_prefix(n, kGroups[g])) known = 1;
            }
            if (!known) print_one(out, &sf->tensors[i]);
        }
    }
}

/* ========================================================================
 * Phase 12: Image preprocessing
 * ======================================================================== */

#include <math.h>

int lrm_triposr_preprocess(const struct lrm_model *m,
                           const iris_image *im,
                           float fg_ratio,
                           float *out_chw) {
    if (!m || !im || !im->data || !out_chw) {
        iris_set_error("preprocess: NULL argument");
        return -1;
    }
    if (im->channels != 3 && im->channels != 4) {
        iris_set_error("preprocess: image must be RGB or RGBA");
        return -1;
    }

    const int target = m->cond_image_size;     /* 512 */
    /* Clamp to a sane range; <= 0 means "use the TripoSR default". */
    if (fg_ratio <= 0.0f) fg_ratio = 0.85f;
    if (fg_ratio > 1.0f)  fg_ratio = 1.0f;
    const int H = im->height;
    const int W = im->width;
    const int C = im->channels;
    const uint8_t *src = im->data;

    /* ---- Step 1: alpha bbox (only if RGBA with non-uniform alpha). */
    int y1, y2, x1, x2;
    int has_alpha_mask = 0;
    if (C == 4) {
        /* Check if any pixel is non-opaque -- TripoSR's
         * remove_background() short-circuits if alpha is uniform 255. */
        for (int y = 0; y < H && !has_alpha_mask; y++) {
            for (int x = 0; x < W; x++) {
                if (src[(y * W + x) * 4 + 3] < 255) {
                    has_alpha_mask = 1;
                    break;
                }
            }
        }
    }

    if (has_alpha_mask) {
        y1 = H; y2 = -1; x1 = W; x2 = -1;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (src[(y * W + x) * 4 + 3] > 0) {
                    if (y < y1) y1 = y;
                    if (y > y2) y2 = y;
                    if (x < x1) x1 = x;
                    if (x > x2) x2 = x;
                }
            }
        }
        if (y2 < y1 || x2 < x1) {
            iris_set_error("preprocess: empty alpha mask");
            return -1;
        }
    } else {
        /* No alpha info: treat the whole image as foreground (caller is
         * responsible for having pre-cleaned it). */
        y1 = 0; y2 = H - 1; x1 = 0; x2 = W - 1;
    }

    const int fg_h = y2 - y1 + 1;
    const int fg_w = x2 - x1 + 1;
    const int sq   = (fg_h > fg_w) ? fg_h : fg_w;
    const int canvas = (int)((float)sq / fg_ratio);
    const int off_y  = (sq - fg_h) / 2 + (canvas - sq) / 2;
    const int off_x  = (sq - fg_w) / 2 + (canvas - sq) / 2;

    /* ---- Step 2: build the canvas RGB image in float, with the alpha
     *               composite over gray 0.5 applied at canvas resolution
     *               (matches Python's tsr/utils.py + run.py exactly:
     *                composite happens BEFORE the resize to 512). */
    float *canvas_rgb = (float *)malloc((size_t)canvas * canvas * 3 * sizeof(float));
    if (!canvas_rgb) {
        iris_set_error("preprocess: out of memory for canvas");
        return -1;
    }

    /* Fill canvas with gray 0.5 (the composite for alpha=0 pixels). */
    for (int i = 0; i < canvas * canvas * 3; i++) canvas_rgb[i] = 0.5f;

    /* Stamp the cropped foreground into the canvas with composite. */
    for (int y = y1; y <= y2; y++) {
        int cy = (y - y1) + off_y;
        for (int x = x1; x <= x2; x++) {
            int cx = (x - x1) + off_x;
            float a = (has_alpha_mask)
                ? (float)src[(y * W + x) * 4 + 3] / 255.0f : 1.0f;
            float r = (float)src[(y * W + x) * C + 0] / 255.0f;
            float g = (float)src[(y * W + x) * C + 1] / 255.0f;
            float b = (float)src[(y * W + x) * C + 2] / 255.0f;
            float *dst = canvas_rgb + (cy * canvas + cx) * 3;
            dst[0] = r * a + 0.5f * (1.0f - a);
            dst[1] = g * a + 0.5f * (1.0f - a);
            dst[2] = b * a + 0.5f * (1.0f - a);
        }
    }

    /* ---- Step 3: bilinear resize canvas_rgb -> target x target, with
     *               align_corners=False and replicate-padding (PyTorch
     *               default for F.interpolate). Output stored as CHW. */
    const float scale = (float)canvas / (float)target;
    for (int oy = 0; oy < target; oy++) {
        float sy_f = ((float)oy + 0.5f) * scale - 0.5f;
        int sy0 = (int)floorf(sy_f);
        int sy1 = sy0 + 1;
        float dy = sy_f - (float)sy0;
        if (sy0 < 0) sy0 = 0;
        if (sy0 > canvas - 1) sy0 = canvas - 1;
        if (sy1 < 0) sy1 = 0;
        if (sy1 > canvas - 1) sy1 = canvas - 1;
        for (int ox = 0; ox < target; ox++) {
            float sx_f = ((float)ox + 0.5f) * scale - 0.5f;
            int sx0 = (int)floorf(sx_f);
            int sx1 = sx0 + 1;
            float dx = sx_f - (float)sx0;
            if (sx0 < 0) sx0 = 0;
            if (sx0 > canvas - 1) sx0 = canvas - 1;
            if (sx1 < 0) sx1 = 0;
            if (sx1 > canvas - 1) sx1 = canvas - 1;
            float w00 = (1.0f - dx) * (1.0f - dy);
            float w10 = dx * (1.0f - dy);
            float w01 = (1.0f - dx) * dy;
            float w11 = dx * dy;
            for (int ch = 0; ch < 3; ch++) {
                float v00 = canvas_rgb[(sy0 * canvas + sx0) * 3 + ch];
                float v10 = canvas_rgb[(sy0 * canvas + sx1) * 3 + ch];
                float v01 = canvas_rgb[(sy1 * canvas + sx0) * 3 + ch];
                float v11 = canvas_rgb[(sy1 * canvas + sx1) * 3 + ch];
                float v = w00*v00 + w10*v10 + w01*v01 + w11*v11;
                /* CHW output. */
                out_chw[ch * target * target + oy * target + ox] = v;
            }
        }
    }
    free(canvas_rgb);
    return 0;
}
