/*
 * lrm_vit_dino.c - DINOv1 ViT-B/16 encoder.
 *
 * Implements the pre-norm BasicTransformerBlock structure used by
 * transformers.ViTModel, plus the surrounding embedding scaffolding:
 *
 *   x = (image - ImageNet_mean) / ImageNet_std
 *   patches = Conv2d(x, kernel=16, stride=16)
 *   tokens  = concat(cls, flatten(patches))
 *   tokens  = tokens + bicubic_interp(pos_embed, 14x14 -> N_patches/side)
 *   for each of 12 blocks:
 *     tokens = tokens + attn_out_proj(MHSA(LayerNorm(tokens)))
 *     tokens = tokens + fc2(GELU(fc1(LayerNorm(tokens))))
 *   tokens = LayerNorm(tokens)
 *
 * Match notes vs the reference implementation:
 *   - ImageNet normalization is done inside lrm_vit_dino_forward to mirror
 *     DINOSingleImageTokenizer.forward (the values come in [0,1] from
 *     ImagePreprocessor); see tsr/models/tokenizers/image.py:51.
 *   - Position-embedding interpolation uses HF's specific
 *     `scale_factor = (target/source + 0.1) / source ... wait no:
 *     h0 = target_size // patch_size; h0 + 0.1; scale = (h0 + 0.1) / sqrt(num_pos).
 *     The +0.1 is HF's floating-point-precision hack; transformers/models/vit/modeling_vit.py.
 *     Bit-exact match requires it.
 *   - GELU is the *exact* form (erf-based), not the tanh approximation.
 *   - The final LayerNorm is `image_tokenizer.model.layernorm.{w,b}`. HF's
 *     pooler (CLS->768 dense) is present in the safetensors but unused by
 *     TripoSR; we don't load or call it.
 */

#include "lrm_vit_dino.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"
#include "iris_kernels.h"

/* ImageNet normalization constants (mean/std on the [0,1] image range). */
static const float kImageNetMean[3] = { 0.485f, 0.456f, 0.406f };
static const float kImageNetStd[3]  = { 0.229f, 0.224f, 0.225f };

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int set_err(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    iris_set_error(buf);
    return -1;
}

/* Find a tensor by name, require f32 dtype, and (if ndim>0) require the
 * exact shape. On success returns a pointer to the f32 data inside the
 * mmap; on failure returns NULL with iris_set_error set. */
static const float *find_f32(const safetensors_file_t *sf, const char *name,
                             int expected_ndim, const int64_t *expected_shape) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (t == NULL) {
        set_err("DINO: missing tensor '%s'", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        set_err("DINO: '%s' has dtype %d, expected f32", name, (int)t->dtype);
        return NULL;
    }
    if (expected_ndim > 0) {
        if (t->ndim != expected_ndim) {
            set_err("DINO: '%s' ndim=%d, expected %d", name, t->ndim, expected_ndim);
            return NULL;
        }
        for (int i = 0; i < expected_ndim; i++) {
            if (t->shape[i] != expected_shape[i]) {
                set_err("DINO: '%s' shape mismatch at axis %d "
                        "(got %lld, expected %lld)",
                        name, i, (long long)t->shape[i],
                        (long long)expected_shape[i]);
                return NULL;
            }
        }
    }
    return (const float *)safetensors_data(sf, t);
}

/* ========================================================================
 * Bicubic 2D interpolation matching HF ViT
 * ======================================================================== */

static double cubic_kernel(double d) {
    /* PyTorch's default a = -0.75 (a.k.a. Catmull-Rom-ish). */
    const double a = -0.75;
    d = fabs(d);
    if (d <= 1.0) {
        return ((a + 2.0) * d - (a + 3.0)) * d * d + 1.0;
    } else if (d < 2.0) {
        return ((a * d - 5.0 * a) * d + 8.0 * a) * d - 4.0 * a;
    }
    return 0.0;
}

static int iclamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Interpolate the pos_embed grid (1, 768, 14, 14) -> (1, 768, target, target).
 *
 * `src` is in HF's safetensors layout: positions 1..196 are a 14x14 grid in
 * row-major order with the channel index varying fastest, i.e. [196, hidden].
 * The CLS row at index 0 is handled by the caller (it is copied verbatim).
 *
 * Output `dst` has layout [target*target, hidden] (also row-major, with
 * channels fastest) so the caller can just append it after the CLS row.
 *
 * Matches PyTorch's F.interpolate(mode='bicubic', align_corners=False) with
 * scale_factor = (target + 0.1) / source, which is the HF hack for FP
 * precision (transformers/models/vit/modeling_vit.py::interpolate_pos_encoding).
 */
static void bicubic_interp_pos(const float *src, int source, int hidden,
                               float *dst, int target) {
    const double scale = ((double)target + 0.1) / (double)source;
    const int src_max = source - 1;

    for (int oy = 0; oy < target; oy++) {
        const double sy = ((double)oy + 0.5) / scale - 0.5;
        const int iy = (int)floor(sy);
        const double dy = sy - (double)iy;
        double Wy[4];
        Wy[0] = cubic_kernel(1.0 + dy);
        Wy[1] = cubic_kernel(dy);
        Wy[2] = cubic_kernel(1.0 - dy);
        Wy[3] = cubic_kernel(2.0 - dy);

        for (int ox = 0; ox < target; ox++) {
            const double sx = ((double)ox + 0.5) / scale - 0.5;
            const int ix = (int)floor(sx);
            const double dx = sx - (double)ix;
            double Wx[4];
            Wx[0] = cubic_kernel(1.0 + dx);
            Wx[1] = cubic_kernel(dx);
            Wx[2] = cubic_kernel(1.0 - dx);
            Wx[3] = cubic_kernel(2.0 - dx);

            /* Cache the four clamped row indices and four clamped col indices. */
            int rows[4];
            int cols[4];
            for (int k = 0; k < 4; k++) {
                rows[k] = iclamp(iy + k - 1, 0, src_max);
                cols[k] = iclamp(ix + k - 1, 0, src_max);
            }

            float *out_row = dst + ((size_t)oy * target + ox) * hidden;
            for (int c = 0; c < hidden; c++) {
                double acc = 0.0;
                for (int ky = 0; ky < 4; ky++) {
                    const float *row_base = src + (size_t)rows[ky] * source * hidden;
                    double wy = Wy[ky];
                    for (int kx = 0; kx < 4; kx++) {
                        double w = wy * Wx[kx];
                        acc += w * (double)row_base[(size_t)cols[kx] * hidden + c];
                    }
                }
                out_row[c] = (float)acc;
            }
        }
    }
}

/* ========================================================================
 * Weight binding (init)
 * ======================================================================== */

static int bind_block(lrm_vit_dino *vit, const safetensors_file_t *sf, int i) {
    char nm[256];
    const int H = vit->hidden_dim;
    const int M = vit->mlp_dim;

#define BIND_F32(field, fmt, ndim, ...) do { \
        int64_t s[] = { __VA_ARGS__ }; \
        snprintf(nm, sizeof(nm), fmt, i); \
        const float *p = find_f32(sf, nm, ndim, s); \
        if (!p) return -1; \
        vit->blocks[i].field = p; \
    } while (0)

    BIND_F32(ln_before_w, "image_tokenizer.model.encoder.layer.%d.layernorm_before.weight", 1, H);
    BIND_F32(ln_before_b, "image_tokenizer.model.encoder.layer.%d.layernorm_before.bias",   1, H);
    BIND_F32(q_w,         "image_tokenizer.model.encoder.layer.%d.attention.attention.query.weight", 2, H, H);
    BIND_F32(q_b,         "image_tokenizer.model.encoder.layer.%d.attention.attention.query.bias",   1, H);
    BIND_F32(k_w,         "image_tokenizer.model.encoder.layer.%d.attention.attention.key.weight",   2, H, H);
    BIND_F32(k_b,         "image_tokenizer.model.encoder.layer.%d.attention.attention.key.bias",     1, H);
    BIND_F32(v_w,         "image_tokenizer.model.encoder.layer.%d.attention.attention.value.weight", 2, H, H);
    BIND_F32(v_b,         "image_tokenizer.model.encoder.layer.%d.attention.attention.value.bias",   1, H);
    BIND_F32(attn_out_w,  "image_tokenizer.model.encoder.layer.%d.attention.output.dense.weight",    2, H, H);
    BIND_F32(attn_out_b,  "image_tokenizer.model.encoder.layer.%d.attention.output.dense.bias",      1, H);
    BIND_F32(ln_after_w,  "image_tokenizer.model.encoder.layer.%d.layernorm_after.weight",            1, H);
    BIND_F32(ln_after_b,  "image_tokenizer.model.encoder.layer.%d.layernorm_after.bias",              1, H);
    BIND_F32(fc1_w,       "image_tokenizer.model.encoder.layer.%d.intermediate.dense.weight",        2, M, H);
    BIND_F32(fc1_b,       "image_tokenizer.model.encoder.layer.%d.intermediate.dense.bias",          1, M);
    BIND_F32(fc2_w,       "image_tokenizer.model.encoder.layer.%d.output.dense.weight",              2, H, M);
    BIND_F32(fc2_b,       "image_tokenizer.model.encoder.layer.%d.output.dense.bias",                1, H);
#undef BIND_F32
    return 0;
}

int lrm_vit_dino_init(lrm_vit_dino *vit, const safetensors_file_t *sf,
                      int input_size) {
    memset(vit, 0, sizeof(*vit));

    /* Architectural constants (DINO ViT-B/16). */
    vit->input_size       = input_size;
    vit->patch_size       = 16;
    vit->hidden_dim       = 768;
    vit->num_layers       = 12;
    vit->num_heads        = 12;
    vit->head_dim         = 64;
    vit->mlp_dim          = 3072;
    vit->patches_per_side = input_size / vit->patch_size;
    vit->num_tokens       = 1 + vit->patches_per_side * vit->patches_per_side;

    if (vit->num_layers > LRM_DINO_MAX_LAYERS) {
        return set_err("DINO: num_layers=%d exceeds LRM_DINO_MAX_LAYERS=%d",
                       vit->num_layers, LRM_DINO_MAX_LAYERS);
    }

    /* Embeddings. */
    int64_t s_proj_w[] = { 768, 3, 16, 16 };
    vit->patch_embed_w = find_f32(sf, "image_tokenizer.model.embeddings.patch_embeddings.projection.weight",
                                  4, s_proj_w);
    if (!vit->patch_embed_w) return -1;
    int64_t s_proj_b[] = { 768 };
    vit->patch_embed_b = find_f32(sf, "image_tokenizer.model.embeddings.patch_embeddings.projection.bias",
                                  1, s_proj_b);
    if (!vit->patch_embed_b) return -1;
    int64_t s_cls[] = { 1, 1, 768 };
    vit->cls_token = find_f32(sf, "image_tokenizer.model.embeddings.cls_token", 3, s_cls);
    if (!vit->cls_token) return -1;

    /* Position embeddings: [1, 197, 768] in the safetensors; we interpolate
     * the 14x14 patch grid to patches_per_side and keep the CLS row as-is. */
    int64_t s_pos[] = { 1, 197, 768 };
    const float *pos_orig = find_f32(sf, "image_tokenizer.model.embeddings.position_embeddings",
                                     3, s_pos);
    if (!pos_orig) return -1;

    vit->pos_embed = (float *)malloc((size_t)vit->num_tokens * vit->hidden_dim * sizeof(float));
    if (!vit->pos_embed) return set_err("DINO: out of memory for pos_embed");

    /* CLS row: positions[0] verbatim. */
    memcpy(vit->pos_embed, pos_orig, vit->hidden_dim * sizeof(float));

    /* Patch rows: bicubic-interpolate the 14x14 grid to (pps, pps).
     * `pos_orig + hidden_dim` is the start of the 196 patch positions. */
    bicubic_interp_pos(pos_orig + vit->hidden_dim,
                       /*source=*/14, /*hidden=*/vit->hidden_dim,
                       vit->pos_embed + vit->hidden_dim,
                       /*target=*/vit->patches_per_side);

    /* Per-block weights. */
    for (int i = 0; i < vit->num_layers; i++) {
        if (bind_block(vit, sf, i) != 0) return -1;
    }

    /* Final LayerNorm. */
    int64_t s_ln[] = { 768 };
    vit->final_ln_w = find_f32(sf, "image_tokenizer.model.layernorm.weight", 1, s_ln);
    if (!vit->final_ln_w) return -1;
    vit->final_ln_b = find_f32(sf, "image_tokenizer.model.layernorm.bias",   1, s_ln);
    if (!vit->final_ln_b) return -1;

    return 0;
}

void lrm_vit_dino_release(lrm_vit_dino *vit) {
    if (!vit) return;
    free(vit->pos_embed);
    vit->pos_embed = NULL;
}

/* ========================================================================
 * Forward
 * ======================================================================== */

/* Workspace layout (one allocation, sub-buffers carved out by lrm_vit_dino_forward):
 *
 *   normed_x   : [num_tokens, hidden]   -- pre-attn / pre-mlp LayerNorm output
 *   Q          : [num_tokens, hidden]
 *   K          : [num_tokens, hidden]
 *   V          : [num_tokens, hidden]
 *   attn_out   : [num_tokens, hidden]   -- post-MHSA, pre-residual-add
 *   mlp_inter  : [num_tokens, mlp_dim]  -- after fc1+GELU
 *   patch_buf  : [hidden, pps, pps]     -- raw Conv2d output (pre-reshape)
 *
 * We reuse patch_buf across calls (just one allocation). The actual size
 * derived in lrm_vit_dino_workspace_bytes.
 */

static void workspace_layout(const lrm_vit_dino *vit,
                             size_t *o_normed, size_t *o_q, size_t *o_k,
                             size_t *o_v, size_t *o_attn, size_t *o_mlp,
                             size_t *o_patch, size_t *o_total) {
    size_t tok = (size_t)vit->num_tokens * vit->hidden_dim;
    size_t mlp = (size_t)vit->num_tokens * vit->mlp_dim;
    size_t patch = (size_t)vit->hidden_dim * vit->patches_per_side * vit->patches_per_side;

    size_t off = 0;
    *o_normed = off; off += tok;
    *o_q      = off; off += tok;
    *o_k      = off; off += tok;
    *o_v      = off; off += tok;
    *o_attn   = off; off += tok;
    *o_mlp    = off; off += mlp;
    *o_patch  = off; off += patch;
    *o_total  = off;
}

size_t lrm_vit_dino_workspace_bytes(const lrm_vit_dino *vit) {
    size_t a, b, c, d, e, f, g, total;
    workspace_layout(vit, &a, &b, &c, &d, &e, &f, &g, &total);
    return total * sizeof(float);
}

/* ImageNet-normalize a CHW image in-place into `out` (3 * H * W floats).
 * `image` is the same layout (channel-major). */
static void imagenet_normalize(const float *image, float *out, int H, int W) {
    int hw = H * W;
    for (int c = 0; c < 3; c++) {
        float m = kImageNetMean[c];
        float s = kImageNetStd[c];
        const float *in_c = image + (size_t)c * hw;
        float *out_c = out + (size_t)c * hw;
        for (int i = 0; i < hw; i++) {
            out_c[i] = (in_c[i] - m) / s;
        }
    }
}

/* Reshape Conv2d output [hidden, pps, pps] (channel-major) into the
 * patch-token rows [num_patches, hidden] (channels-last per token). */
static void patches_to_tokens(const float *patch_chw, float *tokens,
                              int hidden, int pps) {
    int n_patches = pps * pps;
    int hw = pps * pps;
    for (int s = 0; s < n_patches; s++) {
        for (int c = 0; c < hidden; c++) {
            tokens[(size_t)s * hidden + c] = patch_chw[(size_t)c * hw + s];
        }
    }
}

int lrm_vit_dino_forward(const lrm_vit_dino *vit,
                         const float *image,
                         float *tokens_out,
                         float *workspace) {
    if (!vit || !image || !tokens_out || !workspace) {
        return set_err("DINO: NULL argument to lrm_vit_dino_forward");
    }
    const int H = vit->input_size;
    const int W = vit->input_size;
    const int hidden = vit->hidden_dim;
    const int pps = vit->patches_per_side;
    const int num_tokens = vit->num_tokens;
    const int num_heads = vit->num_heads;
    const int head_dim = vit->head_dim;
    const int mlp_dim = vit->mlp_dim;
    const float attn_scale = 1.0f / sqrtf((float)head_dim);

    /* Carve workspace. */
    size_t o_n, o_q, o_k, o_v, o_attn, o_mlp, o_patch, o_total;
    workspace_layout(vit, &o_n, &o_q, &o_k, &o_v, &o_attn, &o_mlp, &o_patch, &o_total);
    float *normed   = workspace + o_n;
    float *Q        = workspace + o_q;
    float *K        = workspace + o_k;
    float *V        = workspace + o_v;
    float *attn_out = workspace + o_attn;
    float *mlp_inter= workspace + o_mlp;
    float *patch_chw= workspace + o_patch;

    /* ----- 1. ImageNet normalize (image is [3, H, W]).
     *           Stash result in patch_chw temporarily; we'll overwrite it
     *           when we run the conv below. */
    /* We need to keep the normalized image around until conv finishes, so
     * borrow a chunk inside attn_out (which is unused until after attn). */
    float *normalized_img = attn_out;  /* needs >= 3*H*W floats */
    {
        size_t need = (size_t)3 * H * W;
        size_t have = (size_t)num_tokens * hidden;
        if (have < need) {
            return set_err("DINO: attn_out scratch (%zu) < normalized image (%zu)",
                           have, need);
        }
    }
    imagenet_normalize(image, normalized_img, H, W);

    /* ----- 2. Patch embedding: Conv2d, kernel=16, stride=16, padding=0.
     *           Output shape [768, 32, 32] (channel-major). */
    iris_conv2d(patch_chw, normalized_img, vit->patch_embed_w, vit->patch_embed_b,
                /*batch=*/1, /*in_ch=*/3, /*out_ch=*/hidden, H, W,
                /*kH=*/vit->patch_size, /*kW=*/vit->patch_size,
                /*stride=*/vit->patch_size, /*padding=*/0);

    /* attn_out can be reused now. */

    /* ----- 3. Token assembly: tokens_out[0] = cls_token; tokens_out[1..] = patch rows.
     *           Then add pos_embed. */
    memcpy(tokens_out, vit->cls_token, hidden * sizeof(float));
    patches_to_tokens(patch_chw, tokens_out + hidden, hidden, pps);

    /* Add interpolated pos_embed (size num_tokens * hidden). */
    iris_add_inplace(tokens_out, vit->pos_embed, num_tokens * hidden);

    /* ----- 4. 12 transformer blocks. */
    for (int b = 0; b < vit->num_layers; b++) {
        /* Pre-attention LayerNorm. */
        iris_layer_norm(normed, tokens_out,
                        vit->blocks[b].ln_before_w, vit->blocks[b].ln_before_b,
                        num_tokens, hidden, 1e-12f);  /* HF ViT eps */

        /* Q/K/V projections: each is [num_tokens, hidden]. */
        iris_linear(Q, normed, vit->blocks[b].q_w, vit->blocks[b].q_b,
                    num_tokens, hidden, hidden);
        iris_linear(K, normed, vit->blocks[b].k_w, vit->blocks[b].k_b,
                    num_tokens, hidden, hidden);
        iris_linear(V, normed, vit->blocks[b].v_w, vit->blocks[b].v_b,
                    num_tokens, hidden, hidden);

        /* Multi-head self-attention. Flash takes the natural [seq, heads*hd]
         * layout, no head-permute needed. */
        iris_flash_attention(attn_out, Q, K, V,
                             /*seq_q=*/num_tokens, /*seq_k=*/num_tokens,
                             num_heads, head_dim, attn_scale);

        /* Attention output projection + residual.
         * We re-use Q as scratch for the projection, then add into tokens_out. */
        iris_linear(Q, attn_out, vit->blocks[b].attn_out_w, vit->blocks[b].attn_out_b,
                    num_tokens, hidden, hidden);
        iris_add_inplace(tokens_out, Q, num_tokens * hidden);

        /* Pre-MLP LayerNorm. */
        iris_layer_norm(normed, tokens_out,
                        vit->blocks[b].ln_after_w, vit->blocks[b].ln_after_b,
                        num_tokens, hidden, 1e-12f);

        /* MLP: FC1 (hidden -> 3072), GELU (inplace), FC2 (3072 -> hidden). */
        iris_linear(mlp_inter, normed,
                    vit->blocks[b].fc1_w, vit->blocks[b].fc1_b,
                    num_tokens, hidden, mlp_dim);
        iris_gelu(mlp_inter, num_tokens * mlp_dim);
        iris_linear(Q, mlp_inter,
                    vit->blocks[b].fc2_w, vit->blocks[b].fc2_b,
                    num_tokens, mlp_dim, hidden);  /* re-use Q as out */
        iris_add_inplace(tokens_out, Q, num_tokens * hidden);
    }

    /* ----- 5. Final LayerNorm (in-place into tokens_out via the normed scratch). */
    iris_layer_norm(normed, tokens_out,
                    vit->final_ln_w, vit->final_ln_b,
                    num_tokens, hidden, 1e-12f);
    memcpy(tokens_out, normed, (size_t)num_tokens * hidden * sizeof(float));

    return 0;
}
