/*
 * lrm_triplane_decoder.c - 16-block image-to-triplane cross-attention transformer.
 *
 * See lrm_triplane_decoder.h for the architecture summary. This file ports the
 * Transformer1D + BasicTransformerBlock(activation_fn='geglu', norm_type='layer_norm')
 * stack from diffusers, matching the layout TripoSR ships at HF revision
 * 5b521936.
 *
 * Layouts inside forward():
 *   queries_chl  : [1024, 3072]    Ct-major, the cached learned tokens
 *   x_lc         : [3072, 1024]    Nt-major, the working hidden state through the blocks
 *
 * The conversion from [Ct, Nt] -> [Nt, Ct] happens once after GroupNorm; from
 * there everything stays in [Nt, Ct] (which is what iris_linear,
 * iris_flash_attention and friends consume natively).
 */

#include "lrm_triplane_decoder.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"
#include "iris_kernels.h"

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

static const float *find_f32(const safetensors_file_t *sf, const char *name,
                             int expected_ndim, const int64_t *expected_shape) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (t == NULL) {
        set_err("triplane_decoder: missing tensor '%s'", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        set_err("triplane_decoder: '%s' has dtype %d, expected f32",
                name, (int)t->dtype);
        return NULL;
    }
    if (expected_ndim > 0) {
        if (t->ndim != expected_ndim) {
            set_err("triplane_decoder: '%s' ndim=%d, expected %d",
                    name, t->ndim, expected_ndim);
            return NULL;
        }
        for (int i = 0; i < expected_ndim; i++) {
            if (t->shape[i] != expected_shape[i]) {
                set_err("triplane_decoder: '%s' shape mismatch at axis %d "
                        "(got %lld, expected %lld)",
                        name, i, (long long)t->shape[i],
                        (long long)expected_shape[i]);
                return NULL;
            }
        }
    }
    return (const float *)safetensors_data(sf, t);
}

/* Transpose [rows, cols] row-major -> [cols, rows] row-major. */
static void transpose_rc(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
        }
    }
}

/* Chunked GEGLU: out[r, c] = proj[r, c] * gelu(proj[r, c + chunk]),
 * for r in [0, rows), c in [0, chunk). proj has 2*chunk cols.
 * Safe to alias `out` with the first `rows*chunk` floats of `proj` (we
 * read each proj row fully before writing into the row's first half). */
static void geglu_chunked(float *out, const float *proj, int rows, int chunk) {
    static const float kInvSqrt2 = 0.70710678118654752440f;
    const int stride = 2 * chunk;
    for (int r = 0; r < rows; r++) {
        const float *row_in = proj + (size_t)r * stride;
        float *row_out = out + (size_t)r * chunk;
        for (int c = 0; c < chunk; c++) {
            float h = row_in[c];
            float g = row_in[c + chunk];
            float g_act = 0.5f * g * (1.0f + erff(g * kInvSqrt2));
            row_out[c] = h * g_act;
        }
    }
}

/* ========================================================================
 * Init / Release
 * ======================================================================== */

static int bind_block(lrm_triplane_decoder *dec,
                      const safetensors_file_t *sf, int i) {
    char nm[256];
    const int H = dec->hidden_dim;       /* 1024 */
    const int KV = dec->cross_kv_dim;    /* 768 */
    const int F2 = 2 * dec->ff_inner;    /* 8192 */
    const int F1 = dec->ff_inner;        /* 4096 */

#define BIND(field, fmt, ndim, ...) do {              \
    int64_t s[] = { __VA_ARGS__ };                    \
    snprintf(nm, sizeof(nm), fmt, i);                 \
    const float *p = find_f32(sf, nm, ndim, s);       \
    if (!p) return -1;                                \
    dec->blocks[i].field = p;                         \
} while (0)

    BIND(norm1_w,    "backbone.transformer_blocks.%d.norm1.weight", 1, H);
    BIND(norm1_b,    "backbone.transformer_blocks.%d.norm1.bias",   1, H);
    BIND(norm2_w,    "backbone.transformer_blocks.%d.norm2.weight", 1, H);
    BIND(norm2_b,    "backbone.transformer_blocks.%d.norm2.bias",   1, H);
    BIND(norm3_w,    "backbone.transformer_blocks.%d.norm3.weight", 1, H);
    BIND(norm3_b,    "backbone.transformer_blocks.%d.norm3.bias",   1, H);

    /* Self-attn (Q/K/V no bias). */
    BIND(self_q_w,   "backbone.transformer_blocks.%d.attn1.to_q.weight", 2, H, H);
    BIND(self_k_w,   "backbone.transformer_blocks.%d.attn1.to_k.weight", 2, H, H);
    BIND(self_v_w,   "backbone.transformer_blocks.%d.attn1.to_v.weight", 2, H, H);
    BIND(self_out_w, "backbone.transformer_blocks.%d.attn1.to_out.0.weight", 2, H, H);
    BIND(self_out_b, "backbone.transformer_blocks.%d.attn1.to_out.0.bias",   1, H);

    /* Cross-attn: Q from triplane (H), K/V from image tokens (KV=768). */
    BIND(cross_q_w,   "backbone.transformer_blocks.%d.attn2.to_q.weight", 2, H, H);
    BIND(cross_k_w,   "backbone.transformer_blocks.%d.attn2.to_k.weight", 2, H, KV);
    BIND(cross_v_w,   "backbone.transformer_blocks.%d.attn2.to_v.weight", 2, H, KV);
    BIND(cross_out_w, "backbone.transformer_blocks.%d.attn2.to_out.0.weight", 2, H, H);
    BIND(cross_out_b, "backbone.transformer_blocks.%d.attn2.to_out.0.bias",   1, H);

    /* GEGLU FFN. */
    BIND(ff_geglu_w, "backbone.transformer_blocks.%d.ff.net.0.proj.weight", 2, F2, H);
    BIND(ff_geglu_b, "backbone.transformer_blocks.%d.ff.net.0.proj.bias",   1, F2);
    BIND(ff_out_w,   "backbone.transformer_blocks.%d.ff.net.2.weight",      2, H, F1);
    BIND(ff_out_b,   "backbone.transformer_blocks.%d.ff.net.2.bias",        1, H);
#undef BIND
    return 0;
}

int lrm_triplane_decoder_init(lrm_triplane_decoder *dec,
                              const safetensors_file_t *sf) {
    memset(dec, 0, sizeof(*dec));

    dec->hidden_dim     = 1024;
    dec->num_layers     = 16;
    dec->num_heads      = 16;
    dec->head_dim       = 64;
    dec->ff_inner       = 4096;
    dec->cross_kv_dim   = 768;
    dec->triplane_planes= 3;
    dec->triplane_size  = 32;
    dec->num_tokens     = dec->triplane_planes * dec->triplane_size * dec->triplane_size;

    /* Top-level GroupNorm + proj_in/out. */
    int64_t s_h[]   = { 1024 };
    int64_t s_hh[]  = { 1024, 1024 };
    dec->gn_w       = find_f32(sf, "backbone.norm.weight", 1, s_h);
    dec->gn_b       = find_f32(sf, "backbone.norm.bias",   1, s_h);
    dec->proj_in_w  = find_f32(sf, "backbone.proj_in.weight",  2, s_hh);
    dec->proj_in_b  = find_f32(sf, "backbone.proj_in.bias",    1, s_h);
    dec->proj_out_w = find_f32(sf, "backbone.proj_out.weight", 2, s_hh);
    dec->proj_out_b = find_f32(sf, "backbone.proj_out.bias",   1, s_h);
    if (!dec->gn_w || !dec->gn_b || !dec->proj_in_w || !dec->proj_in_b ||
        !dec->proj_out_w || !dec->proj_out_b) {
        return -1;
    }

    /* Triplane query tokens: stored as [Np=3, Ct=1024, Hp=32, Wp=32],
     * cached as [Ct=1024, Np*Hp*Wp=3072] to match Triplane1DTokenizer.forward
     * (rearrange "Np Ct Hp Wp -> Ct (Np Hp Wp)" with B=1 squeezed out). */
    int64_t s_emb[] = { dec->triplane_planes, dec->hidden_dim,
                        dec->triplane_size, dec->triplane_size };
    const float *emb_src = find_f32(sf, "tokenizer.embeddings", 4, s_emb);
    if (!emb_src) return -1;

    const int Np = dec->triplane_planes;
    const int Ct = dec->hidden_dim;
    const int Hp = dec->triplane_size;
    const int Wp = dec->triplane_size;
    const int HW = Hp * Wp;            /* 1024 */
    const int Nt = Np * HW;            /* 3072 */

    dec->queries_chl = (float *)malloc((size_t)Ct * Nt * sizeof(float));
    if (!dec->queries_chl) return set_err("triplane_decoder: oom for queries");

    for (int np = 0; np < Np; np++) {
        for (int ct = 0; ct < Ct; ct++) {
            const float *src = emb_src + ((size_t)np * Ct + ct) * HW;
            float *dst       = dec->queries_chl + (size_t)ct * Nt + (size_t)np * HW;
            memcpy(dst, src, (size_t)HW * sizeof(float));
        }
    }

    /* 16 block weights. */
    for (int i = 0; i < dec->num_layers; i++) {
        if (bind_block(dec, sf, i) != 0) return -1;
    }
    return 0;
}

void lrm_triplane_decoder_release(lrm_triplane_decoder *dec) {
    if (!dec) return;
    free(dec->queries_chl);
    dec->queries_chl = NULL;
}

/* ========================================================================
 * Workspace
 * ======================================================================== */

/* Workspace layout (allocated as one float buffer):
 *
 *   gn_out_chl  : [Ct=1024, Nt=3072]   GroupNorm output, channels-first
 *   x_lc        : [Nt=3072, Ct=1024]   working hidden state
 *   normed_lc   : [Nt=3072, Ct=1024]   LN scratch each block
 *   q_lc        : [Nt=3072, Ct=1024]   Q projection (sized for self-attn;
 *                                       cross-attn Q is same shape)
 *   k_lc        : [Nt=3072, Ct=1024]   K (self-attn); cross-attn K is
 *                                       [1025, 1024] which fits
 *   v_lc        : [Nt=3072, Ct=1024]   V same as K
 *   attn_out    : [Nt=3072, Ct=1024]   flash_attention output, also
 *                                       reused for proj-out scratch
 *   ff_proj     : [Nt=3072, 2*ff_inner=8192]  GEGLU input projection;
 *                                              ff_act overlays the first
 *                                              [Nt, ff_inner=4096] floats
 *
 * Total ~184 MB for input_size=512.
 */

static void workspace_layout(const lrm_triplane_decoder *dec,
                             size_t *o_gn, size_t *o_x, size_t *o_n,
                             size_t *o_q, size_t *o_k, size_t *o_v,
                             size_t *o_a, size_t *o_ff,
                             size_t *o_total) {
    size_t tok   = (size_t)dec->num_tokens * dec->hidden_dim;
    size_t ffproj= (size_t)dec->num_tokens * (2 * dec->ff_inner);
    size_t off = 0;
    *o_gn = off; off += tok;
    *o_x  = off; off += tok;
    *o_n  = off; off += tok;
    *o_q  = off; off += tok;
    *o_k  = off; off += tok;
    *o_v  = off; off += tok;
    *o_a  = off; off += tok;
    *o_ff = off; off += ffproj;
    *o_total = off;
}

size_t lrm_triplane_decoder_workspace_bytes(const lrm_triplane_decoder *dec) {
    size_t a,b,c,d,e,f,g,h,total;
    workspace_layout(dec, &a, &b, &c, &d, &e, &f, &g, &h, &total);
    return total * sizeof(float);
}

/* ========================================================================
 * Forward
 * ======================================================================== */

int lrm_triplane_decoder_forward(const lrm_triplane_decoder *dec,
                                 const float *image_tokens,
                                 int num_image_tokens,
                                 float *triplane_out,
                                 float *workspace) {
    if (!dec || !image_tokens || !triplane_out || !workspace) {
        return set_err("triplane_decoder: NULL argument to forward");
    }

    const int Nt = dec->num_tokens;       /* 3072 */
    const int Ct = dec->hidden_dim;       /* 1024 */
    const int Nh = dec->num_heads;        /* 16 */
    const int Hd = dec->head_dim;         /* 64 */
    const int Nx = num_image_tokens;      /* 1025 */
    const int KV = dec->cross_kv_dim;     /* 768 */
    const int F1 = dec->ff_inner;         /* 4096 */
    const int F2 = 2 * F1;                /* 8192 */
    const float attn_scale = 1.0f / sqrtf((float)Hd);

    size_t o_gn, o_x, o_n, o_q, o_k, o_v, o_a, o_ff, o_total;
    workspace_layout(dec, &o_gn, &o_x, &o_n, &o_q, &o_k, &o_v, &o_a, &o_ff, &o_total);
    float *gn_out_chl = workspace + o_gn;
    float *x_lc       = workspace + o_x;
    float *normed_lc  = workspace + o_n;
    float *q_lc       = workspace + o_q;
    float *k_lc       = workspace + o_k;
    float *v_lc       = workspace + o_v;
    float *attn_out   = workspace + o_a;
    float *ff_proj    = workspace + o_ff;
    /* GEGLU activation overlays the start of ff_proj. */
    float *ff_act     = ff_proj;

    /* ----- 1. GroupNorm over channels (groups=32) on queries_chl.
     *           Treat the [Ct, Nt] buffer as [B=1, C=1024, H=1, W=3072]. */
    iris_group_norm(gn_out_chl, dec->queries_chl,
                    dec->gn_w, dec->gn_b,
                    /*batch=*/1, /*channels=*/Ct,
                    /*H=*/1, /*W=*/Nt,
                    /*num_groups=*/32, /*eps=*/1e-6f);

    /* ----- 2. Transpose [Ct, Nt] -> [Nt, Ct], stash result in x_lc as the
     *           input to proj_in. */
    transpose_rc(x_lc, gn_out_chl, /*rows=*/Ct, /*cols=*/Nt);

    /* ----- 3. proj_in: y = x @ W^T + b, [Nt, Ct] -> [Nt, Ct].
     *           Use normed_lc as the transient output, then swap pointers
     *           by copy back to x_lc (next stages mutate it). */
    iris_linear(normed_lc, x_lc, dec->proj_in_w, dec->proj_in_b,
                Nt, Ct, Ct);
    memcpy(x_lc, normed_lc, (size_t)Nt * Ct * sizeof(float));

    /* ----- 4. 16 transformer blocks. */
    for (int b = 0; b < dec->num_layers; b++) {
        /* Pre-attn LayerNorm. */
        iris_layer_norm(normed_lc, x_lc,
                        dec->blocks[b].norm1_w, dec->blocks[b].norm1_b,
                        Nt, Ct, 1e-5f);

        /* Self-attn: Q/K/V from triplane, no bias. */
        iris_linear_nobias(q_lc, normed_lc, dec->blocks[b].self_q_w, Nt, Ct, Ct);
        iris_linear_nobias(k_lc, normed_lc, dec->blocks[b].self_k_w, Nt, Ct, Ct);
        iris_linear_nobias(v_lc, normed_lc, dec->blocks[b].self_v_w, Nt, Ct, Ct);
        iris_flash_attention(attn_out, q_lc, k_lc, v_lc,
                             /*seq_q=*/Nt, /*seq_k=*/Nt,
                             Nh, Hd, attn_scale);
        /* Attention output projection + residual. */
        iris_linear(q_lc, attn_out, dec->blocks[b].self_out_w,
                    dec->blocks[b].self_out_b, Nt, Ct, Ct);
        iris_add_inplace(x_lc, q_lc, Nt * Ct);

        /* Pre-cross-attn LayerNorm. */
        iris_layer_norm(normed_lc, x_lc,
                        dec->blocks[b].norm2_w, dec->blocks[b].norm2_b,
                        Nt, Ct, 1e-5f);

        /* Cross-attn: Q from triplane (1024-d), K/V from image tokens (768-d).
         * K and V live in k_lc / v_lc at shape [Nx=1025, Ct=1024]. */
        iris_linear_nobias(q_lc, normed_lc, dec->blocks[b].cross_q_w, Nt, Ct, Ct);
        iris_linear_nobias(k_lc, image_tokens, dec->blocks[b].cross_k_w, Nx, KV, Ct);
        iris_linear_nobias(v_lc, image_tokens, dec->blocks[b].cross_v_w, Nx, KV, Ct);
        iris_flash_attention(attn_out, q_lc, k_lc, v_lc,
                             /*seq_q=*/Nt, /*seq_k=*/Nx,
                             Nh, Hd, attn_scale);
        iris_linear(q_lc, attn_out, dec->blocks[b].cross_out_w,
                    dec->blocks[b].cross_out_b, Nt, Ct, Ct);
        iris_add_inplace(x_lc, q_lc, Nt * Ct);

        /* Pre-FFN LayerNorm. */
        iris_layer_norm(normed_lc, x_lc,
                        dec->blocks[b].norm3_w, dec->blocks[b].norm3_b,
                        Nt, Ct, 1e-5f);

        /* GEGLU: project to 2*F1, chunk, mix. */
        iris_linear(ff_proj, normed_lc,
                    dec->blocks[b].ff_geglu_w, dec->blocks[b].ff_geglu_b,
                    Nt, Ct, F2);
        geglu_chunked(ff_act, ff_proj, Nt, F1);
        /* FF output projection F1 -> Ct + residual. */
        iris_linear(q_lc, ff_act,
                    dec->blocks[b].ff_out_w, dec->blocks[b].ff_out_b,
                    Nt, F1, Ct);
        iris_add_inplace(x_lc, q_lc, Nt * Ct);
    }

    /* ----- 5. proj_out: [Nt, Ct] -> [Nt, Ct]. */
    iris_linear(normed_lc, x_lc, dec->proj_out_w, dec->proj_out_b,
                Nt, Ct, Ct);

    /* ----- 6. Residual add: out_lc = normed_lc + queries_lc.
     *           We have queries in [Ct, Nt] cache; reconstruct queries
     *           per element on the fly to avoid another scratch buffer. */
    {
        const float *qch = dec->queries_chl;
        for (int n = 0; n < Nt; n++) {
            float *row = normed_lc + (size_t)n * Ct;
            for (int c = 0; c < Ct; c++) {
                row[c] += qch[(size_t)c * Nt + n];
            }
        }
    }

    /* ----- 7. Detokenize: [Nt, Ct] -> [Np, Ct, Hp, Wp]. */
    const int Np = dec->triplane_planes;
    const int Hp = dec->triplane_size;
    const int Wp = dec->triplane_size;
    const int HW = Hp * Wp;
    for (int np = 0; np < Np; np++) {
        for (int hp = 0; hp < Hp; hp++) {
            for (int wp = 0; wp < Wp; wp++) {
                int n = np * HW + hp * Wp + wp;
                const float *src = normed_lc + (size_t)n * Ct;
                for (int c = 0; c < Ct; c++) {
                    triplane_out[((size_t)np * Ct + c) * HW
                                 + (size_t)hp * Wp + wp] = src[c];
                }
            }
        }
    }
    return 0;
}
