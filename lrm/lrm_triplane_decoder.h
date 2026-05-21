/*
 * lrm_triplane_decoder.h - 16-block image-to-triplane cross-attention transformer.
 *
 * Mirrors diffusers' Transformer1D + BasicTransformerBlock (Apache-2.0, TripoSR
 * carries the same code under tsr/models/transformer/).
 *
 * Pipeline (per Transformer1D.forward):
 *   tokens0 = Triplane1DTokenizer.forward(B=1) shape [1, 1024, 3072]
 *   residual = tokens0
 *   hidden   = GroupNorm(tokens0, groups=32, eps=1e-6, channels-axis-1)
 *   hidden   = hidden.transpose(1, 2)                     -> [1, 3072, 1024]
 *   hidden   = proj_in(hidden)                            -> [1, 3072, 1024]
 *   for each of 16 BasicTransformerBlocks:
 *     # Pre-norm self-attention
 *     n = LayerNorm(hidden, norm1)
 *     hidden += SelfAttn(n)                                -- 16 heads x 64 d
 *     # Pre-norm cross-attention (queries=triplane, K/V=image tokens 768d)
 *     n = LayerNorm(hidden, norm2)
 *     hidden += CrossAttn(n, encoder=image_tokens)
 *     # Pre-norm GEGLU FFN
 *     n = LayerNorm(hidden, norm3)
 *     proj   = Linear(n, 1024 -> 8192)
 *     act    = proj[:, :4096] * GELU(proj[:, 4096:])
 *     hidden += Linear(act, 4096 -> 1024)
 *   hidden = proj_out(hidden)                              -> [1, 3072, 1024]
 *   output = hidden.transpose(1,2) + residual              -> [1, 1024, 3072]
 *   triplane = detokenize(output)                          -> [1, 3, 1024, 32, 32]
 *
 * Notes vs the reference implementation:
 *   - Q/K/V projections have NO bias (attention_bias=False in the config).
 *     to_out (post-attention projection) DOES have bias.
 *   - LayerNorm eps defaults to 1e-5 in nn.LayerNorm (different from DINO's 1e-12).
 *   - SelfAttn uses Q from triplane only; the diffusers `only_cross_attention`
 *     flag is False here, so KV come from triplane too.
 *   - The "first chunk is hidden, second is gate" in GEGLU matters - matches
 *     diffusers/models/attention.py::GEGLU.forward().
 *   - Output layout is [Np=3, Ct=1024, Hp=32, Wp=32]; we expose this directly.
 *
 * Weight pointers borrowed from the safetensors mmap; the only owned
 * allocation is the cached triplane query tokens (the learned parameter is
 * stored in safetensors as [3, 1024, 32, 32] and we flatten/cache it as
 * [1024, 3072] to skip the rearrange at every forward).
 */

#ifndef LRM_LRM_TRIPLANE_DECODER_H
#define LRM_LRM_TRIPLANE_DECODER_H

#include <stddef.h>

#include "iris_safetensors.h"

#define LRM_TRIPLANE_MAX_LAYERS 16

typedef struct lrm_triplane_decoder {
    int hidden_dim;       /* 1024 */
    int num_layers;       /* 16 */
    int num_heads;        /* 16 */
    int head_dim;         /* 64 */
    int ff_inner;         /* 4096 (post-GEGLU split); GEGLU proj is 2x this */
    int cross_kv_dim;     /* 768 (image tokens) */
    int triplane_planes;  /* 3 */
    int triplane_size;    /* 32 */
    int num_tokens;       /* 3 * 32 * 32 = 3072 */

    /* Initial GroupNorm (32 groups over channels-axis-1) + proj_in/out. */
    const float *gn_w, *gn_b;        /* both [1024] */
    const float *proj_in_w;          /* [1024, 1024] */
    const float *proj_in_b;          /* [1024] */
    const float *proj_out_w;         /* [1024, 1024] */
    const float *proj_out_b;         /* [1024] */

    /* Cached triplane query tokens in the layout the forward consumes.
     * Source: tokenizer.embeddings [3, 1024, 32, 32] (Np Ct Hp Wp).
     * Cache:  [1024, 3072] = [Ct, Np*Hp*Wp], matching Triplane1DTokenizer.forward.
     * Owned. */
    float *queries_chl;  /* channels-first layout, [1024, 3072] */

    /* Per-block weights. */
    struct {
        /* Pre-attn LayerNorms (default torch eps=1e-5). */
        const float *norm1_w, *norm1_b;     /* [1024] */
        const float *norm2_w, *norm2_b;     /* [1024] */
        const float *norm3_w, *norm3_b;     /* [1024] */

        /* Self-attn (Q/K/V from triplane). No bias on Q/K/V; bias on to_out. */
        const float *self_q_w, *self_k_w, *self_v_w;   /* [1024, 1024] */
        const float *self_out_w, *self_out_b;          /* [1024,1024], [1024] */

        /* Cross-attn (Q from triplane, K/V from image tokens 768-d). */
        const float *cross_q_w;                        /* [1024, 1024] */
        const float *cross_k_w, *cross_v_w;            /* [1024, 768] */
        const float *cross_out_w, *cross_out_b;        /* [1024,1024], [1024] */

        /* GEGLU FFN: proj 1024 -> 8192 (split into hidden+gate of 4096 each),
         * then proj 4096 -> 1024. */
        const float *ff_geglu_w, *ff_geglu_b;          /* [8192,1024], [8192] */
        const float *ff_out_w, *ff_out_b;              /* [1024,4096], [1024] */
    } blocks[LRM_TRIPLANE_MAX_LAYERS];
} lrm_triplane_decoder;

/* Bind weight pointers + cache flattened triplane queries.
 * Returns 0 on success; -1 with iris_get_error() set on failure. */
int  lrm_triplane_decoder_init(lrm_triplane_decoder *dec,
                               const safetensors_file_t *sf);

/* Release owned allocations. */
void lrm_triplane_decoder_release(lrm_triplane_decoder *dec);

/* Workspace size for forward(). */
size_t lrm_triplane_decoder_workspace_bytes(const lrm_triplane_decoder *dec);

/*
 * Forward.
 *   image_tokens : [num_image_tokens, cross_kv_dim] -- DINO output, 1025 x 768
 *   num_image_tokens : number of image tokens (1025 for 512-input DINO)
 *   triplane_out : output buffer of size 3 * 1024 * 32 * 32 = 3,145,728 floats.
 *                  Layout matches PyTorch's tokenizer.detokenize():
 *                  [Np=3, Ct=1024, Hp=32, Wp=32] C-contiguous.
 *   workspace    : scratch >= lrm_triplane_decoder_workspace_bytes(dec).
 *
 * Returns 0 on success.
 */
int lrm_triplane_decoder_forward(const lrm_triplane_decoder *dec,
                                 const float *image_tokens,
                                 int num_image_tokens,
                                 float *triplane_out,
                                 float *workspace);

#endif /* LRM_LRM_TRIPLANE_DECODER_H */
