/*
 * lrm_vit_dino.h - DINOv1 ViT-B/16 image encoder.
 *
 * Mirrors `facebook/dino-vitb16` as it ships in `transformers.ViTModel`, with
 * the pre-norm BasicTransformerBlock layout HF uses. Output retains all
 * 1 + (H/16)^2 tokens (CLS included), to match TripoSR's cross-attention
 * input contract.
 *
 * Architecture (when input_size=512):
 *   patch_embed (Conv2d 3->768, kernel 16, stride 16)
 *     -> [768, 32, 32] -> flatten to 1024 tokens of dim 768
 *   prepend learned CLS token -> [1025, 768]
 *   add bicubic-interpolated position embeddings (197 -> 1025)
 *   12 blocks:
 *     x = x + attn(layernorm_before(x))
 *     x = x + mlp_gelu(layernorm_after(x))
 *   final LayerNorm
 *   -> [1, 1025, 768]
 *
 * Weight pointers borrowed from a safetensors file; the only owned
 * allocation is the bicubic-interpolated position embedding cache.
 */

#ifndef LRM_LRM_VIT_DINO_H
#define LRM_LRM_VIT_DINO_H

#include <stddef.h>

#include "iris_safetensors.h"

#define LRM_DINO_MAX_LAYERS 12

typedef struct lrm_vit_dino {
    int input_size;             /* 512 for TripoSR */
    int patch_size;             /* 16 */
    int hidden_dim;             /* 768 */
    int num_layers;             /* 12 */
    int num_heads;              /* 12 */
    int head_dim;               /* 64 */
    int mlp_dim;                /* 3072 */
    int patches_per_side;       /* input_size / patch_size = 32 */
    int num_tokens;             /* 1 + patches_per_side^2 = 1025 */

    /* Embedding-stage pointers (borrowed from the safetensors mmap). */
    const float *patch_embed_w; /* [768, 3, 16, 16] */
    const float *patch_embed_b; /* [768] */
    const float *cls_token;     /* [1, 1, 768] */

    /* Position embeddings, bicubic-interpolated at init from the original
     * 14x14 grid to patches_per_side x patches_per_side, with the CLS row
     * preserved verbatim. Layout: [num_tokens, hidden_dim]. Owned. */
    float *pos_embed;

    /* Per-block weight pointers (borrowed). HF's BertSelfAttention layout:
     * each of Q/K/V is a separate Linear (weight + bias), the attention
     * output is another Linear, and the MLP is FC1 (with GELU) + FC2.
     * LayerNorms wrap the attention and MLP sub-blocks in pre-norm style. */
    struct {
        const float *ln_before_w, *ln_before_b;     /* applied before attn */
        const float *q_w, *q_b;
        const float *k_w, *k_b;
        const float *v_w, *v_b;
        const float *attn_out_w, *attn_out_b;       /* attention output proj */
        const float *ln_after_w, *ln_after_b;       /* applied before mlp */
        const float *fc1_w, *fc1_b;                 /* 768 -> 3072 */
        const float *fc2_w, *fc2_b;                 /* 3072 -> 768 */
    } blocks[LRM_DINO_MAX_LAYERS];

    /* Final LayerNorm (HF ViT applies one after the 12 blocks). */
    const float *final_ln_w, *final_ln_b;
} lrm_vit_dino;

/*
 * Bind weight pointers from the given safetensors file and pre-compute the
 * interpolated position embedding cache. Returns 0 on success; on failure
 * returns -1 and sets iris_get_error(). The safetensors handle must outlive
 * the vit (this borrows pointers into its mmap).
 */
int lrm_vit_dino_init(lrm_vit_dino *vit, const safetensors_file_t *sf,
                      int input_size);

/* Release owned allocations. */
void lrm_vit_dino_release(lrm_vit_dino *vit);

/*
 * Total scratch buffer size required by lrm_vit_dino_forward, in bytes.
 * Caller allocates a single buffer of this size and passes it in.
 */
size_t lrm_vit_dino_workspace_bytes(const lrm_vit_dino *vit);

/*
 * Forward pass.
 * image:        input [1, 3, H, W] (where H=W=input_size), already
 *               preprocessed (rembg + fg-rescale + gray composite + ImageNet
 *               normalize). Layout: C-contiguous BCHW.
 * tokens_out:   output [1, num_tokens, hidden_dim]. Caller allocates.
 * workspace:    scratch of >= lrm_vit_dino_workspace_bytes(vit).
 *
 * Returns 0 on success.
 */
int lrm_vit_dino_forward(const lrm_vit_dino *vit,
                         const float *image,
                         float *tokens_out,
                         float *workspace);

#endif /* LRM_LRM_VIT_DINO_H */
