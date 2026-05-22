/*
 * lrm_triposr.h - TripoSR-specific loader and metadata.
 *
 * Phase 5 deliverable: mmap the safetensors file, validate that the set of
 * tensors matches the expected TripoSR layout, and expose a print-tree
 * diagnostic for the `lrmc info` subcommand. Per-module forward passes
 * arrive in Phases 6-10.
 *
 * TripoSR weight layout (verified against the canonical model.ckpt at HF
 * revision 5b521936 by inspecting all 549 tensors):
 *
 *   image_tokenizer.*      200 tensors  -- DINO ViT-B/16, hidden=768, 12 layers
 *   tokenizer.embeddings     1 tensor   -- learned triplane queries [3, 1024, 32, 32]
 *   backbone.*             326 tensors  -- Transformer1D, hidden=1024, 16 blocks
 *                                          (norm + proj_in/out + 20 per block: norm1/2/3,
 *                                          attn1.{q,k,v,to_out}, attn2.{q,k,v,to_out},
 *                                          ff.net.0.proj, ff.net.2)
 *   post_processor.*         2 tensors  -- ConvTranspose2d 1024->40, kernel 2x2, stride 2
 *   decoder.*               20 tensors  -- NeRFMLP, Linear layers at even indices
 *                                          0,2,...,18 (10 layers; odd indices are SiLU)
 *                                          Layer 0: 120->64; layers 2-16: 64->64;
 *                                          layer 18: 64->4
 *   ---
 *   total                  549 tensors, all float32
 */

#ifndef LRM_LRM_TRIPOSR_H
#define LRM_LRM_TRIPOSR_H

#include <stdio.h>

#include "iris.h"
#include "iris_safetensors.h"
#include "lrm.h"
#include "lrm_nerf_mlp.h"
#include "lrm_triplane_decoder.h"
#include "lrm_triplane_sample.h"
#include "lrm_triplane_upsample.h"
#include "lrm_vit_dino.h"

/* TripoSR model state. `struct lrm_model` is currently an alias for this:
 * we only ship one model kind. When OpenLRM lands in Phase 15 we'll
 * introduce a kind tag in struct lrm_model and refactor.
 *
 * The safetensors handle owns the mmap; tensor pointers obtained via
 * safetensors_data() are borrowed into that mmap and must not be freed. */
struct lrm_model {
    safetensors_file_t *st;

    /* Cached architectural constants (filled in by lrm_triposr_load).
     * These mirror tsr/config.yaml of the pinned TripoSR commit. */
    int   cond_image_size;     /* 512 */
    int   dino_hidden;         /* 768 */
    int   dino_layers;         /* 12 */
    int   backbone_hidden;     /* 1024 */
    int   backbone_layers;     /* 16 */
    int   backbone_kv_dim;     /* 768 (cross-attn K/V from DINO) */
    int   triplane_planes;     /* 3 */
    int   triplane_channels;   /* 40 (post-upsample) */
    int   triplane_size;       /* 64 */
    int   decoder_linear_layers; /* 10 */
    float radius;              /* 0.87 */
    float density_threshold;   /* 25.0 */

    /* Initialized sub-modules (Phase 12). All weight pointers inside
     * these structs borrow from `st`; release lifetime follows `st`. */
    lrm_vit_dino             vit;
    lrm_triplane_decoder     decoder;
    lrm_triplane_upsample    upsample;
    lrm_triplane_sample_cfg  sample_cfg;
    lrm_nerf_mlp             mlp;
    int                      modules_ready;  /* 1 once all sub-inits succeeded */
};

/* Preprocess an input image for the DINO encoder (Phase 12).
 *
 *   - If the image is RGBA with non-255 alpha, crop to alpha bbox, pad to
 *     square, pad to fg_ratio (0.85) of canvas, composite over gray 0.5.
 *   - If the image is RGB or fully-opaque RGBA, no rembg-style cleanup
 *     is done (caller is responsible).
 *   - Resize to cond_image_size x cond_image_size with plain bilinear.
 *   - Convert to f32 [0, 1] and transpose to CHW.
 *
 * out must point to 3 * cond_image_size * cond_image_size floats.
 * Returns 0 on success, -1 with iris_get_error() set on failure.
 *
 * Note: this uses plain bilinear instead of PyTorch's antialiased
 * bilinear. Differences are small (<1% per-pixel) and the model-level
 * parity tests (Phases 6-10) already validate the rest of the chain.
 */
int lrm_triposr_preprocess(const struct lrm_model *m,
                           const iris_image *im,
                           float *out_chw);

/* Open a TripoSR safetensors file (path can be a directory containing
 * model.safetensors, or a direct .safetensors path). On success, returns
 * a populated model; on failure, returns NULL and sets iris_get_error(). */
struct lrm_model *lrm_triposr_load(const char *model_dir);

/* Release the model state and the underlying mmap. Safe on NULL. */
void lrm_triposr_free(struct lrm_model *m);

/* Diagnostic: print the loaded tensor tree to `out`. Groups by top-level
 * prefix; the header includes file size + total tensor count. Used by
 * `lrmc info`. */
void lrm_triposr_print_tree(const struct lrm_model *m, FILE *out);

#endif /* LRM_LRM_TRIPOSR_H */
