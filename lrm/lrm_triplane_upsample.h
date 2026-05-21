/*
 * lrm_triplane_upsample.h - Post-processor (ConvTranspose2d 1024 -> 40, 2x2 / s=2).
 *
 * TripoSR's post_processor is a single nn.ConvTranspose2d with kernel=2,
 * stride=2 applied per triplane. Because kernel size equals stride, the
 * output 2x2 blocks do NOT overlap and the operation is mathematically
 * equivalent to:
 *
 *   1. A linear projection in_ch -> (out_ch * kernel * kernel)
 *      i.e. 1024 -> 160 over each input pixel
 *   2. PixelShuffle(2): scatter the 160 channels into a 2x2 spatial block
 *      per output channel, producing [out_ch=40, 2H, 2W]
 *
 * That lets us express the whole post-processor as one BLAS sgemm of shape
 * (160, 1024) @ (1024, 32*32) plus a small scatter. On CPU this is ~30x
 * faster than the equivalent nested-loop ConvTranspose2d and matches the
 * f32 output bit-for-bit (single big accumulation vs many small ones).
 *
 * Weights are reordered at init into a packed [160, 1024] row-major matrix
 * suitable for iris_matmul; the original PyTorch [1024, 40, 2, 2] tensor
 * has the C_in axis on the outside, which would force a transpose in the
 * hot path.
 */

#ifndef LRM_LRM_TRIPLANE_UPSAMPLE_H
#define LRM_LRM_TRIPLANE_UPSAMPLE_H

#include <stddef.h>

#include "iris_safetensors.h"

typedef struct lrm_triplane_upsample {
    int in_ch;       /* 1024 */
    int out_ch;      /* 40 */
    int kernel;      /* 2 (== stride; no overlap) */
    int stride;      /* 2 */
    int planes;      /* 3 */
    int in_size;     /* 32 */
    int out_size;    /* 64 = (in_size - 1) * stride + kernel */

    /* Packed weight in [out_ch * kernel * kernel, in_ch] = [160, 1024]
     * row-major. Re-ordered at init from the safetensors ConvTranspose2d
     * weight [in_ch, out_ch, kH, kW]. Owned. */
    float *packed_w;

    /* Bias [out_ch=40], borrowed from the safetensors mmap. */
    const float *bias;
} lrm_triplane_upsample;

int  lrm_triplane_upsample_init(lrm_triplane_upsample *up,
                                const safetensors_file_t *sf);
void lrm_triplane_upsample_release(lrm_triplane_upsample *up);

/* Workspace size in bytes for a single forward call. */
size_t lrm_triplane_upsample_workspace_bytes(const lrm_triplane_upsample *up);

/*
 * Forward.
 *   in    : [Np=3, in_ch=1024, in_size=32, in_size=32]   C-contiguous
 *   out   : [Np=3, out_ch=40,  out_size=64, out_size=64] C-contiguous
 *   work  : scratch >= lrm_triplane_upsample_workspace_bytes(up)
 *
 * Returns 0 on success.
 */
int lrm_triplane_upsample_forward(const lrm_triplane_upsample *up,
                                  const float *in,
                                  float *out,
                                  float *work);

#endif /* LRM_LRM_TRIPLANE_UPSAMPLE_H */
