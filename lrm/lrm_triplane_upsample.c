/*
 * lrm_triplane_upsample.c - ConvTranspose2d (1024 -> 40, 2x2 / stride 2) via
 * a single BLAS sgemm + pixel-shuffle scatter. See lrm_triplane_upsample.h
 * for the math justifying this rewrite.
 *
 * Hot path per plane:
 *   C = A @ B            iris_matmul, shape (160, 1024) @ (1024, 32*32)
 *     A = packed weights [160, 1024]   (out_ch*kk x in_ch)
 *     B = input plane    [1024, 32*32] (= [in_ch, H*W] C-contiguous)
 *     C = shuffled out   [160, 32*32]
 *   scatter: for each (c_out, ky, kx, y, x): out[c_out, 2y+ky, 2x+kx]
 *            = C[c_out*4 + ky*2 + kx, y*32 + x] + bias[c_out]
 *
 * The scatter is O(out_ch * out_H * out_W) = 40 * 64 * 64 = 163,840 writes
 * per plane, negligible next to the 167 MFlops sgemm.
 */

#include "lrm_triplane_upsample.h"

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
        set_err("triplane_upsample: missing tensor '%s'", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        set_err("triplane_upsample: '%s' dtype %d, expected f32",
                name, (int)t->dtype);
        return NULL;
    }
    if (expected_ndim > 0) {
        if (t->ndim != expected_ndim) {
            set_err("triplane_upsample: '%s' ndim %d, expected %d",
                    name, t->ndim, expected_ndim);
            return NULL;
        }
        for (int i = 0; i < expected_ndim; i++) {
            if (t->shape[i] != expected_shape[i]) {
                set_err("triplane_upsample: '%s' shape axis %d mismatch "
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
 * Init / Release
 * ======================================================================== */

int lrm_triplane_upsample_init(lrm_triplane_upsample *up,
                               const safetensors_file_t *sf) {
    memset(up, 0, sizeof(*up));
    up->in_ch   = 1024;
    up->out_ch  = 40;
    up->kernel  = 2;
    up->stride  = 2;
    up->planes  = 3;
    up->in_size = 32;
    up->out_size = (up->in_size - 1) * up->stride + up->kernel;  /* 64 */

    int64_t s_w[] = { up->in_ch, up->out_ch, up->kernel, up->kernel };  /* [1024, 40, 2, 2] */
    int64_t s_b[] = { up->out_ch };
    const float *src_w = find_f32(sf, "post_processor.upsample.weight", 4, s_w);
    const float *src_b = find_f32(sf, "post_processor.upsample.bias",   1, s_b);
    if (!src_w || !src_b) return -1;

    const int kk = up->kernel * up->kernel;              /* 4 */
    const int out_packed = up->out_ch * kk;              /* 160 */
    const size_t packed_n = (size_t)out_packed * up->in_ch;
    up->packed_w = (float *)malloc(packed_n * sizeof(float));
    if (!up->packed_w) return set_err("triplane_upsample: oom for packed weight");

    /* Rearrange [c_in, c_out, ky, kx] -> [(c_out*4 + ky*2 + kx), c_in].
     * Source stride: c_in major (c_in dim stride = out_ch * kH * kW = 160).
     * Target stride: r major (r dim stride = in_ch = 1024). */
    for (int c_in = 0; c_in < up->in_ch; c_in++) {
        for (int c_out = 0; c_out < up->out_ch; c_out++) {
            for (int ky = 0; ky < up->kernel; ky++) {
                for (int kx = 0; kx < up->kernel; kx++) {
                    int r = c_out * kk + ky * up->kernel + kx;
                    up->packed_w[(size_t)r * up->in_ch + c_in] =
                        src_w[((size_t)c_in * up->out_ch + c_out) * kk
                              + ky * up->kernel + kx];
                }
            }
        }
    }

    up->bias = src_b;
    return 0;
}

void lrm_triplane_upsample_release(lrm_triplane_upsample *up) {
    if (!up) return;
    free(up->packed_w);
    up->packed_w = NULL;
}

/* ========================================================================
 * Workspace
 * ======================================================================== */

size_t lrm_triplane_upsample_workspace_bytes(const lrm_triplane_upsample *up) {
    /* One per-plane scratch holding the GEMM output before scatter. */
    const int kk = up->kernel * up->kernel;
    size_t floats = (size_t)up->out_ch * kk
                  * (size_t)up->in_size * up->in_size;
    return floats * sizeof(float);
}

/* ========================================================================
 * Forward
 * ======================================================================== */

int lrm_triplane_upsample_forward(const lrm_triplane_upsample *up,
                                  const float *in,
                                  float *out,
                                  float *work) {
    if (!up || !in || !out || !work) {
        return set_err("triplane_upsample: NULL argument to forward");
    }

    const int C_in   = up->in_ch;       /* 1024 */
    const int C_out  = up->out_ch;      /* 40   */
    const int K      = up->kernel;      /* 2    */
    const int kk     = K * K;            /* 4    */
    const int H_in   = up->in_size;     /* 32   */
    const int W_in   = up->in_size;     /* 32   */
    const int H_out  = up->out_size;    /* 64   */
    const int W_out  = up->out_size;    /* 64   */
    const int SP_in  = H_in * W_in;     /* 1024 */
    const int Rout   = C_out * kk;      /* 160  */

    const size_t in_plane_stride  = (size_t)C_in  * SP_in;
    const size_t out_plane_stride = (size_t)C_out * H_out * W_out;

    for (int p = 0; p < up->planes; p++) {
        const float *in_plane  = in  + p * in_plane_stride;
        float       *out_plane = out + p * out_plane_stride;

        /* GEMM: shuffled[Rout, SP_in] = packed_w[Rout, C_in] @ in_plane[C_in, SP_in].
         * in_plane is [C_in, H, W] C-contiguous, which is the same row-major
         * [C_in, H*W] memory layout that iris_matmul consumes. */
        iris_matmul(work, up->packed_w, in_plane,
                    /*M=*/Rout, /*K=*/C_in, /*N=*/SP_in);

        /* Pixel-shuffle scatter + bias add.
         *
         * shuffled[r=c_out*4 + ky*2 + kx, s=y*32 + x]
         *   -> out[c_out, 2y+ky, 2x+kx]
         *
         * Iteration order favors output-locality: for each (c_out, y, x)
         * we write the 4 outputs for (ky, kx). Each write reads one entry
         * from `work` with stride (W_in*H_in) inside the c_out group. */
        for (int c_out = 0; c_out < C_out; c_out++) {
            float b = up->bias[c_out];
            for (int y = 0; y < H_in; y++) {
                for (int x = 0; x < W_in; x++) {
                    int s = y * W_in + x;
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int r = c_out * kk + ky * K + kx;
                            float v = work[(size_t)r * SP_in + s] + b;
                            int oy = 2 * y + ky;
                            int ox = 2 * x + kx;
                            out_plane[((size_t)c_out * H_out + oy) * W_out + ox] = v;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
