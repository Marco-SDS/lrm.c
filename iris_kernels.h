/*
 * Iris Math Kernels - Header
 *
 * Low-level math operations for the Iris inference engine.
 * All operations work on float32 tensors in row-major order.
 */

#ifndef IRIS_KERNELS_H
#define IRIS_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* Forward declarations */
struct iris_image;

/* Fast exponential approximation using range reduction + degree-5 polynomial.
 * Relative error < 2e-6 across the full float range.
 * Inlined so the compiler can auto-vectorize loops that use it. */
static inline float fast_expf(float x) {
    if (x < -87.3f) return 0.0f;
    if (x > 88.7f) return 1e38f;
    float n = floorf(x * 1.4426950408889634f + 0.5f);
    float r = x - n * 0.6931471805599453f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f +
              r * (0.04166667f + r * 0.00833333f))));
    union { float f; int32_t i; } v;
    v.f = p;
    v.i += (int32_t)n << 23;
    return v.f;
}

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

/* Element-wise operations */
void iris_add(float *out, const float *a, const float *b, int n);

/* In-place variants */
void iris_add_inplace(float *a, const float *b, int n);
void iris_mul_inplace(float *a, const float *b, int n);

/* Accumulate: a += scale * b */
void iris_axpy(float *a, float scale, const float *b, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/*
 * General matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 */
void iris_matmul(float *C, const float *A, const float *B,
                 int M, int K, int N);

/*
 * Matrix multiplication with transposed B: C = A @ B^T
 * A: [M, K], B: [N, K], C: [M, N]
 */
void iris_matmul_t(float *C, const float *A, const float *B,
                   int M, int K, int N);

/*
 * Linear layer: y = x @ W^T + b (if b != NULL)
 * x: [seq_len, in_dim], W: [out_dim, in_dim], b: [out_dim], y: [seq_len, out_dim]
 */
void iris_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim);

/*
 * Linear layer without bias: y = x @ W^T
 */
void iris_linear_nobias(float *y, const float *x, const float *W,
                        int seq_len, int in_dim, int out_dim);

/*
 * Linear layer without bias using bf16 weights
 * x: [seq_len, in_dim] (f32), W: [out_dim, in_dim] (bf16), y: [seq_len, out_dim] (f32)
 */
void iris_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                             int seq_len, int in_dim, int out_dim);

/* ========================================================================
 * GPU Batch Operations
 * These functions allow batching multiple GPU operations to reduce sync overhead.
 * On non-GPU builds, these are no-ops.
 * ======================================================================== */

/*
 * Begin a batch of GPU operations.
 * Operations after this call are queued but not executed until iris_gpu_end_batch().
 * NOTE: Only use for INDEPENDENT operations (outputs don't feed into subsequent inputs).
 */
void iris_gpu_begin_batch(void);

/*
 * End a batch of GPU operations.
 * Executes all queued operations and waits for completion.
 */
void iris_gpu_end_batch(void);

/* ========================================================================
 * Convolution Operations
 * ======================================================================== */

/*
 * 2D Convolution: out = conv2d(in, weight, bias)
 * in: [batch, in_ch, H, W]
 * weight: [out_ch, in_ch, kH, kW]
 * bias: [out_ch] (can be NULL)
 * out: [batch, out_ch, outH, outW]
 */
void iris_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int batch, int in_ch, int out_ch, int H, int W,
                 int kH, int kW, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/*
 * RMS Normalization (no mean centering, no bias)
 * x: [seq_len, hidden], weight: [hidden]
 */
void iris_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps);

/*
 * Group Normalization
 * x: [batch, channels, H, W], gamma/beta: [channels]
 */
void iris_group_norm(float *out, const float *x, const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, int num_groups, float eps);

/*
 * Batch Normalization (inference mode with running stats)
 * x: [batch, channels, H, W]
 */
void iris_batch_norm(float *out, const float *x,
                     const float *running_mean, const float *running_var,
                     const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, float eps);

/*
 * Layer Normalization with affine params.
 * Normalizes over the last dimension (`hidden`) per row, then applies the
 * affine transform `gamma * normalized + beta`. Pass NULL for either gamma
 * or beta to skip that part of the affine. Used by DINO ViT (12 LayerNorms
 * per block) and the TripoSR triplane decoder.
 * x: [seq_len, hidden], gamma: [hidden] or NULL, beta: [hidden] or NULL
 */
void iris_layer_norm(float *out, const float *x,
                     const float *gamma, const float *beta,
                     int seq_len, int hidden, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* SiLU / Swish activation: x * sigmoid(x) */
void iris_silu(float *x, int n);

/* Fused SiLU(gate) * up - single pass for SwiGLU */
void iris_silu_mul(float *gate, const float *up, int n);

/* Exact GELU activation: 0.5 * x * (1 + erf(x / sqrt(2))). Inplace.
 * Used by DINO ViT MLP. Note: this is the *exact* form, not the tanh
 * approximation used elsewhere. */
void iris_gelu(float *x, int n);

/* GEGLU activation: out[i] = hidden[i] * gelu(gate[i]).
 * Used by the TripoSR triplane decoder FFN. Diffusers' BasicTransformerBlock
 * computes proj(x) -> [..., 2*dim], splits chunks (hidden, gate), then
 * applies this kernel. Callers do the split via pointer arithmetic. */
void iris_geglu(float *out, const float *hidden, const float *gate, int n);

/* Softmax over last dimension */
void iris_softmax(float *x, int rows, int cols);
void iris_softmax_cpu(float *x, int rows, int cols);

/* ========================================================================
 * Spatial Sampling
 * ======================================================================== */

/*
 * Bilinear grid_sample (PyTorch-compatible).
 * For each of N_planes feature maps, samples N_points locations using
 * bilinear interpolation. Grid coordinates are normalized [-1, +1] in
 * (x=column, y=row) order. Uses padding_mode='border' (out-of-range
 * coordinates clamp to the edge) and align_corners=False (so a normalized
 * coordinate of -1 maps to half a pixel left of pixel 0).
 *
 * input:  [N_planes, C, H, W]
 * grid:   [N_planes, N_points, 2]  -- (x, y) per point, normalized [-1, 1]
 * out:    [N_planes, C, N_points]
 *
 * Used by the TripoSR triplane sampler (N_planes=3, C=40, H=W=64). At MC
 * resolution 256, this kernel is called with N_points ~= 16.78M; it is
 * the wall-clock bottleneck after the decoder. Phase 4 ships a clean CPU
 * implementation; Phase 13 adds a tiled Metal version.
 */
void iris_grid_sample_bilinear(float *out, const float *input,
                               const float *grid,
                               int N_planes, int C, int H, int W,
                               int N_points);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Scaled dot-product attention
 * Q: [batch, heads, seq_q, head_dim]
 * K: [batch, heads, seq_k, head_dim]
 * V: [batch, heads, seq_k, head_dim]
 * out: [batch, heads, seq_q, head_dim]
 * scale: typically 1/sqrt(head_dim)
 */
void iris_attention(float *out, const float *Q, const float *K, const float *V,
                    int batch, int heads, int seq_q, int seq_k, int head_dim,
                    float scale);

/*
 * Flash attention - memory-efficient tiled attention.
 * Uses online softmax to avoid materializing O(n²) attention matrix.
 * Memory: O(seq_q + tile_size²) instead of O(seq_q × seq_k).
 *
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 */
void iris_flash_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int heads, int head_dim, float scale);

/*
 * Apply rotary position embeddings (RoPE)
 * x: [batch, seq, heads, head_dim]
 * freqs: [seq, head_dim/2, 2] (cos, sin pairs)
 */
void iris_apply_rope(float *x, const float *freqs,
                     int batch, int seq, int heads, int head_dim);

/*
 * Compute RoPE frequencies
 * pos: position indices [seq]
 * freqs: output [seq, dim/2, 2]
 */
void iris_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta);

/* ========================================================================
 * Pooling and Reshape
 * ======================================================================== */

/* Upsample with nearest neighbor */
void iris_upsample_nearest(float *out, const float *in,
                           int batch, int channels, int H, int W,
                           int scale_h, int scale_w);

/* Patchify: [B, C, H, W] -> [B, C*p*p, H/p, W/p] */
void iris_patchify(float *out, const float *in,
                   int batch, int channels, int H, int W, int patch_size);

/* Unpatchify: [B, C*p*p, H, W] -> [B, C, H*p, W*p] */
void iris_unpatchify(float *out, const float *in,
                     int batch, int channels, int H, int W, int patch_size);

/* ========================================================================
 * Random Number Generation
 * ======================================================================== */

/* Initialize RNG with seed */
void iris_rng_seed(uint64_t seed);

/* Generate uniform random [0, 1) */
float iris_random_uniform(void);

/* Generate standard normal using Box-Muller */
float iris_random_normal(void);

/* Fill tensor with random normal values */
void iris_randn(float *out, int n);

/* Fill tensor with uniform random [0, 1) */
void iris_rand(float *out, int n);

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/* Copy tensor */
void iris_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Progress Callbacks
 * ======================================================================== */

/* Substep types during transformer forward pass */
typedef enum {
    IRIS_SUBSTEP_DOUBLE_BLOCK,   /* Double-stream block completed */
    IRIS_SUBSTEP_SINGLE_BLOCK,   /* Single-stream block completed */
    IRIS_SUBSTEP_FINAL_LAYER,    /* Final layer completed */
} iris_substep_type_t;

/*
 * Substep callback - called during transformer forward pass.
 * type: which operation completed
 * index: 0-based index of this substep within its type
 * total: total count for this substep type
 */
typedef void (*iris_substep_callback_t)(iris_substep_type_t type, int index, int total);

/*
 * Step callback - called at sampling step boundaries.
 * step: current step (1-based), or 0 to indicate sampling is starting
 * total: total number of steps
 */
typedef void (*iris_step_callback_t)(int step, int total);

/* Global callback pointers - set by caller before inference */
extern iris_substep_callback_t iris_substep_callback;
extern iris_step_callback_t iris_step_callback;

/*
 * Phase callback - called at major phase boundaries.
 * phase: descriptive name ("encoding text", "decoding image", etc.)
 * done: 0 when starting, 1 when finished
 */
typedef void (*iris_phase_callback_t)(const char *phase, int done);
extern iris_phase_callback_t iris_phase_callback;

/* The diffusion-era step_image / text_progress / vae_progress callback
 * extern declarations used to live here. They were tied to the iterative
 * Euler denoising loop, the Qwen3 text encoder, and the VAE decoder - all
 * removed in Phase 2. LRM-appropriate stage callbacks (encoder progress,
 * decoder progress, MC progress) will be reintroduced in Phase 4. */

/* Global verbose flag - when 0, library code suppresses diagnostic output */
extern int iris_verbose;

#endif /* IRIS_KERNELS_H */
