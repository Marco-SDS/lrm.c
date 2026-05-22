/*
 * lrm_nerf_mlp.c - NeRF-style decoder MLP.
 *
 * 10 Linear layers, SiLU between them. The hot path is a chain of GEMMs:
 *   y_0 = SiLU(Linear_{120->64}(features))
 *   y_1 = SiLU(Linear_{64->64}(y_0))   ... 8 times ...
 *   y_8 = SiLU(Linear_{64->64}(y_7))
 *   out = Linear_{64->4}(y_8)         (no activation)
 * Then channel 0 -> exp(raw + density_bias), channels 1..3 -> sigmoid.
 *
 * At N_points=262144 (the 64^3 golden grid), the workload is:
 *   - layer 0:    262144 * 120 *  64 =   2.0 GFLOPS
 *   - layers 1-8: 262144 *  64 *  64 * 8 = 8.6 GFLOPS
 *   - layer 9:    262144 *  64 *   4 =   0.07 GFLOPS
 * Roughly 11 GFLOPS total, dominated by the middle 8 layers. With BLAS
 * sgemm on M-series hardware this is ~tens of ms; with naive C several
 * hundred ms. The forward calls iris_linear which routes to Accelerate.
 *
 * Memory:
 *   - One double-buffered scratch of [N_points, 64] for the hidden state.
 *   - One [N_points, 4] for the raw output before activations.
 * Both can be inside the workspace; the bigger of the two
 * (N_points * 64) dominates.
 */

#include "lrm_nerf_mlp.h"

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
        set_err("nerf_mlp: missing tensor '%s'", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        set_err("nerf_mlp: '%s' dtype %d, expected f32", name, (int)t->dtype);
        return NULL;
    }
    if (expected_ndim > 0) {
        if (t->ndim != expected_ndim) {
            set_err("nerf_mlp: '%s' ndim %d, expected %d",
                    name, t->ndim, expected_ndim);
            return NULL;
        }
        for (int i = 0; i < expected_ndim; i++) {
            if (t->shape[i] != expected_shape[i]) {
                set_err("nerf_mlp: '%s' shape axis %d "
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
 * Init
 * ======================================================================== */

int lrm_nerf_mlp_init(lrm_nerf_mlp *mlp, const safetensors_file_t *sf) {
    memset(mlp, 0, sizeof(*mlp));
    mlp->in_channels  = 120;
    mlp->hidden_dim   = 64;
    mlp->out_channels = 4;
    mlp->num_layers   = 10;
    mlp->density_bias = -1.0f;

    /* TripoSR's NeRFMLP places its 10 Linears at Sequential indices
     * 0, 2, 4, ..., 18 with SiLU layers in the odd slots. */
    for (int i = 0; i < mlp->num_layers; i++) {
        mlp->seq_index[i] = 2 * i;
    }

    char nm[128];
    /* Layer 0: in -> hidden. */
    {
        int64_t s_w[] = { mlp->hidden_dim, mlp->in_channels };  /* [64, 120] */
        int64_t s_b[] = { mlp->hidden_dim };
        snprintf(nm, sizeof(nm), "decoder.layers.%d.weight", mlp->seq_index[0]);
        mlp->weight[0] = find_f32(sf, nm, 2, s_w);
        snprintf(nm, sizeof(nm), "decoder.layers.%d.bias", mlp->seq_index[0]);
        mlp->bias[0] = find_f32(sf, nm, 1, s_b);
        if (!mlp->weight[0] || !mlp->bias[0]) return -1;
    }
    /* Layers 1..(num_layers - 2): hidden -> hidden. */
    for (int i = 1; i < mlp->num_layers - 1; i++) {
        int64_t s_w[] = { mlp->hidden_dim, mlp->hidden_dim };
        int64_t s_b[] = { mlp->hidden_dim };
        snprintf(nm, sizeof(nm), "decoder.layers.%d.weight", mlp->seq_index[i]);
        mlp->weight[i] = find_f32(sf, nm, 2, s_w);
        snprintf(nm, sizeof(nm), "decoder.layers.%d.bias", mlp->seq_index[i]);
        mlp->bias[i] = find_f32(sf, nm, 1, s_b);
        if (!mlp->weight[i] || !mlp->bias[i]) return -1;
    }
    /* Final layer: hidden -> out. */
    {
        int last = mlp->num_layers - 1;
        int64_t s_w[] = { mlp->out_channels, mlp->hidden_dim };
        int64_t s_b[] = { mlp->out_channels };
        snprintf(nm, sizeof(nm), "decoder.layers.%d.weight", mlp->seq_index[last]);
        mlp->weight[last] = find_f32(sf, nm, 2, s_w);
        snprintf(nm, sizeof(nm), "decoder.layers.%d.bias", mlp->seq_index[last]);
        mlp->bias[last] = find_f32(sf, nm, 1, s_b);
        if (!mlp->weight[last] || !mlp->bias[last]) return -1;
    }
    return 0;
}

/* ========================================================================
 * Workspace + forward
 * ======================================================================== */

/* We need two ping-pong buffers of [N_points, hidden_dim] plus one
 * [N_points, out_channels]. */
size_t lrm_nerf_mlp_workspace_bytes(const lrm_nerf_mlp *mlp, int N_points) {
    size_t hidden_f = (size_t)N_points * mlp->hidden_dim;
    size_t out_f    = (size_t)N_points * mlp->out_channels;
    return (2 * hidden_f + out_f) * sizeof(float);
}

int lrm_nerf_mlp_forward(const lrm_nerf_mlp *mlp,
                         const float *features,
                         int N_points,
                         float *density,
                         float *color,
                         float *work) {
    if (!mlp || !features || !density || !color || !work) {
        return set_err("nerf_mlp: NULL argument");
    }

    const int H  = mlp->hidden_dim;
    const int O  = mlp->out_channels;

    /* Carve workspace. */
    float *bufA = work;
    float *bufB = work + (size_t)N_points * H;
    float *raw  = work + (size_t)2 * N_points * H;

    /* Layer 0: features [N, 120] -> bufA [N, 64], SiLU. */
    iris_linear(bufA, features, mlp->weight[0], mlp->bias[0],
                N_points, mlp->in_channels, H);
    iris_silu(bufA, N_points * H);

    /* Middle layers: ping-pong bufA <-> bufB. */
    float *src = bufA;
    float *dst = bufB;
    for (int i = 1; i < mlp->num_layers - 1; i++) {
        iris_linear(dst, src, mlp->weight[i], mlp->bias[i],
                    N_points, H, H);
        iris_silu(dst, N_points * H);
        float *tmp = src; src = dst; dst = tmp;
    }

    /* Final layer: src [N, 64] -> raw [N, 4], no activation. */
    iris_linear(raw, src, mlp->weight[mlp->num_layers - 1],
                mlp->bias[mlp->num_layers - 1],
                N_points, H, O);

    /* Split + activations:
     *   density[n] = exp(raw[n, 0] + density_bias)
     *   color  [n, c] = sigmoid(raw[n, 1 + c])  for c in 0..2
     */
    const float dbias = mlp->density_bias;
    for (int n = 0; n < N_points; n++) {
        const float *r = raw + (size_t)n * O;
        density[n] = expf(r[0] + dbias);
        float *cd = color + (size_t)n * 3;
        for (int c = 0; c < 3; c++) {
            float v = r[1 + c];
            cd[c] = 1.0f / (1.0f + expf(-v));
        }
    }
    return 0;
}
