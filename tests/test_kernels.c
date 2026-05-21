/*
 * test_kernels.c - Parity tests for the kernels introduced in Phase 4.
 *
 * Compares each of iris_layer_norm, iris_gelu, iris_geglu,
 * iris_grid_sample_bilinear against reference outputs computed in PyTorch
 * (see the comment header on each test for the exact reference snippet).
 * Tolerance is atol=1e-5 in f32 (LRMengine.md section 10.2).
 *
 * Build:  make test (added in Phase 4)
 * Run:    ./test_kernels
 *
 * Exit 0 on all-pass, 1 on any failure. Each failed test prints index,
 * expected vs got, and absolute error so the offender is easy to spot.
 */

#include "iris_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Reference vectors (generated via the Python snippet in each section).
 * ======================================================================== */

/* ---- LayerNorm ----
 * Python:
 *   x = torch.tensor([[1,2,3,4],[-1,0,1,2]], dtype=torch.float32)
 *   gamma = torch.tensor([0.5, 1.0, 1.5, 2.0])
 *   beta  = torch.tensor([0.1, -0.1, 0.2, -0.2])
 *   F.layer_norm(x, [4], weight=gamma, bias=beta, eps=1e-5)
 */
static const float ref_ln_x[8] = {
    +1.00000000e+00f, +2.00000000e+00f, +3.00000000e+00f, +4.00000000e+00f,
    -1.00000000e+00f, +0.00000000e+00f, +1.00000000e+00f, +2.00000000e+00f
};
static const float ref_ln_gamma[4] = {
    +5.00000000e-01f, +1.00000000e+00f, +1.50000000e+00f, +2.00000000e+00f
};
static const float ref_ln_beta[4] = {
    +1.00000001e-01f, -1.00000001e-01f, +2.00000003e-01f, -2.00000003e-01f
};
static const float ref_ln_out[8] = {
    -5.70817709e-01f, -5.47211826e-01f, +8.70817721e-01f, +2.48327088e+00f,
    -5.70817709e-01f, -5.47211826e-01f, +8.70817721e-01f, +2.48327088e+00f
};

/* ---- GELU exact ----
 * Python:
 *   x = torch.linspace(-3, 3, 8)
 *   F.gelu(x, approximate='none')
 */
static const float ref_gelu_in[8] = {
    -3.00000000e+00f, -2.14285707e+00f, -1.28571427e+00f, -4.28571463e-01f,
    +4.28571463e-01f, +1.28571427e+00f, +2.14285707e+00f, +3.00000000e+00f
};
static const float ref_gelu_out[8] = {
    -4.04986739e-03f, -3.44190635e-02f, -1.27634749e-01f, -1.43193260e-01f,
    +2.85378188e-01f, +1.15807950e+00f, +2.10843801e+00f, +2.99595022e+00f
};

/* ---- GEGLU ----
 * Python:
 *   hidden = torch.tensor([1, 2, -1, 0.5, 3, -2])
 *   gate   = torch.tensor([0.5, -0.5, 2, -2, 0, 1])
 *   hidden * F.gelu(gate, approximate='none')
 */
static const float ref_geglu_hidden[6] = {
    +1.00000000e+00f, +2.00000000e+00f, -1.00000000e+00f,
    +5.00000000e-01f, +3.00000000e+00f, -2.00000000e+00f
};
static const float ref_geglu_gate[6] = {
    +5.00000000e-01f, -5.00000000e-01f, +2.00000000e+00f,
    -2.00000000e+00f, +0.00000000e+00f, +1.00000000e+00f
};
static const float ref_geglu_out[6] = {
    +3.45731199e-01f, -3.08537573e-01f, -1.95449996e+00f,
    -2.27500498e-02f, +0.00000000e+00f, -1.68268943e+00f
};

/* ---- grid_sample bilinear ----
 * Python:
 *   input  = arange(32, dtype=float32).reshape(2, 1, 4, 4)
 *   grid   = the (x, y) coords below, shape (2, 5, 2)
 *   F.grid_sample(input, grid[:, None, ...],
 *                 mode='bilinear', padding_mode='border', align_corners=False)
 *
 * Plane 0 input is values 0..15 laid row-major; plane 1 is 16..31.
 * The 5 queries per plane cover: center, the two padding corners, an
 * out-of-range point, and an off-grid interior point.
 */
static const float ref_gs_input[32] = {
    +0.00000000e+00f, +1.00000000e+00f, +2.00000000e+00f, +3.00000000e+00f,
    +4.00000000e+00f, +5.00000000e+00f, +6.00000000e+00f, +7.00000000e+00f,
    +8.00000000e+00f, +9.00000000e+00f, +1.00000000e+01f, +1.10000000e+01f,
    +1.20000000e+01f, +1.30000000e+01f, +1.40000000e+01f, +1.50000000e+01f,
    +1.60000000e+01f, +1.70000000e+01f, +1.80000000e+01f, +1.90000000e+01f,
    +2.00000000e+01f, +2.10000000e+01f, +2.20000000e+01f, +2.30000000e+01f,
    +2.40000000e+01f, +2.50000000e+01f, +2.60000000e+01f, +2.70000000e+01f,
    +2.80000000e+01f, +2.90000000e+01f, +3.00000000e+01f, +3.10000000e+01f
};
static const float ref_gs_grid[20] = {
    +0.00000000e+00f, +0.00000000e+00f, -1.00000000e+00f, -1.00000000e+00f,
    +1.00000000e+00f, +1.00000000e+00f, -2.00000000e+00f, +5.00000000e-01f,
    +3.00000012e-01f, -6.99999988e-01f, +2.50000000e-01f, +2.50000000e-01f,
    -5.00000000e-01f, -5.00000000e-01f, +5.00000000e-01f, +5.00000000e-01f,
    -1.00000000e+00f, +1.00000000e+00f, +0.00000000e+00f, -3.00000012e-01f
};
static const float ref_gs_out[10] = {
    +7.50000000e+00f,  +0.00000000e+00f, +1.50000000e+01f, +1.00000000e+01f,
    +2.50000000e+00f,  +2.60000000e+01f, +1.85000000e+01f, +2.85000000e+01f,
    +2.80000000e+01f,  +2.11000004e+01f
};

/* ========================================================================
 * Comparison helper
 * ======================================================================== */

#define ATOL_F32 1.0e-5f

static int compare_array(const char *test, const float *expected,
                         const float *got, int n) {
    int failures = 0;
    for (int i = 0; i < n; i++) {
        float e = expected[i];
        float g = got[i];
        float diff = fabsf(e - g);
        if (!(diff <= ATOL_F32)) {  /* NaN-safe: fails if diff is NaN */
            if (failures < 8) {
                fprintf(stderr,
                        "  [%s] idx=%d expected=% .8e got=% .8e |err|=%.3e\n",
                        test, i, (double)e, (double)g, (double)diff);
            }
            failures++;
        }
    }
    if (failures == 0) {
        printf("PASS  %s  (%d elements, atol=%g)\n", test, n, (double)ATOL_F32);
        return 0;
    }
    fprintf(stderr, "FAIL  %s  (%d/%d elements exceed atol=%g)\n",
            test, failures, n, (double)ATOL_F32);
    return 1;
}

/* ========================================================================
 * Tests
 * ======================================================================== */

static int test_layer_norm(void) {
    float out[8];
    iris_layer_norm(out, ref_ln_x, ref_ln_gamma, ref_ln_beta,
                    /*seq_len=*/2, /*hidden=*/4, /*eps=*/1.0e-5f);
    return compare_array("layer_norm[2x4]", ref_ln_out, out, 8);
}

static int test_layer_norm_no_affine(void) {
    /* Sanity-check the gamma=NULL/beta=NULL branches separately by feeding
     * gamma=ones, beta=zeros and confirming the same row stats. */
    float gamma_ones[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float zeros[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float ref[8], got[8];
    iris_layer_norm(ref, ref_ln_x, gamma_ones, zeros, 2, 4, 1.0e-5f);
    iris_layer_norm(got, ref_ln_x, NULL, NULL, 2, 4, 1.0e-5f);
    return compare_array("layer_norm[no-affine]", ref, got, 8);
}

static int test_gelu(void) {
    float out[8];
    memcpy(out, ref_gelu_in, sizeof(out));
    iris_gelu(out, 8);
    return compare_array("gelu[exact]", ref_gelu_out, out, 8);
}

static int test_geglu(void) {
    float out[6];
    iris_geglu(out, ref_geglu_hidden, ref_geglu_gate, 6);
    return compare_array("geglu", ref_geglu_out, out, 6);
}

static int test_grid_sample(void) {
    float out[10];
    iris_grid_sample_bilinear(out, ref_gs_input, ref_gs_grid,
                              /*N_planes=*/2, /*C=*/1,
                              /*H=*/4, /*W=*/4, /*N_points=*/5);
    return compare_array("grid_sample[2x1x4x4 -> 5 pts]", ref_gs_out, out, 10);
}

int main(void) {
    int failures = 0;
    failures += test_layer_norm();
    failures += test_layer_norm_no_affine();
    failures += test_gelu();
    failures += test_geglu();
    failures += test_grid_sample();

    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    }
    fprintf(stderr, "\n%d test(s) FAILED\n", failures);
    return 1;
}
