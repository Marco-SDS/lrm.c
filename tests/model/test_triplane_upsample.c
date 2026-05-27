/*
 * test_triplane_upsample.c - Parity test for lrm_triplane_upsample_forward.
 *
 * Loads triplane_pre_upsample.bin (the golden pre-post_processor scene code,
 * [1, 3, 1024, 32, 32]) as input and triplane.bin ([1, 3, 40, 64, 64]) as
 * the expected post_processor output, runs the C upsample, and compares.
 *
 * Tolerance: atol=5e-4, rtol=1e-4. The op is a single GEMM + scatter, so
 * the f32 floor here is the same as the kernel-test parity (~1e-5). The
 * 5e-4 atol just leaves a small margin for libm/Accelerate variation.
 */

#include "iris.h"
#include "iris_safetensors.h"
#include "lrm/lrm_triplane_upsample.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kModelPath  = "triposr_env/model.safetensors";
static const char *kInPath     = "tests/golden/triposr/triplane_pre_upsample.bin";
static const char *kGoldenPath = "tests/golden/triposr/triplane.bin";

#define ATOL_UPSAMPLE 5.0e-4f
#define RTOL_UPSAMPLE 1.0e-4f

static const float *mmap_f32(const char *path, size_t *out_bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror(path); close(fd); return NULL; }
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror(path); return NULL; }
    *out_bytes = (size_t)st.st_size;
    return (const float *)p;
}

static int compare(const float *expected, const float *got, size_t n,
                   float atol, float rtol) {
    int failures = 0;
    double max_err = 0.0;
    double sum_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        float e = expected[i];
        float diff = fabsf(e - got[i]);
        sum_err += diff;
        if (diff > max_err) { max_err = diff; worst = i; }
        float tol = atol + rtol * fabsf(e);
        if (!(diff <= tol)) {
            if (failures < 8) {
                fprintf(stderr,
                        "  idx=%zu expected=% .8e got=% .8e |err|=%.3e tol=%.3e\n",
                        i, (double)e, (double)got[i],
                        (double)diff, (double)tol);
            }
            failures++;
        }
    }
    fprintf(stderr,
            "  summary: %zu elements, max |err|=%.3e at idx=%zu "
            "(expected=% .3e), mean |err|=%.3e\n",
            n, max_err, worst, (double)expected[worst],
            sum_err / (double)n);
    return failures;
}

int main(void) {
    safetensors_file_t *sf = safetensors_open(kModelPath);
    if (!sf) {
        fprintf(stderr, "FAIL: cannot open %s\n", kModelPath);
        return 1;
    }
    lrm_triplane_upsample up;
    if (lrm_triplane_upsample_init(&up, sf) != 0) {
        fprintf(stderr, "FAIL: init: %s\n", iris_get_error());
        safetensors_close(sf);
        return 1;
    }
    fprintf(stderr,
            "loaded triplane upsample: in_ch=%d -> out_ch=%d, kernel=%dx%d "
            "stride=%d, planes=%d, %d x %d -> %d x %d\n",
            up.in_ch, up.out_ch, up.kernel, up.kernel, up.stride,
            up.planes, up.in_size, up.in_size, up.out_size, up.out_size);

    size_t in_bytes = 0;
    const float *in = mmap_f32(kInPath, &in_bytes);
    if (!in) goto fail;
    const size_t expected_in = (size_t)up.planes * up.in_ch
                             * up.in_size * up.in_size * sizeof(float);
    if (in_bytes != expected_in) {
        fprintf(stderr, "FAIL: %s has %zu bytes, expected %zu\n",
                kInPath, in_bytes, expected_in);
        goto fail;
    }

    size_t gold_bytes = 0;
    const float *gold = mmap_f32(kGoldenPath, &gold_bytes);
    if (!gold) goto fail;
    const size_t out_floats = (size_t)up.planes * up.out_ch
                            * up.out_size * up.out_size;
    const size_t expected_out = out_floats * sizeof(float);
    if (gold_bytes != expected_out) {
        fprintf(stderr, "FAIL: %s has %zu bytes, expected %zu\n",
                kGoldenPath, gold_bytes, expected_out);
        goto fail;
    }

    float *out = (float *)malloc(out_floats * sizeof(float));
    if (!out) { fprintf(stderr, "FAIL: oom out\n"); goto fail; }
    size_t ws = lrm_triplane_upsample_workspace_bytes(&up);
    fprintf(stderr, "workspace: %.2f KB\n", (double)ws / 1024.0);
    float *work = (float *)malloc(ws);
    if (!work) {
        fprintf(stderr, "FAIL: oom work\n");
        free(out); goto fail;
    }

    fprintf(stderr, "running lrm_triplane_upsample_forward ...\n");
    if (lrm_triplane_upsample_forward(&up, in, out, work) != 0) {
        fprintf(stderr, "FAIL: forward: %s\n", iris_get_error());
        free(work); free(out); goto fail;
    }

    fprintf(stderr, "comparing %zu floats at atol=%.0e rtol=%.0e ...\n",
            out_floats, (double)ATOL_UPSAMPLE, (double)RTOL_UPSAMPLE);
    int failures = compare(gold, out, out_floats, ATOL_UPSAMPLE, RTOL_UPSAMPLE);

    free(work);
    free(out);
    lrm_triplane_upsample_release(&up);
    safetensors_close(sf);

    if (failures == 0) {
        printf("\nPASS  triplane_upsample_forward parity "
               "(atol=%.0e rtol=%.0e, %zu floats)\n",
               (double)ATOL_UPSAMPLE, (double)RTOL_UPSAMPLE, out_floats);
        return 0;
    }
    fprintf(stderr,
            "\nFAIL  triplane_upsample_forward: %d/%zu exceed tol\n",
            failures, out_floats);
    return 1;

fail:
    lrm_triplane_upsample_release(&up);
    safetensors_close(sf);
    return 1;
}
