/*
 * test_u2net.c - U2Net forward parity vs the PyTorch reference (Phase 16).
 *
 * Feeds the golden preprocessed input [3,320,320] through the C U2Net and
 * compares sigmoid(d0) against the golden mask [320,320] produced by
 * tools/u2net_extract_golden.py.
 *
 * Inputs (regenerate via tools/u2net_extract_golden.py):
 *   triposr_env/u2net.safetensors
 *   tests/golden/triposr/u2net_input.bin   [3,320,320] f32
 *   tests/golden/triposr/u2net_mask.bin    [320,320]   f32
 *
 * Build: make test-u2net
 */

#include "iris.h"
#include "lrm/lrm_u2net.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kModel = "triposr_env/u2net.safetensors";
static const char *kInput = "tests/golden/triposr/u2net_input.bin";
static const char *kMask  = "tests/golden/triposr/u2net_mask.bin";

#define SZ 320
#define ATOL 2.0e-3f   /* sigmoid output of a deep BN+conv net, f32 */

static const float *mmap_f32(const char *path, size_t *n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror(path); close(fd); return NULL; }
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror(path); return NULL; }
    *n = st.st_size / sizeof(float);
    return (const float *)p;
}

int main(void) {
    lrm_u2net *m = lrm_u2net_load(kModel);
    if (!m) { fprintf(stderr, "FAIL: load %s: %s\n", kModel, iris_get_error()); return 1; }

    size_t ni, nm;
    const float *input = mmap_f32(kInput, &ni);
    const float *gold  = mmap_f32(kMask, &nm);
    if (!input || !gold) { lrm_u2net_free(m); return 1; }
    if (ni != (size_t)3 * SZ * SZ || nm != (size_t)SZ * SZ) {
        fprintf(stderr, "FAIL: bad golden sizes (input=%zu mask=%zu)\n", ni, nm);
        lrm_u2net_free(m); return 1;
    }

    float *got = (float *)malloc((size_t)SZ * SZ * sizeof(float));
    if (!got) { lrm_u2net_free(m); return 1; }

    fprintf(stderr, "running U2Net forward (320x320) ...\n");
    if (lrm_u2net_forward(m, input, SZ, SZ, got) != 0) {
        fprintf(stderr, "FAIL: forward: %s\n", iris_get_error());
        free(got); lrm_u2net_free(m); return 1;
    }

    double max_abs = 0.0, sum_abs = 0.0;
    size_t worst = 0, fails = 0;
    for (size_t i = 0; i < (size_t)SZ * SZ; i++) {
        double e = fabs((double)got[i] - (double)gold[i]);
        sum_abs += e;
        if (e > max_abs) { max_abs = e; worst = i; }
        if (e > ATOL) fails++;
    }
    fprintf(stderr,
            "  max|err|=%.3e at %zu (got=%.5f gold=%.5f)  mean|err|=%.3e  atol=%.0e\n",
            max_abs, worst, got[worst], gold[worst],
            sum_abs / ((double)SZ * SZ), (double)ATOL);

    /* IoU at 0.5 threshold as a sanity check on the mask shape. */
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < (size_t)SZ * SZ; i++) {
        int a = got[i] > 0.5f, b = gold[i] > 0.5f;
        inter += (a && b);
        uni += (a || b);
    }
    fprintf(stderr, "  IoU@0.5 = %.4f\n", uni ? (double)inter / (double)uni : 1.0);

    free(got);
    lrm_u2net_free(m);

    if (fails == 0) {
        printf("\nPASS  u2net forward parity (%d^2 mask, atol=%.0e)\n", SZ, (double)ATOL);
        return 0;
    }
    fprintf(stderr, "\nFAIL  u2net parity: %zu/%d px over atol\n", fails, SZ * SZ);
    return 1;
}
