/*
 * test_vit_dino.c - Parity test for lrm_vit_dino_forward vs the golden run.
 *
 * Test gate for Phase 6 (LRMengine.md section 8 row #6). Loads the TripoSR
 * safetensors, runs the C DINO encoder on the preprocessed input image, and
 * compares against the reference dino_tokens dump produced by PyTorch via
 * tools/extract_golden.py.
 *
 * Tolerance: atol=5e-4 in f32 (LRMengine.md section 10.2).
 *
 * Inputs (must be regenerated via tools/extract_golden.py before running):
 *   triposr_env/model.safetensors          - converted TripoSR weights
 *   tests/golden/triposr/input_512.bin     - [1, 1, 512, 512, 3] f32
 *   tests/golden/triposr/dino_tokens.bin   - [1, 1025, 768] f32
 *
 * Build: make test-dino
 * Run:   ./test_vit_dino
 *
 * Exit 0 on parity pass, 1 on parity fail or setup error.
 */

#include "iris.h"
#include "iris_safetensors.h"
#include "lrm/lrm_vit_dino.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kModelPath  = "triposr_env/model.safetensors";
static const char *kInputPath  = "tests/golden/triposr/input_512.bin";
static const char *kGoldenPath = "tests/golden/triposr/dino_tokens.bin";

#define ATOL_DINO 5.0e-4f

/* mmap a file as f32; bytes returned via *out_bytes. */
static const float *mmap_f32(const char *path, size_t *out_bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror(path);
        close(fd);
        return NULL;
    }
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        perror(path);
        return NULL;
    }
    *out_bytes = (size_t)st.st_size;
    return (const float *)p;
}

/* Transpose [1, 1, H, W, C] (channel-last, as dumped by extract_golden.py)
 * into [1, C, H, W] (the layout lrm_vit_dino_forward consumes). */
static void hwc_to_chw(const float *src, float *dst, int H, int W, int C) {
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                dst[(size_t)c * H * W + (size_t)y * W + x] =
                    src[((size_t)y * W + x) * C + c];
            }
        }
    }
}

/* Compare two flat float arrays and report. Returns failure count. */
static int compare(const float *expected, const float *got, size_t n,
                   float atol) {
    int failures = 0;
    double max_err = 0.0;
    double sum_err = 0.0;
    size_t worst_idx = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(expected[i] - got[i]);
        sum_err += diff;
        if (diff > max_err) {
            max_err = diff;
            worst_idx = i;
        }
        if (!(diff <= atol)) {
            if (failures < 8) {
                fprintf(stderr,
                        "  idx=%zu  expected=% .8e  got=% .8e  |err|=%.3e\n",
                        i, (double)expected[i], (double)got[i], (double)diff);
            }
            failures++;
        }
    }
    double mae = sum_err / (double)n;
    fprintf(stderr,
            "  summary: %zu elements, max |err|=%.3e at idx=%zu, mean |err|=%.3e\n",
            n, max_err, worst_idx, mae);
    return failures;
}

int main(void) {
    /* ----- 1. Open safetensors + bind DINO weights. */
    safetensors_file_t *sf = safetensors_open(kModelPath);
    if (!sf) {
        fprintf(stderr, "FAIL: cannot open %s\n", kModelPath);
        return 1;
    }

    lrm_vit_dino vit;
    if (lrm_vit_dino_init(&vit, sf, /*input_size=*/512) != 0) {
        fprintf(stderr, "FAIL: lrm_vit_dino_init: %s\n", iris_get_error());
        safetensors_close(sf);
        return 1;
    }
    fprintf(stderr, "loaded DINO ViT-B/16: %d layers, %d hidden, %d tokens\n",
            vit.num_layers, vit.hidden_dim, vit.num_tokens);

    /* ----- 2. Read the golden input image (HWC f32) and the golden output. */
    size_t in_bytes = 0;
    const float *input_hwc = mmap_f32(kInputPath, &in_bytes);
    if (!input_hwc) goto fail;
    /* extract_golden.py dumps [1, 1, 512, 512, 3] f32 = 1*1*512*512*3*4 bytes. */
    const size_t expected_in_bytes = (size_t)512 * 512 * 3 * sizeof(float);
    if (in_bytes != expected_in_bytes) {
        fprintf(stderr, "FAIL: %s has %zu bytes, expected %zu\n",
                kInputPath, in_bytes, expected_in_bytes);
        goto fail;
    }

    size_t gold_bytes = 0;
    const float *golden_tokens = mmap_f32(kGoldenPath, &gold_bytes);
    if (!golden_tokens) goto fail;
    const size_t expected_gold_bytes =
        (size_t)vit.num_tokens * vit.hidden_dim * sizeof(float);
    if (gold_bytes != expected_gold_bytes) {
        fprintf(stderr, "FAIL: %s has %zu bytes, expected %zu\n",
                kGoldenPath, gold_bytes, expected_gold_bytes);
        goto fail;
    }

    /* ----- 3. HWC -> CHW transpose into a heap buffer. */
    size_t img_floats = (size_t)3 * 512 * 512;
    float *image_chw = (float *)malloc(img_floats * sizeof(float));
    if (!image_chw) { fprintf(stderr, "FAIL: oom\n"); goto fail; }
    hwc_to_chw(input_hwc, image_chw, 512, 512, 3);

    /* ----- 4. Allocate output and workspace. */
    size_t out_floats = (size_t)vit.num_tokens * vit.hidden_dim;
    float *tokens_out = (float *)malloc(out_floats * sizeof(float));
    if (!tokens_out) { fprintf(stderr, "FAIL: oom\n"); free(image_chw); goto fail; }

    size_t ws_bytes = lrm_vit_dino_workspace_bytes(&vit);
    fprintf(stderr, "workspace: %.2f MB\n", (double)ws_bytes / (1024.0 * 1024.0));
    float *workspace = (float *)malloc(ws_bytes);
    if (!workspace) {
        fprintf(stderr, "FAIL: oom for workspace\n");
        free(tokens_out); free(image_chw); goto fail;
    }

    /* ----- 5. Forward. */
    fprintf(stderr, "running lrm_vit_dino_forward ...\n");
    if (lrm_vit_dino_forward(&vit, image_chw, tokens_out, workspace) != 0) {
        fprintf(stderr, "FAIL: forward: %s\n", iris_get_error());
        free(workspace); free(tokens_out); free(image_chw);
        goto fail;
    }

    /* ----- 6. Compare. */
    fprintf(stderr, "comparing %zu floats at atol=%.0e ...\n",
            out_floats, (double)ATOL_DINO);
    int failures = compare(golden_tokens, tokens_out, out_floats, ATOL_DINO);

    free(workspace);
    free(tokens_out);
    free(image_chw);
    lrm_vit_dino_release(&vit);
    safetensors_close(sf);

    if (failures == 0) {
        printf("\nPASS  vit_dino_forward parity (%.0e tolerance, %zu floats)\n",
               (double)ATOL_DINO, out_floats);
        return 0;
    }
    fprintf(stderr, "\nFAIL  vit_dino_forward: %d/%zu elements exceed atol\n",
            failures, out_floats);
    return 1;

fail:
    lrm_vit_dino_release(&vit);
    safetensors_close(sf);
    return 1;
}
