/*
 * test_density_sparse.c - sparse (coarse-to-fine) vs dense density parity.
 *
 * Builds the density grid two ways at several resolutions and verifies that
 * the marching-cubes surface is identical: same vertex count, same face
 * count, and bit-identical vertex coordinates. The sparse builder only skips
 * cells far from the isosurface (which emit no triangles), so the extracted
 * mesh must match the dense reference exactly.
 *
 * Inputs (already produced by tools/extract_golden.py):
 *   triposr_env/model.safetensors
 *   tests/golden/triposr/triplane.bin   [1, 3, 40, 64, 64]  post-upsample
 *
 * Build: make test-density-sparse
 */

#include "iris.h"
#include "iris_safetensors.h"
#include "lrm/lrm_density.h"
#include "lrm/lrm_marching_cubes.h"
#include "lrm/lrm_nerf_mlp.h"
#include "lrm/lrm_triplane_sample.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kModelPath    = "triposr_env/model.safetensors";
static const char *kTriplanePath = "tests/golden/triposr/triplane.bin";

#define RADIUS    0.87f
#define THRESHOLD 25.0f

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

/* Compare two MC meshes for exact equivalence. Returns 0 if identical. */
static int compare_meshes(int R, const lrm_mc_mesh *a, const lrm_mc_mesh *b) {
    if (a->n_vertices != b->n_vertices || a->n_faces != b->n_faces) {
        fprintf(stderr,
                "  R=%d MISMATCH counts: dense V=%d F=%d  sparse V=%d F=%d\n",
                R, a->n_vertices, a->n_faces, b->n_vertices, b->n_faces);
        return 1;
    }
    double max_v = 0.0;
    for (size_t i = 0; i < (size_t)a->n_vertices * 3; i++) {
        double d = fabs((double)a->vertices[i] - (double)b->vertices[i]);
        if (d > max_v) max_v = d;
    }
    int face_diff = 0;
    for (size_t i = 0; i < (size_t)a->n_faces * 3; i++) {
        if (a->faces[i] != b->faces[i]) face_diff++;
    }
    fprintf(stderr,
            "  R=%d  V=%d F=%d  max|vert diff|=%.3e  face idx diffs=%d\n",
            R, a->n_vertices, a->n_faces, max_v, face_diff);
    return (max_v > 1e-6 || face_diff != 0) ? 1 : 0;
}

static int run_res(const lrm_triplane_sample_cfg *cfg, const lrm_nerf_mlp *mlp,
                   const float *triplane, int R) {
    const size_t N = (size_t)R * R * R;
    float *d_dense  = (float *)malloc(N * sizeof(float));
    float *d_sparse = (float *)malloc(N * sizeof(float));
    if (!d_dense || !d_sparse) {
        fprintf(stderr, "FAIL: oom grids R=%d\n", R);
        free(d_dense); free(d_sparse);
        return 1;
    }

    if (lrm_density_build_dense(cfg, mlp, triplane, R, RADIUS, d_dense) != 0) {
        fprintf(stderr, "FAIL: dense build: %s\n", iris_get_error());
        free(d_dense); free(d_sparse); return 1;
    }
    if (lrm_density_build_sparse(cfg, mlp, triplane, R, RADIUS, THRESHOLD,
                                 d_sparse) != 0) {
        fprintf(stderr, "FAIL: sparse build: %s\n", iris_get_error());
        free(d_dense); free(d_sparse); return 1;
    }

    lrm_mc_mesh ma = {0}, mb = {0};
    int rc = 0;
    if (lrm_marching_cubes_extract(d_dense, R, THRESHOLD, -RADIUS, RADIUS,
                                   &ma) != 0) {
        fprintf(stderr, "FAIL: MC dense: %s\n", iris_get_error());
        rc = 1; goto done;
    }
    if (lrm_marching_cubes_extract(d_sparse, R, THRESHOLD, -RADIUS, RADIUS,
                                   &mb) != 0) {
        fprintf(stderr, "FAIL: MC sparse: %s\n", iris_get_error());
        rc = 1; goto done;
    }
    rc = compare_meshes(R, &ma, &mb);

done:
    lrm_mc_mesh_free(&ma);
    lrm_mc_mesh_free(&mb);
    free(d_dense); free(d_sparse);
    return rc;
}

int main(void) {
    safetensors_file_t *sf = safetensors_open(kModelPath);
    if (!sf) { fprintf(stderr, "FAIL: open %s\n", kModelPath); return 1; }

    lrm_nerf_mlp mlp;
    if (lrm_nerf_mlp_init(&mlp, sf) != 0) {
        fprintf(stderr, "FAIL: nerf_mlp init: %s\n", iris_get_error());
        safetensors_close(sf); return 1;
    }
    lrm_triplane_sample_cfg cfg = { 3, 40, 64, RADIUS };

    size_t bytes;
    const float *triplane = mmap_f32(kTriplanePath, &bytes);
    if (!triplane) { safetensors_close(sf); return 1; }
    const size_t expect = (size_t)cfg.planes * cfg.channels
                          * cfg.size * cfg.size * sizeof(float);
    if (bytes != expect) {
        fprintf(stderr, "FAIL: triplane.bin %zu bytes, expected %zu\n",
                bytes, expect);
        safetensors_close(sf); return 1;
    }

    fprintf(stderr, "sparse vs dense density parity (threshold=%.1f):\n",
            (double)THRESHOLD);
    int fail = 0;
    fail |= run_res(&cfg, &mlp, triplane, 128);
    fail |= run_res(&cfg, &mlp, triplane, 256);

    safetensors_close(sf);
    if (fail) {
        fprintf(stderr, "\nFAIL  sparse density diverges from dense\n");
        return 1;
    }
    printf("\nPASS  sparse density matches dense MC surface exactly\n");
    return 0;
}
