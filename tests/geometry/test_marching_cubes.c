/*
 * test_marching_cubes.c - Structural parity test for Phase 10.
 *
 * Loads the golden density volume (64^3), runs the C marching cubes
 * extraction, and compares against the reference mesh produced by
 * torchmcubes (via tools/extract_golden.py):
 *
 *   vertex count  : within +/- 2% of golden
 *   face   count  : within +/- 2% of golden
 *   surface area  : within +/- 1% of golden
 *   Chamfer dist  : < 1e-3 (mean nearest-neighbour, both directions,
 *                  normalized by the model's bounding-box diagonal)
 *
 * Element-wise comparison isn't expected: different MC implementations
 * legitimately differ in vertex ordering, triangle winding, and may
 * choose different edges to interpolate on ambiguous "saddle" cells.
 * The structural checks above are the standard way to validate.
 *
 * Inputs (regenerate via tools/extract_golden.py):
 *   tests/golden/triposr/density.bin           [64, 64, 64] f32
 *   tests/golden/triposr/mesh_vertices.bin     [N, 3]      f32
 *   tests/golden/triposr/mesh_faces.bin        [F, 3]      i32
 *
 * Build: make test-mc
 * Run:   ./test_marching_cubes
 */

#include "iris.h"
#include "lrm/lrm_marching_cubes.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kDensityPath  = "tests/golden/triposr/density.bin";
static const char *kVertsPath    = "tests/golden/triposr/mesh_vertices.bin";
static const char *kFacesPath    = "tests/golden/triposr/mesh_faces.bin";

#define GRID_RES 64
#define RADIUS   0.87f
#define THRESH   25.0f

#define VFTOL    0.02   /* vertex count tolerance (+/-) */
#define AREA_TOL 0.01   /* surface area tolerance (+/-) */
#define CHAMFER_TOL 1.0e-3  /* mean nearest-neighbour distance */

static const void *mmap_bytes(const char *path, size_t *out_bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror(path); close(fd); return NULL; }
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror(path); return NULL; }
    *out_bytes = (size_t)st.st_size;
    return p;
}

/* Triangle surface area via cross product magnitude. */
static double tri_area(const float *p0, const float *p1, const float *p2) {
    double ax = p1[0] - p0[0], ay = p1[1] - p0[1], az = p1[2] - p0[2];
    double bx = p2[0] - p0[0], by = p2[1] - p0[1], bz = p2[2] - p0[2];
    double cx = ay * bz - az * by;
    double cy = az * bx - ax * bz;
    double cz = ax * by - ay * bx;
    return 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
}

static double total_surface_area(const float *verts, int n_verts,
                                 const int32_t *faces, int n_faces) {
    (void)n_verts;
    double total = 0.0;
    for (int f = 0; f < n_faces; f++) {
        const float *p0 = verts + (size_t)faces[f * 3 + 0] * 3;
        const float *p1 = verts + (size_t)faces[f * 3 + 1] * 3;
        const float *p2 = verts + (size_t)faces[f * 3 + 2] * 3;
        total += tri_area(p0, p1, p2);
    }
    return total;
}

/* Mean nearest-neighbour distance from each point in A to its closest
 * point in B (one-sided Chamfer). O(|A| * |B|). */
static double oneway_chamfer(const float *A, int Na,
                             const float *B, int Nb) {
    double total = 0.0;
    for (int i = 0; i < Na; i++) {
        float ax = A[i*3+0], ay = A[i*3+1], az = A[i*3+2];
        float best = 1e30f;
        for (int j = 0; j < Nb; j++) {
            float dx = ax - B[j*3+0];
            float dy = ay - B[j*3+1];
            float dz = az - B[j*3+2];
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best) best = d2;
        }
        total += sqrt((double)best);
    }
    return total / (double)Na;
}

static double bbox_diagonal(const float *v, int n) {
    if (n == 0) return 1.0;
    float mn[3] = { v[0], v[1], v[2] };
    float mx[3] = { v[0], v[1], v[2] };
    for (int i = 1; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            float c = v[i*3+k];
            if (c < mn[k]) mn[k] = c;
            if (c > mx[k]) mx[k] = c;
        }
    }
    double dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/* Synthetic test: a big solid block plus a tiny isolated blob in a corner.
 * lrm_mc_remove_small_components must drop the blob and keep the block.
 * Returns the number of failed checks. */
static int test_floater_removal(void) {
    const int R = 16;
    float *vol = (float *)calloc((size_t)R * R * R, sizeof(float));
    if (!vol) { fprintf(stderr, "FAIL: oom synthetic volume\n"); return 1; }
#define V(i,j,k) vol[(((size_t)(i) * R + (j)) * R + (k))]
    for (int i = 4; i <= 11; i++)        /* big central block */
        for (int j = 4; j <= 11; j++)
            for (int k = 4; k <= 11; k++) V(i,j,k) = 100.0f;
    for (int i = 1; i <= 2; i++)         /* tiny corner blob */
        for (int j = 1; j <= 2; j++)
            for (int k = 1; k <= 2; k++) V(i,j,k) = 100.0f;
#undef V

    lrm_mc_mesh m = {0};
    if (lrm_marching_cubes_extract(vol, R, THRESH, -RADIUS, +RADIUS, &m) != 0) {
        fprintf(stderr, "FAIL: synthetic extract: %s\n", iris_get_error());
        free(vol); return 1;
    }
    int nf_before = m.n_faces;

    /* fraction<=0 must be a no-op. */
    int fails = 0;
    lrm_mc_mesh probe = {0};
    lrm_marching_cubes_extract(vol, R, THRESH, -RADIUS, +RADIUS, &probe);
    if (lrm_mc_remove_small_components(&probe, 0.0f) != 0 ||
        probe.n_faces != nf_before) {
        fprintf(stderr, "  FAIL floater no-op: %d -> %d (expected unchanged)\n",
                nf_before, probe.n_faces);
        fails++;
    }
    lrm_mc_mesh_free(&probe);

    if (lrm_mc_remove_small_components(&m, 0.1f) != 0) {
        fprintf(stderr, "  FAIL floater removal: %s\n", iris_get_error());
        free(vol); lrm_mc_mesh_free(&m); return fails + 1;
    }
    /* The corner blob lives near world x ~ -0.75; the block spans ~[-0.4,0.4].
     * After removal no surviving vertex should sit in the far-negative
     * corner. */
    float minc = 1e30f;
    for (int v = 0; v < m.n_vertices; v++)
        for (int c = 0; c < 3; c++)
            if (m.vertices[v*3+c] < minc) minc = m.vertices[v*3+c];
    if (m.n_faces >= nf_before || m.n_faces == 0) {
        fprintf(stderr, "  FAIL floater face count: %d -> %d\n",
                nf_before, m.n_faces);
        fails++;
    }
    if (minc < -0.6f) {
        fprintf(stderr, "  FAIL floater not removed: min coord %.3f\n", minc);
        fails++;
    }
    if (fails == 0) {
        fprintf(stderr, "  PASS floater removal: faces %d -> %d, min coord %.3f\n",
                nf_before, m.n_faces, minc);
    }
    free(vol);
    lrm_mc_mesh_free(&m);
    return fails;
}

int main(void) {
    /* ----- Load density volume. */
    size_t db;
    const float *density = (const float *)mmap_bytes(kDensityPath, &db);
    if (!density) return 1;
    const size_t N = (size_t)GRID_RES * GRID_RES * GRID_RES;
    if (db != N * sizeof(float)) {
        fprintf(stderr, "FAIL: density.bin %zu bytes, expected %zu\n",
                db, N * sizeof(float));
        return 1;
    }

    /* ----- Load golden mesh. */
    size_t vb;
    const float *gold_verts = (const float *)mmap_bytes(kVertsPath, &vb);
    if (!gold_verts) return 1;
    int gold_nverts = (int)(vb / (sizeof(float) * 3));
    size_t fb;
    const int32_t *gold_faces = (const int32_t *)mmap_bytes(kFacesPath, &fb);
    if (!gold_faces) return 1;
    int gold_nfaces = (int)(fb / (sizeof(int32_t) * 3));

    fprintf(stderr,
            "golden mesh: %d vertices, %d faces\n",
            gold_nverts, gold_nfaces);

    /* ----- Run C marching cubes. */
    fprintf(stderr, "running lrm_marching_cubes_extract ...\n");
    lrm_mc_mesh mesh = {0};
    if (lrm_marching_cubes_extract(density, GRID_RES, THRESH,
                                   -RADIUS, +RADIUS, &mesh) != 0) {
        fprintf(stderr, "FAIL: extract: %s\n", iris_get_error());
        return 1;
    }
    fprintf(stderr, "C mesh:     %d vertices, %d faces\n",
            mesh.n_vertices, mesh.n_faces);

    /* ----- Count comparisons. */
    int failures = 0;
    double v_ratio = (double)mesh.n_vertices / (double)gold_nverts;
    double f_ratio = (double)mesh.n_faces    / (double)gold_nfaces;
    if (v_ratio < 1.0 - VFTOL || v_ratio > 1.0 + VFTOL) {
        fprintf(stderr,
                "  FAIL vertex count: %d vs %d (ratio %.4f, tol +/- %.1f%%)\n",
                mesh.n_vertices, gold_nverts, v_ratio, 100.0 * VFTOL);
        failures++;
    } else {
        fprintf(stderr,
                "  PASS vertex count: %d vs %d (ratio %.4f)\n",
                mesh.n_vertices, gold_nverts, v_ratio);
    }
    if (f_ratio < 1.0 - VFTOL || f_ratio > 1.0 + VFTOL) {
        fprintf(stderr,
                "  FAIL face count: %d vs %d (ratio %.4f, tol +/- %.1f%%)\n",
                mesh.n_faces, gold_nfaces, f_ratio, 100.0 * VFTOL);
        failures++;
    } else {
        fprintf(stderr,
                "  PASS face count: %d vs %d (ratio %.4f)\n",
                mesh.n_faces, gold_nfaces, f_ratio);
    }

    /* ----- Surface area. */
    double area_gold = total_surface_area(gold_verts, gold_nverts,
                                          gold_faces, gold_nfaces);
    double area_c    = total_surface_area(mesh.vertices, mesh.n_vertices,
                                          mesh.faces, mesh.n_faces);
    double area_ratio = area_c / area_gold;
    if (area_ratio < 1.0 - AREA_TOL || area_ratio > 1.0 + AREA_TOL) {
        fprintf(stderr,
                "  FAIL surface area: %.6f vs %.6f (ratio %.4f, tol +/- %.1f%%)\n",
                area_c, area_gold, area_ratio, 100.0 * AREA_TOL);
        failures++;
    } else {
        fprintf(stderr,
                "  PASS surface area: %.6f vs %.6f (ratio %.4f)\n",
                area_c, area_gold, area_ratio);
    }

    /* ----- Symmetric Chamfer distance, normalized by bbox diagonal. */
    double diag = bbox_diagonal(gold_verts, gold_nverts);
    fprintf(stderr, "  bbox diagonal (gold): %.6f\n", diag);
    double ab = oneway_chamfer(gold_verts, gold_nverts,
                               mesh.vertices, mesh.n_vertices);
    double ba = oneway_chamfer(mesh.vertices, mesh.n_vertices,
                               gold_verts, gold_nverts);
    double chamfer = (ab * gold_nverts + ba * mesh.n_vertices)
                   / (double)(gold_nverts + mesh.n_vertices);
    double chamfer_norm = chamfer / diag;
    if (chamfer_norm > CHAMFER_TOL) {
        fprintf(stderr,
                "  FAIL Chamfer: raw %.6e, normalized %.6e (tol %.0e)\n",
                chamfer, chamfer_norm, CHAMFER_TOL);
        failures++;
    } else {
        fprintf(stderr,
                "  PASS Chamfer: raw %.6e, normalized %.6e\n",
                chamfer, chamfer_norm);
    }

    lrm_mc_mesh_free(&mesh);

    /* ----- Floater-removal synthetic test. */
    fprintf(stderr, "floater removal (synthetic two-component volume):\n");
    failures += test_floater_removal();

    if (failures == 0) {
        printf("\nPASS  marching_cubes structural parity + floater removal\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL  marching_cubes (%d checks failed)\n",
            failures);
    return 1;
}
