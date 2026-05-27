/*
 * lrm_density.c - dense and coarse-to-fine density-grid builders.
 *
 * See lrm_density.h for the contract. The sparse builder exploits the fact
 * that the TripoSR density field is band-limited at the triplane resolution:
 * the marching-cubes isosurface is a thin 2D shell, so the vast majority of
 * the R^3 grid sits far from it and never produces triangles. We:
 *
 *   1. Evaluate a coarse lattice (stride B, chosen so the coarse grid is
 *      ~64 nodes per axis = the triplane's native resolution).
 *   2. Mark a coarse block "active" if its 8 corner densities straddle the
 *      iso-threshold or come near it, then dilate the active set by one
 *      block (26-neighborhood) so the surface can never sit in an
 *      unrefined block.
 *   3. Exactly evaluate (sample + MLP) every grid node touched by an active
 *      block; trilinearly interpolate the coarse lattice everywhere else.
 *
 * Because every cell that can contain the surface lies entirely inside an
 * exactly-evaluated active block, and inactive cells have all 8 corners on
 * the same side of the threshold (sign-consistent trilinear fill), the MC
 * output is identical to the dense path. The sparse/dense parity test
 * (tests/test_density_sparse.c) verifies this on the real model.
 */

#include "lrm_density.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"

/* Coarse lattice target resolution (nodes per axis). The triplane is
 * 64x64 per plane, so a coarse grid at this resolution captures every
 * feature the density field can represent. */
#define LRM_DENSITY_COARSE_RES 64

/* ------------------------------------------------------------------ */
/* Reusable per-chunk scratch for sample + MLP evaluation. */
typedef struct {
    int    chunk;
    float *positions;     /* [chunk*3]   */
    float *features;      /* [chunk*120] */
    float *density;       /* [chunk]     */
    float *color;         /* [chunk*3] (discarded) */
    float *ts_work;
    float *nm_work;
} dens_scratch;

static int scratch_alloc(dens_scratch *s, int chunk,
                         const lrm_triplane_sample_cfg *sample_cfg,
                         const lrm_nerf_mlp *mlp) {
    memset(s, 0, sizeof(*s));
    s->chunk = chunk;
    size_t ts_ws = lrm_triplane_sample_workspace_bytes(sample_cfg, chunk);
    size_t nm_ws = lrm_nerf_mlp_workspace_bytes(mlp, chunk);
    s->positions = (float *)malloc((size_t)chunk * 3 * sizeof(float));
    s->features  = (float *)malloc((size_t)chunk * 120 * sizeof(float));
    s->density   = (float *)malloc((size_t)chunk * sizeof(float));
    s->color     = (float *)malloc((size_t)chunk * 3 * sizeof(float));
    s->ts_work   = (float *)malloc(ts_ws);
    s->nm_work   = (float *)malloc(nm_ws);
    if (!s->positions || !s->features || !s->density || !s->color
        || !s->ts_work || !s->nm_work) {
        return -1;
    }
    return 0;
}

static void scratch_free(dens_scratch *s) {
    free(s->positions); free(s->features); free(s->density);
    free(s->color); free(s->ts_work); free(s->nm_work);
    memset(s, 0, sizeof(*s));
}

/* Evaluate density at `n` positions (already packed into s->positions,
 * n <= s->chunk) and write to out[0..n). Color is computed and discarded. */
static int eval_chunk(const lrm_triplane_sample_cfg *sample_cfg,
                      const lrm_nerf_mlp *mlp, const float *triplane,
                      dens_scratch *s, int n, float *out) {
    if (lrm_triplane_sample_forward(sample_cfg, triplane, s->positions,
                                    n, s->features, s->ts_work) != 0) {
        return -1;
    }
    if (lrm_nerf_mlp_forward(mlp, s->features, n, out, s->color,
                             s->nm_work) != 0) {
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Dense reference path
 * ======================================================================== */

int lrm_density_build_dense(const lrm_triplane_sample_cfg *sample_cfg,
                            const lrm_nerf_mlp *mlp,
                            const float *triplane,
                            int R, float radius,
                            float *density_out) {
    if (!sample_cfg || !mlp || !triplane || !density_out || R < 2) {
        iris_set_error("lrm_density_build_dense: bad arguments");
        return -1;
    }
    const int chunk = 8192;
    const size_t N = (size_t)R * R * R;

    dens_scratch s;
    if (scratch_alloc(&s, chunk, sample_cfg, mlp) != 0) {
        scratch_free(&s);
        iris_set_error("lrm_density_build_dense: oom in scratch");
        return -1;
    }
    float *lin = (float *)malloc((size_t)R * sizeof(float));
    if (!lin) {
        scratch_free(&s);
        iris_set_error("lrm_density_build_dense: oom for linspace");
        return -1;
    }
    for (int i = 0; i < R; i++) {
        lin[i] = -radius + (2.0f * radius) * (float)i / (float)(R - 1);
    }

    for (size_t off = 0; off < N; off += (size_t)chunk) {
        int n_this = (off + chunk <= N) ? chunk : (int)(N - off);
        for (int q = 0; q < n_this; q++) {
            size_t p = off + (size_t)q;
            size_t i = p / ((size_t)R * R);
            size_t j = (p / R) % R;
            size_t k = p % R;
            float *pos = s.positions + (size_t)q * 3;
            pos[0] = lin[i]; pos[1] = lin[j]; pos[2] = lin[k];
        }
        if (eval_chunk(sample_cfg, mlp, triplane, &s, n_this,
                       density_out + off) != 0) {
            free(lin); scratch_free(&s);
            return -1;
        }
    }
    free(lin);
    scratch_free(&s);
    return 0;
}

/* ========================================================================
 * Coarse-to-fine path
 * ======================================================================== */

int lrm_density_build_sparse(const lrm_triplane_sample_cfg *sample_cfg,
                             const lrm_nerf_mlp *mlp,
                             const float *triplane,
                             int R, float radius, float threshold,
                             float *density_out) {
    if (!sample_cfg || !mlp || !triplane || !density_out || R < 2) {
        iris_set_error("lrm_density_build_sparse: bad arguments");
        return -1;
    }

    /* Block stride. B==1 degenerates to a full evaluation (no speedup, but
     * still correct); that only happens for small R where the dense cost is
     * already negligible. */
    int B = R / LRM_DENSITY_COARSE_RES;
    if (B < 1) B = 1;

    /* Coarse node indices along one axis: idx[c] = min(c*B, R-1), so the
     * last node always lands exactly on R-1 (the final block may be
     * shorter than B). CN = number of coarse nodes per axis. */
    const int CN = (R - 1 + B - 1) / B + 1;   /* ceil((R-1)/B) + 1 */
    int *idx = (int *)malloc((size_t)CN * sizeof(int));
    float *lin = (float *)malloc((size_t)R * sizeof(float));
    if (!idx || !lin) {
        free(idx); free(lin);
        iris_set_error("lrm_density_build_sparse: oom (lattice)");
        return -1;
    }
    for (int c = 0; c < CN; c++) {
        int v = c * B;
        idx[c] = (v < R - 1) ? v : (R - 1);
    }
    for (int i = 0; i < R; i++) {
        lin[i] = -radius + (2.0f * radius) * (float)i / (float)(R - 1);
    }

    const int chunk = 8192;
    dens_scratch s;
    if (scratch_alloc(&s, chunk, sample_cfg, mlp) != 0) {
        scratch_free(&s); free(idx); free(lin);
        iris_set_error("lrm_density_build_sparse: oom in scratch");
        return -1;
    }

    /* ----- 1. Evaluate the coarse lattice [CN, CN, CN]. ----- */
    const size_t CN3 = (size_t)CN * CN * CN;
    float *coarse = (float *)malloc(CN3 * sizeof(float));
    if (!coarse) {
        scratch_free(&s); free(idx); free(lin);
        iris_set_error("lrm_density_build_sparse: oom (coarse grid)");
        return -1;
    }
    {
        size_t filled = 0;
        while (filled < CN3) {
            int n_this = (filled + chunk <= CN3) ? chunk : (int)(CN3 - filled);
            for (int q = 0; q < n_this; q++) {
                size_t p = filled + (size_t)q;
                int ci = (int)(p / ((size_t)CN * CN));
                int cj = (int)((p / CN) % CN);
                int ck = (int)(p % CN);
                float *pos = s.positions + (size_t)q * 3;
                pos[0] = lin[idx[ci]];
                pos[1] = lin[idx[cj]];
                pos[2] = lin[idx[ck]];
            }
            if (eval_chunk(sample_cfg, mlp, triplane, &s, n_this,
                           coarse + filled) != 0) {
                free(coarse); scratch_free(&s); free(idx); free(lin);
                return -1;
            }
            filled += (size_t)n_this;
        }
    }

    /* ----- 2. Mark active blocks, then dilate by one block. ----- */
    const int NB = CN - 1;                    /* blocks per axis */
    const size_t NB3 = (size_t)NB * NB * NB;
    uint8_t *active = (uint8_t *)calloc(NB3, 1);
    uint8_t *active2 = (uint8_t *)calloc(NB3, 1);
    if (!active || !active2) {
        free(active); free(active2);
        free(coarse); scratch_free(&s); free(idx); free(lin);
        iris_set_error("lrm_density_build_sparse: oom (block flags)");
        return -1;
    }

    /* A block is near the surface if its corner densities straddle the
     * threshold, or any corner lies within a generous band around it. The
     * exp() density activation makes the field steep at the surface, so a
     * relative band [0.25*t, 4*t] comfortably catches near-surface blocks
     * even when the corners do not bracket the iso-value. */
    const float band_lo = threshold * 0.25f;
    const float band_hi = threshold * 4.0f;
#define CIDX(a,b,c) (((size_t)(a) * CN + (b)) * CN + (c))
    for (int bi = 0; bi < NB; bi++) {
        for (int bj = 0; bj < NB; bj++) {
            for (int bk = 0; bk < NB; bk++) {
                float mn = coarse[CIDX(bi, bj, bk)];
                float mx = mn;
                for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++) {
                    float v = coarse[CIDX(bi + di, bj + dj, bk + dk)];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                int straddle = (mn <= threshold && mx >= threshold);
                int near_band = (mx >= band_lo && mn <= band_hi);
                if (straddle || near_band) {
                    active[((size_t)bi * NB + bj) * NB + bk] = 1;
                }
            }
        }
    }
#undef CIDX

    /* Dilate: a block is in active2 if itself or any 26-neighbor is active. */
    for (int bi = 0; bi < NB; bi++) {
        for (int bj = 0; bj < NB; bj++) {
            for (int bk = 0; bk < NB; bk++) {
                int hit = 0;
                for (int di = -1; di <= 1 && !hit; di++) {
                    int ni = bi + di; if (ni < 0 || ni >= NB) continue;
                    for (int dj = -1; dj <= 1 && !hit; dj++) {
                        int nj = bj + dj; if (nj < 0 || nj >= NB) continue;
                        for (int dk = -1; dk <= 1 && !hit; dk++) {
                            int nk = bk + dk; if (nk < 0 || nk >= NB) continue;
                            if (active[((size_t)ni * NB + nj) * NB + nk]) hit = 1;
                        }
                    }
                }
                active2[((size_t)bi * NB + bj) * NB + bk] = (uint8_t)hit;
            }
        }
    }
    free(active);

    /* ----- 3. Mark exactly-evaluated grid nodes. Each active block covers
     * the inclusive node range [idx[b], idx[b+1]] on every axis; the
     * inclusive upper bound makes adjacent active blocks share their common
     * face, so every cell inside an active block has all 8 corners marked
     * exact. We splat directly from the 3D active2 block grid (a per-axis
     * mask would over-mark a full cuboid). ----- */
    const size_t N = (size_t)R * R * R;
    uint8_t *exact = (uint8_t *)calloc(N, 1);
    if (!exact) {
        free(active2); free(coarse); scratch_free(&s); free(idx); free(lin);
        iris_set_error("lrm_density_build_sparse: oom (exact mask)");
        return -1;
    }
    for (int bi = 0; bi < NB; bi++) {
        int i0 = idx[bi], i1 = idx[bi + 1];
        for (int bj = 0; bj < NB; bj++) {
            int j0 = idx[bj], j1 = idx[bj + 1];
            for (int bk = 0; bk < NB; bk++) {
                if (!active2[((size_t)bi * NB + bj) * NB + bk]) continue;
                int k0 = idx[bk], k1 = idx[bk + 1];
                for (int i = i0; i <= i1; i++) {
                    for (int j = j0; j <= j1; j++) {
                        uint8_t *row = exact + ((size_t)i * R + j) * R;
                        for (int k = k0; k <= k1; k++) row[k] = 1;
                    }
                }
            }
        }
    }
    free(active2);

    /* ----- 4. Trilinearly fill the whole grid from the coarse lattice. This
     * gives every node a sign-correct value; exact nodes are overwritten in
     * step 5. Inactive cells never cross the threshold so they emit no
     * triangles regardless. ----- */
    for (int i = 0; i < R; i++) {
        int ci = i / B; if (ci > CN - 2) ci = CN - 2;
        int i0 = idx[ci], i1 = idx[ci + 1];
        float ti = (i1 > i0) ? (float)(i - i0) / (float)(i1 - i0) : 0.0f;
        for (int j = 0; j < R; j++) {
            int cj = j / B; if (cj > CN - 2) cj = CN - 2;
            int j0 = idx[cj], j1 = idx[cj + 1];
            float tj = (j1 > j0) ? (float)(j - j0) / (float)(j1 - j0) : 0.0f;
            float *out_row = density_out + ((size_t)i * R + j) * R;
            for (int k = 0; k < R; k++) {
                int ck = k / B; if (ck > CN - 2) ck = CN - 2;
                int k0 = idx[ck], k1 = idx[ck + 1];
                float tk = (k1 > k0) ? (float)(k - k0) / (float)(k1 - k0) : 0.0f;
                /* 8 coarse corners. */
                const float *c000 = coarse + (((size_t)ci * CN + cj) * CN + ck);
                float c00 = c000[0]      + tk * (c000[1]      - c000[0]);
                float c01 = c000[CN]     + tk * (c000[CN + 1] - c000[CN]);
                float c10 = c000[(size_t)CN*CN]     + tk * (c000[(size_t)CN*CN + 1]     - c000[(size_t)CN*CN]);
                float c11 = c000[(size_t)CN*CN + CN] + tk * (c000[(size_t)CN*CN + CN + 1] - c000[(size_t)CN*CN + CN]);
                float c0 = c00 + tj * (c01 - c00);
                float c1 = c10 + tj * (c11 - c10);
                out_row[k] = c0 + ti * (c1 - c0);
            }
        }
    }

    /* ----- 5. Exactly evaluate the marked nodes, chunked. ----- */
    {
        /* Indices buffer for the current chunk so we can scatter results. */
        size_t *idxbuf = (size_t *)malloc((size_t)chunk * sizeof(size_t));
        if (!idxbuf) {
            free(exact); free(coarse); scratch_free(&s); free(idx); free(lin);
            iris_set_error("lrm_density_build_sparse: oom (index buffer)");
            return -1;
        }
        int n_pack = 0;
        size_t n_exact = 0;
        for (size_t p = 0; p < N; p++) {
            if (!exact[p]) continue;
            n_exact++;
            int i = (int)(p / ((size_t)R * R));
            int j = (int)((p / R) % R);
            int k = (int)(p % R);
            float *pos = s.positions + (size_t)n_pack * 3;
            pos[0] = lin[i]; pos[1] = lin[j]; pos[2] = lin[k];
            idxbuf[n_pack] = p;
            n_pack++;
            if (n_pack == chunk) {
                if (eval_chunk(sample_cfg, mlp, triplane, &s, n_pack,
                               s.density) != 0) {
                    free(idxbuf); free(exact); free(coarse);
                    scratch_free(&s); free(idx); free(lin);
                    return -1;
                }
                for (int q = 0; q < n_pack; q++) density_out[idxbuf[q]] = s.density[q];
                n_pack = 0;
            }
        }
        if (n_pack > 0) {
            if (eval_chunk(sample_cfg, mlp, triplane, &s, n_pack,
                           s.density) != 0) {
                free(idxbuf); free(exact); free(coarse);
                scratch_free(&s); free(idx); free(lin);
                return -1;
            }
            for (int q = 0; q < n_pack; q++) density_out[idxbuf[q]] = s.density[q];
        }
        free(idxbuf);

        if (getenv("LRM_TIMING")) {
            fprintf(stderr,
                    "lrmc:   (density sparse: B=%d coarse=%d^3 exact=%zu/%zu = %.1f%%)\n",
                    B, CN, n_exact, N, 100.0 * (double)n_exact / (double)N);
        }
    }

    free(exact);
    free(coarse);
    scratch_free(&s);
    free(idx);
    free(lin);
    return 0;
}

/* ========================================================================
 * Analytic gradient normals
 * ======================================================================== */

/* Trilinear sample of the density grid at fractional grid coords (gi,gj,gk),
 * clamped to the valid [0, R-1] range. */
static float sample_density(const float *d, int R, float gi, float gj, float gk) {
    if (gi < 0.0f) gi = 0.0f; else if (gi > (float)(R - 1)) gi = (float)(R - 1);
    if (gj < 0.0f) gj = 0.0f; else if (gj > (float)(R - 1)) gj = (float)(R - 1);
    if (gk < 0.0f) gk = 0.0f; else if (gk > (float)(R - 1)) gk = (float)(R - 1);
    int i0 = (int)gi, j0 = (int)gj, k0 = (int)gk;
    int i1 = (i0 < R - 1) ? i0 + 1 : i0;
    int j1 = (j0 < R - 1) ? j0 + 1 : j0;
    int k1 = (k0 < R - 1) ? k0 + 1 : k0;
    float ti = gi - (float)i0, tj = gj - (float)j0, tk = gk - (float)k0;
#define D(a,b,c) d[(((size_t)(a) * R + (b)) * R + (c))]
    float c00 = D(i0,j0,k0) + tk * (D(i0,j0,k1) - D(i0,j0,k0));
    float c01 = D(i0,j1,k0) + tk * (D(i0,j1,k1) - D(i0,j1,k0));
    float c10 = D(i1,j0,k0) + tk * (D(i1,j0,k1) - D(i1,j0,k0));
    float c11 = D(i1,j1,k0) + tk * (D(i1,j1,k1) - D(i1,j1,k0));
#undef D
    float c0 = c00 + tj * (c01 - c00);
    float c1 = c10 + tj * (c11 - c10);
    return c0 + ti * (c1 - c0);
}

int lrm_density_gradient_normals(const float *density, int R,
                                 float world_min, float world_max,
                                 const float *vertices, int Nv,
                                 const int32_t *faces, int Nf,
                                 float *out_normals) {
    if (!density || !vertices || !out_normals || R < 2 || Nv <= 0) {
        iris_set_error("lrm_density_gradient_normals: bad arguments");
        return -1;
    }
    const float span = world_max - world_min;
    const float to_grid = (span != 0.0f) ? (float)(R - 1) / span : 0.0f;
    const float h = 1.0f;   /* one grid cell, in grid coords */
    (void)faces; (void)Nf;

    /* The density field unambiguously defines inside vs outside: density is
     * large inside the object and small outside, so its gradient points
     * inward (toward increasing density) and the outward surface normal is
     * exactly the negated, normalized gradient. No reference to the triangle
     * winding is needed (and indeed must be avoided: the marching-cubes
     * winding here is inverted, so aligning to it would point normals
     * inward). Central differences over one grid cell give a smooth field
     * normal that removes MC stair-stepping from the shading. */
    for (int v = 0; v < Nv; v++) {
        const float *p = vertices + (size_t)v * 3;
        float gi = (p[0] - world_min) * to_grid;
        float gj = (p[1] - world_min) * to_grid;
        float gk = (p[2] - world_min) * to_grid;
        float dx = sample_density(density, R, gi + h, gj, gk)
                 - sample_density(density, R, gi - h, gj, gk);
        float dy = sample_density(density, R, gi, gj + h, gk)
                 - sample_density(density, R, gi, gj - h, gk);
        float dz = sample_density(density, R, gi, gj, gk + h)
                 - sample_density(density, R, gi, gj, gk - h);
        float nx = -dx, ny = -dy, nz = -dz;   /* outward = -gradient */
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) {
            float inv = 1.0f / len;
            nx *= inv; ny *= inv; nz *= inv;
        } else {
            /* Degenerate (flat field at the vertex) - rare. Leave a unit
             * +Z so the GLB stays valid; with doubleSided shading this has
             * no visible effect. */
            nx = 0.0f; ny = 0.0f; nz = 1.0f;
        }
        out_normals[v*3+0] = nx;
        out_normals[v*3+1] = ny;
        out_normals[v*3+2] = nz;
    }
    return 0;
}
