/*
 * lrm_triplane_sample.c - Per-point triplane feature extraction.
 *
 * Hot path:
 *   - Scale positions [-radius, +radius] -> [-1, +1].
 *   - Stack into 3 sets of 2D coords (xy, xz, yz) at layout
 *     [planes=3, N_points, 2] - what iris_grid_sample_bilinear consumes.
 *   - Bilinear sample with padding_mode=zeros, output [3, 40, N_points].
 *   - Transpose into [N_points, 120] (the layout the NeRF MLP wants).
 */

#include "lrm_triplane_sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"
#include "iris_kernels.h"

/* Workspace layout:
 *   grid    : [planes, N_points, 2]            = 6*N      floats
 *   sampled : [planes, channels, N_points]     = 120*N    floats   (for cfg defaults)
 * Total ~126 * N floats. */

size_t lrm_triplane_sample_workspace_bytes(const lrm_triplane_sample_cfg *cfg,
                                           int N_points) {
    size_t grid_f    = (size_t)cfg->planes * N_points * 2;
    size_t sampled_f = (size_t)cfg->planes * cfg->channels * N_points;
    return (grid_f + sampled_f) * sizeof(float);
}

int lrm_triplane_sample_forward(const lrm_triplane_sample_cfg *cfg,
                                const float *triplane,
                                const float *positions,
                                int N_points,
                                float *out,
                                float *work) {
    if (!cfg || !triplane || !positions || !out || !work) {
        iris_set_error("triplane_sample: NULL argument");
        return -1;
    }
    const int Np = cfg->planes;
    const int C  = cfg->channels;
    const int S  = cfg->size;
    const float inv_radius = 1.0f / cfg->radius;

    /* Carve workspace. */
    float *grid    = work;
    float *sampled = work + (size_t)Np * N_points * 2;

    /* ----- 1. Build per-plane (col, row) normalized coords.
     *           Layout: grid[plane, point, axis] with axis 0 = col (x),
     *           axis 1 = row (y) per the grid_sample contract.
     *
     *           Plane 0 (xy): col = x, row = y
     *           Plane 1 (xz): col = x, row = z
     *           Plane 2 (yz): col = y, row = z
     */
    float *g0 = grid + (size_t)0 * N_points * 2;
    float *g1 = grid + (size_t)1 * N_points * 2;
    float *g2 = grid + (size_t)2 * N_points * 2;
    for (int n = 0; n < N_points; n++) {
        float x = positions[(size_t)n * 3 + 0] * inv_radius;
        float y = positions[(size_t)n * 3 + 1] * inv_radius;
        float z = positions[(size_t)n * 3 + 2] * inv_radius;
        g0[n * 2 + 0] = x; g0[n * 2 + 1] = y;
        g1[n * 2 + 0] = x; g1[n * 2 + 1] = z;
        g2[n * 2 + 0] = y; g2[n * 2 + 1] = z;
    }

    /* ----- 2. Bilinear grid_sample, padding_mode='zeros' (TripoSR default). */
    iris_grid_sample_bilinear(sampled, triplane, grid,
                              Np, C, /*H=*/S, /*W=*/S, N_points,
                              IRIS_GS_PAD_ZEROS);

    /* ----- 3. Rearrange "Np Cp N -> N (Np Cp)" into `out`.
     *           sampled layout: sampled[plane, channel, n]
     *           target layout:  out[n, plane*Cp + channel]
     */
    const int feat_dim = Np * C;  /* 120 */
    for (int p = 0; p < Np; p++) {
        const float *plane_in = sampled + (size_t)p * C * N_points;
        for (int c = 0; c < C; c++) {
            const float *row = plane_in + (size_t)c * N_points;
            int out_col = p * C + c;
            for (int n = 0; n < N_points; n++) {
                out[(size_t)n * feat_dim + out_col] = row[n];
            }
        }
    }
    return 0;
}
