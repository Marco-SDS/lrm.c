/*
 * lrm_triplane_sample.h - Per-point triplane feature extraction.
 *
 * Mirrors TripoSR's TriplaneNeRFRenderer.query_triplane (tsr/models/nerf_renderer.py).
 * For each 3D query point in world coordinates [-radius, +radius]^3:
 *
 *   1. Scale position to [-1, +1]^3 (grid_sample normalized space).
 *   2. Form three 2D coordinates per point:
 *        xy plane: (x, y)
 *        xz plane: (x, z)
 *        yz plane: (y, z)
 *      The triplet matches TripoSR's index_pairs [[0,1],[0,2],[1,2]].
 *   3. Bilinear grid_sample on each plane, padding_mode='zeros' (PyTorch
 *      default; matches what TripoSR calls).
 *   4. Concatenate the 3 plane outputs into a 120-d feature vector per point.
 *
 * The N_points dimension is taken in chunks: at MC resolution 256, the full
 * 16.78M query set would need ~2 GB of intermediate buffers, so callers
 * should iterate in chunks of ~8192 (TripoSR's chunk_size default).
 *
 * The triplane input is the post-upsample output [planes=3, C=40, H=64, W=64].
 * Output layout is [N_points, planes * C] = [N_points, 120] row-major.
 */

#ifndef LRM_LRM_TRIPLANE_SAMPLE_H
#define LRM_LRM_TRIPLANE_SAMPLE_H

#include <stddef.h>

typedef struct lrm_triplane_sample_cfg {
    int planes;          /* 3 */
    int channels;        /* 40 */
    int size;            /* 64 */
    float radius;        /* 0.87 */
} lrm_triplane_sample_cfg;

/* Workspace bytes needed for a single forward of N_points queries. */
size_t lrm_triplane_sample_workspace_bytes(const lrm_triplane_sample_cfg *cfg,
                                           int N_points);

/*
 * Sample the triplane at N_points query positions.
 *
 *   triplane : [planes, channels, size, size]   borrowed
 *   positions: [N_points, 3] in world space, components in [-radius, +radius]
 *   out      : [N_points, planes * channels = 120] row-major
 *   work     : scratch >= lrm_triplane_sample_workspace_bytes(cfg, N_points)
 *
 * Returns 0 on success.
 */
int lrm_triplane_sample_forward(const lrm_triplane_sample_cfg *cfg,
                                const float *triplane,
                                const float *positions,
                                int N_points,
                                float *out,
                                float *work);

#endif /* LRM_LRM_TRIPLANE_SAMPLE_H */
