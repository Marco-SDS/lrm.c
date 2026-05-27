/*
 * lrm_density.h - density-grid construction for marching cubes.
 *
 * Two builders with identical signature and identical [R,R,R] output:
 *
 *   lrm_density_build_dense  - reference path. Queries triplane_sample +
 *                              NeRF MLP at every one of the R^3 grid nodes.
 *                              This is exactly what TripoSR upstream does.
 *
 *   lrm_density_build_sparse - coarse-to-fine path. The TripoSR density
 *                              field is band-limited at the triplane
 *                              resolution (64x64 per plane bilinearly
 *                              sampled), so it carries no detail finer than
 *                              ~R/64 voxels. We evaluate a coarse lattice
 *                              (~64^3), refine only blocks straddling the
 *                              iso-threshold (plus a 1-block dilation), and
 *                              trilinearly fill the rest. The marching-cubes
 *                              surface is bit-identical to the dense path
 *                              (only cells far from the surface differ, and
 *                              those produce no triangles). At R=256 this
 *                              cuts MLP queries ~9x.
 *
 * The grid layout is [R, R, R] C-contiguous: density[(i*R + j)*R + k]
 * corresponds to world position (lin[i], lin[j], lin[k]) with
 * lin[t] = -radius + 2*radius*t/(R-1), matching extract_golden.py's
 * meshgrid('ij') view and lrm_marching_cubes_extract's convention.
 */

#ifndef LRM_LRM_DENSITY_H
#define LRM_LRM_DENSITY_H

#include <stdint.h>

#include "lrm_nerf_mlp.h"
#include "lrm_triplane_sample.h"

/* Reference dense builder. density_out must hold R*R*R floats.
 * Returns 0 on success, -1 with iris_get_error() set on failure. */
int lrm_density_build_dense(const lrm_triplane_sample_cfg *sample_cfg,
                            const lrm_nerf_mlp *mlp,
                            const float *triplane,
                            int R, float radius,
                            float *density_out);

/* Coarse-to-fine builder. `threshold` is the marching-cubes iso-value used
 * to decide which blocks straddle the surface and must be refined.
 * density_out must hold R*R*R floats.
 * Returns 0 on success, -1 with iris_get_error() set on failure. */
int lrm_density_build_sparse(const lrm_triplane_sample_cfg *sample_cfg,
                             const lrm_nerf_mlp *mlp,
                             const float *triplane,
                             int R, float radius, float threshold,
                             float *density_out);

/*
 * Compute smooth per-vertex normals as the (normalized, negated) gradient
 * of the density field at each mesh vertex. This is the standard way to get
 * high-quality shading normals from an implicit field: it removes the
 * stair-stepping that face-averaged normals bake in, without touching the
 * geometry. The global sign is aligned to the marching-cubes winding (via
 * the face-averaged normals) so the orientation matches the rest of the
 * pipeline; vertices where the gradient is degenerate fall back to the
 * face-averaged normal.
 *
 *   density            : [R, R, R] grid used for marching cubes
 *   world_min/world_max: the same world mapping passed to MC
 *   vertices           : [Nv, 3] world-space MC vertices
 *   faces              : [Nf, 3] used only for the orientation reference
 *   out_normals        : [Nv, 3] unit normals (caller-owned)
 *
 * Returns 0 on success, -1 with iris_get_error() set on failure.
 */
int lrm_density_gradient_normals(const float *density, int R,
                                 float world_min, float world_max,
                                 const float *vertices, int Nv,
                                 const int32_t *faces, int Nf,
                                 float *out_normals);

#endif /* LRM_LRM_DENSITY_H */
