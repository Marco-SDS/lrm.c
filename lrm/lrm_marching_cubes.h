/*
 * lrm_marching_cubes.h - Lorensen-Cline marching cubes on a 3D density grid.
 *
 * Extracts a triangle mesh from an isosurface at level=threshold of an
 * R x R x R density volume. The input convention matches TripoSR's
 * MarchingCubeHelper: callers pass the post-activation density (e.g.
 * exp(raw_density + bias)) and a threshold; internally we compute
 * `level[i,j,k] = threshold - density[i,j,k]` and march on the zero
 * level, which is the same surface as `density == threshold`.
 *
 * Vertex deduplication is enforced via a global edge ID table so each
 * unique grid edge produces exactly one vertex (matches torchmcubes'
 * behavior; without dedup the vertex count would be ~3x).
 *
 * Output coordinate convention: matches TripoSR's MarchingCubeHelper
 * which applies a [2, 1, 0] axis swap on the raw mcubes output. After
 * that and the (0, 1) -> (world_min, world_max) scale, vertex[0]
 * corresponds to the input's K axis (third dim), vertex[1] to J,
 * vertex[2] to I.
 *
 * Memory: vertex/face buffers grow dynamically with realloc. Peak edge
 * dedup table is 3*R^3 ints = ~3.1 MB at R=64, ~200 MB at R=256.
 */

#ifndef LRM_LRM_MARCHING_CUBES_H
#define LRM_LRM_MARCHING_CUBES_H

#include <stdint.h>

typedef struct {
    int       n_vertices;
    int       n_faces;
    float    *vertices;  /* [n_vertices, 3] world-space, owned */
    int32_t  *faces;     /* [n_faces, 3] vertex indices, owned */
} lrm_mc_mesh;

/*
 * Extract a triangle mesh from a density volume.
 *
 *   volume    : [R, R, R] f32 C-contiguous density grid
 *   R         : resolution
 *   threshold : iso-value (typically 25.0 for TripoSR)
 *   world_min, world_max : linearly map grid coords [0, R-1] / (R-1) to
 *                          [world_min, world_max] for vertex output.
 *                          Pass (-radius, +radius) for TripoSR.
 *   mesh      : output struct, populated on success. Buffers owned by the
 *               caller; release with lrm_mc_mesh_free.
 *
 * Returns 0 on success; -1 with iris_get_error() set on failure.
 */
int lrm_marching_cubes_extract(const float *volume, int R,
                               float threshold,
                               float world_min, float world_max,
                               lrm_mc_mesh *mesh);

/*
 * Remove small disconnected components ("floaters") in place. Vertices are
 * grouped into connected components via shared faces (union-find); any
 * component whose triangle count is below `min_fraction` of the largest
 * component's triangle count is discarded, and the surviving vertices/faces
 * are compacted (indices remapped). The largest component is always kept.
 *
 * This cleans up the spurious detached blobs TripoSR density fields produce
 * around the main object without affecting the main surface. Pass
 * min_fraction <= 0 to disable (no-op).
 *
 * Returns 0 on success, -1 with iris_get_error() set on failure. On success
 * mesh->n_vertices / n_faces and the buffers reflect the cleaned mesh.
 */
int lrm_mc_remove_small_components(lrm_mc_mesh *mesh, float min_fraction);

void lrm_mc_mesh_free(lrm_mc_mesh *mesh);

#endif /* LRM_LRM_MARCHING_CUBES_H */
