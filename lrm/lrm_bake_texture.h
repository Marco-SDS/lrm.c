/*
 * lrm_bake_texture.h - Per-triangle grid UV atlas + texture rasterization.
 *
 * Replaces TripoSR's xatlas + moderngl approach with a self-contained
 * C path. Each triangle gets its own square cell in the output texture
 * (grid layout: S = ceil(sqrt(N_faces)) cells per side, cell width =
 * tex_res / S pixels). Inside each cell the triangle is rasterized as
 * an inscribed right triangle. There is no triangle overlap and no
 * UV optimization — quality is below xatlas-packed atlases, but the
 * resulting texture is a real spatial atlas (each texel maps to a
 * unique world-space position) and the pipeline is dependency-free.
 *
 * Trade-off vs xatlas:
 *   - About half the texture area is unused (the unused half-cell per
 *     triangle plus the unused tail at the bottom-right of the grid).
 *   - No chart-based spatial coherence: adjacent triangles in the mesh
 *     can end up far apart in the texture. Mipmaps will not work
 *     correctly without manual margin handling, but for diffuse
 *     baseColor this is fine.
 *   - Per-vertex-per-triangle UVs (3 UVs per face) → the GLB needs to
 *     duplicate vertices: 3 * N_faces vertices total, not N_vertices.
 *
 * Inputs: the existing scene_code (triplane) and the post-MC mesh.
 * Outputs: UVs (one per triangle vertex) + texture (RGBA u8).
 */

#ifndef LRM_LRM_BAKE_TEXTURE_H
#define LRM_LRM_BAKE_TEXTURE_H

#include <stddef.h>
#include <stdint.h>

#include "lrm_nerf_mlp.h"
#include "lrm_triplane_sample.h"

typedef struct {
    int   texture_resolution;  /* e.g. 2048 */
    int   padding;             /* texel inset within each cell, default 2 */
    int   chunk_size;          /* NeRF MLP query chunk, default 8192 */
    /* Background color for unused texels (RGBA u8). Default (128, 128, 128, 0). */
    uint8_t bg_r, bg_g, bg_b, bg_a;
} lrm_bake_cfg;

#define LRM_BAKE_CFG_DEFAULT { 2048, 2, 8192, 128, 128, 128, 0 }

/*
 * Build the texture atlas.
 *
 *   ts_cfg          : triplane sampler config (used as in lrm_infer)
 *   mlp             : initialized NeRF MLP
 *   triplane        : [3, 40, 64, 64] scene code
 *   vertices        : [n_vertices, 3] world-space mesh vertices
 *   faces           : [n_faces, 3] vertex indices
 *   n_vertices, n_faces : counts
 *   cfg             : bake configuration (or NULL for LRM_BAKE_CFG_DEFAULT)
 *
 *   out_uvs         : malloc'd [n_faces * 3, 2] (caller frees). UVs are
 *                     ordered (v0, v1, v2) per face matching `faces`.
 *   out_tex_rgba    : malloc'd [tex_res * tex_res * 4] uint8 (caller frees).
 *
 * Returns 0 on success.
 */
int lrm_bake_texture(const lrm_triplane_sample_cfg *ts_cfg,
                     const lrm_nerf_mlp *mlp,
                     const float *triplane,
                     const float *vertices, int n_vertices,
                     const int32_t *faces,  int n_faces,
                     const lrm_bake_cfg *cfg,
                     float **out_uvs,
                     uint8_t **out_tex_rgba);

#endif /* LRM_LRM_BAKE_TEXTURE_H */
