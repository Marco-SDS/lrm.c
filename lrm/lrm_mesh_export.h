/*
 * lrm_mesh_export.h - mesh struct + GLB writer.
 *
 * Phase 11 deliverable. The public `lrm_mesh` type (declared opaque in
 * lrm.h) is defined here as a plain owned-buffer container, and the
 * single output we ship is binary glTF 2.0 with embedded vertex colors.
 *
 * We deliberately do NOT vendor cgltf or any other glTF library: the
 * subset of the format we need (one primitive, three accessors, one
 * buffer) is ~150 lines of straightforward serialization. Keeping the
 * dependency footprint at libc + math matches the project's philosophy
 * (LRMengine.md section 2 tenet 1).
 */

#ifndef LRM_LRM_MESH_EXPORT_H
#define LRM_LRM_MESH_EXPORT_H

#include <stdint.h>

#include "lrm.h"

/* The public lrm_mesh type (opaque in lrm.h) is defined here so the
 * GLB writer and the inference glue can both touch the fields.
 *
 * Three modes the writer supports:
 *
 *   1) Indexed + vertex colors (Phase 11/12 default):
 *        vertices [Nv, 3], faces [Nf, 3], vertex_colors [Nv, 4],
 *        uvs = NULL, texture_png = NULL.
 *
 *   2) Indexed + UVs + texture (Phase 18 --bake-texture):
 *        vertices [Nv, 3], faces [Nf, 3], uvs [Nv, 2],
 *        texture_png = encoded PNG bytes, texture_png_size = byte length.
 *        Use vertex_colors = NULL in this mode (the texture replaces it).
 *        Note: per-triangle UVs require duplicating shared vertices, so
 *        in practice Nv = 3 * Nf and `faces` are trivial [0,1,2,3,...].
 *
 *   3) Both vertex_colors + texture are also legal; the glTF spec
 *      multiplies them per the material's PBR rules. lrm.c doesn't
 *      emit this combination today.
 */
struct lrm_mesh {
    int      n_vertices;
    int      n_faces;
    float   *vertices;        /* [n_vertices, 3] world-space, owned */
    int32_t *faces;           /* [n_faces, 3] vertex indices, owned */
    float   *vertex_colors;   /* [n_vertices, 4] RGBA in [0, 1], owned, or NULL */
    float   *uvs;             /* [n_vertices, 2] in [0, 1], owned, or NULL */
    uint8_t *texture_png;     /* encoded PNG bytes (whole file), owned, or NULL */
    size_t   texture_png_size;
};

/*
 * Build a mesh by taking ownership of the given buffers. After this
 * call the caller must not free them - lrm_mesh_free will. Pass NULL
 * for any optional field to skip emitting the corresponding glTF
 * attribute / material binding.
 *
 * Returns NULL on OOM.
 */
lrm_mesh *lrm_mesh_from_buffers(int n_vertices, float *vertices,
                                int n_faces, int32_t *faces,
                                float *vertex_colors);

/* Attach UVs and a PNG-encoded texture to an existing mesh. Transfers
 * ownership of both buffers. */
void lrm_mesh_set_texture(lrm_mesh *mesh,
                          float *uvs,
                          uint8_t *texture_png, size_t texture_png_size);

#endif /* LRM_LRM_MESH_EXPORT_H */
