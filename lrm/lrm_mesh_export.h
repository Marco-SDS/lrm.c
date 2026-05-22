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
 * GLB writer and the inference glue can both touch the fields. */
struct lrm_mesh {
    int      n_vertices;
    int      n_faces;
    float   *vertices;       /* [n_vertices, 3] world-space, owned */
    int32_t *faces;          /* [n_faces, 3] vertex indices, owned */
    float   *vertex_colors;  /* [n_vertices, 4] RGBA in [0, 1], owned;
                              * NULL means the writer emits no COLOR_0. */
};

/*
 * Build a mesh by taking ownership of the given buffers. After this
 * call the caller must not free them - lrm_mesh_free will. Pass NULL
 * for `vertex_colors` to skip emitting a COLOR_0 attribute.
 *
 * Returns NULL on OOM.
 */
lrm_mesh *lrm_mesh_from_buffers(int n_vertices, float *vertices,
                                int n_faces, int32_t *faces,
                                float *vertex_colors);

#endif /* LRM_LRM_MESH_EXPORT_H */
