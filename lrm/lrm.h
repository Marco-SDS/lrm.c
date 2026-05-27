/*
 * lrm.h - Public API for LRM-family image-to-3D inference.
 *
 * Companion to iris.h (which exposes only generic image I/O + opaque types).
 * Everything model-specific lives under the lrm/ subdirectory and behind
 * this header. See LRMengine.md for the architectural plan.
 *
 * Phase 3 state: API skeleton. All functions are stubs that return safe
 * failure values and set iris_get_error() to indicate the phase that will
 * implement them. The shape of the public surface is committed; the
 * implementations land in Phase 5+ (load), 6-9 (forward), 10 (mesh),
 * 11 (GLB export).
 *
 * Threading: every lrm_* function is single-threaded by contract.
 * Concurrency is the caller's responsibility (separate lrm_model
 * instances, or shared model + per-thread arena). See LRMengine.md
 * section 12.6.
 */

#ifndef LRM_H
#define LRM_H

#include <stddef.h>
#include <stdint.h>

#include "iris.h"   /* for iris_image, iris_get_error */

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Opaque types
 * ======================================================================== */

typedef struct lrm_model lrm_model;
typedef struct lrm_mesh  lrm_mesh;

/* ========================================================================
 * Inference options
 * ======================================================================== */

typedef struct {
    /* Marching cubes grid resolution. TripoSR default is 256; golden tests
     * use 64 for speed. Smaller = faster + lower-res mesh, larger = slower
     * + smoother mesh. */
    int   mc_resolution;

    /* Density threshold for marching cubes, applied to the post-activation
     * density (i.e. after the exp(raw - 1.0) transform for TripoSR).
     * Higher values produce smaller / smoother meshes. */
    float density_threshold;

    /* World-space half-extent for the density grid. The MC grid samples
     * uniformly in [-radius, +radius]^3. TripoSR default 0.87. */
    float radius;

    /* If non-zero, bake a per-triangle UV atlas + PNG texture instead of
     * emitting per-vertex colors. The resulting GLB references the texture
     * via materials[0].pbrMetallicRoughness.baseColorTexture and carries
     * TEXCOORD_0. */
    int   bake_texture;

    /* Texture atlas resolution when bake_texture != 0. Default 2048.
     * Memory: tex_res^2 * 4 bytes + small scratch. */
    int   texture_resolution;

    /* Fraction of the (square) canvas the foreground occupies during
     * preprocessing. TripoSR default 0.85. Lower values shrink the object
     * (more context, less detail); higher values zoom in (more detail, risk
     * of silhouette clipping). Sweeping this is the single most effective
     * input-quality knob per the TripoSR demo. Valid range (0, 1]. */
    float foreground_ratio;
} lrm_infer_opts;

#define LRM_INFER_OPTS_DEFAULT { 256, 25.0f, 0.87f, 0, 2048, 0.85f }

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

/*
 * Load an LRM model from a directory. Currently only the TripoSR layout is
 * supported (a directory containing config.yaml and model.safetensors).
 * Returns NULL on error; the reason is available via iris_get_error().
 */
lrm_model *lrm_load(const char *model_dir);

/*
 * Release all resources held by the model. Safe on NULL.
 */
void lrm_free(lrm_model *m);

/* ========================================================================
 * Inference
 * ======================================================================== */

/*
 * Run end-to-end image-to-mesh inference. `im` is borrowed by the call and
 * not freed. `opts` may be NULL to accept all defaults (LRM_INFER_OPTS_DEFAULT).
 * Returns a freshly allocated mesh on success, or NULL with an error set
 * via iris_get_error() on failure.
 */
lrm_mesh *lrm_infer(lrm_model *m, const iris_image *im,
                    const lrm_infer_opts *opts);

/* ========================================================================
 * Mesh I/O
 * ======================================================================== */

/*
 * Serialize a mesh to GLB (binary glTF 2.0). Vertex colors are embedded;
 * texture baking is not yet supported. Returns 0 on success, -1 on error.
 */
int lrm_mesh_save_glb(const lrm_mesh *mesh, const char *path);

/*
 * Free a mesh. Safe on NULL.
 */
void lrm_mesh_free(lrm_mesh *mesh);

#ifdef __cplusplus
}
#endif

#endif /* LRM_H */
