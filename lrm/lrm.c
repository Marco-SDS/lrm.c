/*
 * lrm.c - LRM coordinator: dispatch table for model-kind specific code.
 *
 * Phase 5 state: lrm_load() routes to the TripoSR loader (only model kind
 * currently supported); lrm_free() releases the model. lrm_infer() and
 * lrm_mesh_save_glb() are still stubs - their implementations come in
 * Phases 6-11.
 *
 * When a second model kind lands (OpenLRM in Phase 15), this file gains a
 * dispatch table keyed on a `kind` field in struct lrm_model.
 */

#include "lrm.h"
#include "lrm_triposr.h"

#include <stddef.h>

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

lrm_model *lrm_load(const char *model_dir) {
    /* Only one model kind today; future variants will dispatch here based
     * on a small probe (config.yaml model_class, or filename heuristics). */
    return lrm_triposr_load(model_dir);
}

void lrm_free(lrm_model *m) {
    lrm_triposr_free(m);
}

/* ========================================================================
 * Inference
 * ======================================================================== */

lrm_mesh *lrm_infer(lrm_model *m, const iris_image *im,
                    const lrm_infer_opts *opts) {
    (void)m;
    (void)im;
    (void)opts;
    iris_set_error(
        "lrm_infer: not implemented yet "
        "(Phases 6-10 - encoder, decoder, sampler, MC)");
    return NULL;
}

/* ========================================================================
 * Mesh I/O
 * ======================================================================== */

int lrm_mesh_save_glb(const lrm_mesh *mesh, const char *path) {
    (void)mesh;
    (void)path;
    iris_set_error("lrm_mesh_save_glb: not implemented yet (Phase 11 - GLB)");
    return -1;
}

void lrm_mesh_free(lrm_mesh *mesh) {
    (void)mesh;
    /* No state to release yet. */
}
