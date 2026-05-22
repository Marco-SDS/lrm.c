/*
 * lrm.c - LRM coordinator: dispatch table for model-kind specific code.
 *
 * Phase 11 state: load/free/save_glb are live, lrm_infer is still a
 * stub awaiting Phase 12 (the end-to-end pipeline wiring + CLI).
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
        "lrm_infer: end-to-end pipeline wiring lands in Phase 12; "
        "individual stages (encoder/decoder/sampler/MC/GLB) are all tested.");
    return NULL;
}

/* lrm_mesh_save_glb and lrm_mesh_free live in lrm_mesh_export.c. */
