/*
 * lrm.c - LRM coordinator: dispatch table for model-kind specific code.
 *
 * Phase 3 state: every entry point is a stub that fails predictably with
 * a message identifying the phase in LRMengine.md that will implement it.
 * The skeleton exists so callers (and the Makefile) can be wired up now,
 * and individual modules under lrm/ can be filled in over Phases 5-11.
 */

#include "lrm.h"

#include <stddef.h>

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

lrm_model *lrm_load(const char *model_dir) {
    (void)model_dir;
    iris_set_error(
        "lrm_load: not implemented yet (Phase 5 - safetensors loader)");
    return NULL;
}

void lrm_free(lrm_model *m) {
    (void)m;
    /* No state to release yet. */
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
