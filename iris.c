/*
 * iris.c - Generic engine coordinator stub for lrm.c.
 *
 * Phase 2 cleanup state: the original iris.c (1797 lines) routed Flux and
 * Z-Image diffusion pipelines. All that code is gone with the fork. This
 * file is now an intentionally tiny stub that will grow back to dispatch
 * LRM model kinds (TripoSR first) in Phase 3+ when the lrm/ subdirectory
 * lands.
 *
 * For now it just holds the iris_get_error() implementation used by the
 * image I/O code.
 */

#include "iris.h"

#include <stdio.h>
#include <string.h>

/* Last-error scratch buffer. Single-threaded by contract; concurrency is
 * the caller's responsibility (see LRMengine.md section 12.6). */
static char iris_last_error[512] = "";

void iris_set_error(const char *msg) {
    if (msg == NULL) {
        iris_last_error[0] = '\0';
        return;
    }
    /* Copy and ensure NUL-termination even on truncation. */
    size_t n = sizeof(iris_last_error) - 1;
    strncpy(iris_last_error, msg, n);
    iris_last_error[n] = '\0';
}

const char *iris_get_error(void) {
    return iris_last_error;
}
