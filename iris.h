/*
 * iris.h - Generic infrastructure header for lrm.c.
 *
 * This header used to expose the iris.c image-generation API (Flux/Z-Image).
 * Following the fork to LRM-family models, this surface is reduced to just
 * the pieces that are useful as engine infrastructure for ANY image-input
 * model: an RGB image struct, image I/O helpers, and the opaque iris_ctx
 * type. The actual model APIs live under lrm/ (lrm.h).
 *
 * Phase 2 cleanup state: this is a deliberately minimal stub. The opaque
 * iris_ctx will be reintroduced when the LRM coordinator lands in Phase 3+.
 */

#ifndef IRIS_H
#define IRIS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Opaque types
 * ======================================================================== */

typedef struct iris_ctx iris_ctx;

/* ========================================================================
 * Image
 * ======================================================================== */

typedef struct iris_image {
    int width;
    int height;
    int channels;       /* 3 for RGB, 4 for RGBA */
    uint8_t *data;      /* row-major, channel-interleaved */
} iris_image;

/*
 * Load image from file. Supports PNG and PPM (and JPEG via jpeg.h).
 * Returns NULL on error.
 */
iris_image *iris_image_load(const char *path);

/*
 * Save image to file. Format is selected by extension (.png, .ppm).
 * Returns 0 on success, -1 on error.
 */
int iris_image_save(const iris_image *img, const char *path);

/*
 * Create a new image with the given dimensions.
 */
iris_image *iris_image_create(int width, int height, int channels);

/*
 * Free image memory (safe on NULL).
 */
void iris_image_free(iris_image *img);

/*
 * Resize image using bilinear interpolation.
 */
iris_image *iris_image_resize(const iris_image *img, int new_width, int new_height);

/* ========================================================================
 * Diagnostics
 * ======================================================================== */

/*
 * Return the most recent error message set by any iris_* call. Static
 * storage, do not free.
 */
const char *iris_get_error(void);

/*
 * Set the last-error string. Internal helper for engine code; callers do
 * not need this. Pass NULL to clear the slot. The message is copied into
 * a static buffer (~512 bytes) and truncated if longer.
 */
void iris_set_error(const char *msg);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_H */
