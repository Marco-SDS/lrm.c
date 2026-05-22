/*
 * lrm_mesh_export.c - mesh struct + binary glTF 2.0 writer.
 *
 * GLB layout (https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html):
 *
 *   [ 12-byte header: magic="glTF" | version=2 | total_length ]
 *   [ 4-byte chunk0_length | 4-byte chunk0_type="JSON" ]
 *   [ JSON UTF-8 bytes, padded with 0x20 (space) to a 4-byte boundary ]
 *   [ 4-byte chunk1_length | 4-byte chunk1_type="BIN\0" ]
 *   [ raw binary, padded with 0x00 to a 4-byte boundary ]
 *
 * We emit one primitive per mesh with three accessors:
 *   - POSITION       VEC3 f32  (required by spec to carry min/max)
 *   - COLOR_0        VEC4 f32  (optional; only if vertex_colors != NULL)
 *   - indices        SCALAR    (componentType picked based on N: u16 if
 *                               N <= 65535 else u32)
 */

#include "lrm_mesh_export.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"

/* ========================================================================
 * Mesh container
 * ======================================================================== */

lrm_mesh *lrm_mesh_from_buffers(int n_vertices, float *vertices,
                                int n_faces, int32_t *faces,
                                float *vertex_colors) {
    lrm_mesh *m = (lrm_mesh *)calloc(1, sizeof(lrm_mesh));
    if (!m) {
        iris_set_error("lrm_mesh: out of memory");
        return NULL;
    }
    m->n_vertices    = n_vertices;
    m->n_faces       = n_faces;
    m->vertices      = vertices;
    m->faces         = faces;
    m->vertex_colors = vertex_colors;
    return m;
}

void lrm_mesh_free(lrm_mesh *m) {
    if (!m) return;
    free(m->vertices);
    free(m->faces);
    free(m->vertex_colors);
    free(m);
}

/* ========================================================================
 * GLB writer
 * ======================================================================== */

/* glTF 2.0 magic numbers used in this file. */
#define GLTF_MAGIC          0x46546C67u  /* 'glTF' little-endian */
#define GLTF_VERSION        2u
#define GLTF_CHUNK_JSON     0x4E4F534Au  /* 'JSON' */
#define GLTF_CHUNK_BIN      0x004E4942u  /* 'BIN\0' */
#define GLTF_COMP_U16       5123
#define GLTF_COMP_U32       5125
#define GLTF_COMP_F32       5126
#define GLTF_TARGET_ARRAY   34962
#define GLTF_TARGET_INDEX   34963
#define GLTF_MODE_TRIANGLES 4

/* Append data to a growable byte buffer. */
typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} bytebuf;

static int bb_reserve(bytebuf *b, size_t need) {
    if (b->cap >= need) return 0;
    size_t cap = b->cap > 0 ? b->cap : 256;
    while (cap < need) cap *= 2;
    uint8_t *p = (uint8_t *)realloc(b->data, cap);
    if (!p) return -1;
    b->data = p;
    b->cap  = cap;
    return 0;
}
static int bb_write(bytebuf *b, const void *src, size_t n) {
    if (bb_reserve(b, b->len + n) != 0) return -1;
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}
static int bb_pad_to_4(bytebuf *b, uint8_t fill) {
    while (b->len & 3) {
        if (bb_reserve(b, b->len + 1) != 0) return -1;
        b->data[b->len++] = fill;
    }
    return 0;
}

/* Append a printf-formatted string (no terminator). */
static int bb_printf(bytebuf *b, const char *fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n >= sizeof(tmp)) {
        /* Larger printf, allocate. */
        size_t big = (size_t)n + 1;
        char *buf = (char *)malloc(big);
        if (!buf) return -1;
        va_start(ap, fmt);
        vsnprintf(buf, big, fmt, ap);
        va_end(ap);
        int r = bb_write(b, buf, big - 1);
        free(buf);
        return r;
    }
    return bb_write(b, tmp, (size_t)n);
}

int lrm_mesh_save_glb(const lrm_mesh *mesh, const char *path) {
    if (!mesh || !path) {
        iris_set_error("lrm_mesh_save_glb: NULL argument");
        return -1;
    }
    if (mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        iris_set_error("lrm_mesh_save_glb: empty mesh");
        return -1;
    }

    const int  Nv  = mesh->n_vertices;
    const int  Nf  = mesh->n_faces;
    const int  Ni  = Nf * 3;
    const int  has_color = (mesh->vertex_colors != NULL);

    /* Pick index componentType. Most LRM meshes at 64^3 have a few
     * thousand verts so u16 suffices; at 256^3 we may need u32. */
    const int  use_u32 = (Nv > 65535);
    const int  idx_comp = use_u32 ? GLTF_COMP_U32 : GLTF_COMP_U16;
    const size_t idx_elem = use_u32 ? 4 : 2;

    /* Compute POSITION min/max (required for the accessor). */
    float mn[3] = { mesh->vertices[0], mesh->vertices[1], mesh->vertices[2] };
    float mx[3] = { mn[0], mn[1], mn[2] };
    for (int i = 1; i < Nv; i++) {
        for (int k = 0; k < 3; k++) {
            float c = mesh->vertices[i * 3 + k];
            if (c < mn[k]) mn[k] = c;
            if (c > mx[k]) mx[k] = c;
        }
    }

    /* Build the binary chunk first so we know its layout, then JSON
     * references it via byteOffset / byteLength. */
    bytebuf bin = {0};
    /* POSITION */
    const size_t pos_off = bin.len;
    const size_t pos_len = (size_t)Nv * 3 * sizeof(float);
    if (bb_write(&bin, mesh->vertices, pos_len) != 0) goto oom;

    /* COLOR_0 (optional) */
    size_t col_off = 0, col_len = 0;
    if (has_color) {
        if (bb_pad_to_4(&bin, 0) != 0) goto oom;
        col_off = bin.len;
        col_len = (size_t)Nv * 4 * sizeof(float);
        if (bb_write(&bin, mesh->vertex_colors, col_len) != 0) goto oom;
    }

    /* indices */
    if (bb_pad_to_4(&bin, 0) != 0) goto oom;
    const size_t idx_off = bin.len;
    const size_t idx_len = (size_t)Ni * idx_elem;
    if (use_u32) {
        if (bb_write(&bin, mesh->faces, idx_len) != 0) goto oom;
    } else {
        /* Convert i32 -> u16 inline. */
        uint16_t *tmp = (uint16_t *)malloc(idx_len);
        if (!tmp) goto oom;
        for (int i = 0; i < Ni; i++) {
            int32_t v = mesh->faces[i];
            if (v < 0 || v > 65535) {
                free(tmp);
                iris_set_error("lrm_mesh_save_glb: index out of u16 range");
                free(bin.data);
                return -1;
            }
            tmp[i] = (uint16_t)v;
        }
        int rc = bb_write(&bin, tmp, idx_len);
        free(tmp);
        if (rc != 0) goto oom;
    }

    /* glTF spec requires both chunks be 4-byte aligned in the file;
     * pad the binary chunk too. */
    if (bb_pad_to_4(&bin, 0) != 0) goto oom;

    /* Build the JSON chunk. */
    bytebuf js = {0};
    if (bb_printf(&js,
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"lrm.c\"},"
        "\"scene\":0,"
        "\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{"
        "\"attributes\":{\"POSITION\":0%s},"
        "\"indices\":%d,"
        "\"mode\":%d"
        "}]}],"
        "\"accessors\":["
        /* 0: POSITION */
        "{\"bufferView\":0,\"componentType\":%d,\"count\":%d,\"type\":\"VEC3\","
        "\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]}",
        has_color ? ",\"COLOR_0\":1" : "",
        has_color ? 2 : 1,
        GLTF_MODE_TRIANGLES,
        GLTF_COMP_F32, Nv,
        (double)mn[0], (double)mn[1], (double)mn[2],
        (double)mx[0], (double)mx[1], (double)mx[2]) != 0) goto oom_js;

    if (has_color) {
        if (bb_printf(&js,
            ",{\"bufferView\":1,\"componentType\":%d,\"count\":%d,\"type\":\"VEC4\"}",
            GLTF_COMP_F32, Nv) != 0) goto oom_js;
    }
    if (bb_printf(&js,
        ",{\"bufferView\":%d,\"componentType\":%d,\"count\":%d,\"type\":\"SCALAR\"}"
        "],",
        has_color ? 2 : 1, idx_comp, Ni) != 0) goto oom_js;

    /* bufferViews. */
    if (bb_printf(&js, "\"bufferViews\":[") != 0) goto oom_js;
    if (bb_printf(&js,
        "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
        pos_off, pos_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    if (has_color) {
        if (bb_printf(&js,
            ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
            col_off, col_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    }
    if (bb_printf(&js,
        ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}"
        "],",
        idx_off, idx_len, GLTF_TARGET_INDEX) != 0) goto oom_js;

    /* The single buffer. */
    if (bb_printf(&js, "\"buffers\":[{\"byteLength\":%zu}]}", bin.len) != 0) goto oom_js;

    /* JSON chunk must be padded with 0x20 (space). */
    if (bb_pad_to_4(&js, 0x20) != 0) goto oom_js;

    /* Compute total file length. */
    const size_t header_len = 12;
    const size_t chunk_hdr  = 8;       /* 4 bytes len + 4 bytes type */
    const size_t total_len  = header_len + chunk_hdr + js.len
                            + chunk_hdr + bin.len;
    if (total_len > 0xFFFFFFFFul) {
        iris_set_error("lrm_mesh_save_glb: GLB exceeds 4 GB");
        free(js.data); free(bin.data);
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        char err[512];
        snprintf(err, sizeof(err), "lrm_mesh_save_glb: cannot open '%s'", path);
        iris_set_error(err);
        free(js.data); free(bin.data);
        return -1;
    }

#define W(buf, n) do { \
    if (fwrite((buf), 1, (n), fp) != (size_t)(n)) goto fail_io; \
} while (0)
#define W_U32(v) do { \
    uint32_t _v = (uint32_t)(v); W(&_v, 4); \
} while (0)

    /* GLB header. */
    W_U32(GLTF_MAGIC);
    W_U32(GLTF_VERSION);
    W_U32(total_len);

    /* JSON chunk. */
    W_U32(js.len);
    W_U32(GLTF_CHUNK_JSON);
    W(js.data, js.len);

    /* BIN chunk. */
    W_U32(bin.len);
    W_U32(GLTF_CHUNK_BIN);
    W(bin.data, bin.len);

#undef W
#undef W_U32

    if (fclose(fp) != 0) {
        iris_set_error("lrm_mesh_save_glb: fclose failed");
        free(js.data); free(bin.data);
        return -1;
    }
    free(js.data);
    free(bin.data);
    return 0;

fail_io:
    fclose(fp);
    iris_set_error("lrm_mesh_save_glb: write error");
    free(js.data); free(bin.data);
    return -1;

oom_js:
    free(js.data);
oom:
    free(bin.data);
    iris_set_error("lrm_mesh_save_glb: out of memory");
    return -1;
}
