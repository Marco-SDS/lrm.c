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

#include <math.h>
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
    free(m->uvs);
    free(m->texture_png);
    free(m);
}

void lrm_mesh_set_texture(lrm_mesh *m, float *uvs,
                          uint8_t *texture_png, size_t texture_png_size) {
    if (!m) return;
    free(m->uvs);
    free(m->texture_png);
    m->uvs              = uvs;
    m->texture_png      = texture_png;
    m->texture_png_size = texture_png_size;
}

/* ========================================================================
 * GLB writer
 * ======================================================================== */

/* glTF 2.0 magic numbers used in this file. */
#define GLTF_MAGIC          0x46546C67u  /* 'glTF' little-endian */
#define GLTF_VERSION        2u
#define GLTF_CHUNK_JSON     0x4E4F534Au  /* 'JSON' */
#define GLTF_CHUNK_BIN      0x004E4942u  /* 'BIN\0' */
#define GLTF_COMP_U8        5121
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

/* ------------------------------------------------------------------ */
/* Compute per-vertex normals by area-weighted face-normal averaging.
 * Each face contributes its (unnormalized) normal to each of its three
 * vertices; we then normalize per vertex. This matches what trimesh does
 * by default when normals aren't already present. */
static int compute_vertex_normals(const float *vertices, int Nv,
                                  const int32_t *faces, int Nf,
                                  float *out_normals /* [Nv*3] */) {
    memset(out_normals, 0, (size_t)Nv * 3 * sizeof(float));
    for (int f = 0; f < Nf; f++) {
        int i0 = faces[f * 3 + 0];
        int i1 = faces[f * 3 + 1];
        int i2 = faces[f * 3 + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0
            || i0 >= Nv || i1 >= Nv || i2 >= Nv) {
            continue; /* skip degenerate / out-of-range; should not happen */
        }
        const float *p0 = vertices + (size_t)i0 * 3;
        const float *p1 = vertices + (size_t)i1 * 3;
        const float *p2 = vertices + (size_t)i2 * 3;
        float ax = p1[0]-p0[0], ay = p1[1]-p0[1], az = p1[2]-p0[2];
        float bx = p2[0]-p0[0], by = p2[1]-p0[1], bz = p2[2]-p0[2];
        /* Cross product magnitude = 2 * triangle area, which gives area
         * weighting "for free" in the accumulation. */
        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;
        out_normals[i0*3+0] += nx; out_normals[i0*3+1] += ny; out_normals[i0*3+2] += nz;
        out_normals[i1*3+0] += nx; out_normals[i1*3+1] += ny; out_normals[i1*3+2] += nz;
        out_normals[i2*3+0] += nx; out_normals[i2*3+1] += ny; out_normals[i2*3+2] += nz;
    }
    for (int v = 0; v < Nv; v++) {
        float nx = out_normals[v*3+0];
        float ny = out_normals[v*3+1];
        float nz = out_normals[v*3+2];
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) {
            float inv = 1.0f / len;
            out_normals[v*3+0] = nx * inv;
            out_normals[v*3+1] = ny * inv;
            out_normals[v*3+2] = nz * inv;
        } else {
            /* Vertex with no triangles (shouldn't happen for MC output) -
             * fall back to +Z so the GLB stays valid. */
            out_normals[v*3+0] = 0.0f;
            out_normals[v*3+1] = 0.0f;
            out_normals[v*3+2] = 1.0f;
        }
    }
    return 0;
}

/* Clamp+quantize an f32 channel in [0, 1] to u8 [0, 255]. */
static inline uint8_t f32_to_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    /* Standard glTF rule for "normalized" component: round-half-to-even
     * is not strictly required, but match what trimesh does (just *255
     * with implicit truncation gives slightly different results). The
     * +0.5 rounding is what PIL.Image.fromarray((arr*255).astype(uint8))
     * effectively does after `arr.clip(0, 1)`. */
    return (uint8_t)(v * 255.0f + 0.5f);
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
    const int  has_color   = (mesh->vertex_colors != NULL);
    const int  has_texture = (mesh->uvs != NULL && mesh->texture_png != NULL);

    /* Pick index componentType. Most LRM meshes at 64^3 have a few
     * thousand verts so u16 suffices; at 256^3 we may need u32. With
     * --bake-texture vertex count = 3*Nf which can overflow u16 quickly. */
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

    /* Compute per-vertex normals (required for material shading). */
    float *normals = (float *)malloc((size_t)Nv * 3 * sizeof(float));
    if (!normals) {
        iris_set_error("lrm_mesh_save_glb: oom for normals");
        return -1;
    }
    compute_vertex_normals(mesh->vertices, Nv, mesh->faces, Nf, normals);

    /* Convert vertex colors to u8 normalized RGBA. Smaller (4x) and
     * universally readable by viewers; matches trimesh's exported GLB. */
    uint8_t *colors_u8 = NULL;
    if (has_color) {
        colors_u8 = (uint8_t *)malloc((size_t)Nv * 4 * sizeof(uint8_t));
        if (!colors_u8) {
            free(normals);
            iris_set_error("lrm_mesh_save_glb: oom for color quantization");
            return -1;
        }
        for (int i = 0; i < Nv * 4; i++) {
            colors_u8[i] = f32_to_u8(mesh->vertex_colors[i]);
        }
    }

    /* Build the binary chunk first so we know its layout, then JSON
     * references it via byteOffset / byteLength.
     *
     * Layout: POSITION | NORMAL | COLOR_0 (u8, padded) | indices (padded).
     */
    bytebuf bin = {0};

    /* POSITION */
    const size_t pos_off = bin.len;
    const size_t pos_len = (size_t)Nv * 3 * sizeof(float);
    if (bb_write(&bin, mesh->vertices, pos_len) != 0) goto oom;

    /* NORMAL */
    if (bb_pad_to_4(&bin, 0) != 0) goto oom;
    const size_t nor_off = bin.len;
    const size_t nor_len = (size_t)Nv * 3 * sizeof(float);
    if (bb_write(&bin, normals, nor_len) != 0) goto oom;

    /* COLOR_0 (u8 normalized RGBA) */
    size_t col_off = 0, col_len = 0;
    if (has_color) {
        if (bb_pad_to_4(&bin, 0) != 0) goto oom;
        col_off = bin.len;
        col_len = (size_t)Nv * 4 * sizeof(uint8_t);
        if (bb_write(&bin, colors_u8, col_len) != 0) goto oom;
    }

    /* TEXCOORD_0 (f32 VEC2) */
    size_t tex_off = 0, tex_len = 0;
    if (has_texture) {
        if (bb_pad_to_4(&bin, 0) != 0) goto oom;
        tex_off = bin.len;
        tex_len = (size_t)Nv * 2 * sizeof(float);
        if (bb_write(&bin, mesh->uvs, tex_len) != 0) goto oom;
    }

    /* indices */
    if (bb_pad_to_4(&bin, 0) != 0) goto oom;
    const size_t idx_off = bin.len;
    const size_t idx_len = (size_t)Ni * idx_elem;
    if (use_u32) {
        if (bb_write(&bin, mesh->faces, idx_len) != 0) goto oom;
    } else {
        uint16_t *tmp = (uint16_t *)malloc(idx_len);
        if (!tmp) goto oom;
        for (int i = 0; i < Ni; i++) {
            int32_t v = mesh->faces[i];
            if (v < 0 || v > 65535) {
                free(tmp); free(normals); free(colors_u8);
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
    /* PNG texture bytes (if any). Stored after indices; image bufferView
     * has no `target` (per spec, only accessor-backed bufferViews do). */
    size_t img_off = 0, img_len = 0;
    if (has_texture) {
        if (bb_pad_to_4(&bin, 0) != 0) goto oom;
        img_off = bin.len;
        img_len = mesh->texture_png_size;
        if (bb_write(&bin, mesh->texture_png, img_len) != 0) goto oom;
    }

    /* glTF spec requires both chunks be 4-byte aligned in the file. */
    if (bb_pad_to_4(&bin, 0) != 0) goto oom;

    /* Done with the per-vertex scratch. */
    free(normals);   normals   = NULL;
    free(colors_u8); colors_u8 = NULL;

    /* ----- Build the JSON chunk. Accessor indices (running counter):
     *   0  POSITION   (always)
     *   1  NORMAL     (always)
     *   2  COLOR_0    (only if has_color)
     *   .  TEXCOORD_0 (only if has_texture)
     *   N  indices    (last)
     *
     * BufferView indices follow the same order, plus one EXTRA entry
     * at the end for the image PNG bytes (only if has_texture).
     */
    int next_acc = 2;
    const int acc_pos = 0;
    const int acc_nor = 1;
    const int acc_col = has_color   ? next_acc++ : -1;
    const int acc_tex = has_texture ? next_acc++ : -1;
    const int acc_idx = next_acc;

    int next_bv = 2;
    const int bv_pos = 0;
    const int bv_nor = 1;
    const int bv_col = has_color   ? next_bv++ : -1;
    const int bv_tex = has_texture ? next_bv++ : -1;
    const int bv_idx = next_bv++;
    const int bv_img = has_texture ? next_bv : -1;

    bytebuf js = {0};

    /* Asset + top-level scene/nodes. */
    if (bb_printf(&js,
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"lrm.c\"},"
        "\"scene\":0,"
        "\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],") != 0) goto oom_js;

    /* Material: PBR matte. With a texture present we ALSO set
     * baseColorTexture; viewers multiply the texture into the
     * baseColorFactor (and any COLOR_0) per spec. */
    if (bb_printf(&js,
        "\"materials\":[{"
        "\"name\":\"lrm_default\","
        "\"pbrMetallicRoughness\":{"
        "\"baseColorFactor\":[1,1,1,1],"
        "\"metallicFactor\":0,"
        "\"roughnessFactor\":1") != 0) goto oom_js;
    if (has_texture) {
        if (bb_printf(&js,
            ",\"baseColorTexture\":{\"index\":0,\"texCoord\":0}") != 0) goto oom_js;
    }
    if (bb_printf(&js,
        "},\"doubleSided\":true}],") != 0) goto oom_js;

    /* Textures + samplers + images (only if has_texture). */
    if (has_texture) {
        if (bb_printf(&js,
            "\"textures\":[{\"sampler\":0,\"source\":0}],"
            "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,"
            "\"wrapS\":33071,\"wrapT\":33071}],"
            /* 9729=LINEAR, 9987=LINEAR_MIPMAP_LINEAR, 33071=CLAMP_TO_EDGE */
            "\"images\":[{\"bufferView\":%d,\"mimeType\":\"image/png\"}],",
            bv_img) != 0) goto oom_js;
    }

    /* Mesh + primitive. Build the attributes map dynamically. */
    if (bb_printf(&js,
        "\"meshes\":[{\"primitives\":[{"
        "\"attributes\":{\"POSITION\":%d,\"NORMAL\":%d",
        acc_pos, acc_nor) != 0) goto oom_js;
    if (has_color)   { if (bb_printf(&js, ",\"COLOR_0\":%d", acc_col) != 0) goto oom_js; }
    if (has_texture) { if (bb_printf(&js, ",\"TEXCOORD_0\":%d", acc_tex) != 0) goto oom_js; }
    if (bb_printf(&js,
        "},\"indices\":%d,\"material\":0,\"mode\":%d}]}],",
        acc_idx, GLTF_MODE_TRIANGLES) != 0) goto oom_js;

    /* Accessors. */
    if (bb_printf(&js, "\"accessors\":[") != 0) goto oom_js;

    /* Accessor 0: POSITION. */
    if (bb_printf(&js,
        "{\"bufferView\":%d,\"componentType\":%d,\"count\":%d,\"type\":\"VEC3\","
        "\"min\":[%.8g,%.8g,%.8g],\"max\":[%.8g,%.8g,%.8g]}",
        bv_pos, GLTF_COMP_F32, Nv,
        (double)mn[0], (double)mn[1], (double)mn[2],
        (double)mx[0], (double)mx[1], (double)mx[2]) != 0) goto oom_js;
    /* Accessor 1: NORMAL. */
    if (bb_printf(&js,
        ",{\"bufferView\":%d,\"componentType\":%d,\"count\":%d,\"type\":\"VEC3\"}",
        bv_nor, GLTF_COMP_F32, Nv) != 0) goto oom_js;
    /* Accessor 2: COLOR_0 (u8 normalized) - only if present. */
    if (has_color) {
        if (bb_printf(&js,
            ",{\"bufferView\":%d,\"componentType\":%d,"
            "\"normalized\":true,\"count\":%d,\"type\":\"VEC4\"}",
            bv_col, GLTF_COMP_U8, Nv) != 0) goto oom_js;
    }
    /* Accessor for TEXCOORD_0 (f32 VEC2). */
    if (has_texture) {
        if (bb_printf(&js,
            ",{\"bufferView\":%d,\"componentType\":%d,\"count\":%d,\"type\":\"VEC2\"}",
            bv_tex, GLTF_COMP_F32, Nv) != 0) goto oom_js;
    }
    /* Last accessor: indices. */
    if (bb_printf(&js,
        ",{\"bufferView\":%d,\"componentType\":%d,\"count\":%d,\"type\":\"SCALAR\"}"
        "],",
        bv_idx, idx_comp, Ni) != 0) goto oom_js;

    /* bufferViews. */
    if (bb_printf(&js, "\"bufferViews\":[") != 0) goto oom_js;
    if (bb_printf(&js,
        "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
        pos_off, pos_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    if (bb_printf(&js,
        ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
        nor_off, nor_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    if (has_color) {
        if (bb_printf(&js,
            ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
            col_off, col_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    }
    if (has_texture) {
        if (bb_printf(&js,
            ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
            tex_off, tex_len, GLTF_TARGET_ARRAY) != 0) goto oom_js;
    }
    if (bb_printf(&js,
        ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}",
        idx_off, idx_len, GLTF_TARGET_INDEX) != 0) goto oom_js;
    /* Image bufferView has no `target` per spec. */
    if (has_texture) {
        if (bb_printf(&js,
            ",{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}",
            img_off, img_len) != 0) goto oom_js;
    }
    if (bb_printf(&js, "],") != 0) goto oom_js;

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
    /* normals / colors_u8 may still be live if we OOM'd while writing
     * the BIN chunk (they are freed and nulled after BIN completes). */
    free(normals);
    free(colors_u8);
    free(bin.data);
    iris_set_error("lrm_mesh_save_glb: out of memory");
    return -1;
}
