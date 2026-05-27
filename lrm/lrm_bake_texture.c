/*
 * lrm_bake_texture.c - area-weighted per-triangle atlas + scanline raster.
 *
 * Algorithm:
 *   1. UV layout: each triangle gets its own square cell whose side is
 *      proportional to sqrt(its 3D surface area), so texel density is roughly
 *      uniform across the surface (large triangles are no longer
 *      under-sampled). Cells are shelf-packed (next-fit decreasing); a binary
 *      search on the area->side scale picks the largest cells that fit the
 *      atlas (pack_faces). Within each cell the 3 vertex UVs map to a
 *      right-triangle inscribed with a `padding` texel inset, so:
 *        v0 -> (cx*cs + p, (cy+1)*cs - 1 - p)     bottom-left
 *        v1 -> ((cx+1)*cs - 1 - p, (cy+1)*cs - 1 - p)   bottom-right
 *        v2 -> (cx*cs + p, cy*cs + p)             top-left
 *      Both texel and UV are reported in (u=x, v=y) order with V going
 *      DOWN, matching glTF convention.
 *
 *   2. Rasterize each triangle in texel space using a half-plane edge
 *      function. For each texel inside the triangle, compute barycentric
 *      coords (a, b, c) and the world-space position
 *        P = a * P0 + b * P1 + c * P2
 *      Stash (texel_index, P) in a chunk buffer.
 *
 *   3. When the chunk fills (or after the last triangle), pump the
 *      collected positions through lrm_triplane_sample + lrm_nerf_mlp
 *      to get RGB colors. Quantize to u8 and write back at the recorded
 *      texel offsets.
 *
 * Memory:
 *   - texture: tex_res^2 * 4 bytes (16 MB at 2048)
 *   - chunk:   chunk_size * (4 + 12) ~= 130 KB at chunk=8192
 *   - sample/MLP workspaces: same as the rest of lrm_infer.
 *
 * Quality notes:
 *   - The inscribed-right-triangle layout fills ~50 % of each cell.
 *   - 2-pixel padding around each triangle prevents bilinear sampling
 *     into the adjacent cell. With tex_res=2048 and ~9000 triangles
 *     (cell_size ~21), padding=2 leaves ~17x17 usable pixels and ~145
 *     valid texels per triangle - roughly 200x more color samples than
 *     plain vertex colors.
 */

#include "lrm_bake_texture.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"
#include "iris_kernels.h"

/* ---- Helpers --------------------------------------------------------- */

static int set_err(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    iris_set_error(buf);
    return -1;
}

static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }

static inline uint8_t f32_to_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (uint8_t)(v * 255.0f + 0.5f);
}

/* Edge-function (signed twice-area of the triangle through p, b, c). */
static inline float edge_fn(float px, float py,
                            float bx, float by,
                            float cx, float cy) {
    return (cx - bx) * (py - by) - (cy - by) * (px - bx);
}

/* ---- Chunk batching -------------------------------------------------- */

typedef struct {
    int          cap;            /* chunk_size */
    int          n;              /* current fill */
    int         *texel_off;      /* y * W + x for each texel */
    float       *positions;      /* [cap, 3] world coords */
    float       *features;       /* [cap, 120] triplane sample output */
    float       *density;        /* [cap]      MLP density output (discarded) */
    float       *color;          /* [cap, 3]   MLP color output */
    float       *ts_work;        /* triplane_sample workspace */
    float       *nm_work;        /* nerf_mlp workspace */
} chunk_t;

static void chunk_free(chunk_t *c) {
    free(c->texel_off); free(c->positions);
    free(c->features);  free(c->density); free(c->color);
    free(c->ts_work);   free(c->nm_work);
    memset(c, 0, sizeof(*c));
}

static int chunk_init(chunk_t *c, int cap,
                      const lrm_triplane_sample_cfg *ts_cfg,
                      const lrm_nerf_mlp *mlp) {
    memset(c, 0, sizeof(*c));
    c->cap        = cap;
    c->texel_off  = (int *)  malloc((size_t)cap * sizeof(int));
    c->positions  = (float *)malloc((size_t)cap * 3 * sizeof(float));
    c->features   = (float *)malloc((size_t)cap * 120 * sizeof(float));
    c->density    = (float *)malloc((size_t)cap * sizeof(float));
    c->color      = (float *)malloc((size_t)cap * 3 * sizeof(float));
    size_t ts_n = lrm_triplane_sample_workspace_bytes(ts_cfg, cap);
    size_t nm_n = lrm_nerf_mlp_workspace_bytes(mlp, cap);
    c->ts_work    = (float *)malloc(ts_n);
    c->nm_work    = (float *)malloc(nm_n);
    if (!c->texel_off || !c->positions || !c->features
        || !c->density || !c->color || !c->ts_work || !c->nm_work) {
        chunk_free(c);
        return -1;
    }
    return 0;
}

/* Flush the current chunk: sample triplane, run NeRF MLP, write colors
 * back to the texture at the recorded texel offsets. */
static int chunk_flush(chunk_t *c,
                       const lrm_triplane_sample_cfg *ts_cfg,
                       const lrm_nerf_mlp *mlp,
                       const float *triplane,
                       uint8_t *tex_rgba) {
    if (c->n == 0) return 0;
    if (lrm_triplane_sample_forward(ts_cfg, triplane, c->positions,
                                    c->n, c->features, c->ts_work) != 0) {
        return -1;
    }
    if (lrm_nerf_mlp_forward(mlp, c->features, c->n,
                             c->density, c->color, c->nm_work) != 0) {
        return -1;
    }
    for (int i = 0; i < c->n; i++) {
        int off = c->texel_off[i];
        const float *rgb = c->color + (size_t)i * 3;
        uint8_t *p = tex_rgba + (size_t)off * 4;
        p[0] = f32_to_u8(rgb[0]);
        p[1] = f32_to_u8(rgb[1]);
        p[2] = f32_to_u8(rgb[2]);
        p[3] = 255;
    }
    c->n = 0;
    return 0;
}

/* ---- Area-weighted atlas packing ------------------------------------- */
/*
 * Each triangle gets its own square cell, but the cell side is proportional
 * to sqrt(triangle 3D area) instead of being uniform. This gives a roughly
 * constant texel density per unit surface area, so large triangles are no
 * longer under-sampled (blurry) while tiny ones waste texels. Cells are
 * placed with shelf (next-fit-decreasing) packing; a binary search on the
 * area->side scale picks the largest cells that still fit the atlas.
 */
typedef struct {
    int *x, *y, *s;   /* per-face cell origin (texels) and side (texels) */
} packing_t;

static void packing_free(packing_t *p) {
    free(p->x); free(p->y); free(p->s);
    memset(p, 0, sizeof(*p));
}

typedef struct { int idx; float sa; } face_sa_t;
static int cmp_sa_desc(const void *a, const void *b) {
    float da = ((const face_sa_t *)a)->sa, db = ((const face_sa_t *)b)->sa;
    return (da < db) - (da > db);   /* descending */
}

/* Shelf-pack cells sized side_i = max(min_side, k*sa + base). If `store`,
 * record origins into pk->x/y; otherwise just test feasibility. Returns 1 if
 * everything fits in W x H, else 0. */
static int shelf_pack(const face_sa_t *ord, int n, float k, int base,
                      int min_side, int W, int H, packing_t *pk, int store) {
    int x = 0, y = 0, row_h = 0;
    for (int t = 0; t < n; t++) {
        int i = ord[t].idx;
        int s = (int)(k * ord[t].sa) + base;
        if (s < min_side) s = min_side;
        if (s > W) return 0;
        if (x + s > W) { y += row_h; x = 0; row_h = 0; }  /* next shelf */
        if (y + s > H) return 0;
        if (store) { pk->x[i] = x; pk->y[i] = y; pk->s[i] = s; }
        x += s;
        if (s > row_h) row_h = s;
    }
    return 1;
}

static int pack_faces(const float *vertices, int nv, const int32_t *faces,
                      int nf, int W, int H, int pad, packing_t *pk) {
    const int base = 2 * pad + 1;       /* padding both sides + 1 */
    const int min_side = 2 * pad + 3;   /* leave >=2 usable texels */
    face_sa_t *ord = (face_sa_t *)malloc((size_t)nf * sizeof(face_sa_t));
    pk->x = (int *)malloc((size_t)nf * sizeof(int));
    pk->y = (int *)malloc((size_t)nf * sizeof(int));
    pk->s = (int *)malloc((size_t)nf * sizeof(int));
    if (!ord || !pk->x || !pk->y || !pk->s) { free(ord); packing_free(pk); return -1; }

    float max_sa = 0.0f;
    for (int i = 0; i < nf; i++) {
        int a = faces[i*3+0], b = faces[i*3+1], c = faces[i*3+2];
        float sa = 0.0f;
        if (a >= 0 && b >= 0 && c >= 0 && a < nv && b < nv && c < nv) {
            const float *A = vertices + (size_t)a*3, *B = vertices + (size_t)b*3,
                        *C = vertices + (size_t)c*3;
            float e1x=B[0]-A[0], e1y=B[1]-A[1], e1z=B[2]-A[2];
            float e2x=C[0]-A[0], e2y=C[1]-A[1], e2z=C[2]-A[2];
            float cx=e1y*e2z-e1z*e2y, cy=e1z*e2x-e1x*e2z, cz=e1x*e2y-e1y*e2x;
            sa = sqrtf(0.5f * sqrtf(cx*cx + cy*cy + cz*cz)); /* sqrt(area) */
        }
        ord[i].idx = i; ord[i].sa = sa;
        if (sa > max_sa) max_sa = sa;
    }
    qsort(ord, nf, sizeof(face_sa_t), cmp_sa_desc);

    /* Binary search the largest scale k for which the shelf pack fits. */
    float lo = 0.0f, hi = (max_sa > 1e-9f) ? ((float)(W - base) / max_sa) : 1.0f;
    if (hi < 0.0f) hi = 0.0f;
    if (!shelf_pack(ord, nf, lo, base, min_side, W, H, pk, 0)) {
        free(ord); packing_free(pk);
        return set_err("lrm_bake_texture: too many faces (%d) for tex_res=%d",
                       nf, W);   /* even minimum cells don't fit */
    }
    for (int it = 0; it < 40; it++) {
        float mid = 0.5f * (lo + hi);
        if (shelf_pack(ord, nf, mid, base, min_side, W, H, pk, 0)) lo = mid;
        else hi = mid;
    }
    shelf_pack(ord, nf, lo, base, min_side, W, H, pk, 1);  /* final store */

    if (getenv("LRM_TIMING")) {
        long long used = 0;
        for (int i = 0; i < nf; i++) used += (long long)pk->s[i] * pk->s[i];
        fprintf(stderr, "lrmc:   (bake atlas: %d cells, scale=%.2f, "
                "cell-square fill=%.1f%%)\n", nf, (double)lo,
                100.0 * (double)used / ((double)W * H));
    }
    free(ord);
    return 0;
}

/* ---- Main entry ------------------------------------------------------ */

int lrm_bake_texture(const lrm_triplane_sample_cfg *ts_cfg,
                     const lrm_nerf_mlp *mlp,
                     const float *triplane,
                     const float *vertices, int n_vertices,
                     const int32_t *faces,  int n_faces,
                     const lrm_bake_cfg *cfg_in,
                     float **out_uvs,
                     uint8_t **out_tex_rgba) {
    if (!ts_cfg || !mlp || !triplane || !vertices || !faces
        || !out_uvs || !out_tex_rgba) {
        return set_err("lrm_bake_texture: NULL argument");
    }
    if (n_faces <= 0 || n_vertices <= 0) {
        return set_err("lrm_bake_texture: empty mesh (Nv=%d Nf=%d)",
                       n_vertices, n_faces);
    }

    lrm_bake_cfg cfg = LRM_BAKE_CFG_DEFAULT;
    if (cfg_in) cfg = *cfg_in;
    if (cfg.texture_resolution < 32) {
        return set_err("lrm_bake_texture: texture_resolution=%d too small",
                       cfg.texture_resolution);
    }
    if (cfg.chunk_size < 256) cfg.chunk_size = 256;
    if (cfg.padding < 1) cfg.padding = 1;

    const int W = cfg.texture_resolution;
    const int H = cfg.texture_resolution;
    const int pad = cfg.padding;

    /* Allocate outputs. */
    uint8_t *tex = (uint8_t *)malloc((size_t)W * H * 4);
    if (!tex) return set_err("lrm_bake_texture: oom for texture buffer");
    /* Fill with background color. */
    for (int i = 0; i < W * H; i++) {
        tex[i * 4 + 0] = cfg.bg_r;
        tex[i * 4 + 1] = cfg.bg_g;
        tex[i * 4 + 2] = cfg.bg_b;
        tex[i * 4 + 3] = cfg.bg_a;
    }
    float *uvs = (float *)malloc((size_t)n_faces * 3 * 2 * sizeof(float));
    if (!uvs) { free(tex); return set_err("lrm_bake_texture: oom for UVs"); }

    /* Area-weighted atlas packing: cell side proportional to sqrt(3D area),
     * shelf-packed (replaces the old uniform S x S grid). */
    packing_t pk = {0};
    if (pack_faces(vertices, n_vertices, faces, n_faces, W, H, pad, &pk) != 0) {
        free(tex); free(uvs);
        return -1;   /* error already set */
    }

    /* Compute per-face UVs (3 per face) from the packed cells. */
    const float inv_W = 1.0f / (float)W;
    const float inv_H = 1.0f / (float)H;
    for (int i = 0; i < n_faces; i++) {
        int x0 = pk.x[i] + pad;                  /* left  */
        int x1 = pk.x[i] + pk.s[i] - 1 - pad;    /* right */
        int y0 = pk.y[i] + pad;                  /* top   */
        int y1 = pk.y[i] + pk.s[i] - 1 - pad;    /* bottom*/
        /* v0 = bottom-left, v1 = bottom-right, v2 = top-left
         * (texel positions in pixel space, with V going down). */
        float u_v0 = (float)x0 * inv_W, v_v0 = (float)y1 * inv_H;
        float u_v1 = (float)x1 * inv_W, v_v1 = (float)y1 * inv_H;
        float u_v2 = (float)x0 * inv_W, v_v2 = (float)y0 * inv_H;
        float *u = uvs + (size_t)i * 6;
        u[0] = u_v0; u[1] = v_v0;
        u[2] = u_v1; u[3] = v_v1;
        u[4] = u_v2; u[5] = v_v2;
    }

    /* Rasterize each triangle: scanline edge-function inside the cell,
     * accumulate into the chunk batcher, flush as it fills. */
    chunk_t chunk;
    if (chunk_init(&chunk, cfg.chunk_size, ts_cfg, mlp) != 0) {
        packing_free(&pk); free(tex); free(uvs);
        return set_err("lrm_bake_texture: oom for chunk scratch");
    }

    for (int fi = 0; fi < n_faces; fi++) {
        /* Texel-space vertex positions (the inscribed right triangle). */
        int x0 = pk.x[fi] + pad;
        int x1 = pk.x[fi] + pk.s[fi] - 1 - pad;
        int y0 = pk.y[fi] + pad;
        int y1 = pk.y[fi] + pk.s[fi] - 1 - pad;
        float p0x = (float)x0, p0y = (float)y1;  /* bottom-left  */
        float p1x = (float)x1, p1y = (float)y1;  /* bottom-right */
        float p2x = (float)x0, p2y = (float)y0;  /* top-left     */

        /* World-space vertices of this triangle. */
        int i0 = faces[fi * 3 + 0];
        int i1 = faces[fi * 3 + 1];
        int i2 = faces[fi * 3 + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0
            || i0 >= n_vertices || i1 >= n_vertices || i2 >= n_vertices) {
            continue;
        }
        const float *V0 = vertices + (size_t)i0 * 3;
        const float *V1 = vertices + (size_t)i1 * 3;
        const float *V2 = vertices + (size_t)i2 * 3;

        float area = edge_fn(p0x, p0y, p1x, p1y, p2x, p2y);
        if (fabsf(area) < 1e-6f) continue;  /* degenerate */
        float inv_area = 1.0f / area;

        /* Bounding box in texel space, clipped to texture. */
        int bxmin = imax(0, imin(imin(x0, x1), x0));
        int bxmax = imin(W - 1, imax(imax(x0, x1), x0));
        int bymin = imax(0, imin(imin(y0, y1), y0));
        int bymax = imin(H - 1, imax(imax(y0, y1), y0));

        for (int py = bymin; py <= bymax; py++) {
            for (int px = bxmin; px <= bxmax; px++) {
                /* Pixel center: (px + 0.5, py + 0.5). */
                float fx = (float)px + 0.5f;
                float fy = (float)py + 0.5f;
                float w0 = edge_fn(fx, fy, p1x, p1y, p2x, p2y) * inv_area;
                float w1 = edge_fn(fx, fy, p2x, p2y, p0x, p0y) * inv_area;
                float w2 = edge_fn(fx, fy, p0x, p0y, p1x, p1y) * inv_area;
                /* Accept on/inside (>= 0). Numerical fuzz for borders. */
                if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f) continue;
                /* World position via barycentric interpolation. */
                float wx = w0 * V0[0] + w1 * V1[0] + w2 * V2[0];
                float wy = w0 * V0[1] + w1 * V1[1] + w2 * V2[1];
                float wz = w0 * V0[2] + w1 * V1[2] + w2 * V2[2];

                if (chunk.n == chunk.cap) {
                    if (chunk_flush(&chunk, ts_cfg, mlp, triplane, tex) != 0) {
                        chunk_free(&chunk); packing_free(&pk); free(tex); free(uvs);
                        return -1;
                    }
                }
                int idx = chunk.n++;
                chunk.texel_off[idx] = py * W + px;
                chunk.positions[idx * 3 + 0] = wx;
                chunk.positions[idx * 3 + 1] = wy;
                chunk.positions[idx * 3 + 2] = wz;
            }
        }
    }
    if (chunk_flush(&chunk, ts_cfg, mlp, triplane, tex) != 0) {
        chunk_free(&chunk); packing_free(&pk); free(tex); free(uvs);
        return -1;
    }

    chunk_free(&chunk);
    packing_free(&pk);
    *out_uvs = uvs;
    *out_tex_rgba = tex;
    return 0;
}
