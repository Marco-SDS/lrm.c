/*
 * lrm_u2net.c - U2Net salient-object segmentation (xuebinqin/U-2-Net,
 * Apache-2.0) in pure C. See lrm_u2net.h.
 *
 * The net is a nested U-structure: six encoder stages (RSU7, RSU6, RSU5,
 * RSU4, RSU4F, RSU4F), five symmetric decoder stages, and six side outputs
 * fused by a 1x1 conv into d0. Each RSU is itself a small U-Net of
 * REBNCONV = conv3x3 (dilated) + BatchNorm + ReLU. We implement the RSU
 * generically (pooling variant + dilation-only "F" variant) so the whole
 * graph is a handful of calls.
 *
 * Buffers (fmap) are malloc'd per intermediate and freed as soon as they are
 * consumed, so the success path is leak-free; a mid-graph allocation failure
 * is treated as fatal (the caller aborts the run).
 */

#include "lrm_u2net.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"
#include "iris_kernels.h"

struct lrm_u2net {
    safetensors_file_t *st;
};

/* ---- feature map ---- */
typedef struct { float *d; int c, h, w; } fmap;

static int fmap_alloc(fmap *f, int c, int h, int w) {
    f->c = c; f->h = h; f->w = w;
    f->d = (float *)malloc((size_t)c * h * w * sizeof(float));
    return f->d ? 0 : -1;
}
static void fmap_free(fmap *f) { free(f->d); f->d = NULL; }

/* ---- weight lookup (zero-copy into the mmap) ---- */
static const float *wt(const lrm_u2net *m, const char *name) {
    const safetensor_t *t = safetensors_find(m->st, name);
    if (!t) {
        char e[256];
        snprintf(e, sizeof e, "u2net: missing tensor '%s'", name);
        iris_set_error(e);
        return NULL;
    }
    return (const float *)safetensors_data(m->st, t);
}

/* REBNCONV: conv3x3(pad=dirate, dil=dirate) + BatchNorm + ReLU.
 * out is allocated here (out_ch x in->h x in->w). */
static int rebnconv(const lrm_u2net *m, const char *prefix, int out_ch,
                    int dirate, const fmap *in, fmap *out) {
    char nm[256];
    const float *cw, *cb, *bw, *bb, *rm, *rv;
    snprintf(nm, sizeof nm, "%s.conv_s1.weight", prefix);       if (!(cw = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "%s.conv_s1.bias", prefix);         if (!(cb = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "%s.bn_s1.weight", prefix);         if (!(bw = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "%s.bn_s1.bias", prefix);           if (!(bb = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "%s.bn_s1.running_mean", prefix);   if (!(rm = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "%s.bn_s1.running_var", prefix);    if (!(rv = wt(m, nm))) return -1;
    if (fmap_alloc(out, out_ch, in->h, in->w) != 0) return -1;
    iris_conv2d(out->d, in->d, cw, cb, 1, in->c, out_ch, in->h, in->w,
                3, 3, /*stride=*/1, /*pad=*/dirate, /*dilation=*/dirate);
    iris_batch_norm(out->d, out->d, rm, rv, bw, bb, 1, out_ch,
                    in->h, in->w, 1e-5f);
    iris_relu(out->d, out_ch * in->h * in->w);
    return 0;
}

static int pool2(const fmap *in, fmap *out) {
    int oh = (in->h - 2) / 2 + 1; if ((in->h - 2) % 2) oh++;   /* ceil_mode */
    int ow = (in->w - 2) / 2 + 1; if ((in->w - 2) % 2) ow++;
    if (fmap_alloc(out, in->c, oh, ow) != 0) return -1;
    iris_maxpool2d(out->d, in->d, 1, in->c, in->h, in->w, 2, 2, 1, &oh, &ow);
    return 0;
}

static int up_like(const fmap *src, int tH, int tW, fmap *out) {
    if (fmap_alloc(out, src->c, tH, tW) != 0) return -1;
    iris_upsample_bilinear(out->d, src->d, 1, src->c, src->h, src->w, tH, tW);
    return 0;
}

/* Channel concat (a then b), same spatial size. */
static int fcat(const fmap *a, const fmap *b, fmap *out) {
    if (a->h != b->h || a->w != b->w) return -1;
    size_t hw = (size_t)a->h * a->w;
    if (fmap_alloc(out, a->c + b->c, a->h, a->w) != 0) return -1;
    memcpy(out->d, a->d, (size_t)a->c * hw * sizeof(float));
    memcpy(out->d + (size_t)a->c * hw, b->d, (size_t)b->c * hw * sizeof(float));
    return 0;
}

/* Generic pooling RSU of height L (RSU7/6/5/4). in_ch is taken from x->c. */
static int rsu(const lrm_u2net *m, const char *stage, int L, int mid,
               int out_ch, const fmap *x, fmap *y) {
    char p[256];
    fmap hxin = {0}, enc[8] = {{0}}, dec = {0};
    int hw = x->h * x->w;

    snprintf(p, sizeof p, "%s.rebnconvin", stage);
    if (rebnconv(m, p, out_ch, 1, x, &hxin) != 0) goto fail;

    /* encoder: enc[1] = rebnconv1(hxin); enc[k] = rebnconv_k(pool(enc[k-1])) */
    snprintf(p, sizeof p, "%s.rebnconv1", stage);
    if (rebnconv(m, p, mid, 1, &hxin, &enc[1]) != 0) goto fail;
    for (int k = 2; k <= L - 1; k++) {
        fmap pooled = {0};
        if (pool2(&enc[k - 1], &pooled) != 0) goto fail;
        snprintf(p, sizeof p, "%s.rebnconv%d", stage, k);
        int rc = rebnconv(m, p, mid, 1, &pooled, &enc[k]);
        fmap_free(&pooled);
        if (rc != 0) goto fail;
    }
    /* bottleneck enc[L] = rebnconvL(enc[L-1]), dilation 2, no pooling */
    snprintf(p, sizeof p, "%s.rebnconv%d", stage, L);
    if (rebnconv(m, p, mid, 2, &enc[L - 1], &enc[L]) != 0) goto fail;

    /* decoder: dec_{L-1} = rebnconv(L-1)d(cat(enc[L], enc[L-1])) */
    {
        fmap c = {0};
        if (fcat(&enc[L], &enc[L - 1], &c) != 0) goto fail;
        fmap_free(&enc[L]); fmap_free(&enc[L - 1]);
        snprintf(p, sizeof p, "%s.rebnconv%dd", stage, L - 1);
        int oc = (L - 1 == 1) ? out_ch : mid;
        int rc = rebnconv(m, p, oc, 1, &c, &dec);
        fmap_free(&c);
        if (rc != 0) goto fail;
    }
    for (int k = L - 2; k >= 1; k--) {
        fmap up = {0}, c = {0};
        if (up_like(&dec, enc[k].h, enc[k].w, &up) != 0) goto fail;
        fmap_free(&dec);
        if (fcat(&up, &enc[k], &c) != 0) { fmap_free(&up); goto fail; }
        fmap_free(&up); fmap_free(&enc[k]);
        snprintf(p, sizeof p, "%s.rebnconv%dd", stage, k);
        int oc = (k == 1) ? out_ch : mid;
        int rc = rebnconv(m, p, oc, 1, &c, &dec);
        fmap_free(&c);
        if (rc != 0) goto fail;
    }
    /* residual: dec += hxin */
    iris_add_inplace(dec.d, hxin.d, out_ch * hw);
    fmap_free(&hxin);
    *y = dec;
    return 0;

fail:
    fmap_free(&hxin); fmap_free(&dec);
    for (int k = 0; k < 8; k++) fmap_free(&enc[k]);
    return -1;
}

/* Dilation-only RSU4F (no pooling / upsampling). */
static int rsu4f(const lrm_u2net *m, const char *stage, int mid, int out_ch,
                 const fmap *x, fmap *y) {
    char p[256];
    fmap hxin = {0}, h1 = {0}, h2 = {0}, h3 = {0}, h4 = {0}, c = {0}, d = {0};
    int hw = x->h * x->w;
#define RC(call) do { if ((call) != 0) goto fail; } while (0)
    snprintf(p, sizeof p, "%s.rebnconvin", stage); RC(rebnconv(m, p, out_ch, 1, x, &hxin));
    snprintf(p, sizeof p, "%s.rebnconv1", stage);   RC(rebnconv(m, p, mid, 1, &hxin, &h1));
    snprintf(p, sizeof p, "%s.rebnconv2", stage);   RC(rebnconv(m, p, mid, 2, &h1, &h2));
    snprintf(p, sizeof p, "%s.rebnconv3", stage);   RC(rebnconv(m, p, mid, 4, &h2, &h3));
    snprintf(p, sizeof p, "%s.rebnconv4", stage);   RC(rebnconv(m, p, mid, 8, &h3, &h4));

    RC(fcat(&h4, &h3, &c)); fmap_free(&h4); fmap_free(&h3);
    snprintf(p, sizeof p, "%s.rebnconv3d", stage);  RC(rebnconv(m, p, mid, 4, &c, &d)); fmap_free(&c);
    RC(fcat(&d, &h2, &c)); fmap_free(&d); fmap_free(&h2);
    snprintf(p, sizeof p, "%s.rebnconv2d", stage);  RC(rebnconv(m, p, mid, 2, &c, &d)); fmap_free(&c);
    RC(fcat(&d, &h1, &c)); fmap_free(&d); fmap_free(&h1);
    snprintf(p, sizeof p, "%s.rebnconv1d", stage);  RC(rebnconv(m, p, out_ch, 1, &c, &d)); fmap_free(&c);
#undef RC
    iris_add_inplace(d.d, hxin.d, out_ch * hw);
    fmap_free(&hxin);
    *y = d;
    return 0;
fail:
    fmap_free(&hxin); fmap_free(&h1); fmap_free(&h2); fmap_free(&h3);
    fmap_free(&h4); fmap_free(&c); fmap_free(&d);
    return -1;
}

/* side conv (3x3 pad1 -> 1 channel) then upsample to (tH,tW). */
static int side(const lrm_u2net *m, int n, const fmap *in, int tH, int tW,
                fmap *out) {
    char nm[64];
    const float *cw, *cb;
    snprintf(nm, sizeof nm, "side%d.weight", n); if (!(cw = wt(m, nm))) return -1;
    snprintf(nm, sizeof nm, "side%d.bias", n);   if (!(cb = wt(m, nm))) return -1;
    fmap d = {0};
    if (fmap_alloc(&d, 1, in->h, in->w) != 0) return -1;
    iris_conv2d(d.d, in->d, cw, cb, 1, in->c, 1, in->h, in->w, 3, 3, 1, 1, 1);
    if (in->h == tH && in->w == tW) { *out = d; return 0; }
    int rc = up_like(&d, tH, tW, out);
    fmap_free(&d);
    return rc;
}

/* ======================================================================== */

lrm_u2net *lrm_u2net_load(const char *path) {
    lrm_u2net *m = (lrm_u2net *)calloc(1, sizeof(*m));
    if (!m) { iris_set_error("u2net: oom"); return NULL; }
    m->st = safetensors_open(path);
    if (!m->st) { free(m); return NULL; }  /* error already set */
    return m;
}

void lrm_u2net_free(lrm_u2net *m) {
    if (!m) return;
    if (m->st) safetensors_close(m->st);
    free(m);
}

int lrm_u2net_forward(const lrm_u2net *m, const float *input_chw,
                      int H, int W, float *out_mask) {
    if (!m || !input_chw || !out_mask || H < 8 || W < 8) {
        iris_set_error("u2net_forward: bad arguments");
        return -1;
    }
    /* Copy input into an owned fmap (the convs read it repeatedly). */
    fmap x = {0};
    if (fmap_alloc(&x, 3, H, W) != 0) { iris_set_error("u2net: oom input"); return -1; }
    memcpy(x.d, input_chw, (size_t)3 * H * W * sizeof(float));

    fmap hx1 = {0}, hx2 = {0}, hx3 = {0}, hx4 = {0}, hx5 = {0}, hx6 = {0};
    fmap d5 = {0}, d4 = {0}, d3 = {0}, d2 = {0}, d1 = {0};
    fmap pool = {0}, up = {0}, cat = {0};
    fmap s1 = {0}, s2 = {0}, s3 = {0}, s4 = {0}, s5 = {0}, s6 = {0}, d0 = {0};
    int rc = -1;
#define RC(call) do { if ((call) != 0) goto done; } while (0)

    /* ---- encoder ---- */
    RC(rsu(m, "stage1", 7, 32, 64, &x, &hx1));
    RC(pool2(&hx1, &pool));
    RC(rsu(m, "stage2", 6, 32, 128, &pool, &hx2)); fmap_free(&pool);
    RC(pool2(&hx2, &pool));
    RC(rsu(m, "stage3", 5, 64, 256, &pool, &hx3)); fmap_free(&pool);
    RC(pool2(&hx3, &pool));
    RC(rsu(m, "stage4", 4, 128, 512, &pool, &hx4)); fmap_free(&pool);
    RC(pool2(&hx4, &pool));
    RC(rsu4f(m, "stage5", 256, 512, &pool, &hx5)); fmap_free(&pool);
    RC(pool2(&hx5, &pool));
    RC(rsu4f(m, "stage6", 256, 512, &pool, &hx6)); fmap_free(&pool);

    /* ---- decoder ---- */
    RC(up_like(&hx6, hx5.h, hx5.w, &up));
    RC(fcat(&up, &hx5, &cat)); fmap_free(&up); fmap_free(&hx5);
    RC(rsu4f(m, "stage5d", 256, 512, &cat, &d5)); fmap_free(&cat);

    RC(up_like(&d5, hx4.h, hx4.w, &up));
    RC(fcat(&up, &hx4, &cat)); fmap_free(&up); fmap_free(&hx4);
    RC(rsu(m, "stage4d", 4, 128, 256, &cat, &d4)); fmap_free(&cat);

    RC(up_like(&d4, hx3.h, hx3.w, &up));
    RC(fcat(&up, &hx3, &cat)); fmap_free(&up); fmap_free(&hx3);
    RC(rsu(m, "stage3d", 5, 64, 128, &cat, &d3)); fmap_free(&cat);

    RC(up_like(&d3, hx2.h, hx2.w, &up));
    RC(fcat(&up, &hx2, &cat)); fmap_free(&up); fmap_free(&hx2);
    RC(rsu(m, "stage2d", 6, 32, 64, &cat, &d2)); fmap_free(&cat);

    RC(up_like(&d2, hx1.h, hx1.w, &up));
    RC(fcat(&up, &hx1, &cat)); fmap_free(&up); fmap_free(&hx1);
    RC(rsu(m, "stage1d", 7, 16, 64, &cat, &d1)); fmap_free(&cat);

    /* ---- side outputs fused by outconv (1x1, 6 -> 1) ---- */
    RC(side(m, 1, &d1, d1.h, d1.w, &s1));
    RC(side(m, 2, &d2, d1.h, d1.w, &s2));
    RC(side(m, 3, &d3, d1.h, d1.w, &s3));
    RC(side(m, 4, &d4, d1.h, d1.w, &s4));
    RC(side(m, 5, &d5, d1.h, d1.w, &s5));
    RC(side(m, 6, &hx6, d1.h, d1.w, &s6));
    fmap_free(&d1); fmap_free(&d2); fmap_free(&d3);
    fmap_free(&d4); fmap_free(&d5); fmap_free(&hx6);

    /* concat s1..s6 (6 channels) then 1x1 conv */
    {
        const float *ow = wt(m, "outconv.weight");
        const float *ob = wt(m, "outconv.bias");
        if (!ow || !ob) goto done;
        int hw = s1.h * s1.w;
        if (fmap_alloc(&cat, 6, s1.h, s1.w) != 0) goto done;
        memcpy(cat.d + 0 * hw, s1.d, hw * sizeof(float));
        memcpy(cat.d + 1 * hw, s2.d, hw * sizeof(float));
        memcpy(cat.d + 2 * hw, s3.d, hw * sizeof(float));
        memcpy(cat.d + 3 * hw, s4.d, hw * sizeof(float));
        memcpy(cat.d + 4 * hw, s5.d, hw * sizeof(float));
        memcpy(cat.d + 5 * hw, s6.d, hw * sizeof(float));
        if (fmap_alloc(&d0, 1, s1.h, s1.w) != 0) goto done;
        iris_conv2d(d0.d, cat.d, ow, ob, 1, 6, 1, s1.h, s1.w, 1, 1, 1, 0, 1);
        iris_sigmoid(d0.d, hw);
        /* d0 is at d1 resolution (= input resolution); copy out. */
        memcpy(out_mask, d0.d, (size_t)d0.h * d0.w * sizeof(float));
        rc = 0;
    }

done:
#undef RC
    fmap_free(&x);
    fmap_free(&hx1); fmap_free(&hx2); fmap_free(&hx3); fmap_free(&hx4);
    fmap_free(&hx5); fmap_free(&hx6);
    fmap_free(&d1); fmap_free(&d2); fmap_free(&d3); fmap_free(&d4); fmap_free(&d5);
    fmap_free(&pool); fmap_free(&up); fmap_free(&cat);
    fmap_free(&s1); fmap_free(&s2); fmap_free(&s3);
    fmap_free(&s4); fmap_free(&s5); fmap_free(&s6); fmap_free(&d0);
    return rc;
}

/* ---- end-to-end alpha from a raw image ---- */

#define U2NET_SIZE 320

static void resize_bilinear_rgb(const unsigned char *src, int H, int W, int C,
                                float *dst /* [SIZE,SIZE,3] */) {
    float sh = (float)H / U2NET_SIZE, sw = (float)W / U2NET_SIZE;
    for (int oy = 0; oy < U2NET_SIZE; oy++) {
        float sy = ((float)oy + 0.5f) * sh - 0.5f;
        int y0 = (int)floorf(sy); float dy = sy - y0; int y1 = y0 + 1;
        if (y0 < 0) y0 = 0; else if (y0 > H - 1) y0 = H - 1;
        if (y1 < 0) y1 = 0; else if (y1 > H - 1) y1 = H - 1;
        for (int ox = 0; ox < U2NET_SIZE; ox++) {
            float sx = ((float)ox + 0.5f) * sw - 0.5f;
            int x0 = (int)floorf(sx); float dx = sx - x0; int x1 = x0 + 1;
            if (x0 < 0) x0 = 0; else if (x0 > W - 1) x0 = W - 1;
            if (x1 < 0) x1 = 0; else if (x1 > W - 1) x1 = W - 1;
            for (int c = 0; c < 3; c++) {
                float v00 = src[((size_t)y0 * W + x0) * C + c];
                float v01 = src[((size_t)y0 * W + x1) * C + c];
                float v10 = src[((size_t)y1 * W + x0) * C + c];
                float v11 = src[((size_t)y1 * W + x1) * C + c];
                float top = v00 + dx * (v01 - v00);
                float bot = v10 + dx * (v11 - v10);
                dst[(oy * U2NET_SIZE + ox) * 3 + c] = top + dy * (bot - top);
            }
        }
    }
}

int lrm_u2net_alpha(const lrm_u2net *m, const unsigned char *rgb,
                    int H, int W, int channels, float *out_alpha) {
    if (!m || !rgb || !out_alpha || (channels != 3 && channels != 4)) {
        iris_set_error("u2net_alpha: bad arguments");
        return -1;
    }
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    float *rs = (float *)malloc((size_t)U2NET_SIZE * U2NET_SIZE * 3 * sizeof(float));
    float *chw = (float *)malloc((size_t)3 * U2NET_SIZE * U2NET_SIZE * sizeof(float));
    float *mask = (float *)malloc((size_t)U2NET_SIZE * U2NET_SIZE * sizeof(float));
    if (!rs || !chw || !mask) { free(rs); free(chw); free(mask);
        iris_set_error("u2net_alpha: oom"); return -1; }

    resize_bilinear_rgb(rgb, H, W, channels, rs);
    /* divide by max pixel (rembg), then per-channel normalize, to CHW */
    float mx = 0.0f;
    for (int i = 0; i < U2NET_SIZE * U2NET_SIZE * 3; i++) if (rs[i] > mx) mx = rs[i];
    if (mx <= 0.0f) mx = 1.0f;
    for (int y = 0; y < U2NET_SIZE; y++)
        for (int xx = 0; xx < U2NET_SIZE; xx++)
            for (int c = 0; c < 3; c++) {
                float v = rs[(y * U2NET_SIZE + xx) * 3 + c] / mx;
                chw[((size_t)c * U2NET_SIZE + y) * U2NET_SIZE + xx] =
                    (v - mean[c]) / std[c];
            }

    if (lrm_u2net_forward(m, chw, U2NET_SIZE, U2NET_SIZE, mask) != 0) {
        free(rs); free(chw); free(mask); return -1;
    }

    /* min-max normalize (rembg) then bilinear resize 320 -> WxH */
    float mn = mask[0], mxx = mask[0];
    for (int i = 1; i < U2NET_SIZE * U2NET_SIZE; i++) {
        if (mask[i] < mn) mn = mask[i];
        if (mask[i] > mxx) mxx = mask[i];
    }
    float inv = (mxx - mn > 1e-8f) ? 1.0f / (mxx - mn) : 1.0f;
    for (int i = 0; i < U2NET_SIZE * U2NET_SIZE; i++) mask[i] = (mask[i] - mn) * inv;

    iris_upsample_bilinear(out_alpha, mask, 1, 1, U2NET_SIZE, U2NET_SIZE, H, W);
    for (int i = 0; i < H * W; i++) {
        if (out_alpha[i] < 0.0f) out_alpha[i] = 0.0f;
        else if (out_alpha[i] > 1.0f) out_alpha[i] = 1.0f;
    }
    free(rs); free(chw); free(mask);
    return 0;
}
