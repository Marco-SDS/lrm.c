/*
 * lrm_u2net.h - U2Net salient-object segmentation for background removal.
 *
 * Ports U2Net (xuebinqin/U-2-Net, Apache-2.0), the model rembg uses by
 * default, into pure C on top of iris_kernels (conv2d w/ dilation, maxpool2d,
 * bilinear upsample, batch_norm, relu, sigmoid). The full 6-stage nested
 * U-structure (RSU7/6/5/4 + RSU4F blocks) runs single-pass on CPU.
 *
 * Used by Phase 16 background removal: an arbitrary RGB photo is segmented to
 * a soft foreground mask, which becomes the alpha channel fed to the existing
 * TripoSR preprocessing (alpha bbox + 85% canvas + gray composite).
 *
 * Weights: convert u2net.pth with tools/u2net_to_safetensors.py.
 */

#ifndef LRM_LRM_U2NET_H
#define LRM_LRM_U2NET_H

#include "iris_safetensors.h"

typedef struct lrm_u2net lrm_u2net;

/* Open a u2net.safetensors file. Returns NULL on error (iris_get_error set). */
lrm_u2net *lrm_u2net_load(const char *path);

/* Release the model and its mmap. Safe on NULL. */
void lrm_u2net_free(lrm_u2net *m);

/*
 * Run the net on a preprocessed, normalized CHW input [3, H, W] and write the
 * foreground probability mask sigmoid(d0) to out_mask [H, W] (row-major).
 * Caller owns both buffers. Returns 0 on success, -1 with iris_get_error().
 *
 * This expects the same preprocessing the reference uses (resize to HxW,
 * divide by max pixel, per-channel (x-mean)/std). For the end-to-end helper
 * that takes a raw image and returns an alpha mask, see lrm_u2net_alpha.
 */
int lrm_u2net_forward(const lrm_u2net *m, const float *input_chw,
                      int H, int W, float *out_mask);

/*
 * End-to-end foreground alpha from a raw RGB(A) image. Resizes to 320x320,
 * normalizes (rembg-style), runs the net, min-max normalizes the mask, and
 * resizes it back to the original WxH. out_alpha must hold W*H floats in
 * [0,1]. Returns 0 on success, -1 with iris_get_error().
 *
 *   rgb       : [H, W, channels] uint8 (channels 3 or 4; alpha ignored)
 */
int lrm_u2net_alpha(const lrm_u2net *m, const unsigned char *rgb,
                    int H, int W, int channels, float *out_alpha);

#endif /* LRM_LRM_U2NET_H */
