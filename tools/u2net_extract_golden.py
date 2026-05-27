#!/usr/bin/env python3
"""
U2Net golden extractor (parity reference for lrm/lrm_u2net.c).

Loads the U2Net salient-object net (xuebinqin/U-2-Net, Apache-2.0), runs it on
an image with rembg-style preprocessing (resize 320x320, divide by max pixel,
per-channel (x-mean)/std), and dumps the parity goldens:

  tests/golden/triposr/u2net_input.bin   [3,320,320] f32  net input (CHW)
  tests/golden/triposr/u2net_mask.bin    [320,320]   f32  sigmoid(d0)

plus /tmp/u2net_mask.png (the min-max-normalized mask) and
/tmp/u2net_cutout.png (RGBA cutout) for eyeballing.

The C net (fed u2net_input.bin) must reproduce u2net_mask.bin.

Expects the Apache model definition + weights under triposr_env/u2net_ref/:
  u2net.py    (from raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/)
  u2net.pth   (176 MB checkpoint)

Usage:
    triposr_env/.venv/bin/python tools/u2net_extract_golden.py [image]
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

REF = "triposr_env/u2net_ref"
GOLD = "tests/golden/triposr"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 320

sys.path.insert(0, REF)
from u2net import U2NET  # noqa: E402


def preprocess(path):
    im = Image.open(path).convert("RGB")
    orig = np.array(im)
    r = im.resize((SIZE, SIZE), Image.BILINEAR)
    a = np.array(r).astype(np.float32)
    a = a / a.max()                          # rembg: divide by max pixel
    chw = ((a - MEAN) / STD).transpose(2, 0, 1).copy()
    return chw, orig, im.size


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 \
        else "triposr_env/examples/captured.jpeg"
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(f"{REF}/u2net.pth", map_location="cpu"))
    net.eval()

    chw, orig, (W, H) = preprocess(img_path)
    with torch.no_grad():
        d0 = net(torch.from_numpy(chw[None]))[0]   # sigmoid(d0)
    mask = d0[0, 0].numpy().astype(np.float32)

    os.makedirs(GOLD, exist_ok=True)
    chw.astype(np.float32).tofile(f"{GOLD}/u2net_input.bin")
    mask.tofile(f"{GOLD}/u2net_mask.bin")
    print(f"input {chw.shape} range[{chw.min():.3f},{chw.max():.3f}]")
    print(f"mask  {mask.shape} range[{mask.min():.4f},{mask.max():.4f}] mean={mask.mean():.4f}")
    print(f"-> {GOLD}/u2net_input.bin, {GOLD}/u2net_mask.bin")

    norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    m = Image.fromarray((norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    m.save("/tmp/u2net_mask.png")
    Image.fromarray(np.dstack([orig, np.array(m)[..., None]]).astype(np.uint8),
                    "RGBA").save("/tmp/u2net_cutout.png")
    print(f"image {img_path} {W}x{H} -> /tmp/u2net_mask.png /tmp/u2net_cutout.png")


if __name__ == "__main__":
    main()
