#!/usr/bin/env python3
"""
debug_stage_compare.py - inspect / diff a TripoSR pipeline stage tensor.

The lrm.c parity tests (make test-dino / test-decoder / test-upsample /
test-density) only print pass/fail plus a summary. When a stage regresses you
usually want to know *where* it diverges - which token, channel, or spatial
region. This tool loads the raw float32 .bin dumps and localizes the
difference.

The golden .bin files are produced by tools/extract_golden.py (the PyTorch
reference); the "got" dump is whatever your C code wrote (add a one-off
fwrite of the stage output to a .bin while debugging).

Known stages (shape inferred automatically):

    input_512              [3, 512, 512]        preprocessed CHW image
    dino_tokens            [1025, 768]          DINO encoder output
    triplane_pre_upsample  [3, 1024, 32, 32]    decoder output
    triplane               [3, 40, 64, 64]      post-upsample triplane
    density                [64, 64, 64]         post-exp density (golden grid)
    color                  [64, 64, 64, 3]      post-sigmoid color

Usage:
    # inspect a golden (range / NaN / distribution / structure)
    python debug/debug_stage_compare.py --stage density

    # diff a C dump against the golden, localizing the worst divergences
    python debug/debug_stage_compare.py --stage triplane --got /tmp/triplane_c.bin

    # arbitrary files with an explicit shape
    python debug/debug_stage_compare.py --got a.bin --ref b.bin --shape 3,40,64,64
"""

import argparse
import os
import sys

import numpy as np

GOLD = "tests/golden/triposr"
STAGES = {
    "input_512":             (f"{GOLD}/input_512.bin",             (3, 512, 512)),
    "dino_tokens":           (f"{GOLD}/dino_tokens.bin",           (1025, 768)),
    "triplane_pre_upsample": (f"{GOLD}/triplane_pre_upsample.bin", (3, 1024, 32, 32)),
    "triplane":              (f"{GOLD}/triplane.bin",              (3, 40, 64, 64)),
    "density":               (f"{GOLD}/density.bin",               (64, 64, 64)),
    "color":                 (f"{GOLD}/color.bin",                 (64, 64, 64, 3)),
}


def load_f32(path, shape=None):
    a = np.fromfile(path, dtype="<f4")
    if shape is not None:
        n = int(np.prod(shape))
        if a.size != n:
            print(f"  WARNING: {path} has {a.size} floats, shape {shape} "
                  f"wants {n}; leaving flat", file=sys.stderr)
        else:
            a = a.reshape(shape)
    return a


def inspect(name, a):
    flat = a.ravel()
    nan = int(np.isnan(flat).sum())
    inf = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    print(f"{name}: shape={a.shape} dtype={a.dtype} count={flat.size}")
    if finite.size:
        qs = np.percentile(finite, [0, 1, 50, 99, 100])
        print(f"  min={qs[0]:+.4e} p1={qs[1]:+.4e} median={qs[2]:+.4e} "
              f"p99={qs[3]:+.4e} max={qs[4]:+.4e}")
        print(f"  mean={finite.mean():+.4e} std={finite.std():.4e}")
    if nan or inf:
        print(f"  !! NaN={nan} Inf={inf}")


def diff(got, ref, shape):
    if got.shape != ref.shape:
        print(f"  shape mismatch: got {got.shape} vs ref {ref.shape}",
              file=sys.stderr)
        # fall back to flat compare on the common length
        n = min(got.size, ref.size)
        got, ref = got.ravel()[:n], ref.ravel()[:n]
        shape = None
    g, r = got.astype(np.float64), ref.astype(np.float64)
    absd = np.abs(g - r)
    reld = absd / (np.abs(r) + 1e-12)
    k = 8
    worst = np.argsort(absd.ravel())[::-1][:k]
    print(f"  max|abs|={absd.max():.4e}  mean|abs|={absd.mean():.4e}  "
          f"max|rel|={reld.max():.4e}  mean|rel|={reld.mean():.4e}")
    print(f"  worst {k} by abs error:")
    for idx in worst:
        coord = np.unravel_index(idx, shape) if shape else (int(idx),)
        print(f"    {tuple(int(c) for c in coord)}: "
              f"got={g.ravel()[idx]:+.6e} ref={r.ravel()[idx]:+.6e} "
              f"|abs|={absd.ravel()[idx]:.3e}")
    # Per-leading-axis error map (helps spot a single bad plane/channel/token)
    if shape and len(shape) >= 2 and shape[0] <= 64:
        per = absd.reshape(shape[0], -1).mean(1)
        top = np.argsort(per)[::-1][:min(5, shape[0])]
        summary = ", ".join(f"[{int(i)}]={per[i]:.3e}" for i in top)
        print(f"  worst leading-axis slices (mean abs): {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES.keys()),
                    help="known stage (sets golden path + shape)")
    ap.add_argument("--got", help="C-produced .bin to inspect / compare")
    ap.add_argument("--ref", help="reference .bin (defaults to the stage golden)")
    ap.add_argument("--shape", help="comma-separated shape, e.g. 3,40,64,64")
    args = ap.parse_args()

    shape = None
    ref_path = args.ref
    if args.stage:
        ref_path = ref_path or STAGES[args.stage][0]
        shape = STAGES[args.stage][1]
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))

    if not args.got and not ref_path:
        ap.error("need --stage, or --got / --ref")

    # Inspect-only mode (no --got): describe the reference/golden.
    if not args.got:
        if not os.path.exists(ref_path):
            print(f"missing: {ref_path} (run tools/extract_golden.py?)",
                  file=sys.stderr)
            return 1
        inspect(os.path.basename(ref_path), load_f32(ref_path, shape))
        return 0

    got = load_f32(args.got, shape)
    inspect(f"got  ({os.path.basename(args.got)})", got)
    if ref_path:
        if not os.path.exists(ref_path):
            print(f"missing reference: {ref_path}", file=sys.stderr)
            return 1
        ref = load_f32(ref_path, shape)
        inspect(f"ref  ({os.path.basename(ref_path)})", ref)
        print("diff:")
        diff(got, ref, shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
