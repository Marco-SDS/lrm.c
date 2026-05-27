#!/usr/bin/env python3
"""
mesh_metrics.py - geometry quality metrics between two meshes.

Computes the same family of measures the TripoSR paper uses (Chamfer Distance
+ F-score) plus basic counts, so the engine's output can be evaluated
objectively instead of by eye. Two uses:

  1. Fidelity check: compare lrmc's end-to-end mesh against the PyTorch
     TripoSR reference (the golden mesh) -- closes the loop on end-to-end
     accuracy (per-stage parity tests only cover individual stages).
  2. Lever tuning: compare two lrmc outputs (e.g. --foreground-ratio 0.85 vs
     0.95, or two --threshold values) to quantify how much a knob moves the
     surface.

Points are sampled uniformly on each surface; nearest-neighbour distances are
brute-forced in NumPy (chunked) so no scipy/KD-tree dependency is needed.
Chamfer and the F-score thresholds are reported both raw (world units) and
normalized by the reference bounding-box diagonal, so numbers are scale-robust.

A mesh argument is either a .glb file or a golden-bin prefix P (loads
P_vertices.bin [N,3] f32 + P_faces.bin [F,3] i32), e.g.
`tests/golden/triposr/mesh`.

Usage:
    python tools/mesh_metrics.py <mesh_a> <mesh_b> \
        [--samples N] [--taus 0.01,0.02,0.05] [--max-chamfer T]

Exit code is non-zero only if --max-chamfer is given and the normalized
Chamfer exceeds it (lets `make eval` act as a gate).
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def load_mesh(arg):
    if arg.endswith(".glb") or arg.endswith(".gltf") or arg.endswith(".obj"):
        m = trimesh.load(arg, process=False, force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        return m
    v = np.fromfile(f"{arg}_vertices.bin", dtype="<f4").reshape(-1, 3)
    f = np.fromfile(f"{arg}_faces.bin", dtype="<i4").reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def nn_dist(A, B, chunk=1000):
    """For each point in A, Euclidean distance to its nearest point in B."""
    out = np.empty(len(A), dtype=np.float64)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = ((a[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d2.min(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_a")
    ap.add_argument("mesh_b", help="reference (bbox used for normalization)")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--taus", default="0.01,0.02,0.05")
    ap.add_argument("--max-chamfer", type=float, default=None,
                    help="fail (exit 1) if normalized Chamfer exceeds this")
    args = ap.parse_args()
    np.random.seed(0)  # deterministic surface sampling

    A = load_mesh(args.mesh_a)
    B = load_mesh(args.mesh_b)
    print(f"A: {os.path.basename(args.mesh_a)}  V={len(A.vertices)} F={len(A.faces)}")
    print(f"B: {os.path.basename(args.mesh_b)}  V={len(B.vertices)} F={len(B.faces)}")

    diag = float(np.linalg.norm(B.bounds[1] - B.bounds[0]))
    pa, _ = trimesh.sample.sample_surface(A, args.samples)
    pb, _ = trimesh.sample.sample_surface(B, args.samples)
    pa = np.asarray(pa); pb = np.asarray(pb)

    d_ab = nn_dist(pa, pb)   # A -> B (precision)
    d_ba = nn_dist(pb, pa)   # B -> A (recall)
    chamfer = 0.5 * (d_ab.mean() + d_ba.mean())

    print(f"  bbox diagonal (B) = {diag:.5f}")
    print(f"  Chamfer (mean, bidir) = {chamfer:.6e}  "
          f"(normalized {chamfer / diag:.6e})")
    print(f"  one-sided: A->B mean={d_ab.mean():.3e} max={d_ab.max():.3e}  "
          f"B->A mean={d_ba.mean():.3e} max={d_ba.max():.3e}")

    for ts in args.taus.split(","):
        frac = float(ts)
        tau = frac * diag
        prec = float((d_ab < tau).mean())
        rec = float((d_ba < tau).mean())
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  F-score @ {frac:.3f}*diag (tau={tau:.4f}): "
              f"P={prec:.4f} R={rec:.4f} F={f:.4f}")

    if args.max_chamfer is not None:
        nc = chamfer / diag
        if nc > args.max_chamfer:
            print(f"\nFAIL  normalized Chamfer {nc:.3e} > {args.max_chamfer:.3e}")
            return 1
        print(f"\nPASS  normalized Chamfer {nc:.3e} <= {args.max_chamfer:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
