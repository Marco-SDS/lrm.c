#!/usr/bin/env python3
"""
debug_mesh.py - geometry diagnostics for an lrm.c GLB.

Parses a binary glTF 2.0 file directly (numpy only, no trimesh) and reports
the kind of issues the lrm.c geometry pipeline can produce:

  - vertex / face counts, index component type, bounding box
  - surface orientation:  signed volume (>0 = canonical CCW-outward winding)
  - normals:              unit-length check + outward flux (>0 = outward)
  - degenerate faces (zero area) and exact-duplicate faces
  - connected components (floaters): count + size of the largest few
  - attributes present (NORMAL / COLOR_0 / TEXCOORD_0) + material doubleSided
  - embedded texture image (mime + byte size) if present

This is the geometry-side counterpart to tools/check_glb.py (which focuses on
the glTF structure / trimesh round-trip). Use it to localize a bad mesh.

Usage:
    python debug/debug_mesh.py /tmp/robot.glb
"""

import json
import struct
import sys

import numpy as np


def parse_glb(path):
    data = open(path, "rb").read()
    if data[:4] != b"glTF":
        raise ValueError("not a GLB (bad magic)")
    version, total = struct.unpack("<II", data[4:12])
    off, chunks = 12, {}
    while off < len(data):
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        off += 8
        chunks[ctype] = data[off:off + clen]
        off += clen
    js = json.loads(chunks[b"JSON"].decode("utf-8"))
    binc = chunks.get(b"BIN\x00", b"")
    return js, binc, len(data)


def accessor(js, binc, idx):
    a = js["accessors"][idx]
    bv = js["bufferViews"][a["bufferView"]]
    o = bv.get("byteOffset", 0)
    comp = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "<u1"}[a["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    arr = np.frombuffer(binc, dtype=np.dtype(comp),
                        count=a["count"] * ncomp, offset=o)
    return arr.reshape(a["count"], ncomp) if ncomp > 1 else arr


def connected_components(nv, faces):
    """Union-find over shared faces; return component sizes (by vertex)."""
    parent = np.arange(nv)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for f in faces:
        a = find(f[0])
        for v in f[1:]:
            b = find(v)
            if a != b:
                parent[b] = a
    roots = np.array([find(v) for v in range(nv)])
    _, counts = np.unique(roots, return_counts=True)
    return np.sort(counts)[::-1]


def main():
    if len(sys.argv) < 2:
        print("usage: python debug/debug_mesh.py <file.glb>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    js, binc, total = parse_glb(path)

    prim = js["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    P = accessor(js, binc, attrs["POSITION"]).astype(np.float64)
    F = accessor(js, binc, prim["indices"]).astype(np.int64).reshape(-1, 3)
    nv, nf = len(P), len(F)

    print(f"file: {path}  ({total/1e6:.2f} MB total)")
    print(f"  vertices={nv}  faces={nf}  "
          f"indices={'u32' if js['accessors'][prim['indices']]['componentType']==5125 else 'u16'}")
    print(f"  attributes={list(attrs.keys())}")
    mat = js.get("materials", [{}])[0]
    print(f"  material: doubleSided={mat.get('doubleSided')} "
          f"name={mat.get('name')}")

    # Bounding box
    mn, mx = P.min(0), P.max(0)
    print(f"  bbox min=[{mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}] "
          f"max=[{mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}] "
          f"size=[{(mx-mn)[0]:.3f},{(mx-mn)[1]:.3f},{(mx-mn)[2]:.3f}]")

    # Orientation: signed volume from winding (>0 = CCW outward)
    v0, v1, v2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    sv = (v0 * np.cross(v1, v2)).sum() / 6.0
    print(f"  signed volume = {sv:+.5f}  "
          f"({'OK outward' if sv > 0 else 'INWARD winding!'})")

    # Normals
    if "NORMAL" in attrs:
        N = accessor(js, binc, attrs["NORMAL"]).astype(np.float64)
        lens = np.linalg.norm(N, axis=1)
        fc = (v0 + v1 + v2) / 3.0
        area = np.linalg.norm(fn, axis=1) / 2.0
        navg = (N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]]) / 3.0
        flux = ((fc * navg).sum(1) * area).sum()
        # agreement with winding face normals
        vn = np.zeros_like(P)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vnl = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = vn / np.maximum(vnl, 1e-12)
        agree = (N * vn).sum(1)
        print(f"  normals: |n| mean={lens.mean():.4f} "
              f"(min={lens.min():.4f} max={lens.max():.4f}) "
              f"flux={flux:+.4f} ({'outward' if flux > 0 else 'INWARD!'})  "
              f"winding-agreement={(agree > 0).mean()*100:.1f}%")

    # Degenerate / duplicate faces
    area_all = np.linalg.norm(fn, axis=1) / 2.0
    n_degen = int((area_all < 1e-12).sum())
    sorted_faces = np.sort(F, axis=1)
    _, uniq_idx = np.unique(sorted_faces, axis=0, return_index=True)
    n_dup = nf - len(uniq_idx)
    print(f"  degenerate faces={n_degen}  duplicate faces={n_dup}")

    # Connected components (floaters)
    comps = connected_components(nv, F)
    head = ", ".join(str(int(c)) for c in comps[:6])
    print(f"  connected components={len(comps)} "
          f"(largest by verts: {head}{'...' if len(comps) > 6 else ''})")
    if len(comps) > 1:
        frac = comps[1] / comps[0]
        print(f"    -> {len(comps)-1} smaller component(s); "
              f"2nd/1st size ratio={frac:.4f}"
              f"{'  (likely floaters)' if frac < 0.1 else ''}")

    # Texture
    if js.get("images"):
        img = js["images"][0]
        bv = js["bufferViews"][img["bufferView"]]
        print(f"  texture: {img.get('mimeType')} "
              f"{bv['byteLength']/1e6:.2f} MB embedded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
