"""Inspect a GLB produced by lrm.c. Reports both what trimesh's high-level
loader sees and what the raw glTF JSON+BIN actually contains.

trimesh classifies a primitive with both a material AND a COLOR_0
attribute as `TextureVisuals` (rather than `ColorVisuals`), which
hides `vertex_colors` behind a different API surface. The data is
still in the file - any compliant viewer (Blender, three.js, the
Khronos sample viewer) will combine COLOR_0 with the material's
baseColorFactor per the glTF 2.0 spec. We confirm that by parsing
the binary chunk directly.

Usage:
    python tools/check_glb.py <path.glb>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def parse_glb(path: Path):
    """Return (json_dict, bin_bytes)."""
    with path.open("rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:  # 'glTF'
            raise SystemExit(f"FAIL: not a glTF file (magic 0x{magic:08x})")
        if version != 2:
            raise SystemExit(f"FAIL: glTF version {version}, expected 2")
        clen, _ctype = struct.unpack("<II", f.read(8))
        js_raw = f.read(clen).rstrip(b"\x20\x00")
        js = json.loads(js_raw)
        if length > f.tell():
            blen, _btype = struct.unpack("<II", f.read(8))
            bin_data = f.read(blen)
        else:
            bin_data = b""
    return js, bin_data


def report_via_trimesh(path: Path) -> None:
    try:
        import trimesh
    except ImportError:
        print("  trimesh not installed; skipping loader check")
        return
    import warnings

    warnings.filterwarnings("ignore")
    m = trimesh.load(str(path))
    g = list(m.geometry.values())[0] if hasattr(m, "geometry") and m.geometry else m
    print(f"  trimesh load: vertices={len(g.vertices)} faces={len(g.faces)}")
    print(f"               visual_type={type(g.visual).__name__}")
    if hasattr(g.visual, "material"):
        mat = g.visual.material
        print(f"               material={type(mat).__name__}")


def report_via_raw(path: Path) -> None:
    import numpy as np
    js, bin_data = parse_glb(path)
    print(f"  glTF version: {js['asset']['version']} "
          f"({js['asset'].get('generator', '?')})")

    materials = js.get("materials", [])
    if not materials:
        print("  materials: (NONE - viewers will fall back to flat shading)")
    else:
        for i, mat in enumerate(materials):
            pbr = mat.get("pbrMetallicRoughness", {})
            print(f"  material[{i}]: name={mat.get('name', '?')} "
                  f"baseColor={pbr.get('baseColorFactor', '?')} "
                  f"metallic={pbr.get('metallicFactor', '?')} "
                  f"roughness={pbr.get('roughnessFactor', '?')} "
                  f"doubleSided={mat.get('doubleSided', False)}")

    for mi, mesh in enumerate(js.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = list(prim.get("attributes", {}).keys())
            mat = prim.get("material", "(none)")
            print(f"  mesh[{mi}].prim[{pi}]: attributes={attrs} material={mat}")

    # Sanity-check COLOR_0 data if present.
    for mi, mesh in enumerate(js.get("meshes", [])):
        for prim in mesh.get("primitives", []):
            color_acc_idx = prim.get("attributes", {}).get("COLOR_0")
            if color_acc_idx is None:
                continue
            acc = js["accessors"][color_acc_idx]
            bv = js["bufferViews"][acc["bufferView"]]
            normalized = acc.get("normalized", False)
            ct = acc["componentType"]
            ct_name = {5121: "u8", 5123: "u16", 5126: "f32"}.get(ct, f"?({ct})")
            print(f"  COLOR_0: count={acc['count']} type={acc['type']} "
                  f"componentType={ct_name} normalized={normalized}")
            if ct == 5121:  # u8
                raw = np.frombuffer(
                    bin_data[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]],
                    dtype=np.uint8,
                ).reshape(-1, 4)
                print(f"           raw u8 sample: rgb=[{raw[0,0]},{raw[0,1]},{raw[0,2]}] "
                      f"alpha={raw[0,3]}  (RGB range "
                      f"[{raw[:,:3].min()},{raw[:,:3].max()}])")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    print(f"Inspecting {path}:")
    report_via_trimesh(path)
    report_via_raw(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
