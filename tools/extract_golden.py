"""Capture golden parity checkpoints from the official TripoSR reference run.

Runs the canonical TripoSR pipeline on a fixed input image and dumps six
artifacts under tests/golden/triposr/ for the C engine to validate against:

  1. input_512.npz       - preprocessed image, shape [1, 512, 512, 3]
  2. dino_tokens.npz     - encoder output, shape [1, 1025, 768]
  3. triplane.npz        - post-upsample scene code, shape [1, 3, 40, 64, 64]
  4. density_64.npz      - density+color on a 64^3 query grid (262144 pts)
  5. mesh.npz            - vertices, faces, vertex_colors at 64^3 MC resolution
  6. output.glb          - final mesh exported via trimesh

A meta.json file records the run configuration (pinned commit, input image,
device, resolution, threshold) for reproducibility.

The reference TripoSR repo must be checked out at the pinned commit under
`triposr_env/` at the repo root. See LRMengine.md section 3 for the pin.

Usage:
    python tools/extract_golden.py [--image PATH] [--mc-resolution N]
                                   [--no-remove-bg] [--device DEV]

By default, uses triposr_env/examples/chair.png as input (a stable canonical
example shipped with the upstream repo), 64^3 MC resolution, automatic device
selection (mps -> cuda -> cpu), and rembg-based background removal.

Requirements (install in a venv under triposr_env/):
    pip install torch torchvision  # match your platform
    pip install -r triposr_env/requirements.txt
    pip install safetensors        # for ckpt_to_safetensors.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Locate the lrm.c repo root and the pinned TripoSR reference under triposr_env/.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRIPOSR_DIR = REPO_ROOT / "triposr_env"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "triposr"

if not (TRIPOSR_DIR / "tsr" / "system.py").exists():
    raise SystemExit(
        f"TripoSR reference repo not found at {TRIPOSR_DIR}. "
        "Clone it with:\n"
        "  git clone https://github.com/VAST-AI-Research/TripoSR.git triposr_env\n"
        "  cd triposr_env && git checkout d26e33181947bbbc4c6fc0f5734e1ec6c080956e"
    )

sys.path.insert(0, str(TRIPOSR_DIR))

# tsr/utils.py does `import rembg` at module load time. We don't actually need
# rembg for inputs that already carry an alpha channel (TripoSR's
# remove_background() short-circuits in that case), and rembg transitively
# pulls numba+llvmlite which often fails to build on newer Pythons. Inject a
# stub so the top-level import succeeds; if rembg is genuinely needed at run
# time the stub's attribute access will raise a clear error.
if "rembg" not in sys.modules:
    import types

    _rembg_stub = types.ModuleType("rembg")

    def _rembg_missing(*_args, **_kwargs):
        raise RuntimeError(
            "rembg is not installed in this venv. The input image must already "
            "carry an alpha channel (TripoSR examples do)."
        )

    _rembg_stub.new_session = _rembg_missing
    _rembg_stub.remove = _rembg_missing
    sys.modules["rembg"] = _rembg_stub


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def preprocess_image(image_path: Path, remove_bg: bool, fg_ratio: float):
    """Reproduce run.py's preprocessing exactly (rembg + 85% fg + gray comp).

    rembg is imported lazily so the script also works in environments where
    rembg cannot be installed (it transitively pulls numba/llvmlite which
    sometimes fails to build). When remove_bg=True but the input already has
    a meaningful alpha channel, TripoSR's remove_background() is a no-op
    anyway, so we still skip the rembg import in that case.
    """
    import numpy as np
    from PIL import Image
    from tsr.utils import resize_foreground

    img = Image.open(image_path)

    if remove_bg:
        already_has_alpha = img.mode == "RGBA" and img.getextrema()[3][0] < 255
        if already_has_alpha:
            # TripoSR's remove_background() short-circuits in this case; we do
            # the same without touching rembg so we avoid the heavy import.
            pass
        else:
            import rembg  # only required for truly opaque inputs
            from tsr.utils import remove_background

            session = rembg.new_session()
            img = remove_background(img, session)

        img = resize_foreground(img, fg_ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        # Composite RGBA over gray 0.5 background, exactly as run.py does.
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
    else:
        img = img.convert("RGB")
    return img


def pinned_commit() -> str:
    """Return the SHA currently checked out in triposr_env/ for traceability."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(TRIPOSR_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=TRIPOSR_DIR / "examples" / "chair.png",
        help="Input image (default: triposr_env/examples/chair.png).",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=64,
        help="Marching cubes grid resolution for golden capture (default: 64). "
        "Use 256 for full production-equivalent golden (much slower, ~64 MB dump).",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Skip rembg background removal (input must already be clean).",
    )
    parser.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Foreground rescale ratio (matches run.py default).",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=25.0,
        help="MC density threshold on post-exp density (matches TripoSR default).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto (mps/cuda/cpu), mps, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--pretrained",
        default="stabilityai/TripoSR",
        help="HF repo id or local dir holding config.yaml + model.ckpt.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    # Heavy imports happen here so --help works without torch installed.
    import numpy as np
    import torch
    from einops import rearrange
    from tsr.system import TSR
    from tsr.utils import scale_tensor

    device = select_device(args.device)
    logging.info("device = %s", device)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Preprocess the input image (rembg + foreground rescale).
    pil_image = preprocess_image(args.image, not args.no_remove_bg, args.foreground_ratio)

    # ----- 2. Load TSR with the canonical config.yaml + model.ckpt.
    logging.info("loading TSR from %s", args.pretrained)
    model = TSR.from_pretrained(
        args.pretrained,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    model.eval()

    cond_size = model.cfg.cond_image_size  # 512

    # ----- 3. Replicate TSR.forward() but capture intermediates.
    with torch.no_grad():
        rgb_cond = model.image_processor([pil_image], cond_size)[:, None].to(device)
        # rgb_cond shape: [B=1, Nv=1, H=512, W=512, C=3]
        input_image_tokens = model.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )
        # input_image_tokens shape: [B=1, 1025, 768]

        triplane_tokens = model.tokenizer(rgb_cond.shape[0])
        triplane_tokens = model.backbone(
            triplane_tokens,
            encoder_hidden_states=input_image_tokens,
        )
        scene_codes = model.post_processor(model.tokenizer.detokenize(triplane_tokens))
        # scene_codes shape: [B=1, 3, 40, 64, 64]

    # ----- 4. Query density+color on a uniform mc_resolution^3 grid.
    n = args.mc_resolution
    radius = model.renderer.cfg.radius
    with torch.no_grad():
        # Build a [n,n,n,3] grid in [-radius, radius]^3.
        lin = torch.linspace(-radius, radius, n, device=device)
        xs, ys, zs = torch.meshgrid(lin, lin, lin, indexing="ij")
        grid = torch.stack([xs, ys, zs], dim=-1).view(-1, 3)  # [n^3, 3]
        net = model.renderer.query_triplane(model.decoder, grid, scene_codes[0])
        density_grid = net["density_act"].view(n, n, n).cpu().numpy()
        color_grid = net["color"].view(n, n, n, 3).cpu().numpy()

    # ----- 5. Extract mesh at the golden mc_resolution.
    with torch.no_grad():
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=n,
            threshold=args.density_threshold,
        )
    mesh = meshes[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)

    # ----- 6. Dump artifacts.
    rgb_cond_np = rgb_cond.detach().cpu().numpy().astype(np.float32)
    dino_tokens_np = input_image_tokens.detach().cpu().numpy().astype(np.float32)
    triplane_np = scene_codes.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(GOLDEN_DIR / "input_512.npz", image=rgb_cond_np)
    np.savez_compressed(GOLDEN_DIR / "dino_tokens.npz", tokens=dino_tokens_np)
    np.savez_compressed(GOLDEN_DIR / "triplane.npz", triplane=triplane_np)
    np.savez_compressed(
        GOLDEN_DIR / "density_64.npz",
        density=density_grid,
        color=color_grid,
        resolution=np.int32(n),
        radius=np.float32(radius),
    )
    np.savez_compressed(
        GOLDEN_DIR / "mesh.npz",
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
    )
    glb_path = GOLDEN_DIR / "output.glb"
    mesh.export(str(glb_path), file_type="glb")

    # ----- 7. Write meta.json for reproducibility.
    meta = {
        "triposr_commit": pinned_commit(),
        "image_path": str(args.image.relative_to(REPO_ROOT))
        if args.image.is_relative_to(REPO_ROOT)
        else str(args.image),
        "image_sha256": file_sha256(args.image),
        "mc_resolution": n,
        "density_threshold": args.density_threshold,
        "foreground_ratio": args.foreground_ratio,
        "remove_bg": not args.no_remove_bg,
        "device": device,
        "cond_image_size": int(cond_size),
        "radius": float(radius),
        "shapes": {
            "input_512": list(rgb_cond_np.shape),
            "dino_tokens": list(dino_tokens_np.shape),
            "triplane": list(triplane_np.shape),
            "density_grid": [n, n, n],
            "color_grid": [n, n, n, 3],
            "vertices": list(vertices.shape),
            "faces": list(faces.shape),
        },
    }
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    logging.info("dumped golden artifacts to %s", GOLDEN_DIR)
    logging.info(
        "  vertices=%d faces=%d density_grid=%d^3", len(vertices), len(faces), n
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
