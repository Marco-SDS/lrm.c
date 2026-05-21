"""One-shot converter: TripoSR model.ckpt (PyTorch pickle) -> model.safetensors.

The lrm.c engine reads only safetensors. TripoSR distributes its weights as a
PyTorch pickle (`model.ckpt`); this script does the one-time conversion. Run
once after downloading the checkpoint; never needed at engine runtime.

Usage:
    python tools/ckpt_to_safetensors.py <input.ckpt> <output.safetensors>

Requirements: torch, safetensors. Install via `pip install torch safetensors`.

Security note: torch.load on a pickle can execute arbitrary code. We pass
`weights_only=True` (PyTorch >= 2.0) so the loader refuses any non-tensor
payload. If your torch is older, upgrade rather than relaxing the flag.
"""

from __future__ import annotations

import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    obj = torch.load(input_path, map_location="cpu", weights_only=True)

    # TripoSR's released model.ckpt is a flat state_dict, but some forks wrap
    # it under a "state_dict" key. Unwrap defensively.
    if isinstance(obj, dict) and "state_dict" in obj and not any(
        hasattr(v, "shape") for v in obj.values()
    ):
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise SystemExit(f"unexpected checkpoint structure: {type(obj).__name__}")

    state_dict = {}
    for k, v in obj.items():
        if not hasattr(v, "detach"):
            raise SystemExit(f"non-tensor entry in checkpoint: {k!r} -> {type(v).__name__}")
        state_dict[k] = v.detach().contiguous().cpu()

    save_file(state_dict, str(output_path))
    print(f"wrote {output_path} ({len(state_dict)} tensors)")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    convert(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
