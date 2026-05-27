#!/usr/bin/env python3
"""One-shot converter: U2Net u2net.pth (PyTorch pickle) -> u2net.safetensors.

The lrm.c background-removal stage (lrm/lrm_u2net.c) reads only safetensors.
U2Net (xuebinqin/U-2-Net, Apache-2.0) distributes its salient-object weights
as a raw state_dict pickle; this passes every tensor through to safetensors as
contiguous float32, preserving the original key names (stageN[d].rebnconv*.
{conv_s1,bn_s1}.*, sideN.*, outconv.*).

Usage:
    python tools/u2net_to_safetensors.py <u2net.pth> <u2net.safetensors>

Requirements: torch, safetensors (already in triposr_env/.venv).
"""
from __future__ import annotations

import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    sd = torch.load(str(input_path), map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit(f"unexpected checkpoint type: {type(sd)}")
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        out[k] = v.detach().to(torch.float32).contiguous()
    save_file(out, str(output_path))
    print(f"wrote {len(out)} tensors -> {output_path} "
          f"({output_path.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    convert(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
