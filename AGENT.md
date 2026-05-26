This is a C implementation of LRM-family image-to-3D inference, starting
with TripoSR. The project is a fork of `antirez/iris.c` (originally an
image-generation engine for Flux and Z-Image) stripped down and
specialized for single-view 3D reconstruction.

The project is called **lrm.c**.
The binary is **lrmc** (the directory `lrm/` houses the model code, so
the binary name has the trailing `c` of the project name to avoid the
filesystem collision; see LRMengine.md §3).

# High-level capabilities (MVP, Phase 12.5 complete)

Single end-to-end command:

```
./lrmc infer <model_dir> <image> -o <output.glb> [--mc-resolution N]
```

produces a binary glTF 2.0 file with PBR material, vertex normals, and
vertex colors. The pipeline is image → DINO ViT-B/16 → 16-block triplane
decoder → ConvTranspose upsample → density grid (sample + NeRF MLP) →
marching cubes → color re-query → GLB.

All pipeline stages are validated stage-by-stage against the canonical
PyTorch reference (see `tests/` and the per-stage commit messages on
the `features` branch).

# Naming convention (post-fork)

- **`iris_`** prefix for all shared/generic identifiers (kernels,
  safetensors, image I/O) — inherited from the parent project.
- **`lrm_`** prefix for everything model-specific under `lrm/`.
- Files under `lrm/` may include root headers; root code never
  includes from `lrm/`. Unidirectional dependency.

# File structure

```
iris.c               - generic coordinator (error string, dispatch)
iris.h               - minimal public API (image I/O, opaque types)
iris_kernels.c/.h    - CPU compute primitives (matmul, attention, conv2d,
                       LN/GN/BN, GELU/GEGLU/SiLU, grid_sample, RoPE)
iris_metal.m/.h      - Metal runtime (kept; kernels TBD in Phase 13)
iris_shaders.metal   - Metal compute shaders (kept; trimmed in Phase 13)
iris_safetensors.c/.h - mmap'd weight loader (zero-copy)
iris_image.c         - PNG/JPEG/PPM image I/O
png.c/.h jpeg.c/.h   - codec headers (vendored single-file libs)
main.c               - CLI entry point

lrm/                 - ALL LRM-specific code
  lrm.c/.h           - public LRM API + end-to-end orchestration
  lrm_triposr.c/.h   - TripoSR loader + image preprocessing
  lrm_vit_dino.c/.h  - DINOv1 ViT-B/16 encoder
  lrm_triplane_decoder.c/.h    - 16-block transformer (self+cross+GEGLU)
  lrm_triplane_upsample.c/.h   - ConvTranspose2d via packed GEMM + pixel shuffle
  lrm_triplane_sample.c/.h     - per-point bilinear sampling, 3 planes -> 120-d
  lrm_nerf_mlp.c/.h            - 10-layer NeRF MLP (density + RGB)
  lrm_marching_cubes.c/.h      - Lorensen-Cline MC + global-edge dedup
  lrm_mesh_export.c/.h         - binary glTF 2.0 writer (PBR material, normals)

tests/               - parity tests per module
  test_kernels.c
  test_vit_dino.c
  test_triplane_decoder.c
  test_triplane_upsample.c
  test_density_64.c
  test_marching_cubes.c
  test_glb.c
  golden/triposr/    - pinned reference goldens (gitignored)

tools/               - dev-time Python helpers (NOT in hot path)
  ckpt_to_safetensors.py
  extract_golden.py
  check_glb.py

docs/                - architectural reference + dev guide
  ARCHITECTURE.md
  CONTRIBUTING.md

triposr_env/         - pinned TripoSR Python repo + venv (gitignored)
LRMengine.md         - canonical project plan + decisions log + roadmap
README.md            - user-facing intro + quickstart
SPEED.md             - measured performance baseline
```

# Build targets

```
make generic   - Pure C, no system deps. Slow (~10x). Always works.
make blas      - Accelerate (macOS) / OpenBLAS (Linux). Default.
make mps       - Apple Silicon + Metal. Build set up; kernels TBD (Phase 13).
```

Per-module parity tests (each pulls in only the .c files it needs):

```
make test           - kernel parity (atol=1e-5 vs NumPy)
make test-dino      - DINO forward parity (atol=5e-4 vs PyTorch)
make test-decoder   - triplane decoder parity (atol=4e-3 rtol=1e-4)
make test-upsample  - post-processor parity (atol=5e-4 rtol=1e-4)
make test-density   - sampler + NeRF MLP on 64^3 grid
make test-mc        - marching cubes structural (count + area + Chamfer)
make test-glb       - GLB writer structural + trimesh round-trip
make test-e2e       - end-to-end TripoSR inference (~50 s on i9 CPU)
```

# Performance baseline (Phase 14, 2026-05-22)

On Intel i9-9880H + Accelerate (BLAS), end-to-end TripoSR inference on
the canonical robot.png golden image:

- 64³ MC resolution:  ~50 s total (decoder ~47 s = 93 %)
- 256³ MC resolution: ~83 s total (decoder ~50 s + density ~30 s)

The triplane decoder is BLAS-bandwidth-bound at ~22 GFLOPS sustained.
Closing the gap to the LRMengine.md §11 targets (3 s on M3 + Metal)
requires Phase 13.

See [SPEED.md](SPEED.md) for the full per-stage breakdown.

# Development rules

- No additional project dependencies. Acceptable external deps are
  BLAS/OpenBLAS and Metal/MPS from macOS.
- Reject tiny speed gains that add complexity; prefer substantial wins.
- Always test code modifications with `make test` and the relevant
  per-stage test. Run `make test-e2e` before committing anything that
  touches the pipeline.
- Once changes are validated, commit them.
- Never add or commit unrelated unstaged files.
- Keep code simple and understandable; leave no dead code.
- If you optimize one backend, verify others were not regressed.
- Stick to standard C; avoid compiler-specific tricks/pragmas unless
  strictly required.
- Plan-level decisions live in [LRMengine.md §3](LRMengine.md). Re-open
  them only with explicit justification recorded as a new dated entry.

# How to run TripoSR inference

```bash
make blas

# Once: convert the upstream ckpt to safetensors (the engine reads
# only safetensors; conversion is a one-shot Python script).
triposr_env/.venv/bin/python tools/ckpt_to_safetensors.py \
    ~/.cache/huggingface/.../model.ckpt triposr_env/model.safetensors

# Inference (~50 s at 64^3 on Intel i9 + Accelerate).
./lrmc infer triposr_env triposr_env/examples/robot.png \
    -o /tmp/robot.glb --mc-resolution 64

# Open /tmp/robot.glb in any glTF viewer (Blender, gltf.report, etc.).
```

Set `LRM_TIMING=1` in the env to get per-stage walltime breakdown.

# Python reference implementations

For parity checks / debugging:

TripoSR references (pinned):
- Python venv in `./triposr_env/.venv/`
- Official TripoSR code in `./triposr_env/tsr/` (commit `d26e33181`)
- HuggingFace weights: `stabilityai/TripoSR` revision `5b521936b01fbe1890f6f9baed0254ab6351c04a`
- `tools/extract_golden.py` regenerates the 6 golden tensors used by
  the parity tests.

Diffusers references (for Transformer1D + BasicTransformerBlock):
- `triposr_env/tsr/models/transformer/transformer_1d.py`
- `triposr_env/tsr/models/transformer/basic_transformer_block.py`
- `triposr_env/tsr/models/transformer/attention.py`

Rules:
- Never add/commit `triposr_env/` (the .gitignore covers it).
- If missing, ask user before recreating/downloading (~5 GB of deps
  + 1.7 GB of weights).

# TripoSR architecture pinned facts

(See LRMengine.md §4 for the full reference and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for per-module deep-dive.)

```
1. Image preprocessing: alpha bbox + 85 % canvas + composite gray 0.5
   + bilinear resize to 512x512.
2. DINO ViT-B/16 (frozen): 12 layers, hidden=768, 12 heads x 64 dim.
   Output: [1, 1025, 768] (1 CLS + 32x32 patch tokens).
3. Triplane decoder (Transformer1D, 16 blocks):
   - GroupNorm(32) + proj_in(1024 -> 1024)
   - per block: pre-LN(1e-5) + self-attn + +residual
                + pre-LN + cross-attn(K/V from image tokens 768d) + +residual
                + pre-LN + GEGLU FFN(1024 -> 8192 split -> 1024) + +residual
   - proj_out + residual against the cached learned triplane queries
4. Post-processor: ConvTranspose2d 1024 -> 40, kernel 2, stride 2.
   Triplane out: [3, 40, 64, 64].
5. NeRF query: for each 3D point in [-0.87, +0.87]^3, bilinear-sample
   3 planes (xy, xz, yz; padding_mode='zeros'), concat -> 120-d.
   NeRF MLP: 10 Linears with SiLU between, output 4 channels.
   density = exp(raw + density_bias=-1.0), color = sigmoid(features).
6. Marching cubes at threshold 25.0, default 256^3 grid.
7. Vertex color re-query + GLB write with PBR material.
```

# Critical implementation details (gotchas)

- **HF ViT pos-embed interpolation** uses `scale_factor =
  (target+0.1)/source` (the `+0.1` is HF's FP-precision hack). Skipping
  it costs ~3-4 decimal places of parity.
- **GELU is exact** (`erff`-based), not the tanh approximation.
- **LayerNorm eps**: DINO ViT uses **1e-12**; diffusers
  BasicTransformerBlock uses default **1e-5**. Don't mix them up.
- **No bias on Q/K/V** in the decoder (`attention_bias=False`); only
  `to_out` and `proj_*` carry bias.
- **GEGLU layout**: `proj(x).chunk(2, -1)` gives `(hidden, gate)`; output
  is `hidden * gelu(gate)`. The first half is "hidden", the second is "gate".
- **Transformer1D residual** is added against the **original learned
  queries**, not the post-GroupNorm version.
- **TripoSR has NO camera conditioning, NO AdaLN**. Deliberate
  departure from the LRM paper; the model assumes a canonical orbit view.
- **grid_sample padding_mode**: TripoSR uses **'zeros'** (PyTorch default).
  Using 'border' shifts the silhouette outward at the bounding box edges.
- **Marching cubes axis swap**: TripoSR's `MarchingCubeHelper` applies a
  `[2, 1, 0]` swap to `torchmcubes`' output because `torchmcubes` treats
  the input as (depth, height, width). Our C MC emits in natural
  (i, j, k) order directly, bypassing the round-trip.
- **GLB vertex colors as u8 normalized**: f32 COLOR_0 is allowed by
  spec but not universally supported. We quantize at write time.
- **GLB needs explicit material** for vertex_colors to render reliably
  across viewers. We emit a single PBR matte (`metallicFactor=0,
  roughnessFactor=1, doubleSided=true`) so the COLOR_0 attribute is
  multiplied into the base color.

# Communication language

The user works in Italian. Respond in Italian by default unless the
user explicitly switches to English. Keep code identifiers, file
paths, error messages in English.
