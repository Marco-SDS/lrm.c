# lrm.c

A minimalist, dependency-free C inference engine for **Large Reconstruction
Models** (image → 3D mesh). A fork of [`antirez/iris.c`](https://github.com/antirez/iris.c)
specialized for the LRM family, starting with **TripoSR**.

> One repo, one binary, zero Python in the hot path — the same philosophy
> [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and
> [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
> applied to LLMs and image generation, brought to single-view 3D
> reconstruction.

## Status

**MVP working** (Phases 1–12, 12.5, 14 of [LRMengine.md](LRMengine.md) complete).
The full TripoSR pipeline runs end-to-end in C, validated stage-by-stage
against the official PyTorch reference. End-to-end output is a viewer-ready
binary glTF with PBR material, vertex normals, and vertex colors.

```
✅  DINO ViT-B/16 encoder        max  err  1.3 × 10⁻⁵ vs PyTorch
✅  Triplane decoder (16 blocks) mean err  8 × 10⁻⁵ (f32 precision floor)
✅  Post-processor upsample      max  err  1.3 × 10⁻⁶
✅  Triplane sampler + NeRF MLP  density 9e-5, color 2e-7
✅  Marching cubes               Chamfer normalized 1.5 × 10⁻⁷
✅  GLB writer                   PBR material + vertex normals + u8 colors
```

**Walltime end-to-end** (Intel i9-9880H + Accelerate, see [SPEED.md](SPEED.md)):

| MC resolution | total | vertices | output GLB |
|---|---:|---:|---:|
| 64³ (golden)  | ~50 s | ~4,500   | ~180 KB |
| 256³ (prod)   | ~83 s | ~76,000  | ~3.9 MB |

The triplane decoder is the bottleneck (~50 s on this hardware,
at Accelerate sgemm bandwidth limit). Apple Silicon Metal kernels
(Phase 13) target ~3 s end-to-end at 256³.

## Quick start

```bash
# 1. Build (pick a backend; Accelerate is automatic on macOS).
git clone <repo> lrm.c && cd lrm.c
make blas

# 2. Convert the TripoSR checkpoint once (one-shot pickle -> safetensors).
git clone https://github.com/VAST-AIResearch/TripoSR.git triposr_env
git -C triposr_env checkout d26e33181947bbbc4c6fc0f5734e1ec6c080956e
python3 -m venv triposr_env/.venv
triposr_env/.venv/bin/pip install torch safetensors huggingface_hub
triposr_env/.venv/bin/python tools/ckpt_to_safetensors.py \
    ~/.cache/huggingface/.../model.ckpt triposr_env/model.safetensors

# 3. Infer a mesh from an image.
./lrmc infer triposr_env triposr_env/examples/robot.png \
    -o robot.glb --mc-resolution 256
```

The output `robot.glb` opens in
[gltf.report](https://gltf.report),
[Blender](https://www.blender.org/) (File → Import → glTF 2.0),
[gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com/),
or any other glTF 2.0 viewer with PBR material, normals, and vertex colors.

The input image should be a 512×512 PNG with an alpha mask (TripoSR is
fragile to background noise). The example PNGs under `triposr_env/examples/`
are pre-cleaned; for arbitrary photos, run [`rembg`](https://github.com/danielgatis/rembg)
externally first. A C-native background-removal step is on the roadmap
([LRMengine.md](LRMengine.md) Phase 16).

## Build matrix

| Target       | Backend         | Status              |
|--------------|-----------------|---------------------|
| `make generic` | Pure C (libm only) | ✅ works (~10× slower) |
| `make blas`    | Accelerate (macOS) / OpenBLAS (Linux) | ✅ macOS verified |
| `make mps`     | Apple Metal     | ⚠️ build set up; kernels TBD ([Phase 13](LRMengine.md)) |

Pure C requires no system dependencies. BLAS picks up Apple Accelerate
automatically on macOS; on Linux it needs `libopenblas-dev` at
`/usr/include/openblas/cblas.h`.

## CLI

```
lrmc info  <model_dir|.safetensors>
    Print the loaded tensor tree (549 tensors, ~1.7 GB for TripoSR).

lrmc infer <model_dir> <image> -o <output.glb> [options]
    --mc-resolution N   Marching cubes grid resolution (default 256).
    --threshold     V   Density threshold (default 25.0).

LRM_TIMING=1 lrmc infer ...
    Per-stage walltime to stderr (preprocess, encoder, decoder, ...).
```

## Tests

```bash
make test           # numerical kernel parity (layer_norm, GELU, GEGLU, grid_sample)
make test-dino      # DINO ViT-B/16 forward vs golden
make test-decoder   # 16-block triplane decoder vs golden
make test-upsample  # ConvTranspose post-processor vs golden
make test-density   # sampler + NeRF MLP on 64³ grid vs golden
make test-mc        # marching cubes structural parity (count + area + Chamfer)
make test-glb       # GLB writer structural + trimesh round-trip
make test-e2e       # end-to-end TripoSR pipeline (~50 s)
```

Goldens are regenerated from the canonical TripoSR reference run with
`tools/extract_golden.py` (see [tools/](tools/)). They are gitignored
(~50 MB total) and reproducible from the pinned upstream commit
(`d26e33181` on `VAST-AI-Research/TripoSR`).

## Architecture

Stage-by-stage in `lrm/`:

| File                          | Role                                                                     |
|-------------------------------|--------------------------------------------------------------------------|
| `lrm.c`                       | Coordinator + `lrm_infer` orchestration                                  |
| `lrm_triposr.{c,h}`           | TripoSR loader + image preprocessing                                     |
| `lrm_vit_dino.{c,h}`          | DINOv1 ViT-B/16 image encoder (1025×768 token output)                    |
| `lrm_triplane_decoder.{c,h}`  | 16-block Transformer1D: self-attn + cross-attn + GEGLU                   |
| `lrm_triplane_upsample.{c,h}` | ConvTranspose2d 1024→40 via packed BLAS sgemm + pixel shuffle             |
| `lrm_triplane_sample.{c,h}`   | Bilinear triplane feature sampling (3 axis-aligned planes → 120-d)       |
| `lrm_nerf_mlp.{c,h}`          | 10-layer NeRF MLP → density (exp) + RGB (sigmoid)                        |
| `lrm_marching_cubes.{c,h}`    | Lorensen–Cline MC with global-edge vertex deduplication                  |
| `lrm_mesh_export.{c,h}`       | Direct binary glTF 2.0 writer with PBR material                          |

Generic infrastructure inherited from `iris.c` at the repo root:
`iris_kernels.{c,h}` (CPU primitives), `iris_metal.{m,h}` (Metal runtime
+ shaders), `iris_safetensors.{c,h}` (zero-copy mmap loader),
`iris_image.c` (PNG/JPEG I/O).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deep technical
reference and [LRMengine.md](LRMengine.md) for the project plan.

## Known limitations

- **One model so far**: only TripoSR. OpenLRM and others are part of the
  abstraction validation phase ([LRMengine.md](LRMengine.md) Phase 15).
- **No texture baking**: vertex colors only. TripoSR's `--bake-texture`
  output (UV atlas + texture PNG) is proposed as Phase 18.
- **No background removal in C**: input must already have a meaningful
  alpha channel. Phase 16 is the in-C `rembg`-equivalent.
- **No Metal yet**: build target exists but kernels are CPU-only.
  Phase 13 closes this gap, requires Apple Silicon hardware to develop
  and validate.
- **Linux OpenBLAS not yet end-to-end tested**: the `cblas_*` API is
  identical to Accelerate so the path should work; it just has not
  run on a Linux box yet.

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by Stability AI
  / Tripo (MIT) — the Day-1 model.
- [`antirez/iris.c`](https://github.com/antirez/iris.c) by Salvatore
  Sanfilippo (MIT) — the parent project we forked from; the
  Metal/BLAS/pure-C kernels layer and the safetensors loader are
  inherited directly.
- [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) and
  [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
  for the architectural inspiration.
- **Lorensen & Cline, 1987** — the marching cubes 256-entry case table
  (used verbatim from Paul Bourke's public-domain reference at
  paulbourke.net/geometry/polygonise/).
- [`facebook/dino-vitb16`](https://huggingface.co/facebook/dino-vitb16)
  — DINOv1 weights TripoSR builds on.

## License

MIT, matching the parent `iris.c`.
