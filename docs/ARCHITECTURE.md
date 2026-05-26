# lrm.c — Architecture reference

Technical reference for the codebase, organized stage-by-stage along
the TripoSR pipeline. For the high-level plan and decisions log see
[LRMengine.md](../LRMengine.md); for current performance numbers see
[SPEED.md](../SPEED.md).

## Repo layout

```
lrm.c/
├── iris.{c,h}                    Generic coordinator (image I/O, error string)
├── iris_kernels.{c,h}            CPU compute primitives (matmul, attention,
│                                 conv2d, LN/GN/BN, GELU, GEGLU, grid_sample, ...)
├── iris_metal.{m,h}              Metal runtime (kept from iris.c; kernels TBD)
├── iris_shaders.metal            Metal compute shaders (kept; trimmed in Phase 13)
├── iris_safetensors.{c,h}        mmap'd weight loader, zero-copy
├── iris_image.c                  PNG / JPEG / PPM loader-saver
├── png.{c,h}  jpeg.{c,h}         Codec headers (vendored single-file libs)
├── main.c                        CLI entry: subcommand dispatch
│
├── lrm/                          ALL LRM-specific code (Phase 3+)
│   ├── lrm.{c,h}                 Public API + end-to-end orchestration
│   ├── lrm_triposr.{c,h}         TripoSR loader + image preprocessing
│   ├── lrm_vit_dino.{c,h}        DINOv1 ViT-B/16 image encoder
│   ├── lrm_triplane_decoder.{c,h} 16-block image-to-triplane transformer
│   ├── lrm_triplane_upsample.{c,h} Post-processor (ConvT 1024→40, 2×)
│   ├── lrm_triplane_sample.{c,h} Per-point feature sampling (3 bilinears + concat)
│   ├── lrm_nerf_mlp.{c,h}        10-layer NeRF MLP → density + RGB
│   ├── lrm_marching_cubes.{c,h}  Lorensen-Cline MC with edge dedup
│   └── lrm_mesh_export.{c,h}     Binary glTF 2.0 writer
│
├── tests/                        Per-module parity tests
│   ├── test_kernels.c            layer_norm, gelu, geglu, grid_sample
│   ├── test_vit_dino.c           DINO forward parity
│   ├── test_triplane_decoder.c   Decoder forward parity
│   ├── test_triplane_upsample.c  ConvT forward parity
│   ├── test_density_64.c         Sampler + MLP on 64³ grid
│   ├── test_marching_cubes.c     Structural (count + area + Chamfer)
│   ├── test_glb.c                GLB writer structural + trimesh round-trip
│   └── golden/triposr/           Pinned reference goldens (gitignored)
│
├── tools/                        Dev-time Python helpers (not in hot path)
│   ├── ckpt_to_safetensors.py    One-shot TripoSR ckpt → safetensors conversion
│   ├── extract_golden.py         Regenerate the 6 golden .npz/.bin sidecars
│   └── check_glb.py              Inspect a GLB (trimesh + raw JSON)
│
└── triposr_env/                  Pinned reference Python repo + venv (gitignored)
    ├── tsr/                      Original TripoSR Python implementation
    └── .venv/                    PyTorch + transformers for golden regeneration
```

## Public API

Defined in `iris.h` + `lrm/lrm.h`:

```c
// Image I/O (libc + iris_image.c)
iris_image *iris_image_load(const char *path);
int         iris_image_save(const iris_image *img, const char *path);
void        iris_image_free(iris_image *img);

// LRM model lifecycle (lrm/lrm.h)
lrm_model *lrm_load(const char *model_dir);
void       lrm_free(lrm_model *m);

// Single end-to-end inference call
lrm_mesh  *lrm_infer(lrm_model *m, const iris_image *im,
                     const lrm_infer_opts *opts);

// Mesh I/O
int        lrm_mesh_save_glb(const lrm_mesh *mesh, const char *path);
void       lrm_mesh_free(lrm_mesh *mesh);

// Diagnostics
const char *iris_get_error(void);
```

The opaque `struct lrm_model` and `struct lrm_mesh` are defined under
`lrm/` (model: `lrm_triposr.h`; mesh: `lrm_mesh_export.h`). The
generic API in the root never references LRM-specific types.

## Pipeline stages

The full TripoSR inference flow, with the C function that owns each
stage and the tensor shapes between stages:

```
input image (RGB or RGBA, any size)
   │
   ▼  lrm_triposr_preprocess (lrm/lrm_triposr.c)
   │   - Alpha bbox (if RGBA)
   │   - Foreground rescale to 85 % canvas
   │   - Composite over gray 0.5 at canvas resolution
   │   - Bilinear resize to 512×512 (no antialias; small parity gap vs PyTorch)
   │   - HWC u8 → CHW f32 in [0, 1]
   │
   ▼  [3, 512, 512] f32 image
   │
   ▼  lrm_vit_dino_forward (lrm/lrm_vit_dino.c)
   │   - ImageNet normalize (mean/std hardcoded)
   │   - Conv2D patch embed (16×16 stride 16, 3 → 768)
   │   - CLS token concat + bicubic-interpolated pos embed (197 → 1025)
   │     (with HuggingFace's (target+0.1)/source scale-factor hack for
   │      bit-exact parity)
   │   - 12 transformer blocks: LN + MHSA + LN + MLP(GELU exact)
   │   - Final LN
   │
   ▼  [1025, 768] image tokens
   │
   ▼  lrm_triplane_decoder_forward (lrm/lrm_triplane_decoder.c)
   │   - GroupNorm(32) + transpose + proj_in (1024 → 1024)
   │   - 16 BasicTransformerBlocks:
   │       pre-LN + self-attn (no bias on Q/K/V)
   │       pre-LN + cross-attn(K/V from image tokens 768)
   │       pre-LN + GEGLU FFN(1024 → 8192 → 1024)
   │   - proj_out + residual against the cached learned triplane queries
   │   - detokenize [3072, 1024] → [3, 1024, 32, 32]
   │
   ▼  [3, 1024, 32, 32] pre-upsample triplane
   │
   ▼  lrm_triplane_upsample_forward (lrm/lrm_triplane_upsample.c)
   │   - Packed BLAS sgemm (160, 1024) @ (1024, 32*32) → (160, 32*32)
   │   - PixelShuffle scatter into [3, 40, 64, 64] + bias add
   │     (mathematically equivalent to ConvTranspose2d kernel=stride=2,
   │      ~30× faster than nested loops)
   │
   ▼  [3, 40, 64, 64] scene code
   │
   ▼  For each chunk of N=8192 query points (default; MC res³ total):
   │   lrm_triplane_sample_forward (lrm/lrm_triplane_sample.c)
   │     - Scale world coords [-r, +r] → [-1, +1]
   │     - Form 3 sets of (xy, xz, yz) 2D coords
   │     - Bilinear grid_sample with padding_mode='zeros' (PyTorch default)
   │     - Rearrange [3, 40, N] → [N, 120]
   │   lrm_nerf_mlp_forward (lrm/lrm_nerf_mlp.c)
   │     - 10 BLAS-routed Linear layers (120 → 64 → 64×8 → 4)
   │     - SiLU between layers (none after the last)
   │     - density[n] = exp(raw[0] - 1.0)
   │     - color  [n,c] = sigmoid(raw[1+c]) for c in 0..2
   │
   ▼  density [mc_res^3] + color [mc_res^3, 3]
   │
   ▼  lrm_marching_cubes_extract (lrm/lrm_marching_cubes.c)
   │   - 256-entry edge_table + tri_table (Lorensen-Cline / Bourke, public domain)
   │   - Cube classification, linear edge interpolation
   │   - Global-edge dedup table (3 * R^3 ints) so each grid edge produces
   │     exactly one vertex (matches torchmcubes)
   │   - Output vertex (i, j, k) in [-radius, +radius]^3 world space
   │
   ▼  raw mesh: vertices [Nv, 3], faces [Nf, 3] int32
   │
   ▼  Vertex color re-query: lrm_triplane_sample + lrm_nerf_mlp on the
   │   vertex positions, in chunks of 8192. RGB outputs become RGBA with
   │   alpha = 1.
   │
   ▼  lrm_mesh_from_buffers + lrm_mesh_save_glb (lrm/lrm_mesh_export.c)
   │   - Compute per-vertex normals (area-weighted face-normal averaging)
   │   - Quantize colors f32 → u8 normalized
   │   - Build 12-byte GLB header + JSON chunk + BIN chunk
   │   - Emit single PBR material with COLOR_0 multiplied into baseColor
   │
   ▼  output.glb
```

## Module reference

### `iris_kernels.{c,h}` — CPU compute primitives

All operations work on f32 row-major tensors. The kernels are
self-contained; `lrm/` files only depend on this header.

| Function | Notes |
|---|---|
| `iris_matmul`, `iris_matmul_t`, `iris_linear`, `iris_linear_nobias`, `iris_linear_nobias_bf16` | Dispatch to BLAS sgemm when `-DUSE_BLAS`, naïve triple loop otherwise. |
| `iris_conv2d` | im2col + GEMM. Used by DINO patch embed. |
| `iris_layer_norm` | Mean/var in fp64 accumulator (fp32 sum lost ~3e-6 per call, compounded over 48 LN calls in the decoder). |
| `iris_group_norm` | 32 groups for the decoder pre-projection. |
| `iris_rms_norm`, `iris_batch_norm` | Inherited from iris.c; not used in the TripoSR path. |
| `iris_gelu` | Exact via libm `erff`. DINO ViT MLP. |
| `iris_geglu` | `hidden * gelu(gate)`. Decoder FFN. |
| `iris_silu`, `iris_silu_mul` | NeRF MLP activation; SwiGLU pattern (unused in TripoSR). |
| `iris_softmax` | Over last dim. |
| `iris_attention`, `iris_flash_attention` | Self- and cross-attention; flash takes the natural `[seq, heads*head_dim]` layout. |
| `iris_grid_sample_bilinear` | PyTorch-compatible: ZEROS / BORDER padding, align_corners=False. |
| `iris_apply_rope`, `iris_compute_rope_freqs` | Unused by TripoSR; kept for future LRM variants. |

### `lrm/lrm_vit_dino.{c,h}` — DINOv1 ViT-B/16 encoder

`facebook/dino-vitb16` (Apache 2.0). Frozen pretrained backbone, used
as-is by TripoSR.

| Layer | Shape |
|---|---|
| input | [B=1, 3, 512, 512] f32 in [0, 1] |
| ImageNet normalize | inline, mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` |
| patch_embed (Conv2D 16×16/16) | [768, 32, 32] |
| reshape + concat CLS | [1025, 768] |
| + interpolated pos_embed | [1025, 768] |
| 12× transformer block | (LN→MHSA→+residual→LN→MLP(GELU)→+residual) |
| final LN | output [1, 1025, 768] |

**Position embedding interpolation**. DINO was trained at 224×224 (14×14
patch grid, 197 pos embeddings = 1 CLS + 196 patch). TripoSR feeds
512×512 (32×32 patch grid, 1025 pos). We bicubic-interpolate at init
time, including HuggingFace's `(target+0.1)/source` scale-factor
floating-point precision hack — without it parity drops by 3–4 decimals.

**Forward parity gate**: atol 5e-4 vs PyTorch reference; measured max
abs err **1.4 × 10⁻⁵**, mean **5.7 × 10⁻⁷**.

### `lrm/lrm_triplane_decoder.{c,h}` — Image-to-triplane transformer

Diffusers' `Transformer1D` + `BasicTransformerBlock` ported in C.

| Block sub-layer | Op |
|---|---|
| GroupNorm(32) + proj_in | [3072, 1024] |
| 16× | LN + self-attn + +residual + LN + cross-attn(K/V from image) + +residual + LN + GEGLU FFN(8192 → 4096 split) + +residual |
| proj_out | [3072, 1024] |
| + residual (cached learned queries) | [3072, 1024] |
| detokenize | [3, 1024, 32, 32] |

**Notable details for parity:**
- `nn.LayerNorm` eps = **1e-5** (not the DINO ViT 1e-12)
- Q/K/V projections have **no bias** (`attention_bias=False`); only
  the attention output projection carries bias
- The final residual is added against the **original learned queries**,
  not the post-GroupNorm version
- GEGLU: `proj(x).chunk(2, -1)` gives (hidden, gate); output is
  `hidden * gelu(gate)`

**Forward parity gate**: atol 4e-3 + rtol 1e-4 (LRMengine.md §10.2
deep-network floor); measured mean abs err 8 × 10⁻⁵, max relative
error 4 × 10⁻⁶ at f32 mantissa precision.

### `lrm/lrm_triplane_upsample.{c,h}` — Post-processor

ConvTranspose2d with `kernel == stride == 2` factors exactly into one
sgemm + pixel shuffle:

```
input [1024, H, W] → linear(160) → output [40, 2H, 2W]
```

where the 160 = 40 * 2 * 2 output channels are scattered into a 2×2
block per output pixel. Same FLOPs as the direct conv, **~30× faster
on CPU**, and trivially Metal-portable as a single matmul.

**Forward parity gate**: atol 5e-4 rtol 1e-4; measured mean err
1.1 × 10⁻⁴, max relative 1.3 × 10⁻⁶.

### `lrm/lrm_triplane_sample.{c,h}` — Bilinear feature sampling

For each 3D query in [-r, +r]³:
- scale to [-1, +1]
- form 3 sets of (col, row) 2D coords matching TripoSR's plane order
  ([[0,1],[0,2],[1,2]] = xy, xz, yz)
- `iris_grid_sample_bilinear` with **`padding_mode='zeros'`** (PyTorch
  default; using BORDER would shift the silhouette outward at edges)
- rearrange [3, 40, N] → [N, 120]

At MC resolution 256 this is called with **16,777,216** query points
in chunks of 8192. The kernel is currently the wall-clock #2 hotspot
after the decoder (~30 s at 256³).

### `lrm/lrm_nerf_mlp.{c,h}` — Density + color MLP

10 Linear layers with SiLU between them. Layer indices inside the
original `nn.Sequential`: 0, 2, 4, ..., 18 (the odd indices are SiLU
modules and have no weights). All 10 layers route through
`iris_linear` → BLAS sgemm.

| Layer | Shape |
|---|---|
| `decoder.layers.0`  | [64, 120] |
| `decoder.layers.{2..16}` (×8) | [64, 64] |
| `decoder.layers.18` | [4, 64] (no activation) |

Output channel 0 → density via `exp(raw - 1.0)`; channels 1..3 → RGB
via `sigmoid`. Constants from `config.yaml`:
`density_activation: exp`, `density_bias: -1.0`,
`color_activation: sigmoid`.

**Forward parity gate** on 64³ grid (262,144 queries):
- density mean abs err 9.2 × 10⁻⁵, max 7.3 × 10⁻² on value 2735
  (relative 2.7 × 10⁻⁵)
- color   mean abs err 2.0 × 10⁻⁷, max 4.4 × 10⁻⁵

### `lrm/lrm_marching_cubes.{c,h}` — Triangle extraction

Classical Lorensen–Cline (1987) with Paul Bourke's 256-entry edge
table and 256×16 triangle table (public domain). Corner numbering
follows Bourke's convention (corner 0 at (0,0,0), edges 0-3 on z=0
face CCW, 4-7 on z=1 face, 8-11 vertical).

**Vertex deduplication**: a `3 × R³` int table maps `(direction, i, j, k)`
to the unique vertex index for that grid edge. Without dedup the
vertex count would be ~3× larger (each interior grid edge is shared
by up to 4 cubes). Memory: 3.1 MB at R=64, ~200 MB at R=256.

**Coordinate convention**: output vertex `(i_pos, j_pos, k_pos)` in
world space. Note: TripoSR's `MarchingCubeHelper` applies a `[2, 1, 0]`
swap to torchmcubes' output because torchmcubes treats the input as
`(depth, height, width) = (z, y, x)`. We emit in the natural
`(i, j, k)` order directly, bypassing the round-trip.

**Structural parity gate**: vertex count ±2%, face count ±2%, surface
area ±1%, normalized Chamfer < 1e-3. Measured (64³, robot.png):
4612/9208 vs 4634/9220 (0.5%/0.1%), areas 2.612/2.615 (0.12%),
Chamfer **1.5 × 10⁻⁷**.

### `lrm/lrm_mesh_export.{c,h}` — Binary glTF 2.0 writer

Single-pass writer, ~250 lines, no vendored library.

**File structure**:
```
[ 12-byte header: magic 'glTF' | version 2 | total_length ]
[ JSON chunk: 4-byte length + 'JSON' + UTF-8 payload, 4-byte aligned ]
[ BIN chunk:  4-byte length + 'BIN\0' + binary payload, 4-byte aligned ]
```

**Accessors** (POSITION + NORMAL + COLOR_0 + indices):

| # | Attribute | Component | Type |
|---|---|---|---|
| 0 | POSITION | f32 | VEC3 |
| 1 | NORMAL   | f32 | VEC3 (area-weighted face-normal averages) |
| 2 | COLOR_0  | **u8 normalized** | VEC4 (RGBA) |
| 3 | indices  | u16 if Nv≤65535, else u32 | SCALAR |

**Material**: single PBR matte (`metallicFactor=0`, `roughnessFactor=1`,
`baseColorFactor=[1,1,1,1]`, `doubleSided=true`). The COLOR_0 attribute
is multiplied into the base color per the glTF 2.0 spec, so vertex
colors show through.

## Threading model

`lrm_infer` is single-threaded by default. Parallelism is the
caller's responsibility:

- BLAS / Accelerate sgemm uses multiple threads internally (this is
  what most of the decoder time goes into).
- The MC density grid loop is per-chunk sequential; an outer parallel
  loop over chunks would be safe to add and would scale linearly
  with cores. Not yet wired up.
- No threads are spawned by the engine itself.

If the caller wants concurrent inferences, they spawn N inference
threads with separate `lrm_model` instances (or a shared model + a
per-thread workspace).

## Memory profile

Steady-state allocations during a single `lrm_infer` call:

| Buffer | 64³ | 256³ |
|---|---:|---:|
| safetensors mmap (read-only) | 1.7 GB | 1.7 GB |
| DINO workspace               | 30 MB | 30 MB |
| Decoder workspace            | 180 MB | 180 MB |
| Upsample workspace           | 640 KB | 640 KB |
| Triplane (post-upsample)     | 1.9 MB | 1.9 MB |
| Density grid                 | 1 MB | 64 MB |
| MC edge-dedup table          | 3 MB | 200 MB |
| Output mesh                  | ~0.2 MB | ~4 MB |
| **Peak resident**            | **~2 GB** | **~2.2 GB** |

Mostly stable across resolutions. The MC dedup table is the only
allocation that scales with `R^3`.

## Build matrix internals

All three backends share the same kernels file (`iris_kernels.c`).
The differentiator is conditional compilation:

```c
#ifdef USE_BLAS
  #ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif
#ifdef USE_METAL
  #include "iris_metal.h"
#endif
```

- `make generic` — no flags, pure C nested loops.
- `make blas` — `-DUSE_BLAS [+ -DACCELERATE_NEW_LAPACK on macOS]`,
  links `-framework Accelerate` or `-lopenblas`.
- `make mps` — `-DUSE_BLAS -DUSE_METAL`, links Accelerate + Metal
  frameworks; the Metal shader source is embedded as a C array at
  build time (`xxd -i iris_shaders.metal > iris_shaders_source.h`).

## Adding a new LRM model

1. Add the model loader: `lrm/lrm_<modelname>.{c,h}`, mirroring
   `lrm_triposr.{c,h}`. Bind sub-modules into a model-specific struct.
2. Extend `struct lrm_model` (currently a typedef for the TripoSR
   model) with a `kind` enum; switch the public API entry points to
   dispatch on it. This refactor lands when the second model arrives.
3. Add a per-model loader + freer in `lrm/lrm.c`'s `lrm_load` switch.
4. If the new model needs kernels we don't have, add them to
   `iris_kernels.{c,h}` (not the model file).
5. Pin a reference Python repo + a `tools/extract_golden_<name>.py`,
   add `tests/golden/<name>/` artifacts (gitignored).
6. Add per-module parity tests under `tests/`, then an end-to-end
   test target in the Makefile.

The kernels stay model-agnostic. The plan rule (LRMengine.md tenet 3)
is **the model is data, the engine is code** — a new model adds one
file and one switch case, not a refactor.

## See also

- [`../LRMengine.md`](../LRMengine.md) — full project plan + decisions
  log + roadmap.
- [`../SPEED.md`](../SPEED.md) — measured performance numbers.
- [`./CONTRIBUTING.md`](./CONTRIBUTING.md) — dev workflow + style.
