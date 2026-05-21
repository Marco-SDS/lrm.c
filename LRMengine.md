# LRMengine — A pure-C inference engine for LRM-family models

> A minimalist, dependency-free C runtime for **Large Reconstruction Models**
> (image → 3D mesh). A fork of `antirez/iris.c`, specialized for the LRM
> family starting with **TripoSR**. Same philosophy as `llama.cpp` and
> `stable-diffusion.cpp`: one repo, one binary, zero Python in the hot path.
>
> **Status**: planning v2.0 (2026-05-21). Architectural research complete,
> roadmap committed, ready to begin Phase 1 (golden rig).

---

## 0. Tooling-free in five seconds

```bash
git clone https://github.com/<you>/lrm.c.git
cd lrm.c
make generic                          # produces ./lrmc (single ~3 MB binary)
./lrmc download triposr               # ~1.7 GB checkpoint, auto-converted to safetensors
./lrmc infer photo.png -o out.glb
# → out.glb (binary glTF 2.0 with vertex colors)
```

That is the user-facing surface, in full. Everything else is plumbing.

---

## 1. Why this project exists

### 1.1 The gap

LRM-family models — TripoSR, OpenLRM, and their open-source successors —
turn a single photograph into a textured 3D mesh in seconds. They are
**architecturally simple** (ViT encoder + cross-attention transformer +
small MLP + marching cubes) and their open weights are **permissively
licensed** (MIT for TripoSR and OpenLRM).

The reference implementations, however, sit on the Python + PyTorch +
diffusers + transformers stack:

- **Image footprint**: 3–4 GB minimum to ship them
- **Cold start**: 30–60 seconds just to `import torch` + load weights
- **Distribution**: requires Python, pip, and a venv on the target host
- **Optimization ceiling**: framework overhead dominates inference cost
- **Supply chain**: dozens of transitive deps, each a security surface

For the LLM half of the ML stack, llama.cpp solved this problem by
replacing the entire PyTorch runtime with hand-rolled C. For the
image-generation half, stable-diffusion.cpp did the same. **LRMengine is
the equivalent move for 3D reconstruction.**

### 1.2 Fork heritage

This project is a fork of [`antirez/iris.c`](https://github.com/antirez/iris.c),
itself a pure-C inference engine for Flux.2 Klein and Z-Image-Turbo
text-to-image models. We inherit from iris.c:

- The Metal/BLAS/pure-C three-backend kernels layer (`iris_kernels.c`,
  `iris_metal.m`, `iris_shaders.metal`)
- The mmap'd safetensors loader (`iris_safetensors.c`)
- The image I/O (`iris_image.c`, `png.c`, `jpeg.c`)
- The Makefile structure and the autodetected backend selection

We strip everything diffusion-specific: the Flux MMDiT transformer, the
Z-Image S3-DiT transformer, the Euler ODE sampler, the VAE encoder/decoder,
the Qwen3 text encoder + tokenizer, the embedding cache, the AdaLN /
modulation / CFG / schedule machinery. LRM-family models have a single
forward pass with no text and no iterative denoising, so all of that is
dead weight.

In its place we add an `lrm/` subdirectory housing the LRM-specific
modules: DINOv1 ViT encoder, triplane decoder, triplane sampler, NeRF MLP,
marching cubes, GLB writer. Strict invariant: files under `lrm/` may
include the root headers, but the root never includes from `lrm/`.

### 1.3 What we get back

| Property | Reference (PyTorch) | LRMengine target |
|---|---|---|
| Binary size | ~3.5 GB image | ~3 MB binary + weights |
| Cold start (process → ready) | 30–60 s | < 1 s |
| Single-image inference (warm, M-series Metal) | 5–15 s | < 3 s |
| Distribution | Docker / pip | Single binary, copy and run |
| Native arch on Apple Silicon | x86_64 emulation typical | arm64 first-class |
| External dependencies | torch, diffusers, transformers, … | libc + Metal/BLAS |

Targets, not guarantees — but each is realistic and verified by the
equivalent achievements of llama.cpp and stable-diffusion.cpp.

---

## 2. Philosophy

The five tenets, ordered. When in doubt, the higher tenet wins.

1. **No dependencies beyond libc**, except optional, single-file,
   MIT/zlib-licensed libraries vendored into the tree. No git submodules,
   no package manager, no system-wide installs. `make` produces the
   binary on a fresh checkout.
2. **One file per responsibility**. A reader unfamiliar with the codebase
   should be able to point at `lrm/lrm_triposr.c` and know exactly which
   model it speaks for, with no surprise indirection through other files.
3. **The model is data, the engine is code**. Adding a new LRM variant
   means writing one new file (`lrm/lrm_<modelname>.c`) and editing one
   switch in the loader. Architectural change happens in the kernels
   layer, shared by every model.
4. **Correctness before speed**. Every model module ships with a "golden
   parity test" that compares its output to a reference PyTorch run on a
   fixed input. We optimize only what we can re-verify with that test
   still passing.
5. **Apple Silicon and x86_64 are equal first-class citizens**. The MPS
   (Metal) backend is not an afterthought; it is one of three peer
   backends along with BLAS and pure-C.

These are non-negotiable. Anything that proposes weakening tenet 1
("just this once we'll add libxxx") triggers an explicit design-review
escalation.

---

## 3. Decisions log

Choices made during planning that subsequent work must respect. Reopen
only with explicit justification.

**2026-05-21 — Initial planning round.**

- **Directory layout.** Root keeps generic infrastructure (kernels,
  Metal runtime, safetensors, image I/O) with the `iris_` prefix.
  LRM-specific additions go under `lrm/` with the `lrm_` prefix.
  Dependency arrow is unidirectional: `lrm/*` may include root headers,
  never the reverse.
- **Cleanup approach.** Aggressive single-commit removal of all
  diffusion-specific files (Flux transformer, Z-Image transformer, VAE,
  sampler, Qwen3, tokenizer, embedding cache) plus surgical removal of
  the AdaLN/CFG/schedule machinery in shared files. Git history
  preserves the diffusion code if ever needed.
- **Checkpoint format.** TripoSR ships `model.ckpt` (PyTorch pickle).
  We add `tools/ckpt_to_safetensors.py` as a one-shot conversion step.
  The C engine reads only safetensors — no pickle reader in C.
- **Background removal (rembg).** Long-term goal: vendor a small
  segmentation model (RMBG-1.4 / IS-Net / U2NetP class) ported to C so
  the engine has zero Python at runtime. This is a *post-MVP* phase
  (Phase 16). v1 ships with "user provides PNG with alpha; transparent
  regions become gray 0.5" documented in the README.
- **Golden test resolution.** 64³ density grid for CI parity (~1 MB
  dump, 262K queries, seconds in CI). 256³ for production runs. The
  marching cubes path has no known resolution-dependent bug, so this
  split is safe.
- **Golden capture strategy.** Pin TripoSR repo at a specific commit,
  dump 6 checkpoint tensors at architectural boundaries (preprocessed
  image, DINO tokens, post-upsample triplane, density field 64³, MC
  vertices/faces, GLB). No per-layer dumps in CI. Per-layer parity is
  for debugging when a module test fails.
- **Threading.** Inherit iris.c's single-threaded-per-call model.
  Concurrency stays the caller's responsibility (OpenMP inside BLAS
  GEMM is fine).
- **Binary name.** Project is `lrm.c`. Binary is `./lrmc` (the trailing
  `c` echoes the "X.c → Xc" pattern: bison → ybacc-like). We picked
  this in Phase 3 because the binary `./lrm` and the directory `./lrm/`
  cannot coexist in the same parent on a POSIX filesystem, and the
  directory name is the one referenced throughout the design. Static
  library is `liblrmc.a`.
- **Pin recommendations for parity reference:**
  - TripoSR repo @ `d26e33181947bbbc4c6fc0f5734e1ec6c080956e`
  - HF `stabilityai/TripoSR` @ `5b521936b01fbe1890f6f9baed0254ab6351c04a`
  - HF `facebook/dino-vitb16` HEAD (stable since 2021)

---

## 4. Reference architecture: TripoSR

The Day-1 target. Every architectural fact below is verified against the
official repo, the HF model card, and the TripoSR tech report
(arXiv:2403.02151).

### 4.1 Pipeline

```
Input RGB image (any res, foreground pre-extracted)
        │
        ▼  resize → 512×512, normalize (ImageNet mean/std)
[3, 512, 512]
        │
        ▼  DINOv1 ViT-B/16 (frozen)
Image tokens: [1025, 768]                  (CLS + 32×32 patch tokens)
        │
        ▼  16-layer Transformer1D (self-attn + cross-attn into image tokens + GEGLU)
Triplane query tokens: [3072, 1024]        (3 planes × 32 × 32 learned queries)
        │
        ▼  detokenize → [3, 1024, 32, 32]
        │  TriplaneUpsampleNetwork (ConvTranspose, 1024→40 ch, 2× spatial)
Triplane feature volume: [3, 40, 64, 64]   (the "scene code")
        │
        ▼  For each 3D query point in [-0.87, +0.87]³:
        │    project onto xy/xz/yz, bilinear grid_sample (border, align_corners=False),
        │    concat 3 planes → 120-dim feature
        │
        ▼  NeRFMLP (11 linear layers, 64 hidden, SiLU)
density (1) + features (3, sigmoid → RGB)
        │
        ▼  Marching Cubes on density grid (default 256³, ~16.78M queries)
Mesh: vertices + faces + vertex colors
        │
        ▼  GLB writer (cgltf)
output.glb
```

### 4.2 Encoder

- Backbone: `facebook/dino-vitb16` (DINOv1, **not** v2)
- Config: `hidden=768`, `layers=12`, `heads=12`, `intermediate=3072`,
  `patch_size=16`
- Frozen during TripoSR training; we load and use as-is
- Preprocessing: foreground centered/scaled to 85% of canvas, composited
  on gray 0.5, resized to 512×512, normalized with ImageNet mean
  `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`
- All 1025 tokens (CLS + patches) are forwarded to the decoder

### 4.3 Decoder

- 16 blocks, hidden 1024, 16 heads × 64 head_dim
- Each block: `LN → SelfAttn → LN → CrossAttn(KV=image) → LN →
  FF(GEGLU, 1024→4096→1024)`
- **No camera conditioning, no AdaLN.** Deliberate simplification vs
  the LRM paper. The model assumes a canonical orbit-view camera.
- Triplane query tokens are a learned `nn.Parameter` of shape
  `(3, 1024, 32, 32)`, flattened to a length-3072 sequence
- Pre-transformer projection uses GroupNorm(32 groups)

### 4.4 Post-processor

- `TriplaneUpsampleNetwork`: ConvTranspose 1024→40 ch with 2× spatial
  upsample. Output `[3, 40, 64, 64]`.

### 4.5 Triplane sampler + NeRF MLP

- Sampling: bilinear `grid_sample`, `padding_mode='border'`,
  `align_corners=False`. Plane axis pairs in order: `(xy, xz, yz)` with
  index pairs `[0,1], [0,2], [1,2]`.
- Feature reduction: **concat** (3 × 40 = 120-dim per point).
- NeRFMLP: `Linear(120→64) + SiLU`, then 9 × `Linear(64→64) + SiLU`,
  then `Linear(64→4)`. ~40K params total.
- Density activation: `exp(raw - 1.0)`. MC threshold = 25.0 on the
  *post-activation* value (raw equivalent ≈ 4.22).
- Color activation: sigmoid on the 3 trailing channels.

### 4.6 Marching cubes + export

- Standard Lorensen–Cline 256-entry case table
- Default grid resolution 256³
- After MC: re-query the NeRF MLP at each vertex position to get
  vertex colors
- Export: GLB with vertex colors (no texture baking in v1; that's the
  `--bake-texture` path from the May 2024 PR, deferred)

### 4.7 Parameter budget

| Module | Approx params |
|---|---:|
| DINO ViT-B/16 (frozen) | ~86 M |
| Triplane query embedding | ~3 M |
| Transformer1D decoder | ~400 M |
| TriplaneUpsampleNetwork | ~3–6 M |
| NeRFMLP | ~0.04 M |
| **Total** | **~500 M** |

### 4.8 Computational profile

- **FLOPs dominator**: 16-layer decoder over 3072 triplane tokens
  (self-attn O(N²·d), cross-attn O(N·M·d)).
- **Wall-clock dominator at 256³**: the 16.78M NeRF MLP queries for
  marching cubes. This is where optimization pays off most. Chunked
  batched matmul plus early-termination on low-density voxels are the
  two levers.
- **Memory dominator**: naive 256³ density field allocation (~8 GB if
  not chunked). The reference renderer chunks queries; we must too.
- DINO encoder is comparatively cheap.

### 4.9 Checkpoint format

- File: `model.ckpt` (1.68 GB, PyTorch pickle — **not** safetensors)
- Config: `config.yaml` (OmegaConf-style, ~1 KB)
- Top-level weight prefixes:
  - `image_tokenizer.model.embeddings.*` and
    `image_tokenizer.model.encoder.layer.{0..11}.*` (DINO)
  - `tokenizer.embeddings` (learned triplane query parameter)
  - `backbone.proj_in`, `backbone.transformer_blocks.{0..15}.*`,
    `backbone.proj_out`
  - `post_processor.*`
  - `decoder.layers.{0..10}.*`

### 4.10 Non-obvious gotchas

- No camera input. Do not allocate AdaLN paths.
- Background removal is required upstream of the encoder; the model is
  fragile to background noise.
- Foreground rescale = 85%, gray composite = 0.5 exactly.
- Coordinate system: `[-0.87, +0.87]³`. Plane axis order
  `(xy, xz, yz)` — wrong order silently mirrors geometry.
- `align_corners=False` in grid_sample. Half-texel offsets matter.
- CLS token is included in cross-attention. Do not drop it.
- Triplane token flattening order: `[plane, h, w]` C-contiguous.

---

## 5. Inventory of the fork

What we inherit from iris.c, what we remove, what we add.

### 5.1 Kept as-is (generic infrastructure)

| File | Role |
|---|---|
| `iris_safetensors.c/.h` | mmap'd safetensors loader, zero-copy |
| `iris_image.c` | RGB image struct, resize, save/load |
| `png.c/.h` | PNG codec |
| `jpeg.c/.h` | JPEG decoder |
| `Makefile` (snellito) | Three-backend autodetect |

### 5.2 Kept with surgery

| File | Removed | Kept |
|---|---|---|
| `iris_kernels.c/.h` | nothing significant (5 diffusion refs total) | matmul, linear, conv2d, attention, softmax, silu, RMSNorm, GroupNorm, BatchNorm, RoPE, upsample_nearest, patchify, randn. **Add**: LayerNorm, GELU exact, GEGLU, grid_sample bilinear |
| `iris_metal.m/.h` | all `*_flux_*`, `*_zimage_*`, `*_vae_*`, `*_qwen_*` pipeline dispatch | Metal runtime, allocators, cached/uncached SGEMM, generic attention, RoPE primitive, conv2d |
| `iris_shaders.metal` | shaders nominati `*_flux_*`, `*_zimage_*`, `*_vae_*` | all generic primitive shaders |
| `iris.c` | router for Flux/Z-Image, `load(vae/transformer/tokenizer)`, `generate/img2img/multiref`, CFG/schedule logic | dispatch skeleton (model kind autodetect) |
| `iris.h` | `iris_generate*`, `iris_img2img*`, `iris_encode_text`, `iris_denoise_step`, sigma schedules, `iris_params` | `iris_image` struct, image I/O, `iris_get_error` |
| `main.c` + `iris_cli.c` | REPL with text prompts, `--steps`/`--guidance`, schedule flags | argparse skeleton, subcommand dispatch |

### 5.3 Removed entirely

| File | Reason |
|---|---|
| `iris_transformer_flux.c` (229 KB) | Flux MMDiT — wrong architecture |
| `iris_transformer_zimage.c` (103 KB) | Z-Image S3-DiT — wrong architecture |
| `iris_sample.c` (45 KB) | Euler ODE denoising — LRM has no iterative sampling |
| `iris_vae.c` (57 KB) | VAE — LRM produces triplanes/density directly |
| `iris_qwen3.c/.h` + `iris_qwen3_tokenizer.c` (~112 KB) | Text encoder — LRM is image-only |
| `iris_tokenizer.c` (16 KB) | Tokenizer — no text |
| `embcache.c/.h` (11 KB) | Text embedding cache — no text |

**Total removed: ~570 KB of source.**

### 5.4 Added under `lrm/`

| File | Role |
|---|---|
| `lrm/lrm.h` | LRM public API: `lrm_load`, `lrm_infer`, `lrm_mesh_save_glb`, `lrm_free` |
| `lrm/lrm_vit_dino.c/.h` | DINOv1 ViT-B/16 encoder |
| `lrm/lrm_triplane_decoder.c/.h` | 16-block Transformer1D with self+cross attention + GEGLU |
| `lrm/lrm_triplane_upsample.c/.h` | Post-processor (ConvTranspose-equivalent) |
| `lrm/lrm_triplane_sample.c/.h` | Bilinear grid_sample over 3 planes |
| `lrm/lrm_nerf_mlp.c/.h` | 11-layer NeRF MLP head |
| `lrm/lrm_marching_cubes.c/.h` | Lorensen–Cline MC |
| `lrm/lrm_mesh_export.c/.h` | GLB writer (cgltf) |
| `lrm/lrm_triposr.c/.h` | TripoSR-specific glue (config parsing, weight remap) |

### 5.5 Added under `third_party/`

| File | License | Role |
|---|---|---|
| `cgltf.h` | MIT, single header | GLB writer |

### 5.6 Added under `tools/`

| File | Role |
|---|---|
| `ckpt_to_safetensors.py` | One-shot PyTorch pickle → safetensors converter |
| `extract_golden.py` | Pin TripoSR repo, dump the 6 golden checkpoint tensors |
| `perf_benchmark.py` | CSV of cold/warm/e2e timings |

---

## 6. Repo layout (target)

```
lrm.c/
├── Makefile
├── README.md  AGENT.md  LRMengine.md  SPEED.md
├── LICENSE
│
├── iris.c                       — coordinator: load, dispatch by model kind
├── iris.h                       — minimal public API (image I/O, opaque types)
├── iris_kernels.c/.h            — CPU kernels (incl. new LayerNorm/GELU/GEGLU/grid_sample)
├── iris_metal.m/.h              — Metal runtime
├── iris_shaders.metal           — Metal compute shaders
├── iris_safetensors.c/.h        — mmap'd weight loader
├── iris_image.c                 — image struct + utilities
├── png.c/.h    jpeg.c/.h        — image codecs
├── main.c                       — CLI: subcommand dispatch
│
├── lrm/                         — ALL LRM-specific code
│   ├── lrm.h
│   ├── lrm_vit_dino.c/.h
│   ├── lrm_triplane_decoder.c/.h
│   ├── lrm_triplane_upsample.c/.h
│   ├── lrm_triplane_sample.c/.h
│   ├── lrm_nerf_mlp.c/.h
│   ├── lrm_marching_cubes.c/.h
│   ├── lrm_mesh_export.c/.h
│   └── lrm_triposr.c/.h
│
├── third_party/
│   └── cgltf.h
│
├── tests/
│   ├── runner.c
│   ├── golden/triposr/
│   │   ├── input_512.npz
│   │   ├── dino_tokens.npz
│   │   ├── triplane.npz
│   │   ├── density_64.npz
│   │   ├── mesh.npz
│   │   └── output.glb
│   ├── test_kernels.c           — LayerNorm, GELU, GEGLU, grid_sample parity
│   ├── test_safetensors.c
│   ├── test_vit_dino.c
│   ├── test_triplane_decoder.c
│   ├── test_triplane_upsample.c
│   ├── test_triplane_sample.c
│   ├── test_nerf_mlp.c
│   ├── test_marching_cubes.c
│   └── test_e2e_triposr.c
│
├── tools/
│   ├── ckpt_to_safetensors.py
│   ├── extract_golden.py
│   └── perf_benchmark.py
│
├── triposr_env/                 — gitignored: pinned TripoSR python repo + venv
│
├── debug/    test_vectors/    jpg_test/    images/
```

---

## 7. Data-flow contracts

Two opaque structs cross all module boundaries.

```c
/* iris.h — image I/O lives here */

typedef struct iris_image iris_image;
struct iris_image {
    int width;
    int height;
    int channels;     /* 3 = RGB, 4 = RGBA */
    uint8_t *data;    /* row-major, channel-interleaved */
};

iris_image *iris_image_load(const char *path);
int         iris_image_save(const iris_image *img, const char *path);
void        iris_image_free(iris_image *img);


/* lrm/lrm.h — LRM-specific API */

typedef struct lrm_model lrm_model;
typedef struct lrm_mesh  lrm_mesh;

typedef struct {
    int   mc_resolution;       /* default 256 */
    float density_threshold;   /* default 25.0 (post-exp) */
    float radius;              /* default 0.87 */
    int   bake_texture;        /* 0 = vertex colors only (v1) */
} lrm_infer_opts;

#define LRM_INFER_OPTS_DEFAULT { 256, 25.0f, 0.87f, 0 }

lrm_model *lrm_load(const char *model_dir);
void       lrm_free(lrm_model *m);

lrm_mesh  *lrm_infer(lrm_model *m, const iris_image *im,
                     const lrm_infer_opts *opts);

int        lrm_mesh_save_glb(const lrm_mesh *mesh, const char *path);
void       lrm_mesh_free(lrm_mesh *mesh);
```

No global state, no implicit threads, no malloc in inner loops.
Activations live in a single arena passed in by the caller.

---

## 8. Roadmap

The plan is sized for one focused senior engineer comfortable with C and
transformer internals. Multiply by 1.3–1.5 if either is new.

Each phase ends with a concrete, verifiable artifact and a passing test
(except where noted). No phase is "done" without its test.

| # | Phase | Deliverable | Test gate | Effort |
|---|---|---|---|---:|
| **0** | TripoSR architectural analysis | This document | — | ✅ done |
| **1** | Golden rig | `triposr_env/`, `tools/ckpt_to_safetensors.py`, `tools/extract_golden.py`, all 6 `.npz` checkpoints under `tests/golden/triposr/` | the `.npz` files load and replay in Python | 1 wk |
| **2** | Cleanup | Single commit removing all diffusion-specific files and the AdaLN/CFG/schedule paths from shared files | `make` builds a stub binary that prints "lrm: no model loaded" | 3 d |
| **3** | Restructure | `lrm/` dir created, `iris.h` reduced to minimal public surface, `lrm/lrm.h` skeleton added, Makefile updated to build `lrm/*.c` | `make` builds with `lrm_infer` stub returning NULL | 2 d |
| **4** | Kernels gap fill | Add **LayerNorm**, **GELU exact**, **GEGLU**, **grid_sample bilinear** to `iris_kernels.c` + `iris_shaders.metal`. Verify `iris_attention` supports cross-attn. Remove AdaLN paths | `test_kernels.c` parity vs NumPy reference, atol 1e-5 (f32) | 1.5 wk |
| **5** | Safetensors load + weight remap | `lrm/lrm_triposr.c` opens the converted safetensors, enumerates tensors, validates names/shapes/dtypes against `config.yaml` | `lrm info model.safetensors` prints the weight tree | 2 d |
| **6** | DINOv1 ViT-B/16 encoder | `lrm/lrm_vit_dino.c`: patch_embed + pos_embed + 12 blocks (LN + MHSA + LN + MLP-GELU) | parity vs `dino_tokens.npz`, atol 5e-4 | 2 wk |
| **7** | Triplane decoder | `lrm/lrm_triplane_decoder.c`: GroupNorm + proj_in + 16 BasicTransformerBlocks (self-attn + cross-attn + GEGLU + 3 LN) + proj_out + detokenize | parity vs `triplane_pre_upsample.npz` | 2.5 wk |
| **8** | Post-processor | `lrm/lrm_triplane_upsample.c`: either true ConvTranspose or `upsample_nearest + conv2d` equivalent (verify against golden) | parity vs `triplane.npz` (3×40×64×64) | 4 d |
| **9** | Triplane sample + NeRF MLP | `lrm/lrm_triplane_sample.c` (grid_sample, border, align_corners=False) + `lrm/lrm_nerf_mlp.c` | parity vs `density_64.npz` (262K queries on 64³ grid) | 1 wk |
| **10** | Marching cubes | `lrm/lrm_marching_cubes.c` (Lorensen–Cline 256-entry table) + vertex color re-query | parity vs golden mesh: vertices ±2%, faces ±2%, Chamfer < 1e-3 | 1.5 wk |
| **11** | GLB export | Vendor `cgltf.h`, write `lrm/lrm_mesh_export.c` (vertex colors only) | GLB round-trip opens in Blender / online viewer | 3 d |
| **12** | TripoSR glue + CLI | `lrm/lrm_triposr.c` config parsing, `main.c` subcommands `download / convert / infer / info` | `./lrmc infer image.png -o out.glb` end-to-end | 1 wk |
| **13** | Metal optimization | Targeted shaders for (a) decoder cross-attention (b) batched grid_sample (c) batched MLP over 256³ with early-termination on low-density voxels | benchmark e2e < 3 s on M-series, 512×512 input | 2 wk |
| **14** | BLAS path validation | Run on Linux x86_64 without Metal, profile BLAS GEMM in decoder | end-to-end runs, e2e < 8 s | 4 d |
| **15** | OpenLRM (optional) | Second model `lrm/lrm_openlrm.c`, validates the abstraction | OpenLRM golden parity | 1.5 wk |
| **16** | Background removal in C | Vendor a small segmentation model (candidates: RMBG-1.4, IS-Net, U2NetP — research phase first). Reuse existing kernels (conv2d, LN, GELU) | `lrm infer --auto-bg` matches a manually pre-processed reference | **3–6 wk** |
| **17** | Hardening + release | `README.md`, `SPEED.md`, `docs/ARCHITECTURE.md`, perf table, signed binaries, GitHub release | tagged release | 2 wk |

**MVP estimate (phases 1–14): ~13–14 weeks.** Phase 16 (bg removal) is
an independent track that adds 3–6 weeks.

### 8.1 Phase gates

Each phase ends with three checkboxes:

- ☐ Code merged via PR, reviewed by at least one external contributor
- ☐ Test gate (see table) passes on the CI matrix
- ☐ A 200-word entry added to `docs/CHANGELOG.md` documenting design
  choices

A phase that misses a checkbox blocks the next. No "we'll add tests
later" — the codebase becomes irrecoverable in months if we let that
slip.

---

## 9. Build & distribution

### 9.1 Build matrix

| Target | Compiler | Backend | Distribution form |
|---|---|---|---|
| `macos-arm64` | clang | Metal | universal binary |
| `macos-x86_64` | clang | BLAS (Accelerate) | x86_64 binary |
| `linux-x86_64` | gcc/clang | OpenBLAS / pure C | static binary |
| `linux-arm64` | gcc/clang | pure C | static binary |
| `windows-x86_64` | clang-cl | pure C | static .exe |

For Apple Silicon, the binary statically embeds the Metal shaders so
users never touch `xcrun`.

### 9.2 Releases

GitHub releases ship pre-built binaries for the matrix above. The
artifact is the binary plus a `weights/` tarball per model. A
`lrm.sha256` is published alongside; a `cosign` or `minisign` signature
follows from Day 1.

### 9.3 No Docker required

Docker is *supported* (a small `Dockerfile` is in the repo for users
who want it) but it is **not** the primary distribution channel.

---

## 10. Testing strategy — the golden-parity rig

The single most important component of this project, after the code
itself, is the test rig.

### 10.1 What we capture

For TripoSR we run the **reference PyTorch implementation** on a fixed
input and dump six `.npz` files under `tests/golden/triposr/`:

1. `input_512.npz` — preprocessed image `[3, 512, 512]` (post-rembg,
   85% rescale, gray composite, ImageNet norm)
2. `dino_tokens.npz` — encoder output `[1025, 768]`
3. `triplane.npz` — post-upsample triplane `[3, 40, 64, 64]`
4. `density_64.npz` — density+RGB on a 64³ query grid (262K queries)
5. `mesh.npz` — vertices, faces, vertex colors from marching cubes at
   64³ resolution
6. `output.glb` — the final GLB file

Each is a few MB at most. The full golden bundle is < 50 MB.

The reference run is reproducible with `tools/extract_golden.py` and
the pinned commit hashes from §3.

### 10.2 Per-module parity tests

Each module file ships with a test that loads the relevant golden
inputs, runs the C implementation, and `assert_allclose`s against the
golden output:

- f32 kernels (single-op): atol = `1e-4`, rtol = `1e-4`
- f32 deep networks (compound, >10 residual adds): atol = `4e-3`, rtol = `1e-4`
- bf16 kernels: atol = `5e-3`, rtol = `5e-3`

The deeper-network tolerance reflects the f32 floor for compound graphs.
Concretely (measured in Phase 7 on the 16-block triplane decoder, 3.15M
output floats, fp32 throughout): mean |err| is ~8e-5, the worst absolute
error is ~1e-2 but only on values of magnitude ~2700 (relative error 4e-6,
well below the f32 mantissa). Tightening below 4e-3 atol on the deep
networks would require selectively upgrading matmul/attention accumulation
to f64 - a Phase 13 / Phase 14 optimization, not a correctness gate.

A PR that breaks tolerance is blocked.

### 10.3 End-to-end parity

`test_e2e_triposr.c` runs the full pipeline at 64³ MC resolution and
compares the final mesh:

- Vertex count ±2%
- Face count ±2%
- Surface area ±1%
- Chamfer distance to reference mesh < `1e-3` (normalized units)

### 10.4 Per-layer parity (debug-only)

Per-layer dumps are *not* part of CI. When a per-module test fails,
`tools/extract_golden.py --per-layer <module>` regenerates intermediate
layer outputs for that module so the failure can be isolated. These
intermediates are not committed.

### 10.5 Performance test

`tools/perf_benchmark.py` reports cold-start, warm-load, and e2e
latency. Run on each release tag; results land in `docs/PERFORMANCE.md`
as a tracked table.

---

## 11. Performance targets

These are **commitment targets**. If we miss them by Phase 14, we
revisit the design before adding new models.

| Stage | Hardware | Target | Verified by |
|---|---|---|---|
| Cold start (exec → ready) | M3 Pro, Metal | < 1.0 s | `perf_benchmark.py` |
| End-to-end, 512×512 input | M3 Pro, Metal | < 3.0 s | `test_e2e_triposr.c` |
| End-to-end, 512×512 input | i7-12700, OpenBLAS | < 8.0 s | `test_e2e_triposr.c` |
| Peak RSS during inference | any | < 4 GB | `perf_benchmark.py` |
| Binary size, stripped | any | < 5 MB | release CI |

The PyTorch baseline numbers go in `docs/PERFORMANCE.md` so we are
honest about the speedup.

---

## 12. Critical technical decisions

### 12.1 Why C, not C++ or Rust

C is chosen for:
- Stable ABI with everything (Python ctypes, Go cgo, Node N-API, JNI)
- Minimal compile times on cold checkouts
- No template metaprogramming traps
- Predictable code generation when reasoning about perf

C++ subset (no STL, no exceptions) is acceptable in the Metal-side
files where RAII is convenient. Rust is rejected for the FFI cost when
callers want to embed `liblrm`.

### 12.2 Why one-shot ckpt → safetensors conversion

TripoSR's `model.ckpt` is a PyTorch pickle. We do **not** parse pickle
in C, for two reasons:

- Pickle = arbitrary code execution if mishandled. We do not want a
  C-side pickle parser in our supply chain.
- A pickle parser would be ~500 LOC for no runtime benefit. Users run
  conversion once at download time; the engine reads safetensors only.

The converter (`tools/ckpt_to_safetensors.py`) is ~30 lines and
documented in the README.

### 12.3 Why mmap'd safetensors

Safetensors is:
- The format most reference Python repos publish (after conversion)
- mmap-friendly with zero deserialization step
- Audited for safe loading

`tools/ckpt_to_safetensors.py` bridges the gap for models distributed
in other formats.

### 12.4 Why per-model files, not a generic graph

A generic "load any HuggingFace model" loader would be ten times the
code, and the abstraction tax would show up in every kernel call as a
pointer chase through a graph node. The LRM variants we target all use
*almost the same* code; the delta is captured in `lrm/lrm_<model>.c`
and is small enough to debug directly. When two model files contain
copy-pasted code, it gets lifted into a shared block.

### 12.5 Why marching cubes, not differentiable rendering

The use case is **export a mesh**, not optimize rendering parameters.
Marching cubes is mature, deterministic, single-pass, and works on the
density field LRMs naturally produce. Differentiable renderers
(nvdiffrast, gsplat) carry a torch dependency and solve a different
problem.

### 12.6 Threading model

Default: single-threaded. Each kernel call is on the caller's thread.

OpenMP is enabled at build time for the BLAS and pure-C backends
(`make omp`). On Apple Silicon, Metal naturally parallelizes inside
the GPU.

We do not spawn background threads. If the caller wants concurrency,
they spawn N inference threads themselves (`lrm_infer` is thread-safe
given separate `lrm_model` instances or shared model + per-thread
arena).

### 12.7 Quantization (deferred to v2)

The Day-1 path is f32 / bf16. Quantization is deferred until end-to-end
parity is stable. When added:

- int8 weight-only first (the easy win on memory bandwidth)
- int4 K-quant (llama.cpp-style) second, only after measuring quality
  loss on the Chamfer-distance test

Quantization MUST NOT degrade Chamfer distance by more than 5% vs f32.

---

## 13. Risks & open questions

| Risk | Mitigation |
|---|---|
| TripoSR repo or HF revision changes silently | Pinned commit hashes recorded in §3 and in `tools/extract_golden.py`. |
| Apple Metal shader development takes longer than expected | Phase 13 (Metal optimization) is sequenced *after* the pure-C + BLAS pipeline works end-to-end. We ship a working slow engine first. |
| Safetensors format evolves | Reader is feature-gated to `header_version >= 1`; we vendor the spec. |
| GLB writer (`cgltf`) lacks a feature we need | Out of scope for v1. If needed later, we either upstream a PR or write a 200-LOC GLB writer. |
| Marching cubes produces noisy artefacts on imperfect density fields | Day-1: classic MC + one Taubin smoothing pass. v2: evaluate dual contouring. |
| `iris_attention` may not natively support cross-attention with distinct K/V source | Verified in Phase 4 before relying on it. Worst case: add a tiny wrapper. |
| Background removal in C (Phase 16) drags timeline | Phase 16 is post-MVP. v1 ships with documented manual preprocessing. |
| ConvTranspose required for post-processor | Either implement true ConvTranspose or verify that `upsample_nearest + conv2d` is mathematically equivalent for the specific TripoSR configuration. |

---

## 14. Out-of-scope decisions

Parked for v2 or rejected outright:

- GPU backends other than Apple Metal (CUDA, Vulkan, ROCm)
- WebAssembly build
- A Python bindings layer beyond what `ctypes` already enables
- Distributed inference (irrelevant for ~500M parameter models)
- Online weight downloading from arbitrary URLs — `lrm download` is
  hardcoded to known HF repos only (security gate against
  typo-squatting)
- Text-to-3D (DreamFusion-style SDS optimization)
- Multi-view stereo pipelines
- Training. LRMengine is inference-only.
- Wrapping ONNX Runtime / TensorRT / torch::jit. Same dependency
  footprint we are trying to escape.

---

## 15. Open-source posture

- License: MIT
- Code style: a single `.clang-format` rules everything; CI rejects
  unformatted PRs
- Commit style: Conventional Commits (`feat:`, `fix:`, `perf:`,
  `docs:`) — helps generate release notes automatically
- Reviews: every PR needs one approval. Performance-sensitive PRs need
  a benchmark in the description showing before/after
- Issues triaged weekly. `good-first-issue` label maintained
- No CLA. DCO sign-off is sufficient (`Signed-off-by:` in commit)

---

## 16. Inspirations & references

- **[`antirez/iris.c`](https://github.com/antirez/iris.c)** — the
  fork's parent and the architectural philosophy of this project. Read
  it before contributing.
- **[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)** —
  the industrial reference for arena allocation, mmap'd weights, build
  matrices, and the ggml kernels layer.
- **[`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)** —
  same family, image-gen side. Useful for the multi-backend pattern.
- **[TripoSR technical report (arXiv:2403.02151)](https://arxiv.org/abs/2403.02151)** —
  the Day-1 target model.
- **[TripoSR repo](https://github.com/VAST-AI-Research/TripoSR)** — the
  reference Python implementation.
- **[`stabilityai/TripoSR` on HF](https://huggingface.co/stabilityai/TripoSR)** —
  weights and config.
- **[LRM paper (arXiv:2311.04400)](https://arxiv.org/abs/2311.04400)** —
  the model family origin.
- **[`facebook/dino-vitb16`](https://huggingface.co/facebook/dino-vitb16)** —
  the encoder backbone.
- **[`cgltf`](https://github.com/jkuhlmann/cgltf)** — vendored MIT
  single-file GLB writer.
- **Lorensen & Cline, 1987** — *Marching Cubes: A High Resolution 3D
  Surface Construction Algorithm*. Use the standard 256-entry case
  table from the paper directly.

---

## 17. The contract with future maintainers

If you take over this project, please honor:

1. **No new dependencies.** Vendor or write it yourself.
2. **No tests deferred.** Phase gates exist for a reason.
3. **Performance numbers in `docs/PERFORMANCE.md` are commitments.** A
   refactor that regresses them by more than 10% is reverted or the
   regression is explained in the PR description.
4. **Models are added as files, not as flags.** If a model needs
   special handling, that handling lives in `lrm/lrm_<model>.c`. The
   kernels stay model-agnostic.
5. **The README is the contract with users.** Whatever it claims, must
   work on a fresh clone with `make`.
6. **Decisions in §3 are durable.** Reopen only with explicit
   justification recorded in a new dated entry of the same section.

If you disagree with any of these, you are welcome to fork. They are
not negotiable inside this repo.

---

*Project: lrm.c — a fork of antirez/iris.c specialized for LRM-family
image-to-3D models. Initial author: marco. License: MIT.*
