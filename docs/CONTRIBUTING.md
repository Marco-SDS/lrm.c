# Contributing to lrm.c

Short version: keep it minimal, keep it tested, keep it fast. The
[LRMengine.md](../LRMengine.md) tenets are the ground truth — read
them before opening a PR that changes anything structural.

## Tenets (recap)

1. **No new dependencies.** Libc + Accelerate/OpenBLAS/Metal only.
   Vendor a single-file MIT/zlib library if you must, but not before
   trying to write the ~150 lines yourself. (The GLB writer is a
   reference point — we declined to vendor cgltf.)
2. **One file per responsibility.** The reader should be able to
   point at `lrm/lrm_<thing>.c` and know exactly which sub-system it
   speaks for.
3. **Models are data, the engine is code.** A new LRM variant is one
   file in `lrm/` + one case in the dispatcher. Architectural changes
   happen in the kernels layer.
4. **Correctness before speed.** Every module ships with a parity
   test against the canonical PyTorch reference. Optimizations land
   only when the test still passes.
5. **Three backends, equal first-class.** A change that helps Metal
   but regresses BLAS by >10 % is not accepted.

## Local dev loop

```bash
git checkout -b feat/<short-name>

# Build with the backend you care about (Accelerate is automatic on macOS).
make blas

# Run the full regression sweep before committing.
make test           # kernel parity (always cheap, ~1 s)
make test-dino      # DINO forward parity (~3 s)
make test-decoder   # decoder forward parity (~60 s)
make test-upsample  # post-processor parity (~3 s)
make test-density   # sampler + NeRF MLP on 64^3 (~10 s)
make test-mc        # marching cubes structural (~5 s)
make test-glb       # GLB writer structural + trimesh round-trip
make test-e2e       # end-to-end inference at 64^3 (~50 s)
```

Every change must pass `make generic` AND `make blas`. The MPS build
needs Apple Silicon hardware; if you have an M-series Mac, run
`make mps` too.

## Code style

- C99 with the GNU extensions clang/gcc allow by default. No C++.
- Prefer descriptive names over short ones in module APIs;
  one-letter names are fine inside hot loops.
- Static helpers go above the functions that use them.
- Every public function gets a `/** ... */`-style comment explaining
  shape contracts and pre/post-conditions. Look at
  `lrm/lrm_vit_dino.h` for the reference style.
- Error reporting: return -1 / NULL, set the string via
  `iris_set_error(...)`. Don't `abort()` and don't `errno`.
- No global mutable state except the last-error buffer in `iris.c`
  and the per-stage timing counter in `lrm/lrm.c`.

Format with a single space after commas and around binary operators;
brace style is K&R (`if (...) {` on the same line). There is no
`.clang-format` yet — match the existing files. We'll commit a
`.clang-format` in Phase 17 follow-up if formatting drift becomes a
problem.

## Adding a new kernel

Kernels live in `iris_kernels.{c,h}`. To add one (say, a new
activation function):

1. Add the declaration to `iris_kernels.h`, with a comment that
   states the input/output shapes and the PyTorch op it mirrors.
2. Add the implementation to `iris_kernels.c` next to peer kernels
   (activations grouped together, etc.).
3. Add a parity test to `tests/model/test_kernels.c`:
   - Generate the reference output in PyTorch (a small Python helper
     in the test docstring) and embed the result as
     `static const float ref_<name>[...]`.
   - Add `test_<name>()` that calls the kernel and uses the existing
     `compare_array(...)` helper.
4. Wire `test_<name>()` into `main()`.
5. Tolerance: f32 single-op atol=1e-5 is the baseline. If the kernel
   has natural compound error (e.g. a reduction over many elements),
   document the chosen tolerance in the test docstring.
6. Run `make test`. Then run the upstream regression sweep
   (`make test-dino`, `make test-decoder`, ...) — kernel-level
   changes can ripple.

If a kernel needs a Metal counterpart, add it to `iris_shaders.metal`
and the dispatch in `iris_metal.m`. Mark Metal kernels with the same
`atol=5e-3, rtol=1e-3` tolerance trimesh's MPS path uses; bf16
inputs get more slack (atol=5e-2). This is sequenced for Phase 13.

## Adding a new LRM model variant

The architecture lets you add a sibling to TripoSR with one new file
in `lrm/`. Walk-through:

1. **Pin the reference**: clone the upstream repo at a specific
   commit into `<modelname>_env/`, add it to `.gitignore`. Pin the
   HuggingFace weight revision too — include the SHAs in a comment
   block at the top of `lrm/lrm_<modelname>.h`.
2. **Inspect the safetensors**: convert the weights via a one-shot
   Python script in `tools/`, then `./lrmc info` to dump the tensor
   tree. Diff against TripoSR; if the new model shares modules
   (DINO encoder, NeRF MLP), reuse those `lrm/` files.
3. **Capture goldens**: extend `tools/extract_golden.py` (or add
   `tools/extract_golden_<name>.py`) to dump the same 6 boundary
   tensors used for TripoSR: preprocessed image, encoder tokens,
   decoder triplane, density grid, mesh, output GLB.
4. **Write the loader + glue**: `lrm/lrm_<modelname>.{c,h}` with the
   model-specific weight bindings, image preprocessing, and an init
   function that sets up the sub-modules. Mirror the layout of
   `lrm_triposr.{c,h}`.
5. **Add the `kind` enum**: `struct lrm_model` gains an `enum
   lrm_kind { LRM_KIND_TRIPOSR, LRM_KIND_<MODELNAME> }`. Update
   `lrm_load` and `lrm_free` in `lrm/lrm.c` to dispatch on it. The
   `lrm_infer` entry point also dispatches.
6. **Per-module parity tests**: add `tests/model/test_<modelname>_*.c`
   for each sub-stage that differs from TripoSR (mesh-construction
   tests go under `tests/geometry/`). Re-use the test structure
   (mmap goldens, run forward, compare with the same helper).
7. **End-to-end test**: extend `make test-e2e` (or add
   `make test-e2e-<modelname>`) to run the full pipeline at low
   resolution.

The kernels stay model-agnostic. Per the third tenet, do NOT add
model-specific branches inside `iris_kernels.c`.

## Performance changes

- Performance-sensitive PRs need a before/after comparison in the
  description. Use `LRM_TIMING=1 ./lrmc infer ...` for stage-level
  numbers; report mean of 3 runs.
- A change that regresses `make test-e2e` walltime by >10 % is
  blocked unless it improves correctness or memory in a way that
  warrants the trade-off.
- `SPEED.md` is the canonical baseline; update it in the same
  commit if your change moves any of the headline numbers.

## Git workflow

- One logical change per commit. Phase commits are large by their
  nature but should still tell a single story.
- Use the [Conventional Commits](https://www.conventionalcommits.org/)
  prefix style we already use in this branch:
  - `Phase N <description>` for plan-aligned phase commits.
  - `feat:` / `fix:` / `perf:` / `docs:` / `refactor:` for routine
    work between phases.
- Co-authored AI commits are fine; tag them with the standard
  `Co-Authored-By:` trailer.
- PRs: one reviewer approval minimum; performance-sensitive ones
  need the benchmark in the description.
- DCO sign-off (`Signed-off-by:`) is sufficient. No CLA.

## Issues and triage

- Mark known limitations in the README as you discover them.
- `good-first-issue` is reserved for items where the answer is in
  one of `lrm/*.c` and adding a parity test would catch the bug.

## When in doubt

Read [LRMengine.md](../LRMengine.md) sections 2 (philosophy) and 3
(decisions log) before opening a debate. Those are the ground rules;
they were chosen with explicit trade-off discussion and are not
re-litigated lightly.
