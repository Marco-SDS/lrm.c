# lrm.c — Performance notes

Per-stage walltime for the canonical TripoSR end-to-end inference
(`triposr_env/examples/robot.png` -> GLB) on the BLAS backend.
Profiled via `LRM_TIMING=1 ./lrmc infer ...` after Phase 12.

## Methodology

```bash
make blas
LRM_TIMING=1 ./lrmc infer triposr_env triposr_env/examples/robot.png \
    -o /tmp/lrm.glb --mc-resolution {64,256}
```

Stage timing is opt-in via the `LRM_TIMING` env var; the CLI then
prints walltime per pipeline stage to stderr.

## Baseline: Intel macOS + Accelerate (i9-9880H @ 2.3 GHz, 8 cores)

This is the only CPU we have on hand. The 64³ table is from 2026-05-22
(Phase 12/14); the 256³ table was re-measured on 2026-05-27 after Phase 19
(coarse-to-fine density) and Phase 20 (floater removal + gradient normals).

### Resolution 64³ (262,144 MC queries, 4,520 vertices output)

| Stage                                  |   Time | % of total |
|----------------------------------------|-------:|-----------:|
| Preprocess (foreground + 512 resize)   |    6 ms |     0.0 % |
| DINO ViT-B/16 encoder                  |  2,643 ms |     5.2 % |
| Triplane decoder (16 cross-attn blocks)| 47,221 ms |    93.5 % |
| Post-processor (ConvT 1024→40, 2×)     |    6 ms |     0.0 % |
| Density grid (262,144 sample+MLP)      |   520 ms |     1.0 % |
| Marching cubes                         |    2 ms |     0.0 % |
| Vertex color re-query (4,520 verts)    |    9 ms |     0.0 % |
| GLB write                              |    1 ms |     0.0 % |
| **Total**                              | **50.4 s** |  **100 %** |

### Resolution 256³ (coarse-to-fine density, ~75,600 vertices output)

Since Phase 19 the density grid is built coarse-to-fine (see below), so
only ~6 % of the 16.78M grid nodes are evaluated through sample+MLP. Phase
20 adds floater removal + analytic gradient normals, both negligible.

| Stage                                  |    Time | % of total |
|----------------------------------------|--------:|-----------:|
| Preprocess                             |     8 ms |     0.0 % |
| DINO ViT-B/16 encoder                  |  2,760 ms |     5.6 % |
| Triplane decoder                       | 43,700 ms |    88.3 % |
| Post-processor                         |     6 ms |     0.0 % |
| Density grid (coarse-to-fine, 5.9%)    |  2,510 ms |     5.1 % |
| Marching cubes                         |    238 ms |     0.5 % |
| Floater removal (union-find)           |      7 ms |     0.0 % |
| Gradient normals                       |     10 ms |     0.0 % |
| Vertex color re-query (75,630 verts)   |    125 ms |     0.3 % |
| GLB write                              |     5 ms |     0.0 % |
| **Total**                              |  **49.5 s** |  **100 %** |

Before Phase 19 the dense density grid took **30.5 s** here (36.6 % of an
83.3 s total); coarse-to-fine cut it to **2.5 s** (~12×) with a
bit-identical marching-cubes surface, dropping the e2e total from 83.3 s to
49.5 s on the same machine. GLB output sizes: 181 KB at 64³, ~3.9 MB at 256³.

## Where the time goes

### Triplane decoder dominates (60-93%)

The decoder is 16 blocks, each running on a (3072, 1024) token state.
Per block, the heaviest individual op is the GEGLU FFN projection
(3072 × 1024 -> 3072 × 8192 ≈ 26 GFLOPS per block, 16 blocks = 420
GFLOPS just for that projection). The Q/K/V + cross-attn + GEGLU
+ FF-out projections together total ~1.1 TFLOPS for the whole decoder.

On Accelerate sgemm we measure ~22 GFLOPS sustained on the
benchmark machine, putting the theoretical decoder floor at
~50 s — and that's where we are. The decoder is essentially
**at BLAS bandwidth limit on this hardware**.

### Density grid: coarse-to-fine (Phase 19)

The marching-cubes isosurface is a thin 2D shell, and the TripoSR density
field is band-limited at the triplane resolution (64×64 per plane), so it
carries no detail finer than ~R/64 voxels. `lrm_density_build_sparse`
exploits both facts: it evaluates a coarse ~64³ lattice, marks only the
blocks straddling the iso-threshold (plus a 1-block dilation and a relative
band for the steep `exp()` activation), exactly evaluates just those, and
trilinearly fills the rest. Every cell that can contain the surface lies in
an exactly-evaluated block, so the extracted mesh is **bit-identical** to a
dense evaluation (verified by `make test-density-sparse`).

At 256³ this evaluates 5.9 % of the 16.78M nodes — 30.5 s → 2.5 s (~12×).
At 64³ the block stride degenerates to 1 (coarse = full grid), so it
matches the dense cost (~520 ms); that's fine since the density stage is
~1 % of the 64³ total. The win scales with resolution, which is exactly
where dense evaluation hurt.

The per-query hot path is still `iris_grid_sample_bilinear` (CPU loops,
not BLAS) plus the 10 small BLAS sgemms inside the NeRF MLP; coarse-to-fine
simply runs ~16× fewer of them at 256³.

### Everything else is in the noise

DINO encoder (2.7 s) is dominated by the 12 GEMM-heavy ViT blocks;
similar BLAS efficiency to the decoder per FLOP. The remaining
stages (preprocess, upsample, MC, vertex colors, GLB write) sum to
under 250 ms regardless of resolution and are not worth optimizing
on CPU.

## Comparison to LRMengine.md §11 targets

| Stage                       | LRMengine target            | Measured                  | Status |
|-----------------------------|-----------------------------|---------------------------|--------|
| Cold start (exec → ready)   | < 1.0 s (M3 Pro, Metal)     | 0.05 s model load on CPU  | ✅ trivially met |
| End-to-end 512×512          | < 3.0 s (M3 Pro, Metal)     | 49-50 s (i9-9880H, CPU)   | ❌ requires Phase 13 |
| End-to-end 512×512          | < 8.0 s (i7-12700, OpenBLAS)| ~50 s on slower i9 (CPU)  | ❌ likely ~22-25 s on i7-12700 |
| Peak RSS                    | < 4 GB                       | ~2.0 GB at 256³           | ✅ |
| Binary size, stripped       | < 5 MB                       | ~210 KB                   | ✅ ~25× headroom |

The two e2e targets were set for a Metal-accelerated backend
(Phase 13) and a faster x86 desktop. The current Intel laptop CPU
+ Accelerate path is BLAS-limited; closing the gap to either target
requires Phase 13's Metal kernels (decoder + grid_sample + tiled
MLP) on Apple Silicon, or roughly 2-3× more memory bandwidth on x86.

## What would move the needle

Ranked by expected impact at 256³ (the decoder is now ~88 % of the total
after the density stage was collapsed in Phase 19):

1. **Metal sgemm for decoder Q/K/V/GEGLU** (Phase 13). At ~5-10× the
   sustained GFLOPS of Accelerate sgemm, this collapses the decoder
   to ~5-10 s and would close most of the gap to the 3 s M3 target.
   With density already cheap, this is now the *only* large lever left.
2. **Bigger BLAS** (e.g. OpenBLAS multi-threaded on x86). Each
   sgemm in the decoder is large enough to scale with cores; on
   an 8-core desktop OpenBLAS this would cut the decoder roughly
   in half (from ~44 s to ~22-25 s on i7-12700).
3. **bf16 weight cache** for the decoder. Reduces bandwidth pressure
   by 2× without changing parity (already pattern-validated in the
   inherited iris.c kernels). Modest win on CPU, larger win on
   memory-bound GPUs.
4. **Tiled Metal compute for `iris_grid_sample_bilinear` + NeRF MLP**
   (Phase 13). Now only ~2.5 s on CPU after coarse-to-fine, so this
   is no longer urgent, but a GPU pass would still help at 512³.

DONE — **coarse-to-fine density (Phase 19)**. This was previously listed
here as "early-termination on low-density voxels"; it landed as a
band-limited octree-style refinement and cut the 256³ density stage from
30.5 s to 2.5 s. Anything beyond the items above is in the noise.

## Repro

```bash
make blas
LRM_TIMING=1 ./lrmc infer triposr_env triposr_env/examples/robot.png \
    -o /tmp/lrm_64.glb  --mc-resolution 64
LRM_TIMING=1 ./lrmc infer triposr_env triposr_env/examples/robot.png \
    -o /tmp/lrm_256.glb --mc-resolution 256
```

Add to a `docs/PERFORMANCE.md` table on each release tag (Phase 17
deliverable; not yet wired into release CI).
