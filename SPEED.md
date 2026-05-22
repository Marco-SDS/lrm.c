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

This is the only CPU we have on hand; numbers below are measured on
2026-05-22 with commit `7778c79` (Phase 12 + Phase 14 timing).

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

### Resolution 256³ (16,777,216 MC queries, 75,918 vertices output)

| Stage                                  |    Time | % of total |
|----------------------------------------|--------:|-----------:|
| Preprocess                             |     7 ms |     0.0 % |
| DINO ViT-B/16 encoder                  |  2,693 ms |     3.2 % |
| Triplane decoder                       | 49,721 ms |    59.7 % |
| Post-processor                         |     5 ms |     0.0 % |
| Density grid (16.78M sample+MLP)       | 30,479 ms |    36.6 % |
| Marching cubes                         |    201 ms |     0.2 % |
| Vertex color re-query (75,918 verts)   |    150 ms |     0.2 % |
| GLB write                              |     3 ms |     0.0 % |
| **Total**                              |  **83.3 s** |  **100 %** |

GLB output sizes: 181 KB at 64³, 3.9 MB at 256³.

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

### Density grid scales linearly with MC res³

The cost is `mc_res³` × (one bilinear grid_sample + one tiny NeRF MLP
forward) per query. At 64³ this is 520 ms; at 256³ it's 30.5 s, almost
exactly 64× as expected.

The kernel hot path is `iris_grid_sample_bilinear` (CPU loops, not
BLAS) plus the 10 BLAS sgemms inside the NeRF MLP. The MLP is small
(120 → 64 → 4) so the GEMM aspect ratio is poor for cache reuse;
this is also where a tiled / parallel Metal implementation would
make the biggest delta.

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
| End-to-end 512×512          | < 3.0 s (M3 Pro, Metal)     | 50-83 s (i9-9880H, CPU)   | ❌ requires Phase 13 |
| End-to-end 512×512          | < 8.0 s (i7-12700, OpenBLAS)| ~50 s on slower i9 (CPU)  | ❌ likely ~25-30 s on i7-12700 |
| Peak RSS                    | < 4 GB                       | ~2.0 GB at 256³           | ✅ |
| Binary size, stripped       | < 5 MB                       | ~210 KB                   | ✅ ~25× headroom |

The two e2e targets were set for a Metal-accelerated backend
(Phase 13) and a faster x86 desktop. The current Intel laptop CPU
+ Accelerate path is BLAS-limited; closing the gap to either target
requires Phase 13's Metal kernels (decoder + grid_sample + tiled
MLP) on Apple Silicon, or roughly 2-3× more memory bandwidth on x86.

## What would move the needle

Ranked by expected impact at 256³:

1. **Metal sgemm for decoder Q/K/V/GEGLU** (Phase 13). At ~5-10× the
   sustained GFLOPS of Accelerate sgemm, this collapses the decoder
   to ~5-10 s and would close most of the gap to the 3 s M3 target.
2. **Tiled Metal compute for `iris_grid_sample_bilinear` + NeRF MLP
   over the MC grid** (Phase 13). At 256³ this is 30 s; a tiled GPU
   pass with early-termination on near-zero density values should
   get it under 5 s.
3. **Bigger BLAS** (e.g. OpenBLAS multi-threaded on x86). Each
   sgemm in the decoder is large enough to scale with cores; on
   an 8-core desktop OpenBLAS this would cut the decoder roughly
   in half (from ~50 s to ~25-30 s on i7-12700).
4. **bf16 weight cache** for the decoder. Reduces bandwidth pressure
   by 2× without changing parity (already pattern-validated in the
   inherited iris.c kernels). Modest win on CPU, larger win on
   memory-bound GPUs.
5. **Early-termination on low-density voxels for MC**. Skipping the
   ~50-70 % of voxels that are clearly empty would cut the density
   grid roughly proportionally. Heuristic and easy to add but only
   matters at 256³ (where the density grid is 37 % of total time).

Anything beyond these is in the noise.

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
