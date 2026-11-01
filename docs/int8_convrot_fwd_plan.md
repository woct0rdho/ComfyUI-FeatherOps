# Int8 ConvRot Forward Kernel on gfx1151

Scope: the existing int8 ConvRot weights (already rotated, per-output-channel scale, G=256) and the existing numerics contract `C = (xH)(WH)^T * s_a * s_b`, bf16 out. Only the gfx1151 HIP path. The baseline is comfy-kitchen's `gemm_wmma_kernel<MmaInt8, EpiRowwise, bf16, 128, 128, 128, 4, 2, 2, 4>` as selected by `launch_gemm_wmma` on this part.

What shipped: a runtime-selected config table (`launch_int8_gemm_cfg_kernel` in `ops/gemm_int8.hip`) plus a shape heuristic in `backends/hip/__init__.py` on top of the same kernel. It buys 1.02-1.11x per model block over the shipped schedule (block-weighted: Krea 2 1.05x, H3 1.06x, Z-Image 1.05x, MOSS 1.05x, Anima 1.10x, Klein 9B 1.02x, Qwen-Image 2.1 1.03x), i.e. 37-42 TFLOPS of a ~50 TFLOPS MMA-issue peak on the big cells. Every id keeps the reduction order, so the output is bit-identical to the previous release. `tests/test_int8_gemm_cfg.py` enforces that.

The ceiling is what kept this small: with all memory traffic, barriers and stores removed the kernel still needs 79% of the pure MMA time, so no memory-side change could ever have recovered the 15-30% the first version of this plan assumed. The sections below are the profile, the ids and heuristic, the full per-model results, what was accepted, what was rejected, and what is left.

## Where the time goes

Kernel-level profile (rocprofv3, 4112x6144x6144, real weights):

| kernel | time | VGPR | LDS | threads | waves/SIMD | L2 hit | LDS bank conflict |
|---|---|---|---|---|---|---|---|
| CK int8 (ConvRot) | 7.85-7.95 ms | 224 | 34816 | 256 | 6 (37.5%) | 32.2% | 0.0% |
| CK fp16 | 10.96 ms | 224 | 34816 | 256 | 6 | 24.1% | 0.0% |
| hipBLASLt bf16 | 15.32 ms | 256 | 65024 | 128 | 2 | 38.5% | 47.4% |
| ConvRot producer | 0.46-0.70 ms | 32 | 36964 | 768 | - | 50.2% | 53.5% |

Rooflines measured on this machine: WMMA is 32 cycles per 16x16x16 instruction for fp16, bf16 and i8 alike (55 TFLOP/s at 2.68 GHz, ~50 at the ~2.5 GHz sustained clock). DRAM read 241 GB/s, copy 212 GB/s.

Why the kernel is not memory-bound, even though the traffic model says it should be: at 128x128x128-byte tiles, 1584 blocks and 48 K-tiles it issues 2.49 GB of L2 requests, and the measured 32.2% hit rate means 1.69 GB from DRAM in 7.95 ms = 212 GB/s = 88% of the read roof. That looks decisive but is a coincidence of the tile size. Removing costs from the same kernel shows what is actually exposed:

| variant (4112x6144x6144) | ms | delta |
|---|---:|---:|
| full | 8.383 | - |
| no barriers | 8.196 | -2.2% |
| 7/8 of the C stores removed | 8.175 | -2.5% |
| no barriers + no stores | 7.990 | -4.7% |
| no A/B refill, no barriers, no stores | 7.678 | -8.4% |

With every memory and sync cost removed the kernel still needs 7.678 ms = 40.4 TFLOPS, ~79% of the 51 TFLOPS WMMA-issue peak at the sustained clock. The rest is the LDS-read -> MMA dependency chain plus issue overhead. So the whole memory-side budget was 8.4%, and occupancy cannot be raised either (240-allocated VGPR and 34.8 KB LDS both cap at 6 waves/SIMD).

The 32.2% L2 hit rate is also below what the shipped block order should give: 60 resident blocks cover 4 M-tiles x 15 N-tiles, i.e. 19 distinct tiles per K slice against 60 blocks' worth of requests, which is ~68% if the blocks stayed in phase. They drift by more than the ~4 K-slice window a 2 MB L2 holds, which is why neither an L2 supertile nor a streaming hint recovered much (see the rejected list).

Per-module split at 1024x1024 (M=4112), real Krea 2 weights, shipped schedule:

| layer | bf16 | int8 ConvRot | int8 GEMM | rotate+quant | speedup |
|---|---|---|---|---|---|
| 6144x6144 (wq/wo/gate) | 15.4 ms | 8.4 ms | 7.9 | 0.46 | 1.83x |
| 1536x6144 (wk/wv) | 4.15 ms | 2.54 ms | 2.08 | 0.46 (18%) | 1.63x |
| 16384x6144 (up/gate) | 41.3 ms | 21.3 ms | 20.8 | 0.45 | 1.94x |
| 6144x16384 (down) | 41.3 ms | 22.2 ms | 21.0 | 1.24 | 1.86x |

The heuristic trims 1-9% off the GEMM column, moving those four layers to 37.2 / 31.5 / 39.7 / 36.4 TFLOPS, so int8 sits at 1.63-1.98x of bf16 per linear. A whole Krea 2 block runs at 36.1 -> 37.8 TFLOPS, i.e. 5.0 -> 2.7 s of linear work per denoising step over 28 blocks. The same int8-vs-bf16 comparison on the other families is 1.80-1.92x (Krea 2 at both sizes), 1.71-1.74x (H3), 1.87x (Qwen-Image 2.1) and 1.45-2.51x on the 192/16/1-row tail cells. The producer ("rotate+quant") is 4-6% of a plain module and 18% of the narrow `1536x6144` one. Across a Krea 2 block it is 3.5 ms of 95 ms.

The producer's own profile: 163 GB/s against the 241 GB/s roof (75 MB moved in 0.46 ms at M=4112, K=6144) and `LDSBankConflict = 53.5%`, caused by the FHT sub-buffer being indexed `lane * 4`, so every exchange stage is a 4-way conflict. Because it is already at 68% of the roof, the whole item is bounded at ~1.3x on 3.7% of a block (~0.9%), and ~10% of a fused-activation cell - see the open list.

## The kernel now: config ids and the shape heuristic

All ids keep the same reduction order, so `torch.equal` against id 0 holds for every shape (14 shapes x 26 ids x bias on/off, including tile counts that do not divide. Plus six real Kroma layers against the previous release). Geometry per id:

| id | tile | BKB | warps (TM,TN) | threads | LDS | pad | C store | A layout | schedule | role |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 128x128 | 128 | 4x2 (2,4) | 256 | 34.8 KB | 8 | normal | row-major | `kGroupM=4` | shipped baseline, also the `M<=64`/GEMV path |
| 1 | 128x128 | 128 | 4x2 | 256 | 36.9 KB | 16 | streaming | row-major | supertile | measured, off by default |
| 2 | 128x128 | 128 | 4x2 | 256 | 36.9 KB | 16 | streaming | row-major | supertile + coalesced staging | measured, off |
| 3 | 128x256 | 128 | 4x4 (2,4) | 512 | 52.2 KB | 16 | streaming | row-major | supertile | measured, off (streaming costs it ~1%) |
| 4 | 256x128 | 128 | 8x2 (2,4) | 512 | 52.2 KB | 16 | streaming | row-major | supertile | rejected: collapses above 8192 rows |
| 5 | 128x192 | 128 | 4x2 (2,6) | 256 | 43.5 KB | 16 | streaming | row-major | supertile | rejected: slower everywhere |
| 6/9/15 | 64x64 | 128/64 | 2x2 (2,2) | 128 | 17-35 KB | 8-16 | mixed | row-major | supertile | small-M experiments, all slower than id 0 |
| 7 | 128x128 dual-M | 128 | - | 512 | 51 KB | 8 | normal | tiled only | legacy | gfx12 kernel, needs tiled A, not faster here |
| 8 | 256x256 | 64 | 4x4 (4,4) | 512 | 37 KB | 16 | streaming | row-major | supertile | rejected: slower |
| 10/11/12/13 | 128x128 | 128 | 4x2 | 256 | 36.9 KB | 8/16 | mixed | row-major | mixed | isolation ids for pad/stream/supertile. 12 = pad 16, no stream, no supertile |
| 27 | 128x128 | 128 | 4x2 | 256 | 36.9 KB | 16 | streaming | row-major | legacy order | shipped for `64 < M <= 1024` |
| 28 | 128x256 | 128 | 4x4 | 512 | 52.2 KB | 16 | normal | row-major | legacy order | shipped for mid-M and partial-N |
| 128 (=29 + tiled A) | 128x256 | 128 | 4x4 | 512 | 52.2 KB | 16 | normal | tiled `[m_tile,k_tile,128,128]` | legacy order | shipped for `M >= 4096` |
| 112 | 128x128 | 128 | 4x2 | 256 | 36.9 KB | 16 | normal | tiled | legacy order | measured, no longer used by the heuristic |
| 20-26 | 128x128 | 128 | 4x2 | 256 | - | 16 | - | row-major | - | ablation ids: refill/barrier/store switches (`flags` bits 0-3), wrong results, timing only |
| 31/32/33 | 128x256 | 128 | 4x4 | 512 | 52.2 KB | 16 | streaming | tiled | legacy order | `flags` bits 4/5: `s_setprio`, ifetch distance 3 lines |

`int8_gemm_cfg(...)` takes the id plus (gm, gn) for the supertile order and a tiled-A flag. `hip._set_int8_gemm_config(id)` forces an id for A/B runs and nothing in the model path sets it. The tiled activation layout stores fixed 128x128-byte tiles, so it is only defined for BKB=128: `int8_cfg_supports_tiled()` rejects the other ids and the test suite covers that.

The heuristic (`_int8_auto_cfg`, gfx1151 only. Every other architecture keeps the shipped `launch_gemm_wmma` selection, which carries the gfx12/gfx117 schedules):

| condition | id | basis |
|---|---|---|
| `M <= 64` | 0 | 64x64 tile. The 128-row tiles waste 2-8x the MMA work at these row counts |
| `64 < M <= 1024`, `K < 2048` | 28 | too few K-steps to amortise a 128x128 tile's barriers (Anima cross-attn k/v) |
| `64 < M <= 1024` | 27 | too few tiles for a 256-wide tile or the tiled-A pass to pay off |
| `K >= 8192 && N <= 8192` | 12 | K-dominant: the weight stream dominates, the narrow tile wins by up to 6% |
| `K` a power of two, `K >= 4096`, `N >= 8192` | 12 | power-of-two row stride plus a 256-wide tile aliases the L2 sets (the B panel is exactly 1 MB at K=4096). Measured -2 to -3% for the narrow tile |
| `N % 256 != 0` | 28 | a partial 256-wide N tile still beats 128x128: the tail waste is smaller than the extra A traffic |
| `M < 4096` | 28 | too few M tiles to fill a resident group with a 256-wide tile |
| else | 128 | 128x256 + tiled A. The bulk of the model flops |

Two known misses, both left to the future autotune rather than fitted: H3 `fc2` (`5376x14336`, K-dominant) is 2% faster on id 128, and the synthetic `path.m512` cell (`512x4096x4096`) is 3% faster on id 12. Every other measured cell is at or within noise of its best id.

A streaming C store (`glc slc dlc`, SLC=1 = GL2 STREAM) and the L2 supertile are implemented but off the default path: both measured within noise once the launch covers whole block groups, and streaming costs the 128x256 tile ~1%.

## Results

Achieved TFLOPS per cell (`2*M*N*K / time`), shipped schedule (id 0) against the heuristic. Absolute TFLOPS drift ~3% between sessions with the clock state, so compare ratios and the two columns of a row rather than across rows. Tier A is 5 rounds x 4 iterations, Tier B 3 x 3, ids interleaved inside one process, median of per-round medians. A model's rows are ordered by size: the first group is the cheaper M, the second (where present) the larger one. `=` means both columns run the same id, so the difference is measurement noise. `per block` rows are the effective TFLOPS of one block's mixture (total MACs over total time), which is the number that moves a diffusion step.

| model | cell (M x N x K) | shipped (id 0) TFLOPS | heuristic TFLOPS | speedup |
|---|---|---:|---:|---:|
| Krea 2 | 4112x6144x6144 | 35.8 | 37.2 | 1.040x |
|  | 4112x1536x6144 | 28.9 | 31.5 | 1.091x |
|  | 4112x16384x6144 | 37.3 | 39.7 | 1.064x |
|  | 4112x6144x16384 | 35.8 | 36.4 | 1.018x |
|  | 9232x6144x6144 | 36.4 | 38.8 | 1.066x |
|  | 9232x1536x6144 | 30.1 | 32.7 | 1.086x |
|  | 9232x16384x6144 | 37.6 | 40.7 | 1.081x |
|  | 9232x6144x16384 | 36.8 | 37.3 | 1.013x |
| | per block: 3x6144x6144 + 2x1536x6144 + 2x16384x6144 + 1x6144x16384, 4112 rows | 36.1 | 37.8 | 1.048x |
| | per block: same mixture, 9232 rows | 36.7 | 39.0 | 1.061x |
| Qwen-Image 2.1 | 4112x4096x4096 | 31.2 | 32.2 | 1.032x |
|  | 4112x24576x4096 | 34.8 | 36.0 | 1.035x |
|  | 4112x4096x12288 (swiglu) | 32.4 | 32.9 | 1.015x |
|  | 9232x4096x4096 | 32.3 | 35.0 | 1.084x |
|  | 9232x24576x4096 | 36.0 | 37.0 | 1.029x |
|  | 9232x4096x12288 (swiglu) | 33.7 | 34.4 | 1.021x |
| | per block: 4x4096x4096 + 24576x4096 + 4096x12288, 4112 rows | 33.1 | 34.0 | 1.029x |
| | per block: same mixture, 9232 rows | 34.2 | 35.7 | 1.044x |
| Z-Image | 4128x11520x3840 | 36.8 | 39.0 | 1.060x |
|  | 4128x3840x3840 | 34.6 | 35.8 | 1.035x |
|  | 4128x10240x3840 | 36.8 | 39.6 | 1.076x |
|  | 4128x3840x10240 | 35.6 | 36.5 | 1.025x |
|  | 9248x11520x3840 | 37.7 | 39.8 | 1.057x |
|  | 9248x3840x3840 | 34.7 | 36.5 | 1.051x |
|  | 9248x10240x3840 | 36.7 | 38.9 | 1.060x |
|  | 9248x3840x10240 | 35.8 | 36.5 | 1.020x |
| | per block: one of each, 4128 rows | 36.2 | 38.0 | 1.051x |
| | per block: one of each, 9248 rows | 36.5 | 38.2 | 1.046x |
| H3 | 5376x21504x5376 | 39.1 | 42.0 | 1.072x |
|  | 5376x5376x7168 | 36.5 | 38.8 | 1.063x |
|  | 5376x28672x5376 | 38.9 | 42.2 | 1.087x |
|  | 5376x5376x14336 (swiglu) | 36.6 | 37.2 | 1.015x |
|  | 25344x21504x5376 | 38.7 | 41.4 | 1.070x |
|  | 25344x5376x7168 | 37.0 | 39.3 | 1.063x |
|  | 25344x28672x5376 | 38.6 | 41.2 | 1.069x |
|  | 25344x5376x14336 (swiglu) | 35.9 | 36.3 | 1.012x |
| | per block: one of each, 5376 rows | 38.2 | 40.7 | 1.064x |
| | per block: one of each, 25344 rows | 37.9 | 40.0 | 1.056x |
| MOSS-Image | 4112x9216x3072 | 34.6 | 36.5 | 1.056x |
|  | 4112x3072x3072 | 31.0 | 34.0 | 1.097x |
|  | 4112x16384x3072 | 35.0 | 36.9 | 1.055x |
|  | 4112x3072x8192 (situ) | 28.7 | 29.5 | 1.028x |
|  | 9232x9216x3072 | 35.3 | 36.6 | 1.036x |
|  | 9232x3072x3072 | 32.0 | 34.4 | 1.075x |
|  | 9232x16384x3072 | 36.7 | 38.6 | 1.053x |
|  | 9232x3072x8192 (situ) | 30.8 | 31.2 | 1.011x |
| | per block: one of each, 4112 rows | 32.9 | 34.6 | 1.052x |
| | per block: one of each, 9232 rows | 34.5 | 35.8 | 1.040x |
| Anima | 4096x2048x2048 (self q/k/v/out) | 24.3 | 29.0 | 1.193x |
|  | 512x2048x1024 (cross k/v) | 16.6 | 18.0 | 1.088x |
|  | 4096x8192x2048 | 31.4 | 34.6 | 1.101x |
|  | 4096x2048x8192 | 29.8 | 29.9 | 1.004x |
|  | 9216x2048x2048 | 24.3 | 28.7 | 1.183x |
|  | 9216x8192x2048 | 33.1 | 35.2 | 1.064x |
|  | 9216x2048x8192 | 31.4 | 31.6 | 1.005x |
| | per block: 5x2048x2048 + 2x2048x1024 + mlp.0 + mlp.2, 4096 rows | 27.7 | 30.6 | 1.108x |
| | per block: same mixture, 9216 rows | 28.5 | 31.3 | 1.096x |
| Klein 9B | 4096x12288x4096 (double qkv) | 33.9 | 35.2 | 1.037x |
|  | 512x12288x4096 (double qkv, text) | 31.3 | 34.4 | 1.098x |
|  | 4096x4096x4096 (double proj) | 30.8 | 32.6 | 1.058x |
|  | 512x24576x4096 (double mlp.0, text) | 34.4 | 35.1 | 1.021x |
|  | 4096x4096x12288 (double mlp.2) | 34.1 | 34.3 | 1.006x |
|  | 512x4096x12288 (double mlp.2, text) | 29.4 | 29.2 | = |
|  | 4608x36864x4096 (single linear1) | 36.8 | 37.7 | 1.024x |
|  | 4608x4096x16384 (single linear2) | 34.6 | 35.2 | 1.016x |
|  | 9216x12288x4096 | 34.8 | 35.5 | 1.019x |
|  | 9216x4096x4096 | 32.0 | 34.6 | 1.080x |
|  | 9216x4096x12288 | 34.4 | 35.5 | 1.030x |
|  | 9728x36864x4096 | 36.4 | 37.1 | 1.020x |
|  | 9728x4096x16384 | 35.1 | 35.4 | 1.009x |
| | per block: the eight double-block linears, image + text rows | 33.9 | 34.9 | 1.031x |
| | per block: single linear1 + linear2, 4608 rows | 36.1 | 36.9 | 1.021x |
| | per block: 8 double + 24 single blocks, 4096/4608 rows | 35.5 | 36.4 | 1.024x |
| | per block: same, 9216/9728 rows | 35.7 | 36.4 | 1.020x |
| path cells | 1x36864x6144 (GEMV, tproj) | 0.4 | 0.4 | at the DRAM roof |
|  | 16x6144x6144 | 4.1 | 4.0 | = |
|  | 512x4096x4096 | 31.2 | 31.6 | = |
|  | 192x2560x2560 | 21.5 | 22.7 | 1.056x |
|  | 2048x6144x6144 | 37.3 | 38.7 | 1.037x |
|  | 3072x6144x6144 | 36.5 | 38.0 | 1.041x |
|  | 4096x3456x1152 | 29.6 | 29.5 | = |
|  | 2048x4304x1152 | 31.5 | 32.2 | 1.022x |
|  | 16x4096x4096 | 4.9 | 4.9 | = |
|  | 16x24576x4096 | 2.2 | 2.2 | = |
|  | 16x4096x12288 (swiglu) | 2.7 | 2.5 | = |

Klein's `double mlp.0` at image rows is the same `(24576, 4096)` shape as Qwen-Image 2.1's `gate_up` and reuses its numbers. `double mlp.0/proj` at text rows reuse the 512-row cells. Klein's per-step figure is an approximation from the measured cells (8 double blocks x the eight double-block cells, 24 single blocks x the two single cells). Its single-block cells are the only int8 cells at M=4608/9728 and the two largest kernels in the whole grid (35.7 / 15.9 ms per call at 39 TFLOPS). With 24 of each they are ~40% of Klein's flops, so they deserve their own id if a tile change is ever chased.

The fused-activation cells (`swiglu` on Qwen-Image 2.1 `img_mlp.out` and H3 `fc2`, `situ` on MOSS `ffn_down`) gain least (1.004-1.028x): their producer reads a 2K-wide row, which is outside the GEMM schedule. For `h3.fc2.720p` that is 1.8 GB of a 107 ms cell.

## Accepted optimizations

| change | mechanism | measured | id |
|---|---|---|---|
| LDS pad 16 | keeps 16-byte row alignment, so a fragment read is one `ds_read_b128` instead of four `ds_read_b32` and a staged chunk one `ds_write_b128` instead of two `ds_write_b64`. The two-way conflict pad 16 causes on a duplicated 16-lane read is exactly the two cycles that read needs anyway, so no XOR swizzle is needed | -1.5% to -3% on every cell | all new ids |
| 128x256 tile | doubles the MACs per byte of A, and 512 threads with 52.2 KB LDS still fits 2 blocks/WGP (8 waves/SIMD by LDS) | -3% on the wide-N cells, up to -6% on the narrow `1536x6144` row | 28/128 |
| tiled-A activation layout | the ConvRot producer already emits `q[m_tile, k_tile, 128, 128]`. `launch_int8_gemm_a_tiled128_kernel` already reads it. Only `_use_tiled_int8()` gated it to gfx12. Tiled A is a global-memory layout, independent of the WMMA operand layout. A wave instruction then reads one dense 512 B span instead of four 128 B transactions | up to -9% (1536x6144), 0 to -4% elsewhere | 128 |
| per-shape config table + heuristic | pick tile/pad/tiled-A/K-block per `(M, N, K)` instead of one schedule | the ratios in Results | `_int8_auto_cfg` |
| tiled-A x BKB guard | the tiled layout is only defined for BKB=128. Combining it with another K block silently reads the wrong A (measured rel 1.42 error) | correctness, not speed | `int8_cfg_supports_tiled` |
| `flags` diagnostic bits | refill/barrier/store ablations plus `s_setprio`/ifetch-distance switches, constant-folded away on the shipped path | produced the 8.4% ceiling measurement | 20-26, 31-33 |
| bit-identity test | `tests/test_int8_gemm_cfg.py`: 14 shapes x 26 ids x bias on/off, non-dividing tile counts included. Also checked against the previous release on six real Kroma layers | caught the supertile launch bug | - |

Verification status: `tests/test_int8.py`, `test_hip_dispatch.py`, `test_convrot_unpacked_quantization.py`, `test_int8_input_act.py`, `test_int8_residual.py` and `test_int8_gemm_cfg.py` give 249 passed and one pre-existing failure, `test_int8_linear_correctness[hip]`, which fails identically with the unmodified `_C.abi3.so` (its 1% tolerance is tighter than the eager and HIP backends agree at that shape). fp16 output can differ by 1 ulp between ids because the epilogue instantiations contract the scale multiply differently. The model path is bf16 and is bit-identical.

## Tested and rejected

| item | why it is not shipped |
|---|---|
| 2D L2 supertile order | resident blocks are 60 (34 KB tile) / 40 (52 KB), which floors `GM + GN` at ~16 distinct tiles per K slice against 19 for the shipped `kGroupM=4` order, and the refill it optimises is only 8.4% of runtime. The one measured "6% win" was a launch bug: the grid was sized `mt*nt` instead of `ceil(mt/gm)*ceil(nt/gn)*gm*gn`, so groups that did not divide the tile counts skipped their tail tiles. After the fix it is within noise. Implemented (ids 1/2/3), off by default |
| streaming C stores (`glc slc dlc`) | within noise (+/-1%) on the 128x128 tile and ~1% negative on the 128x256 tile. Implemented, off by default |
| streaming A/B loads | impossible on gfx11: every non-tied FLAT load carries a hardcoded zero cache-policy operand in LLVM (`FLAT_Load_Pseudo`), so `!nontemporal` is dropped. Inline asm is unsafe (no `vmcnt` accounting, and LLVM overlaps the vector result registers). A buffer-descriptor load would need 32-bit MUBUF addressing for <=1% |
| BKB=192 (fewer barriers) | 3-10% slower: 2 blocks/WGP, 12-warp barrier groups and a coarser K tail outweigh 32 fewer barriers |
| 256x128, 128x192, 256x256/BKB64, 64x64 | slower on every cell. 256x128 collapses to 23 TFLOPS above 8192 rows |
| 128x512 (halve A traffic again) | infeasible: 92 KB LDS at BKB=128 exceeds the 64 KB per-workgroup cap, and at BKB=64 `TileStager` needs each thread to own a whole 16-byte chunk (512 chunks for 1024 threads) |
| dual-M kernel (gfx12) | its A indexing already assumed the tiled layout, so it is only reachable through the tiled ids. Not faster than 128x256 here |
| epilogue scale hoist | measured identical to the per-element form (8.496 vs 8.500 ms): the compiler already keeps the scales, so the 2.5% store ablation is store cost, not load cost. Reverted to keep bit-identity |
| `s_setprio`, ifetch distance 3 lines | +0.5-0.8% for setprio, inside the run-to-run spread, even though FeatherOps measured +5% for the same knob on their kernels and the gfx1151 reference records instruction-fetch stalls from long (96-bit) encodings that the prefetch distance targets. Both remain as `flags` bits |
| producer (ConvRot) layout rewrite | the producer is already at 163 GB/s of the 241 GB/s roof, so removing its 53.5% bank conflicts is bounded at ~1.3x on 3.7% of a block, and it would change the rotation's rounding (every activation currently matches the previous release bit for bit) |
| split-K / stream-K | the big shapes are within 8.4% of their memory-free time, so marching K together cannot pay. The GEMV path is already at the DRAM roof and the 16-row prefix cells are a one-shot ~2 ms per generation. If it is ever revisited for the low-block-count shapes: split-K with `S = clamp(next_pow2(40/blocks), 2, 8)`, fp32 partials in a workspace plus a deterministic two-pass reduce (or a semaphore fixup), `blocks < 2*WGP`, gate `rel L2 <= 1e-3` against full-K |
| exact-K bodies | not implemented: many instantiations for <=3% once the compute-side gap is attacked. The shape would be one `K`-templated body per whitelisted K ({1024, 1152, 2560, 4096, 5120, 5376, 6144, 7168, 8192, 12288, 14336, 16384, 25600}) plus exact M/N bounds to drop the tail checks, as torch-ggml-ops did for 9-22% on their kernels |
| fp8 requantization | out of scope: e4m3 holds only 4-bit integers exactly, so it would change outputs, and CK's gfx11 fp8 path widens to bf16 anyway (no fp8 WMMA on RDNA3.5) |

Lessons worth keeping:
- Do not trust a kernel win before the launch is proven exhaustive. The supertile's 6% was a skipped-tile bug that the timing harness could not see. A bit-identity test against the shipped schedule did. Keep `tests/test_int8_gemm_cfg.py` in any future tuning loop.
- The DRAM-traffic model misled: 88% of the read roof looked decisive, yet removing all refill traffic buys 8.4%. Measure the exposed refill (`flags` bits 0-3) before optimising for bandwidth.
- Benchmark discipline: interleave ids inside one process, compare medians of per-round medians, and do not gate on a fixed percentage - clock state alone swings the same kernel by 20-30%. A measured 0.5-0.8% did not survive as a shipping decision.
- Occupancy is capped at 6 waves/SIMD by VGPR (224 used -> 240 allocated) and LDS. Do not chase it, and do not grow the register tile (16 accumulators = 128 VGPR alone) - it risks spills for reuse the LDS bandwidth does not need.
- Do not spend effort on the epilogue's arithmetic: hoisting its scale loads measured identical, and ggtensile's ablation replaced their whole epilogue with an integer sink and got 20% slower. int8 WMMA is not faster than bf16 per instruction (32 cycles each), so only bytes per MAC matter.
- The tiled activation layout is tied to BKB=128. Guard any new id against it.
- Do not add stage-4 staging (rejected twice upstream): without `vmem-to-lds` on gfx1151, deeper staging costs VGPRs and LDS without removing a barrier.

## Still open

- Producer (worth ~0.9% of a Krea 2 block, ~10% of a fused `fc2` cell): the FHT sub-buffer is indexed `lane * 4`, so every exchange stage is a 4-way conflict (53.5% stall). A stride-64 per-lane ownership (`buf[lane + 64*k]`) is conflict-free for every radix stage but needs the `convrot_fht_stage64` helpers rewritten, with `tests/test_convrot_unpacked_quantization.py` as the gate, and it changes the rotation's rounding. The same kernel's final quantize loop writes int8 one byte per element, so vectorizing its global load/store to 16 B is part of the same item. Concrete target: 0.46 ms -> ~0.35 ms for (4112, 6144), where the DRAM floor for the 75 MB it moves is 0.31 ms.
- Model-side producer savings: Krea 2 re-quantizes the same `pre` for wq/wk/wv/gate, so one prequantize per shared activation removes three of six producer calls (1.9% of a block). Folding pre-norm + modulation into the producer removes a read/write of x. Both need ComfyUI-side plumbing, not a kernel change.
- C-shuffle stores: the remaining share of the 2.5% store ablation - 128-bit coalesced writes instead of 32 B per instruction. Packing two bf16 inside a lane is impossible here: the gfx11 int8 accumulator layout gives lane L column `L%16` for all eight `e` values, so a lane owns eight rows of one column and adjacent columns always live in different lanes, which is why the tile has to go through LDS.
- Exact-K bodies: ~1-3%, and only meaningful after the compute-side gap. The kernel is at 79% of the MMA issue peak with memory removed.
- Autotune: the harness and the id table are in place, and a full sweep is ~7.5 min for 8 ids x 6 rounds over 68 cells (22.6 min for 24 ids). The two known misses (`h3.fc2`, `path.m512`) plus the power-of-two-K exception are exactly what a per-shape search would settle without hand-fitted rules.

## Hardware constraints that shaped it

Source: `~/rdna35-isa-markdown/rdna35_instruction_set_architecture.md` and `~/amd-llvm-project/llvm/lib/Target/AMDGPU/` (`FeatureISAVersion11_5_1` in `GCNFeatures.td`, the `MmaInt8` policies in `mma.h`, `SIMemoryLegalizer.cpp`).

| capability | on gfx1151 | consequence |
|---|---|---|
| `v_wmma_i32_16x16x16_iu8`, `f32_16x16x16_f16/bf16` | yes, 32 cycles each (measured 55 TF) | int8 can never beat bf16 per instruction. Only bytes matter |
| int8 operand layout | 4 VGPR/lane (16 B), lanes 0-15 duplicated into 16-31 | a fragment read can never be cheaper than 16 B/lane. The duplication is free but the unique bytes are 16 B/row |
| `FeatureWMMA256bInsts` | yes (gfx11.5 only) | fp16 fragments are 32 B/lane (256-bit), which is why int8 needs half the LDS bytes per MMA |
| fp8 conversion insts / fp8 WMMA | no | an fp8 path here widens to bf16, so it is slower per byte than int8. requantizing int8->fp8 would also change results |
| SWMMAC (sparse) | no (gfx12/13 only) | dense 2:4 decomposition is a wash at best. Ignore |
| `global_load_lds` / `buffer_load_...lds` (vmem-to-lds) | no (LLVM gates on gfx9/gfx10) | refill is global->VGPR->LDS. Deeper staging costs VGPRs without removing a barrier |
| `S_PREFETCH_DATA`, `S_BUFFER_PREFETCH_DATA` | no (gfx12/13 only) | no register-free L2 prefetch. Only the in-kernel VGPR prefetch exists |
| `S_SET_INST_PREFETCH_DISTANCE` | yes (1-3 lines) | measured +0.3%, kept as a `flags` bit |
| `S_SETPRIO` (USER_PRIO 2 bits) | yes | measured +0.5-0.8%, kept as a `flags` bit |
| SLC/DLC/GLC cache bits | yes. `!nontemporal` -> SLC=1 (GL2 STREAM, L1 HIT_EVICT) + DLC=1, but stores only | non-tied FLAT loads have a hardcoded zero cache policy operand in LLVM, so load hints are unreachable |
| LDS shape | 128 KB/WGP = 2 x 32 banks per SIMD pair, 1R+1W per bank per cycle, 64 KB cap per workgroup | 34.8 KB tile -> 3 blocks/WGP. 52.2 KB -> 2 |
| VGPR | 1536/SIMD (`Feature1536VGPRs`), allocated in blocks of 24 | 224 used -> 240 allocated -> 6 waves/SIMD hard cap |
| WMMA dependency rules | same-type chained D==C legal but may stall. D overlapping next A/B needs one `v_nop` | keep >=8 independent accumulators per warp (already the case) |
| VOPD | VALU only, wave32 | LDS instructions never dual-issue, so LDS instruction count is the currency |

## Tuning grid and harness

Share of one full sweep over both documented sizes, counting only the quantized block linears:

| model | families | share |
|---|---|---|
| H3 | 4 | 52.6% |
| Krea 2 | 4 | 14.4% |
| Klein 9B | 6 | 11.1% |
| Qwen-Image 2.1 | 3 | 8.3% |
| Z-Image | 4 | 6.3% |
| MOSS-Image | 4 | 5.6% |
| Anima | 4 | 1.7% |

H3 dominates through 720p (three of the four largest cells in the set), so a global flop ranking would tune H3's video rows and skip everything else. The grid keeps one cell per `(N, K)` family, so every model is represented, and uses the cheaper M of each family for tuning and the expensive one for validation only.

Which shapes exist at all. Counts come from the local `comfy_quant` inventories (Krea 2, H3, MOSS, Anima, and the Klein 9B file, which was converted here - see the conversion chapter) and from the per-model statements in `input_shapes_*.md` for Qwen-Image 2.1 and Z-Image (the Z-Image family list is inferred from CK's int8 perf shapes, `(4128,3840)` and `(4128,10240)`):

| model | quantized linears | (N, K) families | runtime M | fused input act |
|---|---|---|---|---|
| Krea 2 / Kroma | 224 = 28 blocks x 8 | (6144,6144) (1536,6144) (16384,6144) (6144,16384) | 4112, 9232 | - |
| Qwen-Image 2.1 | 192 of 200 = 32 x 6 | (4096,4096) (24576,4096) (4096,12288) | 4112, 9232, 16 (prefix pass), 4096/9216 (cache-target pass) | swiglu on (4096,12288) |
| Z-Image | 120 = 30 x 4 | (11520,3840) (3840,3840) (10240,3840) (3840,10240) | 4128, 9248 | - |
| H3 / MiniMax | 200 = 50 x 4 | (21504,5376) (5376,7168) (28672,5376) (5376,14336) | 5376, 25344 | swiglu on (5376,14336) |
| MOSS-Image | 168 = 42 x 4, fused qkv repack | (9216,3072) (3072,3072) (16384,3072) (3072,8192) | 4112, 9232 | situ on (3072,8192) |
| Anima | 254 of 672 = attention q/k/v/out (self and cross) in blocks 1-26 + mlp in blocks 4-26. Blocks 0 and 27 fully bf16 | (2048,2048) (2048,1024) (8192,2048) (2048,8192) | 4096, 9216 image rows. 512 for cross-attn k/v | - |
| Klein 9B (Flux 2) | 112 of 121 = 8 double x 8 + 24 single x 2 | (12288,4096) (4096,4096) (24576,4096) (4096,12288) (36864,4096) (4096,16384) | double img 4096, 9216. Double txt 512. Single 4608, 9728 | - |

Every int8 GEMM in these models is `TN` (`x[M,K] @ W[N,K]^T`, contiguous x): the kernel enforces it (`x.reshape(-1, k).contiguous()`, and no NT/NN/TT int8 kernel exists), and a transposed weight flips a layout flag and falls back to dequantize + bf16. Verified on the real blocks (`~/tmp/comfy-kitchen/probe_convrot_layout.py`, ConvRot weights, ComfyUI module classes): Krea 2 8/8 quantized linears, Qwen-Image 2.1 `to_q/k/v/out` + `gate_up`, MOSS `qkv`/`o`/`ffn_up_gate` all receive a contiguous `[1, L, K]`. The `TT` rows in `input_shapes_tt.md` (`img_in`, `video_embedder`, `audio_patch_proj`, `patchify_proj`, `x_embedder`, `txtfusion.projector`) are unquantized in every int8 ConvRot checkpoint, and `NT`/`NN` rows are training-only. `TT` still matters on the Feather side: its bf16/fp16 models do issue real TT GEMMs, so a shared grid needs TT cells there.

Three cells are not plain linears: H3 `fc2`, Qwen-Image 2.1 `img_mlp.out` and MOSS `ffn_down` call `linear_input_act(..., "swiglu"/"situ")`, which fuses the gated activation into the producer. The GEMM's `K` is the post-activation width and the producer reads a 2K-wide row (verified: fused == eager composition to bf16 rounding, max abs 3.9e-3, cos 0.99995). Two stream structures are also real and verified in the module code: Anima's cross-attention runs `k_proj`/`v_proj` on the 512-token adapter output (so `(2048,1024)` is a real int8 cell at M=512), and Klein/Flux 2 runs the `txt_*` linears on 512 padded text rows while single blocks run on `torch.cat((txt, img), 1)` (M = 512 + image rows = 4608 / 9728).

Real int8 ConvRot users outside this set, all one-shot per prompt and therefore outside a per-step tune: Qwen3.5-9B TE (317 linears: `(12288,4096)` `(4096,4096)` `(8192,4096)` `(4096,12288)` `(1152,1152)` `(3456,1152)` `(4304,1152)` `(1152,4304)` `(1024,4096)` `(248320,4096)`, 512-4096 rows) and Qwen3-VL-32B TE (350: `(25600,5120)` `(1024,5120)` `(5120,25600)` `(5120,8192)` `(8192,5120)`). The 1152/3456/4304-wide rows are the only partial-N shapes in play, which is why the path cells include them.

Tier A (tune, the cheaper M of each family. 31 cells, 336 ms per sweep at 39 TFLOPS) is `TIER_A` in `bench_cfg.py`. Tier B (validation only, one forced id: 9232 rows for Krea 2/Qwen-Image 2.1/MOSS, 9248 Z-Image, 25344 H3, 9216 Anima, 9216/9728 Klein. 27 cells, 999 ms per sweep) is `TIER_B`. The prefix-cache cells and the synthetic path cells (GEMV, 16/192/512/2048/3072 rows, partial-N 1152-wide) are `PATH`. The harness takes a 5th `act` field: fused cells allocate `x` of width 2K and pass `input_act`, and the ConvRot group size is chosen to divide K (1152-wide text-encoder rows use G=64). Weight source: every local int8 ConvRot checkpoint is indexed by `(N, K)` (Kroma, Anima, Klein 9B, MOSS, H3), so all model cells now run on real weights. Cells without a match use synthetic int8 + per-row scales, which is equivalent for kernel selection.

Build loop (the editable install does not work here, because setup.py builds the CUDA extension first and the local CUDA toolkit fails to compile): `COMFY_KITCHEN_BUILD_HIP=1 COMFY_HIP_ARCHS=gfx1151 python setup.py build_ext --inplace --no-cuda`, with the site-packages copy renamed to `comfy_kitchen.bak.orig` and a `comfy_kitchen_repo.pth` pointing at `~/comfy-kitchen`. An incremental rebuild after a `.hip` edit is ~1 min. The final runs behind the Results table are `r_tierA2.json` (31 cells) and `r_tierB2.json` (27 cells) in `~/tmp/comfy-kitchen`.

| grid | 8 ids x 6 rounds | 24 ids x 6 rounds |
|---|---|---|
| Tier A only (31 cells, 336 ms/sweep) | 1.9 min | 5.7 min |
| Tier A + B (58 cells, 1.34 s) | 7.5 min | 22.4 min |
| + prefix/path cells (68 cells, ~1.35 s) | 7.5 min | 22.6 min |

Which branch each cell class exercises, and what it measured:

| condition | id | cells | measured |
|---|---|---|---|
| `M <= 64` | 0 | 16-row prefix cells, `m16` tail | already at the memory roof. Unchanged |
| `K < 2048`, `M <= 1024` | 28 | Anima cross k/v (512 rows, K=1024) | 1.09x. id 27 was 23% slower here |
| `64 < M <= 1024` | 27 | the whole Klein text stream, `path.m512` | 1.02-1.10x. `path.m512` is 3% better on id 12 (known miss) |
| `K >= 8192 && N <= 8192` | 12 | the five down/`swiglu`/`situ` cells, Anima `mlp.2`, Klein `double mlp.2`/`single linear2` | wins by 5-6% on Krea 2 `mlp.down`, 2-3% on Z-Image/MOSS. H3 `fc2` is 2% better on 128 (known miss) |
| power-of-two K, N >= 8192 | 12 | Qwen-Image 2.1 `gate_up`, Klein `double qkv`, Klein `single linear1` | 2-3% over 128x256 |
| `N % 256 != 0` | 28 | 1152/3456/4304-wide text-encoder rows | 1.09x (4304), tie (3456) |
| `M < 4096` | 28 | 2048/3072-row cells | 1.037x / 1.041x |
| else | 128 | the bulk of the Tier A/B cells | 1.01-1.19x |

## Model conversion (bf16 -> int8 ConvRot)

No converter lives in `~/ComfyUI` or `~/comfy-kitchen` (they ship only the runtime `TensorWiseINT8Layout` and the HIP kernel). The canonical one is `Comfy-Org/comfy-model-tools` `quant_int8_convrot.py`, fetched to `~/tmp/comfy-kitchen/comfy-model-tools/`. Alternatives with the same output format are `Comfy-Org/comfy-quants` (per-family configs), the `ComfyUI-Kitchen-nvfp4-converter` node, `silveroxides/convert_to_quant` and `lodestones/Kromoss quantize_convrot.py`. Recipe used: fp32 upcast, group-Hadamard rotation at the largest power-of-4 groupsize dividing K (256 here), per-row absmax scale, `.comfy_quant = {format: int8_tensorwise, convrot: true, convrot_groupsize: 256}`.

Validated against the released Anima pair: re-quantizing the bf16 reproduces the released scales bit-exactly and the weights within +/-1 LSB (tie rounding), so tool and runtime kernel agree.

Converted here (both into `~/ComfyUI/models/diffusion_models/`):

| model | quantized | params | bf16 -> int8 | mean / max weight relerr |
|---|---|---|---|---|
| `flux-2-klein-9b` | 112 = 8 double x 8 + 24 single x 2 | 8.72 B | 16.91 -> 8.79 GB | 0.909% / 1.120% |
| `kroma-sensei-...-v0.3` | 224 = 28 blocks x 8 (`txtfusion` excluded, as in the released file) | 12.16 B | 23.96 -> 12.64 GB | 0.913% / 1.447% |
| `moss_image_sft` | 168 = `layers.N.proj.*` | 4.76 B | 9.47 -> 5.04 GB | 0.876% / 0.981% |

All three detect in ComfyUI (`Flux2` / `Krea2` / `MOSSImage`, `mixed_ops`) and a layer rebuilt through `comfy.ops` mixed-precision Linear runs at 1.54-1.84x the bf16 time with output cos 0.99990-0.99992 (out relerr 1.24-1.41%).

Format audit of every int8 ConvRot file on disk (`audit_int8_formats.py`, `audit_int8_quality.py`):
- All five released files plus the three new ones use the same runtime-effective format: `int8_tensorwise` + `convrot` + gs 256 + per-row F32 absmax scale, and the weights are genuinely rotated (as-stored excess kurtosis ~0.0 vs 2.7-5.4 after un-rotation. H3 checked against its own bf16 source, cos 0.99993, relerr 0.88-1.20%).
- Metadata generations differ - `kroma-v0.3.1-turbo-opd` still carries a legacy `__metadata__._quantization_metadata` block (plus merge-pipeline keys `base`/`target`/`trim`/`copy_prefixes`/`finetune`), `anima` adds `per_row`, the rest are minimal - but the loader reads only `format`, `convrot`, `convrot_groupsize`. `per_row`, `orig_dtype` and `format_version` are never read (`orig_dtype` is overwritten with the compute dtype at load, `comfy/ops.py`). So no speed or quality difference, only the extra `convert_old_quants` pass on the legacy file.
- Real size/quality items: no file uses MSE clipping (measured 1.9-3.6% lower weight error for free with `--mseclip`), and `kroma-v0.3.1-turbo-opd` stores 1.22 GiB of fp32 2-D weights (9.1% of the file: `tproj`, `tmlp.2`, `txtmlp.1/3`, `first`, `last.linear`) that ComfyUI casts to bf16 at use - `--downcast-fp32` reclaims it on a re-convert.
- Layer-selection policy (the knob that actually trades quality for speed) differs per family and is matched to the released variants above: Anima skips blocks 0 and 27 entirely and the MLP of blocks 0-3.
