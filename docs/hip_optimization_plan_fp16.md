# gfx1151 HIP FP16 Matmul Kernel Optimization Plan

## Scope and Metric

- Pure-FP16 forward `mm_fp16` (`csrc/hip_kernel_fp16.cu`, `feather_ops/mm_fp16.py`): fp16 A, fp16 prepacked B, fp16 C, optional fp16 bias. A is the usual row-major `[M, K]`. B is prepacked once. No backward, no bf16.
- Prepack contract: `[K, N]` fp16 -> `[K_pad/8, N, 8]`, with `K_pad = ceil(K/128)*128`. The padded rows are zero, which is what lets the kernel run a ragged K without a staging branch.
- Accuracy gate: relative L2 <= 1e-2 and max abs <= 1.0 against the fp32 naive reference.
- Metric: kernel-only milliseconds on the shared grid in `bench/cells.py`, converted to `2*M*N*K/t` TFLOPS.  Prepack and output allocation are outside the timed region. Report fp16 beside the fp8 control.
- fp8 control: the best compatible tile from `SCALED_MM_TILES` on the same cell, so the comparison measures kernel quality rather than a stale heuristic. The shipped fp8 heuristic is also reported where it differs.
- Keep rule: correctness passes and paired TFLOPS improve or hold on the grid. There is no fixed promotion threshold. A change is kept on repeated paired evidence and its cell-to-cell profile.
- The production path in this repo is `scaled_mm` (fp8e5m2 weight, fp16/bf16 activations), not `mm_fp16`.  The FP16 family is the pure-FP16 baseline and the reference for the quantization comparison.

## Current Status

- The fp16 kernel now owns the full shape contract: M >= 1, N % 16 == 0, K % 16 == 0, zero-padded prepack, and a tail-free fast path when the extents divide the tile. The old kernel required M % block_m == 0 and could not run most of the grid.
- The runtime autotuner is gone. `feather_ops/tiles.py::MM_FP16_TILES` is a static 15-tile table, the heuristic picks one of them, and the public wrapper allocates the output outside the custom-op boundary.
- Native C++/direct-op `8192^3`, tile `(2,8,2,8,2)` (256x256): `~35-37 TFLOPS`, up from `28.8` for the old `(1,8,2,8,2)` kernel. The fp8 best-of-tiles control is `39.9`. fp8/fp16 = `1.15`.
- Representative paired results (prepack and output excluded, medians, one session. Session drift is ±3%, so read them as representative, not as a leaderboard):

| shape | fp16 TFLOPS | fp16 tile | fp8 best TFLOPS | fp8/fp16 |
|---|---:|---|---:|---:|
| `2048^3` | 39.1 | `(2,4,2,4,2)` 128x128 | 37.8 | 0.97 |
| `4096^3` | 35.3 | `(2,4,2,4,2)` 128x128 | 35.8 | 1.01 |
| `8192^3` | 34.8 | `(2,8,2,8,2)` 256x256 | 39.9 | 1.15 |
| `512x5120x5120` | 32.9 | `(2,4,2,4,2)` 128x128 | 39.8 | 1.21 |
| `512x13824x5120` | 34.4 | `(2,8,2,8,2)` 256x256 | 39.1 | 1.14 |
| `256x2560x2560` | 28.5 | `(2,4,2,4,2)` 128x128 | 31.7 | 1.11 |
| `128x1024x5120` | 13.1 | `(1,1,2,1,1)` 16x16 | 11.4 | 0.87 |
| `16x3072x3072` | 6.7 | `(1,1,2,1,1)` 16x16 | 4.4 | 0.66 |
| `512x248320x4096` | 36.0 | `(2,8,2,8,2)` 256x256 | 39.9 | 1.11 |
| `4096x4608x4608` | 38.1 | `(2,8,2,8,2)` 256x256 | 39.5 | 1.04 |
| `12032x5120x5120` | 34.8 | `(2,8,2,8,2)` 256x256 | 42.6 | 1.22 |
| `25344x21504x5376` | 39.6 | `(2,8,2,8,2)` 256x256 | 40.2 | 1.02 |
| `1024x1280x1280` | 34.5 | `(2,4,2,4,2)` 128x128 | 34.4 | 1.00 |
| `2304x1280x1280` | 34.2 | `(2,4,2,4,2)` 128x128 | 36.0 | 1.05 |
| `4096x4096x12288` | 33.8 | `(1,8,2,8,2)` 128x256 | 32.8 | 0.97 |
| `4112x6144x6144` | 26.4 | `(1,8,2,8,2)` 128x256 | 36.1 | 1.37 |

- fp16 now wins or ties the compute-bound cells (`2048^3`, `4096^3`, `1024x1280`, `2304x1280`, `4096x4096x12288`) and the small-M cells (`16x3072`, `128x1024`). fp8 keeps a 1.1-1.4x advantage where the weight footprint or the 2-byte LDS payload binds: `8192^3`, `512x5120`, `12032x5120`, `248320x4096`, and the ragged `4112x6144` cell.
- The shipped fp8 heuristic was also repaired: the `prefetch=1` rows on 16x64/32x64/64x128 measured 30-50% slower than `prefetch=0` on every tested cell, so the heuristic now uses the no-prefetch form and the four bad rows were removed from `SCALED_MM_TILES`.
- `1x96768x2688` remains the outlier: 0.24 fp16 vs 0.44 fp8 TFLOPS. The weight is a single 260 MB pass and a 16-row WMMA tile wastes 16x the compute. Both are DRAM-bound. A dedicated GEMV/dot-product path is the only way to close it, and it is not worth it for one cell.
- The production `scaled_mm` path still measures `45.5-45.9 TFLOPS` at `8192^3`, so the pure-fp16 kernel is not the throughput path. CK is still absent. The current hipBLASLt is slower than this fp16 kernel in the current session.

## Implemented Plan

### Tails and shape contract

- M is masked, N is clamped/masked at 16-column granularity, K is zero-padded in the prepack. A staging clamps out-of-range rows/K to in-bounds memory. The epilogue drops the dead rows/columns.
- `kTailFree` is selected by `M % block_m == 0 && N % block_n == 0 && K % chunk_k == 0`. The masked path is never entered for the common divisible shapes.
- `prepack_b_for_mm_fp16` pads `K` to 128 and the C++ op checks `K_pad/8`. The Python wrapper raises on a stale prepack shape.

### Static tiles and heuristic

`MM_FP16_TILES` keeps the union of the winning families from the 106-cell sweep:
- `16x16`, `32x16`, `32x64`, `32x128` for small M. `64x64` for narrow N with a short K.
- `128x64` for M <= 128 and wide N. `128x128` as the general default.
- `128x192`/`128x256` for long-K / odd-N. `256x128` and `256x256` for wide N and weight-bound cells.
- `16x256` is the M <= 16 wide-N tile. `32x16`, `256x128 u2`, `64x128 u4`, and the 8-warp `128x128` are retained as sweep alternatives.

`select_mm_fp16_config` is the range heuristic in `feather_ops/mm_fp16.py`, fitted on the shared grid. It is deliberately small: tiny M uses 16x16, 16x256 at M<=16 with N>=16384, and 32x128 at M=32 with wide N. M <= 128 uses 128x64 for wide N. M <= 256 uses 256x128 u4 for wide N. M <= 512 uses 64x64 for narrow N with a short K, 128x128 as the general default, and 256x256 for N >= 12288. Above 512 the default is 128x128, with the long-K and wide-N exceptions. It returns 11 of the 15 instantiated tiles. The other four stay for sweeps and future shape classes.

### Pipeline experiments

- Kept: the existing B-only register prefetch on the tiles whose heuristic enables it.
- A double-buffering: tested and rejected. A second LDS A buffer, selected when the occupancy model said it was free, regressed `128x8192` 24.3 -> 16.3 TFLOPS, `512x5120` 34.3 -> 30.7, `2048^3` 40.1 -> 35.7, `512x13824` 35.9 -> 33.2, and `25344x21504` 41.0 -> 37.6, while helping only a few cells. The LDS model was not sufficient: the extra A stores contend with the LDS reads and change the compiler schedule.
- 2D block grouping: only the weight-stream exception survives. `group_m x group_n` was swept on the measured cells. The default 1x1 (N-fastest) is best almost everywhere. Grouping generally loses the A-panel locality. The exception is a small-M cell with `weight_bytes >= 512 MiB`, where `group_m=8, group_n=1` improved `512x248320x4096` from ~34.6 to ~37-39 in paired runs. Applying it to a large-M weight-stream cell (`25344x21504x5376`) regressed, so the policy is gated on `M <= 512`. The kernel keeps the general 2D mapping with that one policy.
- 512-row tiles: tested, never selected. `512x128`, `512x64`, `512x256` were swept. None won a cell. They were removed from the final tile table.

### API, autotune, and repo cleanup

- `mm_fp16_hip(a, b_prepacked, bias=None)` allocates outside the op and calls `torch.ops.feather_ops.mm_fp16`. `_mm_fp16_auto` is the opaque mutating op that applies the heuristic, so `torch.compile` sees one graph. Tests, benchmarks, and profilers pin a tile by calling `torch.ops.feather_ops.mm_fp16.default` with their own output buffer.
- Removed `feather_ops/autotune.py`, `_CONFIGS`, `_configured_op`, `_autotuned_op`, `register_custom_op_autotuning`, `old_autotune`, and the bf16/out-dtype branches from the fp16 path.
- `feather_ops/tiles.py` now owns `MM_FP16_TILES`. `tools/gen_tiles.py` emits `FEATHER_MM_FP16_TILES`. The kernel dispatch is generated from it. `tests/test_tiles.py` checks the header.
- `tests/test_mm_hip_fp16.py` covers the heuristic and every tile on a tail shape with a preallocated low-level output buffer.  `tests/test_mm_hip_fp16_compile.py` covers `torch.compile` eager parity.
- `bench/benchmark_mm_hip_fp16.py` uses the shared grid, preallocates output, excludes prepack, reports TFLOPS for fp16 and fp8, and supports `--tiles`/`--fp8-tiles`.
- The C++ benchmark pads B to 128 so ragged K can be measured. Prepack remains outside timing.

## Hardware Profiling Insights

The P1-P6 sections below are the earlier `(1,8,2,8,2)` campaign. They are kept because the relative deltas and the heuristics carried forward. The current kernel and its results are above. Absolute TFLOPS are session-specific, and the autotune references in P1-P6 are historical: the current `mm_fp16.py` has no autotuner.

### The FP16 vs FP8 Gap

We have established a highly optimized FP8 kernel (`scaled_mm_hip`) that achieves 45.5-45.9 TFLOPS in the native C++ harness. The primary difference between our optimized FP8 kernel and this baseline FP16 kernel is the memory footprint and LDS layout:
- LDS Footprint: FP16 data is 2 bytes per element, whereas FP8 is 1 byte. To load an equivalent `128x256` chunk, the FP16 kernel requires exactly twice as much LDS memory (`32 KB` vs `16 KB`).
- VGPR Footprint: Loading and holding FP16 data requires twice as many vector registers (`uint4` loads 8 `fp16` elements instead of 16 `fp8` elements).
- LDS Banking: The old swizzled-B baseline had a substantial LDS bank-conflict rate (`14.8%`), but the current identity-order B path reduces that to `0.0%` without increasing LDS usage.
- Latency Hiding Limitation: Because FP16 uses double the LDS memory, the old `(1,8,2,8,2)` tile used 24 KB of LDS and the `u=4` variant `(1,8,4,8,2)` nearly 32 KB per workgroup. This drastically lowers the hardware occupancy compared to FP8. Reduced occupancy means the macro-level scheduler cannot easily hide the `vmcnt` (global memory) stall latency, leading to a performance drop-off at large matrix sizes (the current 256x256 tile reaches 35.7 TFLOPS at N=8192. The old 128x256 tile reached 28.8).

### P1: Identity-Order B Prepack/Load - KEEP

- Isolation profiling on the old swizzled-B baseline (`N=8192`, forced `(1,8,2,8,2)`) showed:
  - baseline (`rocprof_fp16_bank_base/`): `LDSBankConflict 14.764%`
  - C-shuffle bypass (`rocprof_fp16_bank_noc/`): `14.815%` -> C-shuffle is not the source
  - temporary A-swizzle variant (`rocprof_fp16_bank_a_swizzle/`): `46.512%` -> A-side swizzle is actively worse
  - temporary B-identity-load variant (`rocprof_fp16_bank_b_identity/`): `0.0%` -> conflicts came from the B LDS load remap in `wmma_compute_stage`
- Implemented fix: change FP16 prepack to keep logical N order while preserving `(K/8, N, 8)` vec8 chunks, and change compute-side B LDS reads to identity mapping.
- Final validation (`rocprof_fp16_bank_identity_final/`, forced `(1,8,2,8,2)`): `LDSBankConflict = 0.0%`, average profiled kernel time `37.9 ms`.
- Full benchmark result (`python ~/feather-ops/bench/benchmark_mm_hip_fp16.py`): `N=8192` 27.4 TFLOPS (up from 26.7), with the winner config still `(1,8,2,8,2)`.
- Key takeaway: B prepack is useful for 128-bit transport, but it does not need the old swizzled-N order. Identity-order prepack keeps the same LDS footprint and removes bank conflicts.

### P2: Bottleneck Profiling For `(1,8,2,8,2)` - KEEP analysis

- Verified PC sampling at `N=8192` (`fp16_pc_verify_8192_results.db`, forced `(1,8,2,8,2)`, dispatches `4+`) shows:
  - `Sync 46.7%`
  - `LDS Read 36.8%`
  - `WMMA 8.6%`
  - `SALU 4.3%`
- Top sampled instructions are dominated by chunk-boundary synchronization and LDS read issue pressure:
  - `s_barrier`
  - `s_waitcnt vmcnt(0)` / `s_waitcnt lgkmcnt(*)`
  - many `ds_load_b128`
- Reduced detailed ATT at `N=2048` (`thread_trace_fp16_valid_small/`) confirms the true long-latency events are global-memory waits, not WMMA or conversion:
  - `s_waitcnt vmcnt(1) 223,978 cycles`
  - `s_waitcnt vmcnt(0) 102,154 cycles`
  - aggregated trace latency is dominated by `Sync` (`9.45M cycles`) while `WMMA` contributes only `0.95M cycles`
- No-detail ATT at `N=8192` (`thread_trace_fp16_8192_nodetail/`) succeeds without GPU reset and is qualitatively wait-dominated, but still reports `Thread trace buffer full!`. It is useful for occupancy/wave-state analysis, not detailed per-instruction timing.
- PMC check (`rocprof_fp16_bottleneck_pmcs_try/`) shows:
  - `LDSBankConflict = 0.0`
  - `L2CacheHit 32.5`
  - `VALUInsts = 11406`
- Main conclusion: after fixing B bank conflicts, the `(1,8,2,8,2)` kernel is bottlenecked by serialized chunk-boundary refill + synchronization. The current loop computes the whole current chunk, then hits a barrier, then fetches the next chunk, then hits another barrier. There is effectively no overlap between next-chunk global refill and current-chunk WMMA.
- Controlled C++ ablation benchmark (`~/tmp/feather_ops/fp16_analysis/mm_fp16_vram_lds_ablation.cu`, `~/tmp/feather_ops/fp16_analysis/benchmark_mm_fp16_vram_lds_ablation.cpp`) now isolates the chunk-refill path without Python/Torch overhead and without zero-input bias:
  - Forced config only: `(1,8,2,8,2)`
  - Kept method: load the first chunk from real random fp16 inputs, then for later chunks selectively reuse the already-random LDS contents instead of refilling from global memory. This avoids the known `WMMA-with-zeros` speedup while also avoiding synthetic-PRNG overhead in the timed loop.
  - Repeated `N=8192` runs (2 passes, 100 timed iters each) give mean kernel times:
    - `full`: `40.16 ms` (`27.38 TFLOPS`)
    - `reuse_a`: `34.76 ms` (`delta 5.41 ms`, `13.5%` of full time)
    - `reuse_b`: `26.51 ms` (`delta 13.65 ms`, `34.0%` of full time)
    - `reuse_ab`: `24.96 ms` (`delta 15.21 ms`, `37.9%` of full time, `44.06 TFLOPS`)
  - Interpretation:
    - `15.21 ms` of the `40.16 ms` end-to-end kernel time is tied to the repeated global/L2->LDS refill path for this config.
    - The B-side refill dominates the exposed cost. That matches the larger per-chunk B payload (`16 KB`) versus A (`8 KB`) for `(1,8,2,8,2)`.
    - The A-only and B-only deltas are not additive (`5.4 + 13.7 > 15.2 ms`), so the refill costs share fixed sync/issue overhead and interact with each other.
    - With both refills removed, the kernel rises to `44.06 TFLOPS`, which is strong evidence that the math/LDS-read body is already capable of near-FP8-class throughput once chunk refills are taken out of the critical path.
- Controlled C++ ablation benchmark for the compute-side LDS->VGPR path (`~/tmp/feather_ops/fp16_analysis/mm_fp16_lds_vgpr_ablation.cu`, `~/tmp/feather_ops/fp16_analysis/benchmark_mm_fp16_lds_vgpr_ablation.cpp`) now measures the exposed cost of the inner `ds_read -> VGPR` fragment loads while keeping global refill intact:
  - Forced config only: `(1,8,2,8,2)`
  - Kept method: always use real random fp16 A/B input tensors. Inside the compute stage, selectively reuse already-loaded random fragments to remove part of the LDS->VGPR traffic without introducing zero operands.
  - To avoid changing occupancy with long-lived fragment caches, the A-side study uses a pairwise reuse mode (`reuse_a_p2`): the kernel loads A from LDS on even `rm` and reuses that fragment for the next odd `rm`, so 50% of A LDS->VGPR loads are removed. The B-side study (`reuse_b`) loads only one of the two `rn` fragments and reuses it for the second tile, so 50% of B LDS->VGPR loads are removed.
  - Repeated `N=8192` runs (2 passes, 100 timed iters each) give mean kernel times:
    - `full`: `40.37 ms` (`27.23 TFLOPS`)
    - `reuse_a_p2` (50% A LDS->VGPR removed): `40.23 ms` (`delta 0.14 ms`, `0.36%`)
    - `reuse_b` (50% B LDS->VGPR removed): `36.93 ms` (`delta 3.44 ms`, `8.53%`)
    - `reuse_ab_p2` (50% A + 50% B removed): `37.71 ms` (`delta 2.67 ms`, `6.61%`)
  - Interpretation:
    - The exposed LDS->VGPR cost is heavily B-side dominated on this config.
    - A-side LDS->VGPR reads are close to noise-floor in this experiment. Either their direct cost is very small, or they are largely hidden by the existing schedule.
    - The `reuse_b` delta implies that the full B LDS->VGPR fragment load cost is on the order of `6-7 ms` if the effect scaled linearly, but treat that as an upper estimate rather than a strict decomposition.
    - The combined half-ablation is not additive, which means these inner LDS reads interact with the surrounding instruction schedule. Simple per-side deltas should be read as exposed cost, not exact standalone service time.

### P3: B-Only Full-Chunk Prefetch Under WMMA - KEEP

- Kept target: `(1,8,2,8,2)` only.
- Final kept schedule:
  - issue both next-chunk B stages as global->VGPR `uint4` prefetches before current-chunk WMMA,
  - keep those values in four explicit scalar `uint4` temporaries,
  - compute the current two WMMA stages unchanged,
  - after the existing chunk barrier, commit prefetched B into LDS and refill A on the old path,
  - keep the original two-barrier structure.
- Generated-code / metadata confirmation on the kept variant (`extracted hsaco metadata`):
  - `(1,8,2,8,2)` now uses `VGPR=190`, `LDS=24576 B`, `private=0`, `vgpr_spills=0`.
  - This preserves the old LDS footprint while adding only modest register pressure.
- Full benchmark result (`python ~/feather-ops/bench/benchmark_mm_hip_fp16.py`) after adding a small/medium-N autotune guard:
  - `N=2048`: 15.7 TFLOPS, autotuned to `(4,2,2,2,4)`
  - `N=4096`: 34.3 TFLOPS, autotuned to `(1,8,2,8,2)`
  - `N=8192`: 27.9 TFLOPS, autotuned to `(1,8,2,8,2)`
- Net effect versus the previous baseline:
  - `N=8192` improved from 27.4 to 27.9 TFLOPS.
  - `N=4096` improved materially and moved onto the wider `128x256` tile.
  - `N=2048` stays on the narrower path after guarding the wide tile out of the runtime autotune set below `N=4096`.
- Two rejected sub-variants from this P3 search:
  - `REJECT`: rolling one-stage B prefetch with an array/ref-based buffer
    - correctness passed, but extracted metadata showed `(1,8,2,8,2)` inflated to `LDS=57344 B`, and `N=8192` collapsed to 22.2 TFLOPS.
    - regression reason: the intended register-prefetch state was effectively lowered into compiler-managed LDS, destroying occupancy.
  - `REJECT`: scalarized rolling one-stage B prefetch with per-stage commit
    - correctness passed and metadata returned to `LDS=24576 B`, but the added mid-chunk barrier/control cost limited `N=8192` to 26.9 TFLOPS.
    - regression reason: preserving the current LDS footprint was not enough. The extra synchronization erased the overlap gain.
- Why the small/medium-N autotune guard is kept:
  - On `N=2048`, the runtime `old_autotune` short probe window incorrectly preferred `(1,8,2,8,2)` even though the benchmark harness (`triton.testing.do_bench`) measured that config slower than `(4,2,2,2,4)` / `(4,2,4,2,4)`.
  - Restricting `(1,8,2,8,2)` to `N >= 4096` matches the observed crossover after P3 and removes the end-to-end `2048` regression.

### P5: Native B-Prefetch Sweep And Runtime Heuristic - KEEP

- Added native `auto|on|off` B-prefetch control to the FP16 benchmark/kernel path:
  - benchmark harness: `~/feather-ops/cpp_benchmarks/benchmark_mm_hip_fp16.cpp`
  - build target: `benchmark_mm_hip_fp16` in `~/feather-ops/cpp_benchmarks/CMakeLists.txt`
  - sweep runner: `~/tmp/feather_ops/fp16_analysis/run_b_prefetch_sweep.py`
  - transition sweep runner: `~/tmp/feather_ops/fp16_analysis/run_b_prefetch_region_sweep.py`
  - raw sweep results: `~/tmp/feather_ops/fp16_analysis/b_prefetch_sweep.csv`
  - focused transition sweep log: `~/tmp/feather_ops/fp16_analysis/b_prefetch_region_sweep.txt`
- The kernel now supports the simplified policy requested for this search:
  - either prefetch all B stages in the next chunk,
  - or prefetch none of them.
- Sweep protocol:
  - run native C++ benchmark only (`--no-bias`) to avoid Python/Torch overhead,
  - test both `b_prefetch=off` and `b_prefetch=on`,
  - sweep all compatible autotune configs over these shapes:
    - target Qwen-Image shapes: `(32,12288,2048)`, `(32,2048,12288)`, `(8192,12288,2048)`, `(8192,2048,12288)`
    - extra crossover shapes: `(128,12288,2048)`, `(128,2048,12288)`, `(512,12288,2048)`, `(512,2048,12288)`, `(2048,12288,2048)`, `(2048,2048,12288)`
- Target-shape result summary from the sweep:
  - `(32,12288,2048)`: best `off 6.99 TFLOPS`, best `on 7.21 TFLOPS`
  - `(32,2048,12288)`: best `off 3.09 TFLOPS`, best `on 5.43 TFLOPS`
  - `(8192,12288,2048)`: best `off 29.06 TFLOPS`, best `on 28.37 TFLOPS`
  - `(8192,2048,12288)`: best `off 34.48 TFLOPS`, best `on 37.71 TFLOPS`
- Main sweep conclusions:
  - B prefetch is not a monotonic "large shape = on" optimization.
  - Long-K shapes (`K >= 8192`) benefit much more consistently than short-K shapes.
  - The strongest always-positive config families in this sweep were:
    - `(2,4,2,4,2)`
    - `(2,4,4,4,2)`
    - `(2,1,4,2,2)`
  - The wide `128x256` family `(1,8,*,8,2)` benefits on most shapes, but the large wide-N / short-K case `(8192,12288,2048)` is a real exception and should keep prefetch off.
  - Very small-`M` row paths need a small explicit exception: `(1,8,2,1,2)` helps on `(32,12288,2048)` even though the general short-K rule is more conservative.
- Focused transition sweep (`~/tmp/feather_ops/fp16_analysis/b_prefetch_region_sweep.txt`) tightened the boundaries:
  - short-K wide-N `128x256` tile `(1,8,2,8,2)` stays positive through `M=2048`, then flips negative by `M=4096`
  - long-K `128x128` / `warps_n=2` families split into two coarse regions:
    - balanced `warps_m == warps_n` stays positive only through `M=512`
    - asymmetric `warps_m != warps_n` is positive at very small `M` and again from `M=2048`
- Kept runtime heuristic in `hip_kernel_fp16.cu`, written as shape regions instead of tuple-specific exceptions:
  - Region A, long-K: `K >= 8192`
    - if `block_warps_n >= 4`, enable B-prefetch
    - else if `block_warps_m == block_warps_n`, enable only for `M <= 512`
    - else enable for `M <= 256` or `M >= 2048`
  - Region B, short-K wide-N: `K <= 2048 && N >= 8192`
    - enable only when `block_n >= 256` and `M <= 2048`
  - otherwise keep B-prefetch off
- Validation after keeping the heuristic:
  - native `--b_prefetch auto` best-config scan on the four target Qwen shapes selected:
    - `(32,12288,2048)` -> `(1,8,2,1,2)` at `7.08 TFLOPS`
    - `(32,2048,12288)` -> `(1,1,4,1,2)` at `5.38 TFLOPS`
    - `(8192,12288,2048)` -> `(1,8,2,8,2)` at `29.99 TFLOPS`
    - `(8192,2048,12288)` -> `(4,2,4,2,4)` at `39.02 TFLOPS`
- Keep reason:
  - this simpler region heuristic still matches the measured sweep winner on the studied shapes while avoiding the clear regression on the large wide-N / short-K case.

### P6: Native A-WSGR Sweep And Runtime Heuristic - KEEP

- Restored native FP16 benchmark controls and added explicit A-side WSGR mode selection:
  - benchmark harness: `~/feather-ops/cpp_benchmarks/benchmark_mm_hip_fp16.cpp`
  - kernel launch entrypoint: `launch_mm_fp16_tuning_mode(...)`
  - full sweep runner: `~/tmp/feather_ops/fp16_analysis/run_a_wsgr_sweep.py`
  - focused transition runner: `~/tmp/feather_ops/fp16_analysis/run_a_wsgr_region_sweep.py`
  - raw sweep results: `~/tmp/feather_ops/fp16_analysis/a_wsgr_sweep.csv`
  - focused transition log: `~/tmp/feather_ops/fp16_analysis/a_wsgr_region_sweep.txt`
  - target-shape auto validation: `~/tmp/feather_ops/fp16_analysis/a_wsgr_auto_validation.txt`
- Sweep protocol:
  - keep `B-prefetch=auto`,
  - benchmark only native C++ `--no-bias`,
  - compare `a_wsgr=off` vs `a_wsgr=on`,
  - sweep the 20 config families where A-side WSGR actually changes codegen (`block_warps_n > 1`),
  - shapes:
    - target / crossover set: `(32,12288,2048)`, `(32,2048,12288)`, `(128,12288,2048)`, `(128,2048,12288)`, `(512,12288,2048)`, `(512,2048,12288)`, `(2048,12288,2048)`, `(2048,2048,12288)`, `(8192,12288,2048)`, `(8192,2048,12288)`
- Main sweep result:
  - the old A-side rule was effectively just `repeat_m <= 4`. With B-prefetch left on auto, that is too broad.
  - A-WSGR is clearly negative on the wide `128x256` family `(1,8,*,8,2)`:
    - focused sweep confirms `(1,8,2,8,2)` is negative on every tested point, both short-K and long-K.
  - A-WSGR is clearly positive on the balanced square `128x128` / `u=2` family `(2,2,2,4,4)`:
    - focused sweep shows positive deltas across the full tested `M=128..8192` range for both `K=2048` and `K=12288`.
  - A-WSGR is also consistently positive on the tall `128x128` / `repeat_m=2` / `u=2` family `(4,2,2,2,4)`:
    - gain is small but stable across the same `M=128..8192` range.
  - The only important crossover inside the positive families is `(4,2,4,2,4)`:
    - focused sweep shows it is negative at `M=128` and `M=256`,
    - it flips positive from `M=512` upward for both short-K and long-K.
  - Most remaining N-heavy families are mixed or negative:
    - `(1,2,*,2,2)` is always negative in the sweep,
    - `(1,4,4,4,2)`, `(2,4,2,4,2)`, `(2,4,4,4,2)`, `(2,4,4,4,4)` are mostly negative,
    - `(1,4,2,4,2)` and `(1,8,4,8,2)` become positive in some long-K / large-M regions, but they never become the best autotune winner on the studied target shapes.
- Kept runtime heuristic in `hip_kernel_fp16.cu`, written as coarse regions:
  - Region A: `repeat_m == 1` or `repeat_m >= 8`
    - keep A-WSGR off
  - Region B: `repeat_m == 2`
    - enable only on tall wave layouts `block_warps_m > block_warps_n`
    - for `unroll_k == 4`, require `M >= 512`
  - Region C: `repeat_m == 4`
    - enable only on balanced square layouts `block_warps_m == block_warps_n` with `unroll_k == 2`
  - otherwise keep A-WSGR off
- Replay result on the 10-shape native sweep:
  - this kept heuristic matches the per-shape best winner across the studied sweep set.
  - the old `repeat_m <= 4` rule missed the best winner on `(128,2048,12288)` by over-enabling WSGR on a `repeat_m=1` path.
- Auto validation on the four target Qwen-Image shapes (`~/tmp/feather_ops/fp16_analysis/a_wsgr_auto_validation.txt`):
  - `(32,12288,2048)` -> best `(1,8,2,1,2)` at `7.23 TFLOPS`
  - `(32,2048,12288)` -> best `(1,1,4,1,2)` at `5.40 TFLOPS`
  - `(8192,12288,2048)` -> best `(1,8,2,8,2)` at `29.58 TFLOPS`
  - `(8192,2048,12288)` -> best `(4,2,4,2,4)` at `38.21 TFLOPS`
- Keep reason:
  - this region heuristic is simpler than the old implicit `repeat_m <= 4` rule,
  - it preserves the measured best winner on the studied target shapes,
  - it avoids the large regressions from turning A-WSGR on for the wide `repeat_m=8` family.

### Durable Findings

- Zero-conflict B LDS access is achievable without padding or larger LDS allocation.
- For this FP16 kernel family, B swizzle was the source of LDS bank conflicts. Identity-order B prepack is better.
- A-side swizzle is a bad direction on the current winner path (`46.5%` conflict rate in the isolation build).
- The purpose of FP16 B prepack is to preserve vec8 / 128-bit transport, not specifically to preserve the old swizzled order.
- After removing LDS bank conflicts, the dominant large-`N` bottleneck is global-refill latency amplified by workgroup-wide synchronization, not WMMA throughput.
- A direct refill ablation on `(1,8,2,8,2)` attributes `38%` of `N=8192` wall time to repeated global/L2->LDS chunk refill. B-side refill is the larger exposed component.
- A direct compute-side LDS->VGPR ablation shows the exposed inner-fragment load cost is much smaller than the global-refill cost and is mostly on the B side. A-side LDS->VGPR reads are near the noise floor in this schedule.
- `FP Convert` and general `VALU` are not important bottlenecks on the FP16 kernel (`PC sampling: FP Convert 0.1%, VALU Other 1.3%`).
- `LDS Read` remains a secondary issue-pressure hotspot, but the first thing to attack is the serialized refill/barrier structure around `s_waitcnt vmcnt(*)` and `s_barrier`.
- For unstable large-`N` ATT on gfx1151, use no-detail ATT first. Detailed `N=8192` ATT is still unsafe on the current tooling stack.
- On gfx1151, practical latency hiding for this kernel should be treated as a software-pipelining / occupancy problem, not a search for a magical async global->LDS primitive.
- gfx10+ HIP compute defaults to WGP mode. CU mode is not the default path. That matters because some LDS-direct mechanisms are CU-mode-specific, so any CU-mode experiment should be treated as a separate, toolchain-dependent branch rather than the default optimization direction.
- Direct global->LDS is not a primary near-term path for this kernel on gfx1151:
  - the older async-to-LDS intrinsics are documented for gfx9/gfx10,
  - the newer `global.load.async.to.lds.b{8,32,64,128}` path is gfx1250+,
  - LLVM MC currently marks `global_load_lds_*` unsupported on gfx11.
- Even if a direct-to-LDS path becomes usable later, it carries strong layout constraints: each lane transfers one DWORD and lanes in a wave must write consecutive DWORDs into LDS. That does not map naturally onto the current vec8 / `uint4` fp16 transport without non-trivial repartitioning.
- Full AB ping-pong LDS buffering is risky for `(1,8,2,8,2)`: doubling LDS from `24576 B` to `49152 B` would cut the occupancy model from `50%` to `25%`, so it should not be the first overlap experiment.
- Because the refill ablation is strongly B-side dominated, the first overlap attempts should prioritize hiding B refill earlier than A refill, or more aggressively than A refill, before trying symmetric AB double-buffering.
- For this kernel family, explicit scalar `uint4` prefetch state is materially safer than array/ref-style prefetch buffers. The latter can trigger compiler-generated LDS growth even when the source change looks like a register-only prefetch.
- On gfx1151 for `(1,8,2,8,2)`, keeping the original two-barrier chunk structure mattered: the final kept P3 shape worked only after removing the extra mid-chunk barrier from the rolling prefetch attempt.
- Config crossover is driven by A reuse vs control overhead:
  - `(4,2,2,2,4)` / `(4,2,4,2,4)` remain the better small/medium-N benchmark-harness choices at `N=2048`.
  - `(1,8,2,8,2)` with kept P3 B-prefetch now wins at `N=4096..8192`.
  - The runtime autotune path should respect that crossover explicitly, because the short probe window can mis-rank configs near the boundary.
- For FP16 B-prefetch, the useful decision boundary is config-sensitive plus K-sensitive:
  - `K >= 8192` is the strongest generic signal for turning prefetch on.
  - `block_warps_n >= 4` is a good coarse long-K proxy for a prefetch-positive region.
  - balanced `warps_m == warps_n` narrow-N families only want B-prefetch at small `M`.
  - the large wide-N / short-K crossover is mainly an `M` threshold: keep prefetch on through `M=2048`, then turn it off by `M=4096`.
- For A-side WSGR, the useful boundary is tile-family-sensitive rather than a generic `repeat_m <= 4` rule:
  - `repeat_m=8` wide tiles should keep WSGR off.
  - `repeat_m=1` paths are near noise-floor and are safer to keep off.
  - square `128x128` / `u=2` tiles are consistently WSGR-positive.
  - tall `128x128` / `repeat_m=2` tiles are WSGR-positive, but the `u=4` family needs `M >= 512`.
  - most N-heavy `repeat_m=4` families stay negative or too mixed to justify enabling by default.
- The hipBLASLt `8192^3` winner is a different kernel family: `128x96x64` with `(4,1,4,2,6)`, 128 threads, 256 VGPR, `30336 B` LDS, a single-LDS-buffer software pipeline, and no direct-to-LDS/VGPR. See Native Library References.
- Cloning that family was not needed: the current `(1,8,2,8,2)` kernel is faster in the current session, and `(4,1,4,2,6)` was never added to the search space. The hipBLASLt resource facts are kept below as a reference for reading a tuned library kernel, not as a target.
- If a future library kernel is faster, compare its schedule and assembly rather than assuming compiler magic. ROCm is open source. Every optimization it uses should be reproducible in principle.

## Closed Directions

The gap-closing campaign against hipBLASLt is closed: the current kernel is faster than the available CK/hipBLASLt references, and the production path is `scaled_mm`. Do not restart these without a new, faster library reference:
- G1, clone the `128x96x64` `(4,1,4,2,6)` hipBLASLt family: not pursued. The current `(1,8,2,8,2)` with B-prefetch is faster, and the family differs in K depth, N width, wave count and VGPR budget (reference below).
- G2, a true single-LDS-buffer pipeline: the kept overlap is the B-only full-chunk prefetch (P3/P5/P6), still inside the original two-barrier structure. The remaining A-side refill is the main known exposed cost (P2 measured B-side 34% and A-side 13.5% before the B prefetch), but with the current references beaten there is no active target for it.
- G3, asm diff against hipBLASLt: keep as a method only if a future library kernel wins. The source-level comparison was enough to show that the two kernels are different families.
- G4, a new layout branch: identity-order B prepack `(K/8, N, 8)` is the kept layout. No layout work is pending.
- G5, reconnect Python/autotune: done. The fp16 autotuner is removed. The static heuristic and the native out= path are what ship. The Python/C++ split remains useful for isolating wrapper overhead.

## Native Library References

Kept from the three predecessor reference files because the schedules and failure modes are still useful when reading a library kernel or reviving a comparison. They are not performance targets: the current kernel is faster. Numbers are historical unless marked current.

### CK

- Old wrapper flow: `fp8_to_half_kernel` upcast the whole B from fp8e4m3fn to fp16 in global memory, then `gemm_impl_wmma_noswap` instantiated `DeviceGemmWmma_CShuffle`, then an optional `scale_bias_kernel`. CK received fp16 B. The fp8 decode was not fused.
- Instance: `128x256x64`, 256 threads, `MPerWmma/NPerWmma=16`, `MRepeat/NRepeat=4`. A used vectorized global reads (`SrcScalarPerVector=8`), vectorized LDS writes (`DstScalarPerVector=8`) and `M+1` LDS padding. B used `SrcScalarPerVector=1` on the global read (not vectorized), vec8 LDS writes, and extra N padding. The epilogue was C-shuffle.
- The wrapper (`kernel/ck`) is no longer in this tree and the current kernel is faster. The durable lesson is the A-path bundle: vectorized read, vectorized LDS write and `M+1` padding were designed together. Matching only one of them is usually not enough. CK's non-vectorized B global read is the opposite trade.
- If CK is ever revived, the entry points are `DeviceGemmWmma_CShuffle` / `GridwiseGemm_Wmma` in `~/rocm-libraries/projects/composablekernel/`.

### hipBLASLt

- Identify the selected kernel from the log, not the integer index: `HIPBLASLT_LOG_MASK=96 ~/feather-ops/cpp_benchmarks/build/benchmark_mm_hipblaslt_fp16 --warmup 0 --iters 1 --workspace-mb 64`, then match `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x96x64_MI16x16x1_...`. The old public index `1112` mapped to Tensile `SolutionIndex: 71`. Both are version-dependent. Tensile logic: `~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Ailk_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`.
- The 128x96x64 family: MacroTile `128x96`, DepthU `64`, `MatrixInstruction=[16,16,16,1]`, wave32, 128 threads, `WorkGroup=[64,2,1]`, `MIWaveGroup=[4,1]`, `MIWaveTile=[2,6]`, `ThreadTile=[16,6]`. Equivalent custom tuple `(4,1,4,2,6)` (`128=16*4*2`, `96=16*1*6`, `64=16*4`). `LDSB1`, LDS `30336 B`, `LdsPadA/B=8`, `GRVWA/B=8`, `LRVW=16`, `NumLoadsA/B=8/6`, `PGR2`, `PLR0`, `SIA3`, `TransposeLDS1`, `UnrollMajorLDSB=true`, `SourceSwap=true`, `StaggerU=32`, `WorkGroupMapping=8`, `DirectToLds/Vgpr=false`. `VGPR=256`, `SGPR=128`.
- Interpretation: single-LDS-buffer software pipeline, no direct global->LDS or direct-to-VGPR. 4 waves/block, 256 VGPR -> 6 waves/SIMD (37.5%), deeper K (64 vs 32), narrower N (96 vs 256), higher register pressure than the FP16 winner.
- ABI pitfall: build any extension against the installed system headers that match the loaded `libhipblaslt.so`. A local-`rocm-libraries` header build produced mismatched `hipblaslt_ext::GemmPreference`/`GemmInputs` layouts, `setMaxWorkspaceBytes` wrote garbage offsets, hipBLASLt rejected the fast solution and fell back to a 26-28 TFLOPS kernel. The chosen solution used 0 MiB of workspace, but the preference still had to be ABI-correct.
- The wrapper and benches live in `csrc/hipblaslt_kernel_fp16.cu`, `feather_ops/mm_hipblaslt.py`, `bench/benchmark_mm_hipblaslt_fp16.py`, and `cpp_benchmarks/benchmark_mm_hipblaslt_fp16.cpp`.
- Profiling: single-iteration `rocprofv3` runs are useful for kernel metadata (`vgpr_count`, `sgpr_count`, `lds_size`, `workgroup`), not for timing. Performance is ROCm-version and thermal sensitive, see `benchmark_tracking.md`. The old `32-38 TFLOPS` range is not a constant, and the current session's `25.8` is below the current FP16 kernel's `35-37`.

### torch.compile

The target is `profile/profile_scaled_mm_torch.py` at `N=8192`:
- torch.compile decomposes the graph into three kernels: a Triton fp8->fp16 cast (`triton_poi_fused__to_copy_0`), an ATen/rocBLAS/Tensile GEMM (`extern_kernels.mm`), and a Triton fused `mul(scale)+add(bias)` epilogue (`triton_poi_fused__to_copy_add_mul_1`). The cast is not fused into the GEMM. At `8192^3` it moved ~201 MB (fp8 read + fp16 write) in ~0.95 ms in that session, and the separate epilogue was ~1.04 ms. The fused fp8-in-LDS design avoids both passes.
- CK is not used on gfx1151 by torch.compile: the default GEMM backends are `ATEN,TRITON,CPP`, and the ROCm CK arch list is gfx90a/gfx942/gfx950. The GEMM therefore comes from rocBLAS/Tensile, not Inductor CK.
- The selected Tensile GEMM then: `MT128x128x32`, `1LDSB1`, `WG32_4_1` (128 threads), 4 waves/block, `17408 B` LDS, `256 VGPR`, `128 SGPR`, 6 waves/SIMD (37.5% occupancy), `PGR1`, `PLR0`, `SIA3`, `SU32/SUS256`, `WSGRA1/WSGRB1`, `LRVW16`, `WGM8`, grid `(8192,64,1)`. Its GEMM time was ~40.2 ms at `8192^3` in that session. The symbol begins `Cijk_Ailk_Bljk_HHS_BH_MT128x128x32_MI16x16x16x1_SN_...`.
- Matching tile/workgroup geometry alone was not enough: the remaining difference was fine-grained pipeline/dataflow scheduling (`PGR/PLR`, `SIA`, stagger-U, wave-separated global reads), not occupancy. That is the lesson to carry to any future library comparison.
- Token decoding (Tensile naming): `MT` macro tile, `MI` matrix instruction, `WG` workgroup, `TT` thread tile, `LDSB` LDS-buffer count, `PGR/PLR` global/local-read prefetch, `GRVW/GLVW` global-read vector width, `LRVW` local-read vector width, `LPA/LPB/LBSPP` LDS padding, `SIA` schedule iteration algorithm, `SU/SUS` stagger-U, `TLDS/UMLDS` transpose/unroll-major LDS, `WSGR` wave-separated global read, `WGM` workgroup mapping, `ISA` target. Parameter semantics: `~/rocm-libraries/shared/tensile/docs/src/conceptual/kernel-parameters.rst`, `Tensile/Common.py`, `Tensile/SolutionStructs.py`. Scheduling: `KernelWriter.py`. Stagger-U: `KernelWriterAssembly.py`.
- Thread tracing of that Tensile kernel: the global pipeline was fully interlocked and the fp8->fp16 tax was zero inside the loop (the cast was a previous kernel), but ~35% of the math loop went to LDS reads and pointer arithmetic, leaving WMMA active ~65%. With 256 VGPR and 37.5% occupancy there was less latency-hiding headroom than the 8-wave FP16 kernel. That is the mechanism behind a well-tuned library GEMM losing to the fused custom kernel, and it is why the cast + separate epilogue are not automatically a better trade.
