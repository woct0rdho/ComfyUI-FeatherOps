# gfx1151 HIP FP16 Matmul Kernel Optimization Plan

## Scope and Metric

- Prepack contract:
  - B is pre-transposed into identity-order `(K/8, N, 8)` layout, preserving vec8 / 128-bit chunks.
  - The C++ kernel consumes this prepacked storage to perform direct loads into LDS without additional transposition.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance metric: `benchmark_mm_hip_fp16.py` GFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - `N=8192` does not regress,
  - large-N trend (`2048`, `4096`) does not regress materially.

## Current Baseline

- Latest full benchmark (`python benchmark_mm_hip_fp16.py`):
  - `N=2048`: `~15.7k` GFLOPS
  - `N=4096`: `~34.3k` GFLOPS
  - `N=8192`: `~27.9k` GFLOPS
- Current autotuned winner:
  - `N=2048`: `(4,2,2,2,4)`
  - `N=4096`: `(1,8,2,8,2)`
  - `N=8192`: `(1,8,2,8,2)`
- Current B path keeps 128-bit transport via identity-order prepack `(K/8, N, 8)` plus identity B LDS loads.

## Hardware Profiling Insights

### The FP16 vs FP8 Gap

We have established a highly optimized FP8 kernel (`scaled_mm_hip`) that achieves ~44 TFLOPS. The primary difference between our optimized FP8 kernel and this baseline FP16 kernel is the memory footprint and LDS layout:
1. **LDS Footprint:** FP16 data is 2 bytes per element, whereas FP8 is 1 byte. To load an equivalent `128x256` chunk, the FP16 kernel requires exactly twice as much LDS memory (`32 KB` vs `16 KB`).
2. **VGPR Footprint:** Loading and holding FP16 data requires twice as many vector registers (`uint4` loads 8 `fp16` elements instead of 16 `fp8` elements).
3. **LDS Banking:** The old swizzled-B baseline had a substantial LDS bank-conflict rate (`~14.8%`), but the current identity-order B path reduces that to `0.0%` without increasing LDS usage.
4. **Latency Hiding Limitation:** Because FP16 uses double the LDS memory, large block sizes like `(1, 8, 4, 8, 2)` (which corresponds to `128x256`) use nearly 32KB of LDS per wave group. This drastically lowers the hardware occupancy compared to FP8. Reduced occupancy means the macro-level scheduler cannot easily hide the `vmcnt` (global memory) stall latency, leading to a performance drop-off at large matrix sizes (e.g. ~27.4k GFLOPS at N=8192).

### P1: Identity-Order B Prepack/Load - KEEP

- Isolation profiling on the old swizzled-B baseline (`N=8192`, forced `(1,8,2,8,2)`) showed:
  - baseline (`rocprof_fp16_bank_base/`): `LDSBankConflict ~14.764%`
  - C-shuffle bypass (`rocprof_fp16_bank_noc/`): `~14.815%` -> C-shuffle is not the source
  - temporary A-swizzle variant (`rocprof_fp16_bank_a_swizzle/`): `~46.512%` -> A-side swizzle is actively worse
  - temporary B-identity-load variant (`rocprof_fp16_bank_b_identity/`): `0.0%` -> conflicts came from the B LDS load remap in `wmma_compute_stage`
- Implemented fix: change FP16 prepack to keep logical N order while preserving `(K/8, N, 8)` vec8 chunks, and change compute-side B LDS reads to identity mapping.
- Final validation (`rocprof_fp16_bank_identity_final/`, forced `(1,8,2,8,2)`): `LDSBankConflict = 0.0%`, average profiled kernel time `~37.9 ms`.
- Full benchmark result (`python benchmark_mm_hip_fp16.py`): `N=8192 ~27.4k` GFLOPS (up from `~26.7k`), with the winner config still `(1,8,2,8,2)`.
- Key takeaway: B prepack is useful for 128-bit transport, but it does not need the old swizzled-N order; identity-order prepack keeps the same LDS footprint and removes bank conflicts.

### P2: Bottleneck Profiling For `(1,8,2,8,2)` - KEEP analysis

- Verified PC sampling at `N=8192` (`fp16_pc_verify_8192_results.db`, forced `(1,8,2,8,2)`, dispatches `4+`) shows:
  - `Sync ~46.7%`
  - `LDS Read ~36.8%`
  - `WMMA ~8.6%`
  - `SALU ~4.3%`
- Top sampled instructions are dominated by chunk-boundary synchronization and LDS read issue pressure:
  - `s_barrier`
  - `s_waitcnt vmcnt(0)` / `s_waitcnt lgkmcnt(*)`
  - many `ds_load_b128`
- Reduced detailed ATT at `N=2048` (`thread_trace_fp16_valid_small/`) confirms the true long-latency events are global-memory waits, not WMMA or conversion:
  - `s_waitcnt vmcnt(1) ~223,978 cycles`
  - `s_waitcnt vmcnt(0) ~102,154 cycles`
  - aggregated trace latency is dominated by `Sync` (`~9.45M cycles`) while `WMMA` contributes only `~0.95M cycles`
- No-detail ATT at `N=8192` (`thread_trace_fp16_8192_nodetail/`) succeeds without GPU reset and is qualitatively wait-dominated, but still reports `Thread trace buffer full!`; it is useful for occupancy/wave-state analysis, not detailed per-instruction timing.
- PMC check (`rocprof_fp16_bottleneck_pmcs_try/`) shows:
  - `LDSBankConflict = 0.0`
  - `L2CacheHit ~32.5`
  - `VALUInsts = 11406`
- Main conclusion: after fixing B bank conflicts, the `(1,8,2,8,2)` kernel is bottlenecked by serialized chunk-boundary refill + synchronization. The current loop computes the whole current chunk, then hits a barrier, then fetches the next chunk, then hits another barrier. There is effectively no overlap between next-chunk global refill and current-chunk WMMA.
- Controlled C++ ablation benchmark (`tmp_fp16_analysis/mm_fp16_vram_lds_ablation.cu`, `tmp_fp16_analysis/benchmark_mm_fp16_vram_lds_ablation.cpp`) now isolates the chunk-refill path without Python/Torch overhead and without zero-input bias:
  - Forced config only: `(1,8,2,8,2)`
  - Kept method: load the first chunk from real random fp16 inputs, then for later chunks selectively **reuse** the already-random LDS contents instead of refilling from global memory; this avoids the known `WMMA-with-zeros` speedup while also avoiding synthetic-PRNG overhead in the timed loop.
  - Repeated `N=8192` runs (2 passes, 100 timed iters each) give mean kernel times:
    - `full`: `~40.16 ms` (`~27.38 TFLOPS`)
    - `reuse_a`: `~34.76 ms` (`delta ~5.41 ms`, `~13.5%` of full time)
    - `reuse_b`: `~26.51 ms` (`delta ~13.65 ms`, `~34.0%` of full time)
    - `reuse_ab`: `~24.96 ms` (`delta ~15.21 ms`, `~37.9%` of full time, `~44.06 TFLOPS`)
  - Interpretation:
    - Roughly `15 ms` of the `~40 ms` end-to-end kernel time is tied to the repeated global/L2->LDS refill path for this config.
    - The B-side refill dominates the exposed cost; that matches the larger per-chunk B payload (`16 KB`) versus A (`8 KB`) for `(1,8,2,8,2)`.
    - The A-only and B-only deltas are not additive (`5.4 + 13.7 > 15.2 ms`), so the refill costs share fixed sync/issue overhead and interact with each other.
    - With both refills removed, the kernel rises to `~44 TFLOPS`, which is strong evidence that the math/LDS-read body is already capable of near-FP8-class throughput once chunk refills are taken out of the critical path.
- Controlled C++ ablation benchmark for the compute-side LDS->VGPR path (`tmp_fp16_analysis/mm_fp16_lds_vgpr_ablation.cu`, `tmp_fp16_analysis/benchmark_mm_fp16_lds_vgpr_ablation.cpp`) now measures the *exposed* cost of the inner `ds_read -> VGPR` fragment loads while keeping global refill intact:
  - Forced config only: `(1,8,2,8,2)`
  - Kept method: always use real random fp16 A/B input tensors; inside the compute stage, selectively reuse already-loaded random fragments to remove part of the LDS->VGPR traffic without introducing zero operands.
  - To avoid changing occupancy with long-lived fragment caches, the A-side study uses a *pairwise* reuse mode (`reuse_a_p2`): the kernel loads A from LDS on even `rm` and reuses that fragment for the next odd `rm`, so **50% of A LDS->VGPR loads are removed**. The B-side study (`reuse_b`) loads only one of the two `rn` fragments and reuses it for the second tile, so **50% of B LDS->VGPR loads are removed**.
  - Repeated `N=8192` runs (2 passes, 100 timed iters each) give mean kernel times:
    - `full`: `~40.37 ms` (`~27.23 TFLOPS`)
    - `reuse_a_p2` (50% A LDS->VGPR removed): `~40.23 ms` (`delta ~0.14 ms`, `~0.36%`)
    - `reuse_b` (50% B LDS->VGPR removed): `~36.93 ms` (`delta ~3.44 ms`, `~8.53%`)
    - `reuse_ab_p2` (50% A + 50% B removed): `~37.71 ms` (`delta ~2.67 ms`, `~6.61%`)
  - Interpretation:
    - The exposed LDS->VGPR cost is heavily **B-side dominated** on this config.
    - A-side LDS->VGPR reads are close to noise-floor in this experiment; either their direct cost is very small, or they are largely hidden by the existing schedule.
    - The `reuse_b` delta implies that the full B LDS->VGPR fragment load cost is on the order of `~6-7 ms` if the effect scaled linearly, but treat that as a rough upper estimate rather than a strict decomposition.
    - The combined half-ablation is not additive, which means these inner LDS reads interact with the surrounding instruction schedule; simple per-side deltas should be read as *exposed* cost, not exact standalone service time.

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
- Full benchmark result (`python benchmark_mm_hip_fp16.py`) after adding a small/medium-N autotune guard:
  - `N=2048`: `~15.7k` GFLOPS, autotuned to `(4,2,2,2,4)`
  - `N=4096`: `~34.3k` GFLOPS, autotuned to `(1,8,2,8,2)`
  - `N=8192`: `~27.9k` GFLOPS, autotuned to `(1,8,2,8,2)`
- Net effect versus the previous baseline:
  - `N=8192` improved from about `27.4k` to about `27.9k` GFLOPS.
  - `N=4096` improved materially and moved onto the wider `128x256` tile.
  - `N=2048` stays on the narrower path after guarding the wide tile out of the runtime autotune set below `N=4096`.
- Two rejected sub-variants from this P3 search:
  - `REJECT`: rolling one-stage B prefetch with an array/ref-based buffer
    - correctness passed, but extracted metadata showed `(1,8,2,8,2)` inflated to `LDS=57344 B`, and `N=8192` collapsed to about `22.2k` GFLOPS.
    - regression reason: the intended register-prefetch state was effectively lowered into compiler-managed LDS, destroying occupancy.
  - `REJECT`: scalarized rolling one-stage B prefetch with per-stage commit
    - correctness passed and metadata returned to `LDS=24576 B`, but the added mid-chunk barrier/control cost limited `N=8192` to about `26.9k` GFLOPS.
    - regression reason: preserving the current LDS footprint was not enough; the extra synchronization erased the overlap gain.
- Why the small/medium-N autotune guard is kept:
  - On `N=2048`, the runtime `old_autotune` short probe window incorrectly preferred `(1,8,2,8,2)` even though the benchmark harness (`triton.testing.do_bench`) measured that config slower than `(4,2,2,2,4)` / `(4,2,4,2,4)`.
  - Restricting `(1,8,2,8,2)` to `N >= 4096` matches the observed crossover after P3 and removes the end-to-end `2048` regression.

### P5: Native B-Prefetch Sweep And Runtime Heuristic - KEEP

- Added native `auto|on|off` B-prefetch control to the FP16 benchmark/kernel path:
  - benchmark harness: `cpp_benchmarks/benchmark_mm_hip_fp16.cpp`
  - build script: `cpp_benchmarks/build_benchmark_mm_hip_fp16.sh`
  - sweep runner: `tmp_fp16_analysis/run_b_prefetch_sweep.py`
  - transition sweep runner: `tmp_fp16_analysis/run_b_prefetch_region_sweep.py`
  - raw sweep results: `tmp_fp16_analysis/b_prefetch_sweep.csv`
  - focused transition sweep log: `tmp_fp16_analysis/b_prefetch_region_sweep.txt`
- The kernel now supports the simplified policy requested for this search:
  - either prefetch **all** B stages in the next chunk,
  - or prefetch **none** of them.
- Sweep protocol:
  - run native C++ benchmark only (`--no-bias`) to avoid Python/Torch overhead,
  - test both `b_prefetch=off` and `b_prefetch=on`,
  - sweep all compatible autotune configs over these shapes:
    - target Qwen-Image shapes: `(32,12288,2048)`, `(32,2048,12288)`, `(8192,12288,2048)`, `(8192,2048,12288)`
    - extra crossover shapes: `(128,12288,2048)`, `(128,2048,12288)`, `(512,12288,2048)`, `(512,2048,12288)`, `(2048,12288,2048)`, `(2048,2048,12288)`
- Target-shape result summary from the sweep:
  - `(32,12288,2048)`: best `off ~6.99 TFLOPS`, best `on ~7.21 TFLOPS`
  - `(32,2048,12288)`: best `off ~3.09 TFLOPS`, best `on ~5.43 TFLOPS`
  - `(8192,12288,2048)`: best `off ~29.06 TFLOPS`, best `on ~28.37 TFLOPS`
  - `(8192,2048,12288)`: best `off ~34.48 TFLOPS`, best `on ~37.71 TFLOPS`
- Main sweep conclusions:
  - B prefetch is **not** a monotonic “large shape = on” optimization.
  - Long-K shapes (`K >= 8192`) benefit much more consistently than short-K shapes.
  - The strongest always-positive config families in this sweep were:
    - `(2,4,2,4,2)`
    - `(2,4,4,4,2)`
    - `(2,1,4,2,2)`
  - The wide `128x256` family `(1,8,*,8,2)` benefits on most shapes, but the large wide-N / short-K case `(8192,12288,2048)` is a real exception and should keep prefetch off.
  - Very small-`M` row paths need a small explicit exception: `(1,8,2,1,2)` helps on `(32,12288,2048)` even though the general short-K rule is more conservative.
- Focused transition sweep (`tmp_fp16_analysis/b_prefetch_region_sweep.txt`) tightened the boundaries:
  - short-K wide-N `128x256` tile `(1,8,2,8,2)` stays positive through `M=2048`, then flips negative by `M=4096`
  - long-K `128x128` / `warps_n=2` families split into two coarse regions:
    - balanced `warps_m == warps_n` stays positive only up to about `M=512`
    - asymmetric `warps_m != warps_n` is positive at very small `M` and again from about `M=2048`
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
    - `(32,12288,2048)` -> `(1,8,2,1,2)` at about `7.08 TFLOPS`
    - `(32,2048,12288)` -> `(1,1,4,1,2)` at about `5.38 TFLOPS`
    - `(8192,12288,2048)` -> `(1,8,2,8,2)` at about `29.99 TFLOPS`
    - `(8192,2048,12288)` -> `(4,2,4,2,4)` at about `39.02 TFLOPS`
- Keep reason:
  - this simpler region heuristic still matches the measured sweep winner on the studied shapes while avoiding the clear regression on the large wide-N / short-K case.

### P6: Native A-WSGR Sweep And Runtime Heuristic - KEEP

- Restored native FP16 benchmark controls and added explicit A-side WSGR mode selection:
  - benchmark harness: `cpp_benchmarks/benchmark_mm_hip_fp16.cpp`
  - kernel launch entrypoint: `launch_mm_fp16_tuning_mode(...)`
  - full sweep runner: `tmp_fp16_analysis/run_a_wsgr_sweep.py`
  - focused transition runner: `tmp_fp16_analysis/run_a_wsgr_region_sweep.py`
  - raw sweep results: `tmp_fp16_analysis/a_wsgr_sweep.csv`
  - focused transition log: `tmp_fp16_analysis/a_wsgr_region_sweep.txt`
  - target-shape auto validation: `tmp_fp16_analysis/a_wsgr_auto_validation.txt`
- Sweep protocol:
  - keep `B-prefetch=auto`,
  - benchmark only native C++ `--no-bias`,
  - compare `a_wsgr=off` vs `a_wsgr=on`,
  - sweep the 20 config families where A-side WSGR actually changes codegen (`block_warps_n > 1`),
  - shapes:
    - target / crossover set: `(32,12288,2048)`, `(32,2048,12288)`, `(128,12288,2048)`, `(128,2048,12288)`, `(512,12288,2048)`, `(512,2048,12288)`, `(2048,12288,2048)`, `(2048,2048,12288)`, `(8192,12288,2048)`, `(8192,2048,12288)`
- Main sweep result:
  - the old A-side rule was effectively just `repeat_m <= 4`; with B-prefetch left on auto, that is too broad.
  - A-WSGR is clearly **negative** on the wide `128x256` family `(1,8,*,8,2)`:
    - focused sweep confirms `(1,8,2,8,2)` is negative on every tested point, both short-K and long-K.
  - A-WSGR is clearly **positive** on the balanced square `128x128` / `u=2` family `(2,2,2,4,4)`:
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
- Auto validation on the four target Qwen-Image shapes (`tmp_fp16_analysis/a_wsgr_auto_validation.txt`):
  - `(32,12288,2048)` -> best `(1,8,2,1,2)` at about `7.23 TFLOPS`
  - `(32,2048,12288)` -> best `(1,1,4,1,2)` at about `5.40 TFLOPS`
  - `(8192,12288,2048)` -> best `(1,8,2,8,2)` at about `29.58 TFLOPS`
  - `(8192,2048,12288)` -> best `(4,2,4,2,4)` at about `38.21 TFLOPS`
- Keep reason:
  - this region heuristic is simpler than the old implicit `repeat_m <= 4` rule,
  - it preserves the measured best winner on the studied target shapes,
  - it avoids the large regressions from turning A-WSGR on for the wide `repeat_m=8` family.

### Durable Findings

- Zero-conflict B LDS access is achievable without padding or larger LDS allocation.
- For this FP16 kernel family, B swizzle was the source of LDS bank conflicts; identity-order B prepack is better.
- A-side swizzle is a bad direction on the current winner path (`~46.5%` conflict rate in the isolation build).
- The purpose of FP16 B prepack is to preserve vec8 / 128-bit transport, not specifically to preserve the old swizzled order.
- After removing LDS bank conflicts, the dominant large-`N` bottleneck is global-refill latency amplified by workgroup-wide synchronization, not WMMA throughput.
- A direct refill ablation on `(1,8,2,8,2)` attributes about `38%` of `N=8192` wall time to repeated global/L2->LDS chunk refill; B-side refill is the larger exposed component.
- A direct compute-side LDS->VGPR ablation shows the exposed inner-fragment load cost is much smaller than the global-refill cost and is mostly on the B side; A-side LDS->VGPR reads are near the noise floor in this schedule.
- `FP Convert` and general `VALU` are not important bottlenecks on the FP16 kernel (`PC sampling: FP Convert ~0.1%, VALU Other ~1.3%`).
- `LDS Read` remains a secondary issue-pressure hotspot, but the first thing to attack is the serialized refill/barrier structure around `s_waitcnt vmcnt(*)` and `s_barrier`.
- For unstable large-`N` ATT on gfx1151, use no-detail ATT first; detailed `N=8192` ATT is still unsafe on the current tooling stack.
- On gfx1151, practical latency hiding for this kernel should be treated as a **software-pipelining / occupancy** problem, not a search for a magical async global->LDS primitive.
- gfx10+ HIP compute defaults to **WGP mode**; CU mode is not the default path. That matters because some LDS-direct mechanisms are CU-mode-specific, so any CU-mode experiment should be treated as a separate, toolchain-dependent branch rather than the default optimization direction.
- Direct global->LDS is **not a primary near-term path** for this kernel on gfx1151:
  - the older async-to-LDS intrinsics are documented for gfx9/gfx10,
  - the newer `global.load.async.to.lds.b{8,32,64,128}` path is gfx1250+,
  - LLVM MC currently marks `global_load_lds_*` unsupported on gfx11.
- Even if a direct-to-LDS path becomes usable later, it carries strong layout constraints: each lane transfers one DWORD and lanes in a wave must write consecutive DWORDs into LDS. That does not map naturally onto the current vec8 / `uint4` fp16 transport without non-trivial repartitioning.
- Full AB ping-pong LDS buffering is risky for `(1,8,2,8,2)`: doubling LDS from `24576 B` to `49152 B` would cut the occupancy model from roughly `50%` to roughly `25%`, so it should not be the first overlap experiment.
- Because the refill ablation is strongly **B-side dominated**, the first overlap attempts should prioritize hiding **B refill** earlier than A refill, or more aggressively than A refill, before trying symmetric AB double-buffering.
- For this kernel family, explicit scalar `uint4` prefetch state is materially safer than array/ref-style prefetch buffers. The latter can trigger compiler-generated LDS growth even when the source change looks like a register-only prefetch.
- On gfx1151 for `(1,8,2,8,2)`, keeping the original two-barrier chunk structure mattered: the final kept P3 shape worked only after removing the extra mid-chunk barrier from the rolling prefetch attempt.
- Config crossover is driven by A reuse vs control overhead:
  - `(4,2,2,2,4)` / `(4,2,4,2,4)` remain the better small/medium-N benchmark-harness choices around `N=2048`.
  - `(1,8,2,8,2)` with kept P3 B-prefetch now wins at `N=4096..8192`.
  - The runtime autotune path should respect that crossover explicitly, because the short probe window can mis-rank configs near the boundary.
- For FP16 B-prefetch, the useful decision boundary is **config-sensitive plus K-sensitive**:
  - `K >= 8192` is the strongest generic signal for turning prefetch on.
  - `block_warps_n >= 4` is a good coarse long-K proxy for a prefetch-positive region.
  - balanced `warps_m == warps_n` narrow-N families only want B-prefetch at small `M`.
  - the large wide-N / short-K crossover is mainly an `M` threshold: keep prefetch on through about `M=2048`, then turn it off by `M=4096`.
- For A-side WSGR, the useful boundary is **tile-family-sensitive** rather than a generic `repeat_m <= 4` rule:
  - `repeat_m=8` wide tiles should keep WSGR off.
  - `repeat_m=1` paths are near noise-floor and are safer to keep off.
  - square `128x128` / `u=2` tiles are consistently WSGR-positive.
  - tall `128x128` / `repeat_m=2` tiles are WSGR-positive, but the `u=4` family needs `M >= 512`.
  - most N-heavy `repeat_m=4` families stay negative or too mixed to justify enabling by default.

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_mm_hip_fp16.py`
   - `python benchmark_mm_hip_fp16.py`
   - If it regresses, explain the reason by inspecting the generated code and/or profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Next Experiments

### P4: Add A-Light Overlap On Top Of Kept B Prefetch

- **Rationale:** B-prefetch is now working and large-N improved, so the remaining exposed chunk-boundary cost is most likely the A refill plus the waits/control still surrounding it.
- **Method:** Build only on top of the kept P3 kernel. Avoid reintroducing extra barriers or any compiler pattern that inflates LDS beyond `24576 B`.
- **Candidate directions:**
  - try a one-stage or partial A prefetch only if codegen keeps VGPR near the current `~190` level and preserves `LDS=24576 B`,
  - hoist only one A stage first, leaving the other on the existing path,
  - prefer scalarized state over arrays/references if any A prefetch buffer is introduced,
  - reject immediately if the change adds a third chunk barrier or pushes the wide tile back behind the `N=4096` crossover.
- **Decision:** Keep only if `N=8192` improves again and `2048` / `4096` stay at least as good as the current kept baseline.

### Future Branches Worth Revisiting Only With New Preconditions

- **F1: CU-mode / direct-to-LDS branch**
  - **Rationale:** RDNA supports some memory->LDS direct paths, but they are not the default practical path on gfx1151 today.
  - **Preconditions:** reliable CU-mode compilation path for this kernel, toolchain support that actually assembles and schedules the needed instructions on gfx1151, and generated code confirmation that the loads are truly direct-to-LDS.
  - **Reason for low priority:** current gfx1151 evidence points to software pipelining as the primary path, while the direct-to-LDS route is constrained or unsupported in the current gfx11 toolchain stack.
