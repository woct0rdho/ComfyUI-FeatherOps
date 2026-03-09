# gfx1151 HIP FP16 Matmul Kernel Optimization Plan

## Scope and Metric

- Prepack contract:
  - B is pre-transposed into identity-order `[K/16, 2, N, 8]` layout, preserving vec8 / 128-bit chunks.
  - The C++ kernel consumes this prepacked storage to perform direct loads into LDS without additional transposition.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance metric: `benchmark_mm_hip_fp16.py` GFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - `N=8192` does not regress,
  - large-N trend (`2048`, `4096`) does not regress materially.

## Current Baseline

- Latest full benchmark (`python benchmark_mm_hip_fp16.py`):
  - `N=8192`: `~27.4k` GFLOPS
- Current autotuned/profiled winner at `N=8192`: `(1,8,2,8,2)`
- Current B path keeps 128-bit transport via identity-order prepack `[K/16, 2, N, 8]` plus identity B LDS loads.

## Hardware Profiling Insights

### The FP16 vs FP8 Gap

We have established a highly optimized FP8 kernel (`scaled_mm_hip_prepacked`) that achieves ~44 TFLOPS. The primary difference between our optimized FP8 kernel and this baseline FP16 kernel is the memory footprint and LDS layout:
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
- Implemented fix: change FP16 prepack to keep logical N order while preserving `[K/16, 2, N, 8]` vec8 chunks, and change compute-side B LDS reads to identity mapping.
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

### Durable Findings

- Zero-conflict B LDS access is achievable without padding or larger LDS allocation.
- For this FP16 kernel family, B swizzle was the source of LDS bank conflicts; identity-order B prepack is better.
- A-side swizzle is a bad direction on the current winner path (`~46.5%` conflict rate in the isolation build).
- The purpose of FP16 B prepack is to preserve vec8 / 128-bit transport, not specifically to preserve the old swizzled order.
- After removing LDS bank conflicts, the dominant large-`N` bottleneck is global-refill latency amplified by workgroup-wide synchronization, not WMMA throughput.
- `FP Convert` and general `VALU` are not important bottlenecks on the FP16 kernel (`PC sampling: FP Convert ~0.1%, VALU Other ~1.3%`).
- `LDS Read` remains a secondary issue-pressure hotspot, but the first thing to attack is the serialized refill/barrier structure around `s_waitcnt vmcnt(*)` and `s_barrier`.
- For unstable large-`N` ATT on gfx1151, use no-detail ATT first; detailed `N=8192` ATT is still unsafe on the current tooling stack.
- Config crossover is driven by A reuse vs control overhead:
  - `(4,2,4,2,4)` = `128x128`, `chunkK=64`, `VGPR=144`, `LDS=32768 B`; it wins at `N=512..4096` because it halves the number of K refills / barriers and has lower per-wave state.
  - `(1,8,2,8,2)` = `128x256`, `chunkK=32`, `VGPR=184`, `LDS=24576 B`; it wins at `N=8192` because the wider `N` tile halves duplicated A-tile reloads across columns.
  - Forced profile check confirms the crossover: at `N=4096`, `(4,2,4,2,4)` is faster (`~3.69 ms` vs `~5.07 ms`), while at `N=8192`, `(1,8,2,8,2)` is faster (`~38.37 ms` vs `~43.37 ms`).

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

### P3: Overlap Next-Chunk Refill With Current-Chunk Compute

- **Rationale:** Profiling now shows the winner `(1,8,2,8,2)` is dominated by serialized refill + barrier cost. The most promising remaining headroom is to hide the next chunk's global-memory latency behind the current chunk's WMMA work.
- **Method:** Rework the main loop schedule so next-chunk A/B global loads begin earlier, while preserving correctness and preferably keeping the same LDS footprint / occupancy. Start with schedules that do not add extra stages or padding.
- **Decision:** Keep only if correctness passes and `N=8192` improves without materially regressing `2048` / `4096`.

### P4: If P3 Fails, Reduce Chunk-Boundary Control Cost

- **Rationale:** If true overlap is too expensive or unsafe, the fallback is to reduce the frequency or cost of chunk-boundary synchronization/refill events.
- **Method:** Explore alternatives that preserve the current `128x256` A-reuse advantage while reducing refill/barrier overhead, but do not re-open solved bank-conflict work.
- **Decision:** Keep only if the large-`N` metric improves and the crossover against `(4,2,4,2,4)` remains favorable at `N=8192`.
