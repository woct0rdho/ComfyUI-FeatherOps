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
  - `N=8192`: `~27.4k` GFLOPS
- Current autotuned/profiled winner at `N=8192`: `(1,8,2,8,2)`
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

### P3: Software-Pipeline Next-Chunk Refill Under Current-Chunk WMMA

- **Rationale:** Profiling now shows the winner `(1,8,2,8,2)` is dominated by serialized refill + barrier cost. The most promising remaining headroom is to hide the next chunk's global-memory latency behind the current chunk's WMMA work.
- **Method:** Rework the main loop schedule so next-chunk global loads are issued earlier and only waited on near first use. Prioritize schedules that preserve the current `24576 B` LDS footprint and current occupancy model.
- **Order of attack:**
  - **P3-A:** `B-first overlap without extra LDS stage`
    - Start with the B path, because the refill ablation shows the exposed refill cost is B-dominated.
    - Goal: issue next-chunk B global loads during current-chunk WMMA and delay `vmcnt` waits until B is about to be consumed.
    - Preferred forms: register prefetch, partial-prefetch, or wave-specialized B refill that does **not** add a full extra LDS stage.
  - **P3-B:** `Asymmetric overlap (B aggressive, A conservative)`
    - If P3-A helps but is incomplete, let B use the more aggressive overlap scheme while keeping A on a lighter-weight path.
    - Rationale: A refill is smaller and A LDS->VGPR cost is near noise-floor in the current schedule, so symmetric AB treatment is unlikely to be optimal.
  - **P3-C:** `Symmetric AB overlap only if lighter schemes stall`
    - Only after P3-A/P3-B fail to capture enough headroom, try fuller AB overlap schemes.
    - Treat full LDS ping-pong as high-risk because doubling LDS to `49152 B` would likely reduce occupancy from about `50%` to about `25%` for this config.
- **Decision:** Keep only if correctness passes and `N=8192` improves without materially regressing `2048` / `4096`.

### P4: If P3 Fails, Reduce Chunk-Boundary Control Cost

- **Rationale:** If true overlap is too expensive or unsafe, the fallback is to reduce the frequency or cost of chunk-boundary synchronization/refill events.
- **Method:** Explore alternatives that preserve the current `128x256` A-reuse advantage while reducing refill/barrier overhead, but do not re-open solved bank-conflict work.
- **Candidate directions:**
  - reduce unnecessary whole-workgroup synchronization around refill/consume handoff,
  - move waits later and make them more local/partial where correctness allows,
  - shrink the exposed B refill slice without changing the winning macro-tile shape,
  - consider modest control-structure changes that reduce chunk-boundary overhead without adding a second full LDS stage.
- **Decision:** Keep only if the large-`N` metric improves and the crossover against `(4,2,4,2,4)` remains favorable at `N=8192`.

### Future Branches Worth Revisiting Only With New Preconditions

- **F1: CU-mode / direct-to-LDS branch**
  - **Rationale:** RDNA supports some memory->LDS direct paths, but they are not the default practical path on gfx1151 today.
  - **Preconditions:** reliable CU-mode compilation path for this kernel, toolchain support that actually assembles and schedules the needed instructions on gfx1151, and generated code confirmation that the loads are truly direct-to-LDS.
  - **Reason for low priority:** current gfx1151 evidence points to software pipelining as the primary path, while the direct-to-LDS route is constrained or unsupported in the current gfx11 toolchain stack.
