# gfx1151 HIP Prepacked-B fp8e5m2 Matmul Kernel Optimization Plan

## Scope and Metric

- Prepack contract:
  - keep B physically as fp8 values in prepack output,
  - kernel consumes prepacked storage as raw bytes (`uint8_t*`) in device code.
- Python wrapper contract:
  - infer `b_dtype` from fp8 prepacked tensor dtype.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance metric: `benchmark_scaled_mm_hip.py` TFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - `N=8192` does not regress,
  - large-N trend (`2048`, `4096`) does not regress materially.

## Current Baseline

- Latest full benchmark (`python benchmark_scaled_mm_hip.py`):
  - `N=8192`: 44.0 TFLOPS
  - May be 42-44 TFLOPS due to thermal throttling

## Profiling Insights

### PC Sampling

Extensive PC sampling was conducted, moving from legacy `SQ_IND` (which read stale `SQ_WAVE_INST` buffers) to precise `ttmp0:1` host-trap, and finally zero-skid hardware stochastic sampling. With the optimal `(1,8,4,8,2)` configuration, the pure stall distribution is:
- **LDS Read (`ds_load_b128`):** 50% of samples
- **FP Convert (`v_perm_b32`):** 15%
- **Sync (`s_waitcnt` / barrier):** 14%
- **WMMA:** 10%

**Crucial Insight (Issue vs. Execution Latency):** Hardware stochastic sampling records the instruction the program counter (PC) is currently pointing to, which is the instruction *waiting to be issued*, not necessarily the instruction currently executing in the pipeline.
1. **The `v_perm` Fetch Stall:** Even though `v_wmma` executes in 32 cycles and `v_perm_b32` executes in 1 cycle, `v_perm_b32` receives significantly more PC samples! This happens because `v_perm_b32` with inline literal constants (e.g., `0x30c020c`) is a massive 96-bit (3 DWORD) instruction. Fetching 32 consecutive massive instructions for 8 concurrent waves completely chokes the sequencer's instruction fetch frontend. The PC gets stuck pointing at `v_perm_b32` waiting for instruction memory. This front-end starvation physically robs the pipeline of the clock cycles needed to issue `v_wmma`.
2. **The LDS Read Queue Stall:** `ds_load_b128` instructions account for 50% of all samples. A single `ds_load_b128` for a full wave (32 threads) requests 512 bytes, which takes the hardware LDS unit 4 clock cycles (at 128 bytes/cycle bandwidth) to process. With 8 waves constantly issuing these large loads, the internal LDS memory instruction queue becomes fully saturated. The sequencer must stall and wait for the LDS unit to drain its queue before it can issue the next `ds_load`. During this structural stall, the PC remains pointing at the blocked `ds_load` instruction, racking up massive sample counts.

### Thread Tracing

Cycle-accurate thread tracing was used to plot execution timelines, revealing the true underlying bottlenecks that statistical sampling obscured:
1. **Global Memory Wait (The `vmcnt` Gap):** Although PC sampling showed very few `global_load` samples (1%), thread tracing proved that individual waves experience massive `s_waitcnt` (sync) stalls waiting on `vmcnt`. A single wave's timeline reveals a **3,500-cycle bubble** between outer chunk iterations where it completely halts waiting for global memory to return from L2/VRAM. However, as detailed below, this single-wave stall does not translate to global starvation.
2. **Instruction Fetch Bottleneck (`v_perm_b32`):** The FP8->FP16 conversion uses `v_perm_b32` with inline 32-bit literal constants (e.g., `0x010c000cu`). This makes it a 96-bit (3 DWORD) instruction. When multiple waves attempt to unroll 16 of these massive instructions simultaneously, it overwhelms the SIMD instruction cache and fetch/decode frontend. What should take 16 cycles stretches into hundreds of wall-clock cycles, starving the VALU and heavily delaying WMMA issue. The apparent 1:4 time ratio of convert vs WMMA on the timeline (compared to the 1:32 theoretical ratio) is driven by this fetch stall, making it a severe secondary bottleneck.

### The Final Bottleneck

Through thread tracing and hardware occupancy analysis of the chosen autotune config `(1,8,4,8,2)`, we have proven the kernel is near the absolute limit of the hardware:
1. **Hardware Latency Hiding:** While an individual wave stalls for 3,500 cycles on global memory fetches, the `(1,8,4,8,2)` configuration achieves 50% occupancy. It uses 184 VGPRs (fitting 8 waves per SIMD within the 1536 limit) and 32 KB of LDS per 8-wave workgroup (fitting 4 workgroups into the 128 KB per-WGP limit). This allows 8 resident waves per SIMD. The hardware scheduler seamlessly context-switches among these 8 waves during memory stalls. Combined with macro-level scheduling staggering across the 80 SIMDs, the global memory latency is well hidden by the hardware without needing an explicit software pipeline.
2. **FP Convert:** RDNA3.5 (`gfx1151`) lacks native `fp8` instructions. Unpacking `fp8` data requires `v_perm_b32` (VALU), which cannot execute concurrently with `v_wmma` (matrix core). Thread tracing reveals that the VALU unpacking consumes **23%** of the math pipeline execution time, creating a hard ceiling of **77%** for matrix math (WMMA) execution time.
   - Theoretical Peak (assuming 100% WMMA time): 59.4 TFLOPS
   - Hard Cap (due to 77% WMMA time): 45.7 TFLOPS
   - Actually Achieved: 44.0 TFLOPS

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_scaled_mm_hip.py`
   - `python benchmark_scaled_mm_hip.py`
   - If it regresses, explain the reason by inspecting the generated code and/or profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Condensed Experiment Ledger

| ID | Keep? | Change | Key Result | Why |
|---|---|---|---|---|
| P0 | KEEP baseline | initial e5m2 baseline | `N=8192 42.86 TFLOPS` | starting point |
| P1 | REJECT | direct-write decode staging removal | `+0.15%` only | noise-level |
| P2 | REJECT | `cvt_pk_f16_fp8` builtin path | compile fail | requires gfx1250 insts |
| P3 | REJECT | add tile `(1,8,2,16,1)` | tiny `+0.24%`, unstable small-N | not robust |
| P4 | REJECT | remove load-phase `s_setprio` | large-N regression | metric down |
| P5 | KEEP analysis | no-scale/no-bias ablation | `+0.13%` delta | epilogue not bottleneck |
| P6 | KEEP analysis infra | `HIP_PREPACKED_OVERLAP_MODE` runtime modes | decomposition enabled | needed for overlap study |
| P6-A | KEEP analysis | overlap decomposition | low headroom (`2-3%`) | overlap headroom exists but limited |
| P6-B | KEEP analysis | SQ + extra PMCs | busy-cycles shift, VALU/LDS flat | no larger overlap |
| P7 | REJECT | `s_setprio 2` | profile tiny up, benchmark down | non-robust |
| P8 | REJECT config / KEEP infra | stage4 candidate + stage-index fix | correctness fixed, no speedup | keep fix, drop stage4 cfg |
| P9 | KEEP baseline refinement | prepack output kept fp8 + wrapper dtype inference | correctness pass, `N=8192 43.77 TFLOPS` | cleaner API, no perf regression |
| P10 | REJECT | stage2 split-phase software pipeline (winner path, A/B global->VGPR prefetch + wait + commit) | first version broke swizzle correctness; non-swizzle-restricted retry passed correctness but benchmark regressed (`N=8192 42.04 TFLOPS`) | overlap attempt cost exceeded gain; reverted |
| P11 | REJECT | stage2 interleaved refill (`compute stage0 -> refill0 -> compute stage1 -> refill1`) | correctness failed | unsafe stage reuse hazard in 2-stage path; reverted |
| P12 | REJECT | stage2 B-only split-phase prefetch (`prefetch B -> compute -> wait -> load A + commit B`) | correctness failed | hazard-prone on current schedule; reverted |
| P13 | REJECT | stage2 B-only split-phase retry with `volatile uint4` prefetch buffer | compile failed | invalid implementation direction; reverted |
| P14 | KEEP baseline simplification | remove swizzle toggle and overlap profiling modes; enforce stages=2 only | correctness pass; baseline maintained | matches winner path and scope |
| PCS-1/2/3 | KEEP analysis | PC sampling (SQ_IND, Host-Trap, Stochastic) | Consistent 45% LDS, 23% Sync stall distribution | Uncovered `SQ_WAVE_INST` drift issue, confirmed accurate stall breakdown |
| TT-1 | KEEP analysis | thread tracing timeline | Found 3,500-cycle gap between chunks | Proved wave is global-memory bound on `vmcnt`, plus instruction fetch bottleneck on `v_perm_b32` |
| P15 | REJECT | hoist `v_perm_b32` literals to VGPRs | `N=8192` regressed 42.62 TFLOPS | did not fix VOP3 issue latency (VGPR read ports), compiler barrier worsened global memory scheduling (`vmcnt` stalls up 18%) |
| P16 | KEEP | chunk size expansion (`unroll_k`=4,8) + `stages` cleanup | `N=8192` flat (43.1 TFLOPS), but HUGE gains on `N=1024..4096` (+15-35%) | verified code has no A/B double buffering (synchronous wait); increasing chunk size drastically reduces frequency of hitting the 3,500-cycle `vmcnt` stall |
| P17 | KEEP | remove `kBPad` and `kCPad` LDS padding | `N=8192` flat (44.6 TFLOPS) | `LDSBankConflict` PMC profiling proved bank conflicts are virtually zero (0.3%), padding was wasting LDS capacity without providing performance benefits |
| P18 | KEEP | mixed-precision A WSGR auto heuristic refresh | current-nightly replays: square `6/7` exact, broad `12/14` exact, residual loss `<=0.329 TFLOPS` | refreshed sweeps moved the surviving `repeat_m=1` win region to `u=4 && K<=1024`; keep tall `repeat_m=2` and square `repeat_m=4` rules conservative |
| P19 | KEEP analysis | mixed-precision VRAM->LDS refill ablation | exposed refill share is config-sensitive; current `8192^3` winner only gains `2%` from `reuse_ab` | unlike the older FP16 wide-tile path, current mixed winners are not uniformly refill-dominated |
| P20 | KEEP analysis | mixed-precision fp8->fp16 decode ablation | current `8192^3` winner only gains `1.7%` from skipping decode while keeping the LDS->VGPR B load intact | on the new nightly, exposed decode cost is real but still smaller than the earlier suspected front-end bound; the square winner remains only weakly sensitive to this decode path in isolation |
| P21 | KEEP analysis | mixed-precision LDS->VGPR fragment-load ablation | current `8192^3` winner gains `3.3%` from removing half of A LDS loads, `1.5%` from removing half of B LDS loads, and `6.1%` from removing half of both | compute-side fragment-load cost is measurable on the square winner, and the exposed A-side cost is larger than the exposed B-side cost on current nightly |
| P22 | REJECT | mixed-precision A-only global->VGPR prefetch sweep | giant wins only on weak fixed configs (up to `1.99x`), but autotuned target-shape delta was only `+0.20%`, `-14.73%`, `+1.56%`, `-0.59%` | not robust on the winner path; reverted |
| P23 | REJECT | mixed-precision B-only global->VGPR prefetch sweep | forced-`on` beat the new forced-`off` variants, but still lost to the last clean baseline on all four Qwen target shapes (`-1.15%`, `-20.91%`, `-3.23%`, `-4.26%` autotuned) | not a real keeper; reverted |

### P22: Mixed-Precision A-Only Prefetch Sweep - REJECT

- Goal: test the FP16-style split schedule on the mixed-precision kernel by prefetching only A (`global -> VGPR`) before WMMA, then committing A to LDS after compute while leaving B on the old synchronous refill path.
- Validation:
  - correctness passed with the forced-on experiment across all 28 configs;
  - analysis artifacts live under `tmp_mixed_precision_analysis/`, especially:
    - `qwen_shape_sweep_prefetch_off.csv` / `qwen_shape_sweep_prefetch_on.csv`
    - `orientation_region_prefetch_off.csv` / `orientation_region_prefetch_on.csv`
- Target Qwen-Image shapes:
  - `(32, 12288, 2048)`: autotuned `6.774 -> 6.788 TFLOPS` (`+0.20%`)
  - `(32, 2048, 12288)`: autotuned `6.419 -> 5.473 TFLOPS` (`-14.73%`)
  - `(8192, 12288, 2048)`: autotuned `41.199 -> 41.842 TFLOPS` (`+1.56%`)
  - `(8192, 2048, 12288)`: autotuned `37.919 -> 37.695 TFLOPS` (`-0.59%`)
- Best fixed-config outliers:
  - max relative win: `(8192, 2048, 12288)` with `(2,1,2,2,2)` improved `7.910 -> 15.762 TFLOPS` (`+99.25%`)
  - max absolute win: `(8192, 12288, 2048)` with `(2,1,2,2,2)` improved `16.583 -> 30.384 TFLOPS` (`+83.23%`)
- Interpretation:
  - the large gains are limited to weak, non-winning configs and do not translate into a meaningful end-to-end autotuned gain;
  - in the short-K / wide-N orientation, some tall/wide configs such as `(1,8,4,8,2)` improved by `3-5%` once `M >= 1024`, but the strongest square winner `(2,4,4,4,4)` regressed by `2-4%`;
  - in the long-K / narrow-N orientation, the prefetch path was usually negative on good configs, with several double-digit regressions.
- Decision:
  - do not keep A prefetch in the mixed-precision kernel on the current nightly;
  - the shape/config dependence is too strong, and the actual autotuned winner path does not improve enough to justify a runtime heuristic.

### P23: Mixed-Precision B-Only Prefetch Sweep - REJECT

- Goal: port the kept FP16 split schedule to the mixed-precision kernel by prefetching only B (`global -> VGPR`) before WMMA, then committing B into LDS after compute while refilling A on the old path.
- Validation:
  - correctness passed both with the default path and with forced `FEATHER_SCALED_MM_B_PREFETCH=on`;
  - analysis artifacts live under `tmp_mixed_precision_analysis/`, especially:
    - `qwen_shape_bprefetch_off.csv` / `qwen_shape_bprefetch_on.csv`
- Important measurement note:
  - within the patched kernel, forced `on` looked much faster than forced `off` on many shapes, but that comparison was misleading because the non-prefetch instantiations were materially slower than the last clean pre-change baseline;
  - keeper decisions therefore used the last clean baseline (`qwen_shape_baseline_sweep_before.csv`) rather than the patched forced-`off` variants.
- Target Qwen-Image shapes, autotuned comparison versus the clean pre-change baseline:
  - `(32, 12288, 2048)`: `6.790 -> 6.712 TFLOPS` (`-1.15%`)
  - `(32, 2048, 12288)`: `6.400 -> 5.062 TFLOPS` (`-20.91%`)
  - `(8192, 12288, 2048)`: `41.300 -> 39.965 TFLOPS` (`-3.23%`)
  - `(8192, 2048, 12288)`: `36.804 -> 35.238 TFLOPS` (`-4.26%`)
- Interpretation:
  - despite large apparent `on` vs patched-`off` gains, the prefetch path did not beat the last clean winner path on any of the target Qwen shapes;
  - the strongest pre-change winners were still better than the prefetched winners on all four target shapes, so there is no heuristic worth shipping from this experiment.
- Decision:
  - do not keep B prefetch in the mixed-precision kernel on the current nightly;
  - revert the experiment and treat any future revisit as requiring a new precondition plus a cleaner non-prefetch control for benchmarking.

## Durable Findings

- The `v_perm_b32` e5m2 unpacking acts as a severe fetch/decode bottleneck due to 96-bit bloated instructions.
- The winner path is not DRAM-bandwidth-limited; remaining headroom is dominated by on-chip issue/decode/scheduling and math-pipeline constraints.
- Epilogue (scale/bias) is not the bottleneck (`0.13%` delta).
- Overlap decomposition showed limited headroom because the internal loop is already perfectly overlapping WMMA and LDS.
- Removing `v_perm_b32` textually is not sufficient; replacement decode sequences can consume similar or worse front-end / VALU budget and fail to improve benchmark performance.
- Moving fp8 decode into the load phase is a poor direction because it shifts work into a less-overlapped region and tends to increase payload / pressure.
- Finer chunking tends to lose to control overhead on this kernel family; smaller chunks or more frequent refill schedules need a clearly new precondition to be viable.
- Hot-loop address generation is a low-probability target; prior non-prepacked ASM analysis showed most address work was already hoisted into precomputed bases plus compile-time offsets.
- `V_PK_*` / VOP3P packed-half instructions are not VOPD-eligible, so conversion rewrites that depend on packed 16-bit ops gaining dual-issue are unlikely to help.
- `ds_permute_b32` only permutes within a wave32; it cannot be used for cross-wave fragment sharing or broadcast across the 8-wave workgroup.
- Rigid inline ASM / manual instruction ordering can block compiler interleaving and worsen `vmcnt` hiding or register scheduling; use ASM only when it unlocks an otherwise impossible instruction form.
- Mixed-precision A WSGR should stay conservative and region-based on current ROCm nightly:
  - `repeat_m == 1`: enable only for `unroll_k == 4 && K <= 1024`
  - `repeat_m == 2`: enable only for tall tiles (`block_warps_m > block_warps_n`) with `unroll_k == 2`
  - `repeat_m == 4`: enable only for square tiles (`block_warps_m == block_warps_n`) with `unroll_k == 2`
  - Otherwise keep it off; the remaining misses in refreshed sweeps were small enough to ignore.
- Mixed-precision global->LDS refill cost is strongly config-dependent on the current nightly:
  - wide `repeat_m=1` path `(1,8,4,1,2)` at `2048^3` is clearly B-dominated: `reuse_a 4%`, `reuse_b 14%`, `reuse_ab 15%`
  - balanced mid-size path `(1,4,4,4,2)` at `4096^3` is roughly even between A and B: both are `6%`
  - tall WSGR-on path `(4,2,2,2,4)` at `(8192,2048,12288)` is A-dominated: `reuse_a 15%`, `reuse_b 8%`
  - current large square winner `(2,4,4,4,4)` at `8192^3` only gains `2%` from removing both refills, so refill is no longer the dominant exposed bottleneck there
- Mixed-precision fp8->fp16 decode cost on the current large square winner is also modest in isolation:
  - on `8192^3` with `(2,4,4,4,4)`, skipping only the decode while preserving the LDS `uint4` load path reduced time from `24.251 ms` to `23.846 ms`, an exposed delta of `0.405 ms` (`1.7%`)
  - this ablation intentionally feeds WMMA garbage half bits derived from the raw fp8 bytes, so it is only an exposed-cost estimate, not a numerically valid kernel
- Mixed-precision LDS->VGPR fragment-load cost is more visible than decode on the current `8192^3` square winner:
  - on `8192^3` with `(2,4,4,4,4)`, replacing every other A fragment load with a fixed synthetic fragment reduced time from `24.418 ms` to `23.621 ms`, an exposed delta of `0.797 ms` (`3.3%`)
  - replacing every other B fragment load with a fixed synthetic fragment, while keeping the fp8->fp16 decode count unchanged, reduced time from `24.418 ms` to `24.063 ms`, an exposed delta of `0.355 ms` (`1.5%`)
  - replacing every other A and B fragment load together reduced time to `22.933 ms`, an exposed delta of `1.485 ms` (`6.1%`)
  - this is still an ablation, not a valid kernel: the removed loads are replaced with fixed synthetic fragments, so treat the numbers as exposed-load estimates rather than a strict additive decomposition
- Mixed-precision A-only prefetch is not a good keeper on the current nightly:
  - it can produce huge gains on weak fixed configs, but the best measured autotuned improvement on the target Qwen-Image shapes was only `+1.6%`, and two of the four target shapes regressed
  - the short-K / wide-N orientation has a small positive region for some tall/wide configs, but the strongest square winner still regresses there
  - the long-K / narrow-N orientation is broadly negative on strong configs, so there is no robust runtime heuristic worth shipping from the current data
- Mixed-precision B-only prefetch is also not a keeper on the current nightly:
  - comparing only forced `on` vs forced `off` inside the patched kernel can overstate the benefit because the non-prefetch instantiations themselves can move;
  - against the last clean baseline, the prefetched autotuned winner regressed on all four target Qwen-Image shapes, including the two large target shapes.

## Do-Not-Repeat (Unless New Preconditions)

- P1 direct-write decode change (no new condition).
- Any `__builtin_amdgcn_cvt_pk_f16_fp8` path on gfx1151.
- Removing load-phase `s_setprio` or tweaking to `s_setprio 2`.
- Stage2 split-phase, interleaved, or B-only overlapping schedules, or even `stages > 2`.
- Optimizing inner-loop LDS access patterns (thread tracing shows WMMA/LDS already overlap perfectly).
- Load-phase fp8->fp16 conversion / decode hoisting.
- "Remove `v_perm_b32`" rewrites unless the full replacement sequence clearly reduces total issue/front-end cost.
- Finer-grained chunking / extra-control schedules unless a new precondition reduces the control cost.
- Any conversion rewrite whose thesis depends on `V_PK_*` / VOP3P dual-issue.
- Cross-wave broadcast/shuffle ideas built on `ds_permute_b32`.
- Hot-loop address-calc micro-optimizations unless new generated code shows materially higher address VALU than the prior kernels.
- Mixed-precision A-only prefetch on the current nightly, unless a new precondition changes the winner-config set or shows a materially larger autotuned gain on the real Qwen-Image shapes.
- Mixed-precision B-only prefetch on the current nightly, unless a new precondition both restores a clean non-prefetch baseline and shows a real autotuned gain on the actual Qwen-Image target shapes.

## Reference

- `~/amd-llvm-project/`, especially `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst` - hipcc source code
- `~/rdna35-isa-markdown/`
