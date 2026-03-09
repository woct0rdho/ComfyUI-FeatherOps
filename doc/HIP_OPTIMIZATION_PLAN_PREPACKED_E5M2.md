# gfx1151 HIP Prepacked-B fp8e5m2 Matmul Kernel Optimization Plan

## Scope and Metric

- Prepack contract:
  - keep B physically as fp8 values in prepack output,
  - kernel consumes prepacked storage as raw bytes (`uint8_t*`) in device code.
- Python wrapper contract:
  - infer `b_dtype` from fp8 prepacked tensor dtype.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance metric: `benchmark_scaled_mm_hip_prepacked_e5m2.py` GFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - `N=8192` does not regress,
  - large-N trend (`2048`, `4096`) does not regress materially.

## Current Baseline

- Latest full benchmark (`python benchmark_scaled_mm_hip_prepacked_e5m2.py`):
  - `N=8192`: ~44.0 TFLOPS

## Profiling Insights

### PC Sampling

Extensive PC sampling was conducted, moving from legacy `SQ_IND` (which read stale `SQ_WAVE_INST` buffers) to precise `ttmp0:1` host-trap, and finally zero-skid hardware stochastic sampling. With the optimal `(1,8,4,8,2)` configuration, the pure stall distribution is:
- **LDS Read (`ds_load_b128`):** ~50% of samples
- **FP Convert (`v_perm_b32`):** ~15%
- **Sync (`s_waitcnt` / barrier):** ~14%
- **WMMA:** ~10%

**Crucial Insight (Issue vs. Execution Latency):** Hardware stochastic sampling records the instruction the program counter (PC) is currently pointing to, which is the instruction *waiting to be issued*, not necessarily the instruction currently executing in the pipeline.
1. **The `v_perm` Fetch Stall:** Even though `v_wmma` executes in 32 cycles and `v_perm_b32` executes in 1 cycle, `v_perm_b32` receives significantly more PC samples! This happens because `v_perm_b32` with inline literal constants (e.g., `0x30c020c`) is a massive 96-bit (3 DWORD) instruction. Fetching 32 consecutive massive instructions for 8 concurrent waves completely chokes the sequencer's instruction fetch frontend. The PC gets stuck pointing at `v_perm_b32` waiting for instruction memory. This front-end starvation physically robs the pipeline of the clock cycles needed to issue `v_wmma`.
2. **The LDS Read Queue Stall:** `ds_load_b128` instructions account for ~50% of all samples. A single `ds_load_b128` for a full wave (32 threads) requests 512 bytes, which takes the hardware LDS unit 4 clock cycles (at 128 bytes/cycle bandwidth) to process. With 8 waves constantly issuing these large loads, the internal LDS memory instruction queue becomes fully saturated. The sequencer must stall and wait for the LDS unit to drain its queue before it can issue the next `ds_load`. During this structural stall, the PC remains pointing at the blocked `ds_load` instruction, racking up massive sample counts.

### Thread Tracing

Cycle-accurate thread tracing was used to plot execution timelines, revealing the true underlying bottlenecks that statistical sampling obscured:
1. **Global Memory Wait (The `vmcnt` Gap):** Although PC sampling showed very few `global_load` samples (~1%), thread tracing proved that individual waves experience massive `s_waitcnt` (sync) stalls waiting on `vmcnt`. A single wave's timeline reveals a **~3,500 cycle bubble** between outer chunk iterations where it completely halts waiting for global memory to return from L2/VRAM. However, as detailed below, this single-wave stall does not translate to global starvation.
2. **Instruction Fetch Bottleneck (`v_perm_b32`):** The FP8->FP16 conversion uses `v_perm_b32` with inline 32-bit literal constants (e.g., `0x010c000cu`). This makes it a 96-bit (3 DWORD) instruction. When multiple waves attempt to unroll 16 of these massive instructions simultaneously, it overwhelms the SIMD instruction cache and fetch/decode frontend. What should take 16 cycles stretches into hundreds of wall-clock cycles, starving the VALU and heavily delaying WMMA issue. The apparent 1:4 time ratio of convert vs WMMA on the timeline (compared to the 1:32 theoretical ratio) is driven by this fetch stall, making it a severe secondary bottleneck.

### The Final Bottleneck

Through thread tracing and hardware occupancy analysis of the chosen autotune config `(1,8,4,8,2)`, we have proven the kernel is near the absolute limit of the hardware:
1. **Hardware Latency Hiding:** While an individual wave stalls for ~3,500 cycles on global memory fetches, the `(1,8,4,8,2)` configuration achieves 50% occupancy. It uses 184 VGPRs (fitting 8 waves per SIMD within the 1536 limit) and 32 KB of LDS per 8-wave workgroup (fitting 4 workgroups into the 128 KB per-WGP limit). This allows 8 resident waves per SIMD. The hardware scheduler seamlessly context-switches among these 8 waves during memory stalls. Combined with macro-level scheduling staggering across the 80 SIMDs, the global memory latency is well hidden by the hardware without needing an explicit software pipeline.
2. **FP Convert:** RDNA3.5 (`gfx1151`) lacks native `fp8` instructions. Unpacking `fp8` data requires `v_perm_b32` (VALU), which cannot execute concurrently with `v_wmma` (matrix core). Thread tracing reveals that the VALU unpacking consumes **~23%** of the math pipeline execution time, creating a hard ceiling of **~77%** for matrix math (WMMA) execution time.
   - Theoretical Peak (assuming 100% WMMA time): ~59.4 TFLOPS
   - Hard Cap (due to ~77% WMMA time): ~45.7 TFLOPS
   - Actually Achieved: ~44.0 TFLOPS

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_scaled_mm_hip_prepacked_e5m2.py`
   - `python benchmark_scaled_mm_hip_prepacked_e5m2.py`
   - If it regresses, explain the reason by inspecting the generated code and/or profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Condensed Experiment Ledger

| ID | Keep? | Change | Key Result | Why |
|---|---|---|---|---|
| P0 | KEEP baseline | initial e5m2 baseline | `N=8192 ~42.86k` | starting point |
| P1 | REJECT | direct-write decode staging removal | `~+0.15%` only | noise-level |
| P2 | REJECT | `cvt_pk_f16_fp8` builtin path | compile fail | requires gfx1250 insts |
| P3 | REJECT | add tile `(1,8,2,16,1)` | tiny `~+0.24%`, unstable small-N | not robust |
| P4 | REJECT | remove load-phase `s_setprio` | large-N regression | metric down |
| P5 | KEEP analysis | no-scale/no-bias ablation | `~+0.13%` delta | epilogue not bottleneck |
| P6 | KEEP analysis infra | `HIP_PREPACKED_OVERLAP_MODE` runtime modes | decomposition enabled | needed for overlap study |
| P6-A | KEEP analysis | overlap decomposition | low headroom (`~2-3%`) | overlap headroom exists but limited |
| P6-B | KEEP analysis | SQ + extra PMCs | busy-cycles shift, VALU/LDS flat | no larger overlap |
| P7 | REJECT | `s_setprio 2` | profile tiny up, benchmark down | non-robust |
| P8 | REJECT config / KEEP infra | stage4 candidate + stage-index fix | correctness fixed, no speedup | keep fix, drop stage4 cfg |
| P9 | KEEP baseline refinement | prepack output kept fp8 + wrapper dtype inference | correctness pass, `N=8192 ~43.77k` | cleaner API, no perf regression |
| P10 | REJECT | stage2 split-phase software pipeline (winner path, A/B global->VGPR prefetch + wait + commit) | first version broke swizzle correctness; non-swizzle-restricted retry passed correctness but benchmark regressed (`N=8192 ~42.04k`) | overlap attempt cost exceeded gain; reverted |
| P11 | REJECT | stage2 interleaved refill (`compute stage0 -> refill0 -> compute stage1 -> refill1`) | correctness failed | unsafe stage reuse hazard in 2-stage path; reverted |
| P12 | REJECT | stage2 B-only split-phase prefetch (`prefetch B -> compute -> wait -> load A + commit B`) | correctness failed | hazard-prone on current schedule; reverted |
| P13 | REJECT | stage2 B-only split-phase retry with `volatile uint4` prefetch buffer | compile failed | invalid implementation direction; reverted |
| P14 | KEEP baseline simplification | remove swizzle toggle and overlap profiling modes; enforce stages=2 only | correctness pass; baseline maintained | matches winner path and scope |
| PCS-1/2/3 | KEEP analysis | PC sampling (SQ_IND, Host-Trap, Stochastic) | Consistent 45% LDS, 23% Sync stall distribution | Uncovered `SQ_WAVE_INST` drift issue, confirmed accurate stall breakdown |
| TT-1 | KEEP analysis | thread tracing timeline | Found ~3,500 cycle gap between chunks | Proved wave is global-memory bound on `vmcnt`, plus instruction fetch bottleneck on `v_perm_b32` |
| P15 | REJECT | hoist `v_perm_b32` literals to VGPRs | `N=8192` regressed ~42.62k | did not fix VOP3 issue latency (VGPR read ports), compiler barrier worsened global memory scheduling (`vmcnt` stalls up 18%) |
| P16 | KEEP | chunk size expansion (`unroll_k`=4,8) + `stages` cleanup | `N=8192` flat (~43.1k), but HUGE gains on `N=1024..4096` (+15-35%) | verified code has no A/B double buffering (synchronous wait); increasing chunk size drastically reduces frequency of hitting the 3,500-cycle `vmcnt` stall |
| P17 | KEEP | remove `kBPad` and `kCPad` LDS padding | `N=8192` flat (~44.6k) | `LDSBankConflict` PMC profiling proved bank conflicts are virtually zero (~0.3%), padding was wasting LDS capacity without providing performance benefits |

## Durable Findings

- The `v_perm_b32` e5m2 unpacking acts as a severe fetch/decode bottleneck due to 96-bit bloated instructions.
- The winner path is not DRAM-bandwidth-limited; remaining headroom is dominated by on-chip issue/decode/scheduling and math-pipeline constraints.
- Epilogue (scale/bias) is not the bottleneck (`~0.13%` delta).
- Overlap decomposition showed limited headroom because the internal loop is already perfectly overlapping WMMA and LDS.
- Removing `v_perm_b32` textually is not sufficient; replacement decode sequences can consume similar or worse front-end / VALU budget and fail to improve benchmark performance.
- Moving fp8 decode into the load phase is a poor direction because it shifts work into a less-overlapped region and tends to increase payload / pressure.
- Finer chunking tends to lose to control overhead on this kernel family; smaller chunks or more frequent refill schedules need a clearly new precondition to be viable.
- Hot-loop address generation is a low-probability target; prior non-prepacked ASM analysis showed most address work was already hoisted into precomputed bases plus compile-time offsets.
- `V_PK_*` / VOP3P packed-half instructions are not VOPD-eligible, so conversion rewrites that depend on packed 16-bit ops gaining dual-issue are unlikely to help.
- `ds_permute_b32` only permutes within a wave32; it cannot be used for cross-wave fragment sharing or broadcast across the 8-wave workgroup.
- Rigid inline ASM / manual instruction ordering can block compiler interleaving and worsen `vmcnt` hiding or register scheduling; use ASM only when it unlocks an otherwise impossible instruction form.

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

## Reference

- `~/amd-llvm-project/`, especially `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst` - hipcc source code
- `~/rdna35-isa-markdown/`
