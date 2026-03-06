# HIP Prepacked-B fp8e5m2 Optimization Plan (gfx1151)

## Scope and metric

- Target kernel path: prepacked-B HIP (`b_dtype=1`, fp8e5m2) on gfx1151.
- Prepack contract:
  - keep B physically as fp8 values in prepack output,
  - kernel consumes prepacked storage as raw bytes (`uint8_t*`) in device code.
- Python wrapper contract:
  - infer `b_dtype` from fp8 prepacked tensor dtype.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance metric: `benchmark_scaled_mm_hip_prepacked_e5m2.py` GFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - forced `N=8192` does not regress,
  - large-N trend (`4096`, `8192`) does not regress materially.

## Current Baseline Snapshot

- Correctness:
  - `python test_scaled_mm_hip_prepacked_e5m2.py` -> `5/5` pass.
  - `python test_scaled_mm_hip_prepacked.py` -> `5/5` pass.
- Latest full benchmark (`python benchmark_scaled_mm_hip_prepacked_e5m2.py`):
  - `N=4096`: `~35.89k` GFLOPS
  - `N=8192`: `~42.68k` GFLOPS
- Forced-config anchor (`HIP_FORCE_CONFIG=1,8,2,2,8,2`):
  - `N=8192`: `24.996 ms` (`~43.99k` GFLOPS).
- Latest profile anchor:
  - `rocprofv3 --kernel-trace --stats -d tmp_fp8e5m2_analysis/p14_stage2_noswizzle_only_profile -o p14_stage2_noswizzle_only_profile -- python -u profile_scaled_mm_hip_prepacked_e5m2.py`
  - target kernel average: `~25.653 us`.

## Hardware Profiling Insights

### PC Sampling Evolution (SQ_IND -> Host-Trap -> Stochastic)

Extensive PC sampling was conducted, moving from legacy SQ_IND (which read stale `SQ_WAVE_INST` buffers) to precise `ttmp0:1` host-trap, and finally zero-skid hardware stochastic sampling (~827K samples).
All modern methods consistently agree on the stall distribution:
- **LDS Read:** ~45% of samples
- **Sync (waitcnt/barrier):** ~23-24%
- **FP Convert (`v_perm_b32`):** ~10-11%
- **WMMA:** ~8-9%

While PC sampling showed LDS was the dominant memory operation within the active wave execution, it did not fully explain the *duration* of the sync stalls or the reason for the low WMMA throughput.

### Thread Tracing (ATT) Ground Truth

Cycle-accurate Thread Tracing (`--att`) was used to plot execution timelines, revealing the true underlying bottlenecks that statistical sampling obscured:

1. **Global Memory Starvation (The `vmcnt` Gap)**
   Although PC Sampling showed very few `global_load` samples (~1%), Thread Tracing proved that the massive `s_waitcnt` (Sync) stalls are actually waiting on `vmcnt(3)` and `vmcnt(1)`. The timeline reveals a **~3,500 cycle bubble** between outer chunk iterations where the kernel completely halts waiting for global memory to return from L2/VRAM. The `stages=2` double-buffering perfectly hides LDS latency *within* the chunk, but is insufficient to hide the global load latency across chunks.

2. **Instruction Fetch Bottleneck (`v_perm_b32`)**
   The FP8->FP16 conversion uses `v_perm_b32` with inline 32-bit literal constants (e.g., `0x010c000cu`). This makes it a 96-bit (3 DWORD) instruction. When 30+ waves attempt to unroll 16 of these massive instructions simultaneously, it overwhelms the SIMD instruction cache and fetch/decode frontend. What should take 16 cycles stretches into hundreds of wall-clock cycles, starving the VALU and heavily delaying WMMA issue. The apparent 1:4 time ratio of convert vs WMMA on the timeline (compared to the 1:32 theoretical ratio) is driven by this fetch stall, making it a severe secondary bottleneck.

## Durable Findings (Keep in Mind)

- The kernel is definitively bounded by **Global Memory Latency/Bandwidth at the fetch edges** between chunk iterations, causing massive `vmcnt` wait bubbles.
- `stages=2` software pipelining is too shallow to hide global memory latency on this hardware.
- The `v_perm_b32` e5m2 unpacking acts as a severe fetch/decode bottleneck due to 96-bit bloated instructions.
- Epilogue (scale/bias) is not the bottleneck (`~0.13%` delta).
- Overlap decomposition showed limited headroom because the internal loop is already perfectly overlapping WMMA and LDS; the problem is the inter-loop global boundary.
- Platform prior (from `doc/HIP_OPTIMIZATION_PLAN.md` + `kernel/hip/hip_kernel.cu`):
  - stage>2 historically hurts on gfx1151 via occupancy loss, but this was tested before understanding the exact `vmcnt` starvation.
  - prior stage2 split-phase attempts with extra sync overhead regressed.

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

## Reference Note (When in Doubt)

- You may consult ISA/compiler references before changing low-level decode/packing logic:
  - `~/amd-llvm-project/`
  - `~/rdna35-isa-markdown/`

## Condensed Experiment Ledger

| ID | Keep? | Change | Key Result | Why |
|---|---|---|---|---|
| P0 | KEEP baseline | initial e5m2 baseline | `N=8192 ~42.86k` | starting point |
| P1 | REJECT | direct-write decode staging removal | `~+0.15%` only | noise-level |
| P2 | REJECT | `cvt_pk_f16_fp8` builtin path | compile fail | requires gfx1250 insts |
| P3 | REJECT | add tile `(1,8,2,2,16,1)` | tiny `~+0.24%`, unstable small-N | not robust |
| P4 | REJECT | remove load-phase `s_setprio` | large-N regression | metric down |
| P5 | KEEP analysis | no-scale/no-bias ablation | `~+0.13%` delta | epilogue not bottleneck |
| P6 | KEEP analysis infra | `HIP_PREPACKED_OVERLAP_MODE` runtime modes | decomposition enabled | needed for overlap study |
| P6-A | KEEP analysis | overlap decomposition | low headroom (`~2-3%`) | overlap exists but limited |
| P6-B | KEEP analysis | SQ + extra PMCs | busy-cycles shift, VALU/LDS flat | no large hidden overlap |
| P7 | REJECT | `s_setprio 2` | profile tiny up, benchmark down | non-robust |
| P8 | REJECT config / KEEP infra | stage4 candidate + stage-index fix | correctness fixed, no speedup | keep fix, drop stage4 cfg |
| P9 | KEEP baseline refinement | prepack output kept fp8 + wrapper dtype inference | correctness pass, `N=8192 ~43.77k` | cleaner API, no perf regression |
| P10 | REJECT | stage2 split-phase software pipeline (winner path, A/B global->VGPR prefetch + wait + commit) | first version broke swizzle correctness; non-swizzle-restricted retry passed correctness but benchmark regressed (`N=8192 ~42.04k`) | overlap attempt cost exceeded gain on gfx1151; reverted |
| P11 | REJECT | stage2 interleaved refill (`compute stage0 -> refill0 -> compute stage1 -> refill1`) | correctness failed | unsafe stage reuse hazard in 2-stage path; reverted |
| P12 | REJECT | stage2 B-only split-phase prefetch (`prefetch B -> compute -> wait -> load A + commit B`) | correctness failed | hazard-prone on current schedule; reverted |
| P13 | REJECT | stage2 B-only split-phase retry with `volatile uint4` prefetch buffer | compile failed | invalid implementation direction; reverted |
| P14 | KEEP baseline simplification | remove swizzle toggle and overlap profiling modes; enforce stages=2 only | correctness pass; baseline maintained | matches gfx1151 winner path and scope |
| PCS-1/2/3 | KEEP analysis | PC sampling (SQ_IND, Host-Trap, Stochastic) | Consistent 45% LDS, 23% Sync stall distribution | Uncovered `SQ_WAVE_INST` drift issue, confirmed accurate stall breakdown |
| TT-1 | KEEP analysis | Thread Tracing (ATT) timeline | Found ~3,500 cycle gap between chunks | Proved kernel is global-memory bound on `vmcnt`, plus instruction fetch bottleneck on `v_perm_b32` |
| P15 | REJECT | hoist `v_perm_b32` literals to VGPRs | `N=8192` regressed ~42.62k | did not fix VOP3 issue latency (VGPR read ports), compiler barrier worsened global memory scheduling (`vmcnt` stalls up 18%) |

## Do-Not-Repeat (Unless New Preconditions)

- P1 direct-write decode change (no new condition).
- Any `__builtin_amdgcn_cvt_pk_f16_fp8` path on gfx1151.
- Removing load-phase `s_setprio` or tweaking to `s_setprio 2`.
- Stage2 split-phase, interleaved, or B-only overlapping schedules without restructuring the loop entirely.
- Optimizing inner-loop LDS access patterns expecting massive gains (TT shows WMMA/LDS already overlap perfectly; inter-loop is the problem).
- Stages > 2 (e.g. `stages=3` or `stages=4`). Platform prior definitively proves the occupancy loss hurts more than the latency hiding helps on gfx1151.

## Next Experiments (Thread-Tracing-Informed)

Based on the definitive Thread Tracing evidence, the previous hypothesis of "LDS bank conflicts" is invalid, and the true culprits are global memory starvation and `v_perm_b32` instruction bloat.

### P16: Chunk Size Expansion (K-Dimension)

- **Rationale:** Since stages > 2 is non-viable on gfx1151 due to occupancy loss, we cannot pipeline deeper. To reduce the impact of the ~3,500 cycle `vmcnt` global memory starvation penalty that occurs *between* chunks, we must decrease the *frequency* of these chunk boundaries.
- **Method:** Increase the `unroll_k` parameter (which dictates the K-chunk size per iteration). If `unroll_k` is increased from 2 to 4 or 8, the inner loop will execute 2x or 4x more WMMA math before hitting the global memory fetch boundary, amortizing the 3,500 cycle stall over a much larger block of compute.
- **Decision:** Requires increasing LDS allocation per stage. Monitor LDS capacity limits and occupancy. Keep if N=8192 GFLOPS improves.
