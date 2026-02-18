# HIP Prepacked-B fp8e5m2 Optimization Plan (gfx1151)

## Scope and KPI

- Target kernel path: prepacked-B HIP (`b_dtype=1`, fp8e5m2) on gfx1151.
- Prepack contract:
  - keep B physically as fp8 values in prepack output,
  - kernel consumes prepacked storage as raw bytes (`uint8_t*`) in device code.
- Python wrapper contract:
  - infer `b_dtype` from fp8 prepacked tensor dtype by default,
  - keep explicit `b_dtype` only for backward-compatible `uint8` prepacked input.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Performance KPI: `benchmark_scaled_mm_hip_prepacked_e5m2.py` GFLOPS (prepack excluded from timed region).
- Keep rule:
  - correctness passes,
  - forced `N=8192` does not regress,
  - large-N trend (`4096`, `8192`) does not regress materially.

## Current Baseline Snapshot (2026-02-18)

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

## Durable Findings (Keep in Mind)

- e5m2 conversion arithmetic is not the dominant limiter now; local scheduling/overlap is the main remaining lever.
- Epilogue (scale/bias) is not the bottleneck (`~0.13%` delta in ablation profile).
- Overlap decomposition at forced winner config shows only modest headroom:
  - benchmark: `full 24.999 ms` vs `no_overlap 25.651 ms` (`~2.6%`)
  - profile: `full 25.695 us` vs `no_overlap 26.230 us` (`~2.1%`).
- PMCs indicate no large hidden compute/comm reservoir:
  - no-overlap penalty mainly increases `SQ_BUSY_CYCLES`; VALU/LDS counts are effectively flat.
- Known non-starters on gfx1151 for current path:
  - `__builtin_amdgcn_cvt_pk_f16_fp8` (`gfx1250-insts` required),
  - stage4 candidate `(1,8,2,4,8,2)` (correct after fix but no speedup).
- Runtime overlap/no-overlap profiling toggles are now removed from the kernel path.
- Platform prior (from `doc/HIP_OPTIMIZATION_PLAN.md` + `kernel/hip/hip_kernel.cu`):
  - stage>2 historically hurts on gfx1151 via occupancy loss,
  - prior stage2 split-phase attempts with extra sync overhead regressed.

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, gate with:
   - `ps -eo pid,cmd | rg -n "benchmark_scaled_mm_hip_prepacked_e5m2.py|profile_scaled_mm_hip_prepacked_e5m2.py|rocprofv3" -S`
2. Per-step order:
   - `python test_scaled_mm_hip_prepacked_e5m2.py`
   - `python benchmark_scaled_mm_hip_prepacked_e5m2.py`
   - `rocprofv3 --kernel-trace --stats -d ... -o ... -- python -u profile_scaled_mm_hip_prepacked_e5m2.py`
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless locked by unrecoverable error or the user explicitly interrupted.

## Reference Note (When in Doubt)

- You may consult ISA/compiler references before changing low-level decode/packing logic:
  - `~/amd-llvm-project`
  - `doc/rdna35_instruction_set_architecture.md`
  - `doc/amdgpu_isa_rdna3_5.xml`

## Condensed Experiment Ledger

| ID | Keep? | Change | Key Result | Why |
|---|---|---|---|---|
| P0 | KEEP baseline | initial e5m2 baseline | `N=8192 ~42.86k` | starting point |
| P1 | REJECT | direct-write decode staging removal | `~+0.15%` only | noise-level |
| P2 | REJECT | `cvt_pk_f16_fp8` builtin path | compile fail | requires gfx1250 insts |
| P3 | REJECT | add tile `(1,8,2,2,16,1)` | tiny `~+0.24%`, unstable small-N | not robust |
| P4 | REJECT | remove load-phase `s_setprio` | large-N regression | KPI down |
| P5 | KEEP analysis | no-scale/no-bias ablation | `~+0.13%` delta | epilogue not bottleneck |
| P6 | KEEP analysis infra | `HIP_PREPACKED_OVERLAP_MODE` runtime modes | decomposition enabled | needed for overlap study |
| P6-A | KEEP analysis | overlap decomposition | low headroom (`~2-3%`) | overlap exists but limited |
| P6-B | KEEP analysis | SQ + extra PMCs | busy-cycles shift, VALU/LDS flat | no large hidden overlap |
| P7 | REJECT | `s_setprio 2` | profile tiny up, benchmark down | non-robust |
| P8 | REJECT config / KEEP infra | stage4 candidate + stage-index fix | correctness fixed, no speedup | keep fix, drop stage4 cfg |
| P9 | KEEP baseline refinement | prepack output kept fp8 + wrapper dtype inference | correctness pass, `N=8192 ~43.77k` | cleaner API, no perf regression |
| P10 | REJECT | stage2 split-phase software pipeline (winner path, A/B global->VGPR prefetch + wait + commit) | first version broke swizzle correctness; non-swizzle-restricted retry passed correctness but benchmark regressed (`N=8192 ~42.04k`) and autotune drifted to `(2,2,2,2,4,4)` | overlap attempt cost exceeded gain on gfx1151; reverted, skipped profile per protocol |
| P11 | REJECT | stage2 interleaved refill (`compute stage0 -> refill0 -> compute stage1 -> refill1`) | correctness failed even in non-swizzle (`rel_l2` up to `~0.068`, max error `~30.6`) | unsafe stage reuse hazard in 2-stage path; reverted, skipped benchmark/profile per protocol |
| P12 | REJECT | stage2 B-only split-phase prefetch (`prefetch B -> compute -> wait -> load A + commit B`) | correctness failed in non-swizzle (`rel_l2` up to `~0.034`, max error `~39.2`) | stage2 pipelining around same-stage refill remains hazard-prone on current schedule; reverted, skipped benchmark/profile per protocol |
| P13 | REJECT | stage2 B-only split-phase retry with `volatile uint4` prefetch buffer | compile failed (`volatile uint4` assignment/copy not supported by HIP vector type operators) | invalid implementation direction; reverted, skipped benchmark/profile per protocol |
| P14 | KEEP baseline simplification | remove swizzle toggle and overlap profiling modes; enforce stages=2 only | correctness pass; benchmark/profile remain in prior noise band (`N=8192 ~42.68k`, kernel `~25.65 us`) | matches gfx1151 winner path and current optimization scope |

## Do-Not-Repeat (Unless New Preconditions)

- P1 direct-write decode change (no new condition).
- Any `__builtin_amdgcn_cvt_pk_f16_fp8` path on gfx1151.
- Tile `(1,8,2,2,16,1)` re-test without new condition.
- Removing load-phase `s_setprio`.
- `s_setprio 2` load-phase tweak.
- Re-adding stage4 candidate `(1,8,2,4,8,2)` without new condition.
- Re-testing the P10 split-phase stage2 implementation without a new precondition.
- Re-testing P11 interleaved stage refill schedule without a new precondition.
- Re-testing P12 B-only stage2 split-phase schedule without a new precondition.
- Re-trying P13 volatile-`uint4` variant (compiler-incompatible).
- Re-introducing overlap-mode toggles or swizzle runtime toggle without a new requirement.

## Next Experiments (Stage2-Focused)

### P11: Stage2 software pipeline (no stage count increase)

- Constraint: keep `kStages=2` focus (known that `kStages>2` does not help on gfx1151).
- P10/P11/P12/P13 stage2 software-pipeline variants are all rejected (performance, correctness, or compile viability).
- Any further stage2 pipeline attempt must include an explicit hazard-proof stage ownership argument before code changes.

### P12: Baseline Hold

- Stage2 software-pipeline attempts currently failed (one performance, two correctness, one compile failure). Keep current baseline endpoint.
- Next pivot (only if requested): algorithmic/ISA-level changes beyond local reordering.
