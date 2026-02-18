# HIP Prepacked-B Kernel Optimization Plan (gfx1151)

## Scope and KPI

- Target kernel family: prepacked B (`uint8` fp8 bytes), gfx1151, winner track around `1,8,2,2,8,2`.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Primary KPI: benchmark GFLOPS (kernel runtime); profile kernel-time is diagnostic only.
- Keep rule:
  - correctness passes,
  - forced `N=8192` does not regress,
  - large-N sweep (`4096/6144/8192`) does not regress materially.

## Current Stable Baseline (2026-02-17)

- Data path kept: prepacked B as fp8 bytes in `[K/16, N, 16]`.
- Current production winner: `1,8,2,2,8,2` (no-swizzle).
- Representative full benchmark (`benchmark_scaled_mm_hip_prepacked.py`):
  - `N=8192`: `hip_prepacked ~40.5k`, `hip ~36.2k`, `torch_compiled ~32.4k` GFLOPS.
- Forced thermal-anchor comparison (same process window):
  - baseline hip `2,4,2,2,4,4`: ~`35.5k` GFLOPS,
  - prepacked winner `1,8,2,2,8,2`: ~`40.3k` GFLOPS.
- Thermal note: gains under ~2% are treated as noise unless repeated.

## Current Bottleneck Model (Important)

From target-kernel-only profiling (`P11-B6a`, forced `1,8,2,2,8,2`, `N=8192`):

- Avg kernel time: `27.755 ms` (~`39.61 TFLOPS`).
- Theoretical peak (59.4 TFLOPS) lower-bound time: `18.510 ms`.
- Gap to peak-time bound: `+9.245 ms` (~`+49.9%`).
- Bandwidth sanity:
  - bytes floor (A fp16 + B fp8 + C fp16): `335.5 MB`,
  - bandwidth floor @ 200 GB/s: `1.678 ms`,
  - arithmetic intensity very high; this path is not DRAM-bandwidth-limited.
- Instruction mix (target kernel):
  - VALU ~58.2%, LDS ~13.6%, OTHER ~28.3% (`SQ_WAVE32_INSTS` basis).

Interpretation:

- Dominant limiter is on-chip issue/scheduling pressure (decode/VALU + control/other), not DRAM.
- LDS is non-trivial but no longer the primary limiter for the winner path.

## Durable Findings

1. fp8-byte transport is mandatory; fp16-prepack paths were large regressions.
2. `1,8,2,2,8,2` no-swizzle is the strongest robust winner at large N.
3. Stage4 overlap direction is non-viable on this hardware/kernel family.
4. Epilogue is a minor contributor (no-scale/no-bias gave only ~`+0.35%`).
5. Conversion approximation slack is small (aggressive approximations fail gate).
6. Removing `v_perm` alone does not guarantee speedup; replacement ops can cancel gains.
7. Forced-config cache pitfall is real (`_get_forced_config` cache must be cleared in same-process A/B scripts).
8. Benchmark wins are the decision metric; profile-time and counters are supporting evidence.

## Experiment Ledger (Condensed, keep/reject reasons)

| ID | Keep? | Change | Result | Reason |
|---|---|---|---|---|
| P0/P1 | REJECT | fp16-prepack transport variants | ~24k class | bandwidth/traffic regression |
| P3 | KEEP | fp8-byte prepack path | major recovery | restored transport efficiency |
| P4a | REJECT | `kStages=4` overlap | strong regression | occupancy/resource loss |
| P4b | KEEP | add `1,8,2,2,8,2` | major gain | better decode/work balance |
| P6 | KEEP | no-swizzle for `1,8` | faster | swizzle not helping winner |
| P8 | KEEP | bottleneck isolation (`hip 2x4`, prepacked `1x8`, prepacked `2x4`) | `1x8` wins | winner not memory-limited |
| P9a | REJECT | tile `1,16,2,2,16,1` | regression | poorer efficiency |
| P9b | KEEP | compile-time swizzle specialization | small gain | safe positive change |
| P9c | REJECT | stage4 overlap retry | strong regression | repeated non-viable direction |
| P10b | KEEP | signfold conversion + perm pair-build | ~40k class | best robust conversion path |
| P10c | REJECT | exp-only conversion approx | fails all configs | too inaccurate |
| P10d | REJECT | spread-multiply pair-build | fails all configs | not bit-exact (carry coupling) |
| P11-A | KEEP | epilogue ablation + pmc + asm counts | epilogue ~0.35% | bottleneck remains decode/issue |
| P11-B1 | REJECT | mask/shift pair-build replacement | `N=8192` ~38.6k | clear perf regression |
| P11-B2a/B2b/B2c | REJECT | low-risk decode/WMMA scheduling tweaks | no robust gain | benchmark flat/down, profile worse |
| P11-B3a | REJECT | compute-only `s_setprio` | strong regression | priority scope wrong |
| P11-B3b/B3c | REJECT | asymmetric A/B-only priority | correctness fail | unstable correctness |
| P11-B4 | REJECT | direct `uint4` conversion pack | correctness fail | swizzle-path breakage |
| P11-B5 | REJECT | no-swizzle-only direct pack | no robust gain | profile worse |
| P11-B6a | KEEP (analysis) | roofline/issue model | quantified peak gap | confirmed compute-side bottleneck |
| P11-B6b | REJECT | prepack lane transform + mask/shift decode | no robust gain | `v_perm` removed but replaced by similar-cost ops |
| P11-B6c | REJECT | load-phase fp8->fp16 conversion | large regression | moved decode to non-overlapped path + larger LDS payload |
| P11-B6d | REJECT | remove load-phase `s_setprio` | no robust gain | slightly worse forced KPI |
| P11-B6e | REJECT | lazy decode at `rm==0` | no robust gain | profile worse |
| P11-B6f | REJECT | add config `1,8,1,2,8,2` | NaN correctness | exposed `kStages > kUnrollK` stage-write issue |
| P11-B6g | REJECT | fix stage-write issue then re-test `1,8,1,2,8,2` | ~38.7k forced | finer chunking loses to control overhead |

## Do-Not-Repeat (Unless New Preconditions)

- fp16-prepack transport variants.
- stage4 overlap variants in this kernel family.
- exp-only conversion approximation.
- non-bit-exact pair-build shortcuts.
- pair-build swap to plain mask/shift replacing current `v_perm` path.
- low-risk B2 scheduling variants already tested (B2a/B2b/B2c).
- asymmetric priority scopes (`B`-only or `A`-only).
- direct conversion-pack rewrites that break swizzle correctness.
- `1,8,1,2,8,2` as a performance path (validated slower after bugfix).
- re-running rejected items without a clear new precondition.

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, gate with:
   - `ps -eo pid,cmd | rg -n "benchmark_scaled_mm_hip_prepacked.py|profile_scaled_mm_hip_prepacked.py|rocprofv3" -S`
2. Per-step order:
   - `python test_scaled_mm_hip_prepacked.py`
   - `python benchmark_scaled_mm_hip_prepacked.py`
   - `rocprofv3 --kernel-trace --stats -d ... -o ... -- python -u profile_scaled_mm_hip_prepacked.py`
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless locked by unrecoverable error or the user explicitly interrupted.

## Next Plan (P12)

Objective: quantify remaining overlap ceiling before any new invasive code change.

### P12-A Overlap Decomposition (Target Kernel)

- Build controlled run modes for winner config (`1,8,2,2,8,2`):
  - `full`
  - `no_overlap`
  - `comm_only`
  - `comp_only`
- Use `N=8192`, forced config, and collect:
  - benchmark GFLOPS,
  - profile kernel time,
  - target-kernel PMCs (`SQ_*`, `LDSBankConflict`, `L2CacheHit`, `VALUInsts`).
- Derived metrics:
  - direct overlap gain = `T_no_overlap - T_full`
  - decomposition gain = `T_comm_only + T_comp_only - T_full`

### P12-B Decision Gate

- If measured overlap headroom suggests >2% realistic upside, proceed to one high-impact schedule change.
- If overlap headroom is small, stop schedule micro-tuning and pivot to new algorithmic direction (not another local reorder).

### P12-C Candidate (Only if Gate Passes)

- Single candidate only: decode/compute placement change predicted by P12-A data.
- Maintain constraints: fp8-byte prepack, no stage4, same correctness gate.
