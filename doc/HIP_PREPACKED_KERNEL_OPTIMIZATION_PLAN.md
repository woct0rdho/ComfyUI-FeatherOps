# HIP Prepacked-B Kernel Optimization Plan (gfx1151)

## Current State (2026-02-16)

- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
- Best kept prepacked path:
  - B prepack format is **fp8 bytes** (`uint8`), layout `[K/16, N, 16]`.
  - Kernel default config family includes `1,8,2,2,8,2` and currently picks it for large square shapes.
  - Default swizzle for prepacked path is **off**.
- Latest default benchmark at `N=8192`:
  - `torch_compiled`: 32,279 GFLOPS
  - `hip`: 33,956 GFLOPS
  - `hip_prepacked`: 36,698 GFLOPS
- Large-N forced sweep (kernel-only, prepack excluded): prepacked `1,8,2,2,8,2` wins vs baseline hip `2,4,2,2,4,4` at 4096/6144/8192.

## Performance Target Policy

- Performance target is **benchmark GFLOPS** from `benchmark_scaled_mm_hip_prepacked.py` (prepack excluded).
- Profiled kernel duration (`rocprofv3`) is for attribution only (instruction mix, bottleneck direction), not the final target metric.

## Stable Checkpoints

- `cfe1781` - fp8-byte prepack path kept as baseline.
- `c8ae4df` - added winning `1,8,2,2,8,2` kernel config.
- `b798f83` - defaulted prepacked path to `1,8` + no-swizzle flow in wrapper/scripts.

## Key Findings (Keep These Reasonings)

1. **Never prepack B to fp16 for this path**
   - fp16 prepack doubled B traffic and caused major regression.
   - fp8-byte transport is the core benefit of prepacked path.

2. **Low LDS bank conflict alone is not sufficient**
   - One rejected path had near-zero bank conflict but was still slower.
   - Bottleneck can shift to VMEM/VALU/control overhead.

3. **Tile shape can reduce duplicated B conversion work**
   - `1,8,2,2,8,2` reduced VALU pressure vs `2,4,2,2,4,4` prepacked path and won clearly.

4. **`kStages=4` overlap attempt was not viable here**
   - Regressed hard; likely occupancy/resource tradeoff dominated.

5. **Swizzle is config-dependent**
   - For the current `1,8` winner, no-swizzle is slightly faster and simpler.

6. **Thermal drift is real; always co-measure baseline hip**
   - Baseline hip can move by a few percent across runs.

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, gate with:
   - `ps -eo pid,cmd | rg -n "benchmark_scaled_mm_hip_prepacked.py|profile_scaled_mm_hip_prepacked.py|rocprofv3" -S`
2. Per-step order:
   - `python test_scaled_mm_hip_prepacked.py`
   - `python benchmark_scaled_mm_hip_prepacked.py`
   - `rocprofv3 --kernel-trace --stats -d ... -o ... -- python -u profile_scaled_mm_hip_prepacked.py -N 8192 --iters 20`
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless locked by unrecoverable error or the user explicitly interrupted.

## Condensed Experiment Log

| ID | Change | Keep? | N=8192 hip_prepacked | Key Evidence | Why / Decision |
|---|---|---|---:|---|---|
| P0 | Initial prepacked (`[K/16,N,16]` fp16) | REJECT | 24,396 | bank conflict 29.62, kernel ~45.2 us | Strong regression; bad LDS pattern + fp16 traffic |
| P1 | Alternate LDS mapping (`[K/16,16,N]` fp16 path) | REJECT | 24,141 | bank conflict ~0.23 but still slow | Conflicts fixed but fp16 traffic still dominant |
| P2a | Add selective `s_setprio` | KEEP | 25,526 | kernel ~44.8 us | Small but real gain |
| P3 | Switch prepack to fp8 bytes (`uint8`) | KEEP | 32,398 | kernel ~34.2 us | Big recovery; confirmed fp8-byte transport is critical |
| P4a | Try stages=4 overlap (`2,4,2,4,4,4`) | REJECT | 25,639 | large regression | Occupancy/resource tradeoff unfavorable |
| P4b | Add tile `1,8,2,2,8,2` | KEEP | 37,188 (forced), 36,992 recheck | kernel ~30 us, VGPR 176, low bank conflict | Reduced duplicate B-decode work; major win |
| P5 | Default prepacked selection includes `1,8` | KEEP | 37,007 (default run) | profile shows `1,8` dispatch | Default path now picks winning config |
| P6 | Swizzle on/off study for `1,8` | KEEP (no-swizzle) | no-swizzle mean 37,288 vs swizzle mean 36,916 | bank conflict unchanged | Swizzle not helping this winner |
| P7 | Large-N validation (4096/6144/8192) | KEEP | 33,483 / 36,609 / 36,763 | all > baseline hip | Win is not 8192-only |
| P8 | Bottleneck isolation pass (3 kernels @8192) | KEEP | 37,283 (`1x8`) / 33,185 (`2x4`) | `1x8` vs `2x4`: VALU 0.55x, busy cycles 0.85x, similar L2 hit | Winner is not memory-limited; VALU/decode pressure is primary remaining limiter |
| P9a | Candidate tile `1,16,2,2,16,1` | REJECT | 32,159 | large regression vs `1x8` | Too slow; reverted |
| P9b | Compile-time swizzle specialization (`kUseSwizzle` template bool) | KEEP | 37,341 (`1x8`, forced) | correctness unchanged; benchmark slight gain | Small but positive; kept |
| P9c | Re-try stage4 overlap on `1,8,2,4,8,2` | REJECT | 29,659 | strong regression | Reverted immediately; skipped profile after revert per protocol |

## Thermal Tracking (Baseline Hip Proxy)

Forced baseline config: `HIP_FORCE_CONFIG=2,4,2,2,4,4`.

| Point | Baseline hip GFLOPS (N=8192) |
|---|---:|
| P0 | 34,124 |
| P1 | 34,250 |
| P2a | 34,375 |
| P3 | 34,371 |
| P4a | 34,403 |
| P4b | 34,402 |
| P5 | 34,276 |
| Recheck | 35,376 |
| P6 | 35,052 |
| P8 | 34,411 |
| P9a | 34,078 |
| P9b | 34,894 |
| P9c | 35,162 |

Interpretation:
- Baseline drift can reach a few percent.
- Compare candidate and baseline in the same run window whenever possible.

## Do Not Repeat Without New Preconditions

- fp16 prepack transport path (bandwidth regression).
- stages=4 overlap path for this prepacked kernel family (both prior and re-try).
- assuming low bank conflict automatically means higher GFLOPS.

## P8 Isolation Summary (Completed)

Controlled kernels at `N=8192`:
- baseline hip `2,4,2,2,4,4`
- prepacked winner `1,8,2,2,8,2`
- prepacked reference `2,4,2,2,4,4`

Benchmark GFLOPS:
- hip `2x4`: **34,411**
- prepacked `1x8`: **37,283**
- prepacked `2x4`: **33,185**

Key counters (target kernels, normalized to prepacked `1x8`):
- prepacked `2x4` vs prepacked `1x8`:
  - `SQ_INSTS_VALU`: **1.81x**
  - `SQ_BUSY_CYCLES`: **1.18x**
  - `SQ_WAVE_CYCLES`: **1.18x**
  - `SQ_INSTS_LDS`: **0.71x**
- baseline hip `2x4` vs prepacked `1x8`:
  - `SQ_INSTS_LDS`: **3.70x**
  - `SQ_INSTS_VALU`: **0.56x**
  - `LDSBankConflict`: **22.8x** higher

Inference:
- Current winner (`1x8`) is primarily constrained by **VALU/decode work**, not by LDS conflicts or L2 behavior.
- Reducing VALU conversion pressure is the highest-value direction.

## New Plan: Isolate Bottleneck and Optimize Further

### Objective

Push prepacked `1,8,2,2,8,2` further by reducing VALU/decode pressure while preserving fp8-byte transport.

### Targets

- Primary: keep prepacked > baseline hip at `N={4096,6144,8192}`.
- Stretch: raise forced `1x8` at `N=8192` from ~37.3k toward 38k+ GFLOPS.

### P10 - Branch B Execution (VALU/Decode)

1. **Conversion placement A/B test**
   - Variant A (current): convert fp8->fp16 in compute stage.
   - Variant B: convert during load/commit phase and store fp16 fragments in LDS for compute.
   - Compare benchmark GFLOPS first; profile only if benchmark improves.

2. **Conversion micro-op reduction**
   - Optimize `fp8x4_to_half2x2` instruction sequence (avoid non-VOPD-friendly choices).
   - Keep numerical behavior within existing accuracy gate.

3. **Decode scheduling tweaks**
   - Reorder decode and WMMA issue to improve overlap on independent units without `kStages=4`.
   - Keep LDS footprint and occupancy unchanged.

### P11 - Config-Robustness Sweep

1. Add targeted script (`benchmark_scaled_mm_hip_configs.py`) for fixed-config comparisons.
2. Benchmark prepacked candidates on square and rectangular large shapes.
3. Keep default autotune set minimal: only configs that never regress badly.

### Exit Criteria

- Correctness remains green (`rel_l2 <= 0.01`, `max abs <= 1.0`).
- Benchmark (not profile time) improves for forced `1x8` at `N=8192`.
- No regression of default prepacked path on large-N sweep.
