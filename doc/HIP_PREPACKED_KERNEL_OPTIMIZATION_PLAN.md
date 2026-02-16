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

## Thermal Tracking (Baseline Hip Proxy)

Forced baseline config: `HIP_K0MK1_FORCE_CONFIG=2,4,2,2,4,4`.

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

Interpretation:
- Baseline drift can reach a few percent.
- Compare candidate and baseline in the same run window whenever possible.

## Do Not Repeat Without New Preconditions

- fp16 prepack transport path (bandwidth regression).
- stages=4 overlap path for this kernel shape/config family.
- assuming low bank conflict automatically means higher GFLOPS.

## New Plan: Isolate Bottleneck and Optimize Further

### Objective

Identify the dominant limiter of current winner `1,8,2,2,8,2` and improve further while keeping the current gains stable.

### Target

- Primary: sustain prepacked > baseline hip across `N={4096,6144,8192}`.
- Stretch: improve N=8192 from ~36.7-37.2k to >38k GFLOPS without accuracy regression.

### P8 - Bottleneck Isolation Pass (must-run first)

Run three controlled kernels at `N=8192`:
- Baseline hip forced `2,4,2,2,4,4`
- Prepacked forced `1,8,2,2,8,2` (winner)
- Prepacked forced `2,4,2,2,4,4` (reference prepacked shape)

For each, collect:
- Benchmark GFLOPS and top kernel us
- PMCs set A: `SQ_BUSY_CYCLES SQ_INSTS_LDS SQ_INSTS_TEX_LOAD SQ_INSTS_VALU SQ_WAVES SQ_WAVE_CYCLES`
- PMCs set B: `SQ_INSTS_FLAT SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_TEX_STORE SQ_INSTS_WAVE32_VALU SQ_WAVE32_INSTS`
- Single-counter runs: `LDSBankConflict`, `L2CacheHit`, `VALUInsts`

Decision criteria:
- **Memory-side bottleneck** if busy cycles track TEX/L2 pressure while VALU is not dominant.
- **VALU/decode bottleneck** if VALUInsts or wave32 VALU share dominates and correlates with cycle growth.
- **Control/wait bottleneck** if SALU/FLAT/other share rises while LDS/VALU are not dominant.

Deliverable:
- One compact table with per-kernel normalized ratios vs winner (`1x8`).

### P9 - Targeted Optimization Branches (choose by P8 result)

#### Branch A: Memory-side dominant
1. Test prepack ordering variants that preserve fp8 bytes but improve L2 locality.
2. Test B-load issue pattern: one vs two columns/thread prefetch grouping (same arithmetic, different VMEM issue pattern).
3. Keep only variants that improve both kernel us and GFLOPS in same thermal window.

#### Branch B: VALU/decode dominant
1. Optimize `fp8x4_to_half2x2` instruction mix for VOPD-friendliness (guided by ISA/ASM).
2. Reorder decode/WMMA schedule to increase decode-hide opportunities without increasing LDS footprint.
3. Verify with VALUInsts + busy cycles + correctness.

#### Branch C: Control/wait dominant
1. Tighten waitcnt/priority window placement and measure effect.
2. Prefer minimal schedule surgery; avoid stages=4 occupancy penalties unless new evidence appears.

### P10 - Robustness and Selection Policy

1. Add lightweight prepacked-only autotune across `_PREPACKED_CONFIGS` with cache by shape/stride.
2. Validate selected config on mixed shapes (square and tall/wide) and ensure no 8192 regression.
3. Keep forced-config override behavior for deterministic profiling.

### Exit Criteria for This Plan Revision

- Correctness stays green (`rel_l2 <= 0.01`, `max abs <= 1.0`).
- Prepacked remains above baseline hip at 4096/6144/8192.
- Clear bottleneck attribution from P8 with one selected optimization branch and measurable gain.
