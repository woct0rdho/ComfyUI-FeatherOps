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

## PC Sampling Profiles (config 1,8,2,2,8,2, N=8192, 200 iters)

All runs used `rocprofv3 --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 5000`.
See `doc/GFX1151_REFERENCE.md` for PC sampling setup details.

### PCS-1: SQ_IND Approach (Feb 2026)

Two independent runs (ROCm driver, mainline kernel) produced ~572K samples
each with identical distributions (all categories within +/-0.1pp), confirming
reproducibility. This approach reads `SQ_WAVE_INST_DW0/1` via SQ indirect
debug registers, capturing ALL active waves (~480) per scan.

| Category | Key Opcodes | Samples | % |
|---|---|---:|---:|
| LDS Read | `ds_load_b128` | 200,843 | 35.1 |
| WMMA | `v_wmma_f32_16x16x16_f16` | 100,504 | 17.6 |
| FP Convert | `v_perm_b32`, `v_pk_*`, `v_cvt_*` | 76,879 | 13.4 |
| LDS Write | `ds_store_b128`, `ds_store_b16` | 69,748 | 12.2 |
| Sync | `s_barrier`, `s_waitcnt`, `buffer_gl0_inv` | 64,016 | 11.2 |
| SALU | `s_add_i32`, `s_ashr_i32`, `s_or_b32`, etc. | 30,771 | 5.4 |
| VALU Other | `v_add_co_*`, `v_lshlrev_b64`, etc. | 23,417 | 4.1 |
| Global Load | `global_load_b128` | 5,912 | 1.0 |
| Global Store | `global_store_b128` | 295 | 0.1 |

### PCS-2: Host-Trap Approach (Mar 2026)

Uses the mainline kernel with CWSR daisy-chain trap handler (SQ_CMD
BROADCAST+CHECK_VMID). The trap handler captures `ttmp0:1` (the wave's
actual PC at trap delivery). Traps one wave per targeted SIMD/slot per
SQ_CMD, yielding ~9.2 samples per trap across 40 CUs. Total: 10,469
samples from 1,136 traps, 67 dispatches sampled.

| Category | Key Opcodes | Samples | % |
|---|---|---:|---:|
| LDS Read | `ds_load_b128` | 4,641 | 44.3 |
| Sync | `s_barrier`, `s_waitcnt`, `buffer_gl0_inv` | 2,542 | 24.3 |
| FP Convert | `v_perm_b32`, `v_pk_*`, `v_cvt_*` | 1,163 | 11.1 |
| WMMA | `v_wmma_f32_16x16x16_f16` | 867 | 8.3 |
| SALU | `s_mul_i32`, `s_cmp_*`, `s_add_i32`, etc. | 817 | 7.8 |
| VALU Other | `v_lshlrev_b64`, `v_add_co_*`, etc. | 322 | 3.1 |
| Global Load | `global_load_b128` | 98 | 0.9 |
| LDS Write | `ds_store_b128`, `ds_store_b16` | 19 | 0.2 |

### PCS-1 vs PCS-2 Comparison

The kernel binary is unchanged between the two runs (confirmed by git log;
only comment and variable-rename changes since PCS-1).

| Category | PCS-1 (SQ_IND) | PCS-2 (host-trap) | Delta |
|---|---:|---:|---:|
| LDS Read | 35.1% | 44.3% | +9.2 |
| WMMA | 17.6% | 8.3% | -9.3 |
| FP Convert | 13.4% | 11.1% | -2.3 |
| LDS Write | 12.2% | 0.2% | -12.0 |
| Sync | 11.2% | 24.3% | +13.1 |
| SALU | 5.4% | 7.8% | +2.4 |
| VALU Other | 4.1% | 3.1% | -1.0 |
| Global Load | 1.0% | 0.9% | -0.1 |

**Root cause of discrepancy -- SQ_WAVE_INST vs ttmp0:1 semantics:**

The SQ_IND approach reads `SQ_WAVE_INST_DW0/1`, the instruction word in
the wave's instruction buffer. When a wave is stalled at a barrier or
waitcnt, the instruction buffer can retain the previously-decoded
instruction (e.g., the `ds_store_b128` that preceded the barrier). The
host-trap approach captures `ttmp0:1`, the wave's actual program counter
at trap delivery, which correctly points to the stalling instruction.

Smoking gun: `ds_store_b128 v137, v[142:145]` (identical encoding in both
binaries) has **54,014 samples (9.4%)** in PCS-1 vs **1 sample** in PCS-2.
Those 54K "store" samples were actually waves stalled at the subsequent
`s_barrier`, but `SQ_WAVE_INST` still held the preceding store instruction.
PCS-2 correctly attributes the stalls to `s_barrier` and `s_waitcnt`,
explaining the +13pp increase in Sync.

Similarly, WMMA drops from 17.6% to 8.3%: WMMA occupies the instruction
buffer for 32 cycles, but once completed the wave may wait at a subsequent
`s_waitcnt`. SQ_WAVE_INST shows WMMA; ttmp0:1 shows the waitcnt.

**What is consistent across both methods:**

- Memory operations (LDS + global) dominate: 48.4% (PCS-1) vs 45.4% (PCS-2)
- WMMA and LDS co-occur in 96-99% of time bins (double-buffering works)
- Global memory is negligible (~1%)
- The kernel is LDS-bandwidth-bound, not compute-bound

**Interpretation:**

PCS-1 (SQ_IND) shows *what instructions were recently in-flight* (biased
toward long-latency issued instructions). PCS-2 (host-trap) shows *where
waves actually stall* (biased toward stalling instructions). Both are
valid views; PCS-2 is the more accurate measure of wave stall distribution.

### WMMA / Load Overlap

Temporal analysis using 5ms bins:

| Metric | PCS-1 | PCS-2 |
|---|---:|---:|
| Bins with both WMMA and LDS | 98.8% | 95.8% |
| Avg WMMA % per bin | 17.5% | 8.3% |
| Avg LDS % per bin | 47.0% | 44.6% |
| Avg Sync % per bin | 11.1% | 23.9% |

Both profiles confirm double-buffering (stages=2) keeps WMMA and LDS
pipelines fed throughout the main loop.

### Key Takeaways

1. **LDS-bandwidth-bound**: ~45-48% memory vs ~19-31% compute (depending on
   method). LDS accounts for >97% of all memory samples in both profiles.
2. **Sync is larger than SQ_IND suggested**: PCS-2 shows barriers + waitcnts
   at 24.3% (vs 11.2% in PCS-1). This is the true stall distribution --
   waves spend nearly a quarter of their time waiting.
3. **FP8->FP16 conversion is significant**: 11-13% in `v_perm_b32` and
   related ops. Unavoidable on gfx1151 which lacks native fp8 WMMA.
4. **Good overlap confirmed by both methods**: WMMA and LDS co-occur in
   96-99% of time bins.
5. **Sample throughput trade-off**: SQ_IND gives ~55x more samples but reads
   stale instruction buffer contents. Host-trap gives fewer samples but
   captures the true PC.

### PC Sampling vs Overlap Measurement Cross-Validation

PC sampling confirms the overlap decomposition results (P6-A, P6-B):

- P6-A found only ~2-3% overlap headroom. PC sampling explains: 96-99% of
  time bins already contain both WMMA and LDS simultaneously --
  double-buffering is working, there is little more overlap to extract.
- P6-B showed VALU/LDS instruction counts flat between full and no_overlap,
  with penalty only in `SQ_BUSY_CYCLES`. PC sampling explains: waves spend
  the bulk of time stalled on memory (mostly LDS reads and associated
  waitcnts). Disabling overlap just makes waves wait longer on the same
  LDS operations.
- "e5m2 conversion is not the dominant limiter" is confirmed: FP Convert is
  11-13% in both profiles, behind LDS Read and Sync.
- Global memory is not a concern: ~1% of samples. The 2-stage pipeline hides
  global latency almost perfectly.

### Data Files

- `pc_sampling_profile_mainline/rocprof_out/db_results.db` -- 572K samples (PCS-1, SQ_IND, native rocpd DB)
- `pc_sampling_profile/rocprof_out/x2/` -- 572K samples CSV (PCS-1, ROCm kernel run)
- `pc_sampling_profile/kernel_disasm_full.txt` -- full disassembly (11,739 lines)
- `pc_sampling_hosttrap_2026_03_05/rocprof_out/pcs_results.db` -- 10K samples (PCS-2, host-trap, native rocpd DB)

## Durable Findings (Keep in Mind)

- e5m2 conversion arithmetic is not the dominant limiter now; local scheduling/overlap is the main remaining lever.
- Epilogue (scale/bias) is not the bottleneck (`~0.13%` delta in ablation profile).
- Overlap decomposition at forced winner config shows only modest headroom:
  - benchmark: `full 24.999 ms` vs `no_overlap 25.651 ms` (`~2.6%`)
  - profile: `full 25.695 us` vs `no_overlap 26.230 us` (`~2.1%`).
- PMCs indicate no large hidden compute/comm reservoir:
  - no-overlap penalty mainly increases `SQ_BUSY_CYCLES`; VALU/LDS counts are effectively flat.
- PC sampling methodology matters:
  - SQ_IND (reading `SQ_WAVE_INST`) shows stale instruction buffer contents; inflates ds_store and WMMA.
  - Host-trap (capturing `ttmp0:1`) shows the true PC at trap delivery; more accurate stall distribution.
  - True sync cost is ~24% (PCS-2), not ~11% (PCS-1). Barriers and waitcnts dominate stalls.
  - Both methods agree: kernel is LDS-bandwidth-bound with effective double-buffering.
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
| PCS-1 | KEEP analysis | PC sampling via SQ_IND (host_trap, 5ms, 200 iters N=8192) | 572K samples: LDS 48.4%, compute 31.0%, sync 11.2%, other 9.5% | kernel is LDS-bandwidth-bound; but SQ_WAVE_INST reads stale instruction buffer |
| PCS-2 | KEEP analysis | PC sampling via host-trap ttmp0:1 (mainline CWSR, 5ms, 200 iters N=8192) | 10K samples: LDS 44.5%, sync 24.3%, compute 19.4%, other 10.9%; same kernel binary as PCS-1 | true stall profile: sync is 24% not 11%; ds_store inflation in PCS-1 was SQ_WAVE_INST artifact |

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

## Next Experiments (PC-Sampling-Informed)

PCS-2 (host-trap) identifies three cost centers: LDS read (44.3%),
sync (24.3%), compute (19.4%). Sync is larger than PCS-1 suggested
(24% vs 11%). Stage2 software-pipeline attempts (P10-P13) are exhausted.
The following experiments target the cost centers directly.

### P15: LDS Bank Conflict Analysis

- Rationale: LDS reads are 44.3% of stall time. If bank conflicts inflate
  this, fixing access patterns could reduce stalls without changing tile shape.
- Method: `rocprofv3 --pmc LDSBankConflict` on the forced winner config.
- Decision: if bank conflict rate is high (>10%), investigate swizzled LDS
  layout for A/B tiles. If low, LDS bandwidth is the hard limit -- skip.

### P16: 4-Warp Tile Exploration

- Rationale: PCS-2 shows barriers + waitcnts at 24.3%. Halving warp count
  (4 warps, 128x128 tile) would roughly halve barrier cost and reduce LDS
  contention.
- Risk: prior platform knowledge says fewer warps hurts occupancy on gfx1151.
  Only attempt if P15 shows LDS contention (not bandwidth) is the limiter.
- Method: add config `(1,4,2,2,8,2)` or `(2,2,2,2,8,2)`, run correctness +
  benchmark. Keep only if N=8192 GFLOPS improves.

### P17: Baseline Hold (Default)

- If P15 shows low bank conflicts and P16 regresses, the kernel is at the
  gfx1151 hardware limit for this tile shape and dtype.
- The 11-13% FP8->FP16 conversion cost disappears on gfx1250 (native fp8 WMMA).
  No further gfx1151-specific optimization is warranted unless a new
  algorithmic approach emerges.

### PCS-3: True Stochastic PC Sampling (Mar 2026)

Uses the hardware-driven stochastic timer (`--pc-sampling-method stochastic`, `cycles`, `1048576` interval). The kernel programs the hardware timer, which autonomously counts and fires a `PERF_SNAPSHOT` trap. Captured 827,514 samples.

| Category | Samples | % | Delta vs PCS-2 (Host-Trap) |
|---|---:|---:|---:|
| LDS Read | 373,684 | 45.2 | +0.9 |
| Sync | 190,328 | 23.0 | -1.3 |
| FP Convert | 84,690 | 10.2 | -0.9 |
| WMMA | 76,954 | 9.3 | +1.0 |
| SALU | 64,877 | 7.8 |  0.0 |
| VALU Other | 24,936 | 3.0 | -0.1 |
| Global Load | 10,013 | 1.2 | +0.3 |
| LDS Write | 1,925 | 0.2 |  0.0 |
| Global Store | 107 | 0.0 |  0.0 |

**Temporal Overlap (5ms Bins):**
* Bins with both WMMA and LDS: **99.7%** (1027/1030)
* Avg WMMA % per bin: **9.3%**
* Avg LDS Read % per bin: **45.0%**
* Avg Sync % per bin: **23.0%**

**Conclusions:**
1. **Host-trap (PCS-2) is heavily validated:** The distribution profiles between host-trap and stochastic are virtually identical (all major categories are within ~1.3 percentage points). This confirms that capturing `ttmp0:1` in host-trap accurately reflects true hardware stall states, solving the `SQ_WAVE_INST` stale-instruction problem seen in PCS-1.
2. **Bandwidth Limited:** The kernel remains heavily LDS-bandwidth-bound (LDS Read = ~45%).
3. **Perfect Overlap:** Double buffering works flawlessly; WMMA and LDS execute concurrently in 99.7% of all 5ms time bins.
4. **Sample Volume:** Stochastic mode easily delivered 800K+ samples without the CPU-overhead or skid associated with software-issued host traps.
