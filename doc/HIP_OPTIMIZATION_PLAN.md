# HIP GEMM Kernel Optimization Plan

## Target

Kernel: `scaled_mm_kernel` — fp8×fp16 mixed-precision GEMM on RDNA 3.5 (gfx1151).
Best config: `(2,4,2,2,4,4)` → BlockM=128, BlockN=256, 256 threads, VGPR=189, SGPR=105, LDS=25088.
Peak: 40 CUs × 2 SIMD/CU × 32 ALUs/SIMD × 2 (VOPD/WMMA) × 2 (fp16 pack) × 2 (FMA) × 2.9 GHz = **59.4 TFLOPS**.
No native fp8 conversion instructions on gfx1151.

Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0`.
Approximation policy: denorm/NaN exact behavior may be relaxed if gate stays green.

## Performance History

| Change | N=8192 GFLOPS | % of peak | Delta |
|---|---|---|---|
| Step24: removed coarse preload barrier | 28780 | 48.5% | — |
| Step33: split preload order A then B | 30336 | 51.1% | +5.4% |
| Step42: compile-time contig fastpath | 30846 | 51.9% | +1.7% |
| Step65: selective s_setprio | 32404 | 54.6% | +5.0% |
| StepB12: A phys/inv mapping + WSGR A-store | 34404 | 57.9% | +6.2% |
| StepC04: C-shuffle physical row mapping | 34747 | 58.5% | +1.0% |
| + packed fp8→fp16 conversion (fp8x4_to_half2x2) | 35485 | 59.7% | +2.1% |
| + register-tiled compute | **36134** | **60.8%** | +1.8% |

Beats `torch_compiled` (31494) at N=8192 by +14.7%.
Note: benchmarks affected by thermal throttling; gains <2% should be interpreted with caution.
We've updated the benchmark scripts to use longer repetition time for more steady results. The above benchmark results are now invalid and we need to benchmark thee baseline again.

## Current Bottleneck Analysis

### PMC profiling (per-SIMD averages, N=8192, baseline = 36134 GFLOPS)

| Counter | Value | % of total |
|---|---|---|
| SQ_WAVE32_INSTS | 97,787,855 | 100% |
| SQ_INSTS_VALU | 34,939,501 | 35.7% |
| SQ_INSTS_LDS | 35,940,394 | 36.8% |
| SQ_INSTS_SALU | 4,734,686 | 4.8% |
| SQ_INSTS_FLAT | 1,039,678 | 1.1% |
| Other (waitcnt, barrier, branch) | ~21,133,596 | 21.6% |

### Issue model

- LDS cannot dual-issue (VOPD is VALU-only, ISA §7.6). Each LDS op = 1 full issue cycle.
- VALU can dual-issue via VOPD → effective cost ~half.
- Effective issue-cycle breakdown: **LDS ~48%**, VALU ~23%, Other ~29%.
- **LDS is the dominant bottleneck.**

### ASM-verified main loop breakdown (config 2,4,2,2,4,4, kContigFastPath=true)

Precise instruction counts from ASM (`.LBB7_8` through `.LBB7_16`):

| Category | Instruction | Count | Notes |
|---|---|---|---|
| **B LDS reads** | `ds_load_u16_d16` | 64 | Even K values |
| | `ds_load_u16_d16_hi` | 64 | Odd K values |
| **A LDS reads** | `ds_load_b128` | 16 | 2 per rm tile × 4 tiles × 2 stages |
| **LDS writes** | `ds_store_b128` | 12 | A commit (8) + B commit (4) |
| **WMMA** | `v_wmma_f32_16x16x16_f16` | 32 | 16 per stage × 2 stages |
| **Conversion VALU** | `v_lshlrev_b32` | 32 | fp8→fp16 |
| | `v_perm_b32` | 16 | fp8→fp16 |
| | `v_and_or_b32` | 16 | fp8→fp16 |
| | `v_and_b32` | 16 | fp8→fp16 |
| | `v_add_nc_u32` | 16 | fp8→fp16 |
| | `v_mov_b16` | 9 | fp8→fp16 |
| | `v_lshrrev_b32` | 8 | fp8→fp16 |
| **Address VALU** | mul/mad/add_co/etc | ~14 | A+B global addr |
| **Global loads** | `global_load_b128` | 10 | A prefetch (8) + B prefetch (2) |
| **Waits** | `s_waitcnt` | 82 | 64 d16 WAW + 18 other |

**Key findings:**
- B LDS reads (128 ops) = **66% of all LDS ops** (192 total)
- Conversion VALU (~121 ops) fills LDS stall slots — effectively free
- Address calc VALU is only ~14 ops in the hot loop (negligible)
- The ~305 addr calc from whole-function static analysis is mostly prologue/epilogue
- All B/A read addresses use precomputed base registers (v175, v176) with compile-time offsets — no per-iteration address calc in compute phase

### B read d16 WAW hazard pattern

```asm
ds_load_u16_d16     v128, v175           // B[0][col] → v128.lo
s_waitcnt lgkmcnt(0)                      // WAIT (WAW on v128)
ds_load_u16_d16_hi  v128, v175 offset:528 // B[1][col] → v128.hi
ds_load_u16_d16     v129, v175 offset:1056
s_waitcnt lgkmcnt(0)                      // WAIT (WAW on v129)
... (repeats 8× per tile, 4 tiles per stage, 2 stages = 64 waits)
```

Compiler hides these waits by interleaving conversion VALU. E.8 proved eliminating the waits doesn't help (insight #7).

## Hardware Constraints

| Constraint | Value | Impact |
|---|---|---|
| LDS per CU | 64 KB | Occupancy=2 WG requires ≤32 KB/WG |
| kStages=2 LDS | 25 KB | Occupancy=2 ✓ |
| kStages=4 LDS | 50 KB | Occupancy=1 → -20% (rejected) |
| VGPR per SIMD | 1536 | 189 VGPR → 8 waves/SIMD |
| kStages >= kUnrollK | Enforced by static_assert | kUnrollK=4 requires kStages≥4 |

## Critical Insights

### 1. Conversion VALU fills LDS stall slots — cannot be reduced

The compiler interleaves fp8→fp16 conversion ops between d16 WAW waits. Reducing conversion VALU removes useful latency-hiding work. This is why ALL conversion optimization attempts failed (Direction A: 6 variants, all negative).

### 2. d16 WAW waits are not the real bottleneck

E.8 proved that eliminating the 64 `s_waitcnt lgkmcnt(0)` per iteration (via ASM `ds_load_u16` into separate VGPRs + pack) doesn't help. The batched approach regressed -2.3% because rigid ASM blocks prevent compiler interleaving. The compiler already hides wait latency effectively.

### 3. K-pair interleaved layout shifts bottleneck from LDS to VALU

E.2 profile: LDS dropped 69% (35.9M → 11.2M), VALU increased 28% (34.9M → 44.7M). Any approach eliminating B LDS reads requires per-wave conversion → +28%+ VALU.

### 4. Address calc is negligible in the hot loop

ASM analysis shows only ~14 address calc VALU per main loop iteration. The compiler precomputes base addresses (v175 for B, v176 for A) and uses compile-time offsets. The ~305 addr calc from whole-function static analysis is mostly prologue/epilogue/C-shuffle. **Direction F is not viable.**

### 5. V_PK_* instructions are NOT VOPD-eligible

VOP3P instructions (`V_PK_LSHLREV_B16`, `V_PK_ADD_U16`, etc.) cannot dual-issue via VOPD. pk_u16 conversion is strictly worse in this kernel context.

### 6. ds_permute_b32 cannot cross wave boundaries

Operates within wave32 only. Cannot distribute data between the 8 waves in our workgroup.

### 7. Inline ASM ds_load_u16 works with uint32_t LDS offset cast

`static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared_ptr))` extracts the LDS offset. Verified correct on gfx1151. Useful for future ASM work.

### 8. Tile shape changes cannot improve LDS throughput alone

Direction H (kRepeatM=8,kRepeatN=2) proved that reducing total LDS instruction count by 43% yields zero speedup. The reason: in the (4,4) config, B reads' WAW waits are filled by conversion VALU (effectively free work). Replacing B reads with A reads trades "B read + free VALU filling stalls" for "A reads + idle stall slots". The LDS+VALU+WMMA interleaving is a tightly coupled system; reducing any one component doesn't help because it removes the latency-hiding work for the others. **Any approach that merely reshuffles LDS reads between A and B will be neutral or negative.**

## Rejected Experiments (Do Not Repeat)

### Direction A: Reduce conversion VALU — REJECTED
Kernel is LDS-bound; reducing VALU removes latency-hiding work (insight #1).

### Direction C: kUnrollK=4 — REJECTED
Requires kStages≥4 → 50KB LDS → occupancy=1 → -20%.

### Direction E: Reduce LDS instruction count — REJECTED (all approaches)

| Experiment | Result | Root cause |
|---|---|---|
| E.1: K-pair interleaved B layout | -5.1% | 2× global loads + interleave VALU |
| E.2: K-pair interleaved + ds_read_b32 | -2.6% | Shifted LDS→VALU bound |
| E.2+pk_u16 | -6.1% | VOP3P can't dual-issue |
| E.6: Separate-VGPR B loads (ASM) | Failed | LDS address space mismatch |
| E.6: Separate-VGPR B loads (volatile) | Failed | Compiler generates wrong code |
| E.8: ASM ds_load_u16 batched (2×8) | -2.3% | Pack VALU + lost interleaving |
| E.8: ASM ds_load_u16 separate stmts | Failed | Compiler reorders past s_waitcnt |

### Direction F: Reduce address calc VALU — REJECTED
ASM analysis shows only ~14 addr calc VALU in hot loop. Not a bottleneck (insight #4).

### Direction I: Reduce s_waitcnt overhead — REJECTED
E.8 proved waits are already hidden by compiler interleaving (insight #2).

### Other rejected experiments

| Experiment | Result | Reason |
|---|---|---|
| B write swizzle + kBPad=2 | -5% | Alignment regression |
| Naive kUnrollK=1 overlap | -4% to -8% | 2× sync, no real overlap |
| Split-phase prefetch, kStages=2, 2× sync | -7% | Barrier overhead |
| Split-phase prefetch, kStages=4, 1× sync | -20% | Occupancy halved |
| WSGRB owner-wave variants (B13/B14) | -5% to -15% | Load issue bottleneck |
| B physical/inverse row mapping (B15) | -3% | Overhead > gain |
| rn-outer loop reorder | -4% | Worse locality |
| rm-outer with A hoist | Neutral | No benefit |
| K0×N×K1 for B (any K1) | Dead end | Write cost increase dominates |
| K-major global load for B | Dead end | Destroys coalescing |
| ds_load_2addr_b32 for B reads | Dead end | 4-byte alignment fails for odd lanes |
| uint32 B read (avoid d16 packing) | -3.2% | VGPR pressure + pack VALU > benefit |
| packed-u8 pair conversion (B17) | -3% | Benchmark KPI down |
| asm-u8 conversion (B18) | -16% | VALU/scheduling cost |
| C compile-time has_scale/has_bias (C03) | -1% | No benefit |
| C bias half2 pairing (C05) | -1% | No benefit |
| C kCPad=0 (C06) | -1% | Bank conflicts |
| C kCPad=16 (C07) | -2% | Wasted LDS |
| WSGRC (C08) | Infeasible | Accumulators are wave-private |
| s_waitcnt_depctr tweak (Step108) | Neutral | No benefit |
| One-buffer interleave/refill (Step80/87/94) | Negative | No benefit |
| E.7: ds_permute cross-wave broadcast | Infeasible | ds_permute only works within wave32 |
| E.4: Skip LDS for B (global→register) | Likely negative | Per-wave conversion +28% VALU |
| Direction H: (2,4,2,2,8,2) kRepeatM=8,kRepeatN=2 | Neutral (-0.2%) | -43% LDS, -40% VALU, but +4.3% busy cycles. Lost B-read/conversion-VALU interleaving synergy: conversion was free (filled WAW stalls), extra A reads not free (2 WMMAs can't hide A-read latency). See insight #8. |

## Next Steps

### Fresh baseline (2026-02-13)

Profiled at N=8192, 20 iters. Benchmark: **35,057 GFLOPS** (59.0% of peak).

| Counter | Value | % of total |
|---|---|---|
| SQ_WAVE32_INSTS | 85,144,463 | 100% |
| SQ_INSTS_VALU | 31,126,701 | 36.6% |
| SQ_INSTS_LDS | 30,870,137 | 36.2% |
| SQ_INSTS_SALU | 2,824,395 | 3.3% |
| SQ_INSTS_FLAT | 887,399 | 1.0% |
| Other (waitcnt, barrier, branch) | ~19,436,000 | 22.8% |
| Bank conflict rate | 6.45% | (3.07M / 47.6M) |

Effective issue-cycle breakdown: **LDS ~48%**, VALU ~24%, Other ~28%.
VALU has ~50% headroom (dual-issues via VOPD).

### Direction H: Tile config `(2,4,2,2,8,2)` — REJECTED

Tested: -43% LDS, -40% VALU, same occupancy (184 VGPR, 8 waves/SIMD), but **neutral performance** (-0.2%). Lost interleaving synergy (insight #8). Tile shape changes alone cannot help.

### Direction J: Interleave B reads with WMMA compute — MEDIUM PRIORITY

Currently all B fragments are pre-loaded before any WMMA. Interleaving would:
- Load one B tile → compute with it → load next → compute
- Allow LDS B reads to overlap with WMMA execution on the matrix unit
- Key difference from Direction H: doesn't reduce total LDS or VALU, but overlaps them with WMMA on the independent matrix unit
- May break the current tight LDS↔VALU interleaving (risk)

### B store XOR swizzle — MEDIUM PRIORITY

Bank conflict rate is 6.45% (3.07M / 47.6M). B writes cause 50% conflicts. Row-dependent XOR on B store column addresses can reduce this.

**Proposed swizzle**: `col_swizzled = col ^ ((row & 7) * 4)` on B store, with matching inverse on B reads.

### Open question: what can actually improve beyond 60.8%?

Insight #8 shows the kernel's LDS/VALU/WMMA scheduling is tightly coupled — reducing one component removes latency-hiding for the others. Possible remaining levers:
1. **True overlap** (not reduction): overlap LDS with WMMA on independent execution units (Direction J)
2. **Reduce bank conflicts**: 6.45% → ~0% saves wasted LDS cycles without changing instruction mix
3. **Reduce "Other" 22.8%**: waitcnt/barrier/branch overhead — 19.4M instructions, potentially reducible
4. **Better compiler scheduling**: ASM-level tuning of instruction ordering

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, gate with:
   - `ps -eo pid,cmd | rg -n "benchmark_scaled_mm_hip.py|profile_scaled_mm_hip.py|rocprofv3" -S`
2. Per-step order:
   - `python test_scaled_mm_hip.py`
   - `python benchmark_scaled_mm_hip.py`
   - `rocprofv3 --kernel-trace --stats -d ... -o ... -- python -u profile_scaled_mm_hip.py -N 8192 --iters 20`
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless locked by unrecoverable error or the user explicitly interrupted.

## Profiling Quick Reference

rocprofv3 on gfx1151: max ~6 PMC counters per run (more causes crash). Use two runs:
- Run 1: `SQ_BUSY_CYCLES SQ_INSTS_LDS SQ_INSTS_TEX_LOAD SQ_INSTS_VALU SQ_WAVES SQ_WAVE_CYCLES`
- Run 2: `SQ_INSTS_FLAT SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_TEX_STORE SQ_INSTS_WAVE32_VALU SQ_WAVE32_INSTS`

```bash
rocprofv3 --pmc <COUNTER1> <COUNTER2> ... \
  -- python -u profile_scaled_mm_hip.py -N 8192 --iters 20

# Results in x2/<pid>_results.db
sqlite3 x2/<pid>_results.db "
SELECT ipm.name, AVG(p.value) as avg_val
FROM rocpd_pmc_event_<UUID> p
JOIN rocpd_info_pmc_<UUID> ipm ON p.pmc_id = ipm.id
GROUP BY ipm.name ORDER BY avg_val DESC;"
```

## ASM Inspection

Compile single config with `-save-temps` (edit `_CONFIGS` and dispatch table to 1 config first):
```bash
hipcc -save-temps [flags] -c kernel/hip/hip_kernel.hip -o /tmp/out.o
# GPU ASM: hip_kernel-hip-amdgcn-amd-amdhsa-gfx1151.s
# Hot kernel: kContigFastPath=true variant (Lb0ELb1E in mangled name)
# Config 2,4,2,2,4,4: Li2ELi4ELi2ELi2ELi4ELi4ELb0ELb1E (lines ~36919-38617)
```

Can disable autotune with `HIP_FORCE_CONFIG=2,4,2,2,4,4` env var.

## ISA Notes

Authoritative references: `doc/rdna35_instruction_set_architecture.md` (ISA manual) and `doc/amdgpu_isa_rdna3_5.xml` (instruction XML). Grep, don't read — these files are large. When in doubt, verify claims against these sources.

- VOPD X-opcodes: FMAC, FMAAK, FMAMK, MUL_F32, ADD_F32, SUB_F32, SUBREV_F32, MUL_DX9_ZERO_F32, MOV_B32, CNDMASK_B32, MAX_F32, MIN_F32, DOT2ACC_F32_F16, DOT2ACC_F32_BF16
- VOPD Y-opcodes: same as X plus ADD_NC_U32, LSHLREV_B32, AND_B32
- V_PK_* (VOP3P, NOT VOPD-eligible): LSHLREV_B16, ADD_U16, MUL_F16, ADD_F16, FMA_F16, FMAC_F16
- `DS_LOAD_2ADDR_B32`: loads 2×32-bit from `ADDR+OFF0*4` and `ADDR+OFF1*4` (offsets 0-255, ×4)
- `ds_load_u16` works via inline ASM with `static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared_ptr))`
- No native fp8 conversion instructions

## Autotune Config Notes

- `(2,4,2,2,2,4)` invalid: fails `kShASize >= kCShuffleSize` static assert
- Square/wide-N: best `(2,4,2,2,4,4)`
- Tall-narrow-N: `(2,4,2,2,4,2)` or `(4,2,2,2,2,4)` can be stronger
- kCPad=8 is best among {0, 8, 16}

## File Reference

- `kernel/hip/hip_kernel.cu` — Main kernel (**baseline, 36134 GFLOPS**)
- `kernel/hip/hip_kernel.py` — Python wrapper, JIT, autotune, `HIP_FORCE_CONFIG`
- `test_scaled_mm_hip.py` — Correctness test (all configs × 5 sizes)
- `benchmark_scaled_mm_hip.py` — Full benchmark (N=128..8192)
- `profile_scaled_mm_hip.py` — Profiling script for rocprofv3
- `profile_out/` — Profiling output directory
- `doc/rdna35_instruction_set_architecture.md` — ISA reference (large, grep don't read)
