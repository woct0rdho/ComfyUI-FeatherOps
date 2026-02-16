# Triton vs HIP Plan (gfx1151)

## Goal
- Compare Triton and HIP fairly on the same tile shape and launch width.
- Understand why Triton codegen is slower on gfx1151.
- Define a concrete plan to improve Triton-generated code quality.

## Fixed-Config Baseline (Autotune Off)
Comparison mapping used:
- HIP config: `2,4,2,2,4,4`
  - `BLOCK_M=128`, `BLOCK_N=256`, `BLOCK_K=32`, workgroup size `256`.
- Triton forced config: `128,256,32,1,8,2`
  - `BLOCK_M=128`, `BLOCK_N=256`, `BLOCK_K=32`, `num_warps=8`, `num_stages=2`.

Knobs used:
- HIP: `HIP_FORCE_CONFIG=2,4,2,2,4,4`
- Triton: `TRITON_SCALED_MM_FORCE_CONFIG=128,256,32,1,8,2`, `AUTOTUNE_DISABLE=1`

## Repro Artifacts
- Benchmark log: `triton_hip_compare_fixed_step01/benchmark_fixed.log`
- Triton dump: `triton_hip_compare_fixed_step01/triton_dump/*/_scaled_mm_kernel.{ttir,ttgir,llir,amdgcn,hsaco}`
- Triton profile DB: `triton_hip_compare_fixed_step01/rocprof_triton_fixed/triton_fixed_results.db`
- HIP profile DB: `triton_hip_compare_fixed_step01/rocprof_hip_fixed/hip_fixed_results.db`

## Measured Results (N=8192)
Unprofiled benchmark (median):
- Triton fixed: `47.820 ms`, `22992.886 GFLOPS`
- HIP fixed: `31.527 ms`, `34875.003 GFLOPS`

Profiled kernel metrics:
- Triton (`_scaled_mm_kernel`):
  - median `48.422 ms`, avg `48.638 ms`
  - `VGPR=224`, `SGPR=128`, `LDS=16384`, `WG=256`
- HIP (`scaled_mm_kernel<2,4,2,2,4,4,...>`):
  - median `32.052 ms`, avg `31.982 ms`
  - `VGPR=192`, `SGPR=128`, `LDS=25088`, `WG=256`

Key point:
- Triton uses less LDS but is much slower. Current bottleneck is not LDS capacity; it is schedule/instruction quality and register pressure.

## Codegen Findings (Triton vs HIP)
### 1) Triton keeps modulo indexing in hot path
From TTGIR:
- `arith.remsi` exists for `offs_am` and `offs_bn` (`% M`, `% N` style wraparound).
- This adds integer ALU overhead and prevents a clean interior fast path.

HIP path:
- Uses specialized interior launch (`kCheckBounds=false`, contiguous fast path), no modulo in hot path.

### 2) Triton B path converts fp8->fp16 in the inner MMA loop
From TTGIR:
- Sequence appears after `ttg.local_load` of B each iteration:
  - `bitcast -> extui -> and/shl/add/or/select -> bitcast`
- This means conversion work stays on the critical MMA path.

HIP path:
- Converts fp8->fp16 during global->LDS stage; local reads for MMA are already fp16.

### 3) Triton B shared layout is not optimized for vectorized dot operand loads
From TTGIR:
- A shared: `#ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>`
- B shared: `#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>`
- B path looks like a generic layout and likely causes extra register shuffle cost.

From AMDGCN disassembly (static counts):
- `v_perm_b32`: `128`
- `v_wmma_f32_16x16x16_f16`: `64`
- `s_waitcnt`: `108`
- `s_barrier`: `5`

Key point:
- Very high permute pressure is consistent with expensive layout/conversion handling on B.

### 4) Triton register pressure is significantly higher
- Triton VGPR `224` vs HIP VGPR `192` with same block sizes/workgroup.
- This reduces scheduling flexibility and likely hurts occupancy/sustained throughput.

### 5) HIP has structural optimizations not represented in current Triton kernel
- WSGR A-store ownership.
- A physical/inverse LDS row mapping.
- Explicit no-bounds contiguous interior path.
- Hand-controlled load/compute cadence.

## Triton Compiler Knob Checks (gfx1151)
Tested quickly with the same forced config:
- `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`:
  - Compiles, but severe regression: `180.352 ms`, `6096.469 GFLOPS`.
  - Not usable as a blanket knob for this kernel on gfx1151.
- `TRITON_HIP_USE_BLOCK_PINGPONG=1`:
  - Compile pipeline failure at `TritonAMDGPUBlockPingpong`.
  - Log: `triton_hip_compare_fixed_step01/benchmark_triton_block_pingpong.log`
- `TRITON_HIP_USE_ASYNC_COPY=1`:
  - Compile pipeline failure at `ConvertTritonAMDGPUToLLVM`.
  - Log: `triton_hip_compare_fixed_step01/benchmark_triton_async_copy.log`

Conclusion:
- Backend-wide knobs are not drop-in wins on gfx1151 for this kernel shape.
- Improvements should focus first on kernel-source structure and then selective compiler options.

## How To Compare Triton Generated Code vs HIP (Repeatable Method)
1. Fix both kernels to the same tile config (no autotune).
2. Benchmark without profiler overhead for primary KPI.
3. Profile both with rocprof for kernel time + resources (VGPR/SGPR/LDS/WG).
4. Dump Triton TTGIR/LLIR/AMDGPU and inspect:
   - shared layout attributes (`vec/perPhase/maxPhase/order`)
   - whether fp8->fp16 conversion is on critical path
   - wait/barrier density and permute density
5. Compare against HIP source schedule and profiling counters.

## Parity Gaps To Close
1. Hot-path index overhead:
   Triton still carries `%M/%N` (`arith.remsi`) in the main path; HIP interior path does not.
2. B operand critical-path overhead:
   Triton performs fp8->fp16 unpack/convert around dot-operand loads; HIP pre-converts during global->LDS.
3. Dot operand packing/permutation overhead:
   Triton B shared layout is effectively generic (`vec=1`) and drives high `v_perm_b32`.
4. Register pressure:
   Triton VGPR is higher (224 vs 192), reducing scheduling headroom.
5. Missing schedule-level specialization on gfx1151:
   Useful backend transforms are either disabled by default or not stable for this kernel/arch.

## Easy-First Parity Plan (Kernel + Compiler)
### P0: Measurement Guardrails (do once, keep fixed)
- [ ] Keep fixed-config comparison while closing gaps:
  - HIP: `HIP_FORCE_CONFIG=2,4,2,2,4,4`
  - Triton: `TRITON_SCALED_MM_FORCE_CONFIG=128,256,32,1,8,2`, `AUTOTUNE_DISABLE=1`
- [ ] For every step: benchmark (median GFLOPS), then profile, then inspect TTGIR/AMDGPU deltas.

### P1: Triton Python Kernel Easy Wins (no compiler changes)
- [ ] Split launch into interior and edge kernels in Python wrapper.
- [ ] Remove `%M/%N` wraparound from interior kernel address math.
- [ ] Keep masked edge kernel for correctness only on boundary tiles.
- [ ] Add stronger shape/stride hints (`tl.multiple_of`, `tl.max_contiguous`) for A/B/C pointers.
- [ ] Re-check TTGIR to confirm interior kernel has no `arith.remsi` in hot path.

Expected win:
- Less integer ALU/control overhead and better code motion freedom; should reduce wait pressure slightly.

### P2: Triton Python Kernel B-Path Restructure (highest impact, medium effort)
- [ ] Move B fp8 unpack/convert earlier so MMA loop consumes fp16-ready shared fragments.
- [ ] Restructure B load path to encourage vectorized packed loads/unpack in fewer ops.
- [ ] Keep data treated as integer bytes (gfx1151 has no native fp8) and avoid semantic fp8 ops.
- [ ] Validate TTGIR/AMDGPU: reduced unpack-per-iteration pressure and reduced `v_perm_b32`.

Expected win:
- Shorter MMA critical path and lower permute density; target VGPR reduction from 224 toward HIP.

### P3: Triton Compiler C++ Changes (targeted, not global knobs)
- [ ] In `~/triton/third_party/amd/backend/compiler.py`, add gfx1151-targeted option plumbing for this kernel only (not blanket env forcing).
- [ ] In AMD pass pipeline, add a guarded path to improve B shared encoding/operand packing for WMMA fp16-dot from byte source.
- [ ] Extend/adjust `OptimizeDotOperands`-related logic for gfx11 byte-source->fp16 flow to reduce late permutes.
- [ ] Keep new pass behavior shape- and arch-guarded (`gfx1151`, matching layout) to avoid regressions elsewhere.

Expected win:
- Structural reduction in permutes and live ranges beyond what Python-source rewrites can force.

### P4: Schedule-Level Compiler Work (hard, after P1-P3)
- [ ] Revisit ping-pong/in-thread-transpose only after kernel structure is cleaned up.
- [ ] If needed, patch pass legality for gfx1151 shape/layout used here, then re-evaluate.
- [ ] Accept only if compile is stable and benchmark KPI improves.

Expected win:
- Additional overlap/scheduling gains, but only after operand path quality is fixed.

### P5: Parity Acceptance
- [ ] Primary KPI: unprofiled median GFLOPS at `N=8192` (same fixed config).
- [ ] Secondary: profiled median kernel time, VGPR, `v_perm_b32`/waitcnt density.
- [ ] Keep only changes that improve KPI and pass accuracy gate.
- [ ] After fixed-config parity work, re-enable autotune and verify gains persist.
