# Triton vs HIP Plan (gfx1151)

## Goal
- Compare Triton and HIP fairly on the same tile shape and launch width.
- Explain why Triton is slower on gfx1151 in this kernel.
- Close the highest-impact parity gaps with measurable wins.

## Reference Source Code and ISA
- `~/triton`
- `~/amd-llvm-project`
- `doc/rdna35_instruction_set_architecture.md`
- `doc/amdgpu_isa_rdna3_5.xml`

## Fixed Comparison Contract (do not drift)
- HIP fixed config: `HIP_FORCE_CONFIG=2,4,2,2,4,4`
  - `BLOCK_M=128`, `BLOCK_N=256`, `BLOCK_K=32`, `WG=256`
- Triton fixed config: `TRITON_SCALED_MM_FORCE_CONFIG=128,256,32,1,8,2`, `AUTOTUNE_DISABLE=1`
  - `BLOCK_M=128`, `BLOCK_N=256`, `BLOCK_K=32`, `num_warps=8`, `num_stages=2`

## Benchmark Scope (temporary)
- Per user direction, active iteration benchmarking is now `N=8192` only to save autotune time.
- Smaller matrix sweeps are deferred for later validation.

## Current Status (latest kept kernel)
- Active kernel policy:
  - interior-only path
  - constrained inputs (contiguous + divisible)
  - direct B conversion call in-loop (`fp8e4m3fn_to_fp16(b)`)
- Latest compiler-path status:
  - Step11 non-k B vec2 shared-layout change is currently kept in local Triton (`~/triton` `feather`).
  - Step12 vec4 follow-up was rejected (no stable KPI gain; backend output effectively unchanged vs Step11).
  - Step13 perPhase/maxPhase follow-up was rejected (no spill, but clear regression); local Triton is reverted to Step11 vec2.
  - Previous failed compiler-path attempts (Step08/Step09) remain reverted.
- Best current fixed-config KPI (`N=8192`):
  - Triton: `45.028 ms` (`24418.396 GFLOPS`) from repeat run
  - HIP: `30.932 ms` (`35545.538 GFLOPS`) from repeat run
  - Gap: Triton is ~`1.46x` slower in time

## Stable Findings (carry forward)
1. Baseline Triton had avoidable hot-path index overhead.
   - `%M/%N` (`arith.remsi`) appeared in hot path before interior specialization.
2. Dominant remaining bottleneck is B operand path quality.
   - Improving B shared layout vectorization (`vec=1 -> vec=2`) helps, but B conversion/reorder pressure remains the main optimization axis.
3. Permute pressure remains high and sticky.
   - Kept variants still show high static `v_perm_b32` (`128` to `188`).
4. gfx1151 is highly schedule-sensitive.
   - Source rewrites that look simpler can worsen `s_waitcnt` and regress heavily.
5. Source-level P2 rewrites are near saturation.
   - Best kept source delta (Step07) gave only a small KPI win with no structural ISA change.
6. Forced B shared-order override in compiler is currently a do-not-repeat path.
   - On gfx1151 it triggered a spill cliff (`VGPR=256`, scratch traffic, private segment allocation) and severe regression.
7. Even simplified B shared layout forcing (`vec=1`, `order=[0,1]`) still spills.
   - It reduces spill volume vs Step08 but remains far slower than Step07 and keeps `VGPR=256`.
8. Forcing Triton away from buffer ops is not a shortcut to HIP parity.
   - It makes opcodes look more HIP-like (`global_load/global_store`) but triggers severe spill and register-pressure regression.
9. Non-k B-path vectorization can help if order is preserved.
   - Moving B shared layout from `vec=1` to `vec=2` with preserved `order=[1,0]` improved fixed `N=8192` performance without spills.
10. Increasing non-k B vec width from `2` to `4` is not a reliable next lever for this kernel.
   - TTGIR changed (`vec=4`) but final AMDGCN binary hash matched Step11 (`vec=2`), and repeat KPI did not beat gate.
11. Swizzle phase tuning (`perPhase/maxPhase`) can regress without spills.
   - `vec=2, perPhase=2, maxPhase=8` raised wait pressure and VGPR usage and regressed runtime despite zero scratch/private segment.

## Compiler Knob Status (do not repeat without new precondition)
- `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`:
  - severe regression (`180.352 ms`, `6096.469 GFLOPS`)
  - log: `triton_hip_compare_fixed_step01/benchmark_triton_inthread_transpose.log`
- `TRITON_HIP_USE_BLOCK_PINGPONG=1`:
  - compile failure at `TritonAMDGPUBlockPingpong`
  - log: `triton_hip_compare_fixed_step01/benchmark_triton_block_pingpong.log`
- `TRITON_HIP_USE_ASYNC_COPY=1`:
  - compile failure at `ConvertTritonAMDGPUToLLVM`
  - log: `triton_hip_compare_fixed_step01/benchmark_triton_async_copy.log`
- `AMDGCN_USE_BUFFER_OPS=0`:
  - severe regression (`104.486 ms`, repeat `104.529 ms`), `VGPR=256`, scratch/private segment enabled
  - log/artifacts: `triton_hip_compare_fixed_step11/*`

Conclusion:
- Backend-wide env knobs are not drop-in wins on this arch/shape.
- Continue with guarded compiler-path work, not blanket toggles.

## Experiment Log (concise)
### Step01 (kept) - fixed baseline
- KPI: Triton `47.820 ms` vs HIP `31.527 ms`.
- Profiling: Triton `VGPR=224`, HIP `VGPR=192`.
- Why it matters: reference point and parity gap definition.
- Artifacts: `triton_hip_compare_fixed_step01/*`.

### Step02 (rejected, reverted) - strict fast-path constraints only
- Change: removed modulo/masked behavior aggressively.
- KPI: regressed to Triton `50.277 ms`.
- Key reason: `s_waitcnt` rose (`108 -> 148`) while B-path bottleneck stayed.
- Artifacts: `triton_hip_compare_fixed_step02/*`.

### Step03 (kept) - interior/edge split
- Change: interior no modulo; edge masked correctness path.
- KPI: improved to Triton `47.234 ms`.
- Why kept: measurable win with stable counters.
- Artifacts: `triton_hip_compare_fixed_step03/*`.

### Step04 (kept) - interior-only policy + constrained tests
- Change: removed edge path; enforced constrained inputs; pruned incompatible configs.
- KPI: near-neutral vs Step03 (`47.266 ms`), still better than Step01.
- Notes: early runs hit intermittent KFD/rocprof instability; later runs recovered.
- Artifacts: `triton_hip_compare_fixed_step05/*`.

### Step05 (rejected, reverted) - manual K-loop prefetch
- Change: software-pipelined `a_next/b_next` pattern.
- KPI: catastrophic regression to Triton `163.340 ms`.
- Key reason: `VGPR` hit `256`, `s_waitcnt`/barrier pressure jumped.
- Artifacts: `triton_hip_compare_fixed_step06/*`.

### Step06 (rejected, reverted) - remove interior leftovers
- Change: removed `pid>=num_pid` guard and K-tail branch.
- KPI: regressed to Triton `50.115 ms`.
- Key reason: worse scheduling (`s_waitcnt 108 -> 148`, `VGPR 210 -> 215`).
- Artifacts: `triton_hip_compare_fixed_step07/*`.

### Step07 (kept) - minimal B conversion rewrite
- Change: use direct `fp8e4m3fn_to_fp16(b)` call in loop; keep control structure.
- KPI: Triton `47.171 ms` (repeat `47.071 ms`).
- Profiling: Triton median ~`48.164 ms`; static counters unchanged from Step04.
- Why kept: small positive fixed-config gain, no counter regression.
- Artifacts: `triton_hip_compare_fixed_step08/*`.

### Step08 (rejected, reverted) - compiler-path B shared-order override (gfx1151)
- Change (in `~/triton`): in `LowerLoops.cpp`, for `gfx1151` + WMMA + dot operand B (`opIdx=1`) + fp8 byte-source (`bitWidth=8`), force shared order toward K-contiguous (`repOrder`) when current order is not K-contiguous.
- KPI (fixed `N=8192`): Triton `183.233 ms` (repeat `184.044 ms`) vs HIP `30.841/30.774 ms`.
- Profiling/codegen deltas vs Step07 baseline:
  - Triton profile median `48.164 -> 189.809 ms`.
  - `VGPR: 216 -> 256` (rocprof kernel metadata), `private_segment_fixed_size: 0 -> 596`.
  - Scratch instructions introduced (`scratch_store`/`scratch_load`), static `v_perm_b32` rose `128 -> 320`.
  - TTGIR B shared layout changed from `#shared1 vec=1 order=[1,0]` to `vec=16 order=[0,1]`.
- Reject reason: layout forcing caused register-pressure/spill collapse and ~`3.9x` Triton time regression.
- Action taken: reverted `LowerLoops.cpp` patch in `~/triton`, rebuilt wheel, reinstalled local Triton.
- Artifacts: `triton_hip_compare_fixed_step09/*`.

### Step09 (rejected, reverted) - compiler-path simple B shared layout (gfx1151)
- Change (in `~/triton`): in `LowerLoops.cpp` dot-operand path for `gfx1151` WMMA B fp8 (`opIdx=1`, `bitWidth=8`), bypass composed swizzle and force simple shared encoding with `vec=1, perPhase=1, maxPhase=1, order=repOrder`.
- KPI (fixed `N=8192`): Triton `93.505 ms` (repeat `93.640 ms`) vs HIP `30.675/30.836 ms`.
- Profiling/codegen deltas vs Step07 baseline:
  - Triton profile median `48.164 -> 94.143 ms`.
  - `VGPR: 216 -> 256`; scratch/private segment still present (`scratch_size=128`, private segment fixed size `128`).
  - Static `v_perm_b32` stayed high at `320` (vs Step07 `128`), scratch ops reduced vs Step08 but not eliminated.
  - TTGIR B shared layout became `#shared1 vec=1 order=[0,1]`.
- Reject reason: despite improving over Step08, it remains ~`2.0x` slower than Step07 and violates no-spill guardrails.
- Action taken: reverted `LowerLoops.cpp` patch in `~/triton`, rebuilt wheel, reinstalled local Triton.
- Artifacts: `triton_hip_compare_fixed_step10/*`.

### Step10 (rejected) - N8192-only env trial `AMDGCN_USE_BUFFER_OPS=0`
- Change: disable Triton AMD buffer ops to move generated memory ops closer to HIP-style globals.
- KPI (fixed `N=8192`): Triton `104.486 ms` (repeat `104.529 ms`) vs HIP `30.670/30.544 ms`.
- Profiling/codegen deltas vs Step07 baseline:
  - Triton profile median `48.164 -> 107.994 ms`.
  - `VGPR: 216 -> 256` (rocprof metadata), scratch/private segment enabled (`scratch_size=580`, private segment fixed size `580`).
  - Static ISA shifted to global memory ops (`buffer_load/store -> 0`, `global_load/global_store` increased), but `v_perm_b32` and waits worsened (`128 -> 188`, `108 -> 169`).
- Reject reason: opcode-style proximity to HIP did not translate to performance; register pressure and spill overhead dominate.
- Action taken: keep default `AMDGCN_USE_BUFFER_OPS=1` as required baseline behavior.
- Artifacts: `triton_hip_compare_fixed_step11/*`.

### Step11 (kept) - compiler-path non-k B vec2 shared layout (gfx1151)
- Change (in `~/triton` `LowerLoops.cpp`): for gfx1151 WMMA B fp8 byte-source (`opIdx=1`, `bitWidth=8`) in non-k-contig shared order path, force swizzled-shared encoding to `vec=2, perPhase=1, maxPhase=1`, preserving order (`[1,0]` for this kernel).
- KPI (fixed `N=8192`): Triton `45.113 ms` (repeat `45.028 ms`) vs HIP `30.651/30.932 ms`.
- Profiling/codegen deltas vs Step07 baseline:
  - Triton profile median `48.164 -> 46.532 ms`.
  - No spill regression: `VGPR=216`, `scratch_size=0`, private segment fixed size `0`.
  - TTGIR B shared layout changed `#shared1 vec=1 order=[1,0] -> vec=2 order=[1,0]`.
  - Static counters changed (`v_perm_b32 128 -> 188`, `s_waitcnt 108 -> 117`) but runtime still improved.
- Keep reason: first compiler-path variant that improves fixed KPI while preserving no-spill guardrails.
- Artifacts: `triton_hip_compare_fixed_step12/*`.

### Step12 (rejected, reverted) - compiler-path non-k B vec4 follow-up (gfx1151)
- Change (in `~/triton` `LowerLoops.cpp`): for the same Step11-matched path (`gfx1151`, WMMA, B operand fp8 byte-source, non-k-contig order), set swizzled-shared encoding to `vec=4, perPhase=1, maxPhase=1`, preserving order.
- KPI (fixed `N=8192`): Triton `44.997 ms` (repeat `45.125 ms`) vs HIP `30.961/30.842 ms`.
- Profiling/codegen deltas vs Step11 baseline:
  - Triton profile median `46.532 -> 46.619 ms` (neutral/slightly worse).
  - No spill regression: rocprof `VGPR=216`, private segment fixed size `0`.
  - TTGIR changed (`#shared1 vec=2 -> vec=4`), but final AMDGCN hash was identical to Step11 (`f2ef3faf7b0a29ed96044ffbf7f3f3e551b03a492ab5f75338d8ab02a59dcdb6`), with unchanged static counters (`v_perm_b32=188`, `s_waitcnt=117`).
- Reject reason: no stable KPI improvement and no backend codegen delta worth keeping.
- Action taken: reverted local Triton back to Step11 vec2 setting, rebuilt wheel, reinstalled local Triton.
- Artifacts: `triton_hip_compare_fixed_step13/*`.

### Step13 (rejected, reverted) - compiler-path non-k B swizzle-phase tuning (gfx1151)
- Change (in `~/triton` `LowerLoops.cpp`): keep Step11 `vec=2` preserved-order path, but change swizzled-shared params to `perPhase=2, maxPhase=8` for the same non-k B path.
- KPI (fixed `N=8192`): Triton `48.767 ms` (repeat `48.895 ms`) vs HIP `30.872/30.679 ms`.
- Profiling/codegen deltas vs Step11 baseline:
  - Triton profile median `46.532 -> 50.457 ms`.
  - No spill regression: private segment fixed size `0`, scratch disabled; but rocprof `VGPR` increased `216 -> 224`.
  - TTGIR B shared layout changed `#shared1 vec=2, perPhase=1, maxPhase=1 -> vec=2, perPhase=2, maxPhase=8`.
  - Static counters worsened (`s_waitcnt 117 -> 144`, `.amdhsa_next_free_vgpr 210 -> 220`) while `v_perm_b32` stayed `188`.
- Reject reason: clear runtime regression despite meeting no-spill guardrails; added scheduling/wait pressure outweighs any layout benefit.
- Action taken: reverted local Triton back to Step11 vec2 (`perPhase=1`, `maxPhase=1`), rebuilt wheel, reinstalled local Triton.
- Artifacts: `triton_hip_compare_fixed_step14/*`.

## Artifacts Index
- Step01: `triton_hip_compare_fixed_step01/`
- Step02: `triton_hip_compare_fixed_step02/`
- Step03: `triton_hip_compare_fixed_step03/`
- Step04: `triton_hip_compare_fixed_step05/`
- Step05: `triton_hip_compare_fixed_step06/`
- Step06: `triton_hip_compare_fixed_step07/`
- Step07: `triton_hip_compare_fixed_step08/`
- Step08: `triton_hip_compare_fixed_step09/`
- Step09: `triton_hip_compare_fixed_step10/`
- Step10: `triton_hip_compare_fixed_step11/`
- Step11: `triton_hip_compare_fixed_step12/`
- Step12: `triton_hip_compare_fixed_step13/`
- Step13: `triton_hip_compare_fixed_step14/`

## Triton Rebuild/Reinstall Note (local)
- Rebuild from source (in `~/triton`):
  - `python setup.py bdist_wheel`
- Reinstall rebuilt wheel:
  - `pip install --force-reinstall dist/triton-3.6.0+git917dbde5-cp313-cp313-linux_x86_64.whl`
- Verify active install:
  - `pip show triton`
- Important: wheel filename/version may stay unchanged across local rebuilds; treat each rebuild as new binary content even when name is identical.

## Non-Negotiable Run Protocol
1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_scaled_mm.py`
   - `python benchmark_scaled_mm.py`
   - If it regresses, explain the reason by inspecting the generated code and profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Active Checklist
- [x] Fixed-config baseline and profiling guardrails established.
- [x] Interior-only constrained runtime path is implemented and validated.
- [x] Three P2 source-level variants tested after interior-only policy (Step05/06/07).
- [x] Keep best source-level P2 result (Step07).
- [x] Start P3 compiler-path prototype (gfx1151-guarded).
- [x] Validate/reject first compiler-path prototype and revert failed patch.
- [x] Validate/reject second compiler-path follow-up and revert failed patch.
- [x] Validate/reject N8192 env-level HIP-parity trial (`AMDGCN_USE_BUFFER_OPS=0`).
- [x] Land first no-spill compiler-path win over Step07 baseline.
- [x] Validate/reject vec4 follow-up around Step11 baseline.
- [x] Validate/reject swizzle-phase (`perPhase/maxPhase`) follow-up around Step11 baseline.
- [ ] Reduce B operand permute/wait pressure via compiler changes.
- [x] Beat Step07 fixed-config KPI while preserving correctness.

## Next Plan (Step14: HIP-parity-oriented compiler follow-up)
1. Keep Step11 compiler baseline as the active reference (local Triton patch applied).
2. Avoid repeating forced B shared-order override path unless there is a new precondition that addresses spill risk.
3. Scope next compiler experiments to HIP-parity-oriented changes that preserve no-spill guardrails:
   - `.amdhsa_next_free_vgpr < 256`
   - `.amdhsa_private_segment_fixed_size == 0`
4. Keep default `AMDGCN_USE_BUFFER_OPS=1`; no more env-level buffer-op toggles without a new precondition.
5. New baseline for compiler-path gating is Step11 fixed repeat (`45.028 ms`).
6. Candidate directions:
   - target B conversion/permute lowering in compiler passes (e.g., `OptimizeDotOperands`) while preserving Step11 shared-order behavior,
   - reduce wait-heavy scheduling side effects in non-k B operand lowering without increasing VGPR.
7. Rerun fixed protocol for each new compiler change and gate on beating Step11 (`45.028 ms`) without correctness regression.
8. If no guarded compiler variant improves KPI, document an upstream follow-up item with minimal repro and side-by-side Triton/HIP codegen evidence.
