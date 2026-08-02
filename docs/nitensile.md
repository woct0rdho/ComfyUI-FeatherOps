# NiTensile: I have two dtypes, one is long, the other is short.

## Current Goal

The TensileLite kernel is already faster than the HIP kernel. The goal is to further optimize the TensileLite kernel. Keep measured TFLOP/s as the primary decision metric.

Public contract:
- `X`: fp16 row-major activation, logical `[M, K]`.
- `W`: fp8e5m2/B8 logical `[K, N]`, physically packed as AK16 `(K/16, N, 16)`.
- `Y`: fp16 row-major output, logical `[M, N]`.
- Epilogue: scalar fp32 scale plus fp16 output-vector bias over public `N`.

Current status:
- Average-speed recommendation is `SU32/SUS128`. Longer same-input direct comparison, `--warmup 12 --iters 64`: baseline `SU32/SUS256` TensileLite avg `23.811 ms / 46.177 TFLOP/s`, best `46.635`; `SU32/SUS128` TensileLite avg `23.780 ms / 46.237 TFLOP/s`, best `46.663`. Both are correct with no NaNs, `mismatches(abs>0.25)=0`, and max abs diff `0.25`.
- `SU32/SUS128` generated TensileLite client config was not hot-loop by default (`num-enqueues-per-sync=1`, `num-warmups=3`, `sleep-percent=300`). Hot-loop retime of the same generated code object with `num-warmups=20`, `num-enqueues-per-sync=10`, and `sleep-percent=0` passed all 10 samples: median `46.121 TFLOP/s`, mean `46.109 TFLOP/s`, best `46.330 TFLOP/s`, derived median time `23.840 ms`.
- Prior `SU32/SUS256` reference run, `--warmup 10 --iters 32`: HIP avg `24.410 ms / 45.043 TFLOP/s`; TensileLite avg `23.759 ms / 46.277 TFLOP/s`, best `46.715`. No NaNs, `mismatches(abs>0.25)=0`, max abs diff `0.25`.

Direct TensileLite launch and the TensileLite client agree in the same performance regime. For routine tuning, compare TensileLite client benchmark results against the HIP C++ benchmark. Use the direct same-input runner for semantic changes, ABI/kernarg changes, or final claims.

## Tensor Mapping

PyTorch row-major GEMM is executed as a transposed column-major BLAS view:

```text
Y_row[M,N] = X_row[M,K] @ W[K,N]
D_col[N,M] = A_col[N,K] @ B_col[K,M]
```

TensileLite `Cijk_Ailk_Bljk` mapping:
- `i -> public N`
- `j -> public M`
- `l -> public K`
- `Exact = [N, M, batch, K]`
- Public `W` maps to TensileLite operand `A`.
- Public `X` maps to TensileLite operand `B`.
- Public `Y_row[M,N]` shares physical storage with `D_col[N,M]`.

Final problem type:

```yaml
OperationType: GEMM
DataType: H
DataTypeA: B8
DataTypeB: H
MacDataTypeA: H
MacDataTypeB: H
DestDataType: H
ComputeDataType: S
HighPrecisionAccumulate: True
ConvertAfterDS: True
TransposeA: False
TransposeB: False
UseBeta: False
TensorALayoutA: 1
```

Scale/bias mapping:
- HIP scalar `scale[0]` maps to TensileLite `alpha`.
- HIP public-N fp16 bias maps to `UseBias: 1`, `BiasDataTypeList: [h]`, `FactorDimArgs: [0]`.
- Direct launch must pass `BiasStride=0` for this single vector. `BiasStride=1` restricts the bias SRD to one element and effectively drops bias for all columns except zero.

Terminology: `X/W/Y` are public tensors. `A/B/C/D` are TensileLite operands. Old `H/B8/H` runs put fp8 on TensileLite operand `B`. Keep them only as B8 codegen evidence, not final-interface results.

## Current Recommendation

Current average-speed recommendation is the exact-HIP shape with `LdsPadA=16`, `LdsPadB=8`, and retuned `StaggerUStride=128`. The prior `StaggerUStride=256` run remains a conservative baseline because the timing margin is small.

Core parameters:

```yaml
MatrixInstruction: [16, 16, 16, 1, 1, 2, 8, 8, 1]
MacroTile0: 256
MacroTile1: 128
DepthU: 64
TensorALayoutA: 1
GlobalReadVectorWidthA: 16
GlobalReadVectorWidthB: 8
TransposeLDS: 2
WaveSeparateGlobalReadB: 1
WorkGroupMapping: 4
StaggerU: 32
StaggerUStride: 128
StorePriorityOpt: True
GroupLoadStore: False
SourceSwap: 1
StoreRemapVectorWidth: 0
StoreVectorWidth: -1
NumElementsPerBatchStore: 0
StoreSyncOpt: 0
LdsPadA: 16
LdsPadB: 8
LdsBlockSizePerPadA: 128
LdsBlockSizePerPadB: 128
```

Retained resource state:
- `.amdhsa_next_free_vgpr 256`
- `.amdhsa_next_free_sgpr 74`
- `.amdhsa_group_segment_fixed_size 36864`
- `LdsNumBytes=36864`, `LdsNumElementsAlignedA=18432`, `LdsNumElementsAlignedB=18432`

Core matmul finite-input performance:
- TensileLite client exact-HIP `A16/B8`: best `46.702 TFLOP/s`, median `46.375 TFLOP/s`, mean `46.125 TFLOP/s`.
- Direct same-input core finite run: TensileLite avg `46.244 TFLOP/s`, best `46.707 TFLOP/s`; HIP core avg `45.103 TFLOP/s`.

Scale+bias performance:
- Current `SU32/SUS128` direct same-input, `alpha=2.34`, `--warmup 12 --iters 64`: TensileLite avg `46.237 TFLOP/s`, best `46.663 TFLOP/s`. Matched baseline correctness. This has the best measured direct average so far.
- Current `SU32/SUS128` TensileLite client hot-loop, `alpha=Two`: median `46.121 TFLOP/s`, mean `46.109 TFLOP/s`, best `46.330 TFLOP/s`. All 10 validation runs passed. This uses the generated client/reference path, not the direct same-input runner.
- Prior `SU32/SUS256` direct same-input, `--warmup 10 --iters 32`: TensileLite avg `46.277 TFLOP/s`, best `46.715 TFLOP/s`; HIP avg `45.043 TFLOP/s`. Keep it as a conservative reference because timing noise is non-trivial.
- Prior `SU32/SUS256` TensileLite client with `alpha=Two`: passed validation, best `46.294 TFLOP/s`, mean `45.633 TFLOP/s`.

Current code/config paths:
- Core matmul reference config: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_ldspad_a16_b8_thermal_8192.yaml`
- Core matmul reference output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_ldspad_a16_b8_thermal_8192_out`
- Scale+bias validation config: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_small.yaml`
- Scale+bias validation output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_small_out`
- Prior `SUS256` scale+bias benchmark config: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_thermal_8192.yaml`
- Prior `SUS256` scale+bias benchmark output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_thermal_8192_out`
- Current average-speed candidate config: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192.yaml`
- Current average-speed candidate output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out`
- Current average-speed candidate hot-loop override: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_hot_loop_override.ini`
- Current average-speed candidate hot-loop CSV: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_hot_loop.csv`
- Same-input direct runner: `tmp_nitensile/scripts/same_input_hip_tensile.cu`
- Same-input direct runner binary: `tmp_nitensile/scripts/same_input_hip_tensile`
- Same-input log: `tmp_nitensile/logs/same_input_hip_tensile_8192.log`

## Build And Run

Environment used for TensileLite generation and benchmarking:

```bash
export ROCM_PATH=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel
export PYTHONPATH=~/rocm-libraries/projects/hipblaslt/tensilelite:$ROCM_PATH/share/hipblaslt/tensilelite
export TENSILE=~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile
export TENSILE_CLIENT=~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client
```

Run a TensileLite client benchmark from a YAML config:

```bash
CONFIG=~/ComfyUI-FeatherOps/tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192.yaml
OUT=~/ComfyUI-FeatherOps/tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out
$TENSILE "$CONFIG" "$OUT" --prebuilt-client "$TENSILE_CLIENT" --global-parameters CodeObjectVersion=4
```

The generated `ClientParameters.ini` uses default cool-loop timing, not hot-loop timing. Retime the already generated current candidate with:

```bash
~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --config-file ~/ComfyUI-FeatherOps/tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs_00/00_Final/caches/bf7d336000e4/source/ClientParameters.ini \
  --config-file ~/ComfyUI-FeatherOps/tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_hot_loop_override.ini
```

Always use a Python CSV reader for TensileLite result CSVs. The first `GFlops` column can be zero. The actual GFLOP/s values are in the long per-solution kernel-name column after `TotalFlops`.

```python
import csv
from pathlib import Path
from statistics import mean, median

path = Path("<results.csv>")
with path.open(newline="") as f:
    rows = list(csv.reader(f))

header, data = rows[0], rows[1:]
total_idx = header.index(" TotalFlops") if " TotalFlops" in header else header.index("TotalFlops")
perf_cols = []
for idx, name in enumerate(header):
    if idx <= total_idx:
        continue
    try:
        values = [float(row[idx]) for row in data]
    except ValueError:
        continue
    if any(v != 0 for v in values):
        perf_cols.append((name.strip(), values))

name, gflops = perf_cols[-1]
print(name)
print("median_gflops", median(gflops))
print("mean_gflops", mean(gflops))
print("median_tflops", median(gflops) / 1000.0)
```

Build the direct same-input HIP/TensileLite runner:

```bash
$ROCM_PATH/lib/llvm/bin/clang++ -O3 --offload-arch=gfx1151 -x hip -DNO_PYTORCH \
  ~/ComfyUI-FeatherOps/tmp_nitensile/scripts/same_input_hip_tensile.cu \
  ~/ComfyUI-FeatherOps/kernel/hip/hip_kernel.cu \
  -o ~/ComfyUI-FeatherOps/tmp_nitensile/scripts/same_input_hip_tensile
```

Run the direct same-input scale+bias benchmark against a generated code object. Extract the exact exported symbol instead of manually reconstructing the long kernel name:

```bash
CO=~/ComfyUI-FeatherOps/tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs_00/00_Final/caches/bf7d336000e4/source/library/gfx1151/TensileLibrary_gfx1151.co
KERNEL_NAME=$($ROCM_PATH/lib/llvm/bin/llvm-readelf --wide -s "$CO" | awk '/FUNC/ && /Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs/ {print $NF; exit}')
~/ComfyUI-FeatherOps/tmp_nitensile/scripts/same_input_hip_tensile \
  --m 8192 --n 8192 --k 8192 \
  --warmup 12 --iters 64 --bench \
  --validate-elems 1048576 \
  --finite-b8-inputs --scale-bias --tensile-first --hip-ext-launch \
  --stagger-u 32 --stagger-u-stride 128 \
  --code-object "$CO" --kernel-name "$KERNEL_NAME"
```

For non-default `StaggerU` or `StaggerUStride`, pass matching launch metadata with `--stagger-u` and `--stagger-u-stride`. For direct launch, `WorkGroupMapping` is currently fixed in the runner as `4`, matching the active candidates. Do not use it for `WGM!=4` candidates without updating `kernel_info1` plumbing.

## Resource And Occupancy Notes

One possible direction is matching HIP's lower VGPR and LDS footprint. This may increase occupancy and combine with normal TensileLite search-space tuning to increase speed, but resource reduction is not the goal by itself. Prefer candidates that improve direct same-input or full-client TFLOP/s, even if they do not reduce VGPR/LDS.

Occupancy formula is in `doc/gfx1151_reference.md`:

```text
vgpr_allocated_per_wave = ceil(vgpr_used_per_wave / 24) * 24
waves_by_vgpr = floor(1536 / vgpr_allocated_per_wave)
workgroups_by_lds = floor(131072 / lds_per_workgroup_in_bytes)
waves_by_lds_per_simd = floor(workgroups_by_lds * waves_per_workgroup / 4)
occupancy_per_simd = min(waves_by_vgpr, waves_by_lds_per_simd, 16) / 16
```

Current TensileLite exact-HIP `A16/B8`, 8-wave workgroup:
- VGPR used `256`, allocated `264` per wave.
- `waves_by_vgpr = floor(1536 / 264) = 5`.
- LDS `36864`, `workgroups_by_lds = floor(131072 / 36864) = 3`.
- `waves_by_lds_per_simd = floor(3 * 8 / 4) = 6`.
- Occupancy limit: `min(5,6,16)/16 = 31.25%`.

HIP target resource point, 8-wave workgroup:
- VGPR used `184`, allocated `192` per wave.
- LDS `32768`.
- `waves_by_vgpr = 8`, `waves_by_lds_per_simd = 8`.
- Occupancy limit: `50%`.

Useful thresholds:
- Reducing VGPR from `256` to `<=240` increases VGPR-limited waves from `5` to `6`. With current `36864` LDS, this reaches `37.5%` occupancy.
- Reducing LDS from `36864` to `32768` alone does not improve occupancy while VGPR remains `256`. VGPR still limits to `5` waves/SIMD.
- Reducing VGPR to `<=216` and LDS to `32768` reaches `43.75%` occupancy.
- Matching HIP-like `<=192` allocated VGPR and `32768` LDS reaches `50%` occupancy.

Do not assume higher occupancy alone wins. Validate with multiple timed rows because the kernel is near the roofline and row-to-row noise is significant.

## Tuning History

Speed sweep tooling:
- Added `tmp_nitensile/scripts/make_du64_speed_sweep.py` to generate one-solution full-size YAMLs. Use one solution per YAML. Previous broad multi-solution sweeps hit generator/library failures.
- Added `tmp_nitensile/scripts/run_du64_speed_sweep.sh` and `tmp_nitensile/scripts/summarize_tensile_logs.py` to run and summarize sweep batches.
- Updated `tmp_nitensile/scripts/same_input_hip_tensile.cu` with `--stagger-u` and `--stagger-u-stride`. Rebuilt with `-DNO_PYTORCH`. Direct launch still assumes `WorkGroupMapping=4` in `kernel_info1`.

Speed sweep results around `DepthU=64`:
- `StaggerU x StaggerUStride`: all full-client candidates passed. Direct validation favored `SU32/SUS128` for average speed. Longer `--warmup 12 --iters 64` direct comparison: baseline `SU32/SUS256` TensileLite avg `46.177 TFLOP/s`, best `46.635`; candidate `SU32/SUS128` avg `46.237`, best `46.663`. Both correct with no NaNs and `mismatches(abs>0.25)=0`.
- `SU64/SUS256` passed client timing with mean `46.205`, but direct validation was slower than baseline: TensileLite avg `45.999`, best `46.597`. Do not promote.
- Schedule/global-read batch anchored on `SU32/SUS128`: only `WaveSeparateGlobalReadB=0, ScheduleGlobalRead=1, ScheduleLocalWrite=1` passed, and it was slower (`best 46.472`, mean `45.819`). `ScheduleLocalWrite=0` hit `_makeSubIterSchedule` `IndexError: pop from empty list`. `ScheduleGlobalRead=0, ScheduleLocalWrite=1` generated wrong results.
- Store-path batch anchored on `SU32/SUS128`: client noise favored `SPO=False/GLS=True/NEPBS=8/16`, but direct validation was slower (`NEPBS=16` avg `45.537`, `NEPBS=8` avg `45.611`). Keep `StorePriorityOpt=True`, `GroupLoadStore=False`, `NEPBS=0`.
- LDS pad/block batch anchored on `SU32/SUS128`: all valid candidates were slower than `LdsPadA=16`, `LdsPadB=8`, `LdsBlockSizePerPadA/B=128/128`. Severe regressions included `A16/B4` at `15.0 TFLOP/s` and `A8/B8` at `38.4`. Invalid/rejected cases included `LdsPadB=12`, `LdsPadA=24`, and block `128/64`.
- Local-read/prefetch batch anchored on `SU32/SUS128`: `PrefetchLocalRead=1/2` and `ClusterLocalRead=1` produced no valid solutions. `LocalReadVectorWidth=8/32` were rejected. Keep `PLR=0`, `CLR=0`, `LRVW=16`.
- `WorkGroupMapping` batch anchored on `SU32/SUS128`: all passed but were slower than `WGM=4`. Client means: `WGM=1` `45.349`, `WGM=2` `45.645`, `WGM=8` `45.026`, `WGM=16` `42.894`.
- `ScheduleIterAlg` batch anchored on `SU32/SUS128`: `SIA=0/1` were rejected and `SIA=2` produced no valid solution. Keep `ScheduleIterAlg=3`.

Resource and occupancy experiments:
- Retained DU64 assembly high-water: `ValuC v0-v127`; address temps `v128-v139`; `vgprBase=140`; `ValuA/B` and conversion temp `v140-v222`; `vgprSerial=223`; A G2L `v224-v239`; B G2L `v240-v255`.
- `PGR=0 + NEPBS=16` reduced resources to `226` VGPR / `36864` LDS but lost too much speed at full size: best `45.449`, mean `45.203`. Losing global prefetch costs more than the occupancy gain.
- Balanced wave/store-cap variant `MatrixInstruction=[16,16,16,1,1,4,4,4,2]`, `PGR=2`, `NEPBS=16` reached `240` VGPR / `36864` LDS but was slower: best `45.559`, mean `44.765`.
- `PGR=2` G2L/ValuB aliasing is unsafe: second prefetched G2LB data must survive until late local-write while ValuB is overwritten by local reads during compute.
- `DepthU=32 + PGR=2 + NEPBS=16` is the best resource point so far at `236` VGPR / `18432` LDS. Full client best `46.331`, mean `45.281`; direct `SU32` avg `46.034`; direct `SU64` avg `46.190`. It is useful if resource footprint matters, but it is not the speed winner.
- DU32 store-cap retune did not improve speed: `NEPBS=20` full client best `46.126`, mean `45.391`. `NEPBS=0/24/32` exceeded the clean VGPR target or slowed down.
- DU32 WGM/stagger retune did not beat DU64: `WGM=1/8` were slower, `SU16` was neutral/slower, and `SU64` improved mean but still did not beat DU64.
- DU64 `GRVWB=4` and `GRVWA=8` both failed generation with the same AK16 `260`-VGPR overflow layout. Do not pursue lower DU64 global-read vector widths without allocator changes.

Closed or deferred tuning paths:
- Do not debug `SGR=0`, `SLW=0`, or non-default `SIA` unless profiling later shows a strong reason. Current evidence is either wrong results, codegen crashes, or no valid solutions.
- Do not spend time on simple YAML store batching, WGM, local-read prefetch, or nearby LDS pad/block variants for the current DU64 shape. All tested candidates lost in direct validation or full-client screening.
- Do not pursue simple PGR2 G2L/Valu overlap, `DirectToLdsA`, compact `A0/B0` LDS, or StoreRemap/direct wide stores as quick YAML fixes. These are structural generator/toolchain projects.
- Public-`W`/TensileLite-`A` fragment reuse is already present. Do not spend time on a generic `all_reg_b` prototype unless a future retained assembly contradicts this.

## Next Optimization Plan

Primary rule: prioritize measured speed. Treat VGPR/LDS reduction as useful only when it improves direct same-input or full-client TFLOP/s.

Priority 1: Stabilize the current recommendation.
- Use `DepthU=64`, `StaggerU=32`, `StaggerUStride=128`, `WGM=4`, default store path, and current `A16/B8` LDS padding as the average-speed recommendation.
- For final claims, run direct same-input with finite B8 inputs, fused scale+bias, exact exported kernel symbol, and matching `--stagger-u 32 --stagger-u-stride 128`.
- Keep `SU32/SUS256` available as the conservative prior winner because the `SUS128` margin is small and timing noise is non-trivial.

Priority 2: Profile before more YAML sweeps.
- Use `rocprofv3` counters from `doc/gfx1151_reference.md`: `LDSBankConflict`, `L2CacheHit`, `VALUInsts`.
- Use PC sampling or ATT only after a stable repro and reduced scope. Large traces can be partial.
- Interpret PC samples carefully: high `v_perm_b32` samples can be instruction-fetch stalls from large inline literal instructions. High `ds_load_b128` samples can be LDS queue pressure.

Priority 3: Only do structural work if profiling points to it.
- If LDS pressure dominates, investigate a HIP-like LDS bank permutation rather than simple pad/block YAML changes.
- If output/epilogue dominates, investigate HIP-like C-shuffle or StoreRemap/128-bit output-store generator work.
- If occupancy is the limiter, revisit generator-level VGPR high-water reduction while preserving `PGR=2`. Avoid simple G2L/ValuB aliasing.

Priority 4: Keep resource candidate as fallback.
- Use DU32/SU64 (`236` VGPR / `18432` LDS) if lower resource footprint becomes more important than peak speed.
- Reopen DU32 tuning only if a workload or profiler result indicates the lower LDS/VGPR footprint should win despite current direct timings.

## Evidence To Preserve

Finite fp8 input caveat:
- The old HIP C++ benchmark filled fp8 with arbitrary raw `uint8_t`, including e5m2 Inf/NaN encodings.
- At `K=512` and `K=8192`, same-input validation showed all outputs became NaN.
- Raw-byte direct launch reached `49 TFLOP/s`, but that is an all-NaN/special-value timing, not representative finite matmul.
- HIP C++ benchmark initializers were patched to use finite e5m2 encodings for `{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0}`.

Direct/client consistency:
- Raw-byte HIP and TensileLite matched bitwise but all outputs were NaN.
- Finite core direct comparison: no NaNs, `mismatches(abs>0.25)=0`, max abs diff `0.125`.
- Scale+bias direct comparison: no NaNs, `mismatches(abs>0.25)=0`, max abs diff `0.25` over sampled `8192^3` outputs.
- Remaining scale+bias bit mismatches are expected: HIP rounds `acc*scale` to fp16 before adding fp16 bias. TensileLite adds fp16 bias in f32 before final fp16 conversion.

AK16 layout:

```text
addr(k, n) = (k >> 4) * (N * 16) + n * 16 + (k & 15)
```

- `TensorALayoutA: 1` marks the dedicated AK16 layout.
- Operand-A GPU upload is repacked in the TensileLite client.
- `GlobalReadVectorWidthA=16` vectorizes across the 16 K bytes inside one AK16 record.
- `TransposeLDS=2` is required for direct AK16 `buffer_load_b128` to `ds_store_b128` because A must be unroll-major/K-contiguous in LDS.
- Direct same-input launcher must pass AK16 operand-A `strideAL=N`, not `N*16`. The AK16 address macro already multiplies record addresses by `16`. Passing `N*16` caused GPU TCP page faults.

B8 conversion selectors for e5m2:

```asm
v_perm_b32 dst_lo, 0, src, 0x010c000c  ; b0,b1 -> two fp16 lanes
v_perm_b32 dst_hi, 0, src, 0x030c020c  ; b2,b3 -> two fp16 lanes
```

Do not assume these selectors are correct for `B8N`/FNUZ without a separate check.

Fragment reuse audit:
- HIP `all_reg_b` maps to public `W`, which is TensileLite operand `A` in final orientation.
- Retained exact-HIP TensileLite assembly already matches HIP reuse: public-`W`/TensileLite-`A` `ds_load_b128=8` and `v_perm_b32=64` per wave per `DepthU=64`.
- A reload-per-repeated-M implementation would show `ds_load_b128=64` and `v_perm_b32=512`. That is not what the retained loop bodies show.

LDS findings:
- `38912`/`36864` LDS in current generated kernels is padded A/B LDS, not hidden C-shuffle allocation.
- Compact exact-HIP `32768` A/B LDS validates but runs only `14.24~14.33 TFLOP/s`, indicating severe LDS bank conflicts.
- `LdsPadA=16`, `LdsPadB=8`, block interval `128` is the confirmed best LDS setting so far. Refinements around B pad/block interval did not beat it.

Closed or limited paths:
- `DirectToLdsA` is blocked on gfx1151: `HasDirectToLds` and `HasDirectToLdsx4` report unavailable, and the TensileLite validator rejects B8-to-H conversion with DTL-A.
- DU64 `GRVWA=8` and `GRVWB=4` overflow the current AK16 VGPR allocator (`260` VGPR). Earlier lower-width experiments remain historical only.
- `GRVWA=4` is correct but slower due to byte LDS stores and extra shifts.
- StoreRemap/direct wide stores are validator/dataflow blocked for this final orientation: SourceSwap and vector-width rules reject the useful settings.
- Native fp8/bf8 conversion opcodes are not available in this gfx1151 toolchain. Optimize the `v_perm_b32` path.

## Patches Made

Patches are under `~/rocm-libraries/projects/hipblaslt/tensilelite` unless otherwise noted.

Generator and problem plumbing:
- `Tensile/SolutionStructs/Solution.py`: allow B8/B8N `ConvertAfterDS` when MAC type is fp16.
- `Tensile/SolutionStructs/Solution.py`: add AK16 constraints and allow `GlobalReadVectorWidthA in (1, 4, 8, 16)`.
- `Tensile/SolutionStructs/Problem.py`, `Tensile/Contractions.py`, client plumbing: serialize/pass `TensorALayoutA`.
- `Tensile/KernelWriter.py`: disable unsafe generic GRO path for AK16 and add serial-padding allocator fix.

B8 conversion and AK16 movement:
- `Tensile/Components/LocalRead.py`: B8/B8N to fp16 expansion using `v_perm_b32` after LDS reads.
- `Tensile/Components/LocalRead.py`: restrict one-element LDS read special case to operands where storage and MAC bytes differ.
- `Tensile/KernelWriterAssembly.py`: B8 selector SGPRs and conversion path.
- `Tensile/KernelWriterAssembly.py`: AK16 `GLOBAL_OFFSET_A`, scalar byte G2L fix, `GRVWA=4/8/16` vector-coordinate/graShift logic, and direct b64/b128 local-write paths.

Client/runtime/reference:
- `client/src/DataInitialization.cpp`: pack operand `A` for AK16 GPU upload.
- `client/include/DataInitialization.hpp`: AK16 packing hook and finite init definitions.
- `client/include/TypedId.hpp`, `client/src/Reference.cpp`: minimal runtime reference dispatch for H/B8 storage variants.

Repository benchmark/control patches:
- `cpp_benchmarks/benchmark_scaled_mm_hip.cpp`: finite fp8 initializer.
- `cpp_benchmarks/benchmark_scaled_mm_hip_libtorch.cpp`: finite fp8 initializer.
- `tmp_mixed_precision_analysis/benchmark_scaled_mm_fp8_convert_ablation.cpp`: finite fp8 initializer.
- `tmp_mixed_precision_analysis/benchmark_scaled_mm_lds_vgpr_ablation.cpp`: finite fp8 initializer.
- `tmp_mixed_precision_analysis/benchmark_scaled_mm_vram_lds_ablation.cpp`: finite fp8 initializer.
- `tmp_nitensile/scripts/same_input_hip_tensile.cu`: direct-launch runner with core and `--scale-bias` modes, NaN-aware validation, `hipExtModuleLaunchKernel` control, finite B8 input control, and `--stagger-u`/`--stagger-u-stride` metadata controls.

## Key Logs And Paths

Important current artifacts:
- Plan: `doc/nitensile.md`
- Hardware/occupancy reference: `doc/gfx1151_reference.md`
- HIP source: `kernel/hip/hip_kernel.cu`
- HIP finite benchmark binary: `cpp_benchmarks/build/benchmark_scaled_mm_hip`
- Same-input runner: `tmp_nitensile/scripts/same_input_hip_tensile.cu`
- Same-input runner binary: `tmp_nitensile/scripts/same_input_hip_tensile`
- Core matmul reference output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_ldspad_a16_b8_thermal_8192_out`
- Current scale+bias output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out`
- Current scale+bias hot-loop override: `tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_hot_loop_override.ini`
- Current scale+bias hot-loop CSV: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_hot_loop.csv`
- Prior `SUS256` scale+bias output: `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_thermal_8192_out`

Useful historical outputs:
- `tmp_nitensile/outputs/b8h_ak16_grvwa16_tlds2_best_confirm_thermal_8192_out`: prior non-exact thermal/noise confirmation.
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_serialpad_buildonly_out`: exact HIP shape unblocked by serial-padding allocator.
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_compactlds_thermal_8192_out`: compact LDS correctness but severe bank-conflict slowdown.
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_ldspad_3x3_8192_out`: LDS pad sweep identifying `A16/B8`.
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_ldspad_refine_8192_out`: refinement around `A16/B8`. No better variant.
- `tmp_nitensile/logs/same_input_hip_tensile_8192.log`: raw-byte/finite/direct-launch controls.
- `tmp_nitensile/logs/client_exact_a16b8_directlike.csv`: directlike TensileLite client control.

Generated code object used by current direct scale+bias runner:
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs_00/00_Final/caches/bf7d336000e4/source/library/gfx1151/TensileLibrary_gfx1151.co`

Retained scale+bias assembly:
- `tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs_00/00_Final/caches/bf7d336000e4/source/build_tmp/SOURCE/assembly/Cijk_Ailk_Bljk_B8H_HHS_AK16_H_Bias_H_UserArgs_MT0sQspodbWToMkZuGKDqgdqWFcxv8p410l0lU6JeRV74=.s`

## TensileLite Usage Notes

- Persistent work area: `tmp_nitensile/` with `configs/`, `outputs/`, `logs/`, `scripts/`.
- Run TensileLite through `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile`.
- Put source checkout first in `PYTHONPATH`: `PYTHONPATH=~/rocm-libraries/projects/hipblaslt/tensilelite:$ROCM_PATH/share/hipblaslt/tensilelite`.
- Use the centralized prebuilt client: `~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client`.
- Use `--global-parameters CodeObjectVersion=4` for gfx1151 runs.
- Use `KeepBuildTmp=True` to retain generated `.s`.
- `ConvertAfterDS=True` belongs in solution/fork parameters, not the problem type.
- Do not edit generated `.hip` files. Edit `.cu` files.

Example command:

```bash
ROCM_PATH=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel \
PYTHONPATH=~/rocm-libraries/projects/hipblaslt/tensilelite:$ROCM_PATH/share/hipblaslt/tensilelite \
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile \
  ~/ComfyUI-FeatherOps/tmp_nitensile/configs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192.yaml \
  ~/ComfyUI-FeatherOps/tmp_nitensile/outputs/b8h_ak16_hiptile_exact_a16b8_bias_alpha_su32_sus128_thermal_8192_out \
  --prebuilt-client ~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --global-parameters CodeObjectVersion=4
```

## Environment Notes

- `$ROCM_PATH` is `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel`.
- `$ROCM_PATH/bin/amdclang++` wrapper is broken in this SDK layout. Use `$ROCM_PATH/lib/llvm/bin/clang++` or `$ROCM_PATH/lib/llvm/bin/amdclang++` as appropriate.
- Build the TensileLite client with `~/rocm-libraries/build_tensilelite_client.sh`.
- The centralized client build lives under `~/rocm-libraries/build/tensilelite-client/`.
- Build folders are gitignored and may not appear in repo search tools. Use direct paths.

## Legacy Evidence

Old `H/B8/H` orientation results are retained only to show B8 conversion/codegen behavior:
- Smoke validation passed: `/tmp/opencode/nitensile_hb8_out_keep5`.
- Square repeat around `29.85-30.28 TFLOP/s`: `/tmp/opencode/nitensile_hb8_phase2_best_repeat_8192_out`.
- Consolidated by-shape repeat: `/tmp/opencode/nitensile_hb8_phase2_best_by_shape_repeat_out`.
