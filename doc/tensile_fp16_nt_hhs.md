# TensileLite FP16 NT HHS Plan

## Current State

- Target: gfx1151 NT GEMM through TensileLite with fp16 inputs, fp32 accumulation, fp16 output, fp32 scale-alpha-vector, and fp16 bias.
- Problem shape: `8192 x 8192 x 8192`, batch count `1`.
- Public NT maps to TensileLite `Cijk_Ailk_Bjlk` with `TransposeA: False`, `TransposeB: True`.
- Current best guarded/no-`ForceStaticWGM8` candidate: `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe`, aggregated median `45.253 TFLOP/s` over 30 hot-loop samples across three independent passes (best single-pass median `46.342 TFLOP/s`). This beats the historical forced-WGM8 best by ~12%.
- Historical forced-WGM8 best with legacy `ForceStaticWGM8=True`: `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe`, median `40.484 TFLOP/s`.
- Source now removes the public `ForceStaticWGM8` knob and keeps the guarded static-WGM8 path. The winning configuration uses `ScheduleIterAlg=3` (key differentiator over the rocBLAS-like `SIA2`), `StorePriorityOpt=False`, and `NumElementsPerBatchStore=10`.

## Benchmark Protocol

Use hot-loop timing for performance claims, because that matches how PyTorch benchmarks repeated GEMMs. Generated TensileLite `ClientParameters.ini` files usually use default cool-loop timing (`num-enqueues-per-sync=1`, `num-warmups=3`, `sleep-percent=300`). Treat those as screening numbers only.

Hot-loop override pattern for performance claims:

```ini
num-benchmarks=10
num-warmups=20
num-enqueues-per-sync=10
num-syncs-per-benchmark=1
sleep-percent=0
hardware-monitor=False
results-file=/path/to/results.csv
```

Run an already generated TensileLite candidate with hot-loop timing:

```bash
~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --config-file <generated ClientParameters.ini> \
  --config-file <hot-loop override.ini>
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

## Build And Run

Build or rebuild the TensileLite client:

```bash
~/rocm-libraries/build_tensilelite_client.sh
```

- Output client: `~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client`.

Build or rebuild the full Tensile client:

```bash
~/rocm-libraries/build_tensile_client.sh
```

- Output client: `~/rocm-libraries/build/tensile-client/tensile-client`.
- Full Tensile client is required for full Tensile generated `TensileLibrary.dat`. The TensileLite client does not parse full Tensile's binary/msgpack library file.

List HHS scale/bias variants:

```bash
python3 ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/make_fp16_nt_hhs_scale_bias_sweep.py --list
```

Generate and benchmark one or more HHS scale/bias variants with default generated timing:

```bash
bash ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/run_fp16_nt_hhs_variant.sh <variant> [<variant> ...]
```

Use the default runner to screen candidates. Before making a performance claim, retime shortlisted generated candidates with the hot-loop override above.

## Layout And Epilogue Mapping

- Public benchmark `NT` maps to TensileLite operation `Cijk_Ailk_Bjlk`.
- Wrapper output `D` is hipBLASLt N-layout/column-major, with `stride(0) == 1` and `stride(1) == M`.
- Current wrapper contract requires scale and bias vectors with `M` elements. The square benchmark creates length-`N` vectors, which is equivalent only because `M == N`.
- Scale/bias settings: `UseBias: 1`, `UseScaleAlphaVec: 1`, `BiasDataTypeList: [h]`, `BiasTypeArgs: ['h']`, `FactorDimArgs: [0]`.
- `scaleAlphaVec` dtype is TensileLite `ComputeDataType`, so this HHS path takes fp32 scale vectors.
- Wrapper semantics: scalar `alpha=1.0`, fp32 scale-alpha-vector of ones, fp16 bias, and scalar `beta=0.0`. Standalone configs should use `UseBeta: True` and `DataInitTypeBeta: 0`.

Direct same-input benchmark runner:

```bash
~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++ \
  -x hip --offload-arch=gfx1151 \
  --rocm-path=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel \
  -I~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/include \
  -L~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/lib \
  -D__HIP_PLATFORM_AMD__ -DNO_PYTORCH -O3 -std=c++17 \
  ~/ComfyUI-FeatherOps/kernel/hip/hip_kernel_fp16.cu \
  ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/same_input_hip_tensile_hhs_nt.cu \
  -lamdhip64 \
  -o ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/same_input_hip_tensile_hhs_nt
```

Validation command:

```bash
~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/same_input_hip_tensile_hhs_nt \
  --m 1024 --n 1024 --k 1024 \
  --validate-elems 1048576 --hip-ext-launch
```

Direct benchmark command with the same hot-loop shape as the TensileLite client retime:

```bash
~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/scripts/same_input_hip_tensile_hhs_nt \
  --m 8192 --n 8192 --k 8192 \
  --bench --warmup 20 --iters 10 --enqueues-per-sync 10 \
  --tensile-first --hip-ext-launch
```

Direct runner notes:
- The direct-runner default still points at the historical forced-WGM8 `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe` artifacts under cache `70928ca179f0`. It has not yet been retargeted to the current guarded SIA3/no-store-priority winner.
- The direct TensileLite launch uses scalar `alpha=1.0`, scalar `beta=0.0`, an fp32 `scaleAlphaVec` filled with `1.0f`, and one shared device half-bias pointer.
- Default bias initialization mirrors `cpp_benchmarks`: `std::mt19937 gen(42)`, `std::uniform_real_distribution<float>(-1.0f, 1.0f)`, converted to `half`. Use `--tensile-random-bias` only to reproduce the TensileLite benchmark-style `Random` half integer distribution in `[-3, 3]`.
- A/B input generation defaults to TensileLite-style half `Random` integer values in `[-3, 3]`. Use `--small-random-inputs` only for diagnostic `[-0.25, 0.25]` input sensitivity checks. The performance dependency on inputs is explained by transistor switching power, see https://www.thonking.ai/p/strangely-matrix-multiplications
- Runtime metadata matches the generated solution: `StaggerU=32`, `staggerStrideShift=3`, `GSU=1`, `WorkGroupMapping=8`, and `WorkGroupMappingXCC=1`.
- Validation compares against `kernel/hip/hip_kernel_fp16.cu` with the same A, B, and bias buffers by transposing the HIP reference problem. HIP receives `A_hip = B^T` as row-major `N x K` and `B_hip = A^T` prepacked as `K x M`, computes `(B^T @ A^T) + bias[column]`, and the validator compares `HIP[j, i]` to TensileLite `D[i, j]`. This maps HIP's column-bias epilogue to TensileLite's factor-dim `I/M` bias epilogue.
- Small/non-WGM8-divisible shapes are not reliable for this forced static-WGM8 generated kernel. Use shapes where `ceil(N / 128) % 8 == 0` for direct correctness sanity checks. For square aligned shapes this means `N % 1024 == 0`.
- `1024^3` full-output validation with default `cpp_benchmarks` bias passed: `mismatches(abs>4)=0`, `max abs diff=0.125`.
- `1536^3` full-output validation failed because `ceil(1536 / 128)=12` is not divisible by `8`. This confirms the limitation is WGM8 divisibility, not a simple lower-size threshold.
- `2048^3` full-output validation passed: `mismatches(abs>4)=0`, `max abs diff=0.25`.
- Full `8192^3` validation with default `cpp_benchmarks` bias passed: `mismatches(abs>4)=0`, `max abs diff=0.5` over all `67108864` output elements.
- Full direct `8192^3` with default `cpp_benchmarks` bias reached median `38.593 TFLOP/s`, mean `38.690 TFLOP/s`, range `38.210 - 39.018 TFLOP/s` in the latest rerun. An earlier run measured median `36.226 TFLOP/s`. Treat the newer result as the current direct measurement.
- Full direct `8192^3` with `--tensile-random-bias` reached median `39.987 TFLOP/s`, mean `39.979 TFLOP/s`, range `39.698 - 40.155 TFLOP/s`. This is close to the generated-client hot-loop median `40.484 TFLOP/s`.

## Hot-Loop Results

The table keeps decision-relevant hot-loop results. Full per-probe details and CSV paths are in the experiment log and artifact section below.

| Kernel/Path | Median GFLOP/s | Mean GFLOP/s | Notes |
| --- | ---: | ---: | --- |
| TensileLite scale/bias VWB2 guarded static-WGM8, NEPBS10, SIA3, no store priority | `45252.9` | `45491.6` | **Current best no-`ForceStaticWGM8`.** Aggregated over 3 passes (30 total samples). `ScheduleIterAlg=3`, `StorePriorityOpt=False`, `NumElementsPerBatchStore=10`, `PGR1/PLR1/VWB2/1LDSB1`. Resources: vgpr=219, sgpr=74, LDS=8192 |
| TensileLite scale/bias VWB2 guarded static-WGM8, NEPBS10, SIA3 | `40968.9` | `40880.1` | SIA3 with store priority on, same resources. ~10% slower sister config |
| TensileLite scale/bias VWB2 guarded static-WGM8, NEPBS10 | `38292.3` | `38375.1` | Best repeat-confirmed SIA2 config in the original NEPBS sweep |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS8 | `40483.9` | `40432.8` | Historical best for `8192^3` with `ForceStaticWGM8=True` |
| TensileLite scale/bias VWB2 guarded static-WGM8, NEPBS10, SIA3, no store priority (single pass peak) | `46341.6` | `46404.0` | Best single 10-sample pass observed; included for reference, not the aggregated claim |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS4 | `40407.6` | `40395.5` | `8192^3` target validated. Slower than NEPBS8 |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS16 | `40365.0` | `40386.2` | Prior best for `8192^3`. `NumElementsPerBatchStore=16` |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS12 | `40344.4` | `40378.0` | `8192^3` target validated. Slower than NEPBS8/16 |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS16, no store priority | `40272.6` | `40240.6` | `8192^3` target validated. Slightly slower than NEPBS16 alone |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS8, GroupLoadStore | `40425.1` | `40440.8` | `8192^3` target validated but slower than NEPBS8. `GroupLoadStore=True` |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS8, StoreSyncOpt=4 | `40417.2` | `40390.4` | `8192^3` target validated but slower than NEPBS8 |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS8, StoreSyncOpt=1 | `40388.1` | `40430.0` | `8192^3` target validated but slower than NEPBS8 |
| TensileLite scale/bias VWB2 forced static-WGM8, NEPBS8, no activation | `40340.0` | `40362.9` | `8192^3` target validated but slower. `Activation=False`, `ActivationType=none` |
| TensileLite scale/bias VWB2 forced static-WGM8, no store priority | `37309.9` | `37288.2` | Small win over forced-WGM8 baseline for `8192^3`. `StorePriorityOpt=False` |
| TensileLite scale/bias VWB2 forced static-WGM8 | `37220.4` | `37184.7` | Prior best for `8192^3`. Opt-in `ForceStaticWGM8=True`, no dynamic WGM fallback emitted |
| TensileLite scale/bias VWB2 guarded static-WGM8 | `36890.8` | `36834.4` | Previous best. Runtime guard/fallback retained |
| TensileLite pure-HHS VWB2 WGM1 config control | `14544.3` | `14559.9` | Runtime-skips positive WGM remap but destroys traversal/cache behavior. Validated and ruled out |

Historical numbers benchmarked with cool loop, such as the older `mt128x128_tlds0_pgr1` median `29453.9 GFLOP/s`, are retained only as screening evidence. They are not directly comparable to PyTorch or hot-loop Tensile client results.

Current TensileLite summary:
- The best guarded/no-`ForceStaticWGM8` config found is `NEPBS10 + SIA3 + StorePriorityOpt=False`: aggregated 30-sample median `45252.9 GFLOP/s`. It is about 12% faster than the historical forced-WGM8 NEPBS8 best (`45252.9` vs `40483.9 GFLOP/s`) and also faster than the SIA3-with-store-priority sister config (`45252.9` vs `40968.9 GFLOP/s`).
- The combined SIA3/no-store-priority change is approximately 18% faster than the selected SIA2 NEPBS10 baseline (`45252.9` vs `38292.3 GFLOP/s`). SIA3 alone is about 7% faster than that SIA2 baseline (`40968.9` vs `38292.3 GFLOP/s`), and disabling store priority adds about 10% on top of SIA3.
- Historical forced static-WGM8 with `NumElementsPerBatchStore=8` was the prior absolute best for the exact `8192^3` target (`40483.9 GFLOP/s`) but is not shape-generic.
- The winning config keeps the guarded runtime WGM/divisibility fallback, so it is correct for off-grid shapes in a GridBased library.
- Remaining work: the SIA3 + StorePriorityOpt=False combination is a source-code neighborhood worth inspecting in assembly before further config-only probes. Do not pursue broad ABI changes.

## rocBLAS And Full-Tensile Reference

PyTorch/rocBLAS call and kernel identity:
- PyTorch entry point: `~/pytorch/aten/src/ATen/native/cuda/Blas.cpp`.
- `cublasCommonArgs` in `~/pytorch/aten/src/ATen/native/cuda/cuBlasCommonArgs.h` reformulates row-major output as a column-major BLAS problem with operand swap and transpose-flag flip.
- ROCm fp16 GEMM fallback in `~/pytorch/aten/src/ATen/cuda/CUDABlas.cpp` calls `rocblas_gemm_ex` with fp16 A/B/C/D and fp32 compute.
- rocBLAS log: `tmp_tensile_fp16_nt_hhs/logs/pytorch_rocblas_nt_8192`.
- rocprof DB: `tmp_tensile_fp16_nt_hhs/rocprof_pytorch_rocblas_nt_8192/pytorch_rocblas_nt_8192_results.db`.
- rocBLAS call: `transA=N`, `transB=T`, `M=N=K=8192`, fp16 A/B/C/D, fp32 compute, `alpha=1.0`, `beta=0.0`, `solution_index=0`.
- Selected kernel: `Cijk_Ailk_Bjlk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB1_AMAS0_BL1_BS1_EPS1_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1151_IU1_K1_KLA_LBSPPA0_LBSPPB0_LPA0_LPB0_LRVW16_MIAV1_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA2_SS1_SU32_SUS256_SVW1_TT4_64_TLDS0_UMLDSA0_UMLDSB0_USFGROn1_VAW2_VSn1_VW1_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGM8`.
- rocprof resources: grid `(8192, 64, 1)`, workgroup `(128, 1, 1)`, LDS `8192`, VGPR `256`, SGPR `128`, no scratch.

Exact rocBLAS logic entry:
- File: `~/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/strixhalo/strixhalo_Cijk_Ailk_Bjlk_HHS_BH.yaml`.
- `SolutionIndex: 18`.
- Key structure: `MT128x128x16`, `MI16x16x16x1`, `1LDSBuffer: 1`, `PrefetchGlobalRead: 1`, `PrefetchLocalRead: 1`, `ScheduleIterAlg: 2`, `ScheduleGlobalRead: 1`, `ScheduleLocalWrite: 1`, `ThreadTile: [4, 64]`, `WorkGroup: [32, 4, 1]`, `WorkGroupMapping: 8`, `GlobalLoadVectorWidthA/B: 8`, `GlobalReadVectorWidth: 8`, `LocalReadVectorWidth: 16`, `StoreVectorWidth: 1`, `VectorWidth: 1`, `VectorAtomicWidth: 2`, `StaggerU: 32`, `StaggerUStride: 256`.

Packaged rocBLAS loading and extraction:
- Packaged lazy library: `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/rocblas/library/gfx1151/TensileLibrary_lazy_gfx1151.dat`.
- Packaged compressed object: `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/rocblas/library/gfx1151/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.co`.
- Helper HSACO: `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/rocblas/library/gfx1151/Kernels.so-000-gfx1151.hsaco`.
- The packaged `.co` is a compressed Clang offload bundle (`CCOB`). `clang-offload-bundler --list --type=o` reports `host-x86_64-unknown-linux-gnu-` and `hipv4-amdgcn-amd-amdhsa--gfx1151`.
- Raw unbundled HSACO: `tmp_tensile_fp16_nt/outputs_full_tensile/packaged_rocblas_unbundled/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1151.raw.hsaco`.

Useful client configs and CSVs:
- Packaged CCOB client config: `tmp_tensile_fp16_nt/configs/packaged_rocblas_hhs_bh_8192_client.ini`.
- Packaged hot-loop override: `tmp_tensile_fp16_nt/configs/packaged_rocblas_hot_loop_override.ini`.
- Packaged hot-loop CSV: `tmp_tensile_fp16_nt/outputs_full_tensile/packaged_rocblas_hhs_bh_8192_hot_loop.csv`.
- Raw HSACO client config: `tmp_tensile_fp16_nt/configs/raw_rocblas_hhs_bh_8192_client.ini`.
- Raw hot-loop override: `tmp_tensile_fp16_nt/configs/raw_rocblas_hot_loop_override.ini`.
- Raw hot-loop CSV: `tmp_tensile_fp16_nt/outputs_full_tensile/raw_rocblas_hhs_bh_8192_hot_loop.csv`.
- Full-Tensile beta-zero generated config: `tmp_tensile_fp16_nt/outputs_full_tensile/rocblas_hhs_bh_solution18_beta0_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_00/00_Final/source/ClientParameters.ini`.
- Full-Tensile hot-loop override: `tmp_tensile_fp16_nt/configs/full_tensile_solution18_hot_loop_override.ini`.
- Full-Tensile hot-loop CSV: `tmp_tensile_fp16_nt/outputs_full_tensile/rocblas_hhs_bh_solution18_beta0_8192_hot_loop.csv`.

Full-Tensile notes:
- rocBLAS builds from full Tensile under `~/rocm-libraries/shared/tensile`; hipBLASLt uses TensileLite under `~/rocm-libraries/projects/hipblaslt/tensilelite`.
- Full Tensile here reports `v4.47.0`. Reference configs need `MinimumRequiredVersion: 4.47.0` or lower.
- Full Tensile uses `GlobalLoadVectorWidthA/B`; TensileLite uses `GlobalReadVectorWidthA/B`.
- Full Tensile uses `VectorWidth` plus `VectorWidthB`; TensileLite uses `VectorWidthA` plus `VectorWidthB`.
- Pass the compiler with `--cxx-compiler <path>`. Do not use `--global-parameters CxxCompiler=<path>` because full Tensile parses global parameter values through Python `eval`.
- Do not pass `--global-parameters CodeObjectVersion=4`. Use full Tensile's `--code-object-version V4` when needed.
- `TensileCreateLibrary` exact-object output is useful for metadata/assembly, but current full client crashes when benchmarking that YAML library directly: hard-index mode in `AllSolutionsIterator::getSolution`, best-solution mode in `BestSolutionIterator::preProblem`.

## TensileLite Evidence And Blockers

### Key TensileLite Artifacts

- Current guarded winner `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe`: generated source root `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00/00_Final/caches/1d7647921045/source/`, final h2h CSVs `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe_8192_final_h2h{1,4,5}.csv`, aggregated result median `45252.9 GFLOP/s`, mean `45491.6`.
- Historical forced-WGM8 best `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe`: generated source root `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00/00_Final/caches/70928ca179f0/source/`, hot-loop override `tmp_tensile_fp16_nt_hhs/configs/tensilelite_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe_hot_loop_override.ini`, CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe_8192_hot_loop.csv`, result median `40483.9 GFLOP/s`, mean `40432.8`, best `40556.7`, derived median `27.159 ms`.
- Static-WGM history: guarded static-WGM hot-loop CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_probe_8192_hot_loop.csv`, post-tightening CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_probe_8192_rebaseline_hot_loop.csv`, forced-WGM8 baseline CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_probe_8192_hot_loop.csv`.
- Store scheduling CSVs: no-store-priority `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nostoreprio_probe_8192_hot_loop.csv`, NEPBS16 `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_probe_8192_hot_loop.csv`, NEPBS16+no-store-priority `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_nostoreprio_probe_8192_hot_loop.csv`.
- Later config-only probe CSVs: NEPBS4/12/24/32, no-activation, `StoreSyncOpt=1`, `StoreSyncOpt=4`, and `GroupLoadStore=True` all use the same output naming pattern `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_<variant>_8192_hot_loop.csv`. Exact variant names and outcomes are in the experiment log and summary table.
- Older anchor `mt128x128_tlds0_pgr1`: generated config `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_pgr1_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00/00_Final/caches/d49d8309b3e0/source/ClientParameters.ini`, hot-loop override `tmp_tensile_fp16_nt_hhs/configs/tensilelite_mt128x128_tlds0_pgr1_hot_loop_override.ini`, CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_pgr1_8192_hot_loop.csv`, result median `33942.5 GFLOP/s`, mean `33947.3`.

### Experiment Log

- Historical `ForceStaticWGM8` opt-in probe completed. When explicitly enabled on WGM8 it emitted the scalar WGM8 remap directly and omitted runtime WGM/divisibility guards plus dynamic WGM fallback code. This was validated only for target/divisible shapes, not for arbitrary runtime problem sizes. The public parameter has since been removed from source.
- Pure-HHS forced-WGM8 validates for the exact target probe. Assembly removes `label_WGMStaticFallback`, `label_WGMPositive`, and the WGM dynamic `v_cvt_f64_u32`/`v_rcp_f64` block. Resources stay `.sgpr_count: 74`, `.vgpr_count: 255`, `vgprSerial=218`.
- Scale/bias forced-WGM8 validates for the exact target probe with unchanged resources `.sgpr_count: 74`, `.vgpr_count: 256`, `vgprSerial=218`. Hot-loop median is `37220.4 GFLOP/s`. This was the best before store-scheduling probes.
- WGMXCC no-op skip probe was rejected. It removed the XCC remap block and pure-HHS validated, but scale/bias regressed to median `33097.9 GFLOP/s`. The source edit was reverted.
- After reverting the WGMXCC skip, `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_probe` was regenerated so the output directory again matches the keeper source patch set.
- Store scheduling probes were added on top of forced-WGM8: `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nostoreprio_probe`, `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_probe`, and `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_nostoreprio_probe`.
- Store scheduling validation/timing started. `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nostoreprio_probe` generated, validated, and hot-loop retimed at median `37309.9 GFLOP/s`, mean `37288.2`, best `37532.2`. This is a small win over baseline.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_probe` generated, validated, and hot-loop retimed at median `40365.0 GFLOP/s`, mean `40386.2`, best `40666.0`. This became the best at that point, then was superseded by NEPBS8.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs16_nostoreprio_probe` generated, validated, and hot-loop retimed at median `40272.6 GFLOP/s`, mean `40240.6`, best `40417.1`. Reject because it is slightly slower than NEPBS16 alone.
- NEPBS16 assembly inspection: it changes the non-edge OptNLL store path from `4` global-write batches of about `32` elements to `8` batches of `16`, while dropping resources from `.vgpr_count: 256` to `.vgpr_count: 238` and keeping `.sgpr_count: 74`. The keeper still uses `StorePriorityOpt=True`. Removing it with NEPBS16 regressed slightly. This motivated the later `NumElementsPerBatchStore` sweep that promoted NEPBS8.
- Added forced-WGM8 config-only batch-size probes `NEPBS8`, `NEPBS12`, `NEPBS24`, and `NEPBS32` in `tmp_tensile_fp16_nt_hhs/scripts/make_fp16_nt_hhs_scale_bias_sweep.py`, leaving `StorePriorityOpt=True`.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_probe` generated, validated, and hot-loop retimed at median `40483.9 GFLOP/s`, mean `40432.8`, best `40556.7`. This remained the best within the forced-WGM8 subgroup after NEPBS12/24/32, NEPBS4, no-activation, StoreSyncOpt, and GroupLoadStore probes, but was later superseded by the guarded SIA3/no-store-priority config.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs12_probe` generated, validated, and hot-loop retimed at median `40344.4 GFLOP/s`, mean `40378.0`, best `40601.3`. Reject because it is slower than both NEPBS8 and NEPBS16.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs24_probe` generated, validated, and hot-loop retimed at median `37142.7 GFLOP/s`, mean `37204.6`, best `37649.8`. Reject because it regresses to the old forced-WGM8 baseline band.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs32_probe` generated, validated, and hot-loop retimed at median `37119.5 GFLOP/s`, mean `37130.4`, best `37523.6`. Reject because it also regresses to the old forced-WGM8 baseline band.
- `NEPBS8` and `NEPBS12` both compile with `.vgpr_count: 220`. `NEPBS24` rises to `.vgpr_count: 252`. Added one final smaller-batch config probe, `NEPBS4`, because intermediate `5-7` values would round down to `4` on this `MIWaveTile[0]=4` path.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs4_probe` generated, validated, and hot-loop retimed at median `40407.6 GFLOP/s`, mean `40395.5`, best `40505.4`. Reject because it is slower than the forced-WGM8 NEPBS8 baseline.
- Forced-WGM8 `NEPBS8` assembly still emits activation dispatch and per-element `s_swappc` calls because the problem type is `Activation: True`, `ActivationType: hipblaslt_all`, even though standalone benchmarking uses `ActivationArgs: [Enum: none]`. Added config-only `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_noact_probe` with `Activation: False`, `ActivationType: none` to test whether a no-activation specialization is valid and faster for this target.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_noact_probe` generated, validated, and hot-loop retimed at median `40340.0 GFLOP/s`, mean `40362.9`, best `40504.9`. Reject because it is slower than the `Activation: True` forced-WGM8 NEPBS8 baseline despite removing activation dispatch. Keep the wrapper-compatible activation-capable path.
- Store-vector/remap are not viable low-risk probes here: `SourceSwap=True` and `VectorWidthA=1` constrain `StoreVectorWidth` to `1`, while `StoreRemapVectorWidth>0` rejects with `VectorWidthB>1`/GSU constraints. Added config-only `NEPBS8` probes for `StoreSyncOpt=1`, `StoreSyncOpt=4`, and `GroupLoadStore=True`.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_storesync1_probe` generated, validated, and hot-loop retimed at median `40388.1 GFLOP/s`, mean `40430.0`, best `40667.6`, derived median `27.224 ms`. Reject because it is below the forced-WGM8 NEPBS8 baseline median `40483.9 GFLOP/s`.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_storesync4_probe` generated, validated, and hot-loop retimed at median `40417.2 GFLOP/s`, mean `40390.4`, best `40567.8`, derived median `27.204 ms`. Reject because it is below the forced-WGM8 NEPBS8 baseline median `40483.9 GFLOP/s`.
- `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_forced_nepbs8_grouploadstore_probe` generated, validated, and hot-loop retimed at median `40425.1 GFLOP/s`, mean `40440.8`, best `40670.0`, derived median `27.199 ms`. Reject because it is below the forced-WGM8 NEPBS8 baseline median `40483.9 GFLOP/s`.

### No-ForceStaticWGM8 Tuning Sweeps

- Removed `ForceStaticWGM8` from all public solution registries. All configs below are guarded static-WGM8 with runtime WGM/divisibility fallback.
- Base sweep over `NumElementsPerBatchStore` values (1-20) on the `SIA2/PGR1/PLR1/VWB2/1LDSBuffer` rocBLAS-like schedule found `NEPBS10` as the best repeatable SIA2 config (median `38292.3 GFLOP/s`), but run-to-run noise was high.
- Probed `StoreSyncOpt`, `GroupLoadStore`, `StorePriorityOpt`, no-activation, stagger, and PGR/WGM/XCC variants around NEPBS10. None improved on the SIA2 baseline substantially.
- `ScheduleIterAlg=3` (SIA3) sweep on NEPBS10 was the breakthrough: first pass showed `NEPBS10+SIA3` median `40014 GFLOP/s` and `NEPBS10+SIA3+nostoreprio` median `39990 GFLOP/s`, both close to the historical forced-WGM8 best. Second pass: `SIA3+nostoreprio` at `44536`, `SIA3` at `42241`.
- Follow-up NEPBS sweep under SIA3 (values 8/10/12/14/16) plus store/sync/stagger/WGMXCC combos: NEPBS10+SIA3+nostoreprio was the clear winner.
- Final confirmation on the winner (`NEPBS10 + SIA3 + StorePriorityOpt=False`): three alternating head-to-head passes (30 total hot-loop samples), aggregated median `45252.9 GFLOP/s`, mean `45491.6 GFLOP/s`. SIA3 baseline (store priority on) aggregated median `40968.9 GFLOP/s`.
- The winner uses `vgpr=219`, `sgpr=74`, `LDS=8192`, `PGR1/PLR1/VWB2/1LDSBuffer/WGM8`, no `ForceStaticWGM8`, with correct `WGMStaticFallback`/`WGMPositive` guard labels in assembly.
- The combined SIA3/no-store-priority change was the largest gain: approximately 18% over the selected SIA2 NEPBS10 baseline. SIA3 alone was about 7% over that baseline, and `StorePriorityOpt=False` contributed an additional ~10% on top of SIA3.

### VWB2 Correctness Fixes

- rocBLAS uses `VW1_VWB2` in the selected NT kernel. Early TensileLite VWB2 probes either failed validation or failed generation, but the lane-mapping issue is now fixed for the tested WMMA_V1 half/BF16 path.
- Pre-fix pure-HHS `hhs_mt128x128_tlds0_rocblas_vwb2_probe` generated and ran but failed validation, matching the known wrong-but-fast NT `VectorWidth >= 2` class.
- Pre-fix scale/bias `mt128x128_tlds0_rocblas_1ldsb_vwb2_probe` first hit `ImportError: cannot import name 'SAddU64' from rocisa._rocisa.instruction` in `Components/GL2Prefetch.py`. After the GL2 compatibility patch it generated but failed validation with a first-run sample of `126` incorrect values out of `128` in tensor `d`.
- `GL2Prefetch.py` local compatibility patch: replace unavailable `SAddU64` and `VAddNCU64` pseudo-instruction use with explicit `SAddU32` + `SAddCU32` and `VAddCOU32` + `VAddCCOU32`. Guard optional `GlobalPrefetchB8` import and `GLOBALModifiers` construction.
- `KernelWriterAssembly.py` VWB2 lane fix: for WMMA_V1 half/BF16 A/B, non-converting `lrvwTile > 1`, non-transposed local reads, force `ds_load_u16` instead of `ds_load_b32`. This matches full Tensile's paired `_ds_load_u16` / `_ds_load_u16_d16_hi` input layout.
- `LocalRead.py` offset fix: clamp sub-dword WMMA half/BF16 local-read `numElementPerRead` to at least one K element when `lrvwTile > 1`. Without this, B K-lane offsets collapsed to `0`/`2`/`128`/`130`.
- `KernelWriter.py` read-count fix: for the same sub-dword path, do not divide emitted d16 local-read count by `VectorWidthA/B`. Generated comments now report `readsPerIterA=64`, `readsPerIterB=64`, matching full Tensile and the emitted read stream.
- Correctness status: pure-HHS `hhs_mt128x128_tlds0_rocblas_vwb2_probe` and scale/bias `mt128x128_tlds0_rocblas_1ldsb_vwb2_probe` pass generated-client validation after these patches.
- Guardrail: do not broadly enable `VectorWidthA/B >= 2` for NT `Ailk_Bjlk`. The validated predicate is the tested WMMA_V1 half/BF16 non-converting `lrvwTile > 1` path.

### VWB2 Performance Findings

- Correctness-only patched pure-HHS VWB2 hot-loop CSV: `tmp_tensile_fp16_nt/outputs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_probe_8192_hot_loop.csv`. Median moved around `31.1 TFLOP/s` before allocation compaction.
- Correctness-only patched scale/bias VWB2 hot-loop CSV: `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_probe_8192_hot_loop.csv`. Median moved around `30.5-30.6 TFLOP/s` before allocation compaction.
- `KernelWriter.py` PLR buffer compaction: for one-iteration WMMA_V1 half/BF16 loops where `PrefetchLocalRead >= LoopIters` and PLR modulo already collapses to zero, cap `numVgprBuffer` to `1`. Pure-HHS VWB2 improved to median `33158.2 GFLOP/s`, scale/bias VWB2 improved to median `32850.0 GFLOP/s`.
- `KernelWriter.py` allocation-order experiment: for the same one-iteration WMMA_V1 half/BF16 non-converting `PGR1/PLR1` path, re-place A/B operands immediately after C and assign local-write/global-read-offset/G2L/local-read address VGPRs after A/B, matching full Tensile's main-loop VGPR layout.
- Patched allocation-order pure-HHS VWB2 result: `.sgpr_count: 74`, `.vgpr_count: 255`, `vgprValuA=130`, `vgprValuB=163`, `vgprG2LA=200`, `vgprG2LB=208`, `vgprSerial=218`, CSV `tmp_tensile_fp16_nt/outputs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_probe_8192_hot_loop.csv`, median `33551.2 GFLOP/s`, mean `33502.5 GFLOP/s`, best `33775.1 GFLOP/s`, derived median `32.771 ms`.
- Patched allocation-order scale/bias VWB2 result: `.sgpr_count: 74`, `.vgpr_count: 256`, same main-loop VGPR placement, CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_probe_8192_hot_loop.csv`, median `33157.6 GFLOP/s`, mean `33192.9 GFLOP/s`, best `33656.4 GFLOP/s`, derived median `33.160 ms`.
- Same-session re-baseline kept existing generated artifacts and added separate override CSVs so prior results remain intact. Pure-HHS VWB2 CSV `tmp_tensile_fp16_nt/outputs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_probe_8192_rebaseline_hot_loop.csv`: `10` rows, median `33444.6 GFLOP/s`, mean `33389.4 GFLOP/s`, best `33824.3 GFLOP/s`, derived median `32.876 ms`. Scale/bias VWB2 CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_probe_8192_rebaseline_hot_loop.csv`: `10` rows, median `32016.2 GFLOP/s`, mean `32013.5 GFLOP/s`, best `32344.2 GFLOP/s`, derived median `34.342 ms`. Then-current-best `mt128x128_tlds0_pgr1` scale/bias CSV `tmp_tensile_fp16_nt_hhs/outputs/hhs_nt_scale_bias_mt128x128_tlds0_pgr1_8192_rebaseline_hot_loop.csv`: `10` rows, median `33652.1 GFLOP/s`, mean `33542.5 GFLOP/s`, best `33798.8 GFLOP/s`, derived median `32.673 ms`.
- Re-baseline interpretation: existing VWB2 artifacts still validate and have the expected patched assembly shape (`ds_load_u16`/`ds_load_u16_d16_hi`, `.sgpr_count: 74`, compact `vgprSerial=218`). Before static-WGM, scale/bias VWB2 was not competitive with `mt128x128_tlds0_pgr1`. Guarded static-WGM plus later SIA3/no-store-priority tuning ultimately made this path the winner.

### Prior Gap Isolation

- Purpose: this is historical evidence from before static-WGM/forced-WGM8. It explains why we stopped looking at main-loop opcodes, L2/LDS behavior, and broad launch/ABI rewrites.
- Full-Tensile no-beta VWB2 control used `HHS_H`, `UseBeta: False`, `readsPerIterA/B=64`, `.sgpr_count: 64`, `.vgpr_count: 256`, LDS `8192`. Hot-loop CSV `tmp_tensile_fp16_nt/outputs_full_tensile/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_probe_8192_hot_loop.csv`: median `35233.2 GFLOP/s`, mean `35247.5`, best `35576.9`, derived median `31.207 ms`.
- Normalized steady-loop opcode stream was effectively matched: first unroll-half count `187`, including `4` `buffer_load_b128`, `4` `ds_store_b128`, `64` `ds_load_u16`, `64` `ds_load_u16_d16_hi`, `16` WMMA, and `2` barriers. The only spelling difference was TL `s_waitcnt` versus full `s_waitcnt_lgkmcnt`, targeting the same wait class here.
- Static object check showed identical main-loop size (`0xa9c` bytes), but much larger TL non-loop code: TL entry/open/end `0x2800/0x3c08/0xb368`. Full entry/open/end `0x1b00/0x20d4/0x7094`. Conclusion: the old gap was prologue/alternate-path/ABI state, not the compiled main loop.
- Resource/prologue contrast worth remembering: TL kept `.sgpr_count: 74`, UserArgs/general-batch routing, dynamic GSU setup, and runtime workgroup/WGM quotient-remainder code. Full no-beta was `.sgpr_count: 64` with more compact precomputed fields. rocprof rounded both to `vgpr_count=256`, `sgpr_count=128`, LDS `8192`, scratch `0`, so this was not an occupancy-level difference in that profile.
- rocprof counter DBs: TL `tmp_tensile_fp16_nt/rocprof_tl_hhs_vwb2_rebaseline/tl_hhs_vwb2_results.db`. Full `tmp_tensile_fp16_nt/rocprof_full_hhs_vwb2_rebaseline/full_hhs_vwb2_results.db`. Counters over 12 GEMM dispatches showed similar L2 hit rate (`18.99%` TL vs `18.50%` full) and `LDSBankConflict=0`. TL `VALUInsts` median `9224`, full `9140`.
- Launch mapping observation: TL dispatch was flattened `(grid_x=524288, grid_y=1)`, full used `(8192,64)`, both with `4096` workgroups. `ContractionSolution::generateSingleCall` deliberately flattens when `internalArgsSupport.version >= 1`. Assembly reconstructs `WorkGroup0/1/2`. Direct 2D launch remains a broad runtime/codegen ABI refactor, not a local tuning patch.
- Ruled-out mapping probes from this investigation are summarized below: scalar `NumWorkGroups0/1` regressed to `31909.6 GFLOP/s`, WGMXCC1 regressed to `31691.3`, and WGM1 regressed to `14544.3`.
- Static-WGM followed from this isolation and is now summarized in the artifact section and experiment log. Important implementation note: the first valid static-WGM compile crashed because a temp SGPR overwrite lost the original WGM block id before `blockId * WGM` was added back to `WorkGroup1`. Preserving the original block id fixed validation.

### Ruled-Out Low-Risk Controls

- `MIArchVgpr: 0`: generated and validated, but the kernel still had `MIAV1` in the name and unchanged `.sgpr_count: 74`, `.vgpr_count: 255`, `vgprValuA=136`, `vgprValuB=201`, `vgprG2LA=234`, `vgprG2LB=242`, `vgprSerial=250` before the later allocation fixes.
- `LocalWriteUseSgprA/B`: rejected before generation as invalid TensileLite fork parameters. Log path `tmp_tensile_fp16_nt/logs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_lwsgpr_probe_8192.log`.
- `SupportUserArgs: False`: generated as `HHS_H` and validated, but kept the same general/multigemm prologue and `.sgpr_count: 74`. Hot-loop CSV `tmp_tensile_fp16_nt/outputs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_no_userargs_probe_8192_hot_loop.csv`, median `33372.1 GFLOP/s`.
- `Batched: False`: after switching to 3-index `ProblemSizes: [8192, 8192, 8192]`, generation reached `Cij_Aik_Bjk_HHS_H_UserArgs` but failed in `KernelWriterAssembly.py::computeLoadSrd` with `UnboundLocalError: cannot access local variable 'stridedBatchedGemmLoad_End'`. Log path `tmp_tensile_fp16_nt/logs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_nonbatched_probe_8192.log`.
- `GlobalSplitU: 0`: rejected before assembly generation with `reject: Either GSU or StreamK must be enabled`, then `0 valid solutions`. Log path `tmp_tensile_fp16_nt/logs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_gsu0_probe_8192.log`.
- `PreloadKernArgs: True`: generated and validated but forced back to `PKA0` on gfx1151 and left resources/prologue unchanged.
- `GlobalSplitUAlgorithm: SingleBuffer`: generated and validated but worsened resources to `.sgpr_count: 76`, `.vgpr_count: 255`. Hot-loop CSV `tmp_tensile_fp16_nt/outputs/hhs_nt_hhs_mt128x128_tlds0_rocblas_vwb2_singlebuffer_probe_8192_hot_loop.csv`, median `32445.7 GFLOP/s`.
- Scalar `NumWorkGroups0/1` calculation: source patch generated and validated, but pure-HHS VWB2 hot-loop median regressed to `31909.6 GFLOP/s`. Skip scale/bias follow-up and keep the original vector ceil-divide path.
- Explicit WGMXCC1 config: pure-HHS generated and validated, but did not remove the WGMXCC prologue and regressed to median `31691.3 GFLOP/s`. Skip scale/bias follow-up.
- `WorkGroupMapping: 1`: pure-HHS generated and validated, but collapsed to median `14544.3 GFLOP/s`. WGM8 cache/traversal behavior is required for this shape.
- Static-GSU/UserArgs removal is not a narrow patch: runtime GSU is threaded through fixed SGPR allocation, common argument packing/loading, `GSUOn.graWorkGroup`, `computeLoadSrd`, global-read increments, store/workspace paths, and serialized `InternalSupportParams[SupportUserGSU]`. Benchmark YAML rejects `InternalSupportParams` except for custom kernels.

### Historical PLR/VGPR Data

- rocBLAS-like VWB1 scale/bias variants `mt128x128_tlds0_rocblas_sched_vwb1` and `mt128x128_tlds0_rocblas_1ldsb_vwb1` failed assembly generation with `total vgpr: 283 not in [0, 256]`.
- `mt128x128_tlds0_plr1` failed at `266` VGPR. `mt128x128_tlds0_pgr1_plr1_sia3` and `mt128x128_tlds0_pgr1_sia2` failed at `283` VGPR.
- Pure-HHS controls show the PLR cliff is not caused by the scale/bias epilogue: `hhs_mt128x128_tlds0_plr1`, `hhs_mt128x128_tlds0_pgr1_plr1_sia3`, and `hhs_mt128x128_tlds0_rocblas_sched_vwb1` also failed at `283` VGPR.
- Temporary diagnostic for `hhs_mt128x128_tlds0_plr1`: `total=283`, `cValu=128`, `A valu=136+64`, `B valu=201+64`, `A g2l=266+8`, `B g2l=274+8`, `serial=282`.
- Interpretation update: VWB1 PLR1 overflow remains useful history, but rocBLAS's exact selected kernel uses `VW1_VWB2`, reducing the B operand footprint. The current actionable path is the fixed VWB2 schedule, not broad VWB1 PLR1 repair.

### Source Patch Assessment

- Keep locally: VWB2 lane-selection, sub-dword local-read offset clamp, read-count accounting, PLR buffer compaction, allocation-order changes, and the WGM8-only guarded static-WGM fast path. The legacy opt-in `ForceStaticWGM8` source knob has been removed because it was not shape-generic without the divisibility proof in `Deriving Static WGM8`.
- Static-WGM predicate tightening: the fast path was narrowed from all power-of-two positive `WorkGroupMapping` values to only `WorkGroupMapping == 8`, matching the rocBLAS-like path actually validated here. Regenerate and revalidate the static-WGM probes after this source change before relying on old artifacts.
- Key CSVs for static-WGM/forced-WGM8 are listed in the artifact section and experiment log above.
- Tighten before upstreaming: the sub-dword local-read clamp and `numVgprBuffer` cap predicates should be narrowed and covered by targeted correctness tests.
- Keep locally for compatibility: `GL2Prefetch.py` import/instruction fallback. `PrefetchGL2>0` remains lightly validated because the exercised target path used `PrefetchGL2=0`.

### Deriving Static WGM8

- `WorkGroupMapping=8` remains a solution/tuning choice. Removing the runtime WGM fallback should be derived from correctness predicates, not exposed as a performance knob.
- The unguarded static-WGM8 path is safe only when both fallback checks are statically impossible: the effective runtime WGM is guaranteed to be `8`, and `NumWorkGroups1 % 8 == 0` for the selected problem.
- For the current target, the proof is exact: `MacroTile1=128`, free-1 size `8192`, so `NumWorkGroups1=ceil(8192 / 128)=64`, and `64 % 8 == 0`. The benchmark also does not override runtime WGM, and the tested path does not use SFC, cluster, or StreamK mapping semantics.
- `ForceStaticWGM8=True` is therefore not a shape-generic optimization. It deliberately omits the guarded path's `NumWorkGroups1 % 8 == 0` fallback check. Using the generated object on non-divisible shapes can produce zero output tiles or illegal memory accesses.
- For this `MT128x128` NT HHS artifact, the practical aligned-shape rule is `ceil(N / 128) % 8 == 0`. When `N` is a multiple of `128`, this reduces to `N % 1024 == 0`. Passing `1024^3` does not imply all larger shapes are valid: `1536^3` fails, while `2048^3` and `8192^3` pass.
- Direct validation evidence for the current artifact: `256^3` produced zero tiles, `512^3` faulted, `768^3` produced zero tiles, `1536^3` failed validation. `1024^3`, `2048^3`, and `8192^3` passed full-output validation with the transposed HIP reference.
- Do not derive unguarded static-WGM8 from `WorkGroupMapping == 8` alone. Generic kernels may see runtime problem sizes where `ceil(freeSize1 / MacroTile1)` is not divisible by `8`, and runtime `wgm` overrides can change the effective WGM away from the logic value.
- Implemented conservative first step: public `ForceStaticWGM8` is removed from default, valid, and required solution parameters, and codegen always emits the guarded static-WGM8 path.
- To regain the old unguarded fast path safely, add a new internal specialization only after host-side solution selection proves the same grid calculation used by `ContractionSolution::calculateGrid`: no SFC, no cluster remap, no StreamK remap, effective WGM `8`, and `ceil(problem.freeSizeB-packed / macroTile.y) % 8 == 0` after `transposeC01`/packed-batch handling.
- If that proof is not available, emit the guarded static-WGM8 fast path. The guarded path still removes the expensive dynamic WGM divide on valid WGM8/divisible shapes, while preserving the dynamic fallback for other shapes.
- If implementing this for library logic, add a selection predicate rather than a YAML tuning axis. Candidate forms are a host-evaluated `NumWorkGroups1MultipleOf: 8` predicate or an exact-shape predicate for generated standalone kernels.

### Historical screening data

- Best old cool-loop TLDS2 scale/bias neighborhood: `mt128x128_tlds2_pad128x8_nostoreprio`, median `28761.0 GFLOP/s`, mean `28657.2 GFLOP/s`.
- `mt128x128_tlds2_pad128x8`: `.amdhsa_next_free_vgpr 244`, `.amdhsa_next_free_sgpr 72`, `.amdhsa_group_segment_fixed_size 25600`, vector `ds_load_b128` local reads.
- `mt128x128_tlds0_pgr2`: `.amdhsa_next_free_vgpr 256`, `.amdhsa_next_free_sgpr 72`, `.amdhsa_group_segment_fixed_size 16384`, direct sub-dword local reads.
- `mt128x128_tlds0`: `.amdhsa_next_free_vgpr 256`, `.amdhsa_next_free_sgpr 64`, `.amdhsa_group_segment_fixed_size 8192`, direct sub-dword local reads.
- Depth-64 TN/NN-inspired NT attempts did not close the gap. The fastest TLDS0 rectangular attempts failed validation. Valid TLDS2/TLDS0 variants were much slower than `MT128x128x16`.

## Issue #5314 Guardrails

- The linked issue's controlled test used pure HHS GEMM with `TransposeA=N`, `TransposeB=T`, `UseBias=0`, HPA.
- It warned that wrong-but-fast NT tiles use `VectorWidth >= 2` and can mis-lane WMMA inputs with relative error around `1.4`.
- NT `Ailk_Bjlk` has `TLUA=TLUB=True`, which blocks some TensileLite LDS-transpose paths. This makes the gap a TensileLite NT WMMA/codegen limitation, not just epilogue overhead.
- Keep vector scale in scope: prior gfx1151 TN/NN wins were tuned with scale-alpha-vector support, and the FeatherOps wrapper uses that path.

## Next Steps

- Keep `mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe` as the current guarded/WGM-safe best. Aggregated median `45252.9 GFLOP/s` beats the historical forced-WGM8 best by ~12%.
- The winning knobs: `ScheduleIterAlg=3`, `StorePriorityOpt=False`, `NumElementsPerBatchStore=10`, with `PGR1/PLR1/VWB2/1LDSBuffer/WGM8`. Keep activation-capable `Activation: True`/`ActivationType: hipblaslt_all`.
- The combined SIA3/no-store-priority change was the biggest config-only gain found (~18% over the selected SIA2 NEPBS10 baseline). Inspect SIA3 vs SIA2 and store-priority-on vs off assembly before more config probes.
- `StorePriorityOpt=False` on the SIA3 kernel contributed an additional ~10% - opposite to the SIA2 forced-WGM8 experience where `StorePriorityOpt=True` was better. Do not blindly carry store-priority assumptions across schedule algorithms.
- Stop config-only epilogue tuning unless a new targeted knob is identified from assembly. Inspect the guarded SIA3/no-store-priority winner's scale/bias assembly for a narrow source change only if the risk/reward is clear.
- Add a host-side selection predicate before reintroducing an unguarded static-WGM8 specialization for promoted libraries. Keep guarded static-WGM8 whenever the WGM/divisibility proof is unavailable.
- If epilogue tuning is exhausted, inspect remaining prologue for a narrow opt-in static-shape cleanup that does not remove UserArgs/GSU ABI support globally. Avoid WGMXCC skip because it already regressed.
- Do not make speculative source patches for dynamic GSU/UserArgs removal or direct 2D launch. Those require a deliberate ABI/codegen refactor with tests.
- For any new codegen patch, validate pure-HHS VWB2 first, then validate scale/bias VWB2, then retime with hot-loop timing and update this document before moving on.

## Key Files

- HHS scale/bias generator: `tmp_tensile_fp16_nt_hhs/scripts/make_fp16_nt_hhs_scale_bias_sweep.py`.
- HHS scale/bias runner: `tmp_tensile_fp16_nt_hhs/scripts/run_fp16_nt_hhs_variant.sh`.
- PyTorch rocBLAS benchmark: `tmp_tensile_fp16_nt_hhs/scripts/benchmark_pytorch_rocblas_nt.py`.
- Pure-HHS/HHH generator: `tmp_tensile_fp16_nt/scripts/make_fp16_nt_speed_sweep.py`.
- Pure-HHS/HHH runner: `tmp_tensile_fp16_nt/scripts/run_fp16_nt_variant.sh`.
- TensileLite source to patch: `~/rocm-libraries/projects/hipblaslt/tensilelite`.
- Full Tensile source reference: `~/rocm-libraries/shared/tensile`.
- rocBLAS logic reference: `~/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/strixhalo/strixhalo_Cijk_Ailk_Bjlk_HHS_BH.yaml`.
- FeatherOps wrapper benchmark: `benchmark_mm_hipblaslt_fp16.py`.
- FeatherOps hipBLASLt wrapper: `kernel/hip/hipblaslt_kernel_fp16.cu`.
- gfx1151 profiling reference: `doc/gfx1151_reference.md`.
