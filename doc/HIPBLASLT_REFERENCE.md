# hipBLASLt FP16 Reference

## Purpose

- Keep the verified facts needed to identify and reproduce the gfx1151 hipBLASLt FP16 kernel used for `M=N=K=8192`.
- This doc is about the native hipBLASLt kernel path, not the custom HIP kernel.

## Current Verified Status

- Local native benchmark:
  - `cpp_benchmarks/build_benchmark_mm_hipblaslt_fp16.sh`
  - `cpp_benchmarks/benchmark_mm_hipblaslt_fp16.cpp`
- Logging command:

```bash
HIPBLASLT_LOG_MASK=96 ./cpp_benchmarks/benchmark_mm_hipblaslt_fp16 --warmup 0 --iters 1 --workspace-mb 64
```

- For `8192^3`, hipBLASLt selects public `solution_index 1112`.
- Important log line:

```text
hipblaslt-bench --api_method cpp ... --solution_index 1112 --activation_type relu ...
```

- Important solution prefix:

```text
Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x96x64_MI16x16x1_...
```

- Expected steady-state performance from earlier full benchmark runs:
  - native C++ path: about `37-38 TFLOPS`
  - FeatherOps fast-layout Python wrapper: about `36-37 TFLOPS`
- Single-iteration `rocprofv3` runs are useful for kernel metadata only. Their timing is not representative.

## Source Of Truth

- Tensile logic entry:

```text
~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Ailk_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml
```

- Match this kernel by its long solution or kernel name prefix, not by Tensile `SolutionIndex`.
- Important nuance:
  - runtime public algo index: `1112`
  - matching Tensile YAML entry: `SolutionIndex: 71`
- The reliable join is the kernel or solution name string, not the integer index.

## Exact Kernel Facts

- Matching gfx1151 Tensile entry facts:
  - `MacroTile0 = 128`
  - `MacroTile1 = 96`
  - `DepthU = 64`
  - `MatrixInstruction = [16, 16, 16, 1]`
  - `WavefrontSize = 32`
  - `NumThreads = 128`
  - `WorkGroup = [64, 2, 1]`
  - `MIWaveGroup = [4, 1]`
  - `MIWaveTile = [2, 6]`
  - `ThreadTile0 = 16`
  - `ThreadTile1 = 6`

- Equivalent custom-kernel tuple:
  - `warps_m = 4`
  - `warps_n = 1`
  - `unroll_k = 4`
  - `repeat_m = 2`
  - `repeat_n = 6`

- Why this mapping is correct:
  - `MacroTile0 = 16 * 4 * 2 = 128`
  - `MacroTile1 = 16 * 1 * 6 = 96`
  - `DepthU = 16 * 4 = 64`

- Memory and scheduling properties:
  - `1LDSBuffer = 1`
  - `LdsNumBytes = 30336`
  - `LdsPadA = 8`
  - `LdsPadB = 8`
  - `GlobalReadVectorWidthA = 8`
  - `GlobalReadVectorWidthB = 8`
  - `LocalReadVectorWidth = 16`
  - `NumLoadsA = 8`
  - `NumLoadsB = 6`
  - `PrefetchGlobalRead = 2`
  - `PrefetchLocalRead = 0`
  - `ScheduleIterAlg = 3`
  - `ScheduleGlobalRead = 1`
  - `ScheduleLocalWrite = 1`
  - `TransposeLDS = 1`
  - `UnrollMajorLDSB = true`
  - `SourceSwap = true`
  - `StaggerU = 32`
  - `WorkGroupMapping = 8`
  - `DirectToLdsA/B = false`
  - `DirectToVgprA/B = false`

- rocprof kernel metadata from a single-iteration trace:
  - `vgpr_count = 256`
  - `sgpr_count = 128`
  - `lds_size = 30336`
  - `workgroup_x = 128`

## Important Interpretation

- This is a software-pipelined single-LDS-buffer kernel.
- It does not rely on direct global-to-LDS or direct-to-VGPR paths.
- The key comparison points versus our current custom HIP winner are:
  - deeper K pipeline: `64` instead of `32`
  - fewer waves per block: `4` instead of `8`
  - narrower N tile: `128x96` instead of `128x256`
  - higher register pressure: about `256 VGPR` instead of about `192 VGPR`

## Historical Pitfall That Still Matters

- Old symptom:
  - native or standalone hipBLASLt path picked `1112`
  - Torch extension path picked `1002`
  - extension performance stayed around `26-28 TFLOPS`

- Root cause:
  - the extension was compiled against local `rocm-libraries` hipBLASLt headers
  - runtime linked against the installed system `libhipblaslt.so`
  - `hipblaslt_ext::GemmPreference` and `hipblaslt_ext::GemmInputs` layouts did not match
  - `setMaxWorkspaceBytes` wrote garbage offsets
  - hipBLASLt rejected the fast solution and fell back

- Fix:
  - build the extension only against the installed system headers that match the loaded system library

- Result:
  - the FeatherOps fast-layout wrapper can dispatch the target kernel and recover the expected `~36-37 TFLOPS`

- Important note:
  - current logs show the chosen solution uses `0 MiB` workspace at runtime
  - the workspace preference still had to be ABI-correct, because the library used that preference when deciding whether the fast solution was supported

## Relevant Local Files

- `cpp_benchmarks/build_benchmark_mm_hipblaslt_fp16.sh`
- `cpp_benchmarks/benchmark_mm_hipblaslt_fp16.cpp`
- `kernel/hip/hipblaslt_kernel_fp16.cu`
- `kernel/hip/hipblaslt_kernel_fp16.py`
- `benchmark_mm_hipblaslt_fp16.py`
