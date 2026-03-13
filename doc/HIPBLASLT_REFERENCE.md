# hipBLASLt FP16 Matmul Wrapper Reference

## Goal

Make the ComfyUI FeatherOps hipBLASLt fp16 benchmark hit the same gfx1151 kernel path as the standalone benchmark in `~/rocm-libraries/hipblaslt_fp16_bench`.

## Files changed so far

- `kernel/hip/hipblaslt_kernel_fp16.cu`
- `kernel/hip/hipblaslt_kernel_fp16.py`
- `benchmark_mm_hipblaslt_fp16.py`

## Reference files

- Standalone binary: `~/rocm-libraries/hipblaslt_fp16_bench`
- Standalone source: `~/rocm-libraries/hipblaslt_fp16_bench.cpp`
- Existing FeatherOps HIP wrapper pattern: `~/ComfyUI-FeatherOps/kernel/hip/hip_kernel_fp16.cu`
- Existing FeatherOps HIP Python wrapper pattern: `~/ComfyUI-FeatherOps/kernel/hip/hip_kernel_fp16.py`

## What works now

- The FeatherOps hipBLASLt extension builds and loads.
- The Python wrapper works numerically for the supported fp16 path.
- The benchmark script exists and can exercise both the semantic wrapper path and a column-major fast-layout path.
- The extension build cache was deleted after the last unsafe experiment, so the next import should rebuild from source.

## Current findings

### Standalone benchmark

Command:

```bash
HIPBLASLT_LOG_MASK=96 ~/rocm-libraries/hipblaslt_fp16_bench
```

Observed:

- Picks `solution_index 1112`
- Kernel name starts with:

```text
Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x96x64_...
```

- Performance is about `38 TFLOP/s` average / `38.4+ TFLOP/s` best on `8192^3`

### FeatherOps/Torch extension path

Current wrapper paths still pick `solution_index 1002`, not `1112`, even after:

- switching benchmark tensors to column-major layout
- matching the standalone `setProblem(M, N, K, ...)` path
- using a separate internal `C` buffer instead of `C == D`
- preferring installed hipBLASLt headers before repo headers

Observed command from logging:

```text
hipblaslt-bench --api_method cpp ... --solution_index 1002 --activation_type relu
```

Measured performance from the extension path is still about `26-28 TFLOP/s`.

## Current findings (resolved)

The initial discrepancy between the standalone C++ benchmark (hitting ~38 TFLOP/s with `solution_index 1112`) and the Torch extension (hitting ~27 TFLOP/s with `solution_index 1002`) was traced down to an **ABI mismatch**.

### The root cause

The PyTorch extension (`hipblaslt_kernel_fp16.cu`) was being compiled with custom headers included from the local `~/rocm-libraries/projects/hipblaslt/library/include` directory. However, at runtime, the Python environment (via `rocm_sdk`) was dynamically linking against the installed system library `libhipblaslt.so.1` (from `_rocm_sdk_devel`).

These two versions of the headers had diverging definitions for `hipblaslt_ext::GemmPreference` and `hipblaslt_ext::GemmInputs`. Because of this struct layout misalignment:
- When the Python C++ wrapper called `setMaxWorkspaceBytes`, it wrote to the incorrect memory offsets.
- Consequently, the system `hipBLASLt` library read garbage workspace sizes, deemed the `1112` configuration unsupported due to "insufficient workspace", and fell back to the slower `1002` heuristic.

### The fix

The custom include path (`-I$HOME/rocm-libraries/...`) was removed from the compilation flags in `kernel/hip/hipblaslt_kernel_fp16.py`. The extension is now compiled strictly against the same system headers that match the dynamically linked system library (`_rocm_sdk_devel`).

## Verification

After wiping the build cache and forcing a clean recompile, the fix was verified:

1. **Python Benchmark**: Running `benchmark_mm_hipblaslt_fp16.py` now correctly hits **~36.7 TFLOPS** on the 8192^3 problem size using the `hipblaslt_fast_layout` path, closely matching the standalone C++ benchmark.
2. **Kernel Profiling**: Running a minimal fast layout script under `rocprofv3 --kernel-trace` empirically confirmed that the Python extension successfully dispatches the target optimal kernel:
   ```text
   Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x96x64_MI16x16x1_SN_LDSB1...
   ```

## Current source status notes

- `kernel/hip/hipblaslt_kernel_fp16.cu` contains:
  - the semantic row-major wrapper
  - a column-major fast-layout wrapper
  - a `benchmark_raw_buffers` path (added during debugging to isolate Torch memory allocators)
  - safe `getAllAlgos`-based forced-solution fallback logic
- `kernel/hip/hipblaslt_kernel_fp16.py` exposes:
  - `mm_hipblaslt_fp16(...)`
  - `mm_hipblaslt_fp16_colmajor(...)`
  - `to_col_major(...)`
  - compilation flags pointing *only* to system headers.
- `benchmark_mm_hipblaslt_fp16.py` successfully benchmarks the fast layout via `mm_hipblaslt_fp16_colmajor` using `solution_index=1112`.
