# HIP FP16 Matmul Optimization Plan (gfx1151)

## Scope and Metric

- Target kernel path: FP16 (`a`=fp16, `b`=fp16) on gfx1151.
- Prepack contract:
  - `b` is pre-transposed and swizzled into `[K/16, 2, N, 8]` layout.
  - The C++ kernel consumes this prepacked storage to perform direct loads into LDS without additional transposition.
- Accuracy gate: `relative L2 <= 0.01`, `max abs <= 1.0` (some slight relaxation for extreme N=8192 accumulations).
- Performance metric: `benchmark_mm_hip_fp16.py` GFLOPS.

## Current Baseline Snapshot

- Correctness:
  - `python test_mm_hip_fp16.py` -> `28/28` pass (fixed LDS Swizzle and `__syncthreads` missing barrier).
- Latest full benchmark (`python benchmark_mm_hip_fp16.py`):
  - `N=4096`: `~32.1k` GFLOPS
  - `N=8192`: `~26.7k` GFLOPS

## Hardware Profiling Insights

### The FP16 vs FP8 Gap

We have established a highly optimized FP8 kernel (`scaled_mm_hip_prepacked`) that achieves ~44.7 TFLOPS. The primary difference between our optimized FP8 kernel and this baseline FP16 kernel is the memory footprint and LDS layout:
1. **LDS Footprint:** FP16 data is 2 bytes per element, whereas FP8 is 1 byte. To load an equivalent `128x256` chunk, the FP16 kernel requires exactly twice as much LDS memory (`32 KB` vs `16 KB`).
2. **VGPR Footprint:** Loading and holding FP16 data requires twice as many vector registers (`uint4` loads 8 `fp16` elements instead of 16 `fp8` elements).
3. **LDS Swizzle (Bank Conflicts):** Our baseline FP16 kernel now successfully maps logical indices to physical indices to avoid bank conflicts using the `[K/16, 2, N, 8]` layout.
4. **Latency Hiding Limitation:** Because FP16 uses double the LDS memory, large block sizes like `(1, 8, 4, 8, 2)` (which corresponds to `128x256`) use nearly 32KB of LDS per wave group. This drastically lowers the hardware occupancy compared to FP8. Reduced occupancy means the macro-level scheduler cannot easily hide the `vmcnt` (global memory) stall latency, leading to a performance drop-off at large matrix sizes (e.g. 26.7k GFLOPS at N=8192).

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_mm_hip_fp16.py`
   - `python benchmark_mm_hip_fp16.py`
   - If it regresses, explain the reason by inspecting the generated code and/or profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Next Experiments

### P1: Analyze Thread Trace for Bottleneck (N=8192)

- **Rationale:** We currently achieve 32.1k TFLOPS at `N=4096` and drop to 26.7k TFLOPS at `N=8192`. We need to use `rocprofv3 --att` to verify if the global memory `vmcnt` stall is the true reason for the performance drop.
- **Method:** Run a Thread Trace over a single N=8192 execution and extract the instruction cycle timings to determine what percentage of the wave is spent waiting for global memory.
- **Decision:** Use the trace data to decide whether we need to adjust occupancy, or implement a split-phase global load to overlap fetch with the compute stage.
