# gfx1151 HIP fp16/fp8e5m2 Attention Kernel Optimization Plan

## Scope and Metric

- Kernel scope:
  - optimize `kernel_attn/hip/hip_kernel.cu` and its Python wrapper in `kernel_attn/hip/hip_kernel.py`;
  - input/output contract stays fp16 tensors;
  - K/V may be quantized internally to fp8e5m2, or accepted through an explicit prepacked path if an experiment proves that is useful;
  - initial scope stays non-causal forward attention.
- Target shape family:
  - Qwen-Image attention: `B=1`, `H=24`, `S in {1024, 2048, 4096, 8192}`, `D=128`;
  - primary decision shape: `B=1`, `H=24`, `S=4096`, `D=128`.
- Performance metric:
  - `benchmark_attn_hip.py` and generated `attn_hip.csv`;
  - compare against AITER FlashAttention in the same benchmark;
  - report median TFLOPS and derived milliseconds when interpreting results.
- Accuracy gate:
  - compare against fp16 PyTorch SDPA reference for the public fp16-input contract;
  - use combined elementwise tolerance, not separate relative and absolute checks:
    `abs(out - ref_fp16) <= 0.05 * abs(ref_fp16) + 0.05`;
  - keep the quantized-K/V PyTorch reference as a diagnostic, not the primary pass/fail gate;
  - rationale: SageAttention commonly targets `<5%` relative tolerance versus fp16 attention on random inputs, but relative-only gates are too strict near zero.
- Tolerance policy:
  - the starting tolerance is `rtol=0.05`, `atol=0.05`, combined as `rtol * abs(ref_fp16) + atol`;
  - later experiments may loosen `rtol` and `atol` within a reasonable range if it unlocks meaningful speed;
  - start tightening tolerances only after Qwen-Image visual comparisons are available;
  - every tolerance change must be recorded in this file with the benchmark and visual-quality reason.
- Keep rule:
  - correctness passes under the combined tolerance gate;
  - `S=4096` improves or stays flat;
  - `S=8192` does not regress materially;
  - the large-S trend (`2048`, `4096`, `8192`) does not regress materially;
  - if an experiment optimizes only an explicitly separate prepacked path, report packed and end-to-end timings separately.

## Current Baseline

Latest benchmark CSV: `attn_hip.csv`.

| S | AITER TFLOPS | HIP TFLOPS | HIP Prepacked TFLOPS | HIP/AITER | AITER ms | HIP ms | HIP Prepacked ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 22.206809 | 17.142909 | 21.126201 | 77.2% | 0.580 | 0.752 | 0.610 |
| 2048 | 26.918666 | 21.987428 | 24.316010 | 81.7% | 1.915 | 2.344 | 2.120 |
| 4096 | 27.158958 | 22.175404 | 23.854425 | 81.6% | 7.591 | 9.296 | 8.643 |
| 8192 | 27.421504 | 21.874637 | 22.315050 | 79.8% | 30.073 | 37.698 | 36.954 |

Interpretation:
- The HIP kernel is about `1.2-1.3x` slower than AITER on large S after padded `Si` stride.
- HIP TFLOPS is still flatter than AITER from `S=1024` to `8192`, but A4-B/A4-C plus A9-A plus padded `Si` stride lifted the large-S plateau from about `10 TFLOPS` to about `22-24 TFLOPS`.
- The large-S gap is too large for tuning-only polish; the current algorithm structure likely dominates.
- Internal K/V quantization is `O(BHSD)`, while attention is `O(BHS^2D)`, so quantization cannot explain the `S=4096` and `8192` gap by itself. It can still matter at `S=1024` and must be decomposed.
- A standalone V-transposed prepack prototype was tested, rejected, and removed from the code. See A3/A6 notes for the historical measurements.

## Current Kernel Observations

- `attn_fp16_fp8kv` currently launches two kernels:
  - `quantize_kv_e5m2_kernel` converts fp16 K/V to contiguous fp8e5m2 buffers;
  - `fwd_kernel` runs tiled attention over one `(B,H,Br)` query block per workgroup.
- The default/general forward kernel materializes intermediate state in LDS:
  - `Si` stores the `Br x Bc` QK score/probability tile in fp16;
  - `Oi` stores the `Br x D` output accumulator in fp16;
  - every K/V tile does QK WMMA, `__syncthreads`, scalar-ish row softmax, `__syncthreads`, PV WMMA, `__syncthreads`.
- The softmax update is mostly assigned to `tx < Br`, so only one thread per row performs max, exp, sum, probability conversion, and output rescale over full row fragments.
- The output accumulator is repeatedly rounded through fp16 in LDS after each K/V tile; this is both a performance cost and an accuracy compromise.
- The specialized A4-B/A4-C path for `(Br=128, Bc=32, N_WAVES in {8, 16}, D=128)` keeps the output accumulator in fp32 registers and stores only `Si`, `alpha`, and final `inv_l` in LDS.
- The specialized path now pads `Si` row stride from `Bc=32` to `40` half elements to avoid LDS bank conflicts.
- K loads for QK are naturally row-major and contiguous over `D`.
- V loads for PV are currently hostile to coalescing: the WMMA fragment needs 16 tokens for a fixed output-dim lane, but row-major V means those 16 fp8 bytes are separated by `D=128` bytes. A standalone `[B,H,D,S]` VT prepack did not help enough; revisit V packing only as part of a new forward structure.
- For kept shape-specialized paths, prefer divisibility-only contracts and remove boundary / partial-block special handling whenever possible; keep the generic fallback separate if the public API still needs it.

## External Inspirations

### AITER FlashAttention

- Relevant reference: `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`.
- RDNA default uses `BLOCK_M=128`, `BLOCK_N=32` on most RDNA targets, `num_warps=8`, and `waves_per_eu=6`.
- The inner loop keeps online softmax state in registers: `m_i`, `l_i`, and `acc`.
- It avoids materializing the full score tile and output accumulator to LDS between K/V blocks.
- It has separate full-block and masked-block logic; our non-causal fixed-size path can start with the full-block fast path only.
- It uses `exp2`/log2 scaling consistently to avoid slower base-e exponentials.

### SageAttention

- Relevant references:
  - `~/SageAttention/sageattention/triton/quant_per_block.py`;
  - `~/SageAttention/sageattention/triton/quant_per_thread.py`;
  - `~/SageAttention/sageattention/triton/attn_qk_int8_per_block.py`;
  - `~/SageAttention/sageattention/core.py`.
- SageAttention uses practical approximate-attention tolerances versus fp16 attention; use the combined `5% rtol + 0.05 atol` elementwise gate for this kernel.
- SageAttention supports both `HND` (`B,H,S,D`) and `NHD` (`B,S,H,D`) APIs by passing explicit strides.
- Its quantization pipeline separates quantization granularity from the attention kernel: per-block or per-thread Q/K scales, optional K mean smoothing, and architecture-specific attention kernels.
- For this HIP e5m2 path, do not copy int8 quantization blindly. gfx1151 does not have native fp8 WMMA for this path, and Q quantization introduces extra conversion/scaling work. Treat SageAttention as guidance for tolerances, layout flexibility, prepack contracts, and smoothing experiments.

## Layout Assessment

### `B,H,S,D` / HND

- Current HIP layout.
- Best immediate fit for the current one-head-per-workgroup kernel:
  - each head's full sequence is contiguous as `[S,D]`;
  - Q/K row loads are contiguous over `D`;
  - K/V quantization writes contiguous fp8 per head;
  - workgroups do not need cross-head coordination.
- Downsides:
  - adjacent heads for the same token are far apart;
  - if the producer model naturally emits `B,S,H,D`, a transpose or non-contiguous stride path may be needed outside the kernel.

### `B,S,H,D` / NHD

- AITER's benchmark-facing shape uses this layout, and SageAttention supports it.
- It can be better for framework integration and token-major producers.
- For the current one-head-per-workgroup HIP kernel, it is probably worse for fwd memory streaming because advancing one token for a fixed head jumps by `H*D` elements instead of `D`.
- NHD becomes more attractive only if a new kernel processes multiple heads for the same sequence tile in one workgroup or if avoiding external layout conversion dominates the total runtime.

Decision for now:
- Optimize HND first because it matches the current HIP work decomposition and gives contiguous per-head sequence streams.
- Add an explicit layout experiment before committing to the public API:
  - support NHD strides without an internal transpose;
  - benchmark HND vs NHD with identical logical data and no hidden transpose in the timed region;
  - keep NHD only if it wins end-to-end or if app integration requires it.
- Independently of tensor API layout, V-specific packing remains a possible tile-local implementation detail, but the standalone `[B,H,D,S]` VT prepack path is rejected as-is.

## Planned Experiments

| ID | Keep? | Change | Key Result | Why |
|---|---|---|---|---|
| A0 | KEEP | update correctness gate to combined `0.05 * abs(ref) + 0.05` tolerance | target sizes pass `8/8`; small `N=128/512` random cases fail up to `tol_ratio=1.46` | scope test to Qwen target family for now |
| A1 | KEEP infra | decompose benchmark into quantize-only, fwd-prepacked-only, and end-to-end | `S=4096`: quant `0.60 ms`, prepacked fwd `21.59 ms`, end-to-end `21.76 ms` | quantization is not the large-S bottleneck |
| A2 | KEEP | tile sweep around AITER RDNA shape, especially `Br=128`, `Bc=32`, `N_WAVES=8/16` | end-to-end `S=4096` improved `9.53 -> 9.95 TFLOPS`, `S=8192` improved `9.71 -> 9.85 TFLOPS` | small but real primary-metric gain; no large-S regression |
| A3 | REJECT | V fp8 transposed/tile-packed prepack path | `S=4096`: row prepacked `21.59 ms`, VT prepacked `22.19 ms`; VT prepack `2.54 ms` vs row `0.60 ms` | simple `[B,H,D,S]` V transpose did not overcome current forward bottlenecks; code path removed |
| A4 | KEEP partial | register-resident online softmax/PV prototype for fixed `Br=128`, `Bc=32`, `D=128` | A4-C 8-wave register-output path lifts `S=4096` HIP to `16.53 TFLOPS` and prepacked to `17.48 TFLOPS`; full score-tile rewrite still pending | remove LDS score/prob/output round trips |
| A5 | TODO | HND vs NHD stride/layout benchmark | pending | decide public layout strategy |
| A6 | KEEP findings | profile kept or surprising variants with generated-code inspection, PC sampling, bank-conflict PMCs, or thread tracing | padded `Si` stride removes HIP `LDSBankConflict` and lifts `S=4096` prepacked to `23.85 TFLOPS` | explain wins/regressions before further tuning |
| A7 | TODO | optional Sage-inspired K smoothing or scale-bearing quantization experiment | pending | accuracy/headroom only after core fwd is faster |
| A8 | REJECT | shape-specialized cleanup and boundary removal | boundary removal on the fixed fast path regressed because the compile-time specialization raised VGPR pressure and lowered throughput | keep only divisibility-asserted fast paths for fixed target shapes |
| A9 | KEEP partial | perm-based `fp8e5m2x4_to_half2x2` decode | A9-A lifts `S=4096` prepacked to `19.55 TFLOPS`; V-side strided fp8 loads still scalarized | reduce conversion overhead if the combined tolerance gate still passes |
| A10 | KEEP findings | 128-bit load/store audit | Q/global and LDS tile movement use 128-bit ops; strided fp8 V loads and final output stores are narrower | verify hot-path reads/writes stay at the widest legal vector width |

## Next Steps

### A0: Correctness Gate

- Status: complete.
- Update `test_attn_hip.py` so public pass/fail compares to fp16 SDPA using `tol = 0.05 * ref_fp16.abs() + 0.05`.
- Keep current quantized-K/V reference metrics in the printed diagnostics.
- Do not compare relative and absolute tolerances as separate failure criteria.
- Treat `rtol=0.05`, `atol=0.05` as the starting point, not a permanent final quality target.
- If a later optimization needs looser tolerance, adjust both the test and this plan before using the result as a keeper.
- Tighten tolerance only after visual Qwen-Image comparisons show enough quality margin.
- Re-run `python test_attn_hip.py` before any performance experiment.
- Finding: the combined gate passes target sizes `S=1024` and `S=4096`; small random `N=128/512, D=128` cases exceed the starting tolerance, so the test is currently scoped to Qwen target-family sizes.

### A1: Benchmark Decomposition

- Status: complete.
- Add or temporarily expose a prepacked attention path:
  - `quantize_kv_e5m2(k, v) -> (k_fp8, v_fp8)`;
  - `attn_hip_prepacked(q, k_fp8, v_fp8)`.
- Benchmark three timings for all target S values:
  - quantize-only;
  - fwd-prepacked-only;
  - current end-to-end `attn_hip`.
- Keep any prepacked API only if it clarifies performance or provides a useful integration path.
- Finding: prepacked fwd-only is only modestly faster than end-to-end and the gap shrinks at large S, so the next high-impact work must optimize `fwd_kernel`, not quantization/allocation.

### A2: Cheap Tile Sweep

- Status: complete.
- Add configs inspired by AITER RDNA defaults:
  - `(128, 32, 8)`;
  - `(128, 32, 16)`;
  - `(128, 64, 8)`;
  - `(64, 32, 16)`;
  - optionally `(128, 16, 8)` only if generated code stays reasonable.
- Run full correctness and benchmark after each kept config set.
- Reject configs that only improve `S=1024` while regressing `S=4096` or `S=8192`.
- Finding: keep `(128, 64, 8)`, `(128, 32, 16)`, `(128, 32, 8)`, and `(64, 32, 16)`. The autotuner now picks `(128, 32, 16)` at `S=8192` and sometimes for prepacked fwd. Do not add `(128, 16, 8)` unless the PV loop is rewritten for `Bc < 32`.

### A3: V-Specific Prepack

- Status: rejected as implemented and reverted from code.
- Split K and V fp8 layout contracts:
  - keep K row-major `[B,H,S,D]` because QK loads contiguous `D` fragments;
  - pack V for PV as tile-transposed or `[B,H,D,S]`-like storage so a 16-token fragment for a fixed output-dim tile can be loaded contiguously.
- First prototype can be fixed to `D=128` and `Bc=32/64`.
- Measure fwd-prepacked-only first, then include quantize/prepack cost.
- If V prepack wins only when prepack is excluded, document it as a possible cache-reuse API rather than replacing end-to-end attention.
- Finding: the current `[B,H,D,S]` VT prototype is not a keeper. In the normal benchmark, `S=4096` row-major prepacked fwd is `21.589 ms`, while VT prepacked fwd is `22.191 ms`; `S=8192` row-major prepacked fwd is `83.311 ms`, while VT is `88.805 ms`.
- Finding: VT prepack cost is much worse: at `S=4096`, row quantization is `0.600 ms`, while VT quantization is `2.541 ms`; at `S=8192`, row quantization is `1.136 ms`, while VT quantization is `5.044 ms`.
- Interpretation: the current forward bottleneck is not fixed by only making V token fragments contiguous. The extra address arithmetic and transpose/prepack cost erase any possible PV-load benefit. Revisit V packing only as a tile-local layout inside an A4-style forward rewrite, not as this standalone `[B,H,D,S]` path.
- Revert result: removed the standalone VT C++ kernels, stable ops, Python wrappers, benchmark provider, profiling modes, and VT benchmark CSV columns. Historical profile artifacts remain under `tmp_attn_fp8kv_analysis/` for reference.

### A4: Register-Resident Online Attention Rewrite

- Status: in progress.
- Build a specialized fixed-shape prototype before generalizing:
  - `D=128`;
  - `Br=128`;
  - `Bc=32` initially;
  - non-causal, full blocks only.
- Target structure:
  - load Q tile once per query block;
  - loop over K/V blocks;
  - compute QK with WMMA;
  - reduce row max/sum without writing the full `Br x Bc` tile to LDS;
  - update `m_i`, `l_i`, and output accumulator in registers;
  - perform PV immediately after softmax probabilities are available;
  - store normalized output once at the end.
- Primary expected gains:
  - remove `Si` score/probability LDS round trip;
  - remove repeated `Oi` fp16 LDS rescale and round trip;
  - reduce `__syncthreads` frequency;
  - parallelize softmax work across lanes/waves instead of one thread per row.
- Main risk:
  - register pressure for `Br x D` accumulator may lower occupancy. If so, test smaller `Br=64` or split `D` tiles while preserving online softmax state.

#### A4-A: First-Tile Output Fast Path

- Status: rejected and reverted.
- Change tested:
  - remove initial `Oi` zero-fill and initial barrier;
  - use a first-tile `C = P @ V` path instead of `C += P @ V` from zero;
  - skip the first tile's useless `Oi *= rowmax_diff_exp` rescale.
- Correctness: `python test_attn_hip.py` passed `12/12`.
- Same-session benchmark comparison:
  - baseline `S=4096`: HIP `9.444903 TFLOPS`, prepacked `9.866963 TFLOPS`, decomposed prepacked `21.2817 ms`, end-to-end `22.0324 ms`;
  - A4-A `S=4096`: HIP `9.675851 TFLOPS`, prepacked `9.758676 TFLOPS`, decomposed prepacked `21.7418 ms`, end-to-end `21.7234 ms`;
  - baseline `S=8192`: HIP `9.732768 TFLOPS`, prepacked `9.999134 TFLOPS`, decomposed prepacked `85.2115 ms`, end-to-end `84.6056 ms`;
  - A4-A `S=8192`: HIP `9.554699 TFLOPS`, prepacked `9.653398 TFLOPS`, decomposed prepacked `85.2666 ms`, end-to-end `86.5936 ms`.
- Reject reason:
  - `S=8192` regressed materially, especially prepacked benchmark and decomposed end-to-end;
  - prepacked `S=4096` also regressed;
  - the runtime first-tile branch/extra template variant changed generated code and autotune choices (`S=8192` switched to `(64,128,16)`), while the removed first-tile zero/rescale work is too small relative to the repeated LDS softmax/PV structure.
- Revert result: `kernel_attn/hip/hip_kernel.cu` was restored to the previous committed baseline. Per protocol, no test/benchmark/profile was run after the revert.

#### A4-B: Register-Resident Output Accumulator

- Status: kept.
- Change:
  - add specialized `fwd_kernel_reg_o_d128` for `(Br=128, Bc=32, N_WAVES=16, D=128)`;
  - keep the `Br x Bc` score/probability tile in fp16 LDS for this step;
  - replace `Oi` fp16 LDS with per-wave fp32 WMMA output fragments held in registers across all K/V tiles;
  - store per-row `alpha` and final `inv_l` in LDS for the register-fragment rescale/final normalization;
  - reduce this path's dynamic LDS from `40 KB` to about `9 KB` (`Si` plus row scalars).
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Main benchmark results:
  - `S=1024`: HIP `10.568325 TFLOPS`, prepacked `12.201548 TFLOPS`;
  - `S=2048`: HIP `11.787876 TFLOPS`, prepacked `12.597967 TFLOPS`;
  - `S=4096`: HIP `12.451936 TFLOPS`, prepacked `13.024492 TFLOPS`;
  - `S=8192`: HIP `12.561168 TFLOPS`, prepacked `12.659433 TFLOPS`.
- Decomposed timings:
  - `S=1024`: quant kernel `0.1691 ms`, prepacked fwd `1.0617 ms`, end-to-end `1.2268 ms`;
  - `S=2048`: quant kernel `0.3208 ms`, prepacked fwd `4.0704 ms`, end-to-end `4.3701 ms`;
  - `S=4096`: quant kernel `0.6013 ms`, prepacked fwd `15.7722 ms`, end-to-end `16.4859 ms`;
  - `S=8192`: quant kernel `1.1401 ms`, prepacked fwd `64.6722 ms`, end-to-end `66.2321 ms`.
- Interpretation:
  - The output accumulator LDS round trip was a large bottleneck; removing it gives a broad `25-32%` large-S TFLOPS improvement.
  - The remaining gap to AITER is still about `2.1-2.2x`, so A4-C should target the remaining `Si` LDS score/probability round trip and serial row softmax.
  - This result supports the structural direction: using more registers and less LDS wins on gfx1151 even before fully matching AITER's tensor-register online-softmax structure.

#### A4-C: 8-Wave Register-Output Variant

- Status: kept.
- Change:
  - extend the A4-B register-output path to `(Br=128, Bc=32, N_WAVES=8, D=128)`;
  - keep the 16-wave variant available, but let autotune choose between both existing configs;
  - this matches AITER's 256-thread workgroup shape more closely while each wave holds more output fragments in registers.
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Autotune result: `(128, 32, 8)` selected for `hip` and `hip_prepacked` at every target size.
- Main benchmark results:
  - `S=1024`: HIP `12.525824 TFLOPS`, prepacked `14.578468 TFLOPS`;
  - `S=2048`: HIP `15.489495 TFLOPS`, prepacked `16.885610 TFLOPS`;
  - `S=4096`: HIP `16.533739 TFLOPS`, prepacked `17.479504 TFLOPS`;
  - `S=8192`: HIP `16.309928 TFLOPS`, prepacked `16.655114 TFLOPS`.
- Decomposed timings:
  - `S=1024`: quant kernel `0.1676 ms`, prepacked fwd `0.8853 ms`, end-to-end `1.0348 ms`;
  - `S=2048`: quant kernel `0.3179 ms`, prepacked fwd `3.0435 ms`, end-to-end `3.3219 ms`;
  - `S=4096`: quant kernel `0.5970 ms`, prepacked fwd `11.9459 ms`, end-to-end `12.4869 ms`;
  - `S=8192`: quant kernel `1.1390 ms`, prepacked fwd `50.5253 ms`, end-to-end `51.0373 ms`.
- Interpretation:
  - The 8-wave version is clearly better than the 16-wave register-output path despite higher per-wave register pressure.
  - The result confirms the workgroup-thread count was part of the post-A4-B bottleneck; fewer waves with more register-resident output per wave better match gfx1151 and AITER's RDNA default.
  - The remaining large-S gap to AITER is now roughly `1.6-1.7x` for end-to-end HIP and `1.55-1.65x` for prepacked fwd.
  - Next structural target remains the `Si` LDS round trip and row softmax distribution.
  - Before rewriting `Si`, profile LDS bank conflicts against AITER and use ablations to locate whether conflicts come from QK score stores, row-softmax `Si` read/write, or PV fragment loads.

### A5: Layout Experiment

- Add an NHD wrapper or benchmark provider that passes explicit strides to the same kernel logic.
- Benchmark:
  - HND contiguous `q/k/v` as today;
  - NHD contiguous `q/k/v` with no hidden transpose in the timed function;
  - optional non-contiguous views only as integration diagnostics, not primary performance data.
- Decision criteria:
  - if HND wins for fwd and quantize, keep HND as the optimized kernel layout;
  - if NHD wins end-to-end because the model avoids a transpose, support NHD as a public wrapper but keep the internal fwd memory path honest;
  - if neither dominates, expose layout-specific configured ops and let benchmark/autotune choose.

### A6: Profiling Protocol

- Inspect generated ISA after each surprising result.
- If the kernel remains below `15 TFLOPS` after A3/A4, profile before further micro-optimizations.
- Prioritize these questions:
  - is the stall dominated by global V loads, LDS traffic, VALU softmax/exp, barriers, or WMMA under-issue;
  - did V prepack convert strided memory into coalesced loads;
  - did register pressure reduce occupancy enough to erase online-attention gains;
  - are `exp2f` and fp16/fp32 conversions generating unexpected code;
  - are the hot-path reads/writes actually using the widest legal 128-bit vector ops, or did the compiler scalarize any of the critical load/store sites;
  - does AITER have lower or zero LDS bank conflicts, and if so which exact LDS access pattern in our kernel is responsible for the difference.

#### A6-A: AITER Kernel-Trace Baseline

- Artifacts:
  - `tmp_attn_fp8kv_analysis/profile_aiter_4096/aiter_results.db`;
  - `tmp_attn_fp8kv_analysis/profile_aiter_4096_venv/aiter_results_results.db`.
- Workload: `python tmp_attn_fp8kv_analysis/profile_attn_aiter.py -N 4096 --iters 5 --warmup 2`.
- Captured dispatches:
  - 7 `attn_fwd` dispatches, one per warmup/profile call;
  - 3 random-normal setup kernels;
  - 7 fp16 fill kernels and 7 fp32 fill kernels for AITER output/LSE scratch setup.
- Dominant kernel:
  - `attn_fwd`: `97.34%` of GPU kernel time;
  - average duration `7371.192 us` with `rocprofv3`, `7216.647 us` in the earlier build-tree trace;
  - logical performance at `B=1,H=24,S=4096,D=128`: about `28.0 TFLOPS` from the venv profile and about `28.6 TFLOPS` from the earlier profile;
  - dispatch metadata: grid reported as `grid_x=6144`, `grid_y=32`, `grid_z=1`, `workgroup_x=256`, `lds_size=32768`, `vgpr_count=224`, `sgpr_count=128`, `scratch_size=0`.
- Interpretation:
  - ROCProfiler's `grid_x` is total work-items in x, not logical Triton programs. AITER's source grid is `(H=24, ceil(S/BLOCK_M)=32, B=1)`, and `24 * workgroup_x(256) = 6144` explains the profile row.
  - AITER's fwd path is a single dominant kernel with `BLOCK_M=128`, RDNA `BLOCK_N=32`, 8 waves per workgroup, 32 KB LDS, and high VGPR use (`224`).
  - AITER spends negligible time in helper fill kernels relative to attention; our optimization target should remain the main fwd kernel structure.
  - The 32 KB LDS footprint suggests AITER is not materializing both a full score tile and a full fp16 output accumulator in LDS the way our current HIP path does; this reinforces A4's register-resident online-softmax/PV direction.

#### A6-B: HIP vs AITER Kernel-Trace Comparison

- HIP artifacts:
  - `tmp_attn_fp8kv_analysis/profile_hip_prepacked_configured_4096/hip_prepacked_configured_results_results.db`;
  - `tmp_attn_fp8kv_analysis/profile_hip_prepacked_vt_configured_4096/hip_prepacked_vt_configured_results_results.db`.
- Workloads:
  - row-major prepacked: `python tmp_attn_fp8kv_analysis/profile_attn_hip.py -N 4096 --mode prepacked_configured --iters 5 --warmup 2 --config 128,32,16`;
  - VT prepacked: `python tmp_attn_fp8kv_analysis/profile_attn_hip.py -N 4096 --mode prepacked_vt_configured --iters 5 --warmup 2 --config 128,32,16`.
- Main fwd kernel comparison at `S=4096`:
  - AITER `attn_fwd`: `7371.192 us` average, about `28.0 TFLOPS`, `workgroup_x=256`, `lds_size=32768`, `vgpr_count=224`, `scratch_size=0`;
  - HIP row prepacked `fwd_kernel<128,32,16>`: `21118.214 us` average, about `9.76 TFLOPS`, `workgroup_x=512`, `lds_size=40960`, `vgpr_count=88`, `scratch_size=0`;
  - HIP VT prepacked `fwd_kernel_vt<128,32,16>`: `20841.611 us` average, about `9.89 TFLOPS`, `workgroup_x=512`, `lds_size=40960`, `vgpr_count=88`, `scratch_size=0`.
- Launch-shape interpretation:
  - AITER's profile grid is `grid_x=6144`, `grid_y=32`, `grid_z=1`; ROCProfiler reports total x work-items, so this corresponds to `24` logical head programs times `256` threads, with `32` query blocks in y.
  - HIP's profile grid is `grid_x=512`, `grid_y=24`, `grid_z=32`, matching our launch `grid=(B=1,H=24,Tr=32)` and `block=512`; ROCProfiler reports x total work-items as `1 * 512`.
- Key differences:
  - AITER is about `2.86x` faster than HIP row prepacked in the venv kernel trace, matching the benchmark gap.
  - HIP uses twice the workgroup threads (`512` vs `256`) but much lower VGPR count (`88` vs `224`), indicating our problem is not register spilling or scratch. We are likely underusing registers and overusing LDS/barriers/serial softmax work.
  - Estimated occupancy from the reference calculator is not the likely blocker: AITER is roughly VGPR-limited to about `37.5%` occupancy, while HIP is roughly LDS/workgroup-limited to about `75%` occupancy for this config. AITER is faster despite lower occupancy because its per-wave work is more efficient.
  - HIP uses more LDS (`40 KB` vs `32 KB`) because it stores both `Br x Bc` scores/probabilities and `Br x D` fp16 output state in LDS. AITER's high VGPR count and lower LDS align with register-resident online attention.
  - Both AITER and HIP report `scratch_size=0`, so stack/scratch is not the immediate issue.
  - VT's profile-only fwd average is slightly faster than row-major, but the normal benchmark shows VT is slower; the effect is too small and unstable to matter, and the prepack cost is much worse.
- Next optimization implication:
  - Stop pursuing standalone VT prepack as A3.
  - Build A4 around a specialized `Br=128`, `Bc=32`, `D=128`, non-causal, full-block kernel that keeps online softmax state and output accumulators out of LDS as much as possible.

#### A6-C: LDS Bank Conflict Profile and Swizzle Design

- Status: TODO.
- Goal:
  - measure LDS bank conflict rate for AITER and the current HIP A4-C path at the same logical workload, starting with `B=1`, `H=24`, `S=4096`, `D=128`;
  - if AITER reports zero or negligible bank conflicts, treat zero/negligible conflicts as the target for the HIP path too;
  - do not assume row-major `Si` is acceptable just because A4-C improved throughput.
- Initial profiler command shape:
  - AITER: use `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace --stats --pmc LDSBankConflict ... -- ~/venv_torch/bin/python tmp_attn_fp8kv_analysis/profile_attn_aiter.py -N 4096 --iters 5 --warmup 2`;
  - HIP: use the same profiler binary and PMC with `tmp_attn_fp8kv_analysis/profile_attn_hip.py -N 4096 --mode prepacked_configured --iters 5 --warmup 2 --config 128,32,8`;
  - collect kernel duration, `LDSBankConflict`, VGPR, LDS size, and generated kernel name in the same ledger row.
- Normalize/interpretation:
  - compare raw `LDSBankConflict` for the dominant fwd kernels first;
  - if extra LDS counters are available, also normalize by LDS instructions/accesses so a shorter kernel is not mistaken for a lower conflict rate;
  - if counter availability differs by ROCProfiler build, record that explicitly rather than mixing profiler versions.
- HIP ablations to locate conflict source:
  - full A4-C kernel as baseline;
  - QK + row-softmax only: keep `mul_A_BT` and `Si` softmax read/write, skip PV and final output;
  - PV-only synthetic path: pre-fill or generate a stable probability tile in `Si`, skip QK and softmax, run only the PV fragment-load/update path;
  - softmax-only path: exercise row-wise `Si` vector read/write without WMMA to isolate the `tx < Br` access pattern;
  - scalar-row path: isolate `alpha`/`inv_l` LDS traffic to verify it is not the measured conflict source.
- Swizzle design candidates:
  - start with a minimal padded `Si` row stride to test whether the current `Bc=32` row stride causes same-bank row starts;
  - then test an AITER-like phase/swizzled layout for the `Si` tile that preserves efficient row softmax reads while improving PV dot-operand LDS reads;
  - inspect generated ISA for `ds_load`/`ds_store` width and address pattern after each layout change.
- Keep rule for swizzle experiments:
  - correctness must still pass `~/venv_torch/bin/python test_attn_hip.py`;
  - `S=4096` and `S=8192` benchmark must improve or stay flat;
  - `LDSBankConflict` must move materially toward AITER's rate;
  - if conflicts drop but runtime regresses, reject the swizzle and document whether the cost was address arithmetic, poorer vectorization, or register pressure.
- Finding:
  - AITER `attn_fwd` at `S=4096` has `0.0` average `LDSBankConflict`, avg `7492.884 us`, `32768` LDS bytes, and `224` VGPRs;
  - HIP A4-C before A9-A has `65.753` average `LDSBankConflict` (`460.271` total over 7 fwd dispatches), avg `12134.075 us`, `9216` LDS bytes, and `200` VGPRs;
  - HIP A9-A keeps the same bank-conflict count (`65.753` average) while reducing avg fwd profile duration to `10375.665 us` and VGPRs to `192`;
  - padding `Si` row stride to `Bc + 8` removes the conflict count completely and improves runtime materially.

#### A6-D: Padded `Si` Stride

- Status: kept.
- Change:
  - only on `fwd_kernel_reg_o_d128`, set `kSiStride = Bc + 8` for the score/probability tile;
  - use the padded stride in QK stores, row softmax, and PV reads;
  - dynamic LDS increases from `9216` to `11264` bytes for the kept `(128,32,8)` fast path.
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Main benchmark results:
  - `S=1024`: HIP `17.142909 TFLOPS`, prepacked `21.126201 TFLOPS`;
  - `S=2048`: HIP `21.987428 TFLOPS`, prepacked `24.316010 TFLOPS`;
  - `S=4096`: HIP `22.175404 TFLOPS`, prepacked `23.854425 TFLOPS`;
  - `S=8192`: HIP `21.874637 TFLOPS`, prepacked `22.315050 TFLOPS`.
- Decomposed timings:
  - `S=1024`: quant kernel `0.1666 ms`, prepacked fwd `0.6130 ms`, end-to-end `0.7542 ms`;
  - `S=2048`: quant kernel `0.3173 ms`, prepacked fwd `2.1054 ms`, end-to-end `2.3524 ms`;
  - `S=4096`: quant kernel `0.5979 ms`, prepacked fwd `8.7996 ms`, end-to-end `9.3883 ms`;
  - `S=8192`: quant kernel `1.1398 ms`, prepacked fwd `37.4236 ms`, end-to-end `38.1249 ms`.
- Profile: `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/bin/rocprofv3 --kernel-trace --stats --pmc LDSBankConflict ... -- ~/venv_torch/bin/python tmp_attn_fp8kv_analysis/profile_attn_hip.py -N 4096 --mode prepacked_configured --iters 5 --warmup 2 --config 128,32,8`.
- Profile findings:
  - `fwd_kernel_reg_o_d128<128,32,8>` average duration `8788.344 us`;
  - `LDSBankConflict` average `0.0`, total `0.0`, matching AITER's zero-conflict baseline;
  - LDS size `11264` bytes, VGPR `192`, scratch `0`.
- ISA findings:
  - static instruction count remains about `1996`, effectively unchanged from A9-A;
  - static selected memory/decode counts are unchanged: `48` `global_load_b128`, `48` `global_load_d16_u8`, `48` `global_load_d16_hi_u8`, `36` `ds_load_b128`, and `128` `v_perm_b32`;
  - the speedup is from avoiding bank conflicts in the existing LDS traffic, not from changing global V-load width.

### A7: Sage-Inspired Accuracy/Quantization Follow-Ups

- Only run after a faster fwd baseline exists.
- Candidate ideas:
  - K mean smoothing before e5m2 quantization if fp16-reference tolerance failures cluster on biased inputs;
  - optional per-block scale-bearing int8 or fp8-like quantization for K/V if e5m2 unscaled quantization becomes the accuracy bottleneck;
  - Q quantization only if gfx1151 generated code shows a real path to faster QK than fp16 WMMA plus conversion overhead.
- Reject any quantization idea that improves accuracy but loses the performance advantage versus AITER by a large margin.

### A8: Shape-Specialized Cleanup

- Status: rejected and reverted.
- Change tested:
  - remove the `blk_x` / `blk_y` boundary guards from the fixed `fwd_kernel_reg_o_d128` fast path;
  - make the fixed `Br=128`, `Bc=32`, `D=128` path compile-time full-tile only while leaving the generic fallback path guarded.
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Main benchmark results:
  - `S=1024`: HIP `13.289322 TFLOPS`, prepacked `16.092992 TFLOPS`;
  - `S=2048`: HIP `15.968934 TFLOPS`, prepacked `17.394867 TFLOPS`;
  - `S=4096`: HIP `16.762173 TFLOPS`, prepacked `17.818590 TFLOPS`;
  - `S=8192`: HIP `16.804797 TFLOPS`, prepacked `17.177785 TFLOPS`.
- Rejection reason:
  - the compile-time `FULL_TILES` split shrank static control flow, but the fixed 8-wave symbol grew to about `256` VGPRs and lost about `1.73 TFLOPS` at `S=4096` compared with A9-A;
  - the 8-wave kernel still has the same `global_load_d16_u8` / `global_load_d16_hi_u8` V-side pressure, so the structural removal did not attack the real bottleneck;
  - the 16-wave variant remains the autotune winner on the regressed code path, showing the change made the preferred 8-wave fast path worse instead of better.
- Revert result: `kernel_attn/hip/hip_kernel.cu` was restored to the committed A9-A baseline. Per protocol, no test/benchmark/profile was run after the revert.

### A9: Perm-Based FP8 Decode

- Prototype `fp8e5m2x4_to_half2x2` using only `__builtin_amdgcn_perm` or equivalent byte/half shuffle operations.
- Do not add special inf/nan/denormal handling in the first version.
- Rely on the existing combined tolerance gate to decide whether the approximation is acceptable for the target shapes.
- Benchmark the decode path separately from the attention kernel when possible, then include it in the end-to-end benchmark.
- Keep rule:
  - if the decode path is faster but fails the tolerance gate, reject it;
  - if it passes tolerance but does not move end-to-end time, reject it;
  - if it improves both, keep it as the preferred conversion primitive.

#### A9-A: Contiguous K Fragment Decode

- Status: kept.
- Change:
  - add `fp8e5m2x4_to_half2x2` using `__builtin_amdgcn_perm`;
  - use four packed 32-bit chunks in `load_e5m2x16_as_fp16` for contiguous fp8 K fragments;
  - leave the strided V fp8 path unchanged for this step.
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Main benchmark results:
  - `S=1024`: HIP `13.720099 TFLOPS`, prepacked `16.452199 TFLOPS`;
  - `S=2048`: HIP `18.079397 TFLOPS`, prepacked `19.902636 TFLOPS`;
  - `S=4096`: HIP `18.418717 TFLOPS`, prepacked `19.548234 TFLOPS`;
  - `S=8192`: HIP `18.570807 TFLOPS`, prepacked `18.972084 TFLOPS`.
- Decomposed timings:
  - `S=1024`: quant kernel `0.1683 ms`, prepacked fwd `0.7935 ms`, end-to-end `0.9441 ms`;
  - `S=2048`: quant kernel `0.3189 ms`, prepacked fwd `2.5955 ms`, end-to-end `2.8414 ms`;
  - `S=4096`: quant kernel `0.5956 ms`, prepacked fwd `10.7136 ms`, end-to-end `11.2972 ms`;
  - `S=8192`: quant kernel `1.1429 ms`, prepacked fwd `43.8912 ms`, end-to-end `45.0654 ms`.
- ISA/profile findings:
  - A4-C fwd static instruction count drops from about `2324` to `1995` instructions;
  - fwd VGPR count drops from `200` to `192` in the PMC profile;
  - `global_load_d16_u8` / `global_load_d16_hi_u8` remain at `48` each in the fwd symbol, so these are likely the strided V-side fp8 loads rather than the contiguous K path;
  - fwd static `v_perm_b32` count becomes `128`; despite prior concerns about `v_perm` front-end cost, the shorter decode sequence wins here.

### A10: 128-bit Load/Store Audit

- Audit the hot path to confirm it keeps the widest practical 128-bit read/write form everywhere possible.
- Check the generated ISA for:
  - `buffer_load_b128` / `buffer_store_b128` on global-memory paths;
  - `ds_load_b128` / `ds_store_b128` on LDS paths;
  - any accidental `u16`/`u32` scalarization in the hot loops.
- Compare our hot-path access width against AITER's generated code and keep the audit tied to the same `S=4096` trace.
- If a narrower access appears, identify whether the cause is layout, alignment, pointer math, masking, or the decode primitive.
- Keep rule:
  - only keep a change if correctness passes and benchmark does not regress;
  - if the compiler already emits the widest useful vector ops, do not add extra shuffles or casts just to force them.
- Finding:
  - HIP A4-C/A9-A uses `global_load_b128` for wide fp16/global fragments and `ds_load_b128` / `ds_store_b128` for most LDS tile movement;
  - the remaining narrow hot-path global loads are `global_load_d16_u8` / `global_load_d16_hi_u8`, likely from the strided V fp8 fragment path;
  - final output stores are scalar half stores (`global_store_b16` / `global_store_d16_hi_b16`), but AITER also emits scalar output stores, so this is not the first target;
  - next width-focused experiment should target V-side fp8 access/decode rather than forcing output stores to 128-bit.

#### A10-A: V Tile Transpose to LDS

- Status: rejected and reverted.
- Change tested:
  - preload each `Bc x D` row-major fp8 V tile once per K/V block;
  - convert it to fp16 in LDS as transposed `[D,Bc]`;
  - read PV fragments from contiguous LDS rows with `HALF16` instead of repeated strided global fp8 byte loads.
- Correctness: `~/venv_torch/bin/python test_attn_hip.py` passed `12/12` configs.
- Benchmark: `~/venv_torch/bin/python benchmark_attn_hip.py`.
- Main benchmark results:
  - `S=1024`: HIP `9.117587 TFLOPS`, prepacked `10.356332 TFLOPS`;
  - `S=2048`: HIP `10.890340 TFLOPS`, prepacked `11.633517 TFLOPS`;
  - `S=4096`: HIP `11.625828 TFLOPS`, prepacked `12.107011 TFLOPS`;
  - `S=8192`: HIP `11.941535 TFLOPS`, prepacked `12.126874 TFLOPS`.
- Rejection reason:
  - the change removes the static `global_load_d16_u8` / `global_load_d16_hi_u8` V-side pattern, but replaces it with a per-K/V-tile transpose loop, scalar LDS stores, extra `v_perm_b32`, and a larger dynamic LDS footprint (`9216 -> 17408` bytes);
  - autotune switches from `(128,32,8)` to `(128,32,16)`, indicating the 8-wave version lost its previous advantage;
  - generated code for the rejected 8-wave symbol has about `1750` static instructions, `151` VGPRs, `130` static `v_perm_b32`, `48` `ds_load_b128`, `18` `ds_store_b16`, and only `1` static `global_load_b32`; the static code is shorter, but dynamic transpose/preload work dominates;
  - generated code for the rejected 16-wave symbol has about `1126` static instructions, `115` VGPRs, and `66` static `v_perm_b32`, but runtime remains far below A9-A.
- Revert result: `kernel_attn/hip/hip_kernel.cu` was restored to the committed A9-A baseline. Per protocol, no test/benchmark/profile was run after the revert.

## Non-Negotiable Run Protocol

1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, use `ps` to check for any running job.
2. Per-step order:
   - `python test_attn_hip.py`
   - `python benchmark_attn_hip.py`
   - If it regresses, explain the reason by inspecting the generated code and/or profiling.
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert.
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition.
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Durable Findings

- The pre-A4 baseline was structurally behind AITER by about `2.8-2.9x` at large S.
- A4-B/A4-C reduced the large-S gap from about `2.8-2.9x` to about `1.6-1.7x`; the current path is still structurally behind AITER but no longer dominated by the output-accumulator LDS round trip.
- A9-A further reduced the large-S gap to about `1.45-1.5x` by shrinking the contiguous K fp8 decode path.
- Padded `Si` stride further reduced the large-S gap to about `1.2-1.3x` and removed HIP `LDSBankConflict` (`65.753 -> 0.0` average per fwd dispatch at `S=4096`).
- The 128-bit audit shows Q/global and LDS tile movement are already wide; the next memory-width target is V-side fp8 access/decode, not output stores.
- The high-priority issues are not likely fixed by only changing autotune order:
  - full score/probability materialization in LDS;
  - repeated output accumulator materialization in fp16 LDS;
  - row softmax work concentrated into `tx < Br`;
  - strided V loads for PV WMMA.
- HND is probably the better internal layout for a one-head-per-workgroup HIP kernel, but this must be verified against NHD because AITER/SageAttention APIs are NHD-capable and model integration may matter.
- V layout should be considered separately from public tensor layout. A V-specific packed fp8 layout can improve PV loads without forcing public NHD tensors.
- Accuracy should use the combined tolerance `0.05 * abs(ref_fp16) + 0.05`, not independent relative and absolute gates.
- Tolerances are a speed/quality policy lever: loosen only with a documented speed reason, and tighten after visual Qwen-Image validation exists.
- K/V quantization is not the large-S bottleneck. At `S=4096`, row-major quantization is about `0.60 ms` while prepacked forward is about `21.59 ms`.
- The current prepacked row-major fp8 path gives only a small fwd-only gain (`S=4096`: `9.55 TFLOPS` prepacked vs `9.47 TFLOPS` end-to-end in decomposition), so the forward kernel must be rewritten.
- The standalone VT fp8 path was rejected and removed. It was slower in normal benchmark fwd time and increased `S=4096` quantization/prepack from about `0.60 ms` to about `2.54 ms`.
- Adding AITER-inspired `Br=128/Bc=32` configs is a small keeper. It improves the primary `S=4096` end-to-end result to about `9.95 TFLOPS`, but it does not change the conclusion that the forward algorithm is structurally behind AITER.
- Profiling confirms the algorithmic gap: AITER `attn_fwd` is about `7.37 ms` with `32 KB` LDS and `224` VGPRs; HIP `fwd_kernel<128,32,16>` is about `21.12 ms` with `40 KB` LDS and `88` VGPRs. HIP has no scratch, so the next target is LDS/barrier/softmax structure, not spill cleanup.

## Do-Not-Repeat (Unless New Preconditions)

- Do not spend multiple iterations on minor config reordering before decomposing quantize/fwd cost.
- Do not optimize NHD first unless HND vs NHD benchmarking proves it matters end-to-end.
- Do not add Q quantization before the fwd kernel removes its obvious LDS/softmax bottlenecks.
- Do not claim a prepacked-only win as an end-to-end win unless prepack cost is also measured.
- Do not continue standalone `[B,H,D,S]` VT prepack unless a new forward kernel changes the memory-access preconditions; it is already slower as implemented.
- Do not copy SageAttention int8 strategies directly without checking gfx1151 ISA and generated code. RDNA3.5 does not have faster int8 wmma than fp16.

## Reference

- `~/amd-llvm-project/`, especially `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst` - hipcc source code
- `~/rdna35-isa-markdown/`
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py` - AITER FlashAttention forward reference
- `~/SageAttention/sageattention/triton/quant_per_block.py` - SageAttention per-block int8 quantization reference
- `~/SageAttention/sageattention/triton/quant_per_thread.py` - SageAttention per-thread int8 quantization reference
- `~/SageAttention/sageattention/triton/attn_qk_int8_per_block.py` - SageAttention Triton int8-QK attention reference
- `~/SageAttention/sageattention/core.py` - SageAttention public API, layout, smoothing, and tolerance context
