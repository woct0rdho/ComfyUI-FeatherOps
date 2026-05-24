# gfx1151 HIP fp16/fp8e5m2 Attention Kernel Optimization Plan

## Scope And Metric

- Target kernel: `kernel_attn/hip/hip_kernel.cu` with wrapper `kernel_attn/hip/hip_kernel.py`.
- Public contract: fp16 Q/K/V inputs and fp16 output. Internal/prepacked K/V may use fp8e5m2.
- Target shape family: Qwen-Image attention with `B=1`, `H=24`, `S in {1024,2048,4096,8192}`, `D=128`.
- Primary decision shape: `S=4096`; large-S trend must include `S=8192`.
- Current scope: non-causal, full-block forward only, `D=128` only.
- Accuracy gate: compare to fp16 PyTorch SDPA with `abs(out - ref_fp16) <= 0.05 * abs(ref_fp16) + 0.05`.
- Primary performance command: `python benchmark_attn_hip.py`, compared with AITER FlashAttention in the same run.
- Keep rule: correctness passes, `S=4096` improves or stays flat, `S=8192` does not materially regress, no scratch/private segment, and bank conflicts stay zero/negligible unless the runtime win clearly justifies further investigation.

## Current State

- Current fp8 config list: `(64,32,8)` first, with `(128,32,8)` retained as the previous stable fallback and current `S=8192` prepacked winner.
- Current forward symbols: `fwd_kernel_kv_staged_d128<64,32,8>` and `fwd_kernel_kv_staged_d128<128,32,8>`.
- Removed obviously unneeded A13-A probes from autotune: `(64,64,8)`, `(64,32,16)`, and `(32,32,8)`.

### Previous Baseline A12-D6

| S | AITER TFLOPS | HIP TFLOPS | HIP Prepacked TFLOPS | HIP/AITER | AITER ms | HIP ms | HIP Prepacked ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 22.218375 | 17.699625 | 21.715846 | 79.7% | 0.580 | 0.728 | 0.593 |
| 2048 | 27.502781 | 22.206512 | 24.004859 | 80.7% | 1.874 | 2.321 | 2.147 |
| 4096 | 27.477566 | 23.033576 | 23.751923 | 83.8% | 7.503 | 8.950 | 8.680 |
| 8192 | 27.353322 | 22.736288 | 23.170572 | 83.1% | 30.147 | 36.269 | 35.590 |

- Profile at `S=4096`, forced `(128,32,8)`: `8546.388 us`, `LDSBankConflict=0.0`, VGPR `226`, scratch `0`.
- Static fwd counts: `34` `global_load_b128`, `0/0` `global_load_d16_u8`/`global_load_d16_hi_u8`, `52` `ds_load_b128`, `6` `ds_store_b128`, `5` `s_barrier`, `185` `s_waitcnt`, `32` WMMA, `33` `v_exp_f32`.

### Current Baseline A13-A

- Final config list: `(64,32,8)`, `(128,32,8)`.
- Temporary configs tested and removed: `(64,64,8)`, `(64,32,16)`, `(32,32,8)`.
- Correctness: `test_attn_hip.py` passed `2/2` final configs.
- Benchmark selected `(64,32,8)` for all end-to-end sizes and most prepacked sizes; `(128,32,8)` won prepacked `S=8192`.

| S | AITER TFLOPS | HIP TFLOPS | HIP Prepacked TFLOPS | HIP Config | Prepacked Config |
|---:|---:|---:|---:|---|---|
| 1024 | 22.236013 | 19.219206 | 24.799499 | `(64,32,8)` | `(64,32,8)` |
| 2048 | 27.309833 | 22.816377 | 25.306275 | `(64,32,8)` | `(64,32,8)` |
| 4096 | 27.295072 | 23.050048 | 24.206595 | `(64,32,8)` | `(64,32,8)` |
| 8192 | 27.463785 | 23.154828 | 23.532040 | `(64,32,8)` | `(128,32,8)` |

- Decomposed selected-path timings: `S=4096` prepacked `8.4499 ms`, end-to-end `8.8896 ms`; `S=8192` prepacked `35.3245 ms`, end-to-end `35.4353 ms`.
- Forced profile for `(64,32,8)` at `S=4096`: `8345.861 us`, `LDSBankConflict=0.0`.
- Current generated metadata: `(64,32,8)` VGPR `187`, scratch `0`, text size `0x1fd0`; `(128,32,8)` VGPR `222`, text size `0x3374`.
- Interpretation: halving `Br` halves output accumulator tiles per workgroup (`kRegTiles` `8 -> 4`) and lowers VGPR/code size enough to offset twice as many query CTAs. Keeping `Bc=32` preserves the zero-conflict LDS pattern.
- Keep reason: `(64,32,8)` is selected for end-to-end at every target size and improves the current primary/large-S baseline; `(128,32,8)` remains as a low-risk fallback and still wins prepacked `S=8192`.
- Rejection reason for removed probes: `(64,64,8)` repeats the already-rejected `Bc=64` direction; `(64,32,16)` and `(32,32,8)` were correct but not selected and add autotune/compile surface without current benefit.

## Current Kernel Structure

- Quantization kernel converts fp16 K/V to contiguous fp8e5m2 buffers with branchless packed RNE.
- Forward stages raw fp8 K and V tiles in LDS with K/V stride `D + 16`.
- Forward keeps the output accumulator in fp32 registers and stores only once at the end.
- Forward still materializes `Si`, the `Br x Bc` fp16 score/probability tile, in LDS.
- Row softmax is still concentrated in `tx < Br`, one thread per row.
- `Si` stride is `Bc + 8`, which removed measured LDS bank conflicts for the kept `Bc=32` paths.
- HND layout (`B,H,S,D`) is the optimized internal/public fast path; NHD view and one-call NHD pack/swizzle variants were slower.

## External Reference Points

- AITER RDNA path uses `BLOCK_M=128`, `BLOCK_N=32`, 8 warps, WMMA, register online softmax state, and swizzled LDS. It profiles around `7.4-7.5 ms` at `S=4096`, `LDSBankConflict=0.0`, VGPR about `224`, scratch `0`.
- CK-Tile FMHA is useful as a dataflow/layout reference: Q register residency, K/V LDS staging, distributed score/P/output register tiles, and careful LDS layouts. Do not copy CDNA MFMA assumptions for gfx1151.
- SageAttention references are useful for quantization/tolerance context, not as direct int8 kernel templates. RDNA3.5 does not provide a faster int8 WMMA path than the current fp16 WMMA path.

## Experiment Ledger

| ID | Status | Result | Keep/Reject Reason |
|---|---|---|---|
| A0 | Kept | Combined fp16-reference tolerance `0.05 * abs(ref) + 0.05` | Target Qwen sizes pass; relative-only near-zero failures avoided. |
| A1 | Kept infra | Added quant/prepacked/end-to-end decomposition | Quantization is not the large-S bottleneck. |
| A2 | Superseded | Added early AITER-inspired tile configs | Useful for initial lift, but D=128 staged path later narrowed configs. |
| A3 | Rejected | Standalone VT / `[B,H,D,S]` V prepack | Slower forward and much higher prepack cost. Revisit only inside a different forward dataflow. |
| A4-B/C | Kept | Register-resident fp32 output accumulator, then 8-wave variant | Removed output accumulator LDS round trip; major large-S improvement. |
| A5 | Kept finding | HND vs NHD layout assessment | HND-contiguous wins; NHD view is mainly hurt by strided K/V quantization. |
| A6-D | Kept | Padded `Si` stride to `Bc + 8` | Removed HIP `LDSBankConflict` and improved runtime. |
| A7-B | Kept | Branchless packed RNE fp16-to-e5m2 quantization | Passed tolerance and improved quantization/end-to-end timing. |
| A8 | Mostly rejected | Boundary/branch cleanup | Full/helper branch removal regressed; only final output store guard removal survived. |
| A9-A | Kept | Perm-based contiguous K fp8 decode | Reduced fp8 decode overhead and improved prepacked path. |
| A10 | Findings | 128-bit load/store audit | Q/global and LDS tile movement are wide; V global scalar loads were the key memory-width problem. |
| A11 | Rejected | Local softmax split/cache patches | Did not improve the structural `Si`/softmax/PV bottleneck. |
| A12-A | Rejected | Row-owned skeleton with existing `Si` | Higher VGPR and worse V scalar-load footprint; old path still selected. |
| A12-B/C1 | Rejected | First register-P/PV wave-shuffle sentinel `(128,32,83)` | Correctness failed; must validate WMMA C-to-A fragment transpose in a minimal fixture before full attention. |
| A12-D1 | Rejected | Q LDS load-once on row-owned skeleton | LDS `44 KB`, VGPR `252`, private segment use. |
| A12-D2 | Rejected | Raw K-only LDS staging | Reduced K vector loads but added LDS/barrier/wait pressure while V scalar loads remained. |
| A12-D3 | Kept | Raw fp8 K/V LDS staging sentinel `(128,32,82)` | Removed V global scalar loads, preserved zero bank conflicts, improved large-S. |
| A12-D4 | Rejected | K/V LDS stride `D + 8` | Static counts unchanged, forced profile regressed to `12443.569 us`; keep `D + 16`. |
| A12-D5 | Rejected | `Bc=64` raw K/V staging | Not selected; forced profile slower than `Bc=32` and introduced nonzero bank conflicts. |
| A12-D6 | Kept | Made staged path normal `(128,32,8)` and removed sentinel/dead non-staged code | Simplified source, preserved codegen/profile properties. |
| A13-A | Kept partial | Staged autotune around `Br=64` | Keep `(64,32,8)` plus `(128,32,8)` fallback; remove unselected `(64,64,8)`, `(64,32,16)`, and `(32,32,8)`. |

## Logs And Reasoning

### Kept Structural Wins

- Register output accumulator was the first major structural win: moving `Oi` from fp16 LDS to fp32 registers lifted `S=4096` from roughly `10 TFLOPS` into the mid/high teens, then 8-wave raised it further.
- `Si` padding to `Bc + 8` is non-negotiable for current `Bc=32` paths: it took measured HIP `LDSBankConflict` from nonzero to `0.0` without changing the algorithm.
- Raw K/V LDS staging fixed the V-side global scalar-load problem: fwd-symbol `global_load_d16_u8`/`global_load_d16_hi_u8` dropped to `0/0` while scratch stayed `0`.
- Making the staged path the normal config avoided sentinel/autotune clutter and left only the code path that matters for the fp8 D=128 kernel.
- `(64,32,8)` appears to be a better staged shape than `(128,32,8)` because it lowers VGPR and output-fragment pressure while preserving `Bc=32` LDS behavior.

### Durable Findings

- The remaining core bottleneck is the `Si` score/probability LDS round trip plus serial row softmax work, not K/V quantization or output accumulator storage.
- HND (`B,H,S,D`) is the optimized layout for this kernel family; NHD should be handled only if integration requires it.
- Keep zero/negligible LDS bank conflicts as a hard target. AITER and the best HIP paths both achieve `LDSBankConflict=0.0`.
- K/V quantization is `O(BHSD)` and stays small at large S; prepacked and end-to-end timings should still be reported separately.
- More registers can be good on gfx1151 when they remove LDS traffic, but VGPR pressure still matters: `(64,32,8)` currently beats `(128,32,8)` by reducing output-fragment pressure.
- `Bc=32` is the safest current K/V tile width because it preserves the known clean LDS pattern.

### Rejected Paths Worth Remembering

- Standalone V transpose/prepack did not pay. It should only be reconsidered inside a new register-P/PV forward layout.
- Smaller `Si` or K/V padding is not automatically better. `Si + 4` and K/V `D + 8` both preserved or reported low conflicts but regressed runtime, showing dynamic LDS phase/scheduling matters.
- Row ownership alone is not useful if `Si` remains and V access worsens. A viable row-owned kernel must remove `Si` materialization or have a verified register-P/PV mapping.
- Q LDS staging on top of the row-owned/`Si` scaffold is not viable due to LDS/VGPR/private-segment pressure.
- K-only LDS staging is not enough; K staging became worthwhile only when paired with V staging.
- `Bc=64` staging is currently worse because it loses the clean bank-conflict behavior and does not beat `(64/128,32,8)` when forced.
- The first wave-shuffle register-P/PV attempt failed correctness. Before retrying full attention, build a minimal WMMA-fragment mapping test that checks C-fragment row/column extraction into a PV A-fragment.

### Do-Not-Repeat (Unless New Preconditions)

- Do not re-add temporary sentinel configs for the staged path; normal configs should map directly to real staged instantiations.
- Do not re-add standalone VT prepack unless a new forward dataflow changes the V access preconditions.
- Do not re-add row-owned skeletons that keep full `Si` materialization and worsen V access.
- Do not re-add Q LDS staging on the rejected row-owned/`Si` scaffold.
- Do not re-add K-only staging on the current dataflow.
- Do not re-add `Bc=64` staging unless a new LDS layout removes its conflicts and forced profile beats `Bc=32`.
- Do not add Q quantization before the forward kernel removes the score-LDS/softmax bottleneck.
- Do not copy CDNA MFMA-specific CK implementation details for gfx1151; use CK only for transferable dataflow/layout ideas.

## Next Work

1. Add a small WMMA fragment-layout validation fixture before attempting another register-P/PV attention kernel.
2. Revisit register-P/PV only after the fragment mapping is correct; the goal is to remove the `Si` score/probability LDS round trip and distribute row softmax work.
3. Only after register-P/PV exists, revisit load/compute overlap or V-specific tile-local layout. Do not tune waitcnt/barrier placement in isolation.

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

## Reference

- `~/amd-llvm-project/`, especially `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst` - hipcc source code
- `~/rdna35-isa-markdown/`
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py` - AITER FlashAttention forward reference
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp` - CK-Tile QR FMHA dataflow reference
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp` - CK-Tile Q-load, K/V LDS, and V-shuffle policy reference
- `~/rocm-libraries/projects/composablekernel/dispatcher/codegen/fmha/fmha_arch_specs.json` - CK FMHA tile/wave/warp config reference
- `~/rocm-libraries/shared/rocroller/docs/src/LDSSwizzling.md` - LDS swizzle reference for bank-conflict reasoning
- `~/SageAttention/sageattention/triton/quant_per_block.py` - SageAttention per-block int8 quantization reference
- `~/SageAttention/sageattention/triton/quant_per_thread.py` - SageAttention per-thread int8 quantization reference
- `~/SageAttention/sageattention/triton/attn_qk_int8_per_block.py` - SageAttention Triton int8-QK attention reference
- `~/SageAttention/sageattention/core.py` - SageAttention public API, layout, smoothing, and tolerance context
