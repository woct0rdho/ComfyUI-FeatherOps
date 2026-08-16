# FeatherAttn gfx1151 Backward Kernel

## Status

The current production backward surface contains one explicit implementation:

| Head dimension | Explicit implementation | Current status |
| ---: | --- | --- |
| 64 | `implementation="fused"` | Seven-GEMM direct-output kernel, linked and contract-tested |

The saved-state contract is contiguous HND `[B,H,N,D]`, FP16 `(Q,K,V,O,dO)`, FP32 natural-log LSE, dense non-causal attention, equal query/KV heads, current-device validation, and current-stream execution. The current operator returns FP16 `dQ`, `dK`, `dV`, and FP32 `Delta`.

## Public Contract and Source Layout

The public wrapper is:

```text
feather_attn_backward(Q, K, V, O, LSE, dO, *, implementation, sm_scale=None, ...)
```

The backward arguments are fixed HND tensors. There is no backward `layout` selector and no full-gradient workspace argument. The seven-GEMM kernel owns each output element and writes the final FP16 gradients directly; this removes the old FP32 accumulation buffers, atomic updates, clear kernel, and conversion kernels.

| File | Role |
| --- | --- |
| `kernel_attn/hip/hip_kernel.py` | Explicit fused-only Python selector and extension source list |
| `kernel_attn/hip/hip_kernel.cpp` | Shared validation, fused dispatch, current-stream lookup, and Torch registration |
| `kernel_attn/hip/hip_kernel.h` | Raw forward/backward launch ABI |
| `kernel_attn/hip/featherattn_bwd_fused_d64.cu` | D64 Delta helper and seven-GEMM device launcher |

## Current Production Design

The D64 kernel uses one 128-thread workgroup per head and ownership tile. The grid has one owner-tile axis sized to the larger of the Q and KV tile counts, so irregular query/KV lengths share one launch without a second selector.

| Parameter | Value |
| --- | ---: |
| Head dimension | 64 |
| Workgroup | 128 threads, four wave32 waves |
| Ownership tile | 64 rows |
| Inner WMMA tile | 16 rows/elements |
| KV storage | FP16, XOR-swizzled LDS rows |
| K/Q/dO transpose stride | 20 FP16 elements |
| Main LDS | 13,312 bytes with lifetime-local aliasing |
| Gradient ownership | Unique output ownership, no atomics |
| Output | Direct FP16 `dQ`, `dK`, `dV` stores |
| Saved state | Natural-log FP32 LSE plus stream-local FP32 Delta |

The KV-owned portion stages V in LDS, caches K rows, cooperatively stages each Q and dO tile into the stride-20 transpose layout, and computes four GEMM-equivalent operations: QK score, dO/V dP, dS/Q dK, and P/dO dV. The Q-owned portion stages K and V in two 16-row tiles, caches Q and dO rows, and computes three operations: QK score, dO/V dP, and dS/K dQ. The C-to-A WMMA handoff converts each FP32 C fragment to the FP16 A-fragment layout with lane permutes.

The complete launch is a Delta helper followed by the seven-GEMM main kernel. Both launches use the caller's current stream. No initialization, workspace clear, FP32-to-FP16 gradient conversion, or reduction helper remains in the current path.

## D128 Scope

D128 is intentionally absent from the current backward implementation. The former scalar D128 reference kernel and its four-launch reconstruction path were removed because the production objective is a new seven-GEMM D128 design, not preservation of a slow duplicate. No D128 backward benchmark or resource claim is made here.

## Throughput Convention

All backward throughput uses the seven-GEMM convention:

```text
F7 = 7 * 2 * B * H * NQ * NKV * D FLOPs
TFLOPS = F7 / elapsed_seconds / 1e12
Feather / AITER = AITER elapsed time / Feather elapsed time
```

A ratio above `1.000x` means the Feather elapsed time is lower. Complete timing includes the Delta launch, main launch, initialization outside the measured call only when shared by both providers, synchronization, and all helper work inside the provider call. Outputs and Delta were preallocated before the current matrix timing.

## Current Benchmarks

The current matrix uses B1, contiguous HND inputs, equal Q/KV lengths, H `{16,32}`, N `{2048,4096,8192}`, eight warmups, and 30 timed samples per provider. AITER is the FlashAttention Triton backward control. The values below are recomputed with `F7`, not the older five-GEMM field.

Every H/N row in the current D64 matrix is shown. The current direct-output Feather kernel does not allocate gradient workspaces.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 2048 | 25.986 | 27.620 | 1.063x |
| HND | 16 | 4096 | 26.704 | 28.167 | 1.055x |
| HND | 16 | 8192 | 26.493 | 29.226 | 1.103x |
| HND | 32 | 2048 | 25.726 | 27.748 | 1.079x |
| HND | 32 | 4096 | 25.395 | 29.050 | 1.144x |
| HND | 32 | 8192 | 25.342 | 29.242 | 1.154x |

The public D64 contract screen additionally covers `(NQ,NKV) = (33,35), (65,67), (65,129)` and batch two. The private candidate screen covers `(65,64), (129,65), (256,256), (257,257), (512,513)`, plus batch two and a cancellation pattern.

## Resource and Profile Results

The symbol-matched private leader that was promoted into the repository has the following gfx1151 profile:

| Symbol/workgroup | Logical VGPRs | Allocated VGPRs | SGPRs | LDS | Private/spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| D64 seven-GEMM main, 128 threads | 178 | 192 | 46 | 13,312 B | 0 / 0 |
| Delta helper | 66 | 72 | 10 | 0 B | 0 / 0 |

The private six-case correctness screen reports maximum relative L2 `0.00030204`, minimum cosine `0.99999988`, and maximum absolute error `0.00121403`. The full-image counter pass reports occupancy `46.62%`, `426.889 M` VALU instructions, `77.611 M` LDS instructions, `12.457 B` wave cycles, `2.440 B` wait-any cycles, `1.076 B` barrier waits, LDS latency `106.73` cycles, LDS conflict metric `24.32`, and ALU stalled by LDS `2.10%`.

The private H16/N4096/D64 complete result is `8.495935 ms` (`28.310 TFLOPS` under F7). The component result is:

| Portion | GEMM equivalents | Net ms | Normalized TFLOPS |
| --- | ---: | ---: | ---: |
| KV-owned | 4 | 4.980566 | 27.595 |
| Q-owned | 3 | 3.399451 | 30.322 |
| Full seven-GEMM image | 7 | 8.495935 | 28.310 |

The KV portion is the slower side per nominal GEMM. It remains the first target for any FP8 experiment, but the experiment must reduce complete time rather than only LDS bytes.

- The current D64 leader is direct-output, 13,312-byte LDS, 178 logical/192 allocated VGPRs, and no private memory or spills. Its P and dS values currently remain in FP32 C fragments and go through the exact FP32-to-FP16 C-to-A handoff.
- The forward Q path already stores E5M2 bytes through a `uint8x16` lane-vector pattern in `featherattn_fwd_kernel.h`. That path uses RNE and log2 prescaling for forward Q, so its accuracy policy cannot be copied blindly to backward P or dS.
- `kernel/hip/hip_kernel.cu` contains the compact `fp8e5m2x4_to_half2x2` decode: one packed `uint32_t` input produces two packed half2 outputs through two `v_perm_b32` operations. This is a useful decode primitive and an instruction-count reference, not proof that a backward hot-loop decode will be profitable.
- The archived seven-GEMM packed screen measured exact `167` logical VGPRs, `168` allocated VGPRs, `20,480` bytes LDS, and 8 permutes for the unquantized control. P-only used `165/168` VGPRs and 19,456 bytes LDS but 40 permutes; dS-only used `169/192` and 44 permutes; P+dS used `164/168`, 18,432 bytes LDS, and 76 permutes. None changed the useful allocation class in the tested schedule.
- The same screen found truncating P or dS numerically admissible under the broad public tolerance but not preferred: P maximum relative L2 was about `0.10045`, dS about `0.10021`, and P+dS about `0.17681`. Exact FP16 had maximum relative L2 about `0.0002318` in that screen. This is a strong reason to test truncation as an optimization candidate, not silently use it.
- Prior fixed-scale, power-of-two-scale, and linear-scale attempts consumed conversion/register margin and were rejected. gfx1151 has no useful native FP8 WMMA or native E5M2 conversion instruction, so the byte savings compete directly with VALU and `v_perm_b32` work.

## Historical D64 Campaign Evidence

These results are retained as provenance for the selector and topology decisions. They are not current production timings because those kernels used the removed accumulation/workspace ABI or private source images.

| Comparison or candidate | Result | Resource/evidence summary | Decision |
| --- | --- | --- | --- |
| Triton / CK external | `0.2333805x` geometric ratio; `0/6` frozen-row Feather wins | CK backward receipt omits `--targets gfx1151` in the default build | Retained as blocked external evidence |
| Triton / owned | `0.5459785x` geometric ratio; `0/6` frozen-row wins | Qualified at 126 logical / 144 physical VGPRs, 46 SGPRs, 17,152 B LDS | Removed from production |
| Triton / fused workspace kernel | `0.2039779x` geometric ratio; `0/6` frozen-row wins | 174 logical / 192 allocated VGPRs, 54 SGPRs, 17,152 B LDS, FP32 caller workspace and atomic updates | Removed and replaced by direct output |
| Paired ownership | `1.41%` improvement over owned | 191 logical / 192 physical VGPRs and 29,696 B LDS | Rejected as insufficient |
| Native atomics | Site-local owned result about `12.0452 ms` | Lost every frozen row to Triton despite qualified atomics | Retained only as evidence |
| Step 11 cooperative staging/caching | `10.9995437 ms` | 178 logical VGPRs, 47 SGPRs, 24,576 B LDS, zero spills | Superseded by the seven-GEMM leader |
| Five-GEMM, partition workspace, M16, atomic-reduction variants | No promotion gate cleared | Extra workspace, launches, or reduction cost dominated the complete path | Deferred or rejected |

The historical owned variants `owned-order`, `owned-lifetime`, `owned-exp2`, `owned-half-pds`, `owned-qreuse`, and `owned-q32` were individually rejected. Their measurements remain in the campaign artifacts, while the current source contains none of their selectors or workspace paths.

## Accepted, Rejected, and Deferred Work

| Work | Decision |
| --- | --- |
| Direct-output seven-GEMM D64 schedule | Accepted and linked as the current production kernel |
| Unique output ownership, C-to-A handoff, stride-20 transpose, lifetime-local LDS | Accepted mechanisms |
| Old atomic FP32 D64 kernel | Removed; no source, selector, workspace, or linked image remains |
| `owned`, `paired`, automatic selection, and duplicate D64 reference paths | Removed from production |
| FP8 P/dS truncation | Deferred to the gated plan above; not in current production |
| Six-GEMM N-squared workspace design | Deferred; it requires approximately 256 MiB intermediate storage and 512 MiB extra traffic at the reference shape |
| Stock AITER CK backward | Blocked on gfx1151 receipt generation; retained only as external evidence |

## Next Work

- Keep the `168/168` forward contract and the current six-row D64 backward matrix as regression gates.
- Design D128 as a separate seven-GEMM effort; do not restore the deleted reference kernel.
- Evaluate FP8 E5M2 only through the isolated truncation, vectorized pack/decode, resource, correctness, and complete-timing gates above.
- Preserve HND, natural-log LSE, current-device validation, and current-stream execution while changing the internal schedule.

### Proposed Sequence

- Freeze the current baseline. Record symbol-matched metadata, disassembly counts, six-case exact correctness, full H/N timings, and complete H16/N4096 timing. Keep the admission target below `8.417913 ms` and retain the seven-GEMM F7 convention.
- Build an isolated truncation fixture. For an FP16 input, bit-cast to `uint16_t` and take the high byte as E5M2: `e5m2 = half_bits >> 8`. Test all 65,536 FP16 encodings, including signed zero, subnormals, infinities, NaNs, and negative values. Define the treatment of exponent-zero and non-finite values before integrating.
- Vectorize the pack path. Start with eight half values per lane and pack two `uint32_t` words containing eight E5M2 bytes. Compare a shift/pack sequence against `__builtin_amdgcn_perm` byte assembly. Use the `fp8e5m2x4_to_half2x2` layout as the decode oracle. Measure VALU, permute, LDS byte traffic, and register counts in a standalone fixture before inserting the code into attention.
- Try P-only first. P has the lower historical allocation result and is consumed by dV in the KV-owned portion and by dQ in the Q-owned portion. Replace only the current C-to-A P handoff with truncating FP8 pack plus vectorized decode. Keep dS exact. Do not add per-tile scales in the first candidate.
- Test only real reuse. A transient pack followed immediately by a decode is a win only if it reduces live VGPRs or replaces an existing expensive conversion. If it only adds pack and permute instructions, close the candidate. The next version may store packed P in the existing lifetime-local LDS reservation, but only if the stored bytes eliminate a global/LDS round trip or cross a resource class.
- Try dS-only second. Use the same path for dS, preserving exact P. The old screen suggests dS is more likely to hit the 192-VGPR class, so this candidate must show a resource crossing rather than just a smaller LDS number.
- Try P+dS only after both single-state tests. Keep the two state layouts disjoint in lifetime and alias them where possible. Do not carry two decoders or two packed representations simultaneously unless metadata proves the added code still fits the current resource gates.
- Run correctness before timing. Require finite outputs, maximum relative L2 at most `0.01`, minimum per-head cosine at least `0.9995`, norm-ratio error at most `0.01`, and the existing public allclose gate across the six exact cases and the cancellation case. The broad `0.10` allclose threshold is not sufficient for promotion because the current exact leader is much more accurate.
- Check symbol-scoped resources and ISA. Require at most `192` allocated VGPRs, at most `13,312` bytes LDS unless the smaller LDS crosses a measured occupancy class, zero private memory, zero spills, and no materially higher permute/VALU count without a corresponding timing reduction. A candidate that remains in the same allocation class and adds more than the measured conversion work is closed.
- Use complete timing as the gate. First require at least a repeatable `5%` complete improvement over the current D64 leader and a result below `8.417913 ms` at H16/N4096. Then run every D64 H/N row in the table. No isolated kernel-phase result, lower LDS byte count, or single `8.385254 ms` observation is sufficient for promotion.

### Expected Outcomes and Stop Conditions

The highest-probability outcome is that direct P/dS FP8 compression fails to pay: the current leader does not store these states in LDS, and gfx1151's permute-based decode can consume the saved bytes. If the vectorized truncation fixture shows no allocation-class crossing, stop before a full attention integration. If P-only reaches a lower class with preferred numerics, continue to dS-only; otherwise keep the exact current kernel.

Do not add dynamic scaling, approximate exponentials, FP8 WMMA assumptions, or an ABI selector for experimental quantization. Any successful candidate must remain an internal compile-time variant until it passes symbol metadata, exact correctness, complete timing, and the full D64 matrix.

## Artifacts

- Forward matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Current D64 timing run: `/tmp/feather_current_bwd_matrix.json`.
- Private D64 leader timing: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/timing_phases_clean.json`.
- Private D64 leader correctness: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/correctness.json`.
- Private D64 metadata/disassembly: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/build/attn_stride20_all_lds13312_hip_ext/`.
- Historical packed FP8 screen: `~/tmp/feather_attn/candidates/seven_gemm_packed/qualification_summary.json`.
- FP16/FP8 conversion reference: `kernel/hip/hip_kernel.cu`.
