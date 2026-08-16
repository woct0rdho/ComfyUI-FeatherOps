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

The D64 kernel uses one 128-thread workgroup per head and ownership tile. Short and irregular inputs use one fused kernel with an owner-tile axis sized to the larger of the Q and KV tile counts. When both query and KV lengths are at least 4096, the launcher uses separate compile-time KV-only and Q-only kernels on the current stream so each phase gets its own register allocation.

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

The complete launch is a Delta helper followed by either the fused seven-GEMM kernel or ordered KV-only and Q-only phase kernels. All launches use the caller's current stream. No initialization, workspace clear, FP32-to-FP16 gradient conversion, or reduction helper remains in the current path.

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

## Backward Accuracy Tiers

Backward quantization is judged primarily by gradient direction, not by Rel-L2 alone. The SageBwd validation in `~/sageattention-autotune/tests/test_sagebwd_triton.py` uses `1 - cosine < 0.003` as its direction gate, with Rel-L2 targets of `0.07` for dQ/dK and `0.06` for dV. Feather experiments use the following tiers:

| Tier | Required evidence |
| --- | --- |
| Directional candidate | All finite and minimum flattened gradient cosine at least `0.997` (`1 - cosine < 0.003`) for dQ, dK, and dV |
| Preferred numerical candidate | Directional tier plus maximum Rel-L2 at most `0.10` for each gradient and reported norm-ratio error |
| Exploratory numerical candidate | Directional tier plus maximum Rel-L2 below `0.20`; this can be retained for a measured speed win, but remains explicitly approximate |
| Production exact baseline | Existing public elementwise contract and the exact leader's approximately `0.0003` Rel-L2 behavior |

The relaxed exploratory tier is for internal candidate comparison and does not silently change the public API contract. A candidate that passes the directional tier but exceeds `0.10` Rel-L2 is not rejected solely for that reason; complete timing and resource evidence decide whether it is worth retaining.

## Current Benchmarks

The current matrix uses B1, contiguous HND inputs, equal Q/KV lengths, H `{16,32}`, N `{2048,4096,8192}`, eight warmups, and 30 timed samples per provider. AITER is the FlashAttention Triton backward control. The values below are recomputed with `F7`, not the older five-GEMM field.

Every H/N row in the current D64 matrix is shown. The current direct-output Feather kernel does not allocate gradient workspaces.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 2048 | 26.215 | 27.635 | 1.054x |
| HND | 16 | 4096 | 26.467 | 28.340 | 1.071x |
| HND | 16 | 8192 | 26.447 | 29.696 | 1.123x |
| HND | 32 | 2048 | 25.585 | 27.836 | 1.088x |
| HND | 32 | 4096 | 25.403 | 29.221 | 1.150x |
| HND | 32 | 8192 | 25.106 | 29.784 | 1.186x |

The public D64 contract screen additionally covers `(NQ,NKV) = (33,35), (65,67), (65,129)` and batch two. The private candidate screen covers `(65,64), (129,65), (256,256), (257,257), (512,513)`, plus batch two and a cancellation pattern.

## Resource and Profile Results

The symbol-matched linked production image has the following gfx1151 profile:

| Symbol/workgroup | Logical VGPRs | Allocated VGPRs | SGPRs | LDS | Private/spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| D64 fused main, 128 threads | 178 | 192 | 45 | 13,312 B | 0 / 0 |
| D64 KV-only long phase, 128 threads | 175 | 192 | 33 | 13,312 B | 0 / 0 |
| D64 Q-only long phase, 128 threads | 162 | 168 | 34 | 13,312 B | 0 / 0 |
| Delta helper | 66 | 72 | 10 | 0 B | 0 / 0 |

The linked fused symbol contains 2,748 static instructions: 40 WMMA, 1,540 VALU, 941 SALU, 32 `v_perm_b32`, 16 cross-half permutes, 72 LDS loads, 40 LDS stores, 54 global loads, and 96 global stores. The linked KV-only and Q-only symbols contain 1,529 and 1,242 static instructions respectively, with 16 WMMA each for KV and 24 WMMA for Q. The six-case production correctness snapshot reports maximum relative L2 `0.00030144`, minimum per-head cosine `0.99999988`, and maximum absolute error `0.00119209`. The earlier full-image counter pass reports occupancy `46.62%`, `426.889 M` VALU instructions, `77.611 M` LDS instructions, `12.457 B` wave cycles, `2.440 B` wait-any cycles, `1.076 B` barrier waits, LDS latency `106.73` cycles, LDS conflict metric `24.32`, and ALU stalled by LDS `2.10%` for the fused baseline.

The post-integration H16/N4096/D64 production result is `8.486947 ms` (`28.340 TFLOPS` under F7). The private exact leader result was `8.495935 ms` (`28.310 TFLOPS`), and the former campaign target was `8.417913 ms`; none is a fixed promotion gate. A candidate may be retained at any margin when repeated paired measurements provide credible evidence that it is faster. The component result remains the phase attribution for the exact leader:

| Portion | GEMM equivalents | Net ms | Normalized TFLOPS |
| --- | ---: | ---: | ---: |
| KV-owned | 4 | 4.980566 | 27.595 |
| Q-owned | 3 | 3.399451 | 30.322 |
| Full seven-GEMM image | 7 | 8.495935 | 28.310 |

The KV portion is the slower side per nominal GEMM. It remains the first target for phase-specific work, but an experiment must reduce complete time rather than only static resources.

The compile-time phase-fission fixture built from the linked exact source produces three symbol-matched kernels:

| Exact symbol | Logical / allocated VGPRs | SGPRs | LDS | Private/spills | Static instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fused KV+Q | 178 / 192 | 45 | 13,312 B | 0 / 0 | 2,748 |
| KV-only | 175 / 192 | 33 | 13,312 B | 0 / 0 | 1,529 |
| Q-only | 162 / 168 | 34 | 13,312 B | 0 / 0 | 1,242 |

Fission is bit-identical to the fused fixture across all six research correctness cases. Fifty-sample complete timing, including Delta and both fission launches, gives:

| H | N | Fission ms | Fission / matched fused | Fission / production |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2048 | 2.168165 | 0.9994x | 0.9918x |
| 16 | 4096 | 8.390506 | 1.0122x | 1.0106x |
| 16 | 8192 | 32.261520 | 1.0161x | 1.0184x |
| 32 | 2048 | 4.371885 | 1.0052x | 0.9994x |
| 32 | 4096 | 16.644175 | 1.0105x | 1.0105x |
| 32 | 8192 | 65.940956 | 1.0136x | 1.0139x |

The six-row geometric mean is `1.00950x` versus the matched fused image and `1.00739x` versus current production. A subsequent 120-pair confirmation gives a `1.01264x` geometric speedup across the four N4096/N8192 rows. Every long row wins all 12 ten-sample timing blocks, and paired bootstrap 95% intervals range from `[1.00867, 1.01304]` at H16/N4096 to `[1.01375, 1.01496]` at H32/N8192. N2048 remains inconsistent between runs, so the retained selector requires both Q and KV lengths to be at least 4096.

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
| FP8 campaign baseline freeze | Completed: linked metadata/disassembly, six-case correctness, six-row timing matrix, and H16/N4096 component timing recorded |
| Raw high-byte E5M2 truncation | Accepted as an experimental finite-only primitive: all 65,536 GPU encodings match `half_bits >> 8`; signed zero, infinity, sign, and non-increasing finite magnitude are verified |
| Explicit two-permute E5M2 pack | Accepted for candidate integration: bit-exact, 42.1% faster than scalar shift/OR pack and 29.9% faster for pack/decode round-trip in the isolated fixture; 15 versus 16 logical VGPRs |
| Vectorized P-only truncation | Rejected after six-row complete timing: directionally valid, but `0.979607x` geometric mean versus the matched exact image and slower on every row; it adds 68 static instructions, 70 VALU operations, and 10 `v_perm_b32` while remaining in the 192-VGPR class |
| Vectorized dS-only truncation | Rejected after six-row complete timing: directionally valid, but `0.996503x` geometric mean versus the matched exact image and no row wins; it adds 21 static instructions and 18 `v_perm_b32` while remaining in the 192-VGPR class |
| Vectorized P+dS truncation | Rejected before timing: maximum Rel-L2 `0.189359` is exploratory-eligible, but minimum flattened cosine `0.996022` and minimum per-head cosine `0.994891` fail the primary direction gate; 172 logical VGPRs still allocate 192 and SGPRs rise to 49 |
| Transient P/dS pack followed by immediate decode | Rejected for the current topology: no LDS/global traffic is removed, no allocation class is crossed, and software pack/decode work increases complete time |
| Compile-time KV/Q phase fission | Accepted and integrated for NQ/NKV at least 4096: bit-identical outputs, Q-only falls from 192 to 168 allocated VGPRs, the 120-pair long-row confirmation is `1.01264x` geometric with 48/48 timing-block wins, and short rows remain fused |
| Old atomic FP32 D64 kernel | Removed; no source, selector, workspace, or linked image remains |
| `owned`, `paired`, automatic selection, and duplicate D64 reference paths | Removed from production |
| FP8 P/dS truncation | Rejected for the current direct-fragment topology; revisit only if a future schedule creates persistent P/dS storage whose byte reduction can repay software conversion |
| Six-GEMM N-squared workspace design | Deferred; it requires approximately 256 MiB intermediate storage and 512 MiB extra traffic at the reference shape |
| Stock AITER CK backward | Blocked on gfx1151 receipt generation; retained only as external evidence |

## Next Work

- Keep the `168/168` forward contract and the current six-row D64 backward matrix as regression gates.
- Design D128 as a separate seven-GEMM effort; do not restore the deleted reference kernel.
- Continue phase-specific KV-only work. Its measured `175/192` VGPR profile leaves a possible allocation-class reduction as the next high-upside exact optimization.
- Preserve HND, natural-log LSE, current-device validation, and current-stream execution while changing the internal schedule.

### Proposed Sequence

- Isolate the KV-only symbol and target a reduction to at most 168 logical VGPRs or a measured LDS/traffic reduction without changing exact numerics.
- Rebuild and run the full resource, correctness, public-contract, forward-regression, and six-row complete-timing gates for any KV-only change; retain it only with repeated paired evidence.
- Use complete timing as the gate. Smaller improvements may be retained when repeated paired samples show credible positive evidence and the six-row matrix has no material regression. No isolated phase result, lower LDS byte count, or one fast observation is sufficient for promotion.

### Expected Outcomes and Stop Conditions

The direct P/dS FP8 campaign is closed for the current topology: the exact numerical path remains production, and phase fission is the retained exact optimization. Future work should reduce the KV-only resource class or traffic; do not reopen approximate state unless a future schedule creates persistent P/dS storage whose byte reduction can repay conversion.

Do not add dynamic scaling, approximate exponentials, FP8 WMMA assumptions, or an ABI selector for experimental quantization. Raw truncation is finite-only: 510 of 2,046 FP16 subnormals become signed zero and 510 low-payload NaNs become infinity. Attention integration therefore requires finite P/dS inputs and the explicit finite-output gate. Any successful candidate must remain an internal compile-time variant until it passes symbol metadata, the directional accuracy tier, complete timing, and the full D64 matrix.

## Artifacts

- Forward matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Current D64 timing run: `/tmp/feather_current_bwd_matrix.json`.
- Private D64 leader timing: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/timing_phases_clean.json`.
- Private D64 leader correctness: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/correctness.json`.
- Private D64 metadata/disassembly: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/build/attn_stride20_all_lds13312_hip_ext/`.
- Historical packed FP8 screen: `~/tmp/feather_attn/candidates/seven_gemm_packed/qualification_summary.json`.
- FP16/FP8 conversion reference: `kernel/hip/hip_kernel.cu`.
- Active FP8 campaign: `~/tmp/feather_attn/e5m2_truncation_campaign/`.
- Phase-fission fixture and paired evidence: `~/tmp/feather_attn/phase_fission_campaign/`.
