# FeatherAttn gfx1151 Backward Kernel

## Status

The current production backward surface contains one explicit implementation:

| Head dimension | Explicit implementation | Current status |
| ---: | --- | --- |
| 64 | `implementation="fused"` | Seven-GEMM direct-output kernel, linked and contract-tested |

The saved-state contract is contiguous HND `[B,H,N,D]` or NHD `[B,N,H,D]`, FP16 `(Q,K,V,O,dO)`, FP32 natural-log LSE `[B,H,NQ]`, dense non-causal attention, equal query/KV heads, current-device validation, and current-stream execution. The current operator returns FP16 `dQ`, `dK`, `dV` in the input layout and FP32 `Delta` in logical `[B,H,NQ]` order.

The 18-row dual-layout performance matrix is an optimization target, not a support whitelist. Query and KV lengths remain independent arbitrary positive values within the existing signed-int32 dimension and addressability checks. Runtime ceil-divided tile counts and row guards handle tails. Lengths do not create per-length kernel specializations, and each selected image has fixed per-workgroup VGPR/LDS requirements. A bounded long NHD self-attention range may select the exact HND-copy path; every other valid shape uses the direct layout image.

## Public Contract and Source Layout

The public wrapper is:

```text
feather_attn_backward(Q, K, V, O, LSE, dO, *, implementation, layout="HND", sm_scale=None, ...)
```

The backward arguments are contiguous HND or NHD tensors selected by `layout`; there is no caller workspace argument. The seven-GEMM kernel owns each output element and writes the final FP16 gradients directly; this removes the old FP32 accumulation buffers, atomic updates, clear kernel, and conversion kernels. A bounded NHD long-row dispatch uses internal FP16 HND copies as described below.

| File | Role |
| --- | --- |
| `kernel_attn/hip/hip_kernel.py` | Explicit fused-only selector, bounded NHD-to-HND dispatch, and extension source list |
| `kernel_attn/hip/hip_kernel.cpp` | Shared validation, fused dispatch, current-stream lookup, and Torch registration |
| `kernel_attn/hip/hip_kernel.h` | Raw forward/backward launch ABI |
| `kernel_attn/hip/featherattn_bwd_fused_d64.cu` | D64 Delta helper and seven-GEMM device launcher |
| `benchmark_attn_hip_backward.py` | Paired AITER/Feather D64 primary-matrix benchmark |
| `test_attn_hip_backward.py` | Saved-state D64 correctness and arbitrary-length coverage |

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
| KV-only dO row stride | 72 FP16 elements |
| Main LDS | 13,312 B for fused/Q-only; 15,616 B for KV-only |
| Gradient ownership | Unique output ownership, no atomics |
| Output | Direct FP16 `dQ`, `dK`, `dV` stores |
| Saved state | Natural-log FP32 LSE plus stream-local FP32 Delta |

The KV-owned portion stages V in LDS, caches K rows, and cooperatively stages each Q and dO tile into the stride-20 transpose layout. Its compile-time long-sequence specialization also stages the exact dO rows with a padded 72-half stride, so dP reuses aligned LDS loads instead of loading dO from global memory a second time. It computes four GEMM-equivalent operations: QK score, dO/V dP, dS/Q dK, and P/dO dV. The Q-owned portion stages K and V in two 16-row tiles, caches Q and dO rows, and computes three operations: QK score, dO/V dP, and dS/K dQ. The C-to-A WMMA handoff converts each FP32 C fragment to the FP16 A-fragment layout with lane permutes.

The complete launch is a Delta helper followed by either the fused seven-GEMM kernel or ordered KV-only and Q-only phase kernels. General NHD shapes use direct strided D-vector addressing; their owner ordering is head-fast for partition-periodic `H % 16 == 0` cases, with the forward-derived LLC group permutation enabled when selected, and tile-fast otherwise. Valid contiguous B1 NHD self-attention with H in `{32,56}` and N in `[8192,16384]` instead copies Q/K/V/O/dO to HND, runs the unchanged HND image, and copies dQ/dK/dV back to the caller's NHD outputs. This exact path uses eight temporary FP16 tensors, from 256 MiB at H32/N8192 through 896 MiB at H56/N16384; an internal allocation failure falls back to direct NHD execution. All work stays on the caller's current stream. No initialization, workspace clear, FP32-to-FP16 gradient conversion, or reduction helper remains in either path.

## D128 Scope

D128 is now a production backward path for contiguous HND and NHD tensors. The former scalar D128 reference kernel and its four-launch reconstruction path remain deleted; correctness is checked against the saved-state PyTorch reconstruction and AITER outputs instead of restoring that slow path. The accepted implementation supports arbitrary positive dimensions under the existing signed-int32 checks, has no private memory, spills, or scratch instructions, and beats AITER across the complete target matrix.

### D128 Development Campaign

The D128 pipeline may differ from D64 where doubling the channel fragments changes the active bottleneck. The first direct lift establishes that reusing D64's persistent fragment topology is not resource-valid:

| Step | Result | Decision |
| --- | --- | --- |
| Mechanical D64 seven-GEMM lift to D128 | KV-only reaches 256 VGPRs with 13 spills/56 B private in HND and 5 spills/24 B in NHD; Q-only reaches 256 VGPRs with 268 spills/564 B in HND and 265 spills/552 B in NHD; fused images spill 282-287 values. KV LDS is 30,976 B and Q/fused LDS is 26,624 B. | Rejected before correctness or timing. Doubling persistent dK/dV, Q/dO, and dQ fragment banks is structurally invalid. |
| Independent D128 Q and paired-wave KV baseline | All `22/22` HND/NHD saved-state cases pass, including independent Q/KV tails, batch two, long asymmetric lengths, and uneven heads. After removing dead D64 grouped-owner state that caused 12 B of NHD private memory, both layouts link at 208 logical/216 allocated VGPRs and 27,136 B LDS for KV, 166 logical/168 allocated VGPRs and 13,312 B LDS for Q, with zero private memory and zero spills. Complete H16/H32 N4096 timing beats AITER in all four HND/NHD rows: HND reaches `1.884x`/`1.867x` and NHD `1.595x`/`1.566x`, for a `1.721798x` geometric speedup. | Accepted as the first performance baseline because it clears correctness, resource, and AITER gates; continue optimizing its `10.17-13.85` normalized TFLOPS. |
| Independent D128 baseline phase trace | At H16 N4096 HND, kernel trace medians are about 0.29 ms Delta, 20.9 ms KV, and 14.8 ms Q. KV is 58% of main-kernel time and its 27,136 B allocation limits it to four resident waves/SIMD. | Target KV first; test a real LDS residency transition and retain it only on complete latency. |
| D128 KV direct Q-row score loads | Remove the 4,352 B Q row copy while retaining the transpose view, matching the accepted D64 access pattern. KV falls to 205-206 logical/208 allocated VGPRs and 22,784 B LDS, remains scratch/spill-free, and crosses from four to five resident waves/SIMD. H16/H32 N4096 complete latency improves HND from 34.91/69.45 ms to 33.23/65.84 ms and NHD from 43.35/94.63 ms to 41.39/92.41 ms. | Accepted for HND and head-fast NHD: all four rows improve, geometric AITER speedup rises from `1.721798x` to `1.815338x`. |
| Universal direct dO-row reload | Removing the remaining row copy drops KV LDS to 18,432 B and crosses to seven resident waves/SIMD. HND H16/H32 improves again to 30.22/61.93 ms; NHD H16 improves to 40.42 ms, H32 is noisy at 94.62 ms, but tile-fast NHD H56 collapses to 330.47 ms versus AITER's 244.87 ms because every KV owner repeatedly reloads strided Q/dO rows. | Rejected as a universal policy. Retain the seven-wave form for locality-friendly cases only. |
| D128 KV compile-time row-cache policy | Emit 18,432 B direct/direct, 22,784 B direct-Q/cached-dO, and 27,136 B cached/cached variants. `22/22` correctness passes and every linked image has zero private memory/spills. At N4096 the selected policy reaches 30.60/61.77/106.01 ms HND and 39.59/91.53/278.20 ms NHD for H16/H32/H56. Cached rows recover H56 from 330.47 ms but still lose to AITER's 244.77 ms. | Keep the specializations provisionally, but H56 identifies owner order, not just LDS caching, as the next NHD problem. |
| NHD head-fast owner order for all head counts | Remove the `H % 16` restriction on head-fast block order. H56 N4096 NHD falls from 278.20 ms to 157.90 ms with identical arithmetic, beating AITER's 245.02 ms by `1.552x`. | Accepted. The former tile-fast fallback destroyed NHD locality. |
| Head-fast H56 direct-Q/cached-dO specialization | The 22,784 B five-wave image measures 157.66 ms versus 157.90 ms for the 27,136 B four-wave cached-Q/cached-dO image. | Latency-neutral; use the existing direct-Q specialization and avoid a third selected cache policy, but do not claim a performance gain. |
| D128 long-NHD execution through HND | Extend the existing complete-path transpose policy to D128 H16/H32/H56 at N4096-16384. Including five input and three output transposes, N4096 NHD improves from 39.59/91.53/157.66 ms to 38.34/76.01/121.72 ms for H16/H32/H56; all beat AITER, with `1.939530x` geometric speedup across the three rows. | Accepted. Keep direct NHD for non-target, asymmetric, and tail contract cases. |
| Persistent FP16 Q fragments in the D128 Q owner | Hoist eight Q fragments out of the KV loop while dO remains streamed. Q rises from 166 logical/168 allocated VGPRs to 212/216, falls from nine to seven resident waves/SIMD, and remains zero-private/zero-spill. H16/H32 N4096 HND complete latency drops from 30.60/61.77 ms to 25.01/49.75 ms, reaching about 19.3 normalized TFLOPS. | Accepted: the VMEM instruction reduction decisively outweighs the occupancy loss. |
| Persistent dO instead of Q | The symmetric dO-only image links at 197 logical/216 allocated VGPRs with zero private memory/spills, but complete latency is 25.19/50.01 ms at H16/H32 N4096 versus 25.01/49.75 ms for persistent Q. | Rejected; persistent Q is consistently faster at the same allocation/occupancy tier. |
| Persistent FP16 Q and dO together | Both banks fit at 250 logical VGPRs with 13,312 B LDS, zero private memory, zero spills, and no scratch instructions. H16/H32 N4096 HND complete latency improves again from 25.01/49.75 ms to 24.54/48.55 ms, reaching 19.60/19.82 normalized TFLOPS. | Accepted: complete latency wins despite the Q kernel moving to the 256-VGPR allocation tier. |
| Fold Delta into the persistent-Q+dO Q owner | Q can produce exact Delta before dQ and run ahead of KV, eliminating one launch and the standalone dO read, but the extra live reduction state yields 250 reported VGPRs plus 16 spills, 36 B private memory, and 12 linked scratch instructions in each layout. | Rejected before timing on the zero-private/zero-spill gate; retain the standalone Delta kernel. |
| Post-persistence D128 phase trace | At H16 N4096 HND, kernel-trace averages are 0.30 ms Delta, 17.22 ms KV, and 8.42 ms Q. KV is now about two thirds of main-kernel time. | Return to KV; test sharing probability from dK waves to paired dV waves to remove duplicated score WMMA and exponentials. |
| Share FP32 probability fragments from dK waves to dV waves | A 2,048 B LDS exchange removes duplicated dV-wave score WMMA and exponentials, but adds a workgroup barrier and serializes the two branch phases. KV links scratch-free at 213-214 logical VGPRs and 20,480 B HND LDS, reducing residency from seven to six waves/SIMD. H16/H32 N4096 complete latency regresses from 24.54/48.55 ms to 26.22/51.23 ms. | Rejected. The saved arithmetic does not repay branch serialization, the barrier, and lost residency. |
| Share P with only the branch barrier before dK/dV accumulation | Move the barrier immediately after P publication so dK and dV accumulation can overlap. The same zero-spill 20,480 B image still regresses to 26.76/52.31 ms at H16/H32 N4096. | Rejected; the six-wave LDS cost and synchronization remain more expensive than duplicated score work. |
| 128-row, 256-thread D128 Q owner | Double the Q owner tile and workgroup from 64/128 to 128/256. Each K/V tile is staged once for 128 query rows instead of twice, while the 246-logical-VGPR Q image remains zero-private/zero-spill with 13,312 B LDS. `22/22` correctness passes. H16/H32/H56 N4096 HND complete latency is 23.52/47.02/82.05 ms, about 20.45-20.52 normalized TFLOPS; H16/H32 improve from 24.54/48.55 ms. | Accepted: fewer Q blocks and half the K/V staging per query row improve all measured rows. |
| 256-row, 512-thread D128 Q owner | The 512-thread/256-row form remains zero-private/zero-spill at 246 VGPRs and 13,312 B LDS. It is tied/slightly faster at N4096 (`23.32/46.94/82.05` ms), then consistently faster at N8192 (`92.83/183.95/329.33` ms versus `93.20/186.28/331.97` ms) and reaches `378.56/761.07/1338.30` ms at N16384 for H16/H32/H56, all around 20.1-20.7 normalized TFLOPS. | Accepted as the final Q owner topology: the four-wave/SIMD residency limit is offset by halved block count and staging overhead. |
| Final D128 production qualification | The permanent backward screen passes `44/44` D64/D128 cases plus exact D64/D128 NHD-dispatch checks; forward remains `188/188`. The production D128 matrix wins `18/18` at `2.776753x` geometric (`2.803848x` HND, `2.749920x` NHD). Final linked Delta/KV/Q images all have zero private bytes, zero spills, and zero scratch instructions. | Accepted for production. D64 remains `18/18` at `1.175870x`, within run variance of the prior `1.182570x`. |

The final D128 implementation uses separate Delta, KV, and Q kernels. The 512-thread Q owner keeps exact FP16 Q/dO fragments in registers, accumulates dQ in FP32, and streams one 16-row K/V tile at a time. The 128-thread KV owner assigns dK and dV to separate wave pairs so each lane carries one eight-fragment FP32 output bank rather than both. E5M2 is not used: exact FP16 fits the zero-spill envelope, and no tested byte representation created a justified latency/resource transition.

## Throughput Convention

All backward throughput uses the seven-GEMM convention:

```text
F7 = 7 * 2 * B * H * NQ * NKV * D FLOPs
TFLOPS = F7 / elapsed_seconds / 1e12
Feather / AITER = AITER elapsed time / Feather elapsed time
```

A ratio above `1.000x` means the Feather elapsed time is lower. Complete timing includes the Delta launch, main launch, initialization outside the measured call only when shared by both providers, synchronization, and all helper work inside the provider call. Outputs and Delta were preallocated before the current matrix timing; the accepted NHD fallback's temporary allocations and eight transpose copies remain inside timing.

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

The primary matrix matches the forward plan: B1, contiguous HND and NHD inputs, D64, equal Q/KV lengths, H `{16,32,56}`, and N `{4096,8192,16384}`. `benchmark_attn_hip_backward.py` constructs one saved forward state outside timing, preallocates each provider's outputs, performs eight warmups, and collects 30 samples per provider in alternating order. AITER FlashAttention Triton backward is the control. Throughput uses `F7`; `Feather / AITER` is the median paired elapsed-time ratio.

Every H/N row in the primary D64 matrix is shown. HND wins `9/9` rows with a `1.204397x` geometric mean; NHD wins `9/9` with a `1.161139x` geometric mean. Overall, Feather wins `18/18` rows with a `1.182570x` geometric mean paired speedup. The previous all-direct NHD result was `7/9`, `1.094688x`; overall it was `16/18`, `1.147379x`. The bounded transpose dispatch therefore improves the complete dual-layout geometric mean by `1.03067x` while preserving the general direct NHD path.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 26.011 | 29.490 | 1.139x |
| HND | 16 | 8192 | 26.197 | 30.027 | 1.146x |
| HND | 16 | 16384 | 24.840 | 30.066 | 1.219x |
| HND | 32 | 4096 | 25.232 | 30.042 | 1.199x |
| HND | 32 | 8192 | 24.694 | 29.913 | 1.207x |
| HND | 32 | 16384 | 24.342 | 29.783 | 1.224x |
| HND | 56 | 4096 | 23.692 | 29.726 | 1.249x |
| HND | 56 | 8192 | 24.072 | 29.560 | 1.228x |
| HND | 56 | 16384 | 23.894 | 29.482 | 1.234x |
| NHD | 16 | 4096 | 25.535 | 28.338 | 1.110x |
| NHD | 16 | 8192 | 25.402 | 28.517 | 1.122x |
| NHD | 16 | 16384 | 25.065 | 27.426 | 1.094x |
| NHD | 32 | 4096 | 23.229 | 27.424 | 1.183x |
| NHD | 32 | 8192 | 23.192 | 24.337 | 1.044x |
| NHD | 32 | 16384 | 21.916 | 26.568 | 1.206x |
| NHD | 56 | 4096 | 23.315 | 28.853 | 1.230x |
| NHD | 56 | 8192 | 23.567 | 27.280 | 1.158x |
| NHD | 56 | 16384 | 21.333 | 28.307 | 1.327x |

### D128 Production Matrix

The final D128 matrix uses the production `benchmark_attn_hip_backward.py --head-dim 128` entry point with five warmups and ten alternating samples per provider. Feather wins `18/18` rows at `2.776753x` geometric speedup. HND is `2.803848x`; NHD, including all eight in-provider transposes, is `2.749920x`. The minimum row speedup is `2.377893x`.

Artifact: `/tmp/feather_bwd_d128_final_matrix_production.json`

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 7.004 | 20.307 | 2.876x |
| HND | 16 | 8192 | 7.158 | 20.087 | 2.787x |
| HND | 16 | 16384 | 7.409 | 20.358 | 2.740x |
| HND | 32 | 4096 | 7.150 | 20.679 | 2.873x |
| HND | 32 | 8192 | 7.217 | 20.386 | 2.804x |
| HND | 32 | 16384 | 7.323 | 20.123 | 2.746x |
| HND | 56 | 4096 | 6.971 | 20.255 | 2.861x |
| HND | 56 | 8192 | 7.281 | 20.201 | 2.782x |
| HND | 56 | 16384 | 7.209 | 19.992 | 2.771x |
| NHD | 16 | 4096 | 6.724 | 16.010 | 2.378x |
| NHD | 16 | 8192 | 6.508 | 17.793 | 2.722x |
| NHD | 16 | 16384 | 6.260 | 18.655 | 2.972x |
| NHD | 32 | 4096 | 6.444 | 16.130 | 2.507x |
| NHD | 32 | 8192 | 6.077 | 17.927 | 2.957x |
| NHD | 32 | 16384 | 5.921 | 18.646 | 3.146x |
| NHD | 56 | 4096 | 6.618 | 17.522 | 2.638x |
| NHD | 56 | 8192 | 6.822 | 18.659 | 2.742x |
| NHD | 56 | 16384 | 6.911 | 19.156 | 2.772x |

The D64 regression matrix remains `18/18` at `1.175870x` geometric speedup (`1.200022x` HND, `1.152203x` NHD). Its artifact is `/tmp/feather_bwd_d64_regression_matrix_after_d128.json`.

The public backward contract screen now covers both D64 and D128 over H `{16,32,56}`, batch two, the long-path selector boundary, and arbitrary asymmetric lengths through 16,385: `(4095,4097)`, `(4097,4099)`, `(8191,67)`, `(65,8193)`, `(16383,67)`, and `(65,16385)`. Dedicated H32/N8192 tests also require bit-exact `dQ/dK/dV/Delta` equivalence between the accepted transpose dispatch and the direct NHD image for both head dimensions. The private candidate screen additionally covers `(65,64)`, `(129,65)`, `(256,256)`, `(257,257)`, `(512,513)`, and a cancellation pattern.

## Resource and Profile Results

The symbol-matched linked production image has the following gfx1151 profile:

| Symbol/workgroup | Logical VGPRs | Allocated VGPRs | SGPRs | LDS | Private/spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND D64 fused main, 128 threads | 178 | 192 | 46 | 13,312 B | 0 / 0 |
| HND D64 KV-only long phase, 128 threads | 169 | 192 | 33 | 15,616 B | 0 / 0 |
| HND D64 Q-only long phase, 128 threads | 162 | 168 | 33 | 13,312 B | 0 / 0 |
| NHD D64 fused main, 128 threads | 178 | 192 | 50 | 13,312 B | 0 / 0 |
| NHD D64 KV-only long phase, 128 threads | 169 | 192 | 36 | 15,616 B | 0 / 0 |
| NHD D64 Q-only long phase, 128 threads | 163 | 168 | 38 | 13,312 B | 0 / 0 |
| HND Delta helper | 66 | 72 | 10 | 0 B | 0 / 0 |
| NHD Delta helper | 66 | 72 | 14 | 0 B | 0 / 0 |

The final D128 symbols in the linked gfx1151 image are:

| Symbol/workgroup | Logical / allocated VGPRs | SGPRs | LDS | Private/spills |
| --- | ---: | ---: | ---: | ---: |
| HND Delta | 81 / 96 | 10 | 0 B | 0 / 0 |
| NHD Delta | 81 / 96 | 14 | 0 B | 0 / 0 |
| HND KV direct Q/dO, 128 threads | 209 / 216 | 40 | 18,432 B | 0 / 0 |
| NHD KV direct Q/dO, 128 threads | 210 / 216 | 43 | 18,432 B | 0 / 0 |
| NHD KV direct Q/cached dO, 128 threads | 205 / 216 | 43 | 22,784 B | 0 / 0 |
| HND Q, 512 threads | 246 / 256 | 32 | 13,312 B | 0 / 0 |
| NHD Q, 512 threads | 246 / 256 | 34 | 13,312 B | 0 / 0 |

The entire linked D128 code object contains zero scratch load/store instructions. HND KV reaches seven waves/SIMD from both its 216-VGPR and 18,432-byte LDS limits. Q's 16-wave workgroup and 256-VGPR tier admit one workgroup per WGP, or four waves/SIMD. Final inspection artifacts are in `/tmp/feather_bwd_d128_final_inspect`.

The pre-NHD HND linked fused symbol contains 2,748 static instructions: 40 WMMA, 1,540 VALU, 941 SALU, 32 `v_perm_b32`, 16 cross-half permutes, 72 LDS loads, 40 LDS stores, 54 global loads, and 96 global stores. The linked dO-staged KV-only and unchanged Q-only symbols contain 1,489 and 1,242 static instructions respectively, with 16 WMMA for KV and 24 WMMA for Q. Relative to the pre-stage KV image, the new symbol removes eight static global `b128` loads and adds eight LDS `b128` loads; fused and Q-only code and resources remain unchanged. The six-case production correctness snapshot reports maximum relative L2 `0.00030144`, minimum per-head cosine `0.99999988`, and maximum absolute error `0.00119209`. The earlier full-image counter pass reports occupancy `46.62%`, `426.889 M` VALU instructions, `77.611 M` LDS instructions, `12.457 B` wave cycles, `2.440 B` wait-any cycles, `1.076 B` barrier waits, LDS latency `106.73` cycles, LDS conflict metric `24.32`, and ALU stalled by LDS `2.10%` for the fused baseline.

The post-dO-stage H16/N4096/D64 production matrix result is `8.419482 ms` (`28.567 TFLOPS` under F7). The private exact leader result was `8.495935 ms` (`28.310 TFLOPS`), and the former campaign target was `8.417913 ms`; none is a fixed promotion gate. A candidate may be retained at any margin when repeated paired measurements provide credible evidence that it is faster. The component result remains the phase attribution for the exact leader:

| Portion | GEMM equivalents | Net ms | Normalized TFLOPS |
| --- | ---: | ---: | ---: |
| KV-owned | 4 | 4.980566 | 27.595 |
| Q-owned | 3 | 3.399451 | 30.322 |
| Full seven-GEMM image | 7 | 8.495935 | 28.310 |

The KV portion is the slower side per nominal GEMM. It remains the first target for phase-specific work, but an experiment must reduce complete time rather than only static resources.

The pre-dO-stage compile-time phase-fission fixture built from the exact source produced three symbol-matched kernels:

| Pre-stage symbol | Logical / allocated VGPRs | SGPRs | LDS | Private/spills | Static instructions |
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

The six-row geometric mean was `1.00950x` versus the matched fused image and `1.00739x` versus then-current production. A subsequent 120-pair confirmation gives a `1.01264x` geometric speedup across the four N4096/N8192 rows. Every long row wins all 12 ten-sample timing blocks, and paired bootstrap 95% intervals range from `[1.00867, 1.01304]` at H16/N4096 to `[1.01375, 1.01496]` at H32/N8192. N2048 remains inconsistent between runs, so the retained selector requires both Q and KV lengths to be at least 4096.

The retained phase-local KV optimization stages each 16x64 dO tile twice during the existing cooperative load: once in the transpose layout needed by dV and once in a row-major layout needed by dP. A 72-half row stride rotates each aligned 32-byte row load by 16 bytes across the 128-byte LDS bank cycle. The additional 2,304 bytes are instantiated only for `kPhase == 1`; fused and Q-only symbols retain their original LDS layouts. The production image is bit-identical to the frozen phase-fission image at both selector boundaries and on the long path.

The linked-image confirmation used 120 alternating pairs per row and complete provider timing. Speedup is the frozen phase-fission time divided by the dO-staged production time:

| H | N | Phase-fission ms | dO-stage ms | Paired speedup | Bootstrap 95% CI | Winning blocks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4096 | 8.651481 | 8.424170 | 1.026446x | [1.023860, 1.029158] | 12/12 |
| 16 | 8192 | 33.319862 | 32.880568 | 1.013575x | [1.010733, 1.016255] | 12/12 |
| 32 | 4096 | 16.698993 | 16.628175 | 1.004376x | [1.003474, 1.005291] | 12/12 |
| 32 | 8192 | 66.187344 | 64.988033 | 1.018588x | [1.018025, 1.019151] | 12/12 |

The four-row geometric speedup is `1.015714x`; all 48 timing blocks win and every interval excludes 1. The earlier independent candidate qualification was also positive at `1.014511x` geometric with 48/48 block wins.

- The current D64 leader is direct-output. Its fused symbol uses 13,312-byte LDS and 178 logical/192 allocated VGPRs; its long KV symbol uses 15,616-byte LDS and 169 logical/192 allocated VGPRs. Neither has private memory or spills. P and dS remain in FP32 C fragments and go through the exact FP32-to-FP16 C-to-A handoff.
- The forward Q path already stores E5M2 bytes through a `uint8x16` lane-vector pattern in `featherattn_fwd_kernel.h`. That path uses RNE and log2 prescaling for forward Q, so its accuracy policy cannot be copied blindly to backward P or dS.
- `kernel/hip/hip_kernel.cu` contains the compact `fp8e5m2x4_to_half2x2` decode: one packed `uint32_t` input produces two packed half2 outputs through two `v_perm_b32` operations. This is a useful decode primitive and an instruction-count reference, not proof that a backward hot-loop decode will be profitable.
- The archived seven-GEMM packed screen measured exact `167` logical VGPRs, `168` allocated VGPRs, `20,480` bytes LDS, and 8 permutes for the unquantized control. P-only used `165/168` VGPRs and 19,456 bytes LDS but 40 permutes; dS-only used `169/192` and 44 permutes; P+dS used `164/168`, 18,432 bytes LDS, and 76 permutes. None changed the useful allocation class in the tested schedule.
- The same screen found truncating P or dS numerically admissible under the broad public tolerance but not preferred: P maximum relative L2 was about `0.10045`, dS about `0.10021`, and P+dS about `0.17681`. Exact FP16 had maximum relative L2 about `0.0002318` in that screen. This is a strong reason to test truncation as an optimization candidate, not silently use it.
- Prior fixed-scale, power-of-two-scale, and linear-scale attempts consumed conversion/register margin and were rejected. gfx1151 has no useful native FP8 WMMA or native E5M2 conversion instruction, so the byte savings compete directly with VALU and `v_perm_b32` work.

### Profiling and Compiler-Analysis Limitations

- gfx1151 cannot schedule the complete counter set in one `rocprofv3` pass. Core, traffic, cache, LDS, barrier, and stall groups must be collected in separate legal passes against the same symbol and shape. Wait counters overlap and are not additive. `rocprof-compute` is unavailable in the current environment, so the retained results use serialized `rocprofv3` passes and profiler timestamps.
- The ROCm LLVM package is `23.0.0git` and has no standalone `llc`. Production-correlated MIR must be emitted through clang with `-O3 -fno-gpu-rdc --offload-arch=gfx1151`; unoptimized IR can leave callable `WmmaInPlace` helpers that do not exist in the linked image.
- Older clang pre-greedy MIR is not directly reusable: its `scavengeFI: '%stack.30'` metadata fails parsing despite stack object 30 being present. The optimized phase-isolated `-stop-after=phi-node-elimination` snapshot parses and is the authoritative virtual-register input.
- The post-allocation clang MIR spells two `V_CMP` implicit definitions as `$vcc_lo`, while the same LLVM 23 parser requires `$vcc`. Analysis uses a copy with only those two operands rewritten; the original MIR and linked object remain untouched.
- `LiveIntervals::print` emits only already cached physical register-unit ranges. The analyzer must request every VGPR unit explicitly. Each gfx1151 VGPR has low/high 16-bit units, so unit counts must be grouped by physical VGPR or they double the apparent pressure.
- A main virtual interval can span holes between lane subranges. Summing full register-class weights from only the main interval materially overstates pressure at later WMMA sites; lane/subrange liveness or post-allocation physical unit ranges are required. The final machine snapshot peaks at 163 simultaneously live physical VGPRs while naming `VGPR168`; the six holes below it are fragmented (`133-136`, `138`, and `140`), so the 169-register image is an allocator placement/contiguous-tuple result rather than 169 simultaneous values.

### Reusable Kernel-Inspection Workflow

`rocprofv3` is the authority for executed dispatches, timestamps, and hardware counters. It does not explain the exact linked instruction stream, ABI, spill contract, compiler-stage liveness, allocation fragmentation, or theoretical resource limit. Keep those questions separate and use the following evidence hierarchy for this kernel and future gfx1151 kernels:

| Question | Primary evidence | What it establishes |
| --- | --- | --- |
| Which kernel actually ran? | `rocprofv3` trace/counter CSV | Exact demangled `Kernel_Name`, dispatch ID, grid, workgroup, and phase at the profiled shape |
| What image was shipped? | Linked gfx1151 code-object metadata and symbol table | Mangled identity, kernarg ABI, wave size, workgroup limit, logical VGPR/SGPR counts, LDS/private bytes, spill counts, and dynamic-stack use |
| What did the compiler emit? | Symbol-scoped linked disassembly | Static code size, instruction sequence, memory widths/offsets, WMMAs, waits, barriers, lane operations, branches, calls, and scratch operations |
| Why is allocation high? | Optimized MIR plus LLVM `SlotIndexes`, `LiveIntervals`, and register-pressure tracking | Virtual live ranges and pressure at named blocks, slots, and instructions before allocation |
| How was the physical range used? | Post-allocation MIR correlated against linked ISA | Simultaneously live physical VGPRs, holes, tuple-placement effects, and the instructions that touch the highest allocated registers |
| What should occupancy or traffic be? | Metadata plus an explicit architecture/algorithm model | Allocation-rounded resource limits, theoretical occupancy, compulsory bytes, logical output bytes, and phase FLOPs |

#### 1. Freeze Provenance and Runtime Identity

Record the repository commit and diff state, generated translation-unit hash, complete compiler command, `hipcc`/clang/LLVM versions, target ID, extension hash, and shape. Obtain the command from `ninja -t commands` rather than reconstructing include paths or defines. The production-equivalent HIP compile must retain `-O3 -fno-gpu-rdc --offload-arch=gfx1151`.

Run a minimal trace or one legal counter pass first. Save the full profiler `Kernel_Name`, dispatch ID, grid, and workgroup. Treat namespace, template arguments, and phase arguments as part of the identity. A source filename or template stem is not enough. If the trace and inspected image are from different builds, label the result as correlated rather than exact.

#### 2. Extract the Linked gfx1151 Image

The installed LLVM tools can extract every embedded code object directly from the linked extension. Copy the extension into an isolated artifact directory because `llvm-objdump --offloading` writes extracted bundles next to its input. A reusable command skeleton is:

```bash
LLVM=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin
OUT=/tmp/kernel_inspect
EXTENSION=/path/to/module.so
MANGLED='_Z...exact_kernel_symbol...'

mkdir -p "$OUT"
cp "$EXTENSION" "$OUT/module.so"
"$LLVM/llvm-objdump" --offloading "$OUT/module.so" > "$OUT/offloading.txt"
for hsaco in "$OUT"/*.hipv4-amdgcn-amd-amdhsa--gfx1151; do
  "$LLVM/llvm-objdump" --syms "$hsaco" | rg -F "$MANGLED" && printf '%s\n' "$hsaco"
done
HSACO=/tmp/kernel_inspect/module.so.N.hipv4-amdgcn-amd-amdhsa--gfx1151
"$LLVM/llvm-readelf" --notes --wide "$HSACO" > "$OUT/metadata.txt"
"$LLVM/llvm-objdump" --syms "$HSACO" > "$OUT/symbols.txt"
"$LLVM/llvm-objdump" --disassemble-symbols="$MANGLED" "$HSACO" > "$OUT/kernel.s"
sha256sum "$EXTENSION" "$HSACO" > "$OUT/hashes.txt"
```

Select the one gfx1151 bundle containing the exact mangled symbol and require a single match. `llvm-readelf`, not GNU `readelf`, decodes the AMDGPU MessagePack note into readable metadata in this toolchain. Check `.kernarg_segment_size` and argument offsets/kinds, `.wavefront_size`, `.max_flat_workgroup_size`, `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size`, `.private_segment_fixed_size`, both spill counts, and `.uses_dynamic_stack`. Keep logical metadata counts distinct from campaign-rounded hardware allocation.

Inspect only the selected function body. The retained ISA parser counts total instructions and code bytes plus WMMA, transcendental, global-memory, LDS, barrier, wait, permute/lane, branch, SALU, and VALU categories. Extend the opcode categories for a new kernel, and search explicitly for calls and scratch loads/stores. Map source regions through distinctive operation sequences and constants such as WMMA instructions, C-fragment conversion, and LDS offsets, then confirm the corresponding MIR block; a source line alone is not an ISA identity after inlining and scheduling. Static waits and barriers show placement, not elapsed stall cycles; only dynamic counters can establish their exposed cost. Compare candidate and control at the same symbol boundary instead of comparing whole modules.

#### 3. Recover Optimized MIR and Liveness

Replay the exact production command in an isolated directory with compiler temporaries retained, then convert the saved gfx1151 device bitcode to textual LLVM IR. Use that optimized device IR as the common input for all snapshots. With the current LLVM 23 package and no standalone `llc`, clang emits MIR at a named backend stop point:

```bash
CLANG="$LLVM/clang++"
DEVICE_LL=/tmp/kernel_inspect/device.ll
COMMON=(-target amdgcn-amd-amdhsa -mcpu=gfx1151 -O3 -x ir -S)

"$CLANG" "${COMMON[@]}" "$DEVICE_LL" \
  -mllvm -stop-after=phi-node-elimination -o "$OUT/after-phi.mir"
"$CLANG" "${COMMON[@]}" "$DEVICE_LL" \
  -mllvm -stop-after=amdgpu-pre-ra-long-branch-reg -o "$OUT/pre-ra.mir"
python ~/tmp/feather_attn/extract_mir_function.py \
  "$OUT/after-phi.mir" "$OUT/kernel.after-phi.mir" 'exact-function-substring'
```

Use `-mllvm -debug-pass=Structure` to record the pass sequence before choosing another stop point; pass names and ordering are toolchain-specific. Also emit final assembly from the same IR and compare its target, symbol resources, and instruction sequence with the extracted linked image. A MIR snapshot is production-correlated only after that check. The optimized `phi-node-elimination` snapshot was authoritative for virtual pressure here; the later pre-RA snapshot was a schedule/lifetime cross-check, and post-allocation MIR was a physical-placement diagnostic. Linked metadata and linked ISA remain authoritative for final allocation.

Extract one machine function before analysis to avoid accidentally matching a helper or another phase. Build `analyze_feather_pressure.cpp` against the same package reported by `llvm-config --version`, then run it with the exact function substring:

```bash
"$LLVM/clang++" $("$LLVM/llvm-config" --cxxflags) -std=c++20 \
  ~/tmp/feather_attn/analyze_feather_pressure.cpp \
  -o "$OUT/analyze_feather_pressure" \
  $("$LLVM/llvm-config" --ldflags --system-libs --libs all)
"$OUT/analyze_feather_pressure" \
  "$OUT/kernel.after-phi.mir" 'exact-function-substring' \
  > "$OUT/pressure.after-phi.txt"
```

The analyzer parses MIR through LLVM, creates `SlotIndexes`, `MachineDominatorTree`, `LiveIntervals`, and `RegisterClassInfo`, and uses lane-aware `RegPressureTracker` data. For physical analysis it requests every register unit and groups both gfx1151 16-bit units into one VGPR. Record the machine basic block, slot, instruction, live intervals, pressure set, and local peak at each WMMA/inline-assembly boundary.

Never compare virtual-register numbers across separately generated MIR files. Correlate snapshots through function identity, machine block, slot neighborhood, opcode/operand shape, and the final ISA sequence. Likewise, do not equate the highest named physical register with simultaneous pressure: allocation holes and contiguous tuple requirements can make `VGPR168` live in a region whose measured physical peak is lower.

#### 4. Add Static Models, Then Validate Them

- Resource model: start from symbol-matched metadata, apply gfx1151 allocation granules and architectural limits, and take the minimum bound from VGPRs, SGPRs, LDS, waves/threads, and workgroups. Report logical count, rounded allocation, theoretical occupancy, and profiler-observed occupancy as four distinct quantities.
- LDS model: derive every lane's byte address from the source layout, then verify transaction width and immediate offsets in `ds_*` ISA. Map each vector access onto gfx1151 banks and the 128-byte bank cycle. Use LDS conflict/latency counters and a focused layout fixture to validate the model; neither source padding nor a counter alone identifies the winning access order.
- Work and traffic model: derive algorithmic FLOPs and compulsory/logical bytes from tensor shapes and phase ownership. Keep profiler-measured `FETCH_SIZE` separate from modeled output stores and state whether cache effects are measured or assumed. For this backward path use `F7` for complete timing and the documented `4/7` KV and `3/7` Q split only for phase diagnostics.
- Schedule model: use MIR dependencies and linked instruction order to identify independent work that could cover an LDS, barrier, or WMMA dependency. Confirm any claimed improvement with a removed/reordered linked instruction and lower dynamic exposure; a source reorder with unchanged ISA is no change.
- Timing validation: hash both code objects, alternate baseline and candidate in repeated blocks, time the complete provider path, and retain per-block ratios plus a paired bootstrap interval. Phase timestamps, static instruction reductions, theoretical occupancy, and isolated fixtures are triage evidence, not promotion evidence.

For each future inspection, retain one directory containing the provenance manifest, profiler identity row, extracted code object, metadata, symbol table, exact-symbol disassembly, saved device IR, phase-local MIR, pressure reports, static-model inputs, and paired timing result. That bundle is the minimum evidence needed to reproduce a compiler or ISA claim after the build tree changes.

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
| KV probability recomputation | Rejected: bit-identical outputs, but keeping score/LSE state for a second exponential raises KV-only from 175 to 184 logical VGPRs and SGPRs from 33 to 47; H16/N4096 complete timing regresses from `8.470021` to `8.722818 ms` (`2.98%`) |
| KV full K-row reload | Rejected: bit-identical and reduces KV-only to 153 logical/168 allocated VGPRs, 34 SGPRs, 13,312 B LDS, zero private memory/spills, but H16/N4096 regresses from `8.495566` to `8.779339 ms` (`3.34%`) because every Q tile reloads all four K fragments |
| KV partial K-row cache | Rejected: cache-two is bit-identical at 162 logical/168 allocated VGPRs and halves repeated K loads, but loses every block in the H16/H32 x N4096/N8192 paired matrix; row speedups are `0.9907x`, `0.9699x`, `0.9805x`, and `0.9654x` (geometric `0.9766x`, all bootstrap intervals below 1) |
| Dedicated KV kernel extraction | Rejected as redundant: a standalone KV source generates the same 175-VGPR, 33-SGPR, 13,312-byte image as compile-time `kPhase == 1`; phase specialization already removes dead Q state |
| Native FP16-output WMMA accumulation | Rejected as a register lever: gfx1151 FP16 and FP32 WMMA output fragments both occupy eight VGPRs, so changing the output type cannot halve a dK or dV accumulator bank |
| KV Q-row staging | Rejected: bit-identical and `1.0082x` in one focused H16/N4096 run, but raises the KV specialization to 192 logical VGPRs and 15,360 B LDS; it is a materially worse resource image than dO-only staging |
| Linear 64-half dO row staging | Rejected: bit-identical but severe LDS bank conflicts regress H16/N4096 from `8.324604` to `11.090252 ms` |
| Padded 72-half dO row staging | Accepted and integrated only for long-sequence KV-only: bit-identical, 169 logical/192 allocated VGPRs, 33 SGPRs, 15,616 B LDS, zero private memory/spills, and `1.015714x` linked-image paired geometric speedup with 48/48 block wins |
| KV-only interleaved dV/dP WMMA schedule | Rejected: bit-identical, but logical KV VGPRs rise from 169 to 177 in the same 192-VGPR class, static code grows by 28 bytes, and direct alternating timing gives `0.984856x` six-row geometric speedup with `0/36` winning blocks and every row's bootstrap interval below 1 |
| Aligned KV-only dO row-stride sweep | Rejected beyond stride 72: strides 80/88/96/104/112/120 keep 169 logical VGPRs and 33 SGPRs but increase LDS to 15,872-17,152 B; their H16/N4096 paired speedups are `0.990375x`, `0.998818x`, `0.912008x`, `0.993403x`, `0.980309x`, and `0.991942x`. The closest stride 88 confirms at only `0.999945x` geometric across H16/H32 x N4096/N8192 with `22/48` winning blocks and all row intervals crossing 1 |
| KV-only trailing wave-barrier removal | Rejected: bit-identical with unchanged resources, but direct alternating timing gives `0.999392x` six-row geometric speedup and `17/36` winning blocks; per-row bootstrap intervals cross 1, so a redundant-barrier hypothesis does not justify the larger schedule change by itself |
| KV-only ping-pong Q/dO staging | Rejected: corrected pipeline is bit-identical but reaches only `0.958193x` across six HND rows with `0/36` winning blocks; linked LDS rises from 15,616 to 23,040 B, SGPRs from 33 to 39, and logical VGPRs from 169 to 170 |
| KV phase precomputed tile bases | Rejected: bit-identical and removes 76 static bytes in the linked KV symbol, but remains at 169 logical/192 allocated VGPRs, 33 SGPRs, and 15,616 B LDS; six-row paired timing is `0.999524x` with `13/36` winning blocks and no repeatable complete-path gain |
| NHD physical D64 backward layout | Accepted: contiguous `[B,N,H,D]` validation, layout-aware Delta and seven-GEMM images, direct strided D-vector addressing, `22/22` dual-layout saved-state cases passed, and linked HND/NHD images remain within 192 campaign-rounded VGPRs, 32 KiB LDS, zero private memory, and zero spills |
| NHD head-order policy | Accepted for the current topology: NHD uses the exact quotient/remainder LLC group permutation when the forward-derived count is greater than one and `H % 16 == 0`, head-fast ownership for the remaining periodic cases, and tile-fast ownership for other head strides; the policy is compile-time image-preserving and keeps arbitrary head counts supported |
| NHD long phase fusion | Rejected: combining KV and Q ownership for NHD improved some short rows but the focused nine-row NHD screen reached only `0.956310x` geometric Feather/AITER, with H16/N16384 at `0.943x`, H32/N8192 at `0.738x`, and H32/N16384 at `0.760x`; retain long phase fission |
| NHD all-head LLC grouping | Rejected as a blanket policy: applying the forward-style grouped head permutation to non-periodic H56 strides regressed the focused N8192 and N16384 rows to `0.941x` and `1.045x`; grouping is now admitted only when `H % 16 == 0` |
| NHD H32/N16384 group-count sweep | Rejected beyond the forward-derived count 4: corrected uneven permutations with counts 5 and 3 reached only `0.904842x` and `0.911481x` Feather/AITER; count 4 remains selected |
| Initial equal-size NHD head-group permutation | Rejected and corrected before timing: the H32/group-count-5 trial generated an illegal address because its group count did not divide `H`; no result was admitted from that image |
| Uneven NHD head-group permutation | Accepted correctness fix: one-launch grouped ordering now uses quotient/remainder group sizes within each batch, preventing out-of-range physical heads when the group count does not divide `H`; the new H32/NKV16385 asymmetric case passes in both layouts |
| Bounded NHD-to-HND backward dispatch | Accepted for valid contiguous B1 self-attention with H `{32,56}` and N `[8192,16384]`: direct comparison is bit-exact for all four outputs; production-path timing includes five input copies, three output copies, and temporary allocation, yet the primary matrix improves from `16/18` and `1.147379x` to `18/18` and `1.182570x`. NHD H32/N8192 and H32/N16384 rise from `0.981x` and `0.920x` to `1.044x` and `1.206x`; all other shapes, plus optimized shapes without sufficient temporary memory, retain direct NHD execution |
| Combined Q-row and dO-row staging | Rejected: bit-identical at 173 logical VGPRs and 17,408 B LDS, but focused timing is only `1.0014x` versus fission and does not beat dO-only at `1.0205x` |
| Nine-wave launch bound | Rejected as an allocator diagnostic: LLVM's final occupancy is eight waves and cannot satisfy `__launch_bounds__(128, 9)` |
| Forced 168-VGPR attribute | Rejected: deprecated `amdgpu_num_vgpr` changes every specialization and lowers logical counts to 161 for dO-stage, 167 for KV baseline, and 168 for fused without changing the 192-VGPR campaign class. Four-way H16/N4096 timing is uncapped fission `8.471286 ms`, uncapped dO-stage `8.264328 ms`, capped fission `8.326294 ms`, and capped dO-stage `8.284446 ms`; the naturally allocated dO image is faster and has no spills |
| Old atomic FP32 D64 kernel | Removed; no source, selector, workspace, or linked image remains |
| `owned`, `paired`, automatic selection, and duplicate D64 reference paths | Removed from production |
| FP8 P/dS truncation | Rejected for the current direct-fragment topology; revisit only if a future schedule creates persistent P/dS storage whose byte reduction can repay software conversion |
| Six-GEMM N-squared workspace design | Deferred; it requires approximately 256 MiB intermediate storage and 512 MiB extra traffic at the reference shape |
| Stock AITER CK backward | Blocked on gfx1151 receipt generation; retained only as external evidence |

## Completed Bottleneck Diagnosis

The serialized phase-local profiling step is complete. Phase FLOPs use `4/7 * F7` for KV and `3/7 * F7` for Q; the times below are profiler dispatch timestamps and are diagnostic rather than promotion timings.

| Capture | Phase TFLOPS | Occupancy | SIMD utilization | Wait-any / wave cycles | FLOP / normalized byte | Read + logical output GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/N4096 KV, production dO stage | 28.040 | 47.46% | 99.25% | 23.58% | 2,520 | 11.13 |
| H16/N4096 Q, pre-stage control | 33.200 | 52.14% | 97.00% | 16.56% | 2,040 | 16.28 |
| H16/N4096 Q, production companion | 32.809 | 52.28% | 97.20% | 16.57% | 2,038 | 16.10 |
| H32/N8192 KV, production dO stage | 27.648 | 49.14% | 99.24% | 22.69% | 4,372 | 6.32 |
| H32/N8192 Q, production companion | 32.834 | 55.63% | 99.68% | 16.78% | 2,752 | 11.93 |

The byte normalization adds profiler-measured `FETCH_SIZE * 1024` to the minimum logical FP16 output stores for that phase. `FETCH_SIZE` already includes extra video-memory fetches and cache/memory effects; it is not a write counter. The output term is modeled, not measured `WRITE_SIZE`, so the final column must not be presented as a measured bidirectional DRAM rate.

Arithmetic intensity is approximately `2,038-4,372 FLOP/byte`, far above the approximately `232 FLOP/byte` nominal ridge point. Production read-plus-modeled-output bandwidth is only `6.3-16.1 GB/s`. Phase throughput is `27.65-33.20 TFLOPS`, about `46.5-55.3%` of nominal peak, while SIMD utilization is near full, occupancy is roughly half maximum, and wait-any consumes `16.6-23.6%` of wave cycles. The H16 captures also place barrier wait near `7.4-7.5%` of wave cycles. Wait counters overlap and are not additive. These results reject nominal DRAM bandwidth as the dominant explanation and identify instruction issue, LDS dependency, and barrier serialization as the active limit. KV remains first because it is slower per GEMM-equivalent and carries the larger wait-any fraction.

The optimized, phase-isolated MIR aligns the source-level `CFragmentToA`/WMMA handoff with a point at `pressure_total=199`, `SReg_32=29`, and `VGPR_32=170`. Generated tuple-construction regions elsewhere report virtual values as high as `pressure_total=283` and `VGPR_32=263`; those are optimized virtual-pressure observations, not linked physical allocations. The post-allocation snapshot peaks at 163 simultaneously live physical VGPRs while naming `VGPR168` because placement has holes. A useful lifetime change must therefore cause a natural linked allocation or scheduling improvement; a lower source-level count or forced cap is not sufficient.

The remaining hypotheses have the following disposition:

| Hypothesis | Measured disposition |
| --- | --- |
| Generic DRAM/cache optimization | No-go: intensity is far above the ridge and observed bandwidth is low; traffic work must also remove instruction, LDS, or barrier cost |
| K reload or partial cache | Closed for the current topology: both reached a natural 168-VGPR allocation class but regressed complete timing |
| dO LDS layout | Narrowly open: linear stride 64 failed badly and padded stride 72 won; any follow-up must preserve aligned vector loads and improve the remaining LDS wait/conflict behavior |
| Lifetime or occupancy only | No-go by itself: forced VGPR limits lost, and physical pressure shows allocator fragmentation rather than 169 simultaneously live values |
| Scheduling and barrier overlap | Open and highest priority because it directly targets the measured wait fractions without relying on DRAM savings |
| Address calculation | Open only as an ISA-qualified subexperiment; stop when source changes do not remove linked hot-loop instructions or when hoisting lengthens live ranges |
| Persistent K/V cache compression | Deferred: it must qualify numerically, cause a natural allocation or occupancy transition, and improve complete timing; direct P/dS compression remains excluded |

## Ranked Stop/Go Plan

The ranked D64 campaign is complete. The interleave, aligned-stride, barrier, ping-pong, and address-generation ranks failed their complete-path gates; the bounded NHD-to-HND dispatch was accepted after exact-output and full-matrix qualification. No pending kernel-schedule candidate satisfies all promotion gates.

Persistent K/V compression remains deferred because the rejected schedule work did not create a natural resource boundary. Reopen it only after a future topology independently creates an allocation or occupancy transition; direct P/dS compression, software conversion without that transition, and traffic-only improvements remain excluded.

### Admission Gates

- Keep each experiment private and compile-time in the KV-only specialization. Do not add a public selector or alter the HND/NHD layout contract, FP16 inputs/outputs, natural-log FP32 LSE, device validation, stream semantics, or the raw ABI.
- Compile with the production-equivalent `clang -O3 -fno-gpu-rdc --offload-arch=gfx1151` path and inspect the exact linked symbol. Require at most 192 campaign-rounded VGPRs, at most 32,768 bytes of LDS, zero private memory, and zero spills.
- Compare MIR only from matched-generation snapshots using block, slot, interval, and physical-ISA alignment. Virtual register numbers from separate snapshots are not identities.
- Run the six research correctness cases and the public D64 contract. Directional candidates require cosine at least `0.997`; Rel-L2 at most `0.10` is preferred and below `0.20` is exploratory. Exact scheduling/layout changes should retain the current approximately `0.0003` Rel-L2 behavior.
- Use phase-local H16/N4096 and H32/N8192 profiling only for triage. Promotion requires complete provider timing, including Delta, every main/helper launch, synchronization, conversions, and provider work, normalized with `F7`.
- Run repeated alternating paired measurements on all H `{16,32,56}` x N `{4096,8192,16384}` rows. Require credible positive complete-path evidence with no material row regression; a phase-only win, static resource reduction, or isolated fast sample is not sufficient.
- Re-run the `188/188` forward contract, the D64/D128 backward public contract, and exact NHD-to-HND dispatch equivalence before integration.

### Stop Conditions

Stop each candidate at its first failed admission gate and preserve its evidence outside production. If the exact scheduling, padded-layout, barrier, and ISA-qualified address experiments fail to improve complete timing, close the D64 campaign with the current seven-GEMM kernel. Do not continue with generic bandwidth work, forced allocation, direct P/dS compression, or previously rejected K-cache/reload and Q-row-stage variants unless a future topology materially changes their traffic and lifetime assumptions.

D128 remains a separate seven-GEMM design from D64 and is now accepted in production. Preserve its separate Delta/KV/Q topology and do not restore the deleted scalar reference kernel as part of future D64 work.

## Artifacts

- Forward matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Final D128 production matrix: `/tmp/feather_bwd_d128_final_matrix_production.json`.
- Post-D128 D64 regression matrix: `/tmp/feather_bwd_d64_regression_matrix_after_d128.json`.
- Final D128 linked-image inspection: `/tmp/feather_bwd_d128_final_inspect/`.
- Current dual-layout D64 timing run: `/tmp/feather_bwd_primary_matrix_hnd_nhd_transpose_final.json`.
- Previous all-direct NHD timing run: `/tmp/feather_bwd_primary_matrix_hnd_nhd_final.json`.
- Current linked backward resource inspection: `/tmp/feather_bwd_nhd_final_inspect_20260818/`.
- Rejected KV dV/dP interleave campaign: `/tmp/feather_bwd_kv_interleave_campaign/`.
- Rejected aligned KV dO row-stride sweep: `/tmp/feather_bwd_kv_stride_sweep_campaign/`.
- Rejected KV trailing-wave-barrier campaign: `/tmp/feather_bwd_barrier_simple_campaign/`.
- Rejected KV ping-pong staging campaign: `/tmp/feather_bwd_barrier_campaign/`.
- Rejected KV address-base campaign: `/tmp/feather_bwd_address_campaign/`.
- Private D64 leader timing: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/timing_phases_clean.json`.
- Private D64 leader correctness: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/correctness.json`.
- Private D64 metadata/disassembly: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/build/attn_stride20_all_lds13312_hip_ext/`.
- Historical packed FP8 screen: `~/tmp/feather_attn/candidates/seven_gemm_packed/qualification_summary.json`.
- FP16/FP8 conversion reference: `kernel/hip/hip_kernel.cu`.
- Completed FP8 campaign: `~/tmp/feather_attn/e5m2_truncation_campaign/`.
- Phase-fission fixture and paired evidence: `~/tmp/feather_attn/phase_fission_campaign/`.
- KV recomputation fixture: `~/tmp/feather_attn/kv_recompute_campaign/`.
- Full and partial K-cache fixtures: `~/tmp/feather_attn/kv_reload_campaign/` and `~/tmp/feather_attn/kv_partial_cache_campaign/`.
- Dedicated-KV fixture: `~/tmp/feather_attn/kv_dedicated_campaign/`.
- Q/dO row-stage controls: `~/tmp/feather_attn/kv_row_stage_campaign/`.
- Linear dO stage rejection: `~/tmp/feather_attn/kv_dorow_linear_campaign/`.
- Padded dO stage, linked metadata, and paired production evidence: `~/tmp/feather_attn/kv_dorow_stride72_campaign/`.
- Launch-bound and forced-VGPR diagnostics: `~/tmp/feather_attn/kv_dorow_stride72_lb9_campaign/` and `~/tmp/feather_attn/kv_dorow_stride72_vgpr168_campaign/`.
- Phase-local roofline normalization: `~/tmp/feather_attn/current_roofline.json`.
- Serialized KV pre-stage/dO-stage counter summary: `~/tmp/feather_attn/d64_kv_profiles/summary.pretty.json`.
- Optimized virtual-pressure reports: `~/tmp/feather_attn/compiler_inspect/regalloc/pressure_O3_phi_phase1_v4.txt` and `~/tmp/feather_attn/compiler_inspect/regalloc/pressure_O3_prera_phase1_v4.txt`.
- Post-allocation physical-pressure report: `~/tmp/feather_attn/compiler_inspect/regalloc/pressure_O3_regallocfast_phase1_grouped.txt`.
- Saved production-equivalent device compiler outputs: `~/tmp/feather_attn/compiler_inspect/current/` and `~/tmp/feather_attn/compiler_inspect/current_device.ll`.
- MIR function extractor and LLVM pressure analyzer source: `~/tmp/feather_attn/extract_mir_function.py` and `~/tmp/feather_attn/analyze_feather_pressure.cpp`.
- Static linked-ISA category counter used as the parser template: `~/tmp/feather_attn/analyze_final_isa.py`.
