# FeatherAttn gfx1151 Backward Kernel

## Status

The public backward surface is intentionally small:

| Head dimension | Explicit implementation | Role |
| ---: | --- | --- |
| 64 | `implementation="fused"` | Retained custom D64 kernel with caller-owned FP32 workspaces |
| 128 | `implementation="reference"` | Correctness kernel with no full-gradient workspace |

There is no default, `auto`, D64 reference, `owned`, or `paired` implementation in the linked production extension. Low-level IDs are `0=reference` and `1=fused`; the Python argument is required and keyword-only. The saved-state API remains contiguous HND `[B,H,N,D]`, FP16 `(Q,K,V,O,dO)`, FP32 natural-log LSE, FP32 Delta, current-device validation, and current-stream execution.

The qualified performance provider is the FlashAttention AITER Triton backward kernel. The retained Feather kernels are correctness and development surfaces. The main private result is an exact D64 seven-GEMM schedule that is close to the target but is not production-promoted.

## Public Contract and Source Layout

The backward wrapper accepts:

```text
feather_attn_backward(Q, K, V, O, LSE, dO, *, implementation, sm_scale=None, ...)
```

Constraints:
- FP16 Q/K/V/O/dO and FP32 natural-log LSE and Delta.
- Dense, non-causal attention with equal query/KV heads.
- Contiguous HND tensors only.
- D64 selects `fused`; D128 selects `reference`.
- D64 fused requires `dq_acc`, `dk_acc`, and `dv_acc` FP32 workspaces. The wrapper allocates them when omitted, or accepts caller-owned tensors.
- D128 reference does not require full FP32 gradient workspaces.
- All tensors must be on the current device and meet the checked signed-int32 address and dimension limits.

| File | Role |
| --- | --- |
| `kernel_attn/hip/hip_kernel.py` | Explicit Python selector, output/workspace setup, extension source list |
| `kernel_attn/hip/hip_kernel.cpp` | Shared validation, implementation dispatch, current-stream lookup, Torch registration |
| `kernel_attn/hip/hip_kernel.h` | Raw forward/backward launch ABI |
| `kernel_attn/hip/featherattn_bwd_kernel.h` | Reference and fused device templates |
| `kernel_attn/hip/featherattn_bwd_reference_d128.cu` | D128 reference instantiation |
| `kernel_attn/hip/featherattn_bwd_fused_d64.cu` | D64 fused instantiation |

The two backward images compile independently. The linked extension exposes exactly `feather_attn_bwd_d128_reference` and `feather_attn_bwd_d64_fused`.

## Current Production Design

### D64 Fused Kernel

The retained D64 kernel uses a 64-thread workgroup, D64 query tiles of 16 rows, and KV tiles of 32 rows. Its main kernel stages Q/K/V and P/dS state in 17,152 bytes of LDS, computes FP32 gradients, and accumulates dQ, dK, and dV into caller-owned FP32 workspaces. The complete launch sequence is:
- Delta and dQ workspace clear.
- Main fused backward kernel.
- FP32 dQ conversion to FP16.
- FP32 dK/dV conversion to FP16.

The loaded production image uses 174 logical VGPRs, 192 allocated VGPRs, 54 SGPRs, 17,152 bytes LDS, zero private memory, zero spills, and 16 compiler-generated FP32 atomic CAS sites for the workspace accumulation. This implementation is numerically qualified but much slower than AITER.

### D128 Reference Kernel

The D128 reference path launches separate Delta, dQ, dK, and dV kernels. Each gradient kernel reconstructs the saved-state probability path in FP32 and stores the final FP16 gradient. It is deliberately scalar and independent from the D64 fused schedule so it remains a correctness oracle. It is not a throughput candidate and has no current performance/profile qualification.

### Numerical Dataflow

All retained production paths consume the caller's natural-log LSE rather than recomputing it. The reference reconstruction is:

```text
P  = exp(Q @ K^T * sm_scale - LSE)
Delta = rowsum(O * dO)
dP = dO @ V^T
dS = P * (dP - Delta)
dQ = dS @ K * sm_scale
dK = dS^T @ Q * sm_scale
dV = P^T @ dO
```

Intermediate scores, probabilities, Delta, dP, dS, and accumulation are FP32. Caller-visible dQ/dK/dV are FP16.

## Private D64 Seven-GEMM Design

The active research control is separate from the linked production extension. It uses one 128-thread workgroup per head/tile and unique output ownership with no gradient atomics or partition-reduction workspace.

The seven attention-sized GEMM equivalents are:

| Owner | GEMM equivalents | Work |
| --- | ---: | --- |
| KV-owned portion | 4 | QK score, dO/V dP, dS/Q dK, P/dO dV |
| Q-owned portion | 3 | QK score, dO/V dP, dS/K dQ |
| Total | 7 | Exact FP16 arithmetic and FP32 accumulation |

The KV portion uses KV64 ownership. The Q portion uses KV32 grouping. The implementation stages operands cooperatively, converts WMMA C fragments directly to WMMA A fragments for P and dS, and uses a stride-20 K/Q/dO transpose layout. Lifetime-local LDS reservations are aliased so the current leader uses 13,312 bytes rather than the earlier 24,576-byte image. Persistent K, Q, and dO caches remove the scalar transposed global-load network that dominated the initial seven-GEMM source.

The private leader is not part of the public API, is not linked into the production extension, and is not allowed to change the saved-state ABI without a separate promotion decision.

## Backward Throughput Convention

All backward throughput values in this document use the same normalized seven-GEMM work estimate:

```text
F7 = 7 * 2 * B * H * NQ * NKV * D FLOPs
TFLOPS = F7 / elapsed_seconds / 1e12
```

For equal sequence lengths this is `14 * B * H * N^2 * D` FLOPs. The same denominator is applied to AITER and Feather so the ratio is a timing ratio expressed as throughput:

```text
Feather / AITER = AITER elapsed time / Feather elapsed time
```

This convention matches the private D64 schedule. It is a normalized comparison for the public fused path, whose ownership and workspace topology differs. No throughput claim is made for the D128 scalar reference because it has not been performance-qualified.

## Current Benchmarks

### Public D64 Fused Surface

The last qualified six-row saved-state matrix uses identical HND inputs at B1, H `{16,32}`, N `{2048,4096,8192}`, D64, preallocated outputs/workspaces, and complete backward timing. The table below recalculates TFLOPS with the seven-GEMM convention rather than using the older five-GEMM field in the raw JSON.

| Provider | Geomean TFLOPS, seven GEMMs | Feather / AITER | Wins |
| --- | ---: | ---: | ---: |
| Feather D64 fused | 5.436 | 0.204x | 0/6 |

Representative rows:

| H | N | AITER TFLOPS | Feather fused TFLOPS | Feather / AITER |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2048 | 27.082 | 5.328 | 0.197x |
| 16 | 4096 | 27.289 | 5.424 | 0.199x |
| 16 | 8192 | 27.107 | 5.532 | 0.204x |
| 32 | 4096 | 26.063 | 5.450 | 0.209x |
| 32 | 8192 | 25.984 | 5.552 | 0.214x |

Raw matrix: `~/tmp/feather_attn/bwd_d64_final_four_provider_matrix.json`. The old owned and paired columns remain useful as campaign evidence but are not current production results.

### Private Seven-GEMM Leader

The current leader was timed at B1/H16/N4096/D64 with eight warmups and 50 interleaved samples. The AITER row is from the same timing run.

At this shape:

```text
F7 = 240.518 GFLOPs
```

| Candidate | Median ms | Seven-GEMM TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: |
| Stride-20 full image | 9.526261 | 25.248 | 0.962x |
| 21,504 B LDS plus KV prefetch | 8.852319 | 27.170 | 1.035x |
| 13,312 B current leader | 8.495935 | 28.310 | 1.078x |
| Promotion gate | 8.417913 | 28.572 | 1.088x |

The leader's repeat result is `0.927%` slower than the promotion gate. One separate interleaved observation reached `8.385254 ms`, or `28.683 TFLOPS`, but the crossing is too small and not repeatable enough for promotion. The private leader has not been run across the frozen six-row matrix.

### Work Distribution

The 13,312-byte leader's component timing was measured with separate empty, KV, Q, and full launches. Empty-launch overhead is `0.108981 ms`; subtracting it gives the net workload below.

| Portion | GEMM equivalents | Net ms | Normalized TFLOPS |
| --- | ---: | ---: | ---: |
| KV-owned | 4 | 4.980566 | 27.595 |
| Q-owned | 3 | 3.399451 | 30.322 |
| Full leader | 7 | 8.495935 | 28.310 |

The KV portion carries 57.1% of the nominal GEMM work but about 58.6% of net kernel time. Its per-GEMM throughput is approximately 9.0% below the Q portion, so the next useful experiment must change the KV/Q resource balance rather than only reorder another local instruction.

## Current Profile Results

### Private Leader Resources

The symbol-matched gfx1151 metadata for the private leader reports:

| Workgroup | Logical VGPRs | Allocated VGPRs | SGPRs | LDS | Private/scratch | Spills |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 threads, seven-GEMM main | 178 | 192 | 46 | 13,312 B | 0 / 0 | 0 / 0 |
| Delta helper | 66 | 72 | 10 | 0 B | 0 / 0 | 0 / 0 |

The main image is wave32, has no atomics, and remains within the campaign gates. Its correctness screen covers six exact cases, including odd heads, query/KV tails, cancellation, and batch two. Maximum relative L2 is `0.00030204`; minimum global cosine is `0.99999988`; maximum absolute error is `0.00121403`.

### Full-Leader Counters

The following are serialized legal counter results for the current 13,312-byte full image. They are attribution data, not replacements for complete timing.

| Counter | Full leader |
| --- | ---: |
| Occupancy | 46.62% |
| `SQ_INSTS_VALU_sum` | 426.889 M |
| `SQ_INSTS_LDS_sum` | 77.611 M |
| `SQ_WAVE_CYCLES_sum` | 12.457 B |
| `SQ_WAIT_CNT_ANY` | 2.440 B |
| `SQ_WAIT_INST_LDS_sum` | 453.345 M |
| `SQ_WAIT_BARRIER` | 1.076 B |
| LDS latency | 106.73 cycles |
| LDS conflict metric | 24.32 |
| ALU stalled by LDS | 2.10% |

The leader has low global traffic and lower LDS instruction count than the AITER control, but it still exposes long LDS dependency and barrier wait chains. Higher residency is real, but it does not by itself produce a promotion because complete event timing includes every launch, synchronization, conversion, and workspace cost.

## Accepted and Retained Work

| Work | Decision and current role |
| --- | --- |
| Saved-state HND API with natural-log LSE | Accepted public contract |
| Explicit implementation selection | Accepted; no automatic or dimension-inferred selector |
| D128 scalar reference | Accepted correctness oracle |
| D64 shared-tile fused kernel | Retained public diagnostic kernel; not performance-promoted |
| AITER Triton backward | Accepted performance provider |
| Seven-GEMM unique-ownership topology | Accepted private research control; no atomics or N-squared workspace |
| Cooperative K/V and Q staging | Accepted private mechanism; removed the original global-load bottleneck |
| WMMA C-to-A handoff | Accepted private exact transformation |
| Stride-20 transpose layout | Accepted private exact layout |
| Persistent K/Q/dO caching | Accepted private resource-valid improvements |
| 13,312-byte lifetime-aliased LDS | Accepted private leader mechanism |
| Source split into host, template, and two instantiation units | Accepted production organization |

## Rejected Work

| Candidate or idea | Evidence | Decision |
| --- | --- | --- |
| Public D64 fused promotion | `0.204x` AITER geomean and 0/6 wins | Retain for correctness/diagnostics only |
| Direct dK/dV `owned` | `0.546x` AITER geomean and 0/6 wins | Retired from production source |
| Paired KV64 reduction | Only `1.41%` faster than owned with material LDS regressions | Retired |
| Site-local native FP32 atomics | `12.0452 ms` focused latency; still lost every frozen Triton row and changes accumulation semantics | Not integrated |
| Five-GEMM ownership inversion | Lower nominal work but worse complete timing and substantial synchronization/reduction cost | Rejected |
| Q-owned partition reduction | Complete path reached `92.6505 ms` at H16/N4096/D64 | Rejected |
| Four-wave KV-owned M16 exact FP16 | `14.019381 ms` versus AITER `9.192410 ms`; `1.525x` slower | Rejected |
| M16 unscaled E5M2 | `13.591690 ms`; `1.479x` slower with no lower VGPR allocation class | Rejected |
| Fixed, power-of-two, or linear E5M2 scaling | Conversion and scaling cost consumed the register/LDS margin | Rejected before full integration |
| Initial exact seven-GEMM schedule | `20.858525 ms`, `11.531 TFLOPS`, `0.430x` AITER | Rejected as a schedule, retained as control |
| Transient truncating E5M2 | No lower allocation class; dS-only reached 192 VGPRs and failed numerical preference | Rejected |
| First cooperative staged winner | `10.999544 ms`, `1.204x` AITER latency, above the continuation gate | Retained only as profiling evidence |
| Padded LDS rows | `21.0009 ms` and approximately 542-cycle LDS latency | Rejected |
| Q32 synchronization grouping | `10.193416 ms`, `21.29%` slower than the 13,312-byte Q16 control | Rejected |
| Forced 64-bit LDS reads | `+17.11%` latency and higher conflicts/waits | Rejected |
| DPP/register wave transpose | Exhaustively exact, but optimistic complete bound only `4.35%`, below the required `5%` | Closed before integration |
| Six-GEMM N-squared workspace | Requires approximately 256 MiB intermediate storage and 512 MiB extra traffic at the reference shape | Deferred pending explicit workspace approval |
| Stock AITER CK backward | gfx1151 receipt generation omits `--targets gfx1151` | External CK module retained only as evidence |

## Remaining Work

The next private experiment splits the seven-GEMM main image into compile-time KV and Q kernels launched sequentially on the same stream. The extra launch is included in complete timing. Same-layout fission alone is only a control; the candidate must pair fission with a compact Q-stage layout:
- one K16 row-major tile: 2,048 B;
- one V16 row-major tile: 2,048 B;
- one stride-20 K-transpose tile: 2,560 B;
- practical Q-stage LDS target: exactly 6,656 B.

The Q-only image must allocate at most 144 VGPRs so the 6,656-byte image can cross to a useful residency class. Both images must remain at or below 192 allocated VGPRs, 32,768 B LDS, zero private/scratch memory, and zero spills.

Qualification gates:
- Inspect loaded metadata and symbol-scoped ISA before timing.
- Pass the six-case exact correctness screen.
- Improve split-Q32 timing by at least `12.5%` for the compact Q phase.
- Improve complete timing by at least `5%` against the unchanged 13,312-byte leader after charging the extra launch.
- Repeat below `8.417913 ms` with useful margin before running the frozen six-row matrix.

If the Q symbol remains above 144 allocated VGPRs, the compact LDS layout does not change residency and the experiment stops before timing. If the KV transpose handoff or the complete timing gate fails, close the experiment rather than expanding the source surface with more caches, approximations, ownership changes, or a production ABI change.

### Other Deferred Work

- A custom D128 performance schedule, only after it is spill-free and below the current 191/192 VGPR boundary.
- Any FP8 storage follow-up, unless a materially different exact topology first reaches an occupancy or spill cliff that compression can remove.
- The six-GEMM workspace design, pending a separate memory-budget decision.
- An upstream AITER CK gfx1151 recipe fix.
- The broken optional AITER `fused_atomic` wrapper, which currently raises missing-argument `TypeError` and is not a baseline.

There is no active production D64 schedule search outside the compile-time fission experiment. A candidate that fails its gates is closed and the current public source remains unchanged.

## Verification and Artifacts

Repository verification:
- `test_attn_hip_backward.py`: `4/4` explicit D64/D128 saved-state cases.
- `test_attn_hip.py`: `168/168` forward cases after the shared ABI cleanup.
- Clean extension build with separate backward translation units and exactly two linked backward symbols.
- Python bytecode checks, `git diff --check`, and configured Python hooks pass.

Authoritative artifacts:
- Public D64 matrix: `~/tmp/feather_attn/bwd_d64_final_four_provider_matrix.json`.
- Private leader timing and component timing: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/timing_clean.json` and `timing_phases_clean.json`.
- Private leader correctness: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/correctness.json`.
- Private leader metadata and disassembly: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/build/attn_stride20_all_lds13312_hip_ext/`.
- Private leader counters: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/opt_stride20_all_lds13312/profile/h16_n4096/summary.json`.
- Private optimization ladder: `~/tmp/feather_attn/candidates/seven_gemm_packed/profile/final_best/step11_qualification_summary.json`.
- Historical accepted/rejected experiments: `~/tmp/feather_attn/candidates/seven_gemm_packed/` and the neighboring campaign directories.

The historical source and timing logs remain available for decision provenance. These two documents describe the current design, current measurements, accepted/rejected decisions, and the remaining work rather than reproducing the old phase-by-phase campaign.
