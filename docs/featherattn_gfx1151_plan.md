# FeatherAttn gfx1151 Q8-LDS Attention Plan

## Status

This is the active implementation plan for the gfx1151 FP16 attention path in `kernel_attn/hip/`. The accepted implementation substrate is a custom CK Tile kernel body behind the existing raw HIP/PyTorch extension boundary. The stock CK Tile FMHA pipeline is a reference, not the implementation.

`docs/hip_attention_optimization_plan.md` is retained only as an experiment history. Its proposed architecture and next steps are outdated and must not be used as the implementation roadmap for this refactor.

The accepted implementation has one block shape and two head-dimension specializations:

| Parameter | Value |
| --- | ---: |
| Query rows (`Br`, `BLOCK_M`) | 128 |
| Key/value columns (`Bc`, `BLOCK_N`) | 64 |
| Head dimensions | 64, 128 |
| Supported head counts | Any positive count within the checked address/grid contract |
| Workgroup | 256 threads, 8 wave32 waves |
| External Q/K/V type | FP16 |
| External output type | FP16 |
| Q storage inside the kernel | FP8 E5M2 |
| K, P, and V arithmetic/storage | FP16 |
| QK and persistent output accumulators | FP32 |

Do not add another block size, an autotune table, or a generalized FeatherAttn policy matrix without a separate measured justification. Tail support and the D=64/D=128 specializations share this block shape.

The correctness, resource, and two-layout performance qualification through Phase 11 is complete. The cumulative production baseline is commit `01454e3`: bounded D=128 NHD LLC grouping, expanded D=64 HND decoded-Q caching, and partition-aware D=64 NHD strided grouping passed every gate. Linear D=64 online-softmax state, progressive LDS scheduling, D=64 V-load scheduling, alternate arithmetic, larger block shapes, DPP alpha fan-out, and mixed-PV accumulation were measured independently and rejected by their applicable static, resource, or timing gates.

Phase 11 was a separately justified follow-up campaign derived from a findings-only review of the qualified production images, profiler artifacts, AITER, CK Tile, FlashAttention-CK, SageAttention, and the gfx1151 ISA. It began from a byte-for-byte frozen Phase 10C baseline and accepted only Optimization 11B's additional D=64 HND decoded-Q caching and Optimization 11C's guarded partition-aware D=64 NHD reuse schedule. Pure FP16 WMMA score accumulation and persistent FP16 output accumulation remain excluded. QK scores and the cross-key-tile output state remain FP32; the bounded 11H tile-local FP16 PV probe passed correctness but failed timing and was removed.

## Final Benchmarks

This is the sole authoritative full benchmark matrix in this document. It measures commit `01454e3` on gfx1151 with `benchmark_attn_hip.py`, batch one, FP16 inputs, heads `{16,32,56}`, sequence lengths `{4096,8192,16384}`, head dimensions `{64,128}`, and physically contiguous tensors in the selected layout. Each provider uses `triton.testing.do_bench` with `warmup=25` and `rep=100`. Throughput is `4 * B * H * N^2 * D / time`; `Feather / AITER` is the throughput ratio, so values above `1.000x` favor FeatherAttn. For HND, AITER receives zero-copy transposed views while FeatherAttn consumes HND directly.

Raw output is `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.

| Layout | D | AITER geometric mean TFLOPS | FeatherAttn geometric mean TFLOPS | Feather / AITER geometric mean | Feather wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 64 | 30.941 | 34.071 | 1.101x | 9/9 |
| HND | 128 | 31.354 | 34.078 | 1.087x | 8/9 |
| NHD | 64 | 29.248 | 30.643 | 1.048x | 6/9 |
| NHD | 128 | 26.322 | 30.434 | 1.156x | 8/9 |
| All | 64/128 | 29.397 | 32.258 | 1.097x | 31/36 |

### HND

| D | H | N | AITER TFLOPS | FeatherAttn TFLOPS | Feather / AITER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 4096 | 33.981 | 35.898 | 1.056x |
| 64 | 16 | 8192 | 32.144 | 33.999 | 1.058x |
| 64 | 16 | 16384 | 31.747 | 33.580 | 1.058x |
| 64 | 32 | 4096 | 32.698 | 34.870 | 1.066x |
| 64 | 32 | 8192 | 31.047 | 33.793 | 1.088x |
| 64 | 32 | 16384 | 29.395 | 33.654 | 1.145x |
| 64 | 56 | 4096 | 29.749 | 33.776 | 1.135x |
| 64 | 56 | 8192 | 28.972 | 33.386 | 1.152x |
| 64 | 56 | 16384 | 29.140 | 33.761 | 1.159x |
| 128 | 16 | 4096 | 35.015 | 34.262 | 0.978x |
| 128 | 16 | 8192 | 32.737 | 33.444 | 1.022x |
| 128 | 16 | 16384 | 31.653 | 34.120 | 1.078x |
| 128 | 32 | 4096 | 31.800 | 34.731 | 1.092x |
| 128 | 32 | 8192 | 30.872 | 33.641 | 1.090x |
| 128 | 32 | 16384 | 30.822 | 34.419 | 1.117x |
| 128 | 56 | 4096 | 28.927 | 34.007 | 1.176x |
| 128 | 56 | 8192 | 30.109 | 33.727 | 1.120x |
| 128 | 56 | 16384 | 30.628 | 34.372 | 1.122x |

### NHD

| D | H | N | AITER TFLOPS | FeatherAttn TFLOPS | Feather / AITER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 4096 | 32.908 | 32.522 | 0.988x |
| 64 | 16 | 8192 | 32.007 | 31.376 | 0.980x |
| 64 | 16 | 16384 | 31.943 | 31.197 | 0.977x |
| 64 | 32 | 4096 | 31.037 | 31.258 | 1.007x |
| 64 | 32 | 8192 | 28.596 | 29.960 | 1.048x |
| 64 | 32 | 16384 | 22.754 | 29.088 | 1.278x |
| 64 | 56 | 4096 | 29.068 | 30.820 | 1.060x |
| 64 | 56 | 8192 | 28.092 | 29.749 | 1.059x |
| 64 | 56 | 16384 | 28.228 | 29.959 | 1.061x |
| 128 | 16 | 4096 | 31.842 | 31.202 | 0.980x |
| 128 | 16 | 8192 | 30.660 | 30.811 | 1.005x |
| 128 | 16 | 16384 | 24.062 | 30.240 | 1.257x |
| 128 | 32 | 4096 | 26.009 | 30.367 | 1.168x |
| 128 | 32 | 8192 | 21.718 | 30.168 | 1.389x |
| 128 | 32 | 16384 | 21.566 | 27.393 | 1.270x |
| 128 | 56 | 4096 | 26.901 | 30.890 | 1.148x |
| 128 | 56 | 8192 | 27.655 | 31.598 | 1.143x |
| 128 | 56 | 16384 | 28.485 | 31.454 | 1.104x |

### LDS Bank Conflicts

Fresh single-counter profiles of the final production dispatches confirm that both AITER and FeatherAttn are at or near zero LDS bank conflict. On gfx1151, ROCProfiler defines `LDSBankConflict` as the percentage `100 * SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE`, where `0%` is optimal. It is normalized over active LDS indexing and is not a direct whole-kernel slowdown percentage.

| Layout | D | H | N | AITER conflict | FeatherAttn conflict |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 64 | 16 | 4096 | 0.000% | 0.000% |
| HND | 128 | 16 | 4096 | 2.849% | 2.702% |
| NHD | 64 | 16 | 4096 | 0.000% | 0.000% |
| NHD | 128 | 16 | 4096 | 2.849% | 2.702% |
| NHD | 64 | 32 | 16384 | 0.000% | 0.000% |
| NHD | 128 | 32 | 16384 | 2.855% | 2.702% |

D=64 is conflict-free at the counter's resolution, including every captured constituent launch of the partition-aware strided H32/N16384 path. D=128 has a small repeatable residual in both kernels; FeatherAttn is slightly lower than AITER rather than worse. Every dispatch within each profile reported the same value. Earlier pressure profiles put FeatherAttn D=128 `ALUStalledByLDS` at only `0.026-0.029%`, so the residual conflict is not a material bottleneck. Raw per-dispatch results are under `~/tmp/feather_attn/phase11_final/lds_bank_conflicts/`.

## Objective

Refactor the current HIP attention implementation into an AITER-style, row-owned FlashAttention dataflow while using FP8 E5M2 to keep Q compact in LDS. Implement the device pipeline with selected CK Tile compile-time tensor, layout, and WMMA facilities, but retain the existing raw HIP/PyTorch integration boundary and low-level gfx11 escape hatches. The intended gain on gfx1151 is higher occupancy from lower VGPR pressure, not higher low-precision WMMA throughput. gfx1151 has no FP8 WMMA path that is useful for this kernel, so every Q fragment is expanded to FP16 immediately before an FP16 WMMA.

The target resource point is:
- no more than 192 used VGPRs per wave, which allocates 192 VGPRs and permits eight resident waves per SIMD;
- no more than 32,768 bytes LDS per workgroup, which permits four workgroups per WGP for an 8-wave workgroup;
- zero private segment and scratch memory;
- negligible or zero excess LDS bank conflicts.

The external operation remains non-causal FP16 attention:

```text
out = softmax(Q @ K^T / sqrt(head_dim)) @ V
```

### Supported Shape And Indexing Contract

The public FP16 operation supports every positive `num_heads`, `head_dim` in `{64,128}`, and every positive sequence length representable by the input tensors and the checked 32-bit address/launch arithmetic. It accepts contiguous HND `[B,H,N,D]` and NHD `[B,N,H,D]` tensors through the explicit `layout="HND"` or `layout="NHD"` argument, with HND as the compatibility default. Q, K, V, and output use the same selected layout; no physical transpose is performed. Head count only scales grid X, layout-specific strides, and uniform base offsets; `{16,32,56}` are benchmark targets rather than dispatch restrictions. Sequence length need not divide 64 or 128. The fixed kernel uses guarded query and key tails; block size is an implementation detail rather than an API restriction.

Use 32-bit pointer-offset arithmetic, tile counters, and loop indices inside the optimized device path whenever their complete range is proven safe. The host launcher must perform the proof with wider arithmetic before narrowing. For every Q/K/V/O tensor, compute the maximum reachable offset from its sizes and actual strides, account for element size if an intermediate is a byte offset, and require every product or sum formed in 32-bit device code to fit signed `int32_t`. Also validate ceiling-divided grid dimensions before narrowing. Use overflow-safe ceiling division such as `n / tile + (n % tile != 0)` rather than `(n + tile - 1) / tile` in a narrow type.

Do not make 64-bit device arithmetic the default merely to avoid defining these bounds. Reject a tensor that exceeds the checked int32 range until a future wide-index specialization exists. Never truncate a stride, sequence length, grid dimension, or tensor offset silently.

Sequence length must not change the kernel's per-workgroup LDS allocation, VGPR design, or compiled kernel shape. It changes only the runtime number of query workgroups and key-tile loop iterations. Do not split LDS/VGPR state as sequence length grows.

The optimized and benchmarked lengths are 4096, 8192, and 16384. Short and non-aligned lengths always use FeatherAttn and remain correctness requirements. Tests use `rtol=atol=0.10` for `N_KV < 1024`, where Q E5M2 error has less key averaging, and retain `rtol=atol=0.05` for `N_KV >= 1024`. Unsupported masks, causal attention, dropout, non-contiguous tensors, and other head dimensions fail explicitly.

## Why The Current Kernel Must Be Replaced Structurally

The current `fwd_kernel_kv_staged_d128` is not a suitable base for incremental Q packing alone:
- it quantizes K and V in a separate kernel and reads prepacked E5M2 K/V;
- it materializes the complete score/probability tile in the `Si` LDS array;
- only `Br` scalar threads perform the row softmax work;
- QK tile ownership and output tile ownership differ across waves;
- P is read back from LDS for PV;
- K and V occupy two separately allocated FP8 LDS arrays;
- the Python wrapper autotunes `(64, 32, 8)` and `(128, 32, 8)` implementations whose dataflow is different from the intended kernel.

Previous attempts to add register P/PV to that ownership model either required cross-wave sharing or spilled. The new kernel instead assigns one 16-row query group to each wave. That wave produces the four QK score fragments for those rows and owns all eight output fragments for the same rows. P therefore stays within the producing wave.

The refactor was developed beside the existing kernel. After qualification, the legacy FP8-K/V, prepacked, autotuning, fallback, and BF16 attention paths were removed. Production now uses a shared implementation header, four parallel instantiation units, and one checked Torch binding unit.

## Baseline Evidence

An exploratory AITER `128x64`, 8-wave build at `B=1, H=24, S=4096, D=128` measured:

| Metric | AITER control |
| --- | ---: |
| Median runtime | 6.346 ms |
| Throughput | 32.49 TFLOPS |
| Used VGPRs | 233 |
| Allocated VGPRs | 240 |
| LDS | 32,768 bytes |
| Private segment | 0 bytes |
| VGPR-limited waves/SIMD | 6 |

PMC measurements gave `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES = 23.7` active waves per WGP, almost exactly the 24-wave WGP ceiling implied by six waves per SIMD. The occupancy limit is therefore real rather than merely theoretical.

Forcing the unchanged AITER kernel to 192 VGPRs generated 216 bytes of private memory per thread and regressed to 9.16 ms. Smaller block controls also failed to reduce resources cleanly. The implementation must shorten live ranges and remove state rather than use a compiler occupancy hint to force allocation.

`H=24` is not part of the new supported-head benchmark contract; this result only established the resource argument. These numbers are exploratory controls, not permanent benchmark claims. Record new same-run baselines for all six `(H, S)` combinations in `{16, 32, 56} x {4096, 8192}` before each performance experiment because ROCm version, clock, and thermal state materially affect this machine.

The Phase 0 event-based control run used 5 warmups and 30 measured launches per provider. Its six-shape legacy timing table, p20/p80 intervals, and environment metadata remain in `~/tmp/feather_attn/phase0/controls.json`; those historical speeds are superseded by the final matrix above. Every generated AITER control used eight waves, 32,768 bytes of dynamic LDS, 233 used VGPRs, and zero private segment.

## Implementation Substrate Decision

### Recommendation

Implement FeatherAttn primarily as a bespoke CK Tile device kernel, not as an unchanged CK Tile FMHA specialization and not as a monolithic raw HIP kernel. Keep host C++ responsible for PyTorch registration, validation, checked dispatch, and launch. Use CK Tile only where its compile-time abstractions preserve the intended dataflow without adding state.

This split is preferred because:
- CK Tile already models static distributed tensors, tile distributions, tensor descriptors, windows, vectorized tile movement, gfx11 WMMA operand layouts, and the WMMA C-to-A permutation needed by PV;
- those facilities remove substantial fragment-mapping and LDS-addressing risk from a raw HIP implementation;
- the stock FMHA pipeline has the right broad row-owned score/softmax/PV structure, but its complete FP16 Q tile remains live in registers, which is exactly the state FeatherAttn must remove;
- existing Q-in-LDS CK Tile variants still load the complete expanded Q tile before GEMM and therefore do not implement the required packed-Q lifecycle;
- CK Tile's generic gfx11 `bf8_t` conversion path is software conversion and is outside the Q-unpack instruction budget;
- resource use is determined by emitted code, not by the abstraction boundary, so the custom pipeline must still pass the same metadata, ISA, and profiler gates as raw HIP.

Do not derive this kernel by adding a policy flag to `BlockFmhaPipelineQRKSVS`. The Q lifecycle changes the scope and operand interface of the QK block GEMM itself. Create a dedicated fixed-shape FeatherAttn pipeline that loads and expands one Q fragment, reuses it across four WMMAs, and then lets the fragment die.

### Stock CK Tile Compile Control

A compile-only control was generated from Composable Kernel revision `7b4d28bcff` with ROCm clang `23.0.0git`, targeting `gfx1151`. It used the shipped no-bias, no-mask FP16 batch specialization with `D=128`, `128x64` scores, and eight wave32 waves. Project-supported compiler flags did not change the result. An aligned-only ablation then toggled only the two sequence-padding traits to false; that variant is not shipped and is not a production fallback.

| Metric | Shipped tail-capable `pssk` | Aligned-only `npad` ablation |
| --- | ---: | ---: |
| VGPRs | 240 | 215 |
| VGPR spills | 33 | 0 |
| LDS | 9,216 bytes | 9,216 bytes |
| Private segment | 136 bytes/thread | 0 bytes/thread |
| FP16 WMMAs in the static kernel | 64 | 64 |
| `ds_load_b128` in the static kernel | 128 | 128 |

The shipped control fails both the 192-VGPR and zero-scratch gates. The aligned-only ablation removes the spill but still exceeds the VGPR gate by 23 registers. These controls confirm that CK Tile is useful infrastructure but that the shipped FMHA specialization is not a candidate implementation or performance baseline for FeatherAttn.

### Ownership Boundary

| Layer | Responsibility |
| --- | --- |
| C++/PyTorch shell | Operation registration, tensor validation, wide host arithmetic, guarded narrowing, compact kernel arguments, and launch |
| CK Tile core | Static descriptors, tile distributions, distributed tensors, windows, vectorized K/V movement, wave-local reductions, and gfx11 FP16 WMMA wrappers |
| Custom FeatherAttn pipeline | Fixed wave ownership, exact 32 KiB LDS partition, Q8 staging, fragment-scoped QK, log2 online softmax, P consumption, K/V phase reuse, tails, and output epilogue |
| Custom gfx11 primitives | Raw-byte E5M2 encode/decode, one-load/eight-permutation Q expansion, and any narrower C-to-A bridge needed to avoid generic temporary state |
| Intrinsics | `v_perm_b32`, `permlanex16`, shuffle/reduction operations, FP16 WMMA builtins, `exp2`, barriers, and waits when CK wrappers emit the required sequence |
| Inline assembly | Only a measured fallback for a hot primitive whose intrinsic form emits extra conversion, movement, spills, or unusable scheduling |

Use raw packed integers for Q8 in LDS rather than CK Tile `bf8_t` arithmetic. The Q encoder and decoder must define E5M2 rounding and bit placement explicitly. Start with CK Tile's gfx11 WMMA wrapper and `PermuteWarpGemmCToA`; replace either with a narrower local primitive only if generated ISA or liveness fails the gate.

Do not use the stock FMHA code generator, feature dispatch matrix, generic mask/bias/dropout machinery, or host runner. The one fixed kernel needs a small problem type, one custom device pipeline, and a compact launcher. This keeps unsupported features out of the template instantiation and prevents accidental live state from generic branches.

During evaluation, include CK Tile from the read-only `~/rocm-libraries` checkout and record its revision. A committed production build must use a pinned, redistributable CK Tile dependency or vendored header snapshot. It must not depend implicitly on an AITER installation, the evaluator's home-directory checkout, or whichever CK headers happen to be installed system-wide.

## Fixed Kernel Dataflow

### Wave Ownership

The workgroup covers `128 x 64` attention scores. Wave `w` owns query rows:

```text
[16 * w, 16 * w + 15], w in [0, 7]
```

For its 16 rows, each wave owns:
- eight Q fragments, one for each 16-element D slice;
- four FP32 QK C-fragments, one for each 16-column slice of `Bc=64`;
- four FP16 P A-fragments after the C-to-A conversion;
- eight FP32 output C-fragments, one for each 16-column D slice;
- distributed online-softmax maximum and sum state.

K and V tiles are loaded cooperatively by the whole workgroup and consumed by all eight waves.

### LDS Allocation

The steady-state LDS allocation is exactly 32 KiB:

| Region | Logical contents | Bytes |
| --- | --- | ---: |
| Q8 | `128 x 128` E5M2 | 16,384 |
| KV | one `64 x 128` FP16 K or V tile | 16,384 |
| Total | persistent Q8 plus phase-reused K/V | 32,768 |

Do not add row padding, a second K/V buffer, an LDS score tile, or LDS m/l arrays. Any byte above 32 KiB reduces the workgroups that fit per WGP from four to three and removes the intended occupancy gain.

The K and V phases reuse the same 16 KiB region. Keep `PRE_LOAD_V=False` semantics: stage K, finish QK and softmax, then overwrite the region with V and perform PV.

### Q8 Lane-Major Layout

Q uses a padding-free layout arranged exactly for the per-wave WMMA load:

```text
q8_lds[wave][d_tile][lane16][byte_in_fragment]

dimensions = [8][8][16][16]
bytes      = 8 * 8 * 16 * 16 = 16,384
```

For a WMMA fragment, lanes 0-15 load their corresponding `lane16` record and lanes 16-31 load the same addresses as lanes 0-15. This supplies the replication required by wave32 WMMA. One `ds_load_b128` returns 16 packed E5M2 values per lane. Four packed DWORDs are expanded into the eight FP16 operand VGPRs with eight `v_perm_b32` operations.

The 16-byte records for consecutive `lane16` values are consecutive in LDS. This should give balanced 128-bit bank access without padding, but the layout is not accepted until `LDSBankConflict` is measured.

### Q Conversion And Log2 Pre-Scaling

Each Q element is converted once while the workgroup stages its Q tile. The preferred value stored in LDS is:

```text
q8 = e5m2_round(Q_fp16 * softmax_scale * log2(e))
```

For `D=128`, `softmax_scale = 1 / sqrt(128)`. Storing Q in log2-softmax units means QK produces log2 logits directly. The online softmax becomes:

```text
s2        = dequant(q8) @ K
m2_new    = max(m2_old, row_max(s2))
alpha     = exp2(m2_old - m2_new)
p         = exp2(s2 - m2_new)
l_new     = alpha * l_old + row_sum(p)
O_new     = alpha * O_old + p @ V
```

The final output is `O / l`. If an LSE output is added later, convert once with:

```text
lse_natural = (m2 + log2(l)) * ln(2)
```

This pre-scaling removes repeated score scaling and repeated `log2(e)` multiplications from every key tile. It is important to the instruction budget; do not quantize unscaled Q and reintroduce those multiplications in the hot loop.

Use true RNE first. Also implement a fixture variant for round-to-nearest with ties upward, formed by adding `0x80` independently to each FP16 bit lane before retaining the high byte. Numerical probes found it indistinguishable from RNE for this use and cheaper than the current tie-even helper. Literal truncation is not the default: Q-only truncation showed a strong magnitude bias and about twice the output error of RNE. A calibrated truncation compensation is a future ablation, not part of the first kernel.

Keep Q staging in a lexical scope before declaring or initializing the large output accumulator array. One-time conversion temporaries must not overlap the steady-state accumulator live range and inflate the kernel-wide VGPR count.

### QK Loop

For each of the eight D slices:
- Load one packed Q8 fragment from LDS.
- Expand it to one FP16 WMMA operand with eight `v_perm_b32` instructions.
- Load the four FP16 K operands for the four 16-column score fragments.
- Issue four independent FP16-to-FP32 WMMAs using the same expanded Q.

The Q expansion cost is therefore amortized across four WMMAs. The first implementation streams all eight Q fragments. Do not retain packed Q in VGPRs; lane replication and the packed source live range consume too many registers.

### Register Softmax And P Conversion

Keep all four score fragments in FP32 registers. Perform row maximum and row sum reductions with gfx11 lane permutation/shuffle operations matching the WMMA C layout. Keep `m2` and `l` distributed in registers. There is no `Si`, `alpha`, or `inv_l` LDS allocation.

Convert one 16-column probability fragment at a time from the WMMA C layout to the FP16 WMMA A layout. Reuse the proven gfx11 C-to-A pattern:
- convert the FP32 C values to packed FP16;
- exchange the paired row with `permlanex16`;
- interleave the local and exchanged values with `v_perm_b32`;
- immediately consume the resulting P fragment in eight PV WMMAs.

P remains FP16. Packing P as E5M2 does not remove long-lived state in this dataflow and would add both conversion instructions and accuracy loss.

### PV And Output

Maintain eight FP32 output fragments per wave for the full key loop. Before each PV update, multiply all output fragments by `alpha`. Each P fragment is reused across all eight D output tiles while FP16 V fragments are read from the shared KV region.

At the end, divide each output value by its distributed row sum and store FP16 directly in the selected HND `[B,H,N,D]` or NHD `[B,N,H,D]` layout. Do not round-trip output accumulators through LDS.

### Synchronization Budget

The steady-state key-tile loop may use at most the four phase barriers required to reuse the KV buffer:
- ensure prior V readers are finished before K overwrite;
- publish the staged K tile;
- ensure K readers are finished before V overwrite;
- publish the staged V tile and protect it before the next K overwrite.

Do not add a barrier for Q fragment loads, softmax, P conversion, or individual D slices. Q occupies a disjoint persistent LDS region.

## Resource And Instruction Budget

The AITER control keeps 64 Q VGPRs live across the complete key loop. Replacing that state with a packed Q load and one expanded fragment should save 52-56 VGPRs at the QK peak:

```text
233 baseline VGPRs - 64 expanded Q + 8..12 streamed Q = 177..181 VGPRs
```

The custom CK Tile/HIP translation unit will not allocate exactly like Triton, so this is a design estimate, not a waiver of measurement. The hard acceptance threshold remains 192 used VGPRs.

Per key tile and per wave, the intended added Q transport is:

| Instruction class | Added count |
| --- | ---: |
| `ds_load_b128` | 8 |
| `v_perm_b32` E5M2 expansion | 64 |
| barriers | 0 beyond the four phase barriers |

The current AITER loop has 64 WMMAs and 128 LDS loads per key tile. PMC-based estimates put the unoptimized Q streaming increase near 14.4% of VALU instructions and 5.7% of LDS instructions. Log2 Q pre-scaling is expected to remove much of the repeated score-scale/exp2 preparation work and offset most of the added VALU issue cost. Verify this in generated ISA; do not assume the compiler performs the intended simplification.

Do not use `waves_per_eu`, `__launch_bounds__`, or a maximum-register flag to force 192 VGPRs if doing so creates spills. Launch attributes may document the final resource-qualified kernel, but structural liveness must produce the resource result first.

## Numerical Policy

The first kernel quantizes only Q. K, V, and P stay FP16. This is deliberate:
- Q is long-lived in the AITER-style dataflow, so compact Q changes occupancy.
- K and V fragments are short-lived; packed FP8 sources must coexist with their expanded FP16 operands and do not provide the same VGPR saving.
- Direct FP8 K/V decode would add eight permutations per WMMA.
- P is transient and already consumed in register layout, so FP8 P adds work without removing a persistent allocation.

Exploratory random-attention results at input scale 1.0 were:

| Variant | Output relative L2 |
| --- | ---: |
| Q E5M2 RNE | 0.0541 |
| Q E5M2 RNE with log2 pre-scaling | 0.0535 |
| Q E5M2 half-up with log2 pre-scaling | 0.0535 |
| Q E5M2 literal truncation | 0.1217 |
| Q truncation with empirical scale compensation | 0.0557 |
| QKPV E5M2 RNE | 0.1078 |

These probes justify the implementation direction but are not a model quality result. The long-sequence production gate remains comparison with FP16 attention using:

```text
abs(out - ref) <= 0.05 * abs(ref) + 0.05
```

For `N_KV < 1024`, the separately qualified gate is `abs(out - ref) <= 0.10 * abs(ref) + 0.10`. Also report relative L2, maximum absolute error, maximum normalized tolerance ratio, and the fraction of elements passing the tolerance.

## Refactor Strategy

### File-Level Changes

`kernel_attn/hip/featherattn_kernel.h` contains the shared D=64/D=128 and HND/NHD CK Tile device template, E5M2 Q helpers, K/V LDS layouts, online softmax, tied gfx11 WMMA wrapper, and checked launch helper. Layout is compile-time specialized so global address selection adds no runtime branch to the load/store loops.

`kernel_attn/hip/featherattn_launch.h` defines only the compact launch argument structure and 16 specialization declarations. It keeps CK Tile and Torch out of the host/kernel ABI boundary.

`kernel_attn/hip/featherattn_{aligned,query_tail,key_tail,query_key_tail}.cu` each instantiate one tail mode for D=64/D=128 and HND/NHD. Keeping tail modes in separate translation units permits four concurrent hipcc jobs and limits recompilation after local changes.

`kernel_attn/hip/hip_kernel.cpp` owns stable Torch registration, dtype/device/layout/shape checks, wide host arithmetic, head-dimension and tail dispatch, and explicit launch errors. It is host-only so Torch and CK Tile HIP headers do not collide.

`kernel/hip/utils.py` accepts multiple extension sources plus extension-specific host and HIP flags. Linux retains the normal HIP wrapper; Windows alone uses the validated `-nohipwrapperinc` workaround.

`kernel_attn/hip/hip_kernel.py` builds the direct-only extension against `FEATHEROPS_CK_TILE_ROOT` or the default CK checkout and exposes `feather_attn(q,k,v,layout="HND")`. CK Tile is an unconditional compile-time dependency: unavailable headers fail the extension build instead of producing an empty extension with a runtime error path.

`test_attn_hip.py` compares against AITER in both physical layouts and covers both dimensions, benchmark and arbitrary head counts, short/aligned/tail lengths, independent Q/KV lengths, and batch greater than one. HND is exposed to AITER as a zero-copy NHD logical view; no reference transpose copy is included.

`benchmark_attn_hip.py` compares AITER and FeatherAttn across layouts, head dimensions, heads `{16,32,56}`, and lengths `{4096,8192,16384}`. `FEATHER_ATTN_BENCH_LAYOUTS` and `FEATHER_ATTN_BENCH_HEAD_DIMS` filter the corresponding axes.

### Phase 0: Freeze Controls (Complete)

- Record same-run AITER and legacy HIP results for `(H, S)` in `{16, 32, 56} x {4096, 8192}`.
- Save generated metadata and disassembly for the AITER `128x64x8` kernel.
- Record used/allocated VGPRs, SGPRs, LDS, private segment, scratch, static WMMA, LDS, permutation, conversion, and barrier counts.
- Record `LDSBankConflict`, `VALUInsts`, and an occupancy proxy with rocprofv3.

Exit condition: reproducible control measurements and commands are written into the experiment ledger below.

### Phase 1: Q8 LDS Fixture (Complete)

Compile a small custom CK Tile fixture, without instantiating the stock FMHA pipeline, that:
- loads a `128x128` FP16 Q tile;
- applies log2 pre-scaling and E5M2 rounding;
- writes the exact 16 KiB lane-major Q layout;
- reloads all eight fragments with the production address calculation;
- expands each fragment with the production `v_perm_b32` sequence;
- writes decoded values or multiplies by an identity WMMA operand for checking.

Represent packed Q as raw bytes or DWORDs, not `bf8_t`. Test true RNE and half-up conversion. Inspect ISA to ensure the steady fragment load is one `ds_load_b128` plus eight permutations and does not retain the whole Q tile in VGPRs. Compare CK Tile/intrinsic forms of the primitive before considering inline assembly.

Exit conditions:
- fixture output matches its selected E5M2 reference;
- Q LDS allocation is exactly 16,384 bytes;
- no private segment or scratch;
- `LDSBankConflict` is negligible or zero;
- no padded stride is required.

Stop the design if an exact 16 KiB conflict-clean Q layout cannot be produced.

The fixture passed with both true RNE and half-up rounding. Each variant used 68 VGPRs, 16,384 bytes LDS, zero private/scratch memory, and zero `LDSBankConflict`. The post-barrier decode emitted eight `ds_load_b128` operations and 64 `v_perm_b32` operations per wave. RNE is the production default; half-up remains an isolated fallback ablation. Source and ISA artifacts are under `~/tmp/feather_attn/phase1/`.

### Phase 2: Row-Owned QK Skeleton (Complete)

Add the dedicated fixed-shape FeatherAttn CK Tile problem and pipeline, then implement:
- fixed wave-to-query-row ownership;
- fused Q staging from external FP16 input;
- one 16 KiB FP16 K stage;
- four FP32 score fragments per wave;
- streamed Q8 fragment expansion and four-WMMA reuse;
- optional diagnostic score output for a fixture build.

Invoke the gfx11 warp GEMM at fragment scope. Do not call a block GEMM interface that requires a complete expanded Q distributed tensor. Do not implement PV through the old `Si` path. The purpose of this phase is to validate Q/K fragment mappings and generated liveness for the final ownership model.

Exit conditions:
- QK matches the quantized-Q reference;
- Q expansion has the planned static instruction count;
- there is no score LDS allocation;
- the partial kernel shows no structural spill warning.

The fixed QK fixture now stages an exact 16 KiB XOR-swizzled FP16 K tile beside the persistent 16 KiB Q8 tile, invokes 32 gfx11 FP16 WMMA operations per wave, and writes only diagnostic scores to global memory. It matches the quantized-Q reference with maximum absolute error `9.54e-7`; the compiled QK entry uses 129 VGPRs, 32,768 bytes LDS, zero private memory, and zero spills. Static counts are 32 `v_wmma_f32_16x16x16_f16`, 80 `ds_load_b128`, 12 `ds_store_b128`, and 192 `v_perm_b32`; the measured `LDSBankConflict` counter is zero. The extra permutations include Q encode/vector assembly, while the post-barrier Q decode remains the planned eight-load/64-permutation sequence. Artifacts are under `~/tmp/feather_attn/phase2/`.

### Phase 3: Register Softmax And P Fixture (Complete)

Implement distributed row max/sum reductions and online state in log2 units. CK Tile wave-reduction primitives are acceptable only if ISA confirms that they remain wave-local and introduce no LDS or cross-wave synchronization. Extend the WMMA fragment fixture so it validates the exact C-to-A P conversion used by the production wave ownership. Test P through an identity PV WMMA before combining it with real V.

Exit conditions:
- probability rows agree with the quantized-Q reference within the expected FP16 P/exp2 error;
- C-to-A P conversion passes for all four column fragments;
- no `Si`, `alpha`, `inv_l`, or row-state LDS arrays exist;
- all eight waves participate in softmax and PV preparation.

The fixture now performs all row max/sum work with gfx11 DPP operations and keeps scores, probabilities, and row state in registers. The first direct use of CK's block-PV C-to-A helper was rejected because its output is intentionally arranged for CK's transposed block GEMM path, not the standalone A-row WMMA used by this fixture. The accepted gfx1151 primitive converts each FP32 C fragment to FP16, interleaves the two C lane groups with `v_permlanex16`/`v_perm_b32`, then uses a packed four-stage `v_permlane16` butterfly transpose. This removed all 512 `ds_bpermute_b32` operations emitted by the validated scalar-shuffle fallback.

Zero, random, wide-random, and structured one-hot cases match the FP16-quantized log2-domain softmax reference with maximum absolute error at most `7.63e-6`; every identity-PV output is bit-exact with P, and measured row sums stay in `[0.999707, 1.000260]`. The compiled entry uses 90 VGPRs (96 allocated), 25 SGPRs, zero LDS, zero private memory, and zero spills. Static counts are four `v_wmma_f32_16x16x16_f16`, 128 `v_permlane16_b32`, 16 `v_permlanex16_b32`, 32 `v_perm_b32`, 64 DPP moves, 32 lane reads, and zero `ds_bpermute_b32`. Eight profiled dispatches reported `LDSBankConflict=0.0` and `VALUInsts=1227.0`. Artifacts are under `~/tmp/feather_attn/phase3/`.

### Phase 4: Full FP16 PV And Output (Complete)

Add:
- phase-reused FP16 V staging in the same 16 KiB KV region;
- eight persistent FP32 output fragments per wave;
- online output rescaling by `alpha`;
- four P fragments, each reused across eight PV WMMAs;
- final normalization and guarded FP16 output stores.

Run aligned self-attention first: `N=N_KV` and both dimensions divisible by the fixed block sizes.

Exit conditions:
- `test_attn_hip.py` passes the quantized-Q diagnostic and FP16 acceptance gate;
- used VGPRs are at most 192;
- LDS is at most 32,768 bytes;
- private segment and scratch are zero;
- steady-state barriers do not exceed four per key tile;
- LDS bank conflicts remain negligible or zero.

If VGPR usage is above 192, fix liveness in this order:
- scope Q staging before output accumulator initialization;
- consume packed Q DWORDs while overwriting their registers with expanded outputs where the compiler permits;
- remove accidental fragment double buffering;
- convert and consume one P fragment at a time;
- reuse address and reduction temporaries across phases.

Do not force allocation, accept scratch, quantize P, or shrink the block as a substitute for meeting the resource plan.

The aligned D=128 core runs a sequence-independent key loop. Grid X covers every contiguous `(batch, head, 128-row query tile)` combination. The final launcher accepts every positive head count, dispatches compile-time tail variants when needed, requires exact contiguous strides, and proves a representable launch grid plus signed-int32-safe final byte offsets. Unsupported inputs fail explicitly.

Meeting the register gate required two measured changes to the initial Phase 4 sketch. First, QK fragments are accumulated with a `3+1` schedule and compacted to FP16 before softmax, so the second group rereads the eight Q fragments. This retains 32 QK WMMAs but changes the steady key-tile body from eight to sixteen packed-Q LDS loads and from 64 to 128 Q-decode permutations. The extra score quantization is accepted provisionally because aligned fixtures from 64 through 8192 keys stay within `4.0e-4` relative L2 of the quantized-Q/FP16-score reference and pass the public FP16 tolerance at the production lengths. It must still be judged by the six-shape performance matrix and full public tests.

Second, both QK and PV use a local gfx1151 inline-assembly WMMA wrapper with a tied read/write accumulator. The intrinsic generated duplicate loop-carried accumulator tuples: the dynamic kernel reached 256 VGPRs even though the same straight-line body used 143 after tying both WMMA phases. The tied wrapper plus per-iteration rematerialization of swizzled V addresses produces the accepted dynamic entry without changing instruction semantics. This is the isolated compiler defect required by the inline-assembly policy.

The first accepted dynamic `3+1` entry used 192 VGPRs, 41 SGPRs, 32,768 bytes LDS, zero private memory, and zero spills. Its static loop body contained 64 FP16 WMMAs, 144 `ds_load_b128`, 12 `ds_store_b128`, 192 `v_perm_b32`, 192 `v_permlane16_b32`, 16 `v_permlanex16_b32`, 64 DPP moves, 48 lane reads, 48 exponentials, eight logarithms, four barriers, and zero `ds_bpermute_b32`. At `N_KV=4096`, eight profiled fixture dispatches reported `LDSBankConflict=0.0`, `VALUInsts=101911.0`, and no runtime scratch. Artifacts are under `~/tmp/feather_attn/phase4/`.

The first production checkpoint at `B=1, H=16, S=4096, D=128` passed the public FP16 tolerance with maximum absolute error `0.01462` and relative L2 `0.05404`, but failed the performance gate. In the same run, AITER measured `4.263 ms` (`32.24 TFLOPS`) and the aligned FeatherAttn path measured `5.078 ms` (`27.07 TFLOPS`), a `19.1%` median regression. Do not qualify the current `3+1` score-compaction schedule. Phase 5 must first remove the second packed-Q decode pass or demonstrate a different register schedule that recovers its cost without exceeding 192 VGPRs.

### Phase 5: Core Optimization

Only after Phase 4 passes all resource gates:
- schedule the next compact Q load early enough to hide LDS latency without extending more than four packed source VGPRs;
- interleave Q permutations, K LDS loads, and the four independent QK WMMAs;
- verify that log2 pre-scaling removed repeated score-scale and `log2(e)` work;
- use wide global and LDS loads/stores for all tile movement;
- inspect literal-heavy `v_perm_b32` encodings only if profiling identifies instruction fetch as a limiter;
- test caching one expanded Q fragment only if the uncached kernel uses at most 184 VGPRs. One cached fragment adds eight long-lived VGPRs and changes the per-key budget from 64 to 56 permutations and from eight to seven Q loads.

Keep CK Tile wrappers when they emit the intended instructions and live ranges. Introduce a local intrinsic wrapper when a generic CK operation adds state. Use inline assembly only after an isolated intrinsic fixture demonstrates a material instruction, spill, or scheduling defect; never use it to implement tensor bounds, pointer arithmetic, or control flow.

Every optimization is a separate experiment with before/after correctness, resource metadata, ISA counts, and timings. Revert failed experiments without combining them with unrelated changes.

The first Phase 5 ablation replaced `3+1` compaction with one-pass FP32 QK accumulation and delayed FP16 P conversion until C-to-A. Contrary to the earlier liveness estimate, the simpler schedule compiles at 191 VGPRs, 41 SGPRs, 32,768 bytes LDS, zero private memory, and zero spills. It removes eight `ds_load_b128` and 64 `v_perm_b32` instructions from the static key-tile body, leaving 136 loads and 128 permutations. Correctness is unchanged at the public level, but the same-shape timing remains outside the gate: FeatherAttn measured `5.118 ms` versus AITER `4.378 ms`, a `16.9%` regression. Keep the one-pass schedule because it is lower-work and lower-resource, but do not attribute the remaining gap to packed-Q rereads; continue with V staging, synchronization, and instruction-scheduling ablations.

The first V-staging ablation copied Triton's four-row by eight-column per-thread load shape and replaced the packed cross-lane 16x16 transpose with an in-thread transpose followed by eight `ds_store_b64` operations. It retained 191 used/192 allocated VGPRs and reduced the profiled `VALUInsts` counter from `101911.0` to `85528.0`, but the existing LDS XOR maps every fixed-column store to too few banks. `LDSBankConflict` rose from zero to `29.4028`, and `H=16, S=4096` regressed to `5.965 ms` versus AITER `4.402 ms`. Reject that layout. A follow-up may change only the V LDS swizzle and must prove both store and PV-load conflicts before it is retained.

The follow-up uses Triton AMD's rotating-shared phase exactly: the physical N chunk is `n_chunk XOR ((d % 8) XOR ((d / 8) % 8))`. It lowers `LDSBankConflict` to `2.7016`, profiles `88218.0` `VALUInsts`, and measures `5.028 ms` versus AITER `4.456 ms`, a `12.8%` regression before the transposed-score optimization. This recovered `15.7%` relative to the rejected store-conflicted V layout while preserving correctness and the 191-VGPR resource result.

The decisive Phase 5 change transposes the QK result at the WMMA level: each wave computes `K * Q^T`, so each lane owns one query row and two lane groups hold alternating key rows. Softmax becomes one local 32-value reduction plus a `permlanex16` pair exchange; output rescaling uses compile-time `v_readlane` broadcasts from the query-row lanes. The transposed C fragment then needs only the interleave half of the gfx1151 C-to-A conversion, four `v_permlanex16` and eight `v_perm_b32` operations per P fragment, instead of the full four-stage lane transpose. The first corrected multi-tile run matches the quantized-Q reference behavior and passes the public FP16 tolerance at `H=16, S=4096`; it compiles at 191 used/192 allocated VGPRs, 23 SGPRs, 32,768 bytes LDS, zero private memory, and zero spills. In the same run FeatherAttn measured `4.101 ms` versus AITER `4.384 ms`, a `6.5%` speedup. Accept this as the aligned core and remeasure all supported heads before adding tails.

The accepted D=128 core's static body contains 64 FP16 WMMAs, 136 `ds_load_b128`, eight `ds_store_b128`, eight `ds_store_b64`, 128 `v_perm_b32`, zero `v_permlane16_b32`, 18 `v_permlanex16_b32`, 34 exponentials, one logarithm, four barriers, and zero `ds_bpermute_b32`. Eight `N_KV=4096` profile dispatches each reported 192 allocated VGPRs, 32,768 bytes LDS, zero scratch, `VALUInsts=43861.0`, and `LDSBankConflict=2.7016`. The residual conflict is accepted as negligible: it is less than one tenth of the rejected mapping's `29.4028`, and every aligned and representative tail benchmark beats AITER. Artifacts are under `~/tmp/feather_attn/phase5/`.

### Phase 6: Performance Qualification

Benchmark all six `(H, S)` combinations in `{16, 32, 56} x {4096, 8192}` against current AITER and the frozen legacy HIP path. Use identical inputs where layout permits and report median plus the 20th/80th percentile interval. Report each shape separately; do not average the three head counts into one acceptance number.

Acceptance gates:
- no regression at `S=4096` for any supported head count; a production replacement should preferably show at least a 5% repeatable median win on the shape matrix to justify the new complexity;
- no more than a 3% repeatable regression at `S=8192` for any supported head count;
- no private/scratch memory in profiled dispatches;
- measured occupancy reaches the four-workgroup/eight-wave resource point;
- `LDSBankConflict` remains negligible or zero;
- instruction growth agrees with the planned Q load/unpack budget;
- fused Q conversion is included in the measured kernel time.

The expected opportunity is approximately 3-10%, not the nominal 33% increase from six to eight resident waves. Added Q LDS traffic and permutation issue can consume part of the occupancy gain. If the qualified kernel remains slower than AITER after scheduling work, stop rather than adding FP8 K/P/V complexity.

Historical Phase 5/6 event matrices qualified the transposed-score HND core and justified extending the benchmark surface to `S=16384`. Their raw samples remain under `~/tmp/feather_attn/phase5/` and `phase6/`, but their full tables are superseded by Final Benchmarks.

PyTorch `scaled_dot_product_attention` cannot serve as the production-shape oracle in the current PyTorch/ROCm environment: it returns `hipErrorInvalidValue` before FeatherAttn launches for all six aligned shapes. Preserve that failed-oracle artifact at `~/tmp/feather_attn/phase5/aligned_public_correctness.json`; do not classify it as a FeatherAttn correctness failure. Use chunked FP32 `QK -> softmax -> PV` reference computation for the final elementwise gate so the full attention matrix is never materialized.

The chunked FP32 oracle passes all six aligned production shapes through the public `attn_hip` dispatch with zero failed elements. Relative L2 ranges from `0.05363` to `0.05395`, maximum absolute error is at most `0.02125`, and the worst normalized tolerance ratio is `0.380`. Each shape checks every output element against `abs(error) <= 0.05 * abs(reference) + 0.05`; no tolerance change is needed. Full metrics are in `~/tmp/feather_attn/phase5/aligned_public_chunked_correctness.json`.

### Phase 7: Tails, Real Shapes, And Cutover

Keep the same `128x64x8` block and add guarded tails:
- use overflow-safe ceiling division for query and key tile counts;
- zero-fill inactive Q/K/V loads;
- force invalid key logits to negative infinity before row reductions;
- exclude invalid keys from row sums;
- guard final stores for query tails.

Use int32 offsets and counters in the kernel only after the host-side wide-arithmetic checks prove that all reachable strided element and byte offsets, tile counts, and launch dimensions fit. Add boundary-focused host tests for the checker without allocating impractically large tensors.

Test benchmark and arbitrary head counts with aligned and non-aligned lengths. Include lengths on both sides of 64- and 128-token boundaries, independent query/KV tails, batch greater than one, and actual shapes from `docs/input_shapes_attn.md`. Short sequences use the same FeatherAttn kernel with the separately documented accuracy gate.

After aligned and tail paths passed correctness and performance gates, the new kernel became the only `attn_hip` implementation. Autotuning, runtime fallback, prepacked K/V helpers, legacy quantization, development fixture APIs, and BF16 attention code were removed.

Tail support is implemented as compile-time query-only, KV-only, and combined variants, leaving the aligned `false,false` entry unchanged. Query loads outside `N_Q` are zero-filled and final stores are guarded; K/V loads outside `N_KV` are zero-filled and corresponding QK logits are forced to negative infinity before softmax. Host grid construction uses overflow-safe ceiling division. All four entries use 191 used/192 allocated VGPRs, 32,768 bytes LDS, zero private memory, and zero spills; SGPR use is 28 aligned, 26 query-tail, 39 key-tail, and 41 combined-tail.

Representative direct D=128 tail-kernel checks for `(N_Q,N_KV)` in `{(1,1),(16,17),(63,64),(64,63),(65,65),(129,128),(128,129),(129,129)}` match the quantized-Q reference with maximum absolute error at most `9.77e-4`. Against the unquantized FP16 reference, E5M2 Q has less key averaging on short sequences, so `N_KV < 1024` uses the documented `0.10/0.10` test gate. Every length uses FeatherAttn; `N_KV >= 1024` retains `0.05/0.05`. Artifacts are under `~/tmp/feather_attn/phase6/`.

Long-tail public validation also passes with zero failed elements for `(H,N_Q,N_KV)` equal to `(16,4095,4095)`, `(16,4097,4097)`, `(16,4096,1025)`, `(32,1500,1024)`, `(32,8800,8800)`, and `(56,5302,5302)`. This covers combined tails, an independent KV tail, an independent query tail, and real LTX/H3 shapes. Relative L2 stays between `0.05344` and `0.05365`; maximum absolute error is at most `0.04676`, and the worst tolerance ratio is `0.729`. Full metrics are in `~/tmp/feather_attn/phase6/long_tail_correctness.json`.

The final repository public-contract test passes `168/168` AITER-backed cases across HND/NHD and D=64/D=128. It covers benchmark heads at `257`, `4096`, and `8192`; general head counts `{1,2,3,4,7,24,30,40,48}`; short lengths from `1` upward; the `1023/1024/1025` accuracy-gate boundary; independent query/KV tails; query and KV tile boundaries; and `B=2`.

Representative tail timing uses five warmups and 30 event samples per provider. FeatherAttn measures `4.538 ms` versus AITER `5.240 ms` at `H=16,N=4097` (`15.5%` faster), `39.544 ms` versus `58.358 ms` at LTX `H=32,N=8800` (`47.6%` faster), and `25.692 ms` versus `32.453 ms` at H3 `H=56,N=5302` (`26.3%` faster). Tail performance is qualified; raw samples are in `~/tmp/feather_attn/phase6/tail_event_benchmarks.json`.

The production host path now proves every reachable FP16 byte offset with `__int128` arithmetic before any device-side int32 narrowing. It requires positive batch, head, query, and KV dimensions; rejects negative or greater-than-int32 strides; checks Q/K/V/output maximum strided byte offsets against signed int32; and validates tile and grid ranges before launch. Boundary tests cover the arithmetic without allocating impractically large tensors.

### Phase 8: D=64 Specialization And Parallel Build (Complete)

D=64 reuses the accepted `128x64x8` ownership and online-softmax template. It halves Q8 LDS to 8 KiB and phase-reused K/V LDS to 8 KiB, for 16 KiB total. K staging issues two vector loads per thread rather than four. V staging uses the same rotating-shared swizzle but maps the two unused D=128 lane-row groups onto additional N rows and stores two FP16 rows per lane. The D=64 query scale is the exact `log2(e)/sqrt(64)` constant.

The implementation lives in `featherattn_kernel.h`. Four small translation units instantiate aligned, query-tail, KV-tail, and combined-tail kernels for both dimensions and layouts; Ninja compiles those units in parallel. The Torch binding is a host-only C++ translation unit and does not parse CK Tile. A clean build and import with 16 device specializations completed in approximately `8.4 s` with `MAX_JOBS=32`.

D=64 resource metadata passes every gate:

| Variant | Used VGPRs | Allocated VGPRs | SGPRs | LDS bytes | Private / spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned | 146 | 152 | 28 | 16,384 | 0 / 0 |
| Query tail | 146 | 152 | 30 | 16,384 | 0 / 0 |
| KV tail | 168 | 168 | 40 | 16,384 | 0 / 0 |
| Combined tail | 149 | 152 | 44 | 16,384 | 0 / 0 |

The aligned D=64 static body contains 32 FP16 WMMAs, 68 `ds_load_b128`, four `ds_store_b128`, eight `ds_store_b32`, 80 `v_perm_b32`, zero `v_permlane16_b32`, 18 `v_permlanex16_b32`, 34 exponentials, one logarithm, four barriers, and zero `ds_bpermute_b32`. Eight `N_KV=4096` profile dispatches each report 152 allocated VGPRs, 16,384 bytes LDS, zero scratch, `VALUInsts=29979.0`, and `LDSBankConflict=0.0`. Metadata, ISA, and profiler artifacts are under `~/tmp/feather_attn/`.

The combined HND/NHD and D=64/D=128 public contract suite passes `168/168` cases. Historical D=64-only timing and profiling artifacts remain under `~/tmp/feather_attn/d64_benchmark/`; current cross-layout performance is reported only in Final Benchmarks.

### Phase 9: Native HND/NHD Layouts (Complete)

The public wrapper accepts explicit `HND` and `NHD` layouts. Both keep D innermost and contiguous, so Q/K/V vector loads and output stores retain their existing vector widths. Layout is a compile-time template parameter. HND preserves the original flattened `(batch,head)` ownership and contiguous row stride; NHD decomposes batch/head and uses `num_heads * D` as the sequence-row stride. The four instantiation units now produce 16 kernels across layout, dimension, and tail mode.

The first correct NHD implementation assigned consecutive grid blocks to query tiles of one head. On power-of-two `H*D` row strides this caused severe memory-partition aliasing; the worst measured point, `D=64,H=32,N=16384`, reached only `4.793 TFLOPS`. The accepted NHD mapping makes head the fastest grid axis, interleaving adjacent head offsets across active workgroups. The same point rises to `22.553 TFLOPS`, and the full NHD matrix becomes competitive with AITER. HND block ordering is unchanged.

All 16 final kernels remain within the resource gates. HND metadata is unchanged. NHD D=64 uses 130 VGPRs for aligned/query-tail and 151 for KV/combined tails; NHD D=128 uses 191 VGPRs for every tail mode. NHD uses 16,384 bytes LDS for D=64 and 32,768 bytes for D=128, with zero private memory and zero SGPR/VGPR spills in every variant.

The Phase 9 full benchmark established that both physical layouts were viable and exposed the long-NHD partition problem that Phase 10 and 11 later addressed. Its raw output remains under `~/tmp/feather_attn/layout_benchmark/final/`, but the historical 36-row table is intentionally omitted. See Final Benchmarks for the current production TFLOPS and AITER ratios.

### Phase 10: Post-Qualification Performance Review And Optimization Plan

The post-qualification review compared FeatherAttn with the active AITER Triton path, CK Tile FMHA pipelines, FlashAttention-CK head grouping, and the SageAttention optimization history. No production implementation files were changed during the review. The main artifacts are under `~/tmp/feather_attn/review/`.

#### Roofline And Long-NHD Findings

Use three separate traffic quantities in all future reports:
- useful unique tensor bytes: one read of Q/K/V and one write of O;
- schedule-implied bytes: Q once, K/V once per 128-row query CTA, and O once;
- measured memory-controller traffic from `FETCH_SIZE` or the equivalent GCEA read-size counter.

Do not infer byte traffic from `L2CacheHit`. It is a request-based percentage. `FETCH_SIZE` is measured controller traffic and includes cache and transaction effects.

For a `128x64` CTA, the dominant dense-attention work and K/V stream are:

```text
FLOPs per CTA = 4 * 128 * N_KV * D
K/V bytes     = 4 * N_KV * D
intensity     = 128 FLOP/byte
```

The no-cross-CTA-reuse memory roof is therefore approximately `32.8 TFLOPS` at the theoretical `256 GB/s` memory ceiling. This is distinct from the `59.4 TFLOPS` FP16 WMMA ceiling.

The isolated long-NHD profile used `B=1,H=32,N=16384`, two warmups, and final-two-dispatch medians. Its traffic pass is self-consistent: timing and traffic come from the same dispatches.

| Provider | D | Time | Measured fetch | Achieved bandwidth | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| FeatherAttn | 64 | 97.372 ms | 15.837 GiB | 174.64 GB/s | 22.584 TFLOPS |
| AITER | 64 | 95.807 ms | 15.675 GiB | 175.68 GB/s | 22.953 TFLOPS |
| FeatherAttn | 128 | 197.169 ms | 31.504 GiB | 171.56 GB/s | 22.306 TFLOPS |
| AITER | 128 | 203.680 ms | 31.840 GiB | 167.85 GB/s | 21.593 TFLOPS |

For FeatherAttn, useful Q/K/V/O bytes are only `0.250 GiB` for D=64 and `0.500 GiB` for D=128. The schedule-implied Q plus repeated K/V reads are `16.062 GiB` and `32.125 GiB`. Measured fetch is close to that schedule, so the long NHD mapping obtains effectively no cross-CTA K/V reuse. Request-based L2 hit rates are only about `2.0-2.3%` for D=64 and `2.7-2.9%` for D=128.

The GCEA `RDRAM_SIZE_REQ` pass independently reports approximately `174.5 GB/s` for FeatherAttn D=64, using ROCm's documented `32 * GCEA_RDRAM_SIZE_REQ / duration` formula. This corroborates `FETCH_SIZE`. The per-instance GCEA bank and system-arbiter event descriptions are blank, so they must not be used to claim a specific partition-aliasing mechanism without a controlled schedule A/B test.

Long-grid occupancy counters are duration-averaged, not static residency limits. The long FeatherAttn grid has 4,096 workgroups and reports approximately `99.6%` SIMD utilization while measured occupancy falls to about `25%` for D=64 and `12%` for D=128. Static resources still permit the already-qualified residency point; the lower duration average does not justify an occupancy redesign by itself.

#### On-Chip Attribution

Static aligned ISA shows a substantial D-independent loop cost:

| Variant | Instructions | WMMA | Exp | Log | Rcp | LDS loads | Permutations | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HND D=64 | 1,175 | 32 | 34 | 1 | 2 | 68 | 80 | 4 | 50 |
| HND D=128 | 1,949 | 64 | 34 | 1 | 2 | 136 | 128 | 4 | 94 |
| NHD D=64 | 1,334 | 32 | 34 | 1 | 3 | 68 | 80 | 4 | 50 |
| NHD D=128 | 2,060 | 64 | 34 | 1 | 3 | 136 | 128 | 4 | 94 |

NHD adds no branch to the core key loop. Its extra static work is principally block decomposition, address construction, scalar work, and scheduling delays.

At `H=16,N=4096`, FeatherAttn executes `1.264x/1.537x` AITER's VALU instructions for HND D=64/D=128 and `1.296x/1.557x` for NHD. AITER uses the same active `128x64`, eight-wave geometry on gfx1151, so it is a useful instruction-schedule control even though its HND memory traffic is not cache-equivalent.

PC sampling for FeatherAttn HND shows:
- D=64: `36.03%` other VALU, `25.35%` LDS reads, `14.03%` synchronization, `9.94%` WMMA, `7.11%` conversion, and `5.76%` transcendental samples;
- D=128: `31.87%` LDS reads and `14.63%` WMMA, with `38.41%` ALU-dependency and `14.93%` waitcnt stalls.

`ALUStalledByLDS` is only `0.024-0.043%`, and LDS issue waits stay below `0.08%` of wave cycles. The sampled LDS stalls are predominantly dependent-address or dependent-data waits, not an LDS queue-full condition. Optimize dependency chains and load/use distance before adding deeper prefetch.

Barrier waits are approximately `12.1%/4.8%` of wave cycles for HND D=64/D=128 and `12.1%/5.4%` for NHD. Wait-count stalls are approximately `10.1%/21.3%` for HND and `11.4%/30.7%` for NHD. D=128 therefore has more waitcnt exposure but almost no register headroom for buffering.

#### Optimization 10A: Bounded LLC-Aware NHD Grouping

Status: accepted for D=128 NHD; rejected for D=64. The retained implementation keeps the Phase 9 head-fast mapping inside each launch and divides the physical head range into sequential launch subsets. Physical NHD strides continue to use the full head count; `head_start` and `launch_heads` affect only block decomposition. HND remains one launch.

The automatic policy is deliberately narrower than the initial model:

```text
per_head_KV_bytes = 4 * N_KV * D
activate only for D=128 and total_KV_bytes >= 1.5 * 32 MiB
group_size = floor(32 MiB / per_head_KV_bytes)
require 4 <= group_size < H; otherwise use one launch
```

This bounds every complete group K/V set by the 32 MiB LLC and handles the modeled `H=56,N=16384,D=128` edge with four-head groups and 14 launches rather than CK's over-capacity seven-head group. Checked wide host arithmetic and signed-int32 device indexing remain intact.

The focused `{4,8,16}` sweep rejected grouping for D=64. At `H=32,N=16384`, groups `{4,8,16}` regressed by `72.0%`, `54.9%`, and `24.8%`; smaller contiguous head subsets revive the partition-pressure behavior that the Phase 9 head-fast grid fixed. D=64 therefore always uses one launch.

A paired 30-sample D=128 NHD qualification compared the automatic policy with an explicitly ungrouped launch. It won all nine benchmark shapes, with a `1.135x` geometric-mean speedup factor. The largest focused gains were `28.4%` at `H=32,N=8192` and `27.6%` at `H=32,N=16384`; the two H=16 cases that were already cache-favorable changed by only `+0.40%` and `+0.06%`.

Isolated profiler passes sum counters across all sublaunches in one API call and use the final two post-warmup calls:

| Shape | Policy | Group | Time | `FETCH_SIZE` | L2 request hit |
| --- | --- | ---: | ---: | ---: | ---: |
| H32/N8192/D128 | Ungrouped | 32 | 46.125 ms | 7.581 GiB | 6.68% |
| H32/N8192/D128 | Accepted | 8 | 36.721 ms | 2.810 GiB | 54.00% |
| H32/N16384/D128 | Ungrouped | 32 | 197.248 ms | 31.435 GiB | 2.71% |
| H32/N16384/D128 | Accepted | 4 | 167.441 ms | 14.822 GiB | 34.75% |
| H56/N16384/D128 | Ungrouped | 56 | 275.769 ms | 56.078 GiB | N/A |
| H56/N16384/D128 | Accepted | 4 | 249.111 ms | 23.361 GiB | N/A |

The independent H32/N16384 GCEA pass reports `31.529 GiB` ungrouped and `13.543 GiB` grouped, corroborating the traffic reduction. GCEA and `FETCH_SIZE` came from separate profiler runs and are not expected to match exactly. Per-sublaunch `SIMD_UTILIZATION` cannot be combined into a meaningful whole-call percentage because rocprof can report values above one on later sequential launches; it is excluded from the acceptance claim.

The complete 36-case matrix passes. D=128 NHD is now `0.972x-1.376x` AITER, versus `0.972x-1.084x` in Phase 9, and its geometric-mean throughput improves `1.113x` over the Phase 9 table. HND and D=64 geometric means change by only `+0.7%`, `+0.4%`, and `+0.1%`, respectively. The public contract passes `168/168`, including arbitrary heads, independent tails, and batch two. Fresh metadata for all 16 kernels reports at most 191 used/192 allocated VGPRs, 32 KiB LDS, zero private memory, and zero spills; Kargs is 56 bytes.

Artifacts are under `~/tmp/feather_attn/phase10_group_*`, including the focused sweep, paired qualification, profile summary, contract log, metadata, and complete matrix. Phase 10A becomes the production baseline for subsequent experiments.

#### Optimization 10B: Persistent Linear Online `(m,l)` State

Status: rejected for D=64; D=128 not attempted. FeatherAttn reconstructs a log-domain state on every key tile:

```text
old_term  = exp2(old_lse - new_max)
tile_term = exp2(tile_max - new_max)
combined  = old_term + tile_term * tile_sum
alpha     = old_term / combined
beta      = tile_term / combined
lane_lse  = new_max + log2(combined)
```

Replace this with the AITER/FlashAttention linear online state:

```text
new_max     = max(running_max, tile_max)
alpha       = exp2(running_max - new_max)
beta        = exp2(tile_max - new_max)
running_sum = alpha * running_sum + beta * tile_sum
output      = alpha * output + beta * P @ V
running_max = new_max
```

Normalize output once in the epilogue with `rcp(running_sum)`. If probabilities remain `exp2(score - tile_max)`, multiplying P by `beta` still requires two state exponentials per tile. To reach AITER's 33-exponential structure, form P directly as `exp2(score - new_max)` and keep only `alpha = exp2(running_max - new_max)`. The exact generated dataflow, not source-level algebra, determines whether the second state exponential is actually removed.

The primary target is to remove the per-key-tile logarithm and reciprocal and shorten the serial softmax/update chain. A secondary target is reducing 34 exponentials to 33 without increasing reduction or broadcast cost. The 32 probability exponentials remain. Treat the standalone expected gain as `2-6%`, not as removal of the whole softmax cost.

Implement and qualify D=64 first. It has material register headroom. The state changes from one persistent `lane_lse` scalar to at least `running_max` plus `running_sum`, approximately one additional persistent scalar before compiler scheduling. D=128 is already at 191 used/192 allocated VGPRs. Do not retain the D=128 variant if it exceeds 192 allocated VGPRs, creates any private/scratch memory, spills, or requires less favorable LDS/wave residency.

Correctness must be rerun because changing the recurrence can alter rounding and all-masked/initial-state behavior. Preserve the public tolerance gates and compare relative L2 and maximum normalized tolerance ratio against the Phase 9 baseline, not only AITER.

The D=64 experiment formed P directly as `exp2(score - new_max)`, retained linear `running_max/running_sum`, and normalized once in the epilogue. Focused HND/NHD aligned, independent-tail, short, long, arbitrary-head, and batch-two checks passed `16/16`. Exact active gfx1151 images for all 16 kernels remained spill-free with at most 168 used VGPRs for D=64, 16 KiB LDS, and zero private memory; D=128 remained at 191 used VGPRs and 32 KiB LDS.

The generated aligned D=64 ISA changed as intended: exponentials fell from 34 to 33, the logarithm disappeared, and normalization moved to the epilogue. However, HND static instructions rose from 1,175 to 1,186 and aligned HND used 155 VGPRs in the active image. A randomized 80-sample same-stream comparison of exact baseline/candidate code objects regressed `0.938%` geometrically on the six representative HND/NHD cases and won only one. The separate complete 18-row D=64 matrix was also negative, though its raw `1.98%` loss includes cross-run clock drift. An in-place epilogue lifetime refinement remained negative on all six focused cases and increased HND static instructions to 1,220.

The recurrence therefore failed the repeatable timing gate before profiler attribution or the complete public contract. D=128 was not attempted because the prerequisite D=64 benefit was absent and D=128 has no register margin. Production source returns to the Phase 10A log-domain recurrence. Artifacts are `~/tmp/feather_attn/phase10b_linear_*` and `phase10b_epilogue_*`.

#### Optimization 10C: D=64 Q Decode And Dependency Reduction

Status: accepted for D=64 HND only; rejected for NHD. D=64 has 136 allocated VGPRs in aligned NHD and 152 in aligned HND, zero measured LDS bank conflicts, and proportionally more fixed softmax/conversion work than D=128. It is the only dimension where selective Q-fragment caching or a longer-lived decoded fragment is currently reasonable.

Test one narrowly scoped change at a time:
- retain one decoded FP16 Q fragment across its four QK WMMAs while consuming and overwriting it promptly;
- increase independent work between each Q LDS load and first use;
- simplify/rematerialize Q and K LDS addresses to reduce dependent `ds_load_b128` issue;
- interleave Q decode permutations with independent K loads and WMMAs without adding another K/V buffer.

The purpose is to reduce dependency stalls and selected loads/permutations, not merely to increase nominal prefetch depth. Expected, unvalidated gain is `1-4%` for D=64. Reject the experiment if aligned HND exceeds 168 allocated VGPRs, any tail exceeds 192, bank conflicts appear, or the complete D=64 matrix regresses.

Do not generalize Q caching to D=128. Its 191-VGPR result makes an eight-register decoded fragment incompatible with the current resource contract.

The accepted loop already decodes each Q fragment once and retains it across the four same-fragment QK WMMAs. The bounded experiment additionally caches D64 `d_tile=0` across key-loop iterations. Applying it to both layouts exposed a schedule tradeoff: a randomized same-stream comparison of exact saved code objects improved HND but regressed the traffic-bound NHD H32 long cases. Production therefore enables the cache only when `D=64` and the compile-time layout is HND. NHD and all D=128 generated instruction counts remain unchanged.

The public contract passes `168/168`, and paired aligned outputs are bit-identical for all 18 D64 benchmark shapes. Exact active metadata for all 16 variants remains spill-free. D64 HND aligned/query-tail rise from 146 to 147 used VGPRs, combined-tail uses 162, and KV-tail rises from 168 to 172; all use 16 KiB LDS and zero private memory. NHD resources remain at Phase 10A values, and D=128 remains at 191 used VGPRs and 32 KiB LDS.

The first-iteration decode branch grows the static aligned HND body from 1,175 to 1,208 instructions, while dynamic work falls on later key tiles. At `H=16,N=4096`, isolated final-two-dispatch profiler medians report `VALUInsts 29,979 -> 28,719` (`-4.20%`) and `SQ_INSTS_LDS_sum 20,455,424 -> 20,197,376` (`-1.26%`). Normalized wait-count and LDS-issue exposure fall `26.6%` and `27.8%`; `ALUStalledByLDS` falls `28.0%`. Both variants retain zero LDS bank conflicts, 16 KiB LDS, zero scratch, approximately 152 allocated VGPRs for the aligned kernel, and effectively identical measured traffic.

The decisive 60-sample, randomized, same-stream exact-code-object comparison wins all nine HND rows by `1.07-1.64%`, with a `1.294%` geometric-mean gain. NHD, compiled through the original path, changes by `-0.135%` geometrically in the same control and is treated as unchanged. The complete public 36-case benchmark remains competitive with AITER: D64 HND is `1.037x-1.136x`, D64 NHD `0.978x-1.057x`, D128 HND `0.995x-1.136x`, and D128 NHD `0.973x-1.369x`. Cross-run absolute throughput was clock-shifted, so acceptance uses the paired baseline/candidate control rather than comparing raw values from separate full-matrix runs. Artifacts are under `~/tmp/feather_attn/phase10c_hnd_qcache*` and `phase10c_qcache_profiles/`.

#### Optimization 10D: Barrier And Waitcnt Scheduling

Status: D64 V-load hoist rejected; no production change. Evaluate this only after 10B and the D=64 portion of 10C, because both can change the dependency graph. Keep the single phase-reused K/V LDS buffer and the four correctness barriers. Do not remove a barrier unless a formal producer/consumer argument covers all eight waves and the next overwrite phase.

Permitted experiments are instruction scheduling changes:
- move independent pointer arithmetic before LDS waits;
- start address construction for the next phase while current WMMA work is independent;
- increase legal LDS load/use distance;
- replace overly conservative compiler waits only when ISA inspection and repeated correctness tests prove the narrower wait is valid.

CK Tile's async and transpose-load pipelines are scheduling references, not drop-in replacements. A second K/V buffer would exceed the D=128 32 KiB LDS budget and is outside this campaign. Expected gain is `0-3%` with low confidence.

The bounded experiment moved D64 V global loads before the existing K-to-V LDS barrier while leaving V LDS stores after it. D128 source ordering remained unchanged. Exact gfx1151 ISA confirmed that the two `global_load_b128` operations moved from immediately after the barrier into the independent softmax-reduction window; all four barriers, 50 waits, 1,175/1,335 HND/NHD D64 instructions, and the complete resource profile were unchanged. Focused HND/NHD correctness passed `16/16`, including all tail combinations and a D128 control.

A randomized 80-sample same-stream comparison of exact baseline/candidate code objects regressed all six representative cases and fell `0.349%` geometrically. The earlier separate-process focused run was more negative because of cross-run clock drift. The candidate therefore failed before the complete-matrix gate. Production source restores the Phase 10A load/barrier schedule. Artifacts are `~/tmp/feather_attn/phase10d_vload_*` and `phase10_paired_robust.*`.

#### Deprioritized Phase 10 Work

- Deep K/V prefetch is secondary because SIMD utilization is already high and LDS queue-full pressure is negligible.
- Split-K/split-sequence remains conditional on low-grid, long-K workloads. The profiled long grid already has 4,096 CTAs and is traffic-bound.
- Additional block shapes and a broad autotuner remain outside scope.
- FP8 K, V, or P remain unjustified because they add conversion work without removing the persistent state that motivated Q8.
- D=128 Q caching is outside the resource envelope.
- Targeted AITER/NHD PC sampling is optional and should be run only if a candidate changes attribution that existing traffic, SQ, and occupancy counters cannot explain.

#### Phase 10 Execution Order And Gates

- Done: bounded D=128 NHD grouping passed traffic, correctness, resource, and complete-matrix gates. D=64 grouping was measured and rejected.
- Rejected: persistent linear `(m,l)` for D=64 passed focused correctness/resources and transformed the ISA, but regressed the complete D=64 matrix. D=128 was not attempted.
- Done: one D=64 decoded-Q fragment cached across key tiles passes for HND. The NHD specialization retains the Phase 10A schedule.
- Rejected: moving D64 V global loads before the K-to-V LDS barrier preserved correctness/resources and changed ISA scheduling, but regressed the focused timing set.
- Stop: Phase 10A and the HND-only Phase 10C cache are the accepted production changes. The remaining modeled on-chip candidates failed their independent timing gates.

Phase 10A plus the HND-only Phase 10C cache form the completed production baseline. Rejected candidates remain documented with their independent attribution.

### Phase 11: Separately Justified Follow-Up Campaign

Status: complete. Phase 10 reached its documented stop condition, and Phase 11 opened a separate campaign because the qualified production images exposed two distinct remaining bottleneck regimes with separately testable causes. Optimization 11B and 11C were accepted; every other generated-kernel experiment was rejected and restored. Modeled speedups below remain historical planning bounds unless followed by measured results.

#### Production Baseline And Bottleneck Split

The campaign's initial baseline was commit `31faf96`, using exact active gfx1151 images extracted from rebuilt `.cuda.o` fatbins. The cumulative final baseline is commit `01454e3`, where 11C was implemented and qualified on top of accepted 11B. The historical Phase 10C wrapper matrix is `~/tmp/feather_attn/phase10c_hnd_qcache_full_matrix.csv`; candidate acceptance used saved baseline and candidate code objects on one stream rather than cross-run wrapper timing.

The execution freeze is under `~/tmp/feather_attn/phase11_baseline_freeze/`. The active images extracted from the current `.cuda.o` `.hip_fatbin` sections match all four qualified Phase 10C images byte-for-byte. `Kargs` remains 56 bytes.

| Regime | Production evidence | Current conclusion |
| --- | --- | --- |
| D=64 HND | `32.007-34.576 TFLOPS`; the accepted first decoded-Q cache adds `1.294%` geometrically with 9/9 wins | Primarily on-chip instruction and dependency limited |
| D=128 HND | `31.992-33.781 TFLOPS`; 191 used/192 allocated VGPRs and 32 KiB LDS | On-chip limited, but structurally pinned at both resource gates |
| Long D=64 NHD | `H=32,N=16384` is `22.399 TFLOPS` versus `32.478 TFLOPS` for HND; the unchanged NHD profile fetched `15.837 GiB` at `174.64 GB/s` | Primarily controller-traffic and cross-CTA-reuse limited |
| Grouped D=128 NHD | `H=32,N=16384` is `27.248 TFLOPS` versus `33.781 TFLOPS` for HND; grouping reduced fetch `31.435 -> 14.822 GiB` | Existing grouping captures a real LLC opportunity, but residual cache and on-chip limits remain |

The on-chip diagnosis is specific:
- the active gfx1151 images already contain extensive `v_dual_*` instructions, so generic VOPD enablement is not an optimization;
- `ALUStalledByLDS` remains only `0.024-0.043%`, LDS issue waits remain below `0.08%`, and D=64 has zero measured bank conflicts;
- the dominant opportunity is longer distance between LDS issue and first use, plus fewer dependent address, conversion, and lane-broadcast chains;
- Feather frequently consumes each two-load QK or PV batch after `s_waitcnt lgkmcnt(0)`, while the active AITER control begins with eight outstanding LDS loads and consumes them with descending `lgkmcnt(6/4/2/0)` waits.

The traffic diagnosis is also specific:
- long D=64 NHD measured traffic is effectively the no-cross-CTA K/V stream implied by the `128x64` schedule;
- the theoretical no-reuse intensity is about `128 FLOP/byte`, and measured traffic intensity is about `129 FLOP/byte`;
- contiguous D=64 head groups are not a fallback: at `H=32,N=16384`, group sizes `{4,8,16}` regressed by `72.0%`, `54.9%`, and `24.8%`;
- any new D=64 grouping experiment must preserve head diversity within each LLC-bounded sequential launch and prove its effect with timing and traffic counters rather than infer a partition mechanism from undocumented per-instance counters.

#### Phase 11 Numerical And Resource Policy

The public operation, layout contract, tail variants, and signed-int32 device-indexing proof remain unchanged. Internal rounding and temporary arithmetic precision may change when that creates a demonstrably shorter or less resource-intensive active instruction sequence. Permission to change internal arithmetic does not permit an undocumented or shape-specific correctness exception.

Phase 11 must preserve these arithmetic boundaries:
- QK score accumulation remains FP32;
- the persistent normalized PV output and every cross-key-tile output update remain FP32;
- do not use `v_wmma_f16_16x16x16_f16` for QK or feed the persistent output through it; Optimization 11H alone may use that instruction for a zero-initialized `Bc=64` PV partial that is widened after exactly four `K=16` updates;
- K, P, and V remain FP16 WMMA operands;
- an RTZ packed conversion, FP16 temporary exponential, or other approximation is acceptable only if the complete operation passes the Phase 11 correctness policy and realistic activation checks;
- do not replace a one-instruction hardware exponential with a polynomial unless the extracted active ISA proves a lower total instruction and dependency cost.

Use the current public thresholds first: `rtol=atol=0.10` for `N_KV < 1024` and `0.05` otherwise. An arithmetic candidate that fails only the long-sequence `0.05` threshold may request a uniform Phase 11 relaxation in this order:
- retry all `N_KV >= 1024` cases at `rtol=atol=0.075`;
- if still justified by a material timing or resource win, retry at an absolute Phase 11 cap of `rtol=atol=0.10`;
- never exceed `0.10`, create per-shape tolerances, or loosen only the failing tensor elements.

A relaxed candidate is not accepted merely because every element falls inside the new envelope. It must also show finite outputs, no NaN/Inf regressions, bounded maximum normalized error, no material deterioration of relative L2 or high-percentile absolute error versus the production kernel, and realistic activation/model-quality checks. Record both the production-threshold failure count and relaxed-threshold result. If a relaxed candidate is retained, update the public test threshold uniformly and document the arithmetic change and measured speedup in the ledger.

The existing resource gates remain hard: at most 192 allocated VGPRs, at most 32,768 bytes LDS, zero private/scratch memory, zero spills, and no material new bank conflict. D=128 starts at 191 used VGPRs and 32 KiB LDS, so a D=128 candidate must first remove resources before adding persistent state or buffering. D=64 is the default feasibility target.

#### Phase 11 Common Qualification Gates

Apply these gates to every candidate before it can become cumulative:
- Exact active image: rebuild all affected specializations, extract `.hip_fatbin`, unbundle the gfx1151 image, and inspect that image rather than stale `.cuda.o.0.hipv4-*` sidecars. If Kargs changes, rederive and validate its by-value ABI size.
- Correctness: run a focused aligned/tail/layout/batch-two screen first, then the complete `168/168` public contract for any candidate that changes generated kernel behavior. Arithmetic candidates must report production-threshold and any approved relaxed-threshold results plus error-distribution comparisons.
- Resources: inspect every affected aligned/query-tail/KV-tail/combined-tail image. Reject any allocated-VGPR or LDS gate violation, private segment, scratch, spill, or sequence-dependent workgroup resource.
- Attribution: require the intended ISA transformation and collect only counters that test the hypothesis. Fewer source operations without the expected active-image change do not qualify.
- Close timing: alternate exact baseline and candidate code objects on one stream and the same tensors. A local schedule change should provide at least a `0.5%` target-domain geometric-mean win, win a majority of the paired cases, and show no repeatable regression larger than `1%` before the wrapper matrix.
- Structural timing: an additional block shape or launch policy must provide at least a `5%` geometric-mean win over the production dispatch on the domain where it would be selected. It must not alter non-target dispatches.
- Traffic candidates: sum byte counters across constituent launches, keep `FETCH_SIZE`, GCEA read size, and request-based `L2CacheHit` separate, and require both lower controller traffic and lower wall time. Do not aggregate per-sublaunch `SIMD_UTILIZATION`.
- Final matrix: rerun the authoritative 36-case wrapper matrix after every candidate promoted past focused timing. Do not combine two unaccepted candidates; independently qualify each against the same production baseline first.

#### Optimization 11A: Progressive LDS Issue And Consumption

Status: rejected and restored to production. Start with D=64 aligned HND and NHD as separate code-object candidates. Keep the current LDS layouts, rotating V mapping, single K/V buffer, and all four phase barriers.

For QK, issue the four independent `k_lo/k_hi` pairs for one `d_tile` before the first WMMA, then consume the eight outstanding LDS operations with descending waits such as `lgkmcnt(6/4/2/0)`. For PV, independently test the same schedule across the four D=64 output fragments. Do not combine QK and PV rescheduling until each has its own timing result.

The source-level loop order is not the acceptance condition. The active image must show:
- a larger batch of independent `ds_load_b128` instructions before first use;
- nonzero descending `lgkmcnt` thresholds before the final drain;
- unchanged WMMA count, barrier count, and global-memory traffic;
- no increase in bank conflicts or LDS queue-full attribution.

The expected gain is low single digit and unvalidated. The primary profiler targets are normalized wait-count exposure, ALU dependency, LDS issue exposure, `VALUInsts`, and `SQ_INSTS_LDS_sum`. If four load pairs cross the resource gate in a tail variant, retry a two-pair schedule before rejecting the whole idea.

The first QK candidate issued all four K pairs before consumption. The extracted D=64 active image showed the intended eight-read batch with descending waits, normally `lgkmcnt(6/4/2/0)`, while D=128 remained byte-identical to production. It was spill-free with zero private memory; D=64 HND aligned/query/KV/combined-tail used `{145,145,168,168}` VGPRs and NHD used at most 149. Focused aligned outputs were bitwise identical, but the 60-sample exact-code-object comparison regressed HND by `2.263%` geometrically and NHD by `1.149%`, with 1/6 wins overall. The two-pair fallback was also spill-free and used at most 169 VGPRs. It won 5/6 focused cases but improved only `0.312%` geometrically for HND and `0.239%` for NHD, below the `0.5%` gate, and regressed long H32/N16384 NHD by `0.85%`. Reject both QK schedules and restore the production QK loop before testing PV independently.

The independent four-fragment PV candidate also emitted the intended eight-read batch and `lgkmcnt(6/4/2/0)` consumption. D=64 HND aligned/query/KV/combined-tail used `{172,172,177,176}` VGPRs and NHD used 152; all images had zero private memory and spills, and D=128 remained byte-identical. Focused outputs were bitwise identical. Exact-code-object timing regressed HND by `3.165%` geometrically with 0/3 wins; NHD was mixed at `+0.371%`, with only 1/3 wins and two regressions. Reject PV progressive issue without a two-pair retry because the four-pair form passed resources but failed timing materially. Restore the complete production QK/PV loop before 11B.

#### Optimization 11B: Additional D=64 HND Decoded-Q Caching

Status: accepted; HND only. The first persistent D=64 HND decoded-Q fragment already reduced dynamic VALU and LDS dependency work and won all nine HND benchmark rows. The accepted extension caches all four fragments for aligned, query-tail, and combined-tail kernels and caps the KV-tail kernel at three fragments. NHD retains its production decode path.

Each decoded `Wmma::AVecType` is logically eight VGPRs. First-order used-VGPR estimates, before compiler scheduling, are:
- aligned and query-tail: cache all four fragments, approximately `147 + 24 = 171` used VGPRs;
- combined-tail: cache all four fragments, approximately `162 + 24 = 186`;
- KV-tail: cap the experiment at three total cached fragments, approximately `172 + 16 = 188`.

These are feasibility estimates, not resource claims. Exact extracted metadata decides acceptance. Keep NHD on its current decode path because the all-layout Phase 10C candidate regressed long H32 NHD cases. Qualify 11B independently from 11A because both consume the same D=64 register headroom; combine them only if each wins alone and the combined image remains within 192 allocated VGPRs.

The extracted D=64 HND aligned/query/KV/combined-tail images use `{175,175,171,171}` VGPRs, 16 KiB LDS, zero private memory, and zero spills. D=64 NHD remains at `{130,130,151,151}` VGPRs, and byte-level ELF symbol comparison confirms that every D=64 NHD and D=128 kernel body is identical to the frozen Phase 10C image. `Kargs` remains 56 bytes. The candidate passes the complete public contract at `168/168`; exact-code-object checks are bitwise identical.

The first 60-sample focused bracket improved HND by `0.918%` geometrically with 3/3 wins. The first nine-shape bracket was noisy at `+0.523%` with 8/9 wins, so acceptance used an independent 100-sample bracket: all nine HND shapes won by `0.60-0.92%`, for a `+0.789%` geometric mean. On the isolated H16/N4096 attribution dispatch, `VALUInsts` fell `28,719 -> 27,745` (`-3.39%`), `SQ_INSTS_VALU_sum` fell by the same proportion, and `SQ_INSTS_LDS_sum` fell `20,197,376 -> 19,423,232` (`-3.83%`), while `LDSBankConflict` remained zero. Accept the HND-only cache extension and use its exact rebuilt images as the cumulative baseline for 11C and later experiments.

#### Optimization 11C: Strided LLC-Bounded D=64 NHD Head Groups

Status: accepted with a partition-aware activation policy. Retain sequential launches and the 32 MiB hard working-set cap, but replace contiguous physical head subsets with a bijective strided permutation that spreads each launch across the complete head range.

One candidate mapping is:

```text
group_size  = floor(32 MiB / per_head_KV_bytes)
group_count = ceil(H / group_size)
physical_head(group, local) = group + local * group_count
```

Only physical heads below `H` participate, so partial final groups remain valid and no head is duplicated or omitted. Physical NHD tensor strides continue to use the full head count. The device mapping therefore needs an explicit group stride or equivalent checked argument; do not reinterpret `head_start` as contiguous if the mapping is strided.

Start with `H=32,N=16384,D=64`, where an eight-head group occupies exactly 32 MiB and the production gap is largest. Then test `H=32,N=8192` and the H=16/H=56 long cases. Keep D=128 on the accepted Phase 10A grouping policy unless a separate D=128 schedule A/B test justifies changing it.

Reject 11C unless the same candidate simultaneously:
- reduces summed `FETCH_SIZE` and GCEA read size;
- improves exact same-stream whole-call timing;
- preserves enough launch-local head diversity to avoid the severe contiguous-group regressions;
- handles arbitrary positive heads, partial groups, both tails, and batch two with checked host arithmetic.

The broad LLC-only policy was directionally positive but failed the structural gate: its robust six-shape geometric mean was `+3.500%`, below `5%`. Schedule sweeps exposed a stronger and repeatable partition rule. Minimum LLC-safe physical-head strides generally won, but strides divisible by eight recreated severe aliasing: H64/N16384 with stride eight regressed `28.90%`, H128/N8192 with stride eight regressed `64.11%`, and H128/N16384 with stride 16 regressed `80.07%`. Incrementing those strides to 9 or 17 changed the same controls to gains of `32.79%`, `29.88%`, and `19.46%` respectively.

The accepted selector therefore activates only for D=64 NHD when all of these hold:
- the physical head count is divisible by 16;
- the existing `1.5 * LLC` activation threshold and 32 MiB per-launch cap hold;
- the cap permits at least four heads per launch and requires at least three launches;
- the minimum LLC-safe `group_count = ceil(H / group_size)` is incremented by one when divisible by eight.

The accepted mapping is `physical_head = group_index + local_head * group_count`; exact partial-group sizes are computed on the host, and physical tensor strides continue to use the complete head count. It uses a separate strided kernel entry with a 64-byte by-value `Kargs`, while every existing 56-byte HND/NHD D64/D128 production image remains whole-file byte-identical to 11B. The four strided aligned/query/KV/combined-tail variants use `{130,130,151,151}` VGPRs, 16 KiB LDS, zero private memory, and zero spills.

Selected-path aligned/query/KV/combined-tail and batch-two outputs are bitwise identical, including partial plans `[11,11,10]` and `[7,7,7,7,6,6,6,6,6,6]`; the final public contract passes `168/168`. Exact whole-call timing, including every sequential sublaunch, wins all 12 selected-domain controls across H32/H48/H64/H96/H128 by `4.08-39.57%`, for a `+21.280%` geometric mean. At H32/N16384, summed `FETCH_SIZE` falls `15.830 -> 13.554 GiB` (`-14.38%`), summed GCEA reads fall `15.858 -> 13.460 GiB` (`-15.12%`), and request-based `L2CacheHit` rises `2.08% -> 12.93%`. The corrected 36-case matrix reaches `28.879 TFLOPS` at D64/NHD H32/N16384; all non-selected rows retain their established dispatches and performance ranges.

#### Optimization 11D: Replace Scalar Alpha Fan-Out

Status: rejected and restored to production. The current online update extracts row-specific `alpha` through 16 compile-time `v_readlane_b32` operations per key tile, followed by scalar-to-vector moves and selects before output rescaling. Build a D=64 fixture that forms the same row-alpha vectors with fixed lane permutations, DPP, or another vector-only mapping.

Use the existing arithmetic first. The active image must materially reduce readlane and scalar/vector dependency work without adding a barrier, LDS traffic, or persistent VGPR state. Promote to D=128 only if the D=64 fixture wins and the D=128 image remains at or below 192 allocated VGPRs. Expected gain is `0-2%`, unvalidated.

The D=64 fixture used one adjacent-lane DPP swap and one lane-group select, then let the compiler fold eight row-local broadcasts directly into 32 `v_mul_f32_dpp ... row_share:*` output scales. It eliminated all 16 `v_readlane_b32` instructions, reduced aligned HND/NHD static instruction counts by 15/31, and lowered aligned NHD from 130 to 127 VGPRs. All D=64 variants remained spill-free with zero private memory, and every D=128 body stayed byte-identical. Outputs were bitwise identical.

Exact-code-object timing nevertheless rejected the schedule. The first 60-sample bracket regressed HND by `0.313%` geometrically and NHD by `0.825%`, with 0/6 wins. An independent 100-sample bracket confirmed `-0.743%` HND and `-0.602%` NHD, with 1/6 wins and long NHD regressions of `1.04-1.26%`. The cleaner static sequence creates slower row-share output dependencies on gfx1151; restore the scalar readlane fan-out.

#### Optimization 11E: Relaxed Temporary P Arithmetic

Status: completed; all isolated probes rejected and production arithmetic restored. Arithmetic relaxation is useful only when it removes active instructions, shortens a dependency chain, or frees registers for another accepted schedule. Test D=64 first and isolate each change.

Permitted probes include:
- `v_cvt_pk_rtz_f16_f32` for adjacent P values when the complete beta-scale-plus-conversion sequence is shorter than the current `v_fma_mixlo/hi_f16` pair;
- converting beta to FP16 and using packed FP16 multiplication only if the active sequence is no longer than the current fused conversion and has a shorter dependency chain;
- `v_exp_f16` for temporary shifted logits only as a register-lifetime enabler, while row max, row sum, reciprocal, LSE state, and output accumulation remain FP32;
- alternate Q E5M2 encoding rounding when it removes instructions or enables a better decode/cache schedule.

The static ISA gate comes before full timing. Reject a packed-conversion candidate if separate scaling restores the same or greater instruction count. Reject an FP16 exponential candidate if conversions back to FP32 for the row sum erase the liveness or instruction benefit. Do not combine relaxed conversion, relaxed exponential, or a different Q8 rounding mode in one first experiment. Apply the bounded tolerance escalation only after a candidate passes ISA, resource, and timing gates at the smallest tolerance that works.

The first isolated probe forced D=64 beta scaling followed by adjacent `__builtin_amdgcn_cvt_pkrtz` conversion, while D=128 retained the production path. It failed the static gate. Aligned HND/NHD instruction counts increased `1250 -> 1258` and `1335 -> 1354`, and VALU counts increased `907 -> 912` and `942 -> 960`; the strided NHD body likewise increased `1351 -> 1370` instructions and `945 -> 963` VALU instructions. The candidate replaced 32 fused scale-and-convert mixed-FMA operations with 16 packed conversions plus 26-36 additional FP32 multiplies. Restore it without correctness or timing; no tolerance relaxation is justified.

The second probe attempted an RNE packed-FP16 conversion followed by `v_pk_mul_f16` with beta converted to FP16. LLVM rejects `v_cvt_pk_f16_f32` for gfx1151 as an unsupported instruction. The available packed RTZ conversion is the independently rejected 11E1 path; retaining RNE requires two scalar mixed conversions per pair and then adds one packed multiply, which is strictly longer than the production two fused beta-scale-and-convert mixed-FMAs. Reject this form at compile/static feasibility without combining it with RTZ arithmetic.

The third probe routed only the 32 temporary score exponentials through the OCML FP16 entry, immediately widened each result for the mandatory FP32 row sum, and retained FP32 state exponentials and accumulation. LLVM did not select `v_exp_f16` on gfx1151. The active D=64 bodies retained all 34 `v_exp_f32` instructions and added 64 FP32-to-FP16 conversions. Aligned HND/NHD grew `1250 -> 1355` and `1335 -> 1437` instructions, and VALU grew `907 -> 1009` and `942 -> 1041`; aligned HND also rose to 178 VGPRs. Reject before correctness or timing because the conversion path erases both instruction and liveness benefit.

The fourth probe changed only D=64 Q encoding from tie-even RNE to the previously qualified half-up form `bits + 0x80`, retaining the exact Q8 LDS layout, decoder, and all FP32 accumulation. It passed the static and arithmetic gates: D=64 aligned HND/NHD instruction counts fell `1250 -> 1152` and `1335 -> 1239`, VALU fell equally, all D=128 bodies remained byte-identical, resources stayed within 175 HND and 162 NHD VGPRs with zero private memory or spills, and the public contract passed `168/168` at the original thresholds. Exact-code-object timing did not retain it. The first bracket was neutral at `-0.019%` HND and `+0.011%` NHD; the independent 100-sample bracket was `-0.461%` HND and `+0.109%` NHD, for `-0.176%` overall. The removed encoder work runs once per query tile, not once per key tile, and misses the `0.5%` gate. Restore tie-even RNE without requesting tolerance relaxation.

#### Optimization 11F: Rejected D=64 NHD `Br=256`

Status: rejected at the aligned structural timing gate. Attempt this only if 11C fails or leaves the long D=64 NHD path clearly traffic limited. This is not approval for a generalized block-size policy.

A Q8 `Br=256,Bc=64,D=64` specialization would use 16 KiB persistent Q LDS plus 8 KiB phase-reused K/V LDS, or 24 KiB total. Sixteen wave32 waves would each retain ownership of 16 query rows while sharing one K/V tile, halving schedule-implied K/V reads per query row and raising the modeled no-reuse intensity from about `128` to `256 FLOP/byte`.

The risks are substantial:
- a 16-wave workgroup can provide only 16 resident waves per WGP where the current 8-wave workgroup can reach the qualified 24-wave point;
- barrier scope doubles;
- the current V staging uses an eight-wave ownership map and must be redesigned rather than extended with `wave * 8`;
- query-tail behavior and launch granularity change;
- the prior generic Triton `M256N64W16` control was rejected, although its 64 KiB FP16-Q shared-memory footprint is not resource-equivalent to this 24 KiB Q8 proposal.

Compile and test only aligned `H=32/56,N=16384,D=64,NHD` first. Stop before tail implementation unless measured fetch falls by at least `25%`, the target-domain geometric mean improves at least `5%`, and the image remains below 192 allocated VGPRs with zero private memory and spills.

The standalone aligned fixture used a 512-thread, 16-wave workgroup, 16 KiB Q8 LDS plus 8 KiB phase-reused K/V LDS, the accepted 64-byte strided ABI, and only the first eight waves for the rotating V transpose. It compiled to 131 VGPRs, 24 KiB LDS, zero private memory, and zero spills. The active body retained 32 WMMAs and four barriers per key tile and was slightly shorter than the `Br=128` strided body. Outputs were bitwise identical on the focused controls.

Whole-call timing rejected it before traffic profiling or tails. H32/N16384, including four accepted partition-aware launches on both sides, improved only `0.90%`; H56/N16384 regressed `4.60%`. The two-shape geometric mean was `-1.889%` with 1/2 wins, far below the structural `5%` gate. Do not add a production `Br=256` entry.

#### Optimization 11G: Rejected D=64 `Bc=128`

Status: rejected at the compile-time resource gate. A D=64 `Br=128,Bc=128` specialization would use 8 KiB Q LDS plus 16 KiB K/V LDS, or 24 KiB total, while retaining the 8-wave workgroup. It does not reduce K/V controller bytes, but it halves key-loop barrier, reciprocal, logarithm, and state-update counts.

The score fragments double from four to eight, so live score pressure is the primary gate. Test aligned HND first, where controller traffic is already low and fixed on-chip overhead matters most. Do not attempt D=128: its Q plus K/V LDS requirement would be 48 KiB and violate the accepted 32 KiB resource contract. As an additional specialization, 11G must meet the structural `5%` timing gate before tail and dispatch work.

The standalone aligned HND fixture retained the accepted four-fragment decoded-Q cache and instantiated eight FP32 score fragments, eight PV updates, the 8-wave workgroup, and 24 KiB LDS. The extracted gfx1151 image required 242 VGPRs. Although private memory and spills remained zero, this exceeds the hard 192-VGPR gate by 50. Reject without correctness, timing, tails, or dispatch work.

#### Optimization 11H: SageAttention-Style Tile-Local FP16 PV Buffer

Status: rejected and restored to production. SageAttention's `pv_accum_dtype="fp16+fp32"` path is materially different from carrying the attention output in FP16. `compute_fp16_sv_permuted_inst_buf` keeps `RO` as a persistent FP32 fragment, initializes a temporary FP16 instruction buffer for the first `K=16` PV MMA, performs the remaining inner MMAs into that buffer, then widens its eight results and adds them to `RO`. The buffer is flushed once per outer key tile rather than carried across the sequence.

The corresponding FeatherAttn recurrence is:

```text
O32 = alpha * O32
T16 = WMMA_F16(P_beta[0], V[0], 0)
T16 = WMMA_F16(P_beta[1], V[1], T16)
T16 = WMMA_F16(P_beta[2], V[2], T16)
T16 = WMMA_F16(P_beta[3], V[3], T16)
O32 = O32 + widen(T16)
```

Here each `P_beta` is the existing beta-scaled FP16 probability fragment, the four updates cover only the current `Bc=64` tile, and no FP16 output partial survives the key-loop iteration. Before FP16 conversion, the beta-scaled tile probabilities are nonnegative and have row mass at most one, so the exact tile contribution is a bounded weighted-V partial rather than an unnormalized 64-term sum. Rounding each probability and each FP16 FMA can weaken that bound, including overflow near the FP16 limit, so the candidate must pass the complete arithmetic-candidate policy. SageAttention's use is evidence for the decomposition, not evidence that FeatherAttn meets its error or model-quality gates.

gfx1151 supports `v_wmma_f16_16x16x16_f16` with round-to-nearest-even, but its register economics differ from NVIDIA's. In wave32, both the FP16-output and FP32-output `16x16x16` WMMA C/D fragments occupy eight 32-bit VGPRs per lane. With `op_sel=0`, the eight meaningful FP16 results are the low halves at logical `half16` indices `{0,2,...,14}`; widen only those values into the corresponding eight FP32 C slots. The unused halves are not additional results. This means 11H is not a register-compression optimization on gfx1151.

Use one eight-VGPR FP16 scratch fragment and serialize output-fragment ownership. Convert the four score fragments to packed P in storage whose FP32 score lifetime has ended; do not retain both score and P arrays. The measured D=64 HND baselines for aligned, query-tail, KV-tail, and combined-tail are `{147,147,172,162}` used VGPRs. Adding only the logical scratch gives first-order estimates `{155,155,180,170}`, but compiler scheduling can change other live ranges, so exact metadata for all four variants decides feasibility. The corresponding NHD baselines are no higher than 151 VGPRs. A scratch fragment per D tile would add 32 logical VGPRs and is rejected without a compile. D=128 is excluded because its production image already uses 191 VGPRs.

Apply a static gate before full timing:
- D=64 QK must retain its 16 `v_wmma_f32_16x16x16_f16` instructions per key tile, while only the 16 PV instructions change to `v_wmma_f16_16x16x16_f16`;
- the active image must show one tile-local FP16 scratch lifetime, FP32 alpha scaling, and an immediate FP32 widening-add merge, with no new barrier, LDS operation, or global-memory transaction;
- the merge should require at most one mixed widening-add instruction per output value; reject separate conversion/add expansion unless a focused dependency fixture demonstrates a compensating schedule benefit;
- inspect all D=64 aligned and tail images and reject retained score/P duplication, more than 192 allocated VGPRs, private memory, scratch, or spills;
- run focused zero, random, alternating-sign, high-dynamic-range V, long-sequence, both-layout, and tail checks at the production thresholds before considering the bounded relaxation policy.

Public sources do not establish a gfx1151 throughput advantage for FP16-output over FP32-output WMMA, and 11H adds 32 FP32 merges per D=64 key tile. Its only plausible wins are a shorter PV accumulator dependency or a better compiler schedule. Start with HND because that is the dependency-limited regime; leave NHD on the FP32 path unless a separate NHD comparison passes the same gates. Qualify 11H independently from 11A and 11B, retain it only after the normal `0.5%` exact-code-object timing gate, and do not request a tolerance relaxation unless that material timing or resource benefit has already been demonstrated.

The extracted D=64 HND fixture exactly matched the bounded design: 16 FP32 QK WMMAs remained, 16 PV operations changed to FP16-output WMMAs, and 32 `v_fma_mix_f32` widening-add merges immediately folded the tile scratch into persistent FP32 output. Barrier, LDS, and global-memory instruction counts were unchanged. D=64 NHD, the strided entry, and every D=128 body were byte-identical. Aligned/query/combined-tail HND allocated 192 VGPRs and KV-tail allocated 189, all with 16 KiB LDS and zero private memory or spills.

The candidate passed the complete `168/168` contract at production thresholds and a direct adversarial screen covering zero, random, alternating-sign, 256x dynamic-range, near-FP16-limit, long-sequence, tail, both-layout, and batch-two controls. All outputs were finite; worst candidate/production relative-L2 ratio was `1.0002`, so no tolerance relaxation was needed. Performance nevertheless rejected it decisively: exact-code-object HND timing regressed `10.92%`, `7.70%`, and `7.55%` on the focused controls, for `-8.739%` geometrically with 0/3 wins. The FP16 WMMA dependency does not compensate for 32 merges and 192-VGPR pressure on gfx1151. Restore the FP32 PV path.

#### Explicitly Deprioritized Or Excluded In Phase 11

- Pure FP16 WMMA score accumulation or persistent output accumulation is excluded because of known accuracy risk. Optimization 11H is the sole exception: it rounds only one tile-local PV partial before an immediate FP32 merge.
- Generic VOPD tuning is deprioritized because the compiler already emits extensive `v_dual_*` instructions.
- Persistent linear `(m,l)`, the D=64 V-load hoist, and contiguous D=64 grouping remain rejected by their Phase 10 timing results.
- Full K/V double buffering is not a general option; D=128 already consumes the complete 32 KiB LDS budget.
- D=128 decoded-Q caching remains outside the 191/192-VGPR envelope.
- CK async/TDM pipelines remain scheduling references rather than drop-in gfx1151 implementations.
- Native BF8 conversion remains unproven on gfx1151; keep byte-permute E5M2 expansion unless an isolated compiler/ISA probe demonstrates a supported instruction and end-to-end win.
- Broad block-shape sweeps, a new autotuner, FP8 K/V/P transport, and polynomial transcendental replacements remain outside scope.

#### Phase 11 Execution Order

- Freeze exact Phase 10C production code objects, loaded-image hashes, the 36-case matrix, and focused profiler controls.
- Run 11A QK progressive LDS consumption, then its PV counterpart, as independent D=64 experiments.
- Run 11B additional HND Q caching independently against production.
- Run 11C strided D=64 NHD grouping with traffic attribution.
- Run 11D alpha fan-out only after the leading dependency and traffic candidates are resolved.
- Run the isolated 11E arithmetic probes only when their static ISA can beat the current instruction or liveness shape. Escalate tolerance only after the candidate proves a material performance or resource benefit.
- Run the 11H tile-local mixed-PV fixture only after its active ISA proves the exact four-update FP16 buffer and immediate FP32 merge. Qualify it independently from 11A/11B.
- Attempt 11F only if long D=64 NHD remains traffic limited after 11C.
- Attempt 11G only if D=64 HND remains on-chip limited and the aligned resource model fits after earlier accepted work.
- Combine only independently accepted candidates, then rerun resources, correctness at the accepted uniform tolerance, profiler attribution, exact-code-object paired timing, and the authoritative 36-case matrix.

The sequence is complete. No separate combination patch was required: 11C was developed and qualified on the accepted 11B baseline, so commit `01454e3` already contains the complete accepted Phase 11 source state.

#### Phase 11 Final Qualification

A forced rebuild removed every generated `.cuda.o` before compiling, then extracted the five active `.hip_fatbin` gfx1151 images with the explicit `hipv4-amdgcn-amd-amdhsa--gfx1151` target. Every rebuilt image matches the frozen accepted 11C image byte-for-byte.

The rebuilt extension passes the complete public contract at `168/168` with the unchanged production thresholds. Authoritative metadata covers 20 kernels: the 16 existing aligned/tail HND/NHD D64/D128 entries retain 56-byte `Kargs`, the four isolated strided D64 NHD entries use 64-byte `Kargs`, maximum used VGPRs are 191 and therefore allocate 192, maximum LDS is 32,768 bytes, and private memory and SGPR/VGPR spills are zero.

Because these images are byte-identical to the accepted 11C freeze, the cumulative profiler attribution and exact-code-object timing are the accepted 11B and 11C measurements: HND dynamic VALU fell `3.39%` from 11B, while selected D64 NHD grouping improved 12/12 controls by `21.280%` geometrically and reduced H32/N16384 `FETCH_SIZE` by `14.38%` and GCEA reads by `15.12%`. The final wrapper matrix completes all 36 cases. Relative to AITER, Feather's geometric-mean ratios are `1.101x` for D64 HND, `1.087x` for D128 HND, `1.048x` for D64 NHD, and `1.156x` for D128 NHD; the overall ratio is `1.097x` with 31/36 wins. The selected D64 NHD H32/N16384 row reaches `29.088 TFLOPS`. A separate-run comparison with the corrected 11C matrix is `+0.917%` geometrically and is used only as a stability check, not an acceptance measurement.

Final artifacts are under `~/tmp/feather_attn/phase11_final/` and `~/tmp/feather_attn/phase11_final_images/`. Phase 11 stops at `01454e3`; only this documentation ledger remains as an uncommitted tracked change.

## Stop Rules

Stop or redesign before further optimization if any of these remain true:
- used VGPRs exceed 192 after the explicit liveness fixes;
- LDS exceeds 32 KiB;
- any private segment or scratch remains;
- the exact Q layout has material bank conflicts;
- a supported shape can overflow a narrowed device offset, counter, or launch dimension;
- realistic activation tests fail the accepted uniform Phase 11 tolerance or model-quality check;
- `S=4096` remains more than 5% slower than AITER after basic scheduling;
- profiling shows Q LDS traffic or `v_perm_b32` issue consumes the complete occupancy gain.

## Deferred Work

The accepted fixed kernel does not include:
- production additional `Br`, `Bc`, or wave-count variants; Phase 11F/11G permit only isolated D=64 feasibility probes until their structural gates pass;
- a production `Br=256` or `Bc=128` dispatch entry;
- FP16 WMMA score accumulators or persistent FP16 output accumulators; 11H permits only a non-production tile-local mixed-PV probe;
- FP8 K or V transport and cooperative expansion;
- FP8 P;
- causal attention, arbitrary masks, dropout, ALiBi, or softcap;
- `D=256`;
- a new autotuner.

If another block size or head dimension is later justified by measured shape coverage, add it as a separate specialization with its own resource proof.

## Experiment Ledger

Add one row per isolated experiment. Include links or paths to profile artifacts when available.

| ID | Status | Change | Correctness | VGPR / LDS / private | `S=4096` | `S=8192` | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| B0 | Done | Stable-extension Linux/Windows compiler compatibility | Extension import and direct legacy launches pass | Unchanged | N/A | N/A | Keep cross-platform include order and builtin math |
| CK0 | Done | Stock CK Tile `128x64x8` FP16 compile control and aligned ablation | Compile-only | `pssk`: 240 / 9,216 / 136; `npad`: 215 / 9,216 / 0 | N/A | N/A | Reject stock pipeline; retain CK core |
| F0 | Done | Freeze six-shape AITER and legacy controls | N/A | AITER: 233 / 32,768 / 0 | See Phase 0 table | See Phase 0 table | Re-run beside final candidate |
| F1 | Done | Q8 log2-pre-scaled LDS fixture, RNE and half-up | Bit-exact fixture reference | 68 / 16,384 / 0 | N/A | N/A | Use RNE; retain half-up ablation |
| F2 | Done | Row-owned QK skeleton with XOR-swizzled K LDS | Quantized-Q QK max abs `9.54e-7` | 129 / 32,768 / 0 | 32 WMMA; LDS conflicts 0 | 80 loads; 192 perms | Add online softmax/P without score LDS |
| F3 | Done | Register softmax and packed C-to-A P fixture | FP16-P max abs `7.63e-6`; identity PV exact | 90 used (96 allocated) / 0 / 0 | 0 LDS conflicts | 1,227 VALU instructions | Fold into QK/PV |
| F4 | Done | Initial aligned online `128x64x8` Q8/FP16 kernel | Quantized reference relative L2 at most `4.0e-4`; public tolerance passes | 191 / 32,768 / 0 | `5.118 ms` vs AITER `4.378 ms` | Superseded by F5d | Retain as optimization history |
| F5a | Done | Replace `3+1` with one-pass FP32 scores | Public tolerance passes; max abs `0.01465` | 191 / 32,768 / 0 | `5.118 ms`; 16.9% behind AITER | Qualification pending | Keep lower-work schedule |
| F5b | Rejected | Four-row in-thread V transpose with original XOR | Public tolerance passes | 191 used (192 allocated) / 32,768 / 0 | `5.965 ms`; conflicts `29.4028` | N/A | Reject store-conflicted LDS mapping |
| F5c | Done | In-thread V transpose with AMD rotating-shared XOR | Public tolerance passes | 191 used (192 allocated) / 32,768 / 0 | `5.028 ms`; conflicts `2.7016` | Retained in F5d | Accept residual conflict after full timing matrix |
| F5d | Done | Transposed QK `K * Q^T`, pair softmax, compact C-to-A | Public tolerance passes; max abs `0.01465` | 191 used (192 allocated) / 32,768 / 0; conflicts `2.7016` | `4.101 ms`; 6.5% faster than AITER | Full matrix passes | Accept D=128 core |
| F6p | Preliminary | Six-shape aligned `do_bench` matrix | Timing only | Accepted core resources | Wins `7.1%` to `37.5%` | Wins `8.8%` to `47.8%` | Directionally pass; repeat with 30 explicit event samples |
| F6a | Done | Six-shape aligned event matrix, 30 samples/provider | Timing only | Accepted core resources | Wins `5.0%` to `28.9%` | Wins `8.2%` to `54.7%` | Aligned performance passes; run correctness and tails |
| F6b | Done | Six-shape public aligned correctness with chunked FP32 oracle | Zero failed elements; relative L2 `0.05363-0.05395` | Accepted core resources | All heads pass | All heads pass | Aligned correctness passes at default tolerance |
| F6c | Done | Three-shape `S=16384` event extension | Timing only | Accepted core resources | N/A | `S=16384` wins `19.2-55.1%` | Add `16384` to standard benchmark matrix |
| F7a | Done | Compile-time Q/KV tail guards and direct short path | Quantized tail max abs `9.77e-4`; public short gate passes | 191 used (192 allocated) / 32,768 / 0 for all variants | N/A | N/A | Keep variants; no runtime fallback |
| F7b | Done | Long-tail and real-shape public correctness | Zero failed elements; relative L2 `0.05344-0.05365` | Tail variant resources pass | `4095/4097`, `1500/1024`, and `5302` pass | `8800` passes | Tail correctness passes; profile representative tails |
| F7c | Done | Representative tail event benchmarks | Timing only | Tail variant resources pass | `N=4097` wins 15.5%; `N=5302` wins 26.3% | `N=8800` wins 47.6% | Tail performance passes |
| F7d | Done | Wide host arithmetic and checked int32 narrowing | Boundary arithmetic passes | No kernel change | N/A | N/A | Keep mandatory pre-launch checks |
| F7e | Done | D=128 direct-only public contract suite | 35/35 cases pass; zero failed elements | All D=128 variants | Production and tail cases pass | Production cases pass | D=128 correctness gate complete |
| F8a | Done | Shared D=64 specialization and four parallel instantiation units | Focused quantized/public checks pass | 146 aligned; 168 max tail / 16,384 / 0 | N/A | N/A | Keep shared template and parallel build |
| F8b | Done | Combined D=64/D=128 HND contract suite | 84/84 cases pass | All eight HND variants pass gates | Production and independent tails pass | Production cases pass | Superseded by the two-layout F9b suite |
| F8c | Done | D=64 nine-shape event matrix, 30 samples/provider | Timing only | Aligned: 146 used (152 allocated) / 16,384 / 0; conflicts 0 | Wins `3.9-12.5%` | Wins `1.7-17.1%`; S=16384 wins `2.3-43.0%` | D=64 performance qualified |
| F9a | Rejected | NHD with query tile as fastest grid axis | Layout equivalence passes | All NHD variants pass resource gates | Mixed | `D64/H32/N16384`: `4.793 TFLOPS` | Reject memory-partition aliasing |
| F9b | Done | Compile-time HND/NHD with head-interleaved NHD grid | 168/168 AITER-backed cases pass | All 16 variants pass; 191 VGPR max, 32 KiB LDS max, zero private/spills | `0.972-1.165x` vs AITER | `0.979-1.139x`; N=16384 `0.974-1.135x` | Accept both layouts |
| R10a | Done | Post-qualification static ISA, PC sampling, and 40-pass counter review | Read-only review; no kernel change | Qualified Phase 9 resources | Feather executes `1.264-1.557x` AITER VALU work | Dependency, barrier, waitcnt, and traffic regimes identified | Use findings to order Phase 10 |
| R10b | Done | Long-NHD Feather/AITER traffic and roofline profile at `H=32,N=16384` | Read-only review; no kernel change | Qualified Phase 9 resources | D64 Feather `22.584 TFLOPS`, `174.64 GB/s` | D128 Feather `22.306 TFLOPS`, `171.56 GB/s` | Confirm no-reuse K/V stream and grouping opportunity |
| R10c | Done | FlashAttention-CK LLC grouping policy model for gfx1151 | Source/model review | No kernel change | Candidate groups bounded by 32 MiB LLC | CK eight-group floor can exceed LLC at `H=56,N=16384,D=128` | Test bounded grouping, not unrestricted head-major order |
| F10a | Done | D=128-only bounded LLC-aware NHD head groups; D64 grouping rejected | 168/168 contract; grouped output bit-identical to ungrouped | 191 used/192 allocated max; 32 KiB LDS; zero private/spills | D128 NHD `0.972-1.171x` AITER | D128 NHD up to `1.376x` AITER; H32/N16384 traffic `31.435 -> 14.822 GiB` | Accept as production baseline; retain one launch for D64 |
| F10b | Rejected | Persistent linear online `(m,l)`, D=64 | Focused HND/NHD set passes 16/16 | 168 used max; 16 KiB LDS; zero private/spills | Paired focused geometric mean `-0.938%` | One of six paired cases wins; full matrix also negative | Restore Phase 10A recurrence; do not attempt D128 |
| F10c | Done | Cache one D64 HND decoded-Q fragment across key tiles | 168/168 contract; paired output bit-identical | 172 used max; 16 KiB LDS; zero private/spills | HND paired geometric mean `+1.294%` | All 9 HND rows win `1.07-1.64%`; NHD unchanged | Accept HND only; retain Phase 10A NHD schedule |
| F10d | Rejected | Hoist D64 V loads before K-to-V LDS barrier | Focused HND/NHD set passes 16/16 | Resources unchanged; 16 KiB LDS; zero private/spills | Paired focused geometric mean `-0.349%` | All six paired cases regress | Restore Phase 10A scheduling |
| R11a | Done | Findings-only review of final production ISA, profiles, and external schedules | Read-only review; no behavior change | Production resources unchanged | HND remains about `32-35 TFLOPS` | Long D64 NHD remains traffic limited | Open a separate measured Phase 11 campaign |
| R11b | Done | Freeze Phase 10C code objects and verify current build images | Byte-for-byte image comparison | Production resources and 56-byte Kargs unchanged | N/A | N/A | Use `phase11_baseline_freeze` for all Phase 11 paired controls |
| D11a | Deferred | Pure FP16 WMMA score or persistent output accumulation | Excluded because of known accuracy risk | N/A | N/A | N/A | Keep FP32 score and cross-key-tile output accumulators |
| F11a | Rejected | Progressive D=64 QK/PV LDS issue and descending `lgkmcnt` consumption | Focused outputs bitwise identical | All candidates <=177 VGPR; zero private/spills | QK4 `-2.263%`, QK2 `+0.312%`, PV4 `-3.165%` | QK4 `-1.149%`, QK2 `+0.239%`, PV4 `+0.371%` | Restore production QK/PV loops |
| F11a1 | Rejected | D64 QK issue all four K pairs before four WMMAs | Focused aligned outputs bitwise identical | HND `{145,145,168,168}`; NHD <=149; zero private/spills | HND geometric mean `-2.263%` | NHD geometric mean `-1.149%`; 1/6 wins overall | Reject full batch; try two-pair fallback |
| F11a2 | Rejected | D64 QK issue two K pairs before two WMMAs | Focused aligned outputs bitwise identical | At most 169 VGPRs; zero private/spills | HND geometric mean `+0.312%` | NHD geometric mean `+0.239%`; long H32/N16384 `-0.85%` | Below 0.5% gate; restore production QK |
| F11a3 | Rejected | D64 PV issue four V pairs before four WMMAs | Focused aligned outputs bitwise identical | HND `{172,172,177,176}`; NHD 152; zero private/spills | HND geometric mean `-3.165%` | NHD geometric mean `+0.371%`; 1/6 wins overall | Restore production PV loop |
| F11b | Done | Cache four D64 HND decoded-Q fragments, capped at three for KV-tail | 168/168 contract; paired outputs bitwise identical | HND `{175,175,171,171}` VGPRs; 16 KiB LDS; zero private/spills | Robust HND geometric mean `+0.789%`; 9/9 wins | NHD and D128 bodies byte-identical; dynamic VALU `-3.39%` | Accept as cumulative Phase 11 baseline |
| F11c | Done | Partition-aware strided, LLC-bounded D64 NHD groups | 168/168; selected tails/batch two bitwise identical | New 64-byte Kargs; `{130,130,151,151}` VGPRs; 16 KiB LDS; zero private/spills | Selected-domain geometric mean `+21.280%`; 12/12 wins | H32/N16384 fetch `-14.38%`, GCEA reads `-15.12%`; existing images byte-identical | Accept guarded selector; avoid strides divisible by eight |
| F11d | Rejected | D64 DPP row-share alpha fan-out | Focused outputs bitwise identical | HND <=175, NHD <=150 VGPRs; zero private/spills; D128 byte-identical | Robust HND geometric mean `-0.743%` | Robust NHD geometric mean `-0.602%`; 1/6 wins | Restore scalar readlane fan-out |
| F11e | Rejected | Isolated relaxed temporary P conversion/exp/Q-rounding probes | Half-up probe passed `168/168`; others stopped at static gate | FP32 QK/PV accumulators retained throughout | No probe passed timing | No probe passed timing | Restore production arithmetic; no tolerance relaxation |
| F11e1 | Rejected | D64 explicit adjacent-P packed RTZ conversion after FP32 beta scaling | Static gate only; no arithmetic relaxation requested | FP32 accumulators retained; D128 source path unchanged | Aligned HND instructions `1250 -> 1258` | Aligned/strided NHD instructions `1335/1351 -> 1354/1370` | Separate scaling erased the packed-conversion benefit; restore without timing |
| F11e2 | Rejected | D64 RNE P packing plus packed-FP16 beta multiply | Compile/static gate only | `v_cvt_pk_f16_f32` unsupported on gfx1151 | N/A | N/A | Scalar RNE converts plus packed multiply are strictly longer; do not combine with rejected RTZ path |
| F11e3 | Rejected | D64 FP16 temporary score exponential with immediate FP32 widening | Static gate only; FP32 sum/state retained | Aligned HND rose to 178 VGPRs; zero private/spills | HND instructions/VALU `1250/907 -> 1355/1009` | NHD instructions/VALU `1335/942 -> 1437/1041` | LLVM retained `v_exp_f32` and added 64 conversions; restore without timing |
| F11e4 | Rejected | D64 half-up Q E5M2 encoding | `168/168` at production thresholds | HND <=175, NHD <=162 VGPRs; zero private/spills; D128 byte-identical | Robust HND geometric mean `-0.461%` | Robust NHD geometric mean `+0.109%`; overall `-0.176%` | Static work is outside the key loop and misses the 0.5% gate; restore RNE |
| F11f | Rejected | Standalone D64 NHD Q8 `Br=256,Bc=64`, 16 waves | Focused aligned outputs bitwise identical | 131 VGPRs; 24 KiB LDS; zero private/spills | N/A | H32 `+0.90%`, H56 `-4.60%`; geometric mean `-1.889%` | Stop before profiling/tails; no production entry |
| F11g | Rejected | Standalone D64 HND `Br=128,Bc=128`, 8 waves | Compile-only | 242 VGPRs; 24 KiB LDS; zero private/spills | N/A | N/A | Exceeds 192-VGPR gate by 50; stop before timing/tails |
| F11h | Rejected | D64 HND SageAttention-style four-WMMA FP16 PV tile buffer with immediate FP32 merge | `168/168`; adversarial screen finite and passes without relaxation | HND `{192,192,189,192}` VGPRs; 16 KiB LDS; zero private/spills; NHD/D128 byte-identical | Focused HND geometric mean `-8.739%` | NHD unchanged | Merge cost and 192-VGPR pressure dominate; restore FP32 PV |
| F11z | Done | Cumulative 11B+11C final qualification at `01454e3` | `168/168`; final 36-case matrix complete | 20 kernels; 191 used/192 allocated max; 32 KiB LDS max; zero private/spills | D64/D128 HND geometric means `1.101x/1.087x` AITER | D64/D128 NHD geometric means `1.048x/1.156x`; overall `1.097x` | Close Phase 11; accepted source already committed |

## Verification Commands

Primary correctness and benchmark commands:

```bash
python test_attn_hip.py
python benchmark_attn_hip.py
```

The default benchmark output must contain separate rows for both layouts and head dimensions: all 36 `HND/NHD x D{64,128} x {16,32,56} x {4096,8192,16384}` cases. The correctness output identifies layout, batch, head count, query length, KV length, and head dimension for every case so tail failures cannot be hidden by an aggregate result.

Resource and ISA checks must report, for the exact new kernel symbol:
- used and allocated VGPRs;
- SGPRs;
- LDS bytes;
- private segment and scratch bytes;
- static counts for WMMA, `ds_load`, `ds_store`, `v_perm`, conversion, wait, and barrier instructions.

Profile at least:

```bash
rocprofv3 --kernel-trace --stats \
  --pmc L2CacheHit VALUInsts LDSBankConflict \
  --kernel-include-regex '<new-kernel-symbol>' \
  -d <out-dir> -o <prefix> -- python <single-kernel-profile-script>.py
```

Use a separate PMC run for occupancy-related counters when required. Single-run profiler durations are metadata aids, not benchmark results.

For traffic-bound investigations, also use isolated passes because rocprofv3 rejects some counter combinations:

```bash
rocprofv3 --pmc FETCH_SIZE \
  --kernel-include-regex '<new-kernel-symbol>' \
  -d <fetch-dir> -o counters --output-format csv -- \
  python <single-kernel-profile-script>.py

rocprofv3 --pmc GCEA_RDRAM_SIZE_REQ \
  --kernel-include-regex '<new-kernel-symbol>' \
  -d <gcea-dir> -o counters --output-format csv -- \
  python <single-kernel-profile-script>.py
```

Normalize `GCEA_RDRAM_SIZE_REQ` as 32-byte increments over dispatch duration. Keep `FETCH_SIZE`, GCEA bandwidth, and request-based L2 hit rate as separate reported measurements. Use final dispatches after two warmups and do not combine incompatible counters into one pass.

## References

- `kernel_attn/hip/featherattn_kernel.h`: shared D=64/D=128 CK Tile implementation and E5M2 helpers.
- `kernel_attn/hip/featherattn_{aligned,query_tail,key_tail,query_key_tail}.cu`: parallel specialization units.
- `kernel_attn/hip/hip_kernel.cpp`: checked Torch binding and direct dispatch.
- `kernel_attn/hip/hip_kernel.py`: extension loader and public wrapper.
- `test_attn_hip.py`: FP16 correctness gate.
- `benchmark_attn_hip.py`: AITER and HIP benchmark harness.
- `docs/gfx1151_reference.md`: resource allocation, profiling, and WMMA facts.
- `docs/input_shapes_attn.md`: real model attention shape inventory.
- `docs/hip_attention_optimization_plan.md`: outdated experiment history only.
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`: row-owned register score/softmax/PV reference.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`: stock register-Q pipeline and compile control.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`: custom-policy reference, not a packed-Q solution.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp`: gfx11 WMMA C-to-A permutation reference.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp`: gfx11 FP16 WMMA backend.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/core/numeric/float8.hpp`: evidence that generic gfx11 BF8 conversion is not the hot-loop decoder.
- `~/flash-attention/csrc/flash_attn_ck/mha_fwd_head_grouping_utils.hpp`: FlashAttention-CK grouped-forward dispatch integration.
- `~/flash-attention/csrc/composable_kernel/example/ck_tile/01_fmha/fmha_fwd_head_grouping.hpp`: RDNA LLC sizing, activation threshold, and head-group policy reference.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`: async/load scheduling reference only.
- `~/sageattention-autotune/docs/qattn_cutlass_fwd_plan.md`: comparison evidence for online denominator state, Q caching, and prefetch tradeoffs.
- `~/tmp/feather_attn/review/counter_matrix_summary.txt`: normalized HND/NHD Feather/AITER counter matrix.
- `~/tmp/feather_attn/review/model_grouping_and_roofline.txt`: long-NHD byte accounting, roofline, and grouping model.
- `~/tmp/feather_attn/review/model_optimization_bounds.txt`: grouped working-set and online-state resource bounds.
- `~/tmp/feather_attn/phase10c_hnd_qcache_full_matrix.csv`: historical Phase 10C wrapper benchmark artifact.
- `~/tmp/feather_attn/phase11_final/matrix/attn.csv`: authoritative final `01454e3` AITER/FeatherAttn benchmark matrix.
- `~/tmp/feather_attn/phase10c_hnd_qcache_images/featherattn_aligned.active.s`: qualified production aligned disassembly.
- `~/tmp/feather_attn/review/triton_exact_hnd_d64/*/attn_fwd.amdgcn`: active AITER LDS issue/consume schedule control.
- `~/tmp/feather_attn/phase10_group_sweep.log`: contiguous D64/D128 head-group timing controls.
- `~/sageattention-autotune/csrc/qattn/attn_utils.cuh`: `compute_fp16_sv_permuted_inst_buf` mixed FP16/FP32 PV reference.
- `~/rocm-libraries/projects/composablekernel/include/ck/utility/amd_wmma.hpp`: gfx11 FP16-output WMMA builtin and `op_sel` reference.
- `https://gpuopen.com/learn/wmma_on_rdna3/`: gfx11 WMMA C/D register count and FP16 `op_sel` element-selection reference.
- `~/rdna35-isa-markdown/rdna35_instruction_set_architecture.md`: gfx1151 waitcnt, packed conversion, FP16 transcendental, VOPD, and WMMA ISA reference.
