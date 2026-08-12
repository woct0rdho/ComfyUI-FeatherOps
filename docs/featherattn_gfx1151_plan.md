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
| Score and output accumulators | FP32 |

Do not add another block size, an autotune table, or a generalized FeatherAttn policy matrix without a separate measured justification. Tail support and the D=64/D=128 specializations share this block shape.

The correctness, resource, and two-layout performance qualification through Phase 9 is complete. Phase 10 is the active post-qualification optimization campaign. Its priorities are bounded NHD LLC grouping, linear online-softmax state, and D=64 dependency-chain reduction. These are proposals until their individual acceptance gates pass; the Phase 9 implementation remains the production baseline.

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

The Phase 0 event-based control run used 5 warmups and 30 measured launches per provider. Full p20/p80 intervals and environment metadata are in `~/tmp/feather_attn/phase0/controls.json`.

| H | S | AITER ms | Legacy end-to-end ms | Legacy prepacked ms |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 4096 | 4.304 | 5.800 | 5.563 |
| 16 | 8192 | 17.843 | 22.634 | 22.356 |
| 32 | 4096 | 10.373 | 11.502 | 11.141 |
| 32 | 8192 | 50.089 | 45.357 | 44.623 |
| 56 | 4096 | 17.615 | 20.882 | 20.128 |
| 56 | 8192 | 68.594 | 87.053 | 85.012 |

Every generated AITER control used eight waves, 32,768 bytes of dynamic LDS, 233 used VGPRs, and zero private segment. The `H=32, S=8192` ordering differs from the surrounding shapes and must be remeasured during final qualification rather than treated as a stable legacy win.

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

Historical Phase 5/6 event matrices qualified the transposed-score HND core and justified extending the benchmark surface to `S=16384`. Their raw samples remain under `~/tmp/feather_attn/phase5/` and `phase6/`, but their full tables are superseded by the final two-layout matrix in Phase 9.

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

The combined HND/NHD and D=64/D=128 public contract suite passes `168/168` cases. Historical D=64-only timing and profiling artifacts remain under `~/tmp/feather_attn/d64_benchmark/`; the current cross-layout performance results are reported once in Phase 9.

### Phase 9: Native HND/NHD Layouts (Complete)

The public wrapper accepts explicit `HND` and `NHD` layouts. Both keep D innermost and contiguous, so Q/K/V vector loads and output stores retain their existing vector widths. Layout is a compile-time template parameter. HND preserves the original flattened `(batch,head)` ownership and contiguous row stride; NHD decomposes batch/head and uses `num_heads * D` as the sequence-row stride. The four instantiation units now produce 16 kernels across layout, dimension, and tail mode.

The first correct NHD implementation assigned consecutive grid blocks to query tiles of one head. On power-of-two `H*D` row strides this caused severe memory-partition aliasing; the worst measured point, `D=64,H=32,N=16384`, reached only `4.793 TFLOPS`. The accepted NHD mapping makes head the fastest grid axis, interleaving adjacent head offsets across active workgroups. The same point rises to `22.553 TFLOPS`, and the full NHD matrix becomes competitive with AITER. HND block ordering is unchanged.

All 16 final kernels remain within the resource gates. HND metadata is unchanged. NHD D=64 uses 130 VGPRs for aligned/query-tail and 151 for KV/combined tails; NHD D=128 uses 191 VGPRs for every tail mode. NHD uses 16,384 bytes LDS for D=64 and 32,768 bytes for D=128, with zero private memory and zero SGPR/VGPR spills in every variant.

The final benchmark uses `triton.testing.do_bench` with 25 ms warmup and a 100 ms measurement budget per provider. Inputs are physically contiguous in the row's selected layout. For HND, AITER receives zero-copy transposed views because its interface interprets tensors as NHD; FeatherAttn receives HND directly. Throughput is `4 * B * H * N^2 * D / time`, and the ratio is FeatherAttn throughput divided by AITER throughput. Raw output is under `~/tmp/feather_attn/layout_benchmark/final/`.

#### HND

| D | H | N | AITER TFLOPS | FeatherAttn TFLOPS | Feather / AITER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 4096 | 33.779 | 34.762 | 1.029x |
| 64 | 16 | 8192 | 32.154 | 33.132 | 1.030x |
| 64 | 16 | 16384 | 31.706 | 32.684 | 1.031x |
| 64 | 32 | 4096 | 32.752 | 33.829 | 1.033x |
| 64 | 32 | 8192 | 31.054 | 32.753 | 1.055x |
| 64 | 32 | 16384 | 29.330 | 32.877 | 1.121x |
| 64 | 56 | 4096 | 29.752 | 33.083 | 1.112x |
| 64 | 56 | 8192 | 28.669 | 32.654 | 1.139x |
| 64 | 56 | 16384 | 29.092 | 33.013 | 1.135x |
| 128 | 16 | 4096 | 35.019 | 34.032 | 0.972x |
| 128 | 16 | 8192 | 32.479 | 33.154 | 1.021x |
| 128 | 16 | 16384 | 31.884 | 33.639 | 1.055x |
| 128 | 32 | 4096 | 31.553 | 34.570 | 1.096x |
| 128 | 32 | 8192 | 30.660 | 33.315 | 1.087x |
| 128 | 32 | 16384 | 30.756 | 34.443 | 1.120x |
| 128 | 56 | 4096 | 28.917 | 33.700 | 1.165x |
| 128 | 56 | 8192 | 29.855 | 33.180 | 1.111x |
| 128 | 56 | 16384 | 30.574 | 34.399 | 1.125x |

#### NHD

| D | H | N | AITER TFLOPS | FeatherAttn TFLOPS | Feather / AITER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 4096 | 32.804 | 32.601 | 0.994x |
| 64 | 16 | 8192 | 32.044 | 31.360 | 0.979x |
| 64 | 16 | 16384 | 32.026 | 31.204 | 0.974x |
| 64 | 32 | 4096 | 31.057 | 31.048 | 1.000x |
| 64 | 32 | 8192 | 28.517 | 29.866 | 1.047x |
| 64 | 32 | 16384 | 22.933 | 22.553 | 0.983x |
| 64 | 56 | 4096 | 29.143 | 30.747 | 1.055x |
| 64 | 56 | 8192 | 27.870 | 29.572 | 1.061x |
| 64 | 56 | 16384 | 28.193 | 29.860 | 1.059x |
| 128 | 16 | 4096 | 31.736 | 30.862 | 0.972x |
| 128 | 16 | 8192 | 30.291 | 30.885 | 1.020x |
| 128 | 16 | 16384 | 23.565 | 25.010 | 1.061x |
| 128 | 32 | 4096 | 26.290 | 28.430 | 1.081x |
| 128 | 32 | 8192 | 22.097 | 23.943 | 1.084x |
| 128 | 32 | 16384 | 21.521 | 22.293 | 1.036x |
| 128 | 56 | 4096 | 26.945 | 27.243 | 1.011x |
| 128 | 56 | 8192 | 27.565 | 27.068 | 0.982x |
| 128 | 56 | 16384 | 28.409 | 27.657 | 0.974x |

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

**Status: accepted for D=128 NHD; rejected for D=64.** The retained implementation keeps the Phase 9 head-fast mapping inside each launch and divides the physical head range into sequential launch subsets. Physical NHD strides continue to use the full head count; `head_start` and `launch_heads` affect only block decomposition. HND remains one launch.

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

The complete 36-case matrix passes. D=128 NHD is now `0.972x-1.376x` AITER, versus `0.972x-1.084x` in Phase 9, and its geometric-mean throughput improves `1.113x` over the Phase 9 table. HND and D=64 geometric means change by only `+0.7%`, `+0.4%`, and `+0.1%`, respectively. The public contract passes `168/168`, including arbitrary heads, independent tails, and batch two. Fresh metadata for all 16 kernels reports at most 191 used/192 allocated VGPRs, 32 KiB LDS, zero private memory, and zero spills; Kargs is 48 bytes.

Artifacts are under `~/tmp/feather_attn/phase10_group_*`, including the focused sweep, paired qualification, profile summary, contract log, metadata, and complete matrix. Phase 10A becomes the production baseline for subsequent experiments.

#### Optimization 10B: Persistent Linear Online `(m,l)` State

FeatherAttn currently reconstructs a log-domain state on every key tile:

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
new_max = max(running_max, tile_max)
alpha   = exp2(running_max - new_max)
beta    = exp2(tile_max - new_max)
running_sum = alpha * running_sum + beta * tile_sum
output      = alpha * output + beta * P @ V
running_max = new_max
```

Normalize output once in the epilogue with `rcp(running_sum)`. If probabilities remain `exp2(score - tile_max)`, multiplying P by `beta` still requires two state exponentials per tile. To reach AITER's 33-exponential structure, form P directly as `exp2(score - new_max)` and keep only `alpha = exp2(running_max - new_max)`. The exact generated dataflow, not source-level algebra, determines whether the second state exponential is actually removed.

The primary target is to remove the per-key-tile logarithm and reciprocal and shorten the serial softmax/update chain. A secondary target is reducing 34 exponentials to 33 without increasing reduction or broadcast cost. The 32 probability exponentials remain. Treat the standalone expected gain as `2-6%`, not as removal of the whole softmax cost.

Implement and qualify D=64 first. It has material register headroom. The state changes from one persistent `lane_lse` scalar to at least `running_max` plus `running_sum`, approximately one additional persistent scalar before compiler scheduling. D=128 is already at 191 used/192 allocated VGPRs. Do not retain the D=128 variant if it exceeds 192 allocated VGPRs, creates any private/scratch memory, spills, or requires less favorable LDS/wave residency.

Correctness must be rerun because changing the recurrence can alter rounding and all-masked/initial-state behavior. Preserve the public tolerance gates and compare relative L2 and maximum normalized tolerance ratio against the Phase 9 baseline, not only AITER.

#### Optimization 10C: D=64 Q Decode And Dependency Reduction

D=64 has 136 allocated VGPRs in aligned NHD and 152 in aligned HND, zero measured LDS bank conflicts, and proportionally more fixed softmax/conversion work than D=128. It is the only dimension where selective Q-fragment caching or a longer-lived decoded fragment is currently reasonable.

Test one narrowly scoped change at a time:
- retain one decoded FP16 Q fragment across its four QK WMMAs while consuming and overwriting it promptly;
- increase independent work between each Q LDS load and first use;
- simplify/rematerialize Q and K LDS addresses to reduce dependent `ds_load_b128` issue;
- interleave Q decode permutations with independent K loads and WMMAs without adding another K/V buffer.

The purpose is to reduce dependency stalls and selected loads/permutations, not merely to increase nominal prefetch depth. Expected, unvalidated gain is `1-4%` for D=64. Reject the experiment if aligned HND exceeds 168 allocated VGPRs, any tail exceeds 192, bank conflicts appear, or the complete D=64 matrix regresses.

Do not generalize Q caching to D=128. Its 191-VGPR result makes an eight-register decoded fragment incompatible with the current resource contract.

#### Optimization 10D: Barrier And Waitcnt Scheduling

Evaluate this only after 10B and the D=64 portion of 10C, because both can change the dependency graph. Keep the single phase-reused K/V LDS buffer and the four correctness barriers. Do not remove a barrier unless a formal producer/consumer argument covers all eight waves and the next overwrite phase.

Permitted experiments are instruction scheduling changes:
- move independent pointer arithmetic before LDS waits;
- start address construction for the next phase while current WMMA work is independent;
- increase legal LDS load/use distance;
- replace overly conservative compiler waits only when ISA inspection and repeated correctness tests prove the narrower wait is valid.

CK Tile's async and transpose-load pipelines are scheduling references, not drop-in replacements. A second K/V buffer would exceed the D=128 32 KiB LDS budget and is outside this campaign. Expected gain is `0-3%` with low confidence.

#### Deprioritized Phase 10 Work

- Deep K/V prefetch is secondary because SIMD utilization is already high and LDS queue-full pressure is negligible.
- Split-K/split-sequence remains conditional on low-grid, long-K workloads. The profiled long grid already has 4,096 CTAs and is traffic-bound.
- Additional block shapes and a broad autotuner remain outside scope.
- FP8 K, V, or P remain unjustified because they add conversion work without removing the persistent state that motivated Q8.
- D=128 Q caching is outside the resource envelope.
- Targeted AITER/NHD PC sampling is optional and should be run only if a candidate changes attribution that existing traffic, SQ, and occupancy counters cannot explain.

#### Phase 10 Execution Order And Gates

1. **Done:** bounded D=128 NHD grouping passed traffic, correctness, resource, and complete-matrix gates. D=64 grouping was measured and rejected.
2. Implement persistent linear `(m,l)` for D=64. Re-run correctness, resource metadata, ISA counts, focused PC sampling, and the D=64 matrix.
3. Attempt D=128 `(m,l)` only if the generated D=64 recurrence is clearly beneficial and a compile fixture indicates no resource-tier regression.
4. Test one D=64 Q-decode/dependency experiment at a time.
5. Revisit barrier/waitcnt placement after the recurrence and decode schedules settle.
6. Stop when a candidate fails its resource, correctness, or repeatable timing gate. Do not combine failed ideas to hide their individual attribution.

Phase 10A is the production baseline while later Phase 10 candidates are evaluated independently against it.

## Stop Rules

Stop or redesign before further optimization if any of these remain true:
- used VGPRs exceed 192 after the explicit liveness fixes;
- LDS exceeds 32 KiB;
- any private segment or scratch remains;
- the exact Q layout has material bank conflicts;
- a supported shape can overflow a narrowed device offset, counter, or launch dimension;
- realistic activation tests fail the FP16 tolerance or model quality check;
- `S=4096` remains more than 5% slower than AITER after basic scheduling;
- profiling shows Q LDS traffic or `v_perm_b32` issue consumes the complete occupancy gain.

## Deferred Work

The accepted fixed kernel does not include:
- additional `Br`, `Bc`, or wave-count variants;
- `Br=256`, 16-wave workgroups;
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
| F10b | Planned | Persistent linear online `(m,l)` state, D=64 first | Re-run public and baseline-relative numerical gates | D64 must remain within gates; D128 must stay at most 192 VGPRs with zero scratch | Expected `2-6%`, unvalidated | Long and intermediate sequences | Normalize once in epilogue; reject resource-tier regression |
| F10c | Planned | D=64 decoded-Q/dependency-chain reduction | Re-run D=64 correctness | Aligned HND target at most 168 allocated VGPRs; all tails at most 192 | Expected `1-4%`, unvalidated | D=64 only | Do not apply to D=128 |
| F10d | Planned | Barrier/waitcnt instruction rescheduling | Full recurrence and phase-order correctness | No second K/V LDS buffer; four correctness barriers retained | Expected `0-3%`, low confidence | Evaluate after F10b/F10c | Keep only with ISA proof and repeatable gain |

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
