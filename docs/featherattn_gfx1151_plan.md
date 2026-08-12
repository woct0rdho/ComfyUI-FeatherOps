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

The public FP16 operation supports every positive `num_heads`, `head_dim` in `{64,128}`, and every positive sequence length representable by the input tensors and the checked 32-bit address/launch arithmetic. Head count only scales grid X and uniform base offsets; `{16,32,56}` are benchmark targets rather than dispatch restrictions. Sequence length need not divide 64 or 128. The fixed kernel uses guarded query and key tails; block size is an implementation detail rather than an API restriction.

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

At the end, divide each output value by its distributed row sum and store FP16 directly in `[B, H, N, D]` layout. Do not round-trip output accumulators through LDS.

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

`kernel_attn/hip/featherattn_kernel.h` contains the shared D=64/D=128 CK Tile device template, E5M2 Q helpers, K/V LDS layouts, online softmax, tied gfx11 WMMA wrapper, and checked launch helper.

`kernel_attn/hip/featherattn_launch.h` defines only the compact launch argument structure and eight specialization declarations. It keeps CK Tile and Torch out of the host/kernel ABI boundary.

`kernel_attn/hip/featherattn_{aligned,query_tail,key_tail,query_key_tail}.cu` each instantiate one tail mode for D=64 and D=128. Keeping tail modes in separate translation units permits four concurrent hipcc jobs and limits recompilation after local changes.

`kernel_attn/hip/hip_kernel.cpp` owns stable Torch registration, dtype/device/layout/shape checks, wide host arithmetic, head-dimension and tail dispatch, and explicit launch errors. It is host-only so Torch and CK Tile HIP headers do not collide.

`kernel/hip/utils.py` accepts multiple extension sources plus extension-specific host and HIP flags. Linux retains the normal HIP wrapper; Windows alone uses the validated `-nohipwrapperinc` workaround.

`kernel_attn/hip/hip_kernel.py` builds the direct-only extension against `FEATHEROPS_CK_TILE_ROOT` or the default CK checkout and exposes only `attn_hip -> feather_attn_internal::attn_fp16`. CK Tile is an unconditional compile-time dependency: unavailable headers fail the extension build instead of producing an empty extension with a runtime error path.

`test_attn_hip.py` covers both dimensions, benchmark and arbitrary head counts, short/aligned/tail lengths, independent Q/KV lengths, batch greater than one, int32 boundaries, and expected rejection cases.

`benchmark_attn_hip.py` compares AITER and FeatherAttn across head dimensions, heads `{16,32,56}`, and lengths `{4096,8192,16384}`. The `FEATHER_ATTN_BENCH_HEAD_DIMS` environment variable filters the head-dimension axis.

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

A preliminary `triton.testing.do_bench` matrix reports FeatherAttn wins on all six aligned shapes: `7.1%` and `8.8%` at `H=16`, `37.5%` and `47.8%` at `H=32`, and `24.8%` and `20.9%` at `H=56` for `S=4096` and `8192`, respectively. These are directional results, not final qualification: Triton's `rep=30` argument is a millisecond budget, so the long-shape p20/p80 values collapsed to a single timed sample. Repeat the matrix with 30 explicit HIP-event measurements per provider before accepting these numbers. The preliminary artifact is `~/tmp/feather_attn/phase5/aligned_matrix.json`.

The explicit event-based matrix used five warmups and 30 independently timed launches per provider and passes every aligned timing gate:

| H | S | AITER median ms (p20-p80) | FeatherAttn median ms (p20-p80) | FeatherAttn speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 4096 | 4.347 (4.308-4.385) | 4.141 (4.101-4.213) | 5.0% |
| 16 | 8192 | 17.947 (17.758-18.058) | 16.582 (16.433-16.686) | 8.2% |
| 32 | 4096 | 10.400 (10.273-10.668) | 8.071 (8.027-8.153) | 28.9% |
| 32 | 8192 | 50.164 (49.385-50.895) | 32.435 (32.176-32.827) | 54.7% |
| 56 | 4096 | 17.645 (17.583-17.707) | 14.074 (14.021-14.162) | 25.4% |
| 56 | 8192 | 68.707 (68.636-68.991) | 56.153 (55.783-56.387) | 22.4% |

These timings include fused Q conversion. The final direct shell removed the legacy temporary FP8 allocations. Raw samples and extrema are in `~/tmp/feather_attn/phase5/aligned_matrix_events.json`.

After the `4096/8192` matrix passed, the standard benchmark surface was extended to `S=16384`. With the same five-warmup/30-event protocol, FeatherAttn measures `65.221 ms` versus AITER `91.135 ms` at `H=16` (`39.7%` faster), `131.495 ms` versus `203.903 ms` at `H=32` (`55.1%` faster), and `229.407 ms` versus `273.443 ms` at `H=56` (`19.2%` faster). The corresponding p20-p80 intervals are `64.657-65.424`, `130.737-132.147`, and `228.598-230.557 ms` for FeatherAttn. `benchmark_attn_hip.py` now covers all nine `{16,32,56} x {4096,8192,16384}` shapes; raw `16384` samples are in `~/tmp/feather_attn/phase6/benchmark_16384_events.json`.

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

The final repository public-contract test passes `84/84` cases across D=64 and D=128. It covers benchmark heads at `257`, `4096`, and `8192`; general head counts `{1,2,3,4,7,24,30,40,48}`; short lengths from `1` upward; the `1023/1024/1025` accuracy-gate boundary; independent query/KV tails; query and KV tile boundaries; and `B=2`. It also verifies explicit rejection of unsupported head dimensions, dtypes, and non-contiguous layouts.

Representative tail timing uses five warmups and 30 event samples per provider. FeatherAttn measures `4.538 ms` versus AITER `5.240 ms` at `H=16,N=4097` (`15.5%` faster), `39.544 ms` versus `58.358 ms` at LTX `H=32,N=8800` (`47.6%` faster), and `25.692 ms` versus `32.453 ms` at H3 `H=56,N=5302` (`26.3%` faster). Tail performance is qualified; raw samples are in `~/tmp/feather_attn/phase6/tail_event_benchmarks.json`.

The production host path now proves every reachable FP16 byte offset with `__int128` arithmetic before any device-side int32 narrowing. It requires positive batch, head, query, and KV dimensions; rejects negative or greater-than-int32 strides; checks Q/K/V/output maximum strided byte offsets against signed int32; and validates tile and grid ranges before launch. Boundary tests cover the arithmetic without allocating impractically large tensors.

### Phase 8: D=64 Specialization And Parallel Build (Complete)

D=64 reuses the accepted `128x64x8` ownership and online-softmax template. It halves Q8 LDS to 8 KiB and phase-reused K/V LDS to 8 KiB, for 16 KiB total. K staging issues two vector loads per thread rather than four. V staging uses the same rotating-shared swizzle but maps the two unused D=128 lane-row groups onto additional N rows and stores two FP16 rows per lane. The D=64 query scale is the exact `log2(e)/sqrt(64)` constant.

The implementation lives in `featherattn_kernel.h`. Four small translation units instantiate aligned, query-tail, KV-tail, and combined-tail pairs for D=64 and D=128; Ninja compiles those units in parallel. The Torch binding is a host-only C++ translation unit and does not parse CK Tile. A clean build and import completed in approximately `7.0 s` with `MAX_JOBS=32`.

D=64 resource metadata passes every gate:

| Variant | Used VGPRs | Allocated VGPRs | SGPRs | LDS bytes | Private / spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned | 146 | 152 | 28 | 16,384 | 0 / 0 |
| Query tail | 146 | 152 | 30 | 16,384 | 0 / 0 |
| KV tail | 168 | 168 | 40 | 16,384 | 0 / 0 |
| Combined tail | 149 | 152 | 44 | 16,384 | 0 / 0 |

The aligned D=64 static body contains 32 FP16 WMMAs, 68 `ds_load_b128`, four `ds_store_b128`, eight `ds_store_b32`, 80 `v_perm_b32`, zero `v_permlane16_b32`, 18 `v_permlanex16_b32`, 34 exponentials, one logarithm, four barriers, and zero `ds_bpermute_b32`. Eight `N_KV=4096` profile dispatches each report 152 allocated VGPRs, 16,384 bytes LDS, zero scratch, `VALUInsts=29979.0`, and `LDSBankConflict=0.0`. Metadata, ISA, and profiler artifacts are under `~/tmp/feather_attn/`.

The combined D=64/D=128 public contract suite passes `84/84` cases. On D=64, aligned production cases have relative L2 `0.0531-0.0547`; the largest long-case absolute error is `0.0419`, and the worst long tolerance ratio is `0.740`. The largest short-case absolute error is `0.144`, with worst tolerance ratio `0.879` under the documented `0.10/0.10` gate.

The final D=64 event matrix uses five warmups and 30 independently timed launches per provider and beats AITER on all nine target shapes:

| H | S | AITER median ms (p20-p80) | FeatherAttn median ms (p20-p80) | FeatherAttn speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 4096 | 2.172 (2.149-2.182) | 2.091 (2.080-2.146) | 3.9% |
| 16 | 8192 | 8.493 (8.477-8.529) | 8.347 (8.317-8.385) | 1.7% |
| 16 | 16384 | 34.102 (34.046-34.209) | 33.332 (33.253-33.421) | 2.3% |
| 32 | 4096 | 4.544 (4.518-4.552) | 4.198 (4.192-4.216) | 8.2% |
| 32 | 8192 | 19.088 (18.507-19.652) | 16.660 (16.595-16.762) | 14.6% |
| 32 | 16384 | 95.529 (94.939-96.124) | 66.798 (66.673-66.905) | 43.0% |
| 56 | 4096 | 8.259 (8.193-8.372) | 7.340 (7.287-7.359) | 12.5% |
| 56 | 8192 | 34.151 (34.047-34.315) | 29.175 (29.145-29.254) | 17.1% |
| 56 | 16384 | 137.747 (136.396-137.998) | 119.429 (118.672-119.559) | 15.3% |

The standard benchmark now treats D as an axis and defaults to both `64` and `128`. Set `FEATHER_ATTN_BENCH_HEAD_DIMS=64` or `128` to measure one specialization. Raw D=64 samples are in `~/tmp/feather_attn/d64_benchmark/events.json`; the preliminary budget-based matrix remains beside it for comparison.

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
| F8b | Done | Combined D=64/D=128 contract suite | 84/84 cases pass; explicit rejection checks pass | All eight variants pass gates | Production and independent tails pass | Production cases pass | Correctness complete |
| F8c | Done | D=64 nine-shape event matrix, 30 samples/provider | Timing only | Aligned: 146 used (152 allocated) / 16,384 / 0; conflicts 0 | Wins `3.9-12.5%` | Wins `1.7-17.1%`; S=16384 wins `2.3-43.0%` | D=64 performance qualified |

## Verification Commands

Primary correctness and benchmark commands:

```bash
python test_attn_hip.py
python benchmark_attn_hip.py
```

The default benchmark output must contain separate rows for both head dimensions and all 18 `{16,32,56} x {4096,8192,16384}` shapes. The correctness output identifies batch, head count, query length, KV length, and head dimension for every case so tail failures cannot be hidden by an aggregate result.

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
