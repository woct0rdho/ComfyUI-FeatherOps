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
| KV-only dO row stride | 72 FP16 elements |
| Main LDS | 13,312 B for fused/Q-only; 15,616 B for KV-only |
| Gradient ownership | Unique output ownership, no atomics |
| Output | Direct FP16 `dQ`, `dK`, `dV` stores |
| Saved state | Natural-log FP32 LSE plus stream-local FP32 Delta |

The KV-owned portion stages V in LDS, caches K rows, and cooperatively stages each Q and dO tile into the stride-20 transpose layout. Its compile-time long-sequence specialization also stages the exact dO rows with a padded 72-half stride, so dP reuses aligned LDS loads instead of loading dO from global memory a second time. It computes four GEMM-equivalent operations: QK score, dO/V dP, dS/Q dK, and P/dO dV. The Q-owned portion stages K and V in two 16-row tiles, caches Q and dO rows, and computes three operations: QK score, dO/V dP, and dS/K dQ. The C-to-A WMMA handoff converts each FP32 C fragment to the FP16 A-fragment layout with lane permutes.

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
| HND | 16 | 2048 | 25.286 | 27.641 | 1.093x |
| HND | 16 | 4096 | 26.290 | 28.567 | 1.087x |
| HND | 16 | 8192 | 26.459 | 30.107 | 1.138x |
| HND | 32 | 2048 | 25.483 | 27.667 | 1.086x |
| HND | 32 | 4096 | 25.381 | 29.878 | 1.177x |
| HND | 32 | 8192 | 25.174 | 30.281 | 1.203x |

The public D64 contract screen additionally covers `(NQ,NKV) = (33,35), (65,67), (65,129)` and batch two. The private candidate screen covers `(65,64), (129,65), (256,256), (257,257), (512,513)`, plus batch two and a cancellation pattern.

## Resource and Profile Results

The symbol-matched linked production image has the following gfx1151 profile:

| Symbol/workgroup | Logical VGPRs | Allocated VGPRs | SGPRs | LDS | Private/spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| D64 fused main, 128 threads | 178 | 192 | 45 | 13,312 B | 0 / 0 |
| D64 KV-only long phase, 128 threads | 169 | 192 | 33 | 15,616 B | 0 / 0 |
| D64 Q-only long phase, 128 threads | 162 | 168 | 34 | 13,312 B | 0 / 0 |
| Delta helper | 66 | 72 | 10 | 0 B | 0 / 0 |

The linked fused symbol contains 2,748 static instructions: 40 WMMA, 1,540 VALU, 941 SALU, 32 `v_perm_b32`, 16 cross-half permutes, 72 LDS loads, 40 LDS stores, 54 global loads, and 96 global stores. The linked dO-staged KV-only and unchanged Q-only symbols contain 1,489 and 1,242 static instructions respectively, with 16 WMMA for KV and 24 WMMA for Q. Relative to the pre-stage KV image, the new symbol removes eight static global `b128` loads and adds eight LDS `b128` loads; fused and Q-only code and resources remain unchanged. The six-case production correctness snapshot reports maximum relative L2 `0.00030144`, minimum per-head cosine `0.99999988`, and maximum absolute error `0.00119209`. The earlier full-image counter pass reports occupancy `46.62%`, `426.889 M` VALU instructions, `77.611 M` LDS instructions, `12.457 B` wave cycles, `2.440 B` wait-any cycles, `1.076 B` barrier waits, LDS latency `106.73` cycles, LDS conflict metric `24.32`, and ALU stalled by LDS `2.10%` for the fused baseline.

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

No current candidate satisfies all promotion gates, so production remains unchanged. If another D64 campaign is opened, evaluate only compile-time phase-local variants in this order:

| Rank | Experiment | Go signal | Immediate stop signal |
| ---: | --- | --- | --- |
| 1 | Reorder the dO-staged KV hot loop to shorten `CFragmentToA` and WMMA operand lifetimes and place independent LDS work across dependency gaps | Symbol-matched ISA removes dependency-chain instructions or waits, the MIR change is understood, and both H16/N4096 and H32/N8192 KV phase timings improve | Same linked schedule, resource-cap failure, spills, or no repeatable phase win |
| 2 | Micro-sweep the KV-only dO row layout/read order around the accepted padded design while preserving aligned 32-byte row reads | Lower LDS conflict/latency or wait-any at both profile shapes followed by a complete-path win | More barriers, scalarized LDS access, resource-cap failure, or a shape-specific win only |
| 3 | Test one barrier-overlap schedule, using ping-pong Q/dO staging only if it removes or overlaps a full workgroup barrier | A barrier or exposed wait is removed in linked ISA/counters and resources remain within the campaign limits | A second buffer retains the same synchronization points, raises waits, or trades the gain for instruction/register cost |
| 4 | Simplify phase-local address generation with precomputed bases and immediate LDS offsets | Fewer linked 64-bit address/offset instructions in the repeated KV body with unchanged vector memory operations | No ISA delta, extra pointer VGPRs, or wider lifetimes through the WMMA region |
| 5 | Reconsider persistent K/V compression only after ranks 1-4 create a schedule in which reduced storage crosses a natural resource boundary | Directional and preferred numerical qualification, a natural allocation/occupancy transition, and repeated complete-path wins | Direct P/dS compression, software conversion without a resource transition, or traffic-only improvement |

### Admission Gates

- Keep each experiment private and compile-time in the KV-only specialization. Do not add a public selector or alter HND, FP16 inputs/outputs, natural-log FP32 LSE, device validation, stream semantics, or the raw ABI.
- Compile with the production-equivalent `clang -O3 -fno-gpu-rdc --offload-arch=gfx1151` path and inspect the exact linked symbol. Require at most 192 campaign-rounded VGPRs, at most 32,768 bytes of LDS, zero private memory, and zero spills.
- Compare MIR only from matched-generation snapshots using block, slot, interval, and physical-ISA alignment. Virtual register numbers from separate snapshots are not identities.
- Run the six research correctness cases and the public D64 contract. Directional candidates require cosine at least `0.997`; Rel-L2 at most `0.10` is preferred and below `0.20` is exploratory. Exact scheduling/layout changes should retain the current approximately `0.0003` Rel-L2 behavior.
- Use phase-local H16/N4096 and H32/N8192 profiling only for triage. Promotion requires complete provider timing, including Delta, every main/helper launch, synchronization, conversions, and provider work, normalized with `F7`.
- Run repeated alternating paired measurements on all H `{16,32}` x N `{2048,4096,8192}` rows. Require credible positive complete-path evidence with no material row regression; a phase-only win, static resource reduction, or isolated fast sample is not sufficient.
- Re-run the `168/168` forward contract, the backward public contract, and explicit D128 rejection before integration.

### Stop Conditions

Stop each candidate at its first failed admission gate and preserve its evidence outside production. If the exact scheduling, padded-layout, barrier, and ISA-qualified address experiments fail to improve complete timing, close the D64 campaign with the current seven-GEMM kernel. Do not continue with generic bandwidth work, forced allocation, direct P/dS compression, or previously rejected K-cache/reload and Q-row-stage variants unless a future topology materially changes their traffic and lifetime assumptions.

D128 is a separate seven-GEMM design effort. It remains unsupported and must continue to fail validation; do not restore the deleted scalar reference kernel as part of D64 work.

## Artifacts

- Forward matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Current D64 timing run: `/tmp/feather_current_bwd_matrix.json`.
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
