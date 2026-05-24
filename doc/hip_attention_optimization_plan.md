# gfx1151 HIP fp16/fp8e5m2 Attention Kernel Optimization Plan

## Scope and Metric

- Target kernel: `kernel_attn/hip/hip_kernel.cu` with wrapper `kernel_attn/hip/hip_kernel.py`.
- Interface contract: fp16 Q/K/V inputs and fp16 output. Internal/prepacked K/V may use fp8e5m2.
- Target shape: Qwen-Image attention with `B=1`, `H=24`, `S in {1024,2048,4096,8192}`, `D=128`.
- Accuracy gate: `python test_attn_hip.py`, compare to fp16 PyTorch SDPA with `abs(out - ref_fp16) <= 0.05 * abs(ref_fp16) + 0.05`.
- Benchmark: `python benchmark_attn_hip.py`, compared with fp16 AITER Triton attention kernel.
- Keep rule: accuracy passes, `S=4096` improves or stays flat, `S=8192` does not materially regress. No scratch/private segment, and bank conflicts stay zero/negligible, unless the runtime win clearly justifies them.

## Current Baseline Metric

| S | AITER TFLOPS | HIP TFLOPS | HIP Prepacked TFLOPS | HIP Config | Prepacked Config |
|---:|---:|---:|---:|---|---|
| 1024 | 22.236013 | 19.219206 | 24.799499 | `(64,32,8)` | `(64,32,8)` |
| 2048 | 27.309833 | 22.816377 | 25.306275 | `(64,32,8)` | `(64,32,8)` |
| 4096 | 27.295072 | 23.050048 | 24.206595 | `(64,32,8)` | `(64,32,8)` |
| 8192 | 27.463785 | 23.154828 | 23.532040 | `(64,32,8)` | `(128,32,8)` |

- Config list: `(64,32,8)`, `(128,32,8)`.
- Benchmark selected `(64,32,8)` for all end-to-end sizes and most prepacked sizes. `(128,32,8)` won prepacked `S=8192`.
- Decomposed selected-path timings: `S=4096` prepacked `8.4499 ms`, end-to-end `8.8896 ms`; `S=8192` prepacked `35.3245 ms`, end-to-end `35.4353 ms`.
- Forced profile for `(64,32,8)` at `S=4096`: `8345.861 us`, `LDSBankConflict=0.0`.
- Space usage: `(64,32,8)` VGPR `187`, scratch `0`, text size `0x1fd0`; `(128,32,8)` VGPR `222`, text size `0x3374`.
- Interpretation: halving `Br` halves output accumulator tiles per workgroup (`kRegTiles` `8 -> 4`) and lowers VGPR/code size enough to offset twice as many query CTAs. Keeping `Bc=32` preserves the zero-conflict LDS pattern.

## Current Kernel Structure

- Quantization kernel converts fp16 K/V to contiguous fp8e5m2 buffers with branchless packed RNE.
- Forward stages raw fp8 K and V tiles in LDS with K/V stride `D + 16`.
- Forward keeps the output accumulator in fp32 registers and stores only once at the end.
- Forward still materializes `Si`, the `Br x Bc` fp16 score/probability tile, in LDS.
- Row softmax is still concentrated in `tx < Br`, one thread per row.
- `Si` stride is `Bc + 8`, which removed measured LDS bank conflicts for the kept `Bc=32` paths.
- HND layout (`B,H,S,D`) is the optimized internal/public fast path. NHD view and one-call NHD pack/swizzle variants were slower.

## Structural Comparison with AITER (gfx1151)

AITER Triton attention for RDNA uses `BLOCK_M=128, BLOCK_N=32, 8 warps (256 threads)` with `PRE_LOAD_V=False`. Key structural differences from our HIP kernel:

| Aspect | AITER (Triton) | Our HIP (A13-A baseline) |
|---|---|---|
| Score tile S | WMMA C-fragment → fp32 registers | WMMA C-fragment → fp16 LDS (Si write) |
| Softmax | `tl.max(qk,1)` → warp shuffle → exp in registers | Barrier → `tx<Br` scalar threads read Si from LDS |
| Probability P | Registers (C-fragment modified in-place) | Written back to LDS |
| PV dot | P in registers (converted to A-fragment by Triton layout) | P read from LDS into A-fragment |
| Barriers per K-iter | 0 (all register-to-register) | 3 (K/V load, Si write, Si softmax write) |
| Q residency | Load once from DRAM, held in registers for all K/V tiles | Re-read from DRAM every K iteration |
| V load | Loaded after P is ready (PRE_LOAD_V=False) | Pre-staged in LDS alongside K |
| Row softmax threads | All 256 threads (via Triton warp reductions) | `tx < Br` (64 threads); 192 idle |
| m_i / l_i storage | [BLOCK_M] fp32 vectors in registers per thread | LDS arrays, accessed by `tx < Br` threads |
| Config | BLOCK_M=128, BLOCK_N=32, 8 warps | Br=64, Bc=32, 8 waves (dropped to 64 due to Si LDS regpressure) |

Root cause: The Si LDS materialization forces 3 barriers, serialized row-softmax, doubled LDS traffic per score element, and drove Br from 128→64 to contain VGPR. AITER avoids all of these by keeping the entire QK→score→softmax→P pipeline in registers.

Why we can't just do BLOCK_M=128: With 128 rows and Si in LDS, the LDS allocation doubles, output accumulator tiles double (kRegTiles 4→8), and the `tx < Br` softmax serialization gets 2× worse. Moving score to registers first (A16-A) is a prerequisite for any Br=128 experiment.

## Structural Comparison with CK-Tile FMHA (gfx1151)

CK-Tile's QR (Q-in-Registers) pipeline uses `BlockGemmARegBSmemCRegV2` for both QK and PV gemms. Key structural differences beyond the AITER comparison above:

| Aspect | CK-Tile QR Pipeline | Our HIP (A13-A baseline) |
|---|---|---|
| K LDS layout | 3D+padding: `(K/kKPack, N, kKPack)` with inter-chunk pad of `(N+1)*kKPack` between K-column chunks | 2D row-major: `N × (D+16)` with 16-byte pad between full rows |
| K load pipelining | Double-buffered in D-loop: load K0[i+1] while computing GEMM0 on K0[i] | Load entire K tile at once, then compute all D chunks |
| V load timing | Post-softmax on gfx11 (VPrefetchPoint::AfterGemm0Tail) | Pre-staged alongside K before QK gemm |
| V register shuffle | `shuffle_tile` converts DRAM RowMajor V → bank-conflict-free LDS layout | Not needed (V is ColumnMajor, D-contiguous) |
| K/V LDS sharing | Single buffer: `max(SingleKSize, SingleVSize)`. K and V share LDS space | Separate Ks and Vs arrays, both always allocated |
| sched_group_barrier | `__builtin_amdgcn_sched_group_barrier(DS_READ, N)` and `(MFMA, N)` interleave LDS reads with WMMA | None |
| C→A P conversion | `PermuteWarpGemmCToA` on gfx11: single-call layout conversion for P from C-fragment to A-fragment | `cfrag_value_at` + manual `__shfl` (used only in WMMA fixture so far) |
| Score softmax | Row reductions via `block_tile_reduce` + `block_tile_reduce_sync` on register C-fragment | Sequential scalar on `tx < Br` threads reading Si from LDS |
| m_i / l_i | Distributed register tiles matching S's C-fragment distribution (all threads own rows) | LDS float arrays, accessed by Br threads only |

Key CK design decisions for gfx1151:

1. VPrefetchPoint::AfterGemm0Tail on gfx11 (pipeline body line 676-679): CK deliberately loads V after the QK gemm completes on RDNA targets. This trades V-load latency hiding for lower LDS pressure. The comment says pre-softmax V prefetch is for CDNA only.

2. K double-buffering in the D-loop (pipeline body lines 632-665): CK loads K0[i+1] from DRAM while computing GEMM0 on K0[i]. This is a software pipeline inside the inner D-loop, interleaving `load_tile(k_dram_window)` with `gemm_0(s_acc, q_slice, k_lds_window)`. Our kernel loads all K at once per K/V tile.

3. K LDS inter-chunk padding (policy lines 599-619): CK pads `kKPack` bytes between each K-column chunk (not between full K-rows). For D=128 with kKPack=16: K is organized as 8 chunks of `[32 rows × 16 elements]` with 16 bytes of padding between chunks. Total LDS = `(32+1) × 8 × 16 = 4224` bytes (vs our `32 × 144 = 4608` bytes). The inter-chunk spacing avoids bank conflicts when warps read adjacent K columns during WMMA.

4. LDS buffer sequencing (policy lines 372-414): In the async path, CK's `LdsBufferSequence` maps each K0/K1 iteration to an LDS buffer index, with hand-tuned specializations for common loop counts (4x4, 4x2, 3x3, etc). This prevents the last V GEMM1 write from colliding with the next K GEMM0 read when K/V share buffers.

Implications for our kernel: The CK comparison surfaces microarchitectural optimizations (K double-buffering, sched_group_barrier, inter-chunk LDS padding) that are orthogonal to the A16-A register score/softmax work. These can be pursued either independently or as additive improvements on top of a register-P/PV baseline.

## Structural Comparison with SageAttention (gfx12 HIP kernel)

SageAttention's GFX12 native kernel (`qk_int_sv_gfx12_native.cu`, ~9300 lines) uses int8 quantization for both Q and K, with online per-block scaling. The kernel is compiled 5 times with different build-defines for aux/prepare/attn_f16/attn_fp8/rawq_fp8 variants. Key structural differences from our kernel:

| Aspect | SageAttention GFX12 | Our HIP (A13-A baseline) |
|---|---|---|
| QK WMMA type | `__builtin_amdgcn_wmma_i32_16x16x16_iu8` (int8 Q × uint8 K → i32) | `__builtin_amdgcn_wmma_f32_16x16x16_f16` (fp16 Q × fp16 K → fp32) |
| Q quantization | Per-block int8 quantized with pre-multiplied scale (q_scale × sm_scale × log2e) | fp16, no quantization |
| K quantization | Per-block uint8 with smooth-K (per-head mean centering before quantization) | fp8e5m2 with branchless packed RNE, no mean centering |
| Score tile residency | `float8_vec score_cache[ColTiles]` in registers (1Q) or streamed in groups of 2-4 col-tiles (2Q with `StreamColTiles`) | fp16 Si tile fully materialized in LDS |
| Probability tile residency | `half8_vec prob_cache[ColTiles]` in registers, or converted on-the-fly from streamed scores | Written back to Si LDS |
| Query groups per wave | 2 (each wave processes 2×16=32 query rows simultaneously, `QGroups=2`) | 1 (each wave processes 16 query rows) |
| V-load strategies | 6+ strategies: standard shared, transposed, lane-major, transpose-on-load, hardware transposed global load (gfx12xx), prepacked lane-major | 1: fp8 pre-staged in LDS, decoded on read |
| K-load strategies | Lane-major global load (eliminates decode), shared-memory lane-major, prepacked lane-major | fp8 pre-staged in LDS, decoded via perm on read |
| P decode for PV WMMA | Zero-decode: P already in WMMA A-fragment fp16 layout (from `prob_cache`) | P read from LDS, decoded fp16→A-fragment |
| Softmax | Online with `exp2(score - m + kF16SoftmaxOffset)`, reduction via `__shfl_xor` on register cache | Barrier → sequential scalar on `tx < Br` |
| Smooth-K | Per-head K mean subtraction before int8 quantization (default on) | None |
| Block sizes | `BlockRows` in {64,128,256,512,1024}, `BlockCols` in {32,64,128}, `HeadDim` in {16,64,128} | `Br` in {64,128}, `Bc=32`, `D=128` only |
| Hardware features | `global_load_tr_b128` on gfx1200+ (hardware transposed load), `__builtin_amdgcn_wmma_*_gfx12` intrinsics | WMMA v2 intrinsics (same ISA family, gfx1151 supports all gfx12 WMMA builtins) |

Key SageAttention design decisions:

1. Int8 quantization for Q and K (not just K/V like us): The Q @ K^T dot product uses int8 WMMA (`v_wmma_i32_16x16x16_iu8`), which has the same 32-cycle latency as fp16 WMMA on gfx1151. The win is memory: int8 packs 4 elements per uint32 (vs 2 for fp16), halving K LDS size and reducing Q register pressure (i32x2 holds 8 int8 values vs fp16x16 holding 16). The Q @ K score is accumulated as i32, then converted to fp32 for softmax.

2. 2-query groups per wave (2Q): The most optimized kernel variant processes two 16-row query groups per wave (`QGroups=2`). Each wave has two independent sets of Q registers, score caches, m/l states, and output accumulators. This doubles Q throughput for the same K/V LDS bandwidth. The K/V tiles are loaded once and shared across both Q groups. For `BlockRows=128`, 8 waves × 2 groups = 256 query rows processed per workgroup.

3. Streaming col-tile softmax (`StreamColTiles`): Instead of computing all `BlockCols/16` score tiles before softmax, scores are computed in groups of 2 col-tiles, with softmax and PV accumulation happening incrementally per group. This reduces score register pressure from `float8_vec × ColTiles` to `float8_vec × 2`, crucial for 2Q mode where score caches exist per Q-group.

4. Lane-major data layout: K, V, and P are stored in "lane-major" order — data arranged exactly as WMMA instructions consume it (grouped by [col_tile][d_tile][lane_id]). This eliminates all runtime repacking. Our kernel uses `fp8e5m2x4_to_half2x2` + `__builtin_amdgcn_perm` to decode K on every LDS read.

5. Prepacked lane-major optional path (`PrepackedLaneMajorKV`): K/V data can be pre-packed offline into WMMA lane-major order, enabling direct `global_load→register` without LDS staging or decode. This is the ultimate zero-overhead load path but requires offline prepacking (similar in spirit to our offline fp8 K/V quantization).

6. exp2 with offset (`kF16SoftmaxOffset = 8.807f`): SageAttention adds a constant offset to the exponent argument to improve numerical stability for the int8 score range. This is specific to int8 quantization and not applicable to our fp16 scores.

Relevance to gfx1151:
- All GFX12 WMMA builtins (`v_wmma_*_gfx12`) are available on gfx1151 (RDNA3.5 supports the same WMMA ISA as RDNA4 gfx12xx).
- `global_load_tr_b128` is NOT available on gfx1151 (gfx12xx only). Fallback is scalar load.
- The 2Q, streaming softmax, and lane-major layout patterns are all applicable to gfx1151.
- Int8 WMMA has the same throughput as fp16 WMMA on gfx1151, so the int8 quantization choice is about memory bandwidth/register pressure, not compute throughput.

Implications for our kernel:
1. Lane-major K layout (A19-B) is the most directly actionable: eliminate fp8→fp16 perm decode on every K LDS read by storing K in WMMA B-fragment lane order. This is an independent optimization applicable to any kernel variant.
2. Streaming col-tile softmax (A19-A) complements the register-P/PV plan (A16-A): if register pressure from score caches is a concern, streaming reduces the per-wave score cache from 2 col-tiles (for Bc=32) to 2 streamed groups (no change for Bc=32). More relevant if we later increase Bc or Br.
3. 2Q architecture (A19-C) is a major structural change that doubles Q throughput. Best pursued after A16-A register-P/PV is stable, since 2Q requires per-wave score/softmax/PV state for each Q group.
4. Int8 Q quantization is NOT worth pursuing on gfx1151: the int8 WMMA throughput equals fp16 WMMA, so the only benefit would be reduced Q register pressure, which doesn't justify the quantization accuracy risk and complexity of per-block scale management.

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
| A14-A | Kept infra | WMMA fragment-layout fixture | Validates C-fragment to PV A-fragment shuffle mapping and caught the failed dynamic-index shuffle pattern. |
| A14-B | Rejected | Register-P/PV retry `(64,32,4)` | Correctness passed, but autotune did not select it; forced profile regressed to `16274.958 us` with VGPR `256` and private segment `0x38`. |
| A15-A | Rejected | QK-produced partial row maxima | Correctness passed, but `S=4096` fell to `18.416490` TFLOPS; codegen added `32` hot-path `ds_bpermute` reductions. |
| A16-A | Rejected | Parallel per-wave register softmax (keep Si LDS) | Correctness passed with wave_x==0 guard. Regressed 34-39%: only 4/8 waves active during softmax, scalar LDS reads replaced vector reads, shuffle overhead exceeded baseline. |
| A16-B | Blocked | Q-in-Registers (additive) | Blocked by A16-A rejection; Q-in-Registers is independent but adds ~64 VGPR per thread with no register-P to benefit from it. |
| A16-C | Blocked | Integrated benchmark and tuning | Blocked by A16-A rejection. |
| A16-D | Blocked | Warp-parallel softmax with Si in LDS | Blocked: the underlying QK/output tile mismatch (D/Bc=4) means any per-wave softmax either halves active waves or requires cross-wave LDS merging. A16-A demonstrated both approaches regress. |
| A16-E | Blocked | V just-in-time load | Blocked by A17-B rejection (same mechanism — added V-load barrier regressed 1-4%). |
| A17-A | Blocked | K double-buffering in D-loop | Blocked: CK-style double-buffering loads K from global during D-loop, but our kernel pre-loads all K into LDS. Adding global loads in the hot D-loop competes with Q loads and WMMA; not viable without async copy (unavailable on gfx1151). |
| A17-B | Rejected | K/V LDS space sharing + JIT V load | Correctness passed; benchmark regressed 1-4% due to added V-load barrier. LDS savings didn't improve occupancy (14.5KB→10.2KB, same WG/WGP count). |
| A17-C | Rejected | sched_group_barrier compute/load hints | Regressed 13-19% across all sizes. Strict 1:1 DS:MFMA alternation broke compiler's LDS/WMMA interleaving; also shifted autotune from (64,32,8) to (128,32,8). |
| A19-A | Blocked | Streaming col-tile incremental softmax+PV | Blocked: Bc=32 gives only 2 col-tiles, making streaming a no-op (1 group of 2 tiles). Streaming only benefits Bc ≥ 64, which is rejected. |
| A19-B | Rejected | Lane-major K layout (zero-pad decode) | NaN output (byte-layout bug in zero-pad conversion). Even if correct, theoretical analysis shows <0.2% benefit: saving 4 perm ops per WMMA pair (~32 cycles per K-iteration) is offset by doubled LDS write bandwidth (~32 cycles additional). Net benefit too small to measure. |
| A19-C | Blocked | 2-query-groups per wave (2Q architecture) | Blocked by register-P/PV rejection: 2Q doubles per-wave state (Q ×2, score caches ×2, out_frag ×2), which without register-P/PV would exacerbate VGPR pressure. Requires register-P/PV as prerequisite.

## Logs and Reasoning

### Kept Structural Wins

- Register output accumulator was the first major structural win: moving `Oi` from fp16 LDS to fp32 registers lifted `S=4096` from roughly `10 TFLOPS` into the mid/high teens, then 8-wave raised it further.
- `Si` padding to `Bc + 8` is non-negotiable for current `Bc=32` paths: it took measured HIP `LDSBankConflict` from nonzero to `0.0` without changing the algorithm.
- Raw K/V LDS staging fixed the V-side global scalar-load problem: fwd-symbol `global_load_d16_u8`/`global_load_d16_hi_u8` dropped to `0/0` while scratch stayed `0`.
- Making the staged path the normal config avoided sentinel/autotune clutter and left only the code path that matters for the fp8 D=128 kernel.
- `(64,32,8)` appears to be a better staged shape than `(128,32,8)` because it lowers VGPR and output-fragment pressure while preserving `Bc=32` LDS behavior.
- `test_wmma_fragment_layout.py` validates the register-P/PV fragment shuffle using a synthetic C-fragment, the corrected shuffle, and an identity PV WMMA.
- The corrected register-P/PV mapping alone is not enough. The first correct full-attention retry spilled and was slower than the staged baseline.

### Durable Findings

- The remaining core bottleneck is the `Si` score/probability LDS round trip plus serial row softmax work, not K/V quantization or output accumulator storage.
- HND (`B,H,S,D`) is the optimized layout for this kernel family. NHD should be handled only if integration requires it.
- Keep zero/negligible LDS bank conflicts as a hard target. AITER and the best HIP paths both achieve `LDSBankConflict=0.0`.
- K/V quantization is `O(BHSD)` and stays small at large S. Prepacked and end-to-end timings should still be reported separately.
- More registers can be good on gfx1151 when they remove LDS traffic, but VGPR pressure still matters: `(64,32,8)` currently beats `(128,32,8)` by reducing output-fragment pressure.
- `Bc=32` is the safest current K/V tile width because it preserves the known clean LDS pattern.
- Do not spend cross-lane reductions just to remove the first `Si` max-read pass. The score/probability LDS traffic must be removed more completely to pay off.
- QK/output tile mismatch is the fundamental barrier to register-P/PV: For `(64,32,8)` with D=128, per-wave QK tiles = 1 but per-wave output tiles = 4 (ratio `D/Bc = 4`). Each wave computes P for 16 rows but needs P for 64 rows for its output tiles. Full register-P/PV requires either (a) tile redistribution matching QK and output per-wave counts, (b) each wave computing QK for all its output row slices (4× redundant QK work), or (c) cross-wave P sharing via LDS (defeating the register-only goal). None are viable without a larger architectural redesign.

### Rejected Paths Worth Remembering

- Standalone V transpose/prepack did not pay. It should only be reconsidered inside a new register-P/PV forward layout.
- Smaller `Si` or K/V padding is not automatically better. `Si + 4` and K/V `D + 8` both preserved or reported low conflicts but regressed runtime, showing dynamic LDS phase/scheduling matters.
- Row ownership alone is not useful if `Si` remains and V access worsens. A viable row-owned kernel must remove `Si` materialization or have a verified register-P/PV mapping.
- Q LDS staging on top of the row-owned/`Si` scaffold is not viable due to LDS/VGPR/private-segment pressure.
- K-only LDS staging is not enough. K staging became worthwhile only when paired with V staging.
- `Bc=64` staging is currently worse because it loses the clean bank-conflict behavior and does not beat `(64/128,32,8)` when forced.
- The first wave-shuffle register-P/PV attempt failed correctness. Before retrying full attention, build a minimal WMMA-fragment mapping test that checks C-fragment row/column extraction into a PV A-fragment.
- The failed register-P/PV shuffle used `__shfl(fragS[row_ele], src_lane, ...)` with `row_ele` varying by destination lane. That is wrong because the source lane evaluates its own `row_ele`. The corrected pattern runs one shuffle per fixed `ele` and selects the desired candidate after the shuffle.
- The correct `(64,32,4)` register-P/PV retry is also rejected: it used VGPR `256`, private segment `0x38`, text size `0x447c`, and forced profile `16274.958 us` at `S=4096` despite `LDSBankConflict=0.0`.
- The QK partial-row-max probe is rejected: it avoided one softmax max-read pass but added `ds_bpermute` reductions in the hot QK path and regressed `S=4096` HIP to `18.416490` TFLOPS.

### Do-Not-Repeat (Unless New Preconditions)

- Do not re-add temporary sentinel configs for the staged path. Normal configs should map directly to real staged instantiations.
- Do not re-add standalone VT prepack unless a new forward dataflow changes the V access preconditions.
- Do not re-add row-owned skeletons that keep full `Si` materialization and worsen V access.
- Do not re-add Q LDS staging on the rejected row-owned/`Si` scaffold.
- Do not re-add K-only staging on the current dataflow.
- Do not re-add `Bc=64` staging unless a new LDS layout removes its conflicts and forced profile beats `Bc=32`.
- Do not re-add the `(64,32,4)` register-P/PV retry unless the live range/register plan changes enough to avoid private segment use.
- Do not re-add QK-produced partial row maxima unless the row maxima are effectively free as part of a larger no-`Si` dataflow.
- Do not add Q quantization before the forward kernel removes the score-LDS/softmax bottleneck.
- Do not copy CDNA MFMA-specific CK implementation details for gfx1151. Use CK only for transferable dataflow/layout ideas.
- Do not pursue int8 Q quantization (SageAttention-style) on gfx1151. The int8 WMMA instruction (`v_wmma_i32_16x16x16_iu8`) has the same 32-cycle latency as fp16 WMMA on gfx1151. The only benefit would be reduced K LDS/reregister pressure, which does not justify the added Q quantization complexity and per-block scale management given that K is already fp8 (same LDS size as int8).
- Do not use SageAttention's `global_load_tr_b128` pattern directly: the `__builtin_amdgcn_global_load_tr_b128_v8i16` intrinsic is gfx12xx-only (RDNA4). On gfx1151, transposed V loads must use software transposition.

## Next Experiments

### Experiment Dependency Graph

```
A13-A (current baseline)
  │
  ├── A16-A: Register Score/Softmax (remove Si LDS) ── HIGH PRIORITY
  │     │
  │     ├── (if passes + improves) A16-B: Q-in-Registers
  │     │     │
  │     │     └── A16-C: Integrated benchmark/tune
  │     │
  │     └── (if regresses) A16-D: Warp-parallel softmax (Si in LDS)
  │
  ├── A16-E: JIT V load (opportunistic, only if LDS occupancy is the limiter)
  │
  ├── A17-A: K double-buffering in D-loop ── independent, CK-inspired
  │     │
  │     └── A17-B: K/V LDS space sharing (pairs with A17-A)
  │
  ├── A17-C: sched_group_barrier hints ── independent, low risk
  │
  ├── A19-A: Streaming col-tile incremental softmax ── SageAttention-inspired
  │     │                                               (complements A16-A reg-P/PV)
  │     └── A19-C: 2Q architecture (major, depends on A19-A or A16-A)
  │
  └── A19-B: Lane-major K layout ── independent, SageAttention-inspired
```

### A16-A: Register-Resident Score/Softmax — Remove Si LDS

Goal: Eliminate the Si LDS materialization entirely. Keep the QK WMMA C-fragment in fp32 registers, run row-softmax reductions directly on it via warp shuffles, convert to P A-fragment in registers, and feed directly into PV WMMA.

Config: `(64, 32, 8)` — keeps each wave at 1 C-fragment + 4 fragO tiles, the same register-tile pressure as current baseline.

Why this avoids A14-B's spill trap: A14-B used `(64, 32, 4)`, which doubled per-wave tiles to 2 C-fragments + 8 fragO tiles, blowing VGPR to 256 with private segment use. With 8 waves, per-wave tile count stays at 1 C-fragment + 4 fragO, adding only the P A-fragment (~8 VGPR) and softmax state (~2 VGPR) over current baseline (187 VGPR). Target: stay under 216 allocated / spill-free.

Structural changes:

1. Fused QK-softmax-PV inner loop: Replace the three-phase `mul_A_BT → barrier → row_softmax → barrier → rescale_mul_add_A_B_reg` sequence with a single fused per-K-tile pass per wave:
   - Load K tile into LDS (unchanged)
   - Each wave computes its 16×16 QK C-fragment in fp32 registers
   - Row-max reduction: For the 8 rows owned by this wave (within its C-fragment), compute row maxima across the 16 columns using half-wave `__shfl_xor` (lane groups 0 and 1 exchange). This replaces the LDS-read sequential-max loop.
   - Online softmax update: Update per-wave `m_i[8]` (running max) and `l_i[8]` (running sum) in registers. Compute `alpha = exp2(m_old - m_new)`. Rescale the fragO tiles by alpha.
   - In-place exp on C-fragment: Apply `exp2(val - m_new)` elementwise on the C-fragment to get P in fp32 registers.
   - C→A fragment conversion: Use the validated `cfrag_value_at` pattern to build an fp16 A-fragment (P) from the fp32 C-fragment. Each wave converts 8 rows × 1 C-fragment — 8 `cfrag_value_at` calls per row × 16 columns = 128 shuffles per wave per K-iteration.
   - PV WMMA: Use the P A-fragment with V from LDS (current staging unchanged) to accumulate into fragO.
   - Barrier only after K/V LDS load: Down from 3 barriers per iteration to 1.

2. Remove Si LDS allocation: The `Si[Br * kSiStride]` LDS partition is no longer needed, reducing total LDS usage and potentially improving occupancy.

3. Per-wave m_i/l_i storage: `alpha[]` and `inv_l[]` LDS arrays replaced by per-wave register arrays `m_i[8]`, `l_i[8]`. Final normalization still needs a cross-wave combine (or store partial results via registers).

4. V LDS staging unchanged: V stays in LDS (`Vs[Bc * kKVLdsStride]`) using the proven `kKVLdsStride = D + 16` layout.

C-fragment to A-fragment conversion details (per wave, per K-iteration):
- The wave's C-fragment covers rows `r0..r0+7` of the `Br=64` query block
- For each owned row `r` (0..7), the 16 columns of P must be gathered:
  - For column `c` (0..15): source lane = `(r & 1) * 16 + c`, element index = `r >> 1`
  - `cfrag_value_at(fragC, r, c)` returns `__shfl(fragC[r>>1], src_lane)`
- Pack 16 values into `fp16x16_t` for the WMMA A-fragment
- Total: 8 rows × 16 columns = 128 `__shfl` per wave per K-iteration

Risk factors:
- 128 cross-lane shuffles per wave per K-iteration may become a hotspot (similar to how A15-A's `ds_bpermute` regressed). Mitigation: these are `__shfl` (VALU, single-cycle) not `ds_bpermute` (LDS, multi-cycle), and they replace the LDS Si read/write traffic of the baseline.
- The C→A conversion was validated in isolation (`test_wmma_fragment_layout.py`) but has not been stressed in a full attention hot loop.
- Per-wave m_i/l_i means the final `1/l_i` normalization must combine across waves. This can be done with a small LDS reduction at the end (8 waves × 8 rows = 64 elements) — much cheaper than the current per-iteration Si LDS traffic.

Expected win: Eliminates 3→1 barriers per K-iteration, removes all Si LDS traffic, eliminates `Br`-only-thread serialization, and drops LDS usage by `Br * kSiStride * sizeof(half_bits_t) ≈ 4.6 KB`.

Validation order:
1. `test_attn_hip.py` — must pass `2/2` configs with combined tolerance
2. `benchmark_attn_hip.py` — must not regress vs A13-A baseline
3. If benchmark regresses: profile (forced config) to check VGPR, scratch, LDSBankConflict, and instruction counts. Generate unbundled hsaco to count `__shfl`/`v_wmma`/`ds_load` counts.

### A16-B: Q-in-Registers (additive, lower priority)

Goal: Load Q[Br=64, D=128] once per workgroup into per-wave registers before the Tc loop, eliminating repeated Q DRAM reads in the K-loop.

Prerequisites: A16-A committed or source is stable enough to land this on top without merge conflict.

Changes:
- Before the Tc loop: each wave loads its Q tile subset from DRAM into a register array `q_reg[kQRegTiles]`. For `(64, 32, 8)`, each wave owns 8 rows × `D=128` → 8 × (`D/16`) = 8 × 8 = 64 WMMA A-fragment slots. Stored as `fp16x16_t q_reg[8][8]`.
- Inside the fused QK-softmax-PV inner loop (from A16-A): replace DRAM Q loads with reads from `q_reg`, indexed by `k0` (D-chunk iteration).
- This removes `Tc × Br × D` fp16 DRAM reads per workgroup.

Risk: Adds ~16 VGPR per thread for Q storage. With A16-A baseline, total VGPRs may approach the 216 boundary. If VGPR exceeds 216 (causing spill or occupancy drop), revert and keep Q streaming from DRAM.

### A16-C: Integrated Benchmark and Tuning

Goal: After A16-A and optionally A16-B are committed, run full benchmark and profile. Compare end-to-end and prepacked TFLOPS against AITER across all 4 target sizes. Tune waves_per_eu or config if needed.

Expected end-state:
- VGPR ≤ 216, scratch = 0, LDSBankConflict = 0.0
- S=4096 runtime approaching AITER's ~7.5 ms (from current ~8.3 ms)
- Remaining gap, if any, from Triton compiler-level WMMA scheduling — not from an LDS round-trip structural bottleneck.

### Fallback Experiments (if A16-A regresses)

A16-D: Si in LDS but warp-parallel softmax: Keep Si LDS but parallelize row softmax across all 256 threads instead of `tx < Br`. Each thread handles a column subset, reducing the serialization cost. Lowers the barrier from 3→2 if QK output can go straight to PV. Lower risk than A16-A but smaller win.

A16-E: V just-in-time load (eliminate V pre-staging): Load V from DRAM only after P is ready (like AITER PRE_LOAD_V=False). Reduces LDS usage by `Bc * kKVLdsStride * sizeof(fp8)` ≈ 4.6 KB, potentially improving occupancy. Only worth pursuing if A16-A/B already reduce LDS enough that V LDS becomes the occupancy limiter.

### A17-A: K Double-Buffering in D-Loop

Goal: Software-pipeline K loads from DRAM while the QK WMMA computes on the previous K0 chunk, overlapping global memory latency with matrix compute.

Inspiration: CK's QR pipeline (lines 632-665) loads `k_block_tile[i+1]` while computing GEMM0 on K0 chunk `i`. On gfx1151 this uses sync loads (not async copy) but the interleaving still hides DRAM latency.

Changes:
- Split the K tile load into D-chunk-sized pieces (k0 = D/16 = 8 chunks for D=128).
- Before the QK loop: load K0[0] into LDS buffer A.
- Loop body: start computing QK on buffer A, simultaneously load K0[i+1] into buffer B. Alternate buffers (ping-pong).
- Requires 2× K LDS space (double-buffered) OR K/V sharing to reuse V space.

Risk: Barriers between QK and K-load may serialize rather than overlap if the compiler doesn't schedule well. The existing baseline has no such interleaving; this adds complexity but is independent of the register score/softmax path.

Config: `(64, 32, 8)`. K LDS grows by factor of 2 (current Ks = `Bc * kKVLdsStride` = 32 * 144 = 4608 bytes → 9216 bytes). If combined with K/V sharing (A17-B), the increase is absorbed.

Validation: Same protocol. Profile to verify that double-buffered K loads reduce `global_load` waitcnt stalls in the hot QK loop.

### A17-B: K/V LDS Space Sharing

Goal: Allocate LDS as `max(sizeof(Ks), sizeof(Vs))` instead of `sizeof(Ks) + sizeof(Vs)`, nearly halving the K/V staging LDS.

Inspiration: CK's policy returns `max(SingleKSize, SingleVSize)` for single-element LDS space (policy lines 548-595). Our kernel pre-allocates both Ks and Vs separately (sram_sz includes `2 * Bc * kKVLdsStride * sizeof(fp8e5m2_t)`).

Changes:
- K and V are never live simultaneously in the current dataflow: V is used only after QK softmax, when K is no longer needed.
- Allocate one buffer for both, sized to the larger of K and V. For our config `Bc=32, D=128`: Ks needs 32 × 144 = 4608 bytes, Vs needs the same → allocate 4608 bytes total instead of 9216.
- After QK gemm + softmax completes, load V over the same LDS region that held K.
- Total sram_sz drops from ~14 KB to ~9.5 KB, potentially improving occupancy.

Risk: Requires a barrier between last K read and first V write to the shared buffer. Our current kernel already has this barrier (after QK softmax), so no new sync is needed.

Dependency: Pairs naturally with A17-A (K double-buffering), which would otherwise double K LDS.

Validation: Same protocol. Check `LDSBankConflict=0.0` persists with the new shared layout.

### A17-C: sched_group_barrier Compute/Load Hints

Goal: Use `__builtin_amdgcn_sched_group_barrier` to tell the hardware scheduler to interleave LDS reads with WMMA operations in the hot QK loop.

Inspiration: CK's QR pipeline (lines 492-506) uses:
```cpp
__builtin_amdgcn_sched_group_barrier(DS_READ, 2, 0); // allow up to 2 DS reads
__builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);    // then up to 2 MFMA/WMMA
__builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0);
__builtin_amdgcn_sched_group_barrier(MFMA, 4, 0);
```

Explanation: By default, the hardware scheduler may serialize all LDS reads before any WMMA, or vice versa. `sched_group_barrier` defines the maximum number of instructions from a given class that can be issued before allowing the other class to issue. This optimizes compute-load overlap at the instruction-issue level.

Changes (add to `mul_A_BT` hot inner loop, gfx1151 only):
```cpp
#if defined(__gfx1151__)
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS_READ
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // WMMA
#endif
```
- `0x100` = DS_READ barrier (LDS loads)
- `0x008` = MFMA/WMMA barrier

Risk: Very low. These are scheduler hints, not correctness barriers. They affect instruction issue ordering but not data dependencies. The `waitcnt` instructions still enforce data hazards.

Validation: Benchmark only; correctness is not affected. May show a small improvement from better WMMA/LDS interleaving on gfx1151.

### A19-A: Streaming Col-Tile Incremental Softmax + PV

Goal: Compute score, softmax, and PV accumulation incrementally in groups of col-tiles instead of computing all score columns before any softmax. Reduces score register pressure and enables 2Q mode later.

Inspiration: SageAttention's 2Q kernel with `StreamColTiles` computes scores in groups of 2 col-tiles, applies softmax, and does PV accumulation before moving to the next group. This limits the live score cache to `StreamCols` groups instead of all `ColTiles` groups.

How it differs from our current approach: Currently we compute all QK scores for a KV tile (writes to Si LDS), then apply softmax to all scores, then accumulate all PV. With streaming: compute QK for col-tiles [0..1], apply online softmax (update m/l), accumulate PV for col-tiles [0..1], then repeat for col-tiles [2..3], etc.

For Bc=32 (2 col-tiles of 16): Streaming 1 group of 2 col-tiles is equivalent to computing all at once. The benefit appears only with larger Bc (e.g., Bc=64 → 4 col-tiles, stream 2+2) or in 2Q mode where score caches exist per Q-group.

Relevance to A16-A: If register-P/PV with `(64,32,8)` works without streaming (only 2 col-tiles for Bc=32, so 2 score caches of 8 fp32 = 16 VGPR), streaming is not immediately needed. But if we later increase Br or Bc, streaming becomes critical for register pressure.

Changes (on top of register-P/PV):
- Refactor per-K-tile loop: split into outer `col_tile` loop (groups of `StreamCols` columns) and inner `d_tile` loop (chunks of D/16).
- Per group: compute QK for `StreamCols` col-tiles, apply online softmax, rescale O accumulator, compute PV for `StreamCols` col-tiles.
- `StreamCols=2` for Bc=32 (whole tile in one group, no streaming needed). `StreamCols=2` for Bc=64 (two groups of 2).

Risk: If `StreamCols` is larger than 1, need to handle K reload or K buffering per col-tile group (each group needs its K columns). For fp8 K: K LDS already has all columns, so just index into different column ranges per group — no extra loads.

Validation: Same protocol. For Bc=32, this is effectively a no-op (only 1 group). Test benefit with Bc=64 if register-P/PV path exists for it.

### A19-B: Lane-Major K Layout (Zero-Decode K Load)

Goal: Eliminate the `fp8e5m2x4_to_half2x2` + `__builtin_amdgcn_perm` fp8→fp16 decode on every K LDS read by storing K in WMMA B-fragment lane order during quantization. The decode moves from the hot attention loop to the quantization kernel (O(BHSD), done once).

Inspiration: SageAttention's lane-major K layout stores K data in the exact per-lane, per-d_tile order that WMMA B-fragment loads consume. The load function `pack_k_i8_wmma_b_regs_from_lane_major_shared` reads directly without any perm/decode.

Current state: Our quantization kernel writes fp8 K/V in native row-major order. The attention kernel's `load_e5m2x16_as_fp16` decodes 4 fp8 values → 2 fp16 values × 4 iterations = 1 fp16x16 B-fragment per call. This decode runs in the hot QK loop.

Proposed change:
- In `quantize_kv_e5m2_kernel` or a new lane-major packing pass: reorganize K from `[N, D]` to `[ColTiles, DTiles, NPerTile, Lanes, KPack]` where:
  - ColTile = N / 16 (2 for Bc=32)
  - DTile = D / 16 (8 for D=128)
  - NPerTile = 16 (rows per WMMA tile)
  - Lanes = 16 (lanes in WMMA N-dimension)
  - KPack = 4 (contiguous fp8 elements per lane)
- In the attention kernel: replace `load_e5m2x16_as_fp16` with direct load from lane-major LDS using simple pointer arithmetic (no perm needed).

Tradeoff: Quantization kernel becomes slightly more complex (needs to write K in lane-major order), but the hot QK loop becomes simpler. The total fp8 data is the same (K stays fp8), only the storage order changes.

Risk: Moderate. Lane-major layout changes the quantization output format. Must ensure the V layout is compatible (V can stay row-major for now since the V decode is separate from K decode). The quantization kernel already runs once per forward pass, so adding lane-major reordering there is cheap.

Validation: `test_attn_hip.py` must still pass. Profile to verify reduced instruction count in hot QK loop (fewer `v_perm_b32` / `__builtin_amdgcn_perm`).

### A19-C: 2-Query-Groups Per Wave (2Q Architecture)

Goal: Process two 16-row query groups per wave instead of one. Doubles query throughput for the same K/V LDS bandwidth. Equivalent to moving from Br=64 with 8 waves (each wave handles 8 rows) to Br=128 with 8 waves (each wave handles 16 rows × 2 groups = 32 rows).

Inspiration: SageAttention's 2Q kernel (`qk_int8_sv_f16_d64_native_2q_kernel`) has `QGroups=2`, processing 2 sets of Q rows per wave. K/V tiles are loaded once and shared across both Q groups.

Precondition: Requires register-P/PV (A16-A) or streaming softmax (A19-A) to keep score caches small enough for 2× copies per wave.

Changes:
- Double per-wave state: `q_regs[2][DTiles]`, `out_frag[2][DTiles]`, `m[2]`, `l[2]`, `score_cache[2][ColTiles]`
- In inner loop: for each K col-tile group, compute QK for Q-group 0, then for Q-group 1 (or interleave). Share K LDS reads.
- For PV: accumulate `out_frag[0]` and `out_frag[1]` independently, each using its own P and the shared V LDS.
- Grid: Br=128, Tc = n_kv / Bc, grid = (b, h, Tr) where Tr = ceil(n / 128). Each workgroup processes 128 query rows.

Register budget (critical): For Br=128, D=128, Bc=32, 8 waves, QGroups=2:
- kRegTiles per group: (128×128)/(16×16)/8 = 8 tiles → per wave × 2 groups = 16 tiles × 8 fp32 = 128 VGPR for output alone + Q registers + score caches. This WILL exceed register budget.
- Mitigation: use streaming col-tile softmax (A19-A) to keep score caches small, and/or reduce to Br=64 with QGroups=2 (effectively same as Br=128 but with per-wave doubling).
- Better: Test Br=64, QGroups=2 first (8 waves, each processes 8 rows × 2 = 16 rows per wave, 128 total rows). This is the sage path: keep the total query rows per workgroup at 128 but increase per-wave utilization.

Risk: HIGH. Register pressure is the primary concern. 2Q mode doubles the per-wave Q/score/out state. Only viable with aggressive register management (streaming softmax, Q-in-registers, etc). This is the most transformative but also most complex experiment.

Validation: Same protocol. If VGPR blows past 256 or spills, reject and document register count.

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
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py` - AITER FlashAttention forward kernel source
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py` - AITER GPU arch detection and config selection (RDNA vs CDNA paths)
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py` - AITER shared helpers (fp8 cast, rotary, ALiBi)
- `~/triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp` - Triton AMD WMMA lowering (C-fragment layout, intrinsic emission)
- `~/triton/third_party/amd/lib/TritonAMDGPUTransforms/WmmaGroup.cpp` - Triton WMMA group intrinsic database
- `~/triton/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp` - Triton AMD LDS lowering with swizzle/padded layout support
- `~/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` - Triton linear layout to thread/warp/register mapping
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp` - CK-Tile QR FMHA dataflow reference
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp` - CK-Tile Q-load, K/V LDS, and V-shuffle policy reference
- `~/rocm-libraries/projects/composablekernel/dispatcher/codegen/fmha/fmha_arch_specs.json` - CK FMHA tile/wave/warp config reference
- `~/rocm-libraries/shared/rocroller/docs/src/LDSSwizzling.md` - LDS swizzle reference for bank-conflict reasoning
- `~/SageAttention/sageattention/triton/quant_per_block.py` - SageAttention per-block int8 quantization reference
- `~/SageAttention/sageattention/triton/quant_per_thread.py` - SageAttention per-thread int8 quantization reference
- `~/SageAttention/sageattention/triton/attn_qk_int8_per_block.py` - SageAttention Triton int8-QK attention reference
- `~/SageAttention/sageattention/core.py` - SageAttention public API, layout, smoothing, and tolerance context
