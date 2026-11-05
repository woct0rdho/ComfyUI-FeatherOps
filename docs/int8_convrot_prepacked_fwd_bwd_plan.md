# Int8 ConvRot Prepacked Forward/Backward Kernels on gfx1151

Scope: the ConvRot int8 weight format that ComfyUI ships (`weight` int8 `[N, K]` in the rotated basis, `weight_scale` f32 per output channel, group 256), one prepacked layout that serves both GEMMs a training step needs, and the two HIP kernels that consume it. The forward is `Y = X x W_rot^T`. The backward is the input gradient `dX = (dY x W_rot) x R^T` (no weight gradient: the base model is frozen, `W` is a constant). Implementation lives in `~/feather-ops` (`csrc/hip_kernel_int8.cu`, `feather_ops/int8_convrot.py`). This document is maintained here next to the fp8 documents.

The forward baseline is the shipped comfy-kitchen gfx1151 int8 kernel with its shape heuristic (`int8_convrot_fwd_plan.md`): 37-42 TFLOPS on the big cells, 29 TFLOPS on Anima's `4096x2048x2048`. It is at 79% of the pure MMA-issue time with every memory and sync cost ablated, so it cannot be beaten by a memory-side change. The only headroom is the fp8-vs-int8 operand width (an int8 fragment is 16 B per lane where the fp8 kernel has to widen 16 fp8 bytes to 16 bf16 in registers) and the per-shape schedule. There is no backward baseline: sd-scripts dequantizes the weight per step (226 ms/step at 1024x1024) or materialises `wq.t().contiguous()` per call (1.18 ms at 2048^2, 14.8 ms at 8192x2048), so the bar to beat is the bf16 GEMM it replaces (17-28 TFLOPS on these shapes).

## Why the layout question exists

On gfx11 a `v_wmma_i32_16x16x16_iu8` fragment is the whole 16-byte contraction step of one lane's row: lane `L` of A holds `A[row = L % 16][k0 : k0+16]`, lane `L` of B holds `B[k0 : k0+16][col = L % 16]`, and the two half-waves are duplicated. The contraction axis is therefore always the fragment's contiguous axis, and it differs between the two GEMMs:

| | contraction | weight fragment needed | natural weight layout |
|---|---|---|---|
| forward | `K` (= weight's input dim) | 16 contiguous `k` of one `n` | `[N, K]`, `k` contiguous (today's file) |
| backward | `N` (= weight's output dim) | 16 contiguous `n` of one `k` | `[K, N]`, `n` contiguous (the transpose) |

One copy cannot be contiguous along both axes, and every alternative has a measured or structural cost:
- a second transposed copy: exactly the VRAM the quantization buys (1.4 GB for Anima, 8.8 GB for Klein 9B).
- `wq.t().contiguous()` per step: 1.2-14.9 ms per call (torchao's int8 training pays the same and leaves "TODO: check if transpose+quantize are actually fused" in `int8_mixed_precision.py`).
- byte gathers at fragment size: 16 x `ds_read_u8` per fragment, i.e. 16x the LDS instruction count of the hot loop (LDS reads can never be cheaper than one 16 B instruction per lane).
- ggml/GGTensile solve it by decoding into an LDS image in the reader's orientation, which is free for them because a q4/q8 block has to be decoded value by value anyway - int8 ConvRot has no decode step to hide the transpose in.

## The prepacked layout

A 4x4 byte micro-tile, `k` contiguous inside the tile:

```
P[(k // 4), (n // 4), (n % 4) * 4 + (k % 4)]     # 16 B per tile, K * N bytes total
```

- The forward's fragment (16 `k` of one `n`) is four dwords from four consecutive `k4` tiles. The backward's fragment (16 `n` of one `k`) is four dwords from four consecutive `n4` tiles. Both directions get 4-byte-granular access, one direction as four contiguous dwords of a 16 B chunk, the other as four byte-`i` gathers that cost 3 x `v_perm_b32` on the staging path (see below) rather than on the fragment path.
- Consecutive `n4` tiles are contiguous, so one `k4` plane is a `4 x N` contiguous span: a warp reading 32 consecutive `n4` of one `k4` issues one fully coalesced 512 B request, in both directions (the forward loops `k4` outer / `n4` across lanes, the backward `n4` outer / `k4` across lanes).
- It is a pure permutation of the bytes in the released checkpoints: no requantization, no scale change, `int8 x int8 -> int32` accumulation is exact so the products are bit-identical to the `[N, K]` layout, and `unpack` restores the released form exactly for export. The per-output-channel scale vector is untouched.
- 16 B per tile means an LDS store of any 4 consecutive tiles is a 64 B span, so the staging `b128` load / `b32` store pattern has no sub-dword work in the direction that keeps `k` contiguous, and 3 perms per 16 B chunk in the other one.

Rejected variants: the fp8 documents' `(K/16, N, 16)` (16 contiguous `k` per `n`, forward-free) stops the backward from doing anything but 16 single-byte LDS loads per fragment, i.e. it would need a second copy. A `(N/16, K, 16)` mirrored form has the same problem on the forward side. An 8x8 micro-tile needs one axis to be `stride 8` inside the tile, which no LDS instruction can use. Per-`k`-group scales (AWQ/GPTQ style) and 4-bit operands would change the quantization, so they are out of scope by construction.

## Kernels

One templated kernel, two instantiations (`kTransposedB` = forward / backward), because the GEMM structure is identical and only the roles of the weight's axes, the B staging and the epilogue differ:

| | forward | backward |
|---|---|---|
| `A` | `X [M, K]` int8, `k` contiguous | `dY [M, N]` int8, `n` contiguous |
| `B` prepacked | `P`, fragment = 4 `k4` tiles | `P`, fragment = 4 `n4` tiles |
| LDS B rows | `[n][chunk_k]`, 16 B rows | `[k][chunk_k]`, 16 B rows |
| epilogue | `x a_scale[m] x b_scale[n] (+ bias[n])` | `x a_scale[m]` |
| output | `[M, N]`, `n` contiguous | `[M, K]`, `k` contiguous |

- A: the fp8 kernel's `[stage][m][16]` LDS rows (16 B = one K-step), 128-bit loads and stores, one `b128` fragment read per lane. int8 removes the fp8 path's LDS -> VGPR widening entirely, so a fragment is a raw `v4i`.
- B: staged with 128-bit global loads and four `b32` LDS stores per chunk, into the fragment-native row layout. The backward's byte-`i` gather is 3 `v_perm_b32` per chunk (`t01 = perm(d0,d1)`, `t23 = perm(d2,d3)`, `out_i = perm(t01,t23)`), which is ~1 instruction per 4 bytes staged and amortised over the M-repeat factor and the K-tile's reuse.
- Inner loop: `kRepeatN` B fragments loaded once per M-repeat, then `kRepeatM` A fragments, 8 independent accumulators per warp (the gfx11 rule: `D == C` is legal but stalls, so >= 8 independent accumulators).
- Epilogue: the fp8 kernel's C-shuffle (accumulator -> LDS in the gfx11 interleaved row order -> row-major 16 B global stores), with the scale in float and an RNE bf16 conversion. Partial M/N tiles are masked on the store only. Staging clamps indices so no read is out of range.
- Pipelining: two LDS buffers plus one register image of the chunk being fetched (`int4`/`uint4` per thread, ~12 VGPRs). The next chunk's global reads issue before this chunk's WMMAs and land in the other buffer, so there is one barrier per chunk instead of two, and only the buffer that nothing has read since the previous barrier is written. `s_setprio 1` covers the read region. Deeper staging was not needed: gfx1151 has no `global_load_lds`, so a third stage would cost VGPRs and LDS without removing the barrier that is left. The two halves of this design have since been priced separately on the fp8 sibling (+7.7-23% from the overlap, -28% from the second buffer where LDS binds), so a single-buffer + register prefetch form is an open item below. Here the second buffer buys one barrier per chunk at 12-16 KB of the workgroups' LDS budget, and it is only free where VGPRs cap occupancy.
- Grid: output-dim-major (`blockIdx.x` = output tile) so all output tiles of one M tile run together and the A panel stays in L2.
- Dispatch: the schedule comes in as five integers from `select_int8_config`. The kernel's only shape decision is the tail split, which is a property of `M` and the tile rather than tuned data.

## Schedule table

Configs are `(warps_m, warps_n, unroll_k, repeat_m, repeat_n)`, i.e. `block_m = 16 * warps_m * repeat_m`, `block_n = 16 * warps_n * repeat_n`, `chunk_k = 16 * unroll_k`, threads = `32 * warps_m * warps_n`. The table is measured per shape (kernel-only, interleaved configs inside one process, median of per-round medians per `~/evotensile/docs/noisy_measurements.md`) and reduced to a heuristic keyed on `(M, N, K, dir)`, since M is the token count and so varies with the image size:

| condition | schedule | why |
|---|---|---|
| `M <= 16` | 16x64, 64 threads | a row or two: the block count is all that matters, and 16 rows gives 4x the blocks of 64 (+4 to +21% on the four one-row cells, +15-20% on the 16-row cells in the gradient direction) |
| `M <= 64` | 64x64, 128 threads | prefix passes and single-row cells are DRAM bound. A bigger tile only adds wasted MMA work, and a 16-row tile loses to 64x64 at 32-64 rows |
| `M <= 128` | 64x64 up to output 8192, else 128x128 | one 64x64 launch beats a 128-row launch plus its 64-row tail launch (two single-wave grids). Past ~8192 the output is wide enough for the 128x128 tile's N panel to win |
| `M <= 256` | 64x64 up to output 4096, else 128x128 | same, with the crossover at half the width (`M = 192-256`. 1.35-1.6x at `out = 2560-4096`). The `out > 4096` side stays 128x128: `192 x 12288 x 4096` gains 6% on 64x64 but `256 x 12288 x 4096` in the gradient direction loses 30% |
| `M <= 512` | 64x64 below output 8192, 128x128 from 8192, 256x256 when the output is >= 16384 or the reduction >= 8192 | four 128-row tiles is enough height, so the output width decides. Below 8192 the narrow tile wins (`512 x 2048 x 1024` +29%, `512 x 4096 x 4096` +4%), from 8192 the 128x128 panel (+9% at `8192 x 4096`, +5-11% at `12288 x 4096`), and a long or enormous reduction puts the weight traffic back in charge (`512 x 4096 x 12288` +4%, `512 x 248320 x 4096` +8%) |
| output `<= 1280`, `M >= 1024` | 256x128, 256 threads | a 256-wide tile is mostly tail when the output is 1152-1280 wide. The `M` guard keeps the block count above the CU count |
| output `>= 6144` | 256x256, 512 threads | halves the number of blocks and of A panels |
| backward, output `>= 3072`, reduction `>= 8192` | 256x256, 512 threads | the input gradient's dY panel then feeds 16 output tiles instead of 8, and at `M = 512` it is 22-29% faster than 128x128 |
| output `< 2048` with a reduction `>= 4096` (forward) | 128x256, 256 threads | keeps twice the A panel per block over a long K loop |
| else | 128x128, 256 threads | more blocks per cell, so the tail blocks spread over 40 CUs |

The rules are ranges on `(M, output width, reduction width)` rather than fitted shapes, so a new image size (which moves `M`) lands in the same class as the cells it was tuned on, and so does a model whose weight shapes are between the measured families.

## Shapes

One cell per `(N, K)` family from `int8_convrot_fwd_plan.md`, forward and backward, at the M values each model actually runs:

| model | (N, K) families | forward M | backward (N, K) swapped |
|---|---|---|---|
| Krea 2 | (6144,6144) (1536,6144) (16384,6144) (6144,16384) | 4112, 9232 | same set |
| Qwen-Image 2.1 | (4096,4096) (24576,4096) (4096,12288) | 4112, 9232 | same set |
| Z-Image | (11520,3840) (3840,3840) (10240,3840) (3840,10240) | 4128, 9248 | same set |
| H3 | (21504,5376) (5376,7168) (28672,5376) (5376,14336) | 5376, 25344 | same set |
| MOSS-Image | (9216,3072) (3072,3072) (16384,3072) (3072,8192) | 4112, 9232 | same set |
| Anima | (2048,2048) (2048,1024) (8192,2048) (2048,8192) | 512, 4096, 9216 | same set |
| Klein 9B | (12288,4096) (4096,4096) (24576,4096) (4096,12288) (36864,4096) (4096,16384) | 512, 4096, 4608, 9216, 9728 | same set |
| Qwen3.5-9B TE | (12288,4096) (4096,4096) (8192,4096) (4096,12288) (1152,1152) (3456,1152) (4304,1152) (1152,4304) (248320,4096) | 512-4096 | same set |
| Qwen3-VL-32B TE | (25600,5120) (1024,5120) (5120,25600) (5120,8192) (8192,5120) | 512-4096 | same set |

The backward swaps the weight's axes, so its cell set is the same families with `(N, K) -> (K, N)` and a different contraction (`N` instead of `K`). The tuning grid therefore only needs one entry per `(contraction, output)` pair and both directions of each family are covered.

### Training coverage audit

The table above is a tuning set, not the full training cell set: it lists two M per model and only the token-width families. Enumerating the quantized Linears from the model code (musubi-tuner's Krea 2 and MiniMax-H3 plus the H3 text encoder, with the same target/exclude keys and group-size rule the loaders use) gives:

| model | quantized Linears | families `(N, K)` | fwd calls | dgrad calls |
|---|---:|---|---:|---:|
| Krea 2 | 224 | `(6144,6144)x3 (1536,6144)x2 (16384,6144)x2 (6144,16384)x1` | 8/block | 8/block (block 0's base Linears skip it when the latent does not require grad) |
| H3 full INT8 | 250 | `(21504,5376) (5376,7168) (28672,5376) (5376,14336)` | 4/block | 4/block |
| | | `(96768, 2688)` = `adaln_proj.linear`, group 64 | 1/block, `M` = batch (=1) | none |
| H3 pruned / `--prune_adaln` | 200 | adaln becomes `[96768, 8]` (curve table) or `[96768, 9]` (rank-8 basis): `K` divides neither 64 nor 256, so it stays BF16 | - | - |
| H3 Qwen3-VL-32B text encoder | 350 | `(1024,5120)x100 (25600,5120)x100 (8192,5120)x50 (5120,25600)x50 (5120,8192)x50` | forward only | none |

- One family is missing from the tables: H3's unpruned `adaln_proj` (`96768 x 2688`, group 64), whose activation is the timestep embedding. Its forward cell is `(M=1, out=96768, contraction=2688)`. Its input gradient never runs in training (the timestep tensor does not require grad), so `(1, 2688, 96768)` is a benchmark-only cell. It is the widest int8 output of any model and is pure weight streaming (260 MB per call, 50 calls/step ~ 13 GB/step). `M <= 128` selects the 64x64 tile, which is already at the DRAM roof. Keep it in the cell matrix so a schedule change cannot move it to a 256-row tile.
- M distribution: the table lists two M per model, but training is bucketed and the integrations differ.
  - musubi Krea 2 pads the token sequence to a multiple of 256 before the blocks (`padlen = (-full) % 256`), so `M` in `{256 k}` (1024^2 -> 4352, 1536^2 -> 9472, 768^2 -> 2560) and the ragged 4112/9232 cells never occur there. The tail-aware launch is idle and only the exact-M cells matter.
  - H3's packed sequence is video + audio + text rows with no alignment, so ragged M is the norm and the tail launch matters.
  - The H3 text encoder runs at the prompt token count (arbitrary, ~5..4096), so short prompts hit the `M <= 128` class. It is forward-only.
- Direction counts differ from the family symmetry: 8 fwd + 8 dgrad per Krea 2 block, 5 fwd + 4 dgrad per H3 block, and the producer widths differ (`K` for the forward, `N` for the gradient).
- The cells to add to `tests/test_int8_convrot.py` and `bench/benchmark_int8_convrot.py`: `(1, 96768, 2688)` (group 64), the 256-aligned Krea 2 M set, ragged-M rows for H3 and the text encoder, and the TE's short-prompt rows. They are in the bench now (`--groups h3-adaln,krea2-musubi,h3-ragged,text-enc-short`) and measure as expected: adaln forward 1.34 ms = 194 GB/s (weight streaming at the DRAM roof. Its gradient is 2.24 ms = 116 GB/s because a 64x64 tile over 2688 columns launches only 42 blocks, and it never runs in training), the text encoder's 32-row prompts are bandwidth bound at 4-11 TFLOPS, Krea 2's 256-aligned rows (2560/4352/9472) run at 37-39 TFLOPS, and a ragged 5251-row H3 cell runs at 38.0 (the tail launch working).

## Numerics

- int8 x int8 -> int32 is exact, so the prepacked layout, the tile order, the K-loop order and a future split-K cannot change a product. The only float work is `acc * a_scale * b_scale (+ bias)` and the RNE store, exactly as the current kernels do it.
- The backward folds `b_scale[n]` (the weight's per-output-channel scale) into the row-wise quantization of `dY`, because the contraction axis is `n` and a per-`n` factor cannot be pulled out of the sum. That is the same identity torchao uses ("row-wise scaling for B.T is column-wise scaling for B"), it is what sd-scripts' current int8 backward already does, and it only changes which rounding the gradient quantizer performs.
- The backward's `x R^T` (block-diagonal Hadamard, group 256) is the consumer's basis change: `dX` feeds the residual add, the LayerNorm backward and the LoRA branch, all in the original basis, so it cannot be deferred to the next layer's GEMM. It starts as the existing separate rotation pass and becomes an epilogue stage (accumulate a full 256-wide group per CTA, `H` in LDS in 64-column chunks, fp32 rotation before the bf16 store) once the GEMM itself is at target. That removes a bf16 round trip and is the only place where the fused form can be more accurate, not less.

## Where the remaining time goes

Resources of the shipped schedules. Every tile is covered, not just the two the profiler run happened to use (Anima's `4096x2048x2048` and Krea 2's wide `4112x16384x6144`):

| schedule | warps | VGPR/wave | LDS/workgroup | waves/SIMD | limit |
|---|---:|---:|---:|---:|---|
| 64x64 `(2,2,2,2,2)` | 4 | 76 | 8.3 KB | 15 (93%) | LDS |
| 128x128 `(2,4,2,4,2)` | 8 | 118 | 16.5 KB | 12 (75%) | LDS |
| 128x256 `(2,4,2,4,4)` | 8 | 204 | 24.7 KB | 7 (43%) | VGPR |
| 256x128 `(4,2,2,4,4)` | 8 | 192 | 24.7 KB | 8 (50%) | VGPR |
| 256x256 `(4,4,2,4,4)` | 16 | 188 | 32.9 KB | 8 (50%) | VGPR |

Read from the built code objects (`hipFuncGetAttributes`, `~/tmp/feather_ops/dual/int8_attrs.cpp`) and converted with the model in `gfx1151_reference.md` (wave32 VGPR blocks of 24, 131072 B of LDS per WGP, 16 waves per SIMD). The earlier version of this table quoted two rows from the profiler (184/120 VGPR, 32/16 KB) and no limit column. The artifact reports 188/118 for those two, and the difference is rounding between the two sources. Every 128/256-row tile is VGPR-capped, not LDS-capped, so freeing LDS does nothing for them, and the 64x64 and 128x128 tiles are the mirror case (LDS-capped at 76-118 VGPR, which is where a single-buffer staging would pay).

Stochastic PC sampling (cycle interval) on those two cells, share of samples per stall reason:

| cell | top reasons |
|---|---|
| `4112x16384x6144`, 256x256 | `ARBITER_NOT_WIN` 76%, `ARBITER_WIN_EX_STALL` 17%, `WAITCNT` 6%, `ALU_DEPENDENCY` 0.7% |
| `4096x2048x2048`, 128x128 | `ARBITER_NOT_WIN` 60%, `ALU_DEPENDENCY` 18-19%, `BARRIER_WAIT` 13.5-14%, `WAITCNT` 3% |

The wide cell is issue-saturated with WMMA: another wave usually owns the issue slot, and the remaining 17% is the pipeline hazard that the WMMA issue rate itself creates. There is nothing left to win there from memory-side work, which matches the ablation CK published for their kernel (79% of pure MMA time with all memory and sync removed). The small cells are a different regime: barrier waits and dependent LDS chains are ~30% of the samples, which is where a staging change still pays.

The per-cell numbers localize the biggest single loss, and it is not in the kernel body at all: every cell whose M is an exact multiple of the tile height already runs at ~40 TFLOPS (H3 `5376`, Anima `4096`/`9216`), while every `M = 4112` cell sits at 36.6-36.9 in both directions. `4112 = 16 x 257`, so a 256-row tile computes `4096 + 240` rows of dead MMA work, i.e. 5.8% of those cells is tail padding: 36.9 x 1.058 = 39.0, which is exactly an exact-M cell. Anima's own sizes (1024x1024 -> 4096 rows, 1536x1536 -> 9216) are tile-aligned, but bucketed training is not, so this applies to the shapes a training run actually produces.

Of the three structural items this diagnosis produced, two shipped and one is still open:
- Shipped: tail-aware launch. Full M tiles run with the chosen schedule, then the `M % block_m` rows get their own launch with a 64-row tile, so the dead MMA work disappears. The cost is one extra pass over the weight for those rows (`N*K` bytes: 0.16 ms at `6144^2`, 0.42 ms at `16384x6144`) against 5.8% of the cell, i.e. ~+4% net on the 4112-row cells. Exact-M cells take a single launch and are unchanged.
- Shipped: the backward's B staging now walks the layout's contiguous axis (see below), which closed the forward/backward gap.
- Still open: both directions read A one 16-byte row per lane, 32 lines per instruction, because a chunk carries only 32 bytes of K per row. A tiled activation layout makes the same instruction 4 lines and, with `chunk_k = 128`, quarters the barrier count. That is now the top open item because the producer is ours.

The two staging problems, and why the backward one was worth 3-8% on its own:
- The backward's B staging walked the wrong axis. The prepacked layout's contiguous axis is the weight's output axis. The staging loop assigns lanes to the output index, which is contiguous for the forward (`stride_out = 1`, one 512 B request per warp instruction) but for the backward means a stride of `4 x N` bytes, i.e. 32 separate cache lines per instruction. The backward measures 4-10% below the forward on identical shapes (Krea 2 `16384x6144`: 36.6 forward vs 33.2 backward) even though it does less epilogue work.
- The A side is the mirror image, and the lane remap that works with the current row-major activation (2 lanes per row at `chunk_k = 32`) measured inside noise, so it waits for the tiled layout.

What the same evidence rules out: stream-K / persistent CTAs (the wide cell has ~110 waves/SIMD and the small cell ~26, and the small cell is not slower per flop, so block quantization is already smoothed by the backfill). `unroll_k = 1` for occupancy (both schedules are VGPR-capped, not LDS-capped). A third LDS stage (the 16-warp tile has 8 VGPRs of headroom, and the guard-free-loop experiment spilled to half throughput), and any instruction-level shortcut (gfx1151 has only `V_WMMA_I32_16X16X16_IU8`, no `16X16X32` variant).

The producer had its own version of the occupancy problem, and it was the largest single loss outside the GEMMs. Its launch budget was a hardcoded 32 KB of total footprint, so `waves = (32768 - K*2) / 1024` collapsed to zero for `K >= 16384` and was clamped to one: 32 threads and 33 KB of LDS per block, i.e. under one resident wave per SIMD, at 43 GB/s. That is exactly the input-gradient width of Krea 2's and Klein 9B's `mlp.gate`/`up` (`N_w = 16384`, two calls per block), and H3's `K_w = 14336` was already part-way down the slope (4 waves, 138 GB/s). The budget now comes from the 64 KB per-workgroup limit (minus the static wave-maxima array), and rows that no longer leave room for a full block of waves (`K > 16384`, up to the 248320-wide vocab projection of the 32B text encoder) drop the bf16 row buffer and recompute each group from global memory in the quantize pass, so every width in the shape tables works. Measured at `M = 4112`, gradient producer (no rotation):

| K | before | after |
|---:|---:|---:|
| 6144 | 0.45 ms (167 GB/s) | 0.37 ms (205 GB/s) |
| 8192 | 0.45 ms (223 GB/s) | 0.54 ms (189 GB/s) |
| 12288 | 0.73 ms (208 GB/s) | 0.86 ms (177 GB/s) |
| 14336 | 1.28 ms (138 GB/s) | 0.90 ms (197 GB/s) |
| 16384 | 4.66 ms (43 GB/s) | 1.16 ms (175 GB/s) |
| 21504-36864 | launch failed (`smem > 64 KB`) | streamed, 144-171 GB/s |
| 248320 | launch failed | streamed, 153 GB/s |

The Krea 2 gradient cell `4112x16384x6144` goes from 26.6 to 23.3 ms (its producer 4.66 -> 1.16 ms), and an int8 whole-DiT step on the kroma turbo checkpoint from 17.2 to 16.6 s. The 12288 regression is the wave count outgrowing the divisibility of `groups_per_row` (32 waves for 48 groups leaves 16 waves with twice the work). It is inside the noise of what that width is used for and is not worth a balance heuristic yet.

## The int8 path in the training step

Torch-profiler view of one Anima step (1024x1024, batch 1, int8 base, per-block `torch.compile`). Profiled totals carry ~1.3-1.4x of profiler overhead, so read the shares:

| what | profiled ms | calls | share of the profiled step |
|---|---:|---:|---:|
| int8 GEMMs, forward | 343 | 252 | 11% |
| int8 GEMMs, input gradient | 385 | 252 | 13% |
| activation quantizer (rotate in the forward, plain in the gradient) | 148 | 504 | 5% |
| the rest of the step (attention fwd+bwd, LoRA, norms, pointwise, optimizer) | ~2170 | - | 71% |

Two calls per quantized Linear, one per direction, and no other int8-specific kernel in the step: the weight prepack happens once at load, and the gradient path no longer materialises `w.t().contiguous()` or a dequantized weight. What is left inside the int8 path is the separate rotation pass after the input-gradient GEMM (`dx = (g @ W) @ H`, currently an `aten::mm` over the `[M, K]` output: 0.28 ms per 2048-wide layer and 0.85 ms per 8192-wide layer at 4096 tokens, 60-80 ms/step for Anima) and the A-staging request pattern below.

## Results

Kernel-only, `bench/benchmark_int8_convrot.py --heuristic` over the shared grid in `bench/cells.py` (110 cells, both directions): the schedule `select_int8_config` picks, three rounds, medians of per-round medians. Absolute TFLOPS drifts ~3% between sessions with the clock state, so compare the pairs inside a row. bf16 is the same session's `torch.matmul` on the same shapes, and the CK column is that document's shipped heuristic from a different session.

Forward and backward, shipped schedules:

| model | cell (M x N_w x K_w) | CK fwd | int8 fwd | bf16 | fwd/bf16 | int8 bwd | bwd/bf16 |
|---|---|---:|---:|---:|---:|---:|---:|
| Krea 2 | 4112x6144x6144 | 37.2 | 38.3 | 24.2 | 1.58x | 37.6 | 1.55x |
| | 4112x1536x6144 | 31.5 | 31.8 | 22.3 | 1.43x | 38.8 | 1.74x |
| | 4112x16384x6144 | 39.7 | 36.9 | 24.5 | 1.51x | 37.5 | 1.53x |
| | 4112x6144x16384 | 36.4 | 37.8 | 24.4 | 1.55x | 38.8 | 1.59x |
| | 9232x6144x6144 | 38.8 | 38.6 | 24.4 | 1.58x | 39.3 | 1.61x |
| | 9232x16384x6144 | 40.7 | 38.8 | 24.6 | 1.58x | 39.0 | 1.59x |
| Qwen-Image 2.1 | 4112x4096x4096 | 32.2 | 36.5 | 23.7 | 1.54x | 35.6 | 1.50x |
| | 4112x24576x4096 | 36.0 | 37.3 | 24.8 | 1.50x | 36.1 | 1.46x |
| | 4112x4096x12288 | 32.9 | 35.7 | 24.0 | 1.49x | 36.3 | 1.51x |
| Z-Image | 4128x11520x3840 | 39.0 | 39.8 | 25.0 | 1.59x | 42.2 | 1.69x |
| | 4128x3840x3840 | 35.8 | 36.9 | 24.4 | 1.51x | 38.8 | 1.59x |
| | 4128x10240x3840 | 39.6 | 38.2 | 24.9 | 1.53x | 38.7 | 1.55x |
| | 4128x3840x10240 | 36.5 | 35.7 | 24.7 | 1.45x | 40.5 | 1.64x |
| H3 | 5376x21504x5376 | 42.0 | 38.4 | 25.0 | 1.54x | 40.5 | 1.62x |
| | 5376x5376x7168 | 38.8 | 37.3 | 25.0 | 1.49x | 41.6 | 1.66x |
| | 5376x28672x5376 | 42.2 | 39.9 | 24.8 | 1.61x | 39.4 | 1.59x |
| | 5376x5376x14336 | 37.2 | 37.0 | 24.4 | 1.52x | 41.2 | 1.69x |
| | 25344x21504x5376 | 41.4 | 39.7 | 24.7 | 1.61x | 41.7 | 1.69x |
| MOSS-Image | 4112x9216x3072 | 36.5 | 37.2 | 24.4 | 1.52x | 37.1 | 1.52x |
| | 4112x3072x3072 | 34.0 | 36.6 | 24.4 | 1.50x | 35.4 | 1.45x |
| | 4112x16384x3072 | 36.9 | 36.7 | 24.1 | 1.52x | 35.2 | 1.46x |
| | 4112x3072x8192 | 29.5 | 34.8 | 23.7 | 1.47x | 37.0 | 1.56x |
| Anima | 4096x2048x2048 | 29.0 | 34.1 | 16.3 | 2.09x | 33.2 | 2.04x |
| | 512x2048x1024 | 18.0 | 26.7 | 19.5 | 1.37x | 15.7 | 0.81x |
| | 4096x8192x2048 | 34.6 | 39.4 | 26.7 | 1.48x | 31.1 | 1.16x |
| | 4096x2048x8192 | 29.9 | 31.6 | 16.9 | 1.87x | 36.5 | 2.16x |
| | 9216x2048x2048 | 28.7 | 35.6 | 16.8 | 2.12x | 36.5 | 2.17x |
| | 9216x8192x2048 | 35.2 | 37.6 | 23.3 | 1.61x | 35.9 | 1.54x |
| Klein 9B | 4096x12288x4096 | 35.2 | 38.2 | 24.2 | 1.58x | 35.7 | 1.48x |
| | 4096x4096x4096 | 32.6 | 35.8 | 23.5 | 1.52x | 35.2 | 1.50x |
| | 4096x24576x4096 | 35.1 | 38.8 | 23.7 | 1.64x | 37.1 | 1.57x |
| | 4096x4096x12288 | 34.3 | 36.1 | 23.8 | 1.52x | 37.2 | 1.56x |
| | 4608x36864x4096 | 37.7 | 39.3 | 24.8 | 1.58x | 37.1 | 1.50x |
| | 4608x4096x16384 | 35.2 | 36.1 | 23.9 | 1.51x | 38.4 | 1.61x |
| | 9728x4096x4096 | 34.6 | 36.6 | 24.3 | 1.51x | 37.5 | 1.54x |
| TE (Qwen3.5) | 2048x4304x1152 | 32.2 | 38.5 | 23.4 | 1.65x | 39.3 | 1.68x |
| | 2048x1152x4304 | - | 39.1 | 20.7 | 1.89x | 40.9 | 1.98x |

Forward is now at or above CK on 30 of 34 measured cells (the four misses are the widest outputs, 4-6%, and well inside the session drift). The backward is a new capability and runs at 1.45-2.17x the bf16 GEMM it replaces, i.e. now at the same throughput as the forward rather than 4-10% below it. Anima's cells - the ones sd-scripts trains - are 1.37-2.12x bf16 in the forward and 1.16-2.17x in the backward (the one exception, the 512-row cross-attention gradient, is a 0.14 ms kernel whose output is only 512x1024).

Per training step (Anima, 1024x1024, 28 blocks, per-block `torch.compile`, `max-autotune-no-cudagraphs`, no gradient checkpointing, `bench_anima_step.py`, same session, 3 steps x 3 cycles):

| base weights | forward | input gradient | activation quantization | step | peak VRAM |
|---|---|---|---|---:|---:|
| bf16 | bf16 | bf16 | torch matmul + row quantize | 2465 ms | 15.60 GiB |
| ConvRot int8 | int8 GEMM | int8 GEMM, no transposed copy | torch matmul + row quantize | 2300 ms | 13.89 GiB |
| ConvRot int8 | int8 GEMM | int8 GEMM | fused radix-4 producer | 2250 ms | 13.76 GiB |

That is -8.5% step time against bf16 for half the base-weight bytes, and the losses of every configuration agree to the third digit, so the quantization behaves identically throughout.

Activation quantizer, kernel-only (`bench_quant_feather.py`), against the torch-matmul rotation plus the Triton row quantize it replaces:

| M x K | torch rotate | Triton quantize | pair | fused | speedup |
|---|---:|---:|---:|---:|---:|
| 4096x2048 | 0.279 | 0.603 | 0.882 | 0.368 | 2.40x |
| 4096x1024 | 0.153 | 0.550 | 0.703 | 0.193 | 3.64x |
| 4096x8192 | 0.849 | 0.753 | 1.602 | 1.312 | 1.22x |
| 9216x2048 | 0.517 | 0.649 | 1.167 | 0.806 | 1.45x |
| 4112x6144 | 0.670 | 0.637 | 1.307 | 1.189 | 1.10x |
| 512x2048 | 0.060 | 0.625 | 0.685 | 0.058 | 11.84x |

Producer after the in-register rotation (current). The stages used to publish and re-read the group through LDS after every stage (one fp32 group slot per wave). They now run in registers - stage 0 is intra-lane, stage 1 exchanges its `xor 1` partner, and stages 2 and 3 fetch their four group values with `xor 2/4/6` and `xor 8/16/24` (the complement is the mirror lane, so four shuffles per value cover it). The arithmetic and the summation order are unchanged, so the outputs are bit-identical to the previous kernel on the 13 captured cases (staged and streamed rows, bf16 and fp16, with and without the column scale). Kernel-only, median of three `do_bench` rounds, spread <= 1.4% (`~/tmp/feather_ops/bench_producer.py`):

| forward cell (M x K) | before | after | speedup | gradient cell (M x K) | before | after | speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| 4096x2048 | 0.369 | 0.204 | 1.81x | 4112x1536 | 0.150 | 0.154 | - |
| 4112x3072 | 0.573 | 0.289 | 1.98x | 4096x2048 | 0.199 | 0.202 | - |
| 4128x3840 | 0.732 | 0.339 | 2.16x | 4128x3840 | 0.331 | 0.325 | 1.02x |
| 4112x4096 | 0.751 | 0.354 | 2.12x | 4112x4096 | 0.341 | 0.335 | 1.02x |
| 5376x5376 | 1.326 | 0.561 | 2.36x | 5376x5376 | 0.525 | 0.496 | 1.06x |
| 4112x6144 | 1.126 | 0.489 | 2.30x | 4112x6144 | 0.466 | 0.447 | 1.04x |
| 9232x6144 | 2.489 | 0.990 | 2.51x | 9232x6144 | 0.923 | 0.874 | 1.06x |
| 5376x7168 | 1.935 | 0.714 | 2.71x | 5376x7168 | 0.705 | 0.639 | 1.10x |
| 4112x12288 | 2.424 | 0.883 | 2.74x | 4112x16384 | 1.241 | 1.002 | 1.24x |
| 5376x14336 | 3.593 | 1.256 | 2.86x | 25344x21504 | 11.565 | 8.094 | 1.43x |
| 4112x16384 | 3.331 | 1.089 | 3.06x | 4608x36864 | 3.189 | 3.084 | 1.03x |

The forward producer moves from 60-68 GB/s to 123-186 GB/s, i.e. to the gradient path's bandwidth, and the gradient path gains where it used to spend LDS on the group slot (the 16384-wide rows now stage at four workgroups per WGP, and the streamed 21504-wide row is 1.43x faster). Per Krea 2 step the producer total drops from 0.42 s to 0.18 s (forward 0.30 -> 0.13, gradient 0.12 -> 0.10, 28 blocks), 3.4% of the int8 path.

## Where this kernel stands against the fp8 one

The sibling kernel in `hip_optimization_plan_prepacked_e5m2.md` computes the same `[M, K] x [N, K]` GEMM on the same part with the same `16x16x16` WMMA, so the two are directly comparable. `python -m bench.benchmark_kernels --backends fp8,int8_fwd,int8_bwd,bf16` measures every backend on one cell in an interleaved pass, so the columns are paired. The grid-wide result (110 cells, one session):

| backend | mean TFLOPS | median |
|---|---:|---:|
| `fp8` | 32.82 | 35.05 |
| `int8_fwd` | 32.87 | 35.29 |
| `int8_bwd` | 34.02 | 37.63 |
| `bf16` (control) | 16.49 | 18.07 |

`int8_fwd / fp8` is 0.989 at the median, and the per-group medians say where: this kernel wins the 1536-wide ladder (1.18x), Anima's short-K cells (1.17x), the 6144x6144 ladder (1.04x) and Krea 2's 16384-wide and 6912 cells (1.05x). It loses `sdxl` (0.90x), `wan` (0.91x), `text-1152` (0.93x), H3 (0.94x) and the 129-256-row cells (0.96x). The same cells, as TFLOPS:

| cell (fwd) | int8 | fp8 | int8/fp8 |
|---|---:|---:|---:|
| `5376 x 5376 x 7168` | 38.63 | 41.15 | 0.939 |
| `12032 x 5120 x 5120` | 39.65 | 40.60 | 0.977 |
| `32 x 25600 x 5120` | 11.88 | 13.21 | 0.899 |
| `4112 x 6144 x 6144` | 38.90 | 35.73 | 1.089 |
| `4112 x 1536 x 6144` | 31.33 | 27.16 | 1.154 |
| `9232 x 1536 x 6144` | 33.46 | 25.22 | 1.327 |
| `4096 x 2048 x 8192` | 34.72 | 28.91 | 1.201 |
| `256 x 4096 x 4096` | 34.51 | 32.71 | 1.055 |
| `1 x 96768 x 2688` | 0.47 | 0.45 | 1.043 |
| all 106 | mean 32.87 | mean 32.82 | median 0.989, mean 0.992 |

Three facts come out of it, and they are the ones that shape both plans:
- This kernel is MMA-issue-bound: 37-42 TFLOPS is 73-78% of the 54 TF issue ceiling, and it sits at 35-47% of the DRAM roof on the wide cells of that list (63-79% on the narrow ones, where the modelled traffic is closer to the roof). The fp8 kernel is the opposite: 168-183 GB/s (79-86% of the 212 GB/s roof) where it wins, 61-80% of the issue ceiling. So the two are limited by different resources on the same silicon, which is why the win flips with the cell instead of with the dtype.
- Where this kernel wins, the mechanism is measured and not a mystery: one-byte A operands and 128/256-row tiles, i.e. half the activation traffic and 2-4x fewer weight passes. On `4112 x 1536 x 6144` it runs its `128x256` schedule at 32.4 while the fp8 kernel's own `128x256` measures 20.7 on the same box - same tile, same bytes, so part of that gap is the fp8 kernel's decode plus its 33 KB LDS footprint, and the rest was its heuristic: it now drops to a 16-row tile there and reaches 27.3.
- No runtime router follows from this. The two paths consume different weight formats (the 4x4 int8 ConvRot prepack here, `(K/16, N, 16)` fp8e5m2 there) and different activations, so the choice is a checkpoint and integration decision.

The list also re-prices this kernel's own weak cells: the wide-output cells where it trails CK are the cells where the fp8 kernel is fastest (H3 `5376 x 21504`, `4096 x 12288 x 4096`), i.e. this kernel's remaining 5-10% there is issue efficiency on the 256x256 schedule, not traffic.

## Accepted

| change | mechanism | measured |
|---|---|---|
| 4x4 byte micro-tile prepack | one weight copy serves both directions: the forward reads four dwords of four consecutive `k4` tiles, the backward four byte-`i` gathers of four consecutive `n4` tiles | backward exists at all, no transposed copy (the previous path paid 1.2-14.9 ms per call, plus 161 ms/step of weight dequantization) |
| double-buffered LDS + register prefetch | the next chunk's global reads issue before the current chunk's WMMAs and store into the other buffer, so one barrier per chunk instead of two | 33.4 -> 38.2 TFLOPS on `4112x6144x6144` (+14%), every cell improved. The fp8 kernel separated the two halves of this design and found that the overlap is what pays and the second buffer is what costs there (same-buffer register prefetch +7.7% mean, two buffers -28% on its 128x256 tile). Re-pricing it here shows the second buffer is free on the VGPR-capped tiles, so this form stays and the single-buffer alternative is in Rejected |
| 256x256 (16 warps) for wide outputs | halves the number of A panels and the number of blocks | the bulk body for `out >= 6144`, up to 39.8 TFLOPS forward and 42.2 backward |
| 128x128 (8 warps) for mid outputs | more blocks per cell, so the tail blocks of a 128-wide tile spread over 40 CUs | Anima `4096x2048x2048` 34.6 vs 30.2 for 256x256 |
| 256x128 for narrow outputs | a 256-wide tile is mostly tail when the output is 1152-wide | 34.9-37.8 TFLOPS on the text-encoder cells |
| 64x64 for `M <= 128` | prefix passes and single-row cells are bandwidth bound, a big tile only wastes MMA work | `16x6144x6144` 8.3 TFLOPS vs 2.3 for 32x512 |
| per-direction schedule | the direction fixes which axis is the output, so the same cell gets a different tile in the forward and the backward | Anima `4096x2048x8192`: 128x128 forward (30.1), 256x256 backward (33.7) |
| RNE output conversion | the fp8 kernel truncates because its values come from an fp8 widening. Here the scale multiply is the only rounding | matches the reference kernels bit for bit |
| tail-aware launch | full M tiles with the chosen schedule, then one launch for the `M % block_m` rows with a 64-row tile: a 256-row tile otherwise computes `ceil(M/256)*256` rows of MMA work | the 4112-row cells (Krea 2 / Qwen / MOSS image rows) went from 36.9 to 38.3 fwd and from 34.1 to 37.6 bwd. Exact-M cells unchanged |
| backward B staging on the layout's contiguous axis | the prepacked layout stores the weight's output axis contiguously, so the backward assigns lanes to the contraction index: one 512 B span per warp instruction instead of 32 scattered lines, with a 16 B pad on the LDS stage stride so a lane group spanning stages does not alias banks | backward 33-36 -> 35-42 TFLOPS, i.e. the forward/backward gap closed |
| hoisted staging offsets | per-thread byte offsets advance by one constant per chunk instead of a 64-bit multiply. `SALU` was 33% of all issued instructions | part of the same measurement (37.9 -> 38.3 on the 256x256 cells) |
| int8 pack via `cvt_pk_i16` + `byte_perm` | two saturating i32->i16 packs and one permute replace four shifts, four masks and three ors per four codes | part of the producer's 1.10-11.8x |
| in-register radix-4 producer stages | the four Hadamard stages act on index digits {0,1}/{2,3}/{4,5}/{6,7}: stage 0 is intra-lane, stage 1 needs the `xor 1` partner, stages 2/3 the three partners at `xor 2/4/6` and `xor 8/16/24`, and a stage's complement is always the mirror lane. The whole rotation runs in registers (14 shuffle instructions per lane per group) with no LDS group slot | forward producer 60-68 -> 123-186 GB/s, 1.8-3.1x per width, bit-identical outputs. The freed slot also raises the wide-row occupancy (16384-wide gradient 1.24x, streamed 21504-wide 1.43x) |
| small-M tile tier (64x64 for `M <= 256` up to `out = 4096`, and `M <= 128` up to 8192) | a 128-row tile splits into a full launch plus a 64-row tail launch - two single-wave grids - and at 192-256 rows the per-CTA fill/drain dominates. One 64x64 launch keeps the whole input in one grid. Past the output crossover the 128x128 tile's N panel wins, and 256x256 is 1.1-2.2x worse | `192x2560x2560` -37.1% fwd / -25.7% bwd, `512x4096x4096` fwd -1.7% / bwd -20.9%. 122 other cells unchanged |
| power-of-two-contraction exception bounded to `M >= 1024` | the L2-aliasing exception was fitted at `M = 4112`, but at `M = 512` the 256x256 tile is 22-29% faster, so it now only applies from 1024 rows up | `512x4096x4096` gradient 19.3 -> 25.0 TFLOPS |
| mid-width output rule (256x256 for `out >= 3072` with a real reduction) | the gradient direction already chose 256x256 there. The forward did not, and the 256-wide tile halves the A panels | +3.4% to +7.9% on Z-Image 3840x*, H3 out_proj/fc2 5376x*, MOSS 3072x*, and both directions of Qwen-Image 9232x4096x4096. The power-of-two-contraction-at-small-M exception keeps Qwen/Klein 4096x4096 on 128x128 (3.5%) |
| fused activation quantizer | one feather-ops kernel: radix-4 Hadamard (the regular H is `h4^(x)4`, and each row of h4 has one -1, so a stage is `sum - 2*partner` with no multiplies), row absmax, int8 codes. fp32 only for the group being rotated, bf16 row buffer, one block per row | 1.10-11.8x the rotate+quantize pair (1.6x on the shape sum), -2% step time, and it removes the separate `grad * w_scale` pass |
| producer wave budget from the workgroup LDS limit, plus a streaming path | the budget was a hardcoded 32 KB of total footprint, so `K >= 16384` computed zero waves and clamped to one (32 threads, 33 KB of LDS per block, <1 resident wave per SIMD). It now uses the 64 KB per-workgroup limit minus the static wave-maxima array, and rows that no longer leave room for a full block of waves (`K > 16384`) skip the bf16 row buffer and recompute each group from global memory in the quantize pass. Both paths share one `produce_group`, so the scale and the codes are identical | 16384-wide gradient rows 4.66 -> 1.16 ms (43 -> 175 GB/s), 14336 1.28 -> 0.90 ms. The Krea 2 gradient cell `4112x16384x6144` 26.6 -> 23.3 ms. 21504-248320-wide rows (previously a failed launch) stream at 144-171 GB/s |
| small-M and 512-row schedule refit | sixteen tiles measured per cell in a balanced sweep (eight providers, eight rounds), each proposed change re-tested paired against a duplicate of the shipped tile, then the whole matrix A/B'd | +6.7% on the 11 cells that changed, up to +58% on `1 x 2688 x 96768`. The other 125 cells are unchanged to +0.01%. Net +0.24% of the kernel sum, because the cells it fixes are the small ones |
| `weight_packed` flag on the sd-scripts op | the prepack is done once after `load_state_dict`, the shape stays `[N, K]` so LoRA creation and state matching are untouched | 280/254 weights packed at load, step time above |

## Rejected

| item | why it is not shipped |
|---|---|
| grouping M tiles (`gm > 1`, for B-panel reuse) | measured within 0.5-2% of n-major on every cell once the launch was correct (the first attempt under-sized the grid and produced invalid 1.8x-looking numbers) |
| 16 contiguous `k` per `n` (the fp8 documents' `(K/16, N, 16)` layout) | the backward would need 16 single-byte LDS loads per fragment, i.e. a second copy or a 16x LDS instruction count |
| 8x8 micro-tiles | one axis is then `stride 8` inside the tile, which no LDS instruction can use |
| `v_perm` based 16x16 byte transpose in the staging path | 3 perms per 4 bytes staged, versus 3 perms per 16 bytes for the 4x4 tile |
| per-`k`-group (AWQ/GPTQ style) scales, 4-bit operands | changes the quantization and the checkpoint format, out of scope by construction |
| split-K / stream-K | the big shapes are at 74-78% of the MMA-issue peak and the starved small-M cells are limited by per-CTA fill/drain, not by the grid: the fused-equivalent upper bound is 1.05-1.4x at best, and the reduction that separates the slices costs 20-100% of the cell, so the net is negative everywhere except 32-row cells (0.19 ms). Also needs exact-shape instantiations to reach one-slice parity. Small-M section above. The fix is the tile rule |
| separate autocast/tile policy per model | the heuristic is a function of `(M, out, K)`, so image-size changes (the M axis) land in the right class without a per-model table |
| clamp-free int8 scale (`amax / 127.5` with a rounding margin, as ~/sageattention-autotune does) | removes two ops per code, but the quantizer is latency bound and not ALU bound: measured 4.004 ms vs 3.925 ms on the same sweep (no gain), and it shifts ~half the codes by one relative to the `/127` reference recipe. Kept the reference convention |
| one wave per row in the activation quantizer | with the row in LDS at 4 B per value, K=6144 fell to one wave per block (1 wave/SIMD): 33 GB/s and slower than the pair it replaces. Spreading a row over the block's waves (one group per wave) fixed it |
| gap-free radix-4 stages that round to bf16 between stages | four mantissa roundings instead of one: values off by ~1.6%, i.e. more quantization error than the reference recipe |
| LDS stage padding as a bank-conflict knob | the conflicts are real (40 per dispatch on `4112x6144x6144` in the forward against 10 in the backward, so they are the forward's B staging) but they are not padding-sensitive: 0, 32 and 48 bytes per stage against the shipped 16 measure -1.3%, -0.4% and -0.5% on a 52-case fwd+bwd subset, i.e. the shipped 16 B pad is already the best of the four. The forward's 4-byte B stores land on 16 banks because the global layout is output-contiguous, which sets the lane-to-store mapping. The alternatives are the backward's lane mapping (coalesced LDS, 32 scattered global lines) or a `v_perm` transpose in staging (3 perms per 4 bytes, already rejected). Structural, not tunable |
| single-buffer staging with a register prefetch image (the fp8 kernel's form) | re-priced and dropped: only the 64x64 schedule is LDS-capped (15 waves/SIMD against the 16-wave ceiling), so freeing half of its 8.3 KB buys at most one wave on a tile that serves `M <= 256`, while the 128x128 (12 waves) and 128/256-row (7-8 waves) tiles are VGPR-capped and would only lose the second buffer's barrier saving. The fp8 kernel's +7.7-23% from the overlap does not transfer because its tiles are LDS-capped at 50-75% occupancy and these are not |
| two dozen configs ranked in one interleaved sweep | the screen under-measured the 256x256 tile by 10-17%, so it proposed 60 cell changes that a paired re-test (shipped vs shipped-duplicate vs candidate) cut to 11. A screen is only usable with a duplicate provider measuring its own noise floor, and with `rounds >= providers` so the rotating order visits every position |
| accumulators freed early in the epilogue | the accumulator count is set by the tile's register-tile rectangle and is live for the whole K loop, not by the epilogue: writing each 16x16 tile out as it completes frees registers only after the last MMA has already been issued. The 188-204 VGPR of the 256-row tiles is a property of `repeat_m x repeat_n`, so the way to more resident blocks is fewer register tiles, not a restructured epilogue |
| guard-free stage loop for full chunks | specialising the tail out of the hot loop duplicates the unrolled body. The 16-warp tiles are the largest register consumer (188-204 VGPR per wave, 128 of them accumulators) and the duplicate pushed them over budget, halving their throughput (37.7 -> 17.8 TFLOPS). The uniform per-stage predicate the shipped loop uses is free |

## Optimization plan

Kernel-only budget of the int8 path for Krea 2 at `M = 4112` (28 blocks x 8 Linears, real weights, measured with `~/tmp/feather_ops/bench_producer.py` and the GEMM harness):

| component | s/step | share | current state | next step |
|---|---:|---:|---|---|
| forward GEMM | 2.68 | 45% | 37-40 TFLOPS. Behind CK only on wide outputs | tiled A + `chunk_k = 128` (below) |
| input-gradient GEMM | 2.62 | 44% | 37.8-39.1 TFLOPS | none larger |
| forward producer (rotate) | 0.13 | 2% | 123-186 GB/s after the in-register rewrite | none |
| gradient producer | 0.10 | 2% | 123-202 GB/s, bandwidth-bound | none |
| gradient epilogue rotation | 0.17 | 3% | ~= a bandwidth pass | fuse into the dgrad epilogue |

What is left, ranked by expected gain per unit of risk, with the gate each item must pass. Everything the earlier passes finished is in Accepted and Rejected. The two open kernel items are:
- Gradient epilogue rotation. Fuse `dx = (g @ W) @ H` into the dgrad epilogue (accumulate a full 256-wide k group per CTA, rotate from LDS in 64-column chunks). Cost is `256/K` of extra MMA-equivalent (4.2% at K=6144, 1.6% at 16384) against a separate pass at 6-7% of the dgrad: net 2-4% of the dgrad, plus one fewer bf16 round trip. Gate: not bit-identical, so gradient-cosine tests instead. This is the largest item left on the path, because it is the only one that removes work outside the GEMM.
- Tiled A + `chunk_k = 128` for the wide-output cells. All cells where we trail CK are wide-output or K-dominant: Krea 2 `16384x6144` (36.9 vs 39.7), H3 `21504x5376` (38.4 vs 42.0), `28672x5376` (39.9 vs 42.2), `25344x21504x5376` (39.7 vs 41.4), Z-Image `10240x3840` (38.2 vs 39.6). Design and decision below.

Done and recorded elsewhere: the forward producer's in-register radix-4 rotation (Accepted), the schedule refit that followed the failure of the "the heuristic is already optimal" claim (Accepted, and the mis-ranked 22-config screen that claim rested on is in Rejected), the backward's B staging on the layout's contiguous axis (Accepted), and the single-buffer and accumulator-epilogue alternatives (Rejected). Not worth doing: a fused group-64 producer (only H3's `M=1` adaln uses it, <=1 ms/step), split-K (within 8.4% of memory-free time), deeper staging (spills the 16-warp tiles), fp8 requantization (changes outputs).

### Tiled A: what it does and does not require of the prepacked weight

The A tiling is a global-memory layout only. It is independent of the WMMA operand layout and of the prepacked weight bytes (that is already stated in `int8_convrot_fwd_plan.md`, where the tiled form shipped). The real coupling is the K block: the tiled layout is defined for `chunk_k = 128` (BKB 128) and nothing else - the earlier kernel measured a rel. 1.42 error when a tiled A was fed to another K block, so the guard must be explicit. The current prepacked 4x4 micro-tile needs no change for any `chunk_k`: the forward stages B by walking `n4` across lanes (contiguous at every `k4`) and the backward by walking `k4` across lanes, so a 128-k chunk is 4x fewer barriers with unchanged coalescing.

The measured prerequisite: on the narrow `1536x6144` cell the old forward-only kernel's tiled-A + BKB128 form reached 31.5 TFLOPS, while the current double-buffered chunk-32 staging already reaches 32.4, so that win is effectively captured. What remains is the widest outputs: Krea 2 `16384x6144` forward 37.9 vs CK 39.7, H3 `21504x5376` 39.6 vs 42.0 and `28672x5376` 38.9 vs 42.2 - cells where the 256x256 tile re-reads a 32-byte A row per lane over a very wide output. That is the case the tiled layout (with `chunk_k = 128`, one LDS stage) has to be measured on before anything else changes.

A chunk-grouped weight layout (`(K/128, N, 128)`) would lengthen the forward's B runs but make the backward's contraction stride 128 bytes, i.e. 16 scattered byte loads per fragment - a second copy or a large regression. A layer's forward and gradient flops are equal, so a layout that buys the forward `+x%` and costs the gradient `-y%` only pays if `x > y`, and here `y` is structural. Decision: keep the direction-symmetric micro-tile. The tiled A is A-only. That removes any load-time weight-layout decision.

The A layout itself is selected per call: the producer always runs immediately before its GEMM, and the GEMM id comes from `select_int8_config(M, N, K, transposed_b)` with the runtime M, so `tiled` is a property of the selected id and row-major is emitted below the threshold (the forward-only plan used `M >= 4096`). A per-`(weight, direction)` hint can cache the decision at load time from `(N, K)`: the fwd cell is `(M, N, K)` and the gradient cell is `(M, K, N)`, the model's M class comes from the table above (Krea 2 256-aligned 1280-9728, H3 ragged 2k-25k, text encoders 5-4096), and the default is tiled for the classes that trail CK. Implementation stages: (1) extend the cell matrix and keep the bit-identity harness, (2) add `unroll_k = 8` staging variants for 256x256 (A 32 KB + B 32 KB single-buffered = 64 KB, still 2 workgroups/WGP because VGPR caps it) and 128x256 (48 KB), with the tiled addressing as a template parameter and the `tiled <=> chunk_k == 128` guard, (3) producer output layout `q[m_tile][k_tile=128][128][128]` (all group-256 axes in the tables are multiples of 256 and Krea 2 pads M to 256. The tail store still masks M for H3 and the text encoders), (4) measure per cell and both directions, keep the id only where the fwd+dgrad sum improves beyond the +-3% session spread.

### Small-M cells: split-K, stream-K, or a schedule rule?

The cells that stay well below the model's 38-42 TFLOPS are the ones whose grid is starved. Geometry at the heuristic schedule (a 256x256 tile keeps 10 CTAs resident on the 160-SIMD chip, a 128x128 tile 20, a 64x64 tile 80) and the measured throughput (`~/tmp/feather_ops/bench_small_shapes.py --parts 1`):

| cell | dir | tile | CTAs | ms | TFLOPS | note |
|---|---|---|---:|---:|---:|---|
| `512x2048x1024` | fwd | 128x128 | 64 | 0.080 | 26.7 | Anima cross-attn kv (text side) |
| `512x2048x1024` | bwd | 128x128 | 32 | 0.135 | 15.6 | 1.6 waves. Below the bf16 `F.linear` at 19.5 |
| `512x4096x4096` | fwd / bwd | 128x128 | 128 | 0.564 / 0.904 | 30.5 / 19.0 | text encoder at 512 tokens |
| `512x12288x4096` | bwd | 256x256 | 32 | 1.686 | 30.6 | text encoder `qkv`/`o` gradient |
| `192x2560x2560` | fwd / bwd | 128x128 | 40 | 0.128 / 0.154 | 19.6 / 16.4 | two single-wave launches (see below) |
| `128x8192x5120` | fwd / bwd | 64x64 | 256 / 160 | 0.383 / 0.796 | 28.1 / 13.5 | short prompt, bandwidth bound |
| `32x25600x5120` | fwd / bwd | 64x64 | 400 / 80 | 0.879 / 1.149 | 9.5 / 7.3 | 32-token prompt, weight streaming |

Every one of these is far from the roofline, so the question is whether the fix is a different work decomposition (split-K / stream-K) or a different tile. Measured answer: it is the tile.

Split-K, measured as an upper bound. Splitting the contraction into `s` slices gives `s` times as many CTAs with `1/s` of the K loop each. The same MAC count and the same bytes are read by the single fused cell `(M, s*N, K/s)` (forward) or `(M, s*K, N/s)` (gradient), which therefore times exactly the best a split-K with `s` slices could do before the reduction. Against that, the cheapest possible fixup is one `s`-way bf16 sum of the `(M, out)` partials - no fp32 staging, no extra kernel. `--parts 3`, same process, medians of 3 rounds:

| cell | dir | s=2 | s=4 | s=8 | reduction at s=2 | net at s=2 |
|---|---|---:|---:|---:|---:|---:|
| `512x2048x1024` | fwd | 0.96x | 0.73x | 0.58x | 0.051 ms | -68% |
| `512x2048x1024` | bwd | 0.89x | 0.84x | 0.98x | 0.028 ms | -34% |
| `192x2560x2560` | fwd | 1.15x | 1.22x | - | 0.027 ms | -7.8% |
| `192x2560x2560` | bwd | 1.17x | 1.41x | - | 0.025 ms | -1.2% |
| `512x4096x4096` | fwd | 1.08x | 1.09x | 1.05x | 0.104 ms | -11% |
| `512x4096x4096` | bwd | 1.05x | 0.94x | 1.14x | 0.105 ms | -6.7% |
| `1024x4096x4096` | bwd | 1.09x | 1.03x | 1.16x | 0.204 ms | -6.8% |
| `128x8192x5120` | fwd / bwd | 0.65x | 0.62x | - | 0.051 ms | -67% |
| `32x25600x5120` | fwd | 1.07x | 1.05x | 1.01x | 0.040 ms | +1.7% |
| `32x25600x5120` | bwd | 1.20x | 1.04x | 1.27x | 0.011 ms | +15.8% |

Two conclusions. First, the starved cells are not limited by the grid: quadrupling the CTA count buys 1.05-1.4x at best, and on the smallest cells (`512x2048x1024`, `128x8192x5120`) the fused equivalents are slower, because the per-CTA pipeline fill/drain and the epilogue are amortized over 1/s of the K loop - the same cost that makes these cells slow. Second, the reduction is not free at this size: one bf16 pass over `(M, out)` costs 0.03-0.2 ms, which is 20-100% of the cell. The net is negative everywhere except the 32-row cells, where it saves 0.19 ms.

What the cells actually lose to. Sweeping all 16 configs per cell (`--parts 2,4`) and an M sweep from 64 to 1024 at three shapes (`--parts 5`) shows one systematic miss and one exception:
- `M` in 192-256 wants a 64-wide M tile in one launch. With a 128-row tile the tail launch (`m_full = 128`, then a 64-row launch) splits the grid into two single-wave launches of 20 CTAs, and the fill/drain is paid twice: `192x2560x2560` runs at 19.6 fwd / 16.4 bwd against 31.1 / 22.3 for one 64x64 launch (1.35-1.6x, three shapes, both directions. At `M = 256` the gain is 10%). The crossover back is shape dependent: the short-contraction cell (`N=2048, K=1024`) still prefers 64x64 at `M = 512` (30.1 vs 26.8 fwd), while the wide cells (`N=4096/12288`) switch back at `M = 384`. The `M <= 128` small-M rule already picks 64x64.
- `M = 512` with a wide output wants 256x256, not 128x128: `512x4096x4096` bwd 25.0 vs 19.3 (1.29x), `512x12288x4096` bwd 30.5 vs 24.9 (1.22x). The heuristic blocks it through the power-of-two-contraction exception (fitted at `M = 4112`, where 128x128 really is 3.5% better), so the exception needs an `M` bound rather than being dropped.
- A single exception in the other direction: `M = 128` with a very wide output (`12288`) prefers 128x128 by 1.38x over the 64x64 that the small-M rule picks, so the small-M tier needs an output-width bound too.

Verdict. No split-K and no stream-K. The mechanism only pays where the grid is starved and the cell is large enough to absorb a fixup pass, and there is no such cell in a training step: the whole small-M class is 6 ms per Anima step (0.27%, the cross-attn kv forward plus its input gradient) and the text encoder's 512-row cells are 1-2% of a step only when the TE itself is trained. torch-ggml-ops reached the same conclusion on packed quantized GEMMs from the other side: their split-contraction body needed exact-shape instantiations to reach parity at one slice and then won 1.92x at M=64, 1.28x at M=128, 1.20x at M=256 and 1.06x at M=512, and they screened stream-K out over 100 deployed keys because any contraction of 2048 rows or more already yields 64+ slices. Below M=256 a 64x64 tile recovers most of it for free.

Done (the tile rules). `select_int8_config` now carries the small-M tier and the bounded exception:

| condition | schedule |
|---|---|
| `M <= 64` | 64x64 |
| `M <= 128` | 64x64 up to output 8192, else 128x128 |
| `M <= 256` | 64x64 up to output 4096, else 128x128 |
| power-of-two contraction in [4096, 8192] aliasing exception | only for `1024 <= M < 8192` (was `M < 8192`) |

A/B over all 126 bench cells (`~/tmp/feather_ops/verify_heuristic.py`: previous rules vs new, interleaved rounds, then the full 16-config sweep on every cell that moved) - four cells changed, all faster, none slower:

| cell | dir | before | after | delta | best of 16 |
|---|---|---|---:|---:|---|
| `192x2560x2560` | fwd | 128x128, 0.131 ms | 64x64, 0.082 ms | -37.1% | 64x64 |
| `192x2560x2560` | bwd | 128x128, 0.155 ms | 64x64, 0.115 ms | -25.7% | 64x128 (-3.3%, noise) |
| `512x4096x4096` | fwd | 128x128, 0.561 ms | 256x256, 0.551 ms | -1.7% | 64x64 (+0.3%, noise) |
| `512x4096x4096` | bwd | 128x128, 0.888 ms | 256x256, 0.702 ms | -20.9% | 256x256 |

The other 122 cells are schedule-identical by construction (the new tiers only cover `M <= 256`, and no other bench cell has a sub-1024-M power-of-two shape), so the bulk rows in Results stand. Worth 1.35-1.6x on the text-side cross-attention and on a short-prompt text encoder, ~0.3% of an Anima step.

Two residuals are deliberately not fitted, because the optimum is not monotone in M and these are 0.1-1 ms kernels whose numbers move by 5-10% between sessions: `M = 256` with `out >= 4096` in the gradient direction prefers 128x256 (17.0 vs 12.2 TFLOPS at `4096x4096`), while at `M = 128` the same tile is 10-11% worse, and `M = 512` on the short-contraction `2048x1024` cell prefers 64x64 in the forward (30.1 vs 26.8) while its gradient prefers 128x128. `512x2048x1024`'s gradient is left alone - 128x128 is already the best of the 16 configs and the cell is 0.14 ms.

If a small cell ever becomes hot (a per-token or per-frame loop, `M <= 128` with `>= 1 ms`), the design would be: `slices = ceil(target_ctas / grid)`, each CTA taking a contiguous `K/s` range that is a multiple of 128 so the tiled-A form stays legal, writing fp32 partials to a workspace (bf16 output cannot take racy atomics), and one reduction kernel summing in ascending slice order and rounding once - which is also what torch-ggml-ops deployed. It needs exact-shape instantiations to reach one-slice parity, so it is a per-cell mechanism, not a switch.

## Still open (ranked by measured or estimated gain)

The two ranked kernel items - the fused dgrad rotation and the tiled A layout - are in the plan above with their cost and gate, and the tiled A design has its own section. What follows is everything else, which is integration or axis-coverage work rather than a known kernel gap:
- Permuted (HND) epilogue store, optional. The training loop's attention backward packs five tensors from BSHD to HND per block (~17 ms/step at 1024^2, 4.5x that at 1536^2 buckets). Three of them (`q`, `k`, `v`) leave our GEMM epilogue and one (`dout`) leaves our dgrad epilogue. The epilogue already writes each 16-byte output piece through LDS, so emitting HND is an address change rather than a restructure (16 B never straddles a head: `head_dim 128` = 256 B), and it needs no per-head GEMMs. `out` additionally needs the consuming `o_proj` GEMM to read HND, which is the same layout freedom as item 2. Cost: a layout mode on the GEMM op plus (heads, head_dim) on the epilogue.
- Backward gather instead of the `v_perm` chain: twelve `v_perm_b32` per 16-byte chunk, whose inline 32-bit selectors become 96-bit instructions on this part. The alternative is a chunk-order LDS layout for the backward with 4 x `ds_read_b32` fragment reads. It is a wash on instruction count, so it only pays if PC sampling shows fetch stalls.
- Schedule sweep over the occupancy axis, the one axis the sixteen-tile sweep did not cover. The 128/256-row tiles are VGPR-capped at 188-204 per wave for 16 register tiles (128 accumulators), and that count is a property of the `repeat_m x repeat_n` rectangle, not of the epilogue - so the lever is fewer register tiles at the same box, not a restructured epilogue (which is why the accumulator item sits in Rejected). Two points are untested: `64x256` with 8 warps and 8 register tiles (~104 VGPR -> 14 waves/SIMD, at 2.5x worse bytes/MAC) for the small cells that are not traffic-bound, and `512x256` with 32 warps (traffic-optimal, LDS-capped at 4 waves/SIMD) for the wide ones. Both are safe now that the tail has its own launch, and both are quick to add and measure with the same balanced sweep the refit used.
- `--convrot_int8_bwd int8` for models with `groupsize != 256`: the fused quantizer implements the group-256 Hadamard only. 64-group text encoders still take the Triton producer path (the GEMM is group-size agnostic).

Constraints that shaped the list above, kept so a future change does not re-learn them:
- Register budget of the 16-warp tiles. The 256x256 schedule measures 192 VGPR per wave (16 register tiles = 128 accumulators), which is 8 waves per SIMD, so one 16-wave block spans two SIMDs and only two blocks fit per WGP. Any growth of the staging image or the loop body spills and halves its throughput, as the guard-free-loop experiment showed. The same tile with 8 warps would need 256 VGPR per wave and does not fit.
- The producer's neighbourhood. Rotation and quantize run in one kernel (see Accepted), and the weight scale for the gradient path is folded into it. The producer is done: 123-202 GB/s against a 241 GB/s roof, bandwidth-bound in the gradient and latency-bound in the forward until the radix-4 stages moved into registers. What is left there is only a tiled-A output layout, if the GEMM ever wants one (see the tiled-A item above).
- Tile choice stability. Only the `M = 512` cells with a large contraction (32-128 blocks over 40 CUs) switch winners between sessions, so the heuristic is deliberately coarse there. The two small-M residuals listed above (`M = 256` wide-output gradient, `M = 512` short-contraction forward) are the same effect and are left unfitted.

## ComfyUI inference wiring

The same kernel now serves inference through `~/ComfyUI-FeatherOps`. `comfy_feather/quant.py` registers `FeatherInt8ConvRotPackedLayout` as a `comfy.quant_ops.QuantizedLayout`: `quantize` rotates group-256 blocks with the same regular Hadamard the kernel's activation pass applies and quantizes per output channel, `dequantize` un-rotates, `get_plain_tensors` hands the prepacked micro-tiles and the `[N]` scale to `int8_scaled_mm`, and `state_dict_tensors` reproduces the released `[N, K]` int8 + `[N, 1]` scale exactly. Because it is a layout rather than a bespoke class, ComfyUI's own patch machinery works on it unchanged: a LoRA is merged in float and requantized by `quantize`, and ComfyUI asks for that with `want_requant` on the offloaded path (`comfy_feather/ops.py`).

`FeatherUNetLoader` picks the format per layer from the checkpoint, and a checkpoint that is already quantized is never quantized twice:
- int8 with a `comfy_quant` marker saying `convrot: true, convrot_groupsize: 256` and one scale per output channel goes through `make_feather_int8_weight`, which is only a byte permutation - no rotation, no re-quantization, so the load is bit-exact.
- fp8e5m2 is prepacked from its stored bytes with its own scale, or with scale 1 when the file has none (Kijai's `fp8_e5m2_scaled` files bake the scale into the values), and runs on the fp16-fp8 kernel.
- packed codes (nvfp4, mxfp8, int4, grouped int8) are dequantized through `comfy.ops._load_quantized_module`, so that key-to-params table stays ComfyUI's rather than a copy of it here.
- and the layers the exporter deliberately kept in float stay in float.

A file that stores neither int8 nor fp8e5m2 is a float model: the comfy-quants recipe decides which modules are cast to fp8e5m2. The recipe is not consulted for an already-quantized file, because the exporter's per-layer split is the accuracy decision. That is also what ComfyUI's own `mixed_precision_ops` Linear does.

Measured end to end on Anima (`anima-base-v1.0-int8_convrot.safetensors`, one KSampler step at 512^2, medians of five, one session): 254 layers on this kernel, 261 in float, 277.4 ms against ComfyUI's own int8 path at 325.2 ms on the same file (1.17x), landing 1.6e-2 from the bf16 model against ComfyUI's 1.6e-2.

The stored-fp8e5m2 case measures 3236.9 ms against ComfyUI's 4777.3 ms on Wan 2.2 LOW `fp8_e5m2_scaled` (1.48x, one 832x480x5 step), with the latents 2.6e-3 apart. The fp8 path on the bf16 checkpoint is 257.6 ms and lands 7.4e-2 away, so on a file that only quantizes part of its layers the fp8 cast is faster but half an order of magnitude less accurate. On a fully int8 file the two kernels measure within a few percent of each other in TFLOPS (`Where this kernel stands against the fp8 one`), and this kernel's per-row/per-channel scaling is what buys the accuracy.

## sd-scripts wiring

The kernels are consumed through `~/sd-scripts/library/convrot_int8_kernels.py` (feather-ops backend region) and `library/convrot_int8_utils.py`:
- One pack, at load time. `apply_convrot_int8_monkey_patch` registers a `load_state_dict` post hook. The hook rewrites every quantized `nn.Linear`'s int8 weight into the prepacked order with `prepare_int8_weight`. The tensor keeps its `[N, K]` shape (the bytes are permuted, the element count is unchanged) so LoRA creation, state dict matching and every shape-based check are untouched, and `module._convrot_weight_packed` tells the op which layout it holds. Weights whose axes are not multiples of 16 keep the released layout and stay on Triton, and `released_int8_weight` is the exact inverse for the dequantizing fallback path.
- Forward (`convrot_int8_linear` custom op): `quantize_convrot_activation(x, rotate=True)` (the feather-ops radix-4 producer when the group size is 256, otherwise the rotation matmul plus the Triton row quantizer), then `int8_scaled_mm(..., transposed_b=False)` with the per-row activation scale, the per-channel weight scale and the bias.
- Input gradient (`convrot_int8_linear_backward`, `bwd_mode="int8"`): `quantize_convrot_activation(g, rotate=False, col_scale=w_scale)` folds the weight's per-output-channel scale (which cannot be pulled out of a sum over that axis) into the quantizer, then `int8_scaled_mm(..., transposed_b=True)` against the same weight copy, then the block Hadamard on the output. The previous path materialised `wq.t().contiguous()` per call (1.2-14.9 ms per layer) plus a separate `g * w_scale` pass, and in the default `bf16` mode dequantized the whole weight every step (161 ms/step).
- Backend selection is resolved once: feather-ops when the extension imports (ROCm only), Triton otherwise. `SD_SCRIPTS_CONVROT_INT8_BACKEND=triton|feather-ops|auto` overrides it for A/B runs, and the log line `INT8 linear backend: ...` states what was chosen. Triton is still required for the non-256 group sizes and for the non-ROCm fallback, but not for the fast path's quantizer any more.

The training-loop view of the same numbers (what else the step spends time on, which copies are left, and the fusion candidates outside the int8 path) is in `~/sd-scripts/plan_optimizations.md`.

## Verification

- `~/ComfyUI-FeatherOps/tests/test_feather_int8.py`: the layout's eligibility gates (int8, per-output-channel scale, `comfy_quant` marker with `convrot`/group 256, K a whole number of groups), that `_load_from_state_dict` picks this layout for an int8 entry and fp8 for a float one, that the dispatcher calls `int8_scaled_mm`, that `state_dict_tensors` reproduces the checkpoint bytes, that a merged LoRA requantizes back into this layout, and that the `comfy_quant` marker is consumed rather than leaked.
- `tests/test_int8_convrot.py` (feather-ops): prepack/unpack round trip is lossless and byte-count preserving, forward and backward match a dequantized int8 float64 reference within the output dtype's rounding on every model cell and every table config, partial M/N tiles, `M = 1`, fp16 and bf16 outputs, and the shape heuristic's classes. The activation quantizer matches the reference recipe on every group-256 width in the shape tables (1024 through 248320, i.e. both the staged and the streamed rows), with and without the rotation and the column-scale fold, in fp16 and bf16.
- `tests/test_convrot_int8.py` (sd-scripts): the packed path agrees with the plain path to 1e-3 (forward and input gradient), the packed bytes round-trip to the released `[N, K]` form exactly, and the load-time prepack keeps `weight.shape`.
- `bench/benchmark_int8_convrot.py` is the tuning loop: kernel-only, configs interleaved, medians, one line per `(dir, M, N, K)` cell, `--verbose` for the full config matrix. It runs the shared grid in `bench/cells.py` (106 cells over 20 groups), the same one the fp8 bench and `bench/benchmark_kernels.py` use, and `--groups` selects a subset in all three.
- `python -m bench.benchmark_kernels --backends fp8,int8_fwd,int8_bwd,bf16` is the cross-kernel report: one cell per row, every backend interleaved inside it, `bf16` as the control. It replaces the one-off script the earlier versions of this comparison used. The protocol rotates the provider order one place per round rather than reversing it, which is what makes an interleaved comparison fair: with a reversal the middle position belongs to the same provider every round, and two providers running the same kernel measured 1.6% apart that way (the fp8 plan's protocol section has the experiment).
- `~/tmp/feather_ops/dual/compare_fp8_int8.py`: the cross-kernel comparison of this kernel against the fp8 one on 30 shared shapes, `torch.mm` as the control. `int8_attrs.cpp` / `lds_attrs.cpp` print the per-instantiation LDS, VGPR, scratch and the occupancy model used above.
- `~/tmp/feather_ops/int8_refit_sweep.py` + `i8_refit_*.json`: the balanced sixteen-tile sweep (eight providers, eight rounds) the schedule was refitted on. `int8_fit_tree.py` fits a rule tree to it offline. `int8_verify4.py` / `int8_verify5.py` are the paired re-tests with a duplicate provider. `int8_selector_ab.py` is the whole-matrix old-vs-new A/B (136 cases, both directions).
