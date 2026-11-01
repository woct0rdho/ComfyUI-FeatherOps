# gfx1151 HIP Prepacked-B fp8 Matmul Kernel Optimization Plan

Scope: `feather_ops::scaled_mm` in `~/feather-ops` (`csrc/hip_kernel.cu`, `feather_ops/scaled_mm.py`), the forward `Y[M, N] = (X[M, K] fp16|bf16) @ (W[N, K] fp8)^T * scale + bias` against a weight prepacked once as `(K/16, N, 16)`. Forward only: there is no weight-gradient or input-gradient direction in this operator, and none is planned. gfx1151 only, one device, one kernel family.

## Signature And Contract

| operand | shape | dtype | layout |
|---|---|---|---|
| `a` (activations) | `[M, K]` | fp16 or bf16 (widened to fp16 while staging) | `k` contiguous |
| `b_prepacked` (weight) | `[K_pad/16, N, 16]`, `K_pad = ceil(K/128)*128` | fp8 e5m2 (the tuned flavor) or e4m3 (supported) | one 16-byte `k` run per `n`, zero-padded and prepacked once at load |
| `scale` | 1 element | fp32 | per-tensor |
| `bias` | `[N]` | output dtype | optional |
| `c` (output) | `[M, N]` | fp16 or bf16 | `n` contiguous, `out=` variant |

The kernel is an implicit GEMM over `M = tokens`, `N = output width`, `K = reduction width`, with `v_wmma_f32_16x16x16_{f16,bf16}_w32` on wave32 and 8 fp32 accumulators per lane per tile. B is staged as raw fp8 and decoded to half in registers, because gfx11 has no fp8 arithmetic and no `global_load_lds`.

The prepack keeps the weight physically as fp8: one pass at load time writes `(K_pad/16, N, 16)` and the kernel consumes those bytes as a raw `uint8_t*`. No dequantized copy of the weight exists at runtime, and the weight scale stays a per-tensor scalar applied in the epilogue.

Metric: kernel-only milliseconds from `bench/benchmark_scaled_mm_hip.py` and `~/tmp/feather_ops/fp8_shapes/`, converted to `2*M*N*K/t`, measured steady state and against `torch.mm` on an already-dequantized weight. Medians of per-round medians, per `~/evotensile/docs/noisy_measurements.md`. Session drift on this machine is ±3%. Accuracy gate for every change: relative L2 <= 1e-2 and max abs <= 1.0 against the float32 `feather_ops.naive.scaled_mm_naive`, both fp8 flavours and both output dtypes, which is the gate `tests/test_scaled_mm_hip.py` already enforces.

## The Shape Problem

The shape grid is the forward cell set of `int8_convrot_prepacked_fwd_bwd_plan.md`: 17 `(N, K)` families over the M values each model runs, 139 cells. Auditing all of them against the gate the runtime then applied (`autotune._config_compatible`: `M % block_m == 0`, `N % block_n == 0`, `K % chunk_k == 0`, and `block_m, block_n >= 64` for large shapes):

| class | cells | share of forward flops | what happened before this pass |
|---|---:|---:|---|
| unavailable at the operator boundary | 39 | 21.8% | no config passed the gate, so `scaled_mm_hip` raised |
| runnable, weight-traffic-capped | 100 | 78.2% | works, but the tallest tile the heuristic can return is 128 rows, which caps the weight-traffic roof at 50.1 TFLOPS |
| runnable, compute-bound | 0 | 0% | none |

The rejected set was not exotic: every Krea 2 cell at `M = 4112` and `M = 9232`, every Qwen-Image 2.1 cell at both of those M, every Z-Image cell at `M = 4128/9248`, every MOSS cell, H3's ragged `M = 5251`, all six Qwen3.5 text-encoder cells whose width is 4304, and H3's `adaln_proj` at `M = 1`. Those are the cells the image models generate at. The M values that did work (4096, 4608, 5376, 9216, 9728, 25344, 512, 2048) are the ones that happen to be 64- or 128-aligned. This audit is what the tails were built for, and it is kept because it is the reason the shape contract is what it is.

The integration used to pay to avoid that error: `comfy_feather/ops.py` padded `M` up to a multiple of 16/32/64/128 (128 above 256 rows) before the call and sliced the output afterwards. So the production cost of a ragged M was a full activation copy plus the wasted MMA rows, not a fallback: on `4112 x 24576 x 4096` the pad is 34 MB read plus 35 MB written, ~0.33 ms against a ~2.1 ms GEMM, and 112 of 4224 rows are dead MMA (2.7%). The tails in the kernel removed both, and the padding block is gone from the wrapper. The N and K tails had no workaround at all: a 4304-wide layer passed the quantizer (`K % 16 == 0` was the only check) and then raised at the GEMM, which is why those six cells were a hard error rather than a slow path.

Why this is a cliff and not a slope. Each M tile re-reads its whole B panel (`block_n * K` bytes). The N-fastest grid lets L2 carry a panel from one M pass to the next only while the whole weight is a few MB, so for the 118 cells whose weight is larger than that the weight traffic of a cell is `ceil(M / block_m) * N * K`. Dividing flops by that traffic gives a roof that depends on the tile height alone:

| block_m | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| weight-traffic roof (TFLOPS at 212 GB/s) | 6.8 | 13.6 | 27.2 | 54.3 | 108.6 |

That formula is a worst case rather than a law, and the measurements say where it stops applying: `4096 x 2048 x 2048` has a 4 MB weight that stays in L2 across M passes and runs at 30.7 TFLOPS on a 16-row tile, 4.5x what the 6.8 row predicts, and the 1.3 MB `4096 x 1152 x 1152` runs at 40.2 on a 64-row tile. Where the weight does not fit, though, the row applies: a cell forced to 16 rows is not 5% slower, it is roofed at 6.8, and that is the shape of the ragged-M problem. `4112 = 16 * 257` and `9232 = 16 * 577` leave nothing but 16-row tiles once the large-shape floor rejects anything under 64 rows, and the floor then rejects those too, which is why those cells are unavailable rather than merely slow. The 4304-wide text-encoder cells and the `M = 1` adaln cell fail the same gate for the same reason.

Splitting the audit by weight footprint makes the priority order explicit: the 118 cells with `N * K > 8 MB` carry 99.5% of the forward flops and 33 of them are unavailable (21.8% of that class's flops), while the 21 cells with a small weight carry 0.5% of the flops and lose 6 cells to the same gate. For the 100 cells that ran, the formula explains the plateau they sat on: big-weight cells capped at a 128-row tile, i.e. a 50.1 TFLOPS roof against a 54.0 compute ceiling, measuring 41-42 under the cold-L2 protocol of the time.

Three secondary findings from the same audit:
- The two prepack helpers take opposite orientations: `feather_ops.scaled_mm.prepack_b_for_scaled_mm` takes the GEMM's B (`[K, N]`), `comfy_feather.quant.prepack_transpose` takes the weight (`[N, K]`). Nothing was wrong with either, but with no docstring and a square-only bench the difference was invisible, and a non-square run of the bench fed the helper the wrong way round. Both have docstrings now and every non-square test exercises one of them.
- `tests/test_scaled_mm_hip.py` only generated square, divisible shapes, so no test exercised a tail, a non-square weight, or a 16-row tile. That is now the bulk of the file.
- ComfyUI's fp8 path (`comfy_feather/ops.py`) cast `x` to fp16 and the result back to `dtype_orig` on every call, because the kernel only accepted fp16 A. On `4112 x 24576 x 4096` that was 67 MB of extra read+write for the activation cast, ~0.3 ms against a ~2.1 ms GEMM, on top of the M-padding copy above. The cast is gone.

## Current Results

Kernel-only, steady state (warmup, then the median per-launch time over back-to-back launches, which is how a model calls the same GEMM across steps), bf16 activations, e5m2 weights, bf16 output, the tile the heuristic picks. The cells are the shared grid in `bench/cells.py` (see the measurement protocol below): 106 cells over 12 model families, three resolution-ladder groups and five special-regime groups. `torch` is `torch.mm` against an already-dequantized bf16 weight, i.e. the optimistic control: it does not pay for the dequantization. Whole table: `bench/benchmark_scaled_mm_hip.py`.

| grid group | cells | TFLOPS | vs torch (mean / median) |
|---|---:|---:|---:|
| `ladder-wide` (6144x6144 at 512^2..2048^2) | 6 | 32.5-39.4 (mean 37.2) | 1.83x / 1.89x |
| `ladder-narrow` (1536x6144 over the same ladder) | 6 | 26.8-32.8 (mean 28.8) | 1.65x / 1.66x |
| `ladder-small` (1024x5120 over the same ladder) | 5 | 27.7-28.4 (mean 28.2) | 1.73x / 1.72x |
| `sdxl` | 7 | 32.6-41.4 (mean 37.7) | 2.02x / 2.16x |
| `anima` | 8 | 26.4-37.0 (mean 30.9) | 1.80x / 1.78x |
| `qwen21` | 4 | 37.5-43.3 (mean 40.3) | 1.91x / 1.92x |
| `qwen2_1` | 7 | 0.8-39.8 (mean 29.3) | 2.17x / 1.93x |
| `zimage` | 6 | 20.6-41.7 (mean 36.4) | 2.20x / 2.06x |
| `krea2` | 5 | 7.0-39.1 (mean 28.7) | 2.37x / 2.25x |
| `ideogram4` | 5 | 33.4-41.0 (mean 37.1) | 1.67x / 1.51x |
| `klein9b` | 7 | 33.4-42.5 (mean 39.1) | 2.04x / 2.04x |
| `wan` | 5 | 39.3-43.8 (mean 41.4) | 2.13x / 2.10x |
| `ltx2_3` | 4 | 34.9-42.4 (mean 38.2) | 2.11x / 2.04x |
| `h3` | 9 | 15.6-43.3 (mean 37.2) | 2.21x / 1.90x |
| `moss` | 5 | 35.1-40.2 (mean 38.4) | 1.93x / 1.96x |
| `short-prompt` | 4 | 12.9-32.8 (mean 20.7) | 2.71x / 2.42x |
| `text-16` | 1 | 8.6 | 1.77x |
| `pad-256` | 5 | 31.5-36.5 (mean 34.4) | 3.61x / 3.62x |
| `text-1152` | 5 | 32.7-38.5 (mean 35.5) | 2.38x / 2.28x |
| `weight-stream` | 2 | 0.4-42.2 (mean 21.3) | 2.08x / 2.08x |
| all | 106 | mean 34.0, median 36.6 | mean 2.11x, median 1.95x |

Readings:
- No cell is slower than `torch.mm` any more, and the median is 1.95x. The two cells that used to lose were the 16- and 2-row ones: the tiny-M class shipped a 16x256 tile, and 16x64 with the refill overlapped is 1.6-2.7x faster there (see the heuristic below), which is what took `qwen2_1` from 0.41 to 0.8 TF and `pad-256` from 2.97x to 3.61x over torch.
- The image-size ladder is flat above 1024^2 on the wide box (4112/6416/9232/16400 rows all land at 36-40 TFLOPS) and drops at 512^2 (32.5), where the weight stops fitting L2 across M tiles. The narrow 1536-wide box sits 8-11 TFLOPS below it at every resolution: that is the class the int8 kernel wins.
- `weight-stream` is two different things at the same 2.08x: `512 x 248320 x 4096` at 42.0 TFLOPS (the weight is 1 GB, so 393 GB/s of streaming would be needed to starve it - it is compute bound) and `1 x 96768 x 2688` at 0.45 (227 GB/s, the DRAM roof for a single 260 MB pass).
- `short-prompt` (mean 20.7) is the worst class by ratio-free absolute speed: 32-128 rows make the weight traffic dominate, and there is nothing tile-side left to buy.
- `ideogram4`'s 1.51x median is the 53248-long contraction against 4608 outputs: a 245 MB weight read once, so the control is unusually good there rather than the kernel being slow (33-41 TFLOPS).
- The grid is a redesign of the old per-family table (139 cells over 10 families), which under-sampled exactly the regimes above. It is smaller (110 cells), covers 12 models plus the resolution ladder, and both kernels now run it.

## Where This Kernel Stands Against The Int8 One

Both kernels are `[M, K] x [N, K]` GEMMs on the same part and both issue the same `16x16x16` WMMA, so achieved TFLOPS are directly comparable. `python -m bench.benchmark_kernels --backends fp8,int8_fwd,int8_bwd,bf16` measures all of them on the shared grid in one interleaved pass per cell, so the columns are paired. The numbers below are the 110-cell grid, one session:

| backend | mean TFLOPS | median | what it computes |
|---|---:|---:|---|
| `fp8` | 32.82 | 35.05 | bf16 activations x fp8e5m2 weight, per-tensor scale |
| `int8_fwd` | 32.87 | 35.29 | int8 x int8 ConvRot, per-row and per-channel scales |
| `int8_bwd` | 34.02 | 37.63 | the same kernel in the input-gradient direction |
| `bf16` | 16.49 | 18.07 | `torch.mm` on a dequantized bf16 weight (the control) |

`fp8 / int8_fwd`: median 1.011, mean 1.008, fp8 faster on 55 of 106 cells. The per-group medians are where the structure is:

| fp8 wins | | fp8 loses | |
|---|---:|---|---:|
| `sdxl` (1280-wide) | 1.105 | `ladder-narrow` (1536-wide) | 0.847 |
| `wan` (5120-wide video) | 1.093 | `anima` (1024/2048-wide at short K) | 0.853 |
| `text-1152` | 1.070 | `krea2` (its 16384-wide and 6912 cells) | 0.949 |
| `h3` (5376-wide) | 1.062 | `ladder-wide` (6144x6144 ladder) | 0.965 |
| `pad-256` (129-256 rows) | 1.037 | `qwen2_1` (4096-wide) | 0.973 |

Nothing here decides "the faster kernel": the median is a tie and the sign flips with the group, which is predictable from the cell but not from the dtype. The two kernels are limited by different resources, and that is the useful part:
- The int8 kernel is MMA-issue-bound. 37-42 TFLOPS is 73-78% of the 54 TF issue ceiling, and it sits at 35-47% of the DRAM roof on the wide cells of that list (63-79% on the narrow ones). Its 128/256-row tiles re-read the weight 17-33 times, and its A operand is one byte per element.
- This kernel is DRAM-traffic-bound. 168-183 GB/s on the cells it wins = 79-86% of the 212 GB/s roof, and 97-121% of the modelled roof on the cells it loses (modelled as `ceil(M/block_m) * N * K` for the weight plus one pass each over A and C, which over-counts where a block's A panel fits L2). Its shipped tiles are 64 or 128 rows, so the weight is re-read 33-65 times per call, and its A operand is two bytes.

So fp8 wins where the cell is large enough for the 128x256 tile to reach 72-80% of the issue ceiling, and it loses wherever its tile choice is forced down to 64 rows (narrow `N`, short `K`) or the grid is starved (`M <= 256`). The one cell worth reading twice is `4112 x 1536 x 6144`: int8 runs its `128x256` schedule there while this kernel's heuristic drops to `64x128` (its own `128x256` measures 20.7 on that cell against int8's 32.6 on the same box), so the gap is not the tile choice - it is decode plus the 2-byte A plus a 33 KB LDS footprint against int8's 25 KB.

What that implies for the plan: the tall tier at `u = 4` (below) is the way to attack the narrow cells from inside this kernel, and the A-operand width is the only way to close the rest, since it is a property of the operand format and not of the schedule. A per-layer router between the two kernels is not available at runtime: they consume different weight formats (fp8e5m2 versus the 4x4 int8 ConvRot prepack) and different activations, so the choice is made at the checkpoint and integration level, not in the kernel selector.

## The Measurement Protocol

The kernels had four different timing recipes (`do_bench` in seven benches, an interleaved median in the conv2d bench, a local event pair in the int8 bench, an ad-hoc one in the fp8 bench). They now share one, `bench/protocol.py`. It lives next to the benchmarks rather than in the installed package, and the benchmarks are run as `python -m bench.<module>` from the repository root:
- Steady state, not cold. `do_bench` flushes L2 between iterations, so every launch is timed cold. That is right for a one-shot streaming kernel and wrong for a model that calls the same weight repeatedly: `4096 x 2048 x 2048` with `(1,8,4,8,2)` measures 22.9 TFLOPS cold and 32.9 warm. `--cold` keeps the flushed regime for the cases where it is the real one instead of making it the default.
- Adaptive iteration count. One round times `iters` back-to-back launches inside one event pair, with `iters = target_ms / measured_launch` clamped to `[1, 32]`, so a 20 us kernel and a 200 ms kernel both get a full round instead of a fixed `warmup=25ms`.
- Providers interleaved, order rotated. Every candidate on a case (torch, the kernel, a tile, a schedule) is timed round-robin with the order rotated one place per round, so the sustained-clock drift of this machine hits all of them alike and each candidate visits each position equally. Rotation rather than reversal matters: reversing leaves the middle position owned by the same provider in every round, and a three-provider run measured two providers that launch the same kernel 1.6% apart because one of them always sat there (52 cells, mean -2.05%, median -1.62%). With rotation the same experiment has no sign. The ratio column is a paired median of per-round ratios, not a ratio of medians measured minutes apart.
- A spread column. p20/p80 over rounds, so a 1% difference can be told from noise. Typical spread on these benchmarks is 0.2-1.2%.
- One shape table, shared by every kernel. `bench/cells.py` holds the grid and both GEMM benches consume it, so `--groups` selects the same cells in either. It is a deliberate sample rather than a cross product, with three jobs: generalize to the workload in `input_shapes_tn.md` (every model family contributes two or more of its real `(out, in)` pairs at its real row bands), cover the image-size ladder (`ladder-*`: 512^2, 768^2, 1024^2, 1280^2, 1536^2, 2048^2 on a wide, a narrow and a small-grid box, which the workload itself only exercises at 1024^2 and 1536^2), and hit the regimes the plans call special (`pad-256` launch-fill, `short-prompt`, `text-1152`, `weight-stream`, ragged rows against their aligned neighbours, outputs past 16384, contractions past 8192). 106 cells, down from the 139-cell per-family table it replaces.
- Three reports, one grid. `python -m bench.benchmark_scaled_mm_hip` is this kernel against `torch.mm`, `python -m bench.benchmark_int8_convrot` is the int8 kernel in both directions, and `python -m bench.benchmark_kernels --backends fp8,int8_fwd,int8_bwd,bf16` puts all of them on the same cell in one interleaved pass. `--tiles` turns the tile choice into an extra provider set in the fp8 bench, `--square` keeps the `8192^3` regression guard, `--json` writes the per-round samples.

Every number in this document comes from that protocol. Results from before it are only quoted where they were re-measured, and are marked as pre-pass where they were not.

## Kernel Design

### Tails: masked launch, plus a tail-free fast path

The kernel owns the whole shape contract, so no caller pads anything:
- `M >= 1`, any value. The grid rounds up. The A staging clamps out-of-range rows to `M - 1` (in-bounds memory, garbage products). The epilogue drops the rows it is not asked for.
- `N % 16 == 0` is required (a whole 16-column tile is the store granularity, so the epilogue only tests `tile_n_base < N`). B staging clamps the column the same way.
- `K % 16 == 0` is required and the prepacked weight is zero-padded to a multiple of 128, so the last chunk of a ragged K reads zeros instead of running off the end. A clamped A times a zero B row is zero, so no staging site needs a branch for the K tail at all.

Two launches come out of one kernel template: when `M % block_m == 0 && N % block_n == 0 && K % chunk_k == 0` the host picks the `kTailFree` instantiation, whose code is exactly the pre-tail kernel (no clamps, no masks). Everything else takes the masked one. The two are measured against each other per cell, so the cheap path does not silently disappear.

What the tails cost where they are real: `4112` against `4224` on the same `(6144, 6144)` cell is 34.7 vs 35.0 TFLOPS, i.e. under 1%. The 3-7% that the ragged cells give up against their aligned neighbours is the dead rows of the last tile (112 of 4224 at `block_m = 128`), not the masking.

Padded prepack also removed a piece of the old design: the first attempt zero-filled the tail stages in LDS after the load, which put a dynamic loop inside the hot path and cost 5-20% of the kernel on cells where it never ran. Padding the weight once at pack time was free.

### Tile set: what the sweep actually picks

`SCALED_MM_TILES` instantiates 31 tile boxes (the 28 the autotuner searched plus `64x64 (2,2,2,2,2)`, `256x128 (4,4,2,4,2)` and `256x256 (4,4,2,4,4)`), four of which also exist with `prefetch` for 35 rows in total. The measured winners over 139 cells:

| tile | config | prefetch | wins on |
|---|---|---|---|
| 128x256 | `(1,8,4,8,2)`, u=4 | no | the wide-weight, wide-output bulk: H3, Qwen/Klein `24576`/`36864`, Krea `16384`, Z-Image `11520`/`10240`, text-encoder `25600`, `248320` |
| 64x128 | `(1,4,2,4,2)` | yes | mid weights with `N <= 4096`, every `1152`-wide text-encoder cell, long-K narrow-output cells |
| 128x128 | `(2,2,2,4,4)` | no | 512-row cells with `N` in 3072-6144 |
| 128x256 | `(2,4,4,4,2)` | no | 512-row cells with `N <= 2560` |
| 32x128 | `(1,2,4,1,2)` | yes | `N <= 1024` |
| 32x32 | `(1,2,2,2,2)` | yes | 512-row, `N <= 1280` |
| 64x128 | `(1,4,2,4,2)` | yes | 129-256 rows: 1.3-1.4x the 64-wide tile (`192/256 x 2560 x 2560`, `256 x 4096 x 4096`) |
| 16x64 | `(1,2,4,1,2)` | yes | `M <= 128` except the widest and longest cells: 1.2-2.7x the 64-wide and 16x256 tiles |
| 32x64 | `(1,2,2,2,2)` | yes | `M <= 128` with `N >= 8192`, and `M <= 256` with `N <= 4096, K >= 8192` |
| 16x256 | `(1,8,4,1,2)` | no | `M <= 16` with `N >= 16384` (the weight-streaming cell) |

The 256-row tiers have never won a cell, in either protocol - but every one of them is instantiated with `unroll_k = 2`, while the tiles that do win use `u = 4`. Measured on the same cells (`~/tmp/feather_ops/dual/tall_tiles.py`), `4112 x 6144 x 6144`: `128x256 u=4` 39.0, `256x128 u=2` 27.3 (-30%), `256x256 u=2` 26.0 (-33%). `P16` measured chunk 64 over chunk 32 at +15-35% for `N = 1024..4096`, i.e. exactly the narrow cells where the tall tile is supposed to pay, so the negative is confounded: the tier was rejected while carrying the handicap the ledger already documents. What is true is that 256 rows costs MMA waste on the M tail (5.8% at `M = 4112`, which the sibling int8 kernel removes with a separate tail launch) and a wider VREG/LDS footprint. They stay instantiated and the heuristic still never returns them. A `256x128`/`256x256` pair at `u = 4` is the untested corner (see What Remains).

### Heuristic instead of autotune

`select_scaled_mm_config(M, N, K, a_dtype, b_dtype, out_dtype)` in `feather_ops/scaled_mm.py`, fitted to the 139-cell sweep, in the same style as `select_int8_config`: range rules on `(M, N, K)` so a new image size lands in the right class without a per-model table. The kernel keeps no shape heuristic of its own: it takes the tile as six integers (five tile parameters plus `prefetch`) and the only branch it still makes is `tail_free`, which is a property of the extents (`M`, `N`, `K` divide the tile) rather than tuned data.

| condition | tile | prefetch |
|---|---|---|
| `M <= 16` | 16x64, 16x256 when `N >= 16384` | yes |
| `M <= 128` | 16x64, 32x64 when `N >= 8192` | yes |
| `M <= 256` | 64x128, 32x64 when `N <= 4096` and `K >= 8192` | yes |
| `M <= 512`, `N <= 2560` | 64x128 | yes |
| `M <= 512`, `N <= 6144` | 64x128 if `K <= 8192`, 128x128 if `K <= 16384`, else 128x256 | yes |
| `M <= 512`, `N <= 12288` | 64x128 | yes |
| `M <= 512`, `N > 12288` | 128x256 | no |
| `M > 512`, `N <= 1024` | 16x64 | yes |
| `M > 512`, `1280 < N <= 1792` and `M <= 4608` | 16x64 | yes |
| `M > 512`, `1025 <= N <= 1280` | 64x128 | yes |
| `M > 512`, `N <= 2560`, `K <= 2048`, `M <= 4096` | 128x256 (`2,4,4,4,2`) | no |
| `M > 512`, `N <= 4096` or weight `<= 8 MB` | 64x128 | yes |
| else | 128x256 (`1,8,4,8,2`) | no |

Two classes are worth reading in the data rather than the rules. The 16-row tiles exist because at 1024 and 1536 output width a 64-wide N panel over a 64- or 128-row M panel leaves too few blocks to fill 160 SIMDs, and a 16-row tile is the only way to raise the block count without shrinking the N panel below 64. Past 4608 rows the grid is already large and the tall tile's weight traffic wins again, which is why the 1536 row carries an `M` bound. The `128x256` at `N <= 2560, K <= 2048` is the short-contraction case where the weight fits in L2, so the tile height is what buys the time and the 4-warps-in-M variant has the best register/occupancy mix.

Fit quality, measured three ways. Against the per-cell best of the 31 baseline tiles the pre-refit rule was 1.35% behind (median 0.9%, worst 25.7%). Against the ten tiles of the refit sweep, including both `prefetch` states, it was 3.42% behind. The refit below closes most of that: over the same ten-tile sweep the shipped rule now sums to 1.8% behind the per-cell best, and the GPU A/B of the two rules (139 cells, paired, order rotated) measures +1.65% summed and +2.16% mean per cell, with 26 cells better and 5 worse.

`prefetch` is a property of the tile, not of the cell, and that is a simplification rather than a fit: the same 64x128 box with the refill overlapped is +10.2% at the median over its 50 cells but -11% on `4112 x 1536 x 6144` and -4% on two long-K cells, so a cell-level condition would recover a few percent on the losers. It is left as one flag per tile because the per-cell fit needs a model of why those cells differ, not another threshold. The refit instead moved those cells to a different box.

What the fit is not: a proof. The rules are coarse on purpose, several 3-9% misses sit on cells whose neighbours prefer the other side of a boundary (`3840`-wide outputs, `4096`-wide with `K = 12288`), and the differences between the top two tiles on a cell are usually inside the run-to-run spread. The harness to re-fit is `python -m bench.benchmark_scaled_mm_hip --tiles` plus the sweep in `~/tmp/feather_ops/fp8_shapes/`.

### Python API, allocation, and `torch.compile`

`scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype, tile=None)` is the whole public surface. It picks the tile, allocates `out` in Python, and calls `torch.ops.feather_ops.scaled_mm.default(a, b_prepacked, scale, bias, out, *tile)`. The op returns nothing. Deleted with the autotuner: `_configured_op`, `_autotuned_op`, both fake kernels, `register_custom_op_autotuning`, `old_autotune`, the 28-config list, `autotune._config_compatible`, and the `torch.compiler.is_compiling()` branch. This is the shape `mm_hipblaslt.py` and `conv2d.py` already had, and `tests/test_scaled_mm_hip_compile.py` checks that `torch.compile(fullgraph=True)` at `dynamic=False` and `dynamic=True` matches eager on non-square, ragged shapes.

The autotuner is not missed: it resolved differences at its own noise floor (0.1-2.3% against the best fixed tile), and it cost 3.5 s on the first call of every new `(M, N, K)`, which for image models means every new resolution.

### Prepack

`prepack_b_for_scaled_mm(b)` takes the GEMM's B, `[K, N]`, and writes `[K_pad/16, N, 16]` with `K_pad` rounded up to 128. `comfy_feather/quant.py::prepack_transpose` is the same permutation for a `[N, K]` weight and pads the same way. The two orientations are the one place where it is easy to lose an hour, so both have a docstring now and the kernel checks the padded row count.

### bf16 activations: widen while staging, and do not use the bf16 MMA

The A operand accepts fp16 or bf16. bf16 is widened to fp16 in the staging path (2 ops per pair of halves), so the whole body - decode, WMMA, epilogue - stays the fp16 one. Measured on five cells and four tiles (`~/tmp/feather_ops/fp8_shapes/ab_a_dtype.py`): the staging cast is within 1% of a native fp16 activation everywhere, while the native bf16 MMA is 1-25% slower on the cells that matter (`5376 x 21504 x 5376`: 41.15 with the cast against 37.32 with the bf16 MMA. `2048 x 248320 x 4096`: 40.66 against 38.12), because e5m2 -> bf16 decode costs a perm plus three ALU per pair where e5m2 -> fp16 is the perm alone. The native bf16 instantiation was deleted, which halved the kernel count.

`comfy_feather/ops.py` no longer casts `x` to fp16, so a bf16 model now hands its activations straight to the kernel.

### Everything else, unchanged

The compute body is the same one that was profiled to death: same A layout and swizzle, same B staging, same register-tiled WMMA loop, same C-shuffle epilogue, same `s_setprio 1` around the refill. The `if constexpr` axes that were added are tail-free, A dtype, the output dtype parameter, and `prefetch` (the refill of the next chunk issued into registers before the WMMAs and stored into the same LDS buffer after the barrier that publishes the reads).

## Rejected Or Closed, And What Was Tried In This Pass

Closed with measurements from this work:

| tried | result | why it is gone |
|---|---|---|
| native bf16 WMMA (`v_wmma_..._bf16`) instead of the staging cast | 1-25% slower on five cells * four tiles | e5m2 -> bf16 decode is dearer than e5m2 -> fp16, and the fp16 body wins outright |
| runtime output dtype (one instantiation fewer) | 5-20% slower on aligned cells | the per-element branch in the epilogue grew the code and, more importantly, changed the loop's register/schedule layout. Back to a template parameter |
| K-tail guard inside the staging loader (`if constexpr`, then a runtime select) | 5-20% slower even on shapes that never take the branch | a branch or select around the global load breaks the unrolled, branch-free staging. Replaced by padding the prepacked weight to 128 |
| zero-filling the tail stages in LDS inside the loop | same 5-20% | a dynamic loop in the hot path. The padding removed the need |
| `__syncthreads`-free single-buffer refill (the pre-tail loop) | unchanged, kept | nothing to win |
| 256-row tiles (`256x128`, `256x256`) | never won a cell. The `u = 2` rows are 30-34% behind, and the `u = 4` retest (8 and 16 warps) is still 8% behind on average | the chunk size was a real handicap (`u = 4` recovers most of it: 27.3 -> 36.5 for `256x128`), but even rectified the tier loses to the short tiles everywhere the heuristic can reach, because the weight passes it saves are already amortized by L2 at 128 rows (T7, T15, T18) |

Still closed from the previous campaign, with the evidence intact: direct-write decode staging (P1, +0.15% only), `cvt_pk_f16_fp8` (P2, gfx1250-only), `s_setprio` variants (P4/P7), stage4 and `stages > 2` as an LDS-buffer count (P8), hoisting the `v_perm` literals into VGPRs (P15, -2.7%), `kBPad`/`kCPad` padding stays removed (P17), M-tile grouping for B-panel reuse (0.5-2% of N-major, i.e. noise), and split-K / stream-K (net-negative on every starved cell of the sibling kernel).

Closed in this pass:
- Overlap/prefetch (P10, P22, P23). The refill now overlaps with the WMMAs on the tiles that pay for it, and the earlier rejections are explained. Two designs were built and measured per tile over all 139 cells with the shipped protocol: a register image of the next chunk stored into the same LDS buffer after a barrier (no extra LDS), and the int8 kernel's two-LDS-buffer form (one barrier per chunk). The second is dominated: it doubles LDS, which halves the workgroups resident on the WGP's 128 KB, and it never beat the first by more than noise. The first is now the `prefetch` field of four tiles (`64x128`, `32x64`, `16x64`, `64x64`) and wins by +7.7% mean on the 64x128 tiles (50 cells), +14.6% on 32x64, +23.2% on 16x64 and +17.7..+53.1% on 64x64 (five small-M shapes), while losing 15% on 128x256 and 83% on 128x128, so it is a per-tile choice like `unroll_k` rather than a switch. What closed the question is that the win tracks how much of a chunk is refill: a 64x128 tile buys 64 flops per byte of chunk payload, a 128x256 tile 131, and only the short tile has enough exposed latency to recover. The same pass found a compiler artifact on one instantiation: `(2,2,2,2,2)` with `prefetch` and fp16 activations asked for 55296 B of LDS against its 6144-byte union, because `AMDGPUPromoteAlloca` promoted a 384-byte per-thread array there. Disabling that promotion (`tools/hip_flags.py`) fixes it, worth +19% on those four instantiations (`T16`).
- The peak correction. The old "hard cap 45.7 / we are at 96%" framing is retracted. The measured issue ceiling is 54.0 TFLOPS. The 128x256 tile reaches 80% of it on the H3 cells (43.2), the class means are 32-41, and the cells the heuristic brings down to 64 rows are DRAM-bound at 79-97% of the 212 GB/s roof instead. The remaining gap is therefore two different gaps, traffic on the short tiles and issue efficiency on the tall ones, not a single unknown item.

## What Remains

Ranked by measured or estimated gain, with the mechanism and the gate each item has to pass:
- The A-operand width. On the cells the int8 kernel wins, its A is one byte per element where this kernel's is two: on `4112 x 1536 x 6144` that is 25 MB against 50 MB per pass over 12-24 passes. Quantizing activations to fp8 per row and decoding both operands halves it. There is no fp8 WMMA on gfx1151 (`cvt_pk_f16_fp8` is gfx1250-only), so this buys traffic and decode only, and it changes the numerics of the A side. Gate: the accuracy harness plus the traffic model. It is an integration decision before it is a kernel one. This is the only item left with a measured double-digit gap behind it, and it is the one that explains that gap.
- The remaining heuristic gap, 1.8% of summed TFLOPS after the refit. The residue is concentrated in the wide cells where the best box flips non-monotonically: `4096 x 8192 x 2048` wants the pipelined 64x128 (+16.6%) while `5376 x 5376 x 7168` and `25344 x 5376 x 7168` both want 128x256. A boundary there needs a model of the tile's traffic, not another threshold. The per-cell data for it is the ten-tile sweep in `~/tmp/feather_ops/dual/refit_*.json`. Fourteen of the 31 baseline boxes still never win a cell, and dropping them would cut 14 of the 35 instantiations out of the dispatch chain.
- M-tail split launch. The grid rounds `M` up to `block_m`, so `4112` at 128 rows computes 0.9 dead tiles (1.2%) and the ragged training rows (5251, 4352) 1-2%. The int8 kernel recovers its 5.8% (its tile is 256 rows) with a second launch over the remainder. Here the win is small enough to sit behind the items above.
- e4m3 stays deferred: it costs 7.5% more than e5m2 in decode, the integration quantizes to e5m2 itself, and it only becomes interesting if native fp8 e4m3 checkpoints are routed here, at which point it becomes a class in the selector rather than its own kernel.

## History: The One-Shape Campaign

The previous version of this file tuned `8192^3` and only `8192^3`, with an autotuner, and retired ideas when a one-shape comparison inside ±3% went the wrong way. Its ledger is kept because the individual measurements are still valid. Only the conclusions drawn from them about headroom are not.

| ID | Keep? | Change | Key result |
|---|---|---|---|
| P0 | KEEP baseline | initial e5m2 baseline | `8192^3` 42.86 TFLOPS |
| P1 | REJECT | direct-write decode staging removal | +0.15% |
| P2 | REJECT | `cvt_pk_f16_fp8` builtin path | requires gfx1250 |
| P3 | REOPEN | add tile `(1,8,2,16,1)` | +0.24%, unstable at small N. An unbalanced tile, not the tall tier |
| P4 | REJECT | remove load-phase `s_setprio` | large-N regression |
| P5 | KEEP analysis | no-scale/no-bias ablation | 0.13%: epilogue is not a bottleneck |
| P6 | KEEP analysis | overlap decomposition | 2-3% exposed overlap headroom |
| P7 | REJECT | `s_setprio 2` | non-robust |
| P8 | REJECT config, KEEP fix | stage4 candidate + stage-index fix | correctness fixed, no speedup |
| P9 | KEEP | prepack output kept fp8, wrapper infers the flavour | 43.77 TFLOPS, cleaner API |
| P10 | REOPEN | stage2 split-phase A/B prefetch | 42.04 TFLOPS vs 43.77 control, inside noise |
| P11-P13 | REOPEN as one experiment | interleaved / B-only / volatile prefetch | correctness failures, not measurements |
| P14 | KEEP simplification | remove the swizzle toggle, stages=2 only | baseline maintained |
| PCS, TT | KEEP analysis | PC sampling and thread tracing | LDS-read 50%, convert 15%, sync 14% of samples. A 3,500-cycle `vmcnt` bubble per chunk |
| P15 | REJECT | hoist `v_perm` literals to VGPRs | 42.62 TFLOPS, -2.7%, `vmcnt` stalls +18% |
| P16 | KEEP | `unroll_k` 4/8 | flat at `8192^3` (43.1), +15-35% at `N = 1024..4096` |
| P17 | KEEP | remove kBPad/kCPad | bank conflicts 0.3%, LDS capacity back |
| P18 | KEEP | A-WSGR region heuristic refresh | replay: square 6/7, broad 12/14 exact, residual <= 0.33 TFLOPS |
| P19-P21 | KEEP analysis | refill / decode / fragment-load ablations at `8192^3` | refill 2%, decode 1.7%, fragment loads 6.1% |
| P22 | REOPEN | A-only global->VGPR prefetch sweep | autotuned target deltas +0.20 / -14.73 / +1.56 / -0.59% |
| P23 | REOPEN | B-only prefetch sweep | -1.15 / -20.91 / -3.23 / -4.26% vs the clean baseline |
| N1 | KEEP analysis | 139-cell shape audit | 39 unavailable (21.8% of flops), 100 traffic-capped, 0 compute-bound |
| N2 | KEEP analysis | pure-WMMA issue ceiling | 54.04 TF (f16) / 54.06 TF (bf16), retires the 59.4 peak |
| N3 | KEEP analysis | e5m2 vs e4m3 decode | 42.46 vs 39.27 TF, same cell and config |
| T1 | KEEP | masked M/N/K tails + `kTailFree` fast path for divisible shapes | all 139 cells run. Ragged M costs 3-7%, not 5x |
| T2 | KEEP | prepacked weight zero-padded to 128, no K guard in the kernel | removed the only branch in the staging path |
| T3 | REJECT | K-tail zero-fill in LDS inside the loop, and a select in the loader | 5-20% slower even on shapes that never take the branch |
| T4 | REJECT | native bf16 WMMA | 1-25% slower than the fp16 body with a staging cast, on 5 cells x 4 tiles |
| T5 | REJECT | runtime output dtype | 5-20% slower. Back to a template parameter |
| T6 | KEEP | static heuristic (139-cell sweep) replacing the autotuner | mean 1.35% behind the per-cell best. 0 s of tuning |
| T7 | REJECT, qualified | 256-row tiles | never won a cell, but every 256-row row is `unroll_k = 2` and loses 30-34% to the 128x256 `u = 4`. The tier was rejected while carrying the chunk-size handicap `P16` documents. Re-open as `u = 4` (see What Remains) |
| T8 | KEEP | steady-state timing protocol | cold-L2 `do_bench` understated small-weight cells by up to 30% |
| T9 | KEEP | tile table generated from `feather_ops/tiles.py` | instantiation list and dispatch cannot drift. `tests/test_tiles.py` checks the header |
| T10 | REJECT then DELETE | A-WSGR, exposed as a Python gate first | 34.5 -> 30.8 TF on `512x1152x1152`, -2.6% on `512x4096x4096`, +-0.15% on the big cells: no positive region under the current tile set, so the template axis and its 48 instantiations were removed |
| T11 | KEEP | heuristic behind a mutating `custom_op` | 1 Dynamo graph for 6 shape classes instead of 2, `empty_strided` still in the Inductor output, +0.1% runtime |
| T12 | KEEP | `prefetch` tile field: next chunk into registers during the WMMAs, same LDS buffer | +7.7% mean on 64x128 (50 cells), +14.6% on 32x64, +23.2% on 16x64, +17.7..+53.1% on 64x64. -15.1% on 128x256, -82.9% on 128x128, so it is per tile. 139-cell mean 36.82 TF, 1.96x `torch.mm` |
| T13 | REJECT then DELETE | two LDS buffers, one barrier per chunk (the int8 design) | -28.3% mean on 128x256 and equal to T12 where T12 wins: halving the resident workgroups costs more than the second barrier |
| T14 | KEEP analysis | fp8-vs-int8 head to head on 30 shared cells, with the per-tile resource report (`lds_attrs.cpp`, `int8_attrs.cpp`) | median 0.998, fp8 wins 15/30. This kernel is DRAM-bound (79-86% of 212 GB/s where it wins), the int8 one is issue-bound (73-78% of the 54 TF ceiling, 35-47% of the roof). The sign follows tile height and A-operand width, not dtype |
| T15 | KEEP analysis | 256-row tiles measured against `128x256 u=4` | 256x128 `u=2` -30%, 256x256 `u=2` -33%, so the tall tier's negative is confounded by its chunk size. Turned T7 into an open item |
| T16 | FIXED | LDS size audit per instantiation | `(2,2,2,2,2)` + `prefetch` with fp16 A was allocated 55296 B against its 6144 B union: `AMDGPUPromoteAlloca` promoted a 384 B per-thread alloca into LDS (128 threads x 384 B). `-mllvm -disable-promote-alloca-to-lds` (`tools/hip_flags.py`) removes it, and `tools/check_lds.py` - which compiles with the same flags - reports all 560 instantiations at exactly their union. The flag is neutral for bf16 (no bf16 instantiation was promoted) and worth +19% on the fp16 variants of that tile |
| T17 | KEEP | selector refit on the ten-tile sweep | +1.65% summed / +2.16% mean per cell over 139 cells, 26 better and 5 worse. The 19 cells that change tile gain 14.4% on their own sum. The 1536-wide class alone went from 19-30 to 26-32 TFLOPS |
| T18 | REJECT | tall tier at `unroll_k = 4` (`256x128`, `256x256`, 8 and 16 warps) | `u=4` recovers the chunk-size handicap (`256x128` 27.3 -> 36.5 on `4112x6144x6144`) but still loses 8-19% on average to the short tiles and only wins cells where the heuristic was already wrong. Pruned from the table |
| T19 | KEEP | rotation instead of reversal in `bench/protocol.py` | removes a 1.6% middle-position bias measured on two identical providers in a three-provider run |
| T20 | CLOSE, unsubstantiated | the 5-16% scheduling gap against the pre-tail kernel | not reproducible from the tree: `4096 x 2048 x 2048` measures 30.0 with the same tile as the old build's 35.7, and the other cell of the claim (`4096 x 8192 x 2048`) reaches the old number (38.7) with `(1,4,2,4,2)` plus the overlap. The claim came from a pre-protocol session. A rebuild of the pre-tail kernel would be needed to pursue it |
| T22 | KEEP | the shared grid (`bench/cells.py`) replacing two per-family lists | 106 cells instead of 139 + 68, both GEMM benches and a cross-kernel one consume it, and the regimes the plans call special are now cells rather than prose. It found the tiny-M class on its first run |
| T23 | KEEP | small-M tile classes refitted: 16x64 below 128 rows, 32x64 where the grid needs it, 64x128 at 129-256 | 1.6-2.7x on the `M <= 16` cells (16x256 was the shipped tile), 1.2-2.4x at `M <= 128` (64x64 was), 1.3-1.4x at 129-256. No grid cell is slower than `torch.mm` any more and the median rose 1.90x -> 1.95x |
| T24 | KEEP with workaround | the fp16 LDS allocation audit (`tools/check_lds.py`) | checks all 560 instantiations against the union their source asks for. It is what found the 55296-byte fp16 variant. With 64x64 no longer returned by the heuristic that instantiation is unreachable from the selector, so the dtype routing was dropped |
| T25 | REJECT (int8 kernel) | LDS stage padding, 0 / 32 / 48 bytes against the shipped 16 | measured on the int8 kernel, where the bank-conflict count is non-zero: 16 B is already the best of the four (0 is -1.3%, 32 and 48 are -0.4% and -0.5%). This kernel's conflicts are already 0, so the pad is not a knob left on the table for either |
| T21 | CLOSE, superseded | a cell-level `prefetch` condition | the refit moved the cells it existed for (the 1536-wide class is now a 16-row tile, `4112 x 1536 x 6144` is 27.3 instead of 23.6) and the residue is inside the cell-to-cell spread |

Durable findings that still hold, kept so they do not have to be relearned:
- The e5m2 unpack is a front-end/issue cost rather than a VALU cost, driven by 96-bit `v_perm_b32` with inline selectors, and "delete `v_perm`" rewrites fail because their replacement sequences cost the same or more.
- Moving the decode into the load phase is the wrong direction because it shifts work into a less-overlapped region and grows the payload.
- `V_PK_*`/VOP3P packed-half instructions are not VOPD-eligible so decode rewrites cannot rely on dual issue.
- `ds_permute_b32` only permutes within a wave32 and cannot broadcast fragments across waves.
- The epilogue is not a bottleneck.
- Rigid inline ASM blocks interleaving and worsens `vmcnt` hiding.
- Hot-loop address generation was already mostly hoisted into precomputed bases plus compile-time offsets by an earlier ASM analysis of this kernel.
- Finer chunking loses to control overhead unless something else removes the control (P16 buy came from chunk size coupled with fewer `vmcnt` stalls, not from the chunking itself).
- The overlap win is a function of refill share, not of shape class: 64 flops per byte of chunk payload (64x128) recovers 8-10%, 131 (128x256) loses 15%. That is why `prefetch` is a tile property and why the two-LDS-buffer form never won.
- The two kernels are limited by different resources on the same silicon (traffic here, MMA issue there), so "which kernel is faster" is a question about the cell, not about the dtype. Any future comparison has to be per cell and in one session.
- Occupancy is set by LDS as often as by VGPRs on this part: 8 KB of LDS per workgroup is 16 resident workgroups per WGP, 16 KB is 8, 32 KB is 4. A tile that grows its staging by 8 KB can lose a third of its occupancy while its VGPR count stays flat.
- An instantiation's real LDS footprint is `group_segment_fixed_size` in the code object, not the union size in the source. Read it from the built artifact for every tile. When they disagree the usual cause is a promoted alloca rather than the source: `AMDGPUPromoteAlloca` charges `workgroup_size * size` of LDS for a per-thread array it decides to promote, which is how one fp16 variant came to ask for 55296 B for 5632 B of code. `-mllvm -disable-promote-alloca-to-lds` is the cure here. `tools/check_lds.py` is the guard.
- Interleaving only pairs fairly if every provider visits every position. Reversing the order per round leaves the middle position to the same provider, which measured two identical providers 1.6% apart. Rotating one place per round does not.
- A tile family that loses can be losing for a reason you chose: the 256-row tier was priced while all its instantiations had half the chunk size of the winners. Rectify the handicap before retiring a family (`T15`, `T18`).

What the refill ablation adds, and why the overlap item is config-gated rather than a single switch: the exposed cost of the global->LDS refill is strongly config-dependent at `8192^3`-scale cells. Wide `repeat_m=1` `(1,8,4,1,2)` at `2048^3` is clearly B-dominated (`reuse_a` 4%, `reuse_b` 14%, `reuse_ab` 15%). The balanced `(1,4,4,4,2)` at `4096^3` splits evenly (both 6%). The tall WSGR-on `(4,2,2,2,4)` at `(8192, 2048, 12288)` is A-dominated (`reuse_a` 15%, `reuse_b` 8%), and the large square winner `(2,4,4,4,4)` at `8192^3` gains only 2%, i.e. it is not refill-dominated. So dual buffering should be expected to pay on the 2048-6144-class cells and on the tall tiers, and to do little on the largest square cell.

One old conclusion needs qualifying rather than deleting: "the winner path is not DRAM-bandwidth-limited" is true of `8192^3`, but only just (B traffic there is 4.3 GB, ~83% of the 212 GB/s roof at 24.4 ms), and it is false for most of the real cells, where the 128-row tile ceiling puts the weight-traffic roof at 50.1 TFLOPS. The two statements are the same identity evaluated at different tile heights. Neither contradicts the other.

## Verification

- `tests/test_scaled_mm_hip.py`: forward against `scaled_mm_naive` for both fp8 flavours, both output dtypes and both activation dtypes. `M` in 1, 17, 128, 256, 512, 2048, 4096, 4111, 4112, 4304 and 5376. Every one of the 35 tiles (31 boxes, 4 of them also with `prefetch`) on a shape with an M, an N and a K tail (`67 x 272 x 304`). A check that every tile the heuristic can return is instantiated.
- `tests/test_scaled_mm_hip_compile.py`: `torch.compile(fullgraph=True, dynamic=False)` and `dynamic=True` over an eager-equal check, non-square, with the allocation visible as `empty_strided_cuda` in the Inductor output.
- `bench/cells.py`: the shared grid, 106 cells over 20 groups, consumed by every bench below. `--groups` selects a subset by group name in all of them.
- `python -m bench.benchmark_scaled_mm_hip`: the whole grid against `torch.mm`, `--tiles` to sweep tiles as providers (six integers per tile: `warps_m, warps_n, unroll_k, repeat_m, repeat_n, prefetch`), `--square` as the `8192^3` regression guard, `--json` for the per-round samples.
- `python -m bench.benchmark_kernels --backends fp8,int8_fwd,int8_bwd,bf16`: every kernel on the same cell, one interleaved pass, `bf16` as the control. This is the command the cross-kernel section quotes.
- `bench/protocol.py`: the benchmark protocol every benchmark in `bench/` uses (steady state, adaptive iteration count, interleaved providers, paired ratios, p20/p80 spread). It is not part of the installed package, and `bench/` and `tests/` do not import each other.
- `~/tmp/feather_ops/fp8_shapes/sweep_short.py` + `cell_sweep_steady.csv`: the 139-cell tile sweep the heuristic was fitted on. `verify_fit.py` reports the per-cell loss. It times launches the same way `bench/protocol.py` does.
- `~/tmp/feather_ops/dual/`: `compare_fp8_int8.py` (the 30-cell cross-kernel comparison), `dual_fit.py` + `fit_*.json` (the per-cell `prefetch` A/B over all 139 cells), `refit_sweep.py` + `refit_*.json` (the ten-tile sweep the selector was refitted on), `selector_ab.py` (the paired old-vs-new selector A/B), `fit_tree.py` / `score_rule.py` (offline rule fitting against the sweep), `tall_tiles.py` / `tall_fit.py` (the tall-tier retest), `probe_fp16_lds.py` / `fp16_route.py` (the fp16 LDS artifact), `lds_attrs.cpp` / `int8_attrs.cpp` (per-instantiation LDS/VGPR/scratch and the occupancy model), `isa_mix.py`, `lds_offsets.py`, `count_scratch.py`, `order_bias.py`.
- `python tools/check_lds.py`: compiles the kernel to gfx1151 assembly with the shipped flags and checks every instantiation's `group_segment_fixed_size` against the LDS union its source asks for (560 instantiations, ~40 s, so it is not part of the test suite). It is what found the fp16 55296-byte allocation, and it prints "every allocation matches" on the shipped build.
- `tools/hip_flags.py`: the device backend flags shared by `setup.py` and the audit tool, so the two cannot drift. Today it carries `-mllvm -disable-promote-alloca-to-lds` and the reason it is there.
- kernel-only suites: 695 passed. `~/ComfyUI-FeatherOps` 18 passed.

## Files And Artifacts

- `~/feather-ops/csrc/hip_kernel.cu`: kernel, tail-free/masked instantiations, the `FEATHER_SCALED_MM_TILES` table, `scaled_mm` out= op.
- `~/feather-ops/feather_ops/scaled_mm.py`: public entry point, `select_scaled_mm_config`, padded prepack helper.
- `~/feather-ops/feather_ops/autotune.py`: unchanged, still used by `mm_fp16` and the Triton kernels. The scaled_mm path no longer imports it.
- `~/feather-ops/tests/test_scaled_mm_hip.py`, `tests/test_scaled_mm_hip_compile.py`, `tests/test_tiles.py`.
- `~/feather-ops/tools/`: `gen_tiles.py` (renders `csrc/feather_tiles.h` from `feather_ops/tiles.py`), `hip_flags.py` (the device backend flags), `check_lds.py` (the per-instantiation LDS audit).
- `~/feather-ops/bench/protocol.py`: the shared benchmark protocol, including the rotating provider order.
- `~/feather-ops/bench/cells.py` and `bench/benchmark_kernels.py`: the shape grid and the cross-kernel report.
- `~/ComfyUI-FeatherOps/comfy_feather/ops.py` and `quant.py`: the cast removal, measured separately in that repo.
- `~/tmp/feather_ops/fp8_shapes/`: `cells.py`, `compat_audit.py` + `compat_audit.csv`, `bench_cells.py` + `cells_baseline.csv`, `dtype_probe.py`, `wmma_peak_f16.cu`.
- `~/feather-ops/csrc/hip_kernel_int8.cu` and `int8_convrot_prepacked_fwd_bwd_plan.md`: the sibling kernel this one is measured against. Its occupancy model and prepack layout are the reference for the resource numbers here.
- `~/tmp/feather_ops/mixed_precision_analysis/`: the P6/P19-P23 ablations, sweeps and PC-sampling plots referenced by the history table.
- `~/rdna35-isa-markdown/`, `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst`: instruction and compiler reference.
