# Conv2d 3x3 Stride-1 Plan

## Scope And API

In scope: inference-only `conv2d`, `kernel=3x3`, `stride=1`, `padding=1`, `dilation=1`, `groups=1`, contiguous 4D NCHW input and output, bf16 or fp16.

Out of scope, delegated to torch: `groups>1`, dilation>1, NHWC, transposed conv, 1D/3D conv, fp32 (no WMMA on RDNA3.5). fp16 3x3 stride 1 is delegated to torch too, on purpose: MIOpen's Winograd kernels run 1.03-1.59x faster than this kernel on every 3x3 s1 shape measured, and `select_backend` is the one place that decides (fp16 always, while the input is at most a 512 MiB tensor). bf16 stays here, because MIOpen has no bf16 Winograd and its best bf16 path is 1.5x slower. Winograd of our own is rejected for both dtypes (`Rejected Optimizations`). 3x3 stride 2 and 1x1 are delegated for now and are candidates for later work.

The Python API is a drop-in for `F.conv2d` and `nn.Conv2d`. Everything else is internal: the tile table, the row-count rule and the fp16 delegation window live inside `conv3x3_prepacked` and `select_backend`, so the package exports no selector tables and no tuning knobs beyond the explicit `block_m`/`block_n`/`rows` arguments of `conv3x3_prepacked`. The package root is exactly `conv2d`, `Conv2d`, `conv3x3_prepacked`, `prepack_conv2d_weight_3x3` plus the int8/fp8 matmul entry points. `select_backend` stays importable from `feather_ops.conv2d` because the ComfyUI patch needs the decision before it packs a weight (a packed copy in that integration is a module buffer that follows offload, which a global cache cannot do).

```python
def prepack_conv2d_weight_3x3(weight: Tensor) -> Tensor:
    """[C_out, C_in, 3, 3] -> [ceil(C_in/16) * 18, C_out, 8] in the kernel's k order."""


def conv3x3_prepacked(x: Tensor, w_packed: Tensor, bias: Tensor | None = None, block_m: int | None = None, block_n: int | None = None, rows: int | None = None) -> Tensor:
    """3x3, stride 1, padding 1 over a prepacked weight. No fall back, no extra checking."""


# module-level, not package API - the one helper the ComfyUI patch shares:
#   feather_ops.conv2d.select_backend(dtype, c_in, c_out, height, width) -> 'kernel' | 'miopen' | 'torch'


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    """Drop-in for F.conv2d. Uses the kernel when in scope, else falls back to F.conv2d."""


class Conv2d(nn.Conv2d):
    """Drop-in for nn.Conv2d. state_dict keeps the standard [C_out, C_in, 3, 3] weight."""
```

The rules that keep it torch-like, all covered by tests:

| rule | how |
| --- | --- |
| never mutates `input` or `weight`, always returns a fresh contiguous NCHW tensor | the kernel writes into a `torch.empty` it owns |
| unsupported input falls back instead of raising | `F.conv2d`, bit-identical to a plain call (the tests assert `torch.equal`). No strict mode, the fallback is the public contract |
| kernel path agrees with torch to bf16 arithmetic, not bit-exactly | reduction order differs by construction. The tests gate on relative L2 against an fp64 reference (4e-3 bf16, 6e-4 fp16) |
| standard `state_dict` | `nn.Conv2d` keys unchanged. The prepack is `register_buffer(..., persistent=False)` |
| `load_state_dict`, `.to()`, `.half()`, `torch.save` keep working | cache keyed on `(data_ptr, _version, dtype, device, shape)` via `weight_cache_key`, and the functional cache additionally holds a weak reference to the weight and only hits on the same tensor object, because a freed address can be reused by a new tensor that also starts at version 0 (the 14x6 dispatch matrix test caught exactly that). `_load_from_state_dict` clears it |
| `torch.compile` sees one op, and the output buffer belongs to the compiler | the native op is an out= op (`Tensor(a!) out`, returns `()`), so it needs no custom-op wrapper and no fake kernel: there is no output metadata to infer. `conv3x3_prepacked` allocates the buffer, so Inductor emits `buf0 = empty_strided_cuda(...)` and passes it in, and with a consumer it reuses it (`buf3 = buf0; del buf0  # reuse`). `torch.export` and `torch.compile(fullgraph=True)` both work |
| fp16 always goes to MIOpen, and it says so | `feather_ops.conv2d.select_backend()` is capability + policy: fp16 3x3 s1 is delegated to `F.conv2d` while the input is at most a 512 MiB tensor (MIOpen wins 1.03-1.59x there, and loses beyond it: 0.98x at `384->384 1024^2`, 0.97x at `128->128 2048^2`). The delegated call forces `torch.backends.cudnn.flags(enabled=True, benchmark=False)`, because ComfyUI sets `cudnn.enabled = False` on AMD and that would route the call to im2col, and because exhaustive search is the autotune we do not want. ComfyUI reports it as its own fallback reason (`fp16 is delegated to MIOpen`) and counts it separately |
| no silent behaviour change on patched weights | `weight_function` / `bias_function` (LoRA, low-VRAM) and a lazily attached `weight_lowvram_function` (LoCon) force ComfyUI's own path. `comfy_cast_weights` does not, because that flag is just "weights come from the cast copy" and `_feather_pack` builds the prepack from exactly that copy |
| shapes with no tile go to MIOpen instead of im2col | a VAE's first conv (`C_in % 8`) and last one (`C_out` below the 16-row tile minimum) have no tile, so the choice there is MIOpen against im2col, and im2col is what torch does. On the Anima/Qwen 2D VAE (`3->96` and `96->3`) that is 19.4 -> 5.7 ms for the last conv at 1024^2 and 4.0 -> 3.6 ms for the first: decode 324.6 -> 314.0 ms (3.3%), 512^2 80.6 -> 76.7 ms (4.8%), 2048^2 1645.3 -> 1621.1 ms. fp16 keeps MIOpen at every size measured. bf16 is capped at about a billion input elements, where its implicit-GEMM path collapses (`96->3` at 3584^2: 903 ms against im2col's 226) |

Supported-shape requirement table (anything else falls back, and the fallback is deliberate, not an accident):

| requirement | why |
| --- | --- |
| `C_in % 8 == 0` | the padded channel plane re-reads a real plane and relies on the prepacked zero weight rows. Needs whole 8-channel planes. `C_in` need not be a multiple of 16 (the prepack zero pads) |
| `C_out % block_m == 0` for some `block_m` in `{128, 96, 64, 48, 32, 16}` | a tile that divides exactly avoids recomputing overlapped rows. The two-row tiles additionally want `C_out % 128 == 0` (at `block_n = 128`) or `C_out % 64 == 0` (at `block_n = 256`), and fall back to the single-row form when they cannot divide |
| width `>= 32` | narrower images (the LTX 2.3 mel VAE, `W=16..66`) are out of scope |
| contiguous 4D NCHW, bf16/fp16, stride 1, padding 1, dilation 1, `groups=1`, `padding_mode="zeros"` | the kernel's contract |

## Why This Operator

From `docs/input_shapes_conv2d_vae.md` (9 image models at 1024^2/1536^2. 2048^2 only multiplies `N` by 4):

| family | distinct `(M,K)` | `(M,K,N)` triples | FLOP share per model |
| --- | ---: | ---: | ---: |
| 3x3 s1 | 39 | 86 | 95.9-97.3 % |
| 3x3 s2 | 7 | 14 | 0.9-2.3 % |
| 1x1 | 25 | 52 | 1.0-1.8 % |

- FLOP concentration of the 3x3 s1 family at 1024^2: top-6 `(M,K)` = 71.5 %, top-20 = 98.7 %.
- Top-6 shapes: `(512,4608) (384,3456) (256,2304) (192,1728) (128,1152) (96,864)`.
- `M` set: `{3,4,8,16,32,64,96,128,144,192,256,288,384,512,576,768,1152}`. `K = 9*C_in` set: `{27,36,144,288,576,864,1152,1296,1728,2304,2592,3456,4608,5184,6912,10368}`.
- `N` set: 1024^2 -> `{4096,16384,65536,262144,1048576}`, 1536^2 -> `{9216,36864,147456,589824,2359296}`, 2048^2 -> `{16384,65536,262144,1048576,4194304}`.
- Every in-scope image width is a multiple of 32 (`W ∈ {64,96,128,192,256,384,512,768,1024,1536,2048}`), which is what makes a single 256-wide column tile usable on most layers.
- Roofline: AI 287-3938 FLOP/B against a ~175 FLOP/B ridge point, so the family is unambiguously compute bound. The kernel must maximise WMMA issue efficiency, not bandwidth.
- Machine facts the budget below leans on: DRAM copy 212 GB/s, DRAM read 241 GB/s, measured L2 traffic in this kernel ~230 GB/s (mostly hits), and LDS 128 B/cycle per CU (256 B/cycle per WGP, the unit that matters in WGP mode).

Why the baseline is so slow: ComfyUI turns cuDNN/MIOpen off on RDNA3+ (`comfy/model_management.py:489`), so every VAE conv runs `im2col_kernel` + hipBLASLt `addmm`. A traced SDXL VAE decode spent 27 % of its time in `im2col_kernel` alone (trace in `~/tmp/feather_ops/conv_vae_shapes/`).

Why not simply use MIOpen: its winning solver here is CK `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<256,128,128,64,...>` on NHWGC plus three NCHW <-> NHWC layout-transform kernels:

| case | CK math kernel | transform + launch overhead | effective |
| --- | ---: | ---: | ---: |
| 3x3 96 512^2 | 1.40 ms (31.1 TF) | 1.67 ms of transposes (54 %) | 14.3 TF |
| 3x3 384 256^2 | 6.09 ms (28.6 TF) | 1.69 ms (22 %) | 22.3 TF |
| 3x3 1152 128^2 | ~12 ms (31.3 TF) | small | 33.1 TF |
| 3x3 s2 384 257^2 | GEMM 1.17 ms (37.2 TF) | im2col 0.84 ms (43 %) | 22.7 TF |

Its math is already close to what a fused NCHW kernel achieves. What it loses is the layout round trip, which is exactly what a native-NCHW kernel avoids. One trap when quoting MIOpen: its first-call LGBM choice for `384@256^2` was 7.80 ms (22.3 TF). 6.24-6.65 ms is what its find-db holds after one exhaustive search (`torch.backends.cudnn.benchmark=True`). A fresh install is 25 % slower than the numbers above, which is what a stock user sees with `COMFYUI_ENABLE_MIOPEN=1` on the first run.

## Kernel Design

Formulation. An implicit GEMM over the 3x3 window:

```text
M = C_out,  N = H_out*W_out (row-major pixel order),  K = 9*C_in
A[m, k] = weight[m, c, kh, kw],   k = (group, tap, c%16),  group = c/16,  tap = kh*3 + kw
B[k, n] = x[c, h + kh - 1, w + kw - 1]   (zero outside the image)
C[m, n] = out[m, h, w]
```

`C` is exactly the GEMM's row-major `M*N` output and `N` is NCHW's pixel order, so the existing C-shuffle epilogue of `hip_kernel_fp16.cu` writes straight into `[C_out, H, W]` - no transposed copy anywhere in the pipeline.

Prepack layout. The weight is permuted once per weight version into `[ceil(C_in/16)*18, C_out, 8]` with `k = group*144 + tap*16 + c%16`. Two consequences matter:
- the nine taps of one 16-channel group are one contiguous `k` range, so a group's weight slice is 18 contiguous rows of 16-byte vectors.
- every A fragment is a single 16-byte `uint4` at an address of the form `base + m*16`, i.e. one `ds_read_b128`/`global_load_dwordx4` per lane with the tap/plane offset as an immediate (verified in the ISA: `ds_load_b128 v[104:107], v80 offset:36864`).

`C_in` is zero padded up to a multiple of 16 in the prepack. Those weight rows are exactly zero, which is what lets a `C_in % 16 != 0` layer work without any per-channel masking.

Halo tile in LDS. Activations are not prepacked. Per 16-channel group a block stages a `3 * (BW+2)` halo tile as two 8-channel planes, `lds[k0][row][col][8]`, where the 8 channels of a pixel are contiguous. All nine taps then read that one tile at a constant 16-byte-granule offset `(kh*(BW+2) + kw)*16 B`, so the input is fetched once per block instead of once per tap, and `padding=1` is simply the zero-staged halo row/column. The layout is conflict-free for both the fragment reads and the staging stores because every access is 16 bytes wide and 16 bytes apart across lanes.

The global->LDS step reads NCHW coalesced along `w` and transposes: 8 scalar 2-byte loads per pixel-group (each one coalesced 64 B wave access, so the bytes are efficient even though the instruction count is 8x the minimum) into one staged `uint4` per 8-channel plane. This was the plan's "pick one of three transpose mechanisms by profiling" menu. None of the three was needed - the scalar-load form measured well enough that the register-transpose variants stayed on paper (see `Rejected`).

One WMMA k-step = 16 channels = two 8-channel planes. The two planes are concatenated into one `v16i16` fragment per lane before the WMMA. Getting this wrong (one plane per k-step) silently leaves half of every fragment undefined. The tap-by-tap delta test catches it in milliseconds.

Two output rows per block. `rows = 2` stages a 4-row halo (`kHaloRows = kRows + 2`) for two output rows: the input re-read drops from 3.05x per output row to 2.02x at `block_n = 256` (3.10x -> 2.06x at 128), and the per-block staging, barrier and epilogue work is halved. The tap loop then reads halo row `output row + kh`, so output row 1's B fragments are the same staged tile shifted by one row: two register fragment sets per tap instead of one, and twice the accumulators. That only fits where the single-row tile is M-narrow, so two instantiations exist: `128x128 x2` (16 waves, 8 tiles/wave, 64 accumulator floats, 167 VGPRs) and `64x256 x2` (16 waves, 8 tiles/wave, 64 floats, 174 VGPRs). Odd `H` costs nothing: the last block's spare row is computed and not stored.

Activation staging reads 8 channels of one pixel with 8 scalar 2-byte loads and writes one 16-byte LDS vector, because the channel stride is `H*W` while the tile wants channel-contiguous planes. On the two-row tiles those loads are paired instead: one 32-bit load fetches the same channel of two horizontally adjacent pixels (columns 1..`block_n` are even-aligned because `block_n` divides an even width. The two halo columns stay scalar), halving the VMEM instruction count for 3-9 fewer VGPRs. It measured +0.7..+2.4 % there and -0.8 % on the 128x256 single-row tile, so it is gated by tile.

Loop structure. Per group: stage A (36.9 KB at `block_m = 128`, all nine taps) and B (24.8 KB at `block_n = 256`), barrier, nine taps of `wmma`, barrier. Tap inner and channel group outer is what makes the halo tile reusable nine times. With the gated prefetch, the next group's A and B are fetched into VGPRs before the current group's compute and committed after the existing barrier, so the two-barrier structure and the LDS footprint are unchanged.

Epilogue. The C-shuffle through the `sh` union (a 16x16 tile per wave, 8 `ds_write_b16` then two `ds_read_b128` + one 16-byte global store per tile), with the bias added in fp32 before the conversion, matching what `addmm(bias, ...)` does. The bias value of a row is shared by the 16 lanes of a subgroup, so the epilogue issues one broadcast load per row.

Tiles and dispatch. 38 instantiations, `block_m ∈ {128, 96, 64, 48, 32, 16}` x `block_n ∈ {256, 192, 128, 96, 64, 32}` at one output row per block plus two two-row tiles, both bf16 and fp16, and the prefetch variant for every one of them (76 + 76 device kernels). The tile is a `(warps_m, warps_n, repeat_m, repeat_n, rows)` tuple from `feather_ops/tiles.py`, which `tools/gen_tiles.py` compiles into both the instantiation list and the dispatch chain, so a tile the heuristic can return is always a compiled kernel. `rows` and the prefetch switch are runtime arguments. LDS is `(144*block_m + 16*kHaloRows*(block_n + 2))*2` bytes - concretely 61.6 KB for 128x256 with 16 waves, 156 VGPRs (162 for the prefetch variant) and no spills, and 53.5 KB for 128x128 with two rows and 167 VGPRs.

`conv3x3_prepacked` picks the tile with the same divisibility rules as before (largest `block_m` dividing `c_out`, then the largest `block_n` dividing `width`, then the largest `block_n` that merely fits - the last column block then overlaps its neighbour and recomputes those columns with an identical reduction order, so the duplicate stores are bit-identical) and then decides the row count: two rows for a 128-wide image whose `c_out` is a multiple of 128, two rows for `c_out % 64 == 0` but `% 128 != 0` at `block_n = 256` (192, 576), one row otherwise. That rule is internal (`_select_config`). The only public selector is `select_backend`, which answers kernel / MIOpen / torch for a dtype and shape. The C++ dispatch maps `(block_m, block_n, rows)` to the first instantiation with that shape, and routes an odd width (only reachable through an explicit tile) to the single-row form, which produces the same result by construction.

Two deliberate simplifications. Overlapping last tiles instead of masked tails (an `M` or `W` tail costs a bit of redundant compute, never a second code path), and int64 addressing only where it is needed (`K*N` reaches 4.3e10 at 2048^2, so the activation and output offsets are int64 while the in-tile arithmetic stays 32-bit).

## Current Results

All numbers are bf16, batch 1, on this machine (Radeon 8060S, gfx1151, 20 WGPs at 2.9 GHz), with `torch` measured in the same process as the control.

Two measurement caveats apply to every number here, both measured (see `Open Questions And Risks`). The sustained clock depends on what ran before: the same binary and shape measures 13.02 ms cold and 14.05 ms after two minutes of sustained load, an 8 % band that an independent matmul tracks within 0.5 %, so alternatives are always interleaved rather than run in sequence. And the kernel is 5-13 % faster on degenerate data (zero activations or weights), so every measurement here uses random data.

Per-convolution on the 14-shape 3x3 s1 grid, medians of 3 interleaved rounds of 12 samples each, so every candidate on a shape meets the same clock state (`feather-ops/bench/benchmark_conv3x3.py`):

| shape | torch im2col | torch MIOpen | tile | this kernel | vs im2col | vs MIOpen |
| --- | --- | --- | --- | --- | ---: | ---: |
| 512->512 256^2 | 14.244 ms (21.71 TF) | 14.194 ms (21.79 TF) | 128x256x1 | 8.289 ms (37.31 TF) | 1.72x | 1.71x |
| 384->384 256^2 | 8.990 ms (19.35 TF) | 6.607 ms (26.33 TF) | 128x256x1 | 4.827 ms (36.03 TF) | 1.86x | 1.37x |
| 256->256 256^2 | 4.934 ms (15.67 TF) | 4.177 ms (18.51 TF) | 128x256x1 | 2.142 ms (36.10 TF) | 2.30x | 1.95x |
| 192->192 256^2 | 3.866 ms (11.25 TF) | 2.342 ms (18.57 TF) | 64x256x2 | 1.277 ms (34.06 TF) | 3.03x | 1.83x |
| 128->128 256^2 | 2.043 ms (9.46 TF) | 1.244 ms (15.54 TF) | 128x256x1 | 0.583 ms (33.13 TF) | 3.50x | 2.13x |
| 96->96 256^2 | 1.521 ms (7.15 TF) | 0.737 ms (14.76 TF) | 96x256x1 | 0.359 ms (30.29 TF) | 4.24x | 2.05x |
| 1152->1152 128^2 | 18.047 ms (21.69 TF) | 12.128 ms (32.27 TF) | 128x128x2 | 11.175 ms (35.02 TF) | 1.61x | 1.09x |
| 288->288 256^2 | 6.786 ms (14.42 TF) | 6.080 ms (16.09 TF) | 96x256x1 | 2.935 ms (33.34 TF) | 2.31x | 2.07x |
| 576->576 256^2 | 18.628 ms (21.01 TF) | 18.428 ms (21.24 TF) | 64x256x2 | 10.476 ms (37.36 TF) | 1.78x | 1.76x |
| 144->144 256^2 | 2.461 ms (9.94 TF) | 1.748 ms (14.00 TF) | 48x256x1 | 0.899 ms (27.22 TF) | 2.74x | 1.94x |
| 96->96 512^2 | 5.983 ms (7.27 TF) | 3.488 ms (12.47 TF) | 96x256x1 | 1.464 ms (29.71 TF) | 4.09x | 2.38x |
| 128->128 512^2 | 7.847 ms (9.85 TF) | 5.511 ms (14.03 TF) | 128x256x1 | 2.264 ms (34.15 TF) | 3.47x | 2.43x |
| 384->384 128^2 | 2.304 ms (18.88 TF) | 1.932 ms (22.51 TF) | 128x128x2 | 1.196 ms (36.36 TF) | 1.93x | 1.62x |
| 1152->1152 256^2 | 72.523 ms (21.59 TF) | 59.612 ms (26.26 TF) | 128x256x1 | 42.066 ms (37.22 TF) | 1.72x | 1.42x |
| total | 170.2 ms (19.21 TF) | 138.2 ms (23.65 TF) | | 90.0 ms (36.34 TF) | 1.89x | 1.54x |

The best shape reaches 37.4 TF and the widest layer 37.2 TF, 69 % of the 54 TF WMMA issue ceiling measured below, at 1.72-1.86x the path ComfyUI actually uses. Every shape is now a win over both deployment paths, including `1152->1152 128^2`, which used to be the one shape MIOpen won: two output rows per block took it from 29.3 to 35.0 TF against MIOpen's 32.3. The two-row tiles also carry 192 and 576 (34.1 and 37.4 TF), which used to be the weakest non-narrow shapes.

Tile-selection behaviour worth knowing (measured, not assumed):
- 128x256 is the best tile whenever it applies, +8-12 % over 128x128 and +35 % over 64x128, and it wins from `N = 65 536` to `N = 4 194 304`.
- `block_n` must divide the width, otherwise the overlap recomputes 25-33 % of the columns (128x192 on a 256-wide image costs 35 %: 23.1 TF vs 35.6 TF).
- `block_m` must divide `c_out`: for `c_out ∈ {192, 288, 576, 1152}` a 96-wide block beats a 128-wide one that would overlap, and for `c_out = 144` a 48-wide block beats 96 (27.7 vs 24.2 TF).
- At `N = 16 384` the 128-wide column tile wins simply because `W = 128 < 256`.

End-to-end through `comfy_feather.conv` (kernel on/off on the same VAE instance, 4 iterations, 1024^2):

| VAE | dtype | decode | encode |
| --- | --- | --- | --- |
| SDXL | bf16 | 817.9 -> 455.7 ms (1.79x), image rel 1.4e-3 | 397.9 -> 252.4 ms (1.58x) |
| Flux 2 small | fp16 (delegated to MIOpen) | 604.3 -> 287.1 ms (2.10x), image rel 3.9e-4 | 407.1 -> 238.4 ms (1.71x) |
| Qwen 2D (Anima, Qwen-Image 1, Krea 2) | bf16 | 565.1 -> 315.2 ms (1.79x), image rel 2.6e-3 | 320.9 -> 211.9 ms (1.51x) |
| Qwen-Image 2.1 (2D checkpoint) | bf16 | 1337.4 -> 851.8 ms (1.57x), image rel 1.9e-3 | 335.2 -> 225.1 ms (1.49x) |
| Wan 2.1 (3D VAE. Only its embedded 2D resamplers apply) | bf16 | 600.8 -> 546.5 ms (1.10x) | 354.4 -> 354.8 ms (1.00x) |

At 2048^2 the SDXL VAE decodes in 4670.9 -> 2523.9 ms (1.85x) and encodes in 2217.9 -> 1623.9 ms (1.37x), so the gain does not decay with resolution. Tiled decode (`VAEDecodeTiled`, which is what ComfyUI falls back to when the full decode does not fit) was verified separately at 1.83x with image rel 1.5e-3.

For reference on the decode rows, handing every conv back to MIOpen (`COMFYUI_ENABLE_MIOPEN=1`) measures 718 / 268 / 1169 ms for SDXL / Flux 2 small / Qwen-Image 2.1, i.e. 1.15x, 2.27x and 1.14x over im2col. That is why the fp16 row is a delegation and the bf16 rows are not: on the fp16 VAE MIOpen's gfx11 ASM Winograd (`ConvWinoFuryRxS<2-3>`) beats this kernel by 9-11 %, which moved Flux 2 small from 310.0 to 287.1 ms decode, while the bf16 rows stayed where they were.

Per convolution, fp16, on the same 14-shape grid (`~/tmp/feather_ops/conv2d_3x3/bench_wino.py`, interleaved, MIOpen with `MIOPEN_FIND_MODE=FAST`): this kernel 90.1 ms (36.3 TF) against MIOpen 78.4 ms (41.7 TF), i.e. 1.15x in total and 1.03-1.59x per shape. MIOpen's best shapes reach 47-52 TF effective, which only works because Winograd does 4/9 of the multiplies. With `MIOPEN_FIND_MODE=NORMAL` (MIOpen's own timed search, the autotune) it is worse: 94.1 ms in total, because the search picks a 54.6 ms solver for `1152->1152 256^2` where the heuristic picks a 39.9 ms one, so heuristic-only is both cheaper and better here.

Numerics, measured on the SDXL VAE. The meaningful comparison is against another bf16 implementation, not against fp32: two bf16 convs that differ only in summation order differ by ~2*2^-9 relative on every layer because the inputs are quantized.

| comparison | decode image | encode latent |
| --- | ---: | ---: |
| torch im2col vs torch MIOpen | 1.83e-3 | 7.15e-2 |
| torch im2col vs this kernel | 1.41e-3 | 4.62e-2 |

So the kernel is closer to torch than torch's own two paths are to each other, and the 5-7 % latent difference is amplification of bf16 noise through the encoder's cancellation-sensitive `conv_out` (a per-layer hook shows every conv at 2-7e-3 relative, i.e. the bf16 floor). fp16 VAEs are three orders of magnitude tighter (image rel 1.8e-4).

Measurement noise: absolute TFLOPS move by up to 8 % between runs with the GPU's thermal state, and the torch control moves with it, so the ratios are the stable quantity: 1.81x over im2col and 1.49-1.57x over MIOpen on the 14-shape grid across runs. The end-to-end numbers above are sequential (torch first, then the patched run), which costs the patched side the 2-5 % a warmed GPU gives away, so those ratios are conservative. All per-conv numbers are medians of 12 samples with warm-up excluded. The A/B decisions quoted below used 25 iterations.

## Profile: Where The Time Goes

The ceiling is 54 TF, not 59. `~/tmp/feather_ops/conv2d_3x3/wmma_peak.cu` runs a register-only WMMA loop (no LDS, no memory, no barriers) with 4/8/16 independent accumulators:

| workgroups per multiprocessor | waves=4, tiles=4 | 4,8 | 8,4 | 8,8 | 4,16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.5 TF | 52.2 TF | 50.4 TF | 52.5 TF | 52.4 TF |
| 2 | 51.8 TF | 54.1 TF | 52.0 TF | 54.0 TF | 53.9 TF |

A 16x16x16 WMMA retires every ~35 cycles per SIMD pair at 2.9 GHz, so 54 TF is the issue ceiling - not the 59.4 TF of the 32-cycle model, and not the 37 TF that the matmul kernels had been read as being capped at. The probe is also value-independent: with 1.0, 0, -1 and 0.0098 operand fills it measures 52.1-52.5 TF at one workgroup per MP and 53.8-54.0 TF at two (`wmma_peak_data.cu`), so the ceiling hides no data-dependent fast path. Two consequences: the conv at 27-37 TF is at 50-69 % of what the hardware can issue (69 % for the dominant 256-wide tiles, 50 % for the 48-wide `C_out = 144` case), and the same 30 % is available to the matmul kernels, which is why the findings here are reusable.

Stall taxonomy. Stochastic PC sampling on 384->384 256^2 (2322 samples, `rocprofv3 --pc-sampling-method stochastic --pc-sampling-interval 1048576`, raw CSV in `~/tmp/feather_ops/conv2d_3x3/pcs/`):

| stalled instruction | stall reason | share of samples |
| --- | --- | ---: |
| `ds_load_b128` (weight/activation fragments) | ALU dependency | 24.8 % |
| `v_wmma_f32_16x16x16_bf16` | arbiter not won (pipe busy: healthy) | 17.8 % |
| `s_barrier` | barrier wait | 15.8 % |
| `v_wmma_f32_16x16x16_bf16` | ALU dependency (operand just loaded) | 10.8 % |
| `v_wmma_f32_16x16x16_bf16` | waitcnt (LDS outstanding) | 5.7 % |
| 64-bit address VALU | arbiter not won | ~4 % |
| everything else | | ~21 % |

The generated code is not the problem: one base register per operand, immediate offsets, 12 fragment loads per tap, WMMA batches interleaved with the next loads. What costs is the dependency chain between those loads and their consumers.

Ablations on a lab copy of the kernel (`~/tmp/feather_ops/conv2d_3x3/conv3x3_lab.cu`, tile 128x256 unless noted):

| ablation | what it removes | effect |
| --- | --- | ---: |
| all group loads + epilogue, no WMMA | measures the load path alone | 1.56 ms vs 4.86 ms full (32 %) |
| next-group register prefetch | global refill latency | +3.5-7.9 % on every `block_n = 256` tile with `C_in >= 128`, +6.1 % on the 128x128 two-row tile, -8..-9 % on the 64x256 two-row tile |
| drop the second `__syncthreads()` per group | half the barriers | +1 % (0..+4 % across shapes) |
| reuse the tap-0 B fragment for all 9 taps | 44 % of all fragment loads | +1 % at 128x256, +5-8 % at 96x256 |
| fragment double-buffering by tap | LDS-to-use latency | -18 % |

Three conclusions that correct the naive reading:
- The 15.8 % barrier figure is not 15.8 % of wall time. Wave-level sampling counts a wave parked at a barrier as stalled while the other 15 waves of the workgroup keep issuing. Removing half the barriers buys 1-4 %, so barrier accounting is not worth attacking on its own. (This is why the multi-row tile dropped from "the big win" to a modest one.)
- The staging path costs 32 % in isolation but only 2-3 % exposed: it is nearly fully overlapped already, which is why the one change that attacks it directly (the register prefetch) is worth only 2-3 %.
- Fragment LDS reads are tile-dependent: at 128x256 (1.5 LDS instructions per WMMA) removing 44 % of them is worth ~1 %. At 96x256 (2.0 per WMMA) it is worth 5-8 %. Reducing fragment traffic only pays where the per-wave tile is small.

The kernel is therefore balanced: the LDS pipe (61.7 KB of staging stores plus ~880 KB of fragment reads per group per block, 20-27 % of the 256 B/cycle budget), the epilogue (65 KB of output per block plus its LDS round trip, ~5 %) and issue/operand waits each take a few percent. No single remaining change is worth more than ~5 %, and that is what the ranked list reflects.

## Accepted Optimizations

| optimization | what it buys | evidence |
| --- | --- | --- |
| Weight prepack into `[G*18, C_out, 8]`, `k = (group, tap, c%16)` | one 16-byte `uint4` per A fragment with an immediate tap/plane offset. No per-call layout work. Repack once per weight version per process | ISA shows `ds_load_b128 v[..], v80 offset:36864`. The runtime pack counter is exactly 1 per module across warm-up + 4 timed iterations |
| Halo tile in LDS, channel-contiguous 8-channel planes, taps as constant offsets | the input is read once per block instead of nine times. `padding=1` for free | 3x3 s1 runs at 29-38 TF versus 7-22 TF for im2col. The loader is 8 coalesced scalar loads + one 16-byte store per plane |
| In-place C-shuffle epilogue into NCHW | no transposed copy, no NHWC intermediate, `M*N` output written straight to `[C_out, H, W]` | the whole pipeline is NCHW end to end. Output bandwidth is 50 MB for the 384@256^2 case (~5 % of the kernel time) |
| Bias in fp32 before rounding, one broadcast load per output row | matches `addmm(bias, ...)` semantics without a separate pass | unit tests with and without bias across 15 shapes and 2 dtypes |
| Masked `C_out` so a tile can run a VAE's last conv (`C_out = 3`) | rejected, the kernel would buy 1.2 % where the MIOpen delegation already buys 3.3 %: at 1024^2 the padded 16-row kernel measures 1.97 ms against MIOpen's 5.72 and im2col's 19.41, and it would need a row predicate in the epilogue plus a padded weight buffer (the kernel's `block_m` clamp already overlaps for `C_out >= 16`) | the compute is fixed at 16 rows whatever `C_out` is 3 of, so the tile runs at 2.8 TF of its 36. Routing the shape to MIOpen is 3 lines of policy in `select_backend` and most of the win |
| Static 38-tile table, chosen internally by `conv3x3_prepacked` | no runtime autotuning, no benchmark pass on first use, one code path per `(M, W, rows)` | the tile sweep showed the widest fitting tile wins almost everywhere. The deviations are the overlap penalties quantified above, plus the row-count rule measured in `Accepted Optimizations` |
| Two output rows per block (`rows = 2`) for M-narrow tiles: `128x128` x2 and `64x256` x2 | +17.8 % (1152->1152 128^2), +20.2 % (384->384 128^2), +12.4 % (512->512 128^2), +8.8 % (576->576 256^2), +8.3 % (192->192 256^2). 34.1-37.4 TF where the single-row form gave 29.3-33.8 | interleaved A/B per shape, bit-exact against the single-row kernel (`torch.equal`). Halo re-read 3.05x -> 2.02x, 3.10x -> 2.06x at 128 wide. Costs 10-15 extra VGPRs (167 / 174), still no spills. The 14-shape grid total drops 94.2 -> 90.0 ms (34.70 -> 36.34 TF) with the whole table above moving, and the end-to-end VAEs are unchanged because the 128-wide layers are a small share of decode FLOPs |
| Delegate fp16 3x3 s1 to MIOpen's Winograd (`select_backend`) | 1.15x per conv on the fp16 grid (1.03-1.59x per shape). Flux 2 small decode 310.0 -> 287.1 ms (1.96x -> 2.10x over im2col) and encode 254.6 -> 238.4 ms, image rel 1.8e-4 -> 3.9e-4 (inside the 1e-3 gate) | MIOpen's `ConvWinoFuryRxS<2-3>` / `ConvBinWinogradRxSf2x3g1` reach 47-52 TF effective against this kernel's 36.3 TF at fp16, and no amount of direct-convolution tuning closes a 4/9-multiplies gap. Measured with the heuristic (`MIOPEN_FIND_MODE=FAST`), which also beats MIOpen's own timed search, 78.4 vs 94.1 ms. The delegated call forces `cudnn.enabled=True, benchmark=False`, because ComfyUI disables cuDNN on AMD and because exhaustive search is the autotune we do not want |
| 32-bit paired activation staging on the two-row tiles | +0.7..+2.4 % there and 3-9 fewer VGPRs (halves the VMEM instruction count of the loader) | interleaved A/B on `128x128x2` (+1.1, +2.4 %), `64x256x2` (+0.7, +1.1 %), `96x256x1` (+2.8 %), and -0.8 % on `128x256x1`, which is why it is gated by tile |
| Overlapping last tiles instead of masked tails | no mask code in the epilogue or the A loader, so the hot path stays branch-free | `M ∈ {96,144,288,576}` and any `W` not dividing the tile still run. Duplicate rows/columns are bit-identical because the reduction order is identical |
| Zero weight rows for the padded channel plane | `C_in % 16 != 0` works with no per-channel masking in the loader | unit tests at `C_in = 8, 24, 40, 56`. The earlier guarded loader cost 256 VGPRs + spills, see `History` |
| fp16 as a second instantiation of the same body | the Flux 2 small VAE family (Ideogram 4, Klein 9B, MOSS-Image) runs fp16 | decode 606.1 -> 310.0 ms (1.96x), image rel 1.8e-4 |
| Next-group register prefetch (`kPrefetchGroup`), per-cell switch | the dominant 128x256 tiles gain 3.5-7.9 %, the 128x128 two-row tile 6.1 %, and `C_out = 144` (48x256) 13.7 %. No measurable end-to-end change yet (455.7 vs 456.8 ms SDXL, 310.0 vs 309.9 ms Flux 2 small), because conv is now roughly half of VAE decode time | `python -m bench.benchmark_conv3x3 --variants both`, 14 VAE shapes, 3 rounds: prefetch on vs off is +4.1 % at `512->512 256^2`, +3.5 % at `384->384`, +5.7 % at `256->256`, +4.3 % at `128->128`, +7.9 % at `1152->1152 256^2`, +6.1 % at `1152->1152 128^2` (two-row), +2.3 % at `288->288`, +13.7 % at `144->144`, +1.0 % at `96->96 512^2`. -1.3 % at `96->96 256^2` and -8..-9 % on the 64x256 two-row tile (the doubled accumulator set leaves no room to preload), hence the per-cell gate. Costs 20 VGPRs (3 + 2 `uint4` per thread at 512 threads): 156 -> 162 VGPRs, still no spills |
| Per-instance class relabel from a `FeatherVAELoader` node instead of patching `comfy.ops` | zero-copy fast path, opt-in, `state_dict`/patcher/offload unchanged, and the plain `Load VAE` keeps the torch path. Measured through the node at 512^2 on random latents: Flux 2 small VAE decode 623.5 -> 286.6 ms (2.18x), Qwen 2D 149.5 -> 81.0 ms (1.85x), Qwen-Image 2.1 2D 1349.3 -> 806.9 ms (1.67x), rel 5.4e-4 / 2.2e-3 / 2.7e-3. The Wan 2.1 3D VAE relabels 3 convs (its 2D resampler) and decodes within 1.9e-3, its `CausalConv3d` modules untouched. The multigpu reload factory returns a relabelled patcher (55 of 55 convs on the Flux 2 VAE) | `~/ComfyUI-FeatherOps/tests/test_conv_patch.py`: transparency, inference mode, ineligible fallbacks, `weight_function`, the node contract, and that an unpatched conv is left alone |

## Rejected Optimizations And Dead Ends

Each of these was either measured or ruled out by the ISA/LLVM, and each has a reason that will not change on this hardware.

| idea | outcome | evidence |
| --- | --- | --- |
| Fragment double-buffering by tap (two register sets, prefetch tap+1's A and B) | -18 %, rejected | 206 VGPRs, no spills, yet 384@256^2 went 4.89 -> 5.95 ms. The compiler's schedule after the restructuring is worse than the latency it hides |
| Winograd F(2x3)/F(4x3) of our own, bf16 or fp16 | rejected | the bar is MIOpen's hand-written ASM at 47-52 TF effective, i.e. 96 % of the 54 TF issue ceiling counted on direct FLOPs, which works out to its matrix stage being ~43 % of the time and ~57 % transforms/LDS/epilogue. On gfx1151 fp16 transforms run on `v_pk_fma_f16` (two results per instruction) while bf16 has no packed FMA at all: `v_pk_fma_bf16` and `v_pk_fma_f32` reject on gfx1151 in `llvm-mc` (they exist on gfx1250/gfx13 only), leaving `v_dot2_f32_bf16` (two MACs, one fp32 result) or scalar fp32 FMA, i.e. half the per-result rate, so a bf16 Winograd would land around 33 TF effective, below this kernel's 36.5. The matrix unit is not the constraint: bf16 WMMA measures 53.53 TF against the fp16-accumulate variant's 53.78, so there is no fast fp16 matrix instruction that bf16 lacks. Accuracy stacks on top: bf16 needs fp32 accumulation in the transform domain, doubling that footprint in a kernel already at the 1536-VGPR limit. AITER independently concluded the same for RDNA wave32 ("the three Winograd launches and materialized V/M tensors cost more than the reduced arithmetic saves") |
| Winograd with fp16 accumulation over bf16 data (convert in and out) | rejected | costs two conversion passes, fights fp16's 65504 range against Winograd's 4x transform amplification, and buys nothing: the fp16-accumulate WMMA is not faster than bf16's fp32-accumulate WMMA here (53.78 vs 53.53 TF) |
| Async copy (`global_load_lds`, `buffer_load_*_lds`) | unavailable | LLVM gates vmem->LDS on gfx9/gfx10 (`docs/int8_convrot_fwd_plan.md`). The register prefetch is the only refill-hiding mechanism, which is why it is capped at +2-3 % |
| Streaming / non-temporal stores to protect L2 | unavailable | on gfx11 `FLAT_Load_Pseudo` carries a hardcoded zero cache-policy operand, so `!nontemporal` is dropped. Inline asm is unsafe (loses `vmcnt` accounting, and LLVM overlaps the vector result registers). Both recorded in `docs/int8_convrot_fwd_plan.md` |
| `S_PREFETCH_DATA` / `S_BUFFER_PREFETCH_DATA` | unavailable | gfx12/13 only |
| Deeper LDS staging (3-4 stages) | rejected twice | costs VGPRs and LDS without removing a barrier. Re-confirmed here by the -18 % double-buffer result |
| `s_setprio` as a general knob | noise | +0.5-1 % on the matmul kernels, inside the spread. Used here only to bracket the prefetch issue |
| Split-K across taps | rejected | the output would have to be accumulated in memory three times (65 KB per block per pass), costing more than the barrier it removes |
| 32-channel groups (half the barriers) | does not fit | the weight tile doubles to 73.7 KB at `block_m = 128`, over the 64 KB budget |
| Masked tails instead of overlapping tiles | rejected | a second code path in the epilogue and the loader for a few percent of wasted compute on at most one block per row. The overlap is bit-identical and branch-free |
| Runtime autotuning | rejected | the static table's worst measured miss is ~5 % (tile choice is nearly `N`-independent), versus a benchmark pass on first use and a cache to keep coherent |
| LDS bank-conflict padding / swizzles | unnecessary | measured zero conflicts: every access is 16 bytes wide with a 16-byte lane stride |
| 1-3-tap weight staging to free LDS for `BM = 192..384` | rejected | measured at `BM = 192, BW = 256` with three double-buffered 3-tap stages: 25.2-28.6 TF against 33.6-37.7 TF for the shipped tiles, i.e. -24 % on `512->512`, -25 % on `1152->1152`. The extra 3 stage transitions per channel group (6 barriers instead of 2) and 96 accumulator floats (239 VGPRs) cost far more than the fragment amortisation buys. The same "more work per block" idea is available for free through `rows = 2`, which needs no weight-staging change |
| Three or four output rows per block | rejected | `rows = 3` at `128x128`/`96x128` measures -8..-25 % and `rows = 4` spills outright (256 VGPRs, 152 spills). The accumulator set already doubles at `rows = 2`. Beyond that it exceeds the register file |
| Paired activation staging everywhere | gated, not global | +0.7..+2.4 % on the two-row tiles and +2.8 % on `96x256`, but -0.8 % on the `128x256` single-row tile, so it is compiled in only for the two-row instantiations |
| Manual `ds_bpermute` / `v_perm` staging transpose | not needed | the 8-scalar-load form is already hidden behind compute (staging is 2-3 % exposed), so the potential recovery is smaller than its risk |
| `groups>1`, dilation>1, NHWC, fp32 | out of scope | no WMMA for fp32 on RDNA3.5. The rest are not in any in-scope VAE |

## Open Questions And Risks

The build-context question is settled, and it was never about the build. The 6 % swing that this section used to attribute to code placement or allocation addresses is the machine's power state: the same source in three build contexts, interleaved in one process on the same buffers at `1152->1152 128^2` (tile 128x128), measures 12.795 / 12.808 / 12.823 ms - a standalone build of `csrc/hip_conv2d_3x3.cu` (`-DNO_PYTORCH`, lab flags), the same plus the extension's only codegen-relevant flag `-fno-gpu-rdc`, and the installed `_C.abi3.so`. 0.2 % apart, so SASS placement, the number of kernels in the module and the allocation addresses do not matter here, and no kernel-pack or fatbin mechanism is warranted (the `torch-ggml-ops` / hipBLASLt-style packing exists to ship many variants of one GEMM, not to shield a kernel from its neighbours).

The same binary and shape measures 13.02 ms cold, 13.81 ms after 60 s of sustained conv load, 13.44 ms after 20 s of idle, 14.05 ms after another 120 s of load and 13.06 ms after 60 s of idle, and an independent hipBLASLt bf16 matmul in the same process moves in lockstep: the conv/matmul throughput ratio is 1.217 / 1.211 / 1.207 / 1.213 / 1.217 across those five states in that order, constant to within 0.5 %. That is why earlier readings of this one shape span 12.4-13.7 ms: it is what ran first.

The sysfs clocks cannot show this. `pp_dpm_sclk` sits on level 1 (1.1 GHz) the whole time while the kernel delivers 30 TF, which the 54 TF issue ceiling puts at ~2.6 GHz, and `freq1_input` is a smoothed value between DPM steps. That is presumably the "glitch in thermal throttle policy" `benchmark_tracking.md` refers to when it notes hipBLASLt swinging between 32 and 37 TF.

The kernel's speed also depends on input values, the phenomenon https://www.thonking.ai/p/strangely-matrix-multiplications describes for NVIDIA matmuls. At `1152->1152 128^2`, zero activations measure 12.409 ms against 13.082 ms for random ones (+5.4 %, six alternating rounds, same buffer refilled in place), zero weights are +10 % and both together +13 %. hipBLASLt does the same (a zero `A` is 8.6 % faster, a zero `B` 4.8 %). Three controls rule out the obvious explanations without naming a mechanism: 256 MiB copies, fills and reads run at 213 / 455 / 478 GB/s for random, zero and one data (not transparent memory compression).

The register-only WMMA probe measures 53.8-54.0 TF for 1.0, 0, -1 and 0.0098 operands (not the matrix unit's issue rate), and a random-data matmul immediately before and after each conv reads the same (post/pre 0.994-1.000, so not a clock change either). What is left is data-dependent behaviour in the shared LDS-to-WMMA path. It is not exploitable, since VAE weights and activations are dense, but it means benchmarks must use random data (these do) and that a matmul used as a thermal meter must not be fed zeros.

Practical rules that follow, and are now used everywhere here: interleave the alternatives instead of running one after the other, normalise long comparisons by a compute meter, and treat any single-process difference below ~3 % as unresolvable on this machine.

The fp16 delegation carries its own risks, all measured or bounded:
- accuracy: MIOpen's gfx11 Fury kernel accumulates in fp16 (there is no fp32-accumulate build for gfx11. gfx12 has one). Flux 2 small end-to-end image relative error moved from 1.8e-4 (this kernel) to 3.9e-4 (MIOpen), and the encode latent from 7.5e-4 to 1.6e-3. Both are inside the gates, but the delegated path is the less accurate one, so a future fp16 VAE with a tighter tolerance should be checked rather than assumed.
- solver drift: which MIOpen kernel runs is MIOpen's decision (`ConvWinoFuryRxS<2-3>` on wide shapes, `ConvBinWinogradRxSf2x3g1` on 128-wide ones, on the ROCm build tested). A ROCm update can change it, and there is no per-shape record in our repo beyond the measurement in `Current Results`.
- find mode: only `MIOPEN_FIND_MODE=FAST` guarantees no per-shape autotune. With the default `NORMAL`, MIOpen times solvers on the first call for each new shape and writes a find DB entry.

Other live risks:
- `manual_cast` VAEs (fp8 / mixed-dtype checkpoints) stay on torch. The patched `manual_cast.Conv2d` short-circuits only when the cast would be a no-op. Otherwise it falls back. Supporting them means prepacking from the cast weight, which needs the cast dtype at pack time.
- Shapes that fall back and their FLOP share: `C_in % 8 != 0` (`3/4/10/14/21/28/42` channels, ~1 %), `C_out < 16`, width `< 32`, plus the delegated 1x1 (1.0-1.8 %) and 3x3 s2 (0.9-2.3 %).
- Small-`M` tiles waste work: `C_out = 144` runs a 48-wide block (27.2-28.0 TF, the lowest in the table) because nothing larger divides it. `C_out ∈ {96, 288}` still run a 96-wide single-row block (30.3-33.3 TF) and give up some of the 128-wide tile's efficiency. `C_out ∈ {192, 576}` escaped the same penalty by taking the 64x256 two-row tile instead (34.1-37.4 TF).
- Occupancy was not a lever and probably is not one: the tile sweep prefers the 60 KB / 16-wave 128x256 tile over every smaller (higher-occupancy) tile, and the barrier probe (removing half the barriers is worth 1-4 %) bounds what extra resident waves could hide. That matches the int8 kernel's finding that its 6-waves/SIMD cap at 34.8 KB LDS was not the limit either.
- The 128-wide column tile is no longer a weak case: `1152->1152 128^2` runs 36.4 TF against MIOpen's 32.1, because the two-row tile and the prefetch (which now covers `block_n = 128` too) both apply there. The narrowest remaining weak spot is `C_out = 144`, which can only use a 48-wide M tile (27.2-28.0 TF).
- The prepack is per weight version per process, not persisted to disk: reloading a VAE costs one pass over its weights (a few ms). Not worth a file cache.
- The fallback shapes still pay the im2col path, and they are exactly the traffic im2col handles worst (1x1, stride 2, `C_in % 8`). Routing just those through `torch.backends.cudnn.flags(enabled=True)` is untested in the patched world. When MIOpen did all convs it measured 1.15x on SDXL decode and 2.27x on the fp16 Flux 2 VAE, so the residual here is small but the change is one line.

## What Remains

Ranked by expected value, not effort. Every item is measured-informed and states the gate that decides whether to keep it.

| # | change | expected | effort | risk | keep gate |
| --- | --- | --- | --- | --- | --- |
| 1 | fp16 accumulator variant (`V_WMMA_F16_16x16x16_F16`): halves accumulator registers, doubling the tiles/wave budget. Parked deliberately, not dropped: fp16 accumulation is known to overflow on some workloads, so the accuracy gate is the blocker rather than the code | unknown, potentially +5-8 % | medium | accuracy: overflow and precision loss over `K` up to 10368 | only if it never overflows and relative L2 vs fp32 stays <= 1e-3 on all 15 VAE shapes |
| 2 | Delegate the remaining conv shapes: 1x1 (25 shapes, 1.0-1.8 % of FLOPs, 10.4 TF on the im2col path and 27.5 TF on MIOpen for `384->1152 256^2`) and 3x3 s2 (7 shapes, 0.9-2.3 %. 1.94 ms on MIOpen at 22.5 TF and 2.47 ms at 17.6 TF on im2col for `384->384 257^2`). The s2 body is a parity variant of the same halo tile, targeting ~33 TF and ~1.3 ms. 1x1 wants the existing prepacked-row path with a hipBLASLt solution table (MIOpen's `GemmFwd1x1_0_1` already shows 27.5 TF there and the 37 TF class is available on large `N`) | +2-3 % of VAE wall time (not kernel TFLOPS) | medium: GEMM epilogue for NCHW, phase-indexed stride-2 body | low | end-to-end VAE decode >= +2 % |
| 3 | End-to-end only: ~45 % of SDXL VAE decode is outside conv (norms, SiLU, attention, resample). Fusing SiLU into the conv epilogue, or a norm into the next conv's A-prepack, is the next real win | unknown, potentially large | high | high | separate project |

Not on the list, deliberately: everything in `Rejected Optimizations And Dead Ends`. Winograd of our own is closed for both dtypes (the fp16 carve-out is now served by delegating to MIOpen, see `Accepted Optimizations`), and it should not be reopened on RDNA3.5 without a new argument - on gfx1250 `v_pk_fma_bf16` exists and the bf16 transform penalty disappears, which is the one thing that would change the analysis.

## ComfyUI Integration

What ComfyUI does today. Every VAE conv is a `comfy.ops.disable_weight_init.Conv2d` (`comfy_cast_weights = False`) whose `forward` reaches `F.conv2d` -> `im2col_kernel` + `addmm`, because `torch.backends.cudnn.enabled` is turned off on RDNA3+. Activations between convs are contiguous NCHW. The first conv of `VAE.encode` gets an `movedim` view that is not contiguous in CHW order.

Patch point. `comfy_feather/vae.py` adds a `FeatherVAELoader` node, and `comfy_feather/conv.py` relabels the convs of the VAE it just loaded:

```python
class FeatherVAELoader(nodes.VAELoader):  # same inputs, same load path, same RETURN_TYPES
    def load_vae(self, vae_name):
        (vae,) = super().load_vae(vae_name)  # ComfyUI's own loader, untouched
        patch_conv_modules(vae.first_stage_model)  # relabel 3x3 s1 p1 convs, in place
        return (vae,)
```

`patch_conv_modules` assigns `type(conv)` to a cached subclass of `(mixin, type(conv))`, so the conv objects, their parameters and their buffers are the same objects with the same names: the module tree, `state_dict`, `ModelPatcher` and offload paths see what ComfyUI built, and `copy.deepcopy` (how a multigpu clone copies a model) carries the relabelled class over. Only exact `comfy.ops` Conv2d and bare `torch.nn.Conv2d` types are relabelled - a subclass may override `forward`, and prefixing it would drop that override. Nothing inside `comfy.ops` is replaced, so the plain `Load VAE` node, checkpoints that carry a VAE, and every other consumer keep the torch path: the kernel is opt-in per load.

Reusing `VAELoader.load_vae` is what makes the node cover every VAE ComfyUI can load - the TAESD decoders, `pixel_space`, the 3D VAEs, and VAEs a third-party pack teaches `comfy.sd.VAE` to build, since that pack's `__init__` is what runs inside `super().load_vae`. It also inherits the `cached_patcher_init` reload factory that ComfyUI registers for multigpu deepclones. The node wraps that factory so a clone is relabelled too. Module surgery that replaces modules was rejected and stays rejected: the patcher and `load_state_dict` run inside `VAE.__init__`, and swapping modules afterwards invalidates the patcher's bookkeeping, whereas a class relabel does not touch it at all.

The fp16 delegation lives here too, and it has one ComfyUI-specific trap: `comfy/model_management.py` sets `torch.backends.cudnn.enabled = False` on AMD unless `COMFYUI_ENABLE_MIOPEN=1`, so a bare `super().forward()` would run the im2col path for the delegated convs - 2x slower than this kernel, the opposite of the intent. The patch therefore wraps its delegated call in `torch.backends.cudnn.flags(enabled=True, benchmark=False)`: MIOpen on, exhaustive search off (no autotune), and the caller's global state untouched. The delegation is reported once per module as its own reason (`fp16 is delegated to MIOpen...`), and a delegated weight is never prepacked.

MIOpen's find mode is process-global and read from `MIOPEN_FIND_MODE` (this machine's `~/rocm.sh` sets `FAST`). `FAST` is the mode the numbers above were measured in, and it is also the mode to keep: with `NORMAL` MIOpen runs a timed search per new shape (that is the autotune we are avoiding) and, on this grid, picks a worse solver for `1152->1152 256^2` (54.6 vs 39.9 ms), so heuristic-only is both cheaper to start and better here. A library cannot safely set it (MIOpen reads it at init), so it stays an environment setting.

Fast-path contract (checked per call in `forward`, all cheap attribute tests):
- activation and weight are bf16/fp16 (they need not be the same one: the prepack follows the activation dtype, which is what ComfyUI's own cast path does), `ndim == 4`, contiguous, `C_in == weight.shape[1]`.
- 3x3, stride 1, padding 1, dilation 1, `groups == 1`, `padding_mode == "zeros"`.
- `select_backend(dtype, C_in, C_out, H, W)` (imported from `feather_ops.conv2d`) resolves to `"kernel"`, which also implies `C_in % 8 == 0` and a tile exists (`C_out % block_m == 0`, `W >= 32`). `"miopen"` is the delegation: fp16 under the cache-residency window, and the no-tile shapes at any size.
- no `weight_function`/`bias_function`, no `weight_lowvram_function`.
- `torch.is_grad_enabled()` is false (ComfyUI's execution loop runs nodes under `torch.inference_mode()`).

Anything else falls back to `F.conv2d`. The first fallback per module logs one line with the reason, so a silent 2x slowdown cannot hide, but the reasons a normal VAE always has - its first conv's `C_in % 8`, its last conv's `C_out` below the tile minimum, the 1x1 and stride-2 convs, and the MIOpen delegation that takes over the first two - log at debug instead: they would otherwise print a handful of lines on every load and say nothing actionable. A layout miss does not count as that first fallback, because the first conv of `VAE.encode` gets a non-contiguous view and would otherwise use up the line for a module that is otherwise fine. One ComfyUI detail worth knowing: for the Qwen-family VAEs `VAE.process_input` pads a 3-channel image to 4 with `pad_channel_value = 1.0`, so their first encoder conv sees `C_in = 4` and falls back on `C_in % 8` as well. No `.contiguous()`, no `.to()`, no `.permute()` happens on the fast path.

The registered op is an out variant: `conv3x3_prepacked` allocates the single output with `torch.empty` and the op - and the C++ kernel behind it - writes into it, so the extension itself never allocates (`Tensor(a!) out` by reference) and the allocation sits outside the custom-op boundary, where the compiler can pool and reuse it. Eager ComfyUI does not care either way, since a per-call `torch.empty` is just a caching-allocator hit.

Which VAEs take it (`model_management.vae_dtype()` picks from each VAE's `working_dtypes`):

| VAE | `working_dtypes` | runtime dtype | models |
| --- | --- | --- | --- |
| SDXL VAE | bf16, fp32 | bf16 | SDXL |
| Qwen 2D VAE | bf16, fp16, fp32 | bf16 | Anima, Qwen-Image 1, Krea 2 |
| Qwen-Image 2.1 2D VAE | bf16, fp16, fp32 | bf16 | Qwen-Image 2.1 |
| Flux VAE (`ae.safetensors`) | bf16, fp32 | bf16 | Z-Image |
| Flux 2 small VAE | fp16, bf16, fp32 | fp16 | Ideogram 4, Klein 9B, MOSS-Image |

So the bf16 body covers 6 of the 9 image models and the fp16 instantiation covers the other 3. Forcing the Flux 2 small VAE to bf16 would also work numerically but changes the model dtype globally, so it is not the default.

Prepack lifetime. The prepack is a non-persistent buffer keyed on `(data_ptr, _version, dtype, device)` via `feather_ops.conv2d.weight_cache_key`, so `load_state_dict` or a real device/dtype move repacks and a no-op `module.to(...)` does not. `_apply` deliberately does not invalidate: ComfyUI's load path issues no-op `.to()` calls, and dropping the prepack there would repack on every image. Offload/low-VRAM works because the buffer is a module buffer and moves with the parameters.

The one non-contiguous case. The first conv of `VAE.encode` receives a `movedim` view and has `C_in = 3`, so it falls back on two counts at once (layout and `C_in % 8`). That is why no strided/KC-major loader was written: the only layer that would use it is one the kernel cannot handle anyway, and it is ~0.1 % of the decode+encode FLOPs.

Tiled VAEs and 3D VAEs. `VAEDecodeTiled` passes contiguous CHW tiles, which take the fast path per tile (verified: 1.83x, rel 1.5e-3). 3D VAEs are untouched except their embedded 2D resamplers (Wan 2.1 decode 1.10x). `groups>1`, the audio/mel VAEs and fp8 AIO VAEs fall back automatically.

## Numerics And Tests

The fp16 policy has its own tests: `select_backend` answers for both dtypes, both sides of the 512 MiB gate and the no-tile case. The delegated `conv2d`/`Conv2d` output is `torch.equal` to `F.conv2d` with MIOpen forced on and is not equal to this kernel's fp16 output, and a delegated module never prepacks its weight. The dispatch is covered by a 14x6 `(C_out, W)` matrix test against `F.conv2d` at `C_in = 32, H = 7`, which is how the prepack-cache stale hit described in `Scope And API` was found.

Two invariants are stronger than a tolerance and are asserted as such: the two-row kernel must be bit-identical to the one-row kernel for the same tile (`torch.equal`, because the tiling cannot change any pixel's reduction order), and a prepacked-then-repacked weight must reproduce the original conv. Tolerances otherwise follow bf16 arithmetic, not the fp32 ideal: relative L2 <= 4e-3 for bf16 and <= 6e-4 for fp16 against an fp64 reference, and the end-to-end gates in `Current Results` (image rel <= 1e-2). Bit-identical reduction order versus torch is explicitly not a goal: the kernel path matches to bf16 arithmetic, while every fallback path is asserted `torch.equal` to a plain `F.conv2d` call.

## History: What The Original Plan Got Wrong

Kept so the same estimates are not reused. The plan's own phase ladder (P0 skeleton, P1 tails, P2 autotune, P3 ComfyUI + fp16, P4 edge + scale, P5 optional) ended as: P0-P3 done, P4 only its int64 addressing (2048^2) part, P5 dropped as described below. The P0/P2 TFLOPS gates were met with room (>= 27 TF for `C_in >= 192`, 26.7-32.3 TF for `C_in ∈ {96,128,144}`). Of the three P3 decode targets, SDXL (0.456 s vs a 0.50 s gate) and Flux 2 small (0.310 s vs 0.40 s) cleared, and Qwen-Image 2.1 sits at its gate (0.840-0.852 s vs 0.85 s) within run-to-run noise.

| plan said | reality |
| --- | --- |
| "WMMA 16x16x16 = 32 cycles, ~55 TF, ~37 TF practical kernel ceiling" | the 32-cycle model is right but the sustainable issue rate is 54 TF (measured). 37 TF was just what the existing matmul kernels happened to reach. Every "we are at X % of the ceiling" statement in the first version of this document was therefore ~45 % too pessimistic (54/37). |
| two bodies, `KB1` (3x3 s1) and `KB1-edge` (small `M`/`K`, `BM=16, BK=16`, store-bound, "may use a plain vectorized direct conv") | one body. Small `M` uses a 16- or 48-wide tile of the same body, and `C_out < 16` falls back to torch (~0.1 % of FLOPs). No edge kernel was ever worth writing. |
| a ~1,700-measurement autotune grid over 24 `(M,K)` x 3 `N` x 10-12 configs, a generated per-shape config table, `old_autotune` reuse, `conv2d_configs.py`, `gen_conv_configs.py` | a static tile table plus a divisibility rule inside `conv3x3_prepacked`, no runtime autotuning, no generated table and no public selector. The measured tile sweep (25 shapes x 3 `N` x 10 tiles) showed the widest fitting tile wins almost everywhere, so per-shape tuning would have bought ~5 % at most. |
| kernel file `csrc/hip_conv2d_3x3_bf16.cu`, benchmarks `bench/benchmark_conv3x3_hip.py` + `cpp_benchmarks/benchmark_conv3x3` | `csrc/hip_conv2d_3x3.cu` and `bench/benchmark_conv3x3.py`. No separate C++ benchmark (the Python harness measures the extension itself, with both torch paths as controls). |
| a menu of three global->LDS transpose mechanisms (8x8 `ds_bpermute`, `ds_write_b16` scatter, `v_perm_b32` register transpose), "pick by profiling" | none of them. 8 scalar 2-byte loads + one 16-byte staged store per 8-channel plane measured well enough that the register-transpose variants never became worth their risk. Staging ends up only 2-3 % exposed. |
| masked `M`/`K`/`N` tails (each phase adding one) | overlapping last tiles, which removes the tail code entirely and is bit-identical. Only `C_in % 8` remained as a hard requirement, and for a different reason (the zero-weight padded plane). |
| `BH = 2/4` as "P4, the halo reuse lowers the input re-read from 3x to 2x/1.5x, gain mainly L2 pressure at 2048^2" | implemented as `rows = 2`, but on the opposite tiles and for a different reason than the plan gave. The planned pair (`BM = 96, BW = 256`) is the one that loses (-3.6 % at 192), because 96x256 with two rows needs 96 accumulator floats and 220 VGPRs. The pairs that win are `128x128` x2 (+12..+20 %, the 128-wide family the plan expected nothing from) and `64x256` x2 (+8.3 % at 192, +8.8 % at 576). The gain is not L2 pressure: it is the halved per-block staging, barrier and epilogue work on M-narrow tiles. `rows = 3/4` are dead (see `Rejected Optimizations`). |
| 1-3-tap weight staging "to free LDS for 12-24 tiles/wave" was worth +2-4 % | rejected on measurement: 25.2-28.6 TF against 33.6-37.7 TF for the shipped tiles. The barrier count it adds is worth more than the fragment amortisation, and `rows = 2` gets the same "more work per block" for free |
| the 8 scalar 2-byte staging loads were "well enough", no register transpose worth its risk | half right: the transpose alternatives are still not worth it, but pairing two pixels per 32-bit load is a 40-line loader change worth +0.7..+2.8 % on the M-narrow tiles and -0.8 % on the wide one, so it is compiled only where it pays |
| Winograd F(2x3) as an optional phase worth gating at "+20 % with the accuracy gate met" | rejected as our own kernel, but the target was real: MIOpen's fp16 Winograd does deliver it. The answer is to delegate rather than to write one: fp16 3x3 s1 now goes to MIOpen by default (1.15x per conv, Flux 2 small decode 310 -> 287.1 ms), while bf16 stays on this kernel because gfx1151 has no packed bf16 VALU and MIOpen has no bf16 Winograd at all. AITER's "three launches + materialized V/M tensors" argument is about the unfused variant. MIOpen's is a single fused ASM kernel |
| "MIOpen's best is 33 TF at `C_in=1152`, so for the widest layers the target is parity" | parity was not enough and not where the win came from. For a while 1152@128^2 was the one shape MIOpen won (32.4 TF vs our 29.3), and the plan's 12.2 ms target looked missed by a reading (13.3 ms) that itself sat mid-band in the 12.4-14.05 ms thermal spread. The two-row tile then took the same shape to 11.18 ms (35.0 TF) without touching the 12.2 ms target's assumptions, so the shape is now 1.09x ahead of MIOpen. Everywhere else the kernel is 1.3-2.4x MIOpen's tuned choice. |
| 6 % of the 128-wide tile's time was "build context": code placement in a module that also carries the matmul kernels, or the output allocation living inside the custom op | no build effect at all. Three build contexts of the same source interleaved in one process land within 0.2 % (12.795 / 12.808 / 12.823 ms). The 13.5-13.7 ms readings came from runs that followed sustained load, and the shape spans 13.02-14.05 ms purely by thermal state. No kernel packing is needed. |

Three implementation bugs worth remembering, because each cost real time and each is a trap for the next kernel:
- A WMMA k-step is 16 channels, i.e. two 8-channel LDS planes concatenated into one fragment. Reading one plane per k-step left the upper half of every fragment undefined. The tap-by-tap test (`weight = delta(tap)`, `y` must equal the shifted input) caught it immediately and is the cheapest possible guard.
- Guarding the activation loader for a possibly padded channel plane inside the hot loop pushed the fast path from 156 to 256 VGPRs with 26 spills (35.4 -> 32.2 TF end-to-end). Hoisting the decision out of the inner loop - clamp the plane base, let the zero weight rows mask the tail - restored the allocation. This is why `C_in % 8` is a hard requirement instead of something to handle in the loader.
- The functional prepack cache keyed on `(data_ptr, _version, dtype, device, shape)` can return a stale pack: a freed tensor's address can be reused by a new tensor, which also starts at version 0 with the same shape. The dispatch matrix test caught it (relative error 1.4 on `C_out = 16, W = 192`), and the fix is a weak reference plus an identity check on the way in.
- `torch.inference_mode()` makes inference tensors, which do not track a version counter, so `weight._version` raises `RuntimeError` - and ComfyUI wraps every node body in inference mode. Hence `weight_cache_key` falling back to storage identity, and hence `_apply` not invalidating the prepack.

## Files And Artifacts

```text
~/feather-ops/
  csrc/hip_conv2d_3x3.cu              # kernel, prepack layout, tile dispatch, prefetch variant
  feather_ops/conv2d.py               # prepack, conv3x3_prepacked, conv2d, Conv2d, select_backend, weight_cache_key
  feather_ops/__init__.py             # re-exports
  tests/test_conv3x3.py               # kernel tests (pytest)
  bench/benchmark_conv3x3.py
~/ComfyUI-FeatherOps/
  comfy_feather/conv.py               # patch_conv_modules, the conv mixin's routing
  comfy_feather/vae.py                # FeatherVAELoader: ComfyUI's load_vae + the relabel
  __init__.py                         # registers FeatherUNetLoader and FeatherVAELoader
  tests/test_conv_patch.py            # integration tests for the patch (pytest)
  tests/conftest.py                   # sys.path for comfy_feather and ~/ComfyUI
  pyproject.toml                      # ruff/ty config and the pytest namespace-package setting
  docs/hip_conv2d_3x3_plan.md         # this document
~/tmp/feather_ops/conv2d_3x3/
  conv3x3.cu, conv3x3_lab.cu          # standalone copy, lab copy with the ablation variants
  lab2.cu, lab2lib.py                 # second lab copy: rows, pair loads, 3-tap staging + its ctypes bindings
  lab2bench.py, lab2pair.py, lab2stage.py   # interleaved A/Bs for rows, pair loads, staged weights
  build.py, convlib.py                # hipcc build and ctypes bindings for the standalone copy
  bench.py, sweep_grid.py, labbench*.py   # tile sweeps, ablation A/B, config tables
  grid_results.csv, bench*.txt, labbench_*.txt
  bench_vae_conv_e2e.py               # end-to-end ComfyUI VAE benchmark
  smoke_comfy_nodes.py                # VAEDecode / VAEDecodeTiled under torch.inference_mode()
  wmma_peak.cu                        # pure-WMMA issue ceiling probe (the 54 TF number)
  wmma_peak_data.cu                   # the same probe with selectable operand data (value dependence)
  wmma_peak_both.cu                   # bf16 WMMA vs the fp16-accumulate WMMA, same loop
  alu_peak.cu                         # packed fp16 FMA / bf16 dot2 / fp32 FMA instruction rates
  bench_wino.py                       # ours vs MIOpen per shape and dtype (the Winograd decision)
  probe_miopen_solver.py              # which MIOpen solver torch picks, from MIOPEN log level 6
  build_prod_standalone.py, prodlib.py    # builds/loads csrc/hip_conv2d_3x3.cu standalone (-DNO_PYTORCH)
  ab_context.py                       # 3-way interleaved build-context A/B (standalone vs extension)
  meter.py, clockwatch.py, thermal_probe.py   # conv vs hipBLASLt-matmul meter across thermal states
  value_repro.py, data_dependence*.py, ablate_values.py, mem_values.py   # input-value dependence + controls
  pcs/                                # stochastic PC-sampling CSVs + pcsamp_target.py
  kstats.py, lds.py, stats2.py        # vgpr/spill/LDS metadata for every instantiation
  check_int64.py, check_large.py      # 2048^2 and C_out*H*W > 2^31 addressing checks
  check_noise_floor.py, find_conv_mismatch.py, check_vae_numerics.py
```

The lab copy keeps the ablation variants behind template flags so a future session can re-run any of them: `0` baseline, `1` next-group prefetch, `4` fragment double-buffer, `5` both, `6` no second barrier, `7` B-fragment reuse, `16`/`17` compute-only / loads-only.
