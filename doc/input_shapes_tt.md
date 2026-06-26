# gfx1151 FP16 TT Input Shapes

This note records confirmed `TT` GEMM shapes found while auditing common inference and training paths in `~/ComfyUI/`, `~/sd-scripts/`, and `~/musubi-tuner/`.

Machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/`:
- `tt_gemm_shapes.csv`: normalized confirmed `TT` GEMM rows.
- `tt_gemm_shapes_manifest.json`: assumptions, source paths, counts, and excluded near-misses.

## Layout Rule

For PyTorch `mm`/`matmul`/`linear` with a default row-major output, the backend swaps operands at the BLAS boundary. A TensileLite `TT` row therefore requires both visible PyTorch operands to be column-major-like/view-transposed after that swap:

| PyTorch `lhs` stride layout | PyTorch `rhs` stride layout | TensileLite layout |
| --- | --- | --- |
| `N` column-major/view-transposed | `N` column-major/view-transposed | `TT` |

For `torch.nn.functional.linear(input, weight, bias)`, the visible rhs is `weight.t()`. A default contiguous `weight` makes this rhs column-major-like. The forward path becomes `TT` only when the emitted GEMM also receives a column-major-like `input` instead of a contiguous flattened input.

## Confirmed Rows

Confirmed normalized `TT` rows:

| Dataset | Component | Phase | Exact `[M,N,batch,K]` | Count per size | Notes |
| --- | --- | --- | --- | --- | --- |
| Krea 2 | `text_fusion_projector` | forward inference / frozen-weight forward | `[1,2560,16,12]` | 1 | `Linear(12,1)` over `16` text tokens and `2560` hidden channels. |
| LTX 2.3 | `patchify_proj` | forward inference / autograd forward | `[4096,1536,1,128]` | 1 | Normalized row for `640x480x40`; actual token view has `N=1500` before tuning-row rounding. |
| LTX 2.3 | `patchify_proj` | forward inference / autograd forward | `[4096,8960,1,128]` | 1 | Normalized row for `1280x720x80`; actual token view has `N=8800` before tuning-row rounding. |

Counts:
- Normalized `TT` rows: 4.
- Unique exact `TT` shapes: 3.
- Weighted `TT` occurrences: 4.

## Krea 2 TextFusion Projector

Sources checked:
- `~/ComfyUI/comfy/ldm/krea2/model.py`.
- `~/musubi-tuner/src/musubi_tuner/krea2/krea2_mmdit.py`.

Krea 2 receives a stack of 12 selected Qwen3-VL hidden-state layers with shape `[B, text_tokens, 12, 2560]`. `TextFusionTransformer` first processes the layer axis with two layerwise blocks, then rearranges:

- Before projector: `[B * text_tokens, 12, 2560]`, made contiguous before the layerwise blocks.
- Projector input: `rearrange(x, "(b l) n d -> b l d n")`, shape `[B, text_tokens, 2560, 12]`.
- For `B=1`, `text_tokens=16`, the projector input stride is `(491520, 30720, 1, 2560)`.

In no-grad inference, PyTorch lowers the bias-free `Linear(12,1)` to a batched matmul:

- PyTorch matmul: `[16,2560,12] @ [16,12,1]`.
- PyTorch lhs per batch has row stride `1` and column stride `2560`, so it is column-major-like.
- PyTorch rhs comes from `weight.t()` with row stride `1` and column stride `12`, also column-major-like.
- TensileLite exact shape: `[M=1, N=2560, batch=16, K=12]`.
- TensileLite layout: `TT`.

Autograd caveat:
- If the projector weight requires gradients, PyTorch clones and flattens the non-contiguous input before the forward `mm`, so the trainable full-weight projector forward is not recorded as `TT`.
- If the projector weight is frozen but gradient must flow through its input, such as some frozen-base LoRA/adaptation paths, the base projector forward can still use the same `TT` batched matmul.

## LTX 2.3 Patchify Projector

Sources checked:
- `~/ComfyUI/comfy/ldm/lightricks/model.py`.
- `~/ComfyUI/comfy/ldm/lightricks/symmetric_patchifier.py`.

LTX 2.3 uses `SymmetricPatchifier(1)` for the documented no-audio video path. With patch size 1, the patchifier rearranges latents from `[B,C,F,H,W]` to token form `[B,F*H*W,C]` without materializing a contiguous row-major token matrix:

- Rearranged token view: `rearrange(latents, "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)", p1=1, p2=1, p3=1)`.
- Example profiled stride for `[1,128,3,16,32]`: output shape `[1,1536,128]`, stride `(1536,1,1536)`, non-contiguous.
- `patchify_proj` is `Linear(128,4096)`, so the visible rhs `weight.t()` is also column-major-like.

The emitted PyTorch forward path uses a 2D `mm` over the non-contiguous token view, not a clone-normalized input:

- PyTorch matmul: `[tokens,128] @ [128,4096]`.
- PyTorch lhs has row stride `1` and column stride `tokens`, so it is column-major-like.
- PyTorch rhs comes from `weight.t()` with row stride `1` and column stride `128`, also column-major-like.
- TensileLite layout: `TT`.

Documented normalized video rows follow the existing LTX tuning convention and round actual video-token counts up to multiples of 256:

| Video size | Actual tokens | Normalized `N` | Exact `[M,N,batch,K]` |
| --- | ---: | ---: | --- |
| `640x480x40` | 1500 | 1536 | `[4096,1536,1,128]` |
| `1280x720x80` | 8800 | 8960 | `[4096,8960,1,128]` |

The exact pre-rounding GEMMs are `[4096,1500,1,128]` and `[4096,8800,1,128]`; the CSV keeps the rounded normalized rows used by the rest of the LTX 2.3 shape dataset.

## Excluded Near-Misses

These patterns were checked but not recorded as stable model `TT` GEMM rows:

- SDXL `SpatialTransformer` in ComfyUI uses `movedim(...).flatten(...).contiguous()` before `proj_in`, so its Linears remain default `TN` forward rows.
- Qwen, Z-Image, Krea image patchify, Ideogram 4, and Flux/Klein patchify rearranges materialize contiguous token tensors before their first `Linear`; LTX 2.3 is included because its patchifier preserves a channel-last view into `patchify_proj`.
- ComfyUI Wan uses `flatten(2).transpose(1,2)` after 3D convolution, but the transformer block calls `x.contiguous()` before Linears; reference/context concatenation also materializes contiguous tensors.
- musubi-tuner Wan also starts from `flatten(2).transpose(1,2)`, but padding and `torch.cat` produce contiguous tensors before transformer Linears.
- sd-scripts DyLoRA Conv2d fallback can call `F.linear` on an `NCHW -> NHW,C` transposed view, but those shapes are adapter-target and spatial-size dependent rather than stable checkpoint Linear/GEMM rows.
- Attention-kernel internal matmuls remain excluded; fused-attention shapes are documented separately in `doc/input_shapes_attn.md`.

## Planning Notes

Treat the confirmed rows as explicit `TT` targets rather than evidence for a broad `TT` workload grid. The Krea 2 projector is a small strided-batched `K=12` case; the LTX 2.3 patchify projector is a batch-1 video-token case with `K=128` and large `M=4096`.
