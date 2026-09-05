# gfx1151 FP16 NN Input Shapes

This note documents `NN` activation/input-gradient GEMM shapes for the training workloads already audited in `input_shapes_nt.md`.

Machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/`:
- `nn_gemm_shapes.csv`: exact normalized base-Linear and rank-16 LoRA factor input-gradient rows.
- `nn_gemm_shapes_manifest.json`: derivation rules, conditional execution scope, alternate paths, and per-model counts.

## Scope

- Models: SDXL, Anima, Qwen-Image 1, Qwen-Image 2.1, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, LTX 2.3, H3, and MOSS-Image.
- Workload sizes, prompt/context lengths, video geometry, target-module choices, and activation-row rounding match `input_shapes_nt.md`.
- Exact rows use TensileLite order `[M,N,batch,K]`.
- Standard model inference does not execute these NN Linear input-gradient GEMMs. They are backward/training shapes.
- Base input gradients and unmerged rank-16 LoRA factor input gradients are included. Convolutions and fused-attention internals remain excluded.

## Layout Rule

For a logical Linear weight `weight[out,in]`, PyTorch's input gradient is mathematically:

```text
grad_input[rows,in] = grad_output[rows,out] @ weight[out,in]
```

After the row-major output operand swap used by the backend, both visible operands are column-major/view-transposed. The exact NN row is:

```text
NN [M=in, N=rows, batch=1, K=out]
```

The matching NT parameter-gradient row is `[M=in,N=out,batch=1,K=rows]`, so the derivation is:

```text
NT [in,out,batch,rows] -> NN [in,rows,batch,out]
```

For rank-16 LoRA attached to `(out,in)`:

```text
backward through lora_up:   NN [16,rows,1,out]
backward through lora_down: NN [in,rows,1,16]
```

## Conditional Execution

The CSV is a complete activation-gradient envelope corresponding to the documented NT base and LoRA factor rows. An individual NN row executes only if autograd needs the gradient of that operation's input.

Examples of rows that can be skipped:
- A first patch/input projection whose latent input does not require gradients.
- A conditioning projection fed by a cached, detached text embedding.
- A LoRA-down input gradient at a graph boundary where only the factor's parameter gradient is required.

Conversely, gradients through intermediate frozen base Linears still execute during LoRA training because they propagate gradients to earlier trainable adapters. The `count` column is therefore a logical matrix count and an upper-bound shape inventory, not a claim that every row executes in every trainer configuration.

## Aggregate Counts

Primary derived NN rows:
- 857 aggregate rows.
- 488 unique exact NN shapes.
- 29,417 weighted logical matrix occurrences.

By primary category:

| Category | Aggregate rows | Unique shapes | Weighted occurrences |
| --- | ---: | ---: | ---: |
| Base Linear input gradient | 362 | 244 | 9,891 |
| Rank-16 LoRA factor input gradient | 495 | 244 | 19,526 |

Krea 2 contributes two workload-labelled alternate rows for the frozen-base batched projector path. They collapse to one additional unique exact shape, `NN [12,2560,16,1]`. Including alternatives gives 859 CSV rows and 489 unique shapes, but the alternate rows are not additive with the flattened full-weight projector row.

## Per-Model Counts

Counts below cover the primary derivation. Krea 2's two alternate rows are listed separately.

| Model | Base rows / unique / count | LoRA rows / unique / count |
| --- | --- | --- |
| SDXL | `23 / 21 / 1644` | `31 / 27 / 3288` |
| Anima | `25 / 19 / 969` | `32 / 24 / 1938` |
| Qwen-Image 1 | `26 / 18 / 1692` | `32 / 22 / 3384` |
| Qwen-Image 2.1 | `22 / 15 / 400` | `44 / 24 / 800` |
| Z-Image | `40 / 29 / 416` | `62 / 37 / 832` |
| Krea 2 | `36 / 24 / 528` | `52 / 32 / 1056` |
| Ideogram 4 | `42 / 29 / 842` | `52 / 35 / 1684` |
| Klein 9B | `34 / 25 / 242` | `44 / 33 / 484` |
| Wan | `18 / 13 / 812` | `24 / 17 / 1624` |
| LTX 2.3 | `30 / 20 / 1280` | `34 / 24 / 2560` |
| H3 | `30 / 23 / 528` | `16 / 12 / 800` |
| MOSS-Image | `36 / 19 / 538` | `72 / 27 / 1076` |

Krea 2 alternate: `2 / 1 / 2`, exact shape `[12,2560,16,1]` for both documented image sizes.

## Shape Construction By Model

For each logical base pair `(out,in)` below, the NN base family is `[in,rows,1,out]`. For each targeted rank-16 LoRA pair, the factor families are `[16,rows,1,out]` and `[in,rows,1,16]`. The listed row bands and feature pairs are inventories, not Cartesian products. Use `nn_gemm_shapes.csv` for exact combinations and counts.

| Model | Activation-row bands | Base `(out,in)` feature pairs |
| --- | --- | --- |
| SDXL | `16, 1024, 2304, 4096, 9216` | `(640,640)`, `(640,2048)`, `(640,2560)`, `(768,768)`, `(768,3072)`, `(1280,1280)`, `(1280,2048)`, `(1280,5120)`, `(3072,768)`, `(3840,1280)`, `(5120,640)`, `(5120,1280)`, `(10240,1280)` |
| Anima | `1, 16, 512, 4096, 9216` | `(64,2048)`, `(256,2048)`, `(1024,1024)`, `(1024,4096)`, `(2048,68)`, `(2048,1024)`, `(2048,2048)`, `(2048,8192)`, `(4096,256)`, `(4096,1024)`, `(6144,256)`, `(6144,2048)`, `(8192,2048)` |
| Qwen-Image 1 | `1, 16, 4096, 9216` | `(64,3072)`, `(3072,64)`, `(3072,256)`, `(3072,3072)`, `(3072,3584)`, `(3072,12288)`, `(6144,3072)`, `(12288,3072)`, `(18432,3072)` |
| Qwen-Image 2.1 | `1, 2, 16, 4096, 4112, 9216, 9232` | `(64,4096)`, `(4096,64)`, `(4096,256)`, `(4096,4096)`, `(4096,12288)`, `(16384,4096)`, `(24576,4096)` |
| Z-Image | `1, 16, 32, 4096, 4128, 9216, 9248` | `(64,3840)`, `(256,1024)`, `(1024,256)`, `(3840,64)`, `(3840,256)`, `(3840,2560)`, `(3840,3840)`, `(3840,10240)`, `(10240,3840)`, `(11520,3840)`, `(15360,256)` |
| Krea 2 | `1, 16, 192, 4096, 4112, 9216, 9232, 40960` | `(1,12)`, `(64,6144)`, `(1536,6144)`, `(2560,2560)`, `(2560,6912)`, `(6144,64)`, `(6144,256)`, `(6144,2560)`, `(6144,6144)`, `(6144,16384)`, `(6912,2560)`, `(16384,6144)`, `(36864,6144)` |
| Ideogram 4 | `1, 16, 4096, 4112, 9216, 9232` | `(128,4608)`, `(512,4608)`, `(4608,128)`, `(4608,512)`, `(4608,4608)`, `(4608,12288)`, `(4608,53248)`, `(12288,4608)`, `(13824,4608)`, `(18432,512)` |
| Klein 9B | `1, 512, 4096, 4608, 9216, 9728` | `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,12288)`, `(4096,16384)`, `(8192,4096)`, `(12288,4096)`, `(24576,4096)`, `(36864,4096)` |
| Wan | `1, 512, 12032, 72192` | `(64,5120)`, `(5120,256)`, `(5120,4096)`, `(5120,5120)`, `(5120,13824)`, `(13824,5120)`, `(30720,5120)` |
| LTX 2.3 | `1, 1024, 1536, 8960` | `(32,4096)`, `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,16384)`, `(8192,4096)`, `(16384,4096)`, `(36864,4096)` |
| H3 | `2, 16, 256, 512, 5120, 5376, 25088, 25344` | `(32,5376)`, `(96,5376)`, `(5376,32)`, `(5376,96)`, `(5376,5120)`, `(5376,7168)`, `(5376,14336)`, `(10752,8)`, `(21504,5376)`, `(28672,5376)`, `(96768,8)` |
| MOSS-Image | `1, 16, 4096, 4112, 9216, 9232` | `(128,3072)`, `(3072,128)`, `(3072,256)`, `(3072,3072)`, `(3072,4096)`, `(3072,6144)`, `(3072,8192)`, `(3072,18432)`, `(6144,3072)`, `(8192,3072)`, `(9216,3072)`, `(16384,3072)`, `(18432,3072)` |

## Special Path

### Krea 2 TextFusion Projector

Two different base input-gradient shapes are possible:
- Trainable/full-weight projector: the forward input is cloned and flattened, giving `NN [12,40960,1,1]`.
- Frozen base during LoRA training: the base forward remains batched TT, giving `NN [12,2560,16,1]`.

The rank-16 factor gradients use the flattened trainable-factor path:

```text
backward through lora_up:   NN [16,40960,1,1]
backward through lora_down: NN [12,40960,1,16]
```

The two base alternatives are not additive.

### Qwen-Image 2.1 Image Projection

`img_in` has a `TT` forward and an `NN` input gradient: the forward keeps the column-major-like `[rows,64]` token view, while the backward GEMM `grad_tokens[rows,64] = grad_out[rows,4096] @ weight[4096,64]` lowers to `NN [64,rows,1,4096]`. A rank-16 `lora_down` on it keeps the same forward view and contributes `NN [64,rows,1,16]` through the factor.

The other 199 Linears are ordinary contiguous forwards, so their input gradients are `NN [in,rows,1,out]` with the packed `4112`/`9232` rows for the block Linears.

ComfyUI's prefix K/V cache is an inference-only optimisation: training runs the full packed sequence, so no cache-split NN rows are recorded.

### MOSS-Image Image Projection And Fused QKV

`video_embedder` has a `TT` forward in ComfyUI inference, but the sd-scripts training model materializes the token view with `.contiguous()`, so the forward is `TN` and the training input gradient is the ordinary `NN [128,rows,1,3072]`. The rank-16 `lora_down` factor contributes `NN [128,rows,1,16]` and its `lora_up` contributes `NN [16,rows,1,3072]`.

The fused-vs-unfused split changes the NN inventory too:
- The sd-scripts training model keeps `proj.q`, `proj.k`, and `proj.v` separate, so the block input gradients are four `NN [3072,rows,1,3072]` rows (q, k, v, o) plus `NN [3072,rows,1,16384]` for `ffn_up_gate` and `NN [8192,rows,1,3072]` for `ffn_down`, with `rows` equal to the packed `4112`/`9232`.
- A backward pass over the fused ComfyUI checkpoint would instead produce one `NN [3072,rows,1,9216]` qkv input gradient. That fused variant is not recorded, because ComfyUI has no MOSS-Image training loop and the released checkpoints are inference repacks.
- The text refiner runs before packing, so its input gradients use the 16-text-row band: `NN [3072,16,1,9216]` for `qkv`, `NN [3072,16,1,3072]` for `proj`, `NN [3072,16,1,8192]` for `ffn_gate` and `ffn_up`, and `NN [8192,16,1,3072]` for `ffn_down`.
- `final_linear_video` gets `NN [3072,image_rows,1,128]`, and the timestep/modulation Linears get one row each.

MOSS-Image has no prefix K/V cache, and its attention is a single full-sequence call per block, so no cache-split or segment-split NN rows exist.

## Training Implementations

The ordinary two-Linear LoRA execution pattern was checked in:
- `~/sd-scripts/networks/lora.py` and `~/sd-scripts/networks/lora_anima.py`.
- `~/musubi-tuner/src/musubi_tuner/networks/lora.py` and the model-specific target modules for Qwen-Image, Z-Image, Krea 2, Ideogram 4, Flux2/Klein, Wan, and H3.

MOSS-Image is implemented in sd-scripts: `~/sd-scripts/library/moss_image_models.py` (architecture) and `~/sd-scripts/moss_image_train_network.py` with `networks.lora_generic`. Its default target set is the 252 block projections, and its weight gradients, input gradients, and factor gradients are the same `(in,out,rows,rank)` pattern as the other two-Linear LoRA paths.

There is no Qwen-Image 2.1 arch module in musubi-tuner or sd-scripts yet, so its rows are the theoretical pattern: `QwenImage21TransformerBlock` holds 192 of the 200 Linears, and the remaining eight are `txt_in.in_layer`, `txt_in.out_layer`, `img_in`, the two timestep-embedder linears, `modulation.1`, `norm_out.linear`, and `proj_out`.

These implementations call a base forward followed by `lora_down` and `lora_up`. Their NT parameter-gradient rows, TN factor forwards, and NN factor input gradients therefore share the same logical `(in,out,rows,rank)` dimensions.
