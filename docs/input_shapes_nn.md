# gfx1151 FP16 NN Input Shapes

This note documents `NN` activation/input-gradient GEMM shapes for the training workloads already audited in `input_shapes_nt.md`.

Machine-readable rows are saved in `tmp_tensile_fp16_nt_hhs/shape_data/`:
- `nn_gemm_shapes.csv`: exact normalized base-Linear and rank-16 LoRA factor input-gradient rows.
- `nn_gemm_shapes_manifest.json`: derivation rules, conditional execution scope, alternate paths, and per-model counts.

## Scope

- Models: SDXL, Anima, Qwen, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, LTX 2.3, and H3.
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
- 683 aggregate rows.
- 441 unique exact NN shapes.
- 26,603 weighted logical matrix occurrences.

By primary category:

| Category | Aggregate rows | Unique shapes | Weighted occurrences |
| --- | ---: | ---: | ---: |
| Base Linear input gradient | 304 | 216 | 8,953 |
| Rank-16 LoRA factor input gradient | 379 | 225 | 17,650 |

Krea 2 contributes two workload-labelled alternate rows for the frozen-base batched projector path. They collapse to one additional unique exact shape, `NN [12,2560,16,1]`. Including alternatives gives 685 CSV rows and 442 unique shapes, but the alternate rows are not additive with the flattened full-weight projector row.

## Per-Model Counts

Counts below cover the primary derivation. Krea 2's two alternate rows are listed separately.

| Model | Base rows / unique / count | LoRA rows / unique / count |
| --- | --- | --- |
| SDXL | `23 / 21 / 1644` | `31 / 27 / 3288` |
| Anima | `25 / 19 / 969` | `32 / 24 / 1938` |
| Qwen | `26 / 18 / 1692` | `32 / 22 / 3384` |
| Z-Image | `40 / 29 / 416` | `62 / 37 / 832` |
| Krea 2 | `36 / 24 / 528` | `52 / 32 / 1056` |
| Ideogram 4 | `42 / 29 / 842` | `52 / 35 / 1684` |
| Klein 9B | `34 / 25 / 242` | `44 / 33 / 484` |
| Wan | `18 / 13 / 812` | `24 / 17 / 1624` |
| LTX 2.3 | `30 / 20 / 1280` | `34 / 24 / 2560` |
| H3 | `30 / 23 / 528` | `16 / 12 / 800` |

Krea 2 alternate: `2 / 1 / 2`, exact shape `[12,2560,16,1]` for both documented image sizes.

## Shape Construction By Model

For each logical base pair `(out,in)` below, the NN base family is `[in,rows,1,out]`. For each targeted rank-16 LoRA pair, the factor families are `[16,rows,1,out]` and `[in,rows,1,16]`. The listed row bands and feature pairs are inventories, not Cartesian products; use `nn_gemm_shapes.csv` for exact combinations and counts.

| Model | Activation-row bands | Base `(out,in)` feature pairs |
| --- | --- | --- |
| SDXL | `16, 1024, 2304, 4096, 9216` | `(640,640)`, `(640,2048)`, `(640,2560)`, `(768,768)`, `(768,3072)`, `(1280,1280)`, `(1280,2048)`, `(1280,5120)`, `(3072,768)`, `(3840,1280)`, `(5120,640)`, `(5120,1280)`, `(10240,1280)` |
| Anima | `1, 16, 512, 4096, 9216` | `(64,2048)`, `(256,2048)`, `(1024,1024)`, `(1024,4096)`, `(2048,68)`, `(2048,1024)`, `(2048,2048)`, `(2048,8192)`, `(4096,256)`, `(4096,1024)`, `(6144,256)`, `(6144,2048)`, `(8192,2048)` |
| Qwen | `1, 16, 4096, 9216` | `(64,3072)`, `(3072,64)`, `(3072,256)`, `(3072,3072)`, `(3072,3584)`, `(3072,12288)`, `(6144,3072)`, `(12288,3072)`, `(18432,3072)` |
| Z-Image | `1, 16, 32, 4096, 4128, 9216, 9248` | `(64,3840)`, `(256,1024)`, `(1024,256)`, `(3840,64)`, `(3840,256)`, `(3840,2560)`, `(3840,3840)`, `(3840,10240)`, `(10240,3840)`, `(11520,3840)`, `(15360,256)` |
| Krea 2 | `1, 16, 192, 4096, 4112, 9216, 9232, 40960` | `(1,12)`, `(64,6144)`, `(1536,6144)`, `(2560,2560)`, `(2560,6912)`, `(6144,64)`, `(6144,256)`, `(6144,2560)`, `(6144,6144)`, `(6144,16384)`, `(6912,2560)`, `(16384,6144)`, `(36864,6144)` |
| Ideogram 4 | `1, 16, 4096, 4112, 9216, 9232` | `(128,4608)`, `(512,4608)`, `(4608,128)`, `(4608,512)`, `(4608,4608)`, `(4608,12288)`, `(4608,53248)`, `(12288,4608)`, `(13824,4608)`, `(18432,512)` |
| Klein 9B | `1, 512, 4096, 4608, 9216, 9728` | `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,12288)`, `(4096,16384)`, `(8192,4096)`, `(12288,4096)`, `(24576,4096)`, `(36864,4096)` |
| Wan | `1, 512, 12032, 72192` | `(64,5120)`, `(5120,256)`, `(5120,4096)`, `(5120,5120)`, `(5120,13824)`, `(13824,5120)`, `(30720,5120)` |
| LTX 2.3 | `1, 1024, 1536, 8960` | `(32,4096)`, `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,16384)`, `(8192,4096)`, `(16384,4096)`, `(36864,4096)` |
| H3 | `2, 16, 256, 512, 5120, 5376, 25088, 25344` | `(32,5376)`, `(96,5376)`, `(5376,32)`, `(5376,96)`, `(5376,5120)`, `(5376,7168)`, `(5376,14336)`, `(10752,8)`, `(21504,5376)`, `(28672,5376)`, `(96768,8)` |

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

## Training Implementations

The ordinary two-Linear LoRA execution pattern was checked in:
- `~/sd-scripts/networks/lora.py` and `~/sd-scripts/networks/lora_anima.py`.
- `~/musubi-tuner/src/musubi_tuner/networks/lora.py` and the model-specific target modules for Qwen, Z-Image, Krea 2, Ideogram 4, Flux2/Klein, Wan, and H3.

These implementations call a base forward followed by `lora_down` and `lora_up`. Their NT parameter-gradient rows, TN factor forwards, and NN factor input gradients therefore share the same logical `(in,out,rows,rank)` dimensions.

