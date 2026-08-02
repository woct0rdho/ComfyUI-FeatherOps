# gfx1151 FP16 TN Input Shapes

This note documents `TN` GEMM shapes used by the inference and training workloads already audited in `input_shapes_nt.md`.

Machine-readable rows are saved in `tmp_tensile_fp16_nt_hhs/shape_data/`:
- `tn_gemm_shapes.csv`: exact normalized base-Linear and unmerged rank-16 LoRA factor forward rows.
- `tn_gemm_shapes_manifest.json`: derivation rules, source files, special layouts, and per-model counts.

## Scope

- Models: SDXL, Anima, Qwen, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, LTX 2.3, and H3.
- Workload sizes, prompt/context lengths, video geometry, checkpoint boundaries, and row rounding match `input_shapes_nt.md`.
- Exact rows use TensileLite order `[M,N,batch,K]`.
- Quantized checkpoints are represented by their logical dequantized Linear dimensions. Scale tensors and quantization metadata are excluded.
- Base Linear forwards and separate unmerged rank-16 LoRA factor forwards are included. Convolutions and fused-attention internals remain excluded.
- The base forward remains active in unmerged LoRA mode, where `lora_down` and `lora_up` are additional calls. Merged LoRA inference folds the factor product into the base weight and uses only the base row.

## Layout Rule

For the hipBLASLt convention used by this repository, `T` denotes a row-major tensor stride and `N` denotes a column-major/view-transposed stride. A regular PyTorch Linear over a contiguous activation matrix lowers after the row-major output operand swap as:

```text
weight[out,in] @ input[rows,in].T
TN [M=out, N=rows, batch=1, K=in]
```

The documented `NT` parameter-gradient row for the same matrix is:

```text
NT [M=in, N=out, batch=1, K=rows]
```

Therefore the base or factor forward row can be derived exactly by:

```text
NT [in,out,batch,rows] -> TN [out,rows,batch,in]
```

For rank-16 LoRA attached to a logical Linear `(out,in)`:

```text
lora_down: TN [16,rows,1,in]
lora_up:   TN [out,rows,1,16]
```

These factor rows execute during training and during inference when the LoRA is left unmerged. A merged LoRA uses only the base Linear row.

## Aggregate Counts

The normalized TN dataset contains:
- 679 aggregate rows.
- 437 unique exact TN shapes.
- 26,599 weighted logical matrix occurrences.

The total is a corpus inventory, not a one-call count. In an ordinary unmerged rank-16 path, base and factor rows are additive; full-weight training or inference without an unmerged adapter uses only base rows. Krea 2, LTX 2.3, and H3 have mode-dependent TN/TT alternatives documented below, so their alternate rows must be selected by execution mode.

By category:

| Category | Aggregate rows | Unique shapes | Weighted occurrences |
| --- | ---: | ---: | ---: |
| Base Linear forward | 302 | 214 | 8,951 |
| Unmerged rank-16 LoRA factor forward | 377 | 223 | 17,648 |

The base category includes ordinary inference/autograd forwards, SDXL text-encoder forwards, two H3 training-only audio projection rows, and two Krea 2 full-weight-training projector rows. LTX 2.3's two base patch projections remain `TT` and are not counted here.

## Per-Model Counts

Counts aggregate all documented workloads for each model. `Count` is weighted by the number of logical matrices represented by each row.

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
| LTX 2.3 | `28 / 18 / 1278` | `32 / 22 / 2558` |
| H3 | `30 / 23 / 528` | `16 / 12 / 800` |

## Shape Families

The table below is a dimension inventory, not a Cartesian product. Each base pair is `(out_features,in_features)` and becomes TN `[out,rows,1,in]` only for the exact row bands paired with it in `tn_gemm_shapes.csv`. LoRA rows exist only for the targets represented by the source model's rank-16 NT rows.

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

## Special Layout Paths

### Krea 2 TextFusion Projector

- In no-grad inference and frozen-base LoRA training, the base `Linear(12,1)` remains batched `TT [1,2560,16,12]`.
- If the base projector weight itself requires gradients, PyTorch clones and flattens the high-rank input. The base forward becomes `TN [1,40960,1,12]`.
- Its unmerged rank-16 factors use flattened TN rows `lora_down=[16,40960,1,12]` and `lora_up=[1,40960,1,16]`.

### LTX 2.3 Patch Projection

- Base `patchify_proj` remains TT at `[4096,1536,1,128]` and `[4096,8960,1,128]` during inference and autograd forward.
- The rank-16 `lora_down` sees the same column-major-like 2D patch view and remains TT at `[16,1536,1,128]` and `[16,8960,1,128]`.
- `lora_up` consumes a regular contiguous rank-16 activation and is TN. The exact up row is included in the aggregate LTX output-4096 LoRA rows.

### H3 Audio Projection

- No-reference ComfyUI inference uses TT for `audio_patch_proj` because `pack_audio` preserves stride `(1,audio_rows)`.
- Musubi concatenates/materializes the packed audio input before training, producing TN `[5376,256,1,32]` and `[5376,512,1,32]` under the normalized row convention.
- H3's documented default LoRA target excludes this projection, so there are no rank-16 factor rows for it.

## Model Notes

- SDXL includes UNet Transformer Linears and the CLIP-L/OpenCLIP-G text-encoder Linears represented by the existing NT training rows. The original forward CSV omitted the text-encoder forward aggregates; they are restored here by exact NT-to-TN derivation.
- Qwen processes text and image Linears on separate row counts before joint fused attention, so its GEMM bands are 16 and 4096/9216 rather than the packed attention lengths 4112/9232.
- Z-Image includes context-refiner, noise-refiner, and main packed-stream rows; its padded main bands are 4128 and 9248.
- Ideogram 4 conditional packed and image-only paths are alternatives, as documented in `input_shapes_nt.md`.
- Klein 9B and Wan retain their fixed 512-row diffusion text contexts.
- LTX 2.3 remains the documented no-audio workload.
- H3 uses the pruned-AdaLN checkpoint and the Musubi default 200-linear LoRA target set. INT8 ConvRot weights are treated as logical dequantized BF16 matrices.
