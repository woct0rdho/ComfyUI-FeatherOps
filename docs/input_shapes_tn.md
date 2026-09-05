# gfx1151 FP16 TN Input Shapes

This note documents `TN` GEMM shapes used by the inference and training workloads already audited in `input_shapes_nt.md`.

Machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/`:
- `tn_gemm_shapes.csv`: exact normalized base-Linear and unmerged rank-16 LoRA factor forward rows.
- `tn_gemm_shapes_manifest.json`: derivation rules, source files, special layouts, and per-model counts.

## Scope

- Models: SDXL, Anima, Qwen-Image 1, Qwen-Image 2.1, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, LTX 2.3, H3, and MOSS-Image.
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
- 851 aggregate rows.
- 484 unique exact TN shapes.
- 29,493 weighted logical matrix occurrences.

The total is a corpus inventory, not a one-call count. In an ordinary unmerged rank-16 path, base and factor rows are additive. Full-weight training or inference without an unmerged adapter uses only base rows. Krea 2, LTX 2.3, H3, Qwen-Image 2.1, and MOSS-Image have mode-dependent TN/TT or fused/unfused alternatives documented below, so their alternate rows must be selected by execution mode. Qwen-Image 2.1's prefix-cache rows are kept outside the totals below:

| Set | Aggregate rows | Unique shapes | Weighted occurrences |
| --- | ---: | ---: | ---: |
| Primary base and LoRA rows | 851 | 484 | 29,493 |
| Qwen-Image 2.1 prefix-cache alternates | 12 | 9 | 768 |
| Union | 863 | 486 | 30,261 |

By category:

| Category | Aggregate rows | Unique shapes | Weighted occurrences |
| --- | ---: | ---: | ---: |
| Base Linear forward | 360 | 242 | 9,971 |
| Unmerged rank-16 LoRA factor forward | 491 | 242 | 19,522 |

The base category includes ordinary inference/autograd forwards, SDXL text-encoder forwards, two H3 training-only audio projection rows, and two Krea 2 full-weight-training projector rows. LTX 2.3's two base patch projections and Qwen-Image 2.1's two `img_in` projections remain `TT` and are not counted here. Qwen-Image 2.1 also contributes 20 primary base rows, ten per size: its block Linears run on the packed `4112`/`9232`-row sequence, its text and modulation Linears use the 16- and 2-row bands, and `proj_out` uses the image rows. MOSS-Image contributes 38 primary base rows, 19 per size: 18 from the unfused training model (six block projections, the text embedder, five text-refiner projections, four timestep/modulation Linears, `video_embedder`, and `final_linear_video`) plus one `forward_fused_inference` row for the fused `qkv` weight that the released ComfyUI checkpoints store.

## Per-Model Counts

Counts aggregate all documented workloads for each model. `Count` is weighted by the number of logical matrices represented by each row.

| Model | Base rows / unique / count | LoRA rows / unique / count |
| --- | --- | --- |
| SDXL | `23 / 21 / 1644` | `31 / 27 / 3288` |
| Anima | `25 / 19 / 969` | `32 / 24 / 1938` |
| Qwen-Image 1 | `26 / 18 / 1692` | `32 / 22 / 3384` |
| Qwen-Image 2.1 | `20 / 13 / 398` | `42 / 22 / 798` |
| Z-Image | `40 / 29 / 416` | `62 / 37 / 832` |
| Krea 2 | `36 / 24 / 528` | `52 / 32 / 1056` |
| Ideogram 4 | `42 / 29 / 842` | `52 / 35 / 1684` |
| Klein 9B | `34 / 25 / 242` | `44 / 33 / 484` |
| Wan | `18 / 13 / 812` | `24 / 17 / 1624` |
| LTX 2.3 | `28 / 18 / 1278` | `32 / 22 / 2558` |
| H3 | `30 / 23 / 528` | `16 / 12 / 800` |
| MOSS-Image | `38 / 21 / 622` | `72 / 27 / 1076` |

Qwen-Image 2.1 additionally has `12 / 9 / 768` prefix-cache alternate rows, which are not additive with its base rows. MOSS-Image's 622 base occurrences cover 538 unfused training-model forwards (269 per size) plus 84 fused `qkv` inference forwards (42 per size). The fused row replaces the three per-projection `q/k/v` rows for inference and is not additive with them in a single run.

## Shape Families

The table below is a dimension inventory, not a Cartesian product. Each base pair is `(out_features,in_features)` and becomes TN `[out,rows,1,in]` only for the exact row bands paired with it in `tn_gemm_shapes.csv`. LoRA rows exist only for the targets represented by the source model's rank-16 NT rows.

| Model | Activation-row bands | Base `(out,in)` feature pairs |
| --- | --- | --- |
| SDXL | `16, 1024, 2304, 4096, 9216` | `(640,640)`, `(640,2048)`, `(640,2560)`, `(768,768)`, `(768,3072)`, `(1280,1280)`, `(1280,2048)`, `(1280,5120)`, `(3072,768)`, `(3840,1280)`, `(5120,640)`, `(5120,1280)`, `(10240,1280)` |
| Anima | `1, 16, 512, 4096, 9216` | `(64,2048)`, `(256,2048)`, `(1024,1024)`, `(1024,4096)`, `(2048,68)`, `(2048,1024)`, `(2048,2048)`, `(2048,8192)`, `(4096,256)`, `(4096,1024)`, `(6144,256)`, `(6144,2048)`, `(8192,2048)` |
| Qwen-Image 1 | `1, 16, 4096, 9216` | `(64,3072)`, `(3072,64)`, `(3072,256)`, `(3072,3072)`, `(3072,3584)`, `(3072,12288)`, `(6144,3072)`, `(12288,3072)`, `(18432,3072)` |
| Qwen-Image 2.1 | `1, 2, 16, 4096, 4112, 9216, 9232` | `(64,4096)`, `(4096,256)`, `(4096,4096)`, `(4096,12288)`, `(16384,4096)`, `(24576,4096)` |
| Z-Image | `1, 16, 32, 4096, 4128, 9216, 9248` | `(64,3840)`, `(256,1024)`, `(1024,256)`, `(3840,64)`, `(3840,256)`, `(3840,2560)`, `(3840,3840)`, `(3840,10240)`, `(10240,3840)`, `(11520,3840)`, `(15360,256)` |
| Krea 2 | `1, 16, 192, 4096, 4112, 9216, 9232, 40960` | `(1,12)`, `(64,6144)`, `(1536,6144)`, `(2560,2560)`, `(2560,6912)`, `(6144,64)`, `(6144,256)`, `(6144,2560)`, `(6144,6144)`, `(6144,16384)`, `(6912,2560)`, `(16384,6144)`, `(36864,6144)` |
| Ideogram 4 | `1, 16, 4096, 4112, 9216, 9232` | `(128,4608)`, `(512,4608)`, `(4608,128)`, `(4608,512)`, `(4608,4608)`, `(4608,12288)`, `(4608,53248)`, `(12288,4608)`, `(13824,4608)`, `(18432,512)` |
| Klein 9B | `1, 512, 4096, 4608, 9216, 9728` | `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,12288)`, `(4096,16384)`, `(8192,4096)`, `(12288,4096)`, `(24576,4096)`, `(36864,4096)` |
| Wan | `1, 512, 12032, 72192` | `(64,5120)`, `(5120,256)`, `(5120,4096)`, `(5120,5120)`, `(5120,13824)`, `(13824,5120)`, `(30720,5120)` |
| LTX 2.3 | `1, 1024, 1536, 8960` | `(32,4096)`, `(128,4096)`, `(4096,128)`, `(4096,256)`, `(4096,4096)`, `(4096,16384)`, `(8192,4096)`, `(16384,4096)`, `(36864,4096)` |
| H3 | `2, 16, 256, 512, 5120, 5376, 25088, 25344` | `(32,5376)`, `(96,5376)`, `(5376,32)`, `(5376,96)`, `(5376,5120)`, `(5376,7168)`, `(5376,14336)`, `(10752,8)`, `(21504,5376)`, `(28672,5376)`, `(96768,8)` |
| MOSS-Image | `1, 16, 4096, 4112, 9216, 9232` | `(128,3072)`, `(3072,128)`, `(3072,256)`, `(3072,3072)`, `(3072,4096)`, `(3072,6144)`, `(3072,8192)`, `(3072,18432)`, `(6144,3072)`, `(8192,3072)`, `(9216,3072)`, `(16384,3072)`, `(18432,3072)` |

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

### Qwen-Image 2.1 Image Projection And Prefix Cache

- `img_in = Linear(64,4096)` consumes the `img.flatten(2).transpose(1,2)` token view and stays `TT`, so its `(4096,64)` pair is absent from the TN table above. A rank-16 `lora_down` on it is also `TT`.
- Every other Qwen-Image 2.1 Linear sees a contiguous activation, so `txt_in`, `modulation`, the block attention/MLP linears, `norm_out.linear`, `proj_out`, and all `lora_up` rows are TN.
- The block Linears run on the packed sequence: `16 + 4096 = 4112` rows at `1024x1024` and `16 + 9216 = 9232` rows at `1536x1536`.
- Without weight hooks, ComfyUI enables a per-sampling-run prefix K/V cache and then executes the block Linears twice on the first step: a `16`-row text-prefix pass and a target-only pass on the image rows. Later steps run only the target pass. Those rows are the 12 alternates listed above and are not additive with the packed rows. Hooks such as LoRA disable the cache, which restores the packed row count.

### MOSS-Image Fused QKV And Image Projection

- ComfyUI's released MOSS-Image checkpoints store one fused `layers.N.proj.qkv = Linear(3072,9216)` weight. Inference issues one `TN [9216,4112,1,3072]` GEMM per block at `1024x1024` and `[9216,9232,1,3072]` at `1536x1536`. The row is recorded with phase `forward_fused_inference`.
- The sd-scripts training model, and the official diffusers shards, keep `proj.q`, `proj.k`, and `proj.v` separate. Those three `TN [3072,rows,1,3072]` rows are the primary rows in `tn_gemm_shapes.csv`, so a single training forward issues 269 base GEMMs while a single fused ComfyUI forward issues 185. The fused row and the three per-projection rows describe the same arithmetic and are not additive within one run.
- The text embedder runs before the sequence is packed, so `text_embedder.proj`, the two text-refiner `qkv`/`proj`/`ffn_gate`/`ffn_up`/`ffn_down` projections use the 16-text-row band. The text tokens are then concatenated after the image tokens and every main-block Linear sees the packed `4112`/`9232` rows.
- `video_embedder = Linear(128,3072)` is `TT` in ComfyUI inference and `TN` in sd-scripts training, where the token view is materialized with `.contiguous()`. Both forms are recorded.
- The timestep path is `time_embedding.0` `[3072,1,1,256]`, `time_embedding.2` `[3072,1,1,3072]`, `shared_block_modulation` `[18432,1,1,3072]`, and `video_head_modulation` `[6144,1,1,3072]`, all on one batch-1 row.
- `final_linear_video = Linear(3072,128)` runs on the image rows only, because the text rows are sliced off before the final modulation and projection.
- MOSS-Image has no prefix K/V cache and no block-causal masking, so there are no cache alternates and no per-segment rows.

## Model Notes

- SDXL includes UNet Transformer Linears and the CLIP-L/OpenCLIP-G text-encoder Linears represented by the existing NT training rows. The original forward CSV omitted the text-encoder forward aggregates. They are restored here by exact NT-to-TN derivation.
- Qwen-Image 1 processes text and image Linears on separate row counts before joint fused attention, so its GEMM bands are 16 and 4096/9216 rather than the packed attention lengths 4112/9232.
- Qwen-Image 2.1 is single-stream: the 32 blocks are identical and every block Linear sees the packed rows. Its `modulation` is one shared `Linear(4096,16384)` outside the blocks, and `img_mlp.gate_up` is a fused `[24576,4096]` projection. The text-token band of 16 is the short-prompt convention. The real T2I template keeps about 7 fixed tokens around the prompt, which shifts only the text and projection rows.
- Z-Image includes context-refiner, noise-refiner, and main packed-stream rows. Its padded main bands are 4128 and 9248.
- Ideogram 4 conditional packed and image-only paths are alternatives, as documented in `input_shapes_nt.md`.
- Klein 9B and Wan retain their fixed 512-row diffusion text contexts.
- LTX 2.3 remains the documented no-audio workload.
- H3 uses the pruned-AdaLN checkpoint and the Musubi default 200-linear LoRA target set. INT8 ConvRot weights are treated as logical dequantized BF16 matrices.
- MOSS-Image is a 42-block single-stream DiT with width 3072, FFN 8192, and 24 heads of 128. The released ComfyUI bf16 and INT8 ConvRot files are fused-`qkv` repacks. The INT8 file quantizes the 168 block Linears per row and is treated here as logical dequantized BF16 matrices. The default sd-scripts LoRA target set is the 252 block projections (`proj.q/k/v/o`, `ffn_up_gate`, `ffn_down`). The documented factor rows cover all 269 logical Linears.
