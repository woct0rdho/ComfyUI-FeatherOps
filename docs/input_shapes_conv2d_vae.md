# gfx1151 FP16 2D Conv VAE Input Shapes

This note documents the `Conv2d` tensor shapes and their implicit-GEMM decompositions for the VAEs of the image models in this workload set, plus the LTX 2.3 audio (mel-spectrogram) autoencoder.

Machine-readable rows are saved in `~/tmp/feather_ops/conv_vae_shapes/shape_data/`:
- `conv2d_vae_shapes.csv`: one aggregated row per `(model, VAE, workload, phase, module, effective weight, input shape, output shape)` with the derived `m, n, k, batch` and the number of identical calls in `count`.
- `conv2d_vae_unique_gemm_shapes.csv`: unique `(op_kind, m, n, k)` rows with call counts and the models that use them.
- `conv2d_vae_shapes_manifest.json`: per-model counts, workload geometry, VAE checkpoint/code sources, column semantics, and caveats.

## Scope

- Models: SDXL, Anima, Qwen-Image 1, Qwen-Image 2.1, Z-Image, Krea 2, Ideogram 4, Klein 9B, MOSS-Image, plus the LTX 2.3 audio VAE.
- Inference only: one VAE `encode` and one `decode` per documented workload, batch size 1. Training-only parameter-gradient shapes are not recorded. The VAE is normally frozen even when the diffusion model is trained.
- Image workloads: `1024x1024` and `1536x1536`, matching `input_shapes_nt.md` and `input_shapes_tn.md`. `1536x1536` scales every spatial position count by `2.25` relative to `1024x1024`.
- LTX 2.3 audio workloads: the 5.12 s training-length clip (512 mel frames) and the audio spans of the documented `640x480x40` and `1280x720x80` video clips (165 and 329 mel frames, see the geometry section).
- Every recorded operator is a `torch.nn.Conv2d` (`groups=1`, dilation 1, kernel 3x3 or 1x1). None of these VAEs uses `ConvTranspose2d`: upsampling is `nn.Upsample(nearest-exact)` followed by a 3x3 conv, and downsampling is `ZeroPad2d` followed by a stride-2 3x3 conv.
- Excluded: matmuls inside attention kernels. VAE attention blocks still contribute their 1x1 projection convs (`to_qkv`/`q`/`k`/`v`/`proj_out`), which are recorded because they are ordinary conv operators. Also excluded: GroupNorm/RMSNorm, quantization, and the diffusion-model Linears already covered by the GEMM notes.

## Implicit GEMM Mapping

For a 2D convolution with weight `[C_out, C_in, kh, kw]` applied to `[B, C_in, H_in, W_in]` and producing `[B, C_out, H_out, W_out]`:

```text
M = C_out
N = H_out * W_out          # per batch element
K = C_in * kh * kw
batch = B                  # the conv call's batch axis
MACs = M * N * K * batch
```

- A `1x1` conv is a plain GEMM with `K = C_in`. Those are the attention projections, the residual `conv_shortcut`/`nin_shortcut` layers, the `quant_conv`/`post_quant_conv` layers, and the Wan 2.2 latent projections.
- A `3x3` conv has `K = 9 * C_in`. On implicit-GEMM hardware it is a GEMM followed (or preceded, depending on layout) by im2col/col2im data movement. The `m,n,k` triple here is the GEMM part.
- `n` is the per-call value. `n_total = batch * n` and `flops_per_call = 2 * m * n * k * batch` are also in the CSV, so a pass total is `sum(flops_per_call * count)`.
- All rows have `groups=1`. No grouped or depthwise conv appears in these VAEs.
- ComfyUI can split a VAE pass into spatial tiles (`VAEDecodeTiled`, `VAEEncodeTiled`) or row strips (Flux/Flux2 `strip_apply`). Those reduce `N` per call without changing `M`, `K`, or total MACs. The rows here are the single-pass model calls.

## Model To VAE Mapping

| Model | VAE | Weights | Latent | VAE conv modules | Conv calls per pass (encode + decode) |
| --- | --- | --- | --- | ---: | ---: |
| SDXL | `AutoencoderKL` (SDXL VAE) | embedded in the SDXL checkpoint (`first_stage_model.*`) | 4 ch, `/8` | 72 | 72 |
| Anima | Qwen 2D VAE | `qwen_2d_vae.safetensors` | 16 ch, `/8` | 67 | 67 |
| Qwen-Image 1 | Qwen 2D VAE | `qwen_2d_vae.safetensors` | 16 ch, `/8` | 67 | 67 |
| Krea 2 | Qwen 2D VAE | `qwen_2d_vae.safetensors` | 16 ch, `/8` | 67 | 67 |
| Z-Image | Flux VAE (`ae.safetensors`) | Comfy-Org `z_image_turbo` | 16 ch, `/8` | 70 | 70 |
| Ideogram 4 | Flux 2 small VAE | `flux2_vae_small.safetensors` | 128 packed ch, `/16` | 72 | 72 |
| Klein 9B | Flux 2 small VAE | `flux2_vae_small.safetensors` | 128 packed ch, `/16` | 72 | 72 |
| MOSS-Image | Flux 2 small VAE | `flux2_vae_small.safetensors` | 128 packed ch, `/16` | 72 | 72 |
| Qwen-Image 2.1 | Qwen-Image 2.1 VAE (Wan 2.2 layout, 2D) | `qwen_image_2.1_vae_2d.safetensors` | 64 ch, `/16`, RGBA | 82 | 82 |
| LTX 2.3 (audio) | LTX audio autoencoder (mel 2D) | `audio_vae.*` in `ltx-2.3-22b-dev.safetensors` | 8 ch, `/4` on time and mel | 50 | 50 |

The `Anima` / `Qwen-Image 1` / `Krea 2` rows share one VAE. The local ComfyUI workflows (`user/default/workflows/anima_feather.json`, `krea2_txt2img.json`, `qwen_image*.json`) all load `qwen_2d_vae.safetensors`, which is the Qwen-Image 1 VAE converted to 2D kernels (see below). `Ideogram 4`, `Klein 9B`, and `MOSS-Image` all load `flux2_vae_small.safetensors`.

## Aggregate Counts

The normalized 2D dataset contains:
- 1,432 aggregated rows (one per distinct shape signature and call count).
- 221 unique `(op_kind, m, n, k)` shapes.
- 10 models, 6 distinct VAE architectures, 21 model-workload combinations in the CSV (nine image models x two sizes, plus the three LTX audio clips).

| Model | Unique weight shapes | Unique `(op,m,n,k)` | `1024x1024` calls / GMACs (enc + dec) | `1536x1536` calls / GMACs (enc + dec) |
| --- | ---: | ---: | ---: | ---: |
| SDXL | 18 | 46 | 32 + 40 / 2164.6 + 4960.3 | 32 + 40 / 4870.3 + 11160.7 |
| Anima | 17 | 42 | 30 + 37 / 1219.8 + 2149.0 | 30 + 37 / 2744.4 + 4835.2 |
| Qwen-Image 1 | 17 | 42 | 30 + 37 / 1219.8 + 2149.0 | 30 + 37 / 2744.4 + 4835.2 |
| Krea 2 | 17 | 42 | 30 + 37 / 1219.8 + 2149.0 | 30 + 37 / 2744.4 + 4835.2 |
| Z-Image | 16 | 42 | 31 + 39 / 2166.4 + 4961.2 | 31 + 39 / 4874.4 + 11162.8 |
| Ideogram 4 | 22 | 56 | 32 + 40 / 2168.9 + 2792.5 | 32 + 40 / 4880.0 + 6283.2 |
| Klein 9B | 22 | 56 | 32 + 40 / 2168.9 + 2792.5 | 32 + 40 / 4880.0 + 6283.2 |
| MOSS-Image | 22 | 56 | 32 + 40 / 2168.9 + 2792.5 | 32 + 40 / 4880.0 + 6283.2 |
| Qwen-Image 2.1 | 30 | 76 | 36 + 46 / 1553.5 + 8477.2 | 36 + 46 / 3495.4 + 19073.6 |

LTX 2.3 audio:

| Workload | Mel frames | Latent `[C,T,F]` | Conv calls (enc + dec) | GMACs (enc + dec) |
| --- | ---: | --- | ---: | ---: |
| 5.12 s clip | 512 | `8x128x16` | 22 + 28 | 75.7 + 156.5 |
| `640x480x40` span | 165 | `8x42x16` | 22 + 28 | 24.6 + 51.1 |
| `1280x720x80` span | 329 | `8x83x16` | 22 + 28 | 48.9 + 101.3 |

The image VAEs use four distinct per-call spatial extents at `1024x1024` (`16384, 65536, 262144, 1048576` positions) plus, for Qwen-Image 2.1, `4096` (the `64x64` latent level). At `1536x1536` the same sets scale to `36864, 147456, 589824, 2359296` and `9216` for Qwen-Image 2.1.

## VAE Shape Families

Each VAE is built from a small number of `(M, K)` conv pairs. `M` is the output channel count and `K` is `C_in * kernel_product`. The `N` axis is shared by all modules at the same resolution.

### SDXL

Facts: `AutoencoderKL` with `z_channels=4`, `ch=128`, `ch_mult=[1,2,4,4]`, two residual blocks per level, one attention block in the middle, no attention at other levels. Latent is `4x128x128` at `1024x1024` and `4x192x192` at `1536x1536`. 72 conv modules: 32 encoder calls, 40 decoder calls.

`(M, K)` families (18 families, 72 modules. The last column names a representative module):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 512 | 4608 | 3x3 | 30 | `decoder.mid.block_1.conv1` |
| 128 | 1152 | 3x3 | 10 | `decoder.up.0.block.0.conv2` |
| 256 | 2304 | 3x3 | 10 | `decoder.up.1.block.0.conv2` |
| 512 | 512 | 1x1 | 8 | `decoder.mid.attn_1.k` (mid attention q/k/v/proj_out in both encoder and decoder) |
| 3 | 1152 | 3x3 | 1 | `decoder.conv_out` |
| 4 | 4 | 1x1 | 1 | `post_quant_conv` |
| 8 | 8 | 1x1 | 1 | `quant_conv` |
| 8 | 4608 | 3x3 | 1 | `encoder.conv_out` (2*z) |
| 128 | 27 | 3x3 | 1 | `encoder.conv_in` |
| 128 | 256 | 1x1 | 1 | `decoder.up.0.block.0.nin_shortcut` |
| 128 | 2304 | 3x3 | 1 | `decoder.up.0.block.0.conv1` |
| 256 | 128 | 1x1 | 1 | `encoder.down.1.block.0.nin_shortcut` |
| 256 | 512 | 1x1 | 1 | `decoder.up.1.block.0.nin_shortcut` |
| 256 | 1152 | 3x3 | 1 | `encoder.down.1.block.0.conv1` |
| 256 | 4608 | 3x3 | 1 | `decoder.up.1.block.0.conv1` |
| 512 | 36 | 3x3 | 1 | `decoder.conv_in` |
| 512 | 256 | 1x1 | 1 | `encoder.down.2.block.0.nin_shortcut` |
| 512 | 2304 | 3x3 | 1 | `encoder.down.2.block.0.conv1` |

Per-call inventory at `1024x1024` (encoder then decoder). The `1536x1536` run has identical `M`/`K` and multiplies every `N` by `2.25`.

Encoder:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `128,3,3,3` | 3x3 | 1,1 | 1,1 | `1,3,1024,1024` | `1,128,1024,1024` | 128 | 27 | 1048576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,128,1024,1024` | 128 | 1152 | 1048576 | 4 | 4 |
| `128,128,3,3` | 3x3 | 2,2 | 0,0 | `1,128,1025,1025` | `1,128,512,512` | 128 | 1152 | 262144 | 1 | 1 |
| `256,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,512,512` | `1,256,512,512` | 256 | 1152 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,512,512` | `1,256,512,512` | 256 | 2304 | 262144 | 3 | 3 |
| `256,128,1,1` | 1x1 | 1,1 | 0,0 | `1,128,512,512` | `1,256,512,512` | 256 | 128 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 2,2 | 0,0 | `1,256,513,513` | `1,256,256,256` | 256 | 2304 | 65536 | 1 | 1 |
| `512,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,256,256` | `1,512,256,256` | 512 | 2304 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,256,256` | `1,512,256,256` | 512 | 4608 | 65536 | 3 | 3 |
| `512,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,256,256` | `1,512,256,256` | 512 | 256 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 2,2 | 0,0 | `1,512,257,257` | `1,512,128,128` | 512 | 4608 | 16384 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,512,128,128` | 512 | 4608 | 16384 | 8 | 8 |
| `512,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,128,128` | `1,512,128,128` | 512 | 512 | 16384 | 4 | 4 |
| `8,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,8,128,128` | 8 | 4608 | 16384 | 1 | 1 |
| `8,8,1,1` | 1x1 | 1,1 | 0,0 | `1,8,128,128` | `1,8,128,128` | 8 | 8 | 16384 | 1 | 1 |

Decoder:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `4,4,1,1` | 1x1 | 1,1 | 0,0 | `1,4,128,128` | `1,4,128,128` | 4 | 4 | 16384 | 1 | 1 |
| `512,4,3,3` | 3x3 | 1,1 | 1,1 | `1,4,128,128` | `1,512,128,128` | 512 | 36 | 16384 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,512,128,128` | 512 | 4608 | 16384 | 8 | 8 |
| `512,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,128,128` | `1,512,128,128` | 512 | 512 | 16384 | 4 | 4 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,256,256` | `1,512,256,256` | 512 | 4608 | 65536 | 7 | 7 |
| `256,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,512,512` | `1,256,512,512` | 256 | 4608 | 262144 | 1 | 1 |
| `256,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,512,512` | `1,256,512,512` | 256 | 512 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,512,512` | `1,256,512,512` | 256 | 2304 | 262144 | 5 | 5 |
| `128,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,1024,1024` | `1,128,1024,1024` | 128 | 2304 | 1048576 | 1 | 1 |
| `128,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,1024,1024` | `1,128,1024,1024` | 128 | 256 | 1048576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,128,1024,1024` | 128 | 1152 | 1048576 | 5 | 5 |
| `3,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,3,1024,1024` | 3 | 1152 | 1048576 | 1 | 1 |

The `Upsample` + 3x3 conv pattern shows up as a 3x3 conv whose input `H` is already doubled (`1024 -> 1024` at the same resolution), and the `ZeroPad2d` + stride-2 conv shows up as a `1024+1 -> 512` input (the padding is outside the conv module, so the recorded input is `1025x1025` and `padding=(0,0)`).

### Anima / Qwen-Image 1 / Krea 2 (Qwen 2D VAE)

Facts: `Qwen2DVAE` from `custom_nodes/ComfyUI-Anzhc-Qwen2D/qwen2d_arch.py`, `base_dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`, two residual blocks per level, one mid attention block, RMSNorm (`gamma`), no temporal operators. Latent is `16x128x128` at `1024x1024` and `16x192x192` at `1536x1536`. 67 conv modules: 30 encoder calls, 37 decoder calls.

`(M, K)` families (17 families, 67 modules):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 384 | 3456 | 3x3 | 27 | `decoder.mid_block.resnets.0.conv1` |
| 96 | 864 | 3x3 | 11 | `decoder.up_blocks.3.resnets.0.conv1` |
| 192 | 1728 | 3x3 | 10 | `decoder.up_blocks.2.resnets.0.conv1` |
| 192 | 3456 | 3x3 | 2 | `decoder.up_blocks.0.upsamplers.0.resample.1` (upsampler 3x3 convs) |
| 384 | 192 | 1x1 | 2 | `decoder.up_blocks.1.resnets.0.conv_shortcut` |
| 384 | 384 | 1x1 | 2 | mid attention `proj` (encoder + decoder) |
| 384 | 1728 | 3x3 | 2 | `decoder.up_blocks.1.resnets.0.conv1` |
| 1152 | 384 | 1x1 | 2 | mid attention `to_qkv` (encoder + decoder) |
| 3 | 864 | 3x3 | 1 | `decoder.conv_out` |
| 16 | 16 | 1x1 | 1 | `post_quant_conv` |
| 32 | 32 | 1x1 | 1 | `quant_conv` |
| 32 | 3456 | 3x3 | 1 | `encoder.conv_out` (2*z) |
| 96 | 27 | 3x3 | 1 | `encoder.conv_in` |
| 96 | 1728 | 3x3 | 1 | `decoder.up_blocks.2.upsamplers.0.resample.1` |
| 192 | 96 | 1x1 | 1 | `encoder.down_blocks.3.conv_shortcut` |
| 192 | 864 | 3x3 | 1 | `encoder.down_blocks.3.conv1` |
| 384 | 144 | 3x3 | 1 | `decoder.conv_in` |

Encoder at `1024x1024`:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `96,3,3,3` | 3x3 | 1,1 | 1,1 | `1,3,1024,1024` | `1,96,1024,1024` | 96 | 27 | 1048576 | 1 | 1 |
| `96,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,96,1024,1024` | 96 | 864 | 1048576 | 4 | 4 |
| `96,96,3,3` | 3x3 | 2,2 | 0,0 | `1,96,1025,1025` | `1,96,512,512` | 96 | 864 | 262144 | 1 | 1 |
| `192,96,1,1` | 1x1 | 1,1 | 0,0 | `1,96,512,512` | `1,192,512,512` | 192 | 96 | 262144 | 1 | 1 |
| `192,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,512,512` | `1,192,512,512` | 192 | 864 | 262144 | 1 | 1 |
| `192,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,512,512` | `1,192,512,512` | 192 | 1728 | 262144 | 3 | 3 |
| `192,192,3,3` | 3x3 | 2,2 | 0,0 | `1,192,513,513` | `1,192,256,256` | 192 | 1728 | 65536 | 1 | 1 |
| `384,192,1,1` | 1x1 | 1,1 | 0,0 | `1,192,256,256` | `1,384,256,256` | 384 | 192 | 65536 | 1 | 1 |
| `384,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,256,256` | `1,384,256,256` | 384 | 1728 | 65536 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,256,256` | `1,384,256,256` | 384 | 3456 | 65536 | 3 | 3 |
| `384,384,3,3` | 3x3 | 2,2 | 0,0 | `1,384,257,257` | `1,384,128,128` | 384 | 3456 | 16384 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,128,128` | `1,384,128,128` | 384 | 3456 | 16384 | 8 | 8 |
| `1152,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,1152,128,128` | 1152 | 384 | 16384 | 1 | 1 |
| `384,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,384,128,128` | 384 | 384 | 16384 | 1 | 1 |
| `32,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,128,128` | `1,32,128,128` | 32 | 3456 | 16384 | 1 | 1 |
| `32,32,1,1` | 1x1 | 1,1 | 0,0 | `1,32,128,128` | `1,32,128,128` | 32 | 32 | 16384 | 1 | 1 |

Decoder at `1024x1024`:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `16,16,1,1` | 1x1 | 1,1 | 0,0 | `1,16,128,128` | `1,16,128,128` | 16 | 16 | 16384 | 1 | 1 |
| `384,16,3,3` | 3x3 | 1,1 | 1,1 | `1,16,128,128` | `1,384,128,128` | 384 | 144 | 16384 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,128,128` | `1,384,128,128` | 384 | 3456 | 16384 | 8 | 8 |
| `384,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,384,128,128` | 384 | 384 | 16384 | 1 | 1 |
| `1152,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,1152,128,128` | 1152 | 384 | 16384 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,256,256` | `1,384,256,256` | 384 | 3456 | 65536 | 6 | 6 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,512,512` | `1,192,512,512` | 192 | 3456 | 262144 | 1 | 1 |
| `192,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,512,512` | `1,192,512,512` | 192 | 384 | 262144 | 1 | 1 |
| `192,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,512,512` | `1,192,512,512` | 192 | 1728 | 262144 | 5 | 5 |
| `96,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,1024,1024` | `1,96,1024,1024` | 96 | 1728 | 1048576 | 1 | 1 |
| `96,192,1,1` | 1x1 | 1,1 | 0,0 | `1,192,1024,1024` | `1,96,1024,1024` | 96 | 192 | 1048576 | 1 | 1 |
| `96,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,96,1024,1024` | 96 | 864 | 1048576 | 9 | 9 |
| `3,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,3,1024,1024` | 3 | 864 | 1048576 | 1 | 1 |

3D-origin note: the released `qwen_image_vae.safetensors` for this family (Comfy-Org `Qwen-Image_ComfyUI`, `Krea-2`, and `circlestone-labs/Anima` repos) is a Wan 2.1-shaped causal 3D VAE: 194 tensors, `Conv3d` weights such as `decoder.conv1.weight [384,16,3,3,3]` plus `conv1/conv2` 1x1x1 latent projections and `time_conv` stages. Because a text-to-image encode/decode feeds a single frame (`T=1`) and a causal 3D conv with `padding=(1,1,1)` is executed by ComfyUI with `autopad="causal_zero"`, only the last temporal tap contributes, so every `3x3x3` conv is arithmetically the `3x3` 2D conv on that slice. The 2D checkpoint used here (`qwen_2d_vae.safetensors`, 186 tensors) stores exactly those slices, drops the four `conv1`/`conv2` 1x1x1 tensors and the four `time_conv` tensors, and keeps all other weights unchanged. For the single-image path the GEMM shapes are identical between the two forms: `K = C_in * 1 * 3 * 3` for the 3D form with temporal truncation and `K = C_in * 3 * 3` for the 2D form.

### Z-Image (Flux VAE)

Facts: `AutoencodingEngine` + `AutoencoderKL` regularizer with `z_channels=16`, `ch=128`, `ch_mult=[1,2,4,4]`, two residual blocks per level, one mid attention block. Latent is `16x128x128` at `1024x1024` and `16x192x192` at `1536x1536`. 70 conv modules: 31 encoder calls, 39 decoder calls. Unlike SDXL, this VAE has no `quant_conv`/`post_quant_conv`: the encoder ends in a 32-channel `conv_out` and the decoder starts with a 3x3 conv from the 16 latent channels.

`(M, K)` families (16 families, 70 modules):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 512 | 4608 | 3x3 | 30 | `decoder.mid.block_1.conv1` |
| 128 | 1152 | 3x3 | 10 | `decoder.up.0.block.0.conv2` |
| 256 | 2304 | 3x3 | 10 | `decoder.up.1.block.0.conv2` |
| 512 | 512 | 1x1 | 8 | mid attention `q/k/v/proj_out` (encoder + decoder) |
| 3 | 1152 | 3x3 | 1 | `decoder.conv_out` |
| 32 | 4608 | 3x3 | 1 | `encoder.conv_out` (2*z) |
| 128 | 27 | 3x3 | 1 | `encoder.conv_in` |
| 128 | 256 | 1x1 | 1 | `decoder.up.0.block.0.nin_shortcut` |
| 128 | 2304 | 3x3 | 1 | `decoder.up.0.block.0.conv1` |
| 256 | 128 | 1x1 | 1 | `encoder.down.1.block.0.nin_shortcut` |
| 256 | 512 | 1x1 | 1 | `decoder.up.1.block.0.nin_shortcut` |
| 256 | 1152 | 3x3 | 1 | `encoder.down.1.block.0.conv1` |
| 256 | 4608 | 3x3 | 1 | `decoder.up.1.block.0.conv1` |
| 512 | 144 | 3x3 | 1 | `decoder.conv_in` (z=16) |
| 512 | 256 | 1x1 | 1 | `encoder.down.2.block.0.nin_shortcut` |
| 512 | 2304 | 3x3 | 1 | `encoder.down.2.block.0.conv1` |

Per-call inventory at `1024x1024`, encoder:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `128,3,3,3` | 3x3 | 1,1 | 1,1 | `1,3,1024,1024` | `1,128,1024,1024` | 128 | 27 | 1048576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,128,1024,1024` | 128 | 1152 | 1048576 | 4 | 4 |
| `128,128,3,3` | 3x3 | 2,2 | 0,0 | `1,128,1025,1025` | `1,128,512,512` | 128 | 1152 | 262144 | 1 | 1 |
| `256,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,512,512` | `1,256,512,512` | 256 | 1152 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,512,512` | `1,256,512,512` | 256 | 2304 | 262144 | 3 | 3 |
| `256,128,1,1` | 1x1 | 1,1 | 0,0 | `1,128,512,512` | `1,256,512,512` | 256 | 128 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 2,2 | 0,0 | `1,256,513,513` | `1,256,256,256` | 256 | 2304 | 65536 | 1 | 1 |
| `512,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,256,256` | `1,512,256,256` | 512 | 2304 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,256,256` | `1,512,256,256` | 512 | 4608 | 65536 | 3 | 3 |
| `512,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,256,256` | `1,512,256,256` | 512 | 256 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 2,2 | 0,0 | `1,512,257,257` | `1,512,128,128` | 512 | 4608 | 16384 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,512,128,128` | 512 | 4608 | 16384 | 8 | 8 |
| `512,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,128,128` | `1,512,128,128` | 512 | 512 | 16384 | 4 | 4 |
| `32,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,32,128,128` | 32 | 4608 | 16384 | 1 | 1 |

Decoder:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `512,16,3,3` | 3x3 | 1,1 | 1,1 | `1,16,128,128` | `1,512,128,128` | 512 | 144 | 16384 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,512,128,128` | 512 | 4608 | 16384 | 10 | 10 |
| `512,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,128,128` | `1,512,128,128` | 512 | 512 | 16384 | 4 | 4 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,256,256` | `1,512,256,256` | 512 | 4608 | 65536 | 7 | 7 |
| `256,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,512,512` | `1,256,512,512` | 256 | 4608 | 262144 | 1 | 1 |
| `256,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,512,512` | `1,256,512,512` | 256 | 512 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,512,512` | `1,256,512,512` | 256 | 2304 | 262144 | 5 | 5 |
| `128,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,1024,1024` | `1,128,1024,1024` | 128 | 2304 | 1048576 | 1 | 1 |
| `128,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,1024,1024` | `1,128,1024,1024` | 128 | 256 | 1048576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,128,1024,1024` | 128 | 1152 | 1048576 | 5 | 5 |
| `3,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,3,1024,1024` | 3 | 1152 | 1048576 | 1 | 1 |

The Z-Image VAE has the same architecture as the Flux autoencoder: 16 latent channels, `ch_mult=[1,2,4,4]`, 128 base channels, mid attention, and no `quant_conv`/`post_quant_conv`.

### Ideogram 4 / Klein 9B / MOSS-Image (Flux 2 small VAE)

Facts: `AutoencoderKL` with `batch_norm_latent`. The internal latent is `32x128x128` at `/8`. `encode` then applies batch norm and a 2x2 space-to-depth rearrange (`... c (i pi) (j pj) -> ... (c pi pj) i j`, `pi=pj=2`), producing the packed `128x64x64` latent used by the diffusion model at `1024x1024` (`128x96x96` at `1536x1536`). `decode` reverses it. Encoder base width is 128, decoder base width is 96 (`decoder_ddconfig['ch'] = decoder.conv_in.shape[0] / ch_mult[-1] = 384/4`). 72 conv modules: 32 encoder calls, 40 decoder calls.

The batch-norm/space-to-depth and its inverse are not convolutions, so they do not appear in the conv rows. The last encoder conv (`encoder.conv_out`, `M=64, K=4608`) and the first decoder conv (`decoder.conv_in`, `M=384, K=288`) are the operator boundaries around them.

`(M, K)` families (22 families, 72 modules):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 384 | 3456 | 3x3 | 18 | `decoder.mid.block_1.conv1` |
| 512 | 4608 | 3x3 | 12 | `encoder.down.2.block.0.conv2` |
| 192 | 1728 | 3x3 | 6 | `decoder.up.1.block.0.conv2` |
| 96 | 864 | 3x3 | 5 | `decoder.up.0.block.0.conv2` |
| 128 | 1152 | 3x3 | 5 | `encoder.down.0.block.0.conv1` |
| 256 | 2304 | 3x3 | 4 | `encoder.down.1.block.0.conv2` |
| 384 | 384 | 1x1 | 4 | `decoder.mid.attn_1.k` |
| 512 | 512 | 1x1 | 4 | `encoder.mid.attn_1.k` |
| 3 | 864 | 3x3 | 1 | `decoder.conv_out` |
| 32 | 32 | 1x1 | 1 | `post_quant_conv` |
| 64 | 64 | 1x1 | 1 | `quant_conv` |
| 64 | 4608 | 3x3 | 1 | `encoder.conv_out` (2*z) |
| 96 | 192 | 1x1 | 1 | `decoder.up.0.block.0.nin_shortcut` |
| 96 | 1728 | 3x3 | 1 | `decoder.up.0.block.0.conv1` |
| 128 | 27 | 3x3 | 1 | `encoder.conv_in` |
| 192 | 384 | 1x1 | 1 | `decoder.up.1.block.0.nin_shortcut` |
| 192 | 3456 | 3x3 | 1 | `decoder.up.1.block.0.conv1` |
| 256 | 128 | 1x1 | 1 | `encoder.down.1.block.0.nin_shortcut` |
| 256 | 1152 | 3x3 | 1 | `encoder.down.1.block.0.conv1` |
| 384 | 288 | 3x3 | 1 | `decoder.conv_in` |
| 512 | 256 | 1x1 | 1 | `encoder.down.2.block.0.nin_shortcut` |
| 512 | 2304 | 3x3 | 1 | `encoder.down.2.block.0.conv1` |

Encoder at `1024x1024`:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `128,3,3,3` | 3x3 | 1,1 | 1,1 | `1,3,1024,1024` | `1,128,1024,1024` | 128 | 27 | 1048576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,1024,1024` | `1,128,1024,1024` | 128 | 1152 | 1048576 | 4 | 4 |
| `128,128,3,3` | 3x3 | 2,2 | 0,0 | `1,128,1025,1025` | `1,128,512,512` | 128 | 1152 | 262144 | 1 | 1 |
| `256,128,3,3` | 3x3 | 1,1 | 1,1 | `1,128,512,512` | `1,256,512,512` | 256 | 1152 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,512,512` | `1,256,512,512` | 256 | 2304 | 262144 | 3 | 3 |
| `256,128,1,1` | 1x1 | 1,1 | 0,0 | `1,128,512,512` | `1,256,512,512` | 256 | 128 | 262144 | 1 | 1 |
| `256,256,3,3` | 3x3 | 2,2 | 0,0 | `1,256,513,513` | `1,256,256,256` | 256 | 2304 | 65536 | 1 | 1 |
| `512,256,3,3` | 3x3 | 1,1 | 1,1 | `1,256,256,256` | `1,512,256,256` | 512 | 2304 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,256,256` | `1,512,256,256` | 512 | 4608 | 65536 | 3 | 3 |
| `512,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,256,256` | `1,512,256,256` | 512 | 256 | 65536 | 1 | 1 |
| `512,512,3,3` | 3x3 | 2,2 | 0,0 | `1,512,257,257` | `1,512,128,128` | 512 | 4608 | 16384 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,512,128,128` | 512 | 4608 | 16384 | 8 | 8 |
| `512,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,128,128` | `1,512,128,128` | 512 | 512 | 16384 | 4 | 4 |
| `64,512,3,3` | 3x3 | 1,1 | 1,1 | `1,512,128,128` | `1,64,128,128` | 64 | 4608 | 16384 | 1 | 1 |
| `64,64,1,1` | 1x1 | 1,1 | 0,0 | `1,64,128,128` | `1,64,128,128` | 64 | 64 | 16384 | 1 | 1 |

Decoder at `1024x1024`:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `32,32,1,1` | 1x1 | 1,1 | 0,0 | `1,32,128,128` | `1,32,128,128` | 32 | 32 | 16384 | 1 | 1 |
| `384,32,3,3` | 3x3 | 1,1 | 1,1 | `1,32,128,128` | `1,384,128,128` | 384 | 288 | 16384 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,128,128` | `1,384,128,128` | 384 | 3456 | 16384 | 10 | 10 |
| `384,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,384,128,128` | 384 | 384 | 16384 | 4 | 4 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,256,256` | `1,384,256,256` | 384 | 3456 | 65536 | 7 | 7 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,512,512` | `1,192,512,512` | 192 | 3456 | 262144 | 1 | 1 |
| `192,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,512,512` | `1,192,512,512` | 192 | 384 | 262144 | 1 | 1 |
| `192,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,512,512` | `1,192,512,512` | 192 | 1728 | 262144 | 5 | 5 |
| `96,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,1024,1024` | `1,96,1024,1024` | 96 | 1728 | 1048576 | 1 | 1 |
| `96,192,1,1` | 1x1 | 1,1 | 0,0 | `1,192,1024,1024` | `1,96,1024,1024` | 96 | 192 | 1048576 | 1 | 1 |
| `96,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,96,1024,1024` | 96 | 864 | 1048576 | 5 | 5 |
| `3,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,3,1024,1024` | 3 | 864 | 1048576 | 1 | 1 |

The original `flux2-vae.safetensors` (Comfy-Org `flux2-dev`) has the same module map with 512 channels in the mid and upper decoder blocks instead of 384. It is a drop-in replacement target with identical latent geometry, so only the `flux2_vae_small` rows above are recorded. A wider variant would add `(M=512, K=4608)` 3x3 and `(M=512, K=512)` 1x1 families at the same `N` values.

### Qwen-Image 2.1 (Wan 2.2 VAE, 2D)

Facts: we always run the converted 2D checkpoint `~/ComfyUI/models/vae/qwen_image_2.1_vae_2d.safetensors` (226 tensors), which is the 2D form of the released Wan 2.2-layout VAE. The network has `dim_mult=[1,2,4,8,8]`, encoder base width 96, decoder base width 144, `z_dim=64`, `patch_size=1`, spatial `/16`, and 4 input/output channels (RGBA). Latent is `64x64x64` at `1024x1024` and `64x96x96` at `1536x1536` (4D latent, `latent_dim=2`). 82 conv calls: 36 in the encode pass, 46 in the decode pass, covering 70 conv modules that were 3D in the released file (62 3x3x3 and 8 1x1x1) plus 12 convs that were already 2D (8 per-frame resamplers and 4 mid-attention projections).

Loading: the `ComfyUI-Anzhc-Qwen2D` custom node detects the 2D file by its `encoder.downsamples.*` / `decoder.conv1` / `conv2` keys with 4D conv weights, builds `qwen21_2d_arch.QwenImage21VAE2D`, and binds the shared 2D encode/decode paths (`runtime.py` `_is_qwen21_2d_state_dict` / `_init_qwen21_vae`). It sets `latent_dim=2`, `latent_channels=64`, `output_channels=4`, `pad_channel_value=1.0` (opaque alpha for RGB input) and `upscale_ratio=16`. The original 3D file still loads through ComfyUI's stock `WanVAE` path for anyone who needs multi-frame behaviour.

Provenance: the 2D file is produced from `qwen_image_2.1_vae_bf16.safetensors` by `custom_nodes/ComfyUI-Anzhc-Qwen2D/convert_qwen21_vae_2d.py`, which drops the 12 `time_conv` tensors (6 modules, 3 in the encoder temporal downsamplers and 3 in the decoder upsamplers), squeezes the remaining 70 conv weights `[C_out, C_in, 1, kh, kw] -> [C_out, C_in, kh, kw]`, and squeezes the 60 RMS-norm gammas `[C, 1, 1, 1] -> [C, 1, 1]`. The conversion is exact because every `Conv3d` in the released file has temporal kernel extent 1 (`{1: 76}` by `weight.shape[2]`) and the `T=1` path never executes the temporal resamplers.

Verification (fp32, real weights): the 2D module reproduces the 3D model executed on a single frame. Stage-by-stage, the encoder `avg_shortcut` and decoder `dup_up` shortcuts match their 3D counterparts to the same level as the surrounding residual block, so there is no shortcut-specific divergence. The encode latent differs by `3.3e-06` absolute at 64x64 (`3.5e-07` relative) and `1.0e-04` at 512x512 (`7.4e-06` relative). Decoding the same latent differs by `7.8e-06` absolute at 64x64 and `1.9e-05` at 512x512 (`1.4e-05` relative). A full encode->decode round trip differs by `1.2e-05` relative at 64x64 and `6.8e-04` at 512x512, which is the encode difference amplified by the decoder. RGB inputs are padded to 4 channels with `alpha=1.0` and decode returns 4 channels in both forms.

`(M, K)` families (2D form, 30 families, 82 modules):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 1152 | 10368 | 3x3 | 18 | `decoder.middle.0.residual.2` |
| 768 | 6912 | 3x3 | 12 | `encoder.downsamples.3.downsamples.0.residual.6` |
| 288 | 2592 | 3x3 | 6 | `decoder.upsamples.3.upsamples.0.residual.6` |
| 576 | 5184 | 3x3 | 6 | `decoder.upsamples.2.upsamples.0.residual.6` |
| 96 | 864 | 3x3 | 5 | `encoder.downsamples.0.downsamples.0.residual.2` |
| 144 | 1296 | 3x3 | 5 | `decoder.upsamples.4.upsamples.0.residual.6` |
| 192 | 1728 | 3x3 | 4 | `encoder.downsamples.1.downsamples.0.residual.6` |
| 384 | 3456 | 3x3 | 4 | `encoder.downsamples.2.downsamples.0.residual.6` |
| 4 | 1296 | 3x3 | 1 | `decoder.head.2` (RGBA out) |
| 64 | 64 | 1x1 | 1 | `conv2` latent projection |
| 96 | 36 | 3x3 | 1 | `encoder.conv1` (4 input channels) |
| 128 | 128 | 1x1 | 1 | `conv1` latent projection |
| 128 | 6912 | 3x3 | 1 | `encoder.head.2` (2*z) |
| 144 | 288 | 1x1 | 1 | `decoder.upsamples.4.upsamples.0.shortcut` |
| 144 | 2592 | 3x3 | 1 | `decoder.upsamples.4.upsamples.0.residual.2` |
| 192 | 96 | 1x1 | 1 | `encoder.downsamples.1.downsamples.0.shortcut` |
| 192 | 864 | 3x3 | 1 | `encoder.downsamples.1.downsamples.0.residual.2` |
| 288 | 576 | 1x1 | 1 | `decoder.upsamples.3.upsamples.0.shortcut` |
| 288 | 5184 | 3x3 | 1 | `decoder.upsamples.3.upsamples.0.residual.2` |
| 384 | 192 | 1x1 | 1 | `encoder.downsamples.2.downsamples.0.shortcut` |
| 384 | 1728 | 3x3 | 1 | `encoder.downsamples.2.downsamples.0.residual.2` |
| 576 | 1152 | 1x1 | 1 | `decoder.upsamples.2.upsamples.0.shortcut` |
| 576 | 10368 | 3x3 | 1 | `decoder.upsamples.2.upsamples.0.residual.2` |
| 768 | 384 | 1x1 | 1 | `encoder.downsamples.3.downsamples.0.shortcut` |
| 768 | 768 | 1x1 | 1 | `encoder.middle.1.proj` |
| 768 | 3456 | 3x3 | 1 | `encoder.downsamples.3.downsamples.0.residual.2` |
| 1152 | 576 | 3x3 | 1 | `decoder.conv1` |
| 1152 | 1152 | 1x1 | 1 | `decoder.middle.1.proj` |
| 2304 | 768 | 1x1 | 1 | `encoder.middle.1.to_qkv` |
| 3456 | 1152 | 1x1 | 1 | `decoder.middle.1.to_qkv` |

Encoder at `1024x1024` (the checkpoint stores 4D weights. The `in`/`out` shapes below are 4D too):

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `96,4,3,3` | 3x3 | 1,1 | 1,1 | `1,4,1024,1024` | `1,96,1024,1024` | 96 | 36 | 1048576 | 1 | 1 |
| `96,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,1024,1024` | `1,96,1024,1024` | 96 | 864 | 1048576 | 4 | 4 |
| `96,96,3,3` | 3x3 | 2,2 | 0,0 | `1,96,1025,1025` | `1,96,512,512` | 96 | 864 | 262144 | 1 | 1 |
| `192,96,3,3` | 3x3 | 1,1 | 1,1 | `1,96,512,512` | `1,192,512,512` | 192 | 864 | 262144 | 1 | 1 |
| `192,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,512,512` | `1,192,512,512` | 192 | 1728 | 262144 | 3 | 3 |
| `192,96,1,1` | 1x1 | 1,1 | 0,0 | `1,96,512,512` | `1,192,512,512` | 192 | 96 | 262144 | 1 | 1 |
| `192,192,3,3` | 3x3 | 2,2 | 0,0 | `1,192,513,513` | `1,192,256,256` | 192 | 1728 | 65536 | 1 | 1 |
| `384,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,256,256` | `1,384,256,256` | 384 | 1728 | 65536 | 1 | 1 |
| `384,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,256,256` | `1,384,256,256` | 384 | 3456 | 65536 | 3 | 3 |
| `384,192,1,1` | 1x1 | 1,1 | 0,0 | `1,192,256,256` | `1,384,256,256` | 384 | 192 | 65536 | 1 | 1 |
| `384,384,3,3` | 3x3 | 2,2 | 0,0 | `1,384,257,257` | `1,384,128,128` | 384 | 3456 | 16384 | 1 | 1 |
| `768,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,128,128` | `1,768,128,128` | 768 | 3456 | 16384 | 1 | 1 |
| `768,768,3,3` | 3x3 | 1,1 | 1,1 | `1,768,128,128` | `1,768,128,128` | 768 | 6912 | 16384 | 3 | 3 |
| `768,384,1,1` | 1x1 | 1,1 | 0,0 | `1,384,128,128` | `1,768,128,128` | 768 | 384 | 16384 | 1 | 1 |
| `768,768,3,3` | 3x3 | 2,2 | 0,0 | `1,768,129,129` | `1,768,64,64` | 768 | 6912 | 4096 | 1 | 1 |
| `768,768,3,3` | 3x3 | 1,1 | 1,1 | `1,768,64,64` | `1,768,64,64` | 768 | 6912 | 4096 | 8 | 8 |
| `2304,768,1,1` | 1x1 | 1,1 | 0,0 | `1,768,64,64` | `1,2304,64,64` | 2304 | 768 | 4096 | 1 | 1 |
| `768,768,1,1` | 1x1 | 1,1 | 0,0 | `1,768,64,64` | `1,768,64,64` | 768 | 768 | 4096 | 1 | 1 |
| `128,768,3,3` | 3x3 | 1,1 | 1,1 | `1,768,64,64` | `1,128,64,64` | 128 | 6912 | 4096 | 1 | 1 |
| `128,128,1,1` | 1x1 | 1,1 | 0,0 | `1,128,64,64` | `1,128,64,64` | 128 | 128 | 4096 | 1 | 1 |

Decoder at `1024x1024`:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `64,64,1,1` | 1x1 | 1,1 | 0,0 | `1,64,64,64` | `1,64,64,64` | 64 | 64 | 4096 | 1 | 1 |
| `1152,64,3,3` | 3x3 | 1,1 | 1,1 | `1,64,64,64` | `1,1152,64,64` | 1152 | 576 | 4096 | 1 | 1 |
| `1152,1152,3,3` | 3x3 | 1,1 | 1,1 | `1,1152,64,64` | `1,1152,64,64` | 1152 | 10368 | 4096 | 10 | 10 |
| `3456,1152,1,1` | 1x1 | 1,1 | 0,0 | `1,1152,64,64` | `1,3456,64,64` | 3456 | 1152 | 4096 | 1 | 1 |
| `1152,1152,1,1` | 1x1 | 1,1 | 0,0 | `1,1152,64,64` | `1,1152,64,64` | 1152 | 1152 | 4096 | 1 | 1 |
| `1152,1152,3,3` | 3x3 | 1,1 | 1,1 | `1,1152,128,128` | `1,1152,128,128` | 1152 | 10368 | 16384 | 7 | 7 |
| `576,1152,3,3` | 3x3 | 1,1 | 1,1 | `1,1152,256,256` | `1,576,256,256` | 576 | 10368 | 65536 | 1 | 1 |
| `576,1152,1,1` | 1x1 | 1,1 | 0,0 | `1,1152,256,256` | `1,576,256,256` | 576 | 1152 | 65536 | 1 | 1 |
| `576,576,3,3` | 3x3 | 1,1 | 1,1 | `1,576,256,256` | `1,576,256,256` | 576 | 5184 | 65536 | 5 | 5 |
| `576,576,3,3` | 3x3 | 1,1 | 1,1 | `1,576,512,512` | `1,576,512,512` | 576 | 5184 | 262144 | 1 | 1 |
| `288,576,3,3` | 3x3 | 1,1 | 1,1 | `1,576,512,512` | `1,288,512,512` | 288 | 5184 | 262144 | 1 | 1 |
| `288,576,1,1` | 1x1 | 1,1 | 0,0 | `1,576,512,512` | `1,288,512,512` | 288 | 576 | 262144 | 1 | 1 |
| `288,288,3,3` | 3x3 | 1,1 | 1,1 | `1,288,512,512` | `1,288,512,512` | 288 | 2592 | 262144 | 5 | 5 |
| `288,288,3,3` | 3x3 | 1,1 | 1,1 | `1,288,1024,1024` | `1,288,1024,1024` | 288 | 2592 | 1048576 | 1 | 1 |
| `144,288,3,3` | 3x3 | 1,1 | 1,1 | `1,288,1024,1024` | `1,144,1024,1024` | 144 | 2592 | 1048576 | 1 | 1 |
| `144,288,1,1` | 1x1 | 1,1 | 0,0 | `1,288,1024,1024` | `1,144,1024,1024` | 144 | 288 | 1048576 | 1 | 1 |
| `144,144,3,3` | 3x3 | 1,1 | 1,1 | `1,144,1024,1024` | `1,144,1024,1024` | 144 | 1296 | 1048576 | 5 | 5 |
| `4,144,3,3` | 3x3 | 1,1 | 1,1 | `1,144,1024,1024` | `1,4,1024,1024` | 4 | 1296 | 1048576 | 1 | 1 |

The decoder dominates the Qwen-Image 2.1 VAE cost (`8477` of `10031` GMACs at `1024x1024`): it runs the widest (1152-channel) 3x3 convs at `64x64` and `128x128`, then 576/288/144-channel 3x3 convs at `256/512/1024`, so the largest `N` levels carry the smaller channel counts.

The 2D checkpoint keeps the Wan 2.2 residual shortcut structure. Two details matter numerically and are worth knowing when reading the rows: the temporal-downsample stages still build an average-pool shortcut (`AvgDown2D` with `factor_t=2`), which on a single frame inserts one zero "frame" and therefore leaves half of the shortcut channels at zero, and the decoder up-stages use a `DupUp2D` channel gather (`factor_t` collapsed to the last slice) instead of a temporal shuffle. Both are pure 2D tensor operations in the converted file.

### LTX 2.3 audio VAE (mel 2D)

Facts: `CausalAudioAutoencoder` from `comfy/ldm/lightricks/vae/causal_audio_autoencoder.py` with `ddconfig = {in_channels: 2, out_ch: 2, z_channels: 8, ch: 128, ch_mult: [1,2,4], num_res_blocks: 2, mel_bins: 64, causality_axis: height}`. The operator is a 2D conv over `(time, mel)`, with causal padding on the time axis applied before each conv (`F.pad`), so recorded `padding` is `0,0` and the module input already includes the padded extent. Latent is `8x(T/4)x16` for a mel input of `T` frames with 64 bins. 50 conv modules: 22 encoder calls, 28 decoder calls. The 1D vocoder that turns the decoded mel back into a waveform is documented in `input_shapes_conv1d_vae.md`.

Geometry: the audio config uses 16 kHz sampling, STFT hop 160, 64 mel bins, and a latent downsample factor of 4 in both axes. The 5.12 s training clip is 512 mel frames. The documented video clips map to 165 and 329 mel frames using `round(frames / 24 * 25)` audio latents and `4 * latents - 3` causal mel frames (`40 -> 42 latents -> 165` and `80 -> 83 -> 329`). Only the time/mel extents change between these three workloads. `M`, `K` and the channel ladder are fixed.

`(M, K)` families (15 families, 50 modules):

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 512 | 4608 | 3x3 | 18 | `decoder.mid.block_1.conv1.conv` |
| 128 | 1152 | 3x3 | 10 | `decoder.up.0.block.0.conv2.conv` |
| 256 | 2304 | 3x3 | 10 | `decoder.up.1.block.0.conv2.conv` |
| 2 | 1152 | 3x3 | 1 | `decoder.conv_out.conv` |
| 16 | 4608 | 3x3 | 1 | `encoder.conv_out.conv` (2*z) |
| 128 | 18 | 3x3 | 1 | `encoder.conv_in.conv` (2 input channels) |
| 128 | 256 | 1x1 | 1 | `decoder.up.0.block.0.nin_shortcut.conv` |
| 128 | 2304 | 3x3 | 1 | `decoder.up.0.block.0.conv1.conv` |
| 256 | 128 | 1x1 | 1 | `encoder.down.1.block.0.nin_shortcut.conv` |
| 256 | 512 | 1x1 | 1 | `decoder.up.1.block.0.nin_shortcut.conv` |
| 256 | 1152 | 3x3 | 1 | `encoder.down.1.block.0.conv1.conv` |
| 256 | 4608 | 3x3 | 1 | `decoder.up.1.block.0.conv1.conv` |
| 512 | 72 | 3x3 | 1 | `decoder.conv_in.conv` (z=8) |
| 512 | 256 | 1x1 | 1 | `encoder.down.2.block.0.nin_shortcut.conv` |
| 512 | 2304 | 3x3 | 1 | `encoder.down.2.block.0.conv1.conv` |

Encoder at the 5.12 s workload (512 mel frames):

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `128,2,3,3` | 3x3 | 1,1 | 0,0 | `1,2,514,66` | `1,128,512,64` | 128 | 18 | 32768 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 0,0 | `1,128,514,66` | `1,128,512,64` | 128 | 1152 | 32768 | 4 | 4 |
| `128,128,3,3` | 3x3 | 2,2 | 0,0 | `1,128,514,65` | `1,128,256,32` | 128 | 1152 | 8192 | 1 | 1 |
| `256,128,3,3` | 3x3 | 1,1 | 0,0 | `1,128,258,34` | `1,256,256,32` | 256 | 1152 | 8192 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 0,0 | `1,256,258,34` | `1,256,256,32` | 256 | 2304 | 8192 | 3 | 3 |
| `256,128,1,1` | 1x1 | 1,1 | 0,0 | `1,128,256,32` | `1,256,256,32` | 256 | 128 | 8192 | 1 | 1 |
| `256,256,3,3` | 3x3 | 2,2 | 0,0 | `1,256,258,33` | `1,256,128,16` | 256 | 2304 | 2048 | 1 | 1 |
| `512,256,3,3` | 3x3 | 1,1 | 0,0 | `1,256,130,18` | `1,512,128,16` | 512 | 2304 | 2048 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 0,0 | `1,512,130,18` | `1,512,128,16` | 512 | 4608 | 2048 | 7 | 7 |
| `512,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,128,16` | `1,512,128,16` | 512 | 256 | 2048 | 1 | 1 |
| `16,512,3,3` | 3x3 | 1,1 | 0,0 | `1,512,130,18` | `1,16,128,16` | 16 | 4608 | 2048 | 1 | 1 |

Decoder at the 5.12 s workload:

| weight | k_eff | stride | pad | in | out | M | K | N | calls | modules |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `512,8,3,3` | 3x3 | 1,1 | 0,0 | `1,8,130,18` | `1,512,128,16` | 512 | 72 | 2048 | 1 | 1 |
| `512,512,3,3` | 3x3 | 1,1 | 0,0 | `1,512,130,18` | `1,512,128,16` | 512 | 4608 | 2048 | 10 | 10 |
| `512,512,3,3` | 3x3 | 1,1 | 0,0 | `1,512,258,34` | `1,512,256,32` | 512 | 4608 | 8192 | 1 | 1 |
| `256,512,3,3` | 3x3 | 1,1 | 0,0 | `1,512,257,34` | `1,256,255,32` | 256 | 4608 | 8160 | 1 | 1 |
| `256,512,1,1` | 1x1 | 1,1 | 0,0 | `1,512,255,32` | `1,256,255,32` | 256 | 512 | 8160 | 1 | 1 |
| `256,256,3,3` | 3x3 | 1,1 | 0,0 | `1,256,257,34` | `1,256,255,32` | 256 | 2304 | 8160 | 5 | 5 |
| `256,256,3,3` | 3x3 | 1,1 | 0,0 | `1,256,512,66` | `1,256,510,64` | 256 | 2304 | 32640 | 1 | 1 |
| `128,256,3,3` | 3x3 | 1,1 | 0,0 | `1,256,511,66` | `1,128,509,64` | 128 | 2304 | 32576 | 1 | 1 |
| `128,256,1,1` | 1x1 | 1,1 | 0,0 | `1,256,509,64` | `1,128,509,64` | 128 | 256 | 32576 | 1 | 1 |
| `128,128,3,3` | 3x3 | 1,1 | 0,0 | `1,128,511,66` | `1,128,509,64` | 128 | 1152 | 32576 | 5 | 5 |
| `2,128,3,3` | 3x3 | 1,1 | 0,0 | `1,128,511,66` | `1,2,509,64` | 2 | 1152 | 32576 | 1 | 1 |

The three workloads have identical `M`/`K` families. Only `N` changes: the 5.12 s clip uses `2048/8192/32768`, the `640x480x40` span uses `672/2656/10624`, and the `1280x720x80` span uses `1328/5280/21120`.

## Special Cases

### VAE attention blocks

The mid attention block in SDXL, the Flux VAE, the Qwen 2D VAE, the Flux 2 small VAE, and the Wan 2.2-derived Qwen-Image 2.1 VAE contributes four 1x1 conv calls: `to_qkv` (or `q`/`k`/`v`), and `proj_out`/`proj`. Their `N` is the full latent spatial extent (`16384` at `1024x1024` for the `/8` VAEs), so they are the largest-`N` 1x1 convs in the encode pass. The attention core itself (`optimized_attention`/`slice_attention`) is out of scope. ComfyUI's VAE attention backend is selected by `model_management.pytorch_attention_enabled_vae()` and by free-memory heuristics. That choice changes the attention matmuls but not the four projection convs.

### Latent projections are 1x1 convs

`quant_conv`/`post_quant_conv` (SDXL, Z-Image, Flux 2 small, Qwen 2D, Qwen-Image 2.1) and `conv1`/`conv2` (Qwen-Image 2.1) are 1x1 convolutions, i.e. plain GEMMs with `K` equal to the channel count (`4, 8, 16, 32, 48, 64, 128`). Their MAC counts are negligible (`4x4`, `8x8`, `16x16`, `32x32`, `64x64`, `128x128` per position) but they are listed because they are separate kernel launches on the same `N` axis as the neighbouring 3x3 convs.

### Causal padding outside the module

For the SDXL/Z-Image/Flux 2 downsamplers and the LTX audio autoencoder, padding is applied outside the conv module (`ZeroPad2d`, `F.pad`), so the CSV records `padding=0,0` and an input spatial extent that is one or two larger than the output. The implicit-GEMM decomposition uses `M`, `K` and the output extent and is unaffected.

### Tiled execution

ComfyUI's `VAEDecodeTiled`/`VAEEncodeTiled` reduce per-call `N` (SDXL default `tile_x=512, tile_y=512, overlap=64`. Flux 2 `strip_apply` splits rows when an activation exceeds 16 MiB). The recorded rows are single-pass. Per-call `N` under tiling is `tile_area`-derived, and `M`/`K` are unchanged.

## Reproducing The Data

The rows were produced with the scripts in `~/tmp/feather_ops/conv_vae_shapes/`:
- `fetch_headers.py`: downloads the safetensors headers that are not available locally (Z-Image `ae.safetensors`, the original Flux 2 `flux2-vae.safetensors` used only for the small-VAE comparison, Qwen-Image 2.1 VAE, and the LTX 2.3 `vae.*`/`vocoder.*`/`audio_vae.*` tensors) with HTTP Range requests only. No tensor payloads are downloaded except the Qwen-Image 2.1 VAE, which is kept at `~/ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors` together with its converted 2D form.
- `collect_conv_shapes.py`: builds each VAE from its real safetensors header on the `torch.device("meta")` device, runs one `encode` and one `decode` per workload, and records every `Conv2d` call (module path, weight shape, stride, padding, dilation, groups, input/output shape) through forward hooks.
- `build_datasets.py`: derives the implicit-GEMM columns, aggregates identical calls, and writes the three CSV/JSON datasets.
- `doc_tables.py` and `analyze.py`: emit the tables and totals used in these notes.
- `check_qwen21_2d_conversion.py`: the numerical 3D-vs-2D equivalence check for the Qwen-Image 2.1 VAE.

The converted Qwen-Image 2.1 checkpoint is produced by `~/ComfyUI/custom_nodes/ComfyUI-Anzhc-Qwen2D/convert_qwen21_vae_2d.py` and loaded by that node's `runtime.py` patch (`_is_qwen21_2d_state_dict` / `_init_qwen21_vae`). Re-running `collect_conv_shapes.py` against the 2D module reproduces exactly the same 164 `(phase, module, weight, in, out)` signatures as the rows in `conv2d_vae_shapes.csv`.

The meta-device simulation needs a few `comfy.model_management` hooks (free/total memory, attention backend selection, `intermediate_device`) because the recorded shapes must not depend on the host GPU. Those overrides are in `vae_meta.py`. Row strips (`strip_apply`) and the LTX decode chunking are disabled for the 2D family so that each module reports its canonical full-activation shape once.

## Notes And Caveats

- Inference only. No VAE in this set is trained by the documented workloads, so no `NT`/`NN` parameter-gradient conv shapes are recorded. Training a VAE would add dgrad and wgrad implicit-GEMM shapes for every conv, with `K` replaced by the activation count.
- ComfyUI's `VAE.encode` crops pixel input to a multiple of the VAE stride (`vae_encode_crop_pixels`). Non-multiple-of-8 or 16 requests are cropped before the first conv. The `1536x1536` and `1024x1024` rows are already multiples.
- The Flux 2 small VAE batch-norm uses `bn.running_mean/var` over the 128 packed channels. It is an elementwise affine step, not a GEMM.
- LTX audio rows are included here because they are Conv2d operators. The same LTX 2.3 audio stack's 1D vocoder stage is in the 1D note.
- Qwen-Image 2.1 runs the converted 2D checkpoint (`qwen_image_2.1_vae_2d.safetensors`). The released 3D file is only its source: it has the same `M`/`K` values with a `T=1` axis and six extra `time_conv` modules that never execute on the image path.
- The `m,n,k` triple is per call. A per-sample pass with `batch>1` multiplies `batch`, not `n`, for the non-transposed convs.
