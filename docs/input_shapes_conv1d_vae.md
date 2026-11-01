# gfx1151 FP16 1D Conv VAE Input Shapes

This note documents the `Conv1d` and `ConvTranspose1d` tensor shapes and their implicit-GEMM decompositions for the 1D convolution stacks.

Machine-readable rows are saved in `~/tmp/feather_ops/conv_vae_shapes/shape_data/`:
- `conv1d_vae_shapes.csv`: aggregated per-call rows with `m, n, k, batch`, `implicit_gemm_form` (`fwd` or `dgrad`), and identical-call `count`.
- `conv1d_vae_unique_gemm_shapes.csv`: unique `(op_kind, m, n, k)` rows with call counts and models.
- `conv1d_vae_shapes_manifest.json`: per-model counts, workload geometry, sources, and column semantics.

## Scope

- Models: H3 audio (MiniMax H3 audio VAE) and LTX 2.3 audio (the vocoder stage of the LTX 2.3 AV pipeline).
- Inference only. One `encode` (H3) or one `forward` (LTX vocoder) plus one `decode` per documented clip, batch size 1.
- The LTX 2.3 autoencoder half of the audio pipeline is a 2D conv net over mel frames and is documented in `input_shapes_conv2d_vae.md`. This note covers only its 1D vocoder output stage. It is marked `is_vae: false` in the manifest because the vocoder is a neural audio synthesis stage, not a latent autoencoder. It is included because it is part of the LTX 2.3 AV decode path and consists entirely of 1D convolutions.
- Every recorded operator is a `torch.nn.Conv1d` or `torch.nn.ConvTranspose1d` with `groups=1` and dilation 1, 3, 5, or 9 on the residual convs.
- Excluded: Snake/SnakeBeta activations, mel STFT bases, resamplers (`UpSample1d`), attention (`AttnProjection` in the H3 encoder `pre_block`, which is `Linear`-based), and the `latent_dim`-wide 1x1 projections' surrounding elementwise work.

## Implicit GEMM Mapping

For a 1D convolution with weight `[C_out, C_in, k]` applied to `[B, C_in, L_in]` and producing `[B, C_out, L_out]`:

```text
M = C_out
N = L_out              # per batch element
K = C_in * k
batch = B
MACs = M * N * K * batch
```

A `1x1` conv is a plain GEMM with `K = C_in` (used by the residual `1x1` branches in the H3 DAC encoder and by the `mean_proj`/`dec_in_proj` latent projections).

`ConvTranspose1d` forward is executed by the backend as a convolution *backward-data* (dgrad) call. With PyTorch weight `[C_in_t, C_out_t, k]`, input `[B, C_in_t, L_in]`, and output `[B, C_out_t, L_out]`:

```text
M = C_out_t                 # = weight.shape[1]
N = L_out                   # output positions (dX positions in dgrad terms)
K = C_in_t * k              # = weight.shape[0] * k
batch = B
implicit_gemm_form = dgrad
```

This is the same arithmetic as a forward conv with `C_out = C_out_t` and `C_in = C_in_t`. The difference is the memory access pattern (scatter/col2im instead of gather/im2col). With `stride > 1` the backend splits the `N` axis into `stride` phase groups of `N/stride` positions, so a single logical dgrad call appears as several GEMMs with a smaller `N`.

## Model To VAE Mapping

| Model | Stack | Weights | Latent | 1D conv modules | Calls per pass |
| --- | --- | --- | --- | ---: | ---: |
| H3 audio | `MiniMaxH3AudioVAE` (DAC encoder + BigVGAN decoder) | `minimax_h3_audio_vae_fp32.safetensors` | `[B,32,2,T]`, 800 samples per latent, 40 latents/s | 174 | 38 encode + 136 decode |
| LTX 2.3 audio | `VocoderWithBWE` (HiFiGAN/BigVGAN + BWE) | `vocoder.*` in `ltx-2.3-22b-dev.safetensors` | not a latent (mel -> waveform) | 213 | 213 forward |

## Aggregate Counts

The 1D dataset contains 1,161 aggregated rows: 1,107 `conv1d` and 54 `conv_transpose1d`. There are 291 unique `(op_kind, m, n, k)` shapes.

| Model | Workload | Samples / mel frames | Calls | GMACs |
| --- | --- | --- | ---: | ---: |
| H3 audio | 1 s | 32000 samples / 40 latents | 38 + 136 | 64.2 + 52.2 |
| H3 audio | `640x480x40` span | 74400 samples / 93 latents | 38 + 136 | 149.2 + 121.3 |
| H3 audio | `1280x720x80` span | 120000 samples / 150 latents | 38 + 136 | 240.6 + 195.6 |
| LTX 2.3 vocoder | 5.12 s clip | 512 mel frames | 116 + 97 | 420.2 + 194.4 |
| LTX 2.3 vocoder | `640x480x40` span | 165 mel frames | 116 + 97 | 135.4 + 62.7 |
| LTX 2.3 vocoder | `1280x720x80` span | 329 mel frames | 116 + 97 | 270.0 + 124.9 |

The call counts are constant across workloads because both stacks are fully convolutional: only `N` (the sample or mel-frame extent) changes. `M` and `K` are fixed by the checkpoints.

## H3 Audio VAE

Facts (`comfy/ldm/minimax/audio_vae.py`): `MiniMaxH3AudioVAE(encoder_dim=64, encoder_rates=(2,4,4,5,5), latent_dim=2048, decoder_dim=1024, vae_latent_channels=32)`. Sample rate 32 kHz, hop length `2*4*4*5*5 = 800` samples, 40 latent frames/s, stereo processed as a folded batch axis (`b*s`). Latents are normalized to `[B,32,2,T]`.

Encoder (DAC): `encoder.block.0` `Conv1d(1, 64, 7)` followed by five `EncoderBlock`s, each with three `ResidualUnit`s (`Conv1d(k=7, dilation=d)`, `d in {1,3,9}`, plus a `1x1` branch) and a strided `Conv1d(k=2r, stride=r)` with `r in {2,4,4,5,5}`. Channels `64 -> 128 -> 256 -> 512 -> 1024 -> 2048`, length `L -> L/800`. A final `Conv1d(2048, 2048, 3)` (`encoder.block.7`) and the `1x1` `mean_proj(32,32)` produce the posterior mean. The `pre_block` attention projection is `Linear`-based and contributes no convs.

Decoder (BigVGAN): `dec_in_proj` `Conv1d(32, 2048, 1)`, `conv_pre` `Conv1d(2048, 1024, 7)`, seven up stages `ups.{0..6}` with `ConvTranspose1d` rates `(5,5,2,2,2,2,2)` and channels `512, 256, 128, 64, 32, 16, 8`, each followed by 3 `AMPBlock1` residual blocks with kernel sizes `(3,7,11)` and dilations `(1,3,5)` (`convs1` + `convs2` = 6 convs per kernel, 18 per stage), and `conv_post` `Conv1d(8, 1, 7)`.

### Encoder at 1 s (32000 samples, 40 latent frames)

| weight | k | stride | pad | in | out | M | K | N | calls |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `64,1,7` | 7 | 1 | 3 | `2,1,32000` | `2,64,32000` | 64 | 7 | 32000 | 1 |
| `64,64,7` | 7 | 1 | 3 | `2,64,32000` | `2,64,32000` | 64 | 448 | 32000 | 1 |
| `64,64,1` | 1 | 1 | 0 | `2,64,32000` | `2,64,32000` | 64 | 64 | 32000 | 3 |
| `64,64,7` | 7 | 1 | 9 | `2,64,32000` | `2,64,32000` | 64 | 448 | 32000 | 1 |
| `64,64,7` | 7 | 1 | 27 | `2,64,32000` | `2,64,32000` | 64 | 448 | 32000 | 1 |
| `128,64,4` | 4 | 2 | 1 | `2,64,32000` | `2,128,16000` | 128 | 256 | 16000 | 1 |
| `128,128,7` | 7 | 1 | 3 | `2,128,16000` | `2,128,16000` | 128 | 896 | 16000 | 1 |
| `128,128,1` | 1 | 1 | 0 | `2,128,16000` | `2,128,16000` | 128 | 128 | 16000 | 3 |
| `128,128,7` | 7 | 1 | 9 | `2,128,16000` | `2,128,16000` | 128 | 896 | 16000 | 1 |
| `128,128,7` | 7 | 1 | 27 | `2,128,16000` | `2,128,16000` | 128 | 896 | 16000 | 1 |
| `256,128,8` | 8 | 4 | 2 | `2,128,16000` | `2,256,4000` | 256 | 1024 | 4000 | 1 |
| `256,256,7` | 7 | 1 | 3 | `2,256,4000` | `2,256,4000` | 256 | 1792 | 4000 | 1 |
| `256,256,1` | 1 | 1 | 0 | `2,256,4000` | `2,256,4000` | 256 | 256 | 4000 | 3 |
| `256,256,7` | 7 | 1 | 9 | `2,256,4000` | `2,256,4000` | 256 | 1792 | 4000 | 1 |
| `256,256,7` | 7 | 1 | 27 | `2,256,4000` | `2,256,4000` | 256 | 1792 | 4000 | 1 |
| `512,256,8` | 8 | 4 | 2 | `2,256,4000` | `2,512,1000` | 512 | 2048 | 1000 | 1 |
| `512,512,7` | 7 | 1 | 3 | `2,512,1000` | `2,512,1000` | 512 | 3584 | 1000 | 1 |
| `512,512,1` | 1 | 1 | 0 | `2,512,1000` | `2,512,1000` | 512 | 512 | 1000 | 3 |
| `512,512,7` | 7 | 1 | 9 | `2,512,1000` | `2,512,1000` | 512 | 3584 | 1000 | 1 |
| `512,512,7` | 7 | 1 | 27 | `2,512,1000` | `2,512,1000` | 512 | 3584 | 1000 | 1 |
| `1024,512,10` | 10 | 5 | 3 | `2,512,1000` | `2,1024,200` | 1024 | 5120 | 200 | 1 |
| `1024,1024,7` | 7 | 1 | 3 | `2,1024,200` | `2,1024,200` | 1024 | 7168 | 200 | 1 |
| `1024,1024,1` | 1 | 1 | 0 | `2,1024,200` | `2,1024,200` | 1024 | 1024 | 200 | 3 |
| `1024,1024,7` | 7 | 1 | 9 | `2,1024,200` | `2,1024,200` | 1024 | 7168 | 200 | 1 |
| `1024,1024,7` | 7 | 1 | 27 | `2,1024,200` | `2,1024,200` | 1024 | 7168 | 200 | 1 |
| `2048,1024,10` | 10 | 5 | 3 | `2,1024,200` | `2,2048,40` | 2048 | 10240 | 40 | 1 |
| `2048,2048,3` | 3 | 1 | 1 | `2,2048,40` | `2,2048,40` | 2048 | 6144 | 40 | 1 |
| `32,32,1` | 1 | 1 | 0 | `2,32,40` | `2,32,40` | 32 | 32 | 40 | 1 |

The `pad=9/27` rows are the dilated residual convs (`dilation=3/9` with kernel 7). Their `K` is unaffected by dilation (still `7*C`) but the padding grows to `d*(k-1)/2`. The `batch=2` axis is the folded stereo pair. `b*s = 2` for a stereo input.

### Decoder stage template at 1 s

| stage | upsample conv | resblock ch | K (k=3) | K (k=7) | K (k=11) | N | convs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec_in_proj` | `Conv1d(32,2048,1)` (fwd, K=32) | -| -| -| -| 40 | 1 |
| `conv_pre` | `Conv1d(2048,1024,7)` (fwd, K=14336) | -| -| -| -| 40 | 1 |
| `ups.0` | `ConvTranspose1d(1024,512,9,stride=5)` -> M=512, K=9216 | 512 | 1536 | 3584 | 5632 | 200 | 1 + 18 |
| `ups.1` | `ConvTranspose1d(512,256,9,stride=5)` -> M=256, K=4608 | 256 | 768 | 1792 | 2816 | 1000 | 1 + 18 |
| `ups.2` | `ConvTranspose1d(256,128,4,stride=2)` -> M=128, K=1024 | 128 | 384 | 896 | 1408 | 2000 | 1 + 18 |
| `ups.3` | `ConvTranspose1d(128,64,4,stride=2)` -> M=64, K=512 | 64 | 192 | 448 | 704 | 4000 | 1 + 18 |
| `ups.4` | `ConvTranspose1d(64,32,4,stride=2)` -> M=32, K=256 | 32 | 96 | 224 | 352 | 8000 | 1 + 18 |
| `ups.5` | `ConvTranspose1d(32,16,4,stride=2)` -> M=16, K=128 | 16 | 48 | 112 | 176 | 16000 | 1 + 18 |
| `ups.6` | `ConvTranspose1d(16,8,4,stride=2)` -> M=8, K=64 | 8 | 24 | 56 | 88 | 32000 | 1 + 18 |
| `conv_post` | `Conv1d(8,1,7)` (fwd, M=1, K=56) | -| -| -| -| 32000 | 1 |

Every stage contributes 19 conv rows (1 transposed upsample + 3 kernels x 3 dilations x 2 convs), all with the same `N`. The `K` values are `C*k` for `fwd` rows and `C_in_t*k` for the `dgrad` rows. The `M` of a dgrad row is the transposed conv's output channel count (`weight.shape[1]`), which is also the resblock channel count.

### `(M, K)` families

`fwd`: `(64,7)`, `(64,64)`, `(64,448)`, `(128,256)`, `(128,128)`, `(128,896)`, `(256,1024)`, `(256,256)`, `(256,1792)`, `(512,2048)`, `(512,512)`, `(512,3584)`, `(1024,5120)`, `(1024,1024)`, `(1024,7168)`, `(2048,10240)`, `(2048,6144)`, `(32,32)`, `(2048,32)`, `(1024,14336)`.
`dgrad`: `(512,9216)`, `(256,4608)`, `(128,1024)`, `(64,512)`, `(32,256)`, `(16,128)`, `(8,64)`.

`N` values: `1 s -> {40, 200, 1000, 2000, 4000, 8000, 16000, 32000}`, `640x480x40 -> {93, 465, 2325, 4650, 9300, 18600, 37200, 74400}`, `1280x720x80 -> {150, 750, 3750, 7500, 15000, 30000, 60000, 120000}`.

The N ladder is exactly `L_latent * 800 / {1,5,25,50,100,200,400}` because the decoder's seven up stages multiply the length by `5*5*2*2*2*2*2 = 800`.

### Geometry

Audio length is `ceil(L_samples / 800)` latent frames, so the documented video clips (24 fps, 40 and 80 requested frames, H3 effective 56 and 90 frames) give `round(56/24*40) = 93` and `round(90/24*40) = 150` latent frames and `74400` / `120000` samples. A 1 s clip gives `40` latents and `32000` samples and is included as the smallest reference point.

## LTX 2.3 Vocoder

Facts (`comfy/ldm/lightricks/vocoders/vocoder.py`): `VocoderWithBWE` chains a base `Vocoder` (HiFiGAN/BigVGAN) with a bandwidth-extension `Vocoder`. Input is the LTX mel spectrogram `[B, 2, 64, T_mel]` (stereo flattened to 128 channels by `conv_pre`), output is a stereo waveform. 213 conv modules: 116 in the base vocoder, 97 in the BWE generator.

Base vocoder: `conv_pre Conv1d(128, 1536, 7)`. Six `ConvTranspose1d` up stages with rates `(5,2,2,2,2,2)` and kernel sizes `(11,4,4,4,4,4)` reducing `1536 -> 768 -> 384 -> 192 -> 96 -> 48 -> 24`. Each stage followed by 3 `AMPBlock1` residual blocks with kernel sizes `(3,7,11)` and dilations `(1,3,5)` (18 convs per stage). `conv_post Conv1d(24, 2, 7)`. Total upsampling `5*2*2*2*2*2 = 160`, so 512 mel frames become 81920 samples at 16 kHz (`5.12 s`).

BWE: re-analyses the base output with a 512-point STFT at hop 80 (1024 frames for the 5.12 s clip), runs `conv_pre Conv1d(128, 512, 7)` and five up stages with rates `(6,5,2,2,2)` and kernels `(12,11,4,4,4)` reducing `512 -> 256 -> 128 -> 64 -> 32 -> 16`, with the same 18-conv residual blocks, and `conv_post Conv1d(16, 2, 7)`. Output is `512 * 480 = 245760` samples (`5.12 s` at 48 kHz).

### Stage template at the 5.12 s clip (512 mel frames)

| stage | upsample conv | resblock ch | K (k=3) | K (k=7) | K (k=11) | N | convs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base `conv_pre` | `Conv1d(128,1536,7)` (fwd, M=1536, K=896) | -| -| -| -| 512 | 1 |
| base `ups.0` | `ConvTranspose1d(1536,768,11,stride=5)` -> M=768, K=16896 | 768 | 2304 | 5376 | 8448 | 2560 | 1 + 18 |
| base `ups.1` | `ConvTranspose1d(768,384,4,stride=2)` -> M=384, K=3072 | 384 | 1152 | 2688 | 4224 | 5120 | 1 + 18 |
| base `ups.2` | `ConvTranspose1d(384,192,4,stride=2)` -> M=192, K=1536 | 192 | 576 | 1344 | 2112 | 10240 | 1 + 18 |
| base `ups.3` | `ConvTranspose1d(192,96,4,stride=2)` -> M=96, K=768 | 96 | 288 | 672 | 1056 | 20480 | 1 + 18 |
| base `ups.4` | `ConvTranspose1d(96,48,4,stride=2)` -> M=48, K=384 | 48 | 144 | 336 | 528 | 40960 | 1 + 18 |
| base `ups.5` | `ConvTranspose1d(48,24,4,stride=2)` -> M=24, K=192 | 24 | 72 | 168 | 264 | 81920 | 1 + 18 |
| base `conv_post` | `Conv1d(24,2,7)` (fwd, M=2, K=168) | -| -| -| -| 81920 | 1 |
| BWE `conv_pre` | `Conv1d(128,512,7)` (fwd, M=512, K=896) | -| -| -| -| 1024 | 1 |
| BWE `ups.0` | `ConvTranspose1d(512,256,12,stride=6)` -> M=256, K=6144 | 256 | 768 | 1792 | 2816 | 6144 | 1 + 18 |
| BWE `ups.1` | `ConvTranspose1d(256,128,11,stride=5)` -> M=128, K=2816 | 128 | 384 | 896 | 1408 | 30720 | 1 + 18 |
| BWE `ups.2` | `ConvTranspose1d(128,64,4,stride=2)` -> M=64, K=512 | 64 | 192 | 448 | 704 | 61440 | 1 + 18 |
| BWE `ups.3` | `ConvTranspose1d(64,32,4,stride=2)` -> M=32, K=256 | 32 | 96 | 224 | 352 | 122880 | 1 + 18 |
| BWE `ups.4` | `ConvTranspose1d(32,16,4,stride=2)` -> M=16, K=128 | 16 | 48 | 112 | 176 | 245760 | 1 + 18 |
| BWE `conv_post` | `Conv1d(16,2,7)` (fwd, M=2, K=112) | -| -| -| -| 245760 | 1 |

The transposed-conv rows dominate `N` (up to `245760` samples) while the residual-block rows have `K` up to `8448`. The dilations (1,3,5) appear only as `padding` differences (`3/9/15` for `k=3`, `9/15/21` for `k=7`, `15/25/35` for `k=11`), not as `K` changes.

### `(M, K)` families

`fwd`: `(1536,896)`, `(768,2304)`, `(768,5376)`, `(768,8448)`, `(384,1152)`, `(384,2688)`, `(384,4224)`, `(192,576)`, `(192,1344)`, `(192,2112)`, `(96,288)`, `(96,672)`, `(96,1056)`, `(48,144)`, `(48,336)`, `(48,528)`, `(24,72)`, `(24,168)`, `(24,264)`, `(512,896)`, `(256,768)`, `(256,1792)`, `(256,2816)`, `(128,384)`, `(128,896)`, `(128,1408)`, `(64,192)`, `(64,448)`, `(64,704)`, `(32,96)`, `(32,224)`, `(32,352)`, `(16,48)`, `(16,112)`, `(16,176)`, `(2,168)`, `(2,112)`.
`dgrad`: `(768,16896)`, `(384,3072)`, `(192,1536)`, `(96,768)`, `(48,384)`, `(24,192)`, `(256,6144)`, `(128,2816)`, `(64,512)`, `(32,256)`, `(16,128)`.

`N` values: `5.12 s -> 13 values from 512 to 245760`, `640x480x40 -> 13 values from 165 to 79200`, `1280x720x80 -> 13 values from 329 to 157920`.

### Geometry

The LTX 2.3 audio latent rate is `16000 / 160 / 4 = 25` latents/s, and the mel length of a clip is `4 * latents - 3` (causal). The documented video clips give `round(40/24*25) = 42 -> 165` mel frames and `round(80/24*25) = 83 -> 329` mel frames. The 5.12 s training clip is 512 mel frames. The vocoder resamples the base-vocoder output with a 512-point STFT at hop 80, so the BWE stage sees `81920/80 = 1024` frames for the 5.12 s clip.

## Special Cases

### Constant call count, varying N

Both 1D stacks are fully convolutional and neither tiles nor chunks: the module call count is identical for every clip length, and only `N` scales. That makes these stacks good candidates for a static shape-specialized kernel set keyed on `(M, K, N)`.

### Dilated residual convs

The H3 DAC residual units use dilations `(1,3,9)` with kernel 7, and the BigVGAN `AMPBlock1` residual blocks use `(1,3,5)` with kernels `(3,7,11)`. Dilation does not change `K`, but it changes the effective receptive field and the padding (`pad = d*(k-1)/2`). The CSV records the actual `padding` and `dilation` per row. A kernel that indexes the weight matrix by `K` alone can treat all dilations identically.

### Transposed conv phase splitting

The `ConvTranspose1d` rows carry `implicit_gemm_form=dgrad`. A backend that implements dgrad by phase (one GEMM per stride phase) sees `stride` GEMMs with `N/stride`, e.g. the base `ups.0` (`stride=5`, `N=2560`) becomes five GEMMs with `N=512` and the same `M=768, K=16896`. The CSV keeps the logical single-call tuple. The phase split is a backend detail.

## Reproducing The Data

- `collect_conv_shapes.py` builds the H3 audio VAE from the local `minimax_h3_audio_vae_fp32.safetensors` header and the LTX 2.3 vocoder from the `vocoder.*` tensors of the all-in-one checkpoint header, on `torch.device("meta")`, then records every `Conv1d`/`ConvTranspose1d` call for the documented clip lengths.
- `build_datasets.py` derives the `fwd` and `dgrad` implicit-GEMM columns, aggregates identical calls into `count`, and writes the CSV/JSON files.
- The LTX vocoder modules are constructed directly from the checkpoint's `vocoder`/`bwe` config (`comfy/ldm/lightricks/vocoders/vocoder.py`) because the ComfyUI `VAE` loader path wraps them behind the mel autoencoder.

## Notes And Caveats

- Inference only. No dgrad/wgrad training shapes are recorded for the forward convs, and no parameter-gradient shapes for either stack.
- The H3 audio VAE's `pre_block` (`AttnProjection`) and the `logs_proj` posterior variance projection are excluded: `pre_block` is `Linear`-based and `logs_proj` is unused at inference (the encoder returns the posterior mean directly).
- The LTX 2.3 vocoder is not a VAE. It is recorded here because it is the 1D conv stage of the LTX 2.3 AV decode path. The corresponding mel autoencoder is in the 2D note.
- `batch` is the conv call's batch axis. For the H3 audio stack it is the folded stereo axis (`2`). For the LTX vocoder it is `1` (stereo is flattened into the channel axis of `conv_pre`).
- Sample-length rounding: H3 pads the waveform up to a multiple of 800 samples. LTX uses the mel length implied by its STFT hop. The lengths in the tables are the actual recorded values.
