# gfx1151 FP16 3D Conv VAE Input Shapes

This note documents the `Conv3d` tensor shapes and their implicit-GEMM decompositions for the video VAEs, including the per-frame `Conv2d` resampler calls that the Wan VAE issues from inside its 3D graph.

Machine-readable rows are saved in `~/tmp/feather_ops/conv_vae_shapes/shape_data/`:
- `conv3d_vae_shapes.csv`: one aggregated row per `(model, VAE, workload, phase, module, effective kernel, input shape, output shape)` with derived `m, n, k, batch` and the identical-call `count`.
- `conv3d_vae_unique_gemm_shapes.csv`: unique `(op_kind, m, n, k)` rows with call counts and models.
- `conv3d_vae_shapes_manifest.json`: per-model counts, workload geometry, checkpoint/code sources, and column semantics.

The same directory holds `conv2d_*` and `conv1d_*` datasets for the other two notes. `conv_vae_manifest.json` indexes all three.

## Scope

- Models: Wan (Wan 2.1 VAE, 16-channel `/8` latent), LTX 2.3 (`CausalVideoAutoencoder` version 2, 128-channel `/32` latent), H3 (`MiniMaxH3VideoVAE`, 24-channel `/16` latent).
- Inference only: one VAE `encode` and one `decode` per workload, batch size 1. The VAEs are normally frozen.
- Video workloads: the documented `640x480x40` and `1280x720x80` requests. H3 rounds the effective frame count to its `17n+5` grid (`56` and `90` frames). LTX 2.3 crops 720 height to 704 and 40 frames to 33 before the encoder. Exact geometry is in each model section.
- Every video VAE operator is a `torch.nn.Conv3d` with `groups=1`, dilation 1, and a 3x3x3 or 1x1x1 (or 3x1x1 temporal-only) kernel, except the Wan 2.1 spatial resamplers, which are `torch.nn.Conv2d` applied per frame. Those 2D calls are included with `op_kind=conv2d`.
- No `ConvTranspose3d` appears in these VAEs. LTX 2.3 and H3 upsample with depth-to-space reshapes followed by convs. Wan 2.1 upsamples with `nn.Upsample` + a per-frame 2D conv.
- Excluded: attention-kernel matmuls. The Wan 2.1 mid attention block contributes its four 1x1x1 (or 1x1) projection convs, which are recorded. The H3 video VAE decoder is a ViT whose projections are `Linear` layers, so it contributes no convolutions at all beyond `post_quant_conv`.

## Implicit GEMM Mapping

For a 3D convolution with weight `[C_out, C_in, kd, kh, kw]` applied to `[B, C_in, D_in, H_in, W_in]`:

```text
M = C_out
N = D_out * H_out * W_out       # per batch element
K = C_in * kd_eff * kh * kw
batch = B
MACs = M * N * K * batch
```

`kd_eff` is the temporal kernel extent that is actually used. It is not always the stored `kd`:
- Wan 2.1 `CausalConv3d` (`comfy/ldm/wan/vae.py`) and the H3 `CausalConv3d` (`comfy/ldm/minimax/vae.py`) implement causal temporal padding by prepending zero frames and, when a single frame is available, by truncating the weight to the available taps (`super().forward(x, autopad="causal_zero")`, which slices `weight[:, :, -input.shape[2]:, :, :]`). A `T=1` call therefore has `kd_eff=1` and `K = C_in * 1 * kh * kw`.
- LTX `CausalConv3d` (`comfy/ldm/lightricks/vae/causal_conv3d.py`) instead replicates the first frame to fill the temporal padding (`x[:, :, :1].repeat(..., time_kernel_size - 1, ...)`) and then runs the full kernel, so `kd_eff = kd = 3` even at `T=1`.
- Kernel `3x1x1` convs (`time_conv` in the Wan resamplers) are temporal-only channel projections. Their `K = C_in * 3`.
- Kernel `1x1x1` convs (latent projections, attention projections, shortcuts) are plain GEMMs with `K = C_in`.

For the embedded `Conv2d` rows in the Wan VAE, `batch` is the folded `b*t` frame count produced by ComfyUI's `rearrange('b c t h w -> (b t) c h w')`, `N = H_out * W_out`, and `K = C_in * kh * kw`.

`n_total = batch * n` and `flops_per_call = 2 * m * n * k * batch`. A pass total is `sum(flops_per_call * count)`.

## Model To VAE Mapping

| Model | VAE | Weights | Latent | Conv modules | Conv calls per pass (encode + decode) |
| --- | --- | --- | --- | ---: | ---: |
| Wan | Wan 2.1 VAE | `wan_2.1_vae.safetensors` | 16 ch, `/8` spatial, `/4` temporal | 71 (61 conv3d, 10 conv2d resampler) | 382 + 445 at `480p40`, 767 + 890 at `720p80` |
| LTX 2.3 | `CausalVideoAutoencoder` v2 | `vae.*` in `ltx-2.3-22b-dev.safetensors` | 128 ch, `/32` spatial, `/8` temporal | 84 (all conv3d) | 84 + 42 |
| H3 | `MiniMaxH3VideoVAE` | `minimax_h3_video_vae_fp16.safetensors` | 24 ch, `/16` spatial, `17 frames <-> 5 latents` | 35 (33 encoder conv3d + `quant_conv` + `post_quant_conv`, decoder conv-free) | 1224 + 9 at `480p56`, 5712 + 40 at `720p90` |

## Aggregate Counts

The 3D dataset contains 599 aggregated rows, of which 557 are `conv3d` and 42 are the Wan VAE's embedded `conv2d` resamplers. There are 157 unique `(op_kind, m, n, k)` shapes across 3 models.

| Model | Unique weight shapes | Unique `(op,m,n,k)` | `640x480x40` calls / GMACs (enc + dec) | `1280x720x80` calls / GMACs (enc + dec) |
| --- | ---: | ---: | ---: | ---: |
| Wan | 33 | 78 | 382 + 445 / 27918.2 + 47195.6 | 767 + 890 / 175967.1 + 297243.0 |
| LTX 2.3 | 14 | 61 | 84 + 42 / 12353.8 + 12731.7 | 84 + 42 / 79105.8 + 81584.6 |
| H3 | 14 | 18 | 1224 + 9 / 161188.9 + 0.0 | 5712 + 40 / 752214.7 + 0.1 |

`M` values across the three VAEs: `3, 16, 24, 32, 48, 64, 96, 128, 129, 192, 256, 384, 512, 768, 1024, 4096`. `K` values span `16` (1x1x1 16->16) to `27648` (`1024` channels x 27). The largest single-GEMM `N` in the family is `633600` (LTX 2.3 `480p40` encoder bulk pass). The largest `M*K` is `M=1024, K=27648`.

## Wan 2.1 VAE

Facts: `comfy/ldm/wan/vae.py`, `dim=96` (`decoder.head.0.gamma`), `z_dim=16`, `dim_mult=[1,2,4,4]`, `num_res_blocks=2`, `temperal_downsample=[False,True,True]`, `attn_scales=[]` (a single mid attention block), causal 3x3x3 convs, 3x3 `Conv2d` spatial resamplers, and `CACHE_T=2` temporal caching. Encoder/decoder are 5D end to end. The latent is `[1,16,T,H/8,W/8]`.

Geometry: `T_lat = ((F-1)//4)+1` after the encoder truncates the input to `1 + ((F-1)//4)*4` frames. `640x480x40` uses 37 of the 40 requested frames and produces `[1,16,10,60,80]`. `1280x720x80` uses 77 frames and produces `[1,16,20,90,160]`. Decode returns `(T_lat-1)*4+1 = 37` / `77` frames. The encoder processes the first frame alone and then 2-frame chunks (`iter_ = 1 + (t-1)//2`). The decoder does the same over latent frames (`iter_ = 1 + T_lat//2`).

Temporal truncation: the first chunk of each pass feeds a single frame to every causal conv, so those calls have `kd_eff=1` (`K` is one third of the bulk `K`). Every later chunk feeds a 2-frame slice that the module extends with the 2-frame cache, so `K` is the full `C_in*27`. The tables below split the two.

### Wan encoder, `640x480x40`, bulk 2-frame chunks

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `96,3,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,3,2,480,640` | `1,96,2,480,640` | 96 | 81 | 614400 | 18 |
| `96,96,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,96,2,480,640` | `1,96,2,480,640` | 96 | 2592 | 614400 | 72 |
| `192,96,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,96,2,240,320` | `1,192,2,240,320` | 192 | 2592 | 153600 | 18 |
| `192,192,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,192,2,240,320` | `1,192,2,240,320` | 192 | 5184 | 153600 | 54 |
| `96,96,3,3` | 3x3 | 2,2 | 0,0 | `1,96,481,641` | `1,96,240,320` | 96 | 864 | 76800 | 1 |
| `96,96,3,3` | 3x3 | 2,2 | 0,0 | `2,96,481,641` | `2,96,240,320` | 96 | 864 | 76800 | 18 |
| `384,192,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,192,2,120,160` | `1,384,2,120,160` | 384 | 5184 | 38400 | 9 |
| `384,384,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,384,2,120,160` | `1,384,2,120,160` | 384 | 10368 | 38400 | 27 |
| `192,192,3,3` | 3x3 | 2,2 | 0,0 | `1,192,241,321` | `1,192,120,160` | 192 | 1728 | 19200 | 1 |
| `192,192,3,3` | 3x3 | 2,2 | 0,0 | `2,192,241,321` | `2,192,120,160` | 192 | 1728 | 19200 | 18 |
| `192,192,3,1,1` | 3x1x1 | 2,1,1 | 0,0,0 | `1,192,3,120,160` | `1,192,1,120,160` | 192 | 576 | 19200 | 18 |
| `384,384,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,384,2,60,80` | `1,384,2,60,80` | 384 | 10368 | 9600 | 32 |
| `32,384,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,384,2,60,80` | `1,32,2,60,80` | 32 | 10368 | 9600 | 4 |
| `384,384,3,3` | 3x3 | 2,2 | 0,0 | `1,384,121,161` | `1,384,60,80` | 384 | 3456 | 4800 | 1 |
| `384,384,3,3` | 3x3 | 2,2 | 0,0 | `2,384,121,161` | `2,384,60,80` | 384 | 3456 | 4800 | 9 |
| `384,384,3,1,1` | 3x1x1 | 2,1,1 | 0,0,0 | `1,384,3,60,80` | `1,384,1,60,80` | 384 | 1152 | 4800 | 9 |

The 2D resampler rows are the ones with `in` shaped `[b*t, C, H+1, W+1]` (the `ZeroPad2d((0,1,0,1))` is outside the module, so the recorded input is `481x641`, `241x321`, `121x161`) or with `batch = 2` when two frames are folded into the batch axis. `n=76800/19200/4800` corresponds to the half-resolution outputs `240x320`, `120x160`, `60x80`.

The `time_conv` rows (`3x1x1` and the 1x1x1 latent projections) are separated from the spatial convs in the same table: `192,192,3,1,1` is the encoder temporal downsample (`stride=(2,1,1)`), and `384,384,3,1,1` is the second one. Their `N` is the output spatial extent, so they scale with the stage resolution, not with `T`.

### Wan encoder, first chunk (single frame, `kd_eff=1`)

| weight | k_eff | in | out | M | K | N | calls |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `96,3,3,3,3` | 1x3x3 | `1,3,1,480,640` | `1,96,1,480,640` | 96 | 27 | 307200 | 1 |
| `96,96,3,3,3` | 1x3x3 | `1,96,1,480,640` | `1,96,1,480,640` | 96 | 864 | 307200 | 4 |
| `192,96,3,3,3` | 1x3x3 | `1,96,1,240,320` | `1,192,1,240,320` | 192 | 864 | 76800 | 1 |
| `192,192,3,3,3` | 1x3x3 | `1,192,1,240,320` | `1,192,1,240,320` | 192 | 1728 | 76800 | 3 |
| `384,192,3,3,3` | 1x3x3 | `1,192,1,120,160` | `1,384,1,120,160` | 384 | 1728 | 19200 | 1 |
| `384,384,3,3,3` | 1x3x3 | `1,384,1,120,160` | `1,384,1,120,160` | 384 | 3456 | 19200 | 3 |
| `384,384,3,3,3` | 1x3x3 | `1,384,1,60,80` | `1,384,1,60,80` | 384 | 3456 | 4800 | 16 |
| `32,384,3,3,3` | 1x3x3 | `1,384,1,60,80` | `1,32,1,60,80` | 32 | 3456 | 4800 | 2 |

### Wan decoder, `640x480x40`, bulk 2-frame chunks

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `96,96,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,96,2,480,640` | `1,96,2,480,640` | 96 | 2592 | 614400 | 108 |
| `3,96,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,96,2,480,640` | `1,3,2,480,640` | 3 | 2592 | 614400 | 18 |
| `96,192,3,3` | 3x3 | 1,1 | 1,1 | `1,192,480,640` | `1,96,480,640` | 96 | 1728 | 307200 | 1 |
| `96,192,3,3` | 3x3 | 1,1 | 1,1 | `2,192,480,640` | `2,96,480,640` | 96 | 1728 | 307200 | 18 |
| `192,192,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,192,2,240,320` | `1,192,2,240,320` | 192 | 5184 | 153600 | 108 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,240,320` | `1,192,240,320` | 192 | 3456 | 76800 | 1 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `4,384,240,320` | `4,192,240,320` | 192 | 3456 | 76800 | 9 |
| `384,192,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,192,2,120,160` | `1,384,2,120,160` | 384 | 5184 | 38400 | 9 |
| `384,384,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,384,2,120,160` | `1,384,2,120,160` | 384 | 10368 | 38400 | 45 |
| `768,384,3,1,1` | 3x1x1 | 1,1,1 | 0,0,0 | `1,384,2,120,160` | `1,768,2,120,160` | 768 | 1152 | 38400 | 9 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `1,384,120,160` | `1,192,120,160` | 192 | 3456 | 19200 | 1 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `4,384,120,160` | `4,192,120,160` | 192 | 3456 | 19200 | 4 |
| `192,384,3,3` | 3x3 | 1,1 | 1,1 | `2,384,120,160` | `2,192,120,160` | 192 | 3456 | 19200 | 1 |
| `384,16,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,16,2,60,80` | `1,384,2,60,80` | 384 | 432 | 9600 | 4 |
| `384,384,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,384,2,60,80` | `1,384,2,60,80` | 384 | 10368 | 9600 | 40 |
| `768,384,3,1,1` | 3x1x1 | 1,1,1 | 0,0,0 | `1,384,2,60,80` | `1,768,2,60,80` | 768 | 1152 | 9600 | 4 |

### Wan decoder, first chunk (single frame, `kd_eff=1`)

| weight | k_eff | in | out | M | K | N | calls |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `384,16,3,3,3` | 1x3x3 | `1,16,1,60,80` | `1,384,1,60,80` | 384 | 144 | 4800 | 2 |
| `384,384,3,3,3` | 1x3x3 | `1,384,1,60,80` | `1,384,1,60,80` | 384 | 3456 | 4800 | 20 |
| `384,192,3,3,3` | 1x3x3 | `1,192,1,120,160` | `1,384,1,120,160` | 384 | 1728 | 19200 | 1 |
| `384,384,3,3,3` | 1x3x3 | `1,384,1,120,160` | `1,384,1,120,160` | 384 | 3456 | 19200 | 5 |
| `192,192,3,3,3` | 1x3x3 | `1,192,1,240,320` | `1,192,1,240,320` | 192 | 1728 | 76800 | 6 |
| `96,96,3,3,3` | 1x3x3 | `1,96,1,480,640` | `1,96,1,480,640` | 96 | 864 | 307200 | 6 |
| `3,96,3,3,3` | 1x3x3 | `1,96,1,480,640` | `1,3,1,480,640` | 3 | 864 | 307200 | 1 |

`1280x720x80` scales all spatial extents by `2.25` in area (`60x80 -> 90x160`, `120x160 -> 180x...` in the ratio `90/60 = 1.5` per axis) and doubles the latent frame count (`10 -> 20`) with `77` output frames, for `473.2` TMAC total versus `75.1` TMAC at `480x640x40`.

### Wan `(M, K)` families

| M | K (bulk) | K (first chunk) | kernel | op | modules | representative |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 384 | 10368 | 3456 | 3x3x3 | conv3d | 26 | `decoder.middle.0.residual.2` |
| 96 | 2592 | 864 | 3x3x3 | conv3d | 10 | `encoder.downsamples.0.residual.2` |
| 192 | 5184 | 1728 | 3x3x3 | conv3d | 9 | `encoder.downsamples.1.residual.6` |
| 384 | 5184 | 1728 | 3x3x3 | conv3d | 2 | `decoder.upsamples.4.residual.2` |
| 192 | 2592 | 864 | 3x3x3 | conv3d | 1 | `encoder.downsamples.3.residual.2` |
| 96 | 81 | 27 | 3x3x3 | conv3d | 1 | `encoder.conv1` |
| 32 | 10368 | 3456 | 3x3x3 | conv3d | 1 | `encoder.head.2` |
| 3 | 2592 | 864 | 3x3x3 | conv3d | 1 | `decoder.head.2` |
| 384 | 432 | 144 | 3x3x3 | conv3d | 1 | `decoder.conv1` |
| 192 | 3456 | -| 3x3 | conv2d | 2 | `decoder.upsamples.3.resample.1` |
| 96 | 864 | -| 3x3 | conv2d | 1 | `encoder.downsamples.2.resample.1` |
| 96 | 1728 | -| 3x3 | conv2d | 1 | `decoder.upsamples.11.resample.1` |
| 192 | 1728 | -| 3x3 | conv2d | 1 | `encoder.downsamples.5.resample.1` |
| 384 | 3456 | -| 3x3 | conv2d | 1 | `encoder.downsamples.8.resample.1` |
| 384 | 192 | -| 1x1x1 | conv3d | 2 | `decoder.upsamples.4.shortcut` |
| 192 | 96 | -| 1x1x1 | conv3d | 1 | `encoder.downsamples.3.shortcut` |
| 32 | 32 | -| 1x1x1 | conv3d | 1 | `conv1` |
| 16 | 16 | -| 1x1x1 | conv3d | 1 | `conv2` |
| 1152 | 384 | -| 1x1 | conv2d | 2 | `decoder.middle.1.to_qkv` (mid attention) |
| 384 | 384 | -| 1x1 | conv2d | 2 | `decoder.middle.1.proj` (mid attention) |
| 768 | 1152 | 384 | 3x1x1 | conv3d | 2 | `decoder.upsamples.3.time_conv` |
| 384 | 1152 | -| 3x1x1 | conv3d | 1 | `encoder.downsamples.8.time_conv` |
| 192 | 576 | -| 3x1x1 | conv3d | 1 | `encoder.downsamples.5.time_conv` |

The mid attention block (`decoder.middle.1`, `encoder.middle.1`) folds frames into the batch axis (`rearrange('b c t h w -> (b t) c h w')`), so its `to_qkv`/`proj` convs are `conv2d` rows with `batch = t`.

### Wan `N` values

`480p40`: `{4800, 9600, 19200, 38400, 48000, 76800, 153600, 307200, 614400}` (9 distinct values). `720p80`: `{14400, 28800, 57600, 115200, 230400, 288000, 460800, 921600, 1843200}`.

## LTX 2.3 Video VAE

Facts: `comfy/ldm/lightricks/vae/causal_video_autoencoder.py` `VideoVAE` with the checkpoint's `vae` config: `dims=3`, `in_channels=3`, `out_channels=3`, `latent_channels=128`, `patch_size=4`, `norm_layer=pixel_norm`, `latent_log_var=uniform`, `timestep_conditioning=true`, and the encoder/decoder block lists that mix `res_x` residual stacks with `compress_space_res`, `compress_time_res`, and `compress_all_res` `DepthToSpace` stages. 84 conv modules, all `Conv3d` (a `CausalConv3d` wrapper around an inner `ops.Conv3d`). The encoder ends in a `129`-channel `conv_out` (`128` mean + `1` uniform log-variance). The decoder ends in a `48`-channel `conv_out` that is unpatchified 4x4 back to RGB.

Geometry and staging: the encoder first patches 4x4 spatially (`in_channels = 3*4*4 = 48`) and trims the frame count to `1 + ((F-1)//8)*8`. `640x480x40` becomes 33 frames at `120x160`, and three `(2,2,2)` depth-to-space stages take it to `5x15x20`. `1280x720x80` becomes 73 frames at `320x...` after the 720->704 crop, then `10x22x40`. The decoder reverses this and returns `8*T_lat - 7` frames (`33` and `73`).

`CausalConv3d` replicates the first frame instead of truncating, so the first temporal call has `kd_eff=3` even though it sees a single frame: the encoder runs one `T=1` priming call (recorded input `T=3` after the replication) and then one bulk call over the remaining 32 frames (recorded input `T=34`, output `T=32`). The decoder has a similar first-call/bulk split at every level.

### LTX encoder, `640x480x40`

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `128,48,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,48,34,120,160` | `1,128,32,120,160` | 128 | 1296 | 614400 | 1 |
| `128,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,34,120,160` | `1,128,32,120,160` | 128 | 3456 | 614400 | 8 |
| `64,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,34,120,160` | `1,64,32,120,160` | 64 | 3456 | 614400 | 1 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,256,34,60,80` | `1,256,32,60,80` | 256 | 6912 | 153600 | 13 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,18,60,80` | `1,512,16,60,80` | 512 | 13824 | 76800 | 8 |
| `128,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,18,60,80` | `1,128,16,60,80` | 128 | 13824 | 76800 | 1 |
| `128,48,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,48,3,120,160` | `1,128,1,120,160` | 128 | 1296 | 19200 | 1 |
| `128,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,3,120,160` | `1,128,1,120,160` | 128 | 3456 | 19200 | 8 |
| `64,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,3,120,160` | `1,64,1,120,160` | 64 | 3456 | 19200 | 1 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,256,4,60,80` | `1,256,2,60,80` | 256 | 6912 | 9600 | 1 |
| `128,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,4,60,80` | `1,128,2,60,80` | 128 | 13824 | 9600 | 1 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,10,30,40` | `1,1024,8,30,40` | 1024 | 27648 | 9600 | 4 |
| `128,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,10,30,40` | `1,128,8,30,40` | 128 | 27648 | 9600 | 1 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,256,3,60,80` | `1,256,1,60,80` | 256 | 6912 | 4800 | 12 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,3,60,80` | `1,512,1,60,80` | 512 | 13824 | 4800 | 8 |
| `128,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,4,30,40` | `1,128,2,30,40` | 128 | 27648 | 2400 | 1 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,3,30,40` | `1,1024,1,30,40` | 1024 | 27648 | 1200 | 4 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,6,15,20` | `1,1024,4,15,20` | 1024 | 27648 | 1200 | 4 |
| `129,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,6,15,20` | `1,129,4,15,20` | 129 | 27648 | 1200 | 1 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,3,15,20` | `1,1024,1,15,20` | 1024 | 27648 | 300 | 4 |
| `129,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,3,15,20` | `1,129,1,15,20` | 129 | 27648 | 300 | 1 |

The alternating `T` values in the `1024`-channel rows (`10 -> 8`, `3 -> 1`, `6 -> 4`, `3 -> 1`) are the `compress_time_res`/`compress_all_res` stages: the depth-to-space stages reduce `T` and `H`/`W` while expanding channels, and the following residual stack sees the reduced extent.

### LTX decoder, `640x480x40`

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `128,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,35,120,160` | `1,128,33,120,160` | 128 | 3456 | 633600 | 8 |
| `48,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,35,120,160` | `1,48,33,120,160` | 48 | 3456 | 633600 | 1 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,256,35,60,80` | `1,256,33,60,80` | 256 | 6912 | 158400 | 12 |
| `512,256,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,256,35,60,80` | `1,512,33,60,80` | 512 | 6912 | 158400 | 1 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,19,60,80` | `1,512,17,60,80` | 512 | 13824 | 81600 | 9 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,11,30,40` | `1,512,9,30,40` | 512 | 13824 | 10800 | 4 |
| `4096,512,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,512,11,30,40` | `1,4096,9,30,40` | 4096 | 13824 | 10800 | 1 |
| `1024,128,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,128,7,15,20` | `1,1024,5,15,20` | 1024 | 3456 | 1500 | 1 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,7,15,20` | `1,1024,5,15,20` | 1024 | 27648 | 1500 | 4 |
| `4096,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,1,1 | `1,1024,7,15,20` | `1,4096,5,15,20` | 4096 | 27648 | 1500 | 1 |

The `4096`-channel rows are the depth-to-space upsamplers (`out_channels_reduction_factor=2/4`): a `3x3x3` conv expands `512 -> 4096` (or `1024 -> 4096`) at the pre-upsample resolution, and the following reshape trades channels for `T/H/W`.

### LTX `(M, K)` families

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 256 | 6912 | 3x3x3 | 25 | `decoder.up_blocks.6.res_blocks.0.conv1.conv` |
| 512 | 13824 | 3x3x3 | 21 | `encoder.down_blocks.4.res_blocks.0.conv1.conv` |
| 128 | 3456 | 3x3x3 | 16 | `encoder.down_blocks.0.res_blocks.0.conv1.conv` |
| 1024 | 27648 | 3x3x3 | 12 | `encoder.down_blocks.6.res_blocks.0.conv1.conv` |
| 128 | 1296 | 3x3x3 | 1 | `encoder.conv_in.conv` |
| 64 | 3456 | 3x3x3 | 1 | `encoder.down_blocks.1.conv.conv` |
| 128 | 13824 | 3x3x3 | 1 | `encoder.down_blocks.5.conv.conv` |
| 128 | 27648 | 3x3x3 | 1 | `encoder.down_blocks.7.conv.conv` |
| 129 | 27648 | 3x3x3 | 1 | `encoder.conv_out.conv` |
| 1024 | 3456 | 3x3x3 | 1 | `decoder.conv_in.conv` |
| 4096 | 27648 | 3x3x3 | 1 | `decoder.up_blocks.1.conv.conv` |
| 4096 | 13824 | 3x3x3 | 1 | `decoder.up_blocks.5.conv.conv` |
| 512 | 6912 | 3x3x3 | 1 | `decoder.up_blocks.3.conv.conv` |
| 48 | 3456 | 3x3x3 | 1 | `decoder.conv_out.conv` |

`N` values: `480p40` has 14 distinct values `{300, 1200, 1500, 2400, 4800, 9600, 10800, 19200, 38400, 76800, 81600, 153600, 158400, 633600}`. `720p80` has 16 values spanning `880` to `4111360`.

### LTX decode chunking

ComfyUI's `Decoder.run_up` splits the temporal axis into `num_chunks = ceil(numel * element_size / max_chunk_size)` before each up-block, where `max_chunk_size` comes from `get_max_chunk_size(device)` (32 MiB for <=6 GB of VRAM, 128 MiB for >=24 GB, linear in between). The LTX rows above are the uncapped single-chunk (whole-clip) execution, i.e. the mathematical operator shapes. Chunking preserves `M` and `K` and multiplies the number of calls while shrinking per-call `N`. The CSV `count` column is per whole-clip pass.

## H3 Video VAE

Facts: `comfy/ldm/minimax/vae.py` `MiniMaxH3VideoVAE`, 24 latent channels, spatial `/16`, temporal mapping `17 frames <-> 5 latents` (`vae_ratio_t=4`, `tokens_chunk_size=5`, `token_overlap=2`, `token_drop=3`, `frame_pre_padding=3`), spatial tiling at `tile_size=256` with `tile_overlap_min=64`, and `clip_length=17` for the encoder. The architecture is a 3D conv encoder (`128 -> 256 -> 256 -> 512 -> 512 -> 1024`, `48` output channels = `2*24`) plus a ViT decoder.

The decoder contains no convolutions: `ViT3DDecoder` patches the latent into tokens, runs `transformer_blocks` of `ops.Linear` projections, and unpatchifies. Only `post_quant_conv` (`Conv3d`, `24->24`, 1x1x1, applied per latent tile) is a conv. The conv-heavy part of this VAE is the encoder.

Encoder staging for `640x480x56`: 4 clips (`ceil(56/17)`) x 9 spatial tiles (`split_tiles(480) = 3`, `split_tiles(640) = 3`) = 36 calls per conv module, each on a `1x C x 17 x 256 x 256` tile. `1280x720x90`: 6 clips x 28 tiles (`split_tiles(720) = 4` with overlaps `[112,96,96]`, `split_tiles(1280) = 7`) = 168 calls per module. That is why the per-call shapes below are resolution-independent and only the call counts change with the request.

### H3 encoder, `640x480x56` (per 256x256x17 tile)

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `128,3,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,3,17,256,256` | `1,128,17,256,256` | 128 | 81 | 1114112 | 36 |
| `128,128,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,128,17,256,256` | `1,128,17,256,256` | 128 | 3456 | 1114112 | 144 |
| `128,128,3,3,3` | 3x3x3 | 1,2,2 | 0,0,0 | `1,128,17,256,256` | `1,128,17,128,128` | 128 | 3456 | 278528 | 36 |
| `256,128,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,128,17,128,128` | `1,256,17,128,128` | 256 | 3456 | 278528 | 36 |
| `256,128,1,1,1` | 1x1x1 | 1,1,1 | 0,0,0 | `1,128,17,128,128` | `1,256,17,128,128` | 256 | 128 | 278528 | 36 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,256,17,128,128` | `1,256,17,128,128` | 256 | 6912 | 278528 | 108 |
| `256,256,3,3,3` | 3x3x3 | 2,2,2 | 0,0,0 | `1,256,17,128,128` | `1,256,9,64,64` | 256 | 6912 | 36864 | 36 |
| `256,256,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,256,9,64,64` | `1,256,9,64,64` | 256 | 6912 | 36864 | 144 |
| `256,256,3,3,3` | 3x3x3 | 2,2,2 | 0,0,0 | `1,256,9,64,64` | `1,256,5,32,32` | 256 | 6912 | 5120 | 36 |
| `512,256,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,256,5,32,32` | `1,512,5,32,32` | 512 | 6912 | 5120 | 36 |
| `512,256,1,1,1` | 1x1x1 | 1,1,1 | 0,0,0 | `1,256,5,32,32` | `1,512,5,32,32` | 512 | 256 | 5120 | 36 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,512,5,32,32` | `1,512,5,32,32` | 512 | 13824 | 5120 | 108 |
| `512,512,3,3,3` | 3x3x3 | 1,2,2 | 0,0,0 | `1,512,5,32,32` | `1,512,5,16,16` | 512 | 13824 | 1280 | 36 |
| `512,512,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,512,5,16,16` | `1,512,5,16,16` | 512 | 13824 | 1280 | 144 |
| `1024,512,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,512,5,16,16` | `1,1024,5,16,16` | 1024 | 13824 | 1280 | 36 |
| `1024,512,1,1,1` | 1x1x1 | 1,1,1 | 0,0,0 | `1,512,5,16,16` | `1,1024,5,16,16` | 1024 | 512 | 1280 | 36 |
| `1024,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,1024,5,16,16` | `1,1024,5,16,16` | 1024 | 27648 | 1280 | 108 |
| `48,1024,3,3,3` | 3x3x3 | 1,1,1 | 0,0,0 | `1,1024,5,16,16` | `1,48,5,16,16` | 48 | 27648 | 1280 | 36 |
| `48,48,1,1,1` | 1x1x1 | 1,1,1 | 0,0,0 | `1,48,5,16,16` | `1,48,5,16,16` | 48 | 48 | 1280 | 36 |

The H3 `CausalConv3d` applies spatial padding outside the conv module (reflect `F.pad`), so the `padding` column is `0,0,0` and the recorded `in` shape is already padded. The temporal truncation rule applies only to a `T=1` call. Every tile here has `T=17` for the first clip and `T=5`-ish for the later ones (the recorded variants are collapsed into `count` because the tile shapes repeat).

### H3 decoder, `640x480x56` (only `post_quant_conv`)

| weight | k_eff | stride | pad | in | out | M | K | N | calls |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `24,24,1,1,1` | 1x1x1 | 1,1,1 | 0,0,0 | `3,24,7,16,16` | `3,24,7,16,16` | 24 | 24 | 1792 | 9 |

The `batch=3` axis is the spatial tile batch (the decoder tiles the latent into `16x16` blocks with overlap). `T=7` is `tokens_chunk_size + token_overlap = 5 + 2`. The `480p56` decode issues 9 such batched calls covering 27 latent tiles (3 temporal chunks x 9 spatial tiles). `720p90` issues 40 calls covering 140 tiles (5 temporal chunks x 28 spatial tiles, batched 4 or 3 per call).

### H3 `(M, K)` families

| M | K | kernel | modules | representative |
| ---: | ---: | --- | ---: | --- |
| 256 | 6912 | 3x3x3 | 9 | `encoder.down.2.block.0.conv1` |
| 512 | 13824 | 3x3x3 | 8 | `encoder.down.4.block.0.conv2` |
| 128 | 3456 | 3x3x3 | 5 | `encoder.down.0.block.0.conv1` |
| 1024 | 27648 | 3x3x3 | 3 | `encoder.down.5.block.1.conv2` |
| 128 | 81 | 3x3x3 | 1 | `encoder.conv_in` |
| 256 | 3456 | 3x3x3 | 1 | `encoder.down.1.block.0.conv1` |
| 512 | 6912 | 3x3x3 | 1 | `encoder.down.3.block.0.conv1` |
| 1024 | 13824 | 3x3x3 | 1 | `encoder.down.5.block.0.conv1` |
| 256 | 128 | 1x1x1 | 1 | `encoder.down.1.block.0.nin_shortcut` |
| 512 | 256 | 1x1x1 | 1 | `encoder.down.3.block.0.nin_shortcut` |
| 1024 | 512 | 1x1x1 | 1 | `encoder.down.5.block.0.nin_shortcut` |
| 48 | 27648 | 3x3x3 | 1 | `encoder.conv_out` |
| 48 | 48 | 1x1x1 | 1 | `quant_conv` |
| 24 | 24 | 1x1x1 | 1 | `post_quant_conv` |

`N` values: `{1280, 1792, 5120, 36864, 278528, 1114112}` for both sizes.

## Special Cases

### Causal temporal truncation

The `kernel_eff` column (its temporal entry is what the tables call `kd_eff`) is the key difference between the Wan/H3 and LTX families. For Wan, the first chunk of a pass contributes `K = C_in*9` instead of `C_in*27` at the same output `N`. A per-step profile therefore sees two distinct `K` values for the same module. For LTX the temporal kernel is always full (`K = C_in*27`) because the missing frames are synthesized by replication.

### Frame folding into the batch axis

The Wan 2.1 `AttentionBlock` and `Resample` modules run their 2D convs through `rearrange('b c t h w -> (b t) c h w')`. The recorded `conv2d` rows in the Wan VAE show `batch` values of `1`, `2`, and `4` (the number of frames folded at that point), with `N` equal to the per-frame spatial extent. These are ordinary 2D conv GEMMs. Only the batch axis differs from the `conv2d` dataset.

### Temporal chunking

- Wan 2.1: encoder chunks `1 + (t-1)//2`, decoder chunks `1 + T_lat//2`. The residual caches carry `CACHE_T=2` frames. The different first-chunk `K` and the repeated bulk calls are visible as separate CSV rows and `count`s.
- LTX 2.3: the encoder is a single call over the trimmed clip (with one priming call). The decoder chunks by VRAM as described above.
- H3: temporal chunks of 5 latent tokens with 2 overlap on decode, clips of 17 frames on encode.

### Attention projections

The Wan 2.1 mid attention block contributes `to_qkv` (`M=1152, K=384` at `384` channels) and `proj` (`M=384, K=384`) as 1x1 convs, both folded per frame. LTX 2.3 has no attention in the VAE. H3's ViT decoder attention is `Linear`-based and therefore out of scope for this note.

## Reproducing The Data

- `fetch_headers.py` downloads the LTX 2.3 all-in-one checkpoint header (`vae.*`, `audio_vae.*`, `vocoder.*`, 872 KB of header JSON) over HTTP Range. The 46 GB payload is never downloaded. Wan and H3 use the local `wan_2.1_vae.safetensors`, `minimax_h3_video_vae_fp16.safetensors`.
- `collect_conv_shapes.py` builds each VAE from its header on `torch.device("meta")`, runs one `encode` and one `decode` per workload, and records every conv call through forward hooks. The Wan 2.1 VAEs' internal 1-frame/2-frame chunk loop is executed for real, so the first-chunk and bulk rows come from the actual code path rather than from a model of it.
- `vae_meta.py` overrides `comfy.model_management` memory/attention helpers and disables the LTX decoder's VRAM-driven chunking (`get_max_chunk_size`) so the recorded shapes are the whole-clip operator shapes. See the LTX decode chunking note for the rule ComfyUI applies instead.
- `build_datasets.py` and `doc_tables.py` derive the implicit-GEMM columns, aggregate identical calls, and emit the tables.

## Notes And Caveats

- Inference only. No `NT`/`NN` conv-gradient shapes are recorded. Training these VAEs would add dgrad/wgrad implicit-GEMM shapes for every conv with `K` swapped for the activation extent.
- `M` and `K` are resolution-independent for a fixed VAE. Only `N`, the call count, and the chunking/tiling decomposition change with the request. That is why the tables above list `N` for one workload per model and describe the scaling for the other.
- `n` is per batch element. `batch` is the conv call's batch axis. For the frame-folded 2D resampler rows it is the folded frame count, and for the H3 decoder `post_quant_conv` rows it is the spatial tile count.
- ComfyUI's `VAEDecodeTiled` path is not used by these video VAEs when `handles_tiling=True` (H3) or when the model chunks internally (Wan, LTX). The rows here are the model-internal single-pass calls.
- The LTX 2.3 rows come from the `vae.*` weights inside `ltx-2.3-22b-dev.safetensors` (header fetched over HTTP Range only. The 46 GB payload was not downloaded). The ComfyUI template loads the VAE from that all-in-one checkpoint.
- H3 rows come from the local `minimax_h3_video_vae_fp16.safetensors` (`MiniMaxH3VideoVAE`). The int8 ConvRot variant of the same VAE would report the same logical shapes with different storage.
