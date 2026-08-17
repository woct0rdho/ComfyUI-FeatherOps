# Forward Fused Attention Kernel Shapes

Machine-readable rows are saved in `tmp_attn_fp8kv_analysis/shape_data/input_shapes_attn.json`.

## Scope

- Only forward-pass fused attention calls are recorded.
- Internal attention matmuls such as `Q @ K.T` and `softmax(QK) @ V` are not recorded.
- Shapes use canonical head-split layout: `q=[batch, heads, query_tokens, head_dim]` and `k/v=[batch, heads, key_value_tokens, head_dim]`.
- Counts are per single model forward call with batch size 1. Classifier-free-guidance batching would multiply batch size.
- Wan, LTX 2.3, and H3 use exact fused attention sequence lengths, not the GEMM-rounded row counts used in the GEMM shape documents.
- Z-Image uses the same padded text/image sequence lengths as its GEMM rows because ComfyUI pads both streams to multiples of 32 before fused attention.
- Krea 2 uses a TextFusion adapter before the main DiT; its layerwise attention uses batch 16 and sequence length 12 for the 12 tapped Qwen3-VL hidden states.

## Physical Layout

Physical layout is classified at the fused-attention boundary, after ComfyUI's standard head split when the caller supplies `[B,N,H*D]` tensors. `HND` means head-major storage (`stride(H)=N*D`, `stride(N)=D`); `NHD` means token-major storage (`stride(N)=H*D`, `stride(H)=D`). A logical `[B,H,N,D]` transpose view over token-major storage is therefore labeled NHD.

The label records H/N axis order, not compactness. Some fused-QKV views retain gaps in their outer strides. Such a view is still HND- or NHD-ordered, but a kernel requiring compact 4D tensors must materialize it. Q, K, and V are listed separately where their physical orders differ.

Across the 48 unique shape rows, 38 use NHD for Q/K/V, four use HND for Q/K/V, four use HND Q/K with NHD V, and two Klein rows aggregate two block implementations: eight HND Q/K/V calls and 24 calls with HND Q/K but NHD V. Resolution changes token counts, not layout.

## Shape Summary

| Model | Component | Input | Attention | Physical Q/K/V | Count | Fused shape |
| --- | --- | --- | --- | --- | ---: | --- |
| SDXL | UNet width 640 | `1024x1024` | self | NHD | 10 | `q/k/v=[1,10,4096,64]` |
| SDXL | UNet width 640 | `1024x1024` | text cross | NHD | 10 | `q=[1,10,4096,64]`, `k/v=[1,10,16,64]` |
| SDXL | UNet width 1280 | `1024x1024` | self | NHD | 60 | `q/k/v=[1,20,1024,64]` |
| SDXL | UNet width 1280 | `1024x1024` | text cross | NHD | 60 | `q=[1,20,1024,64]`, `k/v=[1,20,16,64]` |
| SDXL | UNet width 640 | `1536x1536` | self | NHD | 10 | `q/k/v=[1,10,9216,64]` |
| SDXL | UNet width 640 | `1536x1536` | text cross | NHD | 10 | `q=[1,10,9216,64]`, `k/v=[1,10,16,64]` |
| SDXL | UNet width 1280 | `1536x1536` | self | NHD | 60 | `q/k/v=[1,20,2304,64]` |
| SDXL | UNet width 1280 | `1536x1536` | text cross | NHD | 60 | `q=[1,20,2304,64]`, `k/v=[1,20,16,64]` |
| SDXL | CLIP-L text encoder | `prompt_16` | self | NHD | 12 | `q/k/v=[1,12,16,64]` |
| SDXL | OpenCLIP-G text encoder | `prompt_16` | self | NHD | 32 | `q/k/v=[1,20,16,64]` |
| Anima | LLM adapter | `prompt_16` | self | NHD | 6 | `q/k/v=[1,16,16,64]` |
| Anima | LLM adapter | `prompt_16` | text cross | NHD | 6 | `q=[1,16,16,64]`, `k/v=[1,16,16,64]` |
| Anima | Main DiT | `1024x1024` | self | NHD | 28 | `q/k/v=[1,16,4096,128]` |
| Anima | Main DiT | `1024x1024` | text cross | NHD | 28 | `q=[1,16,4096,128]`, `k/v=[1,16,512,128]` |
| Anima | Main DiT | `1536x1536` | self | NHD | 28 | `q/k/v=[1,16,9216,128]` |
| Anima | Main DiT | `1536x1536` | text cross | NHD | 28 | `q=[1,16,9216,128]`, `k/v=[1,16,512,128]` |
| Qwen | Main DiT joint blocks | `1024x1024` | joint text+image self | HND | 60 | `q/k/v=[1,24,4112,128]` |
| Qwen | Main DiT joint blocks | `1536x1536` | joint text+image self | HND | 60 | `q/k/v=[1,24,9232,128]` |
| Z-Image | Context refiner | `1024x1024` | text refiner self | NHD | 2 | `q/k/v=[1,30,32,128]` |
| Z-Image | Noise refiner | `1024x1024` | image self | NHD | 2 | `q/k/v=[1,30,4096,128]` |
| Z-Image | Main DiT layers | `1024x1024` | joint text+image self | NHD | 30 | `q/k/v=[1,30,4128,128]` |
| Z-Image | Context refiner | `1536x1536` | text refiner self | NHD | 2 | `q/k/v=[1,30,32,128]` |
| Z-Image | Noise refiner | `1536x1536` | image self | NHD | 2 | `q/k/v=[1,30,9216,128]` |
| Z-Image | Main DiT layers | `1536x1536` | joint text+image self | NHD | 30 | `q/k/v=[1,30,9248,128]` |
| Krea 2 | TextFusion layerwise blocks | `1024x1024` | text layer-stack self | Q/K HND; V NHD | 2 | `q/k/v=[16,20,12,128]` |
| Krea 2 | TextFusion refiner blocks | `1024x1024` | text refiner self | Q/K HND; V NHD | 2 | `q/k/v=[1,20,16,128]` |
| Krea 2 | Main DiT blocks | `1024x1024` | joint text+image self | HND | 28 | `q/k/v=[1,48,4112,128]` |
| Krea 2 | TextFusion layerwise blocks | `1536x1536` | text layer-stack self | Q/K HND; V NHD | 2 | `q/k/v=[16,20,12,128]` |
| Krea 2 | TextFusion refiner blocks | `1536x1536` | text refiner self | Q/K HND; V NHD | 2 | `q/k/v=[1,20,16,128]` |
| Krea 2 | Main DiT blocks | `1536x1536` | joint text+image self | HND | 28 | `q/k/v=[1,48,9232,128]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | joint text+image masked self | NHD | 34 | `q/k/v=[1,18,4112,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | image-only self | NHD | 34 | `q/k/v=[1,18,4096,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | joint text+image masked self | NHD | 34 | `q/k/v=[1,18,9232,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | image-only self | NHD | 34 | `q/k/v=[1,18,9216,256]` |
| Klein 9B | Flux2 transformer blocks | `1024x1024` | joint text+image self | 8 HND; 24 Q/K HND, V NHD | 32 | `q/k/v=[1,32,4608,128]` |
| Klein 9B | Flux2 transformer blocks | `1536x1536` | joint text+image self | 8 HND; 24 Q/K HND, V NHD | 32 | `q/k/v=[1,32,9728,128]` |
| Wan | Main video blocks | `640x480x40` | self | NHD | 40 | `q/k/v=[1,40,12000,128]` |
| Wan | Main video blocks | `640x480x40` | text cross | NHD | 40 | `q=[1,40,12000,128]`, `k/v=[1,40,512,128]` |
| Wan | Main video blocks | `1280x720x80` | self | NHD | 40 | `q/k/v=[1,40,72000,128]` |
| Wan | Main video blocks | `1280x720x80` | text cross | NHD | 40 | `q=[1,40,72000,128]`, `k/v=[1,40,512,128]` |
| LTX 2.3 | Video text connector | `text_context_1024` | self | NHD | 8 | `q/k/v=[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | self | NHD | 48 | `q/k/v=[1,32,1500,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | text cross | NHD | 48 | `q=[1,32,1500,128]`, `k/v=[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | self | NHD | 48 | `q/k/v=[1,32,8800,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | text cross | NHD | 48 | `q=[1,32,8800,128]`, `k/v=[1,32,1024,128]` |
| H3 | Token refiner | `prompt_16` | self | NHD | 2 | `q/k/v=[1,56,16,128]` |
| H3 | Main omni-transformer | `640x480x40` | joint text+audio+video self | NHD | 50 | `q/k/v=[1,56,5302,128]` |
| H3 | Main omni-transformer | `1280x720x80` | joint text+audio+video self | NHD | 50 | `q/k/v=[1,56,25156,128]` |

## Model Notes

- SDXL UNet attention uses `head_dim=64`; width 640 has 10 heads and width 1280 has 20 heads.
- Anima main DiT cross-attention uses 512 key/value tokens because the LLM adapter pads its output before the main DiT consumes it.
- Qwen and Klein 9B use joint text+image fused attention rather than separate image self-attention and text cross-attention kernels.
- Z-Image uses separate 2-layer text context and image noise refiners, then 30 main joint text+image layers; 16 prompt rows are padded to 32 rows before attention.
- Krea 2 uses 2 TextFusion layerwise blocks over the 12 tapped text-encoder layers, 2 TextFusion refiner blocks over 16 prompt rows, then 28 main joint text+image layers; main K/V use 12 KV heads and are repeated to 48 heads before the fused attention call.
- Ideogram 4 uses single-stream masked self-attention over packed `[text, image]` tokens; image-only rows cover `context is None`.
- Wan T2V has one video self-attention and one text cross-attention call per layer.
- LTX 2.3 rows document the no-audio video workload; audio-only and audio/video cross-attention paths are excluded.
- H3 first applies two 56-head self-attention calls to the 16 text rows, then each of its 50 main blocks attends over one packed text/audio/video sequence. Requested 40- and 80-frame clips become 56 and 90 H3 frames, producing exact packed lengths `16+186+5100=5302` and `16+300+24840=25156`. Keyframe and reference rows are excluded.
