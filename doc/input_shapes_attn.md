# Forward Fused Attention Kernel Shapes

Machine-readable rows are saved in `tmp_attn_fp8kv_analysis/shape_data/input_shapes_attn.json`.

## Scope

- Only forward-pass fused attention calls are recorded.
- Internal attention matmuls such as `Q @ K.T` and `softmax(QK) @ V` are not recorded.
- Shapes use canonical head-split layout: `q=[batch, heads, query_tokens, head_dim]` and `k/v=[batch, heads, key_value_tokens, head_dim]`.
- Counts are per single model forward call with batch size 1. Classifier-free-guidance batching would multiply batch size.
- Wan and LTX 2.3 use exact fused attention sequence lengths, not the GEMM-rounded row counts used in `doc/input_shapes.md`.
- Z-Image uses the same padded text/image sequence lengths as its GEMM rows because ComfyUI pads both streams to multiples of 32 before fused attention.
- Krea 2 uses a TextFusion adapter before the main DiT; its layerwise attention uses batch 16 and sequence length 12 for the 12 tapped Qwen3-VL hidden states.

## Shape Summary

| Model | Component | Input | Attention | Count | Fused shape |
| --- | --- | --- | --- | ---: | --- |
| SDXL | UNet width 640 | `1024x1024` | self | 10 | `q/k/v=[1,10,4096,64]` |
| SDXL | UNet width 640 | `1024x1024` | text cross | 10 | `q=[1,10,4096,64]`, `k/v=[1,10,16,64]` |
| SDXL | UNet width 1280 | `1024x1024` | self | 60 | `q/k/v=[1,20,1024,64]` |
| SDXL | UNet width 1280 | `1024x1024` | text cross | 60 | `q=[1,20,1024,64]`, `k/v=[1,20,16,64]` |
| SDXL | UNet width 640 | `1536x1536` | self | 10 | `q/k/v=[1,10,9216,64]` |
| SDXL | UNet width 640 | `1536x1536` | text cross | 10 | `q=[1,10,9216,64]`, `k/v=[1,10,16,64]` |
| SDXL | UNet width 1280 | `1536x1536` | self | 60 | `q/k/v=[1,20,2304,64]` |
| SDXL | UNet width 1280 | `1536x1536` | text cross | 60 | `q=[1,20,2304,64]`, `k/v=[1,20,16,64]` |
| SDXL | CLIP-L text encoder | `prompt_16` | self | 12 | `q/k/v=[1,12,16,64]` |
| SDXL | OpenCLIP-G text encoder | `prompt_16` | self | 32 | `q/k/v=[1,20,16,64]` |
| Anima | LLM adapter | `prompt_16` | self | 6 | `q/k/v=[1,16,16,64]` |
| Anima | LLM adapter | `prompt_16` | text cross | 6 | `q=[1,16,16,64]`, `k/v=[1,16,16,64]` |
| Anima | Main DiT | `1024x1024` | self | 28 | `q/k/v=[1,16,4096,128]` |
| Anima | Main DiT | `1024x1024` | text cross | 28 | `q=[1,16,4096,128]`, `k/v=[1,16,512,128]` |
| Anima | Main DiT | `1536x1536` | self | 28 | `q/k/v=[1,16,9216,128]` |
| Anima | Main DiT | `1536x1536` | text cross | 28 | `q=[1,16,9216,128]`, `k/v=[1,16,512,128]` |
| Qwen | Main DiT joint blocks | `1024x1024` | joint text+image self | 60 | `q/k/v=[1,24,4112,128]` |
| Qwen | Main DiT joint blocks | `1536x1536` | joint text+image self | 60 | `q/k/v=[1,24,9232,128]` |
| Z-Image | Context refiner | `1024x1024` | text refiner self | 2 | `q/k/v=[1,30,32,128]` |
| Z-Image | Noise refiner | `1024x1024` | image self | 2 | `q/k/v=[1,30,4096,128]` |
| Z-Image | Main DiT layers | `1024x1024` | joint text+image self | 30 | `q/k/v=[1,30,4128,128]` |
| Z-Image | Context refiner | `1536x1536` | text refiner self | 2 | `q/k/v=[1,30,32,128]` |
| Z-Image | Noise refiner | `1536x1536` | image self | 2 | `q/k/v=[1,30,9216,128]` |
| Z-Image | Main DiT layers | `1536x1536` | joint text+image self | 30 | `q/k/v=[1,30,9248,128]` |
| Krea 2 | TextFusion layerwise blocks | `1024x1024` | text layer-stack self | 2 | `q/k/v=[16,20,12,128]` |
| Krea 2 | TextFusion refiner blocks | `1024x1024` | text refiner self | 2 | `q/k/v=[1,20,16,128]` |
| Krea 2 | Main DiT blocks | `1024x1024` | joint text+image self | 28 | `q/k/v=[1,48,4112,128]` |
| Krea 2 | TextFusion layerwise blocks | `1536x1536` | text layer-stack self | 2 | `q/k/v=[16,20,12,128]` |
| Krea 2 | TextFusion refiner blocks | `1536x1536` | text refiner self | 2 | `q/k/v=[1,20,16,128]` |
| Krea 2 | Main DiT blocks | `1536x1536` | joint text+image self | 28 | `q/k/v=[1,48,9232,128]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | joint text+image masked self | 34 | `q/k/v=[1,18,4112,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | image-only self | 34 | `q/k/v=[1,18,4096,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | joint text+image masked self | 34 | `q/k/v=[1,18,9232,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | image-only self | 34 | `q/k/v=[1,18,9216,256]` |
| Klein 9B | Flux2 transformer blocks | `1024x1024` | joint text+image self | 32 | `q/k/v=[1,32,4608,128]` |
| Klein 9B | Flux2 transformer blocks | `1536x1536` | joint text+image self | 32 | `q/k/v=[1,32,9728,128]` |
| Wan | Main video blocks | `640x480x40` | self | 40 | `q/k/v=[1,40,12000,128]` |
| Wan | Main video blocks | `640x480x40` | text cross | 40 | `q=[1,40,12000,128]`, `k/v=[1,40,512,128]` |
| Wan | Main video blocks | `1280x720x80` | self | 40 | `q/k/v=[1,40,72000,128]` |
| Wan | Main video blocks | `1280x720x80` | text cross | 40 | `q=[1,40,72000,128]`, `k/v=[1,40,512,128]` |
| LTX 2.3 | Video text connector | `text_context_1024` | self | 8 | `q/k/v=[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | self | 48 | `q/k/v=[1,32,1500,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | text cross | 48 | `q=[1,32,1500,128]`, `k/v=[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | self | 48 | `q/k/v=[1,32,8800,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | text cross | 48 | `q=[1,32,8800,128]`, `k/v=[1,32,1024,128]` |

## Medium Attention Shapes

These rows have both `query_tokens` and `key_value_tokens` between 1,000 and 10,000 inclusive.

| Model | Component | Input | Attention | Count | Q shape | KV shape |
| --- | --- | --- | --- | ---: | --- | --- |
| SDXL | UNet width 640 | `1024x1024` | self | 10 | `[1,10,4096,64]` | `[1,10,4096,64]` |
| SDXL | UNet width 1280 | `1024x1024` | self | 60 | `[1,20,1024,64]` | `[1,20,1024,64]` |
| SDXL | UNet width 640 | `1536x1536` | self | 10 | `[1,10,9216,64]` | `[1,10,9216,64]` |
| SDXL | UNet width 1280 | `1536x1536` | self | 60 | `[1,20,2304,64]` | `[1,20,2304,64]` |
| Anima | Main DiT | `1024x1024` | self | 28 | `[1,16,4096,128]` | `[1,16,4096,128]` |
| Anima | Main DiT | `1536x1536` | self | 28 | `[1,16,9216,128]` | `[1,16,9216,128]` |
| Qwen | Main DiT joint blocks | `1024x1024` | joint text+image self | 60 | `[1,24,4112,128]` | `[1,24,4112,128]` |
| Qwen | Main DiT joint blocks | `1536x1536` | joint text+image self | 60 | `[1,24,9232,128]` | `[1,24,9232,128]` |
| Z-Image | Noise refiner | `1024x1024` | image self | 2 | `[1,30,4096,128]` | `[1,30,4096,128]` |
| Z-Image | Main DiT layers | `1024x1024` | joint text+image self | 30 | `[1,30,4128,128]` | `[1,30,4128,128]` |
| Z-Image | Noise refiner | `1536x1536` | image self | 2 | `[1,30,9216,128]` | `[1,30,9216,128]` |
| Z-Image | Main DiT layers | `1536x1536` | joint text+image self | 30 | `[1,30,9248,128]` | `[1,30,9248,128]` |
| Krea 2 | Main DiT blocks | `1024x1024` | joint text+image self | 28 | `[1,48,4112,128]` | `[1,48,4112,128]` |
| Krea 2 | Main DiT blocks | `1536x1536` | joint text+image self | 28 | `[1,48,9232,128]` | `[1,48,9232,128]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | joint text+image masked self | 34 | `[1,18,4112,256]` | `[1,18,4112,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1024x1024` | image-only self | 34 | `[1,18,4096,256]` | `[1,18,4096,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | joint text+image masked self | 34 | `[1,18,9232,256]` | `[1,18,9232,256]` |
| Ideogram 4 | Main DiT single-stream blocks | `1536x1536` | image-only self | 34 | `[1,18,9216,256]` | `[1,18,9216,256]` |
| Klein 9B | Flux2 transformer blocks | `1024x1024` | joint text+image self | 32 | `[1,32,4608,128]` | `[1,32,4608,128]` |
| Klein 9B | Flux2 transformer blocks | `1536x1536` | joint text+image self | 32 | `[1,32,9728,128]` | `[1,32,9728,128]` |
| LTX 2.3 | Video text connector | `text_context_1024` | self | 8 | `[1,32,1024,128]` | `[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | self | 48 | `[1,32,1500,128]` | `[1,32,1500,128]` |
| LTX 2.3 | Main video blocks | `640x480x40` | text cross | 48 | `[1,32,1500,128]` | `[1,32,1024,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | self | 48 | `[1,32,8800,128]` | `[1,32,8800,128]` |
| LTX 2.3 | Main video blocks | `1280x720x80` | text cross | 48 | `[1,32,8800,128]` | `[1,32,1024,128]` |

## Model Notes

- SDXL UNet attention uses `head_dim=64`; width 640 has 10 heads and width 1280 has 20 heads.
- Anima main DiT cross-attention uses 512 key/value tokens because the LLM adapter pads its output before the main DiT consumes it.
- Qwen and Klein 9B use joint text+image fused attention rather than separate image self-attention and text cross-attention kernels.
- Z-Image uses separate 2-layer text context and image noise refiners, then 30 main joint text+image layers; 16 prompt rows are padded to 32 rows before attention.
- Krea 2 uses 2 TextFusion layerwise blocks over the 12 tapped text-encoder layers, 2 TextFusion refiner blocks over 16 prompt rows, then 28 main joint text+image layers; main K/V use 12 KV heads and are repeated to 48 heads before the fused attention call.
- Ideogram 4 uses single-stream masked self-attention over packed `[text, image]` tokens; image-only rows cover `context is None`.
- Wan T2V has one video self-attention and one text cross-attention call per layer.
- LTX 2.3 rows document the no-audio video workload; audio-only and audio/video cross-attention paths are excluded.
