# ComfyUI-FeatherOps

See https://github.com/woct0rdho/feather-ops for an introduction and the kernels. Docs are maintained in the current repo.

## Use in ComfyUI

1. Install feather-ops
2. Install [comfy-quants](https://github.com/Comfy-Org/comfy-quants)
3. git clone this repo to `ComfyUI/custom_nodes/`
4. In ComfyUI, use `FeatherUNetLoader` node to load the model. See the [example workflows](https://github.com/woct0rdho/ComfyUI-FeatherOps/tree/master/example_workflows)

I recommend to load fp16/bf16 model, or fp8e5m2 model if you can find one, but not fp8e4m3 or fp4 model, because that would suffer from quantizing twice. For Wan, you can find fp8e5m2 models at https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled

For the best speed, the latent token count should be a multiple of 128. The latent token count is computed by:
- Most image models with 16-channel VAE: `width / 16 * height / 16 * 16`
- LTX 2.3 video-only: `width / 32 * height / 32 * (length + 7) / 8`. I haven't checked what's the size constraint when generating video with audio. If you did, please open an issue.
- Wan 2.1/2.2 14B: `width / 16 * height / 16 * (length + 3) / 4`

LoRA and torch.compile are supported.

For some workloads you may see 30~50% speedup compared to not using `FeatherUNetLoader`, but your mileage may vary.

Note on torch.compile: Recently ComfyUI introduced comfy-aimdo but it does not yet work with torch.compile . You may disable it with `--disable-dynamic-vram` when starting ComfyUI.

Note on TAESD preview: Currently ComfyUI may have bugs when running Anima with TAESD preview. You may disable it.

Note on FlashAttention: If you correctly install FlashAttention with AITER Triton kernel, and tune the configs like https://github.com/woct0rdho/aiter/commit/103969ac5e6f6b79773f7d9c5207da3725e32031 , it can reach ~30 TFLOPS on Strix Halo. Currently FeatherOps only speeds up non-attention linear modules, so its speedup can stack with FlashAttention.
