# ComfyUI-FeatherOps

This node pack provides speedup of int8 model, fp8 model, and VAE on RDNA 3/3.5 GPUs.

See https://github.com/woct0rdho/feather-ops for the kernels. Docs are maintained in the current repo.

## Use in ComfyUI

You can ask AI if you don't know how to install.

1. Install feather-ops
2. Install [comfy-quants](https://github.com/Comfy-Org/comfy-quants)
3. git clone this repo to `ComfyUI/custom_nodes/`
4. In ComfyUI, use `FeatherUNetLoader` node to load the model, and `FeatherVAELoader` to load the VAE. See the [example workflows](https://github.com/woct0rdho/ComfyUI-FeatherOps/tree/master/example_workflows)

LoRA and torch.compile are supported.

Note on torch.compile: Recently ComfyUI introduced comfy-aimdo but it does not yet work with torch.compile . You may disable it with `--disable-dynamic-vram` when starting ComfyUI. On Strix Halo with unified memory we don't need dynamic VRAM anyway.

Note on FlashAttention: If you correctly install FlashAttention with AITER Triton kernel, and tune the configs like https://github.com/woct0rdho/aiter/commit/103969ac5e6f6b79773f7d9c5207da3725e32031 , and add `--use-flash-attention` when starting ComfyUI, it can reach ~30 TFLOPS on Strix Halo. Without FlashAttention, ComfyUI may use slice attention by default and it's easy to OOM. Currently FeatherOps only speeds up non-attention linear modules, so its speedup can stack with FlashAttention.
