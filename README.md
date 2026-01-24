# ComfyUI-FeatherOps

WIP. Use at your own risk.

The idea is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can send fp8 data to smem then upcast to fp16, rather than upcast the whole tensor to fp16 then send it to smem. This saves the bandwidth between VRAM and smem.

However, this is helpful only when the tensors are small. When the tensors go larger, the computation time goes by O(N^3), while the communication time only goes by O(N^2). It cannot achieve speedup on Z-Image and larger models, which are compute-bound rather than memory-bound.

Now I think int8 models (see https://github.com/silveroxides/ComfyUI-QuantOps ) are more preferable than fp8 models on Ampere and older GPUs, which actually reduce the computation time with faster int8 matmul than fp16 matmul. It's still a pity that Strix Halo does not have faster int8 matmul than fp16 matmul.
