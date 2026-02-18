# ComfyUI-FeatherOps

WIP. Use at your own risk.

The idea is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can send fp8 data to smem then upcast to fp16, rather than upcast the whole tensor to fp16 then send it to smem. This saves the bandwidth between VRAM and smem.

However, this is helpful only when the tensors are small, or when doing matrix-vector multiplication (batch-1 LLM decoding). When the tensors go larger, the computation time goes by O(N^3), while the transfer time only goes by O(N^2), so the speedup from optimizing the transfer becomes relatively low.

Now I think int8 models (see https://github.com/silveroxides/ComfyUI-QuantOps ) are more preferable than fp8 models on Ampere and older GPUs, which actually reduce the computation time with faster int8 matmul than fp16 matmul.

It's still a pity that Strix Halo does not have faster int8 matmul than fp16 matmul, and things like HIP compiler, Composable Kernel, Triton compiler are not yet well optimized on it. I'm experimenting some HIP mixed-precision matmul kernels in this repo. Currently when running large matmul, fp16 @ fp8e4m3fn is 27% faster than `torch.compile`, and fp16 @ fp8e5m2 is 35% faster. There is still space for optimization from the theoretical performance of 59.4 TFLOPS, mainly due to lack of compute-transfer overlapping.
