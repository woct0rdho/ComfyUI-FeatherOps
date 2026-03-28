# ComfyUI-FeatherOps

For now it's a proof of concept rather than great speedup in ComfyUI. Let's see how it can be further optimized.

## How it works

The idea is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can load fp8 data to smem (LDS) then upcast to fp16, rather than upcast the whole matrix to fp16 then load it to smem. This saves the VRAM -> smem bandwidth.

However, this is helpful only when the matrices are small, or when doing mat-vec multiplication (as in batch-1 LLM decoding). When the matrices go larger (as in diffusion models), the compute time goes by O(N^3), while the load time only goes by O(N^2), so the speedup from optimizing the VRAM -> smem load diminishes.

On Nvidia consumer Pascal (GTX 10xx, not P100), Turing, and Ampere GPUs, int8 models (see https://github.com/BobJohnson24/ComfyUI-INT8-Fast ) are preferable to fp8 models, which actually reduce the compute time with faster int8 matmul than fp16.

It's a pity that AMD RDNA3/3.5 GPUs do not have faster int8 matmul than fp16, but we can surprisingly see speedup with fp8 in large matmul. Although the load from VRAM to LDS is not the bottleneck, it takes less instructions to load fp8 than fp16 from LDS to VGPR, which improves compute-load overlap in the K-loop. Also, keeping fp8 rather than fp16 in LDS reduces LDS usage per workgroup and improves occupancy.

## Implementation details

* `kernel/hip/hip_kernel.cu` is the kernel used in ComfyUI. Other kernels are for experiments and not used in ComfyUI
* `kernel/hip/hip_kernel.py` contains a lot of boilerplate to support torch.compile in torch 2.10, which can be reduced in torch 2.11
* `comfy_feather/` contains all ComfyUI-related code

The kernel is written in HIP, with intrinsics and asm when needed, without abstraction levels like CK, Tensile, or Triton.

The kernel computes fp16 @ fp8e5m2 -> bf16 mixed precision matmul. fp8 @ fp8 seems not achieving further speedup. We choose fp8e5m2 rather than fp8e4m3, and fp16 rather than bf16, because it's extremely fast to upcast fp8e5m2 to fp16 in the K-loop. We use fp32 accumulator in the wmma, and downcast to bf16 as the output to avoid overflow in ComfyUI workloads.

The kernel requires the inputs to be aligned with the M/N/K block sizes, and there are no branches to handle OOB cases. This is satisfied in most AI models.

We prepack the B matrix into `(K/16, N, 16)` layout to enable fast 128-bit loads from VRAM to LDS, and ensure that threads with adjacent N load adjacent 128-bit elements. Note that the B matrix (weight) is in `(N, K)` rather than `(K, N)` layout in usual ComfyUI workloads.

On RDNA3, there are no ways like cp.async to explicitly control compute-load overlap, so for now we can only write a serial pipeline and hope it work well with the hardware scheduler. It seems setting `s_setprio 1` in the LDS -> VGPR load stages except the first stage improves the overall speed.

Split-K is implemented in the `split-k` branch but for now I can't see speedup with it in ComfyUI workloads.

The kernel is tested on Strix Halo, and it should also work with RDNA3 GPUs.

Benchmarks on Strix Halo, when the matrices are large: (The results may change with your driver, ROCm, and PyTorch versions)
* Theoretical roofline is 59.4 TFLOPS
* fp16 @ fp8e5m2 reaches 46 TFLOPS in C++
* and 43 TFLOPS in Python with dispatch overhead, which can be reduced using torch.compile
* torch fp16 @ fp16 (a Tensile kernel) only reaches 30 TFLOPS in Python

`doc/` contains some experiment logs. They may be outdated, and the remaining performance gap from the theoretical roofline is still not well explained. I've tried PC sampling and thread tracing but I could not fully understand the bottleneck. I guess it's either due to the hardware scheduler or the instruction prefetch, and we need better profiling or even a simulator to investigate it.

## Use in ComfyUI

1. Install the rocm-sdk-devel wheel from TheRock, and set the paths
2. git clone this repo to `ComfyUI/custom_nodes/`
3. Run `python test_scaled_mm_hip.py` to test the correctness
4. In ComfyUI, use `FeatherUNetLoader` node to load the model, which converts fp16/bf16 model to fp8e5m2 with the prepacked layout
5. Replace the text encode node with `FeatherCLIPTextEncodePadded`, which pads the tokens to a multiple of 16

Besides the text token count, it also requires the image token count (`width / 16 * height / 16`) to be a multiple of an M block size. The image token count should be a multiple of 128 for best speed, and the text token count can be a multiple of 16 if the prompt is short.

See the example workflow for details. For now I've only tested Qwen-Image. When running Qwen-Image with 1024x1024 image size, without distill LoRA, with full-graph compile, I can see a 10% speedup compared to the original bf16 model.

The fp8e5m2 quantization quality looks good for the original Qwen-Image model, but degrades when the distill LoRA is used. To debug it, you can compare `FeatherOps` and `DebugOps`, which have the same logic to apply LoRA and only differ in the quantized linear op. A better quantization method is to be implemented.

Note on torch.compile: Recently ComfyUI introduced comfy-aimdo but it does not work well with torch.compile . You may disable it with `--disable-dynamic-vram` when starting ComfyUI.

## TODO

* See what we can do with a fp16 @ fp16 kernel. Tensile is good at tuning parameters, but we still need an HIP kernel to better understand low-level behaviors such as how to utilize the hardware scheduler. We need better profiling or even a simulator to investigate it
* Besides the matmul op, see what we can do with the attention op. Currently the AMD Triton kernel in the FlashAttention repo reaches 20 TFLOPS on Strix Halo, which is still far from the theoretical roofline
* Better fp8e5m2 quantization. For now I just directly cast fp16/bf16 weights to fp8e5m2, and we can implement grid search of the scale and blockwise quantization for better quality
* Better heuristics for the autotuner. Currently the autotuner takes a long time to benchmark every config, including the obviously non-optimal ones. For very large matrices we may directly use the config `1,8,4,8,2`
* Support more models in ComfyUI. We need to exclude modules outside the transformer backbone, and mat-vec multiplications
