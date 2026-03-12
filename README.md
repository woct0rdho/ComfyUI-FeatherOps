# ComfyUI-FeatherOps

The idea is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can load fp8 data to smem (LDS) then upcast to fp16, rather than upcast the whole matrix to fp16 then load it to smem. This saves the bandwidth between VRAM and smem.

However, this is helpful only when the matrices are small, or when doing matrix-vector multiplication (as in batch-1 LLM decoding). When the matrices go larger (as in diffusion models), the compute time goes by O(N^3), while the load time only goes by O(N^2), so the speedup from optimizing the load time diminishes.

On Ampere and older Nvidia GPUs, int8 models (see https://github.com/silveroxides/ComfyUI-QuantOps ) are more preferable than fp8 models, which actually reduce the compute time with faster int8 matmul than fp16 matmul.

It's still a pity that Strix Halo does not have faster int8 matmul than fp16 matmul. I'm experimenting with some HIP mixed-precision matmul kernels in this repo. Although loading from VRAM to LDS is not the bottleneck, it takes less instructions to load fp8 than fp16 from LDS to VGPR, and we can convert fp8 to fp16 in VGPR. This improves compute-load overlap in the K-loop. Also, keeping fp8 rather than fp16 in LDS reduces LDS usage and improves occupancy.

Some benchmarks when running large matmul:
* fp16 @ fp8e5m2 reaches 52 TFLOPS in C++
* and 43 TFLOPS in Python with dispatch overhead, which can be reduced using torch.compile
* fp16 @ fp8e4m3 is a bit slower because it takes more instructions to convert
* torch fp16 @ fp16 (a Tensile kernel) only reaches 30 TFLOPS in Python
* Theoretical roofline on Strix Halo is 59.4 TFLOPS
