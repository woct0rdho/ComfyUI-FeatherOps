#!/usr/bin/env python3

import gc

import torch
import torch.nn.functional as F
import triton

from kernel.hip.hipblaslt_kernel_fp16 import mm_hipblaslt_fp16
from kernel.naive import scaled_mm_naive

providers = {
    "torch_mm_TT": scaled_mm_naive,
    "torch_mm_TN": scaled_mm_naive,
    "torch_mm_NT": scaled_mm_naive,
    "torch_mm_NN": scaled_mm_naive,
    "torch_linear_TT": F.linear,
    "torch_linear_TN": F.linear,
    "torch_linear_NT": F.linear,
    "torch_linear_NN": F.linear,
    "hipblaslt_TT": mm_hipblaslt_fp16,
    "hipblaslt_TN": mm_hipblaslt_fp16,
    "hipblaslt_NT": mm_hipblaslt_fp16,
    "hipblaslt_NN": mm_hipblaslt_fp16,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="TFLOPS",
            plot_name="mm_hipblaslt_fp16",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    a_dtype = torch.float16
    b_dtype = torch.float16
    out_dtype = torch.float16

    a = torch.randn((N, N), device=device, dtype=torch.float32).to(a_dtype)
    b = torch.randn((N, N), device=device, dtype=torch.float32).to(b_dtype)
    scale = torch.ones(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=out_dtype)

    input_layout = provider[-2:]
    if provider.startswith("torch"):
        # PyTorch calls BLAS with swapped operands for row-major outputs
        input_layout = {
            "TT": "NN",
            "TN": "TN",
            "NT": "NT",
            "NN": "TT",
        }[input_layout]

    if input_layout == "TT":
        pass
    elif input_layout == "TN":
        b = b.T
    elif input_layout == "NT":
        a = a.T
    elif input_layout == "NN":
        a = a.T
        b = b.T
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    if provider.startswith("torch_mm"):
        fn = lambda: providers[provider](a, b, scale, bias, out_dtype, bias_dim=0)
    elif provider.startswith("torch_linear"):
        fn = lambda: providers[provider](a, b.T, bias)
    elif provider.startswith("hipblaslt"):
        fn = lambda: providers[provider](a, b, scale, bias, out_dtype, solution_index=-2)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=100, rep=1000, quantiles=quantiles)

    perf = lambda ms: 2 * N**3 / ms * 1e-9
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
