#!/usr/bin/env python3

import gc

import torch
import triton

from kernel.hip.hipblaslt_kernel_fp16 import mm_hipblaslt_fp16_colmajor, to_col_major
from kernel.naive import scaled_mm_naive

scaled_mm_naive_compiled = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune")

providers = {
    "torch": scaled_mm_naive,
    "torch_compiled": scaled_mm_naive_compiled,
    "hipblaslt": mm_hipblaslt_fp16_colmajor,
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

    a_col = to_col_major(a)
    b_col = to_col_major(b)

    if provider in {"torch", "torch_compiled"}:
        fn = lambda: providers[provider](a, b, scale, bias, out_dtype)
    elif provider == "hipblaslt":
        fn = lambda: providers[provider](a_col, b_col, scale, bias, out_dtype, solution_index=-2)
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
