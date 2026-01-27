#!/usr/bin/env python3

import gc

import torch
import triton

from kernel.hip.hip_kernel import scaled_mm_hip
from kernel.naive import scaled_mm_naive

scaled_mm_naive_compiled = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")

providers = {
    "torch": scaled_mm_naive,
    "torch_compiled": scaled_mm_naive_compiled,
    "hip": scaled_mm_hip,
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
            ylabel="GFLOPS",
            plot_name="scaled_mm_hip",
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
    b_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16

    a = torch.randn((N, N), device=device, dtype=torch.float32).to(a_dtype)
    b = torch.randn((N, N), device=device, dtype=torch.float32).to(b_dtype)
    scale = torch.tensor(2.34, device=device, dtype=torch.float32)
    bias = torch.randn((N,), device=device, dtype=out_dtype)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: providers[provider](a, b, scale, bias, out_dtype), quantiles=quantiles)

    perf = lambda ms: 2 * N**3 / ms * 1e-6
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
