#!/usr/bin/env python3

import gc

import torch
import triton

from kernel.convert import bf16_to_fp16


def torch_convert(x):
    return x.to(torch.float16)


torch_convert_compiled = torch.compile(torch_convert, fullgraph=True, dynamic=False, mode="max-autotune")

providers = {
    "torch": torch_convert,
    "torch_compiled": torch_convert_compiled,
    "triton_asm": bf16_to_fp16,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024, 2048, 4096, 8192, 16384],
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GB/s",
            plot_name="convert_bf16_to_fp16",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    x = torch.randn((N, N), device=device, dtype=torch.bfloat16)
    fn = lambda: providers[provider](x)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=100, rep=1000, quantiles=quantiles)

    total_bytes = 4 * N * N
    perf = lambda ms: total_bytes / ms * 1e-6
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
