#!/usr/bin/env python3

import gc

import torch
import triton

from kernel.hip.hipblaslt_kernel_fp16 import mm_hipblaslt_fp16, mm_hipblaslt_fp16_colmajor, to_col_major

providers = {
    "torch_colmajor": lambda a, b, scale, bias, out_dtype, use_relu: torch.relu((a @ b) * scale[:, None] + bias[:, None])
    if use_relu
    else (a @ b) * scale[:, None] + bias[:, None],
    "hipblaslt_fast_layout": lambda a, b, scale, bias, out_dtype, use_relu: mm_hipblaslt_fp16_colmajor(
        a, b, scale, bias, out_dtype, use_relu=use_relu, solution_index=1112
    ),
    "hipblaslt_semantic": lambda a, b, scale, bias, out_dtype, use_relu: mm_hipblaslt_fp16(
        a.contiguous(), b.contiguous(), scale, bias, out_dtype, use_relu=use_relu
    ),
}
provider_names = list(providers)


def check_correctness(a, b, scale, bias, out_dtype, use_relu):
    out_ref = providers["torch_colmajor"](a, b, scale, bias, out_dtype, use_relu)
    out_hip = providers["hipblaslt_fast_layout"](a, b, scale, bias, out_dtype, use_relu)

    diff = (out_hip.float() - out_ref.float()).abs()
    rel_l2 = diff.norm() / out_ref.float().norm().clamp_min(1e-6)
    max_diff = diff.max().item()
    if rel_l2.item() > 0.01 or max_diff > 1.0:
        raise RuntimeError(f"correctness check failed: rel_l2={rel_l2.item():.3g} max_atol={max_diff:.3g}")


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GFLOPS",
            plot_name="mm_hipblaslt_fp16",
            args={"use_relu": True},
        )
    ]
)
def benchmark(N, provider, use_relu):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    out_dtype = torch.float16

    a = to_col_major(torch.randn((N, N), device=device, dtype=torch.float32).to(torch.float16))
    b = to_col_major(torch.randn((N, N), device=device, dtype=torch.float32).to(torch.float16))
    scale = torch.randn(N, device=device, dtype=torch.float32).abs() * 0.125 + 0.5
    bias = torch.randn(N, device=device, dtype=out_dtype)

    check_correctness(a, b, scale, bias, out_dtype, use_relu)

    fn = lambda: providers[provider](a, b, scale, bias, out_dtype, use_relu)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=30, rep=200, quantiles=quantiles)

    perf = lambda cur_ms: 2 * N**3 / cur_ms * 1e-6
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
