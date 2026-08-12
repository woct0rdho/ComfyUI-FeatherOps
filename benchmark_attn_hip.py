#!/usr/bin/env python3

import gc
import os

import torch
import triton
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from kernel_attn.hip.hip_kernel import feather_attn

BATCH = 1
BENCHMARK_HEAD_COUNTS = [16, 32, 56]
HEAD_DIMS = [int(value) for value in os.environ.get("FEATHER_ATTN_BENCH_HEAD_DIMS", "64,128").split(",")]
SEQ_LENS = [4096, 8192, 16384]
BENCHMARK_SHAPES = [(heads, seq_len, head_dim) for head_dim in HEAD_DIMS for heads in BENCHMARK_HEAD_COUNTS for seq_len in SEQ_LENS]


def aiter_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, _, _, _ = flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return out


providers = {
    "aiter": aiter_attn,
    "feather": feather_attn,
}
provider_names = list(providers.keys())


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["H", "N", "D"],
            x_vals=BENCHMARK_SHAPES,
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="TFLOPS",
            plot_name="attn",
            args={},
            xlabel="(num_heads, seq_len, head_dim)",
        )
    ]
)
def benchmark(H, N, D, provider):
    print("H", H, "N", N, "D", D, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    if provider == "aiter":
        q = torch.randn((BATCH, N, H, D), device=device, dtype=torch.float16)
        k = torch.randn((BATCH, N, H, D), device=device, dtype=torch.float16)
        v = torch.randn((BATCH, N, H, D), device=device, dtype=torch.float16)
    elif provider == "feather":
        q = torch.randn((BATCH, H, N, D), device=device, dtype=torch.float16)
        k = torch.randn((BATCH, H, N, D), device=device, dtype=torch.float16)
        v = torch.randn((BATCH, H, N, D), device=device, dtype=torch.float16)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    fn = lambda: providers[provider](q, k, v)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=quantiles)

    perf = lambda ms: 4 * BATCH * H * N**2 * D / ms * 1e-9
    print("H", H, "N", N, "D", D, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
