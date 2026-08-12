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
LAYOUTS = [value.strip().upper() for value in os.environ.get("FEATHER_ATTN_BENCH_LAYOUTS", "HND,NHD").split(",")]
if not LAYOUTS or any(layout not in {"HND", "NHD"} for layout in LAYOUTS):
    raise ValueError("FEATHER_ATTN_BENCH_LAYOUTS must contain only HND and/or NHD")
SEQ_LENS = [4096, 8192, 16384]
BENCHMARK_SHAPES = [(heads, seq_len, head_dim, layout) for layout in LAYOUTS for head_dim in HEAD_DIMS for heads in BENCHMARK_HEAD_COUNTS for seq_len in SEQ_LENS]


def aiter_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    if layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
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
    return out.transpose(1, 2) if layout == "HND" else out


provider_names = ["aiter", "feather"]


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["H", "N", "D", "layout"],
            x_vals=BENCHMARK_SHAPES,
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="TFLOPS",
            plot_name="attn",
            args={},
            xlabel="(num_heads, seq_len, head_dim, layout)",
        )
    ]
)
def benchmark(H, N, D, layout, provider):
    print("H", H, "N", N, "D", D, "layout", layout, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"
    if layout == "HND":
        shape = (BATCH, H, N, D)
    elif layout == "NHD":
        shape = (BATCH, N, H, D)
    else:
        raise RuntimeError(f"Unknown layout: {layout}")
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)

    if provider == "aiter":
        fn = lambda: aiter_attn(q, k, v, layout)
    elif provider == "feather":
        fn = lambda: feather_attn(q, k, v, layout)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=quantiles)

    perf = lambda ms: 4 * BATCH * H * N**2 * D / ms * 1e-9
    print("H", H, "N", N, "D", D, "layout", layout, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
