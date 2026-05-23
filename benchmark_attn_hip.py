#!/usr/bin/env python3

import gc

import torch
import triton
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from kernel_attn.hip.hip_kernel import attn_hip

BATCH = 1
HEADS = 24
HEAD_DIM = 128
SEQ_LENS = [1024, 2048, 4096, 8192]


def aiter_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, _, _, _ = flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return out


providers = {
    "aiter": aiter_attn,
    "hip": attn_hip,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=SEQ_LENS,
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="TFLOPS",
            plot_name="attn_hip",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda"

    if provider == "aiter":
        q = torch.randn((BATCH, N, HEADS, HEAD_DIM), device=device, dtype=torch.float16)
        k = torch.randn((BATCH, N, HEADS, HEAD_DIM), device=device, dtype=torch.float16)
        v = torch.randn((BATCH, N, HEADS, HEAD_DIM), device=device, dtype=torch.float16)
    elif provider == "hip":
        q = torch.randn((BATCH, HEADS, N, HEAD_DIM), device=device, dtype=torch.float16)
        k = torch.randn((BATCH, HEADS, N, HEAD_DIM), device=device, dtype=torch.float16)
        v = torch.randn((BATCH, HEADS, N, HEAD_DIM), device=device, dtype=torch.float16)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    fn = lambda: providers[provider](q, k, v)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=quantiles)

    perf = lambda ms: 4 * BATCH * HEADS * N**2 * HEAD_DIM / ms * 1e-9
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
