#!/usr/bin/env python3

import torch

from kernel.kernel import scaled_mm_triton
from kernel.naive import scaled_mm_naive

# scaled_mm_naive = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
# scaled_mm_triton = torch.compile(scaled_mm_triton, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")


def main():
    M = 128
    N = 128
    K = 128
    device = "cuda"
    a_dtype = torch.float16
    b_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16

    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    scale = torch.tensor(2.34, device=device, dtype=torch.float32)
    bias = torch.randn((N,), device=device, dtype=torch.float32)

    a = a.to(a_dtype)
    b = b.to(b_dtype)

    out_triton = scaled_mm_triton(a, b, scale, bias, out_dtype)

    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype)

    out_triton = out_triton.float()
    out_ref = out_ref.float()
    diff = (out_triton - out_ref).abs()
    rdiff = diff / out_ref.abs()
    print(f"mean_rtol={rdiff.mean().item():.3g} max_rtol={rdiff.max().item():.3g} mean_atol={diff.max().item():.3g} max_atol={diff.max().item():.3g}")


if __name__ == "__main__":
    main()
