#!/usr/bin/env python3

import torch

from kernel.naive import scaled_mm_naive
from kernel.triton.kernel import scaled_mm_triton

# scaled_mm_naive = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune")
# scaled_mm_triton = torch.compile(scaled_mm_triton, fullgraph=True, dynamic=False, mode="max-autotune")


def main():
    M = 256
    N = 256
    K = 256
    device = "cuda"
    a_dtype = torch.float16
    b_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16

    a = torch.randn((M, K), device=device, dtype=torch.float32).to(a_dtype)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(b_dtype)
    scale = torch.tensor(2.34, device=device, dtype=out_dtype)
    bias = torch.randn(N, device=device, dtype=out_dtype)

    out_triton = scaled_mm_triton(a, b, scale, bias, out_dtype)
    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype)

    out_triton = out_triton.float()
    out_ref = out_ref.float()
    diff = (out_triton - out_ref).abs()
    l2_rel_err = diff.norm() / out_ref.abs().norm().clamp_min(1e-6)
    l2_rel_err = l2_rel_err.item()
    max_abs_err = diff.max().item()
    print(f"l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}")


if __name__ == "__main__":
    main()
