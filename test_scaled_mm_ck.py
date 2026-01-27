#!/usr/bin/env python3

import torch

from kernel.ck.ck_kernel import scaled_mm_ck
from kernel.naive import scaled_mm_naive


def run_case(M, N, K, device, out_dtype, with_scale, with_bias):
    a_dtype = torch.float16
    b_dtype = torch.float8_e4m3fn

    a = torch.randn((M, K), device=device, dtype=torch.float32).to(a_dtype)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(b_dtype)

    scale = None
    if with_scale:
        scale = torch.tensor(1.7, device=device, dtype=torch.float32)

    bias = None
    if with_bias:
        bias = torch.randn((N,), device=device, dtype=out_dtype)

    out_ck = scaled_mm_ck(a, b, scale, bias, out_dtype)
    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype)

    out_ck = out_ck.float()
    out_ref = out_ref.float()
    diff = (out_ck - out_ref).abs()
    ref_abs = out_ref.abs()
    l2_rel = diff.norm() / ref_abs.norm().clamp_min(1e-6)
    linf_rel = diff.max() / ref_abs.max().clamp_min(1e-6)
    print(
        f"M={M} N={N} K={K} out={out_dtype} scale={with_scale} bias={with_bias} "
        f"rel_l2={l2_rel.item():.3g} rel_linf={linf_rel.item():.3g} "
        f"mean_atol={diff.mean().item():.3g} max_atol={diff.max().item():.3g}"
    )


def main():
    device = "cuda"
    out_dtype = torch.float16
    run_case(64, 64, 64, device, out_dtype, with_scale=True, with_bias=True)
    run_case(128, 96, 160, device, out_dtype, with_scale=True, with_bias=False)
    run_case(192, 128, 80, device, out_dtype, with_scale=False, with_bias=True)


if __name__ == "__main__":
    main()
