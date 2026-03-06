#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel_fp8 import _FP8_CONFIGS, _load_hip_fp8_extension, prepack_b_for_scaled_mm_hip_fp8
from kernel.naive import scaled_mm_naive


def test_config(ext, cfg, M, N, K, device, with_scale=True, with_bias=True):
    """Test a specific config and return (pass, error_msg)."""
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg

    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float8_e5m2)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float8_e5m2)

    scale = torch.tensor(2.34, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    b_prepacked = prepack_b_for_scaled_mm_hip_fp8(b)

    try:
        out_hip = ext.scaled_mm_fp8(
            a,
            b_prepacked,
            scale,
            bias,
            with_scale,
            with_bias,
            warps_m,
            warps_n,
            unroll_k,
            repeat_m,
            repeat_n,
        )
    except Exception as e:
        return False, f"LAUNCH ERROR: {e}"

    out_ref = scaled_mm_naive(a, b, scale if with_scale else None, bias if with_bias else None, torch.float16)

    # Compare
    diff = (out_hip.float() - out_ref.float()).abs()
    ref_abs = out_ref.float().abs()
    l2_rel = diff.norm() / ref_abs.norm().clamp_min(1e-6)
    max_diff = diff.max().item()

    if l2_rel.item() > 0.01 or max_diff > 1.0:
        return False, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"
    return True, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"


def main():
    device = "cuda"
    ext = _load_hip_fp8_extension()

    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
    ]

    failed_configs = []
    for cfg in sorted(_FP8_CONFIGS):
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        block_m = 16 * warps_m * repeat_m
        block_n = 16 * warps_n * repeat_n
        chunk_k = 16 * unroll_k

        config_passed = True
        for M, N, K in test_sizes:
            if M % block_m != 0 or N % block_n != 0 or K % chunk_k != 0:
                continue

            passed, msg = test_config(ext, cfg, M, N, K, device, with_scale=True, with_bias=True)
            if not passed:
                config_passed = False
                print(f"FAIL: {cfg} M={M} N={N} K={K} - {msg}")
        if not config_passed:
            failed_configs.append(cfg)
        else:
            print(f"PASS: {cfg}")


if __name__ == "__main__":
    main()
