#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel_fp8 import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip_fp8_configured
from kernel.naive import scaled_mm_naive


def test_config(cfg, M, N, K, device, with_scale=True, with_bias=True):
    """Test a specific config and return (pass, error_msg)."""
    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float8_e5m2)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float8_e5m2)

    scale = torch.tensor(2.34, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    b_prepacked = prepack_b_for_scaled_mm(b)

    try:
        out_hip = scaled_mm_hip_fp8_configured(a, b_prepacked, scale if with_scale else None, bias if with_bias else None, torch.float16, *cfg)
    except Exception as e:
        return False, f"LAUNCH ERROR: {e}"

    out_ref = scaled_mm_naive(a, b, scale if with_scale else None, bias if with_bias else None, torch.float16)

    diff = (out_hip.float() - out_ref.float()).abs()
    ref_abs = out_ref.float().abs()
    l2_rel = diff.norm() / ref_abs.norm().clamp_min(1e-6)
    max_diff = diff.max().item()

    if l2_rel.item() > 0.01 or max_diff > 1.0:
        return False, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"
    return True, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"


def main():
    device = "cuda"

    # Test matrix sizes - must be divisible by tile sizes for contiguous fast path.
    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (256, 256, 512),
        (256, 512, 256),
        (512, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    print(f"Testing fp8 HIP kernel ({len(_CONFIGS)} configs across {len(test_sizes)} matrix sizes)\n")
    print("Config format: (warps_m, warps_n, unroll_k, repeat_m, repeat_n)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    for cfg in _CONFIGS:
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        block_m = 16 * warps_m * repeat_m
        block_n = 16 * warps_n * repeat_n
        chunk_k = 16 * unroll_k

        config_passed = True
        config_errors = []

        for M, N, K in test_sizes:
            # Skip sizes that don't satisfy divisibility for this config
            if M % block_m != 0 or N % block_n != 0 or K % chunk_k != 0:
                continue

            passed, msg = test_config(cfg, M, N, K, device, with_scale=True, with_bias=True)
            if not passed:
                config_passed = False
                config_errors.append(f"  M={M} N={N} K={K}: {msg}")

        status = "PASS" if config_passed else "FAIL"
        print(f"[{status}] {cfg} BlockM={block_m} BlockN={block_n}")
        if not config_passed:
            for err in config_errors:
                print(err)
            failed_configs.append(cfg)
        else:
            passed_configs.append(cfg)

    print("=" * 80)
    print(f"\nSummary: {len(passed_configs)}/{len(_CONFIGS)} configs passed")

    if failed_configs:
        print(f"\nFailed configs ({len(failed_configs)}):")
        for cfg in failed_configs:
            print(f"  {cfg}")

    if passed_configs:
        print(f"\nPassed configs ({len(passed_configs)}):")
        for cfg in passed_configs:
            print(f"  {cfg}")


if __name__ == "__main__":
    main()
