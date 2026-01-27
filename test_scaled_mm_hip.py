#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel import _load_hip_extension, _CONFIGS
from kernel.naive import scaled_mm_naive


def test_config(ext, cfg, M, N, K, device, with_scale=True, with_bias=False):
    """Test a specific config and return (pass, error_msg)."""
    warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = cfg

    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    scale = torch.tensor(1.7, device=device, dtype=torch.float32) if with_scale else torch.empty(0, device=device, dtype=torch.float32)
    bias = torch.randn((N,), device=device, dtype=torch.float16) if with_bias else torch.empty(0, device=device, dtype=torch.float16)

    try:
        out_hip = ext.scaled_mm_k0mk1(a, b, scale, bias, with_scale, with_bias, warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n)
    except Exception as e:
        return False, f"LAUNCH ERROR: {e}"

    # Compute reference
    out_ref = scaled_mm_naive(a, b, scale if with_scale else None, bias if with_bias else None, torch.float16)

    # Compare
    diff = (out_hip.float() - out_ref.float()).abs()
    ref_abs = out_ref.float().abs()
    l2_rel = diff.norm() / ref_abs.norm().clamp_min(1e-6)
    max_diff = diff.max().item()

    # Threshold for pass/fail
    if l2_rel.item() > 0.01 or max_diff > 1.0:
        return False, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"
    return True, f"rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}"


def main():
    device = "cuda"
    ext = _load_hip_extension()

    # Test matrix sizes
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (192, 128, 80),
        (256, 256, 256),
        (512, 512, 512),
    ]

    print(f"Testing {len(_CONFIGS)} configs across {len(test_sizes)} matrix sizes\n")
    print("Config format: (warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    for cfg in sorted(_CONFIGS):
        warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = cfg
        block_m = 16 * warps_m * repeat_m
        block_n = 16 * warps_n * repeat_n

        config_passed = True
        config_errors = []

        for M, N, K in test_sizes:
            passed, msg = test_config(ext, cfg, M, N, K, device, with_scale=True, with_bias=False)
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
