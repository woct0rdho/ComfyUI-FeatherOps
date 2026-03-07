#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel_fp16 import _FP16_CONFIGS, _load_hip_fp16_extension, prepack_b_for_mm_fp16
from kernel.naive import scaled_mm_naive


def test_config(ext, cfg, M, N, K, device):
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg

    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    b_prepacked = prepack_b_for_mm_fp16(b)

    try:
        out_hip = ext.mm_fp16(
            a,
            b_prepacked,
            bias,
            True,
            warps_m,
            warps_n,
            unroll_k,
            repeat_m,
            repeat_n,
        )
    except Exception as e:
        return False, f"LAUNCH ERROR: {e}"

    # Compute reference
    out_ref = scaled_mm_naive(a, b, None, bias, torch.float16)

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
    ext = _load_hip_fp16_extension()

    # Test matrix sizes - must be divisible by tile sizes for contiguous fast path.
    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (256, 512, 256),
        (512, 512, 512),
    ]

    print(f"Testing fp16 prepacked HIP kernel ({len(_FP16_CONFIGS)} configs across {len(test_sizes)} matrix sizes)\n")
    print("Config format: (warps_m, warps_n, unroll_k, repeat_m, repeat_n)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    for cfg in sorted(_FP16_CONFIGS):
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

            passed, msg = test_config(ext, cfg, M, N, K, device)
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
    print(f"\nSummary: {len(passed_configs)}/{len(_FP16_CONFIGS)} configs passed")

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
