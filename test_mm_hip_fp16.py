#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel_fp16 import _CONFIGS, mm_hip_fp16_configured, prepack_b_for_mm_fp16
from kernel.naive import scaled_mm_naive


def test_config(cfg, M, N, K, device):
    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)
    out_dtype = torch.float16

    b_prepacked = prepack_b_for_mm_fp16(b)
    out_hip = mm_hip_fp16_configured(a, b_prepacked, bias, out_dtype, *cfg)

    out_ref = scaled_mm_naive(a, b, None, bias, out_dtype)

    out_hip = out_hip.float()
    out_ref = out_ref.float()
    diff = (out_hip - out_ref).abs()
    l2_rel_err = diff.norm() / out_ref.abs().norm().clamp_min(1e-6)
    l2_rel_err = l2_rel_err.item()
    max_abs_err = diff.max().item()

    atol_threshold = 8 if out_dtype == torch.bfloat16 else 1
    if l2_rel_err > 0.01 or max_abs_err > atol_threshold:
        return False, f"l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return True, f"l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}"


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

    print(f"Testing fp16 HIP kernel ({len(_CONFIGS)} configs across {len(test_sizes)} matrix sizes)\n")
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

            passed, msg = test_config(cfg, M, N, K, device)
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
