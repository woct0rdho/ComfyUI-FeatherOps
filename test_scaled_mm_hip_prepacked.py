#!/usr/bin/env python3

import torch

from kernel.hip.hip_kernel_prepacked import _PREPACKED_CONFIGS, _load_hip_prepacked_extension, prepack_b_for_scaled_mm_hip, scaled_mm_hip_prepacked
from kernel.naive import scaled_mm_naive


def test_one(M, N, K, swizzle: bool, with_scale: bool = True, with_bias: bool = True):
    device = "cuda"
    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    scale = torch.tensor(1.7, device=device, dtype=torch.float32) if with_scale else None
    bias = torch.randn((N,), device=device, dtype=torch.float16) if with_bias else None

    b_prepacked = prepack_b_for_scaled_mm_hip(b, swizzle=swizzle)
    out_new = scaled_mm_hip_prepacked(
        a,
        b_prepacked,
        scale,
        bias,
        torch.float16,
        b_is_swizzled=swizzle,
    )
    out_ref = scaled_mm_naive(a, b, scale, bias, torch.float16)

    diff = (out_new.float() - out_ref.float()).abs()
    ref_abs = out_ref.float().abs()
    l2_rel = (diff.norm() / ref_abs.norm().clamp_min(1e-6)).item()
    max_diff = diff.max().item()
    ok = l2_rel <= 0.01 and max_diff <= 1.0
    return ok, l2_rel, max_diff


def main():
    _load_hip_prepacked_extension()

    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (256, 512, 256),
        (512, 512, 512),
    ]

    print("Testing prepacked HIP kernel correctness")
    print("Config format: (warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n)")
    print("=" * 80)

    failed = []
    passed = 0

    for cfg in sorted(_PREPACKED_CONFIGS):
        warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = cfg
        block_m = 16 * warps_m * repeat_m
        block_n = 16 * warps_n * repeat_n
        chunk_k = 16 * unroll_k

        cfg_ok = True
        for M, N, K in test_sizes:
            if M % block_m != 0 or N % block_n != 0 or K % chunk_k != 0:
                continue
            for swizzle in (False, True):
                ok, l2_rel, max_diff = test_one(M, N, K, swizzle=swizzle, with_scale=True, with_bias=True)
                if not ok:
                    cfg_ok = False
                    failed.append((cfg, M, N, K, swizzle, l2_rel, max_diff))

        status = "PASS" if cfg_ok else "FAIL"
        print(f"[{status}] {cfg} BlockM={block_m} BlockN={block_n}")
        if cfg_ok:
            passed += 1

    print("=" * 80)
    print(f"Summary: {passed}/{len(_PREPACKED_CONFIGS)} configs passed")
    if failed:
        print("Failures:")
        for cfg, M, N, K, swizzle, l2_rel, max_diff in failed:
            print(f"  cfg={cfg} M={M} N={N} K={K} swizzle={swizzle} rel_l2={l2_rel:.3g} max_atol={max_diff:.3g}")


if __name__ == "__main__":
    main()
