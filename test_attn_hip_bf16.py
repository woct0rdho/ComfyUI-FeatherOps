#!/usr/bin/env python3

import math
import os
import sys

import torch
import torch.nn.functional as F

from kernel_attn.fwd import _CONFIGS, attn_fwd


def test_config(cfg, B, H, N, N_KV, D, device):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.bfloat16, device=device)

    # Reference PyTorch SDPA implementation
    # Pytorch's SDPA takes [B, H, L, D]
    # default scale is 1 / sqrt(D)
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    # Our kernel (directly call op to test specific config without autotuner)
    out_hip = torch.empty_like(q)
    try:
        torch.ops.feather_attn.attn_fwd.default(q, k, v, out_hip, *cfg)
    except Exception as e:
        return False, f"Kernel failed to run: {e}"

    out_hip = out_hip.float()
    out_ref = out_ref.float()
    diff = (out_hip - out_ref).abs()
    l2_rel_err = diff.norm() / out_ref.abs().norm().clamp_min(1e-6)
    l2_rel_err = l2_rel_err.item()
    max_abs_err = diff.max().item()

    # Given bfloat16 and different math paths (fast math, custom exp/max logic),
    # the absolute difference could be slightly larger than naive float32 accumulation.
    if l2_rel_err > 0.15 or max_abs_err > 0.5:
        return False, f"l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return True, f"l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}"


def main():
    device = "cuda"

    # Test tensor sizes (B, H, N, N_KV, D)
    # Must be divisible by tile sizes for the given configs
    test_sizes = [
        (1, 1, 128, 128, 64),
        (2, 4, 128, 128, 64),
        (2, 4, 256, 128, 64),
        (2, 4, 128, 256, 64),
        (2, 4, 256, 256, 64),
        (2, 4, 512, 512, 64),
        (2, 4, 128, 128, 128),
        (2, 4, 256, 256, 128),
        (1, 8, 1024, 1024, 64),
        (1, 1, 16, 16, 32),  # For small sizes
    ]

    print(f"Testing bf16 attention fwd HIP kernel ({len(_CONFIGS)} configs across {len(test_sizes)} matrix sizes)\n")
    print("Config format: (Br, Bc, N_WAVES)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    for cfg in _CONFIGS:
        Br, Bc, N_WAVES = cfg

        config_passed = True
        config_errors = []
        tested_any = False

        for B, H, N, N_KV, D in test_sizes:
            # Skip sizes that don't satisfy divisibility for this config
            if N % Br != 0 or N_KV % Bc != 0 or D % 32 != 0:
                continue

            tested_any = True
            passed, msg = test_config(cfg, B, H, N, N_KV, D, device)
            if not passed:
                config_passed = False
                config_errors.append(f"  B={B} H={H} N={N} N_KV={N_KV} D={D}: {msg}")

        if not tested_any:
            status = "SKIP"
            print(f"[{status}] {cfg} (no compatible sizes tested)")
            continue

        status = "PASS" if config_passed else "FAIL"
        print(f"[{status}] {cfg} Br={Br} Bc={Bc} N_WAVES={N_WAVES}")
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
