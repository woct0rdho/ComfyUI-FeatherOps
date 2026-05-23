#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from kernel_attn.hip.hip_kernel import _CONFIGS, attn_hip_configured


def test_config(cfg, B, H, N, N_KV, D, device):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

    k_e5m2 = k.to(torch.float8_e5m2).to(torch.float16)
    v_e5m2 = v.to(torch.float8_e5m2).to(torch.float16)
    out_ref_quant = F.scaled_dot_product_attention(q, k_e5m2, v_e5m2, attn_mask=None, dropout_p=0.0, is_causal=False)
    out_ref_fp16 = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    out_hip = attn_hip_configured(q, k, v, *cfg)

    out_hip_f = out_hip.float()
    out_ref_quant_f = out_ref_quant.float()
    out_ref_fp16_f = out_ref_fp16.float()

    diff_quant = out_hip_f - out_ref_quant_f
    quant_l2_rel_err = diff_quant.norm() / out_ref_quant_f.norm().clamp_min(1e-6)
    quant_l2_rel_err = quant_l2_rel_err.item()
    quant_max_abs_err = diff_quant.abs().max().item()

    diff_fp16 = out_hip_f - out_ref_fp16_f
    fp16_l2_rel_err = diff_fp16.norm() / out_ref_fp16_f.norm().clamp_min(1e-6)
    fp16_l2_rel_err = fp16_l2_rel_err.item()
    fp16_max_abs_err = diff_fp16.abs().max().item()

    if quant_l2_rel_err > 0.10 or quant_max_abs_err > 0.75:
        return False, f"quant_ref_l2={quant_l2_rel_err:.3g} quant_ref_max={quant_max_abs_err:.3g} fp16_ref_l2={fp16_l2_rel_err:.3g} fp16_ref_max={fp16_max_abs_err:.3g}"
    if fp16_l2_rel_err > 0.18 or fp16_max_abs_err > 0.75:
        return False, f"fp16_ref_l2={fp16_l2_rel_err:.3g} fp16_ref_max={fp16_max_abs_err:.3g} quant_ref_l2={quant_l2_rel_err:.3g} quant_ref_max={quant_max_abs_err:.3g}"

    return True, f"quant_ref_l2={quant_l2_rel_err:.3g} quant_ref_max={quant_max_abs_err:.3g} fp16_ref_l2={fp16_l2_rel_err:.3g} fp16_ref_max={fp16_max_abs_err:.3g}"


def main():
    device = "cuda"

    # Test tensor sizes (B, H, N, N_KV, D). Qwen-Image attention is typically H=24, N=4096, D=128.
    test_sizes = [
        (1, 1, 128, 128, 64),
        (2, 4, 128, 128, 128),
        (1, 24, 512, 512, 128),
        (1, 24, 4096, 4096, 128),
    ]

    print(f"Testing attention HIP fp16/fp8e5m2 KV kernel ({len(_CONFIGS)} configs across {len(test_sizes)} sizes)\n")
    print("Config format: (Br, Bc, N_WAVES)")
    print("=" * 80)

    failed_configs = []
    passed_configs = []

    for cfg in _CONFIGS:
        Br, Bc, N_WAVES = cfg

        config_passed = True
        config_errors = []

        for B, H, N, N_KV, D in test_sizes:
            if N % Br != 0 or N_KV % Bc != 0 or D % 32 != 0:
                continue

            passed, msg = test_config(cfg, B, H, N, N_KV, D, device)
            if not passed:
                config_passed = False
                config_errors.append(f"  B={B} H={H} N={N} N_KV={N_KV} D={D}: {msg}")

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
        raise SystemExit(1)

    if passed_configs:
        print(f"\nPassed configs ({len(passed_configs)}):")
        for cfg in passed_configs:
            print(f"  {cfg}")


if __name__ == "__main__":
    main()
