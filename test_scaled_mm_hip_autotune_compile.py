#!/usr/bin/env python3

import torch
from torch._inductor import config

from kernel.hip.hip_kernel import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip, scaled_mm_hip_configured
from kernel.naive import scaled_mm_naive


def _make_inputs(m, n, k, device):
    a = torch.randn((m, k), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor(2.34, device=device, dtype=torch.bfloat16)
    bias = torch.randn(n, device=device, dtype=torch.bfloat16)
    out_dtype = torch.bfloat16
    b_prepacked = prepack_b_for_scaled_mm(b)
    return a, b, b_prepacked, scale, bias, out_dtype


def _check_close(out, ref, label, out_dtype=torch.bfloat16):
    out = out.float()
    ref = ref.float()
    diff = (out - ref).abs()
    l2_rel_err = diff.norm() / ref.abs().norm().clamp_min(1e-6)
    l2_rel_err = l2_rel_err.item()
    max_abs_err = diff.max().item()

    atol_threshold = 8 if out_dtype == torch.bfloat16 else 1
    if l2_rel_err > 0.01 or max_abs_err > atol_threshold:
        raise RuntimeError(f"{label} failed: l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}")

    print(f"{label}: l2_rel_err={l2_rel_err:.3g} max_abs_err={max_abs_err:.3g}")


def test_eager_configured(device):
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs(512, 512, 512, device)
    cfg = _CONFIGS[0]
    out = scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *cfg)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _check_close(out, ref, f"eager configured {cfg}", out_dtype)


def test_torch_compile_configured(device):
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs(512, 512, 512, device)
    cfg = _CONFIGS[0]

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *cfg)

    out = compiled_fn(a, b_prepacked, scale, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _check_close(out, ref, f"torch.compile configured {cfg}", out_dtype)


def test_eager_autotuned(device):
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs(512, 512, 512, device)
    out = scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _check_close(out, ref, "eager autotuned", out_dtype)


def test_torch_compile_autotuned(device):
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs(512, 512, 512, device)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)

    out = compiled_fn(a, b_prepacked, scale, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _check_close(out, ref, "torch.compile autotuned", out_dtype)


def test_torch_compile_autotuned_no_scale(device):
    a, b, b_prepacked, _, bias, out_dtype = _make_inputs(512, 512, 512, device)
    scale = None

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, bias):
        return scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)

    out = compiled_fn(a, b_prepacked, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _check_close(out, ref, "torch.compile autotuned no scale", out_dtype)


def test_torch_compile_view_input_no_scale(device):
    x = torch.randn((2, 16, 3072), device=device, dtype=torch.bfloat16)
    b = torch.randn((3072, 3072), device=device, dtype=torch.float32).to(torch.float8_e5m2)
    b_prepacked = prepack_b_for_scaled_mm(b)
    bias = torch.randn(3072, device=device, dtype=torch.bfloat16)
    out_dtype = torch.bfloat16

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(x, b_prepacked, bias):
        x = x.to(torch.float16)
        x = x.view(-1, x.shape[-1])
        return scaled_mm_hip(x, b_prepacked, None, bias, out_dtype)

    out = compiled_fn(x, b_prepacked, bias)
    ref = scaled_mm_naive(x.to(torch.float16).view(-1, 3072), b, None, bias, out_dtype)
    _check_close(out, ref, "torch.compile view input no scale", out_dtype)


def main():
    device = "cuda"
    test_eager_configured(device)
    test_torch_compile_configured(device)
    test_eager_autotuned(device)
    test_torch_compile_autotuned(device)
    test_torch_compile_autotuned_no_scale(device)
    test_torch_compile_view_input_no_scale(device)
    print("Configured and autotuned torch.compile tests passed")


if __name__ == "__main__":
    main()
