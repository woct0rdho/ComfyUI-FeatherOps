#!/usr/bin/env python3

import torch
from torch._inductor import config

from kernel.hip.hip_kernel_prepacked import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip_prepacked, scaled_mm_hip_prepacked_configured
from kernel.naive import scaled_mm_naive


def _make_inputs(m, n, k, device):
    a = torch.randn((m, k), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor(2.34, device=device, dtype=torch.float16)
    bias = torch.randn(n, device=device, dtype=torch.float16)
    b_prepacked = prepack_b_for_scaled_mm(b)
    return a, b, b_prepacked, scale, bias


def _check_close(out, ref, label):
    diff = (out.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    l2_rel = diff.norm() / ref_abs.norm().clamp_min(1e-6)
    max_diff = diff.max().item()

    if l2_rel.item() > 0.01 or max_diff > 1.0:
        raise RuntimeError(f"{label} failed: rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}")

    print(f"{label}: rel_l2={l2_rel.item():.3g} max_atol={max_diff:.3g}")


def test_eager_default(device):
    a, b, b_prepacked, scale, bias = _make_inputs(512, 512, 512, device)
    out = scaled_mm_hip_prepacked(a, b_prepacked, scale, bias, torch.float16)
    ref = scaled_mm_naive(a, b, scale, bias, torch.float16)
    _check_close(out, ref, "eager autotune")


def test_torch_compile_autotune(device):
    a, b, b_prepacked, scale, bias = _make_inputs(512, 512, 512, device)

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip_prepacked(a, b_prepacked, scale, bias, torch.float16)

    torch._dynamo.reset()
    with config.patch(
        max_autotune=True,
        benchmark_kernel=True,
        fx_graph_cache=False,
    ):
        out = compiled_fn(a, b_prepacked, scale, bias)

    ref = scaled_mm_naive(a, b, scale, bias, torch.float16)
    _check_close(out, ref, "torch.compile autotune")


def test_eager_configured(device):
    a, b, b_prepacked, scale, bias = _make_inputs(512, 512, 512, device)
    cfg = _CONFIGS[0]
    out = scaled_mm_hip_prepacked_configured(a, b_prepacked, scale, bias, torch.float16, cfg)
    ref = scaled_mm_naive(a, b, scale, bias, torch.float16)
    _check_close(out, ref, f"eager configured {cfg}")


def test_torch_compile_configured(device):
    a, b, b_prepacked, scale, bias = _make_inputs(512, 512, 512, device)
    cfg = _CONFIGS[0]

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip_prepacked_configured(a, b_prepacked, scale, bias, torch.float16, cfg)

    torch._dynamo.reset()
    with config.patch(
        max_autotune=True,
        benchmark_kernel=True,
        fx_graph_cache=False,
    ):
        out = compiled_fn(a, b_prepacked, scale, bias)

    ref = scaled_mm_naive(a, b, scale, bias, torch.float16)
    _check_close(out, ref, f"torch.compile configured {cfg}")


def main():
    device = "cuda"
    test_eager_default(device)
    test_torch_compile_autotune(device)
    test_eager_configured(device)
    test_torch_compile_configured(device)
    print("Configured and autotuned torch.compile tests passed")


if __name__ == "__main__":
    main()
