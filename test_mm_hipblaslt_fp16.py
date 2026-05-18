#!/usr/bin/env python3

import torch

from kernel.hip.hipblaslt_kernel_fp16 import mm_hipblaslt_fp16
from kernel.naive import scaled_mm_naive


def row_major(shape, device):
    return torch.randn(shape, device=device, dtype=torch.float32).to(torch.float16).contiguous()


def col_major(shape, device):
    return torch.randn((shape[1], shape[0]), device=device, dtype=torch.float32).to(torch.float16).contiguous().T


def make_inputs(layout, M, N, K, device):
    a = row_major((M, K), device) if layout[0] == "T" else col_major((M, K), device)
    b = row_major((K, N), device) if layout[1] == "T" else col_major((K, N), device)
    return a, b


def test_layout(layout, M, N, K, device):
    a, b = make_inputs(layout, M, N, K, device)
    scale = torch.rand(M, device=device, dtype=torch.float32)
    bias = torch.randn(M, device=device, dtype=torch.float16)
    out_dtype = torch.float16

    out_hipblaslt = mm_hipblaslt_fp16(a, b, scale, bias, out_dtype)
    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype, bias_dim=0)

    out_hipblaslt = out_hipblaslt.float()
    out_ref = out_ref.float()
    diff = out_hipblaslt - out_ref
    fro_rel_err = torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(out_ref).clamp(min=1e-6)
    fro_rel_err = fro_rel_err.item()
    max_abs_err = diff.abs().max().item()

    if fro_rel_err > 0.01 or max_abs_err > 1:
        return False, f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"
    return True, f"fro_rel_err={fro_rel_err:.3g} max_abs_err={max_abs_err:.3g}"


def main():
    device = "cuda"
    layouts = ["TT", "TN", "NT", "NN"]
    test_sizes = [
        (128, 128, 128),
        (128, 256, 192),
        (256, 128, 192),
        (256, 384, 512),
        (512, 256, 384),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    print(f"Testing hipBLASLt fp16 kernel ({len(layouts)} layouts across {len(test_sizes)} matrix sizes)\n")
    print("Layouts are hipBLASLt conventions: T means row-major tensor stride, N means column-major tensor stride")
    print("=" * 80)

    failures = []

    for layout in layouts:
        layout_passed = True
        layout_errors = []

        for M, N, K in test_sizes:
            passed, msg = test_layout(layout, M, N, K, device)
            if not passed:
                layout_passed = False
                layout_errors.append(f"  M={M} N={N} K={K}: {msg}")

        status = "PASS" if layout_passed else "FAIL"
        print(f"[{status}] {layout}")
        if not layout_passed:
            for err in layout_errors:
                print(err)
            failures.append(layout)

    print("=" * 80)
    if failures:
        raise SystemExit(f"Failed layouts: {', '.join(failures)}")
    print("All hipBLASLt fp16 layout tests passed")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
