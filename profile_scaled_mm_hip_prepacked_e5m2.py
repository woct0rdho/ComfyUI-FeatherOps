#!/usr/bin/env python3

import argparse

import torch

from kernel.hip.hip_kernel_prepacked import prepack_b_for_scaled_mm_hip, scaled_mm_hip_prepacked


def main():
    # Disable compile workers because they can keep profilers alive.
    torch._inductor.config.compile_threads = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("-N", type=int, default=8192, help="Matrix size (N x N)")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to profile")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    m = n = k = args.N

    print(f"Allocating tensors (N={args.N})...")
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16).to(torch.float8_e5m2)
    b_prepacked = prepack_b_for_scaled_mm_hip(b)
    scale = None if args.no_scale else torch.tensor(2.34, device=device, dtype=torch.float16)
    bias = None if args.no_bias else torch.randn(n, device=device, dtype=torch.float16)

    print("Warming up...")
    for _ in range(3):
        _ = scaled_mm_hip_prepacked(a, b_prepacked, scale, bias, out_dtype=torch.float16)
    torch.cuda.synchronize()

    print(f"Profiling {args.iters} iterations...")
    for _ in range(args.iters):
        _ = scaled_mm_hip_prepacked(a, b_prepacked, scale, bias, out_dtype=torch.float16)
    torch.cuda.synchronize()
    print("Done")


if __name__ == "__main__":
    main()
