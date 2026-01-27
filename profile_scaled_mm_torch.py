#!/usr/bin/env python3

import argparse

import torch

from kernel.naive import scaled_mm_naive

scaled_mm_naive_compiled = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")


def main():
    torch._inductor.config.compile_threads = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=8192, help="Matrix size (N x N)")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to profile")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    m = n = k = args.N

    print(f"Allocating tensors (N={args.N})...")
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    bias = torch.randn(n, device=device, dtype=torch.float16)

    print("Warming up (compilation)...")
    for _ in range(3):
        _ = scaled_mm_naive_compiled(a, b, scale, bias, torch.float16)
    torch.cuda.synchronize()

    print(f"Profiling {args.iters} iterations...")
    for i in range(args.iters):
        _ = scaled_mm_naive_compiled(a, b, scale, bias, torch.float16)
    torch.cuda.synchronize()
    print("Done")


if __name__ == "__main__":
    main()
