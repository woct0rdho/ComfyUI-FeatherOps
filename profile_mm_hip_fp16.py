#!/usr/bin/env python3

import argparse

import torch
from torch._inductor import config

from kernel.hip.hip_kernel_fp16 import mm_hip_fp16_configured, prepack_b_for_mm_fp16


def main():
    # Disable compile workers because they can keep profilers alive.
    torch._inductor.config.compile_threads = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("-N", type=int, default=8192, help="Matrix size (N x N)")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to profile")
    parser.add_argument("--config", type=str, default="1,8,4,8,2", help="Specific config to use, e.g. 1,8,4,8,2")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    m = n = k = args.N

    print(f"Allocating tensors (N={args.N})...")
    a = torch.randn((m, k), device=device, dtype=torch.float32).to(torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float32).to(torch.float16)
    b_prepacked = prepack_b_for_mm_fp16(b)
    bias = None if args.no_bias else torch.randn(n, device=device, dtype=torch.float16)

    cfg = tuple(int(x.strip()) for x in args.config.split(","))

    def run_fn():
        return mm_hip_fp16_configured(a, b_prepacked, bias, torch.float16, cfg)

    print("Warming up...")
    for _ in range(3):
        _ = run_fn()
    torch.cuda.synchronize()

    print(f"Profiling {args.iters} iterations...")
    for _ in range(args.iters):
        _ = run_fn()
    torch.cuda.synchronize()
    print("Done")


if __name__ == "__main__":
    main()
