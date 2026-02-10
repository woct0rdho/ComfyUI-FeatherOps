#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path

import torch

from kernel.hip.hip_kernel import scaled_mm_hip


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure comm/comp overlap for HIP scaled_mm kernel.")
    parser.add_argument("-N", type=int, default=8192, help="Matrix size (N x N)")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per mode")
    parser.add_argument(
        "--config",
        type=str,
        default="2,2,2,4,4,4",
        help="HIP_K0MK1_FORCE_CONFIG string (warps_m,warps_n,unroll_k,stages,repeat_m,repeat_n)",
    )
    parser.add_argument("--no-scale", action="store_true", help="Disable scale in kernel call")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias in kernel call")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="overlap_reports",
        help="Directory to write CSV summary",
    )
    args = parser.parse_args()

    os.environ["HIP_K0MK1_FORCE_CONFIG"] = args.config

    device = "cuda"
    n = args.N
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    scale = None if args.no_scale else torch.tensor(1.0, device=device, dtype=torch.float32)
    bias = None if args.no_bias else torch.randn((n,), device=device, dtype=torch.float16)

    modes = ["full", "no_overlap", "comm_only", "comp_only"]
    timings = {}

    for mode in modes:
        ms = bench_ms(
            lambda m=mode: scaled_mm_hip(a, b, scale, bias, out_dtype=torch.float16, mode=m),
            warmup=args.warmup,
            iters=args.iters,
        )
        timings[mode] = ms
        print(f"{mode:10s} {ms:.6f} ms")

    t_full = timings["full"]
    t_no_overlap = timings["no_overlap"]
    t_comm = timings["comm_only"]
    t_comp = timings["comp_only"]

    overlap_gain_ms = t_no_overlap - t_full
    overlap_by_decomp_ms = (t_comm + t_comp) - t_full
    denom = min(t_comm, t_comp)
    overlap_eff = 0.0 if denom <= 0 else overlap_by_decomp_ms / denom
    overlap_eff = max(0.0, min(1.0, overlap_eff))

    print(f"overlap_gain_ms      {overlap_gain_ms:.6f}")
    print(f"overlap_by_decomp_ms {overlap_by_decomp_ms:.6f}")
    print(f"overlap_eff          {overlap_eff:.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_slug = args.config.replace(",", "x")
    out_csv = out_dir / f"overlap_hip_N{n}_cfg_{config_slug}.csv"

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "config",
                "iters",
                "warmup",
                "with_scale",
                "with_bias",
                "full_ms",
                "no_overlap_ms",
                "comm_only_ms",
                "comp_only_ms",
                "overlap_gain_ms",
                "overlap_by_decomp_ms",
                "overlap_eff",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "N": n,
                "config": args.config,
                "iters": args.iters,
                "warmup": args.warmup,
                "with_scale": int(not args.no_scale),
                "with_bias": int(not args.no_bias),
                "full_ms": f"{t_full:.6f}",
                "no_overlap_ms": f"{t_no_overlap:.6f}",
                "comm_only_ms": f"{t_comm:.6f}",
                "comp_only_ms": f"{t_comp:.6f}",
                "overlap_gain_ms": f"{overlap_gain_ms:.6f}",
                "overlap_by_decomp_ms": f"{overlap_by_decomp_ms:.6f}",
                "overlap_eff": f"{overlap_eff:.6f}",
            }
        )

    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
