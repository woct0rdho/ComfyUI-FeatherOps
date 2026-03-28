#!/usr/bin/env python3

import argparse
import statistics
from typing import List, Optional, Tuple

import torch
import triton

from kernel.hip.hip_kernel import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip_configured


def _parse_shapes(text: str) -> List[Tuple[int, int, int]]:
    shapes = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid shape '{chunk}', expected 3 comma-separated ints: M,N,K")
        shapes.append(tuple(int(p) for p in parts))
    return shapes


def _parse_configs(text: str) -> List[Tuple[int, int, int, int, int]]:
    configs = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Invalid config '{chunk}', expected 5 comma-separated ints")
        configs.append(tuple(int(p) for p in parts))
    return configs


def _run_config(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    cfg: Tuple[int, int, int, int, int],
    rep: int,
    warmup: int,
) -> Tuple[float, float]:
    def run_kernel():
        scaled_mm_hip_configured(a, b_prepacked, scale, bias, torch.bfloat16, *cfg)

    timings_ms = triton.testing.do_bench(run_kernel, warmup=warmup, rep=rep, return_mode="all")
    avg_ms = statistics.mean(timings_ms)
    std_ms = statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return avg_ms, std_ms


def _format_cfg(cfg: Tuple[int, int, int, int, int]) -> str:
    return f"({cfg[0]},{cfg[1]},{cfg[2]},{cfg[3]},{cfg[4]})"


def _iter_tflops(m: int, n: int, k: int, avg_ms: float) -> float:
    return 2 * m * n * k / avg_ms * 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=str, default="8192,8192,8192", help="Semicolon-separated M,N,K tuples, e.g. '32,3072,3072;8192,12288,3072'")
    parser.add_argument("--configs", type=str, default="", help="Semicolon-separated configs, leave empty for all configs")
    parser.add_argument("--rep", type=int, default=1000, help="Repetition time (in ms) per config")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup time (in ms) per config")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    args = parser.parse_args()

    shapes = _parse_shapes(args.shapes)
    if args.configs:
        configs = _parse_configs(args.configs)
    else:
        configs = _CONFIGS

    device = "cuda"

    for m, n, k in shapes:
        print(f"\nShape M={m} N={n} K={k}")
        a = torch.randn(m, k, device=device, dtype=torch.float32).to(torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float32).to(torch.float8_e5m2)
        scale = None if args.no_scale else torch.tensor(2.34, device=device, dtype=torch.bfloat16)
        bias = None if args.no_bias else torch.randn(n, device=device, dtype=torch.bfloat16)

        b_prepacked = prepack_b_for_scaled_mm(b)

        for cfg in configs:
            avg_ms, std_ms = _run_config(
                a,
                b_prepacked,
                scale,
                bias,
                cfg,
                args.rep,
                args.warmup,
            )
            avg_tflops = _iter_tflops(m, n, k, avg_ms)
            print(f"  cfg={_format_cfg(cfg)} avg_ms={avg_ms:.3f} std_ms={std_ms:.3f} avg_TFLOPS={avg_tflops:.3f}")


if __name__ == "__main__":
    main()
