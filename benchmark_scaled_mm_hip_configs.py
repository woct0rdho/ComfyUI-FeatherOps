#!/usr/bin/env python3

import argparse
from typing import List, Tuple

import torch
import triton

from kernel.hip import hip_kernel


def _parse_sizes(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_shapes(text: str) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid shape '{chunk}', expected 3 comma-separated ints: M,N,K")
        m, n, k = (int(p) for p in parts)
        shapes.append((m, n, k))
    return shapes


def _parse_configs(text: str) -> List[Tuple[int, int, int, int, int, int]]:
    configs = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 6:
            raise ValueError(f"Invalid config '{chunk}', expected 6 comma-separated ints")
        configs.append(tuple(int(p) for p in parts))
    return configs


def _get_configs(args: argparse.Namespace) -> List[Tuple[int, int, int, int, int, int]]:
    configs: List[Tuple[int, int, int, int, int, int]] = []
    if args.use_default_configs:
        configs.extend(hip_kernel._CONFIGS)
    if args.configs:
        configs.extend(_parse_configs(args.configs))
    if not configs:
        raise ValueError("No configs specified. Use --use-default-configs or --configs.")
    return sorted(set(configs))


def _run_config(
    ext,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    has_scale: bool,
    has_bias: bool,
    cfg: Tuple[int, int, int, int, int, int],
    rep: int,
    warmup: int,
) -> float:
    warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = cfg

    def run_kernel():
        ext.scaled_mm_k0mk1(
            a,
            b,
            scale,
            bias,
            has_scale,
            has_bias,
            warps_m,
            warps_n,
            unroll_k,
            stages,
            repeat_m,
            repeat_n,
        )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(run_kernel, warmup=warmup, rep=rep, quantiles=quantiles)
    return ms / 1000.0


def _format_cfg(cfg: Tuple[int, int, int, int, int, int]) -> str:
    return f"({cfg[0]},{cfg[1]},{cfg[2]},{cfg[3]},{cfg[4]},{cfg[5]})"


def _iter_gflops(m: int, n: int, k: int, seconds: float) -> float:
    return (2.0 * m * n * k) / seconds / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=str,
        default="4096,8192",
        help="Comma-separated N values for square GEMM (M=N=K). Ignored when --shapes is set.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Semicolon-separated M,N,K tuples, e.g. '4096,8192,4096;8192,4096,8192'",
    )
    parser.add_argument("--configs", type=str, default="", help="Semicolon-separated configs")
    parser.add_argument("--use-default-configs", action="store_true", help="Include hip_kernel._CONFIGS")
    parser.add_argument("--rep", type=int, default=1000, help="Repetition time (in ms) per config")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup time (in ms) per config")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    args = parser.parse_args()

    if args.shapes:
        shapes = _parse_shapes(args.shapes)
    else:
        sizes = _parse_sizes(args.sizes)
        shapes = [(n, n, n) for n in sizes]
    configs = _get_configs(args)

    torch.manual_seed(0)
    device = "cuda"
    ext = hip_kernel._load_hip_extension()

    for m, n, k in shapes:
        print(f"\nShape M={m} N={n} K={k}")
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        bias = torch.randn(n, device=device, dtype=torch.float16)
        has_scale = not args.no_scale
        has_bias = not args.no_bias

        if not has_scale:
            scale = torch.empty(0, device=device, dtype=torch.float32)
        if not has_bias:
            bias = torch.empty(0, device=device, dtype=torch.float16)

        for cfg in configs:
            seconds = _run_config(
                ext,
                a,
                b,
                scale,
                bias,
                has_scale,
                has_bias,
                cfg,
                args.rep,
                args.warmup,
            )
            gflops = _iter_gflops(m, n, k, seconds)
            print(f"  cfg={_format_cfg(cfg)}  {gflops:.3f} GFLOPS  {seconds * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
