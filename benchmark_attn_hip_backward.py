#!/usr/bin/env python3

import argparse
import gc
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.bwd import (
    attention_backward_triton_impl,
)

from kernel_attn.hip.hip_kernel import feather_attn_backward

BATCH = 1
HEAD_DIM = 64
BENCHMARK_HEAD_COUNTS = (16, 32, 56)
SEQ_LENS = (4096, 8192, 16384)
BENCHMARK_SHAPES = tuple((heads, seq_len) for heads in BENCHMARK_HEAD_COUNTS for seq_len in SEQ_LENS)
LAYOUTS = ("HND", "NHD")
PROVIDERS = ("aiter", "feather")


@dataclass
class SavedState:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor
    scale: float
    layout: str


def _bshd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor.transpose(1, 2) if layout == "HND" else tensor


def _saved_state(heads: int, seq_len: int, seed: int, layout: str) -> SavedState:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (BATCH, heads, seq_len, HEAD_DIM) if layout == "HND" else (BATCH, seq_len, heads, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    dout = torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
    scale = HEAD_DIM**-0.5
    out_bshd, lse, _, _ = flash_attn_2.fwd(
        _bshd(q, layout),
        _bshd(k, layout),
        _bshd(v, layout),
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return SavedState(
        q=q,
        k=k,
        v=v,
        out=(out_bshd.transpose(1, 2).contiguous() if layout == "HND" else out_bshd.contiguous()),
        lse=lse.contiguous(),
        dout=dout,
        scale=scale,
        layout=layout,
    )


class Provider:
    def __init__(self, name: str, state: SavedState):
        self.name = name
        self.state = state
        self.dq = torch.empty_like(state.q)
        self.dk = torch.empty_like(state.k)
        self.dv = torch.empty_like(state.v)
        self.delta = torch.empty_like(state.lse)

    def launch(self) -> None:
        x = self.state
        if self.name == "feather":
            feather_attn_backward(
                x.q,
                x.k,
                x.v,
                x.out,
                x.lse,
                x.dout,
                sm_scale=x.scale,
                dq=self.dq,
                dk=self.dk,
                dv=self.dv,
                delta=self.delta,
                implementation="fused",
                layout=x.layout,
            )
            return
        if self.name != "aiter":
            raise ValueError(f"unknown provider: {self.name}")
        seq_len = x.q.shape[2] if x.layout == "HND" else x.q.shape[1]
        attention_backward_triton_impl(
            do=_bshd(x.dout, x.layout),
            q=_bshd(x.q, x.layout),
            k=_bshd(x.k, x.layout),
            v=_bshd(x.v, x.layout),
            o=_bshd(x.out, x.layout),
            softmax_lse=x.lse,
            dq=_bshd(self.dq, x.layout),
            dk=_bshd(self.dk, x.layout),
            dv=_bshd(self.dv, x.layout),
            delta=self.delta,
            sm_scale=x.scale,
            alibi_slopes=None,
            causal=False,
            layout="bshd",
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            dropout_p=0.0,
            philox_seed=None,
            philox_offset=None,
            use_exp2=True,
            mode="fused",
            window_size_left=-1,
            window_size_right=-1,
        )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timed_launch(provider: Provider) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    provider.launch()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


def _timing(samples: list[float], flops: float) -> dict[str, object]:
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "p20_ms": _percentile(samples, 0.2),
        "p80_ms": _percentile(samples, 0.8),
        "tflops_f7": flops / (median_ms * 1.0e9),
        "samples_ms": samples,
    }


def _run_shape(
    heads: int,
    seq_len: int,
    layout: str,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    state = _saved_state(heads, seq_len, seed, layout)
    providers = {name: Provider(name, state) for name in PROVIDERS}
    for provider in providers.values():
        provider.launch()
    torch.cuda.synchronize()
    for repeat in range(warmups):
        order = PROVIDERS if repeat % 2 == 0 else tuple(reversed(PROVIDERS))
        for name in order:
            providers[name].launch()
    torch.cuda.synchronize()

    samples = {name: [] for name in PROVIDERS}
    for repeat in range(repeats):
        order = PROVIDERS if repeat % 2 == 0 else tuple(reversed(PROVIDERS))
        for name in order:
            samples[name].append(_timed_launch(providers[name]))

    flops = 14.0 * BATCH * heads * seq_len * seq_len * HEAD_DIM
    timings = {name: _timing(samples[name], flops) for name in PROVIDERS}
    paired = [aiter_ms / feather_ms for aiter_ms, feather_ms in zip(samples["aiter"], samples["feather"])]
    return {
        "shape": {
            "batch": BATCH,
            "heads": heads,
            "seq_len": seq_len,
            "head_dim": HEAD_DIM,
            "layout": layout,
        },
        "seed": seed,
        "timings": timings,
        "feather_over_aiter": _percentile(paired, 0.5),
        "paired_ratio_p20": _percentile(paired, 0.2),
        "paired_ratio_p80": _percentile(paired, 0.8),
    }


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    if value == "default":
        return list(BENCHMARK_SHAPES)
    shapes = []
    for item in value.split(","):
        heads, seq_len = item.lower().split("x", maxsplit=1)
        shape = (int(heads), int(seq_len))
        if min(shape) <= 0:
            raise ValueError("head counts and sequence lengths must be positive")
        shapes.append(shape)
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _parse_layouts(value: str) -> list[str]:
    layouts = [layout.strip().upper() for layout in value.split(",")]
    if not layouts or any(layout not in LAYOUTS for layout in layouts):
        raise ValueError("layouts must contain only HND and/or NHD")
    return layouts


def _speedups(rows: list[dict[str, object]]) -> list[float]:
    ratios = []
    for row in rows:
        ratio = row["feather_over_aiter"]
        if not isinstance(ratio, float):
            raise TypeError("benchmark speedup must be a float")
        ratios.append(ratio)
    return ratios


def _write_result(
    path: Path,
    rows: list[dict[str, object]],
    *,
    warmups: int,
    repeats: int,
) -> None:
    ratios = _speedups(rows)
    result = {
        "matrix": {
            "batch": BATCH,
            "head_dim": HEAD_DIM,
            "head_counts": list(BENCHMARK_HEAD_COUNTS),
            "seq_lens": list(SEQ_LENS),
            "layouts": list(LAYOUTS),
            "flops": "14 * B * H * N^2 * D",
            "providers": list(PROVIDERS),
            "warmups": warmups,
            "repeats": repeats,
            "forward_in_timing": False,
            "outputs_preallocated": True,
        },
        "rows": rows,
        "geometric_mean_feather_over_aiter": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
        "feather_wins": sum(ratio > 1.0 for ratio in ratios),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="default",
        help="comma-separated HxN rows, or 'default' for the primary matrix",
    )
    parser.add_argument(
        "--layouts",
        default=",".join(LAYOUTS),
        help="comma-separated physical layouts (HND and/or NHD)",
    )
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=640117)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    shapes = _parse_shapes(args.shapes)
    layouts = _parse_layouts(args.layouts)
    benchmark_rows = [(heads, seq_len, layout) for layout in layouts for heads, seq_len in shapes]
    rows = []
    for index, (heads, seq_len, layout) in enumerate(benchmark_rows):
        print(
            f"BEGIN H={heads} N={seq_len} D={HEAD_DIM} layout={layout}",
            flush=True,
        )
        row = _run_shape(
            heads,
            seq_len,
            layout,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed + index,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.output is not None:
            _write_result(
                args.output,
                rows,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        gc.collect()
        torch.cuda.empty_cache()

    ratios = _speedups(rows)
    geometric_mean = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    print(f"Summary: Feather/AITER={geometric_mean:.6f}x wins={sum(ratio > 1.0 for ratio in ratios)}/{len(rows)}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
