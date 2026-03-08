import functools
import os
import time
from typing import Optional

import torch

from .utils import _config_compatible, _get_forced_config, load_hip_extension


@functools.cache
def _load_hip_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return load_hip_extension("scaled_mm_hip_ext", cur_dir, "hip_kernel.cu")


# Autotune configs: (warps_m, warps_n, unroll_k, repeat_m, repeat_n)
_CONFIGS = [
    (2, 2, 2, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 2, 4, 4),
    (4, 2, 2, 2, 4),
]
_AUTOTUNE_CACHE = {}


def _select_config(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    has_scale: bool,
    has_bias: bool,
    ext,
):
    forced = _get_forced_config()
    if forced is not None:
        return forced

    key = (
        tuple(a.shape),
        tuple(b.shape),
        tuple(a.stride()),
        tuple(b.stride()),
        has_scale,
        has_bias,
    )
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    M, K = a.shape
    N = b.shape[1]
    candidates = [c for c in _CONFIGS if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible config for M={M} N={N} K={K}. Dimensions must be divisible by tile sizes.")

    warmup_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_WARMUP", "1")))
    bench_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_ITERS", "10")))
    best_cfg = candidates[0]
    best_ms = None

    def run(cfg):
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        return ext.scaled_mm(
            a,
            b,
            scale,
            bias,
            has_scale,
            has_bias,
            warps_m,
            warps_n,
            unroll_k,
            repeat_m,
            repeat_n,
        )

    # Warm up all candidates once to compile/load kernels.
    for cfg in candidates:
        for _ in range(warmup_iters):
            run(cfg)
    torch.cuda.synchronize()

    # Time each candidate.
    for cfg in candidates:
        start = time.perf_counter()
        for _ in range(bench_iters):
            run(cfg)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / bench_iters
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    _AUTOTUNE_CACHE[key] = best_cfg
    wm, wn, uk, rm, rn = best_cfg
    print(f"HIP autotune M={M} N={N} K={K} warps=({wm},{wn}) unroll_k={uk} repeat=({rm},{rn}) time={best_ms:.3f} ms")
    return best_cfg


def scaled_mm_hip(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.is_cuda
    assert b.device == a.device
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == torch.float16
    assert b.dtype == torch.float8_e4m3fn
    assert out_dtype == torch.float16

    if scale is None:
        scale = torch.empty(0, device=a.device, dtype=out_dtype)
        has_scale = False
    else:
        assert scale.device == a.device
        assert scale.numel() == 1
        scale = scale.to(out_dtype)
        has_scale = True

    if bias is None:
        bias = torch.empty(0, device=a.device, dtype=out_dtype)
        has_bias = False
    else:
        assert bias.device == a.device
        assert bias.numel() == b.shape[1]
        bias = bias.to(out_dtype)
        has_bias = True

    ext = _load_hip_extension()

    warps_m, warps_n, unroll_k, repeat_m, repeat_n = _select_config(
        a,
        b,
        scale,
        bias,
        has_scale,
        has_bias,
        ext,
    )
    return ext.scaled_mm(
        a,
        b,
        scale,
        bias,
        has_scale,
        has_bias,
        warps_m,
        warps_n,
        unroll_k,
        repeat_m,
        repeat_n,
    )
