import functools
import os
import time
from typing import Optional

import torch

from .utils import _config_compatible, _get_forced_config, load_hip_extension


@functools.cache
def _load_hip_prepacked_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return load_hip_extension("scaled_mm_hip_prepacked_ext", cur_dir, "hip_kernel_prepacked.cu")


_PREPACKED_CONFIGS = [
    (1, 1, 2, 2, 2),
    (1, 1, 4, 2, 2),
    (1, 1, 2, 4, 4),
    (1, 1, 4, 4, 4),
    (1, 2, 2, 2, 2),
    (1, 2, 4, 2, 2),
    (2, 1, 2, 2, 2),
    (2, 1, 4, 2, 2),
    (1, 4, 2, 4, 2),
    (1, 4, 4, 4, 2),
    (1, 8, 2, 8, 2),
    (1, 8, 4, 8, 2),
    (2, 2, 2, 4, 4),
    (2, 2, 4, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 4, 4, 2),
    (2, 4, 2, 4, 4),
    (2, 4, 4, 4, 4),
    (4, 2, 2, 2, 4),
    (4, 2, 4, 2, 4),
]
_PREPACKED_AUTOTUNE_CACHE = {}


def _select_config_prepacked(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    has_scale: bool,
    has_bias: bool,
    b_dtype: int,
    ext,
):
    forced = _get_forced_config()
    if forced is not None:
        return forced

    key = (
        tuple(a.shape),
        tuple(b_prepacked.shape),
        tuple(a.stride()),
        tuple(b_prepacked.stride()),
        has_scale,
        has_bias,
        b_dtype,
    )
    cached = _PREPACKED_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    M, K = a.shape
    N = b_prepacked.shape[1]

    candidates = [c for c in _PREPACKED_CONFIGS if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible prepacked config for M={M} N={N} K={K}. Dimensions must be divisible by tile sizes.")

    warmup_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_WARMUP", "1")))
    bench_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_ITERS", "10")))
    best_cfg = candidates[0]
    best_ms = None

    def run(cfg):
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        return ext.scaled_mm_prepacked(
            a,
            b_prepacked,
            scale,
            bias,
            has_scale,
            has_bias,
            warps_m,
            warps_n,
            unroll_k,
            repeat_m,
            repeat_n,
            b_dtype,
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

    _PREPACKED_AUTOTUNE_CACHE[key] = best_cfg
    wm, wn, uk, rm, rn = best_cfg
    dtype_str = "fp8e5m2" if b_dtype == 1 else "fp8e4m3"
    print(f"HIP prepacked autotune M={M} N={N} K={K} dtype={dtype_str} warps=({wm},{wn}) unroll_k={uk} repeat=({rm},{rn}) time={best_ms:.3f} ms")
    return best_cfg


def prepack_b_for_scaled_mm_hip(b: torch.Tensor) -> torch.Tensor:
    assert b.is_cuda
    assert b.ndim == 2
    assert b.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}

    K, N = b.shape
    if K % 16 != 0:
        raise RuntimeError(f"K must be divisible by 16 for prepack, got K={K}")
    if N % 16 != 0:
        raise RuntimeError(f"N must be divisible by 16 for prepack layout, got N={N}")

    kt = K // 16
    packed = b.view(kt, 16, N).permute(0, 2, 1).contiguous()
    return packed


def scaled_mm_hip_prepacked(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Scaled matmul path using prepacked B layout [K/16, N, 16] with dtype fp8."""
    assert a.is_cuda
    assert b_prepacked.device == a.device
    assert a.ndim == 2
    assert b_prepacked.ndim == 3
    assert a.dtype == torch.float16
    assert b_prepacked.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
    assert out_dtype == torch.float16

    if b_prepacked.dtype == torch.float8_e4m3fn:
        b_dtype = 0
    elif b_prepacked.dtype == torch.float8_e5m2:
        b_dtype = 1
    else:
        raise RuntimeError(f"Unsupported b_prepacked.dtype {b_prepacked.dtype} ")

    M, K = a.shape
    K_tiles, N, kfrag = b_prepacked.shape
    assert kfrag == 16
    assert K_tiles * 16 == K

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
        assert bias.numel() == N
        bias = bias.to(out_dtype)
        has_bias = True

    ext = _load_hip_prepacked_extension()
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = _select_config_prepacked(
        a,
        b_prepacked,
        scale,
        bias,
        has_scale,
        has_bias,
        b_dtype,
        ext,
    )

    return ext.scaled_mm_prepacked(
        a,
        b_prepacked,
        scale,
        bias,
        has_scale,
        has_bias,
        warps_m,
        warps_n,
        unroll_k,
        repeat_m,
        repeat_n,
        b_dtype,
    )
