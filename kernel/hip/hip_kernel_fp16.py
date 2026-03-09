import functools
import os
import time
from typing import Optional

import torch

from .utils import _config_compatible, _get_forced_config, load_hip_extension


@functools.cache
def _load_hip_fp16_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return load_hip_extension("mm_hip_fp16_ext", cur_dir, "hip_kernel_fp16.cu")


_FP16_CONFIGS = [
    (1, 1, 2, 2, 2),
    (1, 1, 2, 4, 4),
    (1, 1, 4, 2, 2),
    (1, 1, 4, 4, 4),
    (1, 1, 8, 2, 2),
    (1, 1, 8, 4, 4),
    (1, 2, 2, 2, 2),
    (1, 2, 4, 2, 2),
    (1, 2, 8, 2, 2),
    (1, 4, 2, 4, 2),
    (1, 4, 4, 4, 2),
    (1, 4, 8, 4, 2),
    (1, 8, 2, 8, 2),
    (1, 8, 4, 8, 2),
    (2, 1, 2, 2, 2),
    (2, 1, 4, 2, 2),
    (2, 1, 8, 2, 2),
    (2, 2, 2, 4, 4),
    (2, 2, 4, 4, 4),
    (2, 2, 8, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 2, 4, 4),
    (2, 4, 4, 4, 2),
    (2, 4, 4, 4, 4),
    (2, 4, 8, 4, 2),
    (4, 2, 2, 2, 4),
    (4, 2, 4, 2, 4),
    (4, 2, 8, 2, 4),
]
_FP16_AUTOTUNE_CACHE = {}


def _select_config_fp16(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
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
        has_bias,
    )
    cached = _FP16_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    M, K = a.shape
    N = b_prepacked.shape[2]

    candidates = [c for c in _FP16_CONFIGS if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible fp16 config for M={M} N={N} K={K}. Dimensions must be divisible by tile sizes.")

    warmup_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_WARMUP", "1")))
    bench_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_ITERS", "10")))
    best_cfg = candidates[0]
    best_ms = None

    def run(cfg):
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        return ext.mm_fp16(
            a,
            b_prepacked,
            bias,
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

    _FP16_AUTOTUNE_CACHE[key] = best_cfg
    wm, wn, uk, rm, rn = best_cfg
    print(f"HIP fp16 autotune M={M} N={N} K={K} warps=({wm},{wn}) unroll_k={uk} repeat=({rm},{rn}) time={best_ms:.3f} ms")
    return best_cfg


def prepack_b_for_mm_fp16(b: torch.Tensor) -> torch.Tensor:
    assert b.is_cuda
    assert b.ndim == 2
    assert b.dtype == torch.float16

    K, N = b.shape
    if K % 16 != 0:
        raise RuntimeError(f"K must be divisible by 16 for prepack, got K={K}")
    if N % 16 != 0:
        raise RuntimeError(f"N must be divisible by 16 for prepack layout, got N={N}")

    # Reshape to separate K and N components into K0(2) x K1(8) and
    # N_outer x N_inner_2(2) x N_inner_8(8).
    packed = b.view(K // 16, 2, 8, N // 16, 2, 8)

    # Permute to layout: [kt, k0, n_outer, n_inner_2, n_inner_8, k1].
    # This keeps logical N order while still storing each [k1=8] fragment contiguously
    # for 128-bit global and LDS transfers.
    packed = packed.permute(0, 1, 3, 4, 5, 2).contiguous()

    # Flatten the N components to form the logical N dimension directly.
    packed = packed.view(K // 16, 2, N, 8)
    return packed


def mm_fp16_prepacked(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.is_cuda
    assert b_prepacked.device == a.device
    assert a.ndim == 2
    assert b_prepacked.ndim == 4
    assert a.dtype == torch.float16
    assert b_prepacked.dtype == torch.float16
    assert out_dtype == torch.float16

    M, K = a.shape
    K_tiles, k0_dim, N, k1_dim = b_prepacked.shape
    assert k0_dim == 2
    assert k1_dim == 8
    assert K_tiles * 16 == K

    if bias is None:
        bias = torch.empty(0, device=a.device, dtype=out_dtype)
        has_bias = False
    else:
        assert bias.device == a.device
        assert bias.numel() == N
        bias = bias.to(out_dtype)
        has_bias = True

    ext = _load_hip_fp16_extension()
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = _select_config_fp16(
        a,
        b_prepacked,
        bias,
        has_bias,
        ext,
    )

    return ext.mm_fp16(
        a,
        b_prepacked,
        bias,
        has_bias,
        warps_m,
        warps_n,
        unroll_k,
        repeat_m,
        repeat_n,
    )
