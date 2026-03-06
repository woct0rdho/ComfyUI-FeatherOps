import functools
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import _import_module_from_library, load

from .hip_kernel import _config_compatible, _get_forced_config, get_rocm_lib_dirs


@functools.cache
def _load_hip_fp8_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    name = "scaled_mm_hip_fp8_ext"
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, "hip_kernel_fp8.cu")
    ninja_log = os.path.join(build_dir, ".ninja_log")
    should_rebuild = False

    if os.path.exists(source_file) and os.path.exists(ninja_log):
        if os.path.getmtime(source_file) > os.path.getmtime(ninja_log):
            should_rebuild = True

    if not should_rebuild:
        try:
            return _import_module_from_library(name, build_dir, is_python_module=True)
        except ImportError:
            pass

    includes = []

    try:
        import _rocm_sdk_core

        rocm_sdk_inc = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "include")
        if os.path.exists(rocm_sdk_inc):
            includes.append(rocm_sdk_inc)
    except ImportError:
        pass

    extra_ldflags = []
    for lib_dir in dict.fromkeys(get_rocm_lib_dirs()):
        extra_ldflags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

    module = load(
        name=name,
        sources=[source_file],
        extra_cflags=["-O3", "--std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "--std=c++20",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF2_OPERATORS__",
        ],
        extra_ldflags=extra_ldflags,
        extra_include_paths=includes,
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
    )
    Path(ninja_log).touch(exist_ok=True)
    return module


_FP8_CONFIGS = [
    (1, 8, 2, 8, 2),
    (2, 2, 2, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 2, 4, 4),
    (4, 2, 2, 2, 4),
]
_FP8_AUTOTUNE_CACHE = {}


def _select_config_fp8(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
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
        tuple(b_prepacked.shape),
        tuple(a.stride()),
        tuple(b_prepacked.stride()),
        has_scale,
        has_bias,
    )
    cached = _FP8_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    M, K = a.shape
    N = b_prepacked.shape[1]
    candidates = [c for c in _FP8_CONFIGS if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible fp8 config for M={M} N={N} K={K}. Dimensions must be divisible by tile sizes.")

    warmup_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_WARMUP", "1")))
    bench_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_ITERS", "10")))
    best_cfg = candidates[0]
    best_ms = None

    def run(cfg):
        warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
        return ext.scaled_mm_fp8(
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

    _FP8_AUTOTUNE_CACHE[key] = best_cfg
    wm, wn, uk, rm, rn = best_cfg
    print(f"HIP fp8 autotune M={M} N={N} K={K} warps=({wm},{wn}) unroll_k={uk} repeat=({rm},{rn}) time={best_ms:.3f} ms")
    return best_cfg


def prepack_b_for_scaled_mm_hip_fp8(b: torch.Tensor) -> torch.Tensor:
    assert b.is_cuda
    assert b.ndim == 2
    assert b.dtype == torch.float8_e5m2

    K, N = b.shape
    if K % 16 != 0:
        raise RuntimeError(f"K must be divisible by 16 for prepack, got K={K}")
    if N % 16 != 0:
        raise RuntimeError(f"N must be divisible by 16 for prepack layout, got N={N}")

    kt = K // 16
    packed = b.view(kt, 16, N).permute(0, 2, 1).contiguous()
    return packed


def scaled_mm_hip_fp8(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Scaled matmul path using A fp8e5m2 and prepacked B layout [K/16, N, 16] fp8e5m2."""
    assert a.is_cuda
    assert b_prepacked.device == a.device
    assert a.ndim == 2
    assert b_prepacked.ndim == 3
    assert a.dtype == torch.float8_e5m2
    assert b_prepacked.dtype == torch.float8_e5m2
    assert out_dtype == torch.float16
    assert b_prepacked.shape[2] == 16

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

    ext = _load_hip_fp8_extension()
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = _select_config_fp8(
        a,
        b_prepacked,
        scale,
        bias,
        has_scale,
        has_bias,
        ext,
    )

    return ext.scaled_mm_fp8(
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
    )
