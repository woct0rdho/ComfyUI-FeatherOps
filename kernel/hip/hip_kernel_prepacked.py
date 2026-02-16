import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load, _import_module_from_library

from .hip_kernel import _CONFIGS, _config_compatible, _get_forced_config, get_rocm_lib_dirs


_PREPACKED_CONFIGS = (
    (1, 8, 2, 2, 8, 2),
    *_CONFIGS,
)


@lru_cache(maxsize=1)
def _load_hip_prepacked_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    name = "scaled_mm_hip_prepacked_ext"
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, "hip_kernel_prepacked.cu")
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

    rocwmma_root = os.path.expanduser("~/rocm-libraries/projects/rocwmma")
    includes = [
        os.path.join(rocwmma_root, "library", "include"),
    ]

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


def _pick_config_for_dims(M: int, N: int, K: int):
    forced = _get_forced_config()
    if forced is not None:
        return forced
    for cfg in _PREPACKED_CONFIGS:
        if _config_compatible(cfg, M, N, K):
            return cfg
    raise RuntimeError(f"No compatible K0MK1 config for M={M} N={N} K={K}. Dimensions must be divisible by tile sizes.")


def prepack_b_for_scaled_mm_hip(b: torch.Tensor, *, swizzle: bool = False) -> torch.Tensor:
    assert b.is_cuda
    assert b.ndim == 2
    assert b.dtype == torch.float8_e4m3fn

    K, N = b.shape
    if K % 16 != 0:
        raise RuntimeError(f"K must be divisible by 16 for prepack, got K={K}")
    if N % 16 != 0:
        raise RuntimeError(f"N must be divisible by 16 for prepack layout, got N={N}")

    kt = K // 16
    packed = b.view(torch.uint8).view(kt, 16, N).permute(0, 2, 1).contiguous()
    if not swizzle:
        return packed

    groups = N // 16
    packed_g = packed.view(kt, groups, 16, 16)
    group_ids = torch.arange(groups, device=b.device, dtype=torch.int64).view(1, groups)
    shifts = (torch.arange(kt, device=b.device, dtype=torch.int64) & 7).view(kt, 1)
    swizzled_group_ids = group_ids ^ shifts
    gather_idx = swizzled_group_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
    return packed_g.gather(1, gather_idx).reshape(kt, N, 16).contiguous()


def scaled_mm_hip_prepacked(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    b_is_swizzled: bool = False,
) -> torch.Tensor:
    """Scaled matmul path using prepacked B layout [K/16, N, 16] as fp8 bytes."""
    assert a.is_cuda
    assert b_prepacked.device == a.device
    assert a.ndim == 2
    assert b_prepacked.ndim == 3
    assert a.dtype == torch.float16
    assert b_prepacked.dtype == torch.uint8
    assert out_dtype == torch.float16
    assert b_prepacked.shape[2] == 16

    M, K = a.shape
    K_tiles, N, kfrag = b_prepacked.shape
    assert kfrag == 16
    assert K_tiles * 16 == K

    if scale is None:
        scale_tensor = torch.empty(0, device=a.device, dtype=torch.float32)
        has_scale = False
    else:
        assert scale.device == a.device
        assert scale.numel() == 1
        scale_tensor = scale.to(dtype=torch.float32)
        has_scale = True

    if bias is None:
        bias_tensor = torch.empty(0, device=a.device, dtype=out_dtype)
        has_bias = False
    else:
        assert bias.device == a.device
        assert bias.numel() == N
        bias_tensor = bias.to(dtype=torch.float16)
        has_bias = True

    ext = _load_hip_prepacked_extension()
    warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = _pick_config_for_dims(M, N, K)

    return ext.scaled_mm_prepacked(
        a,
        b_prepacked,
        scale_tensor,
        bias_tensor,
        has_scale,
        has_bias,
        b_is_swizzled,
        warps_m,
        warps_n,
        unroll_k,
        stages,
        repeat_m,
        repeat_n,
    )
