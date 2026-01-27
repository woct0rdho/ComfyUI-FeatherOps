import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load, _import_module_from_library


def get_rocm_lib_dirs() -> list[str]:
    rocm_lib_dirs = []
    for env_var in ("ROCM_HOME", "ROCM_PATH"):
        rocm_home = os.environ.get(env_var)
        if rocm_home:
            rocm_lib_dirs.append(os.path.join(rocm_home, "lib"))
            rocm_lib_dirs.append(os.path.join(rocm_home, "lib64"))
    for mod_name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
        try:
            mod = __import__(mod_name)
            mod_dir = os.path.dirname(mod.__file__)
            rocm_lib_dirs.append(os.path.join(mod_dir, "lib"))
        except Exception:
            continue
    return [d for d in rocm_lib_dirs if os.path.isdir(d)]


@lru_cache(maxsize=1)
def _load_hip_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    name = "scaled_mm_hip_ext"
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, "hip_kernel.cu")
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


# K0MK1 configs: (warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n)
_CONFIGS = []
# 128x128 tile matching CK's MT128x128x32 (various thread configs)
_CONFIGS.append((2, 2, 2, 2, 4, 4))  # BlockM=128, BlockN=128, 128 threads
_CONFIGS.append((4, 2, 2, 2, 2, 4))  # BlockM=128, BlockN=128, 256 threads
_CONFIGS.append((2, 4, 2, 2, 4, 2))  # BlockM=128, BlockN=128, 256 threads
_CONFIGS.append((4, 4, 2, 2, 2, 2))  # BlockM=128, BlockN=128, 512 threads
# Main configs: warps=(2,4), repeat=(4,4)
_CONFIGS.append((2, 4, 4, 4, 4, 4))
_CONFIGS.append((2, 4, 2, 2, 4, 4))
# Smaller tiles for small matrices
_CONFIGS.append((2, 4, 4, 4, 2, 2))
_CONFIGS.append((2, 4, 2, 2, 2, 2))
_CONFIGS.append((4, 4, 4, 4, 2, 2))
# Single stage for minimal LDS
_CONFIGS.append((2, 4, 1, 1, 4, 4))
# Large matrix optimized: 1x4 warps for higher N coverage
_CONFIGS.append((1, 4, 4, 4, 8, 4))
# Large matrix: 1x8 warps (matching original HIP best config)
_CONFIGS.append((1, 8, 4, 4, 8, 2))
# 2x2 warps with larger repeat
_CONFIGS.append((2, 2, 4, 4, 4, 8))
_CONFIGS.append((2, 2, 4, 4, 8, 4))
# Large matrix optimized: 4x4 warps for better load balancing
# _CONFIGS.append((4, 4, 4, 4, 4, 4))  # BlockM=256, BlockN=256 - LDS overflow
# 2x8 warps for wide N coverage
_CONFIGS.append((2, 8, 4, 4, 4, 2))  # BlockM=128, BlockN=256
# 4x2 warps for tall M coverage
_CONFIGS.append((4, 2, 4, 4, 4, 4))  # BlockM=256, BlockN=128
_CONFIGS = sorted(set(_CONFIGS))

_AUTOTUNE_CACHE = {}


def _select_config(a, b, scale, bias, has_scale, has_bias, ext):
    key = (a.shape, b.shape, a.stride(), b.stride(), has_scale, has_bias)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    best = _CONFIGS[0]
    best_ms = None

    def run(cfg):
        warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = cfg
        return ext.scaled_mm_k0mk1(
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

    for cfg in _CONFIGS:
        run(cfg)
    torch.cuda.synchronize()

    for cfg in _CONFIGS:
        start = time.perf_counter()
        for _ in range(5):
            run(cfg)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000.0 / 5.0
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = cfg

    _AUTOTUNE_CACHE[key] = best
    warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = best
    print(
        f"HIP K0MK1 autotune M={a.shape[0]} N={b.shape[1]} K={a.shape[1]} "
        f"warps=({warps_m},{warps_n}) unroll_k={unroll_k} "
        f"stages={stages} repeat=({repeat_m},{repeat_n}) "
        f"config={best}"
    )
    return warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n


def scaled_mm_hip(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Scaled matmul using K0MK1 LDS layout kernel (optimized for gfx11 wave32)."""
    assert a.is_cuda
    assert b.device == a.device
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == torch.float16
    assert b.dtype == torch.float8_e4m3fn
    assert out_dtype == torch.float16

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
        assert bias.numel() == b.shape[1]
        bias_tensor = bias.to(dtype=torch.float16)
        has_bias = True

    ext = _load_hip_extension()

    warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n = _select_config(a, b, scale_tensor, bias_tensor, has_scale, has_bias, ext)
    return ext.scaled_mm_k0mk1(
        a,
        b,
        scale_tensor,
        bias_tensor,
        has_scale,
        has_bias,
        warps_m,
        warps_n,
        unroll_k,
        stages,
        repeat_m,
        repeat_n,
    )
