import os
import functools
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


@functools.cache
def _load_ck_extension():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    name = "scaled_mm_ck_ext"
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, "ck_kernel.cu")
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

    ck_root = os.path.expanduser("~/rocm-libraries/projects/composablekernel")
    includes = [
        os.path.join(cur_dir, "include"),
        os.path.join(ck_root, "include"),
        os.path.join(ck_root, "library", "include"),
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


def scaled_mm_ck(
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

    ext = _load_ck_extension()
    return ext.scaled_mm_ck(a, b, scale, bias, has_scale, has_bias)
