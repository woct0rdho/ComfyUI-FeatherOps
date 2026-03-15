import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import _import_module_from_library, load

from .utils import get_rocm_lib_dirs


def load_hipblaslt_stable_extension(name: str, cur_dir: str, source_filename: str):
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, source_filename)
    ninja_log = os.path.join(build_dir, ".ninja_log")
    should_rebuild = False

    if os.path.exists(source_file) and os.path.exists(ninja_log):
        if os.path.getmtime(source_file) > os.path.getmtime(ninja_log):
            should_rebuild = True
    else:
        should_rebuild = True

    if not should_rebuild:
        try:
            return _import_module_from_library(name, build_dir, is_python_module=False)
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

    extra_cflags = [
        "-O3",
        "--std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-deprecated-copy-with-user-provided-copy",
        "-Wno-ignored-qualifiers",
        "-Wno-unused-parameter",
        "-DPy_LIMITED_API=0x03090000",
    ]

    extra_ldflags = ["-lhipblaslt"]
    for lib_dir in dict.fromkeys(get_rocm_lib_dirs()):
        extra_ldflags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

    load(
        name=name,
        sources=[os.path.join(cur_dir, source_filename)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cflags
        + [
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF2_OPERATORS__",
        ],
        extra_ldflags=extra_ldflags,
        extra_include_paths=includes,
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
        is_python_module=False,
    )
    Path(ninja_log).touch(exist_ok=True)


cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hipblaslt_stable_extension("mm_hipblaslt_fp16_ext", cur_dir, "hipblaslt_kernel_fp16.cu")


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype):
    if out_dtype != torch.float16:
        raise RuntimeError("hipBLASLt fp16 wrapper only supports out_dtype=torch.float16")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError("a and b must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise RuntimeError("a and b must be float16")
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError("a and b must be 2D")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(f"incompatible matmul shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if a.stride(0) != 1 or b.stride(0) != 1:
        raise RuntimeError("a and b must be column-major contiguous with stride(0) == 1")
    if a.stride(1) != a.shape[0] or b.stride(1) != b.shape[0]:
        raise RuntimeError("a and b must be column-major contiguous to match the 40 TFLOPS hipBLASLt path")


@torch.library.custom_op("feather_ops_internal::mm_hipblaslt_fp16_colmajor", mutates_args=())
def _op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    out = torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.mm_hipblaslt_fp16_colmajor.default(
        a,
        b,
        scale,
        bias,
        out,
        alpha_scalar,
        use_relu,
        solution_index,
    )
    return out


@_op.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    return torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=out_dtype)


def mm_hipblaslt_fp16_colmajor(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    use_relu: bool = False,
    solution_index: int = -1,
) -> torch.Tensor:
    _validate_inputs(a, b, out_dtype)
    alpha_scalar = 1.0
    if scale is not None and scale.numel() == 1:
        alpha_scalar = float(scale.item())
        scale = None

    return _op(
        a,
        b,
        scale,
        bias,
        out_dtype,
        alpha_scalar,
        use_relu,
        solution_index,
    )


def to_col_major(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_strided(x.shape, (1, x.shape[0]), device=x.device, dtype=x.dtype)
    out.copy_(x)
    return out
