import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from .utils import get_rocm_lib_dirs

HIPBLASLT_ROOT = os.path.expanduser("~/rocm-libraries/projects/hipblaslt")
HIPBLASLT_INSTALLED_INCLUDE = os.path.join(os.environ.get("ROCM_PATH", ""), "include")
HIPBLASLT_INCLUDES = [
    HIPBLASLT_INSTALLED_INCLUDE,
]


def load_hipblaslt_stable_extension(name: str, cur_dir: str, source_filename: str):
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

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

    include_paths = [path for path in HIPBLASLT_INCLUDES if os.path.isdir(path)]
    missing_paths = [path for path in HIPBLASLT_INCLUDES if not os.path.isdir(path)]
    if missing_paths:
        raise RuntimeError(f"Missing hipBLASLt include paths: {missing_paths}")

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
        extra_include_paths=include_paths,
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
        is_python_module=False,
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hipblaslt_stable_extension("mm_hipblaslt_fp16_ext", cur_dir, "hipblaslt_kernel_fp16.cu")


def _canonicalize_scale(scale: Optional[torch.Tensor], n: int, device: torch.device) -> tuple[Optional[torch.Tensor], float]:
    if scale is None:
        return None, 1.0

    scale_t = scale.to(device=device)
    if scale_t.numel() == 1:
        return None, float(scale_t.item())

    if scale_t.dim() != 1 or scale_t.numel() != n:
        raise RuntimeError(f"scale must be None, a scalar, or a length-{n} vector")

    return scale_t.to(torch.float32).contiguous(), 1.0


def _canonicalize_bias(bias: Optional[torch.Tensor], n: int, device: torch.device, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
    if bias is None:
        return None
    bias_t = bias.to(device=device, dtype=out_dtype)
    if bias_t.dim() != 1 or bias_t.numel() != n:
        raise RuntimeError(f"bias must be None or a length-{n} vector")
    return bias_t.contiguous()


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
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise RuntimeError("a and b must have contiguous innermost dimensions")
    if a.stride(0) != a.shape[1] or b.stride(0) != b.shape[1]:
        raise RuntimeError("a and b must be row-major contiguous to stay on the fast hipBLASLt path")


def _validate_colmajor_inputs(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype):
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


@torch.library.custom_op("feather_ops_internal::mm_hipblaslt_fp16", mutates_args=())
def _op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    has_scale: bool,
    has_bias: bool,
    alpha_scalar: float,
    use_relu: bool,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    torch.ops.feather_ops.mm_hipblaslt_fp16.default(
        a,
        b,
        scale,
        bias,
        has_scale,
        has_bias,
        out,
        alpha_scalar,
        use_relu,
    )
    return out


@torch.library.custom_op("feather_ops_internal::mm_hipblaslt_fp16_colmajor", mutates_args=())
def _op_colmajor(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    has_scale: bool,
    has_bias: bool,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    out = torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=torch.float16)
    torch.ops.feather_ops.mm_hipblaslt_fp16_colmajor.default(
        a,
        b,
        scale,
        bias,
        has_scale,
        has_bias,
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
    has_scale: bool,
    has_bias: bool,
    alpha_scalar: float,
    use_relu: bool,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)


@_op_colmajor.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    has_scale: bool,
    has_bias: bool,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    return torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=torch.float16)


def mm_hipblaslt_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    use_relu: bool = False,
) -> torch.Tensor:
    _validate_inputs(a, b, out_dtype)
    scale_t, alpha_scalar = _canonicalize_scale(scale, b.shape[1], a.device)
    bias_t = _canonicalize_bias(bias, b.shape[1], a.device, out_dtype)
    return _op(
        a,
        b,
        scale_t,
        bias_t,
        scale_t is not None,
        bias_t is not None,
        alpha_scalar,
        use_relu,
    )


def mm_hipblaslt_fp16_relu(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return mm_hipblaslt_fp16(a, b, scale, bias, out_dtype, use_relu=True)


def to_col_major(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_strided(x.shape, (1, x.shape[0]), device=x.device, dtype=x.dtype)
    out.copy_(x)
    return out


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
    _validate_colmajor_inputs(a, b, out_dtype)
    scale_t, alpha_scalar = _canonicalize_scale(scale, a.shape[0], a.device)
    bias_t = _canonicalize_bias(bias, a.shape[0], a.device, out_dtype)
    return _op_colmajor(
        a,
        b,
        scale_t,
        bias_t,
        scale_t is not None,
        bias_t is not None,
        alpha_scalar,
        use_relu,
        solution_index,
    )

def benchmark_raw_buffers(
    dummy: torch.Tensor,
    m: int,
    n: int,
    k: int,
    warmup_iters: int = 10,
    iters: int = 100,
    solution_index: int = 1112,
    use_relu: bool = True,
) -> float:
    return torch.ops.feather_ops.benchmark_raw_buffers.default(dummy, m, n, k, warmup_iters, iters, solution_index, use_relu)
