import os
from typing import Optional

import torch
from torch._inductor.kernel.custom_op import register_custom_op_autotuning

from .utils import generate_autotune_configs, get_compatible_config, load_hip_stable_extension, old_autotune

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hip_stable_extension("mm_hip_fp16_ext", cur_dir, "hip_kernel_fp16.cu")


@torch.library.custom_op("feather_ops_internal::mm_fp16_configured", mutates_args=())
def _configured_op(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int,
    block_warps_n: int,
    unroll_k: int,
    repeat_m: int,
    repeat_n: int,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b_prepacked.shape[2]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.mm_fp16.default(
        a,
        b_prepacked,
        bias,
        out,
        block_warps_m,
        block_warps_n,
        unroll_k,
        repeat_m,
        repeat_n,
    )
    return out


@_configured_op.register_fake
def _(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int,
    block_warps_n: int,
    unroll_k: int,
    repeat_m: int,
    repeat_n: int,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b_prepacked.shape[2]), device=a.device, dtype=out_dtype)


def mm_hip_fp16_configured(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    config: tuple[int, int, int, int, int],
) -> torch.Tensor:
    return _configured_op(a, b_prepacked, bias, out_dtype, *config)


_CONFIGS = [
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


@torch.library.custom_op("feather_ops_internal::mm_fp16_autotuned", mutates_args=())
def _autotuned_op(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int = 0,
    block_warps_n: int = 0,
    unroll_k: int = 0,
    repeat_m: int = 0,
    repeat_n: int = 0,
) -> torch.Tensor:
    if min(block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n) <= 0:
        block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n = get_compatible_config(a, b_prepacked, 2, _CONFIGS)
    return _configured_op(a, b_prepacked, bias, out_dtype, block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n)


@_autotuned_op.register_fake
def _(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int = 0,
    block_warps_n: int = 0,
    unroll_k: int = 0,
    repeat_m: int = 0,
    repeat_n: int = 0,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b_prepacked.shape[2]), device=a.device, dtype=out_dtype)


register_custom_op_autotuning(_autotuned_op, config_generator=lambda fake_tensors: generate_autotune_configs(fake_tensors, _CONFIGS, 2))


def mm_hip_fp16(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _autotuned_op(a, b_prepacked, bias, out_dtype)

    def run_fn(cfg):
        return _configured_op(a, b_prepacked, bias, out_dtype, *cfg)

    best_cfg = old_autotune(
        a.shape[0],
        b_prepacked.shape[2],
        a.shape[1],
        _CONFIGS,
        run_fn,
        "fp16",
    )
    return _configured_op(a, b_prepacked, bias, out_dtype, *best_cfg)


def prepack_b_for_mm_fp16(b: torch.Tensor) -> torch.Tensor:
    return b.view(b.shape[0] // 16, 2, 8, b.shape[1] // 16, 2, 8).permute(0, 1, 3, 4, 5, 2).contiguous().view(b.shape[0] // 16, 2, b.shape[1], 8)
