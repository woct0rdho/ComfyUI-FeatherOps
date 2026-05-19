import os
from typing import Optional

import torch
from torch._inductor.kernel.custom_op import register_custom_op_autotuning

from .utils import generate_autotune_configs, get_compatible_config, load_hip_stable_extension, old_autotune

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hip_stable_extension("scaled_mm_hip_ext", cur_dir, "hip_kernel.cu")


@torch.library.custom_op("feather_ops_internal::scaled_mm_configured", mutates_args=())
def _configured_op(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int,
    block_warps_n: int,
    unroll_k: int,
    repeat_m: int,
    repeat_n: int,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b_prepacked.shape[1]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.scaled_mm.default(
        a,
        b_prepacked,
        scale,
        bias,
        out,
        block_warps_m,
        block_warps_n,
        unroll_k,
        repeat_m,
        repeat_n,
        0 if b_prepacked.dtype == torch.float8_e4m3fn else 1,
    )
    return out


@_configured_op.register_fake
def _(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    block_warps_m: int,
    block_warps_n: int,
    unroll_k: int,
    repeat_m: int,
    repeat_n: int,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b_prepacked.shape[1]), device=a.device, dtype=out_dtype)


scaled_mm_hip_configured = _configured_op

_CONFIGS = [
    (1, 1, 2, 1, 2),
    (1, 1, 4, 1, 2),
    (1, 2, 2, 1, 2),
    (1, 2, 4, 1, 2),
    (1, 4, 2, 1, 2),
    (1, 4, 4, 1, 2),
    (1, 8, 2, 1, 2),
    (1, 8, 4, 1, 2),
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
# TODO: Sort configs with a better heuristic to find the fastest one
_CONFIGS = sorted(_CONFIGS, key=lambda x: (x[0], x[1], x[3], x[4], x[2]))


# TODO: When torch custom op can handle multiple optional inputs, we should pass scale and bias as Optional[Tensor]
@torch.library.custom_op("feather_ops_internal::scaled_mm_autotuned", mutates_args=())
def _autotuned_op(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype,
    has_scale: bool,
    has_bias: bool,
    block_warps_m: int = 0,
    block_warps_n: int = 0,
    unroll_k: int = 0,
    repeat_m: int = 0,
    repeat_n: int = 0,
) -> torch.Tensor:
    if min(block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n) <= 0:
        block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n = get_compatible_config(a, b_prepacked, 1, _CONFIGS)
    return _configured_op(a, b_prepacked, scale if has_scale else None, bias if has_bias else None, out_dtype, block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n)


@_autotuned_op.register_fake
def _(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype,
    has_scale: bool,
    has_bias: bool,
    block_warps_m: int = 0,
    block_warps_n: int = 0,
    unroll_k: int = 0,
    repeat_m: int = 0,
    repeat_n: int = 0,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b_prepacked.shape[1]), device=a.device, dtype=out_dtype)


register_custom_op_autotuning(_autotuned_op, config_generator=lambda fake_tensors: generate_autotune_configs(fake_tensors, _CONFIGS, 1))


def scaled_mm_hip(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _autotuned_op(
            a,
            b_prepacked,
            scale if scale is not None else a,
            bias if bias is not None else a,
            out_dtype,
            scale is not None,
            bias is not None,
        )

    def run_fn(cfg):
        return _configured_op(a, b_prepacked, scale, bias, out_dtype, *cfg)

    best_cfg = old_autotune(
        a.shape[0],
        b_prepacked.shape[1],
        a.shape[1],
        _CONFIGS,
        run_fn,
        "mixed",
        b_prepacked.dtype,
    )
    return _configured_op(a, b_prepacked, scale, bias, out_dtype, *best_cfg)


def prepack_b_for_scaled_mm(b: torch.Tensor) -> torch.Tensor:
    return b.view(b.shape[0] // 16, 16, b.shape[1]).permute(0, 2, 1).contiguous()
