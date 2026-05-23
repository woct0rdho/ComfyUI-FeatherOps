import os

import torch
from torch._inductor.kernel.custom_op import register_custom_op_autotuning

from kernel.hip.utils import load_hip_stable_extension

from .utils import CONFIGS as _CONFIGS
from .utils import generate_autotune_configs, get_compatible_config, old_autotune

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hip_stable_extension("attn_hip_ext", cur_dir, "hip_kernel.cu")


@torch.library.custom_op("feather_attn_internal::attn_fp16_fp8kv_configured", mutates_args=())
def _configured_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int,
    bc: int,
    n_waves: int,
) -> torch.Tensor:
    k_fp8 = torch.empty(k.shape, device=k.device, dtype=torch.float8_e5m2)
    v_fp8 = torch.empty(v.shape, device=v.device, dtype=torch.float8_e5m2)
    out = torch.empty_like(q)
    torch.ops.feather_attn_fp16.attn_fp16_fp8kv.default(q, k, v, k_fp8, v_fp8, out, br, bc, n_waves)
    return out


@_configured_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int,
    bc: int,
    n_waves: int,
) -> torch.Tensor:
    return torch.empty_like(q)


attn_hip_configured = _configured_op


@torch.library.custom_op("feather_attn_internal::attn_fp16_fp8kv_autotuned", mutates_args=())
def _autotuned_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int = 0,
    bc: int = 0,
    n_waves: int = 0,
) -> torch.Tensor:
    if min(br, bc, n_waves) <= 0:
        br, bc, n_waves = get_compatible_config(q, k, _CONFIGS)
    return _configured_op(q, k, v, br, bc, n_waves)


@_autotuned_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int = 0,
    bc: int = 0,
    n_waves: int = 0,
) -> torch.Tensor:
    return torch.empty_like(q)


register_custom_op_autotuning(_autotuned_op, config_generator=lambda fake_tensors: generate_autotune_configs(fake_tensors, _CONFIGS))


def attn_hip(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _autotuned_op(q, k, v)

    b, h, n, d = q.shape
    n_kv = k.shape[2]

    def run_fn(cfg):
        return _configured_op(q, k, v, *cfg)

    best_cfg = old_autotune(b, h, n, n_kv, d, _CONFIGS, run_fn, "attn_fp16_fp8kv")
    return _configured_op(q, k, v, *best_cfg)
