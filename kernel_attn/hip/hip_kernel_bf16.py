import os
from typing import Callable, Dict, List, Tuple

import torch
import triton
from torch._inductor.kernel.custom_op import CustomOpConfig, register_custom_op_autotuning
from torch.fx.experimental.symbolic_shapes import optimization_hint

from kernel.hip.utils import load_hip_stable_extension

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hip_stable_extension("attn_hip_bf16_ext", cur_dir, "hip_kernel_bf16.cu")

_CONFIGS = [
    (64, 64, 16),
    (64, 64, 8),
    (64, 128, 16),
    (128, 64, 16),
    (32, 64, 8),
    (64, 32, 8),
    (32, 32, 8),
    (16, 32, 2),
]


@torch.library.custom_op("feather_attn_internal::attn_bf16_configured", mutates_args=())
def _configured_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int,
    bc: int,
    n_waves: int,
) -> torch.Tensor:
    out = torch.empty_like(q)
    torch.ops.feather_attn.attn_bf16.default(q, k, v, out, br, bc, n_waves)
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


attn_hip_bf16_configured = _configured_op


def _config_compatible(cfg, n, n_kv, d):
    Br, Bc, _ = cfg
    return (n % Br == 0) and (n_kv % Bc == 0) and (d % 32 == 0)


def _size_hint(value: int) -> int:
    try:
        return int(value)
    except TypeError:
        return optimization_hint(value)


def _get_compatible_config(q: torch.Tensor, k: torch.Tensor, configs: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    n = _size_hint(q.shape[2])
    n_kv = _size_hint(k.shape[2])
    d = _size_hint(q.shape[3])
    for cfg in configs:
        if _config_compatible(cfg, n, n_kv, d):
            return cfg
    raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")


def _generate_autotune_configs(fake_tensors: Dict[str, torch.Tensor], configs: List[Tuple[int, int, int]]) -> List[CustomOpConfig]:
    q = fake_tensors["q"]
    k = fake_tensors["k"]
    n = _size_hint(q.shape[2])
    n_kv = _size_hint(k.shape[2])
    d = _size_hint(q.shape[3])
    compatible = [cfg for cfg in configs if _config_compatible(cfg, n, n_kv, d)]
    if not compatible:
        raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")

    return [CustomOpConfig(br=cfg[0], bc=cfg[1], n_waves=cfg[2]) for cfg in compatible]


@torch.library.custom_op("feather_attn_internal::attn_bf16_autotuned", mutates_args=())
def _autotuned_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    br: int = 0,
    bc: int = 0,
    n_waves: int = 0,
) -> torch.Tensor:
    if min(br, bc, n_waves) <= 0:
        br, bc, n_waves = _get_compatible_config(q, k, _CONFIGS)
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


register_custom_op_autotuning(_autotuned_op, config_generator=lambda fake_tensors: _generate_autotune_configs(fake_tensors, _CONFIGS))


_AUTOTUNE_CACHE = {}


def _old_autotune(
    b: int,
    h: int,
    n: int,
    n_kv: int,
    d: int,
    configs: List[Tuple[int, int, int]],
    run_fn: Callable[[Tuple[int, int, int]], torch.Tensor],
) -> Tuple[int, int, int]:
    key = (b, h, n, n_kv, d)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = [c for c in configs if _config_compatible(c, n, n_kv, d)]
    if not candidates:
        raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")

    # Same defaults as triton.autotune.
    warmup_ms = max(1, int(os.environ.get("AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("AUTOTUNE_REP_MS", "100")))
    best_cfg = candidates[0]
    best_ms = None

    for cfg in candidates:
        ms = triton.testing.do_bench(lambda: run_fn(cfg), warmup=warmup_ms, rep=rep_ms)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    _AUTOTUNE_CACHE[key] = best_cfg
    print(f"attn_bf16 autotune key={key} cfg={best_cfg} time={best_ms:.3f} ms")
    return best_cfg


def attn_hip_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _autotuned_op(q, k, v)

    b, h, n, d = q.shape
    n_kv = k.shape[2]

    def run_fn(cfg):
        return _configured_op(q, k, v, *cfg)

    best_cfg = _old_autotune(b, h, n, n_kv, d, _CONFIGS, run_fn)
    return _configured_op(q, k, v, *best_cfg)
