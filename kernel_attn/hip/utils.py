import os
from typing import Callable, Dict, List, Tuple

import torch
import triton
from torch._inductor.kernel.custom_op import CustomOpConfig
from torch.fx.experimental.symbolic_shapes import optimization_hint

AttnConfig = Tuple[int, int, int]

CONFIGS: List[AttnConfig] = [
    (64, 64, 16),
    (64, 64, 8),
    (64, 128, 16),
    (128, 64, 16),
    (32, 64, 8),
    (64, 32, 8),
    (32, 32, 8),
    (16, 32, 2),
]


def config_compatible(cfg: AttnConfig, n: int, n_kv: int, d: int) -> bool:
    Br, Bc, _ = cfg
    return (n % Br == 0) and (n_kv % Bc == 0) and (d % 32 == 0)


def size_hint(value: int) -> int:
    try:
        return int(value)
    except TypeError:
        return optimization_hint(value)


def get_compatible_config(q: torch.Tensor, k: torch.Tensor, configs: List[AttnConfig]) -> AttnConfig:
    n = size_hint(q.shape[2])
    n_kv = size_hint(k.shape[2])
    d = size_hint(q.shape[3])
    for cfg in configs:
        if config_compatible(cfg, n, n_kv, d):
            return cfg
    raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")


def generate_autotune_configs(fake_tensors: Dict[str, torch.Tensor], configs: List[AttnConfig]) -> List[CustomOpConfig]:
    q = fake_tensors["q"]
    k = fake_tensors["k"]
    n = size_hint(q.shape[2])
    n_kv = size_hint(k.shape[2])
    d = size_hint(q.shape[3])
    compatible = [cfg for cfg in configs if config_compatible(cfg, n, n_kv, d)]
    if not compatible:
        raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")

    return [CustomOpConfig(br=cfg[0], bc=cfg[1], n_waves=cfg[2]) for cfg in compatible]


_AUTOTUNE_CACHE = {}


def old_autotune(
    b: int,
    h: int,
    n: int,
    n_kv: int,
    d: int,
    configs: List[AttnConfig],
    run_fn: Callable[[AttnConfig], torch.Tensor],
    op_name: str,
) -> AttnConfig:
    key = (op_name, b, h, n, n_kv, d)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = [c for c in configs if config_compatible(c, n, n_kv, d)]
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
    print(f"{op_name} autotune key={(b, h, n, n_kv, d)} cfg={best_cfg} time={best_ms:.3f} ms")
    return best_cfg
