import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel.hip.utils import load_hip_stable_extension

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_hip_stable_extension("feather_attn_ext", cur_dir, "fwd.cu")

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


def _config_compatible(cfg, n, n_kv, d):
    Br, Bc, N_WAVES = cfg
    return (n % Br == 0) and (n_kv % Bc == 0) and (d % 32 == 0)


_AUTOTUNE_CACHE = {}


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, h, n, d = q.shape
    n_kv = k.shape[2]

    out = torch.empty_like(q)

    key = (b, h, n, n_kv, d)
    if key in _AUTOTUNE_CACHE:
        best_cfg = _AUTOTUNE_CACHE[key]
        torch.ops.feather_attn.attn_fwd.default(q, k, v, out, *best_cfg)
        return out

    candidates = [c for c in _CONFIGS if _config_compatible(c, n, n_kv, d)]
    if not candidates:
        raise RuntimeError(f"No compatible config for n={n} n_kv={n_kv} d={d}")

    best_cfg = candidates[0]
    best_ms = None

    warmup_iters = 2
    bench_iters = 5

    for cfg in candidates:
        try:
            for _ in range(warmup_iters):
                torch.ops.feather_attn.attn_fwd.default(q, k, v, out, *cfg)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(bench_iters):
                torch.ops.feather_attn.attn_fwd.default(q, k, v, out, *cfg)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000 / bench_iters

            if best_ms is None or ms < best_ms:
                best_ms = ms
                best_cfg = cfg
        except Exception as e:
            # If a config fails (e.g. out of LDS), skip it
            print(f"Config {cfg} failed: {e}")
            continue

    if best_ms is None:
        raise RuntimeError("All configs failed.")

    print(f"attn_fwd autotune key={key} cfg={best_cfg} time={best_ms:.3f} ms")
    _AUTOTUNE_CACHE[key] = best_cfg

    # Run once more with best config to return correct output
    torch.ops.feather_attn.attn_fwd.default(q, k, v, out, *best_cfg)
    return out
