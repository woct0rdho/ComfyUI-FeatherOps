import functools
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._inductor.kernel.custom_op import CustomOpConfig
from torch.fx.experimental.symbolic_shapes import hint_int
from torch.utils.cpp_extension import load


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
        except ImportError:
            continue
    return [d for d in rocm_lib_dirs if os.path.isdir(d)]


def load_hip_stable_extension(name: str, cur_dir: str, source_filename: str):
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

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
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-DPy_LIMITED_API=0x03090000",
    ]

    extra_ldflags = []
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


def _config_compatible(cfg, M, N, K):
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    block_m = 16 * warps_m * repeat_m
    block_n = 16 * warps_n * repeat_n
    chunk_k = 16 * unroll_k
    return M % block_m == 0 and N % block_n == 0 and K % chunk_k == 0


def _size_hint(value: int) -> int:
    try:
        return int(value)
    except TypeError:
        return hint_int(value)


def generate_autotune_configs(
    fake_tensors: Dict[str, torch.Tensor],
    configs: List[Tuple[int, int, int, int, int]],
    b_dim: int,
) -> List[CustomOpConfig]:
    a = fake_tensors["a"]
    b_prepacked = fake_tensors["b_prepacked"]
    M = _size_hint(a.shape[0])
    N = _size_hint(b_prepacked.shape[b_dim])
    K = _size_hint(a.shape[1])
    compatible = [cfg for cfg in configs if _config_compatible(cfg, M, N, K)]
    if not compatible:
        raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")

    return [
        CustomOpConfig(
            block_warps_m=cfg[0],
            block_warps_n=cfg[1],
            unroll_k=cfg[2],
            repeat_m=cfg[3],
            repeat_n=cfg[4],
        )
        for cfg in compatible
    ]


def get_compatible_config(a: torch.Tensor, b_prepacked: torch.Tensor, b_dim: int, configs: List[Tuple[int, int, int, int, int]]) -> Tuple[int, int, int, int, int]:
    M = _size_hint(a.shape[0])
    N = _size_hint(b_prepacked.shape[b_dim])
    K = _size_hint(a.shape[1])
    for cfg in configs:
        if _config_compatible(cfg, M, N, K):
            return cfg
    raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")


_AUTOTUNE_CACHE = {}


# torch autotuner introduces overhead, so we use the old autotuner in benchmarks
def old_autotune(
    M: int,
    N: int,
    K: int,
    configs: List[Tuple[int, int, int, int, int]],
    run_fn: Callable[[Tuple[int, int, int, int, int]], Any],
    *extra_keys: Any,
) -> Tuple[int, int, int, int, int]:
    key = (M, N, K, *extra_keys)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = [c for c in configs if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")

    warmup_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_WARMUP", "1")))
    bench_iters = max(1, int(os.environ.get("HIP_AUTOTUNE_ITERS", "10")))
    best_cfg = candidates[0]
    best_ms = None

    for cfg in candidates:
        for _ in range(warmup_iters):
            run_fn(cfg)
    torch.cuda.synchronize()

    for cfg in candidates:
        start = time.perf_counter()
        for _ in range(bench_iters):
            run_fn(cfg)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / bench_iters
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    _AUTOTUNE_CACHE[key] = best_cfg
    print(f"autotune key={key} cfg={best_cfg} time={best_ms:.3f} ms")
    return best_cfg
