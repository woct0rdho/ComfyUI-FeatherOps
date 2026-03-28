import functools
import inspect
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._inductor.kernel.custom_op as inductor_custom_op
import triton
from packaging import version
from torch._inductor.kernel.custom_op import CustomOpConfig
from torch._inductor.select_algorithm import realize_inputs
from torch.utils.cpp_extension import _import_module_from_library, load

try:
    # torch >= 2.12
    from torch.fx.experimental.symbolic_shapes import optimization_hint as hint_int
except ImportError:
    from torch.fx.experimental.symbolic_shapes import hint_int


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
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF2_OPERATORS__",
            "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT16_OPERATORS__",
            "-U__HIP_NO_BFLOAT162_OPERATORS__",
        ],
        extra_ldflags=extra_ldflags,
        extra_include_paths=includes,
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
        is_python_module=False,
    )
    Path(ninja_log).touch(exist_ok=True)


def patch_inductor_custom_op_autotune_realize_inputs():
    if version.parse(torch.__version__) >= version.parse("2.11"):
        return

    if getattr(inductor_custom_op.autotune_custom_op, "_featherops_realize_inputs_patch", False):
        return

    original_autotune_custom_op = inductor_custom_op.autotune_custom_op
    signature = inspect.signature(original_autotune_custom_op)

    @functools.wraps(original_autotune_custom_op)
    def wrapped_autotune_custom_op(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        inputs = bound.arguments.get("inputs")
        if inputs is not None:
            bound.arguments["inputs"] = realize_inputs(*inputs)
        return original_autotune_custom_op(*bound.args, **bound.kwargs)

    wrapped_autotune_custom_op._featherops_realize_inputs_patch = True
    inductor_custom_op.autotune_custom_op = wrapped_autotune_custom_op


def _config_compatible(cfg, M, N, K):
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    block_m = 16 * warps_m * repeat_m
    block_n = 16 * warps_n * repeat_n
    chunk_k = 16 * unroll_k

    if M % block_m != 0 or N % block_n != 0 or K % chunk_k != 0:
        return False

    is_large = K >= 3072 and min(M, N) >= 3072 and max(M, N) >= 4096
    if is_large and (block_m < 64 or block_n < 64):
        return False

    return True


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

    # Same defaults as triton.autotune
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
    print(f"autotune key={key} cfg={best_cfg} time={best_ms:.3f} ms")
    return best_cfg
