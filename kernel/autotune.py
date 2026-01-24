import os
from itertools import product
from math import sqrt
from typing import Any, Callable

import torch
import triton

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

DEFAULT_BLOCK_SIZES_M = [16, 32, 64, 128, 256]
DEFAULT_BLOCK_SIZES_N = [16, 32, 64, 128, 256]
DEFAULT_BLOCK_SIZES_K = [16, 32, 64, 128, 256]
if torch.version.hip:
    DEFAULT_GROUP_SIZES_M = [4, 6, 8]
    DEFAULT_NUM_WARPS = [4, 8]
    DEFAULT_NUM_STAGES = [2]
else:
    DEFAULT_GROUP_SIZES_M = [8]
    DEFAULT_NUM_WARPS = [8]
    DEFAULT_NUM_STAGES = [3, 4]

# Currently get_device_properties cannot be traced by torch.compile
SMEM_SIZE = triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())["max_shared_mem"]


def get_autotune_configs() -> list[triton.Config]:
    configs = []
    for m, n, k, g, w, s in product(
        DEFAULT_BLOCK_SIZES_M,
        DEFAULT_BLOCK_SIZES_N,
        DEFAULT_BLOCK_SIZES_K,
        DEFAULT_GROUP_SIZES_M,
        DEFAULT_NUM_WARPS,
        DEFAULT_NUM_STAGES,
    ):
        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE_M": m,
                    "BLOCK_SIZE_N": n,
                    "BLOCK_SIZE_K": k,
                    "GROUP_SIZE_M": g,
                },
                num_warps=w,
                num_stages=s,
            )
        )
    return configs


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    smem_size: int,
) -> bool:
    a_size = BLOCK_SIZE_M * BLOCK_SIZE_K * a_dtype.itemsize
    b_size = BLOCK_SIZE_K * BLOCK_SIZE_N * b_dtype.itemsize
    if num_stages <= 1:
        size = max(a_size, b_size)
    else:
        # (num_stages - 1) stages of both tiles will be cached in smem
        size = (num_stages - 1) * (a_size + b_size)
    return size > smem_size


def _common_prune_criteria(smem_criteria: Callable, config: triton.Config, kwargs: dict[str, Any]) -> bool:
    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    a_dtype = kwargs["a_ptr"].dtype
    b_dtype = kwargs["b_ptr"].dtype
    if smem_criteria(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, a_dtype, b_dtype, SMEM_SIZE):
        return True

    M = kwargs["M"]
    N = kwargs["N"]
    K = kwargs["K"]
    max_block_size_M = max(M, DEFAULT_BLOCK_SIZES_M[0])
    max_block_size_N = max(N, DEFAULT_BLOCK_SIZES_N[0])
    max_block_size_K = max(K, DEFAULT_BLOCK_SIZES_K[0])
    if BLOCK_SIZE_M > max_block_size_M:
        return True
    if BLOCK_SIZE_N > max_block_size_N:
        return True
    if BLOCK_SIZE_K > max_block_size_K:
        return True

    min_block_size_M = min(sqrt(M), DEFAULT_BLOCK_SIZES_M[-1])
    min_block_size_N = min(sqrt(N), DEFAULT_BLOCK_SIZES_N[-1])
    min_block_size_K = min(sqrt(K), DEFAULT_BLOCK_SIZES_K[-1])
    if BLOCK_SIZE_M * BLOCK_SIZE_N < min_block_size_M * min_block_size_N:
        return True
    if BLOCK_SIZE_M * BLOCK_SIZE_K < min_block_size_M * min_block_size_K:
        return True
    if BLOCK_SIZE_N * BLOCK_SIZE_K < min_block_size_N * min_block_size_K:
        return True

    return False


def prune_configs(smem_criteria: Callable, configs: list[triton.Config], args, **kwargs) -> list[triton.Config]:
    pruned_configs = []
    for config in configs:
        if _common_prune_criteria(smem_criteria, config, args):
            continue
        pruned_configs.append(config)

    if os.getenv("AUTOTUNE_DISABLE", "0") == "1":
        # Return one config in the middle
        return [pruned_configs[len(pruned_configs) // 2]]

    return pruned_configs
