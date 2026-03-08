import functools
import os
from pathlib import Path

from torch.utils.cpp_extension import _import_module_from_library, load


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


def load_hip_extension(name: str, cur_dir: str, source_filename: str):
    build_dir = os.path.join(cur_dir, "build", name)
    os.makedirs(build_dir, exist_ok=True)

    source_file = os.path.join(cur_dir, source_filename)
    ninja_log = os.path.join(build_dir, ".ninja_log")
    should_rebuild = False

    if os.path.exists(source_file) and os.path.exists(ninja_log):
        if os.path.getmtime(source_file) > os.path.getmtime(ninja_log):
            should_rebuild = True

    if not should_rebuild:
        try:
            return _import_module_from_library(name, build_dir, is_python_module=True)
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
    ]

    extra_ldflags = []
    for lib_dir in dict.fromkeys(get_rocm_lib_dirs()):
        extra_ldflags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

    module = load(
        name=name,
        sources=[source_file],
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
    )
    Path(ninja_log).touch(exist_ok=True)
    return module


def _config_compatible(cfg, M, N, K):
    """Check if a config's tile sizes evenly divide the matrix dimensions."""
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    block_m = 16 * warps_m * repeat_m
    block_n = 16 * warps_n * repeat_n
    chunk_k = 16 * unroll_k
    return M % block_m == 0 and N % block_n == 0 and K % chunk_k == 0


@functools.cache
def _get_forced_config():
    cfg = os.environ.get("HIP_FORCE_CONFIG")
    if not cfg:
        return None
    values = tuple(int(v.strip()) for v in cfg.split(",") if v.strip())
    if len(values) != 5:
        raise RuntimeError("HIP_FORCE_CONFIG must contain 5 comma-separated integers: warps_m,warps_n,unroll_k,repeat_m,repeat_n")
    return values
