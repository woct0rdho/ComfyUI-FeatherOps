import os
from pathlib import Path

import torch

from kernel.hip.utils import load_hip_stable_extension

cur_dir = os.path.dirname(os.path.abspath(__file__))
_ck_tile_root = os.environ.get("FEATHEROPS_CK_TILE_ROOT", "~/rocm-libraries/projects/composablekernel")
_ck_tile_root = Path(_ck_tile_root).expanduser()
_extension_sources = [
    "hip_kernel.cpp",
    "featherattn_aligned.cu",
    "featherattn_query_tail.cu",
    "featherattn_key_tail.cu",
    "featherattn_query_key_tail.cu",
]
_extension_cuda_flags = [
    f"-I{_ck_tile_root / 'include'}",
    "-DCK_USE_WMMA=1",
    "-DCK_TILE_USE_WMMA=1",
    "-Wno-unknown-warning-option",
    "-Wno-lifetime-safety-intra-tu-suggestions",
    "-Wno-lifetime-safety-lifetimebound-violation",
]
load_hip_stable_extension(
    "attn_hip_ext",
    cur_dir,
    _extension_sources,
    extra_cuda_cflags=_extension_cuda_flags,
)


@torch.library.custom_op("feather_attn_internal::attn_fp16", mutates_args=())
def _feather_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: int) -> torch.Tensor:
    out = torch.empty_like(q)
    torch.ops.feather_attn_fp16.attn_fp16_feather.default(q, k, v, out, layout)
    return out


@_feather_attn.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: int) -> torch.Tensor:
    return torch.empty_like(q)


def feather_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str = "HND",
) -> torch.Tensor:
    normalized_layout = layout.upper()
    if normalized_layout == "HND":
        layout_id = 0
    elif normalized_layout == "NHD":
        layout_id = 1
    else:
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout!r}")
    return _feather_attn(q, k, v, layout_id)
