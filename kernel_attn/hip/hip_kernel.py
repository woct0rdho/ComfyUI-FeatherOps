import math
import os
from pathlib import Path

import torch

from kernel.hip.utils import load_hip_stable_extension

cur_dir = os.path.dirname(os.path.abspath(__file__))
_ck_tile_root = os.environ.get("FEATHEROPS_CK_TILE_ROOT", "~/rocm-libraries/projects/composablekernel")
_ck_tile_root = Path(_ck_tile_root).expanduser()
_extension_sources = [
    "hip_kernel.cpp",
    "featherattn_bwd_fused_d64.cu",
    "featherattn_fwd_aligned.cu",
    "featherattn_fwd_query_tail.cu",
    "featherattn_fwd_key_tail.cu",
    "featherattn_fwd_query_key_tail.cu",
    "featherattn_fwd_strided.cu",
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


def _use_nhd_hnd_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    delta: torch.Tensor,
    sm_scale: float,
) -> bool:
    attention = (q, k, v, out, dout, dq, dk, dv)
    if not all(tensor.is_cuda and tensor.dtype == torch.float16 and tensor.ndim == 4 for tensor in attention):
        return False
    if not all(tensor.device == q.device and tensor.is_contiguous() for tensor in attention):
        return False
    if lse.device != q.device or delta.device != q.device:
        return False
    if lse.dtype != torch.float32 or delta.dtype != torch.float32:
        return False
    if lse.ndim != 3 or delta.ndim != 3 or not lse.is_contiguous() or not delta.is_contiguous():
        return False
    batch, n_q, heads, head_dim = q.shape
    if batch != 1 or heads not in (32, 56) or n_q < 8192 or n_q > 16384 or head_dim != 64:
        return False
    if k.shape != q.shape or v.shape != q.shape:
        return False
    if out.shape != q.shape or dout.shape != q.shape or dq.shape != q.shape:
        return False
    if dk.shape != k.shape or dv.shape != v.shape:
        return False
    if lse.shape != (batch, heads, n_q) or delta.shape != lse.shape:
        return False
    if not math.isfinite(sm_scale) or sm_scale <= 0.0:
        return False
    max_int32 = 2**31 - 1
    if q.numel() * q.element_size() - q.element_size() > max_int32:
        return False
    return lse.numel() * lse.element_size() - lse.element_size() <= max_int32


def _nhd_hnd_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    delta: torch.Tensor,
    sm_scale: float,
    implementation_id: int,
) -> None:
    q_hnd = q.transpose(1, 2).contiguous()
    k_hnd = k.transpose(1, 2).contiguous()
    v_hnd = v.transpose(1, 2).contiguous()
    out_hnd = out.transpose(1, 2).contiguous()
    dout_hnd = dout.transpose(1, 2).contiguous()
    dq_hnd = torch.empty_like(q_hnd)
    dk_hnd = torch.empty_like(k_hnd)
    dv_hnd = torch.empty_like(v_hnd)
    torch.ops.feather_attn_fp16.attn_bwd_fp16_feather.default(
        q_hnd,
        k_hnd,
        v_hnd,
        out_hnd,
        lse,
        dout_hnd,
        dq_hnd,
        dk_hnd,
        dv_hnd,
        delta,
        sm_scale,
        implementation_id,
        0,
    )
    dq.copy_(dq_hnd.transpose(1, 2))
    dk.copy_(dk_hnd.transpose(1, 2))
    dv.copy_(dv_hnd.transpose(1, 2))


def feather_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    sm_scale: float | None = None,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    *,
    implementation: str,
    layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    implementation_ids = {"fused": 1}
    normalized_implementation = implementation.lower()
    if normalized_implementation not in implementation_ids:
        raise ValueError("only implementation='fused' is currently supported")
    if q.shape[-1] != 64:
        raise ValueError("only D64 fused backward is currently supported")
    normalized_layout = layout.upper()
    if normalized_layout == "HND":
        layout_id = 0
    elif normalized_layout == "NHD":
        layout_id = 1
    else:
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout!r}")
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    delta = torch.empty(lse.shape, dtype=torch.float32, device=lse.device) if delta is None else delta
    implementation_id = implementation_ids[normalized_implementation]
    if layout_id == 1 and _use_nhd_hnd_backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        delta,
        float(sm_scale),
    ):
        try:
            _nhd_hnd_backward(
                q,
                k,
                v,
                out,
                lse,
                dout,
                dq,
                dk,
                dv,
                delta,
                float(sm_scale),
                implementation_id,
            )
        except torch.OutOfMemoryError:
            pass
        else:
            return dq, dk, dv, delta
    torch.ops.feather_attn_fp16.attn_bwd_fp16_feather.default(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        delta,
        float(sm_scale),
        implementation_id,
        layout_id,
    )
    return dq, dk, dv, delta
