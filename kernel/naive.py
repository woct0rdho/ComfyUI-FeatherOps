from typing import Optional

import torch


def scaled_mm_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    b_prepacked: bool = False,
    bias_dim: int = 1,
) -> torch.Tensor:
    if b_prepacked:
        # (K/16, N, 16) -> (K, N)
        k, N, _ = b.shape
        b = b.permute(0, 2, 1).reshape(k * 16, N)
    if bias_dim not in {0, 1}:
        raise RuntimeError(f"bias_dim must be 0 or 1, got {bias_dim}")

    mm_dtype = torch.float16
    a = a.to(mm_dtype)
    b = b.to(mm_dtype)
    c = a @ b
    c = c.to(out_dtype)
    if scale is not None:
        scale = scale.to(c.dtype)
        if scale.dim() == 0:
            c *= scale
        else:
            if bias_dim == 0:
                c *= scale[:, None]
            else:
                c *= scale[None, :]
    if bias is not None:
        bias = bias.to(c.dtype)
        if bias_dim == 0:
            c += bias[:, None]
        else:
            c += bias[None, :]
    return c
