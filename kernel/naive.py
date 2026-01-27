from typing import Optional

import torch


def scaled_mm_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    mm_dtype = torch.float16
    a = a.to(mm_dtype)
    b = b.to(mm_dtype)
    c = a @ b
    c = c.to(out_dtype)
    if scale is not None:
        c *= scale.to(c.dtype)
    if bias is not None:
        c += bias.to(c.dtype)
    return c
