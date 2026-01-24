from typing import Optional

import torch
import triton
import triton.language as tl


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
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


@triton.jit
def _scaled_mm_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    scale_ptr,
    bias_ptr,
    c_ptr,
    # Dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # Metadata
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    mm_dtype = tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_k[None, :]).to(mm_dtype)
        b = tl.load(b_ptrs, mask=mask_k[:, None]).to(mm_dtype)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc.to(c_ptr.dtype.element_ty)

    if HAS_SCALE:
        scale = tl.load(scale_ptr).to(acc.dtype)
        acc *= scale

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N).to(acc.dtype)
        acc += bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    # Metadata
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 8,
    NUM_WARPS: int = 8,
    NUM_STAGES: int = 4,
) -> torch.Tensor:
    assert a.is_cuda
    assert b.device == a.device
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    if scale is not None:
        assert scale.device == a.device
        assert scale.numel() == 1
    if bias is not None:
        assert bias.device == a.device
        assert bias.is_contiguous()
        assert bias.numel() == b.shape[1]

    native_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if a.dtype in native_dtypes and b.dtype in native_dtypes:
        return scaled_mm_naive(a, b, scale, bias, out_dtype)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _scaled_mm_kernel[grid](
        # Pointers
        a,
        b,
        scale,
        bias,
        c,
        # Dimensions
        M,
        N,
        K,
        # Strides
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_SCALE=(scale is not None),
        HAS_BIAS=(bias is not None),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c
