import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_to_fp16_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)

    out = tl.inline_asm_elementwise(
        "v_lshlrev_b32 $1, 16, $2\nv_and_b32 $0, 0xffff0000, $2\nv_cvt_pkrtz_f16_f32 $0, $1, $0",
        "=v,=v,v",
        [x],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )

    tl.store(out_ptr + offsets, out[0], mask=mask)


def bf16_to_fp16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert x.is_contiguous()
    assert x.numel() % 2 == 0

    x_int32 = x.view(torch.int32)
    out_int32 = torch.empty_like(x_int32)
    n_elements = x_int32.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _bf16_to_fp16_kernel[grid](x_int32, out_int32, n_elements, BLOCK_SIZE=1024)
    return out_int32.view(torch.float16)
