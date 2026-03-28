import torch

from kernel.convert import bf16_to_fp16


def test_conversion():
    x_bf16 = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    y_fp16_custom = bf16_to_fp16(x_bf16)
    y_fp16_torch = x_bf16.to(torch.float16)

    diff = (y_fp16_custom.float() - y_fp16_torch.float()).abs().max().item()
    print(f"Max diff: {diff}")
    assert diff < 1e-5, f"Max diff is too large: {diff}"
    print("Test passed!")


if __name__ == "__main__":
    test_conversion()
