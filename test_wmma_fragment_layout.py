import torch

from kernel_attn.hip.hip_kernel import wmma_fragment_probe


def main() -> None:
    torch._inductor.config.compile_threads = 1

    out = wmma_fragment_probe().cpu().to(torch.float32)
    expected = torch.empty((4, 16, 16), dtype=torch.float32)
    for row in range(16):
        for col in range(16):
            expected[0, row, col] = row * 32 + col + 1
            expected[1, row, col] = row * 32 + 16 + col + 1
            expected[2, row, col] = row * 32 + col + 1
            expected[3, row, col] = row * 32 + 16 + col + 1

    diff = (out - expected).abs()
    max_diff = diff.max().item()
    if max_diff != 0:
        idx = diff.argmax().item()
        tile = idx // 256
        rem = idx % 256
        row = rem // 16
        col = rem % 16
        raise AssertionError(
            f"WMMA fragment probe failed max_diff={max_diff} at tile={tile} row={row} col={col}: got={out[tile, row, col].item()} expected={expected[tile, row, col].item()}"
        )

    print("[PASS] WMMA C-fragment to PV A-fragment shuffle mapping")


if __name__ == "__main__":
    main()
