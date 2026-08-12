#!/usr/bin/env python3

import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from kernel_attn.hip.hip_kernel import feather_attn

BATCH = 1
BENCHMARK_HEAD_COUNTS = (16, 32, 56)
GENERAL_HEAD_COUNTS = (1, 2, 3, 4, 7, 24, 30, 40, 48)
HEAD_DIMS = (64, 128)
BENCHMARK_SEQ_LENS = (257, 4096, 8192)
GENERAL_SEQ_LENS = (
    1,
    16,
    17,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    1000,
    1023,
    1024,
    1025,
    4095,
    4096,
    4097,
    8192,
)


def _contract_cases() -> list[tuple[int, int, int, int, int]]:
    cases = []
    for head_dim in HEAD_DIMS:
        cases.extend((BATCH, heads, seq_len, seq_len, head_dim) for heads in BENCHMARK_HEAD_COUNTS for seq_len in BENCHMARK_SEQ_LENS)
        cases.extend((BATCH, BENCHMARK_HEAD_COUNTS[0], seq_len, seq_len, head_dim) for seq_len in GENERAL_SEQ_LENS if seq_len not in BENCHMARK_SEQ_LENS)
        cases.extend((BATCH, heads, BENCHMARK_SEQ_LENS[0], BENCHMARK_SEQ_LENS[0], head_dim) for heads in GENERAL_HEAD_COUNTS if heads not in BENCHMARK_HEAD_COUNTS)
        cases.extend(
            (
                (1, 3, 1, 65, head_dim),
                (1, 3, 65, 64, head_dim),
                (1, 3, 128, 129, head_dim),
                (1, 3, 129, 128, head_dim),
                (1, 3, 150, 1024, head_dim),
                (1, 3, 256, 1025, head_dim),
                (2, 3, 129, 65, head_dim),
            )
        )
    return cases


def aiter_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, _, _, _ = flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return out


def test_case(batch: int, heads: int, n_q: int, n_kv: int, head_dim: int, device: str) -> tuple[bool, str]:
    torch.manual_seed(0)
    q = torch.randn((batch, heads, n_q, head_dim), dtype=torch.float16, device=device)
    k = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.float16, device=device)
    v = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.float16, device=device)

    out_ref = (
        aiter_attn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )
        .transpose(1, 2)
        .contiguous()
    )
    out_hip = feather_attn(q, k, v)

    out_ref_f = out_ref.float()
    diff = out_hip.float() - out_ref_f
    rel_l2 = (diff.norm() / out_ref_f.norm().clamp_min(1e-6)).item()
    max_abs = diff.abs().max().item()
    tolerance_factor = 0.10 if n_kv < 1024 else 0.05
    tolerance = tolerance_factor * out_ref_f.abs() + tolerance_factor
    tolerance_ratio = (diff.abs() / tolerance).max().item()
    pass_fraction = (diff.abs() <= tolerance).float().mean().item()
    passed = bool(torch.all(diff.abs() <= tolerance))
    msg = f"rel_l2={rel_l2:.3g} max_abs={max_abs:.3g} gate={tolerance_factor:.2f}/{tolerance_factor:.2f} tol_ratio={tolerance_ratio:.3g} pass_fraction={pass_fraction:.6f}"
    return passed, msg


def main() -> None:
    device = "cuda"
    cases = _contract_cases()

    print("Testing attention HIP public FP16 contract")
    print(f"benchmark_heads={BENCHMARK_HEAD_COUNTS} general_heads={GENERAL_HEAD_COUNTS} head_dims={HEAD_DIMS} cases={len(cases)}")
    print("=" * 96)

    failures = []
    for batch, heads, n_q, n_kv, head_dim in cases:
        try:
            passed, msg = test_case(batch, heads, n_q, n_kv, head_dim, device)
        except (RuntimeError, ValueError) as exc:
            # Report every unsupported tail instead of stopping at the first one.
            passed = False
            msg = f"{type(exc).__name__}: {exc}"

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] B={batch} H={heads} NQ={n_q} NKV={n_kv} D={head_dim}: {msg}")
        if not passed:
            failures.append((batch, heads, n_q, n_kv, head_dim, msg))

    print("=" * 96)
    print(f"Summary: {len(cases) - len(failures)}/{len(cases)} contract cases passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
