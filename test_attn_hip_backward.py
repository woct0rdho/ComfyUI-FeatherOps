#!/usr/bin/env python3

import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from kernel_attn.hip.hip_kernel import feather_attn_backward

CASES = (
    (1, 2, 33, 35, 64),
    (1, 3, 65, 67, 64),
    (2, 2, 65, 129, 64),
)


def _reference(q, k, v, out, lse, dout, scale):
    qf, kf, vf = q.float(), k.float(), v.float()
    outf, doutf, lsef = out.float(), dout.float(), lse.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    probabilities = torch.exp(scores - lsef.unsqueeze(-1))
    delta = (outf * doutf).sum(-1)
    d_probability = torch.matmul(doutf, vf.transpose(-1, -2))
    d_score = probabilities * (d_probability - delta.unsqueeze(-1))
    return (
        torch.matmul(d_score, kf) * scale,
        torch.matmul(d_score.transpose(-1, -2), qf) * scale,
        torch.matmul(probabilities.transpose(-1, -2), doutf),
        delta,
    )


def _saved_state(batch, heads, n_q, n_kv, head_dim, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_q = (batch, heads, n_q, head_dim)
    shape_kv = (batch, heads, n_kv, head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16, generator=generator)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16, generator=generator)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16, generator=generator)
    dout = torch.randn(shape_q, device="cuda", dtype=torch.float16, generator=generator)
    scale = head_dim**-0.5
    out_bshd, lse, _, _ = flash_attn_2.fwd(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return q, k, v, out_bshd.transpose(1, 2).contiguous(), lse.contiguous(), dout, scale


def _check(actual, expected, n_kv):
    tolerance_factor = 0.10 if n_kv < 1024 else 0.05
    tolerance = tolerance_factor + tolerance_factor * expected.float().abs()
    difference = actual.float() - expected.float()
    assert bool(torch.isfinite(actual).all())
    assert bool((difference.abs() <= tolerance).all())
    assert float((difference.abs() / tolerance).max()) <= 1.0


def main():
    total = 0
    for case_index, (batch, heads, n_q, n_kv, head_dim) in enumerate(CASES):
        state = _saved_state(batch, heads, n_q, n_kv, head_dim, seed=20260813 + case_index)
        q, k, v, out, lse, dout, scale = state
        expected = _reference(q, k, v, out, lse, dout, scale)
        implementation = "fused"
        actual = feather_attn_backward(
            q,
            k,
            v,
            out,
            lse,
            dout,
            sm_scale=scale,
            implementation=implementation,
        )
        for result, reference in zip(actual, expected):
            _check(result, reference, n_kv)
        total += 1
        print(f"PASS implementation={implementation} B={batch} H={heads} NQ={n_q} NKV={n_kv} D={head_dim}")
    print(f"Summary: {total} D64 saved-state cases passed")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
