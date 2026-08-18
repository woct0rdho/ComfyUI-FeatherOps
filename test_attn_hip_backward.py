#!/usr/bin/env python3

import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from kernel_attn.hip.hip_kernel import feather_attn_backward

LAYOUTS = ("HND", "NHD")
CASES = (
    (1, 16, 33, 35, 64),
    (1, 32, 65, 67, 64),
    (1, 56, 65, 129, 64),
    (2, 2, 65, 129, 64),
    (1, 1, 4095, 4097, 64),
    (1, 1, 4097, 4099, 64),
    (1, 1, 8191, 67, 64),
    (1, 1, 65, 8193, 64),
    (1, 1, 16383, 67, 64),
    (1, 1, 65, 16385, 64),
    (1, 32, 65, 16385, 64),
)
D128_CASES = tuple((*case[:4], 128) for case in CASES)


def _hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def _reference(q, k, v, out, lse, dout, scale, layout):
    q, k, v = (_hnd(tensor, layout) for tensor in (q, k, v))
    out, dout = (_hnd(tensor, layout) for tensor in (out, dout))
    qf, kf, vf = q.float(), k.float(), v.float()
    outf, doutf, lsef = out.float(), dout.float(), lse.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    probabilities = torch.exp(scores - lsef.unsqueeze(-1))
    delta = (outf * doutf).sum(-1)
    d_probability = torch.matmul(doutf, vf.transpose(-1, -2))
    d_score = probabilities * (d_probability - delta.unsqueeze(-1))
    dq = torch.matmul(d_score, kf) * scale
    dk = torch.matmul(d_score.transpose(-1, -2), qf) * scale
    dv = torch.matmul(probabilities.transpose(-1, -2), doutf)
    if layout == "NHD":
        dq, dk, dv = (tensor.transpose(1, 2) for tensor in (dq, dk, dv))
    return dq, dk, dv, delta


def _saved_state(batch, heads, n_q, n_kv, head_dim, seed, layout):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_q = (batch, heads, n_q, head_dim)
    shape_kv = (batch, heads, n_kv, head_dim)
    q_hnd = torch.randn(shape_q, device="cuda", dtype=torch.float16, generator=generator)
    k_hnd = torch.randn(shape_kv, device="cuda", dtype=torch.float16, generator=generator)
    v_hnd = torch.randn(shape_kv, device="cuda", dtype=torch.float16, generator=generator)
    dout_hnd = torch.randn(shape_q, device="cuda", dtype=torch.float16, generator=generator)
    scale = head_dim**-0.5
    out_bshd, lse, _, _ = flash_attn_2.fwd(
        q_hnd.transpose(1, 2),
        k_hnd.transpose(1, 2),
        v_hnd.transpose(1, 2),
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
    if layout == "HND":
        q, k, v, dout = q_hnd, k_hnd, v_hnd, dout_hnd
        out = out_bshd.transpose(1, 2).contiguous()
    else:
        q, k, v, dout = (tensor.transpose(1, 2).contiguous() for tensor in (q_hnd, k_hnd, v_hnd, dout_hnd))
        out = out_bshd.contiguous()
    return q, k, v, out, lse.contiguous(), dout, scale


def _check(actual, expected, n_kv):
    tolerance_factor = 0.10 if n_kv < 1024 else 0.05
    tolerance = tolerance_factor + tolerance_factor * expected.float().abs()
    difference = actual.float() - expected.float()
    assert bool(torch.isfinite(actual).all())
    assert bool((difference.abs() <= tolerance).all())
    assert float((difference.abs() / tolerance).max()) <= 1.0


def _check_nhd_hnd_dispatch(head_dim):
    q, k, v, out, lse, dout, scale = _saved_state(
        1,
        32,
        8192,
        8192,
        head_dim,
        seed=20260818 + head_dim,
        layout="NHD",
    )
    expected = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    expected_delta = torch.empty_like(lse)
    torch.ops.feather_attn_fp16.attn_bwd_fp16_feather.default(
        q,
        k,
        v,
        out,
        lse,
        dout,
        *expected,
        expected_delta,
        scale,
        1,
        1,
    )
    actual = feather_attn_backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        sm_scale=scale,
        implementation="fused",
        layout="NHD",
    )
    for result, direct in zip(actual, (*expected, expected_delta)):
        assert torch.equal(result, direct)
    print(f"PASS NHD H32 N8192 D{head_dim} HND-dispatch exact equivalence")


def main():
    total = 0
    for layout in LAYOUTS:
        for dimension_index, cases in enumerate((CASES, D128_CASES)):
            for case_index, (batch, heads, n_q, n_kv, head_dim) in enumerate(cases):
                state = _saved_state(
                    batch,
                    heads,
                    n_q,
                    n_kv,
                    head_dim,
                    seed=20260813 + dimension_index * 1000 + case_index,
                    layout=layout,
                )
                q, k, v, out, lse, dout, scale = state
                expected = _reference(q, k, v, out, lse, dout, scale, layout)
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
                    layout=layout,
                )
                for result, reference in zip(actual, expected):
                    _check(result, reference, n_kv)
                total += 1
                print(f"PASS implementation={implementation} layout={layout} B={batch} H={heads} NQ={n_q} NKV={n_kv} D={head_dim}")
    for head_dim in (64, 128):
        _check_nhd_hnd_dispatch(head_dim)
    print(f"Summary: {total} D64/D128 saved-state cases passed")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
