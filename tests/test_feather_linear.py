"""Integration tests for the FeatherOps quantized Linear and its LoRA path.

The LoRA policy is ComfyUI's, not ours: a patched weight is baked through `convert_weight` ->
`calculate_weight` -> `set_weight` (fully loaded model) or requantized per call by
`cast_bias_weight(..., want_requant=True)` (dynamic/offloaded model). Both are exercised here
against `comfy.lora.calculate_weight`, which is the reference for every adapter type.
"""

import comfy.lora
import comfy.model_patcher
import pytest
import torch
from comfy.ops import cast_bias_weight, uncast_bias_weight
from comfy.weight_adapter.lora import LoRAAdapter
from feather_ops import scaled_mm_hip

from comfy_feather import ops as feather_ops_module
from comfy_feather.ops import feather_ops
from comfy_feather.quant import is_feather_quantized_weight

IN, OUT, RANK = 512, 256, 16
DTYPE = torch.bfloat16
KEY = "lin.weight"


@pytest.fixture(autouse=True)
def _seed():
    """Random weights and LoRAs at a fixed seed, so the fp8 tolerances mean the same thing each run."""
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="module")
def ops():
    return feather_ops(policy=(("*",), ()), out_dtype=DTYPE)


class _Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.lin = linear


def _build(ops):
    """A quantized Linear loaded the way the loader loads one: a plain bf16 weight plus a scale."""
    model = _Model(ops.Linear(IN, OUT, bias=True))
    sd = {"weight": torch.randn(OUT, IN) * 0.05, "bias": torch.randn(OUT) * 0.1, "weight_scale": torch.ones(())}
    model.lin._load_from_state_dict(sd, "", {}, False, [], [], [])
    return model.cuda()


def _lora_patches(weight, rank=RANK, alpha=32.0):
    """One rank-`rank` LoRA, in the two shapes ComfyUI uses: patcher value and patch list."""
    up = torch.randn(OUT, rank, dtype=torch.float32) * 0.02
    down = torch.randn(rank, weight.shape[1], dtype=torch.float32) * 0.02
    adapter = LoRAAdapter({KEY}, (up, down, torch.tensor(alpha), None, None, None))
    return {KEY: adapter}, {KEY: [(1.0, adapter, 1.0, None, None)]}


def _reference(linear, x, patches=None):
    """`F.linear` of the dequantized weight, with every patch applied the way ComfyUI applies it."""
    weight = linear.weight.dequantize().float()
    if patches:
        weight = comfy.lora.calculate_weight(patches[KEY], weight, KEY, intermediate_dtype=torch.float32)
    return torch.nn.functional.linear(x, weight.to(DTYPE), linear.bias)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_quantized_linear_matches_the_packed_kernel(ops):
    model = _build(ops)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        out = model.lin(x)
        ref = _reference(model.lin, x)
    assert is_feather_quantized_weight(model.lin.weight)
    assert _rel(out, ref) < 2e-2, "fp8 e5m2 round trip of the weight"
    assert model.lin.is_quantized


def test_baked_lora_keeps_the_packed_weight(ops):
    """A fully loaded model gets its LoRA baked once, and the weight stays packed for the kernel."""
    model = _build(ops)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    unpatched = _reference(model.lin, x)
    _, patches = _lora_patches(model.lin.weight)
    with torch.no_grad():
        patcher = comfy.model_patcher.ModelPatcher(model, load_device="cuda", offload_device="cpu")
        patcher.add_patches({KEY: patches[KEY][0][1]}, 1.0)
        patcher.patch_weight_to_device(KEY)
        assert is_feather_quantized_weight(model.lin.weight), "set_weight must repack, not store bf16"
        out = model.lin(x)
        ref = _reference(model.lin, x)
        patched = _reference(model.lin, x, patches)
    # the LoRA is in the packed weight: the output follows the patched reference, not the base one
    assert _rel(out, ref) < 2e-2
    assert _rel(patched, unpatched) > 5e-2, "the test LoRA must move the output for this to mean anything"


def test_lora_patch_is_not_applied_twice_or_dropped(ops):
    """The output tracks `calculate_weight` and backtracking the patch restores the base output."""
    model = _build(ops)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        base = model.lin(x).clone()
        patcher = comfy.model_patcher.ModelPatcher(model, load_device="cuda", offload_device="cpu")
        patcher.add_patches({KEY: _lora_patches(model.lin.weight)[0][KEY]}, 1.0)
        patcher.patch_weight_to_device(KEY)
        patched = model.lin(x)
        assert _rel(patched, base) > 5e-2
        # the same patcher with the patch removed goes back to the unpatched weight
        patcher.unpatch_model()
        assert _rel(model.lin(x), base) < 2e-2


def test_requantize_from_float_is_what_the_dynamic_path_calls(ops):
    """`cast_bias_weight(want_requant=True)` ends in `QuantizedTensor.requantize_from_float`.

    On the offloaded path ComfyUI dequantizes the weight, applies the patch and asks the layout to
    repack the result, so this is the piece that has to round trip for a patched weight to stay on
    the fp8 kernel. The requantized tensor must still run through `scaled_mm_hip`.
    """
    model = _build(ops)
    linear = model.lin
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        _, patches = _lora_patches(linear.weight)
        patched_float = comfy.lora.calculate_weight(patches[KEY], linear.weight.dequantize().float(), KEY, intermediate_dtype=torch.float32)
        requant = linear.weight.requantize_from_float(patched_float, scale="recalculate")
        assert is_feather_quantized_weight(requant)
        assert requant.shape == linear.weight.shape
        plain, scale = feather_ops_module.get_feather_plain_tensors(requant)
        out = scaled_mm_hip(x, plain, scale, linear.bias, out_dtype=DTYPE)
        requantized = requant.dequantize()
        ref = torch.nn.functional.linear(x, requantized.to(DTYPE), linear.bias)
        base = _reference(linear, x)
        # the exact call `comfy/ops.py` makes when it requantizes a lazily patched weight
        seeded = linear.weight.requantize_from_float(patched_float, scale="recalculate", stochastic_rounding=1234)
        assert is_feather_quantized_weight(seeded)
    # e5m2 has two mantissa bits, so this is the quantization error and not something we control
    assert _rel(requantized, patched_float) < 0.08
    # the kernel must compute what the requantized weight says, not something else
    assert _rel(out, ref) < 1e-2
    assert _rel(out, base) > 5e-2, "the patch has to survive the requant"


def test_requant_flag_keeps_a_resident_weight_packed(ops):
    """`want_requant` must not itself dequantize: with no patch the packed weight comes back as is."""
    model = _build(ops)
    linear = model.lin
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        weight, bias, stream = cast_bias_weight(linear, x, dtype=linear.weight.dtype, bias_dtype=DTYPE, offloadable=True, want_requant=True)
        uncast_bias_weight(linear, weight, bias, stream)
        assert is_feather_quantized_weight(weight)
        plain, scale = feather_ops_module.get_feather_plain_tensors(weight)
        out = scaled_mm_hip(x, plain, scale, bias, out_dtype=DTYPE)
    assert _rel(out, _reference(linear, x)) < 2e-2


def test_offloaded_patch_flag_is_passed(ops, monkeypatch):
    """The Linear asks for a requant exactly when ComfyUI left a lazy patch on the weight."""
    calls = {}

    def spy(s, x, **kwargs):
        calls.update(kwargs)
        return cast_bias_weight(s, x, **kwargs)

    monkeypatch.setattr(feather_ops_module, "cast_bias_weight", spy)
    model = _build(ops)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        model.lin(x)
        assert calls["want_requant"] is False, "a resident, unpatched weight needs no requant"
        model.lin.weight_lowvram_function = comfy.model_patcher.LowVramPatch(KEY, _lora_patches(model.lin.weight)[1])
        model.lin(x)
        assert calls["want_requant"] is True, "a lazy patch has to be requantized into the pack"
        del model.lin.weight_lowvram_function


def test_weight_wrapper_falls_back_to_bf16_linear(ops):
    """A `weight_function` wrapper that ComfyUI cannot requantize still applies, via F.linear."""
    model = _build(ops)
    linear = model.lin
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        base = linear(x)
        linear.weight_function = [lambda weight: weight * 1.5]
        try:
            out = linear(x)
            ref = torch.nn.functional.linear(x, linear.weight.dequantize().to(DTYPE) * 1.5, linear.bias)
        finally:
            linear.weight_function = []
    assert is_feather_quantized_weight(linear.weight), "a wrapper must not overwrite the packed weight"
    assert _rel(out, ref) < 2e-2
    assert _rel(out, base) > 5e-2


def test_unquantized_linear_still_goes_through_comfy_ops(ops):
    """Layers the policy leaves in full precision keep ComfyUI's plain path."""
    # policy None keeps every Linear in full precision, which is how a recipe declines a layer
    linear = feather_ops(policy=None, out_dtype=DTYPE).Linear(IN, OUT, bias=True)
    sd = {"weight": torch.randn(OUT, IN) * 0.05, "bias": torch.randn(OUT) * 0.1}
    linear._load_from_state_dict(sd, "", {}, False, [], [], [])
    linear = linear.cuda().to(DTYPE)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        out = linear(x)
        ref = torch.nn.functional.linear(x, linear.weight, linear.bias)
    assert not linear.is_quantized
    assert torch.equal(out, ref)
