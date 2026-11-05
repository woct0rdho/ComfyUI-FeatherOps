"""The loader's format policy, at the module level where it is decided.

- int8 / fp8e5m2 checkpoints are already quantized: their layers are prepacked as stored, and the
  layers the exporter left in float stay in float.
- any other float dtype is cast to fp8e5m2, and the comfy-quants recipe decides which layers.
"""

import json

import pytest
import torch

from comfy_feather import loader as loader_module
from comfy_feather import quant as q
from comfy_feather.ops import feather_ops

IN, OUT = 512, 256
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="module")
def ops():
    return feather_ops(policy=(("*",), ()), out_dtype=DTYPE)


@pytest.fixture(scope="module")
def no_recipe():
    """`policy=None` is what the loader passes for an already-quantized checkpoint: quantize nothing."""
    return feather_ops(policy=None, out_dtype=DTYPE)


def load(ops, state):
    """Load on the GPU from a copy of `state`, and hand the consumed copy back."""
    linear = ops.Linear(IN, OUT, bias=True)
    # weights and scales go to the compute device. The uint8 tensors are markers and packed codes, which stay
    # on the host the way `load_torch_file` hands them over
    state = {key: value.cuda() if torch.is_tensor(value) and value.dtype != torch.uint8 else value for key, value in state.items()}
    missing, unexpected, errors = [], [], []
    linear._load_from_state_dict(state, "", {}, False, missing, unexpected, errors)
    assert not errors and not missing and not unexpected
    return linear, state


def marker_bytes(**conf) -> torch.Tensor:
    payload = {"format": "int8_tensorwise", "per_row": True, "convrot": True, "convrot_groupsize": 256} | conf
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def fp8_state(scale=None):
    """An fp8e5m2 checkpoint entry: stored fp8 bytes, a per-tensor scale, the marker."""
    weight = (torch.randn(OUT, IN) * 0.05).to(torch.float8_e5m2)
    scale = torch.tensor(0.02) if scale is None else scale
    return {"weight": weight, "weight_scale": scale, "bias": torch.randn(OUT) * 0.1, "comfy_quant": marker_bytes(format="float8_e5m2")}, weight, scale


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_fp8_checkpoint_is_prepacked_not_requantized(ops):
    state, weight, scale = fp8_state()
    linear, consumed = load(ops, state)
    assert not consumed, "every key the entry carried must be consumed"
    assert q.is_feather_fp8_weight(linear.weight)
    assert torch.equal(linear.weight._qdata.cpu(), q.prepack_transpose(weight)), "stored bytes, only permuted"
    assert torch.equal(linear.weight._params.scale.cpu(), scale.float())
    assert linear.weight.shape == (OUT, IN)


def test_fp8_layer_survives_a_recipe_that_excludes_it(ops, no_recipe):
    """A stored-quantized layer is prepacked whatever the recipe says. The recipe is for float weights."""
    for ops_class in (ops, no_recipe):
        state, weight, _ = fp8_state()
        linear, _ = load(ops_class, state)
        assert q.is_feather_fp8_weight(linear.weight)
        assert torch.equal(linear.weight._qdata.cpu(), q.prepack_transpose(weight))


def test_fp8_forward_matches_the_stored_weight(ops):
    state, weight, scale = fp8_state()
    linear, _ = load(ops, state)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        out = linear(x)
        ref = torch.nn.functional.linear(x, (weight.float() * scale).to(DTYPE).cuda(), linear.bias)
    assert rel(out, ref) < 2e-2


def test_float_layer_of_a_quantized_checkpoint_stays_float(no_recipe):
    state = {"weight": torch.randn(OUT, IN) * 0.05, "bias": torch.randn(OUT) * 0.1}
    linear, _ = load(no_recipe, state)
    assert not linear.is_quantized
    assert linear.weight.dtype == DTYPE
    assert torch.equal(linear.weight.cpu(), state["weight"].to(DTYPE))


def test_float_checkpoint_follows_the_recipe(ops, no_recipe):
    state = {"weight": torch.randn(OUT, IN) * 0.05, "bias": torch.randn(OUT) * 0.1}
    quantized, _ = load(ops, state)
    excluded, _ = load(no_recipe, state)
    assert q.is_feather_fp8_weight(quantized.weight)
    assert not excluded.is_quantized


def test_fp8e4m3_checkpoint_is_cast_to_e5m2(ops):
    """e4m3 is 'other fp dtype': its scale is applied and the result is requantized to e5m2."""
    weight = (torch.randn(OUT, IN) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.02)
    linear, _ = load(ops, {"weight": weight, "weight_scale": scale, "bias": None, "comfy_quant": marker_bytes(format="float8_e4m3fn")})
    assert q.is_feather_fp8_weight(linear.weight)
    with torch.no_grad():
        values = linear.weight.dequantize().cpu()
    assert rel(values, (weight.float() * scale).to(DTYPE)) < 8e-2, "e4m3 then e5m2, two 3-bit mantissas"


def test_excluded_e4m3_layer_is_dequantized_not_left_raw(no_recipe):
    """The long-standing trap: an excluded quantized layer must apply its scale, not keep the raw codes."""
    weight = (torch.randn(OUT, IN) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.02)
    linear, _ = load(no_recipe, {"weight": weight, "weight_scale": scale, "bias": None, "comfy_quant": marker_bytes(format="float8_e4m3fn")})
    assert not linear.is_quantized
    assert linear.weight.dtype == DTYPE
    assert torch.allclose(linear.weight.cpu().float(), weight.float() * scale, rtol=2e-2, atol=1e-6)


def test_fp8_with_a_per_channel_scale_is_dequantized(no_recipe):
    """The fp8 layout carries one scale for the tensor, so a per-channel-scale file must not be prepacked."""
    weight = (torch.randn(OUT, IN) * 0.05).to(torch.float8_e5m2)
    scale = torch.rand(OUT, 1) * 0.01 + 0.01
    linear, _ = load(no_recipe, {"weight": weight, "weight_scale": scale, "bias": None, "comfy_quant": marker_bytes(format="float8_e5m2")})
    assert not linear.is_quantized
    assert torch.allclose(linear.weight.cpu().float(), weight.float() * scale, rtol=2e-2, atol=1e-8)


def test_int8_rotated_at_another_group_is_unrotated_by_comfy(no_recipe):
    """ConvRot at a group size this kernel does not implement: ComfyUI's loader un-rotates it at that size."""
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    weight = torch.randn(OUT, IN) * 0.05
    qdata, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=64)
    state = {"bias": None, "comfy_quant": marker_bytes(convrot_groupsize=64)}
    for suffix, value in TensorWiseINT8Layout.state_dict_tensors(qdata, params).items():
        state["weight" + suffix] = value
    linear, consumed = load(no_recipe, state)
    assert not consumed
    assert not linear.is_quantized
    assert rel(linear.weight.cpu(), weight) < 3e-2, "dequantized and un-rotated at group 64"


def test_int8_without_the_marker_is_dequantized(no_recipe):
    """An int8 entry our layout cannot verify (no marker) is dequantized with its scale, not run as int8."""
    qdata, scale = q.quantize_int8_convrot_rows(torch.randn(OUT, IN) * 0.05)
    linear, _ = load(no_recipe, {"weight": qdata, "weight_scale": scale.reshape(-1, 1), "bias": None})
    assert not linear.is_quantized
    assert linear.weight.dtype == DTYPE


def test_packed_checkpoint_entry_is_dequantized_by_comfy(no_recipe):
    """nvfp4 and friends hold byte codes plus several scales. ComfyUI's own loader turns them into floats."""
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout

    weight = torch.randn(OUT, IN) * 0.05
    qdata, params = TensorCoreNVFP4Layout.quantize(weight)
    state = {"bias": None, "comfy_quant": marker_bytes(format="nvfp4")}
    for suffix, value in TensorCoreNVFP4Layout.state_dict_tensors(qdata, params).items():
        state["weight" + suffix] = value
    linear, consumed = load(no_recipe, state)
    assert not consumed, "the loader must consume every key it read"
    assert not linear.is_quantized, "a 4-bit layer the recipe keeps is a float layer"
    assert linear.weight.dtype == DTYPE
    assert rel(linear.weight.cpu(), weight) < 0.2, "dequantized 4-bit weights land within their own precision"


def test_quant_policy_skips_the_recipe_for_stored_formats(monkeypatch):
    """The loader-side half of the rule: an already-quantized file never gets the comfy-quants policy."""
    for stored in ({"int8"}, {"fp8e5m2"}, {"int8", "fp8e5m2"}):
        monkeypatch.setattr(loader_module, "_inspect_checkpoint", lambda path, stored=stored: ("", None, DTYPE, frozenset(stored)))
        policy, out_dtype, returned = loader_module.quant_policy("anima", "unused.safetensors")
        assert policy is None and out_dtype == DTYPE and returned == frozenset(stored)

    monkeypatch.setattr(loader_module, "_inspect_checkpoint", lambda path: ("", None, DTYPE, frozenset()))
    policy, _, stored = loader_module.quant_policy("all", "unused.safetensors")
    assert policy == (("*",), ()) and stored == frozenset(), "a float file still gets the recipe"


def test_stored_format_detection():
    fp8 = torch.zeros(4, 4, dtype=torch.float8_e5m2)
    int8 = torch.zeros(4, 4, dtype=torch.int8)
    assert loader_module._stored_formats({"a.weight": int8}) == frozenset({"int8"})
    assert loader_module._stored_formats({"a.weight": fp8}) == frozenset({"fp8e5m2"})
    assert loader_module._stored_formats({"a.weight": int8, "b.weight": fp8}) == frozenset({"int8", "fp8e5m2"})
    assert loader_module._stored_formats({"a.weight": torch.zeros(4, 4)}) == frozenset()
    assert loader_module._stored_formats({"a.weight_scale": int8}) == frozenset(), "only weights count"
