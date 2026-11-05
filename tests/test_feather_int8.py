"""Tests for the int8 ConvRot layout and its kernel dispatch in the FeatherOps Linear.

The checkpoint form is what comfy-quants writes: int8 rows rotated by the group-256 regular Hadamard,
one float32 scale per output channel, plus a `comfy_quant` marker that says the rows are rotated.
"""

import json

import comfy.lora
import comfy.model_patcher
import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from comfy_feather import quant as q
from comfy_feather.ops import feather_ops

IN, OUT, RANK = 512, 256, 16
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="module")
def ops():
    return feather_ops(policy=(("*",), ()), out_dtype=DTYPE)


def marker(**conf) -> bytes:
    """The `comfy_quant` payload. On disk it is json bytes in a uint8 tensor, which is what a state dict carries."""
    payload = {"format": "int8_tensorwise", "per_row": True, "convrot": True, "convrot_groupsize": 256} | conf
    return json.dumps(payload).encode("utf-8")


def marker_tensor(**conf) -> torch.Tensor:
    return torch.tensor(list(marker(**conf)), dtype=torch.uint8)


def int8_state(shape=(OUT, IN)):
    """A ConvRot-quantized weight, as a checkpoint would carry it."""
    n, k = shape
    weight = torch.randn(n, k) * 0.05
    qdata, scale = q.quantize_int8_convrot_rows(weight)
    return {"weight": qdata, "weight_scale": scale.reshape(-1, 1)}, weight


def load(ops, state, prefix=""):
    """Load a checkpoint entry the way `load_state_dict` does, then move the layer to the GPU."""
    linear = ops.Linear(IN, OUT, bias=True)
    state = dict(state)
    state.setdefault(prefix + "bias", torch.randn(OUT) * 0.1)
    state.setdefault(prefix + "comfy_quant", marker_tensor())
    missing, unexpected, errors = [], [], []
    linear._load_from_state_dict(state, prefix, {}, False, missing, unexpected, errors)
    assert not errors
    return linear.cuda(), missing, unexpected


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_eligibility_gates_on_the_marker():
    state, _ = int8_state()
    weight, scale = state["weight"], state["weight_scale"].reshape(-1)
    assert q.int8_convrot_eligible(weight, scale, marker())
    assert not q.int8_convrot_eligible(weight, scale, None), "no marker means no claim that rows are rotated"
    assert not q.int8_convrot_eligible(weight, scale, marker(convrot=False))
    assert not q.int8_convrot_eligible(weight, scale, marker(convrot_groupsize=128))
    assert not q.int8_convrot_eligible(weight.float(), scale, marker()), "float weights are the fp8 path"
    assert not q.int8_convrot_eligible(weight, None, marker()), "no scale, no dequantization"
    assert not q.int8_convrot_eligible(weight, scale[:4], marker()), "one scale per output channel"
    # K has to be a whole number of ConvRot groups for the kernel's activation rotation
    odd = torch.zeros((OUT, IN + 128), dtype=torch.int8)
    assert not q.int8_convrot_eligible(odd, scale, marker())


def test_int8_layer_loads_packed_and_runs_the_int8_kernel(ops, monkeypatch):
    state, weight = int8_state()
    linear, missing, unexpected = load(ops, state)
    assert not missing and not unexpected
    assert linear.is_quantized and q.is_feather_int8_weight(linear.weight)
    assert linear.weight.shape == (OUT, IN)
    assert linear.weight._qdata.dtype == torch.int8, "int8 storage, bf16 logical dtype"

    calls = []
    real = q.int8_scaled_mm

    def spy(*args, **kwargs):
        calls.append(args[0].dtype)
        return real(*args, **kwargs)

    monkeypatch.setattr(q, "int8_scaled_mm", spy)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        out = linear(x)
        ref = torch.nn.functional.linear(x, weight.cuda().to(DTYPE), linear.bias)
    assert calls == [torch.int8], "the dispatcher must pick the int8 kernel"
    assert rel(out, ref) < 2e-2, "int8 ConvRot round trip of the weight"


def test_int8_layer_keeps_the_checkpoint_bytes(ops):
    """`state_dict_tensors` has to reproduce what the exporter wrote, not a re-quantization."""
    state, _ = int8_state()
    weight = q.FeatherInt8ConvRotPackedLayout
    packed = q.make_feather_int8_weight(state["weight"], state["weight_scale"], DTYPE)
    restored = weight.state_dict_tensors(packed._qdata, packed._params)
    assert torch.equal(restored[""], state["weight"])
    assert torch.equal(restored["_scale"].reshape(-1), state["weight_scale"].reshape(-1))


def test_int8_requantize_round_trips_the_layout(ops):
    """A patch merges in float, so `quantize` has to rotate-then-quantize and `dequantize` the inverse."""
    state, weight = int8_state()
    packed = q.make_feather_int8_weight(state["weight"], state["weight_scale"], DTYPE)
    with torch.no_grad():
        logical = packed.dequantize()
        assert rel(logical, weight) < 2e-2, "un-rotation has to give back the released weight"
        again = q.requantize_feather_weight(packed, logical)
    assert q.is_feather_int8_weight(again)
    assert rel(again.dequantize(), logical) < 2e-2, "requantizing a dequantized weight is stable"


def test_patch_on_int8_layer_stays_int8(ops):
    state, weight = int8_state()
    linear, _, _ = load(ops, state)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    up = torch.randn(OUT, RANK) * 0.02
    down = torch.randn(RANK, IN) * 0.02
    adapter = LoRAAdapter({"lin.weight"}, (up, down, torch.tensor(32.0), None, None, None))
    patches = {"lin.weight": [(1.0, adapter, 1.0, None, None)]}
    with torch.no_grad():
        model = torch.nn.Module()
        model.lin = linear
        base = linear(x)
        patcher = comfy.model_patcher.ModelPatcher(model, load_device="cuda", offload_device="cpu")
        patcher.add_patches({"lin.weight": adapter}, 1.0)
        patcher.patch_weight_to_device("lin.weight")
        assert q.is_feather_int8_weight(linear.weight), "set_weight must repack into the same layout"
        out = linear(x)
        ref = torch.nn.functional.linear(
            x, comfy.lora.calculate_weight(patches["lin.weight"], weight.cuda().float(), "lin.weight", intermediate_dtype=torch.float32).to(DTYPE), linear.bias
        )
    assert rel(out, base) > 5e-2, "the LoRA has to move the output"
    assert rel(out, ref) < 3e-2


def test_int8_state_dict_survives_inference_mode(ops):
    """Same inference-mode constraint as the fp8 layout, and the rotation marker has to be saved."""
    state, _ = int8_state()
    linear, _, _ = load(ops, state)
    x = torch.randn(64, IN, device="cuda", dtype=DTYPE)
    with torch.inference_mode():
        linear(x)
        saved = linear.state_dict()
    assert set(saved) == {"weight", "weight_scale", "bias", "comfy_quant"}
    assert torch.equal(saved["weight"].cpu(), state["weight"]), "unprepacked, still in the rotated basis"
    reloaded, _, unexpected = load(ops, {"weight": saved["weight"], "weight_scale": saved["weight_scale"], "comfy_quant": saved["comfy_quant"]})
    assert not unexpected
    assert q.is_feather_int8_weight(reloaded.weight), "the marker keeps a saved int8 model on the int8 kernel"


def test_float_layer_of_an_int8_checkpoint_takes_the_fp8_path(ops):
    """A mixed checkpoint: what the exporter kept in float follows the recipe, i.e. fp8e5m2."""
    int8_part, _ = int8_state()
    linear, _, _ = load(ops, int8_part)
    assert q.is_feather_int8_weight(linear.weight)

    float_linear = ops.Linear(IN, OUT, bias=True).cuda()
    float_state = {"weight": torch.randn(OUT, IN) * 0.05, "bias": torch.randn(OUT) * 0.1}
    float_linear = float_linear.cuda()
    float_linear._load_from_state_dict(float_state, "", {}, False, [], [], [])
    assert float_linear.is_quantized and q.is_feather_fp8_weight(float_linear.weight)

    excluded = feather_ops(policy=None, out_dtype=DTYPE).Linear(IN, OUT, bias=True).cuda()
    excluded._load_from_state_dict({"weight": torch.randn(OUT, IN) * 0.05, "bias": None}, "", {}, False, [], [], [])
    assert not excluded.is_quantized, "a recipe exclusion keeps the layer in full precision"


def test_comfy_quant_marker_is_consumed_not_leaked(ops):
    state, _ = int8_state()
    linear, _, unexpected = load(ops, state)
    assert not unexpected, "the marker must be popped from the state dict"
    assert q.is_feather_int8_weight(linear.weight)
