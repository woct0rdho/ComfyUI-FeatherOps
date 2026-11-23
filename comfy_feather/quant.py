"""The two packed weight formats the kernels consume, and the `F.linear` that runs them.

Both are `comfy.quant_ops.QuantizedLayout`s, so ComfyUI's own plumbing drives them: `cast_bias_weight`
dequantizes a patched weight, applies every LoRA/LyCORIS/DoRA patch in float, and requantizes through
the layout's `quantize`:
- `FeatherFP8E5M2PackedLayout`: per-tensor scale, `(K/16, N, 16)` prepack for `scaled_mm_hip`, which
  takes bf16/fp16 activations against fp8e5m2 weights. A checkpoint in any float format is cast to it.
  Quantizing and dequantizing are `comfy.quant_ops.TensorCoreFP8E5M2Layout`'s own methods, so a
  re-packed weight is byte-identical to what a native fp8 model would hold. Only the storage order
  (and the prepack that follows) is this layout's.
- `FeatherInt8ConvRotPackedLayout`: the comfy-quants int8 ConvRot form, i.e. group-256 regular-Hadamard
  rotated rows, per-output-channel int8 scales, `(K/4, N/4, 4, 4)` micro-tile prepack for
  `int8_scaled_mm`, which takes int8 activations (rotated and per-row quantized by the kernel) against
  int8 weights. An int8 checkpoint already stores exactly this, so loading it is a byte permutation.
  Quantizing, dequantizing and the ConvRot flags are `comfy.quant_ops.TensorWiseINT8Layout`'s, which is
  what makes a LoRA-merged weight come back byte-identical to the native path (dithering included).

Both layouts store the kernel's order, and their `state_dict` keeps it: the weight entry is the resident
`_qdata` and the `comfy_quant` marker says so (`PACKED_STATE_DICT`), which is what makes a `state_dict()`
free - ComfyUI takes one for every memory estimate and every LoRA key scan. Permuting a weight back to
the released order is a full-weight copy, so only `dequantize` does it.
"""

import functools
import json
import logging
import math
from typing import cast

import comfy.float
import comfy.model_management
import comfy.ops
import torch
from comfy.quant_ops import TensorCoreFP8E5M2Layout, TensorWiseINT8Layout
from comfy_kitchen.tensor import BaseLayoutParams, QuantizedLayout, QuantizedTensor, register_layout_class
from feather_ops import convrot_int8_linear, prepack_int8_weight, scaled_mm_hip, unpack_int8_weight
from feather_ops.int8_convrot import CONVROT_GROUP, convrot_int8_supported

logger = logging.getLogger(__name__)

FEATHER_FP8E5M2_LAYOUT = "FeatherFP8E5M2PackedLayout"
FEATHER_INT8_LAYOUT = "FeatherInt8ConvRotPackedLayout"
FEATHER_LAYOUTS = (FEATHER_FP8E5M2_LAYOUT, FEATHER_INT8_LAYOUT)

# ConvRot's group size. The int8 kernel rotates activations with this block-diagonal Hadamard, and
# comfy-quants rotates the weights with the same one, so the two bases only match at this size.
assert CONVROT_GROUP == 256


# (N, K) -> (K/16, N, 16)
def prepack_transpose(x):
    N, K = x.shape
    # Zero-pad K to a multiple of 128 so the kernel's last K chunk reads zeros, not past the weight.
    K_pad = -(-K // 128) * 128
    if K_pad != K:
        padded = torch.zeros((N, K_pad), device=x.device, dtype=x.dtype)
        padded[:, :K] = x
        x = padded
    return x.view(N, K_pad // 16, 16).permute(1, 0, 2).contiguous()


# (K/16, N, 16) -> (N, K)
def unprepack_transpose(x):
    k, N, _ = x.shape
    return x.permute(1, 0, 2).reshape(N, k * 16)


class FeatherFP8E5M2PackedLayout(QuantizedLayout):
    """ComfyUI's fp8e5m2 layout, stored in `scaled_mm_hip`'s `(K/16, N, 16)` order.

    Quantizing and dequantizing are `TensorCoreFP8E5M2Layout`'s, the same code a native fp8 model runs,
    so the bytes (and the scale, and the stochastic-rounding behaviour under it) are the same ones
    ComfyUI would produce. Only the storage order is this layout's, and `state_dict` keeps that order
    (see `packed_marker`): permuting it back is a full-weight copy, and every `state_dict()` ComfyUI
    takes for an estimate or a LoRA key scan would pay it.
    """

    Params = BaseLayoutParams

    @classmethod
    def quantize(cls, tensor, scale="recalculate", stochastic_rounding=0, inplace_ops=False, **kwargs):
        if tensor.ndim != 2:
            raise ValueError(f"Feather FP8 packed weights require a 2D tensor, got {tensor.ndim}D")
        if tensor.shape[1] % 16 != 0:
            raise ValueError(f"Feather FP8 packed weights require K divisible by 16, got {tensor.shape}")
        qdata, comfy_params = TensorCoreFP8E5M2Layout.quantize(tensor, scale=scale, stochastic_rounding=stochastic_rounding, inplace_ops=inplace_ops)
        return prepack_transpose(qdata), cls.Params(scale=comfy_params.scale, orig_dtype=tensor.dtype, orig_shape=tuple(tensor.shape))

    @classmethod
    def dequantize(cls, qdata, params):
        return TensorCoreFP8E5M2Layout.dequantize(unprepack_transpose(qdata), params)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # The resident bytes, which is what `packed_marker` announces: no permutation, so no copy.
        return {
            "": qdata,
            "_scale": params.scale,
        }


class FeatherInt8ConvRotPackedLayout(QuantizedLayout):
    """ComfyUI's int8 ConvRot layout, stored in the 4x4 micro-tile order both GEMMs read.

    The ConvRot flags, the quantizer and the dequantizer are `TensorWiseINT8Layout`'s. Only the storage
    order is this layout's.
    """

    Params = TensorWiseINT8Layout.Params

    @classmethod
    def quantize(cls, tensor, scale="recalculate", stochastic_rounding=0, inplace_ops=False, **kwargs):
        """Quantize with ComfyUI's own ConvRot quantizer, so re-packed bytes match the native path.

        `TensorWiseINT8Layout.quantize` is the function `comfy.ops.MixedPrecisionOps` reaches through
        `requantize_from_float`, so a weight patched here comes out byte-identical to the same weight
        patched by a native int8 model. That matters beyond consistency: it dithers when
        `stochastic_rounding` is set, and a LoRA delta smaller than one int8 step (which is the normal
        case for a rank-16..64 LoRA on a per-row int8 weight) survives dithering but not round-to-nearest.
        """
        if tensor.ndim != 2:
            raise ValueError(f"Feather int8 packed weights require a 2D tensor, got {tensor.ndim}D")
        n, k = tensor.shape
        ok, reason = convrot_int8_supported((n, k), CONVROT_GROUP, backward=False)
        if not ok:
            raise ValueError(f"Feather int8 packed weights cannot hold {tuple(tensor.shape)}: {reason}")
        qdata, comfy_params = TensorWiseINT8Layout.quantize(
            tensor,
            scale=scale,
            stochastic_rounding=stochastic_rounding,
            is_weight=kwargs.get("is_weight", True),
            per_channel=kwargs.get("per_channel", True),
            convrot=kwargs.get("convrot", True),
            convrot_groupsize=kwargs.get("convrot_groupsize", CONVROT_GROUP),
        )
        if not (comfy_params.convrot and comfy_params.convrot_groupsize == CONVROT_GROUP):
            raise RuntimeError(f"comfy_kitchen requantized without the ConvRot rotation this layout stores: {comfy_params}")
        # Per-output-channel scale as comfy_kitchen returns it, `[N, 1]`: its dequantize broadcasts the
        # scale against the row, and `copy_from` compares shapes.
        return prepack_int8_weight(qdata), cls.Params(
            scale=comfy_params.scale.reshape(-1, 1),
            orig_dtype=tensor.dtype,
            orig_shape=tuple(tensor.shape),
            is_weight=True,
            convrot=True,
            convrot_groupsize=CONVROT_GROUP,
        )

    @classmethod
    def dequantize(cls, qdata, params):
        return TensorWiseINT8Layout.dequantize(unpack_int8_weight(qdata), params)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # The resident bytes, which is what `packed_marker` announces: no permutation, so no copy. The
        # scale keeps the `[N, 1]` shape comfy-quants writes.
        return {
            "": packed_state_view(qdata, params.orig_shape),
            "_scale": params.scale.reshape(-1, 1),
        }


def make_feather_quantized_weight(weight, scale, orig_dtype):
    if weight.ndim != 2:
        raise ValueError(f"Feather FP8 packed weights require a 2D tensor, got {weight.ndim}D")

    if scale is None:
        scale = torch.ones((), device=weight.device, dtype=torch.float32)
    else:
        scale = scale.to(device=weight.device, dtype=torch.float32)

    if weight.dtype == torch.float8_e5m2:
        qdata = prepack_transpose(weight.to(torch.float8_e5m2))
    else:
        qdata, params = FeatherFP8E5M2PackedLayout.quantize(weight.to(torch.float32) * scale, scale="recalculate")
        scale = params.scale

    params = BaseLayoutParams(scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(weight.shape))
    return QuantizedTensor(qdata, FEATHER_FP8E5M2_LAYOUT, params)


def make_feather_int8_weight(weight, scale, orig_dtype):
    """Wrap an int8 ConvRot weight that is already rotated, as a comfy-quants checkpoint stores it."""
    if weight.dtype != torch.int8:
        raise TypeError(f"Feather int8 packed weights require int8, got {weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"Feather int8 packed weights require a 2D tensor, got {weight.ndim}D")
    scale = scale.reshape(-1, 1).to(device=weight.device, dtype=torch.float32)
    params = FeatherInt8ConvRotPackedLayout.Params(
        scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(weight.shape), is_weight=True, convrot=True, convrot_groupsize=CONVROT_GROUP
    )
    return QuantizedTensor(prepack_int8_weight(weight), FEATHER_INT8_LAYOUT, params)


def packed_state_view(qdata: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    """`qdata` as the state dict states it: the weight's own `(N, K)` when that is a free view.

    The bytes stay in the kernel's order (the `packed` marker says so), but the shape is what other
    code reads: `comfy.lora.model_lora_keys_unet` splits a fused qkv LoRA by `state_dict()[k].shape[0] // 3`
    and the safetensors header is built from these shapes, so exposing `(K/4, N/4, 4, 4)` there sent a
    third of the rows to the LoRA. A padded fp8 pack (`K_pad > K`) cannot state `(N, K)`. Its marker
    carries the logical shape instead.
    """
    n, k = int(orig_shape[0]), int(orig_shape[1])
    return qdata.reshape(n, k) if qdata.numel() == n * k else qdata


def fp8_packed_shape(orig_shape: tuple[int, int]) -> tuple[int, int, int]:
    """The `(K_pad/16, N, 16)` storage shape this layout holds for an `(N, K)` weight."""
    n, k = orig_shape
    return -(-k // 128) * 128 // 16, n, 16


def int8_packed_shape(orig_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The `(K/4, N/4, 4, 4)` storage shape this layout holds for an `(N, K)` weight."""
    n, k = orig_shape
    return k // 4, n // 4, 4, 4


def make_feather_packed_weight(qdata, scale, orig_dtype, orig_shape):
    """Adopt a state-dict entry `state_dict_tensors` wrote, i.e. bytes already in the kernel's order.

    This is the reload side of `packed_marker`: the permutation is skipped, so a model saved here loads
    without the prepack pass. The storage shape is checked against `orig_shape` rather than trusted,
    because this is the one direction the kernels cannot check: `scaled_mm` accepts a prepack at least
    `K/16` deep (that is what makes the zero-padded K work), so a weight of the wrong depth would be
    multiplied by a partial contraction instead of failing.
    """
    n, k = int(orig_shape[0]), int(orig_shape[1])
    if qdata.dtype == torch.int8:
        expected = int8_packed_shape((n, k))
        if tuple(qdata.shape) != expected:
            # A state dict states the weight's own `(N, K)`. The bytes are the same, so the storage
            # shape is a free view away. Anything else is a real mismatch and must not slip through.
            if qdata.numel() != math.prod(expected):
                raise ValueError(f"int8 packed weight of {tuple(qdata.shape)} does not hold a ({n}, {k}) weight, which is {expected}")
            qdata = qdata.reshape(expected)
        scale = scale.reshape(-1, 1).to(device=qdata.device, dtype=torch.float32)
        params = FeatherInt8ConvRotPackedLayout.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=(n, k), is_weight=True, convrot=True, convrot_groupsize=CONVROT_GROUP)
        return QuantizedTensor(qdata, FEATHER_INT8_LAYOUT, params)

    expected = fp8_packed_shape((n, k))
    if qdata.dtype == torch.float8_e5m2 and tuple(qdata.shape) != expected and qdata.numel() == math.prod(expected):
        qdata = qdata.reshape(expected)
    if qdata.dtype != torch.float8_e5m2 or tuple(qdata.shape) != expected:
        raise ValueError(f"fp8 packed weight of {qdata.dtype} {tuple(qdata.shape)} does not hold a ({n}, {k}) weight, which is {expected}")
    if scale is None:
        scale = torch.ones((), device=qdata.device, dtype=torch.float32)
    else:
        scale = scale.to(device=qdata.device, dtype=torch.float32)
    params = BaseLayoutParams(scale=scale, orig_dtype=orig_dtype, orig_shape=(n, k))
    return QuantizedTensor(qdata, FEATHER_FP8E5M2_LAYOUT, params)


@functools.lru_cache(maxsize=32)
def _quant_marker(raw: bytes) -> dict:
    return json.loads(raw)


# `comfy_quant` key announcing that `weight` is in this layout's storage order instead of the released
# one. Everything this loader writes carries it (see `quant_marker`), so a reload adopts the bytes and a
# `state_dict()` costs no copy. A comfy-quants checkpoint, which stores released rows, has no marker
# value for it and is prepacked on load as before.
PACKED_STATE_DICT = "packed"


def packed_marker(raw: bytes | None) -> bool:
    """Whether a `comfy_quant` payload says its weight is already in the kernel's storage order."""
    return bool(raw) and bool(_quant_marker(raw).get(PACKED_STATE_DICT, False))


def marker_shape(raw: bytes | None) -> tuple[int, int] | None:
    """The `(N, K)` a packed entry stands for, as its marker records it, or None if it does not say.

    The storage shape is not the logical one - int8 is `(K/4, N/4, 4, 4)` and fp8 pads K to 128 - and
    model detection reads weight shapes, so the logical shape has to travel with the file. This is also
    the only exact source for a padded fp8 K: `(K_pad/16, N, 16)` cannot tell you how much of the 128
    block is padding.
    """
    if raw is None:
        return None
    shape = _quant_marker(raw).get("shape")
    return None if shape is None else (int(shape[0]), int(shape[1]))


def int8_convrot_eligible(weight, scale, marker: bytes | None) -> bool:
    """Whether a state-dict entry can drive the int8 kernel as stored.

    The weights have to be int8 with one scale per output channel, and the checkpoint's `comfy_quant`
    marker has to say the rows are ConvRot-rotated at the group size the kernel implements: an
    unrotated int8 weight would be multiplied by Hadamard-rotated activations, which is a wrong answer
    rather than a slower one.
    """
    if weight.dtype != torch.int8 or scale is None or weight.ndim != 2 or marker is None:
        return False
    n, k = weight.shape
    if not convrot_int8_supported((n, k), CONVROT_GROUP, backward=False)[0] or scale.numel() != n:
        return False
    conf = _quant_marker(marker)
    return bool(conf.get("convrot", False)) and int(conf.get("convrot_groupsize", 0)) == CONVROT_GROUP


def dequantize_stored_weight(weight: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """The float value of a stored weight. A checkpoint scale, if any, is applied.

    Scales come as a scalar (per tensor), `[N]` or `[N, 1]` (per output channel). A per-row int8
    checkpoint uses the last two.
    """
    if scale is None:
        return weight.to(dtype)
    scale = scale.reshape(-1)
    factor = scale if scale.numel() == 1 else scale.reshape(-1, 1)
    return (weight.to(torch.float32) * factor).to(dtype)


class _QuantEntryModule(torch.nn.Module):
    """Stand-in module for ComfyUI's quantized state-dict loader, which reads these off the module."""

    def __init__(self, orig_shape, device, dtype):
        super().__init__()
        self._orig_shape = orig_shape
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self._disabled_formats = ()
        self._full_precision_mm = False
        self._full_precision_mm_config = False


def _comfy_quantized_entry(state_dict: dict, prefix: str, orig_shape: tuple[int, int], dtype: torch.dtype):
    """Build ComfyUI's own `QuantizedTensor` for a packed checkpoint entry, consuming its keys.

    nvfp4, mxfp8, int4 and grouped int8 entries are byte codes plus several scale tensors, and the
    mapping from those keys to layout parameters lives in `comfy.ops._load_quantized_module`. A
    throwaway module gives that function what it wants, so the table stays ComfyUI's instead of being
    copied here. If it ever moves, this raises instead of computing something plausible but wrong.
    """
    loader = getattr(comfy.ops, "_load_quantized_module", None)
    if loader is None:
        raise RuntimeError(f"cannot read the packed weight {prefix}weight: this ComfyUI has no quantized state-dict loader")
    shim = _QuantEntryModule(orig_shape, state_dict[prefix + "weight"].device, dtype)
    loader(shim, lambda *args, **kwargs: None, state_dict, prefix, {}, False, [], [], [])
    return shim.weight


def take_stored_weight(state_dict: dict, prefix: str, orig_shape: tuple[int, int], dtype: torch.dtype):
    """Pop `{prefix}weight` (with its scale and `comfy_quant` marker) and return `(weight, scale, marker)`.

    Three shapes of entry, and the caller only has to look at the dtype to tell the first two apart:
    - a float weight, returned as stored with its scale (e5m2 stays e5m2 so it can be prepacked without
      being quantized twice).
    - an int8 weight or this loader's own packed entry, returned as stored for the same reason.
    - packed codes (nvfp4 / mxfp8 / int4 / grouped int8), which are dequantized to float through
      ComfyUI's own quantized loader, since the key-to-params table for those is its, not ours.
    """
    weight = state_dict.get(prefix + "weight")
    if weight is None:
        state_dict.pop(prefix + "weight", None)
        return None, None, None

    scale = state_dict.get(prefix + "weight_scale")
    if scale is None:
        scale = state_dict.get(prefix + "scale_weight")
    marker = state_dict.get(prefix + "comfy_quant")
    marker_bytes = bytes(marker.cpu().tolist()) if marker is not None else None
    if marker_shape(marker_bytes) is not None and not packed_marker(marker_bytes):
        # The two fields state the same thing - the entry is in this layout's storage order - so a file
        # carrying one without the other cannot be read: adopting it would permute the packed bytes a
        # second time, and reading it as released rows would multiply by a permutation of the weight.
        # This loader is the only writer of `shape`, so the combination only appears if a file was edited.
        raise ValueError(f"{prefix}comfy_quant records a logical shape without the packed flag, so the stored byte order is ambiguous")

    # A packed entry of our own is not a 2D ConvRot weight, so it must not go through the eligibility
    # test (which reads the shape as `(N, K)`). `packed_marker` is what tells the loader to adopt it.
    as_stored = marker_bytes is None or packed_marker(marker_bytes) or int8_convrot_eligible(weight, scale, marker_bytes)
    if torch.is_floating_point(weight) or (weight.dtype == torch.int8 and as_stored):
        for key in (prefix + "weight", prefix + "weight_scale", prefix + "scale_weight", prefix + "comfy_quant"):
            state_dict.pop(key, None)
        return weight, scale, marker_bytes

    if marker is None:
        raise ValueError(f"{prefix}weight is packed {weight.dtype} with no comfy_quant marker, so its format is unknowable")
    packed = _comfy_quantized_entry(state_dict, prefix, orig_shape, dtype)
    return packed.dequantize().to(dtype), None, marker_bytes


def packed_state_dict(weight: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """`{key}` and `{key}_scale` for a packed weight, without detaching the wrapper or copying the bytes.

    `key` is the weight's own state-dict key (`...weight`), which is how the layout's suffixes line up with
    what `_load_from_state_dict` reads. `QuantizedTensor.detach()` builds a new wrapper, which torch
    refuses once the tensor is an inference tensor. The layout's own tensors are what a state dict wants,
    and they are the resident ones: `state_dict_tensors` hands back `_qdata` itself, so a
    `state_dict()` here is a dict of views and costs nothing to build.
    """
    return cast(QuantizedTensor, weight).state_dict(key)


def quant_marker(weight: torch.Tensor) -> torch.Tensor:
    """The `comfy_quant` entry for a packed weight: json bytes in a uint8 tensor, as ComfyUI writes it.

    Saving one of these keeps a reload on the same layout instead of re-deriving it: an fp8 checkpoint is
    read back as fp8, and an int8 ConvRot one keeps its rotation marker (without it the rows could not be
    told apart from unrotated int8, which the kernel must not run).

    `PACKED_STATE_DICT` says the weight is in this layout's storage order, and every entry this loader
    writes carries it: `state_dict_tensors` keeps the resident bytes rather than permuting them back, so
    a reload has to adopt them instead of prepacking them a second time. A checkpoint from comfy-quants
    or a ComfyUI fp8 export has no such flag and takes the released path. `shape` is the logical
    `(N, K)` of those bytes, which the storage shape no longer states.
    """
    conf: dict[str, object]
    if is_feather_int8_weight(weight):
        conf = {"format": "int8_tensorwise", "per_row": True, "convrot": True, "convrot_groupsize": CONVROT_GROUP}
    else:
        conf = {"format": "float8_e5m2"}
    conf[PACKED_STATE_DICT] = True
    conf["shape"] = [int(x) for x in weight.shape]
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def is_feather_fp8_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == FEATHER_FP8E5M2_LAYOUT


def is_feather_int8_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == FEATHER_INT8_LAYOUT


def is_feather_packed_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls in FEATHER_LAYOUTS


def get_feather_plain_tensors(weight):
    if not is_feather_fp8_weight(weight):
        raise TypeError(f"Expected {FEATHER_FP8E5M2_LAYOUT}, got {type(weight).__name__}")
    return weight.layout_cls.get_plain_tensors(weight)


def requantize_after_patch(weight, device) -> bool:
    """Whether a patched weight goes back into `weight`'s own packed format, as ComfyUI's Linear does.

    Weight-only formats (int8 ConvRot) are requantized and keep their quantized GEMM on every device, so
    a patched int8 weight is byte-identical to the native path's and the int8 kernel stays in use.

    A format that also quantizes the activation (fp8e5m2/e4m3) is only requantized where ComfyUI can run
    the fp8 GEMM. Everywhere else ComfyUI marks the layer `full_precision_matrix_mult` and multiplies by
    the merged float weight, and following it is also the accurate choice: rounding a merged weight
    back to e5m2 (two mantissa bits) replaces it with dithered fp8 noise that is far larger than the LoRA
    delta it was meant to add.
    """
    if is_feather_fp8_weight(weight):
        return comfy.model_management.supports_fp8_compute(device)
    return True


def feather_linear(x, weight, bias, out_dtype):
    """`x @ weight.T + bias` on a Feather-packed weight, with the layout picking the kernel."""
    if is_feather_int8_weight(weight):
        return convrot_int8_linear(x, weight._qdata.reshape(-1), weight._params.scale, bias, out_dtype, weight_packed=True)
    plain, scale = get_feather_plain_tensors(weight)
    return scaled_mm_hip(x, plain, scale.to(device=x.device, dtype=torch.float32), bias, out_dtype=out_dtype)


register_layout_class(FEATHER_FP8E5M2_LAYOUT, FeatherFP8E5M2PackedLayout)
register_layout_class(FEATHER_INT8_LAYOUT, FeatherInt8ConvRotPackedLayout)
