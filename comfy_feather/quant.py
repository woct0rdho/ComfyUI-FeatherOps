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
"""

import functools
import json
import logging
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
    ComfyUI would produce. Only the storage order is this layout's.
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
        return {
            "": unprepack_transpose(qdata),
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
        # The released form, which is what comfy-quants writes: rotated int8 rows and a [N, 1] scale.
        return {
            "": unpack_int8_weight(qdata),
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


@functools.lru_cache(maxsize=32)
def _quant_marker(raw: bytes) -> dict:
    return json.loads(raw)


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
    - an int8 weight, returned as stored for the same reason.
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

    if torch.is_floating_point(weight) or (weight.dtype == torch.int8 and (marker_bytes is None or int8_convrot_eligible(weight, scale, marker_bytes))):
        for key in (prefix + "weight", prefix + "weight_scale", prefix + "scale_weight", prefix + "comfy_quant"):
            state_dict.pop(key, None)
        return weight, scale, marker_bytes

    if marker is None:
        raise ValueError(f"{prefix}weight is packed {weight.dtype} with no comfy_quant marker, so its format is unknowable")
    packed = _comfy_quantized_entry(state_dict, prefix, orig_shape, dtype)
    return packed.dequantize().to(dtype), None, marker_bytes


def packed_state_dict(weight: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """`{key}` and `{key}_scale` for a packed weight, without detaching the wrapper.

    `key` is the weight's own state-dict key (`...weight`), which is how the layout's suffixes line up with
    what `_load_from_state_dict` reads. `QuantizedTensor.detach()` builds a new wrapper, which torch
    refuses once the tensor is an inference tensor. The layout's own tensors are what a state dict wants.
    """
    return cast(QuantizedTensor, weight).state_dict(key)


def quant_marker(weight: torch.Tensor) -> torch.Tensor:
    """The `comfy_quant` entry for a packed weight: json bytes in a uint8 tensor, as ComfyUI writes it.

    Saving one of these keeps a reload on the same layout instead of re-deriving it: an fp8 checkpoint is
    read back as fp8, and an int8 ConvRot one keeps its rotation marker (without it the rows could not be
    told apart from unrotated int8, which the kernel must not run).
    """
    if is_feather_int8_weight(weight):
        conf = {"format": "int8_tensorwise", "per_row": True, "convrot": True, "convrot_groupsize": CONVROT_GROUP}
    else:
        conf = {"format": "float8_e5m2"}
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
