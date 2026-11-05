"""The two packed weight formats the kernels consume, and the `F.linear` that runs them.

Both are `comfy.quant_ops.QuantizedLayout`s, so ComfyUI's own plumbing drives them: `cast_bias_weight`
dequantizes a patched weight, applies every LoRA/LyCORIS/DoRA patch in float, and requantizes through
the layout's `quantize`:
- `FeatherFP8E5M2PackedLayout`: per-tensor scale, `(K/16, N, 16)` prepack for `scaled_mm_hip`, which
  takes bf16/fp16 activations against fp8e5m2 weights. A checkpoint in any float format is cast to it.
- `FeatherInt8ConvRotPackedLayout`: the comfy-quants int8 ConvRot form, i.e. group-256 regular-Hadamard
  rotated rows, per-output-channel int8 scales, `(K/4, N/4, 4, 4)` micro-tile prepack for
  `int8_scaled_mm`, which takes int8 activations (rotated and per-row quantized by the kernel) against
  int8 weights. An int8 checkpoint already stores exactly this, so loading it is a byte permutation.
"""

import functools
import json
import logging
from typing import cast

import comfy.float
import comfy.ops
import torch
from comfy_kitchen.tensor import BaseLayoutParams, QuantizedLayout, QuantizedTensor, register_layout_class
from comfy_quants.formats.convrot import build_hadamard, rotate_weight
from feather_ops import convrot_quantize_activation, int8_scaled_mm, prepack_int8_convrot_weight, scaled_mm_hip, unpack_int8_convrot_weight

logger = logging.getLogger(__name__)

FEATHER_FP8E5M2_LAYOUT = "FeatherFP8E5M2PackedLayout"
FEATHER_INT8_LAYOUT = "FeatherInt8ConvRotPackedLayout"
FEATHER_LAYOUTS = (FEATHER_FP8E5M2_LAYOUT, FEATHER_INT8_LAYOUT)

# ConvRot's group size. The int8 kernel rotates activations with this block-diagonal Hadamard, and
# comfy-quants rotates the weights with the same one, so the two bases only match at this size.
CONVROT_GROUP = 256


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


def quantize_fp8e5m2_scaled(x, scale="recalculate", seed=None, inplace_ops=False):
    if scale is None or scale == "recalculate":
        scale = torch.amax(x.abs()).to(dtype=torch.float32) / torch.finfo(torch.float8_e5m2).max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        if x.dtype not in {torch.float32, torch.bfloat16}:
            dtype_info = torch.finfo(x.dtype)
            scale = 1.0 / torch.clamp(1.0 / scale, min=dtype_info.min, max=dtype_info.max)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale = scale.to(device=x.device, dtype=torch.float32)

    inv_scale = (1.0 / scale).to(dtype=x.dtype)
    if inplace_ops:
        x *= inv_scale
    else:
        x = x * inv_scale

    if seed is not None and seed > 0:
        x = comfy.float.stochastic_rounding(x, dtype=torch.float8_e5m2, seed=seed)
    else:
        x = x.to(torch.float8_e5m2)
    return x, scale.float()


def _rotate(x, group=CONVROT_GROUP, dtype=torch.float32):
    """`x @ H^T` per group-256 block. H is symmetric and orthogonal, so this also un-rotates."""
    return rotate_weight(x, build_hadamard(group, device=x.device, dtype=dtype), group)


def quantize_int8_convrot_rows(x, scale="recalculate"):
    """Rotate group-256 blocks of `x` and quantize per row, the way comfy-quants stores int8 weights.

    Rows are scaled by `absmax / 127` and rounded to nearest. The rotation happens in the input dtype and
    the division in float32, which is what makes a re-pack of a loaded checkpoint re-derive its own bytes.
    """
    rotated = _rotate(x, dtype=x.dtype)
    if scale is None or scale == "recalculate":
        scale = (rotated.abs().amax(dim=1).float() / 127.0).clamp(min=1e-30)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale = scale.reshape(-1).to(device=x.device, dtype=torch.float32)
    q = torch.clamp(torch.round(rotated.float() / scale[:, None]), -127, 127).to(torch.int8)
    return q, scale


class FeatherFP8E5M2PackedLayout(QuantizedLayout):
    Params = BaseLayoutParams

    @classmethod
    def quantize(cls, tensor, scale="recalculate", stochastic_rounding=0, inplace_ops=False, **kwargs):
        if tensor.ndim != 2:
            raise ValueError(f"Feather FP8 packed weights require a 2D tensor, got {tensor.ndim}D")
        if tensor.shape[1] % 16 != 0:
            raise ValueError(f"Feather FP8 packed weights require K divisible by 16, got {tensor.shape}")

        orig_dtype, orig_shape = tensor.dtype, tuple(tensor.shape)
        qdata, scale = quantize_fp8e5m2_scaled(tensor, scale=scale, seed=stochastic_rounding, inplace_ops=inplace_ops)
        return prepack_transpose(qdata), cls.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape)

    @classmethod
    def dequantize(cls, qdata, params):
        weight = unprepack_transpose(qdata)
        scale = params.scale.to(device=weight.device, dtype=torch.float32)
        return (weight.to(torch.float32) * scale).to(params.orig_dtype)

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
    Params = BaseLayoutParams

    @classmethod
    def quantize(cls, tensor, scale="recalculate", stochastic_rounding=0, inplace_ops=False, **kwargs):
        if tensor.ndim != 2:
            raise ValueError(f"Feather int8 packed weights require a 2D tensor, got {tensor.ndim}D")
        n, k = tensor.shape
        if k % CONVROT_GROUP or n % 4 or k % 4:
            raise ValueError(f"Feather int8 packed weights require K divisible by {CONVROT_GROUP} and both axes by 4, got {tuple(tensor.shape)}")
        qdata, scale = quantize_int8_convrot_rows(tensor, scale=scale)
        return prepack_int8_convrot_weight(qdata), cls.Params(scale=scale, orig_dtype=tensor.dtype, orig_shape=tuple(tensor.shape))

    @classmethod
    def dequantize(cls, qdata, params):
        weight = unpack_int8_convrot_weight(qdata).float() * params.scale.reshape(-1, 1).to(device=qdata.device, dtype=torch.float32)
        return _rotate(weight).to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        # The released form, which is what comfy-quants writes: rotated int8 rows and a [N, 1] scale.
        return {
            "": unpack_int8_convrot_weight(qdata),
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
        qdata, scale = quantize_fp8e5m2_scaled(weight.to(torch.float32) * scale)
        qdata = prepack_transpose(qdata)

    params = BaseLayoutParams(scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(weight.shape))
    return QuantizedTensor(qdata, FEATHER_FP8E5M2_LAYOUT, params)


def make_feather_int8_weight(weight, scale, orig_dtype):
    """Wrap an int8 ConvRot weight that is already rotated, as a comfy-quants checkpoint stores it."""
    if weight.dtype != torch.int8:
        raise TypeError(f"Feather int8 packed weights require int8, got {weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"Feather int8 packed weights require a 2D tensor, got {weight.ndim}D")
    scale = scale.reshape(-1).to(device=weight.device, dtype=torch.float32)
    params = BaseLayoutParams(scale=scale, orig_dtype=orig_dtype, orig_shape=tuple(weight.shape))
    return QuantizedTensor(prepack_int8_convrot_weight(weight), FEATHER_INT8_LAYOUT, params)


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
    if n % 4 or k % 4 or k % CONVROT_GROUP or scale.numel() != n:
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


def requantize_feather_weight(old: torch.Tensor, weight: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Re-derive packed bytes for `weight` with the layout `old` already uses.

    This is the hook ComfyUI calls when it bakes a patch: the fp8 pack re-quantizes per tensor, the int8
    pack re-rotates per group-256 block and quantizes per output channel.
    """
    return cast(QuantizedTensor, old).requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True)


def is_feather_fp8_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == FEATHER_FP8E5M2_LAYOUT


def is_feather_int8_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == FEATHER_INT8_LAYOUT


def is_feather_packed_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls in FEATHER_LAYOUTS


def get_feather_plain_tensors(weight):
    if not is_feather_fp8_weight(weight):
        raise TypeError(f"Expected {FEATHER_FP8E5M2_LAYOUT}, got {type(weight).__name__}")
    return FeatherFP8E5M2PackedLayout.get_plain_tensors(weight)


def feather_linear(x, weight, bias, out_dtype):
    """`x @ weight.T + bias` on a Feather-packed weight, with the layout picking the kernel."""
    if is_feather_int8_weight(weight):
        a, a_scale = convrot_quantize_activation(x)
        return int8_scaled_mm(a, weight._qdata.reshape(-1), a_scale, weight._params.scale, bias, out_dtype=out_dtype)
    plain, scale = get_feather_plain_tensors(weight)
    return scaled_mm_hip(x, plain, scale.to(device=x.device, dtype=torch.float32), bias, out_dtype=out_dtype)


register_layout_class(FEATHER_FP8E5M2_LAYOUT, FeatherFP8E5M2PackedLayout)
register_layout_class(FEATHER_INT8_LAYOUT, FeatherInt8ConvRotPackedLayout)
