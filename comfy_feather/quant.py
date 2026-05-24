from dataclasses import dataclass, fields

import comfy.float
import torch
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_class

FEATHER_FP8E5M2_LAYOUT = "FeatherFP8E5M2PackedLayout"


# (N, K) -> (K/16, N, 16)
def prepack_transpose(x):
    N, K = x.shape
    return x.view(N, K // 16, 16).permute(1, 0, 2).contiguous()


# (K/16, N, 16) -> (N, K)
def unprepack_transpose(x):
    k, N, _ = x.shape
    return x.permute(1, 0, 2).reshape(N, k * 16)


@dataclass(frozen=True)
class FeatherFP8E5M2PackedParams:
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: tuple[int, ...]

    def _tensor_fields(self):
        return ["scale"]

    def to_device(self, device):
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        for field in self._tensor_fields():
            kwargs[field] = kwargs[field].to(device=device)
        return type(self)(**kwargs)

    def clone(self):
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        for field in self._tensor_fields():
            kwargs[field] = kwargs[field].clone()
        return type(self)(**kwargs)

    def copy_from(self, src, non_blocking=False):
        for field in fields(self):
            src_value = getattr(src, field.name)
            if field.name in self._tensor_fields():
                getattr(self, field.name).copy_(src_value, non_blocking=non_blocking)
            else:
                object.__setattr__(self, field.name, src_value)


def quantize_fp8e5m2_scaled(x, scale="recalculate", seed=None, inplace_ops=False):
    if scale is None or scale == "recalculate":
        scale = torch.amax(x.abs()).to(dtype=torch.float32) / torch.finfo(torch.float8_e5m2).max
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


class FeatherFP8E5M2PackedLayout(QuantizedLayout):
    Params = FeatherFP8E5M2PackedParams

    @classmethod
    def quantize(cls, tensor, scale="recalculate", stochastic_rounding=0, inplace_ops=False):
        if tensor.ndim != 2:
            raise ValueError(f"Feather FP8 packed weights require a 2D tensor, got {tensor.ndim}D")
        if tensor.shape[1] % 16 != 0:
            raise ValueError(f"Feather FP8 packed weights require K divisible by 16, got {tensor.shape}")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        qdata, scale = quantize_fp8e5m2_scaled(
            tensor,
            scale=scale,
            seed=stochastic_rounding,
            inplace_ops=inplace_ops,
        )
        params = cls.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape)
        return prepack_transpose(qdata), params

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


def make_feather_quantized_weight(weight, scale, orig_dtype):
    if weight.ndim != 2:
        raise ValueError(f"Feather FP8 packed weights require a 2D tensor, got {weight.ndim}D")

    if scale is None:
        scale = torch.ones((), device=weight.device, dtype=torch.float32)
    else:
        scale = scale.to(device=weight.device, dtype=torch.float32)

    qdata = prepack_transpose(weight.to(torch.float8_e5m2))
    params = FeatherFP8E5M2PackedLayout.Params(
        scale=scale,
        orig_dtype=orig_dtype,
        orig_shape=tuple(weight.shape),
    )
    return QuantizedTensor(qdata, FEATHER_FP8E5M2_LAYOUT, params)


def quantize_feather_weight(weight, seed=None, inplace_ops=False):
    qdata, params = FeatherFP8E5M2PackedLayout.quantize(
        weight,
        scale="recalculate",
        stochastic_rounding=seed,
        inplace_ops=inplace_ops,
    )
    return QuantizedTensor(qdata, FEATHER_FP8E5M2_LAYOUT, params)


def is_feather_quantized_weight(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == FEATHER_FP8E5M2_LAYOUT


def get_feather_plain_tensors(weight):
    if not is_feather_quantized_weight(weight):
        raise TypeError(f"Expected {FEATHER_FP8E5M2_LAYOUT}, got {type(weight).__name__}")
    return FeatherFP8E5M2PackedLayout.get_plain_tensors(weight)


register_layout_class(FEATHER_FP8E5M2_LAYOUT, FeatherFP8E5M2PackedLayout)
