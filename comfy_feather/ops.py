import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from torch import nn

from ..kernel.hip.hip_kernel import scaled_mm_hip
from .lora import apply_lora_patches
from .quant import get_feather_plain_tensors, is_feather_quantized_weight, make_feather_quantized_weight, quantize_feather_weight, unprepack_transpose


class FeatherOps(manual_cast):
    excluded_names = []
    out_dtype = torch.bfloat16

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_quantized = False

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"
            bias_key = prefix + "bias"
            scale_key = prefix + "weight_scale"
            scale_key_legacy = prefix + "scale_weight"

            weight = state_dict.pop(weight_key, None)
            bias = state_dict.pop(bias_key, None)
            weight_scale = state_dict.pop(scale_key, None)
            if weight_scale is None:
                weight_scale = state_dict.pop(scale_key_legacy, None)

            if weight is not None:
                is_excluded = any(x in prefix for x in FeatherOps.excluded_names)
                is_dim1 = self.in_features == 1 or self.out_features == 1 or weight.ndim == 1
                # The kernel requires in_features to be a multiple of 16
                if is_excluded or is_dim1 or self.in_features % 16 != 0:
                    if self.in_features % 16 != 0:
                        print(f"Warning: Not prepacked {weight_key} {tuple(weight.shape)}")
                    self.is_quantized = False
                    self.weight = nn.Parameter(weight, requires_grad=False)
                else:
                    self.is_quantized = True
                    self.weight = nn.Parameter(
                        make_feather_quantized_weight(weight, weight_scale, FeatherOps.out_dtype),
                        requires_grad=False,
                    )
            else:
                missing_keys.append(weight_key)

            if bias is not None:
                self.bias = nn.Parameter(bias.to(FeatherOps.out_dtype), requires_grad=False)
            else:
                self.bias = None

        def convert_weight(self, weight, inplace=False, **kwargs):
            if not self.is_quantized:
                return weight

            if is_feather_quantized_weight(weight):
                return weight.dequantize()

            # Backward-compatible fallback for older in-memory packed weights
            weight = unprepack_transpose(weight)
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            if not self.is_quantized:
                weight = weight.to(self.weight.dtype)
            else:
                weight = quantize_feather_weight(weight, seed=seed, inplace_ops=True)

            if return_weight:
                return weight

            assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            run_every_op()

            if not self.is_quantized:
                weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                y = F.linear(x, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return y

            x_shape_orig = x.shape
            x = x.reshape(-1, x_shape_orig[-1])

            # The kernel requires fp16 x and produces the configured output dtype
            x_fp16 = x.to(torch.float16)

            # Temporarily clear weight_function so cast_bias_weight does not apply low-VRAM patches to prepacked weight
            saved_weight_function = self.weight_function
            self.weight_function = []

            bias_dtype = self.bias.dtype if self.bias is not None else None
            weight, bias, offload_stream = cast_bias_weight(self, x, dtype=self.weight.dtype, bias_dtype=bias_dtype, offloadable=True)
            weight, scale = get_feather_plain_tensors(weight)
            scale = scale.to(device=x.device, dtype=torch.float32)

            y = scaled_mm_hip(x_fp16, weight, scale, bias, out_dtype=FeatherOps.out_dtype)

            uncast_bias_weight(self, weight, bias, offload_stream)

            y = y.to(x.dtype)

            self.weight_function = saved_weight_function
            y = apply_lora_patches(x, y, saved_weight_function)

            return y.view(*x_shape_orig[:-1], y.shape[-1])
