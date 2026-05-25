import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from torch import nn

from ..kernel.hip.hip_kernel import scaled_mm_hip
from .lora import apply_lora_patches
from .quant import get_feather_plain_tensors, is_feather_quantized_weight, make_feather_quantized_weight, quantize_feather_weight


def feather_ops(out_dtype=torch.bfloat16, excluded_names=()):
    excluded_names = tuple(excluded_names)

    class FeatherOps(manual_cast):
        _excluded_names = excluded_names
        _out_dtype = out_dtype

        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.excluded_names = FeatherOps._excluded_names
                self.out_dtype = FeatherOps._out_dtype
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
                    is_excluded = any(x in prefix for x in self.excluded_names)
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
                            make_feather_quantized_weight(weight, weight_scale, self.out_dtype),
                            requires_grad=False,
                        )
                else:
                    missing_keys.append(weight_key)

                if bias is not None:
                    self.bias = nn.Parameter(bias.to(self.out_dtype), requires_grad=False)
                else:
                    self.bias = None

            def convert_weight(self, weight, inplace=False, **kwargs):
                if not self.is_quantized:
                    return weight
                if not is_feather_quantized_weight(weight):
                    raise TypeError(f"Expected Feather quantized weight, got {type(weight).__name__}")
                return weight.dequantize()

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

                m = x.shape[0]
                if m == 0:
                    return x.new_empty(*x_shape_orig[:-1], self.out_features)

                if m > 256:
                    block_size = 128
                elif m > 128:
                    block_size = 64
                elif m > 64:
                    block_size = 32
                else:
                    block_size = 16

                m_padded = (m + block_size - 1) // block_size * block_size
                if m_padded != m:
                    x_padded = F.pad(x, (0, 0, 0, m_padded - m))
                else:
                    x_padded = x

                # The kernel requires fp16 x and produces the configured output dtype
                x_padded = x_padded.contiguous().to(torch.float16)

                # Temporarily clear weight_function so cast_bias_weight does not apply low-VRAM patches to prepacked weight
                saved_weight_function = self.weight_function
                self.weight_function = []

                bias_dtype = self.bias.dtype if self.bias is not None else None
                weight, bias, offload_stream = cast_bias_weight(self, x_padded, dtype=self.weight.dtype, bias_dtype=bias_dtype, offloadable=True)
                weight, scale = get_feather_plain_tensors(weight)
                scale = scale.to(device=x_padded.device, dtype=torch.float32)

                y = scaled_mm_hip(x_padded, weight, scale, bias, out_dtype=self.out_dtype)

                uncast_bias_weight(self, weight, bias, offload_stream)

                y = y.to(x.dtype)

                y = y[:m]

                self.weight_function = saved_weight_function
                y = apply_lora_patches(x, y, saved_weight_function)

                return y.view(*x_shape_orig[:-1], y.shape[-1])

    return FeatherOps
