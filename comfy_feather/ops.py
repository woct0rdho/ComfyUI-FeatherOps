import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from torch import nn

from ..kernel.hip.hip_kernel_prepacked import scaled_mm_hip_prepacked

DEBUG = False


def check_tensor(x, name):
    if not DEBUG:
        return
    if torch.isnan(x).any():
        raise RuntimeError(f"nan {name}")
    if torch.isinf(x).any():
        raise RuntimeError(f"inf {name}")


def stat_tensor(x, name):
    if not DEBUG:
        return
    dtype = x.dtype
    x = x.float()
    print(f"{name} {tuple(x.shape)} {dtype} {x.mean():.3g} {x.std():.3g} {x.abs().max():.3g}")


# (N, K) -> (K/16, N, 16)
def prepack_transpose(x):
    N, K = x.shape
    return x.view(N, K // 16, 16).permute(1, 0, 2).contiguous()


# (K/16, N, 16) -> (N, K)
def unprepack_transpose(x):
    k, N, _ = x.shape
    return x.permute(1, 0, 2).reshape(N, k * 16)


class FeatherOps(manual_cast):
    excluded_names = []

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_quantized = False
            self.weight_scale = None
            self.lora_A = None
            self.lora_B = None
            self.lora_alpha = None
            self.prefix = None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.prefix = prefix

            weight_key = prefix + "weight"
            bias_key = prefix + "bias"
            scale_key = prefix + "weight_scale"

            weight = state_dict.pop(weight_key, None)
            bias = state_dict.pop(bias_key, None)
            weight_scale = state_dict.pop(scale_key, None)

            if weight is not None:
                is_excluded = any(x in prefix for x in FeatherOps.excluded_names)
                is_dim1 = self.in_features == 1 or self.out_features == 1 or weight.ndim == 1
                # The kernel requires the inner dimension (which is in_features) to be a multiple of 16
                if is_excluded or is_dim1 or self.in_features % 16 != 0:
                    if self.in_features % 16 != 0:
                        print(f"Warning: Not prepacked {weight_key} {tuple(weight.shape)}")
                    self.is_quantized = False
                    self.weight = nn.Parameter(weight, requires_grad=False)
                else:
                    self.is_quantized = True

                    # For now we only support fp8e5m2 weight
                    weight = weight.to(torch.float8_e5m2)
                    check_tensor(weight, weight_key)
                    weight = prepack_transpose(weight)
                    self.weight = nn.Parameter(weight, requires_grad=False)

                if weight_scale is not None:
                    self.weight_scale = weight_scale
            else:
                missing_keys.append(weight_key)

            if bias is not None:
                self.bias = nn.Parameter(bias, requires_grad=False)
            else:
                self.bias = None

        def convert_weight(self, weight, inplace=False, **kwargs):
            if not self.is_quantized:
                return weight
            return unprepack_transpose(weight)

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            if not self.is_quantized:
                weight = weight.to(self.weight.dtype)
            else:
                # For now we only support fp8e5m2 weight
                weight = weight.to(torch.float8_e5m2)
                check_tensor(weight, self.prefix + "weight")
                weight = prepack_transpose(weight)

            if return_weight:
                return weight

            assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            run_every_op()

            if not self.is_quantized:
                check_tensor(x, self.prefix + "x in forward")
                stat_tensor(x, self.prefix + "x")
                weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                y = F.linear(x, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                # TODO: Check whether we need to implement lora here
                # We should not use the quantized linear op for lora
                check_tensor(y, self.prefix + "y in forward")
                stat_tensor(y, self.prefix + "y")
                return y

            # For now we only support fp16 x
            check_tensor(x, self.prefix + "x in forward before conversion to fp16")
            x_dtype_orig = x.dtype
            x = x.to(torch.float16)
            check_tensor(x, self.prefix + "x in forward after conversion to fp16")
            stat_tensor(x, self.prefix + "x")

            weight, bias, offload_stream = cast_bias_weight(self, x, dtype=self.weight.dtype, bias_dtype=torch.bfloat16, offloadable=True)
            scale = self.weight_scale.to(device=x.device, dtype=torch.bfloat16) if self.weight_scale is not None else None

            x_shape_orig = x.shape
            x = x.view(-1, x_shape_orig[-1])

            M = x.shape[0]
            pad_len = (16 - (M % 16)) % 16
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))

            y = scaled_mm_hip_prepacked(x, weight, scale, bias, out_dtype=torch.bfloat16)

            if pad_len > 0:
                x = x[:M, :]
                y = y[:M, :]

            check_tensor(y, self.prefix + "y in forward")
            stat_tensor(y, self.prefix + "y")

            uncast_bias_weight(self, weight, bias, offload_stream)

            y = y.to(x_dtype_orig)

            if self.lora_A is not None and self.lora_B is not None:
                lora_A = self.lora_A.to(x.device)
                lora_B = self.lora_B.to(x.device)
                lora_x = F.linear(x.to(lora_A.dtype), lora_A)
                lora_y = F.linear(lora_x, lora_B)
                if self.lora_alpha is not None:
                    lora_alpha = self.lora_alpha.to(x.device)
                    lora_y = lora_y * lora_alpha
                y = y + lora_y.to(y.dtype)

            return y.view(*x_shape_orig[:-1], y.shape[-1])
