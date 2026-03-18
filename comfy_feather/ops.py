import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from torch import nn

from ..kernel.hip.hip_kernel_prepacked import scaled_mm_hip_prepacked


# (N, K) -> (K/16, N, 16)
def prepack_transpose(x):
    N, K = x.shape
    return x.view(N, K // 16, 16).permute(1, 0, 2).contiguous()


# (K/16, N, 16) -> (N, K)
def unprepack_transpose(x):
    k, N = x.shape
    return x.permute(1, 0, 2).reshape(N, k * 16)


class FeatherOps(manual_cast):
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_quantized = False
            self.weight_scale = None
            self.lora_A = None
            self.lora_B = None
            self.lora_alpha = None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"
            bias_key = prefix + "bias"
            scale_key = prefix + "weight_scale"

            weight = state_dict.pop(weight_key, None)
            bias = state_dict.pop(bias_key, None)
            weight_scale = state_dict.pop(scale_key, None)

            if weight is not None:
                is_dim1 = self.in_features == 1 or self.out_features == 1 or weight.ndim == 1
                # The kernel requires the inner dimension (which is in_features) to be a multiple of 16
                if is_dim1 or self.in_features % 16 != 0:
                    if self.in_features % 16 != 0:
                        print(f"Not prepacked {weight_key} {tuple(weight.shape)}")
                    self.is_quantized = False
                    self.weight = nn.Parameter(weight, requires_grad=False)
                else:
                    self.is_quantized = True

                    # For now we only support fp8e5m2 weight
                    weight = weight.to(torch.float8_e5m2)

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
                weight = weight.to(torch.float8_e5m2)
                weight = prepack_transpose(weight)

            if return_weight:
                return weight

            assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            run_every_op()

            if not self.is_quantized:
                weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                out = F.linear(x, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                # TODO: Check whether we need to implement lora here
                # We should not use the quantized linear op for lora
                return out

            # For now we only support fp16 x
            x = x.to(torch.float16)
            out_dtype = torch.float16

            weight, bias, offload_stream = cast_bias_weight(self, x, dtype=self.weight.dtype, bias_dtype=x.dtype, offloadable=True)
            scale = self.weight_scale.to(x.device) if self.weight_scale is not None else None

            x_shape = x.shape
            x = x.view(-1, x_shape[-1])

            M = x.shape[0]
            pad_len = (16 - (M % 16)) % 16
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))

            y = scaled_mm_hip_prepacked(x, weight, scale, bias, out_dtype)

            if pad_len > 0:
                x = x[:M, :]
                y = y[:M, :]

            uncast_bias_weight(self, weight, bias, offload_stream)

            if self.lora_A is not None and self.lora_B is not None:
                lora_A = self.lora_A.to(x.device)
                lora_B = self.lora_B.to(x.device)
                lora_x = F.linear(x.to(lora_A.dtype), lora_A)
                lora_y = F.linear(lora_x, lora_B)
                if self.lora_alpha is not None:
                    lora_alpha = self.lora_alpha.to(x.device)
                    lora_y = lora_y * lora_alpha
                y = y + lora_y.to(y.dtype)

            return y.view(*x_shape[:-1], y.shape[-1])
