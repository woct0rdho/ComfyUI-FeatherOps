import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from torch import nn

from ..kernel.hip.hip_kernel import scaled_mm_hip


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

                    # Currently the kernel only supports fp8e5m2 weight
                    weight = weight.to(torch.float8_e5m2)
                    weight = prepack_transpose(weight)
                    self.weight = nn.Parameter(weight, requires_grad=False)

                if weight_scale is not None:
                    # Currently the kernel only supports bf16 scale
                    self.weight_scale = weight_scale.to(torch.bfloat16)
            else:
                missing_keys.append(weight_key)

            if bias is not None:
                # Currently the kernel only supports bf16 bias
                self.bias = nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)
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
                # Currently the kernel only supports fp8e5m2 weight
                weight = weight.to(torch.float8_e5m2)
                weight = prepack_transpose(weight)

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

            # Currently the kernel only supports fp16 x and bf16 out
            x_fp16 = x.to(torch.float16)

            # Temporarily clear weight_function so cast_bias_weight does not apply patches to prepacked weight
            saved_weight_function = self.weight_function
            self.weight_function = []

            weight, bias, offload_stream = cast_bias_weight(self, dtype=self.weight.dtype, bias_dtype=self.bias.dtype, offloadable=True)
            scale = self.weight_scale.to(device=x.device) if self.weight_scale is not None else None

            y = scaled_mm_hip(x_fp16, weight, scale, bias, out_dtype=torch.bfloat16)

            uncast_bias_weight(self, weight, bias, offload_stream)

            y = y.to(x.dtype)

            # Apply LoRA
            self.weight_function = saved_weight_function
            for patch_fn in self.weight_function:
                if not (hasattr(patch_fn, "patches") and hasattr(patch_fn, "key")):
                    raise NotImplementedError("FeatherOps currently only supports basic LoRA")

                patches = patch_fn.patches.get(patch_fn.key, [])
                for patch_data in patches:
                    # patch_data: (strength_patch, adapter, strength_model, offset, function)
                    strength_patch = patch_data[0]
                    adapter = patch_data[1]
                    strength_model = patch_data[2]

                    if not hasattr(adapter, "weights") or adapter.weights is None:
                        raise NotImplementedError("FeatherOps currently only supports basic LoRA")

                    weights = adapter.weights
                    lora_B = weights[0]
                    lora_A = weights[1]
                    alpha = weights[2] if weights[2] is not None else 1

                    rank = lora_A.shape[0]
                    lora_scale = strength_patch * strength_model * (alpha / rank)

                    lora_A = lora_A.to(device=x.device, dtype=x.dtype)
                    lora_B = lora_B.to(device=x.device, dtype=x.dtype)

                    temp = F.linear(x, lora_A)
                    temp = F.linear(temp, lora_B)
                    y += temp * lora_scale

            return y.view(*x_shape_orig[:-1], y.shape[-1])
