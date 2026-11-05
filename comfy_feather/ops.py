import fnmatch

import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from feather_ops import scaled_mm_hip
from torch import nn

from .quant import get_feather_plain_tensors, is_feather_quantized_weight, make_feather_quantized_weight, quantize_feather_weight


def feather_ops(out_dtype=torch.bfloat16, policy=None):
    """Feather operations with the layer selection taken from a comfy-quants policy.

    policy is `(include, exclude)` of module-name globs, or None to keep every
    Linear in full precision. A Linear is quantized when it matches include (or
    include is empty) and matches no exclude.
    """
    # No recipe means no quantization. "*" in exclude is how that is spelled here.
    include, exclude = policy if policy is not None else ((), ("*",))
    include = tuple(include)
    exclude = tuple(exclude)

    class FeatherOps(manual_cast):
        _include = include
        _exclude = exclude
        _out_dtype = out_dtype

        class Linear(manual_cast.Linear):  # ty: ignore[invalid-attribute-override]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.include = FeatherOps._include
                self.exclude = FeatherOps._exclude
                self.out_dtype = FeatherOps._out_dtype
                self.is_quantized = False

            def _selected(self, name):
                if self.include and not any(fnmatch.fnmatchcase(name, p) for p in self.include):
                    return False
                return not any(fnmatch.fnmatchcase(name, p) for p in self.exclude)

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
                    selected = self._selected(prefix.rstrip("."))
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight.ndim == 1
                    # The kernel requires in_features to be a multiple of 16
                    if not selected or is_dim1 or self.in_features % 16 != 0:
                        if selected and self.in_features % 16 != 0:
                            print(f"Warning: Not prepacked {weight_key} {tuple(weight.shape)}")
                        self.is_quantized = False
                        self.weight = nn.Parameter(weight, requires_grad=False)
                    else:
                        self.is_quantized = True
                        self.weight = nn.Parameter(make_feather_quantized_weight(weight, weight_scale, self.out_dtype), requires_grad=False)
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

            def _forward(self, x):
                dtype_orig = x.dtype
                x = x.contiguous()

                # Patch application is ComfyUI's, not ours: `cast_bias_weight` dequantizes, runs every
                # patch in `weight_lowvram_function`/`weight_function` (LoRA, LoHa, LoKr, DoRA, weight
                # diffs) and, with `want_requant`, repacks the patched weight through this layout's
                # `quantize` instead of handing back a bf16 tensor. That is the contract
                # `MixedPrecisionOps` in `comfy/ops.py` uses for its own quantized Linears. Without
                # `want_requant` the offloaded path would lose the pack, and without the patches the
                # LoRA would be silently dropped.
                lazy_patch = getattr(self, "weight_lowvram_function", None) is not None
                bias_dtype = self.bias.dtype if self.bias is not None else None
                weight, bias, offload_stream = cast_bias_weight(self, x, dtype=self.weight.dtype, bias_dtype=bias_dtype, offloadable=True, want_requant=lazy_patch)

                if is_feather_quantized_weight(weight):
                    plain, scale = get_feather_plain_tensors(weight)
                    scale = scale.to(device=x.device, dtype=torch.float32)
                    y = scaled_mm_hip(x, plain, scale, bias, out_dtype=self.out_dtype)
                else:
                    # A weight wrapper produced a plain tensor (ComfyUI does not requantize those), or
                    # the weight is a layout this kernel does not pack: let `F.linear` handle it.
                    y = F.linear(x, weight, bias)

                uncast_bias_weight(self, weight, bias, offload_stream)
                y = y.to(dtype_orig)
                return y

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # The parameter is named `input` for proper override from `nn.Linear`
                x = input
                del input

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

                y = self._forward(x)
                return y.view(*x_shape_orig[:-1], y.shape[-1])

    return FeatherOps
