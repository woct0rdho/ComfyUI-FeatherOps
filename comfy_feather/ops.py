import fnmatch
from typing import cast

import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor
from torch import nn

from .quant import (
    dequantize_stored_weight,
    feather_linear,
    int8_convrot_eligible,
    is_feather_packed_weight,
    make_feather_int8_weight,
    make_feather_packed_weight,
    make_feather_quantized_weight,
    packed_marker,
    packed_state_dict,
    quant_marker,
    requantize_after_patch,
    take_stored_weight,
)


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

        class Linear(manual_cast.Linear):
            # `comfy.ops.linear_input_act`/`linear()` fold an activation into a generic GEMM by taking
            # over the weight cast themselves. That skips this class's routing, and for a patched fp8
            # weight it requantizes where ComfyUI's own Linear does not (Wan's FFN down-projection is
            # written that way). Declaring the module's own forward as the only entry point makes
            # ComfyUI apply the activation eagerly and then run this forward, which is where the
            # layout decides between its kernel and a float GEMM.
            comfy_force_forward = True

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
                weight, stored_scale, marker = take_stored_weight(state_dict, prefix, (self.out_features, self.in_features), self.out_dtype)
                bias = state_dict.pop(prefix + "bias", None)

                if weight is None:
                    missing_keys.append(weight_key)
                else:
                    selected = self._selected(prefix.rstrip("."))
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight.ndim == 1
                    if packed_marker(marker):
                        # This loader's own state dict: the bytes are already in the kernel's order, so
                        # they are adopted as they are. Permuting them again would be a full-weight copy
                        # on every reload, and for fp8 it would also re-pad the K the file already holds.
                        packed = make_feather_packed_weight(weight, stored_scale, self.out_dtype, (self.out_features, self.in_features))
                    elif int8_convrot_eligible(weight, stored_scale, marker):
                        # Already int8 ConvRot: prepack the stored rows, never quantize them again.
                        packed = make_feather_int8_weight(weight, stored_scale, self.out_dtype)
                    elif weight.dtype == torch.float8_e5m2 and weight.ndim == 2 and self.in_features % 16 == 0 and (stored_scale is None or stored_scale.numel() == 1):
                        # Already fp8e5m2: prepack the stored bytes with their own scale. Re-quantizing
                        # would only add error, so the recipe does not get a say here either. The fp8
                        # layout carries one scale for the whole tensor, so a file with a scale per
                        # output channel goes through `dequantize_stored_weight` below instead.
                        packed = make_feather_quantized_weight(weight, stored_scale, self.out_dtype)
                    elif selected and not is_dim1 and self.in_features % 16 == 0:
                        # A float checkpoint: the comfy-quants recipe decides, and the cast is ours.
                        packed = make_feather_quantized_weight(weight, stored_scale, self.out_dtype)
                    else:
                        # Deliberately unquantized (the recipe excludes it, or the checkpoint kept it in
                        # float): keep it in float, with any checkpoint scale applied.
                        if selected and self.in_features % 16 != 0:
                            print(f"Warning: Not prepacked {weight_key} {tuple(weight.shape)}")
                        weight = dequantize_stored_weight(weight, stored_scale, self.out_dtype)
                        packed = None

                    self.is_quantized = packed is not None
                    self.weight = nn.Parameter(packed if packed is not None else weight, requires_grad=False)

                if bias is not None:
                    self.bias = nn.Parameter(bias.to(self.out_dtype), requires_grad=False)
                else:
                    self.bias = None

            def state_dict(self, destination=None, prefix="", keep_vars=False):
                """The released form of a packed weight, not its storage.

                Overriding `_save_to_state_dict` is not enough: after it runs, `nn.Module.state_dict`
                writes every parameter again from its generic loop, and detaching a `QuantizedTensor`
                unwraps it to the storage (the `(K/4, N/4, 4, 4)` micro-tiles or the fp8
                `(K_pad/16, N, 16)` order). ComfyUI reads these shapes - `comfy.lora.model_lora_keys_unet`
                derives the fused-qkv split of a MOSS/Flux3 file from `state_dict()[...].shape[0] // 3`,
                which sent a third of the qkv rows to the LoRA - and the exporter builds the safetensors
                header from them, so they have to be the `[N, K]` form the layout's `state_dict_tensors`
                writes. That is also why the layout tensors are returned as views rather than a copy.
                """
                if not self.is_quantized:
                    if destination is None:
                        return super().state_dict(prefix=prefix, keep_vars=keep_vars)
                    return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
                destination = {} if destination is None else destination
                destination.update(packed_state_dict(self.weight, prefix + "weight"))
                destination[prefix + "comfy_quant"] = quant_marker(self.weight)
                if self.bias is not None:
                    destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
                return destination

            def convert_weight(self, weight, inplace=False, **kwargs):
                if not self.is_quantized:
                    return weight
                if not is_feather_packed_weight(weight):
                    raise TypeError(f"Expected a Feather packed weight, got {type(weight).__name__}")
                return weight.dequantize()

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                # Layout-agnostic: each pack re-derives its own bytes from the float weight through the
                # layout's own `quantize`, and the int8 one re-rotates and re-scales as it does.
                if self.is_quantized:
                    weight = cast(QuantizedTensor, self.weight).requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True)
                else:
                    weight = weight.to(self.weight.dtype)

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
                # `quantize`. Without it the patch would be dropped on the offloaded path. With it for a
                # format ComfyUI itself does not requantize here, we would answer a different question
                # than the native loader (see `quant.requantize_after_patch`).
                lazy_patch = getattr(self, "weight_lowvram_function", None) is not None
                requant = lazy_patch and requantize_after_patch(self.weight, x.device)
                bias_dtype = self.bias.dtype if self.bias is not None else None
                weight, bias, offload_stream = cast_bias_weight(self, x, dtype=self.weight.dtype, bias_dtype=bias_dtype, offloadable=True, want_requant=requant)

                if is_feather_packed_weight(weight):
                    # `F.linear` cannot reach this kernel: torch lowers `linear` to `t` + `addmm`, and a
                    # transposed packed weight has no packed form (kitchen's 1:1 layouts can, the
                    # packed ones register `aten.t` to flag it). So the kernel is called here.
                    y = feather_linear(x, weight, bias, self.out_dtype)
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
