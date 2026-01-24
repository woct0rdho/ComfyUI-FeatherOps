import torch
from comfy.ops import (
    CastWeightBiasOp,
    cast_bias_weight,
    manual_cast,
    run_every_op,
    uncast_bias_weight,
)

from .kernel.kernel import scaled_mm


def easy_mixed_precision_ops(compute_dtype=torch.bfloat16):
    class EasyMixedPrecisionOps(manual_cast):
        _compute_dtype = compute_dtype

        class Linear(torch.nn.Module, CastWeightBiasOp):
            def __init__(
                self,
                in_features: int,
                out_features: int,
                bias: bool = True,
                device=None,
                dtype=None,
            ) -> None:
                super().__init__()

                factory_kwargs = {"device": device, "dtype": EasyMixedPrecisionOps._compute_dtype}
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
                self.weight_scale = torch.nn.Parameter(torch.tensor(1.0, **factory_kwargs))
                self.has_weight_scale = False
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
                else:
                    self.register_parameter("bias", None)

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight = state_dict.pop(prefix + "weight", None)
                if weight is not None:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

                weight_scale = state_dict.pop(prefix + "weight_scale", None)
                if weight_scale is None:
                    # Old version
                    weight_scale = state_dict.pop(prefix + "scale_weight", None)
                if weight_scale is not None:
                    self.weight_scale = torch.nn.Parameter(weight_scale.to(self.weight.dtype), requires_grad=False)
                    self.has_weight_scale = True
                else:
                    self.has_weight_scale = False

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                if weight is not None and prefix + "weight" in missing_keys:
                    missing_keys.remove(prefix + "weight")
                if prefix + "weight_scale" in missing_keys:
                    missing_keys.remove(prefix + "weight_scale")

                if prefix + "comfy_quant" in unexpected_keys:
                    unexpected_keys.remove(prefix + "comfy_quant")
                if prefix + "input_scale" in unexpected_keys:
                    unexpected_keys.remove(prefix + "input_scale")

            def forward_comfy_cast_weights(self, input):
                weight, bias, offload_stream = cast_bias_weight(self, input, dtype=self.weight.dtype, bias_dtype=None if self.bias is None else self.bias.dtype, offloadable=True)
                in_shape = input.shape[:-1]
                input = input.view(-1, input.shape[-1])
                output = scaled_mm(input, weight.T, self.weight_scale if self.has_weight_scale else None, self.bias, EasyMixedPrecisionOps._compute_dtype)
                output = output.view(in_shape + (output.shape[-1],))
                uncast_bias_weight(self, weight, bias, offload_stream)
                return output

            def forward(self, input):
                run_every_op()
                return self.forward_comfy_cast_weights(input)

    return EasyMixedPrecisionOps
