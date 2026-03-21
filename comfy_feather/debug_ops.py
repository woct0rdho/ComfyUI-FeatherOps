import torch
import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight


class DebugOps(manual_cast):
    class Linear(manual_cast.Linear):
        def forward(self, x):
            run_every_op()

            x_shape_orig = x.shape
            x = x.reshape(-1, x_shape_orig[-1])

            # Temporarily clear weight_function so cast_bias_weight does not apply patches to prepacked weight
            saved_weight_function = self.weight_function
            self.weight_function = []

            weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
            y = F.linear(x, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)

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
