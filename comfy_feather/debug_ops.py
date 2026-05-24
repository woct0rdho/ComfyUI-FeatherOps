import torch.nn.functional as F
from comfy.ops import cast_bias_weight, manual_cast, run_every_op, uncast_bias_weight

from .lora import apply_lora_patches


class DebugOps(manual_cast):
    class Linear(manual_cast.Linear):
        def forward(self, x):
            run_every_op()

            x_shape_orig = x.shape
            x = x.reshape(-1, x_shape_orig[-1])

            # Temporarily clear weight_function so cast_bias_weight does not apply low-VRAM patches to prepacked weight
            saved_weight_function = self.weight_function
            self.weight_function = []

            weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
            y = F.linear(x, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)

            self.weight_function = saved_weight_function
            y = apply_lora_patches(x, y, saved_weight_function)

            return y.view(*x_shape_orig[:-1], y.shape[-1])
