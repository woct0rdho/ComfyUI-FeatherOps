import torch.nn.functional as F


def apply_lora_patches(x, y, weight_functions):
    for patch_fn in weight_functions:
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

            lora_scale = strength_patch * strength_model
            if weights[2] is not None:
                rank = lora_A.shape[0]
                lora_scale *= weights[2] / rank

            lora_A = lora_A.to(device=x.device, dtype=x.dtype)
            lora_B = lora_B.to(device=x.device, dtype=x.dtype)

            temp = F.linear(x, lora_A)
            temp = F.linear(temp, lora_B)
            y += temp * lora_scale

    return y
