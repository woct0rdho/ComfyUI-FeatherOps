import folder_paths
import torch
from comfy.sd import load_diffusion_model

from .ops import feather_ops
from .sampling import install_sampling_model_pin


class FeatherUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (["qwen", "wan", "default"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        if model_type == "qwen":
            excluded_names = [
                # Non-repeating modules
                "time_text_embed",
                "img_in",
                "norm_out",
                "proj_out",
                "txt_in",
                # Modules with time embedding vector as input
                "img_mod",
                "txt_mod",
            ]
            out_dtype = torch.bfloat16
        elif model_type == "wan":
            excluded_names = [
                # Non-repeating modules
                "patch_embedding",
                "text_embedding",
                "time_embedding",
                "time_projection",
                "head",
            ]
            out_dtype = torch.float16
        else:
            excluded_names = []
            out_dtype = torch.bfloat16

        model_options = {"custom_operations": feather_ops(out_dtype=out_dtype, excluded_names=excluded_names)}
        model = load_diffusion_model(unet_path, model_options=model_options)
        install_sampling_model_pin(model)
        return (model,)
