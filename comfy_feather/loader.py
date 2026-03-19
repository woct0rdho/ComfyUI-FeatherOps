import folder_paths
from comfy.sd import load_diffusion_model

from .debug_ops import DebugOps
from .ops import FeatherOps


class FeatherUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "ops": (["feather", "debug"],),
                "model_type": (["qwen"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"

    def load_unet(self, unet_name, ops, model_type):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        if model_type == "qwen":
            FeatherOps.excluded_names = [
                "time_text_embed",
                "img_in",
                "norm_out",
                "proj_out",
                "txt_in",
            ]
        else:
            FeatherOps.excluded_names = []

        if ops == "feather":
            model_options = {"custom_operations": FeatherOps}
        else:
            model_options = {"custom_operations": DebugOps}

        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
