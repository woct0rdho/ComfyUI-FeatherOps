import folder_paths
from comfy.sd import load_diffusion_model

from .ops import FeatherOps


class FeatherUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"

    def load_unet(self, unet_name):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model_options = {"custom_operations": FeatherOps}
        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
