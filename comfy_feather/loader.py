import numbers

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
                "model_type": (["qwen", "default"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"

    def load_unet(self, unet_name, ops, model_type):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        if model_type == "qwen":
            FeatherOps.excluded_names = [
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
        else:
            FeatherOps.excluded_names = []

        if ops == "feather":
            model_options = {"custom_operations": FeatherOps}
        else:
            model_options = {"custom_operations": DebugOps}

        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class FeatherCLIPTextEncodePadded:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
                "multiplier": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, multiplier):
        tokens = clip.tokenize(text)
        empty_tokens = clip.tokenize("")

        for k in tokens:
            if k in empty_tokens and len(empty_tokens[k]) > 0 and len(empty_tokens[k][0]) > 0:
                pad_token = empty_tokens[k][0][-1]
            else:
                pad_token = (0, 1.0)

            for i in range(len(tokens[k])):
                current_len = len(tokens[k][i])

                # Fix for Qwen system prompt
                template_end = 0
                if k == "qwen25_7b":
                    count_im_start = 0
                    t_end = -1
                    for idx, v in enumerate(tokens[k][i]):
                        elem = v[0]
                        if isinstance(elem, numbers.Integral):
                            if elem == 151644 and count_im_start < 2:
                                t_end = idx
                                count_im_start += 1

                    if t_end != -1 and current_len > (t_end + 3):
                        if tokens[k][i][t_end + 1][0] == 872 and tokens[k][i][t_end + 2][0] == 198:
                            t_end += 3

                    if t_end != -1:
                        template_end = t_end

                effective_len = current_len - template_end
                remainder = effective_len % multiplier
                if remainder != 0:
                    pad_len = multiplier - remainder
                    tokens[k][i] = tokens[k][i] + [pad_token] * pad_len

        return (clip.encode_from_tokens_scheduled(tokens),)
