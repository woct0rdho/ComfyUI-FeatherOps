from .loader import FeatherCLIPTextEncodePadded, FeatherUNetLoader
from .preview import patch_taehv_preview

patch_taehv_preview()

NODE_CLASS_MAPPINGS = {
    "FeatherUNetLoader": FeatherUNetLoader,
    "FeatherCLIPTextEncodePadded": FeatherCLIPTextEncodePadded,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatherUNetLoader": "Load Diffusion Model (Feather)",
    "FeatherCLIPTextEncodePadded": "Text Encode Padded (Feather)",
}
