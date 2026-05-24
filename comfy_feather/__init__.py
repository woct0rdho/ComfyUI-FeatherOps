from .loader import FeatherCLIPTextEncodePadded, FeatherUNetLoader

NODE_CLASS_MAPPINGS = {
    "FeatherUNetLoader": FeatherUNetLoader,
    "FeatherCLIPTextEncodePadded": FeatherCLIPTextEncodePadded,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatherUNetLoader": "Load Diffusion Model (Feather)",
    "FeatherCLIPTextEncodePadded": "Text Encode Padded (Feather)",
}
