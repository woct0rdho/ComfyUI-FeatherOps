from .comfy_feather.loader import FeatherUNetLoader  # ty: ignore[import-not-found]
from .comfy_feather.vae import FeatherVAELoader  # ty: ignore[import-not-found]

NODE_CLASS_MAPPINGS = {
    "FeatherUNetLoader": FeatherUNetLoader,
    "FeatherVAELoader": FeatherVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatherUNetLoader": "Load Diffusion Model (Feather)",
    "FeatherVAELoader": "Load VAE (Feather)",
}
