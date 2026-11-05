from .comfy_feather.loader import FeatherUNetLoader  # ty: ignore[unresolved-import]
from .comfy_feather.vae import FeatherVAELoader  # ty: ignore[unresolved-import]

NODE_CLASS_MAPPINGS = {
    "FeatherUNetLoader": FeatherUNetLoader,
    "FeatherVAELoader": FeatherVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatherUNetLoader": "Load Diffusion Model (Feather)",
    "FeatherVAELoader": "Load VAE (Feather)",
}
