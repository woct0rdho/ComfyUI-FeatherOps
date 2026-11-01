from .comfy_feather.loader import FeatherUNetLoader  # ty: ignore[unresolved-import]

NODE_CLASS_MAPPINGS = {
    "FeatherUNetLoader": FeatherUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatherUNetLoader": "Load Diffusion Model (Feather)",
}
