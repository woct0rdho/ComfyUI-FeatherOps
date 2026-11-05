from typing import Any

class VAELoader:
    RETURN_TYPES: tuple[str, ...]
    FUNCTION: str
    CATEGORY: str
    SEARCH_ALIASES: tuple[str, ...]
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]: ...
    def load_vae(self, vae_name: str) -> tuple[Any]: ...
