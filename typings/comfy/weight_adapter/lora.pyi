from typing import Any

import torch

from . import WeightAdapterBase

class LoRAAdapter(WeightAdapterBase):
    name: str
    loaded_keys: set[str]
    weights: tuple[torch.Tensor, ...]
    def __init__(self, loaded_keys: set[str], weights: tuple[Any, ...]) -> None: ...
