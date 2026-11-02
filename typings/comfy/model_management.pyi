from collections.abc import Sequence
from typing import Protocol

import torch

class _LoadedModel:
    model: object | None

class _FreeMemory(Protocol):
    def __call__(self, memory_required: object, device: object, keep_loaded: list[object] | None = None, *args: object, **kwargs: object) -> object: ...

current_loaded_models: Sequence[_LoadedModel]
free_memory: _FreeMemory

def get_torch_device() -> torch.device: ...
def unet_dtype(
    device: torch.device | None = None, model_params: int = 0, supported_dtypes: Sequence[torch.dtype] = ..., weight_dtype: torch.dtype | None = None
) -> torch.dtype: ...
def unet_manual_cast(weight_dtype: torch.dtype, inference_device: torch.device, supported_dtypes: Sequence[torch.dtype] = ...) -> torch.dtype | None: ...
