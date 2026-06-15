from collections.abc import Sequence
from typing import Protocol

class _LoadedModel:
    model: object | None

class _FreeMemory(Protocol):
    def __call__(self, memory_required: object, device: object, keep_loaded: list[object] | None = None, *args: object, **kwargs: object) -> object: ...

current_loaded_models: Sequence[_LoadedModel]
free_memory: _FreeMemory
