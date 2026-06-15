from collections.abc import Callable

import torch
from torch import nn

_TensorTransform = Callable[[torch.Tensor], torch.Tensor]

class disable_weight_init:
    class Linear(nn.Linear):
        comfy_cast_weights: bool
        weight_function: list[_TensorTransform]
        bias_function: list[_TensorTransform]

class manual_cast(disable_weight_init): ...

def run_every_op() -> None: ...
def cast_bias_weight(
    s: object,
    input: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: object | None = None,
    bias_dtype: torch.dtype | None = None,
    offloadable: bool = False,
    compute_dtype: torch.dtype | None = None,
    want_requant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, object]: ...
def uncast_bias_weight(s: object, weight: torch.Tensor, bias: torch.Tensor | None, offload_stream: object) -> None: ...
