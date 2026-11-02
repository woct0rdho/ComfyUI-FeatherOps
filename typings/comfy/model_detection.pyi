from typing import Any

import torch

class ModelConfig:
    unet_config: dict[str, Any]
    supported_inference_dtypes: list[torch.dtype]
    quant_config: object | None

def unet_prefix_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str: ...
def model_config_from_unet(
    state_dict: dict[str, torch.Tensor], unet_key_prefix: str, use_base_if_no_match: bool = False, metadata: dict[str, Any] | None = None
) -> ModelConfig | None: ...
