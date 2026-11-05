import logging

import comfy.model_detection
import comfy.model_management
import comfy.utils
import folder_paths
import torch
from comfy.sd import load_diffusion_model
from comfy_quants import list_model_adapters
from comfy_quants.registry.global_registry import registry

from .ops import feather_ops
from .sampling import install_sampling_model_pin

logger = logging.getLogger(__name__)

# The layer-selection recipe is comfy-quants' per-family policy, so the node offers
# exactly the families that package ships. "auto" resolves the family from the
# checkpoint with ComfyUI's own detector. "all" is the no-recipe escape hatch that
# quantizes every eligible Linear.
AUTO = "auto"
ALL = "all"
MODEL_TYPES = [AUTO, *list_model_adapters(), ALL]

# The Feather matmul kernels write bf16 or fp16.
OUT_DTYPES = (torch.bfloat16, torch.float16)


def _inspect_checkpoint(unet_path):
    """Checkpoint-key prefix, detected model config, and the dtype ComfyUI runs it in.

    Mirrors what `comfy.sd.load_diffusion_model_state_dict` does before it builds the
    module tree: strip the wrapper prefix, detect, then pick the compute dtype from the
    weight dtype and the model's supported dtypes. The prefix is needed to translate
    comfy-quants' checkpoint-key globs into module-name globs, and the dtype is what the
    matmul should write. Reading the file here also warms the page cache for the real load,
    which follows the same path.

    The last value says what the checkpoint already stores: `{"int8"}` and/or `{"fp8e5m2"}` when its
    weights are quantized (those layers are prepacked as they are), empty when it is a float model, in
    which case every layer the comfy-quants recipe selects is cast to fp8e5m2.
    """
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True, safe_load=True)
    stored = _stored_formats(sd)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True) or sd
    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
    if model_config is None:
        return prefix, None, torch.bfloat16, stored

    weight_dtype = comfy.utils.weight_dtype(sd)
    if model_config.quant_config is not None:
        weight_dtype = None
    supported = list(model_config.supported_inference_dtypes)
    out_dtype = comfy.model_management.unet_dtype(model_params=comfy.utils.calculate_parameters(sd), supported_dtypes=supported, weight_dtype=weight_dtype)
    out_dtype = comfy.model_management.unet_manual_cast(out_dtype, comfy.model_management.get_torch_device(), supported) or out_dtype
    return prefix, model_config, out_dtype if out_dtype in OUT_DTYPES else torch.bfloat16, stored


def _stored_formats(sd) -> frozenset[str]:
    """The quantized formats the file already stores, i.e. what must not be quantized again.

    int8 is the comfy-quants ConvRot export and fp8e5m2 is a pre-scaled fp8 checkpoint. Anything else
    (fp32/fp16/bf16/fp8e4m3/fp4) is a float model that this loader casts to fp8e5m2 itself.
    """
    formats = set()
    for key, value in sd.items():
        if key.endswith(".weight"):
            if value.dtype == torch.int8:
                formats.add("int8")
            elif value.dtype == torch.float8_e5m2:
                formats.add("fp8e5m2")
    return frozenset(formats)


def _strip_prefix(pattern, prefix):
    return pattern[len(prefix) :] if prefix and pattern.startswith(prefix) else pattern


def quant_policy(model_type, unet_path):
    """Layer selection for this checkpoint, as `(policy, out_dtype, stored)`.

    `policy` is `(include, exclude)` module-name globs, or None to keep every Linear
    in full precision. `out_dtype` is the dtype ComfyUI will run the checkpoint in,
    i.e. what the Feather matmul should write. `stored` is the set of formats the file
    already stores (`"int8"`, `"fp8e5m2"`), empty for a float model.
    """
    prefix, model_config, out_dtype, stored = _inspect_checkpoint(unet_path)

    if stored:
        # The checkpoint already carries its exporter's per-layer decision: the int8 and fp8e5m2 layers
        # are prepacked as they are, and the layers it kept in float are the ones it wanted in float.
        # Quantizing any of them again would only add error, so the recipe has nothing to say here.
        logger.info("FeatherUNetLoader: %s already stores %s, keeping its per-layer formats", unet_path, ", ".join(sorted(stored)))
        return None, out_dtype, stored

    family = model_type
    if model_type == AUTO:
        family = model_config.unet_config.get("image_model") if model_config is not None else None
        if family not in list_model_adapters():
            logger.warning(
                "FeatherUNetLoader: detected image_model=%r has no comfy-quants recipe. Loading in full precision. Pick the model_type explicitly to quantize (available: %s).",
                family,
                ", ".join(MODEL_TYPES),
            )
            return None, out_dtype, stored

    if family == ALL:
        return (("*",), ()), out_dtype, stored

    policy = registry.get_adapter(family).default_policy(target_dtype="fp8_e5m2")
    # The recipe's globs are checkpoint keys. Modules are named without the wrapper.
    include = tuple(_strip_prefix(p, prefix) for p in policy.include)
    # Components the recipe keeps at full precision, as subtree globs.
    exclude = tuple([_strip_prefix(p, prefix) for p in policy.exclude] + [f"{_strip_prefix(c, prefix)}.*" for c in policy.keep_components])
    return (include, exclude), out_dtype, stored


class FeatherUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (MODEL_TYPES,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        policy, out_dtype, stored = quant_policy(model_type, unet_path)
        if stored:
            kernel = f"{' + '.join(sorted(stored))} as stored"
        else:
            kernel = f"cast to fp8e5m2, out {str(out_dtype).removeprefix('torch.')}"
        logger.info("FeatherUNetLoader: %s -> %s", unet_name, kernel)
        model_options = {"custom_operations": feather_ops(policy=policy, out_dtype=out_dtype)}
        model = load_diffusion_model(unet_path, model_options=model_options)
        install_sampling_model_pin(model)
        return (model,)
