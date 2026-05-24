# Keep the active Feather diffusion model loaded for the duration of sampling.
# Preview decoders such as TAESD can ask ComfyUI's model manager for VRAM mid-sample;
# without pinning, that request can unload/unpatch the currently sampling model and
# leave later denoise steps running with stale weights. This wrapper adds active
# Feather patchers to model_management.free_memory(..., keep_loaded=...) while the
# sampler is running, so ComfyUI can still free other models but not the active one.

import comfy.model_management
from comfy.patcher_extension import WrappersMP

_PINNED_PATCHERS = "_feather_sampling_pinned_patchers"
_ORIGINAL_FREE_MEMORY = "_feather_original_free_memory"


def _get_pinned_patchers():
    pinned = getattr(comfy.model_management, _PINNED_PATCHERS, None)
    if pinned is None:
        pinned = {}
        setattr(comfy.model_management, _PINNED_PATCHERS, pinned)
    return pinned


def _pin_patcher(patcher):
    pinned = _get_pinned_patchers()
    key = id(patcher)
    _, count = pinned.get(key, (patcher, 0))
    pinned[key] = (patcher, count + 1)


def _unpin_patcher(patcher):
    pinned = _get_pinned_patchers()
    key = id(patcher)
    if key not in pinned:
        return
    _, count = pinned[key]
    if count <= 1:
        pinned.pop(key, None)
    else:
        pinned[key] = (patcher, count - 1)


def _install_free_memory_pin():
    if hasattr(comfy.model_management, _ORIGINAL_FREE_MEMORY):
        return

    original_free_memory = comfy.model_management.free_memory
    setattr(comfy.model_management, _ORIGINAL_FREE_MEMORY, original_free_memory)

    def free_memory_with_feather_pins(memory_required, device, keep_loaded=None, *args, **kwargs):
        keep_loaded = [] if keep_loaded is None else list(keep_loaded)
        pinned = _get_pinned_patchers()
        if pinned:
            keep_loaded_ids = {id(x) for x in keep_loaded}
            for loaded_model in comfy.model_management.current_loaded_models:
                patcher = loaded_model.model
                if patcher is not None and id(patcher) in pinned and id(loaded_model) not in keep_loaded_ids:
                    keep_loaded.append(loaded_model)
                    keep_loaded_ids.add(id(loaded_model))
        return original_free_memory(memory_required, device, keep_loaded, *args, **kwargs)

    comfy.model_management.free_memory = free_memory_with_feather_pins


def install_sampling_model_pin(model_patcher):
    if getattr(model_patcher, "_feather_sampling_model_pin", False):
        return

    _install_free_memory_pin()

    def pin_during_outer_sample(executor, *args, **kwargs):
        patcher = getattr(executor.class_obj, "model_patcher", model_patcher)
        _pin_patcher(patcher)
        try:
            return executor(*args, **kwargs)
        finally:
            _unpin_patcher(patcher)

    model_patcher.add_wrapper_with_key(
        WrappersMP.OUTER_SAMPLE,
        "feather_sampling_model_pin",
        pin_during_outer_sample,
    )
    model_patcher._feather_sampling_model_pin = True
