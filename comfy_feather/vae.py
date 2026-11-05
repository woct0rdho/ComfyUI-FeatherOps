"""A `Load VAE` node whose 3x3 convolutions run on the feather-ops kernel.

Same inputs, same VAE, same `state_dict`, same patcher as ComfyUI's `Load VAE`: loading is
ComfyUI's own `VAELoader.load_vae`, and the only thing added is the class relabel in
`conv.patch_conv_modules`. That means every VAE that loader can load works here too (the TAESD
decoders, `pixel_space`, 3D VAEs, and any VAE another custom node teaches `comfy.sd.VAE` to
build), and a VAE loaded from the plain `Load VAE` node or out of a checkpoint keeps the torch
path. The kernel is opt-in per load, with nothing patched inside ComfyUI.
"""

import logging

import comfy.sd
import nodes

from .conv import patch_conv_modules

logger = logging.getLogger(__name__)


def _load_vae_patcher(vae_path, metadata=None, device=None, disable_dynamic=False):
    """`comfy.sd.load_vae_patcher`, then patch: the reload factory for a multigpu deepclone."""
    patcher = comfy.sd.load_vae_patcher(vae_path, metadata, device, disable_dynamic)
    patch_conv_modules(patcher.model)
    return patcher


class FeatherVAELoader(nodes.VAELoader):
    """`Load VAE`, with the loaded VAE's 3x3 convolutions on the feather-ops kernel."""

    SEARCH_ALIASES = ("vae", "load vae")

    def load_vae(self, vae_name):
        (vae,) = super().load_vae(vae_name)
        count = patch_conv_modules(vae.first_stage_model)
        init = getattr(vae.patcher, "cached_patcher_init", None)
        if count and init is not None:
            # ComfyUI registers a factory that rebuilds this VAE from disk for a multigpu deepclone.
            # It has to go through the patch too, or the clone comes back on the torch path.
            vae.patcher.cached_patcher_init = (_load_vae_patcher, init[1])
        logger.info("feather VAE loader: %s, %s convs on the feather kernel", vae_name, count)
        return (vae,)
