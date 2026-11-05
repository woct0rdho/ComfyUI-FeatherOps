"""Run the 3x3 convolutions of a loaded VAE on the feather-ops kernel.

The 2D VAEs spend 96-97 % of their conv FLOPs in 3x3 stride-1 padding-1 convolutions, and ComfyUI
runs those as `im2col` + hipBLASLt `addmm` (because `torch.backends.cudnn.enabled` is turned off
on RDNA3+). `feather_ops.conv2d` replaces them with a fused NCHW kernel and keeps a prepacked copy
of the weight, so a VAE that is loaded once and applied to many images pays the repack once.

`patch_conv_modules` applies this to one module tree, after ComfyUI has built and loaded it, by
relabelling each eligible conv's class. Nothing in `comfy.ops` is replaced, so a VAE that was not
loaded through `FeatherVAELoader` keeps the torch path, and the module tree, `state_dict`,
`ModelPatcher` and offload paths look exactly like the ones ComfyUI built: the conv objects, their
parameters and their buffers are the same objects, only `type(conv)` changed.
"""

import logging
from typing import TYPE_CHECKING

import comfy.ops
import torch
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from feather_ops.conv2d import conv3x3_prepacked, miopen_flags, prepack_conv2d_weight_3x3, select_backend, weight_cache_key
from feather_ops.conv2d_module import ModuleCache, conv3x3_module_reason, prepacked_conv3x3
from feather_ops.modules import relabel_modules

logger = logging.getLogger(__name__)

_MIOPEN = "delegated to MIOpen, which beats im2col on this shape"
_LAYOUT = "input is not a contiguous 4D NCHW tensor"
_CHANNELS = "input channels are not the weight channels or not a multiple of 8"
_NO_TILE = "no kernel tile divides C_out x width"
_NOT_3x3 = "not a 3x3 stride-1 padding-1 zeros conv"
_NOT_STRIDE1 = "not a stride-1 dilation-1 conv"
_GROUPED = "grouped conv"

# A normal VAE always has some of these: its encoder starts with `C_in % 8` (ComfyUI pads 3 channels to
# 4), its decoder ends with a `C_out` below the tile minimum, the 1x1 and stride-2 convs are structural,
# and MIOpen takes over the first two. Logging them at INFO would print a handful of lines on every load
# and say nothing actionable, so they go to debug. The rest are worth one line per module: they mean the
# model is not on the kernel for a reason the user may want to know.
_QUIET_REASONS = frozenset({_MIOPEN, _LAYOUT, _CHANNELS, _NO_TILE, _NOT_3x3, _NOT_STRIDE1, _GROUPED})

# The prepack cache and the reason vocabulary are feather_ops'. The mixin below adds what is ComfyUI's:
# the per-call cast/offload copy, `run_every_op`, the weight-patch checks and the log policy.
_CACHE = ModuleCache()


def _static_reason(module: torch.nn.modules.conv._ConvNd) -> str | None:
    """Why this module's convolutions can never use the kernel, or None if they can."""
    reason = conv3x3_module_reason(module)
    if reason is None:
        return None
    if module.padding_mode != "zeros" or tuple(module.kernel_size) != (3, 3) or tuple(module.padding) != (1, 1):
        return _NOT_3x3
    if tuple(module.stride) != (1, 1) or tuple(module.dilation) != (1, 1):
        return _NOT_STRIDE1
    return _GROUPED


# Under type checking the mixin derives from `nn.Conv2d`, so `self.weight`, `self.padding` and the
# `super().forward()` call resolve. At runtime it derives from `object`, because `relabelled_class` puts it
# in front of an existing conv class and that class is the one carrying the conv behaviour.
_ConvBase = torch.nn.Conv2d if TYPE_CHECKING else object


class _FeatherConv2dMixin(_ConvBase):
    """Mixin placed in front of a conv class whose instances take the feather path.

    There is no `__init__`: instances are never built with this class, `patch_conv_modules` relabels convs
    ComfyUI already built. The prepack state lives in feather_ops' `ModuleCache`, so nothing has to be
    attached.
    """

    comfy_cast_weights: bool
    weight_function: list
    bias_function: list
    _feather_static: str | None
    _feather_warned: bool

    # prepack lifetime

    def _feather_invalidate(self) -> None:
        _CACHE.invalidate(self)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._feather_invalidate()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self._feather_invalidate()
        return result

    def _feather_pack(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The prepacked weight for `input`'s device and dtype, cached on the module.

        With dynamic VRAM ComfyUI sets `comfy_cast_weights` and the parameter may sit on the offload
        device, so the weight the kernel must read is its per-call cast copy: that is what
        `cast_bias_weight` returns, patches included. The cache key therefore includes the compute
        device and dtype, since the parameter itself does not change between those calls.
        """

        def make():
            if getattr(self, "comfy_cast_weights", False) and hasattr(self, "weight_function"):
                cast_weight, cast_bias, offload_stream = cast_bias_weight(self, input, dtype=input.dtype, bias_dtype=input.dtype, offloadable=True)
                try:
                    packed = prepack_conv2d_weight_3x3(cast_weight.detach())
                    packed_bias = None if cast_bias is None else cast_bias.detach().contiguous()
                finally:
                    uncast_bias_weight(self, cast_weight, cast_bias, offload_stream)
                return packed, packed_bias
            weight = self.weight.detach().to(device=input.device, dtype=input.dtype)
            bias = self.bias
            packed_bias = None
            if bias is not None:
                bias = bias.detach().to(device=input.device, dtype=weight.dtype)
                packed_bias = bias if bias.is_contiguous() else bias.contiguous()
            return prepack_conv2d_weight_3x3(weight), packed_bias

        key = (*weight_cache_key(self.weight), input.device, input.dtype)
        return prepacked_conv3x3(self, key=key, make=make)

    # forward

    def _feather_reason(self, input: torch.Tensor) -> str | None:
        """Why this call cannot use the kernel, or None if it can."""
        if self._feather_static is not None:
            return self._feather_static
        if getattr(self, "weight_function", None) or getattr(self, "bias_function", None):
            return "a weight/bias function is installed"
        if getattr(self, "weight_lowvram_function", None) is not None:
            # A lazily applied patch (LoCon and friends). The packed copy is cached, so a strength change
            # would not be visible to it. ComfyUI's own path applies the patch per call instead.
            return "a low-VRAM weight patch is attached"
        # The weight may be stored in any float dtype: with dynamic VRAM ComfyUI keeps the checkpoint's
        # dtype (fp32 VAE files stay fp32) and casts to the compute dtype per call, and `_feather_pack`
        # follows the activation dtype the same way (`cast_bias_weight` with `dtype=input.dtype`), which
        # is the cast ComfyUI itself would do. Only the activation has to be a kernel dtype.
        if input.dtype not in (torch.bfloat16, torch.float16) or not torch.is_floating_point(self.weight):
            return "activation is not bf16/fp16, or the weight is quantized"
        channels = input.shape[1]
        if channels != self.weight.shape[1]:
            return _CHANNELS
        if input.ndim != 4 or not input.is_contiguous():
            return _LAYOUT
        # `select_backend` owns the whole policy, including the shapes this kernel has no tile for: a VAE's
        # first conv (`C_in % 8`) and last one (`C_out = 3`) are both delegated to MIOpen from here.
        backend = select_backend(input.dtype, channels, self.weight.shape[0], input.shape[2], input.shape[3])
        if backend == "miopen":
            return _MIOPEN
        if backend != "kernel":
            return _NO_TILE
        if torch.is_grad_enabled():
            return "autograd is active"
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        reason = self._feather_reason(input)
        if reason is None:
            run_every_op()
            packed, packed_bias = self._feather_pack(input)
            return conv3x3_prepacked(input, packed, packed_bias)
        # The first conv of `VAE.encode` gets a non-contiguous view, so a layout miss says nothing
        # about the shape. It must not use up the one line this module is allowed to log.
        if reason is not _LAYOUT and not self._feather_warned:
            self._feather_warned = True
            log = logger.debug if reason in _QUIET_REASONS else logger.info
            log("feather conv2d does not use the kernel for C_in=%s C_out=%s: %s", self.weight.shape[1], self.weight.shape[0], reason)
        if reason is _MIOPEN:
            # ComfyUI sets torch.backends.cudnn.enabled = False on AMD, which would send this call
            # down the im2col path. Force MIOpen on for it, without exhaustive search (no autotune).
            with miopen_flags():
                return super().forward(input)
        return super().forward(input)


def patch_conv_modules(root: torch.nn.Module) -> int:
    """Relabel every eligible conv in `root` so its 3x3 calls take the feather path.

    Only classes ComfyUI itself builds are relabelled (`comfy.ops` Conv2d, plus a bare
    `torch.nn.Conv2d` for the TAESD decoders), and by exact type: a subclass may override `forward`,
    and putting this mixin in front of it would drop that override. Nothing else is filtered here -
    a 3x3 stride-1 conv that still cannot use the kernel says so per call in `_feather_reason` and
    falls back to `torch.nn.Conv2d.forward`, which is what it did before.

    Returns the number of convs relabelled. A second call is a no-op, because a relabelled class is
    none of the bases.
    """
    bases = {torch.nn.Conv2d, comfy.ops.disable_weight_init.Conv2d, comfy.ops.manual_cast.Conv2d}

    def accept(module: torch.nn.Module) -> bool:
        if type(module) not in bases or not isinstance(module, torch.nn.Conv2d):
            return False
        if tuple(module.kernel_size) != (3, 3) or tuple(module.padding) != (1, 1):
            return False
        module.__dict__.update(_feather_static=_static_reason(module), _feather_warned=False)
        return True

    return relabel_modules(root, _FeatherConv2dMixin, accept)
