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

import comfy.ops
import feather_ops
import torch
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from feather_ops.conv2d import _miopen_flags, prepack_conv2d_weight_3x3, select_backend, weight_cache_key

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

# The feather class of each eligible conv, keyed by the class it was built from. Created once per
# base so `type(module)` is one stable object: `copy.deepcopy` (how a multigpu clone copies a model)
# carries the class over, and every VAE loaded in the process shares it.
_FEATHER_CLASSES: dict[type, type] = {}


def _static_reason(module: torch.nn.modules.conv._ConvNd) -> str | None:
    """Why this module's convolutions can never use the kernel, or None if they can."""
    if tuple(module.kernel_size) != (3, 3) or tuple(module.padding) != (1, 1) or module.padding_mode != "zeros":
        return _NOT_3x3
    if tuple(module.stride) != (1, 1) or tuple(module.dilation) != (1, 1):
        return _NOT_STRIDE1
    if module.groups != 1:
        return _GROUPED
    return None


def _feather_attach(module: torch.nn.modules.conv._ConvNd) -> None:
    """The state a relabelled conv needs. Its `__init__` cannot run on a module that already exists."""
    if "_feather_packed" in module._buffers:
        return
    module.register_buffer("_feather_packed", torch.empty(0), persistent=False)
    module.register_buffer("_feather_bias", None, persistent=False)
    # Attributes the mixin reads. `_ConvNd` does not declare them and cannot, they exist only on a
    # relabelled conv.
    module.__dict__.update(_feather_key=None, _feather_static=_static_reason(module), _feather_warned=False)


class _FeatherConv2dMixin:
    """Mixin placed in front of a conv class whose instances take the feather path.

    The attributes below come from the conv class it is mixed into, and are declared here so type
    checkers can see them through `Self`. There is no `__init__`: instances are never built with
    this class, `patch_conv_modules` relabels convs ComfyUI already built and `_feather_attach`
    gives them the prepack state.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None
    kernel_size: tuple[int, ...]
    padding: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    padding_mode: str
    comfy_cast_weights: bool
    weight_function: list
    bias_function: list
    _feather_packed: torch.Tensor
    _feather_bias: torch.Tensor | None
    _feather_key: tuple | None
    _feather_static: str | None
    _feather_warned: bool

    # prepack lifetime

    def _feather_invalidate(self) -> None:
        self._feather_key = None
        if self._feather_packed.numel():
            self._feather_packed = torch.empty(0, device=self.weight.device, dtype=self.weight.dtype)
        self._feather_bias = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)  # ty: ignore[unresolved-attribute]
        self._feather_invalidate()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)  # ty: ignore[unresolved-attribute]
        self._feather_invalidate()
        return result

    def _feather_stale(self, input: torch.Tensor) -> bool:
        """Whether the prepacked copy has to be rebuilt for this call."""
        return self._feather_key != self._feather_cache_key(input) or not self._feather_packed.numel()

    def _feather_cache_key(self, input: torch.Tensor) -> tuple:
        # The packed buffer has to match the device and dtype the kernel will read, which under dynamic
        # VRAM is the compute copy rather than wherever the parameter happens to live.
        return (*weight_cache_key(self.weight), input.device, input.dtype)

    def _feather_pack(self, input: torch.Tensor) -> None:
        """Build the prepacked weight for `input`'s device and dtype.

        With dynamic VRAM ComfyUI sets `comfy_cast_weights` and the parameter may sit on the offload
        device, so the weight the kernel must read is its per-call cast copy: that is what
        `cast_bias_weight` returns, patches included. The packed buffer is one conv's 3x3 weights (a
        few hundred KB), it follows the module through `_apply`, and it is rebuilt only when the weight
        or the compute device changes.
        """
        if getattr(self, "comfy_cast_weights", False) and hasattr(self, "weight_function"):
            cast_weight, cast_bias, offload_stream = cast_bias_weight(self, input, dtype=input.dtype, bias_dtype=input.dtype, offloadable=True)
            try:
                self._feather_packed = prepack_conv2d_weight_3x3(cast_weight.detach())
                self._feather_bias = None if cast_bias is None else cast_bias.detach().contiguous()
            finally:
                uncast_bias_weight(self, cast_weight, cast_bias, offload_stream)
        else:
            weight = self.weight.detach().to(device=input.device, dtype=input.dtype)
            self._feather_packed = prepack_conv2d_weight_3x3(weight)
            bias = self.bias
            if bias is None:
                self._feather_bias = None
            else:
                bias = bias.detach().to(device=input.device, dtype=weight.dtype)
                self._feather_bias = bias if bias.is_contiguous() else bias.contiguous()
        self._feather_key = self._feather_cache_key(input)

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
            if self._feather_stale(input):
                self._feather_pack(input)
            return feather_ops.conv3x3_prepacked(input, self._feather_packed, self._feather_bias)
        # The first conv of `VAE.encode` gets a non-contiguous view, so a layout miss says nothing
        # about the shape. It must not use up the one line this module is allowed to log.
        if reason is not _LAYOUT and not self._feather_warned:
            self._feather_warned = True
            log = logger.debug if reason in _QUIET_REASONS else logger.info
            log("feather conv2d does not use the kernel for C_in=%s C_out=%s: %s", self.weight.shape[1], self.weight.shape[0], reason)
        if reason is _MIOPEN:
            # ComfyUI sets torch.backends.cudnn.enabled = False on AMD, which would send this call
            # down the im2col path. Force MIOpen on for it, without exhaustive search (no autotune).
            with _miopen_flags():
                return super().forward(input)  # ty: ignore[unresolved-attribute]
        return super().forward(input)  # ty: ignore[unresolved-attribute]


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
    patched = 0
    for module in root.modules():
        if type(module) not in bases or not isinstance(module, torch.nn.modules.conv._ConvNd):
            continue
        if tuple(module.kernel_size) != (3, 3) or tuple(module.padding) != (1, 1):
            continue
        cls = type(module)
        feather_cls = _FEATHER_CLASSES.get(cls)
        if feather_cls is None:
            feather_cls = type(cls.__name__, (_FeatherConv2dMixin, cls), {"__module__": __name__, "__qualname__": cls.__qualname__})
            _FEATHER_CLASSES[cls] = feather_cls
        module.__class__ = feather_cls
        _feather_attach(module)
        patched += 1
    return patched
