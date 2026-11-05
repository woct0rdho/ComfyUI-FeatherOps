"""Integration tests for the feather conv VAE patch (`comfy_feather/conv.py`, `comfy_feather/vae.py`).

Run with `python -m pytest tests` from the repository root. Everything here exercises convs that
`patch_conv_modules` relabelled, i.e. the objects a VAE built from `FeatherVAELoader` runs.
"""

import copy

import comfy.ops
import pytest
import torch
from feather_ops.conv2d import _miopen_flags, select_backend

from comfy_feather import conv as conv_patch
from comfy_feather import vae as vae_loader


@pytest.fixture(autouse=True)
def _no_grad():
    """VAE inference runs under `torch.inference_mode`. Keep tests in that mode, then restore."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    yield
    torch.set_grad_enabled(previous)


def _packed(module):
    """The prepacked weight of a relabelled conv, or None when the kernel never ran on it."""
    return module._buffers.get("_feather_packed")


def _miopen_ref(x, module):
    """The reference for a delegated call: F.conv2d with MIOpen forced on (ComfyUI disables cuDNN)."""
    with _miopen_flags():
        return torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)


def _build(in_ch, out_ch, dtype=torch.bfloat16, bias=True, base=None):
    base = base or comfy.ops.disable_weight_init.Conv2d
    module = base(in_ch, out_ch, 3, padding=1, bias=bias)
    with torch.no_grad():
        module.weight.normal_(0, 0.05)
        if module.bias is not None:
            module.bias.normal_(0, 0.1)
    module = module.to("cuda").to(dtype)
    assert conv_patch.patch_conv_modules(module) == 1
    return module


def test_patch_is_opt_in_per_module():
    """Patching relabels one tree. Nothing global changes, so an unpatched conv stays torch-only."""
    plain = comfy.ops.disable_weight_init.Conv2d(64, 64, 3, padding=1).cuda().to(torch.bfloat16)
    x = torch.randn((1, 64, 32, 32), device="cuda", dtype=torch.bfloat16)
    module = _build(64, 64)
    assert type(module) is not type(plain)
    assert type(plain) is comfy.ops.disable_weight_init.Conv2d
    assert _packed(plain) is None
    # a second pass over the same tree finds nothing left to relabel
    assert conv_patch.patch_conv_modules(module) == 0
    with torch.no_grad():
        out = module(x)
    ref = torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_patch_leaves_ineligible_convs_alone():
    """1x1, stride 2, no padding, a non-zero padding mode and grouped convs keep their class."""
    layers: list[torch.nn.Conv2d] = [
        comfy.ops.disable_weight_init.Conv2d(64, 64, 3, padding=1),
        comfy.ops.disable_weight_init.Conv2d(64, 64, 1),
        comfy.ops.disable_weight_init.Conv2d(64, 64, 3, padding=1, stride=2),
        comfy.ops.disable_weight_init.Conv2d(64, 64, 3),
        comfy.ops.disable_weight_init.Conv2d(64, 64, 3, padding=1, padding_mode="replicate"),
        comfy.ops.disable_weight_init.Conv2d(64, 64, 3, padding=1, groups=8),
        torch.nn.Conv2d(64, 64, 3, padding=1),
        comfy.ops.manual_cast.Conv2d(64, 64, 3, padding=1),
    ]
    # comfy's reset_parameters is a no-op, so leave the state dict as ComfyUI leaves it: normal weights
    with torch.no_grad():
        for layer in layers:
            layer.weight.normal_(0, 0.05)
            if layer.bias is not None:
                layer.bias.normal_(0, 0.1)
    tree = torch.nn.Sequential(*layers).cuda().to(torch.bfloat16)
    # only kernel size and padding are filtered here. The rest declines per call, so the stride-2,
    # replicate-padded and grouped convs, the bare nn.Conv2d and the manual_cast conv all relabel
    assert conv_patch.patch_conv_modules(tree) == 6
    assert conv_patch._FeatherConv2dMixin in type(layers[0]).__mro__
    assert type(layers[1]) is comfy.ops.disable_weight_init.Conv2d
    assert type(layers[3]) is comfy.ops.disable_weight_init.Conv2d
    # a relabelled conv that the policy rejects still returns exactly what it returned before
    x = torch.randn((1, 64, 32, 32), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for index in (2, 4, 5):
            module = layers[index]
            assert torch.equal(module(x), torch.nn.Conv2d.forward(module, x)), f"layer {index} must fall back to torch"
            assert not _packed(module).numel()


def test_patched_conv_class_is_shared_and_survives_deepcopy():
    first = _build(64, 64)
    second = _build(64, 64)
    assert type(first) is type(second)
    clone = copy.deepcopy(first)
    assert type(clone) is type(first)
    assert not _packed(clone).numel()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(96, 96, 64), (128, 128, 128), (512, 512, 32), (8, 16, 64), (48, 96, 48)])
def test_patched_conv_matches_torch(dtype, shape):
    in_ch, out_ch, size = shape
    module = _build(in_ch, out_ch, dtype)
    x = torch.randn((1, in_ch, size, size), device="cuda", dtype=dtype)
    ref = torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)

    out = module(x)
    if select_backend(dtype, in_ch, out_ch, size, size) == "miopen":
        # fp16 3x3 s1 is delegated to MIOpen on purpose (its Winograd kernels are faster)
        assert not _packed(module).numel(), "a delegated weight must not be prepacked"
        assert torch.equal(out, _miopen_ref(x, module)), "the delegated path is F.conv2d + MIOpen"
    else:
        assert _packed(module).numel(), "expected the kernel path, which prepacks the weight"
        assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_fp16_uses_miopen_and_bf16_uses_the_kernel():
    """The dtype split is the documented policy: fp16 -> MIOpen Winograd, bf16 -> our kernel."""
    assert select_backend(torch.float16, 512, 512, 256, 256) == "miopen"
    assert select_backend(torch.bfloat16, 512, 512, 256, 256) == "kernel"
    # beyond a 512 MiB input MIOpen's Winograd loses, so the kernel takes over again
    assert select_backend(torch.float16, 128, 128, 2048, 2048) == "kernel"

    module = _build(96, 96, torch.float16)
    x = torch.randn((1, 96, 128, 128), device="cuda", dtype=torch.float16)
    out = module(x)
    assert not _packed(module).numel(), "a delegated weight must not be prepacked"
    assert torch.equal(out, _miopen_ref(x, module)), "the delegated path is exactly F.conv2d + MIOpen"


def test_inference_mode_matches():
    module = _build(96, 96)
    x = torch.randn((1, 96, 64, 64), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        out = module(x)
        ref = torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_patch_is_transparent():
    module = _build(96, 96)
    reference = torch.nn.Conv2d(96, 96, 3, padding=1, bias=True)
    assert set(module.state_dict()) == set(reference.state_dict())
    x = torch.randn((1, 96, 64, 64), device="cuda", dtype=torch.bfloat16)

    # a load_state_dict drops the prepacked copy
    with torch.no_grad():
        module(x)
    assert _packed(module).numel() > 0
    module.load_state_dict(reference.state_dict())
    assert not _packed(module).numel()

    # a dtype move repacks: the buffer follows the module and the cache key stops matching
    with torch.no_grad():
        module(x)
    module.half()
    with torch.no_grad():
        out = module(x.half())
    ref = torch.nn.functional.conv2d(x.half(), module.weight, module.bias, 1, 1)
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_weight_function_disables_kernel():
    module = _build(96, 96)
    x = torch.randn((1, 96, 64, 64), device="cuda", dtype=torch.bfloat16)
    module.weight_function = [lambda weight: weight]
    out = module(x)
    assert not _packed(module).numel()
    ref = torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)
    assert torch.equal(out, ref)


def test_grad_enabled_falls_back_but_matches():
    module = _build(96, 96)
    x = torch.randn((1, 96, 48, 48), device="cuda", dtype=torch.bfloat16)
    torch.set_grad_enabled(True)
    try:
        out = module(x)
    finally:
        torch.set_grad_enabled(False)
    assert not _packed(module).numel()
    ref = torch.nn.functional.conv2d(x, module.weight, module.bias, 1, 1)
    assert torch.equal(out, ref)


def test_vae_loader_is_a_load_vae_node():
    """Same node contract as ComfyUI's, so it drops into the graph next to it."""
    import nodes

    assert issubclass(vae_loader.FeatherVAELoader, nodes.VAELoader)
    assert vae_loader.FeatherVAELoader.FUNCTION == nodes.VAELoader.FUNCTION == "load_vae"
    assert vae_loader.FeatherVAELoader.RETURN_TYPES == nodes.VAELoader.RETURN_TYPES == ("VAE",)
    assert vae_loader.FeatherVAELoader.CATEGORY == nodes.VAELoader.CATEGORY
    assert vae_loader.FeatherVAELoader.INPUT_TYPES() == nodes.VAELoader.INPUT_TYPES()


def test_vae_loader_patches_the_loaded_vae():
    """`pixel_space` needs no file, so this covers the loader path end to end."""
    (vae,) = vae_loader.FeatherVAELoader().load_vae("pixel_space")
    assert vae.latent_channels == 3
    # nothing to relabel in a pixel-space VAE, and no reload factory to redirect
    assert conv_patch.patch_conv_modules(vae.first_stage_model) == 0
