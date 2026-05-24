import torch


def patch_taehv_preview():
    import comfy.model_management
    import latent_preview

    previewer_cls = latent_preview.TAEHVPreviewerImpl
    if getattr(previewer_cls, "_feather_direct_decode", False):
        return

    def decode_latent_to_preview(self, x0):
        vae = self.taesd
        model = vae.first_stage_model
        device = vae.device
        dtype = vae.vae_dtype

        current_device = getattr(self, "_feather_preview_device", None)
        if current_device != device:
            model.to(device=device, dtype=dtype)
            self._feather_preview_device = device

        x = x0[:1, :, :1].to(device=device, dtype=dtype)
        x_sample = model.decode(x)[0, :, 0].movedim(0, 2)
        x_sample = x_sample.to(device=comfy.model_management.intermediate_device(), dtype=torch.float32)
        return latent_preview.preview_to_image(x_sample, do_scale=False)

    previewer_cls.decode_latent_to_preview = decode_latent_to_preview
    previewer_cls._feather_direct_decode = True
