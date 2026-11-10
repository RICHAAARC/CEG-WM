import pytest
import torch

from cegwm.runtime.stage_protection_disk import DiskSavedTensors
from cegwm.runtime.stage_protection_v2 import StageProtectionProvider, at_latent
from test_stage_protection_runtime_v2 import config
from test_channel_precompensation_provider import Loader
from test_channel_precompensation_runtime import VAE, Processor

pytestmark = pytest.mark.integration


def test_actual_sdxl_and_source_vjp_disk_matches_cpu_copy_with_no_extra_computation(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("CPU only; no HTTP"))
    outputs = []
    events = []
    for disk_mode in (False, True):
        counts = {}
        disk = DiskSavedTensors(tmp_path) if disk_mode else None
        provider = StageProtectionProvider(config(), counters=counts, loader=Loader(),
            reader_loader=lambda _: (VAE(), Processor()), disk_storage=disk,
            resource_observer=lambda event, **kwargs: events.append(event))
        try:
            captured = provider.generate(config()["sources"][0])
            zero = provider.render(captured[8], capture_late=True)
            clean = provider.read(zero["image"])
            design_image = provider.reconstruct(zero["image"], config()["reconstruction"]["design_seed"])
            design_read = provider.read(design_image)
            sensitivity = provider.rgb_gradient(zero["image"], clean, design_image=design_image, design_read=design_read)
            gradients = provider.latent_gradients(at_latent(captured[8], zero["after"], zero["image"]),
                zero["image"], [sensitivity, sensitivity])["gradients"]
            outputs.append((sensitivity.gradient_rgb, gradients, dict(counts), sensitivity.feature))
        finally:
            provider.close()
            if disk:
                disk.close()
    for first, second in zip(outputs[0], outputs[1]):
        assert torch.equal(first, second) if isinstance(first, torch.Tensor) else first == second
    for name in ("before_sdxl_input_vae_encode", "after_sdxl_input_vae_encode", "after_sdxl_step",
                 "before_sdxl_output_vae_decode", "after_sdxl_output_vae_decode",
                 "before_sdxl_backward", "after_sdxl_backward"):
        assert name in events
