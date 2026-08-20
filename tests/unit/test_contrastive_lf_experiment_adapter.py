from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch

from experiments.methods.ceg_wm import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from main import derive_wrong_key_material, identify_root_key
from runtime import (
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeVaeFactors,
    create_runtime_adapter,
)


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value.clone()


class _Backend:
    def __init__(self) -> None:
        self.configuration = None

    def probe_devices(self):
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=0)

    def prepare(self, configuration, selected_device: str):
        self.configuration = configuration
        values = {
            field.name: getattr(configuration, field.name)
            for field in fields(RuntimeBackendIdentity)
            if field.name not in {"runtime_backend_name", "selected_device"}
        }
        return RuntimeBackendIdentity(**values, runtime_backend_name="contrastive_lf_synthetic_backend", selected_device=selected_device)

    def close(self):
        return None

    def run_generation(self, initial_latent, callback):
        return callback(self.configuration.callback_index, initial_latent.clone())

    def vae_decode(self, latent):
        return latent[:, :3].to(torch.float32)

    def vae_factors(self):
        return RuntimeVaeFactors(scaling_factor=1.0, shift_factor=0.0)

    def vae_encode(self, image):
        return _Posterior(image.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1))


@pytest.mark.unit
def test_experiment_adapter_rebuilds_blind_stage_a_observations_from_current_rgb8() -> None:
    runtime = create_runtime_adapter(_Backend())
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(), runtime
    )
    image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4).contiguous()
    root_key = "contrastive-lf-public-test-root"

    registered = adapter.observe_contrastive_lf_raw(
        image, root_key, candidate_id="lf_multiscale_lowpass_contrastive"
    )
    wrong = adapter.observe_contrastive_lf_raw(
        image,
        derive_wrong_key_material(
            identify_root_key(root_key).root_key_public_digest, 0
        ),
        candidate_id="lf_multiscale_lowpass_contrastive",
    )
    hf = adapter.observe_stage_a_hf_raw(image, root_key)

    assert len(registered.raw_feature) == 2
    assert len(registered.internal_decoy_features) == 8
    assert registered.root_key_public_digest == wrong.root_key_public_digest
    assert registered.raw_observation_digest != wrong.raw_observation_digest
    assert registered.key_role == "registered" and wrong.key_role == "wrong"
    assert hf.key_role == "registered"
    for method in (
        CegWmExperimentAdapter.observe_contrastive_lf_raw,
        CegWmExperimentAdapter.observe_contrastive_lf_candidate,
        CegWmExperimentAdapter.observe_stage_a_hf_raw,
    ):
        parameters = inspect.signature(method).parameters
        assert not {"prompt", "reference_image", "embed_record", "private_latent"} & set(parameters)
    runtime.close()
