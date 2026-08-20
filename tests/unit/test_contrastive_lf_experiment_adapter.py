from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch

from experiments.methods.ceg_wm import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.runners.contrastive_lf_branch_attribution import (
    AdapterBackedStageAOperations,
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
        self.vae_encode_count = 0

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
        self.vae_encode_count += 1
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


@pytest.mark.unit
def test_key_free_public_observation_is_reused_before_key_specific_scoring() -> None:
    backend = _Backend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(), runtime
    )
    image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4).contiguous()
    root_key = "contrastive-lf-public-cache-root"
    wrong = derive_wrong_key_material(
        identify_root_key(root_key).root_key_public_digest, 0
    )

    prepared = adapter.prepare_stage_a_public_rgb8_observation(image)
    registered = adapter.score_contrastive_lf_prepared_observation(
        prepared,
        root_key,
        candidate_id="lf_multiscale_lowpass_contrastive",
    )
    wrong_result = adapter.score_contrastive_lf_prepared_observation(
        prepared,
        wrong,
        candidate_id="lf_multiscale_lowpass_contrastive",
    )
    hf = adapter.score_stage_a_hf_prepared_observation(prepared, root_key)

    assert backend.vae_encode_count == 1
    assert registered.root_key_public_digest == wrong_result.root_key_public_digest
    assert registered.raw_observation_digest != wrong_result.raw_observation_digest
    assert hf.key_role == "registered"
    assert not hasattr(prepared, "root_key")
    assert not hasattr(prepared, "candidate_id")
    runtime.close()


@pytest.mark.unit
def test_stage_a_operations_cache_hits_same_rgb8_and_misses_changed_rgb8() -> None:
    backend = _Backend()
    runtime = create_runtime_adapter(backend)
    session = runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(), runtime
    )
    operations = AdapterBackedStageAOperations(
        backend=backend,
        runtime_adapter=runtime,
        session=session,
        adapter=adapter,
        root_key="contrastive-lf-cache-operation-root",
        implementation_revision="1" * 40,
    )
    first = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4).contiguous()
    second = first.clone()
    second[0, 0, 0, 0] += 1

    observed = operations.prepare_public_observation(first)
    replayed = operations.prepare_public_observation(first.clone())
    changed = operations.prepare_public_observation(second)

    assert observed is replayed
    assert observed.cache_identity != changed.cache_identity
    assert operations.cache_diagnostics() == {
        "cache_entry_count": 2,
        "cache_hit_count": 1,
        "cache_miss_count": 2,
        "vae_encode_count": 2,
    }
    operations.close()
