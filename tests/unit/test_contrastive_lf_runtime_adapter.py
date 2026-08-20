from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from runtime import (
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeVaeFactors,
    create_runtime_adapter,
)


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.value.clone()


class _Backend:
    def __init__(self) -> None:
        self.configuration = None
        self.posterior: _Posterior | None = None

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=0)

    def prepare(self, configuration, selected_device: str) -> RuntimeBackendIdentity:
        self.configuration = configuration
        values = {
            field.name: getattr(configuration, field.name)
            for field in fields(RuntimeBackendIdentity)
            if field.name not in {"runtime_backend_name", "selected_device"}
        }
        return RuntimeBackendIdentity(
            **values,
            runtime_backend_name="contrastive_lf_synthetic_backend",
            selected_device=selected_device,
        )

    def close(self) -> None:
        return None

    def run_generation(self, initial_latent, callback):
        assert self.configuration is not None
        return callback(self.configuration.callback_index, initial_latent.clone())

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :3].to(torch.float32)

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=1.0, shift_factor=0.0)

    def vae_encode(self, image: torch.Tensor) -> _Posterior:
        mode = image.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        self.posterior = _Posterior(mode)
        return self.posterior


@pytest.mark.unit
def test_public_rgb8_vae_observation_uses_posterior_mode_and_binary32() -> None:
    backend = _Backend()
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")
    image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4).contiguous()

    observed = adapter.observe_public_rgb8_vae(image)

    assert observed.detection_latent.shape == (1, 16, 4, 4)
    assert observed.detection_latent.dtype is torch.float32
    assert backend.posterior is not None and backend.posterior.mode_calls == 1
    assert len(observed.rgb8_digest) == 64
    assert len(observed.observation_identity) == 64
    adapter.close()


@pytest.mark.unit
def test_public_rgb8_vae_observation_rejects_non_rgb8_boundary() -> None:
    adapter = create_runtime_adapter(_Backend())
    adapter.initialize("cpu")
    with pytest.raises(Exception, match="failed closed"):
        adapter.observe_public_rgb8_vae(torch.zeros((1, 16, 4, 4), dtype=torch.float32))


@pytest.mark.unit
def test_production_factory_constructs_public_backend_runtime_method_chain_without_model_execution(monkeypatch) -> None:
    import experiments.methods as method_module
    import experiments.runners.contrastive_lf_branch_attribution as runner_module
    import runtime as runtime_module

    calls: list[str] = []

    class Backend:
        def __init__(self, **kwargs) -> None:
            calls.append("Sd35PipelineBackend")

    class Runtime:
        def initialize(self, device: str):
            calls.append(f"Sd35RuntimeAdapter:{device}")
            return SimpleNamespace(
                model_id="stabilityai/stable-diffusion-3.5-large",
                model_revision="2a2a0e0f0552f080f622674b79a9f577c0d64936",
                runtime_config_digest="1" * 64,
                runtime_backend_name="diffusers_sd35_pipeline",
                selected_device="cuda:0",
                    image_height=1024,
                    image_width=1024,
                    vae_encode_protocol="posterior_mode_public_rgb8_binary32",
                )

    class MethodAdapter:
        def __init__(self, configuration, runtime_adapter) -> None:
            calls.append("CegWmExperimentAdapter")

    monkeypatch.setattr(runtime_module, "Sd35PipelineBackend", Backend)
    monkeypatch.setattr(runtime_module, "create_runtime_adapter", lambda *args: Runtime())
    monkeypatch.setattr(method_module, "CegWmExperimentAdapter", MethodAdapter)
    monkeypatch.setattr(method_module, "load_ceg_wm_experiment_adapter_configuration", lambda *args: object())
    monkeypatch.setattr(runner_module, "validate_jpeg_capability", lambda: calls.append("Pillow12.3.0"))
    for name, value in (
        ("HF_TOKEN", "token-not-persisted"),
        ("CEG_WM_ROOT_KEY", "root-not-persisted"),
        ("CEG_WM_CACHE_ROOT", "/cache"),
        ("CEG_WM_PERSISTENT_ROOT", "/models"),
    ):
        monkeypatch.setenv(name, value)

    operations = runner_module.create_adapter_backed_stage_a_operations(
        implementation_revision="a" * 40
    )

    assert type(operations) is runner_module.AdapterBackedStageAOperations
    assert calls == [
        "Pillow12.3.0",
        "Sd35PipelineBackend",
        "Sd35RuntimeAdapter:cuda",
        "CegWmExperimentAdapter",
    ]
