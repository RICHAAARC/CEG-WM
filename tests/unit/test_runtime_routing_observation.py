from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from runtime import (
    ROUTING_OBSERVATION_CANDIDATE_ID,
    ROUTING_PROBE_RELATIVE_STEP,
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeVaeFactors,
    create_runtime_adapter,
)
from runtime.configuration import load_runtime_configuration
from runtime.sd35_backend import Sd35PipelineBackend


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value.detach().clone()


class _RoutingBackend:
    def __init__(self, *, omit_previous_callback: bool = False) -> None:
        self.configuration = None
        self.omit_previous_callback = omit_previous_callback
        self.decode_inputs: list[torch.Tensor] = []
        self.callback_snapshots: dict[int, torch.Tensor] = {}
        self.generation_calls = 0

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device: str):
        self.configuration = configuration
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="synthetic_cpu_routing_backend",
            selected_device=selected_device,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            pipeline_class=configuration.pipeline_class,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            guidance_scale=configuration.guidance_scale,
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            generation_seed_device=configuration.generation_seed_device,
            latent_dtype=configuration.latent_dtype,
            template_dtype=configuration.template_dtype,
            score_dtype=configuration.score_dtype,
            callback_index=configuration.callback_index,
            callback_hold_scheduler_intervals=(
                configuration.callback_hold_scheduler_intervals
            ),
            vae_decode_protocol=configuration.vae_decode_protocol,
            vae_encode_protocol=configuration.vae_encode_protocol,
            vae_scaling_factor_source=configuration.vae_scaling_factor_source,
            vae_shift_factor_source=configuration.vae_shift_factor_source,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                configuration.detection_conditioning_protocol
            ),
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def close(self) -> None:
        return None

    def run_generation(self, initial_latent, callback):
        assert self.configuration is not None
        self.generation_calls += 1
        state = initial_latent.detach().clone()
        for index in range(self.configuration.inference_steps):
            state = (
                state.to(dtype=torch.float32) + (index + 1) * 1.0e-4
            ).to(dtype=torch.float16)
            if self.omit_previous_callback and index == 17:
                continue
            self.callback_snapshots[index] = state.detach().clone()
            state = callback(index, state)
        return state

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=1.0, shift_factor=0.0)

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        return torch.clamp(latent.detach().clone(), 0.0, 1.0)

    def vae_encode(self, image: torch.Tensor) -> _Posterior:
        return _Posterior(image)


def _base_latent() -> torch.Tensor:
    return torch.tensor(
        (
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
            0.22,
            0.24,
            0.26,
            0.28,
            0.30,
            0.32,
            0.34,
            0.36,
            0.38,
            0.40,
            0.42,
            0.44,
            0.46,
            0.48,
            0.50,
            0.52,
            0.54,
            0.56,
            0.58,
            0.60,
            0.62,
            0.64,
            0.66,
            0.68,
            0.70,
            0.72,
            0.74,
            0.76,
            0.78,
            0.80,
            0.82,
            0.84,
            0.86,
            0.88,
            0.20,
            0.24,
            0.28,
            0.32,
            0.36,
            0.40,
            0.44,
            0.48,
        ),
        dtype=torch.float16,
    ).reshape(1, 3, 4, 4)


def _adapter(backend: _RoutingBackend):
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")
    return adapter


def _nearest_rank_positive_p95(values: tuple[float, ...]) -> float:
    assert values
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


@pytest.mark.unit
def test_development_prompt_binding_survives_paired_generations(
    tmp_path: Path,
) -> None:
    class PromptPipeline:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def __call__(self, **kwargs: object) -> object:
            self.prompts.append(str(kwargs["prompt"]))
            return SimpleNamespace(images=kwargs["latents"])

    backend = Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="initial development prompt",
    )
    pipeline = PromptPipeline()
    backend._configuration = load_runtime_configuration()  # type: ignore[attr-defined]
    backend._device = torch.device("cpu")  # type: ignore[attr-defined]
    backend._pipeline = pipeline  # type: ignore[attr-defined]
    backend.set_development_generation_prompts("paired development prompt")
    latent = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    backend.run_generation(latent, lambda _index, value: value)
    backend.run_generation(latent, lambda _index, value: value)

    assert pipeline.prompts == [
        "paired development prompt",
        "paired development prompt",
    ]


@pytest.mark.unit
def test_cross_fit_references_normalize_the_same_measurement_without_rerun() -> None:
    backend = _RoutingBackend()
    adapter = _adapter(backend)
    fit_measurement = adapter.measure_generation_routing_reference_inputs(
        _base_latent(),
        sample_index=21,
    )
    probe_measurement = adapter.measure_generation_routing_reference_inputs(
        _base_latent(),
        sample_index=22,
    )
    generation_calls_before_normalize = backend.generation_calls
    decode_calls_before_normalize = len(backend.decode_inputs)
    gradient_reference = _nearest_rank_positive_p95(
        tuple(value for value in fit_measurement.texture_gradient_values if value > 0.0)
    )
    response_reference = _nearest_rank_positive_p95(
        tuple(value for value in fit_measurement.response_ratio_values if value > 0.0)
    )
    sensitivity_reference = _nearest_rank_positive_p95(
        tuple(value for value in fit_measurement.sensitivity_ratio_values if value > 0.0)
    )

    result = adapter.normalize_generation_routing_measurement(
        probe_measurement,
        reference_gradient=gradient_reference,
        reference_response=response_reference,
        reference_sensitivity=sensitivity_reference,
    )

    assert backend.generation_calls == generation_calls_before_normalize == 2
    assert len(backend.decode_inputs) == decode_calls_before_normalize == 4
    assert torch.equal(result.routing_rgb, probe_measurement.semantic_rgb)
    assert result.texture.values == pytest.approx(
        tuple(
            min(value / gradient_reference, 1.0)
            for value in probe_measurement.texture_gradient_values
        )
    )
    assert result.response.values == pytest.approx(
        tuple(
            min(value / response_reference, 1.0)
            for value in probe_measurement.response_ratio_values
        )
    )
    assert result.sensitivity.values == pytest.approx(
        tuple(
            min(value / sensitivity_reference, 1.0)
            for value in probe_measurement.sensitivity_ratio_values
        )
    )
    assert not hasattr(probe_measurement, "identity_mapping")


@pytest.mark.unit
def test_runtime_routing_observation_uses_real_callback_and_vae_path() -> None:
    backend = _RoutingBackend()
    adapter = _adapter(backend)

    result = adapter.observe_generation_routing(
        _base_latent(),
        sample_index=11,
        reference_gradient=4.0,
        reference_response=1.0,
        reference_sensitivity=1.0,
    )

    assert result.candidate_id == ROUTING_OBSERVATION_CANDIDATE_ID
    assert result.sample_index == 11
    assert result.callback_indices == tuple(
        range(adapter.configuration.inference_steps)
    )
    assert torch.equal(
        result.previous_write_latent,
        backend.callback_snapshots[17],
    )
    assert torch.equal(
        result.routing_write_latent,
        backend.callback_snapshots[18],
    )
    assert torch.equal(result.routing_rgb, backend.decode_inputs[0])
    assert result.previous_write_latent.data_ptr() != (
        backend.callback_snapshots[17].data_ptr()
    )
    assert result.routing_write_latent.data_ptr() != (
        backend.callback_snapshots[18].data_ptr()
    )
    assert len(backend.decode_inputs) == 2
    assert result.texture.spatial_shape == (4, 4)
    assert result.response.spatial_shape == (4, 4)
    assert result.sensitivity.spatial_shape == (4, 4)
    assert result.nominal_relative_probe_step == ROUTING_PROBE_RELATIVE_STEP
    assert result.actual_probe_step > 0.0
    assert result.reference_gradient == 4.0
    assert result.reference_response == 1.0
    assert result.reference_sensitivity == 1.0
    assert len(result.public_probe_domain_digest) == 64
    assert len(result.public_probe_values_float32_be_sha256) == 64
    assert not hasattr(result, "identity_mapping")


@pytest.mark.unit
def test_texture_and_response_follow_registered_formulas() -> None:
    backend = _RoutingBackend()
    adapter = _adapter(backend)
    result = adapter.observe_generation_routing(
        _base_latent(),
        sample_index=3,
        reference_gradient=4.0,
        reference_response=0.5,
        reference_sensitivity=1.0,
    )

    rgb = result.routing_rgb.to(dtype=torch.float32)
    grayscale = (
        rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
    )
    padded = torch.nn.functional.pad(
        grayscale.unsqueeze(1),
        (1, 1, 1, 1),
        mode="replicate",
    )
    horizontal = torch.tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).reshape(1, 1, 3, 3)
    vertical = horizontal.transpose(2, 3)
    expected_texture = torch.sqrt(
        torch.nn.functional.conv2d(padded, horizontal).square()
        + torch.nn.functional.conv2d(padded, vertical).square()
    ).squeeze(1) / 4.0
    assert result.texture.values == pytest.approx(
        torch.clamp(expected_texture, 0.0, 1.0).reshape(-1).tolist()
    )

    previous = result.previous_write_latent.to(dtype=torch.float32)
    current = result.routing_write_latent.to(dtype=torch.float32)
    expected_response = torch.sqrt((current - previous).square().mean(dim=1))
    expected_response /= (
        torch.sqrt(previous.square().mean(dim=1))
        + torch.sqrt(current.square().mean(dim=1))
        + 1.0e-12
    )
    expected_response = torch.clamp(expected_response / 0.5, 0.0, 1.0)
    assert result.response.values == pytest.approx(
        expected_response.reshape(-1).tolist()
    )


@pytest.mark.unit
def test_public_probe_is_sample_bound_and_actual_dtype_materialized() -> None:
    first_backend = _RoutingBackend()
    first = _adapter(first_backend).observe_generation_routing(
        _base_latent(),
        sample_index=5,
        reference_gradient=4.0,
        reference_response=1.0,
        reference_sensitivity=2.0,
    )
    second_backend = _RoutingBackend()
    second = _adapter(second_backend).observe_generation_routing(
        _base_latent(),
        sample_index=6,
        reference_gradient=4.0,
        reference_response=1.0,
        reference_sensitivity=2.0,
    )

    assert first.public_probe_domain_digest != second.public_probe_domain_digest
    assert first.public_probe_values_float32_be_sha256 != (
        second.public_probe_values_float32_be_sha256
    )
    actual_delta = (
        first_backend.decode_inputs[1].to(dtype=torch.float32)
        - first_backend.decode_inputs[0].to(dtype=torch.float32)
    )
    assert first.actual_probe_step == pytest.approx(
        sqrt(float(actual_delta.square().mean().item()))
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    (
        ("reference_gradient", 0.0),
        ("reference_response", float("nan")),
        ("reference_sensitivity", -1.0),
    ),
)
def test_runtime_routing_observation_rejects_invalid_references(
    field: str,
    value: float,
) -> None:
    adapter = _adapter(_RoutingBackend())
    arguments = {
        "sample_index": 0,
        "reference_gradient": 1.0,
        "reference_response": 1.0,
        "reference_sensitivity": 1.0,
    }
    arguments[field] = value

    with pytest.raises(RuntimeAdapterError, match="failed closed"):
        adapter.observe_generation_routing(_base_latent(), **arguments)
    assert adapter.state is RuntimeAdapterState.FAILED


@pytest.mark.unit
def test_runtime_routing_observation_rejects_missing_registered_callback() -> None:
    adapter = _adapter(_RoutingBackend(omit_previous_callback=True))

    with pytest.raises(RuntimeAdapterError, match="failed closed"):
        adapter.observe_generation_routing(
            _base_latent(),
            sample_index=0,
            reference_gradient=1.0,
            reference_response=1.0,
            reference_sensitivity=1.0,
        )
    assert adapter.state is RuntimeAdapterState.FAILED
