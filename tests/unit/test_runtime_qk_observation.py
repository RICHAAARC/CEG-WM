from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from experiments.methods import (
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.runners import (
    FormalRuntimeGeometryEstimationOperation,
)
from main import (
    QkLayerObservation,
    derive_public_noise_stream,
    differentiable_qk_relation_objective,
    qk_geometry_sync,
)
from runtime import (
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeBackendIdentity,
    RuntimeDetectionScheduleStep,
    RuntimeDeviceCapabilities,
    RuntimeGenerationWithSuffixContextResult,
    RuntimeQkForwardIdentity,
    RuntimeVaeFactors,
    create_runtime_adapter,
    observe_differentiable_detection_qk,
)


class FakePosterior:
    def __init__(
        self,
        mode_value: torch.Tensor,
        *,
        preserve_gradient: bool = False,
    ) -> None:
        self.mode_value = mode_value
        self.preserve_gradient = preserve_gradient
        self.mode_calls = 0
        self.sample_calls = 0

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        if self.preserve_gradient:
            return self.mode_value.clone()
        return self.mode_value.detach().clone()

    def sample(self) -> torch.Tensor:
        self.sample_calls += 1
        raise AssertionError("posterior sample path is forbidden")


class FakeProjection(torch.nn.Module):
    def __init__(
        self,
        output_width: int,
        *,
        offset: float,
        output_dtype: torch.dtype = torch.float16,
        output_device: str | None = None,
        nonfinite: bool = False,
    ) -> None:
        super().__init__()
        weight = (
            torch.arange(output_width * 2, dtype=torch.float32)
            .reshape(output_width, 2)
            .add(offset)
            .div(32.0)
            .to(dtype=torch.float16)
        )
        self.register_buffer("weight", weight)
        self.output_dtype = output_dtype
        self.output_device = output_device
        self.nonfinite = nonfinite

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(hidden_states, self.weight.transpose(0, 1))
        output = output.to(dtype=self.output_dtype)
        if self.output_device is not None:
            return torch.empty(
                output.shape,
                dtype=self.output_dtype,
                device=self.output_device,
            )
        if self.nonfinite:
            output = output.detach().clone()
            output[..., 0] = float("inf")
        return output


class CountingNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value * torch.tensor(
            0.5,
            dtype=value.dtype,
            device=value.device,
        )


class FakeAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        offset: float,
        query_width: int = 8,
        key_width: int = 8,
        query_offset_delta: float = 0.0,
        key_offset_delta: float = 0.0,
        query_dtype: torch.dtype = torch.float16,
        query_device: str | None = None,
        query_nonfinite: bool = False,
    ) -> None:
        super().__init__()
        self.heads = 2
        self.to_q = FakeProjection(
            query_width,
            offset=offset + query_offset_delta,
            output_dtype=query_dtype,
            output_device=query_device,
            nonfinite=query_nonfinite,
        )
        self.to_k = FakeProjection(
            key_width,
            offset=offset + 1.0 + key_offset_delta,
        )
        self.norm_q = CountingNorm()
        self.norm_k = CountingNorm()

    def project(self, hidden_states: torch.Tensor) -> None:
        self.to_q(hidden_states)
        self.to_k(hidden_states)


class FakeQkBackend:
    def __init__(
        self,
        *,
        missing_layer: str | None = None,
        skip_capture_layer: str | None = None,
        duplicate_capture_layer: str | None = None,
        alias_attention_layers: bool = False,
        query_width: int = 8,
        key_width: int = 8,
        query_offset_delta: float = 0.0,
        key_offset_delta: float = 0.0,
        query_dtype: torch.dtype = torch.float16,
        query_device: str | None = None,
        query_nonfinite: bool = False,
        schedule_index_drift: int | None = None,
        forward_identity_drift: tuple[str, object] | None = None,
        scale_noise_failure: bool = False,
        scale_noise_output_mode: str = "valid",
    ) -> None:
        first = FakeAttention(
            offset=1.0,
            query_width=query_width,
            key_width=key_width,
            query_offset_delta=query_offset_delta,
            key_offset_delta=key_offset_delta,
            query_dtype=query_dtype,
            query_device=query_device,
            query_nonfinite=query_nonfinite,
        )
        second = (
            first
            if alias_attention_layers
            else FakeAttention(
                offset=3.0,
                query_offset_delta=query_offset_delta,
                key_offset_delta=key_offset_delta,
            )
        )
        self.attentions = {
            "transformer_blocks.0.attn": first,
            "transformer_blocks.23.attn": second,
        }
        self.missing_layer = missing_layer
        self.skip_capture_layer = skip_capture_layer
        self.duplicate_capture_layer = duplicate_capture_layer
        self.schedule_index_drift = schedule_index_drift
        self.forward_identity_drift = forward_identity_drift
        self.scale_noise_failure = scale_noise_failure
        self.scale_noise_output_mode = scale_noise_output_mode
        self.configuration = None
        self.selected_device = None
        self.close_calls = 0
        self.posterior: FakePosterior | None = None
        self.conditioning_calls = []
        self.scale_noise_inputs = []
        self.execution_callback = None

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device):
        self.configuration = configuration
        self.selected_device = selected_device
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="synthetic_qk_backend",
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
        self.close_calls += 1

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=2.0, shift_factor=0.25)

    def vae_encode(self, image: torch.Tensor) -> FakePosterior:
        if self.execution_callback is not None:
            self.execution_callback()
        image_mean = image.to(dtype=torch.float32).mean()
        mode = (
            torch.arange(32, dtype=torch.float32)
            .reshape(1, 2, 4, 4)
            .div(64.0)
            .add(image_mean)
        )
        self.posterior = FakePosterior(mode)
        return self.posterior

    def vae_encode_differentiable(self, image: torch.Tensor) -> FakePosterior:
        image_mean = image.to(dtype=torch.float32).mean()
        mode = (
            torch.arange(32, dtype=torch.float32)
            .reshape(1, 2, 4, 4)
            .div(64.0)
            .add(image_mean)
        )
        self.posterior = FakePosterior(mode, preserve_gradient=True)
        return self.posterior

    def create_detection_schedule(
        self,
        inference_steps: int,
    ) -> RuntimeDetectionScheduleStep:
        assert self.configuration is not None
        return RuntimeDetectionScheduleStep(
            scheduler_class=self.configuration.scheduler_class,
            inference_steps=inference_steps,
            detection_schedule_index=(
                self.configuration.detection_schedule_index
                if self.schedule_index_drift is None
                else self.schedule_index_drift
            ),
            detection_timestep=torch.tensor(731.0, dtype=torch.float32),
        )

    def scale_detection_noise(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self.scale_noise_inputs.append(
            (
                detection_latent.detach().clone(),
                public_noise.detach().clone(),
                timestep.detach().clone(),
            )
        )
        if self.scale_noise_failure:
            raise RuntimeError("fake scale_noise failure")
        output = (
            detection_latent.to(dtype=torch.float32)
            + public_noise.to(dtype=torch.float32) * 0.125
        ).to(dtype=torch.float16)
        if self.scale_noise_output_mode == "shape":
            return output[..., :-1]
        if self.scale_noise_output_mode == "dtype":
            return output.to(dtype=torch.float32)
        if self.scale_noise_output_mode == "device":
            return torch.empty(
                output.shape,
                dtype=torch.float16,
                device="meta",
            )
        if self.scale_noise_output_mode == "nonfinite":
            output = output.detach().clone()
            output.reshape(-1)[0] = float("inf")
            return output
        assert self.scale_noise_output_mode == "valid"
        return output

    def scale_detection_noise_differentiable(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.scale_detection_noise(
            detection_latent,
            public_noise,
            timestep,
        )

    def attention_module(self, layer_name: str) -> torch.nn.Module:
        if layer_name == self.missing_layer:
            raise KeyError(layer_name)
        return self.attentions[layer_name]

    def run_qk_detection_forward(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning,
    ) -> RuntimeQkForwardIdentity:
        assert self.configuration is not None
        self.conditioning_calls.append(conditioning)
        hidden_states = noisy_detection_latent.flatten(2).transpose(1, 2)
        for layer_name in self.configuration.qk_layer_names:
            if layer_name == self.skip_capture_layer:
                continue
            self.attentions[layer_name].project(hidden_states)
            if layer_name == self.duplicate_capture_layer:
                self.attentions[layer_name].project(hidden_states)
        identity = RuntimeQkForwardIdentity(
            runtime_config_digest=self.configuration.runtime_config_digest,
            model_id=self.configuration.model_id,
            model_revision=self.configuration.model_revision,
            scheduler_class=self.configuration.scheduler_class,
            inference_steps=self.configuration.inference_steps,
            detection_schedule_index=(
                self.configuration.detection_schedule_index
            ),
            detection_conditioning_protocol=(
                self.configuration.detection_conditioning_protocol
            ),
            prompt=conditioning.prompt,
            prompt_2=conditioning.prompt_2,
            prompt_3=conditioning.prompt_3,
            do_classifier_free_guidance=(
                conditioning.do_classifier_free_guidance
            ),
            qk_layer_names=self.configuration.qk_layer_names,
        )
        if self.forward_identity_drift is not None:
            field, value = self.forward_identity_drift
            identity = replace(identity, **{field: value})
        return identity

    def run_qk_detection_forward_differentiable(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning,
    ) -> RuntimeQkForwardIdentity:
        return self.run_qk_detection_forward(
            noisy_detection_latent,
            timestep,
            conditioning,
        )


@dataclass(frozen=True, slots=True)
class FakeGenerationSuffixContext:
    runtime_config_digest: str
    callback_index: int


class FakeGeometrySynchronizationBackend(FakeQkBackend):
    def __init__(self) -> None:
        super().__init__()
        self.suffix_replay_modes: list[bool] = []

    def run_generation(self, initial_latent, callback):
        state = initial_latent.detach().clone()
        for callback_index in range(20):
            state = callback(callback_index, state)
        return state

    def run_generation_with_suffix_context(
        self,
        initial_latent,
        callback,
    ) -> RuntimeGenerationWithSuffixContextResult:
        assert self.configuration is not None
        return RuntimeGenerationWithSuffixContextResult(
            terminal_latent=self.run_generation(initial_latent, callback),
            suffix_context=FakeGenerationSuffixContext(
                runtime_config_digest=self.configuration.runtime_config_digest,
                callback_index=self.configuration.callback_index,
            ),
        )

    def replay_generation_suffix(
        self,
        callback_latent,
        suffix_context,
        *,
        differentiable,
    ):
        assert self.configuration is not None
        assert suffix_context.runtime_config_digest == (
            self.configuration.runtime_config_digest
        )
        assert suffix_context.callback_index == self.configuration.callback_index
        self.suffix_replay_modes.append(differentiable)
        return callback_latent.to(dtype=torch.float16) * torch.tensor(
            0.5,
            dtype=torch.float16,
            device=callback_latent.device,
        )

    @staticmethod
    def _decoded_image(latent: torch.Tensor) -> torch.Tensor:
        value = torch.sigmoid(latent.to(dtype=torch.float32).mean()).reshape(
            1,
            1,
            1,
            1,
        )
        return value.expand(1, 3, 512, 512).clone()

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoded_image(latent).detach()

    def vae_decode_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoded_image(latent)


def _image(value: float = 0.5) -> torch.Tensor:
    return torch.full((1, 3, 512, 512), value, dtype=torch.float32)


def _observe(backend: FakeQkBackend, image: torch.Tensor | None = None):
    adapter = create_runtime_adapter(backend)
    session = adapter.initialize("cpu")
    assert session.selected_device == "cpu"
    result = adapter.observe_detection_qk(_image() if image is None else image)
    return adapter, result


def _assert_projection_hooks_removed(backend: FakeQkBackend) -> None:
    for attention in backend.attentions.values():
        assert not attention.to_q._forward_hooks
        assert not attention.to_k._forward_hooks


@pytest.mark.quick
def test_formal_geometry_ready_public_call_chain_succeeds() -> None:
    backend = FakeQkBackend()
    runtime_adapter = create_runtime_adapter(backend)
    runtime_adapter.initialize("cpu")
    operation = FormalRuntimeGeometryEstimationOperation(
        runtime_adapter=runtime_adapter,
        adapter_configuration=(
            load_ceg_wm_experiment_adapter_configuration(
                "configs/experiments/internal_execution_components.json"
            )
        ),
        epsilon_inlier=0.8,
        execution_scope="cpu_synthetic_wiring_only",
    )

    result = operation(_image(), "formal-geometry-registered-key")

    assert result.candidate_ids[1] == "qk_relation_similarity"
    assert result.candidate_ids[2] == "rectification_similarity"
    assert result.estimation_identity_digest
    operation._method_adapter.require_no_runtime_binding()


@pytest.mark.unit
def test_qk_observation_uses_mode_public_noise_and_real_projection_hooks() -> None:
    first_backend = FakeQkBackend()
    first_adapter, first = _observe(first_backend)
    second_backend = FakeQkBackend()
    second_adapter, second = _observe(second_backend)

    assert first.candidate_id == "runtime_sd35_flowmatch"
    assert first.detection_schedule_index == 7
    assert first.detection_conditioning_protocol == (
        "sd3_empty_text_triplet_without_cfg"
    )
    assert first.qk_actual_dtype == "float16"
    assert len(first.qk_layer_observations) == 2
    assert all(
        type(observation) is QkLayerObservation
        for observation in first.qk_layer_observations
    )
    assert tuple(
        observation.layer_name
        for observation in first.qk_layer_observations
    ) == (
        "transformer_blocks.0.attn",
        "transformer_blocks.23.attn",
    )
    for observation in first.qk_layer_observations:
        assert observation.query.shape == (2, 16, 4)
        assert observation.attention_key.shape == (2, 16, 4)
        assert observation.query.dtype is torch.float16
        assert observation.attention_key.dtype is torch.float16
        assert torch.isfinite(observation.query).all()
        assert torch.isfinite(observation.attention_key).all()
        assert "sd35_real_to_q_to_k" in observation.operator_identity
        assert "relation_scale=inverse_sqrt_head_width" in (
            observation.operator_identity
        )
    assert first_backend.posterior is not None
    assert first_backend.posterior.mode_calls == 1
    assert first_backend.posterior.sample_calls == 0
    for attention in first_backend.attentions.values():
        assert isinstance(attention.norm_q, CountingNorm)
        assert isinstance(attention.norm_k, CountingNorm)
        assert attention.norm_q.calls == 1
        assert attention.norm_k.calls == 1
    _assert_projection_hooks_removed(first_backend)
    conditioning = first_backend.conditioning_calls[0]
    assert (conditioning.prompt, conditioning.prompt_2, conditioning.prompt_3) == (
        "",
        "",
        "",
    )
    assert conditioning.do_classifier_free_guidance is False
    assert len(first_backend.scale_noise_inputs) == 1
    scale_latent, scale_noise, scale_timestep = (
        first_backend.scale_noise_inputs[0]
    )
    assert first_backend.posterior is not None
    expected_detection_latent = (
        first_backend.posterior.mode_value - 0.25
    ) * 2.0
    assert torch.equal(
        scale_latent,
        expected_detection_latent.to(dtype=torch.float16),
    )
    expected_noise_stream = derive_public_noise_stream(
        {
            "candidate_id": "qk_relation_similarity",
            "operator": "public_image_only_qk_detection_noise",
            "responsibility_domain": "public_noise",
            "model_revision": first.model_revision,
            "schedule_index": 7,
            "conditioning_protocol": (
                "sd3_empty_text_triplet_without_cfg"
            ),
            "tensor_role": "scheduler_noise",
        },
        tuple(int(size) for size in scale_latent.shape),
    )
    expected_noise = torch.tensor(
        expected_noise_stream.values,
        dtype=torch.float32,
    ).reshape(scale_latent.shape).to(dtype=torch.float16)
    assert torch.equal(scale_noise, expected_noise)
    assert scale_timestep.dtype is torch.float32
    assert scale_timestep.numel() == 1
    assert float(scale_timestep.item()) == 731.0
    assert first.public_noise_domain_digest == second.public_noise_domain_digest
    assert (
        first.public_noise_values_float32_be_sha256
        == second.public_noise_values_float32_be_sha256
    )
    assert all(
        torch.equal(left.query, right.query)
        and torch.equal(left.attention_key, right.attention_key)
        for left, right in zip(
            first.qk_layer_observations,
            second.qk_layer_observations,
            strict=True,
        )
    )
    first_adapter.close()
    second_adapter.close()


@pytest.mark.unit
def test_differentiable_qk_observation_preserves_the_public_score_and_gradient(
) -> None:
    backend = FakeQkBackend()
    adapter = create_runtime_adapter(backend)
    session = adapter.initialize("cpu")
    image = _image().requires_grad_(True)

    differentiable = observe_differentiable_detection_qk(
        backend,
        adapter.configuration,
        session,
        image,
    )
    objective = differentiable_qk_relation_objective(
        differentiable.qk_layer_observations,
        "geometry-cpu-synthetic-key",
    )
    gradient = torch.autograd.grad(objective, image)[0]

    assert objective.requires_grad
    assert gradient.shape == image.shape
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0.0
    assert all(
        observation.query.requires_grad
        and observation.attention_key.requires_grad
        for observation in differentiable.qk_layer_observations
    )

    formal = adapter.observe_detection_qk(image.detach())
    formal_score = qk_geometry_sync(
        formal.qk_layer_observations,
        "geometry-cpu-synthetic-key",
    )
    assert float(objective.detach()) == formal_score.relation_score
    assert differentiable.runtime_config_digest == formal.runtime_config_digest
    assert (
        differentiable.public_noise_domain_digest
        == formal.public_noise_domain_digest
    )
    assert (
        differentiable.public_noise_values_float32_be_sha256
        == formal.public_noise_values_float32_be_sha256
    )
    assert all(
        torch.equal(left.query.detach(), right.query)
        and torch.equal(left.attention_key.detach(), right.attention_key)
        for left, right in zip(
            differentiable.qk_layer_observations,
            formal.qk_layer_observations,
            strict=True,
        )
    )
    assert all(
        not observation.query.requires_grad
        and not observation.attention_key.requires_grad
        for observation in formal.qk_layer_observations
    )
    _assert_projection_hooks_removed(backend)
    adapter.close()


@pytest.mark.unit
def test_geometry_suffix_replay_uses_rgb8_ste_for_gradient_and_blind_actual_qk(
) -> None:
    backend = FakeGeometrySynchronizationBackend()
    adapter = create_runtime_adapter(backend)
    session = adapter.initialize("cpu")
    context = FakeGenerationSuffixContext(
        runtime_config_digest=session.runtime_config_digest,
        callback_index=session.callback_index,
    )
    content_written = torch.linspace(
        -0.25,
        0.25,
        steps=32,
        dtype=torch.float16,
    ).reshape(1, 2, 4, 4)

    differentiable = adapter.observe_differentiable_qk_from_generation_suffix(
        context,
        content_written,
    )
    objective = differentiable_qk_relation_objective(
        differentiable.qk_observation.qk_layer_observations,
        "geometry-cpu-synthetic-key",
    )
    gradient = torch.autograd.grad(
        objective,
        differentiable.callback_latent_float32,
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0.0

    candidate_actual = adapter.materialize_geometry_candidate(
        differentiable.callback_latent_float32,
        expected_shape=content_written.shape,
        expected_device=content_written.device,
    )
    actual = adapter.observe_actual_qk_from_generation_suffix(
        context,
        candidate_actual,
    )

    assert backend.suffix_replay_modes == [True, False]
    assert candidate_actual.dtype is torch.float16
    assert torch.equal(
        differentiable.rgb8_ste_image.detach(),
        actual.rgb8_image,
    )
    differentiable_score = qk_geometry_sync(
        differentiable.qk_observation.qk_layer_observations,
        "geometry-cpu-synthetic-key",
    )
    actual_score = qk_geometry_sync(
        actual.qk_observation.qk_layer_observations,
        "geometry-cpu-synthetic-key",
    )
    assert differentiable_score.relation_score == actual_score.relation_score
    assert all(
        not observation.query.requires_grad
        and not observation.attention_key.requires_grad
        for observation in actual.qk_observation.qk_layer_observations
    )
    assert not hasattr(actual, "suffix_context")
    adapter.close()


@pytest.mark.unit
def test_geometry_suffix_replay_rejects_foreign_context_and_closes_runtime(
) -> None:
    backend = FakeGeometrySynchronizationBackend()
    adapter = create_runtime_adapter(backend)
    session = adapter.initialize("cpu")
    foreign_context = FakeGenerationSuffixContext(
        runtime_config_digest="0" * 64,
        callback_index=session.callback_index,
    )

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.observe_differentiable_qk_from_generation_suffix(
            foreign_context,
            torch.zeros((1, 2, 4, 4), dtype=torch.float16),
        )

    assert exc_info.value.__cause__ is not None
    assert "identity drifted" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    assert backend.suffix_replay_modes == []


@pytest.mark.unit
def test_qk_observation_values_depend_on_the_detection_image() -> None:
    first_adapter, first = _observe(FakeQkBackend(), _image(0.25))
    second_adapter, second = _observe(FakeQkBackend(), _image(0.75))

    assert first.public_noise_domain_digest == second.public_noise_domain_digest
    assert any(
        not torch.equal(left.query, right.query)
        or not torch.equal(left.attention_key, right.attention_key)
        for left, right in zip(
            first.qk_layer_observations,
            second.qk_layer_observations,
            strict=True,
        )
    )
    first_adapter.close()
    second_adapter.close()


@pytest.mark.unit
def test_qk_observation_values_depend_on_actual_projection_parameters() -> None:
    image = _image()
    base_adapter, base = _observe(FakeQkBackend(), image)
    query_adapter, query_changed = _observe(
        FakeQkBackend(query_offset_delta=5.0),
        image,
    )
    key_adapter, key_changed = _observe(
        FakeQkBackend(key_offset_delta=7.0),
        image,
    )

    results = (base, query_changed, key_changed)
    assert len({result.public_noise_domain_digest for result in results}) == 1
    assert len(
        {
            result.public_noise_values_float32_be_sha256
            for result in results
        }
    ) == 1
    assert all(
        result.runtime_config_digest == base.runtime_config_digest
        and result.model_revision == base.model_revision
        and result.detection_timestep == base.detection_timestep
        for result in results
    )
    for base_layer, query_layer, key_layer in zip(
        base.qk_layer_observations,
        query_changed.qk_layer_observations,
        key_changed.qk_layer_observations,
        strict=True,
    ):
        assert not torch.equal(base_layer.query, query_layer.query)
        assert torch.equal(
            base_layer.attention_key,
            query_layer.attention_key,
        )
        assert torch.equal(base_layer.query, key_layer.query)
        assert not torch.equal(
            base_layer.attention_key,
            key_layer.attention_key,
        )
    base_adapter.close()
    query_adapter.close()
    key_adapter.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (
            FakeQkBackend(missing_layer="transformer_blocks.23.attn"),
            "unavailable",
        ),
        (
            FakeQkBackend(skip_capture_layer="transformer_blocks.23.attn"),
            "not captured exactly once",
        ),
        (
            FakeQkBackend(duplicate_capture_layer="transformer_blocks.0.attn"),
            "captured more than once",
        ),
        (
            FakeQkBackend(alias_attention_layers=True),
            "alias attention or projection",
        ),
        (
            FakeQkBackend(key_width=6),
            "shapes do not match",
        ),
        (
            FakeQkBackend(query_dtype=torch.float32),
            "actual dtype or device drifted",
        ),
        (
            FakeQkBackend(query_device="meta"),
            "actual dtype or device drifted",
        ),
        (
            FakeQkBackend(query_nonfinite=True),
            "normalized output is invalid",
        ),
        (
            FakeQkBackend(schedule_index_drift=8),
            "schedule identity drifted",
        ),
        (
            FakeQkBackend(
                forward_identity_drift=("model_revision", "main")
            ),
            "model_revision identity drifted",
        ),
        (
            FakeQkBackend(
                forward_identity_drift=(
                    "scheduler_class",
                    "diffusers.DDIMScheduler",
                )
            ),
            "scheduler_class identity drifted",
        ),
        (
            FakeQkBackend(
                forward_identity_drift=(
                    "qk_layer_names",
                    (
                        "transformer_blocks.23.attn",
                        "transformer_blocks.0.attn",
                    ),
                )
            ),
            "qk_layer_names identity drifted",
        ),
        (
            FakeQkBackend(
                forward_identity_drift=(
                    "do_classifier_free_guidance",
                    True,
                )
            ),
            "do_classifier_free_guidance identity drifted",
        ),
    ],
)
def test_qk_observation_failures_close_adapter(
    backend: FakeQkBackend,
    message: str,
) -> None:
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.observe_detection_qk(_image())

    assert exc_info.value.__cause__ is not None
    assert message in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    _assert_projection_hooks_removed(backend)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (
            FakeQkBackend(scale_noise_failure=True),
            "scheduler scale_noise failed",
        ),
        (
            FakeQkBackend(scale_noise_output_mode="shape"),
            "invalid latent",
        ),
        (
            FakeQkBackend(scale_noise_output_mode="dtype"),
            "invalid latent",
        ),
        (
            FakeQkBackend(scale_noise_output_mode="device"),
            "invalid latent",
        ),
        (
            FakeQkBackend(scale_noise_output_mode="nonfinite"),
            "invalid latent",
        ),
    ],
)
def test_qk_observation_scale_noise_failures_close_and_cleanup(
    backend: FakeQkBackend,
    message: str,
) -> None:
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.observe_detection_qk(_image())

    assert exc_info.value.__cause__ is not None
    assert message in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    assert len(backend.scale_noise_inputs) == 1
    _assert_projection_hooks_removed(backend)


@pytest.mark.unit
def test_qk_observation_rejects_wrong_image_boundary() -> None:
    backend = FakeQkBackend()
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.observe_detection_qk(
            torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        )

    assert exc_info.value.__cause__ is not None
    assert "frozen-resolution RGB image" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
