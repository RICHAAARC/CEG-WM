from dataclasses import dataclass, replace
from hashlib import sha256
import inspect
from math import sqrt
from struct import pack, unpack

import pytest
import torch

import runtime.content_write as runtime_content_write
from experiments.methods.ceg_wm import CegWmExperimentAdapter
from main import (
    ContentEmbeddingResult,
    content_actual_budget_accepts,
    content_embedder,
    semantic_texture_content_embedder,
    semantic_texture_content_router,
)
from main.content_chain.hf_carrier import hf_carrier
from main.content_chain.lf_carrier import lf_carrier
from runtime import (
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeGenerationWithSuffixContextResult,
    RuntimeVaeFactors,
    InspyrenetSemanticRuntime,
    create_runtime_adapter,
    materialize_ordinary_rgb8_snapshot,
    measure_content_materialization,
)


TEST_ROOT_KEY = "ceg-wm-runtime-batch-two-root"
TEST_SHAPE = (1, 1, 4, 4)


@pytest.mark.unit
def test_public_ordinary_rgb8_snapshot_matches_runtime_callback_contract() -> None:
    image = torch.tensor(
        [[[
            [-0.1, 0.0, 0.0039, 0.5, 0.9999, 1.0, 1.1],
        ]] * 3],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [[[
            [0, 0, 0, 127, 254, 255, 255],
        ]] * 3],
        dtype=torch.uint8,
    )
    assert torch.equal(materialize_ordinary_rgb8_snapshot(image), expected)
    assert torch.equal(
        runtime_content_write.materialize_ordinary_rgb8_snapshot(image),
        expected,
    )
    with pytest.raises(runtime_content_write.RuntimeContentExecutionError):
        materialize_ordinary_rgb8_snapshot(torch.zeros((1, 1, 2, 2)))
    with pytest.raises(runtime_content_write.RuntimeContentExecutionError):
        materialize_ordinary_rgb8_snapshot("not-a-tensor")
    source = inspect.getsource(
        CegWmExperimentAdapter.prepare_semantic_texture_clean_primary_null
    )
    assert "materialize_ordinary_rgb8_snapshot(clean.clean_image)" in source
    assert "carrier = lf_carrier(\n            detection_key,\n            route.latent_shape,\n        )" in source
    for private_conversion_token in ("clamp(", ".floor(", "uint8"):
        assert private_conversion_token not in source


def _identity(configuration, selected_device: str) -> RuntimeBackendIdentity:
    return RuntimeBackendIdentity(
        candidate_id=configuration.candidate_id,
        runtime_config_digest=configuration.runtime_config_digest,
        runtime_backend_name="synthetic_cpu_content_backend",
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


class FakePosterior:
    def __init__(self, mode_value: torch.Tensor) -> None:
        self._mode_value = mode_value
        self.mode_calls = 0
        self.sample_calls = 0

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self._mode_value.detach().clone()

    def sample(self) -> torch.Tensor:
        self.sample_calls += 1
        raise AssertionError("detection encode must never sample the posterior")


@dataclass(frozen=True, slots=True)
class FakeGenerationSuffixContext:
    runtime_config_digest: str
    callback_index: int


class FakeContentBackend:
    def __init__(
        self,
        callback_sequences: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        *,
        diverge_second_run: bool = False,
        erase_second_suffix: bool = False,
        invalid_posterior: bool = False,
        nonfinite_posterior_mode: bool = False,
        invalid_vae_factors: bool = False,
        generation_failure: bool = False,
    ) -> None:
        default = tuple(range(20))
        self.callback_sequences = callback_sequences or (default, default)
        self.diverge_second_run = diverge_second_run
        self.erase_second_suffix = erase_second_suffix
        self.invalid_posterior = invalid_posterior
        self.nonfinite_posterior_mode = nonfinite_posterior_mode
        self.invalid_vae_factors = invalid_vae_factors
        self.generation_failure = generation_failure
        self.run_calls = 0
        self.suffix_capture_calls = 0
        self.suffix_callback_latents: list[torch.Tensor] = []
        self.close_calls = 0
        self.decode_inputs: list[torch.Tensor] = []
        self.posteriors: list[FakePosterior] = []
        self.configuration = None

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device: str):
        self.configuration = configuration
        return _identity(configuration, selected_device)

    def close(self) -> None:
        self.close_calls += 1

    def run_generation(self, initial_latent, callback):
        if self.generation_failure:
            raise RuntimeError("synthetic clean generation failure")
        run_index = self.run_calls
        self.run_calls += 1
        state = initial_latent.detach().clone()
        for callback_index in self.callback_sequences[run_index]:
            if (
                self.erase_second_suffix
                and run_index == 1
                and callback_index == 19
            ):
                state = initial_latent.detach().clone()
            if (
                self.diverge_second_run
                and run_index == 1
                and callback_index == 0
            ):
                state = (state.to(torch.float32) + 0.25).to(torch.float16)
            state = callback(callback_index, state)
        return state

    def run_generation_with_suffix_context(
        self,
        initial_latent,
        callback,
    ) -> RuntimeGenerationWithSuffixContextResult:
        assert self.configuration is not None
        self.suffix_capture_calls += 1
        def capture_callback(
            callback_index: int,
            latent: torch.Tensor,
        ) -> torch.Tensor:
            returned = callback(callback_index, latent)
            if callback_index == self.configuration.callback_index:
                self.suffix_callback_latents.append(returned.detach().clone())
            return returned

        return RuntimeGenerationWithSuffixContextResult(
            terminal_latent=self.run_generation(initial_latent, capture_callback),
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
        assert suffix_context.runtime_config_digest == (
            self.configuration.runtime_config_digest
        )
        assert type(differentiable) is bool
        return callback_latent.clone()

    def vae_factors(self) -> RuntimeVaeFactors:
        if self.invalid_vae_factors:
            return object()
        return RuntimeVaeFactors(
            scaling_factor=0.5,
            shift_factor=0.25,
        )

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        return latent.detach().clone() * 2.0

    def vae_decode_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clone() * 2.0

    def vae_encode(self, image: torch.Tensor):
        if self.invalid_posterior:
            return object()
        mode_value = image.detach().clone() / 4.0
        if self.nonfinite_posterior_mode:
            mode_value.reshape(-1)[0] = float("nan")
        posterior = FakePosterior(mode_value)
        self.posteriors.append(posterior)
        return posterior


def _base_latent() -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        steps=16,
        dtype=torch.float32,
    ).reshape(TEST_SHAPE).to(torch.float16)


def _full_scale_accepted_latent() -> torch.Tensor:
    return torch.tensor(
        (
            -0.0007700920104980469,
            -0.0102081298828125,
            -0.001689910888671875,
            0.00917816162109375,
            0.01580810546875,
            0.01300811767578125,
            0.01275634765625,
            -0.002010345458984375,
            0.004962921142578125,
            -0.015716552734375,
            0.00966644287109375,
            -0.01148223876953125,
            -0.01158905029296875,
            0.003253936767578125,
            -0.006313323974609375,
            -0.0283966064453125,
        ),
        dtype=torch.float16,
    ).reshape(TEST_SHAPE)


def _embedding_operation(calls: list[tuple[float, ...]]):
    carrier = hf_carrier(TEST_ROOT_KEY, TEST_SHAPE)

    def operation(values: tuple[float, ...]) -> ContentEmbeddingResult:
        calls.append(values)
        return content_embedder(values, carrier)

    return operation


def _initialized_adapter(backend: FakeContentBackend):
    adapter = create_runtime_adapter(backend)
    adapter.initialize("cpu")
    return adapter


class _InputDependentSemanticModel(torch.nn.Module):
    def forward_inspyre(self, value: torch.Tensor) -> dict[str, object]:
        finest_saliency_logit = value[:, :1]
        return {
            "saliency": [finest_saliency_logit] * 4,
            "laplacian": [finest_saliency_logit] * 3,
        }


class _SemanticTextureContentBackend(FakeContentBackend):
    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        return torch.sigmoid(latent[:, :3].to(dtype=torch.float32))


def _semantic_texture_base_latent() -> torch.Tensor:
    spatial_values = torch.linspace(
        -1.0,
        1.0,
        steps=64 * 64,
        dtype=torch.float32,
    ).reshape(1, 1, 64, 64)
    return spatial_values.repeat(1, 16, 1, 1).to(dtype=torch.float16)


def _semantic_texture_composition(
    values: tuple[float, ...],
    shape: tuple[int, ...],
    runtime_observation: object,
):
    route = semantic_texture_content_router(
        shape,
        mode="routing_semantic_texture_soft",
        observations=runtime_observation.observations,
    )
    hf_result = hf_carrier(TEST_ROOT_KEY, shape, routing_result=route)
    lf_result = lf_carrier(TEST_ROOT_KEY, shape, routing_result=route)
    embedding = semantic_texture_content_embedder(
        values,
        hf_result,
        lf_carrier_result=lf_result,
        routing_result=route,
    )
    return route, embedding


@pytest.mark.unit
def test_clean_image_observation_runs_one_generation_decode_encode_and_mode(
) -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    base = _base_latent()
    base_before = base.detach().clone()

    result = adapter.execute_clean_image_and_vae_observation(base)

    assert adapter.state is RuntimeAdapterState.READY
    assert torch.equal(base, base_before)
    assert backend.run_calls == 1
    assert result.clean_callback_indices == tuple(range(20))
    assert len(backend.decode_inputs) == 1
    assert len(backend.posteriors) == 1
    assert backend.posteriors[0].mode_calls == 1
    assert backend.posteriors[0].sample_calls == 0
    assert torch.equal(
        backend.decode_inputs[0],
        result.clean_generation_terminal_latent.to(torch.float32) / 0.5 + 0.25,
    )
    assert torch.equal(
        result.clean_detection_latent,
        (result.clean_image / 4.0 - 0.25) * 0.5,
    )
    assert result.candidate_id == adapter.session.candidate_id
    assert result.runtime_config_digest == adapter.session.runtime_config_digest
    assert result.selected_device == "cpu"
    assert len(result.clean_base_latent_digest) == 64
    assert not hasattr(result, "watermarked_image")
    assert not hasattr(result, "content_materialization")


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend,error_match",
    [
        (
            FakeContentBackend(
                callback_sequences=(
                    tuple(index for index in range(20) if index != 18),
                    tuple(range(20)),
                )
            ),
            "missing or out of order",
        ),
        (FakeContentBackend(invalid_posterior=True), "posterior mode boundary"),
        (
            FakeContentBackend(nonfinite_posterior_mode=True),
            "posterior_mode contains non-finite",
        ),
        (FakeContentBackend(invalid_vae_factors=True), "VAE factors"),
        (FakeContentBackend(generation_failure=True), "generation backend failed"),
    ],
)
def test_clean_image_observation_fails_closed_on_runtime_boundary(
    backend: FakeContentBackend,
    error_match: str,
) -> None:
    adapter = _initialized_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_clean_image_and_vae_observation(_base_latent())

    assert error_match in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1
    assert backend.run_calls <= 1


@pytest.mark.unit
def test_clean_image_observation_requires_ready_content_backend() -> None:
    backend = FakeContentBackend()
    adapter = create_runtime_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="must be ready"):
        adapter.execute_clean_image_and_vae_observation(_base_latent())

    assert backend.run_calls == 0
    assert backend.decode_inputs == []
    assert backend.posteriors == []


@pytest.mark.unit
def test_clean_image_observation_matches_existing_paired_clean_path() -> None:
    clean_backend = FakeContentBackend()
    paired_backend = FakeContentBackend()
    clean_adapter = _initialized_adapter(clean_backend)
    paired_adapter = _initialized_adapter(paired_backend)
    base = _base_latent()

    clean = clean_adapter.execute_clean_image_and_vae_observation(base)
    paired = paired_adapter.execute_content_write_and_vae(
        base,
        _embedding_operation([]),
    )

    assert clean.clean_base_latent_digest == paired.paired_base_latent_digest
    assert clean.clean_callback_indices == paired.clean_callback_indices
    assert torch.equal(
        clean.clean_generation_terminal_latent,
        paired.clean_generation_terminal_latent,
    )
    assert clean.vae_scaling_factor_actual == paired.vae_scaling_factor_actual
    assert clean.vae_shift_factor_actual == paired.vae_shift_factor_actual
    assert torch.equal(clean.clean_image, paired.clean_image)
    assert torch.equal(
        clean.clean_detection_latent,
        paired.clean_detection_latent,
    )
    assert clean_backend.run_calls == 1
    assert paired_backend.run_calls == 2
    assert paired_backend.suffix_capture_calls == 0


@pytest.mark.unit
def test_semantic_texture_production_traversal_rejects_missing_registered_write_callback_index() -> None:
    full_sequence = tuple(range(20))
    missing_callback = tuple(index for index in full_sequence if index != 18)
    backend = _SemanticTextureContentBackend(
        callback_sequences=(full_sequence, missing_callback)
    )
    adapter = _initialized_adapter(backend)
    semantic_runtime = InspyrenetSemanticRuntime.from_injected_model_for_test(
        _InputDependentSemanticModel()
    )

    with pytest.raises(
        runtime_content_write.RuntimeContentExecutionError,
        match="missing or out of order",
    ):
        runtime_content_write.execute_semantic_texture_content_write_and_vae(
            backend,
            adapter.configuration,
            adapter.session,
            _semantic_texture_base_latent(),
            semantic_runtime,
            _semantic_texture_composition,
        )

    assert backend.decode_inputs == []


@pytest.mark.unit
def test_semantic_texture_traversal_rejects_rgb8_witness_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SemanticTextureContentBackend()
    adapter = _initialized_adapter(backend)
    semantic_runtime = InspyrenetSemanticRuntime.from_injected_model_for_test(
        _InputDependentSemanticModel()
    )
    monkeypatch.setattr(
        runtime_content_write,
        "rgb8_image_digest",
        lambda _image: "0" * 64,
    )

    with pytest.raises(
        runtime_content_write.RuntimeContentExecutionError,
        match="RGB8 identity mismatched callback",
    ):
        runtime_content_write.execute_semantic_texture_content_write_and_vae(
            backend,
            adapter.configuration,
            adapter.session,
            _semantic_texture_base_latent(),
            semantic_runtime,
            _semantic_texture_composition,
        )

    assert len(backend.decode_inputs) == 1


@pytest.mark.unit
def test_semantic_texture_traversal_rejects_route_observation_mismatch() -> None:
    backend = _SemanticTextureContentBackend()
    adapter = _initialized_adapter(backend)
    semantic_runtime = InspyrenetSemanticRuntime.from_injected_model_for_test(
        _InputDependentSemanticModel()
    )

    def stale_route_composition(values, shape, runtime_observation):
        route, embedding = _semantic_texture_composition(
            values,
            shape,
            runtime_observation,
        )
        return replace(route, observations=None), embedding

    with pytest.raises(
        runtime_content_write.RuntimeContentExecutionError,
        match="did not consume the callback observation",
    ):
        runtime_content_write.execute_semantic_texture_content_write_and_vae(
            backend,
            adapter.configuration,
            adapter.session,
            _semantic_texture_base_latent(),
            semantic_runtime,
            stale_route_composition,
        )

    assert len(backend.decode_inputs) == 1


@pytest.mark.unit
def test_paired_write_uses_main_budget_result_and_maximal_scale() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    calls: list[tuple[float, ...]] = []
    base = _base_latent()
    base_before = base.detach().clone()

    result = adapter.execute_content_write_and_vae(
        base,
        _embedding_operation(calls),
    )

    measurement = result.content_materialization
    assert adapter.state is RuntimeAdapterState.READY
    assert torch.equal(base, base_before)
    assert result.clean_callback_indices == tuple(range(20))
    assert result.watermarked_callback_indices == tuple(range(20))
    assert len(calls) == 1
    assert calls[0] == tuple(
        float(value)
        for value in measurement.baseline_latent_actual.to(
            torch.float32
        ).reshape(-1)
    )
    assert measurement.callback_index == 18
    assert measurement.delta_content_actual.dtype is torch.float32
    assert measurement.realized_total_l2 > 0.0
    assert measurement.realized_relative_l2 > 0.0
    method_result = result.content_materialization_result
    assert method_result.budget_status == "accepted"
    assert method_result.integrity_status == "passed"
    assert method_result.attempt_count > 1
    assert method_result.attempt_count == len(
        result.content_materialization_attempts
    )
    assert method_result.materialization_scale < 1.0
    assert method_result.budget_utilization <= 1.0
    assert measurement.materialization_scale == (
        method_result.materialization_scale
    )
    assert measurement.scaled_nominal_delta_digest == (
        method_result.observation.scaled_nominal_delta_digest
    )
    assert measurement.materialization_replay_identity == (
        method_result.observation.materialization_replay_identity
    )
    selected_bits = int.from_bytes(
        pack(">f", method_result.materialization_scale),
        byteorder="big",
    )
    next_scale = unpack(
        ">f",
        (selected_bits + 1).to_bytes(4, byteorder="big"),
    )[0]
    next_attempt = [
        attempt
        for attempt in result.content_materialization_attempts
        if attempt.materialization_scale == next_scale
    ]
    assert len(next_attempt) == 1
    assert not content_actual_budget_accepts(
        method_result.observation.baseline_norm,
        next_attempt[0].realized_total_l2,
    )
    assert not hasattr(measurement, "budget_acceptance_status")
    assert not hasattr(measurement, "budget_accepted")
    assert not torch.equal(
        result.clean_generation_terminal_latent,
        result.watermarked_generation_terminal_latent,
    )

    assert len(backend.decode_inputs) == 2
    assert torch.equal(
        backend.decode_inputs[0],
        result.clean_generation_terminal_latent.to(torch.float32)
        / 0.5
        + 0.25,
    )
    assert torch.equal(
        backend.decode_inputs[1],
        result.watermarked_generation_terminal_latent.to(torch.float32)
        / 0.5
        + 0.25,
    )
    assert len(backend.posteriors) == 2
    assert all(posterior.mode_calls == 1 for posterior in backend.posteriors)
    assert all(posterior.sample_calls == 0 for posterior in backend.posteriors)
    assert torch.equal(
        result.clean_detection_latent,
        (result.clean_image / 4.0 - 0.25) * 0.5,
    )
    assert torch.equal(
        result.watermarked_detection_latent,
        (result.watermarked_image / 4.0 - 0.25) * 0.5,
    )
    assert backend.run_calls == 2
    assert backend.suffix_capture_calls == 0


@pytest.mark.unit
def test_content_write_captures_callback_suffix_without_changing_paired_result(
) -> None:
    ordinary_backend = FakeContentBackend()
    captured_backend = FakeContentBackend()
    ordinary_adapter = _initialized_adapter(ordinary_backend)
    captured_adapter = _initialized_adapter(captured_backend)
    base = _base_latent()

    ordinary = ordinary_adapter.execute_content_write_and_vae(
        base,
        _embedding_operation([]),
    )
    captured = captured_adapter.execute_content_write_and_capture_geometry_suffix(
        base,
        _embedding_operation([]),
    )

    paired = captured.content_write_result
    assert captured_backend.run_calls == 2
    assert captured_backend.suffix_capture_calls == 1
    assert len(captured_backend.suffix_callback_latents) == 1
    assert ordinary_backend.suffix_capture_calls == 0
    assert captured.suffix_context.runtime_config_digest == (
        captured_adapter.session.runtime_config_digest
    )
    assert captured.suffix_context.callback_index == 18
    assert paired.clean_callback_indices == tuple(range(20))
    assert paired.watermarked_callback_indices == tuple(range(20))
    assert torch.equal(
        ordinary.clean_generation_terminal_latent,
        paired.clean_generation_terminal_latent,
    )
    assert torch.equal(
        ordinary.watermarked_generation_terminal_latent,
        paired.watermarked_generation_terminal_latent,
    )
    assert torch.equal(ordinary.clean_image, paired.clean_image)
    assert torch.equal(ordinary.watermarked_image, paired.watermarked_image)
    assert torch.equal(
        ordinary.content_materialization.baseline_latent_actual,
        paired.content_materialization.baseline_latent_actual,
    )
    assert torch.equal(
        ordinary.content_materialization.written_latent_actual,
        paired.content_materialization.written_latent_actual,
    )
    assert torch.equal(
        captured_backend.suffix_callback_latents[0],
        paired.content_materialization.written_latent_actual,
    )
    assert ordinary.content_materialization_result == (
        paired.content_materialization_result
    )
    ordinary_adapter.close()
    captured_adapter.close()


@pytest.mark.unit
def test_full_scale_actual_write_can_be_accepted_without_retry() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    calls: list[tuple[float, ...]] = []
    shape = (1, 1, 1, 1)
    carrier = hf_carrier(TEST_ROOT_KEY, shape)
    baseline = torch.tensor(
        carrier.direction, dtype=torch.float32
    ).reshape(shape).to(torch.float16)

    def operation(values: tuple[float, ...]) -> ContentEmbeddingResult:
        calls.append(values)
        return content_embedder(values, carrier)

    result = adapter.execute_content_write_and_vae(
        baseline,
        operation,
    )

    method_result = result.content_materialization_result
    embedding = method_result.embedding_result
    assert len(calls) == 1
    assert embedding.target_relative_l2 == pytest.approx(0.012)
    assert sqrt(
        sum(value * value for value in embedding.delta_content)
    ) == pytest.approx(embedding.target_total_norm, rel=2e-5)
    assert method_result.materialization_scale == 1.0
    assert method_result.attempt_count == 1
    assert len(result.content_materialization_attempts) == 1
    assert result.content_materialization.integrity_status == "passed"
    assert content_actual_budget_accepts(
        method_result.observation.baseline_norm,
        method_result.realized_total_l2,
    )


@pytest.mark.unit
def test_materialization_replay_rejects_adjacent_float16_write() -> None:
    baseline = _base_latent()
    embedding = content_embedder(
        tuple(float(value) for value in baseline.to(torch.float32).reshape(-1)),
        hf_carrier(TEST_ROOT_KEY, TEST_SHAPE),
    )
    delta = torch.tensor(
        embedding.delta_content,
        dtype=torch.float32,
    ).reshape(TEST_SHAPE)
    written = (baseline.to(torch.float32) + delta).to(torch.float16)
    forged = written.detach().clone()
    forged.reshape(-1)[0] = torch.nextafter(
        forged.reshape(-1)[0],
        torch.tensor(float("inf"), dtype=torch.float16),
    )

    with pytest.raises(
        RuntimeError,
        match="deterministic binary16 replay",
    ):
        measure_content_materialization(
            embedding,
            baseline,
            forged,
            materialization_scale=1.0,
            attempt_index=1,
            callback_index=18,
            expected_callback_index=18,
            actual_dtype="float16",
        )


@pytest.mark.unit
def test_materialization_helper_rejects_wrong_actual_callback_index() -> None:
    baseline = _base_latent()
    embedding = content_embedder(
        tuple(float(value) for value in baseline.to(torch.float32).reshape(-1)),
        hf_carrier(TEST_ROOT_KEY, TEST_SHAPE),
    )
    written = (
        baseline.to(torch.float32)
        + torch.tensor(
            embedding.delta_content,
            dtype=torch.float32,
        ).reshape(TEST_SHAPE)
    ).to(torch.float16)

    with pytest.raises(
        RuntimeError,
        match="actual callback index does not match expected",
    ):
        measure_content_materialization(
            embedding,
            baseline,
            written,
            materialization_scale=1.0,
            attempt_index=1,
            callback_index=17,
            expected_callback_index=18,
            actual_dtype="float16",
        )


@pytest.mark.unit
def test_subnormal_scale_returns_write_disappeared_attempt() -> None:
    baseline = _base_latent()
    embedding = content_embedder(
        tuple(float(value) for value in baseline.to(torch.float32).reshape(-1)),
        hf_carrier(TEST_ROOT_KEY, TEST_SHAPE),
    )
    minimum_subnormal_scale = unpack(">f", b"\x00\x00\x00\x01")[0]
    measurement = measure_content_materialization(
        embedding,
        baseline,
        baseline.detach().clone(),
        materialization_scale=minimum_subnormal_scale,
        attempt_index=1,
        callback_index=18,
        expected_callback_index=18,
        actual_dtype="float16",
    )

    assert measurement.materialization_scale == minimum_subnormal_scale
    assert measurement.integrity_status == "write_disappeared"
    assert measurement.realized_total_l2 == 0.0
    assert measurement.realized_relative_l2 == 0.0
    assert measurement.scaled_nominal_delta_digest
    assert measurement.tensor_replay_identity
    assert measurement.materialization_replay_identity


@pytest.mark.unit
def test_runtime_module_does_not_own_budget_policy() -> None:
    assert not hasattr(runtime_content_write, "content_actual_budget_accepts")
    assert not hasattr(runtime_content_write, "CONTENT_RELATIVE_L2_NUMERATOR")
    assert not hasattr(runtime_content_write, "CONTENT_RELATIVE_L2_DENOMINATOR")
    assert not hasattr(runtime_content_write, "tau_actual_budget")


@pytest.mark.unit
@pytest.mark.parametrize(
    "sequences,error_match",
    [
        (
            (
                tuple(index for index in range(20) if index != 18),
                tuple(range(20)),
            ),
            "missing or out of order",
        ),
        (
            (
                tuple(range(19)) + (18, 19),
                tuple(range(20)),
            ),
            "duplicated",
        ),
        (
            (
                tuple(range(20)),
                tuple(range(19)) + (20,),
            ),
            "wrong callback index",
        ),
    ],
)
def test_callback_missing_duplicate_or_wrong_index_fails_closed(
    sequences,
    error_match,
) -> None:
    backend = FakeContentBackend(callback_sequences=sequences)
    adapter = _initialized_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            _embedding_operation([]),
        )

    assert error_match in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_paired_trajectory_divergence_before_write_fails_closed() -> None:
    backend = FakeContentBackend(diverge_second_run=True)
    adapter = _initialized_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            _embedding_operation([]),
        )

    assert "diverged before content write" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_embedding_from_different_baseline_norm_fails_closed() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    carrier = hf_carrier(TEST_ROOT_KEY, TEST_SHAPE)
    foreign_baseline = tuple(
        float(value) * 2.0
        for value in _base_latent().to(torch.float32).reshape(-1)
    )
    foreign_embedding = content_embedder(
        foreign_baseline,
        carrier,
    )

    def foreign_operation(
        values: tuple[float, ...],
    ) -> ContentEmbeddingResult:
        assert values != foreign_baseline
        return foreign_embedding

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            foreign_operation,
        )

    assert "latent norm does not match actual callback baseline" in str(
        exc_info.value.__cause__
    )
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_half_delta_method_result_never_reaches_runtime_acceptance() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    carrier = hf_carrier(TEST_ROOT_KEY, TEST_SHAPE)

    def half_delta_operation(
        values: tuple[float, ...],
    ) -> ContentEmbeddingResult:
        embedding = content_embedder(values, carrier)
        half_delta = tuple(
            unpack(">f", pack(">f", value * 0.5))[0]
            for value in embedding.delta_content
        )
        return replace(
            embedding,
            delta_content=half_delta,
            delta_content_digest=sha256(
                b"".join(pack(">f", value) for value in half_delta)
            ).hexdigest(),
        )

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            half_delta_operation,
        )

    assert "nominal formula replay" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_zero_relative_l2_baseline_fails_closed() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            torch.zeros(TEST_SHAPE, dtype=torch.float16),
            _embedding_operation([]),
        )

    assert "zero L2 energy" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_write_erased_during_frozen_suffix_fails_closed() -> None:
    backend = FakeContentBackend(erase_second_suffix=True)
    adapter = _initialized_adapter(backend)

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            _embedding_operation([]),
        )

    assert "disappeared after scheduler suffix" in str(
        exc_info.value.__cause__
    )
    assert adapter.state is RuntimeAdapterState.FAILED
    assert backend.close_calls == 1


@pytest.mark.unit
def test_nonfinite_delta_and_float16_overflow_fail_closed() -> None:
    carrier = hf_carrier(TEST_ROOT_KEY, TEST_SHAPE)

    def nonfinite_operation(
        values: tuple[float, ...],
    ) -> ContentEmbeddingResult:
        embedding = content_embedder(values, carrier)
        return replace(
            embedding,
            delta_content=(float("inf"),) + embedding.delta_content[1:],
        )

    nonfinite_backend = FakeContentBackend()
    nonfinite_adapter = _initialized_adapter(nonfinite_backend)
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        nonfinite_adapter.execute_content_write_and_vae(
            _base_latent(),
            nonfinite_operation,
        )
    assert "finite" in str(exc_info.value.__cause__)

    overflow_backend = FakeContentBackend()
    overflow_adapter = _initialized_adapter(overflow_backend)
    overflow_base = torch.full(
        TEST_SHAPE,
        65504.0,
        dtype=torch.float16,
    )
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        overflow_adapter.execute_content_write_and_vae(
            overflow_base,
            _embedding_operation([]),
        )
    assert "non-finite" in str(exc_info.value.__cause__)


@pytest.mark.unit
def test_invalid_baseline_dtype_and_posterior_boundary_fail_closed() -> None:
    dtype_backend = FakeContentBackend()
    dtype_adapter = _initialized_adapter(dtype_backend)
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        dtype_adapter.execute_content_write_and_vae(
            _base_latent().to(torch.float32),
            _embedding_operation([]),
        )
    assert "dtype drifted" in str(exc_info.value.__cause__)

    finite_backend = FakeContentBackend()
    finite_adapter = _initialized_adapter(finite_backend)
    nonfinite_base = _base_latent()
    nonfinite_base.reshape(-1)[0] = float("nan")
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        finite_adapter.execute_content_write_and_vae(
            nonfinite_base,
            _embedding_operation([]),
        )
    assert "non-finite" in str(exc_info.value.__cause__)

    posterior_backend = FakeContentBackend(invalid_posterior=True)
    posterior_adapter = _initialized_adapter(posterior_backend)
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        posterior_adapter.execute_content_write_and_vae(
            _base_latent(),
            _embedding_operation([]),
        )
    assert "posterior mode boundary" in str(exc_info.value.__cause__)

    nonfinite_mode_backend = FakeContentBackend(
        nonfinite_posterior_mode=True,
    )
    nonfinite_mode_adapter = _initialized_adapter(
        nonfinite_mode_backend
    )
    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        nonfinite_mode_adapter.execute_content_write_and_vae(
            _base_latent(),
            _embedding_operation([]),
        )
    assert "posterior_mode contains non-finite" in str(
        exc_info.value.__cause__
    )
    assert nonfinite_mode_adapter.state is RuntimeAdapterState.FAILED
    assert nonfinite_mode_backend.close_calls == 1
