from dataclasses import replace
from hashlib import sha256
from struct import pack

import pytest
import torch

from main import ContentEmbeddingResult, content_embedder
from main.content_chain.hf_carrier import hf_carrier
from runtime import (
    RuntimeAdapterError,
    RuntimeAdapterState,
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeVaeFactors,
    create_runtime_adapter,
    measure_content_materialization,
)


TEST_ROOT_KEY = "ceg-wm-runtime-batch-two-root"
TEST_SHAPE = (1, 1, 4, 4)


def _identity(configuration, selected_device: str) -> RuntimeBackendIdentity:
    return RuntimeBackendIdentity(
        candidate_id=configuration.candidate_id,
        runtime_config_digest=configuration.runtime_config_digest,
        runtime_backend_name="cpu-fake-content-backend",
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


class FakeContentBackend:
    def __init__(
        self,
        callback_sequences: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        *,
        diverge_second_run: bool = False,
        erase_second_suffix: bool = False,
        invalid_posterior: bool = False,
        nonfinite_posterior_mode: bool = False,
    ) -> None:
        default = tuple(range(20))
        self.callback_sequences = callback_sequences or (default, default)
        self.diverge_second_run = diverge_second_run
        self.erase_second_suffix = erase_second_suffix
        self.invalid_posterior = invalid_posterior
        self.nonfinite_posterior_mode = nonfinite_posterior_mode
        self.run_calls = 0
        self.close_calls = 0
        self.decode_inputs: list[torch.Tensor] = []
        self.posteriors: list[FakePosterior] = []

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device: str):
        return _identity(configuration, selected_device)

    def close(self) -> None:
        self.close_calls += 1

    def run_generation(self, initial_latent, callback):
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

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(
            scaling_factor=0.5,
            shift_factor=0.25,
        )

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        return latent.detach().clone() * 2.0

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


@pytest.mark.unit
def test_paired_write_returns_actual_measurements_without_budget_decision() -> None:
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
    assert measurement.budget_acceptance_status == "not_evaluated"
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
            callback_index=17,
            expected_callback_index=18,
            actual_dtype="float16",
        )


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
def test_actual_dtype_write_disappearance_fails_closed() -> None:
    backend = FakeContentBackend()
    adapter = _initialized_adapter(backend)
    carrier = hf_carrier(TEST_ROOT_KEY, TEST_SHAPE)
    zeros = (0.0,) * 16

    def zero_operation(values: tuple[float, ...]) -> ContentEmbeddingResult:
        return replace(
            content_embedder(values, carrier),
            delta_content=zeros,
            delta_content_digest=sha256(
                b"".join(pack(">f", value) for value in zeros)
            ).hexdigest(),
        )

    with pytest.raises(RuntimeAdapterError, match="failed closed") as exc_info:
        adapter.execute_content_write_and_vae(
            _base_latent(),
            zero_operation,
        )

    assert "write disappeared" in str(exc_info.value.__cause__)
    assert adapter.state is RuntimeAdapterState.FAILED


@pytest.mark.unit
def test_zero_l2_baseline_fails_closed() -> None:
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
