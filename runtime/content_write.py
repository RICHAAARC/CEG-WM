"""Paired content materialization and VAE execution for runtime content_write_and_vae."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, prod, sqrt
from struct import pack, unpack
from typing import Callable, Literal, Sequence

import torch

from main import (
    ContentEmbeddingResult,
    ContentEmbedderError,
    ContentMaterializationObservation,
    ContentMaterializationResult,
    content_materialization_replay_identity,
    reconcile_content_materialization_budget,
    scale_content_delta_binary32,
)

from .adapter import RuntimeSession
from .backend import (
    RuntimeContentBackend,
    RuntimeVaeFactors,
    RuntimeVaePosterior,
)
from .configuration import Sd35RuntimeConfiguration


ContentEmbeddingOperation = Callable[
    [tuple[float, ...]],
    ContentEmbeddingResult,
]
MaterializationIntegrityStatus = Literal[
    "passed",
    "write_disappeared",
]


class RuntimeContentExecutionError(RuntimeError):
    """The paired callback/materialization/VAE path failed closed."""


@dataclass(frozen=True, slots=True)
class ContentMaterializationMeasurement:
    """一个 scale 的 runtime 物化证据，不拥有预算判定。"""

    attempt_index: int
    callback_index: int
    embedder_config_digest: str
    materialization_scale: float
    scaled_nominal_delta_digest: str
    baseline_latent_actual: torch.Tensor
    written_latent_actual: torch.Tensor
    delta_content_actual: torch.Tensor
    baseline_latent_digest: str
    written_latent_digest: str
    delta_content_actual_digest: str
    tensor_replay_identity: str
    materialization_replay_identity: str
    realized_total_l2: float
    realized_relative_l2: float
    integrity_status: MaterializationIntegrityStatus


@dataclass(frozen=True, slots=True)
class ContentMaterializationAttempt:
    """不保留中间 tensor 的逐次 materialization 审计身份。"""

    attempt_index: int
    materialization_scale: float
    scaled_nominal_delta_digest: str
    written_latent_digest: str
    delta_content_actual_digest: str
    tensor_replay_identity: str
    materialization_replay_identity: str
    realized_total_l2: float
    realized_relative_l2: float
    integrity_status: MaterializationIntegrityStatus


@dataclass(frozen=True, slots=True)
class ContentWriteVaeResult:
    """Actual paired tensors produced by the prepared runtime backend."""

    candidate_id: str
    runtime_config_digest: str
    selected_device: str
    paired_base_latent_digest: str
    clean_callback_indices: tuple[int, ...]
    watermarked_callback_indices: tuple[int, ...]
    content_materialization: ContentMaterializationMeasurement
    content_materialization_result: ContentMaterializationResult
    content_materialization_attempts: tuple[
        ContentMaterializationAttempt,
        ...,
    ]
    clean_generation_terminal_latent: torch.Tensor
    watermarked_generation_terminal_latent: torch.Tensor
    vae_scaling_factor_actual: float
    vae_shift_factor_actual: float
    clean_image: torch.Tensor
    watermarked_image: torch.Tensor
    clean_detection_latent: torch.Tensor
    watermarked_detection_latent: torch.Tensor


@dataclass(frozen=True, slots=True)
class CleanImageVaeObservationResult:
    """Single clean generation and public RGB-to-VAE observation."""

    candidate_id: str
    runtime_config_digest: str
    selected_device: str
    clean_base_latent_digest: str
    clean_callback_indices: tuple[int, ...]
    clean_generation_terminal_latent: torch.Tensor
    vae_scaling_factor_actual: float
    vae_shift_factor_actual: float
    clean_image: torch.Tensor
    clean_detection_latent: torch.Tensor


def _float32(value: float, role: str) -> float:
    if not isfinite(value):
        raise RuntimeContentExecutionError(f"{role} must be finite")
    try:
        converted = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise RuntimeContentExecutionError(
            f"{role} is outside binary32 range"
        ) from exc
    if not isfinite(converted):
        raise RuntimeContentExecutionError(f"{role} must remain finite")
    return converted


def _tensor(
    value: object,
    *,
    role: str,
    shape: torch.Size | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeContentExecutionError(f"{role} must be a tensor")
    if value.layout is not torch.strided or value.numel() == 0:
        raise RuntimeContentExecutionError(
            f"{role} must be a non-empty strided tensor"
        )
    if shape is not None and value.shape != shape:
        raise RuntimeContentExecutionError(f"{role} shape drifted")
    if dtype is not None and value.dtype is not dtype:
        raise RuntimeContentExecutionError(f"{role} dtype drifted")
    if device is not None and value.device != device:
        raise RuntimeContentExecutionError(f"{role} device drifted")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeContentExecutionError(f"{role} contains non-finite values")
    return value


def _float32_values(
    value: torch.Tensor,
    role: str,
) -> tuple[float, ...]:
    checked = _tensor(value, role=role)
    return tuple(
        float(item)
        for item in checked.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).reshape(-1)
    )


def _float32_vector(
    values: Sequence[float],
    expected_size: int,
    role: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RuntimeContentExecutionError(f"{role} must be a numeric sequence")
    if len(values) != expected_size:
        raise RuntimeContentExecutionError(
            f"{role} length does not match callback latent"
        )
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeContentExecutionError(
                f"{role} must contain only finite numbers"
            )
        converted.append(_float32(float(value), role))
    return tuple(converted)


def _l2_float32(values: Sequence[float], role: str) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(
            squared_sum + _float32(value * value, f"{role} squared term"),
            f"{role} squared accumulator",
        )
    return _float32(sqrt(squared_sum), f"{role} L2")


def _float32_digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().to(device="cpu")
    if contiguous.dtype is not torch.float16:
        raise RuntimeContentExecutionError(
            "stable tensor digest currently requires registered float16"
        )
    raw = b"".join(
        bits.to_bytes(2, byteorder="big", signed=False)
        for bits in _actual_float16_bits(contiguous)
    )
    identity = (
        str(contiguous.dtype).encode("ascii")
        + b"\x00"
        + repr(tuple(contiguous.shape)).encode("ascii")
        + b"\x00"
        + raw
    )
    return sha256(identity).hexdigest()


def _float16_bits(value: float) -> int:
    try:
        encoded = pack(">e", value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeContentExecutionError(
            "content write overflowed registered float16"
        ) from exc
    return int.from_bytes(encoded, byteorder="big", signed=False)


def _actual_float16_bits(value: torch.Tensor) -> tuple[int, ...]:
    raw = value.detach().contiguous().to(device="cpu").view(torch.int16)
    return tuple(int(item) & 0xFFFF for item in raw.reshape(-1).tolist())


def measure_content_materialization(
    embedding_result: ContentEmbeddingResult,
    baseline_latent_actual: torch.Tensor,
    written_latent_actual: torch.Tensor,
    *,
    materialization_scale: float,
    attempt_index: int,
    callback_index: int,
    expected_callback_index: int,
    actual_dtype: str,
) -> ContentMaterializationMeasurement:
    """Independently replay and measure one deterministic binary16 write."""

    if type(embedding_result) is not ContentEmbeddingResult:
        raise RuntimeContentExecutionError(
            "materialization requires ContentEmbeddingResult"
        )
    scale = _float32(materialization_scale, "materialization_scale")
    if not 0.0 < scale <= 1.0:
        raise RuntimeContentExecutionError(
            "materialization_scale must be in (0,1]"
        )
    if type(attempt_index) is not int or attempt_index <= 0:
        raise RuntimeContentExecutionError(
            "materialization attempt index must be positive"
        )
    if actual_dtype != "float16":
        raise RuntimeContentExecutionError(
            "actual dtype does not match the registered float16 candidate"
        )
    if (
        type(expected_callback_index) is not int
        or expected_callback_index != 18
    ):
        raise RuntimeContentExecutionError(
            "expected callback index does not match the registered candidate"
        )
    if (
        type(callback_index) is not int
        or callback_index != expected_callback_index
    ):
        raise RuntimeContentExecutionError(
            "actual callback index does not match expected callback index"
        )
    baseline = _tensor(
        baseline_latent_actual,
        role="baseline_latent_actual",
        dtype=torch.float16,
    )
    written = _tensor(
        written_latent_actual,
        role="written_latent_actual",
        shape=baseline.shape,
        dtype=torch.float16,
        device=baseline.device,
    )
    if tuple(baseline.shape) != embedding_result.shape:
        raise RuntimeContentExecutionError(
            "embedding result shape does not match callback latent"
        )
    expected_size = prod(embedding_result.shape)
    baseline_values = _float32_values(
        baseline,
        "baseline_latent_actual",
    )
    baseline_norm = _l2_float32(
        baseline_values,
        "baseline_latent_actual",
    )
    if baseline_norm == 0.0:
        raise RuntimeContentExecutionError(
            "actual callback baseline has zero L2 energy"
        )
    if embedding_result.latent_norm != baseline_norm:
        raise RuntimeContentExecutionError(
            "embedding latent norm does not match actual callback baseline"
        )
    nominal_delta_values = _float32_vector(
        embedding_result.delta_content,
        expected_size,
        "delta_content",
    )
    if (
        _float32_digest(nominal_delta_values)
        != embedding_result.delta_content_digest
    ):
        raise RuntimeContentExecutionError("delta_content digest mismatch")
    try:
        scaled_nominal_delta = scale_content_delta_binary32(
            embedding_result,
            scale,
        )
    except ContentEmbedderError as exc:
        raise RuntimeContentExecutionError(
            f"scaled nominal content delta is invalid: {exc}"
        ) from exc
    scaled_nominal_delta_digest = _float32_digest(
        scaled_nominal_delta
    )

    expected_bits = tuple(
        _float16_bits(_float32(base + scaled, "content write sum"))
        for base, scaled in zip(
            baseline_values,
            scaled_nominal_delta,
            strict=True,
        )
    )
    if _actual_float16_bits(written) != expected_bits:
        raise RuntimeContentExecutionError(
            "written latent failed deterministic binary16 replay"
        )

    delta_actual = (
        written.detach().to(dtype=torch.float32)
        - baseline.detach().to(dtype=torch.float32)
    )
    delta_actual_values = _float32_values(
        delta_actual,
        "delta_content_actual",
    )
    realized_total_l2 = _l2_float32(
        delta_actual_values,
        "delta_content_actual",
    )
    integrity_status: MaterializationIntegrityStatus = (
        "write_disappeared"
        if realized_total_l2 == 0.0
        else "passed"
    )
    realized_relative_l2 = (
        0.0
        if realized_total_l2 == 0.0
        else _float32(
            realized_total_l2 / baseline_norm,
            "realized_relative_l2",
        )
    )
    baseline_digest = _tensor_digest(baseline)
    written_digest = _tensor_digest(written)
    delta_digest = _float32_digest(delta_actual_values)
    tensor_replay_identity = sha256(
        (
            f"float16-rne-v2\0{attempt_index}\0{callback_index}\0"
            f"{embedding_result.embedder_config_digest}\0"
            f"{pack('>f', scale).hex()}\0"
            f"{scaled_nominal_delta_digest}\0"
            f"{baseline_digest}\0{written_digest}\0{delta_digest}\0"
            f"{pack('>f', realized_total_l2).hex()}\0"
            f"{pack('>f', realized_relative_l2).hex()}"
        ).encode("ascii")
    ).hexdigest()
    try:
        replay_identity = content_materialization_replay_identity(
            embedding_result,
            materialization_scale=scale,
            scaled_nominal_delta_digest=scaled_nominal_delta_digest,
            baseline_norm=baseline_norm,
            delta_content_actual=delta_actual_values,
            realized_total_l2=realized_total_l2,
            integrity_status=integrity_status,
        )
    except ContentEmbedderError as exc:
        raise RuntimeContentExecutionError(
            f"materialization replay identity failed: {exc}"
        ) from exc
    return ContentMaterializationMeasurement(
        attempt_index=attempt_index,
        callback_index=callback_index,
        embedder_config_digest=embedding_result.embedder_config_digest,
        materialization_scale=scale,
        scaled_nominal_delta_digest=scaled_nominal_delta_digest,
        baseline_latent_actual=baseline.detach().clone(),
        written_latent_actual=written.detach().clone(),
        delta_content_actual=delta_actual.detach().clone(),
        baseline_latent_digest=baseline_digest,
        written_latent_digest=written_digest,
        delta_content_actual_digest=delta_digest,
        tensor_replay_identity=tensor_replay_identity,
        materialization_replay_identity=replay_identity,
        realized_total_l2=realized_total_l2,
        realized_relative_l2=realized_relative_l2,
        integrity_status=integrity_status,
    )


def _method_observation(
    measurement: ContentMaterializationMeasurement,
) -> ContentMaterializationObservation:
    return ContentMaterializationObservation(
        materialization_scale=measurement.materialization_scale,
        baseline_norm=_l2_float32(
            _float32_values(
                measurement.baseline_latent_actual,
                "baseline_latent_actual",
            ),
            "baseline_latent_actual",
        ),
        scaled_nominal_delta_digest=(
            measurement.scaled_nominal_delta_digest
        ),
        delta_content_actual=_float32_values(
            measurement.delta_content_actual,
            "delta_content_actual",
        ),
        realized_total_l2=measurement.realized_total_l2,
        integrity_status=measurement.integrity_status,
        deterministic_binary16_replay_passed=True,
        materialization_replay_identity=(
            measurement.materialization_replay_identity
        ),
    )


def _attempt_record(
    measurement: ContentMaterializationMeasurement,
) -> ContentMaterializationAttempt:
    return ContentMaterializationAttempt(
        attempt_index=measurement.attempt_index,
        materialization_scale=measurement.materialization_scale,
        scaled_nominal_delta_digest=(
            measurement.scaled_nominal_delta_digest
        ),
        written_latent_digest=measurement.written_latent_digest,
        delta_content_actual_digest=(
            measurement.delta_content_actual_digest
        ),
        tensor_replay_identity=measurement.tensor_replay_identity,
        materialization_replay_identity=(
            measurement.materialization_replay_identity
        ),
        realized_total_l2=measurement.realized_total_l2,
        realized_relative_l2=measurement.realized_relative_l2,
        integrity_status=measurement.integrity_status,
    )


def _decode_generation_latent(
    backend: RuntimeContentBackend,
    latent: torch.Tensor,
    factors: RuntimeVaeFactors,
    role: str,
) -> torch.Tensor:
    decode_input = (
        latent.detach().to(dtype=torch.float32)
        / float(factors.scaling_factor)
        + float(factors.shift_factor)
    )
    return _tensor(
        backend.vae_decode(decode_input),
        role=f"{role}_image",
    ).detach().clone()


def _encode_detection_image(
    backend: RuntimeContentBackend,
    image: torch.Tensor,
    factors: RuntimeVaeFactors,
    role: str,
) -> torch.Tensor:
    posterior = backend.vae_encode(image.detach().clone())
    if not isinstance(posterior, RuntimeVaePosterior):
        raise RuntimeContentExecutionError(
            f"{role} VAE encode did not return a posterior mode boundary"
        )
    mode = _tensor(
        posterior.mode(),
        role=f"{role}_posterior_mode",
    )
    detection_latent = (
        mode.detach().to(dtype=torch.float32)
        - float(factors.shift_factor)
    ) * float(factors.scaling_factor)
    return _tensor(
        detection_latent,
        role=f"{role}_detection_latent",
    ).detach().clone()


def _validated_content_execution_latent(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(backend, RuntimeContentBackend):
        raise RuntimeContentExecutionError(
            "prepared backend lacks the content_write_and_vae execution protocol"
        )
    if type(configuration) is not Sd35RuntimeConfiguration:
        raise RuntimeContentExecutionError(
            "configuration must be Sd35RuntimeConfiguration"
        )
    if type(session) is not RuntimeSession:
        raise RuntimeContentExecutionError("runtime session is invalid")
    if (
        session.candidate_id != configuration.candidate_id
        or session.runtime_config_digest != configuration.runtime_config_digest
        or session.latent_dtype != configuration.latent_dtype
        or session.callback_index != configuration.callback_index
        or session.vae_decode_protocol != configuration.vae_decode_protocol
        or session.vae_encode_protocol != configuration.vae_encode_protocol
    ):
        raise RuntimeContentExecutionError(
            "runtime session does not match the frozen execution identity"
        )
    latent = _tensor(
        base_latent,
        role="base_latent",
        dtype=torch.float16,
    )
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise RuntimeContentExecutionError(
            "base_latent must have shape [1,C,H,W]"
        )
    if str(latent.device) != session.selected_device:
        raise RuntimeContentExecutionError(
            "base_latent device does not match the prepared session"
        )
    if _l2_float32(
        _float32_values(latent, "base_latent"),
        "base_latent",
    ) == 0.0:
        raise RuntimeContentExecutionError(
            "base_latent has zero L2 energy"
        )
    return latent


def _execute_clean_generation(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    initial_latent: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], dict[int, torch.Tensor]]:
    expected_indices = tuple(range(configuration.inference_steps))
    clean_indices: list[int] = []
    clean_trace: dict[int, torch.Tensor] = {}

    def clean_callback(index: int, callback_latent: torch.Tensor) -> torch.Tensor:
        if type(index) is not int or not 0 <= index < configuration.inference_steps:
            raise RuntimeContentExecutionError(
                "clean generation reported a wrong callback index"
            )
        if index in clean_trace:
            raise RuntimeContentExecutionError(
                "clean generation duplicated a callback index"
            )
        current = _tensor(
            callback_latent,
            role="clean_callback_latent",
            shape=initial_latent.shape,
            dtype=torch.float16,
            device=initial_latent.device,
        )
        clean_indices.append(index)
        clean_trace[index] = current.detach().clone()
        return current

    try:
        clean_terminal = _tensor(
            backend.run_generation(initial_latent, clean_callback),
            role="clean_generation_terminal_latent",
            shape=initial_latent.shape,
            dtype=torch.float16,
            device=initial_latent.device,
        ).detach().clone()
    except RuntimeContentExecutionError:
        raise
    except Exception as exc:
        raise RuntimeContentExecutionError(
            "clean generation backend failed"
        ) from exc
    if tuple(clean_indices) != expected_indices:
        raise RuntimeContentExecutionError(
            "clean generation callback sequence is missing or out of order"
        )
    return clean_terminal, tuple(clean_indices), clean_trace


def execute_clean_image_and_vae_observation(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
) -> CleanImageVaeObservationResult:
    """Generate one clean image and return its public posterior-mode observation."""

    latent = _validated_content_execution_latent(
        backend,
        configuration,
        session,
        base_latent,
    )
    clean_initial = latent.detach().clone()
    if _tensor_digest(clean_initial) != _tensor_digest(latent):
        raise RuntimeContentExecutionError(
            "clean generation latent does not match base latent"
        )
    clean_terminal, clean_indices, _clean_trace = _execute_clean_generation(
        backend,
        configuration,
        clean_initial,
    )
    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeContentExecutionError(
            "backend VAE factors do not match the execution protocol"
        )
    clean_image = _decode_generation_latent(
        backend,
        clean_terminal,
        factors,
        "clean",
    )
    clean_detection_latent = _encode_detection_image(
        backend,
        clean_image,
        factors,
        "clean",
    )
    return CleanImageVaeObservationResult(
        candidate_id=session.candidate_id,
        runtime_config_digest=session.runtime_config_digest,
        selected_device=session.selected_device,
        clean_base_latent_digest=_tensor_digest(latent),
        clean_callback_indices=clean_indices,
        clean_generation_terminal_latent=clean_terminal,
        vae_scaling_factor_actual=float(factors.scaling_factor),
        vae_shift_factor_actual=float(factors.shift_factor),
        clean_image=clean_image,
        clean_detection_latent=clean_detection_latent,
    )


def execute_content_write_and_vae(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
    content_embedding_operation: ContentEmbeddingOperation,
) -> ContentWriteVaeResult:
    """让 main 驱动 actual-dtype 预算闭环并执行一对生成。"""

    latent = _validated_content_execution_latent(
        backend,
        configuration,
        session,
        base_latent,
    )
    if not callable(content_embedding_operation):
        raise RuntimeContentExecutionError(
            "content_embedding_operation must be callable"
        )

    clean_initial = latent.detach().clone()
    watermarked_initial = latent.detach().clone()
    if clean_initial.data_ptr() == watermarked_initial.data_ptr():
        raise RuntimeContentExecutionError(
            "paired generation latents unexpectedly share storage"
        )
    base_digest = _tensor_digest(latent)
    if (
        _tensor_digest(clean_initial) != base_digest
        or _tensor_digest(watermarked_initial) != base_digest
    ):
        raise RuntimeContentExecutionError(
            "clean/watermarked paths do not share one base latent"
        )

    clean_terminal, clean_indices, clean_trace = _execute_clean_generation(
        backend,
        configuration,
        clean_initial,
    )

    expected_indices = tuple(range(configuration.inference_steps))

    watermarked_indices: list[int] = []
    watermarked_seen: set[int] = set()
    materialization: ContentMaterializationMeasurement | None = None
    materialization_result: ContentMaterializationResult | None = None
    materialization_attempts: list[ContentMaterializationMeasurement] = []
    target_index = configuration.callback_index

    def watermarked_callback(
        index: int,
        callback_latent: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal materialization, materialization_result
        if type(index) is not int or not 0 <= index < configuration.inference_steps:
            raise RuntimeContentExecutionError(
                "watermarked generation reported a wrong callback index"
            )
        if index in watermarked_seen:
            raise RuntimeContentExecutionError(
                "watermarked generation duplicated a callback index"
            )
        current = _tensor(
            callback_latent,
            role="watermarked_callback_latent",
            shape=latent.shape,
            dtype=torch.float16,
            device=latent.device,
        )
        watermarked_indices.append(index)
        watermarked_seen.add(index)
        if index <= target_index and not torch.equal(
            current.detach(),
            clean_trace[index],
        ):
            raise RuntimeContentExecutionError(
                "paired callback trajectories diverged before content write"
            )
        if index != target_index:
            return current
        if materialization is not None:
            raise RuntimeContentExecutionError(
                "content callback index was triggered more than once"
            )
        embedding_result = content_embedding_operation(
            _float32_values(current, "watermarked_callback_latent")
        )
        if type(embedding_result) is not ContentEmbeddingResult:
            raise RuntimeContentExecutionError(
                "content embedding operation returned an invalid result"
            )
        baseline = current.detach().clone()

        def materializer(
            requested_embedding: ContentEmbeddingResult,
            requested_scale: float,
            /,
        ) -> ContentMaterializationObservation:
            if requested_embedding is not embedding_result:
                raise ContentEmbedderError(
                    "runtime materializer received a different embedding result"
                )
            try:
                scaled_nominal_delta = scale_content_delta_binary32(
                    embedding_result,
                    requested_scale,
                )
                scaled_values = _float32_vector(
                    scaled_nominal_delta,
                    current.numel(),
                    "scaled_nominal_delta",
                )
                written = (
                    baseline.detach().to(dtype=torch.float32)
                    + torch.tensor(
                        scaled_values,
                        dtype=torch.float32,
                        device=baseline.device,
                    ).reshape(baseline.shape)
                ).to(dtype=torch.float16)
                measurement = measure_content_materialization(
                    embedding_result,
                    baseline,
                    written,
                    materialization_scale=requested_scale,
                    attempt_index=len(materialization_attempts) + 1,
                    callback_index=index,
                    expected_callback_index=configuration.callback_index,
                    actual_dtype=configuration.latent_dtype,
                )
            except RuntimeContentExecutionError as exc:
                raise ContentEmbedderError(str(exc)) from exc
            materialization_attempts.append(measurement)
            return _method_observation(measurement)

        try:
            materialization_result = (
                reconcile_content_materialization_budget(
                    embedding_result,
                    materializer,
                )
            )
        except ContentEmbedderError as exc:
            raise RuntimeContentExecutionError(
                f"content actual-dtype reconciliation failed: {exc}"
            ) from exc
        if (
            materialization_result.attempt_count
            != len(materialization_attempts)
        ):
            raise RuntimeContentExecutionError(
                "content materialization attempt count drifted"
            )
        selected_identity = (
            materialization_result.observation.materialization_replay_identity
        )
        selected = [
            attempt
            for attempt in materialization_attempts
            if attempt.materialization_replay_identity == selected_identity
        ]
        if len(selected) != 1:
            raise RuntimeContentExecutionError(
                "accepted materialization attempt identity is not unique"
            )
        materialization = selected[0]
        if materialization.integrity_status != "passed":
            raise RuntimeContentExecutionError(
                "accepted materialization did not pass runtime integrity"
            )
        return materialization.written_latent_actual.detach().clone()

    try:
        watermarked_terminal = _tensor(
            backend.run_generation(
                watermarked_initial,
                watermarked_callback,
            ),
            role="watermarked_generation_terminal_latent",
            shape=latent.shape,
            dtype=torch.float16,
            device=latent.device,
        ).detach().clone()
    except RuntimeContentExecutionError:
        raise
    except Exception as exc:
        raise RuntimeContentExecutionError(
            "watermarked generation backend failed"
        ) from exc
    if tuple(watermarked_indices) != expected_indices:
        raise RuntimeContentExecutionError(
            "watermarked callback sequence is missing or out of order"
        )
    if materialization is None:
        raise RuntimeContentExecutionError(
            "content callback index was not triggered exactly once"
        )
    if materialization_result is None:
        raise RuntimeContentExecutionError(
            "content materialization result is missing"
        )
    if torch.equal(clean_terminal, watermarked_terminal):
        raise RuntimeContentExecutionError(
            "actual-dtype content write disappeared after scheduler suffix"
        )

    factors = backend.vae_factors()
    if type(factors) is not RuntimeVaeFactors:
        raise RuntimeContentExecutionError(
            "backend VAE factors do not match the execution protocol"
        )
    clean_image = _decode_generation_latent(
        backend,
        clean_terminal,
        factors,
        "clean",
    )
    watermarked_image = _decode_generation_latent(
        backend,
        watermarked_terminal,
        factors,
        "watermarked",
    )
    clean_detection_latent = _encode_detection_image(
        backend,
        clean_image,
        factors,
        "clean",
    )
    watermarked_detection_latent = _encode_detection_image(
        backend,
        watermarked_image,
        factors,
        "watermarked",
    )
    return ContentWriteVaeResult(
        candidate_id=session.candidate_id,
        runtime_config_digest=session.runtime_config_digest,
        selected_device=session.selected_device,
        paired_base_latent_digest=base_digest,
        clean_callback_indices=tuple(clean_indices),
        watermarked_callback_indices=tuple(watermarked_indices),
        content_materialization=materialization,
        content_materialization_result=materialization_result,
        content_materialization_attempts=tuple(
            _attempt_record(attempt)
            for attempt in materialization_attempts
        ),
        clean_generation_terminal_latent=clean_terminal,
        watermarked_generation_terminal_latent=watermarked_terminal,
        vae_scaling_factor_actual=float(factors.scaling_factor),
        vae_shift_factor_actual=float(factors.shift_factor),
        clean_image=clean_image,
        watermarked_image=watermarked_image,
        clean_detection_latent=clean_detection_latent,
        watermarked_detection_latent=watermarked_detection_latent,
    )
