"""Paired content materialization and VAE execution for runtime Batch 2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, prod, sqrt
from struct import pack, unpack
from typing import Callable, Literal, Sequence

import torch

from main import ContentEmbeddingResult

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
BudgetAcceptanceStatus = Literal["not_evaluated"]


class RuntimeContentExecutionError(RuntimeError):
    """The paired callback/materialization/VAE path failed closed."""


@dataclass(frozen=True, slots=True)
class ContentMaterializationMeasurement:
    """Runtime measurements only; no content-budget acceptance decision."""

    callback_index: int
    embedder_config_digest: str
    baseline_latent_actual: torch.Tensor
    written_latent_actual: torch.Tensor
    delta_content_actual: torch.Tensor
    baseline_latent_digest: str
    written_latent_digest: str
    delta_content_actual_digest: str
    materialization_replay_identity: str
    realized_total_l2: float
    realized_relative_l2: float
    budget_acceptance_status: BudgetAcceptanceStatus


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
    clean_generation_terminal_latent: torch.Tensor
    watermarked_generation_terminal_latent: torch.Tensor
    vae_scaling_factor_actual: float
    vae_shift_factor_actual: float
    clean_image: torch.Tensor
    watermarked_image: torch.Tensor
    clean_detection_latent: torch.Tensor
    watermarked_detection_latent: torch.Tensor


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
    callback_index: int,
    expected_callback_index: int,
    actual_dtype: str,
) -> ContentMaterializationMeasurement:
    """Independently replay and measure one deterministic binary16 write."""

    if type(embedding_result) is not ContentEmbeddingResult:
        raise RuntimeContentExecutionError(
            "materialization requires ContentEmbeddingResult"
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
    delta_values = _float32_vector(
        embedding_result.delta_content,
        expected_size,
        "delta_content",
    )
    if _float32_digest(delta_values) != embedding_result.delta_content_digest:
        raise RuntimeContentExecutionError("delta_content digest mismatch")

    expected_bits = tuple(
        _float16_bits(_float32(base + delta, "content write sum"))
        for base, delta in zip(
            baseline_values,
            delta_values,
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
    if realized_total_l2 == 0.0:
        raise RuntimeContentExecutionError(
            "actual-dtype content write disappeared"
        )
    realized_relative_l2 = _float32(
        realized_total_l2 / baseline_norm,
        "realized_relative_l2",
    )
    baseline_digest = _tensor_digest(baseline)
    written_digest = _tensor_digest(written)
    delta_digest = _float32_digest(delta_actual_values)
    replay_identity = sha256(
        (
            f"float16-rne-v1\0{callback_index}\0"
            f"{embedding_result.embedder_config_digest}\0"
            f"{baseline_digest}\0{written_digest}\0{delta_digest}\0"
            f"{pack('>f', realized_total_l2).hex()}\0"
            f"{pack('>f', realized_relative_l2).hex()}"
        ).encode("ascii")
    ).hexdigest()
    return ContentMaterializationMeasurement(
        callback_index=callback_index,
        embedder_config_digest=embedding_result.embedder_config_digest,
        baseline_latent_actual=baseline.detach().clone(),
        written_latent_actual=written.detach().clone(),
        delta_content_actual=delta_actual.detach().clone(),
        baseline_latent_digest=baseline_digest,
        written_latent_digest=written_digest,
        delta_content_actual_digest=delta_digest,
        materialization_replay_identity=replay_identity,
        realized_total_l2=realized_total_l2,
        realized_relative_l2=realized_relative_l2,
        budget_acceptance_status="not_evaluated",
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


def execute_content_write_and_vae(
    backend: RuntimeContentBackend,
    configuration: Sd35RuntimeConfiguration,
    session: RuntimeSession,
    base_latent: torch.Tensor,
    content_embedding_operation: ContentEmbeddingOperation,
) -> ContentWriteVaeResult:
    """Run one clean/watermarked pair without making a budget decision."""

    if not isinstance(backend, RuntimeContentBackend):
        raise RuntimeContentExecutionError(
            "prepared backend lacks the Batch-2 execution protocol"
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
    if not callable(content_embedding_operation):
        raise RuntimeContentExecutionError(
            "content_embedding_operation must be callable"
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
            shape=latent.shape,
            dtype=torch.float16,
            device=latent.device,
        )
        clean_indices.append(index)
        clean_trace[index] = current.detach().clone()
        return current

    try:
        clean_terminal = _tensor(
            backend.run_generation(clean_initial, clean_callback),
            role="clean_generation_terminal_latent",
            shape=latent.shape,
            dtype=torch.float16,
            device=latent.device,
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

    watermarked_indices: list[int] = []
    watermarked_seen: set[int] = set()
    materialization: ContentMaterializationMeasurement | None = None
    target_index = configuration.callback_index

    def watermarked_callback(
        index: int,
        callback_latent: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal materialization
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
        delta_values = _float32_vector(
            embedding_result.delta_content,
            current.numel(),
            "delta_content",
        )
        written = (
            current.detach().to(dtype=torch.float32)
            + torch.tensor(
                delta_values,
                dtype=torch.float32,
                device=current.device,
            ).reshape(current.shape)
        ).to(dtype=torch.float16)
        materialization = measure_content_materialization(
            embedding_result,
            current,
            written,
            callback_index=index,
            expected_callback_index=configuration.callback_index,
            actual_dtype=configuration.latent_dtype,
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
        clean_generation_terminal_latent=clean_terminal,
        watermarked_generation_terminal_latent=watermarked_terminal,
        vae_scaling_factor_actual=float(factors.scaling_factor),
        vae_shift_factor_actual=float(factors.shift_factor),
        clean_image=clean_image,
        watermarked_image=watermarked_image,
        clean_detection_latent=clean_detection_latent,
        watermarked_detection_latent=watermarked_detection_latent,
    )
