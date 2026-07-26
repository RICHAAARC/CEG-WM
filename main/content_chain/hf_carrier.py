"""CEG-WM `hf_sparse_tail` 高频载体模板与单位方向。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, prod, sqrt
from struct import pack, unpack
from typing import Sequence

from main.shared.key_schedule import (
    DEFAULT_CONFIG as DEFAULT_KEY_SCHEDULE_CONFIG,
    DerivedWrongKeyMaterial,
    KeyScheduleError,
    derive_wrong_key_stream,
    identify_root_key,
    key_schedule_sha256_counter,
    stable_json_utf8,
)

HF_CANDIDATE_ID = "hf_sparse_tail"
RUNTIME_CANDIDATE_ID = "runtime_sd35_flowmatch"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
TAIL_NUMERATOR = 1
TAIL_DENOMINATOR = 5
AVERAGE_KERNEL_SIZE = 5


class HfCarrierError(ValueError):
    """HF 模板、mask 或方向不符合冻结候选。"""


@dataclass(frozen=True, slots=True)
class HfCarrierResult:
    """HF 载体的不可变模板、方向和身份结果。"""

    candidate_id: str
    shape: tuple[int, int, int, int]
    template: tuple[float, ...]
    direction: tuple[float, ...]
    support_indices: tuple[int, ...]
    template_digest: str
    direction_digest: str
    mask_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    key_domain_digest: str
    carrier_config_digest: str


def _float32(value: float) -> float:
    if not isfinite(value):
        raise HfCarrierError("HF numeric value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise HfCarrierError("HF numeric value is outside binary32 range") from exc
    if not isfinite(rounded):
        raise HfCarrierError("HF binary32 value must be finite")
    return rounded


def _float32_vector(values: Sequence[float], expected_size: int, role: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HfCarrierError(f"{role} must be a numeric sequence")
    if len(values) != expected_size:
        raise HfCarrierError(f"{role} length does not match carrier shape")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HfCarrierError(f"{role} must contain only finite numbers")
        converted.append(_float32(float(value)))
    return tuple(converted)


def _l2_norm_float32(values: Sequence[float]) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    return _float32(sqrt(squared_sum))


def _normalize_float32(values: Sequence[float], role: str) -> tuple[float, ...]:
    norm = _l2_norm_float32(values)
    if norm == 0.0:
        raise HfCarrierError(f"{role} has zero L2 energy")
    return tuple(_float32(value / norm) for value in values)


def _float32_digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise HfCarrierError("HF shape must be [1,C,H,W]")
    normalized = tuple(shape)
    if (
        len(normalized) != 4
        or normalized[0] != 1
        or any(type(size) is not int or size <= 0 for size in normalized)
    ):
        raise HfCarrierError("HF shape must be positive [1,C,H,W]")
    return normalized


def _zero_padded_average_5x5(
    values: Sequence[float],
    shape: tuple[int, int, int, int],
) -> tuple[float, ...]:
    _, channels, height, width = shape
    pooled = [0.0] * len(values)
    for channel in range(channels):
        channel_offset = channel * height * width
        for row in range(height):
            for column in range(width):
                window_sum = 0.0
                for row_offset in range(-2, 3):
                    source_row = row + row_offset
                    for column_offset in range(-2, 3):
                        source_column = column + column_offset
                        if 0 <= source_row < height and 0 <= source_column < width:
                            source_index = (
                                channel_offset + source_row * width + source_column
                            )
                            window_sum = _float32(window_sum + values[source_index])
                        else:
                            window_sum = _float32(window_sum + 0.0)
                output_index = channel_offset + row * width + column
                pooled[output_index] = _float32(window_sum / 25.0)
    return tuple(pooled)


def _sparse_tail(
    residual: Sequence[float],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    support_size = (len(residual) * TAIL_NUMERATOR + TAIL_DENOMINATOR - 1) // (
        TAIL_DENOMINATOR
    )
    ranked_indices = sorted(
        range(len(residual)),
        key=lambda index: (-abs(residual[index]), index),
    )
    support_indices = tuple(sorted(ranked_indices[:support_size]))
    support = set(support_indices)
    sparse = tuple(
        residual[index] if index in support else 0.0
        for index in range(len(residual))
    )
    return sparse, support_indices


def _derive_hf_gaussian(
    detection_key: str | DerivedWrongKeyMaterial,
    shape: tuple[int, int, int, int],
    model_revision: str,
):
    domain_fields = {
        "candidate_id": HF_CANDIDATE_ID,
        "operator": "carrier_template",
        "responsibility_domain": "hf_carrier",
        "model_revision": model_revision,
        "tensor_role": "base_gaussian",
    }
    try:
        if type(detection_key) is str:
            stream = key_schedule_sha256_counter(
                detection_key,
                domain_fields,
                shape,
                distribution="gaussian",
            )
            root_digest = identify_root_key(detection_key).root_key_public_digest
            key_role = "registered"
            wrong_key_index = None
        elif type(detection_key) is DerivedWrongKeyMaterial:
            stream = derive_wrong_key_stream(
                detection_key,
                domain_fields,
                shape,
                distribution="gaussian",
            )
            root_digest = detection_key.registered_root_key_public_digest
            key_role = "wrong"
            wrong_key_index = detection_key.wrong_key_index
        else:
            raise HfCarrierError(
                "detection_key must be root text or DerivedWrongKeyMaterial"
            )
    except KeyScheduleError as exc:
        raise HfCarrierError("HF key stream derivation failed") from exc
    return stream, root_digest, key_role, wrong_key_index


def hf_carrier(
    detection_key: str | DerivedWrongKeyMaterial,
    shape: Sequence[int],
    *,
    mask_hf: Sequence[float] | None = None,
    model_revision: str = MODEL_REVISION,
) -> HfCarrierResult:
    """构造未中心化 sparse-tail 模板及 mask 后单位 HF 写入方向。"""

    normalized_shape = _validate_shape(shape)
    if model_revision != MODEL_REVISION:
        raise HfCarrierError("HF model revision does not match the frozen candidate")
    element_count = prod(normalized_shape)
    if mask_hf is None:
        mask = (1.0,) * element_count
    else:
        mask = _float32_vector(mask_hf, element_count, "mask_hf")
        if any(value < 0.0 or value > 1.0 for value in mask):
            raise HfCarrierError("mask_hf values must be in [0,1]")

    gaussian_stream, root_digest, key_role, wrong_key_index = _derive_hf_gaussian(
        detection_key,
        normalized_shape,
        model_revision,
    )
    gaussian = gaussian_stream.values
    pooled = _zero_padded_average_5x5(gaussian, normalized_shape)
    residual = tuple(
        _float32(value - low_pass)
        for value, low_pass in zip(gaussian, pooled, strict=True)
    )
    sparse, support_indices = _sparse_tail(residual)
    template = _normalize_float32(sparse, "HF sparse template")
    masked_template = tuple(
        _float32(value * weight)
        for value, weight in zip(template, mask, strict=True)
    )
    if all(weight == 1.0 for weight in mask):
        direction = template
    else:
        direction = _normalize_float32(masked_template, "HF masked direction")

    template_digest = _float32_digest(template)
    direction_digest = _float32_digest(direction)
    mask_digest = _float32_digest(mask)
    carrier_identity = {
        "average_kernel_size": AVERAGE_KERNEL_SIZE,
        "candidate_id": HF_CANDIDATE_ID,
        "count_include_pad": True,
        "dtype": "float32",
        "key_schedule_config_digest": DEFAULT_KEY_SCHEDULE_CONFIG.config_digest,
        "mask_digest": mask_digest,
        "model_revision": model_revision,
        "runtime_candidate_id": RUNTIME_CANDIDATE_ID,
        "shape": list(normalized_shape),
        "tail_denominator": TAIL_DENOMINATOR,
        "tail_numerator": TAIL_NUMERATOR,
        "template_time_centering": False,
        "tie_break": "negative_absolute_then_flat_index",
    }
    carrier_config_digest = sha256(stable_json_utf8(carrier_identity)).hexdigest()
    return HfCarrierResult(
        candidate_id=HF_CANDIDATE_ID,
        shape=normalized_shape,
        template=template,
        direction=direction,
        support_indices=support_indices,
        template_digest=template_digest,
        direction_digest=direction_digest,
        mask_digest=mask_digest,
        root_key_public_digest=root_digest,
        key_role=key_role,
        wrong_key_index=wrong_key_index,
        key_domain_digest=gaussian_stream.domain_digest,
        carrier_config_digest=carrier_config_digest,
    )
