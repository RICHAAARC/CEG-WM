"""CEG-WM LF/HF 内容方向组合与共同 target total budget。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from struct import pack, unpack
from typing import Literal, Sequence

from main.shared.key_schedule import stable_json_utf8

from .hf_carrier import HfCarrierResult
from .lf_carrier import LfCarrierResult

CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
FROZEN_MIXING_COEFFICIENTS = (0.25, 0.50, 0.75)
EMBEDDER_CANDIDATE_IDS = (
    "runtime_sd35_flowmatch",
    "hf_sparse_tail",
    "lf_low_pass",
    "routing_stqr",
    "routing_uniform_control",
)
EmbeddingMode = Literal["hf_only", "lf_only", "combined"]


class ContentEmbedderError(ValueError):
    """内容方向、混合系数或共同 target total budget 无效。"""


@dataclass(frozen=True, slots=True)
class ContentEmbeddingResult:
    """未物化 actual dtype 的理论内容更新与可重建分支诊断。"""

    candidate_ids: tuple[str, ...]
    mode: EmbeddingMode
    shape: tuple[int, int, int, int]
    delta_content: tuple[float, ...]
    delta_content_digest: str
    latent_norm: float
    target_total_norm: float
    target_relative_l2: float
    mixing_coefficient: float | None
    gamma_lh: float | None
    combined_pre_normalization_norm: float
    target_component_lf: tuple[float, ...] | None
    target_component_hf: tuple[float, ...] | None
    target_component_lf_norm: float | None
    target_component_hf_norm: float | None
    lf_carrier_config_digest: str | None
    hf_carrier_config_digest: str | None
    embedder_config_digest: str


def _float32(value: float) -> float:
    if not isfinite(value):
        raise ContentEmbedderError("content embedder value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise ContentEmbedderError(
            "content embedder value is outside binary32 range"
        ) from exc
    if not isfinite(rounded):
        raise ContentEmbedderError(
            "content embedder binary32 value must be finite"
        )
    return rounded


def _vector(
    values: Sequence[float],
    size: int,
    role: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContentEmbedderError(f"{role} must be a numeric sequence")
    if len(values) != size:
        raise ContentEmbedderError(f"{role} length does not match carrier shape")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContentEmbedderError(f"{role} must contain only finite numbers")
        converted.append(_float32(float(value)))
    return tuple(converted)


def _l2_norm(values: Sequence[float]) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    return _float32(sqrt(squared_sum))


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _validated_hf_direction(
    carrier: object,
) -> tuple[HfCarrierResult, tuple[float, ...]]:
    if type(carrier) is not HfCarrierResult:
        raise ContentEmbedderError("HF branch requires HfCarrierResult")
    direction = _vector(
        carrier.direction,
        len(carrier.direction),
        "hf_direction",
    )
    if _digest(direction) != carrier.direction_digest:
        raise ContentEmbedderError("HF carrier direction digest mismatch")
    norm = _l2_norm(direction)
    if norm == 0.0:
        raise ContentEmbedderError("active HF direction has zero L2 energy")
    if abs(norm - 1.0) > 1e-5:
        raise ContentEmbedderError("active HF direction must be unit L2")
    return carrier, direction


def _validated_lf_direction(
    carrier: object,
) -> tuple[LfCarrierResult, tuple[float, ...]]:
    if type(carrier) is not LfCarrierResult:
        raise ContentEmbedderError("LF branch requires LfCarrierResult")
    direction = _vector(
        carrier.direction,
        len(carrier.direction),
        "lf_direction",
    )
    if _digest(direction) != carrier.direction_digest:
        raise ContentEmbedderError("LF carrier direction digest mismatch")
    norm = _l2_norm(direction)
    if norm == 0.0:
        raise ContentEmbedderError("active LF direction has zero L2 energy")
    if abs(norm - 1.0) > 1e-5:
        raise ContentEmbedderError("active LF direction must be unit L2")
    return carrier, direction


def _target_component(
    scale: float,
    direction: Sequence[float],
) -> tuple[float, ...]:
    return tuple(_float32(scale * value) for value in direction)


def content_embedder(
    latent_values: Sequence[float],
    hf_carrier_result: HfCarrierResult | None = None,
    *,
    lf_carrier_result: LfCarrierResult | None = None,
    mixing_coefficient: float | None = None,
) -> ContentEmbeddingResult:
    """产生 HF-only、LF-only 或冻结 `a` 的 combined 理论更新。"""

    if hf_carrier_result is None and lf_carrier_result is None:
        raise ContentEmbedderError("at least one content carrier must be active")

    hf_carrier_value: HfCarrierResult | None = None
    hf_direction: tuple[float, ...] | None = None
    lf_carrier_value: LfCarrierResult | None = None
    lf_direction: tuple[float, ...] | None = None
    if hf_carrier_result is not None:
        hf_carrier_value, hf_direction = _validated_hf_direction(
            hf_carrier_result
        )
    if lf_carrier_result is not None:
        lf_carrier_value, lf_direction = _validated_lf_direction(
            lf_carrier_result
        )

    reference_shape = (
        hf_carrier_value.shape
        if hf_carrier_value is not None
        else lf_carrier_value.shape
    )
    element_count = len(
        hf_direction if hf_direction is not None else lf_direction
    )
    if (
        hf_carrier_value is not None
        and lf_carrier_value is not None
        and (
            hf_carrier_value.shape != lf_carrier_value.shape
            or hf_carrier_value.root_key_public_digest
            != lf_carrier_value.root_key_public_digest
            or hf_carrier_value.key_role != lf_carrier_value.key_role
            or hf_carrier_value.wrong_key_index
            != lf_carrier_value.wrong_key_index
        )
    ):
        raise ContentEmbedderError(
            "active LF/HF carriers must share shape and key semantics"
        )

    latent = _vector(latent_values, element_count, "latent_values")
    latent_norm = _l2_norm(latent)
    if latent_norm == 0.0:
        raise ContentEmbedderError("callback latent has zero L2 energy")

    gamma_lh: float | None
    mixing_value: float | None
    lf_scale: float | None
    hf_scale: float | None
    if lf_direction is not None and hf_direction is not None:
        if (
            isinstance(mixing_coefficient, bool)
            or not isinstance(mixing_coefficient, (int, float))
            or float(mixing_coefficient) not in FROZEN_MIXING_COEFFICIENTS
        ):
            raise ContentEmbedderError(
                "combined mode requires a in {0.25,0.50,0.75}"
            )
        mode: EmbeddingMode = "combined"
        mixing_value = float(mixing_coefficient)
        gamma_accumulator = 0.0
        for lf_value, hf_value in zip(
            lf_direction,
            hf_direction,
            strict=True,
        ):
            gamma_accumulator += lf_value * hf_value
        gamma_lh = gamma_accumulator
        if not isfinite(gamma_lh) or not -1.00002 <= gamma_lh <= 1.00002:
            raise ContentEmbedderError("LF/HF direction inner product is invalid")
        one_minus_a = 1.0 - mixing_value
        c_squared = (
            mixing_value * mixing_value
            + one_minus_a * one_minus_a
            + 2.0 * mixing_value * one_minus_a * gamma_lh
        )
        if not isfinite(c_squared) or c_squared <= 0.0:
            raise ContentEmbedderError(
                "combined content direction has zero or invalid norm"
            )
        combined_norm = sqrt(c_squared)
        combined_raw = tuple(
            _float32(mixing_value * lf_value + one_minus_a * hf_value)
            for lf_value, hf_value in zip(
                lf_direction,
                hf_direction,
                strict=True,
            )
        )
        if _l2_norm(combined_raw) == 0.0:
            raise ContentEmbedderError(
                "combined content direction vanished in float32"
            )
        content_direction = tuple(
            _float32(value / combined_norm) for value in combined_raw
        )
        if abs(_l2_norm(content_direction) - 1.0) > 2e-5:
            raise ContentEmbedderError(
                "combined content direction failed unit normalization"
            )
        lf_scale = mixing_value / combined_norm
        hf_scale = one_minus_a / combined_norm
    elif hf_direction is not None:
        if mixing_coefficient is not None:
            raise ContentEmbedderError(
                "HF-only mode does not accept a mixing coefficient"
            )
        mode = "hf_only"
        mixing_value = None
        gamma_lh = None
        combined_norm = 1.0
        content_direction = hf_direction
        lf_scale = None
        hf_scale = 1.0
    else:
        if mixing_coefficient is not None:
            raise ContentEmbedderError(
                "LF-only mode does not accept a mixing coefficient"
            )
        mode = "lf_only"
        mixing_value = None
        gamma_lh = None
        combined_norm = 1.0
        content_direction = lf_direction
        lf_scale = 1.0
        hf_scale = None

    target_relative_l2 = _float32(
        CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    )
    target_total_norm = _float32(target_relative_l2 * latent_norm)
    if target_total_norm == 0.0:
        raise ContentEmbedderError(
            "target content update vanished in float32"
        )
    delta_content = _target_component(
        target_total_norm,
        content_direction,
    )
    if _l2_norm(delta_content) == 0.0:
        raise ContentEmbedderError("delta_content has zero L2 energy")

    target_component_lf = (
        _target_component(target_total_norm * lf_scale, lf_direction)
        if lf_scale is not None and lf_direction is not None
        else None
    )
    target_component_hf = (
        _target_component(target_total_norm * hf_scale, hf_direction)
        if hf_scale is not None and hf_direction is not None
        else None
    )
    identity = {
        "candidate_ids": list(EMBEDDER_CANDIDATE_IDS),
        "hf_carrier_config_digest": (
            hf_carrier_value.carrier_config_digest
            if hf_carrier_value is not None
            else None
        ),
        "lf_carrier_config_digest": (
            lf_carrier_value.carrier_config_digest
            if lf_carrier_value is not None
            else None
        ),
        "mixing_coefficient": (
            int(mixing_value * 100) if mixing_value is not None else None
        ),
        "mixing_coefficient_denominator": (
            100 if mixing_value is not None else None
        ),
        "mode": mode,
        "rho_content_denominator": CONTENT_RELATIVE_L2_DENOMINATOR,
        "rho_content_numerator": CONTENT_RELATIVE_L2_NUMERATOR,
        "shape": list(reference_shape),
    }
    return ContentEmbeddingResult(
        candidate_ids=EMBEDDER_CANDIDATE_IDS,
        mode=mode,
        shape=reference_shape,
        delta_content=delta_content,
        delta_content_digest=_digest(delta_content),
        latent_norm=latent_norm,
        target_total_norm=target_total_norm,
        target_relative_l2=target_relative_l2,
        mixing_coefficient=mixing_value,
        gamma_lh=gamma_lh,
        combined_pre_normalization_norm=combined_norm,
        target_component_lf=target_component_lf,
        target_component_hf=target_component_hf,
        target_component_lf_norm=(
            _l2_norm(target_component_lf)
            if target_component_lf is not None
            else None
        ),
        target_component_hf_norm=(
            _l2_norm(target_component_hf)
            if target_component_hf is not None
            else None
        ),
        lf_carrier_config_digest=(
            lf_carrier_value.carrier_config_digest
            if lf_carrier_value is not None
            else None
        ),
        hf_carrier_config_digest=(
            hf_carrier_value.carrier_config_digest
            if hf_carrier_value is not None
            else None
        ),
        embedder_config_digest=sha256(
            stable_json_utf8(identity)
        ).hexdigest(),
    )
