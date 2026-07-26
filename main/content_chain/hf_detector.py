"""CEG-WM HF direct blind detector。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, sqrt
from struct import pack, unpack
from typing import Sequence

from main.shared.key_schedule import (
    DerivedWrongKeyMaterial,
    stable_json_utf8,
)

from .hf_carrier import MODEL_REVISION, HfCarrierError, hf_carrier

OBSERVATION_PROTOCOL = "final_image_vae_posterior_mode"


class HfDetectorError(ValueError):
    """普通图像侧 HF 观测、模板或 direct score 无效。"""


@dataclass(frozen=True, slots=True)
class HfDetectionObservation:
    """普通检测图像经公共 VAE-mode 编码后的不可变观测。"""

    values: tuple[float, ...]
    shape: tuple[int, int, int, int]
    observation_protocol: str = field(
        default=OBSERVATION_PROTOCOL,
        init=False,
    )
    observation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        normalized_shape = _validate_shape(self.shape)
        normalized_values = _vector(
            self.values,
            _element_count(normalized_shape),
            "detection observation",
        )
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "values", normalized_values)
        object.__setattr__(
            self,
            "observation_digest",
            _digest(normalized_values),
        )

    @classmethod
    def from_public_image_encoding(
        cls,
        values: Sequence[float],
        shape: Sequence[int],
    ) -> HfDetectionObservation:
        """建立不含参考图、embed record 或 callback latent 的检测观测。"""

        return cls(values=tuple(values), shape=tuple(shape))


@dataclass(frozen=True, slots=True)
class HfDetectionResult:
    """独立可观测的 HF blind score 与检测身份。"""

    candidate_id: str
    hf_score: float
    detector_identity: str
    detector_config_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    observation_digest: str
    template_digest: str


def _element_count(shape: tuple[int, int, int, int]) -> int:
    return shape[0] * shape[1] * shape[2] * shape[3]


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise HfDetectorError("HF observation shape must be [1,C,H,W]")
    normalized = tuple(shape)
    if (
        len(normalized) != 4
        or normalized[0] != 1
        or any(type(size) is not int or size <= 0 for size in normalized)
    ):
        raise HfDetectorError("HF observation shape must be positive [1,C,H,W]")
    return normalized


def _float32(value: float) -> float:
    if not isfinite(value):
        raise HfDetectorError("HF detector value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise HfDetectorError("HF detector value is outside binary32 range") from exc
    if not isfinite(rounded):
        raise HfDetectorError("HF detector binary32 value must be finite")
    return rounded


def _vector(values: Sequence[float], size: int, role: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HfDetectorError(f"{role} must be a numeric sequence")
    if len(values) != size:
        raise HfDetectorError(f"{role} length does not match observation shape")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HfDetectorError(f"{role} must contain only finite numbers")
        converted.append(_float32(float(value)))
    return tuple(converted)


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _center(values: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    for value in values:
        total = _float32(total + value)
    mean = _float32(total / len(values))
    return tuple(_float32(value - mean) for value in values)


def _normalize(values: Sequence[float], role: str) -> tuple[float, ...]:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    norm = _float32(sqrt(squared_sum))
    if norm == 0.0:
        raise HfDetectorError(f"{role} has zero centered L2 energy")
    return tuple(_float32(value / norm) for value in values)


def hf_detector(
    observation: HfDetectionObservation,
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    model_revision: str = MODEL_REVISION,
) -> HfDetectionResult:
    """从普通图像侧观测和 key 盲重构模板并计算 HF direct score。"""

    if type(observation) is not HfDetectionObservation:
        raise HfDetectorError(
            "HF detector requires a public-image HfDetectionObservation"
        )
    if observation.observation_protocol != OBSERVATION_PROTOCOL:
        raise HfDetectorError("HF observation protocol identity mismatch")
    if _digest(observation.values) != observation.observation_digest:
        raise HfDetectorError("HF observation digest mismatch")
    try:
        carrier = hf_carrier(
            detection_key,
            observation.shape,
            mask_hf=None,
            model_revision=model_revision,
        )
    except HfCarrierError as exc:
        raise HfDetectorError("HF detector template reconstruction failed") from exc

    centered_observation = _center(observation.values)
    centered_template = _center(carrier.template)
    observation_unit = _normalize(centered_observation, "HF observation")
    template_unit = _normalize(centered_template, "HF template")
    score = 0.0
    for observed_value, template_value in zip(
        observation_unit,
        template_unit,
        strict=True,
    ):
        score = _float32(
            score + _float32(observed_value * template_value)
        )
    if not isfinite(score):
        raise HfDetectorError("HF direct score must be finite")

    detector_config = {
        "candidate_id": "hf_sparse_tail",
        "carrier_config_digest": carrier.carrier_config_digest,
        "dtype": "float32",
        "model_revision": model_revision,
        "observation_protocol": OBSERVATION_PROTOCOL,
        "score_operator": "centered_normalized_correlation",
        "template_time_centering": False,
    }
    detector_config_digest = sha256(
        stable_json_utf8(detector_config)
    ).hexdigest()
    detector_identity_value = {
        "candidate_id": "hf_sparse_tail",
        "detector_config_digest": detector_config_digest,
        "detector_role": "hf_direct_score",
    }
    return HfDetectionResult(
        candidate_id="hf_sparse_tail",
        hf_score=score,
        detector_identity=sha256(
            stable_json_utf8(detector_identity_value)
        ).hexdigest(),
        detector_config_digest=detector_config_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        observation_digest=observation.observation_digest,
        template_digest=carrier.template_digest,
    )
