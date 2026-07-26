"""CEG-WM 独立 LF blind detector。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, sqrt
from struct import pack, unpack
from typing import Sequence

from main.shared.key_schedule import DerivedWrongKeyMaterial, stable_json_utf8

from .hf_carrier import MODEL_REVISION
from .lf_carrier import LfCarrierError, lf_carrier

OBSERVATION_PROTOCOL = "final_image_vae_posterior_mode"


class LfDetectorError(ValueError):
    """普通图像侧 LF 观测、模板或 blind score 无效。"""


def _float32(value: float) -> float:
    if not isfinite(value):
        raise LfDetectorError("LF detector value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise LfDetectorError(
            "LF detector value is outside binary32 range"
        ) from exc
    if not isfinite(rounded):
        raise LfDetectorError("LF detector binary32 value must be finite")
    return rounded


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise LfDetectorError("LF observation shape must be [1,C,H,W]")
    normalized = tuple(shape)
    if (
        len(normalized) != 4
        or normalized[0] != 1
        or any(type(size) is not int or size <= 0 for size in normalized)
    ):
        raise LfDetectorError("LF observation shape must be positive [1,C,H,W]")
    return normalized


def _vector(
    values: Sequence[float],
    size: int,
    role: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LfDetectorError(f"{role} must be a numeric sequence")
    if len(values) != size:
        raise LfDetectorError(f"{role} length does not match observation shape")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LfDetectorError(f"{role} must contain only finite numbers")
        normalized.append(_float32(float(value)))
    return tuple(normalized)


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


@dataclass(frozen=True, slots=True)
class LfDetectionObservation:
    """普通检测图像经公共 VAE-mode 编码后的 LF 观测。"""

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
            normalized_shape[0]
            * normalized_shape[1]
            * normalized_shape[2]
            * normalized_shape[3],
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
    ) -> LfDetectionObservation:
        """建立不含 route、参考图、embed record 或 callback latent 的观测。"""

        return cls(values=tuple(values), shape=tuple(shape))


@dataclass(frozen=True, slots=True)
class LfDetectionResult:
    """独立可观测的 LF blind score 与检测身份。"""

    candidate_id: str
    candidate_ids: tuple[str, ...]
    lf_score: float
    detector_identity: str
    detector_config_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    observation_digest: str
    template_digest: str


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
        raise LfDetectorError(f"{role} has zero centered L2 energy")
    return tuple(_float32(value / norm) for value in values)


def lf_detector(
    observation: LfDetectionObservation,
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    model_revision: str = MODEL_REVISION,
) -> LfDetectionResult:
    """从普通图像观测与 key 盲重构未 mask LF 模板并评分。"""

    if type(observation) is not LfDetectionObservation:
        raise LfDetectorError(
            "LF detector requires a public-image LfDetectionObservation"
        )
    if observation.observation_protocol != OBSERVATION_PROTOCOL:
        raise LfDetectorError("LF observation protocol identity mismatch")
    if _digest(observation.values) != observation.observation_digest:
        raise LfDetectorError("LF observation digest mismatch")
    try:
        carrier = lf_carrier(
            detection_key,
            observation.shape,
            mask_lf=None,
            model_revision=model_revision,
        )
    except LfCarrierError as exc:
        raise LfDetectorError("LF detector template reconstruction failed") from exc

    observation_unit = _normalize(
        _center(observation.values),
        "LF observation",
    )
    template_unit = _normalize(
        _center(carrier.template),
        "LF template",
    )
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
        raise LfDetectorError("LF blind score must be finite")

    detector_config = {
        "candidate_ids": ["key_schedule_sha256_counter", "lf_low_pass"],
        "carrier_config_digest": carrier.carrier_config_digest,
        "dtype": "float32",
        "model_revision": model_revision,
        "observation_protocol": OBSERVATION_PROTOCOL,
        "score_operator": "centered_normalized_correlation",
        "template_mask": "unmasked",
    }
    detector_config_digest = sha256(
        stable_json_utf8(detector_config)
    ).hexdigest()
    detector_identity = sha256(
        stable_json_utf8(
            {
                "candidate_ids": [
                    "key_schedule_sha256_counter",
                    "lf_low_pass",
                ],
                "detector_config_digest": detector_config_digest,
                "detector_role": "lf_blind_score",
            }
        )
    ).hexdigest()
    return LfDetectionResult(
        candidate_id="lf_low_pass",
        candidate_ids=("key_schedule_sha256_counter", "lf_low_pass"),
        lf_score=score,
        detector_identity=detector_identity,
        detector_config_digest=detector_config_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        observation_digest=observation.observation_digest,
        template_digest=carrier.template_digest,
    )
