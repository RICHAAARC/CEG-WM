"""CEG-WM 独立 LF blind detector。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import cos, isfinite, nextafter, sqrt
from struct import pack, unpack
from typing import Sequence

import numpy as np

from main.shared.key_schedule import (
    DerivedInternalLfDecoyMaterial,
    DerivedWrongKeyMaterial,
    derive_internal_lf_decoy_material,
    stable_json_utf8,
)

from .lf_carrier import (
    CONTRASTIVE_LF_CANDIDATE_IDS,
    MULTISCALE_CONTRASTIVE_CANDIDATE_ID,
    ContrastiveLfCarrierResult,
    LfCarrierError,
    contrastive_lf_carrier,
    contrastive_lowpass,
    lf_carrier,
)
from .lf_whitening import (
    LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LF_NULL_WHITENING_BAND_IDENTITY,
    LF_NULL_WHITENING_DETREND_IDENTITY,
    LF_NULL_WHITENING_LATENT_SHAPE,
    LF_NULL_WHITENING_TRANSFORM_IDENTITY,
    LfNullWhiteningAsset,
    LfNullWhiteningAssetError,
    SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID,
    SemanticTextureLfWhiteningAsset,
)
from .routing import (
    SEMANTIC_TEXTURE_CANDIDATE_STATUS,
    SemanticTextureRoutingResult,
    validate_semantic_texture_routing_result,
)

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


@dataclass(frozen=True, slots=True)
class ContrastiveLfRawObservation:
    """Blind registered/wrong raw vector plus separate internal-decoy vectors."""

    candidate_id: str
    raw_feature: tuple[float, ...]
    internal_decoy_features: tuple[tuple[float, ...], ...]
    observation_digest: str
    carrier_config_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    raw_observation_digest: str


@dataclass(frozen=True, slots=True)
class ContrastiveLfNullAsset:
    """Candidate-specific population asset; multiscale stores joint whitening."""

    candidate_id: str
    population_count: int
    raw_feature_population: tuple[tuple[float, ...], ...]
    population_mean: tuple[float, ...]
    population_covariance: tuple[float, ...]
    regularized_covariance: tuple[float, ...]
    whitening_matrix: tuple[float, ...]
    contrastive_population: tuple[float, ...]
    contrastive_population_mean: float
    contrastive_population_variance: float
    contrastive_population_sigma: float
    provisional_tau: float
    null_manifest_digest: str
    detector_config_digest: str
    asset_digest: str

    def validate(self) -> None:
        """Recompute the complete candidate-specific replay authority."""

        if (
            self.candidate_id not in CONTRASTIVE_LF_CANDIDATE_IDS
            or self.population_count != 32
            or len(self.raw_feature_population) != 32
            or len(self.contrastive_population) != 32
            or any(not isfinite(value) for row in self.raw_feature_population for value in row)
            or any(not isfinite(value) for value in self.contrastive_population)
        ):
            raise LfDetectorError("contrastive LF null asset population is invalid")
        dimension = 2 if self.candidate_id == MULTISCALE_CONTRASTIVE_CANDIDATE_ID else 1
        if any(len(row) != dimension for row in self.raw_feature_population):
            raise LfDetectorError("contrastive LF null asset feature dimension drifted")
        expected = _contrastive_lf_asset_payload(
            candidate_id=self.candidate_id,
            raw_feature_population=self.raw_feature_population,
            population_mean=self.population_mean,
            population_covariance=self.population_covariance,
            regularized_covariance=self.regularized_covariance,
            whitening_matrix=self.whitening_matrix,
            contrastive_population=self.contrastive_population,
            contrastive_population_mean=self.contrastive_population_mean,
            contrastive_population_variance=self.contrastive_population_variance,
            contrastive_population_sigma=self.contrastive_population_sigma,
            provisional_tau=self.provisional_tau,
            null_manifest_digest=self.null_manifest_digest,
            detector_config_digest=self.detector_config_digest,
        )
        if self.asset_digest != sha256(stable_json_utf8(expected)).hexdigest():
            raise LfDetectorError("contrastive LF null asset digest drifted")


@dataclass(frozen=True, slots=True)
class ContrastiveLfDetectionResult:
    candidate_id: str
    raw_feature: tuple[float, ...]
    internal_decoy_features: tuple[tuple[float, ...], ...]
    registered_score: float
    internal_decoy_scores: tuple[float, ...]
    contrastive_score: float
    standardized_score: float
    detector_identity: str
    null_asset_digest: str
    observation_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None


@dataclass(frozen=True, slots=True)
class LfNullWhitenedDetectionResult:
    """LF clean-null-whitened blind score with an explicit public asset."""

    candidate_id: str
    candidate_ids: tuple[str, ...]
    lf_score: float
    detector_identity: str
    detector_config_digest: str
    whitening_asset_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    observation_digest: str
    template_digest: str


@dataclass(frozen=True, slots=True)
class SemanticTextureLfDetectionResult:
    """Dedicated-W blind soft-routed LF score with no legacy fallback."""

    candidate_id: str
    candidate_status: str
    candidate_ids: tuple[str, ...]
    lf_score: float
    detector_identity: str
    detector_config_digest: str
    whitening_asset_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    observation_digest: str
    template_digest: str
    route_identity: str


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


def _contrastive_correlation(
    observed: Sequence[float], template: Sequence[float]
) -> float:
    if len(observed) != len(template) or not observed:
        raise LfDetectorError("contrastive LF correlation shape mismatch")
    observed_unit = _normalize(_center(observed), "contrastive LF observation")
    template_unit = _normalize(_center(template), "contrastive LF template")
    score = _float32(0.0)
    for observed_value, template_value in zip(
        observed_unit, template_unit, strict=True
    ):
        score = _float32(score + _float32(observed_value * template_value))
    return score


def _raw_feature(
    observation: LfDetectionObservation,
    carrier: ContrastiveLfCarrierResult,
) -> tuple[float, ...]:
    y_five = contrastive_lowpass(observation.values, observation.shape, 5)
    values = [
        _contrastive_correlation(y_five, carrier.scale_five_template)
    ]
    if carrier.candidate_id == MULTISCALE_CONTRASTIVE_CANDIDATE_ID:
        if carrier.scale_nine_template is None:
            raise LfDetectorError("multiscale LF carrier lost scale nine")
        y_nine = contrastive_lowpass(observation.values, observation.shape, 9)
        values.append(
            _contrastive_correlation(y_nine, carrier.scale_nine_template)
        )
    return tuple(values)


def contrastive_lf_raw_observation(
    observation: LfDetectionObservation,
    detection_key: str | DerivedWrongKeyMaterial,
    *,
    candidate_id: str,
) -> ContrastiveLfRawObservation:
    """Rebuild a blind Stage-A candidate and eight separate internal decoys."""

    if type(observation) is not LfDetectionObservation:
        raise LfDetectorError("contrastive LF detector requires public-image observation")
    if candidate_id not in CONTRASTIVE_LF_CANDIDATE_IDS:
        raise LfDetectorError("contrastive LF candidate is not registered")
    try:
        carrier = contrastive_lf_carrier(
            detection_key, observation.shape, candidate_id=candidate_id
        )
        decoy_features = []
        for index in range(8):
            material: DerivedInternalLfDecoyMaterial = derive_internal_lf_decoy_material(
                carrier.root_key_public_digest, candidate_id, index
            )
            decoy_carrier = contrastive_lf_carrier(
                material, observation.shape, candidate_id=candidate_id
            )
            decoy_features.append(_raw_feature(observation, decoy_carrier))
    except (LfCarrierError, ValueError) as exc:
        raise LfDetectorError("contrastive LF carrier reconstruction failed") from exc
    feature = _raw_feature(observation, carrier)
    identity = {
        "candidate_id": candidate_id,
        "carrier_config_digest": carrier.carrier_config_digest,
        "internal_decoy_feature_digests": [
            sha256(stable_json_utf8([float.hex(value) for value in item])).hexdigest()
            for item in decoy_features
        ],
        "observation_digest": observation.observation_digest,
        "raw_feature": [float.hex(value) for value in feature],
    }
    return ContrastiveLfRawObservation(
        candidate_id=candidate_id,
        raw_feature=feature,
        internal_decoy_features=tuple(decoy_features),
        observation_digest=observation.observation_digest,
        carrier_config_digest=carrier.carrier_config_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        raw_observation_digest=sha256(stable_json_utf8(identity)).hexdigest(),
    )


def _score_feature(
    candidate_id: str,
    feature: Sequence[float],
    mean: Sequence[float],
    whitening: Sequence[float],
) -> float:
    if candidate_id != MULTISCALE_CONTRASTIVE_CANDIDATE_ID:
        return float(feature[0])
    centered = np.asarray(feature, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    matrix = np.asarray(whitening, dtype=np.float64).reshape(2, 2)
    whitened = matrix @ centered
    score = float((whitened[0] + whitened[1]) / sqrt(2.0))
    if not isfinite(score):
        raise LfDetectorError("multiscale LF whitened score is non-finite")
    return score


def _even_median_eight(values: Sequence[float]) -> float:
    if len(values) != 8 or any(not isfinite(value) for value in values):
        raise LfDetectorError("contrastive LF internal decoy roster is invalid")
    ordered = sorted(float(value) for value in values)
    value = (ordered[3] + ordered[4]) / 2.0
    if not isfinite(value):
        raise LfDetectorError("contrastive LF decoy median is non-finite")
    return value


def _contrastive_lf_asset_payload(
    *,
    candidate_id: str,
    raw_feature_population: Sequence[Sequence[float]],
    population_mean: Sequence[float],
    population_covariance: Sequence[float],
    regularized_covariance: Sequence[float],
    whitening_matrix: Sequence[float],
    contrastive_population: Sequence[float],
    contrastive_population_mean: float,
    contrastive_population_variance: float,
    contrastive_population_sigma: float,
    provisional_tau: float,
    null_manifest_digest: str,
    detector_config_digest: str,
) -> dict[str, object]:
    dimension = 2 if candidate_id == MULTISCALE_CONTRASTIVE_CANDIDATE_ID else 1
    return {
        "candidate_id": candidate_id,
        "contrastive": "registered_minus_binary64_even_median_internal_decoy_eight",
        "null_standardization": "binary64_population_divide_by_32",
        "raw_feature": (
            "joint_whitened_equal_direction_scale_five_scale_nine"
            if dimension == 2
            else "scale_five_normalized_correlation"
        ),
        "contrastive_population": [float.hex(value) for value in contrastive_population],
        "contrastive_population_mean": float.hex(contrastive_population_mean),
        "contrastive_population_sigma": float.hex(contrastive_population_sigma),
        "contrastive_population_variance": float.hex(contrastive_population_variance),
        "detector_config_digest": detector_config_digest,
        "null_manifest_digest": null_manifest_digest,
        "population_covariance": [float.hex(value) for value in population_covariance],
        "population_mean": [float.hex(value) for value in population_mean],
        "provisional_tau": float.hex(provisional_tau),
        "raw_feature_population": [
            [float.hex(value) for value in row] for row in raw_feature_population
        ],
        "regularized_covariance": [float.hex(value) for value in regularized_covariance],
        "whitening_matrix": [float.hex(value) for value in whitening_matrix],
    }


def fit_contrastive_lf_null_asset(
    raw_observations: Sequence[ContrastiveLfRawObservation],
    *,
    candidate_id: str,
    null_manifest_digest: str,
) -> ContrastiveLfNullAsset:
    """Fit the frozen 32-item population whitening/standardization asset."""

    if (
        candidate_id not in CONTRASTIVE_LF_CANDIDATE_IDS
        or len(raw_observations) != 32
        or len(null_manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in null_manifest_digest)
        or any(
            type(item) is not ContrastiveLfRawObservation
            or item.candidate_id != candidate_id
            or item.key_role != "registered"
            for item in raw_observations
        )
    ):
        raise LfDetectorError("contrastive LF null population authority is invalid")
    dimension = 2 if candidate_id == MULTISCALE_CONTRASTIVE_CANDIDATE_ID else 1
    matrix = np.asarray([item.raw_feature for item in raw_observations], dtype=np.float64)
    if matrix.shape != (32, dimension) or not np.isfinite(matrix).all():
        raise LfDetectorError("contrastive LF null raw matrix is invalid")
    mean = np.sum(matrix, axis=0, dtype=np.float64) / 32.0
    centered = matrix - mean
    covariance = (centered.T @ centered) / 32.0
    if dimension == 2:
        trace = float(np.trace(covariance))
        if not isfinite(trace) or trace <= 0.0:
            raise LfDetectorError("contrastive LF multiscale covariance trace is invalid")
        ridge = (2.0 ** -10) * (trace / 2.0)
        regularized = covariance + ridge * np.eye(2, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        if np.any(eigenvalues <= 0.0) or not np.isfinite(eigenvalues).all():
            raise LfDetectorError("contrastive LF covariance is not positive definite")
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    else:
        trace = float(covariance[0, 0])
        if not isfinite(trace) or trace <= 0.0:
            raise LfDetectorError("single-scale LF raw population variance is invalid")
        regularized = covariance.copy()
        whitening_matrix = np.eye(1, dtype=np.float64)
    scores = []
    for item in raw_observations:
        registered = _score_feature(candidate_id, item.raw_feature, mean, whitening_matrix.reshape(-1))
        decoys = tuple(
            _score_feature(candidate_id, feature, mean, whitening_matrix.reshape(-1))
            for feature in item.internal_decoy_features
        )
        scores.append(registered - _even_median_eight(decoys))
    contrastive_mean = sum(scores) / 32.0
    contrastive_variance = sum((value - contrastive_mean) ** 2 for value in scores) / 32.0
    contrastive_sigma = sqrt(contrastive_variance)
    if not isfinite(contrastive_sigma) or contrastive_sigma <= 0.0:
        raise LfDetectorError("contrastive LF population sigma is invalid")
    z_values = [(value - contrastive_mean) / contrastive_sigma for value in scores]
    tau = nextafter(sorted(z_values)[-4], float("inf"))
    detector_config = {
        "candidate_id": candidate_id,
        "contrastive": "registered_minus_binary64_even_median_internal_decoy_eight",
        "null_standardization": "binary64_population_divide_by_32",
        "raw_feature": (
            "joint_whitened_equal_direction_scale_five_scale_nine"
            if dimension == 2
            else "scale_five_normalized_correlation"
        ),
    }
    detector_config_digest = sha256(stable_json_utf8(detector_config)).hexdigest()
    raw_population = tuple(tuple(float(value) for value in row) for row in matrix)
    payload = _contrastive_lf_asset_payload(
        candidate_id=candidate_id,
        raw_feature_population=raw_population,
        population_mean=tuple(float(value) for value in mean),
        population_covariance=tuple(float(value) for value in covariance.reshape(-1)),
        regularized_covariance=tuple(float(value) for value in regularized.reshape(-1)),
        whitening_matrix=tuple(float(value) for value in whitening_matrix.reshape(-1)),
        contrastive_population=scores,
        contrastive_population_mean=contrastive_mean,
        contrastive_population_variance=contrastive_variance,
        contrastive_population_sigma=contrastive_sigma,
        provisional_tau=tau,
        null_manifest_digest=null_manifest_digest,
        detector_config_digest=detector_config_digest,
    )
    asset_digest = sha256(stable_json_utf8(payload)).hexdigest()
    asset = ContrastiveLfNullAsset(
        candidate_id=candidate_id,
        population_count=32,
        raw_feature_population=raw_population,
        population_mean=tuple(float(value) for value in mean),
        population_covariance=tuple(float(value) for value in covariance.reshape(-1)),
        regularized_covariance=tuple(float(value) for value in regularized.reshape(-1)),
        whitening_matrix=tuple(float(value) for value in whitening_matrix.reshape(-1)),
        contrastive_population=tuple(scores),
        contrastive_population_mean=contrastive_mean,
        contrastive_population_variance=contrastive_variance,
        contrastive_population_sigma=contrastive_sigma,
        provisional_tau=tau,
        null_manifest_digest=null_manifest_digest,
        detector_config_digest=detector_config_digest,
        asset_digest=asset_digest,
    )
    asset.validate()
    return asset


def contrastive_lf_detector(
    raw_observation: ContrastiveLfRawObservation,
    null_asset: ContrastiveLfNullAsset,
) -> ContrastiveLfDetectionResult:
    """Apply one candidate-specific frozen null asset to a blind raw observation."""

    if (
        type(raw_observation) is not ContrastiveLfRawObservation
        or type(null_asset) is not ContrastiveLfNullAsset
        or raw_observation.candidate_id != null_asset.candidate_id
    ):
        raise LfDetectorError("contrastive LF detector asset binding is invalid")
    null_asset.validate()
    registered = _score_feature(
        null_asset.candidate_id,
        raw_observation.raw_feature,
        null_asset.population_mean,
        null_asset.whitening_matrix,
    )
    decoys = tuple(
        _score_feature(
            null_asset.candidate_id,
            feature,
            null_asset.population_mean,
            null_asset.whitening_matrix,
        )
        for feature in raw_observation.internal_decoy_features
    )
    contrastive = registered - _even_median_eight(decoys)
    z = (contrastive - null_asset.contrastive_population_mean) / null_asset.contrastive_population_sigma
    if not isfinite(z):
        raise LfDetectorError("contrastive LF standardized score is non-finite")
    identity = sha256(
        stable_json_utf8(
            {
                "candidate_id": null_asset.candidate_id,
                "detector_config_digest": null_asset.detector_config_digest,
                "null_asset_digest": null_asset.asset_digest,
            }
        )
    ).hexdigest()
    return ContrastiveLfDetectionResult(
        candidate_id=null_asset.candidate_id,
        raw_feature=raw_observation.raw_feature,
        internal_decoy_features=raw_observation.internal_decoy_features,
        registered_score=registered,
        internal_decoy_scores=decoys,
        contrastive_score=contrastive,
        standardized_score=z,
        detector_identity=identity,
        null_asset_digest=null_asset.asset_digest,
        observation_digest=raw_observation.observation_digest,
        root_key_public_digest=raw_observation.root_key_public_digest,
        key_role=raw_observation.key_role,
        wrong_key_index=raw_observation.wrong_key_index,
    )


_DCT_PI = float.fromhex("0x1.921fb54442d18p+1")
_DCT_SIZE = 64
_DCT_BASIS = tuple(
    tuple(
        sqrt(1.0 / _DCT_SIZE)
        if frequency == 0
        else sqrt(2.0 / _DCT_SIZE)
        * cos(
            _DCT_PI
            * (coordinate + 0.5)
            * frequency
            / _DCT_SIZE
        )
        for coordinate in range(_DCT_SIZE)
    )
    for frequency in range(_DCT_SIZE)
)
_NORMALIZED_COORDINATES = tuple(
    (2.0 * coordinate - 63.0) / 63.0
    for coordinate in range(_DCT_SIZE)
)
_NORMALIZED_COORDINATE_SQUARED_SUM = sum(
    coordinate * coordinate for coordinate in _NORMALIZED_COORDINATES
)


def _affine_detrended_dct(
    values: Sequence[float],
    *,
    role: str,
) -> np.ndarray:
    residual = np.empty((16, 64, 64), dtype=np.float64, order="C")
    offset = 0
    denominator = 64.0 * _NORMALIZED_COORDINATE_SQUARED_SUM
    for channel in range(16):
        channel_values = values[offset : offset + 64 * 64]
        offset += 64 * 64
        constant_sum = 0.0
        for height in range(64):
            row_offset = height * 64
            for width in range(64):
                value = float(channel_values[row_offset + width])
                constant_sum += value
        constant = constant_sum / (64.0 * 64.0)
        height_sum = 0.0
        for height in range(64):
            height_coordinate = _NORMALIZED_COORDINATES[height]
            row_offset = height * 64
            for width in range(64):
                height_sum += (
                    height_coordinate
                    * float(channel_values[row_offset + width])
                )
        height_slope = height_sum / denominator
        width_sum = 0.0
        for height in range(64):
            row_offset = height * 64
            for width in range(64):
                width_sum += (
                    _NORMALIZED_COORDINATES[width]
                    * float(channel_values[row_offset + width])
                )
        width_slope = width_sum / denominator
        for height in range(64):
            row_offset = height * 64
            for width in range(64):
                residual_value = (
                    float(channel_values[row_offset + width])
                    - constant
                    - height_slope * _NORMALIZED_COORDINATES[height]
                    - width_slope * _NORMALIZED_COORDINATES[width]
                )
                if not isfinite(residual_value):
                    raise LfDetectorError(
                        f"{role} affine residual must be finite"
                    )
                residual[channel, height, width] = residual_value

    basis = np.asarray(_DCT_BASIS, dtype=np.float64, order="C")
    coefficients = np.einsum(
        "chw,uh,vw->cuv",
        residual,
        basis,
        basis,
        dtype=np.float64,
        order="C",
        casting="no",
        optimize=False,
    )
    if not coefficients.flags.c_contiguous:
        raise LfDetectorError(f"{role} DCT coefficient order drifted")
    if not np.isfinite(coefficients).all():
        raise LfDetectorError(f"{role} DCT coefficients must be finite")
    return coefficients


_LF_WHITENED_COEFFICIENT_SHAPE = (16, 64, 64)


def _coefficient_digest(coefficients: np.ndarray) -> str:
    return sha256(coefficients.tobytes(order="C")).hexdigest()


def _freeze_coefficients(coefficients: np.ndarray) -> np.ndarray:
    frozen = np.frombuffer(
        coefficients.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(_LF_WHITENED_COEFFICIENT_SHAPE)
    if frozen.flags.writeable:
        raise LfDetectorError("prepared LF coefficients must be immutable")
    return frozen


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_prepared_coefficients(
    coefficients: np.ndarray,
    coefficients_digest: str,
    *,
    role: str,
) -> None:
    if type(coefficients) is not np.ndarray:
        raise LfDetectorError(f"{role} coefficients must be an exact ndarray")
    if coefficients.dtype != np.dtype(np.float64):
        raise LfDetectorError(f"{role} coefficients must use exact float64")
    if coefficients.shape != _LF_WHITENED_COEFFICIENT_SHAPE:
        raise LfDetectorError(f"{role} coefficient shape mismatch")
    if not coefficients.flags.c_contiguous:
        raise LfDetectorError(f"{role} coefficients must be C contiguous")
    if coefficients.flags.writeable:
        raise LfDetectorError(f"{role} coefficients must be read only")
    if not np.isfinite(coefficients).all():
        raise LfDetectorError(f"{role} coefficients must be finite")
    if (
        type(coefficients_digest) is not str
        or _coefficient_digest(coefficients) != coefficients_digest
    ):
        raise LfDetectorError(f"{role} coefficient digest mismatch")


@dataclass(frozen=True, slots=True)
class PreparedLfWhitenedObservation:
    """Immutable deterministic DCT features for one public observation."""

    coefficients: np.ndarray
    coefficients_digest: str
    observation_digest: str
    observation_shape: tuple[int, int, int, int]
    observation_protocol: str
    whitening_asset_digest: str
    detrend_identity: str = LF_NULL_WHITENING_DETREND_IDENTITY
    transform_identity: str = LF_NULL_WHITENING_TRANSFORM_IDENTITY

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_prepared_coefficients(
            self.coefficients,
            self.coefficients_digest,
            role="prepared LF observation",
        )
        if self.observation_shape != LF_NULL_WHITENING_LATENT_SHAPE:
            raise LfDetectorError("prepared LF observation shape mismatch")
        if self.observation_protocol != OBSERVATION_PROTOCOL:
            raise LfDetectorError("prepared LF observation protocol mismatch")
        if self.detrend_identity != LF_NULL_WHITENING_DETREND_IDENTITY:
            raise LfDetectorError("prepared LF observation detrend mismatch")
        if self.transform_identity != LF_NULL_WHITENING_TRANSFORM_IDENTITY:
            raise LfDetectorError("prepared LF observation transform mismatch")
        if not _is_sha256_digest(self.whitening_asset_digest):
            raise LfDetectorError("prepared LF observation asset mismatch")
        if not _is_sha256_digest(self.observation_digest):
            raise LfDetectorError("prepared LF observation digest mismatch")


@dataclass(frozen=True, slots=True)
class PreparedLfWhitenedTemplate:
    """Immutable deterministic DCT features for one reconstructed key template."""

    coefficients: np.ndarray
    coefficients_digest: str
    template_digest: str
    carrier_config_digest: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    template_shape: tuple[int, int, int, int]
    whitening_asset_digest: str
    detrend_identity: str = LF_NULL_WHITENING_DETREND_IDENTITY
    transform_identity: str = LF_NULL_WHITENING_TRANSFORM_IDENTITY

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_prepared_coefficients(
            self.coefficients,
            self.coefficients_digest,
            role="prepared LF template",
        )
        if self.template_shape != LF_NULL_WHITENING_LATENT_SHAPE:
            raise LfDetectorError("prepared LF template shape mismatch")
        if self.detrend_identity != LF_NULL_WHITENING_DETREND_IDENTITY:
            raise LfDetectorError("prepared LF template detrend mismatch")
        if self.transform_identity != LF_NULL_WHITENING_TRANSFORM_IDENTITY:
            raise LfDetectorError("prepared LF template transform mismatch")
        if not _is_sha256_digest(self.whitening_asset_digest):
            raise LfDetectorError("prepared LF template asset mismatch")
        if any(
            not _is_sha256_digest(value)
            for value in (
                self.template_digest,
                self.carrier_config_digest,
                self.root_key_public_digest,
            )
        ):
            raise LfDetectorError("prepared LF template identity mismatch")
        if (
            self.key_role == "registered"
            and self.wrong_key_index is not None
        ) or (
            self.key_role == "wrong"
            and (
                type(self.wrong_key_index) is not int
                or self.wrong_key_index < 0
            )
        ) or self.key_role not in {"registered", "wrong"}:
            raise LfDetectorError("prepared LF template wrong-key mismatch")


def prepare_lf_null_whitened_observation(
    observation: LfDetectionObservation,
    whitening_asset: LfNullWhiteningAsset,
) -> PreparedLfWhitenedObservation:
    """Prepare deterministic observation features without changing detector math."""

    if type(observation) is not LfDetectionObservation:
        raise LfDetectorError(
            "LF whitened detector requires a public-image observation"
        )
    if observation.shape != LF_NULL_WHITENING_LATENT_SHAPE:
        raise LfDetectorError(
            "LF whitened detector requires shape [1,16,64,64]"
        )
    if observation.observation_protocol != OBSERVATION_PROTOCOL:
        raise LfDetectorError("LF observation protocol identity mismatch")
    if _digest(observation.values) != observation.observation_digest:
        raise LfDetectorError("LF observation digest mismatch")
    if type(whitening_asset) is not LfNullWhiteningAsset:
        raise LfDetectorError(
            "LF whitened detector requires a frozen public whitening asset"
        )
    try:
        whitening_asset.validate()
    except LfNullWhiteningAssetError as exc:
        raise LfDetectorError("LF whitening asset validation failed") from exc
    coefficients = _affine_detrended_dct(
        observation.values,
        role="LF observation",
    )
    coefficients = _freeze_coefficients(coefficients)
    return PreparedLfWhitenedObservation(
        coefficients=coefficients,
        coefficients_digest=_coefficient_digest(coefficients),
        observation_digest=observation.observation_digest,
        observation_shape=observation.shape,
        observation_protocol=observation.observation_protocol,
        whitening_asset_digest=whitening_asset.whitening_asset_digest,
    )


def prepare_lf_null_whitened_template(
    detection_key: str | DerivedWrongKeyMaterial,
    whitening_asset: LfNullWhiteningAsset,
    *,
    shape: tuple[int, int, int, int] = LF_NULL_WHITENING_LATENT_SHAPE,
) -> PreparedLfWhitenedTemplate:
    """Prepare deterministic key-template features without persisting them."""

    if shape != LF_NULL_WHITENING_LATENT_SHAPE:
        raise LfDetectorError("prepared LF template shape mismatch")
    if type(whitening_asset) is not LfNullWhiteningAsset:
        raise LfDetectorError(
            "LF whitened detector requires a frozen public whitening asset"
        )
    try:
        whitening_asset.validate()
    except LfNullWhiteningAssetError as exc:
        raise LfDetectorError("LF whitening asset validation failed") from exc
    try:
        carrier = lf_carrier(
            detection_key,
            shape,
            mask_lf=None,
        )
    except LfCarrierError as exc:
        raise LfDetectorError(
            "LF whitened detector template reconstruction failed"
        ) from exc
    coefficients = _affine_detrended_dct(
        carrier.template,
        role="LF template",
    )
    coefficients = _freeze_coefficients(coefficients)
    return PreparedLfWhitenedTemplate(
        coefficients=coefficients,
        coefficients_digest=_coefficient_digest(coefficients),
        template_digest=carrier.template_digest,
        carrier_config_digest=carrier.carrier_config_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        template_shape=shape,
        whitening_asset_digest=whitening_asset.whitening_asset_digest,
    )


def _whitened_cosine(
    observation_coefficients: np.ndarray,
    template_coefficients: np.ndarray,
    asset: LfNullWhiteningAsset,
) -> float:
    observation_whitened: list[float] = []
    template_whitened: list[float] = []
    for channel in range(16):
        for height_frequency in range(64):
            for width_frequency in range(64):
                if height_frequency == 0 and width_frequency == 0:
                    continue
                ring_radius = max(height_frequency, width_frequency)
                band = ring_radius.bit_length() - 1
                weight = asset.weights[channel * 6 + band]
                observed = (
                    weight
                    * float(
                        observation_coefficients[
                            channel,
                            height_frequency,
                            width_frequency,
                        ]
                    )
                )
                template = (
                    weight
                    * float(
                        template_coefficients[
                            channel,
                            height_frequency,
                            width_frequency,
                        ]
                    )
                )
                observation_whitened.append(observed)
                template_whitened.append(template)
    dot = 0.0
    for observed, template in zip(
        observation_whitened,
        template_whitened,
        strict=True,
    ):
        dot += observed * template
    observation_squared_sum = 0.0
    for observed in observation_whitened:
        observation_squared_sum += observed * observed
    template_squared_sum = 0.0
    for template in template_whitened:
        template_squared_sum += template * template
    if (
        not isfinite(dot)
        or not isfinite(observation_squared_sum)
        or not isfinite(template_squared_sum)
    ):
        raise LfDetectorError("LF whitened score accumulators must be finite")
    if observation_squared_sum <= 0.0 or template_squared_sum <= 0.0:
        raise LfDetectorError(
            "LF whitened score requires strictly positive non-DC norms"
        )
    norm_product = observation_squared_sum * template_squared_sum
    if not isfinite(norm_product) or norm_product <= 0.0:
        raise LfDetectorError(
            "LF whitened norm product must be finite and positive"
        )
    score = dot / sqrt(norm_product)
    if not isfinite(score):
        raise LfDetectorError("LF whitened blind score must be finite")
    return score


def lf_null_whitened_matched_detector(
    observation: LfDetectionObservation,
    detection_key: str | DerivedWrongKeyMaterial,
    whitening_asset: LfNullWhiteningAsset | None = None,
    *,
    prepared_observation: PreparedLfWhitenedObservation | None = None,
    prepared_template: PreparedLfWhitenedTemplate | None = None,
) -> LfNullWhitenedDetectionResult:
    """Score a public RGB-to-VAE observation with the frozen whitening asset."""

    if type(observation) is not LfDetectionObservation:
        raise LfDetectorError(
            "LF whitened detector requires a public-image observation"
        )
    if observation.shape != LF_NULL_WHITENING_LATENT_SHAPE:
        raise LfDetectorError(
            "LF whitened detector requires shape [1,16,64,64]"
        )
    if observation.observation_protocol != OBSERVATION_PROTOCOL:
        raise LfDetectorError("LF observation protocol identity mismatch")
    if _digest(observation.values) != observation.observation_digest:
        raise LfDetectorError("LF observation digest mismatch")
    if type(whitening_asset) is not LfNullWhiteningAsset:
        raise LfDetectorError(
            "LF whitened detector requires a frozen public whitening asset"
        )
    try:
        whitening_asset.validate()
    except LfNullWhiteningAssetError as exc:
        raise LfDetectorError("LF whitening asset validation failed") from exc
    try:
        carrier = lf_carrier(
            detection_key,
            observation.shape,
            mask_lf=None,
        )
    except LfCarrierError as exc:
        raise LfDetectorError(
            "LF whitened detector template reconstruction failed"
        ) from exc

    if prepared_observation is None:
        observation_coefficients = _affine_detrended_dct(
            observation.values,
            role="LF observation",
        )
    else:
        if type(prepared_observation) is not PreparedLfWhitenedObservation:
            raise LfDetectorError("prepared LF observation type mismatch")
        prepared_observation.validate()
        if (
            prepared_observation.observation_digest
            != observation.observation_digest
            or prepared_observation.observation_shape != observation.shape
            or prepared_observation.observation_protocol
            != observation.observation_protocol
            or prepared_observation.whitening_asset_digest
            != whitening_asset.whitening_asset_digest
        ):
            raise LfDetectorError("prepared LF observation identity mismatch")
        observation_coefficients = prepared_observation.coefficients
    if prepared_template is None:
        template_coefficients = _affine_detrended_dct(
            carrier.template,
            role="LF template",
        )
    else:
        if type(prepared_template) is not PreparedLfWhitenedTemplate:
            raise LfDetectorError("prepared LF template type mismatch")
        prepared_template.validate()
        if (
            prepared_template.template_digest != carrier.template_digest
            or prepared_template.carrier_config_digest
            != carrier.carrier_config_digest
            or prepared_template.root_key_public_digest
            != carrier.root_key_public_digest
            or prepared_template.key_role != carrier.key_role
            or prepared_template.wrong_key_index != carrier.wrong_key_index
            or prepared_template.template_shape != observation.shape
            or prepared_template.whitening_asset_digest
            != whitening_asset.whitening_asset_digest
        ):
            raise LfDetectorError("prepared LF template identity mismatch")
        template_coefficients = prepared_template.coefficients
    score = _whitened_cosine(
        observation_coefficients,
        template_coefficients,
        whitening_asset,
    )
    candidate_ids = (
        "key_schedule_sha256_counter",
        "lf_low_pass",
        LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    )
    detector_config = {
        "band_identity": LF_NULL_WHITENING_BAND_IDENTITY,
        "candidate_ids": list(candidate_ids),
        "carrier_config_digest": carrier.carrier_config_digest,
        "computation_dtype": "float64",
        "detrend_identity": LF_NULL_WHITENING_DETREND_IDENTITY,
        "input_dtype": "float32",
        "observation_protocol": OBSERVATION_PROTOCOL,
        "score_operator": LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "template_mask": "unmasked",
        "transform_identity": LF_NULL_WHITENING_TRANSFORM_IDENTITY,
        "whitening_asset_digest": whitening_asset.whitening_asset_digest,
    }
    detector_config_digest = sha256(
        stable_json_utf8(detector_config)
    ).hexdigest()
    detector_identity = sha256(
        stable_json_utf8(
            {
                "candidate_ids": list(candidate_ids),
                "detector_config_digest": detector_config_digest,
                "detector_role": "lf_null_whitened_blind_score",
                "whitening_asset_digest": (
                    whitening_asset.whitening_asset_digest
                ),
            }
        )
    ).hexdigest()
    return LfNullWhitenedDetectionResult(
        candidate_id=LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        candidate_ids=candidate_ids,
        lf_score=score,
        detector_identity=detector_identity,
        detector_config_digest=detector_config_digest,
        whitening_asset_digest=whitening_asset.whitening_asset_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        observation_digest=observation.observation_digest,
        template_digest=carrier.template_digest,
    )


def semantic_texture_lf_detector(
    observation: LfDetectionObservation,
    detection_key: str | DerivedWrongKeyMaterial,
    routing_result: SemanticTextureRoutingResult,
    whitening_asset: SemanticTextureLfWhiteningAsset | None,
) -> SemanticTextureLfDetectionResult:
    """Score symmetric ``m_lf`` features using only the dedicated soft-route W."""

    if type(observation) is not LfDetectionObservation:
        raise LfDetectorError(
            "semantic-texture LF detector requires a public-image observation"
        )
    if observation.shape != LF_NULL_WHITENING_LATENT_SHAPE:
        raise LfDetectorError(
            "semantic-texture LF detector requires shape [1,16,64,64]"
        )
    if observation.observation_protocol != OBSERVATION_PROTOCOL or (
        _digest(observation.values) != observation.observation_digest
    ):
        raise LfDetectorError("semantic-texture LF observation identity mismatch")
    if type(whitening_asset) is not SemanticTextureLfWhiteningAsset:
        raise LfDetectorError(
            "semantic-texture LF detector requires its dedicated whitening W"
        )
    try:
        whitening_asset.validate()
    except LfNullWhiteningAssetError as exc:
        raise LfDetectorError(
            "semantic-texture LF whitening asset validation failed"
        ) from exc
    try:
        route = validate_semantic_texture_routing_result(routing_result)
    except ValueError as exc:
        raise LfDetectorError("semantic-texture LF route validation failed") from exc
    if route.latent_shape != observation.shape:
        raise LfDetectorError("semantic-texture LF route shape mismatch")
    try:
        carrier = lf_carrier(
            detection_key,
            observation.shape,
        )
    except LfCarrierError as exc:
        raise LfDetectorError("semantic-texture LF template reconstruction failed") from exc
    if whitening_asset.lf_carrier_config_digest != carrier.carrier_config_digest:
        raise LfDetectorError(
            "semantic-texture LF whitening asset carrier configuration mismatch"
        )
    routed_observation = tuple(
        _float32(value * weight)
        for value, weight in zip(observation.values, route.mask_lf, strict=True)
    )
    routed_template = tuple(
        _float32(value * weight)
        for value, weight in zip(carrier.template, route.mask_lf, strict=True)
    )
    observation_coefficients = _affine_detrended_dct(
        routed_observation,
        role="semantic-texture LF observation",
    )
    template_coefficients = _affine_detrended_dct(
        routed_template,
        role="semantic-texture LF template",
    )
    score = _whitened_cosine(
        observation_coefficients,
        template_coefficients,
        whitening_asset,
    )
    candidate_ids = (
        "key_schedule_sha256_counter",
        "lf_low_pass",
        "routing_semantic_texture_soft",
        SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID,
    )
    config = {
        "band_identity": LF_NULL_WHITENING_BAND_IDENTITY,
        "candidate_ids": list(candidate_ids),
        "candidate_status": SEMANTIC_TEXTURE_CANDIDATE_STATUS,
        "lf_carrier_config_digest": carrier.carrier_config_digest,
        "detrend_identity": LF_NULL_WHITENING_DETREND_IDENTITY,
        "observation_protocol": OBSERVATION_PROTOCOL,
        "route_config_digest": route.route_config_digest,
        "score_operator": SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID,
        "symmetric_route_application": "m_lf_on_observation_and_template_before_detrend",
        "transform_identity": LF_NULL_WHITENING_TRANSFORM_IDENTITY,
        "whitening_asset_digest": whitening_asset.whitening_asset_digest,
    }
    config_digest = sha256(stable_json_utf8(config)).hexdigest()
    identity = sha256(
        stable_json_utf8(
            {
                "candidate_ids": list(candidate_ids),
                "detector_config_digest": config_digest,
                "detector_role": "semantic_texture_soft_lf_blind_score",
                "whitening_asset_digest": whitening_asset.whitening_asset_digest,
            }
        )
    ).hexdigest()
    return SemanticTextureLfDetectionResult(
        candidate_id=SEMANTIC_TEXTURE_LF_WHITENED_CANDIDATE_ID,
        candidate_status=SEMANTIC_TEXTURE_CANDIDATE_STATUS,
        candidate_ids=candidate_ids,
        lf_score=score,
        detector_identity=identity,
        detector_config_digest=config_digest,
        whitening_asset_digest=whitening_asset.whitening_asset_digest,
        root_key_public_digest=carrier.root_key_public_digest,
        key_role=carrier.key_role,
        wrong_key_index=carrier.wrong_key_index,
        observation_digest=observation.observation_digest,
        template_digest=carrier.template_digest,
        route_identity=route.route_identity,
    )
