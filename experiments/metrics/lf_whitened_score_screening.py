"""Frozen LF clean-null whitening fit and threshold-free screening metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import cos, isfinite, sqrt
from struct import pack, unpack
from typing import Sequence

import numpy as np

RING_COUNTS = (3, 12, 48, 192, 768, 3072)
_CANDIDATE_ID = "lf_null_whitened_matched_score"
_ARTIFACT_ROLE = "lf_clean_null_whitening_operator"
_BAND_IDENTITY = "six_dyadic_chebyshev_frequency_rings_without_dc"
_DETREND_IDENTITY = "per_channel_affine_plane_normalized_coordinates"
_FIT_SOURCE_CLUSTER_COUNT = 32
_LATENT_SHAPE = (1, 16, 64, 64)
_OBSERVATION_PROTOCOL = "final_image_vae_posterior_mode"
_REGULARIZATION_RATIO = "0x1.0000000000000p-10"
_TRANSFORM_IDENTITY = "orthonormal_dct_ii"
_WEIGHT_COUNT = 96
_PI = float.fromhex("0x1.921fb54442d18p+1")
_SIZE = 64
_COORDINATES = tuple((2.0 * index - 63.0) / 63.0 for index in range(64))
_COORDINATE_SQUARED_SUM = sum(value * value for value in _COORDINATES)
_BASIS = np.asarray(
    tuple(
        tuple(
            sqrt(1.0 / 64.0)
            if frequency == 0
            else sqrt(2.0 / 64.0)
            * cos(_PI * (coordinate + 0.5) * frequency / 64.0)
            for coordinate in range(64)
        )
        for frequency in range(64)
    ),
    dtype=np.float64,
    order="C",
)


class LfWhitenedScoreMetricError(ValueError):
    """The LF whitening fit or directional screening statistic is invalid."""


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class LfNullWhiteningFitResult:
    """Pure metric output for one frozen public whitening asset payload."""

    canonical_payload: dict[str, object]
    whitening_asset_digest: str

    def validate(self) -> None:
        payload = self.canonical_payload
        expected_keys = {
            "artifact_role",
            "band_identity",
            "candidate_id",
            "detrend_identity",
            "fit_manifest_sha256",
            "fit_source_cluster_count",
            "latent_shape",
            "observation_protocol",
            "regularization_ratio",
            "transform_identity",
            "weights_binary32_be_hex",
        }
        if type(payload) is not dict or set(payload) != expected_keys:
            raise LfWhitenedScoreMetricError(
                "whitening fit payload fields drifted"
            )
        if (
            payload["artifact_role"] != _ARTIFACT_ROLE
            or payload["band_identity"] != _BAND_IDENTITY
            or payload["candidate_id"] != _CANDIDATE_ID
            or payload["detrend_identity"] != _DETREND_IDENTITY
            or payload["fit_source_cluster_count"] != _FIT_SOURCE_CLUSTER_COUNT
            or payload["latent_shape"] != list(_LATENT_SHAPE)
            or payload["observation_protocol"] != _OBSERVATION_PROTOCOL
            or payload["regularization_ratio"] != _REGULARIZATION_RATIO
            or payload["transform_identity"] != _TRANSFORM_IDENTITY
        ):
            raise LfWhitenedScoreMetricError(
                "whitening fit payload identity drifted"
            )
        fit_manifest_sha256 = payload["fit_manifest_sha256"]
        if (
            type(fit_manifest_sha256) is not str
            or len(fit_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fit_manifest_sha256
            )
        ):
            raise LfWhitenedScoreMetricError(
                "whitening fit manifest digest is invalid"
            )
        words = payload["weights_binary32_be_hex"]
        if (
            type(words) is not list
            or len(words) != _WEIGHT_COUNT
            or any(
                type(word) is not str
                or len(word) != 8
                or any(character not in "0123456789abcdef" for character in word)
                for word in words
            )
        ):
            raise LfWhitenedScoreMetricError(
                "whitening fit weight bytes are invalid"
            )
        if self.whitening_asset_digest != _digest(payload):
            raise LfWhitenedScoreMetricError(
                "whitening fit payload digest drifted"
            )


@dataclass(frozen=True, slots=True)
class SemanticTextureLfWhiteningFitResult:
    """Pure metric output for the diagnostic soft-route whitening asset."""

    canonical_payload: dict[str, object]
    whitening_asset_digest: str

    def validate(self) -> None:
        payload = self.canonical_payload
        expected_keys = {
            "artifact_role",
            "band_identity",
            "candidate_id",
            "detrend_identity",
            "fit_manifest_sha256",
            "fit_source_cluster_count",
            "latent_shape",
            "lf_carrier_config_digest",
            "observation_protocol",
            "regularization_ratio",
            "route_candidate_id",
            "transform_identity",
            "weights_binary32_be_hex",
        }
        if type(payload) is not dict or set(payload) != expected_keys:
            raise LfWhitenedScoreMetricError(
                "semantic-texture whitening fit payload fields drifted"
            )
        if (
            payload["artifact_role"]
            != "lf_semantic_texture_soft_clean_null_whitening_operator"
            or payload["candidate_id"]
            != "lf_semantic_texture_soft_whitened_matched_score"
            or payload["route_candidate_id"] != "routing_semantic_texture_soft"
            or payload["band_identity"] != _BAND_IDENTITY
            or payload["detrend_identity"] != _DETREND_IDENTITY
            or payload["fit_source_cluster_count"] != _FIT_SOURCE_CLUSTER_COUNT
            or payload["latent_shape"] != list(_LATENT_SHAPE)
            or payload["observation_protocol"] != _OBSERVATION_PROTOCOL
            or payload["regularization_ratio"] != _REGULARIZATION_RATIO
            or payload["transform_identity"] != _TRANSFORM_IDENTITY
        ):
            raise LfWhitenedScoreMetricError(
                "semantic-texture whitening fit payload identity drifted"
            )
        for field in ("fit_manifest_sha256", "lf_carrier_config_digest"):
            value = payload[field]
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise LfWhitenedScoreMetricError(
                    "semantic-texture whitening digest is invalid"
                )
        words = payload["weights_binary32_be_hex"]
        if (
            type(words) is not list
            or len(words) != _WEIGHT_COUNT
            or any(
                type(word) is not str
                or len(word) != 8
                or any(character not in "0123456789abcdef" for character in word)
                for word in words
            )
        ):
            raise LfWhitenedScoreMetricError(
                "semantic-texture whitening fit weights are invalid"
            )
        if self.whitening_asset_digest != _digest(payload):
            raise LfWhitenedScoreMetricError(
                "semantic-texture whitening fit payload digest drifted"
            )


def _detrended_dct(values: Sequence[float]) -> np.ndarray:
    if len(values) != 16 * 64 * 64:
        raise LfWhitenedScoreMetricError("clean observation shape drifted")
    residual = np.empty((16, 64, 64), dtype=np.float64, order="C")
    denominator = 64.0 * _COORDINATE_SQUARED_SUM
    offset = 0
    for channel in range(16):
        channel_values = values[offset : offset + 4096]
        offset += 4096
        if any(not isfinite(float(value)) for value in channel_values):
            raise LfWhitenedScoreMetricError("clean observation is non-finite")
        constant_sum = 0.0
        for height in range(64):
            for width in range(64):
                constant_sum += float(channel_values[height * 64 + width])
        constant = constant_sum / 4096.0
        height_sum = 0.0
        for height in range(64):
            for width in range(64):
                height_sum += _COORDINATES[height] * float(
                    channel_values[height * 64 + width]
                )
        width_sum = 0.0
        for height in range(64):
            for width in range(64):
                width_sum += _COORDINATES[width] * float(
                    channel_values[height * 64 + width]
                )
        height_slope = height_sum / denominator
        width_slope = width_sum / denominator
        for height in range(64):
            for width in range(64):
                residual[channel, height, width] = (
                    float(channel_values[height * 64 + width])
                    - constant
                    - height_slope * _COORDINATES[height]
                    - width_slope * _COORDINATES[width]
                )
    coefficients = np.einsum(
        "chw,uh,vw->cuv",
        residual,
        _BASIS,
        _BASIS,
        dtype=np.float64,
        order="C",
        casting="no",
        optimize=False,
    )
    if not coefficients.flags.c_contiguous or not np.isfinite(coefficients).all():
        raise LfWhitenedScoreMetricError("clean DCT coefficients are invalid")
    return coefficients


def clean_null_band_energy_sums(values: Sequence[float]) -> tuple[float, ...]:
    """Return the 96 channel-band sufficient statistics for one clean cluster."""

    coefficients = _detrended_dct(values)
    energy: list[float] = []
    for channel in range(16):
        for band in range(6):
            total = 0.0
            for height_frequency in range(64):
                for width_frequency in range(64):
                    if height_frequency == 0 and width_frequency == 0:
                        continue
                    radius = max(height_frequency, width_frequency)
                    if radius.bit_length() - 1 == band:
                        coefficient = float(
                            coefficients[channel, height_frequency, width_frequency]
                        )
                        total += coefficient * coefficient
            if not isfinite(total) or total < 0.0:
                raise LfWhitenedScoreMetricError(
                    "clean null band energy is invalid"
                )
            energy.append(total)
    if len(energy) != _WEIGHT_COUNT:
        raise LfWhitenedScoreMetricError("clean null statistic count drifted")
    return tuple(energy)


def semantic_texture_clean_null_band_energy_sums(
    values: Sequence[float],
    mask_lf: Sequence[float],
) -> tuple[float, ...]:
    """Fit one soft-route W row from the public routed LF observation."""

    values_tuple = tuple(float(value) for value in values)
    mask_tuple = tuple(float(value) for value in mask_lf)
    if (
        len(values_tuple) != 16 * 64 * 64
        or len(mask_tuple) != len(values_tuple)
        or any(not isfinite(value) for value in (*values_tuple, *mask_tuple))
        or any(value < 0.0 or value > 1.0 for value in mask_tuple)
    ):
        raise LfWhitenedScoreMetricError("semantic-texture routed clean observation is invalid")
    def binary32_product(value: float, mask: float) -> float:
        try:
            routed = unpack(">f", pack(">f", value * mask))[0]
        except (OverflowError, ValueError) as exc:
            raise LfWhitenedScoreMetricError(
                "semantic-texture routed clean observation is outside binary32"
            ) from exc
        if not isfinite(routed):
            raise LfWhitenedScoreMetricError(
                "semantic-texture routed clean observation is non-finite"
            )
        return routed

    return clean_null_band_energy_sums(
        tuple(
            binary32_product(value, mask)
            for value, mask in zip(values_tuple, mask_tuple, strict=True)
        )
    )


def fit_lf_null_whitening_asset(
    ordered_energy_sums: Sequence[Sequence[float]],
    *,
    fit_manifest_sha256: str,
) -> LfNullWhiteningFitResult:
    """Fit the unique 16-channel by 6-band public whitening asset."""

    rows = tuple(tuple(float(value) for value in row) for row in ordered_energy_sums)
    if (
        len(rows) != _FIT_SOURCE_CLUSTER_COUNT
        or any(len(row) != _WEIGHT_COUNT for row in rows)
        or any(not isfinite(value) or value < 0.0 for row in rows for value in row)
    ):
        raise LfWhitenedScoreMetricError("clean null fit coverage is invalid")
    variances: list[float] = []
    for channel in range(16):
        for band in range(6):
            total = 0.0
            index = channel * 6 + band
            for row in rows:
                total += row[index]
            variance = total / (32.0 * float(RING_COUNTS[band]))
            if not isfinite(variance) or variance < 0.0:
                raise LfWhitenedScoreMetricError("clean null variance is invalid")
            variances.append(variance)
    global_numerator = 0.0
    for channel in range(16):
        for band in range(6):
            global_numerator += (
                float(RING_COUNTS[band]) * variances[channel * 6 + band]
            )
    global_variance = global_numerator / (
        16.0 * float(sum(RING_COUNTS))
    )
    if not isfinite(global_variance) or global_variance <= 0.0:
        raise LfWhitenedScoreMetricError(
            "global clean null variance must be positive"
        )
    regularization = float.fromhex("0x1.0000000000000p-10")
    words: list[str] = []
    for variance in variances:
        denominator = variance + regularization * global_variance
        if not isfinite(denominator) or denominator <= 0.0:
            raise LfWhitenedScoreMetricError(
                "regularized clean null variance is invalid"
            )
        weight = denominator ** -0.5
        word = pack(">f", weight).hex()
        if not isfinite(float(np.float32(weight))) or float(np.float32(weight)) <= 0:
            raise LfWhitenedScoreMetricError("binary32 whitening weight is invalid")
        words.append(word)
    payload = {
        "artifact_role": _ARTIFACT_ROLE,
        "band_identity": _BAND_IDENTITY,
        "candidate_id": _CANDIDATE_ID,
        "detrend_identity": _DETREND_IDENTITY,
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_source_cluster_count": _FIT_SOURCE_CLUSTER_COUNT,
        "latent_shape": list(_LATENT_SHAPE),
        "observation_protocol": _OBSERVATION_PROTOCOL,
        "regularization_ratio": _REGULARIZATION_RATIO,
        "transform_identity": _TRANSFORM_IDENTITY,
        "weights_binary32_be_hex": words,
    }
    result = LfNullWhiteningFitResult(
        canonical_payload=payload,
        whitening_asset_digest=_digest(payload),
    )
    result.validate()
    return result


def fit_semantic_texture_lf_whitening_asset(
    ordered_energy_sums: Sequence[Sequence[float]],
    *,
    fit_manifest_sha256: str,
    lf_carrier_config_digest: str,
) -> SemanticTextureLfWhiteningFitResult:
    """Fit the distinct soft-route W payload without changing the formula."""

    if (
        type(lf_carrier_config_digest) is not str
        or len(lf_carrier_config_digest) != 64
        or any(character not in "0123456789abcdef" for character in lf_carrier_config_digest)
    ):
        raise LfWhitenedScoreMetricError("semantic-texture LF carrier configuration is invalid")
    base = fit_lf_null_whitening_asset(
        ordered_energy_sums,
        fit_manifest_sha256=fit_manifest_sha256,
    )
    payload = {
        **base.canonical_payload,
        "artifact_role": "lf_semantic_texture_soft_clean_null_whitening_operator",
        "candidate_id": "lf_semantic_texture_soft_whitened_matched_score",
        "route_candidate_id": "routing_semantic_texture_soft",
        "lf_carrier_config_digest": lf_carrier_config_digest,
    }
    result = SemanticTextureLfWhiteningFitResult(
        canonical_payload=payload,
        whitening_asset_digest=_digest(payload),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class LfWhitenedScreeningObservation:
    cluster_ordinal: int
    raw_registered_score: float
    raw_primary_null_score: float
    raw_wrong_key_scores: tuple[float, ...]
    whitened_registered_score: float
    whitened_primary_null_score: float
    whitened_wrong_key_scores: tuple[float, ...]
    whitened_registered_minus_primary_null: float
    whitened_registered_minus_max_wrong: float
    raw_registered_minus_max_wrong: float
    raw_to_whitened_wrong_margin_improvement: float
    whitening_asset_digest: str
    raw_detector_config_digest: str
    whitened_detector_config_digest: str
    observation_identity: str

    def validate(self) -> None:
        if type(self.cluster_ordinal) is not int or not 0 <= self.cluster_ordinal < 8:
            raise LfWhitenedScoreMetricError("screening cluster ordinal is invalid")
        if len(self.raw_wrong_key_scores) != 4 or len(self.whitened_wrong_key_scores) != 4:
            raise LfWhitenedScoreMetricError("screening requires four wrong keys")
        values = (
            self.raw_registered_score,
            self.raw_primary_null_score,
            *self.raw_wrong_key_scores,
            self.whitened_registered_score,
            self.whitened_primary_null_score,
            *self.whitened_wrong_key_scores,
            self.whitened_registered_minus_primary_null,
            self.whitened_registered_minus_max_wrong,
            self.raw_registered_minus_max_wrong,
            self.raw_to_whitened_wrong_margin_improvement,
        )
        if any(not isfinite(float(value)) for value in values):
            raise LfWhitenedScoreMetricError("screening score is non-finite")
        expected_null = self.whitened_registered_score - self.whitened_primary_null_score
        expected_whitened_wrong = self.whitened_registered_score - max(self.whitened_wrong_key_scores)
        expected_raw_wrong = self.raw_registered_score - max(self.raw_wrong_key_scores)
        if (
            self.whitened_registered_minus_primary_null != expected_null
            or self.whitened_registered_minus_max_wrong != expected_whitened_wrong
            or self.raw_registered_minus_max_wrong != expected_raw_wrong
            or self.raw_to_whitened_wrong_margin_improvement
            != expected_whitened_wrong - expected_raw_wrong
        ):
            raise LfWhitenedScoreMetricError("screening paired margin drifted")
        if any(type(value) is not str or len(value) != 64 for value in (
            self.whitening_asset_digest,
            self.raw_detector_config_digest,
            self.whitened_detector_config_digest,
            self.observation_identity,
        )):
            raise LfWhitenedScoreMetricError("screening identity is invalid")
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise LfWhitenedScoreMetricError("screening observation identity drifted")


def create_lf_whitened_screening_observation(
    *,
    cluster_ordinal: int,
    raw_registered_score: float,
    raw_primary_null_score: float,
    raw_wrong_key_scores: Sequence[float],
    whitened_registered_score: float,
    whitened_primary_null_score: float,
    whitened_wrong_key_scores: Sequence[float],
    whitening_asset_digest: str,
    raw_detector_config_digest: str,
    whitened_detector_config_digest: str,
) -> LfWhitenedScreeningObservation:
    raw_wrong = tuple(float(value) for value in raw_wrong_key_scores)
    whitened_wrong = tuple(float(value) for value in whitened_wrong_key_scores)
    payload = {
        "cluster_ordinal": cluster_ordinal,
        "raw_registered_score": float(raw_registered_score),
        "raw_primary_null_score": float(raw_primary_null_score),
        "raw_wrong_key_scores": raw_wrong,
        "whitened_registered_score": float(whitened_registered_score),
        "whitened_primary_null_score": float(whitened_primary_null_score),
        "whitened_wrong_key_scores": whitened_wrong,
        "whitened_registered_minus_primary_null": float(whitened_registered_score - whitened_primary_null_score),
        "whitened_registered_minus_max_wrong": float(whitened_registered_score - max(whitened_wrong)),
        "raw_registered_minus_max_wrong": float(raw_registered_score - max(raw_wrong)),
        "raw_to_whitened_wrong_margin_improvement": float(
            (whitened_registered_score - max(whitened_wrong))
            - (raw_registered_score - max(raw_wrong))
        ),
        "whitening_asset_digest": whitening_asset_digest,
        "raw_detector_config_digest": raw_detector_config_digest,
        "whitened_detector_config_digest": whitened_detector_config_digest,
    }
    observation = LfWhitenedScreeningObservation(
        **payload, observation_identity=_digest(payload)
    )
    observation.validate()
    return observation


@dataclass(frozen=True, slots=True)
class LfWhitenedScreeningDecision:
    cluster_count: int
    registered_primary_null_pass_count: int
    registered_max_wrong_pass_count: int
    positive_raw_to_whitened_improvement_count: int
    mean_raw_to_whitened_improvement: float
    integrity_failure_count: int
    margin_floor: float
    allow_request_for_lf_whitened_directional_validation: bool
    decision_identity: str

    def validate(self) -> None:
        if self.cluster_count != 8 or self.margin_floor != float.fromhex("0x1.0000000000000p-10"):
            raise LfWhitenedScoreMetricError("screening decision boundary drifted")
        if any(type(value) is not int or not 0 <= value <= 8 for value in (
            self.registered_primary_null_pass_count,
            self.registered_max_wrong_pass_count,
            self.positive_raw_to_whitened_improvement_count,
            self.integrity_failure_count,
        )):
            raise LfWhitenedScoreMetricError("screening decision count is invalid")
        if not isfinite(self.mean_raw_to_whitened_improvement):
            raise LfWhitenedScoreMetricError("screening improvement mean is invalid")
        expected = (
            self.registered_primary_null_pass_count >= 7
            and self.registered_max_wrong_pass_count >= 7
            and self.positive_raw_to_whitened_improvement_count >= 6
            and self.mean_raw_to_whitened_improvement > 0.0
            and self.integrity_failure_count == 0
        )
        if self.allow_request_for_lf_whitened_directional_validation is not expected:
            raise LfWhitenedScoreMetricError("screening decision drifted")
        payload = asdict(self)
        identity = payload.pop("decision_identity")
        if identity != _digest(payload):
            raise LfWhitenedScoreMetricError("screening decision identity drifted")


def evaluate_lf_whitened_screening(
    observations: Sequence[LfWhitenedScreeningObservation],
    *,
    integrity_failure_count: int,
    margin_floor: float,
) -> LfWhitenedScreeningDecision:
    values = tuple(observations)
    ordinals = tuple(item.cluster_ordinal for item in values)
    if (
        type(integrity_failure_count) is not int
        or integrity_failure_count < 0
        or len(values) + integrity_failure_count != 8
        or len(set(ordinals)) != len(ordinals)
        or tuple(sorted(ordinals)) != ordinals
    ):
        raise LfWhitenedScoreMetricError(
            "screening terminal coverage is incomplete"
        )
    for item in values:
        item.validate()
    improvements = tuple(item.raw_to_whitened_wrong_margin_improvement for item in values)
    payload = {
        "cluster_count": 8,
        "registered_primary_null_pass_count": sum(item.whitened_registered_minus_primary_null > margin_floor for item in values),
        "registered_max_wrong_pass_count": sum(item.whitened_registered_minus_max_wrong > margin_floor for item in values),
        "positive_raw_to_whitened_improvement_count": sum(value > 0.0 for value in improvements),
        "mean_raw_to_whitened_improvement": sum(improvements) / 8.0,
        "integrity_failure_count": integrity_failure_count,
        "margin_floor": margin_floor,
    }
    expected = (
        payload["registered_primary_null_pass_count"] >= 7
        and payload["registered_max_wrong_pass_count"] >= 7
        and payload["positive_raw_to_whitened_improvement_count"] >= 6
        and payload["mean_raw_to_whitened_improvement"] > 0.0
        and integrity_failure_count == 0
    )
    decision = LfWhitenedScreeningDecision(
        **payload,
        allow_request_for_lf_whitened_directional_validation=expected,
        decision_identity=_digest({**payload, "allow_request_for_lf_whitened_directional_validation": expected}),
    )
    decision.validate()
    return decision


__all__ = [
    "LfNullWhiteningFitResult", "LfWhitenedScoreMetricError",
    "LfWhitenedScreeningDecision", "LfWhitenedScreeningObservation",
    "clean_null_band_energy_sums", "create_lf_whitened_screening_observation",
    "evaluate_lf_whitened_screening", "fit_lf_null_whitening_asset",
    "fit_semantic_texture_lf_whitening_asset",
    "semantic_texture_clean_null_band_energy_sums",
    "SemanticTextureLfWhiteningFitResult",
]
