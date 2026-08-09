"""Development-only LF signal transport summaries without threshold fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence

from experiments.protocol.lf_transmission_diagnostic import SIGNAL_POSITIONS


class LfTransmissionMetricError(ValueError):
    """A directional LF transport statistic is invalid."""


def diagnostic_latent_template_projection(
    values: Sequence[float], template: Sequence[float]
) -> float:
    """Centered cosine used only to locate latent-path signal loss."""

    observed = tuple(float(value) for value in values)
    direction = tuple(float(value) for value in template)
    if (
        not observed
        or len(observed) != len(direction)
        or any(not isfinite(value) for value in (*observed, *direction))
    ):
        raise LfTransmissionMetricError("diagnostic projection input is invalid")
    observed_mean = sum(observed) / len(observed)
    direction_mean = sum(direction) / len(direction)
    centered_observed = tuple(value - observed_mean for value in observed)
    centered_direction = tuple(value - direction_mean for value in direction)
    observed_norm = sum(value * value for value in centered_observed) ** 0.5
    direction_norm = sum(value * value for value in centered_direction) ** 0.5
    if observed_norm == 0.0 or direction_norm == 0.0:
        raise LfTransmissionMetricError("diagnostic projection has zero norm")
    score = sum(
        observed_value * direction_value
        for observed_value, direction_value in zip(
            centered_observed, centered_direction, strict=True
        )
    ) / (observed_norm * direction_norm)
    if not isfinite(score):
        raise LfTransmissionMetricError("diagnostic projection is non-finite")
    return float(score)


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
class LfSignalPositionObservation:
    position_id: str
    statistic_role: str
    registered_score: float
    wrong_key_score: float
    primary_null_score: float
    registered_minus_wrong_key: float
    registered_minus_primary_null: float
    registered_observation_digest: str
    primary_null_observation_digest: str
    registered_statistic_identity: str
    wrong_key_statistic_identity: str
    primary_null_statistic_identity: str
    registered_template_digest: str
    wrong_key_template_digest: str
    primary_null_template_digest: str
    registered_root_key_public_digest: str
    wrong_key_root_key_public_digest: str
    primary_null_root_key_public_digest: str
    registered_key_role: str
    wrong_key_key_role: str
    primary_null_key_role: str
    registered_wrong_key_index: int | None
    wrong_key_index: int | None
    primary_null_wrong_key_index: int | None
    observation_identity: str

    def validate(self) -> None:
        if self.position_id not in SIGNAL_POSITIONS:
            raise LfTransmissionMetricError("signal position is not frozen")
        expected_role = (
            "formal_lf_detector_operation"
            if self.position_id == "rgb_vae_reencoded"
            else "diagnostic_latent_template_projection"
        )
        if self.statistic_role != expected_role:
            raise LfTransmissionMetricError("signal statistic role drifted")
        values = (
            self.registered_score,
            self.wrong_key_score,
            self.primary_null_score,
            self.registered_minus_wrong_key,
            self.registered_minus_primary_null,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values
        ):
            raise LfTransmissionMetricError("signal score is non-finite")
        if self.registered_minus_wrong_key != (
            self.registered_score - self.wrong_key_score
        ) or self.registered_minus_primary_null != (
            self.registered_score - self.primary_null_score
        ):
            raise LfTransmissionMetricError("paired score margin drifted")
        identities = (
            self.registered_observation_digest,
            self.primary_null_observation_digest,
            self.registered_statistic_identity,
            self.wrong_key_statistic_identity,
            self.primary_null_statistic_identity,
            self.registered_template_digest,
            self.wrong_key_template_digest,
            self.primary_null_template_digest,
            self.registered_root_key_public_digest,
            self.wrong_key_root_key_public_digest,
            self.primary_null_root_key_public_digest,
            self.registered_key_role,
            self.wrong_key_key_role,
            self.primary_null_key_role,
        )
        if any(type(value) is not str or not value for value in identities):
            raise LfTransmissionMetricError("signal identity is missing")
        if (
            self.registered_key_role != "registered"
            or self.wrong_key_key_role != "wrong"
            or self.primary_null_key_role != "registered"
            or self.registered_wrong_key_index is not None
            or self.wrong_key_index != 0
            or self.primary_null_wrong_key_index is not None
            or self.registered_root_key_public_digest
            != self.wrong_key_root_key_public_digest
            or self.registered_root_key_public_digest
            != self.primary_null_root_key_public_digest
            or self.registered_template_digest == self.wrong_key_template_digest
            or self.registered_template_digest != self.primary_null_template_digest
        ):
            raise LfTransmissionMetricError("key-specific template binding drifted")
        payload = asdict(self)
        identity = payload.pop("observation_identity")
        if identity != _digest(payload):
            raise LfTransmissionMetricError("signal observation identity drifted")


def create_lf_signal_position_observation(
    *,
    position_id: str,
    statistic_role: str,
    registered_score: float,
    wrong_key_score: float,
    primary_null_score: float,
    registered_observation_digest: str,
    primary_null_observation_digest: str,
    registered_statistic_identity: str,
    wrong_key_statistic_identity: str,
    primary_null_statistic_identity: str,
    registered_template_digest: str,
    wrong_key_template_digest: str,
    primary_null_template_digest: str,
    registered_root_key_public_digest: str,
    wrong_key_root_key_public_digest: str,
    primary_null_root_key_public_digest: str,
    registered_key_role: str,
    wrong_key_key_role: str,
    primary_null_key_role: str,
    registered_wrong_key_index: int | None,
    wrong_key_index: int | None,
    primary_null_wrong_key_index: int | None,
) -> LfSignalPositionObservation:
    payload = {
        "position_id": position_id,
        "statistic_role": statistic_role,
        "registered_score": float(registered_score),
        "wrong_key_score": float(wrong_key_score),
        "primary_null_score": float(primary_null_score),
        "registered_minus_wrong_key": float(registered_score - wrong_key_score),
        "registered_minus_primary_null": float(
            registered_score - primary_null_score
        ),
        "registered_observation_digest": registered_observation_digest,
        "primary_null_observation_digest": primary_null_observation_digest,
        "registered_statistic_identity": registered_statistic_identity,
        "wrong_key_statistic_identity": wrong_key_statistic_identity,
        "primary_null_statistic_identity": primary_null_statistic_identity,
        "registered_template_digest": registered_template_digest,
        "wrong_key_template_digest": wrong_key_template_digest,
        "primary_null_template_digest": primary_null_template_digest,
        "registered_root_key_public_digest": registered_root_key_public_digest,
        "wrong_key_root_key_public_digest": wrong_key_root_key_public_digest,
        "primary_null_root_key_public_digest": primary_null_root_key_public_digest,
        "registered_key_role": registered_key_role,
        "wrong_key_key_role": wrong_key_key_role,
        "primary_null_key_role": primary_null_key_role,
        "registered_wrong_key_index": registered_wrong_key_index,
        "wrong_key_index": wrong_key_index,
        "primary_null_wrong_key_index": primary_null_wrong_key_index,
    }
    observation = LfSignalPositionObservation(
        **payload,
        observation_identity=_digest(payload),
    )
    observation.validate()
    return observation


@dataclass(frozen=True, slots=True)
class LfTransmissionDirectionalDecision:
    cluster_count: int
    registered_minus_wrong_positive_count: int
    registered_minus_null_positive_count: int
    budget_integrity_nonfinite_failure_count: int
    allow_request_for_lf_directional_validation: bool
    decision_identity: str

    def validate(self) -> None:
        if self.cluster_count != 8:
            raise LfTransmissionMetricError("decision requires exactly eight clusters")
        for count in (
            self.registered_minus_wrong_positive_count,
            self.registered_minus_null_positive_count,
            self.budget_integrity_nonfinite_failure_count,
        ):
            if type(count) is not int or not 0 <= count <= self.cluster_count:
                raise LfTransmissionMetricError("decision count is invalid")
        expected = (
            self.registered_minus_wrong_positive_count >= 7
            and self.registered_minus_null_positive_count >= 7
            and self.budget_integrity_nonfinite_failure_count == 0
        )
        if self.allow_request_for_lf_directional_validation is not expected:
            raise LfTransmissionMetricError("directional decision drifted")
        payload = asdict(self)
        identity = payload.pop("decision_identity")
        if identity != _digest(payload):
            raise LfTransmissionMetricError("decision identity drifted")


def evaluate_lf_transmission_direction(
    final_position_observations: Sequence[LfSignalPositionObservation],
    *,
    budget_integrity_nonfinite_failure_count: int,
) -> LfTransmissionDirectionalDecision:
    observations = tuple(final_position_observations)
    if (
        len(observations) + budget_integrity_nonfinite_failure_count != 8
        or any(
            type(item) is not LfSignalPositionObservation
            for item in observations
        )
    ):
        raise LfTransmissionMetricError("directional decision coverage is incomplete")
    for item in observations:
        item.validate()
        if item.position_id != "rgb_vae_reencoded":
            raise LfTransmissionMetricError("decision uses only final re-encoded position")
    if len({item.observation_identity for item in observations}) != len(
        observations
    ):
        raise LfTransmissionMetricError(
            "directional decision observations are duplicated"
        )
    payload = {
        "cluster_count": 8,
        "registered_minus_wrong_positive_count": sum(
            item.registered_minus_wrong_key > 0.0 for item in observations
        ),
        "registered_minus_null_positive_count": sum(
            item.registered_minus_primary_null > 0.0 for item in observations
        ),
        "budget_integrity_nonfinite_failure_count": (
            budget_integrity_nonfinite_failure_count
        ),
    }
    payload["allow_request_for_lf_directional_validation"] = (
        payload["registered_minus_wrong_positive_count"] >= 7
        and payload["registered_minus_null_positive_count"] >= 7
        and budget_integrity_nonfinite_failure_count == 0
    )
    result = LfTransmissionDirectionalDecision(
        **payload,
        decision_identity=_digest(payload),
    )
    result.validate()
    return result


__all__ = [
    "LfSignalPositionObservation",
    "LfTransmissionDirectionalDecision",
    "LfTransmissionMetricError",
    "create_lf_signal_position_observation",
    "diagnostic_latent_template_projection",
    "evaluate_lf_transmission_direction",
]
