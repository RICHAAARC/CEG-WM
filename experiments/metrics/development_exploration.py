"""Development-only metrics computed from real method/runtime results.

These functions never fit a formal threshold and never consume caller supplied
metric values.  Every reported number is derived from typed public results
created by the CEG-WM adapter or runtime in the governed runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import exp, isfinite, sqrt
from statistics import median
from typing import Mapping, Sequence

from experiments.protocol.internal_records import InternalValidationRecord


METRIC_SCHEMA_VERSION = "ceg_wm_development_metric_observation_v1"
DEVELOPMENT_METRIC_ROLE = "development_exploratory_cluster_level"
DEVELOPMENT_THRESHOLD_ROLE = "development_exploratory"


class DevelopmentMetricError(ValueError):
    """Typed development metric inputs or cluster aggregation are invalid."""


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise DevelopmentMetricError(f"{role} is not finite")
    return float(value)


def _l2(values: Sequence[float]) -> float:
    return sqrt(sum(_finite(value, "vector value") ** 2 for value in values))


@dataclass(frozen=True, slots=True)
class DevelopmentMetricObservation:
    schema_version: str
    metric_role: str
    responsibility_id: str
    source_cluster_id: str
    metric_values: tuple[tuple[str, float], ...]
    result_identity_digests: tuple[str, ...]
    threshold_role: str | None
    threshold_identity: str | None
    threshold_fit_source_cluster_digest: str | None
    observation_digest: str

    def payload_without_digest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("observation_digest")
        return payload

    def validate(self) -> None:
        if self.schema_version != METRIC_SCHEMA_VERSION:
            raise DevelopmentMetricError("development metric schema drifted")
        if self.metric_role != DEVELOPMENT_METRIC_ROLE:
            raise DevelopmentMetricError("development metric role drifted")
        if not self.responsibility_id or not self.source_cluster_id:
            raise DevelopmentMetricError("metric identities are missing")
        names = [name for name, _ in self.metric_values]
        if not names or len(names) != len(set(names)):
            raise DevelopmentMetricError("metric value identities are invalid")
        for name, value in self.metric_values:
            if not name:
                raise DevelopmentMetricError("metric name is empty")
            _finite(value, name)
        if not self.result_identity_digests or any(
            len(value) != 64 for value in self.result_identity_digests
        ):
            raise DevelopmentMetricError("result identity digests are invalid")
        threshold_values = (
            self.threshold_role,
            self.threshold_identity,
            self.threshold_fit_source_cluster_digest,
        )
        if any(value is not None for value in threshold_values):
            if self.threshold_role != DEVELOPMENT_THRESHOLD_ROLE or any(
                type(value) is not str or len(value) != 64
                for value in threshold_values[1:]
            ):
                raise DevelopmentMetricError("development threshold binding is invalid")
        if self.observation_digest != _canonical_digest(self.payload_without_digest()):
            raise DevelopmentMetricError("development metric observation digest drifted")


def _observation(
    *,
    responsibility_id: str,
    source_cluster_id: str,
    metric_values: Mapping[str, float],
    result_identity_digests: Sequence[str],
    threshold_role: str | None = None,
    threshold_identity: str | None = None,
    threshold_fit_source_cluster_digest: str | None = None,
) -> DevelopmentMetricObservation:
    payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metric_role": DEVELOPMENT_METRIC_ROLE,
        "responsibility_id": responsibility_id,
        "source_cluster_id": source_cluster_id,
        "metric_values": tuple(sorted((name, _finite(value, name)) for name, value in metric_values.items())),
        "result_identity_digests": tuple(result_identity_digests),
        "threshold_role": threshold_role,
        "threshold_identity": threshold_identity,
        "threshold_fit_source_cluster_digest": threshold_fit_source_cluster_digest,
    }
    observation = DevelopmentMetricObservation(
        **payload,
        observation_digest=_canonical_digest(payload),
    )
    observation.validate()
    return observation


def metric_key_schedule(
    source_cluster_id: str,
    *,
    registered_identity_digest: str,
    replayed_identity_digest: str,
    registered_stream_digest: str,
    wrong_stream_digest: str,
    public_noise_digest: str,
) -> DevelopmentMetricObservation:
    digests = (
        registered_identity_digest,
        replayed_identity_digest,
        registered_stream_digest,
        wrong_stream_digest,
        public_noise_digest,
    )
    if any(type(value) is not str or len(value) != 64 for value in digests):
        raise DevelopmentMetricError("key stream digests are invalid")
    return _observation(
        responsibility_id="key_schedule",
        source_cluster_id=source_cluster_id,
        metric_values={
            "registered_identity_replay_match": float(registered_identity_digest == replayed_identity_digest),
            "registered_wrong_domain_separated": float(registered_stream_digest != wrong_stream_digest),
            "registered_public_noise_domain_separated": float(registered_stream_digest != public_noise_digest),
            "wrong_public_noise_domain_separated": float(wrong_stream_digest != public_noise_digest),
        },
        result_identity_digests=digests,
    )


def metric_content_router(
    source_cluster_id: str,
    *,
    adaptive_latent_shape: Sequence[int],
    uniform_latent_shape: Sequence[int],
    adaptive_routing_map: Sequence[float],
    uniform_routing_map: Sequence[float],
    adaptive_mean_mask_lf: float,
    adaptive_mean_mask_hf: float,
    uniform_mean_mask_lf: float,
    uniform_mean_mask_hf: float,
    adaptive_route_identity: str,
    uniform_route_identity: str,
) -> DevelopmentMetricObservation:
    if tuple(adaptive_latent_shape) != tuple(uniform_latent_shape):
        raise DevelopmentMetricError("routing metric is not matched-budget")
    if not adaptive_routing_map or len(adaptive_routing_map) != len(uniform_routing_map):
        raise DevelopmentMetricError("routing maps are not aligned")
    mean_absolute_difference = sum(
        abs(_finite(left, "adaptive route") - _finite(right, "uniform route"))
        for left, right in zip(adaptive_routing_map, uniform_routing_map)
    ) / len(adaptive_routing_map)
    return _observation(
        responsibility_id="content_router",
        source_cluster_id=source_cluster_id,
        metric_values={
            "routing_map_mean_absolute_difference": mean_absolute_difference,
            "adaptive_low_frequency_allocation": adaptive_mean_mask_lf,
            "uniform_low_frequency_allocation": uniform_mean_mask_lf,
            "allocation_budget_difference": abs(
                adaptive_mean_mask_lf + adaptive_mean_mask_hf
                - uniform_mean_mask_lf - uniform_mean_mask_hf
            ),
        },
        result_identity_digests=(adaptive_route_identity, uniform_route_identity),
    )


def _carrier_metric(
    responsibility_id: str,
    source_cluster_id: str,
    *,
    direction: Sequence[float],
    template: Sequence[float],
    active_support_count: int,
    direction_digest: str,
    template_digest: str,
    carrier_config_digest: str,
) -> DevelopmentMetricObservation:
    if not direction or len(direction) != len(template):
        raise DevelopmentMetricError("carrier vectors are invalid")
    if type(active_support_count) is not int or not 0 <= active_support_count <= len(direction):
        raise DevelopmentMetricError("carrier support count is invalid")
    support_fraction = active_support_count / len(direction)
    return _observation(
        responsibility_id=responsibility_id,
        source_cluster_id=source_cluster_id,
        metric_values={
            "direction_l2": _l2(direction),
            "template_l2": _l2(template),
            "active_support_fraction": support_fraction,
        },
        result_identity_digests=(direction_digest, template_digest, carrier_config_digest),
    )


def metric_lf_carrier(source_cluster_id: str, **case) -> DevelopmentMetricObservation:
    return _carrier_metric("lf_carrier", source_cluster_id, **case)


def metric_hf_carrier(source_cluster_id: str, **case) -> DevelopmentMetricObservation:
    return _carrier_metric("hf_carrier", source_cluster_id, **case)


def metric_content_embedder(
    source_cluster_id: str,
    *,
    nominal_relative_l2: float,
    realized_relative_l2: float,
    clean_watermarked_image_relative_l2: float,
    embedding_result_identity: str,
    materialization_replay_identity: str,
    paired_base_latent_digest: str,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="content_embedder",
        source_cluster_id=source_cluster_id,
        metric_values={
            "nominal_relative_l2": nominal_relative_l2,
            "realized_relative_l2": realized_relative_l2,
            "actual_budget_excess": max(0.0, realized_relative_l2 - nominal_relative_l2),
            "clean_watermarked_image_relative_l2": clean_watermarked_image_relative_l2,
        },
        result_identity_digests=(
            embedding_result_identity,
            materialization_replay_identity,
            paired_base_latent_digest,
        ),
    )


def _detector_metric(
    responsibility_id: str,
    source_cluster_id: str,
    *,
    registered_score: float,
    wrong_score: float,
    primary_null_score: float,
    detector_config_digest: str,
    registered_observation_digest: str,
    wrong_observation_digest: str,
    primary_null_observation_digest: str,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id=responsibility_id,
        source_cluster_id=source_cluster_id,
        metric_values={
            "registered_score": registered_score,
            "wrong_key_score": wrong_score,
            "primary_null_score": primary_null_score,
            "registered_wrong_margin": registered_score - wrong_score,
            "registered_primary_null_margin": registered_score - primary_null_score,
        },
        result_identity_digests=(
            detector_config_digest,
            registered_observation_digest,
            wrong_observation_digest,
            primary_null_observation_digest,
        ),
    )


def metric_lf_detector(*args, **kwargs) -> DevelopmentMetricObservation:
    return _detector_metric("lf_detector", *args, **kwargs)


def metric_hf_detector(*args, **kwargs) -> DevelopmentMetricObservation:
    return _detector_metric("hf_detector", *args, **kwargs)


def metric_content_detector(
    source_cluster_id: str,
    *,
    candidate_score: float,
    hf_only_score: float,
    wrong_key_score: float,
    primary_null_score: float,
    low_frequency_score: float | None,
    candidate_config_digest: str,
    hf_only_config_digest: str,
    wrong_key_config_digest: str,
    primary_null_config_digest: str,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="content_detector",
        source_cluster_id=source_cluster_id,
        metric_values={
            "candidate_score": candidate_score,
            "hf_only_score": hf_only_score,
            "wrong_key_score": wrong_key_score,
            "primary_null_score": primary_null_score,
            "candidate_minus_hf_only": candidate_score - hf_only_score,
            "lf_score": low_frequency_score if low_frequency_score is not None else 0.0,
        },
        result_identity_digests=(
            candidate_config_digest,
            hf_only_config_digest,
            wrong_key_config_digest,
            primary_null_config_digest,
        ),
    )


def metric_qk_geometry_sync(
    source_cluster_id: str,
    *,
    registered_relation_score: float,
    wrong_key_relation_score: float,
    registered_descriptor_digest: str,
    registered_projection_digest: str,
    wrong_projection_digest: str,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="qk_geometry_sync",
        source_cluster_id=source_cluster_id,
        metric_values={
            "registered_relation_score": registered_relation_score,
            "wrong_key_relation_score": wrong_key_relation_score,
            "registered_wrong_relation_margin": registered_relation_score - wrong_key_relation_score,
        },
        result_identity_digests=(registered_descriptor_digest, registered_projection_digest, wrong_projection_digest),
    )


def metric_geometric_transform_estimator(
    source_cluster_id: str,
    *,
    estimated_log_scale: float,
    estimated_rotation_degrees: float,
    estimated_coverage: float,
    mean_residual: float,
    key_margin: float,
    estimation_identity_digest: str,
    search_config_digest: str,
    truth_crop_fraction: float,
    truth_scale: float,
    truth_rotation_degrees: float,
) -> DevelopmentMetricObservation:
    estimated_scale = exp(_finite(estimated_log_scale, "estimated log scale"))
    estimated_crop = 1.0 - _finite(estimated_coverage, "estimated coverage")
    metric_values = {
        "crop_fraction_absolute_error": abs(estimated_crop - _finite(truth_crop_fraction, "truth crop")),
        "scale_absolute_error": abs(estimated_scale - _finite(truth_scale, "truth scale")),
        "rotation_absolute_error_degrees": abs(_finite(estimated_rotation_degrees, "estimated rotation") - _finite(truth_rotation_degrees, "truth rotation")),
        "coverage": estimated_coverage,
        "mean_residual_finite": float(isfinite(mean_residual)),
        "key_margin": key_margin,
    }
    if isfinite(mean_residual):
        metric_values["mean_residual"] = mean_residual
    return _observation(
        responsibility_id="geometric_transform_estimator",
        source_cluster_id=source_cluster_id,
        metric_values=metric_values,
        result_identity_digests=(estimation_identity_digest, search_config_digest),
    )


def metric_geometry_reliability(
    source_cluster_id: str,
    *,
    reliable_case_accepted: bool,
    unreliable_control_accepted: bool,
    reliable_identity_digest: str,
    unreliable_identity_digest: str,
) -> DevelopmentMetricObservation:
    if type(reliable_case_accepted) is not bool or type(unreliable_control_accepted) is not bool:
        raise DevelopmentMetricError("reliability decisions must be booleans")
    return _observation(
        responsibility_id="geometry_reliability",
        source_cluster_id=source_cluster_id,
        metric_values={
            "reliable_case_accepted": float(reliable_case_accepted),
            "unreliable_control_rejected": float(not unreliable_control_accepted),
            "reliability_separation": float(reliable_case_accepted) - float(unreliable_control_accepted),
        },
        result_identity_digests=(reliable_identity_digest, unreliable_identity_digest),
    )


def metric_image_rectifier(
    source_cluster_id: str,
    *,
    attacked_content_score: float,
    rectified_content_score: float,
    token_crop_support: float,
    pixel_crop_support: float,
    rectified_image_digest: str,
    rectification_config_digest: str,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="image_rectifier",
        source_cluster_id=source_cluster_id,
        metric_values={
            "content_score_recovery_delta": _finite(rectified_content_score, "rectified score") - _finite(attacked_content_score, "attacked score"),
            "token_crop_support": token_crop_support,
            "pixel_crop_support": pixel_crop_support,
        },
        result_identity_digests=(rectified_image_digest, rectification_config_digest),
    )


def metric_conditional_recovery_record(
    source_cluster_id: str,
    record: InternalValidationRecord,
    *,
    threshold_fit_source_cluster_digest: str,
) -> DevelopmentMetricObservation:
    """Compute the joint metric from the runner-owned governed record."""

    if type(record) is not InternalValidationRecord or record.execution_status != "success":
        raise DevelopmentMetricError("joint metric requires a successful InternalValidationRecord")
    raw_score = record.detector_trace.raw_content_score
    if raw_score is None:
        raise DevelopmentMetricError("joint record lacks raw content score")
    rectified_score = record.detector_trace.rectified_content_score
    effective_rectified = raw_score if rectified_score is None else rectified_score
    positive = record.decision_trace.watermark_decision == "positive"
    geometry_positive_violation = float(
        positive and record.decision_trace.positive_source not in {"raw_content", "rectified_content"}
    )
    same_detector = float(
        record.detector_trace.raw_detector_identity
        == record.detector_trace.rectified_detector_identity
        and record.detector_trace.raw_detector_config_digest
        == record.detector_trace.rectified_detector_config_digest
        and record.detector_trace.raw_preprocessing_identity
        == record.detector_trace.rectified_preprocessing_identity
        and record.threshold_trace.raw_threshold_identity
        == record.threshold_trace.rectified_threshold_identity
    )
    return _observation(
        responsibility_id="conditional_recovery_decision",
        source_cluster_id=source_cluster_id,
        metric_values={
            "geometry_triggered": float(record.geometry_trace.geometry_triggered),
            "rectified_content_score_delta": effective_rectified - raw_score,
            "content_positive": float(positive),
            "geometry_direct_positive_violation": geometry_positive_violation,
            "same_detector_binding_preserved": same_detector,
        },
        result_identity_digests=(
            record.record_id,
            record.provenance_trace.candidate_config_digest,
            record.threshold_trace.raw_threshold_identity,
        ),
        threshold_role=DEVELOPMENT_THRESHOLD_ROLE,
        threshold_identity=record.threshold_trace.raw_threshold_identity,
        threshold_fit_source_cluster_digest=threshold_fit_source_cluster_digest,
    )


@dataclass(frozen=True, slots=True)
class DevelopmentClusterAggregate:
    responsibility_id: str
    source_cluster_count: int
    metric_medians: tuple[tuple[str, float], ...]
    metric_means: tuple[tuple[str, float], ...]
    source_cluster_digest: str
    aggregate_digest: str


@dataclass(frozen=True, slots=True)
class DevelopmentCrossFitDetectionAggregate:
    responsibility_id: str
    threshold_role: str
    fold_count: int
    source_cluster_count: int
    fold_thresholds: tuple[tuple[int, float], ...]
    primary_null_false_accept_rate: float
    registered_accept_rate: float
    wrong_key_accept_rate: float
    fold_assignment_digest: str
    aggregate_digest: str


def cross_fit_development_detection_metrics(
    responsibility_id: str,
    observations: Sequence[DevelopmentMetricObservation],
    *,
    folds: Sequence[tuple[Sequence[str], Sequence[str]]],
) -> DevelopmentCrossFitDetectionAggregate:
    """Fit only on each fold's primary-null clusters and score held clusters."""

    if not observations or any(type(item) is not DevelopmentMetricObservation for item in observations):
        raise DevelopmentMetricError("cross-fit requires development metric observations")
    for item in observations:
        item.validate()
    by_cluster = {item.source_cluster_id: item for item in observations}
    if len(by_cluster) != len(observations) or len(by_cluster) < 16:
        raise DevelopmentMetricError("cross-fit cluster coverage is insufficient or duplicated")
    if any(item.responsibility_id != responsibility_id for item in observations):
        raise DevelopmentMetricError("cross-fit responsibility drifted")
    normalized_folds = tuple(
        (tuple(fit_ids), tuple(probe_ids)) for fit_ids, probe_ids in folds
    )
    if len(normalized_folds) < 2:
        raise DevelopmentMetricError("cross-fit requires at least two folds")
    covered_probes: list[str] = []
    thresholds: list[tuple[int, float]] = []
    null_accepts = registered_accepts = wrong_accepts = 0
    for fold_index, (fit_ids, probe_ids) in enumerate(normalized_folds):
        if not fit_ids or not probe_ids or set(fit_ids) & set(probe_ids):
            raise DevelopmentMetricError("cross-fit fit/probe clusters overlap or are empty")
        if not set((*fit_ids, *probe_ids)).issubset(by_cluster):
            raise DevelopmentMetricError("cross-fit fold references unknown clusters")
        threshold = max(dict(by_cluster[cluster].metric_values)["primary_null_score"] for cluster in fit_ids)
        thresholds.append((fold_index, threshold))
        covered_probes.extend(probe_ids)
        for cluster in probe_ids:
            values = dict(by_cluster[cluster].metric_values)
            null_accepts += int(values["primary_null_score"] >= threshold)
            registered_accepts += int(
                values.get("registered_score", values.get("candidate_score")) >= threshold
            )
            wrong_accepts += int(values["wrong_key_score"] >= threshold)
    if sorted(covered_probes) != sorted(by_cluster) or len(covered_probes) != len(set(covered_probes)):
        raise DevelopmentMetricError("cross-fit probes must partition all source clusters")
    count = len(covered_probes)
    payload = {
        "responsibility_id": responsibility_id,
        "threshold_role": DEVELOPMENT_THRESHOLD_ROLE,
        "fold_count": len(normalized_folds),
        "source_cluster_count": count,
        "fold_thresholds": tuple(thresholds),
        "primary_null_false_accept_rate": null_accepts / count,
        "registered_accept_rate": registered_accepts / count,
        "wrong_key_accept_rate": wrong_accepts / count,
        "fold_assignment_digest": _canonical_digest(normalized_folds),
    }
    return DevelopmentCrossFitDetectionAggregate(
        **payload,
        aggregate_digest=_canonical_digest(payload),
    )


def aggregate_development_cluster_metrics(
    responsibility_id: str,
    observations: Sequence[DevelopmentMetricObservation],
    *,
    minimum_source_clusters: int,
) -> DevelopmentClusterAggregate:
    if not observations or any(type(item) is not DevelopmentMetricObservation for item in observations):
        raise DevelopmentMetricError("cluster aggregation requires metric observations")
    for item in observations:
        item.validate()
    if any(item.responsibility_id != responsibility_id for item in observations):
        raise DevelopmentMetricError("cluster aggregation responsibility drifted")
    cluster_ids = [item.source_cluster_id for item in observations]
    if len(cluster_ids) != len(set(cluster_ids)) or len(cluster_ids) < minimum_source_clusters:
        raise DevelopmentMetricError("cluster aggregation coverage is insufficient or duplicated")
    metric_names = tuple(name for name, _ in observations[0].metric_values)
    if any(tuple(name for name, _ in item.metric_values) != metric_names for item in observations):
        raise DevelopmentMetricError("cluster metric identities drifted")
    values_by_name = {
        name: [dict(item.metric_values)[name] for item in observations]
        for name in metric_names
    }
    payload = {
        "responsibility_id": responsibility_id,
        "source_cluster_count": len(cluster_ids),
        "metric_medians": tuple((name, median(values)) for name, values in values_by_name.items()),
        "metric_means": tuple((name, sum(values) / len(values)) for name, values in values_by_name.items()),
        "source_cluster_digest": _canonical_digest(tuple(sorted(cluster_ids))),
    }
    return DevelopmentClusterAggregate(**payload, aggregate_digest=_canonical_digest(payload))
