"""Development-only metrics computed from real method/runtime results.

These functions never fit a formal threshold and never consume caller supplied
metric values.  Every reported number is derived from typed public results
created by the CEG-WM adapter or runtime in the governed runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import exp, isfinite, sqrt
from statistics import median
from typing import Mapping, Sequence

from experiments.protocol.development_exploration import FrozenDevelopmentCrossFitPlan
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
    registered_metric_ids: tuple[str, ...]
    candidate_config_digest: str
    paired_ablation_identity: str
    content_branch_id: str
    geometry_case_id: str
    sufficient_statistics: tuple[tuple[str, float], ...]
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
        if not self.registered_metric_ids or len(set(self.registered_metric_ids)) != len(
            self.registered_metric_ids
        ):
            raise DevelopmentMetricError("registered metric identities are invalid")
        if len(self.candidate_config_digest) != 64 or not self.paired_ablation_identity:
            raise DevelopmentMetricError("metric candidate or paired identity is invalid")
        if not self.content_branch_id or not self.geometry_case_id:
            raise DevelopmentMetricError("metric case variant identity is missing")
        names = [name for name, _ in self.sufficient_statistics]
        if not names or len(names) != len(set(names)):
            raise DevelopmentMetricError("metric value identities are invalid")
        for name, value in self.sufficient_statistics:
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
        "registered_metric_ids": tuple(sorted(metric_values)),
        "candidate_config_digest": "0" * 64,
        "paired_ablation_identity": "unbound_paired_ablation",
        "content_branch_id": "not_applicable",
        "geometry_case_id": "not_applicable",
        "sufficient_statistics": tuple(sorted((name, _finite(value, name)) for name, value in metric_values.items())),
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


def bind_development_metric_observation(
    observation: DevelopmentMetricObservation,
    *,
    registered_metric_ids: Sequence[str],
    candidate_config_digest: str,
    paired_ablation_identity: str,
    content_branch_id: str,
    geometry_case_id: str,
) -> DevelopmentMetricObservation:
    """Bind raw sufficient statistics to the frozen scientific case registry."""

    if type(observation) is not DevelopmentMetricObservation:
        raise DevelopmentMetricError("metric observation exact type is required")
    payload = observation.payload_without_digest()
    payload.update(
        registered_metric_ids=tuple(registered_metric_ids),
        candidate_config_digest=candidate_config_digest,
        paired_ablation_identity=paired_ablation_identity,
        content_branch_id=content_branch_id,
        geometry_case_id=geometry_case_id,
    )
    bound = replace(
        observation,
        registered_metric_ids=tuple(registered_metric_ids),
        candidate_config_digest=candidate_config_digest,
        paired_ablation_identity=paired_ablation_identity,
        content_branch_id=content_branch_id,
        geometry_case_id=geometry_case_id,
        observation_digest=_canonical_digest(payload),
    )
    bound.validate()
    return bound


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
            "key_attribution_separation": float(
                registered_identity_digest == replayed_identity_digest
                and registered_stream_digest != wrong_stream_digest
                and registered_stream_digest != public_noise_digest
            ),
            "domain_collision_count": float(
                3
                - len({registered_stream_digest, wrong_stream_digest, public_noise_digest})
            ),
        },
        result_identity_digests=digests,
    )


def metric_content_router(
    source_cluster_id: str,
    *,
    matched_budget_registered_score: float,
    uniform_control_registered_score: float,
    routing_coverage: float,
    matched_budget_quality_delta: float,
    adaptive_route_identity: str,
    uniform_route_identity: str,
    adaptive_detector_config_digest: str,
    uniform_detector_config_digest: str,
    adaptive_runtime_result_digest: str,
    uniform_runtime_result_digest: str,
) -> DevelopmentMetricObservation:
    if adaptive_detector_config_digest != uniform_detector_config_digest:
        raise DevelopmentMetricError("routing pair detector configuration drifted")
    return _observation(
        responsibility_id="content_router",
        source_cluster_id=source_cluster_id,
        metric_values={
            "matched_budget_incremental_tpr": float(
                _finite(matched_budget_registered_score, "routed score")
                > _finite(uniform_control_registered_score, "uniform score")
            ),
            "routing_coverage": _finite(routing_coverage, "routing coverage"),
            "quality_delta": _finite(matched_budget_quality_delta, "quality delta"),
        },
        result_identity_digests=(
            adaptive_route_identity,
            uniform_route_identity,
            adaptive_detector_config_digest,
            adaptive_runtime_result_digest,
            uniform_runtime_result_digest,
        ),
    )


def _carrier_metric(
    responsibility_id: str,
    source_cluster_id: str,
    *,
    registered_score: float,
    primary_null_score: float,
    quality_delta: float,
    direction_digest: str,
    template_digest: str,
    carrier_config_digest: str,
) -> DevelopmentMetricObservation:
    prefix = "lf" if responsibility_id == "lf_carrier" else "hf"
    return _observation(
        responsibility_id=responsibility_id,
        source_cluster_id=source_cluster_id,
        metric_values={
            f"{prefix}_attribution_tpr": float(
                _finite(registered_score, "registered carrier score")
                > _finite(primary_null_score, "primary null carrier score")
            ),
            f"{prefix}_primary_null_fpr": float(
                _finite(primary_null_score, "primary null carrier score")
                >= _finite(registered_score, "registered carrier score")
            ),
            "quality_delta": _finite(quality_delta, "carrier quality delta"),
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
            "realized_total_relative_l2": realized_relative_l2,
            "matched_budget_quality_delta": clean_watermarked_image_relative_l2,
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
    quality_delta: float,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="qk_geometry_sync",
        source_cluster_id=source_cluster_id,
        metric_values={
            "relation_score_gain": registered_relation_score,
            "wrong_key_relation_margin": registered_relation_score - wrong_key_relation_score,
            "quality_delta": _finite(quality_delta, "QK attack quality delta"),
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
    estimated_translation_x: float,
    estimated_translation_y: float,
    truth_translation_x: float,
    truth_translation_y: float,
) -> DevelopmentMetricObservation:
    estimated_scale = exp(_finite(estimated_log_scale, "estimated log scale"))
    metric_values = {
        "rotation_error": abs(_finite(estimated_rotation_degrees, "estimated rotation") - _finite(truth_rotation_degrees, "truth rotation")),
        "scale_error": abs(estimated_scale - _finite(truth_scale, "truth scale")),
        "translation_error": sqrt(
            (_finite(estimated_translation_x, "estimated translation x") - _finite(truth_translation_x, "truth translation x")) ** 2
            + (_finite(estimated_translation_y, "estimated translation y") - _finite(truth_translation_y, "truth translation y")) ** 2
        ),
        "coverage": estimated_coverage,
        "residual": mean_residual if isfinite(mean_residual) else 1.0e30,
    }
    return _observation(
        responsibility_id="geometric_transform_estimator",
        source_cluster_id=source_cluster_id,
        metric_values=metric_values,
        result_identity_digests=(estimation_identity_digest, search_config_digest),
    )


def metric_geometry_reliability(
    source_cluster_id: str,
    *,
    reliability_accepted: bool,
    is_unreliable_control: bool,
    reliability_identity_digest: str,
    estimation_identity_digest: str,
) -> DevelopmentMetricObservation:
    if type(reliability_accepted) is not bool or type(is_unreliable_control) is not bool:
        raise DevelopmentMetricError("reliability decisions must be booleans")
    return _observation(
        responsibility_id="geometry_reliability",
        source_cluster_id=source_cluster_id,
        metric_values={
            "reliable_accept_rate": float(
                reliability_accepted and not is_unreliable_control
            ),
            "unreliable_reject_rate": float(
                not reliability_accepted and is_unreliable_control
            ),
            "false_reliable_rate": float(
                reliability_accepted and is_unreliable_control
            ),
        },
        result_identity_digests=(
            reliability_identity_digest,
            estimation_identity_digest,
        ),
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
    rectification_quality: float,
) -> DevelopmentMetricObservation:
    return _observation(
        responsibility_id="image_rectifier",
        source_cluster_id=source_cluster_id,
        metric_values={
            "rectification_quality": _finite(rectification_quality, "rectification quality"),
            "same_detector_score_delta": _finite(rectified_content_score, "rectified score") - _finite(attacked_content_score, "attacked score"),
            "valid_support": min(
                _finite(token_crop_support, "token support"),
                _finite(pixel_crop_support, "pixel support"),
            ),
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
            "incremental_tpr": float(
                record.decision_trace.positive_source == "rectified_content"
            ),
            "end_to_end_fpr": float(
                positive and record.key_control_trace.key_role != "registered"
            ),
            "trigger_rate": float(record.geometry_trace.geometry_triggered),
            "false_rescue_rate": max(geometry_positive_violation, 1.0 - same_detector),
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
    plan_digest: str
    execution_intent_authority_digest: str
    input_manifest_digest: str
    candidate_config_digest: str
    paired_ablation_identity: str
    registered_metric_ids: tuple[str, ...]
    evidence_observation_digest: str
    aggregate_digest: str


def cross_fit_development_detection_metrics(
    responsibility_id: str,
    observations: Sequence[DevelopmentMetricObservation],
    *,
    plan: FrozenDevelopmentCrossFitPlan,
) -> DevelopmentCrossFitDetectionAggregate:
    """Replay the exact frozen plan; arbitrary caller fold partitions are forbidden."""

    if not observations or any(type(item) is not DevelopmentMetricObservation for item in observations):
        raise DevelopmentMetricError("cross-fit requires development metric observations")
    if type(plan) is not FrozenDevelopmentCrossFitPlan or plan.validate():
        raise DevelopmentMetricError("cross-fit requires a valid frozen development plan")
    if plan.responsibility_id != responsibility_id:
        raise DevelopmentMetricError("cross-fit plan responsibility drifted")
    for item in observations:
        item.validate()
    if any(item.responsibility_id != responsibility_id for item in observations):
        raise DevelopmentMetricError("cross-fit responsibility drifted")
    candidate_config_digest = observations[0].candidate_config_digest
    paired_ablation_identity = observations[0].paired_ablation_identity
    registered_metric_ids = observations[0].registered_metric_ids
    if any(
        item.candidate_config_digest != candidate_config_digest
        or item.paired_ablation_identity != paired_ablation_identity
        or item.registered_metric_ids != registered_metric_ids
        for item in observations
    ):
        raise DevelopmentMetricError(
            "cross-fit metric registry, candidate, or paired identity drifted"
        )
    primary_branch = {
        "lf_detector": "lf_only",
        "hf_detector": "hf_only",
        "content_detector": "lf_hf_routed_combination",
    }.get(responsibility_id)
    if primary_branch is None:
        raise DevelopmentMetricError("cross-fit responsibility is not a detector")
    selected = tuple(
        item for item in observations if item.content_branch_id == primary_branch
    )
    by_cluster = {item.source_cluster_id: item for item in selected}
    if (
        len(by_cluster) != len(selected)
        or tuple(sorted(by_cluster)) != tuple(sorted(plan.source_cluster_ids))
    ):
        raise DevelopmentMetricError("cross-fit observations do not bind the frozen cluster roster")
    normalized_folds = tuple(
        (fold.fit_source_cluster_ids, fold.recovery_probe_source_cluster_ids)
        for fold in plan.folds
    )
    covered_probes: list[str] = []
    thresholds: list[tuple[int, float]] = []
    null_accepts = registered_accepts = wrong_accepts = 0
    for fold_index, (fit_ids, probe_ids) in enumerate(normalized_folds):
        if not fit_ids or not probe_ids or set(fit_ids) & set(probe_ids):
            raise DevelopmentMetricError("cross-fit fit/probe clusters overlap or are empty")
        if not set((*fit_ids, *probe_ids)).issubset(by_cluster):
            raise DevelopmentMetricError("cross-fit fold references unknown clusters")
        threshold = max(dict(by_cluster[cluster].sufficient_statistics)["primary_null_score"] for cluster in fit_ids)
        thresholds.append((fold_index, threshold))
        covered_probes.extend(probe_ids)
        for cluster in probe_ids:
            values = dict(by_cluster[cluster].sufficient_statistics)
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
        "plan_digest": plan.digest(),
        "execution_intent_authority_digest": (
            plan.expected_execution_intent_authority_digest
        ),
        "input_manifest_digest": plan.input_manifest_digest,
        "candidate_config_digest": candidate_config_digest,
        "paired_ablation_identity": paired_ablation_identity,
        "registered_metric_ids": registered_metric_ids,
        "evidence_observation_digest": _canonical_digest(
            tuple(sorted(item.observation_digest for item in observations))
        ),
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
    expected_metric_ids: Sequence[str] | None = None,
    expected_candidate_config_digest: str | None = None,
    expected_paired_ablation_identity: str | None = None,
    expected_content_branch_ids: Sequence[str] = (),
    expected_geometry_case_ids: Sequence[str] = (),
) -> DevelopmentClusterAggregate:
    if not observations or any(type(item) is not DevelopmentMetricObservation for item in observations):
        raise DevelopmentMetricError("cluster aggregation requires metric observations")
    for item in observations:
        item.validate()
    if any(item.responsibility_id != responsibility_id for item in observations):
        raise DevelopmentMetricError("cluster aggregation responsibility drifted")
    registered = tuple(expected_metric_ids or observations[0].registered_metric_ids)
    candidate_digest = expected_candidate_config_digest or observations[0].candidate_config_digest
    paired_identity = expected_paired_ablation_identity or observations[0].paired_ablation_identity
    if any(
        item.registered_metric_ids != registered
        or item.candidate_config_digest != candidate_digest
        or item.paired_ablation_identity != paired_identity
        for item in observations
    ):
        raise DevelopmentMetricError("cluster metric registry or paired identity drifted")
    branches = tuple(expected_content_branch_ids) or tuple(
        sorted({item.content_branch_id for item in observations})
    )
    geometries = tuple(expected_geometry_case_ids) or tuple(
        sorted({item.geometry_case_id for item in observations})
    )
    expected_variants = {
        (branch, geometry)
        for branch in branches
        for geometry in geometries
    }
    grouped: dict[str, dict[tuple[str, str], DevelopmentMetricObservation]] = {}
    for item in observations:
        key = (item.content_branch_id, item.geometry_case_id)
        cluster = grouped.setdefault(item.source_cluster_id, {})
        if key in cluster:
            raise DevelopmentMetricError("paired cluster variant is duplicated")
        cluster[key] = item
    if len(grouped) < minimum_source_clusters or any(
        set(variants) != expected_variants for variants in grouped.values()
    ):
        raise DevelopmentMetricError("paired cluster aggregate is incomplete")
    metric_names = tuple(name for name, _ in observations[0].sufficient_statistics)
    if any(tuple(name for name, _ in item.sufficient_statistics) != metric_names for item in observations):
        raise DevelopmentMetricError("cluster metric identities drifted")
    values_by_name = {
        name: [dict(item.sufficient_statistics)[name] for item in observations]
        for name in metric_names
    }
    raw_means = {name: sum(values) / len(values) for name, values in values_by_name.items()}
    raw_medians = {name: median(values) for name, values in values_by_name.items()}
    if len(registered) == len(metric_names) and set(registered) == set(metric_names):
        registered_means = {name: raw_means[name] for name in registered}
        registered_medians = {name: raw_medians[name] for name in registered}
    else:
        raise DevelopmentMetricError(
            "per-unit sufficient statistics were not reduced to exact registered metric ids"
        )
    payload = {
        "responsibility_id": responsibility_id,
        "source_cluster_count": len(grouped),
        "metric_medians": tuple((name, registered_medians[name]) for name in registered),
        "metric_means": tuple((name, registered_means[name]) for name in registered),
        "source_cluster_digest": _canonical_digest(tuple(sorted(grouped))),
    }
    return DevelopmentClusterAggregate(**payload, aggregate_digest=_canonical_digest(payload))
