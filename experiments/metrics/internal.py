"""Fail-closed per-case and aggregate metrics for internal validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import exp, isfinite, lgamma, log, log1p, nextafter, sqrt
from pathlib import Path
import re
from typing import Sequence

from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    INTERNAL_VALIDATION_SPLITS,
)


DEFAULT_COMPONENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "internal_execution_components.json"
)
FORBIDDEN_SPLIT = "held_out_evaluation"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DETECTION_KEY_ROLES = (
    "registered_positive",
    "unwatermarked_primary_null",
    "wrong_key",
)
REQUIRED_METRIC_IDS = (
    "fixed_fpr_detection",
    "wrong_key_attribution",
    "matched_budget_quality",
    "routing_gain_non_degradation",
    "lf_hf_complementarity",
    "transform_error",
    "reliability_accept_reject",
    "rectification_same_detector_delta",
    "rescue_global_fpr_safety",
)
REQUIRED_METRIC_SPLIT_BINDINGS = (
    (
        "fixed_fpr_detection",
        ("content_threshold_fit", "end_to_end_check"),
    ),
    ("wrong_key_attribution", ("end_to_end_check",)),
    (
        "matched_budget_quality",
        (
            "candidate_selection",
            "untouched_confirmation",
            "end_to_end_check",
        ),
    ),
    (
        "routing_gain_non_degradation",
        ("candidate_selection", "untouched_confirmation"),
    ),
    (
        "lf_hf_complementarity",
        ("candidate_selection", "untouched_confirmation"),
    ),
    ("transform_error", ("reliability_fit", "end_to_end_check")),
    (
        "reliability_accept_reject",
        ("reliability_fit", "end_to_end_check"),
    ),
    (
        "rectification_same_detector_delta",
        ("rescue_threshold_fit", "end_to_end_check"),
    ),
    (
        "rescue_global_fpr_safety",
        ("rescue_threshold_fit", "end_to_end_check"),
    ),
)


class InternalMetricError(ValueError):
    """Metric inputs, identities, or aggregation units failed closed."""


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _finite(value: object, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise InternalMetricError(f"{role} must be a finite number")
    return float(value)


def _validate_case_identity(
    analysis_unit_identity: AnalysisUnitIdentity,
    split: str,
) -> None:
    if type(analysis_unit_identity) is not AnalysisUnitIdentity:
        raise InternalMetricError("metric case requires AnalysisUnitIdentity")
    violations = analysis_unit_identity.validate()
    if violations:
        raise InternalMetricError(
            f"analysis unit identity invalid: {','.join(violations)}"
        )
    if split not in INTERNAL_VALIDATION_SPLITS:
        raise InternalMetricError("metric split is not registered")
    if split == FORBIDDEN_SPLIT:
        raise PermissionError("metrics_forbid_held_out_evaluation_access")


def _require_nonempty_identity(value: object, role: str) -> str:
    if type(value) is not str or not value:
        raise InternalMetricError(f"{role} must be a non-empty string")
    return value


def _require_digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise InternalMetricError(
            f"{role} must be a lowercase SHA-256 digest"
        )
    return value


def _require_cases(cases: Sequence[object]) -> None:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence) or not cases:
        raise InternalMetricError("metric aggregation requires a non-empty sequence")


def _ensure_unique_units(cases: Sequence[object]) -> None:
    keys = []
    for case in cases:
        unit = getattr(case, "analysis_unit_identity", None)
        role = getattr(case, "key_role", None)
        keys.append(
            (
                unit.unit_id,
                unit.case_id,
                unit.source_cluster_id,
                role,
            )
        )
    if len(keys) != len(set(keys)):
        raise InternalMetricError(
            "metric aggregation contains duplicate unit/case/source-cluster roles"
        )


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    log_terms = tuple(
        lgamma(trials + 1)
        - lgamma(index + 1)
        - lgamma(trials - index + 1)
        + index * log(probability)
        + (trials - index) * log1p(-probability)
        for index in range(successes + 1)
    )
    maximum = max(log_terms)
    return exp(maximum) * sum(exp(value - maximum) for value in log_terms)


def _binomial_upper_confidence_bound(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or type(successes) is not int
        or type(trials) is not int
        or trials <= 0
        or successes < 0
        or successes > trials
    ):
        raise InternalMetricError("binomial count inputs are invalid")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise InternalMetricError("confidence_level must be in (0,1)")
    if successes == trials:
        return 1.0
    tail_probability = 1.0 - confidence
    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > tail_probability:
            lower = midpoint
        else:
            upper = midpoint
    return upper


@dataclass(frozen=True, slots=True)
class MetricSplitBinding:
    metric_id: str
    allowed_splits: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_identity(self.metric_id, "metric_id")
        if (
            type(self.allowed_splits) is not tuple
            or not self.allowed_splits
            or any(
                type(split) is not str
                or split not in INTERNAL_VALIDATION_SPLITS
                or split == FORBIDDEN_SPLIT
                for split in self.allowed_splits
            )
            or len(self.allowed_splits) != len(set(self.allowed_splits))
        ):
            raise InternalMetricError(
                "metric split binding contains invalid allowed_splits"
            )


@dataclass(frozen=True, slots=True)
class MetricRegistry:
    schema_version: str
    registry_version: str
    analysis_unit: str
    forbidden_split: str
    metric_ids: tuple[str, ...]
    metric_split_bindings: tuple[MetricSplitBinding, ...]
    registry_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.schema_version != "ceg_wm_internal_execution_components_v1"
            or self.registry_version != "ceg_wm_internal_metric_registry_v1"
            or self.analysis_unit
            != "analysis_unit_identity_case_id_source_cluster_id"
            or self.forbidden_split != FORBIDDEN_SPLIT
            or self.metric_ids != REQUIRED_METRIC_IDS
            or any(
                type(binding) is not MetricSplitBinding
                for binding in self.metric_split_bindings
            )
        ):
            raise InternalMetricError("metric registry semantics drifted")
        split_bindings = tuple(
            (binding.metric_id, binding.allowed_splits)
            for binding in self.metric_split_bindings
        )
        if split_bindings != REQUIRED_METRIC_SPLIT_BINDINGS:
            raise InternalMetricError(
                "metric split bindings drifted from the canonical registry"
            )
        object.__setattr__(
            self,
            "registry_digest",
            _canonical_digest(
                {
                    "analysis_unit": self.analysis_unit,
                    "forbidden_split": self.forbidden_split,
                    "metric_ids": list(self.metric_ids),
                    "metric_split_bindings": [
                        {
                            "allowed_splits": list(binding.allowed_splits),
                            "metric_id": binding.metric_id,
                        }
                        for binding in self.metric_split_bindings
                    ],
                    "registry_version": self.registry_version,
                    "schema_version": self.schema_version,
                }
            ),
        )


def load_metric_registry(
    path: str | Path = DEFAULT_COMPONENT_CONFIG_PATH,
) -> MetricRegistry:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if type(document) is not dict or set(document) != {
        "attack_registry",
        "method_adapter",
        "metric_registry",
        "schema_version",
    }:
        raise InternalMetricError(
            "execution component configuration fields drifted"
        )
    raw = document.get("metric_registry")
    if type(raw) is not dict or set(raw) != {
        "analysis_unit",
        "forbidden_split",
        "metric_ids",
        "metric_split_bindings",
        "registry_version",
    }:
        raise InternalMetricError("metric_registry configuration missing")
    try:
        split_bindings_raw = raw["metric_split_bindings"]
        if (
            type(split_bindings_raw) is not list
            or any(
                type(item) is not dict
                or set(item) != {"allowed_splits", "metric_id"}
                for item in split_bindings_raw
            )
        ):
            raise InternalMetricError(
                "metric split binding fields drifted"
            )
        return MetricRegistry(
            schema_version=document["schema_version"],
            registry_version=raw["registry_version"],
            analysis_unit=raw["analysis_unit"],
            forbidden_split=raw["forbidden_split"],
            metric_ids=tuple(raw["metric_ids"]),
            metric_split_bindings=tuple(
                MetricSplitBinding(
                    metric_id=item["metric_id"],
                    allowed_splits=tuple(item["allowed_splits"]),
                )
                for item in split_bindings_raw
            ),
        )
    except (KeyError, TypeError) as exc:
        raise InternalMetricError("metric registry is incomplete") from exc


def _require_uniform_metric_split(
    cases: Sequence[object],
    *,
    registry: MetricRegistry,
    metric_ids: tuple[str, ...],
    required_split: str | None = None,
) -> str:
    if type(registry) is not MetricRegistry:
        raise InternalMetricError("registry must be MetricRegistry")
    splits = {
        getattr(case, "split", None)
        for case in cases
    }
    if len(splits) != 1:
        raise InternalMetricError(
            "metric aggregation cannot mix split identities"
        )
    split = next(iter(splits))
    if type(split) is not str:
        raise InternalMetricError("metric aggregation split is invalid")
    binding_map = {
        binding.metric_id: frozenset(binding.allowed_splits)
        for binding in registry.metric_split_bindings
    }
    try:
        allowed = set.intersection(
            *(set(binding_map[metric_id]) for metric_id in metric_ids)
        )
    except KeyError as exc:
        raise InternalMetricError(
            "metric split binding is absent from the registry"
        ) from exc
    if split not in allowed:
        raise InternalMetricError(
            f"{'/'.join(metric_ids)} does not allow split {split}"
        )
    if required_split is not None and split != required_split:
        raise InternalMetricError(
            f"{'/'.join(metric_ids)} requires split {required_split}"
        )
    return split


@dataclass(frozen=True, slots=True)
class DetectionMetricCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    detector_identity: str
    key_role: str
    score: float

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        _require_nonempty_identity(self.detector_identity, "detector_identity")
        if self.key_role not in DETECTION_KEY_ROLES:
            raise InternalMetricError("detection key_role is invalid")
        object.__setattr__(self, "score", _finite(self.score, "detection score"))


@dataclass(frozen=True, slots=True)
class FixedFprThresholdResult:
    split: str
    detector_identity: str
    target_fpr: float
    threshold: float
    false_positive_count: int
    primary_null_count: int
    empirical_fpr: float
    fpr_upper_confidence_bound: float
    confidence_level: float
    source_cluster_digest: str
    calibration_case_digest: str
    threshold_identity: str
    metric_registry_digest: str

    def __post_init__(self) -> None:
        _validate_fixed_fpr_threshold_result(self)


def _fixed_fpr_threshold_identity(
    *,
    split: str,
    detector_identity: str,
    target_fpr: float,
    threshold: float,
    false_positive_count: int,
    primary_null_count: int,
    empirical_fpr: float,
    fpr_upper_confidence_bound: float,
    confidence_level: float,
    source_cluster_digest: str,
    calibration_case_digest: str,
    metric_registry_digest: str,
) -> str:
    return _canonical_digest(
        {
            "confidence_level": confidence_level.hex(),
            "calibration_case_digest": calibration_case_digest,
            "detector_identity": detector_identity,
            "empirical_fpr": empirical_fpr.hex(),
            "false_positive_count": false_positive_count,
            "fpr_upper_confidence_bound": (
                fpr_upper_confidence_bound.hex()
            ),
            "metric_registry_digest": metric_registry_digest,
            "primary_null_count": primary_null_count,
            "source_cluster_digest": source_cluster_digest,
            "split": split,
            "target_fpr": target_fpr.hex(),
            "threshold": threshold.hex(),
        }
    )


def _validate_fixed_fpr_threshold_result(
    result: object,
) -> FixedFprThresholdResult:
    if type(result) is not FixedFprThresholdResult:
        raise InternalMetricError(
            "threshold must be FixedFprThresholdResult"
        )
    if result.split != "content_threshold_fit":
        raise InternalMetricError(
            "fixed-FPR threshold split identity is invalid"
        )
    _require_nonempty_identity(
        result.detector_identity,
        "detector_identity",
    )
    target_fpr = _finite(result.target_fpr, "target_fpr")
    threshold = _finite(result.threshold, "threshold")
    empirical_fpr = _finite(result.empirical_fpr, "empirical_fpr")
    upper_bound = _finite(
        result.fpr_upper_confidence_bound,
        "fpr_upper_confidence_bound",
    )
    confidence_level = _finite(
        result.confidence_level,
        "confidence_level",
    )
    if not 0.0 < target_fpr < 1.0:
        raise InternalMetricError("target_fpr must be in (0,1)")
    if not 0.0 <= empirical_fpr <= target_fpr:
        raise InternalMetricError(
            "empirical_fpr must be in [0,target_fpr]"
        )
    if confidence_level != 0.95:
        raise InternalMetricError(
            "fixed-FPR confidence_level must equal 0.95"
        )
    if (
        type(result.false_positive_count) is not int
        or type(result.primary_null_count) is not int
        or result.primary_null_count <= 0
        or result.false_positive_count < 0
        or result.false_positive_count > result.primary_null_count
    ):
        raise InternalMetricError(
            "fixed-FPR counts must be consistent integers"
        )
    expected_empirical = (
        result.false_positive_count / result.primary_null_count
    )
    if empirical_fpr != expected_empirical:
        raise InternalMetricError(
            "empirical_fpr does not match fixed-FPR counts"
        )
    expected_upper = _binomial_upper_confidence_bound(
        result.false_positive_count,
        result.primary_null_count,
        confidence_level=confidence_level,
    )
    if upper_bound != expected_upper or not empirical_fpr <= upper_bound <= 1.0:
        raise InternalMetricError(
            "fixed-FPR confidence bound does not match counts"
        )
    source_cluster_digest = _require_digest(
        result.source_cluster_digest,
        "source_cluster_digest",
    )
    calibration_case_digest = _require_digest(
        result.calibration_case_digest,
        "calibration_case_digest",
    )
    metric_registry_digest = _require_digest(
        result.metric_registry_digest,
        "metric_registry_digest",
    )
    threshold_identity = _require_digest(
        result.threshold_identity,
        "threshold_identity",
    )
    expected_identity = _fixed_fpr_threshold_identity(
        split=result.split,
        detector_identity=result.detector_identity,
        target_fpr=target_fpr,
        threshold=threshold,
        false_positive_count=result.false_positive_count,
        primary_null_count=result.primary_null_count,
        empirical_fpr=empirical_fpr,
        fpr_upper_confidence_bound=upper_bound,
        confidence_level=confidence_level,
        source_cluster_digest=source_cluster_digest,
        calibration_case_digest=calibration_case_digest,
        metric_registry_digest=metric_registry_digest,
    )
    if threshold_identity != expected_identity:
        raise InternalMetricError(
            "fixed-FPR threshold identity mismatch"
        )
    object.__setattr__(result, "target_fpr", target_fpr)
    object.__setattr__(result, "threshold", threshold)
    object.__setattr__(result, "empirical_fpr", empirical_fpr)
    object.__setattr__(
        result,
        "fpr_upper_confidence_bound",
        upper_bound,
    )
    object.__setattr__(
        result,
        "confidence_level",
        confidence_level,
    )
    return result


def fit_fixed_fpr_threshold(
    primary_null_cases: Sequence[DetectionMetricCase],
    *,
    target_fpr: float,
    registry: MetricRegistry,
) -> FixedFprThresholdResult:
    """Fit the lowest finite threshold meeting the calibration null FPR."""

    _require_cases(primary_null_cases)
    if type(registry) is not MetricRegistry:
        raise InternalMetricError("registry must be MetricRegistry")
    alpha = _finite(target_fpr, "target_fpr")
    if not 0.0 < alpha < 1.0:
        raise InternalMetricError("target_fpr must be in (0,1)")
    if any(type(case) is not DetectionMetricCase for case in primary_null_cases):
        raise InternalMetricError("threshold inputs must be DetectionMetricCase")
    _ensure_unique_units(primary_null_cases)
    split = _require_uniform_metric_split(
        primary_null_cases,
        registry=registry,
        metric_ids=("fixed_fpr_detection",),
        required_split="content_threshold_fit",
    )
    if any(
        case.split != "content_threshold_fit"
        or case.key_role != "unwatermarked_primary_null"
        for case in primary_null_cases
    ):
        raise InternalMetricError(
            "threshold fitting requires content_threshold_fit primary nulls"
        )
    detector_identities = {case.detector_identity for case in primary_null_cases}
    if len(detector_identities) != 1:
        raise InternalMetricError("threshold detector identity mismatch")
    scores = tuple(case.score for case in primary_null_cases)
    candidates = {
        nextafter(max(scores), float("inf")),
        *scores,
    }
    eligible = []
    for candidate in candidates:
        false_positives = sum(score >= candidate for score in scores)
        empirical_fpr = false_positives / len(scores)
        if empirical_fpr <= alpha and isfinite(candidate):
            eligible.append((candidate, false_positives, empirical_fpr))
    if not eligible:
        raise InternalMetricError("no finite fixed-FPR threshold is available")
    threshold, false_positive_count, empirical_fpr = min(
        eligible,
        key=lambda item: item[0],
    )
    cluster_ids = sorted(
        case.analysis_unit_identity.source_cluster_id
        for case in primary_null_cases
    )
    cluster_digest = _canonical_digest(cluster_ids)
    ordered_calibration_cases = sorted(
        primary_null_cases,
        key=lambda case: (
            case.analysis_unit_identity.unit_id,
            case.analysis_unit_identity.case_id,
            case.analysis_unit_identity.source_cluster_id,
            case.key_role,
        ),
    )
    calibration_case_digest = _canonical_digest(
        [
            {
                "case_id": case.analysis_unit_identity.case_id,
                "key_role": case.key_role,
                "score": case.score.hex(),
                "source_cluster_id": (
                    case.analysis_unit_identity.source_cluster_id
                ),
                "split": case.split,
                "unit_id": case.analysis_unit_identity.unit_id,
            }
            for case in ordered_calibration_cases
        ]
    )
    detector_identity = next(iter(detector_identities))
    confidence_level = 0.95
    upper_bound = _binomial_upper_confidence_bound(
        false_positive_count,
        len(scores),
        confidence_level=confidence_level,
    )
    threshold_identity = _fixed_fpr_threshold_identity(
        split=split,
        detector_identity=detector_identity,
        target_fpr=alpha,
        threshold=threshold,
        false_positive_count=false_positive_count,
        primary_null_count=len(scores),
        empirical_fpr=empirical_fpr,
        fpr_upper_confidence_bound=upper_bound,
        confidence_level=confidence_level,
        source_cluster_digest=cluster_digest,
        calibration_case_digest=calibration_case_digest,
        metric_registry_digest=registry.registry_digest,
    )
    return FixedFprThresholdResult(
        split=split,
        detector_identity=detector_identity,
        target_fpr=alpha,
        threshold=threshold,
        false_positive_count=false_positive_count,
        primary_null_count=len(scores),
        empirical_fpr=empirical_fpr,
        fpr_upper_confidence_bound=upper_bound,
        confidence_level=confidence_level,
        source_cluster_digest=cluster_digest,
        calibration_case_digest=calibration_case_digest,
        threshold_identity=threshold_identity,
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class DetectionCaseDecision:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    key_role: str
    score: float
    positive: bool


@dataclass(frozen=True, slots=True)
class DetectionAggregate:
    split: str
    decisions: tuple[DetectionCaseDecision, ...]
    registered_tpr: float
    primary_null_fpr: float
    primary_null_fpr_upper_confidence_bound: float
    confidence_level: float
    wrong_key_positive_rate: float
    registered_positive_count: int
    primary_null_count: int
    wrong_key_count: int
    threshold_identity: str
    metric_registry_digest: str


def evaluate_detection_at_threshold(
    cases: Sequence[DetectionMetricCase],
    threshold: FixedFprThresholdResult,
    *,
    registry: MetricRegistry,
) -> DetectionAggregate:
    _require_cases(cases)
    if (
        type(threshold) is not FixedFprThresholdResult
        or type(registry) is not MetricRegistry
    ):
        raise InternalMetricError("detection evaluation identities are invalid")
    _validate_fixed_fpr_threshold_result(threshold)
    if threshold.metric_registry_digest != registry.registry_digest:
        raise InternalMetricError("threshold metric registry identity mismatch")
    if any(type(case) is not DetectionMetricCase for case in cases):
        raise InternalMetricError("evaluation inputs must be DetectionMetricCase")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=(
            "fixed_fpr_detection",
            "wrong_key_attribution",
        ),
        required_split="end_to_end_check",
    )
    if any(case.detector_identity != threshold.detector_identity for case in cases):
        raise InternalMetricError("evaluation detector identity mismatch")
    grouped = {
        role: [case for case in cases if case.key_role == role]
        for role in DETECTION_KEY_ROLES
    }
    if any(not grouped[role] for role in DETECTION_KEY_ROLES):
        raise InternalMetricError("detection evaluation requires all three key roles")
    decisions = tuple(
        DetectionCaseDecision(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            key_role=case.key_role,
            score=case.score,
            positive=case.score >= threshold.threshold,
        )
        for case in cases
    )
    rates = {}
    for role, role_cases in grouped.items():
        rates[role] = (
            sum(case.score >= threshold.threshold for case in role_cases)
            / len(role_cases)
        )
    return DetectionAggregate(
        split=split,
        decisions=decisions,
        registered_tpr=rates["registered_positive"],
        primary_null_fpr=rates["unwatermarked_primary_null"],
        primary_null_fpr_upper_confidence_bound=(
            _binomial_upper_confidence_bound(
                sum(
                    case.score >= threshold.threshold
                    for case in grouped["unwatermarked_primary_null"]
                ),
                len(grouped["unwatermarked_primary_null"]),
            )
        ),
        confidence_level=threshold.confidence_level,
        wrong_key_positive_rate=rates["wrong_key"],
        registered_positive_count=len(grouped["registered_positive"]),
        primary_null_count=len(grouped["unwatermarked_primary_null"]),
        wrong_key_count=len(grouped["wrong_key"]),
        threshold_identity=threshold.threshold_identity,
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class QualityMetricCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    condition_identity: str
    budget_identity: str
    reference_values: tuple[float, ...]
    candidate_values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        _require_nonempty_identity(self.condition_identity, "condition_identity")
        _require_nonempty_identity(self.budget_identity, "budget_identity")
        if not self.reference_values or len(self.reference_values) != len(
            self.candidate_values
        ):
            raise InternalMetricError(
                "quality vectors must be non-empty and shape matched"
            )
        reference = tuple(
            _finite(value, "quality reference value")
            for value in self.reference_values
        )
        candidate = tuple(
            _finite(value, "quality candidate value")
            for value in self.candidate_values
        )
        object.__setattr__(self, "reference_values", reference)
        object.__setattr__(self, "candidate_values", candidate)


@dataclass(frozen=True, slots=True)
class QualityCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    condition_identity: str
    budget_identity: str
    relative_l2: float
    mean_squared_error: float


def compute_quality_case(case: QualityMetricCase) -> QualityCaseResult:
    if type(case) is not QualityMetricCase:
        raise InternalMetricError("quality input must be QualityMetricCase")
    differences = tuple(
        candidate - reference
        for reference, candidate in zip(
            case.reference_values,
            case.candidate_values,
            strict=True,
        )
    )
    reference_norm = sqrt(sum(value * value for value in case.reference_values))
    if reference_norm == 0.0:
        raise InternalMetricError("quality relative L2 reference has zero norm")
    difference_norm = sqrt(sum(value * value for value in differences))
    relative_l2 = difference_norm / reference_norm
    mse = sum(value * value for value in differences) / len(differences)
    if not isfinite(relative_l2) or not isfinite(mse):
        raise InternalMetricError("quality result is non-finite")
    return QualityCaseResult(
        unit_id=case.analysis_unit_identity.unit_id,
        case_id=case.analysis_unit_identity.case_id,
        source_cluster_id=case.analysis_unit_identity.source_cluster_id,
        split=case.split,
        condition_identity=case.condition_identity,
        budget_identity=case.budget_identity,
        relative_l2=relative_l2,
        mean_squared_error=mse,
    )


@dataclass(frozen=True, slots=True)
class MatchedBudgetQualityAggregate:
    split: str
    cases: tuple[QualityCaseResult, ...]
    mean_relative_l2: float
    mean_squared_error: float
    budget_identity: str
    metric_registry_digest: str


def aggregate_matched_budget_quality(
    cases: Sequence[QualityMetricCase],
    *,
    registry: MetricRegistry,
) -> MatchedBudgetQualityAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry:
        raise InternalMetricError("registry must be MetricRegistry")
    if any(type(case) is not QualityMetricCase for case in cases):
        raise InternalMetricError("quality cases have invalid types")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("matched_budget_quality",),
    )
    budget_identities = {case.budget_identity for case in cases}
    if len(budget_identities) != 1:
        raise InternalMetricError("matched-budget aggregation identity mismatch")
    results = tuple(compute_quality_case(case) for case in cases)
    return MatchedBudgetQualityAggregate(
        split=split,
        cases=results,
        mean_relative_l2=sum(case.relative_l2 for case in results) / len(results),
        mean_squared_error=(
            sum(case.mean_squared_error for case in results) / len(results)
        ),
        budget_identity=next(iter(budget_identities)),
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class RoutingPairCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    routed_positive: bool
    uniform_control_positive: bool
    routed_quality_mse: float
    uniform_control_quality_mse: float
    routed_budget_identity: str
    uniform_control_budget_identity: str

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        if type(self.routed_positive) is not bool or type(
            self.uniform_control_positive
        ) is not bool:
            raise InternalMetricError("routing decisions must be boolean")
        object.__setattr__(
            self,
            "routed_quality_mse",
            _finite(self.routed_quality_mse, "routed quality MSE"),
        )
        object.__setattr__(
            self,
            "uniform_control_quality_mse",
            _finite(
                self.uniform_control_quality_mse,
                "uniform-control quality MSE",
            ),
        )
        if (
            self.routed_budget_identity != self.uniform_control_budget_identity
            or not self.routed_budget_identity
        ):
            raise InternalMetricError("routing pair is not matched-budget")


@dataclass(frozen=True, slots=True)
class RoutingGainAggregate:
    split: str
    cases: tuple[RoutingCaseResult, ...]
    per_case_detection_gain: tuple[int, ...]
    per_case_quality_non_degradation: tuple[float, ...]
    mean_detection_gain: float
    mean_quality_non_degradation: float
    metric_registry_digest: str


@dataclass(frozen=True, slots=True)
class RoutingCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    detection_gain: int
    quality_non_degradation: float


def aggregate_routing_gain(
    cases: Sequence[RoutingPairCase],
    *,
    registry: MetricRegistry,
) -> RoutingGainAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not RoutingPairCase for case in cases
    ):
        raise InternalMetricError("routing aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("routing_gain_non_degradation",),
    )
    results = tuple(
        RoutingCaseResult(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            detection_gain=(
                int(case.routed_positive)
                - int(case.uniform_control_positive)
            ),
            quality_non_degradation=(
                case.uniform_control_quality_mse
                - case.routed_quality_mse
            ),
        )
        for case in cases
    )
    detection = tuple(case.detection_gain for case in results)
    quality = tuple(case.quality_non_degradation for case in results)
    return RoutingGainAggregate(
        split=split,
        cases=results,
        per_case_detection_gain=detection,
        per_case_quality_non_degradation=quality,
        mean_detection_gain=sum(detection) / len(detection),
        mean_quality_non_degradation=sum(quality) / len(quality),
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class BranchOutcomeCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    key_role: str
    hf_positive: bool
    lf_positive: bool
    combined_positive: bool

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        if self.key_role not in {"registered_positive", "wrong_key"}:
            raise InternalMetricError("branch outcome key_role is invalid")
        if any(
            type(value) is not bool
            for value in (
                self.hf_positive,
                self.lf_positive,
                self.combined_positive,
            )
        ):
            raise InternalMetricError("branch outcomes must be boolean")


@dataclass(frozen=True, slots=True)
class BranchComplementarityAggregate:
    split: str
    cases: tuple[BranchCaseResult, ...]
    registered_count: int
    wrong_key_count: int
    lf_complements_hf_count: int
    combined_gain_over_hf_count: int
    combined_regression_from_hf_count: int
    wrong_key_combined_positive_rate: float
    metric_registry_digest: str


@dataclass(frozen=True, slots=True)
class BranchCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    key_role: str
    hf_positive: bool
    lf_positive: bool
    combined_positive: bool
    lf_complements_hf: bool
    combined_gain_over_hf: bool
    combined_regression_from_hf: bool


def aggregate_branch_complementarity(
    cases: Sequence[BranchOutcomeCase],
    *,
    registry: MetricRegistry,
) -> BranchComplementarityAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not BranchOutcomeCase for case in cases
    ):
        raise InternalMetricError("branch aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("lf_hf_complementarity",),
    )
    registered = [
        case for case in cases if case.key_role == "registered_positive"
    ]
    wrong = [case for case in cases if case.key_role == "wrong_key"]
    if not registered or not wrong:
        raise InternalMetricError(
            "branch complementarity requires registered and wrong-key cases"
        )
    results = tuple(
        BranchCaseResult(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            key_role=case.key_role,
            hf_positive=case.hf_positive,
            lf_positive=case.lf_positive,
            combined_positive=case.combined_positive,
            lf_complements_hf=(
                case.key_role == "registered_positive"
                and not case.hf_positive
                and case.lf_positive
            ),
            combined_gain_over_hf=(
                case.key_role == "registered_positive"
                and not case.hf_positive
                and case.combined_positive
            ),
            combined_regression_from_hf=(
                case.key_role == "registered_positive"
                and case.hf_positive
                and not case.combined_positive
            ),
        )
        for case in cases
    )
    return BranchComplementarityAggregate(
        split=split,
        cases=results,
        registered_count=len(registered),
        wrong_key_count=len(wrong),
        lf_complements_hf_count=sum(
            not case.hf_positive and case.lf_positive for case in registered
        ),
        combined_gain_over_hf_count=sum(
            not case.hf_positive and case.combined_positive
            for case in registered
        ),
        combined_regression_from_hf_count=sum(
            case.hf_positive and not case.combined_positive
            for case in registered
        ),
        wrong_key_combined_positive_rate=(
            sum(case.combined_positive for case in wrong) / len(wrong)
        ),
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class TransformMetricCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    expected_rotation_degrees: float
    estimated_rotation_degrees: float
    expected_scale: float
    estimated_scale: float
    expected_translation_x: float
    estimated_translation_x: float
    expected_translation_y: float
    estimated_translation_y: float
    coverage: float
    mean_residual: float

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        for name in (
            "expected_rotation_degrees",
            "estimated_rotation_degrees",
            "expected_scale",
            "estimated_scale",
            "expected_translation_x",
            "estimated_translation_x",
            "expected_translation_y",
            "estimated_translation_y",
            "coverage",
            "mean_residual",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.expected_scale <= 0.0 or self.estimated_scale <= 0.0:
            raise InternalMetricError("transform scales must be positive")
        if not 0.0 <= self.coverage <= 1.0 or self.mean_residual < 0.0:
            raise InternalMetricError("transform coverage or residual is invalid")


@dataclass(frozen=True, slots=True)
class TransformCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    rotation_absolute_error: float
    scale_absolute_error: float
    translation_euclidean_error: float
    coverage: float
    mean_residual: float


@dataclass(frozen=True, slots=True)
class TransformErrorAggregate:
    split: str
    cases: tuple[TransformCaseResult, ...]
    mean_rotation_absolute_error: float
    mean_scale_absolute_error: float
    mean_translation_euclidean_error: float
    mean_coverage: float
    mean_residual: float
    metric_registry_digest: str


def aggregate_transform_error(
    cases: Sequence[TransformMetricCase],
    *,
    registry: MetricRegistry,
) -> TransformErrorAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not TransformMetricCase for case in cases
    ):
        raise InternalMetricError("transform aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("transform_error",),
    )
    results = []
    for case in cases:
        rotation_delta = (
            (case.estimated_rotation_degrees - case.expected_rotation_degrees + 180.0)
            % 360.0
        ) - 180.0
        results.append(
            TransformCaseResult(
                unit_id=case.analysis_unit_identity.unit_id,
                case_id=case.analysis_unit_identity.case_id,
                source_cluster_id=case.analysis_unit_identity.source_cluster_id,
                split=case.split,
                rotation_absolute_error=abs(rotation_delta),
                scale_absolute_error=abs(
                    case.estimated_scale - case.expected_scale
                ),
                translation_euclidean_error=sqrt(
                    (
                        case.estimated_translation_x
                        - case.expected_translation_x
                    )
                    ** 2
                    + (
                        case.estimated_translation_y
                        - case.expected_translation_y
                    )
                    ** 2
                ),
                coverage=case.coverage,
                mean_residual=case.mean_residual,
            )
        )
    result_tuple = tuple(results)
    count = len(result_tuple)
    return TransformErrorAggregate(
        split=split,
        cases=result_tuple,
        mean_rotation_absolute_error=sum(
            case.rotation_absolute_error for case in result_tuple
        )
        / count,
        mean_scale_absolute_error=sum(
            case.scale_absolute_error for case in result_tuple
        )
        / count,
        mean_translation_euclidean_error=sum(
            case.translation_euclidean_error for case in result_tuple
        )
        / count,
        mean_coverage=sum(case.coverage for case in result_tuple) / count,
        mean_residual=sum(case.mean_residual for case in result_tuple) / count,
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class ReliabilityMetricCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    expected_recoverable: bool
    reliable: bool

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        if type(self.expected_recoverable) is not bool or type(
            self.reliable
        ) is not bool:
            raise InternalMetricError("reliability labels must be boolean")


@dataclass(frozen=True, slots=True)
class ReliabilityAggregate:
    split: str
    cases: tuple[ReliabilityCaseResult, ...]
    recoverable_accept_rate: float
    unrecoverable_reject_rate: float
    false_reliable_rate: float
    recoverable_count: int
    unrecoverable_count: int
    metric_registry_digest: str


@dataclass(frozen=True, slots=True)
class ReliabilityCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    expected_recoverable: bool
    reliable: bool


def aggregate_reliability(
    cases: Sequence[ReliabilityMetricCase],
    *,
    registry: MetricRegistry,
) -> ReliabilityAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not ReliabilityMetricCase for case in cases
    ):
        raise InternalMetricError("reliability aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("reliability_accept_reject",),
    )
    recoverable = [case for case in cases if case.expected_recoverable]
    unrecoverable = [case for case in cases if not case.expected_recoverable]
    if not recoverable or not unrecoverable:
        raise InternalMetricError(
            "reliability aggregation requires both expected classes"
        )
    accept_rate = sum(case.reliable for case in recoverable) / len(recoverable)
    false_reliable = sum(case.reliable for case in unrecoverable) / len(
        unrecoverable
    )
    results = tuple(
        ReliabilityCaseResult(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            expected_recoverable=case.expected_recoverable,
            reliable=case.reliable,
        )
        for case in cases
    )
    return ReliabilityAggregate(
        split=split,
        cases=results,
        recoverable_accept_rate=accept_rate,
        unrecoverable_reject_rate=1.0 - false_reliable,
        false_reliable_rate=false_reliable,
        recoverable_count=len(recoverable),
        unrecoverable_count=len(unrecoverable),
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class RectificationMetricCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    raw_detector_identity: str
    rectified_detector_identity: str
    raw_threshold_identity: str
    rectified_threshold_identity: str
    raw_score: float
    rectified_score: float

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        for name in (
            "raw_detector_identity",
            "rectified_detector_identity",
            "raw_threshold_identity",
            "rectified_threshold_identity",
        ):
            _require_nonempty_identity(getattr(self, name), name)
        if self.raw_detector_identity != self.rectified_detector_identity:
            raise InternalMetricError("rectification detector identity mismatch")
        if self.raw_threshold_identity != self.rectified_threshold_identity:
            raise InternalMetricError("rectification threshold identity mismatch")
        object.__setattr__(self, "raw_score", _finite(self.raw_score, "raw_score"))
        object.__setattr__(
            self,
            "rectified_score",
            _finite(self.rectified_score, "rectified_score"),
        )


@dataclass(frozen=True, slots=True)
class RectificationDeltaAggregate:
    split: str
    cases: tuple[RectificationCaseResult, ...]
    per_case_score_delta: tuple[float, ...]
    mean_score_delta: float
    improved_fraction: float
    detector_identity: str
    threshold_identity: str
    metric_registry_digest: str


@dataclass(frozen=True, slots=True)
class RectificationCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    score_delta: float


def aggregate_rectification_delta(
    cases: Sequence[RectificationMetricCase],
    *,
    registry: MetricRegistry,
) -> RectificationDeltaAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not RectificationMetricCase for case in cases
    ):
        raise InternalMetricError("rectification aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("rectification_same_detector_delta",),
    )
    detector_identities = {case.raw_detector_identity for case in cases}
    threshold_identities = {case.raw_threshold_identity for case in cases}
    if len(detector_identities) != 1 or len(threshold_identities) != 1:
        raise InternalMetricError(
            "rectification aggregate identity mismatch across cases"
        )
    results = tuple(
        RectificationCaseResult(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            score_delta=case.rectified_score - case.raw_score,
        )
        for case in cases
    )
    deltas = tuple(case.score_delta for case in results)
    return RectificationDeltaAggregate(
        split=split,
        cases=results,
        per_case_score_delta=deltas,
        mean_score_delta=sum(deltas) / len(deltas),
        improved_fraction=sum(delta > 0.0 for delta in deltas) / len(deltas),
        detector_identity=next(iter(detector_identities)),
        threshold_identity=next(iter(threshold_identities)),
        metric_registry_digest=registry.registry_digest,
    )


@dataclass(frozen=True, slots=True)
class RescueSafetyCase:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    raw_detector_identity: str
    rectified_detector_identity: str
    raw_threshold_identity: str
    rectified_threshold_identity: str
    raw_positive: bool
    rescue_triggered: bool
    rectified_positive: bool
    watermark_decision_positive: bool

    def __post_init__(self) -> None:
        _validate_case_identity(self.analysis_unit_identity, self.split)
        for name in (
            "raw_detector_identity",
            "rectified_detector_identity",
            "raw_threshold_identity",
            "rectified_threshold_identity",
        ):
            _require_nonempty_identity(getattr(self, name), name)
        if self.raw_detector_identity != self.rectified_detector_identity:
            raise InternalMetricError("rescue detector identity mismatch")
        if self.raw_threshold_identity != self.rectified_threshold_identity:
            raise InternalMetricError("rescue threshold identity mismatch")
        if any(
            type(value) is not bool
            for value in (
                self.raw_positive,
                self.rescue_triggered,
                self.rectified_positive,
                self.watermark_decision_positive,
            )
        ):
            raise InternalMetricError("rescue decisions must be boolean")
        if self.raw_positive and self.rescue_triggered:
            raise InternalMetricError(
                "raw positive must not trigger rescue"
            )
        if self.rectified_positive and not self.rescue_triggered:
            raise InternalMetricError(
                "rectified positive requires an actual rescue trigger"
            )
        expected_watermark_decision_positive = self.raw_positive or (
            self.rescue_triggered and self.rectified_positive
        )
        if (
            self.watermark_decision_positive
            != expected_watermark_decision_positive
        ):
            raise InternalMetricError(
                "watermark_decision_positive does not match the "
                "raw/rescue trajectory"
            )


@dataclass(frozen=True, slots=True)
class RescueSafetyAggregate:
    split: str
    cases: tuple[RescueSafetyCaseResult, ...]
    raw_fpr: float
    rescue_additional_fpr: float
    global_fpr: float
    global_fpr_upper_confidence_bound: float
    confidence_level: float
    target_fpr: float
    global_fpr_within_target: bool
    primary_null_count: int
    detector_identity: str
    threshold_identity: str
    metric_registry_digest: str


@dataclass(frozen=True, slots=True)
class RescueSafetyCaseResult:
    unit_id: str
    case_id: str
    source_cluster_id: str
    split: str
    watermark_decision_positive: bool
    raw_false_positive: bool
    rescue_additional_false_positive: bool
    global_false_positive: bool


def aggregate_rescue_fpr_safety(
    cases: Sequence[RescueSafetyCase],
    *,
    target_fpr: float,
    registry: MetricRegistry,
) -> RescueSafetyAggregate:
    _require_cases(cases)
    if type(registry) is not MetricRegistry or any(
        type(case) is not RescueSafetyCase for case in cases
    ):
        raise InternalMetricError("rescue aggregation inputs are invalid")
    _ensure_unique_units(cases)
    split = _require_uniform_metric_split(
        cases,
        registry=registry,
        metric_ids=("rescue_global_fpr_safety",),
    )
    alpha = _finite(target_fpr, "target_fpr")
    if not 0.0 < alpha < 1.0:
        raise InternalMetricError("target_fpr must be in (0,1)")
    detector_identities = {case.raw_detector_identity for case in cases}
    threshold_identities = {case.raw_threshold_identity for case in cases}
    if len(detector_identities) != 1 or len(threshold_identities) != 1:
        raise InternalMetricError("rescue aggregate identity mismatch")
    count = len(cases)
    raw_false_positives = sum(case.raw_positive for case in cases)
    rescue_false_positives = sum(
        case.rescue_triggered and case.rectified_positive
        for case in cases
    )
    global_false_positives = sum(
        case.watermark_decision_positive for case in cases
    )
    global_fpr = global_false_positives / count
    results = tuple(
        RescueSafetyCaseResult(
            unit_id=case.analysis_unit_identity.unit_id,
            case_id=case.analysis_unit_identity.case_id,
            source_cluster_id=case.analysis_unit_identity.source_cluster_id,
            split=case.split,
            watermark_decision_positive=(
                case.watermark_decision_positive
            ),
            raw_false_positive=case.raw_positive,
            rescue_additional_false_positive=(
                case.rescue_triggered and case.rectified_positive
            ),
            global_false_positive=case.watermark_decision_positive,
        )
        for case in cases
    )
    global_upper_bound = _binomial_upper_confidence_bound(
        global_false_positives,
        count,
    )
    return RescueSafetyAggregate(
        split=split,
        cases=results,
        raw_fpr=raw_false_positives / count,
        rescue_additional_fpr=rescue_false_positives / count,
        global_fpr=global_fpr,
        global_fpr_upper_confidence_bound=global_upper_bound,
        confidence_level=0.95,
        target_fpr=alpha,
        global_fpr_within_target=(
            global_fpr <= alpha and global_upper_bound <= alpha
        ),
        primary_null_count=count,
        detector_identity=next(iter(detector_identities)),
        threshold_identity=next(iter(threshold_identities)),
        metric_registry_digest=registry.registry_digest,
    )
