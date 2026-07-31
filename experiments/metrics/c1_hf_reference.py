"""Executable C1 HF-reference metrics frozen by the C1-P run specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Sequence

from experiments.protocol.c1_hf_reference import (
    C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
    load_c1_hf_reference_bundle,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
)

from .binomial import clopper_pearson_lower, clopper_pearson_upper
from .internal import MetricRegistry, load_metric_registry, validate_metric_registry


C1_HF_METRIC_IMPLEMENTATION_SCHEMA_VERSION = (
    "ceg_wm_c1_hf_metric_implementation_binding_v2"
)
C1_HF_METRIC_IDS = (
    "c1_hf_tau_fit",
    "c1_hf_primary_null_fixed_fpr",
    "c1_hf_registered_tpr",
    "c1_hf_wrong_key_false_accept",
    "c1_hf_paired_key_attribution",
    "c1_hf_paired_final_image_quality",
    "c1_hf_actual_dtype_integrity",
)
C1_HF_METRIC_SPLIT_BINDINGS = (
    ("c1_hf_tau_fit", ("content_threshold_fit",)),
    ("c1_hf_primary_null_fixed_fpr", ("untouched_confirmation",)),
    ("c1_hf_registered_tpr", ("untouched_confirmation",)),
    ("c1_hf_wrong_key_false_accept", ("untouched_confirmation",)),
    ("c1_hf_paired_key_attribution", ("untouched_confirmation",)),
    ("c1_hf_paired_final_image_quality", ("untouched_confirmation",)),
    ("c1_hf_actual_dtype_integrity", ("untouched_confirmation",)),
)
C1_HF_IMPLEMENTATION_SYMBOLS = [
    {"metric_id": "c1_hf_tau_fit", "public_callable": "fit_c1_hf_tau"},
    {
        "metric_id": "c1_hf_primary_null_fixed_fpr",
        "public_callable": "evaluate_c1_hf_primary_null_fixed_fpr",
    },
    {
        "metric_id": "c1_hf_registered_tpr",
        "public_callable": "evaluate_c1_hf_registered_tpr",
    },
    {
        "metric_id": "c1_hf_wrong_key_false_accept",
        "public_callable": "evaluate_c1_hf_wrong_key_false_accept",
    },
    {
        "metric_id": "c1_hf_paired_key_attribution",
        "public_callable": "evaluate_c1_hf_paired_key_attribution",
    },
    {
        "metric_id": "c1_hf_paired_final_image_quality",
        "public_callable": "evaluate_c1_hf_paired_rgb8_quality",
    },
    {
        "metric_id": "c1_hf_actual_dtype_integrity",
        "public_callable": "evaluate_c1_hf_actual_dtype_integrity",
    },
]
C1_HF_FORMULA_IMPLEMENTATION_IDENTITIES = {
    "actual_dtype_integrity": (
        "per_registered_positive_integrity_and_exact_dtype_and_finite_"
        "nonnegative_actual_relative_l2_at_most_3_over_250"
    ),
    "binomial_intervals": (
        "exact_one_sided_95_percent_clopper_pearson_upper_or_lower"
    ),
    "paired_attribution": (
        "registered_score_strictly_greater_than_wrong_score_tie_is_failure"
    ),
    "paired_rgb8_mse": (
        "mean_of_squared_uint8_channel_differences_after_division_by_255"
    ),
    "paired_rgb8_relative_l2": (
        "l2_of_uint8_channel_difference_after_division_by_255_over_"
        "clean_image_l2_after_division_by_255"
    ),
    "quality_aggregation": (
        "mean_sample_sd_ddof_1_and_two_sided_95_percent_student_t_interval"
    ),
    "formal_confirmation_authority": (
        "recompute_tau_from_exact_fit_cases_and_replay_raw_rgb8_artifacts"
    ),
    "threshold": (
        "binary64_nextafter_above_primary_null_maximum_and_score_at_least_tau"
    ),
}
C1_HF_EXPECTED_SPECIFICATION_DIGEST = (
    "aa21c78233926049fe3cc1d0f8ec14469330fcaa33174558b45254153a10f3be"
)
C1_HF_THRESHOLD_RULE = "float64_nextafter_above_fit_primary_null_maximum"
C1_HF_DECISION_COMPARISON = "score_greater_than_or_equal_to_tau"
C1_HF_PRIMARY_NULL_ROLE = "unwatermarked_primary_null"
C1_HF_REGISTERED_ROLE = "registered_positive"
C1_HF_WRONG_KEY_ROLE = "wrong_key"
C1_HF_PRIMARY_NULL_CONTROL = "unwatermarked_image_with_registered_detection_key"
C1_HF_REGISTERED_CONTROL = "registered_watermarked_image_with_registered_detection_key"
C1_HF_WRONG_KEY_CONTROL = (
    "watermarked_image_with_preregistered_wrong_detection_key"
)
C1_HF_CONTENT_RELATIVE_L2_LIMIT = 3.0 / 250.0
DEFAULT_C1_HF_METRIC_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "c1_hf_metric_implementation.json"
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class C1HfMetricError(ValueError):
    """C1 metric input, authority, or frozen identity failed closed."""


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise C1HfMetricError(f"{role}_invalid")
    return value


def _require_identity(value: object, role: str) -> str:
    if type(value) is not str or not value:
        raise C1HfMetricError(f"{role}_invalid")
    return value


def _finite_float64(value: object, role: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise C1HfMetricError(f"{role}_must_be_finite_float64")
    return value


@dataclass(frozen=True, slots=True)
class C1HfMetricImplementationBinding:
    c1_specification_digest: str
    protocol_digest: str
    fit_manifest_digest: str
    confirmation_manifest_digest: str
    registered_key_family_digest: str
    metric_registry_digest: str
    formula_identity_digest: str
    implementation_source_sha256: str
    binding_digest: str
    fit_analysis_units: frozenset[AnalysisUnitIdentity]
    confirmation_analysis_units: frozenset[AnalysisUnitIdentity]


def load_c1_hf_metric_implementation_binding(
    path: str | Path = DEFAULT_C1_HF_METRIC_IMPLEMENTATION_PATH,
) -> C1HfMetricImplementationBinding:
    binding_path = Path(path)
    raw = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "c1_specification_path",
        "c1_specification_digest",
        "protocol_digest",
        "split_manifest_digests",
        "registered_key_family_digest",
        "metric_registry_path",
        "metric_registry_digest",
        "metric_ids",
        "metric_split_bindings",
        "frozen_formula_identity_digest",
        "formula_implementation_identities",
        "formula_identity_digest",
        "implementation_module_path",
        "implementation_source_sha256",
        "implementation_symbols",
        "formal_confirmation_entrypoint",
        "confirmation_cross_input_validation",
        "binding_digest",
    }
    if type(raw) is not dict or set(raw) != expected_keys:
        raise C1HfMetricError("c1_metric_implementation_binding_fields_invalid")
    if raw["schema_version"] != C1_HF_METRIC_IMPLEMENTATION_SCHEMA_VERSION:
        raise C1HfMetricError("c1_metric_implementation_schema_mismatch")
    repository_root = binding_path.resolve().parents[2]
    reference_bundle = load_c1_hf_reference_bundle(repository_root)
    specification = reference_bundle.specification
    if (
        raw["c1_specification_path"]
        != "configs/experiments/c1_hf_reference_run.json"
        or specification.digest() != C1_HF_EXPECTED_SPECIFICATION_DIGEST
        or raw["c1_specification_digest"] != specification.digest()
    ):
        raise C1HfMetricError("c1_metric_specification_binding_mismatch")
    metric_plan = specification.raw["metric_plan"]
    expected_split_bindings = [
        {"metric_id": metric_id, "allowed_splits": list(splits)}
        for metric_id, splits in C1_HF_METRIC_SPLIT_BINDINGS
    ]
    if (
        raw["metric_ids"] != list(C1_HF_METRIC_IDS)
        or raw["metric_split_bindings"] != expected_split_bindings
        or metric_plan["metric_split_bindings"] != expected_split_bindings
        or raw["implementation_symbols"] != C1_HF_IMPLEMENTATION_SYMBOLS
        or raw["formal_confirmation_entrypoint"]
        != "evaluate_c1_hf_confirmation_metrics"
        or raw["confirmation_cross_input_validation"]
        != "required_fit_tau_and_raw_rgb8_replay_before_confirmation_results"
    ):
        raise C1HfMetricError("c1_metric_identity_or_split_binding_mismatch")
    frozen_formula_digest = _canonical_digest(metric_plan["formula_identities"])
    implementation_formula_digest = _canonical_digest(
        C1_HF_FORMULA_IMPLEMENTATION_IDENTITIES
    )
    if (
        raw["frozen_formula_identity_digest"] != frozen_formula_digest
        or raw["formula_implementation_identities"]
        != C1_HF_FORMULA_IMPLEMENTATION_IDENTITIES
        or raw["formula_identity_digest"] != implementation_formula_digest
    ):
        raise C1HfMetricError("c1_metric_formula_binding_mismatch")
    registry_path = repository_root / raw["metric_registry_path"]
    registry = load_metric_registry(registry_path)
    if validate_metric_registry(registry):
        raise C1HfMetricError("c1_metric_registry_invalid")
    if (
        raw["metric_registry_path"]
        != "configs/experiments/internal_execution_components.json"
        or raw["metric_registry_digest"] != registry.registry_digest
        or any(metric_id not in registry.metric_ids for metric_id in C1_HF_METRIC_IDS)
    ):
        raise C1HfMetricError("c1_metric_registry_binding_mismatch")
    implementation_path = repository_root / raw["implementation_module_path"]
    if (
        raw["implementation_module_path"]
        != "experiments/metrics/c1_hf_reference.py"
        or not implementation_path.is_file()
        or raw["implementation_source_sha256"] != _file_sha256(implementation_path)
    ):
        raise C1HfMetricError("c1_metric_implementation_source_mismatch")
    split_digests = raw["split_manifest_digests"]
    expected_split_digests = {
        name: value["materialized_manifest_digest"]
        for name, value in specification.raw["split_manifests"].items()
    }
    if split_digests != expected_split_digests:
        raise C1HfMetricError("c1_metric_manifest_binding_mismatch")
    materialized_manifests = {
        manifest.assignments[0].split: manifest
        for manifest in reference_bundle.materialized_manifests
    }
    if {
        split: manifest.digest()
        for split, manifest in materialized_manifests.items()
    } != split_digests:
        raise C1HfMetricError("c1_metric_materialized_manifest_mismatch")
    expected_protocol_digest = specification.raw["protocol_id"]
    if (
        expected_protocol_digest != INTERNAL_VALIDATION_PROTOCOL_ID
        or specification.raw["protocol_version"]
        != INTERNAL_VALIDATION_PROTOCOL_VERSION
        or raw["protocol_digest"]
        != "e05cab6af8adf4413bd33d8fca8226103ac5ed617208cdb41e57c0f6cb0f984d"
        or raw["registered_key_family_digest"]
        != specification.raw["key_controls"]["registered_key_family_digest"]
    ):
        raise C1HfMetricError("c1_metric_protocol_or_key_binding_mismatch")
    payload = {key: value for key, value in raw.items() if key != "binding_digest"}
    if raw["binding_digest"] != _canonical_digest(payload):
        raise C1HfMetricError("c1_metric_implementation_binding_digest_mismatch")
    return C1HfMetricImplementationBinding(
        c1_specification_digest=raw["c1_specification_digest"],
        protocol_digest=raw["protocol_digest"],
        fit_manifest_digest=split_digests["content_threshold_fit"],
        confirmation_manifest_digest=split_digests["untouched_confirmation"],
        registered_key_family_digest=raw["registered_key_family_digest"],
        metric_registry_digest=raw["metric_registry_digest"],
        formula_identity_digest=raw["formula_identity_digest"],
        implementation_source_sha256=raw["implementation_source_sha256"],
        binding_digest=raw["binding_digest"],
        fit_analysis_units=frozenset(
            assignment.identity
            for assignment in materialized_manifests[
                "content_threshold_fit"
            ].assignments
        ),
        confirmation_analysis_units=frozenset(
            assignment.identity
            for assignment in materialized_manifests[
                "untouched_confirmation"
            ].assignments
        ),
    )


@dataclass(frozen=True, slots=True)
class C1HfMetricCaseIdentity:
    analysis_unit_identity: AnalysisUnitIdentity
    split: str
    detector_identity: str
    detector_config_digest: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    c1_specification_digest: str
    manifest_digest: str
    metric_registry_digest: str
    registered_key_family_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.analysis_unit_identity) is not AnalysisUnitIdentity
            or self.analysis_unit_identity.validate()
        ):
            raise C1HfMetricError("analysis_unit_identity_invalid")
        _require_identity(self.detector_identity, "detector_identity")
        for name in (
            "detector_config_digest",
            "protocol_digest",
            "c1_specification_digest",
            "manifest_digest",
            "metric_registry_digest",
            "registered_key_family_digest",
        ):
            _require_digest(getattr(self, name), name)

    @property
    def unit_id(self) -> str:
        return self.analysis_unit_identity.unit_id

    @property
    def case_id(self) -> str:
        return self.analysis_unit_identity.case_id

    @property
    def source_cluster_id(self) -> str:
        return self.analysis_unit_identity.source_cluster_id


def _validate_case_identity(
    identity: C1HfMetricCaseIdentity,
    binding: C1HfMetricImplementationBinding,
    required_split: str,
) -> None:
    if type(identity) is not C1HfMetricCaseIdentity:
        raise C1HfMetricError("c1_metric_case_identity_exact_type_required")
    expected_manifest = (
        binding.fit_manifest_digest
        if required_split == "content_threshold_fit"
        else binding.confirmation_manifest_digest
    )
    expected_units = (
        binding.fit_analysis_units
        if required_split == "content_threshold_fit"
        else binding.confirmation_analysis_units
    )
    if (
        identity.analysis_unit_identity not in expected_units
        or
        identity.split != required_split
        or identity.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID
        or identity.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION
        or identity.protocol_digest != binding.protocol_digest
        or identity.c1_specification_digest != binding.c1_specification_digest
        or identity.manifest_digest != expected_manifest
        or identity.metric_registry_digest != binding.metric_registry_digest
        or identity.registered_key_family_digest
        != binding.registered_key_family_digest
    ):
        raise C1HfMetricError("c1_metric_case_authority_identity_mismatch")


def _validate_exact_analysis_unit_set(
    identities: Sequence[C1HfMetricCaseIdentity],
    binding: C1HfMetricImplementationBinding,
    required_split: str,
) -> None:
    actual = {identity.analysis_unit_identity for identity in identities}
    expected = (
        binding.fit_analysis_units
        if required_split == "content_threshold_fit"
        else binding.confirmation_analysis_units
    )
    if len(actual) != len(identities) or actual != expected:
        raise C1HfMetricError("c1_metric_analysis_unit_manifest_set_mismatch")


@dataclass(frozen=True, slots=True)
class C1HfScoreCase:
    identity: C1HfMetricCaseIdentity
    key_role: str
    score: float
    registered_detection_key_public_digest: str
    detection_key_public_digest: str
    control_identity: str
    image_digest: str

    def __post_init__(self) -> None:
        if type(self.identity) is not C1HfMetricCaseIdentity:
            raise C1HfMetricError("c1_score_case_identity_invalid")
        object.__setattr__(
            self,
            "score",
            _finite_float64(self.score, "c1_hf_score"),
        )
        _require_digest(
            self.registered_detection_key_public_digest,
            "registered_detection_key_public_digest",
        )
        _require_digest(
            self.detection_key_public_digest,
            "detection_key_public_digest",
        )
        _require_identity(self.control_identity, "control_identity")
        _require_digest(self.image_digest, "image_digest")


def _ordered_score_case_payload(cases: Sequence[C1HfScoreCase]) -> list[dict[str, object]]:
    return [
        {
            "case_id": case.identity.case_id,
            "control_identity": case.control_identity,
            "detection_key_public_digest": case.detection_key_public_digest,
            "detector_config_digest": case.identity.detector_config_digest,
            "detector_identity": case.identity.detector_identity,
            "image_digest": case.image_digest,
            "key_role": case.key_role,
            "registered_detection_key_public_digest": (
                case.registered_detection_key_public_digest
            ),
            "score": case.score.hex(),
            "source_cluster_id": case.identity.source_cluster_id,
        }
        for case in sorted(
            cases,
            key=lambda item: (
                item.identity.source_cluster_id,
                item.key_role,
                item.identity.case_id,
            ),
        )
    ]


@dataclass(frozen=True, slots=True)
class C1HfThresholdResult:
    metric_id: str
    split: str
    threshold_rule: str
    decision_comparison: str
    tau: float
    tau_float64_hex: str
    fit_false_positive_count: int
    source_cluster_count: int
    detector_identity: str
    detector_config_digest: str
    protocol_id: str
    protocol_version: str
    protocol_digest: str
    c1_specification_digest: str
    fit_manifest_digest: str
    metric_registry_digest: str
    case_digest: str
    source_cluster_digest: str
    threshold_identity: str


def _threshold_payload(result: C1HfThresholdResult) -> dict[str, object]:
    return {
        key: getattr(result, key)
        for key in result.__dataclass_fields__
        if key != "threshold_identity"
    }


def validate_c1_hf_threshold_result(
    result: C1HfThresholdResult,
    binding: C1HfMetricImplementationBinding,
) -> None:
    if type(result) is not C1HfThresholdResult:
        raise C1HfMetricError("c1_threshold_exact_type_required")
    if (
        result.metric_id != C1_HF_METRIC_IDS[0]
        or result.split != "content_threshold_fit"
        or result.threshold_rule != C1_HF_THRESHOLD_RULE
        or result.decision_comparison != C1_HF_DECISION_COMPARISON
        or type(result.tau) is not float
        or not math.isfinite(result.tau)
        or result.tau_float64_hex != result.tau.hex()
        or result.fit_false_positive_count != 0
        or result.source_cluster_count != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or result.protocol_id != INTERNAL_VALIDATION_PROTOCOL_ID
        or result.protocol_version != INTERNAL_VALIDATION_PROTOCOL_VERSION
        or result.protocol_digest != binding.protocol_digest
        or result.c1_specification_digest != binding.c1_specification_digest
        or result.fit_manifest_digest != binding.fit_manifest_digest
        or result.metric_registry_digest != binding.metric_registry_digest
    ):
        raise C1HfMetricError("c1_threshold_frozen_identity_mismatch")
    for value, role in (
        (result.detector_config_digest, "detector_config_digest"),
        (result.case_digest, "case_digest"),
        (result.source_cluster_digest, "source_cluster_digest"),
        (result.threshold_identity, "threshold_identity"),
    ):
        _require_digest(value, role)
    _require_identity(result.detector_identity, "detector_identity")
    if result.threshold_identity != _canonical_digest(_threshold_payload(result)):
        raise C1HfMetricError("c1_threshold_identity_mismatch")


def fit_c1_hf_tau(
    primary_null_cases: Sequence[C1HfScoreCase],
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfThresholdResult:
    """Fit the exact C1 threshold from all 4096 frozen primary-null clusters."""

    if (
        isinstance(primary_null_cases, (str, bytes))
        or not isinstance(primary_null_cases, Sequence)
        or len(primary_null_cases) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or any(type(case) is not C1HfScoreCase for case in primary_null_cases)
    ):
        raise C1HfMetricError("c1_tau_fit_requires_exact_4096_score_cases")
    for case in primary_null_cases:
        _validate_case_identity(case.identity, binding, "content_threshold_fit")
        if (
            case.key_role != C1_HF_PRIMARY_NULL_ROLE
            or case.control_identity != C1_HF_PRIMARY_NULL_CONTROL
            or case.detection_key_public_digest
            != case.registered_detection_key_public_digest
        ):
            raise C1HfMetricError("c1_tau_fit_primary_null_identity_mismatch")
    _validate_exact_analysis_unit_set(
        [case.identity for case in primary_null_cases],
        binding,
        "content_threshold_fit",
    )
    cluster_ids = [case.identity.source_cluster_id for case in primary_null_cases]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise C1HfMetricError("c1_tau_fit_source_cluster_duplicate")
    uniform = {
        (
            case.identity.detector_identity,
            case.identity.detector_config_digest,
            case.identity.protocol_id,
            case.identity.protocol_version,
            case.identity.protocol_digest,
            case.identity.c1_specification_digest,
            case.identity.manifest_digest,
            case.identity.metric_registry_digest,
        )
        for case in primary_null_cases
    }
    if len(uniform) != 1:
        raise C1HfMetricError("c1_tau_fit_authority_identity_drift")
    tau = math.nextafter(
        max(case.score for case in primary_null_cases),
        math.inf,
    )
    if not math.isfinite(tau):
        raise C1HfMetricError("c1_tau_fit_no_finite_nextafter_threshold")
    false_positive_count = sum(case.score >= tau for case in primary_null_cases)
    if false_positive_count != 0:
        raise C1HfMetricError("c1_tau_fit_false_positive_count_nonzero")
    first = primary_null_cases[0].identity
    result_payload = {
        "metric_id": C1_HF_METRIC_IDS[0],
        "split": "content_threshold_fit",
        "threshold_rule": C1_HF_THRESHOLD_RULE,
        "decision_comparison": C1_HF_DECISION_COMPARISON,
        "tau": tau,
        "tau_float64_hex": tau.hex(),
        "fit_false_positive_count": false_positive_count,
        "source_cluster_count": len(cluster_ids),
        "detector_identity": first.detector_identity,
        "detector_config_digest": first.detector_config_digest,
        "protocol_id": first.protocol_id,
        "protocol_version": first.protocol_version,
        "protocol_digest": first.protocol_digest,
        "c1_specification_digest": first.c1_specification_digest,
        "fit_manifest_digest": first.manifest_digest,
        "metric_registry_digest": first.metric_registry_digest,
        "case_digest": _canonical_digest(
            _ordered_score_case_payload(primary_null_cases)
        ),
        "source_cluster_digest": _canonical_digest(sorted(cluster_ids)),
    }
    result = C1HfThresholdResult(
        **result_payload,
        threshold_identity=_canonical_digest(result_payload),
    )
    validate_c1_hf_threshold_result(result, binding)
    return result


@dataclass(frozen=True, slots=True)
class C1HfBinomialMetricResult:
    metric_id: str
    split: str
    event_count: int
    trial_count: int
    empirical_rate: float
    confidence_level: float
    confidence_direction: str
    confidence_bound: float
    threshold_identity: str
    manifest_digest: str
    metric_registry_digest: str
    case_digest: str
    source_cluster_digest: str
    result_identity: str


def _prepare_confirmation_scores(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    binding: C1HfMetricImplementationBinding,
) -> tuple[dict[str, list[C1HfScoreCase]], str, str]:
    validate_c1_hf_threshold_result(threshold, binding)
    required_total = 3 * C1_HF_SOURCE_CLUSTERS_PER_SPLIT
    if (
        isinstance(cases, (str, bytes))
        or not isinstance(cases, Sequence)
        or len(cases) != required_total
        or any(type(case) is not C1HfScoreCase for case in cases)
    ):
        raise C1HfMetricError("c1_confirmation_requires_exact_three_roles_per_4096")
    grouped_by_cluster: dict[str, list[C1HfScoreCase]] = {}
    for case in cases:
        _validate_case_identity(case.identity, binding, "untouched_confirmation")
        grouped_by_cluster.setdefault(case.identity.source_cluster_id, []).append(case)
    if len(grouped_by_cluster) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT:
        raise C1HfMetricError("c1_confirmation_source_cluster_count_mismatch")
    grouped_by_role = {
        role: [] for role in (
            C1_HF_PRIMARY_NULL_ROLE,
            C1_HF_REGISTERED_ROLE,
            C1_HF_WRONG_KEY_ROLE,
        )
    }
    for cluster_cases in grouped_by_cluster.values():
        roles = [case.key_role for case in cluster_cases]
        if sorted(roles) != sorted(grouped_by_role):
            raise C1HfMetricError("c1_confirmation_role_multiplicity_mismatch")
        by_role = {case.key_role: case for case in cluster_cases}
        primary = by_role[C1_HF_PRIMARY_NULL_ROLE]
        registered = by_role[C1_HF_REGISTERED_ROLE]
        wrong = by_role[C1_HF_WRONG_KEY_ROLE]
        if (
            primary.control_identity != C1_HF_PRIMARY_NULL_CONTROL
            or registered.control_identity != C1_HF_REGISTERED_CONTROL
            or wrong.control_identity != C1_HF_WRONG_KEY_CONTROL
            or primary.detection_key_public_digest
            != primary.registered_detection_key_public_digest
            or registered.detection_key_public_digest
            != registered.registered_detection_key_public_digest
            or wrong.detection_key_public_digest
            == wrong.registered_detection_key_public_digest
            or registered.registered_detection_key_public_digest
            != wrong.registered_detection_key_public_digest
            or primary.registered_detection_key_public_digest
            != registered.registered_detection_key_public_digest
            or registered.image_digest != wrong.image_digest
        ):
            raise C1HfMetricError("c1_confirmation_key_control_or_pair_identity_mismatch")
        if primary.image_digest == registered.image_digest:
            raise C1HfMetricError("c1_confirmation_clean_and_marked_image_identity_collision")
        for role, case in by_role.items():
            if (
                case.identity.detector_identity != threshold.detector_identity
                or case.identity.detector_config_digest
                != threshold.detector_config_digest
            ):
                raise C1HfMetricError("c1_confirmation_detector_identity_mismatch")
            grouped_by_role[role].append(case)
    _validate_exact_analysis_unit_set(
        [cluster_cases[0].identity for cluster_cases in grouped_by_cluster.values()],
        binding,
        "untouched_confirmation",
    )
    case_digest = _canonical_digest(_ordered_score_case_payload(cases))
    source_cluster_digest = _canonical_digest(sorted(grouped_by_cluster))
    return grouped_by_role, case_digest, source_cluster_digest


def _binomial_result(
    *,
    metric_id: str,
    event_count: int,
    confidence_direction: str,
    threshold: C1HfThresholdResult,
    binding: C1HfMetricImplementationBinding,
    case_digest: str,
    source_cluster_digest: str,
) -> C1HfBinomialMetricResult:
    trial_count = C1_HF_SOURCE_CLUSTERS_PER_SPLIT
    confidence_bound = (
        clopper_pearson_upper(event_count, trial_count)
        if confidence_direction == "one_sided_upper"
        else clopper_pearson_lower(event_count, trial_count)
    )
    payload = {
        "metric_id": metric_id,
        "split": "untouched_confirmation",
        "event_count": event_count,
        "trial_count": trial_count,
        "empirical_rate": event_count / trial_count,
        "confidence_level": 0.95,
        "confidence_direction": confidence_direction,
        "confidence_bound": confidence_bound,
        "threshold_identity": threshold.threshold_identity,
        "manifest_digest": binding.confirmation_manifest_digest,
        "metric_registry_digest": binding.metric_registry_digest,
        "case_digest": case_digest,
        "source_cluster_digest": source_cluster_digest,
    }
    result = C1HfBinomialMetricResult(
        **payload,
        result_identity=_canonical_digest(
            {
                key: value.hex() if type(value) is float else value
                for key, value in payload.items()
            }
        ),
    )
    validate_c1_hf_binomial_metric_result(result, binding)
    return result


def validate_c1_hf_binomial_metric_result(
    result: C1HfBinomialMetricResult,
    binding: C1HfMetricImplementationBinding,
) -> None:
    if type(result) is not C1HfBinomialMetricResult:
        raise C1HfMetricError("c1_binomial_result_exact_type_required")
    expected_direction = {
        "c1_hf_primary_null_fixed_fpr": "one_sided_upper",
        "c1_hf_registered_tpr": "one_sided_lower",
        "c1_hf_wrong_key_false_accept": "one_sided_upper",
        "c1_hf_paired_key_attribution": "one_sided_lower",
    }.get(result.metric_id)
    if (
        result.metric_id not in C1_HF_METRIC_IDS[1:5]
        or result.split != "untouched_confirmation"
        or type(result.event_count) is not int
        or result.trial_count != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or not 0 <= result.event_count <= result.trial_count
        or result.empirical_rate != result.event_count / result.trial_count
        or result.confidence_level != 0.95
        or result.confidence_direction
        not in {"one_sided_upper", "one_sided_lower"}
        or result.confidence_direction != expected_direction
        or result.manifest_digest != binding.confirmation_manifest_digest
        or result.metric_registry_digest != binding.metric_registry_digest
    ):
        raise C1HfMetricError("c1_binomial_result_identity_or_count_mismatch")
    expected_bound = (
        clopper_pearson_upper(result.event_count, result.trial_count)
        if result.confidence_direction == "one_sided_upper"
        else clopper_pearson_lower(result.event_count, result.trial_count)
    )
    if result.confidence_bound != expected_bound:
        raise C1HfMetricError("c1_binomial_confidence_bound_mismatch")
    for value, role in (
        (result.threshold_identity, "threshold_identity"),
        (result.case_digest, "case_digest"),
        (result.source_cluster_digest, "source_cluster_digest"),
        (result.result_identity, "binomial_result_identity"),
    ):
        _require_digest(value, role)
    payload = {
        key: getattr(result, key)
        for key in result.__dataclass_fields__
        if key != "result_identity"
    }
    expected_identity = _canonical_digest(
        {
            key: value.hex() if type(value) is float else value
            for key, value in payload.items()
        }
    )
    if result.result_identity != expected_identity:
        raise C1HfMetricError("c1_binomial_result_identity_mismatch")


def _confirmation_metric(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    binding: C1HfMetricImplementationBinding,
    metric_id: str,
) -> C1HfBinomialMetricResult:
    grouped, case_digest, cluster_digest = _prepare_confirmation_scores(
        cases,
        threshold,
        binding,
    )
    if metric_id == "c1_hf_primary_null_fixed_fpr":
        events = sum(
            case.score >= threshold.tau
            for case in grouped[C1_HF_PRIMARY_NULL_ROLE]
        )
        direction = "one_sided_upper"
    elif metric_id == "c1_hf_registered_tpr":
        events = sum(
            case.score >= threshold.tau
            for case in grouped[C1_HF_REGISTERED_ROLE]
        )
        direction = "one_sided_lower"
    elif metric_id == "c1_hf_wrong_key_false_accept":
        events = sum(
            case.score >= threshold.tau
            for case in grouped[C1_HF_WRONG_KEY_ROLE]
        )
        direction = "one_sided_upper"
    elif metric_id == "c1_hf_paired_key_attribution":
        registered = {
            case.identity.source_cluster_id: case
            for case in grouped[C1_HF_REGISTERED_ROLE]
        }
        wrong = {
            case.identity.source_cluster_id: case
            for case in grouped[C1_HF_WRONG_KEY_ROLE]
        }
        events = sum(
            registered[cluster].score > wrong[cluster].score
            for cluster in registered
        )
        direction = "one_sided_lower"
    else:
        raise C1HfMetricError("c1_confirmation_metric_id_invalid")
    return _binomial_result(
        metric_id=metric_id,
        event_count=events,
        confidence_direction=direction,
        threshold=threshold,
        binding=binding,
        case_digest=case_digest,
        source_cluster_digest=cluster_digest,
    )


def evaluate_c1_hf_primary_null_fixed_fpr(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfBinomialMetricResult:
    return _confirmation_metric(
        cases,
        threshold,
        binding,
        "c1_hf_primary_null_fixed_fpr",
    )


def evaluate_c1_hf_registered_tpr(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfBinomialMetricResult:
    return _confirmation_metric(
        cases,
        threshold,
        binding,
        "c1_hf_registered_tpr",
    )


def evaluate_c1_hf_wrong_key_false_accept(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfBinomialMetricResult:
    return _confirmation_metric(
        cases,
        threshold,
        binding,
        "c1_hf_wrong_key_false_accept",
    )


def evaluate_c1_hf_paired_key_attribution(
    cases: Sequence[C1HfScoreCase],
    threshold: C1HfThresholdResult,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfBinomialMetricResult:
    return _confirmation_metric(
        cases,
        threshold,
        binding,
        "c1_hf_paired_key_attribution",
    )


@dataclass(frozen=True, slots=True)
class C1Rgb8Image:
    height: int
    width: int
    channels: int
    dtype: str
    values_hwc: bytes

    def __post_init__(self) -> None:
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or self.height <= 0
            or self.width <= 0
            or self.channels != 3
            or self.dtype != "uint8"
            or type(self.values_hwc) is not bytes
            or len(self.values_hwc) != self.height * self.width * self.channels
        ):
            raise C1HfMetricError("c1_rgb8_image_hwc_dtype_or_shape_invalid")

    def digest(self) -> str:
        return hashlib.sha256(
            self.height.to_bytes(8, "big")
            + self.width.to_bytes(8, "big")
            + self.channels.to_bytes(1, "big")
            + self.values_hwc
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class C1HfQualityPair:
    identity: C1HfMetricCaseIdentity
    clean_image: C1Rgb8Image
    registered_watermarked_image: C1Rgb8Image
    clean_image_digest: str
    registered_watermarked_image_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.clean_image) is not C1Rgb8Image
            or type(self.registered_watermarked_image) is not C1Rgb8Image
            or self.clean_image.digest() != self.clean_image_digest
            or self.registered_watermarked_image.digest()
            != self.registered_watermarked_image_digest
        ):
            raise C1HfMetricError("c1_quality_image_digest_or_type_mismatch")


@dataclass(frozen=True, slots=True)
class C1HfRawRgb8QualityArtifact:
    """Lightweight descriptor for independently replayable paired RGB8 bytes."""

    identity: C1HfMetricCaseIdentity
    height: int
    width: int
    channels: int
    dtype: str
    clean_artifact_path: str
    clean_artifact_sha256: str
    registered_watermarked_artifact_path: str
    registered_watermarked_artifact_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not C1HfMetricCaseIdentity
            or type(self.height) is not int
            or type(self.width) is not int
            or self.height <= 0
            or self.width <= 0
            or self.channels != 3
            or self.dtype != "uint8"
        ):
            raise C1HfMetricError("c1_quality_artifact_shape_or_dtype_invalid")
        for value, role in (
            (self.clean_artifact_path, "clean_artifact_path"),
            (
                self.registered_watermarked_artifact_path,
                "registered_watermarked_artifact_path",
            ),
        ):
            _require_identity(value, role)
            if not Path(value).is_absolute():
                raise C1HfMetricError(f"{role}_must_be_absolute")
        _require_digest(self.clean_artifact_sha256, "clean_artifact_sha256")
        _require_digest(
            self.registered_watermarked_artifact_sha256,
            "registered_watermarked_artifact_sha256",
        )


def _read_exact_rgb8_artifact(
    artifact_path: str,
    expected_sha256: str,
    *,
    height: int,
    width: int,
    channels: int,
    dtype: str,
) -> C1Rgb8Image:
    expected_size = height * width * channels
    try:
        with Path(artifact_path).open("rb") as stream:
            values_hwc = stream.read(expected_size + 1)
    except OSError as exc:
        raise C1HfMetricError("c1_quality_artifact_read_failed") from exc
    if len(values_hwc) != expected_size:
        raise C1HfMetricError("c1_quality_artifact_byte_count_mismatch")
    if hashlib.sha256(values_hwc).hexdigest() != expected_sha256:
        raise C1HfMetricError("c1_quality_artifact_sha256_mismatch")
    return C1Rgb8Image(
        height=height,
        width=width,
        channels=channels,
        dtype=dtype,
        values_hwc=values_hwc,
    )


def _read_cached_exact_rgb8_artifact(
    artifact_cache: dict[tuple[object, ...], C1Rgb8Image],
    artifact_path: str,
    expected_sha256: str,
    *,
    height: int,
    width: int,
    channels: int,
    dtype: str,
) -> C1Rgb8Image:
    cache_key = (
        artifact_path,
        expected_sha256,
        height,
        width,
        channels,
        dtype,
    )
    cached = artifact_cache.get(cache_key)
    if cached is not None:
        return cached
    image = _read_exact_rgb8_artifact(
        artifact_path,
        expected_sha256,
        height=height,
        width=width,
        channels=channels,
        dtype=dtype,
    )
    if len(artifact_cache) == 2:
        artifact_cache.pop(next(iter(artifact_cache)))
    artifact_cache[cache_key] = image
    return image


def _replay_c1_hf_raw_rgb8_quality_artifact(
    artifact: C1HfRawRgb8QualityArtifact,
    binding: C1HfMetricImplementationBinding,
    artifact_cache: dict[tuple[object, ...], C1Rgb8Image],
) -> C1HfQualityCaseResult:
    if type(artifact) is not C1HfRawRgb8QualityArtifact:
        raise C1HfMetricError("c1_quality_artifact_exact_type_required")
    _validate_case_identity(artifact.identity, binding, "untouched_confirmation")
    clean = _read_cached_exact_rgb8_artifact(
        artifact_cache,
        artifact.clean_artifact_path,
        artifact.clean_artifact_sha256,
        height=artifact.height,
        width=artifact.width,
        channels=artifact.channels,
        dtype=artifact.dtype,
    )
    marked = _read_cached_exact_rgb8_artifact(
        artifact_cache,
        artifact.registered_watermarked_artifact_path,
        artifact.registered_watermarked_artifact_sha256,
        height=artifact.height,
        width=artifact.width,
        channels=artifact.channels,
        dtype=artifact.dtype,
    )
    return evaluate_c1_hf_rgb8_quality_pair(
        C1HfQualityPair(
            identity=artifact.identity,
            clean_image=clean,
            registered_watermarked_image=marked,
            clean_image_digest=clean.digest(),
            registered_watermarked_image_digest=marked.digest(),
        ),
        binding=binding,
    )


@dataclass(frozen=True, slots=True)
class C1HfQualityCaseResult:
    identity: C1HfMetricCaseIdentity
    relative_l2: float
    normalized_rgb8_mse: float
    clean_image_digest: str
    registered_watermarked_image_digest: str
    formula_identity_digest: str
    result_identity: str


@dataclass(frozen=True, slots=True)
class C1HfQualitySummary:
    metric_id: str
    sample_count: int
    mean: float
    sample_standard_deviation: float
    student_t_degrees_of_freedom: int
    student_t_critical_975: float
    confidence_interval_lower: float
    confidence_interval_upper: float


@dataclass(frozen=True, slots=True)
class C1HfPairedRgb8QualityResult:
    metric_id: str
    split: str
    pair_count: int
    case_results: tuple[C1HfQualityCaseResult, ...]
    relative_l2: C1HfQualitySummary
    normalized_rgb8_mse: C1HfQualitySummary
    manifest_digest: str
    metric_registry_digest: str
    case_digest: str
    source_cluster_digest: str
    result_identity: str


def evaluate_c1_hf_rgb8_quality_pair(
    pair: C1HfQualityPair,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfQualityCaseResult:
    """Compute both frozen quality formulas from the actual paired RGB8 bytes."""

    if type(pair) is not C1HfQualityPair:
        raise C1HfMetricError("c1_quality_pair_exact_type_required")
    _validate_case_identity(pair.identity, binding, "untouched_confirmation")
    clean = pair.clean_image
    marked = pair.registered_watermarked_image
    if (
        clean.height != marked.height
        or clean.width != marked.width
        or clean.channels != marked.channels
        or clean.dtype != marked.dtype
    ):
        raise C1HfMetricError("c1_quality_pair_shape_or_dtype_mismatch")
    clean_square_sum = math.fsum((value / 255.0) ** 2 for value in clean.values_hwc)
    if clean_square_sum == 0.0:
        raise C1HfMetricError("c1_quality_clean_l2_zero")
    difference_square_sum = math.fsum(
        ((candidate - reference) / 255.0) ** 2
        for reference, candidate in zip(
            clean.values_hwc,
            marked.values_hwc,
            strict=True,
        )
    )
    relative_l2 = math.sqrt(difference_square_sum) / math.sqrt(clean_square_sum)
    mse = difference_square_sum / len(clean.values_hwc)
    if not math.isfinite(relative_l2) or not math.isfinite(mse):
        raise C1HfMetricError("c1_quality_non_finite_result")
    payload = {
        "analysis_unit_identity": asdict(pair.identity.analysis_unit_identity),
        "relative_l2": relative_l2.hex(),
        "normalized_rgb8_mse": mse.hex(),
        "clean_image_digest": pair.clean_image_digest,
        "registered_watermarked_image_digest": (
            pair.registered_watermarked_image_digest
        ),
        "formula_identity_digest": binding.formula_identity_digest,
    }
    result = C1HfQualityCaseResult(
        identity=pair.identity,
        relative_l2=relative_l2,
        normalized_rgb8_mse=mse,
        clean_image_digest=pair.clean_image_digest,
        registered_watermarked_image_digest=(
            pair.registered_watermarked_image_digest
        ),
        formula_identity_digest=binding.formula_identity_digest,
        result_identity=_canonical_digest(payload),
    )
    validate_c1_hf_quality_case_result(result, binding)
    return result


def validate_c1_hf_quality_case_result(
    result: C1HfQualityCaseResult,
    binding: C1HfMetricImplementationBinding,
) -> None:
    if type(result) is not C1HfQualityCaseResult:
        raise C1HfMetricError("c1_quality_case_result_exact_type_required")
    _validate_case_identity(result.identity, binding, "untouched_confirmation")
    relative_l2 = _finite_float64(result.relative_l2, "relative_l2")
    mse = _finite_float64(result.normalized_rgb8_mse, "normalized_rgb8_mse")
    if relative_l2 < 0.0 or not 0.0 <= mse <= 1.0:
        raise C1HfMetricError("c1_quality_case_result_range_invalid")
    for value, role in (
        (result.clean_image_digest, "clean_image_digest"),
        (
            result.registered_watermarked_image_digest,
            "registered_watermarked_image_digest",
        ),
        (result.formula_identity_digest, "formula_identity_digest"),
        (result.result_identity, "quality_case_result_identity"),
    ):
        _require_digest(value, role)
    if result.formula_identity_digest != binding.formula_identity_digest:
        raise C1HfMetricError("c1_quality_formula_identity_mismatch")
    payload = {
        "analysis_unit_identity": asdict(result.identity.analysis_unit_identity),
        "relative_l2": relative_l2.hex(),
        "normalized_rgb8_mse": mse.hex(),
        "clean_image_digest": result.clean_image_digest,
        "registered_watermarked_image_digest": (
            result.registered_watermarked_image_digest
        ),
        "formula_identity_digest": result.formula_identity_digest,
    }
    if result.result_identity != _canonical_digest(payload):
        raise C1HfMetricError("c1_quality_case_result_identity_mismatch")


def _continued_beta_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 240
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        numerator = (
            iteration
            * (b - iteration)
            * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + numerator * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + numerator / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        numerator = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + numerator * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + numerator / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise C1HfMetricError("student_t_regularized_beta_did_not_converge")


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta_fraction(a, b, x) / a
    return 1.0 - front * _continued_beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or type(degrees_of_freedom) is not int
        or degrees_of_freedom <= 0
    ):
        raise C1HfMetricError("student_t_input_invalid")
    if value == 0.0:
        return 0.5
    ratio = degrees_of_freedom / (
        degrees_of_freedom + value * value
    )
    tail = 0.5 * _regularized_beta(
        degrees_of_freedom / 2.0,
        0.5,
        ratio,
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_quantile_975(degrees_of_freedom: int) -> float:
    """Numerically invert the exact Student-t CDF; no normal approximation."""

    if type(degrees_of_freedom) is not int or degrees_of_freedom <= 0:
        raise C1HfMetricError("student_t_degrees_of_freedom_invalid")
    lower = 0.0
    upper = 1.0
    while student_t_cdf(float(upper), degrees_of_freedom) < 0.975:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if student_t_cdf(float(midpoint), degrees_of_freedom) < 0.975:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _quality_summary(
    metric_id: str,
    values: Sequence[float],
) -> C1HfQualitySummary:
    sample_count = len(values)
    mean = math.fsum(values) / sample_count
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        sample_count - 1
    )
    sample_standard_deviation = math.sqrt(variance)
    degrees_of_freedom = sample_count - 1
    critical = student_t_quantile_975(degrees_of_freedom)
    half_width = critical * sample_standard_deviation / math.sqrt(sample_count)
    return C1HfQualitySummary(
        metric_id=metric_id,
        sample_count=sample_count,
        mean=mean,
        sample_standard_deviation=sample_standard_deviation,
        student_t_degrees_of_freedom=degrees_of_freedom,
        student_t_critical_975=critical,
        confidence_interval_lower=mean - half_width,
        confidence_interval_upper=mean + half_width,
    )


def _quality_aggregate_identity_payload(
    *,
    pair_count: int,
    relative_l2: C1HfQualitySummary,
    normalized_rgb8_mse: C1HfQualitySummary,
    manifest_digest: str,
    metric_registry_digest: str,
    case_digest: str,
    source_cluster_digest: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_digest": case_digest,
        "manifest_digest": manifest_digest,
        "metric_id": "c1_hf_paired_final_image_quality",
        "metric_registry_digest": metric_registry_digest,
        "normalized_rgb8_mse": {
            field: getattr(normalized_rgb8_mse, field)
            for field in normalized_rgb8_mse.__dataclass_fields__
        },
        "pair_count": pair_count,
        "relative_l2": {
            field: getattr(relative_l2, field)
            for field in relative_l2.__dataclass_fields__
        },
        "source_cluster_digest": source_cluster_digest,
        "split": "untouched_confirmation",
    }
    for summary_name in ("normalized_rgb8_mse", "relative_l2"):
        summary = payload[summary_name]
        assert isinstance(summary, dict)
        payload[summary_name] = {
            key: value.hex() if type(value) is float else value
            for key, value in summary.items()
        }
    return payload


def _quality_case_digest(
    case_results: Sequence[C1HfQualityCaseResult],
) -> str:
    return _canonical_digest(
        [
            {
                "analysis_unit_identity": asdict(
                    result.identity.analysis_unit_identity
                ),
                "clean_image_digest": result.clean_image_digest,
                "normalized_rgb8_mse": result.normalized_rgb8_mse.hex(),
                "registered_watermarked_image_digest": (
                    result.registered_watermarked_image_digest
                ),
                "relative_l2": result.relative_l2.hex(),
                "result_identity": result.result_identity,
            }
            for result in sorted(
                case_results,
                key=lambda item: item.identity.source_cluster_id,
            )
        ]
    )


def evaluate_c1_hf_paired_rgb8_quality(
    case_results: Sequence[C1HfQualityCaseResult],
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfPairedRgb8QualityResult:
    if (
        isinstance(case_results, (str, bytes))
        or not isinstance(case_results, Sequence)
        or len(case_results) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or any(
            type(result) is not C1HfQualityCaseResult
            for result in case_results
        )
    ):
        raise C1HfMetricError("c1_quality_requires_exact_4096_case_results")
    for result in case_results:
        validate_c1_hf_quality_case_result(result, binding)
    _validate_exact_analysis_unit_set(
        [result.identity for result in case_results],
        binding,
        "untouched_confirmation",
    )
    cluster_ids = [
        result.identity.source_cluster_id for result in case_results
    ]
    ordered_results = tuple(
        sorted(
            case_results,
            key=lambda item: item.identity.source_cluster_id,
        )
    )
    relative_summary = _quality_summary(
        "paired_rgb8_relative_l2",
        tuple(result.relative_l2 for result in ordered_results),
    )
    mse_summary = _quality_summary(
        "paired_rgb8_mse",
        tuple(result.normalized_rgb8_mse for result in ordered_results),
    )
    case_digest = _quality_case_digest(ordered_results)
    cluster_digest = _canonical_digest(sorted(cluster_ids))
    result = C1HfPairedRgb8QualityResult(
        metric_id="c1_hf_paired_final_image_quality",
        split="untouched_confirmation",
        pair_count=len(case_results),
        case_results=ordered_results,
        relative_l2=relative_summary,
        normalized_rgb8_mse=mse_summary,
        manifest_digest=binding.confirmation_manifest_digest,
        metric_registry_digest=binding.metric_registry_digest,
        case_digest=case_digest,
        source_cluster_digest=cluster_digest,
        result_identity=_canonical_digest(
            _quality_aggregate_identity_payload(
                pair_count=len(case_results),
                relative_l2=relative_summary,
                normalized_rgb8_mse=mse_summary,
                manifest_digest=binding.confirmation_manifest_digest,
                metric_registry_digest=binding.metric_registry_digest,
                case_digest=case_digest,
                source_cluster_digest=cluster_digest,
            )
        ),
    )
    validate_c1_hf_paired_rgb8_quality_result(result, binding)
    return result


def validate_c1_hf_paired_rgb8_quality_result(
    result: C1HfPairedRgb8QualityResult,
    binding: C1HfMetricImplementationBinding,
) -> None:
    if type(result) is not C1HfPairedRgb8QualityResult:
        raise C1HfMetricError("c1_quality_aggregate_exact_type_required")
    if (
        result.metric_id != "c1_hf_paired_final_image_quality"
        or result.split != "untouched_confirmation"
        or result.pair_count != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or len(result.case_results) != result.pair_count
        or result.manifest_digest != binding.confirmation_manifest_digest
        or result.metric_registry_digest != binding.metric_registry_digest
    ):
        raise C1HfMetricError("c1_quality_aggregate_identity_mismatch")
    for case_result in result.case_results:
        validate_c1_hf_quality_case_result(case_result, binding)
    _validate_exact_analysis_unit_set(
        [case_result.identity for case_result in result.case_results],
        binding,
        "untouched_confirmation",
    )
    expected_case_digest = _quality_case_digest(result.case_results)
    expected_cluster_digest = _canonical_digest(
        sorted(
            case_result.identity.source_cluster_id
            for case_result in result.case_results
        )
    )
    expected_relative = _quality_summary(
        "paired_rgb8_relative_l2",
        tuple(case_result.relative_l2 for case_result in result.case_results),
    )
    expected_mse = _quality_summary(
        "paired_rgb8_mse",
        tuple(
            case_result.normalized_rgb8_mse
            for case_result in result.case_results
        ),
    )
    if (
        result.case_digest != expected_case_digest
        or result.source_cluster_digest != expected_cluster_digest
        or result.relative_l2 != expected_relative
        or result.normalized_rgb8_mse != expected_mse
    ):
        raise C1HfMetricError("c1_quality_aggregate_formula_or_case_mismatch")
    expected = _canonical_digest(
        _quality_aggregate_identity_payload(
            pair_count=result.pair_count,
            relative_l2=result.relative_l2,
            normalized_rgb8_mse=result.normalized_rgb8_mse,
            manifest_digest=result.manifest_digest,
            metric_registry_digest=result.metric_registry_digest,
            case_digest=result.case_digest,
            source_cluster_digest=result.source_cluster_digest,
        )
    )
    if result.result_identity != expected:
        raise C1HfMetricError("c1_quality_aggregate_result_identity_mismatch")


@dataclass(frozen=True, slots=True)
class C1HfActualDtypeIntegrityCase:
    identity: C1HfMetricCaseIdentity
    registered_watermarked_image_digest: str
    materialization_identity_digest: str
    materialization_integrity_passed: bool
    runtime_dtype: str
    measurement_dtype: str
    actual_relative_l2: float

    def __post_init__(self) -> None:
        _require_digest(
            self.registered_watermarked_image_digest,
            "registered_watermarked_image_digest",
        )
        _require_digest(
            self.materialization_identity_digest,
            "materialization_identity_digest",
        )
        if type(self.materialization_integrity_passed) is not bool:
            raise C1HfMetricError("materialization_integrity_passed_invalid")
        _require_identity(self.runtime_dtype, "runtime_dtype")
        _require_identity(self.measurement_dtype, "measurement_dtype")
        if type(self.actual_relative_l2) is not float:
            raise C1HfMetricError("actual_relative_l2_must_be_float64_fact")


@dataclass(frozen=True, slots=True)
class C1HfActualDtypeIntegrityResult:
    metric_id: str
    split: str
    registered_positive_count: int
    case_facts: tuple[C1HfActualDtypeIntegrityCase, ...]
    materialization_integrity_failure_count: int
    runtime_dtype_failure_count: int
    measurement_dtype_failure_count: int
    non_finite_relative_l2_count: int
    negative_relative_l2_count: int
    budget_exceeded_count: int
    failed_case_count: int
    manifest_digest: str
    metric_registry_digest: str
    case_digest: str
    source_cluster_digest: str
    result_identity: str


def evaluate_c1_hf_actual_dtype_integrity(
    cases: Sequence[C1HfActualDtypeIntegrityCase],
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfActualDtypeIntegrityResult:
    if (
        isinstance(cases, (str, bytes))
        or not isinstance(cases, Sequence)
        or len(cases) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or any(type(case) is not C1HfActualDtypeIntegrityCase for case in cases)
    ):
        raise C1HfMetricError("c1_actual_dtype_requires_exact_4096_cases")
    for case in cases:
        _validate_case_identity(case.identity, binding, "untouched_confirmation")
    _validate_exact_analysis_unit_set(
        [case.identity for case in cases],
        binding,
        "untouched_confirmation",
    )
    cluster_ids = [case.identity.source_cluster_id for case in cases]
    integrity_failures = sum(
        not case.materialization_integrity_passed for case in cases
    )
    runtime_dtype_failures = sum(case.runtime_dtype != "float16" for case in cases)
    measurement_dtype_failures = sum(
        case.measurement_dtype != "float32" for case in cases
    )
    non_finite = sum(
        not math.isfinite(case.actual_relative_l2) for case in cases
    )
    negative = sum(
        math.isfinite(case.actual_relative_l2)
        and case.actual_relative_l2 < 0.0
        for case in cases
    )
    budget_exceeded = sum(
        math.isfinite(case.actual_relative_l2)
        and case.actual_relative_l2 > C1_HF_CONTENT_RELATIVE_L2_LIMIT
        for case in cases
    )
    failed_cases = sum(
        (
            not case.materialization_integrity_passed
            or case.runtime_dtype != "float16"
            or case.measurement_dtype != "float32"
            or not math.isfinite(case.actual_relative_l2)
            or case.actual_relative_l2 < 0.0
            or case.actual_relative_l2 > C1_HF_CONTENT_RELATIVE_L2_LIMIT
        )
        for case in cases
    )
    case_payload = [
        {
            "actual_relative_l2": (
                case.actual_relative_l2.hex()
                if math.isfinite(case.actual_relative_l2)
                else repr(case.actual_relative_l2)
            ),
            "materialization_identity_digest": (
                case.materialization_identity_digest
            ),
            "materialization_integrity_passed": (
                case.materialization_integrity_passed
            ),
            "measurement_dtype": case.measurement_dtype,
            "registered_watermarked_image_digest": (
                case.registered_watermarked_image_digest
            ),
            "runtime_dtype": case.runtime_dtype,
            "source_cluster_id": case.identity.source_cluster_id,
        }
        for case in sorted(cases, key=lambda item: item.identity.source_cluster_id)
    ]
    payload = {
        "metric_id": "c1_hf_actual_dtype_integrity",
        "split": "untouched_confirmation",
        "registered_positive_count": len(cases),
        "materialization_integrity_failure_count": integrity_failures,
        "runtime_dtype_failure_count": runtime_dtype_failures,
        "measurement_dtype_failure_count": measurement_dtype_failures,
        "non_finite_relative_l2_count": non_finite,
        "negative_relative_l2_count": negative,
        "budget_exceeded_count": budget_exceeded,
        "failed_case_count": failed_cases,
        "manifest_digest": binding.confirmation_manifest_digest,
        "metric_registry_digest": binding.metric_registry_digest,
        "case_digest": _canonical_digest(case_payload),
        "source_cluster_digest": _canonical_digest(sorted(cluster_ids)),
    }
    ordered_cases = tuple(
        sorted(cases, key=lambda item: item.identity.source_cluster_id)
    )
    result = C1HfActualDtypeIntegrityResult(
        **payload,
        case_facts=ordered_cases,
        result_identity=_canonical_digest(payload),
    )
    validate_c1_hf_actual_dtype_integrity_result(result, binding)
    return result


def validate_c1_hf_actual_dtype_integrity_result(
    result: C1HfActualDtypeIntegrityResult,
    binding: C1HfMetricImplementationBinding,
) -> None:
    if type(result) is not C1HfActualDtypeIntegrityResult:
        raise C1HfMetricError("c1_actual_dtype_result_exact_type_required")
    counts = (
        result.materialization_integrity_failure_count,
        result.runtime_dtype_failure_count,
        result.measurement_dtype_failure_count,
        result.non_finite_relative_l2_count,
        result.negative_relative_l2_count,
        result.budget_exceeded_count,
        result.failed_case_count,
    )
    if (
        result.metric_id != "c1_hf_actual_dtype_integrity"
        or result.split != "untouched_confirmation"
        or result.registered_positive_count != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or len(result.case_facts) != result.registered_positive_count
        or any(type(count) is not int or not 0 <= count <= result.registered_positive_count for count in counts)
        or result.manifest_digest != binding.confirmation_manifest_digest
        or result.metric_registry_digest != binding.metric_registry_digest
    ):
        raise C1HfMetricError("c1_actual_dtype_result_identity_or_count_mismatch")
    for case in result.case_facts:
        if type(case) is not C1HfActualDtypeIntegrityCase:
            raise C1HfMetricError("c1_actual_dtype_result_case_type_invalid")
        _validate_case_identity(case.identity, binding, "untouched_confirmation")
    _validate_exact_analysis_unit_set(
        [case.identity for case in result.case_facts],
        binding,
        "untouched_confirmation",
    )
    recomputed_counts = (
        sum(not case.materialization_integrity_passed for case in result.case_facts),
        sum(case.runtime_dtype != "float16" for case in result.case_facts),
        sum(case.measurement_dtype != "float32" for case in result.case_facts),
        sum(
            not math.isfinite(case.actual_relative_l2)
            for case in result.case_facts
        ),
        sum(
            math.isfinite(case.actual_relative_l2)
            and case.actual_relative_l2 < 0.0
            for case in result.case_facts
        ),
        sum(
            math.isfinite(case.actual_relative_l2)
            and case.actual_relative_l2 > C1_HF_CONTENT_RELATIVE_L2_LIMIT
            for case in result.case_facts
        ),
        sum(
            (
                not case.materialization_integrity_passed
                or case.runtime_dtype != "float16"
                or case.measurement_dtype != "float32"
                or not math.isfinite(case.actual_relative_l2)
                or case.actual_relative_l2 < 0.0
                or case.actual_relative_l2
                > C1_HF_CONTENT_RELATIVE_L2_LIMIT
            )
            for case in result.case_facts
        ),
    )
    if counts != recomputed_counts:
        raise C1HfMetricError("c1_actual_dtype_result_counts_mismatch")
    recomputed_case_payload = [
        {
            "actual_relative_l2": (
                case.actual_relative_l2.hex()
                if math.isfinite(case.actual_relative_l2)
                else repr(case.actual_relative_l2)
            ),
            "materialization_identity_digest": (
                case.materialization_identity_digest
            ),
            "materialization_integrity_passed": (
                case.materialization_integrity_passed
            ),
            "measurement_dtype": case.measurement_dtype,
            "registered_watermarked_image_digest": (
                case.registered_watermarked_image_digest
            ),
            "runtime_dtype": case.runtime_dtype,
            "source_cluster_id": case.identity.source_cluster_id,
        }
        for case in sorted(
            result.case_facts,
            key=lambda item: item.identity.source_cluster_id,
        )
    ]
    if (
        result.case_digest != _canonical_digest(recomputed_case_payload)
        or result.source_cluster_digest
        != _canonical_digest(
            sorted(case.identity.source_cluster_id for case in result.case_facts)
        )
    ):
        raise C1HfMetricError("c1_actual_dtype_result_case_digest_mismatch")
    for value, role in (
        (result.case_digest, "case_digest"),
        (result.source_cluster_digest, "source_cluster_digest"),
        (result.result_identity, "actual_dtype_result_identity"),
    ):
        _require_digest(value, role)
    payload = {
        key: getattr(result, key)
        for key in result.__dataclass_fields__
        if key not in {"case_facts", "result_identity"}
    }
    if result.result_identity != _canonical_digest(payload):
        raise C1HfMetricError("c1_actual_dtype_result_identity_mismatch")


@dataclass(frozen=True, slots=True)
class C1HfConfirmationInputBundle:
    fit_primary_null_cases: tuple[C1HfScoreCase, ...]
    threshold: C1HfThresholdResult
    score_cases: tuple[C1HfScoreCase, ...]
    raw_rgb8_quality_artifacts: tuple[C1HfRawRgb8QualityArtifact, ...]
    actual_dtype_cases: tuple[C1HfActualDtypeIntegrityCase, ...]


@dataclass(frozen=True, slots=True)
class C1HfConfirmationInputValidation:
    source_cluster_count: int
    fit_score_case_count: int
    score_case_count: int
    raw_rgb8_quality_artifact_count: int
    actual_dtype_case_count: int
    cross_input_digest: str


def _replay_c1_hf_confirmation_input_bundle(
    inputs: C1HfConfirmationInputBundle,
    binding: C1HfMetricImplementationBinding,
) -> tuple[
    C1HfConfirmationInputValidation,
    tuple[C1HfQualityCaseResult, ...],
]:
    """Replay fit and raw RGB8 evidence, then cross-bind all confirmation facts."""

    if type(inputs) is not C1HfConfirmationInputBundle:
        raise C1HfMetricError("c1_confirmation_input_bundle_exact_type_required")
    recomputed_threshold = fit_c1_hf_tau(
        inputs.fit_primary_null_cases,
        binding=binding,
    )
    if inputs.threshold != recomputed_threshold:
        raise C1HfMetricError("c1_confirmation_threshold_fit_replay_mismatch")
    grouped_scores, _, _ = _prepare_confirmation_scores(
        inputs.score_cases,
        recomputed_threshold,
        binding,
    )
    if (
        len(inputs.raw_rgb8_quality_artifacts)
        != C1_HF_SOURCE_CLUSTERS_PER_SPLIT
        or any(
            type(artifact) is not C1HfRawRgb8QualityArtifact
            for artifact in inputs.raw_rgb8_quality_artifacts
        )
    ):
        raise C1HfMetricError(
            "c1_confirmation_raw_rgb8_quality_artifact_count_or_type_mismatch"
        )
    if len(inputs.actual_dtype_cases) != C1_HF_SOURCE_CLUSTERS_PER_SPLIT:
        raise C1HfMetricError("c1_confirmation_actual_dtype_case_count_mismatch")
    artifact_cache: dict[tuple[object, ...], C1Rgb8Image] = {}
    quality_case_results = tuple(
        _replay_c1_hf_raw_rgb8_quality_artifact(
            artifact,
            binding,
            artifact_cache,
        )
        for artifact in inputs.raw_rgb8_quality_artifacts
    )
    for case in inputs.actual_dtype_cases:
        _validate_case_identity(case.identity, binding, "untouched_confirmation")
    _validate_exact_analysis_unit_set(
        [result.identity for result in quality_case_results],
        binding,
        "untouched_confirmation",
    )
    _validate_exact_analysis_unit_set(
        [case.identity for case in inputs.actual_dtype_cases],
        binding,
        "untouched_confirmation",
    )
    primary_by_cluster = {
        case.identity.source_cluster_id: case
        for case in grouped_scores[C1_HF_PRIMARY_NULL_ROLE]
    }
    registered_by_cluster = {
        case.identity.source_cluster_id: case
        for case in grouped_scores[C1_HF_REGISTERED_ROLE]
    }
    quality_by_cluster = {
        result.identity.source_cluster_id: result
        for result in quality_case_results
    }
    actual_by_cluster = {
        case.identity.source_cluster_id: case
        for case in inputs.actual_dtype_cases
    }
    cross_payload = []
    for cluster_id in sorted(primary_by_cluster):
        primary = primary_by_cluster[cluster_id]
        registered = registered_by_cluster[cluster_id]
        quality = quality_by_cluster[cluster_id]
        actual = actual_by_cluster[cluster_id]
        if (
            primary.identity != quality.identity
            or registered.identity != quality.identity
            or actual.identity != quality.identity
            or primary.image_digest != quality.clean_image_digest
            or registered.image_digest
            != quality.registered_watermarked_image_digest
            or actual.registered_watermarked_image_digest
            != quality.registered_watermarked_image_digest
        ):
            raise C1HfMetricError(
                "c1_confirmation_score_quality_actual_cross_binding_mismatch"
            )
        cross_payload.append(
            {
                "analysis_unit_identity": asdict(
                    quality.identity.analysis_unit_identity
                ),
                "actual_materialization_identity_digest": (
                    actual.materialization_identity_digest
                ),
                "clean_image_digest": quality.clean_image_digest,
                "marked_image_digest": (
                    quality.registered_watermarked_image_digest
                ),
                "quality_result_identity": quality.result_identity,
            }
        )
    validation = C1HfConfirmationInputValidation(
        source_cluster_count=C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
        fit_score_case_count=len(inputs.fit_primary_null_cases),
        score_case_count=len(inputs.score_cases),
        raw_rgb8_quality_artifact_count=len(
            inputs.raw_rgb8_quality_artifacts
        ),
        actual_dtype_case_count=len(inputs.actual_dtype_cases),
        cross_input_digest=_canonical_digest(
            {
                "fit_case_digest": recomputed_threshold.case_digest,
                "threshold_identity": recomputed_threshold.threshold_identity,
                "confirmation_cases": cross_payload,
            }
        ),
    )
    return validation, quality_case_results


def validate_c1_hf_confirmation_input_bundle(
    inputs: C1HfConfirmationInputBundle,
    binding: C1HfMetricImplementationBinding,
) -> C1HfConfirmationInputValidation:
    """Independently replay every authority-bearing formal input."""

    validation, _ = _replay_c1_hf_confirmation_input_bundle(inputs, binding)
    return validation


@dataclass(frozen=True, slots=True)
class C1HfConfirmationMetricResults:
    primary_null_fixed_fpr: C1HfBinomialMetricResult
    registered_tpr: C1HfBinomialMetricResult
    wrong_key_false_accept: C1HfBinomialMetricResult
    paired_key_attribution: C1HfBinomialMetricResult
    paired_rgb8_quality: C1HfPairedRgb8QualityResult
    actual_dtype_integrity: C1HfActualDtypeIntegrityResult
    cross_input_digest: str
    result_identity: str


def evaluate_c1_hf_confirmation_metrics(
    inputs: C1HfConfirmationInputBundle,
    *,
    binding: C1HfMetricImplementationBinding,
) -> C1HfConfirmationMetricResults:
    """Formal confirmation entrypoint; cross-table validation is mandatory."""

    cross_validation, quality_case_results = (
        _replay_c1_hf_confirmation_input_bundle(inputs, binding)
    )
    primary = evaluate_c1_hf_primary_null_fixed_fpr(
        inputs.score_cases,
        inputs.threshold,
        binding=binding,
    )
    registered = evaluate_c1_hf_registered_tpr(
        inputs.score_cases,
        inputs.threshold,
        binding=binding,
    )
    wrong = evaluate_c1_hf_wrong_key_false_accept(
        inputs.score_cases,
        inputs.threshold,
        binding=binding,
    )
    attribution = evaluate_c1_hf_paired_key_attribution(
        inputs.score_cases,
        inputs.threshold,
        binding=binding,
    )
    quality = evaluate_c1_hf_paired_rgb8_quality(
        quality_case_results,
        binding=binding,
    )
    actual = evaluate_c1_hf_actual_dtype_integrity(
        inputs.actual_dtype_cases,
        binding=binding,
    )
    identity_payload = {
        "actual_dtype_integrity": actual.result_identity,
        "cross_input_digest": cross_validation.cross_input_digest,
        "paired_rgb8_quality": quality.result_identity,
        "paired_key_attribution": attribution.result_identity,
        "primary_null_fixed_fpr": primary.result_identity,
        "registered_tpr": registered.result_identity,
        "wrong_key_false_accept": wrong.result_identity,
    }
    result = C1HfConfirmationMetricResults(
        primary_null_fixed_fpr=primary,
        registered_tpr=registered,
        wrong_key_false_accept=wrong,
        paired_key_attribution=attribution,
        paired_rgb8_quality=quality,
        actual_dtype_integrity=actual,
        cross_input_digest=cross_validation.cross_input_digest,
        result_identity=_canonical_digest(identity_payload),
    )
    return result


def validate_c1_hf_confirmation_metric_results(
    result: C1HfConfirmationMetricResults,
    inputs: C1HfConfirmationInputBundle,
    binding: C1HfMetricImplementationBinding,
) -> None:
    """Recompute the formal composite from its bound cross-table inputs."""

    if type(result) is not C1HfConfirmationMetricResults:
        raise C1HfMetricError("c1_confirmation_metric_results_exact_type_required")
    cross_validation, quality_case_results = (
        _replay_c1_hf_confirmation_input_bundle(inputs, binding)
    )
    expected_children = (
        evaluate_c1_hf_primary_null_fixed_fpr(
            inputs.score_cases,
            inputs.threshold,
            binding=binding,
        ),
        evaluate_c1_hf_registered_tpr(
            inputs.score_cases,
            inputs.threshold,
            binding=binding,
        ),
        evaluate_c1_hf_wrong_key_false_accept(
            inputs.score_cases,
            inputs.threshold,
            binding=binding,
        ),
        evaluate_c1_hf_paired_key_attribution(
            inputs.score_cases,
            inputs.threshold,
            binding=binding,
        ),
        evaluate_c1_hf_paired_rgb8_quality(
            quality_case_results,
            binding=binding,
        ),
        evaluate_c1_hf_actual_dtype_integrity(
            inputs.actual_dtype_cases,
            binding=binding,
        ),
    )
    actual_children = (
        result.primary_null_fixed_fpr,
        result.registered_tpr,
        result.wrong_key_false_accept,
        result.paired_key_attribution,
        result.paired_rgb8_quality,
        result.actual_dtype_integrity,
    )
    if (
        actual_children != expected_children
        or result.cross_input_digest != cross_validation.cross_input_digest
    ):
        raise C1HfMetricError("c1_confirmation_metric_results_input_mismatch")
    identity_payload = {
        "actual_dtype_integrity": result.actual_dtype_integrity.result_identity,
        "cross_input_digest": result.cross_input_digest,
        "paired_rgb8_quality": (
            result.paired_rgb8_quality.result_identity
        ),
        "paired_key_attribution": result.paired_key_attribution.result_identity,
        "primary_null_fixed_fpr": result.primary_null_fixed_fpr.result_identity,
        "registered_tpr": result.registered_tpr.result_identity,
        "wrong_key_false_accept": result.wrong_key_false_accept.result_identity,
    }
    if result.result_identity != _canonical_digest(identity_payload):
        raise C1HfMetricError("c1_confirmation_metric_results_identity_mismatch")
