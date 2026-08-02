"""CPU constraints for the development-only 13-duty exploration protocol."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.protocol.development_exploration import (
    BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT,
    CANDIDATE_RECOMMENDATIONS,
    CHEAP_DETECTION_RESPONSIBILITIES,
    CONTENT_BRANCH_IDS,
    CRITICAL_PAIR_RESPONSIBILITIES,
    DEPENDENCY_STOP_RULE,
    DEVELOPMENT_SPLIT,
    DEVELOPMENT_THRESHOLD_INPUT_ROLES,
    DEVELOPMENT_THRESHOLD_ROLE,
    FORMAL_LATER_SPLIT_DENY_LIST,
    GEOMETRY_NEGATIVE_CONTROL_CASE_IDS,
    GEOMETRY_OPERATION_FAMILIES,
    ISOLATION_DIMENSIONS,
    MODULE_CANDIDATE_IDS,
    MODULE_CANDIDATE_PARAMETERS,
    MODULE_METRIC_IDS,
    MODULE_NEGATIVE_CONTROL_CASE_IDS,
    MODULE_OUTCOMES,
    PREFLIGHT_SOURCE_CLUSTER_COUNT,
    REGISTERED_STUDY_ROLE_BINDINGS,
    WIRING_SOURCE_CLUSTER_COUNT,
    DevelopmentPrimaryNullKeyBinding,
    DevelopmentThresholdFitInput,
    assert_study_role_manifests_isolated,
    authorize_development_provisional_threshold,
    bind_study_role_manifest,
    build_development_cross_fit_plan,
    create_development_module_outcome_record,
    create_development_provisional_threshold,
    create_development_threshold_detector_binding,
    create_development_threshold_fit_input,
    decide_development_module_execution,
    development_assignments_only,
    enumerate_development_study_units,
    load_frozen_development_exploration_protocol,
)
from experiments.protocol.internal_matrix import REQUIRED_METHOD_RESPONSIBILITIES
from experiments.protocol.internal_records import (
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    MAXIMUM_RECORD_ATTEMPTS,
    BranchScoreTrace,
    DecisionTrace,
    DetectorTrace,
    GeometryTrace,
    InternalValidationRecord,
    KeyControlTrace,
    ProvenanceTrace,
    RoutingTrace,
    ThresholdTrace,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    SplitAssignment,
    derive_source_cluster_id,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/development_module_exploration.json"
PROTOCOL_PATH = ROOT / "experiments/protocol/development_exploration.py"


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


def _recompute_candidate_config_digest(entry: dict[str, object]) -> None:
    entry["candidate_config_digest"] = _digest(
        {
            "candidate_identity": entry["candidate_identity"],
            "candidate_ids": entry["candidate_ids"],
            "candidate_parameter_bindings": tuple(
                (
                    item["parameter_id"],
                    tuple(item["values"]),
                )
                for item in entry["candidate_parameter_bindings"]
            ),
            "content_branch_ids": entry["content_branch_ids"],
            "geometry_case_ids": entry["geometry_case_ids"],
            "metric_ids": entry["metric_ids"],
            "negative_control_case_ids": entry["negative_control_case_ids"],
            "paired_ablation_identity": entry["paired_ablation_identity"],
            "responsibility_id": entry["responsibility_id"],
        }
    )


def _redigest_fit_input(
    fit_input: DevelopmentThresholdFitInput,
    source_record: InternalValidationRecord,
) -> DevelopmentThresholdFitInput:
    return replace(
        fit_input,
        source_record=source_record,
        source_record_digest=_digest(
            {
                "case_role": fit_input.case_role,
                "source_record": asdict(source_record),
            }
        ),
    )


def _unit(
    index: int,
    *,
    split: str = DEVELOPMENT_SPLIT,
    case_id: str = "development_primary_null_threshold_fit",
) -> SplitAssignment:
    prompt_digest = f"{index + 1:064x}"
    lineage_digest = f"{index + 1001:064x}"
    key_family_digest = f"{index + 2001:064x}"
    identity = AnalysisUnitIdentity(
        unit_id=f"development_unit_{index}",
        case_id=case_id,
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=prompt_digest,
            generation_seed=index,
            image_lineage_digest=lineage_digest,
            registered_key_family_digest=key_family_digest,
        ),
        prompt_digest=prompt_digest,
        generation_seed=index,
        image_lineage_digest=lineage_digest,
        registered_key_family_digest=key_family_digest,
    )
    return SplitAssignment(identity=identity, split=split)


def _manifest(
    count: int,
    *,
    split: str = DEVELOPMENT_SPLIT,
    start: int = 0,
    case_id: str = "development_primary_null_threshold_fit",
) -> FrozenSplitManifest:
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id=f"{split}_module_exploration_fixture",
        manifest_revision="fixture_revision",
        assignments=tuple(
            _unit(index, split=split, case_id=case_id)
            for index in range(start, start + count)
        ),
    )


def _development_manifest(count: int) -> FrozenSplitManifest:
    return _manifest(count)


def _primary_null_record(
    assignment: SplitAssignment,
    *,
    index: int,
    score: float,
    split_manifest_digest: str,
    detector_binding,
) -> InternalValidationRecord:
    key_binding = next(
        item
        for item in detector_binding.primary_null_key_bindings
        if item.source_cluster_id == assignment.identity.source_cluster_id
    )
    return InternalValidationRecord(
        record_id=f"primary_null_score_record_{index}",
        run_id="development_threshold_fit_run",
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        record_schema_version=INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
        analysis_unit_identity=assignment.identity,
        split=assignment.split,
        record_sequence_index=index,
        record_attempt_index=0,
        execution_status="success",
        failure_class=None,
        failure_reason=None,
        exclusion_reason=None,
        exclusion_rule_id=None,
        retry_of_record_id=None,
        detector_trace=DetectorTrace(
            raw_detector_identity="development_blind_high_frequency_detector",
            rectified_detector_identity="development_blind_high_frequency_detector",
            raw_detector_config_digest=detector_binding.detector_config_digest,
            rectified_detector_config_digest=detector_binding.detector_config_digest,
            raw_preprocessing_identity=detector_binding.preprocessing_identity,
            rectified_preprocessing_identity=detector_binding.preprocessing_identity,
            raw_content_score=score,
            rectified_content_score=None,
        ),
        branch_score_trace=BranchScoreTrace(
            lf_score=None,
            hf_score=score,
            combined_score=None,
        ),
        routing_trace=RoutingTrace(
            routing_identity="routing_not_applicable",
            routing_control="primary_null",
            routing_observation_digest="1" * 64,
            routing_mask_digest="2" * 64,
        ),
        geometry_trace=GeometryTrace(
            geometry_triggered=False,
            geometry_operation_identity="geometry_not_attempted",
            geometry_reliability_config_digest=None,
            geometry_estimation_identity=None,
            geometry_reliability_identity=None,
            geometry_reliable=None,
            geometry_transform=None,
            geometry_raw_metrics=None,
            geometry_failure_reason=None,
            rectification_status="not_attempted",
        ),
        threshold_trace=ThresholdTrace(
            raw_threshold_identity="development_record_collection_threshold",
            rectified_threshold_identity="development_record_collection_threshold",
            tau=10.0,
            tau_rescue=9.0,
        ),
        key_control_trace=KeyControlTrace(
            registered_key_public_digest=key_binding.registered_key_public_digest,
            detection_key_public_digest=key_binding.detection_key_public_digest,
            key_role="unwatermarked_primary_null",
            control_identity="primary_null",
        ),
        decision_trace=DecisionTrace(
            watermark_decision="negative",
            positive_source=None,
            decision_reason="primary_null_below_development_record_threshold",
        ),
        provenance_trace=ProvenanceTrace(
            protocol_digest="5" * 64,
            split_manifest_digest=split_manifest_digest,
            input_manifest_digest="6" * 64,
            method_code_revision="development_method_revision",
            candidate_config_digest="7" * 64,
            method_config_digest="8" * 64,
            execution_config_digest="9" * 64,
            model_revision="frozen_model_revision",
            environment_digest="a" * 64,
            resource_identity_digest="b" * 64,
            input_artifact_digest="c" * 64,
            attack_config_digest="d" * 64,
            metric_set_digest="e" * 64,
        ),
    )


def _cross_fit_plan(cluster_count: int = 16):
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    manifest = _development_manifest(cluster_count)
    return build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        assignments=development_assignments_only(
            manifest,
            protocol=protocol,
            seed_namespace="development_exploration_seed_namespace",
        ),
        expected_source_cluster_count=cluster_count,
    )


def _threshold_material(plan, fold_index: int = 0, manifest=None):
    manifest = (
        _development_manifest(plan.source_cluster_count)
        if manifest is None
        else manifest
    )
    fit = set(plan.folds[fold_index].fit_source_cluster_ids)
    key_bindings = tuple(
        DevelopmentPrimaryNullKeyBinding(
            source_cluster_id=assignment.identity.source_cluster_id,
            registered_key_family_digest=(
                assignment.identity.registered_key_family_digest
            ),
            registered_key_public_digest=_digest(
                {
                    "key_family": assignment.identity.registered_key_family_digest,
                    "role": "registered_public_key",
                }
            ),
            detection_key_public_digest=_digest(
                {
                    "key_family": assignment.identity.registered_key_family_digest,
                    "role": "development_primary_null_detection_key",
                }
            ),
        )
        for assignment in manifest.assignments
        if assignment.identity.source_cluster_id in fit
    )
    detector_binding = create_development_threshold_detector_binding(
        plan,
        fold_index=fold_index,
        input_manifest=manifest,
        detector_identity="development_blind_high_frequency_detector",
        preprocessing_identity="frozen_public_preprocess",
        public_key_relation="registered_detection_public_digests_distinct",
        primary_null_key_bindings=key_bindings,
        detector_base_config_payload={"detector_mode": "hf_only"},
    )
    fit_inputs = tuple(
        create_development_threshold_fit_input(
            source_record=_primary_null_record(
                assignment,
                index=index,
                score=float(index) / 100.0,
                split_manifest_digest=manifest.digest(),
                detector_binding=detector_binding,
            ),
        )
        for index, assignment in enumerate(manifest.assignments)
        if assignment.identity.source_cluster_id in fit
    )
    return manifest, detector_binding, fit_inputs


def _valid_fit_inputs(plan, fold_index: int = 0):
    return _threshold_material(plan, fold_index)[2]


def _create_threshold(plan, fit_inputs=None):
    manifest, detector_binding, default_fit_inputs = _threshold_material(plan)
    return create_development_provisional_threshold(
        plan,
        fold_index=0,
        input_manifest=manifest,
        detector_binding=detector_binding,
        fit_inputs=default_fit_inputs if fit_inputs is None else fit_inputs,
    )


@pytest.mark.unit
def test_protocol_freezes_exact_thirteen_module_scientific_structure() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert protocol.validate() == ()
    assert tuple(item.responsibility_id for item in protocol.module_matrix) == (
        REQUIRED_METHOD_RESPONSIBILITIES
    )
    for field_name in (
        "scientific_question_id",
        "development_case_id",
        "candidate_selection_case_id",
        "candidate_identity",
        "candidate_config_digest",
        "paired_ablation_identity",
    ):
        values = tuple(getattr(item, field_name) for item in protocol.module_matrix)
        assert all(values)
        assert len(values) == len(set(values)) == 13
    for item in protocol.module_matrix:
        assert item.candidate_ids == MODULE_CANDIDATE_IDS[item.responsibility_id]
        assert item.candidate_parameter_bindings == MODULE_CANDIDATE_PARAMETERS[
            item.responsibility_id
        ]
        assert item.negative_control_case_ids == MODULE_NEGATIVE_CONTROL_CASE_IDS[
            item.responsibility_id
        ]
        assert item.metric_ids == MODULE_METRIC_IDS[item.responsibility_id]
        assert item.record_field_names
        assert item.dependency_stop_rule == DEPENDENCY_STOP_RULE
        assert item.allowed_module_outcomes == MODULE_OUTCOMES
        assert len(item.candidate_config_digest) == 64


@pytest.mark.unit
def test_content_and_geometry_families_are_finite_and_semantic() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert protocol.content_study.branch_ids == CONTENT_BRANCH_IDS == (
        "clean_control",
        "hf_only",
        "lf_only",
        "lf_hf_disabled_uniform_control",
        "lf_hf_routed_combination",
    )
    assert tuple(
        item.operation_family for item in protocol.geometry_study.operation_cases
    ) == GEOMETRY_OPERATION_FAMILIES
    assert len(protocol.geometry_study.operation_cases) == 5
    assert protocol.geometry_study.negative_control_case_ids == (
        GEOMETRY_NEGATIVE_CONTROL_CASE_IDS
    )
    assert set(GEOMETRY_NEGATIVE_CONTROL_CASE_IDS) == {
        "ambiguous_transform_control",
        "boundary_transform_control",
        "extreme_crop_control",
    }


@pytest.mark.unit
def test_preflight_and_wiring_counts_are_not_scientific_coverage() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert protocol.preflight.source_cluster_count == PREFLIGHT_SOURCE_CLUSTER_COUNT
    assert protocol.preflight.counts_as_scientific_coverage is False
    assert protocol.study_budget.wiring_source_cluster_count == (
        WIRING_SOURCE_CLUSTER_COUNT
    )
    assert protocol.study_budget.wiring_counts_as_scientific_coverage is False
    with pytest.raises(
        ValueError,
        match="wiring_clusters_do_not_count_as_scientific_coverage",
    ):
        build_development_cross_fit_plan(
            responsibility_id="key_schedule",
            assignments=development_assignments_only(
                _development_manifest(WIRING_SOURCE_CLUSTER_COUNT),
                protocol=protocol,
                seed_namespace="development_exploration_seed_namespace",
            ),
            expected_source_cluster_count=WIRING_SOURCE_CLUSTER_COUNT,
        )


@pytest.mark.unit
def test_study_roster_is_breadth_first_enumerable_and_budget_bounded() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    roster = enumerate_development_study_units(protocol)
    budget = protocol.study_budget
    assert len(roster) == budget.maximum_scientific_units == 2512
    assert budget.maximum_total_record_attempts == sum(
        unit.maximum_record_attempts for unit in roster
    ) == 7536
    assert len(
        {
            (
                unit.responsibility_id,
                unit.source_cluster_ordinal,
                unit.content_branch_id,
                unit.geometry_case_id,
            )
            for unit in roster
        }
    ) == len(roster)
    assert tuple(
        unit.responsibility_id for unit in roster[:13]
    ) == REQUIRED_METHOD_RESPONSIBILITIES
    assert all(
        unit.source_cluster_ordinal < BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT
        for unit in roster[: 13 * BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT]
    )
    counts = {
        responsibility: len(
            {
                unit.source_cluster_ordinal
                for unit in roster
                if unit.responsibility_id == responsibility
            }
        )
        for responsibility in REQUIRED_METHOD_RESPONSIBILITIES
    }
    assert all(counts[item] == 32 for item in CRITICAL_PAIR_RESPONSIBILITIES)
    assert all(counts[item] == 64 for item in CHEAP_DETECTION_RESPONSIBILITIES)
    assert all(
        count == 16
        for responsibility, count in counts.items()
        if responsibility
        not in {*CRITICAL_PAIR_RESPONSIBILITIES, *CHEAP_DETECTION_RESPONSIBILITIES}
    )
    assert budget.score_adaptive_unit_changes_forbidden is True


@pytest.mark.unit
def test_split_isolation_binds_real_manifests_and_exact_role_mapping() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    isolation = protocol.split_isolation
    assert isolation.isolation_dimensions == ISOLATION_DIMENSIONS
    observed = tuple(
        (
            item.role_id,
            item.registered_split,
            item.detector_mode,
            item.requires_frozen_hf_only_tau,
        )
        for item in isolation.role_bindings
    )
    assert observed == REGISTERED_STUDY_ROLE_BINDINGS
    assert isolation.cross_role_identity_overlap_forbidden is True
    by_role = {item.role_id: item for item in isolation.role_bindings}
    assert by_role["candidate_selection_selection"].registered_split == (
        "candidate_selection"
    )
    assert by_role["content_candidate_confirmation"].detector_mode == "combined"
    assert by_role["hf_only_reference_confirmation"].detector_mode == "hf_only"
    assert by_role[
        "hf_only_reference_confirmation"
    ].requires_frozen_hf_only_tau is True
    assert all(
        not item.execution_allowed_in_development
        for item in isolation.role_bindings[1:]
    )
    development = bind_study_role_manifest(
        protocol,
        role_id="development_exploration",
        seed_namespace="development_exploration_seed_namespace",
        manifest=_manifest(16),
    )
    candidate_selection = bind_study_role_manifest(
        protocol,
        role_id="candidate_selection_selection",
        seed_namespace="candidate_selection_seed_namespace",
        manifest=_manifest(16, split="candidate_selection", start=100),
    )
    assert_study_role_manifests_isolated(
        protocol,
        (development, candidate_selection),
    )


@pytest.mark.unit
def test_split_isolation_rejects_reused_real_identity_across_roles() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    development_manifest = _manifest(16)
    reused_manifest = replace(
        development_manifest,
        manifest_id="candidate_selection_reused_identity_fixture",
        assignments=tuple(
            replace(assignment, split="candidate_selection")
            for assignment in development_manifest.assignments
        ),
    )
    development = bind_study_role_manifest(
        protocol,
        role_id="development_exploration",
        seed_namespace="development_exploration_seed_namespace",
        manifest=development_manifest,
    )
    candidate_selection = bind_study_role_manifest(
        protocol,
        role_id="candidate_selection_selection",
        seed_namespace="candidate_selection_seed_namespace",
        manifest=reused_manifest,
    )
    with pytest.raises(PermissionError, match="study_manifest_identity_overlap"):
        assert_study_role_manifests_isolated(
            protocol,
            (development, candidate_selection),
        )
    with pytest.raises(PermissionError, match="study_manifest_identity_overlap"):
        development_assignments_only(
            development_manifest,
            protocol=protocol,
            seed_namespace="development_exploration_seed_namespace",
            known_role_manifest_bindings=(candidate_selection,),
        )


@pytest.mark.unit
def test_development_split_surface_denies_every_formal_later_split() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    development = _development_manifest(16)
    assert development_assignments_only(
        development,
        protocol=protocol,
        seed_namespace="development_exploration_seed_namespace",
    ) == development.assignments
    for forbidden_split in FORMAL_LATER_SPLIT_DENY_LIST:
        contaminated = replace(
            development,
            assignments=(
                *development.assignments[:-1],
                _unit(100, split=forbidden_split),
            ),
        )
        with pytest.raises(ValueError, match="study_manifest_contains_wrong_split"):
            development_assignments_only(
                contaminated,
                protocol=protocol,
                seed_namespace="development_exploration_seed_namespace",
            )


@pytest.mark.unit
@pytest.mark.parametrize("cluster_count", (16, 32, 64))
def test_threshold_cross_fit_excludes_fit_clusters_from_recovery_probes(
    cluster_count: int,
) -> None:
    plan = _cross_fit_plan(cluster_count)
    assert plan.validate() == ()
    for fold in plan.folds:
        assert not (
            set(fold.fit_source_cluster_ids)
            & set(fold.recovery_probe_source_cluster_ids)
        )
    threshold = _create_threshold(plan)
    assert threshold.threshold_role == DEVELOPMENT_THRESHOLD_ROLE
    assert {item.case_role for item in threshold.fit_inputs} == set(
        DEVELOPMENT_THRESHOLD_INPUT_ROLES
    )
    probe = plan.folds[0].recovery_probe_source_cluster_ids[0]
    authorize_development_provisional_threshold(
        threshold,
        plan,
        requested_split=DEVELOPMENT_SPLIT,
        source_cluster_id=probe,
    )
    with pytest.raises(PermissionError, match="fold_leakage"):
        authorize_development_provisional_threshold(
            threshold,
            plan,
            requested_split=DEVELOPMENT_SPLIT,
            source_cluster_id=plan.folds[0].fit_source_cluster_ids[0],
        )


@pytest.mark.unit
@pytest.mark.parametrize("invalid_role", ("registered_positive", "clean_non_null"))
def test_threshold_rejects_non_primary_null_inputs(
    invalid_role: str,
) -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    invalid = (replace(valid[0], case_role=invalid_role), *valid[1:])
    with pytest.raises(ValueError, match="threshold_fit_input_role_invalid"):
        _create_threshold(plan, invalid)


@pytest.mark.unit
def test_threshold_rejects_non_development_input_and_probe_leakage() -> None:
    plan = _cross_fit_plan()
    fold = plan.folds[0]
    valid = _valid_fit_inputs(plan)
    invalid_split = (
        replace(
            valid[0],
            source_record=replace(
                valid[0].source_record,
                split="candidate_selection",
            ),
        ),
        *valid[1:],
    )
    with pytest.raises(ValueError, match="threshold_fit_input_split_invalid"):
        _create_threshold(plan, invalid_split)
    probe_assignment = next(
        assignment
        for assignment in _development_manifest(plan.source_cluster_count).assignments
        if assignment.identity.source_cluster_id
        == fold.recovery_probe_source_cluster_ids[0]
    )
    leaked = (
        replace(
            valid[0],
            source_record=replace(
                valid[0].source_record,
                analysis_unit_identity=probe_assignment.identity,
            ),
        ),
        *valid[1:],
    )
    with pytest.raises(ValueError, match="recovery_probe_leakage"):
        _create_threshold(plan, leaked)


@pytest.mark.unit
def test_threshold_rejects_manifest_case_role_spoof_and_wrong_key_fit() -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    spoofed_record = replace(
        valid[0].source_record,
        analysis_unit_identity=replace(
            valid[0].source_record.analysis_unit_identity,
            case_id="registered_positive_case",
        ),
    )
    spoofed = (replace(valid[0], source_record=spoofed_record), *valid[1:])
    with pytest.raises(ValueError, match="threshold_fit_input_case_identity_invalid"):
        _create_threshold(plan, spoofed)
    wrong_key = (replace(valid[0], case_role="wrong_key_control"), *valid[1:])
    with pytest.raises(ValueError, match="threshold_fit_input_role_invalid"):
        _create_threshold(plan, wrong_key)


@pytest.mark.unit
def test_registered_positive_manifest_cannot_be_relabelled_as_primary_null() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    manifest = _manifest(16, case_id="registered_positive_case")
    assignments = development_assignments_only(
        manifest,
        protocol=protocol,
        seed_namespace="development_exploration_seed_namespace",
    )
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        assignments=assignments,
        expected_source_cluster_count=16,
    )
    _, detector_binding, forged_inputs = _threshold_material(
        plan,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="threshold_fit_input_case_identity_invalid"):
        create_development_provisional_threshold(
            plan,
            fold_index=0,
            input_manifest=manifest,
            detector_binding=detector_binding,
            fit_inputs=forged_inputs,
        )


@pytest.mark.unit
def test_threshold_binds_manifest_detector_rule_and_fold() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    assert threshold.validate(plan) == ()
    detector_payload = json.loads(threshold.detector_config_payload_json)
    assert detector_payload["preprocessing_identity"] == "frozen_public_preprocess"
    assert detector_payload["public_key_relation"] == (
        "registered_detection_public_digests_distinct"
    )
    assert detector_payload["primary_null_key_roster_digest"] == (
        threshold.detector_binding.primary_null_key_roster_digest
    )
    for field_name, reason in (
        ("input_manifest_digest", "manifest_digest_invalid"),
        ("detector_config_digest", "detector_config_digest_invalid"),
        ("threshold_rule_digest", "rule_digest_invalid"),
    ):
        forged = replace(threshold, **{field_name: "not-a-digest"})
        assert any(reason in item for item in forged.validate(plan))
    assert "provisional_threshold_detector_identity_invalid" in replace(
        threshold,
        detector_identity="",
    ).validate(plan)
    forged = replace(threshold, threshold=threshold.threshold + 1.0)
    forged = replace(
        forged,
        threshold_identity=_digest(forged.payload_without_identity()),
    )
    assert "provisional_threshold_value_not_rule_derived" in forged.validate(plan)


@pytest.mark.unit
def test_threshold_rejects_preprocessing_drift_after_source_redigest() -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    source = valid[0].source_record
    detector = replace(
        source.detector_trace,
        raw_preprocessing_identity="alternate_public_preprocess",
        rectified_preprocessing_identity="alternate_public_preprocess",
    )
    forged = _redigest_fit_input(
        valid[0],
        replace(source, detector_trace=detector),
    )
    with pytest.raises(
        ValueError,
        match="threshold_fit_input_preprocessing_identity_mismatch",
    ):
        _create_threshold(plan, (forged, *valid[1:]))


@pytest.mark.unit
def test_threshold_rejects_public_key_remap_after_all_record_redigests() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    forged_inputs = []
    for index, fit_input in enumerate(threshold.fit_inputs):
        source = fit_input.source_record
        key_trace = replace(
            source.key_control_trace,
            registered_key_public_digest=f"{index + 40:064x}",
            detection_key_public_digest=f"{index + 80:064x}",
        )
        forged_inputs.append(
            _redigest_fit_input(
                fit_input,
                replace(source, key_control_trace=key_trace),
            )
        )
    forged = replace(threshold, fit_inputs=tuple(forged_inputs))
    forged = replace(
        forged,
        threshold_identity=_digest(forged.payload_without_identity()),
    )
    violations = forged.validate(plan)
    assert "threshold_fit_input_public_key_mapping_mismatch" in violations


@pytest.mark.unit
def test_threshold_rejects_cross_fit_preprocess_and_key_mismatch() -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    source = valid[-1].source_record
    forged_source = replace(
        source,
        detector_trace=replace(
            source.detector_trace,
            raw_preprocessing_identity="alternate_public_preprocess",
            rectified_preprocessing_identity="alternate_public_preprocess",
        ),
        key_control_trace=replace(
            source.key_control_trace,
            registered_key_public_digest="a" * 64,
            detection_key_public_digest="b" * 64,
        ),
    )
    forged = _redigest_fit_input(valid[-1], forged_source)
    with pytest.raises(ValueError) as captured:
        _create_threshold(plan, (*valid[:-1], forged))
    message = str(captured.value)
    assert "threshold_fit_input_preprocessing_identity_mismatch" in message
    assert "threshold_fit_input_public_key_mapping_mismatch" in message


@pytest.mark.unit
def test_threshold_detector_payload_tamper_fails_after_redigest() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    altered_payload = json.loads(threshold.detector_config_payload_json)
    altered_payload["preprocessing_identity"] = "alternate_public_preprocess"
    altered_json = json.dumps(
        altered_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    forged = replace(
        threshold,
        detector_config_payload_json=altered_json,
        detector_config_digest=_digest(altered_payload),
    )
    forged = replace(
        forged,
        threshold_identity=_digest(forged.payload_without_identity()),
    )
    violations = forged.validate(plan)
    assert "provisional_threshold_detector_config_binding_mismatch" in violations


@pytest.mark.unit
def test_threshold_is_invalid_for_every_later_split() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    probe = plan.folds[0].recovery_probe_source_cluster_ids[0]
    for forbidden_split in FORMAL_LATER_SPLIT_DENY_LIST:
        with pytest.raises(PermissionError, match="invalid_for_split"):
            authorize_development_provisional_threshold(
                threshold,
                plan,
                requested_split=forbidden_split,
                source_cluster_id=probe,
            )


@pytest.mark.unit
def test_dependency_stop_rule_uses_implementation_blocked_semantics() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    missing = decide_development_module_execution(
        protocol,
        "content_embedder",
        {},
    )
    assert not missing.approved
    assert missing.decision_reason == "prerequisite_outcome_missing"
    blocked = decide_development_module_execution(
        protocol,
        "content_embedder",
        {
            "content_router": "mechanism_signal_observed",
            "lf_carrier": "mechanism_signal_not_observed",
            "hf_carrier": "mechanism_signal_observed",
        },
    )
    assert not blocked.approved
    assert blocked.decision_reason == DEPENDENCY_STOP_RULE
    assert blocked.blocking_responsibilities == ("lf_carrier",)
    approved = decide_development_module_execution(
        protocol,
        "content_embedder",
        {
            "content_router": "mechanism_signal_observed",
            "lf_carrier": "mechanism_signal_observed",
            "hf_carrier": "mechanism_signal_observed",
        },
    )
    assert approved.approved


@pytest.mark.unit
def test_outcomes_and_recommendations_are_separate_exact_identities() -> None:
    assert MODULE_OUTCOMES == (
        "mechanism_signal_observed",
        "mechanism_signal_not_observed",
        "implementation_blocked",
        "resource_blocked",
    )
    assert CANDIDATE_RECOMMENDATIONS == (
        "candidate_worth_further_selection",
        "candidate_not_recommended_for_selection",
    )
    forbidden = {
        "closed_negative",
        "proceed_to_candidate_selection",
        "development_signal_observed",
        "development_dependency_blocked",
    }
    assert forbidden.isdisjoint(MODULE_OUTCOMES)
    assert forbidden.isdisjoint(CANDIDATE_RECOMMENDATIONS)


@pytest.mark.unit
@pytest.mark.parametrize("recommendation", CANDIDATE_RECOMMENDATIONS)
def test_observed_signal_allows_independent_candidate_recommendation(
    recommendation: str,
) -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    outcome = create_development_module_outcome_record(
        protocol,
        responsibility_id="hf_detector",
        module_outcome="mechanism_signal_observed",
        candidate_recommendation=recommendation,
        recommendation_reason="development records support this separate recommendation",
        evidence_record_ids=("record_for_high_frequency_detector",),
    )
    assert outcome.validate(protocol) == ()
    assert outcome.source_record_schema_version == (
        INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION
    )
    assert outcome.source_record_collection_schema_version == (
        INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
    )
    assert outcome.scientific_claims_supported is False


@pytest.mark.unit
def test_blocked_and_negative_outcomes_cannot_recommend_selection() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    with pytest.raises(
        ValueError,
        match="candidate_recommendation_not_supported_by_outcome",
    ):
        create_development_module_outcome_record(
            protocol,
            responsibility_id="hf_detector",
            module_outcome="mechanism_signal_not_observed",
            candidate_recommendation="candidate_worth_further_selection",
            recommendation_reason="forged recommendation",
            evidence_record_ids=("record",),
        )
    own_block = create_development_module_outcome_record(
        protocol,
        responsibility_id="key_schedule",
        module_outcome="implementation_blocked",
        candidate_recommendation="candidate_not_recommended_for_selection",
        recommendation_reason="key schedule implementation could not execute",
        blocking_responsibilities=("key_schedule",),
        evidence_record_ids=("key_schedule_implementation_failure_record",),
    )
    assert own_block.validate(protocol) == ()
    with pytest.raises(
        ValueError,
        match="implementation_blocking_responsibility_missing",
    ):
        create_development_module_outcome_record(
            protocol,
            responsibility_id="geometric_transform_estimator",
            module_outcome="implementation_blocked",
            candidate_recommendation="candidate_not_recommended_for_selection",
            recommendation_reason="dependency did not pass its development rule",
            evidence_record_ids=("record",),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation,expected_reason",
    (
        ("candidate_digest", "candidate_config_digest_invalid"),
        ("unregistered_candidate", "candidate_ids_unregistered"),
        ("unregistered_metric", "metric_ids_unregistered"),
        ("unregistered_control", "negative_controls_unregistered"),
        ("ratio_roster_drift", "candidate_parameters_unregistered"),
        ("missing_metric", "metric_ids_missing"),
        ("expanded_split", "development_allowed_split_invalid"),
        ("changed_budget", "maximum_scientific_units_invalid"),
        ("adaptive_order", "score_adaptive_unit_changes_must_be_forbidden"),
        ("missing_clean_branch", "clean_control_missing"),
        ("incomplete_geometry", "geometry_case_coverage_invalid"),
        ("role_split_rebound", "study_role_registered_binding_invalid"),
        ("outcome_expanded", "development_module_outcomes_invalid"),
        ("unknown_field", "keys_invalid"),
    ),
)
def test_config_loader_rejects_per_module_and_protocol_drift(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if mutation == "candidate_digest":
        document["module_matrix"][0]["candidate_config_digest"] = "0" * 64
    elif mutation == "unregistered_candidate":
        entry = document["module_matrix"][0]
        entry["candidate_ids"] = ["unregistered_key_candidate"]
        _recompute_candidate_config_digest(entry)
    elif mutation == "unregistered_metric":
        entry = document["module_matrix"][0]
        entry["metric_ids"] = ["unregistered_key_metric"]
        _recompute_candidate_config_digest(entry)
    elif mutation == "unregistered_control":
        entry = document["module_matrix"][0]
        entry["negative_control_case_ids"] = ["unregistered_key_control"]
        _recompute_candidate_config_digest(entry)
    elif mutation == "ratio_roster_drift":
        entry = document["module_matrix"][8]
        entry["candidate_parameter_bindings"][-1]["values"] = ["1/8"]
        _recompute_candidate_config_digest(entry)
    elif mutation == "missing_metric":
        document["module_matrix"][0]["metric_ids"] = []
    elif mutation == "expanded_split":
        document["split_policy"]["allowed_split"] = "candidate_selection"
    elif mutation == "changed_budget":
        document["study_budget"]["maximum_scientific_units"] += 1
    elif mutation == "adaptive_order":
        document["study_budget"]["score_adaptive_unit_changes_forbidden"] = False
    elif mutation == "missing_clean_branch":
        document["module_matrix"][1]["content_branch_ids"].remove("clean_control")
    elif mutation == "incomplete_geometry":
        document["module_matrix"][8]["geometry_case_ids"].pop()
    elif mutation == "role_split_rebound":
        document["split_isolation"]["role_bindings"][1]["registered_split"] = (
            "development"
        )
    elif mutation == "outcome_expanded":
        document["module_outcomes"]["allowed"].append("development_closed_negative")
    else:
        document["module_matrix"][0]["free_form_extension"] = "forbidden"
    path = tmp_path / "development_protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=expected_reason):
        load_frozen_development_exploration_protocol(path)


@pytest.mark.unit
def test_protocol_surface_has_no_method_runtime_runner_or_governance_dependency() -> None:
    source = PROTOCOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROTOCOL_PATH))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in (
            "main",
            "runtime",
            "experiments.methods",
            "experiments.attacks",
            "experiments.metrics",
            "experiments.runners",
            "governance",
        )
    )
    assert "write_text(" not in source
