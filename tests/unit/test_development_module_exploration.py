"""CPU constraints for the development-only 13-duty exploration protocol."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import experiments.protocol.development_exploration as development_protocol_module

from experiments.metrics.development_exploration import (
    DEVELOPMENT_METRIC_ROLE,
    METRIC_SCHEMA_VERSION,
)

from experiments.protocol.development_exploration import (
    BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT,
    CANDIDATE_RECOMMENDATIONS,
    CHEAP_DETECTION_RESPONSIBILITIES,
    CONTENT_BRANCH_IDS,
    CRITICAL_PAIR_RESPONSIBILITIES,
    DEPENDENCY_STOP_RULE,
    DEVELOPMENT_SPLIT,
    DEVELOPMENT_DEPENDENCY_LAYERS,
    DEVELOPMENT_THRESHOLD_INPUT_ROLES,
    DEVELOPMENT_THRESHOLD_ROLE,
    RECORD_COLLECTION_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    FORMAL_LATER_SPLIT_DENY_LIST,
    GEOMETRY_NEGATIVE_CONTROL_CASE_IDS,
    GEOMETRY_OPERATION_FAMILIES,
    ISOLATION_DIMENSIONS,
    MODULE_CANDIDATE_IDS,
    MODULE_CANDIDATE_PARAMETERS,
    MODULE_METRIC_IDS,
    MODULE_NEGATIVE_CONTROL_CASE_IDS,
    MODULE_OUTCOMES,
    OPERATIONAL_UNIT_PHASES,
    PREFLIGHT_SOURCE_CLUSTER_COUNT,
    REGISTERED_STUDY_ROLE_BINDINGS,
    WIRING_SOURCE_CLUSTER_COUNT,
    THIRTEEN_MODULE_MECHANISM_SCREENING_PROTOCOL_ID,
    DevelopmentPrimaryNullKeyBinding,
    DevelopmentModuleOutcomeRecord,
    DevelopmentThresholdFitInput,
    assert_study_role_manifests_isolated,
    authorize_development_provisional_threshold,
    bind_study_role_manifest,
    build_development_cross_fit_plan,
    create_development_provisional_threshold,
    create_development_threshold_detector_binding,
    create_development_threshold_fit_input,
    create_frozen_development_execution_intent_authority,
    decide_development_module_execution,
    development_assignments_only,
    derive_development_primary_null_key_family_digest,
    enumerate_development_study_units,
    load_frozen_development_exploration_protocol,
)
from experiments.protocol.internal_matrix import REQUIRED_METHOD_RESPONSIBILITIES
from main import identify_root_key
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    DEVELOPMENT_RECORD_COLLECTION_ROLE,
    RECORD_SCHEMA_VERSION as SCIENTIFIC_RECORD_SCHEMA_VERSION,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
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
MECHANISM_SCREENING_CONFIG_PATH = (
    ROOT / "configs/experiments/thirteen_module_mechanism_screening.json"
)
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
            "signal_criteria": tuple(
                (
                    item["metric_id"],
                    item["comparison"],
                    float(item["threshold"]),
                )
                for item in entry["signal_criteria"]
            ),
        }
    )


def _redigest_fit_input(
    fit_input: DevelopmentThresholdFitInput,
    source_record: DevelopmentScientificRecord,
) -> DevelopmentThresholdFitInput:
    return replace(
        fit_input,
        source_record=source_record,
        source_record_digest=_digest(
            {
                "case_role": fit_input.case_role,
                "expected_execution_intent_authority_digest": (
                    fit_input.expected_execution_intent_authority_digest
                ),
                "source_record": asdict(source_record),
            }
        ),
    )


def _redigest_scientific_record(
    record: DevelopmentScientificRecord,
    **changes: object,
) -> DevelopmentScientificRecord:
    changed = replace(record, **changes)
    return replace(
        changed,
        record_id=canonical_development_value_digest(changed.payload_without_record_id()),
    )


def _unit(
    index: int,
    *,
    threshold_authority,
    split: str = DEVELOPMENT_SPLIT,
    case_id: str = "development_primary_null_threshold_fit",
) -> SplitAssignment:
    prompt_digest = f"{index + 1:064x}"
    lineage_digest = f"{index + 1001:064x}"
    if split == DEVELOPMENT_SPLIT:
        registered_public, detection_public = _fixture_public_key_digests(
            prompt_digest
        )
    else:
        registered_public = _digest(
            {
                "prompt_digest": prompt_digest,
                "split": split,
                "role": "registered_public_key",
            }
        )
        detection_public = registered_public
    key_family_digest = derive_development_primary_null_key_family_digest(
        threshold_authority,
        registered_key_public_digest=registered_public,
        detection_key_public_digest=detection_public,
    )
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
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id=f"{split}_module_exploration_fixture",
        manifest_revision="fixture_revision",
        assignments=tuple(
            _unit(
                index,
                threshold_authority=protocol.threshold_detector_authority,
                split=split,
                case_id=case_id,
            )
            for index in range(start, start + count)
        ),
    )


def _development_manifest(count: int) -> FrozenSplitManifest:
    return _manifest(count)


def _fixture_public_key_digests(prompt_digest: str) -> tuple[str, str]:
    del prompt_digest
    registered = identify_root_key(
        "development-runner-cpu-wiring-key"
    ).root_key_public_digest
    return registered, registered


def _primary_null_record(
    assignment: SplitAssignment,
    *,
    index: int,
    score: float,
    split_manifest_digest: str,
    detector_binding,
) -> DevelopmentScientificRecord:
    key_binding = next(
        item
        for item in detector_binding.primary_null_key_bindings
        if item.source_cluster_id == assignment.identity.source_cluster_id
    )
    protocol = detector_binding.protocol
    study = next(
        item
        for item in protocol.module_matrix
        if item.responsibility_id == "hf_detector"
    )
    operation_payload = {"primary_null_score": score}
    metric_payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metric_role": DEVELOPMENT_METRIC_ROLE,
        "responsibility_id": "hf_detector",
        "source_cluster_id": assignment.identity.source_cluster_id,
        "registered_metric_ids": study.metric_ids,
        "candidate_config_digest": study.candidate_config_digest,
        "paired_ablation_identity": study.paired_ablation_identity,
        "content_branch_id": "hf_only",
        "geometry_case_id": "content_geometry_not_applicable",
        "sufficient_statistics": (
            ("primary_null_score", score),
            ("registered_score", score + 1.0),
            ("wrong_key_score", score - 1.0),
        ),
        "result_identity_digests": ("b" * 64,),
        "threshold_role": None,
        "threshold_identity": None,
        "threshold_fit_source_cluster_digest": None,
    }
    metric_payload["observation_digest"] = canonical_development_value_digest(
        metric_payload
    )
    record = DevelopmentScientificRecord(
        schema_version=SCIENTIFIC_RECORD_SCHEMA_VERSION,
        collection_role=DEVELOPMENT_RECORD_COLLECTION_ROLE,
        record_id="0" * 64,
        run_id=detector_binding.execution_intent_authority.run_id,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest=(
            detector_binding.expected_execution_intent_authority_digest
        ),
        method_code_revision="a" * 40,
        unit_index=index,
        phase="scientific_coverage",
        analysis_unit_identity=asdict(assignment.identity),
        responsibility_id="hf_detector",
        scientific_question_id=study.scientific_question_id,
        development_case_id=study.development_case_id,
        candidate_identity=study.candidate_identity,
        candidate_config_digest=study.candidate_config_digest,
        paired_ablation_identity=study.paired_ablation_identity,
        negative_control_case_ids=study.negative_control_case_ids,
        metric_ids=study.metric_ids,
        content_branch_id="hf_only",
        geometry_case_id="content_geometry_not_applicable",
        attempt_index=0,
        execution_status="success",
        failure_class=None,
        failure_reason=None,
        retry_parent_intent_digest=None,
        actual_elapsed_seconds=1.0,
        maximum_duration_seconds=3600,
        duration_limit_exceeded=False,
        operation_result_payload=operation_payload,
        operation_result_digest=canonical_development_value_digest(operation_payload),
        metric_observation=metric_payload,
        routing_trace={},
        branch_score_trace={"hf_score": score},
        detector_trace={
            "raw_detector_identity": protocol.threshold_detector_authority.method_detector_identity,
            "rectified_detector_identity": protocol.threshold_detector_authority.method_detector_identity,
            "primary_null_detector_identity": protocol.threshold_detector_authority.method_detector_identity,
            "raw_detector_config_digest": protocol.threshold_detector_authority.method_detector_config_digest,
            "rectified_detector_config_digest": protocol.threshold_detector_authority.method_detector_config_digest,
            "primary_null_detector_config_digest": protocol.threshold_detector_authority.method_detector_config_digest,
            "raw_preprocessing_identity": detector_binding.preprocessing_identity,
            "rectified_preprocessing_identity": detector_binding.preprocessing_identity,
            "primary_null_preprocessing_identity": detector_binding.preprocessing_identity,
        },
        geometry_trace={},
        threshold_trace={
            "raw_threshold_identity": "development_exploratory_threshold",
            "rectified_threshold_identity": "development_exploratory_threshold",
        },
        key_control_trace={
            "registered_key_public_digest": key_binding.registered_key_public_digest,
            "primary_null_detection_key_public_digest": key_binding.detection_key_public_digest,
            "primary_null_control_identity": "unwatermarked_registered_key_primary_null",
        },
        decision_trace={"positive_source": None},
        provenance_trace={
            "protocol_digest": protocol.digest(),
            "input_manifest_digest": split_manifest_digest,
            "execution_intent_authority_digest": (
                detector_binding.expected_execution_intent_authority_digest
            ),
            "method_code_revision": "a" * 40,
            "candidate_config_digest": study.candidate_config_digest,
        },
        module_outcome=None,
        candidate_recommendation=None,
        scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
    )
    return replace(
        record,
        record_id=canonical_development_value_digest(record.payload_without_record_id()),
    )


def _cross_fit_plan(cluster_count: int = 64):
    manifest = _development_manifest(cluster_count)
    intent = _execution_intent(manifest)
    return build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=cluster_count,
    )


def _execution_intent(
    manifest: FrozenSplitManifest,
    *,
    run_id: str = "development_threshold_fit_run",
):
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    return create_frozen_development_execution_intent_authority(
        protocol,
        run_id=run_id,
        seed_namespace="development_exploration_seed_namespace",
        input_manifest=manifest,
        public_key_roster=_full_key_roster(manifest),
    )


def _full_key_roster(
    manifest: FrozenSplitManifest,
) -> tuple[DevelopmentPrimaryNullKeyBinding, ...]:
    return tuple(
        DevelopmentPrimaryNullKeyBinding(
            source_cluster_id=assignment.identity.source_cluster_id,
            registered_key_family_digest=(
                assignment.identity.registered_key_family_digest
            ),
            registered_key_public_digest=_fixture_public_key_digests(
                assignment.identity.prompt_digest
            )[0],
            detection_key_public_digest=_fixture_public_key_digests(
                assignment.identity.prompt_digest
            )[1],
        )
        for assignment in manifest.assignments
    )


def _rebind_manifest_public_roster(
    manifest: FrozenSplitManifest,
    *,
    assignment_indexes: set[int] | None = None,
) -> tuple[FrozenSplitManifest, tuple[DevelopmentPrimaryNullKeyBinding, ...]]:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    authority = protocol.threshold_detector_authority
    indexes = (
        set(range(len(manifest.assignments)))
        if assignment_indexes is None
        else assignment_indexes
    )
    assignments = []
    roster = []
    existing_roster = {
        item.source_cluster_id: item for item in _full_key_roster(manifest)
    }
    for index, assignment in enumerate(manifest.assignments):
        identity = assignment.identity
        if index in indexes:
            registered_public = _digest(
                {"rebound_index": index, "role": "registered_public_key"}
            )
            detection_public = registered_public
            key_family = derive_development_primary_null_key_family_digest(
                authority,
                registered_key_public_digest=registered_public,
                detection_key_public_digest=detection_public,
            )
            identity = replace(
                identity,
                source_cluster_id=derive_source_cluster_id(
                    prompt_digest=identity.prompt_digest,
                    generation_seed=identity.generation_seed,
                    image_lineage_digest=identity.image_lineage_digest,
                    registered_key_family_digest=key_family,
                ),
                registered_key_family_digest=key_family,
            )
            key_binding = DevelopmentPrimaryNullKeyBinding(
                source_cluster_id=identity.source_cluster_id,
                registered_key_family_digest=key_family,
                registered_key_public_digest=registered_public,
                detection_key_public_digest=detection_public,
            )
        else:
            key_binding = existing_roster[identity.source_cluster_id]
        assignments.append(replace(assignment, identity=identity))
        roster.append(key_binding)
    rebound = replace(
        manifest,
        manifest_revision="public_roster_rebound_fixture_revision",
        assignments=tuple(assignments),
    )
    assert rebound.validate(require_all_splits=False) == ()
    return rebound, tuple(roster)


def _threshold_material(plan, fold_index: int = 0, manifest=None):
    manifest = (
        plan.input_manifest
        if manifest is None
        else manifest
    )
    fit = set(plan.folds[fold_index].fit_source_cluster_ids)
    key_bindings = tuple(
        item
        for item in plan.execution_intent_authority.public_key_roster
        if item.source_cluster_id in fit
    )
    detector_binding = create_development_threshold_detector_binding(
        plan,
        expected_execution_intent_authority_digest=(
            plan.expected_execution_intent_authority_digest
        ),
        fold_index=fold_index,
        input_manifest=manifest,
        primary_null_key_bindings=key_bindings,
    )
    fit_inputs = tuple(
        create_development_threshold_fit_input(
            expected_execution_intent_authority_digest=(
                plan.expected_execution_intent_authority_digest
            ),
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
        expected_execution_intent_authority_digest=(
            plan.expected_execution_intent_authority_digest
        ),
        fold_index=0,
        input_manifest=manifest,
        detector_binding=detector_binding,
        fit_inputs=default_fit_inputs if fit_inputs is None else fit_inputs,
    )


def _plan_identity(plan, source_cluster_id: str) -> AnalysisUnitIdentity:
    return next(
        assignment.identity
        for assignment in plan.input_manifest.assignments
        if assignment.identity.source_cluster_id == source_cluster_id
    )


@pytest.mark.unit
def test_protocol_freezes_exact_thirteen_module_scientific_structure() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert protocol.validate() == ()
    authority = protocol.threshold_detector_authority
    assert authority.responsibility_id == "hf_detector"
    assert authority.detector_mode == "hf_only"
    assert authority.preprocessing_identity == (
        "rgb8_public_image_float32_unit_interval"
    )
    assert authority.registered_candidate_ids == MODULE_CANDIDATE_IDS["hf_detector"]
    assert authority.registered_candidate_parameter_bindings == (
        MODULE_CANDIDATE_PARAMETERS["hf_detector"]
    )
    assert protocol.execution_intent_policy.authority_role == (
        "create_only_before_scientific_records"
    )
    assert protocol.execution_intent_policy.raw_secret_policy == (
        "raw_secret_prohibited"
    )
    assert protocol.execution_intent_policy.later_runner_must_pin_digest is True
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
    studies = {
        item.responsibility_id: item for item in protocol.module_matrix
    }
    assert studies["lf_detector"].prerequisite_responsibility_ids == (
        "key_schedule",
    )
    assert studies["hf_detector"].prerequisite_responsibility_ids == (
        "key_schedule",
    )
    for carrier_id in ("lf_carrier", "hf_carrier"):
        assert not {
            "lf_attribution_tpr",
            "lf_primary_null_fpr",
            "hf_attribution_tpr",
            "hf_primary_null_fpr",
        } & set(studies[carrier_id].metric_ids)


@pytest.mark.unit
def test_every_module_signal_criterion_exactly_covers_registered_metrics() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert all(
        tuple(criterion.metric_id for criterion in study.signal_criteria)
        == study.metric_ids
        for study in protocol.module_matrix
    )
    first = protocol.module_matrix[0]
    drifted = replace(
        protocol,
        module_matrix=(
            replace(first, signal_criteria=first.signal_criteria[:-1]),
            *protocol.module_matrix[1:],
        ),
    )
    assert any(
        "signal_criteria_metric_coverage_invalid" in violation
        for violation in drifted.validate()
    )


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
        wiring_manifest = _development_manifest(WIRING_SOURCE_CLUSTER_COUNT)
        wiring_intent = _execution_intent(wiring_manifest)
        build_development_cross_fit_plan(
            responsibility_id="key_schedule",
            execution_intent_authority=wiring_intent,
            expected_execution_intent_authority_digest=(
                wiring_intent.authority_digest
            ),
            expected_source_cluster_count=WIRING_SOURCE_CLUSTER_COUNT,
        )


@pytest.mark.unit
def test_thirteen_module_mechanism_screening_freezes_exact_budget_and_order() -> None:
    protocol = load_frozen_development_exploration_protocol(
        MECHANISM_SCREENING_CONFIG_PATH
    )
    assert protocol.protocol_id == THIRTEEN_MODULE_MECHANISM_SCREENING_PROTOCOL_ID
    assert protocol.validate() == ()
    assert protocol.study_budget.maximum_operational_units == 42
    assert protocol.study_budget.maximum_scientific_units == 240
    assert protocol.study_budget.maximum_total_units == 282
    assert protocol.study_budget.maximum_total_record_attempts == 846
    assert protocol.study_budget.scientific_source_cluster_scales == (16, 32)
    assert (
        protocol.development_routing_reference_cross_fit.source_cluster_count
        == 32
    )
    roster = protocol.unit_roster
    expected_blocks = (
        (0, 2, "development_environment_preflight", "development_environment_preflight"),
        (2, 10, "development_full_chain_wiring", "development_full_chain_wiring"),
        (10, 26, "development_scientific_responsibility_case", "key_schedule"),
        (26, 42, "development_scientific_responsibility_case", "hf_carrier"),
        (42, 58, "development_scientific_responsibility_case", "hf_detector"),
        (58, 74, "development_scientific_responsibility_case", "lf_carrier"),
        (74, 90, "development_scientific_responsibility_case", "lf_detector"),
        (90, 122, "development_scientific_responsibility_case", "qk_geometry_sync"),
        (122, 154, "development_routing_reference_fit", "content_router"),
        (154, 170, "development_scientific_responsibility_case", "content_router"),
        (170, 186, "development_scientific_responsibility_case", "content_embedder"),
        (186, 202, "development_scientific_responsibility_case", "content_detector"),
        (202, 234, "development_scientific_responsibility_case", "geometric_transform_estimator"),
        (234, 250, "development_scientific_responsibility_case", "geometry_reliability"),
        (250, 266, "development_scientific_responsibility_case", "image_rectifier"),
        (266, 282, "development_scientific_responsibility_case", "conditional_recovery_decision"),
    )
    for start, stop, phase, responsibility_id in expected_blocks:
        assert {
            (unit.phase, unit.responsibility_id) for unit in roster[start:stop]
        } == {(phase, responsibility_id)}
    assert not any(unit.phase == "development_paired_ablation" for unit in roster)
    assert sum(
        unit.phase == "development_routing_reference_fit" for unit in roster
    ) == 32
    module_scales = {
        item.responsibility_id: item.scientific_source_cluster_scale
        for item in protocol.module_matrix
    }
    assert module_scales["qk_geometry_sync"] == 32
    assert module_scales["geometric_transform_estimator"] == 32
    assert {
        module_scales[responsibility_id]
        for responsibility_id in REQUIRED_METHOD_RESPONSIBILITIES
        if responsibility_id
        not in {"qk_geometry_sync", "geometric_transform_estimator"}
    } == {16}


@pytest.mark.unit
def test_study_roster_is_breadth_first_enumerable_and_budget_bounded() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    roster = enumerate_development_study_units(protocol)
    budget = protocol.study_budget
    assert len(roster) == budget.maximum_total_units == 506
    assert budget.maximum_operational_units == 74
    assert budget.maximum_scientific_units == 432
    assert sum(unit.phase in OPERATIONAL_UNIT_PHASES for unit in roster) == 74
    assert {
        unit.maximum_duration_seconds
        for unit in roster
        if unit.phase == "development_full_chain_wiring"
    } == {2100}
    assert all(
        unit.maximum_duration_seconds == 900
        for unit in roster
        if unit.phase != "development_full_chain_wiring"
    )
    assert budget.maximum_total_record_attempts == sum(
        unit.maximum_record_attempts for unit in roster
    ) == 1518
    assert len(
        {
            (
                unit.phase,
                unit.responsibility_id,
                unit.source_cluster_ordinal,
                unit.content_branch_id,
                unit.geometry_case_id,
            )
            for unit in roster
        }
    ) == len(roster)
    assert tuple(
        (roster[index].responsibility_id, roster[index].phase)
        for index in (10, 26, 42, 106, 122, 186, 218, 282, 314, 346, 426, 458, 474, 490)
    ) == (
        ("key_schedule", "development_scientific_responsibility_case"),
        ("hf_carrier", "development_scientific_responsibility_case"),
        ("hf_detector", "development_scientific_responsibility_case"),
        ("lf_carrier", "development_scientific_responsibility_case"),
        ("lf_detector", "development_scientific_responsibility_case"),
        ("qk_geometry_sync", "development_scientific_responsibility_case"),
        ("content_router", "development_routing_reference_fit"),
        ("content_router", "development_scientific_responsibility_case"),
        ("content_embedder", "development_scientific_responsibility_case"),
        ("content_detector", "development_scientific_responsibility_case"),
        ("geometric_transform_estimator", "development_scientific_responsibility_case"),
        ("geometry_reliability", "development_scientific_responsibility_case"),
        ("image_rectifier", "development_scientific_responsibility_case"),
        ("conditional_recovery_decision", "development_scientific_responsibility_case"),
    )
    assert sum(
        unit.phase == "development_paired_ablation" for unit in roster
    ) == 48
    for responsibility_id, content_branch_id in (
        ("lf_carrier", "lf_only"),
        ("hf_carrier", "hf_only"),
    ):
        carrier_units = tuple(
            unit
            for unit in roster
            if unit.responsibility_id == responsibility_id
            and unit.phase == "development_scientific_responsibility_case"
        )
        assert len(carrier_units) == 16
        assert {unit.source_cluster_ordinal for unit in carrier_units} == set(
            range(16)
        )
        assert {unit.content_branch_id for unit in carrier_units} == {
            content_branch_id
        }
    content_detector_core = tuple(
        unit
        for unit in roster
        if unit.responsibility_id == "content_detector"
        and unit.phase == "development_scientific_responsibility_case"
    )
    content_detector_paired = tuple(
        unit
        for unit in roster
        if unit.responsibility_id == "content_detector"
        and unit.phase == "development_paired_ablation"
    )
    assert len(content_detector_core) == 64
    assert {unit.source_cluster_ordinal for unit in content_detector_core} == set(
        range(64)
    )
    assert {unit.content_branch_id for unit in content_detector_core} == {
        "lf_hf_routed_combination"
    }
    assert len(content_detector_paired) == 16
    assert {unit.source_cluster_ordinal for unit in content_detector_paired} == set(
        range(16)
    )
    assert {unit.content_branch_id for unit in content_detector_paired} == {
        "hf_only"
    }
    for responsibility_id, content_branch_id in (
        ("hf_detector", "hf_only"),
        ("lf_detector", "lf_only"),
    ):
        detector_units = tuple(
            unit
            for unit in roster
            if unit.responsibility_id == responsibility_id
            and unit.phase == "development_scientific_responsibility_case"
        )
        assert len(detector_units) == 64
        assert {unit.source_cluster_ordinal for unit in detector_units} == set(
            range(64)
        )
        assert {unit.content_branch_id for unit in detector_units} == {
            content_branch_id
        }
    assert tuple(
        unit.geometry_case_id
        for unit in roster
        if unit.responsibility_id == "qk_geometry_sync"
    )[:16] == (
        tuple(item.case_id for item in protocol.geometry_study.operation_cases)
        + protocol.geometry_study.negative_control_case_ids
    ) * 2
    counts = {
        responsibility: len(
            {
                unit.source_cluster_ordinal
                for unit in roster
                if unit.responsibility_id == responsibility
                and unit.phase not in OPERATIONAL_UNIT_PHASES
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
    final_index = {
        responsibility: max(
            unit.unit_index
            for unit in roster
            if unit.responsibility_id == responsibility
            and unit.phase not in OPERATIONAL_UNIT_PHASES
        )
        for responsibility in REQUIRED_METHOD_RESPONSIBILITIES
    }
    first_index = {
        responsibility: min(
            unit.unit_index
            for unit in roster
            if unit.responsibility_id == responsibility
            and unit.phase not in OPERATIONAL_UNIT_PHASES
        )
        for responsibility in REQUIRED_METHOD_RESPONSIBILITIES
    }
    for study in protocol.module_matrix:
        assert all(
            final_index[prerequisite] < first_index[study.responsibility_id]
            for prerequisite in study.prerequisite_responsibility_ids
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
                    _unit(
                        100,
                        threshold_authority=protocol.threshold_detector_authority,
                        split=forbidden_split,
                    ),
                ),
        )
        with pytest.raises(ValueError, match="study_manifest_contains_wrong_split"):
            development_assignments_only(
                contaminated,
                protocol=protocol,
                seed_namespace="development_exploration_seed_namespace",
            )


@pytest.mark.unit
@pytest.mark.parametrize("cluster_count", (64,))
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
        expected_execution_intent_authority_digest=(
            plan.expected_execution_intent_authority_digest
        ),
        requested_split=DEVELOPMENT_SPLIT,
        requested_analysis_unit_identity=_plan_identity(plan, probe),
    )
    with pytest.raises(PermissionError, match="fold_leakage"):
        authorize_development_provisional_threshold(
            threshold,
            plan,
            expected_execution_intent_authority_digest=(
                plan.expected_execution_intent_authority_digest
            ),
            requested_split=DEVELOPMENT_SPLIT,
            requested_analysis_unit_identity=_plan_identity(
                plan,
                plan.folds[0].fit_source_cluster_ids[0],
            ),
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
    outside_provenance = dict(valid[0].source_record.provenance_trace)
    outside_provenance["input_manifest_digest"] = "f" * 64
    outside_manifest_record = _redigest_scientific_record(
        valid[0].source_record,
        provenance_trace=outside_provenance,
    )
    outside_manifest = (
        _redigest_fit_input(valid[0], outside_manifest_record),
        *valid[1:],
    )
    with pytest.raises(ValueError, match="manifest_digest_mismatch"):
        _create_threshold(plan, outside_manifest)
    probe_assignment = next(
        assignment
        for assignment in _development_manifest(plan.source_cluster_count).assignments
        if assignment.identity.source_cluster_id
        == fold.recovery_probe_source_cluster_ids[0]
    )
    leaked_metric = dict(valid[0].source_record.metric_observation)
    leaked_metric["source_cluster_id"] = probe_assignment.identity.source_cluster_id
    leaked_metric_without_digest = dict(leaked_metric)
    leaked_metric_without_digest.pop("observation_digest")
    leaked_metric["observation_digest"] = canonical_development_value_digest(
        leaked_metric_without_digest
    )
    leaked_record = _redigest_scientific_record(
        valid[0].source_record,
        analysis_unit_identity=asdict(probe_assignment.identity),
        metric_observation=leaked_metric,
    )
    leaked = (
        _redigest_fit_input(valid[0], leaked_record),
        *valid[1:],
    )
    with pytest.raises(ValueError, match="recovery_probe_leakage"):
        _create_threshold(plan, leaked)


@pytest.mark.unit
def test_threshold_rejects_manifest_case_role_spoof_and_wrong_key_fit() -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    spoofed_identity = dict(valid[0].source_record.analysis_unit_identity)
    spoofed_identity["case_id"] = "registered_positive_case"
    spoofed_record = _redigest_scientific_record(
        valid[0].source_record,
        analysis_unit_identity=spoofed_identity,
    )
    spoofed = (_redigest_fit_input(valid[0], spoofed_record), *valid[1:])
    with pytest.raises(ValueError, match="threshold_fit_input_case_identity_invalid"):
        _create_threshold(plan, spoofed)
    wrong_key = (replace(valid[0], case_role="wrong_key_control"), *valid[1:])
    with pytest.raises(ValueError, match="threshold_fit_input_role_invalid"):
        _create_threshold(plan, wrong_key)


@pytest.mark.unit
def test_threshold_bridge_rejects_registered_branch_and_wrong_null_identity() -> None:
    plan = _cross_fit_plan()
    valid = _valid_fit_inputs(plan)
    registered_metric = dict(valid[0].source_record.metric_observation)
    registered_metric["content_branch_id"] = "clean_control"
    registered_metric_without_digest = dict(registered_metric)
    registered_metric_without_digest.pop("observation_digest")
    registered_metric["observation_digest"] = canonical_development_value_digest(
        registered_metric_without_digest
    )
    registered_record = _redigest_scientific_record(
        valid[0].source_record,
        content_branch_id="clean_control",
        metric_observation=registered_metric,
    )
    registered_input = _redigest_fit_input(valid[0], registered_record)
    with pytest.raises(ValueError, match="threshold_fit_input_case_identity_invalid"):
        _create_threshold(plan, (registered_input, *valid[1:]))

    key_trace = dict(valid[0].source_record.key_control_trace)
    key_trace["primary_null_control_identity"] = "registered_key_control"
    wrong_identity_record = _redigest_scientific_record(
        valid[0].source_record,
        key_control_trace=key_trace,
    )
    wrong_identity_input = _redigest_fit_input(valid[0], wrong_identity_record)
    with pytest.raises(ValueError, match="threshold_fit_input_control_identity_invalid"):
        _create_threshold(plan, (wrong_identity_input, *valid[1:]))


@pytest.mark.unit
def test_registered_positive_manifest_cannot_be_relabelled_as_primary_null() -> None:
    manifest = _manifest(64, case_id="registered_positive_case")
    intent = _execution_intent(manifest)
    plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=intent,
        expected_execution_intent_authority_digest=intent.authority_digest,
        expected_source_cluster_count=64,
    )
    _, detector_binding, forged_inputs = _threshold_material(
        plan,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="threshold_fit_input_case_identity_invalid"):
        create_development_provisional_threshold(
            plan,
            expected_execution_intent_authority_digest=(
                plan.expected_execution_intent_authority_digest
            ),
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
    assert detector_payload["preprocessing_identity"] == (
        "rgb8_public_image_float32_unit_interval"
    )
    assert detector_payload["public_key_relation"] == (
        "registered_detection_public_digests_equal"
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
    detector = dict(source.detector_trace)
    detector.update(
        raw_preprocessing_identity="alternate_public_preprocess",
        rectified_preprocessing_identity="alternate_public_preprocess",
        primary_null_preprocessing_identity="alternate_public_preprocess",
    )
    forged = _redigest_fit_input(
        valid[0],
        _redigest_scientific_record(source, detector_trace=detector),
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
        key_trace = dict(source.key_control_trace)
        key_trace.update(
            registered_key_public_digest=f"{index + 40:064x}",
            primary_null_detection_key_public_digest=f"{index + 80:064x}",
        )
        forged_inputs.append(
            _redigest_fit_input(
                fit_input,
                _redigest_scientific_record(source, key_control_trace=key_trace),
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
    detector_trace = dict(source.detector_trace)
    detector_trace.update(
        raw_preprocessing_identity="alternate_public_preprocess",
        rectified_preprocessing_identity="alternate_public_preprocess",
        primary_null_preprocessing_identity="alternate_public_preprocess",
    )
    key_control_trace = dict(source.key_control_trace)
    key_control_trace.update(
        registered_key_public_digest="a" * 64,
        primary_null_detection_key_public_digest="b" * 64,
    )
    forged_source = _redigest_scientific_record(
        source,
        detector_trace=detector_trace,
        key_control_trace=key_control_trace,
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
def test_threshold_authority_rejects_full_chain_preprocess_rebinding() -> None:
    plan = _cross_fit_plan()
    manifest, binding, _ = _threshold_material(plan)
    forged_authority = replace(
        binding.protocol.threshold_detector_authority,
        preprocessing_identity="alternate_public_preprocess",
    )
    forged_protocol = replace(
        binding.protocol,
        threshold_detector_authority=forged_authority,
    )
    assert forged_protocol.digest() != binding.protocol_digest
    with pytest.raises(
        ValueError,
        match="threshold_authority_preprocessing_identity_invalid",
    ):
        create_frozen_development_execution_intent_authority(
            forged_protocol,
            run_id=plan.execution_intent_authority.run_id,
            seed_namespace=plan.execution_intent_authority.seed_namespace,
            input_manifest=manifest,
            public_key_roster=plan.execution_intent_authority.public_key_roster,
        )


@pytest.mark.unit
def test_threshold_authority_rejects_full_public_roster_replacement() -> None:
    plan = _cross_fit_plan()
    manifest, binding, _ = _threshold_material(plan)
    replaced_roster = tuple(
        replace(
            item,
            registered_key_public_digest=f"{index + 400:064x}",
            detection_key_public_digest=f"{index + 400:064x}",
        )
        for index, item in enumerate(
            plan.execution_intent_authority.public_key_roster
        )
    )
    with pytest.raises(
        ValueError,
        match="execution_intent_key_family_roster_mismatch",
    ):
        create_frozen_development_execution_intent_authority(
            binding.protocol,
            run_id=plan.execution_intent_authority.run_id,
            seed_namespace=plan.execution_intent_authority.seed_namespace,
            input_manifest=manifest,
            public_key_roster=replaced_roster,
        )


@pytest.mark.unit
def test_old_execution_intent_rejects_wholesale_manifest_and_result_rebuild() -> None:
    old_plan = _cross_fit_plan()
    old_intent_digest = old_plan.expected_execution_intent_authority_digest
    rebound_manifest, rebound_roster = _rebind_manifest_public_roster(
        old_plan.input_manifest
    )
    new_intent = create_frozen_development_execution_intent_authority(
        old_plan.execution_intent_authority.protocol,
        run_id=old_plan.execution_intent_authority.run_id,
        seed_namespace=old_plan.execution_intent_authority.seed_namespace,
        input_manifest=rebound_manifest,
        public_key_roster=rebound_roster,
    )
    assert new_intent.authority_digest != old_intent_digest
    new_plan = build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        execution_intent_authority=new_intent,
        expected_execution_intent_authority_digest=new_intent.authority_digest,
        expected_source_cluster_count=old_plan.source_cluster_count,
    )
    fit_clusters = set(new_plan.folds[0].fit_source_cluster_ids)
    with pytest.raises(
        PermissionError,
        match="threshold_expected_execution_intent_digest_mismatch",
    ):
        create_development_threshold_detector_binding(
            new_plan,
            expected_execution_intent_authority_digest=old_intent_digest,
            fold_index=0,
            input_manifest=rebound_manifest,
            primary_null_key_bindings=tuple(
                item
                for item in new_intent.public_key_roster
                if item.source_cluster_id in fit_clusters
            ),
        )
    manifest, binding, fit_inputs = _threshold_material(new_plan)
    with pytest.raises(
        PermissionError,
        match="provisional_threshold_execution_intent_mismatch",
    ):
        create_development_provisional_threshold(
            new_plan,
            expected_execution_intent_authority_digest=old_intent_digest,
            fold_index=0,
            input_manifest=manifest,
            detector_binding=binding,
            fit_inputs=fit_inputs,
        )
    rebuilt_threshold = _create_threshold(new_plan)
    probe = new_plan.folds[0].recovery_probe_source_cluster_ids[0]
    with pytest.raises(
        PermissionError,
        match="development_execution_intent_digest_mismatch",
    ):
        authorize_development_provisional_threshold(
            rebuilt_threshold,
            new_plan,
            expected_execution_intent_authority_digest=old_intent_digest,
            requested_split=DEVELOPMENT_SPLIT,
            requested_analysis_unit_identity=_plan_identity(new_plan, probe),
        )


@pytest.mark.unit
def test_recovery_manifest_rebinding_cannot_reuse_pinned_plan_or_removed_probe() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    probe = plan.folds[0].recovery_probe_source_cluster_ids[0]
    old_probe_identity = _plan_identity(plan, probe)
    probe_index = next(
        index
        for index, assignment in enumerate(plan.input_manifest.assignments)
        if assignment.identity == old_probe_identity
    )
    rebound_manifest, _ = _rebind_manifest_public_roster(
        plan.input_manifest,
        assignment_indexes={probe_index},
    )
    assert SplitAssignment(
        identity=old_probe_identity,
        split=DEVELOPMENT_SPLIT,
    ) not in rebound_manifest.assignments
    binding_violations = threshold.detector_binding.validate(
        plan,
        rebound_manifest,
        threshold.fold_index,
    )
    assert "threshold_detector_manifest_plan_mismatch" in binding_violations
    forged = replace(
        threshold,
        input_manifest=rebound_manifest,
        input_manifest_digest=rebound_manifest.digest(),
    )
    forged = replace(
        forged,
        threshold_identity=_digest(forged.payload_without_identity()),
    )
    assert "provisional_threshold_manifest_plan_mismatch" in forged.validate(plan)
    with pytest.raises(
        ValueError,
        match="provisional_threshold_manifest_plan_mismatch",
    ):
        authorize_development_provisional_threshold(
            forged,
            plan,
            expected_execution_intent_authority_digest=(
                plan.expected_execution_intent_authority_digest
            ),
            requested_split=DEVELOPMENT_SPLIT,
            requested_analysis_unit_identity=old_probe_identity,
        )


@pytest.mark.unit
def test_threshold_authority_rejects_full_chain_detector_mode_rebinding() -> None:
    plan = _cross_fit_plan()
    manifest, binding, _ = _threshold_material(plan)
    forged_authority = replace(
        binding.protocol.threshold_detector_authority,
        detector_mode="combined",
    )
    forged_protocol = replace(
        binding.protocol,
        threshold_detector_authority=forged_authority,
    )
    assert forged_protocol.digest() != binding.protocol_digest
    with pytest.raises(
        ValueError,
        match="threshold_authority_detector_mode_invalid",
    ):
        create_frozen_development_execution_intent_authority(
            forged_protocol,
            run_id=plan.execution_intent_authority.run_id,
            seed_namespace=plan.execution_intent_authority.seed_namespace,
            input_manifest=manifest,
            public_key_roster=plan.execution_intent_authority.public_key_roster,
        )


@pytest.mark.unit
def test_threshold_authority_payload_tamper_fails_after_protocol_redigest() -> None:
    plan = _cross_fit_plan()
    manifest, binding, _ = _threshold_material(plan)
    forged_authority = replace(
        binding.protocol.threshold_detector_authority,
        registered_candidate_config_digest="f" * 64,
    )
    forged_protocol = replace(
        binding.protocol,
        threshold_detector_authority=forged_authority,
    )
    assert forged_protocol.digest() != binding.protocol_digest
    with pytest.raises(
        ValueError,
        match="threshold_authority_candidate_config_mismatch",
    ):
        create_frozen_development_execution_intent_authority(
            forged_protocol,
            run_id=plan.execution_intent_authority.run_id,
            seed_namespace=plan.execution_intent_authority.seed_namespace,
            input_manifest=manifest,
            public_key_roster=plan.execution_intent_authority.public_key_roster,
        )


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
                expected_execution_intent_authority_digest=(
                    plan.expected_execution_intent_authority_digest
                ),
                requested_split=forbidden_split,
                requested_analysis_unit_identity=_plan_identity(plan, probe),
            )


@pytest.mark.unit
def test_protocol_dependency_decision_requires_persistent_store_replay() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    with pytest.raises(
        PermissionError,
        match="persistent-store replay",
    ):
        decide_development_module_execution(
            protocol,
            "content_router",
            {},
        )


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
def test_module_outcome_requires_verified_evidence_context() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    assert not hasattr(
        development_protocol_module,
        "create_development_module_outcome_record",
    )
    outcome = DevelopmentModuleOutcomeRecord(
        outcome_record_id="0" * 64,
        responsibility_id="hf_detector",
        module_outcome="mechanism_signal_observed",
        candidate_recommendation="candidate_worth_further_selection",
        recommendation_reason="unverified caller assertion",
        blocking_responsibilities=(),
        evidence_record_ids=("nonexistent_record",),
        evidence_record_digests=("1" * 64,),
        provisional_threshold_identities=(),
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest="2" * 64,
        input_manifest_digest="3" * 64,
        candidate_config_digest="4" * 64,
        signal_criteria_digest="5" * 64,
        cluster_aggregate_digest="6" * 64,
        cross_fit_plan_digest=None,
        evidence_record_bindings=(("nonexistent_record", "1" * 64),),
        committed_marker_bindings=(("nonexistent_record", "1" * 64, "7" * 64),),
        source_record_schema_version=RECORD_SCHEMA_VERSION,
        source_record_collection_schema_version=RECORD_COLLECTION_SCHEMA_VERSION,
        scientific_claims_supported=False,
    )
    assert outcome.validate(protocol) == (
        "module_outcome_verified_evidence_context_required",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation,expected_reason",
    (
        ("candidate_digest", "candidate_config_digest_invalid"),
        ("unregistered_candidate", "candidate_ids_unregistered"),
        ("unregistered_metric", "metric_ids_unregistered"),
        ("unregistered_control", "negative_controls_unregistered"),
        ("qk_quality_criterion_relaxed", "signal_criteria_unregistered"),
        ("joint_fpr_criterion_relaxed", "signal_criteria_unregistered"),
        ("ratio_roster_drift", "candidate_parameters_unregistered"),
        ("missing_metric", "metric_ids_missing"),
        ("expanded_split", "development_allowed_split_invalid"),
        ("changed_budget", "maximum_scientific_units_invalid"),
        ("adaptive_order", "score_adaptive_unit_changes_must_be_forbidden"),
        ("missing_clean_branch", "clean_control_missing"),
        ("incomplete_geometry", "geometry_case_coverage_invalid"),
        ("role_split_rebound", "study_role_registered_binding_invalid"),
        ("authority_preprocess_rebound", "threshold_authority_preprocessing_identity_invalid"),
        ("authority_mode_rebound", "threshold_authority_detector_mode_invalid"),
        ("authority_candidate_rebound", "threshold_authority_candidate_config_mismatch"),
        ("execution_intent_role_rebound", "execution_intent_authority_role_invalid"),
        ("execution_intent_secret_relaxed", "execution_intent_raw_secret_policy_invalid"),
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
    elif mutation == "qk_quality_criterion_relaxed":
        entry = document["module_matrix"][8]
        entry["signal_criteria"][2]["threshold"] = 999.0
        _recompute_candidate_config_digest(entry)
    elif mutation == "joint_fpr_criterion_relaxed":
        entry = document["module_matrix"][12]
        entry["signal_criteria"][1]["threshold"] = 1.0
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
    elif mutation == "authority_preprocess_rebound":
        document["threshold_detector_authority"]["preprocessing_identity"] = (
            "alternate_public_preprocess"
        )
    elif mutation == "authority_mode_rebound":
        document["threshold_detector_authority"]["detector_mode"] = "combined"
    elif mutation == "authority_candidate_rebound":
        document["threshold_detector_authority"][
            "registered_candidate_config_digest"
        ] = "f" * 64
    elif mutation == "execution_intent_role_rebound":
        document["execution_intent_policy"]["authority_role"] = (
            "replaceable_after_scientific_records"
        )
    elif mutation == "execution_intent_secret_relaxed":
        document["execution_intent_policy"]["raw_secret_policy"] = (
            "raw_secret_allowed"
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
