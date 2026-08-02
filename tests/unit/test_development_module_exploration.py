"""CPU constraints for the development-only 13-duty exploration protocol."""

from __future__ import annotations

import ast
from dataclasses import replace
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
    MODULE_OUTCOMES,
    PREFLIGHT_SOURCE_CLUSTER_COUNT,
    REGISTERED_STUDY_ROLE_BINDINGS,
    WIRING_SOURCE_CLUSTER_COUNT,
    DevelopmentThresholdFitInput,
    authorize_development_provisional_threshold,
    build_development_cross_fit_plan,
    create_development_module_outcome_record,
    create_development_provisional_threshold,
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


def _unit(index: int, *, split: str = DEVELOPMENT_SPLIT) -> SplitAssignment:
    prompt_digest = f"{index + 1:064x}"
    lineage_digest = f"{index + 1001:064x}"
    key_family_digest = f"{index + 2001:064x}"
    identity = AnalysisUnitIdentity(
        unit_id=f"development_unit_{index}",
        case_id="development_protocol_constraint_case",
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


def _development_manifest(count: int) -> FrozenSplitManifest:
    return FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id="development_module_exploration_fixture",
        manifest_revision="fixture_revision",
        assignments=tuple(_unit(index) for index in range(count)),
    )


def _cross_fit_plan(cluster_count: int = 16):
    return build_development_cross_fit_plan(
        responsibility_id="hf_detector",
        assignments=development_assignments_only(
            _development_manifest(cluster_count)
        ),
        expected_source_cluster_count=cluster_count,
    )


def _valid_fit_inputs(plan, fold_index: int = 0):
    fit = plan.folds[fold_index].fit_source_cluster_ids
    midpoint = len(fit) // 2
    return (
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="primary_null",
            source_cluster_ids=fit[:midpoint],
        ),
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="wrong_key_control",
            source_cluster_ids=fit[midpoint:],
        ),
    )


def _create_threshold(plan, fit_inputs=None):
    return create_development_provisional_threshold(
        plan,
        fold_index=0,
        threshold=0.25,
        input_manifest_digest="1" * 64,
        detector_identity="development_blind_high_frequency_detector",
        detector_config_digest="2" * 64,
        threshold_rule_digest="3" * 64,
        fit_inputs=_valid_fit_inputs(plan) if fit_inputs is None else fit_inputs,
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
        assert item.negative_control_case_ids
        assert item.metric_ids
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
                _development_manifest(WIRING_SOURCE_CLUSTER_COUNT)
            ),
            expected_source_cluster_count=WIRING_SOURCE_CLUSTER_COUNT,
        )


@pytest.mark.unit
def test_study_roster_is_breadth_first_enumerable_and_budget_bounded() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    roster = enumerate_development_study_units(protocol)
    budget = protocol.study_budget
    assert len(roster) == budget.maximum_scientific_units == 384
    assert sum(max(1, len(unit.content_branch_ids)) for unit in roster) == (
        budget.maximum_total_branch_units
    )
    assert budget.maximum_total_record_attempts == (
        budget.maximum_total_branch_units
    ) * (
        MAXIMUM_RECORD_ATTEMPTS
    )
    assert tuple(
        unit.responsibility_id for unit in roster[:13]
    ) == REQUIRED_METHOD_RESPONSIBILITIES
    assert all(
        unit.source_cluster_ordinal < BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT
        for unit in roster[: 13 * BASE_SCIENTIFIC_SOURCE_CLUSTER_COUNT]
    )
    counts = {
        responsibility: sum(
            unit.responsibility_id == responsibility for unit in roster
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
def test_split_isolation_has_unique_dimension_rosters_and_exact_role_mapping() -> None:
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
    for dimension in ISOLATION_DIMENSIONS:
        digests = {
            dict(item.identity_dimension_digests)[dimension]
            for item in isolation.role_bindings
        }
        assert len(digests) == len(isolation.role_bindings)
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


@pytest.mark.unit
def test_development_split_surface_denies_every_formal_later_split() -> None:
    development = _development_manifest(16)
    assert development_assignments_only(development) == development.assignments
    for forbidden_split in FORMAL_LATER_SPLIT_DENY_LIST:
        contaminated = replace(
            development,
            assignments=(
                *development.assignments[:-1],
                _unit(100, split=forbidden_split),
            ),
        )
        with pytest.raises(
            PermissionError,
            match="development_exploration_split_forbidden",
        ):
            development_assignments_only(contaminated)


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
def test_threshold_rejects_non_primary_null_or_key_control_inputs(
    invalid_role: str,
) -> None:
    plan = _cross_fit_plan()
    fit = plan.folds[0].fit_source_cluster_ids
    invalid = (
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="primary_null",
            source_cluster_ids=fit[:6],
        ),
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role=invalid_role,
            source_cluster_ids=fit[6:],
        ),
    )
    with pytest.raises(ValueError, match="threshold_fit_input_role_invalid"):
        _create_threshold(plan, invalid)


@pytest.mark.unit
def test_threshold_rejects_non_development_input_and_probe_leakage() -> None:
    plan = _cross_fit_plan()
    fold = plan.folds[0]
    invalid_split = (
        DevelopmentThresholdFitInput(
            source_split="candidate_selection",
            case_role="primary_null",
            source_cluster_ids=fold.fit_source_cluster_ids[:6],
        ),
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="wrong_key_control",
            source_cluster_ids=fold.fit_source_cluster_ids[6:],
        ),
    )
    with pytest.raises(ValueError, match="threshold_fit_input_split_invalid"):
        _create_threshold(plan, invalid_split)
    leaked = (
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="primary_null",
            source_cluster_ids=(fold.recovery_probe_source_cluster_ids[0],),
        ),
        DevelopmentThresholdFitInput(
            source_split=DEVELOPMENT_SPLIT,
            case_role="wrong_key_control",
            source_cluster_ids=fold.fit_source_cluster_ids,
        ),
    )
    with pytest.raises(ValueError, match="recovery_probe_leakage"):
        _create_threshold(plan, leaked)


@pytest.mark.unit
def test_threshold_binds_manifest_detector_rule_and_fold() -> None:
    plan = _cross_fit_plan()
    threshold = _create_threshold(plan)
    assert threshold.validate(plan) == ()
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
