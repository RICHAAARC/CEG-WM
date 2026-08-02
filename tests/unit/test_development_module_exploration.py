"""CPU constraints for the development-only 13-duty exploration protocol."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.protocol.development_exploration import (
    CONTENT_BRANCH_IDS,
    CONTENT_COMBINATION_FUNCTION_IDS,
    CONTENT_MIXING_COEFFICIENTS,
    DEVELOPMENT_SPLIT,
    DEVELOPMENT_UNIT_ORDER,
    FORMAL_LATER_SPLIT_DENY_LIST,
    MODULE_OUTCOMES,
    RECOMMENDATION_BY_MODULE_OUTCOME,
    SCIENTIFIC_SOURCE_CLUSTER_SCALES,
    WIRING_SOURCE_CLUSTER_COUNT,
    authorize_development_provisional_threshold,
    build_development_cross_fit_plan,
    create_development_module_outcome_record,
    create_development_provisional_threshold,
    decide_development_module_execution,
    development_assignments_only,
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
CONFIG_PATH = (
    ROOT / "configs/experiments/development_module_exploration.json"
)
PROTOCOL_PATH = (
    ROOT / "experiments/protocol/development_exploration.py"
)


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


@pytest.mark.unit
def test_frozen_development_protocol_has_exact_matrix_budget_and_finite_grids() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)

    assert protocol.validate() == ()
    assert tuple(item.responsibility for item in protocol.module_matrix) == (
        REQUIRED_METHOD_RESPONSIBILITIES
    )
    assert len({item.development_case_id for item in protocol.module_matrix}) == 13
    assert len(
        {item.candidate_selection_case_id for item in protocol.module_matrix}
    ) == 13
    assert protocol.content_study.branch_ids == CONTENT_BRANCH_IDS
    assert (
        protocol.content_study.mixing_coefficients
        == CONTENT_MIXING_COEFFICIENTS
    )
    assert (
        protocol.content_study.combination_function_ids
        == CONTENT_COMBINATION_FUNCTION_IDS
    )
    assert len(protocol.geometry_grid.grid_points) == 45
    assert {
        point.attack_id for point in protocol.geometry_grid.grid_points
    } == {"identity", "crop", "scale", "rotation", "crop_scale_rotation"}
    assert protocol.study_budget.scientific_source_cluster_scales == (
        SCIENTIFIC_SOURCE_CLUSTER_SCALES
    )
    assert protocol.study_budget.maximum_module_cluster_assignments == sum(
        item.scientific_source_cluster_scale
        for item in protocol.module_matrix
    )
    assert protocol.study_budget.maximum_record_attempts_per_unit == (
        MAXIMUM_RECORD_ATTEMPTS
    )
    assert protocol.study_budget.unit_order == DEVELOPMENT_UNIT_ORDER
    assert "4096" not in CONFIG_PATH.read_text(encoding="utf-8")


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
@pytest.mark.parametrize("cluster_count", SCIENTIFIC_SOURCE_CLUSTER_SCALES)
def test_development_threshold_cross_fit_is_balanced_and_development_only(
    cluster_count: int,
) -> None:
    assignments = development_assignments_only(
        _development_manifest(cluster_count)
    )
    plan = build_development_cross_fit_plan(
        responsibility="hf_detector",
        assignments=assignments,
        expected_source_cluster_count=cluster_count,
    )

    assert plan.validate() == ()
    assert len(plan.folds) == 4
    assert {len(fold.score_source_cluster_ids) for fold in plan.folds} == {
        cluster_count // 4
    }
    assert {
        cluster
        for fold in plan.folds
        for cluster in fold.score_source_cluster_ids
    } == {assignment.identity.source_cluster_id for assignment in assignments}

    provisional = create_development_provisional_threshold(
        plan,
        fold_index=0,
        threshold=0.25,
    )
    assert provisional.validate(plan) == ()
    held_out_cluster = plan.folds[0].score_source_cluster_ids[0]
    authorize_development_provisional_threshold(
        provisional,
        plan,
        requested_split=DEVELOPMENT_SPLIT,
        source_cluster_id=held_out_cluster,
    )
    with pytest.raises(PermissionError, match="fold_leakage"):
        authorize_development_provisional_threshold(
            provisional,
            plan,
            requested_split=DEVELOPMENT_SPLIT,
            source_cluster_id=plan.folds[0].fit_source_cluster_ids[0],
        )
    for forbidden_split in FORMAL_LATER_SPLIT_DENY_LIST:
        with pytest.raises(PermissionError, match="invalid_for_split"):
            authorize_development_provisional_threshold(
                provisional,
                plan,
                requested_split=forbidden_split,
                source_cluster_id=held_out_cluster,
            )


@pytest.mark.unit
def test_wiring_cluster_fixture_never_counts_as_scientific_coverage() -> None:
    assignments = development_assignments_only(
        _development_manifest(WIRING_SOURCE_CLUSTER_COUNT)
    )
    with pytest.raises(
        ValueError,
        match="wiring_clusters_do_not_count_as_scientific_coverage",
    ):
        build_development_cross_fit_plan(
            responsibility="key_schedule",
            assignments=assignments,
            expected_source_cluster_count=WIRING_SOURCE_CLUSTER_COUNT,
        )


@pytest.mark.unit
def test_dependency_stop_rule_requires_observed_prerequisite_signal() -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)

    missing = decide_development_module_execution(
        protocol,
        "content_embedder",
        {},
    )
    assert not missing.approved
    assert missing.decision_reason == "prerequisite_outcome_missing"
    assert set(missing.missing_prerequisites) == {
        "content_router",
        "lf_carrier",
        "hf_carrier",
    }

    blocked = decide_development_module_execution(
        protocol,
        "content_embedder",
        {
            "content_router": "development_signal_observed",
            "lf_carrier": "development_signal_not_observed",
            "hf_carrier": "development_signal_observed",
        },
    )
    assert not blocked.approved
    assert blocked.decision_reason == "dependency_stop_rule"
    assert blocked.blocking_prerequisites == ("lf_carrier",)

    approved = decide_development_module_execution(
        protocol,
        "content_embedder",
        {
            "content_router": "development_signal_observed",
            "lf_carrier": "development_signal_observed",
            "hf_carrier": "development_signal_observed",
        },
    )
    assert approved.approved


@pytest.mark.unit
@pytest.mark.parametrize("module_outcome", MODULE_OUTCOMES)
def test_module_outcomes_are_mutually_exclusive_and_fix_recommendation(
    module_outcome: str,
) -> None:
    protocol = load_frozen_development_exploration_protocol(CONFIG_PATH)
    responsibility = (
        "geometric_transform_estimator"
        if module_outcome == "development_dependency_blocked"
        else "hf_detector"
    )
    blocking = (
        ("qk_geometry_sync",)
        if module_outcome == "development_dependency_blocked"
        else ()
    )
    outcome = create_development_module_outcome_record(
        protocol,
        responsibility=responsibility,
        module_outcome=module_outcome,
        recommendation_reason="development evidence was classified by the frozen rule",
        evidence_record_ids=(f"record_for_{responsibility}",),
        blocking_responsibilities=blocking,
    )

    assert outcome.validate(protocol) == ()
    assert outcome.recommended_next_action == (
        RECOMMENDATION_BY_MODULE_OUTCOME[module_outcome]
    )
    assert outcome.source_record_schema_version == (
        INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION
    )
    assert outcome.source_record_collection_schema_version == (
        INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
    )
    assert outcome.scientific_claims_supported is False
    if module_outcome == "development_signal_observed":
        assert outcome.candidate_selection_case_id == (
            "candidate_selection_high_frequency_blind_detection"
        )
    else:
        assert outcome.candidate_selection_case_id is None

    forged = replace(
        outcome,
        recommended_next_action="proceed_to_candidate_selection",
    )
    if module_outcome == "development_signal_observed":
        forged = replace(
            outcome,
            recommended_next_action="record_closed_negative_and_stop_dependent_modules",
        )
    assert "module_outcome_recommendation_mismatch" in forged.validate(protocol)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_candidate_mapping",
        "expanded_split_access",
        "changed_attempt_budget",
        "unknown_top_level_key",
    ),
)
def test_config_loader_rejects_semantic_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if mutation == "duplicate_candidate_mapping":
        document["module_matrix"][1]["candidate_selection_case_id"] = (
            document["module_matrix"][0]["candidate_selection_case_id"]
        )
    elif mutation == "expanded_split_access":
        document["split_policy"]["allowed_split"] = "candidate_selection"
    elif mutation == "changed_attempt_budget":
        document["study_budget"]["maximum_record_attempts_per_unit"] = 4
    else:
        document["unregistered_extension"] = True
    path = tmp_path / "development_protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
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
