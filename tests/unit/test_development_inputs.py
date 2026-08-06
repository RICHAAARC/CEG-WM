"""Lightweight checks for frozen production development input identities."""

from pathlib import Path

import pytest

from experiments.protocol.development_exploration import (
    build_development_cross_fit_plan,
    create_frozen_development_execution_intent_authority,
    development_cross_fit_source_cluster_ids,
    load_frozen_development_exploration_protocol,
)
from experiments.runners.development_inputs import (
    DevelopmentInputError,
    build_development_manifest_and_key_roster,
    exact_positive_nearest_rank_p95,
    load_development_prompt_roster,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.quick
def test_checked_in_prompt_roster_builds_isolated_development_manifest() -> None:
    protocol = load_frozen_development_exploration_protocol(
        ROOT / "configs/experiments/development_module_exploration.json"
    )
    roster = load_development_prompt_roster(
        ROOT / "configs/experiments/development_exploration_prompt_roster.json"
    )
    manifest, key_roster = build_development_manifest_and_key_roster(
        protocol,
        roster,
        "unit-test-development-root-key",
    )
    assert len(roster.entries) == len(manifest.assignments) == len(key_roster) == 64
    assert manifest.validate(require_all_splits=False) == ()
    assert {item.split for item in manifest.assignments} == {"development"}
    assert len({item.identity.source_cluster_id for item in manifest.assignments}) == 64


@pytest.mark.quick
def test_screening_prompt_roster_and_detector_cross_fit_are_frozen_subsets() -> None:
    screening_protocol = load_frozen_development_exploration_protocol(
        ROOT / "configs/experiments/thirteen_module_mechanism_screening.json"
    )
    screening_roster = load_development_prompt_roster(
        ROOT
        / "configs/experiments/thirteen_module_mechanism_screening_prompt_roster.json"
    )
    legacy_roster = load_development_prompt_roster(
        ROOT / "configs/experiments/development_exploration_prompt_roster.json"
    )
    manifest, key_roster = build_development_manifest_and_key_roster(
        screening_protocol,
        screening_roster,
        "unit-test-screening-root-key",
    )
    legacy_protocol = load_frozen_development_exploration_protocol(
        ROOT / "configs/experiments/development_module_exploration.json"
    )
    legacy_manifest, _legacy_key_roster = build_development_manifest_and_key_roster(
        legacy_protocol,
        legacy_roster,
        "unit-test-screening-root-key",
    )
    assert len(screening_roster.entries) == len(manifest.assignments) == 32
    assert screening_roster.digest != legacy_roster.digest
    assert screening_roster.seed_namespace != legacy_roster.seed_namespace
    assert {
        item.generation_seed for item in screening_roster.entries
    }.isdisjoint(item.generation_seed for item in legacy_roster.entries)
    assert manifest.digest() != legacy_manifest.digest()
    assert {
        item.identity.source_cluster_id for item in manifest.assignments
    }.isdisjoint(
        item.identity.source_cluster_id for item in legacy_manifest.assignments
    )
    authority = create_frozen_development_execution_intent_authority(
        screening_protocol,
        run_id="thirteen_module_mechanism_screening_test_run",
        seed_namespace=screening_roster.seed_namespace,
        input_manifest=manifest,
        public_key_roster=key_roster,
    )
    for responsibility_id in ("lf_detector", "hf_detector", "content_detector"):
        expected_ids = development_cross_fit_source_cluster_ids(
            authority,
            responsibility_id=responsibility_id,
        )
        assert len(expected_ids) == 16
        plan = build_development_cross_fit_plan(
            responsibility_id=responsibility_id,
            execution_intent_authority=authority,
            expected_execution_intent_authority_digest=authority.authority_digest,
            expected_source_cluster_ids=expected_ids,
        )
        assert plan.source_cluster_count == 16
        assert plan.input_manifest == manifest
        assert {len(fold.fit_source_cluster_ids) for fold in plan.folds} == {12}
        assert {
            len(fold.recovery_probe_source_cluster_ids) for fold in plan.folds
        } == {4}
        assert plan.validate() == ()
    with pytest.raises(
        ValueError,
        match="development_cross_fit_source_cluster_roster_mismatch",
    ):
        build_development_cross_fit_plan(
            responsibility_id="hf_detector",
            execution_intent_authority=authority,
            expected_execution_intent_authority_digest=authority.authority_digest,
            expected_source_cluster_ids=development_cross_fit_source_cluster_ids(
                authority,
                responsibility_id="hf_detector",
            )[:-1],
        )
    with pytest.raises(
        ValueError,
        match="development_cross_fit_source_cluster_count_mismatch",
    ):
        build_development_cross_fit_plan(
            responsibility_id="hf_detector",
            execution_intent_authority=authority,
            expected_execution_intent_authority_digest=authority.authority_digest,
            expected_source_cluster_count=32,
        )


@pytest.mark.quick
def test_development_reference_uses_strict_positive_exact_nearest_rank_p95() -> None:
    assert exact_positive_nearest_rank_p95((0.0, -1.0, 1.0, 2.0, 3.0, 4.0)) == 4.0
    with pytest.raises(DevelopmentInputError, match="strictly positive"):
        exact_positive_nearest_rank_p95((0.0, -1.0))
