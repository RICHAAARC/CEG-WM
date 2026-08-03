"""Lightweight checks for frozen production development input identities."""

from pathlib import Path

import pytest

from experiments.protocol.development_exploration import (
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
def test_development_reference_uses_strict_positive_exact_nearest_rank_p95() -> None:
    assert exact_positive_nearest_rank_p95((0.0, -1.0, 1.0, 2.0, 3.0, 4.0)) == 4.0
    with pytest.raises(DevelopmentInputError, match="strictly positive"):
        exact_positive_nearest_rank_p95((0.0, -1.0))
