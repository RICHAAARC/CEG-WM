"""Focused tests for the LF directional component source closure."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scripts.experiment_execution.component_source_closure import (
    LF_DIRECTIONAL_COMPONENT_IDS,
    ComponentSourceClosureError,
    build_component_source_closure,
)


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / ".codex/research_state/method_readiness.yaml"
EXPECTED_PATHS = (
    "main/shared/key_schedule.py",
    "main/content_chain/lf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/lf_detector.py",
    "main/content_chain/lf_whitening.py",
)


def _reviewed_components() -> dict[str, dict[str, object]]:
    payload = json.loads(READINESS.read_text("utf-8"))
    return {
        component_id: payload["components"][component_id]
        for component_id in LF_DIRECTIONAL_COMPONENT_IDS
    }


def _copy_component_sources(destination: Path) -> None:
    for relative in EXPECTED_PATHS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


@pytest.mark.unit
def test_lf_directional_component_closure_binds_only_reviewed_method_sources() -> None:
    closure = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS,
        _reviewed_components(),
        ROOT,
    )

    assert closure.ordered_component_ids == LF_DIRECTIONAL_COMPONENT_IDS
    assert tuple(
        binding.implementation_path for binding in closure.source_bindings
    ) == EXPECTED_PATHS
    assert closure.component_implementation_digest == (
        "005e18bd923995dfc09a1d782e74bd1fba69287c465c9d56d311fc7a3e307c66"
    )
    assert not any(
        binding.implementation_path.endswith("/__init__.py")
        or binding.implementation_path.startswith(
            ("runtime/", "experiments/", "scripts/")
        )
        for binding in closure.source_bindings
    )


@pytest.mark.unit
def test_lf_dependency_source_change_updates_component_digest(tmp_path: Path) -> None:
    _copy_component_sources(tmp_path)
    components = _reviewed_components()
    original = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )

    whitening = tmp_path / "main/content_chain/lf_whitening.py"
    whitening.write_text(
        whitening.read_text("utf-8") + "\n# changed LF candidate source\n",
        encoding="utf-8",
    )
    changed = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )

    assert changed.component_implementation_digest != (
        original.component_implementation_digest
    )


@pytest.mark.unit
def test_unrelated_science_and_delivery_sources_do_not_change_component_digest(
    tmp_path: Path,
) -> None:
    _copy_component_sources(tmp_path)
    components = _reviewed_components()
    original = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )
    for relative in (
        "main/content_chain/hf_detector.py",
        "main/geometry_chain/qk_sync.py",
        "scripts/experiment_execution/package_builder.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNRELATED_VALUE = 1\n", encoding="utf-8")

    replay = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )

    assert replay == original


@pytest.mark.unit
def test_candidate_metadata_stays_separate_while_source_rebinding_is_rejected(
    tmp_path: Path,
) -> None:
    _copy_component_sources(tmp_path)
    components = _reviewed_components()
    original = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )
    candidate_rebind = deepcopy(components)
    candidate_rebind["lf_detector"]["candidate_ids"] = [
        "key_schedule_sha256_counter",
        "lf_low_pass",
        "lf_null_whitened_matched_score",
        "lf_unreviewed_score_candidate",
    ]
    candidate_replay = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, candidate_rebind, tmp_path
    )
    assert candidate_replay == original

    facade_rebind = deepcopy(components)
    facade_rebind["lf_detector"]["implementation_path"] = (
        "main/content_chain/__init__.py"
    )
    with pytest.raises(ComponentSourceClosureError, match="concrete Python"):
        build_component_source_closure(
            LF_DIRECTIONAL_COMPONENT_IDS, facade_rebind, tmp_path
        )


@pytest.mark.unit
def test_repository_revision_metadata_does_not_enter_component_digest(
    tmp_path: Path,
) -> None:
    _copy_component_sources(tmp_path)
    components = _reviewed_components()
    original = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )
    git_head = tmp_path / ".git/HEAD"
    git_head.parent.mkdir(parents=True)
    git_head.write_text("ref: refs/heads/reviewed-revision\n", encoding="utf-8")

    replay = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS, components, tmp_path
    )

    assert replay == original
