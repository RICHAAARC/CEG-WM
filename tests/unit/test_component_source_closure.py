"""Focused tests for the named HF and LF component source closures."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from scripts.experiment_execution.component_source_closure import (
    HF_REFERENCE_COMPONENT_IDS,
    LF_DIRECTIONAL_COMPONENT_IDS,
    ComponentSourceClosureError,
    build_component_source_closure,
    build_hf_reference_component_source_closure,
    build_lf_directional_component_source_closure,
)
from tests.helpers.historical_repository import (
    HF_REFERENCE_PRODUCER_PATHS,
    HF_REFERENCE_PRODUCER_REVISION,
    materialize_historical_repository,
)


ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / ".codex/research_state/method_readiness.yaml"
LF_EXPECTED_PATHS = (
    "main/shared/key_schedule.py",
    "main/content_chain/lf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/lf_detector.py",
    "main/content_chain/lf_whitening.py",
)
HF_EXPECTED_PATHS = (
    "main/shared/key_schedule.py",
    "main/content_chain/hf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/hf_detector.py",
    "main/content_chain/detector.py",
)
LF_DIRECTIONAL_PRODUCER_REVISION = "51adb765cdddafcb4c65c357e899c77b4c9f36d2"
LF_DIRECTIONAL_PRODUCER_PATHS = (
    ".codex/research_state/method_readiness.yaml",
    "configs/experiments/lf_whitened_directional_validation.json",
    "docs/design/candidate_specifications.md",
    *LF_EXPECTED_PATHS,
)


def _reviewed_components(
    component_ids: tuple[str, ...] = LF_DIRECTIONAL_COMPONENT_IDS,
) -> dict[str, dict[str, object]]:
    payload = json.loads(READINESS.read_text("utf-8"))
    return {
        component_id: payload["components"][component_id]
        for component_id in component_ids
    }


def _copy_component_sources(
    destination: Path,
    expected_paths: tuple[str, ...] = LF_EXPECTED_PATHS,
) -> None:
    for relative in expected_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def _materialize_historical_component_authority(
    *,
    tmp_path: Path,
    revision: str,
    paths: tuple[str, ...],
    label: str,
) -> Path:
    if not (ROOT / ".git").exists():
        pytest.skip("local Git metadata is required for historical producer replay")
    return materialize_historical_repository(
        source_root=ROOT,
        revision=revision,
        destination=tmp_path / label,
        paths=paths,
    )


def _assert_lf_directional_producer_authority(producer_root: Path) -> None:
    authority = json.loads(
        (
            producer_root
            / "configs/experiments/lf_whitened_directional_validation.json"
        ).read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (
            producer_root / ".codex/research_state/method_readiness.yaml"
        ).read_text(encoding="utf-8")
    )
    ordered_component_ids = tuple(authority["ordered_component_ids"])
    assert ordered_component_ids == LF_DIRECTIONAL_COMPONENT_IDS
    closure = build_lf_directional_component_source_closure(
        readiness["components"],
        producer_root,
    )
    assert tuple(
        binding.implementation_path for binding in closure.source_bindings
    ) == LF_EXPECTED_PATHS
    assert tuple(asdict(item) for item in closure.source_bindings) == tuple(
        authority["component_source_bindings"]
    )
    assert (
        closure.component_implementation_digest
        == authority["component_implementation_digest"]
        == "9e79aaaadf545966f55fd311a0466f718431c21c39c88addac994149399b41f6"
    )
    candidate_specification_sha256 = sha256(
        (producer_root / "docs/design/candidate_specifications.md").read_bytes()
    ).hexdigest()
    assert candidate_specification_sha256 == authority[
        "candidate_specification_sha256"
    ]
    assert readiness["candidate_specification_sha256"] == (
        candidate_specification_sha256
    )
    review = readiness["independent_semantic_review"]
    assert review["review_reference"] == (
        "independent_lf_prepared_feature_semantic_review:"
        "019fe0f3-b8e8-7230-98f1-9ae0450c1f4a:"
        "00bed2baaf60f039868c208291c86b539a54b2f3:APPROVE"
    )
    assert review["reviewed_repository_revision"] == (
        "00bed2baaf60f039868c208291c86b539a54b2f3"
    )
    assert authority["method_review_reference"] == review["review_reference"]
    assert authority["method_reviewed_revision"] == review[
        "reviewed_repository_revision"
    ]
    assert review["candidate_specification_sha256"] == (
        candidate_specification_sha256
    )


def _assert_hf_reference_producer_authority(producer_root: Path) -> None:
    authority = json.loads(
        (
            producer_root / "configs/experiments/hf_only_reference_validation.json"
        ).read_text(encoding="utf-8")
    )["candidate_binding"]
    ordered_component_ids = tuple(authority["ordered_component_ids"])
    assert ordered_component_ids == HF_REFERENCE_COMPONENT_IDS
    components = {
        binding["component_id"]: {
            "implementation_path": binding["implementation_path"],
            "implementation_symbol": binding["implementation_symbol"],
        }
        for binding in authority["component_source_bindings"]
    }
    closure = build_hf_reference_component_source_closure(
        components,
        producer_root,
    )
    assert tuple(
        binding.implementation_path for binding in closure.source_bindings
    ) == HF_EXPECTED_PATHS
    assert all(
        binding.source_role == "component_implementation"
        for binding in closure.source_bindings
    )
    assert tuple(asdict(item) for item in closure.source_bindings) == tuple(
        authority["component_source_bindings"]
    )
    assert (
        closure.component_implementation_digest
        == authority["component_implementation_digest"]
        == "4323073f7df88c6e3abb253932fba8ba132062b6b47fba0f1db31ded45fd4de1"
    )
    candidate_specification_sha256 = sha256(
        (producer_root / authority["candidate_specification_path"]).read_bytes()
    ).hexdigest()
    assert candidate_specification_sha256 == authority[
        "candidate_specification_sha256"
    ]
    assert authority["method_reviewed_revision"] == (
        "ee512b31917fdf31d76e7237d3bba2b9c8ec4c64"
    )


@pytest.mark.unit
def test_lf_directional_component_closure_binds_only_reviewed_method_sources(
    tmp_path: Path,
) -> None:
    producer_root = _materialize_historical_component_authority(
        tmp_path=tmp_path,
        revision=LF_DIRECTIONAL_PRODUCER_REVISION,
        paths=LF_DIRECTIONAL_PRODUCER_PATHS,
        label="lf-directional-component-producer",
    )
    _assert_lf_directional_producer_authority(producer_root)


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
def test_hf_reference_component_closure_binds_exact_scientific_sources(
    tmp_path: Path,
) -> None:
    producer_root = _materialize_historical_component_authority(
        tmp_path=tmp_path,
        revision=HF_REFERENCE_PRODUCER_REVISION,
        paths=HF_REFERENCE_PRODUCER_PATHS,
        label="hf-reference-component-producer",
    )
    _assert_hf_reference_producer_authority(producer_root)


@pytest.mark.unit
def test_lf_directional_component_closure_rejects_historical_source_tamper(
    tmp_path: Path,
) -> None:
    producer_root = _materialize_historical_component_authority(
        tmp_path=tmp_path,
        revision=LF_DIRECTIONAL_PRODUCER_REVISION,
        paths=LF_DIRECTIONAL_PRODUCER_PATHS,
        label="lf-directional-component-source-tamper",
    )
    source = producer_root / "main/content_chain/lf_detector.py"
    source.write_bytes(source.read_bytes() + b"\n# historical source tamper\n")

    with pytest.raises(AssertionError):
        _assert_lf_directional_producer_authority(producer_root)


@pytest.mark.unit
def test_hf_reference_component_closure_rejects_historical_source_tamper(
    tmp_path: Path,
) -> None:
    producer_root = _materialize_historical_component_authority(
        tmp_path=tmp_path,
        revision=HF_REFERENCE_PRODUCER_REVISION,
        paths=HF_REFERENCE_PRODUCER_PATHS,
        label="hf-reference-component-source-tamper",
    )
    source = producer_root / "main/content_chain/hf_detector.py"
    source.write_bytes(source.read_bytes() + b"\n# historical source tamper\n")

    with pytest.raises(AssertionError):
        _assert_hf_reference_producer_authority(producer_root)


@pytest.mark.unit
@pytest.mark.parametrize("changed_path", HF_EXPECTED_PATHS)
def test_each_hf_scientific_source_change_updates_component_digest(
    tmp_path: Path,
    changed_path: str,
) -> None:
    _copy_component_sources(tmp_path, HF_EXPECTED_PATHS)
    components = _reviewed_components(HF_REFERENCE_COMPONENT_IDS)
    original = build_hf_reference_component_source_closure(components, tmp_path)

    source = tmp_path / changed_path
    source.write_text(
        source.read_text("utf-8") + "\n# changed reviewed HF component\n",
        encoding="utf-8",
    )
    changed = build_hf_reference_component_source_closure(components, tmp_path)

    assert changed.component_implementation_digest != (
        original.component_implementation_digest
    )


@pytest.mark.unit
def test_unrelated_planes_do_not_change_hf_component_digest(tmp_path: Path) -> None:
    _copy_component_sources(tmp_path, HF_EXPECTED_PATHS)
    components = _reviewed_components(HF_REFERENCE_COMPONENT_IDS)
    original = build_hf_reference_component_source_closure(components, tmp_path)
    for relative in (
        "main/__init__.py",
        "main/content_chain/__init__.py",
        "main/content_chain/lf_detector.py",
        "main/content_chain/routing.py",
        "main/geometry_chain/qk_sync.py",
        "runtime/adapter.py",
        "experiments/methods/ceg_wm.py",
        "scripts/experiment_execution/package_builder.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNRELATED_VALUE = 1\n", encoding="utf-8")

    replay = build_hf_reference_component_source_closure(components, tmp_path)

    assert replay == original


@pytest.mark.unit
def test_hf_component_closure_rejects_facade_missing_reordered_and_duplicate(
    tmp_path: Path,
) -> None:
    _copy_component_sources(tmp_path, HF_EXPECTED_PATHS)
    components = _reviewed_components(HF_REFERENCE_COMPONENT_IDS)

    facade = deepcopy(components)
    facade["hf_detector"]["implementation_path"] = "main/content_chain/__init__.py"
    with pytest.raises(ComponentSourceClosureError, match="concrete Python"):
        build_hf_reference_component_source_closure(facade, tmp_path)

    missing = deepcopy(components)
    del missing["hf_detector"]
    with pytest.raises(ComponentSourceClosureError, match="mapping is incomplete"):
        build_hf_reference_component_source_closure(missing, tmp_path)

    reordered = (
        HF_REFERENCE_COMPONENT_IDS[1],
        HF_REFERENCE_COMPONENT_IDS[0],
        *HF_REFERENCE_COMPONENT_IDS[2:],
    )
    with pytest.raises(ComponentSourceClosureError, match="order or membership"):
        build_component_source_closure(reordered, components, tmp_path)

    duplicate = (*HF_REFERENCE_COMPONENT_IDS[:-1], HF_REFERENCE_COMPONENT_IDS[-2])
    with pytest.raises(ComponentSourceClosureError, match="order or membership"):
        build_component_source_closure(duplicate, components, tmp_path)


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
