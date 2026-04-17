"""
File purpose: Contract tests for paper_workflow local runtime bundle orchestration helpers.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from paper_workflow.scripts.pw_local_runtime import (
    CLEAN_NEGATIVE_ROLE,
    POSITIVE_SOURCE_ROLE,
    PW00_STAGE_NAME,
    PW01_STAGE_NAME,
    PW03_STAGE_NAME,
    PW04_FINALIZE_STAGE_NAME,
    archive_local_runtime_for_stage,
    discover_expected_shards,
    prepare_local_runtime_for_stage,
    read_bundle_sidecar,
    resolve_stage_dependencies,
    safe_clean_local_runtime,
    verify_bundle_integrity,
)


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    Write one JSON file for tests.

    Args:
        path_obj: Output path.
        payload: JSON payload.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path_obj: Path, text: str) -> None:
    """
    Write one UTF-8 text file for tests.

    Args:
        path_obj: Output path.
        text: File content.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(text, encoding="utf-8")


def _build_pw00_family_tree(
    local_project_root: Path,
    family_id: str,
    *,
    source_shard_count: int,
    attack_shard_count: int,
) -> Path:
    """
    Build one minimal PW00 family tree for local runtime tests.

    Args:
        local_project_root: Local project root.
        family_id: Family identifier.
        source_shard_count: Source shard count.
        attack_shard_count: Attack shard count.

    Returns:
        Family root path.
    """
    family_root = local_project_root / "paper_workflow" / "families" / family_id
    manifests_root = family_root / "manifests"
    runtime_state_root = family_root / "runtime_state"
    snapshots_root = family_root / "snapshots"

    _write_json(
        manifests_root / "paper_eval_family_manifest.json",
        {
            "family_id": family_id,
            "source_parameters": {
                "source_shard_count": source_shard_count,
                "attack_shard_count": attack_shard_count,
            },
            "sample_roles": {
                "active": [POSITIVE_SOURCE_ROLE, CLEAN_NEGATIVE_ROLE],
            },
        },
    )
    _write_text(manifests_root / "source_event_grid.jsonl", '{"event_id": "source_0000"}\n')
    _write_json(
        manifests_root / "source_shard_plan.json",
        {
            "source_shard_count": source_shard_count,
            "sample_role_plans": {
                POSITIVE_SOURCE_ROLE: {"shards": [{"shard_index": index} for index in range(source_shard_count)]},
                CLEAN_NEGATIVE_ROLE: {"shards": [{"shard_index": index} for index in range(source_shard_count)]},
            },
        },
    )
    _write_json(manifests_root / "source_split_plan.json", {"status": "ready"})
    _write_json(
        manifests_root / "attack_shard_plan.json",
        {
            "attack_shard_count": attack_shard_count,
            "shards": [{"attack_shard_index": index} for index in range(attack_shard_count)],
        },
    )
    _write_json(
        runtime_state_root / "pw00_summary.json",
        {
            "family_id": family_id,
            "source_shard_count": source_shard_count,
            "attack_shard_count": attack_shard_count,
        },
    )
    _write_text(snapshots_root / "config_snapshot.yaml", "model_id: test-model\n")
    return family_root


def test_safe_clean_local_runtime_rejects_protected_paths() -> None:
    """
    Verify local runtime cleanup rejects protected or unsafe paths.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(RuntimeError, match="must not equal /content"):
        safe_clean_local_runtime(Path("/content"))

    with pytest.raises(RuntimeError, match="must not contain drive"):
        safe_clean_local_runtime(Path("/content/drive/MyDrive/CEG_WM_PaperWorkflow"))

    with pytest.raises(RuntimeError, match="must not equal /content/ceg_wm_workspace"):
        safe_clean_local_runtime(Path("/content/ceg_wm_workspace"))


def test_safe_clean_local_runtime_does_not_allow_drive_attestation_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify safe local-runtime cleanup rejects Drive-backed attestation roots and
    never touches persistent Drive secrets while cleaning the local runtime root.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    local_project_root = Path("/content/CEG_WM_PaperWorkflow")
    drive_attestation_root = Path("/content/drive/MyDrive/CEG_WM_PaperWorkflow")
    drive_secrets_root = drive_attestation_root / "secrets"
    existing_paths = {
        local_project_root.as_posix(),
        drive_attestation_root.as_posix(),
        drive_secrets_root.as_posix(),
    }
    deleted_paths: list[str] = []
    path_type = type(local_project_root)

    def fake_exists(self: Path) -> bool:
        return self.as_posix() in existing_paths

    def fake_is_dir(self: Path) -> bool:
        return self.as_posix() in existing_paths

    def fake_rmtree(path_obj: Path) -> None:
        path_text = Path(path_obj).as_posix()
        deleted_paths.append(path_text)
        for existing_path in list(existing_paths):
            if existing_path == path_text or existing_path.startswith(f"{path_text}/"):
                existing_paths.discard(existing_path)

    monkeypatch.setattr(path_type, "exists", fake_exists, raising=False)
    monkeypatch.setattr(path_type, "is_dir", fake_is_dir, raising=False)
    monkeypatch.setattr("paper_workflow.scripts.pw_local_runtime.shutil.rmtree", fake_rmtree)

    with pytest.raises(RuntimeError, match="must not contain drive"):
        safe_clean_local_runtime(Path("/content/drive"))

    with pytest.raises(RuntimeError, match="must not contain drive"):
        safe_clean_local_runtime(drive_attestation_root)

    clean_summary = safe_clean_local_runtime(local_project_root)

    assert clean_summary["status"] == "cleaned"
    assert clean_summary["local_project_root"] == local_project_root.as_posix()
    assert deleted_paths == [local_project_root.as_posix()]
    assert drive_secrets_root.as_posix() in existing_paths


def test_read_optional_json_returns_none_for_missing_path(tmp_path: Path) -> None:
    """
    Verify read_optional_json returns None when the path is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    assert read_optional_json(tmp_path / "missing.json") is None


def test_read_optional_json_returns_none_for_directory(tmp_path: Path) -> None:
    """
    Verify read_optional_json returns None for directory paths.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    assert read_optional_json(tmp_path) is None


def test_read_optional_json_returns_none_for_invalid_json(tmp_path: Path) -> None:
    """
    Verify read_optional_json returns None for invalid JSON payloads.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    payload_path = tmp_path / "invalid.json"
    payload_path.write_text("{invalid", encoding="utf-8")

    assert read_optional_json(payload_path) is None


def test_read_optional_json_returns_none_for_non_object_json(tmp_path: Path) -> None:
    """
    Verify read_optional_json returns None when the JSON root is not a mapping.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    payload_path = tmp_path / "list.json"
    payload_path.write_text("[1, 2, 3]", encoding="utf-8")

    assert read_optional_json(payload_path) is None


def test_read_optional_json_reads_json_object(tmp_path: Path) -> None:
    """
    Verify read_optional_json returns parsed mappings for valid JSON objects.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    assert read_optional_json(payload_path) == {"ok": True}


def test_read_optional_json_rejects_non_path_argument() -> None:
    """
    Verify read_optional_json rejects non-Path inputs.

    Args:
        None.

    Returns:
        None.
    """
    from scripts.notebook_runtime_common import read_optional_json

    with pytest.raises(TypeError, match="path_obj must be Path"):
        read_optional_json("not-a-path")  # type: ignore[arg-type]


def test_pw00_bundle_archive_and_pw01_prepare_roundtrip(tmp_path: Path) -> None:
    """
    Verify PW00 bundle archive can be restored by PW01 prepare without Drive live mode.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    family_id = "family_local_runtime_roundtrip"
    producer_root = tmp_path / "producer_root"
    consumer_root = tmp_path / "consumer_root"
    drive_bundle_root = tmp_path / "drive_bundles"

    family_root = _build_pw00_family_tree(
        producer_root,
        family_id,
        source_shard_count=3,
        attack_shard_count=5,
    )

    archive_summary = archive_local_runtime_for_stage(
        stage_name=PW00_STAGE_NAME,
        family_id=family_id,
        local_project_root=producer_root,
        drive_bundle_root=drive_bundle_root,
        clean_after_archive=False,
    )
    sidecar_path = Path(str(archive_summary["sidecar_path"]))
    sidecar_payload = read_bundle_sidecar(sidecar_path)
    integrity_summary = verify_bundle_integrity(sidecar_path)

    assert sidecar_payload["bundle_kind"] == "pw00_bootstrap"
    assert sidecar_payload["source_shard_count"] == 3
    assert sidecar_payload["attack_shard_count"] == 5
    assert integrity_summary["family_id"] == family_id

    prepare_summary = prepare_local_runtime_for_stage(
        stage_name=PW01_STAGE_NAME,
        family_id=family_id,
        local_project_root=consumer_root,
        drive_bundle_root=drive_bundle_root,
        sample_role=POSITIVE_SOURCE_ROLE,
        shard_index=0,
        shard_count=3,
        clean_before_run=False,
        include_optional_control_negative=False,
    )

    restored_family_root = consumer_root / "paper_workflow" / "families" / family_id
    assert prepare_summary["status"] == "ready"
    assert (restored_family_root / "manifests" / "paper_eval_family_manifest.json").exists()
    assert (restored_family_root / "manifests" / "source_event_grid.jsonl").exists()
    assert (restored_family_root / "runtime_state" / "pw00_summary.json").exists()
    assert (restored_family_root / "snapshots" / "config_snapshot.yaml").exists()
    assert str(family_root.name) == family_id


def test_discover_expected_shards_and_resolve_dependencies_are_dynamic(tmp_path: Path) -> None:
    """
    Verify shard discovery and dependency resolution derive counts from plans instead of hardcoded pilot values.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    family_id = "family_dynamic_shards"
    local_project_root = tmp_path / "local_runtime"
    drive_bundle_root = tmp_path / "drive_bundles"

    family_root = _build_pw00_family_tree(
        local_project_root,
        family_id,
        source_shard_count=3,
        attack_shard_count=5,
    )
    _write_json(
        family_root / "exports" / "pw04" / "quality" / "quality_pair_plan.json",
        {
            "shards": [
                {"quality_shard_index": 0},
                {"quality_shard_index": 1},
            ]
        },
    )

    positive_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="source",
        sample_role=POSITIVE_SOURCE_ROLE,
    )
    attack_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="attack",
    )
    quality_discovery = discover_expected_shards(
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
        shard_kind="quality",
    )
    resolution = resolve_stage_dependencies(
        stage_name=PW04_FINALIZE_STAGE_NAME,
        family_id=family_id,
        local_project_root=local_project_root,
        drive_bundle_root=drive_bundle_root,
    )

    pw03_dependencies = [
        bundle for bundle in resolution["required_bundles"] if bundle["stage_name"] == PW03_STAGE_NAME
    ]
    pw04_quality_dependencies = [
        bundle
        for bundle in resolution["required_bundles"]
        if bundle["stage_name"] == "PW04_Attack_Merge_And_Metrics_quality_shard"
    ]

    assert positive_discovery["shard_count"] == 3
    assert positive_discovery["expected_indices"] == [0, 1, 2]
    assert attack_discovery["shard_count"] == 5
    assert attack_discovery["expected_indices"] == [0, 1, 2, 3, 4]
    assert quality_discovery["shard_count"] == 2
    assert quality_discovery["expected_indices"] == [0, 1]
    assert len(pw03_dependencies) == 5
    assert len(pw04_quality_dependencies) == 2
