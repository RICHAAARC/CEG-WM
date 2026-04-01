"""
File purpose: Unit tests for PW01 shard path isolation and rerun protection.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module
from scripts.notebook_runtime_common import (
    apply_notebook_model_snapshot_binding,
    ensure_directory,
    load_yaml_mapping,
    write_json_atomic,
    write_yaml_mapping,
)


def _build_pw00_family(tmp_path: Path, family_id: str = "family_pw01_fixture") -> Dict[str, Any]:
    """
    Build a PW00 family fixture for PW01 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "pw01_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[3, 9],
        source_shard_count=2,
    )


def _write_bound_config_snapshot(drive_project_root: Path, *, marker: str) -> Path:
    """
    Build a notebook-style bound config snapshot for PW01 tests.

    Args:
        drive_project_root: Drive project root.
        marker: Stable marker stored in the bound config.

    Returns:
        Bound config snapshot path.
    """
    snapshot_dir = drive_project_root / "runtime_state" / f"{marker}_model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    bound_cfg = apply_notebook_model_snapshot_binding(
        load_yaml_mapping((pw01_module.REPO_ROOT / "configs" / "default.yaml").resolve()),
        env_mapping={"CEG_WM_MODEL_SNAPSHOT_PATH": snapshot_dir.as_posix()},
    )
    bound_cfg["test_config_origin"] = marker

    bound_config_path = drive_project_root / "runtime_state" / f"{marker}_bound_config.yaml"
    write_yaml_mapping(bound_config_path, bound_cfg)
    return bound_config_path


def _patch_pw01_base_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch PW01 base-runner calls to lightweight CPU-safe stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def fake_build_stage_command(stage_name: str, config_path: Path, run_root: Path) -> list[str]:
        return [stage_name, str(config_path), str(run_root)]

    def fake_prepare_source_pool_preview_artifact(
        *,
        cfg_obj: Dict[str, Any],
        prompt_run_root: Path,
        prompt_text: str,
        prompt_index: int,
        prompt_file_path: str,
    ) -> Dict[str, Any]:
        preview_image_path = prompt_run_root / "artifacts" / "preview" / "preview.png"
        ensure_directory(preview_image_path.parent)
        preview_image_path.write_bytes(b"preview")

        preview_record_path = prompt_run_root / "artifacts" / "preview_generation_record.json"
        preview_record: Dict[str, Any] = {
            "status": "ok",
            "prompt_text": prompt_text,
            "prompt_index": prompt_index,
            "prompt_file": prompt_file_path,
            "persisted_artifact_path": preview_image_path.as_posix(),
        }
        write_json_atomic(preview_record_path, preview_record)

        runtime_cfg: Dict[str, Any] = dict(cfg_obj)
        embed_node = runtime_cfg.get("embed")
        embed_cfg: Dict[str, Any]
        if isinstance(embed_node, dict):
            embed_cfg = cast(Dict[str, Any], embed_node.copy())
        else:
            embed_cfg = {}
        embed_cfg["input_image_path"] = preview_image_path.as_posix()
        runtime_cfg["embed"] = embed_cfg
        return {
            "runtime_cfg": runtime_cfg,
            "preview_record": preview_record,
        }

    def fake_run_stage(stage_name: str, command: list[str], run_root: Path) -> Dict[str, Any]:
        records_root = ensure_directory(run_root / "records")
        logs_root = ensure_directory(run_root / "logs")
        (logs_root / f"{stage_name}_stdout.log").write_text("ok\n", encoding="utf-8")
        (logs_root / f"{stage_name}_stderr.log").write_text("\n", encoding="utf-8")

        if stage_name == "embed":
            write_json_atomic(records_root / "embed_record.json", {"record_type": "embed"})
        elif stage_name == "detect":
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect",
                    "content_evidence_payload": {
                        "status": "ok",
                        "score": 0.75,
                        "content_chain_score": 0.75,
                    },
                    "attestation": {
                        "final_event_attested_decision": {
                            "event_attestation_score_name": "event_attestation_score",
                            "event_attestation_score": 0.61,
                        }
                    },
                },
            )
        else:
            raise ValueError(f"unsupported stage: {stage_name}")

        return {
            "return_code": 0,
            "command": command,
            "stdout_log_path": (logs_root / f"{stage_name}_stdout.log").as_posix(),
            "stderr_log_path": (logs_root / f"{stage_name}_stderr.log").as_posix(),
            "status": "ok",
        }

    def fake_resolve_source_pool_attestation_views(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        artifact_root = ensure_directory(run_root / "artifacts" / "mock_attestation" / f"event_{prompt_index:06d}")
        statement_path = artifact_root / "attestation_statement.json"
        bundle_path = artifact_root / "attestation_bundle.json"
        result_path = artifact_root / "attestation_result.json"
        write_json_atomic(statement_path, {"status": "ok"})
        write_json_atomic(bundle_path, {"status": "ok"})
        write_json_atomic(result_path, {"status": "ok"})
        return {
            "attestation_statement": {
                "exists": True,
                "path": statement_path.as_posix(),
                "package_relative_path": statement_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
            "attestation_bundle": {
                "exists": True,
                "path": bundle_path.as_posix(),
                "package_relative_path": bundle_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
            "attestation_result": {
                "exists": True,
                "path": result_path.as_posix(),
                "package_relative_path": result_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
        }

    def fake_resolve_source_pool_source_image_view(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        source_image_path = run_root / "artifacts" / "mock_source_images" / f"event_{prompt_index:06d}.png"
        ensure_directory(source_image_path.parent)
        source_image_path.write_bytes(b"img")
        return {
            "exists": True,
            "path": source_image_path.as_posix(),
            "package_relative_path": source_image_path.relative_to(run_root).as_posix(),
            "missing_reason": None,
        }

    def fake_resolve_source_pool_preview_generation_record_view(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        record_path = run_root / "artifacts" / "mock_preview_records" / f"event_{prompt_index:06d}.json"
        ensure_directory(record_path.parent)
        write_json_atomic(record_path, {"status": "ok"})
        return {
            "exists": True,
            "path": record_path.as_posix(),
            "package_relative_path": record_path.relative_to(run_root).as_posix(),
            "missing_reason": None,
        }

    def fake_normalize_direct_detect_payload(
        payload: Dict[str, Any],
        *,
        prompt_text: str,
        prompt_index: int,
        prompt_file_path: str,
        record_usage: str,
    ) -> Dict[str, Any]:
        payload = dict(payload)
        payload["label"] = True
        payload["prompt_text"] = prompt_text
        payload["prompt_index"] = prompt_index
        payload["prompt_file"] = prompt_file_path
        payload["record_usage"] = record_usage
        return payload

    monkeypatch.setattr(pw01_module.BASE_RUNNER_MODULE, "_build_stage_command", fake_build_stage_command)
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_prepare_source_pool_preview_artifact",
        fake_prepare_source_pool_preview_artifact,
    )
    monkeypatch.setattr(pw01_module.BASE_RUNNER_MODULE, "_run_stage", fake_run_stage)
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_attestation_views",
        fake_resolve_source_pool_attestation_views,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_source_image_view",
        fake_resolve_source_pool_source_image_view,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_preview_generation_record_view",
        fake_resolve_source_pool_preview_generation_record_view,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_normalize_direct_detect_payload",
        fake_normalize_direct_detect_payload,
    )


def test_pw01_writes_only_current_shard_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify PW01 only writes files under the selected shard directory.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_isolation")
    bound_config_path = _write_bound_config_snapshot(tmp_path / "drive", marker="family_isolation")
    _patch_pw01_base_runner(monkeypatch)

    family_root = Path(str(summary["family_root"]))
    before_files = {path.resolve() for path in family_root.rglob("*") if path.is_file()}

    pw01_summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_isolation",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    shard_root = Path(str(pw01_summary["shard_root"])).resolve()
    after_files = {path.resolve() for path in family_root.rglob("*") if path.is_file()}
    created_files = after_files - before_files

    assert created_files
    for created_path in created_files:
        created_path.relative_to(shard_root)

    assert (shard_root / "events").exists()
    assert (shard_root / "records").exists()
    assert (shard_root / "artifacts").exists()
    assert (shard_root / "logs").exists()
    assert not (family_root / "source_shards" / "positive" / "shard_0001").exists()

    shard_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    assert shard_manifest["status"] == "completed"


def test_pw01_completed_shard_rerun_protection_and_force_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify completed shard rerun is blocked unless force_rerun is enabled.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _build_pw00_family(tmp_path, family_id="family_rerun")
    bound_config_path = _write_bound_config_snapshot(tmp_path / "drive", marker="family_rerun")
    _patch_pw01_base_runner(monkeypatch)

    first_summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_rerun",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )
    shard_root = Path(str(first_summary["shard_root"]))

    with pytest.raises(RuntimeError, match="completed"):
        pw01_module.run_pw01_source_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="family_rerun",
            shard_index=0,
            shard_count=2,
            bound_config_path=bound_config_path,
            force_rerun=False,
        )

    marker_path = shard_root / "stale_marker.txt"
    marker_path.write_text("stale", encoding="utf-8")

    second_summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_rerun",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
        force_rerun=True,
    )

    assert Path(str(second_summary["shard_root"])).resolve() == shard_root.resolve()
    assert not marker_path.exists()
    rerun_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    assert rerun_manifest["status"] == "completed"
