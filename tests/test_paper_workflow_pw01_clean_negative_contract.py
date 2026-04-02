"""
File purpose: Validate PW01 clean_negative shard execution contract.
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


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a PW00 fixture family for clean_negative PW01 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "pw01_negative_prompts.txt"
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


def _patch_clean_negative_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch PW01 clean_negative runtime calls to lightweight stubs.

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
        preview_image_path.write_bytes(b"preview-negative")

        preview_record_path = prompt_run_root / "artifacts" / "preview_generation_record.json"
        preview_record: Dict[str, Any] = {
            "status": "ok",
            "prompt_text": prompt_text,
            "prompt_index": prompt_index,
            "prompt_file": prompt_file_path,
            "persisted_artifact_path": preview_image_path.as_posix(),
        }
        write_json_atomic(preview_record_path, preview_record)
        return {
            "runtime_cfg": dict(cfg_obj),
            "preview_record": preview_record,
        }

    def fake_run_stage(stage_name: str, command: list[str], run_root: Path) -> Dict[str, Any]:
        records_root = ensure_directory(run_root / "records")
        logs_root = ensure_directory(run_root / "logs")
        (logs_root / f"{stage_name}_stdout.log").write_text("ok\n", encoding="utf-8")
        (logs_root / f"{stage_name}_stderr.log").write_text("\n", encoding="utf-8")

        if stage_name == "detect_probe":
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect_probe",
                    "content_evidence_payload": {
                        "status": "absent",
                        "plan_digest": "plan-probe",
                        "basis_digest": "basis-probe",
                    },
                },
            )
        elif stage_name == "detect":
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect",
                    "content_evidence_payload": {
                        "status": "ok",
                        "score": 0.12,
                        "content_chain_score": 0.12,
                        "plan_digest": "plan-probe",
                        "basis_digest": "basis-probe",
                    },
                    "final_decision": {
                        "is_watermarked": False,
                    },
                    "attestation": {
                        "final_event_attested_decision": {
                            "status": "absent",
                            "is_event_attested": False,
                            "event_attestation_score_name": "event_attestation_score",
                            "event_attestation_score": 0.0,
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
    monkeypatch.setattr(
        pw01_module,
        "_build_negative_branch_attestation_provenance",
        lambda **_: {
            "statement": {
                "schema": "gen_attest_v1",
                "model_id": "stub-model",
                "prompt_commit": "prompt-commit",
                "seed_commit": "seed-commit",
                "plan_digest": "plan-probe",
                "event_nonce": "nonce",
                "time_bucket": "2026-01-01",
            },
            "attestation_digest": "digest",
        },
    )


def test_pw01_clean_negative_writes_statement_only_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify clean_negative PW01 shards stage a probe detect record and final provenance.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _build_pw00_family(tmp_path, family_id="family_clean_negative")
    bound_config_path = _write_bound_config_snapshot(tmp_path / "drive", marker="family_clean_negative")
    _patch_clean_negative_runner(monkeypatch)

    summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_clean_negative",
        sample_role="clean_negative",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    assert summary["sample_role"] == "clean_negative"
    shard_root = Path(str(summary["shard_root"]))
    assert shard_root.as_posix().endswith("source_shards/negative/shard_0000")

    shard_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    assert shard_manifest["status"] == "completed"
    assert shard_manifest["sample_role"] == "clean_negative"
    assert shard_manifest["events"]

    first_event = cast(Dict[str, Any], shard_manifest["events"][0])
    assert first_event["negative_branch_source_attestation_provenance"]["attestation_digest"] == "digest"
    assert Path(str(first_event["detect_probe_record_path"])).exists()
    assert first_event["stage_results"]["detect_probe"]["status"] == "ok"
    assert first_event["stage_results"]["detect"]["status"] == "ok"

    staged_embed_record = json.loads(Path(str(first_event["embed_record_path"])).read_text(encoding="utf-8"))
    assert staged_embed_record["negative_branch_source_attestation_provenance"]["statement"]["plan_digest"] == "plan-probe"
    assert staged_embed_record["basis_digest"] == "basis-probe"