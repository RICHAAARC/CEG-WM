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


def _patch_clean_negative_runner(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """
    Patch PW01 clean_negative runtime calls to lightweight stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Captured stage input payloads.
    """

    captures: Dict[str, Any] = {
        "detect_probe_inputs": [],
        "detect_inputs": [],
    }

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

    def _read_detect_input_record(command: list[str]) -> Dict[str, Any]:
        if "--input" not in command:
            raise ValueError("detect command missing --input")
        input_index = command.index("--input")
        if input_index + 1 >= len(command):
            raise ValueError("detect command missing input record path")
        input_record_path = Path(command[input_index + 1])
        return cast(Dict[str, Any], json.loads(input_record_path.read_text(encoding="utf-8")))

    def fake_run_stage(stage_name: str, command: list[str], run_root: Path) -> Dict[str, Any]:
        records_root = ensure_directory(run_root / "records")
        logs_root = ensure_directory(run_root / "logs")
        (logs_root / f"{stage_name}_stdout.log").write_text("ok\n", encoding="utf-8")
        (logs_root / f"{stage_name}_stderr.log").write_text("\n", encoding="utf-8")

        if stage_name == "detect_probe":
            captures["detect_probe_inputs"].append(_read_detect_input_record(command))
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect_probe",
                    "plan_input_digest": "plan-input-probe",
                    "plan_input_schema_version": "v2",
                    "subspace_planner_impl_identity": {
                        "impl_id": "planner_impl",
                    },
                    "subspace_plan": {
                        "rank": 2,
                        "planner_input_digest": "plan-input-probe",
                    },
                    "content_evidence_payload": {
                        "status": "absent",
                        "plan_digest": "plan-probe",
                        "basis_digest": "basis-probe",
                    },
                },
            )
        elif stage_name == "detect":
            detect_input_record = _read_detect_input_record(command)
            captures["detect_inputs"].append(dict(detect_input_record))
            expected_plan_digest = detect_input_record.get("plan_digest")
            expected_basis_digest = detect_input_record.get("basis_digest")
            has_expected_plan = isinstance(expected_plan_digest, str) and bool(expected_plan_digest)
            content_status = "ok" if has_expected_plan else "absent"
            content_score = 0.12 if has_expected_plan else None
            content_failure_reason = None if has_expected_plan else "detector_no_plan_expected"
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect",
                    "plan_digest_expected": expected_plan_digest if has_expected_plan else None,
                    "plan_digest_observed": "plan-probe",
                    "plan_digest_status": "ok" if has_expected_plan else "absent",
                    "plan_digest_validation_status": "ok" if has_expected_plan else "absent",
                    "contract_bound_digest": "contract-bound",
                    "whitelist_bound_digest": "whitelist-bound",
                    "policy_path_semantics_bound_digest": "semantics-bound",
                    "contract_version": "contracts-v1",
                    "contract_digest": "contract-digest",
                    "contract_file_sha256": "contract-file-sha",
                    "contract_canon_sha256": "contract-canon-sha",
                    "whitelist_version": "whitelist-v1",
                    "whitelist_digest": "whitelist-digest",
                    "whitelist_file_sha256": "whitelist-file-sha",
                    "whitelist_canon_sha256": "whitelist-canon-sha",
                    "policy_path_semantics_version": "semantics-v1",
                    "policy_path_semantics_digest": "semantics-digest",
                    "policy_path_semantics_file_sha256": "semantics-file-sha",
                    "policy_path_semantics_canon_sha256": "semantics-canon-sha",
                    "content_evidence_payload": {
                        "status": content_status,
                        "score": content_score,
                        "content_chain_score": content_score,
                        "plan_digest": "plan-probe",
                        "basis_digest": expected_basis_digest if isinstance(expected_basis_digest, str) and expected_basis_digest else "basis-probe",
                        "content_failure_reason": content_failure_reason,
                    },
                    "final_decision": {
                        "is_watermarked": False,
                        "decision_status": "ok" if has_expected_plan else "error",
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

    return captures


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

    runtime_config_path = Path(str(first_event["runtime_config_path"]))
    prompt_run_root = runtime_config_path.parent / "run"
    detect_input_record_path = prompt_run_root / "artifacts" / "neg_preview_input" / "detect_input_record.json"
    assert detect_input_record_path.exists()
    assert not (prompt_run_root / "records" / "embed_record.json").exists()

    detect_input_record = json.loads(detect_input_record_path.read_text(encoding="utf-8"))
    assert detect_input_record["operation"] == "embed_preview_input"
    assert detect_input_record["negative_branch_source_attestation_provenance"]["statement"]["plan_digest"] == "plan-probe"
    assert detect_input_record["plan_digest"] == "plan-probe"
    assert detect_input_record["basis_digest"] == "basis-probe"
    assert detect_input_record["plan_input_digest"] == "plan-input-probe"
    assert detect_input_record["plan_input_schema_version"] == "v2"
    assert detect_input_record["subspace_planner_impl_identity"]["impl_id"] == "planner_impl"
    assert detect_input_record["subspace_plan"]["planner_input_digest"] == "plan-input-probe"

    detect_probe_command = cast(Dict[str, Any], first_event["stage_results"]["detect_probe"])["command"]
    detect_command = cast(Dict[str, Any], first_event["stage_results"]["detect"])["command"]
    assert "--input" in detect_probe_command
    assert detect_probe_command[detect_probe_command.index("--input") + 1] == str(detect_input_record_path)
    assert "--input" in detect_command
    assert detect_command[detect_command.index("--input") + 1] == str(detect_input_record_path)

    staged_embed_record = json.loads(Path(str(first_event["embed_record_path"])).read_text(encoding="utf-8"))
    assert staged_embed_record["negative_branch_source_attestation_provenance"]["statement"]["plan_digest"] == "plan-probe"
    assert staged_embed_record["basis_digest"] == "basis-probe"
    assert staged_embed_record["plan_input_digest"] == "plan-input-probe"
    assert staged_embed_record["plan_input_schema_version"] == "v2"
    assert staged_embed_record["subspace_planner_impl_identity"]["impl_id"] == "planner_impl"
    assert staged_embed_record["subspace_plan"]["planner_input_digest"] == "plan-input-probe"
    assert staged_embed_record["contract_bound_digest"] == "contract-bound"
    assert staged_embed_record["whitelist_bound_digest"] == "whitelist-bound"
    assert staged_embed_record["policy_path_semantics_bound_digest"] == "semantics-bound"
    assert "content_evidence" not in staged_embed_record
    assert "embed_trace" not in staged_embed_record
    assert "injection_evidence" not in staged_embed_record


def test_pw01_clean_negative_final_detect_consumes_probe_plan_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify final detect consumes the probe-derived planner anchors for clean_negative.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _build_pw00_family(tmp_path, family_id="family_clean_negative_regression")
    bound_config_path = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="family_clean_negative_regression",
    )
    _patch_clean_negative_runner(monkeypatch)

    summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_clean_negative_regression",
        sample_role="clean_negative",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    shard_root = Path(str(summary["shard_root"]))
    shard_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    first_event = cast(Dict[str, Any], shard_manifest["events"][0])
    detect_record = json.loads(Path(str(first_event["detect_record_path"])).read_text(encoding="utf-8"))
    detect_input_record_path = (
        Path(str(first_event["runtime_config_path"])).parent
        / "run"
        / "artifacts"
        / "neg_preview_input"
        / "detect_input_record.json"
    )
    detect_input_record = json.loads(detect_input_record_path.read_text(encoding="utf-8"))
    staged_embed_record = json.loads(Path(str(first_event["embed_record_path"])).read_text(encoding="utf-8"))

    assert detect_record["plan_digest_expected"] == "plan-probe"
    assert detect_record["plan_digest_status"] == "ok"
    assert detect_record["plan_digest_validation_status"] == "ok"
    assert detect_record["content_evidence_payload"]["status"] == "ok"
    assert detect_record["content_evidence_payload"].get("content_failure_reason") is None
    assert detect_input_record["plan_digest"] == staged_embed_record["plan_digest"]
    assert detect_input_record["basis_digest"] == staged_embed_record["basis_digest"]
    assert detect_input_record["plan_input_digest"] == staged_embed_record["plan_input_digest"]
    assert detect_input_record["plan_input_schema_version"] == staged_embed_record["plan_input_schema_version"]
    assert detect_input_record["subspace_planner_impl_identity"] == staged_embed_record["subspace_planner_impl_identity"]
    assert detect_input_record["subspace_plan"] == staged_embed_record["subspace_plan"]


def test_pw01_clean_negative_probe_and_final_detect_inputs_freeze_context_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Freeze the current clean_negative boundary between probe and final detect inputs.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Notes:
        The current architecture intentionally keeps probe detect and final detect
        in different attestation-conditioned input layers. This test freezes that
        asymmetry as a design boundary and simultaneously asserts that the final
        detect planner truth is still the probe-derived planner bundle, not a
        second detect-side canonical planner object.
    """
    _build_pw00_family(tmp_path, family_id="family_clean_negative_context_boundary")
    bound_config_path = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="family_clean_negative_context_boundary",
    )
    captures = _patch_clean_negative_runner(monkeypatch)

    summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_clean_negative_context_boundary",
        sample_role="clean_negative",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    shard_root = Path(str(summary["shard_root"]))
    shard_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    first_event = cast(Dict[str, Any], shard_manifest["events"][0])
    staged_embed_record = json.loads(Path(str(first_event["embed_record_path"])).read_text(encoding="utf-8"))

    event_count = len(cast(list[Dict[str, Any]], shard_manifest["events"]))

    assert len(captures["detect_probe_inputs"]) == event_count
    assert len(captures["detect_inputs"]) == event_count

    probe_input_record = cast(Dict[str, Any], captures["detect_probe_inputs"][0])
    final_input_record = cast(Dict[str, Any], captures["detect_inputs"][0])

    assert "negative_branch_source_attestation_provenance" not in probe_input_record
    assert "plan_digest" not in probe_input_record
    assert "basis_digest" not in probe_input_record
    assert "subspace_plan" not in probe_input_record

    assert final_input_record["negative_branch_source_attestation_provenance"]["statement"]["plan_digest"] == "plan-probe"
    assert final_input_record["plan_digest"] == "plan-probe"
    assert final_input_record["basis_digest"] == "basis-probe"
    assert final_input_record["plan_input_digest"] == "plan-input-probe"
    assert final_input_record["plan_input_schema_version"] == "v2"
    assert final_input_record["subspace_planner_impl_identity"] == staged_embed_record["subspace_planner_impl_identity"]
    assert final_input_record["subspace_plan"] == staged_embed_record["subspace_plan"]