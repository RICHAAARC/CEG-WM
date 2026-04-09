"""
File purpose: Validate PW03 attack shard execution, isolation, and artifact contracts.
Module type: General module
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

from PIL import Image
import pytest

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
import paper_workflow.scripts.pw03_run_attack_event_shard as pw03_module
from scripts.notebook_runtime_common import (
    apply_notebook_model_snapshot_binding,
    ensure_directory,
    load_yaml_mapping,
    write_json_atomic,
    write_yaml_mapping,
    build_repo_import_subprocess_env,
)


def _build_pw00_family(
    tmp_path: Path,
    family_id: str,
    attack_shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    Build a minimal PW00 family fixture for PW03 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.
        attack_shard_count: Optional PW03 attack shard count.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / f"{family_id}_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[3],
        source_shard_count=2,
        attack_shard_count=attack_shard_count,
    )


def _write_bound_config_snapshot(drive_project_root: Path, *, marker: str) -> Tuple[Path, Path]:
    """
    Build a notebook-style bound config snapshot and model snapshot root.

    Args:
        drive_project_root: Drive project root.
        marker: Stable fixture marker.

    Returns:
        Tuple of bound config path and model snapshot directory.
    """
    snapshot_dir = drive_project_root / "runtime_state" / f"{marker}_model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    bound_cfg = apply_notebook_model_snapshot_binding(
        load_yaml_mapping((pw03_module.REPO_ROOT / "configs" / "default.yaml").resolve()),
        env_mapping={"CEG_WM_MODEL_SNAPSHOT_PATH": snapshot_dir.as_posix()},
    )
    bound_cfg["test_config_origin"] = marker
    bound_config_path = drive_project_root / "runtime_state" / f"{marker}_bound_config.yaml"
    write_yaml_mapping(bound_config_path, bound_cfg)
    return bound_config_path, snapshot_dir


def _build_positive_source_finalize_fixture(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build minimal PW01 and PW02 artifacts required by PW03.

    Args:
        summary: PW00 summary payload.

    Returns:
        Fixture metadata for PW03 tests.
    """
    family_root = Path(str(summary["family_root"]))
    family_id = str(summary["family_id"])
    source_event_grid_rows = [
        row
        for row in pw03_module.read_jsonl(Path(str(summary["source_event_grid_path"])))
        if row.get("sample_role") in {pw03_module.ACTIVE_SAMPLE_ROLE, pw03_module.CLEAN_NEGATIVE_SAMPLE_ROLE}
    ]
    source_shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))
    event_to_shard_index: Dict[str, int] = {}
    event_to_sample_role: Dict[str, str] = {}
    for sample_role in [pw03_module.ACTIVE_SAMPLE_ROLE, pw03_module.CLEAN_NEGATIVE_SAMPLE_ROLE]:
        role_shards = cast(
            List[Dict[str, Any]],
            cast(Dict[str, Any], source_shard_plan["sample_role_plans"])[sample_role]["shards"],
        )
        for shard_row in role_shards:
            shard_index = int(shard_row["shard_index"])
            for event_id in cast(List[str], shard_row["assigned_event_ids"]):
                event_to_shard_index[event_id] = shard_index
                event_to_sample_role[event_id] = sample_role

    stage_root = ensure_directory(family_root / "source_finalize")
    positive_pool_events: List[Dict[str, Any]] = []
    clean_negative_pool_events: List[Dict[str, Any]] = []
    shard_event_manifests: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for event_row in source_event_grid_rows:
        event_id = str(event_row["event_id"])
        event_index = int(event_row["event_index"])
        shard_index = event_to_shard_index[event_id]
        sample_role = event_to_sample_role[event_id]
        shard_directory_name = "positive" if sample_role == pw03_module.ACTIVE_SAMPLE_ROLE else "negative"
        shard_root = ensure_directory(family_root / "source_shards" / shard_directory_name / f"shard_{shard_index:04d}")
        event_root = ensure_directory(shard_root / "events" / f"event_{event_index:06d}")
        ensure_directory(shard_root / "records")
        image_path = shard_root / "artifacts" / "mock_source_images" / f"event_{event_index:06d}.png"
        ensure_directory(image_path.parent)
        Image.new("RGB", (8, 8), color=(32 + event_index, 16, 96)).save(image_path)

        embed_record_path = shard_root / "records" / f"event_{event_index:06d}_embed_record.json"
        detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_record.json"
        runtime_config_path = event_root / "runtime_config.yaml"
        embed_record = {
            "operation": "embed",
            "watermarked_path": image_path.as_posix(),
            "image_path": image_path.as_posix(),
            "inputs": {
                "input_image_path": image_path.as_posix(),
            },
            "artifact_sha256": pw03_module.compute_file_sha256(image_path),
            "watermarked_artifact_sha256": pw03_module.compute_file_sha256(image_path),
            "is_watermarked": True,
            "plan_digest": f"plan_{event_index:06d}",
            "basis_digest": f"basis_{event_index:06d}",
            "subspace_plan": {"mode": "fixture"},
            "prompt_text": event_row["prompt_text"],
            "prompt_sha256": event_row["prompt_sha256"],
            "seed": event_row["seed"],
        }
        detect_record = {
            "content_evidence_payload": {
                "status": "ok",
                "content_chain_score": 0.8,
            },
            "attestation": {
                "final_event_attested_decision": {
                    "event_attestation_score": 0.7,
                }
            },
            "final_decision": {
                "decision_status": "accept",
                "is_watermarked": True,
            },
        }
        write_json_atomic(embed_record_path, embed_record)
        write_json_atomic(detect_record_path, detect_record)
        write_yaml_mapping(
            runtime_config_path,
            {
                "paper_workflow_event": {
                    "event_id": event_id,
                    "event_index": event_index,
                    "sample_role": sample_role,
                }
            },
        )

        event_manifest_path = event_root / "event_manifest.json"
        event_manifest = {
            "artifact_type": "paper_workflow_source_event",
            "event_id": event_id,
            "sample_role": sample_role,
            "event_index": event_index,
            "prompt_text": event_row["prompt_text"],
            "prompt_sha256": event_row["prompt_sha256"],
            "seed": event_row["seed"],
            "runtime_config_path": runtime_config_path.as_posix(),
            "embed_record_path": embed_record_path.as_posix(),
            "detect_record_path": detect_record_path.as_posix(),
            "source_image": {
                "exists": True,
                "path": image_path.as_posix(),
                "package_relative_path": image_path.relative_to(shard_root).as_posix(),
                "missing_reason": None,
            },
            "preview_generation_record": None,
            "attestation_statement": None,
            "attestation_bundle": None,
            "attestation_result": None,
            "sha256": pw03_module.compute_file_sha256(detect_record_path),
            "stage_results": {},
        }
        write_json_atomic(event_manifest_path, event_manifest)
        event_manifest["event_manifest_path"] = event_manifest_path.as_posix()

        shard_event_manifests.setdefault((sample_role, shard_index), []).append(event_manifest)
        pool_event = {
            "event_id": event_id,
            "event_index": event_index,
            "sample_role": sample_role,
            "detect_record_path": detect_record_path.as_posix(),
            "source_shard_index": shard_index,
            "source_shard_root": shard_root.as_posix(),
            "source_shard_manifest_path": (shard_root / "shard_manifest.json").as_posix(),
        }
        if sample_role == pw03_module.ACTIVE_SAMPLE_ROLE:
            positive_pool_events.append(pool_event)
        else:
            clean_negative_pool_events.append(pool_event)

    for (sample_role, shard_index), event_manifests in shard_event_manifests.items():
        shard_directory_name = "positive" if sample_role == pw03_module.ACTIVE_SAMPLE_ROLE else "negative"
        shard_root = family_root / "source_shards" / shard_directory_name / f"shard_{shard_index:04d}"
        write_json_atomic(
            shard_root / "shard_manifest.json",
            {
                "status": "completed",
                "sample_role": sample_role,
                "event_count": len(event_manifests),
                "events": event_manifests,
            },
        )

    positive_pool_manifest_path = stage_root / "positive_source_pool_manifest.json"
    write_json_atomic(
        positive_pool_manifest_path,
        {
            "artifact_type": "paper_workflow_pw02_source_pool_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "source_role": pw03_module.ACTIVE_SAMPLE_ROLE,
            "event_count": len(positive_pool_events),
            "events": positive_pool_events,
        },
    )

    empty_pool_payload = {
        "artifact_type": "paper_workflow_pw02_source_pool_manifest",
        "schema_version": "pw_stage_02_v1",
        "family_id": family_id,
        "event_count": 0,
        "events": [],
    }
    clean_negative_pool_manifest_path = stage_root / "clean_negative_pool_manifest.json"
    control_pool_manifest_path = stage_root / "planner_conditioned_control_negative_pool_manifest.json"
    write_json_atomic(
        clean_negative_pool_manifest_path,
        {
            **empty_pool_payload,
            "source_role": pw03_module.CLEAN_NEGATIVE_SAMPLE_ROLE,
            "event_count": len(clean_negative_pool_events),
            "events": clean_negative_pool_events,
        },
    )
    write_json_atomic(control_pool_manifest_path, empty_pool_payload)

    content_threshold_path = stage_root / "thresholds" / "content" / "thresholds.json"
    attestation_threshold_path = stage_root / "thresholds" / "attestation" / "thresholds.json"
    ensure_directory(content_threshold_path.parent)
    ensure_directory(attestation_threshold_path.parent)
    write_json_atomic(content_threshold_path, {"threshold": 0.5, "score_name": "content_chain_score"})
    write_json_atomic(attestation_threshold_path, {"threshold": 0.4, "score_name": "event_attestation_score"})

    finalize_manifest_path = stage_root / "paper_source_finalize_manifest.json"
    write_json_atomic(
        finalize_manifest_path,
        {
            "artifact_type": "paper_workflow_pw02_finalize_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "source_pools": {
                pw03_module.ACTIVE_SAMPLE_ROLE: {
                    "manifest_path": positive_pool_manifest_path.as_posix(),
                    "event_count": len(positive_pool_events),
                },
                "clean_negative": {
                    "manifest_path": clean_negative_pool_manifest_path.as_posix(),
                    "event_count": len(clean_negative_pool_events),
                },
                "planner_conditioned_control_negative": {
                    "manifest_path": control_pool_manifest_path.as_posix(),
                    "event_count": 0,
                },
            },
            "threshold_exports": {
                "content": {"path": content_threshold_path.as_posix()},
                "attestation": {"path": attestation_threshold_path.as_posix()},
            },
        },
    )
    write_json_atomic(
        family_root / "runtime_state" / "pw02_summary.json",
        {
            "status": "completed",
            "family_id": family_id,
            "paper_source_finalize_manifest_path": finalize_manifest_path.as_posix(),
        },
    )

    return {
        "family_root": family_root,
        "positive_source_pool_manifest_path": positive_pool_manifest_path,
        "clean_negative_pool_manifest_path": clean_negative_pool_manifest_path,
        "finalize_manifest_path": finalize_manifest_path,
        "positive_event_count": len(positive_pool_events),
        "clean_negative_event_count": len(clean_negative_pool_events),
    }


def _patch_pw03_detect(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """
    Patch PW03 detect execution with lightweight artifact writers.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mutable capture mapping.
    """
    captures: Dict[str, Any] = {
        "run_roots": [],
        "detect_input_paths": [],
        "detect_input_payloads": [],
    }

    def fake_run_command_with_gpu_monitor(
        *,
        command: List[str],
        label: str,
        gpu_summary_path: Path,
        stdout_log_path: Path,
        stderr_log_path: Path,
    ) -> Dict[str, Any]:
        run_root = Path(str(command[command.index("--out") + 1]))
        detect_input_path = Path(str(command[command.index("--input") + 1]))
        detect_input_payload = json.loads(detect_input_path.read_text(encoding="utf-8"))
        captures["run_roots"].append(run_root)
        captures["detect_input_paths"].append(detect_input_path)
        captures["detect_input_payloads"].append(detect_input_payload)

        ensure_directory(stdout_log_path.parent)
        ensure_directory(stderr_log_path.parent)
        ensure_directory(run_root / "records")
        stdout_log_path.write_text("ok\n", encoding="utf-8")
        stderr_log_path.write_text("\n", encoding="utf-8")
        write_json_atomic(
            run_root / "records" / "detect_record.json",
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "content_chain_score": 0.91,
                    "plan_digest": detect_input_payload.get("plan_digest"),
                    "basis_digest": detect_input_payload.get("basis_digest"),
                },
                "geometry_result": {
                    "sync_status": "ok",
                    "sync_result": {
                        "sync_success": True,
                        "failure_reason": None,
                        "sync_quality_metrics": {
                            "match_score": 0.92,
                        },
                        "template_match_metrics": {
                            "peak_value": 0.81,
                        },
                    },
                    "sync_digest": "sync_digest_test",
                    "relation_binding_diagnostics": {
                        "binding_status": "ok",
                    },
                    "relation_digest_bound": True,
                },
                "geometry_evidence_payload": {
                    "status": "ok",
                    "anchor_digest": "anchor_digest_test",
                    "align_metrics": {
                        "inverse_recovery_success": True,
                    },
                    "sync_quality_metrics": {
                        "match_score": 0.92,
                    },
                    "template_match_metrics": {
                        "peak_value": 0.81,
                    },
                },
                "attestation": {
                    "final_event_attested_decision": {
                        "event_attestation_score": 0.77,
                    }
                },
                "final_decision": {
                    "decision_status": "accept",
                    "is_watermarked": True,
                },
            },
        )
        gpu_payload = {
            "status": "ok",
            "session_board_peak_memory_used_mib": 1536 + len(captures["run_roots"]),
            "peak_observed_at_utc": "2026-04-05T00:00:00Z",
            "peak_gpu_name": "Test GPU",
            "visible_gpu_count": 1,
            "visible_gpus": [
                {
                    "index": 0,
                    "uuid": "gpu-0",
                    "name": "Test GPU",
                    "memory_total_mib": 24576,
                    "peak_memory_used_mib": 1536 + len(captures["run_roots"]),
                }
            ],
            "wrapped_return_code": 0,
        }
        write_json_atomic(gpu_summary_path, gpu_payload)
        return {
            "return_code": 0,
            "stdout_log_path": stdout_log_path.as_posix(),
            "stderr_log_path": stderr_log_path.as_posix(),
            "command": [str(item) for item in command],
            "gpu_session_peak_path": gpu_summary_path.as_posix(),
            "gpu_session_peak": gpu_payload,
        }

    monkeypatch.setattr(pw03_module, "_run_command_with_gpu_monitor", fake_run_command_with_gpu_monitor)
    return captures


def _patch_pw03_worker_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch PW03 worker subprocess launches with an in-process executor.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    class _FakePopen:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> type["_FakePopen"]:
            return cls

        def __init__(
            self,
            command: List[str],
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            stdout: Any = None,
            stderr: Any = None,
            text: bool | None = None,
        ) -> None:
            self.command = [str(item) for item in command]
            self.cwd = cwd
            self.env = env
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self._return_code: int | None = None

        def _arg_value(self, flag: str) -> str:
            return self.command[self.command.index(flag) + 1]

        def wait(self) -> int:
            if self._return_code is None:
                if self.stdout is not None:
                    self.stdout.write("worker stdout\n")
                    self.stdout.flush()
                if self.stderr is not None:
                    self.stderr.write("")
                    self.stderr.flush()
                pw03_module.run_pw03_attack_event_worker(
                    drive_project_root=Path(self._arg_value("--drive-project-root")),
                    family_id=self._arg_value("--family-id"),
                    attack_shard_index=int(self._arg_value("--attack-shard-index")),
                    attack_shard_count=int(self._arg_value("--attack-shard-count")),
                    attack_local_worker_count=int(self._arg_value("--attack-local-worker-count")),
                    local_worker_index=int(self._arg_value("--local-worker-index")),
                    worker_plan_path=Path(self._arg_value("--worker-plan-path")),
                )
                self._return_code = 0
            return self._return_code

    monkeypatch.setattr(pw03_module.subprocess, "Popen", _FakePopen)


def _load_attack_event_specs(summary: Dict[str, Any], shard_index: int) -> List[Dict[str, Any]]:
    """
    Load the materialized attack event specs for one shard.

    Args:
        summary: PW00 summary payload.
        shard_index: Attack shard index.

    Returns:
        Materialized attack event specs.
    """
    family_root = Path(str(summary["family_root"]))
    attack_event_lookup = pw03_module._load_attack_event_lookup(Path(str(summary["attack_event_grid_path"])))
    attack_shard_plan = json.loads(Path(str(summary["attack_shard_plan_path"])).read_text(encoding="utf-8"))
    attack_shard_assignment = pw03_module._resolve_attack_shard_assignment(
        attack_shard_plan,
        attack_shard_index=shard_index,
        attack_shard_count=int(attack_shard_plan["attack_shard_count"]),
    )
    finalize_manifest_path = family_root / "source_finalize" / "paper_source_finalize_manifest.json"
    finalize_manifest = json.loads(finalize_manifest_path.read_text(encoding="utf-8"))
    source_pools = cast(Dict[str, Dict[str, Any]], finalize_manifest["source_pools"])
    positive_pool_manifest_path = Path(str(source_pools[pw03_module.ACTIVE_SAMPLE_ROLE]["manifest_path"]))
    positive_source_pool_manifest = json.loads(positive_pool_manifest_path.read_text(encoding="utf-8"))
    clean_negative_pool_manifest_path = Path(str(source_pools[pw03_module.CLEAN_NEGATIVE_SAMPLE_ROLE]["manifest_path"]))
    clean_negative_pool_manifest = json.loads(clean_negative_pool_manifest_path.read_text(encoding="utf-8"))
    parent_event_lookup = pw03_module._load_parent_source_event_lookup(
        positive_source_pool_manifest,
        expected_parent_sample_role=pw03_module.ACTIVE_SAMPLE_ROLE,
    )
    parent_event_lookup.update(
        pw03_module._load_parent_source_event_lookup(
            clean_negative_pool_manifest,
            expected_parent_sample_role=pw03_module.CLEAN_NEGATIVE_SAMPLE_ROLE,
        )
    )
    threshold_binding_reference = pw03_module._build_threshold_binding_reference(
        finalize_manifest_path=finalize_manifest_path,
        finalize_manifest=finalize_manifest,
    )
    return [
        pw03_module._resolve_attack_event_spec(
            attack_event=attack_event_lookup[event_id],
            parent_event_lookup=parent_event_lookup,
            threshold_binding_reference=threshold_binding_reference,
        )
        for event_id in cast(List[str], attack_shard_assignment["assigned_attack_event_ids"])
    ]


def test_pw03_consumes_finalized_positive_pool_and_writes_event_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify PW03 consumes PW02 finalized positive source pool and writes required artifacts.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw03_artifacts")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_artifacts")
    captures = _patch_pw03_detect(monkeypatch)

    pw03_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_artifacts",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=1,
        bound_config_path=bound_config_path,
    )

    assert pw03_summary["status"] == "completed"
    assert pw03_summary["worker_execution_mode"] == "single_process"
    assert Path(str(pw03_summary["gpu_session_peak_path"])).exists()
    assert len(cast(List[Dict[str, Any]], pw03_summary["worker_outcomes"])) == 1
    shard_gpu_summary = json.loads(Path(str(pw03_summary["gpu_session_peak_path"])).read_text(encoding="utf-8"))
    assert shard_gpu_summary["wrapped_command_count"] == 1
    assert len(cast(List[Dict[str, Any]], shard_gpu_summary["worker_local_peaks"])) == 1
    assert shard_gpu_summary["peak_memory_mib"] == int(shard_gpu_summary["worker_local_peaks"][0]["peak_memory_mib"])
    assert shard_gpu_summary["peak_timestamp"] == "2026-04-05T00:00:00Z"
    assert shard_gpu_summary["visible_gpus"][0]["peak_memory_used_mib"] == shard_gpu_summary["peak_memory_mib"]
    assert pw03_summary["completed_event_count"] == pw03_summary["event_count"]
    assert captures["detect_input_payloads"]

    first_event = cast(Dict[str, Any], pw03_summary["events"][0])
    assert first_event["sample_role"] == pw03_module.ATTACKED_POSITIVE_SAMPLE_ROLE
    assert Path(str(first_event["attacked_image_path"])).exists()
    assert Path(str(first_event["runtime_config_snapshot_path"])).exists()
    assert Path(str(first_event["threshold_binding_summary_path"])).exists()
    assert Path(str(first_event["detect_record_path"])).exists()
    assert first_event["parent_event_id"]
    assert first_event["attack_family"]
    assert first_event["attack_params_digest"]
    assert first_event["threshold_binding_summary"]["threshold_artifact_paths"]["content"]
    assert first_event["threshold_binding_summary"]["threshold_artifact_paths"]["attestation"]
    assert first_event["source_finalize_manifest_digest"]
    assert first_event["parent_source_image_path"] != first_event["attacked_image_path"]
    assert cast(Dict[str, Any], first_event["severity_metadata"])["severity_status"] in {"ok", "not_available"}
    assert cast(Dict[str, Any], first_event["geometry_diagnostics"])["sync_status"] == "ok"
    assert cast(Dict[str, Any], first_event["geometry_diagnostics"])["sync_success"] is True
    assert cast(Dict[str, Any], first_event["geometry_diagnostics"])["inverse_transform_success"] is True
    assert cast(Dict[str, Any], first_event["geometry_diagnostics"])["attention_anchor_available"] is True

    detect_input_payload = cast(Dict[str, Any], captures["detect_input_payloads"][0])
    assert detect_input_payload["paper_workflow_parent_event_id"] == first_event["parent_event_id"]
    assert detect_input_payload["paper_workflow_attack_family"] == first_event["attack_family"]
    assert detect_input_payload["paper_workflow_attack_params_digest"] == first_event["attack_params_digest"]
    assert detect_input_payload["watermarked_path"] == first_event["attacked_image_path"]
    assert detect_input_payload["image_path"] == first_event["attacked_image_path"]
    detect_inputs = cast(Dict[str, Any], detect_input_payload["inputs"])
    assert detect_inputs["input_image_path"] == first_event["attacked_image_path"]
    assert detect_input_payload["plan_digest"].startswith("plan_")

    detect_record_payload = json.loads(Path(str(first_event["detect_record_path"])).read_text(encoding="utf-8"))
    assert cast(Dict[str, Any], detect_record_payload["paper_workflow_severity_metadata"])["severity_status"] in {"ok", "not_available"}
    assert cast(Dict[str, Any], detect_record_payload["paper_workflow_geometry_diagnostics"])["sync_success"] is True
    assert cast(Dict[str, Any], detect_record_payload["paper_workflow_geometry_diagnostics"])["inverse_transform_success"] is True
    assert cast(Dict[str, Any], detect_record_payload["paper_workflow_geometry_diagnostics"])["attention_anchor_available"] is True


def test_pw03_attack_shards_remain_isolated_across_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify different PW03 shard sessions produce disjoint event outputs.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw03_isolation")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_isolation")
    _patch_pw03_detect(monkeypatch)

    shard_0_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_isolation",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=1,
        bound_config_path=bound_config_path,
    )
    shard_1_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_isolation",
        attack_shard_index=1,
        attack_shard_count=2,
        attack_local_worker_count=1,
        bound_config_path=bound_config_path,
    )

    assert set(cast(List[str], shard_0_summary["event_ids"]))
    assert set(cast(List[str], shard_1_summary["event_ids"]))
    assert set(cast(List[str], shard_0_summary["event_ids"])).isdisjoint(set(cast(List[str], shard_1_summary["event_ids"])))
    assert Path(str(shard_0_summary["shard_root"])) != Path(str(shard_1_summary["shard_root"]))

    shard_0_attacked_images = {str(event["attacked_image_path"]) for event in cast(List[Dict[str, Any]], shard_0_summary["events"])}
    shard_1_attacked_images = {str(event["attacked_image_path"]) for event in cast(List[Dict[str, Any]], shard_1_summary["events"])}
    assert shard_0_attacked_images.isdisjoint(shard_1_attacked_images)


def test_pw03_local_worker_assignments_do_not_conflict_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify shard-local worker assignments do not write conflicting event outputs.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw03_workers")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_workers")
    _patch_pw03_detect(monkeypatch)
    _, bound_cfg_obj = pw03_module._load_required_bound_config(bound_config_path)

    family_root = Path(str(summary["family_root"]))
    shard_root = ensure_directory(pw03_module._resolve_attack_shard_root(family_root, 0))
    attack_event_specs = _load_attack_event_specs(summary, shard_index=0)
    worker_plans = pw03_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_workers",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=2,
        shard_root=shard_root,
        bound_config_path=bound_config_path,
        assigned_attack_events=attack_event_specs,
    )

    assert len(worker_plans) == 2
    assigned_by_worker = [set(cast(List[str], worker_plan["assigned_attack_event_ids"])) for worker_plan in worker_plans]
    assert assigned_by_worker[0].isdisjoint(assigned_by_worker[1])
    assert all(Path(str(worker_plan["stdout_log_path"])).parent.exists() for worker_plan in worker_plans)
    assert all(Path(str(worker_plan["stderr_log_path"])).parent.exists() for worker_plan in worker_plans)

    worker_results = []
    for worker_plan in worker_plans:
        worker_result = pw03_module._run_attack_event_by_worker(
            family_id="family_pw03_workers",
            attack_shard_index=0,
            attack_shard_count=2,
            attack_local_worker_count=2,
            local_worker_index=int(worker_plan["local_worker_index"]),
            worker_root=Path(str(worker_plan["worker_root"])),
            shard_root=shard_root,
            bound_cfg_obj=bound_cfg_obj,
            assigned_attack_events=[
                cast(Dict[str, Any], event)
                for event in attack_event_specs
                if str(event["event_id"]) in cast(List[str], worker_plan["assigned_attack_event_ids"])
            ],
        )
        worker_results.append(worker_result)

    event_manifest_paths = [
        str(event["event_manifest_path"])
        for worker_result in worker_results
        for event in cast(List[Dict[str, Any]], worker_result["events"])
    ]
    attacked_image_paths = [
        str(event["attacked_image_path"])
        for worker_result in worker_results
        for event in cast(List[Dict[str, Any]], worker_result["events"])
    ]
    assert len(event_manifest_paths) == len(set(event_manifest_paths))
    assert len(attacked_image_paths) == len(set(attacked_image_paths))
    assert all(Path(path).exists() for path in event_manifest_paths)
    assert all(Path(path).exists() for path in attacked_image_paths)


@pytest.mark.parametrize("attack_local_worker_count", [3, 4])
def test_pw03_local_worker_assignments_support_three_and_four_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack_local_worker_count: int,
) -> None:
    """
    Verify PW03 worker planning accepts three and four local workers.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        attack_local_worker_count: Requested local worker count.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id=f"family_pw03_workers_{attack_local_worker_count}")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker=f"pw03_workers_{attack_local_worker_count}",
    )
    _patch_pw03_detect(monkeypatch)

    pw03_module._validate_attack_local_worker_count(attack_local_worker_count)

    family_root = Path(str(summary["family_root"]))
    shard_root = ensure_directory(pw03_module._resolve_attack_shard_root(family_root, 0))
    attack_event_specs = _load_attack_event_specs(summary, shard_index=0)
    worker_plans = pw03_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id=f"family_pw03_workers_{attack_local_worker_count}",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=attack_local_worker_count,
        shard_root=shard_root,
        bound_config_path=bound_config_path,
        assigned_attack_events=attack_event_specs,
    )

    expected_event_ids = {str(event["event_id"]) for event in attack_event_specs}
    assigned_event_ids = {
        event_id
        for worker_plan in worker_plans
        for event_id in cast(List[str], worker_plan["assigned_attack_event_ids"])
    }

    assert len(worker_plans) == attack_local_worker_count
    assert len({str(worker_plan["worker_root"]) for worker_plan in worker_plans}) == attack_local_worker_count
    assert assigned_event_ids == expected_event_ids
    assert sum(len(cast(List[str], worker_plan["assigned_attack_event_ids"])) for worker_plan in worker_plans) == len(expected_event_ids)
    assert all(Path(str(worker_plan["worker_plan_path"])).exists() for worker_plan in worker_plans)
    assert all(Path(str(worker_plan["stdout_log_path"])).parent.exists() for worker_plan in worker_plans)
    assert all(Path(str(worker_plan["stderr_log_path"])).parent.exists() for worker_plan in worker_plans)


@pytest.mark.parametrize("attack_local_worker_count", [3, 4])
def test_pw03_parallel_worker_launch_supports_three_and_four_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack_local_worker_count: int,
) -> None:
    """
    Verify PW03 parallel worker execution preserves isolation for three and four workers.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        attack_local_worker_count: Requested local worker count.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id=f"family_pw03_parallel_{attack_local_worker_count}")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker=f"pw03_parallel_{attack_local_worker_count}",
    )
    _patch_pw03_detect(monkeypatch)
    _patch_pw03_worker_popen(monkeypatch)

    original_prepare_local_worker_plans = pw03_module._prepare_local_worker_plans

    def _prepare_local_worker_plans_without_logs(**kwargs: Any) -> List[Dict[str, Any]]:
        worker_plans = original_prepare_local_worker_plans(**kwargs)
        for worker_plan in worker_plans:
            logs_root = Path(str(worker_plan["stdout_log_path"])).parent
            if logs_root.exists():
                shutil.rmtree(logs_root)
            assert not logs_root.exists()
        return worker_plans

    monkeypatch.setattr(pw03_module, "_prepare_local_worker_plans", _prepare_local_worker_plans_without_logs)

    pw03_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id=f"family_pw03_parallel_{attack_local_worker_count}",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=attack_local_worker_count,
        bound_config_path=bound_config_path,
    )

    worker_outcomes = cast(List[Dict[str, Any]], pw03_summary["worker_outcomes"])
    shard_gpu_summary = json.loads(Path(str(pw03_summary["gpu_session_peak_path"])).read_text(encoding="utf-8"))

    assert pw03_summary["status"] == "completed"
    assert pw03_summary["worker_execution_mode"] == "shard_local_subprocess_parallel"
    assert len(worker_outcomes) == attack_local_worker_count
    assert all(int(worker_outcome["return_code"]) == 0 for worker_outcome in worker_outcomes)
    assert all(bool(worker_outcome["result_exists"]) for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stdout_log_path"])).exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stderr_log_path"])).exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["worker_gpu_session_peak_path"])).exists() for worker_outcome in worker_outcomes)
    assert len(cast(List[Dict[str, Any]], shard_gpu_summary["worker_local_peaks"])) == attack_local_worker_count
    assert shard_gpu_summary["wrapped_command_count"] == attack_local_worker_count
    assert len(cast(List[str], pw03_summary["event_ids"])) == len(set(cast(List[str], pw03_summary["event_ids"])))


def test_pw03_accepts_decoupled_attack_shard_count_when_plan_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify PW03 accepts a decoupled attack shard count when it matches the frozen plan.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(
        tmp_path,
        family_id="family_pw03_attack_decoupled",
        attack_shard_count=3,
    )
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_attack_decoupled")
    _patch_pw03_detect(monkeypatch)

    attack_shard_plan = json.loads(Path(str(summary["attack_shard_plan_path"])).read_text(encoding="utf-8"))
    pw03_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_attack_decoupled",
        attack_shard_index=0,
        attack_shard_count=3,
        attack_local_worker_count=1,
        attack_family_allowlist=["jpeg"],
        bound_config_path=bound_config_path,
    )

    assert attack_shard_plan["attack_shard_count"] == 3
    assert pw03_summary["status"] == "completed"
    assert pw03_summary["attack_shard_count"] == 3
    assert {str(event["attack_family"]) for event in cast(List[Dict[str, Any]], pw03_summary["events"])} == {"jpeg"}


def test_pw03_rejects_attack_shard_count_mismatch_with_frozen_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify PW03 keeps the frozen attack_shard_count consistency check.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(
        tmp_path,
        family_id="family_pw03_attack_mismatch",
        attack_shard_count=3,
    )
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_attack_mismatch")
    _patch_pw03_detect(monkeypatch)

    with pytest.raises(ValueError, match="attack_shard_count mismatch with attack shard plan"):
        pw03_module.run_pw03_attack_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw03_attack_mismatch",
            attack_shard_index=0,
            attack_shard_count=2,
            attack_local_worker_count=1,
            bound_config_path=bound_config_path,
        )


def test_pw03_parallel_worker_launch_recreates_missing_logs_and_keeps_summary_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify PW03 parallel worker launch recreates missing log directories before open.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw03_parallel_logs")
    _build_positive_source_finalize_fixture(summary)
    bound_config_path, _ = _write_bound_config_snapshot(tmp_path / "drive", marker="pw03_parallel_logs")
    _patch_pw03_detect(monkeypatch)
    _patch_pw03_worker_popen(monkeypatch)

    original_prepare_local_worker_plans = pw03_module._prepare_local_worker_plans

    def _prepare_local_worker_plans_without_logs(**kwargs: Any) -> List[Dict[str, Any]]:
        worker_plans = original_prepare_local_worker_plans(**kwargs)
        for worker_plan in worker_plans:
            logs_root = Path(str(worker_plan["stdout_log_path"])).parent
            if logs_root.exists():
                shutil.rmtree(logs_root)
            assert not logs_root.exists()
        return worker_plans

    monkeypatch.setattr(pw03_module, "_prepare_local_worker_plans", _prepare_local_worker_plans_without_logs)

    pw03_summary = pw03_module.run_pw03_attack_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw03_parallel_logs",
        attack_shard_index=0,
        attack_shard_count=2,
        attack_local_worker_count=2,
        bound_config_path=bound_config_path,
    )

    assert pw03_summary["status"] == "completed"
    assert pw03_summary["worker_execution_mode"] == "shard_local_subprocess_parallel"
    worker_outcomes = cast(List[Dict[str, Any]], pw03_summary["worker_outcomes"])
    assert len(worker_outcomes) == 2
    assert all(int(worker_outcome["return_code"]) == 0 for worker_outcome in worker_outcomes)
    assert all(bool(worker_outcome["result_exists"]) for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stdout_log_path"])).exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stderr_log_path"])).exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stdout_log_path"])).parent.exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["stderr_log_path"])).parent.exists() for worker_outcome in worker_outcomes)
    assert all(Path(str(worker_outcome["worker_gpu_session_peak_path"])).exists() for worker_outcome in worker_outcomes)

    shard_gpu_summary = json.loads(Path(str(pw03_summary["gpu_session_peak_path"])).read_text(encoding="utf-8"))
    assert shard_gpu_summary["wrapped_command_count"] == 2
    assert len(cast(List[Dict[str, Any]], shard_gpu_summary["worker_local_peaks"])) == 2
    assert shard_gpu_summary["peak_memory_mib"] == max(
        int(worker_peak["peak_memory_mib"])
        for worker_peak in cast(List[Dict[str, Any]], shard_gpu_summary["worker_local_peaks"])
        if worker_peak.get("peak_memory_mib") is not None
    )
    assert shard_gpu_summary["peak_timestamp"] == "2026-04-05T00:00:00Z"
    assert shard_gpu_summary["visible_gpus"][0]["peak_memory_used_mib"] == shard_gpu_summary["peak_memory_mib"]
    assert Path(str(pw03_summary["gpu_session_peak_path"])).exists()


def test_pw03_gpu_peak_aggregation_keeps_first_timestamp_on_equal_worker_peaks() -> None:
    """
    Verify shard GPU aggregation keeps a stable timestamp when worker peaks tie.

    Args:
        None.

    Returns:
        None.
    """
    shard_gpu_summary = pw03_module._aggregate_gpu_session_peaks(
        family_id="family_gpu_peak_tie",
        attack_shard_index=0,
        attack_local_worker_count=2,
        gpu_peak_payloads=[
            {
                "worker_local_index": 0,
                "peak_memory_mib": 4096,
                "peak_timestamp": "2026-04-05T00:00:00Z",
                "device_name": "Test GPU",
                "visible_gpu_count": 1,
                "visible_gpus": [
                    {
                        "index": 0,
                        "uuid": "gpu-0",
                        "name": "Test GPU",
                        "memory_total_mib": 24576,
                        "peak_memory_used_mib": 4096,
                    }
                ],
                "wrapped_return_code": 0,
                "summary_path": "worker_00.json",
            },
            {
                "worker_local_index": 1,
                "peak_memory_mib": 4096,
                "peak_timestamp": "2026-04-05T00:00:01Z",
                "device_name": "Test GPU",
                "visible_gpu_count": 1,
                "visible_gpus": [
                    {
                        "index": 0,
                        "uuid": "gpu-0",
                        "name": "Test GPU",
                        "memory_total_mib": 24576,
                        "peak_memory_used_mib": 4096,
                    }
                ],
                "wrapped_return_code": 0,
                "summary_path": "worker_01.json",
            },
        ],
    )

    assert shard_gpu_summary["peak_memory_mib"] == 4096
    assert shard_gpu_summary["peak_timestamp"] == "2026-04-05T00:00:00Z"
    assert shard_gpu_summary["visible_gpus"][0]["peak_memory_used_mib"] == 4096


def test_pw03_gpu_peak_aggregation_ignores_null_worker_peak_when_other_worker_valid() -> None:
    """
    Verify shard GPU aggregation keeps the valid worker peak when another worker has null values.

    Args:
        None.

    Returns:
        None.
    """
    shard_gpu_summary = pw03_module._aggregate_gpu_session_peaks(
        family_id="family_gpu_peak_partial",
        attack_shard_index=1,
        attack_local_worker_count=2,
        gpu_peak_payloads=[
            {
                "worker_local_index": 0,
                "peak_memory_mib": None,
                "peak_timestamp": None,
                "device_name": None,
                "visible_gpu_count": 0,
                "visible_gpus": [],
                "wrapped_return_code": 1,
                "summary_path": "worker_00.json",
            },
            {
                "worker_local_index": 1,
                "peak_memory_mib": 8192,
                "peak_timestamp": "2026-04-05T00:00:02Z",
                "device_name": "Test GPU",
                "visible_gpu_count": 1,
                "visible_gpus": [
                    {
                        "index": 0,
                        "uuid": "gpu-0",
                        "name": "Test GPU",
                        "memory_total_mib": 24576,
                        "peak_memory_used_mib": 8192,
                    }
                ],
                "wrapped_return_code": 0,
                "summary_path": "worker_01.json",
            },
        ],
    )

    assert shard_gpu_summary["peak_memory_mib"] == 8192
    assert shard_gpu_summary["peak_timestamp"] == "2026-04-05T00:00:02Z"
    assert shard_gpu_summary["visible_gpus"][0]["peak_memory_used_mib"] == 8192


def test_pw03_gpu_peak_aggregation_falls_back_to_visible_gpu_peak_when_top_level_missing() -> None:
    """
    Verify shard GPU aggregation backfills the top-level peak from visible_gpus when needed.

    Args:
        None.

    Returns:
        None.
    """
    shard_gpu_summary = pw03_module._aggregate_gpu_session_peaks(
        family_id="family_gpu_peak_visible_gpus",
        attack_shard_index=2,
        attack_local_worker_count=1,
        gpu_peak_payloads=[
            {
                "worker_local_index": 0,
                "peak_memory_mib": None,
                "peak_timestamp": "2026-04-05T00:00:03Z",
                "device_name": "Test GPU",
                "visible_gpu_count": 1,
                "visible_gpus": [
                    {
                        "index": 0,
                        "uuid": "gpu-0",
                        "name": "Test GPU",
                        "memory_total_mib": 24576,
                        "peak_memory_used_mib": 38855,
                    }
                ],
                "wrapped_return_code": 0,
                "summary_path": "worker_00.json",
            }
        ],
    )

    assert shard_gpu_summary["peak_memory_mib"] == 38855
    assert shard_gpu_summary["peak_timestamp"] == "2026-04-05T00:00:03Z"
    assert shard_gpu_summary["visible_gpus"][0]["peak_memory_used_mib"] == shard_gpu_summary["peak_memory_mib"]


def test_pw03_cli_help_exposes_attack_shard_arguments() -> None:
    """
    Verify the PW03 CLI exposes the expected shard arguments.

    Args:
        None.

    Returns:
        None.
    """
    script_path = pw03_module.REPO_ROOT / "paper_workflow" / "scripts" / "pw03_run_attack_event_shard.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=pw03_module.REPO_ROOT,
        env=build_repo_import_subprocess_env(repo_root=pw03_module.REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined_output = f"{result.stdout}\n{result.stderr}"
    normalized_output = " ".join(combined_output.split())
    assert result.returncode == 0
    assert "usage:" in combined_output.lower()
    assert "--attack-shard-index" in normalized_output
    assert "--attack-shard-count" in normalized_output
    assert "--attack-local-worker-count" in normalized_output
    assert "--attack-family-allowlist" in normalized_output
    assert "finalized positive source pool" in normalized_output