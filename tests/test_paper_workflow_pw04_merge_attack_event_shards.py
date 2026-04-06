"""
File purpose: Validate PW04 attack merge, formal materialization, and export contracts.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import pytest

import paper_workflow.scripts.pw04_merge_attack_event_shards as pw04_module
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import read_jsonl
from scripts.notebook_runtime_common import compute_file_sha256, ensure_directory, normalize_path_value, write_json_atomic


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a minimal PW00 family fixture for PW04 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / f"{family_id}_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[7],
        source_shard_count=2,
        attack_shard_count=2,
    )


def _load_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    Load one JSON object file.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed JSON object.
    """
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON root must be object: {path_obj}")
    return cast(Dict[str, Any], payload)


def _score_pair_for_index(event_index: int) -> Tuple[float, float]:
    """
    Build deterministic content and attestation scores for one attack event.

    Args:
        event_index: Attack event index.

    Returns:
        Tuple of (content_score, attestation_score).
    """
    pattern_index = event_index % 4
    if pattern_index == 0:
        return 0.91, 0.12
    if pattern_index == 1:
        return 0.21, 0.74
    if pattern_index == 2:
        return 0.19, 0.18
    return 0.83, 0.79


def _build_pw02_fixture(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Materialize the minimal PW02 outputs consumed by PW04.

    Args:
        summary: PW00 summary payload.

    Returns:
        PW02 fixture metadata.
    """
    family_root = Path(str(summary["family_root"]))
    family_id = str(summary["family_id"])
    pw02_root = ensure_directory(family_root / "exports" / "pw02")
    thresholds_root = ensure_directory(pw02_root / "thresholds")
    content_thresholds_artifact_path = thresholds_root / "content" / "thresholds_artifact.json"
    attestation_thresholds_artifact_path = thresholds_root / "attestation" / "thresholds_artifact.json"
    ensure_directory(content_thresholds_artifact_path.parent)
    ensure_directory(attestation_thresholds_artifact_path.parent)

    content_thresholds_artifact = {
        "threshold_id": "content_np_0p01",
        "score_name": "content_chain_score",
        "target_fpr": 0.01,
        "threshold_value": 0.5,
        "threshold_key_used": "0p01",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": 0.5,
    }
    attestation_thresholds_artifact = {
        "threshold_id": "attestation_np_0p01",
        "score_name": "event_attestation_score",
        "target_fpr": 0.01,
        "threshold_value": 0.6,
        "threshold_key_used": "0p01",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": 0.6,
    }
    write_json_atomic(content_thresholds_artifact_path, content_thresholds_artifact)
    write_json_atomic(attestation_thresholds_artifact_path, attestation_thresholds_artifact)

    content_threshold_export_path = thresholds_root / "content" / "thresholds.json"
    attestation_threshold_export_path = thresholds_root / "attestation" / "thresholds.json"
    write_json_atomic(
        content_threshold_export_path,
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "content_chain_score",
            "source_thresholds_artifact_path": normalize_path_value(content_thresholds_artifact_path),
            "thresholds_artifact": content_thresholds_artifact,
            "calibration_record_status": "ok",
        },
    )
    write_json_atomic(
        attestation_threshold_export_path,
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "event_attestation_score",
            "source_thresholds_artifact_path": normalize_path_value(attestation_thresholds_artifact_path),
            "thresholds_artifact": attestation_thresholds_artifact,
            "calibration_record_status": "ok",
        },
    )

    formal_clean_metrics_path = pw02_root / "formal_final_decision_metrics.json"
    derived_clean_metrics_path = pw02_root / "derived_system_union_metrics.json"
    write_json_atomic(
        formal_clean_metrics_path,
        {
            "artifact_type": "paper_workflow_pw02_formal_final_decision_metrics",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "metrics": {
                "final_decision_tpr": 1.0,
                "final_decision_fpr": 0.0,
            },
        },
    )
    write_json_atomic(
        derived_clean_metrics_path,
        {
            "artifact_type": "paper_workflow_pw02_derived_system_union_metrics",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "metrics": {
                "system_tpr": 1.0,
                "system_fpr": 0.0,
            },
        },
    )

    finalize_manifest_path = pw02_root / "paper_source_finalize_manifest.json"
    write_json_atomic(
        finalize_manifest_path,
        {
            "artifact_type": "paper_workflow_pw02_finalize_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "threshold_exports": {
                "content": {"path": normalize_path_value(content_threshold_export_path)},
                "attestation": {"path": normalize_path_value(attestation_threshold_export_path)},
            },
            "source_pools": {},
        },
    )
    finalize_manifest_digest = compute_file_sha256(finalize_manifest_path)

    pw02_summary_path = family_root / "runtime_state" / "pw02_summary.json"
    write_json_atomic(
        pw02_summary_path,
        {
            "status": "completed",
            "family_id": family_id,
            "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
            "formal_final_decision_metrics_artifact_path": normalize_path_value(formal_clean_metrics_path),
            "derived_system_union_metrics_artifact_path": normalize_path_value(derived_clean_metrics_path),
        },
    )

    return {
        "family_root": family_root,
        "family_id": family_id,
        "pw02_summary_path": pw02_summary_path,
        "finalize_manifest_path": finalize_manifest_path,
        "finalize_manifest_digest": finalize_manifest_digest,
        "content_threshold_export_path": content_threshold_export_path,
        "attestation_threshold_export_path": attestation_threshold_export_path,
        "formal_clean_metrics_path": formal_clean_metrics_path,
        "derived_clean_metrics_path": derived_clean_metrics_path,
    }


def _build_pw03_fixture(summary: Dict[str, Any], pw02_fixture: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Materialize completed PW03 shard manifests, event manifests, and staged detect records.

    Args:
        summary: PW00 summary payload.
        pw02_fixture: PW02 fixture metadata.

    Returns:
        PW03 fixture metadata.
    """
    family_root = Path(str(summary["family_root"]))
    attack_event_grid = read_jsonl(Path(str(summary["attack_event_grid_path"])))
    attack_shard_plan = _load_json_dict(Path(str(summary["attack_shard_plan_path"])))
    attack_event_lookup = {
        str(row.get("attack_event_id", row.get("event_id"))): row
        for row in attack_event_grid
    }
    threshold_artifact_paths = {
        "content": normalize_path_value(Path(str(pw02_fixture["content_threshold_export_path"]))),
        "attestation": normalize_path_value(Path(str(pw02_fixture["attestation_threshold_export_path"]))),
    }

    expected_formal_final_positive_count = 0
    expected_formal_attestation_positive_count = 0
    expected_derived_union_positive_count = 0
    detect_record_paths: List[Path] = []
    event_manifest_paths: List[Path] = []

    for shard_row in cast(List[Dict[str, Any]], attack_shard_plan["shards"]):
        attack_shard_index = int(shard_row["attack_shard_index"])
        shard_root = ensure_directory(family_root / "attack_shards" / f"shard_{attack_shard_index:04d}")
        ensure_directory(shard_root / "records")
        events_payload: List[Dict[str, Any]] = []
        for attack_event_id in cast(List[str], shard_row["assigned_attack_event_ids"]):
            attack_event = cast(Dict[str, Any], attack_event_lookup[attack_event_id])
            attack_event_index = int(attack_event["attack_event_index"])
            event_root = ensure_directory(shard_root / "events" / f"event_{attack_event_index:06d}")
            artifacts_root = ensure_directory(event_root / "artifacts")
            attacked_image_path = artifacts_root / f"event_{attack_event_index:06d}.png"
            attacked_image_path.write_bytes(b"pw04_attack_fixture")

            content_score, attestation_score = _score_pair_for_index(attack_event_index)
            if content_score >= 0.5:
                expected_formal_final_positive_count += 1
            if attestation_score >= 0.6:
                expected_formal_attestation_positive_count += 1
            if content_score >= 0.5 or attestation_score >= 0.6:
                expected_derived_union_positive_count += 1

            detect_record_path = shard_root / "records" / f"event_{attack_event_index:06d}_detect_record.json"
            detect_payload = {
                "sample_role": pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE,
                "parent_event_id": None,
                "paper_workflow_parent_event_id": attack_event["parent_event_id"],
                "paper_workflow_attack_event_id": attack_event_id,
                "paper_workflow_attack_family": attack_event["attack_family"],
                "paper_workflow_attack_config_name": attack_event["attack_config_name"],
                "paper_workflow_attack_condition_key": attack_event["attack_condition_key"],
                "paper_workflow_attack_params_digest": attack_event["attack_params_digest"],
                "content_evidence_payload": {
                    "status": "ok",
                    "content_chain_score": content_score,
                },
                "attestation": {
                    "final_event_attested_decision": {
                        "status": "ok",
                        "is_event_attested": attestation_score >= 0.6,
                        "event_attestation_score_name": "event_attestation_score",
                        "event_attestation_score": attestation_score,
                    }
                },
                "final_decision": {
                    "decision_status": "accept" if content_score >= 0.5 else "reject",
                    "is_watermarked": content_score >= 0.5,
                },
                "fusion_result": {
                    "decision_status": "accept" if content_score >= 0.5 else "reject",
                },
                "geometry_evidence_payload": {
                    "status": "ok",
                },
                "lf_detect_variant": "lf_v1",
            }
            write_json_atomic(detect_record_path, detect_payload)
            detect_record_paths.append(detect_record_path)

            event_manifest_path = event_root / "event_manifest.json"
            event_manifest_payload = {
                "artifact_type": "paper_workflow_attack_event",
                "schema_version": "pw_stage_03_v1",
                "stage_name": "PW03_Attack_Event_Shards",
                "status": "completed",
                "event_id": attack_event_id,
                "attack_event_id": attack_event_id,
                "attack_event_index": attack_event_index,
                "sample_role": pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE,
                "parent_event_id": attack_event["parent_event_id"],
                "attack_family": attack_event["attack_family"],
                "attack_config_name": attack_event["attack_config_name"],
                "attack_condition_key": attack_event["attack_condition_key"],
                "attack_params_digest": attack_event["attack_params_digest"],
                "source_finalize_manifest_digest": str(pw02_fixture["finalize_manifest_digest"]),
                "threshold_artifact_paths": threshold_artifact_paths,
                "attacked_image_path": normalize_path_value(attacked_image_path),
                "detect_record_path": normalize_path_value(detect_record_path),
            }
            write_json_atomic(event_manifest_path, event_manifest_payload)
            event_manifest_paths.append(event_manifest_path)
            events_payload.append(event_manifest_payload)

        shard_manifest_path = shard_root / "shard_manifest.json"
        write_json_atomic(
            shard_manifest_path,
            {
                "artifact_type": "paper_workflow_attack_shard_manifest",
                "schema_version": "pw_stage_03_v1",
                "family_id": summary["family_id"],
                "sample_role": pw04_module.ATTACKED_POSITIVE_SAMPLE_ROLE,
                "attack_shard_index": attack_shard_index,
                "status": "completed",
                "event_count": len(events_payload),
                "completed_event_count": len(events_payload),
                "failed_event_count": 0,
                "event_ids": [str(event_payload["event_id"]) for event_payload in events_payload],
                "assigned_attack_event_ids": list(cast(List[str], shard_row["assigned_attack_event_ids"])),
                "events": events_payload,
            },
        )

    expected_attack_event_count = int(attack_shard_plan["attack_event_count"])
    return {
        "family_root": family_root,
        "attack_event_grid": attack_event_grid,
        "attack_shard_plan": attack_shard_plan,
        "attack_event_lookup": attack_event_lookup,
        "event_manifest_paths": event_manifest_paths,
        "detect_record_paths": detect_record_paths,
        "expected_attack_event_count": expected_attack_event_count,
        "expected_formal_final_positive_count": expected_formal_final_positive_count,
        "expected_formal_attestation_positive_count": expected_formal_attestation_positive_count,
        "expected_derived_union_positive_count": expected_derived_union_positive_count,
    }


def _build_pw04_fixture(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build the full PW00 + PW02 + PW03 fixture consumed by PW04.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        Full PW04 fixture metadata.
    """
    summary = _build_pw00_family(tmp_path, family_id)
    pw02_fixture = _build_pw02_fixture(summary)
    pw03_fixture = _build_pw03_fixture(summary, pw02_fixture)
    return {
        "summary": summary,
        "pw02": pw02_fixture,
        "pw03": pw03_fixture,
    }


def test_pw04_merge_attack_event_shards_success_path(tmp_path: Path) -> None:
    """
    Verify PW04 merges all completed PW03 shards and exports the required artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_success")
    summary = cast(Dict[str, Any], fixture["summary"])
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])

    pw04_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_success",
    )

    required_paths = [
        Path(str(pw04_summary["summary_path"])),
        Path(str(pw04_summary["attack_merge_manifest_path"])),
        Path(str(pw04_summary["attack_positive_pool_manifest_path"])),
        Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])),
        Path(str(pw04_summary["formal_attack_attestation_metrics_path"])),
        Path(str(pw04_summary["derived_attack_union_metrics_path"])),
        Path(str(pw04_summary["per_attack_family_metrics_path"])),
        Path(str(pw04_summary["per_attack_condition_metrics_path"])),
        Path(str(pw04_summary["attack_event_table_path"])),
        Path(str(pw04_summary["attack_family_summary_csv_path"])),
        Path(str(pw04_summary["attack_condition_summary_csv_path"])),
        Path(str(pw04_summary["clean_attack_overview_path"])),
    ]
    for path_obj in required_paths:
        assert path_obj.exists(), path_obj

    merge_manifest = _load_json_dict(Path(str(pw04_summary["attack_merge_manifest_path"])))
    pool_manifest = _load_json_dict(Path(str(pw04_summary["attack_positive_pool_manifest_path"])))
    formal_final_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])))
    formal_attestation_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_attestation_metrics_path"])))
    derived_union_metrics = _load_json_dict(Path(str(pw04_summary["derived_attack_union_metrics_path"])))
    attack_event_rows = read_jsonl(Path(str(pw04_summary["attack_event_table_path"])))

    expected_attack_event_count = int(pw03_fixture["expected_attack_event_count"])
    assert pw04_summary["status"] == "completed"
    assert pw04_summary["completed_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["expected_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["completed_attack_event_count"] == expected_attack_event_count
    assert merge_manifest["attack_family_count"] == len({row["attack_family"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert merge_manifest["parent_event_count"] == len({row["parent_event_id"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert pool_manifest["event_count"] == expected_attack_event_count

    assert formal_final_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_formal_final_positive_count"]
    assert formal_final_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_formal_final_positive_count"] / expected_attack_event_count
    )
    assert formal_attestation_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_formal_attestation_positive_count"]
    assert formal_attestation_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_formal_attestation_positive_count"] / expected_attack_event_count
    )
    assert derived_union_metrics["metrics"]["accepted_count"] == pw03_fixture["expected_derived_union_positive_count"]
    assert derived_union_metrics["metrics"]["attack_tpr"] == pytest.approx(
        pw03_fixture["expected_derived_union_positive_count"] / expected_attack_event_count
    )

    attack_event_lookup = cast(Dict[str, Dict[str, Any]], pw03_fixture["attack_event_lookup"])
    for row in attack_event_rows:
        expected_attack_event = attack_event_lookup[row["attack_event_id"]]
        assert row["attack_family"] == expected_attack_event["attack_family"]
        assert row["parent_event_id"] == expected_attack_event["parent_event_id"]

    first_pool_event = cast(List[Dict[str, Any]], pool_manifest["events"])[0]
    assert first_pool_event["formal_record_path"]
    assert Path(str(first_pool_event["formal_record_path"])).exists()
    assert summary["family_id"] == pw04_summary["family_id"]


def test_pw04_fails_fast_when_one_planned_shard_manifest_is_missing(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one planned PW03 shard manifest is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_missing_shard")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    attack_shard_plan = cast(Dict[str, Any], pw03_fixture["attack_shard_plan"])
    missing_shard_index = int(cast(List[Dict[str, Any]], attack_shard_plan["shards"])[0]["attack_shard_index"])
    missing_path = Path(str(cast(Dict[str, Any], fixture["summary"])["family_root"])) / "attack_shards" / f"shard_{missing_shard_index:04d}" / "shard_manifest.json"
    missing_path.unlink()

    with pytest.raises(FileNotFoundError):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_missing_shard",
        )


def test_pw04_fails_fast_when_planned_shard_is_not_completed(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one planned PW03 shard manifest is not completed.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_non_completed_shard")
    shard_manifest_path = Path(str(cast(Dict[str, Any], fixture["summary"])["family_root"])) / "attack_shards" / "shard_0000" / "shard_manifest.json"
    shard_manifest = _load_json_dict(shard_manifest_path)
    shard_manifest["status"] = "failed"
    write_json_atomic(shard_manifest_path, shard_manifest)

    with pytest.raises(RuntimeError, match="completed"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_non_completed_shard",
        )


def test_pw04_fails_fast_on_duplicate_attack_event_id(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when two event manifests expose the same attack_event_id.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_duplicate_attack_event_id")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    event_manifest_paths = cast(List[Path], pw03_fixture["event_manifest_paths"])
    assert len(event_manifest_paths) >= 2

    first_event_manifest = _load_json_dict(event_manifest_paths[0])
    second_event_manifest = _load_json_dict(event_manifest_paths[1])
    second_event_manifest["attack_event_id"] = first_event_manifest["attack_event_id"]
    write_json_atomic(event_manifest_paths[1], second_event_manifest)

    with pytest.raises(ValueError, match="duplicate attack_event_id"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_duplicate_attack_event_id",
        )


def test_pw04_fails_fast_on_threshold_binding_inconsistency(tmp_path: Path) -> None:
    """
    Verify PW04 fails fast when one event carries a mismatched finalize digest binding.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_threshold_binding")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    event_manifest_path = cast(List[Path], pw03_fixture["event_manifest_paths"])[0]
    event_manifest = _load_json_dict(event_manifest_path)
    event_manifest["source_finalize_manifest_digest"] = "broken_finalize_digest"
    write_json_atomic(event_manifest_path, event_manifest)

    with pytest.raises(ValueError, match="source_finalize_manifest_digest mismatch"):
        pw04_module.run_pw04_merge_attack_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw04_threshold_binding",
        )


def test_pw04_uses_event_manifest_parent_event_id_when_detect_parent_is_absent(tmp_path: Path) -> None:
    """
    Verify PW04 does not depend on detect-record top-level parent_event_id.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw04_fixture(tmp_path, "family_pw04_parent_authority")
    pw03_fixture = cast(Dict[str, Any], fixture["pw03"])
    detect_record_path = cast(List[Path], pw03_fixture["detect_record_paths"])[0]
    detect_payload = _load_json_dict(detect_record_path)
    detect_payload["parent_event_id"] = None
    detect_payload["paper_workflow_parent_event_id"] = None
    write_json_atomic(detect_record_path, detect_payload)

    pw04_summary = pw04_module.run_pw04_merge_attack_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw04_parent_authority",
    )
    pool_manifest = _load_json_dict(Path(str(pw04_summary["attack_positive_pool_manifest_path"])))
    first_event = cast(List[Dict[str, Any]], pool_manifest["events"])[0]
    expected_attack_event = cast(Dict[str, Any], pw03_fixture["attack_event_lookup"])[first_event["attack_event_id"]]
    assert first_event["parent_event_id"] == expected_attack_event["parent_event_id"]
