"""
File purpose: Validate PW04 attack merge, formal materialization, and export contracts.
Module type: General module
"""

from __future__ import annotations

import builtins
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

import pytest
from PIL import Image

import paper_workflow.scripts.pw04_merge_attack_event_shards as pw04_module
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import read_jsonl
from scripts.notebook_runtime_common import compute_file_sha256, ensure_directory, normalize_path_value, write_json_atomic


@pytest.fixture(autouse=True)
def _force_pw04_png_fallback(monkeypatch: Any) -> None:
    """
    Force PW04 figure generation tests through the builtin PNG fallback path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    original_import = builtins.__import__

    def patched_import(
        name: str,
        globals_arg: Any = None,
        locals_arg: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("forced matplotlib fallback for PW04 tests")
        return original_import(name, globals_arg, locals_arg, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched_import)


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


def _load_csv_rows(path_obj: Path) -> List[Dict[str, Any]]:
    """
    Load one CSV file into ordered rows.

    Args:
        path_obj: CSV file path.

    Returns:
        Parsed CSV rows.
    """
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        return [dict(cast(Mapping[str, Any], row)) for row in csv.DictReader(handle)]


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


def _image_evidence_payload_for_index(event_index: int) -> Dict[str, Any]:
    """
    Build deterministic image-evidence rescue fields for one attack event.

    Args:
        event_index: Attack event index.

    Returns:
        Image evidence payload.
    """
    pattern_index = event_index % 4
    if pattern_index == 1:
        return {
            "status": "ok",
            "geo_rescue_eligible": True,
            "geo_rescue_applied": True,
            "geo_not_used_reason": None,
        }
    if pattern_index == 2:
        return {
            "status": "ok",
            "geo_rescue_eligible": True,
            "geo_rescue_applied": False,
            "geo_not_used_reason": "geometry_score_below_rescue_min",
        }
    if pattern_index == 0:
        return {
            "status": "ok",
            "geo_rescue_eligible": False,
            "geo_rescue_applied": False,
            "geo_not_used_reason": "content_chain_already_positive",
        }
    return {
        "status": "ok",
        "geo_rescue_eligible": False,
        "geo_rescue_applied": False,
        "geo_not_used_reason": "attestation_already_positive",
    }


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
    quality_root = ensure_directory(pw02_root / "quality")
    quality_metrics_summary_csv_path = quality_root / "quality_metrics_summary.csv"
    quality_metrics_summary_json_path = quality_root / "quality_metrics_summary.json"
    payload_root = ensure_directory(pw02_root / "payload")
    payload_clean_summary_path = payload_root / "payload_clean_summary.json"
    clean_evaluate_root = ensure_directory(pw02_root / "evaluate" / "clean")
    content_clean_evaluate_export_path = clean_evaluate_root / "content" / "evaluate_record.json"
    attestation_clean_evaluate_export_path = clean_evaluate_root / "attestation" / "evaluate_record.json"
    ensure_directory(content_clean_evaluate_export_path.parent)
    ensure_directory(attestation_clean_evaluate_export_path.parent)
    write_json_atomic(
        formal_clean_metrics_path,
        {
            "artifact_type": "paper_workflow_pw02_formal_final_decision_metrics",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "metrics": {
                "scope": "formal_final_decision",
                "n_total": 8,
                "n_positive": 4,
                "n_negative": 4,
                "final_decision_available_rate": 1.0,
                "content_chain_available_rate": 1.0,
                "final_decision_tpr": 1.0,
                "final_decision_fpr": 0.0,
                "final_decision_status_counts": {
                    "accept": 4,
                    "reject": 4,
                },
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
                "scope": "system_final",
                "n_total": 8,
                "n_positive": 4,
                "n_negative": 4,
                "final_decision_available_rate": 1.0,
                "content_chain_available_rate": 1.0,
                "image_evidence_ok_rate": 1.0,
                "event_attestation_available_rate": 1.0,
                "geo_rescue_eligible_rate": 0.5,
                "system_tpr": 1.0,
                "system_fpr": 0.0,
                "final_decision_tpr": 1.0,
                "final_decision_fpr": 0.0,
                "event_attestation_tpr": 0.75,
                "event_attestation_fpr": 0.25,
                "geo_rescue_applied_rate": 0.25,
                "final_decision_status_counts": {
                    "accept": 4,
                    "reject": 4,
                },
                "geo_not_used_reason_counts": {
                    "geometry_score_below_rescue_min": 1,
                },
            },
        },
    )
    write_json_atomic(
        content_clean_evaluate_export_path,
        {
            "artifact_type": "paper_workflow_pw02_clean_evaluate_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "content_chain_score",
            "evaluate_record": {
                "status": "completed",
                "metrics": {
                    "n_total": 8,
                    "n_pos": 4,
                    "n_neg": 4,
                    "tpr_at_fpr_primary": 1.0,
                    "fpr_empirical": 0.0,
                },
                "breakdown": {
                    "confusion": {
                        "tp": 4,
                        "fp": 0,
                        "fn": 0,
                        "tn": 4,
                    }
                },
            },
        },
    )
    write_json_atomic(
        attestation_clean_evaluate_export_path,
        {
            "artifact_type": "paper_workflow_pw02_clean_evaluate_export",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "score_name": "event_attestation_score",
            "evaluate_record": {
                "status": "completed",
                "metrics": {
                    "n_total": 8,
                    "n_pos": 4,
                    "n_neg": 4,
                    "tpr_at_fpr_primary": 0.75,
                    "fpr_empirical": 0.25,
                },
                "breakdown": {
                    "confusion": {
                        "tp": 3,
                        "fp": 1,
                        "fn": 1,
                        "tn": 3,
                    }
                },
            },
        },
    )
    quality_rows = [
        {
            "scope": "content_chain",
            "status": "ok",
            "reason": None,
            "pair_count": 8,
            "expected_pair_count": 8,
            "missing_count": 0,
            "error_count": 0,
            "mean_psnr": 35.0,
            "mean_ssim": 0.99,
            "lpips_status": "not_available",
            "lpips_reason": "requires additional model dependency or upstream implementation",
            "clip_status": "not_available",
            "clip_reason": "requires additional model dependency or upstream implementation",
            "source_analysis_path": normalize_path_value(content_clean_evaluate_export_path),
        },
        {
            "scope": "event_attestation",
            "status": "not_applicable",
            "reason": "quality metrics are only computed for content_chain_score",
            "pair_count": None,
            "expected_pair_count": None,
            "missing_count": None,
            "error_count": None,
            "mean_psnr": None,
            "mean_ssim": None,
            "lpips_status": "not_available",
            "lpips_reason": "requires additional model dependency or upstream implementation",
            "clip_status": "not_available",
            "clip_reason": "requires additional model dependency or upstream implementation",
            "source_analysis_path": normalize_path_value(attestation_clean_evaluate_export_path),
        },
        {
            "scope": "system_final",
            "status": "not_available",
            "reason": "quality payload is only defined for clean content-chain image pairs in current workflow",
            "pair_count": None,
            "expected_pair_count": None,
            "missing_count": None,
            "error_count": None,
            "mean_psnr": None,
            "mean_ssim": None,
            "lpips_status": "not_available",
            "lpips_reason": "requires additional model dependency or upstream implementation",
            "clip_status": "not_available",
            "clip_reason": "requires additional model dependency or upstream implementation",
            "source_analysis_path": None,
        },
    ]
    with quality_metrics_summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "status",
                "reason",
                "pair_count",
                "expected_pair_count",
                "missing_count",
                "error_count",
                "mean_psnr",
                "mean_ssim",
                "lpips_status",
                "lpips_reason",
                "clip_status",
                "clip_reason",
                "source_analysis_path",
            ],
        )
        writer.writeheader()
        for row in quality_rows:
            writer.writerow(row)
    write_json_atomic(
        quality_metrics_summary_json_path,
        {
            "artifact_type": "paper_workflow_pw02_quality_metrics_summary",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "rows": quality_rows,
        },
    )
    write_json_atomic(
        payload_clean_summary_path,
        {
            "artifact_type": "paper_workflow_pw02_payload_clean_summary",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "status": "not_available",
            "reason": "missing upstream decoded bits / reference bits / bit error sidecar",
            "future_upstream_sidecar_required": True,
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
            "quality_metrics_dir": normalize_path_value(quality_root),
            "quality_metrics_summary_csv_path": normalize_path_value(quality_metrics_summary_csv_path),
            "quality_metrics_summary_json_path": normalize_path_value(quality_metrics_summary_json_path),
            "payload_metrics_dir": normalize_path_value(payload_root),
            "payload_clean_summary_path": normalize_path_value(payload_clean_summary_path),
            "clean_evaluate_exports": {
                "content": normalize_path_value(content_clean_evaluate_export_path),
                "attestation": normalize_path_value(attestation_clean_evaluate_export_path),
            },
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
        "quality_metrics_summary_csv_path": quality_metrics_summary_csv_path,
        "quality_metrics_summary_json_path": quality_metrics_summary_json_path,
        "payload_clean_summary_path": payload_clean_summary_path,
        "content_clean_evaluate_export_path": content_clean_evaluate_export_path,
        "attestation_clean_evaluate_export_path": attestation_clean_evaluate_export_path,
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
            parent_source_image_path = artifacts_root / f"parent_{attack_event_index:06d}.png"
            attacked_image_path = artifacts_root / f"event_{attack_event_index:06d}.png"
            parent_source_image = Image.new("RGB", (8, 8), color=(60 + attack_event_index, 90, 120))
            attacked_image = Image.new("RGB", (8, 8), color=(62 + attack_event_index, 90, 120))
            parent_source_image.save(parent_source_image_path)
            attacked_image.save(attacked_image_path)

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
                "paper_workflow_parent_source_image_path": normalize_path_value(parent_source_image_path),
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
                    },
                    "image_evidence_result": _image_evidence_payload_for_index(attack_event_index),
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
                "parent_source_image_path": normalize_path_value(parent_source_image_path),
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

    canonical_metrics_paths = cast(Dict[str, str], pw04_summary["canonical_metrics_paths"])
    paper_tables_paths = cast(Dict[str, str], pw04_summary["paper_tables_paths"])
    paper_figures_paths = cast(Dict[str, str], pw04_summary["paper_figures_paths"])
    tail_estimation_paths = cast(Dict[str, str], pw04_summary["tail_estimation_paths"])
    attack_quality_metrics_path = Path(str(pw04_summary["attack_quality_metrics_path"]))
    robustness_curve_by_family_path = Path(str(pw04_summary["robustness_curve_by_family_path"]))
    robustness_macro_summary_path = Path(str(pw04_summary["robustness_macro_summary_path"]))
    worst_case_attack_summary_path = Path(str(pw04_summary["worst_case_attack_summary_path"]))
    geo_chain_usage_by_family_path = Path(str(pw04_summary["geo_chain_usage_by_family_path"]))
    geo_diagnostics_summary_path = Path(str(pw04_summary["geo_diagnostics_summary_path"]))
    payload_attack_summary_path = Path(str(pw04_summary["payload_attack_summary_path"]))
    quality_robustness_tradeoff_path = Path(str(pw04_summary["quality_robustness_tradeoff_path"]))
    quality_robustness_frontier_path = Path(str(pw04_summary["quality_robustness_frontier_path"]))
    system_final_auxiliary_attack_summary_path = Path(str(pw04_summary["system_final_auxiliary_attack_summary_path"]))
    system_final_auxiliary_attack_by_family_path = Path(str(pw04_summary["system_final_auxiliary_attack_by_family_path"]))
    system_final_auxiliary_attack_by_condition_path = Path(str(pw04_summary["system_final_auxiliary_attack_by_condition_path"]))

    required_paths = [
        Path(str(pw04_summary["summary_path"])),
        Path(str(pw04_summary["attack_merge_manifest_path"])),
        Path(str(pw04_summary["attack_positive_pool_manifest_path"])),
        Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])),
        Path(str(pw04_summary["formal_attack_attestation_metrics_path"])),
        Path(str(pw04_summary["derived_attack_union_metrics_path"])),
        Path(str(pw04_summary["per_attack_family_metrics_path"])),
        Path(str(pw04_summary["per_attack_condition_metrics_path"])),
        attack_quality_metrics_path,
        robustness_curve_by_family_path,
        robustness_macro_summary_path,
        worst_case_attack_summary_path,
        geo_chain_usage_by_family_path,
        geo_diagnostics_summary_path,
        payload_attack_summary_path,
        quality_robustness_tradeoff_path,
        quality_robustness_frontier_path,
        system_final_auxiliary_attack_summary_path,
        system_final_auxiliary_attack_by_family_path,
        system_final_auxiliary_attack_by_condition_path,
        Path(str(pw04_summary["attack_event_table_path"])),
        Path(str(pw04_summary["attack_family_summary_csv_path"])),
        Path(str(pw04_summary["attack_condition_summary_csv_path"])),
        Path(str(pw04_summary["clean_attack_overview_path"])),
        Path(str(pw04_summary["paper_scope_registry_path"])),
        Path(str(pw04_summary["bootstrap_confidence_intervals_path"])),
        Path(str(pw04_summary["bootstrap_confidence_intervals_csv_path"])),
        *[Path(path_value) for path_value in canonical_metrics_paths.values()],
        *[Path(path_value) for path_value in paper_tables_paths.values()],
        *[Path(path_value) for path_value in paper_figures_paths.values()],
        *[Path(path_value) for path_value in tail_estimation_paths.values()],
    ]
    for path_obj in required_paths:
        assert path_obj.exists(), path_obj
    assert Path(str(pw04_summary["robustness_dir"])).is_dir()
    assert Path(str(pw04_summary["geometry_diagnostics_dir"])).is_dir()
    assert Path(str(pw04_summary["payload_robustness_dir"])).is_dir()
    assert Path(str(pw04_summary["tradeoff_dir"])).is_dir()

    merge_manifest = _load_json_dict(Path(str(pw04_summary["attack_merge_manifest_path"])))
    pool_manifest = _load_json_dict(Path(str(pw04_summary["attack_positive_pool_manifest_path"])))
    formal_final_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_final_decision_metrics_path"])))
    formal_attestation_metrics = _load_json_dict(Path(str(pw04_summary["formal_attack_attestation_metrics_path"])))
    derived_union_metrics = _load_json_dict(Path(str(pw04_summary["derived_attack_union_metrics_path"])))
    attack_quality_metrics = _load_json_dict(attack_quality_metrics_path)
    clean_attack_overview = _load_json_dict(Path(str(pw04_summary["clean_attack_overview_path"])))
    paper_metric_registry = _load_json_dict(Path(str(pw04_summary["paper_scope_registry_path"])))
    content_chain_metrics = _load_json_dict(Path(str(canonical_metrics_paths["content_chain"])))
    event_attestation_metrics = _load_json_dict(Path(str(canonical_metrics_paths["event_attestation"])))
    system_final_metrics = _load_json_dict(Path(str(canonical_metrics_paths["system_final"])))
    bootstrap_payload = _load_json_dict(Path(str(pw04_summary["bootstrap_confidence_intervals_path"])))
    bootstrap_csv_rows = _load_csv_rows(Path(str(pw04_summary["bootstrap_confidence_intervals_csv_path"])))
    main_metrics_rows = _load_csv_rows(Path(str(paper_tables_paths["main_metrics_summary_csv_path"])))
    family_paper_rows = _load_csv_rows(Path(str(paper_tables_paths["attack_family_summary_paper_csv_path"])))
    condition_paper_rows = _load_csv_rows(Path(str(paper_tables_paths["attack_condition_summary_paper_csv_path"])))
    rescue_rows = _load_csv_rows(Path(str(paper_tables_paths["rescue_metrics_summary_csv_path"])))
    tail_fpr_1e4 = _load_json_dict(Path(str(tail_estimation_paths["estimated_tail_fpr_1e4_path"])))
    tail_fpr_1e5 = _load_json_dict(Path(str(tail_estimation_paths["estimated_tail_fpr_1e5_path"])))
    tail_fit_diagnostics = _load_json_dict(Path(str(tail_estimation_paths["tail_fit_diagnostics_path"])))
    tail_fit_stability = _load_json_dict(Path(str(tail_estimation_paths["tail_fit_stability_summary_path"])))
    robustness_curve_rows = _load_csv_rows(robustness_curve_by_family_path)
    robustness_macro_rows = _load_csv_rows(robustness_macro_summary_path)
    worst_case_rows = _load_csv_rows(worst_case_attack_summary_path)
    geometry_family_rows = _load_csv_rows(geo_chain_usage_by_family_path)
    geometry_summary_rows = _load_csv_rows(geo_diagnostics_summary_path)
    payload_attack_summary = _load_json_dict(payload_attack_summary_path)
    tradeoff_rows = _load_csv_rows(quality_robustness_tradeoff_path)
    system_final_auxiliary_attack_summary = _load_json_dict(system_final_auxiliary_attack_summary_path)
    system_final_auxiliary_family_rows = _load_csv_rows(system_final_auxiliary_attack_by_family_path)
    system_final_auxiliary_condition_rows = _load_csv_rows(system_final_auxiliary_attack_by_condition_path)
    attack_event_rows = read_jsonl(Path(str(pw04_summary["attack_event_table_path"])))

    expected_attack_event_count = int(pw03_fixture["expected_attack_event_count"])
    assert pw04_summary["status"] == "completed"
    assert pw04_summary["paper_exports_completed"] is True
    assert pw04_summary["tail_estimation_enabled"] is False
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
    assert attack_quality_metrics["overall"]["count"] == expected_attack_event_count
    assert attack_quality_metrics["overall"]["mean_psnr"] is not None
    assert attack_quality_metrics["overall"]["mean_ssim"] is not None
    assert clean_attack_overview["attack_quality_mean_psnr"] == attack_quality_metrics["overall"]["mean_psnr"]
    assert clean_attack_overview["attack_quality_mean_ssim"] == attack_quality_metrics["overall"]["mean_ssim"]

    assert paper_metric_registry["canonical_scopes"] == ["content_chain", "event_attestation", "system_final"]
    assert paper_metric_registry["legacy_scope_mapping"]["content_chain"]["attack"]["legacy_scope_name"] == "formal_attack_final_decision"
    assert paper_metric_registry["legacy_scope_mapping"]["event_attestation"]["clean"]["legacy_scope_name"] == "clean_attestation_evaluate_export"
    assert paper_metric_registry["legacy_scope_mapping"]["system_final"]["attack"]["legacy_scope_name"] == "derived_attack_union"
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["attack_quality_metrics_path"] == normalize_path_value(attack_quality_metrics_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["robustness_curve_by_family_path"] == normalize_path_value(robustness_curve_by_family_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["quality_robustness_tradeoff_path"] == normalize_path_value(quality_robustness_tradeoff_path)
    assert paper_metric_registry["artifact_paths"]["supplemental_metrics"]["system_final_auxiliary_attack_summary_path"] == normalize_path_value(system_final_auxiliary_attack_summary_path)

    assert content_chain_metrics["scope"] == "content_chain"
    assert content_chain_metrics["clean_metrics"]["clean_positive_count"] == 4
    assert content_chain_metrics["clean_metrics"]["clean_negative_count"] == 4
    assert event_attestation_metrics["scope"] == "event_attestation"
    assert event_attestation_metrics["clean_metrics"]["clean_tpr"] == pytest.approx(0.75)
    assert event_attestation_metrics["clean_metrics"]["clean_fpr"] == pytest.approx(0.25)
    assert event_attestation_metrics["clean_metrics"]["accepted_count_clean_positive"] == 3
    assert event_attestation_metrics["clean_metrics"]["accepted_count_clean_negative"] == 1
    assert system_final_metrics["scope"] == "system_final"
    assert system_final_metrics["compatibility"]["attack_legacy_scope_name"] == "derived_attack_union"

    assert [row["scope"] for row in main_metrics_rows] == ["content_chain", "event_attestation", "system_final"]
    event_attestation_row = next(row for row in main_metrics_rows if row["scope"] == "event_attestation")
    assert event_attestation_row["clean_tpr"] == "0.75"
    assert event_attestation_row["clean_fpr"] == "0.25"
    assert event_attestation_row["accepted_count_clean_positive"] == "3"
    assert event_attestation_row["accepted_count_clean_negative"] == "1"
    assert event_attestation_row["bootstrap_ci_clean_tpr_lower"] != ""
    assert event_attestation_row["metric_source_clean"].endswith("/exports/pw02/evaluate/clean/attestation/evaluate_record.json")

    assert family_paper_rows
    assert "content_chain_attack_tpr" in family_paper_rows[0]
    assert "formal_final_decision_attack_tpr" not in family_paper_rows[0]
    assert "attack_mean_psnr" in family_paper_rows[0]
    assert family_paper_rows[0]["attack_mean_psnr"] != ""
    assert condition_paper_rows
    assert "system_final_attack_tpr" in condition_paper_rows[0]
    assert "attack_family" in condition_paper_rows[0]
    assert "attack_mean_ssim" in condition_paper_rows[0]
    assert condition_paper_rows[0]["attack_mean_ssim"] != ""

    assert len(rescue_rows) == 1
    rescue_row = rescue_rows[0]
    assert rescue_row["clean_false_accept_count"] == "0"
    assert rescue_row["attack_true_accept_count"] == str(pw03_fixture["expected_derived_union_positive_count"])
    attack_true_accept_count_by_family = json.loads(rescue_row["attack_true_accept_count_by_family"])
    assert sum(int(value) for value in attack_true_accept_count_by_family.values()) == pw03_fixture["expected_derived_union_positive_count"]
    geo_not_used_reason_counts = json.loads(rescue_row["geo_not_used_reason_counts"])
    assert geo_not_used_reason_counts

    assert set(bootstrap_payload["scopes"].keys()) == {"content_chain", "event_attestation", "system_final"}
    assert bootstrap_payload["scopes"]["content_chain"]["clean_tpr"]["status"] == "ok"
    assert bootstrap_payload["scopes"]["event_attestation"]["clean_fpr"]["status"] == "ok"
    assert bootstrap_payload["scopes"]["system_final"]["attack_tpr"]["lower_bound"] is not None
    assert len(bootstrap_csv_rows) == 9

    assert tail_fpr_1e4["tail_estimation_enabled"] is False
    assert tail_fpr_1e4["scope_estimates"]["content_chain"]["status"] == "disabled"
    assert tail_fpr_1e5["scope_estimates"]["event_attestation"]["status"] == "disabled"
    assert tail_fit_diagnostics["scope_diagnostics"]["system_final"]["status"] == "not_applicable"
    assert tail_fit_stability["scopes"]["system_final"]["reason"] == "system_final_is_decision_union_without_scalar_score"

    attack_family_count = len({row["attack_family"] for row in cast(List[Dict[str, Any]], pw03_fixture["attack_event_grid"])})
    assert len(robustness_curve_rows) == attack_family_count * 3
    assert {row["scope"] for row in robustness_curve_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["severity_level_status"] == "not_available" for row in robustness_curve_rows)
    assert len(robustness_macro_rows) == 3
    assert {row["scope"] for row in robustness_macro_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["severity_level_status"] == "not_available" for row in robustness_macro_rows)
    assert len(worst_case_rows) == 3
    assert {row["scope"] for row in worst_case_rows} == {"content_chain", "event_attestation", "system_final"}

    assert len(geometry_family_rows) == attack_family_count
    assert all(row["sync_success_status"] == "not_available" for row in geometry_family_rows)
    assert all(row["inverse_transform_success_status"] == "not_available" for row in geometry_family_rows)
    assert all(row["attention_anchor_available_status"] == "not_available" for row in geometry_family_rows)
    assert len(geometry_summary_rows) == 1
    assert geometry_summary_rows[0]["event_count"] == str(expected_attack_event_count)
    assert geometry_summary_rows[0]["future_upstream_sidecar_required"] == "True"

    assert payload_attack_summary["status"] == "not_available"
    assert "decoded bits" in str(payload_attack_summary["reason"])
    assert payload_attack_summary["future_upstream_sidecar_required"] is True

    assert system_final_auxiliary_attack_summary["scope"] == "system_final_auxiliary"
    assert system_final_auxiliary_attack_summary["canonical"] is False
    assert system_final_auxiliary_attack_summary["analysis_only"] is True
    assert system_final_auxiliary_attack_summary["overall"]["consistency_status"] == "exact_match"
    assert system_final_auxiliary_attack_summary["overall"]["consistency_mismatch_count"] == 0
    assert system_final_auxiliary_attack_summary["overall"]["system_final_auxiliary_attack_tpr"] == pytest.approx(
        pw03_fixture["expected_derived_union_positive_count"] / expected_attack_event_count
    )
    assert system_final_auxiliary_family_rows
    assert system_final_auxiliary_condition_rows
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])["pw04_system_final_auxiliary_attack_summary"] == normalize_path_value(system_final_auxiliary_attack_summary_path)
    assert cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])["pw04_system_final_auxiliary_attack_summary"] == {
        "canonical": False,
        "analysis_only": True,
    }
    pw04_analysis_only_annotations = cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])
    if "pw02_system_final_auxiliary_operating_semantics" in pw04_analysis_only_annotations:
        assert pw04_analysis_only_annotations["pw02_system_final_auxiliary_operating_semantics"] == {
            "canonical": False,
            "analysis_only": True,
        }

    assert len(tradeoff_rows) == 3
    assert {row["scope"] for row in tradeoff_rows} == {"content_chain", "event_attestation", "system_final"}
    assert all(row["clean_quality_scope"] == "content_chain" for row in tradeoff_rows)
    assert all(row["clean_quality_status"] == "ok" for row in tradeoff_rows)
    assert all(row["lpips_status"] == "not_available" for row in tradeoff_rows)
    assert all(row["clip_status"] == "not_available" for row in tradeoff_rows)
    assert all(Path(str(row["quality_metrics_summary_csv_path"])).exists() for row in tradeoff_rows)
    assert all(Path(str(row["quality_metrics_summary_json_path"])).exists() for row in tradeoff_rows)
    assert all(Path(str(row["robustness_macro_summary_path"])).exists() for row in tradeoff_rows)
    assert quality_robustness_frontier_path.stat().st_size > 0

    attack_event_lookup = cast(Dict[str, Dict[str, Any]], pw03_fixture["attack_event_lookup"])
    for row in attack_event_rows:
        expected_attack_event = attack_event_lookup[row["attack_event_id"]]
        assert row["attack_family"] == expected_attack_event["attack_family"]
        assert row["parent_event_id"] == expected_attack_event["parent_event_id"]
        assert "geo_rescue_eligible" in row
        assert "geo_rescue_applied" in row
        assert "geo_not_used_reason" in row
        assert row["attack_quality_status"] == "ok"
        assert row["attack_quality_psnr"] is not None
        assert row["attack_quality_ssim"] is not None

    assert any(row["geo_rescue_applied"] is True for row in attack_event_rows)
    assert any(isinstance(row["geo_not_used_reason"], str) and row["geo_not_used_reason"] for row in attack_event_rows)

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
