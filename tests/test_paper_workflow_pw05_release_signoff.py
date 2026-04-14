"""
File purpose: Validate PW05 release/signoff packaging over finalized paper_workflow artifacts.
Module type: General module
"""

from __future__ import annotations

import builtins
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast

import pytest

import paper_workflow.scripts.pw05_release_signoff as pw05_module
from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, write_json_atomic


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


def _write_text(path_obj: Path, content: str) -> Path:
    """
    Write a UTF-8 text file.

    Args:
        path_obj: Destination path.
        content: File content.

    Returns:
        Written path.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding="utf-8")
    return path_obj


def _write_json(path_obj: Path, payload: Mapping[str, Any]) -> Path:
    """
    Write one JSON mapping file.

    Args:
        path_obj: Destination path.
        payload: JSON mapping payload.

    Returns:
        Written path.
    """
    write_json_atomic(path_obj, dict(payload))
    return path_obj


def _build_pw05_family_fixture(tmp_path: Path) -> Dict[str, Any]:
    """
    Build the minimal finalized family fixture consumed by PW05.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Fixture metadata mapping.
    """
    drive_root = tmp_path / "drive_root"
    family_id = "family_pw05_demo"
    family_root = drive_root / "paper_workflow" / "families" / family_id
    ensure_directory(family_root / "manifests")
    ensure_directory(family_root / "snapshots")
    ensure_directory(family_root / "runtime_state")
    ensure_directory(family_root / "exports" / "pw02" / "thresholds" / "content")
    ensure_directory(family_root / "exports" / "pw02" / "thresholds" / "attestation")
    ensure_directory(family_root / "exports" / "pw02" / "operating_metrics")
    ensure_directory(family_root / "exports" / "pw02" / "quality")
    ensure_directory(family_root / "exports" / "pw02" / "payload")
    ensure_directory(family_root / "exports" / "pw04" / "manifests")
    ensure_directory(family_root / "exports" / "pw04" / "metrics")
    ensure_directory(family_root / "exports" / "pw04" / "tables")
    ensure_directory(family_root / "exports" / "pw04" / "figures")
    ensure_directory(family_root / "exports" / "pw04" / "tail")
    ensure_directory(family_root / "exports" / "pw04" / "robustness")
    ensure_directory(family_root / "source_finalize")

    positive_source_sidecar_root = ensure_directory(
        family_root / "source_shards" / "positive" / "shard_0000" / "events" / "event_000001" / "artifacts"
    )
    attacked_positive_sidecar_root = ensure_directory(
        family_root / "attack_shards" / "shard_0000" / "events" / "event_000001" / "artifacts"
    )
    pw02_payload_reference_sidecar_path = _write_json(
        positive_source_sidecar_root / "payload_reference_sidecar.json",
        {
            "artifact_type": "paper_workflow_payload_reference_sidecar",
            "schema_version": "pw_payload_sidecar_v1",
            "family_id": family_id,
            "event_id": "source_event_000001",
        },
    )
    pw02_payload_decode_sidecar_path = _write_json(
        positive_source_sidecar_root / "payload_decode_sidecar.json",
        {
            "artifact_type": "paper_workflow_payload_decode_sidecar",
            "schema_version": "pw_payload_sidecar_v1",
            "family_id": family_id,
            "event_id": "source_event_000001",
        },
    )
    pw04_payload_decode_sidecar_path = _write_json(
        attacked_positive_sidecar_root / "payload_decode_sidecar.json",
        {
            "artifact_type": "paper_workflow_payload_decode_sidecar",
            "schema_version": "pw_payload_sidecar_v1",
            "family_id": family_id,
            "event_id": "attack_event_000001",
        },
    )
    positive_source_pool_manifest_path = _write_json(
        family_root / "source_finalize" / "positive_source_pool_manifest.json",
        {
            "artifact_type": "paper_workflow_pw02_source_pool_manifest",
            "schema_version": "pw_stage_02_v1",
            "family_id": family_id,
            "source_role": "positive_source",
            "event_count": 1,
            "events": [
                {
                    "event_id": "source_event_000001",
                    "event_index": 1,
                    "payload_reference_sidecar_path": normalize_path_value(pw02_payload_reference_sidecar_path),
                    "payload_decode_sidecar_path": normalize_path_value(pw02_payload_decode_sidecar_path),
                }
            ],
        },
    )

    family_manifest_path = _write_json(
        family_root / "manifests" / "paper_eval_family_manifest.json",
        {
            "artifact_type": "paper_eval_family_manifest",
            "family_id": family_id,
            "stage_boundary": {
                "implemented": ["PW00", "PW01", "PW02", "PW03", "PW04", "PW05"],
                "excluded": [],
            },
        },
    )
    config_snapshot_path = _write_text(
        family_root / "snapshots" / "config_snapshot.yaml",
        "seed: 7\nfamily_id: family_pw05_demo\n",
    )
    pw02_finalize_manifest_path = _write_json(
        family_root / "exports" / "pw02" / "paper_source_finalize_manifest.json",
        {
            "artifact_type": "paper_workflow_pw02_finalize_manifest",
            "family_id": family_id,
            "status": "completed",
            "source_pools": {
                "positive_source": {
                    "manifest_path": normalize_path_value(positive_source_pool_manifest_path),
                    "event_count": 1,
                }
            },
        },
    )
    pw02_content_threshold_export_path = _write_json(
        family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json",
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "family_id": family_id,
            "score_name": "content_chain_score",
        },
    )
    pw02_attestation_threshold_export_path = _write_json(
        family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json",
        {
            "artifact_type": "paper_workflow_pw02_threshold_export",
            "family_id": family_id,
            "score_name": "event_attestation_score",
        },
    )
    pw02_system_final_auxiliary_operating_semantics_path = _write_json(
        family_root / "exports" / "pw02" / "operating_metrics" / "system_final_auxiliary_operating_semantics.json",
        {
            "artifact_type": "paper_workflow_pw02_system_final_auxiliary_operating_semantics",
            "family_id": family_id,
            "scope": "system_final_auxiliary",
            "canonical": False,
            "analysis_only": True,
        },
    )
    pw02_system_final_auxiliary_roc_curve_path = _write_json(
        family_root / "exports" / "pw02" / "operating_metrics" / "roc_curve_system_final_auxiliary.json",
        {
            "artifact_type": "paper_workflow_pw02_operating_curve",
            "family_id": family_id,
            "scope": "system_final_auxiliary",
            "canonical": False,
            "analysis_only": True,
        },
    )
    pw02_clean_quality_pair_manifest_path = _write_json(
        family_root / "exports" / "pw02" / "quality" / "clean_quality_pair_manifest.json",
        {
            "artifact_type": "paper_workflow_pw02_clean_quality_pair_manifest",
            "family_id": family_id,
            "scope": "content_chain",
            "pair_id_key": "event_id",
            "reference_path_key": "reference_image_path",
            "candidate_path_key": "candidate_image_path",
            "text_key": "prompt_text",
            "pair_rows": [
                {
                    "event_id": "source_event_000001",
                    "reference_image_path": normalize_path_value(family_root / "exports" / "pw02" / "quality" / "plain_preview_000001.png"),
                    "candidate_image_path": normalize_path_value(family_root / "exports" / "pw02" / "quality" / "watermarked_output_000001.png"),
                    "prompt_text": "prompt one",
                    "sample_role": "positive_source",
                }
            ],
        },
    )
    pw02_payload_clean_summary_path = _write_json(
        family_root / "exports" / "pw02" / "payload" / "payload_clean_summary.json",
        {
            "artifact_type": "paper_workflow_pw02_payload_clean_summary",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "future_upstream_sidecar_required": False,
            "readiness": {
                "status": "ready",
                "reason": None,
                "required_for_formal_release": True,
                "blocking": False,
            },
            "overall": {
                "event_count": 1,
                "available_payload_event_count": 1,
                "missing_payload_event_count": 0,
            },
            "rows": [],
        },
    )

    pw04_attack_merge_manifest_path = _write_json(
        family_root / "exports" / "pw04" / "manifests" / "attack_merge_manifest.json",
        {
            "artifact_type": "paper_workflow_pw04_attack_merge_manifest",
            "family_id": family_id,
            "status": "completed",
        },
    )
    pw04_attack_positive_pool_manifest_path = _write_json(
        family_root / "exports" / "pw04" / "attack_positive_pool_manifest.json",
        {
            "artifact_type": "paper_workflow_pw04_attack_positive_pool_manifest",
            "family_id": family_id,
            "event_count": 2,
            "events": [
                {
                    "attack_event_id": "attack_event_000001",
                    "attack_event_index": 1,
                    "payload_decode_sidecar_path": normalize_path_value(pw04_payload_decode_sidecar_path),
                }
            ],
        },
    )
    pw04_attack_negative_pool_manifest_path = _write_json(
        family_root / "exports" / "pw04" / "attack_negative_pool_manifest.json",
        {
            "artifact_type": "paper_workflow_pw04_attack_negative_pool_manifest",
            "family_id": family_id,
            "event_count": 2,
        },
    )
    pw04_formal_attack_final_decision_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "formal_attack_final_decision_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_formal_attack_final_decision_metrics",
            "family_id": family_id,
            "metrics": {"attack_tpr": 0.5},
        },
    )
    pw04_formal_attack_attestation_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "formal_attack_attestation_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_formal_attack_attestation_metrics",
            "family_id": family_id,
            "metrics": {"attack_tpr": 0.4},
        },
    )
    pw04_derived_attack_union_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "derived_attack_union_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_derived_attack_union_metrics",
            "family_id": family_id,
            "metrics": {"attack_tpr": 0.6},
        },
    )
    pw04_formal_attack_negative_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "formal_attack_negative_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_formal_attack_negative_metrics",
            "family_id": family_id,
            "metrics": {"derived_attack_union_attack_fpr": 0.2},
        },
    )
    pw04_clean_quality_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "clean_quality_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_clean_quality_metrics",
            "family_id": family_id,
            "overall": {
                "status": "ok",
                "availability_reason": None,
                "expected_count": 1,
                "count": 1,
                "missing_count": 0,
                "error_count": 0,
                "mean_psnr": 30.0,
                "mean_ssim": 0.95,
                "mean_lpips": 0.10,
                "mean_clip_text_similarity": 0.80,
                "clip_model_name": "ViT-B-32",
                "clip_sample_count": 1,
                "lpips_status": "ok",
                "lpips_reason": None,
                "clip_status": "ok",
                "clip_reason": None,
                "quality_runtime": {
                    "torch_device": "cuda:0",
                    "lpips_batch_size": 2,
                    "clip_batch_size": 2,
                },
                "prompt_text_expected": True,
                "prompt_text_available_count": 1,
                "prompt_text_missing_count": 0,
                "prompt_text_coverage_status": "ok",
                "prompt_text_coverage_reason": None,
                "quality_readiness_status": "ready",
                "quality_readiness_reason": None,
                "quality_readiness_blocking": False,
                "quality_readiness_required_for_formal_release": True,
            },
        },
    )
    pw04_attack_quality_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "attack_quality_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_attack_quality_metrics",
            "family_id": family_id,
            "overall": {
                "status": "ok",
                "availability_reason": None,
                "expected_count": 1,
                "count": 1,
                "missing_count": 0,
                "error_count": 0,
                "mean_psnr": 24.0,
                "mean_ssim": 0.91,
                "mean_lpips": 0.18,
                "mean_clip_text_similarity": 0.77,
                "clip_model_name": "ViT-B-32",
                "clip_sample_count": 1,
                "lpips_status": "ok",
                "lpips_reason": None,
                "clip_status": "ok",
                "clip_reason": None,
                "quality_runtime": {
                    "torch_device": "cuda:0",
                    "lpips_batch_size": 2,
                    "clip_batch_size": 2,
                },
                "prompt_text_expected": True,
                "prompt_text_available_count": 1,
                "prompt_text_missing_count": 0,
                "prompt_text_coverage_status": "ok",
                "prompt_text_coverage_reason": None,
                "quality_readiness_status": "ready",
                "quality_readiness_reason": None,
                "quality_readiness_blocking": False,
                "quality_readiness_required_for_formal_release": True,
            },
        },
    )
    pw04_clean_attack_overview_path = _write_json(
        family_root / "exports" / "pw04" / "clean_attack_overview.json",
        {
            "artifact_type": "paper_workflow_pw04_clean_attack_overview",
            "family_id": family_id,
        },
    )
    pw04_system_final_auxiliary_attack_summary_path = _write_json(
        family_root / "exports" / "pw04" / "robustness" / "system_final_auxiliary_attack_summary.json",
        {
            "artifact_type": "paper_workflow_pw04_system_final_auxiliary_attack_summary",
            "family_id": family_id,
            "scope": "system_final_auxiliary",
            "canonical": False,
            "analysis_only": True,
        },
    )
    pw04_system_final_auxiliary_attack_by_family_path = _write_text(
        family_root / "exports" / "pw04" / "robustness" / "system_final_auxiliary_attack_by_family.csv",
        "attack_family,system_final_auxiliary_attack_tpr\nresize,0.6\n",
    )
    pw04_system_final_auxiliary_attack_by_condition_path = _write_text(
        family_root / "exports" / "pw04" / "robustness" / "system_final_auxiliary_attack_by_condition.csv",
        "attack_condition_key,system_final_auxiliary_attack_tpr\nresize::0.8,0.6\n",
    )
    pw04_conditional_rescue_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "geometry_diagnostics" / "conditional_rescue_metrics.json",
        {
            "artifact_type": "paper_workflow_pw04_geometry_conditional_rescue_metrics_export",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "readiness": {
                "status": "ready",
                "reason": None,
                "required_for_formal_release": False,
                "blocking": False,
            },
            "overall": {
                "event_count": 1,
                "content_failed_subset_event_count": 1,
            },
        },
    )
    pw04_geometry_optional_claim_summary_path = _write_json(
        family_root / "exports" / "pw04" / "geometry_diagnostics" / "geometry_optional_claim_summary.json",
        {
            "artifact_type": "paper_workflow_pw04_geometry_optional_claim_summary",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "readiness": {
                "status": "ready",
                "reason": None,
                "required_for_formal_release": False,
                "blocking": False,
            },
            "overall": {
                "event_count": 1,
                "eligible_event_count": 1,
                "evidence_event_count": 1,
            },
        },
    )
    pw04_geometry_optional_claim_by_family_path = _write_text(
        family_root / "exports" / "pw04" / "geometry_diagnostics" / "geometry_optional_claim_by_family.csv",
        "attack_family,status,eligible_event_count\nresize,ok,1\n",
    )
    pw04_geometry_optional_claim_by_severity_path = _write_text(
        family_root / "exports" / "pw04" / "geometry_diagnostics" / "geometry_optional_claim_by_severity.csv",
        "severity_level_index,severity_label,status,eligible_event_count\n0,scale_factor=0.75,ok,1\n",
    )
    pw04_geometry_optional_claim_example_manifest_path = _write_json(
        family_root / "exports" / "pw04" / "geometry_diagnostics" / "geometry_optional_claim_example_manifest.json",
        {
            "artifact_type": "paper_workflow_pw04_geometry_optional_claim_example_manifest",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "eligible_event_count": 1,
            "example_count": 1,
            "rows": [
                {
                    "attack_event_id": "attack_event_000001",
                    "attack_family": "resize",
                    "eligible_for_optional_claim": True,
                }
            ],
        },
    )
    pw04_payload_attack_summary_path = _write_json(
        family_root / "exports" / "pw04" / "payload_robustness" / "payload_attack_summary.json",
        {
            "artifact_type": "paper_workflow_pw04_payload_attack_summary",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "future_upstream_sidecar_required": False,
            "readiness": {
                "status": "ready",
                "reason": None,
                "required_for_formal_release": True,
                "blocking": False,
            },
            "overall": {
                "event_count": 1,
                "available_payload_event_count": 1,
            },
            "probe_overall": {
                "status": "ready",
                "reason": None,
                "event_count": 1,
                "available_probe_event_count": 1,
                "probe_margin_threshold": None,
                "probe_effective_n_bits": 96,
                "probe_agreement_count": 96,
                "probe_bit_accuracy": 1.0,
                "probe_support_rate": 1.0,
            },
        },
    )
    pw04_wrong_event_attestation_challenge_summary_path = _write_json(
        family_root / "exports" / "pw04" / "robustness" / "wrong_event_attestation_challenge_summary.json",
        {
            "artifact_type": "paper_workflow_pw04_wrong_event_attestation_challenge_summary",
            "family_id": family_id,
            "status": "ok",
            "reason": None,
            "future_upstream_sidecar_required": False,
            "readiness": {
                "status": "ready",
                "reason": None,
                "required_for_formal_release": True,
                "blocking": False,
            },
            "overall": {
                "event_count": 1,
                "attempted_event_count": 1,
                "wrong_event_rejected_count": 1,
            },
            "rows": [],
        },
    )

    pw04_paper_metric_registry_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "paper_metric_registry.json",
        {
            "artifact_type": "paper_workflow_pw04_paper_metric_registry",
            "family_id": family_id,
            "scope_names": ["content_chain", "event_attestation", "system_final"],
        },
    )
    pw04_content_chain_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "content_chain_metrics.json",
        {"scope": "content_chain", "family_id": family_id},
    )
    pw04_event_attestation_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "event_attestation_metrics.json",
        {"scope": "event_attestation", "family_id": family_id},
    )
    pw04_system_final_metrics_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "system_final_metrics.json",
        {"scope": "system_final", "family_id": family_id},
    )
    pw04_bootstrap_confidence_intervals_path = _write_json(
        family_root / "exports" / "pw04" / "metrics" / "bootstrap_confidence_intervals.json",
        {"family_id": family_id, "intervals": []},
    )

    pw04_main_metrics_summary_csv_path = _write_text(
        family_root / "exports" / "pw04" / "tables" / "main_metrics_summary.csv",
        "scope,tpr\ncontent_chain,0.5\n",
    )
    pw04_attack_family_summary_paper_csv_path = _write_text(
        family_root / "exports" / "pw04" / "tables" / "attack_family_summary_paper.csv",
        "attack_family,attack_tpr\nresize,0.5\n",
    )
    pw04_attack_condition_summary_paper_csv_path = _write_text(
        family_root / "exports" / "pw04" / "tables" / "attack_condition_summary_paper.csv",
        "attack_condition_key,attack_tpr\nresize::0.8,0.5\n",
    )
    pw04_rescue_metrics_summary_csv_path = _write_text(
        family_root / "exports" / "pw04" / "tables" / "rescue_metrics_summary.csv",
        "scope,rescue_rate\nsystem_final,0.1\n",
    )
    pw04_bootstrap_confidence_intervals_csv_path = _write_text(
        family_root / "exports" / "pw04" / "tables" / "bootstrap_confidence_intervals.csv",
        "scope,lower,upper\nsystem_final,0.4,0.8\n",
    )

    figure_paths = {
        "attack_tpr_by_family_path": _write_text(
            family_root / "exports" / "pw04" / "figures" / "attack_tpr_by_family.png",
            "png placeholder\n",
        ),
        "clean_vs_attack_scope_overview_path": _write_text(
            family_root / "exports" / "pw04" / "figures" / "clean_vs_attack_scope_overview.png",
            "png placeholder\n",
        ),
        "rescue_breakdown_path": _write_text(
            family_root / "exports" / "pw04" / "figures" / "rescue_breakdown.png",
            "png placeholder\n",
        ),
    }
    tail_paths = {
        "estimated_tail_fpr_1e4_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "estimated_tail_fpr_1e4.json",
            {
                "family_id": family_id,
                "target": "1e-4",
                "readiness": {"status": "disabled", "reason": "tail_estimation_flag_not_enabled"},
            },
        ),
        "estimated_tail_fpr_1e5_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "estimated_tail_fpr_1e5.json",
            {
                "family_id": family_id,
                "target": "1e-5",
                "readiness": {"status": "disabled", "reason": "tail_estimation_flag_not_enabled"},
            },
        ),
        "tail_fit_diagnostics_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "tail_fit_diagnostics.json",
            {
                "family_id": family_id,
                "status": "ok",
                "readiness": {"status": "disabled", "reason": "tail_estimation_flag_not_enabled"},
            },
        ),
        "tail_fit_stability_summary_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "tail_fit_stability_summary.json",
            {
                "family_id": family_id,
                "status": "stable",
                "readiness": {"status": "disabled", "reason": "tail_estimation_flag_not_enabled"},
            },
        ),
    }

    pw04_summary_path = family_root / "runtime_state" / "pw04_summary.json"
    _write_json(
        pw04_summary_path,
        {
            "stage_name": "PW04_Attack_Merge_And_Metrics",
            "stage_run_id": "pw04_run_demo",
            "family_id": family_id,
            "status": "completed",
            "paper_exports_completed": True,
            "completed_attack_event_count": 2,
            "attack_merge_manifest_path": normalize_path_value(pw04_attack_merge_manifest_path),
            "attack_positive_pool_manifest_path": normalize_path_value(pw04_attack_positive_pool_manifest_path),
            "attack_negative_pool_manifest_path": normalize_path_value(pw04_attack_negative_pool_manifest_path),
            "formal_attack_final_decision_metrics_path": normalize_path_value(pw04_formal_attack_final_decision_metrics_path),
            "formal_attack_attestation_metrics_path": normalize_path_value(pw04_formal_attack_attestation_metrics_path),
            "derived_attack_union_metrics_path": normalize_path_value(pw04_derived_attack_union_metrics_path),
            "formal_attack_negative_metrics_path": normalize_path_value(pw04_formal_attack_negative_metrics_path),
            "clean_attack_overview_path": normalize_path_value(pw04_clean_attack_overview_path),
            "clean_quality_metrics_path": normalize_path_value(pw04_clean_quality_metrics_path),
            "attack_quality_metrics_path": normalize_path_value(pw04_attack_quality_metrics_path),
            "paper_scope_registry_path": normalize_path_value(pw04_paper_metric_registry_path),
            "canonical_metrics_paths": {
                "content_chain": normalize_path_value(pw04_content_chain_metrics_path),
                "event_attestation": normalize_path_value(pw04_event_attestation_metrics_path),
                "system_final": normalize_path_value(pw04_system_final_metrics_path),
            },
            "paper_tables_paths": {
                "main_metrics_summary_csv_path": normalize_path_value(pw04_main_metrics_summary_csv_path),
                "attack_family_summary_paper_csv_path": normalize_path_value(pw04_attack_family_summary_paper_csv_path),
                "attack_condition_summary_paper_csv_path": normalize_path_value(pw04_attack_condition_summary_paper_csv_path),
                "rescue_metrics_summary_csv_path": normalize_path_value(pw04_rescue_metrics_summary_csv_path),
            },
            "paper_figures_paths": {
                key_name: normalize_path_value(path_obj)
                for key_name, path_obj in figure_paths.items()
            },
            "bootstrap_confidence_intervals_path": normalize_path_value(pw04_bootstrap_confidence_intervals_path),
            "bootstrap_confidence_intervals_csv_path": normalize_path_value(pw04_bootstrap_confidence_intervals_csv_path),
            "geometry_optional_claim_summary_path": normalize_path_value(pw04_geometry_optional_claim_summary_path),
            "geometry_optional_claim_by_family_path": normalize_path_value(pw04_geometry_optional_claim_by_family_path),
            "geometry_optional_claim_by_severity_path": normalize_path_value(pw04_geometry_optional_claim_by_severity_path),
            "geometry_optional_claim_example_manifest_path": normalize_path_value(pw04_geometry_optional_claim_example_manifest_path),
            "payload_attack_summary_path": normalize_path_value(pw04_payload_attack_summary_path),
            "tail_estimation_paths": {
                key_name: normalize_path_value(path_obj)
                for key_name, path_obj in tail_paths.items()
            },
            "analysis_only_artifact_paths": {
                "pw02_system_final_auxiliary_operating_semantics": normalize_path_value(pw02_system_final_auxiliary_operating_semantics_path),
                "pw02_system_final_auxiliary_roc_curve": normalize_path_value(pw02_system_final_auxiliary_roc_curve_path),
                "pw02_clean_quality_pair_manifest": normalize_path_value(pw02_clean_quality_pair_manifest_path),
                "pw02_payload_clean_summary": normalize_path_value(pw02_payload_clean_summary_path),
                "pw04_system_final_auxiliary_attack_summary": normalize_path_value(pw04_system_final_auxiliary_attack_summary_path),
                "pw04_system_final_auxiliary_attack_by_family": normalize_path_value(pw04_system_final_auxiliary_attack_by_family_path),
                "pw04_system_final_auxiliary_attack_by_condition": normalize_path_value(pw04_system_final_auxiliary_attack_by_condition_path),
                "pw04_conditional_rescue_metrics": normalize_path_value(pw04_conditional_rescue_metrics_path),
                "pw04_geometry_optional_claim_summary": normalize_path_value(pw04_geometry_optional_claim_summary_path),
                "pw04_geometry_optional_claim_by_family": normalize_path_value(pw04_geometry_optional_claim_by_family_path),
                "pw04_geometry_optional_claim_by_severity": normalize_path_value(pw04_geometry_optional_claim_by_severity_path),
                "pw04_geometry_optional_claim_example_manifest": normalize_path_value(pw04_geometry_optional_claim_example_manifest_path),
                "pw04_payload_attack_summary": normalize_path_value(pw04_payload_attack_summary_path),
                "pw04_wrong_event_attestation_challenge_summary": normalize_path_value(pw04_wrong_event_attestation_challenge_summary_path),
            },
            "analysis_only_artifact_annotations": {
                "pw02_system_final_auxiliary_operating_semantics": {"canonical": False, "analysis_only": True},
                "pw02_system_final_auxiliary_roc_curve": {"canonical": False, "analysis_only": True},
                "pw02_clean_quality_pair_manifest": {"canonical": False, "analysis_only": True},
                "pw02_payload_clean_summary": {"canonical": False, "analysis_only": True},
                "pw04_system_final_auxiliary_attack_summary": {"canonical": False, "analysis_only": True},
                "pw04_system_final_auxiliary_attack_by_family": {"canonical": False, "analysis_only": True},
                "pw04_system_final_auxiliary_attack_by_condition": {"canonical": False, "analysis_only": True},
                "pw04_conditional_rescue_metrics": {"canonical": False, "analysis_only": True},
                "pw04_geometry_optional_claim_summary": {"canonical": False, "analysis_only": True},
                "pw04_geometry_optional_claim_by_family": {"canonical": False, "analysis_only": True},
                "pw04_geometry_optional_claim_by_severity": {"canonical": False, "analysis_only": True},
                "pw04_geometry_optional_claim_example_manifest": {"canonical": False, "analysis_only": True},
                "pw04_payload_attack_summary": {"canonical": False, "analysis_only": True},
                "pw04_wrong_event_attestation_challenge_summary": {"canonical": False, "analysis_only": True},
            },
        },
    )

    return {
        "drive_root": drive_root,
        "family_id": family_id,
        "family_root": family_root,
        "family_manifest_path": family_manifest_path,
        "config_snapshot_path": config_snapshot_path,
        "pw02_finalize_manifest_path": pw02_finalize_manifest_path,
        "pw02_content_threshold_export_path": pw02_content_threshold_export_path,
        "pw02_attestation_threshold_export_path": pw02_attestation_threshold_export_path,
        "pw04_summary_path": pw04_summary_path,
    }


def test_pw05_release_signoff_packages_canonical_pw04_exports(tmp_path: Path) -> None:
    """
    Verify PW05 packages the finalized family artifacts and emits signoff outputs.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw05_family_fixture(tmp_path)

    summary = pw05_module.run_pw05_release_signoff(
        drive_project_root=Path(str(fixture["drive_root"])),
        family_id=str(fixture["family_id"]),
        stage_run_id="pw05_release_demo",
    )

    summary_path = Path(str(summary["summary_path"]))
    signoff_report_path = Path(str(summary["signoff_report_path"]))
    release_manifest_path = Path(str(summary["release_manifest_path"]))
    stage_manifest_path = Path(str(summary["stage_manifest_path"]))
    package_manifest_path = Path(str(summary["package_manifest_path"]))
    package_path = Path(str(summary["package_path"]))
    formal_run_readiness_report_path = Path(str(summary["formal_run_readiness_report_path"]))

    assert summary["stage_name"] == "PW05_Release_And_Signoff"
    assert summary["status"] == "completed"
    assert summary["decision"] == "ALLOW_FREEZE"
    assert summary["signoff_status"] == "passed"
    assert summary["release_status"] == "passed"
    assert summary["paper_closure_status"] == "passed"

    for path_obj in [
        summary_path,
        signoff_report_path,
        release_manifest_path,
        stage_manifest_path,
        package_manifest_path,
        package_path,
        formal_run_readiness_report_path,
    ]:
        assert path_obj.exists(), path_obj

    formal_run_readiness_report = _load_json_dict(formal_run_readiness_report_path)
    signoff_report = _load_json_dict(signoff_report_path)
    release_manifest = _load_json_dict(release_manifest_path)
    stage_manifest = _load_json_dict(stage_manifest_path)
    package_manifest = _load_json_dict(package_manifest_path)
    persisted_summary = _load_json_dict(summary_path)

    assert formal_run_readiness_report["overall_status"] == "ready"
    assert formal_run_readiness_report["decision"] == "ALLOW_FREEZE"
    assert formal_run_readiness_report["blocking_components"] == []
    assert formal_run_readiness_report["blocking_reasons"] == []
    assert "tail_estimation" in formal_run_readiness_report["advisory_components"]
    assert "quality_runtime_preflight" not in formal_run_readiness_report
    assert formal_run_readiness_report["components"]["quality_attack"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["quality_clean"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["payload_clean"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["payload_attack"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["payload_clean"]["payload_role"] == "auxiliary_probe"
    assert formal_run_readiness_report["components"]["payload_clean"]["payload_claim_scope"] == "non_primary_decision"
    assert formal_run_readiness_report["components"]["payload_clean"]["payload_primary_release_dependency"] is False
    assert formal_run_readiness_report["components"]["payload_attack"]["payload_role"] == "auxiliary_probe"
    assert formal_run_readiness_report["components"]["payload_attack"]["payload_claim_scope"] == "non_primary_decision"
    assert formal_run_readiness_report["components"]["payload_attack"]["payload_primary_release_dependency"] is False
    assert formal_run_readiness_report["components"]["wrong_event_attack"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["geometry_conditional_rescue"]["blocking"] is False
    assert formal_run_readiness_report["components"]["geometry_optional_claim"]["blocking"] is False
    for component_name in ["quality_clean", "quality_attack"]:
        component_payload = cast(Dict[str, Any], formal_run_readiness_report["components"][component_name])
        assert "lpips_dependency_ready" not in component_payload
        assert "lpips_dependency_reason" not in component_payload
        assert "clip_dependency_ready" not in component_payload
        assert "clip_dependency_reason" not in component_payload
    assert formal_run_readiness_report["recommended_run_plan"]["plan_name"] == "formal_paper_run_minimal_scale_up"
    assert formal_run_readiness_report["recommended_run_plan"]["gates_before_scale_up"] == []
    serialized_readiness_report = json.dumps(formal_run_readiness_report, ensure_ascii=False, sort_keys=True)
    serialized_signoff_report = json.dumps(signoff_report, ensure_ascii=False, sort_keys=True)
    for forbidden_token in [
        "lpips_import_failed",
        "open_clip_import_failed",
        "lpips_dependency_ready",
        "lpips_dependency_reason",
        "clip_dependency_ready",
        "clip_dependency_reason",
        "quality_runtime_preflight",
    ]:
        assert forbidden_token not in serialized_readiness_report
        assert forbidden_token not in serialized_signoff_report
    assert signoff_report["checked_source_artifact_count"] >= 20
    assert signoff_report["analysis_only_artifact_count"] == 14
    assert signoff_report["formal_run_readiness_report_path"] == normalize_path_value(formal_run_readiness_report_path)
    assert "pw04_summary" in release_manifest["release_copy_paths"]
    assert "family_manifest" in release_manifest["source_artifact_index"]
    assert "pw04_clean_quality_metrics" in release_manifest["source_artifact_index"]
    assert "pw04_attack_quality_metrics" in release_manifest["source_artifact_index"]
    assert "pw02_positive_source_payload_reference_sidecar_e000001" in release_manifest["source_artifact_index"]
    assert "pw02_positive_source_payload_decode_sidecar_e000001" in release_manifest["source_artifact_index"]
    assert "pw04_attacked_positive_payload_decode_sidecar_e000001" in release_manifest["source_artifact_index"]
    assert release_manifest["formal_run_readiness_report_path"] == normalize_path_value(formal_run_readiness_report_path)
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_system_final_auxiliary_attack_summary"] == {
        "source_path": normalize_path_value(
            Path(str(fixture["drive_root"]))
            / "paper_workflow"
            / "families"
            / str(fixture["family_id"])
            / "exports"
            / "pw04"
            / "robustness"
            / "system_final_auxiliary_attack_summary.json"
        ),
        "release_copy_path": "source/exports/pw04/robustness/system_final_auxiliary_attack_summary.json",
        "canonical": False,
        "analysis_only": True,
    }
    assert release_manifest["analysis_only_artifact_annotations"]["pw02_clean_quality_pair_manifest"]["release_copy_path"] == (
        "source/exports/pw02/quality/clean_quality_pair_manifest.json"
    )
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_payload_attack_summary"]["release_copy_path"] == (
        "source/exports/pw04/payload_robustness/payload_attack_summary.json"
    )
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_geometry_optional_claim_summary"]["release_copy_path"] == (
        "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_summary.json"
    )
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_geometry_optional_claim_by_family"]["release_copy_path"] == (
        "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_by_family.csv"
    )
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_geometry_optional_claim_by_severity"]["release_copy_path"] == (
        "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_by_severity.csv"
    )
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_geometry_optional_claim_example_manifest"]["release_copy_path"] == (
        "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_example_manifest.json"
    )
    assert "package_zip" in persisted_summary["generated_artifact_index"]
    assert package_manifest["stage_name"] == "PW05_Release_And_Signoff"
    assert stage_manifest["source_stage_name"] == "PW04_Attack_Merge_And_Metrics"
    assert stage_manifest["formal_run_readiness_report_path"] == normalize_path_value(formal_run_readiness_report_path)
    assert persisted_summary["analysis_only_artifact_paths"]["pw02_system_final_auxiliary_operating_semantics"].endswith(
        "/exports/pw02/operating_metrics/system_final_auxiliary_operating_semantics.json"
    )
    assert persisted_summary["formal_run_readiness_report_path"] == normalize_path_value(formal_run_readiness_report_path)

    with zipfile.ZipFile(package_path, "r") as archive:
        members = set(archive.namelist())
    assert "artifacts/stage_manifest.json" in members
    assert "artifacts/readiness/formal_run_readiness_report.json" in members
    assert "artifacts/release/release_manifest.json" in members
    assert "artifacts/signoff/signoff_report.json" in members
    assert "source/runtime_state/pw04_summary.json" in members
    assert "source/exports/pw04/metrics/paper_metric_registry.json" in members
    assert "source/exports/pw04/metrics/clean_quality_metrics.json" in members
    assert "source/exports/pw04/metrics/attack_quality_metrics.json" in members
    assert "source/exports/pw02/thresholds/content/thresholds.json" in members
    assert "source/exports/pw04/attack_negative_pool_manifest.json" in members
    assert "source/exports/pw04/formal_attack_negative_metrics.json" in members
    assert "source/source_shards/positive/shard_0000/events/event_000001/artifacts/payload_reference_sidecar.json" in members
    assert "source/source_shards/positive/shard_0000/events/event_000001/artifacts/payload_decode_sidecar.json" in members
    assert "source/attack_shards/shard_0000/events/event_000001/artifacts/payload_decode_sidecar.json" in members
    assert "source/exports/pw02/operating_metrics/system_final_auxiliary_operating_semantics.json" in members
    assert "source/exports/pw02/quality/clean_quality_pair_manifest.json" in members
    assert "source/exports/pw02/payload/payload_clean_summary.json" in members
    assert "source/exports/pw04/geometry_diagnostics/conditional_rescue_metrics.json" in members
    assert "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_summary.json" in members
    assert "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_by_family.csv" in members
    assert "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_by_severity.csv" in members
    assert "source/exports/pw04/geometry_diagnostics/geometry_optional_claim_example_manifest.json" in members
    assert "source/exports/pw04/payload_robustness/payload_attack_summary.json" in members
    assert "source/exports/pw04/robustness/system_final_auxiliary_attack_summary.json" in members
    assert "source/exports/pw04/robustness/wrong_event_attestation_challenge_summary.json" in members


def test_pw05_backfills_payload_attack_binding_from_top_level_pw04_summary(tmp_path: Path) -> None:
    """
    Verify PW05 recovers the payload attack binding from the top-level PW04 summary field.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw05_family_fixture(tmp_path)
    pw04_summary_path = Path(str(fixture["pw04_summary_path"]))
    pw04_summary = _load_json_dict(pw04_summary_path)
    payload_attack_summary_path = str(pw04_summary["payload_attack_summary_path"])
    analysis_only_artifact_paths = cast(Dict[str, Any], pw04_summary["analysis_only_artifact_paths"])
    analysis_only_artifact_annotations = cast(Dict[str, Any], pw04_summary["analysis_only_artifact_annotations"])
    analysis_only_artifact_paths.pop("pw04_payload_attack_summary")
    analysis_only_artifact_annotations.pop("pw04_payload_attack_summary")
    write_json_atomic(pw04_summary_path, pw04_summary)

    summary = pw05_module.run_pw05_release_signoff(
        drive_project_root=Path(str(fixture["drive_root"])),
        family_id=str(fixture["family_id"]),
        stage_run_id="pw05_release_demo",
    )

    formal_run_readiness_report = _load_json_dict(Path(str(summary["formal_run_readiness_report_path"])))
    signoff_report = _load_json_dict(Path(str(summary["signoff_report_path"])))
    release_manifest = _load_json_dict(Path(str(summary["release_manifest_path"])))
    persisted_summary = _load_json_dict(Path(str(summary["summary_path"])))

    assert summary["decision"] == "ALLOW_FREEZE"
    assert signoff_report["analysis_only_artifact_count"] == 14
    assert formal_run_readiness_report["components"]["payload_attack"]["status"] == "ready"
    assert formal_run_readiness_report["components"]["payload_attack"]["source_path"] == payload_attack_summary_path
    assert release_manifest["analysis_only_artifact_annotations"]["pw04_payload_attack_summary"] == {
        "source_path": payload_attack_summary_path,
        "release_copy_path": "source/exports/pw04/payload_robustness/payload_attack_summary.json",
        "canonical": False,
        "analysis_only": True,
    }
    assert persisted_summary["analysis_only_artifact_paths"]["pw04_payload_attack_summary"] == payload_attack_summary_path


def test_pw05_blocks_freeze_when_formal_run_readiness_has_blocking_component(tmp_path: Path) -> None:
    """
    Verify PW05 blocks freeze when one required readiness component is not ready.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw05_family_fixture(tmp_path)
    family_root = Path(str(fixture["family_root"]))
    attack_quality_metrics_path = family_root / "exports" / "pw04" / "metrics" / "attack_quality_metrics.json"
    attack_quality_metrics = _load_json_dict(attack_quality_metrics_path)
    attack_quality_overall = cast(Dict[str, Any], attack_quality_metrics["overall"])
    attack_quality_overall["clip_status"] = "missing"
    attack_quality_overall["clip_reason"] = "clip_model_not_available"
    write_json_atomic(attack_quality_metrics_path, attack_quality_metrics)

    summary = pw05_module.run_pw05_release_signoff(
        drive_project_root=Path(str(fixture["drive_root"])),
        family_id=str(fixture["family_id"]),
        stage_run_id="pw05_release_demo",
    )

    formal_run_readiness_report = _load_json_dict(Path(str(summary["formal_run_readiness_report_path"])))
    signoff_report = _load_json_dict(Path(str(summary["signoff_report_path"])))
    run_closure = _load_json_dict(Path(str(summary["run_closure_path"])))

    assert summary["decision"] == "BLOCK_FREEZE"
    assert summary["signoff_status"] == "blocked"
    assert summary["release_status"] == "blocked"
    assert summary["paper_closure_status"] == "blocked"
    assert formal_run_readiness_report["overall_status"] == "blocked"
    assert "quality_attack" in formal_run_readiness_report["blocking_components"]
    assert any("clip_model_not_available" in reason for reason in formal_run_readiness_report["blocking_reasons"])
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    assert signoff_report["blocking_reason_count"] >= 1
    assert run_closure["status"] == {
        "ok": False,
        "reason": "formal_run_readiness_blocked",
        "details": {
            "checked_source_artifact_count": signoff_report["checked_source_artifact_count"],
            "blocking_reason_count": signoff_report["blocking_reason_count"],
        },
    }


def test_pw05_ignores_current_quality_import_state_and_omits_dependency_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify PW05 signoff ignores current quality dependency import state and omits dependency fields.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    fixture = _build_pw05_family_fixture(tmp_path)
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_arg: Any = None,
        locals_arg: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name in {"lpips", "open_clip"}:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return original_import(name, globals_arg, locals_arg, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    summary = pw05_module.run_pw05_release_signoff(
        drive_project_root=Path(str(fixture["drive_root"])),
        family_id=str(fixture["family_id"]),
        stage_run_id="pw05_release_demo",
    )

    formal_run_readiness_report = _load_json_dict(Path(str(summary["formal_run_readiness_report_path"])))
    signoff_report = _load_json_dict(Path(str(summary["signoff_report_path"])))
    quality_clean = cast(Dict[str, Any], formal_run_readiness_report["components"]["quality_clean"])
    quality_attack = cast(Dict[str, Any], formal_run_readiness_report["components"]["quality_attack"])
    serialized_readiness_report = json.dumps(formal_run_readiness_report, ensure_ascii=False, sort_keys=True)
    serialized_signoff_report = json.dumps(signoff_report, ensure_ascii=False, sort_keys=True)

    assert summary["decision"] == "ALLOW_FREEZE"
    assert formal_run_readiness_report["overall_status"] == "ready"
    assert "quality_runtime_preflight" not in formal_run_readiness_report
    for component_payload in [quality_clean, quality_attack]:
        assert "lpips_dependency_ready" not in component_payload
        assert "lpips_dependency_reason" not in component_payload
        assert "clip_dependency_ready" not in component_payload
        assert "clip_dependency_reason" not in component_payload
    for forbidden_token in [
        "lpips_import_failed",
        "open_clip_import_failed",
        "lpips_dependency_ready",
        "lpips_dependency_reason",
        "clip_dependency_ready",
        "clip_dependency_reason",
        "quality_runtime_preflight",
    ]:
        assert forbidden_token not in serialized_readiness_report
        assert forbidden_token not in serialized_signoff_report


def test_pw05_requires_completed_pw04_exports(tmp_path: Path) -> None:
    """
    Verify PW05 rejects a family whose PW04 summary does not declare completed paper exports.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    fixture = _build_pw05_family_fixture(tmp_path)
    pw04_summary_path = Path(str(fixture["pw04_summary_path"]))
    pw04_summary = _load_json_dict(pw04_summary_path)
    pw04_summary["paper_exports_completed"] = False
    write_json_atomic(pw04_summary_path, pw04_summary)

    with pytest.raises(ValueError, match="PW04 summary must confirm completed paper exports"):
        pw05_module.run_pw05_release_signoff(
            drive_project_root=Path(str(fixture["drive_root"])),
            family_id=str(fixture["family_id"]),
            stage_run_id="pw05_release_demo",
        )