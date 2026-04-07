"""
File purpose: Validate PW05 release/signoff packaging over finalized paper_workflow artifacts.
Module type: General module
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, cast

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
    ensure_directory(family_root / "exports" / "pw04" / "manifests")
    ensure_directory(family_root / "exports" / "pw04" / "metrics")
    ensure_directory(family_root / "exports" / "pw04" / "tables")
    ensure_directory(family_root / "exports" / "pw04" / "figures")
    ensure_directory(family_root / "exports" / "pw04" / "tail")

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
    pw04_clean_attack_overview_path = _write_json(
        family_root / "exports" / "pw04" / "clean_attack_overview.json",
        {
            "artifact_type": "paper_workflow_pw04_clean_attack_overview",
            "family_id": family_id,
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
            {"family_id": family_id, "target": "1e-4"},
        ),
        "estimated_tail_fpr_1e5_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "estimated_tail_fpr_1e5.json",
            {"family_id": family_id, "target": "1e-5"},
        ),
        "tail_fit_diagnostics_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "tail_fit_diagnostics.json",
            {"family_id": family_id, "status": "ok"},
        ),
        "tail_fit_stability_summary_path": _write_json(
            family_root / "exports" / "pw04" / "tail" / "tail_fit_stability_summary.json",
            {"family_id": family_id, "status": "stable"},
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
            "formal_attack_final_decision_metrics_path": normalize_path_value(pw04_formal_attack_final_decision_metrics_path),
            "formal_attack_attestation_metrics_path": normalize_path_value(pw04_formal_attack_attestation_metrics_path),
            "derived_attack_union_metrics_path": normalize_path_value(pw04_derived_attack_union_metrics_path),
            "clean_attack_overview_path": normalize_path_value(pw04_clean_attack_overview_path),
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
            "tail_estimation_paths": {
                key_name: normalize_path_value(path_obj)
                for key_name, path_obj in tail_paths.items()
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

    assert summary["stage_name"] == "PW05_Release_And_Signoff"
    assert summary["status"] == "completed"
    assert summary["decision"] == "ALLOW_FREEZE"
    assert summary["signoff_status"] == "passed"
    assert summary["release_status"] == "passed"
    assert summary["paper_closure_status"] == "passed"

    for path_obj in [summary_path, signoff_report_path, release_manifest_path, stage_manifest_path, package_manifest_path, package_path]:
        assert path_obj.exists(), path_obj

    signoff_report = _load_json_dict(signoff_report_path)
    release_manifest = _load_json_dict(release_manifest_path)
    stage_manifest = _load_json_dict(stage_manifest_path)
    package_manifest = _load_json_dict(package_manifest_path)
    persisted_summary = _load_json_dict(summary_path)

    assert signoff_report["checked_source_artifact_count"] >= 20
    assert "pw04_summary" in release_manifest["release_copy_paths"]
    assert "family_manifest" in release_manifest["source_artifact_index"]
    assert "package_zip" in persisted_summary["generated_artifact_index"]
    assert package_manifest["stage_name"] == "PW05_Release_And_Signoff"
    assert stage_manifest["source_stage_name"] == "PW04_Attack_Merge_And_Metrics"

    with zipfile.ZipFile(package_path, "r") as archive:
        members = set(archive.namelist())
    assert "artifacts/stage_manifest.json" in members
    assert "artifacts/release/release_manifest.json" in members
    assert "artifacts/signoff/signoff_report.json" in members
    assert "source/runtime_state/pw04_summary.json" in members
    assert "source/exports/pw04/metrics/paper_metric_registry.json" in members
    assert "source/exports/pw02/thresholds/content/thresholds.json" in members


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