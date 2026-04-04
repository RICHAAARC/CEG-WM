"""
文件目的：验证 stage 04 release/signoff 的最小阻断与放行语义。
Module type: General module

职责边界：
1. 仅覆盖 stage package 输入缺失、lineage 不一致、以及完整一致时的 freeze decision。
2. 通过 importlib 按路径加载 scripts/04_Release_And_Signoff.py，避免非法模块导入。
3. 不恢复 archive 的旧 profile 语义，也不触发任何主链机制运行。
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

from scripts.notebook_runtime_common import (
    build_failure_diagnostics_filename,
    compute_file_sha256,
    compute_mapping_sha256,
    finalize_stage_package,
    resolve_export_package_manifest_path,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "04_Release_And_Signoff.py"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

FORMAL_FINAL_DECISION_METRIC_NAME = "formal_final_decision_metrics"
DERIVED_SYSTEM_UNION_METRIC_NAME = "derived_system_union_metrics"
SYSTEM_FINAL_METRIC_NAME = "system_final_metrics"
SYSTEM_FINAL_METRIC_SEMANTICS = "deprecated_alias_of_derived_system_union_metrics"
FORMAL_PRIMARY_DRIVER_MODE = "formal_final_decision_only"


def _load_stage_04_module() -> Any:
    """
    功能：按文件路径加载 stage 04 脚本模块。 

    Load the stage 04 script module from its filesystem path.

    Args:
        None.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location("stage_04_release_and_signoff", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml_text(path_obj: Path, content: str) -> None:
    """
    功能：写入最小 YAML 文本文件。 

    Write a minimal YAML text file.

    Args:
        path_obj: Destination path.
        content: YAML text content.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding="utf-8")


def _create_stage_package(
    base_dir: Path,
    *,
    stage_name: str,
    stage_run_id: str,
    stage_manifest: Dict[str, Any],
    files: Dict[str, Any],
    source_stage_run_id: str | None,
    source_stage_package_path: str | None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    功能：构造一个最小可消费的 stage package。 

    Create one minimal stage package consumable by stage 04.

    Args:
        base_dir: Base temporary directory.
        stage_name: Stage name.
        stage_run_id: Stage run identifier.
        stage_manifest: Stage manifest payload.
        files: Relative file mapping.
        source_stage_run_id: Optional upstream stage run identifier.
        source_stage_package_path: Optional upstream package path.

    Returns:
        Tuple of package ZIP path and external package manifest.
    """
    package_root = base_dir / "package_root" / stage_name
    export_root = base_dir / "exports" / stage_name / stage_run_id
    run_root = base_dir / "run_root" / stage_name / stage_run_id
    package_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    for relative_path, payload in files.items():
        target_path = package_root / relative_path
        if isinstance(payload, dict):
            write_json_atomic(target_path, payload)
        else:
            _write_yaml_text(target_path, str(payload))

    write_json_atomic(package_root / "artifacts" / "stage_manifest.json", stage_manifest)
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name=stage_name,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=source_stage_run_id,
        source_stage_package_path=source_stage_package_path,
        package_manifest_path=package_manifest_path,
    )
    return Path(str(package_manifest["package_path"])), package_manifest


def _build_failure_diagnostics_package(
    base_dir: Path,
    *,
    stage_name: str,
    stage_run_id: str,
) -> Path:
    """
    功能：构造最小 diagnostics package。 

    Build a minimal failure-diagnostics package that must be rejected by the
    stage-04 formal input gate.

    Args:
        base_dir: Base temporary directory.
        stage_name: Diagnostics stage name.
        stage_run_id: Diagnostics stage run identifier.

    Returns:
        Diagnostics ZIP path.
    """
    export_root = base_dir / "exports" / stage_name / stage_run_id
    export_root.mkdir(parents=True, exist_ok=True)
    package_path = export_root / build_failure_diagnostics_filename(stage_name, stage_run_id)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "failure_diagnostics_summary.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                    "failure_reason": "stage_failed_for_test",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "artifacts/stage_manifest.json",
            json.dumps(
                {
                    "stage_name": stage_name,
                    "stage_run_id": stage_run_id,
                    "stage_status": "failed",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    write_json_atomic(
        resolve_export_package_manifest_path(export_root),
        {
            "stage_name": stage_name,
            "stage_run_id": stage_run_id,
            "package_filename": package_path.name,
            "package_path": package_path.as_posix(),
            "package_sha256": compute_file_sha256(package_path),
            "package_role": "failure_diagnostics_package",
            "package_discovery_scope": "excluded_from_formal_discovery",
            "diagnostics_summary_path": (export_root / "failure_diagnostics_summary.json").as_posix(),
            "diagnostics_package_path": package_path.as_posix(),
        },
    )
    return package_path


def _build_stage_01_package(
    base_dir: Path,
    *,
    workflow_status: str = "ok",
    attestation_evidence_status: str = "ok",
    formal_package_status: str = "generated",
    include_root_records: bool = True,
) -> Dict[str, Any]:
    """
    功能：构造最小 stage 01 package。 

    Build the minimal stage 01 package required by stage 04.

    Args:
        base_dir: Base temporary directory.

    Returns:
        Stage 01 package metadata mapping.
    """
    stage_run_id = "stage01_run"
    runtime_config_path = "/drive/runtime_state/01/runtime_metadata/runtime_config_snapshot.yaml"
    thresholds_path = "/drive/runs/01/artifacts/thresholds/thresholds_artifact.json"
    canonical_source_pool_manifest_path = "/drive/runs/01/artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"
    representative_root_records = {
        "view_role": "representative_summary_view",
        "contract_mode": "compatibility_view",
        "source_truth": "canonical_source_pool",
        "root_records_required": False,
        "source_prompt_index": 0,
        "source_entry_package_relative_path": "artifacts/stage_01_canonical_source_pool/entries/000_source_entry.json",
    }
    attestation_resolution = {
        "overall_status": attestation_evidence_status,
        "required_entry_count": 1,
        "checked_entry_count": 1 if attestation_evidence_status == "ok" else 0,
        "missing_evidence_count": 0 if attestation_evidence_status == "ok" else 1,
        "failing_prompt_indices": [] if attestation_evidence_status == "ok" else [0],
        "failing_source_entry_paths": [] if attestation_evidence_status == "ok" else [
            "artifacts/stage_01_canonical_source_pool/entries/000_source_entry.json"
        ],
        "summary_reason": "ok" if attestation_evidence_status == "ok" else "attestation_evidence_failed",
        "failure_reason": None if attestation_evidence_status == "ok" else "attestation_evidence_failed",
        "representative_root_summary": {
            **representative_root_records,
            "resolution_role": "representative_summary_view_only",
        },
    }
    stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": stage_run_id,
        "stage_status": workflow_status,
        "formal_package_status": formal_package_status,
        "formal_package_role": "formal_stage_package",
        "formal_package_discovery_scope": "discoverable_formal_only",
        "runtime_config_snapshot_path": runtime_config_path,
        "thresholds_path": thresholds_path,
        "threshold_metadata_artifact_path": "/drive/runs/01/artifacts/thresholds/threshold_metadata_artifact.json",
        "evaluation_report_path": "/drive/runs/01/artifacts/evaluation_report.json",
        "run_closure_path": "/drive/runs/01/artifacts/run_closure.json",
        "workflow_summary_path": "/drive/runs/01/artifacts/workflow_summary.json",
        "workflow_summary_status": workflow_status,
        "attestation_evidence_status": attestation_evidence_status,
        "attestation_evidence_summary_reason": attestation_resolution["summary_reason"],
        "attestation_evidence_failure_reason": attestation_resolution["failure_reason"],
        "stage_01_canonical_source_pool_manifest_path": canonical_source_pool_manifest_path,
        "stage_01_canonical_source_pool_manifest_package_relative_path": (
            "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"
        ),
        "stage_01_canonical_source_pool_entry_count": 1,
        "stage_01_root_contract_mode": "compatibility_view",
        "stage_01_root_records_required": False,
        "stage_01_source_truth": "canonical_source_pool",
        "stage_01_representative_root_records": representative_root_records,
    }
    files = {
        "records/calibration_record.json": {"record_type": "calibration"},
        "records/evaluate_record.json": {"record_type": "evaluate"},
        "artifacts/thresholds/thresholds_artifact.json": {"threshold_value": 0.5, "thresholds_digest": "thr01"},
        "artifacts/thresholds/threshold_metadata_artifact.json": {"threshold_metadata_digest": "meta01"},
        "artifacts/evaluation_report.json": {
            "cfg_digest": "cfg01",
            "plan_digest": "plan01",
            "thresholds_digest": "thr01",
            "threshold_metadata_digest": "meta01",
            "impl_digest": "impl01",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_v1",
            "attack_protocol_digest": "attack_digest_01",
            "policy_path": "content_np_geo_rescue",
            "ablation_digest": "abl01",
        },
        "artifacts/run_closure.json": {
            "status": {
                "ok": workflow_status == "ok",
                "reason": "ok" if workflow_status == "ok" else "failed",
            }
        },
        "artifacts/workflow_summary.json": {
            "stage_name": "01_Paper_Full_Cuda",
            "stage_run_id": stage_run_id,
            "status": workflow_status,
            "summary_reason": "ok" if workflow_status == "ok" else "stage_01_formal_failed",
            "failure_reason": None if workflow_status == "ok" else "stage_01_formal_failed",
            "canonical_source_pool_entry_count": 1,
            "representative_root_records": representative_root_records,
            "attestation_evidence_resolution": attestation_resolution,
        },
        "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json": {
            "artifact_role": "canonical_source_pool_root",
            "source_truth": "canonical_source_pool",
            "root_contract_mode": "compatibility_view",
            "root_records_required": False,
            "representative_root_role": "representative_summary_view",
            "prompt_count": 1,
            "entry_count": 1,
            "entries_package_relative_root": "artifacts/stage_01_canonical_source_pool/entries",
            "representative_root_records": representative_root_records,
        },
        "runtime_metadata/runtime_config_snapshot.yaml": "policy_path: content_np_geo_rescue\n",
    }
    if include_root_records:
        files["records/embed_record.json"] = {"record_type": "embed"}
        files["records/detect_record.json"] = {"record_type": "detect"}
    package_path, package_manifest = _create_stage_package(
        base_dir,
        stage_name="01_Paper_Full_Cuda",
        stage_run_id=stage_run_id,
        stage_manifest=stage_manifest,
        files=files,
        source_stage_run_id=None,
        source_stage_package_path=None,
    )
    return {
        "package_path": package_path,
        "package_manifest": package_manifest,
        "stage_manifest": stage_manifest,
        "stage_run_id": stage_run_id,
    }


def _build_stage_02_package(base_dir: Path, stage_01_info: Dict[str, Any], *, mismatch_lineage: bool = False) -> Dict[str, Any]:
    """
    功能：构造最小 stage 02 package。 

    Build the minimal stage 02 package required by stage 04.

    Args:
        base_dir: Base temporary directory.
        stage_01_info: Stage 01 package metadata.
        mismatch_lineage: Whether to inject a lineage mismatch.

    Returns:
        Stage 02 package metadata mapping.
    """
    stage_run_id = "stage02_run"
    source_stage_run_id = "wrong_stage01_run" if mismatch_lineage else stage_01_info["stage_run_id"]
    source_package_sha256 = "wrong_sha" if mismatch_lineage else stage_01_info["package_manifest"]["package_sha256"]
    source_package_manifest_digest = "wrong_digest" if mismatch_lineage else compute_mapping_sha256(stage_01_info["package_manifest"])
    stage_manifest = {
        "stage_name": "02_Parallel_Attestation_Statistics",
        "stage_run_id": stage_run_id,
        "stage_status": "ok",
        "source_stage_name": "01_Paper_Full_Cuda",
        "source_stage_run_id": source_stage_run_id,
        "source_package_sha256": source_package_sha256,
        "source_package_manifest_digest": source_package_manifest_digest,
        "source_stage_manifest_path": "/drive/runtime_state/02/lineage/source_stage_manifest.json",
        "source_package_manifest_path": "/drive/runtime_state/02/lineage/source_package_manifest.json",
        "source_stage_manifest_copy_path": "/drive/runtime_state/02/lineage/source_stage_manifest.json",
        "source_runtime_config_snapshot_path": stage_01_info["stage_manifest"]["runtime_config_snapshot_path"],
        "source_thresholds_artifact_path": stage_01_info["stage_manifest"]["thresholds_path"],
        "evaluation_report_path": "/drive/runs/02/artifacts/evaluation_report.json",
        "run_closure_path": "/drive/runs/02/artifacts/run_closure.json",
        "workflow_summary_path": "/drive/runs/02/artifacts/workflow_summary.json",
    }
    files = {
        "records/calibration_record.json": {"record_type": "calibration"},
        "records/evaluate_record.json": {"record_type": "evaluate"},
        "artifacts/thresholds/thresholds_artifact.json": {"threshold_value": 0.6, "thresholds_digest": "thr02"},
        "artifacts/thresholds/threshold_metadata_artifact.json": {"threshold_metadata_digest": "meta02"},
        "artifacts/evaluation_report.json": {
            "cfg_digest": "cfg02",
            "plan_digest": "plan02",
            "thresholds_digest": "thr02",
            "threshold_metadata_digest": "meta02",
            "impl_digest": "impl02",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_v1",
            "attack_protocol_digest": "attack_digest_02",
            "policy_path": "content_np_geo_rescue",
        },
        "artifacts/run_closure.json": {"status": {"ok": True, "reason": "ok"}},
        "artifacts/workflow_summary.json": {
            "stage_name": "02_Parallel_Attestation_Statistics",
            "stage_run_id": stage_run_id,
            "status": "ok",
        },
        "lineage/source_stage_manifest.json": stage_01_info["stage_manifest"],
        "lineage/source_package_manifest.json": stage_01_info["package_manifest"],
        "runtime_metadata/runtime_config_snapshot.yaml": "parallel_attestation_statistics:\n  enabled: true\n",
    }
    package_path, package_manifest = _create_stage_package(
        base_dir,
        stage_name="02_Parallel_Attestation_Statistics",
        stage_run_id=stage_run_id,
        stage_manifest=stage_manifest,
        files=files,
        source_stage_run_id=stage_01_info["stage_run_id"],
        source_stage_package_path=str(stage_01_info["package_path"]),
    )
    return {
        "package_path": package_path,
        "package_manifest": package_manifest,
        "stage_manifest": stage_manifest,
        "stage_run_id": stage_run_id,
    }


def _build_stage_03_package(
    base_dir: Path,
    stage_01_info: Dict[str, Any],
    *,
    mismatch_lineage: bool = False,
    primary_scope: str = "system_final",
    primary_metric_name: str = FORMAL_FINAL_DECISION_METRIC_NAME,
    primary_driver_mode: str = FORMAL_PRIMARY_DRIVER_MODE,
    primary_status_source: str = FORMAL_FINAL_DECISION_METRIC_NAME,
    primary_summary_basis_scope: str = "system_final",
    primary_summary_basis_metric_name: str = FORMAL_FINAL_DECISION_METRIC_NAME,
    include_formal_final_decision_metrics: bool = True,
    include_derived_system_union_metrics: bool = True,
    include_system_final_metrics: bool = True,
    include_auxiliary_scopes: bool = True,
    auxiliary_analysis_runtime_executed: bool = False,
    include_legacy_scalar_contract: bool = False,
    include_internal_scalar_driver_residual: bool = False,
    legacy_scalar_formal_scope: str = "lf_channel",
    legacy_scalar_formal_score_name: str = "lf_channel_score",
) -> Dict[str, Any]:
    """
    功能：构造最小 stage 03 package。 

    Build the minimal stage 03 package required by stage 04.

    Args:
        base_dir: Base temporary directory.
        stage_01_info: Stage 01 package metadata.

    Returns:
        Stage 03 package metadata mapping.
    """
    stage_run_id = "stage03_run"
    source_stage_run_id = "wrong_stage01_run" if mismatch_lineage else stage_01_info["stage_run_id"]
    source_package_sha256 = "wrong_sha" if mismatch_lineage else stage_01_info["package_manifest"]["package_sha256"]
    source_package_manifest_digest = (
        "wrong_digest" if mismatch_lineage else compute_mapping_sha256(stage_01_info["package_manifest"])
    )
    lineage_stage_manifest = dict(stage_01_info["stage_manifest"])
    lineage_package_manifest = dict(stage_01_info["package_manifest"])
    if mismatch_lineage:
        lineage_stage_manifest["stage_run_id"] = source_stage_run_id
        lineage_package_manifest["stage_run_id"] = source_stage_run_id
        lineage_package_manifest["package_sha256"] = source_package_sha256
    auxiliary_scopes = ["content_chain", "lf_channel"] if include_auxiliary_scopes else ["lf_channel"]
    formal_final_decision_metrics_presence = {
        "rows_with_formal_final_decision_metrics": 1 if include_formal_final_decision_metrics else 0,
        "ok_rows_with_formal_final_decision_metrics": 1 if include_formal_final_decision_metrics else 0,
        "rows_total": 1,
    }
    derived_system_union_metrics_presence = {
        "rows_with_derived_system_union_metrics": 1 if include_derived_system_union_metrics else 0,
        "ok_rows_with_derived_system_union_metrics": 1 if include_derived_system_union_metrics else 0,
        "rows_total": 1,
    }
    system_final_metrics_presence = {
        "rows_with_system_final_metrics": 1 if include_system_final_metrics else 0,
        "ok_rows_with_system_final_metrics": 1 if include_system_final_metrics else 0,
        "rows_total": 1,
    }
    scope_manifest: Dict[str, Any] = {
        "primary_scope": primary_scope,
        "primary_metric_name": primary_metric_name,
        "primary_summary_basis_scope": primary_summary_basis_scope,
        "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
        "auxiliary_scopes": auxiliary_scopes,
    }
    if include_legacy_scalar_contract:
        scope_manifest.update(
            {
                "auxiliary_metric_names": {
                    "content_chain": "content_chain_score",
                    "lf_channel": "lf_channel_score",
                },
                "scalar_formal_scope": legacy_scalar_formal_scope,
                "scalar_calibration_scope": legacy_scalar_formal_scope,
                "scalar_formal_score_name": legacy_scalar_formal_score_name,
            }
        )

    stage_manifest = {
        "stage_name": "03_Experiment_Matrix_Full",
        "stage_run_id": stage_run_id,
        "stage_status": "ok",
        "source_stage_name": "01_Paper_Full_Cuda",
        "source_stage_run_id": source_stage_run_id,
        "source_package_sha256": source_package_sha256,
        "source_package_manifest_digest": source_package_manifest_digest,
        "source_stage_manifest_path": "/drive/runtime_state/03/lineage/source_stage_manifest.json",
        "source_package_manifest_path": "/drive/runtime_state/03/lineage/source_package_manifest.json",
        "source_stage_manifest_copy_path": "/drive/runtime_state/03/lineage/source_stage_manifest.json",
        "source_runtime_config_snapshot_path": stage_01_info["stage_manifest"]["runtime_config_snapshot_path"],
        "source_thresholds_artifact_path": stage_01_info["stage_manifest"]["thresholds_path"],
        "evaluation_report_path": "/drive/runs/03/artifacts/aggregate_report.json",
        "run_closure_path": "/drive/runs/03/artifacts/run_closure.json",
        "workflow_summary_path": "/drive/runs/03/artifacts/workflow_summary.json",
        "thresholds_path": "/drive/runs/03/global_calibrate/artifacts/thresholds/thresholds_artifact.json",
        "threshold_metadata_artifact_path": "/drive/runs/03/global_calibrate/artifacts/thresholds/threshold_metadata_artifact.json",
        "primary_evaluation_scope": primary_scope,
        "primary_metric_name": primary_metric_name,
        "primary_driver_mode": primary_driver_mode,
        "primary_status_source": primary_status_source,
        "primary_summary_basis_scope": primary_summary_basis_scope,
        "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
        "auxiliary_scopes": auxiliary_scopes,
        "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
        "scope_manifest": scope_manifest,
        "formal_final_decision_metrics_presence": formal_final_decision_metrics_presence,
        "derived_system_union_metrics_presence": derived_system_union_metrics_presence,
        "system_final_metrics_presence": system_final_metrics_presence,
        "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
    }
    if include_legacy_scalar_contract:
        stage_manifest["scalar_formal_scope"] = legacy_scalar_formal_scope
        stage_manifest["scalar_formal_score_name"] = legacy_scalar_formal_score_name

    anchor_row = {
        "grid_item_digest": "grid01",
        "evaluation_scope": primary_scope,
        "primary_metric_name": primary_metric_name,
        "primary_summary_basis_scope": primary_summary_basis_scope,
        "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
        "cfg_digest": "cfg03",
        "plan_digest": "plan03",
        "thresholds_digest": "thr03",
        "threshold_metadata_digest": "meta03",
        "ablation_digest": "abl03",
        "attack_protocol_digest": "attack_digest_03",
        "impl_digest": "impl03",
        "fusion_rule_version": "fusion_v1",
        "attack_protocol_version": "attack_v1",
        "policy_path": "content_np_geo_rescue",
        "status": "ok",
    }
    metrics_row: Dict[str, Any] = {
        "grid_item_digest": "grid01",
        "status": "ok",
        "evaluation_scope": primary_scope,
        "primary_metric_name": primary_metric_name,
        "primary_driver_mode": primary_driver_mode,
        "primary_status_source": primary_status_source,
        "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
        "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
        "primary_summary_basis_scope": primary_summary_basis_scope,
        "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
        "auxiliary_scope_metrics": {
            "content_chain": {"metric_name": "content_chain_score", "available": True},
            "lf_channel": {"metric_name": "lf_channel_score", "available": True},
        },
    }
    if include_formal_final_decision_metrics:
        metrics_row[FORMAL_FINAL_DECISION_METRIC_NAME] = {
            "scope": "system_final",
            "decision_status": "accepted",
            "formal_pass": True,
        }
    if include_derived_system_union_metrics:
        metrics_row[DERIVED_SYSTEM_UNION_METRIC_NAME] = {
            "scope": "system_final",
            "system_tpr": 1.0,
            "system_fpr": 0.0,
        }
    if include_system_final_metrics:
        derived_alias = metrics_row.get(DERIVED_SYSTEM_UNION_METRIC_NAME)
        metrics_row[SYSTEM_FINAL_METRIC_NAME] = dict(derived_alias) if isinstance(derived_alias, dict) else {
            "scope": "system_final",
            "system_tpr": 1.0,
            "system_fpr": 0.0,
        }

    grid_result_row: Dict[str, Any] = {
        "grid_item_digest": "grid01",
        "status": "ok",
        "evaluation_scope": primary_scope,
        "auxiliary_scopes": auxiliary_scopes,
        "scope_manifest": scope_manifest,
        "primary_metric_name": primary_metric_name,
        "primary_driver_mode": primary_driver_mode,
        "primary_status_source": primary_status_source,
        "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
        "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
        "primary_summary_basis_scope": primary_summary_basis_scope,
        "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
        "cfg_digest": "cfg03",
        "plan_digest": "plan03",
        "thresholds_digest": "thr03",
        "threshold_metadata_digest": "meta03",
        "ablation_digest": "abl03",
        "attack_protocol_digest": "attack_digest_03",
        "attack_protocol_version": "attack_v1",
        "policy_path": "content_np_geo_rescue",
        "impl_digest": "impl03",
        "fusion_rule_version": "fusion_v1",
        "metrics": {
            FORMAL_FINAL_DECISION_METRIC_NAME: metrics_row.get(FORMAL_FINAL_DECISION_METRIC_NAME),
            DERIVED_SYSTEM_UNION_METRIC_NAME: metrics_row.get(DERIVED_SYSTEM_UNION_METRIC_NAME),
            SYSTEM_FINAL_METRIC_NAME: metrics_row.get(SYSTEM_FINAL_METRIC_NAME),
            "auxiliary_scope_metrics": metrics_row["auxiliary_scope_metrics"],
        },
    }
    if include_internal_scalar_driver_residual:
        metrics_row["formal_score_name"] = legacy_scalar_formal_score_name
        grid_result_row["scalar_formal_score_name"] = legacy_scalar_formal_score_name

    files = {
        "artifacts/grid_summary.json": {
            "cfg_digest": "cfg03",
            "thresholds_digest": "thr03",
            "threshold_metadata_digest": "meta03",
            "attack_protocol_version": "attack_v1",
            "attack_protocol_digest": "attack_digest_03",
            "attack_coverage_digest": "coverage03",
            "impl_digest": "impl03",
            "fusion_rule_version": "fusion_v1",
            "policy_path": "content_np_geo_rescue",
            "primary_evaluation_scope": primary_scope,
            "primary_metric_name": primary_metric_name,
            "primary_driver_mode": primary_driver_mode,
            "primary_status_source": primary_status_source,
            "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
            "primary_summary_basis_scope": primary_summary_basis_scope,
            "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
            "auxiliary_scopes": auxiliary_scopes,
            "scope_manifest": scope_manifest,
            "formal_final_decision_metrics_presence": formal_final_decision_metrics_presence,
            "derived_system_union_metrics_presence": derived_system_union_metrics_presence,
            "system_final_metrics_presence": system_final_metrics_presence,
            "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
            "results": [grid_result_row],
        },
        "artifacts/grid_manifest.json": {"grid_manifest_digest": "grid_manifest_03"},
        "artifacts/aggregate_report.json": {
            "aggregate_report_version": "aggregate_v1",
            "primary_evaluation_scope": primary_scope,
            "primary_metric_name": primary_metric_name,
            "primary_driver_mode": primary_driver_mode,
            "primary_status_source": primary_status_source,
            "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
            "primary_summary_basis_scope": primary_summary_basis_scope,
            "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
            "auxiliary_scopes": auxiliary_scopes,
            "scope_manifest": scope_manifest,
            "experiment_matrix_digest": "matrix03",
            "experiment_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "attack_coverage_digest": "coverage03",
            "policy_path": "content_np_geo_rescue",
            "anchors": [anchor_row],
            "metrics_matrix": [metrics_row],
            "formal_final_decision_metrics_presence": formal_final_decision_metrics_presence,
            "derived_system_union_metrics_presence": derived_system_union_metrics_presence,
            "system_final_metrics_presence": system_final_metrics_presence,
            "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
        },
        "artifacts/workflow_summary.json": {
            "stage_name": "03_Experiment_Matrix_Full",
            "stage_run_id": stage_run_id,
            "status": "ok",
            "primary_evaluation_scope": primary_scope,
            "primary_metric_name": primary_metric_name,
            "primary_driver_mode": primary_driver_mode,
            "primary_status_source": primary_status_source,
            "auxiliary_analysis_runtime_executed": auxiliary_analysis_runtime_executed,
            "primary_summary_basis_scope": primary_summary_basis_scope,
            "primary_summary_basis_metric_name": primary_summary_basis_metric_name,
            "auxiliary_scopes": auxiliary_scopes,
            "scope_manifest": scope_manifest,
            "formal_final_decision_metrics_presence": formal_final_decision_metrics_presence,
            "derived_system_union_metrics_presence": derived_system_union_metrics_presence,
            "system_final_metrics_presence": system_final_metrics_presence,
            "system_final_metrics_semantics": SYSTEM_FINAL_METRIC_SEMANTICS,
        },
        "artifacts/run_closure.json": {"status": {"ok": True, "reason": "ok"}},
        "lineage/source_stage_manifest.json": lineage_stage_manifest,
        "lineage/source_package_manifest.json": lineage_package_manifest,
        "global_calibrate/artifacts/thresholds/thresholds_artifact.json": {"threshold_value": 0.7, "thresholds_digest": "thr03"},
        "global_calibrate/artifacts/thresholds/threshold_metadata_artifact.json": {"threshold_metadata_digest": "meta03"},
        "runtime_metadata/runtime_config_snapshot.yaml": "experiment_matrix:\n  allow_failed_semantics_collection: true\n",
    }
    if include_legacy_scalar_contract:
        for payload_name in ("artifacts/grid_summary.json", "artifacts/aggregate_report.json", "artifacts/workflow_summary.json"):
            payload = files[payload_name]
            payload["scalar_formal_scope"] = legacy_scalar_formal_scope
            payload["scalar_formal_score_name"] = legacy_scalar_formal_score_name

    package_path, package_manifest = _create_stage_package(
        base_dir,
        stage_name="03_Experiment_Matrix_Full",
        stage_run_id=stage_run_id,
        stage_manifest=stage_manifest,
        files=files,
        source_stage_run_id=stage_01_info["stage_run_id"],
        source_stage_package_path=str(stage_01_info["package_path"]),
    )
    return {
        "package_path": package_path,
        "package_manifest": package_manifest,
        "stage_manifest": stage_manifest,
        "stage_run_id": stage_run_id,
    }


def test_stage_04_blocks_when_required_stage_02_missing(tmp_path: Path) -> None:
    """
    功能：验证必需 stage 02 缺失时必须 BLOCK_FREEZE。 

    Verify that stage 04 blocks freeze when required stage 02 input is missing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path)

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=None,
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_missing_stage02",
        require_stage_02=True,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_02.required_package_missing" in reason_codes


def test_stage_04_blocks_on_stage_02_lineage_mismatch(tmp_path: Path) -> None:
    """
    功能：验证 stage 02 lineage 不一致时必须 BLOCK_FREEZE。 

    Verify that stage 04 blocks freeze when stage 02 lineage mismatches the provided stage 01 package.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_lineage")
    stage_02_info = _build_stage_02_package(tmp_path / "case_lineage", stage_01_info, mismatch_lineage=True)

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_lineage_mismatch",
        require_stage_02=True,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    assert signoff_report["paper_closure_status"] == "blocked"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_02.lineage_mismatch" in reason_codes
    assert signoff_report["lineage_resolution_summary"]["lineage_match_status"] == "blocked"
    assert signoff_report["lineage_resolution_summary"]["stage_02_lineage_status"]["status"] == "blocked"


def test_stage_04_allows_freeze_when_all_required_stages_align(tmp_path: Path) -> None:
    """
    功能：验证所有必需 stage package 一致时允许 ALLOW_FREEZE。 

    Verify that stage 04 allows freeze when all required stage packages align.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_allow")
    stage_02_info = _build_stage_02_package(tmp_path / "case_allow", stage_01_info)
    stage_03_info = _build_stage_03_package(tmp_path / "case_allow", stage_01_info)

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_allow",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    release_manifest = json.loads(Path(summary["release_manifest_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "ALLOW_FREEZE"
    assert signoff_report["signoff_status"] == "passed"
    assert signoff_report["release_status"] == "passed"
    assert signoff_report["paper_closure_status"] == "passed"
    assert signoff_report["blocking_reasons"] == []
    assert signoff_report["formal_input_summary"]["rejected_inputs"] == []
    assert len(signoff_report["formal_input_summary"]["accepted_formal_inputs"]) == 3
    assert [item["stage_key"] for item in signoff_report["formal_input_summary"]["required_inputs"]] == [
        "stage_01",
        "stage_02",
        "stage_03",
    ]
    assert signoff_report["checked_packages_summary"]["stage_01"]["package_role"] == "formal_stage_package"
    assert signoff_report["checked_packages_summary"]["stage_02"]["input_status"] == "prepared"
    assert signoff_report["checked_packages_summary"]["stage_03"]["input_status"] == "prepared"
    assert signoff_report["checked_packages_summary"]["stage_01"]["formal_gate_summary"]["canonical_source_pool_status"] == "passed"
    assert signoff_report["checked_packages_summary"]["stage_01"]["formal_gate_summary"]["attestation_evidence_status"] == "ok"
    assert signoff_report["lineage_resolution_summary"]["lineage_match_status"] == "passed"
    assert signoff_report["lineage_resolution_summary"]["stage_01_identity_summary"]["stage_name"] == "01_Paper_Full_Cuda"
    assert signoff_report["lineage_resolution_summary"]["stage_01_identity_summary"]["stage_run_id"] == "stage01_run"
    assert signoff_report["lineage_resolution_summary"]["stage_01_identity_summary"]["package_role"] == "formal_stage_package"
    assert Path(summary["package_path"]).exists()
    assert release_manifest["decision"] == "ALLOW_FREEZE"
    assert release_manifest["signoff_status"] == "passed"
    assert summary["signoff_status"] == "passed"
    assert summary["release_status"] == "passed"
    assert summary["paper_closure_status"] == "passed"


def test_stage_04_does_not_require_representative_root_records(tmp_path: Path) -> None:
    """
    功能：验证 stage 04 不再把 representative root records 当作 formal hard dependency。

    Verify that stage 04 no longer treats representative root record files as a
    formal hard dependency when the canonical source pool, attestation status,
    and lineage remain complete.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(
        tmp_path / "case_optional_root",
        include_root_records=False,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=None,
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_optional_root",
        require_stage_02=False,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    stage_01_summary = signoff_report["checked_packages_summary"]["stage_01"]["formal_gate_summary"]

    assert signoff_report["decision"] == "ALLOW_FREEZE"
    assert stage_01_summary["status"] == "passed"
    assert stage_01_summary["canonical_source_pool_status"] == "passed"
    assert stage_01_summary["attestation_evidence_status"] == "ok"
    assert stage_01_summary["representative_root_view_status"] == "optional_missing"
    assert stage_01_summary["representative_root_present"] is False
    assert stage_01_summary["representative_root_contract_mode"] == "compatibility_view"


def test_stage_04_rejects_diagnostics_package_as_formal_input(tmp_path: Path) -> None:
    """
    功能：验证 diagnostics package 不得冒充 formal stage 04 输入。 

    Verify that a diagnostics ZIP cannot pass the stage-04 formal input gate.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_diagnostics_reject")
    diagnostics_package_path = _build_failure_diagnostics_package(
        tmp_path / "case_diagnostics_reject",
        stage_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_diag",
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=diagnostics_package_path,
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_diagnostics_reject",
        require_stage_02=True,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["signoff_status"] == "blocked"
    reason_codes = {item["reason_code"] for item in signoff_report["block_reasons"]}
    assert "stage_02.diagnostics_package_rejected" in reason_codes
    rejected_inputs = signoff_report["formal_input_summary"]["rejected_inputs"]
    assert any(item["stage_key"] == "stage_02" for item in rejected_inputs)
    assert signoff_report["checked_packages_summary"]["stage_02"]["package_role"] == "failure_diagnostics_package"
    assert signoff_report["checked_packages_summary"]["stage_02"]["input_status"] == "rejected_non_formal"


def test_stage_04_blocks_when_stage_01_formal_status_is_failed(tmp_path: Path) -> None:
    """
    功能：验证 stage 01 formal package status 失败时必须阻断。 

    Verify that stage 04 blocks when stage 01 advertises a failed formal-package status.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(
        tmp_path / "case_stage01_formal_failed",
        formal_package_status="failed",
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=None,
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_stage01_formal_failed",
        require_stage_02=False,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["paper_closure_status"] == "blocked"
    reason_codes = {item["reason_code"] for item in signoff_report["block_reasons"]}
    assert "stage_01.formal_package_status_not_success" in reason_codes


def test_stage_04_blocks_when_stage_01_attestation_evidence_status_is_failed(tmp_path: Path) -> None:
    """
    功能：验证 stage 01 attestation evidence 失败时必须阻断。 

    Verify that stage 04 blocks when stage 01 attestation evidence is not successful.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(
        tmp_path / "case_stage01_attestation_failed",
        attestation_evidence_status="failed",
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=None,
        stage_03_package_path=None,
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_stage01_attestation_failed",
        require_stage_02=False,
        require_stage_03=False,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["signoff_status"] == "blocked"
    reason_codes = {item["reason_code"] for item in signoff_report["block_reasons"]}
    assert "stage_01.attestation_evidence_status_not_success" in reason_codes


def test_stage_04_blocks_when_stage_03_lineage_mismatch_breaks_closure(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 指向错误 stage 01 来源时必须阻断闭环。 

    Verify that stage 04 blocks when stage 03 points to the wrong stage-01 source package.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_stage03_lineage_mismatch")
    stage_02_info = _build_stage_02_package(tmp_path / "case_stage03_lineage_mismatch", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_stage03_lineage_mismatch",
        stage_01_info,
        mismatch_lineage=True,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_stage03_lineage_mismatch",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["signoff_status"] == "blocked"
    assert signoff_report["lineage_resolution_summary"]["lineage_match_status"] == "blocked"
    assert signoff_report["lineage_resolution_summary"]["stage_03_lineage_status"]["status"] == "blocked"
    reason_codes = {item["reason_code"] for item in signoff_report["block_reasons"]}
    assert "stage_03.lineage_mismatch" in reason_codes


def test_stage_04_blocks_when_stage_03_primary_scope_is_not_system_final(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 若主评估作用域不是 system_final，则必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 primary_evaluation_scope
    drifts away from system_final.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_scope_mismatch")
    stage_02_info = _build_stage_02_package(tmp_path / "case_scope_mismatch", stage_01_info)
    stage_03_info = _build_stage_03_package(tmp_path / "case_scope_mismatch", stage_01_info, primary_scope="lf_channel")

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_scope_mismatch",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.primary_scope_not_system_final" in reason_codes


def test_stage_04_blocks_when_stage_03_primary_metric_is_legacy_system_final_alias(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 若仍把 system_final_metrics 当作 formal 主指标，则必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 still promotes the
    deprecated system_final_metrics alias as the formal primary metric.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_primary_metric_legacy_alias")
    stage_02_info = _build_stage_02_package(tmp_path / "case_primary_metric_legacy_alias", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_primary_metric_legacy_alias",
        stage_01_info,
        primary_metric_name=SYSTEM_FINAL_METRIC_NAME,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_primary_metric_legacy_alias",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.primary_metric_not_formal_final_decision_metrics" in reason_codes


def test_stage_04_blocks_when_stage_03_primary_driver_mode_is_not_formal_final_decision_only(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 若主驱动模式不是 formal_final_decision_only，则必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 primary_driver_mode
    drifts away from formal_final_decision_only.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_driver_mode_mismatch")
    stage_02_info = _build_stage_02_package(tmp_path / "case_driver_mode_mismatch", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_driver_mode_mismatch",
        stage_01_info,
        primary_driver_mode="lf_channel_primary",
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_driver_mode_mismatch",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.primary_driver_mode_not_formal_final_decision_only" in reason_codes


def test_stage_04_blocks_when_stage_03_lacks_system_final_metrics(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 缺少 system_final_metrics 时必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 does not carry
    dict-valued system_final_metrics on successful rows.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_missing_system_final")
    stage_02_info = _build_stage_02_package(tmp_path / "case_missing_system_final", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_missing_system_final",
        stage_01_info,
        include_system_final_metrics=False,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_missing_system_final",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.system_final_metrics_missing" in reason_codes or "stage_03.metrics_matrix_system_final_rows_missing" in reason_codes


def test_stage_04_blocks_when_stage_03_legacy_scalar_contract_is_still_present(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 若仍输出 legacy scalar formal 顶层合同，则必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 still exposes legacy
    scalar-formal top-level contract fields.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_mixed_primary_scalar")
    stage_02_info = _build_stage_02_package(tmp_path / "case_mixed_primary_scalar", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_mixed_primary_scalar",
        stage_01_info,
        include_legacy_scalar_contract=True,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_primary_scalar_mixed",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.aggregate_report_legacy_scalar_contract_present" in reason_codes


def test_stage_04_blocks_when_stage_03_internal_scalar_primary_driver_residual_exists(tmp_path: Path) -> None:
    """
    功能：验证 stage 03 若仅在内部结果残留 scalar 主驱动字段，也必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 keeps scalar-primary
    driver residuals inside nested results even if the top-level contract is clean.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_internal_scalar_residual")
    stage_02_info = _build_stage_02_package(tmp_path / "case_internal_scalar_residual", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_internal_scalar_residual",
        stage_01_info,
        include_internal_scalar_driver_residual=True,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_internal_scalar_residual",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.aggregate_report_internal_scalar_primary_driver_residual" in reason_codes


def test_stage_04_blocks_when_stage_03_primary_status_source_is_not_formal_final_decision_metrics(tmp_path: Path) -> None:
    """
    功能：验证即使没有显式 forbidden scalar 字段，只要主状态来源不是 formal_final_decision_metrics 也必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when stage 03 still exposes a scalar-first
    primary status source even without legacy scalar field names.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_primary_status_source_residual")
    stage_02_info = _build_stage_02_package(tmp_path / "case_primary_status_source_residual", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_primary_status_source_residual",
        stage_01_info,
        primary_status_source="auxiliary_analysis",
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_primary_status_source_residual",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.primary_status_source_not_formal_final_decision_metrics" in reason_codes


def test_stage_04_blocks_when_stage_03_auxiliary_runtime_was_executed(tmp_path: Path) -> None:
    """
    功能：验证 stage 04 看到 stage 03 显式声明 auxiliary runtime 已执行时必须 BLOCK_FREEZE。

    Verify that stage 04 blocks freeze when the stage 03 package explicitly
    declares auxiliary_analysis_runtime_executed as true.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    module = _load_stage_04_module()
    drive_project_root = tmp_path / "drive_project_root"
    stage_01_info = _build_stage_01_package(tmp_path / "case_auxiliary_runtime_executed")
    stage_02_info = _build_stage_02_package(tmp_path / "case_auxiliary_runtime_executed", stage_01_info)
    stage_03_info = _build_stage_03_package(
        tmp_path / "case_auxiliary_runtime_executed",
        stage_01_info,
        auxiliary_analysis_runtime_executed=True,
    )

    summary = module.run_stage_04(
        drive_project_root=drive_project_root,
        stage_01_package_path=stage_01_info["package_path"],
        stage_02_package_path=stage_02_info["package_path"],
        stage_03_package_path=stage_03_info["package_path"],
        config_path=DEFAULT_CONFIG_PATH,
        notebook_name="04_Release_And_Signoff",
        stage_run_id="stage04_auxiliary_runtime_executed",
        require_stage_02=True,
        require_stage_03=True,
    )

    signoff_report = json.loads(Path(summary["signoff_report_path"]).read_text(encoding="utf-8"))
    assert signoff_report["decision"] == "BLOCK_FREEZE"
    reason_codes = {item["reason_code"] for item in signoff_report["blocking_reasons"]}
    assert "stage_03.aggregate_report_auxiliary_analysis_runtime_executed" in reason_codes