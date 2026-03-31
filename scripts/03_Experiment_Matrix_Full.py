"""
文件目的：03_Experiment_Matrix_Full 独立阶段编排入口。
Module type: General module

职责边界：
1. 仅消费 01 stage package，生成 experiment matrix 运行时配置副本并独立执行矩阵。
2. 通过只读绑定 01 thresholds 工件，避免重跑 01 主链或覆盖 01 正式 records。
3. 输出独立 stage_manifest 与 package，并保留 source package lineage。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    STAGE_03_NAME,
    apply_notebook_model_snapshot_binding,
    collect_attestation_env_summary,
    collect_cuda_summary,
    collect_file_index,
    collect_git_summary,
    collect_model_summary,
    collect_python_summary,
    collect_weight_summary,
    copy_file,
    copy_stage_manifest_snapshot,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    persist_source_package_lineage,
    prepare_source_package,
    read_json_dict,
    resolve_repo_path,
    resolve_model_identity,
    resolve_source_lineage_paths,
    resolve_source_prompt_snapshot_path,
    resolve_stage_roots,
    run_command_with_logs,
    stage_relative_copy,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)
from scripts.workflow_acceptance_common import detect_stage_03_preflight


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
_AUXILIARY_ANALYSIS_RUNTIME_EVIDENCE_FIELD = "auxiliary_analysis_runtime_executed"


def _copy_readonly_thresholds(extracted_root: Path, run_root: Path) -> Dict[str, Path]:
    thresholds_root = ensure_directory(run_root / "global_calibrate" / "artifacts" / "thresholds")
    source_thresholds_path = extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    source_threshold_metadata_path = extracted_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json"
    if not source_thresholds_path.exists():
        raise FileNotFoundError(f"source thresholds artifact missing: {source_thresholds_path}")
    copied_thresholds_path = copy_file(source_thresholds_path, thresholds_root / "thresholds_artifact.json")
    copied_threshold_metadata_path = (
        copy_file(source_threshold_metadata_path, thresholds_root / "threshold_metadata_artifact.json")
        if source_threshold_metadata_path.exists()
        else thresholds_root / "threshold_metadata_artifact.json"
    )
    return {
        "thresholds_artifact": copied_thresholds_path,
        "threshold_metadata_artifact": copied_threshold_metadata_path,
    }


def _build_stage_03_model_binding_summary(
    cfg_obj: Dict[str, Any],
    *,
    binding_source: str = "<absent>",
    binding_env_var: str = "<absent>",
    binding_status: str = "absent",
    binding_reason: str = "stage_03_model_snapshot_binding_absent",
    model_snapshot_path: str = "<absent>",
) -> Dict[str, Any]:
    """
    功能：构造 stage 03 运行时模型绑定摘要。

    Build the normalized stage-03 model snapshot binding summary.

    Args:
        cfg_obj: Runtime configuration mapping.
        binding_source: Binding source label.
        binding_env_var: Environment-variable anchor when available.
        binding_status: Binding status string.
        binding_reason: Binding reason string.
        model_snapshot_path: Bound snapshot directory path.

    Returns:
        Model binding summary mapping.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    model_identity = resolve_model_identity(cfg_obj)
    requested_model_source = (
        str(cfg_obj.get("model_source")).strip()
        if isinstance(cfg_obj.get("model_source"), str) and str(cfg_obj.get("model_source")).strip()
        else "<absent>"
    )
    return {
        "binding_source": binding_source,
        "binding_env_var": binding_env_var,
        "binding_status": binding_status,
        "binding_reason": binding_reason,
        "model_snapshot_path": model_snapshot_path,
        "requested_model_id": model_identity["model_id"],
        "requested_model_source": requested_model_source,
        "requested_hf_revision": model_identity["revision"],
    }


def _normalize_stage_03_model_binding(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：规范化 stage 03 运行时配置中的模型快照绑定字段。

    Normalize the stage-03 runtime config so model_snapshot_path and
    model_source_binding are always explicit and semantically consistent.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Normalized runtime configuration copy.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    cfg_copy = json.loads(json.dumps(cfg_obj))
    binding_obj = cfg_copy.get("model_source_binding") if isinstance(cfg_copy.get("model_source_binding"), dict) else {}
    binding_summary = _build_stage_03_model_binding_summary(cfg_copy)
    for field_name in (
        "binding_source",
        "binding_env_var",
        "binding_status",
        "binding_reason",
        "requested_model_id",
        "requested_model_source",
        "requested_hf_revision",
    ):
        raw_value = binding_obj.get(field_name)
        if isinstance(raw_value, str) and raw_value.strip():
            binding_summary[field_name] = raw_value.strip()

    binding_snapshot_path = normalize_path_value(binding_obj.get("model_snapshot_path"))
    if binding_snapshot_path != "<absent>":
        binding_summary["model_snapshot_path"] = binding_snapshot_path

    runtime_snapshot_path = normalize_path_value(cfg_copy.get("model_snapshot_path"))
    if runtime_snapshot_path == "<absent>" and binding_summary["model_snapshot_path"] != "<absent>":
        runtime_snapshot_path = str(binding_summary["model_snapshot_path"])
    cfg_copy["model_snapshot_path"] = runtime_snapshot_path

    if runtime_snapshot_path == "<absent>":
        if binding_summary["binding_status"] == "bound":
            binding_summary["binding_status"] = "invalid"
            binding_summary["binding_reason"] = "model_snapshot_path_absent"
        binding_summary["model_snapshot_path"] = "<absent>"
        cfg_copy["model_source_binding"] = binding_summary
        return cfg_copy

    snapshot_path_obj = Path(runtime_snapshot_path)
    if not snapshot_path_obj.exists() or not snapshot_path_obj.is_dir():
        if binding_summary["binding_status"] == "bound":
            binding_summary["binding_status"] = "invalid"
            binding_summary["binding_reason"] = "model_snapshot_path_missing_or_not_directory"
        binding_summary["model_snapshot_path"] = runtime_snapshot_path
        cfg_copy["model_source_binding"] = binding_summary
        return cfg_copy

    if binding_summary["binding_status"] == "bound":
        if binding_summary["model_snapshot_path"] == "<absent>":
            binding_summary["model_snapshot_path"] = runtime_snapshot_path
        elif binding_summary["model_snapshot_path"] != runtime_snapshot_path:
            binding_summary["binding_status"] = "invalid"
            binding_summary["binding_reason"] = "model_source_binding_path_mismatch"

    cfg_copy["model_source_binding"] = binding_summary
    return cfg_copy


def _has_valid_stage_03_model_binding(cfg_obj: Dict[str, Any]) -> bool:
    """
    功能：判断 stage 03 运行时配置是否已有有效模型快照绑定。

    Determine whether stage 03 already has one valid authoritative model
    snapshot binding.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        True when the model binding is present, bound, directory-backed, and
        internally path-consistent.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    normalized_cfg = _normalize_stage_03_model_binding(cfg_obj)
    binding_obj = normalized_cfg.get("model_source_binding") if isinstance(normalized_cfg.get("model_source_binding"), dict) else {}
    runtime_snapshot_path = normalize_path_value(normalized_cfg.get("model_snapshot_path"))
    binding_snapshot_path = normalize_path_value(binding_obj.get("model_snapshot_path"))
    return (
        binding_obj.get("binding_status") == "bound"
        and runtime_snapshot_path != "<absent>"
        and Path(runtime_snapshot_path).exists()
        and Path(runtime_snapshot_path).is_dir()
        and binding_snapshot_path == runtime_snapshot_path
    )


def _inherit_source_runtime_model_binding(
    cfg_obj: Dict[str, Any],
    source_runtime_config_snapshot_path: Path,
) -> Dict[str, Any]:
    """
    功能：从 source stage runtime config 中最小继承模型绑定字段。

    Inherit only model_snapshot_path and model_source_binding from the source
    stage runtime config snapshot.

    Args:
        cfg_obj: Current stage-03 runtime configuration mapping.
        source_runtime_config_snapshot_path: Source stage runtime config path.

    Returns:
        Runtime configuration copy after the minimal source-stage binding
        fallback has been applied.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(source_runtime_config_snapshot_path, Path):
        raise TypeError("source_runtime_config_snapshot_path must be Path")

    cfg_copy = json.loads(json.dumps(cfg_obj))
    source_runtime_cfg = load_yaml_mapping(source_runtime_config_snapshot_path)
    if isinstance(source_runtime_cfg.get("model_snapshot_path"), str) and str(source_runtime_cfg.get("model_snapshot_path")).strip():
        cfg_copy["model_snapshot_path"] = str(source_runtime_cfg.get("model_snapshot_path")).strip()
    if isinstance(source_runtime_cfg.get("model_source_binding"), dict):
        cfg_copy["model_source_binding"] = json.loads(json.dumps(source_runtime_cfg["model_source_binding"]))
    return _normalize_stage_03_model_binding(cfg_copy)


def _resolve_stage_03_runtime_config(
    cfg_obj: Dict[str, Any],
    source_runtime_config_snapshot_path: Path,
) -> Dict[str, Any]:
    """
    功能：解析 stage 03 的最终运行时配置并优先保留 notebook 绑定。

    Resolve the authoritative stage-03 runtime config by preferring the
    notebook-provided snapshot binding and falling back to the source-stage
    runtime config only when the notebook binding is absent or invalid.

    Args:
        cfg_obj: Source configuration mapping.
        source_runtime_config_snapshot_path: Source stage runtime config path.

    Returns:
        Stage-03 runtime configuration with explicit model binding fields.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(source_runtime_config_snapshot_path, Path):
        raise TypeError("source_runtime_config_snapshot_path must be Path")

    notebook_bound_cfg = apply_notebook_model_snapshot_binding(cfg_obj)
    if _has_valid_stage_03_model_binding(notebook_bound_cfg):
        return _normalize_stage_03_model_binding(notebook_bound_cfg)
    return _inherit_source_runtime_model_binding(notebook_bound_cfg, source_runtime_config_snapshot_path)


def _build_runtime_config(
    cfg_obj: Dict[str, Any],
    run_root: Path,
    readonly_thresholds_path: Path,
    runtime_config_snapshot_path: Path,
) -> Dict[str, Any]:
    config_copy = _normalize_stage_03_model_binding(cfg_obj)
    experiment_cfg = dict(config_copy.get("experiment_matrix")) if isinstance(config_copy.get("experiment_matrix"), dict) else {}
    experiment_cfg["config_path"] = str(runtime_config_snapshot_path.resolve())
    experiment_cfg["batch_root"] = str(run_root.resolve())
    experiment_cfg["external_shared_thresholds_path"] = str(readonly_thresholds_path.resolve())
    config_copy["experiment_matrix"] = experiment_cfg
    return config_copy


def _package_outputs(run_root: Path, runtime_state_root: Path, stage_manifest_path: Path, runtime_config_snapshot_path: Path, source_stage_manifest_copy_path: Path) -> Path:
    package_root = ensure_directory(runtime_state_root / "package_staging")
    for relative_path, source_path in {
        "artifacts/stage_manifest.json": stage_manifest_path,
        "artifacts/grid_summary.json": run_root / "artifacts" / "grid_summary.json",
        "artifacts/grid_manifest.json": run_root / "artifacts" / "grid_manifest.json",
        "artifacts/aggregate_report.json": run_root / "artifacts" / "aggregate_report.json",
        "artifacts/gpu_memory_profile_breakdown.json": run_root / "artifacts" / "gpu_memory_profile_breakdown.json",
        "artifacts/run_closure.json": run_root / "artifacts" / "run_closure.json",
        "artifacts/workflow_summary.json": run_root / "artifacts" / "workflow_summary.json",
        "global_calibrate/artifacts/thresholds/thresholds_artifact.json": run_root / "global_calibrate" / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "global_calibrate/artifacts/thresholds/threshold_metadata_artifact.json": run_root / "global_calibrate" / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "runtime_metadata/runtime_config_snapshot.yaml": runtime_config_snapshot_path,
        "lineage/source_stage_manifest.json": source_stage_manifest_copy_path,
        "lineage/source_package_manifest.json": runtime_state_root / "lineage" / "source_package_manifest.json",
    }.items():
        if source_path.exists() and source_path.is_file():
            stage_relative_copy(source_path, package_root, relative_path)
    return package_root


def run_stage_03(
    *,
    drive_project_root: Path,
    config_path: Path,
    source_package_path: Path,
    notebook_name: str,
    stage_run_id: str,
) -> Dict[str, Any]:
    stage_roots = resolve_stage_roots(drive_project_root, STAGE_03_NAME, stage_run_id)
    run_root = ensure_directory(stage_roots["run_root"])
    log_root = ensure_directory(stage_roots["log_root"])
    runtime_state_root = ensure_directory(stage_roots["runtime_state_root"])
    export_root = ensure_directory(stage_roots["export_root"])
    for protected_path in (run_root, log_root, runtime_state_root, export_root):
        validate_path_within_base(drive_project_root, protected_path, "stage path")

    source_info = prepare_source_package(source_package_path, runtime_state_root)
    source_manifest = cast(Dict[str, Any], source_info["stage_manifest"])
    if source_manifest.get("stage_name") != "01_Paper_Full_Cuda":
        raise ValueError("stage 03 requires a source package produced by 01_Paper_Full_Cuda")
    extracted_root = Path(str(source_info["extracted_root"]))
    source_lineage_paths = resolve_source_lineage_paths(extracted_root)
    missing_source_lineage = [
        label for label, path_obj in source_lineage_paths.items()
        if label != "source_package_manifest_path" and not path_obj.exists()
    ]
    if missing_source_lineage:
        raise FileNotFoundError(f"stage 03 source lineage files missing: {missing_source_lineage}")
    package_manifest = cast(Dict[str, Any], source_info["package_manifest"])
    if package_manifest.get("stage_name") != "01_Paper_Full_Cuda":
        raise ValueError("stage 03 source package manifest must declare stage_name=01_Paper_Full_Cuda")
    if package_manifest.get("stage_run_id") != source_manifest.get("stage_run_id"):
        raise ValueError("stage 03 source package manifest stage_run_id does not match source stage_manifest")

    readonly_thresholds = _copy_readonly_thresholds(extracted_root, run_root)
    cfg_obj = load_yaml_mapping(config_path)
    runtime_config_snapshot_path = runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    runtime_cfg = _resolve_stage_03_runtime_config(
        cfg_obj,
        source_lineage_paths["source_runtime_config_snapshot_path"],
    )
    runtime_cfg = _build_runtime_config(
        runtime_cfg,
        run_root,
        readonly_thresholds["thresholds_artifact"],
        runtime_config_snapshot_path,
    )
    write_yaml_mapping(runtime_config_snapshot_path, runtime_cfg)

    preflight = detect_stage_03_preflight(
        runtime_config_snapshot_path,
        source_package_path,
        source_lineage_paths["source_thresholds_artifact_path"],
        require_model_binding=True,
        require_authoritative_config_path=True,
    )
    if not bool(preflight.get("ok", False)):
        raise RuntimeError(f"stage 03 preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

    command = [
        sys.executable,
        "-m",
        "main.cli.run_experiment_matrix",
        "--config",
        str(runtime_config_snapshot_path),
        "--strict",
    ]
    runner_result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=log_root / "03_experiment_matrix_stdout.log",
        stderr_log_path=log_root / "03_experiment_matrix_stderr.log",
    )
    if runner_result["return_code"] != 0:
        raise RuntimeError(f"stage 03 experiment matrix failed: return_code={runner_result['return_code']}")

    outputs = {
        "grid_summary": run_root / "artifacts" / "grid_summary.json",
        "grid_manifest": run_root / "artifacts" / "grid_manifest.json",
        "aggregate_report": run_root / "artifacts" / "aggregate_report.json",
        "gpu_memory_profile_breakdown": run_root / "artifacts" / "gpu_memory_profile_breakdown.json",
    }
    missing_outputs = [label for label, path_obj in outputs.items() if not path_obj.exists()]
    if missing_outputs:
        raise FileNotFoundError(f"stage 03 required outputs missing: {missing_outputs}")

    source_lineage_snapshot_paths = persist_source_package_lineage(runtime_state_root, source_info)
    source_stage_manifest_copy_path = source_lineage_snapshot_paths["source_stage_manifest_copy_path"]
    source_package_manifest_copy_path = source_lineage_snapshot_paths["source_package_manifest_copy_path"]
    grid_summary_obj = read_json_dict(outputs["grid_summary"])
    aggregate_report_obj = read_json_dict(outputs["aggregate_report"])
    gpu_memory_summary_obj = (
        cast(Dict[str, Any], aggregate_report_obj["gpu_memory_summary"])
        if isinstance(aggregate_report_obj.get("gpu_memory_summary"), dict)
        else cast(Dict[str, Any], grid_summary_obj["gpu_memory_summary"])
        if isinstance(grid_summary_obj.get("gpu_memory_summary"), dict)
        else {}
    )
    auxiliary_analysis_runtime_executed = bool(
        aggregate_report_obj.get(
            _AUXILIARY_ANALYSIS_RUNTIME_EVIDENCE_FIELD,
            grid_summary_obj.get(_AUXILIARY_ANALYSIS_RUNTIME_EVIDENCE_FIELD, False),
        )
    )
    run_closure_path = run_root / "artifacts" / "run_closure.json"
    if not run_closure_path.exists():
        write_json_atomic(run_closure_path, {
            "stage_name": STAGE_03_NAME,
            "stage_run_id": stage_run_id,
            "source_stage_run_id": source_manifest.get("stage_run_id"),
            "readonly_thresholds_path": normalize_path_value(readonly_thresholds["thresholds_artifact"]),
            "created_at": utc_now_iso(),
        })

    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    write_json_atomic(workflow_summary_path, {
        "stage_name": STAGE_03_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_path": str(source_info["source_package_path"]),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "source_thresholds_artifact_path": normalize_path_value(source_lineage_paths["source_thresholds_artifact_path"]),
        "grid_summary_path": normalize_path_value(outputs["grid_summary"]),
        "aggregate_report_path": normalize_path_value(outputs["aggregate_report"]),
        "primary_evaluation_scope": aggregate_report_obj.get("primary_evaluation_scope", grid_summary_obj.get("primary_evaluation_scope")),
        "primary_metric_name": aggregate_report_obj.get("primary_metric_name", grid_summary_obj.get("primary_metric_name")),
        "primary_driver_mode": aggregate_report_obj.get("primary_driver_mode", grid_summary_obj.get("primary_driver_mode", "system_final_only")),
        "primary_status_source": aggregate_report_obj.get("primary_status_source", grid_summary_obj.get("primary_status_source", "system_final_metrics")),
        "primary_summary_basis_scope": aggregate_report_obj.get("primary_summary_basis_scope", grid_summary_obj.get("primary_summary_basis_scope")),
        "primary_summary_basis_metric_name": aggregate_report_obj.get("primary_summary_basis_metric_name", grid_summary_obj.get("primary_summary_basis_metric_name")),
        "auxiliary_scopes": aggregate_report_obj.get("auxiliary_scopes", grid_summary_obj.get("auxiliary_scopes", [])),
        _AUXILIARY_ANALYSIS_RUNTIME_EVIDENCE_FIELD: auxiliary_analysis_runtime_executed,
        "gpu_memory_summary": gpu_memory_summary_obj,
        "gpu_memory_profile_breakdown_path": normalize_path_value(outputs["gpu_memory_profile_breakdown"]),
        "scope_manifest": aggregate_report_obj.get("scope_manifest", grid_summary_obj.get("scope_manifest", {})),
        "system_final_metrics_presence": aggregate_report_obj.get("system_final_metrics_presence", grid_summary_obj.get("system_final_metrics_presence", {})),
        "created_at": utc_now_iso(),
    })

    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"
    stage_manifest: Dict[str, Any] = {
        "stage_name": STAGE_03_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_name": source_manifest.get("stage_name"),
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "config_source_path": normalize_path_value(config_path),
        "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "logs_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "exports_root": normalize_path_value(export_root),
        "records": collect_file_index(run_root, {}),
        "thresholds_path": normalize_path_value(readonly_thresholds["thresholds_artifact"]),
        "threshold_metadata_artifact_path": normalize_path_value(readonly_thresholds["threshold_metadata_artifact"]),
        "evaluation_report_path": normalize_path_value(outputs["aggregate_report"]),
        "run_closure_path": normalize_path_value(run_closure_path),
        "workflow_summary_path": normalize_path_value(workflow_summary_path),
        "source_package_path": str(source_info["source_package_path"]),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "source_package_manifest_path": normalize_path_value(source_package_manifest_copy_path),
        "source_package_manifest_digest": str(source_info["package_manifest_digest"]),
        "source_stage_manifest_path": normalize_path_value(source_stage_manifest_copy_path),
        "source_runtime_config_snapshot_path": normalize_path_value(source_lineage_paths["source_runtime_config_snapshot_path"]),
        "source_prompt_snapshot_path": resolve_source_prompt_snapshot_path(extracted_root),
        "source_thresholds_artifact_path": normalize_path_value(source_lineage_paths["source_thresholds_artifact_path"]),
        "source_stage_manifest_copy_path": normalize_path_value(source_stage_manifest_copy_path),
        "readonly_shared_thresholds_path": normalize_path_value(readonly_thresholds["thresholds_artifact"]),
        "notebook_name": notebook_name,
        "git": collect_git_summary(REPO_ROOT),
        "python": collect_python_summary(),
        "cuda": collect_cuda_summary(),
        "attestation_env": collect_attestation_env_summary(runtime_cfg),
        "model_summary": collect_model_summary(runtime_cfg),
        "weight_summary": collect_weight_summary(REPO_ROOT, runtime_cfg),
        "created_at": utc_now_iso(),
        "runner_result": runner_result,
        "primary_evaluation_scope": aggregate_report_obj.get("primary_evaluation_scope", grid_summary_obj.get("primary_evaluation_scope")),
        "primary_metric_name": aggregate_report_obj.get("primary_metric_name", grid_summary_obj.get("primary_metric_name")),
        "primary_driver_mode": aggregate_report_obj.get("primary_driver_mode", grid_summary_obj.get("primary_driver_mode", "system_final_only")),
        "primary_status_source": aggregate_report_obj.get("primary_status_source", grid_summary_obj.get("primary_status_source", "system_final_metrics")),
        "primary_summary_basis_scope": aggregate_report_obj.get("primary_summary_basis_scope", grid_summary_obj.get("primary_summary_basis_scope")),
        "primary_summary_basis_metric_name": aggregate_report_obj.get("primary_summary_basis_metric_name", grid_summary_obj.get("primary_summary_basis_metric_name")),
        "auxiliary_scopes": aggregate_report_obj.get("auxiliary_scopes", grid_summary_obj.get("auxiliary_scopes", [])),
        _AUXILIARY_ANALYSIS_RUNTIME_EVIDENCE_FIELD: auxiliary_analysis_runtime_executed,
        "gpu_memory_summary": gpu_memory_summary_obj,
        "gpu_memory_profile_breakdown_path": normalize_path_value(outputs["gpu_memory_profile_breakdown"]),
        "scope_manifest": aggregate_report_obj.get("scope_manifest", grid_summary_obj.get("scope_manifest", {})),
        "system_final_metrics_presence": aggregate_report_obj.get("system_final_metrics_presence", grid_summary_obj.get("system_final_metrics_presence", {})),
        "grid_summary": grid_summary_obj,
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    package_root = _package_outputs(run_root, runtime_state_root, stage_manifest_path, runtime_config_snapshot_path, source_stage_manifest_copy_path)
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name=STAGE_03_NAME,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=str(source_manifest.get("stage_run_id")),
        source_stage_package_path=str(source_info["source_package_path"]),
        package_manifest_path=package_manifest_path,
    )

    summary: Dict[str, Any] = {
        "stage_name": STAGE_03_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_manifest_path": normalize_path_value(source_package_manifest_copy_path),
        "source_package_manifest_digest": str(source_info["package_manifest_digest"]),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "package_path": package_manifest["package_path"],
        "package_sha256": package_manifest["package_sha256"],
        "status": "ok",
    }
    write_json_atomic(runtime_state_root / "stage_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stage-03 detached experiment matrix workflow.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--source-package", required=True, help="Source stage-01 package ZIP.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Source config path.")
    parser.add_argument("--notebook-name", default=STAGE_03_NAME, help="Notebook display name.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    args = parser.parse_args()

    summary = run_stage_03(
        drive_project_root=resolve_repo_path(args.drive_project_root),
        config_path=resolve_repo_path(args.config),
        source_package_path=resolve_repo_path(args.source_package),
        notebook_name=str(args.notebook_name),
        stage_run_id=args.stage_run_id or make_stage_run_id(STAGE_03_NAME),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
