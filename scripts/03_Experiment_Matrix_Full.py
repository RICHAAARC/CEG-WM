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
    prepare_source_package,
    read_json_dict,
    resolve_repo_path,
    resolve_stage_roots,
    run_command_with_logs,
    stage_relative_copy,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)
from scripts.workflow_acceptance_common import detect_formal_gpu_preflight


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _resolve_source_lineage_paths(extracted_root: Path) -> Dict[str, Path]:
    return {
        "source_stage_manifest_path": extracted_root / "artifacts" / "stage_manifest.json",
        "source_package_manifest_path": extracted_root / "artifacts" / "package_manifest.json",
        "source_runtime_config_snapshot_path": extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml",
        "source_thresholds_artifact_path": extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
    }


def _resolve_prompt_snapshot_path(extracted_root: Path) -> str:
    prompt_root = extracted_root / "runtime_metadata" / "prompt_snapshot"
    if prompt_root.exists() and prompt_root.is_dir():
        for prompt_path in sorted(prompt_root.rglob("*")):
            if prompt_path.is_file():
                return normalize_path_value(prompt_path)
    return "<absent>"


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


def _build_runtime_config(cfg_obj: Dict[str, Any], run_root: Path, readonly_thresholds_path: Path) -> Dict[str, Any]:
    config_copy = json.loads(json.dumps(cfg_obj))
    experiment_cfg = dict(config_copy.get("experiment_matrix")) if isinstance(config_copy.get("experiment_matrix"), dict) else {}
    experiment_cfg["config_path"] = str(DEFAULT_CONFIG_PATH.as_posix())
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
    source_lineage_paths = _resolve_source_lineage_paths(extracted_root)
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
    runtime_cfg = _build_runtime_config(cfg_obj, run_root, readonly_thresholds["thresholds_artifact"])
    write_yaml_mapping(runtime_config_snapshot_path, runtime_cfg)

    preflight = detect_formal_gpu_preflight(runtime_config_snapshot_path)
    if not bool(preflight.get("ok", False)):
        raise RuntimeError(f"formal GPU preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

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
    }
    missing_outputs = [label for label, path_obj in outputs.items() if not path_obj.exists()]
    if missing_outputs:
        raise FileNotFoundError(f"stage 03 required outputs missing: {missing_outputs}")

    source_stage_manifest_copy_path = runtime_state_root / "lineage" / "source_stage_manifest.json"
    copy_stage_manifest_snapshot(source_manifest, source_stage_manifest_copy_path)
    source_package_manifest_copy_path = runtime_state_root / "lineage" / "source_package_manifest.json"
    write_json_atomic(source_package_manifest_copy_path, package_manifest)
    grid_summary_obj = read_json_dict(outputs["grid_summary"])
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
        "source_prompt_snapshot_path": _resolve_prompt_snapshot_path(extracted_root),
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
