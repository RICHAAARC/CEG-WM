"""
文件目的：01_Paper_Full_Cuda notebook 正式阶段编排入口。
Module type: General module

职责边界：
1. 仅负责 Drive 阶段目录布局、formal preflight、主链脚本调用、manifest 与 package 写盘。
2. 不自动触发 02 或 03，不引入 archive 目录运行依赖。
3. 所有运行期配置副本均由 configs/default.yaml 深拷贝得到，不回写仓库配置。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    STAGE_01_NAME,
    build_stage_package_filename,
    collect_attestation_env_summary,
    collect_cuda_summary,
    collect_file_index,
    collect_git_summary,
    collect_model_summary,
    collect_python_summary,
    collect_weight_summary,
    compute_file_sha256,
    copy_prompt_snapshot,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    relative_path_under_base,
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
RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda.py")
PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"
STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH = "artifacts/stage_01_pooled_threshold_build_contract.json"


def _load_json_object(path_obj: Path, label: str) -> Dict[str, Any]:
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return payload


def _required_stage_outputs(run_root: Path) -> Dict[str, Path]:
    return {
        "embed_record": run_root / "records" / "embed_record.json",
        "detect_record": run_root / "records" / "detect_record.json",
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "threshold_metadata_artifact": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "parallel_attestation_statistics_input_contract": run_root / PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH,
        "stage_01_pooled_threshold_build_contract": run_root / STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "run_closure": run_root / "artifacts" / "run_closure.json",
        "workflow_summary": run_root / "artifacts" / "workflow_summary.json",
    }


def _package_stage_outputs(
    run_root: Path,
    runtime_state_root: Path,
    stage_manifest_path: Path,
    runtime_config_snapshot_path: Path,
    source_contract_payload: Dict[str, Any],
    pooled_threshold_build_contract_payload: Dict[str, Any],
) -> Path:
    package_root = ensure_directory(runtime_state_root / "package_staging")
    for relative_path in [
        "records/embed_record.json",
        "records/detect_record.json",
        "records/calibration_record.json",
        "records/evaluate_record.json",
        "artifacts/thresholds/thresholds_artifact.json",
        "artifacts/thresholds/threshold_metadata_artifact.json",
        "artifacts/evaluation_report.json",
        PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH,
        STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "artifacts/run_closure.json",
        "artifacts/workflow_summary.json",
        "artifacts/stage_manifest.json",
        "runtime_metadata/runtime_config_snapshot.yaml",
    ]:
        source_path = run_root / relative_path
        if relative_path == "artifacts/stage_manifest.json":
            source_path = stage_manifest_path
        elif relative_path == "runtime_metadata/runtime_config_snapshot.yaml":
            source_path = runtime_config_snapshot_path
        stage_relative_copy(source_path, package_root, relative_path)

    copied_paths: set[str] = set()
    for contract_payload in [source_contract_payload, pooled_threshold_build_contract_payload]:
        contract_records = contract_payload.get("records")
        if not isinstance(contract_records, list):
            continue
        for record_entry in contract_records:
            if not isinstance(record_entry, dict):
                raise ValueError("stage 01 contract records must be objects")
            source_path_value = record_entry.get("staged_path") or record_entry.get("path")
            package_relative_path = record_entry.get("package_relative_path")
            if not isinstance(source_path_value, str) or not source_path_value:
                raise ValueError("stage 01 contract record missing staged_path/path")
            if not isinstance(package_relative_path, str) or not package_relative_path:
                raise ValueError("stage 01 contract record missing package_relative_path")
            if package_relative_path in copied_paths:
                continue
            stage_relative_copy(Path(source_path_value), package_root, package_relative_path)
            copied_paths.add(package_relative_path)
    return package_root


def _read_log_tail(path_value: Any, max_lines: int = 20) -> list[str]:
    """
    功能：读取日志尾部用于 notebook 失败诊断。

    Read the tail lines from one stage log for notebook diagnostics.

    Args:
        path_value: Log path value.
        max_lines: Maximum number of lines to keep.

    Returns:
        Tail lines in original order.
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be positive int")
    if not isinstance(path_value, (str, Path)):
        return []
    path_obj = Path(path_value)
    if not path_obj.exists() or not path_obj.is_file():
        return []
    lines = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _build_runner_failure_payload(
    runner_result: Dict[str, Any],
    run_root: Path,
    log_root: Path,
    stage_run_id: str,
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 主链失败诊断载荷。

    Build the failure payload for the stage-01 mainline runner.

    Args:
        runner_result: Runner execution result mapping.
        run_root: Stage run root.
        log_root: Stage log root.
        stage_run_id: Stage run identifier.

    Returns:
        JSON-serializable failure payload.
    """
    if not isinstance(runner_result, dict):
        raise TypeError("runner_result must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(log_root, Path):
        raise TypeError("log_root must be Path")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")

    stdout_log_path = runner_result.get("stdout_log_path")
    stderr_log_path = runner_result.get("stderr_log_path")
    return {
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "return_code": int(runner_result.get("return_code", 1)),
        "command": runner_result.get("command"),
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
        "stdout_tail": _read_log_tail(stdout_log_path),
        "stderr_tail": _read_log_tail(stderr_log_path),
    }


def run_stage_01(
    *,
    drive_project_root: Path,
    config_path: Path,
    notebook_name: str,
    stage_run_id: str,
) -> Dict[str, Any]:
    stage_roots = resolve_stage_roots(drive_project_root, STAGE_01_NAME, stage_run_id)
    run_root = ensure_directory(stage_roots["run_root"])
    log_root = ensure_directory(stage_roots["log_root"])
    runtime_state_root = ensure_directory(stage_roots["runtime_state_root"])
    export_root = ensure_directory(stage_roots["export_root"])
    for protected_path in (run_root, log_root, runtime_state_root, export_root):
        validate_path_within_base(drive_project_root, protected_path, "stage path")

    cfg_obj = load_yaml_mapping(config_path)
    runtime_config_snapshot_path = runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    write_yaml_mapping(runtime_config_snapshot_path, cfg_obj)
    prompt_snapshot = copy_prompt_snapshot(REPO_ROOT, cfg_obj, runtime_state_root / "runtime_metadata" / "prompt_snapshot")

    preflight = detect_formal_gpu_preflight(runtime_config_snapshot_path)
    if not bool(preflight.get("ok", False)):
        raise RuntimeError(f"formal GPU preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

    runner_command = [
        sys.executable,
        str((REPO_ROOT / RUNNER_SCRIPT_PATH).resolve()),
        "--config",
        str(runtime_config_snapshot_path),
        "--run-root",
        str(run_root),
        "--stage-run-id",
        stage_run_id,
    ]
    runner_result = run_command_with_logs(
        command=runner_command,
        cwd=REPO_ROOT,
        stdout_log_path=log_root / "01_mainline_stdout.log",
        stderr_log_path=log_root / "01_mainline_stderr.log",
    )
    if runner_result["return_code"] != 0:
        failure_payload = _build_runner_failure_payload(
            runner_result,
            run_root,
            log_root,
            stage_run_id,
        )
        raise RuntimeError(
            "stage 01 mainline failed: "
            f"{json.dumps(failure_payload, ensure_ascii=False, sort_keys=True)}"
        )

    outputs = _required_stage_outputs(run_root)
    missing_outputs = [label for label, path_obj in outputs.items() if not path_obj.exists()]
    if missing_outputs:
        raise FileNotFoundError(f"stage 01 required outputs missing: {missing_outputs}")

    stats_contract_path = outputs["parallel_attestation_statistics_input_contract"]
    stats_contract_payload = _load_json_object(
        stats_contract_path,
        "stage 01 parallel_attestation_statistics_input_contract",
    )
    pooled_threshold_build_contract_path = outputs["stage_01_pooled_threshold_build_contract"]
    pooled_threshold_build_contract_payload = _load_json_object(
        pooled_threshold_build_contract_path,
        "stage 01 pooled threshold build contract",
    )

    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"
    stage_manifest: Dict[str, Any] = {
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_name": None,
        "source_stage_run_id": None,
        "config_source_path": normalize_path_value(config_path),
        "config_source_repo_relative": relative_path_under_base(REPO_ROOT, config_path),
        "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
        "prompt_snapshot_path": prompt_snapshot.get("snapshot_path"),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "logs_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "exports_root": normalize_path_value(export_root),
        "records": collect_file_index(run_root, {
            "embed_record": outputs["embed_record"],
            "detect_record": outputs["detect_record"],
            "calibration_record": outputs["calibration_record"],
            "evaluate_record": outputs["evaluate_record"],
        }),
        "parallel_attestation_statistics_input_contract_path": normalize_path_value(stats_contract_path),
        "parallel_attestation_statistics_input_contract_package_relative_path": PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH,
        "parallel_attestation_statistics_input_contract_role": stats_contract_payload["contract_role"],
        "parallel_attestation_statistics_input_contract_status": stats_contract_payload["status"],
        "parallel_attestation_statistics_input_contract_reason": stats_contract_payload["reason"],
        "parallel_attestation_statistics_input_contract_source_records_available": stats_contract_payload["source_records_available"],
        "parallel_attestation_statistics_input_contract_direct_stats_ready": stats_contract_payload["direct_stats_ready"],
        "parallel_attestation_statistics_input_contract_direct_stats_reason": stats_contract_payload["direct_stats_reason"],
        "parallel_attestation_statistics_input_contract_record_count": stats_contract_payload["record_count"],
        "parallel_attestation_statistics_input_contract_score_availability": stats_contract_payload.get("score_availability", {}),
        "stage_01_pooled_threshold_build_contract_path": normalize_path_value(pooled_threshold_build_contract_path),
        "stage_01_pooled_threshold_build_contract_package_relative_path": STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "stage_01_pooled_threshold_build_mode": pooled_threshold_build_contract_payload.get("build_mode"),
        "stage_01_pooled_threshold_requested_build_mode": pooled_threshold_build_contract_payload.get("requested_build_mode"),
        "stage_01_pooled_threshold_direct_record_count": pooled_threshold_build_contract_payload.get("direct_record_count"),
        "stage_01_pooled_threshold_derived_record_count": pooled_threshold_build_contract_payload.get("derived_record_count"),
        "stage_01_pooled_threshold_final_record_count": pooled_threshold_build_contract_payload.get("final_record_count"),
        "stage_01_pooled_threshold_final_label_balanced": pooled_threshold_build_contract_payload.get("final_label_balanced"),
        "stage_01_pooled_threshold_stats_input_set": pooled_threshold_build_contract_payload.get("stats_input_set", {}),
        "thresholds_path": normalize_path_value(outputs["thresholds_artifact"]),
        "threshold_metadata_artifact_path": normalize_path_value(outputs["threshold_metadata_artifact"]),
        "evaluation_report_path": normalize_path_value(outputs["evaluation_report"]),
        "run_closure_path": normalize_path_value(outputs["run_closure"]),
        "workflow_summary_path": normalize_path_value(outputs["workflow_summary"]),
        "prompt_file_path": prompt_snapshot.get("source_path"),
        "prompt_snapshot": prompt_snapshot,
        "notebook_name": notebook_name,
        "git": collect_git_summary(REPO_ROOT),
        "python": collect_python_summary(),
        "cuda": collect_cuda_summary(),
        "attestation_env": collect_attestation_env_summary(cfg_obj),
        "model_summary": collect_model_summary(cfg_obj),
        "weight_summary": collect_weight_summary(REPO_ROOT, cfg_obj),
        "created_at": utc_now_iso(),
        "runner_result": runner_result,
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    package_root = _package_stage_outputs(
        run_root,
        runtime_state_root,
        stage_manifest_path,
        runtime_config_snapshot_path,
        stats_contract_payload,
        pooled_threshold_build_contract_payload,
    )
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name=STAGE_01_NAME,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=None,
        source_stage_package_path=None,
        package_manifest_path=package_manifest_path,
    )

    summary: Dict[str, Any] = {
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "package_filename": build_stage_package_filename(STAGE_01_NAME, stage_run_id),
        "package_path": package_manifest["package_path"],
        "package_sha256": package_manifest["package_sha256"],
        "status": "ok",
    }
    write_json_atomic(runtime_state_root / "stage_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stage-01 Paper_Full_Cuda notebook orchestration.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Source config path.")
    parser.add_argument("--notebook-name", default=STAGE_01_NAME, help="Notebook display name.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    args = parser.parse_args()

    summary = run_stage_01(
        drive_project_root=resolve_repo_path(args.drive_project_root),
        config_path=resolve_repo_path(args.config),
        notebook_name=str(args.notebook_name),
        stage_run_id=args.stage_run_id or make_stage_run_id(STAGE_01_NAME),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
