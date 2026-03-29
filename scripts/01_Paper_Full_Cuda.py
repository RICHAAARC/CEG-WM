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
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from scripts.notebook_runtime_common import (
    EXCLUDED_PACKAGE_DISCOVERY_SCOPE,
    FAILURE_DIAGNOSTICS_PACKAGE_ROLE,
    FORMAL_PACKAGE_DISCOVERY_SCOPE,
    FORMAL_STAGE_PACKAGE_ROLE,
    REPO_ROOT,
    STAGE_01_NAME,
    build_failure_diagnostics_filename,
    build_package_index,
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
    create_zip_archive_from_directory,
    ensure_attestation_env_bootstrap,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    read_required_json_dict,
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
from scripts.workflow_acceptance_common import detect_stage_01_preflight


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda.py")
PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"
STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH = "artifacts/stage_01_pooled_threshold_build_contract.json"
CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH = "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"
FAILURE_DIAGNOSTICS_SUMMARY_FILE_NAME = "failure_diagnostics_summary.json"
FAILURE_DIAGNOSTICS_MANIFEST_FILE_NAME = "failure_diagnostics_manifest.json"
FAILURE_DIAGNOSTICS_INDEX_FILE_NAME = "failure_diagnostics_index.json"
MAX_FAILURE_DIAGNOSTIC_LOG_COUNT = 20
MAX_FAILURE_DIAGNOSTIC_LOG_TAIL_LINES = 40
def _load_optional_json_object(path_obj: Path) -> Dict[str, Any]:
    """
    功能：按容错方式读取可选 JSON 对象文件。

    Read an optional JSON object file and return an empty mapping when absent or
    invalid.

    Args:
        path_obj: Candidate JSON path.

    Returns:
        Parsed JSON mapping, or an empty mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _required_stage_outputs(run_root: Path) -> Dict[str, Path]:
    return {
        "embed_record": run_root / "records" / "embed_record.json",
        "detect_record": run_root / "records" / "detect_record.json",
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "threshold_metadata_artifact": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "canonical_source_pool_manifest": run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        "parallel_attestation_statistics_input_contract": run_root / PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH,
        "stage_01_pooled_threshold_build_contract": run_root / STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "run_closure": run_root / "artifacts" / "run_closure.json",
        "workflow_summary": run_root / "artifacts" / "workflow_summary.json",
    }


def _resolve_attestation_evidence_manifest_fields(
    workflow_summary_payload: Dict[str, Any],
    canonical_source_pool_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：为 stage_manifest 解析 attestation evidence 摘要字段。

    Resolve stage-manifest fields derived from the mainline attestation
    post-check summary.

    Args:
        workflow_summary_payload: Mainline workflow summary payload.
        canonical_source_pool_payload: Canonical source-pool manifest payload.

    Returns:
        Stage-manifest fields with legacy fallback semantics.
    """
    if not isinstance(workflow_summary_payload, dict):
        raise TypeError("workflow_summary_payload must be dict")
    if not isinstance(canonical_source_pool_payload, dict):
        raise TypeError("canonical_source_pool_payload must be dict")

    representative_root_records = (
        cast(Dict[str, Any], canonical_source_pool_payload.get("representative_root_records"))
        if isinstance(canonical_source_pool_payload.get("representative_root_records"), dict)
        else {}
    )
    attestation_resolution = workflow_summary_payload.get("attestation_evidence_resolution")
    if not isinstance(attestation_resolution, dict):
        return {
            "attestation_evidence_status": "legacy_unavailable",
            "attestation_evidence_required_entry_count": canonical_source_pool_payload.get("entry_count"),
            "attestation_evidence_checked_entry_count": 0,
            "attestation_evidence_missing_count": 0,
            "attestation_evidence_failing_prompt_indices": [],
            "attestation_evidence_failing_source_entry_paths": [],
            "attestation_evidence_summary_reason": "workflow_summary_attestation_resolution_missing",
            "attestation_evidence_failure_reason": None,
            "attestation_evidence_representative_root_summary": {
                "view_role": representative_root_records.get("view_role"),
                "source_prompt_index": representative_root_records.get("source_prompt_index"),
                "source_entry_package_relative_path": representative_root_records.get(
                    "source_entry_package_relative_path"
                ),
                "resolution_role": "representative_summary_view_only",
            },
        }

    return {
        "attestation_evidence_status": attestation_resolution.get("overall_status"),
        "attestation_evidence_required_entry_count": attestation_resolution.get("required_entry_count"),
        "attestation_evidence_checked_entry_count": attestation_resolution.get("checked_entry_count"),
        "attestation_evidence_missing_count": attestation_resolution.get("missing_evidence_count"),
        "attestation_evidence_failing_prompt_indices": attestation_resolution.get("failing_prompt_indices", []),
        "attestation_evidence_failing_source_entry_paths": attestation_resolution.get(
            "failing_source_entry_paths",
            [],
        ),
        "attestation_evidence_summary_reason": attestation_resolution.get("summary_reason"),
        "attestation_evidence_failure_reason": attestation_resolution.get("failure_reason"),
        "attestation_evidence_representative_root_summary": attestation_resolution.get(
            "representative_root_summary",
            {
                "view_role": representative_root_records.get("view_role"),
                "source_prompt_index": representative_root_records.get("source_prompt_index"),
                "source_entry_package_relative_path": representative_root_records.get(
                    "source_entry_package_relative_path"
                ),
                "resolution_role": "representative_summary_view_only",
            },
        ),
    }


def _package_stage_outputs(
    run_root: Path,
    runtime_state_root: Path,
    stage_manifest_path: Path,
    runtime_config_snapshot_path: Path,
    canonical_source_pool_payload: Dict[str, Any],
    source_contract_payload: Dict[str, Any],
    pooled_threshold_build_contract_payload: Dict[str, Any],
) -> Path:
    package_root = ensure_directory(runtime_state_root / "package_staging")
    static_relative_paths = [
        "records/embed_record.json",
        "records/detect_record.json",
        "records/calibration_record.json",
        "records/evaluate_record.json",
        "artifacts/thresholds/thresholds_artifact.json",
        "artifacts/thresholds/threshold_metadata_artifact.json",
        "artifacts/evaluation_report.json",
        CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH,
        PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH,
        STAGE_01_POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH,
        "artifacts/run_closure.json",
        "artifacts/workflow_summary.json",
        "artifacts/stage_manifest.json",
        "runtime_metadata/runtime_config_snapshot.yaml",
    ]
    for relative_path in static_relative_paths:
        source_path = run_root / relative_path
        if relative_path == "artifacts/stage_manifest.json":
            source_path = stage_manifest_path
        elif relative_path == "runtime_metadata/runtime_config_snapshot.yaml":
            source_path = runtime_config_snapshot_path
        stage_relative_copy(source_path, package_root, relative_path)

    copied_paths: set[str] = set(static_relative_paths)

    def _copy_dynamic_package_path(source_path_value: Any, package_relative_path: Any, label: str) -> None:
        if not isinstance(source_path_value, str) or not source_path_value:
            raise ValueError(f"{label} missing source path")
        if not isinstance(package_relative_path, str) or not package_relative_path:
            raise ValueError(f"{label} missing package_relative_path")
        if package_relative_path in copied_paths:
            return
        stage_relative_copy(Path(source_path_value), package_root, package_relative_path)
        copied_paths.add(package_relative_path)

    def _copy_optional_artifact_view(artifact_view: Any, label: str) -> None:
        if artifact_view is None:
            return
        if not isinstance(artifact_view, dict):
            raise ValueError(f"{label} view must be object")
        exists_value = artifact_view.get("exists")
        if not isinstance(exists_value, bool):
            raise ValueError(f"{label} view missing exists flag")
        if not exists_value:
            return
        _copy_dynamic_package_path(
            artifact_view.get("path"),
            artifact_view.get("package_relative_path"),
            label,
        )

    for contract_payload in [source_contract_payload, pooled_threshold_build_contract_payload]:
        contract_records = contract_payload.get("records")
        if not isinstance(contract_records, list):
            continue
        for record_entry in contract_records:
            if not isinstance(record_entry, dict):
                raise ValueError("stage 01 contract records must be objects")
            _copy_dynamic_package_path(
                record_entry.get("staged_path") or record_entry.get("path"),
                record_entry.get("package_relative_path"),
                "stage 01 contract record",
            )

    canonical_entries = canonical_source_pool_payload.get("entries")
    if not isinstance(canonical_entries, list):
        raise ValueError("canonical source pool entries must be list")
    for canonical_entry in canonical_entries:
        if not isinstance(canonical_entry, dict):
            raise ValueError("canonical source pool entry must be object")
        entry_package_relative_path = canonical_entry.get("source_entry_package_relative_path")
        if not isinstance(entry_package_relative_path, str) or not entry_package_relative_path:
            raise ValueError("canonical source entry missing source_entry_package_relative_path")
        _copy_dynamic_package_path(
            (run_root / entry_package_relative_path).as_posix(),
            entry_package_relative_path,
            "canonical source entry",
        )
        for label, source_key, package_key in [
            ("canonical source embed record", "embed_record_path", "embed_record_package_relative_path"),
            ("canonical source detect record", "detect_record_path", "detect_record_package_relative_path"),
            ("canonical source runtime config", "runtime_config_path", "runtime_config_package_relative_path"),
        ]:
            _copy_dynamic_package_path(canonical_entry.get(source_key), canonical_entry.get(package_key), label)
        for label, artifact_key in [
            ("canonical source attestation statement", "attestation_statement"),
            ("canonical source attestation bundle", "attestation_bundle"),
            ("canonical source attestation result", "attestation_result"),
            ("canonical source image", "source_image"),
        ]:
            _copy_optional_artifact_view(canonical_entry.get(artifact_key), label)
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


def _resolve_failure_diagnostics_paths(runtime_state_root: Path, stage_run_id: str) -> Dict[str, Path]:
    """
    功能：解析 stage 01 failure diagnostics 的固定目录布局。

    Resolve the canonical failure-diagnostics paths for stage 01.

    Args:
        runtime_state_root: Stage runtime-state root.
        stage_run_id: Stage run identifier.

    Returns:
        Failure-diagnostics path mapping.
    """
    if not isinstance(runtime_state_root, Path):
        raise TypeError("runtime_state_root must be Path")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")

    diagnostics_root = runtime_state_root / "failure_diagnostics"
    return {
        "diagnostics_root": diagnostics_root,
        "staging_root": diagnostics_root / "staging",
        "summary_path": diagnostics_root / FAILURE_DIAGNOSTICS_SUMMARY_FILE_NAME,
        "manifest_path": diagnostics_root / FAILURE_DIAGNOSTICS_MANIFEST_FILE_NAME,
        "index_path": diagnostics_root / FAILURE_DIAGNOSTICS_INDEX_FILE_NAME,
        "package_path": diagnostics_root / build_failure_diagnostics_filename(STAGE_01_NAME, stage_run_id),
    }


def _resolve_formal_package_summary(package_manifest_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析正式 package 的摘要字段。

    Resolve formal-package summary fields from an optional package manifest.

    Args:
        package_manifest_payload: Optional package manifest payload.

    Returns:
        Formal-package summary fields.
    """
    if not isinstance(package_manifest_payload, dict) or not package_manifest_payload:
        return {
            "formal_package_status": "not_generated",
            "formal_package_path": "<absent>",
            "formal_package_filename": "<absent>",
            "formal_package_manifest_path": "<absent>",
        }
    return {
        "formal_package_status": "generated",
        "formal_package_path": str(package_manifest_payload.get("package_path", "<absent>")),
        "formal_package_filename": str(package_manifest_payload.get("package_filename", "<absent>")),
        "formal_package_manifest_path": str(package_manifest_payload.get("package_manifest_path", "<absent>")),
    }


def _collect_relevant_log_paths(log_root: Path, run_root: Path) -> List[Path]:
    """
    功能：收集 stage 01 失败诊断相关日志文件。

    Collect log files relevant to stage-01 failure diagnostics.

    Args:
        log_root: Wrapper log root.
        run_root: Stage run root.

    Returns:
        Recent log-file path list ordered by modification time descending.
    """
    if not isinstance(log_root, Path):
        raise TypeError("log_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    collected: Dict[str, Path] = {}
    for search_root in [log_root, run_root]:
        if not search_root.exists() or not search_root.is_dir():
            continue
        for log_path in search_root.rglob("*.log"):
            if log_path.exists() and log_path.is_file():
                collected[normalize_path_value(log_path)] = log_path
    return sorted(
        collected.values(),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:MAX_FAILURE_DIAGNOSTIC_LOG_COUNT]


def _build_log_file_summary(log_path: Path) -> Dict[str, Any]:
    """
    功能：构造单个日志文件的诊断摘要。

    Build the diagnostics summary for one log file.

    Args:
        log_path: Log file path.

    Returns:
        Log summary mapping.
    """
    if not isinstance(log_path, Path):
        raise TypeError("log_path must be Path")
    return {
        "path": normalize_path_value(log_path),
        "size_bytes": int(log_path.stat().st_size) if log_path.exists() and log_path.is_file() else 0,
        "tail": _read_log_tail(log_path, max_lines=MAX_FAILURE_DIAGNOSTIC_LOG_TAIL_LINES),
    }


def _resolve_canonical_source_pool_summary(
    run_root: Path,
    canonical_source_pool_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：汇总 canonical source pool 的基础状态。

    Summarize the canonical source-pool state for failure diagnostics.

    Args:
        run_root: Stage run root.
        canonical_source_pool_payload: Canonical source-pool manifest payload.

    Returns:
        Canonical source-pool diagnostics summary.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(canonical_source_pool_payload, dict) or not canonical_source_pool_payload:
        return {
            "manifest_exists": False,
            "manifest_path": normalize_path_value(run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH),
            "entry_count": 0,
            "prompt_count": 0,
            "representative_root_records": {},
        }
    representative_root_records = (
        cast(Dict[str, Any], canonical_source_pool_payload.get("representative_root_records"))
        if isinstance(canonical_source_pool_payload.get("representative_root_records"), dict)
        else {}
    )
    return {
        "manifest_exists": True,
        "manifest_path": normalize_path_value(run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH),
        "entry_count": canonical_source_pool_payload.get("entry_count", 0),
        "prompt_count": canonical_source_pool_payload.get("prompt_count", canonical_source_pool_payload.get("entry_count", 0)),
        "entries_package_relative_root": canonical_source_pool_payload.get("entries_package_relative_root"),
        "representative_root_records": representative_root_records,
    }


def _resolve_stage_01_failure_reason(
    *,
    preflight: Dict[str, Any],
    runner_result: Dict[str, Any],
    workflow_summary_payload: Dict[str, Any],
    exc: Exception,
) -> str:
    """
    功能：解析 stage 01 失败原因代码。

    Resolve the stable failure-reason token for stage 01.

    Args:
        preflight: Preflight payload.
        runner_result: Runner execution result.
        workflow_summary_payload: Mainline workflow summary payload.
        exc: Raised exception.

    Returns:
        Stable failure-reason token.
    """
    if not isinstance(preflight, dict):
        raise TypeError("preflight must be dict")
    if not isinstance(runner_result, dict):
        raise TypeError("runner_result must be dict")
    if not isinstance(workflow_summary_payload, dict):
        raise TypeError("workflow_summary_payload must be dict")
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")

    workflow_failure_reason = workflow_summary_payload.get("failure_reason")
    if isinstance(workflow_failure_reason, str) and workflow_failure_reason:
        return workflow_failure_reason
    workflow_summary_reason = workflow_summary_payload.get("summary_reason")
    if isinstance(workflow_summary_reason, str) and workflow_summary_reason and workflow_summary_reason != "ok":
        return workflow_summary_reason
    if not bool(preflight.get("ok", False)):
        return "stage_01_preflight_failed"
    if int(runner_result.get("return_code", 0)) != 0:
        return "stage_01_mainline_failed"
    if isinstance(exc, FileNotFoundError):
        return "stage_01_required_outputs_missing"
    return type(exc).__name__


def _build_failure_diagnostics_summary(
    *,
    stage_run_id: str,
    run_root: Path,
    log_root: Path,
    workflow_summary_path: Path,
    stage_manifest_path: Path,
    preflight: Dict[str, Any],
    runner_result: Dict[str, Any],
    workflow_summary_payload: Dict[str, Any],
    canonical_source_pool_payload: Dict[str, Any],
    missing_required_artifacts: List[str],
    failure_reason: str,
    diagnostics_summary_path: Path,
    diagnostics_package_path: Path,
    diagnostics_manifest_path: Path,
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 failure diagnostics 摘要。

    Build the structured failure-diagnostics summary for stage 01.

    Args:
        stage_run_id: Stage run identifier.
        run_root: Stage run root.
        log_root: Wrapper log root.
        workflow_summary_path: Workflow summary path.
        stage_manifest_path: Stage manifest path.
        preflight: Preflight payload.
        runner_result: Runner execution result.
        workflow_summary_payload: Mainline workflow summary payload.
        canonical_source_pool_payload: Canonical source-pool manifest payload.
        missing_required_artifacts: Missing required artifact labels.
        failure_reason: Stable failure reason.
        diagnostics_summary_path: Diagnostics summary path.
        diagnostics_package_path: Diagnostics ZIP path.
        diagnostics_manifest_path: Diagnostics manifest path.

    Returns:
        Failure-diagnostics summary mapping.
    """
    relevant_logs = _collect_relevant_log_paths(log_root, run_root)
    return {
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "stage_status": "failed",
        "failure_reason": failure_reason,
        "workflow_summary_path": normalize_path_value(workflow_summary_path) if workflow_summary_path.exists() else "<absent>",
        "workflow_summary_status": workflow_summary_payload.get("status"),
        "workflow_summary_summary_reason": workflow_summary_payload.get("summary_reason"),
        "workflow_summary_failure_reason": workflow_summary_payload.get("failure_reason"),
        "stage_manifest_path": normalize_path_value(stage_manifest_path) if stage_manifest_path.exists() else "<absent>",
        "log_root": normalize_path_value(log_root),
        "runner_command": runner_result.get("command"),
        "command_stdout_tail": _read_log_tail(runner_result.get("stdout_log_path"), max_lines=MAX_FAILURE_DIAGNOSTIC_LOG_TAIL_LINES),
        "command_stderr_tail": _read_log_tail(runner_result.get("stderr_log_path"), max_lines=MAX_FAILURE_DIAGNOSTIC_LOG_TAIL_LINES),
        "preflight_summary": preflight,
        "preflight_ok": bool(preflight.get("ok", False)),
        "attestation_evidence_resolution": workflow_summary_payload.get("attestation_evidence_resolution", {}),
        "missing_required_artifacts": missing_required_artifacts,
        "required_artifacts": workflow_summary_payload.get("required_artifacts", {}),
        "canonical_source_pool_summary": _resolve_canonical_source_pool_summary(run_root, canonical_source_pool_payload),
        "latest_relevant_log_files": [_build_log_file_summary(log_path) for log_path in relevant_logs],
        "diagnostics_summary_path": normalize_path_value(diagnostics_summary_path),
        "diagnostics_package_path": normalize_path_value(diagnostics_package_path),
        "diagnostics_manifest_path": normalize_path_value(diagnostics_manifest_path),
    }


def _write_failure_diagnostics_package(
    *,
    stage_run_id: str,
    run_root: Path,
    log_root: Path,
    runtime_state_root: Path,
    workflow_summary_path: Path,
    stage_manifest_path: Path,
    canonical_source_pool_manifest_path: Path,
    diagnostics_summary_payload: Dict[str, Any],
    failure_reason: str,
) -> Dict[str, Any]:
    """
    功能：生成独立 failure diagnostics 目录、ZIP 与 manifest。

    Generate the isolated failure-diagnostics directory, ZIP package, and
    diagnostics manifest.

    Args:
        stage_run_id: Stage run identifier.
        run_root: Stage run root.
        log_root: Wrapper log root.
        runtime_state_root: Stage runtime-state root.
        workflow_summary_path: Workflow summary path.
        stage_manifest_path: Stage manifest path.
        canonical_source_pool_manifest_path: Canonical source-pool manifest path.
        diagnostics_summary_payload: Diagnostics summary payload.
        failure_reason: Stable failure reason.

    Returns:
        Diagnostics path and manifest summary mapping.
    """
    diagnostics_paths = _resolve_failure_diagnostics_paths(runtime_state_root, stage_run_id)
    diagnostics_root = ensure_directory(diagnostics_paths["diagnostics_root"])
    staging_root = diagnostics_paths["staging_root"]
    if staging_root.exists():
        shutil.rmtree(staging_root)
    ensure_directory(staging_root)

    summary_path = diagnostics_paths["summary_path"]
    manifest_path = diagnostics_paths["manifest_path"]
    index_path = diagnostics_paths["index_path"]
    package_path = diagnostics_paths["package_path"]
    write_json_atomic(summary_path, diagnostics_summary_payload)
    stage_relative_copy(summary_path, staging_root, FAILURE_DIAGNOSTICS_SUMMARY_FILE_NAME)

    staged_copy_specs = [
        (workflow_summary_path, "artifacts/workflow_summary.json"),
        (stage_manifest_path, "artifacts/stage_manifest.json"),
        (canonical_source_pool_manifest_path, CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH),
    ]
    for source_path, relative_path in staged_copy_specs:
        if source_path.exists() and source_path.is_file():
            stage_relative_copy(source_path, staging_root, relative_path)

    for log_path in _collect_relevant_log_paths(log_root, run_root):
        if log_root in log_path.parents or log_path == log_root:
            relative_log_path = f"logs/wrapper/{log_path.relative_to(log_root).as_posix()}"
        elif run_root in log_path.parents or log_path == run_root:
            relative_log_path = f"logs/run/{log_path.relative_to(run_root).as_posix()}"
        else:
            relative_log_path = f"logs/misc/{log_path.name}"
        stage_relative_copy(log_path, staging_root, relative_log_path)

    diagnostics_index = build_package_index(staging_root.rglob("*"), staging_root)
    write_json_atomic(index_path, diagnostics_index)
    write_json_atomic(staging_root / FAILURE_DIAGNOSTICS_INDEX_FILE_NAME, diagnostics_index)

    internal_manifest = {
        "artifact_type": "failure_diagnostics_manifest",
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "stage_status": "failed",
        "failure_reason": failure_reason,
        "diagnostics_summary_path": FAILURE_DIAGNOSTICS_SUMMARY_FILE_NAME,
        "diagnostics_package_path": "<external_zip>",
        "diagnostics_package_filename": package_path.name,
        "diagnostics_package_sha256": "<see_external_manifest>",
        "diagnostics_contents_index_path": FAILURE_DIAGNOSTICS_INDEX_FILE_NAME,
        "package_role": FAILURE_DIAGNOSTICS_PACKAGE_ROLE,
        "package_discovery_scope": EXCLUDED_PACKAGE_DISCOVERY_SCOPE,
        "diagnostics_manifest_scope": "internal_copy",
        "created_at": utc_now_iso(),
    }
    write_json_atomic(staging_root / FAILURE_DIAGNOSTICS_MANIFEST_FILE_NAME, internal_manifest)

    create_zip_archive_from_directory(staging_root, package_path)
    diagnostics_manifest = {
        "artifact_type": "failure_diagnostics_manifest",
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "stage_status": "failed",
        "failure_reason": failure_reason,
        "diagnostics_root": normalize_path_value(diagnostics_root),
        "diagnostics_summary_path": normalize_path_value(summary_path),
        "diagnostics_package_path": normalize_path_value(package_path),
        "diagnostics_package_filename": package_path.name,
        "diagnostics_package_sha256": compute_file_sha256(package_path),
        "diagnostics_contents_index_path": normalize_path_value(index_path),
        "package_role": FAILURE_DIAGNOSTICS_PACKAGE_ROLE,
        "package_discovery_scope": EXCLUDED_PACKAGE_DISCOVERY_SCOPE,
        "diagnostics_manifest_scope": "external_copy",
        "created_at": utc_now_iso(),
    }
    write_json_atomic(manifest_path, diagnostics_manifest)
    return {
        "diagnostics_status": "generated",
        "diagnostics_generation_reason": failure_reason,
        "diagnostics_summary_path": normalize_path_value(summary_path),
        "diagnostics_package_path": normalize_path_value(package_path),
        "diagnostics_manifest_path": normalize_path_value(manifest_path),
    }


def _build_failure_stage_manifest(
    *,
    config_path: Path,
    notebook_name: str,
    stage_run_id: str,
    run_root: Path,
    log_root: Path,
    runtime_state_root: Path,
    export_root: Path,
    runtime_config_snapshot_path: Path,
    prompt_snapshot: Dict[str, Any],
    cfg_obj: Dict[str, Any],
    preflight: Dict[str, Any],
    runner_result: Dict[str, Any],
    workflow_summary_payload: Dict[str, Any],
    canonical_source_pool_payload: Dict[str, Any],
    missing_required_artifacts: List[str],
    diagnostics_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造 stage 01 失败态 stage_manifest。

    Build the failure-mode stage_manifest for stage 01.

    Args:
        config_path: Source config path.
        notebook_name: Notebook display name.
        stage_run_id: Stage run identifier.
        run_root: Stage run root.
        log_root: Wrapper log root.
        runtime_state_root: Stage runtime-state root.
        export_root: Stage export root.
        runtime_config_snapshot_path: Runtime config snapshot path.
        prompt_snapshot: Prompt snapshot payload.
        cfg_obj: Runtime config payload.
        preflight: Preflight payload.
        runner_result: Runner execution result.
        workflow_summary_payload: Mainline workflow summary payload.
        canonical_source_pool_payload: Canonical source-pool manifest payload.
        missing_required_artifacts: Missing required artifact labels.
        diagnostics_fields: Diagnostics status fields.

    Returns:
        Failure stage manifest mapping.
    """
    representative_root_records = (
        cast(Dict[str, Any], canonical_source_pool_payload.get("representative_root_records"))
        if isinstance(canonical_source_pool_payload.get("representative_root_records"), dict)
        else {}
    )
    attestation_evidence_fields = _resolve_attestation_evidence_manifest_fields(
        workflow_summary_payload,
        canonical_source_pool_payload,
    )
    records = collect_file_index(
        run_root,
        {
            "embed_record": run_root / "records" / "embed_record.json",
            "detect_record": run_root / "records" / "detect_record.json",
            "calibration_record": run_root / "records" / "calibration_record.json",
            "evaluate_record": run_root / "records" / "evaluate_record.json",
        },
    )
    manifest: Dict[str, Any] = {
        "stage_name": STAGE_01_NAME,
        "stage_run_id": stage_run_id,
        "stage_status": "failed",
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
        "records": records,
        "workflow_summary_path": normalize_path_value(run_root / "artifacts" / "workflow_summary.json"),
        "workflow_summary_status": workflow_summary_payload.get("status"),
        "workflow_summary_summary_reason": workflow_summary_payload.get("summary_reason"),
        "workflow_summary_failure_reason": workflow_summary_payload.get("failure_reason"),
        "stage_01_canonical_source_pool_manifest_path": normalize_path_value(run_root / CANONICAL_SOURCE_POOL_MANIFEST_RELATIVE_PATH),
        "stage_01_canonical_source_pool_manifest_package_relative_path": canonical_source_pool_payload.get("manifest_package_relative_path"),
        "stage_01_canonical_source_pool_entry_count": canonical_source_pool_payload.get("entry_count"),
        "stage_01_representative_root_records": representative_root_records,
        "stage_01_representative_root_prompt_index": representative_root_records.get("source_prompt_index"),
        "stage_01_representative_root_source_entry_package_relative_path": representative_root_records.get(
            "source_entry_package_relative_path"
        ),
        "missing_required_artifacts": missing_required_artifacts,
        "preflight": preflight,
        "runner_result": runner_result,
        "formal_package_status": "not_generated",
        "formal_package_role": FORMAL_STAGE_PACKAGE_ROLE,
        "formal_package_discovery_scope": FORMAL_PACKAGE_DISCOVERY_SCOPE,
        "formal_package_path": "<absent>",
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
    }
    manifest.update(attestation_evidence_fields)
    manifest.update(diagnostics_fields)
    return manifest


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
    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"

    ensure_attestation_env_bootstrap(
        cfg_obj,
        drive_project_root,
        allow_generate=False,
        allow_missing=True,
    )
    preflight = detect_stage_01_preflight(runtime_config_snapshot_path)
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
    runner_result: Dict[str, Any] = {}

    try:
        if not bool(preflight.get("ok", False)):
            raise RuntimeError(f"formal GPU preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

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
        stats_contract_payload = read_required_json_dict(
            stats_contract_path,
            "stage 01 parallel_attestation_statistics_input_contract",
        )
        canonical_source_pool_manifest_path = outputs["canonical_source_pool_manifest"]
        canonical_source_pool_payload = read_required_json_dict(
            canonical_source_pool_manifest_path,
            "stage 01 canonical source pool manifest",
        )
        pooled_threshold_build_contract_path = outputs["stage_01_pooled_threshold_build_contract"]
        pooled_threshold_build_contract_payload = read_required_json_dict(
            pooled_threshold_build_contract_path,
            "stage 01 pooled threshold build contract",
        )
        workflow_summary_payload = read_required_json_dict(
            outputs["workflow_summary"],
            "stage 01 workflow summary",
        )
        representative_root_records = canonical_source_pool_payload.get("representative_root_records", {})
        if not isinstance(representative_root_records, dict):
            representative_root_records = {}
        attestation_evidence_fields = _resolve_attestation_evidence_manifest_fields(
            workflow_summary_payload,
            canonical_source_pool_payload,
        )

        stage_manifest: Dict[str, Any] = {
            "stage_name": STAGE_01_NAME,
            "stage_run_id": stage_run_id,
            "stage_status": "ok",
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
            "stage_01_canonical_source_pool_manifest_path": normalize_path_value(canonical_source_pool_manifest_path),
            "stage_01_canonical_source_pool_manifest_package_relative_path": canonical_source_pool_payload.get(
                "manifest_package_relative_path"
            ),
            "stage_01_canonical_source_pool_root_path": canonical_source_pool_payload.get(
                "canonical_source_pool_root_path"
            ),
            "stage_01_canonical_source_pool_root_package_relative_path": canonical_source_pool_payload.get(
                "canonical_source_pool_root_package_relative_path"
            ),
            "stage_01_canonical_source_pool_entries_package_relative_root": canonical_source_pool_payload.get(
                "entries_package_relative_root"
            ),
            "stage_01_canonical_source_pool_prompt_count": canonical_source_pool_payload.get(
                "prompt_count",
                canonical_source_pool_payload.get("entry_count"),
            ),
            "stage_01_canonical_source_pool_entry_count": canonical_source_pool_payload.get("entry_count"),
            "stage_01_canonical_source_pool_prompt_file": canonical_source_pool_payload.get("prompt_file"),
            "stage_01_canonical_source_pool_prompt_file_path": canonical_source_pool_payload.get("prompt_file"),
            "stage_01_representative_root_records": representative_root_records,
            "stage_01_representative_root_prompt_index": representative_root_records.get("source_prompt_index"),
            "stage_01_representative_root_source_entry_package_relative_path": representative_root_records.get(
                "source_entry_package_relative_path"
            ),
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
            "workflow_summary_status": workflow_summary_payload.get("status"),
            "workflow_summary_summary_reason": workflow_summary_payload.get("summary_reason"),
            "workflow_summary_failure_reason": workflow_summary_payload.get("failure_reason"),
            "formal_package_status": "generated",
            "formal_package_role": FORMAL_STAGE_PACKAGE_ROLE,
            "formal_package_discovery_scope": FORMAL_PACKAGE_DISCOVERY_SCOPE,
            "diagnostics_status": "not_generated",
            "diagnostics_generation_reason": "stage_completed_without_failure",
            "diagnostics_summary_path": "<absent>",
            "diagnostics_package_path": "<absent>",
            "diagnostics_manifest_path": "<absent>",
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
        stage_manifest.update(attestation_evidence_fields)
        write_json_atomic(stage_manifest_path, stage_manifest)

        package_root = _package_stage_outputs(
            run_root,
            runtime_state_root,
            stage_manifest_path,
            runtime_config_snapshot_path,
            canonical_source_pool_payload,
            stats_contract_payload,
            pooled_threshold_build_contract_payload,
        )
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
            "canonical_source_pool_manifest_path": normalize_path_value(canonical_source_pool_manifest_path),
            "canonical_source_pool_entry_count": canonical_source_pool_payload.get("entry_count"),
            "package_filename": build_stage_package_filename(STAGE_01_NAME, stage_run_id),
            "package_path": package_manifest["package_path"],
            "package_sha256": package_manifest["package_sha256"],
            "formal_package_status": "generated",
            "diagnostics_status": "not_generated",
            "diagnostics_generation_reason": "stage_completed_without_failure",
            "diagnostics_summary_path": "<absent>",
            "diagnostics_package_path": "<absent>",
            "diagnostics_manifest_path": "<absent>",
            "status": "ok",
        }
        write_json_atomic(runtime_state_root / "stage_summary.json", summary)
        return summary
    except Exception as exc:
        outputs = _required_stage_outputs(run_root)
        missing_required_artifacts = [label for label, path_obj in outputs.items() if not path_obj.exists()]
        workflow_summary_payload = _load_optional_json_object(workflow_summary_path)
        canonical_source_pool_manifest_path = outputs["canonical_source_pool_manifest"]
        canonical_source_pool_payload = _load_optional_json_object(canonical_source_pool_manifest_path)
        package_manifest_payload = _load_optional_json_object(package_manifest_path)
        if package_manifest_payload:
            package_manifest_payload["package_manifest_path"] = normalize_path_value(package_manifest_path)

        failure_reason = _resolve_stage_01_failure_reason(
            preflight=preflight,
            runner_result=runner_result,
            workflow_summary_payload=workflow_summary_payload,
            exc=exc,
        )
        diagnostics_paths = _resolve_failure_diagnostics_paths(runtime_state_root, stage_run_id)
        diagnostics_fields = {
            "diagnostics_status": "generated",
            "diagnostics_generation_reason": failure_reason,
            "diagnostics_summary_path": normalize_path_value(diagnostics_paths["summary_path"]),
            "diagnostics_package_path": normalize_path_value(diagnostics_paths["package_path"]),
            "diagnostics_manifest_path": normalize_path_value(diagnostics_paths["manifest_path"]),
        }
        formal_package_fields = _resolve_formal_package_summary(package_manifest_payload)

        failure_stage_manifest = _build_failure_stage_manifest(
            config_path=config_path,
            notebook_name=notebook_name,
            stage_run_id=stage_run_id,
            run_root=run_root,
            log_root=log_root,
            runtime_state_root=runtime_state_root,
            export_root=export_root,
            runtime_config_snapshot_path=runtime_config_snapshot_path,
            prompt_snapshot=prompt_snapshot,
            cfg_obj=cfg_obj,
            preflight=preflight,
            runner_result=runner_result,
            workflow_summary_payload=workflow_summary_payload,
            canonical_source_pool_payload=canonical_source_pool_payload,
            missing_required_artifacts=missing_required_artifacts,
            diagnostics_fields=diagnostics_fields,
        )
        failure_stage_manifest.update(formal_package_fields)
        write_json_atomic(stage_manifest_path, failure_stage_manifest)

        diagnostics_summary_payload = _build_failure_diagnostics_summary(
            stage_run_id=stage_run_id,
            run_root=run_root,
            log_root=log_root,
            workflow_summary_path=workflow_summary_path,
            stage_manifest_path=stage_manifest_path,
            preflight=preflight,
            runner_result=runner_result,
            workflow_summary_payload=workflow_summary_payload,
            canonical_source_pool_payload=canonical_source_pool_payload,
            missing_required_artifacts=missing_required_artifacts,
            failure_reason=failure_reason,
            diagnostics_summary_path=diagnostics_paths["summary_path"],
            diagnostics_package_path=diagnostics_paths["package_path"],
            diagnostics_manifest_path=diagnostics_paths["manifest_path"],
        )
        _write_failure_diagnostics_package(
            stage_run_id=stage_run_id,
            run_root=run_root,
            log_root=log_root,
            runtime_state_root=runtime_state_root,
            workflow_summary_path=workflow_summary_path,
            stage_manifest_path=stage_manifest_path,
            canonical_source_pool_manifest_path=canonical_source_pool_manifest_path,
            diagnostics_summary_payload=diagnostics_summary_payload,
            failure_reason=failure_reason,
        )

        failure_summary: Dict[str, Any] = {
            "stage_name": STAGE_01_NAME,
            "stage_run_id": stage_run_id,
            "run_root": normalize_path_value(run_root),
            "log_root": normalize_path_value(log_root),
            "runtime_state_root": normalize_path_value(runtime_state_root),
            "export_root": normalize_path_value(export_root),
            "stage_manifest_path": normalize_path_value(stage_manifest_path),
            "package_manifest_path": formal_package_fields["formal_package_manifest_path"],
            "package_filename": formal_package_fields["formal_package_filename"],
            "package_path": formal_package_fields["formal_package_path"],
            "package_sha256": str(package_manifest_payload.get("package_sha256", "<absent>")) if package_manifest_payload else "<absent>",
            "formal_package_status": formal_package_fields["formal_package_status"],
            "diagnostics_status": diagnostics_fields["diagnostics_status"],
            "diagnostics_generation_reason": diagnostics_fields["diagnostics_generation_reason"],
            "diagnostics_summary_path": diagnostics_fields["diagnostics_summary_path"],
            "diagnostics_package_path": diagnostics_fields["diagnostics_package_path"],
            "diagnostics_manifest_path": diagnostics_fields["diagnostics_manifest_path"],
            "failure_reason": failure_reason,
            "status": "failed",
        }
        write_json_atomic(runtime_state_root / "stage_summary.json", failure_summary)
        raise


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
