"""
文件目的：01_Paper_Full_Cuda notebook 正式运行编排入口。
Module type: General module

职责边界：
1. 仅负责路径解析、配置校验、正式脚本编排、日志落盘与运行摘要输出。
2. 复用既有 scripts/main CLI 入口，不重写 embed/detect/calibrate/evaluate 机制。
3. 不改写正式 records，不绕过 preflight、冻结语义、阈值只读语义与 attestation 语义。
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent
if str(DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO_ROOT))

from scripts.workflow_acceptance_common import (
    build_path_views,
    detect_formal_gpu_preflight,
    normalize_path_value,
    relative_path_under_base,
)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
PAPER_FULL_CONFIG_PATH = Path("configs/paper_full_cuda.yaml")
MAIN_WORKFLOW_SCRIPT_PATH = Path("scripts/run_paper_full_cuda.py")
PARALLEL_STATISTICS_SCRIPT_PATH = Path("scripts/run_parallel_attestation_statistics.py")


def _utc_now_iso() -> str:
    """
    功能：生成 UTC ISO 时间戳。

    Generate a UTC timestamp in ISO 8601 format.

    Args:
        None.

    Returns:
        UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    """
    功能：按仓库根目录解析路径。

    Resolve a path against the repository root unless it is already absolute.

    Args:
        path_value: Raw path string from CLI.
        repo_root: Repository root path.

    Returns:
        Resolved absolute path.
    """
    if not path_value.strip():
        raise TypeError("path_value must be non-empty str")

    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _load_yaml_mapping(path_obj: Path) -> Dict[str, Any]:
    """
    功能：加载 YAML 映射配置。

    Load a YAML file and require its root node to be a mapping.

    Args:
        path_obj: YAML file path.

    Returns:
        Parsed mapping.
    """
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"config file not found: {path_obj}")

    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be mapping: {path_obj}")
    return cast(Dict[str, Any], payload)


def _read_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取可选 JSON 对象文件。

    Read a JSON object file when it exists.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed mapping, or an empty mapping when absent or invalid.
    """
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else {}


def _write_json_atomic(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    功能：以原子方式写出结构化 JSON。

    Persist a JSON mapping with stable formatting using atomic replace.

    Args:
        path_obj: Destination JSON path.
        payload: JSON payload mapping.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path_obj.parent),
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    ) as tmp_file:
        tmp_file.write(serialized)
        tmp_path = Path(tmp_file.name)

    tmp_path.replace(path_obj)


def _normalize_notebook_group_config(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：对 notebook 组默认事实源进行规范化比较。

    Normalize notebook-group config fields so consistency checks ignore only the
    default-source convergence path and no mechanism fields.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Normalized config mapping copy.
    """
    normalized = copy.deepcopy(cfg_obj)
    experiment_matrix_cfg = normalized.get("experiment_matrix")
    if isinstance(experiment_matrix_cfg, dict):
        experiment_matrix_cfg = cast(Dict[str, Any], experiment_matrix_cfg)
        config_path_value = experiment_matrix_cfg.get("config_path")
        if isinstance(config_path_value, str) and config_path_value in {
            DEFAULT_CONFIG_PATH.as_posix(),
            PAPER_FULL_CONFIG_PATH.as_posix(),
        }:
            experiment_matrix_cfg["config_path"] = DEFAULT_CONFIG_PATH.as_posix()
    return normalized


def _resolve_parallel_attestation_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析专项统计配置。

    Resolve the detached parallel attestation statistics configuration.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Normalized detached-statistics config mapping.
    """
    section_node = cfg_obj.get("parallel_attestation_statistics")
    section: Dict[str, Any] = cast(Dict[str, Any], section_node) if isinstance(section_node, dict) else {}
    enabled_value = section.get("enabled", False)
    enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False
    return {
        "enabled": enabled,
        "calibration_score_name": section.get("calibration_score_name"),
        "evaluate_score_name": section.get("evaluate_score_name"),
    }


def _validate_default_config_binding(repo_root: Path, config_path: Path) -> Dict[str, Any]:
    """
    功能：校验 notebook 组唯一默认配置绑定。

    Validate that configs/default.yaml is the unique notebook-group default
    source and that paper_full_cuda remains semantically aligned except for the
    converged default-source path.

    Args:
        repo_root: Repository root path.
        config_path: Effective default config path.

    Returns:
        Config-guard summary mapping.
    """
    default_cfg = _load_yaml_mapping(config_path)
    experiment_matrix_cfg = default_cfg.get("experiment_matrix")
    if not isinstance(experiment_matrix_cfg, dict):
        raise ValueError("default config requires experiment_matrix mapping")
    experiment_matrix_cfg = cast(Dict[str, Any], experiment_matrix_cfg)

    config_path_value = experiment_matrix_cfg.get("config_path")
    if config_path_value != DEFAULT_CONFIG_PATH.as_posix():
        raise ValueError(
            "configs/default.yaml must set experiment_matrix.config_path to configs/default.yaml"
        )

    summary: Dict[str, Any] = {
        "default_config_path": normalize_path_value(config_path),
        "default_config_repo_relative": relative_path_under_base(repo_root, config_path),
        "default_experiment_matrix_config_path": config_path_value,
        "paper_full_config_exists": False,
        "paper_full_consistency": "not_present",
    }

    paper_cfg_path = (repo_root / PAPER_FULL_CONFIG_PATH).resolve()
    if not paper_cfg_path.exists():
        return summary

    paper_cfg = _load_yaml_mapping(paper_cfg_path)
    paper_experiment_matrix_cfg = paper_cfg.get("experiment_matrix")
    summary["paper_full_config_exists"] = True
    summary["paper_full_config_path"] = normalize_path_value(paper_cfg_path)
    summary["paper_full_config_repo_relative"] = relative_path_under_base(repo_root, paper_cfg_path)
    summary["paper_full_experiment_matrix_config_path"] = (
        cast(Dict[str, Any], paper_experiment_matrix_cfg).get("config_path")
        if isinstance(paper_experiment_matrix_cfg, dict)
        else None
    )

    default_normalized = _normalize_notebook_group_config(default_cfg)
    paper_normalized = _normalize_notebook_group_config(paper_cfg)
    configs_consistent = default_normalized == paper_normalized
    summary["paper_full_consistency"] = "consistent" if configs_consistent else "inconsistent"

    if not configs_consistent:
        raise ValueError(
            "configs/default.yaml and configs/paper_full_cuda.yaml diverge beyond the notebook-group default-source convergence field"
        )
    return summary


def _resolve_runtime_state_root(workspace_summary_path: Path, run_request_path: Path) -> Path:
    """
    功能：解析 runtime_state 目录。

    Resolve the runtime-state directory from notebook-provided state files.

    Args:
        workspace_summary_path: Workspace summary JSON path.
        run_request_path: Run request JSON path.

    Returns:
        Runtime-state directory path.
    """
    workspace_state_root = workspace_summary_path.parent.resolve()
    request_state_root = run_request_path.parent.resolve()
    if workspace_state_root != request_state_root:
        raise ValueError(
            "workspace_summary_path and run_request_path must share the same runtime_state directory"
        )
    return workspace_state_root


def _validate_paths_under_drive_root(
    drive_project_root: Path,
    resolved_run_root: Path,
    drive_export_root: Path,
    drive_log_root: Path,
    runtime_state_root: Path,
) -> None:
    """
    功能：校验关键输出路径位于项目持久化根目录下。

    Validate that notebook persistence paths are rooted under the declared
    Google Drive project root.

    Args:
        drive_project_root: Declared Google Drive project root.
        resolved_run_root: Requested workflow run root.
        drive_export_root: Declared export root.
        drive_log_root: Declared log root.
        runtime_state_root: Derived runtime-state root.

    Returns:
        None.
    """
    protected_paths = {
        "resolved_run_root": resolved_run_root,
        "drive_export_root": drive_export_root,
        "drive_log_root": drive_log_root,
        "runtime_state_root": runtime_state_root,
    }
    for label, path_obj in protected_paths.items():
        try:
            path_obj.resolve().relative_to(drive_project_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{label} must be under drive_project_root: {path_obj} not under {drive_project_root}"
            ) from exc


def _ensure_required_scripts(repo_root: Path) -> Dict[str, str]:
    """
    功能：校验正式脚本入口存在。

    Validate that required orchestration scripts exist in the repository.

    Args:
        repo_root: Repository root path.

    Returns:
        Mapping of required script labels to normalized paths.
    """
    required_paths = {
        "main_workflow_script": (repo_root / MAIN_WORKFLOW_SCRIPT_PATH).resolve(),
        "parallel_statistics_script": (repo_root / PARALLEL_STATISTICS_SCRIPT_PATH).resolve(),
    }
    for label, path_obj in required_paths.items():
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"required script missing: {label}={path_obj}")
    return {label: normalize_path_value(path_obj) for label, path_obj in required_paths.items()}


def _build_run_tag(request_run_root: Path, run_request_path: Path) -> str:
    """
    功能：构造稳定的运行标签。

    Build a stable run tag for manifest and log file names.

    Args:
        request_run_root: Requested run root path.
        run_request_path: Notebook run-request path.

    Returns:
        Stable run tag string.
    """
    if request_run_root.name:
        return request_run_root.name
    stem = run_request_path.stem
    if stem.endswith("_run_request"):
        return stem[: -len("_run_request")]
    return stem or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_command_with_logs(
    command: Sequence[str],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行子进程并将输出分别落盘。

    Execute a subprocess and persist stdout and stderr to explicit log files.

    Args:
        command: Command argument sequence.
        cwd: Working directory.
        stdout_log_path: Stdout log file path.
        stderr_log_path: Stderr log file path.

    Returns:
        Execution result mapping with return code and log paths.
    """
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_log_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "return_code": int(result.returncode),
        "stdout_log_path": normalize_path_value(stdout_log_path),
        "stderr_log_path": normalize_path_value(stderr_log_path),
        "command": [str(item) for item in command],
    }


def _build_required_artifacts(run_root: Path, parallel_enabled: bool) -> Dict[str, Path]:
    """
    功能：构造必需产物路径集合。

    Build the required artifact path map for workflow validation.

    Args:
        run_root: Main workflow run root.
        parallel_enabled: Whether detached parallel statistics are enabled.

    Returns:
        Mapping of artifact labels to paths.
    """
    required_paths = {
        "embed_record": run_root / "records" / "embed_record.json",
        "detect_record": run_root / "records" / "detect_record.json",
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "run_closure": run_root / "artifacts" / "run_closure.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
    }
    if parallel_enabled:
        required_paths["parallel_attestation_statistics_summary"] = (
            run_root
            / "outputs"
            / "parallel_attestation_statistics"
            / "artifacts"
            / "parallel_attestation_statistics_summary.json"
        )
    return required_paths


def _collect_artifact_presence(required_paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    功能：只读核验产物存在性。

    Collect read-only existence checks for required artifacts.

    Args:
        required_paths: Mapping of artifact labels to paths.

    Returns:
        Validation mapping with existence flags and path strings.
    """
    result: Dict[str, Any] = {}
    for artifact_name, path_obj in required_paths.items():
        result[artifact_name] = {
            "path": normalize_path_value(path_obj),
            "exists": bool(path_obj.exists()),
        }
    return result


def _all_artifacts_present(validation_result: Dict[str, Any], include_parallel: bool) -> bool:
    """
    功能：判断目标产物是否齐全。

    Determine whether required artifacts are fully present.

    Args:
        validation_result: Artifact validation mapping.
        include_parallel: Whether to require detached parallel output.

    Returns:
        True when the targeted artifact set is complete.
    """
    for artifact_name, artifact_state in validation_result.items():
        if not include_parallel and artifact_name == "parallel_attestation_statistics_summary":
            continue
        if not isinstance(artifact_state, dict):
            return False
        artifact_state = cast(Dict[str, Any], artifact_state)
        if artifact_state.get("exists") is not True:
            return False
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    功能：构建命令行参数解析器。

    Build the CLI parser for notebook orchestration.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the formal notebook orchestration for 01_Paper_Full_Cuda.",
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root path.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Runtime config path (default: configs/default.yaml).",
    )
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--request-run-root", required=True, help="Requested workflow run root.")
    parser.add_argument("--drive-export-root", required=True, help="Google Drive export root.")
    parser.add_argument("--drive-log-root", required=True, help="Google Drive log root.")
    parser.add_argument("--workspace-summary-path", required=True, help="Notebook workspace summary path.")
    parser.add_argument("--run-request-path", required=True, help="Notebook run-request path.")
    return parser


def main() -> int:
    """
    功能：执行 notebook 正式编排入口。

    Execute the notebook-oriented formal orchestration entrypoint.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    repo_root = _resolve_path(args.repo_root, DEFAULT_REPO_ROOT)
    config_path = _resolve_path(args.config, repo_root)
    drive_project_root = _resolve_path(args.drive_project_root, repo_root)
    resolved_run_root = _resolve_path(args.request_run_root, repo_root)
    drive_export_root = _resolve_path(args.drive_export_root, repo_root)
    drive_log_root = _resolve_path(args.drive_log_root, repo_root)
    workspace_summary_path = _resolve_path(args.workspace_summary_path, repo_root)
    run_request_path = _resolve_path(args.run_request_path, repo_root)
    runtime_state_root = _resolve_runtime_state_root(workspace_summary_path, run_request_path)
    run_tag = _build_run_tag(resolved_run_root, run_request_path)

    runtime_state_root.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_state_root / f"{run_tag}_run_manifest.json"
    validation_summary_path = runtime_state_root / f"{run_tag}_validation_summary.json"

    config_guard_summary: Dict[str, Any] = {}
    script_summary: Dict[str, Any] = {}
    preflight: Dict[str, Any] = {"ok": False, "status": "not_run"}
    main_result: Dict[str, Any] = {
        "return_code": None,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "status": "not_run",
    }
    parallel_result: Dict[str, Any] = {
        "enabled": False,
        "status": "not_run",
        "return_code": None,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "summary_path": normalize_path_value(
            resolved_run_root
            / "outputs"
            / "parallel_attestation_statistics"
            / "artifacts"
            / "parallel_attestation_statistics_summary.json"
        ),
    }
    detached_cfg: Dict[str, Any] = {"enabled": False}
    workspace_summary = _read_json_dict(workspace_summary_path)
    run_request = _read_json_dict(run_request_path)

    manifest: Dict[str, Any] = {
        "started_at_utc": _utc_now_iso(),
        "status": "running",
        "repo_root": normalize_path_value(repo_root),
        "repo_root_relative": relative_path_under_base(repo_root, repo_root),
        "config_path": normalize_path_value(config_path),
        "config_repo_relative": relative_path_under_base(repo_root, config_path),
        "drive_project_root": normalize_path_value(drive_project_root),
        "resolved_run_root": normalize_path_value(resolved_run_root),
        "drive_export_root": normalize_path_value(drive_export_root),
        "drive_log_root": normalize_path_value(drive_log_root),
        "workspace_summary_path": normalize_path_value(workspace_summary_path),
        "run_request_path": normalize_path_value(run_request_path),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "main_workflow_return_code": None,
        "parallel_attestation_status": "not_run",
        "stdout_log_path": None,
        "stderr_log_path": None,
        "validation_summary_path": normalize_path_value(validation_summary_path),
        "workspace_summary_exists": bool(workspace_summary_path.exists()),
        "run_request_exists": bool(run_request_path.exists()),
        "workspace_summary": workspace_summary,
        "run_request": run_request,
    }
    _write_json_atomic(manifest_path, manifest)

    try:
        _validate_paths_under_drive_root(
            drive_project_root=drive_project_root,
            resolved_run_root=resolved_run_root,
            drive_export_root=drive_export_root,
            drive_log_root=drive_log_root,
            runtime_state_root=runtime_state_root,
        )
        config_guard_summary = _validate_default_config_binding(repo_root, config_path)
        script_summary = _ensure_required_scripts(repo_root)

        default_cfg = _load_yaml_mapping(config_path)
        detached_cfg = _resolve_parallel_attestation_cfg(default_cfg)
        parallel_result["enabled"] = bool(detached_cfg.get("enabled", False))
        parallel_result["status"] = "disabled" if not parallel_result["enabled"] else "pending"

        preflight = detect_formal_gpu_preflight(config_path)
        preflight["status"] = "ok" if bool(preflight.get("ok", False)) else "blocked"

        if not bool(preflight.get("ok", False)):
            raise RuntimeError(
                f"formal GPU preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}"
            )

        resolved_run_root.mkdir(parents=True, exist_ok=True)
        drive_export_root.mkdir(parents=True, exist_ok=True)
        drive_log_root.mkdir(parents=True, exist_ok=True)

        main_stdout_log_path = drive_log_root / f"{run_tag}_paper_full_cuda_stdout.log"
        main_stderr_log_path = drive_log_root / f"{run_tag}_paper_full_cuda_stderr.log"
        main_command = [
            sys.executable,
            str((repo_root / MAIN_WORKFLOW_SCRIPT_PATH).resolve()),
            "--config",
            str(config_path),
            "--run-root",
            str(resolved_run_root),
        ]
        main_result = _run_command_with_logs(
            command=main_command,
            cwd=repo_root,
            stdout_log_path=main_stdout_log_path,
            stderr_log_path=main_stderr_log_path,
        )
        main_result["status"] = "ok" if main_result["return_code"] == 0 else "failed"

        if bool(detached_cfg.get("enabled", False)):
            if main_result["return_code"] == 0:
                parallel_stdout_log_path = (
                    drive_log_root / f"{run_tag}_parallel_attestation_statistics_stdout.log"
                )
                parallel_stderr_log_path = (
                    drive_log_root / f"{run_tag}_parallel_attestation_statistics_stderr.log"
                )
                parallel_command = [
                    sys.executable,
                    str((repo_root / PARALLEL_STATISTICS_SCRIPT_PATH).resolve()),
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(resolved_run_root),
                ]
                parallel_result = {
                    **parallel_result,
                    **_run_command_with_logs(
                        command=parallel_command,
                        cwd=repo_root,
                        stdout_log_path=parallel_stdout_log_path,
                        stderr_log_path=parallel_stderr_log_path,
                    ),
                }
                parallel_result["status"] = "ok" if parallel_result["return_code"] == 0 else "failed"
            else:
                parallel_result["status"] = "skipped_main_failed"
        else:
            parallel_result["status"] = "disabled"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_message"] = str(exc)
    finally:
        artifact_validation = _collect_artifact_presence(
            _build_required_artifacts(resolved_run_root, bool(detached_cfg.get("enabled", False)))
        )
        main_artifacts_ok = _all_artifacts_present(artifact_validation, include_parallel=False)
        parallel_artifacts_ok = _all_artifacts_present(
            artifact_validation,
            include_parallel=bool(detached_cfg.get("enabled", False)),
        )

        if not bool(preflight.get("ok", False)):
            verdict = "environment_blocked"
        elif main_result.get("return_code") not in (0, None):
            verdict = "main_workflow_failed"
        elif main_result.get("return_code") == 0 and not main_artifacts_ok:
            verdict = "required_artifacts_missing"
        elif bool(detached_cfg.get("enabled", False)) and parallel_result.get("status") == "failed":
            verdict = "main_ok_parallel_failed"
        elif bool(detached_cfg.get("enabled", False)) and parallel_result.get("status") == "ok" and not parallel_artifacts_ok:
            verdict = "parallel_artifacts_missing"
        else:
            verdict = "ok"

        validation_summary: Dict[str, Any] = {
            "generated_at_utc": _utc_now_iso(),
            "config_path": normalize_path_value(config_path),
            "resolved_run_root": normalize_path_value(resolved_run_root),
            "main_workflow_status": main_result.get("status"),
            "main_workflow_return_code": main_result.get("return_code"),
            "detached_parallel_attestation_status": parallel_result.get("status"),
            "detached_parallel_attestation_return_code": parallel_result.get("return_code"),
            "preflight": preflight,
            "required_artifacts": artifact_validation,
            "required_main_artifacts_present": main_artifacts_ok,
            "required_parallel_artifacts_present": parallel_artifacts_ok,
            "verdict": verdict,
        }
        _write_json_atomic(validation_summary_path, validation_summary)

        path_views = build_path_views(
            resolved_run_root,
            {
                "run_root": resolved_run_root,
                "main_stdout_log": Path(main_result["stdout_log_path"]) if isinstance(main_result.get("stdout_log_path"), str) else None,
                "main_stderr_log": Path(main_result["stderr_log_path"]) if isinstance(main_result.get("stderr_log_path"), str) else None,
                "parallel_stdout_log": Path(parallel_result["stdout_log_path"]) if isinstance(parallel_result.get("stdout_log_path"), str) else None,
                "parallel_stderr_log": Path(parallel_result["stderr_log_path"]) if isinstance(parallel_result.get("stderr_log_path"), str) else None,
                "parallel_summary": Path(parallel_result["summary_path"]) if isinstance(parallel_result.get("summary_path"), str) else None,
                "manifest_path": manifest_path,
                "validation_summary_path": validation_summary_path,
            },
        )

        manifest.update(
            {
                "ended_at_utc": _utc_now_iso(),
                "status": "ok" if verdict == "ok" else "failed",
                "config_guard_summary": config_guard_summary,
                "required_scripts": script_summary,
                "preflight": preflight,
                "main_workflow_return_code": main_result.get("return_code"),
                "parallel_attestation_status": parallel_result.get("status"),
                "parallel_attestation_return_code": parallel_result.get("return_code"),
                "stdout_log_path": main_result.get("stdout_log_path"),
                "stderr_log_path": main_result.get("stderr_log_path"),
                "parallel_stdout_log_path": parallel_result.get("stdout_log_path"),
                "parallel_stderr_log_path": parallel_result.get("stderr_log_path"),
                "parallel_attestation_summary_path": parallel_result.get("summary_path"),
                "validation_summary_path": normalize_path_value(validation_summary_path),
                "drive_export_root": normalize_path_value(drive_export_root),
                "paths": path_views["paths"],
                "paths_relative": path_views["paths_relative"],
                "verdict": verdict,
            }
        )
        _write_json_atomic(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if manifest.get("status") == "ok":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())