#!/usr/bin/env python3
"""
文件目的：mini real set 结构验证入口。
Module type: General module

职责边界：
1. 复用 onefile 正式主链执行 mini real validation 配置。
2. 读取现有 run_root 下的 records 与 artifacts，整理结构化失败摘要。
3. 仅提供脚本级诊断视图，不改写正式 records schema，也不改变阈值或 attestation 语义。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.run_onefile_workflow import PROFILE_PAPER_FULL_CUDA, run_onefile_workflow
from scripts.workflow_acceptance_common import (
    collect_workflow_state,
    detect_formal_gpu_preflight,
    load_json_dict,
    load_runtime_config,
    write_workflow_summary,
)


DEFAULT_CONFIG_PATH = Path("configs/paper_full_cuda_mini_real_validation.yaml")
DEFAULT_RUN_ROOT = Path("outputs/onefile_paper_full_cuda_mini_real_validation")


def _resolve_repo_path(path_value: str) -> Path:
    """
    功能：将 CLI 输入解析为仓库内绝对路径。

    Resolve a CLI path against repository root.

    Args:
        path_value: Raw CLI path string.

    Returns:
        Resolved absolute path.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root / candidate).resolve()


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _extract_content_payload(detect_record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(detect_record, dict):
        raise TypeError("detect_record must be dict")
    for key_name in ("content_evidence", "content_evidence_payload"):
        payload = _as_dict(detect_record.get(key_name))
        if payload:
            return payload
    return {}


def _extract_geometry_payload(detect_record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(detect_record, dict):
        raise TypeError("detect_record must be dict")
    for key_name in ("geometry_evidence", "geometry_evidence_payload"):
        payload = _as_dict(detect_record.get(key_name))
        if payload:
            return payload
    return {}


def _build_issue(
    issue: str,
    negative_result_visibility_risk: str,
    evidence: Dict[str, Any],
    severity: str,
    recommended_fix: str,
) -> Dict[str, Any]:
    if not isinstance(issue, str) or not issue:
        raise TypeError("issue must be non-empty str")
    if not isinstance(negative_result_visibility_risk, str) or not negative_result_visibility_risk:
        raise TypeError("negative_result_visibility_risk must be non-empty str")
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be dict")
    if not isinstance(severity, str) or not severity:
        raise TypeError("severity must be non-empty str")
    if not isinstance(recommended_fix, str) or not recommended_fix:
        raise TypeError("recommended_fix must be non-empty str")
    return {
        "issue": issue,
        "negative_result_visibility_risk": negative_result_visibility_risk,
        "evidence": evidence,
        "severity": severity,
        "recommended_fix": recommended_fix,
    }


def _append_issue_once(issues: List[Dict[str, Any]], issue_obj: Dict[str, Any]) -> None:
    if not isinstance(issues, list):
        raise TypeError("issues must be list")
    if not isinstance(issue_obj, dict):
        raise TypeError("issue_obj must be dict")
    issue_name = issue_obj.get("issue")
    evidence = issue_obj.get("evidence")
    for existing in issues:
        if existing.get("issue") == issue_name and existing.get("evidence") == evidence:
            return
    issues.append(issue_obj)


def _load_matrix_summary(run_root: Path) -> Dict[str, Any]:
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    return load_json_dict(run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json")


def _extract_first_failed_matrix_item(matrix_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(matrix_summary, dict):
        raise TypeError("matrix_summary must be dict")
    results = matrix_summary.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if isinstance(item, dict) and item.get("status") != "ok":
            return item
    return {}


def _infer_nearest_failed_stage(
    state: Dict[str, Any],
    matrix_summary: Dict[str, Any],
    preflight: Dict[str, Any],
    workflow_exit_code: int,
) -> str:
    if not isinstance(state, dict):
        raise TypeError("state must be dict")
    if not isinstance(matrix_summary, dict):
        raise TypeError("matrix_summary must be dict")
    if not isinstance(preflight, dict):
        raise TypeError("preflight must be dict")
    if not isinstance(workflow_exit_code, int):
        raise TypeError("workflow_exit_code must be int")

    if not bool(preflight.get("ok", False)):
        return "preflight"

    exists = _as_dict(state.get("exists"))
    if workflow_exit_code == 0:
        return "ok"
    if not bool(exists.get("embed_record", False)):
        return "embed"
    if not bool(exists.get("detect_record", False)):
        return "detect"
    if not bool(exists.get("calibration_record", False)):
        return "calibrate"
    if not bool(exists.get("evaluate_record", False)):
        return "evaluate"
    if int(matrix_summary.get("failed", 0)) > 0 or not bool(exists.get("experiment_matrix_summary", False)):
        return "experiment_matrix"
    if not bool(exists.get("signoff_report", False)):
        return "signoff"
    return "workflow"


def _build_mini_real_validation_summary(
    run_root: Path,
    cfg_path: Path,
    workflow_exit_code: int,
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    if not isinstance(workflow_exit_code, int):
        raise TypeError("workflow_exit_code must be int")
    if not isinstance(preflight, dict):
        raise TypeError("preflight must be dict")

    cfg_obj = load_runtime_config(cfg_path)
    state = collect_workflow_state(run_root)
    run_closure = load_json_dict(run_root / "artifacts" / "run_closure.json")
    matrix_summary = _load_matrix_summary(run_root)
    detect_record = _as_dict(state.get("detect_record"))
    content_payload = _extract_content_payload(detect_record)
    geometry_payload = _extract_geometry_payload(detect_record)
    attestation_payload = _as_dict(detect_record.get("attestation"))
    run_closure_status = _as_dict(run_closure.get("status"))
    matrix_failure = _extract_first_failed_matrix_item(matrix_summary)
    matrix_cfg = _as_dict(cfg_obj.get("experiment_matrix"))
    exists = _as_dict(state.get("exists"))
    paths = dict(_as_dict(state.get("paths")))
    paths["run_closure"] = str(run_root / "artifacts" / "run_closure.json")

    issues: List[Dict[str, Any]] = []
    if not bool(preflight.get("ok", False)):
        _append_issue_once(
            issues,
            _build_issue(
                issue="environment_blocked",
                negative_result_visibility_risk="high",
                evidence={
                    "gpu_tool_available": bool(preflight.get("gpu_tool_available", False)),
                    "missing_attestation_env_vars": preflight.get("missing_attestation_env_vars", []),
                    "nvidia_smi_path": preflight.get("nvidia_smi_path", "<absent>"),
                },
                severity="blocking",
                recommended_fix="补齐 GPU 运行环境与 attestation 所需环境变量后重跑该入口。",
            ),
        )

    run_closure_reason = run_closure_status.get("reason")
    if isinstance(run_closure_reason, str) and run_closure_reason and run_closure_reason != "ok":
        _append_issue_once(
            issues,
            _build_issue(
                issue="run_closure_failed",
                negative_result_visibility_risk="high",
                evidence={
                    "reason": run_closure_reason,
                    "details": run_closure_status.get("details"),
                    "pipeline_build_failure_reason": run_closure.get("pipeline_build_failure_reason"),
                    "path": str(run_root / "artifacts" / "run_closure.json"),
                },
                severity="high",
                recommended_fix="优先查看 run_closure.status.details 与 pipeline_build_failure_reason，按主失败原因修复后再重跑。",
            ),
        )

    content_failure_reason = content_payload.get("content_failure_reason")
    if isinstance(content_failure_reason, str) and content_failure_reason:
        _append_issue_once(
            issues,
            _build_issue(
                issue="detect_content_failed",
                negative_result_visibility_risk="medium",
                evidence={
                    "content_failure_reason": content_failure_reason,
                    "content_status": content_payload.get("status"),
                    "detect_record_path": paths.get("detect_record"),
                },
                severity="medium",
                recommended_fix="检查 detect content 主链输入图像、mask 条件与 content evidence 生成路径。",
            ),
        )

    geometry_failure_reason = geometry_payload.get("geometry_failure_reason")
    if isinstance(geometry_failure_reason, str) and geometry_failure_reason:
        _append_issue_once(
            issues,
            _build_issue(
                issue="detect_geometry_failed",
                negative_result_visibility_risk="medium",
                evidence={
                    "geometry_failure_reason": geometry_failure_reason,
                    "geometry_status": geometry_payload.get("status"),
                    "detect_record_path": paths.get("detect_record"),
                },
                severity="medium",
                recommended_fix="检查 geometry sync、anchor 与 recovered-domain revalidation 路径。",
            ),
        )

    attestation_failure_reason = attestation_payload.get("attestation_failure_reason")
    if isinstance(attestation_failure_reason, str) and attestation_failure_reason:
        _append_issue_once(
            issues,
            _build_issue(
                issue="attestation_failed",
                negative_result_visibility_risk="medium",
                evidence={
                    "attestation_failure_reason": attestation_failure_reason,
                    "detect_record_path": paths.get("detect_record"),
                },
                severity="medium",
                recommended_fix="检查 embed/detect 后置 attestation 构建与验证步骤。",
            ),
        )

    if int(matrix_summary.get("failed", 0)) > 0:
        _append_issue_once(
            issues,
            _build_issue(
                issue="experiment_matrix_failed",
                negative_result_visibility_risk="high",
                evidence={
                    "failed": int(matrix_summary.get("failed", 0)),
                    "total": int(matrix_summary.get("total", 0)),
                    "first_failure_reason": matrix_failure.get("failure_reason", "<absent>"),
                    "grid_summary_path": paths.get("experiment_matrix_summary"),
                },
                severity="high",
                recommended_fix="优先查看 experiment_matrix 的首个 failure_reason，确认是 real neg cache、shared thresholds 还是 forced pair guard 触发。",
            ),
        )
    elif workflow_exit_code != 0 and not bool(exists.get("experiment_matrix_summary", False)):
        _append_issue_once(
            issues,
            _build_issue(
                issue="experiment_matrix_summary_missing",
                negative_result_visibility_risk="high",
                evidence={
                    "workflow_exit_code": workflow_exit_code,
                    "grid_summary_path": paths.get("experiment_matrix_summary"),
                },
                severity="high",
                recommended_fix="检查 onefile workflow 是否在 experiment_matrix 步骤前已失败，并回看前序 records/run_closure。",
            ),
        )

    if not bool(exists.get("thresholds_artifact", False)):
        _append_issue_once(
            issues,
            _build_issue(
                issue="shared_thresholds_artifact_missing",
                negative_result_visibility_risk="high",
                evidence={
                    "thresholds_artifact_path": paths.get("thresholds_artifact"),
                    "require_shared_thresholds": bool(matrix_cfg.get("require_shared_thresholds", False)),
                },
                severity="high",
                recommended_fix="检查 global calibrate 是否拿到了足够的 real negative detect records，并确认 thresholds_artifact.json 已产出。",
            ),
        )

    nearest_failed_stage = _infer_nearest_failed_stage(
        state=state,
        matrix_summary=matrix_summary,
        preflight=preflight,
        workflow_exit_code=workflow_exit_code,
    )
    mini_real_validation_ok = bool(
        bool(preflight.get("ok", False))
        and workflow_exit_code == 0
        and bool(exists.get("thresholds_artifact", False))
        and bool(exists.get("experiment_matrix_summary", False))
        and int(matrix_summary.get("failed", 0)) == 0
        and (run_closure_reason in {None, "ok"})
    )

    return {
        "profile_role": "paper_full_cuda_mini_real_validation",
        "config_path": str(cfg_path),
        "run_root": str(run_root),
        "workflow_exit_code": workflow_exit_code,
        "environment_blocked": not bool(preflight.get("ok", False)),
        "mini_real_validation_ok": mini_real_validation_ok,
        "nearest_failed_stage": nearest_failed_stage,
        "formal_requirements": {
            "require_real_negative_cache": bool(matrix_cfg.get("require_real_negative_cache", False)),
            "require_shared_thresholds": bool(matrix_cfg.get("require_shared_thresholds", False)),
            "disallow_forced_pair_fallback": bool(matrix_cfg.get("disallow_forced_pair_fallback", False)),
            "thresholds_artifact_present": bool(exists.get("thresholds_artifact", False)),
            "experiment_matrix_summary_present": bool(exists.get("experiment_matrix_summary", False)),
        },
        "stage_artifacts": {
            "embed_record": bool(exists.get("embed_record", False)),
            "detect_record": bool(exists.get("detect_record", False)),
            "calibration_record": bool(exists.get("calibration_record", False)),
            "evaluate_record": bool(exists.get("evaluate_record", False)),
            "evaluation_report": bool(exists.get("evaluation_report", False)),
            "run_closure": bool((run_root / "artifacts" / "run_closure.json").exists()),
            "experiment_matrix_summary": bool(exists.get("experiment_matrix_summary", False)),
            "signoff_report": bool(exists.get("signoff_report", False)),
        },
        "issue_count": len(issues),
        "issues": issues,
        "details": {
            "preflight": preflight,
            "run_closure_status": run_closure_status,
            "pipeline_build_failure_reason": run_closure.get("pipeline_build_failure_reason", "<absent>"),
            "detect_content_status": content_payload.get("status", "<absent>"),
            "detect_content_failure_reason": content_failure_reason,
            "detect_geometry_status": geometry_payload.get("status", "<absent>"),
            "detect_geometry_failure_reason": geometry_failure_reason,
            "matrix_total": int(matrix_summary.get("total", 0)),
            "matrix_failed": int(matrix_summary.get("failed", 0)),
            "matrix_first_failure_reason": matrix_failure.get("failure_reason", "<absent>"),
        },
        "paths": paths,
    }


def run_mini_real_validation(
    run_root: Path,
    config_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行 mini real set 结构验证并输出结构化失败摘要。

    Run the mini-real validation workflow and emit a structured failure summary.

    Args:
        run_root: Target workflow run root.
        config_path: Runtime config path for mini-real validation.
        dry_run: Whether to skip subprocess execution.

    Returns:
        Mapping with workflow exit code, summary, and summary path.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be bool")

    preflight = detect_formal_gpu_preflight(config_path)
    workflow_exit_code = 2
    if bool(preflight.get("ok", False)):
        workflow_exit_code = run_onefile_workflow(
            repo_root=_repo_root,
            cfg_path=config_path,
            run_root=run_root,
            profile=PROFILE_PAPER_FULL_CUDA,
            signoff_profile="paper",
            dry_run=dry_run,
        )
    elif dry_run:
        workflow_exit_code = 0

    summary = _build_mini_real_validation_summary(
        run_root=run_root,
        cfg_path=config_path,
        workflow_exit_code=workflow_exit_code,
        preflight=preflight,
    )
    summary_path = write_workflow_summary(run_root, "paper_full_mini_real_validation_summary.json", summary)
    return {
        "workflow_exit_code": workflow_exit_code,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    功能：构建命令行参数解析器。

    Build CLI parser for mini-real validation.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run paper_full mini-real validation and emit structured failure summary."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Mini-real runtime config path (default: configs/paper_full_cuda_mini_real_validation.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT.as_posix()),
        help="Workflow run_root for mini-real validation (default: outputs/onefile_paper_full_cuda_mini_real_validation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print workflow plan only without executing subprocesses.",
    )
    return parser


def main() -> int:
    """
    功能：CLI 主入口。

    Execute mini-real validation and print the structured summary.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)

    result = run_mini_real_validation(
        run_root=run_root,
        config_path=config_path,
        dry_run=bool(args.dry_run),
    )
    summary = result["summary"]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[mini-real] summary_path={result['summary_path']}")

    if bool(summary.get("mini_real_validation_ok", False)):
        return 0
    if bool(summary.get("environment_blocked", False)):
        return 2
    if bool(args.dry_run) and int(result["workflow_exit_code"]) == 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())