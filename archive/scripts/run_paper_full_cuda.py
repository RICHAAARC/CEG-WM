#!/usr/bin/env python3
"""
文件目的：paper_full_cuda 项目输出导向编排脚本。
Module type: General module

职责边界：
1. 仅顺序编排 embed/detect/calibrate/evaluate 与可选 experiment_matrix。
2. 不执行 formal acceptance、signoff、workflow summary 或审计补洞。
3. 不改写论文机制身份字段，直接消费 configs/default.yaml。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_ROOT = Path("outputs/colab_run_default")


def _resolve_repo_path(path_value: str) -> Path:
    """
    功能：将 CLI 路径解析为仓库内绝对路径。

    Resolve a CLI path against the repository root.

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
    return (REPO_ROOT / candidate).resolve()


def _load_runtime_config(config_path: Path) -> Dict[str, object]:
    """
    功能：加载运行期配置。

    Load the runtime YAML config used by the paper_full_cuda workflow.

    Args:
        config_path: Runtime config path.

    Returns:
        Loaded runtime config mapping.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    cfg_obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_obj, dict):
        raise ValueError("paper_full_cuda config root must be mapping")
    return dict(cfg_obj)


def _format_override_arg(field_path: str, value: Any) -> str:
    """
    功能：构造稳定的 CLI override 参数。

    Build a stable CLI override argument.

    Args:
        field_path: Config field path.
        value: Override value.

    Returns:
        Serialized override string.
    """
    if not isinstance(field_path, str) or not field_path:
        raise TypeError("field_path must be non-empty str")
    return f"{field_path}={json.dumps(value, ensure_ascii=False)}"


def _resolve_parallel_attestation_statistics_cfg(cfg_obj: Dict[str, object]) -> Dict[str, object]:
    """
    功能：解析 parallel_attestation_statistics 配置。

    Resolve the output-only parallel attestation statistics configuration.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Normalized parallel-attestation config mapping.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    section_node = cfg_obj.get("parallel_attestation_statistics")
    section = dict(section_node) if isinstance(section_node, dict) else {}
    enabled_value = section.get("enabled", False)
    enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False
    calibration_score_name = section.get("calibration_score_name", "event_attestation_score")
    evaluate_score_name = section.get("evaluate_score_name", "event_attestation_score")
    if enabled:
        if not isinstance(calibration_score_name, str) or not calibration_score_name:
            raise ValueError("parallel_attestation_statistics.calibration_score_name must be non-empty str")
        if not isinstance(evaluate_score_name, str) or not evaluate_score_name:
            raise ValueError("parallel_attestation_statistics.evaluate_score_name must be non-empty str")
    return {
        "enabled": enabled,
        "calibration_score_name": calibration_score_name,
        "evaluate_score_name": evaluate_score_name,
    }


def _build_stage_overrides(stage_name: str, extra_overrides: Sequence[str] | None = None) -> List[str]:
    """
    功能：构造阶段 override 参数列表。

    Build CLI override arguments for a workflow stage.

    Args:
        stage_name: Stage name.
        extra_overrides: Optional additional override strings.

    Returns:
        Flattened CLI override argument list.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")

    override_items = [
        _format_override_arg("run_root_reuse_allowed", True),
        _format_override_arg("run_root_reuse_reason", f"paper_full_cuda_{stage_name}"),
    ]
    if extra_overrides is not None:
        override_items.extend(str(item) for item in extra_overrides)

    command_args: List[str] = []
    for override_arg in override_items:
        command_args.extend(["--override", override_arg])
    return command_args


def _build_stage_command(
    stage_name: str,
    config_path: Path,
    run_root: Path,
    extra_overrides: Sequence[str] | None = None,
) -> List[str]:
    """
    功能：构造单阶段 CLI 命令。

    Build the command for one formal CLI stage.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.
        config_path: Runtime config path.
        run_root: Workflow run root.
        extra_overrides: Optional extra override strings.

    Returns:
        CLI command argument list.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    command.extend(_build_stage_overrides(stage_name, extra_overrides))
    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])
    return command


def _build_parallel_attestation_run_root(run_root: Path) -> Path:
    """
    功能：构造 parallel attestation 统计子流程输出目录。

    Build the output root for parallel attestation statistics.

    Args:
        run_root: Main workflow run root.

    Returns:
        Parallel attestation run root.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    return run_root / "outputs" / "parallel_attestation_statistics"


def _build_experiment_matrix_command(config_path: Path, run_root: Path) -> List[str]:
    """
    功能：构造 experiment_matrix 命令。

    Build the optional experiment_matrix batch command.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.

    Returns:
        CLI command argument list.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment_matrix.py"),
        "--config",
        str(config_path),
        "--batch-root",
        str(run_root / "outputs" / "experiment_matrix"),
    ]


def _emit_workflow_summary(summary_obj: Dict[str, Any]) -> None:
    """
    功能：输出结构化 workflow 摘要。

    Emit a structured workflow summary to stdout.

    Args:
        summary_obj: Summary mapping.

    Returns:
        None.
    """
    if not isinstance(summary_obj, dict):
        raise TypeError("summary_obj must be dict")
    print("[paper_full_cuda] workflow_summary=")
    print(json.dumps(summary_obj, indent=2, ensure_ascii=False, sort_keys=True))


def _persist_workflow_summary(run_root: Path, summary_obj: Dict[str, Any]) -> None:
    """
    功能：将 workflow 摘要写入 artifacts/workflow_summary.json。

    Persist the workflow summary as a JSON artifact under the run root.

    Args:
        run_root: Workflow run root.
        summary_obj: Workflow summary mapping.

    Returns:
        None.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(summary_obj, dict):
        raise TypeError("summary_obj must be dict")

    summary_path = run_root / "artifacts" / "workflow_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_obj, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _read_json_dict_if_exists(path_obj: Path) -> Dict[str, Any] | None:
    """
    功能：读取存在的 JSON 对象文件。

    Read a JSON object file when it exists.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed mapping when the file exists and is valid; otherwise None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return None
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _resolve_main_attestation_evidence(cfg_obj: Dict[str, object], run_root: Path) -> Dict[str, Any]:
    """
    功能：核查主链 attestation 是否真实参与并留下证据。

    Verify that the main workflow produced attestation evidence rather than only
    carrying an enabled configuration flag.

    Args:
        cfg_obj: Runtime config mapping.
        run_root: Workflow run root.

    Returns:
        Attestation evidence summary with status and missing items.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    attestation_cfg = cfg_obj.get("attestation")
    attestation_enabled = False
    if isinstance(attestation_cfg, dict):
        enabled_value = attestation_cfg.get("enabled")
        attestation_enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False

    summary: Dict[str, Any] = {
        "enabled": attestation_enabled,
        "status": "skipped_disabled",
        "missing": [],
        "artifacts": {},
        "detect_record_attestation_present": False,
        "event_attestation_score_present": False,
        "bundle_verification_present": False,
        "trace_artifact_count": 0,
    }
    if not attestation_enabled:
        return summary

    attestation_dir = run_root / "artifacts" / "attestation"
    detect_record_path = run_root / "records" / "detect_record.json"
    result_path = attestation_dir / "attestation_result.json"
    statement_path = attestation_dir / "attestation_statement.json"
    bundle_path = attestation_dir / "attestation_bundle.json"
    trace_paths = sorted(attestation_dir.glob("*attestation_trace.json")) if attestation_dir.exists() else []

    detect_record = _read_json_dict_if_exists(detect_record_path)
    result_payload = _read_json_dict_if_exists(result_path)

    detect_attestation = detect_record.get("attestation") if isinstance(detect_record, dict) else None
    summary["detect_record_attestation_present"] = isinstance(detect_attestation, dict)

    final_decision = result_payload.get("final_event_attested_decision") if isinstance(result_payload, dict) else None
    event_score = final_decision.get("event_attestation_score") if isinstance(final_decision, dict) else None
    summary["event_attestation_score_present"] = isinstance(event_score, (int, float)) and not isinstance(event_score, bool) and math.isfinite(float(event_score))

    bundle_verification = result_payload.get("bundle_verification") if isinstance(result_payload, dict) else None
    summary["bundle_verification_present"] = isinstance(bundle_verification, dict)
    summary["trace_artifact_count"] = len(trace_paths)
    summary["artifacts"] = {
        "attestation_dir": str(attestation_dir),
        "attestation_result_json": result_path.exists(),
        "attestation_statement_json": statement_path.exists(),
        "attestation_bundle_json": bundle_path.exists(),
        "trace_files": [path_obj.name for path_obj in trace_paths],
    }

    missing_items: List[str] = []
    if not summary["detect_record_attestation_present"]:
        missing_items.append("detect_record.attestation")
    if not summary["event_attestation_score_present"]:
        missing_items.append("attestation_result.final_event_attested_decision.event_attestation_score")
    if not result_path.exists():
        missing_items.append("artifacts/attestation/attestation_result.json")
    if not statement_path.exists():
        missing_items.append("artifacts/attestation/attestation_statement.json")
    if not bundle_path.exists():
        missing_items.append("artifacts/attestation/attestation_bundle.json")
    if not summary["bundle_verification_present"]:
        missing_items.append("attestation_result.bundle_verification")
    if len(trace_paths) <= 0:
        missing_items.append("artifacts/attestation/*attestation_trace.json")

    summary["missing"] = missing_items
    summary["status"] = "ok" if len(missing_items) == 0 else "missing_evidence"
    return summary


def _run_step(step_name: str, command: Sequence[str]) -> int:
    """
    功能：执行单个编排步骤并输出结构化日志。

    Execute one orchestration step with structured logging.

    Args:
        step_name: Stable step name.
        command: CLI command sequence.

    Returns:
        Process return code.
    """
    if not isinstance(step_name, str) or not step_name:
        raise TypeError("step_name must be non-empty str")
    if not isinstance(command, Sequence) or not command:
        raise TypeError("command must be non-empty sequence")

    command_list = [str(item) for item in command]
    print(f"[paper_full_cuda] step_start={step_name}")
    print(f"[paper_full_cuda] command={' '.join(command_list)}")
    result = subprocess.run(command_list, cwd=str(REPO_ROOT), check=False)
    print(f"[paper_full_cuda] step_end={step_name} return_code={result.returncode}")
    return int(result.returncode)


def run_paper_full_cuda(config_path: Path, run_root: Path) -> int:
    """
    功能：执行 paper_full_cuda 项目输出导向工作流。

    Run the paper_full_cuda output-oriented workflow without formal acceptance or signoff.

    Args:
        config_path: Runtime config path.
        run_root: Workflow run root.

    Returns:
        Process exit code.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    cfg_obj = _load_runtime_config(config_path)
    parallel_cfg = _resolve_parallel_attestation_statistics_cfg(cfg_obj)
    workflow_summary: Dict[str, Any] = {
        "mode": "project_outputs_only",
        "workflow_status": "running",
        "main_stages": {},
        "main_chain": {
            "required_stages": ["embed", "detect", "calibrate", "evaluate"],
            "status": "running",
            "attestation": {
                "enabled": bool(isinstance(cfg_obj.get("attestation"), dict) and cfg_obj.get("attestation", {}).get("enabled") is True),
                "status": "pending",
            },
        },
        "parallel_attestation_statistics": {
            "enabled": bool(parallel_cfg.get("enabled", False)),
            "required": False,
            "affects_exit_code": False,
            "status": "detached_not_run" if bool(parallel_cfg.get("enabled", False)) else "disabled",
            "execution_mode": "independent_post_flow",
            "script_path": "scripts/run_parallel_attestation_statistics.py",
            "suggested_run_root": str(_build_parallel_attestation_run_root(run_root)),
            "configured_score_names": {
                "calibration": str(parallel_cfg.get("calibration_score_name")),
                "evaluate": str(parallel_cfg.get("evaluate_score_name")),
            },
        },
        "experiment_matrix": {
            "enabled": isinstance(cfg_obj.get("experiment_matrix"), dict),
            "required": False,
            "status": "skipped",
            "return_code": None,
        },
        "exit_policy": "embed_detect_calibrate_evaluate_required_parallel_attestation_detached_experiment_matrix_optional",
    }
    stages = ["embed", "detect", "calibrate", "evaluate"]
    for stage_name in stages:
        return_code = _run_step(stage_name, _build_stage_command(stage_name, config_path, run_root))
        workflow_summary["main_stages"][stage_name] = {"return_code": return_code}
        if return_code != 0:
            workflow_summary["main_status"] = "failed"
            workflow_summary["main_chain"]["status"] = "failed"
            workflow_summary["workflow_status"] = "failed"
            _persist_workflow_summary(run_root, workflow_summary)
            _emit_workflow_summary(workflow_summary)
            return return_code

    attestation_summary = _resolve_main_attestation_evidence(cfg_obj, run_root)
    workflow_summary["main_chain"]["attestation"] = attestation_summary
    if attestation_summary.get("status") != "ok":
        workflow_summary["main_status"] = "failed_attestation_evidence"
        workflow_summary["main_chain"]["status"] = "failed_attestation_evidence"
        workflow_summary["workflow_status"] = "failed"
        _persist_workflow_summary(run_root, workflow_summary)
        _emit_workflow_summary(workflow_summary)
        return 1

    workflow_summary["main_status"] = "ok"
    workflow_summary["main_chain"]["status"] = "ok"
    workflow_summary["workflow_status"] = "ok"

    matrix_cfg_obj = cfg_obj.get("experiment_matrix")
    matrix_enabled = isinstance(matrix_cfg_obj, dict)
    if matrix_enabled:
        return_code = _run_step("experiment_matrix", _build_experiment_matrix_command(config_path, run_root))
        workflow_summary["experiment_matrix"] = {
            "enabled": True,
            "status": "ok" if return_code == 0 else "failed",
            "return_code": return_code,
        }
        if return_code != 0:
            workflow_summary["workflow_status"] = "ok_with_optional_failures"
            print("[default] experiment_matrix failed but main workflow outputs are preserved.")

    _persist_workflow_summary(run_root, workflow_summary)
    _emit_workflow_summary(workflow_summary)

    return 0


def main() -> int:
    """
    功能：CLI 主入口。

    Execute the output-oriented default workflow.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the default output-oriented workflow."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Runtime config path (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT.as_posix()),
        help="Workflow run root (default: outputs/colab_run_default)",
    )
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)

    result = {
        "config_path": str(config_path),
        "run_root": str(run_root),
        "mode": "project_outputs_only",
        "formal_acceptance": False,
        "signoff": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return run_paper_full_cuda(config_path, run_root)


if __name__ == "__main__":
    sys.exit(main())