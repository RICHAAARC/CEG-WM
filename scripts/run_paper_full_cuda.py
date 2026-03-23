#!/usr/bin/env python3
"""
文件目的：paper_full_cuda 项目输出导向编排脚本。
Module type: General module

职责边界：
1. 仅顺序编排 embed/detect/calibrate/evaluate 与可选 experiment_matrix。
2. 不执行 formal acceptance、signoff、workflow summary 或审计补洞。
3. 不改写论文机制身份字段，直接消费 configs/paper_full_cuda.yaml。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG_PATH = Path("configs/paper_full_cuda.yaml")
DEFAULT_RUN_ROOT = Path("outputs/colab_run_paper_full_cuda")


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


def _build_parallel_attestation_command(
    stage_name: str,
    config_path: Path,
    run_root: Path,
    parallel_run_root: Path,
    score_name: str,
) -> List[str]:
    """
    功能：构造 parallel attestation 统计子流程命令。

    Build a parallel attestation statistics command by reusing the formal CLI
    stages with score-name overrides only.

    Args:
        stage_name: Stage name in {calibrate, evaluate}.
        config_path: Runtime config path.
        run_root: Main workflow run root containing the source detect record.
        parallel_run_root: Parallel statistics run root.
        score_name: Target score name.

    Returns:
        CLI command argument list.
    """
    if stage_name not in {"calibrate", "evaluate"}:
        raise ValueError(f"unsupported parallel attestation stage_name: {stage_name}")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    source_detect_glob = (run_root / "records" / "*detect*.json").resolve().as_posix()
    extra_overrides = [
        _format_override_arg(f"{ 'calibration' if stage_name == 'calibrate' else 'evaluate' }.score_name", score_name),
        _format_override_arg(f"{ 'calibration' if stage_name == 'calibrate' else 'evaluate' }.detect_records_glob", source_detect_glob),
    ]
    if stage_name == "evaluate":
        thresholds_path = (parallel_run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").resolve().as_posix()
        extra_overrides.append(_format_override_arg("evaluate.thresholds_path", thresholds_path))
    return _build_stage_command(stage_name, config_path, parallel_run_root, extra_overrides)


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
        "main_stages": {},
        "parallel_attestation_statistics": {
            "enabled": bool(parallel_cfg.get("enabled", False)),
            "status": "skipped",
            "run_root": None,
            "stages": {},
        },
        "experiment_matrix": {
            "enabled": isinstance(cfg_obj.get("experiment_matrix"), dict),
            "status": "skipped",
            "return_code": None,
        },
        "exit_policy": "embed_detect_calibrate_evaluate_and_parallel_required_experiment_matrix_optional",
    }
    stages = ["embed", "detect", "calibrate", "evaluate"]
    for stage_name in stages:
        return_code = _run_step(stage_name, _build_stage_command(stage_name, config_path, run_root))
        workflow_summary["main_stages"][stage_name] = {"return_code": return_code}
        if return_code != 0:
            workflow_summary["main_status"] = "failed"
            _emit_workflow_summary(workflow_summary)
            return return_code

    workflow_summary["main_status"] = "ok"

    if bool(parallel_cfg.get("enabled", False)):
        parallel_run_root = _build_parallel_attestation_run_root(run_root)
        parallel_summary = workflow_summary["parallel_attestation_statistics"]
        parallel_summary["run_root"] = str(parallel_run_root)
        parallel_summary["status"] = "running"
        parallel_plan = [
            ("parallel_attestation_calibrate", "calibrate", str(parallel_cfg.get("calibration_score_name"))),
            ("parallel_attestation_evaluate", "evaluate", str(parallel_cfg.get("evaluate_score_name"))),
        ]
        for summary_stage_name, cli_stage_name, score_name in parallel_plan:
            return_code = _run_step(
                summary_stage_name,
                _build_parallel_attestation_command(
                    cli_stage_name,
                    config_path,
                    run_root,
                    parallel_run_root,
                    score_name,
                ),
            )
            parallel_summary["stages"][summary_stage_name] = {
                "return_code": return_code,
                "score_name": score_name,
            }
            if return_code != 0:
                parallel_summary["status"] = "failed"
                workflow_summary["main_status"] = "failed_parallel_attestation_statistics"
                _emit_workflow_summary(workflow_summary)
                return return_code
        parallel_summary["status"] = "ok"

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
            print("[paper_full_cuda] experiment_matrix failed but main workflow outputs are preserved.")

    _emit_workflow_summary(workflow_summary)

    return 0


def main() -> int:
    """
    功能：CLI 主入口。

    Execute the output-oriented paper_full_cuda workflow.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the paper_full_cuda output-oriented workflow."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Runtime config path (default: configs/paper_full_cuda.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT.as_posix()),
        help="Workflow run root (default: outputs/colab_run_paper_full_cuda)",
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