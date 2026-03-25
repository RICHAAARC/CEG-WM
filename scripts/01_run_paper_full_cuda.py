"""
文件目的：01_run_paper_full_cuda.py 作为 01_Paper_Full_Cuda 的正式主链脚本入口。
Module type: General module

职责边界：
1. 仅顺序编排 embed、detect、calibrate、evaluate 四段正式主链。
2. 不自动执行 parallel attestation statistics 或 experiment matrix。
3. 仅复用 main.cli 现有入口，不改写 main/ 内部机制逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    run_command_with_logs,
    write_json_atomic,
)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RUN_ROOT = Path("outputs/colab_run_paper_full_cuda")


def _resolve_repo_path(path_value: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _format_override_arg(arg_name: str, value: Any) -> str:
    if not isinstance(arg_name, str) or not arg_name:
        raise TypeError("arg_name must be non-empty str")
    return f"{arg_name}={json.dumps(value, ensure_ascii=False)}"


def _build_stage_overrides(stage_name: str, extra_overrides: Optional[Sequence[str]] = None) -> List[str]:
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
    extra_overrides: Optional[Sequence[str]] = None,
) -> List[str]:
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])
    command.extend(_build_stage_overrides(stage_name, extra_overrides))
    return command


def _required_artifacts(run_root: Path) -> Dict[str, Path]:
    return {
        "embed_record": run_root / "records" / "embed_record.json",
        "detect_record": run_root / "records" / "detect_record.json",
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "threshold_metadata_artifact": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "run_closure": run_root / "artifacts" / "run_closure.json",
        "workflow_summary": run_root / "artifacts" / "workflow_summary.json",
    }


def _artifact_presence(artifact_paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {
        key_name: {
            "path": normalize_path_value(path_obj),
            "exists": bool(path_obj.exists() and path_obj.is_file()),
        }
        for key_name, path_obj in artifact_paths.items()
    }


def _all_required_present(artifact_summary: Dict[str, Dict[str, Any]]) -> bool:
    return all(bool(item.get("exists", False)) for item in artifact_summary.values())


def _run_stage(stage_name: str, command: Sequence[str], run_root: Path) -> Dict[str, Any]:
    logs_dir = ensure_directory(run_root / "logs")
    result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=logs_dir / f"{stage_name}_stdout.log",
        stderr_log_path=logs_dir / f"{stage_name}_stderr.log",
    )
    result["status"] = "ok" if result["return_code"] == 0 else "failed"
    return result


def run_paper_full_cuda(config_path: Path, run_root: Path) -> int:
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    cfg_obj = load_yaml_mapping(config_path)
    ensure_directory(run_root)
    ensure_directory(run_root / "artifacts")

    summary: Dict[str, Any] = {
        "stage_name": "01_Paper_Full_Cuda_mainline",
        "config_path": normalize_path_value(config_path),
        "run_root": normalize_path_value(run_root),
        "status": "running",
        "parallel_attestation_statistics": {
            "enabled_in_default_config": bool(
                isinstance(cfg_obj.get("parallel_attestation_statistics"), dict)
                and bool(cast(Dict[str, Any], cfg_obj.get("parallel_attestation_statistics", {})).get("enabled", False))
            ),
            "status": "detached_not_run",
            "execution_mode": "independent_post_flow",
        },
        "experiment_matrix": {
            "status": "detached_not_run",
            "execution_mode": "independent_post_flow",
        },
        "stages": {},
    }

    stage_plan = [
        ("embed", _build_stage_command("embed", config_path, run_root)),
        ("detect", _build_stage_command("detect", config_path, run_root)),
        ("calibrate", _build_stage_command("calibrate", config_path, run_root)),
        ("evaluate", _build_stage_command("evaluate", config_path, run_root)),
    ]

    for stage_name, command in stage_plan:
        result = _run_stage(stage_name, command, run_root)
        summary["stages"][stage_name] = result
        if result["return_code"] != 0:
            summary["status"] = "failed"
            write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary)
            return int(result["return_code"])

    artifact_summary = _artifact_presence(_required_artifacts(run_root))
    summary["required_artifacts"] = artifact_summary
    summary["required_artifacts_ok"] = _all_required_present(artifact_summary)
    summary["status"] = "ok" if summary["required_artifacts_ok"] else "failed"
    write_json_atomic(run_root / "artifacts" / "workflow_summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stage-01 Paper_Full_Cuda mainline workflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Runtime config path.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT.as_posix()), help="Workflow run root.")
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)
    return run_paper_full_cuda(config_path, run_root)


if __name__ == "__main__":
    sys.exit(main())