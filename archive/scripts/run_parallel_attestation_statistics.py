#!/usr/bin/env python3
"""
文件目的：parallel_attestation_statistics 独立专项流程脚本。
Module type: General module

职责边界：
1. 仅对已完成主 run_root 的 event_attestation_score 执行独立 calibrate/evaluate。
2. 不改变主 Paper_Full_Cuda workflow 的成功口径。
3. 不依赖 CLI score_name override，而是生成专项配置副本后调用标准 CLI。
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_paper_full_cuda as paper_full_cuda_workflow


def _build_parallel_statistics_cfg(
    cfg_obj: Dict[str, Any],
    main_run_root: Path,
    parallel_run_root: Path,
) -> Dict[str, Any]:
    """
    功能：构造 parallel attestation 专项流程配置副本。

    Build a dedicated config copy for the detached parallel attestation
    statistics workflow.

    Args:
        cfg_obj: Source runtime config mapping.
        main_run_root: Completed main workflow run root.
        parallel_run_root: Detached parallel statistics run root.

    Returns:
        Mutated config copy for the detached workflow.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(main_run_root, Path):
        raise TypeError("main_run_root must be Path")
    if not isinstance(parallel_run_root, Path):
        raise TypeError("parallel_run_root must be Path")

    parallel_cfg = paper_full_cuda_workflow._resolve_parallel_attestation_statistics_cfg(cfg_obj)
    cfg_copy = copy.deepcopy(cfg_obj)

    detect_records_glob = str((main_run_root / "records" / "*detect*.json").resolve())
    thresholds_path = str((parallel_run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").resolve())

    calibration_cfg = dict(cfg_copy.get("calibration")) if isinstance(cfg_copy.get("calibration"), dict) else {}
    calibration_cfg["score_name"] = str(parallel_cfg.get("calibration_score_name"))
    calibration_cfg["detect_records_glob"] = detect_records_glob
    cfg_copy["calibration"] = calibration_cfg

    evaluate_cfg = dict(cfg_copy.get("evaluate")) if isinstance(cfg_copy.get("evaluate"), dict) else {}
    evaluate_cfg["score_name"] = str(parallel_cfg.get("evaluate_score_name"))
    evaluate_cfg["detect_records_glob"] = detect_records_glob
    evaluate_cfg["thresholds_path"] = thresholds_path
    cfg_copy["evaluate"] = evaluate_cfg

    return cfg_copy


def _write_parallel_statistics_cfg(parallel_run_root: Path, cfg_obj: Dict[str, Any]) -> Path:
    """
    功能：写出 parallel attestation 专项流程配置副本。

    Persist the detached parallel statistics config copy under runtime_metadata.

    Args:
        parallel_run_root: Detached parallel statistics run root.
        cfg_obj: Config copy to persist.

    Returns:
        Path to the written YAML config copy.
    """
    if not isinstance(parallel_run_root, Path):
        raise TypeError("parallel_run_root must be Path")
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    cfg_path = parallel_run_root / "runtime_metadata" / "parallel_attestation_statistics_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg_path


def _build_parallel_stage_command(stage_name: str, config_path: Path, parallel_run_root: Path) -> List[str]:
    """
    功能：构造 parallel attestation 专项流程命令。

    Build the CLI command for one detached parallel statistics stage.

    Args:
        stage_name: Stage name in {calibrate, evaluate}.
        config_path: Detached config copy path.
        parallel_run_root: Detached parallel statistics run root.

    Returns:
        CLI command argument list.
    """
    if stage_name not in {"calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(parallel_run_root, Path):
        raise TypeError("parallel_run_root must be Path")

    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(parallel_run_root),
        "--config",
        str(config_path),
    ]
    command.extend(
        paper_full_cuda_workflow._build_stage_overrides(
            f"parallel_attestation_{stage_name}",
            None,
        )
    )
    return command


def _run_step(step_name: str, command: List[str]) -> int:
    """
    功能：执行专项流程步骤并输出结构化日志。

    Execute one detached parallel statistics step.

    Args:
        step_name: Stable step name.
        command: CLI command arguments.

    Returns:
        Process return code.
    """
    if not isinstance(step_name, str) or not step_name:
        raise TypeError("step_name must be non-empty str")
    if not isinstance(command, list) or not command:
        raise TypeError("command must be non-empty list")

    command_list = [str(item) for item in command]
    print(f"[parallel_attestation_statistics] step_start={step_name}")
    print(f"[parallel_attestation_statistics] command={' '.join(command_list)}")
    result = subprocess.run(command_list, cwd=str(paper_full_cuda_workflow.REPO_ROOT), check=False)
    print(f"[parallel_attestation_statistics] step_end={step_name} return_code={result.returncode}")
    return int(result.returncode)


def _persist_parallel_summary(parallel_run_root: Path, summary_obj: Dict[str, Any]) -> None:
    """
    功能：写出 parallel attestation 专项流程摘要。

    Persist the detached workflow summary under artifacts.

    Args:
        parallel_run_root: Detached parallel statistics run root.
        summary_obj: Summary mapping.

    Returns:
        None.
    """
    if not isinstance(parallel_run_root, Path):
        raise TypeError("parallel_run_root must be Path")
    if not isinstance(summary_obj, dict):
        raise TypeError("summary_obj must be dict")

    summary_path = parallel_run_root / "artifacts" / "parallel_attestation_statistics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_obj, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def run_parallel_attestation_statistics(config_path: Path, run_root: Path) -> int:
    """
    功能：执行独立的 parallel attestation 统计专项流程。

    Execute the detached parallel attestation statistics workflow.

    Args:
        config_path: Source runtime config path.
        run_root: Completed main workflow run root.

    Returns:
        Process exit code.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    cfg_obj = paper_full_cuda_workflow._load_runtime_config(config_path)
    parallel_cfg = paper_full_cuda_workflow._resolve_parallel_attestation_statistics_cfg(cfg_obj)
    parallel_run_root = paper_full_cuda_workflow._build_parallel_attestation_run_root(run_root)

    summary: Dict[str, Any] = {
        "mode": "parallel_attestation_statistics_only",
        "source_run_root": str(run_root),
        "parallel_run_root": str(parallel_run_root),
        "config_path": str(config_path),
        "parallel_cfg_enabled": bool(parallel_cfg.get("enabled", False)),
        "score_names": {
            "calibration": str(parallel_cfg.get("calibration_score_name")),
            "evaluate": str(parallel_cfg.get("evaluate_score_name")),
        },
        "stages": {},
        "status": "running",
    }

    detached_cfg = _build_parallel_statistics_cfg(cfg_obj, run_root, parallel_run_root)
    detached_cfg_path = _write_parallel_statistics_cfg(parallel_run_root, detached_cfg)
    summary["detached_config_path"] = str(detached_cfg_path)

    stage_plan = [
        ("calibrate", _build_parallel_stage_command("calibrate", detached_cfg_path, parallel_run_root)),
        ("evaluate", _build_parallel_stage_command("evaluate", detached_cfg_path, parallel_run_root)),
    ]
    for stage_name, command in stage_plan:
        return_code = _run_step(stage_name, command)
        summary["stages"][stage_name] = {"return_code": return_code}
        if return_code != 0:
            summary["status"] = "failed"
            _persist_parallel_summary(parallel_run_root, summary)
            print("[parallel_attestation_statistics] summary=")
            print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
            return return_code

    summary["status"] = "ok"
    _persist_parallel_summary(parallel_run_root, summary)
    print("[parallel_attestation_statistics] summary=")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    """
    功能：CLI 主入口。

    Execute the detached parallel attestation statistics workflow.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run detached parallel_attestation_statistics for paper_full_cuda outputs.",
    )
    parser.add_argument(
        "--config",
        default=str(paper_full_cuda_workflow.DEFAULT_CONFIG_PATH.as_posix()),
        help="Runtime config path (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(paper_full_cuda_workflow.DEFAULT_RUN_ROOT.as_posix()),
        help="Completed main workflow run root (default: outputs/colab_run_paper_full_cuda)",
    )
    args = parser.parse_args()

    config_path = paper_full_cuda_workflow._resolve_repo_path(args.config)
    run_root = paper_full_cuda_workflow._resolve_repo_path(args.run_root)
    print(json.dumps({
        "config_path": str(config_path),
        "run_root": str(run_root),
        "mode": "parallel_attestation_statistics_only",
    }, indent=2, ensure_ascii=False))
    return run_parallel_attestation_statistics(config_path, run_root)


if __name__ == "__main__":
    sys.exit(main())