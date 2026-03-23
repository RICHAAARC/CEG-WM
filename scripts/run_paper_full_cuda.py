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
from typing import Dict, List, Sequence

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


def _build_stage_command(stage_name: str, config_path: Path, run_root: Path) -> List[str]:
    """
    功能：构造单阶段 CLI 命令。

    Build the command for one formal CLI stage.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.
        config_path: Runtime config path.
        run_root: Workflow run root.

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
        "--override",
        "run_root_reuse_allowed=true",
        "--override",
        f'run_root_reuse_reason="paper_full_cuda_{stage_name}"',
    ]
    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])
    return command


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
    stages = ["embed", "detect", "calibrate", "evaluate"]
    for stage_name in stages:
        return_code = _run_step(stage_name, _build_stage_command(stage_name, config_path, run_root))
        if return_code != 0:
            return return_code

    matrix_cfg_obj = cfg_obj.get("experiment_matrix")
    matrix_enabled = isinstance(matrix_cfg_obj, dict)
    if matrix_enabled:
        return_code = _run_step("experiment_matrix", _build_experiment_matrix_command(config_path, run_root))
        if return_code != 0:
            return return_code

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