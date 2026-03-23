#!/usr/bin/env python3
"""
文件目的：paper_full GPU 正式验收脚本。
Module type: General module

职责边界：
1. 复用 onefile 正式主链，执行 paper_full_cuda profile 的 formal 验收。
2. 在执行前检查 GPU 工具与 attestation 环境变量，区分环境阻断与代码问题。
3. 输出脚本级结构化摘要，不引入平行正式 workflow。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.run_onefile_workflow import PROFILE_PAPER_FULL_CUDA, run_onefile_workflow
from scripts.workflow_acceptance_common import (
    build_formal_gpu_summary,
    detect_formal_gpu_preflight,
    write_workflow_summary,
)


DEFAULT_CONFIG_PATH = Path("configs/paper_full_cuda.yaml")
DEFAULT_RUN_ROOT = Path("outputs/onefile_paper_full_cuda_verify")


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


def run_paper_full_workflow_verification(
    run_root: Path,
    config_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行 paper_full GPU 正式验收。

    Run formal paper_full workflow verification.

    Args:
        run_root: Target workflow run root.
        config_path: Formal runtime config path.
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
    elif bool(dry_run):
        workflow_exit_code = 0

    summary = build_formal_gpu_summary(
        run_root=run_root,
        cfg_path=config_path,
        workflow_exit_code=workflow_exit_code,
        preflight=preflight,
    )
    summary_path = write_workflow_summary(run_root, "paper_full_formal_summary.json", summary)
    return {
        "workflow_exit_code": workflow_exit_code,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    功能：构建命令行参数解析器。

    Build CLI parser for paper_full formal verification.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run formal paper_full CUDA workflow verification."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Formal runtime config path (default: configs/paper_full_cuda.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT.as_posix()),
        help="Workflow run_root for formal verification (default: outputs/onefile_paper_full_cuda_verify)",
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

    Execute paper_full formal verification and emit structured summary.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)

    result = run_paper_full_workflow_verification(
        run_root=run_root,
        config_path=config_path,
        dry_run=bool(args.dry_run),
    )
    summary = result["summary"]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[paper-full] summary_path={result['summary_path']}")

    if bool(summary.get("formal_output_expectation_ok", False)):
        return 0
    if bool(summary.get("environment_blocked", False)):
        return 2
    if bool(args.dry_run) and int(result["workflow_exit_code"]) == 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
