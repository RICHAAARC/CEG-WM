#!/usr/bin/env python3
"""
文件目的：CPU smoke 闭环验收脚本。
Module type: General module

职责边界：
1. 复用 onefile formal wiring，执行轻量 profile 的 embed → detect → calibrate → evaluate → audits → signoff。
2. 默认使用 synthetic pipeline smoke 配置，不依赖真实 SD3.5 权重。
3. 输出脚本级结构化摘要，不改写正式 records schema。
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

from scripts.run_onefile_workflow import PROFILE_CPU_SMOKE, run_onefile_workflow
from scripts.workflow_acceptance_common import build_cpu_smoke_summary, write_workflow_summary


DEFAULT_CONFIG_PATH = Path("configs/smoke_cpu.yaml")
DEFAULT_RUN_ROOT = Path("outputs/onefile_cpu_smoke_verify")


def _resolve_repo_path(path_value: str) -> Path:
    """
    功能：将 CLI 输入解析为仓库内绝对路径。

    Resolve a CLI path against repository root.

    Args:
        path_value: Raw CLI path string.

    Returns:
        Resolved absolute path.
    """
    if not path_value.strip():
        raise TypeError("path_value must be non-empty str")
    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root / candidate).resolve()


def run_cpu_smoke_verification(
    run_root: Path,
    config_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行 CPU smoke 闭环验收。

    Run CPU-first smoke verification using onefile workflow wiring.

    Args:
        run_root: Target workflow run root.
        config_path: Smoke runtime config path.
        dry_run: Whether to skip subprocess execution.

    Returns:
        Mapping with workflow exit code, summary, and summary path.
    """
    workflow_exit_code = run_onefile_workflow(
        repo_root=_repo_root,
        cfg_path=config_path,
        run_root=run_root,
        profile=PROFILE_CPU_SMOKE,
        signoff_profile="baseline",
        dry_run=dry_run,
        device_override="cpu",
    )
    summary = build_cpu_smoke_summary(run_root, config_path, workflow_exit_code)
    summary_path = write_workflow_summary(run_root, "cpu_smoke_summary.json", summary)
    return {
        "workflow_exit_code": workflow_exit_code,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    功能：构建命令行参数解析器。

    Build CLI parser for CPU smoke verification.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run CPU smoke closure verification using onefile workflow wiring."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.as_posix()),
        help="Smoke runtime config path (default: configs/smoke_cpu.yaml)",
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT.as_posix()),
        help="Workflow run_root for smoke verification (default: outputs/onefile_cpu_smoke_verify)",
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

    Execute CPU smoke verification and emit structured summary.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run_root = _resolve_repo_path(args.run_root)

    result = run_cpu_smoke_verification(
        run_root=run_root,
        config_path=config_path,
        dry_run=bool(args.dry_run),
    )
    summary = result["summary"]

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[cpu-smoke] summary_path={result['summary_path']}")

    if bool(summary.get("smoke_verdict", False)):
        return 0
    if bool(args.dry_run) and int(result["workflow_exit_code"]) == 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
