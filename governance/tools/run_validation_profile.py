"""按变更范围运行 CEG-WM 的登记验证档位。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shlex
import subprocess
import sys

PROFILE_NAMES = ("governance", "method", "full")


def commands_for_profile(
    profile: str,
    python_executable: str = sys.executable,
) -> tuple[tuple[str, ...], ...]:
    """返回指定验证档位的顺序命令。"""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown validation profile: {profile}")

    project_pytest = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "tests",
    )
    governance_pytest = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "-c",
        "governance/pytest.ini",
        "governance/tests",
    )
    harness = (
        python_executable,
        "governance/harness/run_all_audits.py",
    )

    if profile == "governance":
        return governance_pytest, harness
    if profile == "method":
        return project_pytest, harness
    return project_pytest, governance_pytest, harness


def run_profile(
    profile: str,
    repository_root: Path,
    *,
    dry_run: bool = False,
) -> int:
    """从仓库根目录运行档位；首个失败立即停止。"""
    commands = commands_for_profile(profile)
    for command in commands:
        print(f"[{profile}] {shlex.join(command)}", flush=True)
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=repository_root, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="运行 CEG-WM governance、method 或 full 验证档位。",
    )
    parser.add_argument("profile", choices=PROFILE_NAMES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将运行的命令。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并从项目根目录运行验证。"""
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    return run_profile(args.profile, repository_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
