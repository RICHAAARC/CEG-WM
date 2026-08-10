"""按变更范围运行 CEG-WM 的登记验证档位。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

PROFILE_NAMES = ("governance", "method", "full")
VALIDATION_ENVIRONMENT_OVERRIDES = {
    "TMPDIR": "/tmp",
    "TEMP": "/tmp",
    "TMP": "/tmp",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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


def command_identity(command: tuple[str, ...]) -> str:
    """返回登记命令的稳定职责身份。"""
    if command[-1:] == ("tests",):
        return "project_pytest"
    if command[-1:] == ("governance/tests",):
        return "governance_pytest"
    if command[-1:] == ("governance/harness/run_all_audits.py",):
        return "harness_audits"
    raise ValueError("validation command identity is unknown")


def run_profile(
    profile: str,
    repository_root: Path,
    *,
    dry_run: bool = False,
) -> int:
    """从仓库根目录运行档位；首个失败立即停止。"""
    commands = commands_for_profile(profile)
    environment = dict(os.environ)
    environment.update(VALIDATION_ENVIRONMENT_OVERRIDES)
    for command in commands:
        identity = command_identity(command)
        print(
            f"[{profile}] command_identity={identity} command={shlex.join(command)}",
            flush=True,
        )
        if dry_run:
            continue
        started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            env=environment,
        )
        elapsed = perf_counter() - started
        print(
            f"[{profile}] command_identity={identity} "
            f"walltime_seconds={elapsed:.3f} returncode={completed.returncode}",
            flush=True,
        )
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
