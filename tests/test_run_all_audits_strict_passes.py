"""
文件目的：严格审计聚合命令回归测试。
Module type: General module
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_all_audits_strict_passes() -> None:
    """
    功能：验证严格审计命令可通过。

    Verify run_all_audits strict mode command exits successfully.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    command = [sys.executable, str(repo_root / "scripts" / "run_all_audits.py"), "--repo-root", str(repo_root), "--strict"]
    result = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, (
        "run_all_audits --strict must pass\n"
        f"stdout_tail={result.stdout[-1000:]}\n"
        f"stderr_tail={result.stderr[-1000:]}"
    )
