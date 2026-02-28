"""
文件目的：严格审计聚合命令回归测试。
Module type: General module
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_all_audits_strict_requires_explicit_run_root() -> None:
    """
    功能：验证严格审计在未显式绑定 run_root 时必须阻断。

    Verify run_all_audits strict mode fails when explicit --run-root is absent.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    command = [sys.executable, str(repo_root / "scripts" / "run_all_audits.py"), "--repo-root", str(repo_root), "--strict"]
    result = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 1, (
        "run_all_audits --strict without --run-root must block\n"
        f"stdout_tail={result.stdout[-1000:]}\n"
        f"stderr_tail={result.stderr[-1000:]}"
    )
    assert "audit.strict_requires_bound_run_root" in result.stdout
