"""
文件目的：为当前 tests 目录提供最小 pytest 基础设施。
Module type: General module

职责边界：
1. 仅确保 pyproject.toml 中声明的 --basetemp=.pytest/tmp 可在干净 checkout 下直接使用。
2. 不恢复 archive/tests/conftest.py 中与当前 stage 04 测试无关的 fixture、monkeypatch 或全局测试策略。
3. 不依赖 archive 路径运行。
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _workspace_root() -> Path:
    """
    功能：返回仓库根目录路径。

    Resolve the repository root from the active tests directory.

    Args:
        None.

    Returns:
        Repository root path.
    """
    return Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    """
    功能：在 pytest 初始化阶段预创建 basetemp 目录。

    Ensure the configured pytest base temporary directory exists before any
    tmp_path-based fixture is requested.

    Args:
        config: Active pytest configuration object.

    Returns:
        None.
    """
    workspace_root = _workspace_root()
    configured_basetemp = getattr(config.option, "basetemp", None)

    if isinstance(configured_basetemp, str) and configured_basetemp.strip():
        basetemp_path = Path(configured_basetemp)
        if not basetemp_path.is_absolute():
            basetemp_path = workspace_root / basetemp_path
    else:
        basetemp_path = workspace_root / ".pytest" / "tmp"

    basetemp_path.parent.mkdir(parents=True, exist_ok=True)
    basetemp_path.mkdir(parents=True, exist_ok=True)