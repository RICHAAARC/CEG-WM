"""
CLI 入口一致性检查

功能说明：
- 用于校验 CLI 入口模块必须以模块方式执行。
"""

from __future__ import annotations

from typing import Any


_SENTINEL = object()


def assert_module_execution(
    entry_name: str,
    *,
    module_package: Any = _SENTINEL,
    module_spec: Any = _SENTINEL,
    strict: bool = False
) -> None:
    """
    功能：校验 CLI 必须以模块方式执行。

    Assert that CLI entry points are executed via module mode.

    Args:
        entry_name: Entry module name without package prefix.
        module_package: Optional override for module __package__.
        module_spec: Optional override for module __spec__.

    Returns:
        None.

    Raises:
        RuntimeError: If executed outside module mode.
        ImportEnvironmentError: If strict check fails under module mode.
        TypeError: If inputs are invalid.
    """
    if not isinstance(entry_name, str) or not entry_name:
        # entry_name 输入不合法，必须 fail-fast。
        raise TypeError("entry_name must be non-empty str")
    if not isinstance(strict, bool):
        # strict 类型不符合预期，必须 fail-fast。
        raise TypeError("strict must be bool")

    package_value = __package__ if module_package is _SENTINEL else module_package
    spec_value = __spec__ if module_spec is _SENTINEL else module_spec
    if package_value is None or spec_value is None:
        message = (
            "Invalid execution environment for entry="
            f"{entry_name}: __package__={package_value}, __spec__={spec_value}. "
            f"Use: python -m main.cli.{entry_name}"
        )
        raise RuntimeError(message)

    if strict:
        from main.core.errors import ImportEnvironmentError
        message = (
            "Strict module execution check failed for entry="
            f"{entry_name}: __package__={package_value}, __spec__={spec_value}. "
            f"Use: python -m main.cli.{entry_name}"
        )
        raise ImportEnvironmentError(message)
