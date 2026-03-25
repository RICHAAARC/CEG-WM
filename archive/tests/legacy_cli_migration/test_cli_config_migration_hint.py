"""
File purpose: CLI 配置迁移提示测试。
Module type: General module
"""

from __future__ import annotations

from main.cli.run_common import build_cli_config_migration_hint


def test_build_cli_config_migration_hint_returns_hint_for_paper_lf_ecc_gate() -> None:
    """
    功能：验证已知门禁错误会生成迁移提示。

    Verify known paper LF ECC gate error generates migration hint text.

    Args:
        None.

    Returns:
        None.
    """
    exc = ValueError("paper_faithfulness requires watermark.lf.ecc='sparse_ldpc'; legacy int ecc is not allowed")
    hint = build_cli_config_migration_hint(exc)

    assert isinstance(hint, str)
    assert "watermark.lf.ecc" in hint
    assert "sparse_ldpc" in hint


def test_build_cli_config_migration_hint_returns_none_for_other_errors() -> None:
    """
    功能：验证无关错误不生成迁移提示。

    Verify unrelated errors do not produce migration hint.

    Args:
        None.

    Returns:
        None.
    """
    exc = ValueError("some unrelated config failure")
    hint = build_cli_config_migration_hint(exc)

    assert hint is None
