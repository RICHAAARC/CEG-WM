"""
File purpose: 配置加载阶段的 paper LF ECC 门禁测试。
Module type: General module
"""

from __future__ import annotations

import pytest

from main.core import config_loader


def test_validate_paper_lf_ecc_gate_blocks_legacy_int_ecc() -> None:
    """
    功能：当 paper 模式启用时，legacy int ecc 必须被配置层阻断。

    Ensure config-stage gate rejects legacy int ecc when paper mode is enabled.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "lf": {
                "enabled": True,
                "ecc": 3,
            }
        },
    }

    with pytest.raises(ValueError, match=r"paper_faithfulness requires watermark\.lf\.ecc='sparse_ldpc'"):
        config_loader._validate_paper_lf_ecc_gate(cfg)  # pyright: ignore[reportPrivateUsage]


def test_validate_paper_lf_ecc_gate_accepts_sparse_ldpc() -> None:
    """
    功能：当 paper 模式启用且 ecc 为 sparse_ldpc 时门禁应通过。

    Ensure config-stage gate allows sparse_ldpc when paper mode is enabled.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "lf": {
                "enabled": True,
                "ecc": "sparse_ldpc",
            }
        },
    }

    config_loader._validate_paper_lf_ecc_gate(cfg)  # pyright: ignore[reportPrivateUsage]


def test_validate_paper_lf_ecc_gate_ignores_when_paper_disabled() -> None:
    """
    功能：当 paper 模式关闭时，不应阻断 legacy int ecc。

    Ensure gate does not block int ecc when paper mode is disabled.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {
        "paper_faithfulness": {"enabled": False},
        "watermark": {
            "lf": {
                "enabled": True,
                "ecc": 3,
            }
        },
    }

    config_loader._validate_paper_lf_ecc_gate(cfg)  # pyright: ignore[reportPrivateUsage]
