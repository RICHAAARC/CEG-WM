"""
测试用例：T4 - fail/mismatch/absent 时 score 必须为 None

功能说明：
- 验证所有非 ok 状态下，score 和 lf_score 必须为 None。
- 覆盖 failed、mismatch、absent 三种失败语义。
- 确保失败路径不产生"看似有效的分数"。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.watermarking.content_chain.low_freq_coder import LowFreqCoder


def test_lf_coder_absent_status_has_no_score() -> None:
    """
    功能：验证 absent 状态下 score 为 None。

    Test that status="absent" has score=None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "enabled": False
            }
        }
    }

    inputs: Dict[str, Any] = {
        "latent_features": [0.1, 0.2],
        "latent_shape": (2,)
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    result = coder.extract(cfg=cfg, inputs=inputs)

    assert result.status == "absent"
    assert result.score is None, "score must be None when status='absent'"
    assert result.lf_score is None, "lf_score must be None when status='absent'"


def test_lf_coder_mismatch_status_has_no_score() -> None:
    """
    功能：验证 mismatch 状态下 score 为 None。

    Test that status="mismatch" has score=None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            # plan_digest 缺失 → mismatch
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": 3,
                "strength": 0.5,
                "delta": 1.0,
                "block_length": 8
            }
        }
    }

    inputs: Dict[str, Any] = {
        "latent_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "latent_shape": (8,)
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    result = coder.extract(cfg=cfg, inputs=inputs)

    assert result.status == "mismatch"
    assert result.score is None, "score must be None when status='mismatch'"
    assert result.lf_score is None, "lf_score must be None when status='mismatch'"


def test_lf_coder_failed_status_has_no_score() -> None:
    """
    功能：验证 failed 状态下 score 为 None。

    Test that status="failed" has score=None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": 3,
                "strength": 0.5,
                "delta": 1.0,
                "block_length": 8
            }
        }
    }

    # 输入缺失 latent_features → failed。
    inputs: Dict[str, Any] = {
        # latent_features 缺失
        "latent_shape": (8,)
    }

    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    result = coder.extract(cfg=cfg, inputs=inputs)

    assert result.status == "failed"
    assert result.score is None, "score must be None when status='failed'"
    assert result.lf_score is None, "lf_score must be None when status='failed'"
    assert result.content_failure_reason is not None, \
        "content_failure_reason must be populated when status='failed'"


def test_lf_coder_all_non_ok_statuses_have_no_score() -> None:
    """
    功能：验证所有非 ok 状态下 score 和 lf_score 都为 None。

    Comprehensive test that all non-ok statuses have score=None and lf_score=None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If any non-ok status has non-None score.
    """
    coder = LowFreqCoder(
        impl_id="low_freq_coder_v1",
        impl_version="v1",
        impl_digest="test_impl_digest"
    )

    # 测试用例列表：(cfg, inputs, expected_status)
    test_cases = [
        # absent：禁用。
        (
            {"watermark": {"lf": {"enabled": False}}},
            {"latent_features": [0.1], "latent_shape": (1,)},
            "absent"
        ),
        # mismatch：plan_digest 缺失。
        (
            {"watermark": {"lf": {"enabled": True}}},
            {"latent_features": [0.1], "latent_shape": (1,)},
            "mismatch"
        ),
        # failed：输入缺失。
        (
            {"watermark": {"plan_digest": "test", "lf": {"enabled": True}}},
            {},  # 输入为空
            "failed"
        ),
    ]

    for cfg, inputs, expected_status in test_cases:
        result = coder.extract(cfg=cfg, inputs=inputs)
        assert result.status == expected_status, \
            f"Expected status={expected_status}, got {result.status}"
        assert result.score is None, \
            f"score must be None when status={expected_status}, got {result.score}"
        assert result.lf_score is None, \
            f"lf_score must be None when status={expected_status}, got {result.lf_score}"
