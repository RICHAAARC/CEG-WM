"""
测试用例：T2 - plan_digest 不一致必须返回 mismatch

功能说明：
- 验证当 plan_digest 不一致时，LowFreqCoder 返回 status="mismatch"。
- 验证 mismatch 状态下 score 必须为 None。
- 验证 content_failure_reason 正确设置。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.watermarking.content_chain.low_freq_coder import LowFreqCoder


def test_lf_coder_plan_digest_mismatch_returns_mismatch_status() -> None:
    """
    功能：验证 plan_digest 缺失时返回 mismatch 状态。

    Test that LowFreqCoder returns status="mismatch" when plan_digest is missing.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "mismatch" or score is not None.
    """
    # 配置缺失 plan_digest。
    cfg: Dict[str, Any] = {
        "watermark": {
            # plan_digest 缺失
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
        impl_digest="test_impl_digest_001"
    )

    result = coder.extract(cfg=cfg, inputs=inputs)

    # 验证失败语义。
    assert result.status == "mismatch", \
        f"Expected status='mismatch' when plan_digest is missing, got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='mismatch', got {result.score}"
    assert result.lf_score is None, \
        f"lf_score must be None when status='mismatch', got {result.lf_score}"
    assert result.content_failure_reason == "lf_coder_no_plan", \
        f"Expected failure_reason='lf_coder_no_plan', got '{result.content_failure_reason}'"
    assert result.lf_trace_digest is not None, \
        "lf_trace_digest should still be populated for audit even when mismatch"


def test_lf_coder_explicit_plan_digest_none_returns_mismatch() -> None:
    """
    功能：验证显式设置 plan_digest=None 时返回 mismatch。

    Test that explicitly setting plan_digest=None returns status="mismatch".

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "mismatch".
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": None,  # 显式设置为 None
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
        impl_digest="test_impl_digest_001"
    )

    result = coder.extract(cfg=cfg, inputs=inputs)

    assert result.status == "mismatch", \
        f"Expected status='mismatch' when plan_digest=None, got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='mismatch', got {result.score}"
