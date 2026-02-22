"""
测试用例：T3 - enable=false 必须返回 absent 且无 score

功能说明：
- 验证当 watermark.lf.enabled=false 时，LowFreqCoder 返回 status="absent"。
- 验证 absent 状态下 score 和 lf_score 必须为 None。
- 验证 absent 是"弃权"语义，不是错误。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.watermarking.content_chain.low_freq_coder import LowFreqCoder


def test_lf_coder_disabled_returns_absent_status() -> None:
    """
    功能：验证 LF 禁用时返回 absent 状态。

    Test that LowFreqCoder returns status="absent" when watermark.lf.enabled=false.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "absent" or score is not None.
    """
    # 配置禁用 LF。
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": False
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

    # 验证 absent 语义。
    assert result.status == "absent", \
        f"Expected status='absent' when LF is disabled, got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='absent', got {result.score}"
    assert result.lf_score is None, \
        f"lf_score must be None when status='absent', got {result.lf_score}"
    assert result.content_failure_reason is None, \
        f"content_failure_reason should be None for absent (not error), got '{result.content_failure_reason}'"
    assert result.lf_trace_digest is not None, \
        "lf_trace_digest should still be populated for audit even when absent"


def test_lf_coder_enabled_false_with_params_still_returns_absent() -> None:
    """
    功能：验证即使配置了参数，enabled=false 仍返回 absent。

    Test that even with LF parameters configured, enabled=false returns status="absent".

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "absent".
    """
    # 虽然配置了 LF 参数，但 enabled=false。
    cfg: Dict[str, Any] = {
        "watermark": {
            "plan_digest": "test_plan_digest_001",
            "lf": {
                "enabled": False,
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

    # 即使配置了参数，enabled=false 仍返回 absent。
    assert result.status == "absent", \
        f"Expected status='absent' when enabled=false, got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='absent', got {result.score}"
