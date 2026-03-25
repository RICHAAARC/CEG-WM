"""
测试用例：T3 - enable=false 必须返回 absent 且无 score

功能说明：
- 验证当 watermark.lf.enabled=false 时，LowFreqTemplateCodec 返回 status="absent"。
- 验证 absent 状态下 score 和 lf_score 必须为 None。
- 验证 absent 是"弃权"语义，不是错误。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodec,
    LOW_FREQ_TEMPLATE_CODEC_ID,
    LOW_FREQ_TEMPLATE_CODEC_VERSION,
)


def _make_coder() -> LowFreqTemplateCodec:
    impl_digest = digests.canonical_sha256({
        "impl_id": LOW_FREQ_TEMPLATE_CODEC_ID,
        "impl_version": LOW_FREQ_TEMPLATE_CODEC_VERSION,
    })
    return LowFreqTemplateCodec(
        LOW_FREQ_TEMPLATE_CODEC_ID,
        LOW_FREQ_TEMPLATE_CODEC_VERSION,
        impl_digest,
    )


def test_lf_coder_disabled_returns_absent_status() -> None:
    """
    功能：验证 LF 禁用时返回 absent 状态。

    Test that LowFreqTemplateCodec returns status="absent" when watermark.lf.enabled=false.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "absent" or score is not None.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": False}}}
    coder = _make_coder()
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        plan_digest="test_plan_digest_001",
        lf_basis=None,
    )
    assert trace["status"] == "absent", f"Expected absent, got {trace['status']}"
    assert score is None
    assert trace.get("lf_score") is None
    assert trace.get("lf_failure_reason") is None
    assert trace.get("lf_trace_digest") is not None, "lf_trace_digest 应存在"


def test_lf_coder_enabled_false_with_params_still_returns_absent() -> None:
    """
    功能：验证即使配置了参数，enabled=false 仍返回 absent。

    Test that enabled=false returns absent even with LF params configured.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "absent".
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "enabled": False,
                "message_length": 16,
                "ecc_sparsity": 3,
                "correlation_scale": 10.0,
            }
        }
    }
    coder = _make_coder()
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=[0.1, 0.2, 0.3, 0.4],
        plan_digest="test_plan_digest_002",
        lf_basis=None,
    )
    assert trace["status"] == "absent"
    assert score is None
    assert trace.get("lf_trace_digest") is not None
