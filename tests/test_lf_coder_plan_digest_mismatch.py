"""
测试用例：T2 - plan_digest 非法时必须抛出 TypeError

功能说明：
- 验证 V2 detect_score 在 plan_digest 为空字符串/None 时抛出 TypeError。
- V2 不再支持"mismatch"状态（plan_digest 直传，非从 cfg 读取），
  非法 plan_digest 应在接口层立即拒绝。
"""

from __future__ import annotations

import pytest

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


def test_lf_coder_plan_digest_mismatch_returns_mismatch_status() -> None:
    """
    功能：验证 plan_digest 为空字符串时 detect_score 抛出 TypeError。

    Test that detect_score raises TypeError when plan_digest is an empty string.
    V2 validates plan_digest at entry and raises TypeError for invalid values.

    Args:
        None.

    Returns:
        None.

    Raises:
        pytest.raises(TypeError): Expected when plan_digest is empty.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": True, "message_length": 16, "ecc_sparsity": 3}}}
    coder = _make_coder()
    # V2 要求 plan_digest 为非空字符串，空字符串触发 TypeError。
    with pytest.raises(TypeError, match="plan_digest must be non-empty str"):
        coder.detect_score(
            cfg=cfg,
            latent_features=[0.1, 0.2, 0.3, 0.4],
            plan_digest="",
            lf_basis=None,
        )


def test_lf_coder_explicit_plan_digest_none_returns_mismatch() -> None:
    """
    功能：验证 plan_digest=None 时 detect_score 抛出 TypeError。

    Test that detect_score raises TypeError when plan_digest is None.

    Args:
        None.

    Returns:
        None.

    Raises:
        pytest.raises(TypeError): Expected when plan_digest is None.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": True, "message_length": 16, "ecc_sparsity": 3}}}
    coder = _make_coder()
    with pytest.raises(TypeError, match="plan_digest must be non-empty str"):
        coder.detect_score(
            cfg=cfg,
            latent_features=[0.1, 0.2, 0.3, 0.4],
            plan_digest=None,
            lf_basis=None,
        )
