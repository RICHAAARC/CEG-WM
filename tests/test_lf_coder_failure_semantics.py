"""
测试用例：T4 - fail/mismatch/absent 时 score 必须为 None

功能说明：
- 验证所有非 ok 状态下，score 和 lf_score 必须为 None。
- 覆盖 failed、absent 两种失败语义。
- 确保失败路径不产生"看似有效的分数"。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodecV2,
    LOW_FREQ_TEMPLATE_CODEC_V2_ID,
    LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
)


def _make_coder() -> LowFreqTemplateCodecV2:
    impl_digest = digests.canonical_sha256({
        "impl_id": LOW_FREQ_TEMPLATE_CODEC_V2_ID,
        "impl_version": LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
    })
    return LowFreqTemplateCodecV2(
        LOW_FREQ_TEMPLATE_CODEC_V2_ID,
        LOW_FREQ_TEMPLATE_CODEC_V2_VERSION,
        impl_digest,
    )


def test_lf_coder_absent_status_has_no_score() -> None:
    """
    功能：验证 absent 状态下 score 为 None。

    Test that status="absent" produces score=None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None when absent.
    """
    cfg: Dict[str, Any] = {"watermark": {"lf": {"enabled": False}}}
    coder = _make_coder()
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=[0.1, 0.2],
        plan_digest="test_plan_digest",
        lf_basis=None,
    )
    assert trace["status"] == "absent"
    assert score is None, "score must be None when status='absent'"
    assert trace.get("lf_score") is None, "lf_score must be None when status='absent'"


def test_lf_coder_mismatch_status_has_no_score() -> None:
    """
    功能：验证 lf_basis 缺失时（failed 状态）score 为 None。

    Test that status="failed" (lf_basis absent) produces score=None.
    V2 does not have a mismatch state; lf_basis_required_but_absent triggers failed.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None when failed.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 16,
                "ecc_sparsity": 3,
            }
        }
    }
    coder = _make_coder()
    # lf_basis=None → status="failed"（lf_basis_required_but_absent）。
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        plan_digest="test_plan_digest_001",
        lf_basis=None,
    )
    assert trace["status"] == "failed"
    assert score is None, "score must be None when status='failed'"
    assert trace.get("lf_score") is None, "lf_score must be None when status='failed'"


def test_lf_coder_failed_status_has_no_score() -> None:
    """
    功能：验证 failed 状态下 score 为 None。

    Test that status="failed" produces score=None and lf_failure_reason is set.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score is not None or failure_reason is missing.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 16,
                "ecc_sparsity": 3,
            }
        }
    }
    coder = _make_coder()
    # lf_basis=None → failed，failure_reason=lf_basis_required_but_absent。
    score, trace = coder.detect_score(
        cfg=cfg,
        latent_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        plan_digest="test_plan_digest_001",
        lf_basis=None,
    )
    assert trace["status"] == "failed"
    assert score is None, "score must be None when status='failed'"
    assert trace.get("lf_score") is None, "lf_score must be None when status='failed'"
    assert trace.get("lf_failure_reason") is not None,         "lf_failure_reason must be populated when status='failed'"


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
    coder = _make_coder()

    test_cases = [
        # absent：disabled。
        (
            {"watermark": {"lf": {"enabled": False}}},
            "test_pd_1",
            None,
            "absent",
        ),
        # failed：lf_basis=None。
        (
            {"watermark": {"lf": {"enabled": True, "message_length": 16, "ecc_sparsity": 3}}},
            "test_pd_2",
            None,
            "failed",
        ),
    ]

    for cfg, plan_digest, lf_basis, expected_status in test_cases:
        score, trace = coder.detect_score(
            cfg=cfg,
            latent_features=[0.1, 0.2, 0.3, 0.4],
            plan_digest=plan_digest,
            lf_basis=lf_basis,
        )
        assert trace["status"] == expected_status,             f"Expected status={expected_status}, got {trace['status']}"
        assert score is None,             f"score must be None when status={expected_status}, got {score}"
        assert trace.get("lf_score") is None,             f"lf_score must be None when status={expected_status}, got {trace.get('lf_score')}"
