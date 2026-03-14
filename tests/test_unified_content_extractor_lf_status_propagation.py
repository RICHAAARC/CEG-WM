"""
测试用例：T6 补充 - UnifiedContentExtractor 状态传播

功能说明：
- 验证 UnifiedContentExtractor 正确传播 LF 失败状态。
- 验证当 lf_status != "ok" 时，content_status = lf_status。
- 验证失败时 content_score = None。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.watermarking.content_chain.unified_content_extractor import UnifiedContentExtractor
from main.watermarking.content_chain.interfaces import ContentEvidence


def test_unified_content_extractor_propagates_lf_absent_status() -> None:
    """
    功能：验证 UnifiedContentExtractor 传播 LF absent 状态。

    Test that UnifiedContentExtractor propagates lf_status="absent" to content_status.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status propagation is incorrect.
    """
    cfg: Dict[str, Any] = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_001"
        }
    }

    # 模拟 LF absent 证据。
    lf_evidence = ContentEvidence(
        status="absent",
        score=None,
        audit={"impl_identity": "low_freq_template_codec", "impl_version": "v1", "impl_digest": "test", "trace_digest": "test"},
        lf_score=None,
        content_failure_reason=None
    )

    inputs: Dict[str, Any] = {
        "lf_evidence": lf_evidence,
        "plan_digest": "test_plan_digest_001"
    }

    detector = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest"
    )

    result = detector.extract(cfg=cfg, inputs=inputs)

    # 验证状态传播。
    assert result.status == "absent", \
        f"Expected status='absent' when lf_status='absent', got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='absent', got {result.score}"


def test_unified_content_extractor_propagates_lf_mismatch_status() -> None:
    """
    功能：验证 UnifiedContentExtractor 传播 LF mismatch 状态。

    Test that UnifiedContentExtractor propagates lf_status="mismatch" to content_status.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status propagation is incorrect.
    """
    cfg: Dict[str, Any] = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_001"
        }
    }

    # 模拟 LF mismatch 证据。
    lf_evidence = ContentEvidence(
        status="mismatch",
        score=None,
        audit={"impl_identity": "low_freq_template_codec", "impl_version": "v1", "impl_digest": "test", "trace_digest": "test"},
        lf_score=None,
        content_failure_reason="lf_coder_plan_mismatch"
    )

    inputs: Dict[str, Any] = {
        "lf_evidence": lf_evidence,
        "plan_digest": "test_plan_digest_001"
    }

    detector = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest"
    )

    result = detector.extract(cfg=cfg, inputs=inputs)

    # 验证状态传播。
    assert result.status == "mismatch", \
        f"Expected status='mismatch' when lf_status='mismatch', got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='mismatch', got {result.score}"
    assert result.content_failure_reason == "lf_coder_plan_mismatch", \
        f"Expected propagated failure_reason, got '{result.content_failure_reason}'"


def test_unified_content_extractor_propagates_lf_failed_status() -> None:
    """
    功能：验证 UnifiedContentExtractor 传播 LF failed 状态。

    Test that UnifiedContentExtractor propagates lf_status="failed" to content_status.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status propagation is incorrect.
    """
    cfg: Dict[str, Any] = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_001"
        }
    }

    # 模拟 LF failed 证据。
    lf_evidence = ContentEvidence(
        status="failed",
        score=None,
        audit={"impl_identity": "low_freq_template_codec", "impl_version": "v1", "impl_digest": "test", "trace_digest": "test"},
        lf_score=None,
        content_failure_reason="lf_coder_encoding_failed"
    )

    inputs: Dict[str, Any] = {
        "lf_evidence": lf_evidence,
        "plan_digest": "test_plan_digest_001"
    }

    detector = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest"
    )

    result = detector.extract(cfg=cfg, inputs=inputs)

    # 验证状态传播。
    assert result.status == "failed", \
        f"Expected status='failed' when lf_status='failed', got '{result.status}'"
    assert result.score is None, \
        f"score must be None when status='failed', got {result.score}"
    assert result.content_failure_reason == "lf_coder_encoding_failed", \
        f"Expected propagated failure_reason, got '{result.content_failure_reason}'"


def test_unified_content_extractor_success_with_lf_score() -> None:
    """
    功能：验证 UnifiedContentExtractor 在 LF 成功时正确融合得分。

    Test that UnifiedContentExtractor correctly fuses lf_score when lf_status="ok".

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If score fusion is incorrect.
    """
    cfg: Dict[str, Any] = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_001"
        }
    }

    # 模拟 LF 成功证据。
    lf_evidence = ContentEvidence(
        status="ok",
        score=0.75,
        audit={"impl_identity": "low_freq_template_codec", "impl_version": "v1", "impl_digest": "test", "trace_digest": "test"},
        lf_score=0.75,
        content_failure_reason=None
    )

    inputs: Dict[str, Any] = {
        "lf_evidence": lf_evidence,
        "plan_digest": "test_plan_digest_001"
    }

    detector = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest"
    )

    result = detector.extract(cfg=cfg, inputs=inputs)

    # 验证成功路径。
    assert result.status == "ok", \
        f"Expected status='ok' when lf_status='ok', got '{result.status}'"
    assert result.score == 0.75, \
        f"Expected content_score=0.75 (same as lf_score), got {result.score}"
    assert result.lf_score == 0.75, \
        f"Expected lf_score=0.75, got {result.lf_score}"
    assert result.score_parts is not None, \
        "score_parts must be populated when status='ok'"
    assert result.score_parts.get("lf_score") == 0.75, \
        f"Expected score_parts['lf_score']=0.75, got {result.score_parts.get('lf_score')}"
    assert result.score_parts.get("hf_score") == "<absent>", \
        f"Expected score_parts['hf_score']='<absent>', got {result.score_parts.get('hf_score')}"


def test_unified_content_extractor_preserves_hf_metrics_when_lf_absent() -> None:
    """
    功能：验证 LF absent 时 canonical HF summary 不会被无差别清空。

    Test that canonical HF summary remains available when LF status is absent.

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "detect": {
            "content": {
                "enabled": True
            }
        },
        "watermark": {
            "hf": {
                "enabled": True
            }
        }
    }

    inputs: Dict[str, Any] = {
        "expected_plan_digest": "test_plan_digest_001",
        "plan_digest": "test_plan_digest_001",
        "lf_evidence": {
            "status": "absent",
            "lf_score": None,
            "content_failure_reason": "formal_profile_sidecar_disabled",
            "lf_evidence_summary": {
                "lf_status": "absent",
                "lf_absent_reason": "formal_profile_sidecar_disabled",
            },
        },
        "hf_evidence": {
            "status": "ok",
            "hf_score": 0.62,
            "hf_trace_digest": "h" * 64,
            "hf_evidence_summary": {
                "hf_status": "ok",
                "hf_detect_variant": "projection_tail_truncation",
                "basis_rank": 8,
            },
        },
    }

    detector = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest"
    )

    result = detector.extract(cfg=cfg, inputs=inputs)

    assert result.status == "absent"
    assert result.score is None
    assert result.hf_trace_digest == "h" * 64
    assert result.score_parts is not None
    hf_metrics = result.score_parts.get("hf_metrics")
    assert isinstance(hf_metrics, dict)
    assert hf_metrics.get("hf_status") == "ok"
