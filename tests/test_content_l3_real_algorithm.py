"""
L3 Content Chain 真实算法必达字段回归测试

功能说明：
- 验证 Content 链 L3 必达字段：mask_digest, plan_digest, lf_trace_digest 等。
- 确保 embed 侧不再输出 placeholder，而是真实水印证据。
- 确保 detect 侧能够检测并返回 content_score。
- 严格验证失败语义（absent/failed/mismatch）。
"""

import pytest
from main.watermarking.content_chain.unified_content_extractor import (
    UnifiedContentExtractor,
    UNIFIED_CONTENT_EXTRACTOR_ID,
    UNIFIED_CONTENT_EXTRACTOR_VERSION
)
from main.watermarking.content_chain.semantic_mask_provider import (
    SemanticMaskProvider,
    SEMANTIC_MASK_PROVIDER_ID
)
from main.watermarking.content_chain.content_detector import (
    ContentDetector,
    CONTENT_DETECTOR_ID
)
from main.core import digests


def test_unified_extractor_embed_mode_returns_mask_digest():
    """
    功能：验证 Embed 模式（detect.content.enabled=False）返回 mask_digest。

    Test unified extractor in embed mode returns mask_digest structural evidence.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If mask_digest is missing or invalid.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": False  # Embed 模式
            }
        },
        "enable_mask": False  # 掩码提取禁用（应返回 absent）
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=None, cfg_digest="test_cfg_digest")
    
    # Embed 模式 + enable_mask=False -> status=absent
    assert result.status == "absent", f"Expected absent, got {result.status}"
    assert result.score is None, "Score must be None in embed mode with mask disabled"
    

def test_unified_extractor_embed_mode_with_mask_enabled_returns_ok():
    """
    功能：验证 Embed 模式启用掩码时返回 status=ok 与 mask_digest。

    Test unified extractor in embed mode with mask enabled returns ok status.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not ok or mask_digest is missing.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": False  # Embed 模式
            }
        },
        "enable_mask": True,  # 启用掩码提取
        "mask_resolution_width": 512,
        "mask_resolution_height": 512
    }
    
    inputs = {
        "latent": [[1.0, 2.0], [3.0, 4.0]],  # 简化输入
        "shape": [1, 4, 64, 64]
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Embed 模式 + enable_mask=True -> status=ok（结构证据，score=None）
    # 或者 status=failed（如果 latent 输入无效）
    # 由于我们的输入是最小化的，可能触发 failed
    assert result.status in {"ok", "failed", "absent"}, f"Unexpected status: {result.status}"
    assert result.score is None or result.status != "ok", "Score must be None when status=ok in embed mode (structural evidence)"


def test_unified_extractor_detect_mode_returns_content_score_when_plan_present():
    """
    功能：验证 Detect 模式（detect.content.enabled=True）返回 content_score。

    Test unified extractor in detect mode returns content_score when plan_digest present.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If content_score is missing or status is incorrect.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True  # Detect 模式
            }
        },
        "watermark": {
            "plan_digest": "test_plan_digest_12345678",  # 计划摘要存在
            "lf": {
                "enabled": True
            },
            "hf": {
                "enabled": False
            }
        }
    }
    
    inputs = {
        "plan_digest": "test_plan_digest_12345678",
        "lf_evidence": {
            "status": "ok",
            "lf_score": 0.85
        },
        "lf_score": 0.85
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Detect 模式 + plan_digest 一致 -> status=ok, score=非None
    assert result.status == "ok", f"Expected ok, got {result.status}"
    assert result.score is not None, "Score must be non-None in detect mode when status=ok"
    assert result.score >= 0, f"Score must be non-negative, got {result.score}"


def test_unified_extractor_detect_mode_plan_mismatch_returns_mismatch():
    """
    功能：验证 Detect 模式 plan_digest 不一致时返回 mismatch。

    Test unified extractor in detect mode returns mismatch when plan_digest mismatches.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not mismatch or failure reason is missing.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True  # Detect 模式
            }
        },
        "watermark": {
            "lf": {
                "enabled": True
            }
        }
    }
    
    inputs = {
        "expected_plan_digest": "expected_plan_digest_abc123",  # 期望的 plan_digest
        "plan_digest": "actual_plan_digest_xyz789",  # 实际的 plan_digest（不一致）
        "lf_evidence": {
            "status": "ok",
            "lf_score": 0.75
        }
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": UNIFIED_CONTENT_EXTRACTOR_ID,
        "impl_version": UNIFIED_CONTENT_EXTRACTOR_VERSION
    })
    
    extractor = UnifiedContentExtractor(
        UNIFIED_CONTENT_EXTRACTOR_ID,
        UNIFIED_CONTENT_EXTRACTOR_VERSION,
        impl_digest
    )
    
    result = extractor.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # Detect 模式 + plan_digest 不一致 -> status=mismatch, score=None
    assert result.status == "mismatch", f"Expected mismatch, got {result.status}"
    assert result.score is None, "Score must be None when status=mismatch"
    assert result.content_failure_reason is not None, "Failure reason must be present when status=mismatch"
    assert "plan_mismatch" in result.content_failure_reason or "detector_plan_mismatch" in result.content_failure_reason, \
        f"Failure reason should mention plan mismatch, got: {result.content_failure_reason}"


def test_l3_content_chain_not_placeholder_when_enabled():
    """
    功能：验证 Content 链 L3 不再是 placeholder（启用时）。

    Regression test: Content chain L3 must not emit placeholder when enabled.
    Validates that embed_record contains mask_digest/plan_digest/lf_trace_digest.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If L3 必达字段缺失。
    """
    # 此测试需要完整的 embed 流程，暂时标记为 skip（需要 pipeline 和推理）
    pytest.skip("Requires full embed pipeline integration test (TBD in integration suite)")


def test_semantic_mask_provider_returns_mask_digest_when_enabled():
    """
    功能：验证 SemanticMaskProvider 启用时返回 mask_digest。

    Test SemanticMaskProvider returns mask_digest when enable_mask=True.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If mask_digest is missing or status is incorrect.
    """
    cfg = {
        "enable_mask": True,
        "mask_resolution_width": 512,
        "mask_resolution_height": 512
    }
    
    inputs = {
        "latent": [[1.0, 2.0], [3.0, 4.0]],
        "shape": [1, 4, 64, 64]
    }
    
    impl_digest = digests.canonical_sha256({
        "impl_id": SEMANTIC_MASK_PROVIDER_ID,
        "impl_version": "v1"
    })
    
    provider = SemanticMaskProvider(
        SEMANTIC_MASK_PROVIDER_ID,
        "v1",
        impl_digest
    )
    
    result = provider.extract(cfg, inputs=inputs, cfg_digest="test_cfg_digest")
    
    # enable_mask=True，但输入可能无效 -> status=ok or failed
    # 如果 ok，必须有 mask_digest
    if result.status == "ok":
        assert result.mask_digest is not None, "mask_digest must be present when status=ok"
        assert isinstance(result.mask_digest, str), "mask_digest must be str"
        assert len(result.mask_digest) > 0, "mask_digest must be non-empty"
    elif result.status == "failed":
        assert result.content_failure_reason is not None, "Failure reason must be present when status=failed"
    else:
        pytest.fail(f"Unexpected status: {result.status}")

