"""
File purpose: ContentDetector formal path plan_digest closure guard tests.
Module type: General module

v3 闭包语义：
- 正式 detector 不再从 cfg 回填 plan_digest；不识别 test_mode 输入参数。
- expected_plan_digest 必须由调用方通过 inputs["expected_plan_digest"] 或
  inputs["plan_digest_expected"] 或 inputs["plan_digest"] 显式提供。
- 若调用方在 inputs 中传入 test_mode=True，formal detector 忽略该字段。
"""

from main.core import digests
from main.watermarking.content_chain.content_detector import (
    ContentDetector,
    CONTENT_DETECTOR_ID,
    CONTENT_DETECTOR_VERSION,
)


def _build_detector() -> ContentDetector:
    return ContentDetector(
        impl_id=CONTENT_DETECTOR_ID,
        impl_version=CONTENT_DETECTOR_VERSION,
        impl_digest=digests.canonical_sha256(
            {
                "impl_id": CONTENT_DETECTOR_ID,
                "impl_version": CONTENT_DETECTOR_VERSION,
            }
        ),
    )


def test_formal_detector_absent_when_no_plan_digest_in_inputs() -> None:
    """
    功能：验证 formal detector 在 inputs 中未提供 plan_digest 时返回 absent。

    Verify that the formal detector returns absent status when no plan_digest
    is provided in inputs, even if cfg.watermark.plan_digest exists.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True,
            }
        },
        "watermark": {
            "plan_digest": "cfg_plan_digest_anchor",
            "hf": {
                "enabled": False,
            },
        },
    }
    detector = _build_detector()

    # inputs 中无 plan_digest → formal detector 应返回 absent，不从 cfg 回填
    result = detector.extract(
        cfg,
        inputs={
            "lf_evidence": {
                "status": "ok",
                "lf_score": 0.8,
            }
        },
        cfg_digest="cfg_digest_anchor",
    )
    assert result.status == "absent"
    assert result.content_failure_reason == "detector_no_plan_expected"


def test_formal_detector_ok_when_plan_digest_explicit_in_inputs() -> None:
    """
    功能：验证 formal detector 在 inputs 中显式提供 plan_digest 时正常返回 ok。

    Verify that the formal detector returns ok status when plan_digest is
    explicitly provided in inputs.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True,
            }
        },
        "watermark": {
            "hf": {
                "enabled": False,
            },
        },
    }
    detector = _build_detector()

    result = detector.extract(
        cfg,
        inputs={
            "plan_digest": "explicit_plan_digest_anchor",
            "lf_evidence": {
                "status": "ok",
                "lf_score": 0.8,
            },
        },
        cfg_digest="cfg_digest_anchor",
    )
    assert result.status == "ok"
    assert result.score is not None


def test_formal_detector_ignores_test_mode_in_inputs() -> None:
    """
    功能：验证 formal detector 忽略 inputs 中的 test_mode 字段，不触发 cfg 回填。

    Verify that test_mode=True in inputs does NOT cause the formal detector
    to fall back to cfg.watermark.plan_digest. The detector must remain absent
    when no explicit plan_digest is in inputs.
    """
    cfg = {
        "detect": {
            "content": {
                "enabled": True,
            }
        },
        "watermark": {
            "plan_digest": "cfg_plan_digest_anchor",
            "hf": {
                "enabled": False,
            },
        },
    }
    detector = _build_detector()

    # test_mode=True 在 inputs 中，但 formal detector 不识别此参数
    # cfg 中有 plan_digest，但不应被回填
    result = detector.extract(
        cfg,
        inputs={
            "test_mode": True,
            "lf_evidence": {
                "status": "ok",
                "lf_score": 0.8,
            },
        },
        cfg_digest="cfg_digest_anchor",
    )
    # formal detector 必须返回 absent，不受 test_mode 影响
    assert result.status == "absent"
    assert result.content_failure_reason == "detector_no_plan_expected"

