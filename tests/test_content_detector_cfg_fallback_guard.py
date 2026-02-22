"""
File purpose: ContentDetector cfg plan_digest fallback guard tests.
Module type: General module
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


def test_content_detector_cfg_plan_digest_fallback_requires_test_mode_or_explicit_allow() -> None:
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

    result_no_test_mode = detector.extract(
        cfg,
        inputs={
            "lf_evidence": {
                "status": "ok",
                "lf_score": 0.8,
            }
        },
        cfg_digest="cfg_digest_anchor",
    )
    assert result_no_test_mode.status == "absent"
    assert result_no_test_mode.content_failure_reason == "detector_no_plan_expected"

    result_test_mode = detector.extract(
        cfg,
        inputs={
            "test_mode": True,
            "plan_digest": "cfg_plan_digest_anchor",
            "lf_evidence": {
                "status": "ok",
                "lf_score": 0.8,
            },
        },
        cfg_digest="cfg_digest_anchor",
    )
    assert result_test_mode.status == "ok"
    assert result_test_mode.score is not None
