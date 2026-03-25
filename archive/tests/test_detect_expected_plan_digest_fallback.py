"""
功能：验证 detect 侧 expected_plan_digest 的注入位点回退解析。

Module type: General module
"""

from __future__ import annotations

from main.watermarking.detect.orchestrator import _resolve_expected_plan_digest


def test_expected_plan_digest_fallback_from_injection_rule_summary() -> None:
    """
    功能：当 content_evidence.plan_digest 缺失时，必须回退到 injection_rule_summary.plan_digest。

    Validate fallback extraction from content_evidence.injection_site_spec.

    Args:
        None.

    Returns:
        None.
    """
    input_record = {
        "content_evidence": {
            "status": "absent",
            "plan_digest": None,
            "injection_site_spec": {
                "injection_rule_summary": {
                    "plan_digest": "plan_digest_from_injection_rule_summary"
                }
            },
        }
    }

    resolved = _resolve_expected_plan_digest(input_record)
    assert resolved == "plan_digest_from_injection_rule_summary"
