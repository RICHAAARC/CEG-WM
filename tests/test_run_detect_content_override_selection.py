"""
功能：验证 run_detect 输入记录的 content override 选择规则。

Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.cli.run_detect import resolve_content_override_from_input_record


def test_resolve_content_override_skips_embed_absent_content_evidence() -> None:
    """
    功能：仅有 embed 侧 absent content_evidence 时不应覆盖 detect 计算。

    Ensure embed-style absent content_evidence is ignored as detect override.

    Args:
        None.

    Returns:
        None.
    """
    input_record: Dict[str, Any] = {
        "content_evidence": {
            "status": "absent",
            "content_failure_reason": "detector_no_plan_expected",
            "score": None,
        }
    }

    resolved = resolve_content_override_from_input_record(input_record)
    assert resolved is None


def test_resolve_content_override_prefers_detect_payload_when_available() -> None:
    """
    功能：存在 detect payload 时应返回可复用覆盖结果。

    Ensure detect-style payload is selected for orchestrator override.

    Args:
        None.

    Returns:
        None.
    """
    input_record: Dict[str, Any] = {
        "content_evidence": {
            "status": "absent",
            "score": None,
        },
        "content_evidence_payload": {
            "status": "ok",
            "score": 0.123,
            "plan_digest": "abc",
        },
    }

    resolved = resolve_content_override_from_input_record(input_record)
    assert isinstance(resolved, dict)
    assert resolved.get("status") == "ok"
    assert resolved.get("score") == 0.123
