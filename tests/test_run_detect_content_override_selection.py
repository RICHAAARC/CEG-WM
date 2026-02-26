"""
功能：验证 run_detect 输入记录的 content override 选择规则。

Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.cli import run_detect
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


def test_inject_attack_condition_fields_when_protocol_has_single_condition(monkeypatch) -> None:
    """
    功能：协议唯一条件时，detect 记录应补齐 attack family 与 params_version。

    Ensure detect record gets attack metadata injected when protocol declares
    exactly one condition.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    protocol_spec = {
        "params_versions": {
            "rotate::v1": {"angle": 5}
        }
    }
    monkeypatch.setattr(run_detect.protocol_loader, "load_attack_protocol_spec", lambda cfg: protocol_spec)

    record: Dict[str, Any] = {"operation": "detect"}
    run_detect._inject_attack_condition_fields(record, {"evaluate": {}})

    assert record.get("attack_family") == "rotate"
    assert record.get("attack_params_version") == "v1"
    assert isinstance(record.get("attack"), dict)
    assert record["attack"].get("family") == "rotate"
    assert record["attack"].get("params_version") == "v1"


def test_inject_attack_condition_fields_skip_when_protocol_is_ambiguous(monkeypatch) -> None:
    """
    功能：协议存在多个条件时，detect 记录不应强行写入 attack 条件。

    Ensure no injection occurs when protocol has multiple conditions.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    protocol_spec = {
        "params_versions": {
            "rotate::v1": {"angle": 5},
            "noise::v2": {"sigma": 0.1},
        }
    }
    monkeypatch.setattr(run_detect.protocol_loader, "load_attack_protocol_spec", lambda cfg: protocol_spec)

    record: Dict[str, Any] = {"operation": "detect"}
    run_detect._inject_attack_condition_fields(record, {"evaluate": {}})

    assert record.get("attack_family") is None
    assert record.get("attack_params_version") is None
    assert record.get("attack") is None
