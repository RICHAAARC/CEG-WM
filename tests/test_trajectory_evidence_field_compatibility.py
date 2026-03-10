"""
File purpose: Trajectory evidence field compatibility and priority tests.
Module type: General module
"""

from __future__ import annotations

from typing import Dict, Any

from main.watermarking.detect.orchestrator import (
    _resolve_trajectory_tap_status,
    _resolve_trajectory_absent_reason,
    _inject_trajectory_audit_fields,
)


def test_trajectory_tap_status_prefers_audit_field() -> None:
    """
    功能：新字段存在时应优先读取 audit.trajectory_tap_status。

    New trajectory tap status field in audit should have higher priority.

    Args:
        None.

    Returns:
        None.
    """
    trajectory_evidence: Dict[str, Any] = {
        "status": "absent",
        "audit": {
            "trajectory_tap_status": "ok"
        }
    }

    assert _resolve_trajectory_tap_status(trajectory_evidence) == "ok"


def test_trajectory_tap_status_uses_flat_status_field_when_audit_absent() -> None:
    """
    功能：新字段缺失时应兼容回退到旧 status 字段。

    Flat status field should be used when audit.trajectory_tap_status is absent.

    Args:
        None.

    Returns:
        None.
    """
    trajectory_evidence: Dict[str, Any] = {
        "status": "absent"
    }

    assert _resolve_trajectory_tap_status(trajectory_evidence) == "absent"


def test_trajectory_absent_reason_prefers_audit_field() -> None:
    """
    功能：新字段存在时应优先读取 audit.trajectory_absent_reason。

    New absent reason field in audit should have higher priority.

    Args:
        None.

    Returns:
        None.
    """
    trajectory_evidence: Dict[str, Any] = {
        "trajectory_absent_reason": "unsupported_pipeline",
        "audit": {
            "trajectory_absent_reason": "tap_disabled"
        }
    }

    assert _resolve_trajectory_absent_reason(trajectory_evidence) == "tap_disabled"


def test_inject_trajectory_audit_fields_from_flat_trajectory_source() -> None:
    """
    功能：旧平铺字段输入应可写入 content_evidence.audit 新字段。

    Flat trajectory fields should be injected into content audit fields.

    Args:
        None.

    Returns:
        None.
    """
    content_evidence_payload: Dict[str, Any] = {
        "status": "ok",
        "score": 0.7,
        "audit": {
            "impl_identity": "test_impl"
        }
    }
    trajectory_evidence: Dict[str, Any] = {
        "status": "absent",
        "trajectory_absent_reason": "unsupported_pipeline"
    }

    _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)

    assert content_evidence_payload["audit"]["trajectory_tap_status"] == "absent"
    assert content_evidence_payload["audit"]["trajectory_absent_reason"] == "unsupported_pipeline"
