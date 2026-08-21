"""Small JSON-compatible audit reports."""

from __future__ import annotations

from typing import Any


def build_report(
    audit_name: str,
    violations: list[dict[str, Any]],
    checked_paths: list[str],
) -> dict[str, Any]:
    return {
        "audit_name": audit_name,
        "decision": "fail" if violations else "pass",
        "violations": violations,
        "checked_paths": checked_paths,
        "summary": {
            "violation_count": len(violations),
            "checked_path_count": len(checked_paths),
        },
    }
