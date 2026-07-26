"""提供统一 JSON 审计报告能力。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_report(audit_name: str, decision: str, violations: list[dict[str, Any]], checked_paths: list[str]) -> dict[str, Any]:
    """构造标准审计报告。"""
    if decision not in {"pass", "fail"}:
        raise ValueError(f"不支持的审计结论: {decision}")
    return {
        "audit_name": audit_name,
        "decision": decision,
        "violations": violations,
        "checked_paths": [str(Path(path)) for path in checked_paths],
        "summary": {
            "violation_count": len(violations),
            "checked_path_count": len(checked_paths),
        },
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> None:
    """将审计报告写入 UTF-8 JSON 文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def exit_with_report(report: dict[str, Any]) -> None:
    """打印报告并用退出码表达审计结果。"""
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.get("decision") == "pass" else 1)
