"""统一执行全部 harness 审计。"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import write_report

AUDIT_MODULE_NAMES = [
    "governance.harness.audits.audit_project_root_registry",
    "governance.harness.audits.audit_naming_conventions",
    "governance.harness.audits.audit_file_organization_contract",
    "governance.harness.audits.audit_utf8_encoding_contract",
    "governance.harness.audits.audit_test_case_constraints",
    "governance.harness.audits.audit_notebook_boundaries",
    "governance.harness.audits.audit_research_definition",
    "governance.harness.audits.audit_method_readiness",
    "governance.harness.audits.audit_skill_file_presence",
    "governance.harness.audits.audit_placeholder_random_fields",
    "governance.harness.audits.audit_dependency_boundaries",
    "governance.harness.audits.audit_release_extraction_contract",
]


def run_all_audits(root: str | Path) -> dict[str, Any]:
    """执行全部审计并写入汇总报告。"""
    root_path = Path(root)
    output_root = root_path / "audit_reports"
    results = []
    for module_name in AUDIT_MODULE_NAMES:
        module = importlib.import_module(module_name)
        report = module.run_audit(root_path)
        write_report(report, output_root / f"{report['audit_name']}.json")
        results.append(report)
    fail_count = sum(1 for report in results if report["decision"] != "pass")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_decision": "fail" if fail_count else "pass",
        "audit_results": results,
        "summary": {
            "total_audits": len(results),
            "pass_count": len(results) - fail_count,
            "fail_count": fail_count,
        },
    }
    write_report(summary, output_root / "harness_audit_summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""
    arguments = argv or sys.argv
    root = Path(arguments[1]) if len(arguments) > 1 else ROOT
    summary = run_all_audits(root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["overall_decision"] == "pass" else 1)


if __name__ == "__main__":
    main(sys.argv)
