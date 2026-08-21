"""Run the intentionally small CEG-WM harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from governance.harness.audits.dependencies import run_audit as run_dependencies
from governance.harness.audits.notebooks import run_audit as run_notebooks


def run_all(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    reports = [run_dependencies(root_path), run_notebooks(root_path)]
    return {
        "overall_decision": "fail" if any(item["decision"] != "pass" for item in reports) else "pass",
        "audits": reports,
        "summary": {
            "audit_count": len(reports),
            "failure_count": sum(item["decision"] != "pass" for item in reports),
        },
    }


def main() -> None:
    report = run_all(Path.cwd())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["overall_decision"] == "pass" else 1)


if __name__ == "__main__":
    main()
