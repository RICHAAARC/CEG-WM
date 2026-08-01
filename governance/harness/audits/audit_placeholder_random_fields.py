"""审计字段登记表中的 placeholder 与 random 规则。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.field_rules import inspect_field_registry, validate_registry_rows
from governance.harness.lib.json_report import build_report, exit_with_report


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    inspection = inspect_field_registry(root_path)
    rows = inspection.rows
    violations = [*inspection.violations, *validate_registry_rows(rows)]
    checked_paths = ["docs/reference/field_registry.md"]
    return build_report("audit_placeholder_random_fields", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
