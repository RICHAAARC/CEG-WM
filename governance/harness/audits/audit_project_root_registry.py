"""审计顶级目录是否已在治理策略中登记。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.project_policy import load_root_policy


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    policy = load_root_policy(root_path)
    registered_roots = set(policy["root_registry"])
    violations = []
    checked_paths = []
    for path in sorted(root_path.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_dir():
            continue
        checked_paths.append(path.name)
        if path.name not in registered_roots:
            violations.append({"path": path.name, "reason": "unregistered_project_root"})
    return build_report("audit_project_root_registry", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
