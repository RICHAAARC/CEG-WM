"""Check notebook placement, size, outputs, and governance imports."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.policy import load_policy
from governance.harness.lib.report import build_report

SKIP_PARTS = {".git", ".venv", ".conda", "__pycache__", ".pytest_cache", "outputs", "release_packages"}


def _imports_governance(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module else []
        else:
            continue
        if any(module == "governance" or module.startswith("governance.") for module in modules):
            return True
    return False


def run_audit(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    policy = load_policy(root_path, "notebooks")
    allowed_root = Path(policy["allowed_root"])
    violations: list[dict[str, Any]] = []
    checked_paths = ["governance/policies/notebooks.json"]

    for path in sorted(root_path.rglob("*.ipynb")):
        relative = path.relative_to(root_path)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        checked_paths.append(relative.as_posix())
        if relative != allowed_root and allowed_root not in relative.parents:
            violations.append({"path": relative.as_posix(), "reason": "notebook_outside_allowed_root"})
            continue
        if path.stat().st_size > int(policy["max_notebook_bytes"]):
            violations.append({"path": relative.as_posix(), "reason": "notebook_size_limit_exceeded"})
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            violations.append({"path": relative.as_posix(), "reason": "notebook_unreadable", "detail": str(error)})
            continue
        for index, cell in enumerate(document.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            source_text = "".join(source) if isinstance(source, list) else str(source)
            if _imports_governance(source_text):
                violations.append({"path": relative.as_posix(), "reason": "control_plane_import_forbidden", "cell_index": index})
            if policy["committed_outputs"] == "fail" and cell.get("outputs"):
                violations.append({"path": relative.as_posix(), "reason": "committed_notebook_output", "cell_index": index})
            if policy["committed_execution_counts"] == "fail" and cell.get("execution_count") is not None:
                violations.append({"path": relative.as_posix(), "reason": "committed_notebook_execution_count", "cell_index": index})

    return build_report("notebooks", violations, checked_paths)


def main() -> None:
    report = run_audit(Path.cwd())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["decision"] == "pass" else 1)


if __name__ == "__main__":
    main()
