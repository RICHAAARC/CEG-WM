"""审计 Notebook 的位置、大小和已提交执行状态。"""

from __future__ import annotations

import json
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.naming_rules import has_weak_semantic_text
from governance.harness.lib.project_policy import governed_roots, load_notebook_policy


def _is_within(relative: Path, allowed_roots: tuple[Path, ...]) -> bool:
    return any(relative == allowed or allowed in relative.parents for allowed in allowed_roots)


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []
    policy_path = root_path / "governance" / "policies" / "notebook_rules.yaml"
    checked_paths.append(str(policy_path.relative_to(root_path)))

    try:
        policy = load_notebook_policy(root_path)
    except (OSError, ValueError) as error:
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "notebook_policy_unreadable", "detail": str(error)})
        return build_report("audit_notebook_boundaries", "fail", violations, checked_paths)

    required_fields = (
        "notebook_root",
        "allowed_notebook_roots",
        "committed_outputs",
        "committed_execution_counts",
        "max_notebook_bytes",
    )
    missing_fields = [field for field in required_fields if field not in policy]
    if missing_fields:
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "notebook_policy_field_missing", "fields": missing_fields})
        return build_report("audit_notebook_boundaries", "fail", violations, checked_paths)

    allowed_roots = tuple(Path(value) for value in policy.get("allowed_notebook_roots", []))
    if Path(policy["notebook_root"]) not in allowed_roots:
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "notebook_root_not_allowed"})
    notebook_paths: set[Path] = set()
    notebook_paths.update(root_path.glob("*.ipynb"))
    for relative_root in governed_roots(root_path):
        candidate = root_path / relative_root
        if candidate.is_dir():
            notebook_paths.update(candidate.rglob("*.ipynb"))

    for path in sorted(notebook_paths):
        relative = path.relative_to(root_path)
        checked_paths.append(relative.as_posix())
        if not _is_within(relative, allowed_roots):
            violations.append({"path": relative.as_posix(), "reason": "notebook_outside_allowed_root"})
            continue
        if path.stat().st_size > int(policy["max_notebook_bytes"]):
            violations.append({"path": relative.as_posix(), "reason": "notebook_size_limit_exceeded"})
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            violations.append({"path": relative.as_posix(), "reason": "notebook_unreadable", "detail": str(error)})
            continue
        for cell_index, cell in enumerate(document.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            source_text = "".join(source) if isinstance(source, list) else str(source)
            if has_weak_semantic_text(source_text):
                violations.append({"path": relative.as_posix(), "reason": "weak_semantic_notebook_source", "cell_index": cell_index})
            try:
                tree = ast.parse(source_text)
            except SyntaxError:
                tree = None
            if tree is not None:
                for node in ast.walk(tree):
                    imported_modules = []
                    if isinstance(node, ast.ImportFrom):
                        imported_modules = [node.module] if node.module else []
                    elif isinstance(node, ast.Import):
                        imported_modules = [alias.name for alias in node.names]
                    if any(
                        module_name == "governance" or module_name.startswith("governance.")
                        for module_name in imported_modules
                    ):
                        violations.append(
                            {
                                "path": relative.as_posix(),
                                "reason": "control_plane_import_forbidden",
                                "cell_index": cell_index,
                            }
                        )
                        break
            if policy.get("committed_outputs") == "fail" and cell.get("outputs"):
                violations.append({"path": relative.as_posix(), "reason": "committed_notebook_output", "cell_index": cell_index})
            if policy.get("committed_execution_counts") == "fail" and cell.get("execution_count") is not None:
                violations.append({"path": relative.as_posix(), "reason": "committed_notebook_execution_count", "cell_index": cell_index})

    return build_report("audit_notebook_boundaries", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
