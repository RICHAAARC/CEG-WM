"""Audit project imports against the minimal layer policy."""

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


def _source_layer(relative: Path, layers: dict[str, Any]) -> str | None:
    normalized = relative.as_posix()
    ordered = sorted(layers.items(), key=lambda item: len(item[1]["path"]), reverse=True)
    for layer_name, rule in ordered:
        layer_path = str(rule["path"]).strip("/")
        if normalized == layer_path or normalized.startswith(f"{layer_path}/"):
            return layer_name
    return None


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _imported_layer(module: str, layer_names: tuple[str, ...]) -> str | None:
    for layer_name in sorted(layer_names, key=len, reverse=True):
        if module == layer_name or module.startswith(f"{layer_name}."):
            return layer_name
    return None


def run_audit(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    policy = load_policy(root_path, "dependencies")
    layers = policy["layers"]
    layer_names = tuple(layers)
    forbidden = str(policy["forbidden_dependency"])
    violations: list[dict[str, Any]] = []
    checked_paths: list[str] = []

    for layer_name, rule in layers.items():
        layer_root = root_path / rule["path"]
        if not layer_root.exists():
            continue
        for path in sorted(layer_root.rglob("*.py")):
            relative = path.relative_to(root_path)
            if any(part in SKIP_PARTS for part in relative.parts):
                continue
            checked_paths.append(relative.as_posix())
            source_layer = _source_layer(relative, layers)
            if source_layer != layer_name:
                violations.append({
                    "path": relative.as_posix(),
                    "reason": "source_layer_resolution_failed",
                    "expected": layer_name,
                    "observed": source_layer,
                })
                continue
            try:
                imported_modules = _imported_modules(path)
            except (OSError, UnicodeError, SyntaxError) as error:
                violations.append({
                    "path": relative.as_posix(),
                    "reason": "python_source_unreadable",
                    "detail": str(error),
                })
                continue
            allowed = set(rule["allowed_project_dependencies"])
            for module in imported_modules:
                if module == forbidden or module.startswith(f"{forbidden}."):
                    violations.append({
                        "path": relative.as_posix(),
                        "reason": "control_plane_import_forbidden",
                        "source_layer": source_layer,
                        "imported_module": module,
                    })
                    continue
                target_layer = _imported_layer(module, layer_names)
                if target_layer and target_layer != source_layer and target_layer not in allowed:
                    violations.append({
                        "path": relative.as_posix(),
                        "reason": "project_layer_dependency_forbidden",
                        "source_layer": source_layer,
                        "imported_layer": target_layer,
                    })

    for delivery_root_name in policy.get("delivery_code_roots", []):
        delivery_root = root_path / delivery_root_name
        if not delivery_root.exists():
            continue
        for path in sorted(delivery_root.rglob("*.py")):
            relative = path.relative_to(root_path)
            checked_paths.append(relative.as_posix())
            for module in _imported_modules(path):
                if module == forbidden or module.startswith(f"{forbidden}."):
                    violations.append({
                        "path": relative.as_posix(),
                        "reason": "control_plane_import_forbidden",
                        "source_layer": delivery_root_name,
                        "imported_module": module,
                    })

    return build_report("dependencies", violations, checked_paths)


def main() -> None:
    report = run_audit(Path.cwd())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["decision"] == "pass" else 1)


if __name__ == "__main__":
    main()
