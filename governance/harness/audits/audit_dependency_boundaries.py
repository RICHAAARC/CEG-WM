"""审计 policy 登记的项目层依赖方向。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.dependency_rules import (
    dependency_violation_reason,
    extract_filesystem_write_calls,
    extract_imported_modules,
    get_source_layer,
    layer_to_path,
    load_dependency_policy,
)
from governance.harness.lib.file_scanner import should_skip_path
from governance.harness.lib.json_report import build_report, exit_with_report


def run_audit(root: str | Path) -> dict:
    """检查全部登记代码层的导入方向。"""
    root_path = Path(root)
    policy = load_dependency_policy(root_path)
    layer_names = tuple(policy["layers"])
    violations = []
    checked_paths = []

    for layer_name in layer_names:
        layer_root = root_path / layer_to_path(layer_name)
        if not layer_root.exists():
            continue
        for path in layer_root.rglob("*.py"):
            relative = path.relative_to(root_path)
            if should_skip_path(relative):
                continue
            checked_paths.append(str(relative))
            source_layer = get_source_layer(relative, layer_names)
            if source_layer is None:
                continue
            for module_name in extract_imported_modules(path):
                reason = dependency_violation_reason(source_layer, module_name, policy)
                if reason:
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": reason,
                            "source_layer": source_layer,
                            "imported_module": module_name,
                        }
                    )
            record_writer_layers = set(policy.get("record_writer_layers", []))
            if (
                layer_name.startswith("experiments.")
                and layer_name not in record_writer_layers
            ):
                for call_name in extract_filesystem_write_calls(path):
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": "record_write_outside_authorized_layer",
                            "source_layer": source_layer,
                            "write_call": call_name,
                        }
                    )
    forbidden_dependency = str(policy["forbidden_dependency"])
    layer_paths = {layer_to_path(layer_name) for layer_name in layer_names}
    for delivery_root in policy.get("delivery_code_roots", []):
        if delivery_root in layer_paths:
            continue
        source_root = root_path / delivery_root
        if not source_root.exists():
            continue
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(root_path)
            if should_skip_path(relative):
                continue
            checked_paths.append(str(relative))
            for module_name in extract_imported_modules(path):
                if module_name == forbidden_dependency or module_name.startswith(f"{forbidden_dependency}."):
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": "control_plane_import_forbidden",
                            "source_layer": delivery_root,
                            "imported_module": module_name,
                        }
                    )
    return build_report("audit_dependency_boundaries", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
