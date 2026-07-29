"""提供 policy 驱动的项目层依赖审计能力。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from governance.harness.lib.project_policy import load_json_compatible_yaml


def load_dependency_policy(root: str | Path) -> dict[str, Any]:
    """加载分层依赖策略。"""
    root_path = Path(root)
    return load_json_compatible_yaml(root_path / "governance" / "policies" / "dependency_rules.yaml")


def extract_imported_modules(path: Path) -> list[str]:
    """从 Python 文件中提取顶层导入模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def extract_filesystem_write_calls(path: Path) -> list[str]:
    """Return explicit filesystem mutation calls used by one Python module."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    writes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in {
            "Path.write_text",
            "Path.write_bytes",
            "os.replace",
            "os.rename",
            "tempfile.NamedTemporaryFile",
        }:
            writes.append(call_name)
            continue
        if call_name not in {"open", "Path.open"}:
            continue
        mode_node = None
        if call_name == "Path.open" and node.args:
            mode_node = node.args[0]
        elif len(node.args) >= 2:
            mode_node = node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_node = keyword.value
        if (
            isinstance(mode_node, ast.Constant)
            and isinstance(mode_node.value, str)
            and any(flag in mode_node.value for flag in "wax+")
        ):
            writes.append(call_name)
    return writes


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return None
    if node.attr in {"write_text", "write_bytes", "open"}:
        return f"Path.{node.attr}"
    if isinstance(node.value, ast.Name):
        owner = node.value.id
        if owner and owner[0].isupper():
            owner = "Path"
        return f"{owner}.{node.attr}"
    return node.attr


def layer_to_path(layer_name: str) -> str:
    """将 Python 模块层名转换为仓库相对路径。"""
    return layer_name.replace(".", "/")


def get_source_layer(relative_path: Path, layer_names: tuple[str, ...]) -> str | None:
    """按最长前缀匹配源码所属层。"""
    normalized = relative_path.as_posix()
    for layer_name in sorted(layer_names, key=len, reverse=True):
        layer_path = layer_to_path(layer_name)
        if normalized == layer_path or normalized.startswith(f"{layer_path}/"):
            return layer_name
    return None


def get_imported_layer(module_name: str, layer_names: tuple[str, ...]) -> str | None:
    """按最长模块前缀匹配被导入的项目层。"""
    for layer_name in sorted(layer_names, key=len, reverse=True):
        if module_name == layer_name or module_name.startswith(f"{layer_name}."):
            return layer_name
    return None


def dependency_violation_reason(
    source_layer: str,
    imported_module: str,
    policy: dict[str, Any],
) -> str | None:
    """返回依赖违规原因；外部依赖和同层导入不构成跨层违规。"""
    forbidden_dependency = policy["forbidden_dependency"]
    if imported_module == forbidden_dependency or imported_module.startswith(f"{forbidden_dependency}."):
        return "control_plane_import_forbidden"

    layer_names = tuple(policy["layers"])
    imported_layer = get_imported_layer(imported_module, layer_names)
    if imported_layer is None or imported_layer == source_layer:
        return None
    allowed = set(policy["layers"][source_layer]["allowed_project_dependencies"])
    if imported_layer not in allowed:
        return "project_layer_dependency_forbidden"
    return None
