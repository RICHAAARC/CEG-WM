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
