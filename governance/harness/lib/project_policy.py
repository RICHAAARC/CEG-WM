"""加载项目根目录与依赖治理策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_compatible_yaml(path: Path) -> dict[str, Any]:
    """加载采用 JSON 语法子集编写的 YAML 1.2 策略。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_root_policy(root: str | Path) -> dict[str, Any]:
    """加载根目录登记策略。"""
    root_path = Path(root)
    return load_json_compatible_yaml(root_path / "governance" / "policies" / "project_roots.yaml")


def load_skill_policy(root: str | Path) -> dict[str, Any]:
    """加载项目级 Codex skill 登记策略。"""
    root_path = Path(root)
    return load_json_compatible_yaml(root_path / "governance" / "policies" / "project_skills.yaml")


def load_notebook_policy(root: str | Path) -> dict[str, Any]:
    """加载 Notebook 位置与提交状态策略。"""
    root_path = Path(root)
    return load_json_compatible_yaml(root_path / "governance" / "policies" / "notebook_rules.yaml")


def governed_roots(root: str | Path) -> tuple[str, ...]:
    """返回明确登记且需要内容审计的根目录。"""
    policy = load_root_policy(root)
    return tuple(
        name
        for name, metadata in policy["root_registry"].items()
        if metadata.get("audited", False)
    )
