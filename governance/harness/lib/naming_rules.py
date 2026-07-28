"""提供通用命名治理规则。"""

from __future__ import annotations

import re
from pathlib import Path

SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_WEAK_TOKEN_PATTERN = re.compile(
    r"(?:^|[_-])(?:new|old|best|final|proxy|v\d+(?:v\d+)*|p\d+|stage[_-]?\d+)(?:$|[_-])",
    re.IGNORECASE,
)
FORBIDDEN_WEAK_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:new|old|best|final|proxy|v\d+(?:v\d+)*|p\d+|stage[_-]?\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
ALLOWED_LITERAL_FILE_NAMES = {"README.md", "AGENTS.md", ".gitignore", "pyproject.toml", "__init__.py"}
ALLOWED_DIRECTORY_NAMES = {".codex", ".git", ".pytest_cache", "__pycache__"}
ALLOWED_FILE_SUFFIXES = {
    ".drawio",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def is_snake_case_name(name: str) -> bool:
    """判断名称是否为 snake_case。"""
    return bool(SNAKE_CASE_PATTERN.fullmatch(name))


def is_allowed_directory_name(name: str) -> bool:
    """判断目录名是否满足正式命名规则。"""
    return name in ALLOWED_DIRECTORY_NAMES or is_snake_case_name(name)


def is_allowed_file_name(name: str) -> bool:
    """判断文件名是否满足正式命名规则。"""
    if name in ALLOWED_LITERAL_FILE_NAMES:
        return True
    if name.endswith(".skill.md"):
        return is_snake_case_name(name[: -len(".skill.md")])
    path = Path(name)
    return path.suffix in ALLOWED_FILE_SUFFIXES and is_snake_case_name(path.stem)


def has_weak_semantic_token(name: str) -> bool:
    """判断名称是否包含弱语义词。"""
    return bool(FORBIDDEN_WEAK_TOKEN_PATTERN.search(name))


def has_weak_semantic_text(text: str) -> bool:
    """判断注释、docstring 或 Notebook source 是否包含独立弱语义词。"""
    return bool(FORBIDDEN_WEAK_TEXT_PATTERN.search(text))
