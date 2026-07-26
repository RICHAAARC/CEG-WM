"""提供受治理文本文件扫描能力。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from governance.harness.lib.project_policy import governed_roots, load_root_policy

SKIP_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "outputs",
    "audit_reports",
    "dist",
    "build",
}

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".tar", ".gz", ".7z", ".exe", ".dll", ".so", ".pyc", ".pyd",
}

def should_skip_path(path: str | Path) -> bool:
    """判断路径是否属于缓存、输出或构建产物。"""
    candidate = Path(path)
    return any(part in SKIP_DIRECTORY_NAMES for part in candidate.parts)


def iter_text_files(root: str | Path) -> Iterator[Path]:
    """遍历目录下的文本候选文件。"""
    root_path = Path(root)
    if not root_path.exists():
        return
    for path in root_path.rglob("*"):
        if not path.is_file() or should_skip_path(path):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        yield path


def iter_governed_paths(root: str | Path) -> Iterator[Path]:
    """只遍历 policy 明确登记为 audited 的路径。"""
    root_path = Path(root)
    policy = load_root_policy(root_path)
    for relative_root in governed_roots(root_path):
        candidate = root_path / relative_root
        if not candidate.exists() or should_skip_path(candidate):
            continue
        if candidate.is_file():
            yield candidate
        else:
            yield candidate
            for path in candidate.rglob("*"):
                if not should_skip_path(path):
                    yield path
    for relative_file in policy.get("governed_files", []):
        candidate = root_path / relative_file
        if candidate.exists() and not should_skip_path(candidate):
            yield candidate


def iter_governed_text_files(root: str | Path) -> Iterator[Path]:
    """按默认治理根遍历文本文件。"""
    root_path = Path(root)
    for path in iter_governed_paths(root_path):
        if path.is_file() and path.suffix.lower() not in BINARY_SUFFIXES:
            yield path
