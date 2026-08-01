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
FORBIDDEN_ORDINAL_IDENTITY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[ap](?:[-_]?\d+)[a-z]?(?:[-_][a-z0-9]+)*|"
    r"[crs](?:[-_]?\d+)[a-z]?(?:[-_][a-z0-9]+)*|"
    r"(?:runtime[-_ ]*)?batch[-_ ]?\d+|stage[-_ ]?\d+)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_LOCAL_MATH_NOTATION_PATTERN = re.compile(
    r"`[cs]_\d+(?:\(w\))?`|(?<![A-Za-z0-9_])[cs]_\d+(?:\(w\))?(?=\s*=)",
    re.IGNORECASE,
)
MALFORMED_SEMANTIC_NUMERIC_SUFFIX_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[a-z][a-z0-9]*_)+[a-z][a-z0-9]*/\d+"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# These are exact scientific/platform tokens, not ordinal work-package identities.
ALLOWED_NARROW_SEMANTIC_LITERALS = frozenset(
    {
        "relative_l2",
        "F32",
        "RGB8",
        "P95",
        "x86_64",
        "L4",
        "SHA-256",
        "SHA256",
    }
)
_ALLOWED_NARROW_LITERAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(
        sorted(
            (re.escape(value) for value in ALLOWED_NARROW_SEMANTIC_LITERALS),
            key=len,
            reverse=True,
        )
    )
    + r")(?![A-Za-z0-9])",
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
    return bool(FORBIDDEN_WEAK_TOKEN_PATTERN.search(_scrub_allowed_literals(name)))


def has_weak_semantic_text(text: str) -> bool:
    """判断注释、docstring 或 Notebook source 是否包含独立弱语义词。"""
    return bool(FORBIDDEN_WEAK_TEXT_PATTERN.search(_scrub_allowed_literals(text)))


def has_weak_semantic_identity_value(text: str) -> bool:
    """Reject a weak token only when it is the complete formal identity value."""
    scrubbed = _scrub_allowed_literals(text).strip()
    return bool(scrubbed and FORBIDDEN_WEAK_TEXT_PATTERN.fullmatch(scrubbed))


def has_ordinal_identity_text(text: str) -> bool:
    """Reject legacy ordinal identities while preserving narrow literal terms."""
    return bool(FORBIDDEN_ORDINAL_IDENTITY_PATTERN.search(_scrub_semantics(text)))


def has_malformed_semantic_numeric_suffix(text: str) -> bool:
    """Reject a semantic snake-case label mechanically suffixed with ``/number``."""
    return bool(MALFORMED_SEMANTIC_NUMERIC_SUFFIX_PATTERN.search(text))


def ordinal_identity_tokens(text: str) -> tuple[str, ...]:
    """Return normalized legacy ordinal identities found in one value."""
    scrubbed = _scrub_semantics(text)
    return tuple(match.group(0).lower() for match in FORBIDDEN_ORDINAL_IDENTITY_PATTERN.finditer(scrubbed))


def has_ordinal_identity_polysemy(
    bindings: list[tuple[str, str]],
) -> bool:
    """Detect one ordinal token bound to multiple formal semantic identities."""
    meanings: dict[str, set[str]] = {}
    for ordinal, identity in bindings:
        for token in ordinal_identity_tokens(ordinal):
            meanings.setdefault(token, set()).add(identity)
    return any(len(values) > 1 for values in meanings.values())


def _scrub_allowed_literals(text: str) -> str:
    return _ALLOWED_NARROW_LITERAL_PATTERN.sub("", text)


def _scrub_semantics(text: str) -> str:
    without_literals = _scrub_allowed_literals(text)
    return _LOCAL_MATH_NOTATION_PATTERN.sub("", without_literals)
