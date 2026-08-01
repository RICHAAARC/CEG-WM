"""提供通用命名治理规则。"""

from __future__ import annotations

import re
from pathlib import Path

SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
UNKNOWN_OR_TEMPORARY_IDENTITY_WORDS = frozenset(
    {
        "tmp",
        "temp",
        "misc",
        "other",
        "todo",
        "tbd",
        "dummy",
        "fake",
        "mock",
        "proxy",
        "new",
        "old",
        "latest",
        "best",
        "final",
        "backup",
        "copy",
        "foo",
        "bar",
    }
)
NUMBERED_RESPONSIBILITY_WORDS = frozenset(
    {
        "phase",
        "step",
        "stage",
        "batch",
        "tier",
        "level",
        "group",
        "track",
        "route",
        "gate",
        "case",
        "option",
        "variant",
        "module",
        "component",
        "method",
        "model",
        "baseline",
        "run",
        "experiment",
        "trial",
    }
)
_NUMBERED_RESPONSIBILITY_ALTERNATION = "|".join(
    sorted(NUMBERED_RESPONSIBILITY_WORDS, key=len, reverse=True)
)
_UNKNOWN_IDENTITY_ALTERNATION = "|".join(
    sorted(UNKNOWN_OR_TEMPORARY_IDENTITY_WORDS, key=len, reverse=True)
)
FORBIDDEN_VERSION_SHORTHAND_PATTERN = re.compile(
    r"(?:^|[_-])(?:v\d+(?:v\d+)*|p\d+)(?:$|[_-])",
    re.IGNORECASE,
)
FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:detector|metric|config|result|method)"
    r"(?:[-_]?v?[-_]?\d+)(?:[-_][a-z0-9]+)*(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
GENERIC_MECHANICAL_NUMERIC_SUFFIX_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z_-]{1,})\d+"
    r"(?![A-Za-z0-9_])",
)
SCIENTIFIC_NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?:binary|float|bfloat|u?int|utf)\d+",
    re.IGNORECASE,
)
FORBIDDEN_UNKNOWN_IDENTITY_PATTERN = re.compile(
    rf"(?:^|[_-])(?:{_UNKNOWN_IDENTITY_ALTERNATION})(?:$|[_-])",
    re.IGNORECASE,
)
FORBIDDEN_ORDINAL_IDENTITY_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?:[a-uw-z](?:[-_]?\d+)[a-z]?(?:[-_][a-z0-9]+)*|"
    rf"(?-i:[A-UW-Z](?:[-_]?\d+)[A-Za-z0-9_]*)|"
    rf"(?-i:[A-Za-z0-9_]*[a-z][A-UW-Z](?:[-_]?\d+)[A-Za-z]?(?:[A-Z_][A-Za-z0-9_]*)*)|"
    rf"[a-z0-9_-]*(?:{_NUMBERED_RESPONSIBILITY_ALTERNATION})[-_ ]*\d+"
    r"(?:[-_][a-z0-9]+)*)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
RESPONSIBILITY_IDENTITY_CORE_PATTERN = re.compile(
    rf"(?:{_NUMBERED_RESPONSIBILITY_ALTERNATION})[-_ ]*\d+",
    re.IGNORECASE,
)
ORDINAL_IDENTITY_CORE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?:[a-uw-z](?:[-_]?\d+)[a-z]?|"
    rf"(?:{_NUMBERED_RESPONSIBILITY_ALTERNATION})[-_ ]*\d+)"
    r"(?![A-Za-z0-9])",
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
SCIENTIFIC_L2_IDENTIFIER_PATTERN = re.compile(
    r"(?:^|_)l2(?:_|$)",
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
        "SD3.5",
    }
)
# These registered field names carry an explicit numeric role in the field
# registry: two positional runtime prompt inputs and one statistical quantile.
# Keep this exact so it cannot become a general numeric-suffix exemption.
ALLOWED_REGISTERED_NUMERIC_FIELD_ROLES = frozenset(
    {
        "prompt_2",
        "prompt_3",
        "student_t_critical_975",
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
    scrubbed = _scrub_allowed_literals(name)
    return bool(
        FORBIDDEN_VERSION_SHORTHAND_PATTERN.search(scrubbed)
        or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
    )


def has_weak_semantic_text(text: str) -> bool:
    """判断注释、docstring 或 Notebook source 是否包含独立弱语义词。"""
    scrubbed = _scrub_allowed_literals(text)
    return bool(
        FORBIDDEN_VERSION_SHORTHAND_PATTERN.search(scrubbed)
        or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
    )


def has_weak_semantic_identity_value(text: str) -> bool:
    """Reject shorthand or unknown tokens only in a formal identity value."""
    scrubbed = _scrub_contextual_semantic_roles(_scrub_allowed_literals(text)).strip()
    return bool(
        scrubbed
        and (
            FORBIDDEN_VERSION_SHORTHAND_PATTERN.fullmatch(scrubbed)
            or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
            or has_generic_mechanical_numeric_suffix(scrubbed)
            or FORBIDDEN_UNKNOWN_IDENTITY_PATTERN.search(scrubbed)
        )
    )


def has_weak_semantic_path_name(name: str) -> bool:
    """Reject weak shorthand or unknown tokens in a business path basename."""
    scrubbed = _scrub_contextual_semantic_roles(_scrub_allowed_literals(name))
    return bool(
        FORBIDDEN_VERSION_SHORTHAND_PATTERN.search(scrubbed)
        or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
        or has_generic_mechanical_numeric_suffix(scrubbed)
        or FORBIDDEN_UNKNOWN_IDENTITY_PATTERN.search(scrubbed)
    )


def has_ordinal_identity_text(text: str) -> bool:
    """Reject legacy ordinal identities while preserving narrow literal terms."""
    return bool(FORBIDDEN_ORDINAL_IDENTITY_PATTERN.search(_scrub_semantics(text)))


def has_malformed_semantic_numeric_suffix(text: str) -> bool:
    """Reject a semantic snake-case label mechanically suffixed with ``/number``."""
    return bool(MALFORMED_SEMANTIC_NUMERIC_SUFFIX_PATTERN.search(text))


def has_generic_mechanical_numeric_suffix(text: str) -> bool:
    """Reject unexplained numeric suffixes while preserving explicit versions and dtypes."""
    candidate_text = _scrub_allowed_literals(text).strip().strip("_-")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", candidate_text):
        return False
    if is_allowed_registered_numeric_field_role(candidate_text):
        return False
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate_text, re.IGNORECASE):
        return False
    if re.search(
        r"(?:^|[_-])(?:binary|float|bfloat|u?int|utf)\d+(?:$|[_-])",
        candidate_text,
        re.IGNORECASE,
    ):
        return False
    if is_scientific_l2_identifier(candidate_text):
        return False
    if re.search(
        r"(?:^|[_-])(?:quantile|critical)[-_]?\d{2,4}$",
        candidate_text,
        re.IGNORECASE,
    ):
        return False
    for match in GENERIC_MECHANICAL_NUMERIC_SUFFIX_PATTERN.finditer(candidate_text):
        candidate = match.group(0)
        if re.search(r"(?:^|[_-])v\d+$", candidate, re.IGNORECASE):
            continue
        if SCIENTIFIC_NUMERIC_LITERAL_PATTERN.fullmatch(candidate):
            continue
        return True
    return False


def is_scientific_l2_identifier(text: str) -> bool:
    """Recognize an explicit L2 norm role inside a semantic identifier."""
    return bool(SCIENTIFIC_L2_IDENTIFIER_PATTERN.search(text))


def is_allowed_registered_numeric_field_role(text: str) -> bool:
    """Recognize an exact registered positional or statistical numeric role."""
    return text in ALLOWED_REGISTERED_NUMERIC_FIELD_ROLES


def ordinal_identity_tokens(text: str) -> tuple[str, ...]:
    """Return normalized legacy ordinal identities found in one value."""
    scrubbed = _scrub_semantics(text)
    tokens = {
        match.group(0).lower()
        for match in ORDINAL_IDENTITY_CORE_PATTERN.finditer(scrubbed)
    }
    tokens.update(
        match.group(0).lower()
        for match in RESPONSIBILITY_IDENTITY_CORE_PATTERN.finditer(scrubbed)
    )
    return tuple(sorted(tokens))


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


def _scrub_contextual_semantic_roles(text: str) -> str:
    """Preserve exact domain/action roles that are not placeholder identities."""
    return re.sub(
        r"(?i)(?:^|[_-])(?:final_image|copy_gate|result_copy)(?=$|[_-])",
        "_",
        text,
    )


def _scrub_semantics(text: str) -> str:
    without_literals = _scrub_allowed_literals(text)
    return _LOCAL_MATH_NOTATION_PATTERN.sub("", without_literals)
