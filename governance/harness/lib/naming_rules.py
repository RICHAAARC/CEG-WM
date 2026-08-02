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
    r"(?<![A-Za-z0-9])(?P<core>[A-Za-z](?:[A-Za-z_-]+?\d+|\d{2,}))"
    r"(?=$|[_-]?[A-Za-z])",
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
LOCAL_MATH_BINDING_NAMES = frozenset({"C_0", "C_1", "S_0"})
_BACKTICKED_LOCAL_MATH_NOTATION_PATTERN = re.compile(
    r"`(?:C_0|C_1(?:\(w\))?|S_0)`"
)
_IMMEDIATELY_DEFINED_LOCAL_MATH_NOTATION_PATTERN = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?:\#[ \t]*)?)(?:C_0|C_1(?:\(w\))?|S_0)(?=[ \t]*=)"
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
EXACT_VERSION_CONTEXTS = frozenset(
    {
        "access_identity",
        "bootstrap_failure_schema_version",
        "bootstrap_schema_version",
        "content_operation_semantic_version",
        "current_execution_access_identity",
        "declaration_contract",
        "delivery_manifest_schema_version",
        "diagnostic_schema_version",
        "entrypoint_schema_version",
        "execution_identity",
        "execution_schema_version",
        "expected_run_phase_id",
        "expected_bootstrap_schema_version",
        "frozen_protocol_id",
        "frozen_record_collection_schema_version",
        "frozen_record_schema_version",
        "geometry_operation_identity",
        "geometry_operation_semantic_version",
        "hf_only_reference_compact_manifest_schema_version",
        "hf_only_reference_metric_implementation_schema_version",
        "hf_only_reference_prompt_roster_schema_version",
        "hf_only_reference_schema_version",
        "hf_only_threshold_fit_record_schema_version",
        "identity_schema_version",
        "input_manifest_schema_version",
        "internal_record_field_registry_version",
        "internal_validation_protocol_id",
        "internal_validation_record_collection_schema_version",
        "internal_validation_record_schema_version",
        "legacy_internal_validation_protocol_id",
        "legacy_protocol_compatibility",
        "legacy_protocol_id",
        "manifest_id",
        "manifest_schema_version",
        "metric_schema_version",
        "model_revision",
        "operation_identity",
        "output_artifact_schema",
        "package_schema_version",
        "prompt_identity",
        "protocol_id",
        "record_collection_schema_version",
        "record_schema_version",
        "registered_key_family_id",
        "registry_version",
        "result_schema_version",
        "run_phase_id",
        "runtime_schema_version",
        "schema_version",
        "semantic_version",
        "split_manifest_protocol_id",
        "synthetic_model_revision",
        "upstream_commit",
        "wrong_key_roster_id",
    }
)
_COMPACT_VERSION_CONTEXTS = tuple(
    sorted(
        ((context, context.replace("_", "")) for context in EXACT_VERSION_CONTEXTS),
        key=lambda item: len(item[1]),
        reverse=True,
    )
)
_COMPACT_CONTEXT_MORPHOLOGICAL_CONTINUATIONS = {
    "_id": ("entity", "entifier"),
    "_version": ("ing", "ingpolicy"),
}
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
    scrubbed = _scrub_semantics(text)
    return bool(
        FORBIDDEN_VERSION_SHORTHAND_PATTERN.search(scrubbed)
        or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
    )


def has_mechanical_identity_token_in_text(text: str) -> bool:
    """Detect identifier-shaped mechanical suffixes inside code prose."""
    scrubbed = _scrub_semantics(text)
    for token in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", scrubbed):
        if has_generic_mechanical_numeric_suffix(token):
            return True
    return False


def has_weak_semantic_identity_value(
    text: str,
    *,
    version_context: str | None = None,
) -> bool:
    """Reject shorthand or unknown tokens only in a formal identity value."""
    scrubbed = _scrub_contextual_semantic_roles(
        _normalize_identity_boundaries(_scrub_allowed_literals(text))
    ).strip()
    return bool(
        scrubbed
        and (
            FORBIDDEN_VERSION_SHORTHAND_PATTERN.fullmatch(scrubbed)
            or FORBIDDEN_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(scrubbed)
            or has_generic_mechanical_numeric_suffix(
                scrubbed,
                version_context=version_context,
            )
            or FORBIDDEN_UNKNOWN_IDENTITY_PATTERN.search(scrubbed)
        )
    )


def has_weak_semantic_path_name(name: str) -> bool:
    """Reject weak shorthand or unknown tokens in a business path basename."""
    scrubbed = _scrub_contextual_semantic_roles(
        _normalize_identity_boundaries(_scrub_allowed_literals(name))
    )
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


def has_generic_mechanical_numeric_suffix(
    text: str,
    *,
    version_context: str | None = None,
) -> bool:
    """Reject unexplained numeric suffixes while preserving explicit versions and dtypes."""
    candidate_text = _scrub_allowed_literals(text).strip().strip("_-")
    candidate_text = _scrub_compact_numeric_domain_roles(
        candidate_text,
        version_context=version_context,
    )
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", candidate_text):
        return False
    if re.search(r"(?:^|[_-])v\d+(?=[_-])", candidate_text, re.IGNORECASE):
        return True
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
    if candidate_text.lower() in {"log1p", "adaptive_avg_pool2d"}:
        return False
    if re.search(
        r"(?:^|[_-])(?:average|kernel)[_-]\d+x\d+(?:$|[_-])",
        candidate_text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"(?:^|[_-])(?:quantile|critical)[-_]?\d{2,4}$",
        candidate_text,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"(?:^|[_-])cp_upper_95$", candidate_text, re.IGNORECASE):
        return False
    if re.fullmatch(r"(?:utf[-_]?8|ieee[-_]?754)", candidate_text, re.IGNORECASE):
        return False
    return GENERIC_MECHANICAL_NUMERIC_SUFFIX_PATTERN.search(candidate_text) is not None


def _scrub_compact_numeric_domain_roles(
    text: str,
    *,
    version_context: str | None = None,
) -> str:
    """Scrub compact numeric tokens only when the full name proves a real domain role."""
    if re.fullmatch(r"(?:b64(?:encode|decode)|Base64Error)", text):
        return re.sub(r"(?:b64|Base64)", "encoding", text, count=1)
    if re.fullmatch(r"StableDiffusion3Pipeline", text):
        return "StableDiffusionPipeline"

    normalized = _normalize_identity_boundaries(text)
    if re.fullmatch(r"Stable_Diffusion3_Pipeline", normalized):
        return "Stable_Diffusion_Pipeline"
    normalized = re.sub(
        r"(?:^|(?<=_))one_sided_95(?=_|$)",
        "one_sided_confidence",
        normalized,
        flags=re.IGNORECASE,
    )
    if is_explicit_version_context(version_context):
        normalized_context = canonical_version_context(version_context or "")
        if normalized_context == "legacy_protocol_compatibility" and re.fullmatch(
            r"v\d+_structure_readable_but_semantically_incompatible_and_not_revalidatable_as_v\d+",
            normalized,
            re.IGNORECASE,
        ):
            normalized = re.sub(r"v\d+", "version", normalized, flags=re.IGNORECASE)
        else:
            normalized = re.sub(
                r"(?:^|(?<=[_-]))v\d+$",
                "version",
                normalized,
                flags=re.IGNORECASE,
            )
            if (
                normalized_context == "model_revision"
                and normalized == "model_revision_1"
            ):
                normalized = "registered_model_revision"
    if re.search(r"(?:^|_)sd(?:3|35)(?:_|$)", normalized, re.IGNORECASE) and (
        re.search(
            r"(?:^|_)(?:runtime|backend|configuration|adapter|pipeline|flowmatch|gpu|prompt|conditioning)(?:_|$)",
            normalized,
            re.IGNORECASE,
        )
        or re.search(r"(?:^|_)empty_text(?:_|$)", normalized, re.IGNORECASE)
        or re.search(r"(?:^|_)real_to_q_to_k(?:_|$)", normalized, re.IGNORECASE)
    ):
        normalized = re.sub(
            r"(?:^|(?<=_))sd(?:3|35)(?=_|$)",
            "sd_model",
            normalized,
            flags=re.IGNORECASE,
        )
    normalized = re.sub(
        r"(?:^|(?<=_))Rgb8(?=_(?:Image|Quality)(?:_|$))",
        "rgb",
        normalized,
    )
    return normalized


def canonical_version_context(context: str) -> str:
    """Canonicalize a possible version context for exact outer-audit matching."""
    normalized = _normalize_identity_boundaries(context).strip("_-").lower()
    return re.sub(r"-+", "_", normalized)


def is_explicit_version_context(context: str | None) -> bool:
    """Recognize outer-audit contexts that explicitly own a persisted revision."""
    if context is None:
        return False
    return canonical_version_context(context) in EXACT_VERSION_CONTEXTS


def is_noncanonical_version_context(
    context: str | None,
    *,
    surface: str = "python",
) -> bool:
    """Detect compact aliases or wrappers around any exact version context."""
    if context is None:
        return False
    canonical = canonical_version_context(context)
    if canonical in EXACT_VERSION_CONTEXTS:
        allowed_spellings = {canonical}
        if surface == "python":
            allowed_spellings.add(canonical.upper())
        return context not in allowed_spellings
    if any(
        re.search(rf"(?:^|_){re.escape(exact_context)}(?:_|$)", canonical)
        for exact_context in EXACT_VERSION_CONTEXTS
    ):
        return True
    compact = re.sub(r"[_-]+", "", context).lower()
    for exact_context, compact_exact in _COMPACT_VERSION_CONTEXTS:
        start = compact.find(compact_exact)
        while start >= 0:
            continuation = compact[start + len(compact_exact) :]
            is_word_continuation = any(
                exact_context.endswith(role)
                and continuation in allowed_continuations
                for role, allowed_continuations in (
                    _COMPACT_CONTEXT_MORPHOLOGICAL_CONTINUATIONS.items()
                )
            )
            if not is_word_continuation:
                return True
            start = compact.find(compact_exact, start + 1)
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


def _normalize_identity_boundaries(text: str) -> str:
    """Expose CamelCase word boundaries without changing the governed value."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)


def _scrub_contextual_semantic_roles(text: str) -> str:
    """Preserve exact domain/action roles that are not placeholder identities."""
    return re.sub(
        r"(?i)(?:^|[_-])(?:final_image|copy_gate|result_copy|atomic_copy|fail_closed_for_copy)(?=$|[_-])",
        "_",
        text,
    )


def _scrub_semantics(text: str) -> str:
    without_literals = _scrub_allowed_literals(text)
    without_backticked_math = _BACKTICKED_LOCAL_MATH_NOTATION_PATTERN.sub(
        "",
        without_literals,
    )
    return _IMMEDIATELY_DEFINED_LOCAL_MATH_NOTATION_PATTERN.sub(
        lambda match: match.group("prefix"),
        without_backticked_math,
    )
