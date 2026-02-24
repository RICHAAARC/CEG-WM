"""
配置覆写校验收敛

功能说明：
- 统一校验配置中的 override 段落，禁止 YAML 中出现 CLI override 字段。
- 解析 CLI override 参数，验证并应用到配置中，生成 override_applied 审计段。
- 要求 override_applied 在提供 overrides 时必须存在。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple

from main.core import digests
from main.core.contracts import ContractInterpretation
from main.policy.runtime_whitelist import RuntimeWhitelist


_DISALLOWED_OVERRIDE_KEYS = {
    "override",
    "overrides",
    "override_applied"
}

_PROTECTED_ROOT_PREFIXES = (
    "contract_",
    "whitelist_",
    "policy_path_semantics_"
)
_PROTECTED_ROOT_EXACT = {
    "external_fact_sources"
}
_PROTECTED_SUFFIXES = (
    "_bound_digest",
)


@dataclass(frozen=True)
class ParsedOverride:
    """
    功能：解析后的 CLI override 条目。

    Parsed CLI override entry.

    Args:
        key: Override key (arg_name or field_path).
        raw_value: Raw value string from CLI.
        value: Parsed JSON value.

    Returns:
        None.
    """

    key: str
    raw_value: str
    value: Any


@dataclass(frozen=True)
class ResolvedOverride:
    """
    功能：已解析并绑定 whitelist 的 override 条目。

    Resolved override entry bound to whitelist specification.

    Args:
        arg_name: Canonical arg_name from whitelist.
        field_path: Canonical field_path from whitelist.
        override_mode: Override mode (e.g., "set").
        source: Override source (e.g., "cli").
        value: Parsed override value.
        raw_key: Original key from CLI.
        raw_value: Original raw value from CLI.

    Returns:
        None.
    """

    arg_name: str
    field_path: str
    override_mode: str
    source: str
    value: Any
    raw_key: str
    raw_value: str


def validate_overrides(cfg: Dict[str, Any], interpretation: ContractInterpretation) -> None:
    """
    功能：统一校验配置中的 override 段落。

    Validate overrides embedded in config files. CLI overrides are handled
    separately and are not permitted to appear in YAML.

    Args:
        cfg: Loaded configuration mapping.
        interpretation: Structured frozen contract interpretation.

    Raises:
        TypeError: If cfg or interpretation types are invalid.
        ValueError: If overrides are present but not allowed.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    disallowed_paths = _collect_disallowed_override_paths(cfg)
    if disallowed_paths:
        # override 未被允许，必须 fail-fast。
        first_path = disallowed_paths[0]
        raise ValueError(f"override_not_allowed: field_path={first_path}")


def apply_cli_overrides(
    cfg: Dict[str, Any],
    override_args: List[str],
    whitelist: RuntimeWhitelist,
    interpretation: ContractInterpretation
) -> Dict[str, Any]:
    """
    功能：解析并应用 CLI overrides，生成 override_applied 审计段。

    Parse CLI override args, validate against whitelist, apply to cfg,
    and return override_applied audit block.

    Args:
        cfg: Configuration mapping to mutate.
        override_args: CLI override arguments (key=value) list.
        whitelist: Loaded RuntimeWhitelist for allowed overrides.
        interpretation: Frozen contract interpretation for protected fields.

    Returns:
        override_applied audit mapping.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If overrides are invalid or not allowed.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(override_args, list):
        # override_args 类型不合法，必须 fail-fast。
        raise TypeError("override_args must be list")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不合法，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不合法，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    parsed = [_parse_override_arg(item) for item in override_args]
    resolved = _resolve_overrides(parsed, whitelist)
    _require_no_duplicate_arg_names(resolved)

    applied_fields: List[Dict[str, Any]] = []
    requested_overrides: List[Dict[str, Any]] = []
    requested_kv: Dict[str, Any] = {}

    for item in resolved:
        _require_override_target_allowed(item.field_path, interpretation)
        found, old_value = _get_value_by_field_path(cfg, item.field_path)
        if not found:
            raise ValueError(f"override_field_path_missing: field_path={item.field_path}")

        new_value = _resolve_override_value(item, whitelist)
        _set_value_by_field_path(cfg, item.field_path, new_value)

        audit_old_value = old_value if old_value is not None else "<absent>"
        audit_new_value = new_value if new_value is not None else "<absent>"

        applied_fields.append({
            "arg_name": item.arg_name,
            "field_path": item.field_path,
            "override_mode": item.override_mode,
            "source": item.source,
            "old_value": audit_old_value,
            "new_value": audit_new_value
        })
        requested_overrides.append({
            "arg_name": item.arg_name,
            "field_path": item.field_path,
            "override_mode": item.override_mode,
            "source": item.source,
            "raw_key": item.raw_key,
            "raw_value": item.raw_value,
            "value": item.value
        })
        requested_kv[item.arg_name] = item.value

    override_applied = {
        "source": "cli",
        "allowed_fields_version": whitelist.whitelist_version,
        "requested_overrides": requested_overrides,
        "requested_kv": requested_kv,
        "applied_fields": applied_fields,
        "rejected_fields": []
    }
    # override_applied 必须可稳定序列化。
    digests.normalize_for_digest(override_applied)
    return override_applied


def require_override_applied(
    override_args: List[str],
    override_applied: Optional[Dict[str, Any]]
) -> None:
    """
    功能：要求 overrides 非空时必须产生 override_applied。

    Require override_applied audit block when overrides are provided.

    Args:
        override_args: CLI override arguments list.
        override_applied: override_applied audit mapping or None.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If override_applied is missing when overrides exist.
    """
    if not isinstance(override_args, list):
        # override_args 类型不合法，必须 fail-fast。
        raise TypeError("override_args must be list")
    if override_applied is None:
        if override_args:
            raise ValueError("override_applied missing for non-empty overrides")
        return
    if not isinstance(override_applied, dict):
        # override_applied 类型不合法，必须 fail-fast。
        raise TypeError("override_applied must be dict")


def _parse_override_arg(arg: str) -> ParsedOverride:
    """
    功能：解析单个 CLI override 参数。

    Parse a single CLI override argument (key=value) with JSON value.

    Args:
        arg: Override argument string.

    Returns:
        ParsedOverride instance.

    Raises:
        TypeError: If arg is invalid type.
        ValueError: If format or JSON value is invalid.
    """
    if not isinstance(arg, str) or not arg:
        # arg 输入不合法，必须 fail-fast。
        raise TypeError("override arg must be non-empty str")
    if "=" not in arg:
        raise ValueError(f"override arg must be key=value: {arg}")
    key, raw_value = arg.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ValueError(f"override key must be non-empty: {arg}")
    if raw_value == "":
        raise ValueError(f"override value must be non-empty: {arg}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"override value must be JSON: {arg}") from exc
    digests.normalize_for_digest(value)
    return ParsedOverride(key=key, raw_value=raw_value, value=value)


def _resolve_overrides(
    parsed: List[ParsedOverride],
    whitelist: RuntimeWhitelist
) -> List[ResolvedOverride]:
    """
    功能：将 override 条目绑定到 whitelist 规则。

    Resolve CLI overrides against runtime whitelist allowed_overrides.

    Args:
        parsed: Parsed overrides list.
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        List of ResolvedOverride entries.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If override is not allowed or whitelist is inconsistent.
    """
    if not isinstance(parsed, list):
        # parsed 类型不合法，必须 fail-fast。
        raise TypeError("parsed must be list")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不合法，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")

    override_section = whitelist.data.get("override", {})
    allowed_overrides = override_section.get("allowed_overrides", [])
    arg_allowed = override_section.get("arg_name_enum", {}).get("allowed", [])
    source_allowed = override_section.get("source_enum", {}).get("allowed", [])
    if not isinstance(allowed_overrides, list):
        raise TypeError("override.allowed_overrides must be list")
    if not isinstance(arg_allowed, list):
        raise TypeError("override.arg_name_enum.allowed must be list")
    if not isinstance(source_allowed, list):
        raise TypeError("override.source_enum.allowed must be list")

    index_by_arg: Dict[str, Dict[str, Any]] = {}
    index_by_field: Dict[str, Dict[str, Any]] = {}
    for entry in allowed_overrides:
        if not isinstance(entry, dict):
            raise TypeError("override.allowed_overrides entries must be dict")
        arg_name = entry.get("arg_name")
        field_path = entry.get("field_path")
        if not isinstance(arg_name, str) or not arg_name:
            raise ValueError("override.allowed_overrides arg_name must be non-empty str")
        if not isinstance(field_path, str) or not field_path:
            raise ValueError("override.allowed_overrides field_path must be non-empty str")
        if arg_name in index_by_arg:
            raise ValueError(f"override.arg_name duplicated in whitelist: {arg_name}")
        # field_path 可以被多个 arg_name 引用。
        # 只在构建索引时记录 arg_name，允许多个 arg_name 指向同一 field_path。
        index_by_arg[arg_name] = entry
        if field_path not in index_by_field:
            index_by_field[field_path] = []
        if not isinstance(index_by_field[field_path], list):
            index_by_field[field_path] = [index_by_field[field_path]]
        index_by_field[field_path].append(entry)

    resolved: List[ResolvedOverride] = []
    for item in parsed:
        if not isinstance(item, ParsedOverride):
            raise TypeError("parsed overrides must be ParsedOverride")
        # 首先尝试从 arg_name 索引查找。
        entry = index_by_arg.get(item.key)
        raw_key = item.key
        # 如果没有找到，可能输入的是 field_path；不推荐但兼容。
        if entry is None:
            field_entries = index_by_field.get(item.key)
            if isinstance(field_entries, list) and len(field_entries) == 1:
                entry = field_entries[0]
            elif isinstance(field_entries, list) and len(field_entries) > 1:
                # 多个 arg_name 指向同一字段，无法通过 field_path 唯一确定，必须 fail-fast。
                raise ValueError(f"override_field_path_ambiguous: field_path={item.key} has multiple arg_names")
        if entry is None:
            raise ValueError(f"override_not_allowed: key={item.key}")

        arg_name = entry.get("arg_name")
        field_path = entry.get("field_path")
        override_mode = entry.get("override_mode")
        source = entry.get("source")
        if arg_name not in arg_allowed:
            raise ValueError(f"override_arg_name_not_allowed: arg_name={arg_name}")
        if source not in source_allowed:
            raise ValueError(f"override_source_not_allowed: source={source}")
        if override_mode not in {"set"}:
            raise ValueError(f"override_mode_not_supported: {override_mode}")

        resolved.append(ResolvedOverride(
            arg_name=arg_name,
            field_path=field_path,
            override_mode=override_mode,
            source=source,
            value=item.value,
            raw_key=raw_key,
            raw_value=item.raw_value
        ))

    return resolved


def _require_no_duplicate_arg_names(overrides: List[ResolvedOverride]) -> None:
    """
    功能：禁止 override arg_name 重复。

    Require no duplicate arg_name entries in overrides.

    Args:
        overrides: Resolved override list.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If duplicates are found.
    """
    if not isinstance(overrides, list):
        # overrides 类型不合法，必须 fail-fast。
        raise TypeError("overrides must be list")
    seen = set()
    for item in overrides:
        if not isinstance(item, ResolvedOverride):
            raise TypeError("overrides must contain ResolvedOverride items")
        if item.arg_name in seen:
            raise ValueError(f"override_arg_name_duplicated: {item.arg_name}")
        seen.add(item.arg_name)


def _require_override_target_allowed(
    field_path: str,
    interpretation: ContractInterpretation
) -> None:
    """
    功能：禁止覆盖冻结与事实源锚点字段。

    Reject overrides targeting protected fact source or frozen fields.

    Args:
        field_path: Target field path.
        interpretation: Frozen contract interpretation.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is protected.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不合法，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    root = field_path.split(".", 1)[0]
    if root in _PROTECTED_ROOT_EXACT:
        raise ValueError(f"override_not_allowed: protected_root={root}")
    for prefix in _PROTECTED_ROOT_PREFIXES:
        if root.startswith(prefix):
            raise ValueError(f"override_not_allowed: protected_root={root}")
    for suffix in _PROTECTED_SUFFIXES:
        if root.endswith(suffix) or field_path.endswith(suffix):
            raise ValueError(f"override_not_allowed: protected_field={field_path}")


def _resolve_override_value(item: ResolvedOverride, whitelist: RuntimeWhitelist) -> Any:
    """
    功能：解析 override 值并执行 whitelist 约束。

    Resolve override value using whitelist constraints like set_value.

    Args:
        item: Resolved override entry.
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        Resolved override value.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If value violates whitelist constraints.
    """
    if not isinstance(item, ResolvedOverride):
        # item 类型不合法，必须 fail-fast。
        raise TypeError("item must be ResolvedOverride")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不合法，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")

    allowed_overrides = whitelist.data.get("override", {}).get("allowed_overrides", [])
    if not isinstance(allowed_overrides, list):
        raise TypeError("override.allowed_overrides must be list")

    set_value: Any = None
    found = False
    for entry in allowed_overrides:
        if not isinstance(entry, dict):
            continue
        if entry.get("arg_name") == item.arg_name and entry.get("field_path") == item.field_path:
            found = True
            if "set_value" in entry:
                set_value = entry.get("set_value")
            elif "allowed_values" in entry:
                allowed_values = entry.get("allowed_values")
                if isinstance(allowed_values, list) and item.value not in allowed_values:
                    raise ValueError(
                        f"override_value_not_allowed: arg_name={item.arg_name}, "
                        f"value={item.value}, allowed={allowed_values}"
                    )
            break
    if not found:
        raise ValueError(f"override_not_allowed: arg_name={item.arg_name}")
    if "set_value" in (entry if found else {}):
        if item.value != set_value:
            raise ValueError(
                f"override_value_mismatch: arg_name={item.arg_name}, "
                f"expected={set_value}, actual={item.value}"
            )
        return set_value
    return item.value


def _get_value_by_field_path(mapping: Dict[str, Any], field_path: str) -> Tuple[bool, Any]:
    """
    功能：按点路径读取映射字段值。

    Read a nested mapping value by dotted path.

    Args:
        mapping: Source mapping.
        field_path: Dotted path string.

    Returns:
        Tuple of (found, value).

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    for segment in field_path.split("."):
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _set_value_by_field_path(mapping: Dict[str, Any], field_path: str, value: Any) -> None:
    """
    功能：按点路径写入映射字段值。

    Set a nested mapping value by dotted path, disallowing new paths.

    Args:
        mapping: Target mapping.
        field_path: Dotted path string.
        value: Value to set.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid or missing.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if segment not in current or not isinstance(current[segment], dict):
            raise ValueError(f"override_field_path_missing: field_path={field_path}")
        current = current[segment]
    last = segments[-1]
    if not last:
        # field_path 末段为空，必须 fail-fast。
        raise ValueError(f"Invalid field_path segment in {field_path}")
    if last not in current:
        raise ValueError(f"override_field_path_missing: field_path={field_path}")
    current[last] = value


def _collect_disallowed_override_paths(cfg: Dict[str, Any]) -> List[str]:
    """
    功能：收集配置中 override 相关字段路径。

    Collect paths of override-related keys for fail-fast validation.

    Args:
        cfg: Configuration mapping.

    Returns:
        List of field paths that violate override policy.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    violations: List[str] = []

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in _DISALLOWED_OVERRIDE_KEYS:
                    violations.append(next_path)
                _walk(value, next_path)
            return
        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk(item, f"{path}[{idx}]")

    _walk(cfg, "")
    return violations
