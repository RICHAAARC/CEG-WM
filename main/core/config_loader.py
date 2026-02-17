"""
YAML 加载唯一入口

功能说明：
- 统一所有 YAML 加载操作的入口，禁止其他模块直接调用 yaml.safe_load。
- 加载时同时计算文件 SHA256 和对象的 canonical JSON SHA256，提供溯源信息。
- 提供加载并校验配置的函数，验证 policy_path 与白名单和语义表的一致性，并应用 CLI 覆盖规则。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 未来可以扩展为支持更多格式或提供更丰富的校验功能。
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Union
from dataclasses import dataclass

from . import digests
from .errors import YAMLLoadError


# 冻结配置权威路径常量（必须为相对路径字面量，避免解释面分叉）。
FROZEN_CONTRACTS_PATH = "configs/frozen_contracts.yaml"
RUNTIME_WHITELIST_PATH = "configs/runtime_whitelist.yaml"
POLICY_PATH_SEMANTICS_PATH = "configs/policy_path_semantics.yaml"
INJECTION_SCOPE_MANIFEST_PATH = "configs/injection_scope_manifest.yaml"
RECORDS_SCHEMA_EXTENSIONS_PATH = "configs/records_schema_extensions.yaml"


def load_frozen_contracts_interpretation(
    *,
    allow_non_authoritative: bool = False
):
    """
    功能：加载冻结合约配置并构造唯一解释对象。
    
    Load frozen_contracts.yaml and create ContractInterpretation (唯一入口).
    Must be called before any run_closure or policy enforcement operations.
    
    Returns:
        Tuple of (FrozenContracts instance, ContractInterpretation instance).
    
    Raises:
        YAMLLoadError: If contracts cannot be loaded or interpreted.
    """
    from main.core.contracts import load_frozen_contracts, get_contract_interpretation

    contracts = load_frozen_contracts(
        FROZEN_CONTRACTS_PATH,
        allow_non_authoritative=allow_non_authoritative
    )
    interpretation = get_contract_interpretation(contracts)

    return contracts, interpretation


def load_runtime_whitelist(
    *,
    allow_non_authoritative: bool = False
):
    """
    功能：加载运行时白名单配置。
    
    Load runtime_whitelist.yaml.
    
    Returns:
        RuntimeWhitelist instance.
    
    Raises:
        YAMLLoadError: If whitelist cannot be loaded.
    """
    from main.policy.runtime_whitelist import load_runtime_whitelist as _load_runtime_whitelist

    return _load_runtime_whitelist(
        RUNTIME_WHITELIST_PATH,
        allow_non_authoritative=allow_non_authoritative
    )


def load_policy_path_semantics(
    *,
    allow_non_authoritative: bool = False
):
    """
    功能：加载策略路径语义表配置。
    
    Load policy_path_semantics.yaml.
    
    Returns:
        PolicyPathSemantics instance.
    
    Raises:
        YAMLLoadError: If semantics cannot be loaded.
    """
    from main.policy.runtime_whitelist import load_policy_path_semantics as _load_policy_path_semantics

    return _load_policy_path_semantics(
        POLICY_PATH_SEMANTICS_PATH,
        allow_non_authoritative=allow_non_authoritative
    )


def load_injection_scope_manifest(
    *,
    allow_non_authoritative: bool = False
):
    """
    功能：加载注入范围事实源配置。

    Load injection_scope_manifest.yaml.

    Returns:
        InjectionScopeManifest instance.

    Raises:
        YAMLLoadError: If manifest cannot be loaded.
    """
    from main.core.injection_scope import load_injection_scope_manifest as _load_injection_scope_manifest

    return _load_injection_scope_manifest(
        INJECTION_SCOPE_MANIFEST_PATH,
        allow_non_authoritative=allow_non_authoritative
    )


def load_records_schema_extensions(
    *,
    allow_non_authoritative: bool = False,
    allow_missing: bool = False
):
    """
    功能：加载记录 schema 扩展配置。

    Load records_schema_extensions.yaml.
    Supports backward compatibility: if file is missing and allow_missing=True,
    returns empty extensions structure.

    Args:
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.
        allow_missing: Whether to allow missing file (for backward compatibility).

    Returns:
        RecordsSchemaExtensions or EmptyRecordsSchemaExtensions instance.

    Raises:
        YAMLLoadError: If extensions cannot be loaded (unless allow_missing=True).
    """
    from main.core.schema_extensions import load_records_schema_extensions as _load_records_schema_extensions

    return _load_records_schema_extensions(
        RECORDS_SCHEMA_EXTENSIONS_PATH,
        allow_non_authoritative=allow_non_authoritative,
        allow_missing=allow_missing
    )


@dataclass
class YAMLProvenance:
    """
    功能：YAML 文件加载溯源信息。
    
    Provenance information for loaded YAML file.
    
    Attributes:
        path: Normalized relative or absolute path.
        file_sha256: SHA256 of raw file bytes.
        canon_sha256: SHA256 of canonical JSON of parsed object.
    """
    path: str
    file_sha256: str
    canon_sha256: str


def load_yaml_with_provenance(
    path: Union[str, Path]
) -> tuple[Dict[str, Any], YAMLProvenance]:
    """
    功能：加载 YAML 并生成溯源信息。
    
    Load YAML file and compute provenance (file_sha256, canon_sha256).
    
    Args:
        path: Path to YAML file.
    
    Returns:
        Tuple of (parsed_object, provenance).
    
    Raises:
        YAMLLoadError: If file cannot be read or parsed.
    """
    path = Path(path)
    
    # 读取并解析 YAML。
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
    except Exception as e:
        raise YAMLLoadError(
            f"Failed to load YAML from {path}: {e}"
        ) from e
    
    # 计算 file_sha256。
    try:
        file_sha256_value = digests.file_sha256(path)
    except Exception as e:
        raise YAMLLoadError(
            f"Failed to compute file_sha256 for {path}: {e}"
        ) from e
    
    # 计算 canon_sha256。
    try:
        canon_sha256_value = digests.canonical_sha256(obj)
    except Exception as e:
        raise YAMLLoadError(
            f"Failed to compute canon_sha256 for {path}: {e}"
        ) from e
    
    # 构造 provenance。
    provenance = YAMLProvenance(
        path=str(path),
        file_sha256=file_sha256_value,
        canon_sha256=canon_sha256_value
    )
    
    return obj, provenance


def compute_cfg_digest(
    cfg: Dict[str, Any],
    include_paths_spec: List[str],
    include_override_applied: bool
) -> str:
    """
    功能：按 include_paths 计算 cfg_digest。

    Compute cfg_digest on a pruned config derived from include_paths.

    Args:
        cfg: Full config mapping.
        include_paths_spec: List of dotted field paths to include.
        include_override_applied: Whether override_applied is included.

    Returns:
        Canonical SHA256 digest string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If include_paths_spec is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(include_paths_spec, list):
        # include_paths_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("include_paths_spec must be list")
    if not isinstance(include_override_applied, bool):
        # include_override_applied 类型不符合预期，必须 fail-fast。
        raise TypeError("include_override_applied must be bool")

    for path in include_paths_spec:
        if not isinstance(path, str) or not path:
            # include_paths_spec 成员不合法，必须 fail-fast。
            raise ValueError("include_paths_spec entries must be non-empty str")
        if not include_override_applied and path.startswith("override_applied"):
            raise ValueError("override_applied must be excluded from cfg_digest")

    effective_paths = list(include_paths_spec)
    if include_override_applied and "override_applied" not in effective_paths:
        effective_paths.append("override_applied")

    pruned_cfg = _prune_cfg_by_include_paths(cfg, effective_paths)
    return digests.canonical_sha256(pruned_cfg)


def _prune_cfg_by_include_paths(cfg: Dict[str, Any], include_paths_spec: List[str]) -> Dict[str, Any]:
    """
    功能：按 include_paths 裁剪 cfg。

    Prune a config mapping by include_paths.

    Args:
        cfg: Full config mapping.
        include_paths_spec: List of dotted field paths to include.

    Returns:
        Pruned config mapping.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If include_paths_spec is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(include_paths_spec, list):
        # include_paths_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("include_paths_spec must be list")

    pruned: Dict[str, Any] = {}
    for field_path in include_paths_spec:
        if not isinstance(field_path, str) or not field_path:
            raise ValueError("include_paths_spec entries must be non-empty str")
        found, value = _get_value_by_field_path(cfg, field_path)
        if not found:
            value = None
        _set_value_by_field_path(pruned, field_path, value)
    return pruned


def _get_value_by_field_path(mapping: Dict[str, Any], field_path: str) -> tuple[bool, Any]:
    """
    功能：按点路径读取配置字段值。

    Read a config field value by dotted field path.

    Args:
        mapping: Config mapping to read from.
        field_path: Dotted field path string.

    Returns:
        Tuple of (found: bool, value: Any).

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    for segment in field_path.split("."):
        if not segment:
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _set_value_by_field_path(mapping: Dict[str, Any], field_path: str, value: Any) -> None:
    """
    功能：按点路径写入配置字段值。

    Set a config field value by dotted field path.

    Args:
        mapping: Config mapping to write to.
        field_path: Dotted field path string.
        value: Value to set.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            raise ValueError(f"Invalid field_path segment in {field_path}")
        next_value = current.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            current[segment] = next_value
        current = next_value
    last = segments[-1]
    if not last:
        raise ValueError(f"Invalid field_path segment in {field_path}")
    current[last] = value


def load_and_validate_config(
    config_path: Union[str, Path],
    whitelist: Any,
    semantics: Any,
    contracts: Any,
    interpretation: Any,
    overrides: list[str] | None = None
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    功能：加载并校验配置，同时返回 cfg_digest 和 cfg_audit_metadata。

    Load YAML config via load_yaml_with_provenance, validate policy_path and overrides,
    apply CLI overrides (if provided), then compute cfg_digest using include_paths.
    D5: Also generates cfg_audit metadata for artifacts output.

    Args:
        config_path: Path to config YAML.
        whitelist: RuntimeWhitelist instance.
        semantics: PolicyPathSemantics instance.
        contracts: FrozenContracts instance.
        interpretation: ContractInterpretation instance.
        overrides: Optional list of CLI override args (key=value).

    Returns:
        Tuple of (cfg_dict, cfg_digest, cfg_audit_metadata).
        cfg_audit_metadata contains:
          - config_path: str
          - overrides_applied_summary: dict or None
          - cfg_pruned_for_digest_canon_sha256: str
          - cfg_digest: str
          - cfg_audit_canon_sha256: str

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If policy_path or overrides are invalid.
        YAMLLoadError: If YAML cannot be loaded.
    """
    if not isinstance(config_path, (str, Path)):
        # config_path 类型不符合预期，必须 fail-fast。
        raise TypeError("config_path must be str or Path")
    if whitelist is None or not hasattr(whitelist, "data"):
        # whitelist 输入不合法，必须 fail-fast。
        raise TypeError("whitelist must provide data field")
    if semantics is None or not hasattr(semantics, "data"):
        # semantics 输入不合法，必须 fail-fast。
        raise TypeError("semantics must provide data field")
    if contracts is None:
        # contracts 输入不合法，必须 fail-fast。
        raise TypeError("contracts must not be None")

    from main.core.contracts import ContractInterpretation
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")
    if overrides is not None and not isinstance(overrides, list):
        # overrides 类型不符合预期，必须 fail-fast。
        raise TypeError("overrides must be list or None")

    cfg, _ = load_yaml_with_provenance(config_path)
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("config root must be dict")

    from main.policy import override_rules
    override_rules.validate_overrides(cfg, interpretation)

    override_args = overrides or []
    override_applied = None
    if override_args:
        override_applied = override_rules.apply_cli_overrides(
            cfg,
            override_args,
            whitelist,
            interpretation
        )
        override_rules.require_override_applied(override_args, override_applied)
        cfg["override_applied"] = override_applied

    _require_run_root_reuse_override(cfg, override_applied)

    policy_path = cfg.get("policy_path")
    if not isinstance(policy_path, str) or not policy_path:
        # policy_path 缺失或非法，必须 fail-fast。
        raise ValueError("policy_path must be non-empty str")

    whitelist_allowed = whitelist.data.get("policy_path", {}).get("allowed", [])
    if policy_path not in whitelist_allowed:
        # policy_path 不在白名单，必须 fail-fast。
        raise ValueError(f"policy_path not allowed: {policy_path}")

    semantics_paths = semantics.data.get("policy_paths", {})
    if policy_path not in semantics_paths:
        # policy_path 在语义表中缺失，必须 fail-fast。
        raise ValueError(f"policy_path not defined in semantics: {policy_path}")

    include_paths_spec = interpretation.config_loader.cfg_digest_include_paths
    include_override_applied = interpretation.config_loader.cfg_digest_override_applied_included
    cfg_digest = compute_cfg_digest(cfg, include_paths_spec, include_override_applied)
    
    # 生成 cfg_audit_metadata。
    cfg_pruned_for_digest = _prune_cfg_by_include_paths(cfg, include_paths_spec)
    cfg_pruned_for_digest_canon_sha256 = digests.canonical_sha256(cfg_pruned_for_digest)
    
    overrides_applied_summary = None
    if override_applied is not None and isinstance(override_applied, dict):
        overrides_applied_summary = {
            "count": len(override_applied),
            "keys": sorted(override_applied.keys())
        }
    
    cfg_audit_record = {
        "config_path": str(config_path),
        "overrides_applied_summary": overrides_applied_summary,
        "cfg_pruned_for_digest_canon_sha256": cfg_pruned_for_digest_canon_sha256,
        "cfg_digest": cfg_digest
    }
    cfg_audit_canon_sha256 = digests.canonical_sha256(cfg_audit_record)
    
    cfg_audit_metadata = {
        **cfg_audit_record,
        "cfg_audit_canon_sha256": cfg_audit_canon_sha256
    }
    
    return cfg, cfg_digest, cfg_audit_metadata


def _require_run_root_reuse_override(
    cfg: Dict[str, Any],
    override_applied: Dict[str, Any] | None
) -> None:
    """
    功能：强制 run_root 复用必须来自 CLI override。

    Require run_root reuse to be explicitly enabled via CLI overrides.

    Args:
        cfg: Config mapping.
        override_applied: override_applied audit mapping or None.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If reuse is enabled without required overrides.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    allow_nonempty_run_root = cfg.get("allow_nonempty_run_root", False)
    if allow_nonempty_run_root is None:
        allow_nonempty_run_root = False
    if not isinstance(allow_nonempty_run_root, bool):
        # allow_nonempty_run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("allow_nonempty_run_root must be bool")

    allow_nonempty_run_root_reason = cfg.get("allow_nonempty_run_root_reason")
    if allow_nonempty_run_root_reason is not None and not isinstance(allow_nonempty_run_root_reason, str):
        # allow_nonempty_run_root_reason 类型不符合预期，必须 fail-fast。
        raise TypeError("allow_nonempty_run_root_reason must be str or None")

    if allow_nonempty_run_root:
        if override_applied is None:
            raise ValueError("allow_nonempty_run_root requires override_applied")
        if not isinstance(allow_nonempty_run_root_reason, str) or not allow_nonempty_run_root_reason:
            raise ValueError("allow_nonempty_run_root_reason must be non-empty str")
        if not _override_applied_includes_field(override_applied, "allow_nonempty_run_root"):
            raise ValueError("override_applied missing allow_nonempty_run_root")
        if not _override_applied_includes_field(override_applied, "allow_nonempty_run_root_reason"):
            raise ValueError("override_applied missing allow_nonempty_run_root_reason")
    else:
        if allow_nonempty_run_root_reason is not None:
            raise ValueError("allow_nonempty_run_root_reason must be None when reuse disabled")


def _override_applied_includes_field(override_applied: Dict[str, Any], field_path: str) -> bool:
    """
    功能：检查 override_applied 是否覆盖指定字段路径。

    Check whether override_applied contains a given field_path.

    Args:
        override_applied: override_applied audit mapping.
        field_path: Target field path string.

    Returns:
        True if field_path is present in applied_fields; otherwise False.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(override_applied, dict):
        # override_applied 类型不符合预期，必须 fail-fast。
        raise TypeError("override_applied must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 类型不符合预期，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    applied_fields = override_applied.get("applied_fields")
    if not isinstance(applied_fields, list):
        return False
    for item in applied_fields:
        if not isinstance(item, dict):
            continue
        if item.get("field_path") == field_path:
            return True
    return False
