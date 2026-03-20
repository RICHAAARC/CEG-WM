"""
YAML 加载唯一入口

功能说明：
- 统一所有 YAML 加载操作的入口，禁止其他模块直接调用 yaml.safe_load。
- 加载时同时计算文件 SHA256 和对象的 canonical JSON SHA256,提供溯源信息。
- 提供加载并校验配置的函数，验证 policy_path 与白名单和语义表的一致性，并应用 CLI 覆盖规则。
- 实现 ablation 统一开关归一化（normalize_ablation_flags）：
  - 解析用户输入的 ablation.enable_* 开关，处理互斥约束（lf_only / hf_only）。
  - 生成 normalized 字段记录最终生效的开关状态，纳入 cfg_digest 计算。
  - 禁止手动设置 normalized 字段，由归一化函数自动生成。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 额外能力需通过版本化追加接入，且不得改变既有配置语义与校验口径。
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Union, cast
from dataclasses import dataclass

from . import digests
from .errors import YAMLLoadError


# 冻结配置权威路径常量（必须为相对路径字面量，避免解释面分叉）。
FROZEN_CONTRACTS_PATH = "configs/frozen_contracts.yaml"
RUNTIME_WHITELIST_PATH = "configs/runtime_whitelist.yaml"
POLICY_PATH_SEMANTICS_PATH = "configs/policy_path_semantics.yaml"
INJECTION_SCOPE_MANIFEST_PATH = "configs/injection_scope_manifest.yaml"
RECORDS_SCHEMA_EXTENSIONS_PATH = "configs/records_schema_extensions.yaml"
ATTACK_PROTOCOL_PATH = "configs/attack_protocol.yaml"


def normalize_ablation_flags(cfg: Dict[str, Any]) -> None:
    """
    功能：归一化 ablation 实验开关，生成 normalized 字段并写入 cfg。

    Normalize ablation flags and generate normalized field.

    Processes ablation.enable_* flags, resolves mutual exclusion constraints
    (lf_only/hf_only), and generates ablation.normalized field recording final
    effective switch states. Mutates cfg in place.

    Rules:
        1. User-provided enable_* (null/bool) are resolved to bool (null → default).
        2. lf_only / hf_only mutual exclusion enforced (both true → fail-fast).
        3. lf_only=true → enable_lf=true, enable_hf=false.
        4. hf_only=true → enable_hf=true, enable_lf=false.
        5. ablation.normalized pre-existing in cfg → fail-fast (must auto-generate).
        6. ablation.normalized is written to cfg for cfg_digest inclusion.

    Args:
        cfg: Configuration dict with optional ablation section.

    Returns:
        None (mutates cfg in place).

    Raises:
        TypeError: If cfg or ablation types are invalid.
        ValueError: If mutual exclusion violated or normalized is manually set.
    """
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    ablation_obj: Any = cfg_mapping.get("ablation")
    if ablation_obj is None:
        # ablation 段缺失，设置默认值（全部启用）。
        ablation_obj = {}
        cfg_mapping["ablation"] = ablation_obj

    if not isinstance(ablation_obj, dict):
        # ablation 类型不合法，必须 fail-fast。
        raise TypeError("ablation must be dict")

    ablation = cast(Dict[str, Any], ablation_obj)

    # 禁止用户手动设置 normalized 字段。
    if "normalized" in ablation and ablation["normalized"] is not None:
        raise ValueError("ablation.normalized must not be manually set (auto-generated)")

    # 读取用户输入的开关值（null 或 bool）。
    enable_content: Any = ablation.get("enable_content")
    enable_geometry: Any = ablation.get("enable_geometry")
    enable_fusion: Any = ablation.get("enable_fusion")
    enable_mask: Any = ablation.get("enable_mask")
    enable_subspace: Any = ablation.get("enable_subspace")
    enable_rescue: Any = ablation.get("enable_rescue")
    enable_lf: Any = ablation.get("enable_lf")
    enable_hf: Any = ablation.get("enable_hf")
    enable_sync: Any = ablation.get("enable_sync")
    enable_anchor: Any = ablation.get("enable_anchor")
    enable_image_sidecar: Any = ablation.get("enable_image_sidecar")
    lf_only: Any = ablation.get("lf_only", False)
    hf_only: Any = ablation.get("hf_only", False)

    # 类型校验：lf_only / hf_only 必须为 bool。
    if not isinstance(lf_only, bool):
        raise TypeError("ablation.lf_only must be bool")
    if not isinstance(hf_only, bool):
        raise TypeError("ablation.hf_only must be bool")

    # 互斥约束：lf_only 和 hf_only 不可同时为 true。
    if lf_only and hf_only:
        raise ValueError("ablation.lf_only and ablation.hf_only cannot both be true")

    # 默认值解析（null → 默认启用）。
    def _resolve_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            raise TypeError(f"ablation enable flag must be bool or null, got {type(value).__name__}")
        return value

    enable_content_resolved = _resolve_bool(enable_content, True)
    enable_geometry_resolved = _resolve_bool(enable_geometry, True)
    enable_fusion_resolved = _resolve_bool(enable_fusion, True)
    enable_mask_resolved = _resolve_bool(enable_mask, True)
    enable_subspace_resolved = _resolve_bool(enable_subspace, True)
    enable_rescue_resolved = _resolve_bool(enable_rescue, False)
    enable_lf_resolved = _resolve_bool(enable_lf, True)
    enable_hf_resolved = _resolve_bool(enable_hf, False)
    enable_sync_resolved = _resolve_bool(enable_sync, True)
    enable_anchor_resolved = _resolve_bool(enable_anchor, True)
    enable_image_sidecar_resolved = _resolve_bool(enable_image_sidecar, True)

    # 应用互斥约束覆写。
    if lf_only:
        enable_lf_resolved = True
        enable_hf_resolved = False
    if hf_only:
        enable_lf_resolved = False
        enable_hf_resolved = True

    # 生成 normalized 字段（记录最终生效的开关状态）。
    normalized = {
        "enable_content": enable_content_resolved,
        "enable_geometry": enable_geometry_resolved,
        "enable_fusion": enable_fusion_resolved,
        "enable_mask": enable_mask_resolved,
        "enable_subspace": enable_subspace_resolved,
        "enable_rescue": enable_rescue_resolved,
        "enable_lf": enable_lf_resolved,
        "enable_hf": enable_hf_resolved,
        "enable_sync": enable_sync_resolved,
        "enable_anchor": enable_anchor_resolved,
        "enable_image_sidecar": enable_image_sidecar_resolved,
        "lf_only": lf_only,
        "hf_only": hf_only,
    }

    # 写入 cfg，纳入 cfg_digest 计算。
    ablation["enable_content"] = enable_content_resolved
    ablation["enable_geometry"] = enable_geometry_resolved
    ablation["enable_fusion"] = enable_fusion_resolved
    ablation["enable_mask"] = enable_mask_resolved
    ablation["enable_subspace"] = enable_subspace_resolved
    ablation["enable_rescue"] = enable_rescue_resolved
    ablation["enable_lf"] = enable_lf_resolved
    ablation["enable_hf"] = enable_hf_resolved
    ablation["enable_sync"] = enable_sync_resolved
    ablation["enable_anchor"] = enable_anchor_resolved
    ablation["enable_image_sidecar"] = enable_image_sidecar_resolved
    ablation["normalized"] = normalized


def _validate_paper_lf_ecc_gate(cfg: Dict[str, Any]) -> None:
    """
    功能：在配置加载阶段执行 paper 模式 LF ECC 门禁。 

    Enforce LF ECC gate at config loading stage for paper faithfulness mode.

    Args:
        cfg: Configuration mapping.

    Returns:
        None.

    Raises:
        TypeError: If cfg structure is invalid.
        ValueError: If paper mode is enabled while LF ECC uses legacy int branch.
    """
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    paper_cfg_obj: Any = cfg_mapping.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_obj) if isinstance(paper_cfg_obj, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    if not paper_enabled:
        return

    watermark_cfg_obj: Any = cfg_mapping.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_cfg_obj) if isinstance(watermark_cfg_obj, dict) else {}
    lf_cfg_obj: Any = watermark_cfg.get("lf")
    lf_cfg = cast(Dict[str, Any], lf_cfg_obj) if isinstance(lf_cfg_obj, dict) else {}
    ecc_value: Any = lf_cfg.get("ecc", "sparse_ldpc")

    if isinstance(ecc_value, int):
        # paper 模式禁止 legacy int ecc，必须使用 sparse_ldpc 单一路径。
        raise ValueError(
            "paper_faithfulness requires watermark.lf.ecc='sparse_ldpc'; "
            f"legacy int ecc is not allowed (got {ecc_value})"
        )


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
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    include_paths_spec_obj: Any = include_paths_spec
    if not isinstance(include_paths_spec_obj, list):
        # include_paths_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("include_paths_spec must be list")
    include_override_applied_obj: Any = include_override_applied
    if not isinstance(include_override_applied_obj, bool):
        # include_override_applied 类型不符合预期，必须 fail-fast。
        raise TypeError("include_override_applied must be bool")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    include_paths = cast(List[str], include_paths_spec_obj)
    include_override = include_override_applied_obj

    for path in include_paths:
        path_obj: Any = path
        if not isinstance(path_obj, str) or not path_obj:
            # include_paths_spec 成员不合法，必须 fail-fast。
            raise ValueError("include_paths_spec entries must be non-empty str")
        normalized_path = path_obj
        if not include_override and normalized_path.startswith("override_applied"):
            raise ValueError("override_applied must be excluded from cfg_digest")

    effective_paths = list(include_paths)
    if include_override and "override_applied" not in effective_paths:
        effective_paths.append("override_applied")

    pruned_cfg = _prune_cfg_by_include_paths(cfg_mapping, effective_paths)
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
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    include_paths_spec_obj: Any = include_paths_spec
    if not isinstance(include_paths_spec_obj, list):
        # include_paths_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("include_paths_spec must be list")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    include_paths = cast(List[str], include_paths_spec_obj)

    pruned: Dict[str, Any] = {}
    for field_path in include_paths:
        field_path_obj: Any = field_path
        if not isinstance(field_path_obj, str) or not field_path_obj:
            raise ValueError("include_paths_spec entries must be non-empty str")
        normalized_field_path = field_path_obj
        found, value = _get_value_by_field_path(cfg_mapping, normalized_field_path)
        if not found:
            value = None
        _set_value_by_field_path(pruned, normalized_field_path, value)
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
    mapping_obj: Any = mapping
    if not isinstance(mapping_obj, dict):
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError("mapping must be dict")
    field_path_obj: Any = field_path
    if not isinstance(field_path_obj, str) or not field_path_obj:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current = cast(Dict[str, Any], mapping_obj)
    normalized_field_path = field_path_obj
    segments = normalized_field_path.split(".")

    for index, segment in enumerate(segments):
        if not segment:
            raise ValueError(f"Invalid field_path segment in {normalized_field_path}")
        if segment not in current:
            return False, None
        current_value: Any = current[segment]
        if index < len(segments) - 1 and not isinstance(current_value, dict):
            return False, None
        if isinstance(current_value, dict):
            current = cast(Dict[str, Any], current_value)
            continue
        return True, current_value
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
    mapping_obj: Any = mapping
    if not isinstance(mapping_obj, dict):
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError("mapping must be dict")
    field_path_obj: Any = field_path
    if not isinstance(field_path_obj, str) or not field_path_obj:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current = cast(Dict[str, Any], mapping_obj)
    normalized_field_path = field_path_obj
    segments = normalized_field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            raise ValueError(f"Invalid field_path segment in {normalized_field_path}")
        next_value: Any = current.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            current[segment] = next_value
        current = cast(Dict[str, Any], next_value)
    last = segments[-1]
    if not last:
        raise ValueError(f"Invalid field_path segment in {normalized_field_path}")
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
    apply CLI overrides (if provided), normalize ablation flags, then compute cfg_digest
    using include_paths.

    Workflow:
        1. Load YAML config from file.
        2. Validate overrides embedded in config (YAML-level overrides forbidden).
        3. Apply CLI overrides (if provided).
        4. Normalize ablation flags (generate ablation.normalized field).
        5. Validate policy_path against whitelist and semantics.
        6. Compute cfg_digest on effective config.

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
    config_path_obj: Any = config_path
    if not isinstance(config_path_obj, (str, Path)):
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
    overrides_obj: Any = overrides
    if overrides_obj is not None and not isinstance(overrides_obj, list):
        # overrides 类型不符合预期，必须 fail-fast。
        raise TypeError("overrides must be list or None")

    normalized_config_path = config_path_obj
    cfg, _ = load_yaml_with_provenance(normalized_config_path)
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("config root must be dict")
    cfg = cast(Dict[str, Any], cfg_obj)

    from main.policy import override_rules
    override_rules.validate_overrides(cfg, interpretation)

    override_args = cast(list[str], overrides_obj) if overrides_obj is not None else []
    override_applied: Dict[str, Any] | None = None
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

    # Ablation 归一化（在 override 应用之后、cfg_digest 计算之前）。
    # 必须在 policy_path 验证之前完成，以便 ablation.normalized 纳入 cfg_digest。
    normalize_ablation_flags(cfg)

    # Paper 模式 LF ECC 门禁前移到配置阶段，避免运行时才发现分支不兼容。
    _validate_paper_lf_ecc_gate(cfg)

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
    
    overrides_applied_summary: Dict[str, Any] | None = None
    if override_applied is not None:
        overrides_applied_summary = {
            "count": len(override_applied),
            "keys": sorted(override_applied.keys())
        }

    cfg_audit_record: Dict[str, Any] = {
        "config_path": str(config_path),
        "overrides_applied_summary": overrides_applied_summary,
        "cfg_pruned_for_digest_canon_sha256": cfg_pruned_for_digest_canon_sha256,
        "cfg_digest": cfg_digest
    }
    cfg_audit_canon_sha256 = digests.canonical_sha256(cfg_audit_record)

    cfg_audit_metadata: Dict[str, Any] = {
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
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)

    allow_nonempty_run_root: Any = cfg_mapping.get("allow_nonempty_run_root", False)
    if allow_nonempty_run_root is None:
        allow_nonempty_run_root = False
    if not isinstance(allow_nonempty_run_root, bool):
        # allow_nonempty_run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("allow_nonempty_run_root must be bool")

    allow_nonempty_run_root_reason: Any = cfg_mapping.get("allow_nonempty_run_root_reason")
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
    override_applied_obj: Any = override_applied
    if not isinstance(override_applied_obj, dict):
        # override_applied 类型不符合预期，必须 fail-fast。
        raise TypeError("override_applied must be dict")
    field_path_obj: Any = field_path
    if not isinstance(field_path_obj, str) or not field_path_obj:
        # field_path 类型不符合预期，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    override_applied_mapping = cast(Dict[str, Any], override_applied_obj)
    normalized_field_path = field_path_obj

    applied_fields: Any = override_applied_mapping.get("applied_fields")
    if not isinstance(applied_fields, list):
        return False
    applied_field_items = cast(List[Any], applied_fields)
    for item in applied_field_items:
        item_obj: Any = item
        if not isinstance(item_obj, dict):
            continue
        item_mapping = cast(Dict[str, Any], item_obj)
        if item_mapping.get("field_path") == normalized_field_path:
            return True
    return False
