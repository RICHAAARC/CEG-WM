"""
冻结契约加载与绑定

功能说明：
- 加载 frozen_contracts.yaml 文件，解析并计算相关 digest，提供结构化解释面 ContractInterpretation。
- 提供将契约字段绑定到记录的函数，确保记录中包含契约版本和 digest 信息。
- 解释面 ContractInterpretation 包含合同中定义的权威性声明、外部事实源规范、门禁执行要求等内容，供其他模块使用。
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from . import config_loader
from . import digests
from .errors import MissingRequiredFieldError, FrozenContractPathNotAuthoritativeError


@dataclass
class FrozenContracts:
    """
    功能：冻结契约加载结果。
    
    Parsed frozen contracts with computed digests.
    
    Attributes:
        data: Full YAML object.
        contract_version: Version from YAML.
        contract_digest: Semantic digest of data.
        contract_file_sha256: SHA256 of raw file bytes.
        contract_canon_sha256: SHA256 of canonical JSON.
        contract_bound_digest: Bound digest of all components.
        contract_source_path: Canonical source path for audit.
    """
    data: Dict[str, Any]
    contract_version: str
    contract_digest: str
    contract_file_sha256: str
    contract_canon_sha256: str
    contract_bound_digest: str
    contract_source_path: str


@dataclass(frozen=True)
class AuthoritySpec:
    """
    功能：冻结契约权威性声明。
    
    Authority declaration fields in frozen contracts.
    
    Attributes:
        location: Canonical contract location.
        uniqueness: Uniqueness statement.
        modification_policy: Modification policy statement.
    """
    location: str
    uniqueness: str
    modification_policy: str


@dataclass(frozen=True)
class ExternalFactSourceSpec:
    """
    功能：外部事实源声明。
    
    External fact source specification.
    
    Attributes:
        path: Source path.
        required_record_fields: Record field names required for binding.
    """
    path: str
    required_record_fields: List[str]


@dataclass(frozen=True)
class DigestCanonicalizationParameters:
    """
    
    功能：digest 规范化参数。
    
    Digest canonicalization parameters.
    
    Attributes:
        separators: JSON separators.
        sort_keys: Whether to sort keys.
        ensure_ascii: Whether to ensure ASCII output.
    """
    separators: List[str]
    sort_keys: bool
    ensure_ascii: bool


@dataclass(frozen=True)
class DigestCanonicalizationSpec:
    """
    功能：digest 规范化段落解释。
    
    Digest canonicalization specification.
    
    Attributes:
        canonical_module: Canonical module path.
        frozen_parameters: Frozen parameters for canonicalization.
        delegation_rule: Delegation rule statement.
    """
    canonical_module: str
    frozen_parameters: DigestCanonicalizationParameters
    delegation_rule: str


@dataclass(frozen=True)
class GateEnforcementRequirement:
    """
    功能：门禁执行要求条目。
    
    Gate enforcement requirement entry.
    
    Attributes:
        enforcement: Enforcement level.
        requirement: Requirement description.
        authority_source: Optional authority source path.
        planned_state: Optional planned enforcement state for recommended items.
        target_version: Optional target version for planned upgrade.
        upgrade_condition: Optional planned upgrade condition for recommended items.
        owner_gate: Optional owner gate identifier for recommended items.
    """
    enforcement: str
    requirement: str
    authority_source: Optional[str]
    planned_state: Optional[str]
    target_version: Optional[str]
    upgrade_condition: Optional[str]
    owner_gate: Optional[str]


@dataclass(frozen=True)
class FactsExtraKeysPolicySpec:
    """
    功能：facts extra keys 策略规范。

    Facts extra keys policy specification.

    Attributes:
        policy: Policy string (fail or warn).
        allowed_top_level_keys: Allowed top-level keys per fact source.
        allowed_policy_path_keys: Allowed keys for policy_paths entries.
    """
    policy: str
    allowed_top_level_keys: Dict[str, List[str]]
    allowed_policy_path_keys: List[str]


@dataclass(frozen=True)
class PolicyPathMembershipPolicySpec:
    """
    功能：policy_path 值域闭包策略规范。

    Policy path membership policy specification.

    Attributes:
        policy: Policy string (fail or warn).
        record_field_path: Record field path to check.
        whitelist_allowed_path: Whitelist allowed list path.
    """
    policy: str
    record_field_path: str
    whitelist_allowed_path: str


@dataclass(frozen=True)
class OverrideEnumPolicySpec:
    """
    功能：override 枚举闭包策略规范。

    Override enum policy specification.

    Attributes:
        policy: Policy string (fail or warn).
        override_entries_path: Path to override entries list.
        override_arg_name_allowed_path: Path to arg_name allowed list.
        override_source_allowed_path: Path to source allowed list.
        override_entry_fields: Mapping for entry field names.
    """
    policy: str
    override_entries_path: str
    override_arg_name_allowed_path: str
    override_source_allowed_path: str
    override_entry_fields: Dict[str, str]


@dataclass(frozen=True)
class DecoderTypePolicySpec:
    """
    功能：decoder_type 值域闭包策略规范。

    Decoder type policy specification.

    Attributes:
        policy: Policy string (fail or warn).
        semantics_policy_paths_path: Path to policy_paths mapping.
        semantics_decoder_type_field: Field name for decoder_type.
        whitelist_decoder_type_allowed_path: Path to allowed decoder types.
    """
    policy: str
    semantics_policy_paths_path: str
    semantics_decoder_type_field: str
    whitelist_decoder_type_allowed_path: str


@dataclass(frozen=True)
class GateEnforcementRequirements:
    """
    功能：门禁闭包策略集合。

    Gate enforcement policy collection.

    Attributes:
        facts_extra_keys_policy: Facts extra keys policy spec.
        policy_path_membership_policy: Policy path membership policy spec.
        override_enum_policy: Override enum policy spec.
        decoder_type_policy: Decoder type policy spec.
    """
    facts_extra_keys_policy: FactsExtraKeysPolicySpec
    policy_path_membership_policy: PolicyPathMembershipPolicySpec
    override_enum_policy: OverrideEnumPolicySpec
    decoder_type_policy: DecoderTypePolicySpec


@dataclass(frozen=True)
class RecordAccessorsDelegationScope:
    """
    功能：records accessor 委托范围。
    
    Delegation scope for record accessors.
    
    Attributes:
        scripts_mandatory: Whether scripts must use accessors.
        main_chain_gradual: Whether main chain enforcement is gradual.
    """
    scripts_mandatory: bool
    main_chain_gradual: bool


@dataclass(frozen=True)
class RecordAccessorsSpec:
    """
    功能：records accessor 规范段落。
    
    Records accessor specification.
    
    Attributes:
        canonical_module: Canonical module path.
        minimum_required_accessors: Minimum required accessor list.
        delegation_scope: Delegation scope details.
        notes: Additional notes.
    """
    canonical_module: str
    minimum_required_accessors: List[str]
    delegation_scope: RecordAccessorsDelegationScope
    notes: str


@dataclass(frozen=True)
class RecordsSchemaSpec:
    """
    功能：records schema 解释段落。
    
    Records schema specification.
    
    Attributes:
        chain_enablement_fields: Mapping for chain enablement fields.
        decision_field_path: Decision field path.
        required_record_fields: Required record field list.
        decision_obligation_presence: Decision field presence obligation (from audit_obligations).
        decision_allow_null: Decision field allow null obligation (from audit_obligations).
        field_paths_registry_semantics: Registry semantics statement.
        field_paths_registry: Registry field paths.
    """
    chain_enablement_fields: Dict[str, str]
    decision_field_path: str
    required_record_fields: List[str]
    decision_obligation_presence: bool
    decision_allow_null: bool
    field_paths_registry_semantics: str
    field_paths_registry: List[str]


@dataclass(frozen=True)
class ConfigLoaderSpec:
    """
    功能：config_loader 解释段落。

    Config loader specification.

    Attributes:
        cfg_digest_include_paths: Include paths for cfg_digest.
        cfg_digest_override_applied_included: Whether override_applied is included.
    """
    cfg_digest_include_paths: List[str]
    cfg_digest_override_applied_included: bool


@dataclass(frozen=True)
class ImplIdentitySpec:
    """
    功能：impl 身份绑定解释段落。
    
    Impl identity binding specification.
    
    Attributes:
        required: Whether impl identity is required.
        field_paths_by_domain: Field paths by domain.
        version_field_paths_by_domain: Version field paths by domain.
        digest_field_paths_by_domain: Digest field paths by domain.
        validation_rule: Validation rule text.
    """
    required: bool
    field_paths_by_domain: Dict[str, str]
    version_field_paths_by_domain: Dict[str, str]
    digest_field_paths_by_domain: Dict[str, str]
    validation_rule: str


@dataclass(frozen=True)
class StatusEnumSpec:
    """
    功能：status 枚举解释段落。
    
    Status enum specification.
    
    Attributes:
        allowed_values: Allowed status values.
        definitions: Definition mapping.
        immutable: Whether the enum is immutable.
        notes: Notes for status semantics.
    """
    allowed_values: List[str]
    definitions: Dict[str, str]
    immutable: bool
    notes: str


@dataclass(frozen=True)
class FailReasonEnumSpec:
    """
    功能：fail_reason 枚举解释段落。
    
    Fail reason enum specification.
    
    Attributes:
        allowed_values: Allowed fail reason values.
        usage: Usage constraint statement.
        immutable: Whether the enum is immutable.
    """
    allowed_values: List[str]
    usage: str
    immutable: bool


@dataclass(frozen=True)
class MismatchReasonEnumSpec:
    """
    功能：mismatch_reason 枚举解释段落。
    
    Mismatch reason enum specification.
    
    Attributes:
        allowed_values: Allowed mismatch reason values.
        usage: Usage constraint statement.
        immutable: Whether the enum is immutable.
    """
    allowed_values: List[str]
    usage: str
    immutable: bool


@dataclass(frozen=True)
class ContractInterpretation:
    """
    功能：冻结契约解释面（不可变）。
    
    Structured and fail-fast interpretation of frozen contracts.
    
    Attributes:
        required_record_fields: Required record fields for fact binding.
        authority: Authority declaration.
        deprecated_locations: Deprecated locations list.
        external_fact_sources: External fact sources mapping.
        io_write_entrypoints: Allowed write entrypoints.
        io_read_entrypoints: Allowed read entrypoints.
        io_exceptions: I/O exceptions list.
        digest_canonicalization: Digest canonicalization specification.
        gate_enforcement_requirements: Gate enforcement requirements mapping.
        record_accessors: Record accessor specification.
        records_schema: Records schema specification.
        config_loader: Config loader specification.
        field_aliases: Field alias mapping.
        impl_identity: Impl identity specification.
        status_enum: Status enum specification.
        fail_reason_enum: Fail reason enum specification.
        mismatch_reason_enum: Mismatch reason enum specification.
    """
    required_record_fields: List[str]
    authority: AuthoritySpec
    deprecated_locations: List[str]
    external_fact_sources: Dict[str, ExternalFactSourceSpec]
    io_write_entrypoints: List[str]
    io_read_entrypoints: List[str]
    io_exceptions: List[str]
    digest_canonicalization: DigestCanonicalizationSpec
    gate_enforcement_requirements: Dict[str, GateEnforcementRequirement]
    gate_enforcement_policies: GateEnforcementRequirements
    record_accessors: RecordAccessorsSpec
    records_schema: RecordsSchemaSpec
    config_loader: ConfigLoaderSpec
    field_aliases: Dict[str, Any]
    impl_identity: ImplIdentitySpec
    status_enum: StatusEnumSpec
    fail_reason_enum: FailReasonEnumSpec
    mismatch_reason_enum: MismatchReasonEnumSpec


_INTERPRETATION_CACHE: Dict[int, ContractInterpretation] = {}


def _is_test_environment() -> bool:
    """
    功能：判断当前是否在测试环境。
    
    Detect whether the current process is running under pytest.
    Uses PYTEST_CURRENT_TEST environment variable which is set during test execution.
    
    Returns:
        True if executing in test environment, False otherwise.
    """
    # pytest 设置的环保变量，在测试执行时此值存在
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def load_frozen_contracts(
    path: str = "configs/frozen_contracts.yaml",
    *,
    allow_non_authoritative: bool = False
) -> FrozenContracts:
    """
    功能：加载冻结契约并计算所有 digest。
    
    Load frozen_contracts.yaml and compute all digests.
    
    Args:
        path: Path to frozen_contracts.yaml.
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.
    
    Returns:
        FrozenContracts object.
    
    Raises:
        YAMLLoadError: If loading or parsing fails.
        MissingRequiredFieldError: If contract_version is missing.
        FrozenContractPathNotAuthoritativeError: If non-authoritative path is used.
    """
    if not isinstance(path, str) or not path:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    if not isinstance(allow_non_authoritative, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")

    normalized_path = Path(path).as_posix()
    if normalized_path != "configs/frozen_contracts.yaml":
        # 非权威路径例外仅测试环境可用
        is_test = _is_test_environment()
        if not (allow_non_authoritative and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "frozen_contracts path is not authoritative",
                field_path="contract_source_path",
                actual_path=normalized_path
            )
    # 加载 YAML 与 provenance
    obj, provenance = config_loader.load_yaml_with_provenance(path)
    
    # 解析 contract_version
    contract_version = obj.get("contract_version")
    if not contract_version:
        raise MissingRequiredFieldError(
            f"contract_version missing in {path}"
        )
    
    # 计算 digest
    contract_digest_value = digests.semantic_digest(obj)
    contract_file_sha256_value = provenance.file_sha256
    contract_canon_sha256_value = provenance.canon_sha256
    contract_bound_digest_value = digests.bound_digest(
        version=contract_version,
        semantic_digest_value=contract_digest_value,
        file_sha256_value=contract_file_sha256_value,
        canon_sha256_value=contract_canon_sha256_value
    )
    
    return FrozenContracts(
        data=obj,
        contract_version=contract_version,
        contract_digest=contract_digest_value,
        contract_file_sha256=contract_file_sha256_value,
        contract_canon_sha256=contract_canon_sha256_value,
        contract_bound_digest=contract_bound_digest_value,
        contract_source_path=normalized_path
    )


def bind_contract_to_record(
    record: Dict[str, Any],
    contracts: FrozenContracts
) -> None:
    """
    功能：将契约字段写入 record。
    
    Bind contract version and digests to record (mutates in-place).
    Field names are frozen in frozen_contracts.yaml:external_fact_sources.frozen_contracts.required_record_fields.
    
    Args:
        record: Record dict to mutate.
        contracts: Loaded FrozenContracts.
    """
    record["contract_version"] = contracts.contract_version
    record["contract_digest"] = contracts.contract_digest
    record["contract_file_sha256"] = contracts.contract_file_sha256
    record["contract_canon_sha256"] = contracts.contract_canon_sha256
    record["contract_bound_digest"] = contracts.contract_bound_digest


def interpret_frozen_contracts(
    contracts: FrozenContracts
) -> ContractInterpretation:
    """
    功能：冻结契约解释与结构化。
    
    Interpret frozen contracts into a structured, fail-fast representation.
    
    Args:
        contracts: Loaded FrozenContracts.
    
    Returns:
        ContractInterpretation object.
    
    Raises:
        MissingRequiredFieldError: If required fields are missing.
        TypeError: If field types do not match expectations.
    """
    _require_frozen_contracts(contracts)
    root = _require_mapping(contracts.data, "<root>")
    authority = _parse_authority(root)
    deprecated_locations = _require_list_of_str(
        _get_required_field(root, "deprecated_locations", "<root>.deprecated_locations"),
        "<root>.deprecated_locations"
    )
    external_fact_sources = _parse_external_fact_sources(root)
    io_write_entrypoints = _require_list_of_str(
        _get_required_field(root, "io_write_entrypoints", "<root>.io_write_entrypoints"),
        "<root>.io_write_entrypoints"
    )
    io_read_entrypoints = _require_list_of_str(
        _get_required_field(root, "io_read_entrypoints", "<root>.io_read_entrypoints"),
        "<root>.io_read_entrypoints"
    )
    io_exceptions = _require_list_of_str(
        _get_required_field(root, "io_exceptions", "<root>.io_exceptions"),
        "<root>.io_exceptions"
    )
    digest_canonicalization = _parse_digest_canonicalization(root)
    gate_enforcement_requirements = _parse_gate_enforcement_requirements(root)
    gate_enforcement_policies = _parse_gate_enforcement_policies(root)
    record_accessors = _parse_record_accessors(root)
    records_schema = _parse_records_schema(root)
    config_loader = _parse_config_loader(root)
    _validate_records_schema_required_fields(records_schema)
    field_aliases = _require_mapping(
        _get_required_field(root, "field_aliases", "<root>.field_aliases"),
        "<root>.field_aliases"
    )
    impl_identity = _parse_impl_identity(root)
    status_enum = _parse_status_enum(root)
    fail_reason_enum = _parse_fail_reason_enum(root)
    mismatch_reason_enum = _parse_mismatch_reason_enum(root)
    required_record_fields = _collect_required_record_fields(
        external_fact_sources,
        records_schema
    )
    return ContractInterpretation(
        required_record_fields=required_record_fields,
        authority=authority,
        deprecated_locations=deprecated_locations,
        external_fact_sources=external_fact_sources,
        io_write_entrypoints=io_write_entrypoints,
        io_read_entrypoints=io_read_entrypoints,
        io_exceptions=io_exceptions,
        digest_canonicalization=digest_canonicalization,
        gate_enforcement_requirements=gate_enforcement_requirements,
        gate_enforcement_policies=gate_enforcement_policies,
        record_accessors=record_accessors,
        records_schema=records_schema,
        config_loader=config_loader,
        field_aliases=field_aliases,
        impl_identity=impl_identity,
        status_enum=status_enum,
        fail_reason_enum=fail_reason_enum,
        mismatch_reason_enum=mismatch_reason_enum
    )


def required_record_fields_for_facts(
    contracts: FrozenContracts
) -> List[str]:
    """
    功能：从 YAML 读取所有事实源的必需字段集合。
    
    Collect required_record_fields from external_fact_sources and records_schema.
    
    Args:
        contracts: Loaded FrozenContracts.
    
    Returns:
        List of required field names.
    """
    _require_frozen_contracts(contracts)
    return get_required_record_fields(contracts)


def get_contract_interpretation(
    contracts: FrozenContracts
) -> ContractInterpretation:
    """
    功能：获取冻结契约解释面。

    Get contract interpretation with optional lightweight cache.

    Args:
        contracts: Loaded FrozenContracts.

    Returns:
        ContractInterpretation object.

    Raises:
        TypeError: If contracts is not FrozenContracts.
    """
    _require_frozen_contracts(contracts)
    cache_key = id(contracts)
    cached = _INTERPRETATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    interpretation = interpret_frozen_contracts(contracts)
    _INTERPRETATION_CACHE[cache_key] = interpretation
    return interpretation


def get_required_record_fields(
    contracts: FrozenContracts
) -> List[str]:
    """
    功能：获取 records 必需字段集合。

    Get required record fields from contract interpretation.

    Args:
        contracts: Loaded FrozenContracts.

    Returns:
        List of required record field names.

    Raises:
        TypeError: If contracts is not FrozenContracts.
    """
    interpretation = get_contract_interpretation(contracts)
    return list(interpretation.required_record_fields)


def _require_frozen_contracts(contracts: FrozenContracts) -> None:
    """
    功能：校验 FrozenContracts 输入合法性。
    
    Validate FrozenContracts input for interpretation.
    
    Args:
        contracts: FrozenContracts instance.
    
    Raises:
        TypeError: If input is not FrozenContracts or data is not mapping.
    """
    if not isinstance(contracts, FrozenContracts):
        # 输入类型不符合预期，必须 fail-fast。
        raise TypeError("Type mismatch at <root>: expected FrozenContracts")
    if not isinstance(contracts.data, dict):
        # data 类型不符合预期，必须 fail-fast。
        raise TypeError("Type mismatch at <root>: expected mapping for contracts.data")


def _get_required_field(mapping: Dict[str, Any], key: str, field_path: str) -> Any:
    """
    功能：读取必需字段，缺失即失败。
    
    Get a required field from mapping, fail fast if missing.
    
    Args:
        mapping: Source mapping.
        key: Key name.
        field_path: Field path for error message.
    
    Returns:
        Value from mapping.
    
    Raises:
        MissingRequiredFieldError: If key is missing.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError(f"Type mismatch at {field_path}: expected mapping")
    if not isinstance(key, str) or not key:
        # key 输入不合法，必须 fail-fast。
        raise ValueError("key must be non-empty str")
    if key not in mapping:
        # 缺失必需字段，必须 fail-fast。
        raise MissingRequiredFieldError(f"Missing required field: {field_path}")
    return mapping[key]


def _require_mapping(value: Any, field_path: str) -> Dict[str, Any]:
    """
    功能：确保对象为 dict。
    
    Require a mapping type.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        Mapping value.
    
    Raises:
        TypeError: If value is not a mapping.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if not isinstance(value, dict):
        # 字段类型不匹配，必须 fail-fast。
        raise TypeError(
            f"Type mismatch at {field_path}: expected mapping, got {type(value).__name__}"
        )
    return value


def _require_str(value: Any, field_path: str) -> str:
    """
    功能：确保对象为 str。
    
    Require a string type.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        String value.
    
    Raises:
        TypeError: If value is not a string.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if not isinstance(value, str):
        # 字段类型不匹配，必须 fail-fast。
        raise TypeError(
            f"Type mismatch at {field_path}: expected str, got {type(value).__name__}"
        )
    return value


def _require_bool(value: Any, field_path: str) -> bool:
    """
    功能：确保对象为 bool。
    
    Require a boolean type.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        Boolean value.
    
    Raises:
        TypeError: If value is not a boolean.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if not isinstance(value, bool):
        # 字段类型不匹配，必须 fail-fast。
        raise TypeError(
            f"Type mismatch at {field_path}: expected bool, got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, field_path: str) -> List[Any]:
    """
    功能：确保对象为 list。
    
    Require a list type.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        List value.
    
    Raises:
        TypeError: If value is not a list.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if not isinstance(value, list):
        # 字段类型不匹配，必须 fail-fast。
        raise TypeError(
            f"Type mismatch at {field_path}: expected list, got {type(value).__name__}"
        )
    return value


def _require_list_of_str(value: Any, field_path: str) -> List[str]:
    """
    功能：确保对象为 List[str]。
    
    Require a list of strings.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        List of strings.
    
    Raises:
        TypeError: If list or element types do not match.
    """
    items = _require_list(value, field_path)
    validated: List[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            # 列表元素类型不匹配，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {field_path}[{index}]: expected str, got {type(item).__name__}"
            )
        validated.append(item)
    return validated


def _require_mapping_of_str_to_list_of_str(value: Any, field_path: str) -> Dict[str, List[str]]:
    """
    功能：确保对象为 Dict[str, List[str]]。

    Require a mapping from str to list of strings.

    Args:
        value: Value to validate.
        field_path: Field path for error message.

    Returns:
        Mapping from str to list of strings.

    Raises:
        TypeError: If mapping key or value types do not match.
    """
    mapping = _require_mapping(value, field_path)
    validated: Dict[str, List[str]] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            # 映射键类型不匹配，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {field_path}: expected str keys, got {type(key).__name__}"
            )
        list_value = _require_list_of_str(item, f"{field_path}.{key}")
        validated[key] = list_value
    return validated


def _require_mapping_of_str_to_str(value: Any, field_path: str) -> Dict[str, str]:
    """
    功能：确保对象为 Dict[str, str]。
    
    Require a mapping from str to str.
    
    Args:
        value: Value to validate.
        field_path: Field path for error message.
    
    Returns:
        Mapping from str to str.
    
    Raises:
        TypeError: If mapping key or value types do not match.
    """
    mapping = _require_mapping(value, field_path)
    validated: Dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            # 映射键类型不匹配，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {field_path}: expected str keys, got {type(key).__name__}"
            )
        if not isinstance(item, str):
            # 映射值类型不匹配，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {field_path}.{key}: expected str, got {type(item).__name__}"
            )
        validated[key] = item
    return validated


def _parse_authority(root: Dict[str, Any]) -> AuthoritySpec:
    """
    功能：解析 authority 段落。
    
    Parse authority section.
    
    Args:
        root: Root mapping.
    
    Returns:
        AuthoritySpec object.
    """
    authority = _require_mapping(
        _get_required_field(root, "authority", "<root>.authority"),
        "<root>.authority"
    )
    location = _require_str(
        _get_required_field(authority, "location", "<root>.authority.location"),
        "<root>.authority.location"
    )
    uniqueness = _require_str(
        _get_required_field(authority, "uniqueness", "<root>.authority.uniqueness"),
        "<root>.authority.uniqueness"
    )
    modification_policy = _require_str(
        _get_required_field(authority, "modification_policy", "<root>.authority.modification_policy"),
        "<root>.authority.modification_policy"
    )
    return AuthoritySpec(
        location=location,
        uniqueness=uniqueness,
        modification_policy=modification_policy
    )


def _parse_external_fact_sources(root: Dict[str, Any]) -> Dict[str, ExternalFactSourceSpec]:
    """
    功能：解析 external_fact_sources 段落。
    
    Parse external_fact_sources section.
    
    Args:
        root: Root mapping.
    
    Returns:
        Mapping of external fact source specs.
    """
    sources = _require_mapping(
        _get_required_field(root, "external_fact_sources", "<root>.external_fact_sources"),
        "<root>.external_fact_sources"
    )
    parsed: Dict[str, ExternalFactSourceSpec] = {}
    for source_name, source_value in sources.items():
        if not isinstance(source_name, str):
            # 外部事实源名称类型不匹配，必须 fail-fast。
            raise TypeError(
                "Type mismatch at <root>.external_fact_sources: expected str keys"
            )
        source_path = f"<root>.external_fact_sources.{source_name}"
        source_mapping = _require_mapping(source_value, source_path)
        path_value = _require_str(
            _get_required_field(source_mapping, "path", f"{source_path}.path"),
            f"{source_path}.path"
        )
        required_fields = _require_list_of_str(
            _get_required_field(
                source_mapping,
                "required_record_fields",
                f"{source_path}.required_record_fields"
            ),
            f"{source_path}.required_record_fields"
        )
        parsed[source_name] = ExternalFactSourceSpec(
            path=path_value,
            required_record_fields=required_fields
        )
    return parsed


def _parse_digest_canonicalization(root: Dict[str, Any]) -> DigestCanonicalizationSpec:
    """
    功能：解析 digest_canonicalization 段落。
    
    Parse digest_canonicalization section.
    
    Args:
        root: Root mapping.
    
    Returns:
        DigestCanonicalizationSpec object.
    """
    digest_section = _require_mapping(
        _get_required_field(root, "digest_canonicalization", "<root>.digest_canonicalization"),
        "<root>.digest_canonicalization"
    )
    canonical_module = _require_str(
        _get_required_field(digest_section, "canonical_module", "<root>.digest_canonicalization.canonical_module"),
        "<root>.digest_canonicalization.canonical_module"
    )
    frozen_parameters = _require_mapping(
        _get_required_field(digest_section, "frozen_parameters", "<root>.digest_canonicalization.frozen_parameters"),
        "<root>.digest_canonicalization.frozen_parameters"
    )
    separators = _require_list_of_str(
        _get_required_field(frozen_parameters, "separators", "<root>.digest_canonicalization.frozen_parameters.separators"),
        "<root>.digest_canonicalization.frozen_parameters.separators"
    )
    sort_keys = _require_bool(
        _get_required_field(frozen_parameters, "sort_keys", "<root>.digest_canonicalization.frozen_parameters.sort_keys"),
        "<root>.digest_canonicalization.frozen_parameters.sort_keys"
    )
    ensure_ascii = _require_bool(
        _get_required_field(frozen_parameters, "ensure_ascii", "<root>.digest_canonicalization.frozen_parameters.ensure_ascii"),
        "<root>.digest_canonicalization.frozen_parameters.ensure_ascii"
    )
    delegation_rule = _require_str(
        _get_required_field(digest_section, "delegation_rule", "<root>.digest_canonicalization.delegation_rule"),
        "<root>.digest_canonicalization.delegation_rule"
    )
    parameters = DigestCanonicalizationParameters(
        separators=separators,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii
    )
    return DigestCanonicalizationSpec(
        canonical_module=canonical_module,
        frozen_parameters=parameters,
        delegation_rule=delegation_rule
    )


def _parse_gate_enforcement_requirements(root: Dict[str, Any]) -> Dict[str, GateEnforcementRequirement]:
    """
    功能：解析 gate_enforcement_requirements 段落。
    
    Parse gate_enforcement_requirements section.
    
    Args:
        root: Root mapping.
    
    Returns:
        Mapping of gate enforcement requirements.
    """
    requirements = _require_mapping(
        _get_required_field(root, "gate_enforcement_requirements", "<root>.gate_enforcement_requirements"),
        "<root>.gate_enforcement_requirements"
    )
    parsed: Dict[str, GateEnforcementRequirement] = {}
    for requirement_name, requirement_value in requirements.items():
        if not isinstance(requirement_name, str):
            # 门禁要求名称类型不匹配，必须 fail-fast。
            raise TypeError(
                "Type mismatch at <root>.gate_enforcement_requirements: expected str keys"
            )
        requirement_path = f"<root>.gate_enforcement_requirements.{requirement_name}"
        requirement_mapping = _require_mapping(requirement_value, requirement_path)
        enforcement = _require_str(
            _get_required_field(requirement_mapping, "enforcement", f"{requirement_path}.enforcement"),
            f"{requirement_path}.enforcement"
        )
        requirement_text = _require_str(
            _get_required_field(requirement_mapping, "requirement", f"{requirement_path}.requirement"),
            f"{requirement_path}.requirement"
        )
        authority_source_value = requirement_mapping.get("authority_source")
        authority_source = None
        if authority_source_value is not None:
            authority_source = _require_str(
                authority_source_value,
                f"{requirement_path}.authority_source"
            )
        planned_state = None
        target_version = None
        upgrade_condition = None
        owner_gate = None
        if enforcement == "recommended_enforce":
            planned_state = _require_str(
                _get_required_field(requirement_mapping, "planned_state", f"{requirement_path}.planned_state"),
                f"{requirement_path}.planned_state"
            )
            target_version = _require_str(
                _get_required_field(requirement_mapping, "target_version", f"{requirement_path}.target_version"),
                f"{requirement_path}.target_version"
            )
            upgrade_condition = _require_str(
                _get_required_field(requirement_mapping, "upgrade_condition", f"{requirement_path}.upgrade_condition"),
                f"{requirement_path}.upgrade_condition"
            )
            owner_gate = _require_str(
                _get_required_field(requirement_mapping, "owner_gate", f"{requirement_path}.owner_gate"),
                f"{requirement_path}.owner_gate"
            )
        else:
            planned_state_value = requirement_mapping.get("planned_state")
            if planned_state_value is not None:
                planned_state = _require_str(
                    planned_state_value,
                    f"{requirement_path}.planned_state"
                )
            target_version_value = requirement_mapping.get("target_version")
            if target_version_value is not None:
                target_version = _require_str(
                    target_version_value,
                    f"{requirement_path}.target_version"
                )
            upgrade_condition_value = requirement_mapping.get("upgrade_condition")
            if upgrade_condition_value is not None:
                upgrade_condition = _require_str(
                    upgrade_condition_value,
                    f"{requirement_path}.upgrade_condition"
                )
            owner_gate_value = requirement_mapping.get("owner_gate")
            if owner_gate_value is not None:
                owner_gate = _require_str(
                    owner_gate_value,
                    f"{requirement_path}.owner_gate"
                )
        parsed[requirement_name] = GateEnforcementRequirement(
            enforcement=enforcement,
            requirement=requirement_text,
            authority_source=authority_source,
            planned_state=planned_state,
            target_version=target_version,
            upgrade_condition=upgrade_condition,
            owner_gate=owner_gate
        )
    return parsed


def _parse_gate_enforcement_policies(root: Dict[str, Any]) -> GateEnforcementRequirements:
    """
    功能：解析 gate_enforcement_requirements 中的闭包策略。

    Parse closure policies defined in gate_enforcement_requirements.

    Args:
        root: Root mapping.

    Returns:
        GateEnforcementRequirements instance.

    Raises:
        MissingRequiredFieldError: If required policy fields are missing.
        TypeError: If policy fields are invalid.
    """
    requirements = _require_mapping(
        _get_required_field(root, "gate_enforcement_requirements", "<root>.gate_enforcement_requirements"),
        "<root>.gate_enforcement_requirements"
    )

    facts_policy = _require_mapping(
        _get_required_field(
            requirements,
            "facts_extra_keys_policy",
            "<root>.gate_enforcement_requirements.facts_extra_keys_policy"
        ),
        "<root>.gate_enforcement_requirements.facts_extra_keys_policy"
    )
    facts_policy_value = _require_str(
        _get_required_field(facts_policy, "policy", "<root>.gate_enforcement_requirements.facts_extra_keys_policy.policy"),
        "<root>.gate_enforcement_requirements.facts_extra_keys_policy.policy"
    )
    facts_scope = _require_mapping(
        _get_required_field(facts_policy, "scope", "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope"),
        "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope"
    )
    facts_allowed_top = _require_mapping_of_str_to_list_of_str(
        _get_required_field(
            facts_scope,
            "allowed_top_level_keys",
            "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope.allowed_top_level_keys"
        ),
        "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope.allowed_top_level_keys"
    )
    facts_allowed_policy_keys = _require_list_of_str(
        _get_required_field(
            facts_scope,
            "allowed_policy_path_keys",
            "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope.allowed_policy_path_keys"
        ),
        "<root>.gate_enforcement_requirements.facts_extra_keys_policy.scope.allowed_policy_path_keys"
    )

    policy_path_policy = _require_mapping(
        _get_required_field(
            requirements,
            "policy_path_membership_policy",
            "<root>.gate_enforcement_requirements.policy_path_membership_policy"
        ),
        "<root>.gate_enforcement_requirements.policy_path_membership_policy"
    )
    policy_path_value = _require_str(
        _get_required_field(policy_path_policy, "policy", "<root>.gate_enforcement_requirements.policy_path_membership_policy.policy"),
        "<root>.gate_enforcement_requirements.policy_path_membership_policy.policy"
    )
    policy_path_scope = _require_mapping(
        _get_required_field(
            policy_path_policy,
            "scope",
            "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope"
        ),
        "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope"
    )
    record_field_path = _require_str(
        _get_required_field(policy_path_scope, "record_field_path", "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope.record_field_path"),
        "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope.record_field_path"
    )
    whitelist_allowed_path = _require_str(
        _get_required_field(policy_path_scope, "whitelist_allowed_path", "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope.whitelist_allowed_path"),
        "<root>.gate_enforcement_requirements.policy_path_membership_policy.scope.whitelist_allowed_path"
    )

    override_policy = _require_mapping(
        _get_required_field(
            requirements,
            "override_enum_policy",
            "<root>.gate_enforcement_requirements.override_enum_policy"
        ),
        "<root>.gate_enforcement_requirements.override_enum_policy"
    )
    override_policy_value = _require_str(
        _get_required_field(override_policy, "policy", "<root>.gate_enforcement_requirements.override_enum_policy.policy"),
        "<root>.gate_enforcement_requirements.override_enum_policy.policy"
    )
    override_scope = _require_mapping(
        _get_required_field(
            override_policy,
            "scope",
            "<root>.gate_enforcement_requirements.override_enum_policy.scope"
        ),
        "<root>.gate_enforcement_requirements.override_enum_policy.scope"
    )
    override_entries_path = _require_str(
        _get_required_field(override_scope, "override_entries_path", "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_entries_path"),
        "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_entries_path"
    )
    override_arg_allowed_path = _require_str(
        _get_required_field(override_scope, "override_arg_name_allowed_path", "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_arg_name_allowed_path"),
        "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_arg_name_allowed_path"
    )
    override_source_allowed_path = _require_str(
        _get_required_field(override_scope, "override_source_allowed_path", "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_source_allowed_path"),
        "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_source_allowed_path"
    )
    override_entry_fields = _require_mapping_of_str_to_str(
        _get_required_field(override_scope, "override_entry_fields", "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_entry_fields"),
        "<root>.gate_enforcement_requirements.override_enum_policy.scope.override_entry_fields"
    )

    decoder_policy = _require_mapping(
        _get_required_field(
            requirements,
            "decoder_type_policy",
            "<root>.gate_enforcement_requirements.decoder_type_policy"
        ),
        "<root>.gate_enforcement_requirements.decoder_type_policy"
    )
    decoder_policy_value = _require_str(
        _get_required_field(decoder_policy, "policy", "<root>.gate_enforcement_requirements.decoder_type_policy.policy"),
        "<root>.gate_enforcement_requirements.decoder_type_policy.policy"
    )
    decoder_scope = _require_mapping(
        _get_required_field(
            decoder_policy,
            "scope",
            "<root>.gate_enforcement_requirements.decoder_type_policy.scope"
        ),
        "<root>.gate_enforcement_requirements.decoder_type_policy.scope"
    )
    semantics_policy_paths_path = _require_str(
        _get_required_field(decoder_scope, "semantics_policy_paths_path", "<root>.gate_enforcement_requirements.decoder_type_policy.scope.semantics_policy_paths_path"),
        "<root>.gate_enforcement_requirements.decoder_type_policy.scope.semantics_policy_paths_path"
    )
    semantics_decoder_type_field = _require_str(
        _get_required_field(decoder_scope, "semantics_decoder_type_field", "<root>.gate_enforcement_requirements.decoder_type_policy.scope.semantics_decoder_type_field"),
        "<root>.gate_enforcement_requirements.decoder_type_policy.scope.semantics_decoder_type_field"
    )
    whitelist_decoder_type_allowed_path = _require_str(
        _get_required_field(decoder_scope, "whitelist_decoder_type_allowed_path", "<root>.gate_enforcement_requirements.decoder_type_policy.scope.whitelist_decoder_type_allowed_path"),
        "<root>.gate_enforcement_requirements.decoder_type_policy.scope.whitelist_decoder_type_allowed_path"
    )

    return GateEnforcementRequirements(
        facts_extra_keys_policy=FactsExtraKeysPolicySpec(
            policy=facts_policy_value,
            allowed_top_level_keys=facts_allowed_top,
            allowed_policy_path_keys=facts_allowed_policy_keys
        ),
        policy_path_membership_policy=PolicyPathMembershipPolicySpec(
            policy=policy_path_value,
            record_field_path=record_field_path,
            whitelist_allowed_path=whitelist_allowed_path
        ),
        override_enum_policy=OverrideEnumPolicySpec(
            policy=override_policy_value,
            override_entries_path=override_entries_path,
            override_arg_name_allowed_path=override_arg_allowed_path,
            override_source_allowed_path=override_source_allowed_path,
            override_entry_fields=override_entry_fields
        ),
        decoder_type_policy=DecoderTypePolicySpec(
            policy=decoder_policy_value,
            semantics_policy_paths_path=semantics_policy_paths_path,
            semantics_decoder_type_field=semantics_decoder_type_field,
            whitelist_decoder_type_allowed_path=whitelist_decoder_type_allowed_path
        )
    )


def _parse_record_accessors(root: Dict[str, Any]) -> RecordAccessorsSpec:
    """
    功能：解析 record_accessors 段落。
    
    Parse record_accessors section.
    
    Args:
        root: Root mapping.
    
    Returns:
        RecordAccessorsSpec object.
    """
    accessors = _require_mapping(
        _get_required_field(root, "record_accessors", "<root>.record_accessors"),
        "<root>.record_accessors"
    )
    canonical_module = _require_str(
        _get_required_field(accessors, "canonical_module", "<root>.record_accessors.canonical_module"),
        "<root>.record_accessors.canonical_module"
    )
    minimum_required_accessors = _require_list_of_str(
        _get_required_field(
            accessors,
            "minimum_required_accessors",
            "<root>.record_accessors.minimum_required_accessors"
        ),
        "<root>.record_accessors.minimum_required_accessors"
    )
    delegation_scope = _require_mapping(
        _get_required_field(accessors, "delegation_scope", "<root>.record_accessors.delegation_scope"),
        "<root>.record_accessors.delegation_scope"
    )
    scripts_mandatory = _require_bool(
        _get_required_field(
            delegation_scope,
            "scripts_mandatory",
            "<root>.record_accessors.delegation_scope.scripts_mandatory"
        ),
        "<root>.record_accessors.delegation_scope.scripts_mandatory"
    )
    main_chain_gradual = _require_bool(
        _get_required_field(
            delegation_scope,
            "main_chain_gradual",
            "<root>.record_accessors.delegation_scope.main_chain_gradual"
        ),
        "<root>.record_accessors.delegation_scope.main_chain_gradual"
    )
    notes = _require_str(
        _get_required_field(accessors, "notes", "<root>.record_accessors.notes"),
        "<root>.record_accessors.notes"
    )
    delegation = RecordAccessorsDelegationScope(
        scripts_mandatory=scripts_mandatory,
        main_chain_gradual=main_chain_gradual
    )
    return RecordAccessorsSpec(
        canonical_module=canonical_module,
        minimum_required_accessors=minimum_required_accessors,
        delegation_scope=delegation,
        notes=notes
    )


def _parse_records_schema(root: Dict[str, Any]) -> RecordsSchemaSpec:
    """
    功能：解析 records_schema 段落。
    
    Parse records_schema section.
    
    Args:
        root: Root mapping.
    
    Returns:
        RecordsSchemaSpec object.
    """
    schema = _require_mapping(
        _get_required_field(root, "records_schema", "<root>.records_schema"),
        "<root>.records_schema"
    )
    chain_enablement_fields = _require_mapping_of_str_to_str(
        _get_required_field(
            schema,
            "chain_enablement_fields",
            "<root>.records_schema.chain_enablement_fields"
        ),
        "<root>.records_schema.chain_enablement_fields"
    )
    decision_field_path = _require_str(
        _get_required_field(schema, "decision_field_path", "<root>.records_schema.decision_field_path"),
        "<root>.records_schema.decision_field_path"
    )
    required_record_fields = _require_list_of_str(
        _get_required_field(schema, "required_record_fields", "<root>.records_schema.required_record_fields"),
        "<root>.records_schema.required_record_fields"
    )
    field_paths_registry_semantics = _require_str(
        _get_required_field(
            schema,
            "field_paths_registry_semantics",
            "<root>.records_schema.field_paths_registry_semantics"
        ),
        "<root>.records_schema.field_paths_registry_semantics"
    )
    field_paths_registry = _require_list_of_str(
        _get_required_field(schema, "field_paths_registry", "<root>.records_schema.field_paths_registry"),
        "<root>.records_schema.field_paths_registry"
    )
    
    # 解析 audit_obligations，若缺失则使用默认值。
    audit_obligations = schema.get("audit_obligations")
    decision_obligation_presence = False
    decision_allow_null = False
    if audit_obligations is not None:
        if not isinstance(audit_obligations, dict):
            raise TypeError("Type mismatch at <root>.records_schema.audit_obligations: expected mapping")
        decision_field_presence_str = audit_obligations.get("decision_field_presence")
        if decision_field_presence_str == "required":
            decision_obligation_presence = True
        decision_allow_null = audit_obligations.get("decision_field_allow_null", False)
        if not isinstance(decision_allow_null, bool):
            raise TypeError("Type mismatch at <root>.records_schema.audit_obligations.decision_field_allow_null: expected bool")
    
    return RecordsSchemaSpec(
        chain_enablement_fields=chain_enablement_fields,
        decision_field_path=decision_field_path,
        required_record_fields=required_record_fields,
        decision_obligation_presence=decision_obligation_presence,
        decision_allow_null=decision_allow_null,
        field_paths_registry_semantics=field_paths_registry_semantics,
        field_paths_registry=field_paths_registry
    )


def _parse_config_loader(root: Dict[str, Any]) -> ConfigLoaderSpec:
    """
    功能：解析 config_loader 段落。

    Parse config_loader section.

    Args:
        root: Root mapping.

    Returns:
        ConfigLoaderSpec object.
    """
    config_loader = _require_mapping(
        _get_required_field(root, "config_loader", "<root>.config_loader"),
        "<root>.config_loader"
    )
    include_paths = _require_list_of_str(
        _get_required_field(
            config_loader,
            "cfg_digest_include_paths",
            "<root>.config_loader.cfg_digest_include_paths"
        ),
        "<root>.config_loader.cfg_digest_include_paths"
    )
    override_included = _get_required_field(
        config_loader,
        "cfg_digest_override_applied_included",
        "<root>.config_loader.cfg_digest_override_applied_included"
    )
    override_included = _require_bool(
        override_included,
        "<root>.config_loader.cfg_digest_override_applied_included"
    )
    return ConfigLoaderSpec(
        cfg_digest_include_paths=include_paths,
        cfg_digest_override_applied_included=override_included
    )


def _parse_impl_identity(root: Dict[str, Any]) -> ImplIdentitySpec:
    """
    功能：解析 impl_identity 段落。
    
    Parse impl_identity section.
    
    Args:
        root: Root mapping.
    
    Returns:
        ImplIdentitySpec object.
    """
    impl_identity = _require_mapping(
        _get_required_field(root, "impl_identity", "<root>.impl_identity"),
        "<root>.impl_identity"
    )
    required = _require_bool(
        _get_required_field(impl_identity, "required", "<root>.impl_identity.required"),
        "<root>.impl_identity.required"
    )
    field_paths_by_domain = _require_mapping_of_str_to_str(
        _get_required_field(
            impl_identity,
            "field_paths_by_domain",
            "<root>.impl_identity.field_paths_by_domain"
        ),
        "<root>.impl_identity.field_paths_by_domain"
    )
    version_field_paths_by_domain = _require_mapping_of_str_to_str(
        _get_required_field(
            impl_identity,
            "version_field_paths_by_domain",
            "<root>.impl_identity.version_field_paths_by_domain"
        ),
        "<root>.impl_identity.version_field_paths_by_domain"
    )
    digest_field_paths_by_domain = _require_mapping_of_str_to_str(
        _get_required_field(
            impl_identity,
            "digest_field_paths_by_domain",
            "<root>.impl_identity.digest_field_paths_by_domain"
        ),
        "<root>.impl_identity.digest_field_paths_by_domain"
    )
    validation_rule = _require_str(
        _get_required_field(impl_identity, "validation_rule", "<root>.impl_identity.validation_rule"),
        "<root>.impl_identity.validation_rule"
    )
    return ImplIdentitySpec(
        required=required,
        field_paths_by_domain=field_paths_by_domain,
        version_field_paths_by_domain=version_field_paths_by_domain,
        digest_field_paths_by_domain=digest_field_paths_by_domain,
        validation_rule=validation_rule
    )


def _parse_status_enum(root: Dict[str, Any]) -> StatusEnumSpec:
    """
    功能：解析 status_enum 段落。
    
    Parse status_enum section.
    
    Args:
        root: Root mapping.
    
    Returns:
        StatusEnumSpec object.
    """
    status_enum = _require_mapping(
        _get_required_field(root, "status_enum", "<root>.status_enum"),
        "<root>.status_enum"
    )
    allowed_values = _require_list_of_str(
        _get_required_field(status_enum, "allowed_values", "<root>.status_enum.allowed_values"),
        "<root>.status_enum.allowed_values"
    )
    definitions = _require_mapping_of_str_to_str(
        _get_required_field(status_enum, "definitions", "<root>.status_enum.definitions"),
        "<root>.status_enum.definitions"
    )
    immutable = _require_bool(
        _get_required_field(status_enum, "immutable", "<root>.status_enum.immutable"),
        "<root>.status_enum.immutable"
    )
    notes = _require_str(
        _get_required_field(status_enum, "notes", "<root>.status_enum.notes"),
        "<root>.status_enum.notes"
    )
    return StatusEnumSpec(
        allowed_values=allowed_values,
        definitions=definitions,
        immutable=immutable,
        notes=notes
    )


def _parse_fail_reason_enum(root: Dict[str, Any]) -> FailReasonEnumSpec:
    """
    功能：解析 fail_reason_enum 段落。
    
    Parse fail_reason_enum section.
    
    Args:
        root: Root mapping.
    
    Returns:
        FailReasonEnumSpec object.
    """
    fail_reason_enum = _require_mapping(
        _get_required_field(root, "fail_reason_enum", "<root>.fail_reason_enum"),
        "<root>.fail_reason_enum"
    )
    allowed_values = _require_list_of_str(
        _get_required_field(
            fail_reason_enum,
            "allowed_values",
            "<root>.fail_reason_enum.allowed_values"
        ),
        "<root>.fail_reason_enum.allowed_values"
    )
    usage = _require_str(
        _get_required_field(fail_reason_enum, "usage", "<root>.fail_reason_enum.usage"),
        "<root>.fail_reason_enum.usage"
    )
    immutable = _require_bool(
        _get_required_field(fail_reason_enum, "immutable", "<root>.fail_reason_enum.immutable"),
        "<root>.fail_reason_enum.immutable"
    )
    return FailReasonEnumSpec(
        allowed_values=allowed_values,
        usage=usage,
        immutable=immutable
    )


def _parse_mismatch_reason_enum(root: Dict[str, Any]) -> MismatchReasonEnumSpec:
    """
    功能：解析 mismatch_reason_enum 段落。
    
    Parse mismatch_reason_enum section.
    
    Args:
        root: Root mapping.
    
    Returns:
        MismatchReasonEnumSpec object.
    """
    mismatch_reason_enum = _require_mapping(
        _get_required_field(root, "mismatch_reason_enum", "<root>.mismatch_reason_enum"),
        "<root>.mismatch_reason_enum"
    )
    allowed_values = _require_list_of_str(
        _get_required_field(
            mismatch_reason_enum,
            "allowed_values",
            "<root>.mismatch_reason_enum.allowed_values"
        ),
        "<root>.mismatch_reason_enum.allowed_values"
    )
    usage = _require_str(
        _get_required_field(mismatch_reason_enum, "usage", "<root>.mismatch_reason_enum.usage"),
        "<root>.mismatch_reason_enum.usage"
    )
    immutable = _require_bool(
        _get_required_field(mismatch_reason_enum, "immutable", "<root>.mismatch_reason_enum.immutable"),
        "<root>.mismatch_reason_enum.immutable"
    )
    return MismatchReasonEnumSpec(
        allowed_values=allowed_values,
        usage=usage,
        immutable=immutable
    )


def _collect_required_record_fields(
    external_fact_sources: Dict[str, ExternalFactSourceSpec],
    records_schema: RecordsSchemaSpec
) -> List[str]:
    """
    功能：汇总外部事实源必需字段集合。
    
    Collect required record fields from external fact sources and records_schema.
    
    Args:
        external_fact_sources: Mapping of external fact sources.
    
    Returns:
        List of required record field names.
    """
    if not isinstance(external_fact_sources, dict):
        # 外部事实源类型不符合预期，必须 fail-fast。
        raise TypeError("Type mismatch at <root>.external_fact_sources: expected mapping")
    if not isinstance(records_schema, RecordsSchemaSpec):
        # records_schema 类型不符合预期，必须 fail-fast。
        raise TypeError("Type mismatch at <root>.records_schema: expected RecordsSchemaSpec")
    required: List[str] = []
    for source_name, source_spec in external_fact_sources.items():
        if not isinstance(source_name, str):
            # 外部事实源名称类型不匹配，必须 fail-fast。
            raise TypeError(
                "Type mismatch at <root>.external_fact_sources: expected str keys"
            )
        if not isinstance(source_spec, ExternalFactSourceSpec):
            # 外部事实源结构不符合预期，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at <root>.external_fact_sources.{source_name}: expected ExternalFactSourceSpec"
            )
        required.extend(source_spec.required_record_fields)
    return _merge_required_record_fields(required, records_schema.required_record_fields)


def _merge_required_record_fields(
    primary_fields: List[str],
    secondary_fields: List[str]
) -> List[str]:
    """
    功能：合并必需字段并去重。

    Merge required field lists while preserving order and uniqueness.

    Args:
        primary_fields: Primary ordered field list.
        secondary_fields: Secondary ordered field list.

    Returns:
        Merged list with stable order.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(primary_fields, list):
        # primary_fields 类型不符合预期，必须 fail-fast。
        raise TypeError("primary_fields must be list")
    if not isinstance(secondary_fields, list):
        # secondary_fields 类型不符合预期，必须 fail-fast。
        raise TypeError("secondary_fields must be list")

    merged: List[str] = []
    seen = set()
    for field_name in primary_fields + secondary_fields:
        if not isinstance(field_name, str) or not field_name:
            # field_name 类型不合法，必须 fail-fast。
            raise TypeError("required_record_fields entries must be non-empty str")
        if field_name in seen:
            continue
        seen.add(field_name)
        merged.append(field_name)
    return merged


def _validate_records_schema_required_fields(records_schema: RecordsSchemaSpec) -> None:
    """
    功能：校验 records_schema.required_record_fields 注册一致性。

    Validate that required_record_fields are registered in field_paths_registry.

    Args:
        records_schema: Parsed RecordsSchemaSpec.

    Raises:
        TypeError: If records_schema is invalid.
        ValueError: If required fields are not registered.
    """
    if not isinstance(records_schema, RecordsSchemaSpec):
        # records_schema 类型不符合预期，必须 fail-fast。
        raise TypeError("records_schema must be RecordsSchemaSpec")

    registry = set(records_schema.field_paths_registry)
    missing = [
        field_name
        for field_name in records_schema.required_record_fields
        if field_name not in registry
    ]
    if missing:
        # 注册表缺失必需字段，必须 fail-fast。
        raise ValueError(
            "<root>.records_schema.required_record_fields not registered: "
            f"{missing}"
        )
