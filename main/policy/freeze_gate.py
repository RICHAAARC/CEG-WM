"""
写盘前门禁校验

功能说明：
- 根据契约解释面定义的 schema 验证记录结构和字段。
- 门禁要求执行：执行 must_enforce 的门禁要求。
- 契约化闭包策略执行：执行契约中定义的闭包策略。
- 记录 recommended_enforce 的检查结果。
"""

from typing import Dict, Any, List

from main.core.errors import (
    MissingRequiredFieldError,
    DigestMismatchError,
    ContractVersionMismatchError,
    GateRequirementNotImplementedError,
    GateEnforcementError,
    FrozenContractPathNotAuthoritativeError
)
from main.core import schema
from main.core.contracts import (
    FrozenContracts,
    get_contract_interpretation,
    ContractInterpretation,
    FactsExtraKeysPolicySpec,
    PolicyPathMembershipPolicySpec,
    OverrideEnumPolicySpec,
    DecoderTypePolicySpec
)
from main.policy.runtime_whitelist import (
    RuntimeWhitelist,
    PolicyPathSemantics,
    enforce_must_enforce_rules,
    assert_pipeline_impl_allowed,
    assert_pipeline_provenance_digest_consistent,
    assert_pipeline_model_source_allowed,
    assert_pipeline_hf_revision_required,
    assert_pipeline_weights_snapshot_required,
    assert_pipeline_hf_hub_download_allowed,
    assert_env_fingerprint_required,
    assert_inference_device_allowed,
    assert_inference_precision_allowed
)
from main.registries import pipeline_registry
from main.diffusion.sd3 import weights_snapshot


def assert_prewrite(
    record: Dict[str, Any],
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> Dict[str, Any]:
    """
    功能：写盘前门禁校验。
    
    Validate record before write:
    (a) All required fields from external_fact_sources are present.
    (b) version/digest/sha256 in record match loaded objects.
    (c) Digests in record are consistent with recomputed digests.
    (d) impl_id values are whitelist-allowed.
    (e) recommended_enforce items are checked and recorded (but do not fail).
    (f) policy_path_semantics.version == runtime_whitelist.version binding check.
    
    Args:
        record: Record dict to validate.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
    
    Returns:
        Audit record for recommended_enforce findings (empty if none).
    
    Raises:
        MissingRequiredFieldError: If required field is missing.
        ContractVersionMismatchError: If version mismatch.
        DigestMismatchError: If digest mismatch.
        GateEnforcementError: If impl_id is not whitelist-allowed or versions mismatch.
    """
    contract_source_path = getattr(contracts, "contract_source_path", None)
    if contract_source_path != "configs/frozen_contracts.yaml":
        raise FrozenContractPathNotAuthoritativeError(
            "frozen_contracts path must be authoritative",
            field_path="contract_source_path",
            actual_path=str(contract_source_path)
        )
    # policy_path_semantics 与 runtime_whitelist 版本强绑定检查。
    _enforce_policy_semantics_whitelist_version_binding(whitelist, semantics)
    
    # 解释面驱动的权威 schema 校验。
    interpretation = get_contract_interpretation(contracts)
    schema.validate_record(record, interpretation=interpretation)
    _enforce_schema_version_consistency(record)
    
    # 检查 records_schema_extensions 绑定状态（向后兼容过渡）。
    _enforce_records_schema_extensions_binding(record, interpretation)
    
    #  runtime_whitelist must_enforce 规则执行。
    enforce_must_enforce_rules(whitelist, semantics, contracts, record)

    # pipeline_shell 门禁：记录必须绑定 pipeline allowlist 与 provenance digest。
    _enforce_pipeline_shell_binding(record, whitelist)
    # pipeline_realization 门禁：真实加载的附加约束。
    _enforce_pipeline_realization_requirements(record, whitelist)

    # 解释面驱动的门禁执行要求。
    enforce_gate_requirements(record, interpretation, contracts, whitelist, semantics)
    # 契约化闭包策略执行。
    enforce_gate_policies(record, interpretation, whitelist, semantics)
    
    # recommended_enforce 项的检查结果。
    recommendations = enforce_recommended_requirements(record, interpretation, whitelist, semantics)
    
    # 阶段 4：门禁硬化执行（enforce 模式）
    _validate_statistical_fields(record, interpretation, warn_mode=False)
    _validate_semantic_driven_execution(record, semantics, interpretation, warn_mode=False)
    _check_field_override_conflict(record, interpretation, warn_mode=False)
    
    return recommendations


def _enforce_pipeline_shell_binding(record: Dict[str, Any], whitelist: RuntimeWhitelist) -> None:
    """
    功能：执行 pipeline_shell 门禁校验。

    Enforce pipeline_impl_id allowlist (pipeline registry) and provenance digest consistency.

    Args:
        record: Record dict to validate.
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If pipeline shell binding checks fail.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")

    if not _should_enforce_pipeline_shell(record):
        return

    pipeline_impl_id = record.get("pipeline_impl_id")
    if not isinstance(pipeline_impl_id, str) or not pipeline_impl_id:
        raise GateEnforcementError(
            "pipeline_impl_id missing or invalid",
            gate_name="pipeline_impl_id.allowlist",
            field_path="pipeline_impl_id"
        )

    allowlist = pipeline_registry.list_pipeline_impl_ids()
    if pipeline_impl_id not in allowlist:
        raise GateEnforcementError(
            "pipeline_impl_id not allowed",
            gate_name="pipeline_impl_id.allowlist",
            field_path="pipeline_impl_id",
            expected=str(allowlist),
            actual=pipeline_impl_id
        )
    assert_pipeline_provenance_digest_consistent(record)


def _should_enforce_pipeline_shell(record: Dict[str, Any]) -> bool:
    """
    功能：判定是否需要执行 pipeline_shell 门禁。

    Decide whether pipeline shell enforcement is required for this record.

    Args:
        record: Record mapping.

    Returns:
        True if pipeline enforcement should be applied; otherwise False.

    Raises:
        TypeError: If record is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")

    if "pipeline_impl_id" in record or "pipeline_provenance" in record or "pipeline_provenance_canon_sha256" in record:
        return True
    operation = record.get("operation")
    return operation in {"embed", "detect"}


def _enforce_pipeline_realization_requirements(
    record: Dict[str, Any],
    whitelist: RuntimeWhitelist
) -> None:
    """
    功能：执行 pipeline_realization 门禁校验。

    Enforce pipeline realization requirements for real pipeline impl.

    Args:
        record: Record dict to validate.
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If realization requirements fail.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")

    pipeline_impl_id = record.get("pipeline_impl_id")
    if pipeline_impl_id != pipeline_registry.SD3_DIFFUSERS_REAL_ID:
        return

    pipeline_provenance = record.get("pipeline_provenance")
    if not isinstance(pipeline_provenance, dict):
        raise GateEnforcementError(
            "pipeline_provenance required",
            gate_name="pipeline_realization.provenance.required",
            field_path="pipeline_provenance"
        )

    model_source = pipeline_provenance.get("model_source")
    if not isinstance(model_source, str) or not model_source or model_source == "<absent>":
        raise GateEnforcementError(
            "model_source required",
            gate_name="pipeline_realization.model_source.required",
            field_path="pipeline_provenance.model_source",
            expected="non-empty",
            actual=str(model_source)
        )
    assert_pipeline_model_source_allowed(whitelist, model_source)

    hf_revision = pipeline_provenance.get("hf_revision")
    assert_pipeline_hf_revision_required(whitelist, hf_revision, "pipeline_provenance.hf_revision")

    # 支持 "hf" 和 "hf_hub" 两种标识（向后兼容）
    if model_source in ("hf", "hf_hub"):
        local_files_only = pipeline_provenance.get("local_files_only")
        if not isinstance(local_files_only, bool):
            raise GateEnforcementError(
                "local_files_only required for hf_hub",
                gate_name="pipeline_realization.hf_hub_download.allowed",
                field_path="pipeline_provenance.local_files_only",
                expected="bool",
                actual=str(local_files_only)
            )
        download_allowed = not local_files_only
        assert_pipeline_hf_hub_download_allowed(
            whitelist,
            download_allowed,
            "pipeline_provenance.local_files_only"
        )

    env_fingerprint_canon_sha256 = record.get("env_fingerprint_canon_sha256")
    assert_env_fingerprint_required(whitelist, env_fingerprint_canon_sha256)

    runtime_meta = record.get("pipeline_runtime_meta")
    weights_snapshot_sha256 = None
    if isinstance(runtime_meta, dict):
        weights_snapshot_sha256 = runtime_meta.get("weights_snapshot_sha256")
    assert_pipeline_weights_snapshot_required(
        whitelist,
        weights_snapshot_sha256,
        "pipeline_runtime_meta.weights_snapshot_sha256"
    )

    if isinstance(weights_snapshot_sha256, str) and weights_snapshot_sha256 and weights_snapshot_sha256 != "<absent>":
        model_weights_sha256 = pipeline_provenance.get("model_weights_sha256")
        if not isinstance(model_weights_sha256, str) or not model_weights_sha256:
            raise GateEnforcementError(
                "model_weights_sha256 required for weights_snapshot_sha256",
                gate_name="pipeline_realization.weights_snapshot_sha256.consistency",
                field_path="pipeline_provenance.model_weights_sha256",
                expected=weights_snapshot_sha256,
                actual=str(model_weights_sha256)
            )
        if model_weights_sha256 != weights_snapshot_sha256:
            raise GateEnforcementError(
                "weights_snapshot_sha256 mismatch",
                gate_name="pipeline_realization.weights_snapshot_sha256.consistency",
                field_path="pipeline_runtime_meta.weights_snapshot_sha256",
                expected=model_weights_sha256,
                actual=weights_snapshot_sha256
            )

        provenance_snapshot = pipeline_provenance.get("weights_snapshot_sha256")
        if not isinstance(provenance_snapshot, str) or not provenance_snapshot:
            raise GateEnforcementError(
                "pipeline_provenance.weights_snapshot_sha256 required",
                gate_name="pipeline_realization.weights_snapshot_sha256.provenance",
                field_path="pipeline_provenance.weights_snapshot_sha256",
                expected=weights_snapshot_sha256,
                actual=str(provenance_snapshot)
            )
        if provenance_snapshot != weights_snapshot_sha256:
            raise GateEnforcementError(
                "pipeline_provenance.weights_snapshot_sha256 mismatch",
                gate_name="pipeline_realization.weights_snapshot_sha256.provenance",
                field_path="pipeline_provenance.weights_snapshot_sha256",
                expected=weights_snapshot_sha256,
                actual=provenance_snapshot
            )

        model_id = pipeline_provenance.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id == "<absent>":
            raise GateEnforcementError(
                "model_id required for weights_snapshot recompute",
                gate_name="pipeline_realization.weights_snapshot_sha256.recompute",
                field_path="pipeline_provenance.model_id",
                expected="non-empty",
                actual=str(model_id)
            )

        local_files_only = pipeline_provenance.get("local_files_only")
        if not isinstance(local_files_only, bool):
            local_files_only = None

        cache_dir = None
        if isinstance(runtime_meta, dict):
            build_kwargs = runtime_meta.get("build_kwargs")
            if isinstance(build_kwargs, dict):
                cache_dir = build_kwargs.get("cache_dir") if isinstance(build_kwargs.get("cache_dir"), str) else None

        recomputed, snapshot_meta, snapshot_error = weights_snapshot.compute_weights_snapshot_sha256(
            model_id=model_id,
            model_source=model_source,
            hf_revision=hf_revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
        if snapshot_error is not None:
            raise GateEnforcementError(
                "weights_snapshot_sha256 recompute failed",
                gate_name="pipeline_realization.weights_snapshot_sha256.recompute",
                field_path="pipeline_runtime_meta.weights_snapshot_sha256",
                expected=weights_snapshot_sha256,
                actual=str(snapshot_error)
            )
        if recomputed != weights_snapshot_sha256:
            raise GateEnforcementError(
                "weights_snapshot_sha256 recompute mismatch",
                gate_name="pipeline_realization.weights_snapshot_sha256.recompute",
                field_path="pipeline_runtime_meta.weights_snapshot_sha256",
                expected=recomputed,
                actual=weights_snapshot_sha256
            )

    # (7.7) Real Dataflow Smoke: inference 策略断言
    inference_runtime_meta = record.get("inference_runtime_meta")
    if isinstance(inference_runtime_meta, dict):
        device = inference_runtime_meta.get("device")
        if isinstance(device, str) and device and device != "<absent>":
            assert_inference_device_allowed(whitelist, device, "inference_runtime_meta.device")
        
        # precision 校验（预留扩展）
        precision = inference_runtime_meta.get("precision")
        if isinstance(precision, str) and precision and precision != "<absent>":
            assert_inference_precision_allowed(whitelist, precision, "inference_runtime_meta.precision")


def enforce_gate_requirements(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：执行冻结契约中的 must_enforce 门禁要求。

    Enforce must_enforce gate requirements defined by contract interpretation.

    Args:
        record: Record dict to validate.
        interpretation: Parsed contract interpretation.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.

    Returns:
        None.

    Raises:
        GateRequirementNotImplementedError: If a must_enforce requirement has no handler.
        MissingRequiredFieldError: If required fields are missing.
        DigestMismatchError: If digest mismatch occurs.
        ContractVersionMismatchError: If version mismatch occurs.
        TypeError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")
    if not isinstance(contracts, FrozenContracts):
        # contracts 类型不符合预期，必须 fail-fast。
        raise TypeError("contracts must be FrozenContracts")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(semantics, PolicyPathSemantics):
        # semantics 类型不符合预期，必须 fail-fast。
        raise TypeError("semantics must be PolicyPathSemantics")

    handlers = {
        "impl_identity_domain_binding": _enforce_impl_identity_domain_binding,
        "fact_source_binding_integrity": _enforce_fact_source_binding_integrity,
        "cfg_digest_computation_order": _enforce_cfg_digest_computation_order,
        "policy_path_semantic_driven_execution": lambda record, contracts, whitelist, semantics, interpretation: _validate_semantic_driven_execution(record, semantics, interpretation, warn_mode=False)
    }
    policy_requirement_names = {
        "facts_extra_keys_policy",
        "policy_path_membership_policy",
        "override_enum_policy",
        "decoder_type_policy"
    }

    for requirement_name, requirement in interpretation.gate_enforcement_requirements.items():
        if requirement.enforcement != "must_enforce":
            continue
        if requirement_name in policy_requirement_names:
            # 该类策略由 enforce_gate_policies 统一执行。
            continue
        handler = handlers.get(requirement_name)
        if handler is None:
            rule_path = f"<root>.gate_enforcement_requirements.{requirement_name}"
            raise GateRequirementNotImplementedError(
                rule_name=requirement_name,
                rule_path=rule_path
            )
        handler(record, contracts, whitelist, semantics, interpretation)


def _enforce_schema_version_consistency(record: Dict[str, Any]) -> None:
    """
    功能：强制 schema_version 一致性。

    Enforce record schema_version matches authoritative version.

    Args:
        record: Record dict to validate.

    Raises:
        GateEnforcementError: If schema_version is missing or mismatched.
    """
    schema_version = record.get("schema_version")
    if schema_version != schema.RECORD_SCHEMA_VERSION:
        raise GateEnforcementError(
            "schema_version mismatch",
            gate_name="records_schema.schema_version.consistency",
            field_path="schema_version",
            expected=schema.RECORD_SCHEMA_VERSION,
            actual=str(schema_version)
        )


def _enforce_records_schema_extensions_binding(
    record: Dict[str, Any],
    interpretation: ContractInterpretation
) -> None:
    """
    功能：执行 records_schema_extensions 绑定检查（向后兼容过渡）。

    Enforce records_schema_extensions binding according to enablement status.
    
    If extensions are enabled:
      - Recommended anchor fields SHOULD be present (but do not fail in transition)
    If extensions are not enabled:
      - Extensions fields are not required (absent_ok)
      - But audit_obligations may require recording the "not enabled" reason
    
    This implements the transition strategy where:
      - Old records (without extension binding) are allowed to pass
      - New records (with extension binding) are encouraged to follow the spec
      - In future phases, this can be upgraded to "must_enforce"

    Args:
        record: Record dict to validate.
        interpretation: Contract interpretation with extensions spec.

    Raises:
        None (current behavior is audit-record-only for recommended fields).
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    extensions_spec = interpretation.records_schema_extensions_spec
    
    if not extensions_spec.enabled:
        # 扩展未启用：旧 records 允许不绑定
        # NOTE: 可选地在 audit_obligations 中记录 "schema_extensions_not_enabled"
        return

    # 扩展已启用：检查推荐的锚点字段。
    # Enforcement: 当前行为仅记录审计信息，不触发失败。
    # Version-bound: 如需升级为强制失败，必须通过契约版本化变更。
    recommended_fields = [
        "records_schema_extensions_version",
        "records_schema_extensions_digest",
        "records_schema_extensions_file_sha256",
        "records_schema_extensions_canon_sha256",
        "records_schema_extensions_bound_digest"
    ]
    
    missing_recommended = []
    for field_path in recommended_fields:
        value = record.get(field_path)
        if value is None:
            missing_recommended.append(field_path)
    
    if missing_recommended:
        # 推荐项缺失仅记录审计信息，不改变现行门禁判定。
        pass


def enforce_gate_policies(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：执行契约化闭包策略。

    Enforce closure policies defined by gate_enforcement_requirements.

    Args:
        record: Record dict to validate.
        interpretation: Parsed contract interpretation.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.

    Returns:
        None.

    Raises:
        GateEnforcementError: If a policy fails under fail mode.
        TypeError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(semantics, PolicyPathSemantics):
        # semantics 类型不符合预期，必须 fail-fast。
        raise TypeError("semantics must be PolicyPathSemantics")

    policies = interpretation.gate_enforcement_policies
    _enforce_facts_extra_keys_policy(policies.facts_extra_keys_policy, whitelist, semantics)
    _enforce_policy_path_membership_policy(record, whitelist, policies.policy_path_membership_policy)
    _enforce_override_enum_policy(whitelist, policies.override_enum_policy)
    _enforce_decoder_type_policy(whitelist, semantics, policies.decoder_type_policy)


def enforce_recommended_requirements(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> Dict[str, Any]:
    """
    功能：执行 recommended_enforce 项检查。

    Execute recommendations from gate_enforcement_requirements with enforcement="recommended_enforce".
    
    Args:
        record: Record dict to check.
        interpretation: Parsed contract interpretation.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.

    Returns:
        Audit report dict with explicit execution status and reasons.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，但不 fail-fast，仅记录。
        pass
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，但不 fail-fast，仅记录。
        pass

    from main.core import time_utils

    items = []
    recommended_enforce_mode = "warn"
    # recommended_enforce 在 warn 模式下执行检查但不 fail，记录结果供审计

    for requirement_name, requirement in interpretation.gate_enforcement_requirements.items():
        if requirement.enforcement != "recommended_enforce":
            continue
        
        executed = False
        passed = False
        reason = None
        
        try:
            # 针对每个 recommended 项执行相应检查
            if requirement_name == "policy_path_semantic_driven_execution":
                _validate_semantic_driven_execution(record, semantics, interpretation, warn_mode=True)
                executed = True
                passed = True
                reason = "policy_path_semantic_driven_execution checked"
                
            elif requirement_name == "enumeration_whitelist_authority":
                # 检查枚举值是否仅来自 runtime_whitelist
                _check_enumeration_whitelist_authority(record, whitelist)
                executed = True
                passed = True
                reason = "enumeration_whitelist_authority checked"
                
            elif requirement_name == "args_digest_factor_separation":
                # 检查 args_digest 是否包含方法与非方法因子
                _check_args_digest_factor_separation(record)
                executed = True
                passed = True
                reason = "args_digest_factor_separation checked"
                
            else:
                # 未登记执行路径的 recommended 项
                executed = False
                passed = False
                reason = f"recommended_enforce execution path not registered: {requirement_name}"
                
        except Exception as exc:
            # 检查执行失败，记录异常但不中断流程
            executed = True
            passed = False
            reason = f"check failed: {type(exc).__name__}: {str(exc)[:100]}"
        
        # 三态语义映射：enforced / warned / not_executed
        if executed and passed:
            status = "enforced"
        elif executed and not passed:
            status = "warned"
        else:
            # not executed
            status = "not_executed"
        
        item = {
            "item_name": requirement_name,
            "mode": recommended_enforce_mode,
            "status": status,
            "executed": executed,
            "passed": passed,
            "reason": reason,
            "planned_state": requirement.planned_state,
            "target_version": requirement.target_version,
            "upgrade_condition": requirement.upgrade_condition,
            "owner_gate": requirement.owner_gate
        }
        items.append(item)

    return {
        "record_type": "gate_recommendations",
        "timestamp": time_utils.now_utc_iso_z(),
        "recommended_enforce_mode": recommended_enforce_mode,
        "item_count": len(items),
        "items": items
    }


def _emit_gate_warning(message: str) -> None:
    """
    功能：输出门禁警告信息。

    Emit a clear gate warning message.

    Args:
        message: Warning message.

    Returns:
        None.
    """
    print(f"[Gate][WARN] {message}")


def _require_policy_value(policy: str, policy_name: str) -> str:
    """
    功能：校验策略值是否合法。

    Validate policy value is fail or warn.

    Args:
        policy: Policy string.
        policy_name: Policy name for context.

    Returns:
        Policy string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If policy is invalid.
    """
    if not isinstance(policy_name, str) or not policy_name:
        # policy_name 输入不合法，必须 fail-fast。
        raise TypeError("policy_name must be non-empty str")
    if not isinstance(policy, str) or not policy:
        # policy 类型不合法，必须 fail-fast。
        raise TypeError(f"policy must be non-empty str: policy_name={policy_name}")
    if policy not in {"fail", "warn"}:
        # policy 取值非法，必须 fail-fast。
        raise ValueError(
            f"invalid gate policy value: policy_name={policy_name}, policy={policy}"
        )
    return policy


def _get_value_by_path(mapping: Dict[str, Any], field_path: str) -> tuple[bool, Any]:
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
        # mapping 类型不符合预期，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    for segment in field_path.split("."):
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict):
            return False, None
        if segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _handle_policy_failure(
    policy: str,
    message: str,
    gate_name: str | None = None,
    field_path: str | None = None,
    expected: str | None = None,
    actual: str | None = None
) -> None:
    """
    功能：按策略处理门禁失败。

    Handle gate failure with fail or warn behavior.
    Passes structured gate info for better error localization and auditing.

    Args:
        policy: Policy string ("fail" or "warn").
        message: Failure message summary.
        gate_name: Optional gate identifier (e.g. "gate.facts_extra_keys_policy").
        field_path: Optional field path causing failure (e.g. "policy_path_semantics.semantics_version").
        expected: Optional description of expected value.
        actual: Optional description of actual value observed.

    Returns:
        None.

    Raises:
        GateEnforcementError: If policy is "fail".
    """
    if policy == "fail":
        raise GateEnforcementError(
            message,
            gate_name=gate_name,
            field_path=field_path,
            expected=expected,
            actual=actual
        )
    _emit_gate_warning(message)


def _enforce_facts_extra_keys_policy(
    policy_spec: FactsExtraKeysPolicySpec,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：执行 facts extra keys 策略。

    Enforce extra keys policy for facts YAML objects.

    Args:
        policy_spec: Facts extra keys policy spec.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.

    Returns:
        None.
    """
    policy = _require_policy_value(policy_spec.policy, "facts_extra_keys_policy")

    whitelist_allowed = set(policy_spec.allowed_top_level_keys.get("runtime_whitelist", []))
    whitelist_keys = set(whitelist.data.keys())
    whitelist_extra = sorted(whitelist_keys - whitelist_allowed)
    if whitelist_extra:
        # 指定具体的字段路径以便定位
        field_path = f"runtime_whitelist.{whitelist_extra[0]}" if whitelist_extra else "runtime_whitelist"
        _handle_policy_failure(
            policy,
            "facts_extra_keys_policy violation: source=runtime_whitelist, "
            f"extra_keys={whitelist_extra}, allowed_keys={sorted(whitelist_allowed)}",
            gate_name="gate.facts_extra_keys_policy",
            field_path=field_path,
            expected=f"keys from {sorted(whitelist_allowed)}",
            actual=f"extra_keys={whitelist_extra}"
        )

    semantics_allowed = set(policy_spec.allowed_top_level_keys.get("policy_path_semantics", []))
    semantics_keys = set(semantics.data.keys())
    semantics_extra = sorted(semantics_keys - semantics_allowed)
    if semantics_extra:
        # 指定具体的字段路径以便定位
        field_path = f"policy_path_semantics.{semantics_extra[0]}" if semantics_extra else "policy_path_semantics"
        _handle_policy_failure(
            policy,
            "facts_extra_keys_policy violation: source=policy_path_semantics, "
            f"extra_keys={semantics_extra}, allowed_keys={sorted(semantics_allowed)}",
            gate_name="gate.facts_extra_keys_policy",
            field_path=field_path,
            expected=f"keys from {sorted(semantics_allowed)}",
            actual=f"extra_keys={semantics_extra}"
        )

    policy_paths = semantics.data.get("policy_paths", {})
    if isinstance(policy_paths, dict):
        allowed_policy_path_keys = set(policy_spec.allowed_policy_path_keys)
        for policy_name, policy_value in policy_paths.items():
            if not isinstance(policy_value, dict):
                continue
            extra = sorted(set(policy_value.keys()) - allowed_policy_path_keys)
            if extra:
                field_path = f"policy_path_semantics.policy_paths.{policy_name}.{extra[0]}"
                _handle_policy_failure(
                    policy,
                    "facts_extra_keys_policy violation: source=policy_path_semantics, "
                    f"policy_path={policy_name}, extra_keys={extra}, "
                    f"allowed_keys={sorted(allowed_policy_path_keys)}",
                    gate_name="gate.facts_extra_keys_policy",
                    field_path=field_path,
                    expected=f"keys from {sorted(allowed_policy_path_keys)}",
                    actual=f"extra_keys={extra}"
                )


def _enforce_policy_path_membership_policy(
    record: Dict[str, Any],
    whitelist: RuntimeWhitelist,
    policy_spec: PolicyPathMembershipPolicySpec
) -> None:
    """
    功能：执行 policy_path 值域闭包策略。

    Enforce policy_path membership against whitelist.

    Args:
        record: Record dict.
        whitelist: Loaded RuntimeWhitelist.
        policy_spec: Policy path membership policy spec.

    Returns:
        None.
    """
    policy = _require_policy_value(policy_spec.policy, "policy_path_membership_policy")
    found, policy_value = _get_value_by_path(record, policy_spec.record_field_path)
    if not found:
        _handle_policy_failure(
            policy,
            "policy_path_membership_policy missing: "
            f"field_path={policy_spec.record_field_path}"
        )
        return
    if not isinstance(policy_value, str) or not policy_value:
        _handle_policy_failure(
            policy,
            "policy_path_membership_policy type mismatch: "
            f"field_path={policy_spec.record_field_path}, actual_type={type(policy_value).__name__}"
        )
        return

    found_allowed, allowed_values = _get_value_by_path(whitelist.data, policy_spec.whitelist_allowed_path)
    if not found_allowed or not isinstance(allowed_values, list):
        _handle_policy_failure(
            policy,
            "policy_path_membership_policy whitelist path missing: "
            f"field_path={policy_spec.whitelist_allowed_path}"
        )
        return
    if policy_value not in allowed_values:
        _handle_policy_failure(
            policy,
            "policy_path_membership_policy violation: "
            f"value={policy_value}, allowed={allowed_values}"
        )


def _enforce_override_enum_policy(
    whitelist: RuntimeWhitelist,
    policy_spec: OverrideEnumPolicySpec
) -> None:
    """
    功能：执行 override 枚举闭包策略。

    Enforce override arg_name/source enums in runtime whitelist.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        policy_spec: Override enum policy spec.

    Returns:
        None.
    """
    policy = _require_policy_value(policy_spec.policy, "override_enum_policy")
    found_entries, entries = _get_value_by_path(whitelist.data, policy_spec.override_entries_path)
    if not found_entries or not isinstance(entries, list):
        _handle_policy_failure(
            policy,
            "override_enum_policy missing entries: "
            f"field_path={policy_spec.override_entries_path}"
        )
        return

    found_arg_allowed, arg_allowed = _get_value_by_path(
        whitelist.data, policy_spec.override_arg_name_allowed_path
    )
    found_source_allowed, source_allowed = _get_value_by_path(
        whitelist.data, policy_spec.override_source_allowed_path
    )
    if not found_arg_allowed or not isinstance(arg_allowed, list):
        _handle_policy_failure(
            policy,
            "override_enum_policy missing arg_name allowed list: "
            f"field_path={policy_spec.override_arg_name_allowed_path}"
        )
        return
    if not found_source_allowed or not isinstance(source_allowed, list):
        _handle_policy_failure(
            policy,
            "override_enum_policy missing source allowed list: "
            f"field_path={policy_spec.override_source_allowed_path}"
        )
        return

    for entry in entries:
        if not isinstance(entry, dict):
            _handle_policy_failure(
                policy,
                "override_enum_policy entry type mismatch: "
                f"actual_type={type(entry).__name__}"
            )
            continue
        arg_name = entry.get(policy_spec.override_entry_fields.get("arg_name", "arg_name"))
        source = entry.get(policy_spec.override_entry_fields.get("source", "source"))
        if arg_name not in arg_allowed:
            _handle_policy_failure(
                policy,
                "override_enum_policy arg_name not allowed: "
                f"arg_name={arg_name}, allowed={arg_allowed}"
            )
        if source not in source_allowed:
            _handle_policy_failure(
                policy,
                "override_enum_policy source not allowed: "
                f"source={source}, allowed={source_allowed}"
            )


def _enforce_decoder_type_policy(
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    policy_spec: DecoderTypePolicySpec
) -> None:
    """
    功能：执行 decoder_type 值域闭包策略。

    Enforce decoder_type values in policy_path_semantics against whitelist enums.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        policy_spec: Decoder type policy spec.

    Returns:
        None.
    """
    policy = _require_policy_value(policy_spec.policy, "decoder_type_policy")
    found_allowed, allowed_values = _get_value_by_path(
        whitelist.data, policy_spec.whitelist_decoder_type_allowed_path
    )
    if not found_allowed or not isinstance(allowed_values, list):
        _handle_policy_failure(
            policy,
            "decoder_type_policy missing allowed list: "
            f"field_path={policy_spec.whitelist_decoder_type_allowed_path}"
        )
        return

    found_paths, policy_paths = _get_value_by_path(
        semantics.data, policy_spec.semantics_policy_paths_path
    )
    if not found_paths or not isinstance(policy_paths, dict):
        _handle_policy_failure(
            policy,
            "decoder_type_policy missing policy_paths: "
            f"field_path={policy_spec.semantics_policy_paths_path}"
        )
        return

    decoder_field = policy_spec.semantics_decoder_type_field
    for policy_name, policy_value in policy_paths.items():
        if not isinstance(policy_value, dict):
            continue
        decoder_type = policy_value.get(decoder_field)
        if decoder_type not in allowed_values:
            _handle_policy_failure(
                policy,
                "decoder_type_policy violation: "
                f"policy_path={policy_name}, decoder_type={decoder_type}, "
                f"allowed={allowed_values}"
            )


def _require_field_paths_present(record: Dict[str, Any], field_paths: List[str]) -> None:
    """
    功能：校验字段路径存在性。

    Require that all dotted field paths exist in record.

    Args:
        record: Record dict to validate.
        field_paths: List of dotted field paths.

    Raises:
        MissingRequiredFieldError: If any field path is missing.
        TypeError: If inputs are invalid.
        ValueError: If field path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_paths, list):
        # field_paths 类型不符合预期，必须 fail-fast。
        raise TypeError("field_paths must be list")

    for field_path in field_paths:
        found, _ = _get_value_by_field_path_simple(record, field_path)
        if not found:
            # 必需字段缺失，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required field in record before write: {field_path}"
            )


def _get_value_by_field_path_simple(
    record: Dict[str, Any],
    field_path: str
) -> tuple[bool, Any]:
    """
    功能：按点路径读取 record 字段值。

    Read a nested record value by dotted field path.

    Args:
        record: Record dict to read.
        field_path: Dotted field path.

    Returns:
        Tuple of (found, value).

    Raises:
        TypeError: If record types are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = record
    for segment in field_path.split("."):
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict):
            return False, None
        if segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _enforce_impl_identity_domain_binding(
    record: Dict[str, Any],
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    interpretation: ContractInterpretation
) -> None:
    """
    功能：执行 impl_identity 域绑定校验。

    Enforce impl identity domain binding checks.

    Args:
        record: Record dict to validate.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        interpretation: ContractInterpretation instance.

    Raises:
        MissingRequiredFieldError: If required impl fields are missing.
        DigestMismatchError: If impl_id is not allowed.
        TypeError: If inputs are invalid.
    """
    _ = semantics
    _ = interpretation
    _validate_impl_identity_fields(record, contracts, whitelist)


def _enforce_fact_source_binding_integrity(
    record: Dict[str, Any],
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    interpretation: ContractInterpretation
) -> None:
    """
    功能：执行事实源绑定完整性校验。

    Enforce fact source binding integrity checks.

    Args:
        record: Record dict to validate.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        interpretation: ContractInterpretation instance.

    Raises:
        ContractVersionMismatchError: If version mismatch.
        DigestMismatchError: If digest mismatch.
        TypeError: If inputs are invalid.
    """
    _ = interpretation
    _validate_contract_fields(record, contracts)
    _validate_whitelist_fields(record, whitelist)
    _validate_semantics_fields(record, semantics)
    
    # 版本一致性检查：record 中的锚定版本必须与各自事实源当前版本一致（去除硬编码）。
    # 动态获取当前加载的事实源版本，而不是硬编码为 v1.0。
    expected_versions = {
        "contract_version": contracts.contract_version,
        "whitelist_version": whitelist.whitelist_version,
        "policy_path_semantics_version": semantics.policy_path_semantics_version
    }
    
    for version_field, expected_value in expected_versions.items():
        record_value = record.get(version_field)
        if record_value != expected_value:
            raise ContractVersionMismatchError(
                f"{version_field} must be {expected_value}, got {record_value}"
            )


def _enforce_cfg_digest_computation_order(
    record: Dict[str, Any],
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    interpretation: ContractInterpretation
) -> None:
    """
    功能：执行 cfg_digest 存在性校验。

    Enforce cfg_digest presence for computation order requirement.

    Args:
        record: Record dict to validate.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        interpretation: ContractInterpretation instance.

    Raises:
        MissingRequiredFieldError: If cfg_digest is missing.
        TypeError: If inputs are invalid.
    """
    _ = contracts
    _ = whitelist
    _ = semantics
    _ = interpretation
    cfg_digest = record.get("cfg_digest")
    if not isinstance(cfg_digest, str) or not cfg_digest:
        # cfg_digest 缺失或类型不合法，必须 fail-fast。
        raise MissingRequiredFieldError("Missing required field: cfg_digest")

    override_applied = record.get("override_applied")
    if override_applied is None:
        return
    if not isinstance(override_applied, dict):
        raise MissingRequiredFieldError("override_applied must be dict when present")
    requested = override_applied.get("requested_overrides", [])
    if not isinstance(requested, list):
        raise MissingRequiredFieldError("override_applied.requested_overrides must be list")
    applied = override_applied.get("applied_fields", [])
    if requested and not isinstance(applied, list):
        raise MissingRequiredFieldError("override_applied.applied_fields must be list")
    if requested and not applied:
        raise MissingRequiredFieldError("override_applied.applied_fields required when overrides provided")


def _validate_contract_fields(
    record: Dict[str, Any],
    contracts: FrozenContracts
) -> None:
    """
    功能：校验 record 中的 contract 字段。
    
    Validate contract_version and digests in record match loaded contracts.
    
    Args:
        record: Record dict.
        contracts: Loaded FrozenContracts.
    
    Raises:
        ContractVersionMismatchError: If version mismatch.
        DigestMismatchError: If digest mismatch.
    """
    # contract_version
    record_version = record.get("contract_version")
    if record_version != contracts.contract_version:
        raise ContractVersionMismatchError(
            f"contract_version mismatch: record={record_version}, loaded={contracts.contract_version}"
        )
    
    # contract_digest
    _assert_digest_match(
        record, "contract_digest", contracts.contract_digest,
        "frozen_contracts.yaml semantic digest"
    )
    
    # contract_file_sha256
    _assert_digest_match(
        record, "contract_file_sha256", contracts.contract_file_sha256,
        "frozen_contracts.yaml file SHA256"
    )
    
    # contract_canon_sha256
    _assert_digest_match(
        record, "contract_canon_sha256", contracts.contract_canon_sha256,
        "frozen_contracts.yaml canonical SHA256"
    )
    
    # contract_bound_digest
    _assert_digest_match(
        record, "contract_bound_digest", contracts.contract_bound_digest,
        "frozen_contracts.yaml bound digest"
    )


def _validate_whitelist_fields(
    record: Dict[str, Any],
    whitelist: RuntimeWhitelist
) -> None:
    """
    功能：校验 record 中的 whitelist 字段。
    
    Validate whitelist_version and digests in record match loaded whitelist.
    
    Args:
        record: Record dict.
        whitelist: Loaded RuntimeWhitelist.
    
    Raises:
        ContractVersionMismatchError: If version mismatch.
        DigestMismatchError: If digest mismatch.
    """
    # whitelist_version
    record_version = record.get("whitelist_version")
    if record_version != whitelist.whitelist_version:
        raise ContractVersionMismatchError(
            f"whitelist_version mismatch: record={record_version}, loaded={whitelist.whitelist_version}"
        )
    
    # whitelist_digest
    _assert_digest_match(
        record, "whitelist_digest", whitelist.whitelist_digest,
        "runtime_whitelist.yaml semantic digest"
    )
    
    # whitelist_file_sha256
    _assert_digest_match(
        record, "whitelist_file_sha256", whitelist.whitelist_file_sha256,
        "runtime_whitelist.yaml file SHA256"
    )
    
    # whitelist_canon_sha256
    _assert_digest_match(
        record, "whitelist_canon_sha256", whitelist.whitelist_canon_sha256,
        "runtime_whitelist.yaml canonical SHA256"
    )
    
    # whitelist_bound_digest
    _assert_digest_match(
        record, "whitelist_bound_digest", whitelist.whitelist_bound_digest,
        "runtime_whitelist.yaml bound digest"
    )


def _validate_semantics_fields(
    record: Dict[str, Any],
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：校验 record 中的 semantics 字段。
    
    Validate policy_path_semantics_version and digests in record match loaded semantics.
    
    Args:
        record: Record dict.
        semantics: Loaded PolicyPathSemantics.
    
    Raises:
        ContractVersionMismatchError: If version mismatch.
        DigestMismatchError: If digest mismatch.
    """
    # policy_path_semantics_version
    record_version = record.get("policy_path_semantics_version")
    if record_version != semantics.policy_path_semantics_version:
        raise ContractVersionMismatchError(
            f"policy_path_semantics_version mismatch: record={record_version}, loaded={semantics.policy_path_semantics_version}"
        )
    
    # policy_path_semantics_digest
    _assert_digest_match(
        record, "policy_path_semantics_digest", semantics.policy_path_semantics_digest,
        "policy_path_semantics.yaml semantic digest"
    )
    
    # policy_path_semantics_file_sha256
    _assert_digest_match(
        record, "policy_path_semantics_file_sha256", semantics.policy_path_semantics_file_sha256,
        "policy_path_semantics.yaml file SHA256"
    )
    
    # policy_path_semantics_canon_sha256
    _assert_digest_match(
        record, "policy_path_semantics_canon_sha256", semantics.policy_path_semantics_canon_sha256,
        "policy_path_semantics.yaml canonical SHA256"
    )
    
    # policy_path_semantics_bound_digest
    _assert_digest_match(
        record, "policy_path_semantics_bound_digest", semantics.policy_path_semantics_bound_digest,
        "policy_path_semantics.yaml bound digest"
    )


def _assert_digest_match(
    record: Dict[str, Any],
    field_name: str,
    expected_value: str,
    description: str
) -> None:
    """
    功能：断言 record 中的 digest 字段与期望值匹配。
    
    Assert that digest field in record matches expected value.
    
    Args:
        record: Record dict.
        field_name: Field name to check.
        expected_value: Expected digest value.
        description: Human-readable description for error message.
    
    Raises:
        DigestMismatchError: If value mismatch.
    """
    actual_value = record.get(field_name)
    if actual_value != expected_value:
        raise DigestMismatchError(
            f"{description} mismatch in field '{field_name}': "
            f"record={actual_value}, expected={expected_value}"
        )


def _get_value_by_field_path(
    record: Dict[str, Any],
    field_path: str,
    domain: str
) -> Any:
    """
    功能：按点路径读取 record 字段值。
    
    Read a nested record value by dotted field path.
    
    Args:
        record: Record dict to read.
        field_path: Dotted field path.
        domain: Impl domain name for error context.
    
    Returns:
        Field value.
    
    Raises:
        MissingRequiredFieldError: If any path segment is missing.
        TypeError: If record is not a dict.
        ValueError: If field_path or domain is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if not isinstance(domain, str) or not domain:
        # domain 输入不合法，必须 fail-fast。
        raise ValueError("domain must be non-empty str")

    current: Any = record
    for segment in field_path.split("."):
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict) or segment not in current:
            # 缺失必需字段，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required field for impl_identity: domain={domain}, field_path={field_path}"
            )
        current = current[segment]
    return current


def _validate_impl_identity_fields(
    record: Dict[str, Any],
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist
) -> None:
    """
    功能：校验 impl_identity 字段绑定。
    
    Validate impl identity fields against per-domain allowlists.
    
    Args:
        record: Record dict.
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
    
    Raises:
        MissingRequiredFieldError: If required impl fields are missing or invalid.
        DigestMismatchError: If impl_id is not allowed by domain.
        GateEnforcementError: If impl contains non-whitelisted keys.
    """
    interpretation = get_contract_interpretation(contracts)
    impl_identity = interpretation.impl_identity
    if not impl_identity.required:
        # impl_identity.required 冻结为 true，必须 fail-fast。
        raise MissingRequiredFieldError(
            "impl_identity.required must be true at <root>.impl_identity.required"
        )

    impl_config = whitelist.data.get("impl_id", {})
    allowed_by_domain = impl_config.get("allowed_by_domain", {})
    allowed_flat = impl_config.get("allowed_flat", [])

    if not isinstance(allowed_by_domain, dict):
        # allowed_by_domain 类型不合法，必须 fail-fast。
        raise MissingRequiredFieldError("runtime_whitelist.impl_id.allowed_by_domain must be mapping")
    if not isinstance(allowed_flat, list):
        # allowed_flat 类型不合法，必须 fail-fast。
        raise MissingRequiredFieldError("runtime_whitelist.impl_id.allowed_flat must be list")

    # (1) 检查 impl_id 值是否属于对应 domain 的允许集合。
    for domain, field_path in impl_identity.field_paths_by_domain.items():
        impl_id = _get_value_by_field_path(record, field_path, domain)
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_id for domain={domain}, field_path={field_path}"
            )
        allowed_list = allowed_by_domain.get(domain, [])
        if not isinstance(allowed_list, list):
            # allowed_by_domain 条目类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"impl_id.allowed_by_domain[{domain}] must be list"
            )
        if impl_id not in allowed_list:
            raise DigestMismatchError(
                "impl_id not allowed for domain="
                f"{domain}, field_path={field_path}, value={impl_id}, "
                f"allowed_by_domain={allowed_list}, allowed_flat={allowed_flat}"
            )
    
    # (2) 检查 impl.versions.* 字段：必须为非空字符串，且必须匹配正则 ^[A-Za-z0-9_.-]{1,64}$。
    import re
    version_regex = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
    for domain, version_field_path in impl_identity.version_field_paths_by_domain.items():
        version_value = _get_value_by_field_path(record, version_field_path, domain)
        if not isinstance(version_value, str) or not version_value:
            raise MissingRequiredFieldError(
                f"impl version must be non-empty str: domain={domain}, field_path={version_field_path}"
            )
        if not version_regex.match(version_value):
            raise GateEnforcementError(
                f"impl version format invalid: domain={domain}, value={version_value}, "
                f"expected ^[A-Za-z0-9_.-]{{1,64}}$"
            )
    
    # (3) 检查 impl.digests.* 字段：必须为 64 位小写十六进制字符串。
    hex64_regex = re.compile(r"^[0-9a-f]{64}$")
    for domain, digest_field_path in impl_identity.digest_field_paths_by_domain.items():
        digest_value = _get_value_by_field_path(record, digest_field_path, domain)
        if not isinstance(digest_value, str) or not digest_value:
            raise MissingRequiredFieldError(
                f"impl digest must be non-empty str: domain={domain}, field_path={digest_field_path}"
            )
        if not hex64_regex.match(digest_value):
            raise GateEnforcementError(
                f"impl digest format invalid: domain={domain}, value={digest_value}, "
                f"expected 64-char lowercase hex"
            )
    
    # (4) 检查 impl 对象是否包含非白名单键。
    # 白名单键集合：impl.*_id 各域字段名 + versions + digests。
    impl_obj = record.get("impl")
    if not isinstance(impl_obj, dict):
        raise MissingRequiredFieldError("impl must be mapping")
    
    allowed_top_level_keys = set()
    for domain, field_path in impl_identity.field_paths_by_domain.items():
        # field_path 格式为 "impl.xxx_id"，提取 xxx_id。
        parts = field_path.split(".")
        if len(parts) >= 2 and parts[0] == "impl":
            allowed_top_level_keys.add(parts[1])
    allowed_top_level_keys.add("versions")
    allowed_top_level_keys.add("digests")
    
    extra_keys = set(impl_obj.keys()) - allowed_top_level_keys
    if extra_keys:
        raise GateEnforcementError(
            f"impl contains non-whitelisted keys: {sorted(extra_keys)}. "
            f"Allowed keys: {sorted(allowed_top_level_keys)}. "
            f"This prevents injection of environment-related fields like build_time, cwd, machine, pythonpath."
        )


def _validate_statistical_fields(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
    warn_mode: bool = False
) -> None:
    """
    功能：统计口径规范化校验。
    
    Validate statistical fields completeness and consistency.
    
    Args:
        record: Record dict to validate.
        interpretation: Contract interpretation.
        warn_mode: If True, print warnings instead of raising errors.
    
    Raises:
        GateEnforcementError: If validation fails and warn_mode is False.
    """
    operation = record.get("operation")
    if operation not in {"calibrate", "detect", "evaluate"}:
        return
    
    # 统计类记录必需字段
    required_stats_fields = [
        "target_fpr",
        "threshold_source",
        "stats_applicability"
    ]
    
    missing_fields = [f for f in required_stats_fields if f not in record]
    if missing_fields:
        msg = (
            f"[统计口径冻结] Statistical record missing required fields: "
            f"operation={operation}, missing={missing_fields}"
        )
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="statistical.record.required_fields.completeness",
                field_path="stats_applicability"
            )
    
    # threshold_source 必须为 'np_canonical'。
    # 例外：若 fusion_result.audit.allow_threshold_fallback_for_tests=True，
    # 则该记录为 calibrate 前的中间 detect 步骤（onefile 主流程合法场景），
    # 降级为警告不阻断，以允许后续 calibrate → re-detect 流程继续。
    threshold_source = record.get("threshold_source")
    if threshold_source != "np_canonical":
        _fallback_authorized = False
        _fusion_audit = None
        _fr = record.get("fusion_result")
        if isinstance(_fr, dict):
            _fusion_audit = _fr.get("audit")
            if isinstance(_fusion_audit, dict):
                _fallback_authorized = bool(_fusion_audit.get("allow_threshold_fallback_for_tests", False))
        _effective_warn = warn_mode or _fallback_authorized
        msg = (
            f"[统计口径冻结] threshold_source must be 'np_canonical': "
            f"operation={operation}, actual={threshold_source}"
        )
        if _effective_warn:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="statistical.threshold.source.np_canonical",
                field_path="threshold_source"
            )
    
    # stats_applicability 必须为 'applicable'
    stats_applicability = record.get("stats_applicability")
    if stats_applicability != "applicable":
        msg = (
            f"[统计口径冻结] stats_applicability must be 'applicable' for statistical records: "
            f"operation={operation}, actual={stats_applicability}"
        )
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="statistical.applicability.must_be_applicable",
                field_path="stats_applicability"
            )

    required_fields = set(interpretation.required_record_fields)
    record_target_fpr = record.get("target_fpr")
    if not isinstance(record_target_fpr, (int, float)):
        msg = f"[统计口径冻结] target_fpr must be number: actual={record_target_fpr}"
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="statistical.target_fpr.type_check",
                field_path="target_fpr"
            )

    thresholds_spec = schema.build_thresholds_spec({"target_fpr": float(record_target_fpr)})
    thresholds_digest = schema.compute_thresholds_digest(thresholds_spec)
    if "thresholds_digest" in required_fields or "thresholds_digest" in record:
        record_digest = record.get("thresholds_digest")
        if not isinstance(record_digest, str) or not record_digest:
            msg = "[统计口径冻结] thresholds_digest missing or invalid"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="statistical.thresholds_digest.presence",
                    field_path="thresholds_digest"
                )
        elif record_digest != thresholds_digest:
            msg = "[统计口径冻结] thresholds_digest mismatch"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="statistical.thresholds_digest.consistency",
                    field_path="thresholds_digest",
                    expected=thresholds_digest,
                    actual=record_digest
                )

    if "threshold_metadata_digest" in required_fields or "threshold_metadata_digest" in record:
        from main.watermarking.fusion import neyman_pearson

        threshold_metadata_artifact = record.get("threshold_metadata_artifact")
        if isinstance(threshold_metadata_artifact, dict):
            threshold_metadata = threshold_metadata_artifact
        else:
            threshold_metadata = neyman_pearson.build_threshold_metadata(thresholds_spec)
        threshold_metadata_digest = neyman_pearson.compute_threshold_metadata_digest(threshold_metadata)
        record_metadata_digest = record.get("threshold_metadata_digest")
        if not isinstance(record_metadata_digest, str) or not record_metadata_digest:
            msg = "[统计口径冻结] threshold_metadata_digest missing or invalid"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="statistical.threshold_metadata_digest.presence",
                    field_path="threshold_metadata_digest"
                )
        elif record_metadata_digest != threshold_metadata_digest:
            msg = "[统计口径冻结] threshold_metadata_digest mismatch"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="statistical.threshold_metadata_digest.consistency",
                    field_path="threshold_metadata_digest",
                    expected=threshold_metadata_digest,
                    actual=record_metadata_digest
                )


def _validate_semantic_driven_execution(
    record: Dict[str, Any],
    semantics: PolicyPathSemantics,
    interpretation: ContractInterpretation,
    warn_mode: bool = False
) -> None:
    """
    功能：语义驱动执行校验。
    
    Validate execution_report structure and decision field.
    
    Args:
        record: Record dict to validate.
        semantics: PolicyPathSemantics.
        interpretation: Contract interpretation.
        warn_mode: If True, print warnings instead of raising errors.
    
    Raises:
        GateEnforcementError: If validation fails and warn_mode is False.
    """
    # execution_report 结构完整性与类型检查
    execution_report = record.get("execution_report")
    if execution_report is not None:
        required_fields = [
            "content_chain_status",
            "geometry_chain_status",
            "fusion_status",
            "audit_obligations_satisfied"
        ]
        if not isinstance(execution_report, dict):
            msg = f"[执行语义门禁] execution_report must be dict, got {type(execution_report)}"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="execution_report.structure.type_check",
                    field_path="execution_report"
                )
        else:
            missing_fields = [f for f in required_fields if f not in execution_report]
            if missing_fields:
                msg = f"[执行语义门禁] execution_report missing fields: {missing_fields}"
                if warn_mode:
                    print(f"[FreezeGate][WARN] {msg}")
                else:
                    raise GateEnforcementError(
                        msg,
                        gate_name="execution_report.required_fields.completeness",
                        field_path="execution_report"
                    )
            else:
                status_fields = [
                    "content_chain_status",
                    "geometry_chain_status",
                    "fusion_status"
                ]
                for field_name in status_fields:
                    value = execution_report.get(field_name)
                    try:
                        _normalize_execution_report_chain_status(value, f"execution_report.{field_name}")
                    except ValueError as exc:
                        msg = (
                            f"[执行语义门禁] {exc}"
                        )
                        if warn_mode:
                            print(f"[FreezeGate][WARN] {msg}")
                        else:
                            raise GateEnforcementError(
                                msg,
                                gate_name="execution_report.fields.enum_validation",
                                field_path=f"execution_report.{field_name}"
                            )
                audit_value = execution_report.get("audit_obligations_satisfied")
                if not isinstance(audit_value, bool):
                    msg = "[执行语义门禁] execution_report.audit_obligations_satisfied must be bool"
                    if warn_mode:
                        print(f"[FreezeGate][WARN] {msg}")
                    else:
                        raise GateEnforcementError(
                            msg,
                            gate_name="execution_report.fields.enum_validation",
                            field_path="execution_report.audit_obligations_satisfied"
                        )
    
    policy_path = record.get("policy_path")
    if not isinstance(policy_path, str) or not policy_path:
        msg = "[执行语义门禁] policy_path missing or invalid"
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(msg, gate_name="policy_path.presence", field_path="policy_path")

    policy_paths = semantics.data.get("policy_paths", {}) if isinstance(semantics.data, dict) else {}
    policy_spec = policy_paths.get(policy_path) if isinstance(policy_paths, dict) else None
    if not isinstance(policy_spec, dict):
        msg = f"[执行语义门禁] policy_path semantics missing: policy_path={policy_path}"
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(msg, gate_name="policy_path.semantics.missing", field_path="policy_path")

    required_chains = policy_spec.get("required_chains", {})
    on_chain_failure = policy_spec.get("on_chain_failure", {})
    if not isinstance(required_chains, dict) or not isinstance(on_chain_failure, dict):
        msg = f"[执行语义门禁] policy_path semantics malformed: policy_path={policy_path}"
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="policy_path.semantics.malformed",
                field_path="policy_path"
            )

    chain_status_fields = {
        "content": "content_chain_status",
        "geometry": "geometry_chain_status"
    }
    for chain_name, required in required_chains.items():
        if not required:
            continue
        status_field = chain_status_fields.get(chain_name)
        if status_field is None:
            continue
        if not isinstance(execution_report, dict):
            msg = f"[执行语义门禁] execution_report required for chain={chain_name}"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="execution_report.required_for_chain",
                    field_path="execution_report"
                )
        status_value = execution_report.get(status_field)
        if status_value != "ok":
            chain_policy = on_chain_failure.get(chain_name)
            if not isinstance(chain_policy, dict):
                msg = f"[执行语义门禁] on_chain_failure missing: chain={chain_name}"
                if warn_mode:
                    print(f"[FreezeGate][WARN] {msg}")
                else:
                    raise GateEnforcementError(
                        msg,
                        gate_name="on_chain_failure.presence",
                        field_path=f"policy_path_semantics.on_chain_failure.{chain_name}"
                    )
            action = chain_policy.get("action")
            set_decision_to = chain_policy.get("set_decision_to")
            if not isinstance(action, str) or not action:
                msg = f"[执行语义门禁] on_chain_failure.action missing: chain={chain_name}"
                if warn_mode:
                    print(f"[FreezeGate][WARN] {msg}")
                else:
                    raise GateEnforcementError(
                        msg,
                        gate_name="on_chain_failure.action.validation",
                        field_path=f"policy_path_semantics.on_chain_failure.{chain_name}.action"
                    )
            if action == "not_applicable":
                msg = f"[执行语义门禁] on_chain_failure.action not_applicable for required chain: chain={chain_name}"
                if warn_mode:
                    print(f"[FreezeGate][WARN] {msg}")
                else:
                    raise GateEnforcementError(
                        msg,
                        gate_name="on_chain_failure.action.validation",
                        field_path=f"policy_path_semantics.on_chain_failure.{chain_name}.action"
                    )
            decision_field_path = interpretation.records_schema.decision_field_path
            _, decision_value = _get_value_by_path(record, decision_field_path)
            if set_decision_to is None:
                if decision_value is not None:
                    msg = f"[执行语义门禁] decision must be null on chain failure: chain={chain_name}"
                    if warn_mode:
                        print(f"[FreezeGate][WARN] {msg}")
                    else:
                        raise GateEnforcementError(
                            msg,
                            gate_name="on_chain_failure.decision.invariants",
                            field_path=decision_field_path
                        )
            elif isinstance(set_decision_to, bool):
                if decision_value != set_decision_to:
                    msg = f"[执行语义门禁] decision mismatch on chain failure: chain={chain_name}"
                    if warn_mode:
                        print(f"[FreezeGate][WARN] {msg}")
                    else:
                        raise GateEnforcementError(
                            msg,
                            gate_name="on_chain_failure.decision.invariants",
                            field_path=decision_field_path,
                            expected=str(set_decision_to),
                            actual=str(decision_value)
                        )
            else:
                msg = f"[执行语义门禁] on_chain_failure.set_decision_to invalid: chain={chain_name}"
                if warn_mode:
                    print(f"[FreezeGate][WARN] {msg}")
                else:
                    raise GateEnforcementError(
                        msg,
                        gate_name="on_chain_failure.decision.invariants",
                        field_path=f"policy_path_semantics.on_chain_failure.{chain_name}.set_decision_to"
                    )

    decision_field_path = interpretation.records_schema.decision_field_path
    decision_obligation = interpretation.records_schema.decision_obligation_presence
    decision_allow_null = interpretation.records_schema.decision_allow_null
    found, value = _get_value_by_path(record, decision_field_path)

    if decision_obligation and not found:
        msg = f"[执行语义门禁] decision field missing: field_path={decision_field_path}"
        if warn_mode:
            print(f"[FreezeGate][WARN] {msg}")
        else:
            raise GateEnforcementError(
                msg,
                gate_name="decision.presence",
                field_path=decision_field_path
            )
    if found:
        if value is None and not decision_allow_null:
            msg = f"[执行语义门禁] decision field must not be null: field_path={decision_field_path}"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="decision.not_null",
                    field_path=decision_field_path
                )
        if value is not None and not isinstance(value, bool):
            msg = f"[执行语义门禁] decision field type mismatch: field_path={decision_field_path}"
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="decision.type_check",
                    field_path=decision_field_path
                )


def _normalize_execution_report_chain_status(raw_status: Any, field_path: str) -> str:
    """
    功能：归一化 execution_report 链路状态并阻断非规范值。

    Normalize execution_report chain status and reject non-canonical variants.

    Args:
        raw_status: Raw status token.
        field_path: Field path for error context.

    Returns:
        Canonical status token in {ok, failed, absent}.

    Raises:
        TypeError: If field_path is invalid.
        ValueError: If status token is missing or non-canonical.
    """
    if not isinstance(field_path, str) or not field_path:
        raise TypeError("field_path must be non-empty str")
    if not isinstance(raw_status, str) or not raw_status:
        raise ValueError(f"{field_path} must be non-empty str in {{'ok','failed','absent'}}")

    normalized = raw_status.strip().lower()
    if normalized == "fail":
        raise ValueError(f"{field_path} uses deprecated status 'fail'; use 'failed'")
    if normalized not in {"ok", "failed", "absent"}:
        raise ValueError(f"{field_path} must be one of ['absent', 'failed', 'ok']")
    return normalized


def _check_field_override_conflict(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
    warn_mode: bool = False
) -> None:
    """
    功能：受控字段门禁命令侧覆盖防御检测。
    
    Detect field override conflicts between orchestrator and schema injection.
    
    Args:
        record: Record dict to validate.
        interpretation: Contract interpretation.
        warn_mode: If True, print warnings instead of raising errors.
    
    Raises:
        GateEnforcementError: If conflict detected and warn_mode is False.
    """
    operation = record.get("operation")
    if operation not in {"calibrate", "detect", "evaluate"}:
        return

    expected_values = {
        "stats_applicability": "applicable",
        "threshold_source": "np_canonical",
        "schema_version": schema.RECORD_SCHEMA_VERSION
    }

    for field_name, expected_value in expected_values.items():
        actual_value = record.get(field_name)
        if actual_value is not None and actual_value != expected_value:
            msg = (
                f"[受控字段门禁] Detected potential override conflict: {field_name}={actual_value} "
                f"(expected '{expected_value}' from schema injection)"
            )
            if warn_mode:
                print(f"[FreezeGate][WARN] {msg}")
            else:
                raise GateEnforcementError(
                    msg,
                    gate_name="schema.controlled_fields.override_conflict",
                    field_path=field_name,
                    expected=str(expected_value),
                    actual=str(actual_value)
                )


def _enforce_policy_semantics_whitelist_version_binding(
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：强制执行 policy_path_semantics 与 runtime_whitelist 版本强绑定。
    
    Enforce that policy_path_semantics.version == runtime_whitelist.version.
    If versions mismatch, raise GateEnforcementError with field paths and values.
    
    Args:
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
    
    Returns:
        None.
    
    Raises:
        GateEnforcementError: If versions mismatch.
        TypeError: If inputs are invalid.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        raise TypeError(f"whitelist must be RuntimeWhitelist, got {type(whitelist)}")
    if not isinstance(semantics, PolicyPathSemantics):
        raise TypeError(f"semantics must be PolicyPathSemantics, got {type(semantics)}")
    
    whitelist_version = whitelist.whitelist_version
    semantics_version = semantics.policy_path_semantics_version
    
    if whitelist_version != semantics_version:
        raise GateEnforcementError(
            f"policy_path_semantics and runtime_whitelist versions must match: "
            f"semantics.policy_path_semantics_version={semantics_version} "
            f"!= whitelist.whitelist_version={whitelist_version}"
        )


def _check_enumeration_whitelist_authority(
    record: Dict[str, Any],
    whitelist: RuntimeWhitelist
) -> None:
    """
    功能：验证 record 中的枚举值不包含硬编码内容，需要遵循运行时白名单
    
    Verify that enumeration values in record comply with runtime whitelist authority.
    Ensures no hardcoded enumeration values bypass the whitelist configuration.
    
    Args:
        record: The record object to validate.
        whitelist: The RuntimeWhitelist instance defining allowed enumerations.
    
    Raises:
        GateEnforcementError: If enumeration values violate whitelist authority.
    
    Notes:
        This check enforces that enumeration-based fields must derive from
        runtime_whitelist.yaml, preventing bypass via hardcoded values.
    """
    # (1) 提取 record 中的 enumeration 字段
    if "enumeration_fields" not in record:
        # 如果没有 enumeration_fields，检查通过（不适用）
        return
    
    enum_fields = record.get("enumeration_fields", {})
    if not isinstance(enum_fields, dict):
        raise GateEnforcementError(
            f"enumeration_fields must be dict, got {type(enum_fields).__name__}"
        )
    
    # (2) 对比 whitelist 的允许值 (whitelist.allowed_categories, whitelist.allowed_tags 等)
    allowed_categories = getattr(whitelist, "allowed_categories", set())
    allowed_tags = getattr(whitelist, "allowed_tags", set())
    
    # (3) 验证每个 enumeration_fields 的值在允许列表中
    for field_name, field_value in enum_fields.items():
        if field_name == "category" and allowed_categories:
            if field_value not in allowed_categories:
                raise GateEnforcementError(
                    f"enumeration category '{field_value}' not in whitelist allowed_categories: "
                    f"{allowed_categories}"
                )
        elif field_name == "tag" and allowed_tags:
            if field_value not in allowed_tags:
                raise GateEnforcementError(
                    f"enumeration tag '{field_value}' not in whitelist allowed_tags: "
                    f"{allowed_tags}"
                )


def _check_args_digest_factor_separation(record: Dict[str, Any]) -> None:
    """
    功能：验证 args_digest 正确分离方法因子和 I/O 因子
    
    Verify that args_digest factors are properly separated into method factors
    and I/O factors. Method factors include impl_id, cfg_digest, thresholds_digest.
    I/O factors include output_path, device, dtype.
    
    Args:
        record: The record object to validate.
    
    Raises:
        GateEnforcementError: If factor separation is not correct.
    
    Notes:
        Method factors are deterministic and reproducible within the algorithm.
        I/O factors are deployment-specific and should not pollute args_digest.
    """
    # (1) 检查 record 中是否存在 args_digest 和相关因子信息
    if "args_digest" not in record or "digest_factors" not in record:
        # 不适用（可能是早期版本记录）
        return
    
    digest_factors = record.get("digest_factors", {})
    if not isinstance(digest_factors, dict):
        raise GateEnforcementError(
            f"digest_factors must be dict, got {type(digest_factors).__name__}"
        )
    
    # (2) 定义方法因子和 I/O 因子
    method_factors = {"impl_id", "cfg_digest", "thresholds_digest"}
    io_factors = {"output_path", "device", "dtype"}
    
    # (3) 验证 args_digest 中的因子包含所有方法因子
    included_factors = set(digest_factors.keys())
    
    # 检查方法因子：必须包含
    missing_method_factors = method_factors - included_factors
    if missing_method_factors:
        raise GateEnforcementError(
            f"args_digest missing required method factors: {missing_method_factors}"
        )
    
    # 检查 I/O 因子：不应该包含
    included_io_factors = io_factors & included_factors
    if included_io_factors:
        raise GateEnforcementError(
            f"args_digest incorrectly includes I/O factors: {included_io_factors}; "
            f"these should be excluded from digest computation"
        )
