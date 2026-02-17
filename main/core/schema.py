"""
records 与 run_closure schema 常量与校验

功能说明：
- 定义了 run_closure 和 records bundle 相关的 schema 常量，如字段名称、版本号等。
- 用于校验 run_closure payload 和 record 结构与语义的一致性。
- 包含详细的输入验证和错误处理，确保健壮性和可维护性。
- 校验 record 的权威 schema 约束。
"""

from __future__ import annotations

from typing import Any, Dict, List
from main.core import time_utils
from main.core.contracts import ContractInterpretation
from main.core.errors import (
    MissingRequiredFieldError,
    RunFailureReason,
    ContractInterpretationRequiredError
)


RUN_CLOSURE_SCHEMA_VERSION = "v1.0"
RECORD_SCHEMA_VERSION = "v1.0"
RECORDS_MANIFEST_NAME = "records_manifest.json"
RUN_CLOSURE_NAME = "run_closure.json"

PATH_AUDIT_STATUS_ALLOWED = {"ok", "failed"}
PATH_AUDIT_ERROR_CODE_ALLOWED = {
    "fact_sources_unbound",
    "missing_bound_fields",
    "build_exception",
    "write_blocked",
    "write_failed"
}

# 硬编码的可选字段列表（向后兼容，不再直接使用）
# 这些列表现在由 records_schema_extensions.yaml 定义，通过 ContractInterpretation 传递
# 仅在 interpretation 不可用时作为后备
_FALLBACK_OPTIONAL_STR_FIELDS = [
    "content_evidence.mask_digest",
    "content_evidence.plan_digest",
    "content_evidence.basis_digest",
    "content_evidence.lf_trace_digest",
    "content_evidence.hf_trace_digest",
    "content_evidence.content_failure_reason",
    "geometry_evidence.anchor_digest",
    "geometry_evidence.sync_digest",
    "geometry_evidence.align_trace_digest",
    "geometry_evidence.geo_failure_reason",
    "decision.fusion_rule_version",
    "decision.used_threshold_id",
    "decision.routing_digest"
]

_FALLBACK_OPTIONAL_NUMBER_FIELDS = [
    "content_evidence.lf_score",
    "content_evidence.hf_score",
    "geometry_evidence.geo_score"
]

_FALLBACK_OPTIONAL_MAPPING_FIELDS = [
    "content_evidence.mask_stats",
    "content_evidence.score_parts",
    "geometry_evidence.anchor_metrics",
    "geometry_evidence.sync_metrics",
    "decision.routing_decisions",
    "decision.conditional_fpr_notes"
]


def get_record_type(record: Dict[str, Any]) -> str:
    """
    功能：读取 record_type 字段。

    Read record_type field from record.

    Args:
        record: Record mapping.

    Returns:
        record_type string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "record_type")


def get_sample_id(record: Dict[str, Any]) -> str:
    """
    功能：读取 sample_id 字段。

    Read sample_id field from record.

    Args:
        record: Record mapping.

    Returns:
        sample_id string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "sample_id")


def get_cfg_digest(record: Dict[str, Any]) -> str:
    """
    功能：读取 cfg_digest 字段。

    Read cfg_digest field from record.

    Args:
        record: Record mapping.

    Returns:
        cfg_digest string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "cfg_digest")


def get_policy_path(record: Dict[str, Any]) -> str:
    """
    功能：读取 policy_path 字段。

    Read policy_path field from record.

    Args:
        record: Record mapping.

    Returns:
        policy_path string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "policy_path")


def get_threshold_key_used(record: Dict[str, Any]) -> str:
    """
    功能：读取 threshold_key_used 字段。

    Read threshold_key_used field from record.

    Args:
        record: Record mapping.

    Returns:
        threshold_key_used string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "threshold_key_used")


def get_thresholds_digest(record: Dict[str, Any]) -> str:
    """
    功能：读取 thresholds_digest 字段。

    Read thresholds_digest field from record.

    Args:
        record: Record mapping.

    Returns:
        thresholds_digest string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "thresholds_digest")


def get_threshold_metadata_digest(record: Dict[str, Any]) -> str:
    """
    功能：读取 threshold_metadata_digest 字段。

    Read threshold_metadata_digest field from record.

    Args:
        record: Record mapping.

    Returns:
        threshold_metadata_digest string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "threshold_metadata_digest")


def get_subspace_frame_in_plan(record: Dict[str, Any]) -> str:
    """
    功能：读取 subspace_plan.subspace_frame 字段。

    Read subspace_frame from subspace_plan.

    Args:
        record: Record mapping.

    Returns:
        subspace_frame string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "subspace_plan.subspace_frame")


def get_subspace_frame_used(record: Dict[str, Any]) -> str:
    """
    功能：读取 subspace_frame_used 字段。

    Read subspace_frame_used field from record.

    Args:
        record: Record mapping.

    Returns:
        subspace_frame_used string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "subspace_frame_used")


def get_frame_diff(record: Dict[str, Any]) -> float:
    """
    功能：读取 frame_diff 字段。

    Read frame_diff field from record.

    Args:
        record: Record mapping.

    Returns:
        frame_diff numeric value.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    value = _require_field(record, "frame_diff")
    if not isinstance(value, (int, float)):
        raise TypeError("Type mismatch at frame_diff: expected number")
    return float(value)


def get_contract_version(record: Dict[str, Any]) -> str:
    """
    功能：读取 contract_version 字段。

    Read contract_version field from record.

    Args:
        record: Record mapping.

    Returns:
        contract_version string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "contract_version")


def get_impl_id(record: Dict[str, Any]) -> str:
    """
    功能：读取 impl.content_extractor_id 字段。

    Read impl.content_extractor_id field from record.

    Args:
        record: Record mapping.

    Returns:
        impl_id string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If record or value types are invalid.
    """
    return _require_str_field(record, "impl.content_extractor_id")


def require_present(value: Any, field_path: str, record_hint: Any | None = None) -> Any:
    """
    功能：断言字段存在并返回值。

    Require a field value to be present. If a record mapping is supplied
    as the first argument and record_hint is None, this function will
    resolve the field_path against the record.

    Args:
        value: Extracted value or record mapping.
        field_path: Field path for error context.
        record_hint: Optional record hint for error context.

    Returns:
        Resolved value.

    Raises:
        MissingRequiredFieldError: If value is missing.
        TypeError: If inputs are invalid.
    """
    if not isinstance(field_path, str) or not field_path:
        raise TypeError("field_path must be non-empty str")
    if record_hint is None and isinstance(value, dict):
        found, resolved = _get_value_by_field_path(value, field_path)
        if not found:
            raise MissingRequiredFieldError(f"Missing required field: {field_path}")
        return resolved
    if value is None:
        raise MissingRequiredFieldError(f"Missing required field: {field_path}")
    return value


def require_type(value: Any, expected_type: type | tuple[type, ...], field_path: str) -> Any:
    """
    功能：断言字段类型符合预期。

    Require value to be present and match expected type.

    Args:
        value: Field value or record mapping.
        expected_type: Expected type or tuple of types.
        field_path: Field path for error context.

    Returns:
        Value if type matches.

    Raises:
        MissingRequiredFieldError: If value is missing.
        TypeError: If type does not match.
    """
    if not isinstance(expected_type, (type, tuple)):
        raise TypeError("expected_type must be type or tuple of types")
    value = require_present(value, field_path)
    if not isinstance(value, expected_type):
        raise TypeError(f"Type mismatch at {field_path}: expected {expected_type}")
    return value


def validate_run_closure_payload(payload: Dict[str, Any]) -> None:
    """
    功能：校验 run_closure payload 的最小结构。

    Validate minimal structure for run_closure payload.

    Args:
        payload: Run closure payload mapping.

    Raises:
        TypeError: If payload is not a dict.
        ValueError: If required fields are missing.
    """
    if not isinstance(payload, dict):
        # payload 类型不符合预期，必须 fail-fast。
        raise TypeError("payload must be dict")

    required_keys = [
        "schema_version",
        "run_id",
        "command",
        "created_at_utc",
        "cfg_digest",
        "contract_version",
        "contract_digest",
        "contract_file_sha256",
        "contract_canon_sha256",
        "contract_bound_digest",
        "whitelist_version",
        "whitelist_digest",
        "whitelist_file_sha256",
        "whitelist_canon_sha256",
        "whitelist_bound_digest",
        "policy_path_semantics_version",
        "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256",
        "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest",
        "policy_path",
        "impl_id",
        "impl_version",
        "impl_identity",
        "impl_identity_digest",
        "facts_anchor",
        "records_bundle",
        "status"
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        # 关键字段缺失，必须 fail-fast。
        raise ValueError(f"run_closure missing fields: {missing}")


def validate_run_closure(payload: Dict[str, Any]) -> None:
    """
    功能：校验 run_closure payload 语义一致性。

    Validate run_closure payload semantics and reason enum.

    Args:
        payload: Run closure payload mapping.

    Returns:
        None.

    Raises:
        TypeError: If payload types are invalid.
        ValueError: If required fields are missing or invalid.
    """
    validate_run_closure_payload(payload)
    time_utils.validate_utc_iso_z(payload.get("created_at_utc"), "run_closure.created_at_utc")
    status = payload.get("status")
    if not isinstance(status, dict):
        # status 类型不符合预期，必须 fail-fast。
        raise TypeError("status must be dict")
    status_ok = status.get("ok")
    if not isinstance(status_ok, bool):
        # status.ok 类型不符合预期，必须 fail-fast。
        raise TypeError("status.ok must be bool")
    status_reason = status.get("reason")
    if not isinstance(status_reason, str) or not status_reason:
        # status.reason 类型不符合预期，必须 fail-fast。
        raise TypeError("status.reason must be non-empty str")
    allowed_reasons = {reason.value for reason in RunFailureReason}
    if status_reason not in allowed_reasons:
        # reason 非受控枚举成员，必须 fail-fast。
        raise ValueError("status.reason must be RunFailureReason member")
    if status_ok and status_reason != RunFailureReason.OK.value:
        # status.ok=True 时 reason 必须为 ok。
        raise ValueError("status.reason must be 'ok' when status.ok is True")
    if not status_ok and status_reason == RunFailureReason.OK.value:
        # status.ok=False 时 reason 不能为 ok。
        raise ValueError("status.reason must not be 'ok' when status.ok is False")
    status_details = status.get("details")
    if status_details is not None:
        _validate_json_like(status_details, "status.details")
        if isinstance(status_details, dict):
            upstream_reason = status_details.get("upstream_failure_reason")
            if upstream_reason is not None:
                if not isinstance(upstream_reason, str) or not upstream_reason:
                    # upstream_failure_reason 类型不符合预期，必须 fail-fast。
                    raise TypeError("status.details.upstream_failure_reason must be non-empty str")
                if upstream_reason not in allowed_reasons:
                    # upstream_failure_reason 非受控枚举成员，必须 fail-fast。
                    raise ValueError("status.details.upstream_failure_reason must be RunFailureReason member")
                if upstream_reason == RunFailureReason.OK.value:
                    # upstream_failure_reason 不能为 ok。
                    raise ValueError("status.details.upstream_failure_reason must not be 'ok'")

    if _should_require_bound_fact_sources(payload, status_ok):
        _validate_bound_fact_sources(payload)

    impl_identity = payload.get("impl_identity")
    if impl_identity is not None:
        if not isinstance(impl_identity, dict):
            # impl_identity 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch: field_path=run_closure.impl_identity expected=dict|None")
        _validate_json_like(impl_identity, "impl_identity")
    facts_anchor = payload.get("facts_anchor")
    if isinstance(facts_anchor, str):
        # facts_anchor 类型不符合预期，必须 fail-fast。
        raise TypeError("Type mismatch: field_path=run_closure.facts_anchor expected=dict|None")
    impl_identity_digest = payload.get("impl_identity_digest")
    if impl_identity_digest is not None:
        if not isinstance(impl_identity_digest, str) or not impl_identity_digest:
            # impl_identity_digest 类型不符合预期，必须 fail-fast。
            raise TypeError("impl_identity_digest must be non-empty str or None")
    if status_ok:
        _validate_run_closure_ok_fields(payload)

    path_audit_status = payload.get("path_audit_status")
    if path_audit_status is not None:
        if not isinstance(path_audit_status, str) or not path_audit_status:
            # path_audit_status 类型不符合预期，必须 fail-fast。
            raise TypeError("path_audit_status must be non-empty str or None")
        if path_audit_status not in PATH_AUDIT_STATUS_ALLOWED:
            # path_audit_status 非受控枚举成员，必须 fail-fast。
            raise ValueError("path_audit_status must be one of {'ok','failed'}")

    path_audit_error_code = payload.get("path_audit_error_code")
    if path_audit_error_code is not None:
        if not isinstance(path_audit_error_code, str) or not path_audit_error_code:
            # path_audit_error_code 类型不符合预期，必须 fail-fast。
            raise TypeError("path_audit_error_code must be non-empty str or None")
        if path_audit_error_code != "<absent>" and path_audit_error_code not in PATH_AUDIT_ERROR_CODE_ALLOWED:
            # path_audit_error_code 非受控枚举成员，必须 fail-fast。
            raise ValueError("path_audit_error_code must be in frozen allowlist")

    if path_audit_status == "ok":
        if path_audit_error_code not in {None, "<absent>"}:
            # ok 状态不应携带错误码，必须 fail-fast。
            raise ValueError("path_audit_error_code must be absent when path_audit_status is ok")
    if path_audit_status == "failed":
        if path_audit_error_code in {None, "<absent>"}:
            # failed 状态必须携带错误码，必须 fail-fast。
            raise ValueError("path_audit_error_code is required when path_audit_status is failed")

    # 验证 RNG 审计字段：rng_audit_canon_sha256 必须存在且为 str。
    rng_audit_canon_sha256 = payload.get("rng_audit_canon_sha256")
    if rng_audit_canon_sha256 is not None:
        if not isinstance(rng_audit_canon_sha256, str) or not rng_audit_canon_sha256:
            # rng_audit_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("rng_audit_canon_sha256 must be non-empty str or None")

    # 验证模型来源字段：model_provenance_canon_sha256 必须存在且为 str。
    model_provenance_canon_sha256 = payload.get("model_provenance_canon_sha256")
    if model_provenance_canon_sha256 is not None:
        if not isinstance(model_provenance_canon_sha256, str) or not model_provenance_canon_sha256:
            # model_provenance_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("model_provenance_canon_sha256 must be non-empty str or None")

    # 验证 env_lock 字段：若提供则校验类型。
    env_lock_sha256 = payload.get("env_lock_sha256")
    if env_lock_sha256 is not None:
        if not isinstance(env_lock_sha256, str) or not env_lock_sha256:
            # env_lock_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("env_lock_sha256 must be non-empty str or None")

    pipeline_provenance_canon_sha256 = payload.get("pipeline_provenance_canon_sha256")
    if pipeline_provenance_canon_sha256 is not None:
        if not isinstance(pipeline_provenance_canon_sha256, str) or not pipeline_provenance_canon_sha256:
            # pipeline_provenance_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("pipeline_provenance_canon_sha256 must be non-empty str or None")

    pipeline_status = payload.get("pipeline_status")
    if pipeline_status is not None:
        if not isinstance(pipeline_status, str) or not pipeline_status:
            # pipeline_status 类型不符合预期，必须 fail-fast。
            raise TypeError("pipeline_status must be non-empty str or None")
        allowed_pipeline_status = {"built", "failed", "unbuilt"}
        if pipeline_status not in allowed_pipeline_status:
            # pipeline_status 不在允许列表，必须 fail-fast。
            raise ValueError("pipeline_status must be one of {'built','failed','unbuilt'}")

    pipeline_error = payload.get("pipeline_error")
    if pipeline_error is not None:
        if not isinstance(pipeline_error, str) or not pipeline_error:
            # pipeline_error 类型不符合预期，必须 fail-fast。
            raise TypeError("pipeline_error must be non-empty str or None")

    pipeline_runtime_meta = payload.get("pipeline_runtime_meta")
    if pipeline_runtime_meta is not None:
        if not isinstance(pipeline_runtime_meta, dict):
            # pipeline_runtime_meta 类型不符合预期，必须 fail-fast。
            raise TypeError("pipeline_runtime_meta must be dict or None")
        _validate_json_like(pipeline_runtime_meta, "pipeline_runtime_meta")

    env_fingerprint_canon_sha256 = payload.get("env_fingerprint_canon_sha256")
    if env_fingerprint_canon_sha256 is not None:
        if not isinstance(env_fingerprint_canon_sha256, str) or not env_fingerprint_canon_sha256:
            # env_fingerprint_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("env_fingerprint_canon_sha256 must be non-empty str or None")

    diffusers_version = payload.get("diffusers_version")
    if diffusers_version is not None:
        if not isinstance(diffusers_version, str) or not diffusers_version:
            # diffusers_version 类型不符合预期，必须 fail-fast。
            raise TypeError("diffusers_version must be non-empty str or None")

    transformers_version = payload.get("transformers_version")
    if transformers_version is not None:
        if not isinstance(transformers_version, str) or not transformers_version:
            # transformers_version 类型不符合预期，必须 fail-fast。
            raise TypeError("transformers_version must be non-empty str or None")

    safetensors_version = payload.get("safetensors_version")
    if safetensors_version is not None:
        if not isinstance(safetensors_version, str) or not safetensors_version:
            # safetensors_version 类型不符合预期，必须 fail-fast。
            raise TypeError("safetensors_version must be non-empty str or None")

    # 验证可选的 closure_stage 块。若提供则必须为 dict，并校验内部字段。
    closure_stage = payload.get("closure_stage")
    if closure_stage is not None:
        if not isinstance(closure_stage, dict):
            # closure_stage 类型不符合预期，必须 fail-fast。
            raise TypeError("closure_stage must be dict or None")
        _validate_closure_stage(closure_stage)


def _require_field(record: Dict[str, Any], field_path: str) -> Any:
    """
    功能：读取并要求字段存在。

    Read a field from record by dotted path and require presence.

    Args:
        record: Record mapping.
        field_path: Dotted field path.

    Returns:
        Field value.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    found, value = _get_value_by_field_path(record, field_path)
    if not found:
        raise MissingRequiredFieldError(f"Missing required field: {field_path}")
    return value


def _require_str_field(record: Dict[str, Any], field_path: str) -> str:
    """
    功能：读取并要求字段为非空字符串。

    Read a field by dotted path and require non-empty str.

    Args:
        record: Record mapping.
        field_path: Dotted field path.

    Returns:
        Field value string.

    Raises:
        MissingRequiredFieldError: If field is missing.
        TypeError: If field types are invalid.
        ValueError: If field_path is invalid.
    """
    value = _require_field(record, field_path)
    if not isinstance(value, str):
        raise TypeError(f"Type mismatch at {field_path}: expected str")
    if not value:
        raise TypeError(f"Type mismatch at {field_path}: expected non-empty str")
    return value


def _should_require_bound_fact_sources(payload: Dict[str, Any], status_ok: bool) -> bool:
    """
    功能：判断 run_closure 是否必须携带 bound_fact_sources。

    Decide whether bound_fact_sources is required for the run_closure payload.

    Args:
        payload: Run closure payload mapping.
        status_ok: Parsed status.ok value.

    Returns:
        True if bound_fact_sources must be present; otherwise False.
    """
    if status_ok:
        return True
    closure_stage = payload.get("closure_stage")
    if isinstance(closure_stage, dict):
        records_present = closure_stage.get("records_present")
        return records_present is True
    return False


def _validate_bound_fact_sources(payload: Dict[str, Any]) -> None:
    """
    功能：校验 bound_fact_sources 的完整性与一致性。

    Validate that bound_fact_sources contains required fields and non-sentinel values.

    Args:
        payload: Run closure payload mapping.

    Returns:
        None.

    Raises:
        TypeError: If types are invalid.
        ValueError: If required fields are missing or invalid.
    """
    bound_fact_sources = payload.get("bound_fact_sources")
    if not isinstance(bound_fact_sources, dict):
        raise ValueError("bound_fact_sources must be dict when required")
    bound_status = payload.get("bound_fact_sources_status")
    if bound_status is not None:
        if bound_status != "bound" and bound_status != "unbound":
            raise ValueError("bound_fact_sources_status must be 'bound' or 'unbound'")
        if bound_status == "unbound":
            raise ValueError("bound_fact_sources_status must not be 'unbound' when required")

    required_fields = [
        "contract_version",
        "contract_digest",
        "contract_file_sha256",
        "contract_canon_sha256",
        "contract_bound_digest",
        "whitelist_version",
        "whitelist_digest",
        "whitelist_file_sha256",
        "whitelist_canon_sha256",
        "whitelist_bound_digest",
        "policy_path_semantics_version",
        "policy_path_semantics_digest",
        "policy_path_semantics_file_sha256",
        "policy_path_semantics_canon_sha256",
        "policy_path_semantics_bound_digest",
        "injection_scope_manifest_version",
        "injection_scope_manifest_digest",
        "injection_scope_manifest_file_sha256",
        "injection_scope_manifest_canon_sha256",
        "injection_scope_manifest_bound_digest"
    ]
    for field_name in required_fields:
        value = _require_str_field(bound_fact_sources, field_name)
        if value == "<absent>":
            raise ValueError(
                "bound_fact_sources must not contain '<absent>' sentinel: "
                f"field_name={field_name}"
            )


def _validate_run_closure_ok_fields(payload: Dict[str, Any]) -> None:
    """
    功能：校验 ok 闭包关键字段不得为空或为哨兵值。

    Validate critical fields for ok closures to avoid sentinel placeholders.

    Args:
        payload: Run closure payload mapping.

    Returns:
        None.

    Raises:
        TypeError: If payload types are invalid.
        ValueError: If required fields are missing or sentinel.
    """
    if not isinstance(payload, dict):
        # payload 类型不符合预期，必须 fail-fast。
        raise TypeError("payload must be dict")

    _required_fields = [
        "run_id",
        "cfg_digest",
        "policy_path",
        "impl_id",
        "impl_version"
    ]
    if payload.get("impl_identity") is not None or payload.get("impl_identity_digest") is not None:
        _required_fields.append("impl_identity_digest")

    for field_name in _required_fields:
        _value = payload.get(field_name)
        _assert_non_absent_str(_value, field_name)

    facts_anchor = payload.get("facts_anchor")
    if not isinstance(facts_anchor, dict):
        # facts_anchor 类型不符合预期，必须 fail-fast。
        raise TypeError("facts_anchor must be dict when status.ok is True")
    for field_name in [
        "contract_bound_digest",
        "whitelist_bound_digest",
        "policy_path_semantics_bound_digest"
    ]:
        _value = facts_anchor.get(field_name)
        _assert_non_absent_str(_value, f"facts_anchor.{field_name}")


def _assert_non_absent_str(value: Any, field_path: str) -> None:
    """
    功能：断言字段为非空且非哨兵字符串。

    Assert that a field is a non-empty, non-sentinel string.

    Args:
        value: Field value to validate.
        field_path: Field path for error messages.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If value is empty or sentinel.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(value, str):
        # value 类型不符合预期，必须 fail-fast。
        raise TypeError(f"{field_path} must be str")
    if not value or value == "<absent>":
        # value 为空或为哨兵值，必须 fail-fast。
        raise ValueError(f"{field_path} must be non-empty and not '<absent>'")


def _validate_closure_stage(closure_stage: Dict[str, Any]) -> None:
    """
    功能：校验 closure_stage 审计块。

    Validate closure_stage audit block for run_closure.

    Args:
        closure_stage: Closure stage mapping to validate.

    Returns:
        None.

    Raises:
        TypeError: If field types are invalid.
        ValueError: If field values are invalid.
    """
    if not isinstance(closure_stage, dict):
        # closure_stage 类型不符合预期，必须 fail-fast。
        raise TypeError("closure_stage must be dict")

    # 校验 records_present 字段
    records_present = closure_stage.get("records_present")
    if not isinstance(records_present, bool):
        # records_present 类型不符合预期，必须 fail-fast。
        raise TypeError("closure_stage.records_present must be bool")

    # 校验 bundle_attempted 字段
    bundle_attempted = closure_stage.get("bundle_attempted")
    if not isinstance(bundle_attempted, bool):
        # bundle_attempted 类型不符合预期，必须 fail-fast。
        raise TypeError("closure_stage.bundle_attempted must be bool")

    # 校验 bundle_succeeded 字段
    bundle_succeeded = closure_stage.get("bundle_succeeded")
    if not isinstance(bundle_succeeded, bool):
        # bundle_succeeded 类型不符合预期，必须 fail-fast。
        raise TypeError("closure_stage.bundle_succeeded must be bool")

    # 校验 failure_stage 字段
    failure_stage = closure_stage.get("failure_stage")
    if failure_stage is not None:
        if not isinstance(failure_stage, str) or not failure_stage:
            # failure_stage 类型不符合预期，必须 fail-fast。
            raise TypeError("closure_stage.failure_stage must be non-empty str or None")
        allowed_stages = {"bundle", "anchor_merge", "payload_build", "write_artifact", "unknown"}
        if failure_stage not in allowed_stages:
            # failure_stage 值不在允许列表中，必须 fail-fast。
            raise ValueError(
                f"closure_stage.failure_stage must be one of {allowed_stages}, got '{failure_stage}'"
            )

    # 校验 upstream_status_reason 字段
    upstream_reason = closure_stage.get("upstream_status_reason")
    if upstream_reason is not None:
        if not isinstance(upstream_reason, str) or not upstream_reason:
            # upstream_status_reason 类型不符合预期，必须 fail-fast。
            raise TypeError("closure_stage.upstream_status_reason must be non-empty str or None")
        allowed_reasons = {reason.value for reason in RunFailureReason}
        if upstream_reason not in allowed_reasons:
            # upstream_status_reason 非受控枚举成员，必须 fail-fast。
            raise ValueError(
                f"closure_stage.upstream_status_reason must be RunFailureReason member, got '{upstream_reason}'"
            )


def validate_record(
    record: Dict[str, Any],
    *,
    interpretation: ContractInterpretation | None = None
) -> None:
    """
    功能：校验 record 的权威 schema 约束。

    Validate record against authoritative contract interpretation.

    Args:
        record: Record mapping to validate.
        interpretation: Optional contract interpretation for required fields.

    Returns:
        None.

    Raises:
        MissingRequiredFieldError: If required fields are missing.
        TypeError: If record types are invalid.
        ValueError: If inputs are invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if interpretation is not None and not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation or None")

    if interpretation is None:
        # interpretation 缺失，必须 fail-fast。
        raise ContractInterpretationRequiredError(
            "ContractInterpretation is required for schema.validate_record"
        )

    _validate_json_like(record, "<root>")

    required_fields = interpretation.required_record_fields
    decision_field_path = interpretation.records_schema.decision_field_path
    for field_path in required_fields:
        found, value = _get_value_by_field_path(record, field_path)
        if not found:
            # 必需字段缺失，必须 fail-fast。
            raise MissingRequiredFieldError(f"Missing required field: {field_path}")
        if not isinstance(value, str):
            # 必需字段类型不匹配，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected str")

    # schema_version 必须与权威版本一致，避免调用方绕过 ensure_required_fields。
    schema_version = record.get("schema_version")
    if schema_version != RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected={RECORD_SCHEMA_VERSION}, got={schema_version}"
        )
    
    # decision 字段的校验：若 decision_obligation_presence 为 true，则检查字段存在性与类型。
    if interpretation.records_schema.decision_obligation_presence:
        found, value = _get_value_by_field_path(record, decision_field_path)
        if not found:
            # decision 字段缺失，必须 fail-fast。
            raise MissingRequiredFieldError(f"Missing required field: {decision_field_path}")
        if value is not None and not isinstance(value, bool):
            # decision 字段类型不符（必须是 bool | None），必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {decision_field_path}: expected bool or None"
            )

    impl_identity_spec = interpretation.impl_identity
    for _, field_path in impl_identity_spec.field_paths_by_domain.items():
        found, value = _get_value_by_field_path(record, field_path)
        if not found:
            if impl_identity_spec.required:
                # impl_identity 必需字段缺失，必须 fail-fast。
                raise MissingRequiredFieldError(f"Missing required field: {field_path}")
            continue
        if not isinstance(value, str):
            # impl_identity 字段类型不匹配，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected str")

    pipeline_impl_id = record.get("pipeline_impl_id")
    if pipeline_impl_id is not None:
        if not isinstance(pipeline_impl_id, str) or not pipeline_impl_id:
            # pipeline_impl_id 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at pipeline_impl_id: expected non-empty str")

    pipeline_provenance = record.get("pipeline_provenance")
    if pipeline_provenance is not None:
        if not isinstance(pipeline_provenance, dict):
            # pipeline_provenance 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at pipeline_provenance: expected dict")
        _validate_json_like(pipeline_provenance, "pipeline_provenance")

    pipeline_provenance_canon_sha256 = record.get("pipeline_provenance_canon_sha256")
    if pipeline_provenance_canon_sha256 is not None:
        if not isinstance(pipeline_provenance_canon_sha256, str) or not pipeline_provenance_canon_sha256:
            # pipeline_provenance_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("pipeline_provenance_canon_sha256 must be non-empty str")

    pipeline_runtime_meta = record.get("pipeline_runtime_meta")
    if pipeline_runtime_meta is not None:
        if not isinstance(pipeline_runtime_meta, dict):
            # pipeline_runtime_meta 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at pipeline_runtime_meta: expected dict")
        _validate_json_like(pipeline_runtime_meta, "pipeline_runtime_meta")

    # (7.7) Real Dataflow Smoke: inference 相关字段的存在即校验
    inference_status = record.get("inference_status")
    if inference_status is not None:
        if not isinstance(inference_status, str) or not inference_status:
            # inference_status 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at inference_status: expected non-empty str")
        allowed_inference_status = {"ok", "failed", "disabled"}
        if inference_status not in allowed_inference_status:
            raise ValueError(
                f"inference_status must be one of {allowed_inference_status}, got '{inference_status}'"
            )

    inference_error = record.get("inference_error")
    if inference_error is not None:
        if not isinstance(inference_error, str) or not inference_error:
            # inference_error 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at inference_error: expected non-empty str or None")

    inference_runtime_meta = record.get("inference_runtime_meta")
    if inference_runtime_meta is not None:
        if not isinstance(inference_runtime_meta, dict):
            # inference_runtime_meta 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at inference_runtime_meta: expected dict")
        _validate_json_like(inference_runtime_meta, "inference_runtime_meta")

    infer_trace = record.get("infer_trace")
    if infer_trace is not None:
        if not isinstance(infer_trace, dict):
            # infer_trace 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at infer_trace: expected dict")
        _validate_json_like(infer_trace, "infer_trace")

    infer_trace_canon_sha256 = record.get("infer_trace_canon_sha256")
    if infer_trace_canon_sha256 is not None:
        if not isinstance(infer_trace_canon_sha256, str) or not infer_trace_canon_sha256:
            # infer_trace_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("Type mismatch at infer_trace_canon_sha256: expected non-empty str")

    env_fingerprint_canon_sha256 = record.get("env_fingerprint_canon_sha256")
    if env_fingerprint_canon_sha256 is not None:
        if not isinstance(env_fingerprint_canon_sha256, str) or not env_fingerprint_canon_sha256:
            # env_fingerprint_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("env_fingerprint_canon_sha256 must be non-empty str")

    diffusers_version = record.get("diffusers_version")
    if diffusers_version is not None:
        if not isinstance(diffusers_version, str) or not diffusers_version:
            # diffusers_version 类型不符合预期，必须 fail-fast。
            raise TypeError("diffusers_version must be non-empty str")

    transformers_version = record.get("transformers_version")
    if transformers_version is not None:
        if not isinstance(transformers_version, str) or not transformers_version:
            # transformers_version 类型不符合预期，必须 fail-fast。
            raise TypeError("transformers_version must be non-empty str")

    safetensors_version = record.get("safetensors_version")
    if safetensors_version is not None:
        if not isinstance(safetensors_version, str) or not safetensors_version:
            # safetensors_version 类型不符合预期，必须 fail-fast。
            raise TypeError("safetensors_version must be non-empty str")

    model_provenance_canon_sha256 = record.get("model_provenance_canon_sha256")
    if model_provenance_canon_sha256 is not None:
        if not isinstance(model_provenance_canon_sha256, str) or not model_provenance_canon_sha256:
            # model_provenance_canon_sha256 类型不符合预期，必须 fail-fast。
            raise TypeError("model_provenance_canon_sha256 must be non-empty str")

    # 从解释面获取可选字段列表（若启用扩展则使用动态列表，否则使用后备硬编码列表）
    extensions_spec = interpretation.records_schema_extensions_spec
    if extensions_spec.enabled:
        optional_str_fields = extensions_spec.optional_str_fields
        optional_number_fields = extensions_spec.optional_number_fields
        optional_mapping_fields = extensions_spec.optional_mapping_fields
    else:
        # 向后兼容：若扩展未启用，使用硬编码后备列表
        optional_str_fields = _FALLBACK_OPTIONAL_STR_FIELDS
        optional_number_fields = _FALLBACK_OPTIONAL_NUMBER_FIELDS
        optional_mapping_fields = _FALLBACK_OPTIONAL_MAPPING_FIELDS

    _validate_optional_str_fields(record, optional_str_fields)
    _validate_optional_number_fields(record, optional_number_fields)
    _validate_optional_mapping_fields(record, optional_mapping_fields)


def _get_value_by_field_path(record: Dict[str, Any], field_path: str) -> tuple[bool, Any]:
    """
    功能：按点路径读取 record 字段值。

    Get a nested record value by dotted field path.

    Args:
        record: Record dict to read from.
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
    segments = field_path.split(".")
    for segment in segments:
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if not isinstance(current, dict):
            # 中间段类型不合法，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected mapping")
        if segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _validate_json_like(value: Any, field_path: str) -> None:
    """
    功能：校验 JSON-like 结构。

    Validate that a value is JSON-like (dict/list/scalars).

    Args:
        value: Value to validate.
        field_path: Field path for error messages.

    Returns:
        None.

    Raises:
        TypeError: If value contains non-JSON-like types.
        ValueError: If field_path is invalid.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")
    if _is_json_scalar(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_like(item, f"{field_path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                # JSON key 类型不合法，必须 fail-fast。
                raise TypeError(f"Type mismatch at {field_path}: expected str keys")
            _validate_json_like(item, f"{field_path}.{key}")
        return
    # 非 JSON-like 类型，必须 fail-fast。
    raise TypeError(f"Type mismatch at {field_path}: non-JSON-like {type(value).__name__}")


def _validate_optional_str_fields(record: Dict[str, Any], field_paths: List[str]) -> None:
    """
    功能：校验可选字符串字段集合。

    Validate optional string fields by dotted field paths.

    Args:
        record: Record mapping.
        field_paths: List of dotted field paths.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid or field types mismatch.
        ValueError: If field path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_paths, list):
        # field_paths 类型不符合预期，必须 fail-fast。
        raise TypeError("field_paths must be list")
    for field_path in field_paths:
        if not isinstance(field_path, str) or not field_path:
            # field_path 类型不符合预期，必须 fail-fast。
            raise ValueError("field_path must be non-empty str")
        found, value = _get_value_by_field_path(record, field_path)
        if not found or value is None:
            continue
        if not isinstance(value, str) or not value:
            # 字段类型不符合预期，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected non-empty str")


def _validate_optional_number_fields(record: Dict[str, Any], field_paths: List[str]) -> None:
    """
    功能：校验可选数值字段集合。

    Validate optional numeric fields by dotted field paths.

    Args:
        record: Record mapping.
        field_paths: List of dotted field paths.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid or field types mismatch.
        ValueError: If field path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_paths, list):
        # field_paths 类型不符合预期，必须 fail-fast。
        raise TypeError("field_paths must be list")
    for field_path in field_paths:
        if not isinstance(field_path, str) or not field_path:
            # field_path 类型不符合预期，必须 fail-fast。
            raise ValueError("field_path must be non-empty str")
        found, value = _get_value_by_field_path(record, field_path)
        if not found or value is None:
            continue
        if not isinstance(value, (int, float)):
            # 字段类型不符合预期，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected number")


def _validate_optional_mapping_fields(record: Dict[str, Any], field_paths: List[str]) -> None:
    """
    功能：校验可选映射字段集合。

    Validate optional mapping fields by dotted field paths.

    Args:
        record: Record mapping.
        field_paths: List of dotted field paths.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid or field types mismatch.
        ValueError: If field path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(field_paths, list):
        # field_paths 类型不符合预期，必须 fail-fast。
        raise TypeError("field_paths must be list")
    for field_path in field_paths:
        if not isinstance(field_path, str) or not field_path:
            # field_path 类型不符合预期，必须 fail-fast。
            raise ValueError("field_path must be non-empty str")
        found, value = _get_value_by_field_path(record, field_path)
        if not found or value is None:
            continue
        if not isinstance(value, dict):
            # 字段类型不符合预期，必须 fail-fast。
            raise TypeError(f"Type mismatch at {field_path}: expected dict")
        _validate_json_like(value, field_path)


def _is_json_scalar(value: Any) -> bool:
    """
    功能：判断是否为 JSON 标量。

    Check whether a value is a JSON scalar.

    Args:
        value: Value to check.

    Returns:
        True if value is JSON scalar; otherwise False.
    """
    return isinstance(value, (str, int, float, bool)) or value is None


def build_thresholds_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造阈值占位 spec。

    Build a placeholder thresholds specification for digest derivation.

    Args:
        cfg: Config mapping.

    Returns:
        Thresholds spec mapping.

    Raises:
        TypeError: If cfg or derived fields are invalid.
        ValueError: If required fields are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    from main.watermarking.fusion import neyman_pearson

    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    _validate_json_like(thresholds_spec, "thresholds_spec")
    return thresholds_spec


def compute_thresholds_digest(thresholds_spec: Dict[str, Any]) -> str:
    """
    功能：计算 thresholds_digest。

    Compute thresholds digest using canonical semantic digest.

    Args:
        thresholds_spec: Thresholds spec mapping.

    Returns:
        Thresholds digest string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If digest output is invalid.
    """
    if not isinstance(thresholds_spec, dict):
        # thresholds_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("thresholds_spec must be dict")
    _validate_json_like(thresholds_spec, "thresholds_spec")

    from main.watermarking.fusion import neyman_pearson

    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)
    if not isinstance(thresholds_digest, str) or not thresholds_digest:
        # thresholds_digest 类型不符合预期，必须 fail-fast。
        raise ValueError("thresholds_digest must be non-empty str")
    return thresholds_digest


def _resolve_record_kind(record: Dict[str, Any]) -> str:
    """
    功能：根据 operation 字段推断 record 类型（statistical/non_statistical）。
    
    Resolve record kind based on operation field.
    
    Args:
        record: Record mapping with operation field.
    
    Returns:
        "statistical" or "non_statistical"
    
    Raises:
        MissingRequiredFieldError: If operation field is missing.
    """
    operation = record.get("operation")
    if operation is None:
        raise MissingRequiredFieldError("Missing required field: operation")
    
    # 统计类操作映射（阶段 1 占位实现）
    statistical_operations = {"calibrate", "detect", "evaluate"}
    
    if operation in statistical_operations:
        return "statistical"
    else:
        return "non_statistical"


def ensure_required_fields(
    record: Dict[str, Any],
    cfg: Dict[str, Any],
    interpretation: ContractInterpretation | None = None
) -> Dict[str, Any]:
    """
    功能：补齐 records 必需字段。

    Ensure derived required fields are present using authoritative interpretation.
    Automatically injects schema_version if missing.

    Args:
        record: Record mapping to mutate.
        cfg: Config mapping used to build thresholds spec.
        interpretation: Contract interpretation for required fields.

    Returns:
        Record mapping with required fields ensured.

    Raises:
        TypeError: If inputs or existing fields are invalid.
        ContractInterpretationRequiredError: If interpretation is missing.
        ValueError: If existing digest mismatches derived digest or schema_version mismatch.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 缺失或类型不合法，必须 fail-fast。
        raise ContractInterpretationRequiredError(
            "ContractInterpretation is required for ensure_required_fields"
        )

    # schema_version 自动注入与校验。
    if "schema_version" not in record:
        # 若 record 不含 schema_version，自动写入权威版本。
        record["schema_version"] = RECORD_SCHEMA_VERSION
    else:
        # 若已含 schema_version 且不等于权威版本，必须 fail-fast。
        existing_version = record.get("schema_version")
        if existing_version != RECORD_SCHEMA_VERSION:
            from main.core.errors import GateEnforcementError
            raise GateEnforcementError(
                "schema_version mismatch",
                gate_name="records_schema.schema_version.consistency",
                field_path="schema_version",
                expected=RECORD_SCHEMA_VERSION,
                actual=str(existing_version)
            )

    if "operation" not in record:
        # operation 缺失时仅执行基础注入，避免破坏最小路径兼容性。
        return record

    # 统计适用矩阵注入（阶段 2：扩展字段注入 + warn 模式校验）
    # 根据 operation 字段判断 record 类型，预填充统计相关字段
    record_kind = _resolve_record_kind(record)
    thresholds_spec = None

    # stats_applicability 字段注入
    if "stats_applicability" not in record:
        if record_kind == "statistical":
            record["stats_applicability"] = "applicable"
        else:
            record["stats_applicability"] = "<absent>"

    # threshold_source 字段注入（阶段 2：仅统计类记录）
    if record_kind == "statistical":
        if "threshold_source" not in record:
            record["threshold_source"] = "np_canonical"
        thresholds_spec = build_thresholds_spec(cfg)
        expected_target_fpr = thresholds_spec.get("target_fpr")
        if not isinstance(expected_target_fpr, (int, float)):
            # thresholds_spec.target_fpr 类型不符合预期，必须 fail-fast。
            raise TypeError("thresholds_spec.target_fpr must be number")
        if "target_fpr" not in record:
            record["target_fpr"] = float(expected_target_fpr)
        else:
            record_target_fpr = record.get("target_fpr")
            if not isinstance(record_target_fpr, (int, float)):
                # target_fpr 类型不符合预期，必须 fail-fast。
                raise TypeError("Type mismatch at target_fpr: expected number")
            if float(record_target_fpr) != float(expected_target_fpr):
                # target_fpr 与阈值口径不一致，必须 fail-fast。
                raise ValueError(
                    "target_fpr mismatch with thresholds_spec: "
                    f"expected={expected_target_fpr}, actual={record_target_fpr}"
                )

        # 统计类记录字段完整性校验（warn 模式）
        statistical_fields = [
            "target_fpr",
            "threshold_source",
            "stats_applicability"
        ]
        missing_fields = [f for f in statistical_fields if f not in record]
        if missing_fields:
            print(
                "[Schema][WARN] Statistical record missing fields (阶段 2 warn 模式): "
                f"operation={record.get('operation')}, missing={missing_fields}"
            )

    # 非统计类记录不应包含统计字段（warn 模式）
    elif record_kind == "non_statistical":
        for field_name in ["target_fpr", "threshold_source", "stats_applicability"]:
            if field_name not in record:
                record[field_name] = "<absent>"
        statistical_fields_present = [
            f for f in ["target_fpr", "threshold", "threshold_source", "stats_applicability"]
            if f in record and record[f] not in [None, "<absent>"]
        ]
        if statistical_fields_present:
            print(
                "[Schema][WARN] Non-statistical record contains statistical fields (阶段 2 warn 模式): "
                f"operation={record.get('operation')}, fields={statistical_fields_present}"
            )

    required_fields = set(interpretation.required_record_fields)

    if "thresholds_digest" in required_fields or "thresholds_digest" in record:
        if thresholds_spec is None:
            thresholds_spec = build_thresholds_spec(cfg)
        thresholds_digest = compute_thresholds_digest(thresholds_spec)
        if "thresholds_digest" in record:
            existing_digest = record.get("thresholds_digest")
            if not isinstance(existing_digest, str) or not existing_digest:
                # thresholds_digest 类型不符合预期，必须 fail-fast。
                raise TypeError("Type mismatch at thresholds_digest: expected non-empty str")
            if existing_digest != thresholds_digest:
                # thresholds_digest 与推导值不一致，必须 fail-fast。
                raise ValueError("thresholds_digest mismatch at thresholds_digest")
        elif "thresholds_digest" in required_fields:
            record["thresholds_digest"] = thresholds_digest

    if "threshold_metadata_digest" in required_fields or "threshold_metadata_digest" in record:
        if thresholds_spec is None:
            thresholds_spec = build_thresholds_spec(cfg)
        from main.watermarking.fusion import neyman_pearson

        threshold_metadata = neyman_pearson.build_threshold_metadata(thresholds_spec)
        threshold_metadata_digest = neyman_pearson.compute_threshold_metadata_digest(threshold_metadata)
        if "threshold_metadata_digest" in record:
            existing_digest = record.get("threshold_metadata_digest")
            if not isinstance(existing_digest, str) or not existing_digest:
                # threshold_metadata_digest 类型不符合预期，必须 fail-fast。
                raise TypeError("Type mismatch at threshold_metadata_digest: expected non-empty str")
            if existing_digest != threshold_metadata_digest:
                # threshold_metadata_digest 与推导值不一致，必须 fail-fast。
                raise ValueError("threshold_metadata_digest mismatch at threshold_metadata_digest")
        elif "threshold_metadata_digest" in required_fields:
            record["threshold_metadata_digest"] = threshold_metadata_digest

    # 注入 thresholds_rule_id 和 thresholds_rule_version。
    from main.watermarking.fusion import neyman_pearson as _np
    if "thresholds_rule_id" in required_fields or "thresholds_rule_id" in record:
        rule_id = _np.RULE_ID
        if "thresholds_rule_id" in record:
            existing = record.get("thresholds_rule_id")
            if not isinstance(existing, str) or not existing:
                # thresholds_rule_id 类型不符合预期，必须 fail-fast。
                raise TypeError("Type mismatch at thresholds_rule_id: expected non-empty str")
            if existing != rule_id:
                # thresholds_rule_id 与权威值不一致，必须 fail-fast。
                raise ValueError(
                    f"thresholds_rule_id mismatch: expected={rule_id}, actual={existing}"
                )
        elif "thresholds_rule_id" in required_fields:
            record["thresholds_rule_id"] = rule_id

    if "thresholds_rule_version" in required_fields or "thresholds_rule_version" in record:
        rule_version = _np.RULE_VERSION
        if "thresholds_rule_version" in record:
            existing = record.get("thresholds_rule_version")
            if not isinstance(existing, str) or not existing:
                # thresholds_rule_version 类型不符合预期，必须 fail-fast。
                raise TypeError("Type mismatch at thresholds_rule_version: expected non-empty str")
            if existing != rule_version:
                # thresholds_rule_version 与权威值不一致，必须 fail-fast。
                raise ValueError(
                    f"thresholds_rule_version mismatch: expected={rule_version}, actual={existing}"
                )
        elif "thresholds_rule_version" in required_fields:
            record["thresholds_rule_version"] = rule_version

    decision_field_path = interpretation.records_schema.decision_field_path
    if interpretation.records_schema.decision_obligation_presence:
        found, value = _get_value_by_field_path(record, decision_field_path)
        if found:
            if value is not None and not isinstance(value, bool):
                # decision 字段类型不符合预期，必须 fail-fast。
                raise TypeError(
                    f"Type mismatch at {decision_field_path}: expected bool or None"
                )
        else:
            # 缺失时写入 None，而不是 False。
            _set_value_by_field_path(record, decision_field_path, None)
    else:
        # 若不强制 decision，仅检查存在时的类型。
        found, value = _get_value_by_field_path(record, decision_field_path)
        if found and value is not None and not isinstance(value, bool):
            # decision 字段类型不符合预期，必须 fail-fast。
            raise TypeError(
                f"Type mismatch at {decision_field_path}: expected bool or None"
            )
    return record


def _set_value_by_field_path(record: Dict[str, Any], field_path: str, value: Any) -> None:
    """
    功能：按点路径写入 record 字段值。

    Set a nested record value by dotted field path.

    Args:
        record: Record mapping to mutate.
        field_path: Dotted field path string.
        value: Value to set.

    Returns:
        None.

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
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    last = segments[-1]
    if not last:
        # field_path 末段为空，必须 fail-fast。
        raise ValueError(f"Invalid field_path segment in {field_path}")
    current[last] = value
