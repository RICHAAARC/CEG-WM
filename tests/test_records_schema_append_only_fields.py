"""
File purpose: 记录追加字段的 schema 兼容性回归测试。

Module type: Core innovation module

功能说明：
- 验证新增字段缺失时仍可通过 schema 校验。
- 验证新增字段类型错误时 fail-fast。
- 验证新增字段类型正确时通过校验。
"""

import pytest


def _set_value_by_field_path(mapping, field_path, value):
    """
    功能：按点路径写入映射字段。

    Set a nested mapping value by dotted field path.

    Args:
        mapping: Target mapping to mutate.
        field_path: Dotted field path.
        value: Value to assign.

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
        # field_path 类型不符合预期，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            # segment 类型不符合预期，必须 fail-fast。
            raise ValueError("field_path contains empty segment")
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    if not segments[-1]:
        # 末段为空，必须 fail-fast。
        raise ValueError("field_path contains empty segment")
    current[segments[-1]] = value


def _build_minimal_valid_record(schema_module, interpretation):
    """
    功能：构造最小可通过的 record。

    Build a minimal record that satisfies required fields.

    Args:
        schema_module: schema module providing RECORD_SCHEMA_VERSION.
        interpretation: Contract interpretation instance.

    Returns:
        Minimal valid record mapping.

    Raises:
        TypeError: If inputs are invalid.
    """
    if schema_module is None:
        # schema_module 输入不合法，必须 fail-fast。
        raise TypeError("schema_module must be provided")
    if interpretation is None:
        # interpretation 输入不合法，必须 fail-fast。
        raise TypeError("interpretation must be provided")

    record = {}
    for field_path in interpretation.required_record_fields:
        if field_path == "schema_version":
            _set_value_by_field_path(record, field_path, schema_module.RECORD_SCHEMA_VERSION)
        else:
            _set_value_by_field_path(record, field_path, "stub")

    decision_field_path = interpretation.records_schema.decision_field_path
    _set_value_by_field_path(record, decision_field_path, None)
    return record


def test_records_schema_new_fields_absent_ok(mock_interpretation):
    """
    功能：新增字段缺失时仍可通过校验。

    Validate that new optional fields may be absent.

    Args:
        mock_interpretation: Contract interpretation fixture.

    Returns:
        None.
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")

    record = _build_minimal_valid_record(schema, mock_interpretation)
    try:
        schema.validate_record(record, interpretation=mock_interpretation)
    except Exception as exc:
        pytest.fail(f"validate_record should accept missing optional fields: {exc}")


def test_records_schema_new_fields_type_mismatch_fails(mock_interpretation):
    """
    功能：新增字段类型错误时必须 fail-fast。

    Validate that type mismatch on new fields fails fast.

    Args:
        mock_interpretation: Contract interpretation fixture.

    Returns:
        None.
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")

    record = _build_minimal_valid_record(schema, mock_interpretation)
    _set_value_by_field_path(record, "content_evidence.mask_digest", 123)
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence", ["not", "a", "dict"])

    with pytest.raises(TypeError) as exc_info:
        schema.validate_record(record, interpretation=mock_interpretation)

    assert "content_evidence.mask_digest" in str(exc_info.value)


def test_records_schema_new_fields_valid_types_pass(mock_interpretation):
    """
    功能：新增字段类型正确时应通过校验。

    Validate that correctly typed optional fields pass validation.

    Args:
        mock_interpretation: Contract interpretation fixture.

    Returns:
        None.
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")

    record = _build_minimal_valid_record(schema, mock_interpretation)

    _set_value_by_field_path(record, "content_evidence.mask_digest", "a" * 64)
    _set_value_by_field_path(record, "content_evidence.mask_stats", {"area_ratio": 0.1})
    _set_value_by_field_path(record, "content_evidence.lf_score", 0.12)
    _set_value_by_field_path(record, "content_evidence.injection_status", "ok")
    _set_value_by_field_path(record, "content_evidence.injection_absent_reason", "unsupported_pipeline")
    _set_value_by_field_path(record, "content_evidence.injection_failure_reason", "injection_params_mismatch")
    _set_value_by_field_path(record, "content_evidence.injection_trace_digest", "e" * 64)
    _set_value_by_field_path(record, "content_evidence.injection_params_digest", "f" * 64)
    _set_value_by_field_path(record, "content_evidence.injection_metrics", {"step_count": 4})
    _set_value_by_field_path(record, "content_evidence.subspace_binding_digest", "a" * 64)
    _set_value_by_field_path(record, "content_evidence.lf_statistics_digest", "b" * 64)
    _set_value_by_field_path(record, "content_evidence.hf_statistics_digest", "c" * 64)
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence", {"status": "ok"})
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.status", "ok")
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_spec", {"sample_count": 4})
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_spec_digest", "c" * 64)
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_digest", "d" * 64)
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_metrics", {"shape": [1, 2]})
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_stats", {"shape": [1, 2]})
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.audit", {"trajectory_tap_status": "ok"})
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.audit.trajectory_tap_status", "ok")
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.audit.trajectory_absent_reason", "tap_disabled")
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_absent_reason", "tap_disabled")
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.trajectory_tap_version", "v1")
    _set_value_by_field_path(record, "content_evidence.trajectory_evidence.device", "cpu")
    _set_value_by_field_path(record, "content_evidence.audit.trajectory_tap_status", "ok")
    _set_value_by_field_path(record, "content_evidence.audit.trajectory_absent_reason", "tap_disabled")
    _set_value_by_field_path(record, "geometry_evidence.anchor_metrics", {"stability": 0.9})
    _set_value_by_field_path(record, "decision.routing_decisions", {"content": "enabled"})
    _set_value_by_field_path(record, "decision.routing_digest", "b" * 64)

    try:
        schema.validate_record(record, interpretation=mock_interpretation)
    except Exception as exc:
        pytest.fail(f"validate_record should accept valid optional fields: {exc}")
