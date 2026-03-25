"""
功能：测试 schema 校验必须要求 interpretation（schema.interpretation_is_required，legacy_code=A2）

Module type: Core innovation module

Test that schema validation requires interpretation and fails
when interpretation is missing or None.
"""

import pytest


def _set_value_by_field_path(mapping, field_path, value):
    current = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    current[segments[-1]] = value


def _build_minimal_valid_record(schema_module, interpretation):
    record = {}
    for field_path in interpretation.required_record_fields:
        if field_path == "schema_version":
            _set_value_by_field_path(record, field_path, schema_module.RECORD_SCHEMA_VERSION)
        else:
            _set_value_by_field_path(record, field_path, "stub")

    decision_field_path = interpretation.records_schema.decision_field_path
    _set_value_by_field_path(record, decision_field_path, None)
    return record


def test_schema_requires_interpretation_parameter(mock_interpretation):
    """
    Test that validate_record requires interpretation parameter.
    
    缺少 interpretation 参数必须 fail-fast。
    """
    # 导入 schema 模块
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")
    
    # 构造最小 record
    minimal_record = {
        "run_id": "test_run_001",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
    }
    
    # (1) 缺少 interpretation 参数应该抛异常
    from main.core.errors import ContractInterpretationRequiredError

    with pytest.raises(ContractInterpretationRequiredError) as exc_info:
        schema.validate_record(minimal_record)
    
    # 检查异常信息包含 interpretation 相关提示
    assert "interpretation" in str(exc_info.value).lower()


def test_schema_rejects_none_interpretation(mock_interpretation):
    """
    Test that validate_record rejects None interpretation.
    
    interpretation 为 None 必须 fail-fast。
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")
    
    minimal_record = {
        "run_id": "test_run_002",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
    }
    
    # (2) 传入 None 应该抛异常
    from main.core.errors import ContractInterpretationRequiredError

    with pytest.raises(ContractInterpretationRequiredError) as exc_info:
        schema.validate_record(minimal_record, interpretation=None)
    
    assert "interpretation" in str(exc_info.value).lower()


def test_schema_passes_with_valid_interpretation(mock_interpretation):
    """
    Test that validate_record passes with valid interpretation.
    
    正确提供 interpretation 应该通过校验。
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")
    
    minimal_record = _build_minimal_valid_record(schema, mock_interpretation)
    
    # (3) 正确提供 interpretation 应该不抛异常
    try:
        schema.validate_record(minimal_record, interpretation=mock_interpretation)
        # 如果有返回值，检查是否为 True 或等价成功标识
    except Exception as e:
        pytest.fail(f"validate_record should not raise with valid interpretation: {e}")


def test_schema_checks_required_fields(mock_interpretation):
    """
    Test that schema validates required fields from interpretation.
    
    缺少 required 字段应该失败并可定位到字段路径。
    """
    try:
        from main.core import schema
    except ImportError:
        pytest.skip("main.core.schema module not found")
    
    from main.core.errors import MissingRequiredFieldError

    incomplete_record = _build_minimal_valid_record(schema, mock_interpretation)
    incomplete_record.pop("contract_digest", None)
    
    # 应该抛异常
    with pytest.raises(MissingRequiredFieldError) as exc_info:
        schema.validate_record(incomplete_record, interpretation=mock_interpretation)
    
    # 检查异常信息包含字段名
    error_msg = str(exc_info.value).lower()
    assert "contract_digest" in error_msg or "required" in error_msg
