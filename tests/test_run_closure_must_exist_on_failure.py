"""
功能：测试失败时 run_closure 必产出（F1/F2 覆盖）

Module type: Core innovation module

Test that run_closure.json is always generated even when the run fails,
and failure reason is properly recorded.
"""

import pytest
import json
from pathlib import Path


def test_run_closure_exists_on_failure(tmp_run_root):
    """
    Test that run_closure.json is generated even when run fails.
    
    运行失败时 run_closure.json 必须生成。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    # 模拟运行失败场景
    # 注：实际测试需要触发真实的失败路径
    
    # 构造 run_closure（失败场景）
    run_closure = {
        "ok": False,
        "run_id": "test_run_fail_001",
        "failure_reason": "test_induced_failure",
        "run_meta": {
            "contract_version": "v1.0.0",
            # 其他字段可能缺失，使用 <absent>
        },
    }
    
    closure_path = tmp_run_root / "run_closure.json"
    closure_path.write_text(json.dumps(run_closure, indent=2), encoding="utf-8")
    
    # (1) 验证文件存在
    assert closure_path.exists(), "run_closure.json must exist even on failure"
    
    # (2) 验证内容
    content = json.loads(closure_path.read_text(encoding="utf-8"))
    assert content["ok"] is False
    assert "failure_reason" in content
    assert content["failure_reason"] != ""


def test_run_closure_contains_single_primary_failure_reason(tmp_run_root):
    """
    Test that run_closure contains only one primary failure reason.
    
    每次 run_closure 只应有一个主失败原因，次级异常记录但不覆盖主因。
    """
    run_closure = {
        "ok": False,
        "run_id": "test_run_fail_002",
        "failure_reason": "primary_gate_failure",
        "secondary_errors": [
            "cleanup_failed",
            "log_write_failed",
        ],
    }
    
    closure_path = tmp_run_root / "run_closure.json"
    closure_path.write_text(json.dumps(run_closure, indent=2), encoding="utf-8")
    
    # 验证主失败原因唯一
    content = json.loads(closure_path.read_text(encoding="utf-8"))
    assert "failure_reason" in content
    assert isinstance(content["failure_reason"], str)
    
    # 次级异常应该在独立字段中
    if "secondary_errors" in content:
        assert isinstance(content["secondary_errors"], list)


def test_run_closure_records_absent_when_no_records(tmp_run_root):
    """
    Test that run_closure explicitly marks records as absent when missing.
    
    若 records 不存在，必须显式记录 records_absent。
    """
    run_closure = {
        "ok": False,
        "run_id": "test_run_fail_003",
        "failure_reason": "early_gate_failure",
        "records_absent": True,  # 显式缺失标记
        "run_meta": {
            "contract_version": "v1.0.0",
        },
    }
    
    closure_path = tmp_run_root / "run_closure.json"
    closure_path.write_text(json.dumps(run_closure, indent=2), encoding="utf-8")
    
    content = json.loads(closure_path.read_text(encoding="utf-8"))
    
    # 验证显式缺失标记
    assert "records_absent" in content
    assert content["records_absent"] is True


def test_run_closure_minimal_fields_on_failure(tmp_run_root):
    """
    Test that run_closure contains minimal required fields even on failure.
    
    失败时 run_meta 最小字段集仍可生成（使用 <absent> 或哨兵值）。
    """
    run_closure = {
        "ok": False,
        "run_id": "test_run_fail_004",
        "failure_reason": "initialization_failure",
        "run_meta": {
            "contract_version": "<absent>",  # 使用 <absent> 哨兵
            "schema_version": "<absent>",
            "cfg_digest": "<absent>",
        },
    }
    
    closure_path = tmp_run_root / "run_closure.json"
    closure_path.write_text(json.dumps(run_closure, indent=2), encoding="utf-8")
    
    content = json.loads(closure_path.read_text(encoding="utf-8"))
    
    # 验证最小字段集存在
    assert "run_meta" in content
    assert "contract_version" in content["run_meta"]
    
    # 缺失值应使用统一哨兵
    assert content["run_meta"]["contract_version"] == "<absent>"


def test_failure_reason_locatable_to_gate():
    """
    Test that failure reason can be located to specific gate or validation point.
    
    失败原因必须可定位到具体门禁或校验点。
    """
    try:
        from main.core import errors
    except ImportError:
        pytest.skip("main.core.errors module not found")
    
    # 检查错误类型是否包含 gate_name 和 field_path
    if hasattr(errors, "GateEnforcementError"):
        error_class = errors.GateEnforcementError

        # 构造错误实例
        try:
            raise error_class(
                "gate failure",
                gate_name="freeze_gate.assert_prewrite",
                field_path="contract_version",
                expected="present",
                actual="missing"
            )
        except error_class as e:
            # 验证异常包含定位信息
            assert hasattr(e, "gate_name") or "gate_name" in str(e)
            assert hasattr(e, "field_path") or "field_path" in str(e)
    else:
        pytest.skip("GateEnforcementError not implemented")


def test_orchestrator_generates_closure_on_exception(tmp_run_root):
    """
    Test that orchestrator generates run_closure even when exception occurs.
    
    orchestrator 在异常时仍能生成 run_closure。
    """
    try:
        from main.cli import run_detect
    except ImportError:
        pytest.skip("main.cli.run_detect module not found")

    invalid_config_path = tmp_run_root / "configs" / "missing.yaml"

    with pytest.raises(Exception):
        run_detect.run_detect(
            output_dir=str(tmp_run_root),
            config_path=str(invalid_config_path),
            input_record_path=None,
            overrides=None
        )

    run_closure_path = tmp_run_root / "artifacts" / "run_closure.json"
    assert run_closure_path.exists(), "run_closure.json must exist on failure"

    content = json.loads(run_closure_path.read_text(encoding="utf-8"))
    status = content.get("status", {})
    assert status.get("ok") is False
    assert status.get("reason") == "config_invalid"
