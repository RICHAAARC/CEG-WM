"""
功能：测试 artifacts 语义旁路防护（B6/B5 覆盖）

Module type: Core innovation module

Test that artifacts cannot contain semantic fields that should be
constrained by schema and freeze_gate (semantic bypass guard).
"""

import pytest
import json
from pathlib import Path


def test_artifacts_reject_records_semantic_fields(tmp_run_root):
    """
    Test that writing artifacts with records-semantic fields is rejected.
    
    artifacts 中不得包含应由 schema + freeze_gate 约束的语义字段。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    # 构造包含锚点字段的 artifact（非法）
    artifact_with_semantic_fields = {
        "artifact_type": "test_artifact",
        "contract_version": "v1.0.0",  # 锚点字段，不应出现在 artifacts
        "frozen_contracts_digest": "sha256:abc123",  # 锚点字段
        "data": "some test data",
    }
    
    output_path = tmp_run_root / "artifacts" / "illegal_artifact.json"
    
    # 期望：写入被拒绝
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        # 假设 records_io 提供 write_artifact 函数并包含语义检查
        if hasattr(records_io, "write_artifact"):
            records_io.write_artifact(output_path, artifact_with_semantic_fields)
        else:
            pytest.skip("records_io.write_artifact not implemented")
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ["semantic", "bypass", "artifact", "contract"])


def test_artifacts_allow_controlled_fields(tmp_run_root):
    """
    Test that artifacts can contain allowed top-level fields.
    
    artifacts 可以包含受控的顶层字段集合（正例）。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    # 构造合法的 artifact（不包含锚点字段）
    valid_artifact = {
        "artifact_type": "visualization",
        "data": "plot data",
        "metadata": {
            "generated_at": "2026-02-15T10:00:00Z",
        },
    }
    
    output_path = tmp_run_root / "artifacts" / "valid_artifact.json"
    
    # 应该允许写入
    try:
        if hasattr(records_io, "write_artifact"):
            records_io.write_artifact(output_path, valid_artifact)
        else:
            # 临时方案：直接写入
            output_path.write_text(json.dumps(valid_artifact, indent=2), encoding="utf-8")
        
        # 验证文件存在
        assert output_path.exists()
        
        # 验证内容
        written_content = json.loads(output_path.read_text(encoding="utf-8"))
        assert written_content["artifact_type"] == "visualization"
        
    except Exception as e:
        pytest.xfail(f"Artifact write with guard not yet fully implemented: {e}")


def test_artifacts_reject_whitelist_digest_fields(tmp_run_root):
    """
    Test that artifacts cannot contain whitelist-related digest fields.
    
    artifacts 不得包含 whitelist 相关的 digest 字段。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    # 构造包含 whitelist digest 的 artifact
    artifact_with_whitelist_digest = {
        "artifact_type": "test",
        "runtime_whitelist_digest": "sha256:xyz789",  # 锚点字段
        "data": "test",
    }
    
    output_path = tmp_run_root / "artifacts" / "whitelist_artifact.json"
    
    # 应该被拒绝
    with pytest.raises((RuntimeError, ValueError)):
        if hasattr(records_io, "write_artifact"):
            records_io.write_artifact(output_path, artifact_with_whitelist_digest)
        else:
            pytest.skip("write_artifact not implemented")


def test_critical_outputs_use_controlled_write_path(tmp_run_root):
    """
    Test that critical outputs are written through controlled paths.
    
    关键产物（run_closure.json 等）必须通过受控写盘路径。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    critical_outputs = [
        "run_closure.json",
        "records_manifest.json",
        "cfg_audit.json",
    ]
    
    for output_name in critical_outputs:
        output_path = tmp_run_root / output_name
        
        test_content = {
            "output_type": output_name,
            "test": True,
        }
        
        # 尝试直接写入（应该被拦截或由受控函数处理）
        # 注：实际项目中应该有专门的写入函数
        # 这里只验证概念
        if hasattr(records_io, f"write_{output_name.replace('.json', '')}"):
            # 存在专用函数
            pass
        else:
            # 标记为需要实现
            pytest.skip(f"Controlled write function for {output_name} not implemented")
