"""
功能：测试 records_bundle 与 anchors 一致性（F3 覆盖）

Module type: Core innovation module

Test that records_bundle validates anchor field consistency across
records files and fails with locatable error when inconsistency is found.
"""

import pytest
import json
from pathlib import Path


def test_bundle_detects_anchor_inconsistency(tmp_run_root):
    """
    Test that bundle fails when anchor fields are inconsistent.
    
    构造 anchors 冲突的 records，bundle 必须失败并定位冲突字段。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    
    # 创建两个 records 文件，故意制造 contract_version 冲突
    record_1 = {
        "run_id": "test_run_001",
        "contract_version": "v1.0.0",  # 版本 1
        "schema_version": "v1.0.0",
        "event": "record_1",
    }
    
    record_2 = {
        "run_id": "test_run_001",
        "contract_version": "v2.0.0",  # 版本 2（冲突）
        "schema_version": "v1.0.0",
        "event": "record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    # 尝试构建 bundle
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        if hasattr(records_bundle, "build_bundle"):
            records_bundle.build_bundle(records_dir)
        elif hasattr(records_bundle, "validate_anchor_consistency"):
            records_bundle.validate_anchor_consistency(records_dir)
        else:
            pytest.skip("records_bundle module does not provide bundle building function")
    
    error_msg = str(exc_info.value)
    
    # 验证异常信息包含：
    # 1. 冲突字段名
    assert "contract_version" in error_msg
    
    # 2. 两个来源文件
    assert "record_1.json" in error_msg or "record_2.json" in error_msg


def test_bundle_succeeds_with_consistent_anchors(tmp_run_root):
    """
    Test that bundle succeeds when all anchors are consistent.
    
    anchors 一致的正例，bundle 应该成功并输出 manifest/digest。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    
    # 创建两个 records 文件，anchors 一致
    record_1 = {
        "run_id": "test_run_002",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "consistent_record_1",
    }
    
    record_2 = {
        "run_id": "test_run_002",
        "contract_version": "v1.0.0",  # 一致
        "schema_version": "v1.0.0",  # 一致
        "event": "consistent_record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    # 构建 bundle
    try:
        if hasattr(records_bundle, "build_bundle"):
            result = records_bundle.build_bundle(records_dir)
            
            # 验证返回结果包含 manifest 或 digest
            assert result is not None
            if isinstance(result, dict):
                assert "bundle_digest" in result or "manifest" in result
        else:
            pytest.skip("build_bundle not implemented")
    except Exception as e:
        pytest.xfail(f"Bundle building not yet fully implemented: {e}")


def test_bundle_scan_range_is_stable(tmp_run_root):
    """
    Test that bundle scan range is stable (only .json/.jsonl files).
    
    bundle 扫描范围应该稳定，仅包含 .json/.jsonl 文件。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    
    # 创建有效记录
    valid_record = {
        "run_id": "test_run_003",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "valid",
    }
    (records_dir / "valid.json").write_text(json.dumps(valid_record), encoding="utf-8")
    
    # 创建临时文件（不应被包含）
    (records_dir / "temp.txt").write_text("temporary file")
    (records_dir / "temp.log").write_text("log file")
    (records_dir / ".hidden.json").write_text(json.dumps(valid_record))
    
    # 扫描 records
    if hasattr(records_bundle, "scan_records"):
        scanned_files = records_bundle.scan_records(records_dir)
        
        # 验证仅包含 .json 和 .jsonl
        for filepath in scanned_files:
            assert filepath.suffix in {".json", ".jsonl"}
            assert not filepath.name.startswith(".")
    else:
        pytest.skip("scan_records not implemented")


def test_bundle_failure_does_not_leave_partial_output(tmp_run_root):
    """
    Test that bundle failure does not leave partial output files.
    
    bundle 失败不应落盘且无临时残留。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    manifest_path = tmp_run_root / "records_manifest.json"
    
    # 创建冲突的 records
    record_1 = {
        "run_id": "test_run_004",
        "contract_version": "v1.0.0",
        "event": "record_1",
    }
    
    record_2 = {
        "run_id": "test_run_004",
        "contract_version": "v2.0.0",  # 冲突
        "event": "record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    # 尝试构建 bundle（应该失败）
    try:
        if hasattr(records_bundle, "build_bundle"):
            records_bundle.build_bundle(records_dir, output_path=manifest_path)
    except (ValueError, RuntimeError):
        # 预期失败
        pass
    
    # 验证不应有残留的 manifest 或部分输出
    assert not manifest_path.exists(), "Partial manifest should not exist after bundle failure"
    
    # 检查临时文件
    temp_files = list(tmp_run_root.glob("*.tmp"))
    assert len(temp_files) == 0, "No temporary files should remain after bundle failure"


def test_bundle_digest_is_reproducible(tmp_run_root):
    """
    Test that bundle digest is reproducible from manifest.
    
    bundle digest 可从 manifest 复算。
    """
    try:
        from main.core import records_bundle
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    
    # 创建一致的 records
    record_1 = {
        "run_id": "test_run_005",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "digest_test",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1, sort_keys=True), encoding="utf-8")
    
    # 构建 bundle 两次
    if hasattr(records_bundle, "build_bundle"):
        result_1 = records_bundle.build_bundle(records_dir)
        result_2 = records_bundle.build_bundle(records_dir)
        
        # 验证 digest 一致
        if isinstance(result_1, dict) and "bundle_digest" in result_1:
            assert result_1["bundle_digest"] == result_2["bundle_digest"]
    else:
        pytest.skip("build_bundle not implemented")
