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
        from main.core.errors import RecordBundleError
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    
    # 创建两个 records 文件，故意制造 contract_bound_digest 冲突
    record_1 = {
        "run_id": "test_run_001",
        "contract_bound_digest": "a" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "record_1",
    }
    
    record_2 = {
        "run_id": "test_run_001",
        "contract_bound_digest": "d" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    # 尝试构建 bundle
    with pytest.raises(RecordBundleError) as exc_info:
        records_bundle.close_records_bundle(records_dir)
    
    error_msg = str(exc_info.value)
    
    # 验证异常信息包含：
    # 1. 冲突字段名
    assert "contract_bound_digest" in error_msg
    
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
        "contract_bound_digest": "a" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "consistent_record_1",
    }
    
    record_2 = {
        "run_id": "test_run_002",
        "contract_bound_digest": "a" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "consistent_record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    artifacts_dir = tmp_run_root / "artifacts"

    # 构建 bundle
    manifest_path = records_bundle.close_records_bundle(
        records_dir,
        manifest_dir=artifacts_dir
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "bundle_canon_sha256" in manifest


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
    scanned_files = records_bundle._scan_record_files(records_dir, "records_manifest.json")

    # 验证仅包含 .json 和 .jsonl
    for filepath in scanned_files:
        assert filepath.suffix in {".json", ".jsonl"}
        assert filepath.name != "records_manifest.json"


def test_bundle_failure_does_not_leave_partial_output(tmp_run_root):
    """
    Test that bundle failure does not leave partial output files.
    
    bundle 失败不应落盘且无临时残留。
    """
    try:
        from main.core import records_bundle
        from main.core.errors import RecordBundleError
    except ImportError:
        pytest.skip("main.core.records_bundle module not found")
    
    records_dir = tmp_run_root / "records"
    manifest_path = tmp_run_root / "artifacts" / "records_manifest.json"
    
    # 创建冲突的 records
    record_1 = {
        "run_id": "test_run_004",
        "contract_bound_digest": "a" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "record_1",
    }
    
    record_2 = {
        "run_id": "test_run_004",
        "contract_bound_digest": "d" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "record_2",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1), encoding="utf-8")
    (records_dir / "record_2.json").write_text(json.dumps(record_2), encoding="utf-8")
    
    # 尝试构建 bundle（应该失败）
    try:
        records_bundle.close_records_bundle(
            records_dir,
            manifest_dir=manifest_path.parent
        )
    except RecordBundleError:
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
        "contract_bound_digest": "a" * 64,
        "whitelist_bound_digest": "b" * 64,
        "policy_path_semantics_bound_digest": "c" * 64,
        "event": "digest_test",
    }
    
    (records_dir / "record_1.json").write_text(json.dumps(record_1, sort_keys=True), encoding="utf-8")
    
    artifacts_dir = tmp_run_root / "artifacts"

    # 构建 bundle 两次
    manifest_path_1 = records_bundle.close_records_bundle(
        records_dir,
        manifest_dir=artifacts_dir
    )
    manifest_1 = json.loads(manifest_path_1.read_text(encoding="utf-8"))

    manifest_path_2 = records_bundle.close_records_bundle(
        records_dir,
        manifest_dir=artifacts_dir
    )
    manifest_2 = json.loads(manifest_path_2.read_text(encoding="utf-8"))

    assert manifest_1.get("bundle_canon_sha256") == manifest_2.get("bundle_canon_sha256")
