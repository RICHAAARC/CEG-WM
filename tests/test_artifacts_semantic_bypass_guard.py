"""
artifacts 语义旁路防护测试
"""

import pytest
import json
from pathlib import Path


def _prepare_fact_sources(tmp_run_root: Path):
    """
    功能：准备写盘事实源上下文。

    Prepare fact sources for artifact write tests.

    Args:
        tmp_run_root: Temporary run root directory.

    Returns:
        Tuple of (contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir).

    Raises:
        TypeError: If tmp_run_root is invalid.
    """
    if not isinstance(tmp_run_root, Path):
        # tmp_run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("tmp_run_root must be Path")

    from main.core.contracts import load_frozen_contracts
    from main.policy.runtime_whitelist import load_runtime_whitelist, load_policy_path_semantics
    from main.core.injection_scope import load_injection_scope_manifest

    contracts = load_frozen_contracts()
    whitelist = load_runtime_whitelist()
    semantics = load_policy_path_semantics()
    injection_scope_manifest = load_injection_scope_manifest()
    records_dir = tmp_run_root / "records"
    artifacts_dir = tmp_run_root / "artifacts"
    logs_dir = tmp_run_root / "logs"
    return contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir


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
    
    from main.core.errors import RecordsWritePolicyError

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)

    # 期望：写入被拒绝
    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest
    ):
        with pytest.raises(RecordsWritePolicyError) as exc_info:
            records_io.write_artifact_json(str(output_path), artifact_with_semantic_fields)
    
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
    
    output_path = tmp_run_root / "artifacts" / "path_audits" / "valid_artifact.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)

    # 应该允许写入
    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest
    ):
        records_io.write_artifact_json(str(output_path), valid_artifact)

    # 验证文件存在
    assert output_path.exists()

    # 验证内容
    written_content = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_content["artifact_type"] == "visualization"


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
        "whitelist_bound_digest": "sha256:xyz789",  # 锚点字段
        "data": "test",
    }
    
    output_path = tmp_run_root / "artifacts" / "whitelist_artifact.json"
    
    from main.core.errors import RecordsWritePolicyError

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)

    # 应该被拒绝
    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest
    ):
        with pytest.raises(RecordsWritePolicyError):
            records_io.write_artifact_json(str(output_path), artifact_with_whitelist_digest)


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
    
    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest
    ):
        for output_name in critical_outputs:
            if output_name == "cfg_audit.json":
                output_path = artifacts_dir / "cfg_audit" / "cfg_audit.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                test_content = {
                    "config_path": "configs/default.yaml",
                    "cfg_digest": "a" * 64
                }
            else:
                output_path = artifacts_dir / output_name
                test_content = {
                    "output_type": output_name,
                    "test": True,
                }
            records_io.write_artifact_json(str(output_path), test_content)

    for output_name in critical_outputs:
        if output_name == "cfg_audit.json":
            output_path = artifacts_dir / "cfg_audit" / "cfg_audit.json"
        else:
            output_path = artifacts_dir / output_name
        assert output_path.exists()
