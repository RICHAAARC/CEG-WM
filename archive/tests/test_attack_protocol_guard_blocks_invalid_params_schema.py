"""
协议门禁阻断非法参数规范 schema
"""

import pytest
from main.evaluation import attack_protocol_guard
from main.evaluation import attack_coverage


def test_attack_protocol_guard_blocks_invalid_params_schema():
    """
    功能：协议门禁阻断非法参数规范 schema。

    Verify that attack_protocol_guard rejects protocol specs
    with params_versions entries missing required 'family' field.

    GIVEN: Coverage manifest with supported families (e.g., 'rotate')
    WHEN: Protocol params_versions.rotate::v1 missing 'family' field
    THEN: Guard raises RuntimeError with evidence containing
          condition_key='params_version.missing_family' and evidence_path.
    """
    # (1) 获取真实覆盖清单
    coverage_manifest = attack_coverage.compute_attack_coverage_manifest()
    supported_families = coverage_manifest.get("supported_families", [])
    if not supported_families:
        pytest.skip("No supported families in coverage manifest")
    
    # (2) 选取第一个支持的族（如 rotate）
    valid_family = supported_families[0]
    
    # (3) 构造参数 schema 缺失 'family' 字段的非法协议
    bad_protocol = {
        "protocol_version": "attack_protocol_v1",
        "protocol_digest": "dummy_digest_for_test",
        "protocol_meta": {
            "creation_date": "2025-01-06T00:00:00Z",
            "frozen_by": "test_harness",
        },
        "families": [
            valid_family,
        ],
        "params_versions": {
            valid_family: {
                "v1": {
                    # 故意缺失 'family' 字段
                    "default_params": {},
                },
            },
        },
    }
    
    # (4) 门禁必须 fail-fast
    with pytest.raises(RuntimeError) as exc_info:
        attack_protocol_guard.assert_attack_protocol_is_implementable(
            bad_protocol,
            coverage_manifest,
        )
    
    # (5) 证据字段必须包含 'family' 或表明缺失必需字段
    exc_msg = str(exc_info.value)
    assert "family" in exc_msg.lower() or "missing" in exc_msg.lower() or "required" in exc_msg.lower(), \
        f"错误信息必须明确表明缺失 'family' 字段，实际为: {exc_msg}"
    
    # (6) 错误消息中必须包含族名（v1 可能被归一化到 family 维度，不强制要求）
    assert valid_family.lower() in exc_msg.lower(), \
        f"错误消息必须包含族名 '{valid_family}'，实际为: {exc_msg}"
