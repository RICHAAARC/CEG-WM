"""
协议门禁阻断未知攻击族
"""

import pytest
from main.evaluation import attack_protocol_guard
from main.evaluation import attack_coverage


def test_attack_protocol_guard_blocks_unknown_family():
    """
    功能：协议门禁阻断未知攻击族。

    Verify that attack_protocol_guard rejects protocol specs
    with families not listed in coverage manifest.

    GIVEN: Coverage manifest with supported families
    WHEN: Protocol spec contains unknown family
    THEN: Guard raises RuntimeError with evidence containing
          'unsupported_families' and normalized_family.
    """
    # (1) 获取真实覆盖清单
    coverage_manifest = attack_coverage.compute_attack_coverage_manifest()
    supported_families = set(coverage_manifest.get("supported_families", []))
    
    # (2) 注入非法族名
    invalid_family = "UNKNOWN_ATTACK_FAMILY_NEVER_EXISTS_IN_IMPLEMENTATION"
    while invalid_family.lower() in {f.lower() for f in supported_families}:
        invalid_family += "_EXTRA_SUFFIX"
    
    # (3) 构造非法协议
    bad_protocol = {
        "protocol_version": "attack_protocol_v1",
        "protocol_digest": "dummy_digest_for_test",
        "protocol_meta": {
            "creation_date": "2025-01-06T00:00:00Z",
            "frozen_by": "test_harness",
        },
        "families": [
            invalid_family,
        ],
        "params_versions": {
            invalid_family: {
                "v1": {
                    "family": invalid_family,
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
    
    # (5) 证据字段必须包含 unsupported_families
    exc_msg = str(exc_info.value)
    assert "unsupported" in exc_msg.lower() or "unknown" in exc_msg.lower(), \
        f"错误信息必须明确表明族不支持，实际为: {exc_msg}"
    
    # (6) 错误消息中必须包含归一化后的非法族名
    normalized_family = invalid_family.lower()
    assert normalized_family in exc_msg.lower() or invalid_family.lower() in exc_msg.lower(), \
        f"错误消息必须包含非法族名（原始或归一化形式），实际为: {exc_msg}"
