"""
测试分组指标的规范化键。

Test that attack group keys are canonicalized and deterministic regardless of source.
"""

import pytest
from main.evaluation import metrics


class TestCanonicalConditionKey:
    """规范化分组键测试套件。"""

    def test_canonical_key_format_family_params_version(self):
        """测试：规范化键生成 family::params_version 格式。"""
        # Act
        key = metrics.canonical_condition_key("rotate", "v1")

        # Assert
        assert key == "rotate::v1"
        assert "::" in key
        parts = key.split("::")
        assert len(parts) == 2
        assert parts[0] == "rotate"
        assert parts[1] == "v1"

    def test_canonical_key_deterministic_order(self):
        """测试：相同输入总是生成相同的键。"""
        # Act - 多次调用相同参数
        key1 = metrics.canonical_condition_key("resize", "v2")
        key2 = metrics.canonical_condition_key("resize", "v2")
        key3 = metrics.canonical_condition_key("resize", "v2")

        # Assert - 完全相同
        assert key1 == key2 == key3

    def test_canonical_key_different_inputs_different_keys(self):
        """测试：不同输入生成不同的键。"""
        # Act
        key1 = metrics.canonical_condition_key("rotate", "v1")
        key2 = metrics.canonical_condition_key("rotate", "v2")
        key3 = metrics.canonical_condition_key("resize", "v1")

        # Assert
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_canonical_key_rejects_non_string_family(self):
        """测试：非字符串 family 被拒绝。"""
        # Act & Assert
        with pytest.raises(TypeError, match="family must be str"):
            metrics.canonical_condition_key(123, "v1")

    def test_canonical_key_rejects_non_string_params_version(self):
        """测试：非字符串 params_version 被拒绝。"""
        # Act & Assert
        with pytest.raises(TypeError, match="params_version must be str"):
            metrics.canonical_condition_key("rotate", 2.0)

    def test_build_attack_group_key_uses_canonical_key(self):
        """测试：build_attack_group_key 返回规范化的 family::params_version 格式。"""
        # Arrange
        record = {
            "attack_family": "crop",
            "attack_params_version": "v1",
            "score": 0.8,
            "label": True,
        }
        protocol_spec = {
            "family_field_candidates": ["attack_family", "attack.family"],
            "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        }

        # Act
        key = metrics.build_attack_group_key(record, protocol_spec)

        # Assert
        assert key == "crop::v1"
        assert isinstance(key, str)

    def test_build_attack_group_key_handles_absent_fields(self):
        """测试：缺失字段时使用 unknown 作为默认值。"""
        # Arrange
        record = {}  # Empty record
        protocol_spec = {
            "family_field_candidates": ["attack_family"],
            "params_version_field_candidates": ["attack_params_version"],
        }

        # Act
        key = metrics.build_attack_group_key(record, protocol_spec)

        # Assert - 默认值应为 unknown
        assert "unknown" in key
        assert "::" in key

    def test_compute_attack_group_metrics_returns_sorted_keys(self):
        """测试：compute_attack_group_metrics 返回按分组键排序的结果。"""
        # Arrange
        records = [
            {
                "attack_family": "resize",
                "attack_params_version": "v1",
                "content_evidence_payload": {"status": "ok", "score": 0.9},
                "label": True,
            },
            {
                "attack_family": "rotate",
                "attack_params_version": "v1",
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
            },
            {
                "attack_family": "crop",
                "attack_params_version": "v1",
                "content_evidence_payload": {"status": "ok", "score": 0.7},
                "label": False,
            },
        ]
        protocol_spec = {
            "family_field_candidates": ["attack_family"],
            "params_version_field_candidates": ["attack_params_version"],
        }

        # Act
        result = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)

        # Assert - 结果按 group_key 排序
        group_keys = [item.get("group_key") for item in result]
        assert group_keys == sorted(group_keys)  # Should be sorted

    def test_group_keys_in_result_are_canonical_format(self):
        """测试：返回结果中的 group_key 都是规范化格式。"""
        # Arrange
        records = [
            {
                "attack_family": "rotate",
                "attack_params_version": "v2",
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
            },
        ]
        protocol_spec = {
            "family_field_candidates": ["attack_family"],
            "params_version_field_candidates": ["attack_params_version"],
        }

        # Act
        result = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)

        # Assert
        assert len(result) > 0
        for group_metric in result:
            group_key = group_metric.get("group_key", "")
            assert "::" in group_key
            parts = group_key.split("::")
            assert len(parts) == 2


def test_metrics_group_keys_are_canonical_and_complete() -> None:
    """
    功能：验证分组键 canonical 且每组字段完整。

    Verify group keys are canonical and each group contains required report fields.

    Args:
        None.

    Returns:
        None.
    """
    records = [
        {
            "attack_family": "rotate",
            "attack_params_version": "v1",
            "content_evidence_payload": {"status": "ok", "score": 0.9},
            "label": True,
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.3},
            "decision": {"routing_decisions": {"rescue_triggered": True}},
        },
        {
            "attack_family": "rotate",
            "attack_params_version": "v1",
            "content_evidence_payload": {"status": "ok", "score": 0.1},
            "label": False,
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.2},
            "decision": {"routing_decisions": {"rescue_triggered": False}},
        },
    ]
    protocol_spec = {
        "family_field_candidates": ["attack_family"],
        "params_version_field_candidates": ["attack_params_version"],
    }

    grouped = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)
    assert len(grouped) >= 1

    required_fields = {
        "group_key",
        "tpr_at_fpr_primary",
        "geo_available_rate",
        "rescue_rate",
        "reject_rate_by_reason",
    }
    for item in grouped:
        assert "::" in item.get("group_key", "")
        assert required_fields.issubset(set(item.keys()))

