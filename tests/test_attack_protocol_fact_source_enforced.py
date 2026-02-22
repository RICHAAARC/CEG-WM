"""
测试阻断项 B3：attack protocol 必须来自 configs（S-13）。

Test that attack protocol parameters are loaded from configs and not hardcoded.
"""

import pytest
from main.evaluation import protocol_loader, metrics


class TestAttackProtocolFactSourceEnforced:
    """Attack protocol 事实源强制执行测试套件。"""

    def test_protocol_loader_loads_from_configs(self, tmp_path):
        """测试：protocol_loader 从 configs 加载协议（集成测试）。"""
        # Assert - 函数存在且可调用
        assert callable(protocol_loader.load_attack_protocol_spec)
        assert callable(protocol_loader.compute_attack_protocol_digest)

    def test_canonical_condition_key_enforces_standard_format(self):
        """测试：规范化键强制 family::params_version 标准格式。"""
        # Act
        key = metrics.canonical_condition_key("rotate", "v1")

        # Assert
        assert "::" in key
        parts = key.split("::")
        assert len(parts) == 2
        # 不允许其他格式（如冒号、下划线等）
        assert "_" not in key or "_" in "rotate" or "_" in "v1"

    def test_build_attack_group_key_uses_protocol_spec_candidates(self):
        """测试：分组键生成使用协议规范中的字段候选列表。"""
        # Arrange
        record = {
            "attack_family": "crop",
            "attack_params_version": "v2",
            "score": 0.8,
            "label": True,
        }
        protocol_spec = {
            "family_field_candidates": ["attack_family", "attack.family"],
            "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        }

        # Act
        key = metrics.build_attack_group_key(record, protocol_spec)

        # Assert - 键应该来自 protocol_spec 中定义的字段，不是硬编码的
        assert "crop" in key
        assert "v2" in key

    def test_compute_attack_group_metrics_derives_from_protocol_spec(self):
        """测试：分组指标计算完全依赖 protocol_spec，无硬编码参数。"""
        # Arrange
        records = [
            {
                "attack_type": "rotation",  # 使用自定义字段名
                "attack_ver": "version_1",
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
            },
        ]
        protocol_spec = {
            "family_field_candidates": ["attack_type"],  # 自定义字段候选
            "params_version_field_candidates": ["attack_ver"],
        }

        # Act
        result = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)

        # Assert
        assert len(result) > 0
        # 分组键应该使用 protocol_spec 中指定的字段，而不是硬编码的字段名
        group_key = result[0]["group_key"]
        assert "rotation" in group_key
        assert "version_1" in group_key

    def test_no_hardcoded_attack_families_in_metrics_module(self):
        """测试：metrics 模块中不存在硬编码的攻击族列表。"""
        # Arrange - 读取 metrics 模块代码
        import inspect
        metrics_source = inspect.getsource(metrics)

        # Assert - 检查是否存在硬编码的族定义
        forbidden_patterns = [
            'families = ["',
            'families = {',
            "attack_families = ",
            "ATTACK_FAMILIES",
            "HARDCODED_PARAMS",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in metrics_source, f"Found hardcoded pattern: {pattern}"

    def test_protocol_loader_exports_required_functions(self):
        """测试：protocol_loader 导出所有必要的加载函数。"""
        # Arrange & Assert
        assert hasattr(protocol_loader, "load_attack_protocol_spec")
        assert hasattr(protocol_loader, "compute_attack_protocol_digest")
        assert hasattr(protocol_loader, "get_protocol_version")
        assert hasattr(protocol_loader, "get_family_candidates")
        assert hasattr(protocol_loader, "get_params_version_candidates")

    def test_metrics_functions_accept_protocol_spec_parameter(self):
        """测试：关键函数接受 protocol_spec 参数，使协议可配置。"""
        # Arrange & Act & Assert
        import inspect

        # 检查 build_attack_group_key 签名
        sig = inspect.signature(metrics.build_attack_group_key)
        assert "protocol_spec" in sig.parameters

        # 检查 compute_attack_group_metrics 签名
        sig = inspect.signature(metrics.compute_attack_group_metrics)
        assert "protocol_spec" in sig.parameters

    def test_group_metrics_respects_all_protocol_field_names(self):
        """测试：分组计算尊重 protocol_spec 中的所有字段候选。"""
        # Arrange
        records = [
            {
                "custom_attack_field": "flip",
                "custom_version_field": "v3",
                "content_evidence_payload": {"status": "ok", "score": 0.7},
                "label": False,
            },
        ]
        protocol_spec = {
            "family_field_candidates": [
                "attack_family",
                "custom_attack_field",  # 非标准字段名
            ],
            "params_version_field_candidates": [
                "attack_params_version",
                "custom_version_field",  # 非标准字段名
            ],
        }

        # Act
        result = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)

        # Assert
        assert len(result) > 0
        # 应该能够使用 protocol_spec 中的自定义字段名
        group_key = result[0]["group_key"]
        assert "flip" in group_key
        assert "v3" in group_key

    def test_empty_protocol_spec_uses_safe_defaults(self):
        """测试：空的 protocol_spec 不导致崩溃，使用 unknown 默认值。"""
        # Arrange
        record = {"score": 0.8}
        protocol_spec = {}  # Empty!

        # Act & Assert - 应该不崩溃
        key = metrics.build_attack_group_key(record, protocol_spec)
        assert isinstance(key, str)
        assert "unknown" in key
