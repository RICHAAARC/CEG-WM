"""
测试评测报告锚点字段完整性。

Test that evaluation report contains all required anchor fields for paper reproducibility.
"""

import json
import pytest
from pathlib import Path

from main.evaluation import report_builder


class TestEvaluationReportAnchors:
    """论文级报告锚点字段测试套件。"""

    def test_build_report_with_all_anchors_present(self):
        """测试：所有锚点字段都成功填充到报告中。"""
        # Arrange
        cfg_digest = "sha256_cfg_123"
        plan_digest = "sha256_plan_456"
        thresholds_digest = "sha256_thresh_789"
        threshold_metadata_digest = "sha256_metadata_abc"
        impl_digest = "sha256_impl_def"
        fusion_rule_version = "fusion_v1"
        attack_protocol_version = "attack_protocol_v1"
        attack_protocol_digest = "sha256_proto_ghi"
        policy_path = "policy://test_policy"

        metrics_overall = {
            "tpr_at_fpr_primary": 0.95,
            "fpr_empirical": 0.05,
            "n_total": 1000,
            "n_accepted": 900,
        }
        metrics_by_condition = [
            {"group_key": "rotate::v1", "tpr_at_fpr_primary": 0.92},
        ]

        # Act
        report = report_builder.build_evaluation_report(
            cfg_digest=cfg_digest,
            plan_digest=plan_digest,
            thresholds_digest=thresholds_digest,
            threshold_metadata_digest=threshold_metadata_digest,
            impl_digest=impl_digest,
            fusion_rule_version=fusion_rule_version,
            attack_protocol_version=attack_protocol_version,
            attack_protocol_digest=attack_protocol_digest,
            policy_path=policy_path,
            metrics_overall=metrics_overall,
            metrics_by_attack_condition=metrics_by_condition,
            strict_anchor_validation=False,  # Disable strict validation for this test
        )

        # Assert - 所有锚点字段都在报告中
        assert report["cfg_digest"] == cfg_digest
        assert report["plan_digest"] == plan_digest
        assert report["thresholds_digest"] == thresholds_digest
        assert report["threshold_metadata_digest"] == threshold_metadata_digest
        assert report["impl_digest"] == impl_digest
        assert report["fusion_rule_version"] == fusion_rule_version
        assert report["attack_protocol_version"] == attack_protocol_version
        assert report["attack_protocol_digest"] == attack_protocol_digest
        assert report["policy_path"] == policy_path

        # Assert - anchors 子结构也包含这些字段
        anchors = report.get("anchors", {})
        assert anchors["cfg_digest"] == cfg_digest
        assert anchors["plan_digest"] == plan_digest
        assert anchors["attack_protocol_version"] == attack_protocol_version

    def test_strict_validation_fails_on_missing_cfg_digest(self):
        """测试：启用严格验证，cfg_digest 缺失时 FAIL。"""
        # Arrange
        metrics_overall = {"tpr_at_fpr_primary": 0.95}
        metrics_by_condition = []

        # Act & Assert
        with pytest.raises(RuntimeError, match="cfg_digest.*missing or empty"):
            report_builder.build_evaluation_report(
                cfg_digest=None,  # Missing!
                plan_digest="sha256_plan",
                thresholds_digest="sha256_thresh",
                threshold_metadata_digest="sha256_metadata",
                impl_digest="sha256_impl",
                fusion_rule_version="fusion_v1",
                attack_protocol_version="attack_protocol_v1",
                attack_protocol_digest="sha256_proto",
                policy_path="policy://test",
                metrics_overall=metrics_overall,
                metrics_by_attack_condition=metrics_by_condition,
                strict_anchor_validation=True,  # Enable strict validation
            )

    def test_strict_validation_fails_on_empty_string_digest(self):
        """测试：启用严格验证，空字符串 digest 被视为缺失。"""
        # Arrange
        metrics_overall = {"tpr_at_fpr_primary": 0.95}
        metrics_by_condition = []

        # Act & Assert
        with pytest.raises(RuntimeError, match="plan_digest.*missing or empty"):
            report_builder.build_evaluation_report(
                cfg_digest="sha256_cfg",
                plan_digest="",  # Empty string!
                thresholds_digest="sha256_thresh",
                threshold_metadata_digest="sha256_metadata",
                impl_digest="sha256_impl",
                fusion_rule_version="fusion_v1",
                attack_protocol_version="attack_protocol_v1",
                attack_protocol_digest="sha256_proto",
                policy_path="policy://test",
                metrics_overall=metrics_overall,
                metrics_by_attack_condition=metrics_by_condition,
                strict_anchor_validation=True,
            )

    def test_strict_validation_fails_on_missing_attack_protocol_version(self):
        """测试：启用严格验证，attack_protocol_version 缺失时 FAIL。"""
        # Arrange
        metrics_overall = {"tpr_at_fpr_primary": 0.95}
        metrics_by_condition = []

        # Act & Assert
        with pytest.raises(RuntimeError, match="attack_protocol_version"):
            report_builder.build_evaluation_report(
                cfg_digest="sha256_cfg",
                plan_digest="sha256_plan",
                thresholds_digest="sha256_thresh",
                threshold_metadata_digest="sha256_metadata",
                impl_digest="sha256_impl",
                fusion_rule_version="fusion_v1",
                attack_protocol_version=None,  # Missing!
                attack_protocol_digest="sha256_proto",
                policy_path="policy://test",
                metrics_overall=metrics_overall,
                metrics_by_attack_condition=metrics_by_condition,
                strict_anchor_validation=True,
            )

    def test_report_contains_evaluation_version(self):
        """测试：报告包含 evaluation_version 字段用于版本控制。"""
        # Arrange
        cfg_digest = "sha256_cfg"
        plan_digest = "sha256_plan"
        metrics_overall = {"tpr_at_fpr_primary": 0.95}
        metrics_by_condition = []

        # Act
        report = report_builder.build_evaluation_report(
            cfg_digest=cfg_digest,
            plan_digest=plan_digest,
            thresholds_digest="sha256_thresh",
            threshold_metadata_digest="sha256_metadata",
            impl_digest="sha256_impl",
            fusion_rule_version="fusion_v1",
            attack_protocol_version="attack_protocol_v1",
            attack_protocol_digest="sha256_proto",
            policy_path="policy://test",
            metrics_overall=metrics_overall,
            metrics_by_attack_condition=metrics_by_condition,
            strict_anchor_validation=False,
        )

        # Assert
        assert "evaluation_version" in report
        assert isinstance(report["evaluation_version"], str)
        assert len(report["evaluation_version"]) > 0

    def test_disabling_strict_validation_uses_absent_defaults(self):
        """测试：禁用严格验证时，缺失字段使用 <absent> 默认值。"""
        # Arrange
        metrics_overall = {"tpr_at_fpr_primary": 0.95}
        metrics_by_condition = []

        # Act
        report = report_builder.build_evaluation_report(
            cfg_digest=None,
            plan_digest="sha256_plan",
            thresholds_digest=None,
            threshold_metadata_digest=None,
            impl_digest="sha256_impl",
            fusion_rule_version=None,
            attack_protocol_version="attack_protocol_v1",
            attack_protocol_digest=None,
            policy_path="policy://test",
            metrics_overall=metrics_overall,
            metrics_by_attack_condition=metrics_by_condition,
            strict_anchor_validation=False,  # Disable strict validation
        )

        # Assert - 缺失字段被填充为 <absent>
        assert report["cfg_digest"] == "<absent>"
        assert report["thresholds_digest"] == "<absent>"
        assert report["threshold_metadata_digest"] == "<absent>"
        assert report["attack_protocol_digest"] == "<absent>"
        assert report["fusion_rule_version"] == "<absent>"

        # Assert - 存在的字段保留原值
        assert report["plan_digest"] == "sha256_plan"
        assert report["impl_digest"] == "sha256_impl"
        assert report["attack_protocol_version"] == "attack_protocol_v1"
