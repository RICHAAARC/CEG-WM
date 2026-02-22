"""
测试阻断项 B1：评测报告锚点字段完整性审计（S-13）。

Test that audit_evaluation_report_schema correctly validates report anchor fields.
"""

import json
import tempfile
from pathlib import Path

import pytest
from scripts.audits import audit_evaluation_report_schema


class TestAuditEvaluationReportSchema:
    """评测报告锚点审计测试套件。"""

    def test_audit_returns_pass_with_complete_anchors(self, tmp_path):
        """测试：审计发现完整锚点时返回 PASS。"""
        # Arrange
        report = {
            "cfg_digest": "sha256_cfg",
            "plan_digest": "sha256_plan",
            "thresholds_digest": "sha256_thresh",
            "threshold_metadata_digest": "sha256_metadata",
            "impl_digest": "sha256_impl",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "sha256_proto",
            "policy_path": "policy://test",
            "metrics": {},
        }

        # 创建报告文件
        report_path = tmp_path / "outputs" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        # Act
        result = audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        assert result == 0  # Success

    def test_audit_returns_fail_when_anchor_missing(self, tmp_path, capsys):
        """测试：审计检测到缺失锚点时返回 FAIL。"""
        # Arrange
        report = {
            "cfg_digest": "sha256_cfg",
            "plan_digest": "sha256_plan",
            # Missing thresholds_digest!
            "impl_digest": "sha256_impl",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "sha256_proto",
            "policy_path": "policy://test",
        }

        report_path = tmp_path / "outputs" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        # Act
        result = audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        assert result == 1  # Failure
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["result"] == "FAIL"

    def test_audit_handles_anchors_nested_under_anchors_key(self, tmp_path):
        """测试：审计支持锚点嵌套在 'anchors' 子键下。"""
        # Arrange
        report = {
            "anchors": {
                "cfg_digest": "sha256_cfg",
                "plan_digest": "sha256_plan",
                "thresholds_digest": "sha256_thresh",
                "threshold_metadata_digest": "sha256_metadata",
                "impl_digest": "sha256_impl",
                "fusion_rule_version": "fusion_v1",
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "sha256_proto",
                "policy_path": "policy://test",
            },
            "metrics": {},
        }

        report_path = tmp_path / "artifacts" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        # Act
        result = audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        assert result == 0  # Should pass

    def test_audit_rejects_absent_digest_values(self, tmp_path, capsys):
        """测试：审计拒绝 <absent> 占位符作为有效值。"""
        # Arrange
        report = {
            "cfg_digest": "sha256_cfg",
            "plan_digest": "<absent>",  # Invalid placeholder!
            "thresholds_digest": "sha256_thresh",
            "threshold_metadata_digest": "sha256_metadata",
            "impl_digest": "sha256_impl",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "sha256_proto",
            "policy_path": "policy://test",
        }

        report_path = tmp_path / "outputs" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        # Act
        result = audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        assert result == 1  # Failure
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "plan_digest" in str(output["evidence"]["missing_fields"])

    def test_audit_handles_missing_report_file(self, tmp_path, capsys):
        """测试：审计在报告文件不存在时返回 N.A.（非阻断）。"""
        # Arrange - 空的 tmp_path，无报告文件

        # Act
        result = audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        assert result == 0  # Should be N.A. (non-blocking)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["result"] == "N.A."

    def test_audit_output_format_is_valid_json(self, tmp_path, capsys):
        """测试：审计输出符合 JSON 格式。"""
        # Arrange
        report = {
            "cfg_digest": "sha256_cfg",
            "plan_digest": "sha256_plan",
            "thresholds_digest": "sha256_thresh",
            "threshold_metadata_digest": "sha256_metadata",
            "impl_digest": "sha256_impl",
            "fusion_rule_version": "fusion_v1",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "sha256_proto",
            "policy_path": "policy://test",
        }

        report_path = tmp_path / "outputs" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        # Act
        audit_evaluation_report_schema.main(str(tmp_path))

        # Assert
        captured = capsys.readouterr()
        output = json.loads(captured.out)  # Should not raise JSONDecodeError
        assert "audit_id" in output
        assert "result" in output
