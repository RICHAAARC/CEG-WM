"""
文件目的：multi-protocol runner 与 compare 汇总工件的回归测试
Module type: General module

测试覆盖：
1. 多 protocol 生成独立 run_roots
2. compare_summary.json schema 校验
3. 审计脚本通过/失败路径
4. publish workflow 不被影响
"""

import json
from pathlib import Path

import pytest

# Import needed modules  
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
if str(_repo_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_repo_root))

from scripts.run_multi_protocol_evaluation import (
    _make_protocol_safe_key,
    _load_protocol_spec,
    _build_sub_run_root,
)
from scripts.audits.audit_protocol_compare_outputs_schema import (
    audit_protocol_compare_outputs_schema,
)


class TestMultiProtocolRunnerCreatesSeparateRunRoots:
    """测试：多 protocol 生成独立 run_roots"""

    def test_create_separate_run_roots_for_two_protocols(self, tmp_path):
        """
        Test that two different protocols generate separate run_root directories.
        """
        # Arrange: 创建两个最小 protocol fixtures
        protocol_dir = tmp_path / "protocols"
        protocol_dir.mkdir()

        protocol1_path = protocol_dir / "protocol_v1.yaml"
        protocol1_path.write_text(
            """
version: "attack_protocol_v1"
families:
  rotate:
    description: "rotation attacks"
    params_versions:
      v1:
        degrees: [5, 10]
"""
        )

        protocol2_path = protocol_dir / "protocol_v2.yaml"
        protocol2_path.write_text(
            """
version: "attack_protocol_v2"
families:
  resize:
    description: "resize attacks"
    params_versions:
      v1:
        scale_factors: [0.75, 0.9]
"""
        )

        run_root_base = tmp_path / "multi_run"
        run_root_base.mkdir()

        # Act: 加载两个 protocol specs 并生成 run_roots
        protocol1_spec = _load_protocol_spec(protocol1_path)
        protocol2_spec = _load_protocol_spec(protocol2_path)

        run_root1 = _build_sub_run_root(run_root_base, protocol1_spec, protocol1_path)
        run_root2 = _build_sub_run_root(run_root_base, protocol2_spec, protocol2_path)

        # Assert: run_roots 应该不同且都在 run_root_base 下
        assert run_root1 != run_root2, "Two protocols should generate different run_roots"
        assert str(run_root1).startswith(str(run_root_base)), f"{run_root1} should be under {run_root_base}"
        assert str(run_root2).startswith(str(run_root_base)), f"{run_root2} should be under {run_root_base}"
        assert "protocol_" in str(run_root1), "run_root should contain protocol prefix"
        assert "protocol_" in str(run_root2), "run_root should contain protocol prefix"


class TestCompareSummarySchemaMustContainRequiredAnchors:
    """测试：compare_summary.json schema 必须包含所有锚点字段"""

    def test_protocol_record_contains_required_fields(self, tmp_path):
        """
        Test that protocol records in compare_summary have all required fields.
        """
        # Arrange: 构造最小 compare_summary.json
        protocol_record = {
            "protocol_source_path": "/path/to/protocol_v1.yaml",
            "protocol_source_basename": "protocol_v1.yaml",
            "run_root": str(tmp_path / "run1"),
            "run_root_relative": "protocol_attack_protocol_v1_abc12345/run_20240101_010101",
            "status": "ok",
            "failure_reason": "ok",
            "protocol_id": "attack_protocol_digest_abc123",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "attack_protocol_digest_abc123",
            "all_audits_passed": True,
            "anchors": {
                "cfg_digest": "cfg_digest_abc",
                "plan_digest": "plan_digest_def",
                "thresholds_digest": "thresholds_digest_ghi",
                "threshold_metadata_digest": "threshold_metadata_digest_jkl",
                "impl_digest": "impl_digest_mno",
                "fusion_rule_version": "fusion_v1",
                "attack_coverage_digest": "attack_coverage_digest_pqr",
                "policy_path": "/policy/path",
                "tpr_at_fpr_primary": 0.95,
                "geo_available_rate": 0.85,
                "rescue_rate": 0.10,
                "reject_rate_total": 0.05,
            },
        }

        compare_summary = {
            "schema_version": "protocol_compare_v1",
            "created_at_utc": "2024-01-01T01:01:01Z",
            "run_root_base": str(tmp_path),
            "protocol_count": 1,
            "protocols": [protocol_record],
        }

        protocol_compare_dir = tmp_path / "outputs" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
        protocol_compare_dir.mkdir(parents=True)
        summary_path = protocol_compare_dir / "compare_summary.json"
        summary_path.write_text(json.dumps(compare_summary, indent=2))

        # Act: 运行审计
        repo_root = tmp_path
        result = audit_protocol_compare_outputs_schema(repo_root)

        # Assert: 审计应该 PASS
        assert result["result"] == "PASS", f"Audit should PASS but got {result['result']}: {result.get('evidence')}"
        assert "protocol_count" in result["evidence"]
        assert result["evidence"]["protocol_count"] == 1


class TestAuditProtocolCompareOutputsSchemaPasses:
    """测试：审计脚本对有效 compare_summary.json 必须 PASS"""

    def test_audit_passes_on_valid_compare_summary(self, tmp_path):
        """
        Test that audit PASSES on valid compare_summary.json.
        """
        # Arrange: 构造有效的 compare_summary.json
        multi_protocol_dir = tmp_path / "outputs" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
        multi_protocol_dir.mkdir(parents=True)

        protocol_records = [
            {
                "protocol_source_path": "/path/to/protocol_v1.yaml",
                "protocol_source_basename": "protocol_v1.yaml",
                "run_root": str(tmp_path / "run1"),
                "run_root_relative": "protocol_attack_protocol_v1_abc/run_20240101_010101",
                "status": "ok",
                "failure_reason": "ok",
                "protocol_id": "digest_abc",
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "digest_abc",
                "all_audits_passed": True,
                "anchors": {
                    "cfg_digest": "cfg_abc",
                    "plan_digest": "plan_abc",
                    "thresholds_digest": "thresholds_abc",
                    "threshold_metadata_digest": "threshold_metadata_abc",
                    "impl_digest": "impl_abc",
                    "fusion_rule_version": "fusion_v1",
                    "attack_coverage_digest": "coverage_abc",
                    "policy_path": "/policy/path",
                },
            },
        ]

        compare_summary = {
            "schema_version": "protocol_compare_v1",
            "created_at_utc": "2024-01-01T01:01:01Z",
            "run_root_base": str(tmp_path),
            "protocol_count": 1,
            "protocols": protocol_records,
        }

        summary_path = multi_protocol_dir / "compare_summary.json"
        summary_path.write_text(json.dumps(compare_summary, indent=2))

        # Act: 运行审计
        result = audit_protocol_compare_outputs_schema(tmp_path)

        # Assert
        assert result["result"] == "PASS", f"Expected PASS but got {result['result']}: {result['evidence']}"


class TestAuditProtocolCompareOutputsSchemaFailsOnDuplicateProtocolId:
    """测试：审计脚本对重复 protocol_id 必须 FAIL"""

    def test_audit_fails_on_duplicate_protocol_id(self, tmp_path):
        """
        Test that audit FAILS on duplicate protocol_id.
        """
        # Arrange: 构造有重复 protocol_id 的 compare_summary.json
        multi_protocol_dir = tmp_path / "outputs" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
        multi_protocol_dir.mkdir(parents=True)

        protocol_records = [
            {
                "protocol_source_path": "/path/to/protocol_v1.yaml",
                "protocol_source_basename": "protocol_v1.yaml",
                "run_root": str(tmp_path / "run1"),
                "run_root_relative": "protocol_v1_abc/run_1",
                "status": "ok",
                "failure_reason": "ok",
                "protocol_id": "duplicate_digest",  # 重复
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "duplicate_digest",
                "all_audits_passed": True,
                "anchors": {
                    "cfg_digest": "cfg_abc",
                    "plan_digest": "plan_abc",
                    "thresholds_digest": "thresholds_abc",
                    "threshold_metadata_digest": "threshold_metadata_abc",
                    "impl_digest": "impl_abc",
                    "fusion_rule_version": "fusion_v1",
                    "attack_coverage_digest": "coverage_abc",
                    "policy_path": "/policy/path",
                },
            },
            {
                "protocol_source_path": "/path/to/protocol_v2.yaml",
                "protocol_source_basename": "protocol_v2.yaml",
                "run_root": str(tmp_path / "run2"),
                "run_root_relative": "protocol_v2_def/run_2",
                "status": "ok",
                "failure_reason": "ok",
                "protocol_id": "duplicate_digest",  # 重复
                "attack_protocol_version": "attack_protocol_v2",
                "attack_protocol_digest": "duplicate_digest",
                "all_audits_passed": True,
                "anchors": {
                    "cfg_digest": "cfg_def",
                    "plan_digest": "plan_def",
                    "thresholds_digest": "thresholds_def",
                    "threshold_metadata_digest": "threshold_metadata_def",
                    "impl_digest": "impl_def",
                    "fusion_rule_version": "fusion_v1",
                    "attack_coverage_digest": "coverage_def",
                    "policy_path": "/policy/path",
                },
            },
        ]

        compare_summary = {
            "schema_version": "protocol_compare_v1",
            "created_at_utc": "2024-01-01T01:01:01Z",
            "run_root_base": str(tmp_path),
            "protocol_count": 2,
            "protocols": protocol_records,
        }

        summary_path = multi_protocol_dir / "compare_summary.json"
        summary_path.write_text(json.dumps(compare_summary, indent=2))

        # Act: 运行审计
        result = audit_protocol_compare_outputs_schema(tmp_path)

        # Assert: 审计应该 FAIL
        assert result["result"] == "FAIL", f"Expected FAIL for duplicate protocol_id but got {result['result']}"
        assert "Duplicate protocol_id" in str(result["evidence"]), "Evidence should mention duplicate protocol_id"


class TestAuditProtocolCompareOutputsSchemaIgnoresPytestArtifacts:
    """测试：审计脚本应忽略 pytest 临时目录污染。"""

    def test_audit_prefers_official_compare_summary_over_pytest_tmp(self, tmp_path):
        """
        Test that audit ignores pytest-like temporary compare_summary files.
        """
        official_dir = tmp_path / "outputs" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
        official_dir.mkdir(parents=True)
        official_summary = {
            "schema_version": "protocol_compare_v1",
            "created_at_utc": "2024-01-01T01:01:01Z",
            "run_root_base": str(tmp_path),
            "protocol_count": 1,
            "protocols": [
                {
                    "protocol_source_path": "/path/to/protocol_v1.yaml",
                    "protocol_source_basename": "protocol_v1.yaml",
                    "run_root": str(tmp_path / "run1"),
                    "run_root_relative": "protocol_v1_abc/run_1",
                    "status": "ok",
                    "failure_reason": "ok",
                    "protocol_id": "digest_official",
                    "attack_protocol_version": "attack_protocol_v1",
                    "attack_protocol_digest": "digest_official",
                    "all_audits_passed": True,
                    "anchors": {
                        "cfg_digest": "cfg_ok",
                        "plan_digest": "plan_ok",
                        "thresholds_digest": "thresholds_ok",
                        "threshold_metadata_digest": "threshold_metadata_ok",
                        "impl_digest": "impl_ok",
                        "fusion_rule_version": "fusion_v1",
                        "attack_coverage_digest": "coverage_ok",
                        "policy_path": "/policy/path",
                    },
                }
            ],
        }
        official_path = official_dir / "compare_summary.json"
        official_path.write_text(json.dumps(official_summary, indent=2), encoding="utf-8")

        polluted_dir = tmp_path / "outputs" / "pytesttmp_123" / "artifacts"
        polluted_dir.mkdir(parents=True)
        polluted_path = polluted_dir / "compare_summary.json"
        polluted_path.write_text("{ invalid json", encoding="utf-8")

        result = audit_protocol_compare_outputs_schema(tmp_path)

        assert result["result"] == "PASS", f"Expected PASS but got {result['result']}: {result.get('evidence')}"
        assert result["evidence"]["path"] == str(official_path)


class TestMakeProtocolSafeKey:
    """测试：protocol safe key 生成"""

    def test_safe_key_from_version_and_digest(self):
        """Test stable safe key generation from protocol spec."""
        protocol_spec = {
            "version": "attack_protocol_v1",
            "attack_protocol_digest": "abc1234567890def",
        }

        safe_key = _make_protocol_safe_key(protocol_spec)

        # safe_key 应该包含 version 和 digest 前缀
        assert isinstance(safe_key, str)
        assert len(safe_key) > 0
        assert "attack_protocol_v1" in safe_key.lower()
        assert "abc12345" in safe_key

    def test_safe_key_with_absent_digest(self):
        """Test safe key when digest is absent."""
        protocol_spec = {
            "version": "attack_protocol_v1",
            "attack_protocol_digest": "<absent>",
        }

        safe_key = _make_protocol_safe_key(protocol_spec)

        # 应该能生成有效的 key（不能包含特殊字符）
        assert isinstance(safe_key, str)
        assert len(safe_key) > 0
        assert "<" not in safe_key
        assert ">" not in safe_key
