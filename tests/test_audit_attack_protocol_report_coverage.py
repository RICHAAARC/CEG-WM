"""
File purpose: audit_attack_protocol_report_coverage 回归测试集。
Module type: General module

测试覆盖：
1. 协议声明条件提取（format、结构、空集）
2. 报告条件提取（metrics_by_attack_condition field access）
3. 对齐检查（PASS、缺失、多报）
4. 审计脚本端到端（SKIP when report absent、result json format）
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# 导入审计脚本（假设可直接导入）
import sys
from pathlib import Path as PathlibPath

test_dir = PathlibPath(__file__).resolve().parent
scripts_dir = test_dir.parent / "scripts" / "audits"
sys.path.insert(0, str(scripts_dir))

import audit_attack_protocol_report_coverage as audit_module


class TestExtractDeclaredConditions:
    """
    功能：提取协议声明条件的单元测试。
    """

    def test_extract_from_flat_params_versions(self):
        """
        Scenario: Protocol spec contains flat params_versions dict with condition keys.

        Expected: Returns sorted list of all condition keys.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {
                "rotate::v1": {"family": "rotate", "params": {}},
                "crop::v1": {"family": "crop", "params": {}},
                "resize::v1": {"family": "resize", "params": {}},
            },
        }

        conditions = audit_module.extract_declared_conditions(protocol_spec)

        assert isinstance(conditions, list)
        assert len(conditions) == 3
        assert "rotate::v1" in conditions
        assert "crop::v1" in conditions
        assert "resize::v1" in conditions
        # 验证排序
        assert conditions == sorted(conditions)

    def test_extract_from_nested_families_structure(self):
        """
        Scenario: Protocol spec uses nested families/params_versions structure.

        Expected: Constructs condition keys and returns sorted list.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "families": {
                "rotate": {
                    "description": "rotation attacks",
                    "params_versions": {
                        "v1": {"degrees": [5, 10, 15]},
                        "v2": {"degrees": [30, 45]},
                    },
                },
                "jpeg": {
                    "description": "jpeg compression",
                    "params_versions": {
                        "v1": {"quality": [95, 85, 75]},
                    },
                },
            },
        }

        conditions = audit_module.extract_declared_conditions(protocol_spec)

        assert "rotate::v1" in conditions
        assert "rotate::v2" in conditions
        assert "jpeg::v1" in conditions
        assert len(conditions) == 3
        assert conditions == sorted(conditions)

    def test_empty_protocol_spec(self):
        """
        Scenario: Protocol spec is empty dict.

        Expected: Returns empty list.
        """
        protocol_spec = {"version": "attack_protocol_v1"}

        conditions = audit_module.extract_declared_conditions(protocol_spec)

        assert isinstance(conditions, list)
        assert len(conditions) == 0

    def test_type_error_on_non_dict(self):
        """
        Scenario: Input is not dict.

        Expected: Raises TypeError.
        """
        with pytest.raises(TypeError, match="protocol_spec must be dict"):
            audit_module.extract_declared_conditions("not a dict")


class TestExtractReportedConditions:
    """
    功能：从评测报告提取已报告条件的单元测试。
    """

    def test_extract_from_valid_report(self):
        """
        Scenario: Report contains metrics_by_attack_condition with valid group_key values.

        Expected: Returns sorted list of group_key values.
        """
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [
                {"group_key": "crop::v1", "n_total": 120},
                {"group_key": "gaussian_blur::v1", "n_total": 100},
                {"group_key": "rotate::v1", "n_total": 150},
            ],
        }

        conditions = audit_module.extract_reported_conditions(report)

        assert len(conditions) == 3
        assert "crop::v1" in conditions
        assert "gaussian_blur::v1" in conditions
        assert "rotate::v1" in conditions
        assert conditions == sorted(conditions)

    def test_missing_metrics_by_attack_condition_field(self):
        """
        Scenario: Report lacks metrics_by_attack_condition field.

        Expected: Returns empty list (not an error; report incomplete).
        """
        report = {"report_type": "evaluation_report"}

        conditions = audit_module.extract_reported_conditions(report)

        assert isinstance(conditions, list)
        assert len(conditions) == 0

    def test_malformed_metrics_list(self):
        """
        Scenario: metrics_by_attack_condition is not a list.

        Expected: Returns empty list (graceful handling).
        """
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": "not a list",
        }

        conditions = audit_module.extract_reported_conditions(report)

        assert len(conditions) == 0

    def test_duplicate_group_keys(self):
        """
        Scenario: metrics_by_attack_condition contains duplicate group_key values.

        Expected: Returns deduplicated sorted list.
        """
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [
                {"group_key": "rotate::v1", "n_total": 100},
                {"group_key": "rotate::v1", "n_total": 50},  # 重复
                {"group_key": "crop::v1", "n_total": 80},
            ],
        }

        conditions = audit_module.extract_reported_conditions(report)

        assert len(conditions) == 2
        assert conditions == sorted(set(conditions))


class TestAuditEquality:
    """
    功能：协议—报告对齐检查的集成测试。
    """

    def test_pass_exact_match(self):
        """
        Scenario: Declared and reported conditions match exactly.

        Expected: Audit returns PASS.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {
                "rotate::v1": {},
                "crop::v1": {},
            },
        }
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [
                {"group_key": "crop::v1"},
                {"group_key": "rotate::v1"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 创建 configs/attack_protocol.yaml（mock）
            (tmpdir_path / "configs").mkdir()
            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            # 创建 evaluation_report.json
            report_path = tmpdir_path / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            result = audit_module.audit_attack_protocol_report_coverage(tmpdir_path)

            assert result["result"] == "PASS"
            assert result["evidence"]["missed_conditions"] == []
            assert result["evidence"]["extra_reported_conditions"] == []

    def test_fail_missed_conditions(self):
        """
        Scenario: Declared in protocol but not in report (missed).

        Expected: Audit returns FAIL with missed_conditions list.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {
                "rotate::v1": {},
                "crop::v1": {},
                "resize::v1": {},  # 缺失的
            },
        }
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [
                {"group_key": "crop::v1"},
                {"group_key": "rotate::v1"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            report_path = tmpdir_path / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            result = audit_module.audit_attack_protocol_report_coverage(tmpdir_path)

            assert result["result"] == "FAIL"
            assert "resize::v1" in result["evidence"]["missed_conditions"]
            assert result["evidence"]["extra_reported_conditions"] == []

    def test_fail_extra_reported_conditions(self):
        """
        Scenario: Reported in evaluation but not declared in protocol.

        Expected: Audit returns FAIL with extra_reported_conditions list.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {
                "rotate::v1": {},
            },
        }
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [
                {"group_key": "rotate::v1"},
                {"group_key": "undeclared::v1"},  # 多报的
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            report_path = tmpdir_path / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            result = audit_module.audit_attack_protocol_report_coverage(tmpdir_path)

            assert result["result"] == "FAIL"
            assert result["evidence"]["missed_conditions"] == []
            assert "undeclared::v1" in result["evidence"]["extra_reported_conditions"]

    def test_skip_report_missing(self):
        """
        Scenario: evaluation_report.json does not exist.

        Expected: Audit returns SKIP status (not error).
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {
                "rotate::v1": {},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            # 不创建 evaluation_report.json

            result = audit_module.audit_attack_protocol_report_coverage(tmpdir_path)

            assert result["result"] == "N.A."
            assert result["severity"] == "NON_BLOCK"
            assert "not found" in result["evidence"]["status"].lower()


class TestAuditOutputFormat:
    """
    功能：审计输出 JSON 格式的一致性校验。
    """

    def test_result_dict_structure(self):
        """
        Scenario: Audit returns result dict with required fields.

        Expected: All required fields present and properly typed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_spec = {"version": "v1", "params_versions": {}}
            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            result = audit_module.audit_attack_protocol_report_coverage(tmpdir_path)

            # 校验必要字段
            assert "audit_id" in result
            assert "gate_name" in result
            assert "category" in result
            assert "severity" in result
            assert "result" in result
            assert "rule" in result
            assert "evidence" in result

            # 类型检查
            assert isinstance(result["audit_id"], str)
            assert isinstance(result["evidence"], dict)
            assert isinstance(result["evidence"]["declared_conditions"], list)
            assert isinstance(result["evidence"]["reported_conditions"], list)
            assert isinstance(result["evidence"]["missed_conditions"], list)
            assert isinstance(result["evidence"]["extra_reported_conditions"], list)


class TestAuditScriptMainEntry:
    """
    功能：审计脚本命令行入口的测试。
    """

    def test_main_returns_zero_on_pass(self):
        """
        Scenario: Audit passes (conditions match).

        Expected: main() returns exit code 0.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {"test::v1": {}},
        }
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [{"group_key": "test::v1"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            report_path = tmpdir_path / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            exit_code = audit_module.main(str(tmpdir_path))

            assert exit_code == 0

    def test_main_returns_nonzero_on_fail(self):
        """
        Scenario: Audit fails (conditions mismatch).

        Expected: main() returns exit code 1.
        """
        protocol_spec = {
            "version": "attack_protocol_v1",
            "params_versions": {"test::v1": {}, "missing::v1": {}},
        }
        report = {
            "report_type": "evaluation_report",
            "metrics_by_attack_condition": [{"group_key": "test::v1"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "configs").mkdir()

            protocol_path = tmpdir_path / "configs" / "attack_protocol.yaml"
            import yaml

            with open(protocol_path, "w") as f:
                yaml.dump(protocol_spec, f)

            report_path = tmpdir_path / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            exit_code = audit_module.main(str(tmpdir_path))

            assert exit_code == 1
