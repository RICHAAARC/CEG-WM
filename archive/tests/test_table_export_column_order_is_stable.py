"""
测试表格导出的列顺序稳定性。

Test that CSV export column order is deterministic and does not depend on dict insertion order.
"""

import csv
import io
from pathlib import Path
import tempfile

import pytest
from main.evaluation import table_export


class TestTableExportColumnOrder:
    """表格导出列顺序测试套件。"""

    def test_export_column_names_constant_is_defined(self):
        """测试：导出常量 EXPORT_COLUMN_NAMES 存在且完整。"""
        # Assert
        assert hasattr(table_export, "EXPORT_COLUMN_NAMES")
        assert isinstance(table_export.EXPORT_COLUMN_NAMES, list)
        assert len(table_export.EXPORT_COLUMN_NAMES) > 0

    def test_export_column_order_matches_constant(self):
        """测试：导出 CSV 的列顺序与常量定义相符。"""
        # Arrange
        report = {
            "metrics": {
                "tpr_at_fpr_primary": 0.95,
                "fpr_empirical": 0.05,
                "n_total": 1000,
                "n_accepted": 900,
                "n_pos": 500,
                "n_neg": 500,
                "geo_available_rate": 0.8,
                "rescue_rate": 0.1,
                "reject_rate": 0.1,
            },
            "metrics_by_attack_condition": [
                {
                    "group_key": "rotate::v1",
                    "n_total": 300,
                    "n_accepted": 270,
                    "n_pos": 150,
                    "n_neg": 150,
                    "tp": 140,
                    "fp": 5,
                    "tpr_at_fpr_primary": 0.93,
                    "fpr_empirical": 0.03,
                    "geo_available_rate": 0.75,
                    "rescue_rate": 0.08,
                    "reject_rate_by_reason": {"status_not_ok": 0.1},
                }
            ],
        }

        # Act
        csv_content = table_export.export_metrics_to_csv(report)

        # Assert - 使用 csv.DictReader 安全解析，避免 Windows 换行符问题
        reader = csv.DictReader(io.StringIO(csv_content))
        headers = reader.fieldnames

        # 列顺序必须完全匹配常量
        assert headers == table_export.EXPORT_COLUMN_NAMES

    def test_csv_header_order_independent_of_dict_order(self):
        """测试：CSV 列顺序与 dict 建立顺序无关。"""
        # Arrange - 构造两个语义相同但插入顺序不同的报告
        metrics_v1 = {
            "n_total": 1000,
            "tpr_at_fpr_primary": 0.95,
            "n_accepted": 900,
            "fpr_empirical": 0.05,
            "geo_available_rate": 0.8,
            "rescue_rate": 0.1,
        }
        metrics_v2 = {
            "fpr_empirical": 0.05,
            "n_total": 1000,
            "geo_available_rate": 0.8,
            "tpr_at_fpr_primary": 0.95,
            "rescue_rate": 0.1,
            "n_accepted": 900,
        }

        report_v1 = {
            "metrics": metrics_v1,
            "metrics_by_attack_condition": [],
        }
        report_v2 = {
            "metrics": metrics_v2,
            "metrics_by_attack_condition": [],
        }

        # Act
        csv_v1 = table_export.export_metrics_to_csv(report_v1)
        csv_v2 = table_export.export_metrics_to_csv(report_v2)

        # Assert - 两个 CSV 的列顺序应完全相同
        headers_v1 = csv_v1.strip().split("\n")[0].split(",")
        headers_v2 = csv_v2.strip().split("\n")[0].split(",")
        assert headers_v1 == headers_v2

    def test_csv_content_rows_have_correct_field_count(self):
        """测试：导出的 CSV 行与列数相符。"""
        # Arrange
        report = {
            "metrics": {
                "tpr_at_fpr_primary": 0.95,
                "fpr_empirical": 0.05,
                "n_total": 1000,
                "n_accepted": 900,
                "n_pos": 500,
                "n_neg": 500,
                "geo_available_rate": 0.8,
                "rescue_rate": 0.1,
                "reject_rate": 0.1,
            },
            "metrics_by_attack_condition": [],
        }

        # Act
        csv_content = table_export.export_metrics_to_csv(report)

        # Assert
        lines = csv_content.strip().split("\n")
        header_count = len(lines[0].split(",")) if lines else 0

        for line in lines[1:]:  # Skip header
            if line.strip():
                field_count = len(line.split(","))
                assert field_count == header_count

    def test_overall_row_has_condition_column_set_to_overall(self):
        """测试：overall 行的 condition 列值为 'OVERALL'。"""
        # Arrange
        report = {
            "metrics": {
                "tpr_at_fpr_primary": 0.95,
                "fpr_empirical": 0.05,
                "n_total": 1000,
                "n_accepted": 900,
                "n_pos": 500,
                "n_neg": 500,
                "geo_available_rate": 0.8,
                "rescue_rate": 0.1,
                "reject_rate": 0.1,
            },
            "metrics_by_attack_condition": [],
        }

        # Act
        csv_content = table_export.export_metrics_to_csv(report)

        # Assert
        lines = csv_content.strip().split("\n")
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # 第一行应该是 overall
        if rows:
            assert rows[0]["condition"] == "OVERALL"

    def test_per_condition_rows_have_correct_condition_key(self):
        """测试：按条件的行包含正确的 condition 键。"""
        # Arrange
        report = {
            "metrics": {
                "tpr_at_fpr_primary": 0.95,
                "fpr_empirical": 0.05,
                "n_total": 400,
                "n_accepted": 360,
                "n_pos": 200,
                "n_neg": 200,
                "geo_available_rate": 0.8,
                "rescue_rate": 0.1,
                "reject_rate": 0.1,
            },
            "metrics_by_attack_condition": [
                {
                    "group_key": "rotate::v1",
                    "n_total": 200,
                    "n_accepted": 180,
                    "n_pos": 100,
                    "n_neg": 100,
                    "tp": 95,
                    "fp": 3,
                    "tpr_at_fpr_primary": 0.95,
                    "fpr_empirical": 0.03,
                    "geo_available_rate": 0.78,
                    "rescue_rate_by_reason": {},
                },
                {
                    "group_key": "resize::v1",
                    "n_total": 200,
                    "n_accepted": 180,
                    "n_pos": 100,
                    "n_neg": 100,
                    "tp": 90,
                    "fp": 5,
                    "tpr_at_fpr_primary": 0.90,
                    "fpr_empirical": 0.05,
                    "geo_available_rate": 0.75,
                    "rescue_rate_by_reason": {},
                },
            ],
        }

        # Act
        csv_content = table_export.export_metrics_to_csv(report)

        # Assert
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # 应该有 3 行：1 overall + 2 per-condition
        assert len(rows) == 3
        assert rows[0]["condition"] == "OVERALL"
        # 排序后：resize < rotate
        assert rows[1]["condition"] == "resize::v1"
        assert rows[2]["condition"] == "rotate::v1"

    def test_export_to_file_preserves_column_order(self, tmp_path):
        """测试：导出到文件时，列顺序保持一致。"""
        # Arrange
        report = {
            "metrics": {
                "tpr_at_fpr_primary": 0.95,
                "fpr_empirical": 0.05,
                "n_total": 1000,
                "n_accepted": 900,
                "n_pos": 500,
                "n_neg": 500,
                "geo_available_rate": 0.8,
                "rescue_rate": 0.1,
                "reject_rate": 0.1,
            },
            "metrics_by_attack_condition": [],
        }
        output_file = tmp_path / "test_export.csv"

        # Act
        table_export.export_metrics_to_csv(report, output_file)

        # Assert
        csv_content = output_file.read_text(encoding="utf-8")
        headers = csv_content.strip().split("\n")[0].split(",")
        assert headers == table_export.EXPORT_COLUMN_NAMES
