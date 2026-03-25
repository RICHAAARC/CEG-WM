"""
测试几何锚点强制与完整性

Module type: General module

Test geometry anchor enforcement in policy_path gates.
Verifies that geometry anchor fields are required when geometry sync is successful.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

# 添加 scripts 目录到路径以导入审计函数
import sys
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from audits.audit_paper_faithfulness_runtime_must_have import (
    check_geometry_anchor_completeness,
)


class TestGeometryAnchorEnforcement:
    """测试几何锚点强制。"""

    def test_geometry_anchor_required_when_sync_ok(self, tmp_path: Path):
        """测试 geometry sync 成功时锚点字段必须存在。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_np_geo_rescue",
            "detect": {
                "geometry": {
                    "enabled": True
                }
            },
            "geometry_evidence": {
                "sync": {
                    "status": "ok"
                },
                "anchor_digest": "abc123",
                "anchor_metrics": {"confidence": 0.95},
                "anchor_evidence_level": "high"
            }
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is True
        assert result.get("check") == "geometry_anchor_completeness"

    def test_geometry_anchor_missing_when_sync_ok_fails(self, tmp_path: Path):
        """测试 geometry sync 成功但锚点字段缺失时失败。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_np_geo_rescue",
            "detect": {
                "geometry": {
                    "enabled": True
                }
            },
            "geometry_evidence": {
                "sync": {
                    "status": "ok"
                }
            }
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is False
        missing_fields = result.get("missing_fields", [])
        assert "geometry_evidence.anchor_digest" in missing_fields
        assert "geometry_evidence.anchor_metrics" in missing_fields
        assert "geometry_evidence.anchor_evidence_level" in missing_fields

    def test_geometry_anchor_optional_when_sync_failed(self, tmp_path: Path):
        """测试 geometry sync 失败时锚点字段可以缺失。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_np_geo_rescue",
            "detect": {
                "geometry": {
                    "enabled": True
                }
            },
            "geometry_evidence": {
                "sync": {
                    "status": "failed"
                }
            }
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is True
        assert "failure is legal" in result.get("note", "")

    def test_geometry_anchor_optional_when_geometry_disabled(self, tmp_path: Path):
        """测试几何链禁用时锚点字段检查 N.A.。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_only",
            "detect": {
                "geometry": {
                    "enabled": False
                }
            },
            "geometry_evidence": {}
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is True
        assert "not enabled" in result.get("note", "")

    def test_geometry_anchor_no_detect_records(self, tmp_path: Path):
        """测试没有 detect records 时检查 N.A.。"""
        repo_root = tmp_path
        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is True
        assert "No detect records" in result.get("note", "")

    def test_geometry_anchor_partial_fields_missing(self, tmp_path: Path):
        """测试部分锚点字段缺失时失败。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_np_geo_rescue",
            "detect": {
                "geometry": {
                    "enabled": True
                }
            },
            "geometry_evidence": {
                "sync": {
                    "status": "ok"
                },
                "anchor_digest": "abc123"
            }
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is False
        missing_fields = result.get("missing_fields", [])
        assert "geometry_evidence.anchor_metrics" in missing_fields
        assert "geometry_evidence.anchor_evidence_level" in missing_fields
        assert "geometry_evidence.anchor_digest" not in missing_fields

    def test_geometry_anchor_with_absent_sentinel_value(self, tmp_path: Path):
        """测试锚点字段为 <absent> 哨兵值时被认为缺失。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record = {
            "policy_path": "content_np_geo_rescue",
            "detect": {
                "geometry": {
                    "enabled": True
                }
            },
            "geometry_evidence": {
                "sync": {
                    "status": "ok"
                },
                "anchor_digest": "<absent>",
                "anchor_metrics": "<failed>",
                "anchor_evidence_level": ""
            }
        }

        record_file = detect_dir / "detect_record.json"
        record_file.write_text(json.dumps(record), encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is False
        missing_fields = result.get("missing_fields", [])
        assert len(missing_fields) == 3

    def test_geometry_anchor_record_load_error(self, tmp_path: Path):
        """测试 record 加载失败时返回错误。"""
        repo_root = tmp_path
        detect_dir = repo_root / "tmp" / "cli_smoke" / "detect_run" / "records"
        detect_dir.mkdir(parents=True)

        record_file = detect_dir / "detect_record.json"
        record_file.write_text("{invalid json}", encoding="utf-8")

        result = check_geometry_anchor_completeness(repo_root)
        assert result.get("pass") is False
        assert "error" in result or "Failed to load" in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
