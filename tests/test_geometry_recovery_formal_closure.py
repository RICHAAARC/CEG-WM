"""
File purpose: 几何恢复 formal 配置与注册表闭包回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

from main.core import config_loader


def test_paper_full_cuda_declares_geometry_recovery_ready_fields() -> None:
    """
    功能：paper_full_cuda 必须显式携带几何恢复阈值，并清除旧注释。 

    Verify paper_full_cuda declares geometry recovery thresholds without stale notes.

    Args:
        None.

    Returns:
        None.
    """
    cfg, _ = config_loader.load_yaml_with_provenance("configs/paper_full_cuda.yaml")
    geometry_cfg = cfg["detect"]["geometry"]
    assert geometry_cfg["sync_primary_anchor_secondary"] is True
    assert geometry_cfg["align_max_correspondence_count"] == 24
    assert geometry_cfg["align_template_overlap_min"] == 0.70
    assert geometry_cfg["align_recovered_sync_consistency_min"] == 0.55
    assert geometry_cfg["align_recovered_anchor_consistency_min"] == 0.50

    raw_text = Path("configs/paper_full_cuda.yaml").read_text(encoding="utf-8")
    assert "STATIC-02 未修复" not in raw_text
    assert "matched_correlation" in raw_text


def test_geometry_recovery_fields_are_registered_in_formal_contracts() -> None:
    """
    功能：几何恢复新增字段必须在 schema、registry 与语义面完成闭合。 

    Verify geometry recovery fields are registered across schema, frozen contracts, and semantics.

    Args:
        None.

    Returns:
        None.
    """
    extensions_text = Path("configs/records_schema_extensions.yaml").read_text(encoding="utf-8")
    contracts_text = Path("configs/frozen_contracts.yaml").read_text(encoding="utf-8")
    semantics_text = Path("configs/policy_path_semantics.yaml").read_text(encoding="utf-8")
    spec_text = Path("configs/paper_faithfulness_spec.yaml").read_text(encoding="utf-8")

    required_fields = [
        "geometry_evidence.anchor_observations",
        "geometry_evidence.sync_observations",
        "geometry_evidence.observed_correspondences",
        "geometry_evidence.observed_correspondence_summary",
        "geometry_evidence.align_metrics.template_overlap_consistency",
        "geometry_evidence.align_metrics.recovered_sync_consistency",
        "geometry_evidence.align_metrics.recovered_anchor_consistency",
    ]

    for field_name in required_fields:
        assert field_name in extensions_text
        assert field_name in contracts_text

    assert "geometry_evidence.observed_correspondence_summary" in semantics_text
    assert "geometry_evidence.align_metrics.recovered_sync_consistency" in semantics_text
    assert "无需 BP 迭代" not in spec_text
    assert "sparse LDPC BP decode" in spec_text