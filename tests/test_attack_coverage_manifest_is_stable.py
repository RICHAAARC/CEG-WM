"""
File purpose: 攻击覆盖声明稳定性与报告锚点测试。
Module type: General module
"""

from __future__ import annotations

from main.evaluation import attack_coverage
from main.evaluation import report_builder


def test_attack_coverage_manifest_is_stable() -> None:
    """
    功能：验证同环境下 attack coverage manifest 稳定。

    Assert attack coverage manifest output is stable in same environment.

    Args:
        None.

    Returns:
        None.
    """
    manifest_a = attack_coverage.compute_attack_coverage_manifest()
    manifest_b = attack_coverage.compute_attack_coverage_manifest()

    assert manifest_a == manifest_b
    assert isinstance(manifest_a.get("attack_coverage_digest"), str)
    assert len(manifest_a["attack_coverage_digest"]) == 64


def test_report_includes_attack_coverage_digest() -> None:
    """
    功能：验证评测报告包含 attack_coverage_digest。

    Assert report carries attack_coverage_digest in top-level and anchors.

    Args:
        None.

    Returns:
        None.
    """
    manifest = attack_coverage.compute_attack_coverage_manifest()
    coverage_digest = manifest["attack_coverage_digest"]

    report = report_builder.build_evaluation_report(
        cfg_digest="cfg_digest",
        plan_digest="plan_digest",
        thresholds_digest="thresholds_digest",
        threshold_metadata_digest="threshold_metadata_digest",
        impl_digest="impl_digest",
        fusion_rule_version="fusion_v1",
        attack_protocol_version="attack_protocol_v1",
        attack_protocol_digest="attack_protocol_digest",
        policy_path="policy://test",
        ablation_digest="ablation_digest_v1",
        attack_coverage_digest=coverage_digest,
        metrics_overall={"tpr_at_fpr_primary": 0.9},
        metrics_by_attack_condition=[],
        strict_anchor_validation=True,
    )

    assert report.get("attack_coverage_digest") == coverage_digest
    anchors = report.get("anchors")
    assert isinstance(anchors, dict)
    assert anchors.get("attack_coverage_digest") == coverage_digest
