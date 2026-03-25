"""
File purpose: 评测报告 ablation 锚点回归测试。
Module type: General module
"""

from __future__ import annotations

import pytest

from main.evaluation import report_builder


def test_evaluation_report_includes_ablation_digest_anchor() -> None:
    """
    功能：验证评测报告包含 ablation_digest 锚点。

    Assert evaluation report includes ablation_digest at top-level and anchors.

    Args:
        None.

    Returns:
        None.
    """
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
        metrics_overall={"tpr_at_fpr_primary": 0.9},
        metrics_by_attack_condition=[],
        strict_anchor_validation=True,
    )

    assert report["ablation_digest"] == "ablation_digest_v1"
    anchors = report.get("anchors")
    assert isinstance(anchors, dict)
    assert anchors.get("ablation_digest") == "ablation_digest_v1"


def test_report_anchor_validation_fails_when_ablation_digest_missing() -> None:
    """
    功能：验证严格模式下缺失 ablation_digest 会失败。

    Assert strict anchor validation fails if ablation_digest is missing.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(RuntimeError, match="ablation_digest"):
        report_builder.build_evaluation_report(
            cfg_digest="cfg_digest",
            plan_digest="plan_digest",
            thresholds_digest="thresholds_digest",
            threshold_metadata_digest="threshold_metadata_digest",
            impl_digest="impl_digest",
            fusion_rule_version="fusion_v1",
            attack_protocol_version="attack_protocol_v1",
            attack_protocol_digest="attack_protocol_digest",
            policy_path="policy://test",
            ablation_digest=None,
            metrics_overall={"tpr_at_fpr_primary": 0.9},
            metrics_by_attack_condition=[],
            strict_anchor_validation=True,
        )
