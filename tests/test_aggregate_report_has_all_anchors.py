"""
验证 aggregate_report 锚点字段完备性
"""

from __future__ import annotations

from main.evaluation.experiment_matrix import build_aggregate_report


def test_aggregate_report_has_all_anchors() -> None:
    """
    功能：验证 aggregate_report 的 anchors 包含完整字段集合。

    Assert aggregate report anchors include all required fields.

    Args:
        None.

    Returns:
        None.
    """
    experiment_results = [
        {
            "grid_item_digest": "g" * 64,
            "status": "ok",
            "run_root": "tmp/run_001",
            "cfg_digest": "cfg" * 21 + "x",
            "plan_digest": "plan" * 16,
            "thresholds_digest": "thr" * 21 + "y",
            "threshold_metadata_digest": "meta" * 16,
            "ablation_digest": "abl" * 21 + "z",
            "attack_protocol_digest": "apd" * 21 + "q",
            "attack_protocol_version": "attack_protocol_v1",
            "impl_digest": "imp" * 21 + "w",
            "fusion_rule_version": "fusion_v1",
            "metrics": {
                "tpr_at_fpr": 0.91,
                "geo_available_rate": 0.84,
                "rescue_rate": 0.11,
                "reject_rate": 0.05,
                "reject_rate_breakdown": {"invalid_score": 0.03},
            },
        }
    ]

    report = build_aggregate_report(experiment_results)
    assert isinstance(report, dict)
    assert isinstance(report.get("anchors"), list)
    assert len(report["anchors"]) == 1

    anchor = report["anchors"][0]
    required_fields = [
        "cfg_digest",
        "plan_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "ablation_digest",
        "attack_protocol_digest",
        "impl_digest",
        "fusion_rule_version",
        "attack_protocol_version",
        "grid_item_digest",
    ]
    for field_name in required_fields:
        assert field_name in anchor
