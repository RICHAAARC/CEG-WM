"""
验证 aggregate_report 消融分组计数回归测试
"""

from __future__ import annotations

from main.evaluation.experiment_matrix import build_aggregate_report


def test_aggregate_report_includes_ablation_group_counts() -> None:
    """
    功能：验证 aggregate_report 包含按 ablation 分组的计数字段。

    Assert grouped ablation counts are present in aggregate report.

    Args:
        None.

    Returns:
        None.
    """
    experiment_results = [
        {
            "grid_item_digest": "d" * 64,
            "status": "ok",
            "run_root": "tmp/run_1",
            "attack_family": "rotate",
            "model_id": "model_a",
            "seed": 11,
            "cfg_digest": "cfg_digest_1",
            "plan_digest": "plan_digest_1",
            "thresholds_digest": "thresholds_digest_1",
            "threshold_metadata_digest": "threshold_metadata_digest_1",
            "ablation_digest": "ablation_digest_1",
            "attack_protocol_digest": "attack_protocol_digest_1",
            "attack_protocol_version": "attack_protocol_v1",
            "impl_digest": "impl_digest_1",
            "fusion_rule_version": "fusion_v1",
            "metrics": {
                "tpr_at_fpr": 0.91,
                "geo_available_rate": 0.84,
                "rescue_rate": 0.11,
                "reject_rate": 0.05,
                "reject_rate_breakdown": {
                    "absent": 2,
                    "mismatch": 1,
                },
                "n_total": 10,
                "n_accepted": 7,
                "n_rejected": 3,
                "n_rescue_triggered": 2,
                "n_rescue_success": 1,
                "conditional_fpr_estimate": 0.02,
                "conditional_fpr_n": 10,
            },
        }
    ]

    report = build_aggregate_report(experiment_results)

    grouped_rows = report.get("grouped_metrics")
    assert isinstance(grouped_rows, list)
    assert len(grouped_rows) == 1

    row = grouped_rows[0]
    assert row.get("ablation_digest") == "ablation_digest_1"
    assert isinstance(row.get("ablation_id"), str)
    assert row.get("n_total") == 10
    assert row.get("n_attack_applied") == 10
    assert row.get("n_valid_scored") == 7
    assert row.get("n_rejected_absent") == 2
    assert row.get("n_rejected_mismatch") == 1
    assert row.get("n_rejected_fail") >= 3
    assert row.get("n_rescue_triggered") == 2
    assert row.get("n_rescue_success") == 1


def test_aggregate_report_collects_research_failure_semantics_distribution() -> None:
    """
    功能：研究采集模式下聚合报告必须输出失败语义分布。 

    Verify aggregate report includes failure semantics distribution for relaxed detect gate runs.

    Args:
        None.

    Returns:
        None.
    """
    experiment_results = [
        {
            "grid_item_digest": "a" * 64,
            "status": "ok",
            "run_root": "tmp/run_relaxed_mismatch",
            "detect_gate_relaxed": True,
            "detect_gate_sample_counts": {"status": "mismatch"},
            "metrics": {},
        },
        {
            "grid_item_digest": "b" * 64,
            "status": "ok",
            "run_root": "tmp/run_relaxed_absent",
            "detect_gate_relaxed": True,
            "detect_gate_sample_counts": {"status": "absent"},
            "metrics": {},
        },
        {
            "grid_item_digest": "c" * 64,
            "status": "ok",
            "run_root": "tmp/run_hard_gate",
            "detect_gate_relaxed": False,
            "detect_gate_sample_counts": {"status": "failed"},
            "metrics": {},
        },
    ]

    report = build_aggregate_report(experiment_results)
    distribution = report.get("failure_semantics_distribution")

    assert isinstance(distribution, dict)
    assert distribution.get("scope") == "detect_gate_research_collection"
    assert distribution.get("relaxed_run_count") == 2
    status_counts = distribution.get("status_counts")
    assert isinstance(status_counts, dict)
    assert status_counts.get("mismatch") == 1
    assert status_counts.get("absent") == 1
    assert status_counts.get("failed") == 0
