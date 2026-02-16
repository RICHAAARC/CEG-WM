"""
文件目的：验证聚合层不触发阈值重计算。
Module type: General module
"""

from __future__ import annotations

from main.evaluation.experiment_matrix import build_aggregate_report


def test_no_threshold_recompute_in_aggregate(monkeypatch) -> None:
    """
    功能：验证 build_aggregate_report 不调用 NP 阈值计算函数。

    Assert aggregate builder never invokes NP threshold recomputation functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    def _raise_if_called(*args, **kwargs):
        _ = args
        _ = kwargs
        raise AssertionError("NP threshold recompute must not be called in aggregate")

    monkeypatch.setattr(
        "main.watermarking.fusion.neyman_pearson.compute_thresholds_digest",
        _raise_if_called,
    )
    monkeypatch.setattr(
        "main.watermarking.fusion.neyman_pearson.compute_threshold_metadata_digest",
        _raise_if_called,
    )

    report = build_aggregate_report(
        [
            {
                "grid_item_digest": "d" * 64,
                "status": "ok",
                "run_root": "tmp/run_002",
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
                    "tpr_at_fpr": 0.88,
                    "geo_available_rate": 0.8,
                    "rescue_rate": 0.2,
                    "reject_rate": 0.1,
                    "reject_rate_breakdown": {},
                },
            }
        ]
    )

    assert report["experiment_count"] == 1
    assert report["failure_count"] == 0
