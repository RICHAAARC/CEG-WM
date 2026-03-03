"""
File purpose: Verify synthetic closure filtering is opt-in in calibration sampling.
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, cast

from main.watermarking.detect.orchestrator import load_scores_for_calibration


def test_load_scores_for_calibration_does_not_filter_synthetic_closure_by_default() -> None:
    records = cast(
        list[Dict[str, Any]],
        [
        {
            "label": False,
            "content_evidence_payload": {
                "status": "ok",
                "score": -1.0,
                "calibration_sample_usage": "synthetic_negative_for_ground_truth_closure",
            },
        }
        ],
    )

    scores, strata = load_scores_for_calibration(records, cfg={"calibration": {}})
    assert scores == [-1.0]
    sampling_policy = strata["sampling_policy"]
    assert sampling_policy["exclude_synthetic_negative_closure_marker"] is False
    assert sampling_policy["n_rejected_synthetic_negative_closure"] == 0


def test_load_scores_for_calibration_filters_synthetic_closure_when_opt_in() -> None:
    records = cast(
        list[Dict[str, Any]],
        [
        {
            "label": False,
            "content_evidence_payload": {
                "status": "ok",
                "score": -1.0,
                "calibration_sample_usage": "synthetic_negative_for_ground_truth_closure",
            },
        },
        {
            "label": False,
            "content_evidence_payload": {
                "status": "ok",
                "score": -0.7,
                "calibration_sample_usage": "formal_with_dual_branch_negative_marker",
            },
        },
        ],
    )

    cfg = {"calibration": {"exclude_synthetic_negative_closure_marker": True}}
    scores, strata = load_scores_for_calibration(records, cfg=cfg)
    assert scores == [-0.7]
    sampling_policy = strata["sampling_policy"]
    assert sampling_policy["exclude_synthetic_negative_closure_marker"] is True
    assert sampling_policy["n_rejected_synthetic_negative_closure"] == 1
