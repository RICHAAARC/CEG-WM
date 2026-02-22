"""
File purpose: C4 阶段校准与评测只读阈值闭环回归测试。
Module type: General module
"""

import json
from pathlib import Path

import pytest

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import (
    evaluate_records_against_threshold,
    load_scores_for_calibration,
    load_thresholds_artifact_controlled,
    run_calibrate_orchestrator,
    run_evaluate_orchestrator,
)
from main.watermarking.fusion.neyman_pearson import compute_np_threshold_from_scores


def test_compute_np_threshold_from_scores_higher_quantile() -> None:
    """Validate higher-quantile order-statistics threshold selection."""
    scores = [0.11, 0.20, 0.33, 0.45, 0.50]
    threshold_value, order_stat = compute_np_threshold_from_scores(scores, target_fpr=0.2)

    assert threshold_value == pytest.approx(0.45)
    assert order_stat["n_samples"] == 5
    assert order_stat["order_stat_rank_1based"] == 4
    assert order_stat["order_stat_index_0based"] == 3
    assert order_stat["quantile_rule"] == "higher"


def test_load_scores_for_calibration_filters_invalid_records() -> None:
    """Validate strict filtering semantics for calibration score extraction."""
    records = [
        {"content_evidence_payload": {"status": "ok", "score": 0.5}},
        {"content_evidence_payload": {"status": "ok", "score": 0.7}},
        {"content_evidence_payload": {"status": "failed", "score": 0.8}},
        {"content_evidence_payload": {"status": "ok", "score": "0.2"}},
        {"content_evidence_payload": {"status": "ok", "score": float("nan")}},
        {"content_evidence_payload": None},
    ]

    scores, strata = load_scores_for_calibration(records)

    assert scores == [0.5, 0.7]
    assert strata["global"]["n_total"] == 6
    assert strata["global"]["n_valid"] == 2
    assert strata["global"]["n_rejected"] == 4


def test_load_thresholds_artifact_controlled_requires_fields(tmp_path: Path) -> None:
    """Validate missing required field detection for thresholds artifact."""
    artifact_path = tmp_path / "thresholds.json"
    payload = {
        "threshold_id": "content_score_np_fpr_0_01",
        "score_name": "content_score",
        "target_fpr": 0.01,
        "threshold_value": 0.42,
    }
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="thresholds artifact missing field"):
        load_thresholds_artifact_controlled(str(artifact_path))


def test_evaluate_records_against_threshold_uses_readonly_threshold() -> None:
    """Validate metric computation against precomputed threshold artifact."""
    records = [
        {"content_evidence_payload": {"status": "ok", "score": 0.9}, "label": True},
        {"content_evidence_payload": {"status": "ok", "score": 0.4}, "label": False},
        {"content_evidence_payload": {"status": "ok", "score": 0.8}, "label": False},
        {"content_evidence_payload": {"status": "ok", "score": 0.2}, "label": True},
        {"content_evidence_payload": {"status": "mismatch", "score": None}, "label": True},
    ]
    thresholds_obj = {
        "threshold_id": "content_score_np_fpr_0_01",
        "score_name": "content_score",
        "target_fpr": 0.01,
        "threshold_value": 0.5,
        "threshold_key_used": "fpr_0_01",
    }

    metrics, breakdown = evaluate_records_against_threshold(records, thresholds_obj)

    assert metrics["n_total"] == 5
    assert metrics["n_accepted"] == 4
    assert metrics["n_rejected"] == 1
    assert metrics["tpr_at_fpr"] == pytest.approx(0.5)
    assert metrics["fpr_empirical"] == pytest.approx(0.5)
    assert metrics["reject_rate"] == pytest.approx(0.2)
    assert breakdown["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


class _ExtractorRaiser:
    def extract(self, cfg):
        _ = cfg
        raise AssertionError("extract should not be called in evaluate readonly mode")


class _FusionRaiser:
    def fuse(self, cfg, content_evidence, geometry_evidence):
        _ = cfg
        _ = content_evidence
        _ = geometry_evidence
        raise AssertionError("fuse should not be called in evaluate readonly mode")


def test_run_evaluate_orchestrator_readonly_without_extractors(tmp_path: Path) -> None:
    """Validate evaluate orchestrator runs in readonly mode over artifacts and records."""
    thresholds_path = tmp_path / "thresholds_artifact.json"
    thresholds_payload = {
        "threshold_id": "content_score_np_fpr_0_01",
        "score_name": "content_score",
        "target_fpr": 0.01,
        "threshold_value": 0.5,
        "threshold_key_used": "fpr_0_01",
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")

    detect_record_path = tmp_path / "detect_record.json"
    detect_record_payload = {
        "content_evidence_payload": {"status": "ok", "score": 0.8},
        "label": True,
        "cfg_digest": "cfg_sample",
        "plan_digest": "plan_sample",
        "impl": {"digests": {"content_extractor": "impl_sample"}},
    }
    detect_record_path.write_text(json.dumps(detect_record_payload), encoding="utf-8")

    cfg = {
        "evaluate": {
            "thresholds_path": str(thresholds_path),
            "detect_records_glob": str(detect_record_path),
            "attack_protocol_version": "attack_v1",
        },
        "__evaluate_cfg_digest__": "cfg_eval_digest",
    }
    impl_set = BuiltImplSet(
        content_extractor=_ExtractorRaiser(),
        geometry_extractor=_ExtractorRaiser(),
        fusion_rule=_FusionRaiser(),
        subspace_planner=object(),
        sync_module=object(),
    )

    result = run_evaluate_orchestrator(cfg, impl_set)

    assert result["operation"] == "evaluate"
    assert result["evaluation_is_fallback"] is False
    assert result["evaluation_mode"] == "real"
    assert result["metrics"]["n_total"] == 1
    assert result["threshold_key_used"] == "fpr_0_01"
    assert result["fusion_result"].decision_status == "abstain"
    assert result["fusion_result"].used_threshold_id == "content_score_np_fpr_0_01"


def test_run_calibrate_orchestrator_sets_calibration_mode(tmp_path: Path) -> None:
    """
    功能：验证 calibrate 输出包含规范化运行模式字段。

    Validate calibrate record includes normalized mode field.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    detect_record_path = tmp_path / "detect_record.json"
    detect_record_payload = {
        "content_evidence_payload": {"status": "ok", "score": 0.8},
    }
    detect_record_path.write_text(json.dumps(detect_record_payload), encoding="utf-8")

    cfg = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_record_path)},
    }
    impl_set = BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=object(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=object(),
    )

    record = run_calibrate_orchestrator(cfg, impl_set)

    assert record["calibration_is_fallback"] is False
    assert record["calibration_mode"] == "real"
