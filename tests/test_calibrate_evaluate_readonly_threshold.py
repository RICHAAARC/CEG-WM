"""
File purpose: C4 阶段校准与评测只读阈值闭环回归测试。
Module type: General module
"""

import json
import math
from pathlib import Path

import pytest

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import (
    _load_records_for_evaluate,
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

    assert threshold_value == pytest.approx(math.nextafter(0.45, math.inf))
    assert order_stat["n_samples"] == 5
    assert order_stat["order_stat_rank_1based"] == 4
    assert order_stat["order_stat_index_0based"] == 3
    assert order_stat["selected_order_stat_score"] == pytest.approx(0.45)
    assert order_stat["quantile_rule"] == "higher"
    assert order_stat["ties_policy"] == "strict_upper_bound"


def test_load_scores_for_calibration_filters_invalid_records() -> None:
    """Validate strict filtering semantics for calibration score extraction."""
    records = [
        {"content_evidence_payload": {"status": "ok", "score": 0.5}},
        {"content_evidence_payload": {"status": "ok", "score": 0.7}},
        {
            "content_evidence_payload": {
                "status": "ok",
                "score": 0.6,
                "calibration_sample_origin": "sidecar_disabled_fallback",
                "calibration_sample_is_synthetic_fallback": True,
            }
        },
        {"content_evidence_payload": {"status": "failed", "score": 0.8}},
        {"content_evidence_payload": {"status": "ok", "score": "0.2"}},
        {"content_evidence_payload": {"status": "ok", "score": float("nan")}},
        {"content_evidence_payload": None},
    ]

    scores, strata = load_scores_for_calibration(records)

    assert scores == [0.5, 0.7]
    assert strata["global"]["n_total"] == 7
    assert strata["global"]["n_valid"] == 2
    assert strata["global"]["n_rejected"] == 5
    assert strata["sampling_policy"]["n_rejected_synthetic_fallback"] == 1


def test_load_scores_for_calibration_optional_formal_sidecar_filter() -> None:
    """Validate optional filtering for formal sidecar-disabled marker samples."""
    records = [
        {
            "content_evidence_payload": {
                "status": "ok",
                "score": 0.31,
                "calibration_sample_usage": "formal_with_sidecar_disabled_marker",
            }
        },
        {"content_evidence_payload": {"status": "ok", "score": 0.62}},
    ]

    default_scores, default_strata = load_scores_for_calibration(records)
    assert default_scores == [0.31, 0.62]
    assert default_strata["sampling_policy"]["exclude_formal_sidecar_disabled_marker"] is False
    assert default_strata["sampling_policy"]["n_rejected_formal_sidecar_marker"] == 0

    strict_cfg = {"calibration": {"exclude_formal_sidecar_disabled_marker": True}}
    strict_scores, strict_strata = load_scores_for_calibration(records, strict_cfg)
    assert strict_scores == [0.62]
    assert strict_strata["sampling_policy"]["exclude_formal_sidecar_disabled_marker"] is True
    assert strict_strata["sampling_policy"]["n_rejected_formal_sidecar_marker"] == 1


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

    metrics, breakdown, conditional_metrics = evaluate_records_against_threshold(records, thresholds_obj)

    assert metrics["n_total"] == 5
    assert metrics["n_accepted"] == 4
    assert metrics["n_rejected"] == 1
    assert metrics["tpr_at_fpr"] == pytest.approx(0.5)
    assert metrics["tpr_at_fpr_primary"] == pytest.approx(0.5)
    assert metrics["fpr_empirical"] == pytest.approx(0.5)
    assert metrics["reject_rate"] == pytest.approx(0.2)
    assert isinstance(metrics["reject_rate_by_reason"], dict)
    assert breakdown["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert conditional_metrics["version"] == "conditional_eval_v1"
    assert isinstance(conditional_metrics["items"], list) and len(conditional_metrics["items"]) == 3
    assert isinstance(conditional_metrics["attack_group_metrics"], list)


def test_evaluate_attestation_threshold_ignores_detect_hf_score_fallback() -> None:
    """Validate attestation readonly evaluate never accepts detect_hf_score fallback."""
    records = [
        {
            "attestation": {
                "image_evidence_result": {
                    "status": "ok",
                    "content_attestation_score": 0.8,
                    "content_attestation_score_name": "content_attestation_score",
                }
            },
            "content_evidence_payload": {"status": "ok", "detect_hf_score": 0.1},
            "label": True,
        },
        {
            "attestation": {
                "image_evidence_result": {
                    "status": "absent",
                    "content_attestation_score": None,
                }
            },
            "content_evidence_payload": {"status": "ok", "detect_hf_score": 0.95},
            "label": False,
        },
    ]
    thresholds_obj = {
        "threshold_id": "content_attestation_score_np_fpr_0_01",
        "score_name": "content_attestation_score",
        "target_fpr": 0.01,
        "threshold_value": 0.5,
        "threshold_key_used": "fpr_0_01",
    }

    metrics, breakdown, _ = evaluate_records_against_threshold(records, thresholds_obj)

    assert metrics["n_total"] == 2
    assert metrics["n_accepted"] == 1
    assert metrics["n_rejected"] == 1
    assert metrics["fpr_empirical"] is None
    assert breakdown["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}


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
        "attack": {"family": "jpeg", "params_version": "p1"},
        "decision": {
            "routing_decisions": {
                "rescue_triggered": True,
                "geo_gate_applied": True,
            }
        },
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
            "attack_family_field_candidates": ["attack.family"],
            "attack_params_version_field_candidates": ["attack.params_version"],
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
    assert result["metrics"]["geo_available_rate"] == pytest.approx(0.0)
    assert result["metrics"]["rescue_rate"] == pytest.approx(1.0)
    assert result["metrics"]["rescue_gain_rate"] == pytest.approx(1.0)
    assert result["threshold_key_used"] == "fpr_0_01"
    assert result["conditional_metrics"]["version"] == "conditional_eval_v1"
    assert result["conditional_metrics"]["attack_protocol_version"] == "attack_v1"
    attack_groups = result["conditional_metrics"]["attack_group_metrics"]
    assert len(attack_groups) == 1
    assert attack_groups[0]["group_key"] == "jpeg::p1"
    assert result["fusion_result"].decision_status == "abstain"
    assert result["fusion_result"].used_threshold_id == "content_score_np_fpr_0_01"
    assert result["evaluation_report"]["attack_protocol"]["version"] == "attack_v1"


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
    metadata_artifact = record["threshold_metadata_artifact"]
    assert "null_strata" in metadata_artifact
    assert "conditional_fpr" in metadata_artifact
    assert metadata_artifact["conditional_fpr"]["definition"]
    assert metadata_artifact["threshold_value_semantics"] == "strict_upper_bound"
    assert metadata_artifact["decision_operator"] == "score_greater_equal_threshold_value"
    assert "conditional_fpr_records" in metadata_artifact
    assert isinstance(metadata_artifact["conditional_fpr_records"], list)
    assert len(metadata_artifact["conditional_fpr_records"]) >= 3
    first_item = metadata_artifact["conditional_fpr_records"][0]
    assert "condition_id" in first_item
    assert "definition" in first_item
    assert "sample_count" in first_item
    assert "empirical_fpr" in first_item
    assert "inputs_digest" in first_item


def test_run_calibrate_orchestrator_conditional_fpr_records_are_deterministic(tmp_path: Path) -> None:
    """Validate conditional_fpr_records is deterministic for the same calibration inputs."""
    detect_record_path = tmp_path / "detect_record.json"
    detect_record_payload = {
        "content_evidence_payload": {"status": "ok", "score": 0.8},
        "geometry_evidence_payload": {
            "status": "ok",
            "score": 0.2,
            "sync_metrics": {"align_quality": 0.91},
        },
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

    record_a = run_calibrate_orchestrator(cfg, impl_set)
    record_b = run_calibrate_orchestrator(cfg, impl_set)

    assert record_a["threshold_metadata_artifact"]["conditional_fpr_records"] == record_b["threshold_metadata_artifact"]["conditional_fpr_records"]


def test_load_records_for_evaluate_excludes_synthetic_negative_closure_when_enabled(tmp_path: Path) -> None:
    """Validate evaluate loader filters synthetic negative closure samples when enabled."""
    keep_path = tmp_path / "detect_keep.json"
    keep_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.63,
                },
                "label": False,
            }
        ),
        encoding="utf-8",
    )

    filtered_path = tmp_path / "detect_filtered.json"
    filtered_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": -1.0,
                    "calibration_sample_usage": "synthetic_negative_for_ground_truth_closure",
                },
                "label": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "evaluate": {
            "detect_records_glob": str(tmp_path / "detect_*.json"),
            "exclude_synthetic_negative_closure_marker": True,
        }
    }

    loaded = _load_records_for_evaluate(cfg)
    assert len(loaded) == 1
    assert loaded[0]["content_evidence_payload"]["score"] == 0.63


def test_load_records_for_evaluate_keeps_synthetic_negative_closure_by_default(tmp_path: Path) -> None:
    """Validate evaluate loader keeps synthetic negative closure samples by default."""
    normal_path = tmp_path / "detect_normal.json"
    normal_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.41,
                },
                "label": False,
            }
        ),
        encoding="utf-8",
    )

    closure_path = tmp_path / "detect_closure.json"
    closure_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": -1.0,
                    "calibration_sample_usage": "synthetic_negative_for_ground_truth_closure",
                },
                "label": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "evaluate": {
            "detect_records_glob": str(tmp_path / "detect_*.json"),
        }
    }

    loaded = _load_records_for_evaluate(cfg)
    assert len(loaded) == 2
