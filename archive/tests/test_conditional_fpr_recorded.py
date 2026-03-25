"""
File purpose: 条件 FPR 记录固化与 digest 口径回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import run_calibrate_orchestrator


def _build_impl_set() -> BuiltImplSet:
    return BuiltImplSet(
        content_extractor=object(),
        geometry_extractor=object(),
        fusion_rule=object(),
        subspace_planner=object(),
        sync_module=object(),
    )


def _write_detect_record(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _find_condition(records: List[Dict[str, Any]], condition_id: str) -> Dict[str, Any]:
    for item in records:
        if item.get("condition_id") == condition_id:
            return item
    raise AssertionError(f"condition not found: {condition_id}")


def test_conditional_fpr_rescue_band_uses_config_delta(tmp_path: Path) -> None:
    detect_path = tmp_path / "detect_record.json"
    _write_detect_record(
        detect_path,
        {
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.4, "sync_metrics": {"align_quality": 0.92}},
            "decision": {"routing_decisions": {"geo_gate_applied": True}},
        },
    )

    cfg: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_path)},
        "rescue_band_delta_low": 0.123,
        "rescue_band_delta_high": 0.02,
    }

    record = run_calibrate_orchestrator(cfg, _build_impl_set())
    conditional_records = record["threshold_metadata_artifact"]["conditional_fpr_records"]
    rescue_item = _find_condition(conditional_records, "rescue_band_candidate")

    assert "threshold-0.123" in rescue_item["definition"]


def test_conditional_fpr_digest_changes_when_delta_changes(tmp_path: Path) -> None:
    detect_path = tmp_path / "detect_record.json"
    _write_detect_record(
        detect_path,
        {
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.35, "sync_metrics": {"align_quality": 0.85}},
            "decision": {"routing_decisions": {"geo_gate_applied": True}},
        },
    )

    cfg_a: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_path)},
        "rescue_band_delta_low": 0.05,
    }
    cfg_b: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_path)},
        "rescue_band_delta_low": 0.15,
    }

    record_a = run_calibrate_orchestrator(cfg_a, _build_impl_set())
    record_b = run_calibrate_orchestrator(cfg_b, _build_impl_set())

    rescue_a = _find_condition(record_a["threshold_metadata_artifact"]["conditional_fpr_records"], "rescue_band_candidate")
    rescue_b = _find_condition(record_b["threshold_metadata_artifact"]["conditional_fpr_records"], "rescue_band_candidate")

    assert rescue_a["inputs_digest"] != rescue_b["inputs_digest"]


def test_conditional_fpr_digest_changes_when_geo_gate_params_change(tmp_path: Path) -> None:
    detect_path = tmp_path / "detect_record.json"
    _write_detect_record(
        detect_path,
        {
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.6, "sync_metrics": {"align_quality": 0.95}},
            "decision": {"routing_decisions": {"geo_gate_applied": True}},
        },
    )

    cfg_a: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_path)},
        "geo_gate_lower": 0.3,
        "geo_gate_upper": 0.7,
    }
    cfg_b: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01},
        "calibration": {"detect_records_glob": str(detect_path)},
        "geo_gate_lower": 0.45,
        "geo_gate_upper": 0.9,
    }

    record_a = run_calibrate_orchestrator(cfg_a, _build_impl_set())
    record_b = run_calibrate_orchestrator(cfg_b, _build_impl_set())

    gate_a = _find_condition(record_a["threshold_metadata_artifact"]["conditional_fpr_records"], "geo_gate_applied")
    gate_b = _find_condition(record_b["threshold_metadata_artifact"]["conditional_fpr_records"], "geo_gate_applied")

    assert gate_a["inputs_digest"] != gate_b["inputs_digest"]


def test_conditional_fpr_align_quality_threshold_condition_present(tmp_path: Path) -> None:
    detect_path_a = tmp_path / "detect_record_a.json"
    detect_path_b = tmp_path / "detect_record_b.json"
    _write_detect_record(
        detect_path_a,
        {
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.45, "sync_metrics": {"align_quality": 0.9}},
        },
    )
    _write_detect_record(
        detect_path_b,
        {
            "content_evidence_payload": {"status": "ok", "score": 0.7},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.4},
        },
    )

    cfg: Dict[str, Any] = {
        "evaluate": {"target_fpr": 0.01, "align_quality_threshold": 0.85},
        "calibration": {"detect_records_glob": str(tmp_path / "detect_record_*.json")},
    }

    record = run_calibrate_orchestrator(cfg, _build_impl_set())
    align_item = _find_condition(record["threshold_metadata_artifact"]["conditional_fpr_records"], "align_quality_ge_threshold")

    assert "alignment quality >= 0.85" in align_item["definition"]
    assert "unavailable_count" in align_item
