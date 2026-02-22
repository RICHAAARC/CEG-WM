"""
File purpose: 攻击协议事实源与分组评测输出回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import run_evaluate_orchestrator


class _ExtractorRaiser:
    def extract(self, cfg: Dict[str, Any]) -> None:
        _ = cfg
        raise AssertionError("extract should not be called in evaluate readonly mode")


class _FusionRaiser:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> None:
        _ = cfg
        _ = content_evidence
        _ = geometry_evidence
        raise AssertionError("fuse should not be called in evaluate readonly mode")


def _build_impl_set() -> BuiltImplSet:
    return BuiltImplSet(
        content_extractor=_ExtractorRaiser(),
        geometry_extractor=_ExtractorRaiser(),
        fusion_rule=_FusionRaiser(),
        subspace_planner=object(),
        sync_module=object(),
    )


def test_attack_protocol_is_fact_source_and_versioned(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds_artifact.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "threshold_id": "content_score_np_fpr_0_01",
                "score_name": "content_score",
                "target_fpr": 0.01,
                "threshold_value": 0.5,
                "threshold_key_used": "fpr_0_01",
            }
        ),
        encoding="utf-8",
    )

    detect_record_path = tmp_path / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
                "attack": {"family": "rotate", "params_version": "v1"},
            }
        ),
        encoding="utf-8",
    )

    cfg: Dict[str, Any] = {
        "evaluate": {
            "thresholds_path": str(thresholds_path),
            "detect_records_glob": str(detect_record_path),
        },
        "__evaluate_cfg_digest__": "cfg_eval_digest",
    }

    result = run_evaluate_orchestrator(cfg, _build_impl_set())

    assert result["evaluation_report"]["attack_protocol_version"] == "attack_protocol_v1"
    assert result["evaluation_report"]["attack_protocol_digest"] != "<absent>"


def test_evaluate_outputs_grouped_attack_metrics(tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds_artifact.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "threshold_id": "content_score_np_fpr_0_01",
                "score_name": "content_score",
                "target_fpr": 0.01,
                "threshold_value": 0.5,
                "threshold_key_used": "fpr_0_01",
            }
        ),
        encoding="utf-8",
    )

    detect_path_a = tmp_path / "detect_a.json"
    detect_path_b = tmp_path / "detect_b.json"
    detect_path_c = tmp_path / "detect_c.json"

    detect_path_a.write_text(
        json.dumps(
            {
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
                "attack": {"family": "rotate", "params_version": "v1"},
                "decision": {"routing_decisions": {"rescue_triggered": True}},
                "geometry_evidence_payload": {"status": "ok", "geo_score": 0.4},
            }
        ),
        encoding="utf-8",
    )
    detect_path_b.write_text(
        json.dumps(
            {
                "content_evidence_payload": {"status": "ok", "score": 0.2},
                "label": False,
                "attack": {"family": "resize", "params_version": "v1"},
                "geometry_evidence_payload": {"status": "ok", "geo_score": 0.3},
            }
        ),
        encoding="utf-8",
    )
    detect_path_c.write_text(
        json.dumps(
            {
                "content_evidence_payload": {"status": "failed", "score": 0.2},
                "label": False,
                "attack": {"family": "resize", "params_version": "v1"},
            }
        ),
        encoding="utf-8",
    )

    cfg: Dict[str, Any] = {
        "evaluate": {
            "thresholds_path": str(thresholds_path),
            "detect_records_glob": str(tmp_path / "detect_*.json"),
        },
        "__evaluate_cfg_digest__": "cfg_eval_digest",
    }

    result = run_evaluate_orchestrator(cfg, _build_impl_set())

    grouped = result["conditional_metrics"]["attack_group_metrics"]
    grouped_keys = {item["group_key"] for item in grouped}

    assert "rotate::v1" in grouped_keys
    assert "resize::v1" in grouped_keys
    for item in grouped:
        assert "tpr_at_fpr_primary" in item
        assert "geo_available_rate" in item
        assert "rescue_rate" in item
        assert "rescue_gain_rate" in item
        assert "reject_rate_by_reason" in item
