"""
功能：验证 experiment_matrix 锚点字段的回退提取。

Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from main.evaluation import experiment_matrix


def test_run_single_experiment_anchor_fallback_from_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：当 eval_report 缺少锚点时，回退到 evaluate_record/run_closure。

    Verify anchor fallback order for cfg_digest, threshold_metadata_digest,
    and impl_digest.

    Args:
        tmp_path: pytest temp path fixture.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    run_root = tmp_path / "item_0000"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    evaluate_record = {
        "cfg_digest": "cfg_from_evaluate_record",
        "thresholds_digest": "thresholds_from_evaluate_record",
        "threshold_metadata_digest": "threshold_meta_from_evaluate_record",
        "impl_digest": "",
        "fusion_rule_version": "v1",
    }
    run_closure = {
        "impl_identity_digest": "impl_identity_digest_from_run_closure",
    }

    (records_dir / "evaluate_record.json").write_text(json.dumps(evaluate_record), encoding="utf-8")
    (artifacts_dir / "run_closure.json").write_text(json.dumps(run_closure), encoding="utf-8")

    def _fake_derive_run_root(_cfg: Dict[str, Any]) -> Path:
        return run_root

    def _fake_run_stage_sequence(_cfg: Dict[str, Any], _root: Path) -> None:
        return None

    def _fake_assert_required_run_artifacts(_root: Path) -> None:
        return None

    def _fake_read_evaluation_report_for_run(_root: Path) -> Dict[str, Any]:
        return {
            "metrics": {},
            "cfg_digest": "<absent>",
            "thresholds_digest": "<absent>",
            "threshold_metadata_digest": "<absent>",
            "impl_digest": "<absent>",
            "fusion_rule_version": "v1",
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "attack_digest",
        }

    monkeypatch.setattr(experiment_matrix, "_derive_run_root", _fake_derive_run_root)
    monkeypatch.setattr(experiment_matrix, "_run_stage_sequence", _fake_run_stage_sequence)
    monkeypatch.setattr(experiment_matrix, "_assert_required_run_artifacts", _fake_assert_required_run_artifacts)
    monkeypatch.setattr(
        experiment_matrix,
        "_read_evaluation_report_for_run",
        _fake_read_evaluation_report_for_run,
    )

    grid_item_cfg: Dict[str, Any] = {
            "grid_index": 0,
            "grid_item_digest": "grid_digest",
            "cfg_digest": "cfg_from_grid_item",
            "ablation_digest": "ablation_digest",
            "attack_protocol_digest": "attack_digest",
            "attack_protocol_version": "attack_protocol_v1",
        }
    summary = experiment_matrix.run_single_experiment(grid_item_cfg)

    assert summary.get("status") == "ok"
    assert summary.get("cfg_digest") == "cfg_from_evaluate_record"
    assert summary.get("thresholds_digest") == "thresholds_from_evaluate_record"
    assert summary.get("threshold_metadata_digest") == "threshold_meta_from_evaluate_record"
    assert summary.get("impl_digest") == "impl_identity_digest_from_run_closure"
    comparison_payload = summary.get("t2smark_comparison")
    assert isinstance(comparison_payload, dict)
    assert comparison_payload.get("comparison_source") == "real_t2smark_baseline_required"
