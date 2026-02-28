"""
文件目的：验证批量调度的子实验 run_root 独立性。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

from main.evaluation.experiment_matrix import run_experiment_grid


def test_batch_runner_independent_runs(tmp_path: Path, monkeypatch) -> None:
    """
    功能：验证每个 grid item 的 run_root 不冲突。

    Assert each grid item obtains an independent run_root path.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    created_run_roots: list[Path] = []

    def _fake_run_stage_command(stage_name: str, run_root: Path, config_path: Path, stage_overrides: list[str]) -> None:
        _ = config_path
        _ = stage_overrides
        created_run_roots.append(run_root)
        records_dir = run_root / "records"
        artifacts_dir = run_root / "artifacts"
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if stage_name == "detect":
            detect_record_payload = {
                "operation": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.123,
                },
            }
            (records_dir / "detect_record.json").write_text(
                json.dumps(detect_record_payload),
                encoding="utf-8",
            )
        if stage_name == "evaluate":
            eval_report_payload = {
                "evaluation_report": {
                    "cfg_digest": "cfg_digest_x",
                    "plan_digest": "plan_digest_x",
                    "thresholds_digest": "thresholds_digest_x",
                    "threshold_metadata_digest": "threshold_metadata_digest_x",
                    "attack_protocol_digest": "attack_protocol_digest_x",
                    "attack_protocol_version": "attack_protocol_v1",
                    "impl_digest": "impl_digest_x",
                    "fusion_rule_version": "fusion_v1",
                    "metrics": {
                        "tpr_at_fpr_primary": 1.0,
                        "geo_available_rate": 0.5,
                        "rescue_rate": 0.1,
                        "reject_rate": 0.2,
                        "reject_rate_by_reason": {"status_not_ok": 0.2},
                    },
                }
            }
            (artifacts_dir / "eval_report.json").write_text(
                json.dumps(eval_report_payload),
                encoding="utf-8",
            )
            (artifacts_dir / "run_closure.json").write_text("{}", encoding="utf-8")
            (records_dir / "evaluate_record.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "main.evaluation.experiment_matrix._run_stage_command",
        _fake_run_stage_command,
    )

    grid = [
        {
            "grid_index": 0,
            "grid_item_digest": "a" * 64,
            "cfg_snapshot": {"seed": 1, "model_id": "model_a"},
            "ablation_flags": {"enable_geometry": True},
            "attack_protocol_version": "attack_protocol_v1",
            "batch_root": str(tmp_path / "batch_root"),
            "config_path": "configs/default.yaml",
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "max_samples": 2,
        },
        {
            "grid_index": 1,
            "grid_item_digest": "b" * 64,
            "cfg_snapshot": {"seed": 2, "model_id": "model_b"},
            "ablation_flags": {"enable_geometry": False},
            "attack_protocol_version": "attack_protocol_v1",
            "batch_root": str(tmp_path / "batch_root"),
            "config_path": "configs/default.yaml",
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "max_samples": 2,
        },
    ]

    summary = run_experiment_grid(grid, strict=True)
    run_roots = [item["run_root"] for item in summary["results"]]

    assert len(run_roots) == 2
    assert len(set(run_roots)) == 2
