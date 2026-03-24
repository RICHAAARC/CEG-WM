"""
File purpose: detached parallel_attestation_statistics workflow tests.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import run_parallel_attestation_statistics as module


def test_run_parallel_attestation_statistics_builds_detached_config_and_runs_stages(monkeypatch, tmp_path: Path) -> None:
    """
    功能：独立专项脚本必须使用配置副本执行 calibrate 与 evaluate。

    Verify the detached parallel statistics script writes a config copy and runs
    calibrate/evaluate without score-name CLI overrides.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory.

    Returns:
        None.
    """
    stage_order = []
    main_run_root = tmp_path / "main_run"
    main_records_dir = main_run_root / "records"
    main_records_dir.mkdir(parents=True, exist_ok=True)
    (main_records_dir / "detect_record.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        module.paper_full_cuda_workflow,
        "_load_runtime_config",
        lambda _path: {
            "parallel_attestation_statistics": {
                "enabled": True,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            },
            "calibration": {},
            "evaluate": {},
        },
    )

    def _fake_run_step(step_name, command):
        stage_order.append((step_name, command))
        return 0

    monkeypatch.setattr(module, "_run_step", _fake_run_step)

    result = module.run_parallel_attestation_statistics(Path("configs/paper_full_cuda.yaml"), main_run_root)

    assert result == 0
    assert [item[0] for item in stage_order] == ["calibrate", "evaluate"]

    parallel_run_root = module.paper_full_cuda_workflow._build_parallel_attestation_run_root(main_run_root)
    detached_cfg_path = parallel_run_root / "runtime_metadata" / "parallel_attestation_statistics_config.yaml"
    assert detached_cfg_path.exists()

    detached_cfg = yaml.safe_load(detached_cfg_path.read_text(encoding="utf-8"))
    assert detached_cfg["calibration"]["score_name"] == "event_attestation_score"
    assert detached_cfg["evaluate"]["score_name"] == "event_attestation_score"
    calibration_glob = Path(detached_cfg["calibration"]["detect_records_glob"])
    evaluate_glob = Path(detached_cfg["evaluate"]["detect_records_glob"])
    thresholds_path = Path(detached_cfg["evaluate"]["thresholds_path"])
    assert calibration_glob.parent.name == "records"
    assert calibration_glob.name == "*detect*.json"
    assert evaluate_glob.parent.name == "records"
    assert evaluate_glob.name == "*detect*.json"
    assert thresholds_path.parent.name == "thresholds"
    assert thresholds_path.name == "thresholds_artifact.json"
    assert all("score_name" not in command for _stage_name, command in stage_order)
    assert all("run_root_reuse_allowed=true" in command for _stage_name, command in stage_order)
    assert all(any("run_root_reuse_reason=" in item for item in command) for _stage_name, command in stage_order)

    summary_path = parallel_run_root / "artifacts" / "parallel_attestation_statistics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"


def test_run_parallel_attestation_statistics_stops_on_stage_failure(monkeypatch, tmp_path: Path) -> None:
    """
    功能：独立专项脚本在 calibrate 失败时必须立即失败并保留摘要。

    Verify the detached workflow stops on the first failing stage.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory.

    Returns:
        None.
    """
    main_run_root = tmp_path / "main_run"
    main_records_dir = main_run_root / "records"
    main_records_dir.mkdir(parents=True, exist_ok=True)
    (main_records_dir / "detect_record.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        module.paper_full_cuda_workflow,
        "_load_runtime_config",
        lambda _path: {
            "parallel_attestation_statistics": {
                "enabled": True,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            },
            "calibration": {},
            "evaluate": {},
        },
    )
    monkeypatch.setattr(module, "_run_step", lambda _step_name, _command: 3)

    result = module.run_parallel_attestation_statistics(Path("configs/paper_full_cuda.yaml"), main_run_root)

    assert result == 3
    parallel_run_root = module.paper_full_cuda_workflow._build_parallel_attestation_run_root(main_run_root)
    summary_path = parallel_run_root / "artifacts" / "parallel_attestation_statistics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["stages"]["calibrate"]["return_code"] == 3