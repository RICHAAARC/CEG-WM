"""
File purpose: 验证 workflow acceptance 路径视图的正规化与相对路径补充。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_mini_real_validation as mini_real_validation
import scripts.workflow_acceptance_common as workflow_acceptance_common


def test_collect_workflow_state_emits_relative_paths(tmp_path: Path) -> None:
    """
    功能：验证 collect_workflow_state 会输出 paths_relative。 

    Verify collect_workflow_state emits normalized absolute and run_root-relative path views.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    run_root = tmp_path / "run_root"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "signoff").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (run_root / "outputs" / "experiment_matrix" / "artifacts").mkdir(parents=True, exist_ok=True)

    state = workflow_acceptance_common.collect_workflow_state(run_root)

    assert state["paths"]["run_root"] == run_root.as_posix()
    assert state["paths_relative"]["run_root"] == "."
    assert state["paths_relative"]["embed_record"] == "records/embed_record.json"
    assert state["paths_relative"]["signoff_report"] == "artifacts/signoff/signoff_report.json"
    assert state["paths_relative"]["experiment_matrix_summary"] == "outputs/experiment_matrix/artifacts/grid_summary.json"


def test_mini_real_summary_emits_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能：验证 mini-real summary 会补充 paths_relative 与 repo-relative config path。 

    Verify mini-real validation summary includes relative path views and repo-relative config path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    run_root = tmp_path / "mini_real_run"
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "paper_full_cuda_mini_real_validation.yaml"

    def _fake_load_runtime_config(_cfg_path: Path) -> dict[str, object]:
        return {
            "experiment_matrix": {
                "require_real_negative_cache": True,
                "require_shared_thresholds": True,
                "disallow_forced_pair_fallback": True,
            }
        }

    def _fake_collect_workflow_state(_run_root: Path) -> dict[str, object]:
        return {
            "exists": {
                "embed_record": True,
                "detect_record": True,
                "calibration_record": True,
                "evaluate_record": True,
                "evaluation_report": True,
                "thresholds_artifact": True,
                "experiment_matrix_summary": True,
                "signoff_report": True,
            },
            "paths": {
                "run_root": run_root.as_posix(),
                "detect_record": (run_root / "records" / "detect_record.json").as_posix(),
                "thresholds_artifact": (run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").as_posix(),
                "experiment_matrix_summary": (run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json").as_posix(),
            },
            "paths_relative": {
                "run_root": ".",
                "detect_record": "records/detect_record.json",
                "thresholds_artifact": "artifacts/thresholds/thresholds_artifact.json",
                "experiment_matrix_summary": "outputs/experiment_matrix/artifacts/grid_summary.json",
            },
            "detect_record": {
                "content_evidence": {"status": "ok"},
                "geometry_evidence": {"status": "ok"},
            },
        }

    def _fake_load_json_dict(_path: Path) -> dict[str, object]:
        return {"status": {"reason": "ok", "details": None}}

    def _fake_load_matrix_summary(_run_root: Path) -> dict[str, int]:
        return {"total": 16, "failed": 0}

    def _fake_extract_matrix_log_failure(_run_root: Path) -> str:
        return "<absent>"

    monkeypatch.setattr(mini_real_validation, "load_runtime_config", _fake_load_runtime_config)
    monkeypatch.setattr(mini_real_validation, "collect_workflow_state", _fake_collect_workflow_state)
    monkeypatch.setattr(mini_real_validation, "load_json_dict", _fake_load_json_dict)
    monkeypatch.setattr(mini_real_validation, "_load_matrix_summary", _fake_load_matrix_summary)
    monkeypatch.setattr(mini_real_validation, "_extract_experiment_matrix_log_failure", _fake_extract_matrix_log_failure)

    build_summary = getattr(mini_real_validation, "_build_mini_real_validation_summary")
    summary = build_summary(
        run_root=run_root,
        cfg_path=cfg_path,
        workflow_exit_code=0,
        preflight={"ok": True},
    )

    assert summary["run_root"] == run_root.as_posix()
    assert summary["run_root_relative"] == "."
    assert summary["config_path_repo_relative"] == "configs/paper_full_cuda_mini_real_validation.yaml"
    assert summary["paths_relative"]["run_closure"] == "artifacts/run_closure.json"
    assert summary["paths_relative"]["workflow_log"] == "logs/mini_real_workflow_execution.log"
