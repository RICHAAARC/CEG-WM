from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import run_paper_main_worker as main_worker
from experiments import run_paper_reconstruction_worker as reconstruction_worker
from experiments import run_paper_results_finalize as finalizer


@pytest.mark.unit
def test_worker_validate_only_paths_do_not_load_models(capsys: pytest.CaptureFixture[str]) -> None:
    common = [
        "--job-id", "paper-main-v1", "--expected-exact", "a" * 40,
        "--drive-root", "/unused", "--runtime-root", "/unused", "--validate-only",
    ]
    assert main_worker.main(common) == 0
    assert '"model_execution": false' in capsys.readouterr().out
    assert reconstruction_worker.main([
        "--job-id", "paper-main-reconstruction-v1", "--main-job-id", "paper-main-v1",
        "--expected-exact", "a" * 40, "--drive-root", "/unused",
        "--runtime-root", "/unused", "--validate-only",
    ]) == 0
    output = capsys.readouterr().out
    assert '"pair_count": 100' in output and '"fpr_resolution": 0.01' in output
    assert finalizer.main([
        "--drive-root", "/unused", "--expected-exact", "a" * 40,
        "--baseline-exact", "b" * 40, "--validate-only",
    ]) == 0
    assert '"model_execution": false' in capsys.readouterr().out


@pytest.mark.unit
def test_reconstruction_missing_prerequisites_publishes_terminal_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reconstruction_worker, "_verify_exact", lambda exact: None)
    monkeypatch.setattr(
        reconstruction_worker,
        "_load_reconstruction_pipeline",
        lambda config: pytest.fail("model must not load when prerequisites are missing"),
    )
    assert reconstruction_worker.run_worker(
        job_id="paper-main-reconstruction-v1", main_job_id="paper-main-v1",
        expected_exact="a" * 40, drive_root=tmp_path, runtime_root=tmp_path / "runtime",
    ) == 0
    result = json.loads((
        tmp_path / "reconstruction" / "paper-main-reconstruction-v1" / "reconstruction_final.json"
    ).read_text())
    assert result["status"] == "INCOMPLETE_OPERATIONAL"
    assert result["model_loaded"] is False
    assert result["summaries"]["negative"]["n_missing"] == 100


@pytest.mark.unit
def test_finalizer_preserves_missing_methods_in_unified_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_verify_checkout", lambda exact: None)
    assert finalizer.run_finalize(
        drive_root=tmp_path, expected_exact="a" * 40, baseline_exact="b" * 40,
    ) == 0
    result = json.loads((
        tmp_path / "finalized" / "paper-formal-v1" / "unified_result_package.json"
    ).read_text())
    assert result["status"] == "INCOMPLETE_OPERATIONAL"
    assert set(result["methods"]) == {
        main_worker.METHOD_ID, "t2smark", "tree_ring", "gaussian_shading", "shallow_diffuse",
    }
    assert all(method["evaluation"]["clean_no_attack:negative"]["n_missing"] == 1000 for method in result["methods"].values())
    assert result["reconstruction_supplement"]["summaries"]["positive"]["n_missing"] == 100
    assert (tmp_path / "finalized" / "paper-formal-v1" / "unified_main_table_long.csv").exists()


@pytest.mark.unit
def test_main_ablation_contract_is_minimal_and_fixed() -> None:
    config = main_worker.load_formal_config(main_worker.CONFIG_PATH)
    assert config["ablations"]["subset_size"] == 100
    assert tuple(config["ablations"]["variants"]) == (
        "no_content_adaptive", "lf_only", "hf_only", "no_geometry",
    )
    assert config["ablations"]["variants"]["no_geometry"]["threshold_role"].startswith("controlled")
    summaries = main_worker._empty_ablations(config)
    assert summaries and all(value["n_planned"] == 100 for value in summaries.values())
