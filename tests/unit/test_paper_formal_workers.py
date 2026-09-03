from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

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
def test_reconstruction_missing_prerequisites_waits_without_terminal_or_model(
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
    root = tmp_path / "reconstruction" / "paper-main-reconstruction-v1"
    state = json.loads((root / "job_state.json").read_text())
    assert state["status"] == "WAITING_FOR_PREREQUISITE"
    assert state["science_denominator"] == 0
    assert not (root / "reconstruction_final.json").exists()
    assert not (root / "generation" / "units").exists()

    main_root = tmp_path / "main" / "paper-main-v1"
    main_root.mkdir(parents=True)
    (main_root / "threshold.json").write_text(json.dumps({
        "method_id": main_worker.METHOD_ID,
        "producer_exact": "a" * 40,
        "tau": 0.0,
    }), encoding="utf-8")
    generation = main_root / "evaluation_generation"
    generation.mkdir()
    (generation / "final_result.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        reconstruction_worker, "_prepare_runtime",
        lambda config, runtime_root, threshold: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    assert reconstruction_worker.run_worker(
        job_id="paper-main-reconstruction-v1", main_job_id="paper-main-v1",
        expected_exact="a" * 40, drive_root=tmp_path, runtime_root=tmp_path / "runtime",
    ) == 3
    state = json.loads((root / "job_state.json").read_text())
    assert state["status"] == "PREFLIGHT_FAILED_RECOVERABLE"
    assert not (root / "reconstruction_final.json").exists()


@pytest.mark.unit
def test_reconstruction_explicit_close_publishes_fixed_missing_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reconstruction_worker, "_verify_exact", lambda exact: None)
    assert reconstruction_worker.run_worker(
        job_id="paper-main-reconstruction-v1", main_job_id="paper-main-v1",
        expected_exact="a" * 40, drive_root=tmp_path, runtime_root=tmp_path / "runtime",
        finalize_incomplete=True,
    ) == 0
    result = json.loads((
        tmp_path / "reconstruction" / "paper-main-reconstruction-v1" / "reconstruction_final.json"
    ).read_text())
    assert result["closure_mode"] == "EXPLICIT_FINALIZE_INCOMPLETE"
    assert result["summaries"]["negative"]["n_missing"] == 100


@pytest.mark.unit
def test_finalizer_waits_by_default_and_only_explicitly_closes_missing_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_verify_checkout", lambda exact: None)
    assert finalizer.run_finalize(
        drive_root=tmp_path, expected_exact="a" * 40, baseline_exact="b" * 40,
    ) == 0
    output = tmp_path / "finalized" / "paper-formal-v1"
    state = json.loads((output / "job_state.json").read_text())
    assert state["status"] == "WAITING_FOR_REQUIRED_RESULTS"
    assert not (output / "unified_result_package.json").exists()
    assert not (output / "unified_main_table_long.csv").exists()

    assert finalizer.run_finalize(
        drive_root=tmp_path, expected_exact="a" * 40, baseline_exact="b" * 40,
        finalize_incomplete=True,
    ) == 0
    result = json.loads((
        tmp_path / "finalized" / "paper-formal-v1" / "unified_result_package.json"
    ).read_text())
    assert result["status"] == "INCOMPLETE_OPERATIONAL"
    assert result["closure_mode"] == "EXPLICIT_FINALIZE_INCOMPLETE"
    assert set(result["methods"]) == {
        main_worker.METHOD_ID, "t2smark", "tree_ring", "gaussian_shading", "shallow_diffuse",
    }
    assert all(method["evaluation"]["clean_no_attack:negative"]["n_missing"] == 1000 for method in result["methods"].values())
    assert all(method["quality"]["n_missing_pairs"] == 1000 for method in result["methods"].values())
    assert result["reconstruction_supplement"]["summaries"]["positive"]["n_missing"] == 100
    assert (tmp_path / "finalized" / "paper-formal-v1" / "unified_main_table_long.csv").exists()


@pytest.mark.unit
def test_main_preflight_failure_writes_no_formal_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_worker, "verify_expected_exact", lambda exact: None)
    monkeypatch.setattr(
        main_worker, "_prepare_runtime",
        lambda runtime_root: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    assert main_worker.run_worker(
        job_id="paper-main-v1", expected_exact="a" * 40,
        drive_root=tmp_path, runtime_root=tmp_path / "runtime",
    ) == 3
    root = tmp_path / "paper-main-v1"
    state = json.loads((root / "job_state.json").read_text())
    assert state["status"] == "PREFLIGHT_FAILED_RECOVERABLE"
    assert not (root / "threshold_calibration" / "units").exists()


@pytest.mark.unit
def test_main_canary_exercises_checkpoint_and_resume_with_science_n_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_worker, "verify_expected_exact", lambda exact: None)
    monkeypatch.setattr(main_worker, "_prepare_runtime", lambda runtime_root: {})
    monkeypatch.setattr(
        main_worker, "_main_pair",
        lambda runtime, prompt, seed: (
            Image.new("RGB", (512, 512), (1, 2, 3)),
            Image.new("RGB", (512, 512), (4, 5, 6)),
        ),
    )
    monkeypatch.setattr(main_worker, "_quality", lambda clean, marked: {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1})
    monkeypatch.setattr(
        main_worker, "_detect_payload",
        lambda runtime, image, tau: {"normalized_score": 1.0, "decision": True},
    )
    assert main_worker.run_engineering_canary(
        job_id="paper-main-canary-v1", expected_exact="a" * 40,
        drive_root=tmp_path, runtime_root=tmp_path / "runtime",
    ) == 0
    final = json.loads((tmp_path / "paper-main-canary-v1" / "canary_final.json").read_text())
    assert final["science_denominator"] == 0
    assert final["checkpoint_count"] == 2
    assert final["resume_verified"] is True


@pytest.mark.unit
def test_reconstruction_canary_exercises_checkpoint_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reconstruction_worker, "_verify_exact", lambda exact: None)
    monkeypatch.setattr(reconstruction_worker, "_prepare_runtime", lambda config, root, threshold: (object(), {}))
    monkeypatch.setattr(
        reconstruction_worker, "_reconstruct_once",
        lambda pipeline, source, attack, seed: Image.new("RGB", (512, 512), (7, 8, 9)),
    )
    monkeypatch.setattr(
        reconstruction_worker, "_detect_payload",
        lambda runtime, image, tau: {"normalized_score": 1.0, "decision": True},
    )
    assert reconstruction_worker.run_engineering_canary(
        job_id="paper-reconstruction-canary-v1", expected_exact="a" * 40,
        drive_root=tmp_path, runtime_root=tmp_path / "runtime",
    ) == 0
    final = json.loads((tmp_path / "paper-reconstruction-canary-v1" / "canary_final.json").read_text())
    assert final["status"] == "ENGINEERING_CANARY_COMPLETE"
    assert final["checkpoint_count"] == 2


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
