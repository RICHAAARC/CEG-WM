from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments import run_paper_baseline_worker as worker


@pytest.mark.unit
def test_validate_only_freezes_all_baseline_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = worker.main([
        "--method", "shallow_diffuse",
        "--job-id", "paper-baseline-shallow-diffuse-v1",
        "--expected-exact", "a" * 40,
        "--drive-root", "/unused",
        "--runtime-root", "/unused",
        "--validate-only",
    ])
    assert rc == 0
    output = capsys.readouterr().out
    assert '"model_execution": false' in output
    assert '"threshold_calibration": 2000' in output
    assert worker.METHOD_SPECS["shallow_diffuse"]["score_id"] == "negative_mask_l1diff_mean"


@pytest.mark.unit
def test_formal_worker_has_no_canary_lock_force_or_hash_gate() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    for forbidden in ("RunLock", "force-rerun-all", "sha256_file", "final_manifest.json"):
        assert forbidden not in source


@pytest.mark.unit
def test_unavailable_threshold_keeps_all_planned_evaluation_cells() -> None:
    summaries = worker._empty_evaluation()
    assert len(summaries) == 12
    assert all(summary["n_planned"] == 1000 for summary in summaries.values())
    assert all(summary["n_missing"] == 1000 for summary in summaries.values())


@pytest.mark.unit
def test_baseline_preflight_failure_writes_no_formal_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "verify_expected_exact", lambda exact: None)
    monkeypatch.setattr(
        worker, "_prepare_runtime",
        lambda method, runtime_root, token: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    assert worker.run_worker(
        method="tree_ring", job_id="paper-baseline-treering-v1",
        expected_exact="a" * 40, drive_root=tmp_path,
        runtime_root=tmp_path / "runtime",
    ) == 3
    root = tmp_path / "paper-baseline-treering-v1"
    state = json.loads((root / "job_state.json").read_text())
    assert state["status"] == "PREFLIGHT_FAILED_RECOVERABLE"
    assert not (root / "threshold_calibration" / "units").exists()


@pytest.mark.unit
def test_baseline_canary_exercises_checkpoint_and_resume_with_science_n_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        method = "tree_ring"

        def pair(self, prompt: str, seed: int) -> tuple[Image.Image, Image.Image]:
            del prompt, seed
            return (
                Image.new("RGB", (512, 512), (1, 2, 3)),
                Image.new("RGB", (512, 512), (4, 5, 6)),
            )

    monkeypatch.setattr(worker, "verify_expected_exact", lambda exact: None)
    monkeypatch.setattr(worker, "_prepare_runtime", lambda method, root, token: Runtime())
    monkeypatch.setattr(worker, "_quality", lambda clean, marked: {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1})
    monkeypatch.setattr(
        worker, "_score_payload",
        lambda runtime, image, tau=None: {"normalized_score": 1.0, "decision": True},
    )
    assert worker.run_engineering_canary(
        method="tree_ring", job_id="paper-baseline-treering-canary-v1",
        expected_exact="a" * 40, drive_root=tmp_path,
        runtime_root=tmp_path / "runtime",
    ) == 0
    final = json.loads((
        tmp_path / "paper-baseline-treering-canary-v1" / "canary_final.json"
    ).read_text())
    assert final["science_denominator"] == 0
    assert final["checkpoint_count"] == 2
    assert final["resume_verified"] is True
