from __future__ import annotations

from pathlib import Path

import pytest

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
