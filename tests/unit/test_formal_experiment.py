from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cegwm.formal_experiment import (
    FORMAL_CONDITIONS,
    FormalRunStore,
    OperationalUnitError,
    apply_attack,
    empty_binary_summary,
    execute_job_preflight,
    execute_with_frozen_retry,
    expand_rosters,
    freeze_threshold,
    load_formal_config,
    load_or_recover_pair,
    summarize_binary,
    summarize_quality,
    PreflightFailed,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/paper_experiment/formal_experiment_v1.json"


@pytest.mark.unit
def test_shared_rosters_are_fixed_unique_and_disjoint() -> None:
    config = load_formal_config(CONFIG)
    rosters = expand_rosters(ROOT, config)
    assert {name: len(rows) for name, rows in rosters.items()} == {
        "threshold_calibration": 2000,
        "clean_negative_test": 3000,
        "formal_evaluation_pairs": 1000,
    }
    identities = [
        (row.prompt_id, row.seed)
        for rows in rosters.values()
        for row in rows
    ]
    assert len(identities) == len(set(identities)) == 6000
    assert all("rotation_scale" not in condition for condition in FORMAL_CONDITIONS)


@pytest.mark.unit
def test_threshold_rank_and_strict_equality() -> None:
    threshold = freeze_threshold([float(value) for value in range(2000)])
    assert threshold["rank_one_based"] == 1998
    assert threshold["tau"] == 1997.0
    assert threshold["equality_decision"] == "negative"


@pytest.mark.unit
def test_partial_statistics_keep_conditional_interval_and_planned_bounds() -> None:
    rows = [
        {"terminal_status": "SCORED", "decision": True},
        {"terminal_status": "SCORED", "decision": False},
        {"terminal_status": "OPERATIONAL_FAILURE"},
    ]
    negative = summarize_binary(rows, truth_positive=False, planned=4)
    assert negative["status"] == "INCOMPLETE_OPERATIONAL"
    assert (negative["n_scored"], negative["n_failed"], negative["n_missing"]) == (2, 1, 1)
    assert negative["scored_only_fpr"] == 0.5
    assert (negative["planned_fpr_lower"], negative["planned_fpr_upper"]) == (0.25, 0.75)
    positive = summarize_binary(rows, truth_positive=True, planned=4)
    assert (positive["planned_tpr_lower"], positive["planned_tpr_upper"]) == (0.25, 0.75)

    unavailable = empty_binary_summary(truth_positive=False, planned=3000)
    assert unavailable["n_scored"] == unavailable["n_failed"] == 0
    assert unavailable["n_missing"] == 3000
    assert unavailable["scored_only_fpr"] is None
    assert (unavailable["planned_fpr_lower"], unavailable["planned_fpr_upper"]) == (0.0, 1.0)


@pytest.mark.unit
def test_quality_summary_keeps_valid_failed_and_missing_denominators() -> None:
    rows = [
        {"terminal_status": "SCORED", "quality": {"psnr": 30.0, "ssim": 0.8, "lpips": 0.2}},
        {"terminal_status": "SCORED", "quality": {"psnr": 34.0, "ssim": 0.9, "lpips": 0.1}},
        {"terminal_status": "OPERATIONAL_FAILURE"},
    ]
    summary = summarize_quality(rows, planned=4)
    assert (summary["n_valid_pairs"], summary["n_failed_pairs"], summary["n_missing_pairs"]) == (2, 1, 1)
    assert summary["metrics"]["psnr"]["mean"] == 32.0
    assert summary["status"] == "INCOMPLETE_OPERATIONAL"


@pytest.mark.unit
def test_retry_is_same_unit_allowlisted_bounded_and_attempts_retained() -> None:
    calls: list[int] = []

    def transient(attempt: int) -> dict[str, float]:
        calls.append(attempt)
        if attempt == 1:
            raise OperationalUnitError("CUDA_OOM_TRANSIENT", "score", "oom")
        return {"normalized_score": 1.0}

    row = execute_with_frozen_retry("unit-1", transient)
    assert calls == [1, 2]
    assert row["terminal_status"] == "SCORED"
    assert [attempt["status"] for attempt in row["attempts"]] == ["OPERATIONAL_FAILURE", "SCORED"]

    terminal = execute_with_frozen_retry(
        "unit-2",
        lambda attempt: (_ for _ in ()).throw(
            OperationalUnitError("DATA_CONTRACT", "input", "bad")
        ),
    )
    assert len(terminal["attempts"]) == 1
    assert terminal["terminal_status"] == "OPERATIONAL_FAILURE"

    calls: list[int] = []

    def raw_oom(attempt: int) -> dict[str, float]:
        calls.append(attempt)
        if attempt == 1:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return {"normalized_score": 2.0}

    mapped = execute_with_frozen_retry("unit-3", raw_oom)
    assert calls == [1, 2]
    assert mapped["attempts"][0]["failure_code"] == "CUDA_OOM_TRANSIENT"


@pytest.mark.unit
def test_store_resumes_prefix_without_lock_hash_or_overwrite(tmp_path: Path) -> None:
    identity = {
        "schema_version": "v1", "job_id": "job", "run_id": "run",
        "method_id": "tree_ring", "stage": "test", "expected_exact": "a" * 40,
    }
    store = FormalRunStore(tmp_path, identity, ("u1", "u2"))
    store.initialize()
    store.commit({"unit_id": "u1", "terminal_status": "SCORED", "attempts": [], "normalized_score": 0.0})
    resumed = FormalRunStore(tmp_path, identity, ("u1", "u2"))
    assert len(resumed.rows()) == 1
    resumed.run(lambda unit_id, attempt: {"normalized_score": 1.0})
    resumed.finalize({"status": "COMPLETE"})
    assert json.loads((tmp_path / "final_result.json").read_text())["status"] == "COMPLETE"
    assert not list(tmp_path.rglob("*.sha256"))
    with pytest.raises(RuntimeError, match="out of order"):
        resumed.commit({"unit_id": "u2", "terminal_status": "SCORED"})

    calls: list[str] = []
    FormalRunStore(tmp_path, identity, ("u1", "u2")).run(
        lambda unit_id, attempt: calls.append(unit_id) or {"normalized_score": 9.0}
    )
    assert calls == []


@pytest.mark.unit
def test_store_rejects_a_noncontiguous_prefix(tmp_path: Path) -> None:
    identity = {
        "schema_version": "v1", "job_id": "job", "run_id": "run",
        "method_id": "tree_ring", "stage": "test", "expected_exact": "a" * 40,
    }
    units = tmp_path / "units"
    units.mkdir(parents=True)
    (units / "000001.json").write_text(
        json.dumps({"unit_id": "u2", "terminal_status": "SCORED"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="prefix has a gap"):
        FormalRunStore(tmp_path, identity, ("u1", "u2")).rows()


@pytest.mark.unit
def test_preflight_failure_is_recoverable_and_consumes_no_units(tmp_path: Path) -> None:
    identity = {"job_id": "job", "run_id": "run", "method_id": "baseline", "stage": "preflight", "expected_exact": "a" * 40}
    with pytest.raises(PreflightFailed):
        execute_job_preflight(tmp_path, identity, lambda: (_ for _ in ()).throw(RuntimeError("missing model")))
    state = json.loads((tmp_path / "job_state.json").read_text())
    assert state["status"] == "PREFLIGHT_FAILED_RECOVERABLE"
    assert state["science_denominator"] == 0
    assert not (tmp_path / "units").exists()


@pytest.mark.unit
def test_partial_pair_recovery_preserves_existing_arm(tmp_path: Path) -> None:
    clean, marked = tmp_path / "clean.txt", tmp_path / "marked.txt"
    clean.write_text("existing-clean", encoding="utf-8")
    values = load_or_recover_pair(
        clean, marked,
        lambda: ("regenerated-clean", "regenerated-marked"),
        lambda path: path.read_text(encoding="utf-8"),
        lambda path, value: path.write_text(value, encoding="utf-8"),
    )
    assert values == ("existing-clean", "regenerated-marked", "PAIR_PARTIAL_RECOVERED")
    assert clean.read_text(encoding="utf-8") == "existing-clean"


@pytest.mark.unit
def test_six_attacks_are_deterministic_rgb() -> None:
    image = Image.new("RGB", (32, 32), (100, 120, 140))
    for condition in FORMAL_CONDITIONS:
        first = apply_attack(image, condition)
        second = apply_attack(image, condition)
        assert first.mode == "RGB" and first.size == image.size
        assert first.tobytes() == second.tobytes()
