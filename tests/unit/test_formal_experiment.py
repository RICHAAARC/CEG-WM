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
    execute_with_frozen_retry,
    expand_rosters,
    freeze_threshold,
    load_formal_config,
    summarize_binary,
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


@pytest.mark.unit
def test_six_attacks_are_deterministic_rgb() -> None:
    image = Image.new("RGB", (32, 32), (100, 120, 140))
    for condition in FORMAL_CONDITIONS:
        first = apply_attack(image, condition)
        second = apply_attack(image, condition)
        assert first.mode == "RGB" and first.size == image.size
        assert first.tobytes() == second.tobytes()
