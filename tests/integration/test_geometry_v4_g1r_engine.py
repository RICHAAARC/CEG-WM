from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments import geometry_v4_g1r_engine as engine
from cegwm.protocol.geometry_v4_g1r import ATTACKS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_real_rosters_are_complete_disjoint_and_have_no_subset_interface() -> None:
    development = engine.build_real_roster(ROOT, "development")
    confirmation = engine.build_real_roster(ROOT, "confirmation")
    assert len(development) == len(confirmation) == 20
    assert {item[0] for item in development} == {6201, 6202, 6203, 6204}
    assert {item[0] for item in confirmation} == {6301, 6302, 6303, 6304}
    assert not {item[0] for item in development} & {item[0] for item in confirmation}
    assert tuple(item[2] for item in development) == ATTACKS * 4
    assert tuple(inspect.signature(engine.build_real_roster).parameters) == ("repo_root", "split")


@pytest.mark.integration
def test_truth_is_attached_only_after_blind_three_arm_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tuple(inspect.signature(engine._blind_arms).parameters) == ("attacked_marked", "attacked_negative")
    assert tuple(inspect.signature(engine._apply_attack).parameters) == ("rgb", "attack")
    assert "truth" not in inspect.getsource(engine._apply_attack)
    assert "truth_transform" not in inspect.getsource(engine._blind_arms)
    assert tuple(inspect.signature(engine._evaluate_frozen_arms).parameters) == ("arms", "truth_attacked_to_canonical")

    events: list[str] = []
    unreliable = {"H_hat": None, "corners_hat": (), "support": 0, "reliability": 0.0, "status": "UNRELIABLE"}
    monkeypatch.setattr(engine, "_blind_arms", lambda marked, negative: events.append("blind") or engine.BlindArms(unreliable, unreliable, unreliable))
    monkeypatch.setattr(engine, "_truth_for_attack", lambda attack: events.append("truth") or np.eye(3))
    monkeypatch.setattr(engine, "_evaluate_frozen_arms", lambda arms, truth: events.append("evaluate") or {name: {"status": "UNRELIABLE", "support": 0, "truth_metrics": {}, "unsafe": False} for name in ("correct", "wrong", "negative")})
    records = engine.run_cpu_canary()
    assert len(records) == 20 and events == [item for _ in range(20) for item in ("blind", "truth", "evaluate")]


@pytest.mark.integration
def test_truth_metric_is_attacked_to_canonical_and_normalized_diagonal() -> None:
    truth = engine._truth_for_attack("translation_0.08_0")
    reliable = {"status": "RELIABLE", "H_hat": tuple(float(value) for value in truth.reshape(-1))}
    metrics = engine._truth_metrics(reliable, truth)
    assert metrics == pytest.approx({"mapped_corner_error": 0.0, "center_reprojection_error": 0.0, "rotation_abs_error_degrees": 0.0, "log_scale_abs_error": 0.0})


@pytest.mark.integration
def test_fixed_cpu_three_arm_canary_reaches_engineering_exit() -> None:
    records = engine.run_cpu_canary()
    summary = engine.summarize_cpu_canary(records)
    assert len(records) == 20 and all(record["failure"] is None for record in records)
    assert summary["formal_denominator"] == 0 and summary["units"] == 20
    assert summary["correct_safe_reliable"] >= 18
    assert summary["correct_unsafe"] == summary["wrong_unsafe"] == summary["negative_unsafe"] == 0
    assert all(value >= 3 for value in summary["correct_safe_by_attack"].values())
    assert summary["status"] == "CPU_ENGINEERING_EXIT"
