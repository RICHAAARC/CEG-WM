from __future__ import annotations

import inspect
import math

import pytest

from cegwm.geometry_v7.r2 import (
    Candidate,
    FeatureRow,
    OutcomeRow,
    R2_DEV_NO_FEASIBLE,
    R2_CONDITION_IDS,
    R2_FAILED,
    R2_PASSED_ALL,
    R2_PASSED_PARTIAL,
    evaluate_frozen_candidate,
    feature_row_from_geometry,
    generate_candidates,
    outcome_row_from_repair,
    select_candidate,
)


CONDITIONS = R2_CONDITION_IDS


def _features(split: str, *, raw_offset: float = 0.0, invalid=()) -> tuple[FeatureRow, ...]:
    indices = range(1, 5) if split == "dev" else range(5, 9)
    return tuple(
        FeatureRow(
            split, condition, f"content-v6-iss-eval-{index:04d}",
            (condition, index) not in invalid,
            raw_offset + index / 10.0, 3.0 + index / 100.0,
            0.8 + index / 100.0, 1.0 + index / 100.0,
            () if (condition, index) not in invalid else ("invalid",),
        )
        for condition in CONDITIONS for index in indices
    )


def _outcomes(split: str, *, unsafe=(), membership="N_recovery_negative") -> tuple[OutcomeRow, ...]:
    indices = range(1, 5) if split == "dev" else range(5, 9)
    return tuple(
        OutcomeRow(split, condition, f"content-v6-iss-eval-{index:04d}", membership,
                   True, (condition, index) not in unsafe,
                   membership == "N_recovery_negative" and (condition, index) not in unsafe,
                   membership != "N_recovery_negative", True, False)
        for condition in CONDITIONS for index in indices
    )


@pytest.mark.integration
def test_geometry_features_exact_and_invalid_fail_closed() -> None:
    geometry = {
        "status": "UNRELIABLE", "uncalibrated_sync_logit": -0.25,
        "observed_corners_in_canonical_normalized": ((-1,-1),(1,-1),(1,1),(-1,1)),
        "homography_observed_to_canonical": ((1,0,0),(0,1,0),(0,0,1)),
        "legal": True, "error": None,
    }
    row = feature_row_from_geometry(split="dev", condition_id="a", unit_id="u", geometry=geometry)
    assert row.mandatory_valid
    assert row.raw_logit == -0.25
    assert row.kappa_f == pytest.approx(3.0)
    assert row.coverage == pytest.approx(1.0)
    assert row.area_ratio == pytest.approx(1.0)

    for change in (
        {"uncalibrated_sync_logit": float("nan")},
        {"observed_corners_in_canonical_normalized": ((-1,-1),(1,1),(1,-1),(-1,1))},
        {"homography_observed_to_canonical": ((1,0,0),(0,0,0),(0,0,1))},
        {"legal": False}, {"error": "bad"}, {"status": "ERROR"},
    ):
        assert not feature_row_from_geometry(
            split="dev", condition_id="a", unit_id="u", geometry={**geometry, **change}
        ).mandatory_valid


@pytest.mark.integration
def test_type7_candidates_are_canonical_deduplicated_and_different_feature_pairs() -> None:
    candidates = generate_candidates(_features("dev"))
    assert candidates[0].candidate_id == "A|LEGAL_ONLY"
    singles = [candidate for candidate in candidates if candidate.complexity == 1]
    assert singles[0].candidate_id.startswith("B|raw_logit|le|")
    assert singles[4].candidate_id.startswith("B|raw_logit|ge|")
    assert all(component.threshold.hex() in candidate.candidate_id
               for candidate in singles for component in candidate.components)
    pairs = [candidate for candidate in candidates if candidate.complexity == 2]
    assert pairs and all(pair.components[0].feature != pair.components[1].feature for pair in pairs)
    # Forty pooled values repeat each unit value once per attack; type-7 stays deterministic.
    raw_thresholds = [c.components[0].threshold for c in singles
                      if c.components[0].feature == "raw_logit" and c.components[0].direction == "le"]
    assert raw_thresholds[0] == pytest.approx(0.1)
    assert raw_thresholds[-1] == pytest.approx(0.4)


@pytest.mark.integration
def test_dev_selection_isolated_from_test_and_rejects_wrong_split() -> None:
    dev_features, dev_outcomes = _features("dev"), _outcomes("dev")
    selection = select_candidate(dev_features, dev_outcomes)
    assert selection.selected is not None
    frozen_id = selection.selected.candidate_id
    status_a, _ = evaluate_frozen_candidate(selection.selected, _features("test"), _outcomes("test"))
    status_b, _ = evaluate_frozen_candidate(
        selection.selected, _features("test", raw_offset=1000),
        _outcomes("test", unsafe={(condition, index) for condition in CONDITIONS for index in range(5,9)}),
    )
    assert selection.selected.candidate_id == frozen_id
    assert status_a == R2_PASSED_ALL
    assert status_b == R2_FAILED
    with pytest.raises(ValueError):
        select_candidate((*dev_features, *_features("test")), dev_outcomes)
    with pytest.raises(ValueError):
        select_candidate(_features("test"), _outcomes("test"))
    source = inspect.getsource(evaluate_frozen_candidate)
    assert "generate_candidates" not in source and "quantile" not in source and "rank" not in source


@pytest.mark.integration
def test_fixed_denominator_gates_ranking_and_all_reject() -> None:
    features = _features("dev")
    unsafe = {(condition, 1) for condition in CONDITIONS[:5]}
    selection = select_candidate(features, _outcomes("dev", unsafe=unsafe))
    assert selection.selected is not None
    legal = next(item for item in selection.candidate_table if item.candidate_id == "A|LEGAL_ONLY")
    assert legal.accepted_count == 40
    assert legal.unsafe_accept_count == 5
    assert legal.selective_risk == pytest.approx(0.125)
    invalid = _features("dev", invalid={(condition, index) for condition in CONDITIONS for index in range(1,5)})
    failed = select_candidate(invalid, _outcomes("dev"))
    assert failed.status == R2_DEV_NO_FEASIBLE and failed.selected is None
    inconsistent = list(_features("dev"))
    row = inconsistent[0]
    inconsistent[0] = FeatureRow(row.split, row.condition_id, row.unit_id, True,
                                 None, row.kappa_f, row.coverage, row.area_ratio)
    legal = next(item for item in select_candidate(tuple(inconsistent), _outcomes("dev")).candidate_table
                 if item.candidate_id == "A|LEGAL_ONLY")
    assert legal.accepted_count == 39


@pytest.mark.integration
def test_partial_family_pass_and_bd_outcome_semantics() -> None:
    selection = select_candidate(_features("dev"), _outcomes("dev"))
    invalid = {(condition, index) for condition in CONDITIONS[8:] for index in range(5,9)}
    status, metrics = evaluate_frozen_candidate(
        selection.selected, _features("test", invalid=invalid), _outcomes("test")
    )
    assert metrics.covered_attack_count == 8
    assert status == R2_PASSED_PARTIAL

    scores = {"u": {}, "g": {}, "cg": {},
              "positive_cg_vs_g": {"positive": True}, "negative_g_vs_u": {}}
    base = {"errors": [], "scores": scores, "positive_gate_a_delta": 0.1,
            "positive_gate_b_delta": 0.1, "positive_score_delta": 0.1,
            "improved": True, "recovered_negative": True, "decision_harm": False,
            "observed_negative_false_positive": False}
    n = outcome_row_from_repair(split="dev", condition_id="a", unit_id="u",
                                membership="N_recovery_negative", record=base)
    b = outcome_row_from_repair(split="dev", condition_id="a", unit_id="u",
                                membership="B_boundary", record=base)
    d = outcome_row_from_repair(split="dev", condition_id="a", unit_id="u",
                                membership="D_damage_only", record=base)
    assert n.safe and n.safe_rescue and not n.baseline_positive
    assert b.safe and not b.safe_rescue and b.baseline_positive
    assert d.safe and not d.safe_rescue and d.baseline_positive
