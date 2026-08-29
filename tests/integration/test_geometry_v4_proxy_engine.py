from __future__ import annotations

import numpy as np
import pytest

from experiments.geometry_v4_proxy_engine import enumerate_unit_identities, run_split, run_unit
from cegwm.protocol.geometry_v4_proxy import P1_ATTACKS

KEY = "0123456789abcdef"
WRONG_KEY = "fedcba9876543210"


def _record_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(*(_record_keys(item) for item in value.values()), set())
    if isinstance(value, (tuple, list)):
        return set().union(*(_record_keys(item) for item in value), set())
    return set()


@pytest.mark.integration
def test_roster_enumeration_keeps_splits_attacks_and_missing_units_in_denominator() -> None:
    p1d = enumerate_unit_identities("P1D")
    p1c = enumerate_unit_identities("P1C")
    assert len(p1d) == len(p1c) == 8 * 16
    assert {item["seed"] for item in p1d} == set(range(4101, 4109))
    assert {item["seed"] for item in p1c} == set(range(4201, 4209))
    assert {item["attack"] for item in p1d} == set(P1_ATTACKS)
    missing = run_split({}, KEY, WRONG_KEY, split="P1D", attacks=("identity",))
    assert len(missing) == 8
    assert all(item["failure"] == "missing_image" for item in missing)


@pytest.mark.integration
def test_nontrivial_translation_runs_attacked_rgb_only_three_arm_chain() -> None:
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    record = run_unit(
        rgb,
        KEY,
        WRONG_KEY,
        split="P1D",
        seed=4101,
        attack="translation_+0.10_0",
    )
    assert record["failure"] is None
    assert set(record["arms"]) == {
        "marked_correct_key",
        "attacked_unwatermarked_negative",
        "same_unit_wrong_key",
    }
    correct = record["arms"]["marked_correct_key"]
    negative = record["arms"]["attacked_unwatermarked_negative"]
    wrong = record["arms"]["same_unit_wrong_key"]
    assert correct["failure"] is negative["failure"] is wrong["failure"] is None
    assert correct["detection"]["status"] == "RELIABLE"
    assert correct["detection"]["support"] == len(correct["detection"]["diagnostics"]["matches"])
    assert correct["evaluation"]["corner_error_max_diagonal"] < 0.03
    assert negative["detection"]["status"] == "UNRELIABLE"
    assert wrong["detection"]["status"] == "UNRELIABLE"
    assert not {"key_digest", "derived_key", "root_key", "pattern"} & _record_keys(record)
    assert KEY not in repr(record) and WRONG_KEY not in repr(record)


@pytest.mark.integration
def test_rotation_exercises_log_polar_rs_and_local_similarity_fit() -> None:
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    record = run_unit(rgb, KEY, WRONG_KEY, split="P1D", seed=4102, attack="rotation_+5")
    correct = record["arms"]["marked_correct_key"]
    assert correct["detection"]["diagnostics"]["coarse_log_polar"]["PSR"] > 0.0
    assert correct["detection"]["status"] == "RELIABLE"
    assert correct["evaluation"]["corner_error_max_diagonal"] < 0.02
    assert record["arms"]["attacked_unwatermarked_negative"]["detection"]["status"] == "UNRELIABLE"
    assert record["arms"]["same_unit_wrong_key"]["detection"]["status"] == "UNRELIABLE"


@pytest.mark.integration
def test_runner_rejects_cross_split_seed_without_execution() -> None:
    rgb = np.full((64, 64, 3), 0.5, dtype=np.float64)
    with pytest.raises(ValueError, match="outside the selected split"):
        run_unit(rgb, KEY, WRONG_KEY, split="P1D", seed=4201, attack="identity")
