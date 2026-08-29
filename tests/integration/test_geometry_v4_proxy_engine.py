from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments import geometry_v4_proxy_engine as engine
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
def test_source_identity_and_full_plan_are_deterministic_without_running_full() -> None:
    first_image, first_source = engine.generate_procedural_source(4101)
    repeat_image, repeat_source = engine.generate_procedural_source(4101)
    other_image, other_source = engine.generate_procedural_source(4102)
    assert np.array_equal(first_image, repeat_image)
    assert first_source == repeat_source
    assert first_source["image_identity_sha256"] != other_source["image_identity_sha256"]
    assert not np.array_equal(first_image, other_image)

    plan = engine.plan_full("P1D")
    assert len(plan) == 8 * 16 == 128
    assert {item["seed"] for item in plan} == set(range(4101, 4109))
    assert {item["attack"] for item in plan} == set(P1_ATTACKS)
    assert all(item["execution_mode"] == "P1D_full" for item in plan)
    assert all(item["formal_denominator_member"] is True for item in plan)
    assert all(item["source"]["image_identity_sha256"] for item in plan)
    assert set(inspect.signature(engine.run_full).parameters) == {"detection_key", "wrong_key", "split"}


@pytest.mark.integration
def test_small_canary_is_nonformal_and_keeps_real_three_arm_outputs() -> None:
    records = engine.run_canary(
        KEY,
        WRONG_KEY,
        subset=((4101, "identity"), (4102, "translation_+0.10_0")),
    )
    assert len(records) == 2
    assert all(record["execution_mode"] == "engineering_canary" for record in records)
    assert all(record["formal_denominator_member"] is False for record in records)
    assert all(record["formal_split"] is None for record in records)
    assert all("full" not in record["unit_id"] and "P1D" not in record["unit_id"] for record in records)
    for record in records:
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
        assert correct["detection"]["H_hat"] is not None
        assert correct["detection"]["support"] == len(correct["detection"]["diagnostics"]["matches"])
        assert correct["evaluation"]["corner_error_max_diagonal"] < 0.03
        assert negative["detection"]["status"] == "UNRELIABLE"
        assert wrong["detection"]["status"] == "UNRELIABLE"
        assert record["source"]["image_identity_sha256"]
        assert not {"key_digest", "derived_key", "root_key", "pattern"} & _record_keys(record)
        assert KEY not in repr(record) and WRONG_KEY not in repr(record)


@pytest.mark.integration
def test_failures_keep_three_stopped_arms_and_equal_normalized_wrong_key_is_rejected() -> None:
    stopped = engine._stopped_arms("source: deterministic failure")
    assert set(stopped) == {
        "marked_correct_key",
        "attacked_unwatermarked_negative",
        "same_unit_wrong_key",
    }
    assert all(arm["failure"] for arm in stopped.values())
    assert all(arm["detection"]["status"] == "STOPPED" for arm in stopped.values())
    detector_failure = engine._detect_arm(np.zeros((4, 4, 3), dtype=np.float64), KEY)
    assert detector_failure["failure"] is not None
    assert detector_failure["detection"]["status"] == "STOPPED"

    normalized_first = "e\u0301" * 16
    normalized_second = "é" * 16
    with pytest.raises(ValueError, match="must differ after normalization"):
        engine.run_canary(
            normalized_first,
            normalized_second,
            subset=((4101, "identity"),),
        )


@pytest.mark.integration
def test_canary_cannot_impersonate_full_or_cross_split_seed() -> None:
    full_subset = tuple((seed, attack) for seed in range(4101, 4109) for attack in P1_ATTACKS)
    with pytest.raises(ValueError, match="unique strict subset"):
        engine.run_canary(KEY, WRONG_KEY, subset=full_subset)
    with pytest.raises(ValueError, match="outside every frozen seed roster"):
        engine.run_canary(
            KEY,
            WRONG_KEY,
            subset=((9999, "identity"),),
        )
