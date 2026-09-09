import pytest

from experiments.conditional_threshold_candidate import conditional_threshold_candidate

pytestmark = pytest.mark.unit


def test_condition_max_is_not_pooled_quantile_and_strict_ties():
    result = conditional_threshold_candidate(
        {"clean": [0.0] * 100, "attack": [1.0, 2.0, 2.0, 3.0]}, target_fpr=0.25)
    assert result["tau_candidate"] == 2.0
    assert result["conditions"]["attack"]["empirical_fp"] == 1
    assert result["conditions"]["clean"]["empirical_fp"] == 0
    assert result["population_fpr_guarantee"] is False


def test_small_sample_does_not_manufacture_point_one_percent_evidence():
    result = conditional_threshold_candidate({"clean": list(range(24))})
    assert result["tau_candidate"] == 23.0
    assert result["conditions"]["clean"]["rank_one_based"] == 24
    assert result["claim"] == "empirical_fit_only_requires_independent_testing"


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_invalid_scores_are_not_dropped(values):
    with pytest.raises(ValueError):
        conditional_threshold_candidate({"clean": values})
