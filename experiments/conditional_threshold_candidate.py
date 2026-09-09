"""Development-only per-condition empirical threshold; no population FPR claim."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def conditional_threshold_candidate(
    complete_path_scores: Mapping[str, Sequence[float]], *, target_fpr: float = 0.001
) -> dict:
    """Take the maximum empirical (1-alpha) quantile across negative conditions.

    Inputs must score the whole frozen pre/post/candidate-selection path, not
    individual content candidates. Strict score > tau is the decision rule.
    This is a candidate for subsequent independent testing, never a guarantee.
    Missing/failed scores must be resolved or reported separately by the caller;
    silently dropping them here would alter the stated sample distribution.
    """
    if not math.isfinite(target_fpr) or not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0,1)")
    if not complete_path_scores:
        raise ValueError("at least one negative condition is required")
    summaries = {}
    for condition, values in complete_path_scores.items():
        if not isinstance(condition, str) or not condition:
            raise ValueError("condition names must be nonempty strings")
        scores = [float(value) for value in values]
        if not scores or not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{condition}: nonempty finite complete-path scores required")
        scores.sort()
        rank = max(1, math.ceil((1.0 - target_fpr) * len(scores)))
        summaries[condition] = {
            "n": len(scores), "rank_one_based": rank,
            "empirical_quantile": scores[rank - 1],
        }
    tau = max(row["empirical_quantile"] for row in summaries.values())
    for condition, values in complete_path_scores.items():
        fp = sum(float(value) > tau for value in values)
        summaries[condition].update(empirical_fp=fp,
                                     empirical_fpr=fp / summaries[condition]["n"])
    return {
        "tau_candidate": tau, "target_fpr": target_fpr,
        "decision_rule": "complete_path_score_strictly_greater_than_tau",
        "target_kind": "per_condition_empirical_quantile_maximum",
        "conditions": summaries,
        "claim": "empirical_fit_only_requires_independent_testing",
        "population_fpr_guarantee": False,
    }
