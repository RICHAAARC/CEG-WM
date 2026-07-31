"""Dependency-free exact one-sided Clopper-Pearson interval primitives."""

from __future__ import annotations

from math import exp, isfinite, lgamma, log, log1p


class BinomialIntervalError(ValueError):
    """Binomial counts or confidence inputs are invalid."""


def _validate_inputs(
    successes: int,
    trials: int,
    confidence_level: float,
) -> float:
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials <= 0
        or successes < 0
        or successes > trials
        or isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise BinomialIntervalError("binomial count or confidence input is invalid")
    return float(confidence_level)


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    log_terms = tuple(
        lgamma(trials + 1)
        - lgamma(index + 1)
        - lgamma(trials - index + 1)
        + index * log(probability)
        + (trials - index) * log1p(-probability)
        for index in range(successes + 1)
    )
    maximum = max(log_terms)
    return exp(maximum) * sum(exp(value - maximum) for value in log_terms)


def clopper_pearson_upper(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Return the exact one-sided upper confidence bound."""

    confidence = _validate_inputs(successes, trials, confidence_level)
    if successes == trials:
        return 1.0
    tail_probability = 1.0 - confidence
    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > tail_probability:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def clopper_pearson_lower(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Return the exact one-sided lower confidence bound."""

    confidence = _validate_inputs(successes, trials, confidence_level)
    if successes == 0:
        return 0.0
    lower = 0.0
    upper = successes / trials
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes - 1, trials, midpoint) > confidence:
            lower = midpoint
        else:
            upper = midpoint
    return upper
