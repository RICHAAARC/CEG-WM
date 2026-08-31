"""Non-executing adapter plans for the four Baseline-V1 methods.

Plans name the source-qualified path only.  They deliberately do not import
external code, load a model, or choose SD3.5 conversion parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from cegwm.baselines.registry import baseline_by_id


@dataclass(frozen=True)
class AdapterPlan:
    baseline_id: str
    adapter_kind: str
    source_entrypoint: str
    execution_status: str
    blocker: str | None


def adapter_plan(baseline_id: str) -> AdapterPlan:
    """Return a source-qualified plan without granting execution capability."""

    baseline = baseline_by_id(baseline_id)
    if baseline.official_entrypoint is None:
        raise ValueError("baseline has no audited official entrypoint")
    if baseline_id == "t2smark":
        return AdapterPlan(
            baseline_id, "official_sd35_native", baseline.official_entrypoint,
            "execution_not_authorized", "model and runtime execution are not authorized",
        )
    if baseline.source_status != "qualified":
        return AdapterPlan(
            baseline_id, "method_faithful_sd35_adaptation", baseline.official_entrypoint,
            "blocked", "official source has no license file at the frozen exact",
        )
    return AdapterPlan(
        baseline_id, "method_faithful_sd35_adaptation", baseline.official_entrypoint,
        "semantic_review_required", "SD3.5 adaptation parameters are not yet authorized",
    )
