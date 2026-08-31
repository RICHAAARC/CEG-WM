"""Fail-closed baseline-only main-table rows.

This builder aggregates only post-calibration evaluation records under the
frozen protocol. It never admits a proposed-method row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cegwm.baselines.records import BaselineObservation, validate_observation
from cegwm.baselines.registry import baseline_by_id


PRIMARY_ATTACK_FAMILIES = (
    "clean",
    "compression",
    "geometric",
    "photometric",
    "noise",
    "diffusion",
)


@dataclass(frozen=True)
class BaselineTableRow:
    baseline_id: str
    source_exact: str
    adapter_exact: str
    attack_family: str
    attack_condition: str
    threshold_identity: str
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    failure_count: int
    quality_metric_names: tuple[str, ...]
    runtime_observation_count: int
    status: str = "engineering_only"


def build_baseline_table_row(records: Iterable[BaselineObservation]) -> BaselineTableRow:
    """Build one baseline evaluation row without admitting wrong-key diagnostics."""

    items = tuple(validate_observation(record) for record in records)
    if not items:
        raise ValueError("a baseline table row requires records")
    baseline_id = items[0].baseline_id
    baseline_by_id(baseline_id)
    if any(item.baseline_id != baseline_id for item in items):
        raise ValueError("a table row cannot mix baselines")
    if any(item.sample_role == "wrong_key_diagnostic" for item in items):
        raise ValueError("wrong-key diagnostics are not main-table inputs")
    if any(item.status == "calibration_observed" for item in items):
        raise ValueError("calibration records are not main-table inputs")
    if any(item.status == "confirmation_observed" for item in items):
        raise ValueError("confirmation records are not main-table inputs")
    if any(item.status == "not_available" for item in items):
        raise ValueError("not-available records are not main-table inputs")
    if any(item.sample_role not in {"evaluation_watermarked", "evaluation_unwatermarked_negative"}
           for item in items):
        raise ValueError("main-table inputs require evaluation roles")
    if any(item.protocol_partition != "evaluation" for item in items):
        raise ValueError("main-table inputs require evaluation partition")
    evaluation = tuple(item for item in items if item.status == "observed")
    if not evaluation:
        raise ValueError("a baseline table row requires observed evaluation records")
    identity = (evaluation[0].source_exact, evaluation[0].adapter_exact, evaluation[0].threshold_provenance,
                evaluation[0].attack_family, evaluation[0].attack_condition)
    if identity[3] not in PRIMARY_ATTACK_FAMILIES:
        raise ValueError("attack family is not registered")
    if any((item.source_exact, item.adapter_exact, item.threshold_provenance,
            item.attack_family, item.attack_condition) != identity for item in evaluation):
        raise ValueError("evaluation records must share source, adapter, threshold, and attack identity")
    if any((item.attack_family, item.attack_condition) != identity[3:] for item in items):
        raise ValueError("all table inputs must share attack identity")
    if any(item.status == "failed" and (
        item.source_exact not in {None, identity[0]} or item.adapter_exact not in {None, identity[1]}
    ) for item in items):
        raise ValueError("failed records must match the row source and adapter when present")
    true_positive = sum(item.sample_role == "evaluation_watermarked" and item.decision for item in evaluation)
    false_negative = sum(item.sample_role == "evaluation_watermarked" and not item.decision for item in evaluation)
    false_positive = sum(item.sample_role == "evaluation_unwatermarked_negative" and item.decision for item in evaluation)
    true_negative = sum(item.sample_role == "evaluation_unwatermarked_negative" and not item.decision for item in evaluation)
    failures = sum(item.status == "failed" for item in items)
    quality_names = tuple(sorted({name for item in evaluation for name in item.quality}))
    return BaselineTableRow(
        baseline_id=baseline_id,
        source_exact=identity[0],
        adapter_exact=identity[1],
        attack_family=identity[3],
        attack_condition=identity[4],
        threshold_identity=identity[2],
        true_positive=true_positive,
        false_negative=false_negative,
        false_positive=false_positive,
        true_negative=true_negative,
        failure_count=failures,
        quality_metric_names=quality_names,
        runtime_observation_count=sum(item.runtime_seconds is not None for item in evaluation),
    )
