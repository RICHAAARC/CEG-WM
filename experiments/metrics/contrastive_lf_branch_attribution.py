"""Frozen Stage-A gate aggregation over complete fixed-denominator cases."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from experiments.metrics.binomial import clopper_pearson_lower
from experiments.protocol.contrastive_lf_branch_attribution import (
    ATTACKS,
    CANDIDATE_IDS,
    CLUSTER_COUNT,
    blur_complement_passes,
    condition_false_positive_gate_passes,
    identity_attribution_gate_passes,
    identity_attribution_passes,
    quality_gate_passes,
)


class ContrastiveLfMetricError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StageABranchCase:
    source_cluster_id: str
    attack_id: str
    registered_z: float
    primary_null_z: float
    external_wrong_z: tuple[float, ...]
    hf_registered_z: float
    hf_primary_null_z: float
    hf_external_wrong_z: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StageAQualityCase:
    source_cluster_id: str
    attack_id: str
    candidate_mse: float
    hf_only_mse: float


@dataclass(frozen=True, slots=True)
class StageACandidateGateResult:
    candidate_id: str
    identity_attribution_passed: bool
    condition_false_positive_passed: bool
    blur_complement_success_count: int
    blur_complement_lower: float
    blur_complement_passed: bool
    quality_passed: bool
    candidate_passed: bool


def _validate_cases(cases: Sequence[StageABranchCase]) -> None:
    if len(cases) != CLUSTER_COUNT * len(ATTACKS):
        raise ContrastiveLfMetricError("branch case denominator is incomplete")
    expected = {(cluster, attack) for cluster in {case.source_cluster_id for case in cases} for attack in ATTACKS}
    observed = {(case.source_cluster_id, case.attack_id) for case in cases}
    if len({case.source_cluster_id for case in cases}) != CLUSTER_COUNT or observed != expected:
        raise ContrastiveLfMetricError("branch case identity matrix drifted")
    for case in cases:
        values = (
            case.registered_z,
            case.primary_null_z,
            case.hf_registered_z,
            case.hf_primary_null_z,
            *case.external_wrong_z,
            *case.hf_external_wrong_z,
        )
        if len(case.external_wrong_z) != 8 or len(case.hf_external_wrong_z) != 8 or any(not isfinite(value) for value in values):
            raise ContrastiveLfMetricError("branch case values are invalid")


def evaluate_stage_a_candidate_gates(
    candidate_id: str,
    branch_cases: Sequence[StageABranchCase],
    quality_cases: Sequence[StageAQualityCase],
    *,
    candidate_tau: float,
    hf_tau: float,
) -> StageACandidateGateResult:
    """Evaluate attribution/null/wrong, blur complement, and paired RGB8 quality."""

    if candidate_id not in CANDIDATE_IDS or not isfinite(candidate_tau) or not isfinite(hf_tau):
        raise ContrastiveLfMetricError("candidate gate identity is invalid")
    _validate_cases(branch_cases)
    if len(quality_cases) != CLUSTER_COUNT * len(ATTACKS) or any(
        not isfinite(case.candidate_mse)
        or not isfinite(case.hf_only_mse)
        or case.candidate_mse < 0.0
        or case.hf_only_mse < 0.0
        for case in quality_cases
    ):
        raise ContrastiveLfMetricError("quality case denominator is incomplete")
    identity_cases = [case for case in branch_cases if case.attack_id == "identity"]
    identity_pass = identity_attribution_gate_passes(
        tuple(
            identity_attribution_passes(
                registered_z=case.registered_z,
                primary_null_z=case.primary_null_z,
                external_wrong_z=case.external_wrong_z,
            )
            for case in identity_cases
        )
    )
    condition_pass = True
    for attack in ATTACKS:
        condition = [case for case in branch_cases if case.attack_id == attack]
        primary_count = sum(case.primary_null_z >= candidate_tau for case in condition)
        wrong_counts = tuple(
            sum(case.external_wrong_z[index] >= candidate_tau for case in condition)
            for index in range(8)
        )
        condition_pass = condition_pass and condition_false_positive_gate_passes(
            primary_null_positive_count=primary_count,
            external_wrong_positive_counts=wrong_counts,
        )
    blur = [case for case in branch_cases if case.attack_id == "gaussian_blur_sigma_1"]
    blur_count = sum(
        case.registered_z - max(case.external_wrong_z)
        > case.hf_registered_z - max(case.hf_external_wrong_z)
        for case in blur
    )
    lower = clopper_pearson_lower(blur_count, 32, confidence_level=0.95)
    blur_pass = blur_complement_passes(
        success_count=blur_count, clopper_pearson_lower_value=lower
    )
    quality_pass = all(
        quality_gate_passes(
            tuple(case.candidate_mse for case in quality_cases if case.attack_id == attack),
            tuple(case.hf_only_mse for case in quality_cases if case.attack_id == attack),
        )
        for attack in ATTACKS
    )
    return StageACandidateGateResult(
        candidate_id=candidate_id,
        identity_attribution_passed=identity_pass,
        condition_false_positive_passed=condition_pass,
        blur_complement_success_count=blur_count,
        blur_complement_lower=lower,
        blur_complement_passed=blur_pass,
        quality_passed=quality_pass,
        candidate_passed=identity_pass and condition_pass and blur_pass and quality_pass,
    )


def evaluate_stage_a_hf_anchor(
    cases: Sequence[StageABranchCase], *, hf_tau: float
) -> bool:
    _validate_cases(cases)
    identity = [case for case in cases if case.attack_id == "identity"]
    attribution = identity_attribution_gate_passes(
        tuple(
            identity_attribution_passes(
                registered_z=case.hf_registered_z,
                primary_null_z=case.hf_primary_null_z,
                external_wrong_z=case.hf_external_wrong_z,
            )
            for case in identity
        )
    )
    conditions = True
    for attack in ATTACKS:
        condition = [case for case in cases if case.attack_id == attack]
        conditions = conditions and condition_false_positive_gate_passes(
            primary_null_positive_count=sum(case.hf_primary_null_z >= hf_tau for case in condition),
            external_wrong_positive_counts=tuple(
                sum(case.hf_external_wrong_z[index] >= hf_tau for case in condition)
                for index in range(8)
            ),
        )
    return attribution and conditions


__all__ = [
    "ContrastiveLfMetricError",
    "StageABranchCase",
    "StageACandidateGateResult",
    "StageAQualityCase",
    "evaluate_stage_a_candidate_gates",
    "evaluate_stage_a_hf_anchor",
]
