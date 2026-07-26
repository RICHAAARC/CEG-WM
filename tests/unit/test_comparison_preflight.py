"""验证外部 baseline 公平对比的运行前协议。"""

from __future__ import annotations

import pytest

from experiments.protocol.comparison import (
    ComparisonMethodSpec,
    ComparisonProtocol,
    approve_comparison_protocol,
    validate_comparison_protocol,
)


def _protocol(**changes: object) -> ComparisonProtocol:
    values: dict[str, object] = {
        "comparison_group_name": "primary_comparison",
        "sample_manifest_digest": "a" * 64,
        "split_manifest_digest": "b" * 64,
        "generation_conditions_digest": "c" * 64,
        "seed_policy_digest": "d" * 64,
        "output_specification_digest": "e" * 64,
        "attack_matrix_digest": "f" * 64,
        "metric_set_digest": "1" * 64,
        "calibration_split": "calibration",
        "evaluation_split": "evaluation",
        "tuning_budget_policy_digest": "2" * 64,
        "compute_budget_policy_digest": "3" * 64,
        "failure_policy_digest": "4" * 64,
        "exclusion_policy_digest": "5" * 64,
        "methods": (
            ComparisonMethodSpec("project_method", "project_method", "commit_a", "6" * 64, ""),
            ComparisonMethodSpec("reference_method", "external_baseline", "commit_b", "7" * 64, ""),
        ),
    }
    values.update(changes)
    return ComparisonProtocol(**values)


@pytest.mark.unit
def test_valid_comparison_protocol_receives_stable_approval() -> None:
    protocol = _protocol()
    approval = approve_comparison_protocol(protocol)
    assert approval.protocol_digest == protocol.digest()
    assert approval.sample_manifest_digest == protocol.sample_manifest_digest


@pytest.mark.unit
def test_calibration_and_evaluation_splits_must_differ() -> None:
    violations = validate_comparison_protocol(_protocol(evaluation_split="calibration"))
    assert "calibration_and_evaluation_split_must_differ" in violations


@pytest.mark.unit
def test_external_baseline_is_required() -> None:
    project_method = ComparisonMethodSpec(
        "project_method", "project_method", "commit_a", "6" * 64, ""
    )
    with pytest.raises(ValueError, match="external_baseline_missing"):
        approve_comparison_protocol(_protocol(methods=(project_method,)))
