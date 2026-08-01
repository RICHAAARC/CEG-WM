"""Smoke regression for the hf_only_reference_validation-only experiment package entrypoint."""

from __future__ import annotations

import pytest

from scripts.experiment_execution import experiment_execution_entrypoint


@pytest.mark.smoke
def test_package_entrypoint_exposes_only_verified_threshold_fit() -> None:
    assert (
        experiment_execution_entrypoint.THRESHOLD_FIT_EXECUTION_SCOPE
        == "hf_only_threshold_fit_only"
    )
    assert callable(
        experiment_execution_entrypoint.execute_verified_threshold_fit_shard
    )
    assert not any(
        "synthetic" in name
        for name in vars(experiment_execution_entrypoint)
        if not name.startswith("__")
    )
