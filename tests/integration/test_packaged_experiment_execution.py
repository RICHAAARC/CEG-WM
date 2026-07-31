"""Integration regression for the retired synthetic package surface."""

from __future__ import annotations

import pytest

from scripts.experiment_execution import experiment_execution_entrypoint


@pytest.mark.integration
def test_retired_synthetic_package_api_is_unreachable() -> None:
    assert callable(
        experiment_execution_entrypoint.execute_verified_threshold_fit_shard
    )
    assert not hasattr(
        experiment_execution_entrypoint,
        "prepare_synthetic_wiring",
    )
    assert not hasattr(
        experiment_execution_entrypoint,
        "run_synthetic_wiring",
    )
