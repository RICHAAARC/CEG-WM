"""Regression checks for removal of the obsolete synthetic package entrypoint."""

from __future__ import annotations

import pytest

from scripts.experiment_execution import experiment_execution_entrypoint
from scripts.experiment_execution.build_experiment_execution_package import (
    EXACT_FILES,
)


@pytest.mark.integration
def test_threshold_fit_delivery_has_no_packaged_synthetic_entrypoint() -> None:
    assert not hasattr(
        experiment_execution_entrypoint,
        "prepare_synthetic_wiring",
    )
    assert not hasattr(
        experiment_execution_entrypoint,
        "run_synthetic_wiring",
    )
    assert "experiments/runners/synthetic_runtime.py" not in EXACT_FILES
    assert (
        "tests/integration/test_packaged_experiment_execution.py"
        not in EXACT_FILES
    )
    assert (
        "tests/smoke/test_packaged_experiment_execution.py"
        not in EXACT_FILES
    )
