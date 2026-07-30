"""Package-contained A3a execution check using no repository or Git state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_execution.experiment_execution_entrypoint import (
    run_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_packaged_entrypoint_writes_one_synthetic_record(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "experiment_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = run_synthetic_wiring(
        package_root=ROOT,
        output_root=tmp_path / "result",
        workspace_root=tmp_path / "workspace",
        committed_revision=manifest["committed_revision"],
        expected_candidate_config_digest=(
            manifest["candidate_config_digest"]
        ),
        expected_execution_config_digest=(
            manifest["execution_config_digest"]
        ),
        expected_input_manifest_digest=(
            manifest["input_manifest_digest"]
        ),
        run_id="packaged-smoke",
    )

    assert summary["run_status"] == "completed"
    assert summary["record_count"] == 1
    assert summary["success_count"] == 1
    assert summary["scientific_claims_supported"] is False
    assert summary["gpu_executed"] is False
    assert summary["held_out_evaluation_accessed"] is False
