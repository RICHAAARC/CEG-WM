"""Package-contained identity check using no repository or Git state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_execution.experiment_execution_entrypoint import (
    prepare_synthetic_wiring,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_packaged_preparation_matches_bound_manifest(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "experiment_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    preparation = prepare_synthetic_wiring(
        package_root=ROOT,
        records_root=tmp_path / "records",
        workspace_root=tmp_path / "workspace",
        committed_revision=manifest["committed_revision"],
        run_id="packaged-integration",
    )

    assert preparation.candidate_config_digest == (
        manifest["candidate_config_digest"]
    )
    assert preparation.execution_config_digest == (
        manifest["execution_config_digest"]
    )
    assert preparation.input_manifest_digest == (
        manifest["input_manifest_digest"]
    )
