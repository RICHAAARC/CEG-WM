from __future__ import annotations

# Functional coverage for content ISS fitting.

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import fit_content_iss as runner
from cegwm.method.content_iss import ISSDevelopmentMeasurement, load_iss_asset
from cegwm.protocol.content_iss import ContentISSUnit, ISS_DEVELOPMENT_SPLIT


def _units() -> tuple[ContentISSUnit, ...]:
    return tuple(
        ContentISSUnit(
            f"content-v6-iss-dev-{index+1:04d}", ISS_DEVELOPMENT_SPLIT,
            f"content-v6-iss-dev-source-{index+1:04d}", f"prompt {index}",
            2026082400 + index, 512, 512,
        )
        for index in range(32)
    )


@pytest.mark.integration
def test_fit_runner_calls_32_pairs_clears_secrets_and_publishes_create_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exact = "2" * 40
    calls: list[ContentISSUnit] = []
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner,
        "load_content_iss_data_contract",
        lambda root: SimpleNamespace(development=_units()),
    )
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda token: (object(), object()))
    monkeypatch.setattr(
        runner,
        "run_content_iss_development_pair",
        lambda pipeline, unit, key, assets: calls.append(unit) or ISSDevelopmentMeasurement(
            -0.2 + len(calls)/1000,
            0.1 + len(calls)/1000,
            0.2 + len(calls)/1000,
        ),
    )
    monkeypatch.setenv(runner.KEY_ENV, "content-v6-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    args = argparse.Namespace(
        repo_root=str(tmp_path / "repo"),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )
    assert runner.execute(args) == 0
    assert len(calls) == 32
    assert runner.KEY_ENV not in runner.os.environ and runner.TOKEN_ENV not in runner.os.environ
    asset_path, sidecar_path = runner._destinations(Path(args.artifact_sink), exact)
    asset = load_iss_asset(asset_path, sidecar_path)
    assert asset.payload["producer_exact"] == exact
    assert asset.payload["fit_sample_count"] == 32
    stdout = capsys.readouterr().out.splitlines()
    assert len(stdout) == 1 and stdout[0].startswith(runner.FIT_RECEIPT_PREFIX + " ")
    receipt = json.loads(stdout[0].split(" ", 1)[1])
    assert receipt["asset_sha256"] == hashlib.sha256(asset.json_bytes).hexdigest()
    with pytest.raises(FileExistsError, match="destination exists"):
        runner.execute(args)


@pytest.mark.integration
def test_fit_runner_checks_destination_before_secret_or_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exact = "3" * 40
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(runner, "load_content_iss_data_contract", lambda root: SimpleNamespace(development=_units()))
    asset_path, _ = runner._destinations(tmp_path / "sink", exact)
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"existing")
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda token: pytest.fail("must not load"))
    args = argparse.Namespace(repo_root=str(tmp_path), artifact_sink=str(tmp_path / "sink"), expected_exact=exact)
    with pytest.raises(FileExistsError):
        runner.execute(args)
