from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_content_v9_calibration as runner
from cegwm.method.content_weighted_joint_v9 import LFHFScorePair, load_calibration_asset
from cegwm.protocol.content_chain_v9 import CONTENT_V9_CALIBRATION_SPLIT, ContentV9Unit


def _units() -> tuple[ContentV9Unit, ...]:
    return tuple(
        ContentV9Unit(
            f"content-v9-calibration-{index + 1:04d}", CONTENT_V9_CALIBRATION_SPLIT,
            f"content-v9-calibration-source-{index + 1:04d}", f"prompt {index}",
            2026091000 + index, 512, 512,
        )
        for index in range(32)
    )


@pytest.mark.integration
def test_runner_calls_exactly_32_real_pairs_clears_secrets_and_publishes_create_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exact = "4" * 40
    protocol_digest = "5" * 64
    calls: list[ContentV9Unit] = []
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner, "load_content_v9_phase1_contract",
        lambda root: SimpleNamespace(calibration=_units(), protocol_digest=protocol_digest),
    )
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda token: (object(), object()))

    def fake_unit(pipeline: object, unit: ContentV9Unit, key: bytes, assets: object):
        calls.append(unit)
        offset = len(calls) * 33
        return tuple(
            LFHFScorePair(
                -0.8 + (offset + index) / 2200,
                -0.6 + ((offset + index) * 37 % 997) / 1300,
            )
            for index in range(33)
        )

    monkeypatch.setattr(runner, "run_content_v9_calibration_unit", fake_unit)
    monkeypatch.setenv(runner.KEY_ENV, "content-v9-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    args = argparse.Namespace(
        repo_root=str(tmp_path / "repo"),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )
    assert runner.execute(args) == 0
    assert len(calls) == 32
    assert runner.KEY_ENV not in runner.os.environ and runner.TOKEN_ENV not in runner.os.environ
    public_digest = runner.public_key_digest(
        runner.derive_calibration_key("content-v9-root-key-material")
    )
    run_id = runner.deterministic_calibration_run_id(protocol_digest, public_digest)
    asset_path, sidecar_path = runner._destinations(Path(args.artifact_sink), run_id)
    asset = load_calibration_asset(asset_path, sidecar_path)
    assert asset.payload["producer_exact"] == exact
    assert asset.payload["calibration_pair_count"] == 1056
    stdout = capsys.readouterr().out.splitlines()
    assert len(stdout) == 1 and stdout[0].startswith(runner.RECEIPT_PREFIX + " ")
    receipt = json.loads(stdout[0].split(" ", 1)[1])
    assert receipt["asset_sha256"] == hashlib.sha256(asset.json_bytes).hexdigest()
    assert receipt["calibration_pair_count"] == 1056
    monkeypatch.setenv(runner.KEY_ENV, "content-v9-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    with pytest.raises(FileExistsError, match="destination exists"):
        runner.execute(args)


@pytest.mark.integration
def test_runner_fails_closed_on_nonfinite_or_incomplete_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises((TypeError, ValueError)):
        runner.fit_weighted_joint_calibration(
            [LFHFScorePair(math.nan, 0.0)] * 1056
        )
    with pytest.raises(ValueError, match="exactly 1056"):
        runner.fit_weighted_joint_calibration([LFHFScorePair(0.0, 0.1)] * 1055)


@pytest.mark.integration
def test_create_only_publish_removes_both_partial_files_on_sidecar_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    asset_path = tmp_path / runner.ASSET_FILENAME
    sidecar_path = tmp_path / f"{runner.ASSET_FILENAME}.sha256"
    original_open = Path.open

    class _FailingSidecar:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> "_FailingSidecar":
            self._handle.__enter__()
            return self

        def write(self, payload: bytes) -> None:
            self._handle.write(payload[:7])
            raise OSError("simulated sidecar write failure")

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)

    def patched_open(path: Path, *args: object, **kwargs: object) -> object:
        handle = original_open(path, *args, **kwargs)
        if path == sidecar_path and args and args[0] == "xb":
            return _FailingSidecar(handle)
        return handle

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="sidecar write failure"):
        runner._publish_create_only(asset_path, sidecar_path, b"asset")
    assert not asset_path.exists()
    assert not sidecar_path.exists()
