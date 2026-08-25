from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments import run_content_v8_formal_initial as runner
from cegwm.method.content_iss_v8 import ISSDevelopmentMeasurement
from cegwm.protocol.content_chain_v8 import (
    ContentV8Protocol,
    ContentV8Roster,
    ContentV8Unit,
)


def _units(prefix: str, split: str, count: int) -> tuple[ContentV8Unit, ...]:
    return tuple(
        ContentV8Unit(
            f"{prefix}-{index:04d}", split, f"{prefix}-source-{index:04d}",
            f"prompt {prefix} {index}", 1000 + index, 512, 512,
        )
        for index in range(1, count + 1)
    )


def _protocol() -> ContentV8Protocol:
    return ContentV8Protocol(
        "protocol-v8",
        "scope-v8",
        {"limitations": ("engineering-only",)},
        _units("dev", "content_v6_iss_development_v1", 32),
        (
            ContentV8Roster(
                "content_v2_reference", "v2.jsonl", "d" * 64,
                _units("v2", "content_adaptive_dual_branch_v2_clean_v1", 8),
            ),
            ContentV8Roster(
                "content_v6_current", "v6.jsonl", "2" * 64,
                _units("v6", "content_v6_iss_clean_v1", 8),
            ),
        ),
        "a" * 64,
    )


def _args(tmp_path: Path, exact: str) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(tmp_path / "repo"),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )


@pytest.mark.integration
def test_one_invocation_fits_publishes_then_runs_two_independent_rosters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exact = "1" * 40
    protocol = _protocol()
    order: list[str] = []
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(runner, "load_content_v8_protocol", lambda root: protocol)
    monkeypatch.setattr(
        runner, "_load_pipeline_and_assets",
        lambda token: order.append("load") or (object(), object()),
    )
    calls = 0

    def development(pipeline, unit, key, assets):
        nonlocal calls
        calls += 1
        return ISSDevelopmentMeasurement(
            -0.2 + calls / 1000,
            0.1 + calls / 1000,
            0.2 + calls / 1000,
        )

    monkeypatch.setattr(runner, "run_content_v8_development_pair", development)

    def evaluate(**kwargs):
        role = kwargs["roster"].role
        order.append(role)
        asset_path = (
            Path(_args(tmp_path, exact).artifact_sink)
            / runner._run_id(exact, protocol.protocol_digest)
            / "runtime_asset"
            / runner.ASSET_FILENAME
        )
        assert asset_path.exists()
        return {
            "roster_role": role,
            "manifest": kwargs["roster"].manifest,
            "manifest_sha256": kwargs["roster"].manifest_sha256,
            "rc": 0,
            "fixed_denominator_units": 8,
            "fixed_record_count": 16,
            "records": [],
            "failed_units": [],
            "gate_evidence": {},
            "scientific_status": "not_adjudicated",
        }

    monkeypatch.setattr(runner, "_evaluate_roster", evaluate)
    monkeypatch.setenv(runner.KEY_ENV, "content-v8-root-key")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    assert runner.execute(_args(tmp_path, exact)) == 0
    assert calls == 32
    assert order == ["load", "content_v2_reference", "content_v6_current"]
    assert runner.KEY_ENV not in runner.os.environ
    assert runner.TOKEN_ENV not in runner.os.environ

    run_id = runner._run_id(exact, protocol.protocol_digest)
    run_root, asset_path, sidecar_path, archive_path = runner._paths(
        tmp_path / "sink", run_id
    )
    assert asset_path.exists() and sidecar_path.exists()
    assert sidecar_path.read_text(encoding="ascii").endswith(
        f"  {runner.ASSET_FILENAME}\n"
    )
    terminal_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert archive_path.with_name(f"{archive_path.name}.sha256").read_text(
        encoding="ascii"
    ) == f"{terminal_digest}  {archive_path.name}\n"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "receipt.json",
            "result.json",
            f"runtime_asset/{runner.ASSET_FILENAME}",
            f"runtime_asset/{runner.ASSET_FILENAME}.sha256",
        ]
        result = json.loads(archive.read("result.json"))
    assert [item["roster_role"] for item in result["evaluation_results_in_order"]] == [
        "content_v2_reference", "content_v6_current",
    ]
    assert result["cross_roster_pooling"] is False
    assert result["cross_roster_outcome_control"] is False
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.integration
def test_fit_failure_and_interruption_stop_before_asset_and_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "2" * 40
    protocol = _protocol()
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(runner, "load_content_v8_protocol", lambda root: protocol)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda token: (object(), object()))
    monkeypatch.setattr(
        runner, "_evaluate_roster", lambda **kwargs: pytest.fail("evaluation must not run")
    )
    monkeypatch.setattr(
        runner, "run_content_v8_development_pair",
        lambda *args: ISSDevelopmentMeasurement(0.1, 0.1, 0.1),
    )
    monkeypatch.setenv(runner.KEY_ENV, "content-v8-root-key")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    with pytest.raises(ValueError, match="positive"):
        runner.execute(_args(tmp_path, exact))
    assert not (tmp_path / "sink").exists()

    monkeypatch.setattr(
        runner, "run_content_v8_development_pair",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setenv(runner.KEY_ENV, "content-v8-root-key")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path, exact))
    assert not (tmp_path / "sink").exists()


@pytest.mark.integration
def test_create_only_preflight_precedes_secret_and_model_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "3" * 40
    protocol = _protocol()
    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(runner, "load_content_v8_protocol", lambda root: protocol)
    run_root, *_ = runner._paths(
        tmp_path / "sink", runner._run_id(exact, protocol.protocol_digest)
    )
    run_root.mkdir(parents=True)
    monkeypatch.setattr(
        runner, "_load_pipeline_and_assets", lambda token: pytest.fail("must not load")
    )
    monkeypatch.setenv(runner.KEY_ENV, "still-present")
    monkeypatch.setenv(runner.TOKEN_ENV, "still-present")
    with pytest.raises(FileExistsError, match="create-only"):
        runner.execute(_args(tmp_path, exact))
    assert runner.os.environ[runner.KEY_ENV] == "still-present"
    assert runner.os.environ[runner.TOKEN_ENV] == "still-present"


@pytest.mark.integration
def test_roster_failures_denominators_and_strict_tie_gates_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    identity = {
        "run_id": "run",
        "exact": "1" * 40,
        "protocol_digest": "a" * 64,
        "key_digest": "b" * 64,
    }

    def transaction(*, unit, identity, roster_role, **kwargs):
        registered = 0.5
        scores = {
            f"{branch}__{label}": (
                registered
                if label == "registered"
                else (registered if unit.unit_id.endswith("0001") else 0.1)
            )
            for branch in runner.BRANCHES
            for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))
        }
        null_scores = dict(scores)
        for branch in runner.BRANCHES:
            null_scores[f"{branch}__registered"] = 0.2
        metrics = {
            "combined_relative_l2": 0.012,
            "lf_effective_relative_l2": 0.006,
            "hf_effective_relative_l2": 0.006,
            "probe_evaluation_count": 64.0,
            "paired_rgb_psnr_db": 31.0,
        }
        return [
            runner._record(
                run_id="run", roster_role=roster_role, unit=unit,
                arm=runner.ARMS[0], exact=identity["exact"],
                protocol_digest=identity["protocol_digest"],
                key_digest=identity["key_digest"], status="success",
                scores=scores, metrics=metrics,
            ),
            runner._record(
                run_id="run", roster_role=roster_role, unit=unit,
                arm=runner.ARMS[1], exact=identity["exact"],
                protocol_digest=identity["protocol_digest"],
                key_digest=identity["key_digest"], status="success",
                scores=null_scores, metrics={"paired_rgb_psnr_db": 31.0},
            ),
        ]

    monkeypatch.setattr(runner, "_unit_transaction", transaction)
    first = runner._evaluate_roster(
        roster=protocol.evaluation_rosters[0], pipeline=object(), assets=object(),
        iss_asset=object(), key=b"k" * 32, wrong_keys=(b"w" * 32,) * 16,
        identity=identity,
    )
    assert first["rc"] == 0
    assert first["fixed_denominator_units"] == 8
    assert first["fixed_record_count"] == 16
    assert first["gate_evidence"]["branches"]["lf"]["registered_top_rank_pass_units"] == 7
    assert first["gate_evidence"]["branches"]["lf"]["registered_top_rank_gate_pass"] is True

    def fail_one(**kwargs):
        if kwargs["unit"].unit_id.endswith("0001"):
            raise RuntimeError("unit failure")
        return transaction(**kwargs)

    monkeypatch.setattr(runner, "_unit_transaction", fail_one)
    second = runner._evaluate_roster(
        roster=protocol.evaluation_rosters[1], pipeline=object(), assets=object(),
        iss_asset=object(), key=b"k" * 32, wrong_keys=(b"w" * 32,) * 16,
        identity=identity,
    )
    assert second["rc"] == 2
    assert len(second["records"]) == 16
    assert second["failed_units"] == [
        {"unit_id": "v6-0001", "error_type": "RuntimeError"}
    ]
    assert second["gate_evidence"] is None
    assert first["rc"] == 0 and first["gate_evidence"] is not None
