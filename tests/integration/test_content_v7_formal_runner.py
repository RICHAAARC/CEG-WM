from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import pytest

from experiments import run_content_v7_formal_initial as runner
from cegwm.method.content_iss_v7 import ISSDevelopmentMeasurement
from cegwm.protocol.content_chain_v7 import load_content_v7_formal_protocol
from cegwm.shared.keys import public_key_digest

_ROOT = Path(__file__).resolve().parents[2]


def _result(invocation_id: str, *, rc: int = 0) -> dict[str, object]:
    return {
        "invocation_id": invocation_id,
        "rc": rc,
        "gate_evidence": {"all_predeclared_gates_pass": rc == 0},
    }


@pytest.mark.integration
def test_two_invocations_attempt_independent_fixed_8_by_2_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = load_content_v7_formal_protocol(_ROOT)
    calls: list[tuple[int, str]] = []

    def fail_unit(**kwargs: object) -> list[dict[str, object]]:
        unit = kwargs["unit"]
        calls.append((len(calls) // 8 + 1, unit.unit_id))
        raise RuntimeError("synthetic operational failure")

    monkeypatch.setattr(runner, "_unit_transaction", fail_unit)
    key = b"content-v7-evaluation-key"
    key_digest = public_key_digest(key)
    results = [
        runner._evaluate_invocation(
            invocation_index=index,
            protocol=protocol,
            pipeline=object(),
            assets=runner.ContentV7RunnerAssets(object()),
            key=key,
            exact="1" * 40,
            key_digest=key_digest,
        )
        for index, protocol in enumerate(formal.evaluations, 1)
    ]
    assert len(calls) == 16
    assert [group for group, _ in calls] == [1] * 8 + [2] * 8
    assert [result["fixed_denominator_units"] for result in results] == [8, 8]
    assert [len(result["records"]) for result in results] == [16, 16]
    assert [len(result["failed_units"]) for result in results] == [8, 8]
    assert [result["rc"] for result in results] == [2, 2]


@pytest.mark.integration
def test_integrated_runner_fits_publishes_then_calls_two_rosters_and_one_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "2" * 40
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(runner.engine, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner,
        "_load_pipeline_and_assets",
        lambda model_id, token: (object(), type("Assets", (), {"lf_public_assets": object()})()),
    )
    monkeypatch.setattr(runner, "ContentV7DevelopmentAssets", lambda *args: object())
    monkeypatch.setattr(runner, "ContentV7EvaluationAssets", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "run_content_v7_development_pair",
        lambda pipeline, unit, key, assets: calls.append(("fit", unit.unit_id))
        or ISSDevelopmentMeasurement(-0.2, 0.1, 0.2),
    )

    def evaluate(**kwargs: object) -> dict[str, object]:
        protocol = kwargs["protocol"]
        index = kwargs["invocation_index"]
        asset_path, sidecar, _, _ = runner._paths(tmp_path / "sink", exact)
        assert asset_path.exists() and sidecar.exists()
        calls.append(("evaluation", tuple(unit.unit_id for unit in protocol.roster)))
        return _result(
            protocol.protocol_id.rsplit("/", 1)[-1] + f"-{index}",
            rc=0 if index == 1 else 2,
        )

    monkeypatch.setattr(runner, "_evaluate_invocation", evaluate)
    monkeypatch.setenv(runner.KEY_ENV, "content-v7-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    args = argparse.Namespace(
        repo_root=str(_ROOT),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )
    assert runner.execute(args) == 0
    assert [item[0] for item in calls].count("fit") == 32
    evaluations = [item[1] for item in calls if item[0] == "evaluation"]
    assert len(evaluations) == 2
    assert evaluations[0][0] == "content-adaptive-v2-0001"
    assert evaluations[1][0] == "content-v6-iss-eval-0001"
    assert runner.KEY_ENV not in os.environ and runner.TOKEN_ENV not in os.environ
    asset_path, sidecar, terminal, terminal_sidecar = runner._paths(
        tmp_path / "sink", exact
    )
    assert all(path.exists() for path in (asset_path, sidecar, terminal, terminal_sidecar))
    with zipfile.ZipFile(terminal) as archive:
        assert archive.namelist() == [
            "receipt.json", asset_path.name, sidecar.name, "result.json"
        ]
        result = json.loads(archive.read("result.json"))
    assert result["evaluation_result_count"] == 2
    assert len(result["evaluations"]) == 2
    assert [item["rc"] for item in result["evaluations"]] == [0, 2]
    assert result["pooling_applied"] is False
    assert result["cross_cohort_conjunction_applied"] is False
    assert result["combined_result_produced"] is False
    assert all(
        field not in result
        for field in (
            "rc",
            "scientific_status",
            "scientific_outcome_allowed",
            "joint_min_all_predeclared_gates_pass",
        )
    )
    with pytest.raises(FileExistsError, match="create-only"):
        runner.execute(args)


@pytest.mark.integration
def test_fit_failure_prevents_asset_evaluation_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "3" * 40
    monkeypatch.setattr(runner.engine, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner,
        "_load_pipeline_and_assets",
        lambda model_id, token: (object(), type("Assets", (), {"lf_public_assets": object()})()),
    )
    monkeypatch.setattr(runner, "ContentV7DevelopmentAssets", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "run_content_v7_development_pair",
        lambda *args: ISSDevelopmentMeasurement(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_invocation",
        lambda **kwargs: pytest.fail("evaluation must not start after fit failure"),
    )
    monkeypatch.setenv(runner.KEY_ENV, "content-v7-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    args = argparse.Namespace(
        repo_root=str(_ROOT),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )
    with pytest.raises(ValueError, match="gain must be finite and positive"):
        runner.execute(args)
    assert not any(path.exists() for path in runner._paths(tmp_path / "sink", exact))
