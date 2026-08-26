from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import pytest

from experiments import run_content_v6_formal_initial as runner
from cegwm.method.content_iss_v6 import ISSDevelopmentMeasurement
from cegwm.protocol.content_chain_v6_formal import load_content_v6_formal_protocol
from cegwm.shared.keys import public_key_digest

_ROOT = Path(__file__).resolve().parents[2]


def _result(unit_set: dict[str, object], *, rc: int) -> dict[str, object]:
    return {
        "unit_set": unit_set,
        "rc": rc,
        "gate_evidence": {"all_predeclared_gates_pass": rc == 0},
    }


@pytest.mark.integration
def test_protocol_exposes_32v1_then_independent_8v1_and_8v3() -> None:
    formal = load_content_v6_formal_protocol(_ROOT)
    assert len(formal.development) == 32
    assert [len(protocol.roster) for protocol in formal.evaluations] == [8, 8]
    assert formal.config["execution_flow"]["phase_order"] == [
        "development_32V1", "evaluation_8V1", "evaluation_8V3", "terminal"
    ]
    assert [item["display_label"] for item in formal.config["unit_sets"]] == [
        "[32V1]", "[8V1]", "[8V3]"
    ]
    assert formal.evaluations[0].roster[0].unit_id == "content-adaptive-v2-0001"
    assert formal.evaluations[1].roster[0].unit_id == "content-v6-iss-eval-0001"


@pytest.mark.integration
def test_two_evaluations_retain_independent_fixed_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = load_content_v6_formal_protocol(_ROOT)
    calls: list[str] = []

    def fail_unit(**kwargs: object) -> list[dict[str, object]]:
        unit = kwargs["unit"]
        calls.append(unit.unit_id)
        raise RuntimeError("synthetic operational failure")

    monkeypatch.setattr(runner, "_unit_transaction", fail_unit)
    key = b"content-v6-formal-evaluation-key"
    key_digest = public_key_digest(key)
    results = [
        runner._evaluate_invocation(
            invocation_index=index,
            unit_set=unit_set,
            protocol=protocol,
            pipeline=object(),
            assets=runner.ContentV6FormalRunnerAssets(object()),
            key=key,
            exact="1" * 40,
            key_digest=key_digest,
        )
        for index, (unit_set, protocol) in enumerate(
            zip(
                (runner.UNIT_SET_8V1, runner.UNIT_SET_8V3),
                formal.evaluations,
                strict=True,
            ),
            1,
        )
    ]
    assert len(calls) == 16
    assert [result["fixed_denominator_units"] for result in results] == [8, 8]
    assert [len(result["records"]) for result in results] == [16, 16]
    assert [len(result["failed_units"]) for result in results] == [8, 8]
    assert [result["rc"] for result in results] == [2, 2]


@pytest.mark.integration
def test_integrated_runner_fits_then_evaluates_two_rosters_and_publishes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "2" * 40
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(runner.engine, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner,
        "_load_pipeline_and_assets",
        lambda model_id, token: (
            object(),
            type("Assets", (), {"embed_assets": object(), "lf_public_assets": object()})(),
        ),
    )
    monkeypatch.setattr(runner, "ContentV6DevelopmentAssets", lambda *args: object())
    monkeypatch.setattr(runner, "ContentV6EvaluationAssets", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "run_content_v6_development_pair",
        lambda pipeline, unit, key, assets: calls.append(("fit", unit.unit_id))
        or ISSDevelopmentMeasurement(-0.2, 0.1, 0.2),
    )

    def evaluate(**kwargs: object) -> dict[str, object]:
        protocol = kwargs["protocol"]
        unit_set = kwargs["unit_set"]
        asset_path, sidecar, _, _ = runner._paths(tmp_path / "sink", exact)
        assert asset_path.exists() and sidecar.exists()
        calls.append(("evaluation", tuple(unit.unit_id for unit in protocol.roster)))
        return _result(dict(unit_set), rc=0 if unit_set["display_label"] == "[8V1]" else 2)

    monkeypatch.setattr(runner, "_evaluate_invocation", evaluate)
    monkeypatch.setenv(runner.KEY_ENV, "content-v6-root-key-material")
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
    asset_path, sidecar, terminal, terminal_sidecar = runner._paths(tmp_path / "sink", exact)
    assert all(path.exists() for path in (asset_path, sidecar, terminal, terminal_sidecar))
    with zipfile.ZipFile(terminal) as archive:
        assert archive.namelist() == [
            "receipt.json", asset_path.name, sidecar.name, "result.json"
        ]
        receipt = json.loads(archive.read("receipt.json"))
        result = json.loads(archive.read("result.json"))
    assert receipt["unit_set_labels_in_order"] == ["[32V1]", "[8V1]", "[8V3]"]
    assert [item["unit_set"]["display_label"] for item in result["evaluations"]] == [
        "[8V1]", "[8V3]"
    ]
    assert result["pooling_applied"] is False
    assert result["cross_cohort_conjunction_applied"] is False
    assert result["combined_result_produced"] is False
    with pytest.raises(FileExistsError, match="create-only"):
        runner.execute(args)


@pytest.mark.integration
def test_fit_failure_prevents_evaluation_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exact = "3" * 40
    monkeypatch.setattr(runner.engine, "_git_exact", lambda root, expected: exact)
    monkeypatch.setattr(
        runner,
        "_load_pipeline_and_assets",
        lambda model_id, token: (
            object(),
            type("Assets", (), {"embed_assets": object(), "lf_public_assets": object()})(),
        ),
    )
    monkeypatch.setattr(runner, "ContentV6DevelopmentAssets", lambda *args: object())
    monkeypatch.setattr(
        runner,
        "run_content_v6_development_pair",
        lambda *args: ISSDevelopmentMeasurement(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_invocation",
        lambda **kwargs: pytest.fail("evaluation must not start after fit failure"),
    )
    monkeypatch.setenv(runner.KEY_ENV, "content-v6-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "private-token")
    args = argparse.Namespace(
        repo_root=str(_ROOT),
        artifact_sink=str(tmp_path / "sink"),
        expected_exact=exact,
    )
    with pytest.raises(ValueError, match="gain must be finite and positive"):
        runner.execute(args)
    assert not any(path.exists() for path in runner._paths(tmp_path / "sink", exact))
