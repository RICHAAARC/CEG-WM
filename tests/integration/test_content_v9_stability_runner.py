from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v9_stability as runner
from cegwm.protocol.content_chain_v9_stability import (
    CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST,
)

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_RUN_ID = "content-v9-stability-9bc8a94c1d02-63c17e8200a9-805bc21e173a"


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(_ROOT),
        expected_exact=_EXACT,
        local_work_root=str(tmp_path / "local"),
        artifact_sink=str(tmp_path / "sink"),
    )


def _secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runner.KEY_ENV, "test-root-key-material")
    monkeypatch.setenv(runner.TOKEN_ENV, "test-token")
    monkeypatch.setattr(runner, "public_key_digest", lambda key: CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST)


def _scores(*, candidate: bool) -> dict[str, dict[str, float]]:
    labels = runner.SCORE_LABELS
    lf = {label: (0.1 if label != "registered" else (-0.2 if candidate else 0.0)) for label in labels}
    hf = {label: (0.1 if label != "registered" else (0.8 if candidate else 0.0)) for label in labels}
    weighted = {
        label: (1.0 if label != "registered" else (2.0 if candidate else 0.0))
        for label in labels
    }
    return {"lf": lf, "hf": hf, "weighted_joint": weighted}


def _transaction(unit: object, identity: dict[str, object]) -> list[dict[str, object]]:
    effects = {name: 0.001 for name in engine.COUNTERFACTUAL_EFFECT_FIELDS}
    candidate_metrics = {
        "combined_relative_l2": 0.0119,
        "lf_effective_relative_l2": 0.005,
        "hf_effective_relative_l2": 0.007,
        "lf_branch_share": 0.4,
        "hf_branch_share": 0.6,
        **effects,
        "minimum_counterfactual_effect": 0.001,
        "probe_evaluation_count": 64.0,
        "paired_rgb_psnr_db": 31.0,
    }
    return [
        runner._record(
            identity=identity, unit=unit, arm_index=0, status="success",
            scores=runner._flat_scores(_scores(candidate=True)), metrics=candidate_metrics,
        ),
        runner._record(
            identity=identity, unit=unit, arm_index=1, status="success",
            scores=runner._flat_scores(_scores(candidate=False)),
            metrics={"paired_rgb_psnr_db": 31.0},
        ),
    ]


def _terminal(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    archive_path = tmp_path / "sink" / _RUN_ID / f"{_RUN_ID}.zip"
    payload = archive_path.read_bytes()
    sidecar = archive_path.with_name(f"{archive_path.name}.sha256").read_text(encoding="ascii")
    assert sidecar.split() == [hashlib.sha256(payload).hexdigest(), archive_path.name]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["receipt.json", "result.json"]
        return json.loads(archive.read("receipt.json")), json.loads(archive.read("result.json"))


@pytest.mark.integration
def test_one_invocation_runs_all_80_in_order_with_one_checkpoint_and_independent_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runner.load_content_v9_stability_contract(_ROOT)
    expected_units = runner._ordered_units(contract)
    calls: list[str] = []
    monkeypatch.setattr(engine, "_git_exact", lambda root, expected: expected)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model, token: (object(), object()))
    monkeypatch.setattr(runner, "derive_stability_wrong_keys", lambda key: tuple(bytes([i]) for i in range(16)))

    def transaction(**kwargs: object) -> list[dict[str, object]]:
        unit = kwargs["unit"]
        calls.append(unit.unit_id)
        return _transaction(unit, kwargs["identity"])

    monkeypatch.setattr(runner, "_unit_transaction", transaction)
    times = iter([0.0, 7201.0, *([7201.0] * 79)])
    monkeypatch.setattr(engine, "_now", lambda: next(times))
    _secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 0
    assert calls == [unit.unit_id for unit in expected_units]
    assert runner.KEY_ENV not in runner.os.environ and runner.TOKEN_ENV not in runner.os.environ

    checkpoint_path = tmp_path / "sink" / _RUN_ID / f"{_RUN_ID}.checkpoint-0000.zip"
    with zipfile.ZipFile(checkpoint_path) as archive:
        assert archive.namelist() == ["state.json"]
        checkpoint = json.loads(archive.read("state.json"))
    assert checkpoint["checkpoint_sequence"] == 1
    assert checkpoint["committed_unit_count"] == 1
    assert len(checkpoint["records"]) == 2

    receipt, result = _terminal(tmp_path)
    assert receipt["committed_unit_count"] == 80
    assert receipt["calibration_asset_sha256"] == (
        "63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96"
    )
    assert result["sections_in_order"] == list(runner.SECTION_IDS)
    assert [item["fixed_denominator_units"] for item in result["section_results"]] == [
        8, 8, 32, 32,
    ]
    assert [len(item["records"]) for item in result["section_results"]] == [16, 16, 64, 64]
    assert all(item["gate_evidence"]["all_section_weighted_gates_pass"] for item in result["section_results"])
    assert all(
        item["gate_evidence"]["lf_hf_diagnostics_only_no_hard_veto"]["lf"][
            "gate_a_pass_units"
        ] == 0
        for item in result["section_results"]
    )
    assert len(result["novel_two_seed_prompt_descriptives"]) == 32
    assert result["pooled_denominator_absent"] is True
    assert result["cross_section_conjunction_absent"] is True
    assert result["combined_result_absent"] is True
    assert "all_predeclared_gates_pass" not in result

    model_calls = len(calls)
    monkeypatch.setattr(engine, "_now", lambda: 7201.0)
    _secrets(monkeypatch)
    with pytest.raises(FileExistsError, match="terminal artifact pair already exists"):
        runner.execute(_args(tmp_path))
    assert len(calls) == model_calls


@pytest.mark.integration
def test_unit_failure_is_retained_and_does_not_control_later_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runner.load_content_v9_stability_contract(_ROOT)
    expected_units = runner._ordered_units(contract)
    calls: list[str] = []
    monkeypatch.setattr(engine, "_git_exact", lambda root, expected: expected)
    monkeypatch.setattr(engine, "_now", lambda: 0.0)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model, token: (object(), object()))
    monkeypatch.setattr(runner, "derive_stability_wrong_keys", lambda key: tuple(bytes([i]) for i in range(16)))

    def transaction(**kwargs: object) -> list[dict[str, object]]:
        unit = kwargs["unit"]
        calls.append(unit.unit_id)
        if len(calls) == 1:
            raise RuntimeError("fixed denominator unit failure")
        return _transaction(unit, kwargs["identity"])

    monkeypatch.setattr(runner, "_unit_transaction", transaction)
    _secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 2
    assert calls == [unit.unit_id for unit in expected_units]
    _, result = _terminal(tmp_path)
    sections = result["section_results"]
    assert sections[0]["rc"] == 2
    assert sections[0]["fixed_denominator_units"] == 8
    assert len(sections[0]["records"]) == 16
    assert sections[0]["failed_units"] == [{
        "unit_id": expected_units[0].unit_id,
        "status": "failed",
        "error_type": "RuntimeError",
    }]
    assert [item["rc"] for item in sections[1:]] == [0, 0, 0]
    assert result["committed_unit_count"] == 80
    assert result["section_outcome_controls_later_execution"] is False
    assert result["operational_error_class"] is None


@pytest.mark.integration
def test_state_rejects_cross_section_reorder_and_non_whole_unit_prefix() -> None:
    contract = runner.load_content_v9_stability_contract(_ROOT)
    identity = runner._identity(contract, exact=_EXACT, key_digest=CONTENT_V9_STABILITY_PUBLIC_KEY_DIGEST)
    state = runner._new_state(identity, 0.0)
    state["committed_unit_count"] = 1
    state["records"] = _transaction(runner._ordered_units(contract)[0], identity)
    assert runner._validate_state(state, identity, contract) is state
    state["records"] = state["records"][:1]
    with pytest.raises(ValueError, match="whole-unit prefix"):
        runner._validate_state(state, identity, contract)
