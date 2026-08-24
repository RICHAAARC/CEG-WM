from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import pytest

from cegwm.protocol.content_chain_v2 import ContentChainProtocol
from cegwm.protocol.content_chain_v5 import CONTENT_V5_PROTOCOL_DIGEST
from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v3_clean as v3_runner
from experiments import run_content_v4_clean as v4_runner
from experiments import run_content_v5_clean as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "runner-key-value-01"
_TOKEN = "test-token-value"
_PUBLIC_KEY_DIGEST = (
    "805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77"
)
_RUN_ID = "content-v5-c5a0c4bf7d6d-805bc21e173a"


def _variant(cohort_id: str) -> engine.ContentRunnerVariant:
    return runner.CONTENT_V5_RUNNER_VARIANTS[cohort_id]


def _paired_and_identity():
    paired = runner._load_protocol(_ROOT)
    identity = runner._umbrella_identity(
        paired, exact=_EXACT, key_digest=_PUBLIC_KEY_DIGEST
    )
    return paired, identity


def _branch_scores(registered: float, wrong: float) -> dict[str, float]:
    return {
        "registered": registered,
        **{f"wrong_{index:02d}": wrong for index in range(16)},
    }


def _records(
    arms: tuple[str, str],
    outcomes: list[tuple[bool, bool]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (lf_pass, hf_pass) in enumerate(outcomes):
        lf = _branch_scores(0.8 if lf_pass else 0.2, 0.2)
        hf = _branch_scores(0.8 if hf_pass else 0.2, 0.2)
        joint = _branch_scores(min(lf["registered"], hf["registered"]), -0.5)
        null_lf = _branch_scores(0.2, 0.0)
        null_hf = _branch_scores(0.2, 0.0)
        null_joint = _branch_scores(0.2, 0.0)
        records.extend((
            {
                "unit_id": f"unit-{index}",
                "arm": arms[0],
                "scores": engine._flat_scores({"lf": lf, "hf": hf, "joint": joint}),
            },
            {
                "unit_id": f"unit-{index}",
                "arm": arms[1],
                "scores": engine._flat_scores(
                    {"lf": null_lf, "hf": null_hf, "joint": null_joint}
                ),
            },
        ))
    return records


def _unit_metrics() -> list[dict[str, float | str]]:
    return [
        {
            "unit_id": f"unit-{index}",
            "combined_relative_l2": 0.0119,
            "lf_effective_relative_l2": 0.005,
            "hf_effective_relative_l2": 0.007,
            "lf_branch_share": 0.4,
            "hf_branch_share": 0.6,
            **{name: 0.0 for name in engine.COUNTERFACTUAL_EFFECT_FIELDS},
            "minimum_counterfactual_effect": 0.0,
            "probe_evaluation_count": 64.0,
            "paired_rgb_psnr_db": 31.0,
        }
        for index in range(8)
    ]


def _result_records(
    protocol: ContentChainProtocol,
    variant: engine.ContentRunnerVariant,
    outcomes: list[tuple[bool, bool]],
) -> list[dict[str, Any]]:
    scores = _records(variant.arms, outcomes)
    records: list[dict[str, Any]] = []
    for index, unit in enumerate(protocol.roster):
        lf_share = 0.32 + index * 0.02
        metrics = {
            "combined_relative_l2": 0.0119,
            "lf_effective_relative_l2": 0.005,
            "hf_effective_relative_l2": 0.007,
            "lf_branch_share": lf_share,
            "hf_branch_share": 1.0 - lf_share,
            **{name: 0.0 for name in engine.COUNTERFACTUAL_EFFECT_FIELDS},
            "minimum_counterfactual_effect": 0.0,
            "probe_evaluation_count": 64.0,
            "paired_rgb_psnr_db": 31.0,
        }
        records.extend((
            {
                "unit_id": unit.unit_id,
                "arm": variant.arms[0],
                "status": "success",
                "scores": scores[index * 2]["scores"],
                "metrics": metrics,
            },
            {
                "unit_id": unit.unit_id,
                "arm": variant.arms[1],
                "status": "success",
                "scores": scores[index * 2 + 1]["scores"],
                "metrics": {"paired_rgb_psnr_db": 31.0},
            },
        ))
    return records


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(_ROOT),
        expected_exact=_EXACT,
        local_work_root=str(tmp_path / "local"),
        artifact_sink=str(tmp_path / "sink"),
    )


def _set_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(engine.KEY_ENV, _KEY)
    monkeypatch.setenv(engine.TOKEN_ENV, _TOKEN)


def _terminal(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paired = runner._load_protocol(_ROOT)
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    run_id = runner._umbrella_identity(
        paired, exact=_EXACT, key_digest=key_digest
    )["run_id"]
    archive_path = tmp_path / "sink" / run_id / f"{run_id}.zip"
    payload = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["receipt.json", "result.json", "audit-state.json"]
        receipt = json.loads(archive.read("receipt.json"))
        result = json.loads(archive.read("result.json"))
        state = json.loads(archive.read("audit-state.json"))
    checksum = archive_path.with_name(f"{archive_path.name}.sha256").read_text(
        encoding="ascii"
    )
    assert checksum.split() == [hashlib.sha256(payload).hexdigest(), archive_path.name]
    return receipt, result, state


@pytest.mark.integration
def test_content_v5_aggregate_branchwise_or_has_exact_seven_of_eight_boundary() -> None:
    variant = _variant("primary_1")
    seven = _records(variant.arms, [(True, False)] * 7 + [(False, False)])
    evidence = engine._gate_evidence(seven, _unit_metrics(), variant=variant)
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 7
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 7
    assert evidence["all_decision_gates_pass"] is True
    assert evidence["all_predeclared_gates_pass"] is True
    assert evidence["formal_fpr_claim"] is False
    six = _records(variant.arms, [(True, False)] * 6 + [(False, False)] * 2)
    failed = engine._gate_evidence(six, _unit_metrics(), variant=variant)
    assert failed["branchwise_or"]["gate_a_pass_units"] == 6
    assert failed["branchwise_or"]["gate_b_pass_units"] == 6
    assert failed["all_decision_gates_pass"] is False
    assert failed["all_predeclared_gates_pass"] is False


@pytest.mark.integration
def test_content_v5_individual_branch_counts_are_not_conjunctions() -> None:
    variant = _variant("primary_1")
    records = _records(variant.arms, [(True, False)] * 4 + [(False, True)] * 4)
    evidence = engine._gate_evidence(records, _unit_metrics(), variant=variant)
    assert evidence["branches"]["lf"]["gate_a_pass_units"] == 4
    assert evidence["branches"]["hf"]["gate_a_pass_units"] == 4
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 8
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 8
    assert evidence["all_predeclared_gates_pass"] is True


@pytest.mark.integration
def test_v2_v3_v4_keep_legacy_branch_conjunction_evaluator_and_shape() -> None:
    for variant in (
        engine.V2_RUNNER_VARIANT,
        v3_runner.CONTENT_V3_RUNNER_VARIANT,
        v4_runner.CONTENT_V4_RUNNER_VARIANT,
    ):
        assert variant.decision_evaluator is None
        records = _records(variant.arms, [(True, True)] * 7 + [(False, True)])
        evidence = engine._gate_evidence(records, _unit_metrics(), variant=variant)
        assert tuple(evidence) == (
            "branches", "combined_budget_pass_units", "both_nonzero_branches_pass_units",
            "baseline_differenced_probe_response_pass_units",
            "probe_evaluation_count_64_pass_units",
            "public_branch_share_valid_pass_units", "paired_rgb_psnr_pass_units",
            "all_predeclared_gates_pass", "formal_fpr_claim",
        )
        assert evidence["branches"]["lf"]["gate_a_pass_units"] == 7
        assert evidence["branches"]["hf"]["gate_a_pass_units"] == 8
        assert evidence["branches"]["joint"]["gate_a_pass_units"] == 8
        assert evidence["all_predeclared_gates_pass"] is True


@pytest.mark.integration
def test_content_v5_variants_retain_exact_real_v4_runtime_and_scorer_wiring() -> None:
    for variant in runner.CONTENT_V5_RUNNER_VARIANTS.values():
        assert variant.load_pipeline_and_assets is v4_runner._load_pipeline_and_assets
        assert variant.run_joint is v4_runner._run_joint
        assert variant.lf_scorer is v4_runner.score_content_v4_lf_image
        assert variant.state_schema_id == (
            "content_v5_umbrella_whole_unit_checkpoint_state_v1"
        )
        assert variant.record_contract_id == (
            "content_v5_whitened_lf_adaptive_hf_branchwise_or_record_v1"
        )
        assert variant.decision_evaluator is not None


@pytest.mark.integration
def test_content_v5_has_one_umbrella_run_and_state_identity_for_both_cohorts() -> None:
    paired, identity = _paired_and_identity()
    assert paired.protocol_digest == CONTENT_V5_PROTOCOL_DIGEST
    assert identity["run_id"] == _RUN_ID
    assert [item["cohort_id"] for item in identity["ordered_cohorts"]] == [
        "control_1", "primary_1"
    ]
    assert identity["ordered_cohorts"][0]["ordered_roster"] != (
        identity["ordered_cohorts"][1]["ordered_roster"]
    )
    state = runner._new_state(identity)
    assert runner._validate_state(state, identity, paired) is state
    changed = dict(identity)
    changed["ordered_cohorts"] = list(reversed(identity["ordered_cohorts"]))
    with pytest.raises(ValueError, match="identity differs"):
        runner._validate_state(state, changed, paired)


@pytest.mark.integration
def test_content_v5_cohort_results_are_independent_and_never_transfer_pass() -> None:
    paired, identity = _paired_and_identity()
    primary_protocol = paired.cohort_protocol("primary_1")
    control_protocol = paired.cohort_protocol("control_1")
    primary_variant = _variant("primary_1")
    control_variant = _variant("control_1")
    primary_records = _result_records(
        primary_protocol, primary_variant, [(True, False)] * 8
    )
    control_records = _result_records(
        control_protocol, control_variant, [(True, False)] * 6 + [(False, False)] * 2
    )
    primary = engine._derive_result(
        primary_records, primary_protocol, identity, variant=primary_variant
    )
    control = engine._derive_result(
        control_records, control_protocol, identity, variant=control_variant
    )
    assert primary["fixed_denominator_units"] == control["fixed_denominator_units"] == 8
    assert len(primary["records"]) == len(control["records"]) == 16
    assert primary["gate_evidence"]["all_predeclared_gates_pass"] is True
    assert control["gate_evidence"]["all_predeclared_gates_pass"] is False
    assert primary["execution_scope_id"] != control["execution_scope_id"]
    assert "pooled" not in primary and "pooled" not in control


@pytest.mark.integration
def test_content_v5_one_invocation_runs_control_then_primary_despite_unit_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(engine, "_git_exact", lambda root, exact: exact)
    monkeypatch.setattr(v4_runner, "_load_pipeline_and_assets", lambda model, token: (object(), object()))

    def fail_unit(**kwargs: Any) -> list[dict[str, Any]]:
        unit_id = kwargs["unit"].unit_id
        calls.append(unit_id)
        raise RuntimeError("fixed denominator failure")

    monkeypatch.setattr(engine, "_unit_transaction", fail_unit)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 2
    paired = runner._load_protocol(_ROOT)
    assert calls == [
        *(unit.unit_id for unit in paired.cohorts["control_1"]),
        *(unit.unit_id for unit in paired.cohorts["primary_1"]),
    ]
    receipt, result, state = _terminal(tmp_path)
    assert receipt["run_id"] == result["run_id"] == state["identity"]["run_id"]
    assert receipt["cohorts_in_order"] == ["control_1", "primary_1"]
    assert state["committed_whole_unit_count"] == 16
    assert [item["cohort_id"] for item in result["cohort_results"]] == [
        "control_1", "primary_1"
    ]
    assert all(item["result"]["fixed_denominator_units"] == 8 for item in result["cohort_results"])
    assert all(len(item["result"]["records"]) == 16 for item in result["cohort_results"])
    assert result["completeness"] == runner.UMBRELLA_INCOMPLETE_EXECUTION
    assert result["both_cohorts_attempted"] is True
    assert result["pooled_decision_absent"] is True
    assert result["cross_cohort_conjunction"] is False
    assert result["reference_result_controls_primary_execution"] is False
    assert result["umbrella_rc_operational_only"] is True
    assert result["scientific_decision_scope"] == "independent_cohort_results_only"
    assert "scientific_outcome_allowed" not in result


@pytest.mark.integration
def test_content_v5_fatal_loader_failure_reports_both_independent_cohorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_git_exact", lambda root, exact: exact)

    def fail_loader(model: str, token: str) -> tuple[object, object]:
        raise RuntimeError("loader failure")

    monkeypatch.setattr(v4_runner, "_load_pipeline_and_assets", fail_loader)
    _set_secrets(monkeypatch)
    assert runner.execute(_args(tmp_path)) == 2
    _, result, state = _terminal(tmp_path)
    assert state["committed_whole_unit_count"] == 0
    assert [item["cohort_id"] for item in result["cohort_results"]] == [
        "control_1", "primary_1"
    ]
    assert all(item["result"]["fixed_denominator_units"] == 8 for item in result["cohort_results"])
    assert all(item["result"]["records"] == [] for item in result["cohort_results"])
    assert all(item["result"]["gate_evidence"] is None for item in result["cohort_results"])
    assert result["both_cohorts_attempted"] is False
    assert result["operational_error_class"] == "RuntimeError"


@pytest.mark.integration
def test_content_v5_process_interrupt_leaves_last_whole_unit_and_rerun_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    model_calls = 0
    monkeypatch.setattr(engine, "_git_exact", lambda root, exact: exact)

    def load(model: str, token: str) -> tuple[object, object]:
        nonlocal model_calls
        model_calls += 1
        return object(), object()

    def interrupt(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        raise RuntimeError("first unit recorded")

    monkeypatch.setattr(v4_runner, "_load_pipeline_and_assets", load)
    monkeypatch.setattr(engine, "_unit_transaction", interrupt)
    _set_secrets(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(_args(tmp_path))
    paired = runner._load_protocol(_ROOT)
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    identity = runner._umbrella_identity(paired, exact=_EXACT, key_digest=key_digest)
    state_path = tmp_path / "local" / identity["run_id"] / "audit-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["committed_whole_unit_count"] == 1
    assert state["cohorts"][0]["committed_unit_count"] == 1
    assert len(state["cohorts"][0]["records"]) == 2
    assert not (tmp_path / "sink" / identity["run_id"]).exists()
    _set_secrets(monkeypatch)
    with pytest.raises(FileExistsError, match="resume and retry are forbidden"):
        runner.execute(_args(tmp_path))
    assert model_calls == 1


@pytest.mark.integration
def test_content_v5_existing_sink_root_rejects_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = runner._load_protocol(_ROOT)
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    identity = runner._umbrella_identity(paired, exact=_EXACT, key_digest=key_digest)
    (tmp_path / "sink" / identity["run_id"]).mkdir(parents=True)
    model_calls: list[str] = []
    monkeypatch.setattr(engine, "_git_exact", lambda root, exact: exact)
    monkeypatch.setattr(
        v4_runner,
        "_load_pipeline_and_assets",
        lambda *args: model_calls.append("called"),
    )
    _set_secrets(monkeypatch)
    with pytest.raises(FileExistsError, match="resume and retry are forbidden"):
        runner.execute(_args(tmp_path))
    assert model_calls == []


@pytest.mark.integration
@pytest.mark.parametrize("failure", ("absent", "mismatched"))
def test_content_v5_manifest_failure_stops_before_model_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config_root = tmp_path / "configs" / "content_chain"
    config_root.mkdir(parents=True)
    for name in (
        "content_v5_lf_or_hf_clean_v1.json",
        "content_v5_primary_evaluation_v1.jsonl",
        "content_adaptive_dual_branch_v2_clean.jsonl",
    ):
        shutil.copyfile(_ROOT / "configs" / "content_chain" / name, config_root / name)
    primary = config_root / "content_v5_primary_evaluation_v1.jsonl"
    if failure == "absent":
        primary.unlink()
    else:
        primary.write_bytes(primary.read_bytes().replace(b"beekeeper", b"beekeeqer", 1))
    model_calls: list[str] = []
    monkeypatch.setattr(
        v4_runner,
        "_load_pipeline_and_assets",
        lambda *args: model_calls.append("called"),
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        runner._load_protocol(tmp_path)
    assert model_calls == []


@pytest.mark.integration
def test_content_v5_runner_has_no_cohort_selector_resume_or_historical_import_path() -> None:
    source = inspect.getsource(runner)
    arguments = inspect.getsource(runner._arguments)
    assert "--cohort" not in arguments
    assert "--resume" not in arguments
    assert "--retry" not in arguments
    assert "_resolve_state" not in source
    assert "checkpoint-" not in source
    assert "read_checkpoint" not in source
    assert "load_sink_checkpoint" not in source
    assert os.path.basename(runner.__file__) == "run_content_v5_clean.py"
