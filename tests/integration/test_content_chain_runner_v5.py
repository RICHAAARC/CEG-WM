from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

import pytest

from cegwm.protocol.content_chain_v5 import (
    CONTENT_V5_PROTOCOL_DIGEST,
    ContentChainProtocol,
)
from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v3_clean as v3_runner
from experiments import run_content_v4_clean as v4_runner
from experiments import run_content_v5_clean as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_PUBLIC_KEY_DIGEST = (
    "805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77"
)


def _variant(cohort_id: str) -> engine.ContentRunnerVariant:
    return runner.CONTENT_V5_RUNNER_VARIANTS[cohort_id]


def _protocol_and_identity(cohort_id: str):
    variant = _variant(cohort_id)
    protocol = variant.load_protocol(_ROOT)
    run_id = (
        f"{variant.run_prefix}-{protocol.protocol_digest[:12]}-"
        f"{_PUBLIC_KEY_DIGEST[:12]}"
    )
    identity = engine._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=_PUBLIC_KEY_DIGEST,
        run_id=run_id,
        variant=variant,
    )
    return variant, protocol, identity


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


@pytest.mark.integration
def test_content_v5_aggregate_branchwise_or_has_exact_seven_of_eight_boundary() -> None:
    variant = _variant("primary_1")
    seven = _records(
        variant.arms,
        [(True, False)] * 7 + [(False, False)],
    )
    evidence = engine._gate_evidence(
        seven, _unit_metrics(), variant=variant
    )
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 7
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 7
    assert evidence["all_decision_gates_pass"] is True
    assert evidence["all_predeclared_gates_pass"] is True
    assert evidence["formal_fpr_claim"] is False

    six = _records(
        variant.arms,
        [(True, False)] * 6 + [(False, False)] * 2,
    )
    failed = engine._gate_evidence(
        six, _unit_metrics(), variant=variant
    )
    assert failed["branchwise_or"]["gate_a_pass_units"] == 6
    assert failed["branchwise_or"]["gate_b_pass_units"] == 6
    assert failed["all_decision_gates_pass"] is False
    assert failed["all_predeclared_gates_pass"] is False


@pytest.mark.integration
def test_content_v5_individual_branch_counts_are_not_conjunctions() -> None:
    variant = _variant("primary_1")
    records = _records(
        variant.arms,
        [(True, False)] * 4 + [(False, True)] * 4,
    )
    evidence = engine._gate_evidence(
        records, _unit_metrics(), variant=variant
    )
    assert evidence["branches"]["lf"]["gate_a_pass_units"] == 4
    assert evidence["branches"]["hf"]["gate_a_pass_units"] == 4
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 8
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 8
    assert evidence["all_predeclared_gates_pass"] is True


@pytest.mark.integration
def test_v2_v3_v4_keep_legacy_branch_conjunction_evaluator_and_shape() -> None:
    variants = (
        engine.V2_RUNNER_VARIANT,
        v3_runner.CONTENT_V3_RUNNER_VARIANT,
        v4_runner.CONTENT_V4_RUNNER_VARIANT,
    )
    for variant in variants:
        assert variant.decision_evaluator is None
        records = _records(variant.arms, [(True, True)] * 7 + [(False, True)])
        evidence = engine._gate_evidence(records, _unit_metrics(), variant=variant)
        assert tuple(evidence) == (
            "branches",
            "combined_budget_pass_units",
            "both_nonzero_branches_pass_units",
            "baseline_differenced_probe_response_pass_units",
            "probe_evaluation_count_64_pass_units",
            "public_branch_share_valid_pass_units",
            "paired_rgb_psnr_pass_units",
            "all_predeclared_gates_pass",
            "formal_fpr_claim",
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
        assert variant.state_schema_id == "content_v5_resumable_state_v1"
        assert variant.record_contract_id == (
            "content_v5_whitened_lf_adaptive_hf_branchwise_or_record_v1"
        )
        assert variant.decision_evaluator is not None


@pytest.mark.integration
def test_content_v5_cohort_protocol_run_and_state_identities_are_distinct() -> None:
    primary_variant, primary_protocol, primary_identity = _protocol_and_identity("primary_1")
    control_variant, control_protocol, control_identity = _protocol_and_identity("control_1")
    assert primary_protocol.protocol_digest == control_protocol.protocol_digest
    assert primary_protocol.protocol_digest == CONTENT_V5_PROTOCOL_DIGEST
    assert primary_identity["run_id"] == (
        "content-v5-primary-1-7d8f1ebef662-805bc21e173a"
    )
    assert control_identity["run_id"] == (
        "content-v5-control-1-7d8f1ebef662-805bc21e173a"
    )
    assert primary_identity["execution_scope_id"] != control_identity["execution_scope_id"]
    assert primary_identity["ordered_roster"] != control_identity["ordered_roster"]
    assert primary_identity["exact"] == control_identity["exact"] == _EXACT
    assert primary_identity["public_key_digest"] == control_identity["public_key_digest"]
    primary_state = engine._new_state(primary_identity, 1.0, variant=primary_variant)
    control_state = engine._new_state(control_identity, 1.0, variant=control_variant)
    assert primary_state["identity"] == primary_identity
    assert control_state["identity"] == control_identity
    with pytest.raises(ValueError, match="identity differs"):
        engine._validate_state(
            primary_state,
            control_identity,
            control_protocol,
            variant=control_variant,
        )


@pytest.mark.integration
def test_content_v5_cohort_results_are_independent_and_never_transfer_pass() -> None:
    primary_variant, primary_protocol, primary_identity = _protocol_and_identity("primary_1")
    control_variant, control_protocol, control_identity = _protocol_and_identity("control_1")
    primary_records = _result_records(
        primary_protocol, primary_variant, [(True, False)] * 8
    )
    control_records = _result_records(
        control_protocol, control_variant, [(True, False)] * 6 + [(False, False)] * 2
    )
    primary = engine._derive_result(
        primary_records, primary_protocol, primary_identity, variant=primary_variant
    )
    control = engine._derive_result(
        control_records, control_protocol, control_identity, variant=control_variant
    )
    assert primary["fixed_denominator_units"] == control["fixed_denominator_units"] == 8
    assert len(primary["records"]) == len(control["records"]) == 16
    assert primary["gate_evidence"]["all_predeclared_gates_pass"] is True
    assert control["gate_evidence"]["all_predeclared_gates_pass"] is False
    assert primary["execution_scope_id"] != control["execution_scope_id"]
    assert "pooled" not in primary and "pooled" not in control

    failed_records = [dict(record) for record in control_records]
    for index in (0, 1):
        failed_records[index] = {
            **failed_records[index],
            "status": "operational_failure",
            "failure_reason": "RuntimeError",
            "scores": {},
            "metrics": {},
        }
    failed = engine._derive_result(
        failed_records, control_protocol, control_identity, variant=control_variant
    )
    assert failed["fixed_denominator_units"] == 8
    assert failed["gate_evidence"] is None
    assert failed["failed_units"] == [{
        "unit_id": control_protocol.roster[0].unit_id,
        "status": "failed",
        "error_type": "RuntimeError",
    }]


@pytest.mark.integration
def test_content_v5_entrypoint_requires_explicit_cohort_without_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[engine.ContentRunnerVariant] = []

    def observed(
        args: argparse.Namespace,
        *,
        variant: engine.ContentRunnerVariant,
    ) -> int:
        del args
        called.append(variant)
        return 7

    monkeypatch.setattr(engine, "execute", observed)
    assert runner.execute(argparse.Namespace(cohort="primary_1")) == 7
    assert runner.execute(argparse.Namespace(cohort="control_1")) == 7
    assert called == [
        runner.CONTENT_V5_PRIMARY_RUNNER_VARIANT,
        runner.CONTENT_V5_CONTROL_RUNNER_VARIANT,
    ]
    with pytest.raises(ValueError, match="requires explicit"):
        runner.execute(argparse.Namespace())
    with pytest.raises(ValueError, match="requires explicit"):
        runner.execute(argparse.Namespace(cohort="primary"))


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
        source = _ROOT / "configs" / "content_chain" / name
        shutil.copyfile(source, config_root / name)
    primary = config_root / "content_v5_primary_evaluation_v1.jsonl"
    if failure == "absent":
        primary.unlink()
    else:
        primary.write_bytes(primary.read_bytes().replace(b"beekeeper", b"beekeeqer", 1))
    model_calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        model_calls.append("called")
        raise AssertionError("model loader was reached")

    monkeypatch.setattr(v4_runner, "_load_pipeline_and_assets", forbidden)
    with pytest.raises((FileNotFoundError, ValueError)):
        runner._load_primary_protocol(tmp_path)
    assert model_calls == []
