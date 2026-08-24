from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from cegwm.protocol.content_chain_v5 import ContentV5ManifestBindingRequired
from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v3_clean as v3_runner
from experiments import run_content_v4_clean as v4_runner
from experiments import run_content_v5_clean as runner

_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.integration
def test_content_v5_aggregate_branchwise_or_has_exact_seven_of_eight_boundary() -> None:
    seven = _records(
        runner.CONTENT_V5_RUNNER_VARIANT.arms,
        [(True, False)] * 7 + [(False, False)],
    )
    evidence = engine._gate_evidence(
        seven, _unit_metrics(), variant=runner.CONTENT_V5_RUNNER_VARIANT
    )
    assert evidence["branchwise_or"]["gate_a_pass_units"] == 7
    assert evidence["branchwise_or"]["gate_b_pass_units"] == 7
    assert evidence["all_decision_gates_pass"] is True
    assert evidence["all_predeclared_gates_pass"] is True
    assert evidence["formal_fpr_claim"] is False

    six = _records(
        runner.CONTENT_V5_RUNNER_VARIANT.arms,
        [(True, False)] * 6 + [(False, False)] * 2,
    )
    failed = engine._gate_evidence(
        six, _unit_metrics(), variant=runner.CONTENT_V5_RUNNER_VARIANT
    )
    assert failed["branchwise_or"]["gate_a_pass_units"] == 6
    assert failed["branchwise_or"]["gate_b_pass_units"] == 6
    assert failed["all_decision_gates_pass"] is False
    assert failed["all_predeclared_gates_pass"] is False


@pytest.mark.integration
def test_content_v5_individual_branch_counts_are_not_conjunctions() -> None:
    records = _records(
        runner.CONTENT_V5_RUNNER_VARIANT.arms,
        [(True, False)] * 4 + [(False, True)] * 4,
    )
    evidence = engine._gate_evidence(
        records, _unit_metrics(), variant=runner.CONTENT_V5_RUNNER_VARIANT
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
def test_content_v5_variant_retains_exact_real_v4_runtime_and_scorer_wiring() -> None:
    variant = runner.CONTENT_V5_RUNNER_VARIANT
    assert variant.load_pipeline_and_assets is v4_runner._load_pipeline_and_assets
    assert variant.run_joint is v4_runner._run_joint
    assert variant.lf_scorer is v4_runner.score_content_v4_lf_image
    assert variant.run_prefix == "content-v5"
    assert variant.state_schema_id == "content_v5_resumable_state_v1"
    assert variant.record_contract_id == (
        "content_v5_whitened_lf_adaptive_hf_branchwise_or_record_v1"
    )


@pytest.mark.integration
def test_content_v5_entrypoint_fails_before_engine_or_model_asset_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> int:
        del args, kwargs
        called.append("called")
        raise AssertionError("execution crossed the absent-manifest boundary")

    monkeypatch.setattr(engine, "execute", forbidden)
    monkeypatch.setattr(v4_runner, "_load_pipeline_and_assets", forbidden)
    args = argparse.Namespace(repo_root=str(_ROOT))
    with pytest.raises(
        ContentV5ManifestBindingRequired,
        match="user_frozen_new_disjoint_manifest_binding",
    ):
        runner.execute(args)
    assert called == []
