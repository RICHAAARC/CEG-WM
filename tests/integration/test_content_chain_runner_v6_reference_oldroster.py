from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v6_clean as v6_runner
from experiments import run_content_v6_iss_reference_oldroster as runner
from cegwm.protocol.content_chain_v6_reference_oldroster import (
    CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST,
)

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "runner-key-value-01"


def _identity():
    protocol = runner._load_protocol(_ROOT)
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    run_id = (
        f"{runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT.run_prefix}-"
        f"{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    )
    identity = engine._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=key_digest,
        run_id=run_id,
        variant=runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT,
    )
    return protocol, identity


def _scores(registered: float, wrong: float) -> dict[str, float]:
    values = {
        "registered": registered,
        **{f"wrong_{index:02d}": wrong for index in range(16)},
    }
    return engine._flat_scores({
        "lf": dict(values), "hf": dict(values), "joint": dict(values),
    })


def _success_records() -> tuple[object, dict[str, object], list[dict[str, object]]]:
    protocol, identity = _identity()
    variant = runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT
    records: list[dict[str, object]] = []
    for index, unit in enumerate(protocol.roster):
        effects = {
            name: 0.01 * (effect + 1)
            for effect, name in enumerate(engine.COUNTERFACTUAL_EFFECT_FIELDS)
        }
        lf_share = 0.32 + index * 0.02
        records.extend((
            engine._content_v2_record(
                run_id=identity["run_id"], unit_id=unit.unit_id,
                source_cluster_id=unit.source_id, arm=variant.arms[0],
                condition="clean", code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"], status="success",
                scores=_scores(0.9, 0.1),
                metrics={
                    "combined_relative_l2": 0.0119,
                    "lf_effective_relative_l2": 0.005,
                    "hf_effective_relative_l2": 0.007,
                    "lf_branch_share": lf_share,
                    "hf_branch_share": 1.0 - lf_share,
                    **effects,
                    "minimum_counterfactual_effect": min(effects.values()),
                    "probe_evaluation_count": 64.0,
                    "paired_rgb_psnr_db": 31.0,
                },
                variant=variant,
            ),
            engine._content_v2_record(
                run_id=identity["run_id"], unit_id=unit.unit_id,
                source_cluster_id=unit.source_id, arm=variant.arms[1],
                condition="clean", code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"], status="success",
                scores=_scores(0.2, 0.1), metrics={"paired_rgb_psnr_db": 31.0},
                variant=variant,
            ),
        ))
    return protocol, identity, records


@pytest.mark.integration
def test_reference_runner_reuses_v6_runtime_with_distinct_run_and_resume_bytes(
    tmp_path: Path,
) -> None:
    protocol, identity = _identity()
    variant = runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT
    base = v6_runner.CONTENT_V6_RUNNER_VARIANT
    assert protocol.protocol_digest == CONTENT_V6_REFERENCE_OLDROSTER_PROTOCOL_DIGEST
    assert identity["run_id"] == (
        "content-v6-reference-oldroster-c98175252406-8fac30fb16d4"
    )
    assert variant.load_pipeline_and_assets is base.load_pipeline_and_assets
    assert variant.run_pair is base.run_pair
    assert variant.run_joint is base.run_joint
    assert variant.lf_scorer is base.lf_scorer
    assert identity["ordered_roster"][0] == [
        "content-adaptive-v2-0001", "content-v2-prompt-8101",
    ]
    assert identity["ordered_arms"] == list(variant.arms)
    assert identity["record_contract_id"] == variant.record_contract_id
    assert identity["execution_scope_id"] == variant.execution_scope_id

    state = engine._new_state(identity, 1.0, variant=variant)
    state_path = tmp_path / "state.json"
    engine._write_local_state(state_path, state)
    assert state_path.read_bytes() == engine._json_bytes(state)
    assert state["state_schema_id"] == variant.state_schema_id
    engine._validate_state(state, identity, protocol, variant=variant)

    base_state = engine._new_state(identity, 1.0, variant=base)
    with pytest.raises(ValueError, match="state schema identity differs"):
        engine._validate_state(base_state, identity, protocol, variant=variant)


@pytest.mark.integration
def test_reference_runner_uses_one_v6_pair_for_joint_and_primary_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, identity = _identity()
    unit = protocol.roster[0]
    calls: list[tuple[object, ...]] = []
    measurement = SimpleNamespace()

    def paired(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((*args, kwargs))
        return SimpleNamespace(
            image="joint-image", primary_null="pass1-image", measurement=measurement,
        )

    variant = replace(
        runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT,
        run_pair=paired,
    )
    monkeypatch.setattr(
        engine,
        "run_sd35_plain",
        lambda *args, **kwargs: pytest.fail("reference runner cannot create a third generation"),
    )
    observed_images: list[str] = []

    def blind(image: str, *args: object, **kwargs: object) -> dict[str, dict[str, float]]:
        del args, kwargs
        observed_images.append(image)
        registered = 0.8 if image == "joint-image" else 0.2
        values = {
            "registered": registered,
            **{f"wrong_{index:02d}": 0.1 for index in range(16)},
        }
        return {"lf": dict(values), "hf": dict(values), "joint": dict(values)}

    monkeypatch.setattr(engine, "_blind_scores_with_lf_scorer", blind)
    monkeypatch.setattr(engine, "_psnr", lambda first, second: 31.0)
    effects = {
        name: 0.01 * (index + 1)
        for index, name in enumerate(engine.COUNTERFACTUAL_EFFECT_FIELDS)
    }
    monkeypatch.setattr(
        engine,
        "_candidate_aggregate_metrics",
        lambda *args, **kwargs: {
            "unit_id": unit.unit_id,
            "combined_relative_l2": 0.012,
            "lf_effective_relative_l2": 0.006,
            "hf_effective_relative_l2": 0.006,
            "lf_branch_share": 0.4,
            "hf_branch_share": 0.6,
            **effects,
            "minimum_counterfactual_effect": min(effects.values()),
            "probe_evaluation_count": 64,
            "paired_rgb_psnr_db": 31.0,
        },
    )
    records = engine._unit_transaction(
        unit=unit,
        pipeline=object(),
        assets=SimpleNamespace(hf_public_assets=object(), lf_public_assets=object()),
        key=b"registered-key",
        wrong_keys=tuple(bytes([index]) for index in range(16)),
        identity=identity,
        protocol=protocol,
        variant=variant,
    )
    assert len(calls) == 1
    assert calls[0][-1]["seed"] == unit.seed
    assert observed_images == ["joint-image", "pass1-image"]
    assert [record["arm"] for record in records] == list(variant.arms)
    assert records[0]["scores"]["joint__registered"] == 0.8
    assert records[1]["scores"]["joint__registered"] == 0.2


@pytest.mark.integration
def test_reference_records_preserve_fixed_gates_ties_and_failures() -> None:
    protocol, identity, records = _success_records()
    variant = runner.CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT
    state = engine._new_state(identity, 1.0, variant=variant)
    state["committed_unit_count"] = 8
    state["records"] = records
    engine._validate_state(state, identity, protocol, variant=variant)
    result = engine._derive_result(records, protocol, identity, variant=variant)
    assert result["rc"] == 0
    assert len(result["records"]) == 16
    assert result["fixed_denominator_units"] == 8
    assert result["gate_evidence"]["all_predeclared_gates_pass"] is True

    tied = [dict(record) for record in records]
    tied[0] = dict(tied[0])
    tied[0]["scores"] = _scores(0.1, 0.1)
    gates = engine._gate_evidence(
        tied, result["unit_aggregate_metrics"], variant=variant
    )
    assert all(branch["gate_a_pass_units"] == 7 for branch in gates["branches"].values())
    assert all(branch["strict_ties_fail"] is True for branch in gates["branches"].values())

    failed = [
        engine._content_v2_record(
            run_id=identity["run_id"], unit_id=unit.unit_id,
            source_cluster_id=unit.source_id, arm=arm, condition="clean",
            code_revision=identity["exact"], config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"],
            status="operational_failure", failure_reason="RuntimeError",
            variant=variant,
        )
        for unit in protocol.roster
        for arm in variant.arms
    ]
    failed_result = engine._derive_result(failed, protocol, identity, variant=variant)
    assert failed_result["rc"] == 2
    assert failed_result["fixed_denominator_units"] == 8
    assert len(failed_result["failed_units"]) == 8
    assert protocol.config["execution_flow"]["retry_units_allowed"] is False
