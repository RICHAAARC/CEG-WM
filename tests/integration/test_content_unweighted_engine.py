from __future__ import annotations

# Functional coverage for the content-unweighted engine.

import inspect
from pathlib import Path

from PIL import Image
import pytest

from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.protocol.content_adaptive import (
    load_content_adaptive_protocol,
)
from cegwm.protocol.content_unweighted import CONTENT_UNWEIGHTED_PROTOCOL_DIGEST
from experiments import content_adaptive_engine as engine
from experiments import content_unweighted_engine as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "runner-key-value-01"


def _protocol():
    return runner._load_protocol(_ROOT)


def _identity():
    protocol = _protocol()
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    run_id = f"content-v3-{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    identity = engine._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=key_digest,
        run_id=run_id,
        variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
    )
    return protocol, identity


def _scores(registered: float, wrong: float) -> dict[str, float]:
    values = {
        "registered": registered,
        **{f"wrong_{index:02d}": wrong for index in range(16)},
    }
    return engine._flat_scores({"lf": dict(values), "hf": dict(values), "joint": dict(values)})


def _success_records() -> tuple[object, dict[str, object], list[dict[str, object]]]:
    protocol, identity = _identity()
    records: list[dict[str, object]] = []
    for index, unit in enumerate(protocol.roster):
        lf_share = 0.32 + index * 0.02
        effects = {
            name: 0.01 * (effect + 1)
            for effect, name in enumerate(engine.COUNTERFACTUAL_EFFECT_FIELDS)
        }
        metrics = {
            "combined_relative_l2": 0.0119,
            "lf_effective_relative_l2": 0.005,
            "hf_effective_relative_l2": 0.007,
            "lf_branch_share": lf_share,
            "hf_branch_share": 1.0 - lf_share,
            **effects,
            "minimum_counterfactual_effect": min(effects.values()),
            "probe_evaluation_count": 64.0,
            "paired_rgb_psnr_db": 31.0,
        }
        records.extend((
            engine._content_record(
                run_id=identity["run_id"],
                unit_id=unit.unit_id,
                source_cluster_id=unit.source_id,
                arm=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT.arms[0],
                condition="clean",
                code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"],
                status="success",
                scores=_scores(0.9, 0.1),
                metrics=metrics,
                variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
            ),
            engine._content_record(
                run_id=identity["run_id"],
                unit_id=unit.unit_id,
                source_cluster_id=unit.source_id,
                arm=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT.arms[1],
                condition="clean",
                code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"],
                status="success",
                scores=_scores(0.2, 0.1),
                metrics={"paired_rgb_psnr_db": 31.0},
                variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
            ),
        ))
    return protocol, identity, records


@pytest.mark.integration
def test_content_unweighted_runner_binds_distinct_deterministic_run_record_and_state_identity() -> None:
    protocol, identity = _identity()
    assert protocol.protocol_digest == CONTENT_UNWEIGHTED_PROTOCOL_DIGEST
    assert identity["run_id"] == "content-v3-6b812bbef380-8fac30fb16d4"
    assert identity["public_key_digest"][:12] == "8fac30fb16d4"
    assert identity["ordered_arms"] == list(runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT.arms)
    assert identity["record_contract_id"] == (
        "content_v3_unweighted_lf_adaptive_hf_record_v1"
    )
    assert identity["execution_scope_id"] == (
        "content_v3_unweighted_lf_adaptive_hf_engineering_and_stage_a_evaluation_v1"
    )
    state = engine._new_state(
        identity, 1.0, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    assert state["state_schema_id"] == "content_v3_resumable_state_v1"
    assert state["identity"]["fixed_record_count"] == 16

    root = _ROOT / "configs" / "content_chain"
    adaptive = load_content_adaptive_protocol(
        root / "content_adaptive_dual_branch_v2_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )
    adaptive_identity = engine._public_identity(
        adaptive,
        exact=_EXACT,
        key_digest=identity["public_key_digest"],
        run_id="content-adaptive-v2-e3fe3fd32ca2-805bc21e173a",
    )
    assert adaptive_identity["ordered_arms"] == list(engine.ARMS)
    assert adaptive_identity["record_contract_id"] == engine.RECORD_CONTRACT_ID
    assert adaptive_identity["execution_scope_id"] == engine.EXECUTION_SCOPE_ID
    assert adaptive_identity["public_key_digest"] == identity["public_key_digest"]


@pytest.mark.integration
def test_content_unweighted_runner_fixed_records_gates_strict_ties_and_formal_fpr_false() -> None:
    protocol, identity, records = _success_records()
    state = engine._new_state(
        identity, 1.0, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    state["committed_unit_count"] = 8
    state["records"] = records
    engine._validate_state(
        state, identity, protocol, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    result = engine._derive_result(
        records, protocol, identity, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    assert result["rc"] == 0 and len(result["records"]) == 16
    assert result["fixed_denominator_units"] == 8
    assert result["gate_evidence"]["all_predeclared_gates_pass"] is True
    assert result["gate_evidence"]["formal_fpr_claim"] is False
    assert result["lf_branch_share_population_std"] > 0.0
    assert result["hf_branch_share_population_std"] > 0.0

    tied = [dict(record) for record in records]
    for unit_index in (0,):
        index = unit_index * 2
        tied[index] = dict(tied[index])
        tied[index]["scores"] = _scores(0.1, 0.1)
    gates = engine._gate_evidence(
        tied,
        result["unit_aggregate_metrics"],
        variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
    )
    assert all(branch["gate_a_pass_units"] == 7 for branch in gates["branches"].values())
    assert all(branch["strict_ties_fail"] is True for branch in gates["branches"].values())


@pytest.mark.integration
def test_content_unweighted_wrong_keys_blind_lf_hf_min_joint_and_detector_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _ = _identity()
    key = engine.normalize_detection_key(_KEY)
    wrong_first = engine._wrong_keys(key, protocol)
    wrong_second = engine._wrong_keys(key, protocol)
    assert wrong_first == wrong_second
    assert len(wrong_first) == len(set(wrong_first)) == 16
    assert all(item != key and len(item) == 32 for item in wrong_first)

    class _VAE:
        def encode(self, pixels: object) -> object:
            return pixels

    class _Processor:
        def preprocess(self, image: object) -> object:
            return image

    vae, processor = _VAE(), _Processor()
    hf = FrozenHFPublicAssets(vae, processor, "fixture")
    lf = FrozenLFPublicAssets(
        vae,
        processor,
        "fixture",
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    monkeypatch.setattr(engine, "score_lf_image", lambda image, key, assets: 0.8)
    monkeypatch.setattr(engine, "score_hf_image", lambda image, key, assets: 0.6)
    scores = engine._blind_scores(
        Image.new("RGB", (16, 16)), key, wrong_first, hf, lf
    )
    assert scores["lf"]["registered"] == 0.8
    assert scores["hf"]["registered"] == 0.6
    assert scores["joint"]["registered"] == 0.6
    assert all(
        scores["joint"][label] == min(scores["lf"][label], scores["hf"][label])
        for label in scores["joint"]
    )
    assert tuple(inspect.signature(engine._blind_scores).parameters) == (
        "image", "key", "wrong_keys", "hf_public_assets", "lf_public_assets",
    )


@pytest.mark.integration
def test_content_unweighted_failure_transaction_stays_in_denominator_without_secret_or_retry() -> None:
    protocol, identity = _identity()
    records = [
        engine._content_record(
            run_id=identity["run_id"],
            unit_id=unit.unit_id,
            source_cluster_id=unit.source_id,
            arm=arm,
            condition="clean",
            code_revision=identity["exact"],
            config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"],
            status="operational_failure",
            failure_reason="RuntimeError",
            variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
        )
        for unit in protocol.roster
        for arm in runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT.arms
    ]
    state = engine._new_state(
        identity, 1.0, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    state["committed_unit_count"] = 8
    state["records"] = records
    engine._validate_state(
        state, identity, protocol, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    result = engine._derive_result(
        records, protocol, identity, variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT
    )
    assert result["rc"] == 2
    assert result["fixed_denominator_units"] == 8
    assert result["failed_units"] == [
        {"unit_id": unit.unit_id, "status": "failed", "error_type": "RuntimeError"}
        for unit in protocol.roster
    ]
    fatal = engine._fatal_result(
        records,
        identity,
        RuntimeError("private runner-key-value-01 hf_secret"),
        variant=runner.CONTENT_UNWEIGHTED_RUNNER_VARIANT,
    )
    serialized = engine._json_bytes(fatal)
    assert b"runner-key-value-01" not in serialized and b"hf_secret" not in serialized
    assert fatal["operational_error_class"] == "RuntimeError"
    assert protocol.config["execution_flow"]["retry_units_allowed"] is False
