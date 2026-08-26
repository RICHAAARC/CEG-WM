from __future__ import annotations

# Functional coverage for the content-whitening engine.

import argparse
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.protocol.content_whitening import CONTENT_WHITENING_PROTOCOL_DIGEST
from experiments import content_adaptive_engine as engine
from experiments import content_whitening_engine as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "runner-key-value-01"
_REGISTERED_PUBLIC_KEY_DIGEST = (
    "805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77"
)


def _identity():
    protocol = runner._load_protocol(_ROOT)
    key_digest = engine.public_key_digest(engine.normalize_detection_key(_KEY))
    run_id = f"content-v4-{protocol.protocol_digest[:12]}-{key_digest[:12]}"
    identity = engine._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=key_digest,
        run_id=run_id,
        variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
    )
    return protocol, identity


def _flat_scores(registered: float, wrong: float) -> dict[str, float]:
    values = {"registered": registered, **{f"wrong_{index:02d}": wrong for index in range(16)}}
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
                run_id=identity["run_id"], unit_id=unit.unit_id,
                source_cluster_id=unit.source_id, arm=runner.CONTENT_WHITENING_RUNNER_VARIANT.arms[0],
                condition="clean", code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"], status="success",
                scores=_flat_scores(0.9, 0.1), metrics=metrics,
                variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
            ),
            engine._content_record(
                run_id=identity["run_id"], unit_id=unit.unit_id,
                source_cluster_id=unit.source_id, arm=runner.CONTENT_WHITENING_RUNNER_VARIANT.arms[1],
                condition="clean", code_revision=identity["exact"],
                config_digest=identity["protocol_digest"],
                key_public_digest=identity["public_key_digest"], status="success",
                scores=_flat_scores(0.2, 0.1), metrics={"paired_rgb_psnr_db": 31.0},
                variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
            ),
        ))
    return protocol, identity, records


@pytest.mark.integration
def test_content_whitening_runner_binds_digest_deterministic_run_asset_and_state_identity() -> None:
    protocol, identity = _identity()
    assert protocol.protocol_digest == CONTENT_WHITENING_PROTOCOL_DIGEST
    assert identity["run_id"] == "content-v4-a9fdf3e5d384-8fac30fb16d4"
    assert identity["ordered_arms"] == list(runner.CONTENT_WHITENING_RUNNER_VARIANT.arms)
    assert identity["record_contract_id"] == "content_v4_whitened_lf_adaptive_hf_record_v1"
    assert identity["execution_scope_id"] == (
        "content_v4_whitened_lf_adaptive_hf_engineering_and_stage_a_evaluation_v1"
    )
    state = engine._new_state(identity, 1.0, variant=runner.CONTENT_WHITENING_RUNNER_VARIANT)
    assert state["state_schema_id"] == "content_v4_resumable_state_v1"
    assert state["identity"]["fixed_unit_count"] == 8
    assert state["identity"]["fixed_record_count"] == 16
    registered_run_id = (
        f"content-v4-{protocol.protocol_digest[:12]}-"
        f"{_REGISTERED_PUBLIC_KEY_DIGEST[:12]}"
    )
    registered_identity = engine._public_identity(
        protocol,
        exact=_EXACT,
        key_digest=_REGISTERED_PUBLIC_KEY_DIGEST,
        run_id=registered_run_id,
        variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
    )
    assert registered_identity["run_id"] == "content-v4-a9fdf3e5d384-805bc21e173a"


class _BlindVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels))


class _BlindProcessor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        del image
        return torch.zeros((1, 3, 2, 2))


@pytest.mark.integration
def test_generic_engine_whitening_hook_scores_registered_16_wrong_and_null_with_same_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _ = _identity()
    key = engine.normalize_detection_key(_KEY)
    wrong_keys = engine._wrong_keys(key, protocol)
    hf_assets = FrozenHFPublicAssets(
        _BlindVAE(), _BlindProcessor(), "stabilityai/stable-diffusion-3.5-medium:image_processor"
    )
    lf_asset = object()
    calls: list[tuple[bytes, object]] = []

    def lf_scorer(image: Image.Image, received_key: bytes, assets: object) -> float:
        assert image.mode == "RGB"
        calls.append((received_key, assets))
        return 0.8 if received_key == key else 0.2

    monkeypatch.setattr(engine, "score_hf_image", lambda image, received_key, assets: 0.6)
    image = Image.new("RGB", (8, 8))
    joint = engine._blind_scores_with_lf_scorer(
        image, key, wrong_keys, hf_assets, lf_asset, lf_scorer
    )
    primary_null = engine._blind_scores_with_lf_scorer(
        image, key, wrong_keys, hf_assets, lf_asset, lf_scorer
    )
    assert len(calls) == 34
    assert all(assets is lf_asset for _, assets in calls)
    assert tuple(joint["lf"]) == ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    assert tuple(primary_null["lf"]) == tuple(joint["lf"])
    assert joint["lf"]["registered"] == 0.8
    assert joint["hf"]["registered"] == 0.6
    assert joint["joint"]["registered"] == 0.6
    assert all(
        joint["joint"][label] == min(joint["lf"][label], joint["hf"][label])
        for label in joint["joint"]
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "malformed",
    (
        tuple(b"wrong" for _ in range(15)),
        (*tuple(b"wrong" for _ in range(15)), bytearray(b"not-bytes")),
    ),
)
def test_shared_blind_score_boundary_rejects_wrong_key_count_and_type(
    malformed: tuple[object, ...],
) -> None:
    vae, processor = _BlindVAE(), _BlindProcessor()
    image_processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf_assets = FrozenHFPublicAssets(vae, processor, image_processor_id)
    legacy_lf_assets = FrozenLFPublicAssets(
        vae,
        processor,
        image_processor_id,
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    image = Image.new("RGB", (8, 8))
    key = b"registered"
    with pytest.raises(ValueError, match="exactly 16 normalized external wrong keys"):
        engine._blind_scores(
            image, key, malformed, hf_assets, legacy_lf_assets  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly 16 normalized external wrong keys"):
        engine._blind_scores_with_lf_scorer(
            image,
            key,
            malformed,  # type: ignore[arg-type]
            hf_assets,
            object(),
            lambda image, received_key, assets: 0.0,
        )


@pytest.mark.integration
def test_content_whitening_fixed_records_gates_strict_ties_and_formal_fpr_false() -> None:
    protocol, identity, records = _success_records()
    state = engine._new_state(identity, 1.0, variant=runner.CONTENT_WHITENING_RUNNER_VARIANT)
    state["committed_unit_count"] = 8
    state["records"] = records
    engine._validate_state(state, identity, protocol, variant=runner.CONTENT_WHITENING_RUNNER_VARIANT)
    result = engine._derive_result(
        records, protocol, identity, variant=runner.CONTENT_WHITENING_RUNNER_VARIANT
    )
    assert result["rc"] == 0 and len(result["records"]) == 16
    assert result["gate_evidence"]["all_predeclared_gates_pass"] is True
    assert result["gate_evidence"]["formal_fpr_claim"] is False

    tied = [dict(record) for record in records]
    tied[0] = dict(tied[0])
    tied[0]["scores"] = _flat_scores(0.1, 0.1)
    gates = engine._gate_evidence(
        tied, result["unit_aggregate_metrics"], variant=runner.CONTENT_WHITENING_RUNNER_VARIANT
    )
    assert all(branch["gate_a_pass_units"] == 7 for branch in gates["branches"].values())
    assert all(branch["strict_ties_fail"] is True for branch in gates["branches"].values())


@pytest.mark.integration
def test_content_whitening_failures_remain_in_denominator_without_secret_or_retry() -> None:
    protocol, identity = _identity()
    records = [
        engine._content_record(
            run_id=identity["run_id"], unit_id=unit.unit_id,
            source_cluster_id=unit.source_id, arm=arm, condition="clean",
            code_revision=identity["exact"], config_digest=identity["protocol_digest"],
            key_public_digest=identity["public_key_digest"], status="operational_failure",
            failure_reason="RuntimeError", variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
        )
        for unit in protocol.roster
        for arm in runner.CONTENT_WHITENING_RUNNER_VARIANT.arms
    ]
    result = engine._derive_result(
        records, protocol, identity, variant=runner.CONTENT_WHITENING_RUNNER_VARIANT
    )
    assert result["rc"] == 2 and result["fixed_denominator_units"] == 8
    assert len(result["failed_units"]) == 8
    fatal = engine._fatal_result(
        records,
        identity,
        RuntimeError("private runner-key-value-01 hf_secret"),
        variant=runner.CONTENT_WHITENING_RUNNER_VARIANT,
    )
    serialized = engine._json_bytes(fatal)
    assert b"runner-key-value-01" not in serialized and b"hf_secret" not in serialized
    assert protocol.config["execution_flow"]["retry_units_allowed"] is False


@pytest.mark.integration
def test_content_whitening_entrypoint_is_thin_engine_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[argparse.Namespace, engine.ContentRunnerVariant]] = []
    args = argparse.Namespace()

    def fake_execute(
        received: argparse.Namespace,
        *,
        variant: engine.ContentRunnerVariant,
    ) -> int:
        observed.append((received, variant))
        return 7

    monkeypatch.setattr(engine, "execute", fake_execute)
    assert runner.execute(args) == 7
    assert observed == [(args, runner.CONTENT_WHITENING_RUNNER_VARIANT)]
    assert runner.CONTENT_WHITENING_RUNNER_VARIANT.lf_scorer is runner.score_content_whitened_lf_image
