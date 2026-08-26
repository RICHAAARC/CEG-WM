from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v6_clean as runner
from cegwm.protocol.content_chain_v6 import CONTENT_V6_PROTOCOL_DIGEST

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_v6_runner_binds_final_protocol_and_paired_variant() -> None:
    protocol = runner._load_protocol(_ROOT)
    variant = runner.CONTENT_V6_RUNNER_VARIANT
    assert protocol.protocol_digest == CONTENT_V6_PROTOCOL_DIGEST
    assert len(protocol.roster) == 8
    assert variant.run_pair is runner._run_pair
    assert variant.run_joint is runner._unpaired_forbidden
    assert variant.lf_scorer is runner.score_content_v4_lf_image


@pytest.mark.integration
def test_generic_engine_uses_v6_pair_primary_null_without_third_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runner._load_protocol(_ROOT)
    unit = protocol.roster[0]
    calls: list[tuple[object, ...]] = []
    measurement = SimpleNamespace()

    def paired(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((*args, kwargs))
        return SimpleNamespace(image="joint-image", primary_null="pass1-image", measurement=measurement)

    variant = replace(runner.CONTENT_V6_RUNNER_VARIANT, run_pair=paired)
    monkeypatch.setattr(
        engine,
        "run_sd35_plain",
        lambda *args, **kwargs: pytest.fail("V6 engine must not create a third generation"),
    )
    score_calls: list[str] = []

    def blind(image: str, *args: object, **kwargs: object) -> dict[str, dict[str, float]]:
        del args, kwargs
        score_calls.append(image)
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
    assets = SimpleNamespace(hf_public_assets=object(), lf_public_assets=object())
    records = engine._unit_transaction(
        unit=unit,
        pipeline=object(),
        assets=assets,
        key=b"registered-key",
        wrong_keys=tuple(bytes([index]) for index in range(16)),
        identity={
            "run_id": "content-v6-run",
            "exact": "a" * 40,
            "protocol_digest": protocol.protocol_digest,
            "public_key_digest": "b" * 64,
        },
        protocol=protocol,
        variant=variant,
    )
    assert len(calls) == 1
    assert calls[0][-1]["seed"] == unit.seed
    assert score_calls == ["joint-image", "pass1-image"]
    assert len(records) == 2
    assert records[0]["arm"] == variant.arms[0]
    assert records[1]["arm"] == variant.arms[1]
    assert records[0]["scores"]["joint__registered"] == 0.8
    assert records[1]["scores"]["joint__registered"] == 0.2
