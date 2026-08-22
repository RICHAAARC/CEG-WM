from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
from cegwm.protocol.content_chain import load_content_adaptive_dual_branch_clean_protocol
from experiments import run_content_adaptive_dual_branch_clean as runner

_ROOT = Path(__file__).resolve().parents[2]


class _Generator:
    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed = 0

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _image(seed: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:32, :32]
    pixels = np.stack((xx * 3 + yy, yy * 4 + xx, xx * 2 + yy * 2), axis=-1)
    return Image.fromarray((pixels + seed % 7 + offset + 30).astype(np.uint8), mode="RGB")


def _protocol():
    root = _ROOT / "configs" / "content_chain"
    return load_content_adaptive_dual_branch_clean_protocol(
        root / "content_adaptive_dual_branch_clean_v1.json",
        root / "content_adaptive_dual_branch_clean.jsonl",
    )


@pytest.mark.integration
def test_runner_writes_exact_16_record_transactions_and_strict_three_branch_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    assets = SimpleNamespace(hf_public_assets=object(), lf_public_assets=object())
    monkeypatch.setattr(runner, "_git_exact", lambda repo, exact: exact)
    monkeypatch.setattr(runner, "_load_protocol", lambda repo: protocol)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model, token: (object(), assets))
    monkeypatch.setattr(runner.torch, "Generator", _Generator)

    adaptive_calls = 0

    def adaptive(pipeline: object, prompt: str, key: bytes, received: object, **kwargs: object) -> SimpleNamespace:
        nonlocal adaptive_calls
        del pipeline, prompt, key
        assert received is assets
        seed = kwargs["generator"].seed
        lf_share = 0.20 + 0.05 * adaptive_calls
        hf_share = 1.0 - lf_share
        effects = (
            0.01 + 0.001 * adaptive_calls,
            0.02 + 0.001 * adaptive_calls,
            0.03 + 0.001 * adaptive_calls,
            0.04 + 0.001 * adaptive_calls,
        )
        adaptive_calls += 1
        budget = SimpleNamespace(relative_l2=0.0119)
        measurement = SimpleNamespace(
            combined_budget=budget,
            lf_effective_relative_l2=0.006,
            hf_effective_relative_l2=0.006,
            lf_branch_share=lf_share,
            hf_branch_share=hf_share,
            semantic_attention_counterfactual_effect=effects[0],
            texture_energy_counterfactual_effect=effects[1],
            lf_probe_response_counterfactual_effect=effects[2],
            hf_probe_response_counterfactual_effect=effects[3],
            minimum_counterfactual_effect=min(effects),
            probe_evaluation_count=32,
        )
        return SimpleNamespace(image=_image(seed, 2), measurement=measurement)

    def plain(pipeline: object, prompt: str, **kwargs: object) -> Image.Image:
        del pipeline, prompt
        return _image(kwargs["generator"].seed, 0)

    score_calls = 0

    def scores(
        image: Image.Image,
        key: bytes,
        wrong: tuple[bytes, ...],
        hf_assets: object,
        lf_assets: object,
    ):
        nonlocal score_calls
        del image, key
        assert hf_assets is assets.hf_public_assets
        assert lf_assets is assets.lf_public_assets
        is_joint = score_calls % 2 == 0
        score_calls += 1
        registered = 0.9 if is_joint else 0.2
        values = {"registered": registered, **{f"wrong_{index:02d}": 0.1 for index in range(len(wrong))}}
        return {"lf": dict(values), "hf": dict(values), "joint": dict(values)}

    monkeypatch.setattr(runner, "run_sd35_content_adaptive", adaptive)
    monkeypatch.setattr(runner, "run_sd35_plain", plain)
    monkeypatch.setattr(runner, "_blind_scores", scores)
    monkeypatch.setenv(runner.KEY_ENV, "runner-key-value-01")
    monkeypatch.setenv(runner.TOKEN_ENV, "hf_test")
    output = tmp_path / "result.json"
    args = argparse.Namespace(repo_root=str(_ROOT), expected_exact="a" * 40, output=str(output))
    assert runner.execute(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["rc"] == 0 and result["scientific_outcome_allowed"] is True
    assert len(result["records"]) == 16
    assert [record["arm"] for record in result["records"][:2]] == list(runner.ARMS)
    assert all(len(record["scores"]) == 51 for record in result["records"])
    assert all(record["schema_version"] == 1 for record in result["records"])
    assert all(value["gate_a_pass_units"] == 8 for value in result["gate_evidence"]["branches"].values())
    assert all(value["gate_b_pass_units"] == 8 for value in result["gate_evidence"]["branches"].values())
    assert result["gate_evidence"]["combined_budget_pass_units"] == 8
    assert result["gate_evidence"]["both_nonzero_branches_pass_units"] == 8
    expected_lf_shares = np.asarray([0.20 + 0.05 * index for index in range(8)])
    expected_hf_shares = 1.0 - expected_lf_shares
    assert result["lf_branch_share_population_std"] == pytest.approx(
        float(np.std(expected_lf_shares, ddof=0))
    )
    assert result["hf_branch_share_population_std"] == pytest.approx(
        float(np.std(expected_hf_shares, ddof=0))
    )
    assert result["lf_branch_share_population_std"] > 0.0
    assert result["hf_branch_share_population_std"] > 0.0
    assert result["fixed_roster_allocation_not_all_identical_supported"] is True
    candidate_fields = {
        "combined_relative_l2",
        "lf_effective_relative_l2",
        "hf_effective_relative_l2",
        "lf_branch_share",
        "hf_branch_share",
        "semantic_attention_counterfactual_effect",
        "texture_energy_counterfactual_effect",
        "lf_probe_response_counterfactual_effect",
        "hf_probe_response_counterfactual_effect",
        "minimum_counterfactual_effect",
        "probe_evaluation_count",
        "paired_rgb_psnr_db",
    }
    assert all(
        set(metric) == {"unit_id", *candidate_fields}
        for metric in result["unit_aggregate_metrics"]
    )
    candidate_records = [record for record in result["records"] if record["arm"] == runner.ARMS[0]]
    null_records = [record for record in result["records"] if record["arm"] == runner.ARMS[1]]
    assert all(set(record["metrics"]) == candidate_fields for record in candidate_records)
    assert all(set(record["metrics"]) == {"paired_rgb_psnr_db"} for record in null_records)
    first = result["unit_aggregate_metrics"][0]
    assert [first[name] for name in (
        "semantic_attention_counterfactual_effect",
        "texture_energy_counterfactual_effect",
        "lf_probe_response_counterfactual_effect",
        "hf_probe_response_counterfactual_effect",
    )] == [0.01, 0.02, 0.03, 0.04]
    assert first["minimum_counterfactual_effect"] == 0.01
    assert all(metric["probe_evaluation_count"] == 32 for metric in result["unit_aggregate_metrics"])
    serialized = output.read_text(encoding="utf-8")
    assert all(
        word not in serialized
        for word in ("attention_map", "tile_weights", "latents", "deltas", "probe_state")
    )
    result_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            result_keys.update(str(key) for key in value)
            for item in value.values():
                collect_keys(item)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item)

    collect_keys(result)
    assert result_keys.isdisjoint(
        {"mask", "tile_weights", "attention_map", "latent", "latents", "delta", "deltas", "probe_state"}
    )
    assert "runner-key-value-01" not in serialized
    assert "hf_test" not in serialized

    adaptive_calls = 0
    score_calls = 0
    monkeypatch.setattr(
        runner,
        "_branch_share_population_summary",
        lambda *args, **kwargs: (None, None, False, False),
    )
    identity_invalid_output = tmp_path / "identity-invalid-result.json"
    identity_invalid_args = argparse.Namespace(
        repo_root=str(_ROOT),
        expected_exact="a" * 40,
        output=str(identity_invalid_output),
    )
    assert runner.execute(identity_invalid_args) == 1
    identity_invalid_result = json.loads(identity_invalid_output.read_text(encoding="utf-8"))
    assert identity_invalid_result["scientific_outcome_allowed"] is False
    assert identity_invalid_result["lf_branch_share_population_std"] is None
    assert identity_invalid_result["hf_branch_share_population_std"] is None
    assert identity_invalid_result["gate_evidence"] is None


@pytest.mark.integration
def test_runner_keeps_failed_unit_in_fixed_denominator_and_never_claims_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    assets = SimpleNamespace(hf_public_assets=object(), lf_public_assets=object())
    monkeypatch.setattr(runner, "_git_exact", lambda repo, exact: exact)
    monkeypatch.setattr(runner, "_load_protocol", lambda repo: protocol)
    monkeypatch.setattr(runner, "_load_pipeline_and_assets", lambda model, token: (object(), assets))
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    monkeypatch.setattr(runner, "run_sd35_content_adaptive", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private")))
    monkeypatch.setattr(runner, "run_sd35_plain", lambda *args, **kwargs: Image.new("RGB", (32, 32)))
    monkeypatch.setenv(runner.KEY_ENV, "runner-key-value-01")
    monkeypatch.setenv(runner.TOKEN_ENV, "hf_test")
    output = tmp_path / "failed.json"
    args = argparse.Namespace(repo_root=str(_ROOT), expected_exact="b" * 40, output=str(output))
    assert runner.execute(args) == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scientific_outcome_allowed"] is False
    assert result["completeness"] == runner.INCOMPLETE_EXECUTION
    assert len(result["failed_units"]) == 8
    assert len(result["records"]) == 16
    assert all(record["status"] == "operational_failure" for record in result["records"])
    assert all("private" not in failure for failure in result["failed_units"])
    assert result["lf_branch_share_population_std"] is None
    assert result["hf_branch_share_population_std"] is None
    assert result["fixed_roster_allocation_not_all_identical_supported"] is False


@pytest.mark.integration
def test_population_std_is_null_for_non_rc0_partial_nonfinite_or_identity_invalid() -> None:
    unit_ids = tuple(f"unit-{index}" for index in range(8))
    metrics = [
        {
            "unit_id": unit_id,
            "lf_branch_share": 0.2 + index * 0.05,
            "hf_branch_share": 0.8 - index * 0.05,
        }
        for index, unit_id in enumerate(unit_ids)
    ]
    for rc in (1, 2):
        assert runner._branch_share_population_summary(
            metrics,
            unit_ids,
            rc=rc,
            share_sum_absolute_tolerance=1e-12,
            population_std_absolute_tolerance=1e-12,
        ) == (None, None, False, False)
    assert runner._branch_share_population_summary(
        metrics[:-1],
        unit_ids,
        rc=0,
        share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)
    nonfinite = [dict(metric) for metric in metrics]
    nonfinite[3]["lf_branch_share"] = float("nan")
    assert runner._branch_share_population_summary(
        nonfinite,
        unit_ids,
        rc=0,
        share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)
    identity_invalid = [dict(metric) for metric in metrics]
    identity_invalid[2]["unit_id"] = identity_invalid[1]["unit_id"]
    assert runner._branch_share_population_summary(
        identity_invalid,
        unit_ids,
        rc=0,
        share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)
    share_invalid = [dict(metric) for metric in metrics]
    share_invalid[4]["hf_branch_share"] += 0.01
    assert runner._branch_share_population_summary(
        share_invalid,
        unit_ids,
        rc=0,
        share_sum_absolute_tolerance=1e-12,
        population_std_absolute_tolerance=1e-12,
    ) == (None, None, False, False)


class _BlindVAE(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels))


class _BlindProcessor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        del image
        return torch.zeros((1, 3, 2, 2))


@pytest.mark.integration
def test_recorded_score_helper_accepts_only_ordinary_image_keys_and_frozen_public_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = tuple(inspect.signature(runner._blind_scores).parameters)
    assert parameters == (
        "image", "key", "wrong_keys", "hf_public_assets", "lf_public_assets",
    )
    vae, processor = _BlindVAE(), _BlindProcessor()
    image_processor_id = "stabilityai/stable-diffusion-3.5-medium:image_processor"
    hf_assets = FrozenHFPublicAssets(vae, processor, image_processor_id)
    lf_assets = FrozenLFPublicAssets(
        vae, processor, image_processor_id, LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID, LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    wrong_keys = tuple(f"wrong-{index:02d}".encode() for index in range(16))
    monkeypatch.setattr(runner, "score_lf_image", lambda image, key, assets: float(len(key)))
    monkeypatch.setattr(runner, "score_hf_image", lambda image, key, assets: float(len(key) + 2))
    values = runner._blind_scores(
        Image.new("RGB", (4, 4)), b"registered", wrong_keys, hf_assets, lf_assets,
    )
    assert values["joint"]["registered"] == min(
        values["lf"]["registered"], values["hf"]["registered"]
    )
    assert all(
        values["joint"][label] == min(values["lf"][label], values["hf"][label])
        for label in values["joint"]
    )
    with pytest.raises(ValueError, match="RGB"):
        runner._blind_scores(
            Image.new("L", (4, 4)), b"registered", wrong_keys, hf_assets, lf_assets,
        )
    with pytest.raises(TypeError, match="FrozenHFPublicAssets"):
        runner._blind_scores(
            Image.new("RGB", (4, 4)), b"registered", wrong_keys, object(), lf_assets,
        )
