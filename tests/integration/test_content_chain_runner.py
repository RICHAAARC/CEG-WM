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

    def adaptive(pipeline: object, prompt: str, key: bytes, received: object, **kwargs: object) -> SimpleNamespace:
        del pipeline, prompt, key
        assert received is assets
        seed = kwargs["generator"].seed
        budget = SimpleNamespace(relative_l2=0.0119)
        measurement = SimpleNamespace(
            combined_budget=budget,
            lf_effective_relative_l2=0.006,
            hf_effective_relative_l2=0.006,
            lf_branch_share=0.5,
            hf_branch_share=0.5,
            minimum_counterfactual_effect=0.01,
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
    assert all(metric["probe_evaluation_count"] == 32 for metric in result["unit_aggregate_metrics"])
    serialized = output.read_text(encoding="utf-8")
    assert all(word not in serialized for word in ("attention_map", "tile_weights", "latents", "deltas", "probe_state"))
    assert "runner-key-value-01" not in serialized
    assert "hf_test" not in serialized


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
