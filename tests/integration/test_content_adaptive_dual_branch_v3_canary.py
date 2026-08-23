from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.method.lf import FrozenLFPublicAssets
from experiments import run_content_adaptive_dual_branch_v3_canary as runner

_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "a" * 40
_KEY = "canary-test-root-key-01"
_TOKEN = "canary-test-hf-token"


class _VAE:
    def encode(self, pixels: torch.Tensor) -> object:
        del pixels
        raise AssertionError("controlled canary tests must not invoke the VAE")


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        del image
        raise AssertionError("controlled canary tests must not preprocess an image")


class _Pipeline:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self.vae = _VAE()
        self.image_processor = _Processor()
        self.calls = calls

    def to(self, device: str) -> _Pipeline:
        self.calls.append(("pipeline.to", device))
        return self


class _Dino:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self.calls = calls

    def to(self, device: str) -> _Dino:
        self.calls.append(("dino.to", device))
        return self

    def eval(self) -> _Dino:
        self.calls.append(("dino.eval",))
        return self


class _Generator:
    instances: list[_Generator] = []

    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed: int | None = None
        self.instances.append(self)

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _image(value: int) -> Image.Image:
    return Image.fromarray(
        np.full((runner.HEIGHT, runner.WIDTH, 3), value, dtype=np.uint8),
        mode="RGB",
    )


def _measurement(**updates: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "combined_budget": SimpleNamespace(relative_l2=0.0115),
        "lf_effective_relative_l2": 0.005,
        "hf_effective_relative_l2": 0.006,
        "lf_branch_share": 0.4,
        "hf_branch_share": 0.6,
        "minimum_counterfactual_effect": 0.01,
        "probe_evaluation_count": 64,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _args() -> argparse.Namespace:
    return argparse.Namespace(repo_root=str(_ROOT), expected_exact=_EXACT)


def _set_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)


def _install_controlled_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    measurement_updates: dict[str, Any] | None = None,
    same_images: bool = False,
    failure_at: str | None = None,
) -> tuple[list[tuple[Any, ...]], list[tuple[str, Image.Image, bytes, object]]]:
    calls: list[tuple[Any, ...]] = []
    score_calls: list[tuple[str, Image.Image, bytes, object]] = []
    _Generator.instances = []
    _set_secrets(monkeypatch)

    monkeypatch.setattr(runner, "_resolve_exact", lambda root: _EXACT)

    def validate(root: Path, expected: str, exact: str) -> None:
        calls.append(("identity", root, expected, exact))

    def non_roster(root: Path) -> None:
        calls.append(("non_roster", root))

    monkeypatch.setattr(runner, "_validate_execution_identity", validate)
    monkeypatch.setattr(runner, "_assert_non_roster_identity", non_roster)

    pipeline = _Pipeline(calls)

    def load_pipeline(model_id: str, *, torch_dtype: torch.dtype, token: str) -> _Pipeline:
        calls.append(("load_sd35_pipeline", model_id, torch_dtype, token))
        if failure_at == "pipeline":
            raise RuntimeError(f"private pipeline {_KEY} {_TOKEN}")
        return pipeline

    monkeypatch.setattr(runner, "load_sd35_pipeline", load_pipeline)

    dino = _Dino(calls)
    processor = object()

    def load_dino(*, token: str) -> tuple[_Dino, object]:
        calls.append(("load_dino_content_assets", token))
        if failure_at == "dino":
            raise RuntimeError(f"private dino {_KEY} {_TOKEN}")
        return dino, processor

    monkeypatch.setattr(runner, "load_dino_content_assets", load_dino)

    def content_assets(
        received_dino: _Dino,
        received_processor: object,
        hf_assets: FrozenHFPublicAssets,
        lf_assets: FrozenLFPublicAssets,
    ) -> SimpleNamespace:
        calls.append(("ContentEmbedAssets", received_dino, received_processor))
        assert received_dino is dino and received_processor is processor
        assert isinstance(hf_assets, FrozenHFPublicAssets)
        assert isinstance(lf_assets, FrozenLFPublicAssets)
        return SimpleNamespace(
            hf_public_assets=hf_assets,
            lf_public_assets=lf_assets,
        )

    monkeypatch.setattr(runner, "ContentEmbedAssets", content_assets)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)

    joint_image = _image(11)
    null_image = joint_image.copy() if same_images else _image(10)
    measurement = _measurement(**(measurement_updates or {}))

    def adaptive(
        received_pipeline: _Pipeline,
        prompt: str,
        key: bytes,
        assets: object,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls.append(("run_sd35_content_adaptive", prompt, key, assets, kwargs))
        assert received_pipeline is pipeline
        if failure_at == "joint":
            raise RuntimeError(f"private joint {_KEY} {_TOKEN}")
        return SimpleNamespace(image=joint_image, measurement=measurement)

    def plain(received_pipeline: _Pipeline, prompt: str, **kwargs: Any) -> Image.Image:
        calls.append(("run_sd35_plain", prompt, kwargs))
        assert received_pipeline is pipeline
        if failure_at == "null":
            raise RuntimeError(f"private null {_KEY} {_TOKEN}")
        return null_image

    monkeypatch.setattr(runner, "run_sd35_content_adaptive", adaptive)
    monkeypatch.setattr(runner, "run_sd35_plain", plain)

    def lf_score(image: Image.Image, key: bytes, assets: FrozenLFPublicAssets) -> float:
        score_calls.append(("lf", image, key, assets))
        if failure_at == "scoring":
            raise RuntimeError(f"private score {_KEY} {_TOKEN}")
        return 0.7 if int(np.asarray(image)[0, 0, 0]) == 11 else 0.2

    def hf_score(image: Image.Image, key: bytes, assets: FrozenHFPublicAssets) -> float:
        score_calls.append(("hf", image, key, assets))
        return 0.8 if int(np.asarray(image)[0, 0, 0]) == 11 else 0.3

    monkeypatch.setattr(runner, "score_lf_image", lf_score)
    monkeypatch.setattr(runner, "score_hf_image", hf_score)
    return calls, score_calls


def _payload(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"{runner.PREFIX} ")
    assert len(lines[0]) <= 4096
    return json.loads(lines[0].split(" ", 1)[1]), lines[0]


@pytest.mark.integration
def test_fixed_identity_is_absent_from_the_frozen_roster() -> None:
    runner._assert_non_roster_identity(_ROOT)
    assert (
        runner.CANARY_ID,
        runner.UNIT_ID,
        runner.SOURCE_ID,
        runner.PROMPT,
        runner.SEED,
        runner.HEIGHT,
        runner.WIDTH,
    ) == (
        "content-v3-full-runtime-non-roster-canary-v1",
        "content-v3-canary-0001",
        "content-v3-canary-prompt-9001",
        "A book conservator examining an illuminated manuscript under neutral studio light",
        1415149,
        512,
        512,
    )


@pytest.mark.integration
def test_one_joint_one_null_and_blind_final_image_scoring_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, score_calls = _install_controlled_runtime(monkeypatch)

    assert runner.execute(_args()) == 0
    payload, line = _payload(capsys)

    assert [call[0] for call in calls] == [
        "identity",
        "non_roster",
        "load_sd35_pipeline",
        "pipeline.to",
        "load_dino_content_assets",
        "dino.to",
        "dino.eval",
        "ContentEmbedAssets",
        "run_sd35_content_adaptive",
        "run_sd35_plain",
    ]
    assert calls[2][1:] == (runner.MODEL_ID, torch.float16, _TOKEN)
    assert calls[4] == ("load_dino_content_assets", _TOKEN)
    joint_call = calls[8]
    null_call = calls[9]
    assert joint_call[1] == null_call[1] == runner.PROMPT
    assert joint_call[2] == runner.normalize_detection_key(_KEY)
    assert joint_call[4]["height"] == null_call[2]["height"] == 512
    assert joint_call[4]["width"] == null_call[2]["width"] == 512
    assert len(_Generator.instances) == 2
    assert _Generator.instances[0] is not _Generator.instances[1]
    assert [item.seed for item in _Generator.instances] == [runner.SEED, runner.SEED]

    assert [item[0] for item in score_calls] == ["lf", "hf", "lf", "hf"]
    assert all(isinstance(item[1], Image.Image) and item[1].mode == "RGB" for item in score_calls)
    assert all(item[2] == runner.normalize_detection_key(_KEY) for item in score_calls)
    assert all(isinstance(item[3], (FrozenHFPublicAssets, FrozenLFPublicAssets)) for item in score_calls)

    assert tuple(payload) == tuple(sorted(runner._SUCCESS_FIELDS))
    assert payload["status"] == "operational_canary_pass"
    assert payload["claim_ceiling"] == "full_non_roster_runtime_canary_only"
    assert payload["formal_roster_member"] is False
    assert payload["scientific_denominator_units"] == 0
    assert payload["probe_evaluation_count"] == 64
    assert payload["combined_actual_dtype_relative_l2"] == 0.0115
    assert payload["joint_registered_joint_score"] == 0.7
    assert payload["primary_null_registered_joint_score"] == 0.2
    assert runner.KEY_ENV not in runner.os.environ
    assert runner.TOKEN_ENV not in runner.os.environ
    for forbidden in (
        _KEY,
        _TOKEN,
        runner.PROMPT,
        "latent",
        "delta",
        "carrier",
        "attention",
        "semantic_routing",
        "texture",
        "tile_weight",
        "mask",
        "private_state",
    ):
        assert forbidden not in line


@pytest.mark.integration
@pytest.mark.parametrize(
    ("updates", "same_images"),
    [
        ({"combined_budget": SimpleNamespace(relative_l2="invalid")}, False),
        ({"combined_budget": SimpleNamespace(relative_l2=float("nan"))}, False),
        ({"combined_budget": SimpleNamespace(relative_l2=0.0121)}, False),
        ({"lf_effective_relative_l2": 0.0}, False),
        ({"probe_evaluation_count": 63}, False),
        ({"lf_branch_share": 0.4, "hf_branch_share": 0.7}, False),
        ({}, True),
    ],
    ids=("invalid", "nonfinite", "overbudget", "zero-branch", "wrong-probe", "share", "psnr"),
)
def test_invalid_aggregate_or_quality_is_one_bounded_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    updates: dict[str, Any],
    same_images: bool,
) -> None:
    _install_controlled_runtime(
        monkeypatch,
        measurement_updates=updates,
        same_images=same_images,
    )
    assert runner.execute(_args()) == 1
    payload, line = _payload(capsys)
    assert payload == {
        "status": "operational_failure",
        "canary_id": runner.CANARY_ID,
        "exact": _EXACT,
        "stage": "budget_quality_validation",
        "error_class": payload["error_class"],
    }
    assert payload["error_class"] in {"TypeError", "ValueError"}
    assert _KEY not in line and _TOKEN not in line and runner.PROMPT not in line


@pytest.mark.integration
@pytest.mark.parametrize(
    ("failure_at", "stage"),
    [
        ("pipeline", "sd35_pipeline_load"),
        ("dino", "dino_asset_validation"),
        ("joint", "joint_generation"),
        ("null", "primary_null_generation"),
        ("scoring", "blind_scoring"),
    ],
)
def test_runtime_exceptions_are_one_sanitized_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_at: str,
    stage: str,
) -> None:
    _install_controlled_runtime(monkeypatch, failure_at=failure_at)
    assert runner.execute(_args()) == 1
    payload, line = _payload(capsys)
    assert payload == {
        "status": "operational_failure",
        "canary_id": runner.CANARY_ID,
        "exact": _EXACT,
        "stage": stage,
        "error_class": "RuntimeError",
    }
    assert _KEY not in line and _TOKEN not in line and runner.PROMPT not in line


@pytest.mark.integration
def test_identity_failure_is_sanitized_and_does_not_start_model_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_secrets(monkeypatch)
    monkeypatch.setattr(runner, "_resolve_exact", lambda root: _EXACT)

    def reject(*args: object) -> None:
        del args
        raise RuntimeError(f"private identity {_KEY} {_TOKEN}")

    monkeypatch.setattr(runner, "_validate_execution_identity", reject)
    monkeypatch.setattr(
        runner,
        "load_sd35_pipeline",
        lambda *args, **kwargs: pytest.fail("model work must not start"),
    )
    assert runner.execute(_args()) == 1
    payload, line = _payload(capsys)
    assert payload == {
        "status": "operational_failure",
        "canary_id": runner.CANARY_ID,
        "exact": _EXACT,
        "stage": "identity_validation",
        "error_class": "RuntimeError",
    }
    assert _KEY not in line and _TOKEN not in line
