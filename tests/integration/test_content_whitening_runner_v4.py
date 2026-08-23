from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as functional

from experiments import run_content_v4_clean_null_whitening_fit as runner
from cegwm.method import content_whitening_v4 as v4
from cegwm.runtime.content_whitening_sd35_v4 import materialize_clean_fit_observation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXACT = "34567890abcdef1234567890abcdef1234567890"


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


class _Distribution:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        pooled = functional.avg_pool2d(pixels, kernel_size=8)
        base = pooled.mean(dim=1, keepdim=True)
        channels = torch.cat(
            [base * (1.0 + channel / 16.0) for channel in range(16)], dim=1
        )
        return SimpleNamespace(latent_dist=_Distribution(channels))


class _Pipeline:
    def __init__(self) -> None:
        self.image_processor = _Processor()
        self.vae = _VAE()
        self.calls: list[dict[str, object]] = []

    def to(self, device: str) -> "_Pipeline":
        assert device == "cuda"
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        assert set(kwargs) == {
            "prompt",
            "num_inference_steps",
            "height",
            "width",
            "generator",
            "output_type",
        }
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["height"] == kwargs["width"] == 512
        assert kwargs["output_type"] == "pil"
        generator = kwargs["generator"]
        assert isinstance(generator, torch.Generator)
        prompt = kwargs["prompt"]
        assert isinstance(prompt, str)
        seed = generator.initial_seed()
        y = np.arange(512, dtype=np.uint32)[:, None]
        x = np.arange(512, dtype=np.uint32)[None, :]
        base = (3 * x + 5 * y + (x * y) % 251 + seed + len(prompt)) % 256
        image = np.stack(
            [base, (7 * base + x) % 256, (11 * base + y) % 256], axis=2
        ).astype(np.uint8)
        return SimpleNamespace(images=[Image.fromarray(image, mode="RGB")])


def _args(sink: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(_REPO_ROOT),
        artifact_sink=str(sink),
        expected_exact=_EXACT,
    )


@pytest.mark.integration
def test_controlled_runner_exactly_32_clean_calls_and_public_create_only_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = _Pipeline()
    loaded = []

    def fake_load(token: str) -> _Pipeline:
        loaded.append(token == "fixture-secret-token")
        return pipeline

    monkeypatch.setattr(runner, "_git_exact", lambda root, expected: expected)
    monkeypatch.setattr(runner, "_load_pipeline", fake_load)
    monkeypatch.setattr(
        runner,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setenv(runner.TOKEN_ENV, "fixture-secret-token")
    sink = tmp_path / "sink"
    assert runner.execute(_args(sink)) == 0

    assert loaded == [True]
    assert runner.TOKEN_ENV not in os.environ
    assert len(pipeline.calls) == 32
    manifest = v4.load_fit_manifest(_REPO_ROOT / v4.FIT_MANIFEST_REPO_PATH)
    assert tuple(call["prompt"] for call in pipeline.calls) == tuple(
        entry.prompt for entry in manifest.entries
    )
    assert tuple(
        cast.initial_seed()
        for call in pipeline.calls
        if isinstance((cast := call["generator"]), torch.Generator)
    ) == tuple(entry.generation_seed for entry in manifest.entries)
    assert all(
        not any(name in call for name in ("callback_on_step_end", "detection_key", "key", "watermark"))
        for call in pipeline.calls
    )

    asset_path, checksum_path = runner._destinations(sink.resolve(), _EXACT)
    assert sorted(path.name for path in asset_path.parent.iterdir()) == [
        runner.ASSET_FILENAME,
        f"{runner.ASSET_FILENAME}.sha256",
    ]
    raw = asset_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert checksum_path.read_text(encoding="ascii") == f"{digest}  {runner.ASSET_FILENAME}\n"
    loaded_asset = v4.load_whitening_asset(asset_path)
    payload = loaded_asset.payload
    assert set(payload) == {
        "schema_version",
        "observation_contract_id",
        "whitening_shape",
        "whitening_order",
        "whitening_words_be_hex",
        "fit_sample_count",
        "producer_exact",
    }
    assert payload["producer_exact"] == _EXACT
    assert len(payload["whitening_words_be_hex"]) == 96

    receipt_output = capsys.readouterr().out
    assert receipt_output.count("CEGWM_CONTENT_V4_WHITENING_RECEIPT ") == 1
    assert len(receipt_output.encode("utf-8")) < 512
    assert "fixture-secret-token" not in receipt_output
    assert manifest.entries[0].prompt not in receipt_output
    receipt = json.loads(receipt_output.split(" ", 1)[1])
    assert receipt == {
        "asset_sha256": digest,
        "producer_exact": _EXACT,
        "unit_count": 32,
    }

    before = len(pipeline.calls)
    monkeypatch.setenv(runner.TOKEN_ENV, "unused-secret-token")
    with pytest.raises(FileExistsError, match="create-only"):
        runner.execute(_args(sink))
    assert len(pipeline.calls) == before


@pytest.mark.integration
def test_strict_final_rgb_observation_materialization_and_shape_failure() -> None:
    processor = _Processor()
    vae = _VAE()
    image = Image.new("RGB", (512, 512), color=(10, 20, 30))
    observation = materialize_clean_fit_observation(image, processor, vae)
    assert observation.shape == torch.Size(v4.OBSERVATION_SHAPE)
    assert observation.dtype == torch.float32
    assert observation.device.type == "cpu"
    assert tuple(observation.stride()) == v4.OBSERVATION_STRIDE

    class WrongShapeVAE(_VAE):
        def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
            value = super().encode(pixels).latent_dist.mode()[:, :8]
            return SimpleNamespace(latent_dist=_Distribution(value))

    with pytest.raises(ValueError, match="1x16x64x64"):
        materialize_clean_fit_observation(image, processor, WrongShapeVAE())
    with pytest.raises(ValueError, match="already be RGB"):
        materialize_clean_fit_observation(
            Image.new("RGBA", (512, 512)), processor, vae
        )


@pytest.mark.integration
def test_publication_rejects_existing_sidecar_without_partial_asset(tmp_path: Path) -> None:
    asset = tmp_path / runner.ASSET_FILENAME
    sidecar = tmp_path / f"{runner.ASSET_FILENAME}.sha256"
    sidecar.write_text("occupied", encoding="ascii")
    raw = b"{}"
    with pytest.raises(FileExistsError, match="create-only"):
        runner._publish_create_only(asset, sidecar, raw)
    assert not asset.exists()
    assert sidecar.read_text(encoding="ascii") == "occupied"


@pytest.mark.integration
def test_expected_exact_fails_closed() -> None:
    with pytest.raises(ValueError, match="lowercase 40-character"):
        runner._git_exact(_REPO_ROOT, "not-an-exact")
