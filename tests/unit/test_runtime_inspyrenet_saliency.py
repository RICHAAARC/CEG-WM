"""Tests for the frozen public InSPyReNet saliency observation runtime."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from main import SaliencyProbabilityObservation, rgb8_image_digest
from runtime import (
    INSPYRENET_CHECKPOINT_ASSET_BASENAME,
    INSPYRENET_CHECKPOINT_ASSET_IDENTITY,
    INSPYRENET_CHECKPOINT_SHA256,
    INSPYRENET_CHECKPOINT_SIZE,
    INSPYRENET_PREPROCESS_SPATIAL_SIZE,
    InspyrenetSaliencyRuntime,
    InspyrenetSaliencyRuntimeError,
)
import runtime.inspyrenet_saliency as saliency_runtime


pytestmark = pytest.mark.unit


class _ControlledSaliencyModel(torch.nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.checkpoint_marker = torch.nn.Parameter(torch.tensor(0.0))
        self.output = output
        self.forward_inputs: list[torch.Tensor] = []

    def forward_inspyre(self, model_input: torch.Tensor) -> object:
        self.forward_inputs.append(model_input.detach().cpu().clone())
        return self.output


def _valid_output() -> dict[str, object]:
    raw_finest = torch.tensor(
        [[[[0.0, 2.0], [-2.0, 1.0]]]],
        dtype=torch.float32,
    )
    return {
        "saliency": [
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            raw_finest,
        ],
        "laplacian": [],
    }


def _controlled_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: _ControlledSaliencyModel,
) -> Path:
    checkpoint_path = tmp_path / INSPYRENET_CHECKPOINT_ASSET_BASENAME
    torch.save(model.state_dict(), checkpoint_path)
    payload = checkpoint_path.read_bytes()
    monkeypatch.setattr(saliency_runtime, "INSPYRENET_CHECKPOINT_SIZE", len(payload))
    monkeypatch.setattr(
        saliency_runtime,
        "INSPYRENET_CHECKPOINT_SHA256",
        sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        saliency_runtime,
        "_construct_inspyrenet_model",
        lambda: model,
    )
    return checkpoint_path


def _runtime(
    checkpoint_path: Path,
) -> InspyrenetSaliencyRuntime:
    return InspyrenetSaliencyRuntime(
        checkpoint_path=checkpoint_path,
        checkpoint_asset_identity=INSPYRENET_CHECKPOINT_ASSET_IDENTITY,
        checkpoint_asset_basename=INSPYRENET_CHECKPOINT_ASSET_BASENAME,
        selected_device="cpu",
    )


def test_public_saliency_runtime_applies_frozen_preprocess_selector_and_single_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _valid_output()
    model = _ControlledSaliencyModel(output)
    checkpoint_path = _controlled_checkpoint(tmp_path, monkeypatch, model)
    sigmoid_calls = 0
    original_sigmoid = torch.sigmoid

    def counted_sigmoid(value: torch.Tensor) -> torch.Tensor:
        nonlocal sigmoid_calls
        sigmoid_calls += 1
        return original_sigmoid(value)

    monkeypatch.setattr(saliency_runtime.torch, "sigmoid", counted_sigmoid)
    runtime = _runtime(checkpoint_path)
    image = torch.tensor(
        [
            [
                [[255, 0], [1, 2]],
                [[0, 255], [3, 4]],
                [[127, 128], [5, 6]],
            ]
        ],
        dtype=torch.uint8,
    )

    observation = runtime.observe(image, observation_role="detect_public_rgb8")

    assert type(observation) is SaliencyProbabilityObservation
    assert observation.spatial_shape == (2, 2)
    raw_finest = output["saliency"][-1]  # type: ignore[index]
    assert type(raw_finest) is torch.Tensor
    expected = tuple(float(value) for value in original_sigmoid(raw_finest).flatten())
    twice_applied = tuple(
        float(value)
        for value in original_sigmoid(original_sigmoid(raw_finest)).flatten()
    )
    assert observation.values == expected
    assert observation.values != twice_applied
    assert sigmoid_calls == 1
    assert observation.source_repository == "plemeri/transparent-background"
    assert observation.source_revision == (
        "f0fa91701a98cfc8e955c554e84522f365ec6da3"
    )
    assert observation.checkpoint_repository == "plemeri/InSPyReNet"
    assert observation.checkpoint_revision == (
        "d94c2baaa4d023ab018c6f97be6ef37548e3bd1f"
    )
    assert observation.checkpoint_sha256 == INSPYRENET_CHECKPOINT_SHA256
    assert observation.checkpoint_size == INSPYRENET_CHECKPOINT_SIZE
    assert observation.preprocess_identity == (
        "rgb_static_1024x1024_imagenet_mean_std_float32"
    )
    assert observation.forward_identity == (
        "direct_forward_inspyre_raw_finest_saliency_logit"
    )
    assert observation.sigmoid_identity == "torch_sigmoid_exactly_once"
    assert observation.observation_role == "detect_public_rgb8"
    assert observation.input_image_digest == rgb8_image_digest(image)
    pil_image = saliency_runtime._rgb8_tensor_to_pil(image)
    assert pil_image.mode == "RGB"
    assert pil_image.size == (2, 2)
    assert pil_image.tobytes() == bytes(
        (255, 0, 127, 0, 255, 128, 1, 3, 5, 2, 4, 6)
    )
    assert not hasattr(observation, "checkpoint_path")
    assert not hasattr(observation, "model")
    assert model.training is False
    assert len(model.forward_inputs) == 1
    model_input = model.forward_inputs[0]
    assert model_input.shape == (1, 3, *INSPYRENET_PREPROCESS_SPATIAL_SIZE)
    assert model_input.dtype is torch.float32
    expected_pixel = np.asarray([255, 0, 127], dtype=np.uint8).astype(np.float32)
    expected_pixel = expected_pixel / np.float32(255.0)
    expected_pixel = (
        expected_pixel - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    expected_channels = torch.from_numpy(expected_pixel)
    assert torch.equal(model_input[0, :, 0, 0], expected_channels)


@pytest.mark.parametrize(
    "output",
    (
        [],
        {},
        {"saliency": [torch.zeros((1, 1, 1, 1))] * 3},
        {"saliency": [torch.zeros((1, 1, 1, 1))] * 3 + [torch.zeros((1, 2, 2))]},
        {
            "saliency": [torch.zeros((1, 1, 1, 1))] * 3
            + [torch.tensor([[[[float("nan")]]]])]
        },
    ),
)
def test_public_saliency_runtime_rejects_forward_output_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: object,
) -> None:
    model = _ControlledSaliencyModel(output)
    checkpoint_path = _controlled_checkpoint(tmp_path, monkeypatch, model)
    runtime = _runtime(checkpoint_path)
    image = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)

    with pytest.raises(InspyrenetSaliencyRuntimeError):
        runtime.observe(image, observation_role="detect_public_rgb8")


def test_checkpoint_validation_precedes_model_construction_and_rejects_asset_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_calls = 0

    def record_construction() -> _ControlledSaliencyModel:
        nonlocal construction_calls
        construction_calls += 1
        return _ControlledSaliencyModel(_valid_output())

    monkeypatch.setattr(
        saliency_runtime,
        "_construct_inspyrenet_model",
        record_construction,
    )
    missing = tmp_path / INSPYRENET_CHECKPOINT_ASSET_BASENAME
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="unavailable"):
        _runtime(missing)
    assert construction_calls == 0

    missing.mkdir()
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="regular non-symlink"):
        _runtime(missing)
    assert construction_calls == 0

    missing.rmdir()
    target = tmp_path / "checkpoint_payload"
    target.write_bytes(b"controlled")
    missing.symlink_to(target)
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="regular non-symlink"):
        _runtime(missing)
    assert construction_calls == 0


def test_checkpoint_validation_rejects_size_digest_and_strict_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ControlledSaliencyModel(_valid_output())
    checkpoint_path = tmp_path / INSPYRENET_CHECKPOINT_ASSET_BASENAME
    checkpoint_path.write_bytes(b"wrong-size")
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="size drifted"):
        _runtime(checkpoint_path)

    monkeypatch.setattr(
        saliency_runtime,
        "INSPYRENET_CHECKPOINT_SIZE",
        checkpoint_path.stat().st_size,
    )
    monkeypatch.setattr(saliency_runtime, "INSPYRENET_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="digest drifted"):
        _runtime(checkpoint_path)

    torch.save({"unexpected_parameter": torch.tensor(1.0)}, checkpoint_path)
    payload = checkpoint_path.read_bytes()
    monkeypatch.setattr(saliency_runtime, "INSPYRENET_CHECKPOINT_SIZE", len(payload))
    monkeypatch.setattr(
        saliency_runtime,
        "INSPYRENET_CHECKPOINT_SHA256",
        sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        saliency_runtime,
        "_construct_inspyrenet_model",
        lambda: model,
    )
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="strict"):
        _runtime(checkpoint_path)


def test_checkpoint_semantic_identity_and_rgb8_role_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ControlledSaliencyModel(_valid_output())
    checkpoint_path = _controlled_checkpoint(tmp_path, monkeypatch, model)
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="asset identity"):
        InspyrenetSaliencyRuntime(
            checkpoint_path=checkpoint_path,
            checkpoint_asset_identity="different_saliency_checkpoint",
            checkpoint_asset_basename=INSPYRENET_CHECKPOINT_ASSET_BASENAME,
            selected_device="cpu",
        )
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="basename"):
        InspyrenetSaliencyRuntime(
            checkpoint_path=checkpoint_path,
            checkpoint_asset_identity=INSPYRENET_CHECKPOINT_ASSET_IDENTITY,
            checkpoint_asset_basename="different_checkpoint.pth",
            selected_device="cpu",
        )

    runtime = _runtime(checkpoint_path)
    with pytest.raises(InspyrenetSaliencyRuntimeError, match="role"):
        runtime.observe(
            torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
            observation_role="unsupported_role",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "image",
    (
        object(),
        Image.new("RGB", (2, 2)),
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        torch.zeros((3, 2, 2), dtype=torch.uint8),
        torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
        torch.zeros((1, 1, 2, 2), dtype=torch.uint8),
        torch.zeros((1, 3, 1, 2), dtype=torch.uint8),
        torch.zeros((1, 3, 2, 1), dtype=torch.uint8),
    ),
)
def test_public_saliency_runtime_rejects_nonordinary_rgb8_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: object,
) -> None:
    model = _ControlledSaliencyModel(_valid_output())
    checkpoint_path = _controlled_checkpoint(tmp_path, monkeypatch, model)
    runtime = _runtime(checkpoint_path)

    with pytest.raises(InspyrenetSaliencyRuntimeError, match="ordinary RGB8"):
        runtime.observe(
            image,  # type: ignore[arg-type]
            observation_role="detect_public_rgb8",
        )
    assert model.forward_inputs == []


@pytest.mark.integration
@pytest.mark.slow
def test_real_checkpoint_executes_frozen_public_saliency_observation() -> None:
    checkpoint_value = os.environ.get("CEG_WM_INSPYRENET_CHECKPOINT_PATH")
    if checkpoint_value is None:
        pytest.skip("real InSPyReNet checkpoint was not explicitly provided")
    runtime = InspyrenetSaliencyRuntime(
        checkpoint_path=Path(checkpoint_value),
        checkpoint_asset_identity=INSPYRENET_CHECKPOINT_ASSET_IDENTITY,
        checkpoint_asset_basename=INSPYRENET_CHECKPOINT_ASSET_BASENAME,
        selected_device="cuda:0",
    )
    observation = runtime.observe(
        torch.full((1, 3, 32, 32), 127, dtype=torch.uint8),
        observation_role="detect_public_rgb8",
    )

    assert type(observation) is SaliencyProbabilityObservation
    assert observation.spatial_shape == INSPYRENET_PREPROCESS_SPATIAL_SIZE
