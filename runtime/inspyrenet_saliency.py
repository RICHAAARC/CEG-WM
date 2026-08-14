"""Frozen InSPyReNet public saliency observation runtime."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
from struct import pack
from typing import BinaryIO, Literal

import numpy as np
from PIL import Image
import torch

from main import SaliencyProbabilityObservation


INSPYRENET_CHECKPOINT_ASSET_IDENTITY = "inspyrenet_saliency_checkpoint"
INSPYRENET_CHECKPOINT_ASSET_BASENAME = "ckpt_base.pth"
INSPYRENET_CHECKPOINT_SHA256 = (
    "0a6fe2a73ab0532d6d0b8d82849a9760a226df719e3063d09b4149ece6f80fcd"
)
INSPYRENET_CHECKPOINT_SIZE = 367_520_613
INSPYRENET_PREPROCESS_SPATIAL_SIZE = (1024, 1024)

SaliencyObservationRole = Literal[
    "embed_nonterminal_content_write_callback_latent_rgb8",
    "detect_public_rgb8",
]

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class InspyrenetSaliencyRuntimeError(RuntimeError):
    """The frozen saliency asset, input, model, or public output is invalid."""


def _open_verified_checkpoint(
    checkpoint_path: Path,
    *,
    checkpoint_asset_identity: str,
    checkpoint_asset_basename: str,
) -> BinaryIO:
    if not isinstance(checkpoint_path, Path):
        raise InspyrenetSaliencyRuntimeError("checkpoint path must be explicit")
    if checkpoint_asset_identity != INSPYRENET_CHECKPOINT_ASSET_IDENTITY:
        raise InspyrenetSaliencyRuntimeError("checkpoint asset identity drifted")
    if (
        checkpoint_asset_basename != INSPYRENET_CHECKPOINT_ASSET_BASENAME
        or checkpoint_path.name != INSPYRENET_CHECKPOINT_ASSET_BASENAME
    ):
        raise InspyrenetSaliencyRuntimeError("checkpoint asset basename drifted")
    try:
        checkpoint_stat = checkpoint_path.lstat()
    except OSError:
        raise InspyrenetSaliencyRuntimeError("checkpoint asset is unavailable") from None
    if not stat.S_ISREG(checkpoint_stat.st_mode):
        raise InspyrenetSaliencyRuntimeError(
            "checkpoint asset must be a regular non-symlink file"
        )
    if checkpoint_stat.st_size != INSPYRENET_CHECKPOINT_SIZE:
        raise InspyrenetSaliencyRuntimeError("checkpoint asset size drifted")

    descriptor: int | None = None
    stream: BinaryIO | None = None
    try:
        descriptor = os.open(checkpoint_path, os.O_RDONLY | os.O_NOFOLLOW)
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != checkpoint_stat.st_dev
            or opened_stat.st_ino != checkpoint_stat.st_ino
            or opened_stat.st_size != checkpoint_stat.st_size
        ):
            raise InspyrenetSaliencyRuntimeError("checkpoint asset changed during verification")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
        digest = sha256()
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
        if digest.hexdigest() != INSPYRENET_CHECKPOINT_SHA256:
            raise InspyrenetSaliencyRuntimeError("checkpoint asset digest drifted")
        stream.seek(0)
        return stream
    except InspyrenetSaliencyRuntimeError:
        if stream is not None:
            stream.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if stream is not None:
            stream.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise InspyrenetSaliencyRuntimeError("checkpoint asset verification failed") from None


def _construct_inspyrenet_model() -> torch.nn.Module:
    from runtime._vendor.transparent_background.InSPyReNet import InSPyReNet_SwinB

    return InSPyReNet_SwinB(
        depth=64,
        pretrained=False,
        base_size=[384, 384],
    )


def _rgb8_digest(image: Image.Image) -> str:
    width, height = image.size
    return sha256(
        b"ceg-wm-public-rgb8\x00"
        + pack(">II", width, height)
        + image.tobytes()
    ).hexdigest()


def _preprocess_rgb8(image: Image.Image) -> torch.Tensor:
    if type(image) is not Image.Image or image.mode != "RGB":
        raise InspyrenetSaliencyRuntimeError("saliency input must be a PIL RGB8 image")
    if image.width <= 0 or image.height <= 0:
        raise InspyrenetSaliencyRuntimeError("saliency input dimensions are invalid")
    resized = image.resize(
        INSPYRENET_PREPROCESS_SPATIAL_SIZE,
        resample=Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.uint8).astype(np.float32)
    pixels = pixels / np.float32(255.0)
    pixels = (pixels - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw).unsqueeze(0)


def _select_finest_raw_saliency(output: object) -> torch.Tensor:
    if type(output) is not dict or "saliency" not in output:
        raise InspyrenetSaliencyRuntimeError("forward_inspyre output identity drifted")
    saliency = output["saliency"]
    if type(saliency) is not list or len(saliency) != 4:
        raise InspyrenetSaliencyRuntimeError("saliency pyramid identity drifted")
    raw_finest = saliency[-1]
    if (
        type(raw_finest) is not torch.Tensor
        or raw_finest.ndim != 4
        or tuple(raw_finest.shape[:2]) != (1, 1)
        or raw_finest.shape[2] <= 0
        or raw_finest.shape[3] <= 0
        or not raw_finest.is_floating_point()
        or not torch.isfinite(raw_finest).all().item()
    ):
        raise InspyrenetSaliencyRuntimeError("finest raw saliency logit is invalid")
    return raw_finest


class InspyrenetSaliencyRuntime:
    """Own the frozen model while returning only public probability observations."""

    __slots__ = ("_device", "_model")

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_asset_identity: str,
        checkpoint_asset_basename: str,
        selected_device: str,
    ) -> None:
        try:
            device = torch.device(selected_device)
        except (RuntimeError, TypeError, ValueError):
            raise InspyrenetSaliencyRuntimeError("saliency runtime device is invalid") from None
        checkpoint_stream = _open_verified_checkpoint(
            checkpoint_path,
            checkpoint_asset_identity=checkpoint_asset_identity,
            checkpoint_asset_basename=checkpoint_asset_basename,
        )
        try:
            model = _construct_inspyrenet_model()
            state_dict = torch.load(
                checkpoint_stream,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise InspyrenetSaliencyRuntimeError(
                "strict InSPyReNet model initialization failed"
            ) from exc
        finally:
            checkpoint_stream.close()
        self._device = device
        self._model = model

    def observe(
        self,
        image: Image.Image,
        *,
        observation_role: SaliencyObservationRole,
    ) -> SaliencyProbabilityObservation:
        if observation_role not in {
            "embed_nonterminal_content_write_callback_latent_rgb8",
            "detect_public_rgb8",
        }:
            raise InspyrenetSaliencyRuntimeError("saliency observation role is invalid")
        image_digest = _rgb8_digest(image) if type(image) is Image.Image else None
        model_input = _preprocess_rgb8(image).to(self._device)
        try:
            with torch.no_grad():
                output = self._model.forward_inspyre(model_input)
                raw_finest = _select_finest_raw_saliency(output)
                probability = torch.sigmoid(raw_finest)
        except InspyrenetSaliencyRuntimeError:
            raise
        except Exception as exc:
            raise InspyrenetSaliencyRuntimeError(
                "direct forward_inspyre execution failed"
            ) from exc
        probability_cpu = probability.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        if not torch.isfinite(probability_cpu).all().item():
            raise InspyrenetSaliencyRuntimeError("saliency probability is nonfinite")
        return SaliencyProbabilityObservation(
            values=tuple(float(value) for value in probability_cpu.flatten().tolist()),
            spatial_shape=(probability_cpu.shape[2], probability_cpu.shape[3]),
            observation_role=observation_role,
            input_image_digest=image_digest,
        )


__all__ = [
    "INSPYRENET_CHECKPOINT_ASSET_BASENAME",
    "INSPYRENET_CHECKPOINT_ASSET_IDENTITY",
    "INSPYRENET_CHECKPOINT_SHA256",
    "INSPYRENET_CHECKPOINT_SIZE",
    "INSPYRENET_PREPROCESS_SPATIAL_SIZE",
    "InspyrenetSaliencyRuntime",
    "InspyrenetSaliencyRuntimeError",
]
