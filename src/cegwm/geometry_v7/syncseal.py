"""Official SyncSeal TorchScript adapter at the frozen final-RGB boundary."""

from __future__ import annotations

import math
import shutil
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np
import torch
from PIL import Image

from cegwm.geometry_v7.contracts import (
    GeometryEstimate,
    PUBLIC_IMAGE_HEIGHT,
    PUBLIC_IMAGE_WIDTH,
    estimate_geometry,
    syncseal_raw_to_public_normalized,
)
from cegwm.runtime.observation import require_ordinary_rgb_image


SYNCSEAL_TORCHSCRIPT_URL = (
    "https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt"
)
SYNCSEAL_OFFICIAL_BASE_ALPHA = 0.20


def download_official_syncseal_torchscript(
    destination: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = urllib.request.urlopen,
) -> Path:
    """Create, but never overwrite, the official TorchScript checkpoint.

    The producer publishes no frozen SHA-256 in the adopted P0 source.  The URL
    is therefore recorded, while model-byte verification remains unclaimed.
    """

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with opener(SYNCSEAL_TORCHSCRIPT_URL) as source, path.open("xb") as sink:
            created = True
            shutil.copyfileobj(source, sink)
    except BaseException:
        if created and path.exists():
            path.unlink()
        raise
    return path


def _public_rgb(image: Any) -> Image.Image:
    rgb = require_ordinary_rgb_image(image)
    if rgb.size != (PUBLIC_IMAGE_WIDTH, PUBLIC_IMAGE_HEIGHT):
        raise ValueError("Geometry-V7 public input must be raw 512x512 RGB")
    return rgb


def _to_tensor(image: Any, device: torch.device) -> torch.Tensor:
    rgb = _public_rgb(image)
    array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device).float() / 255.0


def _to_rgb(tensor: torch.Tensor) -> Image.Image:
    if tensor.shape != (1, 3, PUBLIC_IMAGE_HEIGHT, PUBLIC_IMAGE_WIDTH):
        raise ValueError("SyncSeal RGB output must be 1x3x512x512")
    if not tensor.dtype.is_floating_point or not bool(torch.isfinite(tensor).all()):
        raise ValueError("SyncSeal RGB output must be finite floating point")
    if bool(torch.any(tensor < 0.0)) or bool(torch.any(tensor > 1.0)):
        raise ValueError("SyncSeal RGB output must remain in [0,1]")
    pixels = tensor.detach().cpu().mul(255.0).round().to(torch.uint8)
    return Image.fromarray(pixels[0].permute(1, 2, 0).numpy(), mode="RGB")


class SyncSealTorchScript:
    """Thin adapter over the official ``embed`` and ``detect`` methods."""

    def __init__(self, model: Any, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        embed = getattr(model, "embed", None)
        detect = getattr(model, "detect", None)
        if not callable(embed) or not callable(detect):
            raise TypeError("SyncSeal TorchScript must expose embed and detect")
        self.model = model.to(self.device).eval()

    @classmethod
    def from_file(
        cls, checkpoint: str | Path, device: str | torch.device = "cpu"
    ) -> "SyncSealTorchScript":
        model = torch.jit.load(str(Path(checkpoint)), map_location=torch.device(device))
        return cls(model, device)

    def embed_final_rgb(self, image: Any, residual_strength_multiplier: float) -> Image.Image:
        """Apply SyncSeal only after the content chain has produced final RGB.

        Official ``imgs_w`` already contains base alpha 0.20.  Multiplier ``m``
        is exactly ``clamp(I + m * (imgs_w-I), 0, 1)``; alpha is not reapplied.
        """

        if isinstance(residual_strength_multiplier, bool) or not isinstance(
            residual_strength_multiplier, (int, float)
        ):
            raise TypeError("SyncSeal residual strength multiplier must be real")
        multiplier = float(residual_strength_multiplier)
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("SyncSeal residual strength multiplier must be finite and nonnegative")
        current = _to_tensor(image, self.device)
        with torch.no_grad():
            output = self.model.embed(current)
        if not isinstance(output, dict) or set(output) != {"preds_w", "imgs_w"}:
            raise TypeError("SyncSeal embed must return exactly preds_w and imgs_w")
        predicted_residual = output["preds_w"]
        embedded = output["imgs_w"]
        if not isinstance(embedded, torch.Tensor) or embedded.shape != current.shape:
            raise ValueError("SyncSeal imgs_w must have shape 1x3x512x512")
        if not embedded.dtype.is_floating_point or not bool(torch.isfinite(embedded).all()):
            raise ValueError("SyncSeal imgs_w must be finite floating point")
        expected_residual_shape = (1, 1, PUBLIC_IMAGE_HEIGHT, PUBLIC_IMAGE_WIDTH)
        if (
            not isinstance(predicted_residual, torch.Tensor)
            or predicted_residual.shape != expected_residual_shape
        ):
            raise ValueError("SyncSeal preds_w must have shape 1x1x512x512")
        if not predicted_residual.dtype.is_floating_point or not bool(
            torch.isfinite(predicted_residual).all()
        ):
            raise ValueError("SyncSeal preds_w must be finite floating point")
        scaled = torch.clamp(current + multiplier * (embedded.to(current) - current), 0.0, 1.0)
        return _to_rgb(scaled)

    def detect_geometry(self, image: Any) -> GeometryEstimate:
        """Return coordinates and observability only; never a content decision."""

        try:
            current = _to_tensor(image, self.device)
            with torch.no_grad():
                output = self.model.detect(current)
            if not isinstance(output, dict) or set(output) != {"preds", "preds_pts"}:
                raise TypeError("SyncSeal detect must return exactly preds and preds_pts")
            preds = output["preds"]
            points = output["preds_pts"]
            if not isinstance(preds, torch.Tensor) or preds.shape != (1, 9):
                raise ValueError("SyncSeal preds must have shape 1x9")
            if not isinstance(points, torch.Tensor) or points.shape != (1, 8):
                raise ValueError("SyncSeal preds_pts must have shape 1x8")
            if not preds.dtype.is_floating_point or not points.dtype.is_floating_point:
                raise TypeError("SyncSeal detection outputs must be floating point")
            if not bool(torch.isfinite(preds).all()) or not bool(torch.isfinite(points).all()):
                raise ValueError("SyncSeal detection outputs must be finite")
            if not torch.equal(preds[:, 1:].to(points), points):
                raise ValueError("SyncSeal preds_pts must equal the final 8 raw outputs")
            raw_corners = points[0].detach().cpu().tolist()
            public_corners = syncseal_raw_to_public_normalized(raw_corners)
            return estimate_geometry(
                float(preds[0, 0]),
                public_corners,
                raw_syncseal_corners=raw_corners,
            )
        except Exception as error:
            return GeometryEstimate.error_record(error)


__all__ = [
    "SYNCSEAL_OFFICIAL_BASE_ALPHA",
    "SYNCSEAL_TORCHSCRIPT_URL",
    "SyncSealTorchScript",
    "download_official_syncseal_torchscript",
]
