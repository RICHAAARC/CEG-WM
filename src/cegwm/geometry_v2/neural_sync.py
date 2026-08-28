"""Small real PyTorch networks for the frozen Geometry-V2 N0 protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 128
SYNC_CODE_LENGTH = 64
SYNC_GRID_SIZE = 8
MAX_RESIDUAL = 4.0 / 255.0
SYNC_AMPLITUDE = 2.0 / 255.0


@dataclass(frozen=True, slots=True)
class EmbeddedBatch:
    image: Tensor
    residual: Tensor


@dataclass(frozen=True, slots=True)
class CornerPrediction:
    corners: Tensor
    confidence: Tensor
    support: Tensor


class KeyedResidualEmbedder(nn.Module):
    """Write a bounded RGB residual conditioned on one per-sample keyed code."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, 1),
        )

    @staticmethod
    def target_map(code: Tensor, *, height: int, width: int) -> Tensor:
        if code.ndim != 2 or code.shape[1] != SYNC_CODE_LENGTH:
            raise ValueError("sync code must have shape [batch, 64]")
        if not torch.isfinite(code).all() or torch.any(torch.abs(code) != 1):
            raise ValueError("sync code must be finite and bipolar")
        grid = code.reshape(code.shape[0], 1, SYNC_GRID_SIZE, SYNC_GRID_SIZE)
        return F.interpolate(grid, size=(height, width), mode="nearest")

    def forward(self, rgb: Tensor, code: Tensor) -> EmbeddedBatch:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB batch must have shape [batch, 3, height, width]")
        if rgb.shape[2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("N0 RGB resolution must be 128x128")
        if not torch.isfinite(rgb).all() or torch.any(rgb < 0) or torch.any(rgb > 1):
            raise ValueError("RGB input must be finite in [0, 1]")
        target = self.target_map(code, height=rgb.shape[2], width=rgb.shape[3])
        proposed = torch.tanh(self.body(torch.cat((rgb, target), dim=1))) * MAX_RESIDUAL
        image = torch.clamp(rgb + proposed, 0.0, 1.0)
        residual = image - rgb
        return EmbeddedBatch(image=image, residual=residual)


class BlindCornerExtractor(nn.Module):
    """Observe only attacked RGB and predict ordered normalized corners."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 9)

    def forward(self, attacked_rgb: Tensor) -> CornerPrediction:
        if attacked_rgb.ndim != 4 or attacked_rgb.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("extractor input must be attacked RGB [batch,3,128,128]")
        if not torch.isfinite(attacked_rgb).all():
            raise ValueError("extractor input must be finite")
        values = self.head(self.features(attacked_rgb).flatten(1))
        # The frozen geometry contracts admit crop/similarity corners in
        # [-0.25, 1.25].  Parameterize that entire interval so the network can
        # represent out-of-frame source corners without clipping them to RGB.
        corners = (1.5 * torch.sigmoid(values[:, :8]) - 0.25).reshape(-1, 4, 2)
        confidence = torch.sigmoid(values[:, 8])
        # N0 support means that one complete finite four-corner estimate exists.
        support = torch.ones_like(confidence)
        return CornerPrediction(corners=corners, confidence=confidence, support=support)


def n0_joint_loss(
    prediction: CornerPrediction,
    truth_corners: Tensor,
    embedded: EmbeddedBatch,
    code: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Frozen N0 objective: corner + .25 sync reconstruction + .05 residual L2."""

    if truth_corners.shape != prediction.corners.shape:
        raise ValueError("corner truth shape differs")
    corner = F.smooth_l1_loss(prediction.corners, truth_corners)
    pooled = F.adaptive_avg_pool2d(embedded.residual.mean(dim=1, keepdim=True), (8, 8)).flatten(1)
    sync = F.mse_loss(pooled, code * SYNC_AMPLITUDE)
    residual = embedded.residual.square().mean()
    total = corner + 0.25 * sync + 0.05 * residual
    return total, {"corner": corner, "sync_reconstruction": sync, "residual_l2": residual}


__all__ = [
    "BlindCornerExtractor",
    "CornerPrediction",
    "EmbeddedBatch",
    "IMAGE_SIZE",
    "KeyedResidualEmbedder",
    "MAX_RESIDUAL",
    "SYNC_AMPLITUDE",
    "SYNC_CODE_LENGTH",
    "n0_joint_loss",
]
