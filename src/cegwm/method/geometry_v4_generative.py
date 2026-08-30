"""Keyed late-latent anchor writer and RGB-only anchor observability."""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from cegwm.protocol.geometry_v4 import derive_geometry_v4_key
from cegwm.protocol.geometry_v4_generative import ENERGY_SHARES, LUMA_PEAK_CAP, LUMA_RMS_CAP
from cegwm.shared.keys import normalize_detection_key

_REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_LATENT_AMPLITUDE = 0.001


@dataclass(frozen=True, slots=True)
class RGBObservability:
    psnr: float
    ssim: float
    luma_rms: float
    luma_peak: float
    correct_key_anchor: float
    wrong_key_anchor: float
    content_score_drift: float

    @property
    def passed(self) -> bool:
        return self.correct_key_anchor > self.wrong_key_anchor and self.psnr > 40.0 and self.ssim > .98 and self.luma_rms <= LUMA_RMS_CAP and self.luma_peak <= LUMA_PEAK_CAP and self.content_score_drift < .05


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hmac.new(key, label, hashlib.sha256).digest()[:8], "big")


def write_final_latent_anchor(latents: torch.Tensor, detection_key: str | bytes | bytearray | memoryview) -> torch.Tensor:
    """Return one fixed-amplitude keyed residual at the sole final callback state."""
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or not latents.dtype.is_floating_point or not bool(torch.isfinite(latents).all()):
        raise ValueError("final callback latents must be finite floating NCHW")
    root = normalize_detection_key(detection_key)
    key = derive_geometry_v4_key(root)
    generator = torch.Generator(device="cpu").manual_seed(_seed(key, b"g0-g1-final-latent-anchor"))
    global_field = torch.randn(latents.shape, generator=generator, dtype=torch.float32, device="cpu")
    local_field = torch.randn(latents.shape, generator=generator, dtype=torch.float32, device="cpu")
    global_field -= global_field.mean(dim=(-2, -1), keepdim=True)
    local_field -= local_field.mean(dim=(-2, -1), keepdim=True)
    global_field /= torch.linalg.vector_norm(global_field).clamp_min(torch.finfo(torch.float32).eps)
    local_field -= (local_field * global_field).sum() * global_field
    local_field /= torch.linalg.vector_norm(local_field).clamp_min(torch.finfo(torch.float32).eps)
    residual = math.sqrt(ENERGY_SHARES[0]) * global_field + math.sqrt(ENERGY_SHARES[1]) * local_field
    return latents + residual.to(device=latents.device, dtype=latents.dtype) * _LATENT_AMPLITUDE


def rgb_only_anchor_score(rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> float:
    """Read only the current RGB and a normalized key; no truth or source enters."""
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all() or min(image.shape[:2]) < 16:
        raise ValueError("detector requires finite current ordinary RGB")
    if image.max() > 1.0 or image.min() < 0.0:
        raise ValueError("detector requires RGB normalized to [0,1]")
    key = derive_geometry_v4_key(normalize_detection_key(detection_key))
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    luma = image @ _REC709
    luma = luma - luma.mean()
    score = 0.0
    for cycle, orientation in ((8, 0), (16, 45), (32, 90), (16, 135)):
        phase = (_seed(key, f"{cycle}:{orientation}".encode()) / 2**64) * 2 * math.pi
        angle = math.radians(orientation)
        carrier = np.cos(2 * math.pi * cycle * (xx * math.cos(angle) / w + yy * math.sin(angle) / h) + phase)
        score += abs(float((luma * carrier).mean()))
    return score / 4.0


def measure_final_rgb(clean_rgb: np.ndarray, marked_rgb: np.ndarray, detection_key: object, wrong_key: object, content_detector: Callable[[np.ndarray, bytes], float]) -> RGBObservability:
    clean, marked = np.asarray(clean_rgb, dtype=np.float64), np.asarray(marked_rgb, dtype=np.float64)
    if clean.shape != marked.shape or clean.ndim != 3 or clean.shape[2] != 3 or not np.isfinite(clean).all() or not np.isfinite(marked).all():
        raise ValueError("final RGB pair must be equal-shaped finite RGB")
    if normalize_detection_key(detection_key) == normalize_detection_key(wrong_key):
        raise ValueError("correct and wrong detection keys must differ after normalization")
    delta_luma = (marked - clean) @ _REC709
    mse = float(np.mean((marked - clean) ** 2))
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    mx, my = float(clean.mean()), float(marked.mean())
    vx, vy = float(clean.var()), float(marked.var())
    cov = float(((clean - mx) * (marked - my)).mean())
    ssim = ((2 * mx * my + 1e-4) * (2 * cov + 9e-4)) / ((mx * mx + my * my + 1e-4) * (vx + vy + 9e-4))
    root = normalize_detection_key(detection_key)
    drift = abs(float(content_detector(clean, root)) - float(content_detector(marked, root)))
    return RGBObservability(psnr, ssim, float(np.sqrt(np.mean(delta_luma**2))), float(np.max(np.abs(delta_luma))), rgb_only_anchor_score(marked, detection_key), rgb_only_anchor_score(marked, wrong_key), drift)
