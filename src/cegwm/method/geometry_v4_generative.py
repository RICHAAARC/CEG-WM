"""Keyed late-latent anchor writer and RGB-only anchor observability."""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from cegwm.protocol.geometry_v4 import derive_geometry_v4_key
from cegwm.protocol.geometry_v4_generative import ENERGY_SHARES, LUMA_PEAK_CAP, LUMA_RMS_CAP
from cegwm.shared.keys import normalize_detection_key
from cegwm.method.content_whitening import score_content_whitened_lf_image
from cegwm.method.hf import score_hf_image
from cegwm.method.content_weighted_joint import LFHFScorePair, WeightedJointAsset, load_calibration_asset, weighted_joint_score

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


@dataclass(frozen=True, slots=True)
class FrozenWeightedJointContentAdapter:
    """The unchanged content detector: current RGB plus normalized key only."""

    lf_public_assets: Any
    hf_public_assets: Any
    calibration_asset: WeightedJointAsset
    calibration_asset_path: str
    calibration_asset_sha256: str

    def identities(self) -> dict[str, str]:
        return {
            "adapter_id": "geometry_v4_reused_content_v9_weighted_joint_rgb_key_only_v1",
            "lf_scorer": "score_content_whitened_lf_image",
            "hf_scorer": "score_hf_image",
            "joint_operator": "weighted_joint_score",
            "calibration_asset": self.calibration_asset_path,
            "calibration_asset_sha256": self.calibration_asset_sha256,
        }

    def __call__(self, current_rgb: np.ndarray, normalized_key: bytes) -> float:
        image = np.asarray(current_rgb, dtype=np.float64)
        if not isinstance(normalized_key, bytes) or not normalized_key:
            raise TypeError("content adapter requires normalized detection-key bytes")
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("content adapter requires finite current RGB in [0,1]")
        # The existing scorers receive exactly the current ordinary RGB and normalized key.
        ordinary = Image.fromarray((image * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
        lf = float(score_content_whitened_lf_image(ordinary, normalized_key, self.lf_public_assets))
        hf = float(score_hf_image(ordinary, normalized_key, self.hf_public_assets))
        if not math.isfinite(lf) or not math.isfinite(hf) or not -1.0 <= lf <= 1.0 or not -1.0 <= hf <= 1.0:
            raise ValueError("unchanged content branch score must be finite in [-1,1]")
        pair = LFHFScorePair(lf=lf, hf=hf)
        score = float(weighted_joint_score(pair.lf, pair.hf, self.calibration_asset))
        if not math.isfinite(score):
            raise ValueError("unchanged weighted-joint content score must be finite")
        return score


def build_reused_weighted_joint_content_adapter(assets: Any, repo_root: str | Path) -> FrozenWeightedJointContentAdapter:
    """Bind only existing Content ISS public assets and the frozen V9 calibration asset."""
    root = Path(repo_root)
    asset_path = root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json"
    sidecar_path = asset_path.with_name(f"{asset_path.name}.sha256")
    if not hasattr(assets, "lf_public_assets") or not hasattr(assets, "hf_public_assets"):
        raise TypeError("reused content adapter requires ContentISS runner public assets")
    raw = asset_path.read_bytes()
    asset = load_calibration_asset(asset_path, sidecar_path)
    return FrozenWeightedJointContentAdapter(
        assets.lf_public_assets,
        assets.hf_public_assets,
        asset,
        "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json",
        hashlib.sha256(raw).hexdigest(),
    )


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hmac.new(key, label, hashlib.sha256).digest()[:8], "big")


def _unit(field: torch.Tensor) -> torch.Tensor:
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    norm = torch.linalg.vector_norm(field)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0: raise RuntimeError("Geometry-V4 anchor basis is degenerate")
    return field / norm


def rgb_anchor_basis(shape: tuple[int, int], detection_key: str | bytes | bytearray | memoryview, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One keyed 12-component global plus 4x4 local RGB-luma basis for writer and detector."""
    h, w = shape
    if h < 16 or w < 16: raise ValueError("RGB anchor basis requires at least 16 pixels per side")
    key = derive_geometry_v4_key(normalize_detection_key(detection_key))
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
    global_field = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    for cycles in (8, 16, 32):
        for angle_degrees in (0, 45, 90, 135):
            label = f"global:{cycles}:{angle_degrees}".encode()
            phase = (_seed(key, label) / 2**64) * 2 * math.pi
            sign = 1.0 if (_seed(key, b"sign:" + label) & 1) else -1.0
            angle = math.radians(angle_degrees)
            global_field += sign * torch.cos(2 * math.pi * cycles * (xx * math.cos(angle) / w + yy * math.sin(angle) / h) + phase)[None, None]
    local_field = torch.zeros_like(global_field)
    for row in range(4):
        for col in range(4):
            label = f"tile:{row * 4 + col}".encode(); phase = (_seed(key, label) / 2**64) * 2 * math.pi
            mask = ((yy >= row*h/4) & (yy < (row+1)*h/4) & (xx >= col*w/4) & (xx < (col+1)*w/4)).to(dtype)
            local_field += mask[None, None] * torch.cos(2 * math.pi * (2 * xx / w + 2 * yy / h) + phase)
    global_field = _unit(global_field); local_field = local_field - (local_field * global_field).sum() * global_field; local_field = _unit(local_field)
    combined = math.sqrt(ENERGY_SHARES[0]) * global_field + math.sqrt(ENERGY_SHARES[1]) * local_field
    return combined, global_field, local_field


def _vae_rgb(pipeline: Any, latents: torch.Tensor) -> torch.Tensor:
    vae = getattr(pipeline, "vae", None); config = getattr(vae, "config", None)
    scaling, shift = getattr(config, "scaling_factor", None), getattr(config, "shift_factor", None)
    if not callable(getattr(vae, "decode", None)) or not isinstance(scaling, (int, float)) or not isinstance(shift, (int, float)) or not math.isfinite(scaling) or scaling <= 0 or not math.isfinite(shift): raise RuntimeError("Geometry-V4 requires a valid differentiable SD3 VAE")
    decoded = vae.decode(latents / float(scaling) + float(shift), return_dict=True)
    sample = getattr(decoded, "sample", None)
    if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[0] != 1 or sample.shape[1] != 3 or not bool(torch.isfinite(sample).all()): raise RuntimeError("Geometry-V4 VAE decode is invalid")
    return (sample + 1.0) / 2.0


def write_final_latent_anchor(latents: torch.Tensor, detection_key: str | bytes | bytearray | memoryview, pipeline: Any) -> torch.Tensor:
    """One fixed first-order VAE-adjoint update; no search, loop, or final-RGB feedback."""
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or not latents.dtype.is_floating_point or not bool(torch.isfinite(latents).all()): raise ValueError("final callback latents must be finite floating NCHW")
    with torch.enable_grad():
        current = latents.detach().to(torch.float32).requires_grad_(True)
        rgb = _vae_rgb(pipeline, current)
        basis, _, _ = rgb_anchor_basis(tuple(rgb.shape[-2:]), detection_key, device=rgb.device, dtype=rgb.dtype)
        luma = (rgb * torch.tensor(_REC709, device=rgb.device, dtype=rgb.dtype)[None, :, None, None]).sum(1, keepdim=True)
        objective = (luma * basis).sum()
        gradient = torch.autograd.grad(objective, current, allow_unused=False)[0]
    if not bool(torch.isfinite(gradient).all()): raise RuntimeError("Geometry-V4 VAE adjoint gradient is nonfinite")
    gradient = gradient - gradient.mean(dim=(-2, -1), keepdim=True); norm = torch.linalg.vector_norm(gradient)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0: raise RuntimeError("Geometry-V4 VAE adjoint gradient is zero")
    return latents + (gradient / norm).to(dtype=latents.dtype) * _LATENT_AMPLITUDE


def rgb_only_anchor_score(rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> float:
    """Read only the current RGB and a normalized key; no truth or source enters."""
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all() or min(image.shape[:2]) < 16:
        raise ValueError("detector requires finite current ordinary RGB")
    if image.max() > 1.0 or image.min() < 0.0:
        raise ValueError("detector requires RGB normalized to [0,1]")
    basis, _, _ = rgb_anchor_basis(image.shape[:2], detection_key)
    luma = torch.from_numpy((image @ _REC709).astype(np.float32))[None, None]
    return float((luma - luma.mean()).mul(basis.cpu()).sum().item())


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
