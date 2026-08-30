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
from cegwm.protocol.geometry_v4_generative import (
    ENERGY_SHARES,
    G1_MIN_ANCHOR_SCORE,
    G1_MIN_MACRO_REGIONS,
    G1_MIN_SUPPORT,
    G1_MIN_TILE_SCORE,
    G1_MIN_TRANSLATION_PSR,
    G1_ROTATION_COARSE_DEGREES,
    G1_ROTATION_FINE_OFFSETS,
    G1_SCALE_COARSE,
    G1_SCALE_FINE_OFFSETS,
    LUMA_PEAK_CAP,
    LUMA_RMS_CAP,
)
from cegwm.shared.keys import normalize_detection_key
from cegwm.method.geometry_v4_proxy import _corners, _sample_h, _similarity_h, _valid_corners, normalized_phase_correlation
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
    try:
        parameter = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("Geometry-V4 VAE device and dtype cannot be resolved") from error
    if not parameter.dtype.is_floating_point:
        raise RuntimeError("Geometry-V4 VAE must use a floating dtype")
    # Keep `latents`' float32 leaf in the graph; this cast is differentiable.
    coordinate = (latents.to(torch.float32) / float(scaling) + float(shift)).to(device=parameter.device, dtype=parameter.dtype)
    decoded = vae.decode(coordinate, return_dict=True)
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


def _g1_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32 or not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("G1 detector requires finite current ordinary RGB in [0,1]")
    return image


def _g1_luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ _REC709


def _g1_valid_warp(image: np.ndarray, output_to_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fill = np.median(image, axis=(0, 1))
    warped = _sample_h(image, output_to_input, fill)
    valid = _sample_h(np.ones(image.shape[:2], dtype=np.float64), output_to_input, 0.0) >= 0.999999
    return np.clip(warped, 0.0, 1.0), valid


def _g1_phase_translation(rectified: np.ndarray, valid: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    plane = _g1_luma(rectified)
    window = np.outer(np.hanning(plane.shape[0]), np.hanning(plane.shape[1]))
    weight = window * valid.astype(np.float64)
    if float(weight.sum()) <= 1e-12:
        return {"shift_x": 0, "shift_y": 0, "PSR": 0.0}
    observed = (plane - float(np.sum(plane * weight) / np.sum(weight))) * weight
    keyed = (reference - float(np.sum(reference * weight) / np.sum(weight))) * weight
    return normalized_phase_correlation(observed, keyed)


def _g1_translation(tx: float, ty: float) -> np.ndarray:
    return np.asarray(((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)), dtype=np.float64)


def _g1_search_candidate(image: np.ndarray, reference: np.ndarray, angle: float, scale: float) -> dict[str, object]:
    rs = _similarity_h(angle, scale)
    rectified_rs, valid_rs = _g1_valid_warp(image, rs)
    phase = _g1_phase_translation(rectified_rs, valid_rs, reference)
    base_x, base_y = int(phase["shift_x"]), int(phase["shift_y"])
    best: dict[str, object] | None = None
    height, width = image.shape[:2]
    # A fixed local integer refinement removes phase-correlation quantization only.
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            shift_x, shift_y = base_x + ox, base_y + oy
            canonical_to_attacked = rs @ _g1_translation(shift_x / max(1, width - 1), shift_y / max(1, height - 1))
            rectified, valid = _g1_valid_warp(image, canonical_to_attacked)
            plane = _g1_luma(rectified)
            centered = plane - float(np.mean(plane[valid]))
            score = float(np.sum(centered * reference * valid))
            rank = (score, float(phase["PSR"]), -abs(ox) - abs(oy), -shift_y, -shift_x)
            if best is None or rank > best["rank"]:
                best = {
                    "angle": float(angle), "scale": float(scale), "shift_x": shift_x, "shift_y": shift_y,
                    "canonical_to_attacked": canonical_to_attacked, "translation_psr": float(phase["PSR"]),
                    "search_score": score, "rank": rank,
                }
    assert best is not None
    return best


def _g1_search(image: np.ndarray, detection_key: object) -> dict[str, object]:
    search_size = 128
    ordinary = Image.fromarray((image * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    resized = np.asarray(ordinary.resize((search_size, search_size), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    reference = rgb_anchor_basis((search_size, search_size), detection_key)[0][0, 0].numpy().astype(np.float64)
    coarse = max(
        (_g1_search_candidate(resized, reference, angle, scale) for angle in G1_ROTATION_COARSE_DEGREES for scale in G1_SCALE_COARSE),
        key=lambda item: item["rank"],
    )
    candidates = []
    for angle_offset in G1_ROTATION_FINE_OFFSETS:
        for scale_offset in G1_SCALE_FINE_OFFSETS:
            angle = float(coarse["angle"]) + angle_offset
            scale = float(coarse["scale"]) + scale_offset
            if -10.0 <= angle <= 10.0 and 0.84 <= scale <= 1.16:
                candidates.append(_g1_search_candidate(resized, reference, angle, scale))
    return max(candidates, key=lambda item: item["rank"])


def rectify_g1_rgb(attacked_rgb: np.ndarray, h_attacked_to_canonical: tuple[float, ...]) -> np.ndarray:
    """Rectify current RGB using the public attacked-to-canonical estimate."""
    image = _g1_rgb(attacked_rgb)
    homography = np.asarray(h_attacked_to_canonical, dtype=np.float64)
    if homography.shape != (9,) or not np.isfinite(homography).all():
        raise ValueError("G1 H_hat must contain nine finite values")
    homography = homography.reshape(3, 3)
    if abs(float(np.linalg.det(homography))) <= 1e-12:
        raise ValueError("G1 H_hat must be nonsingular")
    return _g1_valid_warp(image, np.linalg.inv(homography))[0]


def detect_g1_geometry(attacked_rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> dict[str, object]:
    """Fixed attacked-RGB plus normalized-key-only G1 similarity recovery."""
    image = _g1_rgb(attacked_rgb)
    normalized = normalize_detection_key(detection_key)
    selected = _g1_search(image, normalized)
    # Search translations are normalized at 128px; the transform itself is resolution independent.
    canonical_to_attacked = np.asarray(selected["canonical_to_attacked"], dtype=np.float64)
    attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
    attacked_to_canonical /= attacked_to_canonical[2, 2]
    corners = _corners(attacked_to_canonical)
    corner_validity = _valid_corners(corners)
    rectified, valid = _g1_valid_warp(image, canonical_to_attacked)
    plane = _g1_luma(rectified)
    centered = plane - float(np.mean(plane[valid]))
    combined_t, global_t, local_t = rgb_anchor_basis(image.shape[:2], normalized)
    combined = combined_t[0, 0].numpy().astype(np.float64)
    global_part = global_t[0, 0].numpy().astype(np.float64)
    local_part = local_t[0, 0].numpy().astype(np.float64)
    anchor_score = float(np.sum(centered * combined * valid))
    global_score = float(np.sum(centered * global_part * valid))
    height, width = image.shape[:2]
    tile_scores: list[float] = []
    macro_regions: set[tuple[int, int]] = set()
    for row in range(4):
        for col in range(4):
            mask = np.zeros((height, width), dtype=bool)
            mask[row * height // 4:(row + 1) * height // 4, col * width // 4:(col + 1) * width // 4] = True
            score = float(np.sum(centered * local_part * valid * mask))
            tile_scores.append(score)
            if score >= G1_MIN_TILE_SCORE:
                macro_regions.add((row // 2, col // 2))
    support = sum(score >= G1_MIN_TILE_SCORE for score in tile_scores)
    reliable = bool(
        corner_validity
        and anchor_score >= G1_MIN_ANCHOR_SCORE
        and global_score > 0.0
        and float(selected["translation_psr"]) >= G1_MIN_TRANSLATION_PSR
        and support >= G1_MIN_SUPPORT
        and len(macro_regions) >= G1_MIN_MACRO_REGIONS
    )
    if not reliable:
        return {
            "H_hat": None, "corners_hat": (), "support": support, "reliability": 0.0, "status": "UNRELIABLE",
            "diagnostics": {"anchor_score": anchor_score, "global_score": global_score, "translation_phase_psr": float(selected["translation_psr"]), "macro_regions": len(macro_regions), "tile_scores": tuple(tile_scores), "public_h_direction": "attacked_to_canonical"},
        }
    return {
        "H_hat": tuple(float(value) for value in attacked_to_canonical.reshape(-1)),
        "corners_hat": corners,
        "support": support,
        "reliability": 1.0,
        "status": "RELIABLE",
        "diagnostics": {"anchor_score": anchor_score, "global_score": global_score, "translation_phase_psr": float(selected["translation_psr"]), "macro_regions": len(macro_regions), "tile_scores": tuple(tile_scores), "rotation_deg": float(selected["angle"]), "scale": float(selected["scale"]), "translation_pixels_search_128": (int(selected["shift_x"]), int(selected["shift_y"])), "public_h_direction": "attacked_to_canonical"},
    }


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
