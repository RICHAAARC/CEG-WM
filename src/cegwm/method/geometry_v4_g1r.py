"""V4-G1R versioned anchor, blind similarity recovery, and holdout gate."""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from cegwm.method.geometry_v4_proxy import (
    _corners,
    _macro_regions,
    _robust_similarity_fit,
    _sample_h,
    _similarity_h,
    _spatial_coverage,
    _valid_corners,
    normalized_phase_correlation,
)
from cegwm.protocol.geometry_v4 import GeometryV4Observation
from cegwm.protocol.geometry_v4_g1r import (
    ENERGY_SHARES,
    FIT_GATES,
    FIT_TILE_IDS,
    HOLDOUT_GATES,
    LUMA_PEAK_CAP,
    LUMA_RMS_CAP,
    SEARCH_TOP_K,
    VALIDATE_TILE_IDS,
    derive_g1r_keys,
)

_REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_LATENT_AMPLITUDE = 0.001
_SEARCH_SIZE = 96
_COARSE_ANGLES = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
_COARSE_SCALES = (0.86, 0.9, 0.95, 1.0, 1.05, 1.1, 1.14)
_FINE_ANGLE_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
_FINE_SCALE_OFFSETS = (-0.02, -0.01, 0.0, 0.01, 0.02)


@dataclass(frozen=True, slots=True)
class G1RAnchorFields:
    combined: np.ndarray
    search: np.ndarray
    fit: np.ndarray
    validate: np.ndarray


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hmac.new(key, label, hashlib.sha256).digest()[:8], "big")


def _unit(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    value = value - float(np.mean(value))
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("V4-G1R anchor field is degenerate")
    return value / norm


def _domain_fields(shape: tuple[int, int], domain_keys: Mapping[str, bytes]) -> G1RAnchorFields:
    height, width = shape
    if height < 32 or width < 32:
        raise ValueError("V4-G1R anchor requires at least 32 pixels per side")
    if set(domain_keys) != {"search", "fit", "validate"} or any(not isinstance(value, bytes) or not value for value in domain_keys.values()):
        raise TypeError("V4-G1R requires three non-empty domain keys")
    yy, xx = np.mgrid[:height, :width]
    xn, yn = xx / width, yy / height
    search = np.zeros((height, width), dtype=np.float64)
    for cycles in (8, 16, 24):
        for angle_degrees in (0, 45, 90, 135):
            label = f"search:{cycles}:{angle_degrees}".encode("ascii")
            phase = _seed(domain_keys["search"], label) / 2**64 * 2.0 * math.pi
            sign = 1.0 if _seed(domain_keys["search"], b"sign:" + label) & 1 else -1.0
            angle = math.radians(angle_degrees)
            search += sign * np.cos(2.0 * math.pi * cycles * (xn * math.cos(angle) + yn * math.sin(angle)) + phase)
    search = _unit(search)

    def partition(name: str, tile_ids: tuple[int, ...]) -> np.ndarray:
        answer = np.zeros_like(search)
        for tile_id in tile_ids:
            row, column = divmod(tile_id, 4)
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = column * width // 4, (column + 1) * width // 4
            local_y, local_x = np.mgrid[: y1 - y0, : x1 - x0]
            local_x = (local_x + 0.5) / max(1, x1 - x0)
            local_y = (local_y + 0.5) / max(1, y1 - y0)
            label = f"{name}:tile:{tile_id}".encode("ascii")
            tile = np.zeros_like(local_x)
            for component, (frequency_x, frequency_y) in enumerate(((2, 3), (3, -5), (5, 2), (6, -3), (4, 5), (7, 1))):
                component_label = label + f":{component}".encode("ascii")
                phase = _seed(domain_keys[name], component_label) / 2**64 * 2.0 * math.pi
                sign = 1.0 if _seed(domain_keys[name], b"sign:" + component_label) & 1 else -1.0
                tile += sign * np.cos(2 * math.pi * (frequency_x * local_x + frequency_y * local_y) + phase)
            tile -= float(np.mean(tile))
            search_tile = search[y0:y1, x0:x1]
            denominator = float(np.sum(search_tile * search_tile))
            if denominator > 1e-12:
                tile -= float(np.sum(tile * search_tile) / denominator) * search_tile
            answer[y0:y1, x0:x1] = tile
        return _unit(answer)

    fit = partition("fit", FIT_TILE_IDS)
    validate = partition("validate", VALIDATE_TILE_IDS)
    combined = math.sqrt(ENERGY_SHARES[0]) * search + math.sqrt(ENERGY_SHARES[1]) * fit + math.sqrt(ENERGY_SHARES[2]) * validate
    return G1RAnchorFields(combined, search, fit, validate)


def g1r_anchor_fields(shape: tuple[int, int], detection_key: str | bytes | bytearray | memoryview) -> G1RAnchorFields:
    return _domain_fields(shape, derive_g1r_keys(detection_key))


def _require_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32 or not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("V4-G1R requires finite ordinary RGB in [0,1]")
    return image


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ _REC709


def write_g1r_rgb(rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> tuple[np.ndarray, Mapping[str, float]]:
    """Synthetic-only ordinary-RGB writer using the same frozen total budget."""
    image = _require_rgb(rgb)
    anchor = g1r_anchor_fields(image.shape[:2], detection_key).combined
    target = 0.75 * LUMA_RMS_CAP
    scale = min(target * math.sqrt(anchor.size), LUMA_PEAK_CAP / max(1e-12, float(np.max(np.abs(anchor)))))
    marked = np.clip(image + (scale * anchor)[..., None], 0.0, 1.0)
    delta = _luma(marked) - _luma(image)
    rms, peak = float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta)))
    if rms > LUMA_RMS_CAP + 1e-12 or peak > LUMA_PEAK_CAP + 1e-12:
        raise RuntimeError("V4-G1R RGB writer exceeded the frozen luma budget")
    return marked, {"luma_rms": rms, "luma_peak": peak, "luma_rms_cap": LUMA_RMS_CAP, "luma_peak_cap": LUMA_PEAK_CAP}


def _torch_anchor(shape: tuple[int, int], detection_key: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    field = g1r_anchor_fields(shape, detection_key).combined
    return torch.as_tensor(field, device=device, dtype=dtype)[None, None]


def _vae_rgb(pipeline: Any, latents: torch.Tensor) -> torch.Tensor:
    vae, config = getattr(pipeline, "vae", None), getattr(getattr(pipeline, "vae", None), "config", None)
    scaling, shift = getattr(config, "scaling_factor", None), getattr(config, "shift_factor", None)
    if not callable(getattr(vae, "decode", None)) or not isinstance(scaling, (int, float)) or not isinstance(shift, (int, float)) or scaling <= 0 or not math.isfinite(float(scaling)) or not math.isfinite(float(shift)):
        raise RuntimeError("V4-G1R requires a differentiable SD3 VAE")
    try:
        parameter = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("V4-G1R cannot resolve VAE dtype/device") from error
    coordinate = (latents.to(torch.float32) / float(scaling) + float(shift)).to(parameter.device, parameter.dtype)
    sample = getattr(vae.decode(coordinate, return_dict=True), "sample", None)
    if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[:2] != (1, 3) or not bool(torch.isfinite(sample).all()):
        raise RuntimeError("V4-G1R VAE decode is invalid")
    return (sample + 1.0) / 2.0


def write_g1r_final_latent(latents: torch.Tensor, detection_key: object, pipeline: Any) -> torch.Tensor:
    """One fixed G1R VAE-adjoint update; no final-RGB feedback or search."""
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or not latents.dtype.is_floating_point or not bool(torch.isfinite(latents).all()):
        raise ValueError("V4-G1R final callback latents must be finite floating NCHW")
    with torch.enable_grad():
        current = latents.detach().to(torch.float32).requires_grad_(True)
        rgb = _vae_rgb(pipeline, current)
        basis = _torch_anchor(tuple(rgb.shape[-2:]), detection_key, device=rgb.device, dtype=rgb.dtype)
        weights = torch.as_tensor(_REC709, device=rgb.device, dtype=rgb.dtype)[None, :, None, None]
        objective = ((rgb * weights).sum(1, keepdim=True) * basis).sum()
        gradient = torch.autograd.grad(objective, current, allow_unused=False)[0]
    gradient = gradient - gradient.mean(dim=(-2, -1), keepdim=True)
    norm = torch.linalg.vector_norm(gradient)
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError("V4-G1R VAE-adjoint gradient is invalid")
    return latents + (gradient / norm).to(latents.dtype) * _LATENT_AMPLITUDE


def _valid_warp(image: np.ndarray, output_to_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fill = np.median(image, axis=(0, 1))
    warped = _sample_h(image, output_to_input, fill)
    valid = _sample_h(np.ones(image.shape[:2], dtype=np.float64), output_to_input, 0.0) >= 0.999999
    return np.clip(warped, 0.0, 1.0), valid


def _normalized_score(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool)
    if left.shape != right.shape or mask.shape != left.shape or int(mask.sum()) < max(16, math.ceil(0.6 * mask.size)):
        return -1.0
    a, b = np.asarray(left)[mask], np.asarray(right)[mask]
    a, b = a - float(np.mean(a)), b - float(np.mean(b))
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return -1.0 if denominator <= 1e-12 else float(np.sum(a * b) / denominator)


def _search_band(plane: np.ndarray) -> np.ndarray:
    """Fixed normalized constellation bands around the 8/16/24-cycle carriers."""
    value = np.asarray(plane, dtype=np.float64) - float(np.mean(plane))
    fy = np.fft.fftfreq(value.shape[0]) * value.shape[0]
    fx = np.fft.fftfreq(value.shape[1]) * value.shape[1]
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = np.minimum.reduce(tuple(np.abs(radius - cycles) for cycles in (8.0, 16.0, 24.0))) <= 1.5
    return np.fft.ifft2(np.fft.fft2(value) * mask).real


def _translation(rs_rgb: np.ndarray, valid: np.ndarray, reference: np.ndarray) -> Mapping[str, object]:
    plane = _search_band(_luma(rs_rgb))
    reference = _search_band(reference)
    window = np.outer(np.hanning(plane.shape[0]), np.hanning(plane.shape[1]))
    weight = valid.astype(np.float64) * window
    if float(weight.sum()) <= 1e-12:
        return {"shift_x": 0, "shift_y": 0, "PSR": 0.0}
    observed = (plane - float(np.sum(plane * weight) / np.sum(weight))) * weight
    keyed = (reference - float(np.sum(reference * weight) / np.sum(weight))) * weight
    return normalized_phase_correlation(observed, keyed)


def _fixed_translation_candidate(image: np.ndarray, reference: np.ndarray, angle: float, scale: float, tx: float, ty: float, phase: Mapping[str, object]) -> dict[str, object]:
    rs = _similarity_h(angle, scale)
    translation = np.asarray(((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)), dtype=np.float64)
    canonical_to_attacked = rs @ translation
    rectified, valid = _valid_warp(image, canonical_to_attacked)
    ncc = _normalized_score(_search_band(_luma(rectified)), _search_band(reference), valid)
    surface = np.asarray(phase["surface"], dtype=np.float64)
    ix, iy = int(round(tx * (image.shape[1] - 1))) % image.shape[1], int(round(ty * (image.shape[0] - 1))) % image.shape[0]
    phase_z = float((surface[iy, ix] - float(np.mean(surface))) / (float(np.std(surface)) + 1e-12))
    phase_consistency = max(0.0, ncc) * max(0.0, phase_z)
    rank = (phase_consistency, ncc, phase_z, float(phase["PSR"]), -abs(angle), -abs(math.log(scale)), -abs(tx) - abs(ty), -angle, -scale, -tx, -ty)
    return {"angle": float(angle), "scale": float(scale), "canonical_to_attacked": canonical_to_attacked, "rank": rank, "ncc": ncc, "translation_psr": float(phase["PSR"])}


def _rs_spectral_score(image: np.ndarray, reference: np.ndarray, angle: float, scale: float) -> tuple[object, ...]:
    rectified, valid = _valid_warp(image, _similarity_h(angle, scale))
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1])) * valid.astype(np.float64)
    observed = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((_luma(rectified) - float(np.mean(_luma(rectified)[valid]))) * window))))
    keyed = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(reference * window))))
    height, width = image.shape[:2]
    fy = np.arange(height) - height // 2
    fx = np.arange(width) - width // 2
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = np.minimum.reduce(tuple(np.abs(radius - cycles) for cycles in (8.0, 16.0, 24.0))) <= 2.0
    left, right = observed[mask], keyed[mask]
    left, right = left - float(np.mean(left)), right - float(np.mean(right))
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    score = -1.0 if denominator <= 1e-12 else float(np.sum(left * right) / denominator)
    return (score, -abs(angle), -abs(math.log(scale)), -angle, -scale)


def _search_candidates(image: np.ndarray, search_key: bytes) -> tuple[dict[str, object], ...]:
    ordinary = Image.fromarray((image * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    resized = np.asarray(ordinary.resize((_SEARCH_SIZE, _SEARCH_SIZE), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    neutral = {"search": search_key, "fit": hashlib.sha256(b"fit-neutral").digest(), "validate": hashlib.sha256(b"validate-neutral").digest()}
    reference = _domain_fields((_SEARCH_SIZE, _SEARCH_SIZE), neutral).search
    coarse = sorted(({"angle": angle, "scale": scale, "rank": _rs_spectral_score(resized, reference, angle, scale)} for angle in _COARSE_ANGLES for scale in _COARSE_SCALES), key=lambda item: item["rank"], reverse=True)[:SEARCH_TOP_K]
    fine: list[dict[str, object]] = []
    for seed in coarse:
        for angle_offset in _FINE_ANGLE_OFFSETS:
            for scale_offset in _FINE_SCALE_OFFSETS:
                angle, scale = float(seed["angle"]) + angle_offset, float(seed["scale"]) + scale_offset
                if -10.0 <= angle <= 10.0 and 0.84 <= scale <= 1.16:
                    fine.append({"angle": angle, "scale": scale, "rank": _rs_spectral_score(resized, reference, angle, scale)})
    refined_rs = sorted(fine, key=lambda item: item["rank"], reverse=True)[:SEARCH_TOP_K]
    translated: list[dict[str, object]] = []
    translation_grid = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
    for seed in refined_rs:
        angle, scale = float(seed["angle"]), float(seed["scale"])
        rs_rgb, rs_valid = _valid_warp(resized, _similarity_h(angle, scale))
        phase = _translation(rs_rgb, rs_valid, reference)
        for tx in translation_grid:
            for ty in translation_grid:
                translated.append(_fixed_translation_candidate(resized, reference, angle, scale, tx, ty, phase))
    return tuple(sorted(translated, key=lambda item: item["rank"], reverse=True)[:SEARCH_TOP_K])


def _patch(array: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray | None:
    if center_x - radius < 0 or center_y - radius < 0 or center_x + radius >= array.shape[1] or center_y + radius >= array.shape[0]:
        return None
    return array[center_y - radius:center_y + radius + 1, center_x - radius:center_x + radius + 1]


def _narrow_patch(patch: np.ndarray) -> np.ndarray:
    """Fixed local narrow band; removes ordinary low-frequency tile content."""
    value = np.asarray(patch, dtype=np.float64) - float(np.mean(patch))
    fy = np.fft.fftfreq(value.shape[0]) * value.shape[0]
    fx = np.fft.fftfreq(value.shape[1]) * value.shape[1]
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (radius >= 1.25) & (radius <= min(value.shape) * 0.42)
    return np.fft.ifft2(np.fft.fft2(value) * mask).real


def _holdout_band(plane: np.ndarray) -> np.ndarray:
    value = np.asarray(plane, dtype=np.float64) - float(np.mean(plane))
    fy = np.fft.fftfreq(value.shape[0]) * value.shape[0]
    fx = np.fft.fftfreq(value.shape[1]) * value.shape[1]
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (radius >= 6.0) & (radius <= min(value.shape) * 0.45)
    return np.fft.ifft2(np.fft.fft2(value) * mask).real


def _tile_matches(image: np.ndarray, canonical_to_attacked: np.ndarray, reference: np.ndarray, tile_ids: tuple[int, ...], *, correlation_min: float, margin_min: float, window_divisor: int = 16, allow_offset: bool = True) -> list[dict[str, object]]:
    rectified, valid = _valid_warp(image, canonical_to_attacked)
    plane, height, width = _luma(rectified), image.shape[0], image.shape[1]
    radius, search_radius = max(4, min(height, width) // window_divisor), max(2, min(height, width) // 48)
    matches: list[dict[str, object]] = []
    for tile_id in tile_ids:
        row, column = divmod(tile_id, 4)
        cx, cy = (column + 0.5) / 4.0, (row + 0.5) / 4.0
        px, py = int(round(cx * (width - 1))), int(round(cy * (height - 1)))
        template = _patch(reference, px, py, radius)
        if template is None:
            continue
        template = _narrow_patch(template)
        candidates: list[tuple[float, int, int]] = []
        for oy in range(-search_radius, search_radius + 1):
            for ox in range(-search_radius, search_radius + 1):
                observed, mask = _patch(plane, px + ox, py + oy, radius), _patch(valid, px + ox, py + oy, radius)
                if observed is not None and mask is not None:
                    score = _normalized_score(_narrow_patch(observed), template, mask)
                    candidates.append((score, ox, oy))
        if not candidates:
            continue
        best = max(candidates, key=lambda item: (item[0], -abs(item[1]) - abs(item[2]), -item[2], -item[1])) if allow_offset else next(item for item in candidates if item[1:] == (0, 0))
        separated = [item for item in candidates if abs(item[1] - best[1]) > 1 or abs(item[2] - best[2]) > 1]
        second = max((item[0] for item in separated), default=-1.0)
        best_patch = _patch(plane, px + best[1], py + best[2], radius)
        psr = 0.0 if best_patch is None else float(normalized_phase_correlation(_narrow_patch(best_patch), template)["PSR"])
        margin = float(best[0] - second)
        if best[0] < correlation_min or margin < margin_min:
            continue
        q = np.asarray(((px + best[1]) / (width - 1), (py + best[2]) / (height - 1), 1.0), dtype=np.float64)
        attacked = canonical_to_attacked @ q
        attacked /= attacked[2]
        matches.append({"tile": (row, column), "tile_id": tile_id, "canonical": (cx, cy), "attacked": (float(attacked[0]), float(attacked[1])), "correlation": float(best[0]), "margin": margin, "PSR": psr})
    return matches


def _fit_candidate(image: np.ndarray, candidate: Mapping[str, object], fit_key: bytes) -> dict[str, object]:
    neutral = {"search": hashlib.sha256(b"search-neutral").digest(), "fit": fit_key, "validate": hashlib.sha256(b"validate-neutral").digest()}
    reference = _domain_fields(image.shape[:2], neutral).fit
    matches = _tile_matches(image, np.asarray(candidate["canonical_to_attacked"]), reference, FIT_TILE_IDS, correlation_min=FIT_GATES["correlation"], margin_min=FIT_GATES["margin"])
    estimate, inliers, residuals, condition = _robust_similarity_fit(matches, inlier_threshold=FIT_GATES["reprojection"], minimum_inliers=FIT_GATES["support"])
    support = len(inliers)
    coverage, macro = _spatial_coverage(inliers), _macro_regions(inliers)
    reprojection = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else math.inf
    ratio = support / len(matches) if matches else 0.0
    valid = bool(estimate is not None and support >= FIT_GATES["support"] and coverage >= FIT_GATES["coverage"] and macro >= FIT_GATES["macro_regions"] and condition <= FIT_GATES["condition"] and reprojection <= FIT_GATES["reprojection"] and ratio >= 0.5)
    rank = (int(valid), support, macro, coverage, ratio, -reprojection, -condition, float(candidate["ncc"]), candidate["rank"])
    return {"valid": valid, "canonical_to_attacked": estimate, "matches": tuple(inliers), "support": support, "coverage": coverage, "macro_regions": macro, "reprojection": reprojection, "condition": condition, "inlier_ratio": ratio, "rank": rank, "search": candidate}


def _rotation_scale(h: np.ndarray) -> tuple[float, float]:
    return math.degrees(math.atan2(float(h[1, 0]), float(h[0, 0]))), math.hypot(float(h[0, 0]), float(h[1, 0]))


def _holdout_metrics(image: np.ndarray, canonical_to_attacked: np.ndarray, validate_key: bytes) -> dict[str, object]:
    neutral = {"search": hashlib.sha256(b"search-neutral").digest(), "fit": hashlib.sha256(b"fit-neutral").digest(), "validate": validate_key}
    reference = _domain_fields(image.shape[:2], neutral).validate
    matches = _tile_matches(image, canonical_to_attacked, reference, VALIDATE_TILE_IDS, correlation_min=HOLDOUT_GATES["correlation"], margin_min=HOLDOUT_GATES["margin"])
    estimate, inliers, residuals, condition = _robust_similarity_fit(matches, inlier_threshold=FIT_GATES["reprojection"], minimum_inliers=6)
    coverage, macro = _spatial_coverage(inliers), _macro_regions(inliers)
    correlations = [float(match["correlation"]) for match in inliers]
    margins = [float(match["margin"]) for match in inliers]
    psrs = [float(match["PSR"]) for match in inliers]
    rectified, valid = _valid_warp(image, canonical_to_attacked)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    weight = valid.astype(np.float64) * window
    holdout_psr = float(normalized_phase_correlation((_luma(rectified) - float(np.mean(_luma(rectified)[valid]))) * weight, reference * weight)["PSR"])
    holdout_correlation = _normalized_score(_holdout_band(_luma(rectified)), _holdout_band(reference), valid)
    secondary_matches = _tile_matches(image, canonical_to_attacked, reference, VALIDATE_TILE_IDS, correlation_min=HOLDOUT_GATES["correlation"], margin_min=HOLDOUT_GATES["margin"], window_divisor=14)
    secondary_estimate, secondary_inliers, _, _ = _robust_similarity_fit(secondary_matches, inlier_threshold=FIT_GATES["reprojection"], minimum_inliers=6)
    rotation_spread = log_scale_spread = corner_consistency = math.inf
    if estimate is not None and secondary_estimate is not None:
        rotation_frozen, scale_frozen = _rotation_scale(canonical_to_attacked)
        rotation_a, scale_a = _rotation_scale(estimate)
        rotation_b, scale_b = _rotation_scale(secondary_estimate)
        rotations = (rotation_frozen, rotation_a, rotation_b)
        rotation_spread = max(abs((left - right + 90.0) % 180.0 - 90.0) for left in rotations for right in rotations)
        logs = (math.log(scale_frozen), math.log(scale_a), math.log(scale_b))
        log_scale_spread = max(logs) - min(logs)
        canonical_points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (.5, .5))
        corner_consistency = 0.0
        for fitted in (estimate, secondary_estimate):
            for x, y in canonical_points:
                frozen_point = canonical_to_attacked @ np.asarray((x, y, 1.0))
                fitted_point = fitted @ np.asarray((x, y, 1.0))
                distance = np.linalg.norm(frozen_point[:2] / frozen_point[2] - fitted_point[:2] / fitted_point[2]) / math.sqrt(2.0)
                corner_consistency = max(corner_consistency, float(distance))
    corners = () if estimate is None else _corners(np.linalg.inv(canonical_to_attacked))
    corner_validity = bool(estimate is not None and _valid_corners(corners))
    median_correlation = float(np.median(correlations)) if correlations else -1.0
    median_margin = float(np.median(margins)) if margins else -1.0
    median_tile_psr = float(np.median(psrs)) if psrs else 0.0
    passed = bool(len(inliers) >= 6 and len(secondary_inliers) >= 6 and coverage >= HOLDOUT_GATES["coverage"] and macro >= HOLDOUT_GATES["macro_regions"] and median_correlation >= HOLDOUT_GATES["correlation"] and holdout_correlation >= HOLDOUT_GATES["correlation"] and median_margin >= HOLDOUT_GATES["margin"] and holdout_psr >= HOLDOUT_GATES["psr"] and rotation_spread <= HOLDOUT_GATES["rotation_spread"] and log_scale_spread <= HOLDOUT_GATES["log_scale_spread"] and corner_consistency <= FIT_GATES["reprojection"] and corner_validity and condition <= FIT_GATES["condition"])
    return {"passed": passed, "support": len(inliers), "secondary_window_support": len(secondary_inliers), "coverage": coverage, "macro_regions": macro, "median_correlation": median_correlation, "fixed_h_narrow_band_correlation": holdout_correlation, "median_margin": median_margin, "holdout_psr": holdout_psr, "median_tile_psr": median_tile_psr, "rotation_spread_degrees": rotation_spread, "log_scale_spread": log_scale_spread, "frozen_h_corner_consistency": corner_consistency, "corner_validity": corner_validity}


def _detect_domains(image: np.ndarray, domain_keys: Mapping[str, bytes]) -> dict[str, object]:
    candidates = _search_candidates(image, domain_keys["search"])
    if not candidates:
        return {"observation": GeometryV4Observation(None, (), 0, 0.0, "UNRELIABLE"), "frozen_h": None, "search_identity": (), "fit": {}, "holdout": {}}
    fitted = tuple(_fit_candidate(image, candidate, domain_keys["fit"]) for candidate in candidates)
    selected = max(fitted, key=lambda item: item["rank"])
    search_identity = tuple((float(item["angle"]), float(item["scale"]), tuple(float(value) for value in np.asarray(item["canonical_to_attacked"]).reshape(-1))) for item in candidates)
    if not selected["valid"] or selected["canonical_to_attacked"] is None:
        return {"observation": GeometryV4Observation(None, (), int(selected["support"]), 0.0, "UNRELIABLE"), "frozen_h": None, "search_identity": search_identity, "fit": selected, "holdout": {}}
    canonical_to_attacked = np.asarray(selected["canonical_to_attacked"], dtype=np.float64)
    attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
    attacked_to_canonical /= attacked_to_canonical[2, 2]
    corners = _corners(attacked_to_canonical)
    frozen_h = tuple(float(value) for value in attacked_to_canonical.reshape(-1))
    holdout = _holdout_metrics(image, canonical_to_attacked, domain_keys["validate"])
    reliable = bool(holdout["passed"] and _valid_corners(corners))
    observation = GeometryV4Observation(frozen_h, corners, int(selected["support"]), 1.0, "RELIABLE") if reliable else GeometryV4Observation(None, (), int(selected["support"]), 0.0, "UNRELIABLE")
    return {"observation": observation, "frozen_h": frozen_h, "search_identity": search_identity, "fit": selected, "holdout": holdout}


def detect_g1r(attacked_rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> Mapping[str, object]:
    """Attacked-RGB plus normalized-key-only public fail-closed detector."""
    image = _require_rgb(attacked_rgb)
    result = _detect_domains(image, derive_g1r_keys(detection_key))
    observation = result["observation"]
    assert isinstance(observation, GeometryV4Observation)
    return {"H_hat": observation.H_hat, "corners_hat": observation.corners_hat, "support": observation.support, "reliability": observation.reliability, "status": observation.status, "diagnostics": {"fit": result["fit"], "holdout": result["holdout"], "public_h_direction": "attacked_to_canonical"}}
