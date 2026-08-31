"""V4-G1R versioned anchor, blind similarity recovery, and holdout gate."""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping

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
    DECODER_DTYPE_GUARD_EPS_MULTIPLIER,
    ENERGY_SHARES,
    CONTENT_SCORE_DRIFT_MAX,
    FINAL_RGB_PSNR_MIN,
    FINAL_RGB_SSIM_MIN,
    FIT_GATES,
    FIT_PATCH_WINDOW_DIVISOR,
    FIT_TILE_IDS,
    HOLDOUT_FREQUENCY_RADIUS,
    HOLDOUT_GATES,
    HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION,
    HOLDOUT_PATCH_WINDOW_DIVISORS,
    LUMA_PEAK_CAP,
    LUMA_RMS_CAP,
    RGB_CHANNEL_PEAK_CAP,
    RGB_CHANNEL_RMS_CAP,
    SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION,
    SPARSE_CHIP_RADIUS_FRACTION,
    SPARSE_DOMAIN_SUPPORT_GRID,
    SPARSE_LOCAL_ACTIVE_MODULUS,
    SPARSE_LOCAL_GRID,
    SPARSE_SEARCH_ACTIVE_MODULUS,
    SPARSE_SEARCH_GRIDS,
    SPARSE_SEARCH_GROUPS,
    SPARSE_SUPPORT_FRACTION,
    SEARCH_TOP_K,
    TRANSLATION_NMS_RADIUS_PIXELS,
    TRANSLATION_PEAKS_PER_RS,
    TRANSLATION_PSR_MIN,
    VALIDATE_TILE_IDS,
    WRITER_TARGET_RMS_FRACTION,
    derive_g1r_keys,
)
from cegwm.shared.keys import normalize_detection_key

_REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
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


@dataclass(frozen=True, slots=True)
class G1RFinalRGBObservability:
    psnr: float
    ssim: float
    luma_rms: float
    luma_peak: float
    rgb_channel_rms_max: float
    rgb_channel_peak: float
    content_score_drift: float
    correct_domain_scores: Mapping[str, float]
    wrong_domain_scores: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return bool(
            self.psnr > FINAL_RGB_PSNR_MIN
            and self.ssim > FINAL_RGB_SSIM_MIN
            and self.luma_rms <= LUMA_RMS_CAP
            and self.luma_peak <= LUMA_PEAK_CAP
            and self.rgb_channel_rms_max <= RGB_CHANNEL_RMS_CAP
            and self.rgb_channel_peak <= RGB_CHANNEL_PEAK_CAP
            and self.content_score_drift < CONTENT_SCORE_DRIFT_MAX
            and set(self.correct_domain_scores) == {"search", "fit", "validate"}
            and set(self.wrong_domain_scores) == {"search", "fit", "validate"}
            and all(self.correct_domain_scores[name] > self.wrong_domain_scores[name] for name in ("search", "fit", "validate"))
        )


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hmac.new(key, label, hashlib.sha256).digest()[:8], "big")


def _unit_on_support(field: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Mean-center and normalize only fixed support, preserving exact zeros."""
    value = np.asarray(field, dtype=np.float64)
    mask = np.asarray(support, dtype=bool)
    active_mask = mask & (value != 0.0)
    if value.shape != mask.shape or int(active_mask.sum()) < 2:
        raise RuntimeError("V4-G1R anchor support is degenerate")
    active = value[active_mask] - float(np.mean(value[active_mask]))
    norm = float(np.linalg.norm(active))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("V4-G1R supported anchor field is degenerate")
    answer = np.zeros_like(value)
    answer[active_mask] = active / norm
    return answer


def _domain_support_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed key-independent compact supports spanning every canonical region."""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cell_y = np.minimum(SPARSE_DOMAIN_SUPPORT_GRID - 1, yy * SPARSE_DOMAIN_SUPPORT_GRID // height)
    cell_x = np.minimum(SPARSE_DOMAIN_SUPPORT_GRID - 1, xx * SPARSE_DOMAIN_SUPPORT_GRID // width)
    search = (cell_y + cell_x) % 2 == 0
    return search, ~search


def _sparse_prn_component(shape: tuple[int, int], key: bytes, label: bytes, grid: int, active_modulus: int, *, support: np.ndarray | None = None) -> np.ndarray:
    """Fixed scale-normalized signed Gaussian chips; no image-dependent inputs."""
    height, width = shape
    field = np.zeros((height, width), dtype=np.float64)
    cell_height, cell_width = height / grid, width / grid
    radius = max(1, int(round(min(cell_height, cell_width) * SPARSE_CHIP_RADIUS_FRACTION)))
    sigma = max(0.7, radius * 0.65)
    for cell_y in range(grid):
        for cell_x in range(grid):
            cell = f":{cell_y}:{cell_x}".encode("ascii")
            if _seed(key, label + b":active" + cell) % active_modulus:
                continue
            sign = 1.0 if _seed(key, label + b":sign" + cell) & 1 else -1.0
            jitter_x = ((_seed(key, label + b":jx" + cell) % 1024) / 1023.0 - 0.5) * 0.30
            jitter_y = ((_seed(key, label + b":jy" + cell) % 1024) / 1023.0 - 0.5) * 0.30
            center_x = (cell_x + 0.5 + jitter_x) * cell_width
            center_y = (cell_y + 0.5 + jitter_y) * cell_height
            x0, x1 = max(0, int(math.floor(center_x)) - radius), min(width, int(math.floor(center_x)) + radius + 1)
            y0, y1 = max(0, int(math.floor(center_y)) - radius), min(height, int(math.floor(center_y)) + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            chip = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
            field[y0:y1, x0:x1] += sign * chip
    fixed_support = np.ones(shape, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    return _unit_on_support(field, fixed_support)


def _domain_fields(shape: tuple[int, int], domain_keys: Mapping[str, bytes]) -> G1RAnchorFields:
    height, width = shape
    if height < 32 or width < 32:
        raise ValueError("V4-G1R anchor requires at least 32 pixels per side")
    if set(domain_keys) != {"search", "fit", "validate"} or any(not isinstance(value, bytes) or not value for value in domain_keys.values()):
        raise TypeError("V4-G1R requires three non-empty domain keys")
    search_support, local_support = _domain_support_masks(shape)
    search_components = _search_macro_fields(shape, domain_keys["search"])
    search = _unit_on_support(np.sum(search_components, axis=0), search_support)

    def partition(name: str, tile_ids: tuple[int, ...]) -> np.ndarray:
        answer = np.zeros_like(search)
        domain_support = np.zeros_like(search_support)
        for tile_id in tile_ids:
            row, column = divmod(tile_id, 4)
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = column * width // 4, (column + 1) * width // 4
            label = f"{name}:tile:{tile_id}".encode("ascii")
            tile_support = local_support[y0:y1, x0:x1]
            tile = _sparse_prn_component((y1 - y0, x1 - x0), domain_keys[name], label, SPARSE_LOCAL_GRID, SPARSE_LOCAL_ACTIVE_MODULUS, support=tile_support)
            answer[y0:y1, x0:x1] = tile
            domain_support[y0:y1, x0:x1] = tile_support
        return _unit_on_support(answer, domain_support)

    fit = partition("fit", FIT_TILE_IDS)
    validate = partition("validate", VALIDATE_TILE_IDS)
    fields = (search, fit, validate)
    gram = np.asarray([[float(np.sum(left * right)) for right in fields] for left in fields])
    if not np.allclose(gram, np.eye(3), atol=2e-14, rtol=0.0):
        raise RuntimeError("V4-G1R domain fields lost exact orthogonality")
    components = tuple(math.sqrt(share) * field for share, field in zip(ENERGY_SHARES, fields, strict=True))
    combined = sum(components, start=np.zeros_like(search))
    component_energies = tuple(float(np.sum(component * component)) for component in components)
    if not np.allclose(component_energies, ENERGY_SHARES, atol=2e-14, rtol=0.0) or not math.isclose(float(np.sum(combined * combined)), 1.0, abs_tol=2e-14, rel_tol=0.0):
        raise RuntimeError("V4-G1R physical energy shares differ")
    return G1RAnchorFields(combined, search, fit, validate)


def _search_macro_fields(shape: tuple[int, int], search_key: bytes) -> tuple[np.ndarray, ...]:
    """Return the fixed 3-scale x 4-group sparse search atlas components."""
    search_support, _ = _domain_support_masks(shape)
    fields: list[np.ndarray] = []
    for grid in SPARSE_SEARCH_GRIDS:
        for group in range(SPARSE_SEARCH_GROUPS):
            label = f"search:sparse:grid:{grid}:group:{group}".encode("ascii")
            fields.append(_sparse_prn_component(shape, search_key, label, grid, SPARSE_SEARCH_ACTIVE_MODULUS, support=search_support))
    return tuple(fields)


def g1r_anchor_fields(shape: tuple[int, int], detection_key: str | bytes | bytearray | memoryview) -> G1RAnchorFields:
    return _domain_fields(shape, derive_g1r_keys(detection_key))


def _require_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32 or not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("V4-G1R requires finite ordinary RGB in [0,1]")
    return image


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ _REC709


def _carrier_plane(rgb: np.ndarray) -> np.ndarray:
    """Fixed key-independent REC709 luma extraction projection."""
    return _luma(rgb)


def _g1r_scalar_delta(shape: tuple[int, int], detection_key: object) -> np.ndarray:
    """Return the single fixed scalar anchor shared by both writer placements."""
    anchor = g1r_anchor_fields(shape, detection_key).combined
    # Fixed downward rounding guard keeps reconstructed float RGB at or below
    # the unchanged hard cap; it is not image/key adaptive.
    target = WRITER_TARGET_RMS_FRACTION * LUMA_RMS_CAP * (1.0 - 1e-12)
    scale = min(target * math.sqrt(anchor.size), LUMA_PEAK_CAP / max(1e-12, float(np.max(np.abs(anchor)))))
    delta = scale * anchor
    rms, peak = float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta)))
    if rms > LUMA_RMS_CAP + 1e-12 or peak > LUMA_PEAK_CAP + 1e-12:
        raise RuntimeError("V4-G1R scalar anchor exceeded the frozen budget")
    return delta


def _equal_rgb_delta(scalar_delta: np.ndarray) -> np.ndarray:
    """Map one scalar anchor equally into RGB so REC709 luma is identical."""
    delta = np.repeat(np.asarray(scalar_delta, dtype=np.float64)[..., None], 3, axis=2)
    channel_rms = np.sqrt(np.mean(delta * delta, axis=(0, 1)))
    if float(np.max(channel_rms)) > RGB_CHANNEL_RMS_CAP + 1e-12 or float(np.max(np.abs(delta))) > RGB_CHANNEL_PEAK_CAP + 1e-12:
        raise RuntimeError("V4-G1R sparse luma writer exceeded the frozen RGB budget")
    if not np.allclose(_carrier_plane(delta), scalar_delta, atol=1e-12, rtol=1e-12):
        raise RuntimeError("V4-G1R sparse writer/extractor identity differs")
    return delta


def write_g1r_rgb(rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> tuple[np.ndarray, Mapping[str, float]]:
    """Synthetic-only ordinary-RGB writer using the same frozen total budget."""
    image = _require_rgb(rgb)
    scalar_delta = _g1r_scalar_delta(image.shape[:2], detection_key)
    marked = np.clip(image + _equal_rgb_delta(scalar_delta), 0.0, 1.0)
    delta_rgb = marked - image
    delta_luma = _luma(marked) - _luma(image)
    delta_carrier = _carrier_plane(marked) - _carrier_plane(image)
    luma_rms, luma_peak = float(np.sqrt(np.mean(delta_luma * delta_luma))), float(np.max(np.abs(delta_luma)))
    channel_rms = np.sqrt(np.mean(delta_rgb * delta_rgb, axis=(0, 1)))
    rgb_rms_max, rgb_peak = float(np.max(channel_rms)), float(np.max(np.abs(delta_rgb)))
    carrier_rms = float(np.sqrt(np.mean(delta_carrier * delta_carrier)))
    if luma_rms > LUMA_RMS_CAP + 1e-12 or luma_peak > LUMA_PEAK_CAP + 1e-12 or rgb_rms_max > RGB_CHANNEL_RMS_CAP + 1e-12 or rgb_peak > RGB_CHANNEL_PEAK_CAP + 1e-12:
        raise RuntimeError("V4-G1R RGB writer exceeded the frozen luma budget")
    return marked, {"carrier_rms": carrier_rms, "luma_rms": luma_rms, "luma_peak": luma_peak, "luma_rms_cap": LUMA_RMS_CAP, "luma_peak_cap": LUMA_PEAK_CAP, "rgb_channel_rms_max": rgb_rms_max, "rgb_channel_rms_cap": RGB_CHANNEL_RMS_CAP, "rgb_channel_peak": rgb_peak, "rgb_channel_peak_cap": RGB_CHANNEL_PEAK_CAP}


def _field_score(plane: np.ndarray, field: np.ndarray) -> float:
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2 or plane.shape != field.shape or not np.isfinite(plane).all():
        raise ValueError("V4-G1R field score requires a finite aligned carrier residual")
    plane = plane - float(np.mean(plane))
    denominator = float(np.linalg.norm(plane) * np.linalg.norm(field))
    return -1.0 if denominator <= 1e-12 else float(np.sum(plane * field) / denominator)


def measure_g1r_final_rgb(
    clean_rgb: np.ndarray,
    marked_rgb: np.ndarray,
    detection_key: object,
    wrong_key: object,
    content_detector: Callable[[np.ndarray, bytes], float],
) -> G1RFinalRGBObservability:
    """Final-RGB-only quality, content drift, and all three G1R domain scores."""
    clean, marked = _require_rgb(clean_rgb), _require_rgb(marked_rgb)
    if clean.shape != marked.shape:
        raise ValueError("V4-G1R final RGB pair shape differs")
    normalized = normalize_detection_key(detection_key)
    normalized_wrong = normalize_detection_key(wrong_key)
    if normalized == normalized_wrong:
        raise ValueError("V4-G1R correct and wrong keys collide")
    delta_rgb = marked - clean
    delta_luma = _luma(marked) - _luma(clean)
    delta_carrier = _carrier_plane(marked) - _carrier_plane(clean)
    mse = float(np.mean((marked - clean) ** 2))
    psnr = 300.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    mx, my = float(clean.mean()), float(marked.mean())
    vx, vy = float(clean.var()), float(marked.var())
    covariance = float(((clean - mx) * (marked - my)).mean())
    ssim = ((2 * mx * my + 1e-4) * (2 * covariance + 9e-4)) / ((mx * mx + my * my + 1e-4) * (vx + vy + 9e-4))
    correct_fields = g1r_anchor_fields(clean.shape[:2], normalized)
    wrong_fields = g1r_anchor_fields(clean.shape[:2], normalized_wrong)
    names = ("search", "fit", "validate")
    correct_scores = {name: _field_score(delta_carrier, getattr(correct_fields, name)) for name in names}
    wrong_scores = {name: _field_score(delta_carrier, getattr(wrong_fields, name)) for name in names}
    drift = abs(float(content_detector(clean, normalized)) - float(content_detector(marked, normalized)))
    channel_rms = np.sqrt(np.mean(delta_rgb * delta_rgb, axis=(0, 1)))
    values = (psnr, ssim, float(np.sqrt(np.mean(delta_luma * delta_luma))), float(np.max(np.abs(delta_luma))), float(np.max(channel_rms)), float(np.max(np.abs(delta_rgb))), drift, *correct_scores.values(), *wrong_scores.values())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("V4-G1R final RGB observation is non-finite")
    return G1RFinalRGBObservability(values[0], values[1], values[2], values[3], values[4], values[5], values[6], correct_scores, wrong_scores)


def write_g1r_decoder_output(decoded: torch.Tensor, detection_key: object) -> torch.Tensor:
    """Apply one fixed update to the real VAE decoder output before RGB postprocess."""
    if not isinstance(decoded, torch.Tensor) or decoded.ndim != 4 or decoded.shape[0] != 1 or decoded.shape[1] != 3 or not decoded.dtype.is_floating_point or not bool(torch.isfinite(decoded).all()):
        raise ValueError("V4-G1R decoder output must be finite floating 1x3xHxW")
    scalar_delta = _g1r_scalar_delta((int(decoded.shape[-2]), int(decoded.shape[-1])), detection_key)
    # Diffusers maps the decoder's nominal [-1,1] sample to [0,1], so a 2x
    # decoder-space update is exactly the frozen final-RGB luma update pre-clip.
    rgb_delta = _equal_rgb_delta(scalar_delta)
    dtype_guard = 1.0 - DECODER_DTYPE_GUARD_EPS_MULTIPLIER * float(torch.finfo(decoded.dtype).eps)
    if not 0.0 < dtype_guard < 1.0:
        raise RuntimeError("V4-G1R decoder dtype guard is invalid")
    delta = torch.as_tensor(2.0 * dtype_guard * rgb_delta, device=decoded.device, dtype=decoded.dtype).permute(2, 0, 1)[None]
    updated = decoded + delta
    # Accumulate the actual post-addition update in float64. There is one fixed
    # guard and one fail-closed check: no retry, rescaling search, or adaptation.
    actual_final_rgb_delta = (updated.to(torch.float64) - decoded.to(torch.float64)) / 2.0
    channel_rms = torch.sqrt(torch.mean(actual_final_rgb_delta * actual_final_rgb_delta, dim=(0, 2, 3)))
    peak = torch.max(torch.abs(actual_final_rgb_delta))
    if not bool(torch.isfinite(channel_rms).all()) or not bool(torch.isfinite(peak)) or float(torch.max(channel_rms).item()) > RGB_CHANNEL_RMS_CAP or float(peak.item()) > RGB_CHANNEL_PEAK_CAP:
        raise RuntimeError("V4-G1R post-cast decoder update exceeded the frozen RGB budget")
    return updated


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


def _translation_surface(rs_rgb: np.ndarray, valid: np.ndarray, references: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Joint macro normalized cross-power on reference-derived near-exact bins."""
    plane = _carrier_plane(rs_rgb)
    window = np.outer(np.hanning(plane.shape[0]), np.hanning(plane.shape[1]))
    weight = valid.astype(np.float64) * window
    if float(weight.sum()) <= 1e-12:
        return np.zeros_like(plane), plane, np.zeros_like(plane), ()
    observed = (plane - float(np.sum(plane * weight) / np.sum(weight))) * weight
    observed_fft = np.fft.fft2(observed)
    surfaces: list[np.ndarray] = []
    observed_bands: list[np.ndarray] = []
    keyed_bands: list[np.ndarray] = []
    for reference in references:
        keyed = (reference - float(np.sum(reference * weight) / np.sum(weight))) * weight
        keyed_fft = np.fft.fft2(keyed)
        cross = observed_fft * np.conj(keyed_fft)
        magnitude, keyed_magnitude = np.abs(cross), np.abs(keyed_fft)
        valid_frequency = (keyed_magnitude >= SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION * float(np.max(keyed_magnitude))) & (magnitude > np.finfo(np.float64).eps * max(1.0, float(np.max(magnitude))))
        normalized = np.zeros_like(cross)
        normalized[valid_frequency] = cross[valid_frequency] / magnitude[valid_frequency]
        component = np.fft.ifft2(normalized).real
        component /= float(np.sqrt(np.mean(component * component))) + 1e-12
        surfaces.append(component)
        observed_bands.append(np.fft.ifft2(observed_fft * valid_frequency).real)
        keyed_bands.append(np.fft.ifft2(keyed_fft * valid_frequency).real)
    joint = np.mean(np.stack(surfaces), axis=0)
    return joint, np.sum(observed_bands, axis=0), np.sum(keyed_bands, axis=0), tuple(surfaces)


def _candidate_psr(surface: np.ndarray, dx: int, dy: int) -> float:
    iy, ix = dy % surface.shape[0], dx % surface.shape[1]
    sidelobe = np.ones(surface.shape, dtype=bool)
    for oy in range(-TRANSLATION_NMS_RADIUS_PIXELS, TRANSLATION_NMS_RADIUS_PIXELS + 1):
        for ox in range(-TRANSLATION_NMS_RADIUS_PIXELS, TRANSLATION_NMS_RADIUS_PIXELS + 1):
            sidelobe[(iy + oy) % surface.shape[0], (ix + ox) % surface.shape[1]] = False
    side = surface[sidelobe]
    return float(max(0.0, (float(surface[iy, ix]) - float(np.mean(side))) / (float(np.std(side)) + 1e-12)))


def _align_translation(plane: np.ndarray, valid: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.roll(plane, shift=(-dy, -dx), axis=(0, 1))
    aligned_valid = np.roll(valid, shift=(-dy, -dx), axis=(0, 1)).copy()
    if dy > 0:
        aligned_valid[-dy:, :] = False
    elif dy < 0:
        aligned_valid[:-dy, :] = False
    if dx > 0:
        aligned_valid[:, -dx:] = False
    elif dx < 0:
        aligned_valid[:, :-dx] = False
    return aligned, aligned_valid


def _translation_peaks(surface: np.ndarray) -> tuple[tuple[int, int], ...]:
    max_x = int(math.floor(0.12 * (surface.shape[1] - 1)))
    max_y = int(math.floor(0.12 * (surface.shape[0] - 1)))
    ranked = sorted(
        ((float(surface[dy % surface.shape[0], dx % surface.shape[1]]), -abs(dx) - abs(dy), -abs(dy), -abs(dx), -dy, -dx, dx, dy) for dy in range(-max_y, max_y + 1) for dx in range(-max_x, max_x + 1)),
        reverse=True,
    )
    selected: list[tuple[int, int]] = []
    for *_, dx, dy in ranked:
        if all(max(abs(dx - prior_x), abs(dy - prior_y)) > TRANSLATION_NMS_RADIUS_PIXELS for prior_x, prior_y in selected):
            selected.append((dx, dy))
            if len(selected) == TRANSLATION_PEAKS_PER_RS:
                break
    return tuple(selected)


def _joint_candidates_for_rs(image: np.ndarray, references: tuple[np.ndarray, ...], angle: float, scale: float) -> tuple[dict[str, object], ...]:
    rs = _similarity_h(angle, scale)
    rectified, valid = _valid_warp(image, rs)
    surface, observed_band, reference_band, component_surfaces = _translation_surface(rectified, valid, references)
    candidates = []
    for dx, dy in _translation_peaks(surface):
        tx, ty = dx / (image.shape[1] - 1), dy / (image.shape[0] - 1)
        aligned, aligned_valid = _align_translation(observed_band, valid, dx, dy)
        ncc = _normalized_score(aligned, reference_band, aligned_valid)
        psr = _candidate_psr(surface, dx, dy)
        phase_peak = float(surface[dy % surface.shape[0], dx % surface.shape[1]])
        phase_consistency = phase_peak / (float(np.sqrt(np.mean(surface * surface))) + 1e-12)
        component_values = sorted(float(item[dy % item.shape[0], dx % item.shape[1]]) for item in component_surfaces)
        robust_component_consensus = float(np.mean(component_values[1:-1])) if len(component_values) > 2 else float(np.mean(component_values))
        translation = np.asarray(((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)), dtype=np.float64)
        canonical_to_attacked = rs @ translation
        rank = (robust_component_consensus, phase_consistency, ncc, psr, -abs(angle), -abs(math.log(scale)), -abs(tx) - abs(ty), -angle, -scale, -tx, -ty)
        candidates.append({"angle": float(angle), "scale": float(scale), "canonical_to_attacked": canonical_to_attacked, "rank": rank, "ncc": ncc, "translation_psr": psr, "phase_consistency": phase_consistency, "component_consensus": robust_component_consensus})
    return tuple(candidates)


def _search_candidates(image: np.ndarray, search_key: bytes) -> tuple[dict[str, object], ...]:
    ordinary = Image.fromarray((image * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    resized = np.asarray(ordinary.resize((_SEARCH_SIZE, _SEARCH_SIZE), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    references = _search_macro_fields((_SEARCH_SIZE, _SEARCH_SIZE), search_key)
    coarse_references = references[:8]
    coarse = []
    for angle in _COARSE_ANGLES:
        for scale in _COARSE_SCALES:
            candidates = _joint_candidates_for_rs(resized, coarse_references, angle, scale)
            if candidates:
                coarse.append(max(candidates, key=lambda item: item["rank"]))
    coarse = sorted(coarse, key=lambda item: item["rank"], reverse=True)[:SEARCH_TOP_K]
    fine_pairs: set[tuple[float, float]] = set()
    for seed in coarse:
        for angle_offset in _FINE_ANGLE_OFFSETS:
            for scale_offset in _FINE_SCALE_OFFSETS:
                angle, scale = float(seed["angle"]) + angle_offset, float(seed["scale"]) + scale_offset
                if -10.0 <= angle <= 10.0 and 0.84 <= scale <= 1.16:
                    fine_pairs.add((round(angle, 12), round(scale, 12)))
    joint = [candidate for angle, scale in sorted(fine_pairs) for candidate in _joint_candidates_for_rs(resized, references, angle, scale)]
    return tuple(sorted(joint, key=lambda item: item["rank"], reverse=True)[:SEARCH_TOP_K])


def _patch(array: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray | None:
    if center_x - radius < 0 or center_y - radius < 0 or center_x + radius >= array.shape[1] or center_y + radius >= array.shape[0]:
        return None
    return array[center_y - radius:center_y + radius + 1, center_x - radius:center_x + radius + 1]


@lru_cache(maxsize=16)
def _fixed_patch_trend_basis(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    design = np.stack(
        (np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy, xx**3, xx * xx * yy, xx * yy * yy, yy**3),
        axis=-1,
    ).reshape(-1, 10)
    basis = np.linalg.qr(design, mode="reduced")[0]
    basis.setflags(write=False)
    return basis


def _narrow_patch(patch: np.ndarray) -> np.ndarray:
    """Fixed local narrow band; removes ordinary low-frequency tile content."""
    value = np.asarray(patch, dtype=np.float64)
    trend_basis = _fixed_patch_trend_basis(*value.shape)
    flattened = value.reshape(-1)
    value = (flattened - trend_basis @ (trend_basis.T @ flattened)).reshape(value.shape)
    fy = np.fft.fftfreq(value.shape[0]) * value.shape[0]
    fx = np.fft.fftfreq(value.shape[1]) * value.shape[1]
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (radius >= 1.25) & (radius <= min(value.shape) * 0.42)
    return np.fft.ifft2(np.fft.fft2(value) * mask).real


def _support_matched_score(observed: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float:
    """Key/reference-derived support-aware normalized matched correlation."""
    keyed = np.asarray(reference, dtype=np.float64)
    support = np.abs(keyed) >= SPARSE_SUPPORT_FRACTION * max(1e-12, float(np.max(np.abs(keyed))))
    mask = np.asarray(valid, dtype=bool) & support
    if observed.shape != keyed.shape or mask.shape != keyed.shape or int(mask.sum()) < 16:
        return -1.0
    left, right = np.asarray(observed, dtype=np.float64)[mask], keyed[mask]
    left, right = left - float(np.mean(left)), right - float(np.mean(right))
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return -1.0 if denominator <= 1e-12 else float(np.sum(left * right) / denominator)


def _keyed_holdout_correlation(plane: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float:
    """Correlation on the fixed strong-frequency support of the keyed holdout."""
    window = np.outer(np.hanning(plane.shape[0]), np.hanning(plane.shape[1]))
    weight = valid.astype(np.float64) * window
    if float(weight.sum()) <= 1e-12:
        return -1.0
    observed = (np.asarray(plane, dtype=np.float64) - float(np.sum(plane * weight) / np.sum(weight))) * weight
    keyed = (np.asarray(reference, dtype=np.float64) - float(np.sum(reference * weight) / np.sum(weight))) * weight
    keyed_fft = np.fft.fft2(keyed)
    fy = np.fft.fftfreq(plane.shape[0]) * plane.shape[0]
    fx = np.fft.fftfreq(plane.shape[1]) * plane.shape[1]
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    support = (radius >= HOLDOUT_FREQUENCY_RADIUS[0]) & (radius <= min(HOLDOUT_FREQUENCY_RADIUS[1], min(plane.shape) * 0.45)) & (np.abs(keyed_fft) >= HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION * float(np.max(np.abs(keyed_fft))))
    observed_band = np.fft.ifft2(np.fft.fft2(observed) * support).real
    keyed_band = np.fft.ifft2(keyed_fft * support).real
    return _normalized_score(observed_band, keyed_band, valid)


def _tile_matches(image: np.ndarray, canonical_to_attacked: np.ndarray, reference: np.ndarray, tile_ids: tuple[int, ...], *, correlation_min: float, margin_min: float, window_divisor: int = 16, allow_offset: bool = True, prethreshold: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    rectified, valid = _valid_warp(image, canonical_to_attacked)
    plane, height, width = _carrier_plane(rectified), image.shape[0], image.shape[1]
    radius, search_radius = max(4, min(height, width) // window_divisor), max(2, min(height, width) // 48)
    matches: list[dict[str, object]] = []
    for tile_id in tile_ids:
        row, column = divmod(tile_id, 4)
        cx, cy = (column + 0.5) / 4.0, (row + 0.5) / 4.0
        px, py = int(round(cx * (width - 1))), int(round(cy * (height - 1)))
        template = _patch(reference, px, py, radius)
        if template is None:
            if prethreshold is not None:
                prethreshold.append({"tile_id": tile_id, "best_correlation": None, "margin": None, "accepted": False, "rejection": "missing_template"})
            continue
        template = _narrow_patch(template)
        candidates: list[tuple[float, int, int]] = []
        for oy in range(-search_radius, search_radius + 1):
            for ox in range(-search_radius, search_radius + 1):
                observed, mask = _patch(plane, px + ox, py + oy, radius), _patch(valid, px + ox, py + oy, radius)
                if observed is not None and mask is not None:
                    score = _support_matched_score(_narrow_patch(observed), template, mask)
                    candidates.append((score, ox, oy))
        if not candidates:
            if prethreshold is not None:
                prethreshold.append({"tile_id": tile_id, "best_correlation": None, "margin": None, "accepted": False, "rejection": "missing_valid_patch"})
            continue
        best = max(candidates, key=lambda item: (item[0], -abs(item[1]) - abs(item[2]), -item[2], -item[1])) if allow_offset else next(item for item in candidates if item[1:] == (0, 0))
        separated = [item for item in candidates if abs(item[1] - best[1]) > 1 or abs(item[2] - best[2]) > 1]
        second = max((item[0] for item in separated), default=-1.0)
        best_patch = _patch(plane, px + best[1], py + best[2], radius)
        psr = 0.0 if best_patch is None else float(normalized_phase_correlation(_narrow_patch(best_patch), template)["PSR"])
        margin = float(best[0] - second)
        correlation_ok, margin_ok = best[0] >= correlation_min, margin >= margin_min
        accepted = bool(correlation_ok and margin_ok)
        if prethreshold is not None:
            rejection = "accepted" if accepted else "correlation_and_margin" if not correlation_ok and not margin_ok else "correlation" if not correlation_ok else "margin"
            prethreshold.append({"tile_id": tile_id, "best_correlation": float(best[0]), "margin": margin, "accepted": accepted, "rejection": rejection})
        if not accepted:
            continue
        q = np.asarray(((px + best[1]) / (width - 1), (py + best[2]) / (height - 1), 1.0), dtype=np.float64)
        attacked = canonical_to_attacked @ q
        attacked /= attacked[2]
        matches.append({"tile": (row, column), "tile_id": tile_id, "canonical": (cx, cy), "attacked": (float(attacked[0]), float(attacked[1])), "correlation": float(best[0]), "margin": margin, "PSR": psr})
    return matches


def _fit_candidate(image: np.ndarray, candidate: Mapping[str, object], fit_key: bytes) -> dict[str, object]:
    neutral = {"search": hashlib.sha256(b"search-neutral").digest(), "fit": fit_key, "validate": hashlib.sha256(b"validate-neutral").digest()}
    reference = _domain_fields(image.shape[:2], neutral).fit
    prethreshold: list[dict[str, object]] = []
    matches = _tile_matches(image, np.asarray(candidate["canonical_to_attacked"]), reference, FIT_TILE_IDS, correlation_min=FIT_GATES["correlation"], margin_min=FIT_GATES["margin"], window_divisor=FIT_PATCH_WINDOW_DIVISOR, prethreshold=prethreshold)
    estimate, inliers, residuals, condition = _robust_similarity_fit(matches, inlier_threshold=FIT_GATES["reprojection"], minimum_inliers=FIT_GATES["support"])
    support = len(inliers)
    coverage, macro = _spatial_coverage(inliers), _macro_regions(inliers)
    reprojection = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else math.inf
    ratio = support / len(matches) if matches else 0.0
    valid = bool(estimate is not None and float(candidate["translation_psr"]) >= TRANSLATION_PSR_MIN and support >= FIT_GATES["support"] and coverage >= FIT_GATES["coverage"] and macro >= FIT_GATES["macro_regions"] and condition <= FIT_GATES["condition"] and reprojection <= FIT_GATES["reprojection"] and ratio >= 0.5)
    rank = (int(valid), support, macro, coverage, ratio, -reprojection, -condition, float(candidate["ncc"]), candidate["rank"])
    return {"valid": valid, "canonical_to_attacked": estimate, "matches": tuple(inliers), "prethreshold": tuple(prethreshold), "support": support, "coverage": coverage, "macro_regions": macro, "reprojection": reprojection, "condition": condition, "inlier_ratio": ratio, "rank": rank, "search": candidate}


def _rotation_scale(h: np.ndarray) -> tuple[float, float]:
    return math.degrees(math.atan2(float(h[1, 0]), float(h[0, 0]))), math.hypot(float(h[0, 0]), float(h[1, 0]))


def _holdout_metrics(image: np.ndarray, canonical_to_attacked: np.ndarray, validate_key: bytes) -> dict[str, object]:
    neutral = {"search": hashlib.sha256(b"search-neutral").digest(), "fit": hashlib.sha256(b"fit-neutral").digest(), "validate": validate_key}
    reference = _domain_fields(image.shape[:2], neutral).validate
    matches = _tile_matches(image, canonical_to_attacked, reference, VALIDATE_TILE_IDS, correlation_min=HOLDOUT_GATES["correlation"], margin_min=HOLDOUT_GATES["margin"], window_divisor=HOLDOUT_PATCH_WINDOW_DIVISORS[0])
    estimate, inliers, residuals, condition = _robust_similarity_fit(matches, inlier_threshold=FIT_GATES["reprojection"], minimum_inliers=6)
    coverage, macro = _spatial_coverage(inliers), _macro_regions(inliers)
    correlations = [float(match["correlation"]) for match in inliers]
    margins = [float(match["margin"]) for match in inliers]
    psrs = [float(match["PSR"]) for match in inliers]
    rectified, valid = _valid_warp(image, canonical_to_attacked)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    weight = valid.astype(np.float64) * window
    carrier = _carrier_plane(rectified)
    holdout_psr = float(normalized_phase_correlation((carrier - float(np.mean(carrier[valid]))) * weight, reference * weight)["PSR"])
    holdout_correlation = _keyed_holdout_correlation(carrier, reference, valid)
    secondary_matches = _tile_matches(image, canonical_to_attacked, reference, VALIDATE_TILE_IDS, correlation_min=HOLDOUT_GATES["correlation"], margin_min=HOLDOUT_GATES["margin"], window_divisor=HOLDOUT_PATCH_WINDOW_DIVISORS[1])
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
        return {"observation": GeometryV4Observation(None, (), 0, 0.0, "UNRELIABLE"), "frozen_h": None, "search_identity": (), "search_candidates": (), "fit": {}, "holdout": {}}
    fitted = tuple(_fit_candidate(image, candidate, domain_keys["fit"]) for candidate in candidates)
    selected = max(fitted, key=lambda item: item["rank"])
    search_identity = tuple((float(item["angle"]), float(item["scale"]), tuple(float(value) for value in np.asarray(item["canonical_to_attacked"]).reshape(-1))) for item in candidates)
    if not selected["valid"] or selected["canonical_to_attacked"] is None:
        return {"observation": GeometryV4Observation(None, (), int(selected["support"]), 0.0, "UNRELIABLE"), "frozen_h": None, "search_identity": search_identity, "search_candidates": candidates, "fit": selected, "holdout": {}}
    canonical_to_attacked = np.asarray(selected["canonical_to_attacked"], dtype=np.float64)
    attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
    attacked_to_canonical /= attacked_to_canonical[2, 2]
    corners = _corners(attacked_to_canonical)
    frozen_h = tuple(float(value) for value in attacked_to_canonical.reshape(-1))
    holdout = _holdout_metrics(image, canonical_to_attacked, domain_keys["validate"])
    reliable = bool(holdout["passed"] and _valid_corners(corners))
    observation = GeometryV4Observation(frozen_h, corners, int(selected["support"]), 1.0, "RELIABLE") if reliable else GeometryV4Observation(None, (), int(selected["support"]), 0.0, "UNRELIABLE")
    return {"observation": observation, "frozen_h": frozen_h, "search_identity": search_identity, "search_candidates": candidates, "fit": selected, "holdout": holdout}


def detect_g1r(attacked_rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> Mapping[str, object]:
    """Attacked-RGB plus normalized-key-only public fail-closed detector."""
    image = _require_rgb(attacked_rgb)
    return _public_detection(_detect_domains(image, derive_g1r_keys(detection_key)))


def _public_detection(result: Mapping[str, object]) -> Mapping[str, object]:
    observation = result["observation"]
    assert isinstance(observation, GeometryV4Observation)
    return {"H_hat": observation.H_hat, "corners_hat": observation.corners_hat, "support": observation.support, "reliability": observation.reliability, "status": observation.status}


def _detect_g1r_engineering(attacked_rgb: np.ndarray, detection_key: str | bytes | bytearray | memoryview) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Private runner path returning sanitized diagnostics from the same blind pass."""
    return _detect_g1r_engineering_from_image(_require_rgb(attacked_rgb), detection_key)


def _detect_g1r_engineering_from_image(image: np.ndarray, detection_key: object) -> tuple[Mapping[str, object], Mapping[str, object]]:
    result = _detect_domains(image, derive_g1r_keys(detection_key))
    public = _public_detection(result)
    fit = result.get("fit") if isinstance(result.get("fit"), Mapping) else {}
    search = fit.get("search") if isinstance(fit.get("search"), Mapping) else {}

    def finite(value: object) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    search_summary = []
    for item in result.get("search_candidates", ()):
        matrix = np.asarray(item["canonical_to_attacked"], dtype=np.float64).reshape(3, 3)
        search_summary.append({
            "angle_degrees": finite(item["angle"]),
            "scale": finite(item["scale"]),
            "translation_x": finite(matrix[0, 2]),
            "translation_y": finite(matrix[1, 2]),
            "translation_psr": finite(item["translation_psr"]),
            "ncc": finite(item["ncc"]),
            "component_consensus": finite(item.get("component_consensus", 0.0)),
        })
    fit_summary = {
        "valid": bool(fit.get("valid", False)),
        "support": int(fit.get("support", 0)),
        "coverage": finite(fit.get("coverage", 0.0)),
        "macro_regions": int(fit.get("macro_regions", 0)),
        "inlier_ratio": finite(fit.get("inlier_ratio", 0.0)),
        "reprojection": finite(fit.get("reprojection", math.inf)),
        "condition": finite(fit.get("condition", math.inf)),
        "translation_psr": finite(search.get("translation_psr", 0.0)),
        "ncc": finite(search.get("ncc", -1.0)),
    }
    prethreshold_summary = []
    rejection_counts: dict[str, int] = {}
    for item in fit.get("prethreshold", ()):
        rejection = str(item.get("rejection", "invalid"))
        rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
        prethreshold_summary.append({
            "tile_id": int(item["tile_id"]),
            "best_correlation": None if item.get("best_correlation") is None else finite(item["best_correlation"]),
            "margin": None if item.get("margin") is None else finite(item["margin"]),
            "accepted": bool(item.get("accepted", False)),
            "rejection": rejection,
        })
    fit_summary["prethreshold_tiles"] = tuple(prethreshold_summary)
    fit_summary["rejection_counts"] = rejection_counts
    holdout = result.get("holdout") if isinstance(result.get("holdout"), Mapping) else {}
    holdout_names = (
        "passed", "support", "secondary_window_support", "coverage", "macro_regions",
        "median_correlation", "fixed_h_narrow_band_correlation", "median_margin",
        "holdout_psr", "median_tile_psr", "rotation_spread_degrees",
        "log_scale_spread", "frozen_h_corner_consistency", "corner_validity",
    )
    holdout_summary = {}
    for name in holdout_names:
        if name not in holdout:
            continue
        value = holdout[name]
        holdout_summary[name] = bool(value) if isinstance(value, (bool, np.bool_)) else int(value) if name in {"support", "secondary_window_support", "macro_regions"} else finite(value)
    diagnostics = {"search_top_k": tuple(search_summary), "selected_fit": fit_summary, "holdout": holdout_summary}
    return public, diagnostics


def _probe_g1r_at_truth(attacked_rgb: np.ndarray, detection_key: object, truth_attacked_to_canonical: np.ndarray) -> Mapping[str, object]:
    """Runner-only post-freeze diagnostic; its output cannot enter detection."""
    image = _require_rgb(attacked_rgb)
    truth = np.asarray(truth_attacked_to_canonical, dtype=np.float64)
    if truth.shape != (3, 3) or not np.isfinite(truth).all() or abs(float(np.linalg.det(truth))) <= 1e-12:
        raise ValueError("V4-G1R truth probe requires a finite invertible 3x3 H")
    canonical_to_attacked = np.linalg.inv(truth)
    canonical_to_attacked /= canonical_to_attacked[2, 2]
    angle, scale = _rotation_scale(canonical_to_attacked)
    rs = _similarity_h(angle, scale)
    translation = np.linalg.inv(rs) @ canonical_to_attacked
    expected_dx = int(round(float(translation[0, 2]) * (_SEARCH_SIZE - 1)))
    expected_dy = int(round(float(translation[1, 2]) * (_SEARCH_SIZE - 1)))

    ordinary = Image.fromarray((image * 255.0).round().clip(0, 255).astype(np.uint8), mode="RGB")
    resized = np.asarray(ordinary.resize((_SEARCH_SIZE, _SEARCH_SIZE), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    rectified, valid = _valid_warp(resized, rs)
    keys = derive_g1r_keys(detection_key)
    references = _search_macro_fields((_SEARCH_SIZE, _SEARCH_SIZE), keys["search"])
    surface, observed_band, reference_band, component_surfaces = _translation_surface(rectified, valid, references)
    peaks = _translation_peaks(surface)
    best_dx, best_dy = peaks[0] if peaks else (0, 0)

    def peak_metrics(dx: int, dy: int) -> Mapping[str, float]:
        aligned, aligned_valid = _align_translation(observed_band, valid, dx, dy)
        values = sorted(float(item[dy % item.shape[0], dx % item.shape[1]]) for item in component_surfaces)
        consensus = float(np.mean(values[1:-1])) if len(values) > 2 else float(np.mean(values)) if values else 0.0
        phase_peak = float(surface[dy % surface.shape[0], dx % surface.shape[1]])
        return {
            "translation_x": float(dx / (_SEARCH_SIZE - 1)),
            "translation_y": float(dy / (_SEARCH_SIZE - 1)),
            "translation_psr": _candidate_psr(surface, dx, dy),
            "ncc": _normalized_score(aligned, reference_band, aligned_valid),
            "joint_peak_over_rms": phase_peak / (float(np.sqrt(np.mean(surface * surface))) + 1e-12),
            "component_consensus": consensus,
        }

    expected_metrics = peak_metrics(expected_dx, expected_dy)
    best_metrics = peak_metrics(best_dx, best_dy)
    candidate = {
        "angle": angle,
        "scale": scale,
        "canonical_to_attacked": canonical_to_attacked,
        "rank": (0.0,),
        "ncc": expected_metrics["ncc"],
        "translation_psr": expected_metrics["translation_psr"],
        "phase_consistency": expected_metrics["joint_peak_over_rms"],
        "component_consensus": expected_metrics["component_consensus"],
    }
    fit = _fit_candidate(image, candidate, keys["fit"])
    prethreshold = tuple({
        "tile_id": int(item["tile_id"]),
        "best_correlation": item.get("best_correlation"),
        "margin": item.get("margin"),
        "accepted": bool(item.get("accepted", False)),
        "rejection": str(item.get("rejection", "invalid")),
    } for item in fit.get("prethreshold", ()))
    holdout_at_truth = _holdout_metrics(image, canonical_to_attacked, keys["validate"])
    holdout_after_fit = {} if fit.get("canonical_to_attacked") is None else _holdout_metrics(image, np.asarray(fit["canonical_to_attacked"], dtype=np.float64), keys["validate"])

    def safe_metrics(value: Mapping[str, object]) -> Mapping[str, object]:
        answer: dict[str, object] = {}
        for name, item in value.items():
            if isinstance(item, (bool, np.bool_)):
                answer[name] = bool(item)
            elif isinstance(item, (int, np.integer)):
                answer[name] = int(item)
            elif isinstance(item, (float, np.floating)):
                number = float(item)
                answer[name] = number if math.isfinite(number) else None
        return answer

    return {
        "search_at_truth": {"angle_degrees": float(angle), "scale": float(scale), **expected_metrics},
        "search_best_translation_at_truth_rs": best_metrics,
        "fit_at_truth": {
            "valid": bool(fit.get("valid", False)),
            "support": int(fit.get("support", 0)),
            "coverage": float(fit.get("coverage", 0.0)),
            "macro_regions": int(fit.get("macro_regions", 0)),
            "inlier_ratio": float(fit.get("inlier_ratio", 0.0)),
            "reprojection": None if not math.isfinite(float(fit.get("reprojection", math.inf))) else float(fit["reprojection"]),
            "condition": None if not math.isfinite(float(fit.get("condition", math.inf))) else float(fit["condition"]),
            "prethreshold_tiles": prethreshold,
        },
        "holdout_at_truth": safe_metrics(holdout_at_truth),
        "holdout_after_fit": safe_metrics(holdout_after_fit),
    }
