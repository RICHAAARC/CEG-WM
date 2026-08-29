"""Blind deterministic NumPy RGB proxy for Geometry-V4 P1 mechanism tests."""

from __future__ import annotations

import hashlib
import hmac
import math
from typing import Iterable

import numpy as np

from cegwm.protocol.geometry_v4 import (
    GEOMETRY_V4_METHOD_ID,
    GEOMETRY_V4_PROTOCOL_ID,
    GEOMETRY_V4_PROXY_WRITER_ID,
    GeometryV4Observation,
    derive_geometry_v4_key,
    reliability_is_reliable,
)
from cegwm.shared.keys import normalize_detection_key

_DIRECTIONS = (0.0, 45.0, 90.0, 135.0)
_SCALES = (8, 16, 32)
_CENTERS = (0.125, 0.375, 0.625, 0.875)
_GLOBAL_ENERGY = 0.40
_LOCAL_ENERGY = 0.60
_LUMA_RMS_TARGET = 1.5 / 255.0
_LUMA_RMS_CAP = 2.0 / 255.0
_LUMA_PEAK_CAP = 8.0 / 255.0
_REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _require_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 48:
        raise ValueError("Geometry-V4 proxy requires HxWx3 RGB with minimum side 48")
    if not np.all(np.isfinite(value)) or np.min(value) < 0.0 or np.max(value) > 1.0:
        raise ValueError("Geometry-V4 proxy RGB must be finite in [0,1]")
    return value


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb, _REC709, axes=([-1], [0]))


def _submaterial(key: bytes, label: str) -> bytes:
    return hmac.new(key, b"CEG-WM/geometry-v4/proxy/" + label.encode("ascii"), hashlib.sha256).digest()


def _phase_sign(key: bytes, label: str) -> tuple[float, float]:
    material = _submaterial(key, label)
    phase = int.from_bytes(material[:8], "big") / float(1 << 64) * 2.0 * math.pi
    sign = 1.0 if material[8] & 1 else -1.0
    return phase, sign


def _mean_unit(field: np.ndarray) -> np.ndarray:
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError("Geometry-V4 proxy anchor is degenerate")
    return centered / rms


def _global_fields(shape: tuple[int, int], key: bytes) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    x = xx / float(width)
    y = yy / float(height)
    all_components: list[np.ndarray] = []
    by_scale: dict[int, np.ndarray] = {}
    for cycles in _SCALES:
        scale_components: list[np.ndarray] = []
        for angle_deg in _DIRECTIONS:
            angle = math.radians(angle_deg)
            phase, sign = _phase_sign(key, f"global/{cycles}/{int(angle_deg)}")
            coordinate = x * math.cos(angle) + y * math.sin(angle)
            component = sign * np.sin(2.0 * math.pi * cycles * coordinate + phase)
            scale_components.append(component)
            all_components.append(component)
        by_scale[cycles] = _mean_unit(np.sum(scale_components, axis=0))
    return _mean_unit(np.sum(all_components, axis=0)), by_scale


def _local_field(shape: tuple[int, int], key: bytes) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    x = xx / float(max(1, width - 1))
    y = yy / float(max(1, height - 1))
    field = np.zeros(shape, dtype=np.float64)
    sigma = 0.055
    for row, cy in enumerate(_CENTERS):
        for column, cx in enumerate(_CENTERS):
            material = _submaterial(key, f"tile/{row}/{column}")
            phase = int.from_bytes(material[:8], "big") / float(1 << 64) * 2.0 * math.pi
            sign = 1.0 if material[8] & 1 else -1.0
            angle = math.radians(_DIRECTIONS[material[9] % len(_DIRECTIONS)])
            cycles = 12.0 + float(material[10] % 9)
            dx = x - cx
            dy = y - cy
            envelope = np.exp(-(np.square(dx) + np.square(dy)) / (2.0 * sigma * sigma))
            carrier = np.sin(2.0 * math.pi * cycles * (dx * math.cos(angle) + dy * math.sin(angle)) + phase)
            field += sign * envelope * carrier
    return _mean_unit(field)


def _anchor_fields(
    shape: tuple[int, int], key: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], dict[str, float | int]]:
    global_field, by_scale = _global_fields(shape, key)
    local_field = _local_field(shape, key)
    projection = float(np.mean(global_field * local_field))
    local_field = _mean_unit(local_field - projection * global_field)
    mixed = math.sqrt(_GLOBAL_ENERGY) * global_field + math.sqrt(_LOCAL_ENERGY) * local_field
    combined = _mean_unit(mixed)
    cross = float(np.mean(global_field * local_field))
    denominator = float(np.mean(np.square(mixed)))
    diagnostics: dict[str, float | int] = {
        "direction_count": len(_DIRECTIONS),
        "scale_count": len(_SCALES),
        "global_component_count": len(_DIRECTIONS) * len(_SCALES),
        "tile_count": len(_CENTERS) ** 2,
        "global_energy_fraction": float(_GLOBAL_ENERGY * np.mean(np.square(global_field)) / denominator),
        "local_energy_fraction": float(_LOCAL_ENERGY * np.mean(np.square(local_field)) / denominator),
        "global_local_cross": cross,
        "joint_rms": float(np.sqrt(np.mean(np.square(combined)))),
    }
    return combined, global_field, local_field, by_scale, diagnostics


def write_proxy(rgb: np.ndarray, detection_key: str | bytes) -> tuple[np.ndarray, dict[str, object]]:
    """Write the fixed keyed anchor and report only public budget diagnostics."""

    source = _require_rgb(rgb)
    root = normalize_detection_key(detection_key)
    geometry_key = derive_geometry_v4_key(root)
    combined, _, _, _, anchor_diagnostics = _anchor_fields(source.shape[:2], geometry_key)
    scale = min(_LUMA_RMS_TARGET, _LUMA_PEAK_CAP / float(np.max(np.abs(combined))))
    candidate = source + (combined * scale)[..., None]
    marked = np.clip(candidate, 0.0, 1.0)
    luma_delta = _luma(marked) - _luma(source)
    luma_rms = float(np.sqrt(np.mean(np.square(luma_delta))))
    luma_peak = float(np.max(np.abs(luma_delta)))
    if luma_rms > _LUMA_RMS_CAP + 1e-12 or luma_peak > _LUMA_PEAK_CAP + 1e-12:
        raise RuntimeError("Geometry-V4 proxy final-RGB budget exceeded")
    budget: dict[str, object] = {
        "method_id": GEOMETRY_V4_METHOD_ID,
        "protocol_id": GEOMETRY_V4_PROTOCOL_ID,
        "writer_id": GEOMETRY_V4_PROXY_WRITER_ID,
        "shape": tuple(int(item) for item in source.shape),
        "luma_rms": luma_rms,
        "luma_peak": luma_peak,
        "luma_rms_cap": _LUMA_RMS_CAP,
        "luma_peak_cap": _LUMA_PEAK_CAP,
        "within_budget": True,
        "anchor_energy": anchor_diagnostics,
    }
    return marked, budget


def normalized_phase_correlation(moving: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Return the cyclic shift of ``moving`` from normalized cross-power only."""

    left = np.asarray(moving, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.ndim != 2 or right.shape != left.shape or min(left.shape) < 4:
        raise ValueError("phase-correlation planes must have the same 2-D shape")
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    left_fft = np.fft.fft2(left)
    right_fft = np.fft.fft2(right)
    cross = left_fft * np.conj(right_fft)
    magnitude = np.abs(cross)
    valid = magnitude > np.finfo(np.float64).eps * max(1.0, float(np.max(magnitude)))
    normalized = np.zeros_like(cross)
    normalized[valid] = cross[valid] / magnitude[valid]
    surface = np.fft.ifft2(normalized).real
    peak_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(surface)), surface.shape))
    dy = peak_index[0] if peak_index[0] <= left.shape[0] // 2 else peak_index[0] - left.shape[0]
    dx = peak_index[1] if peak_index[1] <= left.shape[1] // 2 else peak_index[1] - left.shape[1]
    sidelobe = np.ones(surface.shape, dtype=bool)
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            sidelobe[(peak_index[0] + oy) % left.shape[0], (peak_index[1] + ox) % left.shape[1]] = False
    side = surface[sidelobe]
    side_mean = float(np.mean(side)) if side.size else 0.0
    side_std = float(np.std(side)) if side.size else 0.0
    psr = (float(surface[peak_index]) - side_mean) / (side_std + 1e-12)
    return {
        "shift_y": int(dy),
        "shift_x": int(dx),
        "PSR": float(max(0.0, psr)),
        "peak": float(surface[peak_index]),
        "surface": surface,
    }


def _bilinear_values(image: np.ndarray, x: np.ndarray, y: np.ndarray, fill: float) -> np.ndarray:
    height, width = image.shape[:2]
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = x - x0
    wy = y - y0
    if image.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
        valid = valid[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid, sampled, fill)


def _sample_h(image: np.ndarray, output_to_input: np.ndarray, fill: float) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    xn = xx / float(max(1, width - 1))
    yn = yy / float(max(1, height - 1))
    denominator = output_to_input[2, 0] * xn + output_to_input[2, 1] * yn + output_to_input[2, 2]
    source_x = (output_to_input[0, 0] * xn + output_to_input[0, 1] * yn + output_to_input[0, 2]) / denominator
    source_y = (output_to_input[1, 0] * xn + output_to_input[1, 1] * yn + output_to_input[1, 2]) / denominator
    return _bilinear_values(image, source_x * (width - 1), source_y * (height - 1), fill)


def _similarity_h(angle_deg: float, scale: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle) * float(scale)
    sine = math.sin(angle) * float(scale)
    centered = np.asarray(
        ((cosine, -sine, 0.5 - 0.5 * cosine + 0.5 * sine),
         (sine, cosine, 0.5 - 0.5 * sine - 0.5 * cosine),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    translation = np.asarray(((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)), dtype=np.float64)
    return centered @ translation


def apply_proxy_attack(rgb: np.ndarray, attack_id: str) -> tuple[np.ndarray, tuple[float, ...]]:
    """Apply one frozen public attack and return truth only to the runner."""

    source = _require_rgb(rgb)
    angle = 0.0
    scale = 1.0
    tx = 0.0
    ty = 0.0
    if attack_id == "identity":
        pass
    elif attack_id.startswith("rotation_"):
        angle = float(attack_id.removeprefix("rotation_"))
    elif attack_id.startswith("scale_"):
        scale = float(attack_id.removeprefix("scale_"))
    elif attack_id.startswith("translation_"):
        left, right = attack_id.removeprefix("translation_").split("_")
        tx, ty = float(left), float(right)
    elif attack_id.startswith("crop_rescale_"):
        retained = float(attack_id.removeprefix("crop_rescale_"))
        scale = 1.0 / retained
    elif attack_id == "compound_-7_0.9_+0.05_-0.05":
        angle, scale, tx, ty = -7.0, 0.9, 0.05, -0.05
    elif attack_id == "compound_+7_1.1_-0.05_+0.05":
        angle, scale, tx, ty = 7.0, 1.1, -0.05, 0.05
    else:
        raise ValueError(f"unknown Geometry-V4 P1 attack: {attack_id}")
    truth = _similarity_h(angle, scale, tx, ty)
    attacked = _sample_h(source, np.linalg.inv(truth), 0.5)
    return np.clip(attacked, 0.0, 1.0), tuple(float(value) for value in truth.reshape(-1))


def _spectral_magnitude(plane: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((plane - float(np.mean(plane))) * window))
    return np.log1p(np.abs(spectrum))


def _log_polar(magnitude: np.ndarray, angles: int = 180, radii: int = 64) -> tuple[np.ndarray, float]:
    height, width = magnitude.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    minimum_radius = 2.0
    maximum_radius = max(minimum_radius + 1e-6, min(center_x, center_y) - 1.0)
    log_step = math.log(maximum_radius / minimum_radius) / float(max(1, radii - 1))
    radius = minimum_radius * np.exp(np.arange(radii, dtype=np.float64) * log_step)
    theta = np.arange(angles, dtype=np.float64) * math.pi / float(angles)
    sample_x = center_x + np.cos(theta)[:, None] * radius[None, :]
    sample_y = center_y + np.sin(theta)[:, None] * radius[None, :]
    return _bilinear_values(magnitude, sample_x, sample_y, 0.0), log_step


def _coarse_rotation_scale(observed: np.ndarray, reference: np.ndarray) -> tuple[float, float, dict[str, float]]:
    observed_lp, log_step = _log_polar(_spectral_magnitude(observed))
    reference_lp, _ = _log_polar(_spectral_magnitude(reference))
    result = normalized_phase_correlation(observed_lp, reference_lp)
    rotation = float(result["shift_y"]) * 180.0 / observed_lp.shape[0]
    rotation = ((rotation + 90.0) % 180.0) - 90.0
    rotation = float(np.clip(rotation, -15.0, 15.0))
    log_scale = -float(result["shift_x"]) * log_step
    scale = float(np.clip(math.exp(log_scale), 0.82, 1.22))
    return rotation, scale, {"rotation_deg": rotation, "scale": scale, "PSR": float(result["PSR"])}


def _rectify_rs(attacked: np.ndarray, angle: float, scale: float) -> np.ndarray:
    fill = float(np.median(attacked))
    return _sample_h(attacked, _similarity_h(angle, scale), fill)


def _rs_score(attacked: np.ndarray, reference: np.ndarray, angle: float, scale: float) -> float:
    rectified = _luma(_rectify_rs(attacked, angle, scale))
    return float(normalized_phase_correlation(rectified, reference)["PSR"])


def _refine_rotation_scale(
    attacked: np.ndarray,
    global_reference: np.ndarray,
    by_scale: dict[int, np.ndarray],
    coarse_angle: float,
    coarse_scale: float,
) -> tuple[float, float, list[tuple[float, float]]]:
    candidates: list[tuple[float, float, float]] = []
    for rotation_offset in (-4.0, 0.0, 4.0):
        for log_scale_offset in (-0.04, 0.0, 0.04):
            angle = float(np.clip(coarse_angle + rotation_offset, -16.0, 16.0))
            scale = float(np.clip(coarse_scale * math.exp(log_scale_offset), 0.80, 1.25))
            candidates.append((_rs_score(attacked, global_reference, angle, scale), angle, scale))
    _, angle, scale = max(candidates, key=lambda item: (item[0], -abs(item[1]), -abs(math.log(item[2]))))
    per_scale: list[tuple[float, float]] = []
    for cycles in _SCALES:
        local_candidates: list[tuple[float, float, float]] = []
        for rotation_offset in (-1.0, 0.0, 1.0):
            for log_scale_offset in (-0.01, 0.0, 0.01):
                candidate_angle = angle + rotation_offset
                candidate_scale = scale * math.exp(log_scale_offset)
                local_candidates.append(
                    (_rs_score(attacked, by_scale[cycles], candidate_angle, candidate_scale), candidate_angle, candidate_scale)
                )
        _, selected_angle, selected_scale = max(
            local_candidates, key=lambda item: (item[0], -abs(item[1] - angle), -abs(math.log(item[2] / scale)))
        )
        per_scale.append((float(selected_angle), float(selected_scale)))
    return float(angle), float(scale), per_scale


def _patch(plane: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray | None:
    if center_x - radius < 0 or center_y - radius < 0 or center_x + radius >= plane.shape[1] or center_y + radius >= plane.shape[0]:
        return None
    return plane[center_y - radius : center_y + radius + 1, center_x - radius : center_x + radius + 1]


def _normalized_patch_score(left: np.ndarray, right: np.ndarray) -> float:
    a = left - float(np.mean(left))
    b = right - float(np.mean(right))
    denominator = float(np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b))))
    if denominator <= 1e-12:
        return -1.0
    return float(np.sum(a * b) / denominator)


def _match_tiles(
    rectified_plane: np.ndarray,
    local_reference: np.ndarray,
    shift_x: int,
    shift_y: int,
    h_rs: np.ndarray,
) -> list[dict[str, object]]:
    height, width = rectified_plane.shape
    patch_radius = max(4, int(round(min(height, width) / 16.0)))
    search_radius = max(3, int(round(4.0 * min(height, width) / 64.0)))
    matches: list[dict[str, object]] = []
    for row, cy in enumerate(_CENTERS):
        for column, cx in enumerate(_CENTERS):
            canonical_x = int(round(cx * (width - 1)))
            canonical_y = int(round(cy * (height - 1)))
            template = _patch(local_reference, canonical_x, canonical_y, patch_radius)
            if template is None:
                continue
            predicted_x = canonical_x + shift_x
            predicted_y = canonical_y + shift_y
            candidates: list[tuple[float, int, int]] = []
            for oy in range(-search_radius, search_radius + 1):
                for ox in range(-search_radius, search_radius + 1):
                    observed = _patch(rectified_plane, predicted_x + ox, predicted_y + oy, patch_radius)
                    if observed is not None:
                        candidates.append((_normalized_patch_score(observed, template), predicted_x + ox, predicted_y + oy))
            if not candidates:
                continue
            best = max(candidates, key=lambda item: (item[0], -abs(item[1] - predicted_x) - abs(item[2] - predicted_y)))
            separated = [
                item for item in candidates if abs(item[1] - best[1]) > 1 or abs(item[2] - best[2]) > 1
            ]
            second = max((item[0] for item in separated), default=-1.0)
            margin = best[0] - second
            if best[0] < 0.42 or margin < 0.025:
                continue
            rectified_point = np.asarray((best[1] / (width - 1), best[2] / (height - 1), 1.0), dtype=np.float64)
            attacked_point = h_rs @ rectified_point
            attacked_point /= attacked_point[2]
            matches.append(
                {
                    "tile": (row, column),
                    "canonical": (float(cx), float(cy)),
                    "attacked": (float(attacked_point[0]), float(attacked_point[1])),
                    "correlation": float(best[0]),
                    "margin": float(margin),
                }
            )
    return matches


def _fit_similarity(matches: Iterable[dict[str, object]]) -> tuple[np.ndarray | None, np.ndarray, float]:
    material = list(matches)
    if len(material) < 2:
        return None, np.full(len(material), 1.0, dtype=np.float64), 1e12
    rows: list[list[float]] = []
    targets: list[float] = []
    for match in material:
        u, v = match["canonical"]  # type: ignore[misc]
        x, y = match["attacked"]  # type: ignore[misc]
        rows.extend(([u, -v, 1.0, 0.0], [v, u, 0.0, 1.0]))
        targets.extend((x, y))
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    parameters, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    a, b, tx, ty = (float(item) for item in parameters)
    estimate = np.asarray(((a, -b, tx), (b, a, ty), (0.0, 0.0, 1.0)), dtype=np.float64)
    residuals: list[float] = []
    for match in material:
        u, v = match["canonical"]  # type: ignore[misc]
        x, y = match["attacked"]  # type: ignore[misc]
        predicted = estimate @ np.asarray((u, v, 1.0), dtype=np.float64)
        residuals.append(math.hypot(float(predicted[0]) - x, float(predicted[1]) - y) / math.sqrt(2.0))
    return estimate, np.asarray(residuals, dtype=np.float64), float(np.linalg.cond(design))


def _corners(homography: np.ndarray) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        point = homography @ np.asarray((x, y, 1.0), dtype=np.float64)
        result.append((float(point[0] / point[2]), float(point[1] / point[2])))
    return tuple(result)


def _valid_corners(corners: tuple[tuple[float, float], ...]) -> bool:
    if any(not math.isfinite(value) or value < -1.0 or value > 2.0 for point in corners for value in point):
        return False
    crosses = []
    for index in range(4):
        a = corners[index]
        b = corners[(index + 1) % 4]
        c = corners[(index + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    area = sum(
        corners[index][0] * corners[(index + 1) % 4][1]
        - corners[(index + 1) % 4][0] * corners[index][1]
        for index in range(4)
    )
    return area > 0.0 and all(value > 0.0 for value in crosses)


def _macro_regions(matches: Iterable[dict[str, object]]) -> int:
    regions = set()
    for match in matches:
        x, y = match["canonical"]  # type: ignore[misc]
        regions.add((int(y >= 0.5), int(x >= 0.5)))
    return len(regions)


def _aggregate_reliability(metrics: dict[str, object], mean_tile_correlation: float) -> float:
    psr = float(metrics["PSR"])
    support = int(metrics["support"])
    inlier_ratio = float(metrics["inlier_ratio"])
    coverage = float(metrics["spatial_coverage"])
    macro = int(metrics["macro_regions"])
    reprojection = float(metrics["reprojection_rms_diagonal"])
    condition = float(metrics["condition_number"])
    rotation_spread = float(metrics["cross_scale_rotation_spread_deg"])
    scale_spread = float(metrics["cross_scale_log_scale_spread"])
    qualities = (
        min(1.0, psr / 8.0),
        min(1.0, support / 6.0),
        min(1.0, inlier_ratio / 0.5),
        min(1.0, coverage / 0.75),
        min(1.0, macro / 3.0),
        min(1.0, 0.02 / max(reprojection, 1e-12)),
        min(1.0, 1e4 / max(condition, 1.0)),
        min(1.0, 2.0 / max(rotation_spread, 1e-12)),
        min(1.0, 0.03 / max(scale_spread, 1e-12)),
        max(0.0, min(1.0, (mean_tile_correlation - 0.20) / 0.40)),
    )
    return float(np.mean(qualities))


def detect_proxy(attacked: np.ndarray, detection_key: str | bytes) -> dict[str, object]:
    """Recover a fail-closed similarity using attacked RGB and key only."""

    observed_rgb = _require_rgb(attacked)
    root = normalize_detection_key(detection_key)
    geometry_key = derive_geometry_v4_key(root)
    combined, global_reference, local_reference, by_scale, _ = _anchor_fields(observed_rgb.shape[:2], geometry_key)
    observed_plane = _luma(observed_rgb)
    coarse_angle, coarse_scale, coarse_record = _coarse_rotation_scale(observed_plane, global_reference)
    angle, scale, per_scale = _refine_rotation_scale(
        observed_rgb, global_reference, by_scale, coarse_angle, coarse_scale
    )
    h_rs = _similarity_h(angle, scale)
    rectified_rgb = _rectify_rs(observed_rgb, angle, scale)
    rectified_plane = _luma(rectified_rgb)
    translation = normalized_phase_correlation(rectified_plane, combined)
    shift_x = int(translation["shift_x"])
    shift_y = int(translation["shift_y"])
    matches = _match_tiles(rectified_plane, local_reference, shift_x, shift_y, h_rs)
    fitted_h, residuals, condition = _fit_similarity(matches)
    corners = _corners(fitted_h) if fitted_h is not None else ()
    corner_validity = fitted_h is not None and _valid_corners(corners)
    support = len(matches)
    inlier_ratio = float(np.mean(residuals <= 0.02)) if support else 0.0
    reprojection = float(np.sqrt(np.mean(np.square(residuals)))) if support else 1.0
    macro_regions = _macro_regions(matches)
    spatial_coverage = macro_regions / 4.0
    rotations = np.asarray([item[0] for item in per_scale], dtype=np.float64)
    log_scales = np.log(np.asarray([item[1] for item in per_scale], dtype=np.float64))
    rotation_spread = float(np.max(rotations) - np.min(rotations))
    log_scale_spread = float(np.max(log_scales) - np.min(log_scales))
    mean_tile_correlation = float(np.mean([float(item["correlation"]) for item in matches])) if matches else 0.0
    metrics: dict[str, object] = {
        "PSR": float(translation["PSR"]),
        "support": support,
        "inlier_ratio": inlier_ratio,
        "spatial_coverage": spatial_coverage,
        "macro_regions": macro_regions,
        "reprojection_rms_diagonal": reprojection,
        "condition_number": condition,
        "cross_scale_rotation_spread_deg": rotation_spread,
        "cross_scale_log_scale_spread": log_scale_spread,
        "corner_validity": corner_validity,
        "aggregate_reliability": 0.0,
    }
    aggregate = _aggregate_reliability(metrics, mean_tile_correlation)
    metrics["aggregate_reliability"] = aggregate
    reliable = reliability_is_reliable(metrics)
    status = "RELIABLE" if reliable else "UNRELIABLE"
    h_tuple = tuple(float(value) for value in fitted_h.reshape(-1)) if fitted_h is not None else None
    if corner_validity:
        observation = GeometryV4Observation(h_tuple, corners, support, aggregate, status)
    else:
        observation = GeometryV4Observation(None, (), support, aggregate, "UNRELIABLE")
    diagnostics: dict[str, object] = dict(metrics)
    diagnostics.update(
        {
            "coarse_log_polar": coarse_record,
            "rotation_deg": angle,
            "scale": scale,
            "translation_pixels": (shift_x, shift_y),
            "mean_tile_correlation": mean_tile_correlation,
            "matches": tuple(matches),
            "cross_scale_estimates": tuple((float(a), float(s)) for a, s in per_scale),
        }
    )
    return {
        "method_id": GEOMETRY_V4_METHOD_ID,
        "protocol_id": GEOMETRY_V4_PROTOCOL_ID,
        "H_hat": observation.H_hat,
        "corners_hat": observation.corners_hat,
        "support": observation.support,
        "reliability": observation.reliability,
        "status": observation.status,
        "diagnostics": diagnostics,
    }
