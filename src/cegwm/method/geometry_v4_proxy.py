"""Blind deterministic NumPy RGB proxy for Geometry-V4 P1 mechanism tests."""

from __future__ import annotations

import hashlib
import hmac
import itertools
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
_SCALES = (8, 16, 24)
_CENTERS = (0.125, 0.375, 0.625, 0.875)
_LOCAL_MIN_VALID_FRACTION = 0.60
_GLOBAL_ENERGY = 0.40
_LOCAL_ENERGY = 0.60
_LUMA_RMS_TARGET = 1.5 / 255.0
_LUMA_RMS_CAP = 2.0 / 255.0
_LUMA_PEAK_CAP = 8.0 / 255.0
_RS_SCALE_MIN = 0.65
_RS_SCALE_MAX = 1.55
_REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_MAGNITUDE_RAW_LEVELS = (0.65, 0.85, 1.15, 1.35)
_MAGNITUDE_RMS = math.sqrt(sum(value * value for value in _MAGNITUDE_RAW_LEVELS) / 4.0)
_MAGNITUDE_LEVELS = tuple(value / _MAGNITUDE_RMS for value in _MAGNITUDE_RAW_LEVELS)
_MAGNITUDE_OPTIMAL_RADIAL_RATIO = 1.5847750865051904


def _magnitude_worst_radial_ratio(code: tuple[tuple[int, ...], ...]) -> float:
    return max(
        max(
            sum(_MAGNITUDE_LEVELS[code[scale_index][direction_index]] ** 2 for direction_index in range(4) if (scale_index + direction_index) % 3 == group)
            for scale_index in range(3)
        )
        / min(
            sum(_MAGNITUDE_LEVELS[code[scale_index][direction_index]] ** 2 for direction_index in range(4) if (scale_index + direction_index) % 3 == group)
            for scale_index in range(3)
        )
        for group in range(3)
    )


def _quadratic_peak_delta(left: float, center: float, right: float) -> float | None:
    """Return a bounded three-point peak vertex, or fail closed."""

    if not all(math.isfinite(value) for value in (left, center, right)) or not (center > left and center > right):
        return None
    curvature = left - 2.0 * center + right
    if curvature >= 0.0:
        return None
    delta = 0.5 * (left - right) / curvature
    return float(delta) if -0.5 <= delta <= 0.5 else None


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


def _magnitude_codebook() -> tuple[tuple[tuple[int, ...], ...], ...]:
    rows = tuple(itertools.permutations(range(4)))
    safe = []
    for code in itertools.product(rows, repeat=3):
        radial_optimal = abs(_magnitude_worst_radial_ratio(code) - _MAGNITUDE_OPTIMAL_RADIAL_RATIO) <= 1e-12
        if radial_optimal and all(
            sum(code[scale][direction] != code[(scale + ds) % 3][(direction + dd) % 4] for scale in range(3) for direction in range(4)) >= 6
            for ds in range(3) for dd in range(4) if (ds, dd) != (0, 0)
        ):
            safe.append(code)
    return tuple(safe)


_MAGNITUDE_CODEBOOK = _magnitude_codebook()


def _magnitude_code(key: bytes) -> tuple[tuple[int, ...], ...]:
    if not _MAGNITUDE_CODEBOOK:
        raise RuntimeError("Geometry-V4 magnitude codebook is empty")
    material = _submaterial(key, "global/magnitude-codebook-index/v1")
    return _MAGNITUDE_CODEBOOK[int.from_bytes(material[:8], "big") % len(_MAGNITUDE_CODEBOOK)]


def _mean_unit(field: np.ndarray) -> np.ndarray:
    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError("Geometry-V4 proxy anchor is degenerate")
    return centered / rms


def _unit_global_components(shape: tuple[int, int], key: bytes) -> dict[tuple[int, float], np.ndarray]:
    """Build only fixed keyed phase/sign unit carriers; no RGB enters this path."""

    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    x = xx / float(width)
    y = yy / float(height)
    by_component: dict[tuple[int, float], np.ndarray] = {}
    for cycles in _SCALES:
        for angle_deg in _DIRECTIONS:
            angle = math.radians(angle_deg)
            phase, sign = _phase_sign(key, f"global/{cycles}/{int(angle_deg)}")
            coordinate = x * math.cos(angle) + y * math.sin(angle)
            by_component[(cycles, angle_deg)] = _mean_unit(sign * np.sin(2.0 * math.pi * cycles * coordinate + phase))
    return by_component


def _weighted_global_components(
    unit_components: dict[tuple[int, float], np.ndarray], code: tuple[tuple[int, ...], ...]
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, float], np.ndarray]]:
    all_components: list[np.ndarray] = []
    by_scale: dict[int, np.ndarray] = {}
    by_component: dict[tuple[int, float], np.ndarray] = {}
    for scale_index, cycles in enumerate(_SCALES):
        scale_components: list[np.ndarray] = []
        for direction_index, angle_deg in enumerate(_DIRECTIONS):
            component = unit_components[(cycles, angle_deg)] * _MAGNITUDE_LEVELS[code[scale_index][direction_index]]
            by_component[(cycles, angle_deg)] = component
            scale_components.append(component)
            all_components.append(component)
        by_scale[cycles] = _mean_unit(np.sum(scale_components, axis=0))
    return _mean_unit(np.sum(all_components, axis=0)), by_scale, by_component


def _global_fields(shape: tuple[int, int], key: bytes) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, float], np.ndarray]]:
    unit_components = _unit_global_components(shape, key)
    return _weighted_global_components(unit_components, _magnitude_code(key))


def _constellation_groups(by_component: dict[tuple[int, float], np.ndarray]) -> tuple[tuple[np.ndarray, tuple[tuple[int, float], ...]], ...]:
    groups = []
    for group in range(3):
        identities = tuple((cycles, direction) for s, cycles in enumerate(_SCALES) for d, direction in enumerate(_DIRECTIONS) if (s + d) % 3 == group)
        groups.append((_mean_unit(sum((by_component[item] for item in identities))), identities))
    return tuple(groups)


def _angle_orbits_45(raw_angle: float) -> tuple[float, ...]:
    return tuple(_wrap_rotation_180(raw_angle + 45.0 * q) for q in range(4))


def _orbit_assignments() -> tuple[tuple[int, int, int], ...]:
    return tuple(itertools.product(range(4), repeat=3))


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], dict[tuple[int, float], np.ndarray], dict[str, float | int]]:
    global_field, by_scale, by_component = _global_fields(shape, key)
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
        "magnitude_level_count": len(_MAGNITUDE_LEVELS),
        "magnitude_level_range": (min(_MAGNITUDE_LEVELS), max(_MAGNITUDE_LEVELS)),
        "magnitude_min_gap": min(right - left for left, right in zip(_MAGNITUDE_LEVELS, _MAGNITUDE_LEVELS[1:])),
        "magnitude_composition_per_level": 3,
        "magnitude_orbit_min_hamming": 6,
        "magnitude_codebook_size": len(_MAGNITUDE_CODEBOOK),
    }
    return combined, global_field, local_field, by_scale, by_component, diagnostics


def write_proxy(rgb: np.ndarray, detection_key: str | bytes) -> tuple[np.ndarray, dict[str, object]]:
    """Write the fixed keyed anchor and report only public budget diagnostics."""

    source = _require_rgb(rgb)
    root = normalize_detection_key(detection_key)
    geometry_key = derive_geometry_v4_key(root)
    combined, _, _, _, _, anchor_diagnostics = _anchor_fields(source.shape[:2], geometry_key)
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
    """Apply one attack and return attacked-to-canonical truth to the runner."""

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
    canonical_to_attacked = _similarity_h(angle, scale, tx, ty)
    attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
    attacked = _sample_h(source, attacked_to_canonical, 0.5)
    return np.clip(attacked, 0.0, 1.0), tuple(float(value) for value in attacked_to_canonical.reshape(-1))


def rectify_proxy(attacked: np.ndarray, h_attacked_to_canonical: tuple[float, ...]) -> np.ndarray:
    """Inverse-sample attacked RGB into the canonical frame using public H_hat."""

    source = _require_rgb(attacked)
    homography = np.asarray(h_attacked_to_canonical, dtype=np.float64)
    if homography.shape != (9,) or not np.all(np.isfinite(homography)):
        raise ValueError("Geometry-V4 public H_hat must contain nine finite values")
    homography = homography.reshape(3, 3)
    if abs(float(np.linalg.det(homography))) <= 1e-12:
        raise ValueError("Geometry-V4 public H_hat must be non-singular")
    return np.clip(_sample_h(source, np.linalg.inv(homography), float(np.median(source))), 0.0, 1.0)


def _spectral_magnitude(plane: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((plane - float(np.mean(plane))) * window))
    return np.log1p(np.abs(spectrum))


def _sparse_constellation_diagnostic(
    attacked_plane: np.ndarray, by_component: dict[tuple[int, float], np.ndarray]
) -> tuple[dict[str, object], ...]:
    """Blind sparse-carrier R/S surfaces with independent fail-closed group raws."""

    plane = np.asarray(attacked_plane, dtype=np.float64)
    height, width = plane.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((plane - float(np.mean(plane))) * window)))
    rotations = tuple(-16.0 + 0.5 * index for index in range(65))
    logs = _zero_anchored_log_grid()

    template = np.outer(np.hanning(11), np.hanning(11))

    records = []
    for reference, identities in _constellation_groups(by_component):
        scores = []
        for rotation in rotations:
            theta = math.radians(rotation)
            for log_scale in logs:
                scale = math.exp(log_scale)
                vectors = []
                component_scores = []
                for cycles, direction in identities:
                    angle = math.radians(direction) + theta
                    fy = height / 2.0 + scale * cycles * math.sin(angle)
                    fx = width / 2.0 + scale * cycles * math.cos(angle)
                    positive = _toroidal_bilinear_patch(spectrum, fy, fx)
                    negative = _toroidal_bilinear_patch(spectrum, height - fy, width - fx)[::-1, ::-1]
                    observed = positive if (abs((2.0 * fy) % height) < 1e-9 and abs((2.0 * fx) % width) < 1e-9) else 0.5 * (positive + negative)
                    axis = np.arange(-5, 6, dtype=np.float64)
                    lobe = np.outer(_hann_dtft_lobe(axis), _hann_dtft_lobe(axis))
                    weight = float(np.sqrt(np.mean(np.square(by_component[(cycles, direction)]))))
                    pair = _projected_glrt_vectors(observed, lobe * weight)
                    vectors.append(pair)
                    component_scores.append(_group_glrt_score((pair,)))
                completeness = float(np.prod(np.maximum(component_scores, 0.0)) ** (1.0 / len(component_scores)))
                scores.append((_group_glrt_score(vectors) * completeness, rotation, log_scale))
        scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = scores[0]
        boundary = bool(
            best[1] in (rotations[0], rotations[-1])
            or best[2] in (logs[0], logs[-1])
        )
        records.append(
            {
                "identities": identities,
                "raw_rotation_deg": best[1],
                "raw_log_scale": best[2],
                "score": best[0],
                "margin": best[0] - scores[1][0],
                "boundary": boundary,
                "valid": math.isfinite(best[0]) and best[0] - scores[1][0] > 1e-9 and not boundary,
            }
        )
    return tuple(records)


def _toroidal_bilinear_patch(values: np.ndarray, center_y: float, center_x: float, radius: int = 5) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(center_y + offsets, center_x + offsets, indexing="ij")
    y0 = np.floor(yy).astype(int) % values.shape[0]
    x0 = np.floor(xx).astype(int) % values.shape[1]
    y1, x1 = (y0 + 1) % values.shape[0], (x0 + 1) % values.shape[1]
    dy, dx = yy - np.floor(yy), xx - np.floor(xx)
    return (1 - dy) * ((1 - dx) * values[y0, x0] + dx * values[y0, x1]) + dy * ((1 - dx) * values[y1, x0] + dx * values[y1, x1])


def _hann_dtft_lobe(offsets: np.ndarray, frequency_offset: float = 0.0) -> np.ndarray:
    sample = np.arange(64, dtype=np.float64)
    window = np.hanning(64)
    return np.abs(np.sum(window[:, None] * np.exp(-2j * math.pi * sample[:, None] * (offsets[None, :] - frequency_offset) / 64.0), axis=0))


def _projected_glrt_vectors(observed: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = [(dy, dx) for dy in range(-5, 6) for dx in range(-5, 6)]
    design = np.asarray([[1.0, dx, dy] for dy, dx in offsets], dtype=np.float64)
    annulus = np.asarray([3 <= max(abs(dy), abs(dx)) <= 5 for dy, dx in offsets])
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    t = np.asarray(template, dtype=np.float64).reshape(-1)
    y -= design @ np.linalg.lstsq(design[annulus], y[annulus], rcond=None)[0]
    t -= design @ np.linalg.lstsq(design[annulus], t[annulus], rcond=None)[0]
    mad = float(np.median(np.abs(y[annulus] - np.median(y[annulus])))) + 1e-12
    return y / mad, t


def _group_glrt_score(vectors: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    pairs = tuple(vectors)
    if not pairs:
        return -math.inf
    y = np.concatenate([pair[0] for pair in pairs])
    t = np.concatenate([pair[1] for pair in pairs])
    numerator = max(0.0, float(np.dot(y, t)))
    return numerator * numerator / (float(np.dot(t, t)) * (float(np.dot(y, y)) + 1e-12))


def _zero_anchored_log_grid() -> tuple[float, ...]:
    values = {math.log(_RS_SCALE_MIN), math.log(_RS_SCALE_MAX), 0.0}
    for index in range(-100, 101):
        value = index * 0.01
        if math.log(_RS_SCALE_MIN) <= value <= math.log(_RS_SCALE_MAX):
            values.add(value)
    return tuple(sorted(values))


def _log_polar(magnitude: np.ndarray, angles: int = 360, radii: int = 256) -> tuple[np.ndarray, float]:
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


def _mixed_logpolar_correlation(
    observed_lp: np.ndarray,
    reference_lp: np.ndarray,
    log_step: float,
) -> dict[str, object]:
    """Coherent angular correlation over explicit, non-wrapping radial overlaps."""

    observed = np.asarray(observed_lp, dtype=np.float64)
    reference = np.asarray(reference_lp, dtype=np.float64)
    if observed.ndim != 2 or observed.shape != reference.shape:
        raise ValueError("Geometry-V4 mixed log-polar inputs must be equally-shaped planes")
    angles, radii = observed.shape
    if angles != 360 or radii != 256 or not math.isfinite(float(log_step)) or log_step <= 0.0:
        raise ValueError("Geometry-V4 mixed log-polar correlation requires fixed 360x256 finite sampling")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reference)):
        return {"surface": np.full((angles, 2 * radii - 1), np.nan), "global": {"valid": False, "reason": "nonfinite"}, "measured": {"valid": False, "reason": "nonfinite"}, "top8": ()}
    observed_fft = np.fft.fft(observed - np.mean(observed, axis=0, keepdims=True), axis=0)
    reference_fft = np.fft.fft(reference - np.mean(reference, axis=0, keepdims=True), axis=0)
    radial_shifts = np.arange(-(radii - 1), radii, dtype=int)
    coherence_surface = np.zeros((angles, radial_shifts.size), dtype=np.float64)
    overlap_fraction = np.zeros(radial_shifts.size, dtype=np.float64)
    for column, shift_x in enumerate(radial_shifts):
        if shift_x >= 0:
            left = observed_fft[:, shift_x:]
            right = reference_fft[:, : radii - shift_x]
        else:
            left = observed_fft[:, : radii + shift_x]
            right = reference_fft[:, -shift_x:]
        denominator = float(np.sqrt(np.sum(np.abs(left) ** 2) * np.sum(np.abs(right) ** 2)))
        overlap_fraction[column] = left.shape[1] / float(radii)
        coherence = np.zeros(angles, dtype=np.complex128)
        if denominator > 1e-12:
            coherence = np.sum(left * np.conj(right), axis=1) / denominator
        coherence_surface[:, column] = np.real(np.fft.ifft(coherence))
    surface = coherence_surface * np.sqrt(overlap_fraction)[None, :]
    angular_shifts = np.arange(angles, dtype=int)
    angular_shifts = np.where(angular_shifts <= angles // 2, angular_shifts, angular_shifts - angles)
    finite = np.isfinite(surface)
    if not np.any(finite):
        return {
            "surface": surface,
            "global": {"valid": False, "reason": "nonfinite"},
            "measured": {"valid": False, "reason": "nonfinite"},
            "top8": (),
        }

    def describe(index_y: int, index_x: int, score: float) -> dict[str, object]:
        shift_y = int(angular_shifts[index_y])
        shift_x = int(radial_shifts[index_x])
        return {
            "index": (int(index_y), int(index_x)),
            "shift_y": shift_y,
            "shift_x": shift_x,
            "rotation_deg": _wrap_rotation_180(shift_y * 180.0 / float(angles)),
            "log_scale": -shift_x * float(log_step),
            "score": float(score),
        }

    def ranked(mask: np.ndarray) -> list[tuple[float, int, int]]:
        return sorted(
            ((float(surface[row, column]), int(angular_shifts[row]), int(radial_shifts[column])) for row, column in zip(*np.where(mask))),
            key=lambda item: (-item[0], item[1], item[2]),
        )

    unweighted_finite = np.isfinite(coherence_surface)
    global_score, global_dy, global_dx = sorted(
        ((float(coherence_surface[row, column]), int(angular_shifts[row]), int(radial_shifts[column])) for row, column in zip(*np.where(unweighted_finite))),
        key=lambda item: (-item[0], item[1], item[2]),
    )[0]
    global_index = (int(np.where(angular_shifts == global_dy)[0][0]), int(np.where(radial_shifts == global_dx)[0][0]))
    global_top = describe(global_index[0], global_index[1], global_score)
    global_top["score_identity"] = "unweighted_coherence_diagnostic_only"
    log_min = math.log(_RS_SCALE_MIN)
    log_max = math.log(_RS_SCALE_MAX)
    angular_in_domain = np.abs(angular_shifts) <= 32  # +/-16 degrees at 0.5 degree samples.
    radial_logs = -radial_shifts.astype(np.float64) * float(log_step)
    radial_in_domain = (radial_logs >= log_min - 1e-12) & (radial_logs <= log_max + 1e-12)
    domain_mask = angular_in_domain[:, None] & radial_in_domain[None, :] & finite
    if not np.any(domain_mask):
        return {"surface": surface, "global": global_top | {"valid": True}, "measured": {"valid": False, "reason": "empty_domain"}, "top8": ()}
    measured_score, measured_dy, measured_dx = ranked(domain_mask)[0]
    measured_index = (int(np.where(angular_shifts == measured_dy)[0][0]), int(np.where(radial_shifts == measured_dx)[0][0]))
    eligible_scores = surface[domain_mask]
    flat = bool(np.ptp(eligible_scores) <= 1e-12)
    measured_shift_y = int(angular_shifts[int(measured_index[0])])
    measured_shift_x = int(radial_shifts[int(measured_index[1])])
    radial_domain_shifts = radial_shifts[radial_in_domain]
    boundary = bool(
        abs(measured_shift_y) == 32
        or measured_shift_x in {int(np.min(radial_domain_shifts)), int(np.max(radial_domain_shifts))}
    )
    radial_column = int(measured_index[1])
    neighbor_scores = (
        float(surface[measured_index[0], radial_column - 1]) if radial_column > 0 else math.nan,
        measured_score,
        float(surface[measured_index[0], radial_column + 1]) if radial_column + 1 < surface.shape[1] else math.nan,
    )
    radial_delta = _quadratic_peak_delta(*neighbor_scores)
    measured = describe(int(measured_index[0]), int(measured_index[1]), measured_score)
    refined_shift_x = float(measured_shift_x + radial_delta) if radial_delta is not None else math.nan
    measured["log_scale"] = -refined_shift_x * float(log_step) if math.isfinite(refined_shift_x) else math.nan
    sidelobes = np.ones_like(surface, dtype=bool)
    angular_delta = np.abs(angular_shifts - measured_shift_y)
    angular_distance = np.minimum(angular_delta, angles - angular_delta)
    sidelobes[(angular_distance <= 2.0)[:, None] & (np.abs(radial_shifts - measured_shift_x) <= 2)[None, :]] = False
    sidelobe_values = surface[sidelobes & finite]
    sidelobe_std = float(np.std(sidelobe_values)) if sidelobe_values.size else 0.0
    psr = float((measured_score - float(np.mean(sidelobe_values))) / sidelobe_std) if sidelobe_std > 1e-12 else math.inf
    top8: list[dict[str, object]] = []
    nms_mask = domain_mask.copy()
    while len(top8) < 8 and np.any(nms_mask):
        score, dy, dx = ranked(nms_mask)[0]
        row = int(np.where(angular_shifts == dy)[0][0])
        column = int(np.where(radial_shifts == dx)[0][0])
        top8.append(describe(row, column, score))
        angular_delta = np.abs(angular_shifts - dy)
        angular_near = np.minimum(angular_delta, angles - angular_delta) <= 2.0
        radial_near = np.abs(radial_shifts - dx) <= 2
        nms_mask[angular_near[:, None] & radial_near[None, :]] = False
    measured.update({"valid": bool(math.isfinite(measured_score) and math.isfinite(psr) and radial_delta is not None and not boundary and not flat), "boundary": boundary, "flat": flat, "PSR": psr, "overlap_fraction": float(overlap_fraction[measured_index[1]]), "unweighted_coherence": float(coherence_surface[measured_index]), "score_identity": "support_weighted_primary", "integer_shift_x": measured_shift_x, "radial_delta": radial_delta, "radial_triplet_scores": neighbor_scores})
    global_top["valid"] = bool(math.isfinite(float(global_top["score"])))
    return {"surface": surface, "global": global_top, "measured": measured, "top8": tuple(top8)}


def _coarse_rotation_scale(observed: np.ndarray, reference: np.ndarray) -> tuple[float, float, dict[str, float]]:
    observed_lp, log_step = _log_polar(_spectral_magnitude(observed))
    reference_lp, _ = _log_polar(_spectral_magnitude(reference))
    result = normalized_phase_correlation(observed_lp, reference_lp)
    raw_rotation = _wrap_rotation_180(float(result["shift_y"]) * 180.0 / observed_lp.shape[0])
    raw_log_scale = -float(result["shift_x"]) * log_step
    raw_scale = math.exp(raw_log_scale)
    rotation = float(np.clip(raw_rotation, -15.0, 15.0))
    scale = float(np.clip(raw_scale, _RS_SCALE_MIN, _RS_SCALE_MAX))
    return rotation, scale, {
        "rotation_deg": rotation,
        "scale": scale,
        "raw_rotation_deg": raw_rotation,
        "raw_log_scale": raw_log_scale,
        "raw_scale": raw_scale,
        "PSR": float(result["PSR"]),
    }


def _wrap_rotation_180(angle_deg: float) -> float:
    """Canonical representative for magnitude-spectrum rotations."""

    return float((float(angle_deg) + 90.0) % 180.0 - 90.0)


def _quality_weighted_consensus(estimates: Iterable[tuple[float, float, float]]) -> tuple[float, float]:
    """Fuse measured periodic angles and log-scales without changing raw estimates."""

    material = [(float(a), float(s), max(0.0, float(q))) for a, s, q in estimates]
    if not material or not all(math.isfinite(item) for triple in material for item in triple):
        raise ValueError("Geometry-V4 cross-scale estimates are invalid")
    weights = np.asarray([max(1e-9, item[2]) for item in material], dtype=np.float64)
    doubled = np.radians(np.asarray([item[0] for item in material], dtype=np.float64) * 2.0)
    angle = math.degrees(math.atan2(float(np.sum(weights * np.sin(doubled))), float(np.sum(weights * np.cos(doubled))))) / 2.0
    log_scale = float(np.average(np.log(np.asarray([item[1] for item in material], dtype=np.float64)), weights=weights))
    return _wrap_rotation_180(angle), float(np.clip(math.exp(log_scale), _RS_SCALE_MIN, _RS_SCALE_MAX))


def _periodic_rotation_spread_180(rotations_deg: Iterable[float]) -> float:
    """Largest pairwise angular distance on the 180-degree magnitude circle."""

    values = tuple(_wrap_rotation_180(value) for value in rotations_deg)
    if not values or not all(math.isfinite(value) for value in values):
        return math.inf
    return float(
        max(
            abs(_wrap_rotation_180(left - right))
            for left in values
            for right in values
        )
    )


def _raw_cross_scale_spreads(
    raw_estimates: Iterable[tuple[float, float]], consensus_angle: float, consensus_scale: float
) -> tuple[float, float]:
    """Return raw candidate distances to consensus, never clamped search values."""

    material = tuple((float(angle), float(log_scale)) for angle, log_scale in raw_estimates)
    if (
        not material
        or not all(math.isfinite(value) for pair in material for value in pair)
        or not math.isfinite(consensus_angle)
        or not math.isfinite(consensus_scale)
        or consensus_scale <= 0.0
    ):
        return math.inf, math.inf
    consensus_log_scale = math.log(consensus_scale)
    return (
        float(max(abs(_wrap_rotation_180(angle - consensus_angle)) for angle, _ in material)),
        float(max(abs(log_scale - consensus_log_scale) for _, log_scale in material)),
    )


def _raw_group_spreads(groups: Iterable[tuple[float, float]], consensus: tuple[float, float]) -> tuple[float, float]:
    """Measure raw, resolved group residuals to the raw joint consensus."""

    values = tuple((float(angle), float(log_scale)) for angle, log_scale in groups)
    raw_angle, raw_log_scale = (float(value) for value in consensus)
    if (
        not values
        or not all(math.isfinite(value) for pair in values for value in pair)
        or not math.isfinite(raw_angle)
        or not math.isfinite(raw_log_scale)
    ):
        return math.inf, math.inf
    return (
        float(max(abs(_wrap_rotation_180(angle - raw_angle)) for angle, _ in values)),
        float(max(abs(log_scale - raw_log_scale) for _, log_scale in values)),
    )


def _mixed_group_consensus(raw_groups: list[tuple[float, float, float]]) -> dict[str, object]:
    """Deterministic raw-only medoid/median consensus for mixed-LP group evidence."""

    if len(raw_groups) != 3 or not all(math.isfinite(value) for group in raw_groups for value in group):
        return {"valid": False, "raw_consensus": (math.nan, math.nan), "resolved_groups": ()}
    angles = [float(group[0]) for group in raw_groups]
    consensus_angle = min(
        enumerate(angles),
        key=lambda item: (sum(abs(_wrap_rotation_180(item[1] - candidate)) for candidate in angles), item[0]),
    )[1]
    consensus_log_scale = float(np.median([float(group[1]) for group in raw_groups]))
    return {
        "valid": True,
        "raw_consensus": (float(consensus_angle), consensus_log_scale),
        "resolved_groups": tuple((float(group[0]), float(group[1])) for group in raw_groups),
    }


def _rectify_rs(attacked: np.ndarray, angle: float, scale: float) -> np.ndarray:
    fill = float(np.median(attacked))
    return _sample_h(attacked, _similarity_h(angle, scale), fill)


def _rectify_rs_with_valid(attacked: np.ndarray, angle: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Return rectified RGB and an RGB-independent overlap mask for local evidence."""

    transform = _similarity_h(angle, scale)
    rectified = _sample_h(attacked, transform, float(np.median(attacked)))
    valid = _sample_h(np.ones(attacked.shape[:2], dtype=np.float64), transform, 0.0)
    return rectified, np.asarray(valid >= 0.999999, dtype=bool)


def _translation_phase_correlation(
    rectified_plane: np.ndarray, reference: np.ndarray, valid_overlap: np.ndarray
) -> dict[str, object]:
    """Fixed-window masked Cartesian phase correlation for the public translation PSR."""

    valid = np.asarray(valid_overlap, dtype=np.float64)
    if valid.shape != rectified_plane.shape or reference.shape != rectified_plane.shape:
        raise ValueError("Geometry-V4 translation phase inputs differ")
    window = np.outer(np.hanning(rectified_plane.shape[0]), np.hanning(rectified_plane.shape[1]))
    weight = valid * window
    if float(np.sum(weight)) <= 1e-12:
        raise ValueError("Geometry-V4 translation overlap is empty")
    observed = rectified_plane - float(np.sum(rectified_plane * weight) / np.sum(weight))
    keyed_reference = reference - float(np.sum(reference * weight) / np.sum(weight))
    return normalized_phase_correlation(observed * weight, keyed_reference * weight)


def _bandpass(plane: np.ndarray, cycles: int) -> np.ndarray:
    height, width = plane.shape
    fy = np.fft.fftfreq(height) * height
    fx = np.fft.fftfreq(width) * width
    radius = np.sqrt(np.square(fy[:, None]) + np.square(fx[None, :]))
    mask = (radius >= cycles * 0.45) & (radius <= cycles * 1.8)
    return np.fft.ifft2(np.fft.fft2(plane - float(np.mean(plane))) * mask).real


def _rs_score(
    attacked: np.ndarray, reference: np.ndarray, angle: float, scale: float, *, cycles: int | None = None
) -> float:
    rectified = _luma(_rectify_rs(attacked, angle, scale))
    if cycles is not None:
        rectified = _bandpass(rectified, cycles)
        reference = _bandpass(reference, cycles)
    return float(normalized_phase_correlation(rectified, reference)["PSR"])


def _refine_one_rotation_scale(
    attacked: np.ndarray,
    reference: np.ndarray,
    coarse_angle: float,
    coarse_scale: float,
    *,
    cycles: int | None = None,
) -> tuple[float, float]:
    candidates: list[tuple[float, float, float]] = []
    for rotation_offset in (-4.0, 0.0, 4.0):
        for log_scale_offset in (-0.08, 0.0, 0.08):
            angle = float(np.clip(coarse_angle + rotation_offset, -16.0, 16.0))
            scale = float(np.clip(coarse_scale * math.exp(log_scale_offset), _RS_SCALE_MIN, _RS_SCALE_MAX))
            candidates.append((_rs_score(attacked, reference, angle, scale, cycles=cycles), angle, scale))
    _, angle, scale = max(candidates, key=lambda item: (item[0], -abs(item[1]), -abs(math.log(item[2]))))
    return float(angle), float(scale)


def _refine_rotation_scale(
    attacked: np.ndarray,
    global_reference: np.ndarray,
    by_scale: dict[int, np.ndarray],
    by_component: dict[tuple[int, float], np.ndarray],
) -> dict[str, object]:
    """Use unbounded joint group evidence for rectification, preserving all raw evidence."""

    per_scale: list[tuple[float, float]] = []
    qualities: list[float] = []
    for cycles in _SCALES:
        observed_band = _bandpass(_luma(attacked), cycles)
        reference_band = _bandpass(by_scale[cycles], cycles)
        independent_angle, independent_scale, coarse_record = _coarse_rotation_scale(observed_band, reference_band)
        selected_angle, selected_scale = _refine_one_rotation_scale(
            attacked,
            by_scale[cycles],
            independent_angle,
            independent_scale,
            cycles=cycles,
        )
        per_scale.append((float(selected_angle), float(selected_scale)))
        qualities.append(float(max(0.0, coarse_record["PSR"]) * max(0.0, _rs_score(attacked, by_scale[cycles], selected_angle, selected_scale, cycles=cycles))))

    observed_plane = _luma(attacked)
    sparse_records = _sparse_constellation_diagnostic(observed_plane, by_component)
    raw_groups: list[tuple[float, float, float]] = []
    group_observations: list[dict[str, object]] = []
    sparse_valid = len(sparse_records) == 3
    for sparse in sparse_records:
        valid = bool(sparse["valid"])
        sparse_valid = sparse_valid and valid
        raw_rotation = float(sparse["raw_rotation_deg"])
        raw_log_scale = float(sparse["raw_log_scale"])
        score = float(sparse["score"])
        raw_groups.append((raw_rotation, raw_log_scale, score))
        group_observations.append(
            {
                "identities": tuple((int(cycles), float(direction)) for cycles, direction in sparse["identities"]),
                "raw_rotation_deg": raw_rotation,
                "raw_log_scale": raw_log_scale,
                "sparse_glrt_score": score,
                "sparse_glrt_margin": float(sparse["margin"]),
                "measurement_id": "sparse_keyed_spectrum_glrt_v1",
                "valid": valid,
                "boundary": bool(sparse["boundary"]),
            }
        )
    joint = _mixed_group_consensus(raw_groups) if sparse_valid else {"valid": False, "raw_consensus": (math.nan, math.nan), "resolved_groups": ()}
    raw_angle, raw_log_scale = (float(value) for value in joint["raw_consensus"])
    if not bool(joint["valid"]) or not math.isfinite(raw_angle) or not math.isfinite(raw_log_scale):
        return {
            "rotation_deg": math.nan,
            "scale": math.nan,
            "legacy_per_scale_estimates": tuple(per_scale),
            "legacy_per_scale_quality": tuple(qualities),
            "group_observations": tuple(group_observations),
            "joint": joint,
            "rectification_seed": (math.nan, math.nan),
            "raw_valid": False,
        }
    try:
        raw_scale = math.exp(raw_log_scale)
    except OverflowError:
        raw_scale = math.inf
    if not math.isfinite(raw_scale):
        raise ValueError("Geometry-V4 joint raw consensus scale is non-finite")
    rectification_seed_angle = float(np.clip(raw_angle, -16.0, 16.0))
    rectification_seed_scale = float(np.clip(raw_scale, _RS_SCALE_MIN, _RS_SCALE_MAX))
    angle, scale = _refine_one_rotation_scale(
        attacked, global_reference, rectification_seed_angle, rectification_seed_scale
    )
    return {
        "rotation_deg": float(angle),
        "scale": float(scale),
        "legacy_per_scale_estimates": tuple(per_scale),
        "legacy_per_scale_quality": tuple(qualities),
        "group_observations": tuple(group_observations),
        "joint": joint,
        "rectification_seed": (rectification_seed_angle, rectification_seed_scale),
        "raw_valid": True,
    }


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


def _masked_normalized_patch_score(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float | None:
    """Fixed-mask ZNCC; invalid rectification padding never contributes."""

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != left.shape or left.shape != right.shape:
        raise ValueError("Geometry-V4 tile mask shape differs")
    if int(np.count_nonzero(mask)) < math.ceil(_LOCAL_MIN_VALID_FRACTION * mask.size):
        return None
    a = np.asarray(left, dtype=np.float64)[mask]
    b = np.asarray(right, dtype=np.float64)[mask]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b))))
    if denominator <= 1e-12:
        return None
    return float(np.sum(a * b) / denominator)


def _match_tiles(
    rectified_plane: np.ndarray,
    valid_overlap: np.ndarray,
    local_reference: np.ndarray,
    shift_x: int,
    shift_y: int,
    h_rs: np.ndarray,
) -> list[dict[str, object]]:
    height, width = rectified_plane.shape
    patch_radius = max(4, int(round(min(height, width) / 16.0)))
    search_radius = max(3, int(round(8.0 * min(height, width) / 64.0)))
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
                    observed_valid = _patch(valid_overlap, predicted_x + ox, predicted_y + oy, patch_radius)
                    if observed is not None and observed_valid is not None:
                        score = _masked_normalized_patch_score(observed, template, observed_valid)
                        if score is not None:
                            candidates.append((score, predicted_x + ox, predicted_y + oy))
            if not candidates:
                continue
            best = max(candidates, key=lambda item: (item[0], -abs(item[1] - predicted_x) - abs(item[2] - predicted_y)))
            separated = [
                item for item in candidates if abs(item[1] - best[1]) > 1 or abs(item[2] - best[2]) > 1
            ]
            second = max((item[0] for item in separated), default=-1.0)
            margin = best[0] - second
            sidelobes = np.asarray([item[0] for item in separated], dtype=np.float64)
            sidelobe_psr = float((best[0] - float(np.mean(sidelobes))) / (float(np.std(sidelobes)) + 1e-12)) if sidelobes.size else 0.0
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
                    "PSR": sidelobe_psr,
                }
            )
    return matches


def _similarity_design(material: list[dict[str, object]], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in indices:
        match = material[index]
        u, v = match["canonical"]  # type: ignore[misc]
        x, y = match["attacked"]  # type: ignore[misc]
        rows.extend(([u, -v, 1.0, 0.0], [v, u, 0.0, 1.0]))
        targets.extend((x, y))
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _estimate_from_parameters(parameters: np.ndarray) -> np.ndarray:
    a, b, tx, ty = (float(item) for item in parameters)
    return np.asarray(((a, -b, tx), (b, a, ty), (0.0, 0.0, 1.0)), dtype=np.float64)


def _similarity_residuals(estimate: np.ndarray, material: list[dict[str, object]]) -> np.ndarray:
    residuals: list[float] = []
    for match in material:
        u, v = match["canonical"]  # type: ignore[misc]
        x, y = match["attacked"]  # type: ignore[misc]
        predicted = estimate @ np.asarray((u, v, 1.0), dtype=np.float64)
        residuals.append(math.hypot(float(predicted[0]) - x, float(predicted[1]) - y) / math.sqrt(2.0))
    return np.asarray(residuals, dtype=np.float64)


def _pair_hypothesis(first: dict[str, object], second: dict[str, object]) -> np.ndarray | None:
    u1, v1 = first["canonical"]  # type: ignore[misc]
    u2, v2 = second["canonical"]  # type: ignore[misc]
    x1, y1 = first["attacked"]  # type: ignore[misc]
    x2, y2 = second["attacked"]  # type: ignore[misc]
    canonical_delta = complex(u2 - u1, v2 - v1)
    if abs(canonical_delta) <= 1e-12:
        return None
    multiplier = complex(x2 - x1, y2 - y1) / canonical_delta
    translation = complex(x1, y1) - multiplier * complex(u1, v1)
    return np.asarray(
        (
            (multiplier.real, -multiplier.imag, translation.real),
            (multiplier.imag, multiplier.real, translation.imag),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _weighted_refit(material: list[dict[str, object]], indices: list[int]) -> tuple[np.ndarray, float]:
    design, target = _similarity_design(material, indices)
    point_weights = np.asarray(
        [max(1e-6, float(material[index]["correlation"])) for index in indices], dtype=np.float64
    )
    row_weights = np.repeat(np.sqrt(point_weights), 2)
    weighted_design = design * row_weights[:, None]
    weighted_target = target * row_weights
    parameters, _, _, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    return _estimate_from_parameters(parameters), float(np.linalg.cond(weighted_design))


def _robust_similarity_fit(
    matches: Iterable[dict[str, object]], *, inlier_threshold: float = 0.02, minimum_inliers: int = 6
) -> tuple[np.ndarray | None, list[dict[str, object]], np.ndarray, float]:
    material = list(matches)
    if len(material) < 2:
        return None, [], np.empty(0, dtype=np.float64), 1e12
    best: tuple[tuple[object, ...], list[int]] | None = None
    for first_index in range(len(material) - 1):
        for second_index in range(first_index + 1, len(material)):
            hypothesis = _pair_hypothesis(material[first_index], material[second_index])
            if hypothesis is None:
                continue
            residuals = _similarity_residuals(hypothesis, material)
            inliers = [index for index, value in enumerate(residuals) if value <= inlier_threshold]
            if not inliers:
                continue
            weights = np.asarray(
                [max(1e-6, float(material[index]["correlation"])) for index in inliers], dtype=np.float64
            )
            weighted_rms = float(np.sqrt(np.average(np.square(residuals[inliers]), weights=weights)))
            pair_identity = tuple(sorted((material[first_index]["tile"], material[second_index]["tile"])))
            macro_balance = _macro_regions(material[index] for index in inliers)
            rank: tuple[object, ...] = (-len(inliers), -macro_balance, weighted_rms, pair_identity)
            if best is None or rank < best[0]:
                best = (rank, inliers)
    if best is None:
        return None, [], np.empty(0, dtype=np.float64), 1e12
    selected = best[1]
    estimate, condition = _weighted_refit(material, selected)
    for _ in range(2):
        residuals_all = _similarity_residuals(estimate, material)
        retained = [index for index in selected if residuals_all[index] <= inlier_threshold]
        if retained == selected or len(retained) < 2:
            selected = retained
            break
        selected = retained
        estimate, condition = _weighted_refit(material, selected)
    if len(selected) < minimum_inliers:
        return None, [material[index] for index in selected], _similarity_residuals(estimate, material)[selected], condition
    estimate, condition = _weighted_refit(material, selected)
    final_residuals = _similarity_residuals(estimate, material)[selected]
    return estimate, [material[index] for index in selected], final_residuals, condition


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


def _spatial_coverage(matches: Iterable[dict[str, object]]) -> float:
    points = sorted({tuple(match["canonical"]) for match in matches})  # type: ignore[arg-type]
    if len(points) < 3:
        return 0.0

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    ) / 2.0
    maximum = (_CENTERS[-1] - _CENTERS[0]) ** 2
    return float(min(1.0, area / maximum))


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
    combined, global_reference, local_reference, by_scale, by_component, _ = _anchor_fields(observed_rgb.shape[:2], geometry_key)
    observed_plane = _luma(observed_rgb)
    coarse_angle, coarse_scale, coarse_record = _coarse_rotation_scale(observed_plane, global_reference)
    rotation_scale = _refine_rotation_scale(observed_rgb, global_reference, by_scale, by_component)
    if not bool(rotation_scale["raw_valid"]):
        diagnostics = {
            "PSR": 0.0,
            "support": 0,
            "inlier_ratio": 0.0,
            "spatial_coverage": 0.0,
            "macro_regions": 0,
            "reprojection_rms_diagonal": 1.0,
            "condition_number": math.inf,
            "cross_scale_rotation_spread_deg": math.inf,
            "cross_scale_log_scale_spread": math.inf,
            "corner_validity": False,
            "aggregate_reliability": 0.0,
            "coarse_log_polar": coarse_record,
            "sparse_group_raw_valid": False,
            "joint_group_raw_observations": tuple(rotation_scale["group_observations"]),
            "raw_group_consensus": {"rotation_deg": math.nan, "log_scale": math.nan},
            "resolved_group_raw_estimates": (),
            "cross_scale_estimates": tuple(rotation_scale["legacy_per_scale_estimates"]),
            "cross_scale_estimates_role": "diagnostic_only",
            "cross_scale_quality": tuple(rotation_scale["legacy_per_scale_quality"]),
            "component_raw_observations": (),
            "component_observations_role": "diagnostic_only",
            "public_h_direction": "attacked_to_canonical",
            "rotation_deg": math.nan,
            "scale": math.nan,
            "rectification_seed": {"rotation_deg": math.nan, "scale": math.nan},
            "rectification_estimate": {"rotation_deg": math.nan, "scale": math.nan},
        }
        return {
            "method_id": GEOMETRY_V4_METHOD_ID,
            "protocol_id": GEOMETRY_V4_PROTOCOL_ID,
            "H_hat": None,
            "corners_hat": (),
            "support": 0,
            "reliability": 0.0,
            "status": "UNRELIABLE",
            "diagnostics": diagnostics,
        }
    angle = float(rotation_scale["rotation_deg"])
    scale = float(rotation_scale["scale"])
    per_scale = tuple(rotation_scale["legacy_per_scale_estimates"])
    per_scale_quality = tuple(rotation_scale["legacy_per_scale_quality"])
    group_observations = tuple(rotation_scale["group_observations"])
    joint = dict(rotation_scale["joint"])
    component_observations = []
    for (cycles, direction), component in sorted(by_component.items()):
        _, _, record = _coarse_rotation_scale(_bandpass(observed_plane, cycles), _bandpass(component, cycles))
        component_observations.append((int(cycles), float(direction), float(record["raw_rotation_deg"]), float(record["raw_log_scale"]), float(record["PSR"])))
    h_rs = _similarity_h(angle, scale)
    rectified_rgb, valid_overlap = _rectify_rs_with_valid(observed_rgb, angle, scale)
    rectified_plane = _luma(rectified_rgb)
    translation = _translation_phase_correlation(rectified_plane, combined, valid_overlap)
    shift_x = int(translation["shift_x"])
    shift_y = int(translation["shift_y"])
    matches = _match_tiles(rectified_plane, valid_overlap, local_reference, shift_x, shift_y, h_rs)
    canonical_to_attacked, inlier_matches, residuals, condition = _robust_similarity_fit(matches)
    if canonical_to_attacked is not None:
        attacked_to_canonical = np.linalg.inv(canonical_to_attacked)
        attacked_to_canonical /= attacked_to_canonical[2, 2]
        attacked_to_canonical[2, 2] = 1.0
        corners = _corners(attacked_to_canonical)
    else:
        attacked_to_canonical = None
        corners = ()
    corner_validity = attacked_to_canonical is not None and _valid_corners(corners)
    support = len(inlier_matches)
    inlier_ratio = support / len(matches) if matches else 0.0
    reprojection = float(np.sqrt(np.mean(np.square(residuals)))) if support else 1.0
    macro_regions = _macro_regions(inlier_matches)
    spatial_coverage = _spatial_coverage(inlier_matches)
    raw_consensus = tuple(float(value) for value in joint["raw_consensus"])
    resolved_groups = tuple((float(angle), float(log_scale)) for angle, log_scale in joint["resolved_groups"])
    rotation_spread, log_scale_spread = _raw_group_spreads(resolved_groups, raw_consensus)
    mean_tile_correlation = (
        float(np.mean([float(item["correlation"]) for item in inlier_matches])) if inlier_matches else 0.0
    )
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
    h_tuple = (
        tuple(float(value) for value in attacked_to_canonical.reshape(-1))
        if attacked_to_canonical is not None
        else None
    )
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
            "valid_overlap_fraction": float(np.mean(valid_overlap)),
            "candidate_match_count": len(matches),
            "matches": tuple(inlier_matches),
            "public_h_direction": "attacked_to_canonical",
            "cross_scale_estimates": tuple((float(a), float(s)) for a, s in per_scale),
            "cross_scale_estimates_role": "diagnostic_only",
            "cross_scale_quality": tuple(float(value) for value in per_scale_quality),
            "component_raw_observations": tuple(component_observations),
            "component_observations_role": "diagnostic_only",
            "joint_group_raw_observations": group_observations,
            "sparse_group_raw_valid": True,
            "raw_group_consensus": {
                "rotation_deg": raw_consensus[0],
                "log_scale": raw_consensus[1],
            },
            "resolved_group_raw_estimates": tuple(
                {"rotation_deg": group_angle, "log_scale": group_log_scale}
                for group_angle, group_log_scale in resolved_groups
            ),
            "rectification_seed": {
                "rotation_deg": float(rotation_scale["rectification_seed"][0]),
                "scale": float(rotation_scale["rectification_seed"][1]),
            },
            "rectification_estimate": {"rotation_deg": angle, "scale": scale},
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
