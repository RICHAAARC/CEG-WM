"""Pure Geometry-V5 M0 initial-``z_T`` writer and blind R/S/T helpers.

The writer changes one initial-noise channel only. The detector accepts one
recovered ``z_T`` only: it has no clean latent, prompt, RGB, key, or transform
truth parameter. This module deliberately contains no model loading code.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


M0_TEMPLATE_CHANNEL = 3
M0_TEMPLATE_SCALE = 5
M0_OFFICIAL_X_ANGLES_DEGREES = (1.0, 135.0)
M0_RADIAL_LENGTHS = (0.2, 0.3, 0.4, 0.5)
# This is a relative Tree-Ring-style setting, not a copied byte amplitude:
# target coefficient = pre-write global |FFT(z_T[channel])| mean + 5 * std.
M0_RELATIVE_COEFFICIENT_STD_MULTIPLIER = 5.0
_NUMERIC_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class XTemplatePoint:
    frequency_x: float
    frequency_y: float
    weight: float


@dataclass(frozen=True, slots=True)
class RotationScaleEstimate:
    rotation_degrees: float
    scale: float


@dataclass(frozen=True, slots=True)
class RecoveredZTRotationScaleEstimate:
    """Blind R/S result; score fields are diagnostics, never a verdict."""

    rotation_degrees: float
    scale: float
    score: float
    diagnostics: Mapping[str, float]


def build_hermitian_x_template(
    *, channel: int = M0_TEMPLATE_CHANNEL, scale: int = M0_TEMPLATE_SCALE,
    radial_lengths: Sequence[float] = M0_RADIAL_LENGTHS,
) -> tuple[XTemplatePoint, ...]:
    """Return official MaXsive X [1°,135°] radii with Hermitian partners.

    The original two X rays are adapted to a real latent plane by adding their
    negative-frequency conjugates. That is why four radii by four bins gives
    sixteen points while still representing the two official angles.
    """

    if isinstance(channel, bool) or not isinstance(channel, int) or channel != M0_TEMPLATE_CHANNEL:
        raise ValueError("M0 template channel differs")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale != M0_TEMPLATE_SCALE:
        raise ValueError("M0 template scale differs")
    radii = tuple(_finite(item, "radial length") for item in radial_lengths)
    if radii != M0_RADIAL_LENGTHS:
        raise ValueError("M0 radial lengths differ")
    points: list[XTemplatePoint] = []
    for radius in radii:
        for degrees in M0_OFFICIAL_X_ANGLES_DEGREES:
            angle = math.radians(degrees)
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            points.append(XTemplatePoint(x, y, float(scale)))
            points.append(XTemplatePoint(-x, -y, float(scale)))
    return tuple(points)


def inject_initial_z_t_x_template(
    z_t: Sequence[Sequence[Sequence[float]]], template: Sequence[XTemplatePoint]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Write a finite real/Hermitian template into initial ``z_T`` channel 3.

    The amplitude is recomputed from the unmodified global spectrum. Each
    template bin and its conjugate get a deterministic union-cleared 3x3
    neighbourhood before the explicit real conjugate-pair write.
    """

    latent = _latent(z_t)
    if len(latent) != 4:
        raise ValueError("M0 initial z_T must have four channels")
    height, width = len(latent[0]), len(latent[0][0])
    spectrum = _dft2(latent[M0_TEMPLATE_CHANNEL])
    target = _relative_coefficient_target(spectrum)
    _write_hermitian_template(spectrum, template, target)
    spatial = _idft2(spectrum)
    _require_real_finite_spatial(spatial, "Hermitian inverse transform")
    plane = tuple(tuple(_finite(value.real, "postcast initial z_T") for value in row) for row in spatial)
    updated = list(latent)
    updated[M0_TEMPLATE_CHANNEL] = plane
    return tuple(updated)


def estimate_rotation_scale_from_recovered_z_t(
    recovered_z_t: Sequence[Sequence[Sequence[float]]],
    candidate_grid: Sequence[Sequence[float]],
) -> RecoveredZTRotationScaleEstimate:
    """Blindly score recovered-zT spectral R/S candidates with NMS-aware PSR.

    The candidate score is cosine-normalized template local contrast, rather
    than a raw magnitude point sum. The runner-up is the strongest candidate
    outside the best candidate's R/S basin; adjacent grid cells are not a hard
    ambiguity rejection.
    """

    latent = _latent(recovered_z_t)
    if len(latent) != 4:
        raise ValueError("recovered z_T must have four channels")
    plane = latent[M0_TEMPLATE_CHANNEL]
    height, width = len(plane), len(plane[0])
    spectrum = _dft2(plane)
    candidates = tuple(tuple(_finite(item, "R/S candidate") for item in candidate) for candidate in candidate_grid)
    if not candidates or any(len(candidate) != 2 or candidate[1] <= 0.0 for candidate in candidates):
        raise ValueError("R/S candidate grid must contain finite rotation/positive-scale pairs")
    template = build_hermitian_x_template()
    support = _template_support(template, height, width)
    global_mean, global_std = _spectrum_magnitude_statistics(spectrum)
    if global_mean <= _NUMERIC_FLOOR and global_std <= _NUMERIC_FLOOR:
        raise ValueError("recovered z_T has no usable X-template spectral evidence")
    scored: list[tuple[float, float, float, float, float]] = []
    for forward_rotation_degrees, forward_scale in candidates:
        correlation, local_contrast = _normalized_template_match(
            spectrum, template, forward_rotation_degrees, forward_scale, support, global_mean,
        )
        scored.append((correlation * local_contrast, forward_rotation_degrees, forward_scale, correlation, local_contrast))
    ranked = sorted(scored, key=lambda item: (-item[0], abs(item[1]), item[2]))
    score, forward_rotation_degrees, forward_scale, correlation, local_contrast = ranked[0]
    if not math.isfinite(score) or score <= 0.0:
        raise ValueError("recovered z_T has no usable X-template spectral evidence")
    basin_rotation, basin_scale = _nms_basin_widths(candidates)
    outside = [
        item for item in ranked[1:]
        if abs(_normalize_degrees(item[1] - forward_rotation_degrees)) > basin_rotation
        or abs(item[2] - forward_scale) > basin_scale
    ]
    runner_up = outside[0][0] if outside else 0.0
    noise = [item[0] for item in outside] or [0.0]
    noise_mean = sum(noise) / len(noise)
    noise_std = math.sqrt(sum((item - noise_mean) ** 2 for item in noise) / len(noise))
    psr = (score - noise_mean) / max(noise_std, _NUMERIC_FLOOR)
    diagnostics = {
        "normalized_template_correlation": correlation,
        "local_contrast": local_contrast,
        "nms_runner_up_score": runner_up,
        "nms_psr": psr,
        "nms_rotation_basin_degrees": basin_rotation,
        "nms_scale_basin": basin_scale,
    }
    # k_observed = c R(phi) k_canonical, so public attacked_to_canonical is
    # c R(-phi) in this explicitly frozen V5 coordinate convention.
    return RecoveredZTRotationScaleEstimate(
        _normalize_degrees(-forward_rotation_degrees), forward_scale, score, diagnostics,
    )


def estimate_translation_phase_correlation(
    canonical_plane: Sequence[Sequence[float]], observed_plane: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Return attacked-to-canonical T using template-masked cross power only.

    This never correlates a full plane against a random-to-zero reference. It
    retains only fixed template support after R/S canonicalization.
    """

    reference = _plane(canonical_plane, "canonical plane")
    observed = _plane(observed_plane, "observed plane")
    height, width = len(reference), len(reference[0])
    if (height, width) != (len(observed), len(observed[0])) or height > 64 or width > 64:
        raise ValueError("masked phase-correlation helper requires matching planes no larger than 64x64")
    reference_f = _dft2(reference)
    observed_f = _dft2(observed)
    support = _template_support(build_hermitian_x_template(), height, width)
    cross = [[0j for _ in range(width)] for _ in range(height)]
    for y, x in support:
        value = observed_f[y][x] * reference_f[y][x].conjugate()
        if abs(value) > _NUMERIC_FLOOR:
            cross[y][x] = value / abs(value)
    if not any(abs(value) > 0.0 for row in cross for value in row):
        raise ValueError("masked phase correlation has no usable template support")
    surface = _idft2(cross)
    peak_y, peak_x = max(
        ((y, x) for y in range(height) for x in range(width)),
        key=lambda item: surface[item[0]][item[1]].real,
    )
    shift_x = peak_x if peak_x <= width // 2 else peak_x - width
    shift_y = peak_y if peak_y <= height // 2 else peak_y - height
    return (-shift_x / width, -shift_y / height)


def assemble_attacked_to_canonical_similarity(
    rotation_degrees: float, scale: float, tx: float, ty: float,
) -> tuple[tuple[float, float, float], ...]:
    """Assemble the explicitly named observed/attacked-to-canonical similarity."""

    rotation = math.radians(_finite(rotation_degrees, "rotation_degrees"))
    multiplier = _finite(scale, "scale")
    if multiplier <= 0.0:
        raise ValueError("scale must be positive")
    translation_x, translation_y = _finite(tx, "tx"), _finite(ty, "ty")
    cosine, sine = multiplier * math.cos(rotation), multiplier * math.sin(rotation)
    return ((cosine, -sine, translation_x), (sine, cosine, translation_y), (0.0, 0.0, 1.0))


def estimate_rotation_scale_from_peak_pairs(
    observed_points: Sequence[Sequence[float]], canonical_points: Sequence[Sequence[float]],
) -> RotationScaleEstimate:
    """Estimate an observed/attacked-to-canonical R/S similarity from pairs."""

    observed = _points(observed_points, "observed peaks")
    canonical = _points(canonical_points, "canonical peaks")
    if len(observed) != len(canonical) or len(observed) < 2:
        raise ValueError("R/S estimation requires paired peaks")
    ox = sum(point[0] for point in observed) / len(observed)
    oy = sum(point[1] for point in observed) / len(observed)
    cx = sum(point[0] for point in canonical) / len(canonical)
    cy = sum(point[1] for point in canonical) / len(canonical)
    denominator = sum((x - ox) ** 2 + (y - oy) ** 2 for x, y in observed)
    if denominator <= 0.0:
        raise ValueError("observed peaks are degenerate")
    a = sum((x - ox) * (u - cx) + (y - oy) * (v - cy) for (x, y), (u, v) in zip(observed, canonical, strict=True)) / denominator
    b = sum((x - ox) * (v - cy) - (y - oy) * (u - cx) for (x, y), (u, v) in zip(observed, canonical, strict=True)) / denominator
    scale = math.hypot(a, b)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("R/S estimate is singular")
    return RotationScaleEstimate(math.degrees(math.atan2(b, a)), scale)


def inject_initial_z_t_x_template_torch(latents: Any, template: Sequence[XTemplatePoint]) -> Any:
    """Torch FFT writer with the same relative coefficient and postcast checks."""

    if getattr(latents, "ndim", None) != 4 or tuple(latents.shape[1:]) != (4, 64, 64):
        raise ValueError("M0 torch latents must be 1x4x64x64")
    torch = __import__("torch")
    if not bool(latents.dtype.is_floating_point) or not bool(torch.isfinite(latents).all()):
        raise ValueError("M0 torch latents must be finite floating tensors")
    spectrum = torch.fft.fft2(latents[:, M0_TEMPLATE_CHANNEL].float())
    magnitudes = spectrum.abs()
    mean, std = magnitudes.mean(), magnitudes.std(unbiased=False)
    target = mean + M0_RELATIVE_COEFFICIENT_STD_MULTIPLIER * std
    if not bool(torch.isfinite(target)) or float(target.item()) <= _NUMERIC_FLOOR:
        raise ValueError("pre-write global spectrum magnitude statistics are degenerate")
    pairs = _template_bin_pairs(template, 64, 64)
    for y, x in _neighbourhood_union(pairs, 64, 64):
        spectrum[:, y, x] = 0j
    for (y, x), (cy, cx) in pairs:
        spectrum[:, y, x] = target
        spectrum[:, cy, cx] = target
    spatial = torch.fft.ifft2(spectrum)
    if not bool(torch.isfinite(spatial.real).all()) or not bool(torch.isfinite(spatial.imag).all()):
        raise ValueError("torch Hermitian inverse has non-finite spatial components")
    if float(spatial.imag.abs().max().item()) > _torch_hermitian_residual_tolerance(spatial, torch):
        raise ValueError("torch Hermitian inverse has non-real residual beyond dtype-aware numerical tolerance")
    updated_template_plane = spatial.real.to(dtype=latents.dtype)
    if not bool(torch.isfinite(updated_template_plane).all()):
        raise ValueError("torch Hermitian inverse cast to latent dtype has non-finite components")
    result = latents.clone()
    result[:, M0_TEMPLATE_CHANNEL] = updated_template_plane
    return result


def _relative_coefficient_target(spectrum: Sequence[Sequence[complex]]) -> float:
    mean, std = _spectrum_magnitude_statistics(spectrum)
    target = mean + M0_RELATIVE_COEFFICIENT_STD_MULTIPLIER * std
    if not math.isfinite(target) or target <= _NUMERIC_FLOOR:
        raise ValueError("pre-write global spectrum magnitude statistics are degenerate")
    return target


def _spectrum_magnitude_statistics(spectrum: Sequence[Sequence[complex]]) -> tuple[float, float]:
    magnitudes = tuple(abs(value) for row in spectrum for value in row)
    if not magnitudes or any(not math.isfinite(value) for value in magnitudes):
        raise ValueError("pre-write global spectrum magnitude statistics are non-finite")
    mean = sum(magnitudes) / len(magnitudes)
    std = math.sqrt(sum((value - mean) ** 2 for value in magnitudes) / len(magnitudes))
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("pre-write global spectrum magnitude statistics are non-finite")
    return mean, std


def _write_hermitian_template(spectrum: list[list[complex]], template: Sequence[XTemplatePoint], target: float) -> None:
    height, width = len(spectrum), len(spectrum[0])
    pairs = _template_bin_pairs(template, height, width)
    for y, x in _neighbourhood_union(pairs, height, width):
        spectrum[y][x] = 0j
    for (y, x), (cy, cx) in pairs:
        spectrum[y][x] = complex(target, 0.0)
        spectrum[cy][cx] = complex(target, 0.0)


def _template_bin_pairs(
    template: Sequence[XTemplatePoint], height: int, width: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for point in template:
        if not isinstance(point, XTemplatePoint):
            raise TypeError("template entries must be XTemplatePoint")
        y = _frequency_bin(_finite(point.frequency_y, "frequency_y"), height)
        x = _frequency_bin(_finite(point.frequency_x, "frequency_x"), width)
        _finite(point.weight, "template weight")
        conjugate = ((-y) % height, (-x) % width)
        pairs.add(tuple(sorted(((y, x), conjugate))))
    if not pairs:
        raise ValueError("template must contain support")
    return tuple(sorted(pairs))


def _template_support(template: Sequence[XTemplatePoint], height: int, width: int) -> set[tuple[int, int]]:
    return {bin_ for pair in _template_bin_pairs(template, height, width) for bin_ in pair}


def _neighbourhood_union(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]], height: int, width: int,
) -> tuple[tuple[int, int], ...]:
    locations: set[tuple[int, int]] = set()
    for pair in pairs:
        for center_y, center_x in pair:
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    locations.add(((center_y + delta_y) % height, (center_x + delta_x) % width))
    return tuple(sorted(locations))


def _normalized_template_match(
    spectrum: Sequence[Sequence[complex]], template: Sequence[XTemplatePoint],
    forward_rotation_degrees: float, forward_scale: float, support: set[tuple[int, int]],
    global_mean: float,
) -> tuple[float, float]:
    height, width = len(spectrum), len(spectrum[0])
    angle = math.radians(forward_rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    contrasts: list[float] = []
    for point in template:
        observed_x = forward_scale * (cosine * point.frequency_x - sine * point.frequency_y)
        observed_y = forward_scale * (sine * point.frequency_x + cosine * point.frequency_y)
        if not (-0.5 <= observed_x <= 0.5 and -0.5 <= observed_y <= 0.5):
            continue
        y, x = _frequency_bin(observed_y, height), _frequency_bin(observed_x, width)
        magnitude = abs(spectrum[y][x])
        local_ring = _local_ring_magnitude(spectrum, y, x, support)
        contrasts.append(max(0.0, magnitude - local_ring))
    if not contrasts:
        return 0.0, 0.0
    squared = sum(value * value for value in contrasts)
    correlation = sum(contrasts) / math.sqrt(max(_NUMERIC_FLOOR, len(contrasts) * squared))
    local_contrast = (sum(contrasts) / len(contrasts)) / max(global_mean, _NUMERIC_FLOOR)
    return correlation, local_contrast


def _local_ring_magnitude(
    spectrum: Sequence[Sequence[complex]], y: int, x: int, support: set[tuple[int, int]],
) -> float:
    height, width = len(spectrum), len(spectrum[0])
    values = [
        abs(spectrum[(y + dy) % height][(x + dx) % width])
        for dy in range(-2, 3) for dx in range(-2, 3)
        if max(abs(dy), abs(dx)) == 2 and ((y + dy) % height, (x + dx) % width) not in support
    ]
    return sum(values) / len(values) if values else 0.0


def _nms_basin_widths(candidates: Sequence[Sequence[float]]) -> tuple[float, float]:
    rotations = sorted({float(item[0]) for item in candidates})
    scales = sorted({float(item[1]) for item in candidates})
    rotation_steps = [right - left for left, right in zip(rotations, rotations[1:]) if right > left]
    scale_steps = [right - left for left, right in zip(scales, scales[1:]) if right > left]
    return (
        1.5 * min(rotation_steps) if rotation_steps else 0.0,
        1.5 * min(scale_steps) if scale_steps else 0.0,
    )


def _require_real_finite_spatial(spatial: Sequence[Sequence[complex]], label: str) -> None:
    residual = max(abs(value.imag) for row in spatial for value in row)
    scale = max(1.0, max(abs(value.real) for row in spatial for value in row))
    if not math.isfinite(residual) or residual > 1e-10 * scale:
        raise ValueError(f"{label} has non-real residual")
    if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for row in spatial for value in row):
        raise ValueError(f"{label} has non-finite spatial components")


def _latent(value: Any) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("z_T must be CHW data")
    planes = tuple(_plane(plane, "z_T plane") for plane in value)
    shape = (len(planes[0]), len(planes[0][0]))
    if any((len(plane), len(plane[0])) != shape for plane in planes):
        raise ValueError("z_T planes must have matching shape")
    return planes


def _plane(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be non-empty 2D data")
    rows = tuple(tuple(_finite(item, name) for item in row) for row in value)
    if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _points(value: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be pairs")
    result = tuple(tuple(_finite(item, name) for item in point) for point in value)
    if any(len(point) != 2 for point in result):
        raise ValueError(f"{name} must be pairs")
    return result  # type: ignore[return-value]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite non-bool real")
    return float(value)


def _frequency_bin(value: float, size: int) -> int:
    frequency = _finite(value, "frequency")
    if not -0.5 <= frequency <= 0.5:
        raise ValueError("frequency lies outside normalized Fourier support")
    return int(round(frequency * size)) % size


def _normalize_degrees(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def _torch_hermitian_residual_tolerance(spatial: Any, torch: Any) -> float:
    height, width = (int(spatial.shape[-2]), int(spatial.shape[-1]))
    fft_rounds = math.ceil(math.log2(height)) + math.ceil(math.log2(width))
    real_scale = max(1.0, float(spatial.real.abs().max().item()))
    return float(torch.finfo(spatial.real.dtype).eps * (1 + fft_rounds) * real_scale)


def _dft2(plane: Sequence[Sequence[float]]) -> list[list[complex]]:
    height, width = len(plane), len(plane[0])
    return [[sum(plane[y][x] * cmath.exp(-2j * math.pi * ((u * y / height) + (v * x / width))) for y in range(height) for x in range(width)) for v in range(width)] for u in range(height)]


def _idft2(spectrum: Sequence[Sequence[complex]]) -> list[list[complex]]:
    height, width = len(spectrum), len(spectrum[0])
    return [[sum(spectrum[u][v] * cmath.exp(2j * math.pi * ((u * y / height) + (v * x / width))) for u in range(height) for v in range(width)) / (height * width) for x in range(width)] for y in range(height)]
