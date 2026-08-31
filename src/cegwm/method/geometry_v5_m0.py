"""Pure Geometry-V5 M0 global R/S/T helpers.

These helpers operate on initial/recovered latent data only. They do not read
images, prompts, truth, content scores, or a detector decision.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence


M0_TEMPLATE_CHANNEL = 3
M0_TEMPLATE_SCALE = 5
M0_RADIAL_LENGTHS = (0.2, 0.3, 0.4, 0.5)


@dataclass(frozen=True, slots=True)
class XTemplatePoint:
    frequency_x: float
    frequency_y: float
    weight: float


@dataclass(frozen=True, slots=True)
class RotationScaleEstimate:
    rotation_degrees: float
    scale: float


def build_hermitian_x_template(
    *, channel: int = M0_TEMPLATE_CHANNEL, scale: int = M0_TEMPLATE_SCALE,
    radial_lengths: Sequence[float] = M0_RADIAL_LENGTHS,
) -> tuple[XTemplatePoint, ...]:
    """Build the MaXsive-referenced X geometry with V5 Hermitian adaptation."""

    if isinstance(channel, bool) or not isinstance(channel, int) or channel != M0_TEMPLATE_CHANNEL:
        raise ValueError("M0 template channel differs")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale != M0_TEMPLATE_SCALE:
        raise ValueError("M0 template scale differs")
    radii = tuple(_finite(item, "radial length") for item in radial_lengths)
    if radii != M0_RADIAL_LENGTHS:
        raise ValueError("M0 radial lengths differ")
    points: list[XTemplatePoint] = []
    diagonal = math.sqrt(0.5)
    for radius in radii:
        for sign_x, sign_y in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            x = sign_x * radius * diagonal
            y = sign_y * radius * diagonal
            points.append(XTemplatePoint(x, y, float(scale)))
    return tuple(points)


def inject_initial_z_t_x_template(
    z_t: Sequence[Sequence[Sequence[float]]], template: Sequence[XTemplatePoint]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Inject a real/Hermitian M0 template once into initial `z_T` only."""

    latent = _latent(z_t)
    if len(latent) != 4:
        raise ValueError("M0 initial z_T must have four channels")
    height, width = len(latent[0]), len(latent[0][0])
    updated = [[list(row) for row in plane] for plane in latent]
    for point in template:
        if not isinstance(point, XTemplatePoint):
            raise TypeError("template entries must be XTemplatePoint")
        y = int(round((point.frequency_y + 0.5) * (height - 1)))
        x = int(round((point.frequency_x + 0.5) * (width - 1)))
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("template point lies outside normalized spectrum")
        updated[M0_TEMPLATE_CHANNEL][y][x] += _finite(point.weight, "template weight")
    return tuple(tuple(tuple(row) for row in plane) for plane in updated)


def estimate_rotation_scale_from_peak_pairs(
    observed_points: Sequence[Sequence[float]], canonical_points: Sequence[Sequence[float]]
) -> RotationScaleEstimate:
    """Estimate the observed/attacked-to-canonical R/S similarity from peak pairs."""

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


def estimate_translation_phase_correlation(
    canonical_plane: Sequence[Sequence[float]], observed_plane: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """Return normalized observed/attacked-to-canonical translation by phase correlation.

    The direct DFT implementation is deliberately bounded for deterministic fake
    tests; real 64×64 execution belongs to a later authorized runtime binding.
    """

    reference = _plane(canonical_plane, "canonical plane")
    observed = _plane(observed_plane, "observed plane")
    height, width = len(reference), len(reference[0])
    if (height, width) != (len(observed), len(observed[0])) or height > 16 or width > 16:
        raise ValueError("phase-correlation helper requires matching planes no larger than 16x16")
    reference_f = _dft2(reference)
    observed_f = _dft2(observed)
    cross = []
    for left_row, right_row in zip(observed_f, reference_f, strict=True):
        cross_row: list[complex] = []
        for left, right in zip(left_row, right_row, strict=True):
            value = left * right.conjugate()
            cross_row.append(value / abs(value) if abs(value) > 0.0 else 0j)
        cross.append(cross_row)
    surface = _idft2(cross)
    peak_y, peak_x = max(((y, x) for y in range(height) for x in range(width)), key=lambda item: surface[item[0]][item[1]].real)
    shift_x = peak_x if peak_x <= width // 2 else peak_x - width
    shift_y = peak_y if peak_y <= height // 2 else peak_y - height
    return (-shift_x / width, -shift_y / height)


def assemble_attacked_to_canonical_similarity(
    rotation_degrees: float, scale: float, tx: float, ty: float
) -> tuple[tuple[float, float, float], ...]:
    """Assemble the explicitly named observed/attacked-to-canonical similarity."""

    rotation = math.radians(_finite(rotation_degrees, "rotation_degrees"))
    multiplier = _finite(scale, "scale")
    if multiplier <= 0.0:
        raise ValueError("scale must be positive")
    translation_x, translation_y = _finite(tx, "tx"), _finite(ty, "ty")
    cosine, sine = multiplier * math.cos(rotation), multiplier * math.sin(rotation)
    return ((cosine, -sine, translation_x), (sine, cosine, translation_y), (0.0, 0.0, 1.0))


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


def _dft2(plane: Sequence[Sequence[float]]) -> list[list[complex]]:
    height, width = len(plane), len(plane[0])
    return [[sum(plane[y][x] * cmath.exp(-2j * math.pi * ((u * y / height) + (v * x / width))) for y in range(height) for x in range(width)) for v in range(width)] for u in range(height)]


def _idft2(spectrum: Sequence[Sequence[complex]]) -> list[list[complex]]:
    height, width = len(spectrum), len(spectrum[0])
    return [[sum(spectrum[u][v] * cmath.exp(2j * math.pi * ((u * y / height) + (v * x / width))) for u in range(height) for v in range(width)) / (height * width) for x in range(width)] for y in range(height)]
