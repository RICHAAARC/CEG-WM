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


@dataclass(frozen=True, slots=True)
class RecoveredZTRotationScaleEstimate:
    rotation_degrees: float
    scale: float
    score: float


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
    spectrum = _dft2(latent[M0_TEMPLATE_CHANNEL])
    for point in template:
        if not isinstance(point, XTemplatePoint):
            raise TypeError("template entries must be XTemplatePoint")
        y, x = _frequency_bin(point.frequency_y, height), _frequency_bin(point.frequency_x, width)
        conjugate_y, conjugate_x = (-y) % height, (-x) % width
        weight = _finite(point.weight, "template weight")
        spectrum[y][x] += weight
        spectrum[conjugate_y][conjugate_x] += weight
    spatial = _idft2(spectrum)
    imaginary_residual = max(abs(value.imag) for row in spatial for value in row)
    if imaginary_residual > 1e-10:
        raise ValueError("Hermitian inverse transform has non-real residual")
    updated = list(latent)
    updated[M0_TEMPLATE_CHANNEL] = tuple(tuple(value.real for value in row) for row in spatial)
    return tuple(updated)


def estimate_rotation_scale_from_recovered_z_t(
    recovered_z_t: Sequence[Sequence[Sequence[float]]],
    candidate_grid: Sequence[Sequence[float]],
) -> RecoveredZTRotationScaleEstimate:
    """Search spectral candidates, then return attacked-to-canonical spatial R/S.

    Candidate selection is deterministic and blind: no original latent, prompt,
    clean RGB, transform truth, or attack parameters are accepted by this API.
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
    scored_forward_candidates: list[tuple[float, float, float]] = []
    for forward_rotation_degrees, forward_scale in candidates:
        angle = math.radians(forward_rotation_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        score = 0.0
        for point in template:
            observed_x = forward_scale * (cosine * point.frequency_x - sine * point.frequency_y)
            observed_y = forward_scale * (sine * point.frequency_x + cosine * point.frequency_y)
            y, x = _frequency_bin(observed_y, height), _frequency_bin(observed_x, width)
            score += abs(spectrum[y][x])
        scored_forward_candidates.append((score, forward_rotation_degrees, forward_scale))
    score, forward_rotation_degrees, forward_scale = max(
        scored_forward_candidates, key=lambda item: (item[0], -abs(item[1]), -item[2])
    )
    if not math.isfinite(score) or score <= 0.0:
        raise ValueError("recovered z_T has no usable X-template spectral evidence")
    # k_observed = c R(phi) k_canonical implies the spatial
    # attacked_to_canonical map is c R(-phi): Fourier already carries the
    # inverse-scale relation of the forward spatial attack.
    return RecoveredZTRotationScaleEstimate(
        _normalize_degrees(-forward_rotation_degrees), forward_scale, score
    )


def inject_initial_z_t_x_template_torch(latents: Any, template: Sequence[XTemplatePoint]) -> Any:
    """Torch-native FFT equivalent, without importing torch at module import."""

    if getattr(latents, "ndim", None) != 4 or tuple(latents.shape[1:]) != (4, 64, 64):
        raise ValueError("M0 torch latents must be 1x4x64x64")
    torch = __import__("torch")
    if not bool(latents.dtype.is_floating_point) or not bool(torch.isfinite(latents).all()):
        raise ValueError("M0 torch latents must be finite floating tensors")
    spectrum = torch.fft.fft2(latents[:, M0_TEMPLATE_CHANNEL].float())
    for point in template:
        if not isinstance(point, XTemplatePoint):
            raise TypeError("torch template entries must be XTemplatePoint")
        y, x = _frequency_bin(_finite(point.frequency_y, "frequency_y"), 64), _frequency_bin(_finite(point.frequency_x, "frequency_x"), 64)
        weight = _finite(point.weight, "template weight")
        conjugate_y, conjugate_x = (-y) % 64, (-x) % 64
        spectrum[:, y, x] = spectrum[:, y, x] + weight
        if (conjugate_y, conjugate_x) != (y, x):
            spectrum[:, conjugate_y, conjugate_x] = spectrum[:, conjugate_y, conjugate_x] + weight
    spatial = torch.fft.ifft2(spectrum)
    if float(spatial.imag.abs().max().item()) > _torch_hermitian_residual_tolerance(spatial, torch):
        raise ValueError("torch Hermitian inverse has non-real residual beyond dtype-aware numerical tolerance")
    result = latents.clone()
    result[:, M0_TEMPLATE_CHANNEL] = spatial.real.to(dtype=latents.dtype)
    return result


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


def _frequency_bin(value: float, size: int) -> int:
    frequency = _finite(value, "frequency")
    if not -0.5 <= frequency <= 0.5:
        raise ValueError("frequency lies outside normalized Fourier support")
    return int(round(frequency * size)) % size


def _normalize_degrees(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def _torch_hermitian_residual_tolerance(spatial: Any, torch: Any) -> float:
    """Bound complex FFT roundoff from its real dtype, scale, and dimensions."""

    height, width = (int(spatial.shape[-2]), int(spatial.shape[-1]))
    # `.float()` feeds float32/complex64 FFTs, so budget one dtype epsilon for
    # each radix stage on both spatial axes and scale it by the output amplitude.
    fft_rounds = math.ceil(math.log2(height)) + math.ceil(math.log2(width))
    real_scale = max(1.0, float(spatial.real.abs().max().item()))
    return float(torch.finfo(spatial.real.dtype).eps * (1 + fft_rounds) * real_scale)


def _dft2(plane: Sequence[Sequence[float]]) -> list[list[complex]]:
    height, width = len(plane), len(plane[0])
    return [[sum(plane[y][x] * cmath.exp(-2j * math.pi * ((u * y / height) + (v * x / width))) for y in range(height) for x in range(width)) for v in range(width)] for u in range(height)]


def _idft2(spectrum: Sequence[Sequence[complex]]) -> list[list[complex]]:
    height, width = len(spectrum), len(spectrum[0])
    return [[sum(spectrum[u][v] * cmath.exp(2j * math.pi * ((u * y / height) + (v * x / width))) for u in range(height) for v in range(width)) / (height * width) for x in range(width)] for y in range(height)]
