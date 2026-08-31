"""Geometry-V6 R0: final-latent, frozen-VAE round-trip carrier primitives.

This module deliberately contains no content key, content score, threshold, or
decision.  Geometry is a synchronization observation only.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Any

import torch

from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key

GEOMETRY_V6_METHOD_ID = "geometry_v6_training_free_final_latent_roundtrip_sync"
R0_AMPLITUDE_CANDIDATES = (0.0025, 0.005, 0.01)
R0_RADIUS_MIN = 0.24
R0_RADIUS_MAX = 0.58
_KEY_DOMAIN = b"CEG-WM/geometry-v6/roundtrip/v1\x00"


@dataclass(frozen=True, slots=True)
class GeometryKeySet:
    """Independent geometry subkeys; none is derived from a content key."""

    search: bytes
    fit: bytes
    validate: bytes


@dataclass(frozen=True, slots=True)
class GeometryObservation:
    mode: str
    score: float | None
    support_count: int


def derive_geometry_keys(geometry_key: str | bytes | bytearray | memoryview) -> GeometryKeySet:
    root = normalize_detection_key(geometry_key)

    def derive(label: bytes) -> bytes:
        return hmac.new(_KEY_DOMAIN + root, label, hashlib.sha256).digest()

    return GeometryKeySet(derive(b"k_search"), derive(b"k_fit"), derive(b"k_validate"))


def midfrequency_support(latents: torch.Tensor) -> torch.Tensor:
    """Return the strict 0.24 < normalized radius < 0.58 Fourier support."""

    _require_latents(latents)
    height, width = latents.shape[-2:]
    fy = torch.fft.fftfreq(height, device=latents.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=latents.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    return (radius > R0_RADIUS_MIN) & (radius < R0_RADIUS_MAX)


def keyed_template(latents: torch.Tensor, geometry_key: str | bytes | bytearray | memoryview) -> torch.Tensor:
    """Construct a public-domain keyed, unit-norm spectral template on support."""

    _require_latents(latents)
    key = derive_geometry_keys(geometry_key).validate
    support = midfrequency_support(latents)
    if int(support.sum().item()) == 0:
        raise ValueError("Geometry-V6 support is empty")
    digest = hashlib.sha256(key + repr(tuple(latents.shape)).encode("ascii")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big", signed=False))
    values = torch.randint(0, 2, tuple(latents.shape), generator=generator, dtype=torch.int8)
    template = values.to(device=latents.device, dtype=torch.float32).mul_(2).sub_(1)
    # The template lives in the encoded-latent domain; its *energy* is confined
    # to the named Fourier support, rather than treating a frequency mask as a
    # spatial mask.
    template = _project_midfrequency(template)
    norm = torch.linalg.vector_norm(template)
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0.0:
        raise RuntimeError("Geometry-V6 template normalization failed")
    return template / norm


def frozen_roundtrip_observation(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    """Compute E(D(z)) in the public VAE coordinate convention, without RGB I/O."""

    _require_latents(latents)
    scaling, shift = _vae_coordinates(vae)
    decoded = vae.decode(latents / scaling + shift, return_dict=True)
    sample = getattr(decoded, "sample", None)
    encoded = vae.encode(sample)
    distribution = getattr(encoded, "latent_dist", None)
    mode = getattr(distribution, "mode", None)
    if not callable(mode):
        raise TypeError("frozen VAE encode result must expose latent_dist.mode")
    observation = (mode() - shift) * scaling
    _require_latents(observation)
    if observation.shape != latents.shape:
        raise ValueError("frozen VAE round-trip shape differs from final latent")
    return observation


def apply_roundtrip_adjoint_update(
    latents: torch.Tensor,
    geometry_key: str | bytes | bytearray | memoryview,
    amplitude: float,
    vae: Any,
) -> torch.Tensor:
    """Apply exactly one global-amplitude E(D(z)) matched adjoint update.

    No retry, per-image optimization, content feedback, or detector threshold is
    present here.  The update is projected in the Fourier domain before use.
    """

    _require_latents(latents)
    amplitude = _amplitude(amplitude)
    # Diffusers invokes callbacks in inference mode.  Locally leave that mode
    # and make a fresh normal tensor so an inference tensor cannot enter the
    # one required adjoint graph.  This does not alter global grad state or VAE
    # parameters, and autograd.grad never writes parameter .grad fields.
    with torch.inference_mode(False), torch.enable_grad():
        source = latents.detach().clone().to(dtype=_vae_dtype(vae)).requires_grad_(True)
        template = keyed_template(source, geometry_key)
        observation = frozen_roundtrip_observation(source, vae)
        objective = (observation * template).sum()
        (gradient,) = torch.autograd.grad(objective, source, create_graph=False, only_inputs=True)
        projected = _project_midfrequency(gradient)
        norm = torch.linalg.vector_norm(projected)
        if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0.0:
            raise RuntimeError("Geometry-V6 matched adjoint has no supported energy")
        result = source + amplitude * projected / norm
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Geometry-V6 adjoint update produced nonfinite latents")
        return result.detach().to(dtype=latents.dtype)


def blind_geometry_observation(
    image: Any,
    geometry_key: str | bytes | bytearray | memoryview | None,
    image_processor: Any,
    vae: Any,
) -> GeometryObservation:
    """Blindly match only ordinary RGB against a geometry key, or record no-key."""

    ordinary = require_ordinary_rgb_image(image)
    if geometry_key is None:
        return GeometryObservation("no_key", None, 0)
    observation = encode_final_rgb_image(ordinary, image_processor, vae)
    template = keyed_template(observation, geometry_key)
    support_count = int(midfrequency_support(observation).sum().item()) * observation.shape[0] * observation.shape[1]
    denominator = torch.linalg.vector_norm(observation) * torch.linalg.vector_norm(template)
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise RuntimeError("Geometry-V6 blind score denominator is invalid")
    score = float(((observation * template).sum() / denominator).item())
    if not math.isfinite(score):
        raise RuntimeError("Geometry-V6 blind score is nonfinite")
    return GeometryObservation("keyed", score, support_count)


def _project_midfrequency(value: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(value.to(torch.float32), dim=(-2, -1))
    support = midfrequency_support(value).to(dtype=spectrum.dtype)
    result = torch.fft.ifft2(spectrum * support[None, None, :, :], dim=(-2, -1)).real
    return result


def _require_latents(value: Any) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1:
        raise TypeError("Geometry-V6 final latent must be a finite 1CHW torch Tensor")
    if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
        raise ValueError("Geometry-V6 final latent must be finite floating data")


def _vae_coordinates(vae: Any) -> tuple[float, float]:
    config = getattr(vae, "config", None)
    scaling = getattr(config, "scaling_factor", None)
    shift = getattr(config, "shift_factor", None)
    if not isinstance(scaling, (int, float)) or isinstance(scaling, bool) or not math.isfinite(float(scaling)) or float(scaling) <= 0.0:
        raise ValueError("frozen VAE scaling_factor is invalid")
    if not isinstance(shift, (int, float)) or isinstance(shift, bool) or not math.isfinite(float(shift)):
        raise ValueError("frozen VAE shift_factor is invalid")
    return float(scaling), float(shift)


def _vae_dtype(vae: Any) -> torch.dtype:
    try:
        parameter = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise TypeError("frozen VAE must expose a floating parameter dtype") from error
    if not parameter.dtype.is_floating_point:
        raise TypeError("frozen VAE parameters must use a floating dtype")
    return parameter.dtype


def _amplitude(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise TypeError("Geometry-V6 amplitude must be finite")
    amplitude = float(value)
    if amplitude not in R0_AMPLITUDE_CANDIDATES:
        raise ValueError("Geometry-V6 amplitude is outside the predeclared R0 sequence")
    return amplitude


__all__ = [
    "GEOMETRY_V6_METHOD_ID", "GeometryKeySet", "GeometryObservation", "R0_AMPLITUDE_CANDIDATES",
    "R0_RADIUS_MAX", "R0_RADIUS_MIN", "apply_roundtrip_adjoint_update", "blind_geometry_observation",
    "derive_geometry_keys", "frozen_roundtrip_observation", "keyed_template", "midfrequency_support",
]
