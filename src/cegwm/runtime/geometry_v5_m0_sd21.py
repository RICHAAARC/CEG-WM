"""Lazy, concrete frozen-SD2.1-facing M0 adapter.

Importing this module never imports torch/diffusers, loads a model, or contacts
the network. Concrete adapters are implemented but unexecuted locally.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord


@dataclass(frozen=True, slots=True)
class SD21M0Identity:
    model_family: str = "sd2-community/stable-diffusion-2-1-base"
    model_revision: str = "4e63672c03103b6c636b8fb4119ba982469b2955"
    width: int = 512
    height: int = 512
    latent_shape: tuple[int, int, int] = (4, 64, 64)
    steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 7.5
    inversion_guidance_scale: float = 1.0
    inversion_prompt: str = ""
    vae_encoding: str = "mode_not_sampling"


_FROZEN_IDENTITY = SD21M0Identity()
_LATENT_SIDE = 64


def _validate_frozen_identity(identity: Any) -> SD21M0Identity:
    """Reject every production identity except the byte-bound M0 default."""

    if type(identity) is not SD21M0Identity or not _exact_value(identity, _FROZEN_IDENTITY):
        raise ValueError("M0 concrete runtime identity must equal the frozen SD2.1 default")
    return identity


def _exact_value(received: Any, expected: Any) -> bool:
    if type(received) is not type(expected):
        return False
    if isinstance(expected, SD21M0Identity):
        return all(_exact_value(getattr(received, name), getattr(expected, name)) for name in SD21M0Identity.__dataclass_fields__)
    if isinstance(expected, tuple):
        return len(received) == len(expected) and all(_exact_value(left, right) for left, right in zip(received, expected, strict=True))
    return received == expected


def _unit_image_translation_to_grid(value: Any) -> float:
    """Convert finite public unit-image translation to align-corners grid units.

    Public H translations use centered unit-image coordinates: one full image is
    1.0 and a p-pixel shift is p/64. ``grid_sample(align_corners=True)`` uses
    endpoint coordinates, so this is the sole unit-to-grid conversion.
    """

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("unit-image translation must be finite non-bool real")
    return (2.0 * _LATENT_SIDE / (_LATENT_SIDE - 1)) * float(value)


def _grid_translation_to_unit_image(value: Any) -> float:
    """Convert finite align-corners grid translation back to public H units."""

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("grid translation must be finite non-bool real")
    return ((_LATENT_SIDE - 1) / (2.0 * _LATENT_SIDE)) * float(value)


def require_sd21_optional_dependencies() -> None:
    """Fail closed when a real adapter is requested without installed packages."""

    missing: list[str] = []
    for name in ("torch", "diffusers"):
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    if missing:
        raise RuntimeError(f"M0 real SD2.1 adapter requires installed {', '.join(missing)}")


def run_generation_with_initial_z_t(
    generator: Callable[..., Any], initial_z_t: Any, prompt: str,
    identity: SD21M0Identity = SD21M0Identity(),
) -> Any:
    """Call an injected frozen generator once with fixed M0 generation identity."""

    identity = _validate_frozen_identity(identity)
    if not callable(generator) or not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("M0 generator adapter must be callable")
    if initial_z_t is None:
        raise ValueError("M0 initial z_T is required")
    return generator(
        prompt=prompt,
        latents=initial_z_t,
        width=identity.width,
        height=identity.height,
        num_inference_steps=identity.steps,
        eta=identity.eta,
        guidance_scale=identity.guidance_scale,
    )


def recover_and_estimate_from_attacked_rgb(
    attacked_ordinary_rgb: Any,
    inverter: Callable[..., Any],
    estimator: Callable[[Any], GeometryV5M0RawRecord],
    identity: SD21M0Identity = SD21M0Identity(),
) -> GeometryV5M0RawRecord:
    """Use only attacked ordinary RGB and frozen inversion identities at M0 boundary."""

    identity = _validate_frozen_identity(identity)
    if attacked_ordinary_rgb is None:
        raise ValueError("attacked ordinary RGB is required")
    if not callable(inverter) or not callable(estimator):
        raise TypeError("M0 inversion and estimation adapters must be callable")
    recovered_z_t = inverter(
        attacked_ordinary_rgb,
        prompt=identity.inversion_prompt,
        num_inference_steps=identity.steps,
        eta=identity.eta,
        guidance_scale=identity.inversion_guidance_scale,
        vae_encoding=identity.vae_encoding,
    )
    raw = estimator(recovered_z_t)
    if not isinstance(raw, GeometryV5M0RawRecord):
        raise TypeError("M0 estimator must return a raw M0 record")
    return raw


def public_runtime_capabilities() -> Mapping[str, bool]:
    """Declare boundaries, not availability or a fabricated real-model success."""

    return {
        "imports_torch_or_diffusers_at_module_import": False,
        "loads_model_or_network_at_module_import": False,
        "real_model_adapter_bound": True,
        "fake_injected_adapter_is_real_evidence": False,
        "may_emit_reliable": False,
        "may_rectify": False,
        "may_vote_content": False,
    }


def preprocess_bound_attacked_rgb(attacked_ordinary_rgb: Any) -> Any:
    """Deterministically convert one ordinary 512x512 RGB image to [-1, 1]."""
    torch = __import__("torch")
    if getattr(attacked_ordinary_rgb, "mode", None) != "RGB" or getattr(attacked_ordinary_rgb, "size", None) != (512, 512):
        raise ValueError("blind M0 input must be one ordinary 512x512 RGB image")
    pixels = torch.from_numpy(__import__("numpy").asarray(attacked_ordinary_rgb, dtype="float32")).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels / 127.5 - 1.0
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("attacked RGB preprocessing is non-finite")
    return pixels


def _empty_prompt_embeddings(pipeline: Any, device: Any) -> Any:
    tokens = pipeline.tokenizer([""], padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    ids = tokens.input_ids.to(device)
    encoded = pipeline.text_encoder(ids)
    return encoded[0] if isinstance(encoded, tuple) else encoded.last_hidden_state


def _vae_mode_latent(pipeline: Any, attacked_ordinary_rgb: Any) -> Any:
    torch = __import__("torch")
    pixels = preprocess_bound_attacked_rgb(attacked_ordinary_rgb).to(device=pipeline.device, dtype=pipeline.unet.dtype)
    with torch.no_grad():
        distribution = pipeline.vae.encode(pixels).latent_dist
        latent = distribution.mode()
    factor = float(pipeline.vae.config.scaling_factor)
    latent = latent * factor
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("VAE mode latent is non-finite")
    return latent


def invert_bound_sd21_attacked_rgb(pipeline: Any, attacked_ordinary_rgb: Any, identity: SD21M0Identity = SD21M0Identity()) -> Any:
    """Concrete empty-prompt DDIM/ODE-style image-to-noise inversion.

    The reverse DDIM update follows the Tree-Ring referenced flow but is an
    independent implementation: x_next=sqrt(a_next)x0+sqrt(1-a_next)epsilon.
    """
    identity = _validate_frozen_identity(identity)
    torch = __import__("torch")
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(identity.steps, device=pipeline.device)
    latent = _vae_mode_latent(pipeline, attacked_ordinary_rgb)
    embeds = _empty_prompt_embeddings(pipeline, pipeline.device)
    timesteps = list(reversed(scheduler.timesteps))
    with torch.no_grad():
        for index, timestep in enumerate(timesteps[:-1]):
            next_timestep = timesteps[index + 1]
            output = pipeline.unet(latent, timestep, encoder_hidden_states=embeds)
            epsilon = output.sample if hasattr(output, "sample") else output[0]
            alpha = scheduler.alphas_cumprod[timestep].to(device=latent.device, dtype=latent.dtype)
            alpha_next = scheduler.alphas_cumprod[next_timestep].to(device=latent.device, dtype=latent.dtype)
            x0 = (latent - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()
            latent = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
            if not bool(torch.isfinite(latent).all()):
                raise ValueError("DDIM inversion became non-finite")
    return latent


def estimate_bound_blind_rst(recovered_z_t: Any, identity: SD21M0Identity = SD21M0Identity()) -> GeometryV5M0RawRecord:
    """Blind channel-3 finite-grid R/S then canonicalized phase-correlation T.

    The selected R/S map is ``B=c R(-phi)`` from attacked to canonical
    coordinates. Translation is recovered only after resampling the observed
    plane through ``B^-1`` to the canonical grid. The phase peak is the
    normalized observed relative shift, so it is negated to return the ``u``
    in ``H=B x+u``. Ambiguous or degenerate estimates fail closed.
    """
    try:
        identity = _validate_frozen_identity(identity)
        torch = __import__("torch")
        if getattr(recovered_z_t, "ndim", None) != 4 or tuple(recovered_z_t.shape[1:]) != (4, 64, 64) or not bool(torch.isfinite(recovered_z_t).all()):
            raise ValueError("recovered z_T shape or finiteness differs")
        from cegwm.method.geometry_v5_m0 import (
            assemble_attacked_to_canonical_similarity,
            build_hermitian_x_template,
            inject_initial_z_t_x_template_torch,
        )

        magnitude = torch.fft.fft2(recovered_z_t[:, 3].float()).abs()[0]
        template = build_hermitian_x_template()
        scored_candidates: list[tuple[float, float, float]] = []
        for phi in range(-15, 16):
            for hundredths in range(85, 116):
                c = hundredths / 100.0
                angle = __import__("math").radians(phi)
                score = 0.0
                for point in template:
                    x = c * (__import__("math").cos(angle) * point.frequency_x - __import__("math").sin(angle) * point.frequency_y)
                    y = c * (__import__("math").sin(angle) * point.frequency_x + __import__("math").cos(angle) * point.frequency_y)
                    score += float(_bilinear_periodic(magnitude, y * 64.0, x * 64.0))
                scored_candidates.append((score, float(phi), c))
        ranked = sorted(scored_candidates, key=lambda item: (-item[0], abs(item[1]), item[2]))
        if len(ranked) < 2:
            raise ValueError("blind spectral grid is incomplete")
        score, phi, scale = ranked[0]
        runner_up_score = ranked[1][0]
        spectral_margin = score - runner_up_score
        if (
            not __import__("math").isfinite(score)
            or not __import__("math").isfinite(runner_up_score)
            or score <= 0.0
            or spectral_margin <= max(1e-6, 0.01 * score)
        ):
            raise ValueError("blind spectral grid is flat or ambiguous")
        rotation = -phi
        reference_latent = inject_initial_z_t_x_template_torch(torch.zeros_like(recovered_z_t), template)[:, 3]
        normalized_observed, overlap = _resample_recovered_to_canonical(recovered_z_t[:, 3].float(), scale, rotation)
        if not bool(torch.isfinite(normalized_observed).all()) or overlap <= 0.5:
            raise ValueError("canonicalized observed plane has insufficient overlap")
        cross = torch.fft.fft2(normalized_observed) * torch.fft.fft2(reference_latent.float()).conj()
        if not bool(torch.isfinite(cross.real).all()) or float(cross.abs().max().item()) <= 0.0:
            raise ValueError("phase correlation has no usable spectrum")
        corr = torch.fft.ifft2(cross / cross.abs().clamp_min(1e-12)).real[0]
        if not bool(torch.isfinite(corr).all()) or float(corr.abs().max().item()) <= 0.0:
            raise ValueError("phase correlation is degenerate")
        peak = int(corr.argmax().item()); y, x = divmod(peak, _LATENT_SIDE)
        observed_shift_x = x if x <= _LATENT_SIDE // 2 else x - _LATENT_SIDE
        observed_shift_y = y if y <= _LATENT_SIDE // 2 else y - _LATENT_SIDE
        # A p-pixel phase shift is p/64 in the public centered unit-image H basis.
        tx = -observed_shift_x / _LATENT_SIDE; ty = -observed_shift_y / _LATENT_SIDE
        H = assemble_attacked_to_canonical_similarity(rotation, scale, tx, ty)
        phase_peak = float(corr.max().item())
        psr = float(phase_peak / corr.abs().mean().clamp_min(1e-12).item())
        if not __import__("math").isfinite(psr) or psr <= 1.05:
            raise ValueError("phase correlation has insufficient separation")
        return GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", rotation, scale, tx, ty, H, {
            "blind_spectral_score": score,
            "spectral_margin": spectral_margin,
            "phase_peak": phase_peak,
            "phase_psr": psr,
            "canonical_overlap": overlap,
        })
    except Exception:
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})


def _bilinear_periodic(magnitude: Any, y: float, x: float) -> Any:
    import math

    height, width = magnitude.shape
    y0, x0 = math.floor(y) % height, math.floor(x) % width
    y1, x1 = (y0 + 1) % height, (x0 + 1) % width
    fy, fx = y - math.floor(y), x - math.floor(x)
    return (1 - fy) * ((1 - fx) * magnitude[y0, x0] + fx * magnitude[y0, x1]) + fy * ((1 - fx) * magnitude[y1, x0] + fx * magnitude[y1, x1])


def _resample_recovered_to_canonical(observed_plane: Any, scale: float, rotation_degrees: float) -> tuple[Any, float]:
    """Return ``g(B^-1 q)`` on canonical ``q`` coordinates.

    Coordinates are centred normalized ``[-1, 1]`` with
    ``grid_sample(..., align_corners=True, padding_mode="zeros")``.  For
    ``B=cR(theta)``, a canonical destination point ``q`` samples source
    observed coordinates ``B^-1q=R(-theta)q/c``. The sampled all-ones mask is
    reported as the fixed zero-padding overlap diagnostic.
    """
    torch = __import__("torch")
    functional = torch.nn.functional
    if getattr(observed_plane, "ndim", None) != 3 or tuple(observed_plane.shape[1:]) != (64, 64):
        raise ValueError("observed channel-3 plane must be batchx64x64")
    if not __import__("math").isfinite(scale) or scale <= 0.0 or not __import__("math").isfinite(rotation_degrees):
        raise ValueError("canonicalization similarity is invalid")
    angle = __import__("math").radians(rotation_degrees)
    cosine, sine = __import__("math").cos(angle), __import__("math").sin(angle)
    coordinates = torch.linspace(-1.0, 1.0, 64, device=observed_plane.device, dtype=observed_plane.dtype)
    qy, qx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    # B^-1=R(-theta)/c. x is horizontal grid coordinate, y is vertical.
    source_x = (cosine * qx + sine * qy) / scale
    source_y = (-sine * qx + cosine * qy) / scale
    grid = torch.stack((source_x, source_y), dim=-1).unsqueeze(0).expand(observed_plane.shape[0], -1, -1, -1)
    normalized = functional.grid_sample(
        observed_plane.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, 0]
    support = functional.grid_sample(
        torch.ones_like(observed_plane).unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, 0]
    return normalized, float(support.mean().item())


def recover_and_estimate_bound_sd21(pipeline: Any, attacked_ordinary_rgb: Any, identity: SD21M0Identity = SD21M0Identity()) -> GeometryV5M0RawRecord:
    try:
        identity = _validate_frozen_identity(identity)
        return estimate_bound_blind_rst(invert_bound_sd21_attacked_rgb(pipeline, attacked_ordinary_rgb, identity), identity)
    except Exception:
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})


def load_bound_sd21_pipeline() -> Any:
    """Real-only lazy model binding; callers need CUDA and explicit authorization."""
    identity = _validate_frozen_identity(_FROZEN_IDENTITY)
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("M0 real SD2.1 runner requires CUDA")
    scheduler = diffusers.DDIMScheduler.from_pretrained(identity.model_family, subfolder="scheduler", revision=identity.model_revision)
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(identity.model_family, revision=identity.model_revision, scheduler=scheduler, torch_dtype=torch.float16)
    return pipeline.to("cuda")


def generate_bound_sd21(pipeline: Any, prompt: str, initial_z_t: Any, identity: SD21M0Identity = SD21M0Identity()) -> Any:
    """Use the manifest prompt and initial-zT-only template injection once."""
    identity = _validate_frozen_identity(identity)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generation requires the manifest unit prompt")
    from cegwm.method.geometry_v5_m0 import (
        build_hermitian_x_template,
        inject_initial_z_t_x_template_torch,
    )

    injected = inject_initial_z_t_x_template_torch(initial_z_t, build_hermitian_x_template())
    return pipeline(prompt=prompt, latents=injected, num_inference_steps=identity.steps, eta=identity.eta, guidance_scale=identity.guidance_scale)


def invert_bound_sd21_empty_prompt(inverter: Callable[..., Any], attacked_ordinary_rgb: Any) -> Any:
    """Call an injected DDIM/ODE inversion adapter with the fixed blind prompt."""
    if attacked_ordinary_rgb is None or not callable(inverter):
        raise ValueError("attacked RGB and inversion adapter are required")
    return inverter(attacked_ordinary_rgb, prompt="", guidance_scale=1.0, num_inference_steps=50, eta=0.0, vae_encoding="latent_dist.mode")
