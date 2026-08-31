"""Dependency-injected, frozen-SD2.1-facing M0 adapter.

Importing this module never imports torch/diffusers, loads a model, or contacts
the network. Real adapters remain absent until separately authorized.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
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

    if attacked_ordinary_rgb is None:
        raise ValueError("attacked ordinary RGB is required")
    if not callable(inverter) or not callable(estimator):
        raise TypeError("M0 inversion and estimation adapters must be callable")
    recovered_z_t = inverter(
        attacked_ordinary_rgb,
        prompt=identity.inversion_prompt,
        num_inference_steps=identity.steps,
        eta=identity.eta,
        guidance_scale=identity.guidance_scale,
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
    """Blind channel-3 finite-grid R/S then phase-correlation T estimate only."""
    try:
        torch = __import__("torch")
        if getattr(recovered_z_t, "ndim", None) != 4 or tuple(recovered_z_t.shape[1:]) != (4, 64, 64) or not bool(torch.isfinite(recovered_z_t).all()):
            raise ValueError("recovered z_T shape or finiteness differs")
        from cegwm.method.geometry_v5_m0 import assemble_attacked_to_canonical_similarity, build_hermitian_x_template

        magnitude = torch.fft.fft2(recovered_z_t[:, 3].float()).abs()[0]
        template = build_hermitian_x_template()
        best: tuple[float, float, float] | None = None
        for phi in range(-15, 16):
            for hundredths in range(85, 116):
                c = hundredths / 100.0
                angle = __import__("math").radians(phi)
                score = 0.0
                for point in template:
                    x = c * (__import__("math").cos(angle) * point.frequency_x - __import__("math").sin(angle) * point.frequency_y)
                    y = c * (__import__("math").sin(angle) * point.frequency_x + __import__("math").cos(angle) * point.frequency_y)
                    score += float(_bilinear_periodic(magnitude, y * 64.0, x * 64.0))
                candidate = (score, float(phi), c)
                if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and (abs(candidate[1]), candidate[2]) < (abs(best[1]), best[2])):
                    best = candidate
        if best is None or not best[0] > 0.0:
            raise ValueError("blind spectral grid has no peak")
        score, phi, scale = best
        rotation = -phi
        reference = torch.zeros_like(recovered_z_t[:, 3])
        # Carrier-neutral canonical reference is constructed through the same initial-zT template path.
        from cegwm.method.geometry_v5_m0 import inject_initial_z_t_x_template_torch
        reference_latent = inject_initial_z_t_x_template_torch(torch.zeros_like(recovered_z_t), template)[:, 3]
        cross = torch.fft.fft2(reference_latent.float()) * torch.fft.fft2(recovered_z_t[:, 3].float()).conj()
        corr = torch.fft.ifft2(cross / cross.abs().clamp_min(1e-12)).real[0]
        peak = int(corr.argmax().item()); y, x = divmod(peak, 64)
        tx = -(x if x <= 32 else x - 64) / 64.0; ty = -(y if y <= 32 else y - 64) / 64.0
        H = assemble_attacked_to_canonical_similarity(rotation, scale, tx, ty)
        psr = float(corr.max().item() / corr.abs().mean().clamp_min(1e-12).item())
        return GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", rotation, scale, tx, ty, H, {"blind_spectral_score": score, "phase_peak": float(corr.max().item()), "phase_psr": psr})
    except Exception:
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})


def _bilinear_periodic(magnitude: Any, y: float, x: float) -> Any:
    import math

    height, width = magnitude.shape
    y0, x0 = math.floor(y) % height, math.floor(x) % width
    y1, x1 = (y0 + 1) % height, (x0 + 1) % width
    fy, fx = y - math.floor(y), x - math.floor(x)
    return (1 - fy) * ((1 - fx) * magnitude[y0, x0] + fx * magnitude[y0, x1]) + fy * ((1 - fx) * magnitude[y1, x0] + fx * magnitude[y1, x1])


def recover_and_estimate_bound_sd21(pipeline: Any, attacked_ordinary_rgb: Any, identity: SD21M0Identity = SD21M0Identity()) -> GeometryV5M0RawRecord:
    try:
        return estimate_bound_blind_rst(invert_bound_sd21_attacked_rgb(pipeline, attacked_ordinary_rgb, identity), identity)
    except Exception:
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})


def load_bound_sd21_pipeline() -> Any:
    """Real-only lazy model binding; callers need CUDA and explicit authorization."""
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("M0 real SD2.1 runner requires CUDA")
    model_id = "sd2-community/stable-diffusion-2-1-base"
    revision = "4e63672c03103b6c636b8fb4119ba982469b2955"
    scheduler = diffusers.DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", revision=revision)
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model_id, revision=revision, scheduler=scheduler, torch_dtype=torch.float16)
    return pipeline.to("cuda")


def generate_bound_sd21(pipeline: Any, prompt: str, initial_z_t: Any) -> Any:
    """Use the manifest prompt and initial-zT-only template injection once."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generation requires the manifest unit prompt")
    from cegwm.method.geometry_v5_m0 import (
        build_hermitian_x_template,
        inject_initial_z_t_x_template_torch,
    )

    injected = inject_initial_z_t_x_template_torch(initial_z_t, build_hermitian_x_template())
    return pipeline(prompt=prompt, latents=injected, num_inference_steps=50, eta=0.0, guidance_scale=7.5)


def invert_bound_sd21_empty_prompt(inverter: Callable[..., Any], attacked_ordinary_rgb: Any) -> Any:
    """Call an injected DDIM/ODE inversion adapter with the fixed blind prompt."""
    if attacked_ordinary_rgb is None or not callable(inverter):
        raise ValueError("attacked RGB and inversion adapter are required")
    return inverter(attacked_ordinary_rgb, prompt="", guidance_scale=1.0, num_inference_steps=50, eta=0.0, vae_encoding="latent_dist.mode")
