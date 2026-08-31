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
    model_family: str = "frozen_SD2_1_facing"
    model_revision: str = "unbound_until_future_real_run_authorization"
    width: int = 512
    height: int = 512
    latent_shape: tuple[int, int, int] = (4, 64, 64)
    steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 7.5
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
        "real_model_adapter_bound": False,
        "fake_injected_adapter_is_real_evidence": False,
        "may_emit_reliable": False,
        "may_rectify": False,
        "may_vote_content": False,
    }


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
