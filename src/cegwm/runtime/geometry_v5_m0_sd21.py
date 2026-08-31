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
    generator: Callable[..., Any], initial_z_t: Any, identity: SD21M0Identity = SD21M0Identity()
) -> Any:
    """Call an injected frozen generator once with fixed M0 generation identity."""

    if not callable(generator):
        raise TypeError("M0 generator adapter must be callable")
    if initial_z_t is None:
        raise ValueError("M0 initial z_T is required")
    return generator(
        prompt="",
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
