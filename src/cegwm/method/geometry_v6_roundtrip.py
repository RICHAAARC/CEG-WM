"""Geometry-V6 public pilot: final-latent frozen-VAE observability only."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image

GEOMETRY_V6_METHOD_ID = "geometry_v6_public_fixed_unkeyed_roundtrip_pilot_v1"
PUBLIC_PILOT_SPEC_ID = "public_midband_partitioned_pilot/search-fit-validate-v1"
R0_AMPLITUDE_CANDIDATES = (0.0025, 0.005, 0.01)
R0_RADIUS_MIN, R0_RADIUS_MAX = 0.24, 0.58

@dataclass(frozen=True, slots=True)
class PublicPilotPartition:
    search: torch.Tensor
    fit: torch.Tensor
    validate: torch.Tensor

@dataclass(frozen=True, slots=True)
class GeometryObservation:
    status: str
    aggregate_score: float
    search_score: float
    fit_score: float
    validate_score: float
    support_count: int
    search_count: int
    fit_count: int
    validate_count: int

def midfrequency_support(latents: torch.Tensor) -> torch.Tensor:
    _require_latents(latents)
    h, w = latents.shape[-2:]
    fy = torch.fft.fftfreq(h, device=latents.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(w, device=latents.device, dtype=torch.float32)
    return (torch.sqrt(fy[:, None].square() + fx[None, :].square()) > R0_RADIUS_MIN) & (torch.sqrt(fy[:, None].square() + fx[None, :].square()) < R0_RADIUS_MAX)

def public_pilot_partition(latents: torch.Tensor) -> PublicPilotPartition:
    """Fixed, public, non-overlapping subsets; never a key domain."""
    support = midfrequency_support(latents)
    h, w = support.shape
    ordinal = torch.arange(h * w, device=latents.device).reshape(h, w)
    conjugate = ((-torch.arange(h, device=latents.device)) % h)[:, None] * w + ((-torch.arange(w, device=latents.device)) % w)[None, :]
    pair_ordinal = torch.minimum(ordinal, conjugate)
    masks = tuple(support & (pair_ordinal.remainder(3) == index) for index in range(3))
    if any(int(mask.sum().item()) == 0 for mask in masks): raise ValueError("Geometry-V6 public pilot partition is empty")
    if not bool(torch.equal(masks[0] | masks[1] | masks[2], support)): raise RuntimeError("Geometry-V6 public pilot partition union differs")
    if any(bool((left & right).any()) for left, right in ((masks[0],masks[1]),(masks[0],masks[2]),(masks[1],masks[2]))): raise RuntimeError("Geometry-V6 public pilot partition overlaps")
    return PublicPilotPartition(*masks)

def public_pilot_template(latents: torch.Tensor) -> torch.Tensor:
    """Public shape/constants-only template with energy projected to midband."""
    _require_latents(latents)
    spectrum = midfrequency_support(latents).to(torch.complex64)[None, None].expand_as(latents).clone()
    template = torch.fft.ifft2(spectrum, dim=(-2, -1)).real
    norm = torch.linalg.vector_norm(template)
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0.0: raise RuntimeError("Geometry-V6 public pilot normalization failed")
    return template / norm

def frozen_roundtrip_observation(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    _require_latents(latents); scaling, shift = _vae_coordinates(vae)
    decoded = vae.decode(latents / scaling + shift, return_dict=True)
    mode = getattr(getattr(vae.encode(getattr(decoded, "sample", None)), "latent_dist", None), "mode", None)
    if not callable(mode): raise TypeError("frozen VAE encode result must expose latent_dist.mode")
    observation = (mode() - shift) * scaling
    _require_latents(observation)
    if observation.shape != latents.shape: raise ValueError("frozen VAE round-trip shape differs from final latent")
    return observation

def apply_roundtrip_adjoint_update(latents: torch.Tensor, amplitude: float, vae: Any) -> torch.Tensor:
    """One fixed-amplitude E(D(z)) adjoint update to full public pilot support."""
    _require_latents(latents); amplitude = _amplitude(amplitude)
    with torch.inference_mode(False), torch.enable_grad():
        source = latents.detach().clone().to(dtype=_vae_dtype(vae)).requires_grad_(True)
        objective = (frozen_roundtrip_observation(source, vae) * public_pilot_template(source)).sum()
        (gradient,) = torch.autograd.grad(objective, source, create_graph=False, only_inputs=True)
        projected = _project_midfrequency(gradient); norm = torch.linalg.vector_norm(projected)
        if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0.0: raise RuntimeError("Geometry-V6 matched adjoint has no supported energy")
        result = source + amplitude * projected / norm
        if not bool(torch.isfinite(result).all()): raise RuntimeError("Geometry-V6 adjoint update produced nonfinite latents")
        return result.detach().to(dtype=latents.dtype)

def blind_geometry_observation(image: Any, image_processor: Any, vae: Any) -> GeometryObservation:
    """Blind public-pilot observation from ordinary RGB and frozen public VAE."""
    observation = encode_final_rgb_image(require_ordinary_rgb_image(image), image_processor, vae)
    template, partition, support = public_pilot_template(observation), public_pilot_partition(observation), midfrequency_support(observation)
    observation_fft = torch.fft.fft2(observation.to(torch.float32), dim=(-2, -1))
    template_fft = torch.fft.fft2(template.to(torch.float32), dim=(-2, -1))
    channels = observation.shape[0] * observation.shape[1]
    return GeometryObservation("OBSERVATION_ONLY", _masked_frequency_cosine(observation_fft,template_fft,support), _masked_frequency_cosine(observation_fft,template_fft,partition.search), _masked_frequency_cosine(observation_fft,template_fft,partition.fit), _masked_frequency_cosine(observation_fft,template_fft,partition.validate), int(support.sum().item())*channels, int(partition.search.sum().item())*channels, int(partition.fit.sum().item())*channels, int(partition.validate.sum().item())*channels)

def _masked_frequency_cosine(observation: torch.Tensor, template: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask[None,None].to(dtype=observation.dtype); left, right = observation*expanded, template*expanded
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0: raise RuntimeError("Geometry-V6 public pilot score denominator is invalid")
    score = float(torch.vdot(right.reshape(-1), left.reshape(-1)).real.div(denominator).item())
    if not math.isfinite(score): raise RuntimeError("Geometry-V6 public pilot score is nonfinite")
    return score

def _project_midfrequency(value: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(value.to(torch.float32), dim=(-2,-1)); support = midfrequency_support(value).to(dtype=spectrum.dtype)
    return torch.fft.ifft2(spectrum * support[None,None], dim=(-2,-1)).real
def _require_latents(value: Any) -> None:
    if not isinstance(value,torch.Tensor) or value.ndim != 4 or value.shape[0] != 1: raise TypeError("Geometry-V6 final latent must be a finite 1CHW torch Tensor")
    if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()): raise ValueError("Geometry-V6 final latent must be finite floating data")
def _vae_coordinates(vae: Any) -> tuple[float,float]:
    config=getattr(vae,"config",None); scaling,shift=getattr(config,"scaling_factor",None),getattr(config,"shift_factor",None)
    if not isinstance(scaling,(int,float)) or isinstance(scaling,bool) or not math.isfinite(float(scaling)) or float(scaling)<=0: raise ValueError("frozen VAE scaling_factor is invalid")
    if not isinstance(shift,(int,float)) or isinstance(shift,bool) or not math.isfinite(float(shift)): raise ValueError("frozen VAE shift_factor is invalid")
    return float(scaling),float(shift)
def _vae_dtype(vae: Any) -> torch.dtype:
    try: parameter=next(vae.parameters())
    except (AttributeError,StopIteration,TypeError) as error: raise TypeError("frozen VAE must expose a floating parameter dtype") from error
    if not parameter.dtype.is_floating_point: raise TypeError("frozen VAE parameters must use a floating dtype")
    return parameter.dtype
def _amplitude(value: Any) -> float:
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(float(value)): raise TypeError("Geometry-V6 amplitude must be finite")
    amplitude=float(value)
    if amplitude not in R0_AMPLITUDE_CANDIDATES: raise ValueError("Geometry-V6 amplitude is outside the predeclared R0 sequence")
    return amplitude
__all__=["GEOMETRY_V6_METHOD_ID","GeometryObservation","PUBLIC_PILOT_SPEC_ID","PublicPilotPartition","R0_AMPLITUDE_CANDIDATES","R0_RADIUS_MAX","R0_RADIUS_MIN","apply_roundtrip_adjoint_update","blind_geometry_observation","frozen_roundtrip_observation","midfrequency_support","public_pilot_partition","public_pilot_template"]
