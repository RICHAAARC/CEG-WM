"""Injected-model, full-replay diagnostics for a diffusion reconstruction channel.

This module neither loads models nor makes watermark decisions. Each arm starts
the ordinary SD3 pipeline afresh; there is no custom sampler or latent resume.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from PIL import Image

from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image


ARM_SIGNS = {"zero": 0, "plus": 1, "minus": -1}


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    seed: int
    injection_step: int
    relative_l2: float
    guidance_scale: float
    height: int = 512
    width: int = 512
    num_inference_steps: int = 20
    device: str = "cuda"

    def __post_init__(self) -> None:
        _validate_sampling(self.seed, self.num_inference_steps, self.guidance_scale)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("generation prompt must be nonempty")
        if type(self.injection_step) is not int or not 0 <= self.injection_step < self.num_inference_steps - 1:
            raise ValueError("injection_step must leave at least one actual denoising step")
        if not math.isfinite(self.relative_l2) or self.relative_l2 <= 0:
            raise ValueError("relative_l2 must be finite and positive")
        if any(type(size) is not int or size < 1 for size in (self.height, self.width)):
            raise ValueError("image dimensions must be positive integers")


@dataclass(frozen=True)
class ReconstructionSettings:
    strength: float
    guidance_scale: float
    num_inference_steps: int
    prompt: str = ""
    device: str = "cuda"

    def __post_init__(self) -> None:
        _validate_sampling(0, self.num_inference_steps, self.guidance_scale)
        if not isinstance(self.prompt, str):
            raise TypeError("reconstruction prompt must be text")
        if not math.isfinite(self.strength) or not 0 < self.strength <= 1:
            raise ValueError("reconstruction strength must be in (0, 1]")
        if int(self.strength * self.num_inference_steps) < 1:
            raise ValueError("reconstruction strength must leave at least one img2img step")


@dataclass
class ArmReplay:
    image: Image.Image | None
    feature: torch.Tensor | None
    before_state: dict[str, Any]
    actual_delta: torch.Tensor | None
    injection_metrics: dict[str, float | int | str]
    remaining_steps: tuple[int, ...]
    after_latents: torch.Tensor | None = None
    status: str = "OBSERVED"
    error: str | None = None


@dataclass
class ReplayTriplet:
    settings: GenerationSettings
    direction: torch.Tensor
    arms: dict[str, ArmReplay]
    matched_state: bool = True


@dataclass
class ReconstructedArm:
    image: Image.Image | None
    feature: torch.Tensor | None
    status: str = "OBSERVED"
    error: str | None = None


@dataclass
class ReconstructionTriplet:
    settings: ReconstructionSettings
    seed: int
    arms: dict[str, ReconstructedArm]


class ReplayExecutionError(RuntimeError):
    """Retain completed arms and the failed arm's captured state for a runner."""

    def __init__(self, message: str, *, stage: str, arm: str,
                 completed_arms: Mapping[str, Any], before_state: dict[str, Any] | None = None,
                 partial_arm: Any | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.stage = stage
        self.arm = arm
        self.completed_arms = dict(completed_arms)
        self.arms = {**completed_arms, **({arm: partial_arm} if partial_arm is not None else {})}
        self.attempted_arms = tuple(dict.fromkeys((*completed_arms, arm)))
        self.before_state = before_state
        self.cause_type = type(cause).__name__ if cause is not None else None
        self.mismatch_paths = tuple(getattr(cause, "paths", ()))


class ReplayStateMismatch(ValueError):
    def __init__(self, paths: list[str]):
        self.paths = tuple(paths)
        super().__init__("pre-injection replay state differs: " + ", ".join(paths))


def _validate_sampling(seed: int, steps: int, guidance: float) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(steps) is not int or steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if isinstance(guidance, bool) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu(value)
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    return deepcopy(value)


def _different_paths(first: Any, second: Any, path: str = "state") -> list[str]:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        equal = first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
        return [] if equal else [path]
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            return [path]
        return [item for key in first for item in _different_paths(first[key], second[key], f"{path}.{key}")]
    if isinstance(first, tuple) and isinstance(second, tuple):
        if len(first) != len(second):
            return [path]
        return [item for i, (a, b) in enumerate(zip(first, second))
                for item in _different_paths(a, b, f"{path}[{i}]")]
    return [] if type(first) is type(second) and first == second else [path]


def final_image_feature(image: Image.Image, *, vae: Any, image_processor: Any) -> torch.Tensor:
    """A fresh deterministic public VAE observation of the final RGB, unpooled."""
    return _cpu(encode_final_rgb_image(image, image_processor, vae)).reshape(-1)


class _ReplayCallback:
    tensor_inputs = ("latents", "prompt_embeds", "pooled_prompt_embeds")

    def __init__(self, settings: GenerationSettings, direction: torch.Tensor, sign: int,
                 generator: torch.Generator, reference: dict[str, Any] | None):
        self.settings, self.direction, self.sign = settings, direction, sign
        self.generator, self.reference = generator, reference
        self.before_state: dict[str, Any] | None = None
        self.actual_delta: torch.Tensor | None = None
        self.after_latents: torch.Tensor | None = None
        self.metrics: dict[str, float | int | str] = {}
        self.remaining_steps: list[int] = []

    def __call__(self, pipeline: Any, step_index: int, timestep: Any,
                 callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if step_index > self.settings.injection_step:
            self.remaining_steps.append(step_index)
        if step_index != self.settings.injection_step:
            return callback_kwargs
        if self.before_state is not None:
            raise RuntimeError("injection callback ran twice")
        for name in self.tensor_inputs:
            value = callback_kwargs.get(name)
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"callback requires finite {name}")
        latent = callback_kwargs["latents"]
        if latent.ndim != 4 or latent.shape[0] != 1 or not latent.dtype.is_floating_point:
            raise ValueError("callback latent must be floating single-image NCHW")
        if latent.shape != self.direction.shape:
            raise ValueError("direction shape must match the injection latent exactly")
        scheduler = pipeline.scheduler
        config = dict(scheduler.config)
        if getattr(scheduler, "order", None) != 1:
            raise ValueError("source replay currently requires a first-order scheduler")
        if config.get("stochastic_sampling", False):
            raise ValueError("source replay currently requires deterministic scheduler sampling")
        timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
        if not isinstance(timesteps, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
            raise ValueError("source scheduler must expose tensor timesteps and sigmas")
        if len(timesteps) != self.settings.num_inference_steps or len(sigmas) != len(timesteps) + 1:
            raise ValueError("source scheduler must expose one sigma interval per denoising step")
        if not bool(torch.isfinite(timesteps).all()) or not bool(torch.isfinite(sigmas).all()):
            raise ValueError("source schedule must be finite")
        next_index = scheduler.step_index
        if next_index != step_index + 1:
            raise ValueError("callback must follow scheduler.step at its absolute step index")
        observed_timestep = torch.as_tensor(timestep).detach().cpu()
        if not torch.equal(observed_timestep, _cpu(timesteps[step_index])):
            raise ValueError("callback timestep differs from the actual scheduler entry")
        self.before_state = {
            **{name: _cpu(callback_kwargs[name]) for name in self.tensor_inputs},
            "latent_device": str(latent.device),
            "timestep": observed_timestep.clone(),
            "timesteps": _cpu(timesteps), "sigmas": _cpu(sigmas),
            "interval_start_sigma": _cpu(sigmas[step_index]),
            "latent_sigma": _cpu(sigmas[next_index]),
            "next_sigma": _cpu(sigmas[next_index + 1]),
            "next_step_index": next_index,
            "begin_index": getattr(scheduler, "begin_index", None),
            "scheduler_class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "scheduler_config": _snapshot(config),
            "generator_state": _cpu(self.generator.get_state()),
            "guidance_scale": float(pipeline.guidance_scale),
            "do_classifier_free_guidance": bool(pipeline.do_classifier_free_guidance),
            "clip_skip": getattr(pipeline, "clip_skip", None),
            "joint_attention_kwargs": _snapshot(getattr(pipeline, "joint_attention_kwargs", None)),
        }
        if self.before_state["guidance_scale"] != self.settings.guidance_scale:
            raise ValueError("pipeline did not apply the explicit guidance scale")
        if self.reference is not None:
            differences = _different_paths(self.reference, self.before_state)
            if differences:
                raise ReplayStateMismatch(differences)
        base = latent.detach().to(torch.float32)
        norm = float(torch.linalg.vector_norm(latent.double()).item())
        if norm == 0:
            raise ValueError("relative perturbation requires a nonzero latent")
        direction = self.direction.to(device=latent.device, dtype=torch.float32)
        if not bool(torch.isfinite(direction).all()):
            raise ValueError("direction is nonfinite in the injection arithmetic dtype")
        direction = direction / torch.linalg.vector_norm(direction.double()).to(direction.dtype)
        requested = direction * (self.settings.relative_l2 * norm * self.sign)
        if not bool(torch.isfinite(requested).all()):
            raise ValueError("requested injection is nonfinite")
        updated = latent if self.sign == 0 else (base + requested).to(latent.dtype)
        if not bool(torch.isfinite(updated).all()):
            raise ValueError("actual-dtype injection produced nonfinite latents")
        delta = updated.double() - latent.double()
        self.actual_delta = _cpu(delta)
        self.after_latents = _cpu(updated)
        self.metrics = {
            "sign": self.sign, "latent_dtype": str(latent.dtype),
            "requested_relative_l2": self.settings.relative_l2 if self.sign else 0.0,
            "actual_relative_l2": float(torch.linalg.vector_norm(delta.double()).item()) / norm,
            "nonzero_elements": int(torch.count_nonzero(delta).item()),
            "actual_minus_requested_l2": float(torch.linalg.vector_norm(delta - requested.double()).item()),
        }
        return {**callback_kwargs, "latents": updated}


def run_replay_triplet(pipeline: Any, direction: torch.Tensor, settings: GenerationSettings, *,
                       observation_vae: Any | None = None,
                       observation_processor: Any | None = None) -> ReplayTriplet:
    """Run 0/+/- from an equal observed pre-write state, retaining actual writes.

The source pipeline must expose the SD3 callback API and a deterministic
one-step sigma schedule. A mismatch aborts this triplet without retrying.
"""
    if not isinstance(direction, torch.Tensor) or not direction.dtype.is_floating_point:
        raise TypeError("direction must be a floating Tensor")
    if not bool(torch.isfinite(direction).all()) or float(torch.linalg.vector_norm(direction.double())) == 0:
        raise ValueError("direction must be finite and nonzero")
    fixed_direction = _cpu(direction)
    vae = pipeline.vae if observation_vae is None else observation_vae
    processor = pipeline.image_processor if observation_processor is None else observation_processor
    arms: dict[str, ArmReplay] = {}
    reference = None
    for arm, sign in ARM_SIGNS.items():
        generator = torch.Generator(device=settings.device).manual_seed(settings.seed)
        callback = _ReplayCallback(settings, fixed_direction, sign, generator, reference)
        image = None
        stage = "generation"
        try:
            result = pipeline(
                prompt=settings.prompt, num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale, height=settings.height, width=settings.width,
                generator=generator, output_type="pil", callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=list(callback.tensor_inputs),
            )
            if callback.before_state is None or callback.actual_delta is None:
                raise RuntimeError("pipeline omitted the requested injection step")
            expected = list(range(settings.injection_step + 1, settings.num_inference_steps))
            if callback.remaining_steps != expected:
                raise RuntimeError("pipeline did not execute the full denoising suffix")
            if len(result.images) != 1:
                raise ValueError("source pipeline must return exactly one image")
            image = require_ordinary_rgb_image(result.images[0])
            stage = "clean_feature"
            feature = final_image_feature(image, vae=vae, image_processor=processor)
            arms[arm] = ArmReplay(image, feature, callback.before_state, callback.actual_delta,
                                  callback.metrics, tuple(callback.remaining_steps), callback.after_latents)
            reference = arms["zero"].before_state
        except Exception as error:
            partial = ArmReplay(image, None, callback.before_state or {}, callback.actual_delta,
                                callback.metrics, tuple(callback.remaining_steps), callback.after_latents,
                                "FAILED", f"{type(error).__name__}: {error}")
            raise ReplayExecutionError(str(error), stage=stage, arm=arm, completed_arms=arms,
                                       before_state=callback.before_state, partial_arm=partial, cause=error) from error
    return ReplayTriplet(settings, fixed_direction, arms)


def reconstruct_triplet(pipeline: Any, source: ReplayTriplet, settings: ReconstructionSettings,
                        seed: int, *, observation_vae: Any,
                        observation_processor: Any) -> ReconstructionTriplet:
    """Pair SDXL img2img randomness across arms, then independently read each RGB."""
    _validate_sampling(seed, settings.num_inference_steps, settings.guidance_scale)
    if tuple(source.arms) != tuple(ARM_SIGNS) or not source.matched_state:
        raise ValueError("reconstruction requires a complete matched source triplet")
    arms: dict[str, ReconstructedArm] = {}
    for arm in ARM_SIGNS:
        image = None
        stage = "reconstruction"
        try:
            if source.arms[arm].image is None:
                raise ValueError("source arm has no final image")
            result = pipeline(
                prompt=settings.prompt, image=source.arms[arm].image.copy(),
                strength=settings.strength, guidance_scale=settings.guidance_scale,
                num_inference_steps=settings.num_inference_steps,
                generator=torch.Generator(device=settings.device).manual_seed(seed), output_type="pil",
            )
            if len(result.images) != 1:
                raise ValueError("reconstruction pipeline must return exactly one image")
            image = require_ordinary_rgb_image(result.images[0])
            stage = "reconstruction_feature"
            feature = final_image_feature(image, vae=observation_vae, image_processor=observation_processor)
            arms[arm] = ReconstructedArm(image, feature)
        except Exception as error:
            partial = ReconstructedArm(image, None, "FAILED", f"{type(error).__name__}: {error}")
            raise ReplayExecutionError(str(error), stage=stage, arm=arm,
                                       completed_arms=arms, partial_arm=partial, cause=error) from error
    return ReconstructionTriplet(settings, seed, arms)
