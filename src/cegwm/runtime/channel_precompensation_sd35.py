"""Segmented write-side proxy sensitivities; never a detector or model loader.

The SDXL path calls diffusers 0.39's undecorated official img2img implementation,
including stochastic VAE encoding, all selected scheduler steps and VAE decoding.
Only the uint8 rounding boundary uses a straight-through derivative. Floating
casts and clamp retain PyTorch's derivatives. These are proxy sensitivities, not
derivatives of the discrete saved-image map. All public scores are re-read from
ordinary RGB. Model parameters must be frozen and accelerate hooks removed for
gradient phases. Saved autograd tensors are copied to CPU, with copies of known
stable frozen parameter views reused across steps. The changing-dtype SDXL VAE
is excluded from reuse. Activations and these saved weights still cost host RAM.

Inspected API: diffusers 0.39.0 SD3 lines 1067--1140, SDXL img2img
prepare_latents and __call__ lines 1257--1495, VaeImageProcessor.postprocess.
No model, scheduler or tokenizer is instantiated at import time.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image
import torch

from cegwm.runtime.observation import (
    encode_final_rgb_image, require_ordinary_rgb_image, _vae_device_dtype,
    _vae_config_number,
)
from cegwm.runtime.reconstruction_channel_sd35 import (
    GenerationSettings, ReconstructionSettings, _ReplayCallback, _cpu,
    _different_paths, ReplayStateMismatch,
)


@dataclass
class SourceCapture:
    image: Image.Image
    before_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    remaining_steps: tuple[int, ...]


@dataclass
class RGBSensitivity:
    image: Image.Image
    feature: torch.Tensor
    score: float
    gradient_rgb: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteReplay:
    image: Image.Image
    before_state: dict[str, Any]
    actual_delta: torch.Tensor
    after_latents: torch.Tensor
    injection_metrics: dict[str, Any]
    remaining_steps: tuple[int, ...]


class RuntimeStageError(RuntimeError):
    """Expose partial artifacts and physical attempted-work counters on failure."""

    def __init__(self, stage: str, cause: Exception, partial: dict[str, Any], counters):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.partial = partial
        self.counters = dict(counters)
        self.mismatch_paths = tuple(getattr(cause, "paths", ()))


def _count(counters, name, amount=1):
    counters[name] = counters.get(name, 0) + amount


def _ledger(counters):
    return {} if counters is None else counters


def _finite(tensor: torch.Tensor, name: str):
    if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite tensor")


def image_tensor(image: Image.Image, *, device="cpu") -> torch.Tensor:
    image = require_ordinary_rgb_image(image)
    # Match VaeImageProcessor.pil_to_numpy's CPU float32 normalization exactly,
    # rather than introducing a potentially different CUDA division rounding.
    pixels = np.array(image, dtype=np.float32, copy=True) / np.float32(255.0)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=device)


def quantized_rgb_proxy(rgb: torch.Tensor) -> torch.Tensor:
    """Match diffusers' float32 numpy/PIL rounding; identity derivative only here."""
    _finite(rgb, "decoded RGB")
    if rgb.ndim != 4 or rgb.shape[:2] != (1, 3):
        raise ValueError("decoded RGB must be single-image NCHW")
    if bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("pipeline postprocessed RGB must already be in [0,1]")
    value = rgb.float()
    stored = torch.round(value.detach() * 255.0).to(torch.uint8).cpu().numpy()
    quantized = torch.from_numpy(stored.astype(np.float32) / np.float32(255.0)).to(value.device)
    # A custom function gives exactly quantized forward values, unlike x+(q-x).
    return _RoundBoundary.apply(value, quantized)


class _RoundBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, quantized):
        return quantized

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _to_image(quantized_rgb):
    pixels = torch.round(quantized_rgb.detach().float().cpu()[0] * 255.0)
    return Image.fromarray(pixels.to(torch.uint8).permute(1, 2, 0).numpy(), mode="RGB")


@contextmanager
def _cpu_saved_tensors(*, stable_modules=()):
    """Offload activations and reuse copies of explicitly stable frozen weights.

    Only storages owned by the supplied frozen modules are eligible. Holding
    detached references prevents their storage addresses being recycled; view
    metadata and version counters distinguish saved views and in-place updates.
    Activations are never memoized by address. Modules which temporarily change
    dtype (the official SDXL VAE) must not be listed in ``stable_modules``.
    """
    references = {}
    cache = {}
    stats = {"stable_parameter_cpu_copies": 0, "stable_parameter_cache_hits": 0,
             "uncached_cpu_copies": 0}

    def storage_key(value):
        storage = value.untyped_storage()
        return (value.device, value.dtype, storage.data_ptr(), storage.nbytes())

    for module in stable_modules:
        _require_frozen(module, "stable saved-tensor module")
        for parameter in module.parameters():
            # A detached reference retains the original storage even if a caller
            # replaces parameter.data. Such replacements are not cache entries.
            reference = parameter.detach()
            references.setdefault(storage_key(reference), []).append(reference)

    def pack(value):
        key = storage_key(value)
        owners = references.get(key)
        if owners is not None:
            view = (key, tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                    value._version, tuple(owner._version for owner in owners))
            if view not in cache:
                cache[view] = value.detach().to("cpu", copy=True)
                stats["stable_parameter_cpu_copies"] += 1
            else:
                stats["stable_parameter_cache_hits"] += 1
            saved = cache[view]
        else:
            # Independent copies also protect saved SDXL VAE tensors from its
            # temporary float32/float16 conversions before backward.
            saved = value.detach().to("cpu", copy=True)
            stats["uncached_cpu_copies"] += 1
        return value.device, saved

    def unpack(saved):
        device, value = saved
        return value.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats


def _require_frozen(module, name):
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{name} must be a torch Module")
    if module.training:
        raise ValueError(f"{name} must be in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{name} parameters must be frozen")
    if any(getattr(child, "_hf_hook", None) is not None for child in module.modules()):
        raise ValueError(f"{name} must have accelerate hooks removed before VJP")


@contextmanager
def _count_method(module, method, prefix, counters):
    """Count actual calls, including failures, without changing their graph."""
    original = getattr(module, method)
    had_instance_attribute = method in module.__dict__

    def counted(*args, **kwargs):
        _count(counters, prefix + "_attempted")
        result = original(*args, **kwargs)
        _count(counters, prefix + "_returned")
        return result

    setattr(module, method, counted)
    try:
        yield
    finally:
        if had_instance_attribute:
            setattr(module, method, original)
        else:
            delattr(module, method)


def _reader_graph(rgb, vae, image_processor, counters):
    _require_frozen(vae, "observation VAE")
    device, dtype = _vae_device_dtype(vae)
    # The same frozen public processor is used; the writer's Tensor entry retains
    # autograd, whereas the independently read public feature uses PIL below.
    pixels = image_processor.preprocess(rgb).to(device=device, dtype=dtype)
    _count(counters, "reader_proxy_forward_calls")
    mode = vae.encode(pixels).latent_dist.mode()
    scale = _vae_config_number(vae.config, "scaling_factor", positive=True)
    shift = _vae_config_number(vae.config, "shift_factor", positive=False)
    result = ((mode - shift) * scale).reshape(-1)
    _finite(result, "proxy feature")
    _count(counters, "reader_proxy_forward_returned")
    return result


def _score(feature, vector):
    _finite(vector, "reader vector")
    if vector.ndim != 1 or vector.numel() != feature.numel():
        raise ValueError("reader vector must match the complete flattened public feature")
    return torch.dot(feature.double(), vector.to(feature.device, torch.float64))


def _actual_read(image, proxy_feature, vae, image_processor, vector, counters):
    _count(counters, "reader_ordinary_forward_calls")
    actual = _cpu(encode_final_rgb_image(image, image_processor, vae)).reshape(-1)
    _count(counters, "reader_ordinary_forward_returned")
    proxy = _cpu(proxy_feature)
    # Grad-enabled kernels can differ slightly from their no-grad counterparts.
    # Fixed dtype-based numerical check, never fitted to source or outcome.
    tolerance = {torch.float16: (0.002, 0.002), torch.bfloat16: (0.02, 0.02)}.get(
        actual.dtype, (1e-5, 1e-6)
    )
    if actual.shape != proxy.shape or not torch.allclose(actual, proxy, rtol=tolerance[0], atol=tolerance[1]):
        raise ValueError("ordinary RGB feature and differentiable reader forward disagree")
    ordinary_score = float(_score(actual, vector))
    proxy_score = float(_score(proxy, vector))
    return actual, ordinary_score, {
        "reader_proxy_max_abs_error": float((actual.double() - proxy.double()).abs().max()),
        "reader_proxy_rtol": tolerance[0], "reader_proxy_atol": tolerance[1],
        # Per-coordinate allclose is not a bound at the QIM decision scale.
        "reader_proxy_score": proxy_score, "reader_ordinary_score": ordinary_score,
        "reader_proxy_score_difference": proxy_score - ordinary_score,
        "reader_proxy_score_abs_error": abs(proxy_score - ordinary_score),
        "reader_forward_check": "feature_allclose_only_not_scalar_equivalence",
    }


def rgb_reader_vjp(image, *, vae, image_processor, reader_vector, counters=None) -> RGBSensitivity:
    counters = _ledger(counters)
    partial = {"image": require_ordinary_rgb_image(image)}
    try:
        device, _ = _vae_device_dtype(vae)
        rgb = image_tensor(image, device=device).requires_grad_(True)
        with torch.enable_grad(), _cpu_saved_tensors(stable_modules=(vae,)):
            feature = _reader_graph(rgb, vae, image_processor, counters)
            actual, score, diagnostics = _actual_read(image, feature, vae, image_processor, reader_vector, counters)
            partial.update(feature=actual, score=score)
            _count(counters, "reader_vjp_calls")
            gradient, = torch.autograd.grad(_score(feature, reader_vector), rgb)
        _finite(gradient, "reader RGB sensitivity")
        _count(counters, "reader_vjp_returned")
        return RGBSensitivity(partial["image"], actual, score, _cpu(gradient), diagnostics)
    except Exception as error:
        raise RuntimeStageError("clean_rgb_vjp", error, partial, counters) from error


class _CaptureCallback:
    tensor_inputs = _ReplayCallback.tensor_inputs

    def __init__(self, settings, generator, reference=None, delta=None, counters=None):
        self.settings, self.generator, self.reference = settings, generator, reference
        self.delta, self.counters = delta, _ledger(counters)
        self.delegate = None
        self.scheduler_state = None
        self.actual_delta = self.after_latents = None
        self.metrics = {}

    def __call__(self, pipeline, index, timestep, kwargs):
        _count(self.counters, "sd35_full_denoising_steps")
        if self.delegate is None and index == self.settings.injection_step:
            self.delegate = _ReplayCallback(self.settings, torch.ones_like(kwargs["latents"]),
                                             0, self.generator, self.reference)
        if self.delegate is None:
            return kwargs
        result = self.delegate(pipeline, index, timestep, kwargs)
        if index == self.settings.injection_step:
            # Preserve every scheduler field, not just sigma/index. Configuration
            # objects retain their original type; schedule tensors become CPU.
            self.scheduler_state = deepcopy(pipeline.scheduler.__dict__)
            for key, value in self.scheduler_state.items():
                if isinstance(value, torch.Tensor):
                    self.scheduler_state[key] = _cpu(value)
            latent = kwargs["latents"]
            if self.delta is not None:
                _finite(self.delta, "requested write")
                if self.delta.shape != latent.shape:
                    raise ValueError("requested write shape differs from captured latent")
                updated = (latent.float() + self.delta.to(latent.device, torch.float32)).to(latent.dtype)
                _finite(updated, "actual-dtype write")
            else:
                updated = latent
            self.after_latents = _cpu(updated)
            self.actual_delta = _cpu(updated.double() - latent.double())
            self.metrics = {
                "latent_dtype": str(latent.dtype),
                "actual_relative_l2": float(torch.linalg.vector_norm(self.actual_delta)) /
                                      float(torch.linalg.vector_norm(latent.double())),
                "nonzero_elements": int(torch.count_nonzero(self.actual_delta)),
            }
            result["latents"] = updated
        return result


def _full_replay(pipeline, settings, *, reference=None, delta=None, counters=None):
    counters = _ledger(counters)
    generator = torch.Generator(device=settings.device).manual_seed(settings.seed)
    callback = _CaptureCallback(settings, generator, reference, delta, counters)
    partial = {}
    try:
        _count(counters, "sd35_full_generation_calls")
        with torch.no_grad():
            result = pipeline(prompt=settings.prompt, num_inference_steps=settings.num_inference_steps,
                              guidance_scale=settings.guidance_scale, height=settings.height, width=settings.width,
                              generator=generator, output_type="pil", callback_on_step_end=callback,
                              callback_on_step_end_tensor_inputs=list(callback.tensor_inputs))
        _count(counters, "sd35_full_generation_returned")
        if len(result.images) != 1:
            raise ValueError("source pipeline must return exactly one image")
        image = require_ordinary_rgb_image(result.images[0])
        partial["image"] = image
        if callback.delegate is None or callback.delegate.before_state is None:
            raise ValueError("source omitted the requested capture step")
        remaining = tuple(callback.delegate.remaining_steps)
        if remaining != tuple(range(settings.injection_step + 1, settings.num_inference_steps)):
            raise ValueError("source did not run every remaining step")
        return image, callback, remaining
    except Exception as error:
        partial.update(before_state={} if callback.delegate is None else callback.delegate.before_state,
                       scheduler_state=callback.scheduler_state, actual_delta=callback.actual_delta,
                       after_latents=callback.after_latents, injection_metrics=callback.metrics)
        raise RuntimeStageError("source_full_replay", error, partial, counters) from error


def capture_source(pipeline, settings: GenerationSettings, *, counters=None) -> SourceCapture:
    image, callback, remaining = _full_replay(pipeline, settings, counters=counters)
    return SourceCapture(image, callback.delegate.before_state, callback.scheduler_state, remaining)


def replay_write(pipeline, settings: GenerationSettings, capture: SourceCapture, delta,
                 *, counters=None) -> WriteReplay:
    image, callback, remaining = _full_replay(pipeline, settings, reference=capture.before_state,
                                             delta=delta, counters=counters)
    differences = _different_paths(capture.scheduler_state, callback.scheduler_state, "scheduler_state")
    if differences:
        raise RuntimeStageError("write_scheduler_match", ReplayStateMismatch(differences),
                                {"image": image, "actual_delta": callback.actual_delta,
                                 "after_latents": callback.after_latents}, _ledger(counters))
    return WriteReplay(image, callback.delegate.before_state, callback.actual_delta,
                       callback.after_latents, callback.metrics, remaining)


def source_suffix_vjps(pipeline, capture: SourceCapture, rgb_cotangents: torch.Tensor,
                       *, counters=None, diagnostics=None) -> torch.Tensor:
    """One captured official SD3 suffix forward, multiple independent VJPs.

    Cotangents are write-side auxiliaries; they are never supplied to the reader.
    A pixel-exact ordinary RGB match to full zero replay is mandatory. Optional
    ``diagnostics`` records dtype/zero counts without treating a finite zero
    floating-point proxy gradient as evidence of a zero true channel response.
    """
    counters = _ledger(counters)
    diagnostics = {} if diagnostics is None else diagnostics
    partial = {"before_state": capture.before_state}
    try:
        _require_frozen(pipeline.transformer, "SD3 transformer")
        _require_frozen(pipeline.vae, "SD3 VAE")
        _finite(rgb_cotangents, "RGB cotangents")
        if rgb_cotangents.ndim != 5 or len(rgb_cotangents) < 1:
            raise ValueError("RGB cotangents must have shape [K,1,3,H,W]")
        state = capture.before_state
        if state["joint_attention_kwargs"] is not None:
            raise ValueError("suffix adapter does not support joint attention extensions")
        device = next(pipeline.transformer.parameters()).device
        latent = state["latents"].to(device).detach().requires_grad_(True)
        diagnostics.update(latent_leaf_dtype=str(latent.dtype), forward_latent_dtype=str(latent.dtype),
                           returned_gradient_dtype="torch.float32", gradient_rows=[],
                           zero_gradient_interpretation=(
                               "A finite zero proxy gradient is not proof of zero channel sensitivity; "
                               "internal floating-point backward casts can underflow or round."))
        scheduler = deepcopy(pipeline.scheduler)
        scheduler.__dict__.clear()
        scheduler.__dict__.update(deepcopy(capture.scheduler_state))
        scheduler.timesteps = scheduler.timesteps.to(device)
        if scheduler.step_index != state["next_step_index"]:
            raise ValueError("captured scheduler index is inconsistent")
        prompt = state["prompt_embeds"].to(device)
        pooled = state["pooled_prompt_embeds"].to(device)
        gradients = []
        with torch.enable_grad(), _cpu_saved_tensors(
                stable_modules=(pipeline.transformer, pipeline.vae)) as saved_tensor_stats:
            _count(counters, "sd35_suffix_forward_calls")
            current = latent
            for index in capture.remaining_steps:
                t = scheduler.timesteps[index]
                inputs = torch.cat([current] * 2) if state["do_classifier_free_guidance"] else current
                _count(counters, "sd35_suffix_denoising_steps")
                prediction = pipeline.transformer(hidden_states=inputs, timestep=t.expand(inputs.shape[0]),
                    encoder_hidden_states=prompt, pooled_projections=pooled,
                    joint_attention_kwargs=None, return_dict=False)[0]
                if state["do_classifier_free_guidance"]:
                    unconditional, conditional = prediction.chunk(2)
                    prediction = unconditional + state["guidance_scale"] * (conditional - unconditional)
                current = scheduler.step(prediction, t, current, return_dict=False)[0]
            _count(counters, "sd35_suffix_decode_calls")
            decoded = pipeline.vae.decode(current / pipeline.vae.config.scaling_factor +
                                          pipeline.vae.config.shift_factor, return_dict=False)[0]
            _count(counters, "sd35_suffix_decode_returned")
            rgb = pipeline.image_processor.postprocess(decoded, output_type="pt")
            proxy = quantized_rgb_proxy(rgb)
            image = _to_image(proxy)
            partial["image"] = image
            _count(counters, "sd35_suffix_forward_returned")
            if not np.array_equal(np.asarray(image), np.asarray(capture.image)):
                raise ValueError("gradient suffix RGB does not exactly reproduce captured zero RGB")
            if tuple(rgb_cotangents.shape[1:]) != tuple(proxy.shape):
                raise ValueError("RGB cotangent shape differs from final RGB")
            for index, cotangent in enumerate(rgb_cotangents):
                _count(counters, "sd35_suffix_vjp_calls")
                gradient, = torch.autograd.grad(proxy, latent, cotangent.to(proxy.device, proxy.dtype),
                                                retain_graph=index + 1 < len(rgb_cotangents))
                _finite(gradient, "source latent sensitivity")
                _count(counters, "sd35_suffix_vjp_returned")
                gradients.append(_cpu(gradient).float())
                diagnostics["gradient_rows"].append({
                    "backward_gradient_dtype": str(gradient.dtype),
                    "nonzero_elements": int(torch.count_nonzero(gradient)),
                    "total_elements": gradient.numel(),
                    "all_zero": not bool(torch.count_nonzero(gradient)),
                    "l2_norm": float(torch.linalg.vector_norm(gradient.double())),
                })
                partial["completed_gradients"] = tuple(gradients)
            diagnostics["saved_tensors"] = dict(saved_tensor_stats)
        return torch.stack(gradients)
    except Exception as error:
        raise RuntimeStageError("source_suffix_vjp", error, partial, counters) from error


def reconstruction_reader_vjp(pipeline, image, settings: ReconstructionSettings, seed: int,
                              *, observation_vae, observation_processor, reader_vector,
                              counters=None) -> RGBSensitivity:
    """Official complete SDXL img2img graph; gradient is w.r.t. input RGB.

    The fresh generator controls BOTH posterior reparameterization and scheduler
    noise. No latent input, prompt embeddings, IP adapter or reconstruction truth
    is passed to the reader. This function must run after releasing SD3's
    transformer, keeping only its frozen observation VAE.
    """
    counters = _ledger(counters)
    partial = {"input_image": require_ordinary_rgb_image(image)}
    try:
        if type(seed) is not int or seed < 0:
            raise ValueError("design seed must be a nonnegative integer")
        _require_frozen(pipeline.unet, "SDXL UNet")
        _require_frozen(pipeline.vae, "SDXL VAE")
        _require_frozen(observation_vae, "observation VAE")
        if getattr(pipeline, "watermark", None) is not None:
            raise ValueError("SDXL's optional output watermark must be disabled")
        if getattr(pipeline, "_all_hooks", ()):
            raise ValueError("SDXL pipeline offload hooks must be removed before VJP")
        unwrapped = getattr(type(pipeline).__call__, "__wrapped__", None)
        if not callable(unwrapped):
            raise TypeError("SDXL pipeline must expose its torch.no_grad __wrapped__ call")
        rgb = image_tensor(image, device=settings.device).requires_grad_(True)
        # Prove the tensor entry takes the same ordinary-RGB preprocessing path.
        # This rejects resize/conversion discrepancies before using the proxy.
        with torch.no_grad():
            from_pil = pipeline.image_processor.preprocess(image).cpu()
            from_tensor = pipeline.image_processor.preprocess(rgb.detach()).cpu()
        if from_pil.shape != from_tensor.shape or not torch.equal(from_pil, from_tensor):
            raise ValueError("SDXL tensor/PIL input preprocessing does not agree exactly")
        generator = torch.Generator(device=settings.device).manual_seed(seed)
        initial_rng = _cpu(generator.get_state())
        actual_steps = []

        def counted_step(pipe, index, timestep, kwargs):
            _count(counters, "sdxl_denoising_steps")
            actual_steps.append(float(torch.as_tensor(timestep)))
            return kwargs

        with torch.enable_grad(), _cpu_saved_tensors(stable_modules=(pipeline.unet, observation_vae)), \
             _count_method(pipeline.vae, "encode", "sdxl_input_vae_encode", counters), \
             _count_method(pipeline.vae, "decode", "sdxl_output_vae_decode", counters), \
             _count_method(pipeline.unet, "forward", "sdxl_unet_forward", counters):
            _count(counters, "sdxl_reconstruction_forward_calls")
            result = unwrapped(pipeline, prompt=settings.prompt, image=rgb,
                strength=settings.strength, guidance_scale=settings.guidance_scale,
                num_inference_steps=settings.num_inference_steps, generator=generator,
                output_type="pt", callback_on_step_end=counted_step,
                callback_on_step_end_tensor_inputs=["latents"])
            _count(counters, "sdxl_reconstruction_forward_returned")
            pixels = result.images
            if not isinstance(pixels, torch.Tensor) or not pixels.requires_grad:
                raise ValueError("official SDXL RGB output is detached from the input")
            if not actual_steps:
                raise ValueError("SDXL omitted the actual denoising steps")
            proxy = quantized_rgb_proxy(pixels)
            reconstructed = _to_image(proxy)
            partial["image"] = reconstructed
            feature = _reader_graph(proxy, observation_vae, observation_processor, counters)
            actual, score, diagnostics = _actual_read(reconstructed, feature, observation_vae,
                                                     observation_processor, reader_vector, counters)
            partial.update(feature=actual, score=score)
            _count(counters, "sdxl_reconstruction_vjp_calls")
            _count(counters, "reader_vjp_calls")
            gradient, = torch.autograd.grad(_score(feature, reader_vector), rgb)
        _finite(gradient, "reconstruction RGB sensitivity")
        _count(counters, "sdxl_reconstruction_vjp_returned")
        _count(counters, "reader_vjp_returned")
        diagnostics.update(actual_timesteps=actual_steps, design_seed=seed,
                           generator_state_before=initial_rng,
                           generator_state_after=_cpu(generator.get_state()),
                           sensitivity_kind="complete_channel_uint8_rounding_STE")
        return RGBSensitivity(reconstructed, actual, score, _cpu(gradient), diagnostics)
    except Exception as error:
        raise RuntimeStageError("reconstruction_rgb_vjp", error, partial, counters) from error
