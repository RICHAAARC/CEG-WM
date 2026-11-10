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


RGB_PROXY_COMPATIBILITY_VERSION = "rgb-pil-layout-v2"


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


def _tensor_layout(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "storage_offset": value.storage_offset(), "contiguous": value.is_contiguous(),
            "channels_last": value.ndim == 4 and value.is_contiguous(memory_format=torch.channels_last)}


def _execution_metadata():
    # These are enabled policies, not a claim about the kernel actually selected.
    cuda = torch.backends.cuda
    sdpa = {}
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        getter = getattr(cuda, name, None)
        sdpa[name] = bool(getter()) if callable(getter) else None
    return {"torch_version": str(torch.__version__), "grad_enabled": torch.is_grad_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": cuda.matmul.allow_tf32,
            "sdpa_enabled_policies_not_selected_kernel": sdpa}


def _canonical_preprocessed_pixels(proxy, ordinary, *, diagnostics):
    """Copy only layout; reject changed values, shape or dtype before inference.

    PIL preprocessing normally produces a channels-last tensor, while the
    Tensor branch's cat/interpolate can produce contiguous NCHW. Equal values
    alone do not ensure the same convolution execution in half precision.
    ``copy_`` into the observed ordinary stride is differentiable and has the
    identity derivative; no rounding, value replacement or tolerance is added.
    """
    if not isinstance(proxy, torch.Tensor) or not isinstance(ordinary, torch.Tensor):
        raise TypeError("PIL and Tensor preprocessing must both return tensors")
    diagnostics.update(ordinary=_tensor_layout(ordinary), proxy_before=_tensor_layout(proxy))
    _finite(ordinary, "ordinary preprocessed pixels")
    _finite(proxy, "proxy preprocessed pixels")
    diagnostics.update(shape_equal=proxy.shape == ordinary.shape, dtype_equal=proxy.dtype == ordinary.dtype)
    if proxy.shape != ordinary.shape or proxy.dtype != ordinary.dtype:
        raise ValueError("PIL/Tensor preprocessing shape or dtype differs")
    diagnostics["values_equal"] = torch.equal(proxy.detach().cpu(), ordinary.detach().cpu())
    diagnostics["max_abs_error"] = float((proxy.detach().double().cpu() - ordinary.detach().double().cpu()).abs().max())
    if not diagnostics["values_equal"]:
        raise ValueError("PIL/Tensor input preprocessing does not agree exactly")
    if proxy.stride() == ordinary.stride():
        canonical = proxy
    else:
        canonical = torch.empty_strided(proxy.shape, ordinary.stride(), dtype=proxy.dtype,
                                        device=proxy.device).copy_(proxy)
    diagnostics["proxy_after"] = _tensor_layout(canonical)
    if canonical.stride() != ordinary.stride():
        raise ValueError("could not preserve the ordinary PIL preprocessing layout")
    return canonical


@contextmanager
def _canonical_pipeline_preprocess(processor, rgb, ordinary_image, diagnostics):
    """Wrap only this official pipeline input; always restore the original API."""
    original = processor.preprocess
    had_instance_attribute = "preprocess" in processor.__dict__

    def preprocess(value, *args, **kwargs):
        processed = original(value, *args, **kwargs)
        if value is not rgb:
            return processed
        with torch.no_grad():
            ordinary = original(ordinary_image, *args, **kwargs)
        return _canonical_preprocessed_pixels(processed, ordinary, diagnostics=diagnostics)

    processor.preprocess = preprocess
    try:
        yield
    finally:
        if had_instance_attribute:
            processor.preprocess = original
        else:
            delattr(processor, "preprocess")


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


def _nonempty_dense_nonoverlapping(value):
    """Conservative dense-view check using only public tensor metadata.

    Ignore size-one dimensions: their strides cannot alias distinct elements.
    Every other dimension must exactly extend a contiguous span, in increasing
    stride order. This accepts permutations (including transposes) and scalars,
    but rejects empty, holey and overlapping layouts. Storage offset is checked
    separately by the saved-weight policy; no private PyTorch import is needed.
    """
    if value.layout != torch.strided or value.numel() == 0:
        return False
    expected_stride = 1
    dimensions = sorted((stride, size) for size, stride in zip(value.shape, value.stride()) if size > 1)
    for stride, size in dimensions:
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


@contextmanager
def _cpu_saved_tensors(*, stable_modules=(), retain_stable_storage=False):
    """Offload activations and reuse copies of explicitly stable frozen weights.

    Only storages owned by the supplied frozen modules are eligible. Holding
    detached references prevents their storage addresses being recycled; view
    metadata and version counters distinguish saved views and in-place updates.
    Activations are never memoized by address. Modules which temporarily change
    dtype (the official SDXL VAE) must not be listed in ``stable_modules``.
    V2 source-only opt-in may reference dense, zero-offset stable weight views
    instead of copying them. Version/identity/storage/layout/lifetime guards
    cover the fixed synchronous runtime, not unsupported raw-storage/.data
    writes that bypass PyTorch version tracking. No global dispatch is changed.
    """
    references = {}
    cache = {}
    stats = {"stable_parameter_cpu_copies": 0, "stable_parameter_cache_hits": 0,
             "uncached_cpu_copies": 0, "stable_parameter_reference_saves": 0,
             "stable_parameter_reference_views": 0, "layout_copy_fallbacks": 0,
             "stable_parameter_cpu_copy_bytes": 0, "activation_cpu_copy_bytes": 0,
             "stable_parameter_cpu_bytes_avoided": 0}
    owners_by_storage = {}
    active = True

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
        if retain_stable_storage:
            for child in module.modules():
                for name, parameter in child.named_parameters(recurse=False):
                    owners_by_storage.setdefault(storage_key(parameter), []).append(
                        (child, name, parameter, parameter._version, tuple(parameter.shape),
                         tuple(parameter.stride()), parameter.storage_offset()))

    def check_owners(key):
        for child, name, parameter, version, shape, stride, offset in owners_by_storage[key]:
            if (child._parameters.get(name) is not parameter or parameter._version != version
                    or storage_key(parameter) != key or tuple(parameter.shape) != shape
                    or tuple(parameter.stride()) != stride or parameter.storage_offset() != offset
                    or parameter.requires_grad):
                raise RuntimeError("stable saved weight changed during guarded source VJP")

    def pack(value):
        key = storage_key(value)
        owners = references.get(key)
        if owners is not None:
            view = (key, tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                    value._version, tuple(owner._version for owner in owners))
            if retain_stable_storage:
                check_owners(key)
                # preserve_format copies retain dense (including transposed)
                # strides. Holey/overlapping/offset views conservatively use the
                # old copy path; returning them directly can alter kernel layout.
                if (value.layout == torch.strided and value.numel() > 0
                        and value.storage_offset() == 0 and _nonempty_dense_nonoverlapping(value)):
                    if view not in cache:
                        cache[view] = value.detach()
                        stats["stable_parameter_reference_views"] += 1
                        stats["stable_parameter_cpu_bytes_avoided"] += value.numel() * value.element_size()
                    stats["stable_parameter_reference_saves"] += 1
                    return "reference", cache[view], key, value._version
                stats["layout_copy_fallbacks"] += 1
            if view not in cache:
                cache[view] = value.detach().to("cpu", copy=True)
                stats["stable_parameter_cpu_copies"] += 1
                stats["stable_parameter_cpu_copy_bytes"] += value.numel() * value.element_size()
            else:
                stats["stable_parameter_cache_hits"] += 1
            saved = cache[view]
        else:
            # Independent copies also protect saved SDXL VAE tensors from its
            # temporary float32/float16 conversions before backward.
            saved = value.detach().to("cpu", copy=True)
            stats["uncached_cpu_copies"] += 1
            stats["activation_cpu_copy_bytes"] += value.numel() * value.element_size()
        return value.device, saved

    def unpack(saved):
        if len(saved) == 4:
            _, value, key, version = saved
            if not active:
                raise RuntimeError("stable weight reference outlived its guarded source VJP context")
            check_owners(key)
            if value._version != version:
                raise RuntimeError("stable saved weight view was modified")
            return value
        device, value = saved
        return value.to(device)

    try:
        # Fixed official eval/frozen modules never mutate raw storage or use
        # .data writes in this synchronous interval. Such version-bypassing
        # external writes are unsupported; do not change aten dispatch to police
        # arbitrary third-party code. Ordinary mutations/replacement/migration
        # are checked on pack/unpack and before leaving the context.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield stats
            if retain_stable_storage:
                for key in owners_by_storage:
                    check_owners(key)
    finally:
        active = False


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
def _count_method(module, method, prefix, counters, *, input_diagnostics=None):
    """Count actual calls, including failures, without changing their graph."""
    original = getattr(module, method)
    had_instance_attribute = method in module.__dict__

    def counted(*args, **kwargs):
        _count(counters, prefix + "_attempted")
        if input_diagnostics is not None:
            first = args[0] if args else kwargs.get("x")
            input_diagnostics.append({"tensor": _tensor_layout(first) if isinstance(first, torch.Tensor) else None,
                                      "execution": _execution_metadata()})
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


def _reader_graph(rgb, vae, image_processor, counters, *, ordinary_image, diagnostics):
    _require_frozen(vae, "observation VAE")
    device, dtype = _vae_device_dtype(vae)
    # The same frozen public processor is used; the writer's Tensor entry retains
    # autograd, whereas the independently read public feature uses PIL below.
    with torch.no_grad():
        ordinary_pixels = image_processor.preprocess(ordinary_image)
    pixels = _canonical_preprocessed_pixels(image_processor.preprocess(rgb), ordinary_pixels,
                                            diagnostics=diagnostics.setdefault("reader_preprocess", {}))
    pixels = pixels.to(device=device, dtype=dtype)
    diagnostics["reader_vae_input"] = _tensor_layout(pixels)
    diagnostics["reader_ordinary_vae_input"] = _tensor_layout(ordinary_pixels.to(device=device, dtype=dtype))
    diagnostics["reader_proxy_execution"] = _execution_metadata()
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


def _actual_read(image, proxy_feature, vae, image_processor, vector, counters, *, partial=None, diagnostics=None):
    partial = {} if partial is None else partial
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["compatibility_version"] = RGB_PROXY_COMPATIBILITY_VERSION
    partial["reader_diagnostics"] = diagnostics
    proxy = _cpu(proxy_feature)
    partial["reader_proxy_feature"] = proxy
    _count(counters, "reader_ordinary_forward_calls")
    with torch.no_grad():
        diagnostics["reader_ordinary_execution"] = _execution_metadata()
        actual = _cpu(encode_final_rgb_image(image, image_processor, vae)).reshape(-1)
    partial["reader_ordinary_feature"] = actual
    _count(counters, "reader_ordinary_forward_returned")
    # Grad-enabled kernels can differ slightly from their no-grad counterparts.
    # Fixed dtype-based numerical check, never fitted to source or outcome.
    tolerance = {torch.float16: (0.002, 0.002), torch.bfloat16: (0.02, 0.02)}.get(
        actual.dtype, (1e-5, 1e-6)
    )
    shape_equal = actual.shape == proxy.shape
    diagnostics.update({
        "reader_feature_shape_equal": shape_equal,
        "reader_proxy_rtol": tolerance[0], "reader_proxy_atol": tolerance[1],
        # Per-coordinate allclose is not a bound at the QIM decision scale.
        "reader_forward_check": "feature_allclose_only_not_scalar_equivalence",
    })
    if shape_equal:
        diagnostics["reader_proxy_max_abs_error"] = float((actual.double() - proxy.double()).abs().max())
    # Preserve score errors before the allclose guard, including on failure.
    ordinary_score = float(_score(actual, vector))
    proxy_score = float(_score(proxy, vector))
    diagnostics.update({
        "reader_proxy_score": proxy_score, "reader_ordinary_score": ordinary_score,
        "reader_proxy_score_difference": proxy_score - ordinary_score,
        "reader_proxy_score_abs_error": abs(proxy_score - ordinary_score),
        "reader_feature_allclose": shape_equal and torch.allclose(actual, proxy, rtol=tolerance[0], atol=tolerance[1]),
    })
    if not diagnostics["reader_feature_allclose"]:
        raise ValueError("ordinary RGB feature and differentiable reader forward disagree")
    return actual, ordinary_score, diagnostics


def rgb_reader_vjp(image, *, vae, image_processor, reader_vector, counters=None) -> RGBSensitivity:
    counters = _ledger(counters)
    partial = {"image": require_ordinary_rgb_image(image)}
    diagnostics = {"compatibility_version": RGB_PROXY_COMPATIBILITY_VERSION}
    partial["reader_diagnostics"] = diagnostics
    try:
        device, _ = _vae_device_dtype(vae)
        rgb = image_tensor(image, device=device).requires_grad_(True)
        with torch.enable_grad(), _cpu_saved_tensors(stable_modules=(vae,)):
            feature = _reader_graph(rgb, vae, image_processor, counters, ordinary_image=image, diagnostics=diagnostics)
            actual, score, diagnostics = _actual_read(image, feature, vae, image_processor, reader_vector, counters,
                                                      partial=partial, diagnostics=diagnostics)
            partial.update(feature=actual, score=score)
            _count(counters, "reader_vjp_calls")
            gradient, = torch.autograd.grad(_score(feature, reader_vector), rgb)
            partial.update(gradient_rgb=_cpu(gradient), diagnostics=diagnostics)
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


def _restore_source_schedule_devices(scheduler, state, diagnostics):
    """Restore both schedules, without changing dtype or the saved CPU state.

    Legacy captures save tensors on CPU without recording individual original
    devices. In the fixed diffusers 0.39 SD3 non-XLA FlowMatch path, set_timesteps
    receives the execution device and retains BOTH timesteps and sigmas there
    (pipeline lines 965, 1031--1041; scheduler lines 365--385). The captured
    latent_device is therefore the restoration device for these two tensors.
    """
    target = torch.device(state["latent_device"])
    diagnostics.update(
        basis="diffusers_0.39_SD3_non_XLA_FlowMatch_schedule_device_equals_captured_latent_device",
        inferred_for_legacy_capture=True, captured_latent_device=str(target),
        scheduler_class=state["scheduler_class"], tensors={},
    )
    for name in ("timesteps", "sigmas"):
        saved = getattr(scheduler, name)
        diagnostics["tensors"][name] = {"saved": _tensor_layout(saved)}
        restored = saved.to(device=target)
        setattr(scheduler, name, restored)
        diagnostics["tensors"][name]["restored"] = _tensor_layout(restored)


def _source_suffix_forward(pipeline, capture, *, counters, diagnostics, partial, delta=None,
                           after_step=None, step_observer=None):
    """The production suffix, shared verbatim by VJP and forward-only tracing.

    The caller controls torch grad mode and the CPU saved-tensor context. The
    input leaf always requires grad, including when that context disables it.
    Only detached snapshots are added to ``partial``; the returned tensors keep
    the live graph for the production VJP caller.
    """
    _require_frozen(pipeline.transformer, "SD3 transformer")
    _require_frozen(pipeline.vae, "SD3 VAE")
    state = capture.before_state
    if state["joint_attention_kwargs"] is not None:
        raise ValueError("suffix adapter does not support joint attention extensions")
    device = next(pipeline.transformer.parameters()).device
    original = state["latents"].to(device).detach()
    if delta is None:
        updated = original
    else:
        _finite(delta, "requested suffix write")
        if delta.shape != original.shape:
            raise ValueError("requested suffix write shape differs from captured latent")
        updated = (original.float() + delta.to(device=device, dtype=torch.float32)).to(original.dtype)
        _finite(updated, "actual-dtype suffix write")
    latent = updated.detach().requires_grad_(True)
    if delta is not None:
        partial["actual_delta"] = _cpu(updated.double() - original.double())
        partial["after_latents"] = _cpu(updated)
    diagnostics.update(
        before_latents=_tensor_layout(latent), latent_leaf_dtype=str(latent.dtype),
        forward_latent_dtype=str(latent.dtype), leaf_requires_grad=latent.requires_grad,
        grad_enabled=torch.is_grad_enabled(), execution=_execution_metadata(), steps=[],
        module_modes={"transformer_training": pipeline.transformer.training,
                      "vae_training": pipeline.vae.training,
                      "transformer_parameters_require_grad": any(p.requires_grad for p in pipeline.transformer.parameters()),
                      "vae_parameters_require_grad": any(p.requires_grad for p in pipeline.vae.parameters())},
    )
    partial["before_latents"] = _cpu(latent)
    partial["suffix_before_latents"] = partial["before_latents"]  # Existing failure artifact name.
    partial["steps"] = []
    partial["scheduler_step_outputs"] = []
    scheduler = deepcopy(pipeline.scheduler)
    scheduler.__dict__.clear()
    scheduler.__dict__.update(deepcopy(capture.scheduler_state))
    _restore_source_schedule_devices(scheduler, state, diagnostics.setdefault("schedule_restore", {}))
    if scheduler.step_index != state["next_step_index"]:
        raise ValueError("captured scheduler index is inconsistent")
    diagnostics["scheduler_step_index_before"] = scheduler.step_index
    prompt = state["prompt_embeds"].to(device)
    pooled = state["pooled_prompt_embeds"].to(device)
    diagnostics["prompt_embeds"] = _tensor_layout(prompt)
    diagnostics["pooled_prompt_embeds"] = _tensor_layout(pooled)
    _count(counters, "sd35_suffix_forward_calls")
    current = latent
    for index in capture.remaining_steps:
        if step_observer is not None:
            step_observer("before_step", index)
        t = scheduler.timesteps[index]
        inputs = torch.cat([current] * 2) if state["do_classifier_free_guidance"] else current
        _count(counters, "sd35_suffix_denoising_steps")
        prediction = pipeline.transformer(hidden_states=inputs, timestep=t.expand(inputs.shape[0]),
            encoder_hidden_states=prompt, pooled_projections=pooled,
            joint_attention_kwargs=None, return_dict=False)[0]
        if state["do_classifier_free_guidance"]:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + state["guidance_scale"] * (conditional - unconditional)
        step = {"index": index, "timestep": _cpu(t), "guided_prediction": _cpu(prediction)}
        partial["steps"].append(step)
        step_metadata = {"index": index, "timestep": float(t), "transformer_input": _tensor_layout(inputs),
                         "guided_prediction": _tensor_layout(prediction),
                         "guided_prediction_requires_grad": prediction.requires_grad}
        diagnostics["steps"].append(step_metadata)
        current = scheduler.step(prediction, t, current, return_dict=False)[0]
        step["scheduler_output"] = _cpu(current)
        partial["scheduler_step_outputs"].append({"index": index, "latents": step["scheduler_output"]})
        step_metadata.update(scheduler_output=_tensor_layout(current),
                             scheduler_output_requires_grad=current.requires_grad)
        if after_step is not None:
            after_step(index, current, scheduler)
        if step_observer is not None:
            step_observer("after_step", index)
    diagnostics["scheduler_step_index_after"] = scheduler.step_index
    _count(counters, "sd35_suffix_decode_calls")
    pre_vae = current / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    partial["pre_vae_latents"] = _cpu(pre_vae)
    diagnostics["pre_vae_latents"] = _tensor_layout(pre_vae)
    decoded = pipeline.vae.decode(pre_vae, return_dict=False)[0]
    partial["decoded"] = _cpu(decoded)
    diagnostics["decoded"] = _tensor_layout(decoded)
    diagnostics["decoded_requires_grad"] = decoded.requires_grad
    _count(counters, "sd35_suffix_decode_returned")
    rgb = pipeline.image_processor.postprocess(decoded, output_type="pt")
    partial["postprocessed_rgb"] = _cpu(rgb)
    diagnostics["postprocessed_rgb"] = _tensor_layout(rgb)
    diagnostics["postprocessed_rgb_requires_grad"] = rgb.requires_grad
    proxy = quantized_rgb_proxy(rgb)
    image = _to_image(proxy)
    partial["image"] = image
    partial["uint8_rgb"] = torch.from_numpy(np.array(image, copy=True))
    _count(counters, "sd35_suffix_forward_returned")
    diagnostics["zero_rgb_comparison"] = _rgb_comparison(image, capture.image)
    return latent, proxy


def _rgb_comparison(image, reference):
    actual_pixels, reference_pixels = np.asarray(image), np.asarray(reference)
    comparison = {"exact_equal": np.array_equal(actual_pixels, reference_pixels),
                  "shape_equal": actual_pixels.shape == reference_pixels.shape,
                  "actual_shape": list(actual_pixels.shape), "reference_shape": list(reference_pixels.shape)}
    if comparison["shape_equal"]:
        difference = actual_pixels.astype(np.int16) - reference_pixels.astype(np.int16)
        comparison.update(changed_components=int(np.count_nonzero(difference)),
                          total_components=int(difference.size),
                          max_abs_uint8_difference=int(np.abs(difference).max()),
                          mean_abs_uint8_difference=float(np.abs(difference).mean()))
    return comparison


def replay_suffix_write(pipeline, capture: SourceCapture, delta=None, *, counters=None,
                        diagnostics=None) -> WriteReplay:
    """Render a zero or write from the fixed capture through the same suffix.

    ``delta=None`` defines a new runtime suffix baseline, independently of the
    retained historical full-generation image. Writes use FP32 addition and the
    captured latent dtype, without clipping, budget repair, retries or a full
    generation. This is a distinct synthesis path from ``replay_write``.
    """
    counters = _ledger(counters)
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["write_path"] = "fixed_capture_suffix"
    partial = {"before_state": capture.before_state, "source_suffix_diagnostics": diagnostics}
    try:
        _count(counters, "sd35_suffix_write_calls")
        with torch.no_grad(), _cpu_saved_tensors(
                stable_modules=(pipeline.transformer, pipeline.vae)) as saved_tensor_stats:
            diagnostics["saved_tensors"] = saved_tensor_stats
            _source_suffix_forward(pipeline, capture, counters=counters,
                                   diagnostics=diagnostics, partial=partial, delta=delta)
        original = capture.before_state["latents"].detach().cpu()
        after = partial["before_latents"]
        actual = after.double() - original.double()
        original_norm = float(torch.linalg.vector_norm(original.double()))
        actual_norm = float(torch.linalg.vector_norm(actual))
        metrics = {"latent_dtype": str(after.dtype),
                   "requested_l2": 0.0 if delta is None else float(torch.linalg.vector_norm(delta.detach().double())),
                   "actual_l2": actual_norm,
                   "actual_relative_l2": actual_norm / original_norm if original_norm else None,
                   "nonzero_elements": int(torch.count_nonzero(actual))}
        partial.update(actual_delta=actual, after_latents=after, injection_metrics=metrics)
        diagnostics["injection_metrics"] = metrics
        _count(counters, "sd35_suffix_write_returned")
        return WriteReplay(partial["image"], capture.before_state, actual, after,
                           metrics, capture.remaining_steps)
    except Exception as error:
        raise RuntimeStageError("source_suffix_write", error, partial, counters) from error


def source_suffix_vjps(pipeline, capture: SourceCapture, rgb_cotangents: torch.Tensor,
                       *, counters=None, diagnostics=None, basepoint_mode="historical_capture",
                       reference_image=None, retain_stable_storage=False,
                       resource_observer=None) -> torch.Tensor:
    """One captured official SD3 suffix forward, multiple independent VJPs.

    Cotangents are write-side auxiliaries; they are never supplied to the reader.
    By default, the historical full zero image remains the exact-match gate.
    Explicit ``runtime_suffix`` mode instead requires the independently rendered
    current suffix zero image. Both modes retain an exact reference check before
    backward; a caller must derive its cotangents at the selected RGB basepoint.
    """
    counters = _ledger(counters)
    diagnostics = {} if diagnostics is None else diagnostics
    partial = {"before_state": capture.before_state, "source_suffix_diagnostics": diagnostics}
    try:
        if basepoint_mode not in ("historical_capture", "runtime_suffix"):
            raise ValueError("unknown suffix basepoint mode")
        if basepoint_mode == "historical_capture":
            if reference_image is not None:
                raise ValueError("historical capture mode does not accept a replacement reference")
            reference = capture.image
        else:
            if reference_image is None:
                raise ValueError("runtime suffix mode requires its independently rendered zero reference")
            reference = require_ordinary_rgb_image(reference_image)
        diagnostics["basepoint_mode"] = basepoint_mode
        _finite(rgb_cotangents, "RGB cotangents")
        if rgb_cotangents.ndim != 5 or len(rgb_cotangents) < 1:
            raise ValueError("RGB cotangents must have shape [K,1,3,H,W]")
        diagnostics.update(returned_gradient_dtype="torch.float32", gradient_rows=[],
                           zero_gradient_interpretation=(
                               "A finite zero proxy gradient is not proof of zero channel sensitivity; "
                               "internal floating-point backward casts can underflow or round."))
        gradients = []
        with torch.enable_grad(), _cpu_saved_tensors(
                stable_modules=(pipeline.transformer, pipeline.vae),
                retain_stable_storage=retain_stable_storage) as saved_tensor_stats:
            diagnostics["saved_tensors"] = saved_tensor_stats
            def observe(event, index=None):
                if resource_observer is not None:
                    resource_observer(event, step=index, saved_tensors=dict(saved_tensor_stats))
            observe("forward_start")
            latent, proxy = _source_suffix_forward(pipeline, capture, counters=counters,
                diagnostics=diagnostics, partial=partial,
                step_observer=observe if resource_observer is not None else None)
            observe("forward_returned")
            comparison = _rgb_comparison(partial["image"], reference)
            diagnostics["basepoint_rgb_comparison"] = comparison
            if not comparison["exact_equal"]:
                description = "captured zero RGB" if basepoint_mode == "historical_capture" else "runtime suffix zero RGB"
                raise ValueError(f"gradient suffix RGB does not exactly reproduce {description}")
            if tuple(rgb_cotangents.shape[1:]) != tuple(proxy.shape):
                raise ValueError("RGB cotangent shape differs from final RGB")
            for index, cotangent in enumerate(rgb_cotangents):
                observe("before_backward", index)
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
                observe("after_backward", index)
            diagnostics["saved_tensors"] = dict(saved_tensor_stats)
        return torch.stack(gradients)
    except Exception as error:
        raise RuntimeStageError("source_suffix_vjp", error, partial, counters) from error


def trace_source_suffix(pipeline, capture: SourceCapture, *, grad_enabled: bool, counters=None) -> dict[str, Any]:
    """Trace exactly the captured step-19 suffix, with no VJP or match gate.

    NG/G use the same leaf.requires_grad=True, saved-tensor policy, frozen models,
    captured input values/layout/dtype and freshly restored scheduler. Only torch
    grad mode changes. The ordinary zero-image comparison is diagnostic here;
    production ``source_suffix_vjps`` retains its strict pixel-exact guard.
    """
    counters = _ledger(counters)
    diagnostics = {}
    partial = {"before_state": capture.before_state, "metadata": diagnostics}
    try:
        if type(grad_enabled) is not bool:
            raise TypeError("grad_enabled must be an explicit bool")
        if tuple(capture.remaining_steps) != (19,) or capture.before_state["next_step_index"] != 19:
            raise ValueError("diagnostic trace requires exactly the captured step-19 suffix")
        with torch.set_grad_enabled(grad_enabled), _cpu_saved_tensors(
                stable_modules=(pipeline.transformer, pipeline.vae)) as saved_tensor_stats:
            diagnostics["saved_tensors"] = saved_tensor_stats
            _source_suffix_forward(pipeline, capture, counters=counters,
                                    diagnostics=diagnostics, partial=partial)
        return {name: partial[name] for name in ("image", "uint8_rgb", "before_latents", "steps",
                                                 "pre_vae_latents", "decoded", "postprocessed_rgb", "metadata")}
    except Exception as error:
        raise RuntimeStageError("source_suffix_trace", error, partial, counters) from error


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
    diagnostics = {"compatibility_version": RGB_PROXY_COMPATIBILITY_VERSION,
                   "sdxl_input_preprocess": {}}
    partial["reader_diagnostics"] = diagnostics
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
        generator = torch.Generator(device=settings.device).manual_seed(seed)
        initial_rng = _cpu(generator.get_state())
        actual_steps = []

        def counted_step(pipe, index, timestep, kwargs):
            _count(counters, "sdxl_denoising_steps")
            actual_steps.append(float(torch.as_tensor(timestep)))
            return kwargs

        with torch.enable_grad(), _cpu_saved_tensors(stable_modules=(pipeline.unet, observation_vae)), \
             _canonical_pipeline_preprocess(pipeline.image_processor, rgb, image, diagnostics["sdxl_input_preprocess"]), \
             _count_method(pipeline.vae, "encode", "sdxl_input_vae_encode", counters,
                           input_diagnostics=diagnostics.setdefault("sdxl_vae_encode_inputs", [])), \
             _count_method(pipeline.vae, "decode", "sdxl_output_vae_decode", counters), \
             _count_method(pipeline.unet, "forward", "sdxl_unet_forward", counters):
            _count(counters, "sdxl_reconstruction_forward_calls")
            diagnostics["sdxl_proxy_execution"] = _execution_metadata()
            result = unwrapped(pipeline, prompt=settings.prompt, image=rgb,
                strength=settings.strength, guidance_scale=settings.guidance_scale,
                num_inference_steps=settings.num_inference_steps, generator=generator,
                output_type="pt", callback_on_step_end=counted_step,
                callback_on_step_end_tensor_inputs=["latents"])
            _count(counters, "sdxl_reconstruction_forward_returned")
            if not diagnostics["sdxl_input_preprocess"].get("values_equal", False):
                raise ValueError("official SDXL did not preprocess the expected RGB input")
            pixels = result.images
            if not isinstance(pixels, torch.Tensor) or not pixels.requires_grad:
                raise ValueError("official SDXL RGB output is detached from the input")
            if not actual_steps:
                raise ValueError("SDXL omitted the actual denoising steps")
            proxy = quantized_rgb_proxy(pixels)
            reconstructed = _to_image(proxy)
            partial["image"] = reconstructed
            feature = _reader_graph(proxy, observation_vae, observation_processor, counters,
                                    ordinary_image=reconstructed, diagnostics=diagnostics)
            actual, score, diagnostics = _actual_read(reconstructed, feature, observation_vae,
                                                     observation_processor, reader_vector, counters,
                                                     partial=partial, diagnostics=diagnostics)
            partial.update(feature=actual, score=score)
            _count(counters, "sdxl_reconstruction_vjp_calls")
            _count(counters, "reader_vjp_calls")
            gradient, = torch.autograd.grad(_score(feature, reader_vector), rgb)
            partial.update(gradient_rgb=_cpu(gradient), diagnostics=diagnostics)
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
