"""V2 model adapter: two captures, one canonical suffix, staged model ownership.

No construction-time weights. Uses the retained full-VAE/Rademacher reader and
the retained RGB/latent proxy VJPs. CPU-saved activations do not establish that
eleven-step backward fits a real GPU/host. No large source/SDXL coexistence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from cegwm.channel_precompensation import make_reader
from cegwm.runtime import channel_precompensation_sd35 as rt
from cegwm.runtime.channel_precompensation_provider import PrecompensationProvider, _freeze
from cegwm.runtime.reconstruction_channel_sd35 import _snapshot


def schedule_snapshot(scheduler):
    # Retain FrozenDict/config types used by diffusers scheduler attributes.
    state = deepcopy(scheduler.__dict__)
    return {key: rt._cpu(value) if isinstance(value, torch.Tensor) else value
            for key, value in state.items()}


def capture_two(pipeline, settings, counters):
    generator = torch.Generator(device=settings.device).manual_seed(settings.seed)
    callbacks = {step: rt._ReplayCallback(replace(settings, injection_step=step),
                 torch.empty(0), 0, generator, None) for step in (8, 18)}
    schedules = {}
    partial = {"captures": {}}

    def callback(pipe, index, timestep, kwargs):
        for step, delegate in callbacks.items():
            if index == step:
                delegate.direction = torch.ones_like(kwargs["latents"])
                delegate(pipe, index, timestep, kwargs)
                schedules[step] = schedule_snapshot(pipe.scheduler)
                partial["captures"][step] = dict(before_state=delegate.before_state,
                                                 scheduler_state=schedules[step])
            elif index > step:
                delegate.remaining_steps.append(index)
        return kwargs

    try:
        rt._count(counters, "sd35_full_generation_calls")
        with torch.no_grad(), rt._count_method(pipeline.transformer, "forward", "sd35_transformer", counters):
            result = pipeline(prompt=settings.prompt, num_inference_steps=20,
                guidance_scale=settings.guidance_scale, height=settings.height, width=settings.width,
                generator=generator, output_type="pil", callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=list(rt._ReplayCallback.tensor_inputs))
        rt._count(counters, "sd35_full_generation_returned")
        if len(result.images) != 1:
            raise ValueError("one source RGB required")
        image = rt.require_ordinary_rgb_image(result.images[0])
        partial["image"] = image
        captures = {}
        for step, delegate in callbacks.items():
            remaining = tuple(delegate.remaining_steps)
            if delegate.before_state is None or remaining != tuple(range(step + 1, 20)):
                raise ValueError("incomplete source capture")
            captures[step] = rt.SourceCapture(image, delegate.before_state, schedules[step], remaining)
        return captures
    except Exception as error:
        raise rt.RuntimeStageError("two_source_captures", error, partial, counters) from error


def suffix(pipeline, capture, delta, counters, *, capture_late=False):
    diagnostics, partial, late = {}, {}, {}

    def after_step(index, latent, scheduler):
        if capture_late and index == 18:
            state = deepcopy(capture.before_state)
            state.update(latents=rt._cpu(latent), timestep=rt._cpu(scheduler.timesteps[18]),
                         interval_start_sigma=rt._cpu(scheduler.sigmas[18]),
                         latent_sigma=rt._cpu(scheduler.sigmas[19]),
                         next_sigma=rt._cpu(scheduler.sigmas[20]), next_step_index=19,
                         begin_index=getattr(scheduler, "begin_index", None))
            # The supported first-order source scheduler is deterministic; the
            # captured generator state therefore remains unchanged by suffix.
            late.update(before_state=state, scheduler_state=schedule_snapshot(scheduler))
            partial["late_state"] = late

    try:
        with torch.no_grad(), rt._count_method(pipeline.transformer, "forward", "sd35_transformer", counters):
            rt._source_suffix_forward(pipeline, capture, counters=counters, diagnostics=diagnostics,
                                      partial=partial, delta=delta, after_step=after_step)
        original = capture.before_state["latents"]
        after = partial["before_latents"]
        result = dict(image=partial["image"], actual_delta=after.double() - original.double(),
                      after=after, diagnostics=diagnostics)
        if capture_late:
            if not late:
                raise ValueError("early suffix omitted step18 capture")
            result["late"] = rt.SourceCapture(result["image"], late["before_state"],
                                             late["scheduler_state"], (19,))
        return result
    except Exception as error:
        partial["diagnostics"] = diagnostics
        raise rt.RuntimeStageError("stage_suffix", error, partial, counters) from error


def at_latent(capture, latent, image):
    state = deepcopy(capture.before_state)
    state["latents"] = latent.detach().cpu().clone()
    return rt.SourceCapture(image, state, deepcopy(capture.scheduler_state), capture.remaining_steps)


class StageProtectionProvider(PrecompensationProvider):
    def __init__(self, config, *, counters, **kwargs):
        super().__init__(config, **kwargs)
        self._raw = counters

    def _prepare_reader(self):
        if self._vae is None:
            self._vae, self._processor = self.reader_loader(self.config)
        super()._prepare_reader()

    def _prepare_suffix(self):
        self._release_reconstruction()
        self._load_source()
        self._source.remove_all_hooks()
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self._source, name, None)
            if module is not None:
                module.to("cpu")
        for module in (self._source.transformer, self._source.vae):
            module.to(self.config["generation"]["device"])
            _freeze(module)

    def generate(self, source):
        self._release_reconstruction()
        self._load_source()
        self._inference_policy(self._source, "generation")
        return capture_two(self._source, self._generation_settings(source), self._raw)

    def render(self, capture, delta=None, *, capture_late=False):
        self._prepare_suffix()
        return suffix(self._source, capture, delta, self._raw, capture_late=capture_late)

    def read(self, image):
        self._prepare_reader()
        rt._count(self._raw, "reader_result_ordinary_calls")
        feature = self.public_reader(image, vae=self._vae, image_processor=self._processor)
        rt._count(self._raw, "reader_result_ordinary_returned")
        self._check_reader(feature)
        vector, _ = make_reader(feature.numel(), self.spec)
        score = float(torch.dot(feature.double(), vector))
        if not np.isfinite(score):
            raise ValueError("nonfinite ordinary reader score")
        return dict(feature=feature, score=score)

    def reconstruct(self, image, seed):
        self._release_source()
        if self._reconstruction is None:
            self._load_reconstruction(gradients=True)
        pipe, settings = self._reconstruction, self._reconstruction_settings
        rt._count(self._raw, "sdxl_ordinary_forward_calls")
        with torch.no_grad(), rt._count_method(pipe.unet, "forward", "sdxl_unet_forward", self._raw), \
                rt._count_method(pipe.vae, "encode", "sdxl_input_vae_encode", self._raw), \
                rt._count_method(pipe.vae, "decode", "sdxl_output_vae_decode", self._raw):
            result = pipe(prompt=settings.prompt, image=image, strength=settings.strength,
                guidance_scale=settings.guidance_scale, num_inference_steps=settings.num_inference_steps,
                generator=torch.Generator(device=settings.device).manual_seed(seed), output_type="pil")
        rt._count(self._raw, "sdxl_ordinary_forward_returned")
        if len(result.images) != 1:
            raise ValueError("one reconstruction RGB required")
        return rt.require_ordinary_rgb_image(result.images[0])

    def rgb_gradient(self, image, ordinary_read, *, design_image=None, design_read=None):
        self._release_source()
        self._prepare_reader()
        self._check_reader(ordinary_read["feature"])
        if design_image is None:
            result = rt.rgb_reader_vjp(image, vae=self._vae, image_processor=self._processor,
                                       reader_vector=self._reader_vector, counters=self._raw)
            expected, expected_read = image, ordinary_read
        else:
            if self._reconstruction is None:
                self._load_reconstruction(gradients=True)
            result = rt.reconstruction_reader_vjp(self._reconstruction, image,
                self._reconstruction_settings, self.config["reconstruction"]["design_seed"],
                observation_vae=self._vae, observation_processor=self._processor,
                reader_vector=self._reader_vector, counters=self._raw)
            expected, expected_read = design_image, design_read
        if not np.array_equal(np.asarray(result.image), np.asarray(expected)):
            raise rt.RuntimeStageError("current_RGB_proxy_agreement",
                ValueError("proxy reconstruction differs from current ordinary RGB"),
                {"sensitivity": result}, self._raw)
        if not torch.equal(result.feature, expected_read["feature"]):
            raise rt.RuntimeStageError("current_reader_agreement",
                ValueError("ordinary reread differs at the same RGB basepoint"),
                {"sensitivity": result}, self._raw)
        return result

    def latent_gradients(self, capture, image, sensitivities):
        self._prepare_suffix()
        diagnostics = {}
        with rt._count_method(self._source.transformer, "forward", "sd35_transformer", self._raw):
            gradients = rt.source_suffix_vjps(self._source, capture,
                torch.stack([s.gradient_rgb for s in sensitivities]), counters=self._raw,
                diagnostics=diagnostics, basepoint_mode="runtime_suffix", reference_image=image)
        return dict(gradients=gradients, diagnostics=diagnostics)


def physical_counts(raw):
    """Attempted and returned stay separate, including nested components."""
    result = {}
    for suffix_name, raw_suffix in (("attempted", "calls"), ("returned", "returned")):
        result[f"sd3_forward_{suffix_name}"] = sum(raw.get(f"{p}_{raw_suffix}", 0)
            for p in ("sd35_full_generation", "sd35_suffix_forward"))
        ordinary = raw.get(f"sdxl_ordinary_forward_{raw_suffix}", 0)
        proxy = raw.get(f"sdxl_reconstruction_forward_{raw_suffix}", 0)
        result.update({f"sdxl_ordinary_{suffix_name}": ordinary, f"sdxl_proxy_{suffix_name}": proxy,
                       f"sdxl_forward_{suffix_name}": ordinary + proxy})
        for dest, prefix in (("rgb_vjp", "reader_vjp"), ("latent_vjp", "sd35_suffix_vjp"),
                             ("reader_result_ordinary", "reader_result_ordinary"),
                             ("reader_aux_ordinary", "reader_ordinary_forward"),
                             ("reader_proxy", "reader_proxy_forward")):
            result[f"{dest}_{suffix_name}"] = raw.get(f"{prefix}_{raw_suffix}", 0)
        result[f"reader_encode_{suffix_name}"] = sum(result[f"{p}_{suffix_name}"] for p in
            ("reader_result_ordinary", "reader_aux_ordinary", "reader_proxy"))
        for dest, prefix in (("sd3_denoising_steps", "sd35_transformer"),
                             ("sdxl_denoising_steps", "sdxl_unet_forward")):
            result[f"{dest}_{suffix_name}"] = raw.get(f"{prefix}_{suffix_name}", 0)
    return result
