"""Staged model ownership for the one-shot precompensation experiment.

Construction is inert. Only the explicitly selected runner calls these methods.
SD3 and SDXL large components are never deliberately resident together. The
identical SD3 VAE remains the ordinary-RGB reader across every stage. Inference
uses the configured offload policy; backward phases remove inference hooks,
freeze weights and use the runtime's CPU-saved activations. Host RAM and real
GPU compatibility still require the user-run experiment.
"""

from __future__ import annotations

from copy import deepcopy
import gc
from typing import Any

import numpy as np
import torch

from cegwm.channel_precompensation import ReaderSpec, make_reader
from cegwm.runtime import channel_precompensation_sd35 as default_runtime
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.runtime.reconstruction_channel_sd35 import (
    GenerationSettings, ReconstructionSettings, final_image_feature,
)


def _load_pipeline(config, name):
    # Importing this module never imports/loads a model pipeline.
    from experiments.run_reconstruction_channel_v1 import _load_pipeline as load
    return load(config, name)


def _load_reader(config):
    """Load only the fixed SD3 VAE when resuming existing zero observations."""
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    settings = config["generation"]
    vae = AutoencoderKL.from_pretrained(settings["model_id"], subfolder="vae",
        revision=settings["revision"], torch_dtype=getattr(torch, settings["dtype"]))
    # Identical to StableDiffusion3Pipeline.__init__ in diffusers 0.39.0.
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    return vae, processor


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _freeze(module):
    if module is not None:
        module.eval()
        module.requires_grad_(False)


class PrecompensationProvider:
    """Generator protocol consumed by run_channel_precompensation_v1.

    ``loader`` and ``runtime`` are dependency injection for CPU lifecycle tests;
    neither is a choice made from model results. Every source and arm is tried
    at most once. All counters are attempts/returns, not favorable outcomes.
    """

    def __init__(self, config, *, loader=None, runtime=None, public_reader=None,
                 reader_loader=None, resume_from=None, resume_checkpoint=None):
        if resume_from is not None and resume_checkpoint is not None:
            raise ValueError("pass either resume_from or resume_checkpoint")
        if resume_from is not None:
            from cegwm.runtime.channel_precompensation_resume import load_resume_checkpoint
            resume_checkpoint = load_resume_checkpoint(resume_from, config=config)
        self.config = deepcopy(config)
        self.loader = loader or _load_pipeline
        self.reader_loader = reader_loader or _load_reader
        self.runtime = runtime or default_runtime
        self.public_reader = public_reader or final_image_feature
        self.spec = ReaderSpec(**config["reader"])
        self._raw: dict[str, int] = {}
        self._public_attempted = self._public_returned = 0
        self._evaluation_attempted = self._evaluation_returned = 0
        self._source = self._reconstruction = self._vae = self._processor = None
        self._reader_vector = None
        self._resume = resume_checkpoint
        if resume_checkpoint is not None:
            vector = resume_checkpoint.reader_vector
            expected, phase = make_reader(vector.numel(), self.spec)
            if not torch.equal(vector, expected) or resume_checkpoint.reader_phase != phase:
                raise ValueError("resume reader differs from the frozen configured reader")
            self._reader_vector = vector.detach().cpu().clone()
        self._captures: dict[str, Any] = {}
        self._rgb_gradients: dict[str, dict[str, Any]] = {}
        self._settings = {row["source_id"]: self._generation_settings(row) for row in config["sources"]}
        reconstruction = config["reconstruction"]
        self._reconstruction_settings = ReconstructionSettings(**{
            field: reconstruction[field] for field in
            ("strength", "guidance_scale", "num_inference_steps", "prompt", "device")
        })
        self._state = "new"
        self._closed = False

    @property
    def counts(self):
        raw = self._raw
        # Component counts (e.g. reader and SDXL participation in one backward)
        # overlap. Do not sum every counter as independent model invocations.
        return {
            "final_generation_attempted": raw.get("sd35_full_generation_calls", 0),
            "final_generation_returned": raw.get("sd35_full_generation_returned", 0),
            "reconstruction_attempted": raw.get("sdxl_reconstruction_forward_calls", 0) + self._evaluation_attempted,
            "reconstruction_returned": raw.get("sdxl_reconstruction_forward_returned", 0) + self._evaluation_returned,
            "final_reconstruction_attempted": self._evaluation_attempted,
            "final_reconstruction_returned": self._evaluation_returned,
            "auxiliary_reconstruction_forward_attempted": raw.get("sdxl_reconstruction_forward_calls", 0),
            "auxiliary_reconstruction_forward_returned": raw.get("sdxl_reconstruction_forward_returned", 0),
            "public_feature_reads_attempted": self._public_attempted,
            "public_feature_reads_returned": self._public_returned,
            "latent_vjps_attempted": raw.get("sd35_suffix_vjp_calls", 0),
            "latent_vjps_returned": raw.get("sd35_suffix_vjp_returned", 0),
            "rgb_vjps_attempted": raw.get("reader_vjp_calls", 0),
            "rgb_vjps_returned": raw.get("reader_vjp_returned", 0),
            "autograd_calls_attempted": raw.get("reader_vjp_calls", 0) + raw.get("sd35_suffix_vjp_calls", 0),
            "autograd_calls_returned": raw.get("reader_vjp_returned", 0) + raw.get("sd35_suffix_vjp_returned", 0),
            "auxiliary_ordinary_reader_attempted": raw.get("reader_ordinary_forward_calls", 0),
            "auxiliary_ordinary_reader_returned": raw.get("reader_ordinary_forward_returned", 0),
            "evaluation_reconstruction_attempted": self._evaluation_attempted,
            "evaluation_reconstruction_returned": self._evaluation_returned,
            **raw,
        }

    def _generation_settings(self, source):
        settings = self.config["generation"]
        return GenerationSettings(prompt=source["prompt"], seed=source["seed"],
            relative_l2=self.config["write"]["relative_l2_budget"],
            **{field: settings[field] for field in
               ("injection_step", "guidance_scale", "height", "width", "num_inference_steps", "device")})

    def _inference_policy(self, pipeline, name):
        pipeline.remove_all_hooks()
        settings = self.config[name]
        if settings.get("memory_mode", "direct") == "model_cpu_offload":
            pipeline.to("cpu")
            pipeline.enable_model_cpu_offload(device=settings["device"])
        else:
            pipeline.to(settings["device"])

    def _load_source(self):
        if self._reconstruction is not None:
            raise RuntimeError("SDXL must be released before loading SD3")
        if self._source is None:
            self._raw["source_pipeline_load_attempted"] = self._raw.get("source_pipeline_load_attempted", 0) + 1
            pipeline = self.loader(self.config, "generation")
            self._source = pipeline
            pipeline.remove_all_hooks()
            if self._vae is None:
                self._vae, self._processor = pipeline.vae, pipeline.image_processor
            else:
                # Keep the original reader object, not an independently loaded
                # replacement whose state/dtype could silently diverge.
                pipeline.vae = self._vae
            self._inference_policy(pipeline, "generation")
            self._raw["source_pipeline_load_returned"] = self._raw.get("source_pipeline_load_returned", 0) + 1
        return self._source

    def _release_source(self):
        pipeline, self._source = self._source, None
        if pipeline is not None:
            pipeline.remove_all_hooks()
            del pipeline
        _free_memory()

    def _prepare_reader(self):
        if self._vae is None:
            raise RuntimeError("ordinary reader has not been loaded")
        self._vae.to(self.config["generation"]["device"])
        _freeze(self._vae)

    def _public_feature(self, image):
        self._prepare_reader()
        self._public_attempted += 1
        feature = self.public_reader(image, vae=self._vae, image_processor=self._processor)
        if feature.device.type != "cpu" or feature.ndim != 1 or not bool(torch.isfinite(feature).all()):
            raise ValueError("public reader must return a finite flattened CPU feature")
        self._public_returned += 1
        self._check_reader(feature)
        return feature

    def _check_reader(self, feature):
        if self._reader_vector is None:
            self._reader_vector, _ = make_reader(feature.numel(), self.spec)
        elif self._reader_vector.numel() != feature.numel():
            raise ValueError("frozen public feature dimension changed")

    def _load_reconstruction(self, *, gradients):
        if self._source is not None:
            raise RuntimeError("SD3 must be released before loading SDXL")
        self._prepare_reader()
        self._raw["reconstruction_pipeline_load_attempted"] = self._raw.get("reconstruction_pipeline_load_attempted", 0) + 1
        self._reconstruction = self.loader(self.config, "reconstruction")
        pipeline = self._reconstruction
        # Explicit across both the design and evaluation paths: this candidate
        # does not enable SDXL's optional unrelated output watermark.
        pipeline.watermark = None
        if gradients:
            pipeline.remove_all_hooks()
            pipeline.to(self.config["reconstruction"]["device"])
            for module in getattr(pipeline, "components", {}).values():
                if isinstance(module, torch.nn.Module):
                    _freeze(module)
            _freeze(pipeline.unet)
            _freeze(pipeline.vae)
        self._raw["reconstruction_pipeline_load_returned"] = self._raw.get("reconstruction_pipeline_load_returned", 0) + 1
        return pipeline

    def _release_reconstruction(self):
        pipeline, self._reconstruction = self._reconstruction, None
        if pipeline is not None:
            pipeline.remove_all_hooks()
            del pipeline
        _free_memory()

    @staticmethod
    def _failure(identifier, stage, error):
        return {"kind": "failure", "source_id": identifier, "stage": stage,
                "status": "FAILED", "error": f"{type(error).__name__}: {error}",
                "partial": getattr(error, "partial", {})}

    @staticmethod
    def _gradient_event(identifier, phase, result, *, reusable, reused=False):
        return {"kind": "rgb_gradient", "source_id": identifier, "phase": phase,
                "status": "OBSERVED", "reused": reused, "reusable": reusable,
                "compatibility_version": default_runtime.RGB_PROXY_COMPATIBILITY_VERSION,
                "result": {"image": result.image, "feature": result.feature,
                           "score": result.score, "gradient_rgb": result.gradient_rgb,
                           "diagnostics": result.diagnostics}}

    def _cached_gradient(self, identifier, phase, actual_image):
        if self._resume is None:
            return None
        event = getattr(self._resume, "rgb_gradients", {}).get(identifier, {}).get(phase, {})
        if (event.get("compatibility_version") != default_runtime.RGB_PROXY_COMPATIBILITY_VERSION
                or event.get("reusable") is not True):
            return None
        payload = event.get("result", {})
        gradient = payload.get("gradient_rgb")
        feature = payload.get("feature")
        image = payload.get("image")
        if (not isinstance(gradient, torch.Tensor) or gradient.device.type != "cpu"
                or tuple(gradient.shape) != (1, 3, actual_image.height, actual_image.width)
                or not bool(torch.isfinite(gradient).all())
                or not isinstance(feature, torch.Tensor) or feature.device.type != "cpu"
                or feature.shape != self._reader_vector.shape or not bool(torch.isfinite(feature).all())
                or image is None or not np.array_equal(np.asarray(image), np.asarray(actual_image))):
            return None
        if phase == "design" and payload.get("diagnostics", {}).get("design_seed") != self.config["reconstruction"]["design_seed"]:
            return None
        return default_runtime.RGBSensitivity(**payload)

    def _resume_baseline_events(self, sources):
        for source in sources:
            identifier = source["source_id"]
            baseline = self._resume.baselines[identifier]
            self._check_reader(baseline["feature"])
            self._captures[identifier] = default_runtime.SourceCapture(
                baseline["image"], baseline["before_state"], baseline["scheduler_state"],
                tuple(baseline["remaining_steps"]))
            self._rgb_gradients[identifier] = {}
            yield {**baseline, "kind": "baseline", "source_id": identifier,
                   "status": "OBSERVED", "reused": True}
        for source in sources:
            identifier = source["source_id"]
            for name, kind in (("design_seed", "design"), ("holdout_seed", "holdout_zero")):
                seed = self.config["reconstruction"][name]
                observed = self._resume.zero_reconstructions[(identifier, seed)]
                self._check_reader(observed["feature"])
                yield {**observed, "kind": kind, "source_id": identifier, "seed": seed,
                       "status": "OBSERVED", "reused": True}

    def prepare_sources(self, sources):
        if self._state != "new" or self._closed:
            raise RuntimeError("source preparation is single-use")
        if list(sources) != self.config["sources"]:
            raise ValueError("preparation must use the fixed complete source roster")
        self._state = "preparing"
        if self._resume is not None:
            # No source generation, zero reconstruction or public read is run
            # merely to import an already observed zero sample.
            yield from self._resume_baseline_events(sources)
            self._raw["reader_load_attempted"] = self._raw.get("reader_load_attempted", 0) + 1
            self._vae, self._processor = self.reader_loader(self.config)
            self._raw["reader_load_returned"] = self._raw.get("reader_load_returned", 0) + 1
        else:
            self._load_source()
            try:
                for source in sources:
                    identifier = source["source_id"]
                    partial = {}
                    try:
                        capture = self.runtime.capture_source(self._source, self._settings[identifier], counters=self._raw)
                        self._captures[identifier] = capture
                        partial = {"image": capture.image, "before_state": capture.before_state,
                                   "scheduler_state": capture.scheduler_state, "remaining_steps": capture.remaining_steps}
                        feature = self._public_feature(capture.image)
                        self._rgb_gradients[identifier] = {}
                        yield {"kind": "baseline", "source_id": identifier, "status": "OBSERVED",
                               "image": capture.image, "feature": feature, "before_state": capture.before_state,
                               "scheduler_state": capture.scheduler_state, "remaining_steps": capture.remaining_steps}
                    except Exception as error:
                        if partial:
                            yield {"kind": "baseline", "source_id": identifier, **partial,
                                   "status": "FAILED", "error": f"{type(error).__name__}: {error}"}
                        yield self._failure(identifier, "baseline", error)
            finally:
                self._release_source()

        # All downstream gradient phases see only the frozen original reader;
        # no inference offload hooks are left on its encode/decode operations.
        self._prepare_reader()
        for identifier, record in self._rgb_gradients.items():
            try:
                image = self._captures[identifier].image
                result = self._cached_gradient(identifier, "clean", image)
                reused = result is not None
                if result is None:
                    result = self.runtime.rgb_reader_vjp(image, vae=self._vae, image_processor=self._processor,
                        reader_vector=self._reader_vector, counters=self._raw)
                record["clean"] = result
                yield self._gradient_event(identifier, "clean", result, reusable=True, reused=reused)
            except Exception as error:
                yield self._failure(identifier, "clean_rgb_vjp", error)

        if self._rgb_gradients:
            try:
                for identifier, record in self._rgb_gradients.items():
                    result = actual_image = actual_feature = None
                    try:
                        # The actual design baseline follows the same ordinary
                        # no-grad/PIL path as every marked evaluation. Its graph
                        # replay for the VJP is an ADDITIONAL counted forward.
                        if self._resume is not None:
                            actual = self._resume.zero_reconstructions[(identifier, self.config["reconstruction"]["design_seed"])]
                            actual_image, actual_feature = actual["image"], actual["feature"]
                        else:
                            if self._reconstruction is None:
                                self._load_reconstruction(gradients=True)
                            actual_image = self._ordinary_reconstruction(self._captures[identifier].image,
                                self.config["reconstruction"]["design_seed"])
                            actual_feature = self._public_feature(actual_image)
                            yield {"kind": "design", "source_id": identifier, "status": "OBSERVED",
                                   "image": actual_image, "feature": actual_feature}
                        result = self._cached_gradient(identifier, "design", actual_image)
                        reused = result is not None
                        if result is None:
                            if self._reconstruction is None:
                                self._load_reconstruction(gradients=True)
                            result = self.runtime.reconstruction_reader_vjp(self._reconstruction,
                                self._captures[identifier].image, self._reconstruction_settings,
                                self.config["reconstruction"]["design_seed"], observation_vae=self._vae,
                                observation_processor=self._processor, reader_vector=self._reader_vector,
                                counters=self._raw)
                        self._check_reader(result.feature)
                        if not np.array_equal(np.asarray(actual_image), np.asarray(result.image)):
                            raise ValueError("gradient SDXL forward does not exactly reproduce ordinary design RGB")
                        record["design"] = result
                        yield self._gradient_event(identifier, "design", result, reusable=True, reused=reused)
                    except Exception as error:
                        partial = {**getattr(error, "partial", {})}
                        if result is not None:
                            partial.update(proxy_image=result.image, proxy_feature=result.feature,
                                           proxy_diagnostics=result.diagnostics, gradient_rgb=result.gradient_rgb)
                            yield self._gradient_event(identifier, "design", result, reusable=False)
                        if actual_image is not None:
                            partial.update(ordinary_image=actual_image, ordinary_feature=actual_feature)
                            if actual_feature is None:
                                yield {"kind": "design", "source_id": identifier, "image": actual_image,
                                       "status": "FAILED", "error": f"{type(error).__name__}: {error}"}
                        yield {**self._failure(identifier, "design_rgb_vjp", error), "partial": partial}
            finally:
                self._release_reconstruction()

        ready = [identifier for identifier, record in self._rgb_gradients.items()
                 if "clean" in record and "design" in record]
        if ready:
            self._load_source()
            self._source.remove_all_hooks()
            # Cached conditioning allows the gradient suffix to keep all three
            # SD3 text encoders on CPU, rather than filling GPU with unused text
            # parameters. The official full replay uses offload again below.
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                module = getattr(self._source, name, None)
                if module is not None:
                    module.to("cpu")
            for module in (self._source.transformer, self._source.vae):
                module.to(self.config["generation"]["device"])
                _freeze(module)
            for identifier in ready:
                record = self._rgb_gradients[identifier]
                try:
                    sensitivity_diagnostics = {}
                    gradients = self.runtime.source_suffix_vjps(self._source, self._captures[identifier],
                        torch.stack([record["clean"].gradient_rgb, record["design"].gradient_rgb]), counters=self._raw,
                        diagnostics=sensitivity_diagnostics)
                    yield {"kind": "linearization", "source_id": identifier, "gradients": gradients,
                           "diagnostics": {"kind": "two_proxy_sensitivities_not_discrete_jacobian",
                               "design_seed": self.config["reconstruction"]["design_seed"],
                               "latent_sensitivity": sensitivity_diagnostics,
                               "design_rgb_proxy_equals_ordinary": True,
                               "clean": record["clean"].diagnostics, "design": record["design"].diagnostics}}
                except Exception as error:
                    yield self._failure(identifier, "source_suffix_vjp", error)
            self._inference_policy(self._source, "generation")
        self._state = "prepared"

    def generate_writes(self, requests):
        if self._state != "prepared" or self._closed:
            raise RuntimeError("writes require a completed single preparation")
        self._state = "generating"
        try:
            if requests:
                self._load_source()
            for request in requests:
                identifier = request["source_id"]
                identity = {key: request[key] for key in ("source_id", "method", "bit")}
                partial = {}
                matched_state = False
                try:
                    replay = self.runtime.replay_write(self._source, self._settings[identifier],
                        self._captures[identifier], request["delta"], counters=self._raw)
                    matched_state = True
                    partial = {"image": replay.image, "before_state": replay.before_state,
                               "actual_delta": replay.actual_delta, "after_latents": replay.after_latents,
                               "injection_metrics": replay.injection_metrics, "remaining_steps": replay.remaining_steps}
                    event = {**identity, **partial, "status": "OBSERVED",
                             "feature": self._public_feature(replay.image), "matched_state": True}
                except Exception as error:
                    partial = {**partial, **getattr(error, "partial", {})}
                    event = {**identity, **partial, "status": "FAILED", "matched_state": matched_state,
                             "error": f"{type(error).__name__}: {error}"}
                yield event
        finally:
            self._release_source()
            self._state = "generated"

    def _ordinary_reconstruction(self, image, seed):
        settings = self._reconstruction_settings
        if seed not in (self.config["reconstruction"]["design_seed"], self.config["reconstruction"]["holdout_seed"]):
            raise ValueError("unplanned evaluation seed")
        generator = torch.Generator(device=settings.device).manual_seed(seed)
        self._evaluation_attempted += 1
        with torch.no_grad():
            result = self._reconstruction(prompt=settings.prompt,
                image=require_ordinary_rgb_image(image), strength=settings.strength,
                guidance_scale=settings.guidance_scale, num_inference_steps=settings.num_inference_steps,
                generator=generator, output_type="pil")
        self._evaluation_returned += 1
        if len(result.images) != 1:
            raise ValueError("reconstruction must return exactly one image")
        return require_ordinary_rgb_image(result.images[0])

    def reconstruct_outputs(self, requests):
        if self._state != "generated" or self._closed:
            raise RuntimeError("evaluation requires completed writes")
        self._state = "reconstructing"
        try:
            for request in requests:
                identity = {key: request[key] for key in ("source_id", "method", "bit", "seed")}
                image = None
                try:
                    if self._resume is not None and request["method"] == "zero":
                        cached = self._resume.zero_reconstructions[(request["source_id"], request["seed"])]
                        yield {**identity, **cached, "status": "OBSERVED", "reused": True}
                        continue
                    if self._reconstruction is None:
                        self._load_reconstruction(gradients=False)
                    image = self._ordinary_reconstruction(request["image"], request["seed"])
                    yield {**identity, "status": "OBSERVED", "image": image,
                           "feature": self._public_feature(image)}
                except Exception as error:
                    yield {**identity, "status": "FAILED", "image": image,
                           "error": f"{type(error).__name__}: {error}"}
        finally:
            self._release_reconstruction()
            self._state = "reconstructed"

    def close(self):
        if not self._closed:
            self._release_source()
            self._release_reconstruction()
            if self._vae is not None:
                self._vae.to("cpu")
            self._closed = True
            _free_memory()
