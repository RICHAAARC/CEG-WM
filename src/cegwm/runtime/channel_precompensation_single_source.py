"""One fixed source: linearize and write around the same runtime suffix image.

Historical zero images remain diagnostics. Old RGB cotangents are deliberately
not reused at this different basepoint. No full-prefix regeneration is needed.
"""
from __future__ import annotations

import numpy as np
import torch

from cegwm.runtime.channel_precompensation_provider import PrecompensationProvider, _freeze
from cegwm.runtime.channel_precompensation_sd35 import SourceCapture


class RuntimeSuffixSingleSourceProvider(PrecompensationProvider):
    def _prepare_suffix(self):
        self._load_source()
        self._source.remove_all_hooks()
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self._source, name, None)
            if module is not None:
                module.to("cpu")
        for module in (self._source.transformer, self._source.vae):
            module.to(self.config["generation"]["device"])
            _freeze(module)

    def prepare_sources(self, sources):
        if self._state != "new" or self._closed or self._resume is None:
            raise RuntimeError("single-source method requires a fresh provider and saved capture")
        if list(sources) != self.config["sources"][:1]:
            raise ValueError("single-source scope is the first predeclared source only")
        self._state = "preparing"
        identifier = sources[0]["source_id"]
        saved = self._resume.baselines[identifier]
        capture = SourceCapture(saved["image"], saved["before_state"], saved["scheduler_state"],
                                tuple(saved["remaining_steps"]))
        self._captures[identifier] = capture
        self._runtime_zero = {}
        stage = "runtime_zero"
        partial = {}
        try:
            self._prepare_suffix()
            diagnostics = {}
            replay = self.runtime.replay_suffix_write(self._source, capture, counters=self._raw,
                                                      diagnostics=diagnostics)
            self._runtime_zero[identifier] = replay.image
            partial = {"image": replay.image, "diagnostics": diagnostics}
            feature = self._public_feature(replay.image)
            yield {"kind": "baseline", "source_id": identifier, "status": "OBSERVED",
                   "image": replay.image, "feature": feature, "before_state": capture.before_state,
                   "scheduler_state": capture.scheduler_state, "remaining_steps": capture.remaining_steps,
                   "capture_reused": True, "reused": False,
                   "diagnostics": {"basepoint": "runtime_suffix", "historical_zero": diagnostics,
                       "old_rgb_gradients_used": False,
                       "reason": "Both RGB sensitivities are recomputed at this runtime zero image."}}
            self._release_source()
            stage = "clean_rgb_vjp"
            clean = self.runtime.rgb_reader_vjp(replay.image, vae=self._vae,
                image_processor=self._processor, reader_vector=self._reader_vector, counters=self._raw)
            yield self._gradient_event(identifier, "clean", clean, reusable=False)
            stage = "design_zero"
            self._load_reconstruction(gradients=True)
            seed = self.config["reconstruction"]["design_seed"]
            actual = self._ordinary_reconstruction(replay.image, seed)
            partial = {"image": actual, "source_zero_image": replay.image, "design_seed": seed}
            actual_feature = self._public_feature(actual)
            yield {"kind": "design", "source_id": identifier, "status": "OBSERVED",
                   "image": actual, "feature": actual_feature}
            stage = "design_rgb_vjp"
            design = self.runtime.reconstruction_reader_vjp(self._reconstruction,
                replay.image, self._reconstruction_settings, seed, observation_vae=self._vae,
                observation_processor=self._processor, reader_vector=self._reader_vector, counters=self._raw)
            yield self._gradient_event(identifier, "design", design, reusable=False)
            if not np.array_equal(np.asarray(actual), np.asarray(design.image)):
                raise ValueError("current-basepoint design proxy differs from its ordinary reconstruction")
            self._release_reconstruction()
            stage = "source_suffix_vjp"
            self._prepare_suffix()
            sensitivity = {}
            gradients = self.runtime.source_suffix_vjps(self._source, capture,
                torch.stack([clean.gradient_rgb, design.gradient_rgb]), counters=self._raw,
                diagnostics=sensitivity, basepoint_mode="runtime_suffix", reference_image=replay.image)
            yield {"kind": "linearization", "source_id": identifier, "gradients": gradients,
                   "diagnostics": {"kind": "two_proxy_sensitivities_not_discrete_jacobian",
                       "basepoint": "runtime_suffix", "historical_zero_is_diagnostic_only": True,
                       "design_seed": seed, "latent_sensitivity": sensitivity,
                       "clean": clean.diagnostics, "design": design.diagnostics}}
        except Exception as error:
            yield {**self._failure(identifier, stage, error),
                   "partial": {**partial, **getattr(error, "partial", {})}}
        finally:
            self._release_reconstruction()
            self._state = "prepared"

    def generate_writes(self, requests):
        if self._state != "prepared" or self._closed:
            raise RuntimeError("writes require completed single-source preparation")
        self._state = "generating"
        try:
            for request in requests:
                identifier = request["source_id"]
                identity = {key: request[key] for key in ("source_id", "method", "bit")}
                partial = {}
                matched_state = False
                try:
                    if self._source is None:
                        raise RuntimeError("prepared suffix pipeline is unavailable")
                    replay = self.runtime.replay_suffix_write(self._source, self._captures[identifier],
                                                              request["delta"], counters=self._raw)
                    matched_state = True
                    partial = {"image": replay.image, "before_state": replay.before_state,
                               "actual_delta": replay.actual_delta, "after_latents": replay.after_latents,
                               "injection_metrics": replay.injection_metrics, "remaining_steps": replay.remaining_steps}
                    yield {**identity, **partial, "status": "OBSERVED", "matched_state": True,
                           "feature": self._public_feature(replay.image)}
                except Exception as error:
                    partial = {**partial, **getattr(error, "partial", {})}
                    yield {**identity, **partial, "status": "FAILED", "matched_state": matched_state,
                           "error": f"{type(error).__name__}: {error}"}
        finally:
            self._release_source()
            self._state = "generated"

    def reconstruct_outputs(self, requests):
        if self._state != "generated" or self._closed:
            raise RuntimeError("evaluation requires completed writes")
        self._state = "reconstructing"
        try:
            for request in requests:
                identity = {key: request[key] for key in ("source_id", "method", "bit", "seed")}
                image = None
                try:
                    if self._reconstruction is None:
                        # Match the resident, frozen design-zero execution
                        # policy. _ordinary_reconstruction still uses no_grad;
                        # this flag controls ownership/hooks, not a backward.
                        self._load_reconstruction(gradients=True)
                    image = self._ordinary_reconstruction(request["image"], request["seed"])
                    yield {**identity, "status": "OBSERVED", "image": image,
                           "feature": self._public_feature(image), "reused": False}
                except Exception as error:
                    yield {**identity, "status": "FAILED", "image": image,
                           "error": f"{type(error).__name__}: {error}"}
        finally:
            self._release_reconstruction()
            self._state = "reconstructed"
