"""Experimental image-only reader and actual main ISS callback composition."""
from __future__ import annotations

import torch
from cegwm.method.latent_landmark_sync import carrier, read_observation, pil_sampling_public_h
from cegwm.runtime.observation import require_ordinary_rgb_image, encode_final_rgb_image
from cegwm.runtime.content_iss_sd35 import ContentISSInjectionCallback


class LandmarkCallback:
    tensor_inputs = ("latents",)

    def __init__(self, content_callback, key, rms):
        self.content_callback = content_callback
        self.key, self.rms = key, rms
        self.events = []
        self.writes = 0
        self.timesteps = None

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        updated = self.content_callback(pipeline, step_index, timestep, callback_kwargs)
        self.events.append(dict(index=int(step_index), timestep=float(timestep)))
        if step_index == 18:
            if self.writes:
                raise RuntimeError("duplicate landmark write")
            self.timesteps = pipeline.scheduler.timesteps.detach().cpu().tolist()
            if len(self.timesteps) != 20:
                raise RuntimeError("V1 requires exactly 20 actual timesteps")
            z = updated["latents"]
            if tuple(z.shape) != (1,16,64,64):
                raise ValueError("V1 supports only 1x16x64x64 latents")
            if self.rms:
                delta = torch.as_tensor(carrier(self.key, self.rms), device=z.device, dtype=z.dtype)
                updated = dict(updated, latents=z + delta)
                self.actual_delta_rms = float((updated["latents"].float()-z.float()).square().mean().sqrt())
            else:
                self.actual_delta_rms = 0.
            self.writes += 1
        return updated

    def trace(self):
        if self.writes != 1 or [e["index"] for e in self.events] != list(range(20)):
            raise RuntimeError("actual callback sequence differs from frozen 20-step path")
        return dict(timesteps=self.timesteps, callbacks=self.events, write_index=18,
                    updates_after_write=1, actual_dtype_delta_rms=self.actual_delta_rms,
                    update_count_evidence="standard_diffusers_step_end_callback_sequence_not_scheduler_instrumentation",
                    content_callback=type(self.content_callback).__name__)


def generate_content_landmark(pipeline, prompt, key, iss_assets, beta, seed, device, rms):
    callback = LandmarkCallback(ContentISSInjectionCallback(key, iss_assets, beta), key, rms)
    result = pipeline(prompt=prompt, num_inference_steps=20, height=512, width=512,
                      generator=torch.Generator(device=device).manual_seed(seed), output_type="pil",
                      callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    trace = callback.trace()
    if callback.content_callback.measurement is None:
        raise RuntimeError("missing real content callback measurement")
    return require_ordinary_rgb_image(result.images[0]), trace


def read_image(image, key, image_processor, vae, peak_threshold=0., ambiguity_threshold=0.):
    image = require_ordinary_rgb_image(image)
    if image.size != (512,512):
        raise ValueError("V1 RGB input must be 512x512")
    observation = encode_final_rgb_image(image, image_processor, vae)
    result = read_observation(observation.detach().cpu(), key, peak_threshold, ambiguity_threshold)
    result["input_boundary"] = "final_RGB8_deterministic_VAE_mode"
    result["vae"] = dict(scaling_factor=float(vae.config.scaling_factor), shift_factor=float(vae.config.shift_factor))
    return result


def detect_candidate(image, key, public_assets, tau, peak_threshold, ambiguity_threshold):
    """Separate experimental API: geometry cannot create a positive result.

    public_assets is used by the unchanged main content scorer and public VAE;
    the candidate is never installed into its SyncSeal backend slot.
    """
    from cegwm.runtime.paper_detection_v2 import score_current_rgb
    from cegwm.geometry_v7.r1b import rectify_attacked_rgb
    pre = float(score_current_rgb(image, key, public_assets).value)
    if pre > tau:
        return dict(pre=pre, post=None, accepted=True, route="pre_early_return")
    public = public_assets.content_assets.iss_assets.embed_assets.hf_public_assets
    geometry = read_image(image,key,public.image_processor,public.vae,peak_threshold,ambiguity_threshold)["geometry"]
    if not geometry["accepted"]:
        return dict(pre=pre, post=None, accepted=False, route="geometry_rejected")
    corrected = rectify_attacked_rgb(image, pil_sampling_public_h(geometry["H_latent"]))
    post = float(score_current_rgb(corrected, key, public_assets).value)
    return dict(pre=pre, post=post, accepted=post > tau, route="single_H_post")
