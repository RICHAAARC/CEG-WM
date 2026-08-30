"""Real SD3.5 20-step, step-19 callback adapter for Geometry-V4 G0/G1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.geometry_v4_generative import write_final_latent_anchor
from cegwm.runtime.diffusers_sd35 import _validate_pipeline_callback_api
from cegwm.runtime.observation import require_ordinary_rgb_image


class FinalLatentAnchorCallback:
    tensor_inputs = ("latents",)
    def __init__(self, detection_key: object) -> None: self._key, self.called = detection_key, False
    def __call__(self, pipeline: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        del timestep
        if not isinstance(step_index, int) or not isinstance(callback_kwargs, dict): raise TypeError("invalid diffusers callback state")
        if step_index != 19: return callback_kwargs
        if self.called: raise RuntimeError("Geometry-V4 final callback attempted step 19 more than once")
        self.called = True
        updated = dict(callback_kwargs)
        if pipeline is None or not callable(getattr(getattr(pipeline, "vae", None), "decode", None)):
            raise RuntimeError("Geometry-V4 final callback requires the real SD3 VAE pipeline")
        updated["latents"] = write_final_latent_anchor(updated.get("latents"), self._key, pipeline)
        return updated


@dataclass(frozen=True, slots=True)
class GeneratedPair:
    clean: Image.Image
    marked: Image.Image


def run_sd35_final_latent_pair(pipeline: Any, prompt: str, detection_key: object, *, height: int, width: int, generator: torch.Generator) -> GeneratedPair:
    """Generate one clean pass and one same-seed sole-placement marked pass."""
    if not isinstance(prompt, str) or not prompt.strip(): raise ValueError("prompt must be non-empty")
    if not isinstance(height, int) or not isinstance(width, int) or height < 256 or width < 256: raise ValueError("invalid SD3.5 dimensions")
    _validate_pipeline_callback_api(pipeline)
    seed = generator.initial_seed()
    device = getattr(generator, "device", "cpu")
    clean_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil")
    callback = FinalLatentAnchorCallback(detection_key)
    marked_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil", callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    clean, marked = getattr(clean_result, "images", None), getattr(marked_result, "images", None)
    if not callback.called or not isinstance(clean, (list, tuple)) or not isinstance(marked, (list, tuple)) or len(clean) != 1 or len(marked) != 1: raise RuntimeError("SD3.5 pair failed to materialize exactly one final RGB per arm")
    return GeneratedPair(require_ordinary_rgb_image(clean[0]), require_ordinary_rgb_image(marked[0]))
