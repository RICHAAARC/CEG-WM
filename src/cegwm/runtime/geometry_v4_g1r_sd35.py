"""Versioned SD3.5 step-19 callback for the V4-G1R anchor writer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.geometry_v4_g1r import write_g1r_final_latent
from cegwm.protocol.geometry_v4_g1r import CALLBACK_STEP_INDEX
from cegwm.runtime.diffusers_sd35 import _validate_pipeline_callback_api
from cegwm.runtime.observation import require_ordinary_rgb_image


class G1RFinalLatentCallback:
    tensor_inputs = ("latents",)

    def __init__(self, detection_key: object) -> None:
        self._key = detection_key
        self.called = False

    def __call__(self, pipeline: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        del timestep
        if not isinstance(step_index, int) or not isinstance(callback_kwargs, dict):
            raise TypeError("invalid V4-G1R callback state")
        if step_index != CALLBACK_STEP_INDEX:
            return callback_kwargs
        if self.called:
            raise RuntimeError("V4-G1R callback attempted step 19 more than once")
        if pipeline is None or not callable(getattr(getattr(pipeline, "vae", None), "decode", None)):
            raise RuntimeError("V4-G1R callback requires the real SD3 VAE pipeline")
        self.called = True
        updated = dict(callback_kwargs)
        updated["latents"] = write_g1r_final_latent(updated.get("latents"), self._key, pipeline)
        return updated


@dataclass(frozen=True, slots=True)
class G1RGeneratedPair:
    clean: Image.Image
    marked: Image.Image


def run_g1r_sd35_pair(pipeline: Any, prompt: str, detection_key: object, *, height: int, width: int, generator: torch.Generator) -> G1RGeneratedPair:
    """Real-model entrypoint; this module does not authorize its execution."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("V4-G1R prompt must be non-empty")
    if not isinstance(height, int) or not isinstance(width, int) or height < 256 or width < 256:
        raise ValueError("invalid V4-G1R SD3.5 dimensions")
    _validate_pipeline_callback_api(pipeline)
    seed, device = generator.initial_seed(), getattr(generator, "device", "cpu")
    clean_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil")
    callback = G1RFinalLatentCallback(detection_key)
    marked_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil", callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    clean, marked = getattr(clean_result, "images", None), getattr(marked_result, "images", None)
    if not callback.called or not isinstance(clean, (list, tuple)) or not isinstance(marked, (list, tuple)) or len(clean) != 1 or len(marked) != 1:
        raise RuntimeError("V4-G1R SD3.5 pair did not materialize exactly one RGB per arm")
    return G1RGeneratedPair(require_ordinary_rgb_image(clean[0]), require_ordinary_rgb_image(marked[0]))
