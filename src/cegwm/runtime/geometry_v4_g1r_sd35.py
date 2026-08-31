"""Versioned one-shot VAE decoder-output hook for the V4-G1R v3 opponent writer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.geometry_v4_g1r import write_g1r_decoder_output
from cegwm.protocol.geometry_v4_g1r import DECODER_HOOK_CALLS_REQUIRED
from cegwm.runtime.observation import require_ordinary_rgb_image


class G1RDecoderOutputHook:
    def __init__(self, detection_key: object) -> None:
        self._key = detection_key
        self.calls = 0

    def __call__(self, module: Any, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
        del module, inputs
        self.calls += 1
        if self.calls > DECODER_HOOK_CALLS_REQUIRED:
            raise RuntimeError("V4-G1R decoder hook was invoked more than once")
        return write_g1r_decoder_output(output, self._key)


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
    seed, device = generator.initial_seed(), getattr(generator, "device", "cpu")
    clean_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil")
    decoder = getattr(getattr(pipeline, "vae", None), "decoder", None)
    if not callable(getattr(decoder, "register_forward_hook", None)):
        raise RuntimeError("V4-G1R requires the real AutoencoderKL decoder module")
    hook = G1RDecoderOutputHook(detection_key)
    handle = decoder.register_forward_hook(hook)
    try:
        marked_result = pipeline(prompt=prompt, num_inference_steps=20, height=height, width=width, generator=torch.Generator(device=device).manual_seed(seed), output_type="pil")
    finally:
        handle.remove()
    clean, marked = getattr(clean_result, "images", None), getattr(marked_result, "images", None)
    if hook.calls != DECODER_HOOK_CALLS_REQUIRED or not isinstance(clean, (list, tuple)) or not isinstance(marked, (list, tuple)) or len(clean) != 1 or len(marked) != 1:
        raise RuntimeError("V4-G1R SD3.5 pair did not materialize exactly one RGB per arm")
    return G1RGeneratedPair(require_ordinary_rgb_image(clean[0]), require_ordinary_rgb_image(marked[0]))
