"""Thin SD3.5 diffusers adapter for the real Stage-A HF callback path."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.hf import FrozenHFPublicAssets, inject_hf_carrier
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.numerics import BudgetMeasurement


@dataclass(frozen=True, slots=True)
class HFRunOutput:
    """Runtime output only; it contains no detection decision or positive claim."""

    image: Image.Image
    injection_budget: BudgetMeasurement


class HFInjectionCallback:
    """Inject exactly once at the frozen zero-based denoising step."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        frozen_public_assets: FrozenHFPublicAssets,
    ) -> None:
        self._detection_key = detection_key
        self._assets = frozen_public_assets
        self._measurement: BudgetMeasurement | None = None

    @property
    def measurement(self) -> BudgetMeasurement | None:
        return self._measurement

    def __call__(
        self,
        pipeline: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del pipeline, timestep
        if not isinstance(step_index, int):
            raise TypeError("diffusers callback step_index must be an integer")
        if not isinstance(callback_kwargs, dict):
            raise TypeError("diffusers callback state must be a dict")
        if step_index != self._assets.injection_step_index:
            return callback_kwargs
        if self._measurement is not None:
            raise RuntimeError("HF callback attempted the frozen injection step more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("diffusers callback state must contain torch latents")
        injected, measurement = inject_hf_carrier(latents, self._detection_key, self._assets)
        updated = dict(callback_kwargs)
        updated["latents"] = injected
        self._measurement = measurement
        return updated


def _validate_pipeline_callback_api(pipeline: Any) -> None:
    call = getattr(pipeline, "__call__", None)
    if not callable(call):
        raise TypeError("SD3.5 pipeline must be callable")
    parameters = inspect.signature(call).parameters
    has_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    required = {"callback_on_step_end", "callback_on_step_end_tensor_inputs"}
    if not has_kwargs and not required.issubset(parameters):
        raise TypeError("pipeline lacks the required diffusers callback-on-step-end API")


def load_sd35_pipeline(
    model_id: str,
    *,
    torch_dtype: torch.dtype,
    token: str,
) -> Any:
    """Load the protocol-named SD3.5 model using the Hub default resolution."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("SD3.5 model_id must be non-empty text")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Hugging Face token must be non-empty text")

    try:
        diffusers = importlib.import_module("diffusers")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("diffusers is required for the real SD3.5 runtime") from error
    pipeline_class = getattr(diffusers, "StableDiffusion3Pipeline", None)
    if pipeline_class is None or not callable(getattr(pipeline_class, "from_pretrained", None)):
        raise RuntimeError("installed diffusers lacks StableDiffusion3Pipeline")
    pipeline = pipeline_class.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        token=token,
    )
    _validate_pipeline_callback_api(pipeline)
    return pipeline


def run_sd35_hf(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    frozen_public_assets: FrozenHFPublicAssets,
    *,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
) -> HFRunOutput:
    """Execute the fixed callback path and return the ordinary final RGB image."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("embedding prompt must be non-empty text")
    if not isinstance(height, int) or not isinstance(width, int) or height < 256 or width < 256:
        raise ValueError("Stage-A image dimensions must be integer values of at least 256")
    _validate_pipeline_callback_api(pipeline)
    callback = HFInjectionCallback(detection_key, frozen_public_assets)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=list(HFInjectionCallback.tensor_inputs),
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("SD3.5 pipeline must return exactly one final image")
    image = require_ordinary_rgb_image(images[0])
    if callback.measurement is None:
        raise RuntimeError("SD3.5 pipeline completed without the frozen HF injection")
    return HFRunOutput(image=image, injection_budget=callback.measurement)


def run_sd35_plain(
    pipeline: Any,
    prompt: str,
    *,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Generate the paired primary-null image without installing a callback."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generation prompt must be non-empty text")
    if not isinstance(height, int) or not isinstance(width, int) or height < 256 or width < 256:
        raise ValueError("Stage-A image dimensions must be integer values of at least 256")
    if not callable(pipeline):
        raise TypeError("SD3.5 pipeline must be callable")
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("SD3.5 pipeline must return exactly one primary-null image")
    return require_ordinary_rgb_image(images[0])
