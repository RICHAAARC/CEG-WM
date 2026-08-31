"""SD3.5 adapter for the Geometry-V6 R0 final-latent carrier."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

import torch
from PIL import Image

from cegwm.method.geometry_v6_roundtrip import apply_roundtrip_adjoint_update
from cegwm.runtime.content_adaptive_sd35 import ContentAdaptiveInjectionCallback, ContentEmbedAssets
from cegwm.runtime.observation import require_ordinary_rgb_image

R0Arm = Literal["content_only", "content_geometry", "geometry_only", "unwatermarked"]


@dataclass(frozen=True, slots=True)
class GeometryV6RunOutput:
    image: Image.Image
    content_measurement: Any | None
    arm: R0Arm


class _V6Callback:
    tensor_inputs = ("latents",)

    def __init__(
        self,
        arm: R0Arm,
        content_key: str | bytes | bytearray | memoryview | None,
        amplitude: float | None,
        content_assets: ContentEmbedAssets | None,
    ) -> None:
        self._arm = arm
        self._amplitude = amplitude
        self._content = (
            ContentAdaptiveInjectionCallback(content_key, content_assets)
            if arm in {"content_only", "content_geometry"}
            and content_key is not None and content_assets is not None
            else None
        )
        self._geometry_written = False

    @property
    def content_measurement(self) -> Any | None:
        return None if self._content is None else self._content.measurement

    def __call__(self, pipeline: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(step_index, int) or not isinstance(callback_kwargs, dict):
            raise TypeError("diffusers callback must provide integer step and dict state")
        updated = callback_kwargs
        if self._content is not None:
            # This is the existing step-18 writer unchanged; its own callback
            # returns the input untouched on every other scheduler step.
            updated = self._content(pipeline, step_index, timestep, updated)
        if step_index != 19:
            return updated
        if self._amplitude is None:
            return updated
        if self._geometry_written:
            raise RuntimeError("Geometry-V6 attempted final-latent write more than once")
        latents = updated.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("step-19 callback state must contain latents")
        vae = getattr(pipeline, "vae", None)
        updated = dict(updated)
        updated["latents"] = apply_roundtrip_adjoint_update(latents, self._amplitude, vae)
        self._geometry_written = True
        return updated

    def require_complete(self) -> None:
        if self._content is not None and self._content.measurement is None:
            raise RuntimeError("pipeline completed without the frozen step-18 content write")
        if self._amplitude is not None and not self._geometry_written:
            raise RuntimeError("pipeline completed without the required post-scheduler step-19 geometry write")


def _validate_pipeline(pipeline: Any) -> None:
    call = getattr(pipeline, "__call__", None)
    if not callable(call):
        raise TypeError("SD3.5 pipeline must be callable")
    parameters = inspect.signature(call).parameters
    has_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    if not has_kwargs and not {"callback_on_step_end", "callback_on_step_end_tensor_inputs"}.issubset(parameters):
        raise TypeError("pipeline lacks diffusers callback-on-step-end API")


def run_sd35_geometry_v6_r0_arm(
    pipeline: Any,
    prompt: str,
    arm: R0Arm,
    *,
    content_key: str | bytes | bytearray | memoryview | None,
    amplitude: float | None,
    content_assets: ContentEmbedAssets | None,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
) -> GeometryV6RunOutput:
    """Run one R0 arm.  Callback-on-step-end occurs after scheduler update.

    The adapter requires the real Diffusers callback API and fixes the interface
    expectation in fake-pipeline tests: step 18 is content, then the final
    scheduler update invokes step 19, then the pipeline decodes final RGB.
    """

    if arm not in {"content_only", "content_geometry", "geometry_only", "unwatermarked"}:
        raise ValueError("Geometry-V6 R0 arm is invalid")
    if not isinstance(prompt, str) or not prompt.strip() or height < 256 or width < 256:
        raise ValueError("Geometry-V6 requires a nonempty prompt and dimensions of at least 256")
    needs_content = arm in {"content_only", "content_geometry"}
    needs_geometry = arm in {"content_geometry", "geometry_only"}
    if needs_content != (content_key is not None and content_assets is not None):
        raise ValueError("content arms require exactly the frozen content key and assets")
    if needs_geometry != (amplitude is not None):
        raise ValueError("geometry arms require exactly one global amplitude")
    _validate_pipeline(pipeline)
    callback = _V6Callback(arm, content_key, amplitude, content_assets)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("SD3.5 pipeline must return one final image")
    callback.require_complete()
    return GeometryV6RunOutput(require_ordinary_rgb_image(images[0]), callback.content_measurement, arm)


__all__ = ["GeometryV6RunOutput", "R0Arm", "run_sd35_geometry_v6_r0_arm"]
