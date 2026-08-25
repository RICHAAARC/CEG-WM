"""One parametrized two-pass SD3.5 path for Content V8 fit and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image

from cegwm.method.content_adaptive_v2 import (
    ContentAdaptiveMeasurement,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.method.content_iss_v8 import (
    ISSAsset,
    ISSDevelopmentMeasurement,
    content_v8_h,
    derive_wrong_keys,
    embed_content_v8,
    iss_beta,
)
from cegwm.protocol.content_chain_v8 import ContentV8Unit
from cegwm.runtime.content_adaptive_sd35_v2 import (
    ContentEmbedAssets,
    _decode_callback_latents,
    _probe_observation,
    _validate_pipeline,
)
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image


@dataclass(frozen=True, slots=True)
class ContentV8RunOutput:
    image: Image.Image
    primary_null: Image.Image
    measurement: ContentAdaptiveMeasurement


class ContentV8InjectionCallback:
    """Run the frozen V2 analysis and apply the V8 beta at step 18 once."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        assets: ContentEmbedAssets,
        beta: float,
    ) -> None:
        if not isinstance(assets, ContentEmbedAssets):
            raise TypeError("Content V8 callback requires ContentEmbedAssets")
        self._detection_key = detection_key
        self._assets = assets
        self._beta = beta
        self._measurement: ContentAdaptiveMeasurement | None = None

    @property
    def measurement(self) -> ContentAdaptiveMeasurement | None:
        return self._measurement

    def __call__(
        self,
        pipeline: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del timestep
        if not isinstance(step_index, int):
            raise TypeError("callback step_index must be an integer")
        if not isinstance(callback_kwargs, dict):
            raise TypeError("callback state must be a dict")
        if step_index != 18:
            return callback_kwargs
        if self._measurement is not None:
            raise RuntimeError("Content V8 callback attempted step 18 more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("callback state must contain torch latents")
        if pipeline is None:
            raise RuntimeError("Content V8 analysis requires the real callback pipeline")
        base_image = _decode_callback_latents(pipeline, latents)
        semantic = dino_last_layer_cls_patch_tiles(
            base_image, self._assets.dino_processor, self._assets.dino_model
        )
        texture = rgb_texture_tiles(base_image)
        baseline = _probe_observation(base_image, self._assets)

        def evaluate(branch: str, candidate: torch.Tensor) -> Any:
            if branch not in {"lf", "hf"}:
                raise ValueError("Content V8 probe branch differs")
            return _probe_observation(
                _decode_callback_latents(pipeline, candidate), self._assets
            )

        probes = evaluate_public_probes(latents, baseline, evaluate)
        allocation = allocate_content(ContentSignals(
            semantic,
            texture,
            probes.lf_two_scale_response_consistency,
            probes.hf_two_scale_response_consistency,
            probes.lf_local_perturbation_sensitivity,
            probes.hf_local_perturbation_sensitivity,
        ))
        embedded, measurement = embed_content_v8(
            latents,
            self._detection_key,
            self._assets.hf_public_assets,
            self._assets.lf_public_assets,
            allocation,
            self._beta,
        )
        updated = dict(callback_kwargs)
        updated["latents"] = embedded
        self._measurement = measurement
        return updated


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _run_write(
    pipeline: Any,
    unit: ContentV8Unit,
    detection_key: bytes,
    assets: ContentEmbedAssets,
    beta: float,
) -> tuple[Image.Image, ContentAdaptiveMeasurement]:
    _validate_pipeline(pipeline)
    callback = ContentV8InjectionCallback(detection_key, assets, beta)
    result = pipeline(
        prompt=unit.prompt,
        num_inference_steps=20,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("SD3.5 pipeline must return exactly one final image")
    if callback.measurement is None:
        raise RuntimeError("pipeline completed without Content V8 step-18 write")
    return require_ordinary_rgb_image(images[0]), callback.measurement


def _run_two_pass(
    pipeline: Any,
    unit: ContentV8Unit,
    detection_key: bytes,
    assets: ContentEmbedAssets,
    beta_controller: Callable[[float], float],
) -> tuple[ContentV8RunOutput, float]:
    """Single control path: callback-free pass 1, then same-seed write pass 2."""

    if not isinstance(unit, ContentV8Unit):
        raise TypeError("Content V8 two-pass runtime requires a validated unit")
    if not isinstance(assets, ContentEmbedAssets):
        raise TypeError("Content V8 two-pass runtime requires frozen embed assets")
    if not callable(beta_controller):
        raise TypeError("Content V8 beta controller must be callable")
    primary_null = require_ordinary_rgb_image(run_sd35_plain(
        pipeline,
        unit.prompt,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    ))
    host_score = content_v8_h(
        primary_null, detection_key, assets.lf_public_assets
    )
    beta = beta_controller(host_score)
    image, measurement = _run_write(
        pipeline, unit, detection_key, assets, beta
    )
    return ContentV8RunOutput(image, primary_null, measurement), host_score


def run_content_v8_development_pair(
    pipeline: Any,
    unit: ContentV8Unit,
    development_key: bytes,
    assets: ContentEmbedAssets,
) -> ISSDevelopmentMeasurement:
    """Measure the real beta=1 V2 spatial writer on one frozen dev identity."""

    if unit.split != "content_v6_iss_development_v1":
        raise TypeError("Content V8 development requires the frozen dev32 split")
    output, host_score = _run_two_pass(
        pipeline, unit, development_key, assets, lambda _: 1.0
    )
    beta_one_score = content_v8_h(
        output.image, development_key, assets.lf_public_assets
    )
    wrong_scores = tuple(
        content_v8_h(output.image, wrong, assets.lf_public_assets)
        for wrong in derive_wrong_keys(development_key)
    )
    if len(wrong_scores) != 16:
        raise RuntimeError("Content V8 development requires exactly 16 wrong keys")
    return ISSDevelopmentMeasurement(
        host_score,
        beta_one_score,
        max(host_score, *wrong_scores),
    )


def run_content_v8_evaluation_pair(
    pipeline: Any,
    unit: ContentV8Unit,
    detection_key: bytes,
    assets: ContentEmbedAssets,
    iss_asset: ISSAsset,
) -> ContentV8RunOutput:
    """Run the same control with beta=clamp((m-h)/g,1,2)."""

    if unit.split not in {
        "content_adaptive_dual_branch_v2_clean_v1",
        "content_v6_iss_clean_v1",
    }:
        raise TypeError("Content V8 evaluation unit is outside both frozen rosters")
    output, _ = _run_two_pass(
        pipeline,
        unit,
        detection_key,
        assets,
        lambda host: iss_beta(host, iss_asset),
    )
    return output


__all__ = [
    "ContentV8InjectionCallback", "ContentV8RunOutput",
    "run_content_v8_development_pair", "run_content_v8_evaluation_pair",
]
