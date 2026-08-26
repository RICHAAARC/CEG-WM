"""Real paired SD3.5 development path for the content ISS ISS fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.content_iss import (
    ISSAsset,
    ISSDevelopmentMeasurement,
    score_content_iss_image,
    derive_development_wrong_keys,
    embed_content_iss,
    iss_beta,
)
from cegwm.method.content_whitening import FrozenContentWhiteningLFPublicAssets
from cegwm.method.content_unweighted import (
    ContentAdaptiveMeasurement,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.protocol.content_iss import ContentISSUnit, ISS_DEVELOPMENT_SPLIT
from cegwm.runtime.content_adaptive_sd35 import (
    _decode_callback_latents,
    _probe_observation,
    _validate_pipeline,
)
from cegwm.runtime.content_unweighted_sd35 import ContentUnweightedEmbedAssets, run_sd35_content_unweighted
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image


@dataclass(frozen=True, slots=True)
class ContentISSDevelopmentAssets:
    embed_assets: ContentUnweightedEmbedAssets
    lf_public_assets: FrozenContentWhiteningLFPublicAssets

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentUnweightedEmbedAssets):
            raise TypeError("content ISS development requires content-unweighted embed assets")
        if not isinstance(self.lf_public_assets, FrozenContentWhiteningLFPublicAssets):
            raise TypeError("content ISS development requires frozen content-whitening LF assets")
        if self.lf_public_assets.carrier_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("content ISS embed and detector must share LF carrier assets")


@dataclass(frozen=True, slots=True)
class ContentISSEvaluationAssets:
    """content-whitening public and runtime assets plus the accepted frozen ISS controller asset."""

    embed_assets: ContentUnweightedEmbedAssets
    lf_public_assets: FrozenContentWhiteningLFPublicAssets
    iss_asset: ISSAsset

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentUnweightedEmbedAssets):
            raise TypeError("content ISS evaluation requires content-unweighted embed assets")
        if not isinstance(self.lf_public_assets, FrozenContentWhiteningLFPublicAssets):
            raise TypeError("content ISS evaluation requires frozen content-whitening LF assets")
        if not isinstance(self.iss_asset, ISSAsset):
            raise TypeError("content ISS evaluation requires the frozen ISS asset")
        if self.lf_public_assets.carrier_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("content ISS embed and detector must share LF carrier assets")

    @property
    def hf_public_assets(self) -> Any:
        return self.embed_assets.hf_public_assets


@dataclass(frozen=True, slots=True)
class ContentISSRunOutput:
    """Pass-2 image and aggregates plus the sole pass-1 primary null image."""

    image: Image.Image
    primary_null: Image.Image
    measurement: ContentAdaptiveMeasurement


class ContentISSInjectionCallback:
    """Run the content-whitening analysis, then scale only LF before joint projection."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        assets: ContentISSEvaluationAssets,
        beta: float,
    ) -> None:
        if not isinstance(assets, ContentISSEvaluationAssets):
            raise TypeError("content ISS callback requires evaluation assets")
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
            raise TypeError("diffusers callback step_index must be an integer")
        if not isinstance(callback_kwargs, dict):
            raise TypeError("diffusers callback state must be a dict")
        if step_index != 18:
            return callback_kwargs
        if self._measurement is not None:
            raise RuntimeError("content ISS callback attempted step 18 more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("diffusers callback state must contain torch latents")
        if pipeline is None:
            raise RuntimeError("content ISS analysis requires the real callback pipeline")
        base_image = _decode_callback_latents(pipeline, latents)
        embed = self._assets.embed_assets
        semantic = dino_last_layer_cls_patch_tiles(
            base_image, embed.dino_processor, embed.dino_model
        )
        texture = rgb_texture_tiles(base_image)
        baseline = _probe_observation(base_image, embed)

        def probe_evaluator(branch: str, candidate: torch.Tensor) -> Any:
            if branch not in {"lf", "hf"}:
                raise ValueError("public probe branch is invalid")
            return _probe_observation(_decode_callback_latents(pipeline, candidate), embed)

        probes = evaluate_public_probes(latents, baseline, probe_evaluator)
        allocation = allocate_content(ContentSignals(
            semantic,
            texture,
            probes.lf_two_scale_response_consistency,
            probes.hf_two_scale_response_consistency,
            probes.lf_local_perturbation_sensitivity,
            probes.hf_local_perturbation_sensitivity,
        ))
        embedded, measurement = embed_content_iss(
            latents,
            self._detection_key,
            embed.hf_public_assets,
            embed.lf_public_assets,
            allocation,
            self._beta,
        )
        updated = dict(callback_kwargs)
        updated["latents"] = embedded
        self._measurement = measurement
        return updated


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def run_content_iss_development_pair(
    pipeline: Any,
    unit: ContentISSUnit,
    development_key: bytes,
    assets: ContentISSDevelopmentAssets,
) -> ISSDevelopmentMeasurement:
    """Run plain host then content-whitening beta=1 joint generation for one dev unit."""

    if not isinstance(unit, ContentISSUnit) or unit.split != ISS_DEVELOPMENT_SPLIT:
        raise TypeError("content ISS development runtime requires a validated dev unit")
    if not isinstance(assets, ContentISSDevelopmentAssets):
        raise TypeError("content ISS development runtime requires frozen assets")
    plain = run_sd35_plain(
        pipeline,
        unit.prompt,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    )
    beta_one = run_sd35_content_unweighted(
        pipeline,
        unit.prompt,
        development_key,
        assets.embed_assets,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    )
    host_score = score_content_iss_image(plain, development_key, assets.lf_public_assets)
    beta_one_score = score_content_iss_image(
        beta_one.image, development_key, assets.lf_public_assets
    )
    wrong_scores = tuple(
        score_content_iss_image(beta_one.image, wrong_key, assets.lf_public_assets)
        for wrong_key in derive_development_wrong_keys(development_key)
    )
    if len(wrong_scores) != 16:
        raise RuntimeError("content ISS development requires exactly 16 wrong-key scores")
    return ISSDevelopmentMeasurement(
        host_score,
        beta_one_score,
        max(host_score, *wrong_scores),
    )


def _run_content_iss_pass2(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentISSEvaluationAssets,
    beta: float,
    *,
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[Image.Image, ContentAdaptiveMeasurement]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("embedding prompt must be non-empty text")
    if (
        not isinstance(height, int)
        or not isinstance(width, int)
        or height < 256
        or width < 256
    ):
        raise ValueError("image dimensions must be integers of at least 256")
    _validate_pipeline(pipeline)
    callback = ContentISSInjectionCallback(detection_key, assets, beta)
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
        raise RuntimeError("SD3.5 pipeline must return exactly one final image")
    if callback.measurement is None:
        raise RuntimeError("pipeline completed without content ISS step-18 embedding")
    return require_ordinary_rgb_image(images[0]), callback.measurement


def run_content_iss_evaluation_pair(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentISSEvaluationAssets,
    *,
    height: int,
    width: int,
    seed: int,
) -> ContentISSRunOutput:
    """Run callback-free pass 1, then the sole same-seed ISS pass 2."""

    if not isinstance(assets, ContentISSEvaluationAssets):
        raise TypeError("content ISS evaluation pair requires frozen evaluation assets")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("content ISS evaluation seed must be a nonnegative integer")
    primary_null = require_ordinary_rgb_image(run_sd35_plain(
        pipeline,
        prompt,
        height=height,
        width=width,
        generator=_generator(seed),
    ))
    host_score = score_content_iss_image(primary_null, detection_key, assets.lf_public_assets)
    beta = iss_beta(host_score, assets.iss_asset)
    image, measurement = _run_content_iss_pass2(
        pipeline,
        prompt,
        detection_key,
        assets,
        beta,
        height=height,
        width=width,
        generator=_generator(seed),
    )
    return ContentISSRunOutput(image, primary_null, measurement)


__all__ = [
    "ContentISSDevelopmentAssets",
    "ContentISSEvaluationAssets",
    "ContentISSInjectionCallback",
    "ContentISSRunOutput",
    "run_content_iss_development_pair",
    "run_content_iss_evaluation_pair",
]
