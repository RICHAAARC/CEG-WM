"""Real SD3.5 fit and two-pass evaluation path for Content V7 ISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.content_adaptive_v3 import (
    ContentAdaptiveMeasurement,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.method.content_iss_v7 import (
    ISSAsset,
    ISSDevelopmentMeasurement,
    derive_development_wrong_keys,
    embed_content_v7,
    iss_beta,
    score_content_v7_lf,
)
from cegwm.method.lf import FrozenLFPublicAssets
from cegwm.protocol.content_chain_v7 import ContentV7Unit, V7_DEVELOPMENT_SPLIT
from cegwm.runtime.content_adaptive_sd35_v2 import (
    _decode_callback_latents,
    _probe_observation,
    _validate_pipeline,
)
from cegwm.runtime.content_adaptive_sd35_v3 import (
    ContentV3EmbedAssets,
    run_sd35_content_v3,
)
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image


@dataclass(frozen=True, slots=True)
class ContentV7DevelopmentAssets:
    embed_assets: ContentV3EmbedAssets
    lf_public_assets: FrozenLFPublicAssets

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentV3EmbedAssets):
            raise TypeError("Content V7 development requires Content V3 embed assets")
        if not isinstance(self.lf_public_assets, FrozenLFPublicAssets):
            raise TypeError("Content V7 development requires frozen LF public assets")
        if self.lf_public_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("Content V7 writer and detector must share LF public assets")


@dataclass(frozen=True, slots=True)
class ContentV7EvaluationAssets:
    embed_assets: ContentV3EmbedAssets
    lf_public_assets: FrozenLFPublicAssets
    iss_asset: ISSAsset

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentV3EmbedAssets):
            raise TypeError("Content V7 evaluation requires Content V3 embed assets")
        if not isinstance(self.lf_public_assets, FrozenLFPublicAssets):
            raise TypeError("Content V7 evaluation requires frozen LF public assets")
        if not isinstance(self.iss_asset, ISSAsset):
            raise TypeError("Content V7 evaluation requires its published ISS asset")
        if self.lf_public_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("Content V7 writer and detector must share LF public assets")

    @property
    def hf_public_assets(self) -> Any:
        return self.embed_assets.hf_public_assets


@dataclass(frozen=True, slots=True)
class ContentV7RunOutput:
    image: Image.Image
    primary_null: Image.Image
    measurement: ContentAdaptiveMeasurement


class ContentV7InjectionCallback:
    """Run V3 analysis and apply beta to LF alone before shared projection."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        assets: ContentV7EvaluationAssets,
        beta: float,
    ) -> None:
        if not isinstance(assets, ContentV7EvaluationAssets):
            raise TypeError("Content V7 callback requires evaluation assets")
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
            raise RuntimeError("Content V7 callback attempted step 18 more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("diffusers callback state must contain torch latents")
        if pipeline is None:
            raise RuntimeError("Content V7 analysis requires the real callback pipeline")
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
        embedded, measurement = embed_content_v7(
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


def run_content_v7_development_pair(
    pipeline: Any,
    unit: ContentV7Unit,
    development_key: bytes,
    assets: ContentV7DevelopmentAssets,
) -> ISSDevelopmentMeasurement:
    """Run same-seed plain then unweighted V3 writer and score final RGB."""

    if not isinstance(unit, ContentV7Unit) or unit.split != V7_DEVELOPMENT_SPLIT:
        raise TypeError("Content V7 development runtime requires a validated dev unit")
    if not isinstance(assets, ContentV7DevelopmentAssets):
        raise TypeError("Content V7 development runtime requires frozen assets")
    primary_null = require_ordinary_rgb_image(run_sd35_plain(
        pipeline,
        unit.prompt,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    ))
    beta_one = run_sd35_content_v3(
        pipeline,
        unit.prompt,
        development_key,
        assets.embed_assets,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    )
    beta_one_image = require_ordinary_rgb_image(beta_one.image)
    host_score = score_content_v7_lf(
        primary_null, development_key, assets.lf_public_assets
    )
    beta_one_score = score_content_v7_lf(
        beta_one_image, development_key, assets.lf_public_assets
    )
    wrong_scores = tuple(
        score_content_v7_lf(beta_one_image, wrong_key, assets.lf_public_assets)
        for wrong_key in derive_development_wrong_keys(development_key)
    )
    if len(wrong_scores) != 16:
        raise RuntimeError("Content V7 development requires exactly 16 wrong-key scores")
    return ISSDevelopmentMeasurement(
        host_score,
        beta_one_score,
        max(host_score, *wrong_scores),
    )


def _run_content_v7_pass2(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentV7EvaluationAssets,
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
    callback = ContentV7InjectionCallback(detection_key, assets, beta)
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
        raise RuntimeError("pipeline completed without Content V7 step-18 embedding")
    return require_ordinary_rgb_image(images[0]), callback.measurement


def run_content_v7_evaluation_pair(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentV7EvaluationAssets,
    *,
    height: int,
    width: int,
    seed: int,
) -> ContentV7RunOutput:
    """Run the sole callback-free null then same-seed ISS writer pass."""

    if not isinstance(assets, ContentV7EvaluationAssets):
        raise TypeError("Content V7 evaluation pair requires frozen evaluation assets")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Content V7 evaluation seed must be a nonnegative integer")
    primary_null = require_ordinary_rgb_image(run_sd35_plain(
        pipeline,
        prompt,
        height=height,
        width=width,
        generator=_generator(seed),
    ))
    host_score = score_content_v7_lf(
        primary_null, detection_key, assets.lf_public_assets
    )
    beta = iss_beta(host_score, assets.iss_asset)
    image, measurement = _run_content_v7_pass2(
        pipeline,
        prompt,
        detection_key,
        assets,
        beta,
        height=height,
        width=width,
        generator=_generator(seed),
    )
    return ContentV7RunOutput(image, primary_null, measurement)


__all__ = [
    "ContentV7DevelopmentAssets",
    "ContentV7EvaluationAssets",
    "ContentV7InjectionCallback",
    "ContentV7RunOutput",
    "run_content_v7_development_pair",
    "run_content_v7_evaluation_pair",
]
