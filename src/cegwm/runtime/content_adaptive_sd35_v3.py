"""Real SD3.5 step-18 adapter for the Content V3 method candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from cegwm.method.content_adaptive_v3 import (
    CONTENT_V3_EVALUATED_CANDIDATE_ID,
    CONTENT_V3_METHOD_ID,
    DINO_ASSET_ID,
    HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID,
    LF_CONTENT_V3_EMBEDDING_TRANSFORM_ID,
    ContentAdaptiveMeasurement,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    embed_content_v3,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.method.hf import HF_CANDIDATE_ID, FrozenHFPublicAssets
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.runtime.content_adaptive_sd35_v2 import (
    _decode_callback_latents,
    _probe_observation,
    _validate_dino_assets,
    _validate_pipeline,
    load_dino_content_assets,
)
from cegwm.runtime.observation import require_ordinary_rgb_image

RUNTIME_ASSET_VALIDATION_CONTRACT_ID = (
    "dinov2_small_eager_bit_image_processor_public_size_semantics_v3"
)
_SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_SD35_IMAGE_PROCESSOR_ID = f"{_SD35_MODEL_ID}:image_processor"
_HF_DETECTOR_STATISTIC_ID = (
    "frozen_hf_final_rgb_public_vae_global_normalized_correlation"
)
_HF_EVALUATED_CANDIDATE_ID = "hf_tail_rademacher_v1_rankgate_v2"


@dataclass(frozen=True, slots=True)
class ContentV3EmbedAssets:
    """Embed-only DINO assets plus the frozen blind-detector assets."""

    dino_model: Any
    dino_processor: Any
    hf_public_assets: FrozenHFPublicAssets
    lf_public_assets: FrozenLFPublicAssets
    dino_asset_id: str = DINO_ASSET_ID
    runtime_asset_validation_contract_id: str = field(
        default=RUNTIME_ASSET_VALIDATION_CONTRACT_ID, init=False
    )
    content_method_id: str = field(default=CONTENT_V3_METHOD_ID, init=False)
    hf_detector_statistic_id: str = field(
        default=_HF_DETECTOR_STATISTIC_ID, init=False
    )
    hf_evaluated_candidate_id: str = field(
        default=_HF_EVALUATED_CANDIDATE_ID, init=False
    )
    hf_adaptive_embedding_transform_id: str = field(
        default=HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID, init=False
    )
    lf_embedding_transform_id: str = field(
        default=LF_CONTENT_V3_EMBEDDING_TRANSFORM_ID, init=False
    )
    evaluated_candidate_id: str = field(
        default=CONTENT_V3_EVALUATED_CANDIDATE_ID, init=False
    )

    def __post_init__(self) -> None:
        if self.dino_asset_id != DINO_ASSET_ID:
            raise ValueError("Content V3 embed asset must be facebook/dinov2-small")
        _validate_dino_assets(self.dino_model, self.dino_processor)
        if not isinstance(self.hf_public_assets, FrozenHFPublicAssets):
            raise TypeError("Content V3 HF assets must be FrozenHFPublicAssets")
        if not isinstance(self.lf_public_assets, FrozenLFPublicAssets):
            raise TypeError("Content V3 LF assets must be FrozenLFPublicAssets")
        hf = self.hf_public_assets
        lf = self.lf_public_assets
        if (
            hf.model_id != _SD35_MODEL_ID
            or hf.image_processor_id != _SD35_IMAGE_PROCESSOR_ID
            or hf.candidate_id != HF_CANDIDATE_ID
            or self.hf_detector_statistic_id != _HF_DETECTOR_STATISTIC_ID
            or self.hf_evaluated_candidate_id != _HF_EVALUATED_CANDIDATE_ID
            or self.hf_adaptive_embedding_transform_id
            != HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID
        ):
            raise ValueError("Content V3 HF frozen carrier or detector identity differs")
        if (
            lf.model_id != _SD35_MODEL_ID
            or lf.image_processor_id != _SD35_IMAGE_PROCESSOR_ID
            or lf.candidate_id != LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
            or lf.detector_statistic_id != LF_BLOCKNORM_DETECTOR_STATISTIC_ID
            or lf.evaluated_candidate_id
            != LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
            or self.content_method_id != CONTENT_V3_METHOD_ID
            or self.lf_embedding_transform_id
            != LF_CONTENT_V3_EMBEDDING_TRANSFORM_ID
            or self.evaluated_candidate_id
            != CONTENT_V3_EVALUATED_CANDIDATE_ID
        ):
            raise ValueError("Content V3 LF carrier, detector, or method identity differs")
        if hf.vae is not lf.vae or hf.image_processor is not lf.image_processor:
            raise ValueError(
                "Content V3 HF and LF detectors must share public observation assets"
            )


@dataclass(frozen=True, slots=True)
class ContentV3RunOutput:
    """Final RGB plus irreversible aggregate embedding measurements only."""

    image: Image.Image
    measurement: ContentAdaptiveMeasurement


class ContentV3InjectionCallback:
    """Analyze content and embed both Content V3 branches once at step 18."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        assets: ContentV3EmbedAssets,
    ) -> None:
        if not isinstance(assets, ContentV3EmbedAssets):
            raise TypeError("Content V3 callback requires ContentV3EmbedAssets")
        self._detection_key = detection_key
        self._assets = assets
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
            raise RuntimeError("Content V3 callback attempted step 18 more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("diffusers callback state must contain torch latents")
        if pipeline is None:
            raise RuntimeError("Content V3 analysis requires the real callback pipeline")
        base_image = _decode_callback_latents(pipeline, latents)
        semantic = dino_last_layer_cls_patch_tiles(
            base_image, self._assets.dino_processor, self._assets.dino_model
        )
        texture = rgb_texture_tiles(base_image)
        baseline = _probe_observation(base_image, self._assets)

        def probe_evaluator(branch: str, candidate: torch.Tensor):
            if branch not in {"lf", "hf"}:
                raise ValueError("public probe branch is invalid")
            return _probe_observation(
                _decode_callback_latents(pipeline, candidate), self._assets
            )

        probes = evaluate_public_probes(latents, baseline, probe_evaluator)
        allocation = allocate_content(ContentSignals(
            semantic,
            texture,
            probes.lf_two_scale_response_consistency,
            probes.hf_two_scale_response_consistency,
            probes.lf_local_perturbation_sensitivity,
            probes.hf_local_perturbation_sensitivity,
        ))
        embedded, measurement = embed_content_v3(
            latents,
            self._detection_key,
            self._assets.hf_public_assets,
            self._assets.lf_public_assets,
            allocation,
        )
        updated = dict(callback_kwargs)
        updated["latents"] = embedded
        self._measurement = measurement
        return updated


def run_sd35_content_v3(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentV3EmbedAssets,
    *,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
) -> ContentV3RunOutput:
    """Run one real Content V3 joint embedding and return public outputs only."""

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
    callback = ContentV3InjectionCallback(detection_key, assets)
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
        raise RuntimeError("pipeline completed without Content V3 step-18 embedding")
    return ContentV3RunOutput(
        require_ordinary_rgb_image(images[0]), callback.measurement
    )


__all__ = [
    "ContentV3EmbedAssets",
    "ContentV3InjectionCallback",
    "ContentV3RunOutput",
    "RUNTIME_ASSET_VALIDATION_CONTRACT_ID",
    "load_dino_content_assets",
    "run_sd35_content_v3",
]
