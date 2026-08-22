"""Real SD3.5 step-18 adapter for v2 content-adaptive dual-branch embedding."""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from cegwm.method.content_adaptive_v2 import (
    DINO_ASSET_ID,
    HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID,
    JOINT_EVALUATED_CANDIDATE_ID,
    LF_ADAPTIVE_EMBEDDING_TRANSFORM_ID,
    ContentAdaptiveMeasurement,
    ContentSignals,
    ProbeObservation,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    embed_content_adaptive,
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
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image

_SD35_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_SD35_IMAGE_PROCESSOR_ID = f"{_SD35_MODEL_ID}:image_processor"
_HF_DETECTOR_STATISTIC_ID = "frozen_hf_final_rgb_public_vae_global_normalized_correlation"
_HF_EVALUATED_CANDIDATE_ID = "hf_tail_rademacher_v1_rankgate_v2"


def _dino_source_identity(asset: Any) -> str | None:
    for owner in (asset, getattr(asset, "config", None), getattr(asset, "init_kwargs", None)):
        if isinstance(owner, dict):
            value = owner.get("_name_or_path") or owner.get("name_or_path")
        else:
            value = getattr(owner, "_name_or_path", None) or getattr(owner, "name_or_path", None)
        if isinstance(value, str) and value:
            return value
    return None


def _validate_dino_assets(model: Any, processor: Any) -> None:
    if _dino_source_identity(model) != DINO_ASSET_ID:
        raise RuntimeError("content DINO model identity is missing or differs")
    if _dino_source_identity(processor) != DINO_ASSET_ID:
        raise RuntimeError("content DINO processor identity is missing or differs")
    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "eager":
        raise RuntimeError("content DINO asset must expose eager attention")
    if not callable(processor):
        raise TypeError("content DINO processor must be callable")


@dataclass(frozen=True, slots=True)
class ContentEmbedAssets:
    """Embed-only DINO assets plus frozen blind-detector assets."""

    dino_model: Any
    dino_processor: Any
    hf_public_assets: FrozenHFPublicAssets
    lf_public_assets: FrozenLFPublicAssets
    dino_asset_id: str = DINO_ASSET_ID
    hf_detector_statistic_id: str = field(default=_HF_DETECTOR_STATISTIC_ID, init=False)
    hf_evaluated_candidate_id: str = field(default=_HF_EVALUATED_CANDIDATE_ID, init=False)
    hf_adaptive_embedding_transform_id: str = field(
        default=HF_ADAPTIVE_EMBEDDING_TRANSFORM_ID, init=False
    )
    lf_adaptive_embedding_transform_id: str = field(
        default=LF_ADAPTIVE_EMBEDDING_TRANSFORM_ID, init=False
    )
    evaluated_candidate_id: str = field(default=JOINT_EVALUATED_CANDIDATE_ID, init=False)

    def __post_init__(self) -> None:
        if self.dino_asset_id != DINO_ASSET_ID:
            raise ValueError("content embed asset must be facebook/dinov2-small")
        _validate_dino_assets(self.dino_model, self.dino_processor)
        if not isinstance(self.hf_public_assets, FrozenHFPublicAssets):
            raise TypeError("content HF assets must be FrozenHFPublicAssets")
        if not isinstance(self.lf_public_assets, FrozenLFPublicAssets):
            raise TypeError("content LF assets must be FrozenLFPublicAssets")
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
            raise ValueError("content HF frozen carrier, detector, or evaluated identity differs")
        if (
            lf.model_id != _SD35_MODEL_ID
            or lf.image_processor_id != _SD35_IMAGE_PROCESSOR_ID
            or lf.candidate_id != LF_BALANCED_BLOCKS_CARRIER_METHOD_ID
            or lf.detector_statistic_id != LF_BLOCKNORM_DETECTOR_STATISTIC_ID
            or lf.evaluated_candidate_id != LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID
            or self.lf_adaptive_embedding_transform_id
            != LF_ADAPTIVE_EMBEDDING_TRANSFORM_ID
            or self.evaluated_candidate_id != JOINT_EVALUATED_CANDIDATE_ID
        ):
            raise ValueError("content LF frozen carrier, detector, or evaluated identity differs")
        if hf.vae is not lf.vae or hf.image_processor is not lf.image_processor:
            raise ValueError("content HF and LF detectors must share the same frozen public observation assets")


@dataclass(frozen=True, slots=True)
class ContentAdaptiveRunOutput:
    """Final RGB plus irreversible aggregate embedding measurements only."""

    image: Image.Image
    measurement: ContentAdaptiveMeasurement


def load_dino_content_assets(*, token: str | None = None) -> tuple[Any, Any]:
    """Load the named public embed-only asset with eager attention enabled."""

    try:
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("transformers is required for DINO content analysis") from error
    model_class = getattr(transformers, "AutoModel", None)
    processor_class = getattr(transformers, "AutoImageProcessor", None)
    if model_class is None or processor_class is None:
        raise RuntimeError("installed transformers lacks DINO auto classes")
    kwargs = {"token": token} if token else {}
    processor = processor_class.from_pretrained(DINO_ASSET_ID, **kwargs)
    model = model_class.from_pretrained(
        DINO_ASSET_ID,
        attn_implementation="eager",
        **kwargs,
    )
    _validate_dino_assets(model, processor)
    return model, processor


def _config_scalar(config: Any, name: str, *, positive: bool) -> float:
    value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"SD3 VAE {name} is missing or invalid")
    scalar = float(value)
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        raise RuntimeError(f"SD3 VAE {name} is missing or invalid")
    return scalar


def _decode_callback_latents(pipeline: Any, latents: torch.Tensor) -> Image.Image:
    """Decode the complete current callback latent using SD3 VAE coordinates."""

    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    if not callable(getattr(vae, "decode", None)):
        raise RuntimeError("SD3 pipeline VAE cannot decode callback latents")
    if not callable(getattr(processor, "postprocess", None)):
        raise RuntimeError("SD3 pipeline image processor cannot postprocess decoded latents")
    config = getattr(vae, "config", None)
    scaling = _config_scalar(config, "scaling_factor", positive=True)
    shift = _config_scalar(config, "shift_factor", positive=False)
    try:
        parameter = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("SD3 pipeline VAE device and dtype cannot be resolved") from error
    if not parameter.dtype.is_floating_point:
        raise RuntimeError("SD3 pipeline VAE must use a floating dtype")
    coordinate = (latents.to(torch.float32) / scaling + shift).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    with torch.no_grad():
        decoded = vae.decode(coordinate, return_dict=True)
    sample = getattr(decoded, "sample", None)
    if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[0] != 1:
        raise RuntimeError("SD3 VAE probe decode returned an invalid sample")
    if not bool(torch.isfinite(sample).all()):
        raise RuntimeError("SD3 VAE probe decode returned nonfinite pixels")
    images = processor.postprocess(sample, output_type="pil")
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("SD3 VAE probe decode must return exactly one image")
    return require_ordinary_rgb_image(images[0])


def _probe_observation(
    image: Image.Image,
    assets: ContentEmbedAssets,
) -> ProbeObservation:
    """Return I and E(I); v2 responses are formed only by later subtraction."""

    ordinary = require_ordinary_rgb_image(image)
    pixels = torch.from_numpy(np.asarray(ordinary, dtype=np.float64).copy()).permute(
        2, 0, 1
    ).unsqueeze(0) / 255.0
    public = assets.hf_public_assets
    observation = encode_final_rgb_image(ordinary, public.image_processor, public.vae)
    return ProbeObservation(rgb=pixels, vae=observation)


class ContentAdaptiveInjectionCallback:
    """Analyze and co-embed both branches once at the same frozen step 18."""

    tensor_inputs = ("latents",)

    def __init__(
        self,
        detection_key: str | bytes | bytearray | memoryview,
        assets: ContentEmbedAssets,
    ) -> None:
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
            raise RuntimeError("content-adaptive callback attempted step 18 more than once")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("diffusers callback state must contain torch latents")
        if pipeline is None:
            raise RuntimeError("content analysis requires the real callback pipeline")
        base_image = _decode_callback_latents(pipeline, latents)
        semantic = dino_last_layer_cls_patch_tiles(
            base_image,
            self._assets.dino_processor,
            self._assets.dino_model,
        )
        texture = rgb_texture_tiles(base_image)
        baseline = _probe_observation(base_image, self._assets)

        def probe_evaluator(branch: str, candidate: torch.Tensor) -> ProbeObservation:
            if branch not in {"lf", "hf"}:
                raise ValueError("public probe branch is invalid")
            return _probe_observation(_decode_callback_latents(pipeline, candidate), self._assets)

        probes = evaluate_public_probes(latents, baseline, probe_evaluator)
        allocation = allocate_content(ContentSignals(
            semantic,
            texture,
            probes.lf_two_scale_response_consistency,
            probes.hf_two_scale_response_consistency,
            probes.lf_local_perturbation_sensitivity,
            probes.hf_local_perturbation_sensitivity,
        ))
        embedded, measurement = embed_content_adaptive(
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


def _validate_pipeline(pipeline: Any) -> None:
    call = getattr(pipeline, "__call__", None)
    if not callable(call):
        raise TypeError("SD3.5 pipeline must be callable")
    parameters = inspect.signature(call).parameters
    has_kwargs = any(value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values())
    required = {"callback_on_step_end", "callback_on_step_end_tensor_inputs"}
    if not has_kwargs and not required.issubset(parameters):
        raise TypeError("pipeline lacks the required diffusers callback-on-step-end API")


def run_sd35_content_adaptive(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentEmbedAssets,
    *,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
) -> ContentAdaptiveRunOutput:
    """Run one real joint embedding and return only final RGB plus aggregates."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("embedding prompt must be non-empty text")
    if not isinstance(height, int) or not isinstance(width, int) or height < 256 or width < 256:
        raise ValueError("image dimensions must be integers of at least 256")
    _validate_pipeline(pipeline)
    callback = ContentAdaptiveInjectionCallback(detection_key, assets)
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
        raise RuntimeError("pipeline completed without the frozen step-18 joint embedding")
    return ContentAdaptiveRunOutput(require_ordinary_rgb_image(images[0]), callback.measurement)
