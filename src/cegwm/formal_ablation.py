"""Frozen embedding-side ablations for the paper's minimal mechanism study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from cegwm.method.content_adaptive import (
    COMBINED_RELATIVE_L2,
    TILE_COUNT,
    ContentAllocation,
    ContentSignals,
    allocate_content,
    dino_last_layer_cls_patch_tiles,
    evaluate_public_probes,
    rgb_texture_tiles,
)
from cegwm.method.content_iss import embed_content_iss, iss_beta, score_content_iss_image
from cegwm.method.content_unweighted import _content_unweighted_branch_deltas, _relative_l2
from cegwm.runtime.content_adaptive_sd35 import (
    _decode_callback_latents,
    _probe_observation,
    _validate_pipeline,
)
from cegwm.runtime.content_iss_sd35 import ContentISSEvaluationAssets
from cegwm.runtime.diffusers_sd35 import run_sd35_plain
from cegwm.runtime.observation import require_ordinary_rgb_image


EMBEDDING_ABLATIONS = ("no_content_adaptive", "lf_only", "hf_only")


@dataclass(frozen=True, slots=True)
class FormalAblationPair:
    image: Image.Image
    primary_null: Image.Image
    variant: str


def _uniform_allocation() -> ContentAllocation:
    weights = (1.0,) * TILE_COUNT
    return ContentAllocation(weights, weights, 0.5, 0.5, (0.0,) * 6)


def _adaptive_allocation(
    latents: torch.Tensor,
    pipeline: Any,
    assets: ContentISSEvaluationAssets,
) -> ContentAllocation:
    embed = assets.embed_assets
    base_image = _decode_callback_latents(pipeline, latents)
    semantic = dino_last_layer_cls_patch_tiles(
        base_image, embed.dino_processor, embed.dino_model
    )
    texture = rgb_texture_tiles(base_image)
    baseline = _probe_observation(base_image, embed)

    def probe(branch: str, candidate: torch.Tensor) -> Any:
        if branch not in {"lf", "hf"}:
            raise ValueError("formal ablation probe branch differs")
        return _probe_observation(_decode_callback_latents(pipeline, candidate), embed)

    observations = evaluate_public_probes(latents, baseline, probe)
    return allocate_content(ContentSignals(
        semantic,
        texture,
        observations.lf_two_scale_response_consistency,
        observations.hf_two_scale_response_consistency,
        observations.lf_local_perturbation_sensitivity,
        observations.hf_local_perturbation_sensitivity,
    ))


def _single_branch(
    latents: torch.Tensor,
    key: bytes,
    assets: ContentISSEvaluationAssets,
    allocation: ContentAllocation,
    variant: str,
    beta: float,
) -> torch.Tensor:
    lf_delta, hf_delta = _content_unweighted_branch_deltas(
        latents,
        key,
        assets.embed_assets.hf_public_assets,
        assets.embed_assets.lf_public_assets,
        allocation,
    )
    selected = lf_delta * beta if variant == "lf_only" else hf_delta
    direction = selected.to(torch.float64)
    norm = torch.linalg.vector_norm(direction)
    base64 = latents.to(torch.float64)
    base_norm = torch.linalg.vector_norm(base64)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("single-branch ablation direction is invalid")
    target = direction / norm * base_norm * COMBINED_RELATIVE_L2

    def candidate(scale: float) -> torch.Tensor:
        return (base64 + scale * target).to(latents.dtype)

    low, high = 0.0, 2.0
    best = latents.detach().clone()
    measurement = _relative_l2(latents, best)
    for _ in range(96):
        middle = (low + high) / 2.0
        trial = candidate(middle)
        current = _relative_l2(latents, trial)
        if current.relative_l2 <= COMBINED_RELATIVE_L2:
            low, best, measurement = middle, trial, current
        else:
            high = middle
    if measurement.perturbation_l2 == 0.0 or measurement.relative_l2 > COMBINED_RELATIVE_L2:
        raise RuntimeError("single-branch ablation violates the shared budget")
    return best


class _VariantCallback:
    tensor_inputs = ("latents",)

    def __init__(self, variant: str, key: bytes, assets: ContentISSEvaluationAssets, beta: float) -> None:
        if variant not in EMBEDDING_ABLATIONS:
            raise ValueError("unknown embedding ablation")
        self.variant = variant
        self.key = key
        self.assets = assets
        self.beta = beta
        self.executed = False

    def __call__(self, pipeline: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        del timestep
        if step_index != 18:
            return callback_kwargs
        if self.executed:
            raise RuntimeError("formal ablation callback executed twice")
        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError("formal ablation callback requires latent tensor")
        allocation = (
            _uniform_allocation()
            if self.variant == "no_content_adaptive"
            else _adaptive_allocation(latents, pipeline, self.assets)
        )
        if self.variant == "no_content_adaptive":
            embedded, _ = embed_content_iss(
                latents,
                self.key,
                self.assets.embed_assets.hf_public_assets,
                self.assets.embed_assets.lf_public_assets,
                allocation,
                self.beta,
            )
        else:
            embedded = _single_branch(
                latents, self.key, self.assets, allocation, self.variant, self.beta
            )
        self.executed = True
        updated = dict(callback_kwargs)
        updated["latents"] = embedded
        return updated


def run_formal_ablation_pair(
    pipeline: Any,
    prompt: str,
    key: bytes,
    assets: ContentISSEvaluationAssets,
    variant: str,
    *,
    seed: int,
    height: int = 512,
    width: int = 512,
) -> FormalAblationPair:
    """Run one same-seed clean/ablated pair under the frozen 0.012 budget."""

    if variant not in EMBEDDING_ABLATIONS:
        raise ValueError("unknown embedding ablation")
    _validate_pipeline(pipeline)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    primary_null = require_ordinary_rgb_image(run_sd35_plain(
        pipeline, prompt, height=height, width=width, generator=generator
    ))
    host = score_content_iss_image(primary_null, key, assets.lf_public_assets)
    beta = iss_beta(host, assets.iss_asset)
    callback = _VariantCallback(variant, key, assets, beta)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        height=height,
        width=width,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1 or not callback.executed:
        raise RuntimeError("formal ablation generation did not complete once")
    return FormalAblationPair(
        require_ordinary_rgb_image(images[0]), primary_null, variant
    )


__all__ = ["EMBEDDING_ABLATIONS", "FormalAblationPair", "run_formal_ablation_pair"]
