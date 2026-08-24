"""Real paired SD3.5 development path for the Content V6 ISS fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cegwm.method.content_iss_v6 import (
    ISSDevelopmentMeasurement,
    content_v6_h,
    derive_development_wrong_keys,
)
from cegwm.method.content_whitening_v4 import FrozenContentV4LFPublicAssets
from cegwm.protocol.content_chain_v6 import ContentV6Unit, V6_DEVELOPMENT_SPLIT
from cegwm.runtime.content_adaptive_sd35_v3 import ContentV3EmbedAssets, run_sd35_content_v3
from cegwm.runtime.diffusers_sd35 import run_sd35_plain


@dataclass(frozen=True, slots=True)
class ContentV6DevelopmentAssets:
    embed_assets: ContentV3EmbedAssets
    lf_public_assets: FrozenContentV4LFPublicAssets

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentV3EmbedAssets):
            raise TypeError("Content V6 development requires Content V3 embed assets")
        if not isinstance(self.lf_public_assets, FrozenContentV4LFPublicAssets):
            raise TypeError("Content V6 development requires frozen V4 LF assets")
        if self.lf_public_assets.carrier_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("Content V6 embed and detector must share LF carrier assets")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def run_content_v6_development_pair(
    pipeline: Any,
    unit: ContentV6Unit,
    development_key: bytes,
    assets: ContentV6DevelopmentAssets,
) -> ISSDevelopmentMeasurement:
    """Run plain host then unchanged V4 beta=1 joint generation for one dev unit."""

    if not isinstance(unit, ContentV6Unit) or unit.split != V6_DEVELOPMENT_SPLIT:
        raise TypeError("Content V6 development runtime requires a validated dev unit")
    if not isinstance(assets, ContentV6DevelopmentAssets):
        raise TypeError("Content V6 development runtime requires frozen assets")
    plain = run_sd35_plain(
        pipeline,
        unit.prompt,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    )
    beta_one = run_sd35_content_v3(
        pipeline,
        unit.prompt,
        development_key,
        assets.embed_assets,
        height=unit.height,
        width=unit.width,
        generator=_generator(unit.seed),
    )
    host_score = content_v6_h(plain, development_key, assets.lf_public_assets)
    beta_one_score = content_v6_h(
        beta_one.image, development_key, assets.lf_public_assets
    )
    wrong_scores = tuple(
        content_v6_h(beta_one.image, wrong_key, assets.lf_public_assets)
        for wrong_key in derive_development_wrong_keys(development_key)
    )
    if len(wrong_scores) != 16:
        raise RuntimeError("Content V6 development requires exactly 16 wrong-key scores")
    return ISSDevelopmentMeasurement(
        host_score,
        beta_one_score,
        max(host_score, *wrong_scores),
    )


__all__ = ["ContentV6DevelopmentAssets", "run_content_v6_development_pair"]
