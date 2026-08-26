"""Real Content V6 generation and blind paired-score collection for V9 calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cegwm.method.content_weighted_joint_v9 import (
    LFHFScorePair,
    WeightedJointAsset,
    weighted_joint_score,
)
from cegwm.method.content_whitening_v4 import score_content_v4_lf_image
from cegwm.method.hf import score_hf_image
from cegwm.protocol.content_chain_v9 import (
    CONTENT_V9_CALIBRATION_SPLIT,
    CONTENT_V9_WRONG_KEY_DOMAIN,
    ContentV9Unit,
)
from cegwm.protocol.content_chain_v9_stability import ContentV9StabilityUnit
from cegwm.runtime.content_iss_sd35_v6 import (
    ContentV6EvaluationAssets,
    ContentV6RunOutput,
    run_content_v6_evaluation_pair,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key
from cegwm.shared.prg import prg_bytes


@dataclass(frozen=True, slots=True)
class ContentV9CalibrationAssets:
    v6_assets: ContentV6EvaluationAssets

    def __post_init__(self) -> None:
        if not isinstance(self.v6_assets, ContentV6EvaluationAssets):
            raise TypeError("Content V9 calibration requires real Content V6 assets")


@dataclass(frozen=True, slots=True)
class ContentV9StabilityOutput:
    image: Any
    primary_null: Any
    measurement: Any
    candidate_scores: dict[str, dict[str, float]]
    primary_null_scores: dict[str, dict[str, float]]


def derive_calibration_wrong_keys(calibration_key: bytes) -> tuple[bytes, ...]:
    key = normalize_detection_key(calibration_key)
    return tuple(
        prg_bytes(key, f"{CONTENT_V9_WRONG_KEY_DOMAIN}/index={index}", 32)
        for index in range(16)
    )


def derive_stability_wrong_keys(detection_key: bytes) -> tuple[bytes, ...]:
    """Use the unchanged formal external-wrong-key domain for stability scoring."""

    return derive_calibration_wrong_keys(detection_key)


def _blind_pair(image: Any, key: bytes, assets: ContentV9CalibrationAssets) -> LFHFScorePair:
    ordinary = require_ordinary_rgb_image(image)
    lf = float(score_content_v4_lf_image(ordinary, key, assets.v6_assets.lf_public_assets))
    hf = float(score_hf_image(ordinary, key, assets.v6_assets.hf_public_assets))
    if not math.isfinite(lf) or not math.isfinite(hf) or not -1.0 <= lf <= 1.0 or not -1.0 <= hf <= 1.0:
        raise ValueError("Content V9 blind branch scores must be finite in [-1, 1]")
    return LFHFScorePair(lf, hf)


def run_content_v9_calibration_unit(
    pipeline: Any,
    unit: ContentV9Unit,
    calibration_key: bytes,
    assets: ContentV9CalibrationAssets,
) -> tuple[LFHFScorePair, ...]:
    """Generate one V6 pair and return exactly 33 ordered null pairs.

    Candidate registered scores are deliberately not sampled.
    """

    if not isinstance(unit, ContentV9Unit) or unit.split != CONTENT_V9_CALIBRATION_SPLIT:
        raise TypeError("Content V9 calibration runtime requires a validated calibration unit")
    if not isinstance(assets, ContentV9CalibrationAssets):
        raise TypeError("Content V9 calibration runtime requires frozen assets")
    wrong_keys = derive_calibration_wrong_keys(calibration_key)
    if len(wrong_keys) != 16:
        raise RuntimeError("Content V9 calibration requires exactly 16 wrong keys")
    output = run_content_v6_evaluation_pair(
        pipeline,
        unit.prompt,
        calibration_key,
        assets.v6_assets,
        height=unit.height,
        width=unit.width,
        seed=unit.seed,
    )
    if not isinstance(output, ContentV6RunOutput):
        raise TypeError("Content V9 calibration requires a real Content V6 pair result")
    pairs = [
        *(_blind_pair(output.image, wrong_key, assets) for wrong_key in wrong_keys),
        _blind_pair(output.primary_null, calibration_key, assets),
        *(_blind_pair(output.primary_null, wrong_key, assets) for wrong_key in wrong_keys),
    ]
    if len(pairs) != 33:
        raise RuntimeError("Content V9 calibration unit must yield exactly 33 score pairs")
    return tuple(pairs)


def blind_weighted_scores(
    image: Any,
    detection_key: bytes,
    wrong_keys: Sequence[bytes],
    assets: ContentV9CalibrationAssets,
    calibration_asset: WeightedJointAsset,
) -> dict[str, dict[str, float]]:
    """Score one ordinary image through unchanged LF/HF and frozen V9 J."""

    if len(wrong_keys) != 16 or any(not isinstance(key, bytes) for key in wrong_keys):
        raise ValueError("Content V9 blind scoring requires exactly 16 wrong keys")
    labels = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    keys = (normalize_detection_key(detection_key), *wrong_keys)
    pairs = tuple(_blind_pair(image, key, assets) for key in keys)
    lf = {label: pair.lf for label, pair in zip(labels, pairs, strict=True)}
    hf = {label: pair.hf for label, pair in zip(labels, pairs, strict=True)}
    weighted = {
        label: weighted_joint_score(pair.lf, pair.hf, calibration_asset)
        for label, pair in zip(labels, pairs, strict=True)
    }
    return {"lf": lf, "hf": hf, "weighted_joint": weighted}


def run_content_v9_stability_unit(
    pipeline: Any,
    unit: ContentV9StabilityUnit,
    detection_key: bytes,
    wrong_keys: Sequence[bytes],
    assets: ContentV9CalibrationAssets,
    calibration_asset: WeightedJointAsset,
) -> ContentV9StabilityOutput:
    """Run the unchanged V6 pair and score both final images with frozen V9 J."""

    if not isinstance(unit, ContentV9StabilityUnit):
        raise TypeError("Content V9 stability runtime requires a validated unit")
    if not isinstance(assets, ContentV9CalibrationAssets):
        raise TypeError("Content V9 stability runtime requires frozen V6 assets")
    if not isinstance(calibration_asset, WeightedJointAsset):
        raise TypeError("Content V9 stability runtime requires the accepted calibration asset")
    if len(wrong_keys) != 16 or any(not isinstance(key, bytes) for key in wrong_keys):
        raise ValueError("Content V9 stability runtime requires exactly 16 wrong keys")
    output = run_content_v6_evaluation_pair(
        pipeline,
        unit.prompt,
        detection_key,
        assets.v6_assets,
        height=unit.height,
        width=unit.width,
        seed=unit.seed,
    )
    if not isinstance(output, ContentV6RunOutput):
        raise TypeError("Content V9 stability requires a real Content V6 pair result")
    candidate_scores = blind_weighted_scores(
        output.image, detection_key, wrong_keys, assets, calibration_asset
    )
    primary_null_scores = blind_weighted_scores(
        output.primary_null, detection_key, wrong_keys, assets, calibration_asset
    )
    return ContentV9StabilityOutput(
        output.image,
        output.primary_null,
        output.measurement,
        candidate_scores,
        primary_null_scores,
    )


__all__ = [
    "ContentV9CalibrationAssets",
    "ContentV9StabilityOutput",
    "blind_weighted_scores",
    "derive_calibration_wrong_keys",
    "derive_stability_wrong_keys",
    "run_content_v9_calibration_unit",
    "run_content_v9_stability_unit",
]
