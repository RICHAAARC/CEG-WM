"""Real content ISS generation and blind paired-score collection for content calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cegwm.method.content_weighted_joint import (
    LFHFScorePair,
    WeightedJointAsset,
    weighted_joint_score,
)
from cegwm.method.content_whitening import score_content_whitened_lf_image
from cegwm.method.hf import score_hf_image
from cegwm.protocol.content_calibration import (
    CONTENT_CALIBRATION_CALIBRATION_SPLIT,
    CONTENT_CALIBRATION_WRONG_KEY_DOMAIN,
    ContentCalibrationUnit,
)
from cegwm.protocol.content_chain import ContentChainUnit
from cegwm.runtime.content_iss_sd35 import (
    ContentISSEvaluationAssets,
    ContentISSRunOutput,
    run_content_iss_evaluation_pair,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key
from cegwm.shared.prg import prg_bytes


@dataclass(frozen=True, slots=True)
class ContentCalibrationAssets:
    iss_assets: ContentISSEvaluationAssets

    def __post_init__(self) -> None:
        if not isinstance(self.iss_assets, ContentISSEvaluationAssets):
            raise TypeError("content calibration requires real content ISS assets")


@dataclass(frozen=True, slots=True)
class ContentChainOutput:
    image: Any
    primary_null: Any
    measurement: Any
    candidate_scores: dict[str, dict[str, float]]
    primary_null_scores: dict[str, dict[str, float]]


def derive_calibration_wrong_keys(calibration_key: bytes) -> tuple[bytes, ...]:
    key = normalize_detection_key(calibration_key)
    return tuple(
        prg_bytes(key, f"{CONTENT_CALIBRATION_WRONG_KEY_DOMAIN}/index={index}", 32)
        for index in range(16)
    )


def derive_stability_wrong_keys(detection_key: bytes) -> tuple[bytes, ...]:
    """Use the unchanged formal external-wrong-key domain for stability scoring."""

    return derive_calibration_wrong_keys(detection_key)


def _blind_pair(image: Any, key: bytes, assets: ContentCalibrationAssets) -> LFHFScorePair:
    ordinary = require_ordinary_rgb_image(image)
    lf = float(score_content_whitened_lf_image(ordinary, key, assets.iss_assets.lf_public_assets))
    hf = float(score_hf_image(ordinary, key, assets.iss_assets.hf_public_assets))
    if not math.isfinite(lf) or not math.isfinite(hf) or not -1.0 <= lf <= 1.0 or not -1.0 <= hf <= 1.0:
        raise ValueError("content blind branch scores must be finite in [-1, 1]")
    return LFHFScorePair(lf, hf)


def run_content_calibration_unit(
    pipeline: Any,
    unit: ContentCalibrationUnit,
    calibration_key: bytes,
    assets: ContentCalibrationAssets,
) -> tuple[LFHFScorePair, ...]:
    """Generate one content ISS pair and return exactly 33 ordered null pairs.

    Candidate registered scores are deliberately not sampled.
    """

    if not isinstance(unit, ContentCalibrationUnit) or unit.split != CONTENT_CALIBRATION_CALIBRATION_SPLIT:
        raise TypeError("content calibration runtime requires a validated calibration unit")
    if not isinstance(assets, ContentCalibrationAssets):
        raise TypeError("content calibration runtime requires frozen assets")
    wrong_keys = derive_calibration_wrong_keys(calibration_key)
    if len(wrong_keys) != 16:
        raise RuntimeError("content calibration requires exactly 16 wrong keys")
    output = run_content_iss_evaluation_pair(
        pipeline,
        unit.prompt,
        calibration_key,
        assets.iss_assets,
        height=unit.height,
        width=unit.width,
        seed=unit.seed,
    )
    if not isinstance(output, ContentISSRunOutput):
        raise TypeError("content calibration requires a real content ISS pair result")
    pairs = [
        *(_blind_pair(output.image, wrong_key, assets) for wrong_key in wrong_keys),
        _blind_pair(output.primary_null, calibration_key, assets),
        *(_blind_pair(output.primary_null, wrong_key, assets) for wrong_key in wrong_keys),
    ]
    if len(pairs) != 33:
        raise RuntimeError("content calibration unit must yield exactly 33 score pairs")
    return tuple(pairs)


def blind_weighted_scores(
    image: Any,
    detection_key: bytes,
    wrong_keys: Sequence[bytes],
    assets: ContentCalibrationAssets,
    calibration_asset: WeightedJointAsset,
) -> dict[str, dict[str, float]]:
    """Score one ordinary image through unchanged LF/HF and frozen weighted-joint statistic."""

    if len(wrong_keys) != 16 or any(not isinstance(key, bytes) for key in wrong_keys):
        raise ValueError("content blind scoring requires exactly 16 wrong keys")
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


def run_content_chain_unit(
    pipeline: Any,
    unit: ContentChainUnit,
    detection_key: bytes,
    wrong_keys: Sequence[bytes],
    assets: ContentCalibrationAssets,
    calibration_asset: WeightedJointAsset,
) -> ContentChainOutput:
    """Run the content ISS pair and score both final images with frozen weighted-joint statistic."""

    if not isinstance(unit, ContentChainUnit):
        raise TypeError("content chain runtime requires a validated unit")
    if not isinstance(assets, ContentCalibrationAssets):
        raise TypeError("content chain runtime requires frozen content ISS assets")
    if not isinstance(calibration_asset, WeightedJointAsset):
        raise TypeError("content chain runtime requires the accepted calibration asset")
    if len(wrong_keys) != 16 or any(not isinstance(key, bytes) for key in wrong_keys):
        raise ValueError("content chain runtime requires exactly 16 wrong keys")
    output = run_content_iss_evaluation_pair(
        pipeline,
        unit.prompt,
        detection_key,
        assets.iss_assets,
        height=unit.height,
        width=unit.width,
        seed=unit.seed,
    )
    if not isinstance(output, ContentISSRunOutput):
        raise TypeError("content chain requires a real content ISS pair result")
    candidate_scores = blind_weighted_scores(
        output.image, detection_key, wrong_keys, assets, calibration_asset
    )
    primary_null_scores = blind_weighted_scores(
        output.primary_null, detection_key, wrong_keys, assets, calibration_asset
    )
    return ContentChainOutput(
        output.image,
        output.primary_null,
        output.measurement,
        candidate_scores,
        primary_null_scores,
    )


__all__ = [
    "ContentCalibrationAssets",
    "ContentChainOutput",
    "blind_weighted_scores",
    "derive_calibration_wrong_keys",
    "derive_stability_wrong_keys",
    "run_content_calibration_unit",
    "run_content_chain_unit",
]
