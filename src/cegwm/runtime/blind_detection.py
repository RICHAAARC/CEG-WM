"""BlindDetection-V1 production closure and development calibration runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.method.blind_detection import (
    BLIND_DEV_DENOMINATOR,
    BlindCalibrationRoster,
    BlindCalibrationRow,
    BlindStatistic,
    BlindThresholdAsset,
    statistic_from_weighted_scores,
)
from cegwm.method.content_weighted_joint import WeightedJointAsset
from cegwm.runtime.content_weighted_joint_sd35 import (
    ContentCalibrationAssets,
    blind_weighted_scores,
    derive_stability_wrong_keys,
)
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key, public_key_digest


BLIND_SCORER_ID = "content_v9_weighted_joint_registered_minus_exact16_wrong_max_v1"
BLIND_PREPROCESS_ID = "ordinary_rgb_then_frozen_content_v9_branch_preprocessing_v1"


class ThresholdUnavailableError(RuntimeError):
    """Raised before scoring when the production N_dev=256 asset is absent."""


@dataclass(frozen=True, slots=True)
class BlindPublicAssets:
    """Only public scoring, Geometry, and optional threshold state.

    ``geometry_backend`` must expose ``detect_geometry(current_rgb)``.  The
    official production object is ``SyncSealTorchScript``; tests may inject a
    local method double without loading a model.
    """

    content_assets: ContentCalibrationAssets
    weighted_joint_asset: WeightedJointAsset
    geometry_backend: Any
    threshold_asset: BlindThresholdAsset | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_assets, ContentCalibrationAssets):
            raise TypeError("blind detection requires frozen content public assets")
        if not isinstance(self.weighted_joint_asset, WeightedJointAsset):
            raise TypeError("blind detection requires the frozen weighted-joint asset")
        if not callable(getattr(self.geometry_backend, "detect_geometry", None)):
            raise TypeError("blind detection requires a public Geometry detector")
        if self.threshold_asset is not None and not isinstance(
            self.threshold_asset, BlindThresholdAsset
        ):
            raise TypeError("blind threshold must be a BlindThresholdAsset")


@dataclass(frozen=True, slots=True)
class BlindEmbeddingAssets:
    """Injected real embedding adapters; content always runs before SyncSeal."""

    content_backend: Any
    syncseal_backend: Any
    residual_strength_multiplier: float

    def __post_init__(self) -> None:
        if not callable(getattr(self.content_backend, "embed_content", None)):
            raise TypeError("content backend must expose the existing content embed")
        if not callable(getattr(self.syncseal_backend, "embed_final_rgb", None)):
            raise TypeError("SyncSeal backend must expose final-RGB embedding")
        value = self.residual_strength_multiplier
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("SyncSeal residual multiplier must be real")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("SyncSeal residual multiplier must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class BlindDetectionRecord:
    route: str
    positive: bool
    pre: BlindStatistic | None
    post: BlindStatistic | None
    recovered: bool
    geometry: GeometryEstimate | None
    tau_blind: float
    key_digest: str
    scorer_id: str = BLIND_SCORER_ID
    preprocess_id: str = BLIND_PREPROCESS_ID
    same_scoring_context: bool = True
    error: str | None = None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _score_current_rgb(
    current_rgb: Image.Image, detection_key: bytes, assets: BlindPublicAssets
) -> BlindStatistic:
    image = require_ordinary_rgb_image(current_rgb)
    wrong_keys = derive_stability_wrong_keys(detection_key)
    if len(wrong_keys) != 16:
        raise RuntimeError("blind scoring wrong-key derivation must return exact 16")
    branches = blind_weighted_scores(
        image,
        detection_key,
        wrong_keys,
        assets.content_assets,
        assets.weighted_joint_asset,
    )
    if not isinstance(branches, dict) or set(branches) != {"lf", "hf", "weighted_joint"}:
        raise TypeError("frozen content scorer returned an invalid branch record")
    return statistic_from_weighted_scores(branches["weighted_joint"])


def _raw_h(geometry: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(geometry, GeometryEstimate):
        raise TypeError("Geometry detector must return GeometryEstimate")
    if geometry.error is not None:
        raise RuntimeError(f"Geometry detector error: {geometry.error}")
    if geometry.legal is not True:
        raise ValueError("Geometry returned an invalid raw H")
    matrix = geometry.homography_observed_to_canonical
    if matrix is None:
        raise LookupError("Geometry returned no raw H")
    if not isinstance(matrix, (tuple, list)) or len(matrix) != 3:
        raise ValueError("Geometry raw H must be finite invertible 3x3")
    try:
        parsed = tuple(tuple(_finite(value, "raw_H") for value in row) for row in matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("Geometry raw H must be finite invertible 3x3") from error
    if any(len(row) != 3 for row in parsed):
        raise ValueError("Geometry raw H must be finite invertible 3x3")
    determinant = (
        parsed[0][0] * (parsed[1][1] * parsed[2][2] - parsed[1][2] * parsed[2][1])
        - parsed[0][1] * (parsed[1][0] * parsed[2][2] - parsed[1][2] * parsed[2][0])
        + parsed[0][2] * (parsed[1][0] * parsed[2][1] - parsed[1][1] * parsed[2][0])
    )
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("Geometry raw H must be finite invertible 3x3")
    return parsed


def detect_watermark(
    image: Any,
    key: str | bytes | bytearray | memoryview,
    assets: BlindPublicAssets,
) -> BlindDetectionRecord:
    """Detect from exactly one current ordinary RGB, one key, and public assets.

    Every finite non-positive pre-threshold result enters Geometry-Direct once.
    Geometry can provide coordinates but can never create a positive; the same
    internal content scorer and threshold decide both observations.
    """

    if not isinstance(assets, BlindPublicAssets):
        raise TypeError("detect_watermark requires BlindPublicAssets")
    if assets.threshold_asset is None:
        raise ThresholdUnavailableError(
            "BlindDetection-V1 has no frozen N_dev=256 threshold asset; production detection refused"
        )
    detection_key = normalize_detection_key(key)
    digest = public_key_digest(detection_key)
    tau = assets.threshold_asset.tau_blind
    geometry = None
    pre = None
    recovered = False
    try:
        current = require_ordinary_rgb_image(image)
        pre = _score_current_rgb(current, detection_key, assets)
        if pre.value > tau:
            return BlindDetectionRecord(
                "DIRECT_POSITIVE", True, pre, None, False, None, tau, digest
            )
        geometry = assets.geometry_backend.detect_geometry(current)
        try:
            matrix = _raw_h(geometry)
        except LookupError as error:
            return BlindDetectionRecord(
                "GEOMETRY_NO_H", False, pre, None, False, geometry, tau, digest,
                error=str(error),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            return BlindDetectionRecord(
                "GEOMETRY_FAIL_CLOSED", False, pre, None, False, geometry, tau, digest,
                error=f"{type(error).__name__}: {error}",
            )
        recovered_rgb = rectify_attacked_rgb(current, matrix)
        recovered = True
        post = _score_current_rgb(recovered_rgb, detection_key, assets)
        return BlindDetectionRecord(
            "GEOMETRY_RECOVERED", post.value > tau, pre, post, True, geometry, tau, digest
        )
    except Exception as error:
        return BlindDetectionRecord(
            "ERROR_FAIL_CLOSED", False, pre, None, recovered, geometry, tau, digest,
            error=f"{type(error).__name__}: {error}",
        )


def embed_watermark(
    content_request: Any,
    key: str | bytes | bytearray | memoryview,
    assets: BlindEmbeddingAssets,
) -> Image.Image:
    """Run the existing content embed, then frozen final-RGB SyncSeal exactly once."""

    if not isinstance(assets, BlindEmbeddingAssets):
        raise TypeError("embed_watermark requires BlindEmbeddingAssets")
    detection_key = normalize_detection_key(key)
    content_rgb = assets.content_backend.embed_content(content_request, detection_key)
    final_content_rgb = require_ordinary_rgb_image(content_rgb)
    synced = assets.syncseal_backend.embed_final_rgb(
        final_content_rgb, float(assets.residual_strength_multiplier)
    )
    return require_ordinary_rgb_image(synced)


def _calibration_row(
    *,
    roster_index: int,
    unit_id: str,
    source_stratum: str,
    image: Any,
    detection_key: bytes,
    assets: BlindPublicAssets,
) -> BlindCalibrationRow:
    try:
        current = require_ordinary_rgb_image(image)
        pre = _score_current_rgb(current, detection_key, assets).value
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, None, None, "GEOMETRY_ERROR", False,
            f"content_pre:{type(error).__name__}: {error}",
        )
    try:
        geometry = assets.geometry_backend.detect_geometry(current)
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "GEOMETRY_ERROR", False,
            f"geometry_model:{type(error).__name__}: {error}",
        )
    if isinstance(geometry, GeometryEstimate) and geometry.error is not None:
        normalized_error = geometry.error.lower()
        if any(
            marker in normalized_error
            for marker in ("cuda", "gpu", "out of memory", "checkpoint", "no such file", "i/o")
        ):
            return BlindCalibrationRow(
                roster_index, unit_id, source_stratum, pre, None, "GEOMETRY_ERROR", False,
                f"geometry_model:{geometry.error}",
            )
    try:
        matrix = _raw_h(geometry)
    except LookupError:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "NO_H", True, None
        )
    except RuntimeError:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "GEOMETRY_ERROR", True, None
        )
    except (TypeError, ValueError):
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "INVALID_H", True, None
        )
    try:
        recovered = rectify_attacked_rgb(current, matrix)
    except Exception:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "RECTIFICATION_ERROR", True, None
        )
    try:
        post = _score_current_rgb(recovered, detection_key, assets).value
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None, "RECOVERED", False,
            f"content_post:{type(error).__name__}: {error}",
        )
    return BlindCalibrationRow(
        roster_index, unit_id, source_stratum, pre, post, "RECOVERED", True, None
    )


def run_development_calibration(
    roster: BlindCalibrationRoster,
    key: str | bytes | bytearray | memoryview,
    assets: BlindPublicAssets,
    image_loader: Callable[[str], Any],
) -> tuple[BlindCalibrationRow, ...]:
    """Attempt the frozen N_dev=256 roster once, retaining every row.

    This function prepares the authorized entrypoint but performs work only
    when explicitly called by a future execution.  The roster tuple is fully
    validated/frozen before the first image is loaded or scored.
    """

    if not isinstance(roster, BlindCalibrationRoster):
        raise TypeError("development calibration requires a frozen roster")
    if not isinstance(assets, BlindPublicAssets):
        raise TypeError("development calibration requires BlindPublicAssets")
    if not callable(image_loader):
        raise TypeError("development calibration requires an image loader")
    detection_key = normalize_detection_key(key)
    frozen_units = tuple(roster.units)
    if len(frozen_units) != BLIND_DEV_DENOMINATOR:
        raise RuntimeError("development denominator changed after roster validation")
    rows: list[BlindCalibrationRow] = []
    for index, unit in enumerate(frozen_units):
        try:
            image = image_loader(unit.image_ref)
        except Exception as error:
            rows.append(
                BlindCalibrationRow(
                    index, unit.unit_id, unit.source_stratum, None, None,
                    "GEOMETRY_ERROR", False,
                    f"image_io:{type(error).__name__}: {error}",
                )
            )
            continue
        rows.append(
            _calibration_row(
                roster_index=index,
                unit_id=unit.unit_id,
                source_stratum=unit.source_stratum,
                image=image,
                detection_key=detection_key,
                assets=assets,
            )
        )
    return tuple(rows)


__all__ = [
    "BLIND_PREPROCESS_ID", "BLIND_SCORER_ID", "BlindDetectionRecord",
    "BlindEmbeddingAssets", "BlindPublicAssets", "ThresholdUnavailableError",
    "detect_watermark", "embed_watermark", "run_development_calibration",
]
