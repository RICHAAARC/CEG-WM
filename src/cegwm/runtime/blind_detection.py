"""BlindDetection-V1 production closure and development calibration runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from cegwm.method.blind_detection import (
    BLIND_DEV_DENOMINATOR,
    BlindCalibrationRoster,
    BlindCalibrationRow,
    BlindReplayRow,
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


def _validate_content_assets(content: Any, weighted: Any) -> None:
    if not isinstance(content, ContentCalibrationAssets):
        raise TypeError("blind detection requires frozen content public assets")
    if not isinstance(weighted, WeightedJointAsset):
        raise TypeError("blind detection requires the frozen weighted-joint asset")


@dataclass(frozen=True, slots=True)
class BlindProductionAssets:
    """Exact production boundary with the real frozen SyncSeal adapter type."""

    content_assets: ContentCalibrationAssets
    weighted_joint_asset: WeightedJointAsset
    geometry_backend: SyncSealTorchScript
    threshold_asset: BlindThresholdAsset | None = None

    def __post_init__(self) -> None:
        _validate_content_assets(self.content_assets, self.weighted_joint_asset)
        if type(self.geometry_backend) is not SyncSealTorchScript:
            raise TypeError("production Geometry backend must be SyncSealTorchScript")
        if self.threshold_asset is not None and not isinstance(
            self.threshold_asset, BlindThresholdAsset
        ):
            raise TypeError("blind threshold must be a BlindThresholdAsset")


@dataclass(frozen=True, slots=True)
class BlindTestAssets:
    """Explicit local-test injection boundary, never accepted by production API."""

    content_assets: ContentCalibrationAssets
    weighted_joint_asset: WeightedJointAsset
    geometry_backend: Any

    def __post_init__(self) -> None:
        _validate_content_assets(self.content_assets, self.weighted_joint_asset)
        if not callable(getattr(self.geometry_backend, "detect_geometry", None)):
            raise TypeError("test Geometry backend must expose detect_geometry")


@dataclass(frozen=True, slots=True)
class BlindEmbeddingAssets:
    """Injected existing content embed plus strongly typed final-RGB SyncSeal."""

    content_backend: Any
    syncseal_backend: SyncSealTorchScript
    residual_strength_multiplier: float

    def __post_init__(self) -> None:
        if not callable(getattr(self.content_backend, "embed_content", None)):
            raise TypeError("content backend must expose the existing content embed")
        if type(self.syncseal_backend) is not SyncSealTorchScript:
            raise TypeError("embedding SyncSeal backend must be SyncSealTorchScript")
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
    method_complete: bool
    operational_error: str | None
    scorer_id: str = BLIND_SCORER_ID
    preprocess_id: str = BLIND_PREPROCESS_ID
    same_scoring_context: bool = True


BlindScoringAssets = BlindProductionAssets | BlindTestAssets


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _score_current_rgb(
    current_rgb: Image.Image, detection_key: bytes, assets: BlindScoringAssets
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


def _raw_h(geometry: GeometryEstimate) -> tuple[tuple[float, float, float], ...]:
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


def _geometry_disposition(geometry: GeometryEstimate) -> tuple[str, str | None]:
    """Classify the typed Geometry result before inspecting any raw H."""

    if geometry.status is GeometryStatus.ERROR:
        detail = geometry.error or "GeometryStatus.ERROR without explanatory error"
        return "OPERATIONAL", f"geometry_runtime:{detail}"
    if geometry.status is GeometryStatus.UNSUPPORTED:
        if (
            geometry.legal is not False
            or geometry.basic_observable is not False
            or geometry.homography_observed_to_canonical is not None
            or not isinstance(geometry.error, str)
            or not geometry.error
        ):
            return (
                "OPERATIONAL",
                "geometry_runtime:UNSUPPORTED invariant violation: expected "
                "legal=False, basic_observable=False, H=None, explanatory error",
            )
        return "INVALID_H", None
    if geometry.status is GeometryStatus.RELIABLE:
        if geometry.error is not None:
            return "OPERATIONAL", f"geometry_runtime:RELIABLE carried error: {geometry.error}"
        if geometry.legal is not True or geometry.basic_observable is not True:
            return (
                "OPERATIONAL",
                "geometry_runtime:RELIABLE invariant violation: expected "
                "legal=True, basic_observable=True",
            )
        return "RAW_H", None
    if geometry.status is GeometryStatus.UNRELIABLE:
        if geometry.error is not None:
            return "OPERATIONAL", f"geometry_runtime:UNRELIABLE carried error: {geometry.error}"
        if geometry.legal is not True or geometry.basic_observable is not True:
            return (
                "OPERATIONAL",
                "geometry_runtime:UNRELIABLE invariant violation: expected "
                "legal=True, basic_observable=True",
            )
        return "RAW_H", None
    return "OPERATIONAL", f"geometry_runtime:unknown GeometryStatus: {geometry.status!r}"


def _record(
    route: str,
    positive: bool,
    pre: BlindStatistic | None,
    post: BlindStatistic | None,
    recovered: bool,
    geometry: GeometryEstimate | None,
    tau: float,
    digest: str,
    *,
    method_complete: bool,
    operational_error: str | None = None,
) -> BlindDetectionRecord:
    return BlindDetectionRecord(
        route, positive, pre, post, recovered, geometry, tau, digest,
        method_complete, operational_error,
    )


def _detect_core(
    image: Any,
    key: str | bytes | bytearray | memoryview,
    assets: BlindScoringAssets,
    tau_blind: float,
) -> BlindDetectionRecord:
    """Shared core reached only through production or explicit test/calibration gates."""

    detection_key = normalize_detection_key(key)
    digest = public_key_digest(detection_key)
    tau = _finite(tau_blind, "tau_blind")
    try:
        current = require_ordinary_rgb_image(image)
        pre = _score_current_rgb(current, detection_key, assets)
    except Exception as error:
        return _record(
            "ERROR_FAIL_CLOSED", False, None, None, False, None, tau, digest,
            method_complete=False,
            operational_error=f"content_pre:{type(error).__name__}: {error}",
        )
    if pre.value > tau:
        return _record(
            "DIRECT_POSITIVE", True, pre, None, False, None, tau, digest,
            method_complete=True,
        )
    try:
        geometry = assets.geometry_backend.detect_geometry(current)
    except Exception as error:
        return _record(
            "ERROR_FAIL_CLOSED", False, pre, None, False, None, tau, digest,
            method_complete=False,
            operational_error=f"geometry_runtime:{type(error).__name__}: {error}",
        )
    if not isinstance(geometry, GeometryEstimate):
        return _record(
            "ERROR_FAIL_CLOSED", False, pre, None, False, None, tau, digest,
            method_complete=False,
            operational_error="geometry_runtime:TypeError: detector must return GeometryEstimate",
        )
    disposition, geometry_error = _geometry_disposition(geometry)
    if disposition == "OPERATIONAL":
        return _record(
            "ERROR_FAIL_CLOSED", False, pre, None, False, geometry, tau, digest,
            method_complete=False,
            operational_error=geometry_error,
        )
    if disposition == "INVALID_H":
        return _record(
            "GEOMETRY_FAIL_CLOSED", False, pre, None, False, geometry, tau, digest,
            method_complete=True,
        )
    try:
        matrix = _raw_h(geometry)
    except LookupError:
        return _record(
            "GEOMETRY_NO_H", False, pre, None, False, geometry, tau, digest,
            method_complete=True,
        )
    except (TypeError, ValueError):
        return _record(
            "GEOMETRY_FAIL_CLOSED", False, pre, None, False, geometry, tau, digest,
            method_complete=True,
        )
    try:
        recovered_rgb = rectify_attacked_rgb(current, matrix)
    except Exception:
        return _record(
            "RECTIFICATION_FAIL_CLOSED", False, pre, None, False, geometry, tau, digest,
            method_complete=True,
        )
    try:
        post = _score_current_rgb(recovered_rgb, detection_key, assets)
    except Exception as error:
        return _record(
            "ERROR_FAIL_CLOSED", False, pre, None, True, geometry, tau, digest,
            method_complete=False,
            operational_error=f"content_post:{type(error).__name__}: {error}",
        )
    return _record(
        "GEOMETRY_RECOVERED", post.value > tau, pre, post, True, geometry, tau, digest,
        method_complete=True,
    )


def detect_watermark(
    image: Any,
    key: str | bytes | bytearray | memoryview,
    assets: BlindProductionAssets,
) -> BlindDetectionRecord:
    """Production single-image detection with an exact-bound SyncSeal backend."""

    if type(assets) is not BlindProductionAssets:
        raise TypeError("production detect_watermark requires BlindProductionAssets")
    threshold = assets.threshold_asset
    if threshold is None:
        raise ThresholdUnavailableError(
            "BlindDetection-V1 has no frozen N_dev=256 threshold asset; production detection refused"
        )
    if threshold.test_only:
        raise ThresholdUnavailableError("production detection rejects a test-only threshold asset")
    if threshold.payload["calibration_key_digest"] != public_key_digest(key):
        raise ValueError("production threshold detection-key identity differs")
    return _detect_core(image, key, assets, threshold.tau_blind)


def detect_watermark_test_only(
    image: Any,
    key: str | bytes | bytearray | memoryview,
    assets: BlindTestAssets,
    threshold: BlindThresholdAsset,
) -> BlindDetectionRecord:
    """Explicit local-test entry; production never accepts this asset/backend pair."""

    if not isinstance(assets, BlindTestAssets):
        raise TypeError("test-only detection requires BlindTestAssets")
    if not isinstance(threshold, BlindThresholdAsset) or not threshold.test_only:
        raise TypeError("test-only detection requires an explicit test-only threshold")
    return _detect_core(image, key, assets, threshold.tau_blind)


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
    assets: BlindScoringAssets,
) -> BlindCalibrationRow:
    try:
        current = require_ordinary_rgb_image(image)
        pre = _score_current_rgb(current, detection_key, assets).value
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, None, None,
            "GEOMETRY_ERROR", False, f"content_pre:{type(error).__name__}: {error}",
        )
    try:
        geometry = assets.geometry_backend.detect_geometry(current)
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "GEOMETRY_ERROR", False, f"geometry_runtime:{type(error).__name__}: {error}",
        )
    if not isinstance(geometry, GeometryEstimate):
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "GEOMETRY_ERROR", False,
            "geometry_runtime:TypeError: detector must return GeometryEstimate",
        )
    disposition, geometry_error = _geometry_disposition(geometry)
    if disposition == "OPERATIONAL":
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "GEOMETRY_ERROR", False, geometry_error,
        )
    if disposition == "INVALID_H":
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "INVALID_H", True, None,
        )
    try:
        matrix = _raw_h(geometry)
    except LookupError:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "NO_H", True, None,
        )
    except (TypeError, ValueError):
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "INVALID_H", True, None,
        )
    try:
        recovered = rectify_attacked_rgb(current, matrix)
    except Exception:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "RECTIFICATION_ERROR", True, None,
        )
    try:
        post = _score_current_rgb(recovered, detection_key, assets).value
    except Exception as error:
        return BlindCalibrationRow(
            roster_index, unit_id, source_stratum, pre, None,
            "RECOVERED", False, f"content_post:{type(error).__name__}: {error}",
        )
    return BlindCalibrationRow(
        roster_index, unit_id, source_stratum, pre, post,
        "RECOVERED", True, None,
    )


def _frozen_units(roster: BlindCalibrationRoster) -> tuple[Any, ...]:
    if not isinstance(roster, BlindCalibrationRoster):
        raise TypeError("development calibration requires a frozen roster")
    units = tuple(roster.units)
    if len(units) != BLIND_DEV_DENOMINATOR:
        raise RuntimeError("development denominator changed after roster validation")
    return units


def run_development_calibration(
    roster: BlindCalibrationRoster,
    key: str | bytes | bytearray | memoryview,
    assets: BlindScoringAssets,
    image_loader: Callable[[str], Any],
) -> tuple[BlindCalibrationRow, ...]:
    """First pass: score and attempt Geometry once for every frozen unit."""

    if type(assets) not in (BlindProductionAssets, BlindTestAssets):
        raise TypeError("development calibration assets differ")
    if not callable(image_loader):
        raise TypeError("development calibration requires an image loader")
    detection_key = normalize_detection_key(key)
    rows: list[BlindCalibrationRow] = []
    for index, unit in enumerate(_frozen_units(roster)):
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


def run_development_full_system_replay(
    roster: BlindCalibrationRoster,
    key: str | bytes | bytearray | memoryview,
    assets: BlindScoringAssets,
    image_loader: Callable[[str], Any],
    tau_blind: float,
) -> tuple[BlindReplayRow, ...]:
    """Fresh second pass through the complete pre/route/Geometry/post system."""

    if type(assets) not in (BlindProductionAssets, BlindTestAssets):
        raise TypeError("development replay assets differ")
    if not callable(image_loader):
        raise TypeError("development replay requires an image loader")
    rows: list[BlindReplayRow] = []
    for index, unit in enumerate(_frozen_units(roster)):
        try:
            current = require_ordinary_rgb_image(image_loader(unit.image_ref))
        except Exception as error:
            rows.append(
                BlindReplayRow(
                    index, unit.unit_id, unit.source_stratum, None, None,
                    "ERROR_FAIL_CLOSED", False, False, False,
                    f"image_io:{type(error).__name__}: {error}",
                )
            )
            continue
        record = _detect_core(current, key, assets, tau_blind)
        rows.append(
            BlindReplayRow(
                index, unit.unit_id, unit.source_stratum,
                None if record.pre is None else record.pre.value,
                None if record.post is None else record.post.value,
                record.route, record.positive, record.recovered,
                record.method_complete, record.operational_error,
            )
        )
    return tuple(rows)


__all__ = [
    "BLIND_PREPROCESS_ID", "BLIND_SCORER_ID", "BlindDetectionRecord",
    "BlindEmbeddingAssets", "BlindProductionAssets", "BlindTestAssets",
    "ThresholdUnavailableError", "detect_watermark", "detect_watermark_test_only",
    "embed_watermark", "run_development_calibration",
    "run_development_full_system_replay",
]
