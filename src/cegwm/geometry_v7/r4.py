"""Geometry-V7 refined-R3 reliability and single-image R4 recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.geometry_v7.r2 import feature_row_from_geometry
from cegwm.geometry_v7.r3 import R3_B_LOW
from cegwm.geometry_v7.r3_advanced import predicted_h_regime
from cegwm.runtime.observation import require_ordinary_rgb_image


R4_TAU = 0.0
R4_R2_AREA_RATIO_THRESHOLD = 0.9843414071510957
R4_CYCLE_MAX_PX = 8.0
R4_ZOOM_MAX_ABS_ANGLE_DEG = 2.0
R4_ZOOM_MIN_SCALE = 1.15
R4_ZOOM_MAX_SCALE = 1.35
R4_ZOOM_MAX_TRANSLATION = 0.02
R4_ZOOM_MAX_PERSPECTIVE = 0.01
R4_ENGINEERING_REPLAY_RECORDED = "R4_ENGINEERING_REPLAY_RECORDED"
R4_OPERATIONAL_FAILURE = "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
R4_CLAIM_CEILING = "existing_observed_data_engineering_replay_only_no_r4_promotion"


@dataclass(frozen=True, slots=True)
class RefinedGate:
    valid: bool
    r2_selector_accepted: bool
    pure_rotation: bool
    zoom_out_like: bool
    cycle_score_px: float | None
    reliable: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class R4RuntimeRecord:
    route: str
    pre_score: float | None
    refined_gate: RefinedGate | None
    recovered: bool
    post_score: float | None
    positive: bool
    geometry: GeometryEstimate | None
    error: str | None = None


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("score must be a finite real scalar")
    return value


def score_route(score: object) -> str:
    try:
        value = _finite(score)
    except ValueError:
        return "INVALID_SCORE"
    if value > R4_TAU:
        return "DIRECT_POSITIVE"
    if value <= R3_B_LOW:
        return "DIRECT_NEGATIVE"
    return "BOUNDARY"


def _geometry_payload(geometry: GeometryEstimate) -> Mapping[str, object]:
    return {
        "status": geometry.status.value,
        "uncalibrated_sync_logit": geometry.uncalibrated_sync_logit,
        "observed_corners_in_canonical_normalized": (
            geometry.observed_corners_in_canonical_normalized
        ),
        "homography_observed_to_canonical": geometry.homography_observed_to_canonical,
        "legal": geometry.legal,
        "error": geometry.error,
    }


def refined_gate_from_features(
    *, boundary: bool, r2_selector_accepted: bool,
    regime_valid: bool, angle_degrees: object, scale: object,
    translation: object, perspective: object,
    pure_rotation: bool, cycle_score_px: object | None,
) -> RefinedGate:
    """Frozen predicate with no condition, truth, attack, or outcome input."""

    try:
        angle = _finite(angle_degrees)
        scale_value = _finite(scale)
        translation_value = _finite(translation)
        perspective_value = _finite(perspective)
        if not regime_valid:
            raise ValueError("predicted-H regime invalid")
        zoom = (
            abs(angle) <= R4_ZOOM_MAX_ABS_ANGLE_DEG
            and R4_ZOOM_MIN_SCALE <= scale_value <= R4_ZOOM_MAX_SCALE
            and translation_value <= R4_ZOOM_MAX_TRANSLATION
            and perspective_value <= R4_ZOOM_MAX_PERSPECTIVE
        )
        cycle = None
        if zoom:
            cycle = _finite(cycle_score_px)
        reliable = bool(
            boundary and r2_selector_accepted
            and (pure_rotation or (zoom and cycle is not None and cycle <= R4_CYCLE_MAX_PX))
        )
        return RefinedGate(True, bool(r2_selector_accepted), bool(pure_rotation),
                           zoom, cycle, reliable)
    except (TypeError, ValueError) as error:
        return RefinedGate(False, False, False, False, None, False, (str(error),))


def _prepare_geometry(geometry: GeometryEstimate) -> tuple[object, object]:
    if not isinstance(geometry, GeometryEstimate):
        raise TypeError("detect_geometry must return GeometryEstimate")
    feature = feature_row_from_geometry(
        split="runtime", condition_id="runtime", unit_id="runtime",
        geometry=_geometry_payload(geometry),
    )
    regime = predicted_h_regime(
        geometry.homography_observed_to_canonical,
        geometry_legal=geometry.legal,
        geometry_error=geometry.error,
    )
    return feature, regime


def detect_refine_and_recover(
    image_rgb: Image.Image,
    *,
    score_rgb: Callable[[Image.Image], float],
    detect_geometry: Callable[[Image.Image], GeometryEstimate],
    cycle_score_rgb: Callable[[Image.Image, object], float],
) -> R4RuntimeRecord:
    """Run one blind-image decision; geometry can only authorize one recovery."""

    geometry = None
    try:
        image = require_ordinary_rgb_image(image_rgb)
        pre = _finite(score_rgb(image))
        route = score_route(pre)
        if route == "DIRECT_POSITIVE":
            return R4RuntimeRecord(route, pre, None, False, None, True, None)
        if route == "DIRECT_NEGATIVE":
            return R4RuntimeRecord(route, pre, None, False, None, False, None)
        if route != "BOUNDARY":
            return R4RuntimeRecord(route, None, None, False, None, False, None,
                                   "invalid pre score")
        geometry = detect_geometry(image)
        feature, regime = _prepare_geometry(geometry)
        r2_accepted = bool(
            feature.mandatory_valid and feature.area_ratio is not None
            and feature.area_ratio >= R4_R2_AREA_RATIO_THRESHOLD
        )
        zoom_like = bool(
            regime.valid and regime.angle_degrees is not None and regime.scale is not None
            and regime.translation is not None and regime.perspective is not None
            and abs(regime.angle_degrees) <= R4_ZOOM_MAX_ABS_ANGLE_DEG
            and R4_ZOOM_MIN_SCALE <= regime.scale <= R4_ZOOM_MAX_SCALE
            and regime.translation <= R4_ZOOM_MAX_TRANSLATION
            and regime.perspective <= R4_ZOOM_MAX_PERSPECTIVE
        )
        cycle = cycle_score_rgb(image, geometry.homography_observed_to_canonical) if zoom_like else None
        gate = refined_gate_from_features(
            boundary=True, r2_selector_accepted=r2_accepted,
            regime_valid=regime.valid, angle_degrees=regime.angle_degrees,
            scale=regime.scale, translation=regime.translation,
            perspective=regime.perspective, pure_rotation=regime.pure_rotation_gate,
            cycle_score_px=cycle,
        )
        if not gate.reliable or geometry.homography_observed_to_canonical is None:
            return R4RuntimeRecord(route, pre, gate, False, None, False, geometry)
        recovered = rectify_attacked_rgb(image, geometry.homography_observed_to_canonical)
        post = _finite(score_rgb(recovered))
        return R4RuntimeRecord(route, pre, gate, True, post, post > R4_TAU, geometry)
    except Exception as error:
        return R4RuntimeRecord(
            "ERROR", None, None, False, None, False, geometry,
            f"{type(error).__name__}: {error}",
        )


__all__ = [name for name in globals() if name.startswith("R4_") or name in {
    "RefinedGate", "R4RuntimeRecord", "score_route", "refined_gate_from_features",
    "detect_refine_and_recover",
}]
