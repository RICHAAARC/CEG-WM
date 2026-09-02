"""Frozen Geometry-V7 R1A attack, truth, and fixed-denominator contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from cegwm.geometry_v7.contracts import (
    CANONICAL_CORNERS_NORMALIZED,
    GeometryEstimate,
    GeometryStatus,
    Matrix3x3,
)
from cegwm.runtime.observation import require_ordinary_rgb_image


R1A_FIXED_UNIT_COUNT = 8
R1A_DELTA_NONTRIVIAL_NORMALIZED = 2.0 / 255.0
R1A_SANITY_MAX_CORNER_ERROR_NORMALIZED = 2.0 / 255.0
R1A_MIN_IMPROVED_FRACTION = 0.75
R1A_MIN_IMPROVED_COUNT = math.ceil(
    R1A_MIN_IMPROVED_FRACTION * R1A_FIXED_UNIT_COUNT
)
R1A_ATTACK_SPEC_REQUEST_CHANGES = "ATTACK_SPEC_REQUEST_CHANGES"
R1A_BLOCKING_METHOD_CANARY_PASSED = "R1A_BLOCKING_METHOD_CANARY_PASSED"
R1A_BLOCKING_METHOD_CANARY_FAILED = "R1A_BLOCKING_METHOD_CANARY_FAILED"


class R1AConditionKind(str, Enum):
    SANITY = "sanity_control"
    CORE = "core_nonidentity"


@dataclass(frozen=True, slots=True)
class R1AConditionSpec:
    condition_id: str
    kind: R1AConditionKind
    forward_canonical_to_observed: Matrix3x3
    truth_observed_to_canonical: Matrix3x3
    resize_control_size: int | None = None


@dataclass(frozen=True, slots=True)
class R1ATruthPreflightEntry:
    condition_id: str
    identity_baseline_rmse: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class R1ATruthPreflight:
    entries: tuple[R1ATruthPreflightEntry, ...]
    passed: bool
    status: str


@dataclass(frozen=True, slots=True)
class R1AUnitRecord:
    unit_id: str
    condition_id: str
    condition_kind: R1AConditionKind
    truth_observed_corners_in_canonical_normalized: tuple[
        tuple[float, float], ...
    ]
    identity_baseline_rmse: float
    truth_eligible: bool
    attacked_image: Image.Image | None
    geometry: GeometryEstimate | None
    prediction_rmse: float | None
    paired_delta: float | None
    improved: bool | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R1AConditionRecords:
    spec: R1AConditionSpec
    records: tuple[R1AUnitRecord, ...]


@dataclass(frozen=True, slots=True)
class R1AConditionAggregate:
    condition_id: str
    condition_kind: R1AConditionKind
    roster: tuple[str, ...]
    denominator: int
    truth_eligible_count: int
    valid_prediction_count: int
    improved_count: int
    paired_delta_median: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class R1AEvaluation:
    status: str
    truth_preflight: R1ATruthPreflight
    aggregates: tuple[R1AConditionAggregate, ...]
    all_sanity_passed: bool
    all_core_passed: bool
    blocking_method_canary_passed: bool


GeometryDetector = Callable[[Image.Image], GeometryEstimate]


def _matrix_multiply(left: Matrix3x3, right: Matrix3x3) -> Matrix3x3:
    return tuple(
        tuple(
            math.fsum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_inverse(matrix: Matrix3x3) -> Matrix3x3:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactors = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    determinant = a * cofactors[0][0] + b * cofactors[1][0] + c * cofactors[2][0]
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("R1A transform must be finite and invertible")
    inverse = tuple(
        tuple(value / determinant for value in row) for row in cofactors
    )
    if any(not math.isfinite(value) for row in inverse for value in row):
        raise ValueError("R1A inverse transform must be finite")
    return inverse


def _rotation_forward(degrees: float) -> Matrix3x3:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    # Image y increases down.  Positive display-space CCW moves (1,0) upward.
    return (
        (cosine, sine, 0.0),
        (-sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )


def _scale_forward(scale: float) -> Matrix3x3:
    return ((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, 1.0))


def _translation_forward(dx_pixels: int, dy_pixels: int) -> Matrix3x3:
    return (
        (1.0, 0.0, 2.0 * dx_pixels / 511.0),
        (0.0, 1.0, 2.0 * dy_pixels / 511.0),
        (0.0, 0.0, 1.0),
    )


def _condition(
    condition_id: str,
    kind: R1AConditionKind,
    forward: Matrix3x3,
    *,
    resize_control_size: int | None = None,
    truth: Matrix3x3 | None = None,
) -> R1AConditionSpec:
    return R1AConditionSpec(
        condition_id,
        kind,
        forward,
        _matrix_inverse(forward) if truth is None else truth,
        resize_control_size,
    )


_IDENTITY_MATRIX: Matrix3x3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_OFFSET_CROP_TRUTH: Matrix3x3 = (
    (0.75, 0.0, -0.125),
    (0.0, 0.75, 0.125),
    (0.0, 0.0, 1.0),
)
_COMPOSITE_FORWARD = _matrix_multiply(
    # C_0.85 is a centered 85% crop rescaled to the fixed canvas.
    _scale_forward(1.0 / 0.85),
    _matrix_multiply(
        _translation_forward(16, -16),
        _rotation_forward(10.0),
    ),
)

R1A_SANITY_CONDITIONS = (
    _condition("sanity_identity", R1AConditionKind.SANITY, _IDENTITY_MATRIX),
    _condition(
        "sanity_resize_384_then_512",
        R1AConditionKind.SANITY,
        _IDENTITY_MATRIX,
        resize_control_size=384,
    ),
    _condition(
        "sanity_resize_768_then_512",
        R1AConditionKind.SANITY,
        _IDENTITY_MATRIX,
        resize_control_size=768,
    ),
)
R1A_CORE_CONDITIONS = (
    _condition(
        "core_rotation_neg15",
        R1AConditionKind.CORE,
        _rotation_forward(-15.0),
    ),
    _condition(
        "core_rotation_pos15",
        R1AConditionKind.CORE,
        _rotation_forward(15.0),
    ),
    _condition(
        "core_fixed_canvas_zoom_0_8",
        R1AConditionKind.CORE,
        _scale_forward(0.8),
    ),
    _condition(
        "core_fixed_canvas_zoom_1_2",
        R1AConditionKind.CORE,
        _scale_forward(1.2),
    ),
    _condition(
        "core_translation_pos32_x",
        R1AConditionKind.CORE,
        _translation_forward(32, 0),
    ),
    _condition(
        "core_translation_neg32_x",
        R1AConditionKind.CORE,
        _translation_forward(-32, 0),
    ),
    _condition(
        "core_translation_pos32_y",
        R1AConditionKind.CORE,
        _translation_forward(0, 32),
    ),
    _condition(
        "core_translation_neg32_y",
        R1AConditionKind.CORE,
        _translation_forward(0, -32),
    ),
    _condition(
        "core_offset_crop_rescale",
        R1AConditionKind.CORE,
        _matrix_inverse(_OFFSET_CROP_TRUTH),
        truth=_OFFSET_CROP_TRUTH,
    ),
    _condition(
        "core_composite_c0_85_t16_neg16_r10",
        R1AConditionKind.CORE,
        _COMPOSITE_FORWARD,
    ),
)
R1A_ALL_CONDITIONS = (*R1A_SANITY_CONDITIONS, *R1A_CORE_CONDITIONS)


def apply_homography(
    matrix: Matrix3x3,
    points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    mapped: list[tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("R1A points must be 2D")
        x, y = float(point[0]), float(point[1])
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if not math.isfinite(denominator) or denominator == 0.0:
            raise ValueError("R1A homography point is not finite")
        result = (
            (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2])
            / denominator,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2])
            / denominator,
        )
        if any(not math.isfinite(value) for value in result):
            raise ValueError("R1A homography point is not finite")
        mapped.append(result)
    return tuple(mapped)


def corner_rmse(
    predicted: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
) -> float:
    predicted_points = tuple(tuple(point) for point in predicted)
    truth_points = tuple(tuple(point) for point in truth)
    if (
        len(predicted_points) != 4
        or len(truth_points) != 4
        or any(len(point) != 2 for point in (*predicted_points, *truth_points))
    ):
        raise ValueError("R1A corner RMSE requires two 4x2 point sets")
    squared = []
    for predicted_point, truth_point in zip(
        predicted_points, truth_points, strict=True
    ):
        for predicted_value, truth_value in zip(
            predicted_point, truth_point, strict=True
        ):
            predicted_scalar = float(predicted_value)
            truth_scalar = float(truth_value)
            if not math.isfinite(predicted_scalar) or not math.isfinite(truth_scalar):
                raise ValueError("R1A corner RMSE requires finite points")
            squared.append((predicted_scalar - truth_scalar) ** 2)
    return math.sqrt(math.fsum(squared) / 8.0)


def truth_correspondences(
    spec: R1AConditionSpec,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(spec, R1AConditionSpec):
        raise TypeError("R1A truth requires a frozen condition")
    return apply_homography(
        spec.truth_observed_to_canonical,
        CANONICAL_CORNERS_NORMALIZED,
    )


def r1a_truth_preflight(
    conditions: Sequence[R1AConditionSpec] = R1A_CORE_CONDITIONS,
) -> R1ATruthPreflight:
    supplied = tuple(conditions)
    if supplied != R1A_CORE_CONDITIONS:
        raise ValueError("R1A truth preflight requires the exact ten core conditions")
    entries = tuple(
        R1ATruthPreflightEntry(
            spec.condition_id,
            corner_rmse(CANONICAL_CORNERS_NORMALIZED, truth_correspondences(spec)),
            corner_rmse(CANONICAL_CORNERS_NORMALIZED, truth_correspondences(spec))
            > R1A_DELTA_NONTRIVIAL_NORMALIZED,
        )
        for spec in supplied
    )
    passed = all(entry.eligible for entry in entries)
    return R1ATruthPreflight(
        entries,
        passed,
        "TRUTH_PREFLIGHT_PASSED" if passed else R1A_ATTACK_SPEC_REQUEST_CHANGES,
    )


def _pixel_output_to_source(
    truth_observed_to_canonical: Matrix3x3,
) -> tuple[float, ...]:
    pixel_to_normalized: Matrix3x3 = (
        (2.0 / 511.0, 0.0, -1.0),
        (0.0, 2.0 / 511.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    normalized_to_pixel: Matrix3x3 = (
        (511.0 / 2.0, 0.0, 511.0 / 2.0),
        (0.0, 511.0 / 2.0, 511.0 / 2.0),
        (0.0, 0.0, 1.0),
    )
    mapping = _matrix_multiply(
        normalized_to_pixel,
        _matrix_multiply(truth_observed_to_canonical, pixel_to_normalized),
    )
    scale = mapping[2][2]
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("R1A pixel mapping must be finite")
    normalized = tuple(tuple(value / scale for value in row) for row in mapping)
    return (
        normalized[0][0],
        normalized[0][1],
        normalized[0][2],
        normalized[1][0],
        normalized[1][1],
        normalized[1][2],
        normalized[2][0],
        normalized[2][1],
    )


def render_r1a_attack(image: Any, spec: R1AConditionSpec) -> Image.Image:
    """Render one frozen condition; every core condition resamples exactly once."""

    source = require_ordinary_rgb_image(image)
    if source.size != (512, 512):
        raise ValueError("R1A renderer requires ordinary 512x512 RGB")
    if not isinstance(spec, R1AConditionSpec) or spec not in R1A_ALL_CONDITIONS:
        raise ValueError("R1A renderer requires one exact frozen condition")
    if spec.resize_control_size is not None:
        intermediate = source.resize(
            (spec.resize_control_size, spec.resize_control_size),
            resample=Image.Resampling.BILINEAR,
        )
        return intermediate.resize((512, 512), resample=Image.Resampling.BILINEAR)
    if spec.condition_id == "sanity_identity":
        return source.copy()
    return source.transform(
        (512, 512),
        Image.Transform.PERSPECTIVE,
        _pixel_output_to_source(spec.truth_observed_to_canonical),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def detect_attacked_rgb(
    detector: GeometryDetector,
    attacked_rgb: Image.Image,
) -> GeometryEstimate:
    """Invoke the detector with exactly one attacked-RGB argument."""

    if not callable(detector):
        raise TypeError("R1A detector must be callable")
    result = detector(attacked_rgb)
    if not isinstance(result, GeometryEstimate):
        raise TypeError("R1A detector must return GeometryEstimate")
    return result


def _strict_convex(corners: tuple[tuple[float, float], ...]) -> bool:
    crosses = []
    for index in range(4):
        current = corners[index]
        following = corners[(index + 1) % 4]
        after_following = corners[(index + 2) % 4]
        first = (following[0] - current[0], following[1] - current[1])
        second = (
            after_following[0] - following[0],
            after_following[1] - following[1],
        )
        crosses.append(first[0] * second[1] - first[1] * second[0])
    return all(value > 0.0 for value in crosses) or all(
        value < 0.0 for value in crosses
    )


def _valid_geometry_correspondences(
    geometry: GeometryEstimate | None,
) -> tuple[tuple[float, float], ...] | None:
    if (
        not isinstance(geometry, GeometryEstimate)
        or not geometry.legal
        or geometry.error is not None
        or geometry.observed_corners_in_canonical_normalized is None
        or geometry.homography_observed_to_canonical is None
    ):
        return None
    try:
        corners = tuple(
            tuple(float(value) for value in row)
            for row in geometry.observed_corners_in_canonical_normalized
        )
        homography = tuple(
            tuple(float(value) for value in row)
            for row in geometry.homography_observed_to_canonical
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        len(corners) != 4
        or any(len(row) != 2 for row in corners)
        or len(homography) != 3
        or any(len(row) != 3 for row in homography)
        or any(not math.isfinite(value) for row in corners for value in row)
        or any(not math.isfinite(value) for row in homography for value in row)
        or not _strict_convex(corners)
    ):
        return None
    return corners


def _sanity_direct_valid(geometry: GeometryEstimate | None) -> bool:
    corners = _valid_geometry_correspondences(geometry)
    if corners is None:
        return False
    maximum_error = max(
        abs(observed - canonical)
        for observed_corner, canonical_corner in zip(
            corners, CANONICAL_CORNERS_NORMALIZED, strict=True
        )
        for observed, canonical in zip(
            observed_corner, canonical_corner, strict=True
        )
    )
    return maximum_error <= R1A_SANITY_MAX_CORNER_ERROR_NORMALIZED


def evaluate_r1a_observation(
    *,
    unit_id: str,
    spec: R1AConditionSpec,
    attacked_image: Image.Image | None,
    geometry: GeometryEstimate | None,
    errors: Sequence[str] = (),
) -> R1AUnitRecord:
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("R1A unit id must be nonempty")
    if spec not in R1A_ALL_CONDITIONS:
        raise ValueError("R1A observation requires one exact frozen condition")
    truth = truth_correspondences(spec)
    identity_error = corner_rmse(CANONICAL_CORNERS_NORMALIZED, truth)
    eligible = spec.kind is R1AConditionKind.CORE and (
        identity_error > R1A_DELTA_NONTRIVIAL_NORMALIZED
    )
    prediction_error = None
    paired_delta = None
    improved = None
    recorded_errors = tuple(str(value) for value in errors)
    correspondences = _valid_geometry_correspondences(geometry)
    if correspondences is not None and not recorded_errors:
        prediction_error = corner_rmse(correspondences, truth)
        paired_delta = prediction_error - identity_error
        if not math.isfinite(paired_delta):
            prediction_error = paired_delta = None
        elif eligible:
            improved = paired_delta < 0.0
    elif not recorded_errors:
        recorded_errors = ("geometry_invalid",)
    return R1AUnitRecord(
        unit_id,
        spec.condition_id,
        spec.kind,
        truth,
        identity_error,
        eligible,
        attacked_image,
        geometry,
        prediction_error,
        paired_delta,
        improved,
        recorded_errors,
    )


def run_r1a_unit(
    *,
    unit_id: str,
    source_cg_rgb: Any,
    spec: R1AConditionSpec,
    detector: GeometryDetector,
) -> R1AUnitRecord:
    try:
        attacked = render_r1a_attack(source_cg_rgb, spec)
    except Exception as error:
        return evaluate_r1a_observation(
            unit_id=unit_id,
            spec=spec,
            attacked_image=None,
            geometry=None,
            errors=(f"attack_render:{type(error).__name__}",),
        )
    try:
        geometry = detect_attacked_rgb(detector, attacked)
    except Exception as error:
        return evaluate_r1a_observation(
            unit_id=unit_id,
            spec=spec,
            attacked_image=attacked,
            geometry=None,
            errors=(f"geometry_detect:{type(error).__name__}",),
        )
    errors = (
        ("geometry_detect:reported_error",)
        if geometry.status is GeometryStatus.ERROR
        else ()
    )
    return evaluate_r1a_observation(
        unit_id=unit_id,
        spec=spec,
        attacked_image=attacked,
        geometry=geometry,
        errors=errors,
    )


def r1a_detection_setup_failure_record(
    *,
    unit_id: str,
    spec: R1AConditionSpec,
    attacked_image: Image.Image,
    error: BaseException,
) -> R1AUnitRecord:
    return evaluate_r1a_observation(
        unit_id=unit_id,
        spec=spec,
        attacked_image=attacked_image,
        geometry=None,
        errors=(f"syncseal_runtime_setup:{type(error).__name__}",),
    )


def aggregate_r1a_condition(
    *,
    condition_records: R1AConditionRecords,
    ordered_roster: Sequence[str],
) -> R1AConditionAggregate:
    if not isinstance(condition_records, R1AConditionRecords):
        raise TypeError("R1A aggregate requires R1AConditionRecords")
    spec = condition_records.spec
    if spec not in R1A_ALL_CONDITIONS:
        raise ValueError("R1A aggregate condition identity differs")
    roster = tuple(ordered_roster)
    records = tuple(condition_records.records)
    if (
        len(roster) != R1A_FIXED_UNIT_COUNT
        or len(set(roster)) != R1A_FIXED_UNIT_COUNT
        or len(records) != R1A_FIXED_UNIT_COUNT
        or tuple(record.unit_id for record in records) != roster
        or any(
            record.condition_id != spec.condition_id
            or record.condition_kind is not spec.kind
            for record in records
        )
    ):
        raise ValueError("R1A records must match the fixed ordered eight-unit roster")
    if spec.kind is R1AConditionKind.SANITY:
        valid_count = sum(
            not record.errors and _sanity_direct_valid(record.geometry)
            for record in records
        )
        return R1AConditionAggregate(
            spec.condition_id,
            spec.kind,
            roster,
            R1A_FIXED_UNIT_COUNT,
            0,
            valid_count,
            0,
            None,
            valid_count == R1A_FIXED_UNIT_COUNT,
        )
    eligible_count = sum(record.truth_eligible for record in records)
    deltas = tuple(
        record.paired_delta
        for record in records
        if not record.errors
        and isinstance(record.paired_delta, float)
        and math.isfinite(record.paired_delta)
    )
    valid_count = len(deltas)
    improved_count = sum(record.improved is True for record in records)
    paired_median = (
        float(median(deltas))
        if eligible_count == R1A_FIXED_UNIT_COUNT
        and valid_count == R1A_FIXED_UNIT_COUNT
        else None
    )
    passed = bool(
        eligible_count == R1A_FIXED_UNIT_COUNT
        and valid_count == R1A_FIXED_UNIT_COUNT
        and improved_count >= R1A_MIN_IMPROVED_COUNT
        and paired_median is not None
        and paired_median < 0.0
    )
    return R1AConditionAggregate(
        spec.condition_id,
        spec.kind,
        roster,
        R1A_FIXED_UNIT_COUNT,
        eligible_count,
        valid_count,
        improved_count,
        paired_median,
        passed,
    )


def evaluate_r1a(
    *,
    condition_records: Sequence[R1AConditionRecords],
    ordered_roster: Sequence[str],
) -> R1AEvaluation:
    preflight = r1a_truth_preflight()
    if not preflight.passed:
        return R1AEvaluation(
            R1A_ATTACK_SPEC_REQUEST_CHANGES,
            preflight,
            (),
            False,
            False,
            False,
        )
    supplied = tuple(condition_records)
    if tuple(item.spec for item in supplied) != R1A_ALL_CONDITIONS:
        raise ValueError("R1A evaluation requires all thirteen conditions in order")
    aggregates = tuple(
        aggregate_r1a_condition(
            condition_records=item,
            ordered_roster=ordered_roster,
        )
        for item in supplied
    )
    sanity = aggregates[: len(R1A_SANITY_CONDITIONS)]
    core = aggregates[len(R1A_SANITY_CONDITIONS) :]
    all_sanity_passed = all(item.passed for item in sanity)
    all_core_passed = all(item.passed for item in core)
    passed = all_sanity_passed and all_core_passed
    return R1AEvaluation(
        R1A_BLOCKING_METHOD_CANARY_PASSED
        if passed
        else R1A_BLOCKING_METHOD_CANARY_FAILED,
        preflight,
        aggregates,
        all_sanity_passed,
        all_core_passed,
        passed,
    )


def condition_by_id(condition_id: str) -> R1AConditionSpec:
    matches = tuple(
        spec for spec in R1A_ALL_CONDITIONS if spec.condition_id == condition_id
    )
    if len(matches) != 1:
        raise ValueError("unknown R1A condition identity")
    return matches[0]


__all__ = [
    "R1A_ALL_CONDITIONS",
    "R1A_ATTACK_SPEC_REQUEST_CHANGES",
    "R1A_BLOCKING_METHOD_CANARY_FAILED",
    "R1A_BLOCKING_METHOD_CANARY_PASSED",
    "R1A_CORE_CONDITIONS",
    "R1A_DELTA_NONTRIVIAL_NORMALIZED",
    "R1A_FIXED_UNIT_COUNT",
    "R1A_MIN_IMPROVED_COUNT",
    "R1A_SANITY_CONDITIONS",
    "R1A_SANITY_MAX_CORNER_ERROR_NORMALIZED",
    "R1AConditionAggregate",
    "R1AConditionKind",
    "R1AConditionRecords",
    "R1AConditionSpec",
    "R1AEvaluation",
    "R1ATruthPreflight",
    "R1AUnitRecord",
    "aggregate_r1a_condition",
    "apply_homography",
    "condition_by_id",
    "corner_rmse",
    "detect_attacked_rgb",
    "evaluate_r1a",
    "evaluate_r1a_observation",
    "r1a_detection_setup_failure_record",
    "r1a_truth_preflight",
    "render_r1a_attack",
    "run_r1a_unit",
    "truth_correspondences",
]
