"""R0/F1 paired four-arm final-RGB execution and fail-closed records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from PIL import Image

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, GeometryEstimate
from cegwm.geometry_v7.syncseal import SYNCSEAL_OFFICIAL_BASE_ALPHA
from cegwm.runtime.observation import require_ordinary_rgb_image


DEVELOPMENT_SELECTION_RULE = (
    "first predeclared residual strength multiplier satisfying every frozen gate; "
    "freeze once on development and never reselect on test"
)
R0_DEVELOPMENT_ROSTER_SIZE = 4
R0_EVALUATION_ROSTER_SIZE = 8
R0_NO_WINDOW_STOP = (
    "no carrier-compatibility window found on the preregistered strength grid "
    "and fixed R0 roster; Geometry-V7 stops by contract"
)


class R0Arm(str, Enum):
    U = "U_no_content_no_sync"
    G = "G_no_content_with_sync"
    C = "C_with_content_no_sync"
    CG = "CG_with_content_with_sync"


@dataclass(frozen=True, slots=True)
class R0NumericGates:
    """Frozen R0 carrier-compatibility and identity-interface gates."""

    base_syncseal_alpha: float = SYNCSEAL_OFFICIAL_BASE_ALPHA
    residual_strength_multipliers: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    min_mean_psnr: float = 40.0
    min_mean_ssim: float = 0.98
    max_mean_lpips: float = 0.05
    max_cg_c_decision_flip_rate: float = 0.0
    max_g_content_false_positive_rate: float = 0.0
    min_identity_coordinate_valid_rate: float = 1.0
    # One official 256-grid pixel step after public normalization.  This is
    # implementation tolerance for identity coordinates, not a geometry gate.
    identity_homography_max_error_normalized: float = 2.0 / 255.0

    def __post_init__(self) -> None:
        actual = (
            self.base_syncseal_alpha,
            self.residual_strength_multipliers,
            self.min_mean_psnr,
            self.min_mean_ssim,
            self.max_mean_lpips,
            self.max_cg_c_decision_flip_rate,
            self.max_g_content_false_positive_rate,
            self.min_identity_coordinate_valid_rate,
            self.identity_homography_max_error_normalized,
        )
        expected = (
            0.20,
            (0.25, 0.50, 0.75, 1.00),
            40.0,
            0.98,
            0.05,
            0.0,
            0.0,
            1.0,
            2.0 / 255.0,
        )
        if actual != expected:
            raise ValueError("R0 numeric gates must match the exact user-frozen contract")


@dataclass(frozen=True, slots=True)
class ContentScore:
    lf: float
    hf: float
    weighted_joint: float
    margin: float
    positive: bool

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (self.lf, self.hf, self.weighted_joint, self.margin)
        ):
            raise ValueError("content LF/HF/weighted-joint/margin scores must be finite raw values")
        if not isinstance(self.positive, bool):
            raise TypeError("content positive must come from the unchanged frozen decision callable")


@dataclass(frozen=True, slots=True)
class ImageQuality:
    psnr: float
    ssim: float
    lpips: float

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (self.psnr, self.ssim, self.lpips)
        ):
            raise ValueError("PSNR/SSIM/LPIPS must be finite raw values")


@dataclass(frozen=True, slots=True)
class R0ArmRecord:
    arm: R0Arm
    image: Image.Image | None
    content: ContentScore | None
    geometry: GeometryEstimate | None
    quality_to_unsynchronized_pair: ImageQuality | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R0UnitRecord:
    unit_id: str
    base_syncseal_alpha: float
    residual_strength_multiplier: float
    arms: tuple[R0ArmRecord, ...]
    cg_minus_c_raw: tuple[tuple[str, float], ...] | None
    cg_c_content_flip: bool | None
    g_content_false_positive: bool | None
    negative_arm_denominator: int
    positive_arm_denominator: int
    failure_arm_denominator: int
    failed_arm_count: int


class R0Stage(str, Enum):
    DEVELOPMENT = "development_reference_first_4"
    EVALUATION = "evaluation_fixed_8"


@dataclass(frozen=True, slots=True)
class R0QualityFamilyAggregate:
    pair_family: str
    denominator: int
    valid_count: int
    mean_psnr: float | None
    mean_ssim: float | None
    mean_lpips: float | None
    min_psnr: float | None
    max_psnr: float | None
    min_ssim: float | None
    max_ssim: float | None
    min_lpips: float | None
    max_lpips: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class R0AggregateEvaluation:
    stage: R0Stage
    roster: tuple[str, ...]
    base_syncseal_alpha: float
    residual_strength_multiplier: float
    g_u_quality: R0QualityFamilyAggregate
    cg_c_quality: R0QualityFamilyAggregate
    cg_c_decision_flip_count: int
    cg_c_decision_valid_count: int
    cg_c_decision_flip_denominator: int
    cg_c_decision_flip_rate: float | None
    g_content_false_positive_count: int
    g_content_decision_valid_count: int
    g_content_false_positive_denominator: int
    g_content_false_positive_rate: float | None
    identity_coordinate_valid_count: int
    identity_coordinate_valid_denominator: int
    identity_coordinate_valid_rate: float
    carrier_compatibility_passed: bool


@dataclass(frozen=True, slots=True)
class R0MultiplierRecords:
    residual_strength_multiplier: float
    records: tuple[R0UnitRecord, ...]


@dataclass(frozen=True, slots=True)
class R0DevelopmentSelection:
    attempts: tuple[R0AggregateEvaluation, ...]
    selected_residual_strength_multiplier: float | None
    complete: bool
    stop_reason: str | None


ContentScorer = Callable[[Image.Image], ContentScore]
GeometryDetector = Callable[[Image.Image], GeometryEstimate]
QualityScorer = Callable[[Image.Image, Image.Image], ImageQuality]
SyncEmbedder = Callable[[Image.Image, float], Image.Image]


def _rgb512(image: Any) -> Image.Image:
    rgb = require_ordinary_rgb_image(image)
    if rgb.size != (512, 512):
        raise ValueError("R0/F1 arms require final raw 512x512 RGB")
    return rgb


def _error(stage: str, error: BaseException) -> str:
    return f"{stage}:{type(error).__name__}:{error}"


def run_r0_four_arm_unit(
    *,
    unit_id: str,
    unwatermarked_final_rgb: Any,
    content_watermarked_final_rgb: Any,
    residual_strength_multiplier: float,
    sync_embedder: SyncEmbedder,
    content_scorer: ContentScorer,
    geometry_detector: GeometryDetector,
    quality_scorer: QualityScorer,
) -> R0UnitRecord:
    """Execute exactly U/G/C/CG once each with fixed denominators and no fallback."""

    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("R0 unit_id must be a nonempty string")
    if isinstance(residual_strength_multiplier, bool) or not isinstance(
        residual_strength_multiplier, (int, float)
    ):
        raise TypeError("R0 residual strength multiplier must be real")
    multiplier = float(residual_strength_multiplier)
    if multiplier not in R0NumericGates().residual_strength_multipliers:
        raise ValueError("R0 residual strength multiplier must be on the frozen ordered grid")
    for name, method in (
        ("sync_embedder", sync_embedder),
        ("content_scorer", content_scorer),
        ("geometry_detector", geometry_detector),
        ("quality_scorer", quality_scorer),
    ):
        if not callable(method):
            raise TypeError(f"{name} must be the injected real method callable")

    images: dict[R0Arm, Image.Image | None] = {
        R0Arm.U: _rgb512(unwatermarked_final_rgb),
        R0Arm.C: _rgb512(content_watermarked_final_rgb),
        R0Arm.G: None,
        R0Arm.CG: None,
    }
    errors: dict[R0Arm, list[str]] = {arm: [] for arm in R0Arm}
    for source_arm, target_arm in ((R0Arm.U, R0Arm.G), (R0Arm.C, R0Arm.CG)):
        try:
            images[target_arm] = _rgb512(sync_embedder(images[source_arm], multiplier))
        except Exception as error:
            errors[target_arm].append(_error("sync_embed", error))

    content: dict[R0Arm, ContentScore | None] = {arm: None for arm in R0Arm}
    geometry: dict[R0Arm, GeometryEstimate | None] = {arm: None for arm in R0Arm}
    quality: dict[R0Arm, ImageQuality | None] = {arm: None for arm in R0Arm}
    for arm in R0Arm:
        image = images[arm]
        if image is None:
            continue
        try:
            score = content_scorer(image)
            if not isinstance(score, ContentScore):
                raise TypeError("content scorer must return ContentScore")
            content[arm] = score
        except Exception as error:
            errors[arm].append(_error("content_score", error))
        try:
            estimate = geometry_detector(image)
            if not isinstance(estimate, GeometryEstimate):
                raise TypeError("geometry detector must return GeometryEstimate")
            geometry[arm] = estimate
            if estimate.error is not None:
                errors[arm].append(f"geometry_detect:{estimate.error}")
        except Exception as error:
            errors[arm].append(_error("geometry_detect", error))

    for base_arm, synced_arm in ((R0Arm.U, R0Arm.G), (R0Arm.C, R0Arm.CG)):
        if images[synced_arm] is None:
            continue
        try:
            metrics = quality_scorer(images[base_arm], images[synced_arm])
            if not isinstance(metrics, ImageQuality):
                raise TypeError("quality scorer must return ImageQuality")
            quality[synced_arm] = metrics
        except Exception as error:
            errors[synced_arm].append(_error("quality_score", error))

    c_score = content[R0Arm.C]
    cg_score = content[R0Arm.CG]
    cg_minus_c = None
    cg_c_flip = None
    if c_score is not None and cg_score is not None:
        cg_minus_c = (
            ("lf", cg_score.lf - c_score.lf),
            ("hf", cg_score.hf - c_score.hf),
            ("weighted_joint", cg_score.weighted_joint - c_score.weighted_joint),
            ("margin", cg_score.margin - c_score.margin),
        )
        cg_c_flip = cg_score.positive != c_score.positive
    g_score = content[R0Arm.G]
    g_false_positive = None if g_score is None else g_score.positive

    arm_records = tuple(
        R0ArmRecord(
            arm,
            images[arm],
            content[arm],
            geometry[arm],
            quality[arm],
            tuple(errors[arm]),
        )
        for arm in R0Arm
    )
    return R0UnitRecord(
        unit_id,
        SYNCSEAL_OFFICIAL_BASE_ALPHA,
        multiplier,
        arm_records,
        cg_minus_c,
        cg_c_flip,
        g_false_positive,
        negative_arm_denominator=2,
        positive_arm_denominator=2,
        failure_arm_denominator=4,
        failed_arm_count=sum(bool(record.errors) for record in arm_records),
    )


def r0_record_payload(record: R0UnitRecord) -> dict[str, object]:
    """Project one complete in-memory record to a strict JSON-safe payload."""

    if not isinstance(record, R0UnitRecord):
        raise TypeError("R0 payload requires R0UnitRecord")
    arms: list[dict[str, object]] = []
    for item in record.arms:
        content = item.content
        geometry = item.geometry
        quality = item.quality_to_unsynchronized_pair
        arms.append(
            {
                "arm": item.arm.value,
                "image_present": item.image is not None,
                "content": None
                if content is None
                else {
                    "lf": content.lf,
                    "hf": content.hf,
                    "weighted_joint": content.weighted_joint,
                    "margin": content.margin,
                    "positive": content.positive,
                },
                "geometry": None
                if geometry is None
                else {
                    "status": geometry.status.value,
                    "uncalibrated_sync_logit": geometry.uncalibrated_sync_logit,
                    "raw_syncseal_corners": geometry.raw_syncseal_corners,
                    "corners_current_normalized": geometry.corners_current_normalized,
                    "homography_current_to_canonical": geometry.homography_current_to_canonical,
                    "legal": geometry.legal,
                    "basic_observable": geometry.basic_observable,
                    "error": geometry.error,
                },
                "quality_to_unsynchronized_pair": None
                if quality is None
                else {"psnr": quality.psnr, "ssim": quality.ssim, "lpips": quality.lpips},
                "errors": item.errors,
            }
        )
    return {
        "unit_id": record.unit_id,
        "base_syncseal_alpha": record.base_syncseal_alpha,
        "residual_strength_multiplier": record.residual_strength_multiplier,
        "arms": arms,
        "cg_minus_c_raw": None
        if record.cg_minus_c_raw is None
        else dict(record.cg_minus_c_raw),
        "cg_c_content_flip": record.cg_c_content_flip,
        "g_content_false_positive": record.g_content_false_positive,
        "negative_arm_denominator": record.negative_arm_denominator,
        "positive_arm_denominator": record.positive_arm_denominator,
        "failure_arm_denominator": record.failure_arm_denominator,
        "failed_arm_count": record.failed_arm_count,
    }


def _arm_map(record: R0UnitRecord) -> dict[R0Arm, R0ArmRecord]:
    expected = tuple(R0Arm)
    if not isinstance(record, R0UnitRecord) or tuple(item.arm for item in record.arms) != expected:
        raise ValueError("R0 aggregate requires exactly ordered U/G/C/CG arm records")
    return {item.arm: item for item in record.arms}


def _quality_family(
    records: Sequence[R0UnitRecord],
    *,
    base_arm: R0Arm,
    synced_arm: R0Arm,
    pair_family: str,
    gates: R0NumericGates,
) -> R0QualityFamilyAggregate:
    values: list[ImageQuality] = []
    for record in records:
        arms = _arm_map(record)
        base = arms[base_arm]
        synced = arms[synced_arm]
        quality = synced.quality_to_unsynchronized_pair
        if base.image is None or synced.image is None or not isinstance(quality, ImageQuality):
            continue
        if any(
            not math.isfinite(float(value))
            for value in (quality.psnr, quality.ssim, quality.lpips)
        ):
            continue
        values.append(quality)
    denominator = len(records)
    complete = len(values) == denominator
    psnr = [value.psnr for value in values]
    ssim = [value.ssim for value in values]
    lpips = [value.lpips for value in values]
    mean_psnr = math.fsum(psnr) / denominator if complete else None
    mean_ssim = math.fsum(ssim) / denominator if complete else None
    mean_lpips = math.fsum(lpips) / denominator if complete else None
    passed = bool(
        complete
        and mean_psnr is not None
        and mean_ssim is not None
        and mean_lpips is not None
        and mean_psnr >= gates.min_mean_psnr
        and mean_ssim >= gates.min_mean_ssim
        and mean_lpips <= gates.max_mean_lpips
    )
    return R0QualityFamilyAggregate(
        pair_family,
        denominator,
        len(values),
        mean_psnr,
        mean_ssim,
        mean_lpips,
        min(psnr) if psnr else None,
        max(psnr) if psnr else None,
        min(ssim) if ssim else None,
        max(ssim) if ssim else None,
        min(lpips) if lpips else None,
        max(lpips) if lpips else None,
        passed,
    )


def _identity_coordinate_valid(
    estimate: GeometryEstimate | None, max_error_normalized: float
) -> bool:
    if (
        not isinstance(estimate, GeometryEstimate)
        or not estimate.legal
        or estimate.error is not None
        or estimate.homography_current_to_canonical is None
    ):
        return False
    try:
        matrix = tuple(tuple(float(value) for value in row) for row in estimate.homography_current_to_canonical)
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            return False
        if any(not math.isfinite(value) for row in matrix for value in row):
            return False
        probes = (*CANONICAL_CORNERS_NORMALIZED, (0.0, 0.0))
        errors = []
        for x, y in probes:
            denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
            if not math.isfinite(denominator) or denominator == 0.0:
                return False
            mapped_x = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator
            mapped_y = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator
            if not math.isfinite(mapped_x) or not math.isfinite(mapped_y):
                return False
            errors.append(max(abs(mapped_x - x), abs(mapped_y - y)))
        return max(errors) <= max_error_normalized
    except (TypeError, ValueError, OverflowError):
        return False


def _evaluate_r0_records(
    *,
    stage: R0Stage,
    records: Sequence[R0UnitRecord],
    ordered_roster: Sequence[str],
    residual_strength_multiplier: float,
    gates: R0NumericGates | None = None,
) -> R0AggregateEvaluation:
    """Evaluate one fixed multiplier without retry, fallback, or subset means."""

    frozen = R0NumericGates() if gates is None else gates
    if not isinstance(frozen, R0NumericGates):
        raise TypeError("R0 evaluation requires the frozen R0NumericGates")
    expected_count = {
        R0Stage.DEVELOPMENT: R0_DEVELOPMENT_ROSTER_SIZE,
        R0Stage.EVALUATION: R0_EVALUATION_ROSTER_SIZE,
    }.get(stage)
    if expected_count is None:
        raise TypeError("R0 stage must be development or evaluation")
    roster = tuple(ordered_roster)
    if (
        len(roster) != expected_count
        or len(set(roster)) != expected_count
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in roster)
    ):
        raise ValueError(f"R0 {stage.value} roster must contain {expected_count} unique ordered ids")
    fixed_records = tuple(records)
    if any(not isinstance(record, R0UnitRecord) for record in fixed_records):
        raise TypeError("R0 aggregate requires R0UnitRecord inputs")
    if len(fixed_records) != expected_count or tuple(record.unit_id for record in fixed_records) != roster:
        raise ValueError("R0 records must match the complete fixed roster in exact order")
    if isinstance(residual_strength_multiplier, bool) or not isinstance(
        residual_strength_multiplier, (int, float)
    ):
        raise TypeError("R0 aggregate multiplier must be real")
    multiplier = float(residual_strength_multiplier)
    if multiplier not in frozen.residual_strength_multipliers:
        raise ValueError("R0 aggregate multiplier must be on the frozen ordered grid")
    for record in fixed_records:
        if (
            record.base_syncseal_alpha != frozen.base_syncseal_alpha
            or record.residual_strength_multiplier != multiplier
            or record.negative_arm_denominator != 2
            or record.positive_arm_denominator != 2
            or record.failure_arm_denominator != 4
        ):
            raise ValueError("R0 record identity or fixed denominator drift")
        _arm_map(record)

    g_u_quality = _quality_family(
        fixed_records,
        base_arm=R0Arm.U,
        synced_arm=R0Arm.G,
        pair_family="G_to_U",
        gates=frozen,
    )
    cg_c_quality = _quality_family(
        fixed_records,
        base_arm=R0Arm.C,
        synced_arm=R0Arm.CG,
        pair_family="CG_to_C",
        gates=frozen,
    )

    flip_values: list[bool | None] = []
    g_values: list[bool | None] = []
    for record in fixed_records:
        arms = _arm_map(record)
        c_content = arms[R0Arm.C].content
        cg_content = arms[R0Arm.CG].content
        expected_flip = (
            c_content.positive != cg_content.positive
            if isinstance(c_content, ContentScore) and isinstance(cg_content, ContentScore)
            else None
        )
        flip_values.append(
            record.cg_c_content_flip
            if isinstance(record.cg_c_content_flip, bool)
            and record.cg_c_content_flip == expected_flip
            else None
        )
        g_content = arms[R0Arm.G].content
        expected_g_false_positive = (
            g_content.positive if isinstance(g_content, ContentScore) else None
        )
        g_values.append(
            record.g_content_false_positive
            if isinstance(record.g_content_false_positive, bool)
            and record.g_content_false_positive == expected_g_false_positive
            else None
        )
    flip_valid = sum(isinstance(value, bool) for value in flip_values)
    flip_count = sum(value is True for value in flip_values)
    flip_rate = flip_count / expected_count if flip_valid == expected_count else None
    g_valid = sum(isinstance(value, bool) for value in g_values)
    g_false_positive_count = sum(value is True for value in g_values)
    g_false_positive_rate = (
        g_false_positive_count / expected_count if g_valid == expected_count else None
    )

    identity_valid = 0
    for record in fixed_records:
        arms = _arm_map(record)
        identity_valid += sum(
            _identity_coordinate_valid(
                arms[arm].geometry, frozen.identity_homography_max_error_normalized
            )
            for arm in (R0Arm.G, R0Arm.CG)
        )
    identity_denominator = expected_count * 2
    identity_rate = identity_valid / identity_denominator
    passed = bool(
        g_u_quality.passed
        and cg_c_quality.passed
        and flip_rate is not None
        and flip_rate <= frozen.max_cg_c_decision_flip_rate
        and g_false_positive_rate is not None
        and g_false_positive_rate <= frozen.max_g_content_false_positive_rate
        and identity_rate >= frozen.min_identity_coordinate_valid_rate
    )
    return R0AggregateEvaluation(
        stage,
        roster,
        frozen.base_syncseal_alpha,
        multiplier,
        g_u_quality,
        cg_c_quality,
        flip_count,
        flip_valid,
        expected_count,
        flip_rate,
        g_false_positive_count,
        g_valid,
        expected_count,
        g_false_positive_rate,
        identity_valid,
        identity_denominator,
        identity_rate,
        passed,
    )


def select_r0_development_multiplier(
    *,
    attempts: Sequence[R0MultiplierRecords],
    ordered_reference_roster_first_4: Sequence[str],
    gates: R0NumericGates | None = None,
) -> R0DevelopmentSelection:
    """Select the first passing prefix entry or freeze the bounded stop conclusion."""

    frozen = R0NumericGates() if gates is None else gates
    supplied = tuple(attempts)
    if not supplied or len(supplied) > len(frozen.residual_strength_multipliers):
        raise ValueError("development attempts must be a nonempty frozen-grid prefix")
    aggregates: list[R0AggregateEvaluation] = []
    selected = None
    for index, attempt in enumerate(supplied):
        expected_multiplier = frozen.residual_strength_multipliers[index]
        if (
            not isinstance(attempt, R0MultiplierRecords)
            or attempt.residual_strength_multiplier != expected_multiplier
        ):
            raise ValueError("development attempts must follow the frozen multiplier order")
        aggregate = _evaluate_r0_records(
            stage=R0Stage.DEVELOPMENT,
            records=attempt.records,
            ordered_roster=ordered_reference_roster_first_4,
            residual_strength_multiplier=expected_multiplier,
            gates=frozen,
        )
        aggregates.append(aggregate)
        if aggregate.carrier_compatibility_passed:
            selected = expected_multiplier
            if index != len(supplied) - 1:
                raise ValueError("development must stop immediately after the first passing multiplier")
            break
    complete = selected is not None or len(supplied) == len(frozen.residual_strength_multipliers)
    stop_reason = R0_NO_WINDOW_STOP if complete and selected is None else None
    return R0DevelopmentSelection(tuple(aggregates), selected, complete, stop_reason)


def evaluate_r0_test(
    *,
    records: Sequence[R0UnitRecord],
    ordered_evaluation_roster_8: Sequence[str],
    development_selection: R0DevelopmentSelection,
    gates: R0NumericGates | None = None,
) -> R0AggregateEvaluation:
    """Run the same gates once on the fixed eight-unit test roster."""

    if (
        not isinstance(development_selection, R0DevelopmentSelection)
        or not development_selection.complete
        or development_selection.selected_residual_strength_multiplier is None
        or development_selection.stop_reason is not None
    ):
        raise ValueError("R0 test requires a completed passing development selection")
    return _evaluate_r0_records(
        stage=R0Stage.EVALUATION,
        records=records,
        ordered_roster=ordered_evaluation_roster_8,
        residual_strength_multiplier=development_selection.selected_residual_strength_multiplier,
        gates=gates,
    )


__all__ = [
    "ContentScore",
    "DEVELOPMENT_SELECTION_RULE",
    "ImageQuality",
    "R0Arm",
    "R0ArmRecord",
    "R0AggregateEvaluation",
    "R0DevelopmentSelection",
    "R0MultiplierRecords",
    "R0NumericGates",
    "R0QualityFamilyAggregate",
    "R0Stage",
    "R0UnitRecord",
    "evaluate_r0_test",
    "r0_record_payload",
    "run_r0_four_arm_unit",
    "select_r0_development_multiplier",
]
