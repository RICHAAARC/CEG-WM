"""R0/F1 paired four-arm final-RGB execution and fail-closed records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate
from cegwm.runtime.observation import require_ordinary_rgb_image


DEVELOPMENT_SELECTION_RULE = (
    "first predeclared residual strength satisfying every frozen gate; "
    "freeze once on development and never reselect on test"
)


class R0Arm(str, Enum):
    U = "U_no_content_no_sync"
    G = "G_no_content_with_sync"
    C = "C_with_content_no_sync"
    CG = "CG_with_content_with_sync"


@dataclass(frozen=True, slots=True)
class R0NumericGates:
    """Single compact home for user-frozen R0 numeric gates.

    ``None`` and an empty sequence mean pending confirmation, never pass.
    """

    residual_strengths: tuple[float, ...] = ()
    min_psnr: float | None = None
    min_ssim: float | None = None
    max_lpips: float | None = None
    max_cg_c_flip_rate: float | None = None
    max_g_content_false_positive_rate: float | None = None
    min_sync_basic_observability_rate: float | None = None

    def __post_init__(self) -> None:
        for strength in self.residual_strengths:
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise TypeError("residual strengths must be real")
            if not math.isfinite(float(strength)) or float(strength) < 0.0:
                raise ValueError("residual strengths must be finite and nonnegative")
        if len(set(float(value) for value in self.residual_strengths)) != len(
            self.residual_strengths
        ):
            raise ValueError("residual strengths must be unique and preordered")
        for name in (
            "min_psnr",
            "min_ssim",
            "max_lpips",
            "max_cg_c_flip_rate",
            "max_g_content_false_positive_rate",
            "min_sync_basic_observability_rate",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite when frozen")

    @property
    def pending_fields(self) -> tuple[str, ...]:
        pending = []
        if not self.residual_strengths:
            pending.append("residual_strengths")
        pending.extend(
            name
            for name in (
                "min_psnr",
                "min_ssim",
                "max_lpips",
                "max_cg_c_flip_rate",
                "max_g_content_false_positive_rate",
                "min_sync_basic_observability_rate",
            )
            if getattr(self, name) is None
        )
        return tuple(pending)

    def require_frozen(self) -> None:
        if self.pending_fields:
            raise RuntimeError("R0 numeric gates pending confirmation: " + ", ".join(self.pending_fields))


@dataclass(frozen=True, slots=True)
class ContentScore:
    lf: float
    hf: float
    weighted_joint: float
    positive: bool

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (self.lf, self.hf, self.weighted_joint)
        ):
            raise ValueError("content LF/HF/weighted-joint scores must be finite raw values")
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
    residual_strength: float
    arms: tuple[R0ArmRecord, ...]
    cg_minus_c_raw: tuple[tuple[str, float], ...] | None
    cg_c_content_flip: bool | None
    g_content_false_positive: bool | None
    negative_arm_denominator: int
    positive_arm_denominator: int
    failure_arm_denominator: int
    failed_arm_count: int


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
    residual_strength: float,
    sync_embedder: SyncEmbedder,
    content_scorer: ContentScorer,
    geometry_detector: GeometryDetector,
    quality_scorer: QualityScorer,
) -> R0UnitRecord:
    """Execute exactly U/G/C/CG once each with fixed denominators and no fallback."""

    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("R0 unit_id must be a nonempty string")
    if isinstance(residual_strength, bool) or not isinstance(residual_strength, (int, float)):
        raise TypeError("R0 residual strength must be real")
    strength = float(residual_strength)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("R0 residual strength must be finite and nonnegative")
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
            images[target_arm] = _rgb512(sync_embedder(images[source_arm], strength))
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
        strength,
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
        "residual_strength": record.residual_strength,
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


__all__ = [
    "ContentScore",
    "DEVELOPMENT_SELECTION_RULE",
    "ImageQuality",
    "R0Arm",
    "R0ArmRecord",
    "R0NumericGates",
    "R0UnitRecord",
    "r0_record_payload",
    "run_r0_four_arm_unit",
]
