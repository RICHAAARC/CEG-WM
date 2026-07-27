"""Conditional same-detector recovery for the frozen CEG-WM joint candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Protocol

import torch

from main.content_chain.detector import (
    ContentDetectionResult,
    ContentDetectorError,
    validate_content_detection_result,
)
from main.content_chain.hf_detector import HfDetectionResult, HfDetectorError
from main.content_chain.lf_detector import LfDetectionResult, LfDetectorError
from main.geometry_chain.qk_sync import QkGeometrySyncError
from main.geometry_chain.rectifier import (
    ImageRectificationResult,
    ImageRectifierError,
    image_rectifier,
    validate_image_rectification_result,
)
from main.geometry_chain.reliability import (
    GeometryReliabilityError,
    GeometryReliabilityResult,
    GeometryReliabilityThresholds,
    geometry_reliability,
    validate_geometry_reliability_result,
)
from main.geometry_chain.transform_estimator import (
    GeometricTransformEstimation,
    GeometricTransformEstimatorError,
    validate_geometric_transform_estimation,
)
from main.shared.key_schedule import (
    KeyScheduleError,
    identify_root_key,
    stable_json_utf8,
)
from main.shared.rgb8 import (
    Rgb8ImageError,
    clone_rgb8_image,
    rgb8_image_digest,
    validate_rgb8_image,
    validate_rgb8_image_digest,
)

JOINT_CANDIDATE_ID = "joint_conditional_recovery"

_STATUS_STATE_MACHINE = {
    "raw_content_operation_failure": (
        "raw_content_operation_failure",
        False, False, False, False, False, False,
        False, None, None, True,
    ),
    "raw_content_identity_failure": (
        "raw_content_identity_failure",
        False, False, False, False, False, False,
        False, None, None, True,
    ),
    "raw_positive": (
        "raw_content_at_or_above_tau",
        True, False, False, False, False, False,
        True, "raw_content", "raw_positive", False,
    ),
    "negative_without_geometry": (
        "raw_content_below_tau_rescue",
        True, False, False, False, False, False,
        False, None, None, False,
    ),
    "negative_geometry_operation_failure": (
        "raw_content_in_rescue_window",
        True, True, False, False, False, False,
        False, None, None, True,
    ),
    "negative_geometry_identity_failure": (
        "raw_content_in_rescue_window",
        True, True, False, False, False, False,
        False, None, None, True,
    ),
    "negative_geometry_reliability_failure": (
        "raw_content_in_rescue_window",
        True, True, True, False, False, False,
        False, None, None, True,
    ),
    "negative_geometry_unreliable": (
        "raw_content_in_rescue_window",
        True, True, True, True, False, False,
        False, None, None, True,
    ),
    "negative_rectification_failure": (
        "raw_content_in_rescue_window",
        True, True, True, True, False, False,
        False, None, None, True,
    ),
    "negative_rectified_content_operation_failure": (
        "raw_content_in_rescue_window",
        True, True, True, True, True, False,
        False, None, None, True,
    ),
    "negative_rectified_content_identity_failure": (
        "raw_content_in_rescue_window",
        True, True, True, True, True, False,
        False, None, None, True,
    ),
    "rectified_positive": (
        "raw_content_in_rescue_window",
        True, True, True, True, True, True,
        True, "rectified_content", "rescue_positive", False,
    ),
    "rectified_content_negative": (
        "raw_content_in_rescue_window",
        True, True, True, True, True, True,
        False, None, None, False,
    ),
}


class ConditionalRecoveryError(ValueError):
    """Joint configuration or result identity violates the frozen flow."""


class JointOperationError(RuntimeError):
    """A runtime-backed public-image operation failed as explicitly declared."""


class ContentDetectionOperation(Protocol):
    """Public-image content detection returning a content-owned image binding."""

    def __call__(
        self,
        image: torch.Tensor,
        detection_key: str,
    ) -> ContentDetectionResult: ...


class GeometryEstimationOperation(Protocol):
    """Lazy public-image Q/K extraction followed by the transform estimator."""

    def __call__(
        self,
        image: torch.Tensor,
        registered_key: str,
    ) -> GeometricTransformEstimation: ...


@dataclass(frozen=True, slots=True)
class ContentDetectorBinding:
    """One immutable detector operation and its public identity."""

    content_detection_operation: ContentDetectionOperation
    detector_identity: str
    content_config_digest: str
    hf_detector_identity: str
    hf_detector_config_digest: str
    hf_template_digest: str
    preprocessing_identity: str
    formal_mode: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    detector_binding_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.content_detection_operation):
            raise ConditionalRecoveryError(
                "content detection operation must be callable"
            )
        for role, value in (
            ("detector_identity", self.detector_identity),
            ("content_config_digest", self.content_config_digest),
            ("hf_detector_identity", self.hf_detector_identity),
            ("hf_detector_config_digest", self.hf_detector_config_digest),
            ("hf_template_digest", self.hf_template_digest),
            ("preprocessing_identity", self.preprocessing_identity),
            ("root_key_public_digest", self.root_key_public_digest),
        ):
            if type(value) is not str or not value:
                raise ConditionalRecoveryError(
                    f"{role} must be a non-empty string"
                )
        if self.formal_mode != "hf_only":
            raise ConditionalRecoveryError(
                "batch-5 content binding must preserve formal_mode='hf_only'"
            )
        if self.key_role != "registered" or self.wrong_key_index is not None:
            raise ConditionalRecoveryError(
                "batch-5 detector binding requires registered root-key semantics"
            )
        object.__setattr__(
            self,
            "detector_binding_digest",
            _detector_binding_digest(
                detector_identity=self.detector_identity,
                content_config_digest=self.content_config_digest,
                hf_detector_identity=self.hf_detector_identity,
                hf_detector_config_digest=self.hf_detector_config_digest,
                hf_template_digest=self.hf_template_digest,
                preprocessing_identity=self.preprocessing_identity,
                formal_mode=self.formal_mode,
                root_key_public_digest=self.root_key_public_digest,
                key_role=self.key_role,
                wrong_key_index=self.wrong_key_index,
            ),
        )


@dataclass(frozen=True, slots=True)
class JointDecisionThresholds:
    """Calibration-bound raw and rescue thresholds for one detector binding."""

    tau: float
    tau_rescue: float
    detector_binding_digest: str
    calibration_identity: str
    threshold_identity: str = field(init=False)

    def __post_init__(self) -> None:
        for role, value in (("tau", self.tau), ("tau_rescue", self.tau_rescue)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ConditionalRecoveryError(
                    f"{role} must be a finite number"
                )
        if float(self.tau_rescue) >= float(self.tau):
            raise ConditionalRecoveryError(
                "tau_rescue must be strictly lower than tau"
            )
        for role, value in (
            ("detector_binding_digest", self.detector_binding_digest),
            ("calibration_identity", self.calibration_identity),
        ):
            if type(value) is not str or not value:
                raise ConditionalRecoveryError(
                    f"{role} must be a non-empty string"
                )
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "tau_rescue", float(self.tau_rescue))
        object.__setattr__(
            self,
            "threshold_identity",
            _threshold_identity(
                tau=float(self.tau),
                tau_rescue=float(self.tau_rescue),
                detector_binding_digest=self.detector_binding_digest,
                calibration_identity=self.calibration_identity,
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionalRecoveryResult:
    """Immutable raw/rescue outcome; geometry never supplies positive evidence."""

    candidate_id: str
    formal_mode: str
    full_ceg_wm_eligible: bool
    source_image: torch.Tensor
    source_image_digest: str
    raw_content_result: ContentDetectionResult | None
    raw_content_score: float | None
    geometry_triggered: bool
    trigger_reason: str
    geometry_estimation: GeometricTransformEstimation | None
    geometry_reliability_result: GeometryReliabilityResult | None
    image_rectification_result: ImageRectificationResult | None
    rectified_content_result: ContentDetectionResult | None
    rectified_content_score: float | None
    joint_content_positive: bool
    positive_source: str | None
    positive_path: str | None
    status: str
    failure_reason: str | None
    detector_identity: str
    content_config_digest: str
    hf_detector_identity: str
    hf_detector_config_digest: str
    hf_template_digest: str
    preprocessing_identity: str
    detector_binding_digest: str
    tau: float
    tau_rescue: float
    threshold_identity: str
    calibration_identity: str
    root_key_public_digest: str
    key_role: str
    wrong_key_index: int | None
    decision_identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "source_image",
                clone_rgb8_image(self.source_image),
            )
        except Rgb8ImageError as exc:
            raise ConditionalRecoveryError(
                "joint source image binding is invalid"
            ) from exc
        _validate_conditional_result_types(self)
        object.__setattr__(
            self,
            "decision_identity_digest",
            _decision_identity_digest(self),
        )
        validate_conditional_recovery_result(self)


def _validate_conditional_result_types(
    result: ConditionalRecoveryResult,
) -> None:
    try:
        validate_rgb8_image(result.source_image)
        validate_rgb8_image_digest(result.source_image_digest)
    except Rgb8ImageError as exc:
        raise ConditionalRecoveryError(
            "joint source image binding is invalid"
        ) from exc
    if rgb8_image_digest(result.source_image) != result.source_image_digest:
        raise ConditionalRecoveryError(
            "joint source image digest mismatch"
        )
    optional_types = (
        ("raw_content_result", result.raw_content_result, ContentDetectionResult),
        (
            "geometry_estimation",
            result.geometry_estimation,
            GeometricTransformEstimation,
        ),
        (
            "geometry_reliability_result",
            result.geometry_reliability_result,
            GeometryReliabilityResult,
        ),
        (
            "image_rectification_result",
            result.image_rectification_result,
            ImageRectificationResult,
        ),
        (
            "rectified_content_result",
            result.rectified_content_result,
            ContentDetectionResult,
        ),
    )
    for role, value, expected_type in optional_types:
        if value is not None and type(value) is not expected_type:
            raise ConditionalRecoveryError(
                f"{role} has an invalid result type"
            )
    for role, value in (
        ("raw_content_score", result.raw_content_score),
        ("rectified_content_score", result.rectified_content_score),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ConditionalRecoveryError(
                f"{role} must be a finite number when present"
            )
    for role, value in (
        ("tau", result.tau),
        ("tau_rescue", result.tau_rescue),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ConditionalRecoveryError(f"{role} must be finite")
    for role, value in (
        ("candidate_id", result.candidate_id),
        ("formal_mode", result.formal_mode),
        ("detector_identity", result.detector_identity),
        ("content_config_digest", result.content_config_digest),
        ("hf_detector_identity", result.hf_detector_identity),
        ("hf_detector_config_digest", result.hf_detector_config_digest),
        ("hf_template_digest", result.hf_template_digest),
        ("preprocessing_identity", result.preprocessing_identity),
        ("detector_binding_digest", result.detector_binding_digest),
        ("threshold_identity", result.threshold_identity),
        ("calibration_identity", result.calibration_identity),
        ("root_key_public_digest", result.root_key_public_digest),
        ("key_role", result.key_role),
        ("status", result.status),
        ("trigger_reason", result.trigger_reason),
    ):
        if type(value) is not str or not value:
            raise ConditionalRecoveryError(
                f"{role} must be a non-empty string"
            )
    for role, value in (
        ("positive_source", result.positive_source),
        ("positive_path", result.positive_path),
        ("failure_reason", result.failure_reason),
    ):
        if value is not None and (type(value) is not str or not value):
            raise ConditionalRecoveryError(
                f"{role} must be a non-empty string when present"
            )
    if (
        type(result.geometry_triggered) is not bool
        or type(result.joint_content_positive) is not bool
        or type(result.full_ceg_wm_eligible) is not bool
    ):
        raise ConditionalRecoveryError(
            "joint decision flags must be booleans"
        )
    if (
        result.wrong_key_index is not None
        and (
            type(result.wrong_key_index) is not int
            or result.wrong_key_index < 0
        )
    ):
        raise ConditionalRecoveryError("wrong_key_index is invalid")


def _detector_binding_digest(
    *,
    detector_identity: str,
    content_config_digest: str,
    hf_detector_identity: str,
    hf_detector_config_digest: str,
    hf_template_digest: str,
    preprocessing_identity: str,
    formal_mode: str,
    root_key_public_digest: str,
    key_role: str,
    wrong_key_index: int | None,
) -> str:
    return sha256(
        stable_json_utf8(
            {
                "content_config_digest": content_config_digest,
                "detector_identity": detector_identity,
                "formal_mode": formal_mode,
                "hf_detector_config_digest": hf_detector_config_digest,
                "hf_detector_identity": hf_detector_identity,
                "hf_template_digest": hf_template_digest,
                "key_role": key_role,
                "operation_role": "same_public_image_content_detector",
                "preprocessing_identity": preprocessing_identity,
                "root_key_public_digest": root_key_public_digest,
                "wrong_key_index": wrong_key_index,
            }
        )
    ).hexdigest()


def _threshold_identity(
    *,
    tau: float,
    tau_rescue: float,
    detector_binding_digest: str,
    calibration_identity: str,
) -> str:
    return sha256(
        stable_json_utf8(
            {
                "calibration_identity": calibration_identity,
                "candidate_id": JOINT_CANDIDATE_ID,
                "detector_binding_digest": detector_binding_digest,
                "tau_float64_hex": tau.hex(),
                "tau_rescue_float64_hex": tau_rescue.hex(),
            }
        )
    ).hexdigest()


def _number_identity(value: object) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        return f"invalid:{type(value).__name__}"
    return float(value).hex()


def _content_result_identity_payload(
    result: ContentDetectionResult | None,
) -> dict[str, object] | None:
    if result is None:
        return None
    if type(result) is not ContentDetectionResult:
        return {"invalid_type": type(result).__name__}
    hf = result.hf_result
    lf = result.lf_result
    hf_payload: dict[str, object]
    if type(hf) is not HfDetectionResult:
        hf_payload = {"invalid_type": type(hf).__name__}
    else:
        hf_payload = {
            "candidate_id": hf.candidate_id,
            "detector_config_digest": hf.detector_config_digest,
            "detector_identity": hf.detector_identity,
            "hf_score": _number_identity(hf.hf_score),
            "key_role": hf.key_role,
            "observation_digest": hf.observation_digest,
            "root_key_public_digest": hf.root_key_public_digest,
            "template_digest": hf.template_digest,
            "wrong_key_index": hf.wrong_key_index,
        }
    lf_payload: dict[str, object] | None
    if lf is None:
        lf_payload = None
    elif type(lf) is not LfDetectionResult:
        lf_payload = {"invalid_type": type(lf).__name__}
    else:
        lf_payload = {
            "candidate_id": lf.candidate_id,
            "candidate_ids": list(lf.candidate_ids),
            "detector_config_digest": lf.detector_config_digest,
            "detector_identity": lf.detector_identity,
            "key_role": lf.key_role,
            "lf_score": _number_identity(lf.lf_score),
            "observation_digest": lf.observation_digest,
            "root_key_public_digest": lf.root_key_public_digest,
            "template_digest": lf.template_digest,
            "wrong_key_index": lf.wrong_key_index,
        }
    return {
        "candidate_ids": list(result.candidate_ids),
        "combined_score": (
            None
            if result.combined_score is None
            else _number_identity(result.combined_score)
        ),
        "content_config_digest": result.content_config_digest,
        "content_input_image_digest": result.content_input_image_digest,
        "content_score": _number_identity(result.content_score),
        "detector_identity": result.detector_identity,
        "diagnostic_identity": result.diagnostic_identity,
        "formal_mode": result.formal_mode,
        "hf_result": hf_payload,
        "hf_score": _number_identity(result.hf_score),
        "lf_result": lf_payload,
        "lf_score": (
            None
            if result.lf_score is None
            else _number_identity(result.lf_score)
        ),
    }


def _tensor_payload(tensor: object) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        return {"invalid_type": type(tensor).__name__}
    detached = tensor.detach().to(device="cpu").contiguous()
    byte_tensor = (
        detached.to(dtype=torch.uint8)
        if detached.dtype is torch.bool
        else detached.view(torch.uint8)
    )
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "values_sha256": sha256(
            bytes(byte_tensor.reshape(-1).tolist())
        ).hexdigest(),
    }


def _rectification_identity_payload(
    result: ImageRectificationResult | None,
) -> dict[str, object] | None:
    if result is None:
        return None
    if type(result) is not ImageRectificationResult:
        return {"invalid_type": type(result).__name__}
    try:
        matrix_payload = [
            [_number_identity(value) for value in row]
            for row in result.canonical_to_observed_matrix
        ]
        crop_payload = [
            _number_identity(value) for value in result.crop_support
        ]
    except TypeError:
        matrix_payload = [{"invalid_type": "matrix"}]
        crop_payload = ["invalid:crop_support"]
    return {
        "canonical_to_observed_matrix": matrix_payload,
        "crop_support": crop_payload,
        "pixel_crop_support": _number_identity(
            result.pixel_crop_support
        ),
        "rectification_config_digest": result.rectification_config_digest,
        "rectified_image_digest": result.rectified_image_digest,
        "rectified_image": _tensor_payload(result.rectified_image),
        "source_image_digest": result.source_image_digest,
        "token_crop_support": _number_identity(result.token_crop_support),
        "valid_support_mask": _tensor_payload(result.valid_support_mask),
    }


def _result_identity_attribute(
    value: object,
    expected_type: type,
    attribute: str,
) -> str | None:
    if value is None:
        return None
    if type(value) is not expected_type:
        return f"invalid:{type(value).__name__}"
    identity = getattr(value, attribute, None)
    return (
        identity
        if type(identity) is str and identity
        else f"invalid:{type(identity).__name__}"
    )


def _decision_identity_digest(result: ConditionalRecoveryResult) -> str:
    identity = {
        "calibration_identity": result.calibration_identity,
        "candidate_id": result.candidate_id,
        "content_config_digest": result.content_config_digest,
        "detector_binding_digest": result.detector_binding_digest,
        "detector_identity": result.detector_identity,
        "failure_reason": result.failure_reason,
        "joint_content_positive": result.joint_content_positive,
        "formal_mode": result.formal_mode,
        "full_ceg_wm_eligible": result.full_ceg_wm_eligible,
        "hf_detector_config_digest": result.hf_detector_config_digest,
        "hf_detector_identity": result.hf_detector_identity,
        "hf_template_digest": result.hf_template_digest,
        "geometry_estimation_identity": _result_identity_attribute(
            result.geometry_estimation,
            GeometricTransformEstimation,
            "estimation_identity_digest",
        ),
        "geometry_reliability_identity": _result_identity_attribute(
            result.geometry_reliability_result,
            GeometryReliabilityResult,
            "reliability_identity_digest",
        ),
        "geometry_triggered": result.geometry_triggered,
        "image_rectification_result": _rectification_identity_payload(
            result.image_rectification_result
        ),
        "key_role": result.key_role,
        "positive_source": result.positive_source,
        "positive_path": result.positive_path,
        "preprocessing_identity": result.preprocessing_identity,
        "raw_content_result": _content_result_identity_payload(
            result.raw_content_result
        ),
        "rectified_content_result": _content_result_identity_payload(
            result.rectified_content_result
        ),
        "root_key_public_digest": result.root_key_public_digest,
        "source_image": _tensor_payload(result.source_image),
        "source_image_digest": result.source_image_digest,
        "status": result.status,
        "tau_float64_hex": _number_identity(result.tau),
        "tau_rescue_float64_hex": _number_identity(result.tau_rescue),
        "threshold_identity": result.threshold_identity,
        "trigger_reason": result.trigger_reason,
        "wrong_key_index": result.wrong_key_index,
    }
    return sha256(stable_json_utf8(identity)).hexdigest()


def _validate_bound_content_result(
    result: ContentDetectionResult,
    *,
    actual_image: torch.Tensor,
    detector_identity: str,
    content_config_digest: str,
    hf_detector_identity: str,
    hf_detector_config_digest: str,
    hf_template_digest: str,
    formal_mode: str,
    root_key_public_digest: str,
) -> ContentDetectionResult:
    try:
        validate_content_detection_result(result, actual_image)
    except ContentDetectorError as exc:
        raise ConditionalRecoveryError(
            "content result validation failed"
        ) from exc
    if (
        result.detector_identity != detector_identity
        or result.content_config_digest != content_config_digest
        or result.formal_mode != formal_mode
        or result.hf_result.detector_identity != hf_detector_identity
        or result.hf_result.detector_config_digest
        != hf_detector_config_digest
        or result.hf_result.template_digest != hf_template_digest
    ):
        raise ConditionalRecoveryError(
            "content result does not match the immutable detector binding"
        )
    if (
        result.hf_result.root_key_public_digest != root_key_public_digest
        or result.hf_result.key_role != "registered"
        or result.hf_result.wrong_key_index is not None
    ):
        raise ConditionalRecoveryError(
            "content result does not match registered root-key semantics"
        )
    return result


def _failure_name(exc: Exception) -> str:
    return type(exc).__name__


def _result(
    *,
    binding: ContentDetectorBinding,
    thresholds: JointDecisionThresholds,
    source_image: torch.Tensor,
    root_key_public_digest: str,
    raw_content_result: ContentDetectionResult | None,
    geometry_triggered: bool,
    trigger_reason: str,
    geometry_estimation: GeometricTransformEstimation | None = None,
    geometry_reliability_result: GeometryReliabilityResult | None = None,
    image_rectification_result: ImageRectificationResult | None = None,
    rectified_content_result: ContentDetectionResult | None = None,
    joint_content_positive: bool = False,
    positive_source: str | None = None,
    status: str,
    failure_reason: str | None = None,
) -> ConditionalRecoveryResult:
    positive_path = (
        "raw_positive"
        if positive_source == "raw_content"
        else "rescue_positive"
        if positive_source == "rectified_content"
        else None
    )
    return ConditionalRecoveryResult(
        candidate_id=JOINT_CANDIDATE_ID,
        formal_mode=binding.formal_mode,
        full_ceg_wm_eligible=False,
        source_image=source_image,
        source_image_digest=rgb8_image_digest(source_image),
        raw_content_result=raw_content_result,
        raw_content_score=(
            float(raw_content_result.content_score)
            if raw_content_result is not None
            else None
        ),
        geometry_triggered=geometry_triggered,
        trigger_reason=trigger_reason,
        geometry_estimation=geometry_estimation,
        geometry_reliability_result=geometry_reliability_result,
        image_rectification_result=image_rectification_result,
        rectified_content_result=rectified_content_result,
        rectified_content_score=(
            float(rectified_content_result.content_score)
            if rectified_content_result is not None
            else None
        ),
        joint_content_positive=joint_content_positive,
        positive_source=positive_source,
        positive_path=positive_path,
        status=status,
        failure_reason=failure_reason,
        detector_identity=binding.detector_identity,
        content_config_digest=binding.content_config_digest,
        hf_detector_identity=binding.hf_detector_identity,
        hf_detector_config_digest=binding.hf_detector_config_digest,
        hf_template_digest=binding.hf_template_digest,
        preprocessing_identity=binding.preprocessing_identity,
        detector_binding_digest=binding.detector_binding_digest,
        tau=thresholds.tau,
        tau_rescue=thresholds.tau_rescue,
        threshold_identity=thresholds.threshold_identity,
        calibration_identity=thresholds.calibration_identity,
        root_key_public_digest=root_key_public_digest,
        key_role="registered",
        wrong_key_index=None,
    )


def conditional_recovery_decision(
    image: torch.Tensor,
    detection_key: str,
    *,
    content_detector_binding: ContentDetectorBinding,
    thresholds: JointDecisionThresholds,
    geometry_estimation_operation: GeometryEstimationOperation,
    geometry_reliability_thresholds: GeometryReliabilityThresholds | None,
) -> ConditionalRecoveryResult:
    """Execute the frozen raw/near-threshold/recheck flow, fail closed."""

    try:
        source_image = clone_rgb8_image(image)
    except Rgb8ImageError as exc:
        raise ConditionalRecoveryError(
            "image must be RGB uint8 [1,3,H,W] with H,W > 1"
        ) from exc
    if type(content_detector_binding) is not ContentDetectorBinding:
        raise ConditionalRecoveryError(
            "content_detector_binding must be ContentDetectorBinding"
        )
    if type(thresholds) is not JointDecisionThresholds:
        raise ConditionalRecoveryError(
            "thresholds must be JointDecisionThresholds"
        )
    if thresholds.detector_binding_digest != (
        content_detector_binding.detector_binding_digest
    ):
        raise ConditionalRecoveryError(
            "thresholds do not match the detector binding"
        )
    if not callable(geometry_estimation_operation):
        raise ConditionalRecoveryError(
            "geometry estimation operation must be callable"
        )
    try:
        root_identity = identify_root_key(detection_key)
    except KeyScheduleError as exc:
        raise ConditionalRecoveryError(
            "detection key violates shared root-key semantics"
        ) from exc
    root_digest = root_identity.root_key_public_digest
    if root_digest != content_detector_binding.root_key_public_digest:
        raise ConditionalRecoveryError(
            "detection key does not match the detector binding"
        )

    try:
        raw_result = content_detector_binding.content_detection_operation(
            clone_rgb8_image(source_image),
            detection_key,
        )
    except (
        ContentDetectorError,
        HfDetectorError,
        JointOperationError,
        KeyScheduleError,
        LfDetectorError,
    ) as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=None,
            geometry_triggered=False,
            trigger_reason="raw_content_operation_failure",
            status="raw_content_operation_failure",
            failure_reason=_failure_name(exc),
        )
    try:
        _validate_bound_content_result(
            raw_result,
            actual_image=source_image,
            detector_identity=content_detector_binding.detector_identity,
            content_config_digest=content_detector_binding.content_config_digest,
            hf_detector_identity=content_detector_binding.hf_detector_identity,
            hf_detector_config_digest=(
                content_detector_binding.hf_detector_config_digest
            ),
            hf_template_digest=content_detector_binding.hf_template_digest,
            formal_mode=content_detector_binding.formal_mode,
            root_key_public_digest=root_digest,
        )
    except ConditionalRecoveryError as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=None,
            geometry_triggered=False,
            trigger_reason="raw_content_identity_failure",
            status="raw_content_identity_failure",
            failure_reason=_failure_name(exc),
        )

    raw_score = float(raw_result.content_score)
    if raw_score >= thresholds.tau:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=False,
            trigger_reason="raw_content_at_or_above_tau",
            joint_content_positive=True,
            positive_source="raw_content",
            status="raw_positive",
        )
    if raw_score < thresholds.tau_rescue:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=False,
            trigger_reason="raw_content_below_tau_rescue",
            status="negative_without_geometry",
        )

    try:
        estimation = geometry_estimation_operation(
            clone_rgb8_image(source_image),
            detection_key,
        )
    except (
        GeometricTransformEstimatorError,
        JointOperationError,
        QkGeometrySyncError,
    ) as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            status="negative_geometry_operation_failure",
            failure_reason=_failure_name(exc),
        )
    try:
        validate_geometric_transform_estimation(estimation)
        if estimation.registered_root_key_public_digest != root_digest:
            raise ConditionalRecoveryError(
                "geometry estimation root-key identity mismatch"
            )
    except (ConditionalRecoveryError, GeometricTransformEstimatorError) as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            status="negative_geometry_identity_failure",
            failure_reason=_failure_name(exc),
        )

    try:
        reliability = geometry_reliability(
            estimation,
            geometry_reliability_thresholds,
        )
        reliable = validate_geometry_reliability_result(
            reliability,
            estimation,
        )
    except (GeometryReliabilityError, GeometricTransformEstimatorError) as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            geometry_estimation=estimation,
            status="negative_geometry_reliability_failure",
            failure_reason=_failure_name(exc),
        )
    if not reliable:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            geometry_estimation=estimation,
            geometry_reliability_result=reliability,
            status="negative_geometry_unreliable",
            failure_reason="geometry_unreliable",
        )

    try:
        rectification = image_rectifier(
            source_image,
            estimation,
            reliability,
        )
    except ImageRectifierError as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            geometry_estimation=estimation,
            geometry_reliability_result=reliability,
            status="negative_rectification_failure",
            failure_reason=_failure_name(exc),
        )

    try:
        rectified_result = content_detector_binding.content_detection_operation(
            clone_rgb8_image(rectification.rectified_image),
            detection_key,
        )
    except (
        ContentDetectorError,
        HfDetectorError,
        JointOperationError,
        KeyScheduleError,
        LfDetectorError,
    ) as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            geometry_estimation=estimation,
            geometry_reliability_result=reliability,
            image_rectification_result=rectification,
            status="negative_rectified_content_operation_failure",
            failure_reason=_failure_name(exc),
        )
    try:
        _validate_bound_content_result(
            rectified_result,
            actual_image=rectification.rectified_image,
            detector_identity=content_detector_binding.detector_identity,
            content_config_digest=content_detector_binding.content_config_digest,
            hf_detector_identity=content_detector_binding.hf_detector_identity,
            hf_detector_config_digest=(
                content_detector_binding.hf_detector_config_digest
            ),
            hf_template_digest=content_detector_binding.hf_template_digest,
            formal_mode=content_detector_binding.formal_mode,
            root_key_public_digest=root_digest,
        )
    except ConditionalRecoveryError as exc:
        return _result(
            binding=content_detector_binding,
            thresholds=thresholds,
            source_image=source_image,
            root_key_public_digest=root_digest,
            raw_content_result=raw_result,
            geometry_triggered=True,
            trigger_reason="raw_content_in_rescue_window",
            geometry_estimation=estimation,
            geometry_reliability_result=reliability,
            image_rectification_result=rectification,
            status="negative_rectified_content_identity_failure",
            failure_reason=_failure_name(exc),
        )

    rectified_positive = (
        float(rectified_result.content_score) >= thresholds.tau
    )
    return _result(
        binding=content_detector_binding,
        thresholds=thresholds,
        source_image=source_image,
        root_key_public_digest=root_digest,
        raw_content_result=raw_result,
        geometry_triggered=True,
        trigger_reason="raw_content_in_rescue_window",
        geometry_estimation=estimation,
        geometry_reliability_result=reliability,
        image_rectification_result=rectification,
        rectified_content_result=rectified_result,
        joint_content_positive=rectified_positive,
        positive_source=(
            "rectified_content" if rectified_positive else None
        ),
        status=(
            "rectified_positive"
            if rectified_positive
            else "rectified_content_negative"
        ),
    )


def _validate_content_result_identity(
    result: ConditionalRecoveryResult,
    content_result: ContentDetectionResult,
    actual_image: torch.Tensor,
) -> None:
    _validate_bound_content_result(
        content_result,
        actual_image=actual_image,
        detector_identity=result.detector_identity,
        content_config_digest=result.content_config_digest,
        hf_detector_identity=result.hf_detector_identity,
        hf_detector_config_digest=result.hf_detector_config_digest,
        hf_template_digest=result.hf_template_digest,
        formal_mode=result.formal_mode,
        root_key_public_digest=result.root_key_public_digest,
    )


def validate_conditional_recovery_result(
    result: ConditionalRecoveryResult,
) -> ConditionalRecoveryResult:
    """Replay path, threshold, evidence-source, and identity consistency."""

    if type(result) is not ConditionalRecoveryResult:
        raise ConditionalRecoveryError(
            "result must be ConditionalRecoveryResult"
        )
    _validate_conditional_result_types(result)
    if result.candidate_id != JOINT_CANDIDATE_ID:
        raise ConditionalRecoveryError("joint candidate identity mismatch")
    if result.formal_mode != "hf_only" or result.full_ceg_wm_eligible:
        raise ConditionalRecoveryError(
            "HF-only construction candidate cannot claim full CEG-WM eligibility"
        )
    if result.key_role != "registered" or result.wrong_key_index is not None:
        raise ConditionalRecoveryError(
            "joint result key semantics must remain registered-root"
        )
    expected_binding_digest = _detector_binding_digest(
        detector_identity=result.detector_identity,
        content_config_digest=result.content_config_digest,
        hf_detector_identity=result.hf_detector_identity,
        hf_detector_config_digest=result.hf_detector_config_digest,
        hf_template_digest=result.hf_template_digest,
        preprocessing_identity=result.preprocessing_identity,
        formal_mode=result.formal_mode,
        root_key_public_digest=result.root_key_public_digest,
        key_role=result.key_role,
        wrong_key_index=result.wrong_key_index,
    )
    if result.detector_binding_digest != expected_binding_digest:
        raise ConditionalRecoveryError("detector binding digest mismatch")
    expected_threshold_identity = _threshold_identity(
        tau=result.tau,
        tau_rescue=result.tau_rescue,
        detector_binding_digest=result.detector_binding_digest,
        calibration_identity=result.calibration_identity,
    )
    if (
        result.tau_rescue >= result.tau
        or result.threshold_identity != expected_threshold_identity
    ):
        raise ConditionalRecoveryError("joint threshold identity mismatch")

    expected_state = _STATUS_STATE_MACHINE.get(result.status)
    if expected_state is None:
        raise ConditionalRecoveryError("unknown joint decision status")
    (
        expected_trigger,
        expect_raw,
        expect_geometry_triggered,
        expect_estimation,
        expect_reliability,
        expect_rectification,
        expect_rectified,
        expect_positive,
        expected_positive_source,
        expected_positive_path,
        expect_failure_reason,
    ) = expected_state
    actual_state = (
        result.trigger_reason,
        result.raw_content_result is not None,
        result.geometry_triggered,
        result.geometry_estimation is not None,
        result.geometry_reliability_result is not None,
        result.image_rectification_result is not None,
        result.rectified_content_result is not None,
        result.joint_content_positive,
        result.positive_source,
        result.positive_path,
    )
    expected_actual_state = (
        expected_trigger,
        expect_raw,
        expect_geometry_triggered,
        expect_estimation,
        expect_reliability,
        expect_rectification,
        expect_rectified,
        expect_positive,
        expected_positive_source,
        expected_positive_path,
    )
    if actual_state != expected_actual_state:
        raise ConditionalRecoveryError(
            "joint status state-machine fields are inconsistent"
        )
    if expect_failure_reason:
        if type(result.failure_reason) is not str or not result.failure_reason:
            raise ConditionalRecoveryError(
                "joint failure status requires a non-empty failure reason"
            )
    elif result.failure_reason is not None:
        raise ConditionalRecoveryError(
            "successful decision path must not carry a failure reason"
        )

    raw = result.raw_content_result
    rectified = result.rectified_content_result
    if raw is not None:
        _validate_content_result_identity(
            result,
            raw,
            result.source_image,
        )
        if result.raw_content_score != float(raw.content_score):
            raise ConditionalRecoveryError("raw content score mismatch")
    elif result.raw_content_score is not None:
        raise ConditionalRecoveryError(
            "raw content score requires a raw content result"
        )

    if result.geometry_triggered:
        if (
            raw is None
            or not result.tau_rescue
            <= float(raw.content_score)
            < result.tau
            or result.trigger_reason != "raw_content_in_rescue_window"
        ):
            raise ConditionalRecoveryError(
                "geometry may run only for a near-threshold raw negative"
            )
    elif (
        result.geometry_estimation is not None
        or result.geometry_reliability_result is not None
        or result.image_rectification_result is not None
        or rectified is not None
    ):
        raise ConditionalRecoveryError(
            "non-triggered path must not carry geometry or rectified results"
        )

    replayed_reliable = False
    if result.geometry_estimation is not None:
        try:
            validate_geometric_transform_estimation(
                result.geometry_estimation
            )
        except GeometricTransformEstimatorError as exc:
            raise ConditionalRecoveryError(
                "joint geometry estimation validation failed"
            ) from exc
        if (
            result.geometry_estimation.registered_root_key_public_digest
            != result.root_key_public_digest
        ):
            raise ConditionalRecoveryError(
                "joint geometry estimation key identity mismatch"
            )
    if result.geometry_reliability_result is not None:
        if result.geometry_estimation is None:
            raise ConditionalRecoveryError(
                "geometry reliability requires an estimation"
            )
        try:
            replayed_reliable = validate_geometry_reliability_result(
                result.geometry_reliability_result,
                result.geometry_estimation,
            )
        except GeometryReliabilityError as exc:
            raise ConditionalRecoveryError(
                "joint geometry reliability validation failed"
            ) from exc
    if result.image_rectification_result is not None:
        if not replayed_reliable or result.geometry_estimation is None:
            raise ConditionalRecoveryError(
                "rectification requires reliable geometry"
            )
        try:
            validate_image_rectification_result(
                result.image_rectification_result,
                result.source_image,
                result.geometry_estimation,
                result.geometry_reliability_result,
            )
        except ImageRectifierError as exc:
            raise ConditionalRecoveryError(
                "joint rectification result validation failed"
            ) from exc
    if rectified is not None:
        if result.image_rectification_result is None:
            raise ConditionalRecoveryError(
                "rectified content result requires image rectification"
            )
        _validate_content_result_identity(
            result,
            rectified,
            result.image_rectification_result.rectified_image,
        )
        if result.rectified_content_score != float(rectified.content_score):
            raise ConditionalRecoveryError("rectified content score mismatch")
    elif result.rectified_content_score is not None:
        raise ConditionalRecoveryError(
            "rectified score requires a rectified content result"
        )

    if result.status == "negative_without_geometry":
        if (
            raw is None
            or float(raw.content_score) >= result.tau_rescue
            or result.geometry_triggered
        ):
            raise ConditionalRecoveryError(
                "negative-without-geometry path is inconsistent"
            )
    elif result.status == "raw_positive":
        if raw is None or float(raw.content_score) < result.tau:
            raise ConditionalRecoveryError(
                "raw positive path is inconsistent"
            )
    if result.status == "negative_geometry_unreliable":
        if (
            result.geometry_reliability_result is None
            or replayed_reliable
            or result.image_rectification_result is not None
        ):
            raise ConditionalRecoveryError(
                "unreliable geometry path is inconsistent"
            )
    if result.status in {
        "negative_rectification_failure",
        "negative_rectified_content_operation_failure",
        "negative_rectified_content_identity_failure",
        "rectified_positive",
        "rectified_content_negative",
    } and not replayed_reliable:
        raise ConditionalRecoveryError(
            "post-reliability path requires reliable geometry"
        )
    if result.status == "rectified_positive":
        if rectified is None or float(rectified.content_score) < result.tau:
            raise ConditionalRecoveryError(
                "rectified positive path is inconsistent"
            )
    if result.status == "rectified_content_negative":
        if (
            rectified is None
            or float(rectified.content_score) >= result.tau
            or result.joint_content_positive
        ):
            raise ConditionalRecoveryError(
                "rectified negative path is inconsistent"
            )
    if result.decision_identity_digest != _decision_identity_digest(result):
        raise ConditionalRecoveryError(
            "joint decision identity digest mismatch"
        )
    return result


__all__ = [
    "ConditionalRecoveryError",
    "ConditionalRecoveryResult",
    "ContentDetectionOperation",
    "ContentDetectorBinding",
    "GeometryEstimationOperation",
    "JointOperationError",
    "JointDecisionThresholds",
    "conditional_recovery_decision",
    "validate_conditional_recovery_result",
]
