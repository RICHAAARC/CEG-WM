"""CPU/synthetic checks for conditional same-detector recovery."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
import torch
import torch.nn.functional as functional

from main.content_chain import (
    ContentDetectorError,
    HfDetectionObservation,
    validate_content_detection_result,
)
from main.content_chain.detector import content_detector
from main.content_chain.hf_detector import hf_detector
from main.geometry_chain import (
    GeometricTransformEstimation,
    GeometryReliabilityThresholds,
    SimilarityTransform,
)
from main.geometry_chain.rectifier import image_rectifier
from main.geometry_chain.reliability import geometry_reliability
from main.joint_decision import (
    ConditionalRecoveryError,
    ContentDetectorBinding,
    JointOperationError,
    JointDecisionThresholds,
    validate_conditional_recovery_result,
)
from main.joint_decision.detector import conditional_recovery_decision
from main.shared import identify_root_key, rgb8_image_digest, stable_json_utf8

_ROOT_KEY = "joint-decision-cpu-key"
_SHAPE = (1, 3, 9, 9)


class _ActualContentOperation:
    """Map RGB8 values to the real HF detector and formal content detector."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor]] = []

    def _detect(
        self,
        image: torch.Tensor,
        detection_key: str,
    ):
        observation = HfDetectionObservation.from_public_image_encoding(
            tuple((image.to(dtype=torch.float32) / 255.0).reshape(-1).tolist()),
            tuple(image.shape),
        )
        return replace(
            content_detector(hf_detector(observation, detection_key)),
            content_input_image_digest=rgb8_image_digest(image),
            content_replay_operation=self,
        )

    def __call__(self, image: torch.Tensor, detection_key: str):
        self.calls.append((id(self), image.detach().clone()))
        return self._detect(image, detection_key)

    def replay_validate_content_result(
        self,
        result,
        input_image: object,
        detection_key: str,
    ):
        if not isinstance(input_image, torch.Tensor):
            raise ContentDetectorError(
                "CPU content replay requires an RGB8 tensor"
            )
        expected = self._detect(input_image, detection_key)
        validate_content_detection_result(expected)
        if result != expected:
            raise ContentDetectorError(
                "CPU image-to-observation-to-score replay mismatch"
            )
        return result


class _GeometryOperation:
    def __init__(self, estimation: GeometricTransformEstimation) -> None:
        self.estimation = estimation
        self.calls = 0

    def __call__(self, image: torch.Tensor, registered_key: str):
        assert image.dtype is torch.uint8
        assert registered_key == _ROOT_KEY
        self.calls += 1
        return self.estimation


def _estimation(translation_x: float = 0.0) -> GeometricTransformEstimation:
    translation_float32 = float(
        torch.tensor(translation_x, dtype=torch.float32)
    )
    is_identity = translation_x == 0.0
    transform = SimilarityTransform(
        dihedral="identity",
        residual_rotation_degrees=0.0,
        log_scale=0.0,
        translation_x=float(translation_x),
        translation_y=0.0,
        matrix=(
            (1.0, 0.0, translation_float32),
            (0.0, 1.0, 0.0),
        ),
        is_exact_identity=is_identity,
        continuous_parameter_on_search_boundary=False,
    )
    exact_identity_objective = 1.0 if is_identity else 0.75
    return GeometricTransformEstimation(
        candidate_ids=(
            "key_schedule_sha256_counter",
            "qk_relation_similarity",
            "rectification_similarity",
        ),
        transform=transform,
        registered_objective=1.0,
        second_registered_objective=0.8,
        exact_identity_objective=exact_identity_objective,
        wrong_key_objectives=tuple(0.2 for _ in range(8)),
        canonical_score=1.0,
        observation_score=1.0,
        coverage_forward=1.0,
        coverage_backward=1.0,
        uniqueness_forward=1.0,
        uniqueness_backward=1.0,
        coverage=1.0,
        uniqueness=1.0,
        gap=0.2,
        identity_margin=1.0 - exact_identity_objective,
        key_margin=0.8,
        inlier_ratio=1.0,
        mean_residual=0.0,
        epsilon_inlier=0.8,
        anchor_residuals=tuple(0.0 for _ in range(12)),
        registered_root_key_public_digest=identify_root_key(
            _ROOT_KEY
        ).root_key_public_digest,
        observation_descriptor_digest="1" * 64,
        observation_projection_digest="2" * 64,
        observation_geometry_config_digest="3" * 64,
        search_config_digest=sha256(
            stable_json_utf8(
                {
                    "candidate_id": "rectification_similarity",
                    "coarse_log_scale": ["0", "-log_sqrt2", "+log_sqrt2"],
                    "coarse_rotation_degrees": [0, -32, -16, 16, 32],
                    "coarse_translation": ["0", "-0.28", "+0.28"],
                    "dihedral_order": [
                        "identity",
                        "x_flip",
                        "y_flip",
                        "xy_flip",
                        "rot90",
                        "rot_minus90",
                        "diag",
                        "anti_diag",
                    ],
                    "epsilon_inlier_decimal": format(0.8, ".17g"),
                    "objective_weights": ["0.10", "0.90", "-0.01_deficits"],
                    "refinement_rounds": 3,
                    "wrong_key_indices": list(range(8)),
                    "refinement_strategy": (
                        "joint_greedy_with_axis_isolated_safeguards_v2"
                    ),
                    "axis_safeguard_order": [
                        "rotation_degrees",
                        "log_scale",
                        "translation_x",
                        "translation_y",
                    ],
                    "axis_safeguard_initialization": "coarse_selected",
                    "candidate_matrix_protocol": (
                        "float64_parameter_math_then_single_float32_cast"
                    ),
                }
            )
        ).hexdigest(),
    )


def _reliability_thresholds(
    *,
    gap: float = 0.05,
) -> GeometryReliabilityThresholds:
    return GeometryReliabilityThresholds(
        gamma_coverage=0.45,
        gamma_uniqueness=0.5,
        gamma_gap=gap,
        gamma_key=0.1,
        gamma_inlier=0.5,
        gamma_residual=0.5,
        gamma_identity=0.05,
        epsilon_inlier=0.8,
        fit_identity="joint_geometry_reliability_fit_cpu_synthetic",
    )


def _binding(
    operation: _ActualContentOperation,
    image: torch.Tensor,
) -> ContentDetectorBinding:
    prototype = operation(image, _ROOT_KEY)
    operation.calls.clear()
    return ContentDetectorBinding(
        content_detection_operation=operation,
        detector_identity=prototype.detector_identity,
        content_config_digest=prototype.content_config_digest,
        hf_detector_identity=prototype.hf_result.detector_identity,
        hf_detector_config_digest=prototype.hf_result.detector_config_digest,
        hf_template_digest=prototype.hf_result.template_digest,
        preprocessing_identity=(
            "cpu_synthetic_rgb8_as_image_vae_mode_encoding"
        ),
        formal_mode=prototype.formal_mode,
        root_key_public_digest=prototype.hf_result.root_key_public_digest,
        key_role=prototype.hf_result.key_role,
        wrong_key_index=prototype.hf_result.wrong_key_index,
    )


def _thresholds(
    binding: ContentDetectorBinding,
    *,
    tau: float,
    tau_rescue: float,
    calibration_identity: str,
) -> JointDecisionThresholds:
    return JointDecisionThresholds(
        tau=tau,
        tau_rescue=tau_rescue,
        detector_binding_digest=binding.detector_binding_digest,
        calibration_identity=calibration_identity,
    )


def _null_and_translated_image() -> tuple[torch.Tensor, torch.Tensor]:
    direct_generator = torch.Generator().manual_seed(0)
    direct_null = torch.randint(
        0,
        256,
        _SHAPE,
        dtype=torch.uint8,
        generator=direct_generator,
    )
    rescue_generator = torch.Generator().manual_seed(2)
    rescue_canonical = torch.randint(
        0,
        256,
        _SHAPE,
        dtype=torch.uint8,
        generator=rescue_generator,
    )
    inverse_translation = torch.tensor(
        ((1.0, 0.0, -0.2), (0.0, 1.0, 0.0)),
        dtype=torch.float32,
    )
    grid = functional.affine_grid(
        inverse_translation.unsqueeze(0),
        rescue_canonical.shape,
        align_corners=True,
    )
    observed = torch.floor(
        torch.clamp(
            functional.grid_sample(
                rescue_canonical.to(dtype=torch.float32) / 255.0,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ),
            0.0,
            1.0,
        )
        * 255.0
    ).to(dtype=torch.uint8)
    return direct_null, observed


@pytest.mark.unit
def test_near_threshold_recovery_gate() -> None:
    image = torch.arange(3 * 9 * 9, dtype=torch.uint8).reshape(_SHAPE)
    operation = _ActualContentOperation()
    binding = _binding(operation, image)
    raw_score = operation(image, _ROOT_KEY).content_score
    operation.calls.clear()
    geometry = _GeometryOperation(_estimation())

    raw_positive = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score,
            tau_rescue=raw_score - 0.11,
            calibration_identity="raw-positive-calibration",
        ),
        geometry_estimation_operation=geometry,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    far_negative = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score + 0.2,
            tau_rescue=raw_score + 0.1,
            calibration_identity="far-negative-calibration",
        ),
        geometry_estimation_operation=geometry,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    near_negative = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score + 0.1,
            tau_rescue=raw_score,
            calibration_identity="near-negative-calibration",
        ),
        geometry_estimation_operation=geometry,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )

    assert raw_positive.status == "raw_positive"
    assert raw_positive.raw_content_score == raw_positive.tau
    assert far_negative.status == "negative_without_geometry"
    assert near_negative.status == "rectified_content_negative"
    assert near_negative.raw_content_score == near_negative.tau_rescue
    assert not raw_positive.geometry_triggered
    assert not far_negative.geometry_triggered
    assert near_negative.geometry_triggered
    assert geometry.calls == 1
    assert len(operation.calls) == 4
    assert {call[0] for call in operation.calls} == {id(operation)}
    for invalid_tau, invalid_rescue in (
        (True, raw_score - 0.1),
        (float("nan"), raw_score - 0.1),
        (raw_score + 0.1, True),
        (raw_score + 0.1, float("nan")),
        (raw_score + 0.1, raw_score + 0.1),
    ):
        with pytest.raises(ConditionalRecoveryError):
            JointDecisionThresholds(
                tau=invalid_tau,
                tau_rescue=invalid_rescue,
                detector_binding_digest=binding.detector_binding_digest,
                calibration_identity="invalid-threshold-calibration",
            )

    def programming_error(
        image: torch.Tensor,
        registered_key: str,
    ) -> GeometricTransformEstimation:
        raise AssertionError("programming error must not become a method negative")

    with pytest.raises(AssertionError, match="must not become"):
        conditional_recovery_decision(
            image,
            _ROOT_KEY,
            content_detector_binding=binding,
            thresholds=_thresholds(
                binding,
                tau=raw_score + 0.1,
                tau_rescue=raw_score - 0.1,
                calibration_identity="programming-error-calibration",
            ),
            geometry_estimation_operation=programming_error,
            geometry_reliability_thresholds=_reliability_thresholds(),
        )

    def declared_operation_failure(
        image: torch.Tensor,
        registered_key: str,
    ) -> GeometricTransformEstimation:
        raise JointOperationError("declared geometry operation failure")

    declared_failure = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score + 0.1,
            tau_rescue=raw_score,
            calibration_identity="declared-operation-failure-calibration",
        ),
        geometry_estimation_operation=declared_operation_failure,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    assert declared_failure.status == "negative_geometry_operation_failure"
    assert declared_failure.failure_reason == "JointOperationError"


@pytest.mark.unit
def test_geometry_no_direct_positive() -> None:
    image = torch.arange(3 * 9 * 9, dtype=torch.uint8).reshape(_SHAPE)
    operation = _ActualContentOperation()
    binding = _binding(operation, image)
    raw_score = operation(image, _ROOT_KEY).content_score
    operation.calls.clear()
    estimation = _estimation()
    reliability_thresholds = _reliability_thresholds(gap=0.3)
    directly_unreliable = geometry_reliability(
        estimation,
        reliability_thresholds,
    )
    assert not directly_unreliable.reliable
    assert not directly_unreliable.allow_rectification
    geometry = _GeometryOperation(estimation)
    result = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score + 0.1,
            tau_rescue=raw_score - 0.1,
            calibration_identity="unreliable-geometry-calibration",
        ),
        geometry_estimation_operation=geometry,
        geometry_reliability_thresholds=reliability_thresholds,
    )

    assert result.status == "negative_geometry_unreliable"
    assert result.geometry_reliability_result is not None
    assert not result.geometry_reliability_result.reliable
    assert result.geometry_estimation is not None
    assert result.geometry_estimation.registered_objective == pytest.approx(1.0)
    assert not result.joint_content_positive
    assert result.positive_source is None
    assert result.positive_path is None
    assert result.formal_mode == "hf_only"
    assert not result.full_ceg_wm_eligible
    assert result.image_rectification_result is None
    assert len(operation.calls) == 1

    reliable_geometry = _GeometryOperation(_estimation())
    reliable_but_content_negative = conditional_recovery_decision(
        image,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=raw_score + 0.1,
            tau_rescue=raw_score,
            calibration_identity="reliable-content-negative-calibration",
        ),
        geometry_estimation_operation=reliable_geometry,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    assert reliable_but_content_negative.status == (
        "rectified_content_negative"
    )
    assert reliable_but_content_negative.geometry_reliability_result.reliable
    assert not reliable_but_content_negative.joint_content_positive
    assert reliable_but_content_negative.positive_source is None


@pytest.mark.unit
def test_joint_same_detector_threshold() -> None:
    direct_null, rescue_null = _null_and_translated_image()
    operation = _ActualContentOperation()
    binding = _binding(operation, direct_null)
    raw_score = operation(rescue_null, _ROOT_KEY).content_score
    foreign_content_result = operation(direct_null, _ROOT_KEY)
    direct_score = foreign_content_result.content_score
    replayed_observation = HfDetectionObservation.from_public_image_encoding(
        tuple(
            (rescue_null.to(dtype=torch.float32) / 255.0)
            .reshape(-1)
            .tolist()
        ),
        tuple(rescue_null.shape),
    )
    replayed_content = content_detector(
        hf_detector(replayed_observation, _ROOT_KEY)
    )
    assert replayed_content.content_score == pytest.approx(raw_score)
    operation.calls.clear()
    assert direct_score > raw_score
    tau = (raw_score + direct_score) / 2.0
    coordinated_foreign_content = replace(
        foreign_content_result,
        content_input_image_digest=rgb8_image_digest(rescue_null),
    )
    with pytest.raises(
        ContentDetectorError,
        match="requires both input_image and detection_key",
    ):
        validate_content_detection_result(
            coordinated_foreign_content,
            rescue_null,
        )
    with pytest.raises(
        ContentDetectorError,
        match="image-to-observation-to-score replay",
    ):
        validate_content_detection_result(
            coordinated_foreign_content,
            rescue_null,
            _ROOT_KEY,
        )

    class _ForeignContentOperation(_ActualContentOperation):
        def __init__(self, foreign_result) -> None:
            super().__init__()
            self.foreign_result = foreign_result

        def __call__(self, image: torch.Tensor, detection_key: str):
            self.calls.append(
                (id(self), image.detach().clone())
            )
            return replace(
                self.foreign_result,
                content_input_image_digest=rgb8_image_digest(image),
                content_replay_operation=self,
            )

    foreign_operation = _ForeignContentOperation(foreign_content_result)
    foreign_binding = _binding(foreign_operation, direct_null)
    contaminated_flow = conditional_recovery_decision(
        rescue_null,
        _ROOT_KEY,
        content_detector_binding=foreign_binding,
        thresholds=_thresholds(
            foreign_binding,
            tau=tau,
            tau_rescue=raw_score - 0.1,
            calibration_identity="foreign-content-replay-gatekeeper",
        ),
        geometry_estimation_operation=_GeometryOperation(_estimation(0.2)),
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    assert contaminated_flow.status == "raw_content_identity_failure"
    assert not contaminated_flow.joint_content_positive

    direct_positive = conditional_recovery_decision(
        direct_null,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=tau,
            tau_rescue=raw_score - 0.1,
            calibration_identity="joint-fpr-accounting-calibration",
        ),
        geometry_estimation_operation=_GeometryOperation(_estimation(0.2)),
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    geometry = _GeometryOperation(_estimation(0.2))
    rescue_positive = conditional_recovery_decision(
        rescue_null,
        _ROOT_KEY,
        content_detector_binding=binding,
        thresholds=_thresholds(
            binding,
            tau=tau,
            tau_rescue=raw_score - 0.1,
            calibration_identity="joint-fpr-accounting-calibration",
        ),
        geometry_estimation_operation=geometry,
        geometry_reliability_thresholds=_reliability_thresholds(),
    )

    assert direct_positive.joint_content_positive
    assert direct_positive.positive_source == "raw_content"
    assert rescue_positive.joint_content_positive
    assert rescue_positive.positive_source == "rectified_content"
    assert rescue_positive.raw_content_score < tau
    assert rescue_positive.rectified_content_score >= tau
    assert rescue_positive.raw_content_result.detector_identity == (
        rescue_positive.rectified_content_result.detector_identity
    )
    assert rescue_positive.raw_content_result.content_config_digest == (
        rescue_positive.rectified_content_result.content_config_digest
    )
    assert rescue_positive.threshold_identity == (
        direct_positive.threshold_identity
    )
    assert rescue_positive.root_key_public_digest == (
        rescue_positive.raw_content_result.hf_result.root_key_public_digest
    )
    assert rescue_positive.root_key_public_digest == (
        rescue_positive.rectified_content_result.hf_result.root_key_public_digest
    )
    assert geometry.calls == 1
    synthetic_primary_null_outcomes = (direct_positive, rescue_positive)
    assert (
        sum(
            result.joint_content_positive
            for result in synthetic_primary_null_outcomes
        )
        == 2
    )
    assert {
        direct_positive.positive_path,
        rescue_positive.positive_path,
    } == {"raw_positive", "rescue_positive"}
    assert torch.equal(operation.calls[-2][1], rescue_null)
    assert torch.equal(
        operation.calls[-1][1],
        rescue_positive.image_rectification_result.rectified_image,
    )
    assert rescue_positive.source_image_digest != (
        direct_positive.source_image_digest
    )
    assert torch.equal(rescue_positive.source_image, rescue_null)
    assert rescue_positive.source_image.data_ptr() != rescue_null.data_ptr()
    assert rescue_positive.raw_content_result.content_input_image_digest == (
        rescue_positive.source_image_digest
    )
    assert (
        rescue_positive.rectified_content_result.content_input_image_digest
        == rescue_positive.image_rectification_result.rectified_image_digest
    )
    assert rescue_positive.image_rectification_result.source_image_digest == (
        rescue_positive.source_image_digest
    )
    replayed_rectification = image_rectifier(
        rescue_null,
        rescue_positive.geometry_estimation,
        rescue_positive.geometry_reliability_result,
    )
    joint_rectification = rescue_positive.image_rectification_result
    assert torch.equal(
        replayed_rectification.rectified_image,
        joint_rectification.rectified_image,
    )
    assert torch.equal(
        replayed_rectification.valid_support_mask,
        joint_rectification.valid_support_mask,
    )
    assert replayed_rectification.source_image_digest == (
        joint_rectification.source_image_digest
    )
    assert replayed_rectification.rectified_image_digest == (
        joint_rectification.rectified_image_digest
    )
    assert replayed_rectification.token_crop_support == (
        joint_rectification.token_crop_support
    )
    assert replayed_rectification.pixel_crop_support == (
        joint_rectification.pixel_crop_support
    )
    assert replayed_rectification.crop_support == (
        joint_rectification.crop_support
    )
    assert replayed_rectification.canonical_to_observed_matrix == (
        joint_rectification.canonical_to_observed_matrix
    )
    assert replayed_rectification.rectification_config_digest == (
        joint_rectification.rectification_config_digest
    )
    assert len(rescue_positive.source_image_digest) == 64
    assert rescue_positive.source_image_digest == (
        rescue_positive.source_image_digest.lower()
    )
    assert (
        validate_conditional_recovery_result(rescue_positive)
        is rescue_positive
    )
    assert (
        validate_conditional_recovery_result(
            rescue_positive,
            _ROOT_KEY,
        )
        is rescue_positive
    )
    with pytest.raises(
        ConditionalRecoveryError,
        match="source image binding",
    ):
        replace(rescue_positive, source_image_digest="A" * 64)
    coordinated_source = torch.flip(
        rescue_positive.source_image,
        dims=(3,),
    ).contiguous()
    with pytest.raises(
        ConditionalRecoveryError,
        match="content result validation",
    ):
        replace(
            rescue_positive,
            source_image=coordinated_source,
            source_image_digest=rgb8_image_digest(coordinated_source),
        )
    with pytest.raises(ConditionalRecoveryError):
        replace(rescue_positive, positive_source="raw_content")
    with pytest.raises(ConditionalRecoveryError, match="unknown"):
        replace(rescue_positive, status="forged")
    for field_name, invalid_value in (
        ("tau", "invalid"),
        ("geometry_estimation", "invalid"),
        ("image_rectification_result", "invalid"),
        ("rectified_content_result", "invalid"),
    ):
        with pytest.raises(ConditionalRecoveryError):
            replace(
                rescue_positive,
                **{field_name: invalid_value},
            )
    invalid_diagnostic_content = replace(
        rescue_positive.raw_content_result,
        diagnostic_combination="invalid",
    )
    with pytest.raises(ConditionalRecoveryError):
        replace(
            rescue_positive,
            raw_content_result=invalid_diagnostic_content,
        )
    tampered_hf_result = replace(
        rescue_positive.rectified_content_result.hf_result,
        detector_config_digest="f" * 64,
    )
    tampered_content_result = replace(
        rescue_positive.rectified_content_result,
        hf_result=tampered_hf_result,
    )
    with pytest.raises(ConditionalRecoveryError):
        replace(
            rescue_positive,
            rectified_content_result=tampered_content_result,
        )
    tampered_rectification = replace(
        rescue_positive.image_rectification_result,
        pixel_crop_support=0.0,
    )
    with pytest.raises(
        ConditionalRecoveryError,
        match="rectification result validation",
    ):
        replace(
            rescue_positive,
            image_rectification_result=tampered_rectification,
        )

    other_source = torch.flip(rescue_null, dims=(2,)).contiguous()
    other_rectification = image_rectifier(
        other_source,
        rescue_positive.geometry_estimation,
        rescue_positive.geometry_reliability_result,
    )
    other_rectified_content = operation(
        other_rectification.rectified_image,
        _ROOT_KEY,
    )
    with pytest.raises(
        ConditionalRecoveryError,
        match="rectification result validation",
    ):
        replace(
            rescue_positive,
            image_rectification_result=other_rectification,
            rectified_content_result=other_rectified_content,
            rectified_content_score=other_rectified_content.content_score,
        )
    coordinated_other_rectification = replace(
        other_rectification,
        source_image_digest=rescue_positive.source_image_digest,
    )
    with pytest.raises(
        ConditionalRecoveryError,
        match="rectification result validation",
    ):
        replace(
            rescue_positive,
            image_rectification_result=coordinated_other_rectification,
            rectified_content_result=other_rectified_content,
            rectified_content_score=other_rectified_content.content_score,
        )

    class _SecondCallIdentityDrift(_ActualContentOperation):
        def __call__(self, image: torch.Tensor, detection_key: str):
            content_result = super().__call__(image, detection_key)
            if len(self.calls) == 2:
                return replace(
                    content_result,
                    hf_result=replace(
                        content_result.hf_result,
                        detector_config_digest="d" * 64,
                    ),
                )
            return content_result

    drifting_operation = _SecondCallIdentityDrift()
    drifting_binding = _binding(drifting_operation, direct_null)
    drifting_result = conditional_recovery_decision(
        rescue_null,
        _ROOT_KEY,
        content_detector_binding=drifting_binding,
        thresholds=_thresholds(
            drifting_binding,
            tau=tau,
            tau_rescue=raw_score - 0.1,
            calibration_identity="joint-fpr-accounting-calibration",
        ),
        geometry_estimation_operation=_GeometryOperation(_estimation(0.2)),
        geometry_reliability_thresholds=_reliability_thresholds(),
    )
    assert drifting_result.status == (
        "negative_rectified_content_identity_failure"
    )
    source_mutation = replace(rescue_positive)
    source_mutation.source_image[0, 0, 0, 0] ^= 1
    with pytest.raises(
        ConditionalRecoveryError,
        match="source image digest",
    ):
        validate_conditional_recovery_result(source_mutation)
    rescue_positive.image_rectification_result.rectified_image[
        0, 0, 0, 0
    ] ^= 1
    with pytest.raises(ConditionalRecoveryError):
        validate_conditional_recovery_result(rescue_positive)
