"""CPU/synthetic checks for the frozen CEG-WM geometry chain."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import cos, log, pi, sin, sqrt
from struct import pack

import pytest
import torch
import torch.nn.functional as functional

from main.geometry_chain.qk_sync import (
    QkGeometrySyncError,
    QkLayerObservation,
    _aggregate_descriptor_digest,
    _content_projection,
    geometry_direction_outside_content_span,
    geometry_synchronization_write,
    qk_geometry_sync,
    row_normalized_relation_score,
    validate_qk_geometry_sync_result,
)
from main.geometry_chain.rectifier import (
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
    SimilarityTransform,
    geometric_transform_estimator,
    sampling_matrix,
)
from main.shared.key_schedule import (
    derive_wrong_key_material,
    identify_root_key,
)

_REGISTERED_KEY = "geometry-cpu-synthetic-key"
_LAYER_NAMES = (
    "transformer_blocks.0.attn",
    "transformer_blocks.23.attn",
)


def _actual_qk_observation(side: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    token_count = side * side
    observations = tuple(
        QkLayerObservation(
            layer_name=layer_name,
            query=torch.randn(2, token_count, 5, generator=generator),
            attention_key=torch.randn(2, token_count, 5, generator=generator),
            operator_identity="projected_normalized_attention_qk",
        )
        for layer_name in _LAYER_NAMES
    )
    return observations, qk_geometry_sync(observations, _REGISTERED_KEY)


def _inverse_affine(matrix: torch.Tensor) -> torch.Tensor:
    linear = matrix[:, :2].to(dtype=torch.float64)
    inverse_linear = torch.linalg.inv(linear)
    inverse_translation = -(inverse_linear @ matrix[:, 2].to(dtype=torch.float64))
    return torch.cat(
        (inverse_linear, inverse_translation.unsqueeze(1)), dim=1
    ).to(dtype=torch.float32)


def _warp_relation(sampling: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            sampling @ relation[:, :, channel] @ sampling.transpose(0, 1)
            for channel in range(4)
        ],
        dim=-1,
    )


def _relation_digest(values: tuple[float, ...]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def _synthetic_transformed_observation(base, matrix: torch.Tensor):
    geometry_layer = base.layers[0]
    backward, _ = sampling_matrix(
        _inverse_affine(matrix),
        original_grid_side=geometry_layer.original_grid_side,
        token_indices=geometry_layer.token_indices,
    )
    layers = []
    for layer in base.layers:
        transformed = _warp_relation(backward, layer.projection_tensor())
        values = tuple(float(value) for value in transformed.reshape(-1))
        layers.append(
            replace(
                layer,
                relation_values=values,
                descriptor_digest=_relation_digest(values),
                relation_score=row_normalized_relation_score(
                    transformed,
                    layer.projection_tensor(),
                ),
            )
        )
    layer_values = tuple(layers)
    return replace(
        base,
        layers=layer_values,
        relation_score=sum(layer.relation_score for layer in layer_values)
        / len(layer_values),
        descriptor_digest=_aggregate_descriptor_digest(layer_values),
    )


def _identity_estimation_record() -> GeometricTransformEstimation:
    transform = SimilarityTransform(
        dihedral="identity",
        residual_rotation_degrees=0.0,
        log_scale=0.0,
        translation_x=0.0,
        translation_y=0.0,
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        is_exact_identity=True,
        continuous_parameter_on_search_boundary=False,
    )
    return GeometricTransformEstimation(
        candidate_ids=(
            "key_schedule_sha256_counter",
            "qk_relation_similarity",
            "rectification_similarity",
        ),
        transform=transform,
        registered_objective=1.0,
        second_registered_objective=0.8,
        exact_identity_objective=1.0,
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
        identity_margin=0.0,
        key_margin=0.8,
        inlier_ratio=1.0,
        mean_residual=0.0,
        epsilon_inlier=0.8,
        anchor_residuals=tuple(0.0 for _ in range(12)),
        registered_root_key_public_digest=identify_root_key(
            _REGISTERED_KEY
        ).root_key_public_digest,
        observation_descriptor_digest="1" * 64,
        observation_projection_digest="2" * 64,
        observation_geometry_config_digest="3" * 64,
        search_config_digest="4" * 64,
    )


def _thresholds(
    *,
    coverage: float = 0.45,
    uniqueness: float = 0.0,
    gap: float = 0.0,
    key: float = 0.0,
    inlier: float = 0.0,
    residual: float = 1.0,
    identity: float = 0.0,
) -> GeometryReliabilityThresholds:
    return GeometryReliabilityThresholds(
        gamma_coverage=coverage,
        gamma_uniqueness=uniqueness,
        gamma_gap=gap,
        gamma_key=key,
        gamma_inlier=inlier,
        gamma_residual=residual,
        gamma_identity=identity,
        epsilon_inlier=0.8,
        fit_identity="geometry_reliability_fit_cpu_synthetic",
    )


@pytest.mark.unit
def test_qk_relation_consumption():
    observations, original = _actual_qk_observation(side=4, seed=901)
    query_changed = list(observations)
    changed_query = observations[0].query.clone()
    changed_query[0, 0, 0] += 0.25
    query_changed[0] = replace(observations[0], query=changed_query)
    query_result = qk_geometry_sync(tuple(query_changed), _REGISTERED_KEY)

    key_changed = list(observations)
    changed_attention_key = observations[1].attention_key.clone()
    changed_attention_key[1, 3, 2] -= 0.25
    key_changed[1] = replace(
        observations[1],
        attention_key=changed_attention_key,
    )
    key_result = qk_geometry_sync(tuple(key_changed), _REGISTERED_KEY)
    assert query_result.descriptor_digest != original.descriptor_digest
    assert key_result.descriptor_digest != original.descriptor_digest
    assert query_result.relation_score != original.relation_score
    assert key_result.relation_score != original.relation_score

    wrong_material = derive_wrong_key_material(
        identify_root_key(_REGISTERED_KEY).root_key_public_digest,
        0,
    )
    wrong_key_result = qk_geometry_sync(observations, wrong_material)
    assert wrong_key_result.descriptor_digest == original.descriptor_digest
    assert wrong_key_result.projection_digest != original.projection_digest
    assert wrong_key_result.key_role == "wrong"
    wrong_estimation = geometric_transform_estimator(
        wrong_key_result,
        wrong_material,
        epsilon_inlier=0.8,
    )
    assert wrong_estimation.observation_projection_digest == (
        wrong_key_result.projection_digest
    )
    assert wrong_estimation.registered_root_key_public_digest == (
        wrong_key_result.root_key_public_digest
    )

    large_observations, large_result = _actual_qk_observation(side=10, seed=902)
    layer = large_result.layers[0]
    indices = torch.tensor(layer.token_indices, dtype=torch.long)
    query = large_observations[0].query.index_select(1, indices)
    attention_key = large_observations[0].attention_key.index_select(1, indices)
    attention = query @ attention_key.transpose(-1, -2) / sqrt(query.shape[-1])
    probability = torch.softmax(attention, dim=-1).mean(dim=0)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    coordinates = torch.tensor(
        [
            (
                -1.0 + 2.0 * (index % 10) / 9.0,
                -1.0 + 2.0 * (index // 10) / 9.0,
            )
            for index in layer.token_indices
        ],
        dtype=torch.float32,
    )
    distance = torch.cdist(coordinates, coordinates) / (2.0 * sqrt(2.0))
    expected_distance_channel = (
        probability - probability.mean(dim=-1, keepdim=True)
    ) * (distance - distance.mean(dim=-1, keepdim=True))
    assert torch.allclose(
        layer.relation_tensor()[:, :, 3],
        expected_distance_channel,
        atol=1e-7,
        rtol=1e-6,
    )
    translated = torch.tensor(
        ((1.0, 0.0, 1.0 / 9.0), (0.0, 1.0, 0.0)),
        dtype=torch.float32,
    )
    translated_sampling, translated_valid = sampling_matrix(
        translated,
        original_grid_side=layer.original_grid_side,
        token_indices=layer.token_indices,
    )
    sampled_axis = (0, 1, 3, 4, 5, 6, 8, 9)
    source_row = 1 * len(sampled_axis) + 1
    lower_column = 1 * len(sampled_axis) + 1
    upper_column = 1 * len(sampled_axis) + 2
    assert translated_valid[source_row]
    assert translated_sampling[source_row, lower_column] == pytest.approx(0.75)
    assert translated_sampling[source_row, upper_column] == pytest.approx(0.25)
    assert translated_sampling[source_row].sum() == pytest.approx(1.0)

    changed_values = list(original.layers[0].relation_values)
    changed_values[0] += 0.125
    changed_tuple = tuple(changed_values)
    tampered_layer = replace(
        original.layers[0],
        relation_values=changed_tuple,
        descriptor_digest=_relation_digest(changed_tuple),
        relation_score=row_normalized_relation_score(
            torch.tensor(changed_tuple, dtype=torch.float32).reshape(
                original.layers[0].relation_shape
            ),
            original.layers[0].projection_tensor(),
        ),
    )
    tampered = replace(
        original,
        layers=(tampered_layer, original.layers[1]),
        relation_score=(
            tampered_layer.relation_score + original.layers[1].relation_score
        )
        / 2.0,
    )
    with pytest.raises(QkGeometrySyncError, match="aggregate descriptor"):
        validate_qk_geometry_sync_result(tampered, _REGISTERED_KEY)

    baseline = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    content = baseline.clone()
    content[0, 0, 0, 0] += 0.01
    content = content.to(dtype=torch.float16)
    gradient = torch.zeros_like(baseline)
    gradient[0, 0, 0, 1] = 1.0
    content_direction = torch.zeros_like(baseline)
    content_direction[0, 0, 0, 0] = 1.0
    write = geometry_synchronization_write(
        baseline,
        content,
        gradient,
        (content_direction,),
        geometry_ratio=1.0 / 4.0,
        baseline_score=float(content[0, 0, 0, 1]),
        materialize=lambda value: value.to(dtype=torch.float16),
        replay_score=lambda value: float(value[0, 0, 0, 1]),
    )
    assert write.accepted
    assert write.accepted_score is not None
    assert write.accepted_score > write.baseline_score
    assert write.content_projection_relative is not None
    assert write.content_projection_relative <= 1e-4

    first = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32)
    gradient_second = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32)
    retained_near_collinear = torch.tensor(
        (1.0, 1e-5, 0.0), dtype=torch.float32
    )
    retained_projection, retained_residual = _content_projection(
        gradient_second,
        (first, retained_near_collinear),
    )
    assert torch.linalg.vector_norm(
        retained_projection - gradient_second.to(dtype=torch.float64)
    ) < 1e-8
    assert torch.linalg.vector_norm(retained_residual) < 1e-8
    truncated_near_collinear = torch.tensor(
        (1.0, 1e-7, 0.0), dtype=torch.float32
    )
    truncated_projection, truncated_residual = _content_projection(
        gradient_second,
        (first, truncated_near_collinear),
    )
    assert torch.linalg.vector_norm(truncated_projection) < 1e-6
    assert torch.linalg.vector_norm(
        truncated_residual - gradient_second.to(dtype=torch.float64)
    ) < 1e-6
    truncated_direction = geometry_direction_outside_content_span(
        gradient_second,
        (first, truncated_near_collinear),
    )
    assert torch.allclose(
        truncated_direction,
        gradient_second,
        atol=2e-7,
        rtol=0.0,
    )


@pytest.mark.quick
def test_qk_similarity_transform_identifiability():
    observations, _ = _actual_qk_observation(side=3, seed=2302)
    base_small = qk_geometry_sync(observations, _REGISTERED_KEY)
    structured_observations, _ = _actual_qk_observation(side=5, seed=2305)
    base_structured = qk_geometry_sync(
        structured_observations,
        _REGISTERED_KEY,
    )
    scale_observations, _ = _actual_qk_observation(side=8, seed=2305)
    base_scale = qk_geometry_sync(scale_observations, _REGISTERED_KEY)
    assert base_small.descriptor_digest != base_structured.descriptor_digest
    finest_rotation_resolution = 8.0 / 9.0
    finest_log_scale_resolution = log(sqrt(2.0)) / 18.0
    finest_translation_resolution = 0.14 / 9.0

    def affine(
        rotation_degrees: float,
        log_scale_value: float,
        translation_x: float,
        translation_y: float,
    ) -> torch.Tensor:
        angle = rotation_degrees * pi / 180.0
        scale_value = torch.exp(torch.tensor(log_scale_value)).item()
        return torch.tensor(
            (
                (
                    scale_value * cos(angle),
                    -scale_value * sin(angle),
                    translation_x,
                ),
                (
                    scale_value * sin(angle),
                    scale_value * cos(angle),
                    translation_y,
                ),
            ),
            dtype=torch.float32,
        )

    cases = (
        ("identity", base_small, 0.0, 0.0, 0.0, 0.0, False),
        ("rotation", base_small, 16.0, 0.0, 0.0, 0.0, True),
        (
            "scale",
            base_scale,
            0.0,
            -3.0 * finest_log_scale_resolution,
            0.0,
            0.0,
            True,
        ),
        (
            "translation_crop",
            base_structured,
            0.0,
            0.0,
            0.14,
            0.0,
            True,
        ),
        (
            "combined_crop",
            base_structured,
            8.0,
            0.0,
            0.14,
            -0.14,
            True,
        ),
    )
    for (
        case_name,
        base_result,
        expected_rotation,
        expected_log_scale,
        expected_x,
        expected_y,
        expected_coverage_loss,
    ) in cases:
        expected_matrix = affine(
            expected_rotation,
            expected_log_scale,
            expected_x,
            expected_y,
        )
        transformed = _synthetic_transformed_observation(
            base_result,
            expected_matrix,
        )
        estimation = geometric_transform_estimator(
            transformed,
            _REGISTERED_KEY,
            epsilon_inlier=0.8,
        )
        assert (
            estimation.observation_descriptor_digest
            == transformed.descriptor_digest
        ), case_name
        assert (
            estimation.observation_projection_digest
            == transformed.projection_digest
        ), case_name
        assert (
            estimation.observation_geometry_config_digest
            == transformed.geometry_config_digest
        ), case_name
        transform = estimation.transform
        assert transform.dihedral == "identity", case_name
        assert abs(
            transform.residual_rotation_degrees - expected_rotation
        ) <= finest_rotation_resolution + 1e-8, case_name
        assert abs(
            transform.log_scale - expected_log_scale
        ) <= finest_log_scale_resolution + 1e-8, case_name
        assert abs(
            transform.translation_x - expected_x
        ) <= finest_translation_resolution + 1e-8, case_name
        assert abs(
            transform.translation_y - expected_y
        ) <= finest_translation_resolution + 1e-8, case_name
        if case_name == "identity":
            assert transform.is_exact_identity
        else:
            assert estimation.registered_objective > (
                estimation.exact_identity_objective
            ), case_name
        if expected_coverage_loss:
            assert estimation.coverage < 1.0, case_name
        assert estimation.gap > 0.0, case_name
        assert estimation.key_margin > 0.0, case_name
        assert len(estimation.wrong_key_objectives) == 8
        assert not hasattr(estimation, "reliable")


@pytest.mark.quick
def test_geometry_reliability_fail_closed():
    observations, _ = _actual_qk_observation(side=3, seed=2303)
    base_qk_result = qk_geometry_sync(observations, _REGISTERED_KEY)
    identity_observation = _synthetic_transformed_observation(
        base_qk_result,
        torch.eye(2, 3, dtype=torch.float32),
    )
    identity_estimation = geometric_transform_estimator(
        identity_observation,
        _REGISTERED_KEY,
        epsilon_inlier=0.8,
    )
    not_fitted = geometry_reliability(identity_estimation)
    assert not not_fitted.reliable
    assert not not_fitted.allow_rectification
    assert not_fitted.status == "reliability_not_fitted"
    assert not_fitted.fitted_reliability_thresholds is None
    assert not_fitted.threshold_config_digest is None
    assert not validate_geometry_reliability_result(
        not_fitted,
        identity_estimation,
    )

    identity_thresholds = _thresholds(gap=0.01, key=0.05, inlier=0.5)
    identity_reliable = geometry_reliability(
        identity_estimation,
        identity_thresholds,
    )
    assert identity_reliable.reliable
    assert identity_reliable.allow_rectification
    assert (
        identity_reliable.fitted_reliability_thresholds
        == identity_thresholds
    )
    assert validate_geometry_reliability_result(
        identity_reliable,
        identity_estimation,
    )
    changed_threshold_result = geometry_reliability(
        identity_estimation,
        replace(identity_thresholds, gamma_gap=0.02),
    )
    assert validate_geometry_reliability_result(
        changed_threshold_result,
        identity_estimation,
    )

    forged_results = (
        GeometryReliabilityResult(
            reliable=True,
            allow_rectification=True,
            status="forged",
            failure_reasons=(),
            fitted_reliability_thresholds=None,
            threshold_config_digest=None,
            estimator_search_config_digest=(
                identity_reliable.estimator_search_config_digest
            ),
            estimation_identity_digest=(
                identity_reliable.estimation_identity_digest
            ),
            registered_root_key_public_digest=(
                identity_reliable.registered_root_key_public_digest
            ),
            reliability_identity_digest=(
                identity_reliable.reliability_identity_digest
            ),
        ),
        replace(identity_reliable, status="unreliable"),
        replace(
            identity_reliable,
            fitted_reliability_thresholds=None,
            threshold_config_digest=None,
        ),
        replace(
            identity_reliable,
            fitted_reliability_thresholds=replace(
                identity_thresholds,
                gamma_gap=0.02,
            ),
        ),
        replace(
            identity_reliable,
            fitted_reliability_thresholds=(
                changed_threshold_result.fitted_reliability_thresholds
            ),
            threshold_config_digest=(
                changed_threshold_result.threshold_config_digest
            ),
        ),
        replace(
            identity_reliable,
            failure_reasons=("forged_reliability_reason",),
        ),
        replace(identity_reliable, reliable=False),
        replace(identity_reliable, allow_rectification=False),
        replace(identity_reliable, threshold_config_digest="0" * 64),
        replace(identity_reliable, estimator_search_config_digest="0" * 64),
        replace(identity_reliable, estimation_identity_digest="0" * 64),
        replace(
            identity_reliable,
            registered_root_key_public_digest="0" * 64,
        ),
        replace(identity_reliable, reliability_identity_digest="0" * 64),
    )
    for forged_result in forged_results:
        with pytest.raises(GeometryReliabilityError):
            validate_geometry_reliability_result(
                forged_result,
                identity_estimation,
            )

    low_coverage_estimation = replace(
        identity_estimation,
        coverage_forward=0.4,
        coverage=0.4,
    )
    low_coverage = geometry_reliability(
        low_coverage_estimation,
        _thresholds(coverage=0.8, residual=10.0),
    )
    assert not low_coverage.reliable
    assert "coverage_below_threshold" in low_coverage.failure_reasons

    ambiguous_estimation = replace(
        identity_estimation,
        second_registered_objective=identity_estimation.registered_objective,
        gap=0.0,
    )
    ambiguous = geometry_reliability(
        ambiguous_estimation,
        _thresholds(gap=0.01, residual=10.0),
    )
    assert not ambiguous.reliable
    assert (
        "registered_candidate_gap_below_threshold"
        in ambiguous.failure_reasons
    )

    boundary_angle = 32.0 * pi / 180.0
    boundary_transform = SimilarityTransform(
        dihedral="identity",
        residual_rotation_degrees=32.0,
        log_scale=0.0,
        translation_x=0.0,
        translation_y=0.0,
        matrix=(
            (cos(boundary_angle), -sin(boundary_angle), 0.0),
            (sin(boundary_angle), cos(boundary_angle), 0.0),
        ),
        is_exact_identity=False,
        continuous_parameter_on_search_boundary=True,
    )
    boundary_estimation = replace(
        identity_estimation,
        transform=boundary_transform,
    )
    boundary = geometry_reliability(
        boundary_estimation,
        _thresholds(residual=10.0),
    )
    assert not boundary.reliable
    assert "continuous_parameter_on_search_boundary" in boundary.failure_reasons

    nonfinite_estimation = replace(identity_estimation, mean_residual=float("nan"))
    nonfinite = geometry_reliability(
        nonfinite_estimation,
        _thresholds(residual=10.0),
    )
    assert not nonfinite.reliable
    assert "nonfinite_geometry_metric" in nonfinite.failure_reasons
    assert not hasattr(nonfinite, "positive")

    invalid_domain_estimation = replace(
        identity_estimation,
        coverage_forward=1.2,
        coverage=1.0,
    )
    invalid_domain = geometry_reliability(
        invalid_domain_estimation,
        _thresholds(residual=10.0),
    )
    assert not invalid_domain.reliable
    assert "geometry_metric_outside_domain" in invalid_domain.failure_reasons

    tampered_identity_flag = replace(
        identity_estimation,
        transform=replace(
            identity_estimation.transform,
            is_exact_identity=False,
        ),
    )
    with pytest.raises(
        GeometryReliabilityError,
        match="identity validation failed",
    ):
        geometry_reliability(
            tampered_identity_flag,
            _thresholds(),
        )


@pytest.mark.quick
def test_geometry_reliability_wrong_key_and_raw_metrics():
    observations, _ = _actual_qk_observation(side=3, seed=2304)
    base_qk_result = qk_geometry_sync(observations, _REGISTERED_KEY)
    angle = 16.0 * pi / 180.0
    rotation = torch.tensor(
        (
            (cos(angle), -sin(angle), 0.0),
            (sin(angle), cos(angle), 0.0),
        ),
        dtype=torch.float32,
    )
    transformed = _synthetic_transformed_observation(base_qk_result, rotation)
    rotation_estimation = geometric_transform_estimator(
        transformed,
        _REGISTERED_KEY,
        epsilon_inlier=0.8,
    )
    assert rotation_estimation.key_margin == pytest.approx(
        rotation_estimation.registered_objective
        - max(rotation_estimation.wrong_key_objectives)
    )
    assert rotation_estimation.coverage == pytest.approx(
        min(
            rotation_estimation.coverage_forward,
            rotation_estimation.coverage_backward,
        )
    )
    assert rotation_estimation.uniqueness == pytest.approx(
        min(
            rotation_estimation.uniqueness_forward,
            rotation_estimation.uniqueness_backward,
        )
    )
    assert rotation_estimation.gap == pytest.approx(
        rotation_estimation.registered_objective
        - rotation_estimation.second_registered_objective
    )
    wrong_key_rejected = geometry_reliability(
        rotation_estimation,
        _thresholds(key=rotation_estimation.key_margin + 0.01, residual=10.0),
    )
    assert not wrong_key_rejected.reliable
    assert (
        "wrong_key_margin_below_threshold"
        in wrong_key_rejected.failure_reasons
    )
    assert len(rotation_estimation.anchor_residuals) == 12
    assert rotation_estimation.registered_root_key_public_digest == (
        identify_root_key(_REGISTERED_KEY).root_key_public_digest
    )


@pytest.mark.unit
def test_rectification_coordinate_protocol():
    identity_estimation = _identity_estimation_record()
    reliability = geometry_reliability(
        identity_estimation,
        _thresholds(gap=0.01, key=0.05, inlier=0.5),
    )
    image = torch.arange(3 * 7 * 7, dtype=torch.uint8).reshape(1, 3, 7, 7)
    result = image_rectifier(image, identity_estimation, reliability)
    assert (
        validate_image_rectification_result(
            result,
            image,
            identity_estimation,
            reliability,
        )
        is result
    )

    theta = identity_estimation.transform.tensor().unsqueeze(0)
    grid = functional.affine_grid(theta, image.shape, align_corners=True)
    expected = torch.floor(
        torch.clamp(
            functional.grid_sample(
                image.float() / 255.0,
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
    expected_support = (
        functional.grid_sample(
            torch.ones((1, 1, 7, 7), dtype=torch.float32),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        > 0.5
    )
    assert torch.equal(result.rectified_image, expected)
    assert torch.equal(result.valid_support_mask, expected_support)
    assert result.token_crop_support == pytest.approx(identity_estimation.coverage)
    assert result.pixel_crop_support == pytest.approx(
        float(expected_support.float().mean())
    )
    assert result.crop_support == pytest.approx(
        (result.token_crop_support, result.pixel_crop_support)
    )

    translated_transform = SimilarityTransform(
        dihedral="identity",
        residual_rotation_degrees=0.0,
        log_scale=0.0,
        translation_x=0.20,
        translation_y=0.0,
        matrix=((1.0, 0.0, 0.20), (0.0, 1.0, 0.0)),
        is_exact_identity=False,
        continuous_parameter_on_search_boundary=False,
    )
    translated_estimation = replace(
        identity_estimation,
        transform=translated_transform,
    )
    translated_reliability = geometry_reliability(
        translated_estimation,
        _thresholds(gap=0.01, key=0.05, inlier=0.5),
    )
    translated_result = image_rectifier(
        image,
        translated_estimation,
        translated_reliability,
    )
    translated_theta = translated_transform.tensor().unsqueeze(0)
    translated_grid = functional.affine_grid(
        translated_theta,
        image.shape,
        align_corners=True,
    )
    expected_translated_image = torch.floor(
        torch.clamp(
            functional.grid_sample(
                image.float() / 255.0,
                translated_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ),
            0.0,
            1.0,
        )
        * 255.0
    ).to(dtype=torch.uint8)
    expected_translated_support = (
        functional.grid_sample(
            torch.ones((1, 1, 7, 7), dtype=torch.float32),
            translated_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        > 0.5
    )
    incorrect_border_support = (
        functional.grid_sample(
            torch.ones((1, 1, 7, 7), dtype=torch.float32),
            translated_grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        > 0.5
    )
    assert torch.equal(
        translated_result.rectified_image,
        expected_translated_image,
    )
    assert torch.equal(
        translated_result.valid_support_mask,
        expected_translated_support,
    )
    assert bool(translated_result.valid_support_mask[:, :, :, 0].all())
    assert not bool(translated_result.valid_support_mask[:, :, :, -1].any())
    assert bool(translated_result.rectified_image[:, :, :, -1].any())
    assert not bool(translated_result.valid_support_mask.all())
    assert translated_result.pixel_crop_support < 1.0
    assert translated_result.pixel_crop_support == pytest.approx(
        float(expected_translated_support.float().mean())
    )
    assert bool(incorrect_border_support.all())
    assert not torch.equal(
        translated_result.valid_support_mask,
        incorrect_border_support,
    )
    assert torch.equal(
        torch.tensor(translated_result.canonical_to_observed_matrix),
        translated_transform.tensor(),
    )

    flipped_transform = SimilarityTransform(
        dihedral="x_flip",
        residual_rotation_degrees=0.0,
        log_scale=0.0,
        translation_x=0.0,
        translation_y=0.0,
        matrix=((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        is_exact_identity=False,
        continuous_parameter_on_search_boundary=False,
    )
    different_estimation = replace(
        identity_estimation,
        transform=flipped_transform,
    )
    flipped_reliability = geometry_reliability(
        different_estimation,
        _thresholds(gap=0.01, key=0.05, inlier=0.5),
    )
    flipped_result = image_rectifier(
        image,
        different_estimation,
        flipped_reliability,
    )
    flipped_theta = different_estimation.transform.tensor().unsqueeze(0)
    flipped_grid = functional.affine_grid(
        flipped_theta,
        image.shape,
        align_corners=True,
    )
    expected_flipped_image = torch.floor(
        torch.clamp(
            functional.grid_sample(
                image.float() / 255.0,
                flipped_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ),
            0.0,
            1.0,
        )
        * 255.0
    ).to(dtype=torch.uint8)
    assert torch.equal(
        flipped_result.rectified_image,
        expected_flipped_image,
    )
    assert not torch.equal(flipped_result.rectified_image, image)
    exact_horizontal_flip = torch.flip(image, dims=(3,))
    flip_quantization_difference = torch.abs(
        flipped_result.rectified_image.to(dtype=torch.int16)
        - exact_horizontal_flip.to(dtype=torch.int16)
    )
    assert int(flip_quantization_difference.max()) <= 1
    assert int(flip_quantization_difference[:, :, :, 0].max()) <= 1
    assert int(flip_quantization_difference[:, :, :, -1].max()) <= 1
    assert bool(
        (
            flipped_result.rectified_image[:, :, :, :-1]
            >= flipped_result.rectified_image[:, :, :, 1:]
        ).all()
    )

    with pytest.raises(
        ImageRectifierError,
        match="reliability result validation failed",
    ):
        image_rectifier(image, different_estimation, reliability)

    forged_rectifier_results = (
        GeometryReliabilityResult(
            reliable=True,
            allow_rectification=True,
            status="forged",
            failure_reasons=(),
            fitted_reliability_thresholds=None,
            threshold_config_digest=None,
            estimator_search_config_digest=(
                reliability.estimator_search_config_digest
            ),
            estimation_identity_digest=(
                reliability.estimation_identity_digest
            ),
            registered_root_key_public_digest=(
                reliability.registered_root_key_public_digest
            ),
            reliability_identity_digest=(
                reliability.reliability_identity_digest
            ),
        ),
        replace(reliability, status="unreliable"),
        replace(
            reliability,
            fitted_reliability_thresholds=None,
            threshold_config_digest=None,
        ),
        replace(reliability, allow_rectification=False),
        replace(reliability, reliability_identity_digest="0" * 64),
    )
    for forged_result in forged_rectifier_results:
        with pytest.raises(
            ImageRectifierError,
            match="reliability result validation failed",
        ):
            image_rectifier(image, identity_estimation, forged_result)

    tampered_flag_estimation = replace(
        identity_estimation,
        transform=replace(
            identity_estimation.transform,
            is_exact_identity=False,
        ),
    )
    with pytest.raises(
        ImageRectifierError,
        match="reliability result validation failed",
    ):
        image_rectifier(image, tampered_flag_estimation, reliability)

    with pytest.raises(ImageRectifierError, match="does not allow"):
        image_rectifier(
            image,
            identity_estimation,
            geometry_reliability(identity_estimation),
        )
