"""Candidate-specific CPU checks for the LF clean-null-whitened detector."""

from __future__ import annotations

import ast
from dataclasses import asdict
from hashlib import sha256
import json
from math import cos, sqrt
from pathlib import Path
from struct import pack

import numpy as np
import pytest

from main.content_chain.detector import (
    ContentDetectorError,
    NullScoreRecord,
    SemanticTextureBranchNullCalibration,
    semantic_texture_content_detector,
)
from main.content_chain.hf_carrier import hf_carrier
from main.content_chain.hf_detector import (
    HfDetectionObservation,
    HfDetectorError,
    semantic_texture_hf_detector,
)
from main.content_chain.lf_carrier import lf_carrier
from main.content_chain.lf_detector import (
    LfDetectionObservation,
    LfDetectorError,
    PreparedLfWhitenedObservation,
    _affine_detrended_dct,
    lf_detector,
    lf_null_whitened_matched_detector,
    prepare_lf_null_whitened_observation,
    prepare_lf_null_whitened_template,
    semantic_texture_lf_detector,
)
from main.content_chain.lf_whitening import (
    LF_NULL_WHITENING_LATENT_SHAPE,
    LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfNullWhiteningAsset,
    LfNullWhiteningAssetError,
    SemanticTextureLfWhiteningAsset,
)
from main.content_chain.routing import (
    SemanticTextureRoutingObservations,
    SpatialRoutingObservation,
    semantic_texture_content_router,
)
from main.shared.key_schedule import (
    derive_wrong_key_material,
    identify_root_key,
    stable_json_utf8,
)


ROOT_KEY = "ceg-wm-lf-whitened-detector-unit-key"
LATENT_SHAPE = (1, 16, 64, 64)


def _asset(
    *,
    weight_words: tuple[str, ...] = ("3f800000",) * 96,
    declared_digest: str | None = None,
    fit_manifest_sha256: str = "a" * 64,
) -> LfNullWhiteningAsset:
    payload = {
        "artifact_role": "lf_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": list(weight_words),
    }
    digest = sha256(stable_json_utf8(payload)).hexdigest()
    return LfNullWhiteningAsset.from_canonical_payload(
        payload,
        whitening_asset_digest=declared_digest or digest,
    )


def _semantic_texture_asset() -> SemanticTextureLfWhiteningAsset:
    payload = {
        "artifact_role": "lf_semantic_texture_soft_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": "lf_semantic_texture_soft_whitened_matched_score",
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "b" * 64,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "lf_carrier_config_digest": lf_carrier(
            ROOT_KEY,
            LF_NULL_WHITENING_LATENT_SHAPE,
        ).carrier_config_digest,
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "route_candidate_id": "routing_semantic_texture_soft",
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": ["3f800000"] * 96,
    }
    return SemanticTextureLfWhiteningAsset.from_canonical_payload(
        payload,
        whitening_asset_digest=sha256(stable_json_utf8(payload)).hexdigest(),
    )


def _constant_spatial(value: float, role: str) -> SpatialRoutingObservation:
    return SpatialRoutingObservation(
        values=(value,) * (64 * 64),
        spatial_shape=(64, 64),
        source_identity_digest=sha256(role.encode("utf-8")).hexdigest(),
    )


def _template_with_affine_plane() -> tuple[float, ...]:
    carrier = lf_carrier(ROOT_KEY, LATENT_SHAPE)
    values: list[float] = []
    for channel in range(16):
        offset = channel * 64 * 64
        for height in range(64):
            height_coordinate = (2.0 * height - 63.0) / 63.0
            for width in range(64):
                width_coordinate = (2.0 * width - 63.0) / 63.0
                values.append(
                    carrier.template[offset + height * 64 + width]
                    + 1.5
                    + 0.25 * height_coordinate
                    - 0.5 * width_coordinate
                )
    return tuple(values)


@pytest.mark.unit
def test_lf_whitened_asset_and_detector_are_explicit_no_fallback_candidates() -> None:
    asset = _asset()
    observation = LfDetectionObservation.from_public_image_encoding(
        _template_with_affine_plane(),
        LATENT_SHAPE,
    )
    raw = lf_detector(observation, ROOT_KEY)
    whitened = lf_null_whitened_matched_detector(
        observation,
        ROOT_KEY,
        asset,
    )
    root_identity = identify_root_key(ROOT_KEY)
    wrong_key = derive_wrong_key_material(
        root_identity.root_key_public_digest,
        0,
    )
    wrong = lf_null_whitened_matched_detector(
        observation,
        wrong_key,
        asset,
    )

    assert whitened.candidate_id == LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID
    assert whitened.candidate_ids == (
        "key_schedule_sha256_counter",
        "lf_low_pass",
        "lf_null_whitened_matched_score",
    )
    assert whitened.whitening_asset_digest == asset.whitening_asset_digest
    assert whitened.lf_score > 0.999999
    assert whitened.lf_score > wrong.lf_score
    assert raw.candidate_id == "lf_low_pass"
    assert raw.detector_identity != whitened.detector_identity
    assert raw.detector_config_digest != whitened.detector_config_digest
    assert raw.lf_score != pytest.approx(whitened.lf_score, abs=1e-6)
    assert not hasattr(whitened, "raw_score")
    assert not hasattr(observation, "reference_image")
    assert not hasattr(observation, "embed_record")


@pytest.mark.unit
def test_semantic_texture_blind_branches_require_dedicated_assets_and_cdfs() -> None:
    identity_route = semantic_texture_content_router(
        LATENT_SHAPE,
        mode="routing_semantic_texture_soft",
        observations=SemanticTextureRoutingObservations(
            semantic_probability=_constant_spatial(
                0.25,
                "semantic-detector-identity-anchor",
            ),
            texture_complexity=_constant_spatial(
                0.75,
                "texture-detector-identity-anchor",
            ),
        ),
    )
    routes = (
        semantic_texture_content_router(
            LATENT_SHAPE,
            mode="routing_semantic_texture_soft",
            observations=SemanticTextureRoutingObservations(
                semantic_probability=_constant_spatial(
                    0.2,
                    "semantic-first-public-image",
                ),
                texture_complexity=_constant_spatial(
                    0.8,
                    "texture-first-public-image",
                ),
            ),
        ),
        semantic_texture_content_router(
            LATENT_SHAPE,
            mode="routing_semantic_texture_soft",
            observations=SemanticTextureRoutingObservations(
                semantic_probability=_constant_spatial(
                    0.8,
                    "semantic-second-public-image",
                ),
                texture_complexity=_constant_spatial(
                    0.2,
                    "texture-second-public-image",
                ),
            ),
        ),
    )
    hf_template = hf_carrier(ROOT_KEY, LATENT_SHAPE).template
    lf_template = lf_carrier(ROOT_KEY, LATENT_SHAPE).template
    values = tuple(
        hf_value + lf_value + ((index % 11) - 5) / 1000.0
        for index, (hf_value, lf_value) in enumerate(
            zip(hf_template, lf_template, strict=True)
        )
    )
    hf_observation = HfDetectionObservation.from_public_image_encoding(
        values,
        LATENT_SHAPE,
    )
    lf_observation = LfDetectionObservation.from_public_image_encoding(
        values,
        LATENT_SHAPE,
    )
    dedicated_asset = _semantic_texture_asset()
    identity_values = tuple(
        ((index % 31) - 15) / 31.0
        for index in range(16 * 64 * 64)
    )
    identity_hf_result = semantic_texture_hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            identity_values,
            LATENT_SHAPE,
        ),
        ROOT_KEY,
        identity_route,
    )
    identity_lf_result = semantic_texture_lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            identity_values,
            LATENT_SHAPE,
        ),
        ROOT_KEY,
        identity_route,
        dedicated_asset,
    )

    def fixed_primary_null_calibration(
        branch: str,
        detector_identity: str,
    ) -> SemanticTextureBranchNullCalibration:
        return SemanticTextureBranchNullCalibration(
            branch=branch,  # type: ignore[arg-type]
            detector_identity=detector_identity,
            partition_identity=f"semantic-texture-{branch}-primary-null",
            records=(
                NullScoreRecord(
                    score=-0.75,
                    source_cluster_id=f"{branch}-cluster-0",
                    sample_id=f"{branch}-sample-0",
                ),
                NullScoreRecord(
                    score=-0.25,
                    source_cluster_id=f"{branch}-cluster-1",
                    sample_id=f"{branch}-sample-1",
                ),
                NullScoreRecord(
                    score=0.25,
                    source_cluster_id=f"{branch}-cluster-2",
                    sample_id=f"{branch}-sample-2",
                ),
                NullScoreRecord(
                    score=0.75,
                    source_cluster_id=f"{branch}-cluster-3",
                    sample_id=f"{branch}-sample-3",
                ),
            ),
        )

    hf_null = fixed_primary_null_calibration(
        "hf",
        identity_hf_result.detector_identity,
    )
    lf_null = fixed_primary_null_calibration(
        "lf",
        identity_lf_result.detector_identity,
    )
    hf_results = tuple(
        semantic_texture_hf_detector(
            hf_observation,
            ROOT_KEY,
            route,
        )
        for route in routes
    )
    lf_results = tuple(
        semantic_texture_lf_detector(
            lf_observation,
            ROOT_KEY,
            route,
            dedicated_asset,
        )
        for route in routes
    )
    combined_results = tuple(
        semantic_texture_content_detector(
            hf_result,
            lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )
        for hf_result, lf_result in zip(
            hf_results,
            lf_results,
            strict=True,
        )
    )

    assert routes[0].route_identity != routes[1].route_identity
    assert {
        result.detector_config_digest
        for result in hf_results
    } == {identity_hf_result.detector_config_digest}
    assert {
        result.detector_identity
        for result in hf_results
    } == {identity_hf_result.detector_identity}
    assert {
        result.detector_config_digest
        for result in lf_results
    } == {identity_lf_result.detector_config_digest}
    assert {
        result.detector_identity
        for result in lf_results
    } == {identity_lf_result.detector_identity}
    assert tuple(result.route_identity for result in hf_results) == tuple(
        route.route_identity for route in routes
    )
    assert tuple(result.route_identity for result in lf_results) == tuple(
        route.route_identity for route in routes
    )
    assert all(
        result.whitening_asset_digest == dedicated_asset.whitening_asset_digest
        for result in lf_results
    )
    assert all(
        combined.candidate_status
        == "implemented_not_scientifically_validated"
        for combined in combined_results
    )
    assert combined_results[0].candidate_family == (
        "routing_semantic_texture_soft",
        "content_embedding_semantic_texture_soft_lf_hf",
        "lf_semantic_texture_soft_whitened_matched_score",
        "hf_semantic_texture_soft_direct_score",
        "content_combination_semantic_texture_max_standardized",
    )
    assert all(
        combined.content_score
        == max(
            combined.hf_standardization.z_score,
            combined.lf_standardization.z_score,
        )
        for combined in combined_results
    )
    assert all(combined.diagnostic_only is True for combined in combined_results)
    assert all(combined.promoted is False for combined in combined_results)
    assert {
        combined.hf_standardization.calibration_identity
        for combined in combined_results
    } == {hf_null.calibration_identity}
    assert {
        combined.lf_standardization.calibration_identity
        for combined in combined_results
    } == {lf_null.calibration_identity}
    assert {
        combined.content_config_digest
        for combined in combined_results
    } == {combined_results[0].content_config_digest}
    assert {
        combined.detector_identity
        for combined in combined_results
    } == {combined_results[0].detector_identity}
    assert tuple(
        combined.route_identity for combined in combined_results
    ) == tuple(route.route_identity for route in routes)
    assert all(
        hf_result.observation_digest == lf_result.observation_digest
        for hf_result, lf_result in zip(hf_results, lf_results, strict=True)
    )
    assert not hasattr(hf_results[0], "threshold")
    assert not hasattr(lf_results[0], "threshold")

    with pytest.raises(LfDetectorError, match="dedicated whitening W"):
        semantic_texture_lf_detector(
            lf_observation,
            ROOT_KEY,
            routes[0],
            _asset(),  # type: ignore[arg-type]
        )
    with pytest.raises(LfDetectorError, match="dedicated whitening W"):
        semantic_texture_lf_detector(
            lf_observation,
            ROOT_KEY,
            routes[0],
            None,
        )
    with pytest.raises(ContentDetectorError, match="dedicated branch CDFs"):
        semantic_texture_content_detector(
            hf_results[0],
            lf_results[0],
            hf_null=hf_null,
            lf_null=None,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_semantic_texture_detector_config_branch_key_model_and_whitening_mismatches_fail_closed(
) -> None:
    route = semantic_texture_content_router(
        LATENT_SHAPE,
        mode="routing_semantic_texture_soft",
        observations=SemanticTextureRoutingObservations(
            semantic_probability=_constant_spatial(
                0.4,
                "semantic-failure-boundary",
            ),
            texture_complexity=_constant_spatial(
                0.6,
                "texture-failure-boundary",
            ),
        ),
    )
    identity_values = tuple(
        ((index % 29) - 14) / 29.0
        for index in range(16 * 64 * 64)
    )
    query_values = tuple(
        ((index % 37) - 18) / 37.0
        for index in range(16 * 64 * 64)
    )
    asset = _semantic_texture_asset()
    identity_hf_result = semantic_texture_hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            identity_values,
            LATENT_SHAPE,
        ),
        ROOT_KEY,
        route,
    )
    identity_lf_result = semantic_texture_lf_detector(
        LfDetectionObservation.from_public_image_encoding(
            identity_values,
            LATENT_SHAPE,
        ),
        ROOT_KEY,
        route,
        asset,
    )
    hf_observation = HfDetectionObservation.from_public_image_encoding(
        query_values,
        LATENT_SHAPE,
    )
    lf_observation = LfDetectionObservation.from_public_image_encoding(
        query_values,
        LATENT_SHAPE,
    )
    hf_result = semantic_texture_hf_detector(
        hf_observation,
        ROOT_KEY,
        route,
    )
    lf_result = semantic_texture_lf_detector(
        lf_observation,
        ROOT_KEY,
        route,
        asset,
    )

    def fixed_primary_null_calibration(
        branch: str,
        detector_identity: str,
        partition_role: str,
    ) -> SemanticTextureBranchNullCalibration:
        return SemanticTextureBranchNullCalibration(
            branch=branch,  # type: ignore[arg-type]
            detector_identity=detector_identity,
            partition_identity=partition_role,
            records=(
                NullScoreRecord(
                    score=-0.75,
                    source_cluster_id=f"{branch}-fixed-cluster-negative",
                    sample_id=f"{branch}-fixed-sample-negative",
                ),
                NullScoreRecord(
                    score=0.75,
                    source_cluster_id=f"{branch}-fixed-cluster-positive",
                    sample_id=f"{branch}-fixed-sample-positive",
                ),
            ),
        )

    hf_null = fixed_primary_null_calibration(
        "hf",
        identity_hf_result.detector_identity,
        "semantic-texture-hf-fixed-primary-null",
    )
    lf_null = fixed_primary_null_calibration(
        "lf",
        identity_lf_result.detector_identity,
        "semantic-texture-lf-fixed-primary-null",
    )
    mismatched_detector_config_calibration = fixed_primary_null_calibration(
        "hf",
        sha256(b"mismatched-semantic-texture-hf-detector-config").hexdigest(),
        "semantic-texture-hf-mismatched-config-primary-null",
    )
    mismatched_branch_calibration = fixed_primary_null_calibration(
        "lf",
        identity_hf_result.detector_identity,
        "semantic-texture-hf-mismatched-branch-primary-null",
    )

    with pytest.raises(ContentDetectorError, match="CDF identity mismatch"):
        semantic_texture_content_detector(
            hf_result,
            lf_result,
            hf_null=mismatched_detector_config_calibration,
            lf_null=lf_null,
        )
    with pytest.raises(ContentDetectorError, match="CDF identity mismatch"):
        semantic_texture_content_detector(
            hf_result,
            lf_result,
            hf_null=mismatched_branch_calibration,
            lf_null=lf_null,
        )
    alternate_asset_payload = {
        "artifact_role": "lf_semantic_texture_soft_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": "lf_semantic_texture_soft_whitened_matched_score",
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "b" * 64,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "lf_carrier_config_digest": lf_carrier(
            ROOT_KEY,
            LF_NULL_WHITENING_LATENT_SHAPE,
        ).carrier_config_digest,
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "route_candidate_id": "routing_semantic_texture_soft",
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": ["3f000000"] + ["3f800000"] * 95,
    }
    alternate_asset = SemanticTextureLfWhiteningAsset.from_canonical_payload(
        alternate_asset_payload,
        whitening_asset_digest=sha256(
            stable_json_utf8(alternate_asset_payload)
        ).hexdigest(),
    )
    alternate_lf_result = semantic_texture_lf_detector(
        lf_observation,
        ROOT_KEY,
        route,
        alternate_asset,
    )
    assert alternate_lf_result.detector_identity != lf_result.detector_identity
    with pytest.raises(ContentDetectorError, match="CDF identity mismatch"):
        semantic_texture_content_detector(
            hf_result,
            alternate_lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )

    mismatched_carrier_asset_payload = {
        **asset.canonical_payload,
        "lf_carrier_config_digest": sha256(
            b"mismatched-semantic-texture-lf-carrier-config"
        ).hexdigest(),
    }
    mismatched_carrier_asset = SemanticTextureLfWhiteningAsset.from_canonical_payload(
        mismatched_carrier_asset_payload,
        whitening_asset_digest=sha256(
            stable_json_utf8(mismatched_carrier_asset_payload)
        ).hexdigest(),
    )
    with pytest.raises(LfDetectorError, match="carrier configuration mismatch"):
        semantic_texture_lf_detector(
            lf_observation,
            ROOT_KEY,
            route,
            mismatched_carrier_asset,
        )

    root_identity = identify_root_key(ROOT_KEY)
    wrong_key_zero = derive_wrong_key_material(
        root_identity.root_key_public_digest,
        0,
    )
    wrong_key_one = derive_wrong_key_material(
        root_identity.root_key_public_digest,
        1,
    )
    wrong_zero_hf_result = semantic_texture_hf_detector(
        hf_observation,
        wrong_key_zero,
        route,
    )
    wrong_zero_lf_result = semantic_texture_lf_detector(
        lf_observation,
        wrong_key_zero,
        route,
        asset,
    )
    wrong_one_lf_result = semantic_texture_lf_detector(
        lf_observation,
        wrong_key_one,
        route,
        asset,
    )
    assert wrong_zero_hf_result.detector_identity == (
        identity_hf_result.detector_identity
    )
    assert wrong_zero_lf_result.detector_identity == (
        identity_lf_result.detector_identity
    )
    assert wrong_one_lf_result.detector_identity == (
        identity_lf_result.detector_identity
    )
    with pytest.raises(ContentDetectorError, match="key semantics differ"):
        semantic_texture_content_detector(
            hf_result,
            wrong_zero_lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )
    with pytest.raises(ContentDetectorError, match="key semantics differ"):
        semantic_texture_content_detector(
            wrong_zero_hf_result,
            wrong_one_lf_result,
            hf_null=hf_null,
            lf_null=lf_null,
        )


@pytest.mark.unit
def test_lf_whitened_detector_reuses_one_asset_across_registered_wrong_and_null() -> None:
    weights = tuple(
        f"{0x3F000000 + channel * 6 + band:08x}"
        for channel in range(16)
        for band in range(6)
    )
    asset = _asset(weight_words=weights)
    registered_carrier = lf_carrier(ROOT_KEY, LATENT_SHAPE)
    registered_observation = LfDetectionObservation.from_public_image_encoding(
        registered_carrier.template,
        LATENT_SHAPE,
    )
    null_observation = LfDetectionObservation.from_public_image_encoding(
        tuple(
            ((index % 37) - 18) / 37.0
            for index in range(16 * 64 * 64)
        ),
        LATENT_SHAPE,
    )
    wrong_key = derive_wrong_key_material(
        identify_root_key(ROOT_KEY).root_key_public_digest,
        0,
    )

    registered = lf_null_whitened_matched_detector(
        registered_observation,
        ROOT_KEY,
        asset,
    )
    wrong = lf_null_whitened_matched_detector(
        registered_observation,
        wrong_key,
        asset,
    )
    primary_null = lf_null_whitened_matched_detector(
        null_observation,
        ROOT_KEY,
        asset,
    )

    assert {
        registered.whitening_asset_digest,
        wrong.whitening_asset_digest,
        primary_null.whitening_asset_digest,
    } == {asset.whitening_asset_digest}
    assert {
        registered.detector_config_digest,
        wrong.detector_config_digest,
        primary_null.detector_config_digest,
    } == {registered.detector_config_digest}
    assert registered.lf_score > wrong.lf_score
    assert registered.lf_score > primary_null.lf_score


@pytest.mark.unit
def test_lf_whitened_prepared_features_semantic_domain_binary64_golden() -> None:
    weights = tuple(
        f"{0x3F000000 + channel * 6 + band:08x}"
        for channel in range(16)
        for band in range(6)
    )
    asset = _asset(weight_words=weights)
    registered_observation = LfDetectionObservation.from_public_image_encoding(
        lf_carrier(ROOT_KEY, LATENT_SHAPE).template,
        LATENT_SHAPE,
    )
    null_observation = LfDetectionObservation.from_public_image_encoding(
        tuple(
            ((index % 37) - 18) / 37.0
            for index in range(16 * 64 * 64)
        ),
        LATENT_SHAPE,
    )
    wrong_keys = tuple(
        derive_wrong_key_material(
            identify_root_key(ROOT_KEY).root_key_public_digest,
            index,
        )
        for index in range(4)
    )
    calls = (
        (registered_observation, ROOT_KEY),
        (null_observation, ROOT_KEY),
        *((registered_observation, key) for key in wrong_keys),
    )
    legacy = tuple(
        lf_null_whitened_matched_detector(observation, key, asset)
        for observation, key in calls
    )
    prepared_observations = {
        observation.observation_digest: prepare_lf_null_whitened_observation(
            observation,
            asset,
        )
        for observation in (registered_observation, null_observation)
    }
    prepared_templates = {
        key: prepare_lf_null_whitened_template(key, asset)
        for key in (ROOT_KEY, *wrong_keys)
    }
    optimized = tuple(
        lf_null_whitened_matched_detector(
            observation,
            key,
            asset,
            prepared_observation=prepared_observations[
                observation.observation_digest
            ],
            prepared_template=prepared_templates[key],
        )
        for observation, key in calls
    )
    expected_score_bits = (
        "3ff0000000000000",
        "bf64defed55bde8b",
        "3f9c4a2b8080e3ae",
        "3f93ae326397e761",
        "3f8fe6674c91e727",
        "3f77947690515ad2",
    )
    expected_canonical_sha256 = (
        "8e8e386bb9ff0077984b46543b565f96a739a1e433584df78c1da8f57ae71fab",
        "1fa3099ff969913fca9bb53f9a366906a11d94ac57287c608b6ed6877f877e53",
        "b1a3a351eaba6fb18017149de81585f9b622c21e4b4ca3177282593f450eeb7a",
        "43617d0f42b9def70f1823d2ec2a0d617ff95876fe8cb8a55f76da7ebd8dd6bc",
        "0ef2184f9fb3a5cc27e4d39db77a28062865ce7466785af1f3e58286533cf0ac",
        "746f8c2ee71a484d18bc4cf291ee434535c900c3f97938a9dedf171584f86aca",
    )

    for index, (legacy_result, optimized_result) in enumerate(
        zip(legacy, optimized, strict=True)
    ):
        legacy_payload = json.dumps(
            asdict(legacy_result),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        optimized_payload = json.dumps(
            asdict(optimized_result),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        assert pack(">d", legacy_result.lf_score) == pack(
            ">d", optimized_result.lf_score
        )
        assert pack(">d", optimized_result.lf_score).hex() == (
            expected_score_bits[index]
        )
        assert legacy_payload == optimized_payload
        assert sha256(optimized_payload).hexdigest() == (
            expected_canonical_sha256[index]
        )
        assert asdict(legacy_result) == asdict(optimized_result)
        assert "prepared" not in optimized_payload.decode("utf-8")


@pytest.mark.unit
def test_lf_whitened_prepared_features_fail_closed_on_identity_drift() -> None:
    asset = _asset()
    observation = LfDetectionObservation.from_public_image_encoding(
        lf_carrier(ROOT_KEY, LATENT_SHAPE).template,
        LATENT_SHAPE,
    )
    different_observation = LfDetectionObservation.from_public_image_encoding(
        tuple(-value for value in observation.values),
        LATENT_SHAPE,
    )
    wrong_key = derive_wrong_key_material(
        identify_root_key(ROOT_KEY).root_key_public_digest,
        0,
    )
    prepared_observation = prepare_lf_null_whitened_observation(
        observation,
        asset,
    )
    prepared_template = prepare_lf_null_whitened_template(ROOT_KEY, asset)

    with pytest.raises(LfDetectorError, match="observation identity mismatch"):
        lf_null_whitened_matched_detector(
            different_observation,
            ROOT_KEY,
            asset,
            prepared_observation=prepared_observation,
            prepared_template=prepared_template,
        )
    with pytest.raises(LfDetectorError, match="template identity mismatch"):
        lf_null_whitened_matched_detector(
            observation,
            wrong_key,
            asset,
            prepared_observation=prepared_observation,
            prepared_template=prepared_template,
        )
    changed_asset = _asset(fit_manifest_sha256="b" * 64)
    with pytest.raises(LfDetectorError, match="observation identity mismatch"):
        lf_null_whitened_matched_detector(
            observation,
            ROOT_KEY,
            changed_asset,
            prepared_observation=prepared_observation,
            prepared_template=prepared_template,
        )


@pytest.mark.unit
def test_lf_whitened_prepared_coefficients_reject_mutable_layout_and_data_drift() -> None:
    asset = _asset()
    observation = LfDetectionObservation.from_public_image_encoding(
        lf_carrier(ROOT_KEY, LATENT_SHAPE).template,
        LATENT_SHAPE,
    )
    prepared = prepare_lf_null_whitened_observation(observation, asset)
    assert prepared.coefficients.dtype == np.dtype(np.float64)
    assert prepared.coefficients.flags.c_contiguous
    assert not prepared.coefficients.flags.writeable
    assert np.isfinite(prepared.coefficients).all()
    with pytest.raises(ValueError, match="WRITEABLE"):
        prepared.coefficients.setflags(write=True)

    writable = np.array(prepared.coefficients, dtype=np.float64, order="C")
    object.__setattr__(prepared, "coefficients", writable)
    with pytest.raises(LfDetectorError, match="read only"):
        prepared.validate()

    layout_drifted = prepare_lf_null_whitened_observation(observation, asset)
    non_contiguous = np.asfortranarray(layout_drifted.coefficients)
    non_contiguous.setflags(write=False)
    object.__setattr__(layout_drifted, "coefficients", non_contiguous)
    with pytest.raises(LfDetectorError, match="C contiguous"):
        layout_drifted.validate()

    nonfinite_drifted = prepare_lf_null_whitened_observation(
        observation,
        asset,
    )
    nonfinite = np.array(non_contiguous, dtype=np.float64, order="C")
    nonfinite[0, 0, 0] = np.nan
    nonfinite.setflags(write=False)
    object.__setattr__(nonfinite_drifted, "coefficients", nonfinite)
    with pytest.raises(LfDetectorError, match="finite"):
        nonfinite_drifted.validate()

    valid = prepare_lf_null_whitened_observation(observation, asset)
    object.__setattr__(valid, "coefficients_digest", "0" * 64)
    with pytest.raises(LfDetectorError, match="digest mismatch"):
        lf_null_whitened_matched_detector(
            observation,
            ROOT_KEY,
            asset,
            prepared_observation=valid,
        )

    writable = np.array(valid.coefficients, dtype=np.float64, order="C")
    with pytest.raises(LfDetectorError, match="read only"):
        PreparedLfWhitenedObservation(
            coefficients=writable,
            coefficients_digest=sha256(
                writable.tobytes(order="C")
            ).hexdigest(),
            observation_digest=observation.observation_digest,
            observation_shape=observation.shape,
            observation_protocol=observation.observation_protocol,
            whitening_asset_digest=asset.whitening_asset_digest,
        )


@pytest.mark.unit
def test_lf_whitened_detector_fails_closed_without_valid_asset_shape_or_norm() -> None:
    observation = LfDetectionObservation.from_public_image_encoding(
        lf_carrier(ROOT_KEY, LATENT_SHAPE).template,
        LATENT_SHAPE,
    )
    with pytest.raises(LfDetectorError, match="frozen public whitening asset"):
        lf_null_whitened_matched_detector(
            observation,
            ROOT_KEY,
            None,  # type: ignore[arg-type]
        )
    with pytest.raises(LfNullWhiteningAssetError, match="digest"):
        _asset(declared_digest="b" * 64)
    with pytest.raises(LfNullWhiteningAssetError, match="finite"):
        _asset(weight_words=("7f800000",) + ("3f800000",) * 95)

    wrong_shape = LfDetectionObservation.from_public_image_encoding(
        tuple(range(16)),
        (1, 16, 1, 1),
    )
    with pytest.raises(LfDetectorError, match="shape"):
        lf_null_whitened_matched_detector(
            wrong_shape,
            ROOT_KEY,
            _asset(),
        )

    zero_residual = LfDetectionObservation.from_public_image_encoding(
        (0.0,) * (16 * 64 * 64),
        LATENT_SHAPE,
    )
    with pytest.raises(LfDetectorError, match="strictly positive"):
        lf_null_whitened_matched_detector(
            zero_residual,
            ROOT_KEY,
            _asset(),
        )


@pytest.mark.unit
def test_lf_whitening_payload_rejects_identity_and_weight_shape_drift() -> None:
    asset = _asset()
    payload = asset.canonical_payload
    changed_identity = dict(payload)
    changed_identity["band_identity"] = "different_frequency_partition"
    with pytest.raises(LfNullWhiteningAssetError, match="identities drifted"):
        LfNullWhiteningAsset.from_canonical_payload(
            changed_identity,
            whitening_asset_digest=asset.whitening_asset_digest,
        )

    missing_weight = dict(payload)
    missing_weight["weights_binary32_be_hex"] = list(
        payload["weights_binary32_be_hex"]
    )[:-1]
    with pytest.raises(LfNullWhiteningAssetError, match="exactly 96"):
        LfNullWhiteningAsset.from_canonical_payload(
            missing_weight,
            whitening_asset_digest=asset.whitening_asset_digest,
        )


@pytest.mark.unit
def test_lf_whitened_dct_matches_registered_formula_at_selected_coefficient() -> None:
    values = tuple(
        ((index % 113) - 56) / 113.0
        for index in range(16 * 64 * 64)
    )
    coefficients = _affine_detrended_dct(values, role="test observation")
    coordinates = tuple((2.0 * index - 63.0) / 63.0 for index in range(64))
    coordinate_squared_sum = sum(value * value for value in coordinates)
    channel_values = values[: 64 * 64]
    constant_sum = 0.0
    for height in range(64):
        for width in range(64):
            constant_sum += channel_values[height * 64 + width]
    constant = constant_sum / (64.0 * 64.0)
    height_sum = 0.0
    for height in range(64):
        for width in range(64):
            height_sum += (
                coordinates[height]
                * channel_values[height * 64 + width]
            )
    height_slope = height_sum / (64.0 * coordinate_squared_sum)
    width_sum = 0.0
    for height in range(64):
        for width in range(64):
            width_sum += (
                coordinates[width]
                * channel_values[height * 64 + width]
            )
    width_slope = width_sum / (64.0 * coordinate_squared_sum)
    height_frequency = 3
    width_frequency = 5
    direct_sum = 0.0
    fixed_pi = float.fromhex("0x1.921fb54442d18p+1")
    for height in range(64):
        for width in range(64):
            residual = (
                channel_values[height * 64 + width]
                - constant
                - height_slope * coordinates[height]
                - width_slope * coordinates[width]
            )
            height_basis = sqrt(2.0 / 64.0) * cos(
                fixed_pi
                * (height + 0.5)
                * height_frequency
                / 64.0
            )
            width_basis = sqrt(2.0 / 64.0) * cos(
                fixed_pi
                * (width + 0.5)
                * width_frequency
                / 64.0
            )
            direct_sum += residual * height_basis * width_basis
    assert coefficients[0, height_frequency, width_frequency].item().hex() == (
        direct_sum.hex()
    )


@pytest.mark.constraint
def test_lf_whitening_method_plane_has_no_runtime_experiment_or_governance_import() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / "main/content_chain/lf_detector.py",
        root / "main/content_chain/lf_whitening.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])
        assert imported_roots.isdisjoint({"runtime", "experiments", "governance"})
