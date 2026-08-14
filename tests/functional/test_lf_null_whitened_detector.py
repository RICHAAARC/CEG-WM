"""Lightweight real adapter traversal for the LF whitening candidate."""

from __future__ import annotations

from hashlib import sha256

import pytest

from experiments.methods.ceg_wm import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from main import (
    LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfDetectionObservation,
    SaliencyMaskedLfDetectionObservation,
    LfNullWhiteningAsset,
    lf_carrier,
    prepare_lf_null_whitened_observation,
    prepare_lf_null_whitened_template,
    LF_SALIENCY_MASKED_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfSaliencyMaskedNullWhiteningAsset,
    SaliencyProbabilityObservation,
    inspyrenet_salient_local_lf_router,
    lf_saliency_masked_null_whitened_matched_detector,
)
from main.shared.key_schedule import stable_json_utf8


def _identity_asset() -> LfNullWhiteningAsset:
    payload = {
        "artifact_role": "lf_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "c" * 64,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": ["3f800000"] * 96,
    }
    return LfNullWhiteningAsset.from_canonical_payload(
        payload,
        whitening_asset_digest=sha256(stable_json_utf8(payload)).hexdigest(),
    )


def _masked_identity_asset() -> LfSaliencyMaskedNullWhiteningAsset:
    payload = {
        "artifact_role": "lf_saliency_masked_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": LF_SALIENCY_MASKED_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "d" * 64,
        "fit_source_cluster_count": 32,
        "latent_shape": [1, 16, 64, 64],
        "observation_protocol": "final_image_vae_posterior_mode",
        "regularization_ratio": "0x1.0000000000000p-10",
        "saliency_mask_protocol": (
            "detect_public_rgb8_inspyrenet_probability_bilinear64_threshold_0.5_erosion3"
        ),
        "transform_identity": "orthonormal_dct_ii",
        "weights_binary32_be_hex": ["3f800000"] * 96,
    }
    return LfSaliencyMaskedNullWhiteningAsset.from_canonical_payload(
        payload,
        whitening_asset_digest=sha256(stable_json_utf8(payload)).hexdigest(),
    )


@pytest.mark.quick
def test_saliency_masked_lf_candidate_crosses_main_public_api_without_raw_fallback() -> None:
    root_key = "ceg-wm-saliency-masked-lf-functional-key"
    shape = (1, 16, 64, 64)
    carrier = lf_carrier(root_key, shape)
    image_digest = sha256(b"functional-masked-lf-public-image").hexdigest()
    observation = SaliencyMaskedLfDetectionObservation.from_public_image_encoding(
        carrier.template,
        shape,
        public_input_image_digest=image_digest,
    )
    probability = SaliencyProbabilityObservation(
        values=tuple(
            0.75 if 8 <= row < 56 and 8 <= column < 56 else 0.25
            for row in range(64)
            for column in range(64)
        ),
        spatial_shape=(64, 64),
        observation_role="detect_public_rgb8",
        input_image_digest=image_digest,
    )
    route = inspyrenet_salient_local_lf_router(shape, probability)
    result = lf_saliency_masked_null_whitened_matched_detector(
        observation,
        root_key,
        _masked_identity_asset(),
        route,
    )
    assert result.candidate_id == LF_SALIENCY_MASKED_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID
    assert result.saliency_route_identity == route.route_identity
    assert result.whitening_asset_digest != _identity_asset().whitening_asset_digest


@pytest.mark.quick
def test_lf_whitened_candidate_crosses_real_public_adapter_without_raw_fallback() -> None:
    root_key = "ceg-wm-lf-whitened-functional-key"
    shape = (1, 16, 64, 64)
    carrier = lf_carrier(root_key, shape)
    observation = LfDetectionObservation.from_public_image_encoding(
        carrier.template,
        shape,
    )
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration()
    )

    raw = adapter.detect_lf(observation, root_key)
    asset = _identity_asset()
    whitened = adapter.detect_lf_null_whitened(
        observation,
        root_key,
        asset,
    )
    prepared = adapter.detect_lf_null_whitened(
        observation,
        root_key,
        asset,
        prepared_observation=prepare_lf_null_whitened_observation(
            observation,
            asset,
        ),
        prepared_template=prepare_lf_null_whitened_template(
            root_key,
            asset,
        ),
    )

    assert raw.public_callable == "main.lf_detector"
    assert whitened.public_callable == "main.lf_null_whitened_matched_detector"
    assert raw.result.candidate_id == "lf_low_pass"
    assert (
        whitened.result.candidate_id
        == LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID
    )
    assert raw.result_identity == raw.result.detector_identity
    assert whitened.result_identity == whitened.result.detector_identity
    assert prepared == whitened
    assert raw.result_identity != whitened.result_identity
