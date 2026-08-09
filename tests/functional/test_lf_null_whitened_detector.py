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
    LfNullWhiteningAsset,
    lf_carrier,
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
    whitened = adapter.detect_lf_null_whitened(
        observation,
        root_key,
        _identity_asset(),
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
    assert raw.result_identity != whitened.result_identity
