"""Candidate-specific CPU checks for the LF clean-null-whitened detector."""

from __future__ import annotations

import ast
from hashlib import sha256
from math import cos, sqrt
from pathlib import Path

import pytest

from main.content_chain.lf_carrier import lf_carrier
from main.content_chain.lf_detector import (
    LfDetectionObservation,
    LfDetectorError,
    _affine_detrended_dct,
    lf_detector,
    lf_null_whitened_matched_detector,
)
from main.content_chain.lf_whitening import (
    LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfNullWhiteningAsset,
    LfNullWhiteningAssetError,
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
) -> LfNullWhiteningAsset:
    payload = {
        "artifact_role": "lf_clean_null_whitening_operator",
        "band_identity": "six_dyadic_chebyshev_frequency_rings_without_dc",
        "candidate_id": LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
        "detrend_identity": "per_channel_affine_plane_normalized_coordinates",
        "fit_manifest_sha256": "a" * 64,
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
            direct_sum += (
                residual
                * cos(
                    fixed_pi
                    * (height + 0.5)
                    * height_frequency
                    / 64.0
                )
                * cos(
                    fixed_pi
                    * (width + 0.5)
                    * width_frequency
                    / 64.0
                )
            )
    expected = sqrt(2.0 / 64.0) * sqrt(2.0 / 64.0) * direct_sum
    assert coefficients[0][height_frequency][width_frequency] == pytest.approx(
        expected,
        abs=2e-14,
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
