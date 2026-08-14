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
    NullScoreRecord,
    SaliencyBranchNullCalibration,
    saliency_max_standardized_content_detector,
)
from main.content_chain.hf_detector import HfDetectionObservation, hf_detector
from main.content_chain.lf_carrier import lf_carrier
from main.content_chain.lf_detector import (
    LfDetectionObservation,
    SaliencyMaskedLfDetectionObservation,
    LfDetectorError,
    PreparedLfWhitenedObservation,
    _affine_detrended_dct,
    lf_detector,
    lf_null_whitened_matched_detector,
    prepare_lf_null_whitened_observation,
    prepare_lf_null_whitened_template,
    lf_saliency_masked_null_whitened_matched_detector,
)
from main.content_chain.lf_whitening import (
    LF_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfNullWhiteningAsset,
    LfNullWhiteningAssetError,
    LF_SALIENCY_MASKED_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID,
    LfSaliencyMaskedNullWhiteningAsset,
)
from main.content_chain.routing import (
    SaliencyProbabilityObservation,
    inspyrenet_salient_local_lf_router,
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


def _masked_asset() -> LfSaliencyMaskedNullWhiteningAsset:
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


def _detect_route():
    image_digest = sha256(b"masked-lf-public-image").hexdigest()
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
    return inspyrenet_salient_local_lf_router(LATENT_SHAPE, probability)


@pytest.mark.unit
def test_saliency_masked_whitened_detector_binds_mask_asset_and_wrong_key() -> None:
    asset = _masked_asset()
    route = _detect_route()
    carrier = lf_carrier(ROOT_KEY, LATENT_SHAPE)
    observation = SaliencyMaskedLfDetectionObservation.from_public_image_encoding(
        carrier.template,
        LATENT_SHAPE,
        public_input_image_digest=route.input_image_digest,
    )
    registered = lf_saliency_masked_null_whitened_matched_detector(
        observation,
        ROOT_KEY,
        asset,
        route,
    )
    wrong = lf_saliency_masked_null_whitened_matched_detector(
        observation,
        derive_wrong_key_material(identify_root_key(ROOT_KEY).root_key_public_digest, 0),
        asset,
        route,
    )
    primary_null_observation = SaliencyMaskedLfDetectionObservation.from_public_image_encoding(
        tuple(((index % 37) - 18) / 37.0 for index in range(16 * 64 * 64)),
        LATENT_SHAPE,
        public_input_image_digest=route.input_image_digest,
    )
    primary_null = lf_saliency_masked_null_whitened_matched_detector(
        primary_null_observation,
        ROOT_KEY,
        asset,
        route,
    )
    assert registered.candidate_id == LF_SALIENCY_MASKED_NULL_WHITENED_MATCHED_SCORE_CANDIDATE_ID
    assert registered.whitening_asset_digest == asset.whitening_asset_digest
    assert registered.saliency_route_identity == route.route_identity
    assert primary_null.whitening_asset_digest == asset.whitening_asset_digest
    assert primary_null.detector_config_digest == registered.detector_config_digest
    assert registered.masked_observation_digest != registered.observation_digest
    assert registered.lf_score > wrong.lf_score
    hf_result = hf_detector(
        HfDetectionObservation.from_public_image_encoding(
            observation.values,
            LATENT_SHAPE,
        ),
        ROOT_KEY,
    )
    hf_null = SaliencyBranchNullCalibration(
        branch="hf",
        detector_identity=hf_result.detector_identity,
        partition_identity="masked-lf-null-fit-partition",
        records=tuple(
            NullScoreRecord(
                -1.0 + index / 32.0,
                f"null-{index:02d}",
                f"hf-{index:02d}",
            )
            for index in range(32)
        ),
    )
    lf_null = SaliencyBranchNullCalibration(
        branch="lf",
        detector_identity=registered.detector_identity,
        partition_identity="masked-lf-null-fit-partition",
        records=tuple(
            NullScoreRecord(
                -1.0 + index / 32.0,
                f"null-{index:02d}",
                f"lf-{index:02d}",
            )
            for index in range(32)
        ),
    )
    combined = saliency_max_standardized_content_detector(
        hf_result,
        registered,
        hf_null=hf_null,
        lf_null=lf_null,
    )
    assert combined.candidate_id == "content_combination_saliency_max_standardized"
    assert combined.combined_score == max(
        combined.hf_standardization.z_score,
        combined.lf_standardization.z_score,
    )
    assert combined.diagnostic_only and not combined.promoted
    with pytest.raises(LfDetectorError):
        lf_saliency_masked_null_whitened_matched_detector(
            observation,
            ROOT_KEY,
            _asset(),
            route,
        )
    cross_image_observation = (
        SaliencyMaskedLfDetectionObservation.from_public_image_encoding(
            observation.values,
            LATENT_SHAPE,
            public_input_image_digest=sha256(b"different-public-image").hexdigest(),
        )
    )
    with pytest.raises(LfDetectorError):
        lf_saliency_masked_null_whitened_matched_detector(
            cross_image_observation,
            ROOT_KEY,
            asset,
            route,
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
def test_lf_whitened_prepared_features_preserve_legacy_binary64_results() -> None:
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
        "3f820c26b18e0b5e",
        "bf94abf44cc6955a",
        "bf8d8f62b6f69a5e",
        "bf8bf8832447c2f8",
        "bf7f20b427919497",
    )
    expected_canonical_sha256 = (
        "837dbd2535c224824d3858ae497691c2e5443e885fed52e9d80a35e2c11e72f4",
        "52a7acf32fc9e19dbb036e96de9adecda9b7e99a13aa37eb4ce268960e7bf75d",
        "e2d5a7c990d01d67fd8cbe1344179d24b91ad0b52527369ce95b150ce2122bba",
        "fc737c1a5b3e985a6fd2933c97fe784cc1f4102fbfed102239c7b6c3eab6589a",
        "9f76f94cd2ff33c80872b36fe9144ab67f0d31a357cd3830dbcabec0898f9269",
        "cf82c1469ea4fbca8e6de32834bd9cbb7cb417a892aca6c3d524f31d627ac184",
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
    model_drifted_observation = prepare_lf_null_whitened_observation(
        observation,
        asset,
    )
    object.__setattr__(
        model_drifted_observation,
        "model_revision",
        "different-model-revision",
    )
    with pytest.raises(LfDetectorError, match="observation model mismatch"):
        lf_null_whitened_matched_detector(
            observation,
            ROOT_KEY,
            asset,
            prepared_observation=model_drifted_observation,
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
            model_revision=prepared.model_revision,
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
