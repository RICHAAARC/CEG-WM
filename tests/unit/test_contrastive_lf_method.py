from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

from main import (
    ContrastiveLfRawObservation,
    LfDetectionObservation,
    contrastive_lf_carrier,
    contrastive_lf_detector,
    contrastive_lf_raw_observation,
    contrastive_lowpass,
    derive_internal_lf_decoy_material,
    derive_wrong_key_material,
    fit_contrastive_lf_null_asset,
    identify_root_key,
)
from main.content_chain.lf_carrier import LfCarrierError


ROOT = "stage-a-method-unit-root"
SHAPE = (1, 16, 3, 3)
pytestmark = pytest.mark.unit


def test_contrastive_carriers_use_independent_registered_domains() -> None:
    multiscale = contrastive_lf_carrier(
        ROOT, SHAPE, candidate_id="lf_multiscale_lowpass_contrastive"
    )
    single = contrastive_lf_carrier(
        ROOT, SHAPE, candidate_id="lf_five_by_five_lowpass_contrastive"
    )
    assert multiscale.scale_nine_domain_digest is not None
    assert len(
        {
            multiscale.scale_five_domain_digest,
            multiscale.scale_nine_domain_digest,
            single.scale_five_domain_digest,
        }
    ) == 3
    assert multiscale.scale_five_template != single.scale_five_template
    assert sum(value * value for value in multiscale.direction) == pytest.approx(1.0, abs=2e-5)


def test_internal_decoy_and_external_wrong_are_separate_capabilities() -> None:
    root_digest = identify_root_key(ROOT).root_key_public_digest
    internal = derive_internal_lf_decoy_material(
        root_digest, "lf_multiscale_lowpass_contrastive", 0
    )
    wrong = derive_wrong_key_material(root_digest, 0)
    assert internal.material_text.startswith("ceg-wm-internal-lf-decoy:")
    assert wrong.material_text.startswith("ceg-wm-wrong-key:")
    assert internal.material_text != wrong.material_text
    internal_carrier = contrastive_lf_carrier(
        internal, SHAPE, candidate_id="lf_multiscale_lowpass_contrastive"
    )
    wrong_carrier = contrastive_lf_carrier(
        wrong, SHAPE, candidate_id="lf_multiscale_lowpass_contrastive"
    )
    assert internal_carrier.key_role == "internal_decoy"
    assert wrong_carrier.key_role == "wrong"
    assert internal_carrier.scale_five_domain_digest != wrong_carrier.scale_five_domain_digest


def test_lowpass_is_per_channel_zero_padded_and_count_includes_pad() -> None:
    values = [0.0] * (1 * 16 * 3 * 3)
    values[4] = 25.0
    values[9 + 4] = 50.0
    pooled = contrastive_lowpass(values, SHAPE, 5)
    assert pooled[0] == 1.0
    assert pooled[9] == 2.0
    assert all(value == 0.0 for value in pooled[18:])


def _raw(index: int, candidate: str) -> ContrastiveLfRawObservation:
    dimension = 2 if candidate == "lf_multiscale_lowpass_contrastive" else 1
    feature = (float(index), float((index * 7) % 11))[:dimension]
    decoys = tuple(
        tuple(value - 0.1 * (decoy + 1) for value in feature)
        for decoy in range(8)
    )
    return ContrastiveLfRawObservation(
        candidate_id=candidate,
        raw_feature=feature,
        internal_decoy_features=decoys,
        observation_digest=f"{index + 1:064x}",
        carrier_config_digest="a" * 64,
        root_key_public_digest="b" * 64,
        key_role="registered",
        wrong_key_index=None,
        raw_observation_digest=f"{index + 33:064x}",
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "lf_multiscale_lowpass_contrastive",
        "lf_five_by_five_lowpass_contrastive",
    ],
)
def test_candidate_specific_population_assets_and_scores(candidate: str) -> None:
    population = tuple(_raw(index, candidate) for index in range(32))
    asset = fit_contrastive_lf_null_asset(
        population, candidate_id=candidate, null_manifest_digest="c" * 64
    )
    result = contrastive_lf_detector(population[10], asset)
    assert asset.population_count == 32
    assert asset.contrastive_population_sigma > 0.0
    assert len(result.internal_decoy_scores) == 8
    assert isfinite(result.standardized_score)
    if candidate == "lf_multiscale_lowpass_contrastive":
        assert len(asset.whitening_matrix) == 4
    else:
        assert len(asset.whitening_matrix) == 1


def test_raw_detector_does_not_accept_candidate_mismatched_internal_material() -> None:
    root_digest = identify_root_key(ROOT).root_key_public_digest
    internal = derive_internal_lf_decoy_material(
        root_digest, "lf_multiscale_lowpass_contrastive", 0
    )
    with pytest.raises(LfCarrierError):
        contrastive_lf_carrier(
            internal,
            SHAPE,
            candidate_id="lf_five_by_five_lowpass_contrastive",
        )
