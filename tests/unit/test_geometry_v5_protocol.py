from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from cegwm.protocol import geometry_v5 as geometry


_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = _ROOT / geometry.GEOMETRY_V5_P0_CONTRACT_PATH
_ROOT_KEY = b"geometry-v5-root-key-for-unit-tests-only"
_IDENTITY_H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_UNIT_CORNERS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _conditions(**changes: bool) -> geometry.ReliabilityConditions:
    values = {
        "search_candidate": True,
        "fit_support": True,
        "macro_region_coverage": True,
        "residual": True,
        "holdout_correlation_psr": True,
        "cross_scale_rs_consistency": True,
        "holdout_disjoint": True,
        "legal_conditioning": True,
    }
    values.update(changes)
    return geometry.ReliabilityConditions(**values)


@pytest.mark.unit
def test_p0_contract_bytes_digest_identities_and_stage_order_are_frozen() -> None:
    contract = geometry.load_geometry_v5_p0_contract(_ROOT)
    raw = _CONTRACT_PATH.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == geometry.GEOMETRY_V5_P0_CONTRACT_SHA256
    assert raw == geometry.canonical_json_bytes(json.loads(raw))
    assert contract.byte_sha256 == geometry.GEOMETRY_V5_P0_CONTRACT_SHA256
    assert contract.config["method_id"] == geometry.GEOMETRY_V5_METHOD_ID
    assert contract.config["stage"]["current_stage"] == "V5-P0"
    assert contract.config["stage"]["preferred_progression"] == (
        "V5-M0_SD2_1_faithful_global_RST_reproduction",
        "V5-M1_key_domain_separation_and_holdout_safety",
        "V5-C0_keyed_local_latent_tiles_for_crop_crop_rescale",
        "V5-I0_unchanged_content_detector_integration",
        "V5-SD35_after_method_freeze_and_separately_proven_fixed_repeatable_inversion",
    )
    assert contract.config["evidence_ceiling"] == geometry.GEOMETRY_V5_P0_CLAIM_CEILING


@pytest.mark.unit
def test_p0_contract_loader_rejects_any_byte_or_canonical_json_drift(tmp_path: Path) -> None:
    target = tmp_path / geometry.GEOMETRY_V5_P0_CONTRACT_PATH
    target.parent.mkdir(parents=True)
    shutil.copyfile(_CONTRACT_PATH, target)
    target.write_bytes(target.read_bytes().replace(b'"V5-P0"', b'"V5-PX"'))

    with pytest.raises(ValueError, match="byte digest differs"):
        geometry.load_geometry_v5_p0_contract(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, True])
def test_p0_digest_validation_requires_lowercase_64_hex(digest: object) -> None:
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        geometry.validate_lowercase_sha256(digest)  # type: ignore[arg-type]


@pytest.mark.unit
def test_hkdf_key_domains_are_deterministic_distinct_and_not_content_derived() -> None:
    first = geometry.derive_geometry_v5_key_domain_digests(_ROOT_KEY)
    second = geometry.derive_geometry_v5_key_domain_digests(_ROOT_KEY)
    changed = geometry.derive_geometry_v5_key_domain_digests(_ROOT_KEY + b"x")

    assert tuple(first) == ("k_search", "k_fit", "k_validate")
    assert first == second
    assert len(set(first.values())) == 3
    assert first != changed
    assert all(len(value) == 64 for value in first.values())
    assert "content_subkey" in geometry.load_geometry_v5_p0_contract(_ROOT).config[
        "key_hierarchy"
    ]["source_forbidden"]
    with pytest.raises(ValueError, match="non-empty"):
        geometry.derive_geometry_v5_key_domain_digests(b"")


@pytest.mark.unit
def test_detector_boundary_is_exactly_blind_and_forbids_embed_truth_and_content_inputs() -> None:
    contract = geometry.load_geometry_v5_p0_contract(_ROOT).config
    boundary = contract["detector_boundary"]
    allowed = (
        "attacked_ordinary_RGB",
        "geometry_root_key",
        "frozen_model_scheduler_inversion_identities",
    )

    assert geometry.validate_detector_input_names(allowed) == allowed
    assert set(boundary["forbidden_inputs"]) >= {
        "clean_RGB", "pre_attack_RGB", "original_z_T", "writer_tensors",
        "writer_residuals", "true_transform_parameters", "true_crop_parameters",
        "true_attack_parameters", "content_scores", "content_keys", "evaluation_truth",
        "retry", "fallback",
    }
    with pytest.raises(ValueError, match="exact blind boundary"):
        geometry.validate_detector_input_names((*allowed, "original_z_T"))


@pytest.mark.unit
def test_holdout_role_is_disjoint_and_content_never_votes_or_changes_tau() -> None:
    contract = geometry.load_geometry_v5_p0_contract(_ROOT).config
    validate = contract["roles"]["k_validate"]
    content = contract["content_integration"]

    assert validate["may_participate_in"] == ()
    assert set(validate["forbidden_participation"]) == {
        "candidate_proposal", "correspondence", "parameter_estimation", "tie_break",
        "threshold_tuning", "fallback",
    }
    assert content["geometry_may_add_positive_evidence"] is False
    assert content["s1"] == "exact_same_content_path_and_tau"
    assert content["tau_or_delta"] == "unbound"


@pytest.mark.unit
def test_public_output_is_exact_and_reliable_requires_all_structural_conditions() -> None:
    assert tuple(field.name for field in dataclasses.fields(geometry.GeometryV5Observation)) == (
        "H_hat", "corners_hat", "support", "reliability", "status"
    )
    observation = geometry.GeometryV5Observation(
        _IDENTITY_H, _UNIT_CORNERS, 1, 1.0, geometry.GeometryV5Status.RELIABLE, _conditions()
    )
    assert observation.status is geometry.GeometryV5Status.RELIABLE
    assert observation.H_hat == _IDENTITY_H
    with pytest.raises(ValueError, match="complete legal structural conditions"):
        geometry.GeometryV5Observation(
            _IDENTITY_H, _UNIT_CORNERS, 1, 1.0, "RELIABLE", _conditions(holdout_disjoint=False)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("matrix", "corners", "message"),
    [
        (((True, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), _UNIT_CORNERS, "non-bool"),
        (((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), _UNIT_CORNERS, "finite"),
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.0, 1.0)), _UNIT_CORNERS, "similarity"),
        (((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), _UNIT_CORNERS, "similarity"),
        (_IDENTITY_H, ((0.0, 0.0), (1.0, 0.0), (0.2, 0.2), (0.0, 1.0)), "strict convex"),
        (_IDENTITY_H, ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)), "inconsistent"),
    ],
)
def test_reliable_output_fails_closed_for_nonfinite_malformed_or_mismatched_geometry(
    matrix: object, corners: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        geometry.GeometryV5Observation(
            matrix,  # type: ignore[arg-type]
            corners,  # type: ignore[arg-type]
            1,
            0.5,
            "RELIABLE",
            _conditions(),
        )


@pytest.mark.unit
def test_unreliable_and_stopped_outputs_cannot_fabricate_geometry_or_fallback() -> None:
    unreliable = geometry.GeometryV5Observation(None, None, 0, 0.0, "UNRELIABLE")
    stopped = geometry.GeometryV5Observation(None, None, 0, 0.0, "STOPPED")
    assert unreliable.status is geometry.GeometryV5Status.UNRELIABLE
    assert stopped.status is geometry.GeometryV5Status.STOPPED
    with pytest.raises(ValueError, match="UNRELIABLE must not export"):
        geometry.GeometryV5Observation(_IDENTITY_H, _UNIT_CORNERS, 1, 0.2, "UNRELIABLE")
    with pytest.raises(ValueError, match="STOPPED must not export"):
        geometry.GeometryV5Observation(_IDENTITY_H, _UNIT_CORNERS, 1, 0.2, "STOPPED")


@pytest.mark.unit
def test_claim_ceiling_and_later_parameters_remain_explicitly_unbound() -> None:
    contract = geometry.load_geometry_v5_p0_contract(_ROOT).config
    assert contract["evidence_ceiling"] == "P0_local_static_engineering_only_science_denominator_0"
    assert set(contract["later_pre_run_contracts"]) == {
        "numeric_gates_and_thresholds", "tile_layout", "X_template_parameters",
        "inversion_schedule", "fixed_seeds", "attack_roster", "denominators",
    }
