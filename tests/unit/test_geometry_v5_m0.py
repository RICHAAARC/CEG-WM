from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.method import geometry_v5_m0 as method
from cegwm.protocol import geometry_v5_m0 as protocol


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_m0_byte_bindings_sources_roster_and_engineering_ceiling_are_frozen() -> None:
    contract = protocol.load_geometry_v5_m0_contract(_ROOT)
    assert hashlib.sha256((_ROOT / protocol.M0_CONFIG_PATH).read_bytes()).hexdigest() == protocol.M0_CONFIG_SHA256
    assert hashlib.sha256((_ROOT / protocol.M0_MANIFEST_PATH).read_bytes()).hexdigest() == protocol.M0_MANIFEST_SHA256
    assert tuple(unit.seed for unit in contract.units) == (7501, 7502, 7503, 7504)
    assert all(unit.seed not in {6201, 6202, 6203, 6204} for unit in contract.units)
    assert len(contract.units) * len(contract.config["development"]["attacks"]) == 44
    assert [attack["attack_id"] for attack in contract.config["development"]["attacks"]] == [
        "identity", "rotation_-10", "rotation_+10", "scale_0.9", "scale_1.1",
        "translation_x_-0.08", "translation_x_+0.08", "translation_y_-0.08",
        "translation_y_+0.08", "compound_rot+7_scale0.93_tx+0.05_ty-0.04",
        "compound_rot-7_scale1.07_tx-0.05_ty+0.04",
    ]
    assert contract.config["source_bindings"]["maxsive"]["exact"] == "a9554024aed176e705cc15ca1cbd31b9c7f75bfb"
    assert contract.config["source_bindings"]["tree_ring"]["exact"] == "3015283d9cf82e90b628f02ad2121bd37408ca9a"
    assert contract.config["engineering_evaluation"]["claim_ceiling"] == protocol.M0_CLAIM_CEILING


@pytest.mark.unit
def test_m0_template_initial_z_t_injection_and_similarity_direction_are_pure_math_only() -> None:
    template = method.build_hermitian_x_template()
    assert len(template) == 16
    latent = tuple(tuple(tuple(0.0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    injected = method.inject_initial_z_t_x_template(latent, template)
    assert len(injected) == 4 and any(value != 0.0 for row in injected[3] for value in row)
    estimate = method.estimate_rotation_scale_from_peak_pairs(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    )
    assert estimate.rotation_degrees == pytest.approx(0.0)
    assert estimate.scale == pytest.approx(2.0)
    H = method.assemble_attacked_to_canonical_similarity(0.0, 2.0, -0.5, -0.5)
    assert H == ((2.0, -0.0, -0.5), (0.0, 2.0, -0.5), (0.0, 0.0, 1.0))


@pytest.mark.unit
def test_m0_raw_output_has_no_reliable_rectification_or_fabricated_failure() -> None:
    failed = protocol.GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})
    assert failed.status is protocol.M0RawStatus.FAILED
    with pytest.raises(ValueError, match="fabricate"):
        protocol.GeometryV5M0RawRecord("FAILED", 0.0, None, None, None, None, {})
    available = protocol.GeometryV5M0RawRecord(
        "ESTIMATE_AVAILABLE", 0.0, 1.0, 0.0, 0.0,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), {"phase_peak": 1.0},
    )
    assert available.status is protocol.M0RawStatus.ESTIMATE_AVAILABLE
    scope = protocol.load_geometry_v5_m0_contract(_ROOT).config["scope"]
    assert scope["may_emit_RELIABLE"] is False and scope["may_rectify"] is False and scope["may_vote_content"] is False


@pytest.mark.unit
def test_m0_contract_rejects_noncanonical_bytes_and_forbidden_detector_inputs_are_recorded() -> None:
    raw = (_ROOT / protocol.M0_CONFIG_PATH).read_bytes()
    assert raw == protocol.canonical_json_bytes(json.loads(raw))
    forbidden = protocol.load_geometry_v5_m0_contract(_ROOT).config["scope"]["detector_forbidden_inputs"]
    assert set(forbidden) >= {"original_prompt", "original_z_T", "clean_RGB", "true_H", "evaluation_truth"}
