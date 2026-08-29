from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol import geometry_v4

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "geometry_v4" / geometry_v4.GEOMETRY_V4_CONFIG_NAME


@pytest.mark.unit
def test_p0_contract_is_stable_bounded_and_preserves_content_boundary() -> None:
    contract = geometry_v4.load_geometry_v4_p0_contract(_ROOT)
    raw = _CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == geometry_v4.GEOMETRY_V4_CONFIG_SHA256
    assert raw == (json.dumps(contract, ensure_ascii=True, indent=2) + "\n").encode("ascii")
    assert contract["identities"] == {
        "generated_writer_id": geometry_v4.GEOMETRY_V4_GENERATED_WRITER_ID,
        "method_id": geometry_v4.GEOMETRY_V4_METHOD_ID,
        "proxy_writer_id": geometry_v4.GEOMETRY_V4_PROXY_WRITER_ID,
    }
    content = contract["content_chain_boundary"]
    assert content["content_key_bytes"] == "reuse_existing_identity_unchanged"
    assert content["ordinary_rgb_weighted_joint_score"] == "reuse_unchanged"
    assert content["tau_status"].startswith("absent_on_main")
    assert content["tau_source"] == "F0_only"
    assert content["weighted_joint_boundary"] == "same_score_only_geometry_score_cannot_contribute"
    assert contract["claim_ceiling"].endswith("science_denominator_0")
    assert contract["geometry_key"]["raw_key_material"] == "forbidden_in_artifacts"
    assert contract["geometry_key"]["raw_pattern"] == "forbidden_in_artifacts"


@pytest.mark.unit
def test_p0_contract_freezes_stages_units_and_prohibitions() -> None:
    contract = geometry_v4.load_geometry_v4_p0_contract(_ROOT)
    assert contract["status_enum"] == list(geometry_v4.GEOMETRY_V4_STATUS)
    assert contract["stages"] == [
        "P1D_RGB_PROXY", "P1C_RGB_PROXY",
        "G0_GENERATED_CALLBACK_STEP_19_FINAL_LATENT_BEFORE_VAE_DECODE", "G1", "H0D", "H0C",
        "F0_FULL_TWO_DETECTION_STRATEGY_FIXED_FPR_CALIBRATION", "E0", "R0",
    ]
    assert contract["units"] == {
        "failure_policy": "retain_declared_unit_without_replacement_retry_or_fallback",
        "primary_unit": "physical_image_instance_x_predeclared_attack",
        "wrong_key_policy": "within_same_unit_control_not_additional_denominator",
    }
    assert contract["prohibitions"] == {
        "geometry_positive_watermark_decision": True,
        "replacement_units_allowed": False,
        "retry_units_allowed": False,
        "unit_fallback_allowed": False,
        "v3_inheritance": False,
    }
    tiles = contract["local_tile_contract"]
    assert tiles["coordinate_policy"] == "fixed_canonical"
    assert tiles["content_adaptive_coordinate_change"] == "forbidden"
    assert tiles["content_adaptive_identity_change"] == "forbidden"


@pytest.mark.unit
def test_geometry_key_is_domain_separated_and_never_part_of_observation() -> None:
    key = b"a sufficiently long detection key"
    assert geometry_v4.derive_geometry_v4_key(key) == geometry_v4.derive_geometry_v4_key(key)
    assert geometry_v4.derive_geometry_v4_key(key, salt=b"one") != geometry_v4.derive_geometry_v4_key(key, salt=b"two")
    assert len(geometry_v4.derive_geometry_v4_key(key, length=48)) == 48
    with pytest.raises(TypeError):
        geometry_v4.derive_geometry_v4_key("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        geometry_v4.derive_geometry_v4_key(key, length=0)
    observation = geometry_v4.GeometryV4Observation(None, (), 0, 0.0, "STOPPED")
    assert tuple(observation.__dataclass_fields__) == ("H_hat", "corners_hat", "support", "reliability", "status")
    with pytest.raises(ValueError, match="status"):
        geometry_v4.GeometryV4Observation(None, (), 0, 0.0, "POSITIVE")


@pytest.mark.unit
def test_loader_fails_closed_on_contract_drift(tmp_path: Path) -> None:
    target = tmp_path / "configs" / "geometry_v4"
    target.mkdir(parents=True)
    altered = json.loads(_CONFIG.read_bytes())
    altered["prohibitions"]["retry_units_allowed"] = True
    (target / geometry_v4.GEOMETRY_V4_CONFIG_NAME).write_text(
        json.dumps(altered, ensure_ascii=True, indent=2) + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="bytes differ"):
        geometry_v4.load_geometry_v4_p0_contract(tmp_path)
    geometry_v4.require_geometry_v4_contract_digest(geometry_v4.GEOMETRY_V4_CONFIG_SHA256)
    with pytest.raises(ValueError, match="64-hex"):
        geometry_v4.require_geometry_v4_contract_digest("not-a-digest")
