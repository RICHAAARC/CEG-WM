from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from cegwm.protocol.stage_a import load_stage_a_protocol

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "stage_a" / "stage_a_v1.json"
_SELECTION = _ROOT / "configs" / "stage_a" / "candidate_selection.jsonl"
_CONFIRMATION = _ROOT / "configs" / "stage_a" / "untouched_confirmation.jsonl"


@pytest.mark.unit
def test_frozen_stage_a_protocol_is_finite_disjoint_and_digestible() -> None:
    protocol = load_stage_a_protocol(_CONFIG, _SELECTION, _CONFIRMATION)

    assert protocol.protocol_id == "cegwm-stage-a-v1"
    assert len(protocol.candidate_selection) == 8
    assert len(protocol.untouched_confirmation) == 8
    assert len(protocol.protocol_digest) == 64
    assert {unit.source_id for unit in protocol.candidate_selection}.isdisjoint(
        unit.source_id for unit in protocol.untouched_confirmation
    )
    assert len(protocol.config["lf_candidates"]) == 2
    assert isinstance(protocol.config["detection_access"], MappingProxyType)
    assert protocol.config["execution_flow"]["failure_units_remain_in_denominator"] is True
    assert protocol.config["execution_flow"]["replacement_units_allowed"] is False


@pytest.mark.unit
def test_detection_access_is_exact_and_private_state_is_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["detection_access"]["allowed_inputs"].append("prompt")
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_inputs must be exactly"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_stops_if_failures_are_removed_from_fixed_denominator(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["execution_flow"]["failure_units_remain_in_denominator"] = False
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed denominator"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_rejects_overlapping_lf_hf_bands(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["bands"]["lf_radius"] = [0.04, 0.7]
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)


@pytest.mark.unit
def test_protocol_rejects_candidate_outside_frozen_lf_band(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["lf_candidates"][0]["radial_subband"] = [0.01, 0.14]
    modified = tmp_path / "invalid.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the frozen LF band"):
        load_stage_a_protocol(modified, _SELECTION, _CONFIRMATION)
