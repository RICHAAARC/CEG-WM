from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.stage_a_frequency_response.protocol import CONDITIONS, EVIDENCE_CONTRACT, RECORD_ARMS, expected_pairs, load_plan

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json"
_ROSTER = _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl"


def _payload() -> dict[str, Any]:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))


def _assert_rejected(tmp_path: Path, payload: dict[str, Any], match: str) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_plan(bad, _ROSTER)


@pytest.mark.unit
def test_frequency_response_plan_is_finite_disjoint_and_320_records() -> None:
    plan = load_plan(_CONFIG, _ROSTER)
    assert plan.protocol_id == "standalone-lf-hf-frequency-response-v1"
    assert EVIDENCE_CONTRACT == "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
    assert len(plan.units) == 8 and len(expected_pairs()) == 40
    assert CONDITIONS == (
        "identity", "jpeg_q90", "jpeg_q75", "jpeg_q50", "gaussian_blur_sigma_0_5",
        "gaussian_blur_sigma_1", "gaussian_blur_sigma_2", "gaussian_noise_std_0_005",
        "gaussian_noise_std_0_01", "gaussian_noise_std_0_02",
    )
    assert RECORD_ARMS[0].startswith("hf_") and RECORD_ARMS[2].startswith("lf_")
    assert len({unit.unit_id for unit in plan.units}) == len({unit.source_id for unit in plan.units}) == 8
    assert len(plan.protocol_digest) == len(plan.roster_digest) == 64
    assert plan.config_digest == plan.protocol_digest
    assert plan.protocol_digest != plan.roster_digest
    assert set(plan.method_identities) == {"hf", "lf"}


@pytest.mark.unit
def test_frequency_response_roster_is_globally_fresh_against_existing_stage_a_manifests() -> None:
    plan = load_plan(_CONFIG, _ROSTER)
    prior: dict[str, set[object]] = {"unit_id": set(), "source_id": set(), "prompt": set(), "seed": set()}
    for manifest in (_ROOT / "configs/stage_a").glob("*.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            for name in prior:
                prior[name].add(entry[name])
    for name in prior:
        assert {getattr(unit, name if name != "source_id" else "source_id") for unit in plan.units}.isdisjoint(prior[name])


@pytest.mark.unit
def test_plan_rejects_protocol_and_evidence_identity_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["protocol_id"] = "standalone-lf-hf-frequency-response-v2"
    _assert_rejected(tmp_path, payload, "protocol identity")
    payload = _payload()
    payload["evidence_contract"] = "OTHER_EVIDENCE"
    _assert_rejected(tmp_path, payload, "evidence contract")
    payload = _payload()
    payload["unrecognized"] = True
    _assert_rejected(tmp_path, payload, "config fields")


@pytest.mark.unit
def test_plan_rejects_detection_access_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["detection_access"]["forbidden_inputs"].remove("prompt")
    _assert_rejected(tmp_path, payload, "detection access")


@pytest.mark.unit
def test_plan_rejects_generation_runtime_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["generation_runtime"]["inference_steps"] = 21
    _assert_rejected(tmp_path, payload, "generation runtime")


@pytest.mark.unit
def test_plan_rejects_method_identity_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["methods"]["lf"]["detector_statistic_id"] = "other"
    _assert_rejected(tmp_path, payload, "detector identity")


@pytest.mark.unit
def test_plan_rejects_budget_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["budget"]["independent_full_budget"] = 1
    _assert_rejected(tmp_path, payload, "budget")


@pytest.mark.unit
def test_plan_rejects_wrong_key_contract_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["keying"]["wrong_key_derivation_domain"] = "other"
    _assert_rejected(tmp_path, payload, "wrong-key")


@pytest.mark.unit
def test_plan_rejects_condition_or_arm_order_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["conditions"] = list(reversed(payload["conditions"]))
    _assert_rejected(tmp_path, payload, "condition order")
    payload = _payload()
    payload["record_arms_in_exact_condition_order"] = list(
        reversed(payload["record_arms_in_exact_condition_order"])
    )
    _assert_rejected(tmp_path, payload, "arm order")


@pytest.mark.unit
def test_plan_rejects_transform_contract_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["transform_contract"]["jpeg"]["encode"]["subsampling"] = 0
    _assert_rejected(tmp_path, payload, "transform contract")


@pytest.mark.unit
def test_plan_rejects_execution_failure_contract_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["execution"]["fixed_records"] = 319
    _assert_rejected(tmp_path, payload, "denominator")


@pytest.mark.unit
def test_execution_contract_fixes_atomic_resume_and_artifact_rules() -> None:
    execution = _payload()["execution"]
    assert execution == {
        "fixed_units": 8,
        "records_per_unit": 40,
        "fixed_records": 320,
        "failures_remain_in_denominator": True,
        "replacement_units_allowed": False,
        "complete_rc": 0,
        "automatic_fresh_or_resume": True,
        "checkpoint_interval_hours": 2.0,
        "checkpoint_only_after_new_complete_unit": True,
        "short_run_final_only": True,
        "checkpoint_schema": "standalone-lf-hf-frequency-response-checkpoint-v1",
        "committed_unit_transactions_immutable": True,
        "active_state_location": "local_only",
        "artifact_sink_pairs": [
            "complete_checkpoint_zip_and_sha256",
            "complete_final_zip_and_sha256",
            "complete_failure_zip_and_sha256",
        ],
        "artifact_publication": "create_only",
        "terminal_pair_prevents_rerun": True,
    }


@pytest.mark.unit
def test_protocol_and_roster_digests_are_independent(tmp_path: Path) -> None:
    original = load_plan(_CONFIG, _ROSTER)
    entries = [json.loads(line) for line in _ROSTER.read_text(encoding="utf-8").splitlines()]
    entries[0]["prompt"] += " at dawn"
    changed = tmp_path / "changed.jsonl"
    changed.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    revised = load_plan(_CONFIG, changed)
    assert revised.protocol_digest == original.protocol_digest
    assert revised.roster_digest != original.roster_digest


@pytest.mark.unit
def test_plan_rejects_limitations_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["limitations"] = payload["limitations"][:-1]
    _assert_rejected(tmp_path, payload, "limitation")
