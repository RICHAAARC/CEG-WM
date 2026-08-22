from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.stage_a_frequency_response.protocol import CONDITIONS, EVIDENCE_CONTRACT, RECORD_ARMS, expected_pairs, load_plan

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.json"
_ROSTER = _ROOT / "configs/stage_a_frequency_response/standalone_lf_hf_frequency_response_v1.jsonl"


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
    assert len(plan.config_digest) == 64


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
def test_plan_rejects_budget_or_fixed_denominator_drift(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["budget"]["actual_callback_dtype_relative_l2_per_method_max"] = 0.011
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="budget"):
        load_plan(bad, _ROSTER)
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["execution"]["fixed_records"] = 319
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="320-record"):
        load_plan(bad, _ROSTER)
