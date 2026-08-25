from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol import content_chain_v9 as v9

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_v9_manifests_are_exact_ordered_unique_and_disjoint() -> None:
    contract = v9.load_content_v9_phase1_contract(_ROOT)
    assert len(contract.calibration) == 32
    assert len(contract.evaluation) == 8
    assert tuple(unit.seed for unit in contract.calibration) == tuple(range(2026091000, 2026091032))
    assert tuple(unit.seed for unit in contract.evaluation) == tuple(range(2026092000, 2026092008))
    assert hashlib.sha256(
        (_ROOT / "configs/content_chain/content_v9_calibration_v1.jsonl").read_bytes()
    ).hexdigest() == v9.CONTENT_V9_CALIBRATION_MANIFEST_SHA256
    assert hashlib.sha256(
        (_ROOT / "configs/content_chain/content_v9_clean_evaluation_v1.jsonl").read_bytes()
    ).hexdigest() == v9.CONTENT_V9_EVALUATION_MANIFEST_SHA256
    for field in ("unit_id", "source_id", "prompt", "seed"):
        assert not (
            {getattr(unit, field) for unit in contract.calibration}
            & {getattr(unit, field) for unit in contract.evaluation}
        )
    historical: set[tuple[str, int]] = {
        ("A book conservator examining an illuminated manuscript under neutral studio light", 1415149)
    }
    config_root = _ROOT / "configs/content_chain"
    for name in (
        "content_adaptive_dual_branch_v2_clean.jsonl",
        "content_v6_iss_development_v1.jsonl",
        "content_v6_iss_clean.jsonl",
    ):
        for line in (config_root / name).read_bytes().splitlines():
            value = json.loads(line)
            historical.add((value["prompt"], value["seed"]))
    fit = json.loads((config_root / "content_v4_clean_null_whitening_fit_v1.json").read_bytes())
    historical.update((entry["prompt"], entry["generation_seed"]) for entry in fit["entries"])
    assert not historical.intersection(
        (unit.prompt, unit.seed) for unit in (*contract.calibration, *contract.evaluation)
    )


@pytest.mark.unit
def test_v9_phase1_identity_has_calibration_digest_but_no_fabricated_final_asset() -> None:
    contract = v9.load_content_v9_phase1_contract(_ROOT)
    assert contract.protocol_digest == v9.CONTENT_V9_CALIBRATION_PROTOCOL_DIGEST
    assert contract.config["protocol_id"] == v9.CONTENT_V9_CALIBRATION_PROTOCOL_ID
    assert contract.config["future_evaluation"]["protocol_id"] == (
        v9.CONTENT_V9_FUTURE_EVALUATION_PROTOCOL_ID
    )
    assert contract.config["scientific_status"] == (
        "calibration_not_run_no_final_v9_evaluation_identity"
    )
    assert contract.config["joint_operator"]["lf_hf_gates"] == (
        "diagnostic_only_no_hard_veto"
    )
    assert "asset_sha256" not in contract.config["asset_contract"]
    run_id = v9.deterministic_calibration_run_id(contract.protocol_digest, "a" * 64)
    assert run_id == f"content-v9-calibration-{contract.protocol_digest[:12]}-{'a' * 12}"


@pytest.mark.unit
def test_v9_manifest_drift_fails_closed(tmp_path: Path) -> None:
    source = _ROOT / "configs/content_chain"
    target = tmp_path / "configs/content_chain"
    target.mkdir(parents=True)
    for name in (
        "content_v9_calibration_v1.jsonl",
        "content_v9_clean_evaluation_v1.jsonl",
        "content_v9_calibrated_weighted_joint_phase1_v1.json",
    ):
        (target / name).write_bytes((source / name).read_bytes())
    raw = (target / "content_v9_calibration_v1.jsonl").read_bytes()
    (target / "content_v9_calibration_v1.jsonl").write_bytes(raw.replace(b"bronze", b"silver", 1))
    with pytest.raises(ValueError, match="manifest bytes differ"):
        v9.load_content_v9_phase1_contract(tmp_path)
