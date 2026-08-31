from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "geometry-v4"
SCRIPT = EVIDENCE_ROOT / "scripts" / "recompute_geometry_v4_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("geometry_v4_evidence_recompute", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_original_sidecars_and_derived_files_recompute() -> None:
    module = _module()
    verified = module.verify_sidecars()
    assert len(verified) == 22
    module.check_derived()
    assert all(entry["bytes"] > 0 and len(entry["sha256"]) == 64 for entry in verified)


def test_route_ledger_reproduces_the_freeze_decision() -> None:
    ledger = json.loads((EVIDENCE_ROOT / "derived" / "route_ledger.json").read_text(encoding="ascii"))
    freeze = json.loads((EVIDENCE_ROOT / "derived" / "freeze_decision.json").read_text(encoding="ascii"))
    routes = ledger["routes"]

    assert ledger["frozen_exact"] == "12488ad69bd6d2bf8ccc8d0c8d590cfa44bf372b"
    assert freeze["status"] == "DECODER_OUTPUT_BASELINE_METHOD_PARTIAL"
    assert freeze["formal_science_denominator"] == 0
    assert freeze["geometry_role"] == "coordinate_only_never_positive"
    assert freeze["do_not_merge_back"] is True

    g0 = next(route for route in routes if route["stage"] == "G0")
    g1 = next(route for route in routes if route["stage"] == "G1")
    g1r = [route for route in routes if route["stage"] == "G1R_development"]
    cpu = next(route for route in routes if route.get("route") == "balanced_bipolar_prn_microcode")

    assert (g0["final_rgb_observable"], g0["units"], g0["status"]) == (4, 4, "PASS")
    assert (g1["source_observability_passed"], g1["legacy_attacked_gate_passed"]) == (2, 2)
    assert (g1["correct_reliable"], g1["correct_safe_reliable"], g1["correct_unsafe"]) == (19, 0, 19)
    assert (g1["wrong_reliable"], g1["wrong_safe_reliable"], g1["wrong_unsafe"]) == (18, 0, 18)

    assert len(g1r) == 6
    assert [route["source_observability_passed"] for route in g1r] == [2, 2, 3, 2, 1, 0]
    assert all(route["units"] == 20 and route["failures"] == 0 for route in g1r)
    assert all(route["correct_safe_reliable"] == 0 for route in g1r)
    assert [route["correct_rs_top5"] for route in g1r] == [0, 0, 10, 0, 0, 0]

    sparse = g1r[-1]
    assert sparse["selected_fit_support"] == {"min": 3, "median": 4.0, "max": 7}
    assert sparse["truth_probe"]["fit_valid"] == 0
    assert sparse["truth_probe"]["holdout_passed"] == 0
    assert sparse["truth_probe"]["truth_rs_translation_psr_max"] < 2.679

    assert cpu["formal_denominator"] == 0
    assert cpu["status"] == "CPU_METHOD_PARTIAL"
    assert cpu["correct_safe_reliable"] == 0
    assert cpu["correct_rs_top5"] == 5
    assert cpu["identity_translation_psr_ge_8"] == 2
    assert cpu["failures"] == cpu["stops"] == 0


def test_public_package_contains_no_secret_material_or_positive_geometry_semantics() -> None:
    forbidden = (
        "CEG_WM_ROOT_KEY",
        "HF_TOKEN",
        "geometry_can_form_positive\": true",
        "geometry_positive_vote",
        "raw_key_bytes",
    )
    public_paths = list((EVIDENCE_ROOT / "raw").rglob("*"))
    public_paths += list((EVIDENCE_ROOT / "derived").rglob("*"))
    public_paths += [EVIDENCE_ROOT / "README.md", REPO_ROOT / "docs" / "geometry_v4_evidence_card.md"]
    for path in public_paths:
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path
