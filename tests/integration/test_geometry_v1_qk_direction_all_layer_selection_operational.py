"""CPU/fake contracts for artifact-only all-layer direction selection."""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_direction_all_layer_selection_operational.py"
SPEC = importlib.util.spec_from_file_location("geometry_direction_all_layer", MODULE)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


def _unit(pair: str, kind: str, control: str, layer: int, *, eligible: bool = False, no_common: bool = False) -> dict:
    transform = pair.rsplit("-", 1)[1]
    matched, shuffled = (1.0, 3.0) if eligible else (4.0, 2.0)
    ranks = [None, None, None] if no_common else ([matched, None, matched + 1] if control == "matched_h" else [shuffled, None, shuffled + 1])
    return {"pair_id": pair, "transform_label": transform, "control_label": control, "descriptor_kind": kind, "layer_path": f"transformer_blocks.{layer}.attn", "reference_grid": [32, 32], "attacked_grid": [32, 32], "input_identity": {"sha256": "a" * 64}, "h_identity": {"sha256": "b" * 64}, "status": "calculated", "failure_reason": None, "candidate_correspondences": [], "true_match_ranks": ranks, "coverage": 1.0, "ambiguity_gaps": [1.0], "fit_residual": 0.1, "recovery_error": 0.2}


def _units(*, eligible_layers: set[int] = set(), no_common: tuple[int, str] | None = None) -> list[dict]:
    return [_unit(pair, kind, control, layer, eligible=layer in eligible_layers, no_common=no_common == (layer, kind)) for layer in range(24) for pair in RUNNER.PAIRS for kind in RUNNER.KINDS for control in RUNNER.CONTROLS]


def _source(root: Path, *, sidecar: str | None = None, leak: str | None = None, unit_value: str | None = None) -> Path:
    root.mkdir(); layers = root / "layers"; layers.mkdir(); units = _units(); shards = []
    for layer in range(24):
        target = layers / f"{layer:02d}.zip"; layer_units = [unit for unit in units if unit["layer_path"] == f"transformer_blocks.{layer}.attn"]
        if layer == 0 and unit_value is not None: layer_units[0]["input_identity"] = {"sha256": unit_value}
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for ordinal, unit in enumerate(layer_units): archive.writestr(f"{ordinal:02d}.json", json.dumps(unit, sort_keys=True, separators=(",", ":")))
        shards.append({"filename": f"layers/{layer:02d}.zip", "layer_path": f"transformer_blocks.{layer}.attn", "unit_count": 32, "bytes": target.stat().st_size})
    receipt = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "plan_digest": RUNNER.SOURCE_PLAN_DIGEST, "d0_status": RUNNER.SOURCE_STATUS, "science_denominator": 0, "declared_unit_count": 768, "calculated_unit_count": 768, "failed_unit_count": 0, "artifact_status": "complete", "operational_status": "complete", "operational_failure_point": None, "execution_identity": {"commit": RUNNER.SOURCE_EXACT}, "layer_shards": shards}
    manifest = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "execution_exact": RUNNER.SOURCE_EXACT, "unit_count": 768, "layer_shards": shards}
    terminal = {"run_id": RUNNER.SOURCE_RUN_ID, "d0_status": RUNNER.SOURCE_STATUS}
    sidecars = {"receipt.json": receipt, "manifest.json": manifest, "terminal.json": terminal}
    if sidecar is not None and leak is not None: sidecars[sidecar]["audit_note"] = leak
    for name, value in sidecars.items(): (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def test_source_validates_24_x_32_fixed_order_and_roster_digest(tmp_path) -> None:
    source = _source(tmp_path / "source"); identity, units = RUNNER._validate_source(source)
    assert identity["roster_digest"] == RUNNER.SOURCE_ROSTER_DIGEST and len(units) == 768
    assert [(unit["pair_id"], unit["descriptor_kind"], unit["control_label"]) for unit in units[:4]] == [("reference_a-identity", "q", "matched_h"), ("reference_a-identity", "q", "shuffled_h"), ("reference_a-identity", "k", "matched_h"), ("reference_a-identity", "k", "shuffled_h")]


def test_selection_uses_same_list_indexes_qk_and_k2_ties_without_strata_or_transform_deletion() -> None:
    units = _units(eligible_layers={2, 5, 9})
    status, selected, layers, audit = RUNNER._selection(units)
    assert status == "DIRECTION_TWO_CANDIDATES_FROZEN" and selected == ["transformer_blocks.2.attn", "transformer_blocks.5.attn"]
    assert layers[2]["q_stat"] == -2.0 and layers[2]["k_stat"] == -2.0
    assert all(len(layers[index]["q_audit"]["pair_audit"]) == 8 for index in range(24))
    assert {item["transform_label"] for item in audit["per_transform"]} == set(RUNNER.TRANSFORMS)


def test_no_common_index_or_fewer_than_two_eligible_is_unresolved() -> None:
    status, selected, layers, _audit = RUNNER._selection(_units(eligible_layers={2}, no_common=(2, "q")))
    assert status == "DIRECTION_ALL_LAYER_UNRESOLVED" and selected == [] and not layers[2]["eligible"]
    status, selected, _layers, _audit = RUNNER._selection(_units(eligible_layers={2}))
    assert status == "DIRECTION_ALL_LAYER_UNRESOLVED" and selected == []


def test_route_audit_is_nonblocking_and_marks_all_layer_d4_crop_instability() -> None:
    status, selected, _layers, audit = RUNNER._selection(_units())
    assert status == "DIRECTION_ALL_LAYER_UNRESOLVED" and selected == []
    assert audit["route_level_transform_instability"]
    assert all(item["all_layer_nonnegative"] for item in audit["per_transform"])


@pytest.mark.parametrize("sidecar", ("receipt.json", "manifest.json", "terminal.json"))
@pytest.mark.parametrize("leak", ("audit note /mnt/private/source", "Bearer opaque-value", "authentication token material"))
def test_root_sidecar_leaks_fail_before_selection_or_packaging(monkeypatch, tmp_path, sidecar, leak) -> None:
    source = _source(tmp_path / "source", sidecar=sidecar, leak=leak)
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    monkeypatch.setattr(RUNNER, "_selection", lambda _units: pytest.fail("selection must not run"))
    monkeypatch.setattr(RUNNER, "_package", lambda *_args: pytest.fail("package must not run"))
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", "a" * 40, "--source-root", str(source), "--output-root", str(tmp_path / "out"), "--control-fd", str(write)])
        line = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    control = json.loads(line[len(RUNNER.FAILURE_PREFIX):])
    assert rc == 1 and control["failure_point"] == "source_validation" and control["error_class"] == "validation_error"


def test_unit_value_leak_fails_before_selection_or_packaging(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path / "source", unit_value="embedded /mnt/private/source")
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    monkeypatch.setattr(RUNNER, "_selection", lambda _units: pytest.fail("selection must not run"))
    monkeypatch.setattr(RUNNER, "_package", lambda *_args: pytest.fail("package must not run"))
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", "a" * 40, "--source-root", str(source), "--output-root", str(tmp_path / "out"), "--control-fd", str(write)])
        line = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    control = json.loads(line[len(RUNNER.FAILURE_PREFIX):])
    assert rc == 1 and control["failure_point"] == "source_validation" and control["error_class"] == "validation_error"


def test_package_is_create_only_and_does_not_copy_source(tmp_path, monkeypatch) -> None:
    source = _source(tmp_path / "source"); monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    summary = RUNNER.run_direction_selection(expected_exact="a" * 40, repo_root=tmp_path, source_root=source)
    output = tmp_path / "out"; RUNNER._package(output, summary)
    assert {item.name for item in output.iterdir()} == {"receipt.json", "manifest.json", "terminal.json"}
    with pytest.raises(FileExistsError): RUNNER._package(output, summary)
