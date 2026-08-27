"""CPU/fake contracts for the D1 fixed-layer confirmation transport."""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_d1_independent_confirmation_operational.py"
SPEC = importlib.util.spec_from_file_location("geometry_d1_operational", MODULE)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


def _source(root: Path, *, leaked: dict | None = None) -> Path:
    root.mkdir()
    receipt = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "d01_status": RUNNER.SOURCE_STATUS,
               "science_denominator": 0, "execution_identity": {"commit": RUNNER.SOURCE_EXACT}, "selected_layer_paths": RUNNER.SOURCE_SELECTED_PATHS}
    manifest = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "execution_exact": RUNNER.SOURCE_EXACT, "d01_status": RUNNER.SOURCE_STATUS}
    terminal = {"run_id": RUNNER.SOURCE_RUN_ID, "d01_status": RUNNER.SOURCE_STATUS, "science_denominator": 0}
    if leaked: receipt.update(leaked)
    for name, value in (("receipt.json", receipt), ("manifest.json", manifest), ("terminal.json", terminal)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def _record(pair: dict, path: str, kind: str, control: str, *, matched: float = 1.0, shuffled: float = 3.0, common: bool = True) -> dict:
    ranks = [matched, None, matched + 1] if control == "matched_h" else [shuffled, None, shuffled + 1]
    if not common: ranks = [None, None, None]
    return {"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path,
            "reference_grid": [32, 32], "attacked_grid": [32, 32], "input_identity": {"sha256": "a" * 64}, "h_identity": {"sha256": "b" * 64},
            "status": "calculated", "failure_reason": None, "candidate_correspondences": [], "true_match_ranks": ranks, "coverage": 1.0,
            "ambiguity_gaps": [1.0], "fit_residual": 0.1, "recovery_error": 0.2}


def _run(monkeypatch, tmp_path, *, common: bool = True, reversed_direction: bool = False):
    source = _source(tmp_path / "source"); calls = []
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    monkeypatch.setattr(RUNNER, "_spec", lambda pipeline: object())
    def observer(_image, *, pipeline, spec): calls.append(1); return object()
    def unit(pair, _reference, _attacked, path, kind, control):
        return _record(pair, path, kind, control, matched=4.0 if reversed_direction else 1.0, shuffled=2.0 if reversed_direction else 3.0, common=common)
    monkeypatch.setattr(RUNNER, "_unit", unit)
    summary, units = RUNNER.run_d1(expected_exact="a" * 40, repo_root=tmp_path, source_root=source, hf_token="", loader=lambda *_a, **_k: object(), observer=observer)
    return summary, units, calls


def test_d1_fixed_roster_is_96_and_confirms_only_all_six_negative_statistics(monkeypatch, tmp_path) -> None:
    summary, units, calls = _run(monkeypatch, tmp_path)
    assert len(calls) == 10 and len(units) == 96 and summary["d1_status"] == "D1_CANDIDATES_CONFIRMED"
    assert summary["fixed_layer_paths"] == list(RUNNER.ATTENTION_LAYER_PATHS)
    assert all(item["strictly_negative"] for item in summary["direction_statistics"])
    assert [(u["pair_id"], u["layer_path"], u["descriptor_kind"], u["control_label"]) for u in units[:4]] == [("confirmation_a-identity", "transformer_blocks.6.attn", "q", "matched_h"), ("confirmation_a-identity", "transformer_blocks.6.attn", "q", "shuffled_h"), ("confirmation_a-identity", "transformer_blocks.6.attn", "k", "matched_h"), ("confirmation_a-identity", "transformer_blocks.6.attn", "k", "shuffled_h")]


def test_d1_common_finite_pairing_and_direction_fail_closed_without_dropping_roster(monkeypatch, tmp_path) -> None:
    summary, units, _calls = _run(monkeypatch, tmp_path, common=False)
    assert summary["d1_status"] == "D1_UNRESOLVED" and len(units) == 96
    summary, units, _calls = _run(monkeypatch, tmp_path, reversed_direction=True)
    assert summary["d1_status"] == "D1_UNRESOLVED" and len(units) == 96
    assert any(not item["strictly_negative"] for item in summary["direction_statistics"])


def test_d1_source_identity_and_leaks_fail_before_model_or_selection(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path / "source", leaked={"audit_note": "raw token material"}); calls = []
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    with pytest.raises(ValueError, match="forbidden"):
        RUNNER.run_d1(expected_exact="a" * 40, repo_root=tmp_path, source_root=source, hf_token="", loader=lambda *_a, **_k: calls.append("loader"), observer=lambda *_a, **_k: calls.append("observer"))
    assert calls == []
    source = _source(tmp_path / "wrong")
    receipt = json.loads((source / "receipt.json").read_text()); receipt["selected_layer_paths"] = []
    (source / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="selected"):
        RUNNER._validate_d01_source(source)


def test_d1_three_fixed_shards_are_create_only_and_public(monkeypatch, tmp_path) -> None:
    summary, units, _calls = _run(monkeypatch, tmp_path)
    output = tmp_path / "output"; package = RUNNER._package(output, summary, units)
    assert package["artifact_status"] == "complete" and [path.name for path in sorted((output / "layers").glob("*.zip"))] == ["00.zip", "01.zip", "02.zip"]
    for target in sorted((output / "layers").glob("*.zip")):
        with zipfile.ZipFile(target) as archive: assert archive.namelist() == [f"{index:02d}.json" for index in range(32)]
    with pytest.raises(FileExistsError): RUNNER._package(output, summary, units)


def test_d1_main_preserves_bounded_source_validation_failure(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path / "source", leaked={"audit_note": "/mnt/private/source"})
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", "a" * 40, "--source-root", str(source), "--output-root", str(tmp_path / "output"), "--control-fd", str(write)])
        line = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    control = json.loads(line[len(RUNNER.FAILURE_PREFIX):])
    assert rc == 1 and control["failure_point"] == "source_validation" and control["error_class"] == "validation_error"
