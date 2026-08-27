"""CPU-only contracts for D0.1's immutable D0 artifact selection path."""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_d01_artifact_selection_operational.py"
SPEC = importlib.util.spec_from_file_location("geometry_d01_operational", MODULE)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


def _unit(pair: str, kind: str, control: str, layer: int, *, score: float, no_rank: bool = False) -> dict:
    transform = pair.rsplit("-", 1)[1]
    out_of_view = control == "matched_h" and transform in ("similarity", "crop_rescale")
    ranks = [None] if no_rank and control == "matched_h" else ([None, score + 1.0] if out_of_view else [score + 1.0])
    return {"pair_id": pair, "transform_label": transform, "control_label": control, "descriptor_kind": kind,
            "layer_path": f"transformer_blocks.{layer}.attn", "reference_grid": [32, 32], "attacked_grid": [32, 32],
            "input_identity": None, "h_identity": None, "status": "calculated", "failure_reason": None,
            "candidate_correspondences": [], "true_match_ranks": ranks,
            "coverage": 1.0, "ambiguity_gaps": [score + 2.0], "fit_residual": score + .25, "recovery_error": score}


def _source(root: Path, *, no_rank_layers: set[int] = set()) -> Path:
    root.mkdir(); layers = root / "layers"; layers.mkdir(); shards = []
    for layer in range(24):
        target = layers / f"{layer:02d}.zip"; raw = []
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            ordinal = 0
            for pair in RUNNER.PAIRS:
                for kind in RUNNER.KINDS:
                    for control in RUNNER.CONTROLS:
                        data = _unit(pair, kind, control, layer, score=float(layer), no_rank=layer in no_rank_layers and control == "matched_h")
                        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(); raw.append(encoded)
                        archive.writestr(f"{ordinal:02d}.json", encoded); ordinal += 1
        shards.append({"layer_path": f"transformer_blocks.{layer}.attn", "filename": f"layers/{layer:02d}.zip", "unit_count": 32, "bytes": target.stat().st_size})
        assert sum(map(len, raw)) <= RUNNER.MAX_LAYER_UNIT_BYTES
    receipt = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "plan_digest": RUNNER.SOURCE_PLAN_DIGEST,
               "d0_status": "D0_UNRESOLVED", "science_denominator": 0, "declared_unit_count": 768,
               "calculated_unit_count": 768, "failed_unit_count": 0, "artifact_status": "complete", "operational_status": "complete",
               "operational_failure_point": None, "execution_identity": {"commit": RUNNER.SOURCE_EXACT}, "layer_shards": shards}
    manifest = {"run_id": RUNNER.SOURCE_RUN_ID, "protocol": RUNNER.SOURCE_PROTOCOL, "execution_exact": RUNNER.SOURCE_EXACT,
                "layer_shards": shards, "unit_count": 768}
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "terminal.json").write_text(json.dumps({"run_id": RUNNER.SOURCE_RUN_ID, "d0_status": "D0_UNRESOLVED"}), encoding="utf-8")
    return root


def _run(monkeypatch, tmp_path, source: Path) -> dict:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    return RUNNER.run_d01(expected_exact=RUNNER.SOURCE_EXACT, repo_root=tmp_path, source_root=source)


def test_d01_validates_exact_source_roster_and_selects_three_strata(monkeypatch, tmp_path) -> None:
    summary = _run(monkeypatch, tmp_path, _source(tmp_path / "source"))
    assert summary["d01_status"] == "D01_CANDIDATES_FROZEN"
    assert summary["selected_layer_paths"] == ["transformer_blocks.0.attn", "transformer_blocks.8.attn", "transformer_blocks.16.attn"]
    assert summary["audited_unit_count"] == 768 and summary["science_denominator"] == 0
    assert all(item["null_rank_count"] == 8 and item["finite_rank_count"] == 16 for item in summary["layer_audit"])
    assert all(item["shuffled_record_count"] == 16 and item["shuffled_calculated_count"] == 16 for item in summary["layer_audit"])


def test_d01_none_is_only_out_of_view_missingness_and_missing_all_ranks_is_unresolved(monkeypatch, tmp_path) -> None:
    summary = _run(monkeypatch, tmp_path, _source(tmp_path / "source", no_rank_layers=set(range(8))))
    assert summary["d01_status"] == "D01_UNRESOLVED" and summary["selected_layer_paths"] == []
    assert all(not item["eligible"] for item in summary["layer_audit"][:8])


def test_d01_rejects_source_identity_bounds_and_public_leaks(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path / "source"); receipt = json.loads((source / "receipt.json").read_text())
    receipt["plan_digest"] = "0" * 64; (source / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    with pytest.raises(ValueError, match="receipt"):
        RUNNER.run_d01(expected_exact=RUNNER.SOURCE_EXACT, repo_root=tmp_path, source_root=source)
    source = _source(tmp_path / "leak")
    with zipfile.ZipFile(source / "layers" / "00.zip", "a") as archive:
        record = json.loads(archive.read("00.json")); record["raw_qk"] = "forbidden"; archive.writestr("00.json", json.dumps(record))
    with pytest.raises(ValueError, match="roster|field|zip"):
        RUNNER.run_d01(expected_exact=RUNNER.SOURCE_EXACT, repo_root=tmp_path, source_root=source)


def test_d01_create_only_package_and_bounded_failure_control(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path / "source"); summary = _run(monkeypatch, tmp_path, source)
    output = tmp_path / "output"; package = RUNNER._package(output, summary)
    assert package["artifact_status"] == "complete" and {item.name for item in output.iterdir()} == {"receipt.json", "manifest.json", "terminal.json"}
    with pytest.raises(FileExistsError): RUNNER._package(output, summary)
    monkeypatch.setattr(RUNNER, "run_d01", lambda **_kwargs: (_ for _ in ()).throw(ValueError("secret /content/path")))
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", RUNNER.SOURCE_EXACT,
                           "--source-root", str(source), "--output-root", str(tmp_path / "unused"), "--control-fd", str(write)])
        line = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    assert rc == 1 and line.startswith(RUNNER.FAILURE_PREFIX.encode()) and b"secret" not in line and b"/content" not in line
