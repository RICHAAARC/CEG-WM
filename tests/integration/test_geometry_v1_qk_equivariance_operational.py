"""CPU/fake coverage for the sharded E0 operational experiment runner."""
from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch
from PIL import Image

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_equivariance_operational.py"
SPEC = importlib.util.spec_from_file_location("qk_e0_operational", MODULE)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _indices() -> torch.Tensor:
    rows = torch.linspace(0, 63, 8, dtype=torch.float64).round().to(torch.int64)
    return (rows[:, None] * 64 + rows[None, :]).reshape(-1)


def _image(path: Path) -> str:
    image = Image.new("RGB", (8, 8), (17, 31, 47)); image.save(path)
    return sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def _plan(tmp_path: Path) -> dict:
    reference = _image(tmp_path / "reference.png"); attacked = _image(tmp_path / "attacked.png")
    pairs = []
    for ref in ("r0", "r1"):
        for label in RUNNER._TRANSFORMS:
            pairs.append({"reference_id": ref, "pair_id": f"{ref}-{label}", "transform_label": label,
                          "reference_path": str(tmp_path / "reference.png"), "reference_sha256": reference,
                          "attacked_path": str(tmp_path / "attacked.png"), "attacked_sha256": attacked,
                          "reference_source_grid": [64, 64], "attacked_source_grid": [64, 64],
                          "matched_h": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          "shuffled_h": [[1., 0., 100.], [0., 1., 0.], [0., 0., 1.]]})
    return {"schema": RUNNER.PLAN_SCHEMA, "attention_layer_paths": ["blocks.0.attn", "blocks.1.attn"], "pairs": pairs}


class _Pipeline:
    _commit_hash = "a" * 40
    def __init__(self): self.transformer = torch.nn.Linear(1, 1)
    def to(self, _device): return self
    def encode_prompt(self, **kwargs):
        assert kwargs == {"prompt": "", "prompt_2": "", "prompt_3": "", "do_classifier_free_guidance": False}
        return torch.zeros((1, 2)), None, torch.zeros((1, 2)), None


def _observation(paths: tuple[str, str]):
    values = torch.eye(64, dtype=torch.float32)
    layers = tuple(SimpleNamespace(layer_path=path, query=values, key=values, source_grid=(64, 64), sample_indices=_indices()) for path in paths)
    return SimpleNamespace(layers=layers)


def test_fixed_plan_observes_images_once_and_keeps_64_qk_layer_control_units(tmp_path, monkeypatch) -> None:
    plan, calls = _plan(tmp_path), []
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    def loader(*args, **kwargs):
        assert args == (RUNNER.MODEL_ID,) and "token" in kwargs
        return _Pipeline()
    def observer(image, *, pipeline, spec):
        calls.append((image.size, spec.attention_layer_paths)); return _observation(spec.attention_layer_paths)
    summary, units = RUNNER.run_qk_equivariance_operational(plan, hf_token="hf-secret", expected_exact="b" * 40, repo_root=tmp_path, loader=loader, observer=observer)
    assert len(calls) == 16 and all(item[1] == tuple(plan["attention_layer_paths"]) for item in calls)
    assert len(units) == 64 and summary["declared_unit_count"] == 64
    assert summary["method_status"] == summary["scientific_status"] == "not_adjudicated" and summary["science_denominator"] == 0
    assert [unit["descriptor_kind"] for unit in units[:4]] == ["q", "q", "k", "k"]
    matched, shuffled = units[0], units[1]
    assert matched["candidate_correspondences"] == shuffled["candidate_correspondences"]
    assert set(matched) == RUNNER.HARNESS.public_record_fields()


def test_invalid_plan_rejects_before_loader_or_images(tmp_path) -> None:
    plan = _plan(tmp_path); plan["pairs"] = plan["pairs"][:-1]
    with pytest.raises(ValueError, match="invalid_pair_count"):
        RUNNER.run_qk_equivariance_operational(plan, hf_token="x", expected_exact="c" * 40, repo_root=tmp_path, loader=lambda *_a, **_k: pytest.fail("loader"))


def test_resource_failure_retains_exactly_64_structured_unit_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    summary, units = RUNNER.run_qk_equivariance_operational(_plan(tmp_path), hf_token="x", expected_exact="c" * 40, repo_root=tmp_path, loader=lambda **_k: (_ for _ in ()).throw(RuntimeError("no resource")))
    assert summary["operational_status"] == "failure" and summary["resource_status"] == "unavailable"
    assert len(units) == 64 and {unit["failure_reason"] for unit in units} == {"runtime_not_observed"}


def test_sharded_package_is_create_only_and_does_not_embed_full_units(tmp_path) -> None:
    summary = {"run_id": "geometry-v1-qk-e0-aaaaaaaaaaaa", "operational_status": "complete", "artifact_status": "unavailable", "method_status": "not_adjudicated", "scientific_status": "not_adjudicated", "science_denominator": 0}
    unit = {"pair_id": "ordinary.scientific-words", "transform_label": "identity", "control_label": "matched_h", "descriptor_kind": "q", "layer_path": "blocks.0.attn", "reference_grid": [2, 2], "attacked_grid": [2, 2], "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": "x", "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}
    root = tmp_path / "out"; package = RUNNER._package(root, summary, [unit] * 64, expected_exact="a" * 40)
    assert len(list((root / "units").glob("*.json"))) == 64 and (root / package["archive_filename"]).is_file()
    receipt = json.loads((root / "receipt.json").read_text())
    assert "unit_records" not in receipt and len(receipt["unit_manifest"]) == 64
    with pytest.raises(FileExistsError): RUNNER._package(root, summary, [unit], expected_exact="a" * 40)


def test_unit_and_total_bounds_do_not_truncate(tmp_path) -> None:
    unit = {key: None for key in RUNNER.HARNESS.public_record_fields()}; unit.update({"pair_id": "p", "transform_label": "identity", "control_label": "matched_h", "descriptor_kind": "q", "layer_path": "l", "status": "failed", "failure_reason": "x", "candidate_correspondences": [], "true_match_ranks": [], "ambiguity_gaps": []})
    unit["input_identity"] = "x" * RUNNER.MAX_UNIT_BYTES
    with pytest.raises(ValueError, match="bounded_json_exceeded"):
        RUNNER._package(tmp_path / "too-big", {"run_id": "r", "operational_status": "failure", "artifact_status": "unavailable"}, [unit], expected_exact="a" * 40)


def test_control_is_bounded_one_line_and_public_identifiers_are_not_semantically_scanned() -> None:
    read, write = __import__("os").pipe()
    try:
        RUNNER._emit(write, RUNNER.SUCCESS_PREFIX, {"status": "success", "run_id": "ordinary.scientific-words", "artifact_status": "complete"})
        line = __import__("os").read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        __import__("os").close(read); __import__("os").close(write)
    assert line.endswith(b"\n") and len(line) <= RUNNER.MAX_CONTROL_BYTES and b"scientific-words" in line


def test_main_marks_invalid_plan_as_plan_failure_not_artifact_packaging(tmp_path, monkeypatch) -> None:
    plan_path = tmp_path / "bad.json"; plan_path.write_text("{}")
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--plan", str(plan_path), "--repo-root", str(tmp_path), "--expected-exact", "a" * 40,
                           "--output-root", str(tmp_path / "out"), "--control-fd", str(write)])
        payload = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    assert rc == 1
    assert b'"failure_point":"plan"' in payload


def test_plan_file_bound_is_checked_before_read_or_execution(tmp_path, monkeypatch) -> None:
    plan_path = tmp_path / "large.json"; plan_path.write_bytes(b"x" * (RUNNER.MAX_PLAN_BYTES + 1))
    monkeypatch.setattr(Path, "read_bytes", lambda *_a, **_k: pytest.fail("plan must not be read"))
    with pytest.raises(ValueError, match="plan_bytes_exceeded"):
        RUNNER._read_plan(plan_path)


def test_plan_json_contract_allows_current_bound_and_rejects_max_plus_one_before_loader(tmp_path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    for pair in plan["pairs"]:
        pair["reference_path"] = "r" * RUNNER.MAX_PRIVATE_PATH_BYTES
        pair["attacked_path"] = "a" * RUNNER.MAX_PRIVATE_PATH_BYTES
    encoded = RUNNER._plan_json(plan)
    assert 65536 < len(encoded) <= RUNNER.MAX_PLAN_BYTES
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    summary, units = RUNNER.run_qk_equivariance_operational(plan, hf_token="x", expected_exact="c" * 40,
                                                             repo_root=tmp_path, loader=lambda *_a, **_k: _Pipeline(),
                                                             observer=lambda *_a, **_k: _observation(("blocks.0.attn", "blocks.1.attn")))
    assert summary["plan_digest"] == sha256(encoded).hexdigest() and len(units) == 64
    monkeypatch.setattr(RUNNER, "MAX_PLAN_BYTES", len(encoded) - 1)
    with pytest.raises(ValueError, match="bounded_json_exceeded"):
        RUNNER.run_qk_equivariance_operational(plan, hf_token="x", expected_exact="c" * 40, repo_root=tmp_path,
                                                loader=lambda *_a, **_k: pytest.fail("loader must not run"))


@pytest.mark.parametrize("mutate,reason", [
    (lambda p: p.update(schema="wrong"), "invalid_plan_schema"),
    (lambda p: p.update(attention_layer_paths=["one", "one"]), "invalid_attention_layer_paths"),
    (lambda p: p.update(pairs=p["pairs"][:7]), "invalid_pair_count"),
    (lambda p: p["pairs"][0].update(transform_label="other"), "invalid_pair_manifest"),
    (lambda p: p["pairs"][0].update(reference_source_grid=[0, 2]), "invalid_expected_source_grid"),
])
def test_plan_structural_categories_fail_before_model_work(tmp_path, mutate, reason) -> None:
    plan = _plan(tmp_path); mutate(plan)
    with pytest.raises(ValueError, match=reason):
        RUNNER.run_qk_equivariance_operational(plan, hf_token="x", expected_exact="c" * 40, repo_root=tmp_path,
                                                loader=lambda *_a, **_k: pytest.fail("loader"))


def test_pair_local_failure_retains_only_its_eight_units_and_keeps_fixed_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    calls = []
    def observer(_image, *, pipeline, spec):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("pair-local")
        return _observation(spec.attention_layer_paths)
    summary, units = RUNNER.run_qk_equivariance_operational(_plan(tmp_path), hf_token="x", expected_exact="c" * 40,
                                                             repo_root=tmp_path, loader=lambda *_a, **_k: _Pipeline(), observer=observer)
    # The failed pair is not relaunched; all seven later pairs still make their
    # one reference/attacked observation attempt.
    assert len(calls) == 15 and len(units) == 64 and summary["operational_failure_point"] == "image_observation"
    assert {item["failure_reason"] for item in units[8:16]} == {"image_observation_failed"}
    assert all(item["pair_id"] == "r0-d4" for item in units[8:16])
    assert all(item["pair_id"] == "r1-crop_rescale" for item in units[-8:])


def test_single_layer_source_grid_mismatch_is_retained_without_other_layer_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    def observer(_image, *, pipeline, spec):
        item = _observation(spec.attention_layer_paths)
        layers = list(item.layers)
        layers[0] = SimpleNamespace(**{**vars(layers[0]), "source_grid": (63, 64)})
        return SimpleNamespace(layers=tuple(layers))
    _summary, units = RUNNER.run_qk_equivariance_operational(_plan(tmp_path), hf_token="x", expected_exact="c" * 40,
                                                               repo_root=tmp_path, loader=lambda *_a, **_k: _Pipeline(), observer=observer)
    assert {u["failure_reason"] for u in units if u["layer_path"] == "blocks.0.attn"} == {"source_grid_mismatch"}
    assert {u["status"] for u in units if u["layer_path"] == "blocks.1.attn"} == {"calculated"}


def test_artifact_manifest_zip_sidecar_and_hashes_are_exact(tmp_path) -> None:
    summary = {"run_id": "geometry-v1-qk-e0-aaaaaaaaaaaa", "operational_status": "failure", "artifact_status": "unavailable",
               "method_status": "not_adjudicated", "scientific_status": "not_adjudicated", "science_denominator": 0}
    unit = {"pair_id": "ordinary.scientific-words", "transform_label": "identity", "control_label": "matched_h", "descriptor_kind": "q", "layer_path": "blocks.0.attn", "reference_grid": [2, 2], "attacked_grid": [2, 2], "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": "x", "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}
    root = tmp_path / "out"; package = RUNNER._package(root, summary, [unit] * 64, expected_exact="a" * 40)
    manifest = json.loads((root / "manifest.json").read_text()); assert len(manifest["units"]) == 64
    assert manifest["members"] == ["receipt.json", "failure.json", "checkpoint.json", *[f"units/{i:03d}.json" for i in range(64)], "manifest.json", "SHA256SUMS"]
    with __import__("zipfile").ZipFile(root / package["archive_filename"]) as bundle:
        assert bundle.namelist() == manifest["members"]
    digest, name = (root / package["sidecar_filename"]).read_text().split(); assert name == package["archive_filename"]
    assert digest == sha256((root / name).read_bytes()).hexdigest()


def test_control_fd_failure_has_no_stdout_fallback(tmp_path, capsys) -> None:
    path = tmp_path / "plan.json"; path.write_text("{}")
    assert RUNNER._main(["--plan", str(path), "--repo-root", str(tmp_path), "--expected-exact", "a" * 40,
                         "--output-root", str(tmp_path / "out"), "--control-fd", "-1"]) == 1
    assert capsys.readouterr().out == ""


def test_packaging_failure_preserves_known_operational_failure_in_compact_control(tmp_path, monkeypatch) -> None:
    plan_path = tmp_path / "plan.json"; plan_path.write_text(json.dumps(_plan(tmp_path)))
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    monkeypatch.setattr(RUNNER, "run_qk_equivariance_operational", lambda *_a, **_k: ({"run_id": "geometry-v1-qk-e0-aaaaaaaaaaaa", "operational_status": "failure"}, ()))
    monkeypatch.setattr(RUNNER, "_package", lambda *_a, **_k: (_ for _ in ()).throw(OSError("package")))
    read, write = os.pipe()
    try:
        rc = RUNNER._main(["--plan", str(plan_path), "--repo-root", str(tmp_path), "--expected-exact", "a" * 40,
                           "--output-root", str(tmp_path / "out"), "--control-fd", str(write)])
        control = os.read(read, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read); os.close(write)
    assert rc == 1
    assert b'"artifact_status":"unavailable"' in control
    assert b'"underlying_status":"operational_failure"' in control
    assert b'"failure_point":"artifact_packaging"' in control


def test_public_package_does_not_leak_secret_or_private_paths(tmp_path, monkeypatch) -> None:
    token, private = "hf_token_very_private", "/private/input/reference.png"
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    plan = _plan(tmp_path)
    plan["pairs"][0]["reference_path"] = private
    # The digest mismatch is an operational observation result; the public
    # transport is intentionally assembled only from field allowlists.
    summary, units = RUNNER.run_qk_equivariance_operational(plan, hf_token=token, expected_exact="a" * 40,
                                                             repo_root=tmp_path, loader=lambda *_a, **_k: _Pipeline(), observer=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("raw-tensor-like")))
    root = tmp_path / "out"; package = RUNNER._package(root, summary, units, expected_exact="a" * 40)
    public = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert token.encode() not in public and private.encode() not in public and b"raw-tensor-like" not in public
    assert b"r0-identity" in public


def test_runner_never_references_valid_corners_or_old_relation_transport() -> None:
    source = MODULE.read_text()
    for forbidden in ("valid_" + "corners", "CEG_WM_" + "ROOT_KEY", "keyed_" + "qk_relation", "_sanitize_" + "public"):
        assert forbidden not in source
