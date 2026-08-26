"""CPU/fake coverage for the sharded E0 operational experiment runner."""
from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

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
