"""CPU/fake contract coverage for the independent D0 operational transport."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

MODULE = Path(__file__).parents[2] / "experiments" / "run_geometry_v1_qk_all_layer_discovery_operational.py"
SPEC = importlib.util.spec_from_file_location("geometry_d0_operational", MODULE)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


class _Attn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.to_q = torch.nn.Linear(1, 1); self.to_k = torch.nn.Linear(1, 1); self.heads = 1


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.transformer_blocks = torch.nn.ModuleList([])
        for _ in range(24):
            block = torch.nn.Module(); block.attn = _Attn(); self.transformer_blocks.append(block)


class _Pipeline:
    def __init__(self) -> None: self.transformer = _Transformer(); self._commit_hash = None
    def to(self, _device): return self
    def encode_prompt(self, **_kwargs): return torch.zeros((1, 1)), None, torch.zeros((1, 1)), None


def _observation(paths):
    indices = torch.tensor([0, 1, 2, 3]); values = torch.eye(4)
    return SimpleNamespace(layers=tuple(SimpleNamespace(layer_path=path, query=values, key=values, source_grid=(2, 2), sample_indices=indices) for path in paths))


def test_fixed_plan_is_eight_pairs_and_known_h_has_deterministic_cyclic_shuffle() -> None:
    plan = RUNNER.build_fixed_plan()
    assert plan["protocol"] == RUNNER.PROTOCOL and plan["declared_unit_count"] == 768 and len(plan["pairs"]) == 8
    assert [p["transform_label"] for p in plan["pairs"][:4]] == list(RUNNER.TRANSFORMS)
    assert plan["pairs"][0]["shuffled_h"] == plan["pairs"][1]["matched_h"]
    # PIL uses output pixel indices; H uses pixel centres.  Verify the public
    # inverse-coordinate contract against the actual Pillow affine interface,
    # rather than merely re-evaluating the runner's H constants.
    for pair in plan["pairs"]:
        source = RUNNER._reference(pair["reference_id"])
        attacked, matrix = RUNNER._attack(source, pair["transform_label"])
        assert attacked.size == (512, 512)
        h = np.asarray(matrix)
        if pair["transform_label"] == "identity":
            assert np.allclose(h @ np.array([31.5, 77.5, 1.]), [31.5, 77.5, 1.])
        elif pair["transform_label"] == "d4":
            assert np.allclose(h @ np.array([31.5, 77.5, 1.]), [434.5, 31.5, 1.])
            assert attacked.getpixel((434, 31)) == source.getpixel((31, 77))
        elif pair["transform_label"] == "similarity":
            assert np.allclose(h, _frozen_similarity_h())
        else:
            # The first crop pixel centre remains (48.5,32.5), and the last
            # maps to the last output centre under Pillow crop+resize.
            assert np.allclose(h @ np.array([48.5, 32.5, 1.]), [.5, .5, 1.])
            assert np.allclose(h @ np.array([463.5, 447.5, 1.]), [511.5, 511.5, 1.])


def _marker_image() -> tuple[Image.Image, tuple[tuple[np.ndarray, tuple[int, int, int]], ...]]:
    """Independent colour landmarks, deliberately far from every image edge."""
    image = Image.new("RGB", (512, 512), (0, 0, 0)); draw = ImageDraw.Draw(image)
    markers = ((np.array([121.5, 109.5, 1.]), (255, 31, 19)),
               (np.array([366.5, 143.5, 1.]), (23, 255, 47)),
               (np.array([174.5, 351.5, 1.]), (41, 61, 255)),
               (np.array([341.5, 324.5, 1.]), (255, 223, 29)))
    for centre, colour in markers:
        x, y = (int(centre[0] - .5), int(centre[1] - .5))
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=colour)
    return image, markers


def _colour_centres(image: Image.Image, colours: tuple[tuple[int, int, int], ...]) -> dict[tuple[int, int, int], np.ndarray]:
    values = np.asarray(image, dtype=np.int16); yy, xx = np.indices(values.shape[:2])
    result = {}
    for colour in colours:
        target = np.asarray(colour, dtype=np.int16)
        # A pure 13x13 marker survives BICUBIC with a compact, high-confidence
        # interior; this mask never relies on a production coordinate helper.
        mask = (np.abs(values - target).max(axis=2) <= 28)
        assert mask.sum() >= 20
        result[colour] = np.array([(xx[mask].mean() + .5), (yy[mask].mean() + .5)])
    return result


def _frozen_similarity_h() -> np.ndarray:
    angle, scale, centre, translation = np.deg2rad(12.0), .90, np.array([256., 256.]), np.array([16., -12.])
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = centre + translation - linear @ centre
    return np.array([[linear[0, 0], linear[0, 1], offset[0]], [linear[1, 0], linear[1, 1], offset[1]], [0., 0., 1.]])


def test_actual_pillow_rgb_landmarks_independently_verify_similarity_and_crop_h() -> None:
    source, markers = _marker_image(); colours = tuple(colour for _, colour in markers)
    frozen = {
        "similarity": _frozen_similarity_h(),
        # Crop [48,32,464,448], then resize 416 -> 512: derive centre mapping
        # directly from the frozen crop geometry, independent of runner helpers.
        "crop_rescale": np.array([[512 / 416, 0., .5 - 48.5 * 512 / 416], [0., 512 / 416, .5 - 32.5 * 512 / 416], [0., 0., 1.]]),
    }
    for label, expected_h in frozen.items():
        attacked, returned_h = RUNNER._attack(source, label)
        observed = _colour_centres(attacked, colours)
        for point, colour in markers:
            expected = (expected_h @ point)[:2]
            returned = (np.asarray(returned_h) @ point)[:2]
            # 1.25 px is a predeclared BICUBIC support/quantisation allowance,
            # not a fitted quality threshold.
            assert np.linalg.norm(observed[colour] - expected) <= 1.25
            assert np.linalg.norm(observed[colour] - returned) <= 1.25


def test_discovery_accepts_only_complete_contiguous_sample_side_roster() -> None:
    paths, record = RUNNER._discover(_Pipeline().transformer)
    assert paths == tuple(f"transformer_blocks.{i}.attn" for i in range(24))
    assert record["candidate_count"] == 24
    transformer = _Pipeline().transformer; del transformer.transformer_blocks[23]
    with pytest.raises(ValueError, match="24_layer"): RUNNER._discover(transformer)


def test_d0_observes_ten_images_and_retains_all_768_ordered_units(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected)
    def observer(image, *, pipeline, spec): calls.append(image.size); return _observation(spec.attention_layer_paths)
    summary, units = RUNNER.run_d0(expected_exact="a" * 40, repo_root=tmp_path, hf_token="secret", loader=lambda *_a, **_k: _Pipeline(), observer=observer)
    assert len(calls) == 10 and len(units) == 768 and summary["science_denominator"] == 0
    assert [(u["pair_id"], u["layer_path"], u["descriptor_kind"], u["control_label"]) for u in units[:4]] == [("reference_a-identity", "transformer_blocks.0.attn", "q", "matched_h"), ("reference_a-identity", "transformer_blocks.0.attn", "q", "shuffled_h"), ("reference_a-identity", "transformer_blocks.0.attn", "k", "matched_h"), ("reference_a-identity", "transformer_blocks.0.attn", "k", "shuffled_h")]


def test_reference_failure_still_attempts_ten_observations_and_keeps_768(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected); calls = []
    def observer(image, *, pipeline, spec):
        calls.append(image.size)
        if len(calls) == 1: raise RuntimeError("reference unavailable")
        return _observation(spec.attention_layer_paths)
    _summary, units = RUNNER.run_d0(expected_exact="a" * 40, repo_root=tmp_path, hf_token="secret", loader=lambda *_a, **_k: _Pipeline(), observer=observer)
    assert len(calls) == 10 and len(units) == 768
    assert {item["failure_reason"] for item in units[:384]} == {"reference_observation_failed"}


def test_global_observer_and_model_failures_keep_full_ordered_denominator(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RUNNER, "_exact", lambda expected, root: expected); calls = []
    def observer(image, *, pipeline, spec):
        calls.append(image.size); error = RuntimeError("transformer")
        setattr(error, "geometry_failure_point", "transformer_call"); raise error
    summary, units = RUNNER.run_d0(expected_exact="a" * 40, repo_root=tmp_path, hf_token="secret", loader=lambda *_a, **_k: _Pipeline(), observer=observer)
    assert len(calls) == 10 and len(units) == 768 and {item["failure_reason"] for item in units} == {"global_transformer_failure"}
    summary, units = RUNNER.run_d0(expected_exact="a" * 40, repo_root=tmp_path, hf_token="secret", loader=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("model")), observer=lambda *_a, **_k: pytest.fail("observer must not run"))
    assert summary["d0_status"] == "D0_STOPPED" and len(units) == 768 and {item["failure_reason"] for item in units} == {"model_or_topology_unavailable"}


def test_24_layer_shards_are_exact_and_bounds_fail_closed(tmp_path) -> None:
    summary = {"run_id": "geometry-v1-qk-d0-aaaaaaaaaaaa", "d0_status": "D0_UNRESOLVED", "artifact_status": "unavailable"}
    unit = {"pair_id": "p", "transform_label": "identity", "control_label": "matched_h", "descriptor_kind": "q", "layer_path": "", "reference_grid": None, "attacked_grid": None, "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": "x", "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}
    units = tuple({**unit, "layer_path": f"transformer_blocks.{index}.attn"} for _pair in range(8) for index in range(24) for _kind in ("q", "k") for _control in ("matched_h", "shuffled_h"))
    RUNNER._package(tmp_path / "out", summary, units, exact="a" * 40)
    assert len(list((tmp_path / "out" / "layers").glob("*.zip"))) == 24
