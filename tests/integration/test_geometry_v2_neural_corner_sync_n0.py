from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cegwm.geometry_v2.neural_sync import CornerPrediction, KeyedResidualEmbedder, MAX_RESIDUAL
from cegwm.geometry_v2 import operational as N0


RUNNER_PATH = Path(__file__).parents[2] / "experiments" / "run_geometry_v2_neural_corner_sync_n0.py"
SPEC = importlib.util.spec_from_file_location("geometry_v2_n0_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(RUNNER)


def _unit(error: float = 0.01, *, reliable: bool = True) -> dict[str, object]:
    return {"seed": 3000, "attack": "identity", "status": "calculated", "mean_corner_error": error, "reliable": reliable}


@pytest.mark.integration
def test_frozen_splits_roster_and_status_gates_retain_failures() -> None:
    assert N0.TRAIN_SEEDS == tuple(range(1000, 1128))
    assert N0.VALIDATION_SEEDS == tuple(range(2000, 2032))
    assert N0.CONFIRMATION_SEEDS == tuple(range(3000, 3032))
    assert not (set(N0.TRAIN_SEEDS) & set(N0.VALIDATION_SEEDS) | set(N0.TRAIN_SEEDS) & set(N0.CONFIRMATION_SEEDS) | set(N0.VALIDATION_SEEDS) & set(N0.CONFIRMATION_SEEDS))
    candidate = [_unit() for _ in range(128)]
    status, metrics = N0.decide_n0_status(candidate, actual_residual_max=MAX_RESIDUAL)
    assert status == N0.STATUS_CANDIDATE and metrics["calculated_unit_count"] == 128
    unresolved = [_unit(0.08, reliable=False) for _ in range(128)]
    assert N0.decide_n0_status(unresolved, actual_residual_max=MAX_RESIDUAL)[0] == N0.STATUS_UNRESOLVED
    stopped = candidate.copy(); stopped[17] = {"seed": 3004, "attack": "rotate90", "status": "failed", "failure_class": "geometry_estimation_error"}
    status, metrics = N0.decide_n0_status(stopped, actual_residual_max=MAX_RESIDUAL)
    assert status == N0.STATUS_STOPPED and metrics["failed_unit_count"] == 1 and metrics["declared_unit_count"] == 128


def _gaussian(point: np.ndarray, channel: int) -> torch.Tensor:
    yy, xx = np.indices((128, 128), dtype=np.float64)
    value = 0.05 + 0.9 * np.exp(-((xx + 0.5 - point[0]) ** 2 + (yy + 0.5 - point[1]) ** 2) / (2 * 5.0 ** 2))
    rgb = np.full((128, 128, 3), 0.05, dtype=np.float32); rgb[:, :, channel] = value.astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _centroid(image: torch.Tensor, channel: int, expected: np.ndarray) -> np.ndarray:
    values = image[channel].double().numpy() - 0.05
    cx, cy = int(round(expected[0] - 0.5)), int(round(expected[1] - 0.5))
    radius = 18; left, top = cx - radius, cy - radius
    window = np.maximum(values[top:cy + radius + 1, left:cx + radius + 1], 0.0)
    yy, xx = np.indices(window.shape, dtype=np.float64)
    return np.array(((window * (xx + left + 0.5)).sum() / window.sum(), (window * (yy + top + 0.5)).sum() / window.sum()))


@pytest.mark.integration
def test_actual_pillow_attacks_match_independent_corner_and_h_correspondence() -> None:
    points = (np.array((38.5, 41.5, 1.0)), np.array((88.5, 39.5, 1.0)), np.array((86.5, 88.5, 1.0)), np.array((42.5, 86.5, 1.0)))
    for label in N0.ATTACKS:
        for index, point in enumerate(points):
            attacked, returned_h, corners = N0.apply_pillow_attack(_gaussian(point, index % 3), label)
            normalized = np.array((point[0] / 128.0, point[1] / 128.0, 1.0))
            expected_norm = returned_h @ normalized; expected = expected_norm[:2] / expected_norm[2] * 128.0
            observed = _centroid(attacked, index % 3, expected)
            assert np.linalg.norm(observed - expected) <= 0.15
            canonical = np.array(((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), dtype=np.float64)
            projected = (returned_h @ canonical.T).T; projected = projected[:, :2] / projected[:, 2:3]
            assert np.allclose(projected, corners)


class _DegenerateExtractor(torch.nn.Module):
    def forward(self, attacked: torch.Tensor) -> CornerPrediction:
        batch = attacked.shape[0]
        return CornerPrediction(torch.full((batch, 4, 2), 0.5), torch.ones(batch), torch.ones(batch))


@pytest.mark.integration
def test_evaluation_retains_each_failed_attack_unit() -> None:
    records, errors, _ = N0._evaluate_split(
        (3000,), b"g" * 32, KeyedResidualEmbedder(), _DegenerateExtractor(), torch.device("cpu"), retain_units=True
    )
    assert len(records) == 4 and not errors
    assert [record["attack"] for record in records] == list(N0.ATTACKS)
    assert all(record["status"] == "failed" for record in records)


@pytest.mark.integration
def test_runner_fake_path_packages_bounded_public_artifact_without_sensitive_material(monkeypatch, tmp_path: Path) -> None:
    units = tuple({**_unit(), "seed": 3000 + index // 4, "attack": N0.ATTACKS[index % 4]} for index in range(128))
    result = N0.N0RunResult(
        summary={"protocol": RUNNER.PROTOCOL_IDENTITY, "n0_status": N0.STATUS_UNRESOLVED, "science_denominator": 0,
                 "confirmation": {"declared_unit_count": 128, "calculated_unit_count": 128, "failed_unit_count": 0},
                 "weights_persisted": False, "images_persisted": False, "raw_geometry_key_persisted": False},
        units=units,
    )
    exact = "a" * 40
    monkeypatch.setattr(RUNNER, "_execution_exact", lambda expected, root: exact)
    monkeypatch.setattr(RUNNER, "_geometry_key", lambda: b"k" * 32)
    monkeypatch.setattr(RUNNER, "_device", lambda requested: "cpu")
    monkeypatch.setattr(RUNNER, "run_n0", lambda key, device_name: result)
    read_fd, write_fd = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", exact, "--output-root", str(tmp_path / "out"), "--control-fd", str(write_fd), "--device", "cpu"])
        os.close(write_fd); line = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read_fd)
    assert rc == 0 and line.startswith(RUNNER.SUCCESS_PREFIX.encode()) and len(line) <= RUNNER.MAX_CONTROL_BYTES
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["n0_status"] == N0.STATUS_UNRESOLVED and receipt["science_denominator"] == 0
    artifact = b"".join(path.read_bytes() for path in sorted((tmp_path / "out").iterdir()))
    lowered = artifact.lower()
    for forbidden in (b"raw_qk", b"geometry_key_hex", b"model_weights\" : true", b"image_bytes", b"prompt", b"latent"):
        assert forbidden not in lowered


@pytest.mark.integration
def test_runner_failure_receipt_is_bounded_and_does_not_echo_exception(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(RUNNER, "_execution_exact", lambda expected, root: "b" * 40)
    monkeypatch.setattr(RUNNER, "_geometry_key", lambda: b"k" * 32)
    monkeypatch.setattr(RUNNER, "run_n0", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret private path")))
    read_fd, write_fd = os.pipe()
    try:
        rc = RUNNER._main(["--repo-root", str(tmp_path), "--expected-exact", "b" * 40, "--output-root", str(tmp_path / "out"), "--control-fd", str(write_fd)])
        os.close(write_fd); line = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    finally:
        os.close(read_fd)
    assert rc == 1 and len(line) <= RUNNER.MAX_CONTROL_BYTES
    assert b"failure_point" in line and b"run_n0" in line and b"secret private path" not in line
