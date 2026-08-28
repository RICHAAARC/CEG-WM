from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import cegwm.geometry_v3.operational as OP
from cegwm.geometry_v3.active_writer import P0_CONFIGS


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "experiments" / "run_geometry_v3_qk_active_writer_p0.py"
    spec = importlib.util.spec_from_file_location("geometry_v3_p0_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _gaussian_marker(cx: float, cy: float, sigma: float = 18.0) -> Image.Image:
    y, x = np.mgrid[0:512, 0:512]
    signal = 250.0 * np.exp(-((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) / (2.0 * sigma**2))
    pixels = np.clip(np.rint(signal), 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2), mode="RGB")


def _centroid(image: Image.Image, expected_x: float, expected_y: float) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64).mean(axis=2)
    x0, x1 = max(0, int(expected_x) - 64), min(512, int(expected_x) + 65)
    y0, y1 = max(0, int(expected_y) - 64), min(512, int(expected_y) + 65)
    window = values[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weights = np.maximum(window, 0.0)
    total = float(weights.sum())
    assert total > 0.0
    return float(np.sum((xx + 0.5) * weights) / total), float(np.sum((yy + 0.5) * weights) / total)


@pytest.mark.integration
@pytest.mark.parametrize("attack_id", OP.ATTACK_IDS)
def test_actual_pillow_attack_correspondence_matches_public_h(attack_id: str) -> None:
    for source_x, source_y in ((176.25, 188.75), (331.5, 207.25), (242.75, 318.5)):
        attacked = OP.apply_attack(_gaussian_marker(source_x, source_y), attack_id)
        h = np.asarray(attacked.homography, dtype=np.float64)
        mapped = h @ np.array((source_x, source_y, 1.0))
        expected_x, expected_y = mapped[0] / mapped[2], mapped[1] / mapped[2]
        observed_x, observed_y = _centroid(attacked.image, expected_x, expected_y)
        tolerance = 0.12 if attack_id in {"similarity", "crop_rescale"} else 0.04
        assert abs(observed_x - expected_x) <= tolerance
        assert abs(observed_y - expected_y) <= tolerance


@pytest.mark.integration
def test_fresh_observer_rejects_embedding_tensor_or_cached_qk_before_runtime_use() -> None:
    with pytest.raises(TypeError, match="ordinary RGB"):
        OP.observe_fresh_attacked_rgb(
            SimpleNamespace(),
            torch.zeros((1, 4, 64, 64)),
            P0_CONFIGS[0],
            object(),
            object(),
            np.eye(3),
        )


def _calculated_records(margins: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for config_id, attack, kind, control in OP.fixed_roster():
        q_margin, k_margin = margins[config_id]
        margin = q_margin if kind == "q" else k_margin
        records.append(
            {
                "config_id": config_id,
                "attack_id": attack,
                "feature_kind": kind,
                "control": control,
                "status": "calculated",
                "error_class": None,
                "score": 0.1,
                "margin": margin,
            }
        )
    return records


@pytest.mark.integration
def test_fixed_144_roster_and_predeclared_selection_ties() -> None:
    assert len(OP.fixed_roster()) == 144
    margins = {config.config_id: (0.2, 0.2) for config in P0_CONFIGS}
    winner, summaries = OP.select_writer_candidate(_calculated_records(margins))
    assert winner == P0_CONFIGS[0].config_id
    assert len(summaries) == 6
    assert all(summary["calculated_unit_count"] == 24 for summary in summaries)


@pytest.mark.integration
def test_zero_margin_is_ineligible_and_no_candidate_is_unresolved() -> None:
    margins = {config.config_id: (0.2, 0.0) for config in P0_CONFIGS}
    status, summaries = OP.select_writer_candidate(_calculated_records(margins))
    assert status == OP.P0_STATUS_UNRESOLVED
    assert not any(summary["eligible"] for summary in summaries)


@pytest.mark.integration
def test_writer_generation_failure_retains_all_144_units(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (512, 512), (4, 5, 6))
    monkeypatch.setattr(OP, "generate_no_writer", lambda pipeline: image)

    def generate(pipeline, config, anchor):
        del pipeline, anchor
        if config == P0_CONFIGS[0]:
            raise RuntimeError("private failure text must not escape")
        return OP.GeneratedConfig(image, ())

    monkeypatch.setattr(OP, "generate_writer_config", generate)
    monkeypatch.setattr(
        OP,
        "observe_fresh_attacked_rgb",
        lambda *args, **kwargs: OP.ObservationScores(0.3, 0.1, 0.25, 0.05),
    )
    result = OP.run_p0(object(), "geometry-key-0001")
    assert result.status == OP.P0_STATUS_STOPPED
    assert len(result.records) == 144
    assert sum(record["status"] == "failed" for record in result.records) == 24
    assert {record["error_class"] for record in result.records if record["status"] == "failed"} == {"runtime_error"}


@pytest.mark.integration
def test_bounded_artifact_contains_only_public_derived_data(tmp_path: Path) -> None:
    margins = {config.config_id: (0.2, 0.1) for config in P0_CONFIGS}
    records = tuple(_calculated_records(margins))
    winner, summaries = OP.select_writer_candidate(records)
    result = OP.P0ExecutionResult(
        OP.P0_STATUS_FROZEN,
        winner,
        records,
        summaries,
        ({"config_id": winner, "rgb_mse": 1.0, "rgb_psnr_db": 48.0,
          "content_detector_hook_status": "not_invoked_record_only"},),
        ({"config_id": winner, "feature_kind": "q",
          "module_path": "transformer_blocks.4.attn.to_q",
          "relative_rms_budget": 0.0025, "actual_relative_rms": 0.00249,
          "call_count": 1, "writer_step_index": 18},),
        None,
    )
    root = tmp_path / "p0"
    control = OP.package_p0_artifacts(root, exact="a" * 40, result=result)
    assert control["status"] == OP.P0_STATUS_FROZEN
    assert {path.name for path in root.iterdir()} == {
        "receipt.json", "manifest.json", "terminal.json", "metrics.jsonl"
    }
    payload = b"".join(path.read_bytes() for path in root.iterdir())
    assert len(payload) < OP.P0_ARTIFACT_MAX_BYTES
    lowered = payload.lower()
    for forbidden in (b"geometry-key", b"raw_qk", b"prompt_text", b"latent", b"image_bytes", b"hf_token"):
        assert forbidden not in lowered
    assert not any(path.suffix.lower() in {".png", ".jpg", ".pt", ".bin", ".zip"} for path in root.iterdir())
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["plan_digest"]) == len(receipt["roster_digest"]) == 64
    assert receipt["writer_measurements"][0]["call_count"] == 1


@pytest.mark.integration
def test_execute_plan_uses_one_fake_preloader_and_create_only_packaging(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    pipeline = object()
    result = OP.P0ExecutionResult(
        OP.P0_STATUS_UNRESOLVED, None, tuple(), tuple(), tuple(), tuple(), None
    )
    monkeypatch.setattr(OP, "run_p0", lambda received, key: result if received is pipeline and key == "key" else None)
    monkeypatch.setattr(
        OP,
        "package_p0_artifacts",
        lambda path, exact, result: {
            "run_id": "run", "status": result.status, "artifact_status": "complete",
            "selected_config_id": None, "science_denominator": 0,
        },
    )

    def preloader(model_id: str, token: str):
        calls.append((model_id, token))
        return pipeline

    control = OP.execute_plan(
        {
            "expected_exact": "b" * 40,
            "execution_exact": "b" * 40,
            "output_directory": "/content/drive/MyDrive/CEG-WM/Geometry-V3/P0/run",
        },
        geometry_key="key",
        hf_token="token",
        preloader=preloader,
    )
    assert calls == [(OP.P0_MODEL_ID, "token")]
    assert control["status"] == OP.P0_STATUS_UNRESOLVED


@pytest.mark.integration
def test_runner_fake_preloader_control_receipt_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exact = "c" * 40
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "expected_exact": exact,
                "execution_exact": exact,
                "output_directory": "/content/drive/MyDrive/CEG-WM/Geometry-V3/P0/run",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RUNNER, "_git_exact", lambda expected: expected)
    seen: list[object] = []

    def execute(plan_value, *, geometry_key, hf_token, preloader):
        seen.extend((plan_value, geometry_key, hf_token, preloader))
        return {
            "run_id": "run", "status": OP.P0_STATUS_UNRESOLVED,
            "artifact_status": "complete", "selected_config_id": None,
            "science_denominator": 0,
        }

    monkeypatch.setattr(RUNNER, "execute_plan", execute)
    monkeypatch.setenv(RUNNER.TOKEN_ENV, "token")
    monkeypatch.setenv(RUNNER.KEY_ENV, "key")
    read_fd, write_fd = os.pipe()
    rc = RUNNER._main(["--plan", str(plan), "--control-fd", str(write_fd)], preloader=object())
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    assert rc == 0 and len(payload) <= RUNNER.MAX_CONTROL_BYTES
    assert json.loads(payload)["status"] == "success"
    assert json.loads(payload)["p0_status"] == OP.P0_STATUS_UNRESOLVED
    assert seen[1:3] == ["key", "token"]
    assert RUNNER.TOKEN_ENV not in os.environ and RUNNER.KEY_ENV not in os.environ
