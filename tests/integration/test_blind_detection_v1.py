from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus
from cegwm.method.blind_detection import (
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    build_test_only_threshold_asset,
)
from cegwm.method.content_weighted_joint import WeightedJointAsset
from cegwm.runtime import blind_detection as runtime
from cegwm.runtime.content_weighted_joint_sd35 import ContentCalibrationAssets
from experiments import run_blind_detection_v1 as runner

pytestmark = pytest.mark.integration


KEY = "blind-detection-test-key"
IDENTITY_H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _geometry(*, h=IDENTITY_H, legal=True, error=None) -> GeometryEstimate:
    return GeometryEstimate(
        GeometryStatus.UNRELIABLE if error is None else GeometryStatus.ERROR,
        99.0,
        None,
        None,
        h,
        legal,
        legal,
        error,
    )


class GeometryBackend:
    def __init__(self, estimate: GeometryEstimate):
        self.estimate = estimate
        self.calls = 0

    def detect_geometry(self, image):
        self.calls += 1
        return self.estimate


def _content_assets() -> ContentCalibrationAssets:
    return object.__new__(ContentCalibrationAssets)


def _assets(backend, tau=1.0, *, threshold=True) -> runtime.BlindPublicAssets:
    asset = build_test_only_threshold_asset(tau) if threshold else None
    return runtime.BlindPublicAssets(
        _content_assets(), WeightedJointAsset({}, b""), backend, asset
    )


def _weighted(value: float) -> dict[str, dict[str, float]]:
    weighted = {"registered": value, **{f"wrong_{index:02d}": 0.0 for index in range(16)}}
    return {"lf": dict(weighted), "hf": dict(weighted), "weighted_joint": weighted}


def _install_scores(monkeypatch, values):
    iterator = iter(values)
    seen = []

    def score(image, key, wrong_keys, content_assets, calibration_asset):
        seen.append((image.mode, image.size, key, wrong_keys, content_assets, calibration_asset))
        return _weighted(next(iterator))

    monkeypatch.setattr(runtime, "blind_weighted_scores", score)
    monkeypatch.setattr(
        runtime,
        "derive_stability_wrong_keys",
        lambda key: tuple(bytes([index]) * 32 for index in range(16)),
    )
    return seen


def test_direct_positive_strictly_exceeds_tau_and_short_circuits_geometry(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    seen = _install_scores(monkeypatch, [1.01])
    record = runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _assets(backend))
    assert record.route == "DIRECT_POSITIVE" and record.positive
    assert not record.recovered and record.post is None and backend.calls == 0
    assert len(seen) == 1 and len(seen[0][3]) == 16


def test_equality_is_negative_and_enters_geometry_without_b_low(monkeypatch) -> None:
    backend = GeometryBackend(_geometry(h=None, legal=True))
    _install_scores(monkeypatch, [1.0])
    record = runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _assets(backend))
    assert record.route == "GEOMETRY_NO_H"
    assert not record.positive and backend.calls == 1


def test_single_raw_h_recovery_reuses_same_scorer_key_assets_and_tau(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    seen = _install_scores(monkeypatch, [0.25, 1.5])
    real_rectify = runtime.rectify_attacked_rgb
    rectifications = []

    def rectify_once(image, matrix):
        rectifications.append(matrix)
        return real_rectify(image, matrix)

    monkeypatch.setattr(runtime, "rectify_attacked_rgb", rectify_once)
    assets = _assets(backend)
    record = runtime.detect_watermark(Image.new("RGB", (512, 512), "white"), KEY, assets)
    assert record.route == "GEOMETRY_RECOVERED" and record.positive and record.recovered
    assert backend.calls == len(rectifications) == 1 and len(seen) == 2
    assert seen[0][2:] == seen[1][2:]
    assert record.same_scoring_context and record.tau_blind == 1.0
    assert record.geometry.uncalibrated_sync_logit == 99.0  # recorded, never voted


@pytest.mark.parametrize(
    "estimate,route",
    [
        (_geometry(h=None, legal=True), "GEOMETRY_NO_H"),
        (_geometry(h=((1.0, 0.0, 0.0),) * 3), "GEOMETRY_FAIL_CLOSED"),
        (_geometry(h=IDENTITY_H, legal=False), "GEOMETRY_FAIL_CLOSED"),
        (GeometryEstimate.error_record("detector malformed"), "GEOMETRY_FAIL_CLOSED"),
    ],
)
def test_no_h_invalid_h_and_geometry_error_are_fail_closed(monkeypatch, estimate, route) -> None:
    backend = GeometryBackend(estimate)
    _install_scores(monkeypatch, [0.0])
    record = runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _assets(backend))
    assert record.route == route and not record.positive and record.post is None
    assert backend.calls == 1


def test_post_score_error_is_retained_without_retry(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    calls = 0

    def score(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("post scorer stopped")
        return _weighted(0.0)

    monkeypatch.setattr(runtime, "blind_weighted_scores", score)
    monkeypatch.setattr(runtime, "derive_stability_wrong_keys", lambda key: (b"x" * 32,) * 16)
    record = runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _assets(backend))
    assert record.route == "ERROR_FAIL_CLOSED" and not record.positive
    assert record.recovered and calls == 2 and "post scorer stopped" in record.error


def test_production_detection_refuses_absent_threshold_before_scoring(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    monkeypatch.setattr(runtime, "blind_weighted_scores", lambda *args: pytest.fail("must not score"))
    with pytest.raises(runtime.ThresholdUnavailableError, match="production detection refused"):
        runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _assets(backend, threshold=False))


def test_development_calibration_freezes_256_and_retains_no_h(monkeypatch) -> None:
    backend = GeometryBackend(_geometry(h=None, legal=True))
    seen = _install_scores(monkeypatch, [float(index) for index in range(256)])
    roster = BlindCalibrationRoster(
        tuple(
            BlindCalibrationUnit(f"u-{index}", f"s-{index % 2}", f"ref-{index}", f"base-{index}")
            for index in range(256)
        )
    )
    loaded = []
    rows = runtime.run_development_calibration(
        roster,
        KEY,
        _assets(backend, threshold=False),
        lambda ref: loaded.append(ref) or Image.new("RGB", (512, 512)),
    )
    assert len(rows) == len(loaded) == len(seen) == backend.calls == 256
    assert all(row.method_complete and row.geometry_outcome == "NO_H" for row in rows)
    assert all(row.z == row.pre_score for row in rows)


def test_embedding_order_is_content_then_final_rgb_syncseal_once() -> None:
    calls = []

    class Content:
        def embed_content(self, request, key):
            calls.append(("content", request, key))
            return Image.new("RGB", (512, 512), "white")

    class Sync:
        def embed_final_rgb(self, image, multiplier):
            calls.append(("syncseal", image.mode, multiplier))
            return image

    output = runtime.embed_watermark(
        {"pipeline_request": "injected"},
        KEY,
        runtime.BlindEmbeddingAssets(Content(), Sync(), 1.0),
    )
    assert output.mode == "RGB"
    assert [call[0] for call in calls] == ["content", "syncseal"]


def test_threshold_runner_is_complete_fixed_denominator_and_create_only(tmp_path: Path) -> None:
    units = tuple(
        BlindCalibrationUnit(f"u-{index}", f"s-{index % 2}", f"ref-{index}", f"base-{index}")
        for index in range(256)
    )
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "disjoint_from": [
                    "geometry_v7_development",
                    "future_paper_calibration",
                    "future_paper_test",
                ],
                "units": [asdict(unit) for unit in units],
            }
        ),
        encoding="utf-8",
    )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps(
                {
                    "geometry_outcome": "NO_H",
                    "method_complete": True,
                    "operational_error": None,
                    "post_score": None,
                    "pre_score": float(index),
                    "roster_index": index,
                    "source_stratum": f"s-{index % 2}",
                    "unit_id": f"u-{index}",
                }
            ) + "\n"
            for index in range(256)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "threshold.json"
    assert runner.freeze_threshold(
        roster_path, rows_path, output, producer_exact="c" * 40
    ) == output
    assert output.is_file()
    with pytest.raises(FileExistsError):
        runner.freeze_threshold(roster_path, rows_path, output, producer_exact="c" * 40)


def test_notebook_first_executable_cell_and_static_callback_contract() -> None:
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "blind_detection_v1_callback.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
    ]
    source = "".join("".join(cell["source"]) for cell in code_cells)
    assert "force_remount" not in source
    assert "N_CALLBACK = 4" in source
    assert "--detach" in source and "torch.cuda.is_available()" in source
    assert "RUNNER_CALLS == 0" in source and "RUNNER_CALLS += 1" in source
    assert "OUTPUT.exists()" in source and "--output" in source
    runner_source = (
        Path(__file__).resolve().parents[2] / "experiments" / "run_blind_detection_v1.py"
    ).read_text(encoding="utf-8")
    assert runner_source.count("detect_watermark(current_rgb, detection_key, public_assets)") == 1
    assert 'output.open("xb")' in runner_source
