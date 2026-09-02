from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus, estimate_geometry
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from cegwm.method.blind_detection import (
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    build_test_only_threshold_asset,
    load_threshold_asset,
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
        99.0, None, None, h, legal, legal, error,
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


def _syncseal() -> SyncSealTorchScript:
    return object.__new__(SyncSealTorchScript)


def _test_assets(backend) -> runtime.BlindTestAssets:
    return runtime.BlindTestAssets(_content_assets(), WeightedJointAsset({}, b""), backend)


def _production_assets(threshold=None) -> runtime.BlindProductionAssets:
    return runtime.BlindProductionAssets(
        _content_assets(), WeightedJointAsset({}, b""), _syncseal(), threshold
    )


def _detect(image, backend, tau=1.0):
    return runtime.detect_watermark_test_only(
        image, KEY, _test_assets(backend), build_test_only_threshold_asset(tau)
    )


def _weighted(value: float) -> dict[str, dict[str, float]]:
    weighted = {"registered": value, **{f"wrong_{index:02d}": 0.0 for index in range(16)}}
    return {"lf": dict(weighted), "hf": dict(weighted), "weighted_joint": weighted}


def _install_scores(monkeypatch, values):
    iterator = iter(values)
    seen = []

    def score(image, key, wrong_keys, content_assets, calibration_asset):
        seen.append((image.mode, image.size, key, wrong_keys, content_assets, calibration_asset))
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return _weighted(value)

    monkeypatch.setattr(runtime, "blind_weighted_scores", score)
    monkeypatch.setattr(
        runtime,
        "derive_stability_wrong_keys",
        lambda key: tuple(bytes([index]) * 32 for index in range(16)),
    )
    return seen


def _roster() -> BlindCalibrationRoster:
    return BlindCalibrationRoster(
        tuple(
            BlindCalibrationUnit(f"u-{index}", f"s-{index % 2}", f"ref-{index}", f"base-{index}")
            for index in range(256)
        )
    )


def test_direct_positive_strictly_exceeds_tau_and_short_circuits_geometry(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    seen = _install_scores(monkeypatch, [1.01])
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "DIRECT_POSITIVE" and record.positive and record.method_complete
    assert not record.recovered and record.post is None and backend.calls == 0
    assert len(seen) == 1 and len(seen[0][3]) == 16


def test_equality_is_negative_and_enters_geometry_without_b_low(monkeypatch) -> None:
    backend = GeometryBackend(_geometry(h=None, legal=True))
    _install_scores(monkeypatch, [1.0])
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "GEOMETRY_NO_H" and record.method_complete
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
    record = _detect(Image.new("RGB", (512, 512), "white"), backend)
    assert record.route == "GEOMETRY_RECOVERED" and record.positive and record.recovered
    assert backend.calls == len(rectifications) == 1 and len(seen) == 2
    assert seen[0][2:] == seen[1][2:]
    assert record.same_scoring_context and record.tau_blind == 1.0
    assert record.geometry.uncalibrated_sync_logit == 99.0


@pytest.mark.parametrize(
    "estimate,route",
    [
        (_geometry(h=None, legal=True), "GEOMETRY_NO_H"),
        (_geometry(h=((1.0, 0.0, 0.0),) * 3), "GEOMETRY_FAIL_CLOSED"),
    ],
)
def test_no_h_and_invalid_h_are_complete_fail_closed(monkeypatch, estimate, route) -> None:
    backend = GeometryBackend(estimate)
    _install_scores(monkeypatch, [0.0])
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == route and not record.positive and record.method_complete
    assert backend.calls == 1


def test_real_unsupported_geometry_is_complete_invalid_h_in_detection_and_calibration(
    monkeypatch,
) -> None:
    unsupported = estimate_geometry(0.0, ((0.0, 0.0),) * 4)
    assert unsupported.status is GeometryStatus.UNSUPPORTED
    assert not unsupported.legal and unsupported.homography_observed_to_canonical is None
    assert unsupported.error is not None
    backend = GeometryBackend(unsupported)
    _install_scores(monkeypatch, [0.0] * 257)
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "GEOMETRY_FAIL_CLOSED" and not record.positive
    assert record.method_complete and record.operational_error is None
    assert record.geometry is unsupported
    rows = runtime.run_development_calibration(
        _roster(), KEY, _test_assets(backend), lambda ref: Image.new("RGB", (512, 512))
    )
    assert len(rows) == 256 and backend.calls == 257
    assert all(row.geometry_outcome == "INVALID_H" for row in rows)
    assert all(row.method_complete and row.operational_error is None for row in rows)


def test_every_geometry_error_is_operational_and_retained(monkeypatch) -> None:
    raw = "unclassified backend failure that must not be guessed"
    backend = GeometryBackend(GeometryEstimate.error_record(raw))
    _install_scores(monkeypatch, [0.0] * 257)
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "ERROR_FAIL_CLOSED" and not record.method_complete
    assert record.operational_error == f"geometry_runtime:{raw}"
    rows = runtime.run_development_calibration(
        _roster(), KEY, _test_assets(backend), lambda ref: Image.new("RGB", (512, 512))
    )
    assert len(rows) == 256 and backend.calls == 257
    assert all(not row.method_complete for row in rows)
    assert all(row.operational_error == f"geometry_runtime:{raw}" for row in rows)


def test_unknown_geometry_status_fails_operationally(monkeypatch) -> None:
    unknown = GeometryEstimate(
        "FUTURE_STATUS", 0.0, None, None, None, False, False, None  # type: ignore[arg-type]
    )
    backend = GeometryBackend(unknown)
    _install_scores(monkeypatch, [0.0])
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "ERROR_FAIL_CLOSED" and not record.method_complete
    assert "unknown GeometryStatus" in record.operational_error


def test_inconsistent_observable_geometry_blocks_detection_calibration_and_replay(
    monkeypatch,
) -> None:
    inconsistent = GeometryEstimate(
        GeometryStatus.UNRELIABLE, 0.0, None, None, IDENTITY_H, True, False, None
    )
    backend = GeometryBackend(inconsistent)
    seen = _install_scores(monkeypatch, [0.0] * 513)
    rectifications = []

    def forbidden_rectification(image, matrix):
        rectifications.append((image, matrix))
        raise AssertionError("inconsistent Geometry must never be rectified")

    monkeypatch.setattr(runtime, "rectify_attacked_rgb", forbidden_rectification)
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "ERROR_FAIL_CLOSED" and not record.positive
    assert not record.method_complete and not record.recovered
    assert "UNRELIABLE invariant violation" in record.operational_error
    rows = runtime.run_development_calibration(
        _roster(), KEY, _test_assets(backend), lambda ref: Image.new("RGB", (512, 512))
    )
    assert all(row.geometry_outcome == "GEOMETRY_ERROR" for row in rows)
    assert all(not row.method_complete and row.operational_error for row in rows)
    replay = runtime.run_development_full_system_replay(
        _roster(), KEY, _test_assets(backend),
        lambda ref: Image.new("RGB", (512, 512)), 1.0,
    )
    assert all(row.route == "ERROR_FAIL_CLOSED" and not row.positive for row in replay)
    assert all(not row.method_complete and not row.recovered for row in replay)
    assert len(seen) == backend.calls == 513
    assert rectifications == []


@pytest.mark.parametrize(
    "inconsistent",
    [
        GeometryEstimate(
            GeometryStatus.UNSUPPORTED, 0.0, None, None, None, True, False, "degenerate"
        ),
        GeometryEstimate(
            GeometryStatus.UNSUPPORTED, 0.0, None, None, IDENTITY_H, False, False,
            "degenerate",
        ),
        GeometryEstimate(
            GeometryStatus.UNSUPPORTED, 0.0, None, None, None, False, False, None
        ),
        GeometryEstimate(
            GeometryStatus.RELIABLE, 0.0, None, None, None, False, True, None
        ),
        GeometryEstimate(
            GeometryStatus.UNRELIABLE, 0.0, None, None, IDENTITY_H, False, False, None
        ),
        GeometryEstimate(
            GeometryStatus.UNRELIABLE, 0.0, None, None, None, True, True, "unexpected"
        ),
    ],
)
def test_other_inconsistent_known_geometry_states_block_before_rectification(
    monkeypatch, inconsistent,
) -> None:
    backend = GeometryBackend(inconsistent)
    _install_scores(monkeypatch, [0.0])
    monkeypatch.setattr(
        runtime, "rectify_attacked_rgb",
        lambda image, matrix: pytest.fail("inconsistent Geometry must never be rectified"),
    )
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "ERROR_FAIL_CLOSED" and not record.positive
    assert not record.method_complete and not record.recovered
    assert "geometry_runtime:" in record.operational_error


def test_post_score_error_is_operational_without_retry(monkeypatch) -> None:
    backend = GeometryBackend(_geometry())
    seen = _install_scores(monkeypatch, [0.0, RuntimeError("post scorer stopped")])
    record = _detect(Image.new("RGB", (512, 512)), backend)
    assert record.route == "ERROR_FAIL_CLOSED" and not record.positive
    assert record.recovered and not record.method_complete and len(seen) == 2
    assert "post scorer stopped" in record.operational_error


def test_production_boundary_rejects_arbitrary_backend_and_test_threshold(monkeypatch) -> None:
    with pytest.raises(TypeError, match="must be SyncSealTorchScript"):
        runtime.BlindProductionAssets(
            _content_assets(), WeightedJointAsset({}, b""), GeometryBackend(_geometry())
        )
    monkeypatch.setattr(runtime, "blind_weighted_scores", lambda *args: pytest.fail("must not score"))
    with pytest.raises(runtime.ThresholdUnavailableError, match="no frozen N_dev=256"):
        runtime.detect_watermark(Image.new("RGB", (512, 512)), KEY, _production_assets())
    with pytest.raises(runtime.ThresholdUnavailableError, match="rejects a test-only"):
        runtime.detect_watermark(
            Image.new("RGB", (512, 512)), KEY,
            _production_assets(build_test_only_threshold_asset(1.0)),
        )


def test_development_full_replay_reinvokes_scorer_and_geometry_before_write(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"geometry": 0}

    def detect(self, image):
        calls["geometry"] += 1
        return _geometry(h=None, legal=True)

    monkeypatch.setattr(SyncSealTorchScript, "detect_geometry", detect)
    seen = _install_scores(monkeypatch, [0.0] * 512)
    output = tmp_path / "threshold.json"
    assert runner.freeze_threshold_with_runtime(
        _roster(), KEY.encode(), _production_assets(),
        lambda ref: Image.new("RGB", (512, 512)), output,
        producer_exact="c" * 40,
    ) == output
    assert output.is_file() and len(seen) == calls["geometry"] == 512
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["replay_kind"].startswith("fresh_full_system")
    assert len(payload["full_system_replay_rows"]) == 256
    with pytest.raises(ValueError, match="detection-key identity"):
        runtime.detect_watermark(
            Image.new("RGB", (512, 512)), "different-detection-key",
            _production_assets(load_threshold_asset(output)),
        )
    with pytest.raises(FileExistsError):
        runner.freeze_threshold_with_runtime(
            _roster(), KEY.encode(), _production_assets(),
            lambda ref: Image.new("RGB", (512, 512)), output,
            producer_exact="c" * 40,
        )


def test_replay_time_operational_failure_and_false_positive_block_asset(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        SyncSealTorchScript, "detect_geometry", lambda self, image: _geometry(h=None, legal=True)
    )
    output = tmp_path / "operational.json"
    _install_scores(monkeypatch, [0.0] * 256 + [RuntimeError("replay scorer stopped")])
    with pytest.raises(runner.ThresholdFreezeBlocked, match="operational interruption") as blocked:
        runner.freeze_threshold_with_runtime(
            _roster(), KEY.encode(), _production_assets(),
            lambda ref: Image.new("RGB", (512, 512)), output,
            producer_exact="e" * 40,
        )
    assert len(blocked.value.calibration_rows) == len(blocked.value.replay_rows) == 256
    assert blocked.value.status == "OPERATIONAL_BLOCKED"
    assert not output.exists()

    output = tmp_path / "false-positive.json"
    _install_scores(monkeypatch, [0.0] * 256 + [1.0] + [0.0] * 255)
    with pytest.raises(runner.ThresholdFreezeBlocked, match="0/256") as blocked:
        runner.freeze_threshold_with_runtime(
            _roster(), KEY.encode(), _production_assets(),
            lambda ref: Image.new("RGB", (512, 512)), output,
            producer_exact="e" * 40,
        )
    assert len(blocked.value.replay_rows) == 256
    assert blocked.value.status == "METHOD_FAILED"
    assert not output.exists()


def test_embedding_order_is_content_then_strong_typed_final_rgb_syncseal_once(monkeypatch) -> None:
    calls = []

    class Content:
        def embed_content(self, request, key):
            calls.append(("content", request, key))
            return Image.new("RGB", (512, 512), "white")

    def sync_embed(self, image, multiplier):
        calls.append(("syncseal", image.mode, multiplier))
        return image

    monkeypatch.setattr(SyncSealTorchScript, "embed_final_rgb", sync_embed)
    output = runtime.embed_watermark(
        {"pipeline_request": "injected"}, KEY,
        runtime.BlindEmbeddingAssets(Content(), _syncseal(), 1.0),
    )
    assert output.mode == "RGB"
    assert [call[0] for call in calls] == ["content", "syncseal"]


def test_callback_validates_actual_route_and_retains_method_failure(monkeypatch, tmp_path: Path) -> None:
    image_paths = []
    cases = []
    coverage = (
        "direct_positive", "geometry_recovered_positive",
        "unwatermarked_geometry_negative", "direct_positive",
    )
    for index, label in enumerate(coverage):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (512, 512)).save(path)
        image_paths.append(path)
        cases.append({"case_id": f"c{index}", "coverage": label, "image_path": str(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases, "denominator": 4}), encoding="utf-8")
    key_path = tmp_path / "key.bin"
    key_path.write_bytes(KEY.encode())
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text("ignored", encoding="ascii")
    monkeypatch.setattr(runner, "load_threshold_asset", lambda path: build_test_only_threshold_asset(1.0))
    monkeypatch.setattr(runner, "_load_factory", lambda spec: lambda root: _production_assets())
    results = iter(
        (
            SimpleNamespace(route="DIRECT_POSITIVE", positive=True, recovered=False, method_complete=True, operational_error=None, pre=None, post=None),
            SimpleNamespace(route="GEOMETRY_RECOVERED", positive=False, recovered=True, method_complete=True, operational_error=None, pre=None, post=None),
            SimpleNamespace(route="GEOMETRY_RECOVERED", positive=False, recovered=True, method_complete=True, operational_error=None, pre=None, post=None),
            SimpleNamespace(route="DIRECT_POSITIVE", positive=True, recovered=False, method_complete=True, operational_error=None, pre=None, post=None),
        )
    )
    monkeypatch.setattr(runner, "detect_watermark", lambda image, key, assets: next(results))
    output = tmp_path / "result.json"
    _, status, mismatches = runner.run_callback(
        manifest, key_path, threshold_path, "fake:factory", output
    )
    assert status == "METHOD_FAILED" and mismatches == ("c1",) and output.is_file()
    stored = json.loads(output.read_text(encoding="ascii"))
    assert stored["status"] == "METHOD_FAILED" and stored["mismatched_case_ids"] == ["c1"]
    monkeypatch.setattr(
        runner, "run_callback", lambda *args: (output, "METHOD_FAILED", ("c1",))
    )
    assert runner.main([
        "callback", "--manifest", str(manifest), "--key-file", str(key_path),
        "--threshold", str(threshold_path), "--runtime-factory", "fake:factory",
        "--output", str(tmp_path / "other.json"),
    ]) == 2


def test_notebook_first_executable_cell_and_static_callback_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/blind_detection_v1_callback.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')\n",
    ]
    source = "".join("".join(cell["source"]) for cell in code_cells)
    assert "force_remount" not in source and "N_CALLBACK = 4" in source
    assert "--detach" in source and "torch.cuda.is_available()" in source
    assert "if 'RUNNER_CALLS' not in globals()" in source
    assert source.count("RUNNER_CALLS = 0") == 1 and "RUNNER_CALLS += 1" in source
    assert "METHOD_FAILED" in source and "OPERATIONAL_BLOCKED" in source
    runner_source = (root / "experiments/run_blind_detection_v1.py").read_text()
    assert runner_source.count("detect_watermark(current_rgb, detection_key, public_assets)") == 1
    assert 'output.open("xb")' in runner_source
