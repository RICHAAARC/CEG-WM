from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path
import subprocess
import symtable
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image

from cegwm.geometry_v7.contracts import GeometryEstimate, GeometryStatus, estimate_geometry
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from cegwm.method.blind_detection import (
    BLIND_DEV_DISJOINT_FROM,
    BlindCalibrationRow,
    BlindCalibrationRoster,
    BlindCalibrationUnit,
    BlindReplayRow,
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


def _unresolved_global_references(source: str, filename: str) -> list[tuple[str, str]]:
    table = symtable.symtable(source, filename, "exec")
    module_definitions = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    implicit_globals = {
        "__builtins__", "__cached__", "__file__", "__loader__",
        "__name__", "__package__", "__spec__",
    }
    unresolved = []

    def visit(current) -> None:
        unresolved.extend(
            (current.get_name(), symbol.get_name())
            for symbol in current.get_symbols()
            if symbol.is_referenced()
            and symbol.is_global()
            and symbol.get_name() not in module_definitions
            and symbol.get_name() not in vars(builtins)
            and symbol.get_name() not in implicit_globals
        )
        for child in current.get_children():
            visit(child)

    visit(table)
    return unresolved


def test_full_runner_has_no_unresolved_global_references() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments/run_blind_detection_v1.py"
    assert _unresolved_global_references(
        path.read_text(encoding="utf-8"), str(path)
    ) == []


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


def test_calibration_roster_freezes_logical_cross_product_and_retains_preflight_failure(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    roster, generation_units, summary = runner.load_roster_inputs(root)
    assert len(roster.units) == len(generation_units) == 256
    assert roster.disjoint_from == BLIND_DEV_DISJOINT_FROM
    assert summary["seeds"] == [2026101000, 2026101001, 2026101002, 2026101003]
    assert summary["geometry_v7_excluded_pair_count"] == 12
    assert set(summary["source_strata"].values()) == {32}
    assert len(summary["source_strata"]) == 8
    assert len({unit.calibration_unit.base_image_id for unit in generation_units}) == 256
    assert len({(unit.prompt, unit.seed) for unit in generation_units}) == 256

    monkeypatch.setattr(runner, "_verify_producer_checkout", lambda exact: None)
    monkeypatch.setattr(
        runner, "load_roster_inputs", lambda root: (_ for _ in ()).throw(
            ValueError("blind development logical image references must be unique")
        ),
    )
    monkeypatch.setenv(runner.ROOT_KEY_ENV, KEY)
    monkeypatch.setenv(runner.HF_TOKEN_ENV, "test-token")
    result_path = tmp_path / "preflight-result.json"
    _, candidate, status = runner.calibrate_and_record(
        tmp_path / "runtime", tmp_path / "candidate.json", result_path,
        producer_exact="d" * 40,
    )
    assert candidate is None and status == "OPERATIONAL_BLOCKED"
    assert "logical image references must be unique" in json.loads(
        result_path.read_text(encoding="ascii")
    )["error"]


def test_formal_calibration_retains_complete_method_failure_without_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    rows = tuple(
        BlindCalibrationRow(
            index, f"u-{index}", f"s-{index % 2}", 0.0, None,
            "NO_H", True, None,
        )
        for index in range(256)
    )
    replay = tuple(
        BlindReplayRow(
            index, f"u-{index}", f"s-{index % 2}",
            1.0 if index == 0 else 0.0, None,
            "DIRECT_POSITIVE" if index == 0 else "GEOMETRY_NO_H",
            index == 0, False, True, None,
        )
        for index in range(256)
    )
    monkeypatch.setattr(runner, "_verify_producer_checkout", lambda exact: None)
    monkeypatch.setattr(
        runner, "public_key_digest", lambda key: runner.CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    )
    monkeypatch.setattr(
        runner, "build_production_runtime",
        lambda root, config, hf_token, runtime_root: (object(), _production_assets()),
    )
    monkeypatch.setattr(
        runner, "generate_development_images",
        lambda units, pipeline, device: (
            {
                unit.calibration_unit.image_ref: Image.new("RGB", (512, 512))
                for unit in units
            },
            tuple(
                {
                    "error": None,
                    "roster_index": index,
                    "source_stratum": unit.calibration_unit.source_stratum,
                    "status": "GENERATED",
                    "unit_id": unit.calibration_unit.unit_id,
                }
                for index, unit in enumerate(units)
            ),
        ),
    )

    def fail(*args, **kwargs):
        raise runner.ThresholdFreezeBlocked(
            ValueError("full-system replay must produce exactly 0/256 empirical false positives"),
            rows,
            replay,
        )

    monkeypatch.setattr(runner, "evaluate_threshold_with_runtime", fail)
    monkeypatch.setenv(runner.ROOT_KEY_ENV, KEY)
    monkeypatch.setenv(runner.HF_TOKEN_ENV, "test-token")
    result_path = tmp_path / "calibration_result.json"
    threshold_path = tmp_path / "threshold.json"
    result, threshold, status = runner.calibrate_and_record(
        tmp_path / "runtime", threshold_path, result_path,
        producer_exact="f" * 40,
    )
    assert result == result_path and threshold is None and status == "METHOD_FAILED"
    assert result_path.is_file() and not threshold_path.exists()
    payload = json.loads(result_path.read_text(encoding="ascii"))
    assert payload["denominator"] == 256 and payload["science_denominator"] == 0
    assert len(payload["calibration_rows"]) == len(payload["fresh_replay_rows"]) == 256
    assert payload["calibration_rows"][0]["z_be_hex"] is not None
    assert payload["fresh_replay_false_positives"] == 1
    assert payload["candidate_tau_blind_be_hex"] is not None
    assert payload["frozen_tau_blind_be_hex"] is None
    assert payload["threshold_candidate_ready"] is False
    assert len(payload["generation_records"]) == 256
    with pytest.raises(FileExistsError, match="result is create-only"):
        runner.calibrate_and_record(
            tmp_path / "other-runtime", threshold_path, result_path,
            producer_exact="f" * 40,
        )


def test_formal_calibration_success_retains_rows_replay_and_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    rows = tuple(
        BlindCalibrationRow(
            index, f"u-{index}", f"s-{index % 2}", 0.0, None,
            "NO_H", True, None,
        )
        for index in range(256)
    )
    replay = tuple(
        BlindReplayRow(
            index, f"u-{index}", f"s-{index % 2}", 0.0, None,
            "GEOMETRY_NO_H", False, False, True, None,
        )
        for index in range(256)
    )
    asset = SimpleNamespace(
        payload={"tau_blind_be_hex": "0000000000000000"},
        json_bytes=b'{"threshold":"test-double"}',
    )
    monkeypatch.setattr(runner, "_verify_producer_checkout", lambda exact: None)
    monkeypatch.setattr(
        runner, "public_key_digest", lambda key: runner.CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    )
    monkeypatch.setattr(
        runner, "build_production_runtime",
        lambda root, config, hf_token, runtime_root: (object(), _production_assets()),
    )
    monkeypatch.setattr(
        runner, "generate_development_images",
        lambda units, pipeline, device: (
            {
                unit.calibration_unit.image_ref: Image.new("RGB", (512, 512))
                for unit in units
            },
            tuple(
                {
                    "error": None,
                    "roster_index": index,
                    "source_stratum": unit.calibration_unit.source_stratum,
                    "status": "GENERATED",
                    "unit_id": unit.calibration_unit.unit_id,
                }
                for index, unit in enumerate(units)
            ),
        ),
    )
    monkeypatch.setattr(
        runner, "evaluate_threshold_with_runtime",
        lambda *args, **kwargs: (rows, replay, asset),
    )
    monkeypatch.setenv(runner.ROOT_KEY_ENV, KEY)
    monkeypatch.setenv(runner.HF_TOKEN_ENV, "test-token")
    result_path = tmp_path / "calibration_result.json"
    threshold_path = tmp_path / "threshold.json"
    result, threshold, status = runner.calibrate_and_record(
        tmp_path / "runtime", threshold_path, result_path,
        producer_exact="e" * 40,
    )
    assert (result, threshold) == (result_path, threshold_path)
    assert status == "CALIBRATION_COMPLETE_THRESHOLD_CANDIDATE_READY"
    payload = json.loads(result_path.read_text(encoding="ascii"))
    assert payload["fresh_replay_zero_of_256"] is True
    assert payload["fresh_replay_false_positives"] == 0
    assert payload["candidate_tau_blind_be_hex"] == payload["frozen_tau_blind_be_hex"]
    assert payload["threshold_candidate_ready"] is True
    assert len(payload["generation_records"]) == 256
    assert "threshold_candidate_sha256" not in payload


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


def _fixed_n4_threshold(detection_key: bytes):
    rows = tuple(
        BlindCalibrationRow(
            index,
            f"u-{index}",
            f"s-{index % 2}",
            runner.CALLBACK_N4_TAU if index == 255 else 0.0,
            None,
            "NO_H",
            True,
            None,
        )
        for index in range(256)
    )
    replay = tuple(
        BlindReplayRow(
            index, f"u-{index}", f"s-{index % 2}", 0.0, None,
            "GEOMETRY_NO_H", False, False, True, None,
        )
        for index in range(256)
    )
    roster = BlindCalibrationRoster(
        tuple(
            BlindCalibrationUnit(f"u-{index}", f"s-{index % 2}", f"r-{index}", f"b-{index}")
            for index in range(256)
        )
    )
    return runner.build_threshold_asset(
        rows,
        roster,
        replay,
        producer_exact="a" * 40,
        calibration_key_digest=runner.public_key_digest(detection_key),
    )


def test_callback_n4_discovers_threshold_by_zip_content_semantics(tmp_path: Path) -> None:
    detection_key = runner.normalize_detection_key(KEY)
    threshold = _fixed_n4_threshold(detection_key)
    terminal = tmp_path / "opaque-terminal.zip"
    with zipfile.ZipFile(terminal, mode="x") as archive:
        archive.writestr("arbitrary-result-member", runner.stable_json_bytes({
            "denominator": 256,
            "frozen_tau_blind_be_hex": "3ff2201bf0021293",
            "status": "CALIBRATION_COMPLETE_THRESHOLD_CANDIDATE_READY",
            "threshold_candidate_ready": True,
        }))
        archive.writestr("arbitrary-threshold-member", threshold.json_bytes)
        archive.writestr("runner-output", "completed")
    discovered = runner.discover_callback_n4_threshold(tmp_path, detection_key)
    assert discovered.tau_blind == runner.CALLBACK_N4_TAU
    assert discovered.payload["tau_blind_be_hex"] == "3ff2201bf0021293"
    with pytest.raises(RuntimeError, match="exactly one semantically matching"):
        runner.discover_callback_n4_threshold(tmp_path, b"different-key")


def test_callback_n4_fixed_four_retains_mismatch_and_all_current_rgb(
    monkeypatch, tmp_path: Path
) -> None:
    config, units = runner.load_callback_n4_config(Path(__file__).resolve().parents[2])
    assert config["tau_blind_be_hex"] == "3ff2201bf0021293"
    assert tuple(unit.coverage for unit in units) == runner.CALLBACK_N4_COVERAGE
    assert len({unit.prompt for unit in units}) == len({unit.seed for unit in units}) == 4
    monkeypatch.setattr(runner, "_verify_producer_checkout", lambda exact: None)
    monkeypatch.setattr(
        runner, "public_key_digest", lambda key: runner.CONTENT_CHAIN_PUBLIC_KEY_DIGEST,
    )
    monkeypatch.setattr(
        runner,
        "discover_callback_n4_threshold",
        lambda root, key: build_test_only_threshold_asset(runner.CALLBACK_N4_TAU),
    )
    base_assets = _production_assets()
    object.__setattr__(base_assets.content_assets, "iss_assets", object())
    monkeypatch.setattr(
        runner,
        "build_production_runtime",
        lambda *args, **kwargs: (object(), base_assets),
    )
    monkeypatch.setattr(runner, "BlindEmbeddingAssets", lambda *args: object())
    prepared = []

    def prepare(unit, *args, **kwargs):
        prepared.append(unit.case_id)
        return Image.new("RGB", (512, 512), "white")

    monkeypatch.setattr(runner, "_prepare_callback_n4_current_rgb", prepare)
    results = iter((
        SimpleNamespace(route="DIRECT_POSITIVE", positive=True, recovered=False, method_complete=True, operational_error=None, pre=None, post=None),
        SimpleNamespace(route="GEOMETRY_RECOVERED", positive=False, recovered=True, method_complete=True, operational_error=None, pre=None, post=None),
        SimpleNamespace(route="GEOMETRY_FAIL_CLOSED", positive=False, recovered=False, method_complete=True, operational_error=None, pre=None, post=None),
        SimpleNamespace(route="GEOMETRY_RECOVERED", positive=False, recovered=True, method_complete=True, operational_error=None, pre=None, post=None),
    ))
    monkeypatch.setattr(
        runner, "detect_callback_n4_current_rgb", lambda *args: next(results)
    )
    monkeypatch.setenv(runner.ROOT_KEY_ENV, KEY)
    monkeypatch.setenv(runner.HF_TOKEN_ENV, "test-token")
    result_path = tmp_path / "result.json"
    _, status, mismatches = runner.run_callback_n4(
        tmp_path / "calibration-runs",
        tmp_path / "runtime",
        tmp_path / "current-rgb",
        result_path,
        producer_exact="e" * 40,
    )
    assert status == "METHOD_FAILED"
    assert mismatches == ("blind-callback-n4-02",)
    stored = json.loads(result_path.read_text(encoding="ascii"))
    assert stored["denominator"] == 4 and stored["science_denominator"] == 0
    assert len(stored["records"]) == len(prepared) == 4
    assert len(tuple((tmp_path / "current-rgb").glob("*.png"))) == 4
    assert stored["records"][2]["route"] == "GEOMETRY_FAIL_CLOSED"
    assert stored["records"][2]["case_id"] not in stored["mismatched_case_ids"]
    assert stored["automatic_retries"] == 0
    assert runner.ROOT_KEY_ENV not in runner.os.environ
    assert runner.HF_TOKEN_ENV not in runner.os.environ

    monkeypatch.setattr(
        runner,
        "load_callback_n4_config",
        lambda root: (_ for _ in ()).throw(ValueError("frozen roster stopped")),
    )
    monkeypatch.setenv(runner.ROOT_KEY_ENV, KEY)
    monkeypatch.setenv(runner.HF_TOKEN_ENV, "test-token")
    blocked_path = tmp_path / "blocked.json"
    _, blocked_status, _ = runner.run_callback_n4(
        tmp_path / "other-calibration-runs",
        tmp_path / "other-runtime",
        tmp_path / "other-current-rgb",
        blocked_path,
        producer_exact="e" * 40,
    )
    blocked = json.loads(blocked_path.read_text(encoding="ascii"))
    assert blocked_status == "OPERATIONAL_BLOCKED"
    assert len(blocked["records"]) == 4
    assert all("frozen roster stopped" in row["operational_error"] for row in blocked["records"])


def test_callback_n4_detection_handoff_is_closure_free_and_single_image_only() -> None:
    signature = tuple(inspect.signature(runner.detect_callback_n4_current_rgb).parameters)
    assert signature == ("current_rgb", "detection_key", "public_assets")
    root = Path(__file__).resolve().parents[2]
    source = (root / "experiments/run_blind_detection_v1.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "detect_callback_n4_current_rgb"
    )
    identifiers = {
        node.id for node in ast.walk(helper) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(helper) if isinstance(node, ast.Attribute)
    }
    assert not identifiers.intersection({
        "original", "u", "g", "paired_null", "prompt", "seed", "embed_record",
        "private_latent", "attack", "truth", "outcome", "stored_h",
    })
    assert source.count(
        "detect_watermark(current_rgb, detection_key, public_assets)"
    ) == 1
    assert sum(
        isinstance(node, (ast.FunctionDef, ast.Lambda)) for node in ast.walk(helper)
    ) == 1
    assert "primary_null = pair.primary_null" in source and "del primary_null" in source
    handoff = source.index("record = detect_callback_n4_current_rgb(")
    for released in (
        "del units", "del config", "del pipeline", "del embedding_assets",
        "del base_assets", "del runtime_config", "del threshold",
    ):
        assert source.index(released) < handoff
    assert 'sub.add_parser("callback-n4")' in source
    assert "--key-file" not in source and "--runtime-factory" not in source
    args = runner._parser().parse_args([
        "callback-n4",
        "--producer-exact", "f" * 40,
        "--calibration-runs-root", "/tmp/calibration-runs",
        "--runtime-root", "/tmp/runtime",
        "--current-rgb-output-dir", "/tmp/current-rgb",
        "--result-output", "/tmp/result.json",
    ])
    assert args.command == "callback-n4"


def test_callback_notebook_is_bound_to_p6_and_retains_one_terminal_zip() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/blind_detection_v1_callback.ipynb").read_text()
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')\n",
    ]
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    source = "".join("".join(cell["source"]) for cell in code_cells)
    producer_exact = "0ff9b054c7caeaf487c3488fbcd04164d4db2ad3"
    assert source.count(producer_exact) == 1
    assert f"PRODUCER_EXACT = '{producer_exact}'" in source
    assert "checkout', '--detach', PRODUCER_EXACT" in source
    assert source.count("git_value('rev-parse', 'HEAD')") == 2
    assert source.count("git_value('branch', '--show-current')") == 2
    assert source.count("git_value('status', '--porcelain=v1')") == 2
    assert "sys.executable, '-m', 'pip', 'install'" in source
    assert "torch.cuda.is_available()" in source
    assert "sys.executable, '-m', 'experiments.run_blind_detection_v1', 'callback-n4'" in source
    assert "cwd=CHECKOUT, env=runner_env, check=False" in source
    assert source.count("userdata.get(") == 2
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "secret_markers = ('TOKEN', 'KEY', 'SECRET', 'PASSWORD', 'CREDENTIAL')" in source
    assert "runner_env.pop('CEG_WM_ROOT_KEY', None)" in source
    assert "runner_env.pop('HF_TOKEN', None)" in source
    assert "BLIND_CALLBACK_RUNNER_CALLS += 1" in source
    assert source.count("'callback-n4'") == 1
    assert source.count("    write_terminal_zip()\n") == 1
    assert source.count("zipfile.ZipFile(TERMINAL_ZIP, mode='x'") == 1
    assert "/content/drive/MyDrive/CEG-WM/BlindDetection-V1/calibration-runs" in source
    assert "/content/drive/MyDrive/CEG-WM/BlindDetection-V1/callback-runs" in source
    assert "PUBLIC_N4_CONFIG" in source and "CURRENT_RGB_DIR.glob('current_rgb_*.png')" in source
    assert "if checkout_verified and PUBLIC_N4_CONFIG.is_file():" in source
    assert "'records': []" in source and "'status': 'OPERATIONAL_BLOCKED'" in source
    for forbidden in (
        "force_remount", "CEGWM_BLIND_RUNTIME_FACTORY",
        "CEGWM_BLIND_DETECTION_EXACT", "--key-file", "callback-input",
        "manifest", "sha256", "hashlib", ".zip.sha", "receipt", "signature",
        "LOCAL_THRESHOLD", "blind_detection_v1_thresholds.json",
        "primary_null", "stored_h", "original_image",
    ):
        assert forbidden not in source
    final_source = "".join(code_cells[-1]["source"])
    assert "zipfile.ZipFile(TERMINAL_ZIP, mode='r')" in final_source
    assert "archive.namelist()" in final_source
    assert "public_result.get('status')" in final_source
    assert "public_result.get('denominator')" in final_source
    assert "subprocess" not in final_source and "archive.write" not in final_source

    runner_source = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{producer_exact}:experiments/run_blind_detection_v1.py"],
        text=True,
    )
    config_payload = json.loads(subprocess.check_output(
        [
            "git", "-C", str(root), "show",
            f"{producer_exact}:configs/blind_detection/blind_detection_v1_callback_n4.json",
        ],
        text=True,
    ))
    for required in (
        "def run_callback_n4(", "def discover_callback_n4_threshold(",
        "def detect_callback_n4_current_rgb(",
        "return detect_watermark(current_rgb, detection_key, public_assets)",
        'sub.add_parser("callback-n4")', "run_content_iss_evaluation_pair(",
        "generated = embed_watermark(", "generated = run_sd35_plain(",
        "render_r1a_attack(", "del units", "del embedding_assets",
    ):
        assert required in runner_source
    assert "--key-file" not in runner_source and "--runtime-factory" not in runner_source
    assert config_payload["denominator"] == 4
    assert config_payload["science_denominator"] == 0
    assert config_payload["tau_blind_be_hex"] == "3ff2201bf0021293"
    assert config_payload["attack_condition_id"] == "core_rotation_pos15"
    assert [case["coverage"] for case in config_payload["cases"]] == list(
        runner.CALLBACK_N4_COVERAGE
    )


def test_calibration_notebook_is_exact_bound_and_calls_only_formal_n256_runner_once() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/blind_detection_v1_calibration.ipynb").read_text()
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')\n",
    ]
    source = "".join("".join(cell["source"]) for cell in code_cells)
    assert "force_remount" not in source
    producer_exact = "920561bd264075628d52dcb70deef842284b6a75"
    assert source.count(producer_exact) == 1
    assert f"PRODUCER_EXACT = '{producer_exact}'" in source
    assert "checkout', '--detach', PRODUCER_EXACT" in source
    assert "status', '--porcelain=v1'" in source
    assert "r.load_roster_inputs(r.REPO_ROOT); r.load_runtime_config(r.REPO_ROOT)" in source
    assert source.count("userdata.get(") == 2
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "BLIND_CALIBRATION_RUNNER_CALLS += 1" in source
    assert source.count("'calibrate-and-freeze'") == 1
    assert "sys.executable, '-m', 'experiments.run_blind_detection_v1'" in source
    assert "str(CHECKOUT / 'experiments/run_blind_detection_v1.py')" not in source
    assert "'--runtime-root', str(RUNTIME_ROOT)" in source
    assert "runtime_factory" not in source
    assert "'--result-output', str(LOCAL_RESULT)" in source
    assert "runner.stdout.txt" in source and "runner.stderr.txt" in source
    assert "calibration_rows.json" in source and "fresh_replay_rows.json" in source
    assert "fresh_replay_zero_of_256" in source
    assert source.count("zipfile.ZipFile(TERMINAL_ZIP, mode='x'") == 1
    assert "members.append(LOCAL_THRESHOLD)" in source
    assert "if not terminal_published and BLIND_TERMINAL_ZIP_WRITES == 0" in source
    for forbidden in (
        "sha256", "hashlib", ".zip.sha256", "artifact_manifest", "receipt",
        "signature", "INPUT_ROOT", "KEY_FILE", "CONFIG_FILE", "pending.unlink()",
        ".replace(",
    ):
        assert forbidden not in source
    assert "blind_detection_v1_callback.ipynb" not in source
    assert "N_CALLBACK" not in source and "--manifest" not in source
    for forbidden in ("paired_null", "stored_h", "proxy_rgb", "truth_label"):
        assert forbidden not in source.lower()
    final_source = "".join(code_cells[-1]["source"])
    assert "mode='r'" in final_source and "archive.read('calibration_result.json')" in final_source
    assert "subprocess" not in final_source and "archive.write" not in final_source
    runner_source = subprocess.check_output(
        [
            "git", "-C", str(root), "show",
            f"{producer_exact}:experiments/run_blind_detection_v1.py",
        ],
        text=True,
    )
    threshold_source = subprocess.check_output(
        [
            "git", "-C", str(root), "show",
            f"{producer_exact}:src/cegwm/method/blind_detection.py",
        ],
        text=True,
    )
    for required in (
        "content_unweighted_engine._load_pipeline_and_assets(",
        "load_whitening_asset_semantic(repo_root)",
        "FrozenContentWhiteningLFPublicAssets(",
        "load_iss_asset_semantic(repo_root)",
        "ContentISSEvaluationAssets(",
        "ContentCalibrationAssets(evaluation_assets)",
        "load_weighted_asset_semantic(repo_root)",
        "download_official_syncseal_torchscript(checkpoint)",
        "if not checkpoint.is_file()",
        "SyncSealTorchScript.from_file(checkpoint",
    ):
        assert required in runner_source
    for forbidden in (
        "content_iss_engine", "content_whitening_engine",
        "load_frozen_content_iss_asset", "load_frozen_content_whitening_asset",
        "load_calibration_asset", "sha256", "hashlib", "hexdigest", "getsize",
        "st_size", ".stat()",
    ):
        assert forbidden not in runner_source
    assert "stable_json_bytes(payload) != raw" not in threshold_source
    assert "json_bytes != stable_json_bytes" not in threshold_source
    assert "public_key_digest" in runner_source
    assert "calibration_key_digest" in threshold_source
    runner_tree = ast.parse(runner_source, filename=f"{producer_exact}:runner")
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "cegwm.method.blind_detection"
        and any(alias.name == "decode_binary64" for alias in node.names)
        for node in runner_tree.body
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_binary64"
        for node in ast.walk(runner_tree)
    ) == 4
    assert _unresolved_global_references(
        runner_source, f"{producer_exact}:experiments/run_blind_detection_v1.py"
    ) == []
