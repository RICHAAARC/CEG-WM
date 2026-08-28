from __future__ import annotations

import hashlib
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

import cegwm.geometry_v3.confirmation as P1
from cegwm.geometry_v3.active_writer import P0_CONFIGS, P0_INFERENCE_STEPS
from cegwm.geometry_v3.operational import ObservationScores


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "experiments" / "run_geometry_v3_qk_active_writer_p1_confirmation.py"
    spec = importlib.util.spec_from_file_location("geometry_v3_p1_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for config in P0_CONFIGS:
        intended_margin = 0.05 if config.config_id == P1.P1_CONFIG_ID else -0.01
        scores = {
            "correct_key_anchor": 0.25,
            "wrong_key_anchor": 0.20 if intended_margin > 0.0 else 0.26,
            "no_writer": 0.19 if intended_margin > 0.0 else 0.25,
        }
        margin = scores["correct_key_anchor"] - max(
            scores["wrong_key_anchor"], scores["no_writer"]
        )
        for attack in ("identity", "rotate90", "similarity", "crop_rescale"):
            for kind in P1.P1_KIND_IDS:
                for control in P1.P1_CONTROL_IDS:
                    records.append({
                        "config_id": config.config_id,
                        "attack_id": attack,
                        "feature_kind": kind,
                        "control": control,
                        "status": "calculated",
                        "error_class": None,
                        "score": scores[control],
                        "margin": margin,
                    })
    return records


def _source_fixture(root: Path) -> Path:
    root.mkdir()
    records = _source_records()
    summaries = []
    for config in P0_CONFIGS:
        selected = config.config_id == P1.P1_CONFIG_ID
        fixture_margin = 0.25 - (0.20 if selected else 0.26)
        summaries.append({
            "config_id": config.config_id,
            "block_index": config.block_index,
            "relative_rms_budget": config.relative_rms_budget,
            "calculated_unit_count": 24,
            "q_four_attack_equal_weight_median_margin": fixture_margin,
            "k_four_attack_equal_weight_median_margin": fixture_margin,
            "eligible": selected,
        })
    receipt = {
        "run_id": P1.SOURCE_RUN_ID,
        "protocol": P1.SOURCE_PROTOCOL,
        "execution_exact": P1.SOURCE_EXECUTION_EXACT,
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "prompt_id": "geometry-v3-p0-public-prompt-01",
        "plan_digest": P1.SOURCE_PLAN_DIGEST,
        "roster_digest": P1.SOURCE_ROSTER_DIGEST,
        "status": P1.SOURCE_STATUS,
        "artifact_status": "complete",
        "fixed_unit_count": 144,
        "calculated_unit_count": 144,
        "failed_unit_count": 0,
        "selected_config_id": P1.P1_CONFIG_ID,
        "operational_failure_point": None,
        "science_denominator": 0,
        "config_summaries": summaries,
        "interference": [
            {"config_id": config.config_id, "rgb_mse": 1.0, "rgb_psnr_db": 48.0,
             "content_detector_hook_status": "not_invoked_record_only"}
            for config in P0_CONFIGS
        ],
        "writer_measurements": [
            {"config_id": config.config_id, "feature_kind": kind,
             "module_path": f"{config.layer_path}.to_{kind}",
             "relative_rms_budget": config.relative_rms_budget,
             "actual_relative_rms": config.relative_rms_budget * 0.99,
             "call_count": 1, "writer_step_index": 18}
            for config in P0_CONFIGS for kind in P1.P1_KIND_IDS
        ],
    }
    terminal = {
        "run_id": P1.SOURCE_RUN_ID, "status": P1.SOURCE_STATUS,
        "artifact_status": "complete", "selected_config_id": P1.P1_CONFIG_ID,
        "science_denominator": 0,
    }
    metrics = b"".join(_json_bytes(record) + b"\n" for record in records)
    payloads = {
        "metrics.jsonl": metrics,
        "receipt.json": _json_bytes(receipt),
        "terminal.json": _json_bytes(terminal),
    }
    manifest = {
        "run_id": P1.SOURCE_RUN_ID,
        "protocol": P1.SOURCE_PROTOCOL,
        "execution_exact": P1.SOURCE_EXECUTION_EXACT,
        "plan_digest": P1.SOURCE_PLAN_DIGEST,
        "roster_digest": P1.SOURCE_ROSTER_DIGEST,
        "files": [
            {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(payloads.items())
        ],
        "total_payload_bytes": sum(len(data) for data in payloads.values()),
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    return root


def _rewrite_source(root: Path, filename: str, mutation) -> None:
    value = json.loads((root / filename).read_text(encoding="utf-8"))
    mutation(value)
    (root / filename).write_bytes(_json_bytes(value))
    if filename != "manifest.json":
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            if entry["name"] == filename:
                payload = (root / filename).read_bytes()
                entry["bytes"] = len(payload)
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["total_payload_bytes"] = sum(
            (root / entry["name"]).stat().st_size for entry in manifest["files"]
        )
        (root / "manifest.json").write_bytes(_json_bytes(manifest))


def _rewrite_metrics(root: Path, mutation) -> None:
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_bytes().splitlines()]
    mutation(records)
    (root / "metrics.jsonl").write_bytes(
        b"".join(_json_bytes(record) + b"\n" for record in records)
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == "metrics.jsonl":
            payload = (root / "metrics.jsonl").read_bytes()
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["total_payload_bytes"] = sum(
        (root / entry["name"]).stat().st_size for entry in manifest["files"]
    )
    (root / "manifest.json").write_bytes(_json_bytes(manifest))


def _set_group_margin(
    records: list[dict[str, object]], config_id: str,
    attacks: tuple[str, ...], kinds: tuple[str, ...], margin: float,
) -> None:
    for record in records:
        if (
            record["config_id"] == config_id
            and record["attack_id"] in attacks
            and record["feature_kind"] in kinds
        ):
            control = record["control"]
            record["score"] = (
                0.25 if control == "correct_key_anchor"
                else 0.25 - margin if control == "wrong_key_anchor"
                else 0.24 - margin
            )
    for record in records:
        if (
            record["config_id"] == config_id
            and record["attack_id"] in attacks
            and record["feature_kind"] in kinds
        ):
            group = [
                item for item in records
                if item["config_id"] == config_id
                and item["attack_id"] == record["attack_id"]
                and item["feature_kind"] == record["feature_kind"]
            ]
            by_control = {item["control"]: item for item in group}
            actual_margin = float(by_control["correct_key_anchor"]["score"]) - max(
                float(by_control["wrong_key_anchor"]["score"]),
                float(by_control["no_writer"]["score"]),
            )
            record["margin"] = actual_margin


def _plan(tmp_path: Path, source: Path) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "expected_exact": "a" * 40,
        "execution_exact": "a" * 40,
        "source_directory": str(source),
        "output_directory": "/content/drive/MyDrive/CEG-WM/Geometry-V3/P1/Geometry-V3-P1-test",
    }), encoding="utf-8")
    return plan


def _run_main(plan: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, dict[str, object]]:
    monkeypatch.setenv(RUNNER.TOKEN_ENV, "token")
    monkeypatch.setenv(RUNNER.KEY_ENV, "key")
    monkeypatch.setattr(RUNNER, "_git_exact", lambda expected: expected)
    read_fd, write_fd = os.pipe()
    rc = RUNNER._main(["--plan", str(plan), "--control-fd", str(write_fd)], preloader=lambda *_: pytest.fail("model preloader reached"))
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    return rc, json.loads(payload)


@pytest.mark.integration
def test_protocol_freezes_independent_inputs_and_24_unit_roster() -> None:
    plan = P1.public_plan()
    assert plan["protocol"] == P1.P1_PROTOCOL_ID
    assert plan["prompt_id"] == "geometry-v3-p1-public-prompt-01"
    assert "lighthouse" not in P1.P1_PROMPT_TEXT.lower()
    assert P1.P1_GENERATION_SEED == 173 != 73
    assert P1.P1_OBSERVATION_NOISE_SEED == 19073 != 9073
    assert P1.P1_OBSERVATION_TIMESTEP == 500
    assert P1.fixed_config().config_id == "block12-qk-rms0p0025"
    assert len(P1.fixed_roster()) == 24
    assert set(P1.P1_ATTACK_IDS) == {"identity", "rotate270", "similarity", "crop_rescale"}


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
    total = float(window.sum())
    assert total > 0.0
    return float(np.sum((xx + 0.5) * window) / total), float(np.sum((yy + 0.5) * window) / total)


def _independent_h(attack_id: str) -> np.ndarray:
    if attack_id == "identity":
        return np.eye(3)
    if attack_id == "rotate270":
        return np.array(((0.0, -1.0, 512.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    if attack_id == "similarity":
        angle, scale = math.radians(-11.0), 0.89
        linear = scale * np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
        centre = np.array((256.0, 256.0))
        offset = centre + np.array((-17.0, 9.0)) - linear @ centre
        return np.array(((linear[0, 0], linear[0, 1], offset[0]),
                         (linear[1, 0], linear[1, 1], offset[1]), (0.0, 0.0, 1.0)))
    left, top, right, bottom = 46, 28, 470, 482
    sx, sy = 512.0 / (right - left), 512.0 / (bottom - top)
    return np.array(((sx, 0.0, -left * sx), (0.0, sy, -top * sy), (0.0, 0.0, 1.0)))


@pytest.mark.integration
@pytest.mark.parametrize("attack_id", P1.P1_ATTACK_IDS)
def test_actual_pillow_attack_correspondence_matches_independent_h(attack_id: str) -> None:
    expected_h = _independent_h(attack_id)
    for source_x, source_y in ((176.25, 188.75), (331.5, 207.25), (242.75, 318.5)):
        attacked = P1.apply_attack(_gaussian_marker(source_x, source_y), attack_id)
        assert np.asarray(attacked.homography) == pytest.approx(expected_h, abs=1e-12)
        mapped = expected_h @ np.array((source_x, source_y, 1.0))
        expected_x, expected_y = mapped[0] / mapped[2], mapped[1] / mapped[2]
        observed_x, observed_y = _centroid(attacked.image, expected_x, expected_y)
        tolerance = 0.12 if attack_id in {"similarity", "crop_rescale"} else 0.04
        assert abs(observed_x - expected_x) <= tolerance
        assert abs(observed_y - expected_y) <= tolerance


@pytest.mark.integration
def test_valid_p0_source_fixture_is_fully_cross_bound(tmp_path: Path) -> None:
    identity = P1.validate_p0_source(_source_fixture(tmp_path / "source"))
    assert identity == P1.validate_p0_source_identity(identity)
    assert identity["selected_config_id"] == P1.P1_CONFIG_ID
    assert identity["calculated_unit_count"] == 144
    assert identity["failed_unit_count"] == 0


@pytest.mark.integration
def test_valid_source_real_main_reaches_execution_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_fixture(tmp_path / "source")
    plan = _plan(tmp_path, source)
    monkeypatch.setenv(RUNNER.TOKEN_ENV, "token")
    monkeypatch.setenv(RUNNER.KEY_ENV, "key")
    monkeypatch.setattr(RUNNER, "_git_exact", lambda expected: expected)
    seen: list[object] = []

    def execute(plan_value, *, geometry_key, hf_token, source_identity, preloader):
        seen.extend((plan_value, geometry_key, hf_token, source_identity, preloader))
        return {
            "run_id": "run", "status": P1.P1_STATUS_UNRESOLVED,
            "artifact_status": "complete", "fixed_config_id": P1.P1_CONFIG_ID,
            "science_denominator": 0,
        }

    monkeypatch.setattr(RUNNER, "execute_plan", execute)
    read_fd, write_fd = os.pipe()
    sentinel = object()
    rc = RUNNER._main(["--plan", str(plan), "--control-fd", str(write_fd)], preloader=sentinel)
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    control = json.loads(payload)
    assert rc == 0 and control["p1_status"] == P1.P1_STATUS_UNRESOLVED
    assert seen[1:3] == ["key", "token"]
    assert seen[3]["run_id"] == P1.SOURCE_RUN_ID and seen[4] is sentinel
    assert RUNNER.TOKEN_ENV not in os.environ and RUNNER.KEY_ENV not in os.environ


@pytest.mark.integration
def test_real_main_rejects_incomplete_143_metric_roster_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_fixture(tmp_path / "source")
    lines = (source / "metrics.jsonl").read_bytes().splitlines()
    (source / "metrics.jsonl").write_bytes(b"\n".join(lines[:-1]) + b"\n")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == "metrics.jsonl":
            payload = (source / "metrics.jsonl").read_bytes()
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["total_payload_bytes"] = sum(
        (source / entry["name"]).stat().st_size for entry in manifest["files"]
    )
    (source / "manifest.json").write_bytes(_json_bytes(manifest))
    rc, control = _run_main(_plan(tmp_path, source), monkeypatch)
    assert rc == 1 and control["failure_point"] == "source_validation"


@pytest.mark.integration
@pytest.mark.parametrize("case", ("margin", "median", "eligibility", "winner"))
def test_real_main_replays_p0_statistics_and_rejects_tampered_metrics_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    source = _source_fixture(tmp_path / "source")

    def mutate(records: list[dict[str, object]]) -> None:
        if case == "margin":
            records[0]["margin"] = 999.0
        elif case == "median":
            _set_group_margin(
                records, P1.P1_CONFIG_ID, ("identity", "rotate90"), ("q",), -0.4
            )
        elif case == "eligibility":
            _set_group_margin(records, P1.P1_CONFIG_ID, tuple(P1.P1_ATTACK_IDS), ("k",), -0.2)
        else:
            _set_group_margin(
                records, P0_CONFIGS[0].config_id, tuple(P1.P1_ATTACK_IDS), tuple(P1.P1_KIND_IDS), 0.2
            )
            _set_group_margin(
                records, P1.P1_CONFIG_ID, tuple(P1.P1_ATTACK_IDS), tuple(P1.P1_KIND_IDS), -0.2
            )

    _rewrite_metrics(source, mutate)
    rc, control = _run_main(_plan(tmp_path, source), monkeypatch)
    assert rc == 1
    assert control == {
        "status": "failure", "failure_point": "source_validation",
        "error_class": "validation_error", "science_denominator": 0,
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        ("receipt.json", lambda value: value.__setitem__("run_id", "wrong")),
        ("receipt.json", lambda value: value.__setitem__("protocol", "wrong")),
        ("receipt.json", lambda value: value.__setitem__("execution_exact", "0" * 40)),
        ("receipt.json", lambda value: value.__setitem__("plan_digest", "0" * 64)),
        ("receipt.json", lambda value: value.__setitem__("roster_digest", "0" * 64)),
        ("receipt.json", lambda value: value.__setitem__("artifact_status", "partial")),
        ("receipt.json", lambda value: value.__setitem__("fixed_unit_count", 143)),
        ("receipt.json", lambda value: value.__setitem__("calculated_unit_count", 143)),
        ("receipt.json", lambda value: value.__setitem__("failed_unit_count", 1)),
        ("receipt.json", lambda value: value.__setitem__("status", "P0_UNRESOLVED")),
        ("receipt.json", lambda value: value.__setitem__("selected_config_id", "block4-qk-rms0p0025")),
        ("manifest.json", lambda value: value.__setitem__("run_id", "wrong")),
        ("manifest.json", lambda value: value.__setitem__("protocol", "wrong")),
        ("manifest.json", lambda value: value.__setitem__("execution_exact", "0" * 40)),
        ("manifest.json", lambda value: value.__setitem__("plan_digest", "0" * 64)),
        ("manifest.json", lambda value: value.__setitem__("roster_digest", "0" * 64)),
        ("terminal.json", lambda value: value.__setitem__("run_id", "wrong")),
        ("terminal.json", lambda value: value.__setitem__("status", "P0_UNRESOLVED")),
        ("terminal.json", lambda value: value.__setitem__("selected_config_id", None)),
        ("terminal.json", lambda value: value.__setitem__("science_denominator", 1)),
    ),
)
def test_real_main_rejects_cross_binding_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, mutation,
) -> None:
    source = _source_fixture(tmp_path / "source")
    _rewrite_source(source, filename, mutation)
    rc, control = _run_main(_plan(tmp_path, source), monkeypatch)
    assert rc == 1
    assert control == {
        "status": "failure", "failure_point": "source_validation",
        "error_class": "validation_error", "science_denominator": 0,
    }


@pytest.mark.integration
@pytest.mark.parametrize("leak", ({"raw_qk": "x"}, {"note": "HF token material"}, {"note": "C:\\private\\source"}))
def test_real_main_rejects_public_leaks_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leak: dict[str, str],
) -> None:
    source = _source_fixture(tmp_path / "source")
    _rewrite_source(source, "terminal.json", lambda value: value.update(leak))
    rc, control = _run_main(_plan(tmp_path, source), monkeypatch)
    assert rc == 1 and control["failure_point"] == "source_validation"


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8, bias=False)
        self.to_k = torch.nn.Linear(8, 8, bias=False)
        torch.nn.init.eye_(self.to_q.weight)
        torch.nn.init.eye_(self.to_k.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return (self.to_q(hidden) + self.to_k(hidden)) * 0.5


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden)


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(_Block() for _ in range(21))
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return hidden


class _WriterPipeline:
    def __init__(self) -> None:
        self.transformer = _Transformer()

    def __call__(self, *, callback_on_step_end, callback_on_step_end_tensor_inputs, **kwargs):
        assert kwargs["prompt"] == P1.P1_PROMPT_TEXT
        assert kwargs["num_inference_steps"] == P0_INFERENCE_STEPS
        assert callback_on_step_end_tensor_inputs == ["latents"]
        hidden = torch.linspace(-1.0, 1.0, 16 * 8).reshape(1, 16, 8)
        state = {"latents": torch.zeros((1, 4, 4, 4))}
        for step in range(P0_INFERENCE_STEPS):
            hidden = self.transformer(hidden)
            state = callback_on_step_end(self, step, torch.tensor(step), state)
        return SimpleNamespace(images=[Image.new("RGB", (512, 512), (12, 34, 56))])


@pytest.mark.integration
def test_real_production_writer_hooks_complete_once_on_fake_pipeline() -> None:
    pipeline = _WriterPipeline()
    anchor = P1.derive_canonical_relation_anchor("geometry-key-0001", point_count=16)
    generated = P1.generate_writer_config(pipeline, P1.fixed_config(), anchor)
    assert generated.image.mode == "RGB"
    assert {item.feature_kind for item in generated.measurements} == {"q", "k"}
    assert all(item.call_count == 1 and item.writer_step_index == 18 for item in generated.measurements)
    assert all(0.0 < item.actual_relative_rms <= 0.0025 * 1.0002 for item in generated.measurements)


def _calculated_records(q_margins: tuple[float, ...], k_margins: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    records = []
    for attack, kind, control in P1.fixed_roster():
        index = P1.P1_ATTACK_IDS.index(attack)
        margin = q_margins[index] if kind == "q" else k_margins[index]
        records.append({
            "config_id": P1.P1_CONFIG_ID, "attack_id": attack,
            "feature_kind": kind, "control": control, "status": "calculated",
            "error_class": None, "score": 0.2, "margin": margin,
        })
    return tuple(records)


@pytest.mark.integration
def test_confirmation_uses_only_strict_q_and_k_four_attack_medians() -> None:
    status, q_value, k_value, audits = P1.confirm_active_anchor(
        _calculated_records((-1.0, 0.1, 0.2, 0.3), (-0.1, 0.2, 0.3, 0.4))
    )
    assert status == P1.P1_STATUS_CONFIRMED
    assert q_value == pytest.approx(0.15) and k_value == pytest.approx(0.25)
    assert len(audits) == 8 and any(item["margin"] < 0.0 for item in audits)
    assert P1.confirm_active_anchor(
        _calculated_records((-0.2, 0.0, 0.0, 0.2), (0.1, 0.2, 0.3, 0.4))
    )[0] == P1.P1_STATUS_UNRESOLVED


@pytest.mark.integration
def test_observation_failure_retains_fixed_24_units(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (512, 512), (7, 11, 13))
    monkeypatch.setattr(P1, "generate_no_writer", lambda pipeline: image)
    monkeypatch.setattr(P1, "generate_writer_config", lambda pipeline, config, anchor: P1.GeneratedConfig(image, ()))
    monkeypatch.setattr(P1, "observe_fresh_attacked_rgb", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private")))
    result = P1.run_p1(object(), "geometry-key-0001")
    assert result.status == P1.P1_STATUS_STOPPED
    assert len(result.records) == 24
    assert sum(record["status"] == "failed" for record in result.records) == 24
    assert {record["error_class"] for record in result.records} == {"runtime_error"}


@pytest.mark.integration
def test_bounded_artifact_contains_public_derived_data_only(tmp_path: Path) -> None:
    records = _calculated_records((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))
    status, q_value, k_value, audits = P1.confirm_active_anchor(records)
    result = P1.P1ExecutionResult(status, records, q_value, k_value, audits, (), (), None)
    source = P1.validate_p0_source_identity({
        "run_id": P1.SOURCE_RUN_ID, "protocol": P1.SOURCE_PROTOCOL,
        "execution_exact": P1.SOURCE_EXECUTION_EXACT, "plan_digest": P1.SOURCE_PLAN_DIGEST,
        "roster_digest": P1.SOURCE_ROSTER_DIGEST, "status": P1.SOURCE_STATUS,
        "artifact_status": "complete", "fixed_unit_count": 144,
        "calculated_unit_count": 144, "failed_unit_count": 0,
        "selected_config_id": P1.P1_CONFIG_ID, "science_denominator": 0,
    })
    root = tmp_path / "p1"
    control = P1.package_p1_artifacts(root, exact="b" * 40, source_identity=source, result=result)
    assert control["status"] == P1.P1_STATUS_CONFIRMED
    assert {path.name for path in root.iterdir()} == {"receipt.json", "manifest.json", "terminal.json", "metrics.jsonl"}
    payload = b"".join(path.read_bytes() for path in root.iterdir())
    assert len(payload) < P1.P1_ARTIFACT_MAX_BYTES
    lowered = payload.lower()
    for forbidden in (b"geometry-key", b"raw_qk", b"prompt_text", b"latent", b"image_bytes", b"hf_token", b"model_weights"):
        assert forbidden not in lowered
