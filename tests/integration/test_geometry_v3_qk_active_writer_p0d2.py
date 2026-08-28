from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


def _load_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "run_geometry_v3_qk_active_writer_p0d2.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_v3_p0d2_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


class _FixedProjection(torch.nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "no_correction":
            return torch.full_like(hidden, 0.5, dtype=torch.float32)
        if self.kind == "correction_accepted":
            return (
                torch.randn(
                    hidden.shape,
                    generator=torch.Generator().manual_seed(335),
                )
                * 1000.0
            ).to(torch.float16)
        if self.kind == "hard_rejected":
            smallest_subnormal = torch.finfo(torch.float16).smallest_normal * 2**-10
            return torch.full_like(hidden, smallest_subnormal, dtype=torch.float16)
        raise AssertionError("unknown fixed projection")


class _FakeAttention(torch.nn.Module):
    def __init__(self, q_kind: str) -> None:
        super().__init__()
        self.to_q = _FixedProjection(q_kind)
        self.to_k = _FixedProjection("no_correction")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.to_q(hidden).to(torch.float32) + self.to_k(hidden)


class _FakeBlock(torch.nn.Module):
    def __init__(self, q_kind: str = "no_correction") -> None:
        super().__init__()
        self.attn = _FakeAttention(q_kind)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden)


class _FakeTransformer(torch.nn.Module):
    def __init__(self, q_kind: str) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [_FakeBlock(q_kind if index == 4 else "no_correction") for index in range(5)]
        )
        self.device_anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[4](hidden)


class _FakePipeline:
    def __init__(
        self,
        q_kind: str = "no_correction",
        *,
        step_count: int = 20,
        valid_rgb: bool = True,
    ) -> None:
        self.transformer = _FakeTransformer(q_kind)
        self.step_count = step_count
        self.valid_rgb = valid_rgb

    def __call__(self, **kwargs):
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["callback_on_step_end_tensor_inputs"] == ["latents"]
        hidden = torch.full((1, 4, 4), 0.5)
        callback = kwargs["callback_on_step_end"]
        for step in range(self.step_count):
            self.transformer(hidden)
            returned = callback(self, step, step, {"latents": hidden})
            assert returned == {"latents": hidden}
        image = Image.new(
            "RGB",
            (512, 512) if self.valid_rgb else (256, 256),
            (7, 8, 9),
        )
        return SimpleNamespace(images=[image])


def _preloader_for(pipeline):
    def preloader(model_id: str, token: str):
        assert model_id == RUNNER.P0_MODEL_ID
        assert token == "hf-private-token"
        if isinstance(pipeline, BaseException):
            raise pipeline
        return pipeline

    return preloader


NO_CORRECTION_PATH = (
    "q_output_contract_pass",
    "q_pattern_materialized",
    "q_base_rms_validated",
    "q_delta_materialized",
    "q_ratio_validated",
    "q_initial_budget_comparison_completed",
    "q_hard_budget_accepted",
    "q_budget_validated",
    "q_measurement_recorded",
)
CORRECTION_ACCEPTED_PATH = (
    "q_output_contract_pass",
    "q_pattern_materialized",
    "q_base_rms_validated",
    "q_delta_materialized",
    "q_ratio_validated",
    "q_initial_budget_comparison_completed",
    "q_correction_branch_entered",
    "q_corrected_output_materialized",
    "q_corrected_delta_materialized",
    "q_post_correction_ratio_computed",
    "q_hard_budget_accepted",
    "q_budget_validated",
    "q_measurement_recorded",
)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("q_kind", "expected_path"),
    [
        ("no_correction", NO_CORRECTION_PATH),
        ("correction_accepted", CORRECTION_ACCEPTED_PATH),
    ],
)
def test_real_production_super_hook_completes_each_accepted_branch(
    q_kind: str,
    expected_path: tuple[str, ...],
) -> None:
    result = RUNNER.run_p0d2(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=_preloader_for(_FakePipeline(q_kind)),
    )

    assert result.status == RUNNER.P0D2_STATUS_COMPLETE
    assert result.failure_point == "none"
    for checkpoint in RUNNER.P0D2_Q_DIAGNOSTIC_CHECKPOINTS:
        assert result.counters[f"{checkpoint}_count"] == int(
            checkpoint in expected_path
        )
    assert result.counters["block4_to_q_injection_count"] == 1
    assert result.counters["block4_to_k_injection_count"] == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "stop_checkpoint",
    (
        "q_correction_branch_entered",
        "q_corrected_output_materialized",
        "q_corrected_delta_materialized",
        "q_post_correction_ratio_computed",
        "q_hard_budget_accepted",
    ),
)
def test_correction_branch_adjacent_stops_retain_only_completed_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    stop_checkpoint: str,
) -> None:
    original = RUNNER.DiagnosticWriterSession._record_q_checkpoint

    def stop_after_record(self, checkpoint: str) -> None:
        original(self, checkpoint)
        if checkpoint == stop_checkpoint:
            raise RuntimeError("private diagnostic stop text")

    monkeypatch.setattr(
        RUNNER.DiagnosticWriterSession,
        "_record_q_checkpoint",
        stop_after_record,
    )
    result = RUNNER.run_p0d2(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=_preloader_for(_FakePipeline("correction_accepted")),
    )

    assert result.status == RUNNER.P0D2_STATUS_STOPPED
    assert result.failure_point == "pipeline_callback"
    assert result.error_class == "runtime_error"
    stop_index = CORRECTION_ACCEPTED_PATH.index(stop_checkpoint)
    for checkpoint in RUNNER.P0D2_Q_DIAGNOSTIC_CHECKPOINTS:
        expected = checkpoint in CORRECTION_ACCEPTED_PATH[: stop_index + 1]
        assert result.counters[f"{checkpoint}_count"] == int(expected)
    assert result.counters["block4_to_k_injection_count"] == 0


@pytest.mark.integration
def test_explicit_hard_budget_runtime_stop_retains_rejected_checkpoint() -> None:
    result = RUNNER.run_p0d2(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=_preloader_for(_FakePipeline("hard_rejected")),
    )

    assert result.status == RUNNER.P0D2_STATUS_STOPPED
    assert result.failure_point == "pipeline_callback"
    assert result.error_class == "runtime_error"
    assert result.counters["q_correction_branch_entered_count"] == 1
    assert result.counters["q_corrected_output_materialized_count"] == 0
    assert result.counters["q_post_correction_ratio_computed_count"] == 0
    assert result.counters["q_hard_budget_rejected_count"] == 1
    assert result.counters["q_hard_budget_accepted_count"] == 0
    assert result.counters["q_budget_validated_count"] == 0
    assert result.counters["q_measurement_recorded_count"] == 0
    assert result.counters["block4_to_k_injection_count"] == 0


def _invoke_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline,
    run_name: str,
) -> tuple[int, dict[str, object], Path, bytes]:
    exact = "e" * 40
    root = tmp_path / "drive-root"
    root.mkdir(exist_ok=True)
    output = root / run_name
    monkeypatch.setattr(RUNNER, "P0D2_DRIVE_ROOT", root.as_posix() + "/")
    monkeypatch.setattr(RUNNER, "_git_exact", lambda expected: expected)
    plan = tmp_path / f"{run_name}.json"
    plan.write_text(
        json.dumps(
            {
                "expected_exact": exact,
                "execution_exact": exact,
                "output_directory": output.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(RUNNER.TOKEN_ENV, "hf-private-token")
    monkeypatch.setenv(RUNNER.KEY_ENV, "geometry-private-key-0001")
    read_fd, write_fd = os.pipe()
    rc = RUNNER._main(
        ["--plan", str(plan), "--control-fd", str(write_fd)],
        preloader=_preloader_for(pipeline),
    )
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    assert len(payload) <= RUNNER.MAX_CONTROL_BYTES
    return rc, json.loads(payload), output, payload


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pipeline", "run_name", "status", "failure_point"),
    [
        (_FakePipeline(), "complete", RUNNER.P0D2_STATUS_COMPLETE, "none"),
        (
            _FakePipeline("hard_rejected"),
            "hard-rejected",
            RUNNER.P0D2_STATUS_STOPPED,
            "pipeline_callback",
        ),
        (
            RuntimeError("private model failure"),
            "load-stopped",
            RUNNER.P0D2_STATUS_STOPPED,
            "pipeline_load",
        ),
        (
            _FakePipeline(step_count=19),
            "completion-stopped",
            RUNNER.P0D2_STATUS_STOPPED,
            "session_completion",
        ),
        (
            _FakePipeline(valid_rgb=False),
            "rgb-stopped",
            RUNNER.P0D2_STATUS_STOPPED,
            "rgb_validation",
        ),
    ],
)
def test_real_main_packages_bounded_public_create_only_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline,
    run_name: str,
    status: str,
    failure_point: str,
) -> None:
    rc, control, output, control_bytes = _invoke_main(
        tmp_path, monkeypatch, pipeline, run_name
    )

    assert rc == 0
    assert control["status"] == "success"
    assert control["p0d2_status"] == status
    assert control["failure_point"] == failure_point
    assert control["science_denominator"] == 0
    assert len(control_bytes) <= 1024
    assert {path.name for path in output.iterdir()} == {
        "receipt.json",
        "manifest.json",
        "terminal.json",
    }
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["config_id"] == "block4-qk-rms0p0025"
    assert receipt["counters"] == control["counters"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["files"]] == [
        "receipt.json",
        "terminal.json",
    ]
    payload = b"".join(path.read_bytes() for path in output.iterdir())
    assert len(payload) < RUNNER.P0D2_ARTIFACT_MAX_BYTES
    lowered = payload.lower()
    for forbidden in (
        b"geometry-private-key",
        b"hf-private-token",
        b"private model failure",
        b"raw_qk",
        b"anchor_tensor",
        b"latent",
        b"prompt_text",
        b"model_weights",
    ):
        assert forbidden not in lowered
    assert RUNNER.TOKEN_ENV not in os.environ
    assert RUNNER.KEY_ENV not in os.environ


@pytest.mark.integration
def test_runner_is_one_fixed_production_session_without_alternate_paths() -> None:
    source = Path(RUNNER.__file__).read_text(encoding="utf-8").lower()
    assert "production_hook = super()._feature_hook" in source
    assert "retry" not in source
    assert "fallback" not in source
    assert "print(" not in source
    assert "sys.stdout" not in source
    assert "sys.stderr" not in source
    assert "p0d2_config = p0writerconfig(4, 0.0025)" in source
    assert RUNNER.public_plan()["run_count"] == 1
    assert RUNNER.public_plan()["candidate_selection"] is False
