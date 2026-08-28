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
        / "run_geometry_v3_qk_active_writer_p0d1.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_v3_p0d1_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4, bias=False)
        self.to_k = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.to_q(hidden) + self.to_k(hidden)


class _FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _FakeAttention()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden)


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [_FakeBlock() for _ in range(5)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[4](hidden)


class _FakePipeline:
    def __init__(
        self,
        *,
        step_count: int = 20,
        raise_after_root_count: int | None = None,
        valid_rgb: bool = True,
    ) -> None:
        self.transformer = _FakeTransformer()
        self.step_count = step_count
        self.raise_after_root_count = raise_after_root_count
        self.valid_rgb = valid_rgb

    def __call__(self, **kwargs):
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["callback_on_step_end_tensor_inputs"] == ["latents"]
        hidden = torch.full((1, 4, 4), 0.5)
        callback = kwargs["callback_on_step_end"]
        for step in range(self.step_count):
            self.transformer(hidden)
            if self.raise_after_root_count == step + 1:
                raise RuntimeError("private callback failure text")
            returned = callback(self, step, step, {"latents": hidden})
            assert returned == {"latents": hidden}
        image = (
            Image.new("RGB", (512, 512), (7, 8, 9))
            if self.valid_rgb
            else Image.new("RGB", (256, 256), (7, 8, 9))
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


@pytest.mark.integration
def test_real_p0d1_writer_path_counts_one_fixed_complete_session() -> None:
    result = RUNNER.run_p0d1(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=_preloader_for(_FakePipeline()),
    )
    assert result.status == RUNNER.P0D1_STATUS_COMPLETE
    assert result.failure_point == "none"
    assert result.error_class is None
    assert result.counters == {
        "pipeline_load_count": 1,
        "session_setup_count": 1,
        "pipeline_callback_count": 20,
        "step18_reached_count": 1,
        "transformer_root_call_count": 20,
        "block4_to_q_hook_hit_count": 20,
        "block4_to_k_hook_hit_count": 20,
        "block4_to_q_injection_count": 1,
        "block4_to_k_injection_count": 1,
        "q_output_contract_pass_count": 1,
        "q_pattern_materialized_count": 1,
        "q_base_rms_validated_count": 1,
        "q_delta_materialized_count": 1,
        "q_ratio_validated_count": 1,
        "q_budget_validated_count": 1,
        "q_measurement_recorded_count": 1,
        "session_completion_count": 1,
        "final_rgb_validation_count": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("preloader", "expected_stage", "expected_error", "expected"),
    [
        (
            _preloader_for(RuntimeError("private model failure")),
            "pipeline_load",
            "runtime_error",
            {"pipeline_load_count": 0},
        ),
        (
            _preloader_for(SimpleNamespace(transformer=object())),
            "session_setup",
            "runtime_error",
            {"pipeline_load_count": 1, "session_setup_count": 0},
        ),
        (
            _preloader_for(_FakePipeline(raise_after_root_count=3)),
            "pipeline_callback",
            "runtime_error",
            {
                "pipeline_callback_count": 2,
                "transformer_root_call_count": 3,
                "block4_to_q_hook_hit_count": 3,
                "block4_to_k_hook_hit_count": 3,
            },
        ),
        (
            _preloader_for(_FakePipeline(step_count=19)),
            "session_completion",
            "runtime_error",
            {
                "pipeline_callback_count": 19,
                "step18_reached_count": 1,
                "block4_to_q_injection_count": 1,
                "block4_to_k_injection_count": 1,
                "session_completion_count": 0,
            },
        ),
        (
            _preloader_for(_FakePipeline(valid_rgb=False)),
            "rgb_validation",
            "validation_error",
            {
                "session_completion_count": 1,
                "final_rgb_validation_count": 0,
            },
        ),
    ],
)
def test_p0d1_stages_fail_closed_with_public_counters(
    preloader,
    expected_stage: str,
    expected_error: str,
    expected: dict[str, int],
) -> None:
    result = RUNNER.run_p0d1(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=preloader,
    )
    assert result.status == RUNNER.P0D1_STATUS_STOPPED
    assert result.failure_point == expected_stage
    assert result.error_class == expected_error
    assert tuple(result.counters) == RUNNER.COUNTER_NAMES
    for name, value in expected.items():
        assert result.counters[name] == value


@pytest.mark.integration
@pytest.mark.parametrize(
    "stop_checkpoint",
    (
        "q_output_contract_pass",
        "q_pattern_materialized",
        "q_base_rms_validated",
        "q_delta_materialized",
        "q_ratio_validated",
        "q_budget_validated",
        "q_measurement_recorded",
    ),
)
def test_real_production_q_checkpoints_localize_inner_hook_stops(
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
    result = RUNNER.run_p0d1(
        geometry_key="geometry-private-key-0001",
        hf_token="hf-private-token",
        preloader=_preloader_for(_FakePipeline()),
    )

    assert result.status == RUNNER.P0D1_STATUS_STOPPED
    assert result.failure_point == "pipeline_callback"
    assert result.error_class == "runtime_error"
    checkpoints = (
        "q_output_contract_pass",
        "q_pattern_materialized",
        "q_base_rms_validated",
        "q_delta_materialized",
        "q_ratio_validated",
        "q_budget_validated",
        "q_measurement_recorded",
    )
    stop_index = checkpoints.index(stop_checkpoint)
    for index, checkpoint in enumerate(checkpoints):
        assert result.counters[f"{checkpoint}_count"] == int(index <= stop_index)
    assert result.counters["block4_to_k_injection_count"] == 0


def _invoke_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preloader,
    run_name: str,
) -> tuple[int, dict[str, object], Path]:
    exact = "d" * 40
    root = tmp_path / "drive-root"
    root.mkdir(exist_ok=True)
    output = root / run_name
    monkeypatch.setattr(RUNNER, "P0D1_DRIVE_ROOT", root.as_posix() + "/")
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
        preloader=preloader,
    )
    payload = os.read(read_fd, RUNNER.MAX_CONTROL_BYTES + 1)
    os.close(read_fd)
    assert len(payload) <= RUNNER.MAX_CONTROL_BYTES
    return rc, json.loads(payload), output


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pipeline", "run_name", "status", "failure_point"),
    [
        (_FakePipeline(), "complete", RUNNER.P0D1_STATUS_COMPLETE, "none"),
        (
            RuntimeError("private model failure"),
            "pipeline-load-stopped",
            RUNNER.P0D1_STATUS_STOPPED,
            "pipeline_load",
        ),
        (
            _FakePipeline(raise_after_root_count=3),
            "callback-stopped",
            RUNNER.P0D1_STATUS_STOPPED,
            "pipeline_callback",
        ),
        (
            _FakePipeline(step_count=19),
            "completion-stopped",
            RUNNER.P0D1_STATUS_STOPPED,
            "session_completion",
        ),
        (
            _FakePipeline(valid_rgb=False),
            "rgb-stopped",
            RUNNER.P0D1_STATUS_STOPPED,
            "rgb_validation",
        ),
    ],
)
def test_real_main_packages_bounded_public_diagnostic_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline,
    run_name: str,
    status: str,
    failure_point: str,
) -> None:
    rc, control, output = _invoke_main(
        tmp_path, monkeypatch, _preloader_for(pipeline), run_name
    )
    assert rc == 0
    assert control["status"] == "success"
    assert control["p0d1_status"] == status
    assert control["failure_point"] == failure_point
    assert control["science_denominator"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "receipt.json",
        "manifest.json",
        "terminal.json",
    }
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == status
    assert receipt["failure_point"] == failure_point
    assert receipt["config_id"] == "block4-qk-rms0p0025"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["files"]] == [
        "receipt.json",
        "terminal.json",
    ]
    payload = b"".join(path.read_bytes() for path in output.iterdir())
    assert len(payload) < RUNNER.P0D1_ARTIFACT_MAX_BYTES
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
        b"private diagnostic stop text",
    ):
        assert forbidden not in lowered
    assert RUNNER.TOKEN_ENV not in os.environ
    assert RUNNER.KEY_ENV not in os.environ


@pytest.mark.integration
def test_runner_has_no_retry_fallback_or_stdout_stderr_control_path() -> None:
    source = Path(RUNNER.__file__).read_text(encoding="utf-8").lower()
    assert "retry" not in source
    assert "fallback" not in source
    assert "print(" not in source
    assert "sys.stdout" not in source
    assert "sys.stderr" not in source
    assert "p0d1_config = p0writerconfig(4, 0.0025)" in source

