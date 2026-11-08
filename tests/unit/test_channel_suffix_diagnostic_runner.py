"""Fixed three-arm diagnostic control flow, without models, downloads or GPU."""

from collections import OrderedDict
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments import run_channel_suffix_diagnostic_v1 as runner
from cegwm.runtime.channel_precompensation_sd35 import RuntimeStageError

pytestmark = pytest.mark.unit


def checkpoint():
    scheduler_config = OrderedDict(stochastic_sampling=False, invert_sigmas=False)
    scheduler_config.stochastic_sampling = False
    scheduler_config.invert_sigmas = False
    capture = {"image": Image.new("RGB", (2, 2)),
               "before_state": {"latents": torch.ones(1, 2, 2, 2)},
               "scheduler_state": {"_step_index": 19, "_internal_dict": scheduler_config}, "remaining_steps": (19,)}
    prior = {"final_generation_attempted": 4, "reconstruction_attempted": 16,
             "rgb_vjps_attempted": 15, "sd35_suffix_forward_calls": 8}
    return SimpleNamespace(config={"generation": {"device": "cpu"}, "sources": [
        {"source_id": runner.SOURCE_ID}, {"source_id": "source06_fur_knit"},
        {"source_id": "source07_coastal_scene"}, {"source_id": "source08_indoor_structure"}]},
        baselines={runner.SOURCE_ID: capture}, prior_actual_calls=prior,
        prior_summary={"cumulative_actual_calls": prior})


class Pipeline:
    def __init__(self):
        for name in ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            setattr(self, name, torch.nn.Linear(2, 2))
        self.remove_hooks_calls = 0

    def remove_all_hooks(self):
        self.remove_hooks_calls += 1


def trace():
    tensor = torch.ones(1, 2, 2, 2)
    return {"before_latents": tensor.clone(),
            "steps": [{"index": 19, "timestep": torch.tensor(8.9285717),
                       "guided_prediction": tensor.clone(), "scheduler_output": tensor.clone()}],
            "pre_vae_latents": tensor.clone(), "decoded": torch.zeros(1, 3, 2, 2),
            "postprocessed_rgb": torch.zeros(1, 3, 2, 2),
            "uint8_rgb": torch.zeros(2, 2, 3, dtype=torch.uint8),
            "image": Image.new("RGB", (2, 2)), "metadata": {"diagnostic_only": True}}


def harness(tmp_path, *, failures=(), mutate=None):
    saved, pipeline, calls = checkpoint(), Pipeline(), []
    output = tmp_path / "diagnostic"

    def loader(config):
        # Initial summary must exist before even model construction.
        summary = json.loads((output / "summary.json").read_text())
        assert summary["actual_calls"]["source_pipeline_load_attempted"] == 1
        assert summary["stages"]["load"] == "started"
        return pipeline

    def tracer(actual_pipeline, capture, *, grad_enabled, counters):
        index = len(calls)
        name = runner.ARMS[index][0]
        calls.append((actual_pipeline, grad_enabled))
        recorded = json.loads((output / "summary.json").read_text())
        assert recorded["arms"][name]["status"] == "STARTED"
        assert recorded["actual_calls"]["suffix_arms_attempted"] == index + 1
        assert pipeline.remove_hooks_calls == 1
        for component in (pipeline.transformer, pipeline.vae, pipeline.text_encoder):
            assert not component.training
            assert not any(parameter.requires_grad for parameter in component.parameters())
        counters["sd35_suffix_forward_calls"] += 1
        counters["sd35_suffix_denoising_steps"] += 1
        result = trace()
        if mutate is not None:
            mutate(name, result)
        if name in failures:
            raise RuntimeStageError("source_suffix_trace", ValueError("fake preserved failure"),
                                    {"before_latents": result["before_latents"], "image": result["image"]}, counters)
        counters["sd35_suffix_forward_returned"] += 1
        counters["sd35_suffix_decode_calls"] += 1
        counters["sd35_suffix_decode_returned"] += 1
        return result

    summary = runner.execute_diagnostic(tmp_path / "saved", output,
        checkpoint_loader=lambda path: saved, pipeline_loader=loader, trace_fn=tracer)
    return summary, output, pipeline, calls


def test_default_cli_only_loads_checkpoint_and_prints_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_checkpoint", lambda path: checkpoint())
    monkeypatch.setattr(runner, "load_pipeline", lambda *args: pytest.fail("plan must not load models"))
    assert runner.main(["--resume-from", str(tmp_path / "saved")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan_only" and plan["source_id"] == runner.SOURCE_ID
    assert plan["planned_budget"]["suffix_forwards"] == 3
    assert plan["planned_budget"]["vjps"] == 0
    assert list(tmp_path.iterdir()) == []


def test_same_pipeline_three_fixed_arms_and_saved_reference(tmp_path):
    summary, output, pipeline, calls = harness(tmp_path)
    assert [enabled for _, enabled in calls] == [False, False, True]
    assert all(actual is pipeline for actual, _ in calls)
    assert pipeline.remove_hooks_calls == 2
    assert summary["status"] == "OBSERVED"
    assert summary["classification"]["kind"] == "ALL_OBSERVED_MATCH_REFERENCE"
    assert summary["actual_calls"]["suffix_arms_attempted"] == 3
    assert summary["actual_calls"]["sd35_suffix_forward_calls"] == 3
    assert summary["cumulative_actual_calls"]["sd35_suffix_forward_calls"] == 11
    for name, value in checkpoint().prior_actual_calls.items():
        if name != "sd35_suffix_forward_calls":
            assert summary["cumulative_actual_calls"][name] == value
    assert all(summary["actual_calls"][name] == 0 for name in runner.FORBIDDEN_COUNTS)
    assert {path.name for path in output.iterdir()} == {
        "config.json", "prior_summary.json", "baseline.pt", "reference_zero.png", "summary.json", "NG1", "NG2", "G"}
    for name, _ in runner.ARMS:
        value = torch.load(output / name / "trace.pt", weights_only=True)
        assert "image" not in value and value["uint8_rgb"].dtype == torch.uint8
        assert (output / name / "image.png").is_file()
    assert summary["stages"]["cleanup"] == "complete"


def test_partial_failure_is_retained_and_does_not_remove_other_arms(tmp_path):
    summary, output, _, calls = harness(tmp_path, failures=("NG2",))
    assert len(calls) == 3
    assert summary["arms"]["NG2"]["status"] == "FAILED"
    assert "fake preserved failure" in summary["arms"]["NG2"]["error"]
    assert summary["classification"]["kind"] == "UNKNOWN_INCOMPLETE"
    assert summary["comparisons"]["NG1_vs_NG2"]["first_unavailable"] == "step19/timestep"
    assert (output / "NG2/image.png").is_file()
    partial = torch.load(output / "NG2/trace.pt", weights_only=True)
    assert "before_latents" in partial
    assert summary["actual_calls"]["suffix_arms_attempted"] == 3
    assert summary["actual_calls"]["suffix_arms_returned"] == 2


def test_saved_baseline_preserves_safe_scheduler_metadata_attributes(tmp_path):
    _, output, _, _ = harness(tmp_path)
    saved = torch.load(output / "baseline.pt", map_location="cpu", weights_only=True)
    config = saved["scheduler_state"]["_internal_dict"]
    assert type(config) is OrderedDict
    assert config["stochastic_sampling"] is False and config.stochastic_sampling is False
    assert config["invert_sigmas"] is False and config.invert_sigmas is False
    assert torch.equal(saved["before_state"]["latents"], checkpoint().baselines[runner.SOURCE_ID]["before_state"]["latents"])


@pytest.mark.parametrize("case, expected", [
    ("repeat", "REPEATED_NOGRAD_DIFFERENCE"),
    ("grad", "GRAD_MODE_ASSOCIATED_DIFFERENCE"),
    ("shared", "SHARED_SUFFIX_REFERENCE_MISMATCH"),
    ("mixed", "MIXED_REFERENCE_AND_GRAD_MODE_DIFFERENCES"),
])
def test_first_difference_and_cautious_classification(tmp_path, case, expected):
    def mutate(name, result):
        if case == "repeat" and name == "NG2" or case in {"grad", "mixed"} and name == "G":
            result["steps"][0]["guided_prediction"][0, 0, 0, 0] += 1
        if case in {"shared", "mixed"}:
            result["uint8_rgb"] += 1
            result["image"] = Image.new("RGB", (2, 2), (1, 1, 1))
    summary, _, _, _ = harness(tmp_path, mutate=mutate)
    assert summary["classification"]["kind"] == expected
    if case == "repeat":
        assert summary["comparisons"]["NG1_vs_NG2"]["first_difference"] == "step19/guided_prediction"
    if case in {"grad", "mixed"}:
        assert summary["comparisons"]["NG1_vs_G"]["first_difference"] == "step19/guided_prediction"
    assert "cannot identify a unique" in summary["classification"]["limits"]


def test_existing_output_and_wrong_source_cannot_start_models(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    saved = checkpoint()
    forbidden = lambda *args: pytest.fail("invalid request must not load models")
    with pytest.raises(FileExistsError):
        runner.execute_diagnostic(tmp_path / "saved", output,
            checkpoint_loader=lambda path: saved, pipeline_loader=forbidden)
    with pytest.raises(ValueError, match="outside"):
        runner.execute_diagnostic(tmp_path / "saved", tmp_path / "saved/inside",
            checkpoint_loader=lambda path: saved, pipeline_loader=forbidden)
    saved.config["sources"].reverse()
    with pytest.raises(ValueError, match="original four-source roster"):
        runner.execute_diagnostic(tmp_path / "saved", tmp_path / "wrong",
            checkpoint_loader=lambda path: saved, pipeline_loader=forbidden)
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("arm, field, comparison", [
    ("NG2", "before_latents", "NG1_vs_NG2"),
    ("G", "step19/timestep", "NG1_vs_G"),
])
def test_changed_control_input_cannot_be_classified_as_grad_or_repeat_effect(tmp_path, arm, field, comparison):
    def mutate(name, result):
        if name == arm:
            value = result["before_latents"] if field == "before_latents" else result["steps"][0]["timestep"]
            value.add_(1)
    summary, _, _, _ = harness(tmp_path, mutate=mutate)
    assert summary["comparisons"][comparison]["first_difference"] == field
    assert summary["classification"]["kind"] == "UNKNOWN_CONTROL_MISMATCH"


def test_nonfinite_trace_comparison_is_unknown():
    first, second = trace(), trace()
    second["decoded"].fill_(float("nan"))
    comparison = runner.compare_traces(first, second)
    assert comparison["status"] == "INCOMPLETE"
    assert comparison["first_unavailable"] == "decoded"
    assert comparison["layers"]["decoded"]["status"] == "NONFINITE"


def test_observed_nonfinite_arm_is_not_complete_collection(tmp_path):
    def mutate(name, result):
        if name == "G":
            result["decoded"].fill_(float("nan"))
    summary, _, _, _ = harness(tmp_path, mutate=mutate)
    assert summary["arms"]["G"]["status"] == "OBSERVED"
    assert summary["status"] == "INCOMPLETE"
    assert summary["classification"]["kind"] == "UNKNOWN_INCOMPLETE"


def test_interrupt_keeps_started_arm_and_rejects_same_output(tmp_path):
    output, saved = tmp_path / "diagnostic", checkpoint()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated interruption")
    with pytest.raises(KeyboardInterrupt):
        runner.execute_diagnostic(tmp_path / "saved", output,
            checkpoint_loader=lambda path: saved, pipeline_loader=lambda _: Pipeline(), trace_fn=interrupt)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["arms"]["NG1"]["status"] == "STARTED"
    assert summary["arms"]["NG2"]["status"] == "NOT_STARTED"
    assert summary["actual_calls"]["suffix_arms_attempted"] == 1
    assert summary["stages"]["traces"] == "started"
    assert summary["stages"]["cleanup"] == "complete"
    with pytest.raises(FileExistsError):
        runner.execute_diagnostic(tmp_path / "saved", output, checkpoint_loader=lambda path: saved)
