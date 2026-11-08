"""Entry-point checks; no model initialization or scientific measurements."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments import run_reconstruction_channel_v1 as runner

pytestmark = pytest.mark.unit


def config():
    return {
        "generation": {"model_id": "fake-sd3", "revision": "fake-revision", "device": "cpu", "dtype": "float32",
                       "height": 32, "width": 32, "num_inference_steps": 2, "guidance_scale": 1.0, "injection_step": 0},
        "probe": {"seed": 31, "relative_l2": 0.01}, "feature": "full_vae_mode",
        "reconstruction": {"model_id": "fake-sdxl", "revision": "fake-revision", "device": "cpu", "dtype": "float32",
                           "strength": 0.5, "num_inference_steps": 2, "guidance_scale": 1.0, "prompt": "", "seeds": [9]},
        "sources": [{"source_id": "example", "prompt": "example prompt", "seed": 7}],
    }


def test_plan_does_not_load_models_or_create_output(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    output = tmp_path / "results"
    monkeypatch.setattr(runner, "load_pipelines", lambda _: pytest.fail("plan loaded a model"))
    assert runner.main(["--config", str(path), "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["budget"]["generation_calls"] == 3
    assert plan["budget"]["reconstruction_calls"] == 3
    assert plan["budget"]["public_feature_calls"] == 6
    assert not output.exists()


@pytest.mark.parametrize("section,field", [("generation", "revision"), ("generation", "guidance_scale"),
    ("probe", "relative_l2"), ("reconstruction", "strength"), ("reconstruction", "seeds")])
def test_missing_choices_are_not_inherited(section, field):
    settings = config()
    del settings[section][field]
    with pytest.raises(ValueError):
        runner.validate_config(settings)


def test_probe_is_shared_unit_direction_not_source_conditioned():
    first = runner.make_fixed_probe((1, 2, 8, 8), 31)
    assert torch.equal(first, runner.make_fixed_probe((1, 2, 8, 8), 31))
    assert not torch.equal(first, runner.make_fixed_probe((1, 2, 8, 8), 32))
    assert float(torch.linalg.vector_norm(first)) == pytest.approx(1.0)
    assert abs(float(first.mean())) < 1e-12


def test_setup_failure_preserves_roster_and_output_is_not_overwritten(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("fake loading failure")
    monkeypatch.setattr(runner, "load_pipelines", fail)
    output = tmp_path / "results"
    result = runner.execute_measurement(config(), output)
    assert result["sources"] == [{"source_id": "example", "status": "not_attempted"}]
    assert "fake loading failure" in result["setup_error"]
    saved = (output / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        runner.execute_measurement(config(), output)
    assert (output / "summary.json").read_text() == saved


def test_pixel_distortion_keeps_identity_json_finite():
    image = Image.new("RGB", (4, 4), (30, 50, 80))
    quality = runner.pixel_distortion(image, image)
    assert quality["identical_pixels"] and quality["psnr_db"] is None
    json.dumps(quality, allow_nan=False)


def test_partial_generation_is_saved_and_all_arms_remain(tmp_path, monkeypatch):
    from cegwm.runtime import reconstruction_channel_sd35 as runtime

    arm = SimpleNamespace(image=Image.new("RGB", (4, 4)), feature=torch.zeros(3),
                          before_state={"latents": torch.ones(1, 1, 4, 4)}, actual_delta=torch.zeros(1, 1, 4, 4))
    def fail(*args, **kwargs):
        raise runtime.ReplayExecutionError("fake plus failure", stage="generation", arm="plus", completed_arms={"zero": arm})
    monkeypatch.setattr(runtime, "run_replay_triplet", fail)
    pipeline = SimpleNamespace(vae=object(), image_processor=object(), vae_scale_factor=8,
                               transformer=SimpleNamespace(config=SimpleNamespace(in_channels=1)))
    result = runner.execute_measurement(config(), tmp_path / "results", pipelines=(pipeline, object()))
    source = result["sources"][0]
    assert source["status"] == "failed"
    assert source["measurement"]["expected_arm_count"] == 6
    assert source["measurement"]["clean"]["status"] == "INCOMPLETE"
    assert (tmp_path / "results/example/clean/zero.png").exists()
    assert (tmp_path / "results/example/clean/zero.pt").exists()


@pytest.mark.parametrize("collapsed", [False, True])
def test_complete_runner_keeps_actual_dtype_and_invalid_amplitude_status(tmp_path, monkeypatch, collapsed):
    from cegwm.runtime import reconstruction_channel_sd35 as runtime

    def replay(pipeline, direction, settings, **kwargs):
        base = torch.ones_like(direction, dtype=torch.float16)
        arms = {}
        for name, sign in (("zero", 0), ("plus", 1), ("minus", -1)):
            after = (base.double() + (0 if collapsed else sign * 0.1) * direction).to(base.dtype)
            arms[name] = SimpleNamespace(image=Image.new("RGB", (32, 32)), feature=after.float().flatten(),
                before_state={"latents": base}, after_latents=after,
                actual_delta=after.double() - base.double(), injection_metrics={})
        return SimpleNamespace(arms=arms, matched_state=True)

    def reconstruct(pipeline, triplet, settings, seed, **kwargs):
        return SimpleNamespace(arms={name: SimpleNamespace(image=arm.image, feature=arm.feature + 1)
                                     for name, arm in triplet.arms.items()})

    monkeypatch.setattr(runtime, "run_replay_triplet", replay)
    monkeypatch.setattr(runtime, "reconstruct_triplet", reconstruct)
    pipeline = SimpleNamespace(vae=object(), image_processor=object(), vae_scale_factor=8,
                               transformer=SimpleNamespace(config=SimpleNamespace(in_channels=1)))
    output = tmp_path / "result"
    result = runner.execute_measurement(config(), output, pipelines=(pipeline, object()))
    row = result["sources"][0]
    assert row["status"] == ("incomplete" if collapsed else "measured")
    measurement = row["measurement"]
    assert measurement["probe_geometry"]["actual_dtype"] == "torch.float16"
    assert measurement["expected_arm_count"] == measurement["observed_arm_count"] == 6
    assert measurement["clean"]["status"] == ("INVALID" if collapsed else "MEASURED")
    saved = json.loads((output / "summary.json").read_text())
    assert saved["sources"][0]["status"] == row["status"]
