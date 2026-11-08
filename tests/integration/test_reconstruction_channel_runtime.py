"""CPU fakes exercise real orchestration, actual-dtype writes and RGB readout."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.runtime import reconstruction_channel_sd35 as runtime


pytestmark = pytest.mark.integration


class ImageProcessor:
    def preprocess(self, image):
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        return pixels.float() / 255.0


class VAE(torch.nn.Module):
    def __init__(self, fail_at=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.1)
        self.calls = []
        self.fail_at = fail_at

    def encode(self, pixels):
        self.calls.append(pixels.clone())
        if len(self.calls) == self.fail_at:
            raise RuntimeError("public RGB read failed")
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels[:, :2].clone()))


class SD3:
    """Small deterministic chain with the real step-before-callback convention."""

    def __init__(self, *, dtype=torch.float16, drift=None, fail_at=None, omit_suffix=False):
        self.vae, self.image_processor = VAE(), ImageProcessor()
        self.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=2))
        self.vae_scale_factor = 16
        self.dtype, self.drift, self.fail_at = dtype, drift, fail_at
        self.omit_suffix = omit_suffix
        self.calls = []
        self.scheduler = SimpleNamespace(
            config={"stochastic_sampling": False}, order=1, begin_index=None, step_index=None,
        )
        self.guidance_scale = 0.0
        self.do_classifier_free_guidance = False
        self.clip_skip = None
        self.joint_attention_kwargs = None
        self.last_zero_identity = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("source generation failed")
        steps = kwargs["num_inference_steps"]
        self.guidance_scale = kwargs["guidance_scale"]
        self.do_classifier_free_guidance = self.guidance_scale > 1
        self.scheduler.timesteps = torch.arange(steps, 0, -1, dtype=torch.float32)
        self.scheduler.sigmas = torch.linspace(1, 0, steps + 1)
        self.scheduler.step_index = 0
        current = (torch.randn((1, 2, 2, 2), generator=kwargs["generator"]) + 2).to(self.dtype)
        prompt, pooled = torch.ones(2, 3, 4), torch.ones(2, 4)
        callback = kwargs["callback_on_step_end"]
        for i in range(steps):
            current = (current.float() * 0.99 + 0.01).to(self.dtype)
            self.scheduler.step_index = i + 1
            if len(self.calls) == 2 and i == callback.settings.injection_step:
                if self.drift == "latent":
                    current = current + 0.1
                elif self.drift == "conditioning":
                    pooled = pooled + 0.1
                elif self.drift == "sigma":
                    self.scheduler.sigmas[i + 1] += 0.001
                elif self.drift == "rng":
                    torch.randn(1, generator=kwargs["generator"])
                elif self.drift == "index":
                    self.scheduler.step_index = i
            before = current
            state = callback(self, i, self.scheduler.timesteps[i], {
                "latents": current, "prompt_embeds": prompt, "pooled_prompt_embeds": pooled,
            })
            current = state["latents"]
            if len(self.calls) == 1 and i == callback.settings.injection_step:
                self.last_zero_identity = current is before
            if self.omit_suffix and i == callback.settings.injection_step:
                break
        red = int(torch.round(current.float().mean() * 25 + 90).clamp(0, 255).item())
        return SimpleNamespace(images=[Image.new("RGB", (4, 4), (red, 80, 50))])


class SDXL:
    def __init__(self, fail_at=None):
        self.calls, self.noises, self.fail_at = [], [], fail_at

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("reconstruction failed")
        noise = int(torch.randint(0, 15, (1,), generator=kwargs["generator"]).item())
        self.noises.append(noise)
        source = np.array(kwargs["image"], copy=True)
        source[:, :, 0] = np.clip(source[:, :, 0].astype(int) // 2 + noise, 0, 255)
        return SimpleNamespace(images=[Image.fromarray(source)])


def settings(**changes):
    base = runtime.GenerationSettings("a test image", 21, 18, 0.05, 7.0, device="cpu")
    return replace(base, **changes)


def direction():
    return torch.ones(1, 2, 2, 2)


def test_three_full_replays_match_state_and_read_only_final_rgb():
    pipe = SD3()
    result = runtime.run_replay_triplet(pipe, direction(), settings())
    assert tuple(result.arms) == ("zero", "plus", "minus")
    assert len(pipe.calls) == 3
    assert len({id(call["generator"]) for call in pipe.calls}) == 3
    assert len({id(call["callback_on_step_end"]) for call in pipe.calls}) == 3
    for call in pipe.calls:
        assert call["num_inference_steps"] == 20
        assert call["guidance_scale"] == 7.0
        assert call["generator"].initial_seed() == 21
        assert call["callback_on_step_end_tensor_inputs"] == ["latents", "prompt_embeds", "pooled_prompt_embeds"]
    assert result.matched_state
    assert pipe.last_zero_identity
    assert torch.count_nonzero(result.arms["zero"].actual_delta) == 0
    assert result.arms["zero"].injection_metrics["requested_relative_l2"] == 0
    assert len(pipe.vae.calls) == 3
    for arm in result.arms.values():
        assert arm.remaining_steps == (19,)
        assert arm.before_state["next_step_index"] == 19
        assert torch.equal(arm.before_state["latent_sigma"], pipe.scheduler.sigmas[19])
        assert arm.after_latents.dtype == torch.float16
        assert torch.equal(arm.actual_delta, arm.after_latents.double() - arm.before_state["latents"].double())
        expected = (pipe.image_processor.preprocess(arm.image)[:, :2] - 0.1) * 2
        assert torch.equal(arm.feature, expected.reshape(-1))
        assert arm.feature.numel() == 32  # all public VAE coordinates; no pooling
    assert not torch.equal(result.arms["plus"].feature, result.arms["minus"].feature)


@pytest.mark.parametrize("drift,path", [
    ("latent", "state.latents"), ("conditioning", "state.pooled_prompt_embeds"),
    ("sigma", "state.sigmas"), ("rng", "state.generator_state"),
])
def test_state_mismatch_retains_zero_and_failed_before_state_without_third_attempt(drift, path):
    pipe = SD3(drift=drift)
    with pytest.raises(runtime.ReplayExecutionError) as caught:
        runtime.run_replay_triplet(pipe, direction(), settings())
    error = caught.value
    assert error.arm == "plus" and error.stage == "generation"
    assert error.cause_type == "ReplayStateMismatch"
    assert path in error.mismatch_paths
    assert tuple(error.completed_arms) == ("zero",)
    assert error.attempted_arms == ("zero", "plus")
    assert tuple(error.arms) == ("zero", "plus")
    assert error.arms["plus"].before_state
    assert error.arms["plus"].actual_delta is None
    assert len(pipe.calls) == 2


def test_scheduler_callback_off_by_one_is_rejected():
    with pytest.raises(runtime.ReplayExecutionError, match="must follow scheduler.step"):
        runtime.run_replay_triplet(SD3(drift="index"), direction(), settings())


@pytest.mark.parametrize("kind", ["stochastic", "multistep", "suffix"])
def test_unsupported_or_incomplete_source_path_is_not_scored(kind):
    pipe = SD3(omit_suffix=kind == "suffix")
    if kind == "stochastic":
        pipe.scheduler.config["stochastic_sampling"] = True
    elif kind == "multistep":
        pipe.scheduler.order = 2
    with pytest.raises(runtime.ReplayExecutionError) as caught:
        runtime.run_replay_triplet(pipe, direction(), settings())
    assert caught.value.arm == "zero"
    assert len(pipe.vae.calls) == 0


def test_quantized_zero_probe_is_recorded_without_renormalization_or_retry():
    pipe = SD3(dtype=torch.float16)
    result = runtime.run_replay_triplet(pipe, direction(), settings(relative_l2=1e-10))
    assert len(pipe.calls) == 3
    for name in ("plus", "minus"):
        arm = result.arms[name]
        assert arm.injection_metrics["requested_relative_l2"] == 1e-10
        assert arm.injection_metrics["actual_relative_l2"] == 0
        assert arm.injection_metrics["nonzero_elements"] == 0


def test_source_rgb_read_failure_preserves_generated_image_and_actual_write():
    pipe = SD3()
    vae = VAE(fail_at=2)
    with pytest.raises(runtime.ReplayExecutionError) as caught:
        runtime.run_replay_triplet(pipe, direction(), settings(), observation_vae=vae)
    error = caught.value
    assert error.stage == "clean_feature" and error.arm == "plus"
    assert error.cause_type == "RuntimeError"
    assert error.arms["plus"].image.mode == "RGB"
    assert error.arms["plus"].feature is None
    assert error.arms["plus"].after_latents is not None
    assert error.arms["plus"].actual_delta is not None
    assert error.arms["plus"].status == "FAILED"
    assert len(pipe.calls) == len(vae.calls) == 2


def test_source_pipeline_failure_keeps_completed_arm():
    pipe = SD3(fail_at=2)
    with pytest.raises(runtime.ReplayExecutionError) as caught:
        runtime.run_replay_triplet(pipe, direction(), settings())
    assert caught.value.stage == "generation"
    assert caught.value.arms["zero"].feature is not None
    assert caught.value.arms["plus"].image is None
    assert caught.value.arms["plus"].error == "RuntimeError: source generation failed"


def test_reconstruction_pairs_noise_and_uses_fresh_source_vae_reads():
    source_pipe = SD3()
    source = runtime.run_replay_triplet(source_pipe, direction(), settings())
    recon, vae = SDXL(), VAE()
    spec = runtime.ReconstructionSettings(0.3, 1.0, 20, device="cpu")
    result = runtime.reconstruct_triplet(recon, source, spec, 91,
        observation_vae=vae, observation_processor=source_pipe.image_processor)
    assert result.seed == 91 and len(recon.calls) == len(vae.calls) == 3
    assert len({id(call["generator"]) for call in recon.calls}) == 3
    assert len(set(recon.noises)) == 1
    for name, call in zip(runtime.ARM_SIGNS, recon.calls):
        assert call["strength"] == 0.3 and call["num_inference_steps"] == 20
        assert call["guidance_scale"] == 1.0 and call["prompt"] == ""
        assert call["image"] is not source.arms[name].image
        expected = (source_pipe.image_processor.preprocess(result.arms[name].image)[:, :2] - 0.1) * 2
        assert torch.equal(result.arms[name].feature, expected.reshape(-1))


@pytest.mark.parametrize("failure", ["pipeline", "feature"])
def test_reconstruction_failure_preserves_completed_and_current_rgb(failure):
    pipe = SD3()
    source = runtime.run_replay_triplet(pipe, direction(), settings())
    recon = SDXL(fail_at=2 if failure == "pipeline" else None)
    vae = VAE(fail_at=2 if failure == "feature" else None)
    with pytest.raises(runtime.ReplayExecutionError) as caught:
        runtime.reconstruct_triplet(recon, source,
            runtime.ReconstructionSettings(0.3, 1.0, 20, device="cpu"), 13,
            observation_vae=vae, observation_processor=pipe.image_processor)
    error = caught.value
    assert error.arm == "plus" and tuple(error.completed_arms) == ("zero",)
    assert error.attempted_arms == ("zero", "plus")
    assert error.stage == ("reconstruction" if failure == "pipeline" else "reconstruction_feature")
    assert (error.arms["plus"].image is not None) == (failure == "feature")
    assert len(recon.calls) == 2


@pytest.mark.parametrize("changes", [
    {"injection_step": 19}, {"relative_l2": 0}, {"seed": -1}, {"guidance_scale": float("nan")},
])
def test_invalid_generation_parameters_are_rejected_before_calls(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_zero_effective_img2img_steps_are_rejected():
    with pytest.raises(ValueError, match="at least one img2img step"):
        runtime.ReconstructionSettings(0.001, 1, 20)


def test_complete_runner_uses_real_runtime_api_and_saves_six_observations(tmp_path):
    from experiments import run_reconstruction_channel_v1 as runner

    config = {
        "generation": {
            "model_id": "fake-sd3", "revision": "fake", "device": "cpu", "dtype": "float16",
            "height": 32, "width": 32, "num_inference_steps": 20,
            "guidance_scale": 7.0, "injection_step": 18,
        },
        "probe": {"seed": 31, "relative_l2": 0.01},
        "feature": "full_vae_mode",
        "reconstruction": {
            "model_id": "fake-sdxl", "revision": "fake", "device": "cpu", "dtype": "float32",
            "strength": 0.3, "num_inference_steps": 20, "guidance_scale": 1.0,
            "prompt": "", "seeds": [91],
        },
        "sources": [{"source_id": "source-one", "prompt": "a test image", "seed": 21}],
    }
    generation, reconstruction = SD3(), SDXL()
    output = tmp_path / "measurement"
    result = runner.execute_measurement(config, output, pipelines=(generation, reconstruction))
    assert result["actual_calls"] == {
        "generation_attempted": 3, "generation_returned": 3,
        "reconstruction_attempted": 3, "reconstruction_returned": 3,
        "feature_reads_attempted": 6, "feature_reads_saved": 6,
    }
    assert len(generation.calls) == len(reconstruction.calls) == 3
    assert len(generation.vae.calls) == 6
    row = result["sources"][0]
    assert row["status"] == "measured"
    assert row["measurement"]["status"] == "MEASURED"
    assert row["measurement"]["expected_arm_count"] == row["measurement"]["observed_arm_count"] == 6
    assert row["matched_state"]
    assert len(list((output / "source-one").rglob("*.png"))) == 6
    assert len(list((output / "source-one").rglob("*.pt"))) == 6
    assert (output / "probe.pt").is_file()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["actual_calls"] == result["actual_calls"]
    evidence = torch.load(output / "source-one/clean/plus.pt", weights_only=True)
    assert evidence["after_latents"].dtype == torch.float16
    assert torch.equal(evidence["actual_delta"],
                       evidence["after_latents"].double() - evidence["before_state"]["latents"].double())
    assert evidence["feature"].numel() == 32
