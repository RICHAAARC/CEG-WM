"""CPU-only stage lifecycle checks; no model downloads or scientific claims."""

import json
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments import run_reconstruction_channel_v1 as runner
from cegwm.runtime import reconstruction_channel_sd35 as runtime

pytestmark = pytest.mark.integration


def config(count=2):
    return {
        "generation": {"model_id": "fake-sd3", "revision": "fake", "device": "cpu", "dtype": "float32",
                       "height": 32, "width": 32, "num_inference_steps": 2,
                       "guidance_scale": 1.0, "injection_step": 0},
        "probe": {"seed": 31, "relative_l2": 0.01}, "feature": "full_vae_mode",
        "reconstruction": {"model_id": "fake-sdxl", "revision": "fake", "device": "cpu", "dtype": "float32",
                           "strength": 0.5, "num_inference_steps": 2, "guidance_scale": 1.0,
                           "prompt": "", "seeds": [9]},
        "sources": [{"source_id": f"source_{i}", "prompt": f"source {i}", "seed": i}
                    for i in range(count)],
    }


class Reader(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.events = events

    def to(self, device):
        self.events.append(("reader_device", str(device)))
        return super().to(device)


class Processor:
    def __init__(self, events):
        self.events = events

    def preprocess(self, image):
        self.events.append(("feature", image.getpixel((0, 0))))
        return torch.zeros(1)


class Pipeline:
    def __init__(self, name, events, reader):
        self.name, self.events, self.vae = name, events, reader
        self.image_processor = Processor(events)
        self.vae_scale_factor = 16
        self.transformer = torch.nn.Linear(1, 1)
        self.transformer.config = SimpleNamespace(in_channels=2)
        # Simulate the retained VAE's offload hook pointing to a large component.
        if name == "generation":
            self.vae._hf_hook = SimpleNamespace(previous=self.transformer)

    def __call__(self, **kwargs):
        self.events.append((self.name, kwargs["source"]))
        return Image.new("RGB", (4, 4), (80, 90, 100))

    def remove_all_hooks(self):
        self.events.append(("remove_hooks", self.name))
        if hasattr(self.vae, "_hf_hook"):
            del self.vae._hf_hook


def install_fakes(monkeypatch, *, failed_source=None, reconstruction_load_error=False,
                  collapsed=False, mismatched_feature=False):
    events, weak_models = [], {}
    reader = Reader(events)

    def load(settings, name):
        events.append(("load", name))
        if name == "reconstruction":
            assert weak_models["generation"]() is None, "SD3 pipeline still retained at SDXL load"
            assert weak_models["transformer"]() is None, "SD3 component still retained by offload hooks"
            assert not hasattr(reader, "_hf_hook")
            if reconstruction_load_error:
                raise RuntimeError("fake SDXL load failure")
        pipeline = Pipeline(name, events, reader)
        weak_models[name] = weakref.ref(pipeline)
        if name == "generation":
            weak_models["transformer"] = weakref.ref(pipeline.transformer)
        return pipeline

    def replay(pipeline, direction, settings, *, observation_vae, observation_processor):
        assert observation_vae is reader
        base = torch.ones_like(direction, dtype=torch.float16)
        arms = {}
        for name, sign in (("zero", 0), ("plus", 1), ("minus", -1)):
            image = pipeline(source=settings.seed)
            if settings.seed == failed_source and name == "plus":
                raise runtime.ReplayExecutionError("fake plus failure", stage="generation", arm=name,
                                                   completed_arms=arms)
            observation_processor.preprocess(image)
            after = (base.double() + (0 if collapsed else sign * 0.1) * direction).to(base.dtype)
            arms[name] = SimpleNamespace(image=image, feature=after.float().flatten(), before_state={"latents": base},
                after_latents=after, actual_delta=after.double() - base.double(), injection_metrics={})
        return SimpleNamespace(arms=arms, matched_state=True, source=settings.seed)

    def reconstruct(pipeline, triplet, settings, seed, *, observation_vae, observation_processor):
        assert observation_vae is reader
        arms = {}
        for name, original in triplet.arms.items():
            image = pipeline(source=triplet.source)
            observation_processor.preprocess(image)
            feature = original.feature + 1
            if mismatched_feature and name == "minus":
                feature = feature[:-1]
            arms[name] = SimpleNamespace(image=image, feature=feature)
        return SimpleNamespace(arms=arms)

    monkeypatch.setattr(runner, "_load_pipeline", load)
    monkeypatch.setattr(runtime, "run_replay_triplet", replay)
    monkeypatch.setattr(runtime, "reconstruct_triplet", reconstruct)
    return events, reader, weak_models


def test_all_sources_generated_before_sdxl_load_and_same_reader_survives(tmp_path, monkeypatch):
    events, reader, models = install_fakes(monkeypatch)
    output = tmp_path / "batch"
    result = runner.execute_measurement(config(), output)
    load_index = events.index(("load", "reconstruction"))
    assert sum(event[0] == "generation" for event in events[:load_index]) == 6
    assert not any(event[0] == "generation" for event in events[load_index:])
    assert sum(event[0] == "reconstruction" for event in events[load_index:]) == 6
    assert result["stages"] == {
        "generation_setup": "complete", "generation": "complete", "generation_release": "complete",
        "reconstruction_setup": "complete", "reconstruction": "complete", "cleanup": "complete"}
    assert all(row["collection_complete"] for row in result["sources"])
    assert result["actual_calls"] == {"generation_attempted": 6, "generation_returned": 6,
        "reconstruction_attempted": 6, "reconstruction_returned": 6,
        "feature_reads_attempted": 12, "feature_reads_saved": 12}
    assert len(list(output.rglob("*.png"))) == 12
    assert len(list(output.rglob("*.pt"))) == 13
    assert models["generation"]() is None and models["reconstruction"]() is None
    assert not hasattr(reader, "_hf_hook")
    saved = (output / "summary.json").read_text()
    before = list(events)
    with pytest.raises(FileExistsError):
        runner.execute_measurement(config(), output)
    assert events == before and (output / "summary.json").read_text() == saved


def test_sdxl_setup_failure_keeps_every_clean_source_and_fixed_denominator(tmp_path, monkeypatch):
    events, _, _ = install_fakes(monkeypatch, reconstruction_load_error=True)
    output = tmp_path / "batch"
    result = runner.execute_measurement(config(), output)
    assert "fake SDXL load failure" in result["setup_error"]
    assert result["stages"]["reconstruction_setup"] == "failed"
    assert result["stages"]["cleanup"] == "complete"
    assert result["actual_calls"]["generation_attempted"] == 6
    assert result["actual_calls"]["reconstruction_attempted"] == 0
    assert result["actual_calls"]["feature_reads_saved"] == 6
    assert len(list(output.rglob("*.png"))) == 6
    for row in result["sources"]:
        assert row["status"] == "failed" and not row["collection_complete"]
        assert row["measurement"]["expected_arm_count"] == 6
        assert row["measurement"]["observed_arm_count"] == 3
        assert "reconstruction_setup" in row["errors"]
        assert json.loads((output / row["source_id"] / "measurement.json").read_text()) == row
    assert sum(event == ("load", "reconstruction") for event in events) == 1


def test_partial_generation_saved_while_other_source_is_reconstructed(tmp_path, monkeypatch):
    events, _, _ = install_fakes(monkeypatch, failed_source=0)
    output = tmp_path / "batch"
    result = runner.execute_measurement(config(), output)
    failed, complete = result["sources"]
    assert failed["status"] == "failed" and not failed["collection_complete"]
    assert failed["measurement"]["expected_arm_count"] == 6
    assert failed["measurement"]["observed_arm_count"] == 1
    assert (output / "source_0/clean/zero.png").exists()
    assert complete["collection_complete"]
    assert result["stages"]["generation"] == "completed_with_errors"
    assert result["actual_calls"]["generation_attempted"] == 5
    assert result["actual_calls"]["reconstruction_attempted"] == 3
    assert result["actual_calls"]["feature_reads_saved"] == 7
    assert all(event[1] == 1 for event in events if event[0] == "reconstruction")


def test_zero_actual_amplitude_is_collected_negative_not_collection_failure(tmp_path, monkeypatch):
    install_fakes(monkeypatch, collapsed=True)
    result = runner.execute_measurement(config(1), tmp_path / "batch")
    row = result["sources"][0]
    assert row["collection_complete"]
    assert row["status"] == "incomplete"
    assert row["measurement"]["clean"]["status"] == "INVALID"


def test_feature_dimension_mismatch_is_not_complete_collection(tmp_path, monkeypatch):
    install_fakes(monkeypatch, mismatched_feature=True)
    row = runner.execute_measurement(config(1), tmp_path / "batch")["sources"][0]
    assert not row["collection_complete"]


@pytest.mark.parametrize("memory_mode", ["direct", "model_cpu_offload"])
def test_real_loader_selects_offload_without_full_cuda_to(monkeypatch, memory_mode):
    events = []

    class Loadable:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            events.append(("from_pretrained", model_id, kwargs))
            return cls()

        def to(self, device):
            events.append(("to", device))
            return self

        def enable_model_cpu_offload(self, *, device):
            events.append(("offload", device))

        def set_progress_bar_config(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        StableDiffusion3Pipeline=Loadable, StableDiffusionXLImg2ImgPipeline=Loadable))
    settings = config(1)
    settings["generation"].update(device="cuda:0", memory_mode=memory_mode)
    runner.validate_config(settings)
    runner._load_pipeline(settings, "generation")
    assert [event[0] for event in events] == ["from_pretrained", "offload" if memory_mode == "model_cpu_offload" else "to"]
    assert events[-1][1] == "cuda:0"


def test_offload_requires_cuda_and_valid_memory_mode():
    settings = config(1)
    settings["generation"]["memory_mode"] = "model_cpu_offload"
    with pytest.raises(ValueError, match="requires a CUDA"):
        runner.validate_config(settings)
    settings["generation"]["memory_mode"] = "unknown"
    with pytest.raises(ValueError, match="memory_mode"):
        runner.validate_config(settings)


def test_reader_is_placed_before_observer_preprocess():
    events = []
    reader = Reader(events)
    processor = runner._CountedProcessor(Processor(events), {"feature_reads_attempted": 0},
                                         observation_vae=reader, device="cpu")
    processor.preprocess(Image.new("RGB", (4, 4)))
    assert [event[0] for event in events] == ["reader_device", "feature"]
