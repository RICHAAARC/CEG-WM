from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import cegwm.runtime.geometry_v6_sd35 as runtime
from cegwm.runtime.geometry_v6_sd35 import run_sd35_geometry_v6_r0_arm
from experiments import geometry_v6_r0_engine as r0_engine


class _Distribution:
    def __init__(self, value): self._value = value
    def mode(self): return self._value


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
    def decode(self, value, return_dict=True): return SimpleNamespace(sample=value * self.anchor)
    def encode(self, value): return SimpleNamespace(latent_dist=_Distribution(value * self.anchor))


class _Pipeline:
    def __init__(self):
        self.vae = _VAE()
        self.events = []
    def __call__(self, **kwargs):
        callback = kwargs["callback_on_step_end"]
        latents = torch.zeros(1, 4, 16, 16)
        for step in range(20):
            latents = latents + 1.0  # fake scheduler update happens before callback
            self.events.append(("scheduler", step, float(latents.mean())))
            latents = callback(self, step, None, {"latents": latents})["latents"]
            self.events.append(("callback", step, float(latents.mean())))
        self.events.append(("decode", float(latents.mean())))
        return SimpleNamespace(images=[Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), "RGB")])


def test_fake_pipeline_fixes_step19_after_scheduler_and_before_final_decode():
    pipeline = _Pipeline()
    output = run_sd35_geometry_v6_r0_arm(
        pipeline, "frozen prompt", "geometry_only", content_key=None,
        geometry_key="geometry-key-0001", amplitude=0.0025, content_assets=None,
        height=256, width=256,
    )
    assert output.arm == "geometry_only"
    scheduler_19 = next(event for event in pipeline.events if event[:2] == ("scheduler", 19))
    callback_19 = next(event for event in pipeline.events if event[:2] == ("callback", 19))
    decode = pipeline.events[-1]
    assert scheduler_19[2] == 20.0
    assert callback_19[2] != scheduler_19[2]
    assert decode[0] == "decode" and decode[1] == callback_19[2]


def test_combined_path_keeps_content_step18_then_geometry_step19_then_decode(monkeypatch):
    content_events = []

    class FakeUnchangedContentCallback:
        def __init__(self, key, assets):
            self.measurement = None
        def __call__(self, pipeline, step_index, timestep, callback_kwargs):
            if step_index == 18:
                content_events.append(("content", step_index, float(callback_kwargs["latents"].mean())))
                updated = dict(callback_kwargs)
                updated["latents"] = callback_kwargs["latents"] + 10.0
                self.measurement = "unchanged-content-step18-effect"
                return updated
            return callback_kwargs

    monkeypatch.setattr(runtime, "ContentAdaptiveInjectionCallback", FakeUnchangedContentCallback)
    pipeline = _Pipeline()
    output = run_sd35_geometry_v6_r0_arm(
        pipeline, "frozen prompt", "content_geometry", content_key="content-key-00001",
        geometry_key="geometry-key-0001", amplitude=0.0025, content_assets=object(),
        height=256, width=256,
    )
    scheduler_18 = next(event for event in pipeline.events if event[:2] == ("scheduler", 18))
    callback_18 = next(event for event in pipeline.events if event[:2] == ("callback", 18))
    scheduler_19 = next(event for event in pipeline.events if event[:2] == ("scheduler", 19))
    callback_19 = next(event for event in pipeline.events if event[:2] == ("callback", 19))
    assert output.content_measurement == "unchanged-content-step18-effect"
    assert content_events == [("content", 18, scheduler_18[2])]
    assert callback_18[2] == scheduler_18[2] + 10.0
    assert scheduler_19[2] == callback_18[2] + 1.0
    assert callback_19[2] != scheduler_19[2]
    assert pipeline.events[-1] == ("decode", callback_19[2])


def test_notebook_delegates_the_fixed_full_sequence_to_the_engine():
    notebook = json.loads(Path("notebooks/geometry_v6_r0_colab.ipynb").read_text())
    assert notebook["cells"][0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')\n",
    ]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"])
    serialized = json.dumps(notebook)
    assert "CEG_WM_GEOMETRY_V6_R0_AMPLITUDE" not in serialized
    assert "--amplitude" not in serialized


def test_r0_content_raw_uses_the_frozen_whitened_lf_scorer_for_all_keys(monkeypatch):
    calls = []
    wrong_keys = tuple(f"wrong-{index}".encode() for index in range(16))

    monkeypatch.setattr(r0_engine, "derive_stability_wrong_keys", lambda key: wrong_keys)
    monkeypatch.setattr(r0_engine, "score_content_whitened_lf_image", lambda image, key, assets: calls.append(("lf", key, assets)) or 0.25)
    monkeypatch.setattr(r0_engine, "score_hf_image", lambda image, key, assets: calls.append(("hf", key, assets)) or 0.5)
    monkeypatch.setattr(r0_engine, "weighted_joint_score", lambda lf, hf, calibration: lf + hf)
    assets = SimpleNamespace(hf_public_assets="frozen-hf")
    whitening = object()
    records = r0_engine._content_raw("ordinary-rgb", "content-key-0001", assets, whitening, "calibration")
    assert not hasattr(r0_engine, "score_content_image")
    assert tuple(records) == ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    assert all(item[2] == whitening for item in calls[::2])
    assert [item[1] for item in calls[::2]] == ["content-key-0001", *wrong_keys]
    assert all(item[2] == "frozen-hf" for item in calls[1::2])
    assert all(record["lf"] == 0.25 and record["hf"] == 0.5 and record["weighted_joint"] == 0.75 for record in records.values())
