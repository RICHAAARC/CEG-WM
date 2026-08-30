from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method.geometry_v4_generative import RGBObservability
from cegwm.runtime.geometry_v4_sd35 import FinalLatentAnchorCallback, run_sd35_final_latent_pair
from experiments import geometry_v4_generative_engine as engine


class _FixturePipeline:
    class _VAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))
        def decode(self, value, return_dict=True): return SimpleNamespace(sample=value[:, :3] * self.weight)
    vae = _VAE()
    def __call__(self, *, callback_on_step_end=None, **kwargs):
        state = {"latents": torch.zeros((1, 4, 16, 16), dtype=torch.float32)}
        if callback_on_step_end is not None:
            for step in range(20): state = callback_on_step_end(self, step, None, state)
        return SimpleNamespace(images=[Image.new("RGB", (256, 256), "gray")])


@pytest.mark.integration
def test_final_callback_is_sole_step_19_and_pair_materializes_rgb() -> None:
    callback = FinalLatentAnchorCallback("0123456789abcdef")
    early = {"latents": torch.zeros((1, 4, 16, 16), dtype=torch.float32)}
    fixture = _FixturePipeline()
    assert callback(fixture, 18, None, early) is early
    updated = callback(fixture, 19, None, early)
    assert callback.called and not torch.equal(updated["latents"], early["latents"])
    with pytest.raises(RuntimeError, match="more than once"):
        callback(fixture, 19, None, early)
    pair = run_sd35_final_latent_pair(fixture, "a test prompt", "0123456789abcdef", height=256, width=256, generator=torch.Generator().manual_seed(7))
    assert pair.clean.mode == pair.marked.mode == "RGB"


@pytest.mark.integration
def test_runner_reuses_content_iss_loader_without_a_default_proxy_scorer() -> None:
    source = inspect.getsource(engine.run)
    assert "_load_pipeline_and_assets" in source
    assert "build_reused_weighted_joint_content_adapter" in source
    assert "content_detector" not in inspect.signature(engine.run).parameters
    assert "mean" not in source and "load_sd35_pipeline" not in source


@pytest.mark.integration
def test_cli_consumes_secrets_once_and_emits_no_secret(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    root, token = "a root secret that must not escape", "hf-token-secret"
    env = {"CEG_WM_ROOT_KEY": root, "HF_TOKEN": token, "OTHER": "ok"}
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(engine, "_checkout_state", lambda repo: ("a" * 40, "", True))
    monkeypatch.setattr("cegwm.runtime.content_weighted_joint_sd35.derive_stability_wrong_keys", lambda key: (b"wrong",))
    monkeypatch.setattr(engine, "run", lambda *args, **kwargs: calls.append(args) or tuple({"final_rgb": {"passed": True}} for _ in range(4)))
    code = engine.main(["--stage", "G0", "--repo-root", str(tmp_path), "--artifact-root", str(tmp_path / "out"), "--expected-exact", "a" * 40], environ=env)
    output = capsys.readouterr().out
    assert code == 0 and len(calls) == 1 and root not in output and token not in output
    assert "CEG_WM_ROOT_KEY" not in env and "HF_TOKEN" not in env


@pytest.mark.integration
def test_cli_rejects_bad_checkout_before_runner(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    monkeypatch.setattr(engine, "_checkout_state", lambda repo: ("b" * 40, "Geometry-V4", False))
    monkeypatch.setattr(engine, "run", lambda *args, **kwargs: pytest.fail("runner must not start"))
    assert engine.main(["--stage", "G0", "--repo-root", str(tmp_path), "--artifact-root", str(tmp_path / "out"), "--expected-exact", "a" * 40], environ={}) == 2
    assert "STOPPED" in capsys.readouterr().out


@pytest.mark.integration
def test_g1_summary_rejects_final_rgb_only_false_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    env = {"CEG_WM_ROOT_KEY": "root secret material", "HF_TOKEN": "hf secret"}
    monkeypatch.setattr(engine, "_checkout_state", lambda repo: ("a" * 40, "", True))
    monkeypatch.setattr("cegwm.runtime.content_weighted_joint_sd35.derive_stability_wrong_keys", lambda key: (b"wrong",))
    records = tuple({"final_rgb": {"passed": True}, "attacked_rgb": {"passed": False}} for _ in range(20))
    monkeypatch.setattr(engine, "run", lambda *args, **kwargs: records)
    code = engine.main(["--stage", "G1", "--repo-root", str(tmp_path), "--artifact-root", str(tmp_path / "out"), "--expected-exact", "a" * 40], environ=env)
    output = capsys.readouterr().out
    assert code == 2 and '"passed":0' in output and '"status":"GATE_FAILED"' in output


@pytest.mark.integration
def test_g1_attacked_gate_does_not_consume_paired_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Generator:
        def manual_seed(self, seed): return self

    image = Image.new("RGB", (32, 32), "gray")
    monkeypatch.setattr(engine.torch, "Generator", lambda **kwargs: _Generator())
    monkeypatch.setattr(engine, "run_sd35_final_latent_pair", lambda *args, **kwargs: SimpleNamespace(clean=image, marked=image))
    monkeypatch.setattr(engine, "_g1_attacked_record", lambda *args: {"passed": True})
    detector = SimpleNamespace(identities=lambda: {})
    passing = RGBObservability(50.0, .99, 0.0, 0.0, 4.0, -1.0, 0.0)
    failing = RGBObservability(10.0, .5, 1.0, 1.0, -1.0, 4.0, 1.0)
    monkeypatch.setattr(engine, "measure_final_rgb", lambda *args: passing)
    first = engine._record("G1", 1, "prompt", "identity", object(), b"correct", b"wrong", detector)
    monkeypatch.setattr(engine, "measure_final_rgb", lambda *args: failing)
    second = engine._record("G1", 1, "prompt", "identity", object(), b"correct", b"wrong", detector)
    assert first["final_rgb"]["passed"] is True and second["final_rgb"]["passed"] is False
    assert first["attacked_rgb"]["passed"] is second["attacked_rgb"]["passed"] is True
    monkeypatch.setattr(engine, "measure_final_rgb", lambda *args: (_ for _ in ()).throw(RuntimeError("diagnostic only")))
    third = engine._record("G1", 1, "prompt", "identity", object(), b"correct", b"wrong", detector)
    assert third["final_rgb"]["passed"] is False and "diagnostic_failure" in third["final_rgb"]
    assert third["failure"] is None and third["attacked_rgb"]["passed"] is True


@pytest.mark.integration
def test_g1_key_arms_never_share_h_or_rectified_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    correct_key, wrong_key = b"correct", b"wrong"
    correct_h = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    wrong_h = (1.0, 0.0, .2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    detect_calls: list[object] = []
    rectify_calls: list[tuple[float, ...]] = []
    score_calls: list[object] = []

    def detect(rgb, key):
        detect_calls.append(key)
        if key == correct_key:
            return {"status": "RELIABLE", "H_hat": correct_h, "corners_hat": ((0, 0), (1, 0), (1, 1), (0, 1)), "support": 6}
        return {"status": "UNRELIABLE", "H_hat": wrong_h, "corners_hat": (), "support": 0}

    monkeypatch.setattr(engine, "detect_g1_geometry", detect)
    monkeypatch.setattr(engine, "rectify_g1_rgb", lambda rgb, h: rectify_calls.append(h) or rgb)
    monkeypatch.setattr(engine, "rgb_only_anchor_score", lambda rgb, key: score_calls.append(key) or 3.0)
    result = engine._g1_attacked_record(np.full((32, 32, 3), .5), correct_key, wrong_key)
    assert result["passed"] is True
    assert detect_calls == [correct_key, wrong_key]
    assert rectify_calls == [correct_h]
    assert score_calls == [correct_key]
    assert "rectified_wrong_key_anchor" not in result
