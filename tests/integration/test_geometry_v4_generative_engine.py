from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

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
