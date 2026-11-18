"""CPU fixtures exercise engineering paths, never SD3.5 or scientific efficacy."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from cegwm.self_reference.adapter import Adapter, rgb8
from cegwm.self_reference.config import Config, ARMS, SHIFTS, SPLITS
from cegwm.self_reference.data import Cache, file_hash, json_write
from cegwm.self_reference.engine import (evaluate, load_checkpoint, measure, quality_report,
    select_checkpoints, stage_b_report, train)
from cegwm.self_reference.networks import Reader, Writer
from cegwm.self_reference.protocol import (candidates, digest, key_bits, reference, reference_target,
    relation, schedule, shift_rgb, stage_a, summarize_prediction)

pytestmark = pytest.mark.unit


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=.5, shift_factor=.125)

    def decode(self, z):
        return SimpleNamespace(sample=F.interpolate(z, scale_factor=4, mode="nearest") + self.anchor)

    def encode(self, pixels):
        value = F.avg_pool2d(pixels, 4) + self.anchor
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))


class FakeProcessor:
    def preprocess(self, image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float()/127.5-1


@pytest.fixture
def fixture(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = replace(Config(), width=4, latent_channels=3, rgb_size=32, downsample=4,
                  mask_border=2, updates=4, batch_size=2, checkpoint_every=2, bootstrap_samples=50)
    adapter = Adapter(FakeVAE(), FakeProcessor(), cfg)
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir/"fixture.txt").write_text("FakeVAE: not a model export")
    raw = {"schema": "self-reference-cache-v1", "config_hash": cfg.digest,
           "latent_convention": "final_z0_scaled_before_decode",
           "model": {"family": "SD3.5", "id": "CPU_FIXTURE_NOT_REAL_SD35", "revision": "fixture",
                     "denoiser_sha256": "0"*64, "vae_export": "vae",
                     "vae_files": {"fixture.txt": file_hash(vae_dir/"fixture.txt")}},
           "keys": {s: [f"{s}-key-{i}" for i in range(2)] for s in SPLITS}, "units": []}
    for si, split in enumerate(SPLITS):
        for i in range(2):
            unit = f"{split}-{i}"
            generator = torch.Generator().manual_seed(si*10+i)
            z = torch.randn((3, 8, 8), generator=generator)*.05
            np.save(tmp_path/f"{unit}.npy", z.numpy())
            clean = rgb8(adapter.decode(z[None]))[0].permute(1, 2, 0).numpy()
            Image.fromarray(clean).save(tmp_path/f"{unit}.png")
            raw["units"].append({"unit_id": unit, "split": split, "source_id": unit, "source_group": unit,
                "source_sha256": digest("fixture-source", unit).hex(), "key": raw["keys"][split][i],
                "reference_nonce": digest("fixture-independent-nonce", si*2+i).hex(),
                "latent": f"{unit}.npy", "latent_sha256": file_hash(tmp_path/f"{unit}.npy"),
                "clean": f"{unit}.png", "clean_sha256": file_hash(tmp_path/f"{unit}.png"),
                "provenance": {"generation_seed": si*10+i, "prompt_sha256": digest("fixture-prompt", unit).hex(),
                               "sampler": "FAKE", "steps": 1, "guidance_scale": 0}})
    manifest = tmp_path/"manifest.json"
    json_write(manifest, raw)
    yield cfg, adapter, Cache(manifest, cfg), tmp_path
    torch.set_num_threads(previous)


def test_reference_is_independent_and_relation_is_not_constant_bit_broadcast():
    p1, p2 = reference("nonce-1", 64), reference("nonce-2", 64)
    assert torch.equal(p1, reference("nonce-1", 64))
    assert abs(float(p1.mean())) < 1e-7
    assert torch.allclose(p1.square().mean(), torch.tensor(1.))
    assert not torch.equal(p1, p2)
    assert list(inspect.signature(reference).parameters) == ["nonce", "size"]
    bits = key_bits("key")[None, :, None, None]
    p = p1[None]
    r = relation(p, p*bits, 8)
    assert torch.allclose(r, bits[:, :, 0, 0], atol=1e-6)
    assert torch.equal(relation(torch.zeros_like(p), p*bits, 8), torch.zeros_like(r))
    assert not summarize_prediction(torch.zeros(32), "key")["top1"]
    assert summarize_prediction(key_bits("key"), "key")["exact"]
    # Same-sign fields with scale imbalance must not collapse to cosine=1.
    assert torch.allclose(relation(p, 2*p*bits, 8), .8*bits[:, :, 0, 0], atol=1e-6)
    keys = candidates("key")
    assert len(set(keys)) == len({tuple(key_bits(k).tolist()) for k in keys}) == 17


def test_reader_blind_locality_and_arm_distinctions(fixture):
    cfg, _, _, _ = fixture
    assert list(inspect.signature(Reader.forward).parameters) == ["self", "observation"]
    z = torch.randn(1, 3, 8, 8, requires_grad=True)
    reader = Reader("C", cfg)
    output = reader(z)
    output["q"][0, 0, 4, 4].backward()
    assert torch.equal(z.grad[..., 0, :], torch.zeros_like(z.grad[..., 0, :]))
    assert all(not isinstance(layer, (torch.nn.MultiheadAttention, torch.nn.AdaptiveAvgPool2d)) for layer in reader.modules())
    assert Reader("B0", cfg).reference is None
    assert Reader("B1", cfg).reference is not None
    b, p = key_bits("test")[None], reference("r", 8)[None]
    with pytest.raises(ValueError, match="no reference"):
        Writer("B0", cfg)(z, b, p)
    for arm in ARMS:
        writer = Writer(arm, cfg)
        assert float((writer(z, b, None if arm == "B0" else p)-z).abs().max()) <= cfg.residual_cap
    assert schedule(cfg.seed, 3, 10, 4) == schedule(cfg.seed, 3, 10, 4)
    # Reference draw stays the same even when source population changes.
    assert [r[2] for r in schedule(cfg.seed, 3, 10, 4)] == [r[2] for r in schedule(cfg.seed, 3, 20, 4)]


@pytest.mark.parametrize("shift", SHIFTS)
def test_rgb_integer_shift_and_supervision_coordinates(shift):
    x = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
    x[:, :, 12, 15] = 137
    dx, dy = shift
    y = shift_rgb(x, dx, dy)
    assert torch.equal(y[:, :, 12+dy, 15+dx], torch.full((1, 3), 137, dtype=torch.uint8))
    assert int(y.sum()) == 3*137 and set(y.unique().tolist()) == {0, 137}
    p = torch.arange(64*64).reshape(1, 1, 64, 64).float()
    target = reference_target(p, dx, dy, 8)
    assert torch.allclose(target[..., 8:-8, 8:-8], p[..., 8:-8, 8:-8] - dx/8 - dy/8*64, atol=.001)
    with pytest.raises(TypeError):
        shift_rgb(x, .5, 0)


def test_adapter_matches_formal_numbers_but_has_different_gradient(fixture):
    cfg, adapter, _, _ = fixture
    z = torch.randn(2, 3, 8, 8)*.05
    z.requires_grad_(True)
    pixels = adapter.quantize_ste(adapter.decode(z))
    proxy = adapter.encode_proxy(pixels)
    formal = adapter.observe(rgb8(pixels))
    assert torch.allclose(proxy, formal, atol=2e-7, rtol=0)
    assert proxy.requires_grad and not formal.requires_grad
    proxy.sum().backward()
    assert float(z.grad.abs().sum()) > 0
    assert all(not p.requires_grad and p.grad is None for p in adapter.vae.parameters())
    with pytest.raises(ValueError, match="finite"):
        rgb8(torch.full((1, 3, 32, 32), float("nan")))
    with pytest.raises(ValueError, match="shape"):
        adapter.decode(torch.zeros(1, 3, 4, 4))
    class BadProcessor(FakeProcessor):
        def preprocess(self, image):
            return (super().preprocess(image)+1)/2
    with pytest.raises(ValueError, match="unsupported processor"):
        Adapter(FakeVAE(), BadProcessor(), cfg)


@pytest.mark.parametrize("field", ["source_id", "source_group", "source_sha256", "latent_sha256", "clean_sha256", "reference_nonce"])
def test_manifest_rejects_split_and_nonce_leakage(fixture, field):
    cfg, _, cache, root = fixture
    raw = cache.raw
    raw["units"][2][field] = raw["units"][0][field]
    json_write(root/"invalid.json", raw)
    with pytest.raises(ValueError):
        Cache(root/"invalid.json", cfg, verify_files=False)


def test_manifest_key_and_file_integrity(fixture):
    cfg, _, cache, root = fixture
    raw = json.loads(cache.path.read_text())
    raw["keys"]["dev"][0] = raw["keys"]["train"][0]
    json_write(root/"invalid.json", raw)
    with pytest.raises(ValueError, match="key leakage"):
        Cache(root/"invalid.json", cfg, verify_files=False)
    (root/"vae"/"fixture.txt").write_text("mutated")
    with pytest.raises(ValueError, match="VAE hash"):
        Cache(cache.path, cfg)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_exact_resume_optimizer_rng_and_inputs(fixture):
    cfg, adapter, cache, root = fixture
    full = train(cache, cfg, adapter, "C", root/"full")
    interrupted = train(cache, cfg, adapter, "C", root/"resumed", stop_after=1)
    # Resume restores RNG despite caller consuming random values.
    torch.rand(21)
    np.random.rand(4)
    resumed = train(cache, cfg, adapter, "C", root/"resumed", resume=interrupted)
    a = load_checkpoint(full, cache, cfg, "C", "cpu")
    b = load_checkpoint(resumed, cache, cfg, "C", "cpu")
    for name in ("writer", "reader", "optimizer", "rng", "last_input_signature", "next_input_signature", "loss"):
        assert_nested_equal(a[name], b[name])
    with pytest.raises(ValueError, match="config_hash"):
        load_checkpoint(full, cache, replace(cfg, seed=1), "C", "cpu")


def test_gates_keep_independent_arm_failures_and_quality_missing():
    cfg = Config()
    gate = stage_a({"B0": .98, "B1": .2, "C": .6}, True, cfg)
    assert gate["status"] == "stop_C_v1" and gate["B"] == "skipped"
    assert gate["arm_pass"] == {"B0": True, "B1": False, "C": False}
    assert stage_a(dict.fromkeys(ARMS, .1), True, cfg)["status"] == "implementation_budget_optimization_unresolved"
    assert stage_a(dict.fromkeys(ARMS, 1.), False, cfg)["status"] == "engineering_error"
    assert stage_a({"B0": .98, "B1": .6, "C": .96}, True, cfg)["pass"]
    records = {a: [{"unit_id": "u", "psnr": 35., "lpips": None}] for a in ARMS}
    assert quality_report(records, cfg)["status"] == "not_evaluable"
    for rows in records.values():
        rows[0]["lpips"] = .02
    assert quality_report(records, cfg)["status"] == "pass_pilot_budget"
    records["C"][0]["psnr"] = 34.4
    assert quality_report(records, cfg)["status"] == "fail_pilot_budget"


def test_cli_train_resume_select_evaluate_and_diagnostic(fixture, monkeypatch, capsys):
    from cegwm.self_reference import cli, adapter as adapter_module, engine
    cfg, fake, cache, root = fixture
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(adapter_module, "load_adapter", lambda *args: fake)
    common = ["--manifest", str(cache.path), "--execute"]
    run = str(root/"run")
    for arm in ARMS:
        cli.main(["train", *common, "--run-dir", run, "--arm", arm, "--stop-after", "1"])
        cli.main(["train", *common, "--run-dir", run, "--arm", arm,
                  "--resume", str(root/"run"/arm/"update_000001.pt")])
    selection = str(root/"selection.json")
    original_measure = engine.measure
    calls = []
    def spy_measure(*args, **kwargs):
        calls.append((args[5], tuple(args[6])))
        return original_measure(*args, **kwargs)
    monkeypatch.setattr(engine, "measure", spy_measure)
    cli.main(["select", *common, "--run-dir", run, "--output", selection])
    assert calls and all(split == "model_selection" and shifts == ((0, 0),) for split, shifts in calls)
    calls.clear()
    dev = cli.main(["evaluate", *common, "--selection", selection, "--output", str(root/"dev.json")])
    assert all(dev["A"][arm]["denominator"] == 2 for arm in ARMS)
    assert dev["quality"]["status"] == "not_evaluable"
    assert dev["B"]["status"] == "skipped_by_dev_A_gate"  # Tiny random fixture is not writable evidence.
    assert len(calls) == 3 and all(shifts == ((0, 0),) for _, shifts in calls)
    calls.clear()
    final = cli.main(["evaluate", *common, "--selection", selection, "--split", "test",
                      "--dev-report", str(root/"dev.json"), "--output", str(root/"test.json")])
    assert final["A_gate"]["status"] == "descriptive_only_test_not_for_decisions"
    assert len(calls) == 3 and all(split == "test" and shifts == ((0, 0),) for split, shifts in calls)
    unavailable = cli.main(["evaluate", *common, "--selection", selection,
                            "--lpips-weights", str(root/"missing-metric.pt"), "--output", str(root/"missing_metric.json")])
    assert unavailable["lpips"]["status"] == "unavailable" and "FileNotFoundError" in unavailable["lpips"]["error"]
    assert unavailable["quality"]["status"] == "not_evaluable"
    diagnostic = cli.main(["diagnose", *common, "--selection", selection, "--output", str(root/"diagnose.json")])
    assert len(diagnostic["records"]) == 12
    # Same input schedule across all three independent optimizer states.
    states = [load_checkpoint(root/"run"/a/"update_000004.pt", cache, cfg, a, "cpu") for a in ARMS]
    assert len({s["last_input_signature"] for s in states}) == 1
    assert len({s["next_input_signature"] for s in states}) == 1
    # Exercise B wiring under a deliberately injected synthetic gate, never evidence.
    monkeypatch.setattr(engine, "stage_a", lambda *args: {"pass": True, "status": "synthetic_test_gate"})
    calls.clear()
    forced = engine.evaluate(cache, cfg, fake, selection, "dev", root/"synthetic_B.json")
    assert sum(shifts == tuple(SHIFTS[1:]) for _, shifts in calls) == 3
    assert all(len(forced["B_records"][arm]) == 8 for arm in ARMS)
    assert forced["research_status"] == "not_evaluable"  # Missing LPIPS never becomes PASS.
    capsys.readouterr()


def test_selection_rejects_unequal_budget_and_missing_opportunities(fixture):
    cfg, adapter, cache, root = fixture
    for arm in ARMS:
        train(cache, cfg, adapter, arm, root/"run"/arm, stop_after=2 if arm == "B1" else None)
    with pytest.raises(ValueError, match="complete budget"):
        select_checkpoints(cache, cfg, adapter, root/"run", root/"selection.json")
    assert not (root/"selection.json").exists()


def test_fixed_denominator_on_bad_clean_and_all_four_source_pairing(fixture):
    cfg, adapter, cache, root = fixture
    checkpoint = train(cache, cfg, adapter, "C", root/"run")
    row = cache.split("dev")[0]
    Image.new("RGB", (32, 32), (0, 0, 0)).save(root/row["clean"])
    result = measure(cache, cfg, adapter, "C", checkpoint, "dev", SHIFTS[1:])
    assert len(result["records"]) == 8
    failed = [r for r in result["records"] if r["unit_id"] == row["unit_id"]]
    assert len(failed) == 4 and all(r["status"] == "engineering_error" and not r["top1"] for r in failed)
    by_arm = {a: [dict(r) for r in result["records"]] for a in ARMS}
    report = stage_b_report(by_arm, cfg)
    assert report["source_denominator"] == 2 and not report["complete"] and not report["gain_target_met"]


def test_plan_only_never_loads_a_model(monkeypatch, capsys):
    from cegwm.self_reference import cli, adapter
    def forbidden(*args, **kwargs):
        raise AssertionError("model load in plan mode")
    monkeypatch.setattr(adapter, "load_adapter", forbidden)
    result = cli.main(["train"])
    assert result["mode"] == "plan_only"
    capsys.readouterr()


def test_nan_decode_is_engineering_error_with_fixed_denominator(fixture, monkeypatch):
    cfg, adapter, cache, root = fixture
    checkpoint = train(cache, cfg, adapter, "C", root/"run")
    monkeypatch.setattr(adapter.vae, "decode", lambda z: SimpleNamespace(sample=torch.full((z.shape[0], 3, 32, 32), float("nan"))))
    result = measure(cache, cfg, adapter, "C", checkpoint, "dev", SHIFTS)
    assert len(result["records"]) == 10
    assert all(r["status"] == "engineering_error" and not r["top1"] for r in result["records"])


def test_metric_failure_does_not_erase_A_measurements(fixture):
    cfg, adapter, cache, root = fixture
    checkpoint = train(cache, cfg, adapter, "C", root/"run")
    def broken_metric(*args):
        raise RuntimeError("synthetic metric failure")
    result = measure(cache, cfg, adapter, "C", checkpoint, "dev", [SHIFTS[0]], metric=broken_metric)
    assert len(result["records"]) == 2
    assert all(r["status"] != "engineering_error" and r["lpips"] is None and r["scores"] is not None
               and "synthetic metric failure" in r["quality_error"] for r in result["records"])


def test_cuda_environment_rejected_before_model_load(fixture, monkeypatch):
    from cegwm.self_reference import cli, adapter
    cfg, _, cache, root = fixture
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    calls = []
    monkeypatch.setattr(adapter, "load_adapter", lambda *args: calls.append(args))
    for value in (None, ":16:8", "invalid"):
        if value is None:
            monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        else:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
        with pytest.raises(ValueError, match="before starting Python"):
            cli.main(["train", "--manifest", str(cache.path), "--run-dir", str(root/"cuda"), "--device", "cuda", "--execute"])
    assert calls == []
