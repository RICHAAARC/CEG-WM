"""CPU-only recovery policy, checkpoint corruption and retained cost evidence."""
import json

from PIL import Image
import pytest
import torch

from cegwm.runtime.channel_precompensation_sd35 import SourceCapture, RGBSensitivity, RuntimeStageError
from cegwm.runtime.stage_protection_resume import (
    REPAIR, REQUIRED, TARGET, RepairStore, RepairStopped, PreflightBlocked, preflight,
)
from cegwm.runtime.stage_protection_store import atomic_json, atomic_torch, Unavailable
from experiments import run_stage_protection_v2_resume as runner
from experiments.run_stage_protection_v2 import load_config

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint_run(tmp_path):
    config = load_config()
    z = torch.ones(1, 3, 2, 2)
    image = Image.new("RGB", (2, 2), (127, 128, 129))
    def capture(step):
        return SourceCapture(image, dict(latents=z, prompt_embeds=torch.ones(1, 1, 2),
            pooled_prompt_embeds=torch.ones(1, 2), latent_device="cpu", scheduler_class="fake",
            joint_attention_kwargs=None, do_classifier_free_guidance=False, guidance_scale=7.,
            next_step_index=step + 1), dict(timesteps=torch.arange(20), sigmas=torch.arange(21)),
            tuple(range(step + 1, 20)))
    atomic_json(tmp_path / "config.json", config)
    counts = dict(sd35_transformer_attempted=46, sd35_transformer_returned=45,
                  source_pipeline_load_attempted=6, source_pipeline_load_returned=6,
                  sd35_suffix_forward_calls=7, sd35_suffix_forward_returned=6,
                  sd35_full_generation_calls=1, sd35_full_generation_returned=1)
    atomic_json(tmp_path / "physical_counts.json", counts)
    for key in REQUIRED:
        leaf = key.rsplit("/", 1)[-1]
        if leaf == "source":
            value = {step: capture(step) for step in (8, 18)}
        elif leaf in ("canonical_early_zero", "render"):
            value = dict(image=image, after=z, actual_delta=torch.zeros_like(z), late=capture(18))
        elif leaf in ("clean", "design_read"):
            value = dict(feature=torch.ones(3), score=0.1)
        elif leaf == "design_image":
            value = image
        elif leaf.endswith("rgb_vjp"):
            value = RGBSensitivity(image, torch.ones(3), 0.1, torch.ones_like(z))
        elif leaf == "latent_vjp":
            value = dict(gradients=torch.ones(2, *z.shape))
        elif leaf == "solve":
            value = dict(status="SOLVED", delta=torch.zeros_like(z))
        elif leaf == "action":
            value = dict(after=z, actual=torch.zeros_like(z), relative=0., within_budget=True, nonzero=False)
        folder = tmp_path / "operations" / key
        folder.mkdir(parents=True)
        atomic_json(folder / "status.json", dict(status="RETURNED", key=key, counts_after=counts))
        atomic_torch(folder / "result.pt", value)
    folder = tmp_path / "operations" / TARGET
    folder.mkdir(parents=True)
    atomic_json(folder / "status.json", dict(status="STARTED", key=TARGET,
        counts_before={**counts, "sd35_transformer_attempted": 36, "sd35_transformer_returned": 36}))
    return tmp_path, config


def retry_value():
    return dict(gradients=torch.ones(1, 1, 3, 2, 2))


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_original_cache_no_compute_precise_retry_once_and_costs(checkpoint_run):
    root, config = checkpoint_run
    old = snapshot(root)
    store = RepairStore(root, config)
    for key in REQUIRED:
        store.call(key, lambda: pytest.fail("completed checkpoint must not compute"))
    assert not (root / REPAIR / "operations" / TARGET / "status.json").exists()
    calls = []
    def retry():
        calls.append(TARGET)
        assert json.loads((store.operations / TARGET / "status.json").read_text())["status"] == "STARTED"
        store.counters["sd35_transformer_attempted"] += 11
        store.counters["sd35_transformer_returned"] += 11
        store.counters["sd35_suffix_forward_calls"] += 1
        store.counters["sd35_suffix_forward_returned"] += 1
        return retry_value()
    store.call(TARGET, retry)
    store = RepairStore(root, config)
    store.call(TARGET, lambda: pytest.fail("no second retry"))
    def later():
        store.counters["sd35_transformer_attempted"] += 20
        store.counters["sd35_transformer_returned"] += 20
        return Image.new("RGB", (2, 2))
    store.call("source07_coastal_scene/unmarked/design_image", later)
    cost = store.accounting()
    assert calls == [TARGET]
    assert cost["original_raw_counts"]["sd35_transformer_attempted"] == 46
    assert cost["retry_raw_counts"]["sd35_transformer_attempted"] == 11
    assert cost["later_continuation_raw_counts"]["sd35_transformer_attempted"] == 20
    assert cost["new_raw_counts"]["sd35_transformer_attempted"] == 31
    assert cost["cumulative_raw_counts"]["sd35_transformer_attempted"] == 77
    assert cost["original_physical_counts"]["sd3_denoising_steps_attempted"] == 46
    assert cost["retry_physical_counts"]["sd3_denoising_steps_attempted"] == 11
    assert cost["cumulative_physical_counts"]["sd3_denoising_steps_attempted"] == 77
    assert cost["cumulative_raw_counts"]["sd35_transformer_returned"] == 76
    assert cost["original_maxima_apply_to_cumulative"] is False
    assert {name: (root / name).read_bytes() for name in old} == old
    target_rows = [r for r in store.inventory() if r["key"] == TARGET]
    assert [r["status"] for r in target_rows] == ["STARTED", "RETURNED"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_or_interrupted_retry_consumes_authorization(checkpoint_run, failure):
    root, config = checkpoint_run
    store = RepairStore(root, config)
    def fail():
        store.counters["sd35_transformer_attempted"] += 1
        raise failure("synthetic interruption")
    with pytest.raises(RepairStopped if failure is RuntimeError else KeyboardInterrupt):
        store.call(TARGET, fail)
    status = json.loads((store.operations / TARGET / "status.json").read_text())
    assert status["status"] == ("FAILED" if failure is RuntimeError else "STARTED")
    assert store.accounting()["retry_raw_counts"]["sd35_transformer_attempted"] == 1
    with pytest.raises(PreflightBlocked, match="no second retry"):
        runner.execute(config, root, provider_factory=lambda *a, **k: pytest.fail("must block before provider"))
    with pytest.raises(RepairStopped):
        store.call(TARGET, lambda: pytest.fail("same object must not retry either"))


@pytest.mark.parametrize("damage", ["missing_source", "broken_rgb_vjp", "wrong_payload", "nonfinite", "missing_bit0"])
def test_preflight_blocks_missing_or_corrupt_prerequisites_before_provider(checkpoint_run, damage):
    root, config = checkpoint_run
    path = root / "operations" / (REQUIRED[0] if damage == "missing_source" else REQUIRED[-1]) / "result.pt"
    if damage in ("missing_source", "missing_bit0"):
        if damage == "missing_bit0":
            path = root / "operations" / (REQUIRED[12]) / "result.pt"
        path.unlink()
    elif damage == "broken_rgb_vjp":
        path.write_bytes(b"incomplete pickle")
    elif damage == "wrong_payload":
        atomic_torch(path, None)
    else:
        value = torch.load(path, weights_only=False)
        value.gradient_rgb[0, 0, 0, 0] = float("nan")
        atomic_torch(path, value)
    with pytest.raises(PreflightBlocked):
        runner.execute(config, root, provider_factory=lambda *a, **k: pytest.fail("preflight precedes models"))
    assert not (root / REPAIR).exists()


def test_general_started_failed_policy_is_unchanged(checkpoint_run):
    root, config = checkpoint_run
    store = RepairStore(root, config)
    for state in ("STARTED", "FAILED"):
        key = "source07_coastal_scene/" + state
        folder = store.operations / key
        folder.mkdir(parents=True)
        atomic_json(folder / "status.json", dict(status=state, key=key))
        with pytest.raises(Unavailable, match="no automatic retry"):
            store.call(key, lambda: pytest.fail("only the fixed target has a new attempt"))


def test_corrupt_returned_retry_is_blocked_without_recompute(checkpoint_run):
    root, config = checkpoint_run
    store = RepairStore(root, config)
    store.call(TARGET, retry_value)
    (store.operations / TARGET / "result.pt").write_bytes(b"broken")
    with pytest.raises(PreflightBlocked):
        RepairStore(root, config)


@pytest.mark.parametrize("compatibility", [False, True])
def test_continuation_runner_freezes_before_holdout_and_completed_rerun_no_provider(checkpoint_run, monkeypatch, compatibility):
    root, config = checkpoint_run
    if compatibility:
        previous = RepairStore(root, config)
        def import_failure():
            previous.counters["source_pipeline_load_attempted"] += 1
            previous.counters["source_pipeline_load_returned"] += 1
            raise RuntimeStageError("source_suffix_vjp", ImportError(
                "cannot import name 'is_non_overlapping_and_dense' from 'torch._prims_common'"),
                {"before_state": {}}, previous.counters)
        with pytest.raises(RepairStopped):
            previous.call(TARGET, import_failure)
    events = []
    class SmallCanary:
        # Orchestration test only. Fixed numerical runner has separate tests.
        def __init__(self, config, store, provider):
            self.store = store
        def freeze(self):
            for key in REQUIRED:
                self.store.call(key, lambda: pytest.fail("old result recomputed"))
            self.store.call(TARGET, retry_value)
            events.append("retry")
            frozen = dict(trials=[{}] * 4)
            atomic_torch(self.store.directory / "selection_frozen.pt", frozen)
            return frozen
        def evaluate(self, frozen):
            assert (self.store.directory / "selection_frozen.pt").exists()
            events.append("holdout_after_freeze")
            summary = dict(plan={}, trials=frozen["trials"], engineering_complete=True, verdict={})
            atomic_json(self.store.directory / "summary.json", summary)
            return summary
    monkeypatch.setattr(runner, "Canary", SmallCanary)
    class Provider:
        def __init__(self, *a, **k):
            events.append("provider")
        def close(self):
            events.append("closed")
    summary = runner.execute(config, root, provider_factory=Provider, compatibility=compatibility)
    assert events == ["provider", "retry", "holdout_after_freeze", "closed"]
    assert summary["continuation"]["retry_status"] == "RETURNED"
    resumed = runner.execute(config, root, compatibility=compatibility,
        provider_factory=lambda *a, **k: pytest.fail("cached summary loads no provider"))
    assert resumed == summary
    assert not (root / "summary.json").exists()  # Original evidence untouched.


def test_cached_late_solver_is_not_recomputed(tmp_path, monkeypatch):
    from test_stage_protection_runner_v2 import run
    from experiments import run_stage_protection_v2 as original_runner
    first, _ = run(tmp_path)
    for name in ("selection_frozen.pt", "selection_frozen.json"):
        (tmp_path / name).unlink()  # Synthetic interrupted-before-freeze scenario.
    monkeypatch.setattr(original_runner, "solve_intervals", lambda *a, **k: pytest.fail("cached solve must be used"))
    second, provider = run(tmp_path)
    assert second["trials"] == first["trials"]
    assert provider.log == []
