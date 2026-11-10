"""Recovery accepts real adapter artifacts using only tiny CPU fake modules."""
import pytest
import torch

from cegwm.runtime.stage_protection_resume import RepairStore, REQUIRED, TARGET
from cegwm.runtime.stage_protection_resume import RepairStopped
from cegwm.runtime.channel_precompensation_sd35 import RuntimeStageError
from cegwm.runtime.stage_protection_store import atomic_json, atomic_torch
from cegwm.runtime.stage_protection_v2 import StageProtectionProvider, at_latent
from cegwm.stage_protection_v2 import cast_action, solve_intervals
from test_stage_protection_runtime_v2 import config
from test_channel_precompensation_provider import Loader
from test_channel_precompensation_runtime import VAE, Processor

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("compatibility", [False, True])
def test_actual_adapter_checkpoints_resume_vjp_without_source_regeneration(tmp_path, monkeypatch, compatibility):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request",
        lambda *a, **k: pytest.fail("CPU recovery test must never request the network"))
    c, raw = config(), {}
    provider = StageProtectionProvider(c, counters=raw, loader=Loader(),
                                       reader_loader=lambda _: (VAE(), Processor()))
    try:
        captured = provider.generate(c["sources"][0])
        zero = provider.render(captured[8], capture_late=True)
        clean = provider.read(zero["image"])
        design_image = provider.reconstruct(zero["image"], c["reconstruction"]["design_seed"])
        design_read = provider.read(design_image)
        sensitivity = provider.rgb_gradient(zero["image"], clean, design_image=design_image, design_read=design_read)
        E = provider.render(zero["late"])
        z = captured[8].before_state["latents"]
        gradients = torch.zeros(2, *z.shape)
        # These stored solve/action objects are structural fixtures, not claims
        # about a scientific trial. Source/render/read/RGB-VJP are real adapters.
        solve = solve_intervals(gradients, [0, 0], [0, 0], [.25, .25], 1.)
        action = cast_action(z, torch.zeros_like(z), z)
        for key in REQUIRED:
            leaf = key.rsplit("/", 1)[-1]
            if leaf == "source":
                value = captured
            elif leaf == "canonical_early_zero":
                value = zero
            elif leaf == "render":
                value = E
            elif leaf in ("clean", "design_read"):
                value = clean if leaf == "clean" else design_read
            elif leaf == "design_image":
                value = design_image
            elif leaf.endswith("rgb_vjp"):
                value = sensitivity
            elif leaf == "latent_vjp":
                value = dict(gradients=gradients)
            elif leaf == "solve":
                value = solve
            else:
                value = action
            folder = tmp_path / "operations" / key
            folder.mkdir(parents=True)
            atomic_json(folder / "status.json", dict(status="RETURNED", key=key, counts_after=dict(raw)))
            atomic_torch(folder / "result.pt", value)
    finally:
        provider.close()
    atomic_json(tmp_path / "config.json", c)
    atomic_json(tmp_path / "physical_counts.json", raw)
    target = tmp_path / "operations" / TARGET
    target.mkdir(parents=True)
    atomic_json(target / "status.json", dict(status="STARTED", key=TARGET, counts_before=dict(raw)))
    if compatibility:
        previous = RepairStore(tmp_path, c)
        def import_failure():
            previous.counters["source_pipeline_load_attempted"] += 1
            previous.counters["source_pipeline_load_returned"] += 1
            raise RuntimeStageError("source_suffix_vjp", ImportError(
                "cannot import name 'is_non_overlapping_and_dense' from 'torch._prims_common'"),
                {"before_state": captured[8].before_state}, previous.counters)
        with pytest.raises(RepairStopped):
            previous.call(TARGET, import_failure)
    store = RepairStore(tmp_path, c, compatibility=compatibility)
    loader = Loader()
    provider = StageProtectionProvider(c, counters=store.counters, loader=loader,
                                       reader_loader=lambda _: (VAE(), Processor()))
    try:
        for key in REQUIRED:
            store.call(key, lambda: pytest.fail("saved artifact must be reused"))
        actual = store.call(TARGET, lambda: provider.latent_gradients(
            at_latent(captured[8], zero["after"], zero["image"]), zero["image"], [sensitivity]))
        assert actual["gradients"].shape == (1, *z.shape)
        assert store.counters["sd35_full_generation_calls"] == 1
        cost = store.accounting()
        assert cost["retry_physical_counts"]["sd3_denoising_steps_attempted"] == 11
        assert cost["retry_physical_counts"]["sd3_denoising_steps_returned"] == 11
        assert cost["retry_physical_counts"]["sd3_forward_attempted"] == 1
        loaded = dict(loader.counts)
        store.call(TARGET, lambda: pytest.fail("returned VJP must not repeat"))
        assert loader.counts == loaded
    finally:
        provider.close()
