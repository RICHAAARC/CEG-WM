"""Real adapter with small multilayer CPU fakes. Never scientific evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from cegwm.runtime import channel_precompensation_sd35 as rt
from cegwm.runtime.stage_protection_v2 import StageProtectionProvider, at_latent, physical_counts
from experiments.run_stage_protection_v2 import execute, load_config
from test_channel_precompensation_provider import Loader, PublicReader
from test_channel_precompensation_runtime import VAE, Processor

pytestmark = pytest.mark.integration


def config():
    value = load_config()
    for name in ("generation", "reconstruction"):
        value[name].update(device="cpu", dtype="float32")
    value["generation"].update(height=2, width=2)
    return value


def test_dual_capture_canonical_early_new_basepoint_late_rebase_and_vjps():
    c, raw, loader = config(), {}, Loader()
    p = StageProtectionProvider(c, counters=raw, loader=loader)
    try:
        captured = p.generate(c["sources"][0])
        assert captured[8].remaining_steps == tuple(range(9, 20))
        assert captured[18].remaining_steps == (19,)
        assert captured[8].before_state["next_step_index"] == 9
        assert captured[18].before_state["next_step_index"] == 19
        early = p.render(captured[8], capture_late=True)
        read = p.read(early["image"])
        design_image = p.reconstruct(early["image"], c["reconstruction"]["design_seed"])
        design_read = p.read(design_image)
        sensitivity = p.rgb_gradient(early["image"], read, design_image=design_image, design_read=design_read)
        vjp = p.latent_gradients(captured[8], early["image"], [sensitivity])
        assert vjp["gradients"].shape == (1, 1, 3, 2, 2)
        assert len(vjp["diagnostics"]["steps"]) == 11
        delta = torch.full_like(captured[8].before_state["latents"], .03)
        changed = p.render(captured[8], delta, capture_late=True)
        assert not torch.equal(changed["late"].before_state["latents"], early["late"].before_state["latents"])
        E = p.render(changed["late"])
        E_read = p.read(E["image"])
        E_design_image = p.reconstruct(E["image"], c["reconstruction"]["design_seed"])
        E_design_read = p.read(E_design_image)
        clean = p.rgb_gradient(E["image"], E_read)
        design = p.rgb_gradient(E["image"], E_read, design_image=E_design_image, design_read=E_design_read)
        late_g = p.latent_gradients(changed["late"], E["image"], [clean, design])
        assert len(late_g["diagnostics"]["steps"]) == 1
        assert late_g["diagnostics"]["basepoint_rgb_comparison"]["exact_equal"]
        counts = physical_counts(raw)
        assert counts["sd3_forward_returned"] == 6
        assert counts["sd3_denoising_steps_returned"] == 20 + 11 * 3 + 1 * 2
        assert counts["rgb_vjp_returned"] == counts["latent_vjp_returned"] == 3
        assert counts["reader_encode_returned"] == 4 + 3 + 3
        assert counts["sdxl_forward_returned"] == 4
        assert counts["sdxl_denoising_steps_returned"] == 12  # Fake SDXL has 3 steps.
    finally:
        p.close()


def test_end_to_end_four_denominators_freeze_before_holdout_resume_no_model(tmp_path):
    c, loader = config(), Loader()
    class Provider(StageProtectionProvider):
        def __init__(self, spec, *, counters):
            super().__init__(spec, counters=counters, loader=loader,
                             reader_loader=lambda _: (VAE(), Processor()))
        def reconstruct(self, image, seed):
            if seed == c["reconstruction"]["holdout_seed"]:
                assert (tmp_path / "selection_frozen.pt").exists()
                assert len(torch.load(tmp_path / "selection_frozen.pt", weights_only=False)["trials"]) == 4
            return super().reconstruct(image, seed)
    summary = execute(c, tmp_path, provider_factory=Provider)
    assert len(summary["trials"]) == 4
    assert summary["verdict"]["content_denominator"] == 2
    assert summary["verdict"]["scientific_pass"] is None
    assert all(r["centers"] for r in summary["trials"])
    assert summary["raw_counts"]["sd35_full_generation_calls"] == 2
    assert summary["physical_counts"]["sd3_forward_attempted"] <= 40
    assert summary["physical_counts"]["reader_encode_attempted"] <= 102
    previous_counts = summary["physical_counts"]
    previous_loads = dict(loader.counts)
    resumed = execute(c, tmp_path, provider_factory=Provider)
    assert resumed["physical_counts"] == previous_counts
    assert loader.counts == previous_loads  # No model needed for fully cached output.
    assert resumed["resume_operation_reuse"]


def test_proxy_mismatch_is_engineering_failure_retains_four_trials(tmp_path):
    c, loader = config(), Loader(gradient_offset=.05)
    def factory(spec, *, counters):
        return StageProtectionProvider(spec, counters=counters, loader=loader,
                                       reader_loader=lambda _: (VAE(), Processor()))
    summary = execute(c, tmp_path, provider_factory=factory)
    assert len(summary["trials"]) == 4
    assert not summary["engineering_complete"]
    assert not summary["verdict"]["operational_P2_signal"]
    failures = [o for o in summary["operations"] if o["status"] == "FAILED"]
    assert any("proxy" in o["error"] for o in failures)
    assert summary["verdict"]["scientific_pass"] is None


def test_runtime_full_physical_envelope_includes_auxiliary_reader_encodes():
    """Exercise the V2 maximum operation schedule, independent of solver success.

    This checks nested *physical* accounting with actual autograd fake modules.
    Fake SDXL deliberately has three denoising steps, so its step total is half
    the six-step real-model plan. This is neither a scientific nor GPU run.
    """
    c, raw = config(), {}
    p = StageProtectionProvider(c, counters=raw, loader=Loader())
    holdouts = []
    def observe(image):
        clean = p.read(image)
        design_image = p.reconstruct(image, c["reconstruction"]["design_seed"])
        return image, clean, design_image, p.read(design_image)
    def design_vjp(o):
        return p.rgb_gradient(o[0], o[1], design_image=o[2], design_read=o[3])
    try:
        for source in c["sources"]:
            captures = p.generate(source)
            zero = p.render(captures[8], capture_late=True)
            baseline = observe(zero["image"])
            holdouts.append(zero["image"])
            for bit in (0, 1):
                current, observation = zero, baseline
                for update in (1, 2):
                    sensitivity = design_vjp(observation)
                    state = at_latent(captures[8], current["after"], observation[0])
                    p.latent_gradients(state, observation[0], [sensitivity])
                    delta = torch.full_like(captures[8].before_state["latents"], (2*bit-1)*update*.002)
                    current = p.render(captures[8], delta, capture_late=True)
                    observation = observe(current["image"])
                E = p.render(current["late"])
                eo = observe(E["image"])
                clean = p.rgb_gradient(eo[0], eo[1])
                design = design_vjp(eo)
                p.latent_gradients(current["late"], eo[0], [clean, design])
                holdouts.append(E["image"])
                for arm in (1, 2, 3):
                    delta = torch.full_like(current["late"].before_state["latents"], (2*bit-1)*arm*.002)
                    proposal = p.render(current["late"], delta)
                    observe(proposal["image"])
                    holdouts.append(proposal["image"])
        for image in holdouts:
            p.read(p.reconstruct(image, c["reconstruction"]["holdout_seed"]))
        counts = physical_counts(raw)
        expected = dict(sd3_forward=40, sd3_denoising_steps=258, sdxl_forward=56,
            sdxl_ordinary=44, sdxl_proxy=12, sdxl_denoising_steps=168,
            rgb_vjp=16, latent_vjp=16, reader_result_ordinary=70,
            reader_aux_ordinary=16, reader_proxy=16, reader_encode=102)
        for key, value in expected.items():
            assert counts[key + "_attempted"] == counts[key + "_returned"] == value, key
    finally:
        p.close()
