"""Single-source provider with the actual runtime and small CPU fake models."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from cegwm.channel_precompensation import ReaderSpec, make_reader
from cegwm.runtime import channel_precompensation_sd35 as runtime
from cegwm.runtime.channel_precompensation_single_source import RuntimeSuffixSingleSourceProvider

from test_channel_precompensation_provider import Loader, PublicReader, config, evaluations
from test_channel_precompensation_runtime import Source


pytestmark = pytest.mark.integration


class UnusableOldCache(dict):
    def __getitem__(self, key):
        raise AssertionError("the different historical basepoint must not supply a cache")

    def get(self, *args, **kwargs):
        raise AssertionError("the different historical basepoint must not supply a cache")


def saved_capture(specification):
    source, generation = specification["sources"][0], specification["generation"]
    settings = runtime.GenerationSettings(prompt=source["prompt"], seed=source["seed"],
        relative_l2=specification["write"]["relative_l2_budget"],
        **{key: generation[key] for key in
           ("injection_step", "guidance_scale", "height", "width", "num_inference_steps", "device")})
    captured = runtime.capture_source(Source(), settings)
    historical_image = Image.new("RGB", (2, 2), (0, 0, 0))
    vector, phase = make_reader(12, ReaderSpec(**specification["reader"]))
    return SimpleNamespace(
        baselines={source["source_id"]: {"image": historical_image,
            "before_state": captured.before_state, "scheduler_state": captured.scheduler_state,
            "remaining_steps": captured.remaining_steps}},
        reader_vector=vector, reader_phase=phase,
        rgb_gradients=UnusableOldCache(), zero_reconstructions=UnusableOldCache())


def forbid_full_generation(*args, **kwargs):
    raise AssertionError("the single-source method must never regenerate the full prefix")


def requests(baseline):
    return [{"source_id": baseline["source_id"], "method": method, "bit": bit,
             "delta": torch.full_like(baseline["before_state"]["latents"],
                                      1e-12 if bit == 0 else sign * 0.005)}
            for method, sign in (("joint", 1), ("gradient", -1)) for bit in (0, 1)]


def test_new_zero_real_runtime_vjps_four_suffix_writes_and_fresh_reconstructions(monkeypatch):
    specification = config()
    saved = saved_capture(specification)
    monkeypatch.setattr(Source, "__call__", forbid_full_generation)
    loader, reader = Loader(), PublicReader()
    provider = RuntimeSuffixSingleSourceProvider(specification, loader=loader,
        public_reader=reader, resume_checkpoint=saved)
    assert not loader.log
    try:
        preparation = list(provider.prepare_sources(specification["sources"][:1]))
        assert [row["kind"] for row in preparation] == [
            "baseline", "rgb_gradient", "design", "rgb_gradient", "linearization"]
        zero = preparation[0]
        assert not np.array_equal(np.asarray(zero["image"]),
                                  np.asarray(saved.baselines[zero["source_id"]]["image"]))
        assert zero["capture_reused"] and not zero["reused"]
        assert zero["diagnostics"]["old_rgb_gradients_used"] is False
        gradients = [row for row in preparation if row["kind"] == "rgb_gradient"]
        assert [(row["phase"], row["reused"], row["reusable"]) for row in gradients] == [
            ("clean", False, False), ("design", False, False)]
        linearization = preparation[-1]
        assert linearization["gradients"].shape == (2, 1, 3, 2, 2)
        assert torch.isfinite(linearization["gradients"]).all()
        assert torch.count_nonzero(linearization["gradients"])
        diagnostics = linearization["diagnostics"]["latent_sensitivity"]
        assert diagnostics["basepoint_mode"] == "runtime_suffix"
        assert diagnostics["basepoint_rgb_comparison"]["exact_equal"]
        assert not diagnostics["zero_rgb_comparison"]["exact_equal"]

        marked = list(provider.generate_writes(requests(zero)))
        assert len(marked) == 4
        assert [(row["method"], row["bit"]) for row in marked] == [
            ("joint", 0), ("joint", 1), ("gradient", 0), ("gradient", 1)]
        for row in marked:
            assert row["status"] == "OBSERVED" and row["matched_state"]
            assert row["feature"].shape == (12,)
            assert row["image"].mode == "RGB"
            assert torch.equal(row["actual_delta"],
                row["after_latents"].double() - zero["before_state"]["latents"].double())
            if row["bit"] == 0:
                assert torch.count_nonzero(row["actual_delta"]) == 0
                assert row["injection_metrics"]["requested_l2"] > 0
            else:
                assert torch.count_nonzero(row["actual_delta"]) > 0
        observed = list(provider.reconstruct_outputs(evaluations(preparation, marked, specification)))
        assert len(observed) == 9
        assert all(row["status"] == "OBSERVED" and not row["reused"] for row in observed)
        assert all(row["image"].mode == "RGB" and row["feature"].shape == (12,) for row in observed)
        counts = provider.counts
        for prefix, value in (("final_generation", 0), ("final_reconstruction", 10),
                              ("reconstruction", 11), ("public_feature_reads", 15),
                              ("rgb_vjps", 2), ("latent_vjps", 2), ("autograd_calls", 4)):
            assert counts[prefix + "_attempted"] == counts[prefix + "_returned"] == value
        assert counts["sd35_suffix_write_calls"] == counts["sd35_suffix_write_returned"] == 5
        assert counts["sd35_suffix_forward_calls"] == counts["sd35_suffix_forward_returned"] == 6
        assert counts["sd35_suffix_denoising_steps"] == 6
        assert counts.get("sd35_full_denoising_steps", 0) == 0
        assert loader.counts == {"generation": 2, "reconstruction": 2}
        assert [event[1] for event in loader.log if event[0] == "load"] == [
            "generation", "reconstruction", "generation", "reconstruction"]
        assert len({call[:2] for call in reader.calls}) == 1
        assert len(reader.calls) == 15
    finally:
        provider.close()


def test_current_design_mismatch_is_retained_and_never_uses_old_cache(monkeypatch):
    specification = config()
    saved = saved_capture(specification)
    monkeypatch.setattr(Source, "__call__", forbid_full_generation)
    loader = Loader(gradient_offset=2 / 255)
    provider = RuntimeSuffixSingleSourceProvider(specification, loader=loader, resume_checkpoint=saved)
    try:
        events = list(provider.prepare_sources(specification["sources"][:1]))
        assert events[-1]["kind"] == "failure" and events[-1]["stage"] == "design_rgb_vjp"
        assert not any(row["kind"] == "linearization" for row in events)
        assert len([row for row in events if row["kind"] == "rgb_gradient"]) == 2
        assert provider.counts["rgb_vjps_returned"] == 2
        assert provider.counts["latent_vjps_attempted"] == provider.counts["final_generation_attempted"] == 0
        assert loader.counts == {"generation": 1, "reconstruction": 1}
        assert list(provider.generate_writes([])) == []
    finally:
        provider.close()


def test_failed_public_read_preserves_successful_suffix_and_actual_delta(monkeypatch):
    specification = config()
    saved = saved_capture(specification)
    monkeypatch.setattr(Source, "__call__", forbid_full_generation)
    provider = RuntimeSuffixSingleSourceProvider(specification, loader=Loader(),
        public_reader=PublicReader(fail_at=3), resume_checkpoint=saved)
    try:
        preparation = list(provider.prepare_sources(specification["sources"][:1]))
        baseline = preparation[0]
        rows = list(provider.generate_writes(requests(baseline)[1:2]))
        assert len(rows) == 1 and rows[0]["status"] == "FAILED"
        row = rows[0]
        assert "public read failed" in row["error"]
        assert row["matched_state"] is True
        assert row["image"].mode == "RGB"
        assert torch.count_nonzero(row["actual_delta"]) > 0
        assert torch.equal(row["actual_delta"],
            row["after_latents"].double() - baseline["before_state"]["latents"].double())
        assert provider.counts["sd35_suffix_write_returned"] == 2
        assert provider.counts["public_feature_reads_attempted"] == 3
        assert provider.counts["public_feature_reads_returned"] == 2
        assert provider.counts["final_generation_attempted"] == 0
    finally:
        provider.close()
