"""Public loader arguments and full/suffix lifecycle; no downloaded models."""
import gc
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from cegwm.runtime.stage_protection_v2 import StageProtectionProvider, TEXT_COMPONENTS, load_v2_source
from cegwm.runtime import channel_precompensation_sd35 as rt
from test_stage_protection_runtime_v2 import config
from test_channel_precompensation_provider import Loader
from test_channel_precompensation_runtime import Source, settings
from dataclasses import replace

pytestmark = pytest.mark.integration


def test_public_from_pretrained_overrides_preserve_revision_dtype_and_reader(monkeypatch):
    calls = []
    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return cls()
        def set_progress_bar_config(self, **kwargs):
            assert kwargs == {"disable": True}
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(StableDiffusion3Pipeline=Pipeline))
    c, reader = config(), object()
    load_v2_source(c, suffix_only=True, vae=reader)
    load_v2_source(c, suffix_only=False, vae=reader)
    load_v2_source(c, suffix_only=False)
    for args, kwargs in calls:
        assert args == (c["generation"]["model_id"],)
        assert kwargs["revision"] == c["generation"]["revision"]
        assert kwargs["torch_dtype"] == torch.float32
    assert all(calls[0][1][name] is None for name in TEXT_COMPONENTS)
    assert calls[0][1]["vae"] is calls[1][1]["vae"] is reader
    assert all(name not in calls[1][1] for name in TEXT_COMPONENTS)
    assert "vae" not in calls[2][1]


def test_full_suffix_reconstruction_suffix_new_source_and_reader_identity():
    c, raw, loader = config(), {}, Loader()
    source_calls = []
    def source_factory(spec, *, suffix_only, vae):
        source_calls.append((suffix_only, id(vae) if vae is not None else None))
        pipeline = loader(spec, "generation")
        if vae is not None:
            pipeline.vae = vae
        if suffix_only:
            for name in TEXT_COMPONENTS:
                if hasattr(pipeline, name):
                    delattr(pipeline, name)
        return pipeline
    p = StageProtectionProvider(c, counters=raw, loader=loader, source_loader=source_factory)
    try:
        captured = p.generate(c["sources"][0])
        reader_id = id(p._vae)
        text_refs = [weakref.ref(getattr(p._source, name)) for name in TEXT_COMPONENTS[:3]]
        zero = p.render(captured[8], capture_late=True)
        gc.collect()
        assert all(ref() is None for ref in text_refs)
        assert p._source_kind == "suffix"
        p.reconstruct(zero["image"], c["reconstruction"]["design_seed"])
        assert p._source is None
        p.render(captured[8])
        assert source_calls == [(False, None), (True, reader_id)]
        assert p._reconstruction is None and id(p._vae) == reader_id
        # Even without an intervening reconstruction, suffix-only cannot serve
        # the next complete source: full components must be loaded afresh.
        second = p.generate(c["sources"][1])
        assert source_calls[-1] == (False, reader_id)
        assert p._source_kind == "full" and second[8].remaining_steps == tuple(range(9, 20))
        assert raw["sd35_full_generation_calls"] == 2
        assert raw["sd35_suffix_forward_calls"] == 2
        assert id(p._source.vae) == id(p._vae) == reader_id
    finally:
        p.close()


def test_eleven_step_source_vjp_equivalence_counts_and_observation_boundaries():
    pipeline = Source()
    capture = rt.capture_source(pipeline, replace(settings(), injection_step=8))
    zero = rt.replay_suffix_write(pipeline, capture).image
    outputs, counters, diagnostics, events = [], [], [], []
    for reuse in (False, True):
        raw, diagnostic = {}, {}
        observer = (lambda event, **values: events.append((event, values))) if reuse else None
        outputs.append(rt.source_suffix_vjps(pipeline, capture, torch.ones(2, 1, 3, 2, 2),
            counters=raw, diagnostics=diagnostic, basepoint_mode="runtime_suffix", reference_image=zero,
            retain_stable_storage=reuse, resource_observer=observer))
        counters.append(raw)
        diagnostics.append(diagnostic)
    assert torch.equal(outputs[0], outputs[1])
    assert counters[0] == counters[1]
    assert counters[1]["sd35_suffix_forward_calls"] == 1
    assert counters[1]["sd35_suffix_denoising_steps"] == 11
    assert counters[1]["sd35_suffix_vjp_calls"] == 2
    assert diagnostics[0]["saved_tensors"]["stable_parameter_cpu_copies"] > 0
    assert diagnostics[1]["saved_tensors"]["stable_parameter_cpu_copies"] == 0
    assert [values["step"] for event, values in events if event == "before_step"] == list(range(9, 20))
    assert [event for event, _ in events].count("after_backward") == 2
    assert events[-1][1]["saved_tensors"]["activation_cpu_copy_bytes"] > 0
