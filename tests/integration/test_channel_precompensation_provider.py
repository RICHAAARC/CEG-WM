"""Real provider + real runtime, CPU fake model lifecycle and failure coverage."""

import gc
import json
from pathlib import Path
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cegwm.runtime import channel_precompensation_sd35 as runtime
from cegwm.runtime.channel_precompensation_provider import PrecompensationProvider
from cegwm.channel_precompensation import ReaderSpec, make_reader
from cegwm.runtime.reconstruction_channel_sd35 import final_image_feature

# Reuse the actual multilayer differentiable fakes, rather than stubbing VJPs.
from test_channel_precompensation_runtime import Source, Reconstruction, VAE, Processor


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def config(count=4):
    value = json.loads((ROOT / "configs/channel_precompensation_v1_colab.json").read_text())
    value["sources"] = value["sources"][:count]
    for name in ("generation", "reconstruction"):
        value[name]["device"] = "cpu"
        value[name]["dtype"] = "float32"
        value[name]["memory_mode"] = "model_cpu_offload"
    value["generation"].update(height=2, width=2)
    return value


class TextEncoder(torch.nn.Linear):
    def __init__(self, log, label):
        super().__init__(1, 1)
        self.log, self.label = log, label

    def to(self, *args, **kwargs):
        self.log.append(("text_to", self.label, str(args[0]) if args else str(kwargs)))
        return super().to(*args, **kwargs)


class Policy:
    @property
    def components(self):
        names = ("vae", "transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3")
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def to(self, device):
        self.log.append(("pipeline_to", self.label, str(device)))
        for module in self.components.values():
            module.to(device)
        return self

    def remove_all_hooks(self):
        self.log.append(("remove_hooks", self.label))
        for module in self.components.values():
            for child in module.modules():
                if hasattr(child, "_hf_hook"):
                    del child._hf_hook
        self._all_hooks = []

    def enable_model_cpu_offload(self, *, device):
        self.log.append(("enable_offload", self.label, str(device)))
        for module in self.components.values():
            module._hf_hook = object()
        self._all_hooks = [object()]


class TrackedSource(Policy, Source):
    def __init__(self, log, label):
        super().__init__()
        self.log, self.label = log, label
        self.text_encoder = TextEncoder(log, label + "/text1")
        self.text_encoder_2 = TextEncoder(log, label + "/text2")
        self.text_encoder_3 = TextEncoder(log, label + "/text3")

        def check_suffix(module, args):
            if torch.is_grad_enabled():
                log.append(("source_gradient", label))
                assert not hasattr(module, "_hf_hook")
                assert not module.training
                assert not any(p.requires_grad for p in module.parameters())

        self.transformer.register_forward_pre_hook(check_suffix)


class TrackedReconstruction(Policy, Reconstruction):
    def __init__(self, log, label):
        super().__init__()
        self.log, self.label = log, label
        self.gradient_offset = 0.0
        self.text_encoder, self.text_encoder_2 = TextEncoder(log, label + "/text1"), TextEncoder(log, label + "/text2")

    @torch.no_grad()
    def __call__(self, **kwargs):
        grad = torch.is_grad_enabled()
        self.log.append(("reconstruction_call", self.label, kwargs["generator"].initial_seed(), grad, kwargs["output_type"]))
        if grad:
            assert not self._all_hooks
            assert all(not p.requires_grad for module in self.components.values() for p in module.parameters())
        # Delegate to the fake's complete input-posterior/denoise/decode body.
        # Calling its decorated entry here would silently lose the graph.
        result = Reconstruction.__call__.__wrapped__(self, **kwargs)
        if grad and self.gradient_offset:
            result.images = (result.images + self.gradient_offset).clamp(0, 1)
        return result


class Loader:
    def __init__(self, *, gradient_offset=0.0):
        self.log, self.references, self.counts = [], [], {"generation": 0, "reconstruction": 0}
        self.gradient_offset = gradient_offset

    def __call__(self, specification, name):
        gc.collect()
        other = "reconstruction" if name == "generation" else "generation"
        assert not any(kind == other and reference() is not None for kind, reference in self.references), "large pipelines overlap"
        self.counts[name] += 1
        label = f"{name}{self.counts[name]}"
        self.log.append(("load", name, label))
        pipeline = TrackedSource(self.log, label) if name == "generation" else TrackedReconstruction(self.log, label)
        if name == "reconstruction":
            pipeline.gradient_offset = self.gradient_offset
        self.references.append((name, weakref.ref(pipeline)))
        # The real loader applies inference policy before returning the pipeline.
        pipeline.enable_model_cpu_offload(device=specification[name]["device"])
        return pipeline


class PublicReader:
    def __init__(self, *, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def __call__(self, image, *, vae, image_processor):
        self.calls.append((id(vae), id(image_processor), image.mode))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("public read failed")
        return final_image_feature(image, vae=vae, image_processor=image_processor)


def writes(preparation):
    result = []
    for event in preparation:
        if event["kind"] == "baseline":
            for method in ("joint", "gradient"):
                for bit in (0, 1):
                    result.append({"source_id": event["source_id"], "method": method, "bit": bit,
                                   "delta": torch.full_like(event["before_state"]["latents"], (bit*2-1)*1e-5)})
    return result


def evaluations(preparation, marked, specification):
    design, holdout = (specification["reconstruction"][name] for name in ("design_seed", "holdout_seed"))
    result = [{"source_id": event["source_id"], "method": "zero", "bit": None,
               "seed": holdout, "image": event["image"]} for event in preparation if event["kind"] == "baseline"]
    for event in marked:
        for seed in (design, holdout):
            result.append({**{name: event[name] for name in ("source_id", "method", "bit", "image")}, "seed": seed})
    return result


def test_constructor_is_inert_and_full_real_provider_has_fixed_counts_and_staging():
    loader, reader, specification = Loader(), PublicReader(), config()
    provider = PrecompensationProvider(specification, loader=loader, public_reader=reader)
    assert not loader.log and provider._reader_vector is None
    assert provider.counts["final_generation_attempted"] == 0
    preparation = list(provider.prepare_sources(specification["sources"]))
    assert [event["kind"] for event in preparation if event["kind"] != "rgb_gradient"] == ["baseline"]*4 + ["design"]*4 + ["linearization"]*4
    saved_gradients = [event for event in preparation if event["kind"] == "rgb_gradient"]
    assert len(saved_gradients) == 8 and all(event["reusable"] and not event["reused"] for event in saved_gradients)
    assert all(event["result"]["gradient_rgb"].device.type == "cpu" for event in saved_gradients)
    original_vae, original_processor = id(provider._vae), id(provider._processor)
    assert id(provider._source.vae) == original_vae
    for event in preparation:
        if event["kind"] == "linearization":
            assert event["gradients"].shape == (2, 1, 3, 2, 2)
            assert torch.isfinite(event["gradients"]).all()
    marked = list(provider.generate_writes(writes(preparation)))
    assert len(marked) == 16 and all(event["status"] == "OBSERVED" for event in marked)
    assert provider._source is None
    observed = list(provider.reconstruct_outputs(evaluations(preparation, marked, specification)))
    assert len(observed) == 36 and all(event["status"] == "OBSERVED" for event in observed)
    counts = provider.counts
    for name, expected in (("final_generation", 20), ("final_reconstruction", 40), ("reconstruction", 44),
                           ("auxiliary_reconstruction_forward", 4), ("public_feature_reads", 60),
                           ("latent_vjps", 8), ("rgb_vjps", 8), ("autograd_calls", 16)):
        assert counts[name + "_attempted"] == counts[name + "_returned"] == expected
    assert counts["auxiliary_ordinary_reader_attempted"] == counts["auxiliary_ordinary_reader_returned"] == 8
    assert counts["reader_proxy_forward_calls"] == 8
    assert counts["sd35_suffix_forward_calls"] == 4
    assert counts["sdxl_input_vae_encode_attempted"] == 4
    assert counts["sdxl_unet_forward_attempted"] == 12  # three real fake layers per design path
    assert counts["sdxl_output_vae_decode_attempted"] == 4
    assert counts["sd35_full_denoising_steps"] == 400
    assert all(call == (original_vae, original_processor, "RGB") for call in reader.calls)
    assert loader.counts == {"generation": 2, "reconstruction": 2}
    assert [event[1] for event in loader.log if event[0] == "load"] == ["generation", "reconstruction", "generation", "reconstruction"]
    design_calls = [event for event in loader.log if event[0] == "reconstruction_call" and event[3]]
    assert len(design_calls) == 4
    assert {event[2] for event in design_calls} == {specification["reconstruction"]["design_seed"]}
    assert all(event[4] == "pt" for event in design_calls)
    evaluation_calls = [event for event in loader.log if event[0] == "reconstruction_call" and not event[3]]
    assert len(evaluation_calls) == 40 and all(event[4] == "pil" for event in evaluation_calls)
    assert all(event[2] == specification["reconstruction"]["design_seed"] for event in evaluation_calls[:4])
    suffix_position = next(i for i, event in enumerate(loader.log) if event[0] == "source_gradient")
    assert all(("text_to", f"generation2/text{index}", "cpu") in loader.log[:suffix_position] for index in (1,2,3))
    provider.close()
    provider.close()
    gc.collect()
    assert all(reference() is None for _, reference in loader.references)
    assert provider._vae.weight.device.type == "cpu"


def test_baseline_read_failure_preserves_successful_zero_capture_and_counts():
    loader, specification = Loader(), config(1)
    provider = PrecompensationProvider(specification, loader=loader, public_reader=PublicReader(fail_at=1))
    events = list(provider.prepare_sources(specification["sources"]))
    assert [event["kind"] for event in events] == ["baseline", "failure"]
    assert events[0]["status"] == "FAILED"
    assert events[0]["image"].mode == "RGB"
    assert events[0]["before_state"]["latents"].shape == (1,3,2,2)
    assert provider.counts["final_generation_attempted"] == provider.counts["final_generation_returned"] == 1
    assert provider.counts["public_feature_reads_attempted"] == 1
    assert provider.counts["public_feature_reads_returned"] == 0
    assert provider.counts["autograd_calls_attempted"] == 0
    provider.close()


def test_design_post_read_failure_retains_generated_image_and_feature(monkeypatch):
    specification, loader = config(1), Loader()
    provider = PrecompensationProvider(specification, loader=loader)
    original = provider._check_reader
    calls = 0

    def check(feature):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("post-design check failed")
        return original(feature)

    monkeypatch.setattr(provider, "_check_reader", check)
    events = list(provider.prepare_sources(specification["sources"]))
    design = next(event for event in events if event["kind"] == "design")
    assert design["image"].mode == "RGB" and design["feature"].numel() == 12
    assert design["status"] == "OBSERVED"
    assert any(event["kind"] == "failure" and event["stage"] == "design_rgb_vjp" for event in events)
    failure = next(event for event in events if event["kind"] == "failure")
    assert failure["partial"]["proxy_image"].mode == "RGB"
    assert failure["partial"]["proxy_feature"].numel() == 12
    assert failure["partial"]["ordinary_image"].mode == "RGB"
    assert not any(event["kind"] == "linearization" for event in events)
    assert provider.counts["final_reconstruction_returned"] == 1
    assert provider.counts["reconstruction_returned"] == 2
    assert provider.counts["public_feature_reads_returned"] == 2
    provider.close()


def test_write_read_failure_retains_actual_write_and_matching_state():
    specification, loader, reader = config(1), Loader(), PublicReader(fail_at=3)
    provider = PrecompensationProvider(specification, loader=loader, public_reader=reader)
    preparation = list(provider.prepare_sources(specification["sources"]))
    events = list(provider.generate_writes(writes(preparation)[:1]))
    assert len(events) == 1 and events[0]["status"] == "FAILED"
    event = events[0]
    assert event["image"].mode == "RGB" and event["matched_state"] is True
    assert event["after_latents"].shape == event["actual_delta"].shape == (1,3,2,2)
    assert event["before_state"] and event["remaining_steps"] == (19,)
    assert provider.counts["final_generation_returned"] == 2
    provider.close()


def test_ordinary_and_gradient_reconstruction_mismatch_preserves_both_and_blocks_linearization():
    specification, loader = config(1), Loader(gradient_offset=2 / 255)
    provider = PrecompensationProvider(specification, loader=loader)
    events = list(provider.prepare_sources(specification["sources"]))
    assert [event["kind"] for event in events if event["kind"] != "rgb_gradient"] == ["baseline", "design", "failure"]
    failure = events[-1]
    assert "exactly reproduce ordinary design RGB" in failure["error"]
    partial = failure["partial"]
    assert not np.array_equal(np.asarray(partial["ordinary_image"]), np.asarray(partial["proxy_image"]))
    assert partial["ordinary_feature"].numel() == partial["proxy_feature"].numel() == 12
    gradient_events = [event for event in events if event["kind"] == "rgb_gradient"]
    assert [(event["phase"], event["reusable"]) for event in gradient_events] == [("clean", True), ("design", False)]
    assert torch.equal(partial["gradient_rgb"], gradient_events[-1]["result"]["gradient_rgb"])
    counts = provider.counts
    assert counts["final_reconstruction_returned"] == counts["auxiliary_reconstruction_forward_returned"] == 1
    assert counts["reconstruction_returned"] == 2 and counts["public_feature_reads_returned"] == 2
    assert counts["rgb_vjps_returned"] == 2 and counts["latent_vjps_attempted"] == 0
    assert loader.counts["generation"] == 1
    provider.close()


def test_evaluation_read_failure_keeps_ordinary_image_and_returned_forward():
    specification, loader, reader = config(1), Loader(), PublicReader(fail_at=3)
    provider = PrecompensationProvider(specification, loader=loader, public_reader=reader)
    preparation = list(provider.prepare_sources(specification["sources"]))
    assert list(provider.generate_writes([])) == []
    events = list(provider.reconstruct_outputs(evaluations(preparation, [], specification)))
    assert len(events) == 1 and events[0]["status"] == "FAILED"
    assert events[0]["image"].mode == "RGB"
    counts = provider.counts
    assert counts["final_reconstruction_attempted"] == counts["final_reconstruction_returned"] == 2
    assert counts["public_feature_reads_attempted"] == 3 and counts["public_feature_reads_returned"] == 2
    provider.close()


def test_explicit_close_releases_pipeline_when_preparation_generator_is_abandoned():
    specification, loader = config(1), Loader()
    provider = PrecompensationProvider(specification, loader=loader)
    generator = provider.prepare_sources(specification["sources"])
    assert next(generator)["kind"] == "baseline"
    assert provider._source is not None
    provider.close()
    generator.close()
    gc.collect()
    assert all(reference() is None for _, reference in loader.references)
    with pytest.raises(RuntimeError, match="completed single preparation"):
        list(provider.generate_writes([]))


def test_runtime_failed_capture_still_counts_a_returned_full_pipeline(monkeypatch):
    pipe = Source()
    original = Source.__call__.__wrapped__

    @torch.no_grad()
    def missing_callback(self, **kwargs):
        def identity(pipe, index, timestep, fields):
            return fields
        kwargs["callback_on_step_end"] = identity
        return original(self, **kwargs)

    monkeypatch.setattr(Source, "__call__", missing_callback)
    counts = {}
    settings = runtime.GenerationSettings("example", 23, 18, .01, 7, height=2, width=2, device="cpu")
    with pytest.raises(runtime.RuntimeStageError, match="omitted") as caught:
        runtime.capture_source(pipe, settings, counters=counts)
    assert counts["sd35_full_generation_calls"] == counts["sd35_full_generation_returned"] == 1
    assert caught.value.partial["image"].mode == "RGB"


def checkpoint(specification, *, with_gradients=False):
    """A completed zero-only CPU run, retaining the same real provider events."""
    provider = PrecompensationProvider(specification, loader=Loader())
    preparation = list(provider.prepare_sources(specification["sources"]))
    assert not any(event["kind"] == "failure" for event in preparation)
    assert list(provider.generate_writes([])) == []
    holdouts = list(provider.reconstruct_outputs(evaluations(preparation, [], specification)))
    baselines = {event["source_id"]: event for event in preparation if event["kind"] == "baseline"}
    reconstructions = {(event["source_id"], specification["reconstruction"]["design_seed"]): event
                       for event in preparation if event["kind"] == "design"}
    reconstructions.update({(event["source_id"], event["seed"]): event for event in holdouts})
    gradients = {}
    if with_gradients:
        for event in preparation:
            if event["kind"] == "rgb_gradient":
                gradients.setdefault(event["source_id"], {})[event["phase"]] = event
    vector, phase = make_reader(next(iter(baselines.values()))["feature"].numel(), ReaderSpec(**specification["reader"]))
    provider.close()
    return SimpleNamespace(baselines=baselines, zero_reconstructions=reconstructions,
                           reader_vector=vector, reader_phase=phase, rgb_gradients=gradients), preparation


class ReaderLoader:
    def __init__(self):
        self.calls = []
        self.vae = self.processor = None

    def __call__(self, specification):
        self.calls.append(specification["generation"])
        self.vae, self.processor = VAE(), Processor()
        return self.vae, self.processor


def test_resume_reuses_all_zero_assets_without_new_zero_calls_and_preserves_frozen_reader():
    specification = config()
    saved, original = checkpoint(specification)
    loader, reader_loader, reader = Loader(), ReaderLoader(), PublicReader()
    provider = PrecompensationProvider(specification, loader=loader, reader_loader=reader_loader,
                                      public_reader=reader, resume_checkpoint=saved)
    assert not loader.log and not reader_loader.calls
    preparation = list(provider.prepare_sources(specification["sources"]))
    reused = [event for event in preparation if event.get("reused")]
    assert len(reused) == 12
    assert {kind: sum(event["kind"] == kind for event in reused)
            for kind in ("baseline", "design", "holdout_zero")} == {"baseline": 4, "design": 4, "holdout_zero": 4}
    assert provider.counts["final_generation_attempted"] == provider.counts["final_reconstruction_attempted"] == 0
    assert provider.counts["public_feature_reads_attempted"] == 0
    assert len(reader_loader.calls) == 1 and loader.counts == {"generation": 1, "reconstruction": 1}
    assert provider._source.vae is reader_loader.vae
    original_gradients = {event["source_id"]: event["gradients"] for event in original if event["kind"] == "linearization"}
    for event in preparation:
        if event["kind"] == "linearization":
            torch.testing.assert_close(event["gradients"], original_gradients[event["source_id"]])
    marked = list(provider.generate_writes(writes(preparation)))
    # Even accidental repeated zero requests are served from saved observations.
    observed = list(provider.reconstruct_outputs(evaluations(preparation, marked, specification)))
    assert len(marked) == 16 and len(observed) == 36
    assert sum(event.get("reused", False) for event in observed) == 4
    counts = provider.counts
    for name, expected in (("final_generation", 16), ("final_reconstruction", 32), ("reconstruction", 36),
                           ("auxiliary_reconstruction_forward", 4), ("public_feature_reads", 48),
                           ("latent_vjps", 8), ("rgb_vjps", 8), ("autograd_calls", 16)):
        assert counts[name + "_attempted"] == counts[name + "_returned"] == expected
    assert all(call[:2] == (id(reader_loader.vae), id(reader_loader.processor)) for call in reader.calls)
    gradient_calls = [event for event in loader.log if event[0] == "reconstruction_call" and event[3]]
    assert len(gradient_calls) == 4
    assert {event[2] for event in gradient_calls} == {specification["reconstruction"]["design_seed"]}
    provider.close()


def test_resume_reuses_compatible_successful_gradients_but_never_holdout_gradients():
    specification = config(1)
    saved, _ = checkpoint(specification, with_gradients=True)
    loader = Loader()
    provider = PrecompensationProvider(specification, loader=loader, reader_loader=ReaderLoader(), resume_checkpoint=saved)
    events = list(provider.prepare_sources(specification["sources"]))
    assert len([event for event in events if event["kind"] == "linearization"]) == 1
    gradients = [event for event in events if event["kind"] == "rgb_gradient"]
    assert len(gradients) == 2 and all(event["reused"] for event in gradients)
    assert provider.counts["rgb_vjps_attempted"] == provider.counts["reconstruction_attempted"] == 0
    assert provider.counts["latent_vjps_returned"] == 2
    assert loader.counts == {"generation": 1, "reconstruction": 0}
    provider.close()


@pytest.mark.parametrize("invalid", ["layout", "failed_guard", "holdout_seed"])
def test_resume_recomputes_design_gradient_if_cache_is_incompatible_or_not_accepted(invalid):
    specification = config(1)
    saved, _ = checkpoint(specification, with_gradients=True)
    identifier = specification["sources"][0]["source_id"]
    cached = saved.rgb_gradients[identifier]["design"]
    if invalid == "layout":
        cached["compatibility_version"] = "old-layout"
    elif invalid == "failed_guard":
        cached["reusable"] = False
    else:
        cached["result"]["diagnostics"]["design_seed"] = specification["reconstruction"]["holdout_seed"]
    provider = PrecompensationProvider(specification, loader=Loader(), reader_loader=ReaderLoader(), resume_checkpoint=saved)
    events = list(provider.prepare_sources(specification["sources"]))
    assert not any(event["kind"] == "failure" for event in events)
    gradients = [event for event in events if event["kind"] == "rgb_gradient"]
    assert [(event["phase"], event["reused"]) for event in gradients] == [("clean", True), ("design", False)]
    assert provider.counts["rgb_vjps_returned"] == provider.counts["auxiliary_reconstruction_forward_returned"] == 1
    assert provider.counts["final_generation_attempted"] == provider.counts["final_reconstruction_attempted"] == 0
    provider.close()
