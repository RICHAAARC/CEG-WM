"""CPU fake-provider checks for planning, fixed denominators, and saved failures."""

import json

import pytest
import torch
from PIL import Image

from experiments import run_channel_precompensation_v1 as runner
from cegwm.channel_precompensation import ReaderSpec, make_reader
from cegwm.runtime.channel_precompensation_resume import load_resume_checkpoint

pytestmark = pytest.mark.unit


def config():
    return {
        "generation": {"model_id": "fake-sd3", "revision": "fake", "device": "cpu", "dtype": "float32",
                       "height": 32, "width": 32, "num_inference_steps": 2,
                       "guidance_scale": 1.0, "injection_step": 0},
        "reconstruction": {"model_id": "fake-sdxl", "revision": "fake", "device": "cpu", "dtype": "float32",
                           "strength": 0.5, "num_inference_steps": 2, "guidance_scale": 1.0,
                           "prompt": "", "design_seed": 40, "holdout_seed": 41},
        "reader": {"key_seed": 30, "delta": 1.0, "inner_fraction": 0.25},
        "write": {"relative_l2_budget": 0.01}, "quality": {"minimum_psnr_db": None},
        "sources": [{"source_id": f"new_source_{index}", "prompt": f"new source {index}", "seed": index + 31}
                    for index in range(4)],
    }


class FakeProvider:
    def __init__(self, settings, *, preparation_failure=None, feature_failure=None,
                 infeasible_source=None, holdout_shift=0, omit_reconstruction=False,
                 collapsed=False, mismatched_state=False):
        self.settings = settings
        self.vector, self.phase = make_reader(8, ReaderSpec(**settings["reader"]))
        self.counts = {name: 0 for name in ("generation_attempted", "generation_returned",
            "reconstruction_attempted", "reconstruction_returned", "public_feature_reads_attempted",
            "final_reconstruction_attempted", "final_reconstruction_returned",
            "auxiliary_reconstruction_forward_attempted", "auxiliary_reconstruction_forward_returned",
            "public_feature_reads_returned", "latent_vjps_attempted", "latent_vjps_returned",
            "rgb_vjps_attempted", "rgb_vjps_returned", "auxiliary_forward_attempted", "auxiliary_forward_returned")}
        self.preparation_failure, self.feature_failure = preparation_failure, feature_failure
        self.infeasible_source, self.holdout_shift = infeasible_source, holdout_shift
        self.omit_reconstruction, self.collapsed = omit_reconstruction, collapsed
        self.mismatched_state = mismatched_state
        self.base = torch.full((1, 2, 2, 2), 100.0)
        self.gradients = torch.eye(8, dtype=torch.float64)[:2].reshape(2, *self.base.shape)
        self.design_scores, self.actual_deltas = {}, {}
        self.closed = False
        self.write_requests, self.reconstruction_requests = [], []

    def _feature(self, score, fail=False):
        self.counts["public_feature_reads_attempted"] += 1
        if fail:
            return None
        self.counts["public_feature_reads_returned"] += 1
        return self.vector * score

    def _generation(self):
        self.counts["generation_attempted"] += 1
        self.counts["generation_returned"] += 1

    def _reconstruction(self):
        self.counts["reconstruction_attempted"] += 1
        self.counts["reconstruction_returned"] += 1
        self.counts["final_reconstruction_attempted"] += 1
        self.counts["final_reconstruction_returned"] += 1

    def _capture(self, source):
        steps = self.settings["generation"]["num_inference_steps"]
        injection = self.settings["generation"]["injection_step"]
        timesteps = torch.arange(steps, 0, -1, dtype=torch.float32)
        sigmas = torch.linspace(1, 0, steps + 1)
        state = {"latents": self.base.clone(), "prompt_embeds": torch.ones(1, 2, 3),
            "pooled_prompt_embeds": torch.ones(1, 4), "timesteps": timesteps, "sigmas": sigmas,
            "generator_state": torch.Generator().manual_seed(source["seed"]).get_state(),
            "next_step_index": injection + 1, "guidance_scale": 1.0, "do_classifier_free_guidance": False,
            "latent_device": "cpu", "timestep": timesteps[injection], "interval_start_sigma": sigmas[injection],
            "latent_sigma": sigmas[injection + 1], "next_sigma": sigmas[injection + 2], "begin_index": None,
            "scheduler_class": "fake.Flow", "scheduler_config": {}, "clip_skip": None, "joint_attention_kwargs": None}
        scheduler = {"timesteps": timesteps, "sigmas": sigmas, "_step_index": injection + 1,
                     "_begin_index": None, "num_inference_steps": steps, "_internal_dict": {}}
        return state, scheduler, tuple(range(injection + 1, steps))

    def prepare_sources(self, sources):
        for source in sources:
            identifier = source["source_id"]
            self._generation()
            state, scheduler, remaining = self._capture(source)
            yield {"kind": "baseline", "source_id": identifier, "status": "OBSERVED",
                   "image": Image.new("RGB", (32, 32), (80, 90, 100)),
                   "feature": self._feature(self.phase + 0.1), "before_state": state,
                   "scheduler_state": scheduler, "remaining_steps": remaining}
            if identifier == self.preparation_failure:
                yield {"kind": "failure", "source_id": identifier, "stage": "design", "error": "fake design failure",
                       "partial": {"image": Image.new("RGB", (32, 32), (82, 90, 100)),
                                   "before_state": {"latents": self.base.clone()},
                                   "completed_gradients": (self.gradients[0].clone(),)}}
                continue
            score = self.phase + (1.1 if identifier == self.infeasible_source else 0.2)
            self.design_scores[identifier] = score
            self._reconstruction()
            yield {"kind": "design", "source_id": identifier, "status": "OBSERVED",
                   "image": Image.new("RGB", (32, 32), (82, 90, 100)),
                   "feature": self._feature(score)}
            self.counts["auxiliary_reconstruction_forward_attempted"] += 1
            self.counts["auxiliary_reconstruction_forward_returned"] += 1
            self.counts["reconstruction_attempted"] += 1
            self.counts["reconstruction_returned"] += 1
            for prefix in ("latent_vjps", "rgb_vjps", "auxiliary_forward"):
                self.counts[f"{prefix}_attempted"] += 2
                self.counts[f"{prefix}_returned"] += 2
            gradients = self.gradients.clone()
            if identifier == self.infeasible_source:
                gradients[1] = gradients[0]
            yield {"kind": "linearization", "source_id": identifier, "gradients": gradients,
                   "diagnostics": {"surrogate": "CPU fake"}}

    def generate_writes(self, requests):
        self.write_requests = list(requests)
        for request in requests:
            identifier, method, bit = (request[name] for name in ("source_id", "method", "bit"))
            self._generation()
            requested = request["delta"]
            after = self.base if self.collapsed else (self.base.double() + requested).float()
            delta = after.double() - self.base.double()
            self.actual_deltas[(identifier, method, bit)] = delta
            score = self.phase + 0.1 + float(torch.sum(self.gradients[0] * delta))
            fail = (identifier, method, bit) == self.feature_failure
            yield {"source_id": identifier, "method": method, "bit": bit,
                   "image": Image.new("RGB", (32, 32), (80 + 40 * bit, 90, 100)),
                   "feature": self._feature(score, fail), "actual_delta": delta, "after_latents": after,
                   "matched_state": not self.mismatched_state, "status": "FAILED" if fail else "OBSERVED",
                   "error": "fake feature read failure" if fail else None}

    def reconstruct_outputs(self, requests):
        self.reconstruction_requests = list(requests)
        for index, request in enumerate(requests):
            if self.omit_reconstruction and index == len(requests) - 1:
                continue
            identifier, method, bit, seed = (request[name] for name in ("source_id", "method", "bit", "seed"))
            self._reconstruction()
            delta = torch.zeros_like(self.base) if method == "zero" else self.actual_deltas[(identifier, method, bit)]
            score = self.design_scores.get(identifier, self.phase + 0.2) + float(torch.sum(self.gradients[1] * delta))
            if seed == self.settings["reconstruction"]["holdout_seed"]:
                score += self.holdout_shift
            yield {**{key: request[key] for key in ("source_id", "method", "bit", "seed")},
                   "image": request["image"].copy(), "feature": self._feature(score), "status": "OBSERVED"}

    def close(self):
        self.closed = True


def test_default_is_plan_only_and_counts_marked_zero_and_aux_separately(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    output = tmp_path / "output"
    monkeypatch.setattr(runner, "load_provider", lambda _: pytest.fail("plan initialized a model provider"))
    assert runner.main(["--config", str(path), "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_budget"]["marked_final_images"] == 16
    assert plan["planned_budget"]["shared_zero_images"] == 4
    assert plan["planned_budget"]["final_generation_calls"] == 20
    assert plan["planned_budget"]["final_reconstruction_calls"] == 40
    assert plan["planned_budget"]["auxiliary_reconstruction_forward_calls"] == 4
    assert plan["planned_budget"]["total_reconstruction_forward_calls"] == 44
    assert plan["planned_budget"]["public_feature_reads"] == 60
    assert plan["planned_budget"]["latent_vjps"] == 8
    assert plan["fixed_denominators"] == {"bit_trials_per_method": 8, "both_bits_per_source_per_method": 4}
    assert not output.exists()


@pytest.mark.parametrize("section,field", [("reader", "delta"), ("reader", "key_seed"),
    ("reader", "inner_fraction"), ("write", "relative_l2_budget"), ("quality", "minimum_psnr_db"),
    ("reconstruction", "design_seed"), ("reconstruction", "holdout_seed")])
def test_required_choices_cannot_be_inherited(section, field):
    settings = config()
    del settings[section][field]
    with pytest.raises(ValueError):
        runner.validate_config(settings)


def test_design_holdout_must_differ_and_roster_stays_four():
    settings = config()
    settings["reconstruction"]["holdout_seed"] = settings["reconstruction"]["design_seed"]
    with pytest.raises(ValueError, match="must differ"):
        runner.validate_config(settings)
    settings = config()
    settings["sources"].pop()
    with pytest.raises(ValueError, match="exactly four"):
        runner.validate_config(settings)


def test_complete_collection_saved_with_fixed_public_reader_and_no_quality_gate(tmp_path):
    settings = config()
    provider = FakeProvider(settings)
    output = tmp_path / "output"
    result = runner.execute_measurement(settings, output, provider=provider)
    assert provider.closed
    assert all(row["collection_complete"] for row in result["sources"])
    assert result["actual_calls"]["generation_attempted"] == 20
    assert result["actual_calls"]["reconstruction_attempted"] == 44
    assert result["actual_calls"]["final_reconstruction_attempted"] == 40
    assert result["actual_calls"]["auxiliary_reconstruction_forward_attempted"] == 4
    assert result["actual_calls"]["public_feature_reads_attempted"] == 60
    assert result["actual_calls"]["latent_vjps_attempted"] == 8
    assert result["actual_calls"]["rgb_vjps_attempted"] == 8
    assert result["actual_calls"]["auxiliary_forward_attempted"] == 8
    assert len(list(output.rglob("*.png"))) == 60
    assert (output / "reader.pt").exists()
    for row in result["sources"]:
        assert (output / row["source_id"] / "writer/linearization.pt").exists()
        for method in runner.METHODS:
            for bit in runner.BITS:
                trial = row["trials"][method][str(bit)]
                assert trial["clean"]["status"] == "OBSERVED"
                assert trial["quality"]["gate_enabled"] is False
                assert trial["quality"]["eligible"] is True
                assert "actual_linear_prediction" in trial["write"]
    for method in runner.METHODS:
        assert result["method_summaries"][method]["expected_bit_trials"] == 8
        assert result["method_summaries"][method]["collected_bit_trials"] == 8
    saved = (output / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        runner.execute_measurement(settings, output, provider=provider)
    assert (output / "summary.json").read_text() == saved


def test_infeasible_source_stays_in_denominator_and_is_not_retried(tmp_path):
    provider = FakeProvider(config(), infeasible_source="new_source_0")
    result = runner.execute_measurement(config(), tmp_path / "output", provider=provider)
    assert not any(request["source_id"] == "new_source_0" for request in provider.write_requests)
    failed = result["sources"][0]
    for method in runner.METHODS:
        assert result["method_summaries"][method]["expected_bit_trials"] == 8
        assert result["method_summaries"][method]["collected_bit_trials"] == 6
        for bit in runner.BITS:
            assert failed["trials"][method][str(bit)]["solve"]["status"] != "SOLVED"
            assert failed["trials"][method][str(bit)]["clean"] is None
    assert result["actual_calls"]["generation_attempted"] == 16


def test_prepare_failure_keeps_zero_and_all_four_trial_failures(tmp_path):
    provider = FakeProvider(config(), preparation_failure="new_source_0")
    output = tmp_path / "output"
    result = runner.execute_measurement(config(), output, provider=provider)
    failed = result["sources"][0]
    assert (output / "new_source_0/zero/clean.png").exists()
    assert "fake design failure" in failed["errors"]["design"]
    evidence = failed["writer"]["failure_evidence"][0]
    writer_directory = output / "new_source_0/writer"
    assert (writer_directory / evidence["saved_images"]["image"]).exists()
    partial = torch.load(writer_directory / evidence["partial_file"], weights_only=True)
    assert torch.equal(partial["partial"]["before_state"]["latents"], provider.base)
    assert torch.equal(partial["partial"]["completed_gradients"][0], provider.gradients[0])
    assert all(failed["trials"][method][str(bit)]["errors"] for method in runner.METHODS for bit in runner.BITS)
    assert all(result["method_summaries"][method]["expected_bit_trials"] == 8 for method in runner.METHODS)


def test_failed_public_read_keeps_image_and_does_not_hide_other_outcomes(tmp_path):
    provider = FakeProvider(config(), feature_failure=("new_source_0", "joint", 1))
    output = tmp_path / "output"
    result = runner.execute_measurement(config(), output, provider=provider)
    trial = result["sources"][0]["trials"]["joint"]["1"]
    assert (output / "new_source_0/joint_bit1/clean.png").exists()
    assert trial["clean"]["status"] == "FAILED"
    assert trial["collection_complete"] is False
    assert trial["reconstruction"]["41"]["status"] == "OBSERVED"
    assert result["method_summaries"]["joint"]["collected_bit_trials"] == 7
    assert result["method_summaries"]["joint"]["expected_bit_trials"] == 8
    assert result["actual_calls"]["public_feature_reads_returned"] == 59


def test_wrong_holdout_bits_are_collected_negative_results(tmp_path):
    result = runner.execute_measurement(config(), tmp_path / "output", provider=FakeProvider(config(), holdout_shift=1.0))
    assert all(row["collection_complete"] for row in result["sources"])
    for method in runner.METHODS:
        assert result["method_summaries"][method]["collected_bit_trials"] == 8
        assert result["method_summaries"][method]["blind_all_conditions_correct"] == 0
        assert result["method_summaries"][method]["eligible_joint_successes"] == 0


def test_missing_reconstruction_is_retained_and_not_successful(tmp_path):
    result = runner.execute_measurement(config(), tmp_path / "output", provider=FakeProvider(config(), omit_reconstruction=True))
    source = result["sources"][-1]
    assert source["status"] == "incomplete"
    trial = source["trials"]["gradient"]["1"]
    assert "reconstruction/41" in trial["errors"]
    assert trial["collection_complete"] is False
    assert result["method_summaries"]["gradient"]["expected_bit_trials"] == 8


def test_quantization_collapse_and_state_mismatch_cannot_be_success(tmp_path):
    collapsed = runner.execute_measurement(config(), tmp_path / "collapsed", provider=FakeProvider(config(), collapsed=True))
    assert all(row["collection_complete"] for row in collapsed["sources"])
    trial = collapsed["sources"][0]["trials"]["joint"]["1"]
    assert trial["write"]["quantization_collapsed"]
    assert not trial["eligible_joint_success"]
    mismatch = runner.execute_measurement(config(), tmp_path / "mismatch", provider=FakeProvider(config(), mismatched_state=True))
    assert not any(row["collection_complete"] for row in mismatch["sources"])


def test_setup_failure_keeps_roster_and_fixed_denominators(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("fake loader failed")
    monkeypatch.setattr(runner, "load_provider", fail)
    result = runner.execute_measurement(config(), tmp_path / "output")
    assert "fake loader failed" in result["execution_error"]
    assert len(result["sources"]) == 4
    assert all(result["method_summaries"][method]["expected_bit_trials"] == 8 for method in runner.METHODS)
    assert result["stages"]["setup"] == "failed"


class PreWriteProvider(FakeProvider):
    def generate_writes(self, requests):
        # Completed zero observations with no physically attempted marked writes.
        self.write_requests = list(requests)
        yield from ()


class ResumeProvider(FakeProvider):
    def __init__(self, settings, checkpoint, *, prewrite_only=False, reusable_gradients=True):
        super().__init__(settings)
        self.checkpoint = checkpoint
        self.prewrite_only = prewrite_only
        self.reusable_gradients = reusable_gradients

    def prepare_sources(self, sources):
        for source in sources:
            identifier = source["source_id"]
            yield {**self.checkpoint.baselines[identifier], "kind": "baseline", "source_id": identifier, "reused": True}
            for name, kind in (("design_seed", "design"), ("holdout_seed", "holdout_zero")):
                seed = self.settings["reconstruction"][name]
                observed = self.checkpoint.zero_reconstructions[(identifier, seed)]
                yield {**observed, "kind": kind, "source_id": identifier, "seed": seed, "reused": True}
            self.design_scores[identifier] = float(torch.dot(
                self.checkpoint.zero_reconstructions[(identifier, self.settings["reconstruction"]["design_seed"])]["feature"], self.vector))
            for phase in ("clean", "design"):
                observed = (self.checkpoint.baselines[identifier] if phase == "clean" else
                            self.checkpoint.zero_reconstructions[(identifier, self.settings["reconstruction"]["design_seed"])])
                self.counts["rgb_vjps_attempted"] += 1
                self.counts["rgb_vjps_returned"] += 1
                yield {"kind": "rgb_gradient", "source_id": identifier, "phase": phase,
                    "compatibility_version": "rgb-pil-layout-v2", "reusable": self.reusable_gradients, "reused": False,
                    "result": {"image": observed["image"], "feature": observed["feature"],
                        "score": float(torch.dot(observed["feature"], self.vector)),
                        "gradient_rgb": torch.zeros(1, 3, 32, 32),
                        "diagnostics": {"design_seed": self.settings["reconstruction"]["design_seed"]}}}
            self.counts["auxiliary_reconstruction_forward_attempted"] += 1
            self.counts["auxiliary_reconstruction_forward_returned"] += 1
            self.counts["reconstruction_attempted"] += 1
            self.counts["reconstruction_returned"] += 1
            self.counts["latent_vjps_attempted"] += 2
            self.counts["latent_vjps_returned"] += 2
            yield {"kind": "linearization", "source_id": identifier, "gradients": self.gradients.clone()}

    def generate_writes(self, requests):
        if self.prewrite_only:
            self.write_requests = list(requests)
            yield from ()
        else:
            yield from super().generate_writes(requests)


def make_prewrite_checkpoint(path):
    result = runner.execute_measurement(config(), path, provider=PreWriteProvider(config()))
    assert result["actual_calls"]["generation_attempted"] == 4
    assert result["actual_calls"]["final_reconstruction_attempted"] == 8
    return load_resume_checkpoint(path)


def test_resume_plan_reads_saved_config_without_loading_models(tmp_path, monkeypatch, capsys):
    checkpoint = make_prewrite_checkpoint(tmp_path / "original")
    monkeypatch.setattr(runner, "load_provider", lambda *args, **kwargs: pytest.fail("resume plan loaded models"))
    assert runner.main(["--resume-from", str(checkpoint.batch_directory)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_budget"]["final_generation_calls"] == 16
    assert plan["planned_budget"]["final_reconstruction_calls"] == 32
    assert plan["planned_budget"]["public_feature_reads"] == 48
    assert plan["planned_budget"]["total_reconstruction_forward_calls"] == 36
    assert plan["imported_counts"]["public_zero_features"] == 12


def test_resume_reuses_zero_observations_and_preserves_prior_failures_and_counts(tmp_path):
    original = tmp_path / "original"
    checkpoint = make_prewrite_checkpoint(original)
    original_bytes = {path.relative_to(original): path.read_bytes() for path in original.rglob("*") if path.is_file()}
    provider = ResumeProvider(config(), checkpoint)
    output = tmp_path / "continuation"
    result = runner.execute_measurement(config(), output, provider=provider, resume_from=original)
    assert all(source["collection_complete"] for source in result["sources"])
    assert result["actual_calls"]["generation_attempted"] == 16
    assert result["actual_calls"]["final_reconstruction_attempted"] == 32
    assert result["actual_calls"]["reconstruction_attempted"] == 36
    assert result["actual_calls"]["public_feature_reads_attempted"] == 48
    assert result["cumulative_actual_calls"]["generation_attempted"] == 20
    assert result["cumulative_actual_calls"]["final_reconstruction_attempted"] == 40
    assert result["cumulative_actual_calls"]["reconstruction_attempted"] == 48
    assert result["cumulative_actual_calls"]["public_feature_reads_attempted"] == 60
    assert result["reused_counts"] == {"source_captures": 4, "zero_generation_images": 4,
        "zero_reconstruction_images": 8, "public_zero_features": 12, "reader_assets": 1, "rgb_sensitivities": 0}
    assert all(request["method"] != "zero" for request in provider.reconstruction_requests)
    assert json.loads((output / "prior_summary.json").read_text()) == checkpoint.prior_summary
    assert checkpoint.prior_summary["method_summaries"]["joint"]["collected_bit_trials"] == 0
    assert len(list(output.rglob("rgb_gradient_*.pt"))) == 8
    assert all((original / relative).read_bytes() == value for relative, value in original_bytes.items())
    assert result["method_summaries"]["joint"]["expected_bit_trials"] == 8


def test_resume_loader_rejects_semantic_changes_but_accepts_explanatory_metadata(tmp_path):
    checkpoint = make_prewrite_checkpoint(tmp_path / "original")
    settings = config()
    settings["purpose"] = "new explanation only"
    assert load_resume_checkpoint(checkpoint.batch_directory, config=settings).config["purpose"] == settings["purpose"]
    for section, field, changed in (("reader", "delta", 2), ("reconstruction", "holdout_seed", 42),
                                     ("write", "relative_l2_budget", 0.02)):
        settings = config()
        settings[section][field] = changed
        with pytest.raises(ValueError, match="changes fixed method fields"):
            load_resume_checkpoint(checkpoint.batch_directory, config=settings)


def test_resume_never_overwrites_original_or_restarts_already_attempted_marked_writes(tmp_path):
    checkpoint = make_prewrite_checkpoint(tmp_path / "original")
    with pytest.raises(ValueError, match="original batch"):
        runner.execute_measurement(config(), checkpoint.batch_directory / "new", resume_from=checkpoint.batch_directory)
    assert not (checkpoint.batch_directory / "new").exists()
    complete = tmp_path / "completed"
    runner.execute_measurement(config(), complete, provider=FakeProvider(config()))
    with pytest.raises(ValueError, match="pre-write checkpoints only"):
        load_resume_checkpoint(complete)


def test_resume_loader_rejects_missing_or_nonfinite_saved_evidence(tmp_path):
    checkpoint = make_prewrite_checkpoint(tmp_path / "original")
    path = checkpoint.batch_directory / "new_source_0/zero/clean.pt"
    record = torch.load(path, weights_only=True)
    record["feature"][0] = float("nan")
    torch.save(record, path)
    with pytest.raises(ValueError, match="finite"):
        load_resume_checkpoint(checkpoint.batch_directory)


def test_second_prewrite_continuation_carries_cumulative_counts_and_safe_gradient_cache(tmp_path):
    original = tmp_path / "original"
    checkpoint = make_prewrite_checkpoint(original)
    first = tmp_path / "first"
    result = runner.execute_measurement(config(), first,
        provider=ResumeProvider(config(), checkpoint, prewrite_only=True), resume_from=original)
    restored = load_resume_checkpoint(first)
    assert restored.prior_actual_calls == result["cumulative_actual_calls"]
    assert restored.reused_counts["rgb_sensitivities"] == 8
    second = runner.execute_measurement(config(), tmp_path / "second",
        provider=ResumeProvider(config(), restored, prewrite_only=True, reusable_gradients=False), resume_from=first)
    assert second["cumulative_actual_calls"]["generation_attempted"] == 4
    assert second["cumulative_actual_calls"]["final_reconstruction_attempted"] == 8
    assert second["cumulative_actual_calls"]["reconstruction_attempted"] == 20
    assert second["cumulative_actual_calls"]["public_feature_reads_attempted"] == 12
