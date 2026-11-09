"""Nonlinear synthetic channel exercises all late arms and frozen holdout order."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from cegwm.channel_precompensation import ReaderSpec, make_reader
from cegwm.runtime.channel_precompensation_sd35 import SourceCapture
from cegwm.stage_protection_v2 import grouped_verdict
from experiments.run_stage_protection_v2 import execute, load_config

pytestmark = pytest.mark.unit


class SyntheticChannel:
    """Two scalars stand in for RGB; deliberately biased proxy, no real model."""
    def __init__(self, config, *, counters, directory, fail_holdout=None):
        self.config, self.directory = config, directory
        _, self.phase = make_reader(1, ReaderSpec(**config["reader"]))
        self.log = []
        self.fail_holdout = fail_holdout
        self.holdouts = 0

    def generate(self, source):
        self.log.append(("source", source["source_id"]))
        z = torch.full((1, 1, 1, 2), 100., dtype=torch.float64)
        return {step: SourceCapture(None, {"latents": z.clone(), "source": source["source_id"]}, {},
                                   tuple(range(step + 1, 20))) for step in (8, 18)}

    def render(self, capture, delta=None, *, capture_late=False):
        early = len(capture.remaining_steps) == 11
        self.log.append(("render_early" if early else "render_late",))
        z = capture.before_state["latents"]
        after = (z.float() + (torch.zeros_like(z) if delta is None else delta).float()).to(z.dtype)
        actual = after.double() - z.double()
        if early:
            change = float(after.flatten()[0] - 100)
            design = .5 + 40 * change
            clean = .5 if change == 0 else .6  # Long-path clean differs from late E.
            state = {"latents": torch.full_like(z, 100), "design": design}
        else:
            design = capture.before_state["design"]
            x, y = actual.flatten().tolist()
            direction = 1 if design < .5 else -1
            # A damages R; B narrowly fails; C's different y realizes protection.
            nonlinear = 0 if not bool(torch.count_nonzero(actual)) else direction * (.01 - 6 * max(0., abs(y) - .018))
            design += -10*x + 10*y + nonlinear
            clean = .5 + 10*x
        image = torch.tensor([self.phase + clean, self.phase + design], dtype=torch.float64).numpy()
        result = dict(image=image, actual_delta=actual, after=after)
        if capture_late:
            result["late"] = SourceCapture(image, state, {}, (19,))
        return result

    def read(self, image):
        return dict(score=float(image[0]), feature=torch.tensor(image))

    def reconstruct(self, image, seed):
        if seed == self.config["reconstruction"]["holdout_seed"]:
            assert (self.directory / "selection_frozen.pt").exists()
            self.holdouts += 1
            if self.holdouts == self.fail_holdout:
                raise RuntimeError("synthetic holdout failure")
        return image[1:].repeat(2)

    def rgb_gradient(self, image, ordinary_read, *, design_image=None, design_read=None):
        assert ordinary_read["score"] == image[0]
        self.log.append(("design_vjp" if design_image is not None else "clean_vjp", image.copy()))
        return dict(design=design_image is not None, image=image.copy())

    def latent_gradients(self, capture, image, sensitivities):
        assert all((s["image"] == image).all() for s in sensitivities)
        self.log.append(("latent_vjp", len(capture.remaining_steps), len(sensitivities)))
        if len(capture.remaining_steps) == 11:
            assert len(sensitivities) == 1
            gradients = torch.tensor([[100., 0.]])
        else:
            assert len(sensitivities) == 2
            assert image[0] == pytest.approx(self.phase + .5)  # E, never long-path .6.
            gradients = torch.tensor([[10., 0.], [-10., 10.]])
        return dict(gradients=gradients.reshape(-1, 1, 1, 1, 2))

    def close(self):
        pass


def run(tmp_path, **kwargs):
    config = load_config()
    provider = SyntheticChannel(config, counters={}, directory=tmp_path, **kwargs)
    result = execute(config, tmp_path, provider_factory=lambda *a, **k: provider)
    return result, provider


def test_full_runner_has_two_updates_all_three_arms_and_rebased_sensitivities(tmp_path):
    result, provider = run(tmp_path)
    assert result["engineering_complete"], result["trials"]
    assert len(result["trials"]) == 4
    for row in result["trials"]:
        assert len(row["early"]) == 2
        assert row["late_status"] == "repair_attempted"
        assert row["C_B_actual_different"]
        assert row["long_path_vs_E"]["mse"] > 0
        assert row["A_clean_repair_design_damage"]
        assert row["arms"]["A"]["selected"] == "E"
        assert row["arms"]["B"]["selected"] == "E"
        assert row["arms"]["C"]["selected"] == "proposal"
        assert row["arms"]["C"]["protection_realized"]
        assert row["C_success"] and not row["B_success"]
    assert len([x for x in provider.log if x[0] == "source"]) == 2
    assert len([x for x in provider.log if x[0] == "render_early"]) == 10  # two zero + eight updates
    assert len([x for x in provider.log if x[0] == "render_late"]) == 16  # four E + twelve arms
    gradients = [x for x in provider.log if x[0] == "latent_vjp"]
    assert len(gradients) == 12 and sum(x[1] for x in gradients) == 92
    assert sum(x[2] for x in gradients) == 16


def test_holdout_failure_does_not_skip_other_raw_proposals(tmp_path):
    result, provider = run(tmp_path, fail_holdout=2)
    assert not result["engineering_complete"]
    assert not result["verdict"]["operational_P2_signal"]
    # E holdout failure still leaves the three raw arms independently evaluated.
    assert all("holdout" in a for a in result["trials"][0]["arms"].values())
    assert result["trials"][0]["arms"]["A"]["raw_three_channel_success"] is False
    assert result["trials"][0]["arms"]["A"]["returned_three_channel_success"] is None
    assert len(result["trials"]) == 4


def test_unmarked_holdout_failure_overrides_trial_only_promotion(tmp_path):
    result, provider = run(tmp_path, fail_holdout=1)
    assert result["baseline_holdout_errors"]
    assert not result["engineering_complete"]
    assert len(result["trials"]) == 4
    assert all(row["evaluation_complete"] for row in result["trials"])
    trial_only = grouped_verdict(result["trials"], [s["source_id"] for s in load_config()["sources"]])
    assert trial_only["operational_P2_signal"]  # Isolate baseline failure from trial outcomes.
    verdict = result["verdict"]
    assert verdict["outcome"] == "ENGINEERING_INCOMPLETE"
    assert verdict["operational_P2_signal"] is False
    assert verdict["group_diagnostics_only"] is True
    assert verdict["trial_only_outcome"] == trial_only["outcome"]
    assert verdict["quality_tradeoff_review"] == "deferred_engineering_incomplete"
    for group, diagnostic in zip(verdict["groups"], trial_only["groups"]):
        assert group["operational_signal"] is False
        assert group["trial_only_operational_signal"] is diagnostic["operational_signal"]
        assert group["strict_improvement_bits"] == diagnostic["strict_improvement_bits"]
        assert group["complete"] == diagnostic["complete"]
    assert verdict["P3_authorized"] is False
    assert verdict["scientific_pass"] is verdict["innovation_pass"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["verdict"] == verdict
    assert provider.holdouts > 1  # Independent candidate evaluation still proceeds.
