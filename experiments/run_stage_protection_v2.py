"""P1/P2 fixed canary. Default invocation prints the plan without loading models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cegwm.channel_precompensation import ReaderSpec, _norm
from cegwm.stage_protection_v2 import (
    LIMITS, cast_action, common_accept, early_update, fixed_cells, grouped_verdict,
    protection_radius, read_score, solve_intervals,
)
from cegwm.runtime.stage_protection_store import OperationStore, atomic_json, atomic_torch
from cegwm.runtime.stage_protection_v2 import StageProtectionProvider, at_latent, physical_counts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_protection_v2_colab.json"


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8-sig"))


def describe_plan(config):
    return dict(config=config, phase="P1_ready_for_review_P2_user_selected_only",
        content_denominator=2, bit_trial_denominator=4, maxima=LIMITS,
        stage_relative_budget=0.01 / math.sqrt(2), joint_squared_budget=0.01 ** 2,
        holdout_policy="After every proposal and selection is frozen, including unmarked holdout.",
        resume_policy="Reuse RETURNED; STARTED/FAILED is retained, never automatically retried.",
        resource_status="Eleven-step backward GPU/host RAM/time UNKNOWN; CPU checks are not scientific evidence.",
        scientific_pass=None, innovation_pass=None, automatic_P3_or_P4=False)


def quality(image, baseline):
    a, b = np.asarray(image).astype(float) / 255, np.asarray(baseline).astype(float) / 255
    mse = float(np.mean((a - b) ** 2))
    return dict(mse=mse, psnr_db=None if mse == 0 else -10 * math.log10(mse),
                identical=mse == 0, max_abs_rgb=float(np.max(np.abs(a - b))),
                eligibility_gate=False)


class Canary:
    def __init__(self, config, store, provider):
        self.config, self.store, self.provider = config, store, provider
        self.spec = ReaderSpec(**config["reader"])
        self.rho = config["write"]["relative_l2_budget"]

    def op(self, key, function):
        return self.store.call(key, function)

    def observe(self, key, image):
        clean = self.op(key + "/clean", lambda: self.provider.read(image))
        design_image = self.op(key + "/design_image", lambda: self.provider.reconstruct(
            image, self.config["reconstruction"]["design_seed"]))
        design = self.op(key + "/design_read", lambda: self.provider.read(design_image))
        return dict(image=image, clean=clean, design=design, design_image=design_image)

    def reads(self, observation, centers, bit):
        return {name: read_score(observation[name]["score"], centers[i], bit, self.spec)
                for i, name in enumerate(("clean", "design"))}

    def design_gradient(self, key, observation):
        return self.op(key, lambda: self.provider.rgb_gradient(observation["image"], observation["clean"],
            design_image=observation["design_image"], design_read=observation["design"]))

    def trial(self, source, baseline, bit):
        sid = source["source_id"]
        key = f"{sid}/bit{bit}"
        centers = fixed_cells([baseline["obs"][n]["score"] for n in ("clean", "design")], bit, self.spec)
        early_capture = baseline["captures"][8]
        early_base = early_capture.before_state["latents"]
        late_reference = baseline["captures"][18].before_state["latents"]
        current = baseline["zero"]
        observation = baseline["obs"]
        cumulative = torch.zeros_like(early_base, dtype=torch.float64)
        early_relative = 0.0
        row = dict(source_id=sid, bit=bit, centers=centers, early=[], arms={}, errors=[],
                   baseline_reads=self.reads(observation, centers, bit), evaluation_complete=False)
        try:
            for update in range(2):
                if abs(observation["design"]["score"] - centers[1]) <= 0.25:
                    row["early_stop"] = "already_design_inner" if update == 0 else "design_inner_reached"
                    break
                prefix = key + f"/early{update + 1}"
                design = self.design_gradient(prefix + "/rgb_vjp", observation)
                capture = at_latent(early_capture, current["after"], observation["image"])
                gradients = self.op(prefix + "/latent_vjp", lambda: self.provider.latent_gradients(
                    capture, observation["image"], [design]))
                requested = early_update(cumulative, gradients["gradients"][0], observation["design"]["score"],
                                         centers[1], self.rho / math.sqrt(2) * _norm(early_base.double()))
                action = self.op(prefix + "/action", lambda: cast_action(early_base, requested, early_base, self.rho))
                record = {"action": action, "requested": requested, "basepoint_score": observation["design"]["score"]}
                row["early"].append(record)
                if not action["within_budget"]:
                    raise ValueError("early actual cast exceeds fixed stage budget; no strength repair")
                if torch.equal(action["after"], current["after"]):
                    record["status"] = "noop_cast"
                    break
                replay = self.op(prefix + "/render", lambda: self.provider.render(early_capture, requested, capture_late=True))
                if not torch.equal(replay["actual_delta"], action["actual"]):
                    raise ValueError("early execution differs from preflight actual cast")
                current, cumulative, early_relative = replay, action["actual"], action["relative"]
                observation = self.observe(prefix, replay["image"])
                record.update(status="OBSERVED", reads=self.reads(observation, centers, bit))
            row.update(early_relative=early_relative, early_nonzero=bool(torch.count_nonzero(cumulative)))
            # Always rebase through the same one-step engine used by all arms.
            E_render = self.op(key + "/E/render", lambda: self.provider.render(current["late"]))
            row["E_render"] = E_render
            E = self.observe(key + "/E", E_render["image"])
            E_reads = self.reads(E, centers, bit)
            row.update(E=E, E_reads=E_reads,
                E_quality=quality(E["image"], baseline["obs"]["image"]),
                long_path_vs_E=quality(E["image"], current["image"]),
                active_early_flip=(row["early_nonzero"] and
                    row["baseline_reads"]["design"]["decoded_bit"] != bit and E_reads["design"]["bit_correct"]))
            if not E_reads["design"]["fixed_inner"]:
                row["late_status"] = "early_design_gate_failed"
            elif E_reads["clean"]["fixed_inner"]:
                row["late_status"] = "no_repair_needed"
            else:
                row["late_status"] = "repair_attempted"
                rP = protection_radius(E["design"]["score"], centers[1])
                row["rP"] = rP
                clean = self.op(key + "/late/clean_rgb_vjp", lambda: self.provider.rgb_gradient(E["image"], E["clean"]))
                design = self.design_gradient(key + "/late/design_rgb_vjp", E)
                gradients = self.op(key + "/late/latent_vjp", lambda: self.provider.latent_gradients(
                    current["late"], E["image"], [clean, design]))["gradients"]
                scores = [E[n]["score"] for n in ("clean", "design")]
                radius = self.rho / math.sqrt(2) * _norm(late_reference.double())
                for arm in ("A", "B", "C"):
                    count = 1 if arm == "A" else 2
                    widths = [0.25] if count == 1 else [0.25, rP if arm == "C" else 0.25]
                    solved = solve_intervals(gradients[:count], scores[:count], centers[:count], widths, radius)
                    proposal = dict(solve=solved, status=solved["status"], selected="E", accepted=False,
                                    protection_realized=None, late_relative=None)
                    row["arms"][arm] = proposal
                    prefix = key + "/late/" + arm
                    self.op(prefix + "/solve", lambda: solved)
                    if solved["delta"] is None:
                        continue
                    try:
                        action = self.op(prefix + "/action", lambda: cast_action(
                            current["late"].before_state["latents"], solved["delta"], late_reference, self.rho))
                        proposal.update(action=action, late_relative=action["relative"])
                        if not action["within_budget"] or early_relative ** 2 + action["relative"] ** 2 > self.rho ** 2:
                            proposal["status"] = "infeasible_actual_budget"
                            continue
                        if action["nonzero"]:
                            rendered = self.op(prefix + "/render", lambda: self.provider.render(current["late"], solved["delta"]))
                            if not torch.equal(rendered["actual_delta"], action["actual"]):
                                raise ValueError("late execution differs from preflight actual cast")
                            proposal["render"] = rendered
                            observed = self.observe(prefix, rendered["image"])
                        else:
                            observed = E
                        reads = self.reads(observed, centers, bit)
                        accept = common_accept(reads, early_relative, action["relative"], self.rho)
                        proposal.update(observation=observed, reads=reads, accepted=accept,
                            status="OBSERVED" if action["nonzero"] else "noop_cast",
                            selected="proposal" if accept else "E", quality=quality(observed["image"], baseline["obs"]["image"]),
                            protection_realized=abs(observed["design"]["score"] - centers[1]) <= rP,
                            design_margin_change=reads["design"]["fixed_inner_margin"] - E_reads["design"]["fixed_inner_margin"],
                            design_left_inner=not reads["design"]["fixed_inner"],
                            design_bit_flip=reads["design"]["decoded_bit"] != E_reads["design"]["decoded_bit"])
                    except Exception as error:
                        proposal.update(status="ENGINEERING_FAILED", error=str(error))
                        row["errors"].append(str(error))
                B, C = row["arms"]["B"], row["arms"]["C"]
                b, c = B["solve"]["requested"], C["solve"]["requested"]
                row["C_B_requested_different"] = None if b is None or c is None else not torch.equal(b, c)
                row["C_B_actual_different"] = (not torch.equal(B["action"]["actual"], C["action"]["actual"])) if "action" in B and "action" in C else None
                row["protection_constraint_active"] = (abs(B["solve"]["predicted"][1] - centers[1]) > rP) if "predicted" in B["solve"] else None
            for arm in ("A", "B", "C"):
                row["arms"].setdefault(arm, dict(status=row["late_status"], selected="E", accepted=False))
        except Exception as error:
            row["errors"].append(f"{type(error).__name__}: {error}")
            row.setdefault("late_status", "ENGINEERING_INCOMPLETE")
            for arm in ("A", "B", "C"):
                row["arms"].setdefault(arm, dict(status="ENGINEERING_INCOMPLETE", selected="E", accepted=False))
        return row

    def freeze(self):
        destination = self.store.directory / "selection_frozen.pt"
        if destination.exists():
            return torch.load(destination, map_location="cpu", weights_only=False)
        baselines, trials = {}, []
        for source in self.config["sources"]:
            sid = source["source_id"]
            try:
                captures = self.op(sid + "/source", lambda: self.provider.generate(source))
                zero = self.op(sid + "/canonical_early_zero", lambda: self.provider.render(captures[8], capture_late=True))
                observed = self.observe(sid + "/unmarked", zero["image"])
                baseline = dict(captures=captures, zero=zero, obs=observed,
                                historical_vs_canonical=quality(zero["image"], captures[8].image))
                baselines[sid] = baseline
                for bit in (0, 1):
                    trials.append(self.trial(source, baseline, bit))
            except Exception as error:
                trials.extend(dict(source_id=sid, bit=bit, arms={}, errors=[str(error)],
                                   evaluation_complete=False) for bit in (0, 1))
        frozen = dict(baselines=baselines, trials=trials)
        atomic_torch(destination, frozen)
        atomic_json(self.store.directory / "selection_frozen.json", dict(
            status="FROZEN_BEFORE_HOLDOUT", trial_count=len(trials), content_count=2,
            selections=[dict(source_id=r["source_id"], bit=r["bit"],
                arms={a: p["selected"] for a, p in r["arms"].items()}) for r in trials]))
        return frozen

    def evaluate(self, frozen):
        # All decisions are already saved. Equality reuse is recorded explicitly.
        cache = []
        aliases = []

        def holdout(key, image):
            for previous_key, previous_image, result in cache:
                if np.array_equal(np.asarray(previous_image), np.asarray(image)):
                    aliases.append(dict(key=key, reused_from=previous_key))
                    if isinstance(result, Exception):
                        raise RuntimeError(f"same RGB holdout already failed at {previous_key}; no retry") from result
                    return result
            try:
                reconstructed = self.op(key + "/image", lambda: self.provider.reconstruct(
                    image, self.config["reconstruction"]["holdout_seed"]))
                read = self.op(key + "/read", lambda: self.provider.read(reconstructed))
                result = dict(image=reconstructed, **read)
                cache.append((key, image, result))
                return result
            except Exception as error:
                cache.append((key, image, error))
                raise

        baseline_errors = []
        for sid, baseline in frozen["baselines"].items():
            try:
                baseline["holdout"] = holdout(sid + "/unmarked_holdout", baseline["obs"]["image"])
            except Exception as error:
                baseline_errors.append(str(error))
        for row in frozen["trials"]:
            if "E_render" not in row:
                continue
            key = f"{row['source_id']}/bit{row['bit']}"
            center, bit = row["centers"][1], row["bit"]
            try:
                E_holdout = holdout(key + "/E/holdout", row["E_render"]["image"])
                row["E_holdout"] = read_score(E_holdout["score"], center, bit, self.spec)
            except Exception as error:
                row["errors"].append(f"E holdout: {error}")
            for arm, proposal in row["arms"].items():
                proposal["returned_three_channel_success"] = None
                proposal["raw_three_channel_success"] = None
                try:
                    if "observation" in proposal:
                        raw = holdout(key + f"/late/{arm}/holdout", proposal["observation"]["image"])
                        proposal["holdout"] = read_score(raw["score"], center, bit, self.spec)
                    # Also evaluate a successfully rendered RGB whose read/design failed.
                    elif "render" in proposal:
                        raw = holdout(key + f"/late/{arm}/holdout", proposal["render"]["image"])
                        proposal["holdout"] = read_score(raw["score"], center, bit, self.spec)
                    proposal["raw_three_channel_success"] = (proposal["accepted"] and
                        all(proposal["reads"][n]["bit_correct"] for n in ("clean", "design"))
                        and proposal["holdout"]["bit_correct"]) if "observation" in proposal else None
                    selected = proposal["selected"] == "proposal"
                    reads = proposal["reads"] if selected else row["E_reads"]
                    h = proposal["holdout"] if selected else row["E_holdout"]
                    relative = proposal["late_relative"] if selected else 0.0
                    proposal["returned_reads"] = {**reads, "holdout": h}
                    proposal["returned_three_channel_success"] = (
                        common_accept(reads, row["early_relative"], relative, self.rho)
                        and all(r["bit_correct"] for r in (*reads.values(), h)))
                except Exception as error:
                    row["errors"].append(f"{arm} holdout/returned evaluation: {error}")
            for arm in ("B", "C"):
                row[arm + "_success"] = row["arms"][arm]["returned_three_channel_success"]
            A, B, C = (row["arms"][arm] for arm in ("A", "B", "C"))
            row["A_clean_repair_design_damage"] = bool("reads" in A and A["action"]["nonzero"]
                and A["reads"]["clean"]["fixed_inner"] and not A["reads"]["design"]["fixed_inner"])
            row["mechanism_case"] = bool(row.get("active_early_flip") and
                row.get("late_status") == "repair_attempted" and row["A_clean_repair_design_damage"]
                and row.get("C_B_actual_different") and C.get("action", {}).get("nonzero")
                and C.get("selected") == "proposal" and C.get("protection_realized")
                and C.get("raw_three_channel_success"))
            if "quality" in B and "quality" in C:
                row["C_minus_B_rgb_mse"] = C["quality"]["mse"] - B["quality"]["mse"]
            if "holdout" in C and "E_holdout" in row:
                row["C_lost_E_holdout_correctness"] = row["E_holdout"]["bit_correct"] and not C["holdout"]["bit_correct"]
                row["C_minus_E_holdout_target_margin"] = C["holdout"]["signed_margin"] - row["E_holdout"]["signed_margin"]
            row["evaluation_complete"] = not row["errors"]
        atomic_torch(self.store.directory / "evaluated.pt", frozen)
        # Compact JSON retains metrics/statuses, large tensors/RGB remain in .pt.
        rows = [compact(r) for r in frozen["trials"]]
        summary = dict(plan=describe_plan(self.config), trials=rows,
            verdict=grouped_verdict(rows, [s["source_id"] for s in self.config["sources"]]),
            physical_counts=physical_counts(self.store.counters), raw_counts=dict(self.store.counters),
            baseline_holdout_errors=baseline_errors, holdout_image_reuse=aliases,
            resume_operation_reuse=self.store.cache_reuses, operations=self.store.inventory(),
            engineering_complete=not baseline_errors and all(r["evaluation_complete"] for r in rows))
        if baseline_errors:
            verdict = summary["verdict"]
            # Preserve trial-only findings as diagnostics, while a missing
            # unmarked holdout makes the complete run engineering-incomplete.
            verdict["trial_only_outcome"] = verdict["outcome"]
            verdict.update(operational_P2_signal=False, outcome="ENGINEERING_INCOMPLETE",
                           group_diagnostics_only=True,
                           quality_tradeoff_review="deferred_engineering_incomplete")
            for group in verdict["groups"]:
                group["trial_only_operational_signal"] = group["operational_signal"]
                group["operational_signal"] = False
        atomic_json(self.store.directory / "summary.json", summary)
        return summary


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in {"E", "E_render", "observation", "render", "feature", "image", "design_image", "requested", "delta", "actual", "after"}}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), norm=float(torch.linalg.vector_norm(value.double())))
    return value


def execute(config, output, *, provider_factory=StageProtectionProvider):
    # Prevent overlapping runs in one Linux/Colab VM. Drive does not promise a
    # cross-VM lock; the notebook explicitly requires one active VM per run.
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = OperationStore(output, config)
        provider = provider_factory(config, counters=store.counters)
        try:
            canary = Canary(config, store, provider)
            return canary.evaluate(canary.freeze())
        finally:
            provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly run the fixed P2 on real models")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage-protection-v2/attempt01"))
    args = parser.parse_args(argv)
    config = load_config()
    if not args.execute:
        print(json.dumps(describe_plan(config), indent=2))
        return 0
    summary = execute(config, args.output)
    print(json.dumps({"output": str(args.output), "engineering_complete": summary["engineering_complete"],
                      "verdict": summary["verdict"], "physical_counts": summary["physical_counts"]}, indent=2))
    return 0 if summary["engineering_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
