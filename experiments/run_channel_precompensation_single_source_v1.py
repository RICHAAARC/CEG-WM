"""Fixed first-source method trial with a coherent runtime suffix basepoint.

No new full generation, parameter search, retry, or fitted reader. The existing
four-source configuration remains unchanged; only its first source executes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.run_channel_precompensation_v1 import (
    BITS, METHODS, execute_measurement, validate_config,
)
from experiments.run_reconstruction_channel_v1 import _write_json


def describe_single_plan(config, *, checkpoint=None):
    validate_config(config)
    selected = config["sources"][0]["source_id"]
    result = {
        "mode": "plan_only", "scope": "runtime_suffix_single", "source_id": selected,
        "full_roster": [{"source_id": source["source_id"],
                         "scope_status": "selected_by_fixed_order" if index == 0 else "not_measured_in_this_scope"}
                        for index, source in enumerate(config["sources"])],
        "methods": list(METHODS), "bits": list(BITS),
        "planned_budget": {"source_count": 1, "marked_final_images": 4,
            "new_runtime_zero_images": 1, "final_generation_calls": 0,
            "source_suffix_forward_calls": 6, "source_suffix_denoising_steps": 6,
            "source_suffix_decode_calls": 6, "latent_vjps": 2, "rgb_vjps": 2,
            "autograd_calls": 4, "final_reconstruction_calls": 10,
            "auxiliary_reconstruction_forward_calls": 1,
            "total_reconstruction_forward_calls": 11, "public_feature_reads": 15,
            "convention": "Upper bounds when all four fixed writes solve; failures remain in denominators. No extra probe or retry."},
        "fixed_denominators": {"bit_trials_per_method": 2, "both_bits_per_source_per_method": 1},
        "basepoint": "Recompute clean/design scores and RGB sensitivities at the current suffix zero. Historical zero is diagnostic only.",
        "write_path": "The same captured latent, conditioning and scheduler feed every zero, VJP and marked suffix.",
        "claim": "Single-source falsification of fixed-reader one-bit control; not four-source completion, watermark presence, FPR, or robustness qualification.",
    }
    if checkpoint is not None:
        result.update(resume_from=str(checkpoint.batch_directory),
                      prior_actual_calls=dict(checkpoint.prior_actual_calls))
    return result


def add_response_assessment(summary, config):
    """Compare the actual-cast prediction to the ordinary RGB reads already made.

    No response-based candidate selection, new probe, correction or threshold is
    introduced. Signed changes, residuals and outcomes are reported separately.
    """
    design_seed = str(config["reconstruction"]["design_seed"])
    holdout_seed = str(config["reconstruction"]["holdout_seed"])
    for source in summary["sources"]:
        baseline = [source["zero"]["clean"], source["zero"]["reconstruction"][design_seed]]
        for method in METHODS:
            for bit in BITS:
                trial = source["trials"][method][str(bit)]
                write = trial.get("write") or {}
                prediction = write.get("actual_linear_prediction") or {}
                observed = [trial["clean"], trial["reconstruction"][design_seed]]
                rows = []
                if prediction.get("predicted_scores") is not None:
                    for index, name in enumerate(("clean", "design")):
                        if not baseline[index] or not observed[index]:
                            continue
                        if baseline[index].get("status") != "OBSERVED" or observed[index].get("status") != "OBSERVED":
                            continue
                        base = baseline[index]["score"]
                        predicted_change = prediction["predicted_scores"][index] - base
                        actual_change = observed[index]["score"] - base
                        rows.append({"channel": name, "baseline_score": base,
                            "predicted_score": prediction["predicted_scores"][index],
                            "observed_score": observed[index]["score"],
                            "predicted_change": predicted_change, "observed_change": actual_change,
                            "residual": actual_change - predicted_change,
                            "residual_in_cell_units": (actual_change - predicted_change) / config["reader"]["delta"],
                            "opposite_sign": predicted_change * actual_change < 0,
                            "response_ratio": actual_change / predicted_change if predicted_change != 0 else None})
                mechanisms = []
                if trial.get("errors"):
                    mechanisms.append("ACQUISITION_OR_PREPARATION_FAILURE")
                solve = trial.get("solve")
                if solve and solve["status"] != "SOLVED":
                    mechanisms.append("FIXED_CONSTRAINTS_NOT_SOLVED")
                if write:
                    if not write.get("within_budget"):
                        mechanisms.append("ACTUAL_DTYPE_BUDGET_EXCEEDED")
                    if write.get("quantization_collapsed"):
                        mechanisms.append("ACTUAL_DTYPE_WRITE_COLLAPSED")
                    if prediction and not prediction.get("within_inner_cells"):
                        mechanisms.append("ACTUAL_DTYPE_LINEAR_TARGET_MISS")
                if any(row["opposite_sign"] for row in rows):
                    mechanisms.append("OBSERVED_RESPONSE_OPPOSES_PREDICTION")
                observed_pair = len(rows) == 2
                pair_inner = observed_pair and all(item.get("inside_target_inner_cell") for item in observed)
                if observed_pair and prediction.get("within_inner_cells") and not pair_inner:
                    mechanisms.append("FINITE_BUDGET_PREDICTION_MISS")
                holdout = trial["reconstruction"][holdout_seed]
                if pair_inner and holdout and holdout.get("status") == "OBSERVED" and not holdout.get("inside_target_inner_cell"):
                    mechanisms.append("HOLDOUT_RECONSTRUCTION_TARGET_MISS")
                trial["response_assessment"] = {"channels": rows,
                    "mechanisms": mechanisms,
                    "interpretation": "Finite-budget observation of this fixed write; not an exact derivative proof or an independent validation set."}
    summary["method_validation"] = {
        "marked_rgb_returned": sum(bool((trial.get("clean") or {}).get("status") == "OBSERVED")
            for source in summary["sources"] for trials in source["trials"].values() for trial in trials.values()),
        "both_bits_required": True, "four_source_experiment_complete": False,
        "historical_pixel_match_required": False,
        "current_basepoint_and_rgb_sensitivity_alignment_required": True,
        "reader_or_budget_tuning": False,
    }
    return summary


def execute_single(config, output, *, resume_from, provider=None):
    result = execute_measurement(config, output, provider=provider, resume_from=resume_from,
                                 source_scope="runtime_suffix_single")
    add_response_assessment(result, config)
    # The saved capture checkpoint predates the three-arm diagnostic. Include
    # its already completed work when that exact sibling ledger is available;
    # an unavailable ledger never blocks the method trial or invents counts.
    if resume_from is not None:
        diagnostic_path = Path(resume_from).parent / "suffix_diagnostic_v1/summary.json"
        result["prior_calls_scope"] = "capture checkpoint; later diagnostic not included"
        try:
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            prior = result["prior_actual_calls"]
            ancestor = diagnostic["prior_actual_calls"]
            extra = diagnostic["actual_calls"]
            aligned = all(prior.get(key, 0) == ancestor.get(key, 0) for key in prior.keys() | ancestor.keys())
            stages = diagnostic.get("stages", {})
            terminal = stages.get("cleanup") == "complete" and "started" not in stages.values()
            if aligned and terminal and all(type(value) is int and value >= 0 for value in extra.values()):
                prior = {key: prior.get(key, 0) + extra.get(key, 0) for key in prior.keys() | extra.keys()}
                result["prior_actual_calls"] = prior
                new = result["actual_calls"]
                result["cumulative_actual_calls"] = {key: prior.get(key, 0) + new.get(key, 0)
                                                     for key in prior.keys() | new.keys()}
                result["prior_calls_scope"] = "capture checkpoint plus completed three-arm suffix diagnostic"
                _write_json(Path(output) / "prior_suffix_diagnostic_summary.json", diagnostic)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    _write_json(Path(output) / "summary.json", result)
    for source in result["sources"]:
        _write_json(Path(output) / source["source_id"] / "measurement.json", source)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None and args.resume_from is None:
        parser.error("provide --config or --resume-from")
    path = args.config if args.config is not None else args.resume_from / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(describe_single_plan(config), indent=2, allow_nan=False))
        return 0
    if args.resume_from is None or args.output is None:
        parser.error("execution requires --resume-from and a new --output")
    if args.output.exists():
        summary = args.output / "summary.json"
        if summary.is_file():
            print(summary.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"status": "EXISTING_OUTPUT_READ_ONLY", "path": str(args.output)}))
        return 0
    result = execute_single(config, args.output, resume_from=args.resume_from)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(any(source["status"] != "collected" for source in result["sources"]))


if __name__ == "__main__":
    raise SystemExit(main())
