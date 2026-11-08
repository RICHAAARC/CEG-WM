"""Plan or explicitly collect the fixed one-shot channel-precompensation trial.

The default command validates a configuration without loading models. Data
collection, blind bit readout, quality eligibility, and scientific conclusions
remain separate. No retry, source selection, reader fitting, or scientific PASS
is performed here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from experiments.run_reconstruction_channel_v1 import (
    _integer, _number, _write_json, pixel_distortion, validate_config as _validate_channel,
)

METHODS = ("joint", "gradient")
BITS = (0, 1)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    for name in ("generation", "reconstruction", "reader", "write", "quality"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"{name} must be explicitly specified")
    reader, write, reconstruction = (config[name] for name in ("reader", "write", "reconstruction"))
    _integer(reader.get("key_seed"), "reader.key_seed")
    _number(reader.get("delta"), "reader.delta", positive=True)
    _number(2 * reader["delta"], "reader two-cell period", positive=True)
    fraction = _number(reader.get("inner_fraction"), "reader.inner_fraction", positive=True)
    if fraction >= 0.5:
        raise ValueError("reader.inner_fraction must be strictly between zero and one half")
    _number(write.get("relative_l2_budget"), "write.relative_l2_budget", positive=True)
    if "minimum_psnr_db" not in config["quality"]:
        raise ValueError("quality.minimum_psnr_db must be explicit (null means report only)")
    minimum_psnr = config["quality"]["minimum_psnr_db"]
    if minimum_psnr is not None:
        _number(minimum_psnr, "quality.minimum_psnr_db", positive=True)
    for name in ("design_seed", "holdout_seed"):
        _integer(reconstruction.get(name), f"reconstruction.{name}")
    if reconstruction["design_seed"] == reconstruction["holdout_seed"]:
        raise ValueError("design_seed and holdout_seed must differ")
    if "seeds" in reconstruction:
        raise ValueError("use explicit design_seed and holdout_seed, not reconstruction.seeds")
    if len(config.get("sources", [])) != 4:
        raise ValueError("this first candidate requires exactly four fixed sources")
    # Reuse existing concrete SD3/SDXL configuration checks, without loading a
    # pipeline or inheriting the previous candidate's probe/reader semantics.
    channel = deepcopy(config)
    channel["probe"] = {"seed": reader["key_seed"], "relative_l2": write["relative_l2_budget"]}
    channel["feature"] = "full_vae_mode"
    channel["reconstruction"]["seeds"] = [reconstruction["design_seed"], reconstruction["holdout_seed"]]
    _validate_channel(channel)
    if config.get("feature", "full_vae_mode") != "full_vae_mode":
        raise ValueError("the reader requires the full ordinary-RGB VAE mode feature")
    return config


def describe_plan(config: dict[str, Any], *, resume_checkpoint=None) -> dict[str, Any]:
    validate_config(config)
    count = len(config["sources"])
    plan = {
        "mode": "plan_only", "method": "two_channel_one_shot_interval_precompensation",
        "source_ids": [source["source_id"] for source in config["sources"]],
        "methods": list(METHODS), "bits": list(BITS),
        "planned_budget": {
            "source_count": count, "marked_final_images": 4 * count,
            "shared_zero_images": count, "final_generation_calls": 5 * count,
            "final_reconstruction_calls": 10 * count, "public_feature_reads": 15 * count,
            "auxiliary_reconstruction_forward_calls": count,
            "total_reconstruction_forward_calls": 11 * count,
            "latent_vjps": 2 * count,
            "rgb_vjps": 2 * count,
            "auxiliary_calls": (
                f"The {count} differentiable proxy reconstruction forwards are separate from the "
                f"{10 * count} formal ordinary-RGB reconstructions. All additional suffix, reader, "
                "and RGB-gradient calls are recorded separately; counters may describe overlapping components."
            ),
            "convention": "Final-call ceilings when every write solves; infeasible writes are not generated and remain in fixed denominators.",
        },
        "fixed_denominators": {"bit_trials_per_method": 2 * count, "both_bits_per_source_per_method": count},
        "reader": config["reader"], "write": config["write"], "quality": config["quality"],
        "generation": config["generation"], "reconstruction": config["reconstruction"],
        "claim": "Small falsification trial only; no watermark-presence, FPR, or scientific PASS.",
    }
    if resume_checkpoint is not None:
        plan["resume_from"] = str(resume_checkpoint.batch_directory)
        plan["imported_counts"] = dict(resume_checkpoint.reused_counts)
        plan["prior_actual_calls"] = dict(resume_checkpoint.prior_actual_calls)
        plan["planned_budget"].update(shared_zero_images=0, final_generation_calls=4 * count,
            final_reconstruction_calls=8 * count, public_feature_reads=12 * count,
            total_reconstruction_forward_calls=9 * count,
            auxiliary_calls=(f"New work only: up to {count} proxy reconstruction forwards, {count} source suffix forwards, "
                f"{2 * count} latent and {2 * count} RGB VJPs; additional reader forwards remain separate. "
                "Saved zero images/features are reused. Compatible saved RGB sensitivities may reduce new auxiliary calls."))
    return plan


def load_provider(config, *, resume_from=None, resume_checkpoint=None):
    """Lazy real-model entry; imported only after explicit execution selection."""
    from cegwm.runtime.channel_precompensation_provider import PrecompensationProvider
    if resume_checkpoint is not None:
        return PrecompensationProvider(config, resume_checkpoint=resume_checkpoint)
    if resume_from is not None:
        return PrecompensationProvider(config, resume_from=resume_from)
    return PrecompensationProvider(config)


def _jsonable(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _trial(method, bit, seeds):
    return {"method": method, "bit": bit, "status": "not_attempted", "solve": None,
            "clean": None, "reconstruction": {str(seed): None for seed in seeds},
            "quality": None, "write": None, "errors": {}, "collection_complete": False,
            "blind_all_conditions_correct": False, "blind_all_conditions_inner_margin": False,
            "eligible_joint_success": False}


def _save_failure_partial(directory: Path, event: dict[str, Any], index: int) -> dict[str, Any]:
    """Keep failed writer-side tensors and any already-produced ordinary RGB."""
    import torch
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    partial = event.get("partial", {})
    partial = dict(partial) if isinstance(partial, dict) else {"value": partial}
    saved_images = {}
    for field, value in list(partial.items()):
        if isinstance(value, Image.Image):
            safe_field = "".join(character if character.isalnum() or character == "_" else "_"
                                 for character in str(field))
            filename = f"failure_{index:04d}_{safe_field}.png"
            value.save(directory / filename)
            saved_images[str(field)] = filename
            del partial[field]
    filename = f"failure_{index:04d}.pt"
    torch.save({"stage": event.get("stage"), "error": event.get("error"),
                "partial": partial, "saved_images": saved_images}, directory / filename)
    return {"stage": str(event.get("stage", "prepare")), "error": str(event.get("error", "provider failure")),
            "partial_file": filename, "saved_images": saved_images}


def _finalize_trials(summary, config):
    """Fixed denominators include unsolved, failed, low-quality, and missing trials."""
    required_margin = config["reader"]["delta"] * (0.5 - config["reader"]["inner_fraction"])
    aggregates = {}
    for method in METHODS:
        trials = [source["trials"][method][str(bit)] for source in summary["sources"] for bit in BITS]
        for trial in trials:
            observations = [trial["clean"], *trial["reconstruction"].values()]
            complete = bool(all(item and item.get("status") == "OBSERVED" for item in observations)
                            and trial.get("write", {}).get("matched_state") is True
                            and trial.get("write", {}).get("actual_l2") is not None
                            and not trial["errors"]) if trial.get("write") else False
            trial["collection_complete"] = complete
            trial["blind_all_conditions_correct"] = bool(complete and all(
                item["decoded_bit"] == trial["bit"] for item in observations))
            trial["blind_all_conditions_inner_margin"] = bool(complete and all(
                item.get("inside_target_inner_cell", item["signed_margin"] >= required_margin)
                for item in observations))
            quality = trial.get("quality")
            quality_ok = bool(quality and quality.get("eligible"))
            write_ok = bool(trial.get("write") and not trial["write"].get("quantization_collapsed", False)
                            and trial["write"].get("within_budget", False))
            trial["eligible_joint_success"] = bool(trial["blind_all_conditions_inner_margin"] and quality_ok and write_ok)
            if trial["errors"]:
                trial["status"] = "failed"
            elif complete:
                trial["status"] = "collected"
        aggregates[method] = {
            "expected_bit_trials": 2 * len(summary["sources"]),
            "collected_bit_trials": sum(trial["collection_complete"] for trial in trials),
            "blind_all_conditions_correct": sum(trial["blind_all_conditions_correct"] for trial in trials),
            "blind_all_conditions_inner_margin": sum(trial["blind_all_conditions_inner_margin"] for trial in trials),
            "eligible_joint_successes": sum(trial["eligible_joint_success"] for trial in trials),
            "expected_both_bit_sources": len(summary["sources"]),
            "both_bits_collected_sources": sum(all(source["trials"][method][str(bit)]["collection_complete"]
                                                    for bit in BITS) for source in summary["sources"]),
            "both_bits_blind_all_conditions_correct": sum(all(source["trials"][method][str(bit)]["blind_all_conditions_correct"]
                                                              for bit in BITS) for source in summary["sources"]),
            "both_bits_eligible_joint_successes": sum(all(source["trials"][method][str(bit)]["eligible_joint_success"]
                                                         for bit in BITS) for source in summary["sources"]),
            "status_counts": {status: sum(trial["status"] == status for trial in trials)
                              for status in sorted({trial["status"] for trial in trials})},
        }
    summary["method_summaries"] = aggregates


def execute_measurement(config: dict[str, Any], output: Path, *, provider=None, resume_from=None) -> dict[str, Any]:
    """Collect one create-only batch with a staged real or injected fake provider.

    Provider methods yield dictionaries, allowing each observation/failure to be
    saved immediately. ``prepare_sources`` emits baseline/design/linearization/
    failure events. ``generate_writes`` consumes source_id/method/bit/delta.
    ``reconstruct_outputs`` consumes source_id/method/bit/image/seed. The provider
    owns model lifecycle and exposes cumulative ``counts`` and ``close()``.
    """
    import torch
    from cegwm.channel_precompensation import ReaderSpec, make_reader, decode_score, solve_write, evaluate_delta

    checkpoint = None
    if resume_from is not None:
        from cegwm.runtime.channel_precompensation_resume import load_resume_checkpoint
        checkpoint = load_resume_checkpoint(resume_from, config=config)
    plan = describe_plan(config, resume_checkpoint=checkpoint)
    output = Path(output)
    if checkpoint is not None and output.resolve().is_relative_to(checkpoint.batch_directory):
        raise ValueError("continuation output must not modify the original batch directory")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    seeds = [config["reconstruction"][name] for name in ("design_seed", "holdout_seed")]
    rows = [{"source_id": source["source_id"], "status": "not_attempted", "errors": {},
             "zero": {"clean": None, "reconstruction": {str(seed): None for seed in seeds}},
             "writer": {"status": "not_attempted"},
             "trials": {method: {str(bit): _trial(method, bit, seeds) for bit in BITS} for method in METHODS}}
            for source in config["sources"]]
    indexed = {row["source_id"]: row for row in rows}
    contexts = {identifier: {} for identifier in indexed}
    summary = {"mode": "measurement", "planned_budget": plan["planned_budget"],
               "fixed_denominators": plan["fixed_denominators"], "sources": rows, "actual_calls": {},
               "stages": {name: "not_started" for name in ("setup", "prepare", "solve", "generate", "reconstruct", "cleanup")},
               "claim": plan["claim"]}
    if checkpoint is not None:
        summary.update(resume_from=str(checkpoint.batch_directory),
            prior_actual_calls=dict(checkpoint.prior_actual_calls), imported_counts=dict(checkpoint.reused_counts),
            reused_counts={"source_captures": 0, "zero_generation_images": 0, "zero_reconstruction_images": 0,
                           "public_zero_features": 0, "reader_assets": 1, "rgb_sensitivities": 0},
            prior_evidence="prior_summary.json")
        _write_json(output / "prior_summary.json", checkpoint.prior_summary)
    spec = ReaderSpec(**config["reader"])
    reader_vector = None if checkpoint is None else checkpoint.reader_vector.clone()
    phase = None if checkpoint is None else checkpoint.reader_phase
    if checkpoint is not None:
        torch.save({"vector": reader_vector, "phase": phase, "spec": config["reader"]}, output / "reader.pt")
        summary["reader"] = {**config["reader"], "feature_dimension": reader_vector.numel(), "phase": phase}

    def persist():
        if provider is not None:
            summary["actual_calls"] = _jsonable(dict(provider.counts))
        if checkpoint is not None:
            new, prior = summary["actual_calls"], summary["prior_actual_calls"]
            if any(type(value) is not int or value < 0 for value in new.values()):
                raise ValueError("new physical call counters must be nonnegative integers")
            summary["cumulative_actual_calls"] = {name: prior.get(name, 0) + new.get(name, 0)
                                                 for name in prior.keys() | new.keys()}
        _finalize_trials(summary, config)
        _write_json(output / "summary.json", summary)
        for row in rows:
            directory = output / row["source_id"]
            directory.mkdir(exist_ok=True)
            _write_json(directory / "measurement.json", row)

    def observation(event, relative, bit=None):
        nonlocal reader_vector, phase
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image, feature = event.get("image"), event.get("feature")
        if image is not None:
            image.save(path.with_suffix(".png"))
        torch.save({"feature": feature, "status": event.get("status"), "error": event.get("error")}, path.with_suffix(".pt"))
        if event.get("error") or event.get("status") == "FAILED":
            return {"status": "FAILED", "error": str(event.get("error", "provider reported failure")),
                    "image_saved": image is not None, "feature_saved": feature is not None}
        if image is None or feature is None:
            return {"status": "MISSING", "image_saved": image is not None, "feature_saved": feature is not None}
        if not isinstance(feature, torch.Tensor) or feature.device.type != "cpu" or feature.ndim != 1:
            raise ValueError("public feature must be a flattened CPU tensor")
        if not feature.is_floating_point() or not bool(torch.isfinite(feature).all()):
            raise ValueError("public feature must be finite and floating")
        if reader_vector is None:
            reader_vector, phase = make_reader(feature.numel(), spec)
            torch.save({"vector": reader_vector, "phase": phase, "spec": config["reader"]}, output / "reader.pt")
            summary["reader"] = {**config["reader"], "feature_dimension": feature.numel(), "phase": phase}
        if feature.numel() != reader_vector.numel():
            raise ValueError("public feature dimension changed between observations")
        score = float(torch.dot(feature.double(), reader_vector.double()))
        return {"status": "OBSERVED", "score": score, **decode_score(score, spec, phase, target_bit=bit)}

    def source_failure(identifier, stage, error):
        row = indexed[identifier]
        row["errors"][stage] = str(error)
        row["status"] = "failed"
        row["writer"]["status"] = "failed"

    persist()
    try:
        summary["stages"]["setup"] = "started"
        persist()
        if provider is None:
            provider = (load_provider(config) if checkpoint is None else
                        load_provider(config, resume_from=resume_from, resume_checkpoint=checkpoint))
        summary["stages"]["setup"] = "complete"
        summary["stages"]["prepare"] = "started"
        persist()
        prepared_events = set()
        try:
            for event in provider.prepare_sources(config["sources"]):
                identifier, kind = event["source_id"], event["kind"]
                if identifier not in indexed:
                    raise ValueError("provider emitted an unplanned source")
                row, context = indexed[identifier], contexts[identifier]
                try:
                    if kind != "failure":
                        event_key = (identifier, kind, event.get("phase") if kind == "rgb_gradient" else None)
                        if event_key in prepared_events:
                            raise ValueError("provider repeated a preparation observation")
                        prepared_events.add(event_key)
                    if kind == "baseline":
                        context["baseline"] = event
                        row["zero"]["clean"] = observation(event, Path(identifier) / "zero/clean")
                        writer_dir = output / identifier / "writer"
                        writer_dir.mkdir(exist_ok=True)
                        torch.save({name: value for name, value in event.items() if name not in {"image", "feature"}},
                                   writer_dir / "baseline.pt")
                        row["status"] = "prepared_baseline"
                        if checkpoint is not None and event.get("reused") and row["zero"]["clean"]["status"] == "OBSERVED":
                            summary["reused_counts"]["source_captures"] += 1
                            summary["reused_counts"]["zero_generation_images"] += 1
                            summary["reused_counts"]["public_zero_features"] += 1
                    elif kind == "design":
                        context["design"] = event
                        row["zero"]["reconstruction"][str(seeds[0])] = observation(
                            event, Path(identifier) / f"zero/reconstruction_{seeds[0]}")
                        if checkpoint is not None and event.get("reused") and row["zero"]["reconstruction"][str(seeds[0])]["status"] == "OBSERVED":
                            summary["reused_counts"]["zero_reconstruction_images"] += 1
                            summary["reused_counts"]["public_zero_features"] += 1
                    elif kind == "holdout_zero":
                        if checkpoint is None or event.get("seed", seeds[1]) != seeds[1] or not event.get("reused"):
                            raise ValueError("holdout_zero is reserved for the saved fixed holdout observation")
                        row["zero"]["reconstruction"][str(seeds[1])] = observation(
                            event, Path(identifier) / f"zero/reconstruction_{seeds[1]}")
                        if row["zero"]["reconstruction"][str(seeds[1])]["status"] == "OBSERVED":
                            summary["reused_counts"]["zero_reconstruction_images"] += 1
                            summary["reused_counts"]["public_zero_features"] += 1
                    elif kind == "rgb_gradient":
                        gradient_phase = event.get("phase")
                        if gradient_phase not in {"clean", "design"} or not isinstance(event.get("result"), dict):
                            raise ValueError("RGB sensitivity event requires a clean/design phase and result")
                        writer_dir = output / identifier / "writer"
                        writer_dir.mkdir(exist_ok=True)
                        result = event["result"]
                        if result.get("image") is not None:
                            result["image"].save(writer_dir / f"rgb_gradient_{gradient_phase}.png")
                        evidence = {**event, "result": {name: value for name, value in result.items() if name != "image"}}
                        torch.save(evidence, writer_dir / f"rgb_gradient_{gradient_phase}.pt")
                        row["writer"].setdefault("rgb_gradients", {})[gradient_phase] = {
                            "file": f"rgb_gradient_{gradient_phase}.pt",
                            "compatibility_version": event.get("compatibility_version"),
                            "reusable": event.get("reusable") is True, "reused": event.get("reused") is True}
                        if checkpoint is not None and event.get("reused"):
                            summary["reused_counts"]["rgb_sensitivities"] += 1
                    elif kind == "linearization":
                        context["linearization"] = event
                        writer_dir = output / identifier / "writer"
                        writer_dir.mkdir(exist_ok=True)
                        torch.save(event, writer_dir / "linearization.pt")
                        row["writer"].update(status="failed" if row["errors"] else "linearized",
                                             diagnostics=_jsonable(event.get("diagnostics", {})))
                    elif kind == "failure":
                        records = row["writer"].setdefault("failure_evidence", [])
                        records.append(_save_failure_partial(output / identifier / "writer", event, len(records)))
                        source_failure(identifier, event.get("stage", "prepare"), event.get("error", "provider failure"))
                    else:
                        raise ValueError(f"unknown prepare event kind: {kind}")
                except Exception as error:
                    source_failure(identifier, kind, f"{type(error).__name__}: {error}")
                persist()
        except Exception as error:
            summary["prepare_error"] = f"{type(error).__name__}: {error}"
        summary["stages"]["prepare"] = "complete" if not summary.get("prepare_error") else "failed"
        summary["stages"]["solve"] = "started"
        requests = []
        for row in rows:
            identifier, context = row["source_id"], contexts[row["source_id"]]
            clean, design = row["zero"]["clean"], row["zero"]["reconstruction"][str(seeds[0])]
            if row["errors"] or not all(item and item.get("status") == "OBSERVED" for item in (clean, design)) or "linearization" not in context:
                for method in METHODS:
                    for bit in BITS:
                        trial = row["trials"][method][str(bit)]
                        trial["status"] = "preparation_failed"
                        trial["errors"]["prepare"] = "missing or failed baseline/design/linearization"
                persist()
                continue
            try:
                baseline = context["baseline"]["before_state"]["latents"]
                budget = config["write"]["relative_l2_budget"] * float(torch.linalg.vector_norm(baseline.double()))
                context["l2_budget"] = budget
                gradients = context["linearization"]["gradients"]
                if tuple(gradients.shape) != (2, *baseline.shape):
                    raise ValueError("two writer gradients must match the captured latent shape")
                for method in METHODS:
                    for bit in BITS:
                        trial = row["trials"][method][str(bit)]
                        result = solve_write(gradients, [clean["score"], design["score"]], spec=spec, phase=phase,
                                             bit=bit, l2_budget=budget, method=method)
                        trial["solve"] = _jsonable({key: value for key, value in result.items() if key != "delta"})
                        trial["status"] = result["status"].lower()
                        if result["status"] == "SOLVED":
                            delta = result["delta"]
                            context.setdefault("requested_deltas", {})[(method, bit)] = delta
                            torch.save({"requested_delta": delta, "l2_budget": budget},
                                       output / identifier / "writer" / f"{method}_bit{bit}.pt")
                            requests.append({"source_id": identifier, "method": method, "bit": bit, "delta": delta})
            except Exception as error:
                source_failure(identifier, "solve", f"{type(error).__name__}: {error}")
                for method in METHODS:
                    for bit in BITS:
                        row["trials"][method][str(bit)]["errors"]["solve"] = str(error)
                requests = [request for request in requests if request["source_id"] != identifier]
            persist()
        summary["stages"]["solve"] = "complete"
        summary["stages"]["generate"] = "started"
        requested = {(item["source_id"], item["method"], item["bit"]) for item in requests}
        seen = set()
        try:
            for event in provider.generate_writes(requests):
                key = (event["source_id"], event["method"], event["bit"])
                if key not in requested or key in seen:
                    raise ValueError("provider emitted an unplanned or duplicate write")
                seen.add(key)
                identifier, method, bit = key
                row, context = indexed[identifier], contexts[identifier]
                trial = row["trials"][method][str(bit)]
                try:
                    name = f"{method}_bit{bit}"
                    trial["clean"] = observation(event, Path(identifier) / name / "clean", bit)
                    torch.save({field: value for field, value in event.items() if field not in {"image", "feature"}},
                               output / identifier / "writer" / f"{name}_actual.pt")
                    if event.get("error"):
                        trial["errors"]["generation"] = str(event["error"])
                    delta = event.get("actual_delta")
                    if delta is not None and (not isinstance(delta, torch.Tensor) or delta.device.type != "cpu"
                            or not delta.is_floating_point() or not bool(torch.isfinite(delta).all())):
                        raise ValueError("actual write delta must be a finite floating CPU tensor")
                    if delta is not None and delta.shape != context["baseline"]["before_state"]["latents"].shape:
                        raise ValueError("actual write delta must match the captured latent shape")
                    requested_delta = context["requested_deltas"][(method, bit)]
                    requested_norm = float(torch.linalg.vector_norm(requested_delta.double()))
                    actual_norm = None if delta is None else float(torch.linalg.vector_norm(delta.double()))
                    budget = context["l2_budget"]
                    trial["write"] = {"matched_state": event.get("matched_state") is True,
                        "requested_l2": requested_norm, "actual_l2": actual_norm, "l2_budget": budget,
                        "within_budget": actual_norm is not None and actual_norm <= budget,
                        "intentional_noop": requested_norm == 0,
                        "quantization_collapsed": requested_norm > 0 and actual_norm == 0}
                    if delta is not None:
                        trial["write"]["actual_linear_prediction"] = evaluate_delta(delta,
                            context["linearization"]["gradients"],
                            [row["zero"]["clean"]["score"], row["zero"]["reconstruction"][str(seeds[0])]["score"]],
                            bit=bit, spec=spec, phase=phase, l2_budget=budget)
                    if event.get("image") is not None:
                        context.setdefault("marked_images", {})[(method, bit)] = event["image"]
                        distortion = pixel_distortion(context["baseline"]["image"], event["image"])
                        threshold = config["quality"]["minimum_psnr_db"]
                        quality_ok = distortion.get("status") == "measured" and (threshold is None
                            or distortion["identical_pixels"] or distortion["psnr_db"] >= threshold)
                        trial["quality"] = {**distortion, "minimum_psnr_db": threshold,
                                            "eligible": bool(quality_ok), "gate_enabled": threshold is not None}
                    if not trial["write"]["matched_state"]:
                        trial["errors"]["state"] = "write replay did not match the baseline state"
                except Exception as error:
                    trial["errors"]["generation_record"] = f"{type(error).__name__}: {error}"
                persist()
        except Exception as error:
            summary["generation_error"] = f"{type(error).__name__}: {error}"
        for identifier, method, bit in requested - seen:
            indexed[identifier]["trials"][method][str(bit)]["errors"]["generation"] = "requested write was not returned"
        summary["stages"]["generate"] = "complete" if not summary.get("generation_error") else "failed"
        summary["stages"]["reconstruct"] = "started"
        reconstruction_requests = []
        for identifier, context in contexts.items():
            baseline = context.get("baseline", {})
            saved_holdout = indexed[identifier]["zero"]["reconstruction"][str(seeds[1])]
            if baseline.get("image") is not None and not (saved_holdout and saved_holdout.get("status") == "OBSERVED"):
                # The design-seed zero image was already collected in prepare.
                reconstruction_requests.append({"source_id": identifier, "method": "zero", "bit": None,
                                                "seed": seeds[1], "image": baseline["image"]})
            for (method, bit), image in context.get("marked_images", {}).items():
                for seed in seeds:
                    reconstruction_requests.append({"source_id": identifier, "method": method, "bit": bit,
                                                    "seed": seed, "image": image})
        expected = {(item["source_id"], item["method"], item["bit"], item["seed"]) for item in reconstruction_requests}
        seen = set()
        try:
            for event in provider.reconstruct_outputs(reconstruction_requests):
                key = (event["source_id"], event["method"], event["bit"], event["seed"])
                if key not in expected or key in seen:
                    raise ValueError("provider emitted an unplanned or duplicate reconstruction")
                seen.add(key)
                identifier, method, bit, seed = key
                row = indexed[identifier]
                target = row["zero"] if method == "zero" else row["trials"][method][str(bit)]
                name = "zero" if method == "zero" else f"{method}_bit{bit}"
                try:
                    target["reconstruction"][str(seed)] = observation(
                        event, Path(identifier) / name / f"reconstruction_{seed}", bit)
                    if event.get("error"):
                        (row["errors"] if method == "zero" else target["errors"])[f"reconstruction/{seed}"] = str(event["error"])
                except Exception as error:
                    (row["errors"] if method == "zero" else target["errors"])[f"reconstruction/{seed}"] = f"{type(error).__name__}: {error}"
                persist()
        except Exception as error:
            summary["reconstruction_error"] = f"{type(error).__name__}: {error}"
        for identifier, method, bit, seed in expected - seen:
            row = indexed[identifier]
            errors = row["errors"] if method == "zero" else row["trials"][method][str(bit)]["errors"]
            errors[f"reconstruction/{seed}"] = "requested reconstruction was not returned"
        summary["stages"]["reconstruct"] = "complete" if not summary.get("reconstruction_error") else "failed"
    except Exception as error:
        summary["execution_error"] = f"{type(error).__name__}: {error}"
        for stage, status in summary["stages"].items():
            if status == "started":
                summary["stages"][stage] = "failed"
    finally:
        summary["stages"]["cleanup"] = "started"
        if provider is not None:
            try:
                provider.close()
                summary["stages"]["cleanup"] = "complete"
            except Exception as error:
                summary["stages"]["cleanup"] = "failed"
                summary["cleanup_error"] = f"{type(error).__name__}: {error}"
        else:
            summary["stages"]["cleanup"] = "no_provider"
        _finalize_trials(summary, config)
        for row in rows:
            all_collected = all(row["trials"][method][str(bit)]["collection_complete"] for method in METHODS for bit in BITS)
            zero_collected = all(item and item.get("status") == "OBSERVED"
                                 for item in [row["zero"]["clean"], *row["zero"]["reconstruction"].values()])
            row["collection_complete"] = bool(all_collected and zero_collected and not row["errors"])
            row["status"] = "collected" if row["collection_complete"] else "incomplete"
        persist()
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume-from", type=Path,
                        help="read an original pre-write batch checkpoint; requires a new output for execution")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.config is None and arguments.resume_from is None:
            parser.error("provide --config or --resume-from")
        config_path = arguments.config if arguments.config is not None else arguments.resume_from / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not arguments.execute:
            checkpoint = None
            if arguments.resume_from is not None:
                from cegwm.runtime.channel_precompensation_resume import load_resume_checkpoint
                checkpoint = load_resume_checkpoint(arguments.resume_from, config=config)
            plan = describe_plan(config, resume_checkpoint=checkpoint)
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        if arguments.output is None:
            parser.error("--execute requires a new output directory")
        result = execute_measurement(config, arguments.output, resume_from=arguments.resume_from)
        print(json.dumps(result, indent=2, allow_nan=False))
        # Numerical infeasibility and wrong decoded bits are retained outcomes;
        # the exit code reports acquisition, not a favorable method result.
        return int(result.get("execution_error") is not None or result.get("cleanup_error") is not None
                   or any(source["status"] != "collected" for source in result["sources"]))
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
