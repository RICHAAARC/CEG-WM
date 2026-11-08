"""Three fixed source-suffix traces; planning is the default and loads no model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ARMS = (("NG1", False), ("NG2", False), ("G", True))
SOURCE_ID = "source05_smooth_stilllife"
CLAIM = "Forward-path diagnosis only; neither unique causal attribution nor watermark validation."
FORBIDDEN_COUNTS = ("final_generation_attempted", "final_generation_returned",
                    "reconstruction_attempted", "reconstruction_returned",
                    "public_feature_reads_attempted", "public_feature_reads_returned",
                    "rgb_vjps_attempted", "rgb_vjps_returned",
                    "latent_vjps_attempted", "latent_vjps_returned",
                    "autograd_calls_attempted", "autograd_calls_returned")


def load_checkpoint(path):
    from cegwm.runtime.channel_precompensation_resume import load_resume_checkpoint
    return load_resume_checkpoint(path)


def describe_plan(checkpoint):
    sources = checkpoint.config.get("sources", [])
    if len(sources) != 4 or sources[0].get("source_id") != SOURCE_ID:
        raise ValueError("diagnosis requires the original four-source roster with source05_smooth_stilllife first")
    capture = checkpoint.baselines.get(SOURCE_ID)
    if not capture or tuple(capture.get("remaining_steps", ())) != (19,):
        raise ValueError("diagnosis requires the saved first source with remaining step 19")
    return {"mode": "plan_only", "source_id": SOURCE_ID,
            "source_selection": "first item of the frozen original roster; never selected by results",
            "arms": [{"arm": name, "grad_enabled": enabled} for name, enabled in ARMS],
            "planned_budget": {"source_pipeline_loads": 1, "suffix_forwards": 3,
                "suffix_denoising_steps": 3, "suffix_decodes": 3,
                "full_generation": 0, "reconstruction": 0, "public_reads": 0, "vjps": 0},
            "prior_actual_calls": dict(checkpoint.prior_actual_calls), "claim": CLAIM}


def load_pipeline(config):
    # Reuse the existing model identity/loading policy. Preparation below removes
    # inference hooks before all three traces; no full generation is requested.
    from experiments.run_reconstruction_channel_v1 import _load_pipeline
    return _load_pipeline(config, "generation")


def prepare_pipeline(pipeline, config):
    pipeline.remove_all_hooks()
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.to("cpu")
            module.eval()
            module.requires_grad_(False)
    for module in (pipeline.transformer, pipeline.vae):
        module.to(config["generation"]["device"])
        module.eval()
        module.requires_grad_(False)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items() if not isinstance(item, Image.Image)}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _save_trace(directory, trace):
    directory.mkdir()
    torch.save(_cpu(trace), directory / "trace.pt")
    image = trace.get("image")
    if image is not None:
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("trace image must be ordinary RGB")
        image.save(directory / "image.png")
    return {"trace_file": str(directory.name + "/trace.pt"),
            "image_file": str(directory.name + "/image.png") if image is not None else None}


def _layers(trace):
    layers = {"before_latents": trace.get("before_latents")}
    steps = trace.get("steps", [])
    step = next((row for row in steps if row.get("index") == 19), {})
    for field in ("timestep", "guided_prediction", "scheduler_output"):
        layers["step19/" + field] = step.get(field)
    for field in ("pre_vae_latents", "decoded", "postprocessed_rgb", "uint8_rgb"):
        layers[field] = trace.get(field)
    return layers


def compare_tensors(first, second):
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        return {"status": "MISSING", "exact_equal": None}
    finite = bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
    row = {"status": "OBSERVED" if finite else "NONFINITE", "finite": finite,
           "left_shape": list(first.shape), "right_shape": list(second.shape),
           "left_dtype": str(first.dtype), "right_dtype": str(second.dtype),
           "shape_equal": first.shape == second.shape, "dtype_equal": first.dtype == second.dtype}
    row["exact_equal"] = bool(finite and row["shape_equal"] and row["dtype_equal"] and torch.equal(first, second))
    if finite and row["shape_equal"]:
        delta = first.double() - second.double()
        row.update(max_abs_difference=float(delta.abs().max()) if delta.numel() else 0.0,
                   mean_abs_difference=float(delta.abs().mean()) if delta.numel() else 0.0,
                   changed_components=int(torch.count_nonzero(delta)), total_components=delta.numel())
    return row


def compare_traces(first, second):
    left, right = _layers(first), _layers(second)
    layers = {name: compare_tensors(value, right[name]) for name, value in left.items()}
    complete = all(row["status"] == "OBSERVED" for row in layers.values())
    return {"status": "OBSERVED" if complete else "INCOMPLETE", "layers": layers,
            "all_layers_exact": complete and all(row["exact_equal"] for row in layers.values()),
            "first_difference": next((name for name, row in layers.items() if row["exact_equal"] is False), None),
            "first_unavailable": next((name for name, row in layers.items() if row["status"] != "OBSERVED"), None)}


def classify(comparisons, arms):
    repeated, gradient = comparisons["NG1_vs_NG2"], comparisons["NG1_vs_G"]
    if any(comparison["layers"][name]["exact_equal"] is False
           for comparison in (repeated, gradient)
           for name in ("before_latents", "step19/timestep")):
        kind = "UNKNOWN_CONTROL_MISMATCH"
    elif (any(row["status"] != "OBSERVED" for row in arms.values())
            or repeated["status"] != "OBSERVED" or gradient["status"] != "OBSERVED"):
        kind = "UNKNOWN_INCOMPLETE"
    elif not repeated["all_layers_exact"]:
        kind = "REPEATED_NOGRAD_DIFFERENCE"
    elif not gradient["all_layers_exact"]:
        ng_matches = comparisons["rgb_vs_reference"]["NG1"]["exact_equal"]
        kind = "GRAD_MODE_ASSOCIATED_DIFFERENCE" if ng_matches else "MIXED_REFERENCE_AND_GRAD_MODE_DIFFERENCES"
    elif comparisons["rgb_vs_reference"]["NG1"]["exact_equal"]:
        kind = "ALL_OBSERVED_MATCH_REFERENCE"
    else:
        kind = "SHARED_SUFFIX_REFERENCE_MISMATCH"
    return {"kind": kind, "claim": CLAIM,
            "limits": "One repeated no-grad pair and one grad-mode forward cannot identify a unique kernel or exclude all nondeterminism."}


def execute_diagnostic(resume_from, output, *, checkpoint_loader=None, pipeline_loader=None, trace_fn=None):
    checkpoint = (checkpoint_loader or load_checkpoint)(resume_from)
    plan = describe_plan(checkpoint)
    output, original = Path(output).resolve(), Path(resume_from).resolve()
    if output == original or output.is_relative_to(original):
        raise ValueError("diagnostic output must be a new directory outside the input checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", checkpoint.config)
    _write_json(output / "prior_summary.json", checkpoint.prior_summary)
    baseline = checkpoint.baselines[SOURCE_ID]
    # The checkpoint loader already validates CPU tensors. Preserve the safe
    # OrderedDict metadata and its attribute state used by scheduler.config.
    torch.save({key: baseline[key] for key in ("before_state", "scheduler_state", "remaining_steps")},
               output / "baseline.pt")
    baseline["image"].save(output / "reference_zero.png")
    actual = {key: 0 for key in FORBIDDEN_COUNTS}
    actual.update(source_pipeline_load_attempted=0, source_pipeline_load_returned=0,
                  suffix_arms_attempted=0, suffix_arms_returned=0,
                  sd35_suffix_forward_calls=0, sd35_suffix_forward_returned=0,
                  sd35_suffix_denoising_steps=0, sd35_suffix_decode_calls=0, sd35_suffix_decode_returned=0)
    summary = {"mode": "suffix_diagnostic", "source_id": SOURCE_ID, "resume_from": str(original),
               "planned_budget": plan["planned_budget"], "prior_actual_calls": checkpoint.prior_actual_calls,
               "actual_calls": actual, "status": "RUNNING", "claim": CLAIM,
               "stages": {"load": "not_started", "traces": "not_started", "cleanup": "not_started"},
               "arms": {name: {"status": "NOT_STARTED", "grad_enabled": enabled, "error": None}
                        for name, enabled in ARMS}}

    def persist():
        prior = summary["prior_actual_calls"]
        summary["cumulative_actual_calls"] = {name: prior.get(name, 0) + actual.get(name, 0)
                                               for name in prior.keys() | actual.keys()}
        _write_json(output / "summary.json", summary)

    persist()
    pipeline, traces = None, {}
    try:
        from cegwm.runtime.channel_precompensation_sd35 import SourceCapture
        if trace_fn is None:
            from cegwm.runtime.channel_precompensation_sd35 import trace_source_suffix
            trace_fn = trace_source_suffix
        capture = SourceCapture(baseline["image"], baseline["before_state"],
                                baseline["scheduler_state"], tuple(baseline["remaining_steps"]))
        summary["stages"]["load"] = "started"
        actual["source_pipeline_load_attempted"] += 1
        persist()
        pipeline = (pipeline_loader or load_pipeline)(checkpoint.config)
        actual["source_pipeline_load_returned"] += 1
        prepare_pipeline(pipeline, checkpoint.config)
        summary["stages"]["load"] = "complete"
        summary["stages"]["traces"] = "started"
        persist()
        for name, enabled in ARMS:
            row = summary["arms"][name]
            row["status"] = "STARTED"
            actual["suffix_arms_attempted"] += 1
            persist()  # Reserve the fixed arm before physical work, including interruption.
            try:
                trace = trace_fn(pipeline, capture, grad_enabled=enabled, counters=actual)
                actual["suffix_arms_returned"] += 1
                traces[name] = trace
                row.update(_save_trace(output / name, trace), status="OBSERVED")
            except Exception as error:
                partial = getattr(error, "partial", {})
                traces[name] = partial
                row.update(status="FAILED", error=type(error).__name__ + ": " + str(error),
                           error_stage=getattr(error, "stage", None))
                if not (output / name).exists():
                    row.update(_save_trace(output / name, partial))
            persist()
        summary["stages"]["traces"] = "complete"
        reference = torch.from_numpy(np.array(baseline["image"], dtype=np.uint8, copy=True))
        comparisons = {"NG1_vs_NG2": compare_traces(traces.get("NG1", {}), traces.get("NG2", {})),
                       "NG1_vs_G": compare_traces(traces.get("NG1", {}), traces.get("G", {})),
                       "rgb_vs_reference": {name: compare_tensors(traces.get(name, {}).get("uint8_rgb"), reference)
                                            for name, _ in ARMS}}
        summary["comparisons"] = comparisons
        summary["classification"] = classify(comparisons, summary["arms"])
        complete = (all(row["status"] == "OBSERVED" for row in summary["arms"].values())
                    and comparisons["NG1_vs_NG2"]["status"] == "OBSERVED"
                    and comparisons["NG1_vs_G"]["status"] == "OBSERVED")
        summary["status"] = "OBSERVED" if complete else "INCOMPLETE"
    except Exception as error:
        summary["execution_error"] = type(error).__name__ + ": " + str(error)
        summary["status"] = "INCOMPLETE"
        for stage in ("load", "traces"):
            if summary["stages"][stage] == "started":
                summary["stages"][stage] = "failed"
    finally:
        summary["stages"]["cleanup"] = "started"
        persist()
        try:
            if pipeline is not None:
                pipeline.remove_all_hooks()
            pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary["stages"]["cleanup"] = "complete"
        except Exception as error:
            summary["stages"]["cleanup"] = "failed"
            summary["cleanup_error"] = type(error).__name__ + ": " + str(error)
        persist()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            print(json.dumps(describe_plan(load_checkpoint(args.resume_from)), indent=2, allow_nan=False))
            return 0
        if args.output is None:
            parser.error("--execute requires a new output directory")
        summary = execute_diagnostic(args.resume_from, args.output)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return int(summary["status"] != "OBSERVED" or summary.get("cleanup_error") is not None)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
