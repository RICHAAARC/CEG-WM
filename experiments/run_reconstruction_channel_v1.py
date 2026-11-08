"""Plan or explicitly execute the single-direction reconstruction measurement.

No model is loaded by the default command. This entry point does not fit a
reader, encode bits, calibrate a detector, or adjudicate a scientific result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from cegwm.reconstruction_channel import planned_call_budget


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be explicitly specified as a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a concrete measurement configuration without touching models."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    for field in ("generation", "probe", "reconstruction"):
        if not isinstance(config.get(field), dict):
            raise ValueError(f"{field} must be explicitly specified")
    generation, probe, reconstruction = (config[k] for k in ("generation", "probe", "reconstruction"))
    for label, settings in (("generation", generation), ("reconstruction", reconstruction)):
        for field in ("model_id", "revision", "device"):
            if not isinstance(settings.get(field), str) or not settings[field].strip():
                raise ValueError(f"{label}.{field} must be explicit nonempty text")
        if settings.get("dtype") not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"{label}.dtype must be float16, bfloat16, or float32")
        if settings.get("memory_mode", "direct") not in {"direct", "model_cpu_offload"}:
            raise ValueError(f"{label}.memory_mode must be direct or model_cpu_offload")
        if settings.get("memory_mode") == "model_cpu_offload" and settings["device"].split(":")[0] != "cuda":
            raise ValueError(f"{label}.model_cpu_offload requires a CUDA device")
        _integer(settings.get("num_inference_steps"), f"{label}.num_inference_steps", 1)
        _number(settings.get("guidance_scale"), f"{label}.guidance_scale")
    for field in ("height", "width"):
        size = _integer(generation.get(field), f"generation.{field}", 16)
        if size % 16:
            raise ValueError(f"generation.{field} must be divisible by 16 for this SD3 adapter")
    step = _integer(generation.get("injection_step"), "generation.injection_step")
    if step >= generation["num_inference_steps"] - 1:
        raise ValueError("injection_step must leave at least one actual denoising step")
    _integer(probe.get("seed"), "probe.seed")
    _number(probe.get("relative_l2"), "probe.relative_l2", positive=True)
    if config.get("feature") != "full_vae_mode":
        raise ValueError("the adopted stage-A reader is full_vae_mode")
    strength = _number(reconstruction.get("strength"), "reconstruction.strength", positive=True)
    if strength > 1 or int(strength * reconstruction["num_inference_steps"]) < 1:
        raise ValueError("reconstruction strength must produce at least one img2img step and be <= 1")
    if not isinstance(reconstruction.get("prompt"), str):
        raise ValueError("reconstruction.prompt must be explicit (empty text is allowed)")
    seeds = reconstruction.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("reconstruction.seeds must be a nonempty explicit list")
    for seed in seeds:
        _integer(seed, "reconstruction seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("reconstruction seeds must be unique")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty explicit roster")
    identifiers = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        identifier = source.get("source_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
            raise ValueError("source_id must be a safe, nonempty directory name")
        identifiers.append(identifier)
        if not isinstance(source.get("prompt"), str) or not source["prompt"].strip():
            raise ValueError("each source must have an explicit nonempty prompt")
        _integer(source.get("seed"), "source.seed")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("source_id values must be unique")
    return config


def describe_plan(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    return {
        "mode": "plan_only",
        "method": "one_fixed_hf_family_probe_full_vae_mode",
        "budget": planned_call_budget(len(config["sources"]), len(config["reconstruction"]["seeds"])),
        "source_ids": [row["source_id"] for row in config["sources"]],
        "generation": config["generation"],
        "probe": config["probe"],
        "reconstruction": config["reconstruction"],
        "claim": "finite-amplitude rank-one channel measurement only; no scientific PASS",
    }


def make_fixed_probe(shape: tuple[int, int, int, int], seed: int):
    """A shared HF-family probe; no host/source inputs or old method assets."""
    import torch
    from cegwm.shared.prg import prg_rademacher

    if len(shape) != 4 or shape[0] != 1 or min(shape) < 1:
        raise ValueError("probe requires a positive single-image NCHW shape")
    _integer(seed, "probe.seed")
    _, channels, height, width = shape
    radius = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]) / math.hypot(0.5, 0.5)
    mask = (radius >= 0.58) & (radius <= 1.0)
    key = f"reconstruction-channel-v1:seed={seed}"
    channel_arrays = []
    for channel in range(channels):
        domain = f"reconstruction-channel-v1/hf-probe/{channels}/{height}/{width}/{channel}"
        spectrum = prg_rademacher(key, domain, mask.shape, dtype=np.float64) * mask
        channel_arrays.append(np.fft.irfft2(spectrum, s=(height, width), norm="ortho"))
    array = np.stack(channel_arrays)[None]
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("the fixed HF shell is empty for this shape")
    return torch.from_numpy(array / norm)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pixel_distortion(reference, image) -> dict[str, Any]:
    first, second = np.asarray(reference, dtype=np.float64), np.asarray(image, dtype=np.float64)
    if first.shape != second.shape:
        return {"status": "shape_mismatch", "reference_shape": list(first.shape), "image_shape": list(second.shape)}
    delta = (second - first) / 255.0
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {"status": "measured", "rmse_0_1": rmse, "max_abs_0_1": float(np.abs(delta).max()),
            "psnr_db": None if rmse == 0 else -20 * math.log10(rmse), "identical_pixels": rmse == 0}


def _release_unused_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pipeline(config: dict[str, Any], name: str):
    import torch
    from diffusers import StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline

    settings = config[name]
    pipeline_class = StableDiffusion3Pipeline if name == "generation" else StableDiffusionXLImg2ImgPipeline
    pipeline = pipeline_class.from_pretrained(settings["model_id"], revision=settings["revision"],
                                               torch_dtype=getattr(torch, settings["dtype"]))
    if settings.get("memory_mode", "direct") == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=settings["device"])
    else:
        pipeline = pipeline.to(settings["device"])
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


class _StagedPipelines:
    """Own one large pipeline at a time and retain the identical SD3 VAE reader."""

    def __init__(self, config):
        self.config = config
        self.generation = _load_pipeline(config, "generation")
        self.vae = self.generation.vae
        self.image_processor = self.generation.image_processor
        self.reconstruction = None

    def release_generation(self):
        pipeline, self.generation = self.generation, None
        if pipeline is not None:
            # Diffusers recursively removes Accelerate hooks, including the VAE
            # hook whose previous-module chain otherwise retains SD3 components.
            pipeline.remove_all_hooks()
            del pipeline
        _release_unused_model_memory()

    def load_reconstruction(self):
        if self.generation is not None:
            raise RuntimeError("generation pipeline must be released before loading reconstruction")
        self.vae.to(self.config["generation"]["device"])
        self.reconstruction = _load_pipeline(self.config, "reconstruction")
        return self.reconstruction

    def close(self):
        self.release_generation()
        pipeline, self.reconstruction = self.reconstruction, None
        if pipeline is not None:
            pipeline.remove_all_hooks()
            del pipeline
        self.vae.to("cpu")
        _release_unused_model_memory()


def load_pipelines(config: dict[str, Any]):
    """Load only SD3 now; SDXL is deferred until the generation stage is freed."""
    return _StagedPipelines(config)


def _save_arms(directory: Path, arms: dict[str, Any]) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    for name, arm in arms.items():
        if getattr(arm, "image", None) is not None:
            arm.image.save(directory / f"{name}.png")
        evidence = {field: getattr(arm, field) for field in
                    ("feature", "before_state", "after_latents", "actual_delta", "injection_metrics",
                     "remaining_steps", "status", "error") if hasattr(arm, field)}
        torch.save(evidence, directory / f"{name}.pt")


class _CountedPipeline:
    def __init__(self, pipeline, counts, name):
        self.pipeline, self.counts, self.name = pipeline, counts, name

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def __call__(self, *args, **kwargs):
        self.counts[f"{self.name}_attempted"] += 1
        result = self.pipeline(*args, **kwargs)
        self.counts[f"{self.name}_returned"] += 1
        return result


class _CountedProcessor:
    def __init__(self, processor, counts, *, observation_vae=None, device=None):
        self.processor, self.counts = processor, counts
        self.observation_vae, self.device = observation_vae, device

    def preprocess(self, *args, **kwargs):
        self.counts["feature_reads_attempted"] += 1
        if self.observation_vae is not None:
            # The RGB observer selects the input device from VAE parameters.
            # Accelerate's encode decorator moves the VAE, but not encode's
            # arguments; move this same VAE before the observer reads its device.
            self.observation_vae.to(self.device)
        return self.processor.preprocess(*args, **kwargs)


def _arm_failures(error, prefix: str, arms: dict[str, Any]) -> dict[str, str]:
    message = f"{type(error).__name__}: {error}"
    arm = getattr(error, "arm", None)
    if arm in {"zero", "plus", "minus"}:
        return {f"{prefix}/{arm}": message}
    if not arms:
        return {}
    return {f"{prefix}/{name}": message for name in ("zero", "plus", "minus")
            if name not in arms or getattr(arms[name], "feature", None) is None}


def execute_measurement(config: dict[str, Any], output: Path, *, pipelines=None) -> dict[str, Any]:
    """Generate the whole batch, free SD3, then reconstruct; never retry a batch.

    ``pipelines=(generation, reconstruction)`` remains a caller-owned fake/test
    injection. The real loader owns and releases its models between stages.
    """
    import torch
    from cegwm.reconstruction_channel import measure_probe_geometry, measure_source
    from cegwm.runtime.reconstruction_channel_sd35 import (
        GenerationSettings, ReconstructionSettings, run_replay_triplet, reconstruct_triplet,
    )

    plan = describe_plan(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    rows = [{"source_id": source["source_id"], "status": "not_attempted"} for source in config["sources"]]
    counts = {"generation_attempted": 0, "generation_returned": 0,
              "reconstruction_attempted": 0, "reconstruction_returned": 0,
              "feature_reads_attempted": 0, "feature_reads_saved": 0}
    stages = {"generation_setup": "started", "generation": "not_started",
              "generation_release": "not_started", "reconstruction_setup": "not_started",
              "reconstruction": "not_started", "cleanup": "not_started"}
    summary = {"mode": "measurement", "planned_budget": plan["budget"], "actual_calls": counts, "sources": rows,
               "stages": stages,
               "claim": "measurement only; no watermark or scientific PASS"}
    _write_json(output / "summary.json", summary)
    owner = None
    generation_pipeline = reconstruction_pipeline = None
    try:
        generation, reconstruction = config["generation"], config["reconstruction"]
        if pipelines is None:
            owner = load_pipelines(config)
            generation_pipeline = owner.generation
        else:
            generation_pipeline, reconstruction_pipeline = pipelines
        vae = generation_pipeline.vae
        processor = _CountedProcessor(generation_pipeline.image_processor, counts,
            observation_vae=vae if owner is not None else None,
            device=generation["device"])
        shape = (1, int(generation_pipeline.transformer.config.in_channels),
                 generation["height"] // generation_pipeline.vae_scale_factor,
                 generation["width"] // generation_pipeline.vae_scale_factor)
        direction = make_fixed_probe(shape, config["probe"]["seed"])
        torch.save(direction, output / "probe.pt")
        reconstruction_settings = ReconstructionSettings(**{name: reconstruction[name] for name in
            ("strength", "guidance_scale", "num_inference_steps", "prompt", "device")})
        generation_pipeline = _CountedPipeline(generation_pipeline, counts, "generation")
    except Exception as error:
        stages["generation_setup"] = "failed"
        summary["setup_error"] = f"{type(error).__name__}: {error}"
        generation_pipeline = reconstruction_pipeline = None
        if owner is not None:
            try:
                owner.close()
                stages["cleanup"] = "complete"
            except Exception as cleanup_error:
                stages["cleanup"] = "failed"
                summary["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        _write_json(output / "summary.json", summary)
        return summary

    contexts = []

    def save_context(context):
        row = context["row"]
        clean, reconstructed = context["clean"], context["reconstructed"]
        operation_errors = context["errors"]
        try:
            measurement = measure_source(row["source_id"], clean_features=clean,
                reconstruction_features_by_seed=reconstructed, expected_reconstruction_seeds=reconstruction["seeds"],
                probe_geometry=context["geometry"], failures=context["failures"], include_vectors=False)
            status = "failed" if operation_errors else ("measured" if measurement["status"] == "MEASURED" else "incomplete")
            row.update(status=status, errors=dict(operation_errors), measurement=measurement)
        except Exception as error:
            operation_errors["analysis"] = f"{type(error).__name__}: {error}"
            row.update(status="failed", errors=dict(operation_errors))
        expected_features = [clean.get(name) for name in ("zero", "plus", "minus")]
        expected_features += [reconstructed.get(seed, {}).get(name) for seed in reconstruction["seeds"]
                              for name in ("zero", "plus", "minus")]
        shapes = [tuple(feature.shape) for feature in expected_features if feature is not None]
        # Collection is independent of response size or scientific measurement
        # validity (including actual-dtype perturbations that quantize to zero).
        row["collection_complete"] = bool(not operation_errors and row.get("matched_state")
            and context["all_images_collected"] and len(shapes) == len(expected_features)
            and len(set(shapes)) == 1)
        counts["feature_reads_saved"] = sum(feature is not None for item in contexts
            for feature in list(item["clean"].values())
            + [value for features in item["reconstructed"].values() for value in features.values()])
        _write_json(context["directory"] / "measurement.json", row)
        _write_json(output / "summary.json", summary)

    stages["generation_setup"] = "complete"
    stages["generation"] = "started"
    for index, source in enumerate(config["sources"]):
        directory = output / source["source_id"]
        directory.mkdir()
        row = {"source_id": source["source_id"], "status": "started", "stage": "generation",
               "collection_complete": False}
        rows[index] = row
        context = {"row": row, "directory": directory, "clean": {}, "reconstructed": {},
                   "geometry": None, "failures": {}, "errors": {}, "triplet": None,
                   "all_images_collected": False}
        contexts.append(context)
        _write_json(output / "summary.json", summary)
        try:
            settings = GenerationSettings(prompt=source["prompt"], seed=source["seed"],
                relative_l2=config["probe"]["relative_l2"], **{name: generation[name] for name in
                ("injection_step", "guidance_scale", "height", "width", "num_inference_steps", "device")})
            triplet = run_replay_triplet(generation_pipeline, direction, settings,
                                         observation_vae=vae, observation_processor=processor)
            _save_arms(directory / "clean", triplet.arms)
            context["clean"] = {name: arm.feature for name, arm in triplet.arms.items()}
            if not triplet.matched_state:
                raise ValueError("source replay states did not match")
            zero = triplet.arms["zero"]
            base = zero.before_state["latents"]
            context["geometry"] = measure_probe_geometry(base, triplet.arms["plus"].after_latents,
                triplet.arms["minus"].after_latents, direction,
                nominal_amplitude=config["probe"]["relative_l2"] * float(torch.linalg.vector_norm(base.double())))
            row["matched_state"] = triplet.matched_state
            row["clean_pixel_distortion"] = {name: pixel_distortion(zero.image, triplet.arms[name].image)
                                              for name in ("plus", "minus")}
            if any(getattr(triplet.arms.get(name), "image", None) is None
                   or getattr(triplet.arms.get(name), "feature", None) is None for name in ("zero", "plus", "minus")):
                raise ValueError("generation returned an incomplete RGB/feature triplet")
            context["triplet"] = triplet
            row["stage"] = "generation_complete"
        except Exception as error:
            arms = getattr(error, "arms", {})
            _save_arms(directory / "clean", arms)
            context["clean"].update({name: getattr(arm, "feature", None) for name, arm in arms.items()})
            context["errors"]["generation"] = f"{type(error).__name__}: {error}"
            context["failures"].update(_arm_failures(error, "clean", arms))
            row["stage"] = "generation_failed"
        save_context(context)

    stages["generation"] = "complete" if all(item["triplet"] is not None for item in contexts) else "completed_with_errors"
    complete = [item for item in contexts if item["triplet"] is not None]
    # The wrapper must release its reference before the owner drops SD3. Only
    # CPU triplets, ordinary images, and the shared VAE survive this boundary.
    generation_pipeline = None
    try:
        stages["generation_release"] = "started"
        _write_json(output / "summary.json", summary)
        if owner is not None:
            owner.release_generation()
        stages["generation_release"] = "complete" if owner is not None else "caller_owned"
        if complete:
            stages["reconstruction_setup"] = "started"
            _write_json(output / "summary.json", summary)
            if owner is not None:
                reconstruction_pipeline = owner.load_reconstruction()
            reconstruction_pipeline = _CountedPipeline(reconstruction_pipeline, counts, "reconstruction")
            stages["reconstruction_setup"] = "complete"
            stages["reconstruction"] = "started"
            _write_json(output / "summary.json", summary)
            for context in complete:
                row, directory, triplet = context["row"], context["directory"], context["triplet"]
                row["stage"] = "reconstruction"
                all_images = True
                for seed in reconstruction["seeds"]:
                    try:
                        result = reconstruct_triplet(reconstruction_pipeline, triplet, reconstruction_settings, seed,
                                                     observation_vae=vae, observation_processor=processor)
                        _save_arms(directory / f"reconstruction_{seed}", result.arms)
                        context["reconstructed"][seed] = {name: arm.feature for name, arm in result.arms.items()}
                        all_images = all_images and all(getattr(result.arms.get(name), "image", None) is not None
                                                       for name in ("zero", "plus", "minus"))
                        row.setdefault("reconstruction_pixel_distortion", {})[str(seed)] = {
                            name: pixel_distortion(triplet.arms[name].image, arm.image) for name, arm in result.arms.items()}
                    except Exception as error:
                        arms = getattr(error, "arms", {})
                        _save_arms(directory / f"reconstruction_{seed}", arms)
                        context["reconstructed"].setdefault(seed, {}).update(
                            {name: getattr(arm, "feature", None) for name, arm in arms.items()})
                        context["errors"][f"reconstruction/{seed}"] = f"{type(error).__name__}: {error}"
                        context["failures"].update(_arm_failures(error, f"reconstruction/{seed}", arms))
                        all_images = False
                    save_context(context)
                context["all_images_collected"] = all_images
                row["stage"] = "complete"
                save_context(context)
            stages["reconstruction"] = "complete" if all(not item["errors"] for item in complete) else "completed_with_errors"
        else:
            stages["reconstruction_setup"] = stages["reconstruction"] = "skipped_no_complete_sources"
    except Exception as error:
        stage = "generation_release" if stages["generation_release"] == "started" else "reconstruction_setup"
        stages[stage] = "failed"
        stages["reconstruction"] = "not_started"
        summary["setup_error"] = f"{stage}: {type(error).__name__}: {error}"
        for context in complete:
            context["errors"][stage] = f"{type(error).__name__}: {error}"
            context["row"]["stage"] = f"{stage}_failed"
            save_context(context)
    finally:
        reconstruction_pipeline = None
        if owner is not None:
            try:
                owner.close()
                stages["cleanup"] = "complete"
            except Exception as error:
                stages["cleanup"] = "failed"
                summary["cleanup_error"] = f"{type(error).__name__}: {error}"
        else:
            stages["cleanup"] = "caller_owned"
        _write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="explicitly load models and execute the listed measurement")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        plan = describe_plan(config)
        if not arguments.execute:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        if arguments.output is None:
            parser.error("--execute requires --output pointing to a new directory")
        summary = execute_measurement(config, arguments.output)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 1 if summary.get("setup_error") or any(row["status"] != "measured" for row in summary["sources"]) else 0
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
