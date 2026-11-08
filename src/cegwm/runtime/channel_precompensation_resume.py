"""Read saved precompensation observations without rerunning their models.

This loader restores completed zero-arm evidence and source capture state. It
does not turn prior failed trials into successes, change method parameters, or
execute pickled arbitrary Python objects. Missing writer sensitivities remain
missing. The runner always writes continuation evidence to a new directory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from PIL import Image

from cegwm.channel_precompensation import ReaderSpec, make_reader


RGB_GRADIENT_COMPATIBILITY_VERSION = "rgb-pil-layout-v2"
_METHOD_FIELDS = ("generation", "reconstruction", "reader", "write", "quality", "sources")


@dataclass
class ResumeCheckpoint:
    batch_directory: Path
    config: dict[str, Any]
    reader_vector: torch.Tensor
    reader_phase: float
    baselines: dict[str, dict[str, Any]]
    zero_reconstructions: dict[tuple[str, int], dict[str, Any]]
    prior_summary: dict[str, Any]
    prior_actual_calls: dict[str, int]
    reused_counts: dict[str, int]
    rgb_gradients: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path.name} must contain an object")
    return value


def validate_same_method(saved: dict[str, Any], requested: dict[str, Any]) -> None:
    """Compare method choices, allowing explanatory metadata/key order to differ."""
    differences = [name for name in _METHOD_FIELDS if saved.get(name) != requested.get(name)]
    if saved.get("feature", "full_vae_mode") != requested.get("feature", "full_vae_mode"):
        differences.append("feature")
    if differences:
        raise ValueError("resume configuration changes fixed method fields: " + ", ".join(differences))


def _legacy_frozen_dict_metadata(*args, **kwargs):
    """Restore legacy config data as a safe built-in mapping, without model code.

    Torch 2.14 permits SETITEMS only on exact dict/OrderedDict/Counter types,
    including when the original mapping subclass was explicitly allowlisted.
    OrderedDict also accepts pickle BUILD's attribute state, preserving the
    scheduler's ``config.stochastic_sampling`` and other attribute reads.
    """
    return OrderedDict(*args, **kwargs)


# A callable registered under the one historical pickle name works with both
# old safe_globals APIs and newer (callable, name) APIs. No arbitrary checkpoint
# global is resolved, and the callable can only construct OrderedDict metadata.
_legacy_frozen_dict_metadata.__module__ = "diffusers.configuration_utils"
_legacy_frozen_dict_metadata.__name__ = "FrozenDict"
_legacy_frozen_dict_metadata.__qualname__ = "FrozenDict"


def _load_tensor_record(path: Path) -> dict[str, Any]:
    # No unrestricted pickle fallback, model construction, or tensor conversion.
    with torch.serialization.safe_globals([_legacy_frozen_dict_metadata]):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path.name} must contain a dictionary")
    return value


def _finite_tensor(value, name: str, *, floating=True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty CPU tensor")
    if floating and not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and real")
    return value


def _finite_metadata(value, name="state"):
    if isinstance(value, torch.Tensor):
        _finite_tensor(value, name, floating=False)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_metadata(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_metadata(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as value:
        if value.mode != "RGB" or value.size != size:
            raise ValueError(f"{path.name} must be ordinary RGB with configured dimensions")
        value.load()
        return value.copy()


def _observation(directory: Path, stem: str, dimension: int, size: tuple[int, int]) -> dict[str, Any]:
    record = _load_tensor_record(directory / f"{stem}.pt")
    if record.get("status") != "OBSERVED" or record.get("error"):
        raise ValueError(f"saved zero observation {stem} was not successfully observed")
    feature = _finite_tensor(record.get("feature"), stem)
    if feature.ndim != 1 or feature.numel() != dimension:
        raise ValueError(f"saved zero observation {stem} has a different feature dimension")
    return {"image": _image(directory / f"{stem}.png", size), "feature": feature,
            "status": "OBSERVED", "reused": True}


def _validate_capture(record, source_id, generation):
    if record.get("source_id") != source_id or record.get("status") != "OBSERVED":
        raise ValueError("saved source capture identity/status differs")
    state, scheduler = record.get("before_state"), record.get("scheduler_state")
    if not isinstance(state, dict) or not isinstance(scheduler, dict):
        raise ValueError("saved source capture lacks before_state or scheduler_state")
    for name in ("latents", "prompt_embeds", "pooled_prompt_embeds", "timesteps", "sigmas"):
        _finite_tensor(state.get(name), f"before_state.{name}")
    generator = _finite_tensor(state.get("generator_state"), "before_state.generator_state", floating=False)
    if generator.dtype != torch.uint8 or generator.ndim != 1:
        raise ValueError("saved generator state must be a uint8 vector")
    latent = state["latents"]
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.dtype != getattr(torch, generation["dtype"]):
        raise ValueError("saved latent shape or actual dtype differs from the generation configuration")
    steps, injection = generation["num_inference_steps"], generation["injection_step"]
    if state["timesteps"].ndim != 1 or len(state["timesteps"]) != steps:
        raise ValueError("saved capture does not contain the complete timestep schedule")
    if state["sigmas"].ndim != 1 or len(state["sigmas"]) != steps + 1:
        raise ValueError("saved capture does not contain the complete sigma schedule")
    if state.get("next_step_index") != injection + 1 or scheduler.get("_step_index") != injection + 1:
        raise ValueError("saved capture is not at the configured post-injection step")
    if tuple(record.get("remaining_steps", ())) != tuple(range(injection + 1, steps)):
        raise ValueError("saved capture has different remaining denoising steps")
    if state.get("guidance_scale") != generation["guidance_scale"]:
        raise ValueError("saved capture guidance differs")
    if state.get("do_classifier_free_guidance") is not (generation["guidance_scale"] > 1):
        raise ValueError("saved classifier-free-guidance state differs")
    required = ("latent_device", "timestep", "interval_start_sigma", "latent_sigma", "next_sigma",
                "begin_index", "scheduler_class", "scheduler_config", "clip_skip", "joint_attention_kwargs")
    if any(name not in state for name in required):
        raise ValueError("saved source capture is missing reference-state metadata")
    for name in ("timesteps", "sigmas"):
        if name not in scheduler or not torch.equal(state[name], scheduler[name]):
            raise ValueError("saved scheduler and observed schedule disagree")
    _finite_metadata(state, "before_state")
    _finite_metadata(scheduler, "scheduler_state")


def _prior_counts(summary):
    stages = summary.get("stages", {})
    if not isinstance(stages, dict) or not stages or "started" in stages.values():
        raise ValueError("prior call totals are not terminal; continuation cannot silently infer them")
    if stages.get("cleanup") not in {"complete", "failed", "no_provider"}:
        raise ValueError("prior cleanup has no terminal record")
    counts = summary.get("cumulative_actual_calls", summary.get("actual_calls"))
    if not isinstance(counts, dict):
        raise ValueError("prior actual calls are missing")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("prior call counts must be nonnegative integers")
    return dict(counts)


def _cached_gradients(directory, source_id, dimension, size):
    cached = {}
    for phase in ("clean", "design"):
        path = directory / "writer" / f"rgb_gradient_{phase}.pt"
        if not path.is_file():
            continue
        event = _load_tensor_record(path)
        if event.get("compatibility_version") != RGB_GRADIENT_COMPATIBILITY_VERSION or event.get("reusable") is not True:
            continue
        if event.get("source_id") != source_id or event.get("phase") != phase:
            raise ValueError("saved RGB sensitivity identity differs")
        result = event.get("result")
        if not isinstance(result, dict):
            raise ValueError("saved RGB sensitivity result is missing")
        gradient = _finite_tensor(result.get("gradient_rgb"), "saved RGB sensitivity")
        if tuple(gradient.shape) != (1, 3, size[1], size[0]):
            raise ValueError("saved RGB sensitivity shape differs")
        feature = _finite_tensor(result.get("feature"), "saved RGB sensitivity feature")
        if feature.ndim != 1 or feature.numel() != dimension or not math.isfinite(float(result.get("score", float("nan")))):
            raise ValueError("saved RGB sensitivity readout is invalid")
        event["result"] = {**result, "image": _image(directory / "writer" / f"rgb_gradient_{phase}.png", size)}
        cached[phase] = event
    return cached


def load_resume_checkpoint(path: str | Path, config: dict[str, Any] | None = None) -> ResumeCheckpoint:
    """Load fixed zero observations and source states; no models and no writes."""
    batch = Path(path).resolve()
    saved = _read_json(batch / "config.json")
    if config is not None:
        validate_same_method(saved, config)
    settings = saved if config is None else config
    sources = settings.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("resume requires the fixed four-source roster")
    identifiers = [source.get("source_id") for source in sources]
    if any(not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier)
           for identifier in identifiers) or len(set(identifiers)) != 4:
        raise ValueError("resume source identifiers must be unique safe names")
    summary = _read_json(batch / "summary.json")
    if [row.get("source_id") for row in summary.get("sources", [])] != identifiers:
        raise ValueError("prior summary source roster differs from its fixed configuration")
    counts = _prior_counts(summary)
    attempted_generations = counts.get("final_generation_attempted", counts.get("generation_attempted", 0))
    has_marked_evidence = any(trial.get("write") is not None or trial.get("clean") is not None
        for row in summary.get("sources", []) for trials in row.get("trials", {}).values()
        for trial in trials.values())
    if attempted_generations > len(sources) or has_marked_evidence:
        raise ValueError("this continuation restores pre-write checkpoints only; resuming already-attempted marked writes is not implemented")
    reader = _load_tensor_record(batch / "reader.pt")
    if reader.get("spec") != settings["reader"]:
        raise ValueError("saved public reader specification differs")
    vector = _finite_tensor(reader.get("vector"), "saved public reader")
    if vector.ndim != 1:
        raise ValueError("saved public reader must be a vector")
    expected, phase = make_reader(vector.numel(), ReaderSpec(**settings["reader"]))
    if not torch.equal(vector.double(), expected) or reader.get("phase") != phase:
        raise ValueError("saved reader/phase does not match the frozen key specification")
    generation, reconstruction = settings["generation"], settings["reconstruction"]
    seeds = (reconstruction["design_seed"], reconstruction["holdout_seed"])
    if seeds[0] == seeds[1]:
        raise ValueError("saved design and holdout seeds must differ")
    size = (generation["width"], generation["height"])
    baselines, zero, rgb = {}, {}, {}
    for identifier in identifiers:
        directory = batch / identifier
        capture = _load_tensor_record(directory / "writer/baseline.pt")
        _validate_capture(capture, identifier, generation)
        baselines[identifier] = {**capture, **_observation(directory / "zero", "clean", vector.numel(), size)}
        for seed in seeds:
            zero[(identifier, seed)] = _observation(directory / "zero", f"reconstruction_{seed}", vector.numel(), size)
        cached = _cached_gradients(directory, identifier, vector.numel(), size)
        if cached:
            rgb[identifier] = cached
    return ResumeCheckpoint(batch, settings, vector, phase, baselines, zero, summary, counts,
        {"source_captures": 4, "zero_generation_images": 4, "zero_reconstruction_images": 8,
         "public_zero_features": 12, "reader_assets": 1,
         "rgb_sensitivities": sum(len(phases) for phases in rgb.values())}, rgb)
