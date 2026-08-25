"""One-shot, user-run SD3.5 Geometry-V1 operational preflight.

This module is deliberately an operational probe, not a detector or scientific
experiment.  It loads a supplied public runtime only when launched by the
future user-owned Colab handoff; local tests replace that loader with fakes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from cegwm.geometry.qk_relation import keyed_qk_relation
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.sd35_qk_observation import SD35QKObservation, SD35QKObservationSpec, observe_sd35_image_qk


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
INFERENCE_STEPS = 20
SCHEDULE_INDEX = 7
PUBLIC_NOISE_SEED = 0
MAX_GRID = (8, 8)
_ATTENTION_PATH = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.attn(?:\.|$)")
_PUBLIC_ERRORS = {"FileNotFoundError", "ImportError", "ModuleNotFoundError", "OSError", "RuntimeError", "TypeError", "ValueError"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    return _sha256_bytes(detached.numpy().tobytes())


def _image_digest(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ValueError("input image must be ordinary RGB")
    return _sha256_bytes(np.asarray(image, dtype=np.uint8).tobytes())


def _runtime_record(pipeline: Any) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pillow": getattr(Image, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "pipeline_class": type(pipeline).__qualname__,
        "pipeline_name_or_path": str(getattr(pipeline, "name_or_path", "unknown")),
    }


def _null_conditioning(pipeline: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    encode_prompt = getattr(pipeline, "encode_prompt", None)
    if not callable(encode_prompt):
        raise TypeError("pipeline must expose public encode_prompt for the one null construction")
    result = encode_prompt(prompt="", do_classifier_free_guidance=False)
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ValueError("null encode_prompt must return hidden and pooled tensors")
    hidden, pooled = result[0], result[1]
    if not isinstance(hidden, torch.Tensor) or not isinstance(pooled, torch.Tensor):
        raise TypeError("null encode_prompt outputs must be tensors")
    if hidden.shape[0] != 1 or pooled.shape[0] != 1:
        raise ValueError("null conditioning must be batch one")
    return hidden.detach(), pooled.detach(), {
        "hidden_shape": list(hidden.shape), "hidden_dtype": str(hidden.dtype), "hidden_sha256": _tensor_digest(hidden),
        "pooled_shape": list(pooled.shape), "pooled_dtype": str(pooled.dtype), "pooled_sha256": _tensor_digest(pooled),
    }


def _discover_layers(transformer: torch.nn.Module) -> tuple[str, str]:
    candidates: list[tuple[int, str]] = []
    for path, module in transformer.named_modules():
        match = _ATTENTION_PATH.search(path)
        if match and isinstance(getattr(module, "to_q", None), torch.nn.Module) and isinstance(getattr(module, "to_k", None), torch.nn.Module):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError("no transformer_blocks.<int>.attn module with to_q/to_k was found")
    candidates.sort()
    shallow, deep = candidates[0], candidates[-1]
    if shallow[0] == deep[0]:
        raise ValueError("layer discovery requires distinct shallow and deep block indices")
    return shallow[1], deep[1]


def _observation_record(observation: SD35QKObservation, *, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "latent_shape": list(observation.latent_shape), "schedule_index": observation.schedule_index,
        "timestep": str(observation.timestep.item()), "elapsed_seconds": elapsed_seconds,
        "layers": [
            {"path": layer.layer_path, "query_shape": list(layer.query.shape), "key_shape": list(layer.key.shape),
             "dtype": str(layer.query.dtype), "device": str(layer.query.device), "finite": bool(torch.isfinite(layer.query).all() and torch.isfinite(layer.key).all()),
             "query_sha256": _tensor_digest(layer.query), "key_sha256": _tensor_digest(layer.key),
             "source_grid": list(layer.source_grid), "heads": layer.heads, "head_dim": layer.head_dim}
            for layer in observation.layers
        ],
    }


def operational_preflight(
    images: list[Image.Image], *, hf_token: str, root_key: str | bytes,
    loader: Callable[..., Any] = load_sd35_pipeline,
) -> dict[str, Any]:
    """Run exactly the bounded operational probe and return a sanitized receipt."""
    if len(images) not in (1, 2):
        raise ValueError("operational preflight requires exactly one or two RGB images")
    if not isinstance(hf_token, str) or not hf_token:
        raise ValueError("HF_TOKEN is required")
    if not isinstance(root_key, (str, bytes)) or not root_key:
        raise ValueError("CEG_WM_ROOT_KEY is required")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_geometry_v1_operational_preflight")
    pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
    pipeline = pipeline.to("cuda")
    null_hidden, null_pooled, null_record = _null_conditioning(pipeline)
    transformer = getattr(pipeline, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError("pipeline transformer must be a torch module")
    shallow, deep = _discover_layers(transformer)
    spec = SD35QKObservationSpec(MODEL_ID, str(getattr(pipeline, "revision", "runtime-resolved")), (shallow, deep), INFERENCE_STEPS, SCHEDULE_INDEX, PUBLIC_NOISE_SEED, MAX_GRID, null_hidden, null_pooled)
    results: list[dict[str, Any]] = []
    first: SD35QKObservation | None = None
    for index, image in enumerate(images):
        started = time.monotonic()
        observation = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
        record = _observation_record(observation, elapsed_seconds=time.monotonic() - started)
        record["input_sha256"] = _image_digest(image)
        results.append(record)
        if index == 0:
            first = observation
            repeated = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
            if _tensor_digest(repeated.layers[0].query) != _tensor_digest(observation.layers[0].query) or _tensor_digest(repeated.layers[0].key) != _tensor_digest(observation.layers[0].key):
                raise ValueError("first image observation is not deterministic")
    assert first is not None
    relation = keyed_qk_relation(first.layers[0].query.numpy(), first.layers[0].key.numpy(), root_key)
    if not np.isfinite(relation.relation).all():
        raise ValueError("keyed relation compatibility is nonfinite")
    return {"status": "operational_preflight_complete", "science_denominator": 0, "runtime": _runtime_record(pipeline), "null_conditioning": null_record, "selected_paths": [shallow, deep], "images": results, "relation_compatibility": "pass", "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    args = parser.parse_args(argv)
    try:
        images = [Image.open(path).convert("RGB") for path in args.images]
        receipt = operational_preflight(images, hf_token=os.environ.get("HF_TOKEN", ""), root_key=os.environ.get("CEG_WM_ROOT_KEY", ""))
        print("CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT " + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except BaseException as error:
        error_class = type(error).__name__ if type(error).__name__ in _PUBLIC_ERRORS else "OtherOperationalError"
        print("CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE " + json.dumps({"status": "operational_failure", "stage": "preflight", "error_class": error_class}, sort_keys=True, separators=(",", ":")), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
