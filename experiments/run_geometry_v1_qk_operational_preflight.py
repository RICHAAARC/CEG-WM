"""One-shot, user-run SD3.5 Geometry-V1 operational preflight.

This module is deliberately an operational probe, not a detector or scientific
experiment.  It loads a supplied public runtime only when launched by the
future user-owned Colab handoff; local tests replace that loader with fakes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
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
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    public_config = getattr(pipeline, "config", None)
    revision = None
    resolution_status = "unavailable"
    for source in (pipeline, public_config):
        for name in ("_commit_hash",):
            value = getattr(source, name, None)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,64}", value):
                revision, resolution_status = value, "public_commit_hash"
    name_or_path = getattr(pipeline, "_name_or_path", getattr(pipeline, "name_or_path", None))
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None), "diffusers": version("diffusers"),
        "transformers": version("transformers"), "numpy": getattr(np, "__version__", None), "pillow": version("Pillow"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "requested_model_id": MODEL_ID, "requested_torch_dtype": str(torch.float16),
        "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
        "pipeline_name_or_path": str(name_or_path) if name_or_path is not None else None,
        "public_revision": revision, "revision_resolution_status": resolution_status,
        "vae_class": f"{type(getattr(pipeline, 'vae', None)).__module__}.{type(getattr(pipeline, 'vae', None)).__qualname__}",
        "transformer_class": f"{type(getattr(pipeline, 'transformer', None)).__module__}.{type(getattr(pipeline, 'transformer', None)).__qualname__}",
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


def _discover_candidates(transformer: torch.nn.Module) -> list[dict[str, Any]]:
    candidates: list[tuple[int, str]] = []
    for path, module in transformer.named_modules():
        match = _ATTENTION_PATH.search(path)
        if match and isinstance(getattr(module, "to_q", None), torch.nn.Module) and isinstance(getattr(module, "to_k", None), torch.nn.Module):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError("no transformer_blocks.<int>.attn module with to_q/to_k was found")
    candidates.sort()
    return [{"path": path, "block_index": index} for index, path in candidates]


def _discover_layers(transformer: torch.nn.Module) -> tuple[str, str]:
    candidates = _discover_candidates(transformer)
    shallow, deep = candidates[0], candidates[-1]
    if shallow["block_index"] == deep["block_index"]:
        raise ValueError("layer discovery requires distinct shallow and deep block indices")
    return shallow["path"], deep["path"]


def _observation_record(observation: SD35QKObservation, *, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "latent_shape": list(observation.latent_shape), "latent_dtype": None, "schedule_index": observation.schedule_index,
        "timestep": str(observation.timestep.item()), "elapsed_seconds": elapsed_seconds,
        "layers": [
            {"path": layer.layer_path, "query_shape": list(layer.query.shape), "key_shape": list(layer.key.shape),
             "dtype": str(layer.query.dtype), "source_dtype": str(layer.source_dtype), "device": str(layer.query.device), "finite": bool(torch.isfinite(layer.query).all() and torch.isfinite(layer.key).all()),
             "query_sha256": _tensor_digest(layer.query), "key_sha256": _tensor_digest(layer.key),
             "source_grid": list(layer.source_grid), "heads": layer.heads, "head_dim": layer.head_dim}
            for layer in observation.layers
        ],
    }


def _validate_execution_exact(expected_exact: str, repo_root: Path) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_exact) is None:
        raise ValueError("expected execution exact must be lowercase 40-hex")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected_exact or clean:
        raise RuntimeError("execution checkout identity differs")
    return f"geometry-v1-b2b-{expected_exact[:12]}-operational-01"


def _counter_targets(transformer: torch.nn.Module, paths: tuple[str, str]) -> tuple[dict[str, int], list[Any], dict[str, int]]:
    baseline: dict[str, int] = {}
    counts: dict[str, int] = {}
    handles: list[Any] = []
    for path in paths:
        attention = transformer.get_submodule(path)
        for name in ("to_q", "to_k"):
            module = getattr(attention, name)
            label = f"{path}.{name}"
            baseline[label] = len(module._forward_hooks)
            counts[label] = 0
            def count(_module: Any, _inputs: Any, _output: Any, *, label: str = label) -> None:
                counts[label] += 1
            handles.append(module.register_forward_hook(count))
    return baseline, handles, counts


def operational_preflight(
    images: list[Image.Image], *, hf_token: str, root_key: str | bytes,
    expected_exact: str, repo_root: Path,
    loader: Callable[..., Any] = load_sd35_pipeline,
) -> dict[str, Any]:
    """Run exactly the bounded operational probe and return a sanitized receipt."""
    if len(images) not in (1, 2):
        raise ValueError("operational preflight requires exactly one or two RGB images")
    if not isinstance(hf_token, str) or not hf_token:
        raise ValueError("HF_TOKEN is required")
    if not isinstance(root_key, (str, bytes)) or not root_key:
        raise ValueError("CEG_WM_ROOT_KEY is required")
    run_id = _validate_execution_exact(expected_exact, repo_root)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_geometry_v1_operational_preflight")
    pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
    pipeline = pipeline.to("cuda")
    null_hidden, null_pooled, null_record = _null_conditioning(pipeline)
    transformer = getattr(pipeline, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError("pipeline transformer must be a torch module")
    candidates = _discover_candidates(transformer)
    shallow, deep = _discover_layers(transformer)
    spec = SD35QKObservationSpec(MODEL_ID, str(getattr(pipeline, "revision", "runtime-resolved")), (shallow, deep), INFERENCE_STEPS, SCHEDULE_INDEX, PUBLIC_NOISE_SEED, MAX_GRID, null_hidden, null_pooled)
    results: list[dict[str, Any]] = []
    first: SD35QKObservation | None = None
    baseline, counter_handles, counts = _counter_targets(transformer, (shallow, deep))
    try:
      for index, image in enumerate(images):
        started = time.monotonic()
        before = dict(counts)
        observation = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
        record = _observation_record(observation, elapsed_seconds=time.monotonic() - started)
        record["input_sha256"] = _image_digest(image)
        record["projection_call_counts"] = {name: counts[name] - before[name] for name in counts}
        if any(value != 1 for value in record["projection_call_counts"].values()):
            raise ValueError("selected projections were not called exactly once")
        results.append(record)
        if index == 0:
            first = observation
            before = dict(counts)
            repeated = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
            if any(counts[name] - before[name] != 1 for name in counts):
                raise ValueError("repeated selected projections were not called exactly once")
            if _tensor_digest(repeated.layers[0].query) != _tensor_digest(observation.layers[0].query) or _tensor_digest(repeated.layers[0].key) != _tensor_digest(observation.layers[0].key):
                raise ValueError("first image observation is not deterministic")
    finally:
      for handle in counter_handles:
          handle.remove()
      for path in (shallow, deep):
          attention = transformer.get_submodule(path)
          for name in ("to_q", "to_k"):
              label = f"{path}.{name}"
              if len(getattr(attention, name)._forward_hooks) != baseline[label]:
                  raise RuntimeError("projection hook cleanup differs")
    assert first is not None
    relation = keyed_qk_relation(first.layers[0].query.numpy(), first.layers[0].key.numpy(), root_key)
    if not np.isfinite(relation.relation).all():
        raise ValueError("keyed relation compatibility is nonfinite")
    return {"status": "operational_preflight_complete", "run_id": run_id, "science_denominator": 0, "runtime": _runtime_record(pipeline), "null_conditioning": null_record, "candidate_layers": candidates, "selected_candidates": [{"path": path, "block_index": int(_ATTENTION_PATH.search(path).group(1))} for path in (shallow, deep)], "images": results, "relation_compatibility": "pass", "relation": {"relation_shape": list(relation.relation.shape), "relation_finite": bool(np.isfinite(relation.relation).all()), "projection_scalar_finite": bool(np.isfinite(relation.projection)), "coverage_finite": bool(np.isfinite(relation.coverage)), "gap_finite": bool(np.isfinite(relation.gap)), "wrong_key_margin_finite": bool(np.isfinite(relation.wrong_key_margin))}, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    args = parser.parse_args(argv)
    try:
        images = [Image.open(path).convert("RGB") for path in args.images]
        run_id = f"geometry-v1-b2b-{args.expected_exact[:12]}-operational-01" if re.fullmatch(r"[0-9a-f]{40}", args.expected_exact) else None
        receipt = operational_preflight(images, hf_token=os.environ.get("HF_TOKEN", ""), root_key=os.environ.get("CEG_WM_ROOT_KEY", ""), expected_exact=args.expected_exact, repo_root=Path(args.repo_root))
        print("CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT " + json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except BaseException as error:
        error_class = type(error).__name__ if type(error).__name__ in _PUBLIC_ERRORS else "OtherOperationalError"
        print("CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE " + json.dumps({"status": "operational_failure", "run_id": run_id, "stage": "preflight", "error_class": error_class}, sort_keys=True, separators=(",", ":")), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
