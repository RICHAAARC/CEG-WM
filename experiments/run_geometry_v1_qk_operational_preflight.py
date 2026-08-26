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
import zipfile
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
_SNAPSHOT_HEX = re.compile(r"(?:^|/)snapshots/([0-9a-f]{7,64})(?:/|$)")
_PUBLIC_ERRORS = {"FileNotFoundError", "ImportError", "ModuleNotFoundError", "OSError", "RuntimeError", "TypeError", "ValueError"}
_FAILURE_POINTS = frozenset({
    "model_load",
    "null_conditioning_call",
    "null_conditioning_validate",
    "vae_encode",
    "scheduler",
    "transformer_call",
    "qk_capture",
    "relation",
    "receipt_packaging",
})
MAX_CONTROL_BYTES = 1024
MAX_RECEIPT_BYTES = 262144
MAX_ARCHIVE_BYTES = 524288
MAX_SIDECAR_BYTES = 256
_SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT "
_FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE "


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    return _sha256_bytes(detached.numpy().tobytes())


def _image_digest(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ValueError("input image must be ordinary RGB")
    return _sha256_bytes(np.asarray(image, dtype=np.uint8).tobytes())


def _public_revision(pipeline: Any) -> tuple[str | None, str]:
    """Use only a public commit-shaped identity; unknown is recorded, never guessed."""
    for source in (pipeline, getattr(pipeline, "config", None)):
        value = getattr(source, "_commit_hash", None)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,64}", value):
            return value, "proven_public_commit"
    snapshots = _component_snapshot_candidates(pipeline)
    if len(snapshots) == 1:
        return next(iter(snapshots)), "unique_public_snapshot"
    return None, "unavailable_from_public_runtime"


def _module_dtype(value: Any) -> str | None:
    try:
        return str(next(value.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        return None


def _snapshot_hex(value: Any) -> str | None:
    """Extract only a commit-shaped public snapshot name, never its path."""
    if not isinstance(value, str):
        return None
    match = _SNAPSHOT_HEX.search(value)
    return match.group(1) if match else None


def _public_name_or_path(component: Any) -> Any:
    return getattr(component, "_name_or_path", getattr(component, "name_or_path", None))


def _component_snapshot_candidates(component: Any) -> set[str]:
    config = getattr(component, "config", None)
    return {
        candidate for candidate in (
            _snapshot_hex(_public_name_or_path(component)),
            _snapshot_hex(_public_name_or_path(config)),
        ) if candidate is not None
    }


def _component_identity(component: Any) -> dict[str, Any]:
    """Return the fixed, path-safe public identity record for one component."""
    config = getattr(component, "config", None)
    scalar: dict[str, str | int | float | bool | None] = {}
    for name, value in vars(config).items() if hasattr(config, "__dict__") else ():
        lowered = name.lower()
        if any(token in lowered for token in ("token", "key", "secret", "path", "directory", "file")):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar[name] = value
    encoded = json.dumps(scalar, sort_keys=True, separators=(",", ":")).encode("utf-8")
    commits = [
        getattr(source, "_commit_hash", None)
        for source in (component, config)
    ]
    commit = next((value for value in commits if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,64}", value)), None)
    snapshots = _component_snapshot_candidates(component)
    source = _public_name_or_path(component)
    return {
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "commit_candidate": commit,
        "snapshot_candidate": next(iter(snapshots)) if len(snapshots) == 1 else None,
        "sanitized_config_digest": _sha256_bytes(encoded),
        "public_name_or_path": MODEL_ID if source == MODEL_ID else None,
    }


def _runtime_record(pipeline: Any) -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    revision, resolution_status = _public_revision(pipeline)
    components = {name: _component_identity(getattr(pipeline, name, None)) for name in ("pipeline", "vae", "transformer", "scheduler", "image_processor")}
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None), "diffusers": version("diffusers"),
        "transformers": version("transformers"), "numpy": getattr(np, "__version__", None), "pillow": version("Pillow"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "requested_model_id": MODEL_ID, "requested_revision": None, "requested_torch_dtype": str(torch.float16), "selected_device": "cuda",
        "resolved_revision": revision, "resolution_status": resolution_status,
        "vae_class": f"{type(getattr(pipeline, 'vae', None)).__module__}.{type(getattr(pipeline, 'vae', None)).__qualname__}",
        "transformer_class": f"{type(getattr(pipeline, 'transformer', None)).__module__}.{type(getattr(pipeline, 'transformer', None)).__qualname__}",
        "scheduler_class": f"{type(getattr(pipeline, 'scheduler', None)).__module__}.{type(getattr(pipeline, 'scheduler', None)).__qualname__}",
        "image_processor_class": f"{type(getattr(pipeline, 'image_processor', None)).__module__}.{type(getattr(pipeline, 'image_processor', None)).__qualname__}",
        "vae_dtype": _module_dtype(getattr(pipeline, "vae", None)), "transformer_dtype": _module_dtype(getattr(pipeline, "transformer", None)),
        "components": components,
    }


def _null_conditioning(pipeline: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    encode_prompt = getattr(pipeline, "encode_prompt", None)
    if not callable(encode_prompt):
        raise TypeError("pipeline must expose public encode_prompt for the one null construction")
    try:
        result = encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            do_classifier_free_guidance=False,
        )
    except BaseException as error:
        setattr(error, "geometry_failure_point", "null_conditioning_call")
        raise
    try:
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("null encode_prompt must return the SD3 four-item tuple")
    # StableDiffusion3Pipeline returns (prompt, negative_prompt, pooled,
    # negative_pooled).  The no-CFG negative positions are intentionally not
    # consumed by this image-observation preflight.
        hidden, pooled = result[0], result[2]
        if not isinstance(hidden, torch.Tensor) or not isinstance(pooled, torch.Tensor):
            raise TypeError("selected null encode_prompt outputs must be tensors")
        if not torch.is_floating_point(hidden) or not torch.is_floating_point(pooled):
            raise TypeError("selected null encode_prompt outputs must be floating tensors")
        if hidden.ndim < 2 or pooled.ndim < 2:
            raise ValueError("selected null conditioning tensors must be rank two or higher")
        if hidden.shape[0] != 1 or pooled.shape[0] != 1:
            raise ValueError("null conditioning must be batch one")
        if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(pooled).all()):
            raise ValueError("selected null conditioning tensors must be finite")
    except BaseException as error:
        setattr(error, "geometry_failure_point", "null_conditioning_validate")
        raise
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


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return [int(dimension) for dimension in value.shape]


def _projection_identity(value: Any) -> dict[str, Any]:
    """Bounded public facts about an eligible sample-side projection."""
    return {
        "present": isinstance(value, torch.nn.Module),
        "class": f"{type(value).__module__}.{type(value).__qualname__}" if isinstance(value, torch.nn.Module) else None,
        "in_features": getattr(value, "in_features", None) if isinstance(value, torch.nn.Module) else None,
        "out_features": getattr(value, "out_features", None) if isinstance(value, torch.nn.Module) else None,
        "weight_shape": _shape(getattr(value, "weight", None)) if isinstance(value, torch.nn.Module) else None,
        "dtype": _module_dtype(value) if isinstance(value, torch.nn.Module) else None,
        "device": str(getattr(getattr(value, "weight", None), "device", None)) if isinstance(value, torch.nn.Module) and isinstance(getattr(value, "weight", None), torch.Tensor) else None,
    }


def _architecture_record(transformer: torch.nn.Module, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Record public SD3 attention topology without retaining weights or tensors."""
    config = getattr(transformer, "config", None)
    config_fields = (
        "num_layers", "patch_size", "in_channels", "joint_attention_dim",
        "caption_projection_dim", "num_attention_heads", "attention_head_dim",
    )
    inventory: list[dict[str, Any]] = []
    for candidate in candidates:
        path = candidate["path"]
        attention = transformer.get_submodule(path)
        block = transformer.get_submodule(path.rsplit(".", 1)[0])
        inventory.append({
            "path": path,
            "block_index": candidate["block_index"],
            "attention_class": f"{type(attention).__module__}.{type(attention).__qualname__}",
            "processor_class": f"{type(getattr(attention, 'processor', None)).__module__}.{type(getattr(attention, 'processor', None)).__qualname__}" if getattr(attention, "processor", None) is not None else None,
            "to_q": _projection_identity(getattr(attention, "to_q", None)),
            "to_k": _projection_identity(getattr(attention, "to_k", None)),
            "other_routes": {
                # In SD3 JointTransformerBlock, attn2 is a sibling of the
                # sample-side attn; the added/fused projections belong to attn.
                "attn2": isinstance(getattr(block, "attn2", None), torch.nn.Module),
                **{
                    name: isinstance(getattr(attention, name, None), torch.nn.Module)
                    for name in ("add_q_proj", "add_k_proj", "to_qkv")
                },
            },
        })
    return {
        "transformer_class": f"{type(transformer).__module__}.{type(transformer).__qualname__}",
        "config": {name: getattr(config, name, None) for name in config_fields},
        "attention_candidates": inventory,
    }


def _observation_record(observation: SD35QKObservation, *, elapsed_seconds: float) -> dict[str, Any]:
    source_grid = observation.layers[0].source_grid
    return {
        "latent_shape": list(observation.latent_shape), "schedule_index": observation.schedule_index,
        "latent_grid": list(observation.latent_shape[-2:]), "patch_grid": list(source_grid),
        "token_count": source_grid[0] * source_grid[1],
        "timestep": str(observation.timestep.item()), "elapsed_seconds": elapsed_seconds,
        "layers": [
            {"path": layer.layer_path, "query_shape": list(layer.query.shape), "key_shape": list(layer.key.shape),
             "dtype": str(layer.query.dtype), "source_dtype": str(layer.source_dtype), "source_device": str(layer.source_device), "source_query_shape": list(layer.source_shape), "source_key_shape": list(layer.source_shape), "device": str(layer.query.device), "finite": bool(torch.isfinite(layer.query).all() and torch.isfinite(layer.key).all()),
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


def _counter_targets(transformer: torch.nn.Module, paths: tuple[str, str]) -> tuple[dict[str, int], list[Any], dict[str, int], dict[str, Any]]:
    baseline: dict[str, int] = {}
    counts: dict[str, int] = {}
    handles: list[Any] = []
    transformer_baseline = len(transformer._forward_pre_hooks)
    hidden_metadata: dict[str, Any] = {"count": 0, "device": None, "dtype": None, "baseline": transformer_baseline}
    def prehook(_module: Any, _args: Any, kwargs: Any) -> None:
        hidden = kwargs.get("hidden_states") if isinstance(kwargs, dict) else None
        if not isinstance(hidden, torch.Tensor):
            raise ValueError("transformer prehook did not receive hidden_states")
        hidden_metadata["count"] += 1
        hidden_metadata["device"] = str(hidden.device)
        hidden_metadata["dtype"] = str(hidden.dtype)
    handles.append(transformer.register_forward_pre_hook(prehook, with_kwargs=True))
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
    return baseline, handles, counts, hidden_metadata


def _observation_failure_point(error: BaseException) -> str:
    """Use the adapter's bounded stage tag; never classify exception text."""
    point = getattr(error, "geometry_failure_point", None)
    return point if point in _FAILURE_POINTS else "qk_capture"


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
    try:
        pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
        pipeline = pipeline.to("cuda")
    except BaseException as error:
        setattr(error, "geometry_failure_point", "model_load")
        raise
    runtime = _runtime_record(pipeline)
    architecture: dict[str, Any] | None = None
    try:
        null_hidden, null_pooled, null_record = _null_conditioning(pipeline)
        transformer = getattr(pipeline, "transformer", None)
        if not isinstance(transformer, torch.nn.Module):
            raise TypeError("pipeline transformer must be a torch module")
        candidates = _discover_candidates(transformer)
        architecture = _architecture_record(transformer, candidates)
        shallow, deep = _discover_layers(transformer)
        resolved_revision, _ = _public_revision(pipeline)
        spec = SD35QKObservationSpec(MODEL_ID, resolved_revision, (shallow, deep), INFERENCE_STEPS, SCHEDULE_INDEX, PUBLIC_NOISE_SEED, MAX_GRID, null_hidden, null_pooled)
        results: list[dict[str, Any]] = []
        first: SD35QKObservation | None = None
        baseline, counter_handles, counts, hidden_metadata = _counter_targets(transformer, (shallow, deep))
        try:
          for index, image in enumerate(images):
            started = time.monotonic()
            before = dict(counts)
            try:
                observation = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
            except BaseException as error:
                setattr(error, "geometry_failure_point", _observation_failure_point(error))
                raise
            record = _observation_record(observation, elapsed_seconds=time.monotonic() - started)
            record["input_sha256"] = _image_digest(image)
            record["projection_call_counts"] = {name: counts[name] - before[name] for name in counts}
            if any(value != 1 for value in record["projection_call_counts"].values()):
                raise ValueError("selected projections were not called exactly once")
            results.append(record)
            if index == 0:
                first = observation
                before = dict(counts)
                try:
                    repeated = observe_sd35_image_qk(image, pipeline=pipeline, spec=spec)
                except BaseException as error:
                    setattr(error, "geometry_failure_point", _observation_failure_point(error))
                    raise
                repeat_counts = {name: counts[name] - before[name] for name in counts}
                record["repeat_projection_call_counts"] = repeat_counts
                if any(value != 1 for value in repeat_counts.values()):
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
          if len(transformer._forward_pre_hooks) != hidden_metadata["baseline"]:
              raise RuntimeError("transformer prehook cleanup differs")
        assert first is not None
        try:
            relation = keyed_qk_relation(first.layers[0].query.numpy(), first.layers[0].key.numpy(), root_key)
        except BaseException as error:
            setattr(error, "geometry_failure_point", "relation")
            raise
        if not np.isfinite(relation.relation).all():
            raise ValueError("keyed relation compatibility is nonfinite")
        return {"status": "operational_preflight_complete", "run_id": run_id, "science_denominator": 0, "runtime": runtime, "architecture": architecture, "null_conditioning": null_record, "candidate_layers": candidates, "selected_candidates": [{"path": path, "block_index": int(_ATTENTION_PATH.search(path).group(1))} for path in (shallow, deep)], "transformer_hidden_states": {"device": hidden_metadata["device"], "dtype": hidden_metadata["dtype"], "call_count": hidden_metadata["count"]}, "hook_cleanup": True, "images": results, "relation_compatibility": "pass", "relation": {"relation_shape": list(relation.relation.shape), "relation_finite": bool(np.isfinite(relation.relation).all()), "projection_scalar_finite": bool(np.isfinite(relation.projection)), "coverage_finite": bool(np.isfinite(relation.coverage)), "gap_finite": bool(np.isfinite(relation.gap)), "wrong_key_margin_finite": bool(np.isfinite(relation.wrong_key_margin))}, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}
    except BaseException as error:
        setattr(error, "geometry_runtime_record", runtime)
        if architecture is not None:
            setattr(error, "geometry_architecture_record", architecture)
        raise


def _bounded_json(value: dict[str, Any], limit: int) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > limit:
        raise ValueError("sanitized receipt exceeds bounded package limit")
    return encoded


def _write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _package_receipt(*, output_root: Path, receipt: dict[str, Any], status_name: str, expected_exact: str, run_id: str) -> dict[str, Any]:
    """Create the runner-owned, create-only terminal package before control."""
    if output_root.exists():
        raise FileExistsError("runner output root must be create-only")
    output_root.mkdir()
    receipt_bytes = _bounded_json(receipt, MAX_RECEIPT_BYTES)
    _write_exclusive(output_root / "receipt.json", receipt_bytes)
    _write_exclusive(output_root / status_name, _bounded_json({"status": receipt["status"], "run_id": run_id}, MAX_RECEIPT_BYTES))
    # This is an operational checkpoint only.  It deliberately contains no
    # resume state, tensors, secrets, paths, or scientific conclusion.
    _write_exclusive(output_root / "checkpoint.json", _bounded_json({"status": receipt["status"], "run_id": run_id, "checkpoint": "terminal"}, MAX_RECEIPT_BYTES))
    members = ["receipt.json", status_name, "checkpoint.json", "manifest.json", "SHA256SUMS"]
    _write_exclusive(output_root / "manifest.json", _bounded_json({"execution_exact": expected_exact, "run_id": run_id, "allowed_filenames": members}, MAX_RECEIPT_BYTES))
    sums = "".join(f"{_sha256_bytes((output_root / name).read_bytes())}  {name}\n" for name in members[:-1]).encode("ascii")
    _write_exclusive(output_root / "SHA256SUMS", sums)
    archive_name = f"{run_id}.zip"
    archive = output_root / archive_name
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in members:
            bundle.write(output_root / name, name)
    archive_bytes = archive.stat().st_size
    if archive_bytes > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds bounded package limit")
    digest = _sha256_bytes(archive.read_bytes())
    sidecar_name = f"{archive_name}.sha256"
    sidecar = output_root / sidecar_name
    sidecar_value = f"{digest}  {archive_name}\n".encode("ascii")
    if len(sidecar_value) > MAX_SIDECAR_BYTES:
        raise ValueError("sidecar exceeds bounded package limit")
    _write_exclusive(sidecar, sidecar_value)
    return {"archive_filename": archive_name, "sidecar_filename": sidecar_name, "receipt_bytes": len(receipt_bytes), "receipt_sha256": _sha256_bytes(receipt_bytes), "archive_bytes": archive_bytes}


def _emit_control(fd: int, prefix: str, control: dict[str, Any]) -> None:
    line = prefix.encode("ascii") + _bounded_json(control, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES:
        raise ValueError("compact control exceeds bounded transport limit")
    os.write(fd, line)


def _failure_receipt(error: BaseException, run_id: str | None) -> dict[str, Any]:
    error_class = type(error).__name__ if type(error).__name__ in _PUBLIC_ERRORS else "OtherOperationalError"
    failure_point = getattr(error, "geometry_failure_point", "receipt_packaging")
    if failure_point not in _FAILURE_POINTS:
        failure_point = "receipt_packaging"
    failure: dict[str, Any] = {"status": "operational_failure", "run_id": run_id, "stage": "preflight", "failure_point": failure_point, "error_class": error_class, "science_denominator": 0}
    for name, attribute in (("runtime", "geometry_runtime_record"), ("architecture", "geometry_architecture_record")):
        value = getattr(error, attribute, None)
        if isinstance(value, dict):
            failure[name] = value
    return failure


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv)
    run_id = f"geometry-v1-b2b-{args.expected_exact[:12]}-operational-01" if re.fullmatch(r"[0-9a-f]{40}", args.expected_exact) else None
    try:
        images = [Image.open(path).convert("RGB") for path in args.images]
        receipt = operational_preflight(images, hf_token=os.environ.get("HF_TOKEN", ""), root_key=os.environ.get("CEG_WM_ROOT_KEY", ""), expected_exact=args.expected_exact, repo_root=Path(args.repo_root))
        package = _package_receipt(output_root=Path(args.output_root), receipt=receipt, status_name="success.json", expected_exact=args.expected_exact, run_id=receipt["run_id"])
        _emit_control(args.control_fd, _SUCCESS_PREFIX, {"status": "success", "run_id": receipt["run_id"], "artifact_status": "complete", **package})
        return 0
    except BaseException as error:
        failure = _failure_receipt(error, run_id)
        try:
            package = _package_receipt(output_root=Path(args.output_root), receipt=failure, status_name="failure.json", expected_exact=args.expected_exact, run_id=run_id or "invalid-exact")
            control = {"status": "failure", "underlying_status": "operational_failure", "artifact_status": "complete", "failure_point": failure["failure_point"], "run_id": run_id, **package}
        except BaseException:
            control = {"status": "failure", "underlying_status": "unknown", "artifact_status": "unavailable", "failure_point": "receipt_packaging", "run_id": run_id}
        try:
            _emit_control(args.control_fd, _FAILURE_PREFIX, control)
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
