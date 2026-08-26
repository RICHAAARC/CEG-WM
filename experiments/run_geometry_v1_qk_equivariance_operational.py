"""Bounded, image-only operational runner for the E0 Q/K experiment.

This module is deliberately an experiment transport: its unit calculation is
owned by the E0 harness and its receipt never adjudicates method or science.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.sd35_qk_observation import SD35QKObservation, SD35QKObservationSpec, observe_sd35_image_qk

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MAX_CONTROL_BYTES, MAX_UNIT_COUNT, MAX_UNIT_BYTES = 1024, 64, 16384
MAX_TOTAL_UNIT_BYTES, MAX_SUMMARY_BYTES, MAX_MANIFEST_BYTES = 1048576, 262144, 131072
MAX_ARCHIVE_BYTES, MAX_SIDECAR_BYTES = 2097152, 256
SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_QK_E0 "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_QK_E0_FAILURE "
PLAN_SCHEMA = "geometry-v1-qk-e0-plan-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z", re.ASCII)
_TRANSFORMS = ("identity", "d4", "similarity", "crop_rescale")
_CONTROLS = ("matched_h", "shuffled_h")
_FAILURE_POINTS = frozenset({"plan", "model_load", "image_observation", "artifact_packaging"})

_harness_spec = importlib.util.spec_from_file_location("geometry_e0_harness", Path(__file__).with_name("run_geometry_v1_qk_equivariance_preflight.py"))
assert _harness_spec and _harness_spec.loader
HARNESS = importlib.util.module_from_spec(_harness_spec)
_harness_spec.loader.exec_module(HARNESS)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: Mapping[str, Any] | Sequence[Any], maximum: int) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > maximum:
        raise ValueError("bounded_json_exceeded")
    return data


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def _exact(expected_exact: str, repo_root: Path) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_exact):
        raise ValueError("invalid_expected_exact")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected_exact or clean:
        raise RuntimeError("execution_identity_mismatch")
    return actual


def _valid_identifier(value: Any, maximum: int = 128) -> bool:
    return isinstance(value, str) and len(value) <= maximum and bool(_IDENT.fullmatch(value))


def _validate_plan(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(plan, Mapping) or set(plan) != {"schema", "attention_layer_paths", "pairs"} or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid_plan_schema")
    paths, pairs = plan["attention_layer_paths"], plan["pairs"]
    if not isinstance(paths, list) or len(paths) != 2 or len(set(paths)) != 2 or not all(_valid_identifier(p, 256) for p in paths):
        raise ValueError("invalid_attention_layer_paths")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise ValueError("invalid_pair_count")
    required = {"reference_id", "pair_id", "transform_label", "reference_path", "reference_sha256", "attacked_path", "attacked_sha256", "reference_source_grid", "attacked_source_grid", "matched_h", "shuffled_h"}
    by_reference: dict[str, set[str]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != required:
            raise ValueError("invalid_pair_schema")
        if not _valid_identifier(pair["reference_id"]) or not _valid_identifier(pair["pair_id"]):
            raise ValueError("invalid_pair_identifier")
        if pair["transform_label"] not in _TRANSFORMS or not all(isinstance(pair[name], str) and _HEX.fullmatch(pair[name]) for name in ("reference_sha256", "attacked_sha256")):
            raise ValueError("invalid_pair_manifest")
        if not all(isinstance(pair[name], str) and 1 <= len(pair[name]) <= 4096 for name in ("reference_path", "attacked_path")):
            raise ValueError("invalid_private_input_path")
        for name in ("reference_source_grid", "attacked_source_grid"):
            grid = pair[name]
            if not isinstance(grid, list) or len(grid) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid):
                raise ValueError("invalid_expected_source_grid")
        for name in ("matched_h", "shuffled_h"):
            h = np.asarray(pair[name], dtype=np.float64)
            if h.shape != (3, 3) or not np.isfinite(h).all():
                raise ValueError("invalid_h_reference_to_attacked")
        by_reference.setdefault(pair["reference_id"], set()).add(pair["transform_label"])
    if len(by_reference) != 2 or any(labels != set(_TRANSFORMS) for labels in by_reference.values()):
        raise ValueError("invalid_reference_transform_matrix")
    return list(pairs)


def _failure_unit(pair: Mapping[str, Any], layer_path: str, descriptor_kind: str, control: str, reason: str) -> dict[str, Any]:
    return {"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": descriptor_kind, "layer_path": layer_path, "reference_grid": None, "attacked_grid": None, "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": reason, "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}


def _image(path: str, declared: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    digest = _sha(np.asarray(image, dtype=np.uint8).tobytes())
    if digest != declared:
        raise ValueError("image_digest_mismatch")
    return image


def _null(pipeline: Any) -> tuple[torch.Tensor, torch.Tensor]:
    result = pipeline.encode_prompt(prompt="", prompt_2="", prompt_3="", do_classifier_free_guidance=False)
    if not isinstance(result, (tuple, list)) or len(result) != 4 or not isinstance(result[0], torch.Tensor) or not isinstance(result[2], torch.Tensor):
        raise ValueError("invalid_null_conditioning")
    return result[0].detach(), result[2].detach()


def _spec(pipeline: Any, paths: tuple[str, str]) -> SD35QKObservationSpec:
    hidden, pooled = _null(pipeline)
    return SD35QKObservationSpec(MODEL_ID, getattr(pipeline, "_commit_hash", None), paths, 20, 7, 0, (8, 8), hidden, pooled)


def _layer(observation: SD35QKObservation, path: str) -> Any:
    for layer in observation.layers:
        if layer.layer_path == path:
            return layer
    raise ValueError("selected_layer_not_observed")


def _unit(pair: Mapping[str, Any], reference: Any, attacked: Any, path: str, kind: str, control: str) -> dict[str, Any]:
    rlayer, alayer = _layer(reference, path), _layer(attacked, path)
    descriptor_name = "query" if kind == "q" else "key"
    if rlayer.source_grid != tuple(pair["reference_source_grid"]) or alayer.source_grid != tuple(pair["attacked_source_grid"]):
        return _failure_unit(pair, path, kind, control, "source_grid_mismatch")
    return HARNESS.evaluate_unit({"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path, "reference_descriptors": getattr(rlayer, descriptor_name).detach().cpu().numpy(), "attacked_descriptors": getattr(alayer, descriptor_name).detach().cpu().numpy(), "reference_source_grid": rlayer.source_grid, "attacked_source_grid": alayer.source_grid, "reference_sample_indices": rlayer.sample_indices.detach().cpu().numpy(), "attacked_sample_indices": alayer.sample_indices.detach().cpu().numpy(), "H_reference_to_attacked": pair[control]})


def _runtime(pipeline: Any) -> dict[str, Any]:
    transformer = getattr(pipeline, "transformer", None)
    return {"pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}", "transformer_class": f"{type(transformer).__module__}.{type(transformer).__qualname__}", "device": str(next(transformer.parameters()).device) if transformer is not None else None, "dtype": str(next(transformer.parameters()).dtype) if transformer is not None else None, "resolved_public_revision": getattr(pipeline, "_commit_hash", None)}


def run_qk_equivariance_operational(plan: Mapping[str, Any], *, hf_token: str, expected_exact: str, repo_root: Path, loader: Callable[..., Any] = load_sd35_pipeline, observer: Callable[..., SD35QKObservation] = observe_sd35_image_qk) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Execute the fixed E0 plan once; all method outcomes remain descriptive."""
    pairs = _validate_plan(plan)
    exact = _exact(expected_exact, repo_root)
    paths = tuple(plan["attention_layer_paths"])
    run_id = f"geometry-v1-qk-e0-{exact[:12]}"
    units: list[dict[str, Any]] = []
    operational_status, resource_status, failure_point = "complete", "available", None
    runtime: dict[str, Any] = {}
    pipeline = None
    try:
        pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
        if hasattr(pipeline, "to"):
            pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        runtime = _runtime(pipeline)
        spec = _spec(pipeline, paths)
    except BaseException:
        operational_status, resource_status, failure_point = "failure", "unavailable", "model_load"
        spec = None
    for pair in pairs:
        reference = attacked = None
        pair_reason = None
        if spec is None:
            pair_reason = "runtime_not_observed"
        else:
            try:
                reference = observer(_image(pair["reference_path"], pair["reference_sha256"]), pipeline=pipeline, spec=spec)
                attacked = observer(_image(pair["attacked_path"], pair["attacked_sha256"]), pipeline=pipeline, spec=spec)
            except BaseException:
                operational_status, failure_point = "failure", "image_observation"
                pair_reason = "image_observation_failed"
        for path in paths:
            for kind in ("q", "k"):
                for control in _CONTROLS:
                    if pair_reason:
                        units.append(_failure_unit(pair, path, kind, control, pair_reason))
                    else:
                        try:
                            units.append(_unit(pair, reference, attacked, path, kind, control))
                        except (AttributeError, KeyError, TypeError, ValueError):
                            units.append(_failure_unit(pair, path, kind, control, "selected_layer_observation_invalid"))
    if len(units) != MAX_UNIT_COUNT:
        raise RuntimeError("fixed_unit_expansion_mismatch")
    summary = {"schema": "geometry-v1-qk-e0-operational-v1", "run_id": run_id, "execution_identity": {"commit": exact}, "plan_digest": _sha(_json(plan, 65536)), "operational_status": operational_status, "resource_status": resource_status, "artifact_status": "unavailable", "method_status": "not_adjudicated", "scientific_status": "not_adjudicated", "science_denominator": 0, "operational_failure_point": failure_point, "runtime": runtime, "declared_unit_count": 64, "calculated_unit_count": sum(item["status"] == "calculated" for item in units), "failed_unit_count": sum(item["status"] == "failed" for item in units)}
    return summary, tuple(units)


def _package(output_root: Path, summary: dict[str, Any], units: Sequence[dict[str, Any]], *, expected_exact: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("output_root_must_be_create_only")
    output_root.mkdir(); (output_root / "units").mkdir()
    total, entries = 0, []
    for ordinal, unit in enumerate(units):
        data = _json(unit, MAX_UNIT_BYTES); total += len(data)
        if total > MAX_TOTAL_UNIT_BYTES: raise ValueError("total_unit_bound_exceeded")
        name = f"units/{ordinal:03d}.json"; _write(output_root / name, data)
        entries.append({"ordinal": ordinal, "filename": name, "bytes": len(data), "sha256": _sha(data), **{key: unit[key] for key in ("pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path")}})
    status_name = "success.json" if summary["operational_status"] == "complete" else "failure.json"
    summary["artifact_status"] = "complete"
    summary["unit_manifest"] = entries
    receipt = _json(summary, MAX_SUMMARY_BYTES); _write(output_root / "receipt.json", receipt)
    _write(output_root / status_name, _json({"run_id": summary["run_id"], "operational_status": summary["operational_status"]}, 1024))
    _write(output_root / "checkpoint.json", _json({"run_id": summary["run_id"], "checkpoint": "terminal"}, 1024))
    members = ["receipt.json", status_name, "checkpoint.json", *[entry["filename"] for entry in entries], "manifest.json", "SHA256SUMS"]
    manifest = {"run_id": summary["run_id"], "execution_exact": expected_exact, "members": members, "units": entries}
    _write(output_root / "manifest.json", _json(manifest, MAX_MANIFEST_BYTES))
    sums = b"".join(f"{_sha((output_root / name).read_bytes())}  {name}\n".encode("ascii") for name in members[:-1])
    _write(output_root / "SHA256SUMS", sums)
    archive_name = f"{summary['run_id']}.zip"; archive = output_root / archive_name
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in members: bundle.write(output_root / name, name)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES: raise ValueError("archive_bound_exceeded")
    digest = _sha(archive.read_bytes()); sidecar_name = archive_name + ".sha256"; sidecar = f"{digest}  {archive_name}\n".encode("ascii")
    if len(sidecar) > MAX_SIDECAR_BYTES: raise ValueError("sidecar_bound_exceeded")
    _write(output_root / sidecar_name, sidecar)
    return {"archive_filename": archive_name, "sidecar_filename": sidecar_name, "receipt_bytes": len(receipt), "receipt_sha256": _sha(receipt), "archive_bytes": archive.stat().st_size}


def _emit(fd: int, prefix: str, control: Mapping[str, Any]) -> None:
    line = prefix.encode("ascii") + _json(control, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES: raise ValueError("control_bound_exceeded")
    os.write(fd, line)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", required=True); parser.add_argument("--repo-root", required=True); parser.add_argument("--expected-exact", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv); run_id = f"geometry-v1-qk-e0-{args.expected_exact[:12]}"
    summary: dict[str, Any] | None = None
    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        summary, units = run_qk_equivariance_operational(plan, hf_token=os.environ.get("HF_TOKEN", ""), expected_exact=args.expected_exact, repo_root=Path(args.repo_root))
        package = _package(Path(args.output_root), summary, units, expected_exact=args.expected_exact)
        _emit(args.control_fd, SUCCESS_PREFIX if summary["operational_status"] == "complete" else FAILURE_PREFIX, {"status": "success" if summary["operational_status"] == "complete" else "failure", "artifact_status": "complete", "run_id": summary["run_id"], **package})
        return 0 if summary["operational_status"] == "complete" else 1
    except BaseException:
        control = {"status": "failure", "underlying_status": "operational_failure" if summary and summary["operational_status"] == "failure" else "unknown", "artifact_status": "unavailable", "failure_point": "artifact_packaging", "run_id": run_id}
        try: _emit(args.control_fd, FAILURE_PREFIX, control)
        except BaseException: pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
