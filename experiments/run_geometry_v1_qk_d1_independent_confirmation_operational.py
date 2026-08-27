"""D1 independent, fixed-layer Q/K direction confirmation.

This runner reads one immutable D0.1 sidecar artifact and performs a new,
predeclared image-only observation roster.  It never selects layers, revises
the D0/D0.1 result, or produces a detector or scientific conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import zipfile
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.sd35_qk_observation import SD35QKObservation, SD35QKObservationSpec, observe_sd35_image_qk

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PROTOCOL = "geometry-v1-qk-d1-independent-confirmation-v1"
SCHEMA = "geometry-v1-qk-d1-independent-confirmation-operational-v1"
SOURCE_RUN_ID = "geometry-v1-qk-d01-ccfb7bcefbb1"
SOURCE_EXACT = "ccfb7bcefbb18f9812a4e800bbea18b91b031ebb"
SOURCE_PROTOCOL = "geometry-v1-qk-d01-artifact-selection-v1"
SOURCE_STATUS = "D01_CANDIDATES_FROZEN"
ATTENTION_LAYER_PATHS = ("transformer_blocks.6.attn", "transformer_blocks.13.attn", "transformer_blocks.18.attn")
SOURCE_SELECTED_PATHS = list(ATTENTION_LAYER_PATHS)
TRANSFORMS = ("identity", "d4", "similarity", "crop_rescale")
KINDS, CONTROLS = ("q", "k"), ("matched_h", "shuffled_h")
REFERENCE_IDS = ("confirmation_a", "confirmation_b")
UNIT_COUNT = 96
MAX_CONTROL_BYTES, MAX_ROOT_BYTES = 1024, 262144
MAX_UNIT_BYTES, MAX_LAYER_UNIT_BYTES, MAX_LAYER_ZIP_BYTES = 16384, 524288, 1048576
MAX_RUN_BYTES = 12582912
SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_QK_D1 "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_QK_D1_FAILURE "
UNIT_FIELDS = frozenset(("pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path",
                         "reference_grid", "attacked_grid", "input_identity", "h_identity", "status",
                         "failure_reason", "candidate_correspondences", "true_match_ranks", "coverage",
                         "ambiguity_gaps", "fit_residual", "recovery_error"))
_VALUE_LEAKS = (re.compile(r"\braw\s*(?:q\s*/\s*k|qk|query|key|token(?:\s+material)?)\b", re.I),
                re.compile(r"\b(?:hf[_ -]?token|access[_ -]?token|api[_ -]?key|token\s+(?:material|credential|secret|value|data)|credential(?:s)?\s+(?:material|value|data))\b", re.I))

_hspec = importlib.util.spec_from_file_location("geometry_d1_harness", Path(__file__).with_name("run_geometry_v1_qk_equivariance_preflight.py"))
assert _hspec and _hspec.loader
HARNESS = importlib.util.module_from_spec(_hspec); _hspec.loader.exec_module(HARNESS)


def _json(value: Any, maximum: int) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > maximum: raise ValueError("bounded_json_exceeded")
    return data


def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle: handle.write(data)


def _exact(expected: str, root: Path) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected): raise ValueError("invalid_expected_exact")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected or dirty: raise RuntimeError("execution_identity_mismatch")
    return actual


def _read_json(path: Path, maximum: int) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum: raise ValueError("invalid_bounded_source_file")
    try: value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError("invalid_source_json") from error
    if not isinstance(value, dict): raise ValueError("invalid_source_json")
    return value


def _reject_leak(value: Any, *, depth: int = 0) -> None:
    """Bounded recursive public-artifact leak rejection before identity use."""
    if depth > 64: raise ValueError("public_value_structure_depth_exceeded")
    prohibited = ("raw", "token", "prompt", "latent", "secret", "hf_", "weight", "private", "image_bytes")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in prohibited): raise ValueError("forbidden_public_field")
            _reject_leak(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value: _reject_leak(item, depth=depth + 1)
    elif isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        local_path = (normalized.startswith("//") or normalized.startswith("~/") or "file://" in normalized or
                      bool(re.search(r"\b[a-z]:/", normalized)) or
                      (normalized.startswith("/") and not normalized.startswith("/content/drive/")))
        if any(word in normalized for word in ("hf_", "hf token", "secret", "prompt", "latent")) or any(pattern.search(normalized) for pattern in _VALUE_LEAKS) or local_path:
            raise ValueError("forbidden_public_value")


def _validate_d01_source(source_root: Path) -> dict[str, Any]:
    if source_root.is_symlink() or not source_root.is_dir(): raise ValueError("invalid_source_root")
    files = [item for item in source_root.rglob("*") if item.is_file()]
    if any(item.is_symlink() for item in source_root.rglob("*")) or {item.relative_to(source_root).as_posix() for item in files} != {"receipt.json", "manifest.json", "terminal.json"}:
        raise ValueError("source_file_roster_mismatch")
    if sum(item.stat().st_size for item in files) > MAX_ROOT_BYTES * 2 + MAX_CONTROL_BYTES: raise ValueError("source_bound_exceeded")
    receipt = _read_json(source_root / "receipt.json", MAX_ROOT_BYTES)
    manifest = _read_json(source_root / "manifest.json", MAX_ROOT_BYTES)
    terminal = _read_json(source_root / "terminal.json", MAX_CONTROL_BYTES)
    _reject_leak(receipt); _reject_leak(manifest); _reject_leak(terminal)
    if (receipt.get("run_id"), receipt.get("protocol"), receipt.get("d01_status"), receipt.get("science_denominator"), receipt.get("execution_identity", {}).get("commit")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_STATUS, 0, SOURCE_EXACT):
        raise ValueError("source_receipt_identity_mismatch")
    if receipt.get("selected_layer_paths") != SOURCE_SELECTED_PATHS: raise ValueError("source_selected_layer_mismatch")
    if (manifest.get("run_id"), manifest.get("protocol"), manifest.get("execution_exact"), manifest.get("d01_status")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_EXACT, SOURCE_STATUS):
        raise ValueError("source_manifest_identity_mismatch")
    if (terminal.get("run_id"), terminal.get("d01_status"), terminal.get("science_denominator")) != (SOURCE_RUN_ID, SOURCE_STATUS, 0):
        raise ValueError("source_terminal_identity_mismatch")
    return {"run_id": SOURCE_RUN_ID, "execution_exact": SOURCE_EXACT, "protocol": SOURCE_PROTOCOL,
            "status": SOURCE_STATUS, "selected_layer_paths": SOURCE_SELECTED_PATHS, "science_denominator": 0}


def _reference(reference_id: str) -> Image.Image:
    """New deterministic asymmetric RGB designs; not the D0 recipes or seeds."""
    image = Image.new("RGB", (512, 512), (31, 17, 59)); draw = ImageDraw.Draw(image)
    if reference_id == "confirmation_a":
        for x in range(7, 512, 29): draw.line((x, 0, (x * 5) % 512, 511), fill=((x * 11) % 256, 73, 231), width=3)
        draw.polygon(((71, 64), (257, 102), (118, 406), (39, 261)), fill=(238, 187, 39)); draw.ellipse((294, 208, 465, 458), fill=(57, 203, 135))
    elif reference_id == "confirmation_b":
        for y in range(9, 512, 31): draw.arc((25, y - 68, 486, y + 161), 17, 266, fill=(237, (y * 7) % 256, 91), width=4)
        draw.rectangle((63, 279, 226, 469), fill=(72, 144, 235)); draw.polygon(((312, 47), (475, 138), (412, 341), (279, 217)), fill=(219, 74, 157))
    else: raise ValueError("unknown_confirmation_reference")
    return image


def _similarity_h() -> np.ndarray:
    angle = np.deg2rad(-9.0); scale = .86; c, s = np.cos(angle) * scale, np.sin(angle) * scale
    center, translation = np.array([256., 256.]), np.array([-19., 14.])
    linear = np.array([[c, -s], [s, c]])
    offset = center + translation - linear @ center
    return np.array([[c, -s, offset[0]], [s, c, offset[1]], [0., 0., 1.]])


def _pillow_inverse_affine(h: np.ndarray) -> tuple[float, float, float, float, float, float]:
    inverse = np.linalg.inv(h); linear, offset = inverse[:2, :2], inverse[:2, 2]
    index_offset = linear @ np.array([.5, .5]) + offset - .5
    return (float(linear[0, 0]), float(linear[0, 1]), float(index_offset[0]), float(linear[1, 0]), float(linear[1, 1]), float(index_offset[1]))


def _attack(image: Image.Image, label: str) -> tuple[Image.Image, list[list[float]]]:
    if label == "identity": return image.copy(), np.eye(3).tolist()
    if label == "d4": return image.transpose(Image.Transpose.ROTATE_180), [[-1., 0., 512.], [0., -1., 512.], [0., 0., 1.]]
    if label == "similarity":
        h = _similarity_h()
        return image.transform((512, 512), Image.Transform.AFFINE, _pillow_inverse_affine(h), resample=Image.Resampling.BICUBIC), h.tolist()
    if label == "crop_rescale":
        left, top, right, bottom = 40, 56, 480, 472; sx, sy = 512 / (right - left), 512 / (bottom - top)
        return image.crop((left, top, right, bottom)).resize((512, 512), Image.Resampling.BICUBIC), [[sx, 0., -left * sx], [0., sy, -top * sy], [0., 0., 1.]]
    raise ValueError("unknown_confirmation_transform")


def build_fixed_plan() -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for reference_id in REFERENCE_IDS:
        for index, label in enumerate(TRANSFORMS):
            _, matched = _attack(_reference(reference_id), label)
            _, shuffled = _attack(_reference(reference_id), TRANSFORMS[(index + 1) % len(TRANSFORMS)])
            pairs.append({"reference_id": reference_id, "pair_id": f"{reference_id}-{label}", "transform_label": label,
                          "matched_h": matched, "shuffled_h": shuffled, "resampler": "PIL.Image.Resampling.BICUBIC"})
    return {"schema": "geometry-v1-qk-d1-independent-confirmation-plan-v1", "protocol": PROTOCOL,
            "reference_recipe_ids": {"confirmation_a": "d1-procedural-seed-1701", "confirmation_b": "d1-procedural-seed-2718"},
            "public_observation_seed": 41, "attack_parameters": {"d4": "rotate_180", "similarity": {"degrees": -9.0, "scale": .86, "translation": [-19, 14]}, "crop_rescale": [40, 56, 480, 472]},
            "attention_layer_paths": list(ATTENTION_LAYER_PATHS), "pairs": pairs, "declared_unit_count": UNIT_COUNT}


def _null(pipeline: Any) -> tuple[torch.Tensor, torch.Tensor]:
    result = pipeline.encode_prompt(prompt="", prompt_2="", prompt_3="", do_classifier_free_guidance=False)
    if not isinstance(result, (tuple, list)) or len(result) != 4 or not isinstance(result[0], torch.Tensor) or not isinstance(result[2], torch.Tensor): raise ValueError("invalid_null_conditioning")
    return result[0].detach(), result[2].detach()


def _spec(pipeline: Any) -> SD35QKObservationSpec:
    hidden, pooled = _null(pipeline)
    return SD35QKObservationSpec(MODEL_ID, getattr(pipeline, "_commit_hash", None), ATTENTION_LAYER_PATHS, 20, 7, 41, (8, 8), hidden, pooled)


def _grid_h(h_rgb: Any, reference_grid: Any, attacked_grid: Any) -> np.ndarray:
    def grid(value: Any, name: str) -> tuple[int, int]:
        if not isinstance(value, tuple) or len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value): raise ValueError(f"invalid_{name}_grid")
        return value
    rr, rc = grid(reference_grid, "reference"); ar, ac = grid(attacked_grid, "attacked")
    h = np.asarray(h_rgb, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all(): raise ValueError("invalid_rgb_h")
    converted = np.diag((ac / 512., ar / 512., 1.)) @ h @ np.linalg.inv(np.diag((rc / 512., rr / 512., 1.)))
    if not np.isfinite(converted).all(): raise ValueError("invalid_grid_h")
    return converted


def _failure(pair: Mapping[str, Any], path: str, kind: str, control: str, reason: str) -> dict[str, Any]:
    return {"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path,
            "reference_grid": None, "attacked_grid": None, "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": reason,
            "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}


def _layer(observation: SD35QKObservation, path: str) -> Any:
    for layer in observation.layers:
        if layer.layer_path == path: return layer
    raise ValueError("fixed_layer_not_observed")


def _unit(pair: Mapping[str, Any], reference: SD35QKObservation, attacked: SD35QKObservation, path: str, kind: str, control: str) -> dict[str, Any]:
    reference_layer, attacked_layer = _layer(reference, path), _layer(attacked, path); name = "query" if kind == "q" else "key"
    return HARNESS.evaluate_unit({"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path,
                                  "reference_descriptors": getattr(reference_layer, name).detach().cpu().numpy(), "attacked_descriptors": getattr(attacked_layer, name).detach().cpu().numpy(),
                                  "reference_source_grid": reference_layer.source_grid, "attacked_source_grid": attacked_layer.source_grid,
                                  "reference_sample_indices": reference_layer.sample_indices.detach().cpu().numpy(), "attacked_sample_indices": attacked_layer.sample_indices.detach().cpu().numpy(),
                                  "H_reference_to_attacked": _grid_h(pair[control], reference_layer.source_grid, attacked_layer.source_grid)})


def _expand_failures(plan: Mapping[str, Any], reason: str) -> tuple[dict[str, Any], ...]:
    return tuple(_failure(pair, path, kind, control, reason) for pair in plan["pairs"] for path in ATTENTION_LAYER_PATHS for kind in KINDS for control in CONTROLS)


def _finite_rank(value: Any) -> float | None:
    if value is None: return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)): raise ValueError("invalid_true_match_rank")
    return float(value)


def _direction_statistics(units: Sequence[Mapping[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    stats: list[dict[str, Any]] = []
    for path in ATTENTION_LAYER_PATHS:
        for kind in KINDS:
            pair_medians: list[float] = []; pair_audit: list[dict[str, Any]] = []; valid = True
            for pair in (f"{reference}-{transform}" for reference in REFERENCE_IDS for transform in TRANSFORMS):
                matched = [u for u in units if (u["pair_id"], u["layer_path"], u["descriptor_kind"], u["control_label"]) == (pair, path, kind, "matched_h")]
                shuffled = [u for u in units if (u["pair_id"], u["layer_path"], u["descriptor_kind"], u["control_label"]) == (pair, path, kind, "shuffled_h")]
                if len(matched) != 1 or len(shuffled) != 1 or matched[0]["status"] != "calculated" or shuffled[0]["status"] != "calculated":
                    valid = False; pair_audit.append({"pair_id": pair, "common_finite_count": 0, "pair_median": None}); continue
                left, right = matched[0]["true_match_ranks"], shuffled[0]["true_match_ranks"]
                if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
                    valid = False; pair_audit.append({"pair_id": pair, "common_finite_count": 0, "pair_median": None}); continue
                differences = [a - b for a, b in zip((_finite_rank(v) for v in left), (_finite_rank(v) for v in right)) if a is not None and b is not None]
                if not differences:
                    valid = False; pair_audit.append({"pair_id": pair, "common_finite_count": 0, "pair_median": None}); continue
                value = float(median(differences)); pair_medians.append(value); pair_audit.append({"pair_id": pair, "common_finite_count": len(differences), "pair_median": value})
            aggregate = float(median(pair_medians)) if valid and len(pair_medians) == 8 else None
            stats.append({"layer_path": path, "descriptor_kind": kind, "pair_statistics": pair_audit, "equal_weight_pair_median": aggregate, "strictly_negative": aggregate is not None and aggregate < 0.0})
    return all(item["strictly_negative"] for item in stats), stats


def run_d1(*, expected_exact: str, repo_root: Path, source_root: Path, hf_token: str, loader: Callable[..., Any] = load_sd35_pipeline, observer: Callable[..., SD35QKObservation] = observe_sd35_image_qk) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    source = _validate_d01_source(source_root)
    plan = build_fixed_plan(); plan_bytes = _json(plan, MAX_ROOT_BYTES); exact = _exact(expected_exact, repo_root)
    runtime: dict[str, Any] = {}; status, failure_point, reason, pipeline = "D1_STOPPED", "model_load", "model_or_topology_unavailable", None
    try:
        pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
        if hasattr(pipeline, "to"): pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        spec = _spec(pipeline); runtime = {"pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}", "resolved_public_revision": getattr(pipeline, "_commit_hash", None)}
        status, failure_point, reason = "D1_UNRESOLVED", None, None
    except BaseException: spec = None
    units: list[dict[str, Any]] = []; global_reason = None
    for reference_id in REFERENCE_IDS:
        reference, reference_reason = None, None
        if spec is not None:
            try: reference = observer(_reference(reference_id), pipeline=pipeline, spec=spec)
            except BaseException as error:
                reference_reason, failure_point = "reference_observation_failed", "image_observation"
                if getattr(error, "geometry_failure_point", None) == "transformer_call": global_reason = "global_transformer_failure"
        for pair in (item for item in plan["pairs"] if item["reference_id"] == reference_id):
            attacked, pair_reason = None, reason if spec is None else None
            if spec is not None:
                try: attacked = observer(_attack(_reference(reference_id), pair["transform_label"])[0], pipeline=pipeline, spec=spec)
                except BaseException as error:
                    pair_reason, failure_point = "attacked_observation_failed", "image_observation"
                    if getattr(error, "geometry_failure_point", None) == "transformer_call": global_reason = "global_transformer_failure"
                if reference_reason is not None: pair_reason = reference_reason
            for path in ATTENTION_LAYER_PATHS:
                for kind in KINDS:
                    for control in CONTROLS:
                        try: units.append(_failure(pair, path, kind, control, pair_reason) if pair_reason else _unit(pair, reference, attacked, path, kind, control))
                        except (AttributeError, KeyError, TypeError, ValueError): units.append(_failure(pair, path, kind, control, "layer_observation_or_calculation_failed"))
        del reference
    if len(units) != UNIT_COUNT: raise RuntimeError("d1_fixed_unit_expansion_mismatch")
    if global_reason is not None: units = list(_expand_failures(plan, global_reason)); status, failure_point = "D1_STOPPED", "image_observation"
    if spec is None: units = list(_expand_failures(plan, reason))
    if status != "D1_STOPPED":
        confirmed, statistics = _direction_statistics(units)
        status = "D1_CANDIDATES_CONFIRMED" if confirmed else "D1_UNRESOLVED"
    else: statistics = []
    summary = {"schema": SCHEMA, "protocol": PROTOCOL, "run_id": f"geometry-v1-qk-d1-{exact[:12]}", "execution_identity": {"commit": exact},
               "source_d01_artifact_identity": source, "plan_digest": _sha(plan_bytes), "roster_digest": _sha(_json([u["pair_id"] + ":" + u["layer_path"] + ":" + u["descriptor_kind"] + ":" + u["control_label"] for u in units], MAX_ROOT_BYTES)),
               "fixed_layer_paths": list(ATTENTION_LAYER_PATHS), "direction_statistics": statistics, "d1_status": status, "science_denominator": 0,
               "declared_unit_count": UNIT_COUNT, "calculated_unit_count": sum(u["status"] == "calculated" for u in units), "failed_unit_count": sum(u["status"] == "failed" for u in units),
               "operational_status": "complete" if failure_point is None else "failure", "operational_failure_point": failure_point, "runtime": runtime, "artifact_status": "unavailable"}
    return summary, tuple(units)


def _validate_public_unit(unit: Mapping[str, Any]) -> None:
    if not isinstance(unit, dict) or frozenset(unit) != UNIT_FIELDS: raise ValueError("invalid_public_unit_fields")
    _reject_leak(unit)


def _package(root: Path, summary: dict[str, Any], units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if root.exists(): raise FileExistsError("output_root_must_be_create_only")
    root.mkdir(parents=True); shard_dir = root / "layers"; shard_dir.mkdir(); shards = []
    for index, path in enumerate(ATTENTION_LAYER_PATHS):
        layer_units = [unit for unit in units if unit["layer_path"] == path]
        if len(layer_units) != 32: raise ValueError("layer_shard_count_mismatch")
        raw = []
        for unit in layer_units:
            _validate_public_unit(unit); raw.append(_json(unit, MAX_UNIT_BYTES))
        if sum(map(len, raw)) > MAX_LAYER_UNIT_BYTES: raise ValueError("layer_unit_bound_exceeded")
        target = shard_dir / f"{index:02d}.zip"
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for ordinal, data in enumerate(raw): archive.writestr(f"{ordinal:02d}.json", data)
        if target.stat().st_size > MAX_LAYER_ZIP_BYTES: raise ValueError("layer_zip_bound_exceeded")
        shards.append({"layer_path": path, "filename": f"layers/{index:02d}.zip", "unit_count": 32, "bytes": target.stat().st_size})
    summary["artifact_status"] = "complete"; summary["layer_shards"] = shards
    _write(root / "receipt.json", _json(summary, MAX_ROOT_BYTES))
    _write(root / "manifest.json", _json({"run_id": summary["run_id"], "protocol": PROTOCOL, "execution_exact": summary["execution_identity"]["commit"], "source_d01_artifact_identity": summary["source_d01_artifact_identity"], "d1_status": summary["d1_status"], "unit_count": UNIT_COUNT, "layer_shards": shards}, MAX_ROOT_BYTES))
    _write(root / "terminal.json", _json({"run_id": summary["run_id"], "d1_status": summary["d1_status"], "science_denominator": 0, "runner_execution_identity": summary["execution_identity"], "source_d01_artifact_identity": summary["source_d01_artifact_identity"]}, MAX_CONTROL_BYTES))
    if sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) > MAX_RUN_BYTES: raise ValueError("persistent_run_bound_exceeded")
    return {"artifact_status": "complete", "receipt_filename": "receipt.json", "manifest_filename": "manifest.json"}


def _emit(fd: int, prefix: str, value: Mapping[str, Any]) -> None:
    line = prefix.encode("ascii") + _json(value, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES: raise ValueError("control_bound_exceeded")
    os.write(fd, line)


def _public_error_class(error: BaseException) -> str:
    if isinstance(error, (FileExistsError, FileNotFoundError, PermissionError, OSError)): return "filesystem_error"
    if isinstance(error, (ValueError, TypeError, json.JSONDecodeError, zipfile.BadZipFile)): return "validation_error"
    if isinstance(error, subprocess.SubprocessError): return "subprocess_error"
    if isinstance(error, RuntimeError): return "runtime_error"
    return "unexpected_error"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", required=True); parser.add_argument("--expected-exact", required=True); parser.add_argument("--source-root", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv); stage = "source_validation"; run_id = f"geometry-v1-qk-d1-{args.expected_exact[:12]}"
    try:
        summary, units = run_d1(expected_exact=args.expected_exact, repo_root=Path(args.repo_root), source_root=Path(args.source_root), hf_token=os.environ.get("HF_TOKEN", ""))
        stage = "artifact_packaging"; package = _package(Path(args.output_root), summary, units); stage = "control_channel"
        _emit(args.control_fd, SUCCESS_PREFIX, {"status": "success", "run_id": summary["run_id"], "d1_status": summary["d1_status"], "science_denominator": 0, "fixed_layer_paths": list(ATTENTION_LAYER_PATHS), **package})
        return 0
    except BaseException as error:
        if stage == "control_channel": return 1
        try: _emit(args.control_fd, FAILURE_PREFIX, {"status": "failure", "run_id": run_id, "failure_point": stage, "error_class": _public_error_class(error), "artifact_status": "unavailable"})
        except BaseException: pass
        return 1


if __name__ == "__main__": raise SystemExit(_main())
