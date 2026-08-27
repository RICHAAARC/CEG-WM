"""D0 all-layer Q/K representation/equivariance discovery transport.

This is not a detector, a keyed anchor, or a scientific adjudicator.  It owns
the fixed procedural RGB roster, records bounded public derived records, and
can only freeze candidate layer paths for a future independently authorised C0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.sd35_qk_observation import SD35QKAllLayerObservation, SD35QKObservationSpec, observe_sd35_image_qk_sampled_all_layers

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
PROTOCOL = "geometry-v1-qk-d0-all-layer-discovery-v1"
PLAN_SCHEMA = "geometry-v1-qk-d0-all-layer-plan-v1"
SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_QK_D0 "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_QK_D0_FAILURE "
MAX_CONTROL_BYTES, MAX_UNIT_BYTES, MAX_LAYER_UNIT_BYTES = 1024, 16384, 524288
MAX_LAYER_ZIP_BYTES, MAX_TOTAL_UNIT_BYTES = 1048576, 12582912
MAX_ROOT_BYTES, MAX_RUN_BYTES, UNIT_COUNT = 262144, 50331648, 768
TRANSFORMS = ("identity", "d4", "similarity", "crop_rescale")
KINDS, CONTROLS = ("q", "k"), ("matched_h", "shuffled_h")
LAYER_RE = re.compile(r"transformer_blocks\.(\d+)\.attn\Z")

_hspec = importlib.util.spec_from_file_location("geometry_d0_harness", Path(__file__).with_name("run_geometry_v1_qk_equivariance_preflight.py"))
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


def _reference(reference_id: str) -> Image.Image:
    """Two visibly asymmetric procedural RGB references with no external input."""
    image = Image.new("RGB", (512, 512), (17, 29, 43)); draw = ImageDraw.Draw(image)
    if reference_id == "reference_a":
        for x in range(0, 512, 19): draw.line((x, 0, 511 - x // 2, 511), fill=((x * 3) % 256, 220, 41), width=5)
        draw.rectangle((53, 91, 211, 337), fill=(242, 63, 91)); draw.ellipse((278, 177, 459, 390), fill=(22, 127, 218))
    elif reference_id == "reference_b":
        for y in range(0, 512, 23): draw.line((0, y, 511, (y * 3) % 512), fill=(233, (y * 5) % 256, 66), width=4)
        draw.polygon(((54, 432), (177, 74), (294, 461)), fill=(47, 184, 144)); draw.rectangle((318, 52, 472, 244), fill=(194, 66, 211))
    else: raise ValueError("unknown_reference")
    return image


def _homography_similarity() -> np.ndarray:
    angle = np.deg2rad(12.0); scale = .90; c, s = np.cos(angle) * scale, np.sin(angle) * scale
    center = np.array([255.5, 255.5]); translation = np.array([16.0, -12.0])
    offset = center + translation - np.array([[c, -s], [s, c]]) @ center
    return np.array([[c, -s, offset[0]], [s, c, offset[1]], [0., 0., 1.]])


def _attack(image: Image.Image, label: str) -> tuple[Image.Image, list[list[float]]]:
    if label == "identity": return image.copy(), np.eye(3).tolist()
    if label == "d4":
        # Pillow transpose is an exact clockwise D4 action in the 512 source grid.
        return image.transpose(Image.Transpose.ROTATE_270), [[0., -1., 511.], [1., 0., 0.], [0., 0., 1.]]
    if label == "similarity":
        h = _homography_similarity(); inverse = np.linalg.inv(h)
        coeff = tuple(inverse[:2, :].reshape(-1).tolist())
        return image.transform((512, 512), Image.Transform.AFFINE, coeff, resample=Image.Resampling.BICUBIC), h.tolist()
    if label == "crop_rescale":
        # Source pixel centers map to enlarged crop coordinates.
        return image.crop((48, 32, 464, 448)).resize((512, 512), Image.Resampling.BICUBIC), [[512/416, 0., -48 * 512/416], [0., 512/416, -32 * 512/416], [0., 0., 1.]]
    raise ValueError("unknown_transform")


def build_fixed_plan() -> dict[str, Any]:
    pairs = []
    for ref in ("reference_a", "reference_b"):
        for index, transform in enumerate(TRANSFORMS):
            _, matched = _attack(_reference(ref), transform)
            _, shuffled = _attack(_reference(ref), TRANSFORMS[(index + 1) % len(TRANSFORMS)])
            pairs.append({"reference_id": ref, "pair_id": f"{ref}-{transform}", "transform_label": transform,
                          "matched_h": matched, "shuffled_h": shuffled, "resampler": "PIL.Image.Resampling.BICUBIC"})
    return {"schema": PLAN_SCHEMA, "protocol": PROTOCOL, "pairs": pairs,
            "attack_order": list(TRANSFORMS), "declared_unit_count": UNIT_COUNT}


def _discover(transformer: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    candidates, excluded = [], []
    for path, module in transformer.named_modules():
        match = LAYER_RE.fullmatch(path)
        if match:
            index = int(match.group(1)); q, k = getattr(module, "to_q", None), getattr(module, "to_k", None)
            heads = getattr(module, "heads", None)
            if isinstance(q, torch.nn.Module) and isinstance(k, torch.nn.Module) and isinstance(heads, int) and not isinstance(heads, bool) and heads > 0:
                candidates.append((index, path))
            else: excluded.append(path)
        elif any(token in path for token in ("attn2", "add_q_proj", "add_k_proj", "to_qkv", "context")):
            excluded.append(path)
    candidates.sort()
    paths = tuple(path for _, path in candidates)
    if [index for index, _ in candidates] != list(range(24)):
        raise ValueError("d0_required_24_layer_roster_unavailable")
    return paths, {"candidate_count": len(paths), "candidate_paths": list(paths), "excluded_or_recorded_paths": sorted(set(excluded))}


def _null(pipeline: Any) -> tuple[torch.Tensor, torch.Tensor]:
    result = pipeline.encode_prompt(prompt="", prompt_2="", prompt_3="", do_classifier_free_guidance=False)
    if not isinstance(result, (tuple, list)) or len(result) != 4 or not isinstance(result[0], torch.Tensor) or not isinstance(result[2], torch.Tensor):
        raise ValueError("invalid_null_conditioning")
    return result[0].detach(), result[2].detach()


def _spec(pipeline: Any, paths: tuple[str, ...]) -> SD35QKObservationSpec:
    hidden, pooled = _null(pipeline)
    return SD35QKObservationSpec(MODEL_ID, getattr(pipeline, "_commit_hash", None), paths, 20, 7, 0, (8, 8), hidden, pooled)


def _failure(pair: Mapping[str, Any], path: str, kind: str, control: str, reason: str) -> dict[str, Any]:
    return {"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path,
            "reference_grid": None, "attacked_grid": None, "input_identity": None, "h_identity": None, "status": "failed", "failure_reason": reason,
            "candidate_correspondences": [], "true_match_ranks": [], "coverage": None, "ambiguity_gaps": [], "fit_residual": None, "recovery_error": None}


def _layer(observation: SD35QKAllLayerObservation, path: str) -> Any:
    for layer in observation.layers:
        if layer.layer_path == path: return layer
    raise ValueError("layer_not_observed")


def _unit(pair: Mapping[str, Any], reference: SD35QKAllLayerObservation, attacked: SD35QKAllLayerObservation, path: str, kind: str, control: str) -> dict[str, Any]:
    r, a = _layer(reference, path), _layer(attacked, path); name = "query" if kind == "q" else "key"
    return HARNESS.evaluate_unit({"pair_id": pair["pair_id"], "transform_label": pair["transform_label"], "control_label": control, "descriptor_kind": kind, "layer_path": path,
        "reference_descriptors": getattr(r, name).numpy(), "attacked_descriptors": getattr(a, name).numpy(), "reference_source_grid": r.source_grid, "attacked_source_grid": a.source_grid,
        "reference_sample_indices": r.sample_indices.numpy(), "attacked_sample_indices": a.sample_indices.numpy(), "H_reference_to_attacked": pair[control]})


def _status_and_selection(units: Sequence[Mapping[str, Any]], paths: Sequence[str], plan: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    selected: list[str] = []
    for lower, upper in ((0, 8), (8, 16), (16, 24)):
        eligible = []
        for index, path in enumerate(paths[lower:upper], start=lower):
            records = [u for u in units if u["layer_path"] == path and u["control_label"] == "matched_h"]
            if len(records) != 16 or any(u["status"] != "calculated" for u in records): continue
            fields = ("recovery_error", "fit_residual")
            if any(not np.isfinite(float(u[field])) for u in records for field in fields): continue
            ranks = [value for u in records for value in u["true_match_ranks"]]
            gaps = [value for u in records for value in u["ambiguity_gaps"]]
            if not ranks or not gaps or not np.isfinite(ranks).all() or not np.isfinite(gaps).all(): continue
            eligible.append((float(np.median([u["recovery_error"] for u in records])), float(np.median(ranks)), float(np.median([u["fit_residual"] for u in records])), -float(np.median(gaps)), index, path))
        if not eligible: return "D0_UNRESOLVED", [], {"selection_rule_id": "d0-stratum-median-lexicographic-v1", "plan_digest": _sha(_json(plan, MAX_ROOT_BYTES))}
        selected.append(min(eligible)[-1])
    return "D0_CANDIDATES_FROZEN", selected, {"selection_rule_id": "d0-stratum-median-lexicographic-v1", "plan_digest": _sha(_json(plan, MAX_ROOT_BYTES)), "roster_digest": _sha(_json(list(paths), MAX_ROOT_BYTES)), "selected_layer_paths": selected}


def run_d0(*, expected_exact: str, repo_root: Path, hf_token: str, loader: Callable[..., Any] = load_sd35_pipeline, observer: Callable[..., SD35QKAllLayerObservation] = observe_sd35_image_qk_sampled_all_layers) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    plan, plan_bytes = build_fixed_plan(), None
    plan_bytes = _json(plan, MAX_ROOT_BYTES); exact = _exact(expected_exact, repo_root); run_id = f"geometry-v1-qk-d0-{exact[:12]}"
    paths: tuple[str, ...] = tuple(f"transformer_blocks.{i}.attn" for i in range(24)); units: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}; pipeline = None; status, failure_point = "D0_STOPPED", "model_load"
    try:
        pipeline = loader(MODEL_ID, torch_dtype=torch.float16, token=hf_token)
        if hasattr(pipeline, "to"): pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        paths, topology = _discover(pipeline.transformer)
        runtime = {"pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}", "resolved_public_revision": getattr(pipeline, "_commit_hash", None), "topology": topology}
        spec = _spec(pipeline, paths); status, failure_point = "D0_UNRESOLVED", None
    except BaseException:
        spec = None
    for ref in ("reference_a", "reference_b"):
        reference = None
        if spec is not None:
            try: reference = observer(_reference(ref), pipeline=pipeline, spec=spec)
            except BaseException: failure_point = "image_observation"
        for pair in (item for item in plan["pairs"] if item["reference_id"] == ref):
            attacked = None; reason = None
            if reference is None: reason = "runtime_not_observed"
            else:
                try: attacked = observer(_attack(_reference(ref), pair["transform_label"])[0], pipeline=pipeline, spec=spec)
                except BaseException: reason = "image_observation_failed"; failure_point = "image_observation"
            for path in paths:
                for kind in KINDS:
                    for control in CONTROLS:
                        try: units.append(_failure(pair, path, kind, control, reason) if reason else _unit(pair, reference, attacked, path, kind, control))
                        except (AttributeError, KeyError, TypeError, ValueError): units.append(_failure(pair, path, kind, control, "layer_observation_or_calculation_failed"))
        del reference
    if len(units) != UNIT_COUNT: raise RuntimeError("d0_fixed_unit_expansion_mismatch")
    if spec is not None:
        status, selected, selection = _status_and_selection(units, paths, plan)
    else: selected, selection = [], {"selection_rule_id": "d0-stratum-median-lexicographic-v1", "plan_digest": _sha(plan_bytes)}
    summary = {"schema": "geometry-v1-qk-d0-operational-v1", "protocol": PROTOCOL, "run_id": run_id, "execution_identity": {"commit": exact}, "plan_digest": _sha(plan_bytes), "operational_status": "complete" if failure_point is None else "failure", "d0_status": status, "science_denominator": 0, "declared_unit_count": UNIT_COUNT, "calculated_unit_count": sum(u["status"] == "calculated" for u in units), "failed_unit_count": sum(u["status"] == "failed" for u in units), "operational_failure_point": failure_point, "runtime": runtime, "selection": selection, "artifact_status": "unavailable"}
    return summary, tuple(units)


def _package(root: Path, summary: dict[str, Any], units: Sequence[Mapping[str, Any]], *, exact: str) -> dict[str, Any]:
    if root.exists(): raise FileExistsError("output_root_must_be_create_only")
    root.mkdir(parents=True); shard_dir = root / "layers"; shard_dir.mkdir()
    manifests, total = [], 0
    for index in range(24):
        layer_units = [unit for unit in units if unit["layer_path"] == f"transformer_blocks.{index}.attn"]
        if len(layer_units) != 32: raise RuntimeError("layer_shard_count_mismatch")
        raw = [_json(unit, MAX_UNIT_BYTES) for unit in layer_units]; used = sum(map(len, raw)); total += used
        if used > MAX_LAYER_UNIT_BYTES or total > MAX_TOTAL_UNIT_BYTES: raise ValueError("unit_bound_exceeded")
        name = f"layers/{index:02d}.zip"; target = root / name
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for ordinal, data in enumerate(raw): archive.writestr(f"{ordinal:02d}.json", data)
        if target.stat().st_size > MAX_LAYER_ZIP_BYTES: raise ValueError("layer_zip_bound_exceeded")
        manifests.append({"layer_path": layer_units[0]["layer_path"], "filename": name, "unit_count": 32, "bytes": target.stat().st_size})
    summary["artifact_status"] = "complete"; summary["layer_shards"] = manifests
    receipt = _json(summary, MAX_ROOT_BYTES); _write(root / "receipt.json", receipt)
    manifest = {"run_id": summary["run_id"], "protocol": PROTOCOL, "execution_exact": exact, "layer_shards": manifests, "unit_count": UNIT_COUNT}
    _write(root / "manifest.json", _json(manifest, MAX_ROOT_BYTES)); _write(root / "terminal.json", _json({"run_id": summary["run_id"], "d0_status": summary["d0_status"]}, 1024))
    if sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) > MAX_RUN_BYTES: raise ValueError("persistent_run_bound_exceeded")
    return {"receipt_filename": "receipt.json", "manifest_filename": "manifest.json", "artifact_status": "complete"}


def _emit(fd: int, prefix: str, value: Mapping[str, Any]) -> None:
    line = prefix.encode("ascii") + _json(value, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES: raise ValueError("control_bound_exceeded")
    os.write(fd, line)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", required=True); parser.add_argument("--expected-exact", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv); stage = "execution_identity"; summary = None; run_id = f"geometry-v1-qk-d0-{args.expected_exact[:12]}"
    try:
        summary, units = run_d0(expected_exact=args.expected_exact, repo_root=Path(args.repo_root), hf_token=os.environ.get("HF_TOKEN", ""))
        stage = "artifact_packaging"; package = _package(Path(args.output_root), summary, units, exact=args.expected_exact); stage = "control_channel"
        _emit(args.control_fd, SUCCESS_PREFIX, {"status": "success", "run_id": summary["run_id"], "d0_status": summary["d0_status"], "science_denominator": 0, "selected_layer_paths": summary["selection"].get("selected_layer_paths", []), **package})
        return 0
    except BaseException:
        if stage == "control_channel": return 1
        try: _emit(args.control_fd, FAILURE_PREFIX, {"status": "failure", "run_id": run_id, "failure_point": stage, "artifact_status": "unavailable"})
        except BaseException: pass
        return 1


if __name__ == "__main__": raise SystemExit(_main())
