"""Artifact-only all-layer Q/K direction selection for Geometry-V1.

This CPU-only transport consumes one immutable D0 public artifact. It does not
load a model, observe an image, alter D0, or adjudicate a scientific result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

PROTOCOL = "geometry-v1-qk-direction-all-layer-selection-v1"
SCHEMA = "geometry-v1-qk-direction-all-layer-selection-operational-v1"
SOURCE_RUN_ID = "geometry-v1-qk-d0-4732211beefb"
SOURCE_EXACT = "4732211beefbeface95cb842c117b9719e362f1a"
SOURCE_PROTOCOL = "geometry-v1-qk-d0-all-layer-discovery-v1"
SOURCE_PLAN_DIGEST = "96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8"
SOURCE_ROSTER_DIGEST = "88850de32ae0783427f86d0a5c82c6272a30811931ca0f883f6888cf8b83ac9e"
SOURCE_STATUS = "D0_UNRESOLVED"
PAIRS = tuple(f"{reference}-{transform}" for reference in ("reference_a", "reference_b") for transform in ("identity", "d4", "similarity", "crop_rescale"))
TRANSFORMS, KINDS, CONTROLS = ("identity", "d4", "similarity", "crop_rescale"), ("q", "k"), ("matched_h", "shuffled_h")
UNIT_FIELDS = frozenset(("pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path", "reference_grid", "attacked_grid", "input_identity", "h_identity", "status", "failure_reason", "candidate_correspondences", "true_match_ranks", "coverage", "ambiguity_gaps", "fit_residual", "recovery_error"))
MAX_CONTROL_BYTES, MAX_ROOT_BYTES = 1024, 262144
MAX_UNIT_BYTES, MAX_LAYER_UNIT_BYTES, MAX_LAYER_ZIP_BYTES = 16384, 524288, 1048576
MAX_SOURCE_BYTES, UNIT_COUNT = 50331648, 768
SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_DIRECTION_ALL_LAYER "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_DIRECTION_ALL_LAYER_FAILURE "
_VALUE_LEAKS = (
    re.compile(r"\braw\s*(?:q\s*/\s*k|qk|query|key|token(?:\s+material)?)\b", re.I),
    re.compile(r"\b(?:hf[_ -]?token|access[_ -]?token|auth(?:entication)?[_ -]?token|api[_ -]?key|"
               r"bearer\s+[a-z0-9._-]+|token\s+(?:material|credential|secret|value|data)|"
               r"credential(?:s)?(?:\s+(?:material|value|data))?)\b", re.I),
)


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
    if depth > 64: raise ValueError("public_value_structure_depth_exceeded")
    prohibited = ("raw", "token", "prompt", "latent", "secret", "hf_", "weight", "private", "image_bytes")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in prohibited): raise ValueError("forbidden_public_field")
            _reject_leak(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value: _reject_leak(item, depth=depth + 1)
    elif isinstance(value, str):
        lowered = value.lower(); normalized = lowered.replace("\\", "/")
        forbidden_path = (normalized.startswith("//") or normalized.startswith("~/") or "file://" in normalized or bool(re.search(r"\b[a-z]:/", normalized)) or any(match.group(0) not in ("/content/drive",) and not match.group(0).startswith("/content/drive/") for match in re.finditer(r"(?<![:/a-z0-9._-])/[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*", normalized)))
        if any(word in lowered for word in ("hf_", "hf token", "secret", "prompt", "latent")) or any(pattern.search(lowered) for pattern in _VALUE_LEAKS) or forbidden_path: raise ValueError("forbidden_public_value")


def _expected_unit(layer: int) -> list[tuple[str, str, str, str]]:
    return [(pair, kind, control, f"transformer_blocks.{layer}.attn") for pair in PAIRS for kind in KINDS for control in CONTROLS]


def _validate_unit(value: Any, layer: int, ordinal: int) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != UNIT_FIELDS: raise ValueError("invalid_public_unit_fields")
    _reject_leak(value)
    pair, kind, control, path = _expected_unit(layer)[ordinal]
    if (value["pair_id"], value["transform_label"], value["descriptor_kind"], value["control_label"], value["layer_path"]) != (pair, pair.rsplit("-", 1)[1], kind, control, path): raise ValueError("fixed_unit_roster_mismatch")
    if value["status"] != "calculated" or value["failure_reason"] is not None or not isinstance(value["true_match_ranks"], list): raise ValueError("source_unit_status_or_rank_mismatch")
    for rank in value["true_match_ranks"]:
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(float(rank))): raise ValueError("invalid_true_match_rank")
    return value


def _validate_source(root: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if root.is_symlink() or not root.is_dir(): raise ValueError("invalid_source_root")
    files = [item for item in root.rglob("*") if item.is_file()]
    expected = {"receipt.json", "manifest.json", "terminal.json"} | {f"layers/{index:02d}.zip" for index in range(24)}
    if any(item.is_symlink() for item in root.rglob("*")) or {item.relative_to(root).as_posix() for item in files} != expected: raise ValueError("source_file_roster_mismatch")
    if sum(item.stat().st_size for item in files) > MAX_SOURCE_BYTES: raise ValueError("source_run_bound_exceeded")
    receipt, manifest, terminal = _read_json(root / "receipt.json", MAX_ROOT_BYTES), _read_json(root / "manifest.json", MAX_ROOT_BYTES), _read_json(root / "terminal.json", MAX_CONTROL_BYTES)
    _reject_leak(receipt); _reject_leak(manifest); _reject_leak(terminal)
    if (receipt.get("run_id"), receipt.get("protocol"), receipt.get("plan_digest"), receipt.get("d0_status"), receipt.get("science_denominator"), receipt.get("declared_unit_count"), receipt.get("calculated_unit_count"), receipt.get("failed_unit_count"), receipt.get("artifact_status"), receipt.get("operational_status"), receipt.get("operational_failure_point"), receipt.get("execution_identity", {}).get("commit")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_PLAN_DIGEST, SOURCE_STATUS, 0, 768, 768, 0, "complete", "complete", None, SOURCE_EXACT): raise ValueError("source_receipt_identity_or_status_mismatch")
    if (manifest.get("run_id"), manifest.get("protocol"), manifest.get("execution_exact"), manifest.get("unit_count")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_EXACT, 768): raise ValueError("source_manifest_identity_mismatch")
    if (terminal.get("run_id"), terminal.get("d0_status")) != (SOURCE_RUN_ID, SOURCE_STATUS): raise ValueError("source_terminal_identity_mismatch")
    shards = manifest.get("layer_shards")
    if not isinstance(shards, list) or receipt.get("layer_shards") != shards or len(shards) != 24: raise ValueError("source_layer_manifest_mismatch")
    units: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        filename, path = f"layers/{index:02d}.zip", f"transformer_blocks.{index}.attn"
        target = root / filename
        if not isinstance(shard, dict) or (shard.get("filename"), shard.get("layer_path"), shard.get("unit_count"), shard.get("bytes")) != (filename, path, 32, target.stat().st_size) or target.stat().st_size > MAX_LAYER_ZIP_BYTES: raise ValueError("source_layer_roster_or_bound_mismatch")
        try:
            with zipfile.ZipFile(target) as archive:
                infos = archive.infolist()
                if [info.filename for info in infos] != [f"{ordinal:02d}.json" for ordinal in range(32)] or any(info.is_dir() or info.file_size > MAX_UNIT_BYTES for info in infos): raise ValueError("source_zip_member_roster_mismatch")
                raw = [archive.read(info) for info in infos]
        except (OSError, RuntimeError, zipfile.BadZipFile) as error: raise ValueError("invalid_source_layer_zip") from error
        if sum(map(len, raw)) > MAX_LAYER_UNIT_BYTES: raise ValueError("source_layer_unit_bound_exceeded")
        for ordinal, data in enumerate(raw):
            try: units.append(_validate_unit(json.loads(data), index, ordinal))
            except (UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError("invalid_source_unit_json") from error
    if len(units) != UNIT_COUNT: raise ValueError("source_unit_count_mismatch")
    digest = _sha(_json([unit["layer_path"] for unit in units], MAX_ROOT_BYTES))
    if digest != SOURCE_ROSTER_DIGEST: raise ValueError("source_roster_digest_mismatch")
    return {"run_id": SOURCE_RUN_ID, "execution_exact": SOURCE_EXACT, "protocol": SOURCE_PROTOCOL, "plan_digest": SOURCE_PLAN_DIGEST, "roster_digest": digest, "status": SOURCE_STATUS, "science_denominator": 0}, tuple(units)


def _finite(value: Any) -> float | None:
    if value is None: return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)): raise ValueError("invalid_true_match_rank")
    return float(value)


def _pair_delta(matched: Mapping[str, Any], shuffled: Mapping[str, Any]) -> tuple[float | None, int]:
    left, right = matched["true_match_ranks"], shuffled["true_match_ranks"]
    if len(left) != len(right): return None, 0
    deltas = [left_value - right_value for left_value, right_value in zip((_finite(value) for value in left), (_finite(value) for value in right)) if left_value is not None and right_value is not None]
    return (float(median(deltas)), len(deltas)) if deltas else (None, 0)


def _selection(units: Sequence[Mapping[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    layer_stats: list[dict[str, Any]] = []; eligible: list[tuple[float, float, int, str]] = []
    for layer in range(24):
        path = f"transformer_blocks.{layer}.attn"; kind_stats: dict[str, dict[str, Any]] = {}
        for kind in KINDS:
            pair_audit = []
            for pair in PAIRS:
                matched = [unit for unit in units if (unit["pair_id"], unit["layer_path"], unit["descriptor_kind"], unit["control_label"]) == (pair, path, kind, "matched_h")]
                shuffled = [unit for unit in units if (unit["pair_id"], unit["layer_path"], unit["descriptor_kind"], unit["control_label"]) == (pair, path, kind, "shuffled_h")]
                if len(matched) != 1 or len(shuffled) != 1: raise ValueError("source_pair_roster_mismatch")
                pair_median, common_count = _pair_delta(matched[0], shuffled[0]); pair_audit.append({"pair_id": pair, "transform_label": pair.rsplit("-", 1)[1], "common_finite_count": common_count, "pair_median": pair_median})
            supported = all(item["pair_median"] is not None for item in pair_audit)
            statistic = float(median([item["pair_median"] for item in pair_audit])) if supported else None
            per_transform = []
            for transform in TRANSFORMS:
                records = [item for item in pair_audit if item["transform_label"] == transform]
                values = [item["pair_median"] for item in records if item["pair_median"] is not None]
                per_transform.append({"transform_label": transform, "two_reference_common_counts": [item["common_finite_count"] for item in records], "two_reference_medians": [item["pair_median"] for item in records], "two_reference_equal_weight_median": float(median(values)) if len(values) == 2 else None})
            kind_stats[kind] = {"statistic": statistic, "all_eight_pairs_supported": supported, "pair_audit": pair_audit, "per_transform_audit": per_transform}
        q_stat, k_stat = kind_stats["q"]["statistic"], kind_stats["k"]["statistic"]
        is_eligible = q_stat is not None and k_stat is not None and q_stat < 0.0 and k_stat < 0.0
        record = {"layer_path": path, "block_index": layer, "q_stat": q_stat, "k_stat": k_stat, "q_audit": kind_stats["q"], "k_audit": kind_stats["k"], "eligible": is_eligible}
        if is_eligible:
            selection_tuple = [max(q_stat, k_stat), float(median([q_stat, k_stat])), layer]
            record["selection_tuple"] = selection_tuple; eligible.append((selection_tuple[0], selection_tuple[1], layer, path))
        layer_stats.append(record)
    route_audit = {"per_transform": [], "route_level_transform_instability": False}
    for transform in TRANSFORMS:
        values = [entry[f"{kind}_audit"]["per_transform_audit"][TRANSFORMS.index(transform)]["two_reference_equal_weight_median"] for entry in layer_stats for kind in KINDS]
        finite = [value for value in values if value is not None and math.isfinite(float(value))]
        all_nonnegative = len(finite) == 48 and all(value >= 0.0 for value in finite)
        route_audit["per_transform"].append({"transform_label": transform, "finite_stat_count": len(finite), "nonnegative_stat_count": sum(value >= 0.0 for value in finite), "all_layer_nonnegative": all_nonnegative})
    route_audit["route_level_transform_instability"] = any(item["transform_label"] in ("d4", "crop_rescale") and item["all_layer_nonnegative"] for item in route_audit["per_transform"])
    selected = [entry[-1] for entry in sorted(eligible)[:2]] if len(eligible) >= 2 else []
    return ("DIRECTION_TWO_CANDIDATES_FROZEN" if selected else "DIRECTION_ALL_LAYER_UNRESOLVED"), selected, layer_stats, route_audit


def run_direction_selection(*, expected_exact: str, repo_root: Path, source_root: Path) -> dict[str, Any]:
    source, units = _validate_source(source_root); exact = _exact(expected_exact, repo_root)
    status, selected, layer_stats, route_audit = _selection(units)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "run_id": f"geometry-v1-qk-direction-all-layer-{exact[:12]}", "runner_execution_identity": {"commit": exact}, "source_d0_artifact_identity": source, "selection_rule_id": "direction-all-layer-paired-rank-k2-v1", "audited_unit_count": UNIT_COUNT, "declared_unit_count": UNIT_COUNT, "layer_statistics": layer_stats, "eligible_layer_paths": [entry["layer_path"] for entry in layer_stats if entry["eligible"]], "selected_layer_paths": selected, "route_audit": route_audit, "status": status, "science_denominator": 0, "artifact_status": "unavailable"}


def _package(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if root.exists(): raise FileExistsError("output_root_must_be_create_only")
    root.mkdir(parents=True); summary["artifact_status"] = "complete"
    _write(root / "receipt.json", _json(summary, MAX_ROOT_BYTES))
    _write(root / "manifest.json", _json({"run_id": summary["run_id"], "protocol": PROTOCOL, "runner_execution_exact": summary["runner_execution_identity"]["commit"], "source_d0_artifact_identity": summary["source_d0_artifact_identity"], "status": summary["status"], "unit_count": UNIT_COUNT}, MAX_ROOT_BYTES))
    _write(root / "terminal.json", _json({"run_id": summary["run_id"], "status": summary["status"], "selected_layer_paths": summary["selected_layer_paths"], "science_denominator": 0}, MAX_CONTROL_BYTES))
    if sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) > MAX_ROOT_BYTES * 3: raise ValueError("persistent_run_bound_exceeded")
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
    args = parser.parse_args(argv); stage = "source_validation"; run_id = f"geometry-v1-qk-direction-all-layer-{args.expected_exact[:12]}"
    try:
        summary = run_direction_selection(expected_exact=args.expected_exact, repo_root=Path(args.repo_root), source_root=Path(args.source_root))
        stage = "artifact_packaging"; package = _package(Path(args.output_root), summary); stage = "control_channel"
        _emit(args.control_fd, SUCCESS_PREFIX, {"status": "success", "run_id": summary["run_id"], "selection_status": summary["status"], "selected_layer_paths": summary["selected_layer_paths"], "science_denominator": 0, **package}); return 0
    except BaseException as error:
        if stage == "control_channel": return 1
        try: _emit(args.control_fd, FAILURE_PREFIX, {"status": "failure", "run_id": run_id, "failure_point": stage, "error_class": _public_error_class(error), "artifact_status": "unavailable"})
        except BaseException: pass
        return 1


if __name__ == "__main__": raise SystemExit(_main())
