"""D0.1 bounded post-D0, missingness-aware artifact selection.

This standard-library-only transport reads one immutable D0 public artifact.
It is not a model runner, a detector, a keyed anchor, or a scientific result.
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

PROTOCOL = "geometry-v1-qk-d01-artifact-selection-v1"
SCHEMA = "geometry-v1-qk-d01-artifact-selection-operational-v1"
SOURCE_RUN_ID = "geometry-v1-qk-d0-4732211beefb"
SOURCE_EXACT = "4732211beefbeface95cb842c117b9719e362f1a"
SOURCE_PROTOCOL = "geometry-v1-qk-d0-all-layer-discovery-v1"
SOURCE_PLAN_DIGEST = "96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8"
SUCCESS_PREFIX = "CEGWM_GEOMETRY_V1_QK_D01 "
FAILURE_PREFIX = "CEGWM_GEOMETRY_V1_QK_D01_FAILURE "
MAX_CONTROL_BYTES, MAX_ROOT_BYTES = 1024, 262144
MAX_UNIT_BYTES, MAX_LAYER_UNIT_BYTES, MAX_LAYER_ZIP_BYTES = 16384, 524288, 1048576
MAX_SOURCE_BYTES, UNIT_COUNT = 50331648, 768
PAIRS = tuple(f"{reference}-{transform}" for reference in ("reference_a", "reference_b")
              for transform in ("identity", "d4", "similarity", "crop_rescale"))
KINDS, CONTROLS = ("q", "k"), ("matched_h", "shuffled_h")
UNIT_FIELDS = frozenset(("pair_id", "transform_label", "control_label", "descriptor_kind", "layer_path",
                         "reference_grid", "attacked_grid", "input_identity", "h_identity", "status",
                         "failure_reason", "candidate_correspondences", "true_match_ranks", "coverage",
                         "ambiguity_gaps", "fit_residual", "recovery_error"))
LAYER_RE = re.compile(r"transformer_blocks\.(\d+)\.attn\Z")


def _json(value: Any, maximum: int) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > maximum:
        raise ValueError("bounded_json_exceeded")
    return data


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def _exact(expected: str, root: Path) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("invalid_expected_exact")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected or dirty:
        raise RuntimeError("execution_identity_mismatch")
    return actual


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid_{name}")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"invalid_{name}")
    return number


def _read_json(path: Path, maximum: int) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("invalid_bounded_source_file")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid_source_json") from error
    if not isinstance(value, dict):
        raise ValueError("invalid_source_json")
    return value


def _reject_leak(value: Any, *, field: str = "") -> None:
    """Reject forbidden public-field names and bounded string indicators."""
    prohibited = ("raw", "token", "prompt", "latent", "secret", "hf_", "weight", "private", "image_bytes")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in prohibited):
                raise ValueError("forbidden_public_field")
            _reject_leak(item, field=key)
    elif isinstance(value, list):
        for item in value:
            _reject_leak(item, field=field)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(word in lowered for word in ("hf_", "hf token", "secret", "prompt", "latent", "/content/", "/home/", "c:\\users\\", "\\\\")):
            raise ValueError("forbidden_public_value")


def _expected_unit(layer: int) -> list[tuple[str, str, str, str]]:
    return [(pair, kind, control, f"transformer_blocks.{layer}.attn")
            for pair in PAIRS for kind in KINDS for control in CONTROLS]


def _validate_unit(value: Any, *, layer: int, ordinal: int) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != UNIT_FIELDS:
        raise ValueError("invalid_public_unit_fields")
    _reject_leak(value)
    pair, kind, control, path = _expected_unit(layer)[ordinal]
    transform = pair.rsplit("-", 1)[1]
    if (value["pair_id"], value["transform_label"], value["descriptor_kind"], value["control_label"], value["layer_path"]) != (pair, transform, kind, control, path):
        raise ValueError("fixed_unit_roster_mismatch")
    if value["status"] != "calculated" or value["failure_reason"] is not None:
        raise ValueError("source_unit_not_calculated")
    if not isinstance(value["true_match_ranks"], list) or not isinstance(value["ambiguity_gaps"], list):
        raise ValueError("invalid_public_metric_list")
    _number(value["coverage"], "coverage", positive=True)
    _number(value["fit_residual"], "fit_residual")
    _number(value["recovery_error"], "recovery_error")
    ranks = value["true_match_ranks"]
    if any(rank is not None and (isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(float(rank))) for rank in ranks):
        raise ValueError("invalid_true_match_rank")
    if not value["ambiguity_gaps"]:
        raise ValueError("invalid_ambiguity_gaps")
    for gap in value["ambiguity_gaps"]:
        _number(gap, "ambiguity_gap")
    return value


def _validate_source(source_root: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("invalid_source_root")
    files = [item for item in source_root.rglob("*") if item.is_file()]
    if any(item.is_symlink() for item in source_root.rglob("*")):
        raise ValueError("source_symlink_not_allowed")
    if sum(item.stat().st_size for item in files) > MAX_SOURCE_BYTES:
        raise ValueError("source_run_bound_exceeded")
    expected_names = {"receipt.json", "manifest.json", "terminal.json"} | {f"layers/{index:02d}.zip" for index in range(24)}
    names = {item.relative_to(source_root).as_posix() for item in files}
    if names != expected_names:
        raise ValueError("source_file_roster_mismatch")
    receipt = _read_json(source_root / "receipt.json", MAX_ROOT_BYTES)
    manifest = _read_json(source_root / "manifest.json", MAX_ROOT_BYTES)
    terminal = _read_json(source_root / "terminal.json", MAX_CONTROL_BYTES)
    _reject_leak(receipt)
    _reject_leak(manifest)
    _reject_leak(terminal)
    if (receipt.get("run_id"), receipt.get("protocol"), receipt.get("plan_digest"), receipt.get("d0_status"),
            receipt.get("science_denominator"), receipt.get("declared_unit_count"), receipt.get("calculated_unit_count"),
            receipt.get("failed_unit_count"), receipt.get("artifact_status"), receipt.get("operational_status"),
            receipt.get("operational_failure_point")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_PLAN_DIGEST, "D0_UNRESOLVED", 0, 768, 768, 0, "complete", "complete", None):
        raise ValueError("source_receipt_identity_or_status_mismatch")
    if receipt.get("execution_identity", {}).get("commit") != SOURCE_EXACT:
        raise ValueError("source_receipt_exact_mismatch")
    if (manifest.get("run_id"), manifest.get("protocol"), manifest.get("execution_exact"), manifest.get("unit_count")) != (SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_EXACT, 768):
        raise ValueError("source_manifest_identity_mismatch")
    if (terminal.get("run_id"), terminal.get("d0_status")) != (SOURCE_RUN_ID, "D0_UNRESOLVED"):
        raise ValueError("source_terminal_identity_mismatch")
    shards = manifest.get("layer_shards")
    if not isinstance(shards, list) or len(shards) != 24 or receipt.get("layer_shards") != shards:
        raise ValueError("source_layer_manifest_mismatch")
    units: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        filename, path = f"layers/{index:02d}.zip", f"transformer_blocks.{index}.attn"
        if not isinstance(shard, dict) or (shard.get("filename"), shard.get("layer_path"), shard.get("unit_count")) != (filename, path, 32):
            raise ValueError("source_layer_roster_mismatch")
        target = source_root / filename
        if target.stat().st_size > MAX_LAYER_ZIP_BYTES or shard.get("bytes") != target.stat().st_size:
            raise ValueError("source_layer_bound_mismatch")
        try:
            with zipfile.ZipFile(target) as archive:
                infos = archive.infolist()
                if [info.filename for info in infos] != [f"{ordinal:02d}.json" for ordinal in range(32)] or any(info.is_dir() or info.file_size > MAX_UNIT_BYTES for info in infos):
                    raise ValueError("source_zip_member_roster_mismatch")
                raw = [archive.read(info) for info in infos]
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise ValueError("invalid_source_layer_zip") from error
        if sum(map(len, raw)) > MAX_LAYER_UNIT_BYTES:
            raise ValueError("source_layer_unit_bound_exceeded")
        for ordinal, data in enumerate(raw):
            if len(data) > MAX_UNIT_BYTES:
                raise ValueError("source_unit_bound_exceeded")
            try:
                unit = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("invalid_source_unit_json") from error
            units.append(_validate_unit(unit, layer=index, ordinal=ordinal))
    if len(units) != UNIT_COUNT:
        raise ValueError("source_unit_count_mismatch")
    return {"source_run_id": SOURCE_RUN_ID, "source_execution_exact": SOURCE_EXACT,
            "source_protocol": SOURCE_PROTOCOL, "source_plan_digest": SOURCE_PLAN_DIGEST,
            "source_roster_digest": hashlib.sha256(_json([item["layer_path"] for item in units], MAX_ROOT_BYTES)).hexdigest()}, tuple(units)


def _select(units: Sequence[Mapping[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    selected: list[str] = []
    for lower, upper in ((0, 8), (8, 16), (16, 24)):
        candidates: list[tuple[float, float, float, float, int, str]] = []
        for index in range(lower, upper):
            path = f"transformer_blocks.{index}.attn"
            records = [unit for unit in units if unit["layer_path"] == path and unit["control_label"] == "matched_h"]
            shuffled = [unit for unit in units if unit["layer_path"] == path and unit["control_label"] == "shuffled_h"]
            finite_ranks = [float(rank) for unit in records for rank in unit["true_match_ranks"] if rank is not None]
            null_rank_count = sum(rank is None for unit in records for rank in unit["true_match_ranks"])
            eligible = (len(records) == 16 and all(unit["status"] == "calculated" for unit in records) and
                        all(_number(unit["coverage"], "coverage", positive=True) > 0.0 for unit in records) and
                        all(_number(unit["recovery_error"], "recovery_error") == _number(unit["recovery_error"], "recovery_error") for unit in records) and
                        all(_number(unit["fit_residual"], "fit_residual") == _number(unit["fit_residual"], "fit_residual") for unit in records) and
                        all(any(rank is not None for rank in unit["true_match_ranks"]) for unit in records) and
                        all(unit["ambiguity_gaps"] for unit in records))
            entry = {"layer_path": path, "matched_record_count": len(records), "shuffled_record_count": len(shuffled),
                     "shuffled_calculated_count": sum(unit["status"] == "calculated" for unit in shuffled),
                     "finite_rank_count": len(finite_ranks), "null_rank_count": null_rank_count, "eligible": eligible}
            if eligible:
                recovery = float(median(_number(unit["recovery_error"], "recovery_error") for unit in records))
                rank = float(median(finite_ranks))
                residual = float(median(_number(unit["fit_residual"], "fit_residual") for unit in records))
                gaps = [float(gap) for unit in records for gap in unit["ambiguity_gaps"]]
                gap = float(median(gaps))
                entry["sort_key"] = [recovery, rank, residual, -gap, index]
                candidates.append((recovery, rank, residual, -gap, index, path))
            audit.append(entry)
        if not candidates:
            return "D01_UNRESOLVED", [], audit
        selected.append(min(candidates)[-1])
    return "D01_CANDIDATES_FROZEN", selected, audit


def run_d01(*, expected_exact: str, repo_root: Path, source_root: Path) -> dict[str, Any]:
    exact = _exact(expected_exact, repo_root)
    source, units = _validate_source(source_root)
    status, selected, audit = _select(units)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "run_id": f"geometry-v1-qk-d01-{exact[:12]}",
            "execution_identity": {"commit": exact}, "science_denominator": 0, "d01_status": status,
            "source": source, "declared_unit_count": UNIT_COUNT, "audited_unit_count": len(units),
            "selected_layer_paths": selected, "selection_rule_id": "d01-missingness-aware-stratum-median-lexicographic-v1",
            "layer_audit": audit, "artifact_status": "unavailable"}


def _package(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError("output_root_must_be_create_only")
    root.mkdir(parents=True)
    summary["artifact_status"] = "complete"
    receipt = _json(summary, MAX_ROOT_BYTES)
    manifest = _json({"run_id": summary["run_id"], "protocol": PROTOCOL,
                      "execution_exact": summary["execution_identity"]["commit"], "source": summary["source"],
                      "d01_status": summary["d01_status"], "unit_count": UNIT_COUNT}, MAX_ROOT_BYTES)
    terminal = _json({"run_id": summary["run_id"], "d01_status": summary["d01_status"],
                      "science_denominator": 0}, MAX_CONTROL_BYTES)
    _write(root / "receipt.json", receipt); _write(root / "manifest.json", manifest); _write(root / "terminal.json", terminal)
    if sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) > MAX_SOURCE_BYTES:
        raise ValueError("persistent_run_bound_exceeded")
    return {"receipt_filename": "receipt.json", "manifest_filename": "manifest.json", "artifact_status": "complete"}


def _emit(fd: int, prefix: str, value: Mapping[str, Any]) -> None:
    line = prefix.encode("ascii") + _json(value, MAX_CONTROL_BYTES - len(prefix) - 1) + b"\n"
    if len(line) > MAX_CONTROL_BYTES:
        raise ValueError("control_bound_exceeded")
    os.write(fd, line)


def _public_error_class(error: BaseException) -> str:
    if isinstance(error, (FileExistsError, FileNotFoundError, PermissionError, OSError)):
        return "filesystem_error"
    if isinstance(error, (ValueError, TypeError, json.JSONDecodeError, zipfile.BadZipFile)):
        return "validation_error"
    if isinstance(error, subprocess.SubprocessError):
        return "subprocess_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "unexpected_error"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True); parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv); stage = "source_validation"; run_id = f"geometry-v1-qk-d01-{args.expected_exact[:12]}"
    try:
        summary = run_d01(expected_exact=args.expected_exact, repo_root=Path(args.repo_root), source_root=Path(args.source_root))
        stage = "artifact_packaging"; package = _package(Path(args.output_root), summary); stage = "control_channel"
        _emit(args.control_fd, SUCCESS_PREFIX, {"status": "success", "run_id": summary["run_id"],
              "d01_status": summary["d01_status"], "science_denominator": 0,
              "selected_layer_paths": summary["selected_layer_paths"], **package})
        return 0
    except BaseException as error:
        if stage == "control_channel":
            return 1
        try:
            _emit(args.control_fd, FAILURE_PREFIX, {"status": "failure", "run_id": run_id,
                  "failure_point": stage, "error_class": _public_error_class(error), "artifact_status": "unavailable"})
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
