#!/usr/bin/env python3
"""Verify immutable Geometry-V4 artifacts and rebuild the local evidence index.

The default path is pure stdlib and never runs a model.  ``--capture-cpu`` is
the only mode that imports the frozen Geometry-V4 implementation; it reruns the
fixed 4x5 synthetic CPU canary and writes a zero-formal-denominator snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

EVIDENCE_ROOT = Path(__file__).resolve().parents[1]
RAW = EVIDENCE_ROOT / "raw"
DERIVED = EVIDENCE_ROOT / "derived"
FROZEN_EXACT = "12488ad69bd6d2bf8ccc8d0c8d590cfa44bf372b"

G0_EXACT = "5b29a275151d436dbe1d51789cffe8e6908966b7"
G1_EXACT = "22db26de8ad83b125bbe0d58030f93a9554c8112"
G1R_ROUTES = (
    ("331e56539858d06c062ff4211abcb27c516fc180", "initial_g1r_blind_recovery"),
    ("7b8649ae9c775e52deac7daf964f8419c5ce50f2", "decoder_output_writer"),
    ("c5ed4f167a80591f8d0275fc1afdb662be0301ea", "keyed_phase_search"),
    ("a3877d6491308057547caf5993ec9bb1629e791c", "diffuse_luma_spread_anchor"),
    ("40261d26580e439d76ff93d344ac806a8e4744fc", "opponent_color_carrier"),
    ("c0a7f9b7bfdcf245e32460d919ed03800c5d000c", "sparse_gaussian_fiducial_atlas"),
)

DRIVE_SOURCES = {
    "g0": {
        "source_exact": G0_EXACT,
        "folder_url": "https://drive.google.com/drive/folders/1NB6ClFZVkkSBm4K2TTEOg-va8ArUkSYe",
        "records_file_id": "1R19VRT50LGPl9WNiIJdXrmhW_Z773uuj",
    },
    "g1": {
        "source_exact": G1_EXACT,
        "folder_url": "https://drive.google.com/drive/folders/1-rP0q_7t0IaYbw-aA6NJs-QUoeB9HLqv",
        "records_file_id": "1Y-wDIkee78GHpYXbZrRCnIRFbqkd_G3N",
    },
    "g1r": {
        "331e56539858d06c062ff4211abcb27c516fc180": ("1UyHv9jl-iYWRdayjJN24OXzmR3yiJF-q", "1BduwLpkQQJB6NL5S6SVDMvf9kXtKiYw-"),
        "7b8649ae9c775e52deac7daf964f8419c5ce50f2": ("1ytSv4pLnPGFSClYrmOxg7BXmv4lyY3oO", "1sCEAbfVwe0mmqifIONFq2S3vyKENtCbu"),
        "c5ed4f167a80591f8d0275fc1afdb662be0301ea": ("1ioaOotbhor6nO07Ue-gLQLlT0fViXCwl", "1eYXeHgq_w4EAoZhBPCGNQqEbOEkRBjr0"),
        "a3877d6491308057547caf5993ec9bb1629e791c": ("1bzwGEZgYp2lbLaF-C-sgPLPpr0WiOQbd", "1xE4qkE7Dcp02s_qIQYkusT3YjqiWgXd8"),
        "40261d26580e439d76ff93d344ac806a8e4744fc": ("1P1AJKx7ZQiNtCAOJZzGsgz6c8x8gTNUk", "1kVmitant9P3a6tAMUmZ2ihTVlOsNi6Lx"),
        "c0a7f9b7bfdcf245e32460d919ed03800c5d000c": ("1WIampO8S8yMN6dCWakqiYYBMo8DJXOZg", "1Gq5B27GRFosPceJBoRWz8DDOzMrD9rz0"),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")


def verify_sidecars() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for sidecar in sorted(RAW.rglob("*.sha256")):
        parts = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid sidecar: {sidecar}")
        expected, name = parts
        target = sidecar.parent / name.strip()
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"hash mismatch: {target}")
        results.append({"path": str(target.relative_to(EVIDENCE_ROOT)), "sha256": actual, "bytes": target.stat().st_size})
    return results


def _similarity(angle_deg: float, scale: float, tx: float = 0.0, ty: float = 0.0) -> tuple[float, ...]:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    centered = (c, -s, .5 - .5 * c + .5 * s, s, c, .5 - .5 * s - .5 * c, 0.0, 0.0, 1.0)
    # centered @ translation
    return (centered[0], centered[1], centered[0] * tx + centered[1] * ty + centered[2],
            centered[3], centered[4], centered[3] * tx + centered[4] * ty + centered[5], 0.0, 0.0, 1.0)


def _inverse_affine(h: Iterable[float]) -> tuple[float, ...]:
    a, b, tx, c, d, ty, _, _, _ = tuple(float(v) for v in h)
    det = a * d - b * c
    return (d / det, -b / det, (b * ty - d * tx) / det,
            -c / det, a / det, (c * tx - a * ty) / det, 0.0, 0.0, 1.0)


def _truth(attack: str) -> tuple[float, ...]:
    forward = {
        "identity": _similarity(0.0, 1.0),
        "rotation_5": _similarity(5.0, 1.0),
        "scale_0.9": _similarity(0.0, .9),
        "translation_0.08_0": _similarity(0.0, 1.0, .08, 0.0),
        "crop_0.9": _similarity(0.0, 1.0 / .9),
    }[attack]
    return _inverse_affine(forward)


def _project(h: Iterable[float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f, g, q, i = tuple(float(v) for v in h)
    z = g * x + q * y + i
    return ((a * x + b * y + c) / z, (d * x + e * y + f) / z)


def _geometry_error(h_hat: Iterable[float], attack: str) -> dict[str, float]:
    estimate, truth = tuple(float(v) for v in h_hat), _truth(attack)
    points = ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.5, .5))
    distances = []
    for point in points:
        p, t = _project(estimate, *point), _project(truth, *point)
        distances.append(math.hypot(p[0] - t[0], p[1] - t[1]) / math.sqrt(2.0))
    ea, es = math.degrees(math.atan2(estimate[3], estimate[0])), math.hypot(estimate[0], estimate[3])
    ta, ts = math.degrees(math.atan2(truth[3], truth[0])), math.hypot(truth[0], truth[3])
    return {
        "mapped_corner_error": max(distances[:4]),
        "center_reprojection_error": distances[4],
        "rotation_abs_error_degrees": abs((ea - ta + 90.0) % 180.0 - 90.0),
        "log_scale_abs_error": abs(math.log(es) - math.log(ts)),
    }


def _unsafe(status: str, metrics: Mapping[str, float]) -> bool:
    return status == "RELIABLE" and (
        metrics["mapped_corner_error"] > .02
        or metrics["center_reprojection_error"] > .02
        or metrics["rotation_abs_error_degrees"] > 2.0
        or metrics["log_scale_abs_error"] > .03
    )


def _candidate_truth(attack: str) -> tuple[float, float]:
    return {
        "identity": (0.0, 1.0),
        "rotation_5": (5.0, 1.0),
        "scale_0.9": (0.0, .9),
        "translation_0.08_0": (0.0, 1.0),
        "crop_0.9": (0.0, 1.0 / .9),
    }[attack]


def _top5_hit(record: Mapping[str, object]) -> bool:
    correct = record["arms"]["correct"]
    candidates = correct.get("engineering_diagnostics", {}).get("search_top_k", [])
    truth_angle, truth_scale = _candidate_truth(str(record["attack"]))
    for item in candidates:
        angle, scale = float(item["angle_degrees"]), float(item["scale"])
        angle_error = abs((angle - truth_angle + 90.0) % 180.0 - 90.0)
        if angle_error <= 2.0 and scale > 0.0 and abs(math.log(scale) - math.log(truth_scale)) <= .03:
            return True
    return False


def summarize_g0() -> dict[str, object]:
    data = _load(RAW / "g0" / G0_EXACT / "g0-records.json")
    records = data["records"]
    passed = sum(r["failure"] is None and r["final_rgb"]["passed"] for r in records)
    return {"stage": "G0", "source_exact": G0_EXACT, "units": len(records), "failures": sum(r["failure"] is not None for r in records), "final_rgb_observable": passed, "status": "PASS" if passed == len(records) else "GATE_FAILED"}


def summarize_g1() -> dict[str, object]:
    data = _load(RAW / "g1" / G1_EXACT / "g1-records.json")
    records = data["records"]
    source_observability = {
        int(record["seed"]): bool(record["failure"] is None and record["final_rgb"]["passed"])
        for record in records
    }
    result: dict[str, object] = {
        "stage": "G1",
        "source_exact": G1_EXACT,
        "units": len(records),
        "failures": sum(r["failure"] is not None for r in records),
        "source_observability_passed": sum(source_observability.values()),
        "sources": len(source_observability),
        "legacy_attacked_gate_passed": sum(r["failure"] is None and r["attacked_rgb"]["passed"] for r in records),
    }
    for key, arm_key in (("correct", "correct_key_geometry"), ("wrong", "wrong_key_geometry")):
        reliable = safe = unsafe = 0
        for record in records:
            arm = record["attacked_rgb"][arm_key]
            reliable += arm["status"] == "RELIABLE"
            if arm["status"] == "RELIABLE":
                is_unsafe = _unsafe(arm["status"], _geometry_error(arm["H_hat"], record["attack"]))
                unsafe += is_unsafe
                safe += not is_unsafe
        result[f"{key}_reliable"] = reliable
        result[f"{key}_safe_reliable"] = safe
        result[f"{key}_unsafe"] = unsafe
    result["status"] = "G1_METHOD_PARTIAL_NOT_PASSED"
    return result


def summarize_g1r(exact: str, route: str) -> dict[str, object]:
    base = RAW / "g1r" / exact
    data, recorded = _load(base / "g1r-development-records.json"), _load(base / "g1r-development-summary.json")
    records, sources = data["records"], data["sources"]
    complete = [r for r in records if r.get("failure") is None and isinstance(r.get("arms"), Mapping)]
    result: dict[str, object] = {
        "stage": "G1R_development",
        "route": route,
        "source_exact": exact,
        "records_sha256": _sha256(base / "g1r-development-records.json"),
        "units": len(records),
        "failures": sum(r.get("failure") is not None for r in records),
        "source_observability_passed": sum(s.get("failure") is None and bool((s.get("final_rgb") or {}).get("passed")) for s in sources),
        "correct_reliable": sum(r["arms"]["correct"]["status"] == "RELIABLE" for r in complete),
        "correct_safe_reliable": sum(r["arms"]["correct"]["status"] == "RELIABLE" and not r["arms"]["correct"]["unsafe"] for r in complete),
        "correct_unsafe": sum(bool(r["arms"]["correct"]["unsafe"]) for r in complete),
        "wrong_reliable": sum(r["arms"]["wrong"]["status"] == "RELIABLE" for r in complete),
        "wrong_harmless_reliable": sum(r["arms"]["wrong"]["status"] == "RELIABLE" and not r["arms"]["wrong"]["unsafe"] for r in complete),
        "wrong_unsafe": sum(bool(r["arms"]["wrong"]["unsafe"]) for r in complete),
        "negative_unsafe": sum(bool(r["arms"]["negative"]["unsafe"]) for r in complete),
    }
    supports, psrs = [], []
    top5 = {attack: 0 for attack in ("identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9")}
    for record in complete:
        diagnostics = record["arms"]["correct"].get("engineering_diagnostics", {})
        fit = diagnostics.get("selected_fit", {})
        if "support" in fit:
            supports.append(int(fit["support"]))
        if fit.get("translation_psr") is not None:
            psrs.append(float(fit["translation_psr"]))
        if diagnostics.get("search_top_k"):
            top5[str(record["attack"])] += _top5_hit(record)
    result["correct_rs_top5"] = sum(top5.values())
    result["correct_rs_top5_by_attack"] = top5
    result["selected_fit_support"] = {"min": min(supports) if supports else None, "median": median(supports) if supports else None, "max": max(supports) if supports else None}
    result["selected_translation_psr"] = {"min": min(psrs) if psrs else None, "median": median(psrs) if psrs else None, "max": max(psrs) if psrs else None}
    probes = [r.get("truth_probe") for r in complete if isinstance(r.get("truth_probe"), Mapping) and "failure" not in r["truth_probe"]]
    result["truth_probe"] = {
        "units": len(probes),
        "fit_valid": sum(bool(p.get("fit_at_truth", {}).get("valid")) for p in probes),
        "holdout_passed": sum(bool(p.get("holdout_at_truth", {}).get("passed")) for p in probes),
        "fit_support_max": max((int(p.get("fit_at_truth", {}).get("support", 0)) for p in probes), default=None),
        "truth_rs_translation_psr_max": max((float(p.get("search_at_truth", {}).get("translation_psr", 0.0)) for p in probes), default=None),
    }
    for field in ("correct_safe_reliable", "correct_unsafe", "wrong_unsafe", "negative_unsafe", "failures", "source_observability_passed", "units"):
        if result[field] != recorded[field]:
            raise ValueError(f"recorded G1R summary mismatch for {exact}: {field}")
    result["status"] = "G1R_METHOD_PARTIAL_NOT_PASSED"
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def capture_cpu() -> None:
    from experiments.geometry_v4_g1r_engine import run_cpu_canary, summarize_cpu_canary

    records = run_cpu_canary()
    payload = {"source_exact": FROZEN_EXACT, "stage": "V4-G1R_CPU_SYNTHETIC", "records": _jsonable(records)}
    summary = _jsonable(summarize_cpu_canary(records))
    target = RAW / "cpu" / FROZEN_EXACT
    _dump(target / "cpu-records.json", payload)
    _dump(target / "cpu-summary.json", summary)
    for name in ("cpu-records.json", "cpu-summary.json"):
        path = target / name
        (target / f"{name}.sha256").write_text(f"{_sha256(path)}  {name}\n", encoding="ascii")


def build_evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sidecars = verify_sidecars()
    routes = [summarize_g0(), summarize_g1()]
    routes.extend(summarize_g1r(exact, route) for exact, route in G1R_ROUTES)
    cpu_path = RAW / "cpu" / FROZEN_EXACT / "cpu-summary.json"
    if cpu_path.exists():
        cpu = _load(cpu_path)
        routes.append({"stage": "G1R_CPU_synthetic", "route": "balanced_bipolar_prn_microcode", "source_exact": FROZEN_EXACT, **cpu})
    ledger = {
        "schema": "geometry_v4_evidence_route_ledger_v1",
        "method": "geometry_v4_keyed_multiscale_sync_anchor_v1",
        "frozen_exact": FROZEN_EXACT,
        "routes": routes,
    }
    freeze = {
        "schema": "geometry_v4_freeze_decision_v1",
        "method": "geometry_v4_keyed_multiscale_sync_anchor_v1",
        "frozen_exact": FROZEN_EXACT,
        "status": "DECODER_OUTPUT_BASELINE_METHOD_PARTIAL",
        "formal_science_denominator": 0,
        "geometry_role": "coordinate_only_never_positive",
        "decision": "freeze_V4_and_do_not_continue_this_route_without_a_new_mechanism",
        "reasons": [
            "G0 passed 4/4 and proves only writer_to_final_RGB observability, not blind recovery.",
            "Legacy G1 passed its attacked gate only 2/20 and produced unsafe reliable homographies.",
            "All six real G1R development artifacts retained 0/20 correct safe RELIABLE recoveries.",
            "Real final-RGB source observability varied from 0/4 to 3/4 and never reached the frozen 4/4 requirement.",
            "The final synthetic CPU route remained CPU_METHOD_PARTIAL; synthetic evidence has formal denominator zero.",
            "Changing writer placement, phase search, spread, color carrier, sparsity, and balanced PRN microcode did not close both observability and blind safe-H recovery.",
        ],
        "reuse_rule": "future methods may recompute this public evidence locally but must not relabel it as a positive or scientific result",
        "do_not_merge_back": True,
        "do_not_rerun_frozen_gpu_development_seeds": True,
    }
    provenance = {
        "schema": "geometry_v4_evidence_provenance_v1",
        "source_branch": "Geometry-V4",
        "source_exact": FROZEN_EXACT,
        "evidence_branch": "Geometry-V4-Evidence",
        "drive_sources": _jsonable(DRIVE_SOURCES),
        "verified_sidecars": sidecars,
        "claim_ceiling": "engineering_and_method-development_evidence_only",
    }
    return ledger, freeze, provenance


def _indexable_files() -> list[Path]:
    return sorted(
        path
        for path in EVIDENCE_ROOT.rglob("*")
        if path.is_file()
        and path.name != "index.json"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )


def write_derived() -> None:
    ledger, freeze, provenance = build_evidence()
    _dump(DERIVED / "route_ledger.json", ledger)
    _dump(DERIVED / "freeze_decision.json", freeze)
    _dump(DERIVED / "provenance.json", provenance)
    indexed = []
    for path in _indexable_files():
        indexed.append({"path": str(path.relative_to(EVIDENCE_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    _dump(EVIDENCE_ROOT / "index.json", {"schema": "geometry_v4_evidence_index_v1", "frozen_exact": FROZEN_EXACT, "files": indexed})


def check_derived() -> None:
    ledger, freeze, provenance = build_evidence()
    expected = {"route_ledger.json": ledger, "freeze_decision.json": freeze, "provenance.json": provenance}
    for name, value in expected.items():
        if _load(DERIVED / name) != value:
            raise ValueError(f"derived evidence is stale: {name}")
    index = _load(EVIDENCE_ROOT / "index.json")
    indexed_paths = {entry["path"] for entry in index["files"]}
    actual_paths = {str(path.relative_to(EVIDENCE_ROOT)) for path in _indexable_files()}
    if indexed_paths != actual_paths:
        raise ValueError("index path set differs from evidence package")
    for entry in index["files"]:
        path = EVIDENCE_ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"index mismatch: {entry['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-cpu", action="store_true", help="rerun the frozen local 4x5 synthetic CPU canary")
    parser.add_argument("--write-derived", action="store_true", help="rewrite derived JSON and index")
    args = parser.parse_args()
    if args.capture_cpu:
        capture_cpu()
    if args.write_derived or args.capture_cpu:
        write_derived()
    else:
        check_derived()
    print(json.dumps({"status": "PASS", "frozen_exact": FROZEN_EXACT, "sidecars": len(verify_sidecars())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
