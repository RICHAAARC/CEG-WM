from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "geometry-v2"
ATTACKS = ("identity", "rotate90", "similarity", "crop_rescale")
FIELDS = {
    "attack", "extractor_confidence", "mean_corner_error", "reliability_score",
    "reliable", "seed", "status", "support", "truth_h_finite",
}


def _json(path: str) -> dict:
    return json.loads((EVIDENCE / path).read_text(encoding="utf-8"))


def _records() -> list[dict]:
    lines = (EVIDENCE / "n0" / "metrics_public.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 128 and all(line == line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _walk_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def test_identity_provenance_and_exact_public_metrics_copy() -> None:
    index = _json("index.json")
    provenance = _json("n0/provenance.json")
    metrics_path = EVIDENCE / "n0" / "metrics_public.jsonl"

    assert index["source_branch_final_exact"] == "bc147f8985d5e54477d0bbd47f7d44a73f70a6e6"
    assert index["execution_exact"] == "d82efc292db8a16f60d272635e577a4186ed866a"
    assert index["method_identity"] == "geometry_v2_keyed_neural_corner_sync"
    assert index["run_status"] == "N0_UNRESOLVED"
    assert index["final_status"] == "OPERATIONAL_UNRESOLVED"
    assert index["science_denominator"] == 0
    assert index["merge_policy"] == "do_not_merge_into_Geometry-V2"
    canonical_metrics = metrics_path.read_bytes().replace(b"\r\n", b"\n")
    assert len(canonical_metrics) == 28698
    assert hashlib.sha256(canonical_metrics).hexdigest() == "d10ccb3463ac3c2b8a1df8716ac3e7188f3449ea0cdf37ba158b9d6c88b94cae"
    assert provenance["access"]["drive_writes"] == 0
    assert provenance["access"]["source_total_bytes"] == sum(item["bytes"] for item in provenance["files"])
    assert {item["role"] for item in provenance["files"]} == {"receipt", "manifest", "terminal", "metrics"}
    assert all(len(item["id"]) >= 20 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in provenance["files"])


def test_all_confirmation_records_and_statistics_recompute() -> None:
    rows = _records()
    receipt = _json("n0/receipt_summary.json")
    summary = _json("n0/comparison_summary.json")

    assert all(set(row) == FIELDS for row in rows)
    assert [(row["seed"], row["attack"]) for row in rows] == [
        (seed, attack) for seed in range(3000, 3032) for attack in ATTACKS
    ]
    assert all(row["status"] == "calculated" for row in rows)
    assert all(row["support"] == 1.0 and row["truth_h_finite"] is True for row in rows)
    assert not any(row["reliable"] for row in rows)

    errors = [row["mean_corner_error"] for row in rows]
    assert math.isclose(_percentile(errors, 0.5), receipt["observed_confirmation"]["median_corner_error"], abs_tol=1e-15)
    assert math.isclose(_percentile(errors, 0.95), receipt["observed_confirmation"]["p95_corner_error"], abs_tol=1e-15)
    assert summary["overall"]["record_count"] == 128
    assert summary["overall"]["failed_count"] == 0

    for attack in ATTACKS:
        group = [row for row in rows if row["attack"] == attack]
        public = summary["attacks"][attack]
        assert len(group) == public["count"] == 32
        assert math.isclose(sum(row["mean_corner_error"] for row in group) / 32, public["mean_corner_error_mean"], abs_tol=1e-15)
        assert math.isclose(_percentile([row["mean_corner_error"] for row in group], 0.5), public["mean_corner_error_median"], abs_tol=1e-15)
        assert math.isclose(_percentile([row["mean_corner_error"] for row in group], 0.95), public["mean_corner_error_p95"], abs_tol=1e-15)
        assert public["reliable_count"] == 0
        assert public["support_complete_count"] == public["truth_h_finite_count"] == 32


def test_gate_result_and_public_payload_ceiling() -> None:
    receipt = _json("n0/receipt_summary.json")
    summary = _json("n0/comparison_summary.json")
    provenance = _json("n0/provenance.json")

    assert receipt["gate_results"] == {
        "all_declared_units_calculated": "pass",
        "median_corner_error": "fail",
        "p95_corner_error": "fail",
        "reliable_fraction": "fail",
        "actual_residual_max": "pass",
    }
    assert receipt["route_decision"]["action"] == "stop_after_n0"
    assert receipt["route_decision"]["planned_additional_training_stages"] == 0
    assert summary["science_denominator"] == 0
    assert sum((EVIDENCE / path).stat().st_size for path in [
        "index.json", "n0/provenance.json", "n0/receipt_summary.json",
        "n0/metrics_public.jsonl", "n0/comparison_summary.json",
    ]) < 100_000

    for text in _walk_strings({"receipt": receipt, "summary": summary, "provenance": provenance}):
        normalized = text.lower().replace("\\", "/")
        assert not re.search(r"\bhf_[a-z0-9]{12,}", text, re.IGNORECASE)
        assert not re.search(r"\bbearer\s+[a-z0-9._-]+", text, re.IGNORECASE)
        assert not re.match(r"^[a-z]:/", normalized)
        assert not normalized.startswith("//")
        assert not normalized.startswith("file://")
        assert not normalized.startswith("/mnt/")
        assert not normalized.startswith("/home/")
        assert not (normalized.startswith("/content/") and not normalized.startswith("/content/drive/"))
