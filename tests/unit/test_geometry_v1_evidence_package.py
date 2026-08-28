from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "geometry-v1"
REQUIRED = {
    "stage", "protocol", "run", "layer", "kind", "control", "transform",
    "reference_or_pair", "metric", "value", "support", "count", "status",
    "source_sidecar_id",
}
STAGE_ORDER = {"D0": 0, "D0.1": 1, "D0.2": 2, "D1": 3, "D2": 4}
KIND_ORDER = {None: 0, "Q": 1, "K": 2}
CONTROL_ORDER = {None: 0, "aggregate": 1, "matched": 2, "shuffled": 3, "matched_minus_shuffled": 4}
TRANSFORM_ORDER = {None: 0, "all_transforms": 1, "identity": 2, "d4": 3, "similarity": 4, "crop_rescale": 5}


def _load_json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _records() -> list[dict]:
    lines = (EVIDENCE / "data" / "comparison_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines and all(line == line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def _block(layer: str | None) -> int:
    if layer is None:
        return -1
    match = re.fullmatch(r"transformer_blocks\.(\d+)\.attn", layer)
    assert match
    return int(match.group(1))


def _order(record: dict) -> tuple:
    return (
        STAGE_ORDER[record["stage"]],
        _block(record["layer"]),
        KIND_ORDER[record["kind"]],
        CONTROL_ORDER[record["control"]],
        TRANSFORM_ORDER[record["transform"]],
        record["reference_or_pair"] or "",
        record["metric"],
    )


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert key.lower() not in {"raw_q", "raw_k", "q_tensor", "k_tensor", "image_bytes", "prompt", "latent", "secret", "model_weights", "weights"}
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    elif isinstance(value, str):
        yield value


def test_package_identity_counts_and_deterministic_records() -> None:
    index = _load_json("index.json")
    provenance = _load_json("data/provenance.json")
    summary = _load_json("data/comparison_summary.json")
    records = _records()

    assert index["method_identity"] == "geometry_v1_natural_qk_geometry_observation"
    assert index["final_status"] == "OPERATIONAL_UNRESOLVED"
    assert index["science_denominator"] == 0
    assert index["merge_policy"] == "do_not_merge_into_Geometry-V1"
    assert len(records) == index["data_files"]["comparison_records"]["record_count"]
    assert all(set(record) == REQUIRED for record in records)
    assert [_order(record) for record in records] == sorted(_order(record) for record in records)
    assert len(summary["all_24_layers"]) == 24
    assert [row["block_index"] for row in summary["all_24_layers"]] == list(range(24))
    assert [row["block_index"] for row in summary["focus_layers"]] == [6, 13, 18, 14, 23]
    assert provenance["access"]["bounded_json_objects_parsed"] == 15
    assert provenance["access"]["drive_writes"] == 0
    assert provenance["access"]["zip_content_reads"] == 0
    assert all(sidecar["source_content_sha256"] is None for stage in provenance["stages"] for sidecar in stage["sidecars"])
    assert sum(len(stage["layer_shards"]) for stage in provenance["stages"]) == 29
    for metadata in index["data_files"].values():
        assert (EVIDENCE / metadata["path"]).stat().st_size == metadata["bytes"]


def test_cross_stage_public_facts_are_retained_without_promotion() -> None:
    summary = _load_json("data/comparison_summary.json")
    table = {row["block_index"]: row for row in summary["all_24_layers"]}

    assert table[6]["d01"]["selected"] is True
    assert table[13]["d01"]["selected"] is True
    assert table[18]["d01"]["selected"] is True
    assert table[14]["d02"]["selected"] is True
    assert table[23]["d02"]["selected"] is True
    assert table[6]["d1"] == {"K": 0, "Q": 0}
    assert table[13]["d1"] == {"K": -0.75, "Q": -1}
    assert table[18]["d1"] == {"K": -0.25, "Q": -0.75}
    assert table[14]["d2"] == {"K": -3.5, "Q": -3.5}
    assert table[23]["d2"] == {"K": 0, "Q": -3.5}
    assert summary["d2_gate_facts"]["route_level_transform_instability"] is True
    assert summary["science_denominator"] == 0
    assert "coverage" in summary["unavailable_or_not_directly_comparable"]["D02_fields_not_in_public_sidecar"]


def test_provenance_ids_record_sources_and_payload_is_bounded_and_public() -> None:
    index = _load_json("index.json")
    provenance = _load_json("data/provenance.json")
    summary = _load_json("data/comparison_summary.json")
    records = _records()
    ids = {sidecar["id"] for stage in provenance["stages"] for sidecar in stage["sidecars"]}

    assert {record["source_sidecar_id"] for record in records} <= ids
    assert {record["stage"] for record in records} == set(STAGE_ORDER)
    assert sum((EVIDENCE / "data" / name).stat().st_size for name in [
        "provenance.json", "comparison_records.jsonl", "comparison_summary.json"
    ]) < 1_000_000
    assert index["data_files"]["comparison_records"]["record_count"] > 900

    for text in _walk({"index": index, "provenance": provenance, "summary": summary, "records": records}):
        normalized = text.lower().replace("\\", "/")
        assert not re.search(r"\bhf_[a-z0-9]{12,}", text, re.IGNORECASE)
        assert not re.search(r"\bbearer\s+[a-z0-9._-]+", text, re.IGNORECASE)
        assert not re.match(r"^[a-z]:/", normalized)
        assert not normalized.startswith("//")
        assert not normalized.startswith("file://")
        assert not normalized.startswith("/mnt/")
        assert not normalized.startswith("/home/")
        assert not (normalized.startswith("/content/") and not normalized.startswith("/content/drive/"))
