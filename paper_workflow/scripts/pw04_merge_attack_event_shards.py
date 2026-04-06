"""
File purpose: Merge completed PW03 attack shards and materialize PW04 attack metrics exports.
Module type: General module
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.evaluation import metrics as eval_metrics
from main.evaluation import workflow_inputs as eval_workflow_inputs
from paper_workflow.scripts.pw_common import (
    ATTACKED_POSITIVE_SAMPLE_ROLE,
    build_family_root,
    read_jsonl,
    write_jsonl,
)
from scripts.notebook_runtime_common import (
    compute_file_sha256,
    ensure_directory,
    normalize_path_value,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
)


STAGE_NAME = "PW04_Attack_Merge_And_Metrics"
SCHEMA_VERSION = "pw_stage_04_v1"
CONTENT_SCORE_NAME = eval_metrics.CONTENT_CHAIN_SCORE_NAME
EVENT_ATTESTATION_SCORE_NAME = eval_metrics.EVENT_ATTESTATION_SCORE_NAME
PW02_SUMMARY_FILE_NAME = "pw02_summary.json"
PW04_SUMMARY_FILE_NAME = "pw04_summary.json"
FORMAL_RECORDS_DIRECTORY_NAME = "attack_formal_records"
ATTACK_MERGE_MANIFEST_FILE_NAME = "attack_merge_manifest.json"
ATTACK_POOL_MANIFEST_FILE_NAME = "attack_positive_pool_manifest.json"
FORMAL_ATTACK_FINAL_DECISION_METRICS_FILE_NAME = "formal_attack_final_decision_metrics.json"
FORMAL_ATTACK_ATTESTATION_METRICS_FILE_NAME = "formal_attack_attestation_metrics.json"
DERIVED_ATTACK_UNION_METRICS_FILE_NAME = "derived_attack_union_metrics.json"
PER_ATTACK_FAMILY_METRICS_FILE_NAME = "per_attack_family_metrics.json"
PER_ATTACK_CONDITION_METRICS_FILE_NAME = "per_attack_condition_metrics.json"
ATTACK_EVENT_TABLE_FILE_NAME = "attack_event_table.jsonl"
ATTACK_FAMILY_SUMMARY_CSV_FILE_NAME = "attack_family_summary.csv"
ATTACK_CONDITION_SUMMARY_CSV_FILE_NAME = "attack_condition_summary.csv"
CLEAN_ATTACK_OVERVIEW_FILE_NAME = "clean_attack_overview.json"


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    Load one required JSON object file.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    """
    Build canonical JSON text for stable mapping comparisons.

    Args:
        payload: Mapping payload.

    Returns:
        Canonical JSON text.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be Mapping")
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_threshold_artifact_paths(threshold_artifact_paths: Mapping[str, Any]) -> Dict[str, str]:
    """
    Normalize threshold artifact path bindings for equality checks.

    Args:
        threshold_artifact_paths: Threshold artifact mapping.

    Returns:
        Normalized threshold artifact path mapping.
    """
    if not isinstance(threshold_artifact_paths, Mapping):
        raise TypeError("threshold_artifact_paths must be Mapping")

    normalized: Dict[str, str] = {}
    for key_name in ["content", "attestation"]:
        path_value = threshold_artifact_paths.get(key_name)
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"threshold_artifact_paths missing {key_name}")
        normalized[key_name] = normalize_path_value(Path(path_value).expanduser().resolve())
    return normalized


def _resolve_pw04_paths(family_root: Path) -> Dict[str, Path]:
    """
    Resolve canonical PW04 output paths.

    Args:
        family_root: Family root path.

    Returns:
        Mapping of PW04 output directories and files.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")

    export_root = family_root / "exports" / "pw04"
    manifests_root = export_root / "manifests"
    records_root = export_root / "records" / FORMAL_RECORDS_DIRECTORY_NAME
    tables_root = export_root / "tables"
    summary_path = family_root / "runtime_state" / PW04_SUMMARY_FILE_NAME
    paths = {
        "export_root": export_root,
        "manifests_root": manifests_root,
        "records_root": records_root,
        "tables_root": tables_root,
        "summary_path": summary_path,
        "attack_merge_manifest_path": manifests_root / ATTACK_MERGE_MANIFEST_FILE_NAME,
        "attack_pool_manifest_path": export_root / ATTACK_POOL_MANIFEST_FILE_NAME,
        "formal_attack_final_decision_metrics_path": export_root / FORMAL_ATTACK_FINAL_DECISION_METRICS_FILE_NAME,
        "formal_attack_attestation_metrics_path": export_root / FORMAL_ATTACK_ATTESTATION_METRICS_FILE_NAME,
        "derived_attack_union_metrics_path": export_root / DERIVED_ATTACK_UNION_METRICS_FILE_NAME,
        "per_attack_family_metrics_path": export_root / PER_ATTACK_FAMILY_METRICS_FILE_NAME,
        "per_attack_condition_metrics_path": export_root / PER_ATTACK_CONDITION_METRICS_FILE_NAME,
        "attack_event_table_path": tables_root / ATTACK_EVENT_TABLE_FILE_NAME,
        "attack_family_summary_csv_path": tables_root / ATTACK_FAMILY_SUMMARY_CSV_FILE_NAME,
        "attack_condition_summary_csv_path": tables_root / ATTACK_CONDITION_SUMMARY_CSV_FILE_NAME,
        "clean_attack_overview_path": export_root / CLEAN_ATTACK_OVERVIEW_FILE_NAME,
    }
    for path_obj in paths.values():
        validate_path_within_base(family_root, path_obj, "PW04 output path")
    return paths


def _prepare_pw04_outputs(
    *,
    family_root: Path,
    export_root: Path,
    summary_path: Path,
    force_rerun: bool,
) -> None:
    """
    Prepare the PW04 output root with explicit rerun semantics.

    Args:
        family_root: Family root path.
        export_root: PW04 export root.
        summary_path: PW04 summary path.
        force_rerun: Whether to clear existing outputs.

    Returns:
        None.
    """
    if export_root.exists() or summary_path.exists():
        if not force_rerun:
            raise RuntimeError(
                "PW04 outputs already exist; rerun requires force_rerun: "
                f"export_root={normalize_path_value(export_root)}"
            )
        if export_root.exists():
            shutil.rmtree(export_root)
        if summary_path.exists():
            summary_path.unlink()

    ensure_directory(family_root / "runtime_state")
    ensure_directory(export_root)


def _resolve_authoritative_parent_event_id(
    *,
    event_manifest: Mapping[str, Any],
    detect_payload: Mapping[str, Any],
    attack_event_id: str,
) -> str:
    """
    Resolve the authoritative parent event id for one attacked event.

    Args:
        event_manifest: PW03 event manifest payload.
        detect_payload: PW03 staged detect payload.
        attack_event_id: Attack event identifier.

    Returns:
        Resolved parent event identifier.
    """
    manifest_parent_event_id = event_manifest.get("parent_event_id")
    detect_parent_event_id = detect_payload.get("paper_workflow_parent_event_id")

    normalized_manifest_parent = (
        str(manifest_parent_event_id)
        if isinstance(manifest_parent_event_id, str) and manifest_parent_event_id.strip()
        else None
    )
    normalized_detect_parent = (
        str(detect_parent_event_id)
        if isinstance(detect_parent_event_id, str) and detect_parent_event_id.strip()
        else None
    )
    if normalized_manifest_parent is None and normalized_detect_parent is None:
        raise ValueError(f"attack event missing authoritative parent_event_id: {attack_event_id}")
    if normalized_manifest_parent is not None and normalized_detect_parent is not None:
        if normalized_manifest_parent != normalized_detect_parent:
            raise ValueError(
                "PW04 parent_event_id mismatch between event manifest and staged detect record: "
                f"attack_event_id={attack_event_id}, "
                f"event_manifest.parent_event_id={normalized_manifest_parent}, "
                f"paper_workflow_parent_event_id={normalized_detect_parent}"
            )
    return normalized_manifest_parent or cast(str, normalized_detect_parent)


def _collect_completed_pw03_shard_manifests(
    *,
    family_root: Path,
    attack_shard_plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect all completed PW03 shard manifests declared by the shard plan.

    Args:
        family_root: Family root path.
        attack_shard_plan: PW00 attack shard plan payload.

    Returns:
        Ordered shard manifest rows with plan metadata.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(attack_shard_plan, Mapping):
        raise TypeError("attack_shard_plan must be Mapping")

    shards_node = attack_shard_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError("attack_shard_plan.shards must be list")

    shard_rows: List[Dict[str, Any]] = []
    for shard_node in cast(List[object], shards_node):
        if not isinstance(shard_node, Mapping):
            raise ValueError("attack_shard_plan.shards must contain objects")
        planned_shard = dict(cast(Mapping[str, Any], shard_node))
        attack_shard_index = planned_shard.get("attack_shard_index")
        expected_event_ids = planned_shard.get("assigned_attack_event_ids")
        if not isinstance(attack_shard_index, int) or attack_shard_index < 0:
            raise ValueError("attack shard row missing attack_shard_index")
        if not isinstance(expected_event_ids, list):
            raise ValueError(f"attack shard row missing assigned_attack_event_ids: {attack_shard_index}")

        shard_root = family_root / "attack_shards" / f"shard_{attack_shard_index:04d}"
        validate_path_within_base(family_root, shard_root, "PW04 shard root")
        shard_manifest_path = shard_root / "shard_manifest.json"
        shard_manifest = _load_required_json_dict(
            shard_manifest_path,
            f"PW03 shard manifest {attack_shard_index}",
        )
        if shard_manifest.get("status") != "completed":
            raise RuntimeError(
                "PW04 requires every planned PW03 shard_manifest to be completed: "
                f"attack_shard_index={attack_shard_index}, status={shard_manifest.get('status')}"
            )
        if shard_manifest.get("sample_role") != ATTACKED_POSITIVE_SAMPLE_ROLE:
            raise ValueError(
                "PW03 shard manifest sample_role mismatch: "
                f"attack_shard_index={attack_shard_index}, sample_role={shard_manifest.get('sample_role')}"
            )

        manifest_events_node = shard_manifest.get("events")
        if not isinstance(manifest_events_node, list):
            raise ValueError(f"PW03 shard manifest events must be list: {attack_shard_index}")
        manifest_event_ids: List[str] = []
        for event_node in cast(List[object], manifest_events_node):
            if not isinstance(event_node, Mapping):
                raise ValueError(f"PW03 shard manifest events must contain objects: {attack_shard_index}")
            event_id = event_node.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"PW03 shard manifest event missing event_id: {attack_shard_index}")
            manifest_event_ids.append(event_id)
        if len(manifest_event_ids) != len(cast(List[object], expected_event_ids)):
            raise ValueError(
                "PW03 shard manifest event_count mismatch with attack_shard_plan: "
                f"attack_shard_index={attack_shard_index}"
            )
        if set(manifest_event_ids) != set(str(event_id) for event_id in cast(List[object], expected_event_ids)):
            raise ValueError(
                "PW03 shard manifest assigned events mismatch with attack_shard_plan: "
                f"attack_shard_index={attack_shard_index}"
            )
        if int(shard_manifest.get("completed_event_count", len(manifest_event_ids))) != len(manifest_event_ids):
            raise RuntimeError(
                "PW03 shard manifest completed_event_count mismatch with assigned events: "
                f"attack_shard_index={attack_shard_index}"
            )
        if int(shard_manifest.get("failed_event_count", 0)) != 0:
            raise RuntimeError(
                "PW03 shard manifest reports failed events; PW04 cannot do partial merge: "
                f"attack_shard_index={attack_shard_index}"
            )

        shard_rows.append(
            {
                "attack_shard_index": attack_shard_index,
                "shard_root": shard_root,
                "shard_manifest_path": shard_manifest_path,
                "shard_manifest": shard_manifest,
                "expected_attack_event_ids": [str(event_id) for event_id in cast(List[object], expected_event_ids)],
            }
        )
    return shard_rows


def _collect_completed_attack_events(
    *,
    family_id: str,
    family_root: Path,
    shard_rows: Sequence[Mapping[str, Any]],
    attack_event_lookup: Mapping[str, Mapping[str, Any]],
    expected_source_finalize_manifest_digest: str,
    expected_threshold_artifact_paths: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """
    Collect all completed attack events declared by the shard plan.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        shard_rows: Completed shard manifest summaries.
        attack_event_lookup: Expected attack event rows keyed by attack_event_id.
        expected_source_finalize_manifest_digest: Expected finalize manifest digest.
        expected_threshold_artifact_paths: Expected threshold export path bindings.

    Returns:
        Ordered collected attack event rows.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")

    discovered_attack_event_ids: set[str] = set()
    discovered_parent_condition_keys: set[Tuple[str, str, str]] = set()
    collected_events: List[Dict[str, Any]] = []

    for shard_row in shard_rows:
        shard_root = Path(str(shard_row["shard_root"])).expanduser().resolve()
        expected_attack_event_ids = cast(List[str], shard_row.get("expected_attack_event_ids", []))
        for expected_attack_event_id in expected_attack_event_ids:
            expected_attack_event = attack_event_lookup.get(expected_attack_event_id)
            if not isinstance(expected_attack_event, Mapping):
                raise ValueError(f"attack_shard_plan references unknown attack_event_id: {expected_attack_event_id}")
            expected_attack_event_index = expected_attack_event.get("attack_event_index", expected_attack_event.get("event_index"))
            if not isinstance(expected_attack_event_index, int) or expected_attack_event_index < 0:
                raise ValueError(f"attack_event_grid missing attack_event_index: {expected_attack_event_id}")

            event_root = shard_root / "events" / f"event_{expected_attack_event_index:06d}"
            validate_path_within_base(family_root, event_root, "PW04 event root")
            event_manifest_path = event_root / "event_manifest.json"
            event_manifest = _load_required_json_dict(
                event_manifest_path,
                f"PW03 event manifest {expected_attack_event_id}",
            )
            if event_manifest.get("status") != "completed":
                raise RuntimeError(
                    "PW04 requires every planned PW03 event_manifest to be completed: "
                    f"attack_event_id={expected_attack_event_id}, status={event_manifest.get('status')}"
                )
            if event_manifest.get("sample_role") != ATTACKED_POSITIVE_SAMPLE_ROLE:
                raise ValueError(
                    "PW03 event sample_role must be attacked_positive before PW04: "
                    f"attack_event_id={expected_attack_event_id}, sample_role={event_manifest.get('sample_role')}"
                )

            manifest_event_id = event_manifest.get("event_id")
            if not isinstance(manifest_event_id, str) or not manifest_event_id:
                raise ValueError(f"PW03 event manifest missing event_id: {normalize_path_value(event_manifest_path)}")
            if manifest_event_id != expected_attack_event_id:
                raise ValueError(
                    "PW03 event manifest event_id mismatch with attack_shard_plan: "
                    f"expected={expected_attack_event_id}, actual={manifest_event_id}"
                )

            attack_event_id = event_manifest.get("attack_event_id", manifest_event_id)
            if not isinstance(attack_event_id, str) or not attack_event_id:
                raise ValueError(f"PW03 event manifest missing attack_event_id: {manifest_event_id}")
            if attack_event_id in discovered_attack_event_ids:
                raise ValueError(f"duplicate attack_event_id detected in PW03 events: {attack_event_id}")
            discovered_attack_event_ids.add(attack_event_id)

            detect_record_path_value = event_manifest.get("detect_record_path")
            if not isinstance(detect_record_path_value, str) or not detect_record_path_value.strip():
                raise FileNotFoundError(f"PW03 event manifest missing detect_record_path: {manifest_event_id}")
            detect_record_path = Path(detect_record_path_value).expanduser().resolve()
            detect_payload = _load_required_json_dict(detect_record_path, f"PW03 staged detect record {manifest_event_id}")

            sample_role_value = detect_payload.get("sample_role")
            if sample_role_value is not None and sample_role_value != ATTACKED_POSITIVE_SAMPLE_ROLE:
                raise ValueError(
                    "PW03 staged detect record sample_role must be attacked_positive: "
                    f"attack_event_id={attack_event_id}, sample_role={sample_role_value}"
                )

            parent_event_id = _resolve_authoritative_parent_event_id(
                event_manifest=event_manifest,
                detect_payload=detect_payload,
                attack_event_id=attack_event_id,
            )
            expected_parent_event_id = expected_attack_event.get("parent_event_id")
            if not isinstance(expected_parent_event_id, str) or not expected_parent_event_id:
                raise ValueError(f"attack_event_grid missing parent_event_id: {expected_attack_event_id}")
            if parent_event_id != expected_parent_event_id:
                raise ValueError(
                    "PW03 parent_event_id does not match attack_event_grid: "
                    f"attack_event_id={attack_event_id}, expected={expected_parent_event_id}, actual={parent_event_id}"
                )

            attack_family = event_manifest.get("attack_family")
            attack_config_name = event_manifest.get("attack_config_name")
            attack_condition_key = event_manifest.get("attack_condition_key")
            attack_params_digest = event_manifest.get("attack_params_digest")
            if not isinstance(attack_family, str) or not attack_family:
                raise ValueError(f"PW03 event manifest missing attack_family: {attack_event_id}")
            if not isinstance(attack_config_name, str) or not attack_config_name:
                raise ValueError(f"PW03 event manifest missing attack_config_name: {attack_event_id}")
            if not isinstance(attack_condition_key, str) or not attack_condition_key:
                raise ValueError(f"PW03 event manifest missing attack_condition_key: {attack_event_id}")
            if not isinstance(attack_params_digest, str) or not attack_params_digest:
                raise ValueError(f"PW03 event manifest missing attack_params_digest: {attack_event_id}")

            if attack_family != expected_attack_event.get("attack_family"):
                raise ValueError(f"attack_family mismatch for {attack_event_id}")
            if attack_config_name != expected_attack_event.get("attack_config_name"):
                raise ValueError(f"attack_config_name mismatch for {attack_event_id}")
            if attack_condition_key != expected_attack_event.get("attack_condition_key"):
                raise ValueError(f"attack_condition_key mismatch for {attack_event_id}")
            if attack_params_digest != expected_attack_event.get("attack_params_digest"):
                raise ValueError(f"attack_params_digest mismatch for {attack_event_id}")

            unique_parent_condition_key = (parent_event_id, attack_family, attack_condition_key)
            if unique_parent_condition_key in discovered_parent_condition_keys:
                raise ValueError(
                    "duplicate (parent_event_id, attack_family, attack_condition_key) detected: "
                    f"{unique_parent_condition_key}"
                )
            discovered_parent_condition_keys.add(unique_parent_condition_key)

            source_finalize_manifest_digest = event_manifest.get("source_finalize_manifest_digest")
            if not isinstance(source_finalize_manifest_digest, str) or not source_finalize_manifest_digest:
                raise ValueError(f"PW03 event manifest missing source_finalize_manifest_digest: {attack_event_id}")
            if source_finalize_manifest_digest != expected_source_finalize_manifest_digest:
                raise ValueError(
                    "PW03 event manifest source_finalize_manifest_digest mismatch: "
                    f"attack_event_id={attack_event_id}"
                )

            threshold_artifact_paths_node = event_manifest.get("threshold_artifact_paths")
            if not isinstance(threshold_artifact_paths_node, Mapping):
                raise ValueError(f"PW03 event manifest missing threshold_artifact_paths: {attack_event_id}")
            threshold_artifact_paths = _normalize_threshold_artifact_paths(threshold_artifact_paths_node)
            if threshold_artifact_paths != dict(expected_threshold_artifact_paths):
                raise ValueError(
                    "PW03 event manifest threshold artifact reference mismatch: "
                    f"attack_event_id={attack_event_id}"
                )

            attacked_image_path = event_manifest.get("attacked_image_path")
            if not isinstance(attacked_image_path, str) or not attacked_image_path.strip():
                raise ValueError(f"PW03 event manifest missing attacked_image_path: {attack_event_id}")

            content_score, content_score_source = eval_workflow_inputs._resolve_content_score_source(detect_payload)
            attestation_score, attestation_score_source = eval_workflow_inputs._resolve_event_attestation_score_source(detect_payload)
            content_payload = detect_payload.get("content_evidence_payload")
            geometry_payload = detect_payload.get("geometry_evidence_payload")
            fusion_payload = detect_payload.get("fusion_result")
            final_decision_payload = detect_payload.get("final_decision")

            fusion_status = None
            if isinstance(fusion_payload, Mapping):
                for candidate_key in ["status", "decision_status"]:
                    candidate_value = fusion_payload.get(candidate_key)
                    if isinstance(candidate_value, str) and candidate_value:
                        fusion_status = candidate_value
                        break
            if fusion_status is None and isinstance(final_decision_payload, Mapping):
                candidate_value = final_decision_payload.get("decision_status")
                if isinstance(candidate_value, str) and candidate_value:
                    fusion_status = candidate_value

            lf_detect_variant = detect_payload.get("lf_detect_variant")
            if not isinstance(lf_detect_variant, str) or not lf_detect_variant:
                lf_detect_variant = None

            collected_events.append(
                {
                    "family_id": family_id,
                    "event_manifest_path": normalize_path_value(event_manifest_path),
                    "event_manifest": event_manifest,
                    "detect_record_path": normalize_path_value(detect_record_path),
                    "detect_payload": detect_payload,
                    "attack_event_id": attack_event_id,
                    "event_id": manifest_event_id,
                    "attack_event_index": expected_attack_event_index,
                    "parent_event_id": parent_event_id,
                    "attack_family": attack_family,
                    "attack_config_name": attack_config_name,
                    "attack_condition_key": attack_condition_key,
                    "attack_params_digest": attack_params_digest,
                    "source_finalize_manifest_digest": source_finalize_manifest_digest,
                    "threshold_artifact_paths": threshold_artifact_paths,
                    "attacked_image_path": attacked_image_path,
                    "content_score": content_score,
                    "content_score_source": content_score_source,
                    "event_attestation_score": attestation_score,
                    "event_attestation_score_source": attestation_score_source,
                    "lf_detect_variant": lf_detect_variant,
                    "content_chain_status": content_payload.get("status") if isinstance(content_payload, Mapping) else None,
                    "geometry_chain_status": geometry_payload.get("status") if isinstance(geometry_payload, Mapping) else None,
                    "fusion_status": fusion_status,
                }
            )

    if len(collected_events) != len(attack_event_lookup):
        raise ValueError(
            "discovered PW03 event count does not match attack_shard_plan / attack_event_grid: "
            f"expected={len(attack_event_lookup)}, discovered={len(collected_events)}"
        )

    return sorted(collected_events, key=lambda row: int(row["attack_event_index"]))


def _build_attack_labelled_detect_payload(
    *,
    family_id: str,
    attack_event_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the attack-labelled detect payload copy used by PW04 materialization.

    Args:
        family_id: Family identifier.
        attack_event_row: Collected attack event row.

    Returns:
        Labelled detect payload copy.
    """
    detect_payload = attack_event_row.get("detect_payload")
    if not isinstance(detect_payload, Mapping):
        raise TypeError("attack_event_row.detect_payload must be Mapping")

    payload = copy.deepcopy(dict(cast(Mapping[str, Any], detect_payload)))
    payload["label"] = True
    payload["paper_workflow_stage"] = STAGE_NAME
    payload["paper_workflow_family_id"] = family_id
    payload["paper_workflow_attack_event_id"] = attack_event_row.get("attack_event_id")
    payload["paper_workflow_parent_event_id"] = attack_event_row.get("parent_event_id")
    payload["paper_workflow_attack_family"] = attack_event_row.get("attack_family")
    payload["paper_workflow_attack_condition_key"] = attack_event_row.get("attack_condition_key")
    payload["paper_workflow_attack_config_name"] = attack_event_row.get("attack_config_name")
    payload["paper_workflow_attack_params_digest"] = attack_event_row.get("attack_params_digest")
    return payload


def _materialize_attack_formal_record(
    *,
    family_id: str,
    attack_event_row: Mapping[str, Any],
    content_thresholds_artifact: Mapping[str, Any],
    attestation_thresholds_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Materialize one PW04 formal record copy from a staged detect record.

    Args:
        family_id: Family identifier.
        attack_event_row: Collected attack event row.
        content_thresholds_artifact: PW02 content thresholds artifact.
        attestation_thresholds_artifact: PW02 attestation thresholds artifact.

    Returns:
        Materialized PW04 formal record payload.
    """
    labelled_detect_payload = _build_attack_labelled_detect_payload(
        family_id=family_id,
        attack_event_row=attack_event_row,
    )
    formal_final_decision = eval_workflow_inputs.build_formal_final_decision_overlay(
        labelled_detect_payload,
        CONTENT_SCORE_NAME,
        dict(cast(Mapping[str, Any], content_thresholds_artifact)),
    )
    formal_event_attestation_decision = eval_workflow_inputs.build_formal_final_decision_overlay(
        labelled_detect_payload,
        EVENT_ATTESTATION_SCORE_NAME,
        dict(cast(Mapping[str, Any], attestation_thresholds_artifact)),
    )
    derived_attack_union_positive = bool(
        formal_final_decision.get("is_watermarked") is True
        or formal_event_attestation_decision.get("is_watermarked") is True
    )
    labelled_detect_payload["formal_final_decision"] = formal_final_decision
    labelled_detect_payload["formal_event_attestation_decision"] = formal_event_attestation_decision
    labelled_detect_payload["derived_attack_union_positive"] = derived_attack_union_positive
    return labelled_detect_payload


def _write_attack_formal_records(
    *,
    family_root: Path,
    records_root: Path,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
    content_thresholds_artifact: Mapping[str, Any],
    attestation_thresholds_artifact: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Materialize and write all PW04 formal record copies.

    Args:
        family_root: Family root path.
        records_root: PW04 formal records root.
        family_id: Family identifier.
        attack_event_rows: Collected attack event rows.
        content_thresholds_artifact: PW02 content thresholds artifact.
        attestation_thresholds_artifact: PW02 attestation thresholds artifact.

    Returns:
        Attack event rows augmented with formal record payload and path.
    """
    ensure_directory(records_root)
    augmented_rows: List[Dict[str, Any]] = []
    for attack_event_row in attack_event_rows:
        attack_event_id = str(attack_event_row["attack_event_id"])
        attack_event_index = int(attack_event_row["attack_event_index"])
        formal_record_path = records_root / f"event_{attack_event_index:06d}_formal_record.json"
        validate_path_within_base(family_root, formal_record_path, "PW04 formal record path")
        formal_record = _materialize_attack_formal_record(
            family_id=family_id,
            attack_event_row=attack_event_row,
            content_thresholds_artifact=content_thresholds_artifact,
            attestation_thresholds_artifact=attestation_thresholds_artifact,
        )
        write_json_atomic(formal_record_path, formal_record)
        augmented_row = dict(cast(Mapping[str, Any], attack_event_row))
        augmented_row["formal_record_path"] = normalize_path_value(formal_record_path)
        augmented_row["formal_record"] = formal_record
        augmented_rows.append(augmented_row)
    return augmented_rows


def _build_attack_merge_manifest_payload(
    *,
    family_id: str,
    family_root: Path,
    pw02_summary_path: Path,
    finalize_manifest_path: Path,
    content_threshold_export_path: Path,
    attestation_threshold_export_path: Path,
    attack_shard_plan_path: Path,
    attack_event_grid_path: Path,
    expected_attack_shard_count: int,
    discovered_attack_shard_count: int,
    expected_attack_event_count: int,
    discovered_attack_event_count: int,
    completed_attack_event_count: int,
    attack_event_rows: Sequence[Mapping[str, Any]],
    source_finalize_manifest_digest: str,
) -> Dict[str, Any]:
    """
    Build the PW04 attack merge manifest payload.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw02_summary_path: PW02 summary path.
        finalize_manifest_path: PW02 finalize manifest path.
        content_threshold_export_path: PW02 content threshold export path.
        attestation_threshold_export_path: PW02 attestation threshold export path.
        attack_shard_plan_path: PW00 attack shard plan path.
        attack_event_grid_path: PW00 attack event grid path.
        expected_attack_shard_count: Expected attack shard count.
        discovered_attack_shard_count: Discovered attack shard count.
        expected_attack_event_count: Expected attack event count.
        discovered_attack_event_count: Discovered attack event count.
        completed_attack_event_count: Completed attack event count.
        attack_event_rows: Collected attack event rows.
        source_finalize_manifest_digest: Bound finalize manifest digest.

    Returns:
        Attack merge manifest payload.
    """
    attack_families = sorted({str(row["attack_family"]) for row in attack_event_rows})
    parent_event_ids = sorted({str(row["parent_event_id"]) for row in attack_event_rows})
    return {
        "artifact_type": "paper_workflow_pw04_attack_merge_manifest",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "pw02_summary_path": normalize_path_value(pw02_summary_path),
        "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
        "content_threshold_export_path": normalize_path_value(content_threshold_export_path),
        "attestation_threshold_export_path": normalize_path_value(attestation_threshold_export_path),
        "attack_shard_plan_path": normalize_path_value(attack_shard_plan_path),
        "attack_event_grid_path": normalize_path_value(attack_event_grid_path),
        "expected_attack_shard_count": expected_attack_shard_count,
        "discovered_attack_shard_count": discovered_attack_shard_count,
        "expected_attack_event_count": expected_attack_event_count,
        "discovered_attack_event_count": discovered_attack_event_count,
        "completed_attack_event_count": completed_attack_event_count,
        "attack_family_count": len(attack_families),
        "parent_event_count": len(parent_event_ids),
        "source_finalize_manifest_digest": source_finalize_manifest_digest,
        "status": "completed",
    }


def _build_attack_pool_manifest_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the attacked-positive pool manifest payload.

    Args:
        family_id: Family identifier.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Attacked-positive pool manifest payload.
    """
    ordered_rows = sorted(attack_event_rows, key=lambda row: int(row["attack_event_index"]))
    events: List[Dict[str, Any]] = []
    for attack_event_row in ordered_rows:
        formal_record = cast(Mapping[str, Any], attack_event_row["formal_record"])
        formal_final_decision = cast(Mapping[str, Any], formal_record.get("formal_final_decision", {}))
        formal_event_attestation_decision = cast(
            Mapping[str, Any],
            formal_record.get("formal_event_attestation_decision", {}),
        )
        events.append(
            {
                "attack_event_id": attack_event_row["attack_event_id"],
                "attack_event_index": attack_event_row["attack_event_index"],
                "parent_event_id": attack_event_row["parent_event_id"],
                "attack_family": attack_event_row["attack_family"],
                "attack_config_name": attack_event_row["attack_config_name"],
                "attack_condition_key": attack_event_row["attack_condition_key"],
                "attack_params_digest": attack_event_row["attack_params_digest"],
                "attacked_image_path": attack_event_row["attacked_image_path"],
                "detect_record_path": attack_event_row["detect_record_path"],
                "formal_record_path": attack_event_row["formal_record_path"],
                "content_score": attack_event_row["content_score"],
                "event_attestation_score": attack_event_row["event_attestation_score"],
                "formal_final_decision_is_positive": formal_final_decision.get("is_watermarked") is True,
                "formal_event_attestation_is_positive": formal_event_attestation_decision.get("is_watermarked") is True,
                "derived_attack_union_positive": bool(formal_record.get("derived_attack_union_positive", False)),
            }
        )
    return {
        "artifact_type": "paper_workflow_pw04_attack_positive_pool_manifest",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "source_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "event_count": len(events),
        "attack_family_count": len({str(row["attack_family"]) for row in ordered_rows}),
        "parent_event_count": len({str(row["parent_event_id"]) for row in ordered_rows}),
        "events": events,
    }


def _build_attack_metrics_core(
    *,
    attack_event_rows: Sequence[Mapping[str, Any]],
    decision_key: str,
    positive_field_name: str,
) -> Dict[str, Any]:
    """
    Build a simple attack-only metrics summary from materialized formal records.

    Args:
        attack_event_rows: Collected and materialized attack event rows.
        decision_key: Formal decision key in the formal record payload.
        positive_field_name: Positive flag field name.

    Returns:
        Metrics summary mapping.
    """
    attack_positive_count = len(attack_event_rows)
    accepted_count = 0
    decision_status_counts: Dict[str, int] = {}
    for attack_event_row in attack_event_rows:
        formal_record = cast(Mapping[str, Any], attack_event_row["formal_record"])
        decision_payload = formal_record.get(decision_key)
        decision_mapping = cast(Mapping[str, Any], decision_payload) if isinstance(decision_payload, Mapping) else {}
        decision_status = decision_mapping.get("decision_status")
        if isinstance(decision_status, str) and decision_status:
            decision_status_counts[decision_status] = decision_status_counts.get(decision_status, 0) + 1
        if decision_mapping.get(positive_field_name) is True:
            accepted_count += 1

    attack_tpr = None if attack_positive_count <= 0 else float(accepted_count / attack_positive_count)
    return {
        "attack_positive_count": attack_positive_count,
        "accepted_count": accepted_count,
        "attack_tpr": attack_tpr,
        "decision_status_counts": decision_status_counts,
    }


def _build_formal_attack_final_decision_metrics_export(
    *,
    family_id: str,
    finalize_manifest_path: Path,
    content_threshold_export_path: Path,
    attack_pool_manifest_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the formal content-final attack metrics export payload.

    Args:
        family_id: Family identifier.
        finalize_manifest_path: PW02 finalize manifest path.
        content_threshold_export_path: PW02 content threshold export path.
        attack_pool_manifest_path: PW04 attack pool manifest path.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Formal content-final attack metrics export payload.
    """
    metrics = _build_attack_metrics_core(
        attack_event_rows=attack_event_rows,
        decision_key="formal_final_decision",
        positive_field_name="is_watermarked",
    )
    return {
        "artifact_type": "paper_workflow_pw04_formal_attack_final_decision_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "formal_attack_final_decision",
        "source_kind": "pw04_materialized_from_pw03_staged_detect_records",
        "is_formal_evaluate_record": False,
        "source_score_name": CONTENT_SCORE_NAME,
        "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
        "source_threshold_export_path": normalize_path_value(content_threshold_export_path),
        "attack_positive_pool_manifest_path": normalize_path_value(attack_pool_manifest_path),
        "metrics": metrics,
        "notes": "PW04 materialized formal content-final metrics from PW03 staged detect records using PW02 content thresholds; this is not an independent evaluate run.",
    }


def _build_formal_attack_attestation_metrics_export(
    *,
    family_id: str,
    finalize_manifest_path: Path,
    attestation_threshold_export_path: Path,
    attack_pool_manifest_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the formal attestation attack metrics export payload.

    Args:
        family_id: Family identifier.
        finalize_manifest_path: PW02 finalize manifest path.
        attestation_threshold_export_path: PW02 attestation threshold export path.
        attack_pool_manifest_path: PW04 attack pool manifest path.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Formal attestation attack metrics export payload.
    """
    metrics = _build_attack_metrics_core(
        attack_event_rows=attack_event_rows,
        decision_key="formal_event_attestation_decision",
        positive_field_name="is_watermarked",
    )
    return {
        "artifact_type": "paper_workflow_pw04_formal_attack_attestation_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "formal_attack_attestation",
        "source_kind": "pw04_materialized_from_pw03_staged_detect_records",
        "is_formal_evaluate_record": False,
        "source_score_name": EVENT_ATTESTATION_SCORE_NAME,
        "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
        "source_threshold_export_path": normalize_path_value(attestation_threshold_export_path),
        "attack_positive_pool_manifest_path": normalize_path_value(attack_pool_manifest_path),
        "metrics": metrics,
        "notes": "PW04 materialized formal attestation metrics from PW03 staged detect records using PW02 attestation thresholds; this is not an independent evaluate run.",
    }


def _build_derived_attack_union_metrics_export(
    *,
    family_id: str,
    content_threshold_export_path: Path,
    attestation_threshold_export_path: Path,
    attack_pool_manifest_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the derived attack-union metrics export payload.

    Args:
        family_id: Family identifier.
        content_threshold_export_path: PW02 content threshold export path.
        attestation_threshold_export_path: PW02 attestation threshold export path.
        attack_pool_manifest_path: PW04 attack pool manifest path.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Derived attack-union metrics export payload.
    """
    attack_positive_count = len(attack_event_rows)
    accepted_count = 0
    for attack_event_row in attack_event_rows:
        formal_record = cast(Mapping[str, Any], attack_event_row["formal_record"])
        if bool(formal_record.get("derived_attack_union_positive", False)):
            accepted_count += 1

    attack_tpr = None if attack_positive_count <= 0 else float(accepted_count / attack_positive_count)
    return {
        "artifact_type": "paper_workflow_pw04_derived_attack_union_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "derived_attack_union",
        "source_kind": "derived_union_from_formal_content_and_attestation_overlays",
        "is_formal_evaluate_record": False,
        "content_threshold_export_path": normalize_path_value(content_threshold_export_path),
        "attestation_threshold_export_path": normalize_path_value(attestation_threshold_export_path),
        "attack_positive_pool_manifest_path": normalize_path_value(attack_pool_manifest_path),
        "definition": "derived_attack_union_positive = formal_final_decision.is_watermarked OR formal_event_attestation_decision.is_watermarked",
        "metrics": {
            "attack_positive_count": attack_positive_count,
            "accepted_count": accepted_count,
            "attack_tpr": attack_tpr,
        },
        "notes": "This is a derived metrics export computed from the two PW04 formal overlays; it is not an independent formal evaluate record.",
    }


def _build_grouped_attack_metrics_rows(
    *,
    attack_event_rows: Sequence[Mapping[str, Any]],
    group_key_name: str,
) -> List[Dict[str, Any]]:
    """
    Build grouped attack metrics rows by one categorical field.

    Args:
        attack_event_rows: Collected and materialized attack event rows.
        group_key_name: Grouping field name.

    Returns:
        Grouped metrics rows.
    """
    grouped_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for attack_event_row in attack_event_rows:
        group_value = attack_event_row.get(group_key_name)
        if not isinstance(group_value, str) or not group_value:
            raise ValueError(f"attack event missing {group_key_name}")
        grouped_rows.setdefault(group_value, []).append(attack_event_row)

    output_rows: List[Dict[str, Any]] = []
    for group_value in sorted(grouped_rows):
        rows = grouped_rows[group_value]
        content_scores = [
            float(row["content_score"])
            for row in rows
            if isinstance(row.get("content_score"), (int, float)) and not isinstance(row.get("content_score"), bool)
        ]
        attestation_scores = [
            float(row["event_attestation_score"])
            for row in rows
            if isinstance(row.get("event_attestation_score"), (int, float)) and not isinstance(row.get("event_attestation_score"), bool)
        ]
        formal_final_positive_count = 0
        formal_attestation_positive_count = 0
        derived_union_positive_count = 0
        for row in rows:
            formal_record = cast(Mapping[str, Any], row["formal_record"])
            formal_final_decision = cast(Mapping[str, Any], formal_record.get("formal_final_decision", {}))
            formal_event_attestation_decision = cast(
                Mapping[str, Any],
                formal_record.get("formal_event_attestation_decision", {}),
            )
            if formal_final_decision.get("is_watermarked") is True:
                formal_final_positive_count += 1
            if formal_event_attestation_decision.get("is_watermarked") is True:
                formal_attestation_positive_count += 1
            if bool(formal_record.get("derived_attack_union_positive", False)):
                derived_union_positive_count += 1

        row_payload: Dict[str, Any] = {
            group_key_name: group_value,
            "event_count": len(rows),
            "parent_event_count": len({str(row["parent_event_id"]) for row in rows}),
            "formal_final_decision_attack_tpr": float(formal_final_positive_count / len(rows)) if rows else None,
            "formal_attestation_attack_tpr": float(formal_attestation_positive_count / len(rows)) if rows else None,
            "derived_attack_union_tpr": float(derived_union_positive_count / len(rows)) if rows else None,
            "content_score_mean": float(sum(content_scores) / len(content_scores)) if content_scores else None,
            "event_attestation_score_mean": float(sum(attestation_scores) / len(attestation_scores)) if attestation_scores else None,
        }
        if group_key_name == "attack_condition_key":
            row_payload["attack_family"] = str(rows[0]["attack_family"])
            row_payload["attack_config_name"] = str(rows[0]["attack_config_name"])
        output_rows.append(row_payload)
    return output_rows


def _build_per_attack_family_metrics_export(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the per-attack-family metrics export payload.

    Args:
        family_id: Family identifier.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Per-attack-family metrics export payload.
    """
    grouped_rows = _build_grouped_attack_metrics_rows(
        attack_event_rows=attack_event_rows,
        group_key_name="attack_family",
    )
    return {
        "artifact_type": "paper_workflow_pw04_per_attack_family_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "group_by": "attack_family",
        "rows": grouped_rows,
    }


def _build_per_attack_condition_metrics_export(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the per-attack-condition metrics export payload.

    Args:
        family_id: Family identifier.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Per-attack-condition metrics export payload.
    """
    grouped_rows = _build_grouped_attack_metrics_rows(
        attack_event_rows=attack_event_rows,
        group_key_name="attack_condition_key",
    )
    return {
        "artifact_type": "paper_workflow_pw04_per_attack_condition_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "group_by": "attack_condition_key",
        "rows": grouped_rows,
    }


def _write_attack_event_table_jsonl(
    *,
    output_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Write the flat attack event table JSONL export.

    Args:
        output_path: Attack event table JSONL path.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Flat attack event table rows.
    """
    rows: List[Dict[str, Any]] = []
    for attack_event_row in attack_event_rows:
        formal_record = cast(Mapping[str, Any], attack_event_row["formal_record"])
        formal_final_decision = cast(Mapping[str, Any], formal_record.get("formal_final_decision", {}))
        formal_event_attestation_decision = cast(
            Mapping[str, Any],
            formal_record.get("formal_event_attestation_decision", {}),
        )
        rows.append(
            {
                "attack_event_id": attack_event_row["attack_event_id"],
                "attack_event_index": attack_event_row["attack_event_index"],
                "parent_event_id": attack_event_row["parent_event_id"],
                "attack_family": attack_event_row["attack_family"],
                "attack_config_name": attack_event_row["attack_config_name"],
                "attack_condition_key": attack_event_row["attack_condition_key"],
                "attack_params_digest": attack_event_row["attack_params_digest"],
                "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
                "content_score": attack_event_row["content_score"],
                "content_score_source": attack_event_row["content_score_source"],
                "event_attestation_score": attack_event_row["event_attestation_score"],
                "event_attestation_score_source": attack_event_row["event_attestation_score_source"],
                "formal_final_decision_status": formal_final_decision.get("decision_status"),
                "formal_final_decision_is_positive": formal_final_decision.get("is_watermarked") is True,
                "formal_event_attestation_status": formal_event_attestation_decision.get("decision_status"),
                "formal_event_attestation_is_positive": formal_event_attestation_decision.get("is_watermarked") is True,
                "derived_attack_union_positive": bool(formal_record.get("derived_attack_union_positive", False)),
                "lf_detect_variant": attack_event_row.get("lf_detect_variant"),
                "content_chain_status": attack_event_row.get("content_chain_status"),
                "fusion_status": attack_event_row.get("fusion_status"),
                "geometry_chain_status": attack_event_row.get("geometry_chain_status"),
            }
        )
    ensure_directory(output_path.parent)
    write_jsonl(output_path, rows)
    return rows


def _write_attack_summary_csv(
    *,
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Write one grouped attack summary CSV export.

    Args:
        output_path: Output CSV path.
        rows: Grouped summary rows.

    Returns:
        None.
    """
    ensure_directory(output_path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key_name in row.keys():
            if key_name not in fieldnames:
                fieldnames.append(str(key_name))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def _build_clean_attack_overview_export(
    *,
    family_id: str,
    clean_formal_metrics_path: Path,
    clean_derived_metrics_path: Path,
    attack_formal_metrics_path: Path,
    attack_attestation_metrics_path: Path,
    attack_derived_metrics_path: Path,
    attack_formal_metrics_payload: Mapping[str, Any],
    attack_attestation_metrics_payload: Mapping[str, Any],
    attack_derived_metrics_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the clean-vs-attack overview export payload.

    Args:
        family_id: Family identifier.
        clean_formal_metrics_path: PW02 clean formal metrics path.
        clean_derived_metrics_path: PW02 clean derived metrics path.
        attack_formal_metrics_path: PW04 attack formal metrics path.
        attack_attestation_metrics_path: PW04 attack attestation metrics path.
        attack_derived_metrics_path: PW04 attack derived metrics path.
        attack_formal_metrics_payload: PW04 attack formal metrics payload.
        attack_attestation_metrics_payload: PW04 attack attestation metrics payload.
        attack_derived_metrics_payload: PW04 attack derived metrics payload.

    Returns:
        Clean-vs-attack overview payload.
    """
    clean_formal_metrics_payload = _load_required_json_dict(
        clean_formal_metrics_path,
        "PW02 clean formal final decision metrics",
    )
    clean_derived_metrics_payload = _load_required_json_dict(
        clean_derived_metrics_path,
        "PW02 clean derived system union metrics",
    )
    clean_formal_metrics = cast(Mapping[str, Any], clean_formal_metrics_payload.get("metrics", {}))
    clean_derived_metrics = cast(Mapping[str, Any], clean_derived_metrics_payload.get("metrics", {}))
    attack_formal_metrics = cast(Mapping[str, Any], attack_formal_metrics_payload.get("metrics", {}))
    attack_attestation_metrics = cast(Mapping[str, Any], attack_attestation_metrics_payload.get("metrics", {}))
    attack_derived_metrics = cast(Mapping[str, Any], attack_derived_metrics_payload.get("metrics", {}))
    return {
        "artifact_type": "paper_workflow_pw04_clean_attack_overview",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "clean_formal_final_decision_metrics_path": normalize_path_value(clean_formal_metrics_path),
        "clean_derived_system_union_metrics_path": normalize_path_value(clean_derived_metrics_path),
        "attack_formal_final_decision_metrics_path": normalize_path_value(attack_formal_metrics_path),
        "attack_formal_attestation_metrics_path": normalize_path_value(attack_attestation_metrics_path),
        "attack_derived_union_metrics_path": normalize_path_value(attack_derived_metrics_path),
        "clean_formal_tpr": clean_formal_metrics.get("final_decision_tpr"),
        "clean_formal_fpr": clean_formal_metrics.get("final_decision_fpr"),
        "clean_derived_union_tpr": clean_derived_metrics.get("system_tpr"),
        "attack_formal_tpr": attack_formal_metrics.get("attack_tpr"),
        "attack_formal_attestation_tpr": attack_attestation_metrics.get("attack_tpr"),
        "attack_derived_union_tpr": attack_derived_metrics.get("attack_tpr"),
    }


def _build_pw04_summary_payload(
    *,
    family_id: str,
    family_root: Path,
    summary_path: Path,
    attack_merge_manifest_path: Path,
    attack_pool_manifest_path: Path,
    formal_attack_final_decision_metrics_path: Path,
    formal_attack_attestation_metrics_path: Path,
    derived_attack_union_metrics_path: Path,
    per_attack_family_metrics_path: Path,
    per_attack_condition_metrics_path: Path,
    attack_event_table_path: Path,
    attack_family_summary_csv_path: Path,
    attack_condition_summary_csv_path: Path,
    clean_attack_overview_path: Path,
    expected_attack_shard_count: int,
    discovered_attack_shard_count: int,
    expected_attack_event_count: int,
    discovered_attack_event_count: int,
    completed_attack_event_count: int,
    attack_family_count: int,
    parent_event_count: int,
    formal_record_count: int,
) -> Dict[str, Any]:
    """
    Build the top-level PW04 summary payload.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        summary_path: PW04 summary path.
        attack_merge_manifest_path: Attack merge manifest path.
        attack_pool_manifest_path: Attack pool manifest path.
        formal_attack_final_decision_metrics_path: Formal attack content metrics path.
        formal_attack_attestation_metrics_path: Formal attack attestation metrics path.
        derived_attack_union_metrics_path: Derived attack union metrics path.
        per_attack_family_metrics_path: Per-family metrics path.
        per_attack_condition_metrics_path: Per-condition metrics path.
        attack_event_table_path: Attack event table path.
        attack_family_summary_csv_path: Attack family summary CSV path.
        attack_condition_summary_csv_path: Attack condition summary CSV path.
        clean_attack_overview_path: Clean-vs-attack overview path.
        expected_attack_shard_count: Expected attack shard count.
        discovered_attack_shard_count: Discovered attack shard count.
        expected_attack_event_count: Expected attack event count.
        discovered_attack_event_count: Discovered attack event count.
        completed_attack_event_count: Completed attack event count.
        attack_family_count: Unique attack family count.
        parent_event_count: Unique parent event count.
        formal_record_count: Materialized formal record count.

    Returns:
        PW04 summary payload.
    """
    return {
        "status": "completed",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "summary_path": normalize_path_value(summary_path),
        "attack_merge_manifest_path": normalize_path_value(attack_merge_manifest_path),
        "attack_positive_pool_manifest_path": normalize_path_value(attack_pool_manifest_path),
        "formal_attack_final_decision_metrics_path": normalize_path_value(formal_attack_final_decision_metrics_path),
        "formal_attack_attestation_metrics_path": normalize_path_value(formal_attack_attestation_metrics_path),
        "derived_attack_union_metrics_path": normalize_path_value(derived_attack_union_metrics_path),
        "per_attack_family_metrics_path": normalize_path_value(per_attack_family_metrics_path),
        "per_attack_condition_metrics_path": normalize_path_value(per_attack_condition_metrics_path),
        "attack_event_table_path": normalize_path_value(attack_event_table_path),
        "attack_family_summary_csv_path": normalize_path_value(attack_family_summary_csv_path),
        "attack_condition_summary_csv_path": normalize_path_value(attack_condition_summary_csv_path),
        "clean_attack_overview_path": normalize_path_value(clean_attack_overview_path),
        "expected_attack_shard_count": expected_attack_shard_count,
        "discovered_attack_shard_count": discovered_attack_shard_count,
        "expected_attack_event_count": expected_attack_event_count,
        "discovered_attack_event_count": discovered_attack_event_count,
        "completed_attack_event_count": completed_attack_event_count,
        "attack_family_count": attack_family_count,
        "parent_event_count": parent_event_count,
        "formal_record_count": formal_record_count,
    }


def run_pw04_merge_attack_event_shards(
    *,
    drive_project_root: Path,
    family_id: str,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Execute the PW04 attack merge and metrics materialization stage.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        force_rerun: Whether to clear existing PW04 outputs before rerun.

    Returns:
        PW04 summary payload.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(force_rerun, bool):
        raise TypeError("force_rerun must be bool")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    pw04_paths = _resolve_pw04_paths(family_root)
    _prepare_pw04_outputs(
        family_root=family_root,
        export_root=cast(Path, pw04_paths["export_root"]),
        summary_path=cast(Path, pw04_paths["summary_path"]),
        force_rerun=force_rerun,
    )
    ensure_directory(cast(Path, pw04_paths["manifests_root"]))
    ensure_directory(cast(Path, pw04_paths["records_root"]))
    ensure_directory(cast(Path, pw04_paths["tables_root"]))

    pw02_summary_path = family_root / "runtime_state" / PW02_SUMMARY_FILE_NAME
    finalize_manifest_path = family_root / "exports" / "pw02" / "paper_source_finalize_manifest.json"
    content_threshold_export_path = family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json"
    attestation_threshold_export_path = family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json"
    attack_shard_plan_path = family_root / "manifests" / "attack_shard_plan.json"
    attack_event_grid_path = family_root / "manifests" / "attack_event_grid.jsonl"
    clean_formal_metrics_path = family_root / "exports" / "pw02" / "formal_final_decision_metrics.json"
    clean_derived_metrics_path = family_root / "exports" / "pw02" / "derived_system_union_metrics.json"

    pw02_summary = _load_required_json_dict(pw02_summary_path, "PW02 summary")
    finalize_manifest = _load_required_json_dict(finalize_manifest_path, "paper source finalize manifest")
    content_threshold_export = _load_required_json_dict(content_threshold_export_path, "PW02 content threshold export")
    attestation_threshold_export = _load_required_json_dict(attestation_threshold_export_path, "PW02 attestation threshold export")
    attack_shard_plan = _load_required_json_dict(attack_shard_plan_path, "PW00 attack shard plan")

    finalize_manifest_path_from_summary = pw02_summary.get("paper_source_finalize_manifest_path")
    if not isinstance(finalize_manifest_path_from_summary, str) or not finalize_manifest_path_from_summary.strip():
        raise ValueError("PW02 summary missing paper_source_finalize_manifest_path")
    if normalize_path_value(Path(finalize_manifest_path_from_summary).expanduser().resolve()) != normalize_path_value(finalize_manifest_path):
        raise ValueError("PW02 summary finalize manifest path mismatch with canonical PW02 output")

    content_thresholds_artifact = content_threshold_export.get("thresholds_artifact")
    attestation_thresholds_artifact = attestation_threshold_export.get("thresholds_artifact")
    if not isinstance(content_thresholds_artifact, Mapping):
        raise ValueError("PW02 content threshold export missing thresholds_artifact")
    if not isinstance(attestation_thresholds_artifact, Mapping):
        raise ValueError("PW02 attestation threshold export missing thresholds_artifact")

    attack_event_rows_from_grid = read_jsonl(attack_event_grid_path)
    attack_event_lookup: Dict[str, Dict[str, Any]] = {}
    for attack_event_row in attack_event_rows_from_grid:
        attack_event_id = attack_event_row.get("attack_event_id", attack_event_row.get("event_id"))
        if not isinstance(attack_event_id, str) or not attack_event_id:
            raise ValueError("attack_event_grid contains invalid attack_event_id")
        if attack_event_id in attack_event_lookup:
            raise ValueError(f"duplicate attack_event_id in attack_event_grid: {attack_event_id}")
        attack_event_lookup[attack_event_id] = attack_event_row

    attack_plan_shards = attack_shard_plan.get("shards")
    if not isinstance(attack_plan_shards, list):
        raise ValueError("attack_shard_plan.shards must be list")
    planned_attack_event_ids: List[str] = []
    for shard_node in cast(List[object], attack_plan_shards):
        if not isinstance(shard_node, Mapping):
            raise ValueError("attack_shard_plan.shards must contain objects")
        assigned_attack_event_ids = shard_node.get("assigned_attack_event_ids")
        if not isinstance(assigned_attack_event_ids, list):
            raise ValueError("attack_shard_plan shard missing assigned_attack_event_ids")
        planned_attack_event_ids.extend(str(event_id) for event_id in cast(List[object], assigned_attack_event_ids))

    expected_attack_event_count = attack_shard_plan.get("attack_event_count")
    expected_attack_shard_count = attack_shard_plan.get("attack_shard_count")
    if not isinstance(expected_attack_event_count, int) or expected_attack_event_count < 0:
        raise ValueError("attack_shard_plan missing attack_event_count")
    if not isinstance(expected_attack_shard_count, int) or expected_attack_shard_count <= 0:
        raise ValueError("attack_shard_plan missing attack_shard_count")
    if expected_attack_event_count != len(planned_attack_event_ids):
        raise ValueError("attack_shard_plan attack_event_count mismatch with shard assignments")
    if expected_attack_event_count != len(attack_event_lookup):
        raise ValueError("attack_shard_plan attack_event_count mismatch with attack_event_grid")
    if set(planned_attack_event_ids) != set(attack_event_lookup.keys()):
        raise ValueError("attack_shard_plan expected universe does not match attack_event_grid")

    source_finalize_manifest_digest = compute_file_sha256(finalize_manifest_path)
    expected_threshold_artifact_paths = {
        "content": normalize_path_value(content_threshold_export_path),
        "attestation": normalize_path_value(attestation_threshold_export_path),
    }

    shard_rows = _collect_completed_pw03_shard_manifests(
        family_root=family_root,
        attack_shard_plan=attack_shard_plan,
    )
    attack_event_rows = _collect_completed_attack_events(
        family_id=family_id,
        family_root=family_root,
        shard_rows=shard_rows,
        attack_event_lookup=attack_event_lookup,
        expected_source_finalize_manifest_digest=source_finalize_manifest_digest,
        expected_threshold_artifact_paths=expected_threshold_artifact_paths,
    )
    materialized_attack_event_rows = _write_attack_formal_records(
        family_root=family_root,
        records_root=cast(Path, pw04_paths["records_root"]),
        family_id=family_id,
        attack_event_rows=attack_event_rows,
        content_thresholds_artifact=content_thresholds_artifact,
        attestation_thresholds_artifact=attestation_thresholds_artifact,
    )

    attack_merge_manifest_payload = _build_attack_merge_manifest_payload(
        family_id=family_id,
        family_root=family_root,
        pw02_summary_path=pw02_summary_path,
        finalize_manifest_path=finalize_manifest_path,
        content_threshold_export_path=content_threshold_export_path,
        attestation_threshold_export_path=attestation_threshold_export_path,
        attack_shard_plan_path=attack_shard_plan_path,
        attack_event_grid_path=attack_event_grid_path,
        expected_attack_shard_count=expected_attack_shard_count,
        discovered_attack_shard_count=len(shard_rows),
        expected_attack_event_count=expected_attack_event_count,
        discovered_attack_event_count=len(materialized_attack_event_rows),
        completed_attack_event_count=len(materialized_attack_event_rows),
        attack_event_rows=materialized_attack_event_rows,
        source_finalize_manifest_digest=source_finalize_manifest_digest,
    )
    attack_pool_manifest_payload = _build_attack_pool_manifest_payload(
        family_id=family_id,
        attack_event_rows=materialized_attack_event_rows,
    )
    formal_attack_final_decision_metrics_payload = _build_formal_attack_final_decision_metrics_export(
        family_id=family_id,
        finalize_manifest_path=finalize_manifest_path,
        content_threshold_export_path=content_threshold_export_path,
        attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
        attack_event_rows=materialized_attack_event_rows,
    )
    formal_attack_attestation_metrics_payload = _build_formal_attack_attestation_metrics_export(
        family_id=family_id,
        finalize_manifest_path=finalize_manifest_path,
        attestation_threshold_export_path=attestation_threshold_export_path,
        attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
        attack_event_rows=materialized_attack_event_rows,
    )
    derived_attack_union_metrics_payload = _build_derived_attack_union_metrics_export(
        family_id=family_id,
        content_threshold_export_path=content_threshold_export_path,
        attestation_threshold_export_path=attestation_threshold_export_path,
        attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
        attack_event_rows=materialized_attack_event_rows,
    )
    per_attack_family_metrics_payload = _build_per_attack_family_metrics_export(
        family_id=family_id,
        attack_event_rows=materialized_attack_event_rows,
    )
    per_attack_condition_metrics_payload = _build_per_attack_condition_metrics_export(
        family_id=family_id,
        attack_event_rows=materialized_attack_event_rows,
    )
    attack_event_table_rows = _write_attack_event_table_jsonl(
        output_path=cast(Path, pw04_paths["attack_event_table_path"]),
        attack_event_rows=materialized_attack_event_rows,
    )
    _write_attack_summary_csv(
        output_path=cast(Path, pw04_paths["attack_family_summary_csv_path"]),
        rows=cast(List[Mapping[str, Any]], per_attack_family_metrics_payload["rows"]),
    )
    _write_attack_summary_csv(
        output_path=cast(Path, pw04_paths["attack_condition_summary_csv_path"]),
        rows=cast(List[Mapping[str, Any]], per_attack_condition_metrics_payload["rows"]),
    )
    clean_attack_overview_payload = _build_clean_attack_overview_export(
        family_id=family_id,
        clean_formal_metrics_path=clean_formal_metrics_path,
        clean_derived_metrics_path=clean_derived_metrics_path,
        attack_formal_metrics_path=cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"]),
        attack_attestation_metrics_path=cast(Path, pw04_paths["formal_attack_attestation_metrics_path"]),
        attack_derived_metrics_path=cast(Path, pw04_paths["derived_attack_union_metrics_path"]),
        attack_formal_metrics_payload=formal_attack_final_decision_metrics_payload,
        attack_attestation_metrics_payload=formal_attack_attestation_metrics_payload,
        attack_derived_metrics_payload=derived_attack_union_metrics_payload,
    )

    write_json_atomic(cast(Path, pw04_paths["attack_merge_manifest_path"]), attack_merge_manifest_payload)
    write_json_atomic(cast(Path, pw04_paths["attack_pool_manifest_path"]), attack_pool_manifest_payload)
    write_json_atomic(
        cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"]),
        formal_attack_final_decision_metrics_payload,
    )
    write_json_atomic(
        cast(Path, pw04_paths["formal_attack_attestation_metrics_path"]),
        formal_attack_attestation_metrics_payload,
    )
    write_json_atomic(
        cast(Path, pw04_paths["derived_attack_union_metrics_path"]),
        derived_attack_union_metrics_payload,
    )
    write_json_atomic(cast(Path, pw04_paths["per_attack_family_metrics_path"]), per_attack_family_metrics_payload)
    write_json_atomic(cast(Path, pw04_paths["per_attack_condition_metrics_path"]), per_attack_condition_metrics_payload)
    write_json_atomic(cast(Path, pw04_paths["clean_attack_overview_path"]), clean_attack_overview_payload)

    summary_payload = _build_pw04_summary_payload(
        family_id=family_id,
        family_root=family_root,
        summary_path=cast(Path, pw04_paths["summary_path"]),
        attack_merge_manifest_path=cast(Path, pw04_paths["attack_merge_manifest_path"]),
        attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
        formal_attack_final_decision_metrics_path=cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"]),
        formal_attack_attestation_metrics_path=cast(Path, pw04_paths["formal_attack_attestation_metrics_path"]),
        derived_attack_union_metrics_path=cast(Path, pw04_paths["derived_attack_union_metrics_path"]),
        per_attack_family_metrics_path=cast(Path, pw04_paths["per_attack_family_metrics_path"]),
        per_attack_condition_metrics_path=cast(Path, pw04_paths["per_attack_condition_metrics_path"]),
        attack_event_table_path=cast(Path, pw04_paths["attack_event_table_path"]),
        attack_family_summary_csv_path=cast(Path, pw04_paths["attack_family_summary_csv_path"]),
        attack_condition_summary_csv_path=cast(Path, pw04_paths["attack_condition_summary_csv_path"]),
        clean_attack_overview_path=cast(Path, pw04_paths["clean_attack_overview_path"]),
        expected_attack_shard_count=expected_attack_shard_count,
        discovered_attack_shard_count=len(shard_rows),
        expected_attack_event_count=expected_attack_event_count,
        discovered_attack_event_count=len(attack_event_table_rows),
        completed_attack_event_count=len(materialized_attack_event_rows),
        attack_family_count=len({str(row["attack_family"]) for row in materialized_attack_event_rows}),
        parent_event_count=len({str(row["parent_event_id"]) for row in materialized_attack_event_rows}),
        formal_record_count=len(materialized_attack_event_rows),
    )
    write_json_atomic(cast(Path, pw04_paths["summary_path"]), summary_payload)
    return summary_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the PW04 CLI argument parser.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Merge PW03 attack shards and materialize PW04 metrics exports.")
    parser.add_argument("--drive-project-root", required=True, help="Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Family identifier.")
    parser.add_argument("--force-rerun", action="store_true", help="Clear existing PW04 outputs before rerun.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute the PW04 CLI entrypoint.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process-style exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_pw04_merge_attack_event_shards(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())