"""
File purpose: Merge PW01 source shards and execute PW02 global threshold workflows.
Module type: General module
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from main.evaluation import metrics as eval_metrics
from main.evaluation import workflow_inputs as eval_workflow_inputs
from main.evaluation.experiment_matrix import _build_system_final_metrics_for_run
from scripts.notebook_runtime_common import (
    REPO_ROOT,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    run_command_with_logs,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)

from paper_workflow.scripts.pw_common import (
    ACTIVE_SAMPLE_ROLE,
    CLEAN_NEGATIVE_SAMPLE_ROLE,
    build_family_root,
    build_source_shard_root,
    resolve_family_layout_paths,
    validate_source_sample_role,
)


CONTENT_SCORE_NAME = eval_metrics.CONTENT_CHAIN_SCORE_NAME
EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
POOL_MANIFEST_FILE_NAMES = {
    ACTIVE_SAMPLE_ROLE: "positive_source_pool_manifest.json",
    CLEAN_NEGATIVE_SAMPLE_ROLE: "clean_negative_pool_manifest.json",
}
FINALIZE_MANIFEST_FILE_NAME = "paper_source_finalize_manifest.json"
SYSTEM_FINAL_METRICS_FILE_NAME = "system_final_metrics.json"


def _resolve_top_level_score_directory_name(score_name: str) -> str:
    """
    Resolve the PW02 top-level export directory name for one score.

    Args:
        score_name: Canonical score name.

    Returns:
        Stable export directory token.
    """
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
    if eval_metrics.is_content_chain_score_name(score_name):
        return "content"
    if score_name == EVENT_ATTESTATION_SCORE_NAME:
        return "attestation"
    raise ValueError(f"unsupported score_name: {score_name}")


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


def _resolve_score_value(
    payload: Dict[str, Any],
    score_name: str,
) -> Tuple[float | None, str | None]:
    """
    Resolve one formal score value from a detect-record payload.

    Args:
        payload: Detect-record payload.
        score_name: Requested score name.

    Returns:
        Tuple of (score_value, score_source).
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    if eval_metrics.is_content_chain_score_name(score_name):
        return eval_workflow_inputs._resolve_content_score_source(payload)
    if score_name == EVENT_ATTESTATION_SCORE_NAME:
        return eval_workflow_inputs._resolve_event_attestation_score_source(payload)
    raise ValueError(f"unsupported score_name: {score_name}")


def _resolve_role_plan(source_shard_plan: Mapping[str, Any], sample_role: str) -> Dict[str, Any]:
    """
    Resolve one sample-role plan from the PW00 shard plan.

    Args:
        source_shard_plan: Source shard plan payload.
        sample_role: Supported source sample role.

    Returns:
        Role-specific shard plan mapping.
    """
    if not isinstance(source_shard_plan, Mapping):
        raise TypeError("source_shard_plan must be Mapping")
    normalized_sample_role = validate_source_sample_role(sample_role)
    sample_role_plans_node = source_shard_plan.get("sample_role_plans")
    if not isinstance(sample_role_plans_node, Mapping):
        raise ValueError("source shard plan missing sample_role_plans")
    role_plan_node = sample_role_plans_node.get(normalized_sample_role)
    if not isinstance(role_plan_node, Mapping):
        raise ValueError(f"source shard plan missing {normalized_sample_role} plan")
    return cast(Dict[str, Any], role_plan_node)


def _collect_completed_events_for_role(
    *,
    family_root: Path,
    source_shard_plan: Mapping[str, Any],
    sample_role: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect completed event manifests for one source role.

    Args:
        family_root: Family root path.
        source_shard_plan: Source shard plan payload.
        sample_role: Supported source sample role.

    Returns:
        Event manifest lookup keyed by event_id.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    normalized_sample_role = validate_source_sample_role(sample_role)
    role_plan = _resolve_role_plan(source_shard_plan, normalized_sample_role)

    shards_node = role_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError(f"source shard plan {normalized_sample_role}.shards must be list")

    event_lookup: Dict[str, Dict[str, Any]] = {}
    for shard_row_node in cast(List[object], shards_node):
        if not isinstance(shard_row_node, dict):
            raise ValueError("source shard plan shards must contain objects")
        shard_row = cast(Dict[str, Any], shard_row_node)
        shard_index = shard_row.get("shard_index")
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("source shard row missing shard_index")

        shard_root = build_source_shard_root(family_root, normalized_sample_role, shard_index)
        shard_manifest_path = shard_root / "shard_manifest.json"
        shard_manifest = _load_required_json_dict(
            shard_manifest_path,
            f"PW01 shard manifest {normalized_sample_role}:{shard_index}",
        )
        if shard_manifest.get("status") != "completed":
            raise ValueError(
                f"PW01 shard manifest must be completed before PW02: role={normalized_sample_role}, shard_index={shard_index}"
            )

        events_node = shard_manifest.get("events")
        if not isinstance(events_node, list):
            raise ValueError("PW01 shard manifest events must be list")
        for event_node in cast(List[object], events_node):
            if not isinstance(event_node, dict):
                raise ValueError("PW01 shard manifest events must contain objects")
            event_payload = dict(cast(Dict[str, Any], event_node))
            event_id = event_payload.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("PW01 shard event missing event_id")
            if event_id in event_lookup:
                raise ValueError(f"duplicate event_id across PW01 shard manifests: {event_id}")
            if event_payload.get("sample_role") != normalized_sample_role:
                raise ValueError(
                    f"PW01 shard event sample_role mismatch: expected={normalized_sample_role}, actual={event_payload.get('sample_role')}"
                )
            event_payload["source_shard_index"] = shard_index
            event_payload["source_shard_root"] = normalize_path_value(shard_root)
            event_payload["source_shard_manifest_path"] = normalize_path_value(shard_manifest_path)
            event_lookup[event_id] = event_payload
    return event_lookup


def _build_source_pool_manifest_payload(
    *,
    family_id: str,
    sample_role: str,
    event_lookup: Mapping[str, Dict[str, Any]],
    family_manifest_path: Path,
    source_shard_plan_path: Path,
    source_split_plan_path: Path,
    stage_root: Path,
) -> Dict[str, Any]:
    """
    Build one top-level PW02 source-pool manifest payload.

    Args:
        family_id: Family identifier.
        sample_role: Supported source sample role.
        event_lookup: Event lookup keyed by event_id.
        family_manifest_path: Family manifest path.
        source_shard_plan_path: Source shard plan path.
        source_split_plan_path: Source split plan path.
        stage_root: PW02 stage root.

    Returns:
        Source-pool manifest payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if not isinstance(event_lookup, Mapping):
        raise TypeError("event_lookup must be Mapping")

    ordered_events = sorted(
        (dict(event_payload) for event_payload in event_lookup.values()),
        key=lambda item: int(item.get("event_index", -1)),
    )

    shard_rows: Dict[int, Dict[str, Any]] = {}
    manifest_events: List[Dict[str, Any]] = []
    for event_payload in ordered_events:
        event_id = event_payload.get("event_id")
        event_index = event_payload.get("event_index")
        detect_record_path = event_payload.get("detect_record_path")
        source_shard_index = event_payload.get("source_shard_index")
        source_shard_root = event_payload.get("source_shard_root")
        source_shard_manifest_path = event_payload.get("source_shard_manifest_path")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("source pool event missing event_id")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError(f"source pool event missing event_index: {event_id}")
        if not isinstance(detect_record_path, str) or not detect_record_path:
            raise ValueError(f"source pool event missing detect_record_path: {event_id}")
        if not isinstance(source_shard_index, int) or source_shard_index < 0:
            raise ValueError(f"source pool event missing source_shard_index: {event_id}")
        if not isinstance(source_shard_root, str) or not source_shard_root:
            raise ValueError(f"source pool event missing source_shard_root: {event_id}")
        if not isinstance(source_shard_manifest_path, str) or not source_shard_manifest_path:
            raise ValueError(f"source pool event missing source_shard_manifest_path: {event_id}")

        manifest_events.append(
            {
                "event_id": event_id,
                "event_index": event_index,
                "sample_role": normalized_sample_role,
                "detect_record_path": detect_record_path,
                "source_shard_index": source_shard_index,
                "source_shard_root": source_shard_root,
                "source_shard_manifest_path": source_shard_manifest_path,
            }
        )

        shard_row = shard_rows.get(source_shard_index)
        if shard_row is None:
            shard_row = {
                "sample_role": normalized_sample_role,
                "shard_index": source_shard_index,
                "shard_root": source_shard_root,
                "shard_manifest_path": source_shard_manifest_path,
                "event_count": 0,
                "event_ids": [],
            }
            shard_rows[source_shard_index] = shard_row
        shard_row["event_count"] = int(shard_row["event_count"]) + 1
        cast(List[str], shard_row["event_ids"]).append(event_id)

    ordered_shards = [shard_rows[index] for index in sorted(shard_rows)]
    return {
        "artifact_type": "paper_workflow_pw02_source_pool_manifest",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "source_role": normalized_sample_role,
        "event_count": len(manifest_events),
        "event_ids": [event_payload["event_id"] for event_payload in manifest_events],
        "events": manifest_events,
        "source_shard_count": len(ordered_shards),
        "source_shards": ordered_shards,
        "source_shard_manifest_paths": [
            str(shard_row["shard_manifest_path"])
            for shard_row in ordered_shards
        ],
        "key_paths": {
            "family_manifest_path": normalize_path_value(family_manifest_path),
            "source_shard_plan_path": normalize_path_value(source_shard_plan_path),
            "source_split_plan_path": normalize_path_value(source_split_plan_path),
            "pw02_stage_root": normalize_path_value(stage_root),
        },
    }


def _write_source_pool_manifest(
    *,
    stage_root: Path,
    family_id: str,
    sample_role: str,
    event_lookup: Mapping[str, Dict[str, Any]],
    family_manifest_path: Path,
    source_shard_plan_path: Path,
    source_split_plan_path: Path,
) -> Dict[str, Any]:
    """
    Write one top-level PW02 source-pool manifest.

    Args:
        stage_root: PW02 stage root.
        family_id: Family identifier.
        sample_role: Supported source sample role.
        event_lookup: Event lookup keyed by event_id.
        family_manifest_path: Family manifest path.
        source_shard_plan_path: Source shard plan path.
        source_split_plan_path: Source split plan path.

    Returns:
        Manifest summary with output path and payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    normalized_sample_role = validate_source_sample_role(sample_role)
    manifest_path = stage_root / POOL_MANIFEST_FILE_NAMES[normalized_sample_role]
    payload = _build_source_pool_manifest_payload(
        family_id=family_id,
        sample_role=normalized_sample_role,
        event_lookup=event_lookup,
        family_manifest_path=family_manifest_path,
        source_shard_plan_path=source_shard_plan_path,
        source_split_plan_path=source_split_plan_path,
        stage_root=stage_root,
    )
    write_json_atomic(manifest_path, payload)
    return {
        "source_role": normalized_sample_role,
        "path": normalize_path_value(manifest_path),
        "payload": payload,
    }


def _build_threshold_export(
    *,
    family_id: str,
    stage_root: Path,
    score_name: str,
    score_run: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one top-level threshold export from the real calibrate outputs.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        score_name: Canonical score name.
        score_run: Score-run summary.

    Returns:
        Threshold export summary with path and payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    thresholds_artifact_path = Path(str(score_run.get("thresholds_artifact_path", ""))).expanduser().resolve()
    calibration_record_path = Path(str(score_run.get("calibration_record_path", ""))).expanduser().resolve()
    thresholds_artifact = _load_required_json_dict(thresholds_artifact_path, f"PW02 thresholds artifact {score_name}")
    calibration_record = _load_required_json_dict(calibration_record_path, f"PW02 calibration record {score_name}")
    export_path = stage_root / "thresholds" / _resolve_top_level_score_directory_name(score_name) / "thresholds.json"
    ensure_directory(export_path.parent)
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_threshold_export",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "score_name": score_name,
        "source_calibrate_run_root": str(score_run.get("calibrate_run_root")),
        "source_thresholds_artifact_path": normalize_path_value(thresholds_artifact_path),
        "source_calibration_record_path": normalize_path_value(calibration_record_path),
        "thresholds_artifact": thresholds_artifact,
        "calibration_record_status": calibration_record.get("status", "<absent>"),
    }
    write_json_atomic(export_path, payload)
    return {
        "score_name": score_name,
        "path": normalize_path_value(export_path),
        "payload": payload,
    }


def _count_records_by_role(records_summary: Mapping[str, Any]) -> Dict[str, int]:
    """
    Count prepared records by sample role.

    Args:
        records_summary: Score-run records summary.

    Returns:
        Sample-role counts.
    """
    counts = {
        ACTIVE_SAMPLE_ROLE: 0,
        CLEAN_NEGATIVE_SAMPLE_ROLE: 0,
    }
    records_node = records_summary.get("records")
    if not isinstance(records_node, list):
        return counts
    for record_node in cast(List[object], records_node):
        if not isinstance(record_node, Mapping):
            continue
        sample_role = record_node.get("sample_role")
        if isinstance(sample_role, str) and sample_role in counts:
            counts[sample_role] += 1
    return counts


def _build_clean_evaluate_export(
    *,
    family_id: str,
    stage_root: Path,
    score_name: str,
    score_run: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one top-level clean-evaluate export from the real evaluate outputs.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        score_name: Canonical score name.
        score_run: Score-run summary.

    Returns:
        Evaluate export summary with path and payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    evaluate_record_path = Path(str(score_run.get("evaluate_record_path", ""))).expanduser().resolve()
    evaluate_record = _load_required_json_dict(evaluate_record_path, f"PW02 evaluate record {score_name}")
    export_path = stage_root / "evaluate" / "clean" / _resolve_top_level_score_directory_name(score_name) / "evaluate_record.json"
    ensure_directory(export_path.parent)
    evaluate_inputs = cast(Mapping[str, Any], score_run.get("evaluate_inputs", {}))
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_clean_evaluate_export",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "score_name": score_name,
        "evaluation_scope": "positive_source_vs_clean_negative",
        "positive_source_role": ACTIVE_SAMPLE_ROLE,
        "negative_source_role": CLEAN_NEGATIVE_SAMPLE_ROLE,
        "source_evaluate_run_root": str(score_run.get("evaluate_run_root")),
        "source_evaluate_record_path": normalize_path_value(evaluate_record_path),
        "source_evaluation_report_path": str(score_run.get("evaluation_report_path")),
        "source_evaluate_inputs_glob": evaluate_inputs.get("records_glob"),
        "evaluate_input_counts": _count_records_by_role(evaluate_inputs),
        "evaluate_record": evaluate_record,
    }
    write_json_atomic(export_path, payload)
    return {
        "score_name": score_name,
        "path": normalize_path_value(export_path),
        "payload": payload,
    }


def _build_system_final_metrics_export(
    *,
    family_id: str,
    stage_root: Path,
    content_score_run: Mapping[str, Any],
    system_final_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the honest PW02 system_final derived-metrics export.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        content_score_run: Content score-run summary.
        system_final_metrics: Derived system-final metrics.

    Returns:
        System-final export summary with path and payload.
    """
    export_path = stage_root / SYSTEM_FINAL_METRICS_FILE_NAME
    evaluate_inputs = cast(Mapping[str, Any], content_score_run.get("evaluate_inputs", {}))
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_system_final_metrics",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "system_final",
        "source_kind": "derived_metrics_from_content_evaluate_inputs",
        "is_formal_evaluate_record": False,
        "source_score_name": CONTENT_SCORE_NAME,
        "source_evaluate_run_root": str(content_score_run.get("evaluate_run_root")),
        "source_evaluate_record_path": str(content_score_run.get("evaluate_record_path")),
        "source_evaluate_inputs_glob": evaluate_inputs.get("records_glob"),
        "metrics": dict(system_final_metrics),
        "notes": "Derived via main.evaluation.experiment_matrix._build_system_final_metrics_for_run over content evaluate inputs; this is not an independent formal evaluate record.",
    }
    write_json_atomic(export_path, payload)
    return {
        "path": normalize_path_value(export_path),
        "payload": payload,
    }


def _build_finalize_manifest_payload(
    *,
    family_id: str,
    family_root: Path,
    stage_root: Path,
    summary_path: Path,
    family_manifest_path: Path,
    source_shard_plan_path: Path,
    source_split_plan_path: Path,
    source_merge_manifest_path: Path,
    positive_pool_manifest: Mapping[str, Any],
    clean_negative_pool_manifest: Mapping[str, Any],
    threshold_exports: Mapping[str, Dict[str, Any]],
    clean_evaluate_exports: Mapping[str, Dict[str, Any]],
    system_final_export: Mapping[str, Any],
    score_runs: Mapping[str, Dict[str, Any]],
    split_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the top-level PW02 finalize manifest payload.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        stage_root: PW02 stage root.
        summary_path: PW02 summary path.
        family_manifest_path: Family manifest path.
        source_shard_plan_path: Source shard plan path.
        source_split_plan_path: Source split plan path.
        source_merge_manifest_path: Source merge manifest path.
        positive_pool_manifest: Positive pool manifest summary.
        clean_negative_pool_manifest: Negative pool manifest summary.
        threshold_exports: Threshold export summaries keyed by score directory name.
        clean_evaluate_exports: Evaluate export summaries keyed by score directory name.
        system_final_export: System-final export summary.
        score_runs: Score-run summaries keyed by score name.
        split_counts: Split count summary.

    Returns:
        Finalize-manifest payload.
    """
    return {
        "artifact_type": "paper_workflow_pw02_finalize_manifest",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "pw02_stage_root": normalize_path_value(stage_root),
        "pw02_summary_path": normalize_path_value(summary_path),
        "family_manifest_path": normalize_path_value(family_manifest_path),
        "source_shard_plan_path": normalize_path_value(source_shard_plan_path),
        "source_split_plan_path": normalize_path_value(source_split_plan_path),
        "source_merge_manifest_path": normalize_path_value(source_merge_manifest_path),
        "split_counts": dict(split_counts),
        "source_pools": {
            ACTIVE_SAMPLE_ROLE: {
                "manifest_path": str(positive_pool_manifest.get("path")),
                "event_count": cast(Mapping[str, Any], positive_pool_manifest.get("payload", {})).get("event_count"),
            },
            CLEAN_NEGATIVE_SAMPLE_ROLE: {
                "manifest_path": str(clean_negative_pool_manifest.get("path")),
                "event_count": cast(Mapping[str, Any], clean_negative_pool_manifest.get("payload", {})).get("event_count"),
            },
        },
        "threshold_exports": {
            score_key: {
                "score_name": export_summary.get("score_name"),
                "path": export_summary.get("path"),
                "source_thresholds_artifact_path": cast(Mapping[str, Any], export_summary.get("payload", {})).get("source_thresholds_artifact_path"),
            }
            for score_key, export_summary in threshold_exports.items()
        },
        "clean_evaluate_exports": {
            score_key: {
                "score_name": export_summary.get("score_name"),
                "path": export_summary.get("path"),
                "source_evaluate_run_root": cast(Mapping[str, Any], export_summary.get("payload", {})).get("source_evaluate_run_root"),
                "source_evaluate_record_path": cast(Mapping[str, Any], export_summary.get("payload", {})).get("source_evaluate_record_path"),
            }
            for score_key, export_summary in clean_evaluate_exports.items()
        },
        "system_final": {
            "mode": "derived_metrics_from_content_evaluate_inputs",
            "is_formal_evaluate_record": False,
            "artifact_path": system_final_export.get("path"),
            "source_score_name": cast(Mapping[str, Any], system_final_export.get("payload", {})).get("source_score_name"),
            "source_evaluate_run_root": cast(Mapping[str, Any], system_final_export.get("payload", {})).get("source_evaluate_run_root"),
        },
        "score_runs": {
            score_name: {
                "calibrate_run_root": score_run.get("calibrate_run_root"),
                "evaluate_run_root": score_run.get("evaluate_run_root"),
                "thresholds_artifact_path": score_run.get("thresholds_artifact_path"),
                "evaluate_record_path": score_run.get("evaluate_record_path"),
            }
            for score_name, score_run in score_runs.items()
        },
    }


def _build_labelled_detect_payload(
    *,
    detect_payload: Dict[str, Any],
    family_id: str,
    sample_role: str,
    split_kind: str,
    event_id: str,
    event_index: int,
) -> Dict[str, Any]:
    """
    Build one labelled detect-record artifact copy for PW02.

    Args:
        detect_payload: Source detect-record payload.
        family_id: Family identifier.
        sample_role: Supported source sample role.
        split_kind: Split-kind token.
        event_id: Event identifier.
        event_index: Event index.

    Returns:
        Labelled detect-record copy.
    """
    if not isinstance(detect_payload, dict):
        raise TypeError("detect_payload must be dict")
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if split_kind not in {"calibration", "evaluate"}:
        raise ValueError("split_kind must be calibration or evaluate")
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(event_index, int) or event_index < 0:
        raise TypeError("event_index must be non-negative int")

    payload = eval_workflow_inputs._strip_forbidden_artifact_anchor_fields(copy.deepcopy(detect_payload))
    label_value = normalized_sample_role == ACTIVE_SAMPLE_ROLE
    payload["label"] = label_value
    payload["ground_truth"] = label_value
    payload["is_watermarked"] = label_value
    payload["sample_role"] = normalized_sample_role
    payload["paper_workflow_family_id"] = family_id
    payload["paper_workflow_event_id"] = event_id
    payload["paper_workflow_event_index"] = event_index
    payload["paper_workflow_split_kind"] = split_kind

    if label_value:
        payload["calibration_label_resolution"] = "paper_workflow_positive_source"
        payload["system_final_label_resolution"] = "paper_workflow_positive_source"
    else:
        eval_workflow_inputs._apply_negative_attestation_semantics(payload)
        payload["calibration_label_resolution"] = "paper_workflow_clean_negative"
        payload["system_final_label_resolution"] = "paper_workflow_clean_negative"
        payload["calibration_sample_usage"] = "paper_workflow_clean_negative_real_source"
    return payload


def _build_prepared_records(
    *,
    family_id: str,
    split_kind: str,
    event_ids: Sequence[str],
    sample_role: str,
    event_lookup: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build prepared detect-record copies for one split and sample role.

    Args:
        family_id: Family identifier.
        split_kind: Split-kind token.
        event_ids: Ordered event identifiers for the split.
        sample_role: Supported source sample role.
        event_lookup: Event manifest lookup by event_id.

    Returns:
        Prepared record entries.
    """
    prepared_records: List[Dict[str, Any]] = []
    normalized_sample_role = validate_source_sample_role(sample_role)
    for event_id in event_ids:
        if event_id not in event_lookup:
            raise ValueError(f"PW02 split references missing event_id: {event_id}")
        event_payload = event_lookup[event_id]
        detect_record_path_value = event_payload.get("detect_record_path")
        if not isinstance(detect_record_path_value, str) or not detect_record_path_value:
            raise ValueError(f"PW01 event manifest missing detect_record_path: {event_id}")
        detect_record_path = Path(detect_record_path_value).expanduser().resolve()
        detect_payload = _load_required_json_dict(detect_record_path, f"PW01 detect record {event_id}")

        event_index = event_payload.get("event_index")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError(f"PW01 event manifest missing event_index: {event_id}")

        labelled_payload = _build_labelled_detect_payload(
            detect_payload=detect_payload,
            family_id=family_id,
            sample_role=normalized_sample_role,
            split_kind=split_kind,
            event_id=event_id,
            event_index=event_index,
        )
        prepared_records.append(
            {
                "event_id": event_id,
                "event_index": event_index,
                "sample_role": normalized_sample_role,
                "split_kind": split_kind,
                "detect_record_path": normalize_path_value(detect_record_path),
                "payload": labelled_payload,
            }
        )
    return prepared_records


def _write_score_split_records(
    *,
    score_name: str,
    records_root: Path,
    prepared_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Write score-filtered detect-record copies for one split.

    Args:
        score_name: Requested score name.
        records_root: Output directory for copied detect records.
        prepared_records: Prepared record entries.

    Returns:
        Summary payload with output paths and score sources.
    """
    if not isinstance(records_root, Path):
        raise TypeError("records_root must be Path")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    ensure_directory(records_root)
    written_paths: List[str] = []
    record_summaries: List[Dict[str, Any]] = []

    for ordinal, prepared_record in enumerate(prepared_records):
        payload_node = prepared_record.get("payload")
        if not isinstance(payload_node, dict):
            raise ValueError("prepared_record payload must be dict")
        payload = copy.deepcopy(cast(Dict[str, Any], payload_node))
        score_value, score_source = _resolve_score_value(payload, score_name)
        if score_value is None or not isinstance(score_source, str) or not score_source:
            raise ValueError(
                f"PW02 record missing {score_name}: event_id={prepared_record.get('event_id')}, sample_role={prepared_record.get('sample_role')}"
            )
        payload["paper_workflow_score_name"] = score_name
        payload["paper_workflow_score_source"] = score_source

        event_id = str(prepared_record.get("event_id"))
        sample_role = str(prepared_record.get("sample_role"))
        event_index = int(prepared_record.get("event_index", ordinal))
        role_token = "pos" if sample_role == ACTIVE_SAMPLE_ROLE else "neg"
        target_path = records_root / f"{ordinal:04d}_{role_token}_e{event_index:06d}.json"
        write_json_atomic(target_path, payload)
        written_paths.append(normalize_path_value(target_path))
        record_summaries.append(
            {
                "event_id": event_id,
                "sample_role": sample_role,
                "record_path": normalize_path_value(target_path),
                "score_name": score_name,
                "score_source": score_source,
                "score_value": float(score_value),
            }
        )

    return {
        "score_name": score_name,
        "record_count": len(record_summaries),
        "records_glob": str((records_root / "*.json").resolve()),
        "record_paths": written_paths,
        "records": record_summaries,
    }


def _build_score_runtime_config(
    *,
    base_cfg_obj: Mapping[str, Any],
    score_name: str,
    calibration_records_glob: str,
    evaluate_records_glob: str,
    thresholds_path: Path,
) -> Dict[str, Any]:
    """
    Build one score-specific runtime config for calibrate and evaluate.

    Args:
        base_cfg_obj: Base config mapping.
        score_name: Requested score name.
        calibration_records_glob: Calibration detect-record glob.
        evaluate_records_glob: Evaluate detect-record glob.
        thresholds_path: Expected thresholds artifact path.

    Returns:
        Score-specific runtime config mapping.
    """
    if not isinstance(base_cfg_obj, Mapping):
        raise TypeError("base_cfg_obj must be Mapping")
    cfg_obj = copy.deepcopy(dict(base_cfg_obj))
    cfg_obj["allow_nonempty_run_root"] = True
    cfg_obj["allow_nonempty_run_root_reason"] = "paper_workflow_pw02_prepared_inputs"

    calibration_cfg = copy.deepcopy(cfg_obj.get("calibration")) if isinstance(cfg_obj.get("calibration"), dict) else {}
    calibration_cfg["detect_records_glob"] = calibration_records_glob
    calibration_cfg["score_name"] = score_name
    calibration_cfg["allow_synthetic_minimal_ground_truth"] = False
    cfg_obj["calibration"] = calibration_cfg

    evaluate_cfg = copy.deepcopy(cfg_obj.get("evaluate")) if isinstance(cfg_obj.get("evaluate"), dict) else {}
    evaluate_cfg["detect_records_glob"] = evaluate_records_glob
    evaluate_cfg["score_name"] = score_name
    evaluate_cfg["thresholds_path"] = normalize_path_value(thresholds_path)
    evaluate_cfg["allow_synthetic_minimal_ground_truth"] = False
    cfg_obj["evaluate"] = evaluate_cfg
    return cfg_obj


def _run_python_stage_command(
    *,
    module_name: str,
    output_dir: Path,
    config_path: Path,
    log_prefix: str,
) -> Dict[str, Any]:
    """
    Run one Python CLI stage with explicit stdout and stderr logs.

    Args:
        module_name: Python module name.
        output_dir: Stage output directory.
        config_path: Runtime config path.
        log_prefix: Log-file prefix.

    Returns:
        Command execution summary.
    """
    if not isinstance(module_name, str) or not module_name:
        raise TypeError("module_name must be non-empty str")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be Path")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(log_prefix, str) or not log_prefix:
        raise TypeError("log_prefix must be non-empty str")

    logs_root = ensure_directory(output_dir / "logs")
    command = [
        sys.executable,
        "-m",
        module_name,
        "--out",
        str(output_dir),
        "--config",
        str(config_path),
    ]
    result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=logs_root / f"{log_prefix}_stdout.log",
        stderr_log_path=logs_root / f"{log_prefix}_stderr.log",
    )
    result["status"] = "ok" if int(result.get("return_code", 1)) == 0 else "failed"
    return result


def _run_score_pipeline(
    *,
    score_name: str,
    stage_root: Path,
    base_cfg_obj: Mapping[str, Any],
    calibration_records: Sequence[Mapping[str, Any]],
    evaluate_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Run PW02 calibrate and evaluate stages for one score family.

    Args:
        score_name: Requested score name.
        stage_root: PW02 stage root.
        base_cfg_obj: Base config mapping.
        calibration_records: Calibration split records.
        evaluate_records: Evaluate split records.

    Returns:
        Score-pipeline summary payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")

    score_directory_name = "content" if eval_metrics.is_content_chain_score_name(score_name) else "attest"
    score_root = ensure_directory(stage_root / "score_runs" / score_directory_name)
    calibrate_root = ensure_directory(score_root / "cal")
    evaluate_root = ensure_directory(score_root / "eval")

    calibration_records_root = ensure_directory(calibrate_root / "artifacts" / "calibration_inputs" / "formal_calibration_records")
    evaluate_records_root = ensure_directory(evaluate_root / "artifacts" / "evaluate_inputs" / "formal_evaluate_records")

    calibration_records_summary = _write_score_split_records(
        score_name=score_name,
        records_root=calibration_records_root,
        prepared_records=calibration_records,
    )
    evaluate_records_summary = _write_score_split_records(
        score_name=score_name,
        records_root=evaluate_records_root,
        prepared_records=evaluate_records,
    )

    thresholds_artifact_path = evaluate_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    runtime_cfg_obj = _build_score_runtime_config(
        base_cfg_obj=base_cfg_obj,
        score_name=score_name,
        calibration_records_glob=str((calibration_records_root / "*.json").resolve()),
        evaluate_records_glob=str((evaluate_records_root / "*.json").resolve()),
        thresholds_path=thresholds_artifact_path,
    )
    runtime_cfg_path = score_root / "runtime_config.yaml"
    write_yaml_mapping(runtime_cfg_path, runtime_cfg_obj)

    calibrate_result = _run_python_stage_command(
        module_name="main.cli.run_calibrate",
        output_dir=calibrate_root,
        config_path=runtime_cfg_path,
        log_prefix="pw02_calibrate",
    )
    if int(calibrate_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW02 calibrate failed for {score_name}: {json.dumps(calibrate_result, ensure_ascii=False, sort_keys=True)}"
        )

    thresholds_artifact_path = calibrate_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    runtime_cfg_obj = _build_score_runtime_config(
        base_cfg_obj=base_cfg_obj,
        score_name=score_name,
        calibration_records_glob=str((calibration_records_root / "*.json").resolve()),
        evaluate_records_glob=str((evaluate_records_root / "*.json").resolve()),
        thresholds_path=thresholds_artifact_path,
    )
    write_yaml_mapping(runtime_cfg_path, runtime_cfg_obj)

    evaluate_result = _run_python_stage_command(
        module_name="main.cli.run_evaluate",
        output_dir=evaluate_root,
        config_path=runtime_cfg_path,
        log_prefix="pw02_evaluate",
    )
    if int(evaluate_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW02 evaluate failed for {score_name}: {json.dumps(evaluate_result, ensure_ascii=False, sort_keys=True)}"
        )

    return {
        "score_name": score_name,
        "runtime_config_path": normalize_path_value(runtime_cfg_path),
        "calibration_inputs": calibration_records_summary,
        "evaluate_inputs": evaluate_records_summary,
        "calibrate_run_root": normalize_path_value(calibrate_root),
        "evaluate_run_root": normalize_path_value(evaluate_root),
        "calibrate_result": calibrate_result,
        "evaluate_result": evaluate_result,
        "thresholds_artifact_path": normalize_path_value(thresholds_artifact_path),
        "calibration_record_path": normalize_path_value(calibrate_root / "records" / "calibration_record.json"),
        "evaluate_record_path": normalize_path_value(evaluate_root / "records" / "evaluate_record.json"),
        "evaluation_report_path": normalize_path_value(evaluate_root / "artifacts" / "evaluation_report.json"),
    }


def run_pw02_merge_source_event_shards(
    *,
    drive_project_root: Path,
    family_id: str,
) -> Dict[str, Any]:
    """
    Merge PW01 source shards and execute PW02 threshold workflows.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.

    Returns:
        PW02 summary payload.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    layout = resolve_family_layout_paths(family_root)

    family_manifest = _load_required_json_dict(layout["family_manifest_path"], "paper eval family manifest")
    source_shard_plan = _load_required_json_dict(layout["source_shard_plan_path"], "source shard plan")
    source_split_plan = _load_required_json_dict(layout["source_split_plan_path"], "source split plan")

    base_cfg_path = layout["config_snapshot_path"]
    if not base_cfg_path.exists() or not base_cfg_path.is_file():
        raise FileNotFoundError(f"PW02 config snapshot missing: {normalize_path_value(base_cfg_path)}")
    base_cfg_obj = load_yaml_mapping(base_cfg_path)

    positive_events = _collect_completed_events_for_role(
        family_root=family_root,
        source_shard_plan=source_shard_plan,
        sample_role=ACTIVE_SAMPLE_ROLE,
    )
    clean_negative_events = _collect_completed_events_for_role(
        family_root=family_root,
        source_shard_plan=source_shard_plan,
        sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
    )

    calibration_records = _build_prepared_records(
        family_id=family_id,
        split_kind="calibration",
        event_ids=cast(List[str], source_split_plan.get("calib_pos_event_ids", [])),
        sample_role=ACTIVE_SAMPLE_ROLE,
        event_lookup=positive_events,
    )
    calibration_records.extend(
        _build_prepared_records(
            family_id=family_id,
            split_kind="calibration",
            event_ids=cast(List[str], source_split_plan.get("calib_neg_event_ids", [])),
            sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
            event_lookup=clean_negative_events,
        )
    )
    evaluate_records = _build_prepared_records(
        family_id=family_id,
        split_kind="evaluate",
        event_ids=cast(List[str], source_split_plan.get("eval_pos_event_ids", [])),
        sample_role=ACTIVE_SAMPLE_ROLE,
        event_lookup=positive_events,
    )
    evaluate_records.extend(
        _build_prepared_records(
            family_id=family_id,
            split_kind="evaluate",
            event_ids=cast(List[str], source_split_plan.get("eval_neg_event_ids", [])),
            sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
            event_lookup=clean_negative_events,
        )
    )

    stage_root = ensure_directory(layout["exports_root"] / "pw02")
    validate_path_within_base(family_root, stage_root, "PW02 stage root")
    manifests_root = ensure_directory(stage_root / "manifests")

    merge_manifest_path = manifests_root / "source_merge_manifest.json"
    merge_manifest_payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_source_merge_manifest",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "family_manifest_path": normalize_path_value(layout["family_manifest_path"]),
        "source_shard_plan_path": normalize_path_value(layout["source_shard_plan_path"]),
        "source_split_plan_path": normalize_path_value(layout["source_split_plan_path"]),
        "split_counts": {
            "calibration": len(calibration_records),
            "evaluate": len(evaluate_records),
        },
        "split_event_ids": {
            "calibration": [str(record["event_id"]) for record in calibration_records],
            "evaluate": [str(record["event_id"]) for record in evaluate_records],
        },
    }
    write_json_atomic(merge_manifest_path, merge_manifest_payload)

    positive_pool_manifest = _write_source_pool_manifest(
        stage_root=stage_root,
        family_id=family_id,
        sample_role=ACTIVE_SAMPLE_ROLE,
        event_lookup=positive_events,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
    )
    clean_negative_pool_manifest = _write_source_pool_manifest(
        stage_root=stage_root,
        family_id=family_id,
        sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
        event_lookup=clean_negative_events,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
    )

    score_runs = {
        CONTENT_SCORE_NAME: _run_score_pipeline(
            score_name=CONTENT_SCORE_NAME,
            stage_root=stage_root,
            base_cfg_obj=base_cfg_obj,
            calibration_records=calibration_records,
            evaluate_records=evaluate_records,
        ),
        EVENT_ATTESTATION_SCORE_NAME: _run_score_pipeline(
            score_name=EVENT_ATTESTATION_SCORE_NAME,
            stage_root=stage_root,
            base_cfg_obj=base_cfg_obj,
            calibration_records=calibration_records,
            evaluate_records=evaluate_records,
        ),
    }

    content_evaluate_root = Path(str(score_runs[CONTENT_SCORE_NAME]["evaluate_run_root"]))
    system_final_metrics = _build_system_final_metrics_for_run(content_evaluate_root)

    threshold_exports = {
        _resolve_top_level_score_directory_name(score_name): _build_threshold_export(
            family_id=family_id,
            stage_root=stage_root,
            score_name=score_name,
            score_run=score_run,
        )
        for score_name, score_run in score_runs.items()
    }
    clean_evaluate_exports = {
        _resolve_top_level_score_directory_name(score_name): _build_clean_evaluate_export(
            family_id=family_id,
            stage_root=stage_root,
            score_name=score_name,
            score_run=score_run,
        )
        for score_name, score_run in score_runs.items()
    }
    system_final_export = _build_system_final_metrics_export(
        family_id=family_id,
        stage_root=stage_root,
        content_score_run=score_runs[CONTENT_SCORE_NAME],
        system_final_metrics=system_final_metrics,
    )

    summary_path = layout["runtime_state_root"] / "pw02_summary.json"
    finalize_manifest_path = stage_root / FINALIZE_MANIFEST_FILE_NAME
    finalize_manifest_payload = _build_finalize_manifest_payload(
        family_id=family_id,
        family_root=family_root,
        stage_root=stage_root,
        summary_path=summary_path,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
        source_merge_manifest_path=merge_manifest_path,
        positive_pool_manifest=positive_pool_manifest,
        clean_negative_pool_manifest=clean_negative_pool_manifest,
        threshold_exports=threshold_exports,
        clean_evaluate_exports=clean_evaluate_exports,
        system_final_export=system_final_export,
        score_runs=score_runs,
        split_counts=cast(Mapping[str, Any], merge_manifest_payload["split_counts"]),
    )
    write_json_atomic(finalize_manifest_path, finalize_manifest_payload)
    summary: Dict[str, Any] = {
        "status": "ok",
        "stage_name": "PW02_Source_Merge_And_Global_Thresholds",
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "summary_path": normalize_path_value(summary_path),
        "pw02_stage_root": normalize_path_value(stage_root),
        "source_merge_manifest_path": normalize_path_value(merge_manifest_path),
        "positive_source_pool_manifest_path": str(positive_pool_manifest["path"]),
        "clean_negative_pool_manifest_path": str(clean_negative_pool_manifest["path"]),
        "paper_source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
        "family_manifest_path": normalize_path_value(layout["family_manifest_path"]),
        "source_shard_plan_path": normalize_path_value(layout["source_shard_plan_path"]),
        "source_split_plan_path": normalize_path_value(layout["source_split_plan_path"]),
        "score_runs": score_runs,
        "threshold_exports": {
            score_key: export_summary["path"]
            for score_key, export_summary in threshold_exports.items()
        },
        "clean_evaluate_exports": {
            score_key: export_summary["path"]
            for score_key, export_summary in clean_evaluate_exports.items()
        },
        "system_final_metrics": system_final_metrics,
        "system_final_semantics": "derived_metrics_from_content_evaluate_inputs",
        "system_final_metrics_artifact_path": str(system_final_export["path"]),
        "split_counts": merge_manifest_payload["split_counts"],
    }
    write_json_atomic(summary_path, summary)
    return summary