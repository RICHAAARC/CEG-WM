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
from main.evaluation.experiment_matrix import (
    _build_formal_final_decision_metrics_for_run,
    _build_system_final_metrics_for_run,
)
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
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
    build_family_root,
    build_source_shard_root,
    write_jsonl,
    resolve_family_layout_paths,
    validate_source_sample_role,
)
from paper_workflow.scripts.pw_quality_metrics import (
    resolve_optional_artifact_path,
    resolve_preview_persisted_artifact_path,
)
from paper_workflow.scripts.pw02_metrics_extensions import build_pw02_metrics_extensions


CONTENT_SCORE_NAME = eval_metrics.CONTENT_CHAIN_SCORE_NAME
EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
PW02_RUN_ROOT_REUSE_REASON = "paper_workflow_pw02_prepared_inputs"
FORMAL_REQUIRED_SOURCE_POOL = "formal_required"
OPTIONAL_DIAGNOSTIC_SOURCE_POOL = "optional_diagnostic"
SOURCE_POOL_STATUS_COMPLETED = "completed"
SOURCE_POOL_STATUS_NOT_PROVIDED = "not_provided"
POOL_MANIFEST_FILE_NAMES = {
    ACTIVE_SAMPLE_ROLE: "positive_source_pool_manifest.json",
    CLEAN_NEGATIVE_SAMPLE_ROLE: "clean_negative_pool_manifest.json",
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: "planner_conditioned_control_negative_pool_manifest.json",
}
FINALIZE_MANIFEST_FILE_NAME = "paper_source_finalize_manifest.json"
FORMAL_FINAL_DECISION_METRICS_FILE_NAME = "formal_final_decision_metrics.json"
DERIVED_SYSTEM_UNION_METRICS_FILE_NAME = "derived_system_union_metrics.json"
CLEAN_SCORE_ANALYSIS_FILE_NAME = "clean_score_analysis.json"
CLEAN_QUALITY_PAIR_MANIFEST_FILE_NAME = "clean_quality_pair_manifest.json"
CLEAN_EVENT_TABLE_FILE_NAME = "clean_event_table.jsonl"


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


def _normalize_role_order(node: Any, *, label: str) -> List[str]:
    """
    Normalize one score-pool role order.

    Args:
        node: Candidate role-order node.
        label: Error label.

    Returns:
        Normalized role order.
    """
    if not isinstance(node, list) or not node:
        raise ValueError(f"{label} must be non-empty list[str]")

    normalized_role_order: List[str] = []
    for raw_role in cast(List[object], node):
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ValueError(f"{label} entries must be non-empty str")
        normalized_role = validate_source_sample_role(raw_role.strip())
        if normalized_role not in normalized_role_order:
            normalized_role_order.append(normalized_role)
    return normalized_role_order


def _normalize_role_event_id_keys(node: Any, *, label: str) -> Dict[str, str]:
    """
    Normalize one score-pool event-id key mapping.

    Args:
        node: Candidate mapping.
        label: Error label.

    Returns:
        Normalized mapping from sample_role to split-plan key.
    """
    if not isinstance(node, Mapping):
        raise ValueError(f"{label} must be mapping[str, str]")

    normalized_mapping: Dict[str, str] = {}
    for raw_role, raw_split_key in cast(Mapping[Any, Any], node).items():
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ValueError(f"{label} keys must be non-empty str")
        if not isinstance(raw_split_key, str) or not raw_split_key.strip():
            raise ValueError(f"{label} values must be non-empty str")
        normalized_mapping[validate_source_sample_role(raw_role.strip())] = raw_split_key.strip()
    return normalized_mapping


def _default_score_pool_spec(score_name: str) -> Dict[str, Any]:
    """
    Build the default PW02 score-pool specification.

    Args:
        score_name: Canonical score name.

    Returns:
        Default score-pool spec.
    """
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
    if score_name not in {CONTENT_SCORE_NAME, EVENT_ATTESTATION_SCORE_NAME}:
        raise ValueError(f"unsupported score_name: {score_name}")
    return {
        "calibration_role_order": [ACTIVE_SAMPLE_ROLE, CLEAN_NEGATIVE_SAMPLE_ROLE],
        "calibration_event_id_keys": {
            ACTIVE_SAMPLE_ROLE: "calib_pos_event_ids",
            CLEAN_NEGATIVE_SAMPLE_ROLE: "calib_neg_event_ids",
        },
        "evaluate_role_order": [ACTIVE_SAMPLE_ROLE, CLEAN_NEGATIVE_SAMPLE_ROLE],
        "evaluate_event_id_keys": {
            ACTIVE_SAMPLE_ROLE: "eval_pos_event_ids",
            CLEAN_NEGATIVE_SAMPLE_ROLE: "eval_neg_event_ids",
        },
    }


def _resolve_benchmark_protocol_bundle(family_manifest: Mapping[str, Any]) -> Dict[str, Any] | None:
    """
    Resolve the optional shared benchmark protocol bundle from PW00 outputs.

    Args:
        family_manifest: Family manifest payload.

    Returns:
        Benchmark protocol bundle or None.
    """
    if not isinstance(family_manifest, Mapping):
        raise TypeError("family_manifest must be Mapping")

    benchmark_protocol_node = family_manifest.get("benchmark_protocol")
    benchmark_protocol_payload = (
        dict(cast(Mapping[str, Any], benchmark_protocol_node))
        if isinstance(benchmark_protocol_node, Mapping)
        else None
    )
    benchmark_protocol_config_path_value = family_manifest.get("benchmark_protocol_config_path")
    if benchmark_protocol_payload is None:
        if not isinstance(benchmark_protocol_config_path_value, str) or not benchmark_protocol_config_path_value.strip():
            return None
        benchmark_protocol_payload = load_yaml_mapping(
            Path(benchmark_protocol_config_path_value).expanduser().resolve()
        )
        if not isinstance(benchmark_protocol_payload, dict):
            raise TypeError("benchmark protocol config must load as dict")

    benchmark_provenance_node = family_manifest.get("benchmark_provenance")
    benchmark_provenance = (
        copy.deepcopy(dict(cast(Mapping[str, Any], benchmark_provenance_node)))
        if isinstance(benchmark_provenance_node, Mapping)
        else {}
    )
    if isinstance(benchmark_protocol_config_path_value, str) and benchmark_protocol_config_path_value.strip():
        benchmark_provenance.setdefault(
            "benchmark_protocol_config_path",
            benchmark_protocol_config_path_value,
        )

    return {
        "config_path": benchmark_provenance.get("benchmark_protocol_config_path"),
        "protocol": copy.deepcopy(benchmark_protocol_payload),
        "provenance": benchmark_provenance,
    }


def _resolve_score_pool_spec(
    *,
    score_name: str,
    benchmark_protocol_bundle: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    Resolve one score-pool spec from the optional benchmark protocol.

    Args:
        score_name: Canonical score name.
        benchmark_protocol_bundle: Optional benchmark protocol bundle.

    Returns:
        Normalized score-pool spec.
    """
    default_score_pool_spec = _default_score_pool_spec(score_name)
    if not isinstance(benchmark_protocol_bundle, Mapping):
        return default_score_pool_spec

    benchmark_protocol = benchmark_protocol_bundle.get("protocol")
    if not isinstance(benchmark_protocol, Mapping):
        return default_score_pool_spec

    score_pools_node = benchmark_protocol.get("score_pools")
    if not isinstance(score_pools_node, Mapping):
        raise ValueError("benchmark_protocol.score_pools must be mapping")

    score_pool_node = score_pools_node.get(score_name)
    if not isinstance(score_pool_node, Mapping):
        raise ValueError(f"benchmark protocol missing score_pools.{score_name}")

    return {
        "calibration_role_order": _normalize_role_order(
            score_pool_node.get("calibration_role_order"),
            label=f"benchmark_protocol.score_pools.{score_name}.calibration_role_order",
        ),
        "calibration_event_id_keys": _normalize_role_event_id_keys(
            score_pool_node.get("calibration_event_id_keys"),
            label=f"benchmark_protocol.score_pools.{score_name}.calibration_event_id_keys",
        ),
        "evaluate_role_order": _normalize_role_order(
            score_pool_node.get("evaluate_role_order"),
            label=f"benchmark_protocol.score_pools.{score_name}.evaluate_role_order",
        ),
        "evaluate_event_id_keys": _normalize_role_event_id_keys(
            score_pool_node.get("evaluate_event_id_keys"),
            label=f"benchmark_protocol.score_pools.{score_name}.evaluate_event_id_keys",
        ),
    }


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


def _collect_optional_completed_events_for_role(
    *,
    family_root: Path,
    source_shard_plan: Mapping[str, Any],
    sample_role: str,
) -> Dict[str, Any]:
    """
    Collect one optional source cohort with 0/N, N/N, and partial-state semantics.

    Args:
        family_root: Family root path.
        source_shard_plan: Source shard plan payload.
        sample_role: Supported optional source sample role.

    Returns:
        Optional cohort summary with collection status and event lookup.

    Raises:
        RuntimeError: If the optional cohort is partially provided.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    normalized_sample_role = validate_source_sample_role(sample_role)
    role_plan = _resolve_role_plan(source_shard_plan, normalized_sample_role)

    shards_node = role_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError(f"source shard plan {normalized_sample_role}.shards must be list")

    expected_shard_indices: List[int] = []
    discovered_shard_indices: List[int] = []
    missing_shard_indices: List[int] = []
    for shard_row_node in cast(List[object], shards_node):
        if not isinstance(shard_row_node, dict):
            raise ValueError("source shard plan shards must contain objects")
        shard_row = cast(Dict[str, Any], shard_row_node)
        shard_index = shard_row.get("shard_index")
        if not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError("source shard row missing shard_index")
        expected_shard_indices.append(shard_index)

        shard_root = build_source_shard_root(family_root, normalized_sample_role, shard_index)
        shard_manifest_path = shard_root / "shard_manifest.json"
        if shard_manifest_path.exists() and shard_manifest_path.is_file():
            discovered_shard_indices.append(shard_index)
        else:
            missing_shard_indices.append(shard_index)

    expected_source_shard_count = len(expected_shard_indices)
    discovered_source_shard_count = len(discovered_shard_indices)
    if discovered_source_shard_count == 0:
        return {
            "cohort_status": SOURCE_POOL_STATUS_NOT_PROVIDED,
            "role_requirement": OPTIONAL_DIAGNOSTIC_SOURCE_POOL,
            "event_lookup": {},
            "expected_source_shard_count": expected_source_shard_count,
            "discovered_source_shard_count": 0,
            "missing_source_shard_indices": missing_shard_indices,
        }

    if discovered_source_shard_count < expected_source_shard_count:
        raise RuntimeError(
            "PW02 optional diagnostic cohort is partially provided: "
            f"sample_role={normalized_sample_role}, "
            f"discovered_source_shard_count={discovered_source_shard_count}, "
            f"expected_source_shard_count={expected_source_shard_count}, "
            f"missing_source_shard_indices={missing_shard_indices}"
        )

    return {
        "cohort_status": SOURCE_POOL_STATUS_COMPLETED,
        "role_requirement": OPTIONAL_DIAGNOSTIC_SOURCE_POOL,
        "event_lookup": _collect_completed_events_for_role(
            family_root=family_root,
            source_shard_plan=source_shard_plan,
            sample_role=normalized_sample_role,
        ),
        "expected_source_shard_count": expected_source_shard_count,
        "discovered_source_shard_count": discovered_source_shard_count,
        "missing_source_shard_indices": [],
    }


def _build_source_pool_manifest_payload(
    *,
    family_id: str,
    sample_role: str,
    event_lookup: Mapping[str, Dict[str, Any]],
    family_manifest_path: Path,
    source_shard_plan_path: Path,
    source_split_plan_path: Path,
    stage_root: Path,
    role_requirement: str = FORMAL_REQUIRED_SOURCE_POOL,
    cohort_status: str = SOURCE_POOL_STATUS_COMPLETED,
    expected_source_shard_count: int | None = None,
    discovered_source_shard_count: int | None = None,
    missing_source_shard_indices: Sequence[int] | None = None,
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
    expected_shard_count = (
        expected_source_shard_count
        if isinstance(expected_source_shard_count, int) and expected_source_shard_count >= 0
        else len(ordered_events)
    )
    discovered_shard_count = (
        discovered_source_shard_count
        if isinstance(discovered_source_shard_count, int) and discovered_source_shard_count >= 0
        else len(ordered_events)
    )
    missing_shard_indices = [
        int(index)
        for index in cast(Sequence[int], missing_source_shard_indices or [])
    ]

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
                "payload_reference_sidecar_path": event_payload.get("payload_reference_sidecar_path"),
                "payload_decode_sidecar_path": event_payload.get("payload_decode_sidecar_path"),
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
        "role_requirement": role_requirement,
        "cohort_status": cohort_status,
        "diagnostic_only": role_requirement == OPTIONAL_DIAGNOSTIC_SOURCE_POOL,
        "event_count": len(manifest_events),
        "event_ids": [event_payload["event_id"] for event_payload in manifest_events],
        "events": manifest_events,
        "source_shard_count": len(ordered_shards),
        "expected_source_shard_count": expected_shard_count,
        "discovered_source_shard_count": discovered_shard_count,
        "missing_source_shard_indices": missing_shard_indices,
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
    role_requirement: str = FORMAL_REQUIRED_SOURCE_POOL,
    cohort_status: str = SOURCE_POOL_STATUS_COMPLETED,
    expected_source_shard_count: int | None = None,
    discovered_source_shard_count: int | None = None,
    missing_source_shard_indices: Sequence[int] | None = None,
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
        role_requirement=role_requirement,
        cohort_status=cohort_status,
        expected_source_shard_count=expected_source_shard_count,
        discovered_source_shard_count=discovered_source_shard_count,
        missing_source_shard_indices=missing_source_shard_indices,
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
    benchmark_provenance = score_run.get("benchmark_provenance")
    if isinstance(benchmark_provenance, Mapping):
        payload["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], benchmark_provenance)))
    score_pool = score_run.get("score_pool")
    if isinstance(score_pool, Mapping):
        payload["score_pool"] = copy.deepcopy(dict(cast(Mapping[str, Any], score_pool)))
    write_json_atomic(export_path, payload)
    return {
        "score_name": score_name,
        "path": normalize_path_value(export_path),
        "payload": payload,
    }


def _extract_threshold_value_from_export_summary(threshold_export_summary: Mapping[str, Any]) -> float:
    """
    Extract one finite threshold value from the PW02 export summary.

    Args:
        threshold_export_summary: Threshold export summary mapping.

    Returns:
        Threshold value.
    """
    if not isinstance(threshold_export_summary, Mapping):
        raise TypeError("threshold_export_summary must be Mapping")

    payload = cast(Mapping[str, Any], threshold_export_summary.get("payload", {}))
    thresholds_artifact = cast(Mapping[str, Any], payload.get("thresholds_artifact", {}))
    threshold_value = thresholds_artifact.get("threshold_value")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise ValueError("PW02 threshold export missing finite thresholds_artifact.threshold_value")
    return float(threshold_value)


def _count_records_by_role(records_summary: Mapping[str, Any]) -> Dict[str, int]:
    """
    Count prepared records by sample role.

    Args:
        records_summary: Score-run records summary.

    Returns:
        Sample-role counts.
    """
    counts: Dict[str, int] = {}
    records_node = records_summary.get("records")
    if not isinstance(records_node, list):
        return counts
    for record_node in cast(List[object], records_node):
        if not isinstance(record_node, Mapping):
            continue
        sample_role = record_node.get("sample_role")
        if isinstance(sample_role, str) and sample_role:
            counts[sample_role] = counts.get(sample_role, 0) + 1
    return counts


def _build_clean_evaluate_export(
    *,
    family_id: str,
    stage_root: Path,
    score_name: str,
    score_run: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build one top-level clean evaluate export from the real evaluate outputs.

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
    benchmark_provenance = score_run.get("benchmark_provenance")
    if isinstance(benchmark_provenance, Mapping):
        payload["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], benchmark_provenance)))
    score_pool = score_run.get("score_pool")
    if isinstance(score_pool, Mapping):
        payload["score_pool"] = copy.deepcopy(dict(cast(Mapping[str, Any], score_pool)))
    write_json_atomic(export_path, payload)
    return {
        "score_name": score_name,
        "path": normalize_path_value(export_path),
        "payload": payload,
    }


def _load_records_from_records_summary(records_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Load prepared record payloads from one records-summary mapping.

    Args:
        records_summary: Records summary produced by _write_score_split_records.

    Returns:
        Prepared record payloads.
    """
    if not isinstance(records_summary, Mapping):
        raise TypeError("records_summary must be Mapping")

    records_node = records_summary.get("records")
    if not isinstance(records_node, list):
        raise ValueError("records_summary.records must be list")

    payloads: List[Dict[str, Any]] = []
    for record_node in cast(List[object], records_node):
        if not isinstance(record_node, Mapping):
            raise ValueError("records_summary.records items must be mappings")
        record_path_value = record_node.get("record_path")
        if not isinstance(record_path_value, str) or not record_path_value.strip():
            raise ValueError("records_summary.records record_path must be non-empty str")
        payloads.append(_load_required_json_dict(Path(record_path_value).expanduser().resolve(), "PW02 prepared evaluate record"))
    return payloads


def _extract_evaluate_metrics_payload(evaluate_record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract the stable metrics payload from one evaluate record.

    Args:
        evaluate_record: Evaluate record payload.

    Returns:
        Stable metrics mapping.
    """
    if not isinstance(evaluate_record, Mapping):
        raise TypeError("evaluate_record must be Mapping")

    metrics_node = evaluate_record.get("metrics")
    if isinstance(metrics_node, Mapping):
        return dict(cast(Mapping[str, Any], metrics_node))

    evaluation_report_node = evaluate_record.get("evaluation_report")
    if isinstance(evaluation_report_node, Mapping):
        report_metrics_node = evaluation_report_node.get("metrics")
        if isinstance(report_metrics_node, Mapping):
            return dict(cast(Mapping[str, Any], report_metrics_node))

    return {}


def _extract_evaluate_breakdown_payload(evaluate_record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract the stable breakdown payload from one evaluate record.

    Args:
        evaluate_record: Evaluate record payload.

    Returns:
        Stable breakdown mapping.
    """
    if not isinstance(evaluate_record, Mapping):
        raise TypeError("evaluate_record must be Mapping")

    for key_name in ["evaluation_breakdown", "breakdown"]:
        breakdown_node = evaluate_record.get(key_name)
        if isinstance(breakdown_node, Mapping):
            return dict(cast(Mapping[str, Any], breakdown_node))

    evaluation_report_node = evaluate_record.get("evaluation_report")
    if isinstance(evaluation_report_node, Mapping):
        report_breakdown_node = evaluation_report_node.get("breakdown")
        if isinstance(report_breakdown_node, Mapping):
            return dict(cast(Mapping[str, Any], report_breakdown_node))

    return {}


def _safe_resolve_source_image_path(event_payload: Mapping[str, Any]) -> str | None:
    """
    Resolve the clean source-image path from one PW01 event payload.

    Args:
        event_payload: PW01 event payload surfaced via shard manifest.

    Returns:
        Normalized source image path or None.
    """
    if not isinstance(event_payload, Mapping):
        raise TypeError("event_payload must be Mapping")

    source_image_view = event_payload.get("source_image")
    if not isinstance(source_image_view, Mapping):
        return None
    try:
        source_path = resolve_optional_artifact_path(cast(Mapping[str, Any], source_image_view), "source_image")
    except Exception:
        return None
    if source_path is None:
        return None
    return normalize_path_value(source_path)


def _safe_resolve_preview_artifact_path(event_payload: Mapping[str, Any]) -> str | None:
    """
    Resolve the clean preview artifact path from one PW01 event payload.

    Args:
        event_payload: PW01 event payload surfaced via shard manifest.

    Returns:
        Normalized preview artifact path or None.
    """
    if not isinstance(event_payload, Mapping):
        raise TypeError("event_payload must be Mapping")

    plain_preview_view = event_payload.get("plain_preview_image")
    if isinstance(plain_preview_view, Mapping):
        try:
            plain_preview_path = resolve_optional_artifact_path(
                cast(Mapping[str, Any], plain_preview_view),
                "plain_preview_image",
            )
        except Exception:
            plain_preview_path = None
        if plain_preview_path is not None:
            return normalize_path_value(plain_preview_path)

    preview_record_view = event_payload.get("preview_generation_record")
    if not isinstance(preview_record_view, Mapping):
        return None
    try:
        preview_artifact_path = resolve_preview_persisted_artifact_path(cast(Mapping[str, Any], preview_record_view))
    except Exception:
        return None
    if preview_artifact_path is None:
        return None
    return normalize_path_value(preview_artifact_path)


def _safe_resolve_watermarked_image_path(event_payload: Mapping[str, Any]) -> str | None:
    """
    Resolve the finalized watermarked image path from one PW01 event payload.

    Args:
        event_payload: PW01 event payload surfaced via shard manifest.

    Returns:
        Normalized watermarked image path or None.
    """
    if not isinstance(event_payload, Mapping):
        raise TypeError("event_payload must be Mapping")

    watermarked_output_view = event_payload.get("watermarked_output_image")
    if isinstance(watermarked_output_view, Mapping):
        try:
            watermarked_output_path = resolve_optional_artifact_path(
                cast(Mapping[str, Any], watermarked_output_view),
                "watermarked_output_image",
            )
        except Exception:
            watermarked_output_path = None
        if watermarked_output_path is not None:
            return normalize_path_value(watermarked_output_path)

    embed_record_path_value = event_payload.get("embed_record_path")
    if not isinstance(embed_record_path_value, str) or not embed_record_path_value.strip():
        return None

    try:
        embed_record = _load_required_json_dict(
            Path(embed_record_path_value).expanduser().resolve(),
            "PW01 embed record",
        )
    except Exception:
        return None

    watermarked_path_value = embed_record.get("watermarked_path") or embed_record.get("image_path")
    if not isinstance(watermarked_path_value, str) or not watermarked_path_value.strip():
        return None

    watermarked_path = Path(watermarked_path_value).expanduser().resolve()
    if not watermarked_path.exists() or not watermarked_path.is_file():
        return None
    return normalize_path_value(watermarked_path)


def _safe_resolve_prompt_text(event_payload: Mapping[str, Any]) -> str | None:
    """
    Resolve the prompt text from one PW01 event payload when available.

    Args:
        event_payload: PW01 event payload surfaced via shard manifest.

    Returns:
        Prompt text or None.
    """
    if not isinstance(event_payload, Mapping):
        raise TypeError("event_payload must be Mapping")

    for key_name in ["prompt_text", "prompt"]:
        prompt_value = event_payload.get(key_name)
        if isinstance(prompt_value, str) and prompt_value.strip():
            return prompt_value.strip()
    return None


def _build_clean_quality_pair_manifest(
    *,
    family_id: str,
    stage_root: Path,
    positive_event_lookup: Mapping[str, Dict[str, Any]],
    eval_positive_event_ids: Sequence[str],
    score_run: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build one reusable clean-pair manifest for downstream PW04 quality metrics.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        positive_event_lookup: Positive-source event lookup.
        eval_positive_event_ids: Positive event ids used by the evaluate split.

    Returns:
        Clean-pair manifest export summary.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")
    if not isinstance(positive_event_lookup, Mapping):
        raise TypeError("positive_event_lookup must be Mapping")
    if not isinstance(eval_positive_event_ids, Sequence):
        raise TypeError("eval_positive_event_ids must be Sequence")

    pair_rows: List[Dict[str, Any]] = []
    missing_pair_count = 0
    prompt_text_available_count = 0
    for event_id in eval_positive_event_ids:
        if not isinstance(event_id, str) or not event_id:
            raise TypeError("eval_positive_event_ids items must be non-empty str")
        event_payload = positive_event_lookup.get(event_id)
        if not isinstance(event_payload, Mapping):
            missing_pair_count += 1
            pair_rows.append(
                {
                    "event_id": event_id,
                    "reference_image_path": None,
                    "candidate_image_path": None,
                    "plain_preview_image_path": None,
                    "watermarked_output_image_path": None,
                    "prompt_text": None,
                    "sample_role": ACTIVE_SAMPLE_ROLE,
                }
            )
            continue
        plain_preview_image_path = _safe_resolve_preview_artifact_path(event_payload)
        watermarked_output_image_path = _safe_resolve_watermarked_image_path(event_payload)
        prompt_text = _safe_resolve_prompt_text(event_payload)
        if isinstance(prompt_text, str) and prompt_text:
            prompt_text_available_count += 1
        if (
            not isinstance(plain_preview_image_path, str)
            or not plain_preview_image_path
            or not isinstance(watermarked_output_image_path, str)
            or not watermarked_output_image_path
        ):
            missing_pair_count += 1
        pair_rows.append(
            {
                "event_id": event_id,
                "reference_image_path": plain_preview_image_path,
                "candidate_image_path": watermarked_output_image_path,
                "plain_preview_image_path": plain_preview_image_path,
                "watermarked_output_image_path": watermarked_output_image_path,
                "prompt_text": prompt_text,
                "sample_role": ACTIVE_SAMPLE_ROLE,
            }
        )

    expected_pair_count = len(eval_positive_event_ids)
    complete_pair_count = expected_pair_count - missing_pair_count
    if expected_pair_count <= 0:
        status_value = "not_available"
        reason_value = "no_evaluate_positive_events_available_for_clean_pair_manifest"
    elif missing_pair_count <= 0:
        status_value = "ok"
        reason_value = None
    else:
        status_value = "partial"
        reason_value = (
            f"clean pair manifest complete for {complete_pair_count}/{expected_pair_count} evaluate-positive events"
        )

    output_path = stage_root / "quality" / CLEAN_QUALITY_PAIR_MANIFEST_FILE_NAME
    ensure_directory(output_path.parent)
    payload = {
        "artifact_type": "paper_workflow_pw02_clean_quality_pair_manifest",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "score_name": CONTENT_SCORE_NAME,
        "scope": "content_chain",
        "canonical": False,
        "analysis_only": True,
        "status": status_value,
        "reason": reason_value,
        "pair_id_key": "event_id",
        "reference_path_key": "reference_image_path",
        "candidate_path_key": "candidate_image_path",
        "text_key": "prompt_text",
        "extra_metadata_keys": ["sample_role", "plain_preview_image_path", "watermarked_output_image_path"],
        "reference_artifact_name": "plain_preview_image",
        "candidate_artifact_name": "watermarked_output_image",
        "reference_semantics": "preview_generation_persisted_artifact_vs_watermarked_output_image",
        "sample_role": ACTIVE_SAMPLE_ROLE,
        "split_scope": "evaluate_positive_only",
        "pair_count": expected_pair_count,
        "expected_pair_count": expected_pair_count,
        "complete_pair_count": complete_pair_count,
        "missing_pair_count": missing_pair_count,
        "prompt_text_expected": True,
        "prompt_text_available_count": prompt_text_available_count,
        "prompt_text_missing_count": expected_pair_count - prompt_text_available_count,
        "pair_rows": pair_rows,
    }
    if isinstance(score_run, Mapping):
        benchmark_provenance = score_run.get("benchmark_provenance")
        if isinstance(benchmark_provenance, Mapping):
            payload["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], benchmark_provenance)))
        score_pool = score_run.get("score_pool")
        if isinstance(score_pool, Mapping):
            payload["score_pool"] = copy.deepcopy(dict(cast(Mapping[str, Any], score_pool)))
    write_json_atomic(output_path, payload)
    return {
        "path": normalize_path_value(output_path),
        "payload": payload,
    }


def _extract_clean_event_outcomes(detect_payload: Mapping[str, Any]) -> Dict[str, bool]:
    """
    Extract clean-side formal and system outcomes from one detect payload.

    Args:
        detect_payload: Source detect payload.

    Returns:
        Outcome booleans for formal and system scopes.
    """
    if not isinstance(detect_payload, Mapping):
        raise TypeError("detect_payload must be Mapping")

    final_decision_payload = cast(Mapping[str, Any], detect_payload.get("final_decision", {}))
    attestation_payload = cast(Mapping[str, Any], detect_payload.get("attestation", {}))
    final_attestation_payload = cast(
        Mapping[str, Any],
        attestation_payload.get("final_event_attested_decision", {}),
    )
    formal_final_positive = final_decision_payload.get("is_watermarked") is True
    formal_attestation_positive = final_attestation_payload.get("is_event_attested") is True
    return {
        "formal_final_decision_is_positive": formal_final_positive,
        "formal_event_attestation_is_positive": formal_attestation_positive,
        "system_final_is_positive": bool(formal_final_positive or formal_attestation_positive),
    }


def _build_clean_event_table_rows(
    *,
    positive_event_lookup: Mapping[str, Dict[str, Any]],
    clean_negative_event_lookup: Mapping[str, Dict[str, Any]],
    eval_positive_event_ids: Sequence[str],
    eval_negative_event_ids: Sequence[str],
    content_threshold_value: float,
) -> List[Dict[str, Any]]:
    """
    Build the canonical clean evaluate event table rows.

    Args:
        positive_event_lookup: Positive-source event lookup.
        clean_negative_event_lookup: Clean-negative event lookup.
        eval_positive_event_ids: Evaluate positive event ids.
        eval_negative_event_ids: Evaluate negative event ids.
        content_threshold_value: Frozen PW02 content threshold value.

    Returns:
        Ordered clean event table rows.
    """
    if not isinstance(positive_event_lookup, Mapping):
        raise TypeError("positive_event_lookup must be Mapping")
    if not isinstance(clean_negative_event_lookup, Mapping):
        raise TypeError("clean_negative_event_lookup must be Mapping")
    if not isinstance(eval_positive_event_ids, Sequence):
        raise TypeError("eval_positive_event_ids must be Sequence")
    if not isinstance(eval_negative_event_ids, Sequence):
        raise TypeError("eval_negative_event_ids must be Sequence")
    if not isinstance(content_threshold_value, (int, float)) or isinstance(content_threshold_value, bool):
        raise TypeError("content_threshold_value must be finite float")

    row_specs: List[Tuple[int, bool, str, Dict[str, Any]]] = []
    for event_id in eval_positive_event_ids:
        if not isinstance(event_id, str) or not event_id:
            raise TypeError("eval_positive_event_ids items must be non-empty str")
        event_payload = positive_event_lookup.get(event_id)
        if not isinstance(event_payload, Mapping):
            raise ValueError(f"missing evaluate positive event payload: {event_id}")
        event_index = event_payload.get("event_index")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError(f"evaluate positive event missing event_index: {event_id}")
        row_specs.append((event_index, True, ACTIVE_SAMPLE_ROLE, dict(cast(Mapping[str, Any], event_payload))))
    for event_id in eval_negative_event_ids:
        if not isinstance(event_id, str) or not event_id:
            raise TypeError("eval_negative_event_ids items must be non-empty str")
        event_payload = clean_negative_event_lookup.get(event_id)
        if not isinstance(event_payload, Mapping):
            raise ValueError(f"missing evaluate negative event payload: {event_id}")
        event_index = event_payload.get("event_index")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError(f"evaluate negative event missing event_index: {event_id}")
        row_specs.append((event_index, False, CLEAN_NEGATIVE_SAMPLE_ROLE, dict(cast(Mapping[str, Any], event_payload))))

    clean_event_rows: List[Dict[str, Any]] = []
    for clean_event_table_index, (_event_index, ground_truth_label, sample_role, event_payload) in enumerate(
        sorted(row_specs, key=lambda item: (item[0], item[2], str(item[3].get("event_id"))))
    ):
        event_id = event_payload.get("event_id")
        detect_record_path_value = event_payload.get("detect_record_path")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("clean event row missing event_id")
        if not isinstance(detect_record_path_value, str) or not detect_record_path_value:
            raise ValueError(f"clean event row missing detect_record_path: {event_id}")
        detect_record_path = Path(detect_record_path_value).expanduser().resolve()
        detect_payload = _load_required_json_dict(detect_record_path, f"PW02 clean detect record {event_id}")
        content_score, content_score_source = eval_workflow_inputs._resolve_content_score_source(detect_payload)
        event_attestation_score, event_attestation_score_source = (
            eval_workflow_inputs._resolve_event_attestation_score_source(detect_payload)
        )
        clean_event_rows.append(
            {
                "clean_event_table_index": clean_event_table_index,
                "subset_name": "clean_eval_events",
                "event_id": event_id,
                "event_index": event_payload.get("event_index"),
                "sample_role": sample_role,
                "ground_truth_label": ground_truth_label,
                "prompt_text": event_payload.get("prompt_text"),
                "detect_record_path": normalize_path_value(detect_record_path),
                "content_score": content_score,
                "content_margin": (
                    float(cast(float, content_score) - float(content_threshold_value))
                    if isinstance(content_score, (int, float)) and not isinstance(content_score, bool)
                    else None
                ),
                "content_score_source": content_score_source,
                "event_attestation_score": event_attestation_score,
                "event_attestation_score_source": event_attestation_score_source,
                **_extract_clean_event_outcomes(detect_payload),
            }
        )
    return clean_event_rows


def _build_clean_score_analysis_export(
    *,
    family_id: str,
    stage_root: Path,
    score_name: str,
    score_run: Mapping[str, Any],
    positive_event_lookup: Mapping[str, Dict[str, Any]],
    eval_positive_event_ids: Sequence[str],
    clean_quality_pair_manifest_export: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    Build one append-only clean score analysis export.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        score_name: Canonical score name.
        score_run: Score-run summary.
        positive_event_lookup: Positive-source event lookup.
        eval_positive_event_ids: Positive event ids used by the evaluate split.
        clean_quality_pair_manifest_export: Clean-pair manifest export summary.

    Returns:
        Export summary with path and payload.
    """
    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be Path")

    evaluate_record_path = Path(str(score_run.get("evaluate_record_path", ""))).expanduser().resolve()
    thresholds_artifact_path = Path(str(score_run.get("thresholds_artifact_path", ""))).expanduser().resolve()
    evaluate_record = _load_required_json_dict(evaluate_record_path, f"PW02 evaluate record {score_name}")
    thresholds_artifact = _load_required_json_dict(thresholds_artifact_path, f"PW02 thresholds artifact {score_name}")
    prepared_records = _load_records_from_records_summary(cast(Mapping[str, Any], score_run.get("evaluate_inputs", {})))
    roc_fpr, roc_tpr, roc_thresholds = eval_metrics.compute_roc_curve(prepared_records, score_name=score_name)
    roc_auc_value = None
    if roc_fpr and roc_tpr:
        try:
            roc_auc_value = eval_metrics.compute_auc(roc_fpr, roc_tpr)
        except ValueError:
            roc_auc_value = None

    export_path = stage_root / "analysis" / _resolve_top_level_score_directory_name(score_name) / CLEAN_SCORE_ANALYSIS_FILE_NAME
    ensure_directory(export_path.parent)
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_clean_score_analysis",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "score_name": score_name,
        "evaluation_scope": "positive_source_vs_clean_negative",
        "source_evaluate_run_root": str(score_run.get("evaluate_run_root")),
        "source_evaluate_record_path": normalize_path_value(evaluate_record_path),
        "source_thresholds_artifact_path": normalize_path_value(thresholds_artifact_path),
        "evaluate_input_counts": _count_records_by_role(cast(Mapping[str, Any], score_run.get("evaluate_inputs", {}))),
        "operating_metrics": _extract_evaluate_metrics_payload(evaluate_record),
        "breakdown": _extract_evaluate_breakdown_payload(evaluate_record),
        "threshold_binding": {
            "threshold_id": thresholds_artifact.get("threshold_id"),
            "threshold_key_used": thresholds_artifact.get("threshold_key_used"),
            "threshold_value": thresholds_artifact.get("threshold_value"),
            "target_fpr": thresholds_artifact.get("target_fpr"),
            "decision_operator": thresholds_artifact.get("decision_operator"),
        },
        "roc_auc": {
            "auc": roc_auc_value,
            "roc_curve_points": len(roc_fpr),
            "fpr": roc_fpr,
            "tpr": roc_tpr,
            "thresholds": roc_thresholds,
        },
    }
    benchmark_provenance = score_run.get("benchmark_provenance")
    if isinstance(benchmark_provenance, Mapping):
        payload["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], benchmark_provenance)))
    score_pool = score_run.get("score_pool")
    if isinstance(score_pool, Mapping):
        payload["score_pool"] = copy.deepcopy(dict(cast(Mapping[str, Any], score_pool)))
    if eval_metrics.is_content_chain_score_name(score_name):
        pair_manifest_path = None
        if isinstance(clean_quality_pair_manifest_export, Mapping):
            pair_manifest_path_value = clean_quality_pair_manifest_export.get("path")
            if isinstance(pair_manifest_path_value, str) and pair_manifest_path_value.strip():
                pair_manifest_path = pair_manifest_path_value
        payload["clean_positive_quality_pair_manifest_path"] = pair_manifest_path
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
    export_path = stage_root / DERIVED_SYSTEM_UNION_METRICS_FILE_NAME
    evaluate_inputs = cast(Mapping[str, Any], content_score_run.get("evaluate_inputs", {}))
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_derived_system_union_metrics",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "derived_system_union",
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


def _build_formal_final_decision_metrics_export(
    *,
    family_id: str,
    stage_root: Path,
    content_score_run: Mapping[str, Any],
    formal_final_decision_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the formal final-decision metrics export from content evaluate inputs.

    Args:
        family_id: Family identifier.
        stage_root: PW02 stage root.
        content_score_run: Content score-run summary.
        formal_final_decision_metrics: Final-decision-only metrics.

    Returns:
        Export summary with path and payload.
    """
    export_path = stage_root / FORMAL_FINAL_DECISION_METRICS_FILE_NAME
    evaluate_inputs = cast(Mapping[str, Any], content_score_run.get("evaluate_inputs", {}))
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw02_formal_final_decision_metrics",
        "schema_version": "pw_stage_02_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "scope": "formal_final_decision",
        "source_kind": "formal_final_decision_metrics_from_content_evaluate_inputs",
        "is_formal_evaluate_record": False,
        "source_score_name": CONTENT_SCORE_NAME,
        "source_evaluate_run_root": str(content_score_run.get("evaluate_run_root")),
        "source_evaluate_record_path": str(content_score_run.get("evaluate_record_path")),
        "source_evaluate_inputs_glob": evaluate_inputs.get("records_glob"),
        "metrics": dict(formal_final_decision_metrics),
        "notes": "Aggregated from evaluate-stage formal_final_decision overlays over content evaluate inputs; older artifacts fall back to legacy final_decision only for compatibility. This is separate from the derived system union metrics.",
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
    control_negative_pool_manifest: Mapping[str, Any],
    threshold_exports: Mapping[str, Dict[str, Any]],
    clean_evaluate_exports: Mapping[str, Dict[str, Any]],
    clean_score_analysis_exports: Mapping[str, Dict[str, Any]],
    clean_event_table_path: Path,
    clean_event_table_event_count: int,
    clean_quality_pair_manifest_export: Mapping[str, Any],
    formal_final_decision_export: Mapping[str, Any],
    derived_system_union_export: Mapping[str, Any],
    score_runs: Mapping[str, Dict[str, Any]],
    benchmark_protocol_bundle: Mapping[str, Any] | None,
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
        clean_score_analysis_exports: Score-analysis export summaries keyed by score directory name.
        clean_event_table_path: Canonical clean event table path.
        clean_event_table_event_count: Clean event table row count.
        clean_quality_pair_manifest_export: Clean-pair manifest export summary.
        system_final_export: System-final export summary.
        score_runs: Score-run summaries keyed by score name.
        split_counts: Split count summary.

    Returns:
        Finalize-manifest payload.
    """
    positive_pool_payload = cast(Mapping[str, Any], positive_pool_manifest.get("payload", {}))
    clean_negative_pool_payload = cast(Mapping[str, Any], clean_negative_pool_manifest.get("payload", {}))
    control_negative_pool_payload = cast(Mapping[str, Any], control_negative_pool_manifest.get("payload", {}))
    clean_quality_pair_manifest_payload = cast(
        Mapping[str, Any],
        clean_quality_pair_manifest_export.get("payload", {}),
    )
    payload: Dict[str, Any] = {
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
                "event_count": positive_pool_payload.get("event_count"),
                "cohort_status": positive_pool_payload.get("cohort_status"),
                "role_requirement": positive_pool_payload.get("role_requirement"),
                "expected_source_shard_count": positive_pool_payload.get("expected_source_shard_count"),
                "discovered_source_shard_count": positive_pool_payload.get("discovered_source_shard_count"),
                "diagnostic_only": positive_pool_payload.get("diagnostic_only"),
            },
            CLEAN_NEGATIVE_SAMPLE_ROLE: {
                "manifest_path": str(clean_negative_pool_manifest.get("path")),
                "event_count": clean_negative_pool_payload.get("event_count"),
                "cohort_status": clean_negative_pool_payload.get("cohort_status"),
                "role_requirement": clean_negative_pool_payload.get("role_requirement"),
                "expected_source_shard_count": clean_negative_pool_payload.get("expected_source_shard_count"),
                "discovered_source_shard_count": clean_negative_pool_payload.get("discovered_source_shard_count"),
                "diagnostic_only": clean_negative_pool_payload.get("diagnostic_only"),
            },
            PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: {
                "manifest_path": str(control_negative_pool_manifest.get("path")),
                "event_count": control_negative_pool_payload.get("event_count"),
                "cohort_status": control_negative_pool_payload.get("cohort_status"),
                "role_requirement": control_negative_pool_payload.get("role_requirement"),
                "expected_source_shard_count": control_negative_pool_payload.get("expected_source_shard_count"),
                "discovered_source_shard_count": control_negative_pool_payload.get("discovered_source_shard_count"),
                "diagnostic_only": control_negative_pool_payload.get("diagnostic_only"),
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
        "clean_score_analysis_exports": {
            score_key: {
                "score_name": export_summary.get("score_name"),
                "path": export_summary.get("path"),
                "source_evaluate_run_root": cast(Mapping[str, Any], export_summary.get("payload", {})).get("source_evaluate_run_root"),
                "source_evaluate_record_path": cast(Mapping[str, Any], export_summary.get("payload", {})).get("source_evaluate_record_path"),
            }
            for score_key, export_summary in clean_score_analysis_exports.items()
        },
        "clean_event_table": {
            "path": normalize_path_value(clean_event_table_path),
            "event_count": clean_event_table_event_count,
            "subset_name": "clean_eval_events",
        },
        "clean_quality_pair_artifact": {
            "path": clean_quality_pair_manifest_export.get("path"),
            "status": clean_quality_pair_manifest_payload.get("status"),
            "reason": clean_quality_pair_manifest_payload.get("reason"),
            "pair_count": clean_quality_pair_manifest_payload.get("pair_count"),
            "complete_pair_count": clean_quality_pair_manifest_payload.get("complete_pair_count"),
            "missing_pair_count": clean_quality_pair_manifest_payload.get("missing_pair_count"),
        },
        "formal_final_decision": {
            "mode": "formal_final_decision_metrics_from_content_evaluate_inputs",
            "is_formal_evaluate_record": False,
            "artifact_path": formal_final_decision_export.get("path"),
            "source_score_name": cast(Mapping[str, Any], formal_final_decision_export.get("payload", {})).get("source_score_name"),
            "source_evaluate_run_root": cast(Mapping[str, Any], formal_final_decision_export.get("payload", {})).get("source_evaluate_run_root"),
        },
        "derived_system_union": {
            "mode": "derived_metrics_from_content_evaluate_inputs",
            "is_formal_evaluate_record": False,
            "artifact_path": derived_system_union_export.get("path"),
            "source_score_name": cast(Mapping[str, Any], derived_system_union_export.get("payload", {})).get("source_score_name"),
            "source_evaluate_run_root": cast(Mapping[str, Any], derived_system_union_export.get("payload", {})).get("source_evaluate_run_root"),
        },
        "system_final": {
            "mode": "derived_metrics_from_content_evaluate_inputs",
            "is_formal_evaluate_record": False,
            "artifact_path": derived_system_union_export.get("path"),
            "source_score_name": cast(Mapping[str, Any], derived_system_union_export.get("payload", {})).get("source_score_name"),
            "source_evaluate_run_root": cast(Mapping[str, Any], derived_system_union_export.get("payload", {})).get("source_evaluate_run_root"),
        },
        "score_runs": {
            score_name: {
                "calibrate_run_root": score_run.get("calibrate_run_root"),
                "evaluate_run_root": score_run.get("evaluate_run_root"),
                "thresholds_artifact_path": score_run.get("thresholds_artifact_path"),
                "evaluate_record_path": score_run.get("evaluate_record_path"),
                "score_pool": copy.deepcopy(dict(cast(Mapping[str, Any], score_run.get("score_pool", {})))),
                "benchmark_provenance": copy.deepcopy(
                    dict(cast(Mapping[str, Any], score_run.get("benchmark_provenance", {})))
                ),
            }
            for score_name, score_run in score_runs.items()
        },
    }
    if isinstance(benchmark_protocol_bundle, Mapping):
        protocol_node = benchmark_protocol_bundle.get("protocol")
        if isinstance(protocol_node, Mapping):
            payload["benchmark_protocol"] = copy.deepcopy(dict(cast(Mapping[str, Any], protocol_node)))
        provenance_node = benchmark_protocol_bundle.get("provenance")
        if isinstance(provenance_node, Mapping):
            payload["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], provenance_node)))
    return payload


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
        payload_reference_sidecar_path = event_payload.get("payload_reference_sidecar_path")
        if isinstance(payload_reference_sidecar_path, str) and payload_reference_sidecar_path:
            labelled_payload["paper_workflow_payload_reference_sidecar_path"] = payload_reference_sidecar_path
        payload_decode_sidecar_path = event_payload.get("payload_decode_sidecar_path")
        if isinstance(payload_decode_sidecar_path, str) and payload_decode_sidecar_path:
            labelled_payload["paper_workflow_payload_decode_sidecar_path"] = payload_decode_sidecar_path
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


def _build_records_for_score_pool_split(
    *,
    family_id: str,
    split_kind: str,
    source_split_plan: Mapping[str, Any],
    role_order: Sequence[str],
    event_id_keys_by_role: Mapping[str, str],
    event_lookup_by_role: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Build one split worth of prepared records for a score-specific pool.

    Args:
        family_id: Family identifier.
        split_kind: Split-kind token.
        source_split_plan: Source split plan payload.
        role_order: Ordered sample roles for this split.
        event_id_keys_by_role: Mapping from sample_role to split-plan key.
        event_lookup_by_role: Mapping from sample_role to event lookup.

    Returns:
        Split summary with prepared records and role counts.
    """
    if not isinstance(source_split_plan, Mapping):
        raise TypeError("source_split_plan must be Mapping")

    prepared_records: List[Dict[str, Any]] = []
    included_roles: List[str] = []
    skipped_roles: List[str] = []
    role_counts: Dict[str, int] = {}

    for sample_role in role_order:
        split_event_id_key = event_id_keys_by_role.get(sample_role)
        if not isinstance(split_event_id_key, str) or not split_event_id_key:
            raise ValueError(f"missing split event id key for sample_role={sample_role}")

        role_event_lookup = event_lookup_by_role.get(sample_role)
        if not isinstance(role_event_lookup, Mapping):
            raise ValueError(f"missing event lookup for sample_role={sample_role}")

        if sample_role == PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE and not role_event_lookup:
            skipped_roles.append(sample_role)
            continue

        event_ids_node = source_split_plan.get(split_event_id_key)
        if not isinstance(event_ids_node, list):
            raise ValueError(f"source split plan missing {split_event_id_key}")

        role_prepared_records = _build_prepared_records(
            family_id=family_id,
            split_kind=split_kind,
            event_ids=cast(Sequence[str], event_ids_node),
            sample_role=sample_role,
            event_lookup=cast(Mapping[str, Dict[str, Any]], role_event_lookup),
        )
        prepared_records.extend(role_prepared_records)
        included_roles.append(sample_role)
        role_counts[sample_role] = len(role_prepared_records)

    return {
        "records": prepared_records,
        "included_roles": included_roles,
        "skipped_roles": skipped_roles,
        "role_counts": role_counts,
    }


def _build_score_pool_records(
    *,
    family_id: str,
    score_name: str,
    source_split_plan: Mapping[str, Any],
    positive_events: Mapping[str, Dict[str, Any]],
    clean_negative_events: Mapping[str, Dict[str, Any]],
    control_negative_events: Mapping[str, Dict[str, Any]],
    benchmark_protocol_bundle: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    Build score-specific calibration and evaluate pools.

    Args:
        family_id: Family identifier.
        score_name: Canonical score name.
        source_split_plan: Source split plan payload.
        positive_events: Positive-source event lookup.
        clean_negative_events: Clean-negative event lookup.
        control_negative_events: Control-negative event lookup.
        benchmark_protocol_bundle: Optional benchmark protocol bundle.

    Returns:
        Score-pool summary.
    """
    score_pool_spec = _resolve_score_pool_spec(
        score_name=score_name,
        benchmark_protocol_bundle=benchmark_protocol_bundle,
    )
    event_lookup_by_role: Dict[str, Mapping[str, Dict[str, Any]]] = {
        ACTIVE_SAMPLE_ROLE: positive_events,
        CLEAN_NEGATIVE_SAMPLE_ROLE: clean_negative_events,
        PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: control_negative_events,
    }
    calibration_split = _build_records_for_score_pool_split(
        family_id=family_id,
        split_kind="calibration",
        source_split_plan=source_split_plan,
        role_order=cast(Sequence[str], score_pool_spec["calibration_role_order"]),
        event_id_keys_by_role=cast(Mapping[str, str], score_pool_spec["calibration_event_id_keys"]),
        event_lookup_by_role=event_lookup_by_role,
    )
    evaluate_split = _build_records_for_score_pool_split(
        family_id=family_id,
        split_kind="evaluate",
        source_split_plan=source_split_plan,
        role_order=cast(Sequence[str], score_pool_spec["evaluate_role_order"]),
        event_id_keys_by_role=cast(Mapping[str, str], score_pool_spec["evaluate_event_id_keys"]),
        event_lookup_by_role=event_lookup_by_role,
    )

    benchmark_provenance = None
    if isinstance(benchmark_protocol_bundle, Mapping):
        benchmark_provenance = copy.deepcopy(
            cast(Mapping[str, Any], benchmark_protocol_bundle.get("provenance", {}))
        )
        if isinstance(benchmark_provenance, dict):
            benchmark_provenance["score_name"] = score_name
            benchmark_provenance["calibration_role_order"] = list(
                cast(Sequence[str], score_pool_spec["calibration_role_order"])
            )
            benchmark_provenance["calibration_event_id_keys"] = dict(
                cast(Mapping[str, str], score_pool_spec["calibration_event_id_keys"])
            )
            benchmark_provenance["evaluate_role_order"] = list(
                cast(Sequence[str], score_pool_spec["evaluate_role_order"])
            )
            benchmark_provenance["evaluate_event_id_keys"] = dict(
                cast(Mapping[str, str], score_pool_spec["evaluate_event_id_keys"])
            )

    return {
        "calibration_records": cast(List[Dict[str, Any]], calibration_split["records"]),
        "evaluate_records": cast(List[Dict[str, Any]], evaluate_split["records"]),
        "score_pool": {
            "score_name": score_name,
            "calibration_role_counts": dict(cast(Mapping[str, int], calibration_split["role_counts"])),
            "calibration_included_roles": list(cast(Sequence[str], calibration_split["included_roles"])),
            "calibration_skipped_roles": list(cast(Sequence[str], calibration_split["skipped_roles"])),
            "evaluate_role_counts": dict(cast(Mapping[str, int], evaluate_split["role_counts"])),
            "evaluate_included_roles": list(cast(Sequence[str], evaluate_split["included_roles"])),
            "evaluate_skipped_roles": list(cast(Sequence[str], evaluate_split["skipped_roles"])),
        },
        "benchmark_provenance": benchmark_provenance,
    }


def _write_score_split_records(
    *,
    score_name: str,
    records_root: Path,
    prepared_records: Sequence[Mapping[str, Any]],
    thresholds_artifact: Mapping[str, Any] | None = None,
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
    if thresholds_artifact is not None and not isinstance(thresholds_artifact, Mapping):
        raise TypeError("thresholds_artifact must be Mapping or None")

    ensure_directory(records_root)
    written_paths: List[str] = []
    record_summaries: List[Dict[str, Any]] = []
    thresholds_payload = dict(cast(Dict[str, Any], thresholds_artifact)) if isinstance(thresholds_artifact, Mapping) else None

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
        if thresholds_payload is not None and eval_metrics.is_content_chain_score_name(score_name):
            formal_final_decision = eval_workflow_inputs.build_formal_final_decision_overlay(
                payload,
                score_name=score_name,
                thresholds_artifact=thresholds_payload,
            )
            payload["formal_final_decision"] = formal_final_decision
            payload["formal_final_decision_source"] = str(formal_final_decision["decision_origin"])

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
    cfg_obj["allow_nonempty_run_root_reason"] = PW02_RUN_ROOT_REUSE_REASON

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


def _build_run_root_reuse_override_args(cfg_obj: Mapping[str, Any]) -> List[str]:
    """
    Build CLI override args that authorize PW02 run_root reuse.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        CLI override argument list.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")

    allow_nonempty_run_root = cfg_obj.get("allow_nonempty_run_root")
    if allow_nonempty_run_root is not True:
        raise ValueError("PW02 run_root reuse override requires allow_nonempty_run_root=True")

    allow_nonempty_run_root_reason = cfg_obj.get("allow_nonempty_run_root_reason")
    if not isinstance(allow_nonempty_run_root_reason, str) or not allow_nonempty_run_root_reason:
        raise ValueError("PW02 run_root reuse override requires non-empty allow_nonempty_run_root_reason")

    return [
        f"run_root_reuse_allowed={json.dumps(True)}",
        f"run_root_reuse_reason={json.dumps(allow_nonempty_run_root_reason)}",
    ]


def _run_python_stage_command(
    *,
    module_name: str,
    output_dir: Path,
    config_path: Path,
    log_prefix: str,
    overrides: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Run one Python CLI stage with explicit stdout and stderr logs.

    Args:
        module_name: Python module name.
        output_dir: Stage output directory.
        config_path: Runtime config path.
        log_prefix: Log-file prefix.
        overrides: Optional CLI override args.

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
    overrides_obj: Any = overrides
    if overrides_obj is not None and (
        isinstance(overrides_obj, (str, bytes)) or not isinstance(overrides_obj, Sequence)
    ):
        raise TypeError("overrides must be Sequence[str] or None")

    normalized_overrides: List[str] = []
    if overrides_obj is not None:
        for item in cast(Sequence[Any], overrides_obj):
            if not isinstance(item, str) or not item:
                raise TypeError("overrides items must be non-empty str")
            normalized_overrides.append(item)

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
    for override_arg in normalized_overrides:
        command.extend(["--override", override_arg])
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
    runtime_override_args = _build_run_root_reuse_override_args(runtime_cfg_obj)
    runtime_cfg_path = score_root / "runtime_config.yaml"
    write_yaml_mapping(runtime_cfg_path, runtime_cfg_obj)

    calibrate_result = _run_python_stage_command(
        module_name="main.cli.run_calibrate",
        output_dir=calibrate_root,
        config_path=runtime_cfg_path,
        log_prefix="pw02_calibrate",
        overrides=runtime_override_args,
    )
    if int(calibrate_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW02 calibrate failed for {score_name}: {json.dumps(calibrate_result, ensure_ascii=False, sort_keys=True)}"
        )

    thresholds_artifact_path = calibrate_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    thresholds_artifact = _load_required_json_dict(
        thresholds_artifact_path,
        f"PW02 thresholds artifact for {score_name}",
    )
    evaluate_records_summary = _write_score_split_records(
        score_name=score_name,
        records_root=evaluate_records_root,
        prepared_records=evaluate_records,
        thresholds_artifact=thresholds_artifact,
    )
    runtime_cfg_obj = _build_score_runtime_config(
        base_cfg_obj=base_cfg_obj,
        score_name=score_name,
        calibration_records_glob=str((calibration_records_root / "*.json").resolve()),
        evaluate_records_glob=str((evaluate_records_root / "*.json").resolve()),
        thresholds_path=thresholds_artifact_path,
    )
    runtime_override_args = _build_run_root_reuse_override_args(runtime_cfg_obj)
    write_yaml_mapping(runtime_cfg_path, runtime_cfg_obj)

    evaluate_result = _run_python_stage_command(
        module_name="main.cli.run_evaluate",
        output_dir=evaluate_root,
        config_path=runtime_cfg_path,
        log_prefix="pw02_evaluate",
        overrides=runtime_override_args,
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
    control_negative_collection = _collect_optional_completed_events_for_role(
        family_root=family_root,
        source_shard_plan=source_shard_plan,
        sample_role=PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
    )
    control_negative_events = cast(Dict[str, Dict[str, Any]], control_negative_collection["event_lookup"])
    benchmark_protocol_bundle = _resolve_benchmark_protocol_bundle(family_manifest)
    score_pool_records = {
        CONTENT_SCORE_NAME: _build_score_pool_records(
            family_id=family_id,
            score_name=CONTENT_SCORE_NAME,
            source_split_plan=source_split_plan,
            positive_events=positive_events,
            clean_negative_events=clean_negative_events,
            control_negative_events=control_negative_events,
            benchmark_protocol_bundle=benchmark_protocol_bundle,
        ),
        EVENT_ATTESTATION_SCORE_NAME: _build_score_pool_records(
            family_id=family_id,
            score_name=EVENT_ATTESTATION_SCORE_NAME,
            source_split_plan=source_split_plan,
            positive_events=positive_events,
            clean_negative_events=clean_negative_events,
            control_negative_events=control_negative_events,
            benchmark_protocol_bundle=benchmark_protocol_bundle,
        ),
    }

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
            "calibration": len(cast(List[str], source_split_plan.get("calib_pos_event_ids", [])))
            + len(cast(List[str], source_split_plan.get("calib_neg_event_ids", []))),
            "evaluate": len(cast(List[str], source_split_plan.get("eval_pos_event_ids", [])))
            + len(cast(List[str], source_split_plan.get("eval_neg_event_ids", []))),
        },
        "split_event_ids": {
            "calibration": [
                *[str(event_id) for event_id in cast(List[str], source_split_plan.get("calib_pos_event_ids", []))],
                *[str(event_id) for event_id in cast(List[str], source_split_plan.get("calib_neg_event_ids", []))],
            ],
            "evaluate": [
                *[str(event_id) for event_id in cast(List[str], source_split_plan.get("eval_pos_event_ids", []))],
                *[str(event_id) for event_id in cast(List[str], source_split_plan.get("eval_neg_event_ids", []))],
            ],
        },
    }
    write_json_atomic(merge_manifest_path, merge_manifest_payload)

    positive_role_plan = _resolve_role_plan(source_shard_plan, ACTIVE_SAMPLE_ROLE)
    positive_shards_node = positive_role_plan.get("shards")
    if not isinstance(positive_shards_node, list):
        raise ValueError(f"source shard plan {ACTIVE_SAMPLE_ROLE}.shards must be list")
    expected_positive_source_shard_count = len(positive_shards_node)
    discovered_positive_source_shard_count = len(
        {
            int(cast(Dict[str, Any], event_payload)["source_shard_index"])
            for event_payload in positive_events.values()
        }
    )

    clean_negative_role_plan = _resolve_role_plan(source_shard_plan, CLEAN_NEGATIVE_SAMPLE_ROLE)
    clean_negative_shards_node = clean_negative_role_plan.get("shards")
    if not isinstance(clean_negative_shards_node, list):
        raise ValueError(f"source shard plan {CLEAN_NEGATIVE_SAMPLE_ROLE}.shards must be list")
    expected_clean_negative_source_shard_count = len(clean_negative_shards_node)
    discovered_clean_negative_source_shard_count = len(
        {
            int(cast(Dict[str, Any], event_payload)["source_shard_index"])
            for event_payload in clean_negative_events.values()
        }
    )

    positive_pool_manifest = _write_source_pool_manifest(
        stage_root=stage_root,
        family_id=family_id,
        sample_role=ACTIVE_SAMPLE_ROLE,
        event_lookup=positive_events,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
        expected_source_shard_count=expected_positive_source_shard_count,
        discovered_source_shard_count=discovered_positive_source_shard_count,
    )
    clean_negative_pool_manifest = _write_source_pool_manifest(
        stage_root=stage_root,
        family_id=family_id,
        sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
        event_lookup=clean_negative_events,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
        expected_source_shard_count=expected_clean_negative_source_shard_count,
        discovered_source_shard_count=discovered_clean_negative_source_shard_count,
    )
    control_negative_pool_manifest = _write_source_pool_manifest(
        stage_root=stage_root,
        family_id=family_id,
        sample_role=PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
        event_lookup=control_negative_events,
        family_manifest_path=layout["family_manifest_path"],
        source_shard_plan_path=layout["source_shard_plan_path"],
        source_split_plan_path=layout["source_split_plan_path"],
        role_requirement=str(control_negative_collection["role_requirement"]),
        cohort_status=str(control_negative_collection["cohort_status"]),
        expected_source_shard_count=int(control_negative_collection["expected_source_shard_count"]),
        discovered_source_shard_count=int(control_negative_collection["discovered_source_shard_count"]),
        missing_source_shard_indices=cast(List[int], control_negative_collection["missing_source_shard_indices"]),
    )
    score_runs = {
        CONTENT_SCORE_NAME: _run_score_pipeline(
            score_name=CONTENT_SCORE_NAME,
            stage_root=stage_root,
            base_cfg_obj=base_cfg_obj,
            calibration_records=cast(List[Dict[str, Any]], score_pool_records[CONTENT_SCORE_NAME]["calibration_records"]),
            evaluate_records=cast(List[Dict[str, Any]], score_pool_records[CONTENT_SCORE_NAME]["evaluate_records"]),
        ),
        EVENT_ATTESTATION_SCORE_NAME: _run_score_pipeline(
            score_name=EVENT_ATTESTATION_SCORE_NAME,
            stage_root=stage_root,
            base_cfg_obj=base_cfg_obj,
            calibration_records=cast(
                List[Dict[str, Any]],
                score_pool_records[EVENT_ATTESTATION_SCORE_NAME]["calibration_records"],
            ),
            evaluate_records=cast(
                List[Dict[str, Any]],
                score_pool_records[EVENT_ATTESTATION_SCORE_NAME]["evaluate_records"],
            ),
        ),
    }
    for score_name, score_run in score_runs.items():
        score_run["score_pool"] = copy.deepcopy(cast(Mapping[str, Any], score_pool_records[score_name]["score_pool"]))
        score_run["benchmark_provenance"] = copy.deepcopy(
            cast(Mapping[str, Any], score_pool_records[score_name].get("benchmark_provenance", {}) or {})
        )

    clean_quality_pair_manifest_export = _build_clean_quality_pair_manifest(
        family_id=family_id,
        stage_root=stage_root,
        positive_event_lookup=positive_events,
        eval_positive_event_ids=cast(Sequence[str], source_split_plan.get("eval_pos_event_ids", [])),
        score_run=cast(Mapping[str, Any], score_runs[CONTENT_SCORE_NAME]),
    )

    content_evaluate_root = Path(str(score_runs[CONTENT_SCORE_NAME]["evaluate_run_root"]))
    formal_final_decision_metrics = _build_formal_final_decision_metrics_for_run(content_evaluate_root)
    derived_system_union_metrics = _build_system_final_metrics_for_run(content_evaluate_root)

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
    clean_score_analysis_exports = {
        _resolve_top_level_score_directory_name(score_name): _build_clean_score_analysis_export(
            family_id=family_id,
            stage_root=stage_root,
            score_name=score_name,
            score_run=score_run,
            positive_event_lookup=positive_events,
            eval_positive_event_ids=cast(Sequence[str], source_split_plan.get("eval_pos_event_ids", [])),
            clean_quality_pair_manifest_export=clean_quality_pair_manifest_export,
        )
        for score_name, score_run in score_runs.items()
    }
    clean_event_table_rows = _build_clean_event_table_rows(
        positive_event_lookup=positive_events,
        clean_negative_event_lookup=clean_negative_events,
        eval_positive_event_ids=cast(Sequence[str], source_split_plan.get("eval_pos_event_ids", [])),
        eval_negative_event_ids=cast(Sequence[str], source_split_plan.get("eval_neg_event_ids", [])),
        content_threshold_value=_extract_threshold_value_from_export_summary(
            threshold_exports[_resolve_top_level_score_directory_name(CONTENT_SCORE_NAME)]
        ),
    )
    clean_event_table_path = stage_root / "tables" / CLEAN_EVENT_TABLE_FILE_NAME
    ensure_directory(clean_event_table_path.parent)
    write_jsonl(clean_event_table_path, clean_event_table_rows)
    formal_final_decision_export = _build_formal_final_decision_metrics_export(
        family_id=family_id,
        stage_root=stage_root,
        content_score_run=score_runs[CONTENT_SCORE_NAME],
        formal_final_decision_metrics=formal_final_decision_metrics,
    )
    derived_system_union_export = _build_system_final_metrics_export(
        family_id=family_id,
        stage_root=stage_root,
        content_score_run=score_runs[CONTENT_SCORE_NAME],
        system_final_metrics=derived_system_union_metrics,
    )
    pw02_metrics_extensions = build_pw02_metrics_extensions(
        family_id=family_id,
        stage_root=stage_root,
        clean_score_analysis_exports={
            score_key: str(export_summary["path"])
            for score_key, export_summary in clean_score_analysis_exports.items()
        },
        clean_quality_pair_manifest_path=Path(str(clean_quality_pair_manifest_export["path"])).expanduser().resolve(),
        score_runs=score_runs,
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
        control_negative_pool_manifest=control_negative_pool_manifest,
        threshold_exports=threshold_exports,
        clean_evaluate_exports=clean_evaluate_exports,
        clean_score_analysis_exports=clean_score_analysis_exports,
        clean_event_table_path=clean_event_table_path,
        clean_event_table_event_count=len(clean_event_table_rows),
        clean_quality_pair_manifest_export=clean_quality_pair_manifest_export,
        formal_final_decision_export=formal_final_decision_export,
        derived_system_union_export=derived_system_union_export,
        score_runs=score_runs,
        benchmark_protocol_bundle=benchmark_protocol_bundle,
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
        "planner_conditioned_control_negative_pool_manifest_path": str(control_negative_pool_manifest["path"]),
        "planner_conditioned_control_negative_cohort_status": cast(Mapping[str, Any], control_negative_pool_manifest["payload"]).get("cohort_status"),
        "planner_conditioned_control_negative_role_requirement": cast(Mapping[str, Any], control_negative_pool_manifest["payload"]).get("role_requirement"),
        "planner_conditioned_control_negative_expected_source_shard_count": cast(Mapping[str, Any], control_negative_pool_manifest["payload"]).get("expected_source_shard_count"),
        "planner_conditioned_control_negative_discovered_source_shard_count": cast(Mapping[str, Any], control_negative_pool_manifest["payload"]).get("discovered_source_shard_count"),
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
        "clean_score_analysis_exports": {
            score_key: export_summary["path"]
            for score_key, export_summary in clean_score_analysis_exports.items()
        },
        "clean_event_table_path": normalize_path_value(clean_event_table_path),
        "operating_metrics_dir": pw02_metrics_extensions["operating_metrics_dir"],
        "clean_pair_artifacts_dir": pw02_metrics_extensions["clean_pair_artifacts_dir"],
        "clean_quality_pair_manifest_path": pw02_metrics_extensions["clean_quality_pair_manifest_path"],
        "payload_metrics_dir": pw02_metrics_extensions["payload_metrics_dir"],
        "roc_curve_paths": pw02_metrics_extensions["roc_curve_paths"],
        "system_final_auxiliary_operating_semantics_path": pw02_metrics_extensions["system_final_auxiliary_operating_semantics_path"],
        "auc_summary_path": pw02_metrics_extensions["auc_summary_path"],
        "eer_summary_path": pw02_metrics_extensions["eer_summary_path"],
        "tpr_at_target_fpr_summary_path": pw02_metrics_extensions["tpr_at_target_fpr_summary_path"],
        "payload_clean_summary_path": pw02_metrics_extensions["payload_clean_summary_path"],
        "analysis_only_artifact_paths": pw02_metrics_extensions["analysis_only_artifact_paths"],
        "formal_final_decision_metrics": formal_final_decision_metrics,
        "formal_final_decision_metrics_artifact_path": str(formal_final_decision_export["path"]),
        "derived_system_union_metrics": derived_system_union_metrics,
        "derived_system_union_metrics_artifact_path": str(derived_system_union_export["path"]),
        "system_final_metrics": derived_system_union_metrics,
        "system_final_semantics": "derived_system_union_metrics",
        "system_final_metrics_artifact_path": str(derived_system_union_export["path"]),
        "split_counts": merge_manifest_payload["split_counts"],
    }
    if isinstance(benchmark_protocol_bundle, Mapping):
        protocol_node = benchmark_protocol_bundle.get("protocol")
        if isinstance(protocol_node, Mapping):
            summary["benchmark_protocol"] = copy.deepcopy(dict(cast(Mapping[str, Any], protocol_node)))
        provenance_node = benchmark_protocol_bundle.get("provenance")
        if isinstance(provenance_node, Mapping):
            summary["benchmark_provenance"] = copy.deepcopy(dict(cast(Mapping[str, Any], provenance_node)))
    write_json_atomic(summary_path, summary)
    return summary