"""
File purpose: Execute one PW03 attacked-event shard with worker isolation.
Module type: General module
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from main.evaluation import attack_runner as eval_attack_runner
from paper_workflow.scripts.pw_common import (
    ACTIVE_SAMPLE_ROLE,
    ATTACKED_POSITIVE_SAMPLE_ROLE,
    build_family_root,
    read_jsonl,
)
from scripts.notebook_runtime_common import (
    build_repo_import_subprocess_env,
    compute_file_sha256,
    copy_file,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    run_command_with_logs,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)


STAGE_NAME = "PW03_Attack_Event_Shards"
GPU_PEAK_SCRIPT_PATH = REPO_ROOT / "scripts" / "gpu_session_peak.py"
DEFAULT_SAMPLE_INTERVAL_MS = 200
DEFAULT_ATTACK_LOCAL_WORKER_COUNT = 1
ALLOWED_ATTACK_LOCAL_WORKER_COUNTS = {1, 2, 3, 4}
ALLOWED_ATTACK_LOCAL_WORKER_COUNT_ERROR = "attack_local_worker_count must be one of 1, 2, 3, or 4"
PW02_SUMMARY_FILE_NAME = "pw02_summary.json"
PW03_RUNTIME_SUMMARY_FILE_NAME = "runtime_summary.json"
PW03_SHARD_MANIFEST_FILE_NAME = "shard_manifest.json"
PW03_WORKER_PLAN_FILE_NAME = "worker_plan.json"
PW03_WORKER_RESULT_FILE_NAME = "worker_result.json"


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    Load one required JSON object.

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


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    Normalize one optional mapping payload to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Normalized dict payload.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _extract_attack_severity_metadata(attack_event_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract frozen severity metadata from one attack event spec.

    Args:
        attack_event_spec: Materialized attack event spec.

    Returns:
        Severity metadata mapping.
    """
    if not isinstance(attack_event_spec, Mapping):
        raise TypeError("attack_event_spec must be Mapping")
    return {
        "severity_rule_version": attack_event_spec.get("severity_rule_version"),
        "severity_axis_kind": attack_event_spec.get("severity_axis_kind"),
        "severity_directionality": attack_event_spec.get("severity_directionality"),
        "severity_status": attack_event_spec.get("severity_status"),
        "severity_reason": attack_event_spec.get("severity_reason"),
        "severity_source_param": attack_event_spec.get("severity_source_param"),
        "severity_scalarization": attack_event_spec.get("severity_scalarization"),
        "severity_value": attack_event_spec.get("severity_value"),
        "severity_sort_value": attack_event_spec.get("severity_sort_value"),
        "severity_label": attack_event_spec.get("severity_label"),
        "severity_level_index": attack_event_spec.get("severity_level_index"),
    }


def _extract_attack_geometry_diagnostics(detect_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract stable geometry diagnostics from one attacked detect payload.

    Args:
        detect_payload: PW03 staged detect payload.

    Returns:
        Append-only geometry diagnostics mapping.
    """
    if not isinstance(detect_payload, Mapping):
        raise TypeError("detect_payload must be Mapping")

    geometry_result = _extract_mapping(detect_payload.get("geometry_result"))
    geometry_evidence_payload = _extract_mapping(detect_payload.get("geometry_evidence_payload"))
    sync_result = _extract_mapping(geometry_result.get("sync_result"))
    anchor_result = _extract_mapping(geometry_result.get("anchor_result"))
    align_metrics = _extract_mapping(geometry_evidence_payload.get("align_metrics"))
    if not align_metrics:
        align_metrics = _extract_mapping(anchor_result.get("align_metrics"))

    sync_status = geometry_result.get("sync_status")
    if not isinstance(sync_status, str) or not sync_status:
        sync_status = sync_result.get("sync_status")
    if not isinstance(sync_status, str) or not sync_status:
        sync_status = sync_result.get("status")
    if not isinstance(sync_status, str) or not sync_status:
        sync_status = _extract_mapping(geometry_evidence_payload.get("audit")).get("sync_status_detail")

    sync_success_value = sync_result.get("sync_success")
    sync_success = None
    if isinstance(sync_success_value, bool):
        sync_success = sync_success_value
    elif isinstance(sync_status, str) and sync_status:
        sync_success = sync_status == "ok"

    sync_quality_metrics = _extract_mapping(geometry_result.get("sync_metrics"))
    if not sync_quality_metrics:
        sync_quality_metrics = _extract_mapping(sync_result.get("sync_quality_metrics"))
    if not sync_quality_metrics:
        sync_quality_metrics = _extract_mapping(geometry_evidence_payload.get("sync_quality_metrics"))

    template_match_metrics = _extract_mapping(sync_result.get("template_match_metrics"))
    if not template_match_metrics:
        template_match_metrics = {
            key_name: sync_quality_metrics.get(key_name)
            for key_name in [
                "template_match_score",
                "template_match_p95",
                "template_match_detected",
                "template_match_threshold",
                "template_seed",
                "template_digest",
                "template_confidence",
            ]
            if key_name in sync_quality_metrics
        }

    relation_binding_diagnostics = _extract_mapping(geometry_result.get("relation_binding_diagnostics"))
    relation_digest_bound = geometry_result.get("relation_digest_bound")
    if not isinstance(relation_digest_bound, str) or not relation_digest_bound:
        relation_digest_bound = sync_result.get("relation_digest_bound")

    geometry_failure_reason = geometry_result.get("geometry_failure_reason")
    if not isinstance(geometry_failure_reason, str) or not geometry_failure_reason:
        geometry_failure_reason = sync_result.get("geometry_failure_reason")
    if not isinstance(geometry_failure_reason, str) or not geometry_failure_reason:
        geometry_failure_reason = geometry_evidence_payload.get("geometry_failure_reason")

    anchor_digest = geometry_evidence_payload.get("anchor_digest")
    if not isinstance(anchor_digest, str) or not anchor_digest:
        anchor_digest = geometry_result.get("anchor_digest")
    if not isinstance(anchor_digest, str) or not anchor_digest:
        anchor_digest = anchor_result.get("anchor_digest")

    inverse_transform_success = None
    inverse_recovery_success = align_metrics.get("inverse_recovery_success")
    if isinstance(inverse_recovery_success, bool):
        inverse_transform_success = inverse_recovery_success

    attention_anchor_available = None
    if isinstance(anchor_digest, str) and anchor_digest:
        attention_anchor_available = True
    elif anchor_result or geometry_evidence_payload.get("anchor_metrics") is not None:
        attention_anchor_available = False

    if isinstance(geometry_failure_reason, str) and geometry_failure_reason:
        geometry_failure_status = "ok"
        geometry_failure_reason_reason = None
    else:
        geometry_failure_status = "not_available"
        geometry_failure_reason_reason = "geometry chain did not report failure reason"

    if isinstance(sync_success, bool):
        sync_success_status = "ok"
        sync_success_reason = None
    else:
        sync_success_status = "not_available"
        sync_success_reason = (
            geometry_failure_reason
            if isinstance(geometry_failure_reason, str) and geometry_failure_reason
            else "geometry chain did not report sync_success"
        )

    if sync_quality_metrics:
        sync_quality_metrics_status = "ok"
        sync_quality_metrics_reason = None
    else:
        sync_quality_metrics_status = "not_available"
        sync_quality_metrics_reason = (
            geometry_failure_reason
            if isinstance(geometry_failure_reason, str) and geometry_failure_reason
            else "geometry chain did not expose sync_quality_metrics"
        )

    if isinstance(inverse_transform_success, bool):
        inverse_transform_success_status = "ok"
        inverse_transform_success_reason = None
    else:
        inverse_transform_success_status = "not_available"
        inverse_transform_success_reason = (
            geometry_failure_reason
            if isinstance(geometry_failure_reason, str) and geometry_failure_reason
            else "geometry chain did not report inverse transform status"
        )

    if isinstance(attention_anchor_available, bool):
        attention_anchor_available_status = "ok"
        attention_anchor_available_reason = None
    else:
        attention_anchor_available_status = "not_available"
        attention_anchor_available_reason = (
            geometry_failure_reason
            if isinstance(geometry_failure_reason, str) and geometry_failure_reason
            else "geometry chain did not report attention anchor availability"
        )

    return {
        "sync_status": sync_status if isinstance(sync_status, str) and sync_status else None,
        "sync_success": sync_success,
        "sync_success_status": sync_success_status,
        "sync_success_reason": sync_success_reason,
        "sync_digest": sync_result.get("sync_digest") if isinstance(sync_result.get("sync_digest"), str) else geometry_result.get("sync_digest"),
        "geometry_failure_reason": geometry_failure_reason if isinstance(geometry_failure_reason, str) and geometry_failure_reason else None,
        "geometry_failure_reason_status": geometry_failure_status,
        "geometry_failure_reason_reason": geometry_failure_reason_reason,
        "relation_digest_bound": relation_digest_bound if isinstance(relation_digest_bound, str) and relation_digest_bound else None,
        "relation_binding_diagnostics": relation_binding_diagnostics or None,
        "template_match_metrics": template_match_metrics or None,
        "sync_quality_metrics": sync_quality_metrics or None,
        "sync_quality_metrics_status": sync_quality_metrics_status,
        "sync_quality_metrics_reason": sync_quality_metrics_reason,
        "inverse_transform_success": inverse_transform_success,
        "inverse_transform_success_status": inverse_transform_success_status,
        "inverse_transform_success_reason": inverse_transform_success_reason,
        "attention_anchor_available": attention_anchor_available,
        "attention_anchor_available_status": attention_anchor_available_status,
        "attention_anchor_available_reason": attention_anchor_available_reason,
        "anchor_digest": anchor_digest if isinstance(anchor_digest, str) and anchor_digest else None,
        "align_metrics": align_metrics or None,
    }


def _validate_attack_local_worker_count(attack_local_worker_count: int) -> None:
    """
    Validate the shard-local worker count.

    Args:
        attack_local_worker_count: Requested worker count.

    Returns:
        None.
    """
    if not isinstance(attack_local_worker_count, int) or isinstance(attack_local_worker_count, bool):
        raise ValueError(ALLOWED_ATTACK_LOCAL_WORKER_COUNT_ERROR)
    if attack_local_worker_count not in ALLOWED_ATTACK_LOCAL_WORKER_COUNTS:
        raise ValueError(ALLOWED_ATTACK_LOCAL_WORKER_COUNT_ERROR)


def _parse_attack_family_allowlist(attack_family_allowlist: Sequence[str] | str | None) -> List[str] | None:
    """
    Parse the optional attack-family allowlist.

    Args:
        attack_family_allowlist: Raw allowlist value.

    Returns:
        Normalized allowlist or None.
    """
    if attack_family_allowlist is None:
        return None
    if isinstance(attack_family_allowlist, str):
        allowlist_text = attack_family_allowlist.strip()
        if not allowlist_text:
            return None
        if allowlist_text.startswith("["):
            parsed = json.loads(allowlist_text)
            if not isinstance(parsed, list):
                raise TypeError("attack_family_allowlist JSON must be list")
            raw_values = cast(List[object], parsed)
        else:
            raw_values = [item.strip() for item in allowlist_text.split(",") if item.strip()]
    else:
        raw_values = list(cast(Sequence[object], attack_family_allowlist))

    normalized: List[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise TypeError("attack_family_allowlist items must be non-empty str")
        normalized_value = raw_value.strip()
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized or None


def _resolve_attack_shard_root(family_root: Path, attack_shard_index: int) -> Path:
    """
    Resolve one attack shard root path.

    Args:
        family_root: Family root path.
        attack_shard_index: Attack shard index.

    Returns:
        Attack shard root path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(attack_shard_index, int) or isinstance(attack_shard_index, bool) or attack_shard_index < 0:
        raise TypeError("attack_shard_index must be non-negative int")
    shard_root = family_root / "attack_shards" / f"shard_{attack_shard_index:04d}"
    validate_path_within_base(family_root, shard_root, "PW03 attack shard root")
    return shard_root


def _build_worker_root(shard_root: Path, local_worker_index: int) -> Path:
    """
    Resolve one worker root inside the current attack shard.

    Args:
        shard_root: Attack shard root.
        local_worker_index: Local worker index.

    Returns:
        Worker root path.
    """
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")
    if not isinstance(local_worker_index, int) or isinstance(local_worker_index, bool) or local_worker_index < 0:
        raise TypeError("local_worker_index must be non-negative int")
    worker_root = shard_root / "workers" / f"worker_{local_worker_index:02d}"
    validate_path_within_base(shard_root, worker_root, "PW03 worker root")
    return worker_root


def _build_worker_result_path(worker_root: Path) -> Path:
    """
    Resolve one worker result JSON path.

    Args:
        worker_root: Worker root path.

    Returns:
        Worker result JSON path.
    """
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    return worker_root / PW03_WORKER_RESULT_FILE_NAME


def _build_worker_log_paths(worker_root: Path) -> Tuple[Path, Path]:
    """
    Resolve the worker stdout and stderr log paths.

    Args:
        worker_root: Worker root path.

    Returns:
        Tuple of stdout and stderr log paths.
    """
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    logs_root = worker_root / "logs"
    return logs_root / "worker_stdout.log", logs_root / "worker_stderr.log"


def _resolve_pw02_summary_path(family_root: Path) -> Path:
    """
    Resolve the canonical PW02 summary path.

    Args:
        family_root: Family root path.

    Returns:
        PW02 summary path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    summary_path = family_root / "runtime_state" / PW02_SUMMARY_FILE_NAME
    validate_path_within_base(family_root, summary_path, "PW02 summary path")
    return summary_path


def _resolve_attack_shard_assignment(
    attack_shard_plan: Mapping[str, Any],
    *,
    attack_shard_index: int,
    attack_shard_count: int,
) -> Dict[str, Any]:
    """
    Resolve the selected attack shard assignment row.

    Args:
        attack_shard_plan: Attack shard plan payload.
        attack_shard_index: Selected attack shard index.
        attack_shard_count: Expected attack shard count.

    Returns:
        Selected attack shard row.
    """
    if attack_shard_index < 0:
        raise ValueError("attack_shard_index must be non-negative int")
    if attack_shard_count <= 0:
        raise ValueError("attack_shard_count must be positive int")

    plan_shard_count = attack_shard_plan.get("attack_shard_count")
    if not isinstance(plan_shard_count, int) or plan_shard_count <= 0:
        raise ValueError("attack_shard_plan missing attack_shard_count")
    if int(plan_shard_count) != attack_shard_count:
        raise ValueError(
            f"attack_shard_count mismatch with attack shard plan: expected={plan_shard_count}, actual={attack_shard_count}"
        )

    shards_node = attack_shard_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError("attack_shard_plan.shards must be list")
    for shard_node in cast(List[object], shards_node):
        if not isinstance(shard_node, dict):
            continue
        shard_row = cast(Dict[str, Any], shard_node)
        shard_index_value = shard_row.get("attack_shard_index")
        if isinstance(shard_index_value, int) and int(shard_index_value) == attack_shard_index:
            return shard_row

    raise ValueError(f"attack_shard_index not found in attack_shard_plan: {attack_shard_index}")


def _load_attack_event_lookup(attack_event_grid_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the attack event grid keyed by attack event id.

    Args:
        attack_event_grid_path: Attack event grid JSONL path.

    Returns:
        Attack event lookup keyed by event id.
    """
    attack_event_lookup: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(attack_event_grid_path):
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("attack_event_grid contains invalid event_id")
        if event_id in attack_event_lookup:
            raise ValueError(f"duplicate attack event_id in attack_event_grid: {event_id}")
        attack_event_lookup[event_id] = row
    return attack_event_lookup


def _load_required_bound_config(bound_config_path: Path | None) -> Tuple[Path, Dict[str, Any]]:
    """
    Load the required notebook-bound config snapshot for PW03.

    Args:
        bound_config_path: Candidate bound config path.

    Returns:
        Tuple of resolved bound config path and parsed config mapping.
    """
    if bound_config_path is None:
        raise ValueError("PW03 requires bound_config_path produced by notebook precheck")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")

    resolved_path = bound_config_path.expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"PW03 bound_config_path not found: {normalize_path_value(resolved_path)}")
    bound_cfg = load_yaml_mapping(resolved_path)
    model_snapshot_path = bound_cfg.get("model_snapshot_path")
    if not isinstance(model_snapshot_path, str) or not model_snapshot_path.strip():
        raise RuntimeError("PW03 bound config missing valid model_snapshot_path")
    snapshot_path = Path(model_snapshot_path).expanduser().resolve()
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        raise RuntimeError("PW03 bound config missing valid model_snapshot_path")
    model_source_binding = bound_cfg.get("model_source_binding")
    if not isinstance(model_source_binding, Mapping):
        raise RuntimeError("PW03 bound config missing valid model_source_binding")
    return resolved_path, bound_cfg


def _resolve_parent_event_manifest_path(pool_event: Mapping[str, Any]) -> Path:
    """
    Resolve the parent PW01 event manifest path from the PW02 pool row.

    Args:
        pool_event: Positive source pool event row.

    Returns:
        Parent event manifest path.
    """
    source_shard_root = pool_event.get("source_shard_root")
    event_index = pool_event.get("event_index")
    if not isinstance(source_shard_root, str) or not source_shard_root:
        raise ValueError("positive source pool event missing source_shard_root")
    if not isinstance(event_index, int) or event_index < 0:
        raise ValueError("positive source pool event missing event_index")

    source_shard_root_path = Path(source_shard_root).expanduser().resolve()
    event_manifest_path = source_shard_root_path / "events" / f"event_{event_index:06d}" / "event_manifest.json"
    validate_path_within_base(source_shard_root_path, event_manifest_path, "parent event manifest path")
    return event_manifest_path


def _load_parent_positive_event_lookup(
    positive_source_pool_manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Load the finalized parent positive-event lookup.

    Args:
        positive_source_pool_manifest: Positive source pool manifest payload.

    Returns:
        Lookup keyed by parent event id.
    """
    events_node = positive_source_pool_manifest.get("events")
    if not isinstance(events_node, list):
        raise ValueError("positive source pool manifest missing events list")

    parent_lookup: Dict[str, Dict[str, Any]] = {}
    for event_node in cast(List[object], events_node):
        if not isinstance(event_node, dict):
            raise ValueError("positive source pool manifest events must contain objects")
        pool_event = cast(Dict[str, Any], event_node)
        event_id = pool_event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("positive source pool event missing event_id")
        if pool_event.get("sample_role") != ACTIVE_SAMPLE_ROLE:
            raise ValueError("PW03 parent source pool must contain positive_source events only")
        parent_event_manifest_path = _resolve_parent_event_manifest_path(pool_event)
        parent_event_manifest = _load_required_json_dict(parent_event_manifest_path, f"PW01 parent event manifest {event_id}")
        if parent_event_manifest.get("event_id") != event_id:
            raise ValueError(f"parent event manifest event_id mismatch: expected={event_id}, actual={parent_event_manifest.get('event_id')}")
        parent_lookup[event_id] = {
            "pool_event": pool_event,
            "event_manifest": parent_event_manifest,
            "event_manifest_path": normalize_path_value(parent_event_manifest_path),
        }
    return parent_lookup


def _build_threshold_binding_reference(
    *,
    finalize_manifest_path: Path,
    finalize_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the immutable threshold binding reference used by PW03.

    Args:
        finalize_manifest_path: Finalize manifest path.
        finalize_manifest: Finalize manifest payload.

    Returns:
        Threshold binding reference mapping.
    """
    threshold_exports_node = finalize_manifest.get("threshold_exports")
    threshold_exports = cast(Mapping[str, Any], threshold_exports_node) if isinstance(threshold_exports_node, Mapping) else {}
    threshold_artifact_paths: Dict[str, str] = {}
    threshold_artifact_sha256: Dict[str, str] = {}
    for score_key, export_node in threshold_exports.items():
        if not isinstance(export_node, Mapping):
            continue
        export_path_value = export_node.get("path")
        if not isinstance(export_path_value, str) or not export_path_value:
            continue
        export_path = Path(export_path_value).expanduser().resolve()
        threshold_artifact_paths[str(score_key)] = normalize_path_value(export_path)
        if export_path.exists() and export_path.is_file():
            threshold_artifact_sha256[str(score_key)] = compute_file_sha256(export_path)

    positive_pool_node = finalize_manifest.get("source_pools")
    positive_pool_manifest_path = "<absent>"
    if isinstance(positive_pool_node, Mapping):
        positive_pool_payload = positive_pool_node.get(ACTIVE_SAMPLE_ROLE)
        if isinstance(positive_pool_payload, Mapping):
            manifest_path_value = positive_pool_payload.get("manifest_path")
            if isinstance(manifest_path_value, str) and manifest_path_value:
                positive_pool_manifest_path = normalize_path_value(Path(manifest_path_value).expanduser().resolve())

    return {
        "source_finalize_manifest_path": normalize_path_value(finalize_manifest_path),
        "source_finalize_manifest_digest": compute_file_sha256(finalize_manifest_path),
        "positive_source_pool_manifest_path": positive_pool_manifest_path,
        "threshold_artifact_paths": threshold_artifact_paths,
        "threshold_artifact_sha256": threshold_artifact_sha256,
        "detect_cli_threshold_injection": {
            "applied": False,
            "reason": "separate_content_and_attestation_threshold_exports",
        },
    }


def _load_parent_positive_event(
    *,
    parent_event_id: str,
    parent_event_lookup: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Load one finalized parent positive event.

    Args:
        parent_event_id: Parent positive event id.
        parent_event_lookup: Parent event lookup.

    Returns:
        Parent event payload.
    """
    parent_event = parent_event_lookup.get(parent_event_id)
    if not isinstance(parent_event, Mapping):
        raise ValueError(f"PW03 parent_event_id not found in finalized positive source pool: {parent_event_id}")
    return dict(cast(Mapping[str, Any], parent_event))


def _resolve_attack_event_spec(
    *,
    attack_event: Mapping[str, Any],
    parent_event_lookup: Mapping[str, Mapping[str, Any]],
    threshold_binding_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Resolve one fully materialized PW03 attack event spec.

    Args:
        attack_event: Attack event row from attack_event_grid.
        parent_event_lookup: Finalized parent positive-event lookup.
        threshold_binding_reference: Shared threshold binding reference.

    Returns:
        Materialized attack event spec.
    """
    attack_event_payload = dict(cast(Mapping[str, Any], attack_event))
    event_id = attack_event_payload.get("event_id")
    parent_event_id = attack_event_payload.get("parent_event_id")
    attack_family = attack_event_payload.get("attack_family")
    attack_params_digest = attack_event_payload.get("attack_params_digest")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("attack event missing event_id")
    if not isinstance(parent_event_id, str) or not parent_event_id:
        raise ValueError(f"attack event missing parent_event_id: {event_id}")
    if not isinstance(attack_family, str) or not attack_family:
        raise ValueError(f"attack event missing attack_family: {event_id}")
    if not isinstance(attack_params_digest, str) or not attack_params_digest:
        raise ValueError(f"attack event missing attack_params_digest: {event_id}")

    parent_positive_event = _load_parent_positive_event(
        parent_event_id=parent_event_id,
        parent_event_lookup=parent_event_lookup,
    )
    parent_event_manifest = cast(Dict[str, Any], parent_positive_event["event_manifest"])
    source_image_view = parent_event_manifest.get("source_image")
    if not isinstance(source_image_view, Mapping):
        raise ValueError(f"parent event source_image missing: {parent_event_id}")
    parent_source_image_path = source_image_view.get("path")
    if not isinstance(parent_source_image_path, str) or not parent_source_image_path:
        raise ValueError(f"parent source_image path missing: {parent_event_id}")
    parent_embed_record_path = parent_event_manifest.get("embed_record_path")
    parent_detect_record_path = parent_event_manifest.get("detect_record_path")
    if not isinstance(parent_embed_record_path, str) or not parent_embed_record_path:
        raise ValueError(f"parent embed_record_path missing: {parent_event_id}")
    if not isinstance(parent_detect_record_path, str) or not parent_detect_record_path:
        raise ValueError(f"parent detect_record_path missing: {parent_event_id}")

    return {
        **attack_event_payload,
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "parent_positive_event": parent_positive_event,
        "parent_event_reference": {
            "parent_event_id": parent_event_id,
            "parent_event_index": parent_event_manifest.get("event_index"),
            "parent_event_manifest_path": parent_positive_event.get("event_manifest_path"),
            "parent_embed_record_path": parent_embed_record_path,
            "parent_detect_record_path": parent_detect_record_path,
            "parent_source_image_path": parent_source_image_path,
            "parent_source_shard_root": cast(Mapping[str, Any], parent_positive_event.get("pool_event", {})).get("source_shard_root"),
            "parent_source_shard_index": cast(Mapping[str, Any], parent_positive_event.get("pool_event", {})).get("source_shard_index"),
            "parent_source_shard_manifest_path": cast(Mapping[str, Any], parent_positive_event.get("pool_event", {})).get("source_shard_manifest_path"),
        },
        "threshold_binding_reference": dict(threshold_binding_reference),
    }


def _build_attack_seed(attack_event_spec: Mapping[str, Any]) -> int:
    """
    Build one deterministic attack seed.

    Args:
        attack_event_spec: Materialized attack event spec.

    Returns:
        Deterministic integer seed.
    """
    parent_reference = cast(Mapping[str, Any], attack_event_spec.get("parent_event_reference", {}))
    parent_event_id = str(parent_reference.get("parent_event_id", ""))
    parent_positive_event = cast(Mapping[str, Any], attack_event_spec.get("parent_positive_event", {}))
    parent_event_manifest = cast(Mapping[str, Any], parent_positive_event.get("event_manifest", {}))
    seed_value = parent_event_manifest.get("seed")
    parent_seed = int(seed_value) if isinstance(seed_value, int) else 0
    digest_parts = [
        str(attack_event_spec.get("event_id", "")),
        str(parent_event_id),
        str(attack_event_spec.get("attack_family", "")),
        str(attack_event_spec.get("attack_params_digest", "")),
        str(parent_seed),
    ]
    digest_text = "::".join(digest_parts)
    return int.from_bytes(digest_text.encode("utf-8"), byteorder="little", signed=False) % (2**31 - 1)


def _relative_to_shard(shard_root: Path, path_obj: Path) -> str:
    """
    Convert a path into a shard-relative POSIX path.

    Args:
        shard_root: Attack shard root.
        path_obj: Candidate child path.

    Returns:
        Shard-relative POSIX path.
    """
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    validate_path_within_base(shard_root, path_obj, "PW03 shard child path")
    return path_obj.relative_to(shard_root).as_posix()


def _apply_attack_to_parent_image(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
) -> Dict[str, Any]:
    """
    Apply one attack to the finalized parent source image.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.

    Returns:
        Attack materialization summary.
    """
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")
    if not isinstance(event_root, Path):
        raise TypeError("event_root must be Path")

    parent_reference = cast(Mapping[str, Any], attack_event_spec.get("parent_event_reference", {}))
    parent_source_image_path_value = parent_reference.get("parent_source_image_path")
    if not isinstance(parent_source_image_path_value, str) or not parent_source_image_path_value:
        raise ValueError("attack event spec missing parent_source_image_path")
    parent_source_image_path = Path(parent_source_image_path_value).expanduser().resolve()
    if not parent_source_image_path.exists() or not parent_source_image_path.is_file():
        raise FileNotFoundError(f"parent source image not found: {normalize_path_value(parent_source_image_path)}")

    attacked_image_path = event_root / "artifacts" / "attacked_image.png"
    attack_trace_path = event_root / "artifacts" / "attack_trace.json"
    ensure_directory(attacked_image_path.parent)
    validate_path_within_base(shard_root, attacked_image_path, "attacked image path")
    validate_path_within_base(shard_root, attack_trace_path, "attack trace path")

    attack_seed = _build_attack_seed(attack_event_spec)
    attack_spec = {
        "attack_family": attack_event_spec.get("attack_family"),
        "params_version": attack_event_spec.get("attack_params_version", attack_event_spec.get("attack_config_name", "<absent>")),
        "params": copy.deepcopy(attack_event_spec.get("attack_params", {})),
        "seed": attack_seed,
    }
    with Image.open(parent_source_image_path) as parent_image:
        transformed = eval_attack_runner.apply_attack_transform(parent_image.convert("RGB"), attack_spec, rng=None)
    attacked_payload = transformed.get("payload") if isinstance(transformed, Mapping) else None
    if not isinstance(attacked_payload, Image.Image):
        raise TypeError("PW03 attacked payload must be PIL.Image.Image")
    attacked_payload.save(attacked_image_path)

    attack_trace_core = transformed.get("attack_trace") if isinstance(transformed, Mapping) else None
    if attack_trace_core is not None and not isinstance(attack_trace_core, Mapping):
        raise TypeError("PW03 attack_trace must be mapping when present")
    attack_transform_summary = {
        "attack_digest": transformed.get("attack_digest") if isinstance(transformed, Mapping) else None,
        "attack_trace_digest": transformed.get("attack_trace_digest") if isinstance(transformed, Mapping) else None,
        "attack_trace": copy.deepcopy(dict(cast(Mapping[str, Any], attack_trace_core))) if isinstance(attack_trace_core, Mapping) else None,
        "payload_type": type(attacked_payload).__name__,
    }

    attack_trace_payload = {
        "event_id": attack_event_spec.get("event_id"),
        "parent_event_id": parent_reference.get("parent_event_id"),
        "attack_family": attack_event_spec.get("attack_family"),
        "attack_config_name": attack_event_spec.get("attack_config_name"),
        "attack_condition_key": attack_event_spec.get("attack_condition_key"),
        "attack_params_version": attack_event_spec.get("attack_params_version"),
        "attack_params": copy.deepcopy(attack_event_spec.get("attack_params", {})),
        "attack_params_digest": attack_event_spec.get("attack_params_digest"),
        "attack_seed": attack_seed,
        "attack_transform": attack_transform_summary,
        "parent_source_image_path": normalize_path_value(parent_source_image_path),
        "attacked_image_path": normalize_path_value(attacked_image_path),
    }
    write_json_atomic(attack_trace_path, attack_trace_payload)
    return {
        "attack_seed": attack_seed,
        "attacked_image_path": normalize_path_value(attacked_image_path),
        "attacked_image_package_relative_path": _relative_to_shard(shard_root, attacked_image_path),
        "attack_trace_path": normalize_path_value(attack_trace_path),
        "attack_trace_package_relative_path": _relative_to_shard(shard_root, attack_trace_path),
        "sha256": compute_file_sha256(attacked_image_path),
    }


def _build_attack_detect_input_record(
    *,
    attack_event_spec: Mapping[str, Any],
    attacked_image_path: Path,
) -> Dict[str, Any]:
    """
    Build one attacked-image detect input record from the parent embed record.

    Args:
        attack_event_spec: Materialized attack event spec.
        attacked_image_path: Attacked image path.

    Returns:
        Detect input record payload.
    """
    parent_reference = cast(Mapping[str, Any], attack_event_spec.get("parent_event_reference", {}))
    parent_embed_record_path_value = parent_reference.get("parent_embed_record_path")
    if not isinstance(parent_embed_record_path_value, str) or not parent_embed_record_path_value:
        raise ValueError("attack event spec missing parent_embed_record_path")
    parent_embed_record_path = Path(parent_embed_record_path_value).expanduser().resolve()
    parent_embed_record = _load_required_json_dict(parent_embed_record_path, "PW03 parent embed record")
    attacked_image_sha256 = compute_file_sha256(attacked_image_path)
    attacked_image_path_value = normalize_path_value(attacked_image_path)

    detect_input_record = copy.deepcopy(parent_embed_record)
    detect_input_record["watermarked_path"] = attacked_image_path_value
    detect_input_record["image_path"] = attacked_image_path_value
    detect_input_record["artifact_sha256"] = attacked_image_sha256
    detect_input_record["watermarked_artifact_sha256"] = attacked_image_sha256
    detect_input_record["sample_role"] = ATTACKED_POSITIVE_SAMPLE_ROLE
    inputs_node = detect_input_record.get("inputs")
    if isinstance(inputs_node, dict):
        inputs_payload = cast(Dict[str, Any], inputs_node)
        inputs_payload["input_image_path"] = attacked_image_path_value
    detect_input_record["paper_workflow_attack_stage"] = STAGE_NAME
    detect_input_record["paper_workflow_attack_event_id"] = attack_event_spec.get("event_id")
    detect_input_record["paper_workflow_parent_event_id"] = parent_reference.get("parent_event_id")
    detect_input_record["paper_workflow_attack_family"] = attack_event_spec.get("attack_family")
    detect_input_record["paper_workflow_attack_config_name"] = attack_event_spec.get("attack_config_name")
    detect_input_record["paper_workflow_attack_params_digest"] = attack_event_spec.get("attack_params_digest")
    detect_input_record["paper_workflow_attack_condition_key"] = attack_event_spec.get("attack_condition_key")
    detect_input_record["paper_workflow_parent_source_image_path"] = parent_reference.get("parent_source_image_path")
    return detect_input_record


def _write_runtime_config_snapshot(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
    bound_cfg_obj: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Write the attacked-event runtime config snapshot.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.
        bound_cfg_obj: Notebook-bound config mapping.

    Returns:
        Runtime config snapshot summary.
    """
    runtime_cfg = copy.deepcopy(dict(bound_cfg_obj))
    parent_positive_event = cast(Mapping[str, Any], attack_event_spec.get("parent_positive_event", {}))
    parent_event_manifest = cast(Mapping[str, Any], parent_positive_event.get("event_manifest", {}))
    runtime_cfg["paper_workflow_event"] = {
        "event_id": attack_event_spec.get("event_id"),
        "event_index": attack_event_spec.get("attack_event_index", attack_event_spec.get("event_index")),
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "parent_event_id": attack_event_spec.get("parent_event_id"),
        "parent_event_index": parent_event_manifest.get("event_index"),
        "attack_family": attack_event_spec.get("attack_family"),
        "attack_config_name": attack_event_spec.get("attack_config_name"),
        "attack_params_digest": attack_event_spec.get("attack_params_digest"),
    }
    prompt_text = parent_event_manifest.get("prompt_text")
    if isinstance(prompt_text, str) and prompt_text:
        runtime_cfg["inference_prompt"] = prompt_text
    seed_value = parent_event_manifest.get("seed")
    if isinstance(seed_value, int):
        runtime_cfg["seed"] = seed_value

    runtime_config_path = event_root / "runtime_config.yaml"
    runtime_config_snapshot_path = event_root / "runtime_config_snapshot.json"
    validate_path_within_base(shard_root, runtime_config_path, "PW03 runtime_config path")
    validate_path_within_base(shard_root, runtime_config_snapshot_path, "PW03 runtime_config_snapshot path")
    write_yaml_mapping(runtime_config_path, runtime_cfg)
    write_json_atomic(runtime_config_snapshot_path, runtime_cfg)
    return {
        "runtime_config_path": normalize_path_value(runtime_config_path),
        "runtime_config_package_relative_path": _relative_to_shard(shard_root, runtime_config_path),
        "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
        "runtime_config_snapshot_package_relative_path": _relative_to_shard(shard_root, runtime_config_snapshot_path),
    }


def _write_threshold_binding_summary(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
) -> Dict[str, Any]:
    """
    Write one attacked-event threshold binding summary.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.

    Returns:
        Threshold binding summary.
    """
    summary_path = event_root / "threshold_binding_summary.json"
    validate_path_within_base(shard_root, summary_path, "PW03 threshold binding summary path")
    threshold_binding_reference = dict(cast(Mapping[str, Any], attack_event_spec.get("threshold_binding_reference", {})))
    threshold_binding_summary = {
        **threshold_binding_reference,
        "event_id": attack_event_spec.get("event_id"),
        "parent_event_id": attack_event_spec.get("parent_event_id"),
        "attack_family": attack_event_spec.get("attack_family"),
        "attack_config_name": attack_event_spec.get("attack_config_name"),
        "attack_params_digest": attack_event_spec.get("attack_params_digest"),
    }
    write_json_atomic(summary_path, threshold_binding_summary)
    return {
        "threshold_binding_summary_path": normalize_path_value(summary_path),
        "threshold_binding_summary_package_relative_path": _relative_to_shard(shard_root, summary_path),
        "threshold_binding_summary": threshold_binding_summary,
    }


def _run_command_with_gpu_monitor(
    *,
    command: Sequence[str],
    label: str,
    gpu_summary_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> Dict[str, Any]:
    """
    Execute one command through the GPU session peak wrapper.

    Args:
        command: Wrapped command.
        label: Stable label.
        gpu_summary_path: GPU summary JSON path.
        stdout_log_path: Stdout log path.
        stderr_log_path: Stderr log path.

    Returns:
        Execution summary.
    """
    monitored_command = [
        sys.executable,
        str(GPU_PEAK_SCRIPT_PATH),
        "--output-json",
        str(gpu_summary_path),
        "--label",
        str(label),
        "--sample-interval-ms",
        str(DEFAULT_SAMPLE_INTERVAL_MS),
        "--",
        *[str(item) for item in command],
    ]
    result = run_command_with_logs(monitored_command, REPO_ROOT, stdout_log_path, stderr_log_path)
    result["gpu_session_peak_path"] = normalize_path_value(gpu_summary_path)
    if gpu_summary_path.exists() and gpu_summary_path.is_file():
        result["gpu_session_peak"] = _load_required_json_dict(gpu_summary_path, "PW03 event gpu_session_peak")
    else:
        result["gpu_session_peak"] = None
    return result


def _run_attack_detect_event(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
    runtime_config_path: Path,
    attacked_image_path: Path,
) -> Dict[str, Any]:
    """
    Run detect on one attacked image.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.
        runtime_config_path: Event runtime config path.
        attacked_image_path: Attacked image path.

    Returns:
        Detect execution summary.
    """
    run_root = ensure_directory(event_root / "run")
    logs_root = ensure_directory(event_root / "logs")
    detect_input_record_path = event_root / "detect_input_record.json"
    detect_stdout_log_path = logs_root / "detect_stdout.log"
    detect_stderr_log_path = logs_root / "detect_stderr.log"
    event_gpu_summary_path = event_root / "artifacts" / "gpu_session_peak.json"
    staged_detect_record_path = shard_root / "records" / f"event_{int(attack_event_spec.get('attack_event_index', attack_event_spec.get('event_index', 0))):06d}_detect_record.json"

    validate_path_within_base(shard_root, detect_input_record_path, "PW03 detect_input_record path")
    validate_path_within_base(shard_root, staged_detect_record_path, "PW03 staged detect record path")
    ensure_directory(detect_input_record_path.parent)
    ensure_directory(staged_detect_record_path.parent)
    detect_input_record = _build_attack_detect_input_record(
        attack_event_spec=attack_event_spec,
        attacked_image_path=attacked_image_path,
    )
    write_json_atomic(detect_input_record_path, detect_input_record)

    detect_command = [
        sys.executable,
        "-m",
        "main.cli.run_detect",
        "--out",
        str(run_root),
        "--config",
        str(runtime_config_path),
        "--input",
        str(detect_input_record_path),
    ]
    detect_result = _run_command_with_gpu_monitor(
        command=detect_command,
        label=f"{STAGE_NAME}:attack_detect:{attack_event_spec.get('event_id', '<absent>')}",
        gpu_summary_path=event_gpu_summary_path,
        stdout_log_path=detect_stdout_log_path,
        stderr_log_path=detect_stderr_log_path,
    )
    if int(detect_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            "PW03 detect failed: "
            f"event_id={attack_event_spec.get('event_id')} payload={json.dumps(detect_result, ensure_ascii=False, sort_keys=True)}"
        )

    source_detect_record_path = run_root / "records" / "detect_record.json"
    if not source_detect_record_path.exists() or not source_detect_record_path.is_file():
        raise FileNotFoundError(f"PW03 detect record missing: {normalize_path_value(source_detect_record_path)}")
    detect_payload = _load_required_json_dict(source_detect_record_path, "PW03 attacked detect record")
    severity_metadata = _extract_attack_severity_metadata(attack_event_spec)
    geometry_diagnostics = _extract_attack_geometry_diagnostics(detect_payload)
    parent_reference = cast(Mapping[str, Any], attack_event_spec.get("parent_event_reference", {}))
    detect_payload["sample_role"] = ATTACKED_POSITIVE_SAMPLE_ROLE
    detect_payload["paper_workflow_attack_stage"] = STAGE_NAME
    detect_payload["paper_workflow_attack_event_id"] = attack_event_spec.get("event_id")
    detect_payload["paper_workflow_parent_event_id"] = attack_event_spec.get("parent_event_id")
    detect_payload["paper_workflow_parent_source_image_path"] = parent_reference.get("parent_source_image_path")
    detect_payload["paper_workflow_attack_family"] = attack_event_spec.get("attack_family")
    detect_payload["paper_workflow_attack_config_name"] = attack_event_spec.get("attack_config_name")
    detect_payload["paper_workflow_attack_params_digest"] = attack_event_spec.get("attack_params_digest")
    detect_payload["paper_workflow_severity_metadata"] = severity_metadata
    detect_payload["paper_workflow_geometry_diagnostics"] = geometry_diagnostics
    write_json_atomic(staged_detect_record_path, detect_payload)

    return {
        "run_root": normalize_path_value(run_root),
        "detect_input_record_path": normalize_path_value(detect_input_record_path),
        "detect_input_record_package_relative_path": _relative_to_shard(shard_root, detect_input_record_path),
        "detect_record_path": normalize_path_value(staged_detect_record_path),
        "detect_record_package_relative_path": _relative_to_shard(shard_root, staged_detect_record_path),
        "event_gpu_session_peak_path": normalize_path_value(event_gpu_summary_path),
        "event_gpu_session_peak_package_relative_path": _relative_to_shard(shard_root, event_gpu_summary_path),
        "detect_stage_result": detect_result,
        "severity_metadata": severity_metadata,
        "geometry_diagnostics": geometry_diagnostics,
    }


def _write_attack_event_manifest(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
    runtime_config_summary: Mapping[str, Any],
    threshold_binding_summary: Mapping[str, Any],
    attack_artifacts: Mapping[str, Any],
    detect_summary: Mapping[str, Any],
    worker_local_index: int,
    status: str,
    start_time: str,
    end_time: str,
    failure_reason: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    traceback_text: str | None = None,
) -> Dict[str, Any]:
    """
    Write one attacked-event manifest.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.
        runtime_config_summary: Runtime config snapshot summary.
        threshold_binding_summary: Threshold binding summary.
        attack_artifacts: Attack materialization summary.
        detect_summary: Detect execution summary.
        worker_local_index: Local worker index.
        status: Event status.
        start_time: Event start time.
        end_time: Event end time.
        failure_reason: Optional failure reason.
        exception_type: Optional exception type.
        exception_message: Optional exception message.
        traceback_text: Optional traceback string.

    Returns:
        Event manifest payload.
    """
    event_manifest_path = event_root / "event_manifest.json"
    validate_path_within_base(shard_root, event_manifest_path, "PW03 event manifest path")

    parent_reference = cast(Mapping[str, Any], attack_event_spec.get("parent_event_reference", {}))
    threshold_binding_payload_node = threshold_binding_summary.get("threshold_binding_summary", {})
    threshold_binding_payload = (
        cast(Mapping[str, Any], threshold_binding_payload_node)
        if isinstance(threshold_binding_payload_node, Mapping)
        else {}
    )
    event_manifest_payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_attack_event",
        "schema_version": "pw_stage_03_v1",
        "stage_name": STAGE_NAME,
        "status": status,
        "event_id": attack_event_spec.get("event_id"),
        "attack_event_id": attack_event_spec.get("attack_event_id", attack_event_spec.get("event_id")),
        "attack_event_index": attack_event_spec.get("attack_event_index", attack_event_spec.get("event_index")),
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "parent_event_id": attack_event_spec.get("parent_event_id"),
        "parent_event_reference": dict(parent_reference),
        "parent_source_image_path": parent_reference.get("parent_source_image_path"),
        "attack_family": attack_event_spec.get("attack_family"),
        "attack_config_name": attack_event_spec.get("attack_config_name"),
        "attack_condition_key": attack_event_spec.get("attack_condition_key"),
        "attack_params_version": attack_event_spec.get("attack_params_version"),
        "attack_params": copy.deepcopy(attack_event_spec.get("attack_params", {})),
        "attack_params_digest": attack_event_spec.get("attack_params_digest"),
        "severity_metadata": dict(cast(Mapping[str, Any], detect_summary.get("severity_metadata", {}))),
        "geometry_diagnostics": dict(cast(Mapping[str, Any], detect_summary.get("geometry_diagnostics", {}))),
        "attack_seed": attack_artifacts.get("attack_seed"),
        "runtime_config_path": runtime_config_summary.get("runtime_config_path"),
        "runtime_config_package_relative_path": runtime_config_summary.get("runtime_config_package_relative_path"),
        "runtime_config_snapshot_path": runtime_config_summary.get("runtime_config_snapshot_path"),
        "runtime_config_snapshot_package_relative_path": runtime_config_summary.get("runtime_config_snapshot_package_relative_path"),
        "threshold_binding_summary_path": threshold_binding_summary.get("threshold_binding_summary_path"),
        "threshold_binding_summary_package_relative_path": threshold_binding_summary.get("threshold_binding_summary_package_relative_path"),
        "threshold_binding_summary": dict(threshold_binding_payload),
        "source_finalize_manifest_digest": threshold_binding_payload.get("source_finalize_manifest_digest"),
        "threshold_artifact_paths": threshold_binding_payload.get("threshold_artifact_paths", {}),
        "attacked_image_path": attack_artifacts.get("attacked_image_path"),
        "attacked_image_package_relative_path": attack_artifacts.get("attacked_image_package_relative_path"),
        "attack_trace_path": attack_artifacts.get("attack_trace_path"),
        "attack_trace_package_relative_path": attack_artifacts.get("attack_trace_package_relative_path"),
        "detect_input_record_path": detect_summary.get("detect_input_record_path"),
        "detect_input_record_package_relative_path": detect_summary.get("detect_input_record_package_relative_path"),
        "detect_record_path": detect_summary.get("detect_record_path"),
        "detect_record_package_relative_path": detect_summary.get("detect_record_package_relative_path"),
        "gpu_session_peak_path": detect_summary.get("event_gpu_session_peak_path"),
        "gpu_session_peak_package_relative_path": detect_summary.get("event_gpu_session_peak_package_relative_path"),
        "worker_local_index": worker_local_index,
        "start_time": start_time,
        "end_time": end_time,
        "failure_reason": failure_reason,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "traceback": traceback_text,
        "sha256": attack_artifacts.get("sha256"),
        "stage_results": {
            "attack": {
                "attack_seed": attack_artifacts.get("attack_seed"),
                "attacked_image_path": attack_artifacts.get("attacked_image_path"),
                "attack_trace_path": attack_artifacts.get("attack_trace_path"),
            },
            "detect": detect_summary.get("detect_stage_result"),
        },
    }
    write_json_atomic(event_manifest_path, event_manifest_payload)
    event_manifest_payload["event_manifest_path"] = normalize_path_value(event_manifest_path)
    event_manifest_payload["event_manifest_package_relative_path"] = _relative_to_shard(shard_root, event_manifest_path)
    return event_manifest_payload


def _build_failed_attack_event_manifest(
    *,
    attack_event_spec: Mapping[str, Any],
    shard_root: Path,
    event_root: Path,
    worker_local_index: int,
    start_time: str,
    failure_reason: str,
    exc: Exception,
) -> Dict[str, Any]:
    """
    Build and persist a failed attack-event manifest.

    Args:
        attack_event_spec: Materialized attack event spec.
        shard_root: Attack shard root.
        event_root: Event root.
        worker_local_index: Local worker index.
        start_time: Event start time.
        failure_reason: Stable failure reason.
        exc: Raised exception.

    Returns:
        Failed event manifest payload.
    """
    return _write_attack_event_manifest(
        attack_event_spec=attack_event_spec,
        shard_root=shard_root,
        event_root=event_root,
        runtime_config_summary={},
        threshold_binding_summary={},
        attack_artifacts={},
        detect_summary={},
        worker_local_index=worker_local_index,
        status="failed",
        start_time=start_time,
        end_time=utc_now_iso(),
        failure_reason=failure_reason,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        traceback_text=traceback.format_exc(),
    )


def _build_local_worker_assignments(
    *,
    assigned_attack_events: Sequence[Mapping[str, Any]],
    attack_local_worker_count: int,
) -> List[Dict[str, Any]]:
    """
    Build deterministic shard-local worker assignments.

    Args:
        assigned_attack_events: Ordered attack event specs.
        attack_local_worker_count: Requested worker count.

    Returns:
        Worker assignment list.
    """
    _validate_attack_local_worker_count(attack_local_worker_count)
    assignments: List[Dict[str, Any]] = [
        {
            "local_worker_index": local_worker_index,
            "assigned_attack_event_ids": [],
            "assigned_attack_event_indices": [],
            "local_event_ordinals": [],
            "assigned_attack_events": [],
        }
        for local_worker_index in range(attack_local_worker_count)
    ]

    for local_event_ordinal, attack_event in enumerate(assigned_attack_events):
        local_worker_index = local_event_ordinal % attack_local_worker_count
        assignment = assignments[local_worker_index]
        assignment["assigned_attack_event_ids"].append(str(attack_event.get("event_id")))
        assignment["assigned_attack_event_indices"].append(int(attack_event.get("attack_event_index", attack_event.get("event_index", -1))))
        assignment["local_event_ordinals"].append(local_event_ordinal)
        assignment["assigned_attack_events"].append(dict(cast(Mapping[str, Any], attack_event)))
    return assignments


def _write_worker_plan(
    *,
    worker_root: Path,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    shard_root: Path,
    bound_config_path: Path,
    assignment: Mapping[str, Any],
) -> Path:
    """
    Write one PW03 worker plan.

    Args:
        worker_root: Worker root path.
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        shard_root: Attack shard root.
        bound_config_path: Bound config path.
        assignment: Worker assignment payload.

    Returns:
        Worker plan path.
    """
    plan_path = worker_root / PW03_WORKER_PLAN_FILE_NAME
    validate_path_within_base(shard_root, plan_path, "PW03 worker plan path")
    payload = {
        "artifact_type": "paper_workflow_attack_shard_worker_plan",
        "schema_version": "pw_stage_03_v1",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "attack_shard_index": attack_shard_index,
        "attack_shard_count": attack_shard_count,
        "attack_local_worker_count": attack_local_worker_count,
        "local_worker_index": local_worker_index,
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "shard_root": normalize_path_value(shard_root),
        "bound_config_path": normalize_path_value(bound_config_path),
        "assigned_attack_event_ids": list(cast(List[str], assignment.get("assigned_attack_event_ids", []))),
        "assigned_attack_event_indices": list(cast(List[int], assignment.get("assigned_attack_event_indices", []))),
        "local_event_ordinals": list(cast(List[int], assignment.get("local_event_ordinals", []))),
        "assigned_attack_events": [dict(cast(Mapping[str, Any], event)) for event in cast(List[Mapping[str, Any]], assignment.get("assigned_attack_events", []))],
    }
    write_json_atomic(plan_path, payload)
    return plan_path


def _build_worker_command(
    *,
    drive_project_root: Path,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    worker_plan_path: Path,
) -> List[str]:
    """
    Build one worker subprocess command.

    Args:
        drive_project_root: Drive project root.
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        worker_plan_path: Worker plan path.

    Returns:
        Worker subprocess command.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--drive-project-root",
        str(drive_project_root),
        "--family-id",
        family_id,
        "--attack-shard-index",
        str(attack_shard_index),
        "--attack-shard-count",
        str(attack_shard_count),
        "--attack-local-worker-count",
        str(attack_local_worker_count),
        "--local-worker-index",
        str(local_worker_index),
        "--worker-plan-path",
        str(worker_plan_path),
    ]


def _prepare_local_worker_plans(
    *,
    drive_project_root: Path,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    shard_root: Path,
    bound_config_path: Path,
    assigned_attack_events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare worker plans for the selected attack shard.

    Args:
        drive_project_root: Drive project root.
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        shard_root: Attack shard root.
        bound_config_path: Bound config path.
        assigned_attack_events: Ordered attack event specs.

    Returns:
        Worker plan summaries.
    """
    assignments = _build_local_worker_assignments(
        assigned_attack_events=assigned_attack_events,
        attack_local_worker_count=attack_local_worker_count,
    )
    worker_plans: List[Dict[str, Any]] = []
    for assignment in assignments:
        local_worker_index = int(assignment["local_worker_index"])
        worker_root = ensure_directory(_build_worker_root(shard_root, local_worker_index))
        worker_plan_path = _write_worker_plan(
            worker_root=worker_root,
            family_id=family_id,
            attack_shard_index=attack_shard_index,
            attack_shard_count=attack_shard_count,
            attack_local_worker_count=attack_local_worker_count,
            local_worker_index=local_worker_index,
            shard_root=shard_root,
            bound_config_path=bound_config_path,
            assignment=assignment,
        )
        worker_result_path = _build_worker_result_path(worker_root)
        stdout_log_path, stderr_log_path = _build_worker_log_paths(worker_root)
        ensure_directory(stdout_log_path.parent)
        ensure_directory(stderr_log_path.parent)
        worker_plans.append(
            {
                "local_worker_index": local_worker_index,
                "worker_root": normalize_path_value(worker_root),
                "worker_plan_path": normalize_path_value(worker_plan_path),
                "worker_result_path": normalize_path_value(worker_result_path),
                "stdout_log_path": normalize_path_value(stdout_log_path),
                "stderr_log_path": normalize_path_value(stderr_log_path),
                "assigned_attack_event_ids": list(cast(List[str], assignment["assigned_attack_event_ids"])),
                "assigned_attack_event_indices": list(cast(List[int], assignment["assigned_attack_event_indices"])),
                "local_event_ordinals": list(cast(List[int], assignment["local_event_ordinals"])),
                "command": _build_worker_command(
                    drive_project_root=drive_project_root,
                    family_id=family_id,
                    attack_shard_index=attack_shard_index,
                    attack_shard_count=attack_shard_count,
                    attack_local_worker_count=attack_local_worker_count,
                    local_worker_index=local_worker_index,
                    worker_plan_path=worker_plan_path,
                ),
            }
        )
    return worker_plans


def _run_local_worker_plans(worker_plans: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run worker plans in subprocesses and collect their results.

    Args:
        worker_plans: Worker plan summaries.

    Returns:
        Tuple of worker execution summaries and worker result payloads.
    """
    env_mapping = build_repo_import_subprocess_env(repo_root=REPO_ROOT)
    launches: List[Dict[str, Any]] = []
    for worker_plan in worker_plans:
        worker_root = Path(str(worker_plan["worker_root"]))
        stdout_log_path = Path(str(worker_plan["stdout_log_path"]))
        stderr_log_path = Path(str(worker_plan["stderr_log_path"]))
        ensure_directory(worker_root)
        ensure_directory(stdout_log_path.parent)
        ensure_directory(stderr_log_path.parent)
        stdout_handle = stdout_log_path.open("w", encoding="utf-8")
        stderr_handle = stderr_log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                list(cast(Sequence[str], worker_plan["command"])),
                cwd=str(REPO_ROOT),
                env=env_mapping,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
        launches.append(
            {
                "local_worker_index": int(worker_plan["local_worker_index"]),
                "worker_root": worker_root,
                "worker_plan_path": Path(str(worker_plan["worker_plan_path"])),
                "worker_result_path": Path(str(worker_plan["worker_result_path"])),
                "stdout_log_path": stdout_log_path,
                "stderr_log_path": stderr_log_path,
                "assigned_attack_event_ids": list(cast(Sequence[str], worker_plan["assigned_attack_event_ids"])),
                "command": list(cast(Sequence[str], worker_plan["command"])),
                "process": process,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
            }
        )

    worker_executions: List[Dict[str, Any]] = []
    worker_results: List[Dict[str, Any]] = []
    for launch in launches:
        process = cast(subprocess.Popen[str], launch["process"])
        return_code = int(process.wait())
        cast(Any, launch["stdout_handle"]).close()
        cast(Any, launch["stderr_handle"]).close()
        worker_result_path = cast(Path, launch["worker_result_path"])
        result_exists = worker_result_path.exists() and worker_result_path.is_file()
        worker_execution = {
            "local_worker_index": int(launch["local_worker_index"]),
            "worker_root": normalize_path_value(cast(Path, launch["worker_root"])),
            "worker_plan_path": normalize_path_value(cast(Path, launch["worker_plan_path"])),
            "worker_result_path": normalize_path_value(worker_result_path),
            "stdout_log_path": normalize_path_value(cast(Path, launch["stdout_log_path"])),
            "stderr_log_path": normalize_path_value(cast(Path, launch["stderr_log_path"])),
            "assigned_attack_event_ids": list(cast(List[str], launch["assigned_attack_event_ids"])),
            "command": list(cast(List[str], launch["command"])),
            "return_code": return_code,
            "result_exists": result_exists,
        }
        worker_executions.append(worker_execution)
        if result_exists:
            worker_results.append(
                _load_required_json_dict(
                    worker_result_path,
                    f"PW03 worker result {worker_execution['local_worker_index']}",
                )
            )
    return worker_executions, worker_results


def _extract_visible_gpu_peak_memory_mib(gpu_peak_payload: Mapping[str, Any]) -> int | None:
    """
    Extract the maximum visible-GPU peak memory from one GPU summary payload.

    Args:
        gpu_peak_payload: Raw or aggregated GPU summary payload.

    Returns:
        Maximum visible-GPU peak memory in MiB when available.
    """
    visible_gpus_node = gpu_peak_payload.get("visible_gpus")
    if not isinstance(visible_gpus_node, list):
        return None

    peak_values: List[int] = []
    for gpu_node in cast(List[object], visible_gpus_node):
        if not isinstance(gpu_node, Mapping):
            continue
        peak_value = gpu_node.get("peak_memory_used_mib")
        if isinstance(peak_value, (int, float)) and not isinstance(peak_value, bool):
            peak_values.append(int(peak_value))
    return max(peak_values) if peak_values else None


def _normalize_gpu_peak_payload(gpu_peak_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw and aggregated GPU summaries into one peak-view schema.

    Args:
        gpu_peak_payload: Raw wrapper payload or worker/shard aggregate payload.

    Returns:
        Normalized peak-view payload.
    """
    peak_memory_value = gpu_peak_payload.get("session_board_peak_memory_used_mib")
    if not isinstance(peak_memory_value, (int, float)) or isinstance(peak_memory_value, bool):
        peak_memory_value = gpu_peak_payload.get("peak_memory_mib")
    peak_memory_mib = int(peak_memory_value) if isinstance(peak_memory_value, (int, float)) and not isinstance(peak_memory_value, bool) else None
    if peak_memory_mib is None:
        peak_memory_mib = _extract_visible_gpu_peak_memory_mib(gpu_peak_payload)

    peak_timestamp_value = gpu_peak_payload.get("peak_observed_at_utc")
    if not isinstance(peak_timestamp_value, str) or not peak_timestamp_value:
        peak_timestamp_value = gpu_peak_payload.get("peak_timestamp")
    peak_timestamp = peak_timestamp_value if isinstance(peak_timestamp_value, str) and peak_timestamp_value else None

    device_name_value = gpu_peak_payload.get("peak_gpu_name")
    if not isinstance(device_name_value, str) or not device_name_value:
        device_name_value = gpu_peak_payload.get("device_name")
    device_name = device_name_value if isinstance(device_name_value, str) and device_name_value else None

    visible_gpus_node = gpu_peak_payload.get("visible_gpus")
    visible_gpus = [
        dict(cast(Mapping[str, Any], gpu_node))
        for gpu_node in cast(List[object], visible_gpus_node)
        if isinstance(gpu_node, Mapping)
    ] if isinstance(visible_gpus_node, list) else []
    visible_gpu_count_value = gpu_peak_payload.get("visible_gpu_count")
    visible_gpu_count = int(visible_gpu_count_value) if isinstance(visible_gpu_count_value, int) and not isinstance(visible_gpu_count_value, bool) else len(visible_gpus)

    return {
        "peak_memory_mib": peak_memory_mib,
        "peak_timestamp": peak_timestamp,
        "device_name": device_name,
        "visible_gpu_count": visible_gpu_count,
        "visible_gpus": visible_gpus,
        "wrapped_return_code": gpu_peak_payload.get("wrapped_return_code"),
        "worker_local_index": gpu_peak_payload.get("worker_local_index"),
        "summary_path": gpu_peak_payload.get("summary_path"),
    }


def _aggregate_gpu_session_peaks(
    *,
    family_id: str,
    attack_shard_index: int,
    attack_local_worker_count: int,
    gpu_peak_payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate worker and event GPU peak payloads into one shard summary.

    Args:
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_local_worker_count: Local worker count.
        gpu_peak_payloads: Event or worker GPU payloads.

    Returns:
        Aggregated GPU session peak payload.
    """
    payloads = [dict(cast(Mapping[str, Any], payload)) for payload in gpu_peak_payloads if isinstance(payload, Mapping)]
    if not payloads:
        return {
            "status": "absent",
            "stage_name": STAGE_NAME,
            "family_id": family_id,
            "attack_shard_index": attack_shard_index,
            "attack_local_worker_count": attack_local_worker_count,
            "monitor_source": "nvidia-smi",
            "device_name": None,
            "peak_memory_mib": None,
            "peak_timestamp": None,
            "wrapped_command_count": 0,
            "wrapped_command_return_codes": [],
            "worker_local_peaks": [],
        }

    normalized_payloads = [_normalize_gpu_peak_payload(payload) for payload in payloads]
    peak_payload = normalized_payloads[0]
    for normalized_payload in normalized_payloads[1:]:
        candidate_peak_memory_mib = normalized_payload.get("peak_memory_mib")
        selected_peak_memory_mib = peak_payload.get("peak_memory_mib")
        # Equal peaks keep the first observed payload to make timestamp tie-break stable.
        if isinstance(candidate_peak_memory_mib, int) and (
            not isinstance(selected_peak_memory_mib, int) or candidate_peak_memory_mib > selected_peak_memory_mib
        ):
            peak_payload = normalized_payload

    wrapped_command_return_codes = [
        payload.get("wrapped_return_code")
        for payload in normalized_payloads
    ]
    worker_local_peaks: List[Dict[str, Any]] = []
    for payload in normalized_payloads:
        worker_local_index = payload.get("worker_local_index")
        worker_local_peaks.append(
            {
                "worker_local_index": worker_local_index,
                "peak_memory_mib": payload.get("peak_memory_mib"),
                "peak_timestamp": payload.get("peak_timestamp"),
                "device_name": payload.get("device_name"),
                "summary_path": payload.get("summary_path"),
            }
        )
    return {
        "status": "ok",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "attack_shard_index": attack_shard_index,
        "attack_local_worker_count": attack_local_worker_count,
        "monitor_source": "nvidia-smi",
        "device_name": peak_payload.get("device_name"),
        "peak_memory_mib": peak_payload.get("peak_memory_mib"),
        "peak_timestamp": peak_payload.get("peak_timestamp"),
        "wrapped_command_count": len(normalized_payloads),
        "wrapped_command_return_codes": wrapped_command_return_codes,
        "worker_local_peaks": worker_local_peaks,
        "visible_gpu_count": peak_payload.get("visible_gpu_count"),
        "visible_gpus": peak_payload.get("visible_gpus"),
    }


def _write_worker_gpu_session_peak(
    *,
    family_id: str,
    attack_shard_index: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    worker_root: Path,
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate event GPU peaks into one worker-local summary.

    Args:
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        worker_root: Worker root path.
        events: Event manifest payloads.

    Returns:
        Worker GPU summary payload.
    """
    gpu_peak_payloads: List[Dict[str, Any]] = []
    for event in events:
        gpu_peak_path_value = event.get("gpu_session_peak_path")
        if not isinstance(gpu_peak_path_value, str) or not gpu_peak_path_value:
            continue
        gpu_peak_path = Path(gpu_peak_path_value).expanduser().resolve()
        if not gpu_peak_path.exists() or not gpu_peak_path.is_file():
            continue
        payload = _load_required_json_dict(gpu_peak_path, f"PW03 worker event gpu peak {gpu_peak_path_value}")
        payload["worker_local_index"] = local_worker_index
        payload["summary_path"] = normalize_path_value(gpu_peak_path)
        gpu_peak_payloads.append(payload)

    worker_gpu_summary = _aggregate_gpu_session_peaks(
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_local_worker_count=attack_local_worker_count,
        gpu_peak_payloads=gpu_peak_payloads,
    )
    worker_gpu_summary["worker_local_index"] = local_worker_index
    worker_gpu_summary_path = worker_root / "gpu_session_peak.json"
    write_json_atomic(worker_gpu_summary_path, worker_gpu_summary)
    worker_gpu_summary["summary_path"] = normalize_path_value(worker_gpu_summary_path)
    return worker_gpu_summary


def _build_worker_result_payload(
    *,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    worker_root: Path,
    worker_plan_path: Path,
    assigned_attack_event_ids: Sequence[str],
    assigned_attack_event_indices: Sequence[int],
    events: Sequence[Mapping[str, Any]],
    status: str,
    worker_gpu_summary: Mapping[str, Any],
    failure_reason: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    traceback_text: str | None = None,
) -> Dict[str, Any]:
    """
    Build the structured PW03 worker result payload.

    Args:
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        worker_root: Worker root path.
        worker_plan_path: Worker plan path.
        assigned_attack_event_ids: Assigned attack event ids.
        assigned_attack_event_indices: Assigned attack event indices.
        events: Event manifest payloads.
        status: Worker status.
        worker_gpu_summary: Worker GPU summary.
        failure_reason: Optional failure reason.
        exception_type: Optional exception type.
        exception_message: Optional exception message.
        traceback_text: Optional traceback text.

    Returns:
        Worker result payload.
    """
    completed_event_ids = [
        str(event.get("event_id"))
        for event in events
        if event.get("status") == "completed"
    ]
    failed_event_ids = [
        str(event.get("event_id"))
        for event in events
        if event.get("status") != "completed"
    ]
    return {
        "artifact_type": "paper_workflow_attack_shard_worker_result",
        "schema_version": "pw_stage_03_v1",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "attack_shard_index": attack_shard_index,
        "attack_shard_count": attack_shard_count,
        "attack_local_worker_count": attack_local_worker_count,
        "local_worker_index": local_worker_index,
        "worker_root": normalize_path_value(worker_root),
        "worker_plan_path": normalize_path_value(worker_plan_path),
        "worker_result_path": normalize_path_value(_build_worker_result_path(worker_root)),
        "status": status,
        "assigned_attack_event_ids": [str(event_id) for event_id in assigned_attack_event_ids],
        "assigned_attack_event_indices": [int(event_index) for event_index in assigned_attack_event_indices],
        "assigned_attack_event_count": len(assigned_attack_event_ids),
        "completed_event_ids": completed_event_ids,
        "failed_event_ids": failed_event_ids,
        "completed_event_count": len(completed_event_ids),
        "failed_event_count": len(failed_event_ids),
        "events": [dict(cast(Mapping[str, Any], event)) for event in events],
        "worker_gpu_session_peak_path": worker_gpu_summary.get("summary_path"),
        "worker_gpu_session_peak": dict(cast(Mapping[str, Any], worker_gpu_summary)),
        "failure_reason": failure_reason,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "traceback": traceback_text,
    }


def _run_attack_event_by_worker(
    *,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    worker_root: Path,
    shard_root: Path,
    bound_cfg_obj: Mapping[str, Any],
    assigned_attack_events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Execute the assigned attack events for one worker.

    Args:
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        worker_root: Worker root path.
        shard_root: Attack shard root.
        bound_cfg_obj: Notebook-bound config mapping.
        assigned_attack_events: Assigned attack event specs.

    Returns:
        Worker result payload.
    """
    executed_events: List[Dict[str, Any]] = []
    worker_plan_path = worker_root / PW03_WORKER_PLAN_FILE_NAME
    for attack_event_spec in assigned_attack_events:
        attack_event_index = int(attack_event_spec.get("attack_event_index", attack_event_spec.get("event_index", -1)))
        event_root = ensure_directory(shard_root / "events" / f"event_{attack_event_index:06d}")
        event_start_time = utc_now_iso()
        try:
            attack_artifacts = _apply_attack_to_parent_image(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
            )
            runtime_config_summary = _write_runtime_config_snapshot(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
                bound_cfg_obj=bound_cfg_obj,
            )
            threshold_binding_summary = _write_threshold_binding_summary(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
            )
            detect_summary = _run_attack_detect_event(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
                runtime_config_path=Path(str(runtime_config_summary["runtime_config_path"])),
                attacked_image_path=Path(str(attack_artifacts["attacked_image_path"])),
            )
            event_manifest = _write_attack_event_manifest(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
                runtime_config_summary=runtime_config_summary,
                threshold_binding_summary=threshold_binding_summary,
                attack_artifacts=attack_artifacts,
                detect_summary=detect_summary,
                worker_local_index=local_worker_index,
                status="completed",
                start_time=event_start_time,
                end_time=utc_now_iso(),
            )
        except Exception as exc:
            event_manifest = _build_failed_attack_event_manifest(
                attack_event_spec=attack_event_spec,
                shard_root=shard_root,
                event_root=event_root,
                worker_local_index=local_worker_index,
                start_time=event_start_time,
                failure_reason="pw03_attack_event_failed",
                exc=exc,
            )
        executed_events.append(event_manifest)

    worker_gpu_summary = _write_worker_gpu_session_peak(
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_local_worker_count=attack_local_worker_count,
        local_worker_index=local_worker_index,
        worker_root=worker_root,
        events=executed_events,
    )
    worker_status = "completed" if all(event.get("status") == "completed" for event in executed_events) else "failed"
    worker_result = _build_worker_result_payload(
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_shard_count=attack_shard_count,
        attack_local_worker_count=attack_local_worker_count,
        local_worker_index=local_worker_index,
        worker_root=worker_root,
        worker_plan_path=worker_plan_path,
        assigned_attack_event_ids=[str(event.get("event_id")) for event in assigned_attack_events],
        assigned_attack_event_indices=[int(event.get("attack_event_index", event.get("event_index", -1))) for event in assigned_attack_events],
        events=executed_events,
        status=worker_status,
        worker_gpu_summary=worker_gpu_summary,
        failure_reason=None if worker_status == "completed" else "pw03_worker_event_execution_failed",
    )
    write_json_atomic(_build_worker_result_path(worker_root), worker_result)
    return worker_result


def run_pw03_attack_event_worker(
    *,
    drive_project_root: Path,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    local_worker_index: int,
    worker_plan_path: Path,
) -> Dict[str, Any]:
    """
    功能：执行一个 PW03 shard-local worker。 

    Execute one PW03 shard-local worker from a persisted worker plan.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        local_worker_index: Local worker index.
        worker_plan_path: Worker plan JSON path.

    Returns:
        Worker result payload.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_shard_index, int) or attack_shard_index < 0:
        raise TypeError("attack_shard_index must be non-negative int")
    if not isinstance(attack_shard_count, int) or attack_shard_count <= 0:
        raise TypeError("attack_shard_count must be positive int")
    _validate_attack_local_worker_count(attack_local_worker_count)
    if not isinstance(local_worker_index, int) or local_worker_index < 0 or local_worker_index >= attack_local_worker_count:
        raise TypeError("local_worker_index must be within attack_local_worker_count")
    if not isinstance(worker_plan_path, Path):
        raise TypeError("worker_plan_path must be Path")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    worker_plan = _load_required_json_dict(worker_plan_path, "PW03 worker plan")
    if worker_plan.get("family_id") != family_id:
        raise ValueError("worker plan family_id mismatch")
    if int(worker_plan.get("attack_shard_index", -1)) != attack_shard_index:
        raise ValueError("worker plan attack_shard_index mismatch")
    if int(worker_plan.get("attack_shard_count", -1)) != attack_shard_count:
        raise ValueError("worker plan attack_shard_count mismatch")
    if int(worker_plan.get("attack_local_worker_count", -1)) != attack_local_worker_count:
        raise ValueError("worker plan attack_local_worker_count mismatch")
    if int(worker_plan.get("local_worker_index", -1)) != local_worker_index:
        raise ValueError("worker plan local_worker_index mismatch")

    shard_root_value = worker_plan.get("shard_root")
    if not isinstance(shard_root_value, str) or not shard_root_value:
        raise ValueError("worker plan missing shard_root")
    shard_root = Path(shard_root_value).expanduser().resolve()
    validate_path_within_base(family_root, shard_root, "PW03 worker shard root")
    worker_root = ensure_directory(_build_worker_root(shard_root, local_worker_index))

    bound_config_path_value = worker_plan.get("bound_config_path")
    if not isinstance(bound_config_path_value, str) or not bound_config_path_value:
        raise ValueError("worker plan missing bound_config_path")
    _, bound_cfg_obj = _load_required_bound_config(Path(bound_config_path_value))

    assigned_attack_events_node = worker_plan.get("assigned_attack_events")
    if not isinstance(assigned_attack_events_node, list):
        raise ValueError("worker plan assigned_attack_events must be list")
    assigned_attack_events = [
        cast(Dict[str, Any], event_node)
        for event_node in cast(List[object], assigned_attack_events_node)
        if isinstance(event_node, dict)
    ]
    if len(assigned_attack_events) != len(assigned_attack_events_node):
        raise ValueError("worker plan assigned_attack_events must contain objects")

    return _run_attack_event_by_worker(
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_shard_count=attack_shard_count,
        attack_local_worker_count=attack_local_worker_count,
        local_worker_index=local_worker_index,
        worker_root=worker_root,
        shard_root=shard_root,
        bound_cfg_obj=bound_cfg_obj,
        assigned_attack_events=assigned_attack_events,
    )


def _collect_worker_events(worker_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect worker-completed event manifests in stable event-index order.

    Args:
        worker_results: Worker result payloads.

    Returns:
        Ordered event manifest payloads.
    """
    events: List[Dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for worker_result in worker_results:
        events_node = worker_result.get("events")
        if not isinstance(events_node, list):
            continue
        for event_node in cast(List[object], events_node):
            if not isinstance(event_node, dict):
                raise ValueError("worker result events must contain objects")
            event_payload = cast(Dict[str, Any], event_node)
            event_id = event_payload.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("worker event missing event_id")
            if event_id in seen_event_ids:
                raise ValueError(f"PW03 worker event overlap detected: {event_id}")
            seen_event_ids.add(event_id)
            events.append(event_payload)
    events.sort(key=lambda item: int(item.get("attack_event_index", item.get("event_index", -1))))
    return events


def _build_worker_outcomes(
    *,
    worker_plans: Sequence[Mapping[str, Any]],
    worker_executions: Sequence[Mapping[str, Any]],
    worker_results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build manifest-friendly worker outcome rows.

    Args:
        worker_plans: Worker plan summaries.
        worker_executions: Worker execution summaries.
        worker_results: Worker result payloads.

    Returns:
        Ordered worker outcome rows.
    """
    execution_by_index = {
        int(execution["local_worker_index"]): dict(cast(Mapping[str, Any], execution))
        for execution in worker_executions
    }
    result_by_index = {
        int(result["local_worker_index"]): dict(cast(Mapping[str, Any], result))
        for result in worker_results
        if isinstance(result.get("local_worker_index"), int)
    }
    rows: List[Dict[str, Any]] = []
    for worker_plan in worker_plans:
        local_worker_index = int(worker_plan["local_worker_index"])
        execution = execution_by_index.get(local_worker_index, {})
        result = result_by_index.get(local_worker_index, {})
        rows.append(
            {
                "local_worker_index": local_worker_index,
                "worker_root": worker_plan.get("worker_root"),
                "worker_plan_path": worker_plan.get("worker_plan_path"),
                "worker_result_path": worker_plan.get("worker_result_path"),
                "stdout_log_path": worker_plan.get("stdout_log_path"),
                "stderr_log_path": worker_plan.get("stderr_log_path"),
                "assigned_attack_event_ids": worker_plan.get("assigned_attack_event_ids", []),
                "assigned_attack_event_indices": worker_plan.get("assigned_attack_event_indices", []),
                "local_event_ordinals": worker_plan.get("local_event_ordinals", []),
                "command": worker_plan.get("command"),
                "return_code": execution.get("return_code"),
                "result_exists": execution.get("result_exists", False),
                "status": result.get("status", "not_started"),
                "completed_event_ids": result.get("completed_event_ids", []),
                "failed_event_ids": result.get("failed_event_ids", []),
                "failure_reason": result.get("failure_reason"),
                "exception_type": result.get("exception_type"),
                "exception_message": result.get("exception_message"),
                "worker_gpu_session_peak_path": result.get("worker_gpu_session_peak_path"),
            }
        )
    return rows


def _write_attack_shard_manifest(
    *,
    family_id: str,
    shard_root: Path,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int,
    attack_family_allowlist: Sequence[str] | None,
    bound_config_path: Path,
    worker_execution_mode: str,
    assigned_attack_events: Sequence[Mapping[str, Any]],
    worker_outcomes: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    gpu_session_peak_path: Path,
    runtime_summary_path: Path,
    status: str,
    failure_reason: str | None = None,
    traceback_text: str | None = None,
) -> Dict[str, Any]:
    """
    Write one attacked shard manifest.

    Args:
        family_id: Family identifier.
        shard_root: Attack shard root.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Local worker count.
        attack_family_allowlist: Optional attack-family allowlist.
        bound_config_path: Bound config path.
        worker_execution_mode: Worker execution mode token.
        assigned_attack_events: Ordered attack event specs.
        worker_outcomes: Worker outcome rows.
        events: Ordered attacked event manifests.
        gpu_session_peak_path: Shard GPU summary path.
        runtime_summary_path: Runtime summary path.
        status: Shard status.
        failure_reason: Optional failure reason.
        traceback_text: Optional traceback text.

    Returns:
        Shard manifest payload.
    """
    completed_event_count = sum(1 for event in events if event.get("status") == "completed")
    failed_event_count = sum(1 for event in events if event.get("status") != "completed")
    manifest_path = shard_root / PW03_SHARD_MANIFEST_FILE_NAME
    payload = {
        "artifact_type": "paper_workflow_attack_shard_manifest",
        "schema_version": "pw_stage_03_v1",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "sample_role": ATTACKED_POSITIVE_SAMPLE_ROLE,
        "attack_shard_index": attack_shard_index,
        "attack_shard_count": attack_shard_count,
        "attack_local_worker_count": attack_local_worker_count,
        "stage_worker_count": attack_local_worker_count,
        "worker_execution_mode": worker_execution_mode,
        "attack_family_allowlist": list(attack_family_allowlist) if attack_family_allowlist is not None else None,
        "bound_config_path": normalize_path_value(bound_config_path),
        "shard_root": normalize_path_value(shard_root),
        "status": status,
        "failure_reason": failure_reason,
        "traceback": traceback_text,
        "event_count": len(events),
        "assigned_attack_event_count": len(assigned_attack_events),
        "completed_event_count": completed_event_count,
        "failed_event_count": failed_event_count,
        "event_ids": [str(event.get("event_id")) for event in events],
        "assigned_attack_event_ids": [str(event.get("event_id")) for event in assigned_attack_events],
        "assigned_parent_event_ids": [str(event.get("parent_event_id")) for event in assigned_attack_events],
        "assigned_attack_families": sorted({str(event.get("attack_family")) for event in assigned_attack_events}),
        "assigned_attack_config_names": [str(event.get("attack_config_name")) for event in assigned_attack_events],
        "worker_outcomes": [dict(cast(Mapping[str, Any], row)) for row in worker_outcomes],
        "events": [dict(cast(Mapping[str, Any], event)) for event in events],
        "gpu_session_peak_path": normalize_path_value(gpu_session_peak_path),
        "runtime_summary_path": normalize_path_value(runtime_summary_path),
    }
    write_json_atomic(manifest_path, payload)
    payload["shard_manifest_path"] = normalize_path_value(manifest_path)
    return payload


def run_pw03_attack_event_shard(
    *,
    drive_project_root: Path,
    family_id: str,
    attack_shard_index: int,
    attack_shard_count: int,
    attack_local_worker_count: int = DEFAULT_ATTACK_LOCAL_WORKER_COUNT,
    attack_family_allowlist: Sequence[str] | str | None = None,
    bound_config_path: Path | None = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行一个 PW03 attacked-event shard。 

    Execute one PW03 attacked-event shard from the finalized positive source
    pool and the PW00 attack shard plan.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        attack_shard_index: Attack shard index.
        attack_shard_count: Attack shard count.
        attack_local_worker_count: Shard-local worker count.
        attack_family_allowlist: Optional attack-family allowlist.
        bound_config_path: Notebook-bound config snapshot path.
        force_rerun: Whether to clear the existing shard root before rerun.

    Returns:
        Shard manifest payload.

    Raises:
        FileNotFoundError: If required PW00 or PW02 artifacts are missing.
        ValueError: If shard parameters or assignment metadata are inconsistent.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_shard_index, int) or attack_shard_index < 0:
        raise TypeError("attack_shard_index must be non-negative int")
    if not isinstance(attack_shard_count, int) or attack_shard_count <= 0:
        raise TypeError("attack_shard_count must be positive int")
    _validate_attack_local_worker_count(attack_local_worker_count)

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    family_manifest_path = family_root / "manifests" / "paper_eval_family_manifest.json"
    attack_event_grid_path = family_root / "manifests" / "attack_event_grid.jsonl"
    attack_shard_plan_path = family_root / "manifests" / "attack_shard_plan.json"
    pw02_summary_path = _resolve_pw02_summary_path(family_root)
    resolved_bound_config_path, bound_cfg_obj = _load_required_bound_config(bound_config_path)
    attack_family_allowlist_values = _parse_attack_family_allowlist(attack_family_allowlist)

    _load_required_json_dict(family_manifest_path, "paper eval family manifest")
    attack_shard_plan = _load_required_json_dict(attack_shard_plan_path, "PW03 attack shard plan")
    attack_shard_assignment = _resolve_attack_shard_assignment(
        attack_shard_plan,
        attack_shard_index=attack_shard_index,
        attack_shard_count=attack_shard_count,
    )
    attack_event_lookup = _load_attack_event_lookup(attack_event_grid_path)
    pw02_summary = _load_required_json_dict(pw02_summary_path, "PW02 summary")
    finalize_manifest_path_value = pw02_summary.get("paper_source_finalize_manifest_path")
    if not isinstance(finalize_manifest_path_value, str) or not finalize_manifest_path_value:
        raise ValueError("PW02 summary missing paper_source_finalize_manifest_path")
    finalize_manifest_path = Path(finalize_manifest_path_value).expanduser().resolve()
    finalize_manifest = _load_required_json_dict(finalize_manifest_path, "paper source finalize manifest")
    source_pools_node = finalize_manifest.get("source_pools")
    if not isinstance(source_pools_node, Mapping):
        raise ValueError("paper source finalize manifest missing source_pools")
    positive_source_pool_node = source_pools_node.get(ACTIVE_SAMPLE_ROLE)
    if not isinstance(positive_source_pool_node, Mapping):
        raise ValueError("paper source finalize manifest missing positive_source pool")
    positive_pool_manifest_path_value = positive_source_pool_node.get("manifest_path")
    if not isinstance(positive_pool_manifest_path_value, str) or not positive_pool_manifest_path_value:
        raise ValueError("paper source finalize manifest missing positive_source manifest_path")
    positive_source_pool_manifest = _load_required_json_dict(
        Path(positive_pool_manifest_path_value).expanduser().resolve(),
        "positive_source pool manifest",
    )
    parent_event_lookup = _load_parent_positive_event_lookup(positive_source_pool_manifest)
    threshold_binding_reference = _build_threshold_binding_reference(
        finalize_manifest_path=finalize_manifest_path,
        finalize_manifest=finalize_manifest,
    )

    assigned_attack_event_ids = cast(List[str], attack_shard_assignment.get("assigned_attack_event_ids", []))
    assigned_attack_events: List[Dict[str, Any]] = []
    for attack_event_id in assigned_attack_event_ids:
        attack_event = attack_event_lookup.get(attack_event_id)
        if attack_event is None:
            raise ValueError(f"attack_shard_plan references missing attack_event_id: {attack_event_id}")
        if attack_family_allowlist_values is not None:
            attack_family = attack_event.get("attack_family")
            if attack_family not in attack_family_allowlist_values:
                continue
        assigned_attack_events.append(
            _resolve_attack_event_spec(
                attack_event=attack_event,
                parent_event_lookup=parent_event_lookup,
                threshold_binding_reference=threshold_binding_reference,
            )
        )

    shard_root = _resolve_attack_shard_root(family_root, attack_shard_index)
    shard_manifest_path = shard_root / PW03_SHARD_MANIFEST_FILE_NAME
    if shard_root.exists():
        if force_rerun:
            shutil.rmtree(shard_root)
        elif shard_manifest_path.exists() and shard_manifest_path.is_file():
            existing_manifest = _load_required_json_dict(shard_manifest_path, "existing PW03 shard manifest")
            if existing_manifest.get("status") == "completed":
                return existing_manifest
            raise RuntimeError(
                f"PW03 shard root already exists with non-completed status; rerun requires force_rerun: {normalize_path_value(shard_root)}"
            )
        else:
            raise RuntimeError(
                f"PW03 shard root already exists without shard_manifest; rerun requires force_rerun: {normalize_path_value(shard_root)}"
            )

    ensure_directory(shard_root)
    ensure_directory(shard_root / "records")
    ensure_directory(shard_root / "artifacts")
    ensure_directory(shard_root / "events")
    ensure_directory(shard_root / "workers")

    worker_plans = _prepare_local_worker_plans(
        drive_project_root=normalized_drive_root,
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_shard_count=attack_shard_count,
        attack_local_worker_count=attack_local_worker_count,
        shard_root=shard_root,
        bound_config_path=resolved_bound_config_path,
        assigned_attack_events=assigned_attack_events,
    )

    worker_execution_mode = "single_process" if attack_local_worker_count == 1 else "shard_local_subprocess_parallel"
    if attack_local_worker_count == 1:
        only_plan = worker_plans[0]
        worker_result = run_pw03_attack_event_worker(
            drive_project_root=normalized_drive_root,
            family_id=family_id,
            attack_shard_index=attack_shard_index,
            attack_shard_count=attack_shard_count,
            attack_local_worker_count=attack_local_worker_count,
            local_worker_index=int(only_plan["local_worker_index"]),
            worker_plan_path=Path(str(only_plan["worker_plan_path"])),
        )
        worker_executions = [
            {
                "local_worker_index": int(only_plan["local_worker_index"]),
                "worker_root": only_plan.get("worker_root"),
                "worker_plan_path": only_plan.get("worker_plan_path"),
                "worker_result_path": only_plan.get("worker_result_path"),
                "stdout_log_path": only_plan.get("stdout_log_path"),
                "stderr_log_path": only_plan.get("stderr_log_path"),
                "assigned_attack_event_ids": only_plan.get("assigned_attack_event_ids", []),
                "command": only_plan.get("command"),
                "return_code": 0,
                "result_exists": True,
            }
        ]
        worker_results = [worker_result]
    else:
        worker_executions, worker_results = _run_local_worker_plans(worker_plans)

    events = _collect_worker_events(worker_results)
    worker_outcomes = _build_worker_outcomes(
        worker_plans=worker_plans,
        worker_executions=worker_executions,
        worker_results=worker_results,
    )

    shard_gpu_inputs: List[Dict[str, Any]] = []
    for worker_result in worker_results:
        worker_gpu_payload = worker_result.get("worker_gpu_session_peak")
        if isinstance(worker_gpu_payload, Mapping):
            payload = dict(cast(Mapping[str, Any], worker_gpu_payload))
            payload["summary_path"] = worker_result.get("worker_gpu_session_peak_path")
            payload["worker_local_index"] = worker_result.get("local_worker_index")
            shard_gpu_inputs.append(payload)
    shard_gpu_summary = _aggregate_gpu_session_peaks(
        family_id=family_id,
        attack_shard_index=attack_shard_index,
        attack_local_worker_count=attack_local_worker_count,
        gpu_peak_payloads=shard_gpu_inputs,
    )
    shard_gpu_summary_path = shard_root / "artifacts" / "gpu_session_peak.json"
    write_json_atomic(shard_gpu_summary_path, shard_gpu_summary)

    completed_event_count = sum(1 for event in events if event.get("status") == "completed")
    failed_event_count = sum(1 for event in events if event.get("status") != "completed")
    worker_execution_failures = [
        worker_execution
        for worker_execution in worker_executions
        if int(worker_execution.get("return_code", 1)) != 0 or not bool(worker_execution.get("result_exists", False))
    ]
    shard_status = "completed"
    failure_reason = None
    if worker_execution_failures:
        shard_status = "failed"
        failure_reason = "pw03_worker_execution_failed"
    elif failed_event_count > 0:
        shard_status = "failed"
        failure_reason = "pw03_attack_event_failed"

    runtime_summary_path = shard_root / PW03_RUNTIME_SUMMARY_FILE_NAME
    runtime_summary = {
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "attack_shard_index": attack_shard_index,
        "attack_shard_count": attack_shard_count,
        "attack_local_worker_count": attack_local_worker_count,
        "worker_execution_mode": worker_execution_mode,
        "status": shard_status,
        "failure_reason": failure_reason,
        "event_count": len(events),
        "assigned_attack_event_count": len(assigned_attack_events),
        "completed_event_count": completed_event_count,
        "failed_event_count": failed_event_count,
        "event_ids": [str(event.get("event_id")) for event in events],
        "worker_outcomes": worker_outcomes,
        "gpu_session_peak_path": normalize_path_value(shard_gpu_summary_path),
        "gpu_session_peak": shard_gpu_summary,
    }
    write_json_atomic(runtime_summary_path, runtime_summary)

    return _write_attack_shard_manifest(
        family_id=family_id,
        shard_root=shard_root,
        attack_shard_index=attack_shard_index,
        attack_shard_count=attack_shard_count,
        attack_local_worker_count=attack_local_worker_count,
        attack_family_allowlist=attack_family_allowlist_values,
        bound_config_path=resolved_bound_config_path,
        worker_execution_mode=worker_execution_mode,
        assigned_attack_events=assigned_attack_events,
        worker_outcomes=worker_outcomes,
        events=events,
        gpu_session_peak_path=shard_gpu_summary_path,
        runtime_summary_path=runtime_summary_path,
        status=shard_status,
        failure_reason=failure_reason,
        traceback_text=None,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for PW03.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one PW03 attacked-event shard from the finalized positive source pool. "
            "The shard supports isolated multi-session execution and shard-local worker parallelism."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Paper workflow family identifier.")
    parser.add_argument("--attack-shard-index", required=True, type=int, help="Zero-based attack shard index.")
    parser.add_argument("--attack-shard-count", required=True, type=int, help="Total attack shard count.")
    parser.add_argument(
        "--attack-local-worker-count",
        default=1,
        type=int,
        help="Shard-local attack worker count. Allowed values are 1, 2, 3, and 4.",
    )
    parser.add_argument(
        "--attack-family-allowlist",
        default=None,
        help="Optional comma-separated or JSON-list attack family allowlist.",
    )
    parser.add_argument(
        "--bound-config-path",
        default=None,
        help="Notebook-bound runtime config snapshot path.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Clear completed shard root before rerun.")
    parser.add_argument("--local-worker-index", default=None, type=int, help="Internal shard-local worker index.")
    parser.add_argument("--worker-plan-path", default=None, help="Internal worker plan path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    功能：执行 PW03 CLI 入口。 

    Execute the PW03 CLI entrypoint.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process-style exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.worker_plan_path is not None or args.local_worker_index is not None:
        if args.worker_plan_path is None or args.local_worker_index is None:
            parser.error("--local-worker-index and --worker-plan-path must be provided together")
        summary = run_pw03_attack_event_worker(
            drive_project_root=Path(args.drive_project_root),
            family_id=str(args.family_id),
            attack_shard_index=int(args.attack_shard_index),
            attack_shard_count=int(args.attack_shard_count),
            attack_local_worker_count=int(args.attack_local_worker_count),
            local_worker_index=int(args.local_worker_index),
            worker_plan_path=Path(str(args.worker_plan_path)),
        )
    else:
        if args.bound_config_path is None:
            parser.error("--bound-config-path is required for the top-level PW03 shard runner")
        summary = run_pw03_attack_event_shard(
            drive_project_root=Path(args.drive_project_root),
            family_id=str(args.family_id),
            attack_shard_index=int(args.attack_shard_index),
            attack_shard_count=int(args.attack_shard_count),
            attack_local_worker_count=int(args.attack_local_worker_count),
            attack_family_allowlist=args.attack_family_allowlist,
            bound_config_path=Path(str(args.bound_config_path)),
            force_rerun=bool(args.force_rerun),
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())