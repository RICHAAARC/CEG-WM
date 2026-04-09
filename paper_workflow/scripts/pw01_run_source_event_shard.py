"""
File purpose: Execute one PW01 source event shard in isolation.
Module type: General module
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from main.core import status as run_status
from main.core.errors import RunFailureReason
from scripts.notebook_runtime_common import (
    REPO_ROOT,
    apply_notebook_model_snapshot_binding,
    build_repo_import_subprocess_env,
    compute_file_sha256,
    copy_file,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    read_json_dict,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)

from paper_workflow.scripts.pw_common import (
    ACTIVE_SAMPLE_ROLE,
    CLEAN_NEGATIVE_SAMPLE_ROLE,
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
    DEFAULT_CONFIG_RELATIVE_PATH,
    build_payload_decode_sidecar_payload,
    build_payload_reference_sidecar_payload,
    build_source_shard_root,
    build_family_root,
    read_jsonl,
    resolve_family_id_from_path,
    resolve_family_layout_paths,
    validate_source_sample_role,
)
from paper_workflow.scripts import pw01_stage_runtime_helpers

WORKER_SCRIPT_PATH = Path("paper_workflow/scripts/pw01_run_source_event_shard_worker.py")
BASE_RUNNER_MODULE = pw01_stage_runtime_helpers
EVENT_RECORD_USAGE_BY_SAMPLE_ROLE = {
    ACTIVE_SAMPLE_ROLE: "paper_workflow_positive_source",
    CLEAN_NEGATIVE_SAMPLE_ROLE: "paper_workflow_clean_negative",
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: "paper_workflow_planner_conditioned_control_negative",
}
DEFAULT_PW01_WORKER_COUNT = 1

_CLEAN_NEGATIVE_FREEZE_ANCHOR_FIELDS = (
    "contract_version",
    "contract_digest",
    "contract_file_sha256",
    "contract_canon_sha256",
    "contract_bound_digest",
    "whitelist_version",
    "whitelist_digest",
    "whitelist_file_sha256",
    "whitelist_canon_sha256",
    "whitelist_bound_digest",
    "policy_path_semantics_version",
    "policy_path_semantics_digest",
    "policy_path_semantics_file_sha256",
    "policy_path_semantics_canon_sha256",
    "policy_path_semantics_bound_digest",
)
_CLEAN_NEGATIVE_OPTIONAL_FORMAL_SCALAR_FIELDS = (
    "cfg_digest",
    "plan_input_digest",
    "plan_input_schema_version",
)
_CLEAN_NEGATIVE_OPTIONAL_FORMAL_MAPPING_FIELDS = (
    "subspace_planner_impl_identity",
    "subspace_plan",
)
_CLEAN_NEGATIVE_REQUIRED_PROBE_PLANNER_FIELDS = (
    "subspace_plan",
    "subspace_planner_impl_identity",
    "plan_input_digest",
    "plan_input_schema_version",
)


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    Read one required JSON object file.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON root is not object.
    """
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def _resolve_manifest_shard_count(family_manifest: Mapping[str, Any]) -> int:
    """
    Resolve source shard count from family manifest.

    Args:
        family_manifest: Family manifest payload.

    Returns:
        Source shard count.
    """
    source_parameters_node = family_manifest.get("source_parameters")
    if not isinstance(source_parameters_node, Mapping):
        raise ValueError("family manifest missing source_parameters")
    source_parameters = cast(Dict[str, Any], source_parameters_node)
    shard_count_value = source_parameters.get("source_shard_count")
    if not isinstance(shard_count_value, int) or shard_count_value <= 0:
        raise ValueError("family manifest source_parameters.source_shard_count must be positive int")
    return int(shard_count_value)


def resolve_source_shard_assignment(
    shard_plan: Mapping[str, Any],
    *,
    sample_role: str,
    shard_index: int,
    shard_count: int,
) -> Dict[str, Any]:
    """
    Resolve one source-role shard assignment.

    Args:
        shard_plan: Source shard plan payload.
        sample_role: Supported source sample role.
        shard_index: Shard index.
        shard_count: Expected shard count.

    Returns:
        Selected shard assignment row.

    Raises:
        ValueError: If shard count or shard index is inconsistent.
    """
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative int")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive int")
    normalized_sample_role = validate_source_sample_role(sample_role)

    plan_shard_count = shard_plan.get("source_shard_count")
    if not isinstance(plan_shard_count, int) or plan_shard_count <= 0:
        raise ValueError("source shard plan missing source_shard_count")
    if int(plan_shard_count) != shard_count:
        raise ValueError(
            f"shard_count mismatch with source shard plan: expected={plan_shard_count}, actual={shard_count}"
        )

    sample_role_plans_node = shard_plan.get("sample_role_plans")
    if not isinstance(sample_role_plans_node, Mapping):
        raise ValueError("source shard plan missing sample_role_plans")
    sample_role_plans = cast(Dict[str, Any], sample_role_plans_node)
    role_plan_node = sample_role_plans.get(normalized_sample_role)
    if not isinstance(role_plan_node, Mapping):
        raise ValueError(f"source shard plan missing {normalized_sample_role} plan")
    role_plan = cast(Dict[str, Any], role_plan_node)

    shards_node = role_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError(f"source shard plan {normalized_sample_role}.shards must be list")

    for shard_row_node in cast(List[object], shards_node):
        shard_row = cast(Dict[str, Any], shard_row_node) if isinstance(shard_row_node, dict) else None
        if shard_row is None:
            continue
        row_index = shard_row.get("shard_index")
        if isinstance(row_index, int) and int(row_index) == shard_index:
            assigned_event_ids = shard_row.get("assigned_event_ids")
            if not isinstance(assigned_event_ids, list):
                raise ValueError("assigned_event_ids must be list")
            assigned_event_indices = shard_row.get("assigned_event_indices")
            if not isinstance(assigned_event_indices, list):
                raise ValueError("assigned_event_indices must be list")
            return shard_row

    raise ValueError(f"shard_index not found in {normalized_sample_role} shard plan: {shard_index}")


def resolve_positive_shard_assignment(
    shard_plan: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
) -> Dict[str, Any]:
    """
    Resolve one positive_source shard assignment.

    Args:
        shard_plan: Source shard plan payload.
        shard_index: Shard index.
        shard_count: Expected shard count.

    Returns:
        Selected shard assignment row.
    """
    return resolve_source_shard_assignment(
        shard_plan,
        sample_role=ACTIVE_SAMPLE_ROLE,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _resolve_default_config_path(family_manifest: Mapping[str, Any]) -> Path:
    """
    Resolve default config path for event runtime derivation.

    Args:
        family_manifest: Family manifest payload.

    Returns:
        Default config path.
    """
    configured_path = family_manifest.get("default_config_path")
    if isinstance(configured_path, str) and configured_path.strip():
        return Path(configured_path).expanduser().resolve()
    return (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).resolve()


def _resolve_required_bound_config_path(bound_config_path: Path | None) -> Path:
    """
    Resolve one required notebook-bound config snapshot path.

    Args:
        bound_config_path: Candidate bound config path.

    Returns:
        Resolved bound config path.

    Raises:
        ValueError: If the bound config path is absent.
        FileNotFoundError: If the bound config path does not exist.
    """
    if bound_config_path is None:
        raise ValueError("PW01 requires bound_config_path produced by notebook precheck")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path or None")

    resolved_path = bound_config_path.expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"PW01 bound_config_path not found: {normalize_path_value(resolved_path)}")
    return resolved_path


def _build_bound_config_context(
    *,
    bound_config_path: Path,
    runtime_cfg_path: Path | None,
    event_id: str | None,
) -> str:
    """
    Build one stable diagnostic context for bound-config failures.

    Args:
        bound_config_path: Bound config source path.
        runtime_cfg_path: Event runtime config path when available.
        event_id: Event identifier when available.

    Returns:
        Joined diagnostic context string.
    """
    details = [f"bound_config_path={normalize_path_value(bound_config_path)}"]
    details.append(
        f"runtime_cfg_path={normalize_path_value(runtime_cfg_path) if runtime_cfg_path is not None else '<absent>'}"
    )
    details.append(f"event_id={event_id if isinstance(event_id, str) and event_id else '<absent>'}")
    return ", ".join(details)


def _validate_bound_runtime_cfg(
    cfg_obj: Mapping[str, Any],
    *,
    bound_config_path: Path,
    runtime_cfg_path: Path | None,
    event_id: str | None,
    stage_label: str,
) -> None:
    """
    Validate that one PW01 runtime config keeps the notebook binding.

    Args:
        cfg_obj: Runtime config mapping.
        bound_config_path: Bound config source path.
        runtime_cfg_path: Event runtime config path when available.
        event_id: Event identifier when available.
        stage_label: Stage label used in the error message.

    Returns:
        None.

    Raises:
        RuntimeError: If the model snapshot binding is incomplete.
    """
    if not isinstance(cfg_obj, Mapping):
        raise TypeError("cfg_obj must be Mapping")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")
    if not isinstance(stage_label, str) or not stage_label.strip():
        raise TypeError("stage_label must be non-empty str")

    diagnostic_context = _build_bound_config_context(
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
    )
    model_snapshot_path_value = cfg_obj.get("model_snapshot_path")
    if not isinstance(model_snapshot_path_value, str) or not model_snapshot_path_value.strip():
        raise RuntimeError(
            f"PW01 bound config missing valid model_snapshot_path before {stage_label}: {diagnostic_context}"
        )

    normalized_snapshot_path = normalize_path_value(model_snapshot_path_value)
    if normalized_snapshot_path == "<absent>":
        raise RuntimeError(
            f"PW01 bound config missing valid model_snapshot_path before {stage_label}: {diagnostic_context}"
        )
    snapshot_path = Path(normalized_snapshot_path)
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        raise RuntimeError(
            f"PW01 bound config missing valid model_snapshot_path before {stage_label}: "
            f"{diagnostic_context}, model_snapshot_path={normalized_snapshot_path}"
        )

    model_source_binding = cfg_obj.get("model_source_binding")
    if not isinstance(model_source_binding, Mapping):
        raise RuntimeError(
            f"PW01 bound config missing valid model_source_binding before {stage_label}: "
            f"{diagnostic_context}, model_snapshot_path={normalized_snapshot_path}"
        )


def _load_required_bound_config(
    bound_config_path: Path | None,
    *,
    stage_label: str,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Load one required notebook-bound config snapshot.

    Args:
        bound_config_path: Candidate bound config path.
        stage_label: Stage label used in validation errors.

    Returns:
        Tuple of resolved bound config path and parsed config mapping.
    """
    resolved_bound_config_path = _resolve_required_bound_config_path(bound_config_path)
    bound_cfg_obj = load_yaml_mapping(resolved_bound_config_path)
    _validate_bound_runtime_cfg(
        bound_cfg_obj,
        bound_config_path=resolved_bound_config_path,
        runtime_cfg_path=None,
        event_id=None,
        stage_label=stage_label,
    )
    return resolved_bound_config_path, bound_cfg_obj


def _load_event_lookup(source_event_grid_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load source event grid and build event lookup by event_id.

    Args:
        source_event_grid_path: Event grid JSONL path.

    Returns:
        Event lookup mapping.
    """
    rows = read_jsonl(source_event_grid_path)
    event_lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("source event grid contains invalid event_id")
        if event_id in event_lookup:
            raise ValueError(f"duplicate event_id in source event grid: {event_id}")
        event_lookup[event_id] = row
    return event_lookup


def _validate_pw01_worker_count(pw01_worker_count: int) -> None:
    """
    Validate the shard-local PW01 worker count.

    Args:
        pw01_worker_count: Requested worker count.

    Returns:
        None.

    Raises:
        ValueError: If the worker count is not 1 or 2.
    """
    if not isinstance(pw01_worker_count, int) or isinstance(pw01_worker_count, bool):
        raise ValueError("pw01_worker_count must be 1 or 2")
    if pw01_worker_count not in {1, 2}:
        raise ValueError("pw01_worker_count must be 1 or 2")


def _validate_local_worker_index(local_worker_index: int, pw01_worker_count: int) -> None:
    """
    Validate the local worker index against the shard-local worker count.

    Args:
        local_worker_index: Local worker index.
        pw01_worker_count: Total shard-local worker count.

    Returns:
        None.

    Raises:
        ValueError: If the worker index is outside the allowed range.
    """
    _validate_pw01_worker_count(pw01_worker_count)
    if not isinstance(local_worker_index, int) or isinstance(local_worker_index, bool):
        raise ValueError("local_worker_index must satisfy 0 <= local_worker_index < pw01_worker_count")
    if local_worker_index < 0 or local_worker_index >= pw01_worker_count:
        raise ValueError("local_worker_index must satisfy 0 <= local_worker_index < pw01_worker_count")


def _resolve_source_prompt_index(event: Mapping[str, Any]) -> int:
    """
    Resolve the source prompt index from one event payload.

    Args:
        event: Event payload.

    Returns:
        Source prompt index.

    Raises:
        ValueError: If the source prompt index is missing or invalid.
    """
    source_prompt_index = event.get("source_prompt_index")
    if source_prompt_index is None:
        source_prompt_index = event.get("prompt_index")
    if not isinstance(source_prompt_index, int) or source_prompt_index < 0:
        raise ValueError("source_prompt_index must be non-negative int")
    return int(source_prompt_index)


def _build_local_worker_assignments(
    *,
    assigned_events: Sequence[Mapping[str, Any]],
    sample_role: str,
    pw01_worker_count: int,
) -> List[Dict[str, Any]]:
    """
    Build stable shard-local worker assignments from ordered assigned events.

    Args:
        assigned_events: Ordered shard-assigned events.
        pw01_worker_count: Requested shard-local worker count.

    Returns:
        Ordered worker-assignment payloads.
    """
    _validate_pw01_worker_count(pw01_worker_count)
    normalized_sample_role = validate_source_sample_role(sample_role)

    assignments: List[Dict[str, Any]] = [
        {
            "local_worker_index": local_worker_index,
            "local_event_ordinals": [],
            "assigned_event_ids": [],
            "assigned_event_indices": [],
            "assigned_events": [],
        }
        for local_worker_index in range(pw01_worker_count)
    ]

    for local_event_ordinal, event in enumerate(assigned_events):
        event_id = event.get("event_id")
        event_index = event.get("event_index")
        sample_role = event.get("sample_role")
        prompt_text = event.get("prompt_text")
        prompt_sha256 = event.get("prompt_sha256")
        seed = event.get("seed")
        prompt_file = event.get("prompt_file")
        source_prompt_index = _resolve_source_prompt_index(event)

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty str")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be non-negative int")
        if sample_role != normalized_sample_role:
            raise ValueError(
                f"assigned event sample_role mismatch: expected={normalized_sample_role}, actual={sample_role}"
            )
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError("prompt_text must be non-empty str")
        if not isinstance(prompt_sha256, str) or not prompt_sha256:
            raise ValueError("prompt_sha256 must be non-empty str")
        if not isinstance(seed, int):
            raise ValueError("seed must be int")
        if not isinstance(prompt_file, str) or not prompt_file:
            raise ValueError("prompt_file must be non-empty str")

        local_worker_index = local_event_ordinal % pw01_worker_count
        assignment = assignments[local_worker_index]
        assignment["local_event_ordinals"].append(local_event_ordinal)
        assignment["assigned_event_ids"].append(event_id)
        assignment["assigned_event_indices"].append(event_index)
        assignment["assigned_events"].append(
            {
                "event_id": event_id,
                "event_index": event_index,
                "sample_role": normalized_sample_role,
                "source_prompt_index": source_prompt_index,
                "prompt_text": prompt_text,
                "prompt_sha256": prompt_sha256,
                "seed": seed,
                "prompt_file": prompt_file,
                "local_event_ordinal": local_event_ordinal,
            }
        )

    return assignments


def _build_role_shard_root(family_root: Path, sample_role: str, shard_index: int) -> Path:
    """
    Build shard root path for one source role and shard index.

    Args:
        family_root: Family root path.
        sample_role: Supported source sample role.
        shard_index: Shard index.

    Returns:
        Shard root path.
    """
    return build_source_shard_root(family_root, sample_role, shard_index)


def _build_shard_root(family_root: Path, shard_index: int) -> Path:
    """
    Build shard root path for one positive shard index.

    Args:
        family_root: Family root path.
        shard_index: Shard index.

    Returns:
        Shard root path.
    """
    return _build_role_shard_root(family_root, ACTIVE_SAMPLE_ROLE, shard_index)


def _build_worker_root(shard_root: Path, local_worker_index: int) -> Path:
    """
    Build the isolated worker root under one shard root.

    Args:
        shard_root: Shard root path.
        local_worker_index: Local worker index.

    Returns:
        Worker root path.
    """
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")
    if local_worker_index < 0:
        raise ValueError("local_worker_index must be non-negative int")

    worker_root = shard_root / "workers" / f"worker_{local_worker_index:02d}"
    validate_path_within_base(shard_root, worker_root, "worker root")
    return worker_root


def _build_worker_plan_path(worker_root: Path) -> Path:
    """
    Build the worker plan path under one worker root.

    Args:
        worker_root: Worker root path.

    Returns:
        Worker plan JSON path.
    """
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    return worker_root / "worker_plan.json"


def _build_worker_result_path(worker_root: Path) -> Path:
    """
    Build the worker result path under one worker root.

    Args:
        worker_root: Worker root path.

    Returns:
        Worker result JSON path.
    """
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    return worker_root / "worker_result.json"


def _build_worker_log_paths(worker_root: Path) -> Tuple[Path, Path]:
    """
    Build stdout and stderr log paths for one worker.

    Args:
        worker_root: Worker root path.

    Returns:
        Tuple of stdout and stderr log paths.
    """
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    return worker_root / "stdout.log", worker_root / "stderr.log"


def _write_worker_plan(
    *,
    worker_root: Path,
    family_id: str,
    sample_role: str,
    shard_index: int,
    shard_count: int,
    pw01_worker_count: int,
    default_config_path: Path,
    bound_config_path: Path,
    shard_root: Path,
    assignment: Mapping[str, Any],
) -> Path:
    """
    Write one shard-local worker plan.

    Args:
        worker_root: Worker root path.
        family_id: Family identifier.
        shard_index: Shard index.
        shard_count: Total shard count.
        pw01_worker_count: Total shard-local worker count.
        default_config_path: Default config path.
        shard_root: Shard root path.
        assignment: Worker assignment payload.

    Returns:
        Worker plan path.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if shard_index < 0:
        raise TypeError("shard_index must be non-negative int")
    if shard_count <= 0:
        raise TypeError("shard_count must be positive int")
    if not isinstance(default_config_path, Path):
        raise TypeError("default_config_path must be Path")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")

    local_worker_index = assignment.get("local_worker_index")
    if not isinstance(local_worker_index, int):
        raise ValueError("worker assignment missing local_worker_index")
    _validate_local_worker_index(local_worker_index, pw01_worker_count)

    plan_path = _build_worker_plan_path(worker_root)
    validate_path_within_base(shard_root, plan_path, "worker plan path")
    write_json_atomic(
        plan_path,
        {
            "artifact_type": "paper_workflow_source_shard_worker_plan",
            "schema_version": "pw01_v1",
            "family_id": family_id,
            "sample_role": normalized_sample_role,
            "shard_index": shard_index,
            "source_shard_count": shard_count,
            "pw01_worker_count": pw01_worker_count,
            "local_worker_index": local_worker_index,
            "default_config_path": normalize_path_value(default_config_path),
            "bound_config_path": normalize_path_value(bound_config_path),
            "shard_root": normalize_path_value(shard_root),
            "worker_root": normalize_path_value(worker_root),
            "local_event_ordinals": list(cast(List[int], assignment.get("local_event_ordinals", []))),
            "assigned_event_ids": list(cast(List[str], assignment.get("assigned_event_ids", []))),
            "assigned_event_indices": list(cast(List[int], assignment.get("assigned_event_indices", []))),
            "assigned_events": list(cast(List[Dict[str, Any]], assignment.get("assigned_events", []))),
        },
    )
    return plan_path


def _build_worker_command(
    *,
    drive_project_root: Path,
    family_id: str,
    shard_index: int,
    pw01_worker_count: int,
    local_worker_index: int,
    worker_plan_path: Path,
) -> List[str]:
    """
    Build the subprocess command for one shard-local worker.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        shard_index: Shard index.
        pw01_worker_count: Total shard-local worker count.
        local_worker_index: Local worker index.
        worker_plan_path: Worker plan path.

    Returns:
        Command token list.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(worker_plan_path, Path):
        raise TypeError("worker_plan_path must be Path")
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if shard_index < 0:
        raise TypeError("shard_index must be non-negative int")
    _validate_local_worker_index(local_worker_index, pw01_worker_count)

    return [
        sys.executable,
        str((REPO_ROOT / WORKER_SCRIPT_PATH).resolve()),
        "--drive-project-root",
        str(drive_project_root),
        "--family-id",
        family_id,
        "--shard-index",
        str(shard_index),
        "--pw01-worker-count",
        str(pw01_worker_count),
        "--local-worker-index",
        str(local_worker_index),
        "--worker-plan-path",
        str(worker_plan_path),
    ]


def _prepare_local_worker_plans(
    *,
    drive_project_root: Path,
    family_id: str,
    sample_role: str,
    shard_index: int,
    shard_count: int,
    pw01_worker_count: int,
    shard_root: Path,
    default_config_path: Path,
    bound_config_path: Path,
    assigned_events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare shard-local worker plans and their fixed output paths.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        shard_index: Shard index.
        shard_count: Total shard count.
        pw01_worker_count: Total shard-local worker count.
        shard_root: Shard root path.
        default_config_path: Default config path.
        assigned_events: Ordered shard-assigned events.

    Returns:
        Ordered worker-plan summaries.
    """
    _validate_pw01_worker_count(pw01_worker_count)

    worker_plans: List[Dict[str, Any]] = []
    for assignment in _build_local_worker_assignments(
        assigned_events=assigned_events,
        sample_role=sample_role,
        pw01_worker_count=pw01_worker_count,
    ):
        local_worker_index = int(assignment["local_worker_index"])
        worker_root = ensure_directory(_build_worker_root(shard_root, local_worker_index))
        worker_plan_path = _write_worker_plan(
            worker_root=worker_root,
            family_id=family_id,
            sample_role=sample_role,
            shard_index=shard_index,
            shard_count=shard_count,
            pw01_worker_count=pw01_worker_count,
            default_config_path=default_config_path,
            bound_config_path=bound_config_path,
            shard_root=shard_root,
            assignment=assignment,
        )
        worker_result_path = _build_worker_result_path(worker_root)
        stdout_log_path, stderr_log_path = _build_worker_log_paths(worker_root)
        worker_plans.append(
            {
                "local_worker_index": local_worker_index,
                "worker_root": normalize_path_value(worker_root),
                "worker_plan_path": normalize_path_value(worker_plan_path),
                "worker_result_path": normalize_path_value(worker_result_path),
                "stdout_log_path": normalize_path_value(stdout_log_path),
                "stderr_log_path": normalize_path_value(stderr_log_path),
                "bound_config_path": normalize_path_value(bound_config_path),
                "assigned_event_ids": list(cast(List[str], assignment["assigned_event_ids"])),
                "assigned_event_indices": list(cast(List[int], assignment["assigned_event_indices"])),
                "local_event_ordinals": list(cast(List[int], assignment["local_event_ordinals"])),
                "command": _build_worker_command(
                    drive_project_root=drive_project_root,
                    family_id=family_id,
                    shard_index=shard_index,
                    pw01_worker_count=pw01_worker_count,
                    local_worker_index=local_worker_index,
                    worker_plan_path=worker_plan_path,
                ),
            }
        )
    return worker_plans


def _run_local_worker_plans(worker_plans: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute shard-local worker plans in subprocesses and collect their results.

    Args:
        worker_plans: Worker-plan summaries.

    Returns:
        Tuple of worker execution summaries and parsed worker results.
    """
    env_mapping = build_repo_import_subprocess_env(repo_root=REPO_ROOT)
    launches: List[Dict[str, Any]] = []
    for worker_plan in worker_plans:
        worker_root = Path(str(worker_plan["worker_root"]))
        stdout_log_path = Path(str(worker_plan["stdout_log_path"]))
        stderr_log_path = Path(str(worker_plan["stderr_log_path"]))
        ensure_directory(worker_root)
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
                "command": list(cast(Sequence[str], worker_plan["command"])),
                "assigned_event_ids": list(cast(Sequence[str], worker_plan["assigned_event_ids"])),
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
        execution_summary = {
            "local_worker_index": int(launch["local_worker_index"]),
            "worker_root": normalize_path_value(cast(Path, launch["worker_root"])),
            "worker_plan_path": normalize_path_value(cast(Path, launch["worker_plan_path"])),
            "worker_result_path": normalize_path_value(worker_result_path),
            "stdout_log_path": normalize_path_value(cast(Path, launch["stdout_log_path"])),
            "stderr_log_path": normalize_path_value(cast(Path, launch["stderr_log_path"])),
            "command": list(cast(List[str], launch["command"])),
            "assigned_event_ids": list(cast(List[str], launch["assigned_event_ids"])),
            "return_code": return_code,
            "result_exists": result_exists,
        }
        worker_executions.append(execution_summary)
        if result_exists:
            worker_results.append(
                _load_required_json_dict(
                    worker_result_path,
                    f"PW01 worker result {execution_summary['local_worker_index']}",
                )
            )

    return worker_executions, worker_results


def _collect_partial_worker_events(worker_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect partial worker-completed events in stable event-index order.

    Args:
        worker_results: Worker result payloads.

    Returns:
        Partial event manifest payloads.
    """
    partial_events: List[Dict[str, Any]] = []
    for worker_result in worker_results:
        events_node = worker_result.get("events")
        if not isinstance(events_node, list):
            continue
        for event_node in cast(List[object], events_node):
            if isinstance(event_node, dict):
                partial_events.append(cast(Dict[str, Any], event_node))
    partial_events.sort(key=lambda item: int(item.get("event_index", -1)))
    return partial_events


def _merge_completed_worker_events(
    *,
    worker_results: Sequence[Mapping[str, Any]],
    assigned_event_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Merge worker-completed events and validate non-overlap plus full coverage.

    Args:
        worker_results: Worker result payloads.
        assigned_event_ids: Ordered shard-assigned event ids.

    Returns:
        Ordered event manifest payloads.

    Raises:
        ValueError: If assignments overlap or coverage is incomplete.
    """
    expected_event_ids = [str(event_id) for event_id in assigned_event_ids]
    expected_event_id_set = set(expected_event_ids)
    assigned_by_worker: set[str] = set()
    completed_by_worker: set[str] = set()
    event_payload_by_id: Dict[str, Dict[str, Any]] = {}

    for worker_result in sorted(worker_results, key=lambda item: int(item.get("local_worker_index", -1))):
        worker_status = worker_result.get("status")
        if worker_status != "completed":
            raise ValueError(f"worker_result status must be completed, got: {worker_status}")

        assigned_ids_node = worker_result.get("assigned_event_ids")
        if not isinstance(assigned_ids_node, list):
            raise ValueError("worker_result assigned_event_ids must be list")
        for event_id in cast(List[object], assigned_ids_node):
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("worker_result assigned_event_ids must contain non-empty str")
            if event_id in assigned_by_worker:
                raise ValueError(f"worker assigned events overlap: {event_id}")
            assigned_by_worker.add(event_id)

        events_node = worker_result.get("events")
        if not isinstance(events_node, list):
            raise ValueError("worker_result events must be list")
        for event_node in cast(List[object], events_node):
            if not isinstance(event_node, dict):
                raise ValueError("worker_result events must contain objects")
            event_payload = cast(Dict[str, Any], event_node)
            event_id = event_payload.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("worker_result event missing event_id")
            if event_id not in assigned_by_worker:
                raise ValueError(f"worker completed unassigned event: {event_id}")
            if event_id in completed_by_worker:
                raise ValueError(f"worker completed events overlap: {event_id}")
            completed_by_worker.add(event_id)
            event_payload_by_id[event_id] = event_payload

    if assigned_by_worker != expected_event_id_set:
        raise ValueError(
            "worker assigned events coverage mismatch: "
            f"expected={sorted(expected_event_id_set)} actual={sorted(assigned_by_worker)}"
        )
    if completed_by_worker != expected_event_id_set:
        raise ValueError(
            "worker completed events coverage mismatch: "
            f"expected={sorted(expected_event_id_set)} actual={sorted(completed_by_worker)}"
        )

    return [event_payload_by_id[event_id] for event_id in expected_event_ids]


def _build_worker_state_rows(
    *,
    worker_plans: Sequence[Mapping[str, Any]],
    worker_executions: Sequence[Mapping[str, Any]],
    worker_results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build manifest-friendly worker state rows.

    Args:
        worker_plans: Worker-plan summaries.
        worker_executions: Worker execution summaries.
        worker_results: Parsed worker result payloads.

    Returns:
        Ordered worker-state rows.
    """
    plan_by_index = {
        int(plan["local_worker_index"]): cast(Dict[str, Any], dict(plan))
        for plan in worker_plans
    }
    execution_by_index = {
        int(execution["local_worker_index"]): cast(Dict[str, Any], dict(execution))
        for execution in worker_executions
    }
    result_by_index = {
        int(result["local_worker_index"]): cast(Dict[str, Any], dict(result))
        for result in worker_results
        if isinstance(result.get("local_worker_index"), int)
    }

    rows: List[Dict[str, Any]] = []
    for local_worker_index in sorted(plan_by_index):
        plan = plan_by_index[local_worker_index]
        execution = execution_by_index.get(local_worker_index, {})
        result = result_by_index.get(local_worker_index, {})
        rows.append(
            {
                "local_worker_index": local_worker_index,
                "worker_root": plan.get("worker_root"),
                "worker_plan_path": plan.get("worker_plan_path"),
                "worker_result_path": plan.get("worker_result_path"),
                "stdout_log_path": plan.get("stdout_log_path"),
                "stderr_log_path": plan.get("stderr_log_path"),
                "bound_config_path": plan.get("bound_config_path"),
                "command": plan.get("command"),
                "assigned_event_ids": plan.get("assigned_event_ids", []),
                "assigned_event_indices": plan.get("assigned_event_indices", []),
                "local_event_ordinals": plan.get("local_event_ordinals", []),
                "return_code": execution.get("return_code"),
                "result_exists": execution.get("result_exists", False),
                "status": result.get("status", "not_started"),
                "completed_event_ids": result.get("completed_event_ids", []),
                "failure_reason": result.get("failure_reason"),
                "exception_type": result.get("exception_type"),
                "exception_message": result.get("exception_message"),
            }
        )
    return rows


def _build_worker_execution_failure_payload(worker_executions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Build a stable failure payload for shard-local worker execution failures.

    Args:
        worker_executions: Worker execution summaries.

    Returns:
        Failure payload mapping.
    """
    failed_workers = [
        dict(worker_execution)
        for worker_execution in worker_executions
        if int(worker_execution.get("return_code", 1)) != 0 or not bool(worker_execution.get("result_exists", False))
    ]
    return {
        "stage_name": "PW01_Source_Event_Shards",
        "status": "failed",
        "failure_reason": "pw01_worker_execution_failed",
        "worker_count": len(worker_executions),
        "failed_worker_count": len(failed_workers),
        "worker_executions": [dict(worker_execution) for worker_execution in worker_executions],
    }


def _build_worker_result_payload(
    *,
    family_id: str,
    sample_role: str,
    shard_index: int,
    shard_count: int,
    pw01_worker_count: int,
    local_worker_index: int,
    worker_root: Path,
    worker_plan_path: Path,
    assigned_event_ids: Sequence[str],
    assigned_event_indices: Sequence[int],
    events: Sequence[Mapping[str, Any]],
    status: str,
    failure_reason: str | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> Dict[str, Any]:
    """
    Build the structured worker result payload.

    Args:
        family_id: Family identifier.
        shard_index: Shard index.
        shard_count: Total shard count.
        pw01_worker_count: Total shard-local worker count.
        local_worker_index: Local worker index.
        worker_root: Worker root path.
        worker_plan_path: Worker plan path.
        assigned_event_ids: Assigned event ids.
        assigned_event_indices: Assigned event indices.
        events: Event manifest payloads completed by this worker.
        status: Worker status.
        failure_reason: Optional failure reason.
        exception_type: Optional exception type.
        exception_message: Optional exception message.

    Returns:
        Worker result payload.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if shard_index < 0:
        raise TypeError("shard_index must be non-negative int")
    if shard_count <= 0:
        raise TypeError("shard_count must be positive int")
    _validate_local_worker_index(local_worker_index, pw01_worker_count)
    if not isinstance(worker_root, Path):
        raise TypeError("worker_root must be Path")
    if not isinstance(worker_plan_path, Path):
        raise TypeError("worker_plan_path must be Path")
    if status not in {"completed", "failed"}:
        raise ValueError("worker status must be 'completed' or 'failed'")

    completed_event_ids: List[str] = []
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("worker event manifest missing event_id")
        completed_event_ids.append(event_id)

    return {
        "artifact_type": "paper_workflow_source_shard_worker_result",
        "schema_version": "pw01_v1",
        "stage_name": "PW01_Source_Event_Shards",
        "family_id": family_id,
        "sample_role": normalized_sample_role,
        "shard_index": shard_index,
        "source_shard_count": shard_count,
        "pw01_worker_count": pw01_worker_count,
        "local_worker_index": local_worker_index,
        "worker_root": normalize_path_value(worker_root),
        "worker_plan_path": normalize_path_value(worker_plan_path),
        "worker_result_path": normalize_path_value(_build_worker_result_path(worker_root)),
        "status": status,
        "assigned_event_ids": [str(event_id) for event_id in assigned_event_ids],
        "assigned_event_indices": [int(event_index) for event_index in assigned_event_indices],
        "assigned_event_count": len(assigned_event_ids),
        "completed_event_ids": completed_event_ids,
        "completed_event_count": len(completed_event_ids),
        "events": [dict(event) for event in events],
        "failure_reason": failure_reason,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def _resolve_event_record_usage(sample_role: str) -> str:
    """
    Resolve detect payload usage token for one source role.

    Args:
        sample_role: Supported source sample role.

    Returns:
        Detect payload record usage token.
    """
    normalized_sample_role = validate_source_sample_role(sample_role)
    return EVENT_RECORD_USAGE_BY_SAMPLE_ROLE[normalized_sample_role]


def _resolve_watermarked_output_image_view(
    *,
    shard_root: Path,
    embed_record_path: Path,
) -> Dict[str, Any]:
    """
    Resolve the explicit watermarked-output image view for one PW01 event.

    Args:
        shard_root: Source shard root.
        embed_record_path: Staged embed record path.

    Returns:
        Optional artifact view for the finalized watermarked output image.
    """
    if not isinstance(shard_root, Path):
        raise TypeError("shard_root must be Path")
    if not isinstance(embed_record_path, Path):
        raise TypeError("embed_record_path must be Path")
    if not embed_record_path.exists() or not embed_record_path.is_file():
        return {
            "exists": False,
            "path": None,
            "package_relative_path": None,
            "missing_reason": "embed_record_not_emitted",
        }

    try:
        embed_record = _load_required_json_dict(embed_record_path, "PW01 embed record")
    except Exception:
        return {
            "exists": False,
            "path": None,
            "package_relative_path": None,
            "missing_reason": "embed_record_invalid",
        }

    watermarked_path_value = embed_record.get("watermarked_path") or embed_record.get("image_path")
    if not isinstance(watermarked_path_value, str) or not watermarked_path_value.strip():
        return {
            "exists": False,
            "path": None,
            "package_relative_path": None,
            "missing_reason": "watermarked_output_image_missing",
        }

    watermarked_path = Path(watermarked_path_value).expanduser().resolve()
    if not watermarked_path.exists() or not watermarked_path.is_file():
        return {
            "exists": False,
            "path": normalize_path_value(watermarked_path),
            "package_relative_path": None,
            "missing_reason": "watermarked_output_image_missing",
        }

    try:
        package_relative_path = watermarked_path.relative_to(shard_root).as_posix()
    except ValueError:
        package_relative_path = None

    return {
        "exists": True,
        "path": normalize_path_value(watermarked_path),
        "package_relative_path": package_relative_path,
        "missing_reason": None,
    }


def _extract_detect_probe_plan_anchors(detect_payload_obj: Mapping[str, Any]) -> Dict[str, str]:
    """
    Extract plan and basis digests from one detect probe record.

    Args:
        detect_payload_obj: Detect record payload.

    Returns:
        Mapping with plan_digest and basis_digest.

    Raises:
        ValueError: If required plan anchors are absent.
    """
    if not isinstance(detect_payload_obj, Mapping):
        raise TypeError("detect_payload_obj must be Mapping")

    content_evidence_node = detect_payload_obj.get("content_evidence_payload")
    content_evidence_payload = (
        cast(Dict[str, Any], content_evidence_node)
        if isinstance(content_evidence_node, dict)
        else {}
    )

    plan_digest = detect_payload_obj.get("plan_digest")
    if not isinstance(plan_digest, str) or not plan_digest:
        plan_digest = content_evidence_payload.get("plan_digest")
    basis_digest = detect_payload_obj.get("basis_digest")
    if not isinstance(basis_digest, str) or not basis_digest:
        basis_digest = content_evidence_payload.get("basis_digest")

    if not isinstance(plan_digest, str) or not plan_digest:
        raise ValueError("detect probe record missing plan_digest")
    if not isinstance(basis_digest, str) or not basis_digest:
        raise ValueError("detect probe record missing basis_digest")

    return {
        "plan_digest": plan_digest,
        "basis_digest": basis_digest,
    }


def _validate_clean_negative_probe_planner_payload(detect_payload_obj: Mapping[str, Any]) -> None:
    """
    Validate that one detect probe record exposes the required planner payload.

    Args:
        detect_payload_obj: Detect probe record payload.

    Returns:
        None.

    Raises:
        RuntimeError: If required planner payload fields are absent.
    """
    if not isinstance(detect_payload_obj, Mapping):
        raise TypeError("detect_payload_obj must be Mapping")

    missing_fields: List[str] = []
    for field_name in _CLEAN_NEGATIVE_REQUIRED_PROBE_PLANNER_FIELDS:
        field_value = detect_payload_obj.get(field_name)
        if field_name in _CLEAN_NEGATIVE_OPTIONAL_FORMAL_MAPPING_FIELDS:
            if not isinstance(field_value, dict) or not field_value:
                missing_fields.append(field_name)
            continue
        if not isinstance(field_value, str) or not field_value:
            missing_fields.append(field_name)

    if missing_fields:
        raise RuntimeError(
            "clean_negative_probe_missing_planner_payload: "
            f"missing_fields={','.join(sorted(missing_fields))}"
        )


def _build_clean_negative_run_meta_from_run_closure(run_closure_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Rebuild the minimal run_meta required to refresh one event run closure.

    Args:
        run_closure_payload: Existing run_closure payload.

    Returns:
        Mutable run_meta mapping.
    """
    if not isinstance(run_closure_payload, Mapping):
        raise TypeError("run_closure_payload must be Mapping")

    status_node = run_closure_payload.get("status")
    status_payload = cast(Dict[str, Any], status_node) if isinstance(status_node, dict) else {}
    status_ok = status_payload.get("ok")
    if not isinstance(status_ok, bool):
        raise ValueError("clean_negative run_closure missing boolean status.ok")

    status_reason_value = status_payload.get("reason")
    if not isinstance(status_reason_value, str) or not status_reason_value:
        raise ValueError("clean_negative run_closure missing status.reason")
    try:
        status_reason = RunFailureReason(status_reason_value)
    except ValueError as exc:
        raise ValueError(
            f"clean_negative run_closure has unsupported status.reason: {status_reason_value}"
        ) from exc

    run_meta = dict(cast(Dict[str, Any], run_closure_payload))
    run_meta.pop("status", None)
    run_meta["status_ok"] = status_ok
    run_meta["status_reason"] = status_reason
    run_meta["status_details"] = status_payload.get("details")
    return run_meta


def _refresh_clean_negative_event_run_records_view(prompt_run_root: Path) -> None:
    """
    Refresh the event-level records bundle and run_closure after embed write-back.

    Args:
        prompt_run_root: Event run root.

    Returns:
        None.
    """
    if not isinstance(prompt_run_root, Path):
        raise TypeError("prompt_run_root must be Path")
    if not prompt_run_root.exists() or not prompt_run_root.is_dir():
        raise FileNotFoundError(
            f"clean_negative event run root missing: {normalize_path_value(prompt_run_root)}"
        )

    records_dir = ensure_directory(prompt_run_root / "records")
    artifacts_dir = ensure_directory(prompt_run_root / "artifacts")
    run_closure_path = artifacts_dir / "run_closure.json"
    if not run_closure_path.exists() or not run_closure_path.is_file():
        raise FileNotFoundError(
            f"clean_negative run_closure missing before records refresh: {normalize_path_value(run_closure_path)}"
        )

    run_closure_payload = _load_required_json_dict(run_closure_path, "clean_negative run_closure")
    run_meta = _build_clean_negative_run_meta_from_run_closure(run_closure_payload)
    run_status.finalize_run(prompt_run_root, records_dir, artifacts_dir, run_meta)


def _build_clean_negative_detect_command(
    *,
    config_path: Path,
    run_root: Path,
    input_record_path: Path,
) -> List[str]:
    """
    Build one detect command with an explicit clean-negative input record.

    Args:
        config_path: Runtime config path.
        run_root: Prompt-local run root.
        input_record_path: Explicit detect input record path.

    Returns:
        Detect command token list.
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(input_record_path, Path):
        raise TypeError("input_record_path must be Path")

    command = list(pw01_stage_runtime_helpers._build_stage_command("detect", config_path, run_root))
    input_token = str(input_record_path)
    if "--input" in command:
        input_index = command.index("--input")
        if input_index + 1 < len(command):
            command[input_index + 1] = input_token
        else:
            command.append(input_token)
    else:
        command.extend(["--input", input_token])
    return [str(item) for item in command]


def _build_clean_negative_input_record(
    *,
    preview_image_path: Path,
    sample_role: str,
    event_id: str,
    prompt_text: str,
    prompt_sha256: str,
    seed: int,
    provenance_payload: Mapping[str, Any] | None = None,
    probe_plan_anchors: Mapping[str, Any] | None = None,
    probe_detect_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the clean-negative detect input record consumed by detect.

    Args:
        preview_image_path: Clean preview artifact path.
        sample_role: Negative-branch sample role.
        event_id: Event identifier.
        prompt_text: Prompt text.
        prompt_sha256: Prompt SHA256.
        seed: Event seed.
        provenance_payload: Optional statement-only provenance payload.
        probe_plan_anchors: Optional probe-derived plan anchor mapping.
        probe_detect_payload: Optional detect probe payload carrying planner fields.

    Returns:
        Clean-negative detect input record.
    """
    if not isinstance(preview_image_path, Path):
        raise TypeError("preview_image_path must be Path")
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise FileNotFoundError(f"clean negative preview image not found: {normalize_path_value(preview_image_path)}")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise TypeError("prompt_sha256 must be non-empty str")
    if not isinstance(seed, int):
        raise TypeError("seed must be int")

    normalized_preview_path = normalize_path_value(preview_image_path)
    artifact_sha256 = compute_file_sha256(preview_image_path)
    record_payload: Dict[str, Any] = {
        "operation": "embed_preview_input",
        "watermarked_path": normalized_preview_path,
        "image_path": normalized_preview_path,
        "artifact_sha256": artifact_sha256,
        "watermarked_artifact_sha256": artifact_sha256,
        "is_watermarked": False,
        "negative_branch_note": (
            "paper workflow "
            f"{normalized_sample_role} source uses preview artifact without watermark embed"
        ),
        "event_id": event_id,
        "prompt": prompt_text,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "inputs": {
            "input_image_path": normalized_preview_path,
        },
    }
    if isinstance(provenance_payload, Mapping):
        record_payload["negative_branch_source_attestation_provenance"] = dict(provenance_payload)
    _copy_clean_negative_probe_formal_fields(
        record_payload,
        probe_plan_anchors=probe_plan_anchors,
        probe_detect_payload=probe_detect_payload,
    )
    return record_payload


def _copy_clean_negative_probe_formal_fields(
    target_record: Dict[str, Any],
    *,
    probe_plan_anchors: Mapping[str, Any] | None = None,
    probe_detect_payload: Mapping[str, Any] | None = None,
) -> None:
    """
    Copy probe-derived planner anchors and formal planner fields into one record.

    Args:
        target_record: Mutable target record.
        probe_plan_anchors: Optional probe-derived plan anchor mapping.
        probe_detect_payload: Optional probe detect payload.

    Returns:
        None.
    """
    if not isinstance(target_record, dict):
        raise TypeError("target_record must be dict")
    if probe_plan_anchors is not None and not isinstance(probe_plan_anchors, Mapping):
        raise TypeError("probe_plan_anchors must be Mapping or None")
    if probe_detect_payload is not None and not isinstance(probe_detect_payload, Mapping):
        raise TypeError("probe_detect_payload must be Mapping or None")

    if isinstance(probe_plan_anchors, Mapping):
        plan_digest = probe_plan_anchors.get("plan_digest")
        basis_digest = probe_plan_anchors.get("basis_digest")
        if not isinstance(plan_digest, str) or not plan_digest:
            raise ValueError("probe_plan_anchors missing plan_digest")
        if not isinstance(basis_digest, str) or not basis_digest:
            raise ValueError("probe_plan_anchors missing basis_digest")
        target_record["plan_digest"] = plan_digest
        target_record["basis_digest"] = basis_digest

    if not isinstance(probe_detect_payload, Mapping):
        return

    content_evidence_node = probe_detect_payload.get("content_evidence_payload")
    content_evidence_payload = (
        cast(Dict[str, Any], content_evidence_node)
        if isinstance(content_evidence_node, dict)
        else {}
    )
    for field_name in _CLEAN_NEGATIVE_OPTIONAL_FORMAL_SCALAR_FIELDS:
        field_value = probe_detect_payload.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            field_value = content_evidence_payload.get(field_name)
        if isinstance(field_value, str) and field_value:
            target_record[field_name] = field_value
    for field_name in _CLEAN_NEGATIVE_OPTIONAL_FORMAL_MAPPING_FIELDS:
        field_value = probe_detect_payload.get(field_name)
        if not isinstance(field_value, dict) or not field_value:
            field_value = content_evidence_payload.get(field_name)
        if isinstance(field_value, dict) and field_value:
            target_record[field_name] = copy.deepcopy(cast(Dict[str, Any], field_value))


def _copy_clean_negative_freeze_anchor_fields(
    target_record: Dict[str, Any],
    source_record: Mapping[str, Any],
) -> None:
    """
    Copy the actual freeze-anchor fields from one bound detect record.

    Args:
        target_record: Mutable staged embed record.
        source_record: Bound detect record payload.

    Returns:
        None.
    """
    if not isinstance(target_record, dict):
        raise TypeError("target_record must be dict")
    if not isinstance(source_record, Mapping):
        raise TypeError("source_record must be Mapping")

    for field_name in _CLEAN_NEGATIVE_FREEZE_ANCHOR_FIELDS:
        field_value = source_record.get(field_name)
        if isinstance(field_value, str) and field_value:
            target_record[field_name] = field_value


def _build_clean_negative_staged_embed_record(
    *,
    preview_image_path: Path,
    sample_role: str,
    event_id: str,
    prompt_text: str,
    prompt_sha256: str,
    seed: int,
    probe_plan_anchors: Mapping[str, Any] | None,
    probe_detect_payload: Mapping[str, Any] | None,
    bound_detect_payload: Mapping[str, Any],
    provenance_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the staged clean-negative embed record without reusing the detect input file.

    Args:
        preview_image_path: Clean preview artifact path.
        sample_role: Negative-branch sample role.
        event_id: Event identifier.
        prompt_text: Prompt text.
        prompt_sha256: Prompt SHA256.
        seed: Event seed.
        probe_plan_anchors: Optional probe-derived plan anchor mapping.
        probe_detect_payload: Optional probe detect record payload.
        bound_detect_payload: Final detect record payload carrying freeze anchors.
        provenance_payload: Optional statement-only provenance payload.

    Returns:
        Staged clean-negative embed record payload.
    """
    if not isinstance(preview_image_path, Path):
        raise TypeError("preview_image_path must be Path")
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise FileNotFoundError(f"clean negative preview image not found: {normalize_path_value(preview_image_path)}")
    normalized_sample_role = validate_source_sample_role(sample_role)
    if not isinstance(event_id, str) or not event_id:
        raise TypeError("event_id must be non-empty str")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise TypeError("prompt_sha256 must be non-empty str")
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    if probe_plan_anchors is not None and not isinstance(probe_plan_anchors, Mapping):
        raise TypeError("probe_plan_anchors must be Mapping or None")
    if probe_detect_payload is not None and not isinstance(probe_detect_payload, Mapping):
        raise TypeError("probe_detect_payload must be Mapping or None")
    if not isinstance(bound_detect_payload, Mapping):
        raise TypeError("bound_detect_payload must be Mapping")

    normalized_preview_path = normalize_path_value(preview_image_path)
    artifact_sha256 = compute_file_sha256(preview_image_path)
    staged_record: Dict[str, Any] = {
        "operation": "embed",
        "embed_mode": "explicit_negative_no_watermark",
        "watermarked_path": normalized_preview_path,
        "image_path": normalized_preview_path,
        "artifact_sha256": artifact_sha256,
        "watermarked_artifact_sha256": artifact_sha256,
        "is_watermarked": False,
        "negative_branch_note": (
            "paper workflow "
            f"{normalized_sample_role} source uses preview artifact without watermark embed"
        ),
        "event_id": event_id,
        "prompt": prompt_text,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
    }
    if isinstance(provenance_payload, Mapping):
        staged_record["negative_branch_source_attestation_provenance"] = dict(provenance_payload)

    if isinstance(probe_plan_anchors, Mapping) or isinstance(probe_detect_payload, Mapping):
        _copy_clean_negative_probe_formal_fields(
            staged_record,
            probe_plan_anchors=probe_plan_anchors,
            probe_detect_payload=probe_detect_payload,
        )

    _copy_clean_negative_freeze_anchor_fields(staged_record, bound_detect_payload)
    return staged_record


def _build_negative_branch_attestation_provenance(
    *,
    runtime_cfg: Mapping[str, Any],
    prompt_text: str,
    seed: int,
    plan_digest: str,
) -> Dict[str, Any]:
    """
    Build statement-only negative provenance from runtime config and probe anchors.

    Args:
        runtime_cfg: Bound runtime config mapping.
        prompt_text: Prompt text.
        seed: Event seed.
        plan_digest: Detect-probe plan digest.

    Returns:
        Statement-only provenance payload.

    Raises:
        RuntimeError: If attestation commitment keys are unavailable.
    """
    if not isinstance(runtime_cfg, Mapping):
        raise TypeError("runtime_cfg must be Mapping")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    if not isinstance(plan_digest, str) or not plan_digest:
        raise TypeError("plan_digest must be non-empty str")

    from main.watermarking.provenance.commitments import (
        compute_prompt_commit,
        compute_seed_commit,
    )
    from main.watermarking.provenance.attestation_statement import (
        build_attestation_statement,
        compute_attestation_digest,
    )

    model_id = runtime_cfg.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("clean_negative runtime_cfg missing model_id")

    attestation_node = runtime_cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    k_prompt_env_var = attestation_cfg.get("k_prompt_env_var")
    k_seed_env_var = attestation_cfg.get("k_seed_env_var")
    if not isinstance(k_prompt_env_var, str) or not k_prompt_env_var:
        raise RuntimeError("clean_negative attestation.k_prompt_env_var missing")
    if not isinstance(k_seed_env_var, str) or not k_seed_env_var:
        raise RuntimeError("clean_negative attestation.k_seed_env_var missing")

    k_prompt = os.environ.get(k_prompt_env_var)
    k_seed = os.environ.get(k_seed_env_var)
    if not isinstance(k_prompt, str) or not k_prompt:
        raise RuntimeError(f"clean_negative missing attestation prompt key env var: {k_prompt_env_var}")
    if not isinstance(k_seed, str) or not k_seed:
        raise RuntimeError(f"clean_negative missing attestation seed key env var: {k_seed_env_var}")

    statement = build_attestation_statement(
        model_id=model_id,
        prompt_commit=compute_prompt_commit(k_prompt, prompt_text),
        seed_commit=compute_seed_commit(k_seed, seed),
        plan_digest=plan_digest,
    )
    return {
        "statement": statement.as_dict(),
        "attestation_digest": compute_attestation_digest(statement),
    }


def _run_clean_negative_event(
    *,
    event: Mapping[str, Any],
    shard_root: Path,
    default_cfg_obj: Mapping[str, Any],
    bound_config_path: Path,
) -> Dict[str, Any]:
    """
    Execute one strict clean_negative event without planner-conditioned override.

    Args:
        event: Event payload from source_event_grid.
        shard_root: Shard root path.
        default_cfg_obj: Parsed default config mapping.
        bound_config_path: Bound config source path.

    Returns:
        Event manifest payload.
    """
    event_id = event.get("event_id")
    sample_role = event.get("sample_role")
    event_index = event.get("event_index")
    source_prompt_index = event.get("source_prompt_index")
    if source_prompt_index is None:
        source_prompt_index = event.get("prompt_index")
    prompt_text = event.get("prompt_text")
    prompt_sha256 = event.get("prompt_sha256")
    seed = event.get("seed")
    prompt_file = event.get("prompt_file")

    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be non-empty str")
    if sample_role != CLEAN_NEGATIVE_SAMPLE_ROLE:
        raise ValueError(f"PW01 clean negative executor requires clean_negative, got: {sample_role}")
    if not isinstance(event_index, int) or event_index < 0:
        raise ValueError("event_index must be non-negative int")
    if not isinstance(source_prompt_index, int) or source_prompt_index < 0:
        raise ValueError("prompt_index must be non-negative int")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be non-empty str")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise ValueError("prompt_sha256 must be non-empty str")
    if not isinstance(seed, int):
        raise ValueError("seed must be int")
    if not isinstance(prompt_file, str) or not prompt_file:
        raise ValueError("prompt_file must be non-empty str")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")

    event_root = ensure_directory(shard_root / "events" / f"event_{event_index:06d}")
    prompt_run_root = ensure_directory(event_root / "run")
    validate_path_within_base(shard_root, event_root, "event root")
    validate_path_within_base(shard_root, prompt_run_root, "event run root")

    runtime_cfg = copy.deepcopy(dict(default_cfg_obj))
    runtime_cfg["inference_prompt"] = prompt_text
    runtime_cfg["seed"] = seed
    runtime_cfg["paper_workflow_event"] = {
        "event_id": event_id,
        "event_index": event_index,
        "source_prompt_index": source_prompt_index,
        "sample_role": CLEAN_NEGATIVE_SAMPLE_ROLE,
    }

    runtime_cfg_path = event_root / "runtime_config.yaml"
    validate_path_within_base(shard_root, runtime_cfg_path, "event runtime config")
    runtime_cfg = apply_notebook_model_snapshot_binding(runtime_cfg, env_mapping=os.environ)
    _validate_bound_runtime_cfg(
        runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="preview_precompute",
    )
    write_yaml_mapping(runtime_cfg_path, runtime_cfg)

    preview_precompute = pw01_stage_runtime_helpers._prepare_source_pool_preview_artifact(
        cfg_obj=runtime_cfg,
        prompt_run_root=prompt_run_root,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
    )
    preview_runtime_cfg_node = preview_precompute.get("runtime_cfg")
    if not isinstance(preview_runtime_cfg_node, dict):
        raise ValueError("preview precompute result missing runtime_cfg")
    preview_runtime_cfg = cast(Dict[str, Any], preview_runtime_cfg_node)
    _validate_bound_runtime_cfg(
        preview_runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="clean_negative_detect",
    )
    write_yaml_mapping(runtime_cfg_path, preview_runtime_cfg)

    preview_record_node = preview_precompute.get("preview_record")
    preview_record = cast(Dict[str, Any], preview_record_node) if isinstance(preview_record_node, dict) else {}
    preview_image_path_value = preview_record.get("persisted_artifact_path")
    if not isinstance(preview_image_path_value, str) or not preview_image_path_value:
        raise ValueError("clean_negative preview record missing persisted_artifact_path")
    preview_image_path = Path(preview_image_path_value).expanduser().resolve()
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise FileNotFoundError(f"clean_negative preview image missing: {normalize_path_value(preview_image_path)}")

    detect_input_record_path = prompt_run_root / "artifacts" / "neg_preview_input" / "detect_input_record.json"
    ensure_directory(detect_input_record_path.parent)
    detect_input_record = _build_clean_negative_input_record(
        preview_image_path=preview_image_path,
        sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
        event_id=event_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        seed=seed,
    )
    write_json_atomic(detect_input_record_path, detect_input_record)

    stage_results: Dict[str, Any] = {
        "preview_precompute": preview_record,
    }

    detect_command = _build_clean_negative_detect_command(
        config_path=runtime_cfg_path,
        run_root=prompt_run_root,
        input_record_path=detect_input_record_path,
    )
    detect_result = pw01_stage_runtime_helpers._run_stage("detect", detect_command, prompt_run_root)
    stage_results["detect"] = detect_result
    if int(detect_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW01 clean_negative detect failed: event_id={event_id}, "
            f"payload={json.dumps(detect_result, ensure_ascii=False, sort_keys=True)}"
        )

    source_detect_record_path = prompt_run_root / "records" / "detect_record.json"
    if not source_detect_record_path.exists() or not source_detect_record_path.is_file():
        raise FileNotFoundError(f"clean_negative detect record missing: {normalize_path_value(source_detect_record_path)}")

    staged_embed_record_path = shard_root / "records" / f"event_{event_index:06d}_embed_record.json"
    staged_detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_record.json"
    validate_path_within_base(shard_root, staged_embed_record_path, "staged embed record")
    validate_path_within_base(shard_root, staged_detect_record_path, "staged detect record")

    detect_payload_obj = _load_required_json_dict(source_detect_record_path, "clean_negative detect record")
    staged_embed_record_payload = _build_clean_negative_staged_embed_record(
        preview_image_path=preview_image_path,
        sample_role=CLEAN_NEGATIVE_SAMPLE_ROLE,
        event_id=event_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        seed=seed,
        probe_plan_anchors=None,
        probe_detect_payload=None,
        bound_detect_payload=detect_payload_obj,
        provenance_payload=None,
    )
    write_json_atomic(staged_embed_record_path, staged_embed_record_payload)

    event_embed_record_path = prompt_run_root / "records" / "embed_record.json"
    validate_path_within_base(shard_root, event_embed_record_path, "event embed record")
    write_json_atomic(event_embed_record_path, staged_embed_record_payload)
    _refresh_clean_negative_event_run_records_view(prompt_run_root)

    detect_payload_obj = pw01_stage_runtime_helpers._normalize_direct_detect_payload(
        detect_payload_obj,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
        record_usage=_resolve_event_record_usage(CLEAN_NEGATIVE_SAMPLE_ROLE),
    )
    write_json_atomic(staged_detect_record_path, detect_payload_obj)

    source_image_view = pw01_stage_runtime_helpers._resolve_source_pool_source_image_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    preview_generation_record_view = pw01_stage_runtime_helpers._resolve_source_pool_preview_generation_record_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    plain_preview_image_view = dict(source_image_view)
    watermarked_output_image_view = _resolve_watermarked_output_image_view(
        shard_root=shard_root,
        embed_record_path=staged_embed_record_path,
    )

    shard_relative_runtime_cfg = runtime_cfg_path.relative_to(shard_root).as_posix()
    shard_relative_embed_record = staged_embed_record_path.relative_to(shard_root).as_posix()
    shard_relative_detect_record = staged_detect_record_path.relative_to(shard_root).as_posix()

    event_manifest_payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_source_event",
        "event_id": event_id,
        "sample_role": CLEAN_NEGATIVE_SAMPLE_ROLE,
        "source_prompt_index": source_prompt_index,
        "event_index": event_index,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "runtime_config_path": normalize_path_value(runtime_cfg_path),
        "runtime_config_package_relative_path": shard_relative_runtime_cfg,
        "embed_record_path": normalize_path_value(staged_embed_record_path),
        "embed_record_package_relative_path": shard_relative_embed_record,
        "detect_record_path": normalize_path_value(staged_detect_record_path),
        "detect_record_package_relative_path": shard_relative_detect_record,
        "source_image": source_image_view,
        "preview_generation_record": preview_generation_record_view,
        "attestation_statement": None,
        "attestation_bundle": None,
        "attestation_result": None,
        "negative_branch_source_attestation_provenance": None,
        "detect_probe_record_path": None,
        "sha256": compute_file_sha256(staged_detect_record_path),
        "stage_results": stage_results,
    }

    event_manifest_path = event_root / "event_manifest.json"
    validate_path_within_base(shard_root, event_manifest_path, "event manifest path")
    write_json_atomic(event_manifest_path, event_manifest_payload)
    event_manifest_payload["event_manifest_path"] = normalize_path_value(event_manifest_path)
    return event_manifest_payload


def _run_planner_conditioned_control_negative_event(
    *,
    event: Mapping[str, Any],
    shard_root: Path,
    default_cfg_obj: Mapping[str, Any],
    bound_config_path: Path,
) -> Dict[str, Any]:
    """
    Execute one planner-conditioned control negative event with statement-only provenance.

    Args:
        event: Event payload from source_event_grid.
        shard_root: Shard root path.
        default_cfg_obj: Parsed default config mapping.
        bound_config_path: Bound config source path.

    Returns:
        Event manifest payload.
    """
    event_id = event.get("event_id")
    sample_role = event.get("sample_role")
    event_index = event.get("event_index")
    source_prompt_index = event.get("source_prompt_index")
    if source_prompt_index is None:
        source_prompt_index = event.get("prompt_index")
    prompt_text = event.get("prompt_text")
    prompt_sha256 = event.get("prompt_sha256")
    seed = event.get("seed")
    prompt_file = event.get("prompt_file")

    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be non-empty str")
    if sample_role != PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE:
        raise ValueError(
            "PW01 control negative executor requires planner_conditioned_control_negative, "
            f"got: {sample_role}"
        )
    if not isinstance(event_index, int) or event_index < 0:
        raise ValueError("event_index must be non-negative int")
    if not isinstance(source_prompt_index, int) or source_prompt_index < 0:
        raise ValueError("prompt_index must be non-negative int")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be non-empty str")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise ValueError("prompt_sha256 must be non-empty str")
    if not isinstance(seed, int):
        raise ValueError("seed must be int")
    if not isinstance(prompt_file, str) or not prompt_file:
        raise ValueError("prompt_file must be non-empty str")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")

    event_root = ensure_directory(shard_root / "events" / f"event_{event_index:06d}")
    prompt_run_root = ensure_directory(event_root / "run")
    validate_path_within_base(shard_root, event_root, "event root")
    validate_path_within_base(shard_root, prompt_run_root, "event run root")

    runtime_cfg = copy.deepcopy(dict(default_cfg_obj))
    runtime_cfg["inference_prompt"] = prompt_text
    runtime_cfg["seed"] = seed
    runtime_cfg["paper_workflow_event"] = {
        "event_id": event_id,
        "event_index": event_index,
        "source_prompt_index": source_prompt_index,
        "sample_role": PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
    }

    runtime_cfg_path = event_root / "runtime_config.yaml"
    validate_path_within_base(shard_root, runtime_cfg_path, "event runtime config")
    runtime_cfg = apply_notebook_model_snapshot_binding(runtime_cfg, env_mapping=os.environ)
    _validate_bound_runtime_cfg(
        runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="preview_precompute",
    )
    write_yaml_mapping(runtime_cfg_path, runtime_cfg)

    preview_precompute = pw01_stage_runtime_helpers._prepare_source_pool_preview_artifact(
        cfg_obj=runtime_cfg,
        prompt_run_root=prompt_run_root,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
    )
    preview_runtime_cfg_node = preview_precompute.get("runtime_cfg")
    if not isinstance(preview_runtime_cfg_node, dict):
        raise ValueError("preview precompute result missing runtime_cfg")
    preview_runtime_cfg = cast(Dict[str, Any], preview_runtime_cfg_node)
    _validate_bound_runtime_cfg(
        preview_runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="control_negative_detect",
    )
    write_yaml_mapping(runtime_cfg_path, preview_runtime_cfg)

    preview_record_node = preview_precompute.get("preview_record")
    preview_record = cast(Dict[str, Any], preview_record_node) if isinstance(preview_record_node, dict) else {}
    preview_image_path_value = preview_record.get("persisted_artifact_path")
    if not isinstance(preview_image_path_value, str) or not preview_image_path_value:
        raise ValueError("clean_negative preview record missing persisted_artifact_path")
    preview_image_path = Path(preview_image_path_value).expanduser().resolve()
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise FileNotFoundError(f"clean_negative preview image missing: {normalize_path_value(preview_image_path)}")

    detect_input_record_path = prompt_run_root / "artifacts" / "neg_preview_input" / "detect_input_record.json"
    ensure_directory(detect_input_record_path.parent)
    probe_input_record = _build_clean_negative_input_record(
        preview_image_path=preview_image_path,
        sample_role=PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
        event_id=event_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        seed=seed,
    )
    write_json_atomic(detect_input_record_path, probe_input_record)

    stage_results: Dict[str, Any] = {
        "preview_precompute": preview_record,
    }

    detect_probe_command = _build_clean_negative_detect_command(
        config_path=runtime_cfg_path,
        run_root=prompt_run_root,
        input_record_path=detect_input_record_path,
    )
    detect_probe_result = pw01_stage_runtime_helpers._run_stage("detect_probe", detect_probe_command, prompt_run_root)
    stage_results["detect_probe"] = detect_probe_result
    if int(detect_probe_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW01 clean_negative detect probe failed: event_id={event_id}, "
            f"payload={json.dumps(detect_probe_result, ensure_ascii=False, sort_keys=True)}"
        )

    probe_detect_record_path = prompt_run_root / "records" / "detect_record.json"
    if not probe_detect_record_path.exists() or not probe_detect_record_path.is_file():
        raise FileNotFoundError(f"clean_negative detect probe record missing: {normalize_path_value(probe_detect_record_path)}")
    probe_detect_payload = _load_required_json_dict(probe_detect_record_path, "clean_negative detect probe record")
    probe_plan_anchors = _extract_detect_probe_plan_anchors(probe_detect_payload)
    stage_results["detect_probe_plan_anchors"] = dict(probe_plan_anchors)

    staged_probe_detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_probe_record.json"
    validate_path_within_base(shard_root, staged_probe_detect_record_path, "staged detect probe record")
    write_json_atomic(staged_probe_detect_record_path, probe_detect_payload)
    _validate_clean_negative_probe_planner_payload(probe_detect_payload)

    provenance_payload = _build_negative_branch_attestation_provenance(
        runtime_cfg=preview_runtime_cfg,
        prompt_text=prompt_text,
        seed=seed,
        plan_digest=probe_plan_anchors["plan_digest"],
    )
    final_input_record = _build_clean_negative_input_record(
        preview_image_path=preview_image_path,
        sample_role=PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
        event_id=event_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        seed=seed,
        provenance_payload=provenance_payload,
        probe_plan_anchors=probe_plan_anchors,
        probe_detect_payload=probe_detect_payload,
    )
    write_json_atomic(detect_input_record_path, final_input_record)

    detect_command = _build_clean_negative_detect_command(
        config_path=runtime_cfg_path,
        run_root=prompt_run_root,
        input_record_path=detect_input_record_path,
    )
    detect_result = pw01_stage_runtime_helpers._run_stage("detect", detect_command, prompt_run_root)
    stage_results["detect"] = detect_result
    if int(detect_result.get("return_code", 1)) != 0:
        raise RuntimeError(
            f"PW01 clean_negative detect failed: event_id={event_id}, "
            f"payload={json.dumps(detect_result, ensure_ascii=False, sort_keys=True)}"
        )

    source_detect_record_path = prompt_run_root / "records" / "detect_record.json"
    if not source_detect_record_path.exists() or not source_detect_record_path.is_file():
        raise FileNotFoundError(f"clean_negative detect record missing: {normalize_path_value(source_detect_record_path)}")

    staged_embed_record_path = shard_root / "records" / f"event_{event_index:06d}_embed_record.json"
    staged_detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_record.json"
    validate_path_within_base(shard_root, staged_embed_record_path, "staged embed record")
    validate_path_within_base(shard_root, staged_detect_record_path, "staged detect record")

    detect_payload_obj = _load_required_json_dict(source_detect_record_path, "clean_negative detect record")
    staged_embed_record_payload = _build_clean_negative_staged_embed_record(
        preview_image_path=preview_image_path,
        sample_role=PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
        event_id=event_id,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        seed=seed,
        probe_plan_anchors=probe_plan_anchors,
        probe_detect_payload=probe_detect_payload,
        bound_detect_payload=detect_payload_obj,
        provenance_payload=provenance_payload,
    )
    write_json_atomic(staged_embed_record_path, staged_embed_record_payload)

    event_embed_record_path = prompt_run_root / "records" / "embed_record.json"
    validate_path_within_base(shard_root, event_embed_record_path, "event embed record")
    write_json_atomic(event_embed_record_path, staged_embed_record_payload)
    _refresh_clean_negative_event_run_records_view(prompt_run_root)

    detect_payload_obj = pw01_stage_runtime_helpers._normalize_direct_detect_payload(
        detect_payload_obj,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
        record_usage=_resolve_event_record_usage(PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE),
    )
    write_json_atomic(staged_detect_record_path, detect_payload_obj)

    source_image_view = pw01_stage_runtime_helpers._resolve_source_pool_source_image_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    preview_generation_record_view = pw01_stage_runtime_helpers._resolve_source_pool_preview_generation_record_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )

    shard_relative_runtime_cfg = runtime_cfg_path.relative_to(shard_root).as_posix()
    shard_relative_embed_record = staged_embed_record_path.relative_to(shard_root).as_posix()
    shard_relative_detect_record = staged_detect_record_path.relative_to(shard_root).as_posix()

    event_manifest_payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_source_event",
        "event_id": event_id,
        "sample_role": PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
        "source_prompt_index": source_prompt_index,
        "event_index": event_index,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "runtime_config_path": normalize_path_value(runtime_cfg_path),
        "runtime_config_package_relative_path": shard_relative_runtime_cfg,
        "embed_record_path": normalize_path_value(staged_embed_record_path),
        "embed_record_package_relative_path": shard_relative_embed_record,
        "detect_record_path": normalize_path_value(staged_detect_record_path),
        "detect_record_package_relative_path": shard_relative_detect_record,
        "source_image": source_image_view,
        "preview_generation_record": preview_generation_record_view,
        "attestation_statement": None,
        "attestation_bundle": None,
        "attestation_result": None,
        "negative_branch_source_attestation_provenance": provenance_payload,
        "detect_probe_record_path": normalize_path_value(staged_probe_detect_record_path),
        "sha256": compute_file_sha256(staged_detect_record_path),
        "stage_results": stage_results,
    }

    event_manifest_path = event_root / "event_manifest.json"
    validate_path_within_base(shard_root, event_manifest_path, "event manifest path")
    write_json_atomic(event_manifest_path, event_manifest_payload)
    event_manifest_payload["event_manifest_path"] = normalize_path_value(event_manifest_path)
    return event_manifest_payload


def _run_source_event_by_role(
    *,
    event: Mapping[str, Any],
    shard_root: Path,
    default_cfg_obj: Mapping[str, Any],
    bound_config_path: Path,
) -> Dict[str, Any]:
    """
    Dispatch one PW01 event to the role-specific executor.

    Args:
        event: Event payload from source_event_grid.
        shard_root: Shard root path.
        default_cfg_obj: Parsed default config mapping.
        bound_config_path: Bound config source path.

    Returns:
        Event manifest payload.
    """
    sample_role = validate_source_sample_role(str(event.get("sample_role")))
    if sample_role == ACTIVE_SAMPLE_ROLE:
        return _run_positive_source_event(
            event=event,
            shard_root=shard_root,
            default_cfg_obj=default_cfg_obj,
            bound_config_path=bound_config_path,
        )
    if sample_role == CLEAN_NEGATIVE_SAMPLE_ROLE:
        return _run_clean_negative_event(
            event=event,
            shard_root=shard_root,
            default_cfg_obj=default_cfg_obj,
            bound_config_path=bound_config_path,
        )
    if sample_role == PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE:
        return _run_planner_conditioned_control_negative_event(
            event=event,
            shard_root=shard_root,
            default_cfg_obj=default_cfg_obj,
            bound_config_path=bound_config_path,
        )
    raise ValueError(f"unsupported PW01 sample_role: {sample_role}")


def _run_positive_source_event(
    *,
    event: Mapping[str, Any],
    shard_root: Path,
    default_cfg_obj: Mapping[str, Any],
    bound_config_path: Path,
) -> Dict[str, Any]:
    """
    Execute one positive_source event in event-bound mode.

    Args:
        event: Event payload from source_event_grid.
        shard_root: Shard root path.
        default_cfg_obj: Parsed default config mapping.
        bound_config_path: Bound config source path.

    Returns:
        Event manifest payload.
    """
    event_id = event.get("event_id")
    sample_role = event.get("sample_role")
    event_index = event.get("event_index")
    source_prompt_index = event.get("source_prompt_index")
    if source_prompt_index is None:
        source_prompt_index = event.get("prompt_index")
    prompt_text = event.get("prompt_text")
    prompt_sha256 = event.get("prompt_sha256")
    seed = event.get("seed")
    prompt_file = event.get("prompt_file")

    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be non-empty str")
    if sample_role != ACTIVE_SAMPLE_ROLE:
        raise ValueError(f"PW01 only supports positive_source, got: {sample_role}")
    if not isinstance(event_index, int) or event_index < 0:
        raise ValueError("event_index must be non-negative int")
    if not isinstance(source_prompt_index, int) or source_prompt_index < 0:
        raise ValueError("prompt_index must be non-negative int")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be non-empty str")
    if not isinstance(prompt_sha256, str) or not prompt_sha256:
        raise ValueError("prompt_sha256 must be non-empty str")
    if not isinstance(seed, int):
        raise ValueError("seed must be int")
    if not isinstance(prompt_file, str) or not prompt_file:
        raise ValueError("prompt_file must be non-empty str")
    if not isinstance(bound_config_path, Path):
        raise TypeError("bound_config_path must be Path")

    event_root = ensure_directory(shard_root / "events" / f"event_{event_index:06d}")
    prompt_run_root = ensure_directory(event_root / "run")
    validate_path_within_base(shard_root, event_root, "event root")
    validate_path_within_base(shard_root, prompt_run_root, "event run root")

    runtime_cfg = copy.deepcopy(dict(default_cfg_obj))
    runtime_cfg["inference_prompt"] = prompt_text
    runtime_cfg["seed"] = seed
    runtime_cfg["paper_workflow_event"] = {
        "event_id": event_id,
        "event_index": event_index,
        "source_prompt_index": source_prompt_index,
        "sample_role": ACTIVE_SAMPLE_ROLE,
    }

    runtime_cfg_path = event_root / "runtime_config.yaml"
    validate_path_within_base(shard_root, runtime_cfg_path, "event runtime config")
    runtime_cfg = apply_notebook_model_snapshot_binding(runtime_cfg, env_mapping=os.environ)
    _validate_bound_runtime_cfg(
        runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="preview_precompute",
    )
    write_yaml_mapping(runtime_cfg_path, runtime_cfg)

    preview_precompute = pw01_stage_runtime_helpers._prepare_source_pool_preview_artifact(
        cfg_obj=runtime_cfg,
        prompt_run_root=prompt_run_root,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
    )
    preview_runtime_cfg_node = preview_precompute.get("runtime_cfg")
    if not isinstance(preview_runtime_cfg_node, dict):
        raise ValueError("preview precompute result missing runtime_cfg")
    preview_runtime_cfg = cast(Dict[str, Any], preview_runtime_cfg_node)
    _validate_bound_runtime_cfg(
        preview_runtime_cfg,
        bound_config_path=bound_config_path,
        runtime_cfg_path=runtime_cfg_path,
        event_id=event_id,
        stage_label="embed_detect_execution",
    )
    write_yaml_mapping(runtime_cfg_path, preview_runtime_cfg)

    stage_results: Dict[str, Any] = {
        "preview_precompute": preview_precompute.get("preview_record", {}),
    }
    for stage_name in ["embed", "detect"]:
        command = pw01_stage_runtime_helpers._build_stage_command(stage_name, runtime_cfg_path, prompt_run_root)
        result = pw01_stage_runtime_helpers._run_stage(stage_name, command, prompt_run_root)
        stage_results[stage_name] = result
        if int(result.get("return_code", 1)) != 0:
            raise RuntimeError(
                f"PW01 event stage failed: stage={stage_name}, event_id={event_id}, "
                f"payload={json.dumps(result, ensure_ascii=False, sort_keys=True)}"
            )

    source_embed_record_path = prompt_run_root / "records" / "embed_record.json"
    source_detect_record_path = prompt_run_root / "records" / "detect_record.json"
    if not source_embed_record_path.exists() or not source_embed_record_path.is_file():
        raise FileNotFoundError(f"event embed record missing: {normalize_path_value(source_embed_record_path)}")
    if not source_detect_record_path.exists() or not source_detect_record_path.is_file():
        raise FileNotFoundError(f"event detect record missing: {normalize_path_value(source_detect_record_path)}")

    staged_embed_record_path = shard_root / "records" / f"event_{event_index:06d}_embed_record.json"
    staged_detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_record.json"
    validate_path_within_base(shard_root, staged_embed_record_path, "staged embed record")
    validate_path_within_base(shard_root, staged_detect_record_path, "staged detect record")
    copy_file(source_embed_record_path, staged_embed_record_path)

    family_id = resolve_family_id_from_path(shard_root)
    artifacts_root = ensure_directory(event_root / "artifacts")
    payload_reference_sidecar_path = artifacts_root / "payload_reference_sidecar.json"
    validate_path_within_base(shard_root, payload_reference_sidecar_path, "payload reference sidecar path")
    payload_reference_sidecar_payload: Dict[str, Any] | None = None
    try:
        payload_reference_sidecar_payload = build_payload_reference_sidecar_payload(
            family_id=family_id,
            stage_name="PW01_Source_Event_Shards",
            event_id=event_id,
            event_index=event_index,
            sample_role=ACTIVE_SAMPLE_ROLE,
            prompt_sha256=prompt_sha256,
            seed=seed,
            embed_record=_load_required_json_dict(staged_embed_record_path, "event embed record"),
        )
        write_json_atomic(payload_reference_sidecar_path, payload_reference_sidecar_payload)
    except (TypeError, ValueError):
        # 旧夹具可能未提供完整 payload 参考字段，此时保持 PW01 主流程兼容。
        payload_reference_sidecar_payload = None

    detect_payload_obj = _load_required_json_dict(source_detect_record_path, "event detect record")
    detect_payload_obj = pw01_stage_runtime_helpers._normalize_direct_detect_payload(
        detect_payload_obj,
        prompt_text=prompt_text,
        prompt_index=source_prompt_index,
        prompt_file_path=prompt_file,
        record_usage=_resolve_event_record_usage(ACTIVE_SAMPLE_ROLE),
    )
    write_json_atomic(staged_detect_record_path, detect_payload_obj)

    payload_decode_sidecar_path = artifacts_root / "payload_decode_sidecar.json"
    validate_path_within_base(shard_root, payload_decode_sidecar_path, "payload decode sidecar path")
    payload_decode_sidecar_payload: Dict[str, Any] | None = None
    if payload_reference_sidecar_payload is not None:
        try:
            payload_decode_sidecar_payload = build_payload_decode_sidecar_payload(
                family_id=family_id,
                stage_name="PW01_Source_Event_Shards",
                event_id=event_id,
                event_index=event_index,
                sample_role=ACTIVE_SAMPLE_ROLE,
                reference_event_id=event_id,
                detect_payload=detect_payload_obj,
                reference_sidecar=payload_reference_sidecar_payload,
            )
            write_json_atomic(payload_decode_sidecar_path, payload_decode_sidecar_payload)
        except (TypeError, ValueError):
            # 旧夹具 detect 记录可能缺少 LF 闭环细节，此时保留旧路径回退能力。
            payload_decode_sidecar_payload = None

    attestation_views = pw01_stage_runtime_helpers._resolve_source_pool_attestation_views(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    source_image_view = pw01_stage_runtime_helpers._resolve_source_pool_source_image_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    preview_generation_record_view = pw01_stage_runtime_helpers._resolve_source_pool_preview_generation_record_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )

    shard_relative_runtime_cfg = runtime_cfg_path.relative_to(shard_root).as_posix()
    shard_relative_embed_record = staged_embed_record_path.relative_to(shard_root).as_posix()
    shard_relative_detect_record = staged_detect_record_path.relative_to(shard_root).as_posix()

    event_manifest_payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_source_event",
        "event_id": event_id,
        "sample_role": ACTIVE_SAMPLE_ROLE,
        "source_prompt_index": source_prompt_index,
        "event_index": event_index,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "runtime_config_path": normalize_path_value(runtime_cfg_path),
        "runtime_config_package_relative_path": shard_relative_runtime_cfg,
        "embed_record_path": normalize_path_value(staged_embed_record_path),
        "embed_record_package_relative_path": shard_relative_embed_record,
        "detect_record_path": normalize_path_value(staged_detect_record_path),
        "detect_record_package_relative_path": shard_relative_detect_record,
        "payload_reference_sidecar_path": (
            normalize_path_value(payload_reference_sidecar_path)
            if payload_reference_sidecar_payload is not None
            else None
        ),
        "payload_reference_sidecar_package_relative_path": (
            payload_reference_sidecar_path.relative_to(shard_root).as_posix()
            if payload_reference_sidecar_payload is not None
            else None
        ),
        "payload_decode_sidecar_path": (
            normalize_path_value(payload_decode_sidecar_path)
            if payload_decode_sidecar_payload is not None
            else None
        ),
        "payload_decode_sidecar_package_relative_path": (
            payload_decode_sidecar_path.relative_to(shard_root).as_posix()
            if payload_decode_sidecar_payload is not None
            else None
        ),
        "source_image": source_image_view,
        "plain_preview_image": plain_preview_image_view,
        "preview_generation_record": preview_generation_record_view,
        "watermarked_output_image": watermarked_output_image_view,
        "attestation_statement": attestation_views.get("attestation_statement"),
        "attestation_bundle": attestation_views.get("attestation_bundle"),
        "attestation_result": attestation_views.get("attestation_result"),
        "sha256": compute_file_sha256(staged_detect_record_path),
        "stage_results": stage_results,
    }

    event_manifest_path = event_root / "event_manifest.json"
    validate_path_within_base(shard_root, event_manifest_path, "event manifest path")
    write_json_atomic(event_manifest_path, event_manifest_payload)
    event_manifest_payload["event_manifest_path"] = normalize_path_value(event_manifest_path)
    return event_manifest_payload


def run_pw01_source_event_shard_worker(
    *,
    drive_project_root: Path,
    family_id: str,
    shard_index: int,
    pw01_worker_count: int,
    local_worker_index: int,
    worker_plan_path: Path,
) -> Dict[str, Any]:
    """
    Execute one shard-local PW01 worker from a persisted worker plan.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        shard_index: Shard index.
        pw01_worker_count: Total shard-local worker count.
        local_worker_index: Local worker index.
        worker_plan_path: Worker plan JSON path.

    Returns:
        Worker result payload.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if shard_index < 0:
        raise TypeError("shard_index must be non-negative int")
    _validate_local_worker_index(local_worker_index, pw01_worker_count)
    if not isinstance(worker_plan_path, Path):
        raise TypeError("worker_plan_path must be Path")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    worker_plan = _load_required_json_dict(worker_plan_path, "PW01 worker plan")

    plan_family_id = worker_plan.get("family_id")
    if plan_family_id != family_id:
        raise ValueError(f"worker plan family_id mismatch: expected={family_id}, actual={plan_family_id}")
    plan_shard_index = worker_plan.get("shard_index")
    if not isinstance(plan_shard_index, int) or int(plan_shard_index) != shard_index:
        raise ValueError(f"worker plan shard_index mismatch: expected={shard_index}, actual={plan_shard_index}")
    plan_pw01_worker_count = worker_plan.get("pw01_worker_count")
    if not isinstance(plan_pw01_worker_count, int) or int(plan_pw01_worker_count) != pw01_worker_count:
        raise ValueError(
            "worker plan pw01_worker_count mismatch: "
            f"expected={pw01_worker_count}, actual={plan_pw01_worker_count}"
        )
    plan_local_worker_index = worker_plan.get("local_worker_index")
    if not isinstance(plan_local_worker_index, int) or int(plan_local_worker_index) != local_worker_index:
        raise ValueError(
            "worker plan local_worker_index mismatch: "
            f"expected={local_worker_index}, actual={plan_local_worker_index}"
        )
    plan_sample_role = validate_source_sample_role(str(worker_plan.get("sample_role")))

    default_config_path_value = worker_plan.get("default_config_path")
    if not isinstance(default_config_path_value, str) or not default_config_path_value:
        raise ValueError("worker plan missing default_config_path")
    default_config_path = Path(default_config_path_value).expanduser().resolve()

    bound_config_path_value = worker_plan.get("bound_config_path")
    if not isinstance(bound_config_path_value, str) or not bound_config_path_value:
        raise ValueError("worker plan missing bound_config_path")
    bound_config_path, default_cfg_obj = _load_required_bound_config(
        Path(bound_config_path_value),
        stage_label="worker_execution",
    )

    shard_root_value = worker_plan.get("shard_root")
    if not isinstance(shard_root_value, str) or not shard_root_value:
        raise ValueError("worker plan missing shard_root")
    shard_root = Path(shard_root_value).expanduser().resolve()
    validate_path_within_base(family_root, shard_root, "worker shard root")

    worker_root = _build_worker_root(shard_root, local_worker_index)
    ensure_directory(worker_root)
    worker_result_path = _build_worker_result_path(worker_root)
    validate_path_within_base(shard_root, worker_result_path, "worker result path")

    plan_assigned_event_ids = worker_plan.get("assigned_event_ids")
    if not isinstance(plan_assigned_event_ids, list):
        raise ValueError("worker plan assigned_event_ids must be list")
    assigned_event_ids = [
        str(event_id)
        for event_id in cast(List[object], plan_assigned_event_ids)
        if isinstance(event_id, str) and event_id
    ]
    if len(assigned_event_ids) != len(plan_assigned_event_ids):
        raise ValueError("worker plan assigned_event_ids must contain non-empty str")

    plan_assigned_event_indices = worker_plan.get("assigned_event_indices")
    if not isinstance(plan_assigned_event_indices, list):
        raise ValueError("worker plan assigned_event_indices must be list")
    assigned_event_indices = [
        int(event_index)
        for event_index in cast(List[object], plan_assigned_event_indices)
        if isinstance(event_index, int) and event_index >= 0
    ]
    if len(assigned_event_indices) != len(plan_assigned_event_indices):
        raise ValueError("worker plan assigned_event_indices must contain non-negative int")

    assigned_events_node = worker_plan.get("assigned_events")
    if not isinstance(assigned_events_node, list):
        raise ValueError("worker plan assigned_events must be list")
    assigned_events = [
        cast(Dict[str, Any], event_node)
        for event_node in cast(List[object], assigned_events_node)
        if isinstance(event_node, dict)
    ]
    if len(assigned_events) != len(assigned_events_node):
        raise ValueError("worker plan assigned_events must contain objects")

    executed_events: List[Dict[str, Any]] = []
    try:
        for event in assigned_events:
            executed_events.append(
                _run_source_event_by_role(
                    event=event,
                    shard_root=shard_root,
                    default_cfg_obj=default_cfg_obj,
                    bound_config_path=bound_config_path,
                )
            )

        worker_result = _build_worker_result_payload(
            family_id=family_id,
            sample_role=plan_sample_role,
            shard_index=shard_index,
            shard_count=int(worker_plan.get("source_shard_count", 0)),
            pw01_worker_count=pw01_worker_count,
            local_worker_index=local_worker_index,
            worker_root=worker_root,
            worker_plan_path=worker_plan_path,
            assigned_event_ids=assigned_event_ids,
            assigned_event_indices=assigned_event_indices,
            events=executed_events,
            status="completed",
        )
        write_json_atomic(worker_result_path, worker_result)
        return worker_result
    except Exception as exc:
        worker_result = _build_worker_result_payload(
            family_id=family_id,
            sample_role=plan_sample_role,
            shard_index=shard_index,
            shard_count=int(worker_plan.get("source_shard_count", 0)),
            pw01_worker_count=pw01_worker_count,
            local_worker_index=local_worker_index,
            worker_root=worker_root,
            worker_plan_path=worker_plan_path,
            assigned_event_ids=assigned_event_ids,
            assigned_event_indices=assigned_event_indices,
            events=executed_events,
            status="failed",
            failure_reason="pw01_worker_event_execution_failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        worker_result["traceback"] = traceback.format_exc()
        write_json_atomic(worker_result_path, worker_result)
        return worker_result


def run_pw01_source_event_shard(
    *,
    drive_project_root: Path,
    family_id: str,
    shard_index: int,
    shard_count: int,
    sample_role: str = ACTIVE_SAMPLE_ROLE,
    pw01_worker_count: int = DEFAULT_PW01_WORKER_COUNT,
    bound_config_path: Path | None = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Execute one isolated PW01 source-role shard.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        shard_index: Source shard index.
        shard_count: Source shard count.
        sample_role: Source sample role for the shard.
        pw01_worker_count: Total shard-local worker count.
        bound_config_path: Notebook-bound config snapshot path.
        force_rerun: Whether to clear completed shard and rerun.

    Returns:
        PW01 shard summary payload.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if shard_index < 0:
        raise TypeError("shard_index must be non-negative int")
    if shard_count <= 0:
        raise TypeError("shard_count must be positive int")
    normalized_sample_role = validate_source_sample_role(sample_role)
    _validate_pw01_worker_count(pw01_worker_count)

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    layout = resolve_family_layout_paths(family_root)

    family_manifest_path = layout["family_manifest_path"]
    shard_plan_path = layout["source_shard_plan_path"]
    source_event_grid_path = layout["source_event_grid_path"]

    family_manifest = _load_required_json_dict(family_manifest_path, "paper eval family manifest")
    source_shard_plan = _load_required_json_dict(shard_plan_path, "source shard plan")
    manifest_shard_count = _resolve_manifest_shard_count(family_manifest)
    if manifest_shard_count != shard_count:
        raise ValueError(
            f"shard_count mismatch with family manifest: expected={manifest_shard_count}, actual={shard_count}"
        )

    shard_assignment = resolve_source_shard_assignment(
        source_shard_plan,
        sample_role=normalized_sample_role,
        shard_index=shard_index,
        shard_count=shard_count,
    )

    assigned_event_ids_raw = shard_assignment.get("assigned_event_ids", [])
    if not isinstance(assigned_event_ids_raw, list):
        raise ValueError("assigned_event_ids must be list")
    assigned_event_ids: List[str] = []
    for event_id in cast(List[object], assigned_event_ids_raw):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("assigned_event_ids must contain non-empty str")
        assigned_event_ids.append(event_id)

    event_lookup = _load_event_lookup(source_event_grid_path)
    assigned_events: List[Dict[str, Any]] = []
    for event_id in assigned_event_ids:
        if event_id not in event_lookup:
            raise ValueError(f"assigned event_id not found in source_event_grid: {event_id}")
        assigned_events.append(event_lookup[event_id])

    default_config_path = _resolve_default_config_path(family_manifest)
    resolved_bound_config_path, default_cfg_obj = _load_required_bound_config(
        bound_config_path,
        stage_label="shard_execution",
    )

    shard_root = _build_role_shard_root(family_root, normalized_sample_role, shard_index)
    shard_manifest_path = shard_root / "shard_manifest.json"

    if shard_root.exists():
        existing_manifest = read_json_dict(shard_manifest_path)
        existing_status = existing_manifest.get("status")
        if existing_status == "completed" and not force_rerun:
            raise RuntimeError(
                "shard already completed; use --force-rerun to clear and rerun: "
                f"{normalize_path_value(shard_root)}"
            )
        if not force_rerun:
            raise RuntimeError(
                "shard directory already exists and rerun is blocked without --force-rerun: "
                f"{normalize_path_value(shard_root)}"
            )
        shutil.rmtree(shard_root)

    ensure_directory(shard_root)
    ensure_directory(shard_root / "events")
    ensure_directory(shard_root / "records")
    ensure_directory(shard_root / "artifacts")
    ensure_directory(shard_root / "logs")

    running_manifest: Dict[str, Any] = {
        "artifact_type": "paper_workflow_source_shard_manifest",
        "schema_version": "pw01_v1",
        "family_id": family_id,
        "sample_role": normalized_sample_role,
        "shard_index": shard_index,
        "source_shard_count": shard_count,
        "pw01_worker_count": pw01_worker_count,
        "worker_execution_mode": "single_process" if pw01_worker_count == 1 else "shard_local_subprocess_parallel",
        "status": "running",
        "created_at": utc_now_iso(),
        "force_rerun": bool(force_rerun),
        "family_manifest_path": normalize_path_value(family_manifest_path),
        "source_shard_plan_path": normalize_path_value(shard_plan_path),
        "source_event_grid_path": normalize_path_value(source_event_grid_path),
        "default_config_path": normalize_path_value(default_config_path),
        "bound_config_path": normalize_path_value(resolved_bound_config_path),
        "shard_root": normalize_path_value(shard_root),
        "assigned_event_ids": assigned_event_ids,
        "event_count": len(assigned_event_ids),
        "workers": [],
        "events": [],
        "failure_reason": None,
        "traceback": None,
    }
    write_json_atomic(shard_manifest_path, running_manifest)

    executed_events: List[Dict[str, Any]] = []
    worker_plans: List[Dict[str, Any]] = []
    worker_executions: List[Dict[str, Any]] = []
    worker_results: List[Dict[str, Any]] = []
    try:
        if pw01_worker_count == 1:
            for event in assigned_events:
                executed_events.append(
                    _run_source_event_by_role(
                        event=event,
                        shard_root=shard_root,
                        default_cfg_obj=default_cfg_obj,
                        bound_config_path=resolved_bound_config_path,
                    )
                )
        else:
            worker_plans = _prepare_local_worker_plans(
                drive_project_root=normalized_drive_root,
                family_id=family_id,
                sample_role=normalized_sample_role,
                shard_index=shard_index,
                shard_count=shard_count,
                pw01_worker_count=pw01_worker_count,
                shard_root=shard_root,
                default_config_path=default_config_path,
                bound_config_path=resolved_bound_config_path,
                assigned_events=assigned_events,
            )
            running_manifest["workers"] = _build_worker_state_rows(
                worker_plans=worker_plans,
                worker_executions=[],
                worker_results=[],
            )
            write_json_atomic(shard_manifest_path, running_manifest)

            worker_executions, worker_results = _run_local_worker_plans(worker_plans)
            if len(worker_results) != pw01_worker_count or any(
                worker_result.get("status") != "completed" for worker_result in worker_results
            ):
                raise RuntimeError(
                    "PW01 shard-local worker execution failed: "
                    f"{json.dumps(_build_worker_execution_failure_payload(worker_executions), ensure_ascii=False, sort_keys=True)}"
                )
            executed_events = _merge_completed_worker_events(
                worker_results=worker_results,
                assigned_event_ids=assigned_event_ids,
            )

        completed_manifest = dict(running_manifest)
        completed_manifest["status"] = "completed"
        completed_manifest["completed_at"] = utc_now_iso()
        completed_manifest["workers"] = _build_worker_state_rows(
            worker_plans=worker_plans,
            worker_executions=worker_executions,
            worker_results=worker_results,
        )
        completed_manifest["events"] = executed_events
        write_json_atomic(shard_manifest_path, completed_manifest)

        return {
            "status": "ok",
            "stage_name": "PW01_Source_Event_Shards",
            "family_id": family_id,
            "sample_role": normalized_sample_role,
            "shard_index": shard_index,
            "source_shard_count": shard_count,
            "pw01_worker_count": pw01_worker_count,
            "event_count": len(executed_events),
            "shard_root": normalize_path_value(shard_root),
            "shard_manifest_path": normalize_path_value(shard_manifest_path),
            "bound_config_path": normalize_path_value(resolved_bound_config_path),
        }
    except Exception as exc:
        failed_manifest = dict(running_manifest)
        failed_manifest["status"] = "failed"
        failed_manifest["failed_at"] = utc_now_iso()
        failed_manifest["workers"] = _build_worker_state_rows(
            worker_plans=worker_plans,
            worker_executions=worker_executions,
            worker_results=worker_results,
        )
        failed_manifest["events"] = executed_events or _collect_partial_worker_events(worker_results)
        failed_manifest["failure_reason"] = str(exc)
        failed_manifest["traceback"] = traceback.format_exc()
        write_json_atomic(shard_manifest_path, failed_manifest)
        raise
