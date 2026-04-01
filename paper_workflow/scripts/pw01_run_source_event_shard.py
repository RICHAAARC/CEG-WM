"""
File purpose: Execute one PW01 positive_source event shard in isolation.
Module type: General module
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
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
    DEFAULT_CONFIG_RELATIVE_PATH,
    build_family_root,
    read_jsonl,
    resolve_family_layout_paths,
)

BASE_RUNNER_SCRIPT_PATH = Path("scripts/01_run_paper_full_cuda.py")
EVENT_RECORD_USAGE = "paper_workflow_positive_source"


def _load_base_runner_module() -> ModuleType:
    """
    Load baseline stage-01 runner module from file path.

    Args:
        None.

    Returns:
        Loaded module object.

    Raises:
        RuntimeError: If runner module cannot be loaded.
    """
    runner_path = (REPO_ROOT / BASE_RUNNER_SCRIPT_PATH).resolve()
    spec = importlib.util.spec_from_file_location("pw01_stage_01_base_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load stage-01 runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_RUNNER_MODULE = _load_base_runner_module()


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

    Raises:
        ValueError: If shard count or shard index is inconsistent.
    """
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative int")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive int")

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
    positive_plan_node = sample_role_plans.get(ACTIVE_SAMPLE_ROLE)
    if not isinstance(positive_plan_node, Mapping):
        raise ValueError("source shard plan missing positive_source plan")
    positive_plan = cast(Dict[str, Any], positive_plan_node)

    shards_node = positive_plan.get("shards")
    if not isinstance(shards_node, list):
        raise ValueError("source shard plan positive_source.shards must be list")

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

    raise ValueError(f"shard_index not found in positive_source shard plan: {shard_index}")


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


def _build_shard_root(family_root: Path, shard_index: int) -> Path:
    """
    Build shard root path for one positive shard index.

    Args:
        family_root: Family root path.
        shard_index: Shard index.

    Returns:
        Shard root path.
    """
    shard_root = family_root / "source_shards" / "positive" / f"shard_{shard_index:04d}"
    validate_path_within_base(family_root, shard_root, "source shard root")
    return shard_root


def _run_positive_source_event(
    *,
    event: Mapping[str, Any],
    shard_root: Path,
    default_cfg_obj: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Execute one positive_source event in event-bound mode.

    Args:
        event: Event payload from source_event_grid.
        shard_root: Shard root path.
        default_cfg_obj: Parsed default config mapping.

    Returns:
        Event manifest payload.
    """
    event_id = event.get("event_id")
    sample_role = event.get("sample_role")
    event_index = event.get("event_index")
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
    write_yaml_mapping(runtime_cfg_path, runtime_cfg)

    preview_precompute = BASE_RUNNER_MODULE._prepare_source_pool_preview_artifact(
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
    write_yaml_mapping(runtime_cfg_path, preview_runtime_cfg)

    stage_results: Dict[str, Any] = {
        "preview_precompute": preview_precompute.get("preview_record", {}),
    }
    for stage_name in ["embed", "detect"]:
        command = BASE_RUNNER_MODULE._build_stage_command(stage_name, runtime_cfg_path, prompt_run_root)
        result = BASE_RUNNER_MODULE._run_stage(stage_name, command, prompt_run_root)
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

    detect_payload_obj = _load_required_json_dict(source_detect_record_path, "event detect record")
    normalize_fn = getattr(BASE_RUNNER_MODULE, "_normalize_direct_detect_payload", None)
    if callable(normalize_fn):
        normalized_payload = normalize_fn(
            detect_payload_obj,
            prompt_text=prompt_text,
            prompt_index=source_prompt_index,
            prompt_file_path=prompt_file,
            record_usage=EVENT_RECORD_USAGE,
        )
        if not isinstance(normalized_payload, dict):
            raise ValueError("normalized detect payload must be JSON object")
        detect_payload_obj = cast(Dict[str, Any], normalized_payload)
    write_json_atomic(staged_detect_record_path, detect_payload_obj)

    attestation_views = BASE_RUNNER_MODULE._resolve_source_pool_attestation_views(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    source_image_view = BASE_RUNNER_MODULE._resolve_source_pool_source_image_view(
        cfg_obj=preview_runtime_cfg,
        run_root=shard_root,
        prompt_run_root=prompt_run_root,
        prompt_index=event_index,
    )
    preview_generation_record_view = BASE_RUNNER_MODULE._resolve_source_pool_preview_generation_record_view(
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
        "source_image": source_image_view,
        "preview_generation_record": preview_generation_record_view,
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


def run_pw01_source_event_shard(
    *,
    drive_project_root: Path,
    family_id: str,
    shard_index: int,
    shard_count: int,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Execute one isolated PW01 positive_source shard.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        shard_index: Positive shard index.
        shard_count: Positive shard count.
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

    shard_assignment = resolve_positive_shard_assignment(
        source_shard_plan,
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
    default_cfg_obj = load_yaml_mapping(default_config_path)

    shard_root = _build_shard_root(family_root, shard_index)
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
        "schema_version": "pw_stage_01_v1",
        "family_id": family_id,
        "sample_role": ACTIVE_SAMPLE_ROLE,
        "shard_index": shard_index,
        "source_shard_count": shard_count,
        "status": "running",
        "created_at": utc_now_iso(),
        "force_rerun": bool(force_rerun),
        "family_manifest_path": normalize_path_value(family_manifest_path),
        "source_shard_plan_path": normalize_path_value(shard_plan_path),
        "source_event_grid_path": normalize_path_value(source_event_grid_path),
        "default_config_path": normalize_path_value(default_config_path),
        "shard_root": normalize_path_value(shard_root),
        "assigned_event_ids": assigned_event_ids,
        "event_count": len(assigned_event_ids),
        "events": [],
        "failure_reason": None,
        "traceback": None,
    }
    write_json_atomic(shard_manifest_path, running_manifest)

    executed_events: List[Dict[str, Any]] = []
    try:
        for event in assigned_events:
            executed_events.append(
                _run_positive_source_event(
                    event=event,
                    shard_root=shard_root,
                    default_cfg_obj=default_cfg_obj,
                )
            )

        completed_manifest = dict(running_manifest)
        completed_manifest["status"] = "completed"
        completed_manifest["completed_at"] = utc_now_iso()
        completed_manifest["events"] = executed_events
        write_json_atomic(shard_manifest_path, completed_manifest)

        return {
            "status": "ok",
            "stage_name": "PW01_Source_Event_Shards",
            "family_id": family_id,
            "sample_role": ACTIVE_SAMPLE_ROLE,
            "shard_index": shard_index,
            "source_shard_count": shard_count,
            "event_count": len(executed_events),
            "shard_root": normalize_path_value(shard_root),
            "shard_manifest_path": normalize_path_value(shard_manifest_path),
        }
    except Exception as exc:
        failed_manifest = dict(running_manifest)
        failed_manifest["status"] = "failed"
        failed_manifest["failed_at"] = utc_now_iso()
        failed_manifest["events"] = executed_events
        failed_manifest["failure_reason"] = str(exc)
        failed_manifest["traceback"] = traceback.format_exc()
        write_json_atomic(shard_manifest_path, failed_manifest)
        raise
