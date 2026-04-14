"""
File purpose: Build paper workflow family manifest and source shard plan.
Module type: General module
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    normalize_path_value,
    utc_now_iso,
    write_json_atomic,
    write_yaml_mapping,
)

from paper_workflow.scripts.pw_common import (
    ACTIVE_SAMPLE_ROLE,
    ACTIVE_SOURCE_SAMPLE_ROLES,
    ATTACKED_NEGATIVE_SAMPLE_ROLE,
    ATTACKED_POSITIVE_SAMPLE_ROLE,
    ATTACK_SEVERITY_AXIS_KIND,
    ATTACK_SEVERITY_RULE_VERSION,
    CLEAN_NEGATIVE_SAMPLE_ROLE,
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
    DEFAULT_CONFIG_RELATIVE_PATH,
    DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH,
    GEOMETRY_OPTIONAL_CLAIM_DIRECTIONALITY,
    GEOMETRY_OPTIONAL_CLAIM_MODE,
    GEOMETRY_OPTIONAL_CLAIM_PLAN_FILE_NAME,
    GEOMETRY_OPTIONAL_CLAIM_SCOPE,
    RESERVED_SAMPLE_ROLES,
    SOURCE_TRUTH_STAGE,
    build_attack_condition_catalog,
    build_attack_event_grid,
    build_attack_shard_plan,
    build_family_root,
    build_method_identity_snapshot,
    build_source_event_grid,
    build_source_shard_plan,
    build_source_split_plan,
    ensure_family_layout,
    load_default_config_snapshot,
    load_prompt_lines,
    parse_seed_list,
    resolve_family_layout_paths,
    write_jsonl,
)


def _load_pw_base_cfg() -> Dict[str, Any]:
    """
    Load pw_base config mapping.

    Args:
        None.

    Returns:
        Parsed pw_base config mapping.
    """
    pw_base_path = (REPO_ROOT / DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH).resolve()
    from scripts.notebook_runtime_common import load_yaml_mapping

    return load_yaml_mapping(pw_base_path)


def _resolve_source_alignment_reference_files(pw_base_cfg: Dict[str, Any]) -> list[str]:
    """
    Resolve source alignment file list for method identity snapshot.

    Args:
        pw_base_cfg: Parsed pw_base config mapping.

    Returns:
        Source alignment file list.
    """
    candidates = pw_base_cfg.get("source_alignment_reference_files")
    if isinstance(candidates, list):
        output: list[str] = []
        for item in cast(list[object], candidates):
            if isinstance(item, str) and item:
                output.append(item)
            else:
                output = []
                break
        if output:
            return output
    return [
        "paper_workflow/configs/pw_base.yaml",
        "paper_workflow/scripts/pw_common.py",
        "paper_workflow/scripts/pw00_build_family_manifest.py",
        "paper_workflow/scripts/pw01_stage_runtime_helpers.py",
        "paper_workflow/scripts/pw01_run_source_event_shard.py",
        "paper_workflow/notebook/PW00_Paper_Eval_Family_Manifest.ipynb",
        "paper_workflow/notebook/PW01_Source_Event_Shards.ipynb",
        "scripts/notebook_runtime_common.py",
        "configs/default.yaml",
    ]


def _build_wrong_event_attestation_challenge_plan(
    *,
    family_id: str,
    positive_source_events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the frozen wrong-event attestation challenge plan for PW03.

    Args:
        family_id: Family identifier.
        positive_source_events: Ordered positive-source parent events.

    Returns:
        Challenge plan payload keyed by positive parent event.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(positive_source_events, Sequence):
        raise TypeError("positive_source_events must be Sequence")

    ordered_positive_events = sorted(
        [dict(cast(Mapping[str, Any], event)) for event in positive_source_events],
        key=lambda item: int(item.get("event_index", -1)),
    )
    plan_rows: list[Dict[str, Any]] = []

    def _derive_challenge_type(
        source_event: Mapping[str, Any],
        challenge_event: Mapping[str, Any] | None,
    ) -> str:
        if not isinstance(challenge_event, Mapping):
            return "wrong_statement"

        source_prompt_text = str(source_event.get("prompt_text") or "")
        challenge_prompt_text = str(challenge_event.get("prompt_text") or "")
        source_seed = source_event.get("seed")
        challenge_seed = challenge_event.get("seed")
        same_prompt = bool(source_prompt_text) and source_prompt_text == challenge_prompt_text
        same_seed = source_seed == challenge_seed

        if same_prompt and not same_seed:
            return "same_prompt_wrong_seed"
        if same_seed and not same_prompt:
            return "same_seed_wrong_prompt"
        if same_prompt and same_seed:
            return "same_prompt_same_seed_wrong_event"
        return "different_prompt_and_seed"

    if len(ordered_positive_events) < 2:
        for positive_event in ordered_positive_events:
            plan_rows.append(
                {
                    "challenge_type": _derive_challenge_type(positive_event, None),
                    "source_event_id": positive_event.get("event_id"),
                    "challenged_event_id": None,
                    "parent_event_id": positive_event.get("event_id"),
                    "parent_event_index": positive_event.get("event_index"),
                    "challenge_parent_event_id": None,
                    "challenge_parent_event_index": None,
                    "status": "not_available",
                    "reason": "requires_at_least_two_distinct_positive_parent_events",
                    "assignment_policy": "cyclic_next_positive_parent_event_by_event_index",
                }
            )
        return {
            "artifact_type": "paper_workflow_wrong_event_attestation_challenge_plan",
            "schema_version": "pw_stage_00_v1",
            "created_at": utc_now_iso(),
            "family_id": family_id,
            "plan_policy": "cyclic_next_positive_parent_event_by_event_index",
            "positive_parent_event_count": len(ordered_positive_events),
            "available_assignment_count": 0,
            "rows": plan_rows,
        }

    for event_index, positive_event in enumerate(ordered_positive_events):
        challenge_parent_event = ordered_positive_events[(event_index + 1) % len(ordered_positive_events)]
        plan_rows.append(
            {
                "challenge_type": _derive_challenge_type(positive_event, challenge_parent_event),
                "source_event_id": positive_event.get("event_id"),
                "challenged_event_id": challenge_parent_event.get("event_id"),
                "parent_event_id": positive_event.get("event_id"),
                "parent_event_index": positive_event.get("event_index"),
                "challenge_parent_event_id": challenge_parent_event.get("event_id"),
                "challenge_parent_event_index": challenge_parent_event.get("event_index"),
                "status": "ready",
                "reason": None,
                "assignment_policy": "cyclic_next_positive_parent_event_by_event_index",
            }
        )

    return {
        "artifact_type": "paper_workflow_wrong_event_attestation_challenge_plan",
        "schema_version": "pw_stage_00_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "plan_policy": "cyclic_next_positive_parent_event_by_event_index",
        "positive_parent_event_count": len(ordered_positive_events),
        "available_assignment_count": len(plan_rows),
        "rows": plan_rows,
    }


def _build_geometry_optional_claim_plan(
    *,
    family_id: str,
    attack_event_grid: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the frozen geometry optional-claim evidence plan for PW03/PW04.

    Args:
        family_id: Family identifier.
        attack_event_grid: Ordered attack-event grid rows.

    Returns:
        Geometry optional-claim plan payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_grid, Sequence):
        raise TypeError("attack_event_grid must be Sequence")

    plan_rows: list[Dict[str, Any]] = []
    available_assignment_count = 0
    for attack_event in attack_event_grid:
        attack_event_payload = dict(cast(Mapping[str, Any], attack_event))
        sample_role = str(attack_event_payload.get("sample_role") or "")
        eligible_for_optional_claim = sample_role == ATTACKED_POSITIVE_SAMPLE_ROLE
        row_status = "ready" if eligible_for_optional_claim else "not_applicable"
        row_reason = None if eligible_for_optional_claim else "sample_role_not_attacked_positive"
        if eligible_for_optional_claim:
            available_assignment_count += 1
        plan_rows.append(
            {
                "attack_event_id": attack_event_payload.get("event_id"),
                "attack_event_index": attack_event_payload.get("attack_event_index"),
                "parent_event_id": attack_event_payload.get("parent_event_id"),
                "sample_role": sample_role,
                "attack_family": attack_event_payload.get("attack_family"),
                "attack_condition_key": attack_event_payload.get("attack_condition_key"),
                "attack_config_name": attack_event_payload.get("attack_config_name"),
                "severity_status": attack_event_payload.get("severity_status"),
                "severity_reason": attack_event_payload.get("severity_reason"),
                "severity_label": attack_event_payload.get("severity_label"),
                "severity_level_index": attack_event_payload.get("severity_level_index"),
                "claim_mode": GEOMETRY_OPTIONAL_CLAIM_MODE if eligible_for_optional_claim else None,
                "claim_scope": GEOMETRY_OPTIONAL_CLAIM_SCOPE,
                "content_positive_veto_allowed": False,
                "rescue_directionality": GEOMETRY_OPTIONAL_CLAIM_DIRECTIONALITY,
                "eligible_for_optional_claim": eligible_for_optional_claim,
                "status": row_status,
                "reason": row_reason,
            }
        )

    return {
        "artifact_type": "paper_workflow_geometry_optional_claim_plan",
        "schema_version": "pw_stage_00_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "claim_mode": GEOMETRY_OPTIONAL_CLAIM_MODE,
        "claim_scope": GEOMETRY_OPTIONAL_CLAIM_SCOPE,
        "content_positive_veto_allowed": False,
        "rescue_directionality": GEOMETRY_OPTIONAL_CLAIM_DIRECTIONALITY,
        "attack_event_count": len(plan_rows),
        "available_assignment_count": available_assignment_count,
        "rows": plan_rows,
    }


def run_pw00_build_family_manifest(
    *,
    drive_project_root: Path,
    family_id: str,
    prompt_file: str,
    seed_list: Sequence[int] | str,
    source_shard_count: int,
    attack_shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    Build PW00 family manifest outputs.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        prompt_file: Prompt file path.
        seed_list: Seed list or seed-list text.
        source_shard_count: Source shard count.
        attack_shard_count: Optional attack shard count frozen for PW03.

    Returns:
        PW00 summary payload.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")
    if source_shard_count <= 0:
        raise TypeError("source_shard_count must be positive int")
    if attack_shard_count is not None and (
        not isinstance(attack_shard_count, int)
        or isinstance(attack_shard_count, bool)
        or attack_shard_count <= 0
    ):
        raise TypeError("attack_shard_count must be positive int when provided")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    resolved_attack_shard_count = source_shard_count if attack_shard_count is None else int(attack_shard_count)
    family_root = build_family_root(normalized_drive_root, family_id)
    layout = ensure_family_layout(family_root)
    _ = resolve_family_layout_paths(family_root)

    pw_base_cfg = _load_pw_base_cfg()
    normalized_seeds = parse_seed_list(seed_list)
    prompt_path, prompt_lines = load_prompt_lines(prompt_file)
    prompt_file_normalized = normalize_path_value(prompt_path)

    calibration_fraction_raw = pw_base_cfg.get("calibration_fraction")
    if not isinstance(calibration_fraction_raw, (int, float)) or isinstance(calibration_fraction_raw, bool):
        raise TypeError("paper_workflow/configs/pw_base.yaml calibration_fraction must be numeric")
    calibration_fraction = float(calibration_fraction_raw)

    event_grid = build_source_event_grid(
        family_id=family_id,
        prompt_lines=prompt_lines,
        seeds=normalized_seeds,
        prompt_file=prompt_file_normalized,
        sample_roles=ACTIVE_SOURCE_SAMPLE_ROLES,
    )
    source_shard_plan = build_source_shard_plan(
        family_id=family_id,
        source_shard_count=source_shard_count,
        events=event_grid,
    )
    source_split_plan = build_source_split_plan(
        family_id=family_id,
        events=event_grid,
        calibration_fraction=calibration_fraction,
    )

    positive_source_event_count = sum(
        1
        for event in event_grid
        if str(event.get("sample_role")) == ACTIVE_SAMPLE_ROLE
    )
    clean_negative_event_count = sum(
        1
        for event in event_grid
        if str(event.get("sample_role")) == CLEAN_NEGATIVE_SAMPLE_ROLE
    )
    control_negative_event_count = sum(
        1
        for event in event_grid
        if str(event.get("sample_role")) == PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE
    )

    positive_source_events = [
        event
        for event in event_grid
        if str(event.get("sample_role")) == ACTIVE_SAMPLE_ROLE
    ]
    clean_negative_events = [
        event
        for event in event_grid
        if str(event.get("sample_role")) == CLEAN_NEGATIVE_SAMPLE_ROLE
    ]
    wrong_event_attestation_challenge_plan = _build_wrong_event_attestation_challenge_plan(
        family_id=family_id,
        positive_source_events=positive_source_events,
    )
    attack_conditions = build_attack_condition_catalog()
    severity_status_counts: Dict[str, int] = {}
    severity_family_level_counts: Dict[str, int] = {}
    for attack_condition in attack_conditions:
        severity_status = str(attack_condition.get("severity_status") or "not_available")
        severity_status_counts[severity_status] = severity_status_counts.get(severity_status, 0) + 1
        if severity_status != "ok":
            continue
        attack_family_name = str(attack_condition.get("attack_family") or "<absent>")
        severity_family_level_counts[attack_family_name] = severity_family_level_counts.get(attack_family_name, 0) + 1
    attack_event_grid = build_attack_event_grid(
        family_id=family_id,
        parent_events=[*positive_source_events, *clean_negative_events],
        attack_conditions=attack_conditions,
    )
    attacked_positive_event_count = sum(
        1
        for event in attack_event_grid
        if str(event.get("sample_role")) == ATTACKED_POSITIVE_SAMPLE_ROLE
    )
    attacked_negative_event_count = sum(
        1
        for event in attack_event_grid
        if str(event.get("sample_role")) == ATTACKED_NEGATIVE_SAMPLE_ROLE
    )
    attack_shard_plan = build_attack_shard_plan(
        family_id=family_id,
        attack_shard_count=resolved_attack_shard_count,
        events=attack_event_grid,
    )
    attack_event_grid_path = layout["manifests_root"] / "attack_event_grid.jsonl"
    attack_shard_plan_path = layout["manifests_root"] / "attack_shard_plan.json"
    wrong_event_attestation_challenge_plan_path = (
        layout["manifests_root"] / "wrong_event_attestation_challenge_plan.json"
    )
    geometry_optional_claim_plan_path = layout["manifests_root"] / GEOMETRY_OPTIONAL_CLAIM_PLAN_FILE_NAME
    attack_condition_count = len(attack_conditions)
    attack_event_count = len(attack_event_grid)
    geometry_optional_claim_plan = _build_geometry_optional_claim_plan(
        family_id=family_id,
        attack_event_grid=attack_event_grid,
    )

    default_cfg_path = (REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
    default_cfg_obj = load_default_config_snapshot(REPO_ROOT)
    source_alignment_reference_files = _resolve_source_alignment_reference_files(pw_base_cfg)
    method_identity_snapshot = build_method_identity_snapshot(
        default_cfg_obj=default_cfg_obj,
        default_cfg_path=default_cfg_path,
        source_alignment_reference_files=source_alignment_reference_files,
    )

    config_snapshot = copy.deepcopy(default_cfg_obj)
    family_manifest: Dict[str, Any] = {
        "artifact_type": "paper_eval_family_manifest",
        "schema_version": "pw01_v1",
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "drive_project_root": normalize_path_value(normalized_drive_root),
        "family_root_relative": "paper_workflow/families",
        "source_truth_stage": SOURCE_TRUTH_STAGE,
        "stage_boundary": {
            "implemented": ["PW00", "PW01", "PW02", "PW03", "PW04", "PW05"],
            "excluded": [],
        },
        "sample_roles": {
            "active": list(ACTIVE_SOURCE_SAMPLE_ROLES),
            "reserved": list(RESERVED_SAMPLE_ROLES),
        },
        "event_identity": {
            "unit_definition": "sample_event = (prompt_index, prompt_text, seed, sample_role)",
            "event_id_fields": [
                "family_id",
                "sample_role",
                "prompt_index",
                "prompt_sha256",
                "seed",
            ],
            "event_id_digest": "sha256",
        },
        "source_parameters": {
            "prompt_file": prompt_file_normalized,
            "prompt_count": len(prompt_lines),
            "seed_list": list(normalized_seeds),
            "seed_count": len(normalized_seeds),
            "source_shard_count": source_shard_count,
            "calibration_fraction": calibration_fraction,
        },
        "attack_parameters": {
            "attack_shard_count": resolved_attack_shard_count,
            "materialization_profile": "first_value_per_condition",
            "attack_condition_count": attack_condition_count,
            "attack_event_count": attack_event_count,
            "attacked_positive_event_count": attacked_positive_event_count,
            "attacked_negative_event_count": attacked_negative_event_count,
            "wrong_event_attestation_challenge_plan_frozen": True,
            "wrong_event_challenge_parent_event_count": int(
                wrong_event_attestation_challenge_plan["positive_parent_event_count"]
            ),
            "wrong_event_challenge_available_assignment_count": int(
                wrong_event_attestation_challenge_plan["available_assignment_count"]
            ),
            "geometry_optional_claim_plan_frozen": True,
            "geometry_optional_claim_mode": GEOMETRY_OPTIONAL_CLAIM_MODE,
            "geometry_optional_claim_scope": GEOMETRY_OPTIONAL_CLAIM_SCOPE,
            "geometry_optional_claim_available_assignment_count": int(
                geometry_optional_claim_plan["available_assignment_count"]
            ),
            "severity_metadata_frozen": True,
            "severity_rule_version": ATTACK_SEVERITY_RULE_VERSION,
            "severity_axis_kind": ATTACK_SEVERITY_AXIS_KIND,
            "severity_status_counts": dict(sorted(severity_status_counts.items())),
            "severity_available_family_count": len(severity_family_level_counts),
            "severity_multi_point_family_count": sum(
                1 for count in severity_family_level_counts.values() if count > 1
            ),
        },
        "counts": {
            "positive_source_event_count": positive_source_event_count,
            "clean_negative_event_count": clean_negative_event_count,
            "planner_conditioned_control_negative_event_count": control_negative_event_count,
            "wrong_event_challenge_parent_event_count": int(
                wrong_event_attestation_challenge_plan["positive_parent_event_count"]
            ),
            "attack_condition_count": attack_condition_count,
            "attack_event_count": attack_event_count,
            "attacked_positive_event_count": attacked_positive_event_count,
            "attacked_negative_event_count": attacked_negative_event_count,
            "calibration_event_count": len(source_split_plan["calib_pos_event_ids"]) + len(source_split_plan["calib_neg_event_ids"]),
            "evaluate_event_count": len(source_split_plan["eval_pos_event_ids"]) + len(source_split_plan["eval_neg_event_ids"]),
            "control_calibration_event_count": len(source_split_plan["calib_control_event_ids"]),
            "control_evaluate_event_count": len(source_split_plan["eval_control_event_ids"]),
            "total_event_count": len(event_grid),
        },
        "paths": {
            "paper_eval_family_manifest": normalize_path_value(layout["family_manifest_path"]),
            "source_event_grid": normalize_path_value(layout["source_event_grid_path"]),
            "source_shard_plan": normalize_path_value(layout["source_shard_plan_path"]),
            "source_split_plan": normalize_path_value(layout["source_split_plan_path"]),
            "attack_event_grid": normalize_path_value(attack_event_grid_path),
            "attack_shard_plan": normalize_path_value(attack_shard_plan_path),
            "wrong_event_attestation_challenge_plan": normalize_path_value(
                wrong_event_attestation_challenge_plan_path
            ),
            "geometry_optional_claim_plan": normalize_path_value(geometry_optional_claim_plan_path),
            "prompt_snapshot": normalize_path_value(layout["prompt_snapshot_path"]),
            "method_identity_snapshot": normalize_path_value(layout["method_identity_snapshot_path"]),
            "config_snapshot": normalize_path_value(layout["config_snapshot_path"]),
        },
        "source_split": source_split_plan,
        "attack_plan": {
            "materialization_profile": "first_value_per_condition",
            "attack_shard_count": resolved_attack_shard_count,
            "attack_condition_count": attack_condition_count,
            "attack_event_count": attack_event_count,
            "attacked_positive_event_count": attacked_positive_event_count,
            "attacked_negative_event_count": attacked_negative_event_count,
            "wrong_event_attestation_challenge_plan_frozen": True,
            "wrong_event_attestation_challenge_plan_policy": str(
                wrong_event_attestation_challenge_plan["plan_policy"]
            ),
            "wrong_event_challenge_parent_event_count": int(
                wrong_event_attestation_challenge_plan["positive_parent_event_count"]
            ),
            "wrong_event_challenge_available_assignment_count": int(
                wrong_event_attestation_challenge_plan["available_assignment_count"]
            ),
            "geometry_optional_claim_plan_frozen": True,
            "geometry_optional_claim_mode": GEOMETRY_OPTIONAL_CLAIM_MODE,
            "geometry_optional_claim_scope": GEOMETRY_OPTIONAL_CLAIM_SCOPE,
            "geometry_optional_claim_available_assignment_count": int(
                geometry_optional_claim_plan["available_assignment_count"]
            ),
            "severity_metadata_frozen": True,
            "severity_rule_version": ATTACK_SEVERITY_RULE_VERSION,
            "severity_axis_kind": ATTACK_SEVERITY_AXIS_KIND,
            "severity_status_counts": dict(sorted(severity_status_counts.items())),
            "severity_available_family_count": len(severity_family_level_counts),
            "severity_multi_point_family_count": sum(
                1 for count in severity_family_level_counts.values() if count > 1
            ),
        },
        "default_config_path": normalize_path_value(default_cfg_path),
        "pw_base_config_path": normalize_path_value((REPO_ROOT / DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH).resolve()),
    }

    layout["prompt_snapshot_path"].write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    write_json_atomic(layout["method_identity_snapshot_path"], method_identity_snapshot)
    write_yaml_mapping(layout["config_snapshot_path"], config_snapshot)
    write_jsonl(layout["source_event_grid_path"], event_grid)
    write_json_atomic(layout["source_shard_plan_path"], source_shard_plan)
    write_json_atomic(layout["source_split_plan_path"], source_split_plan)
    write_jsonl(attack_event_grid_path, attack_event_grid)
    write_json_atomic(attack_shard_plan_path, attack_shard_plan)
    write_json_atomic(
        wrong_event_attestation_challenge_plan_path,
        wrong_event_attestation_challenge_plan,
    )
    write_json_atomic(
        geometry_optional_claim_plan_path,
        geometry_optional_claim_plan,
    )
    write_json_atomic(layout["family_manifest_path"], family_manifest)

    summary_path = layout["runtime_state_root"] / "pw00_summary.json"
    summary: Dict[str, Any] = {
        "status": "ok",
        "stage_name": "PW00_Paper_Eval_Family_Manifest",
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "summary_path": normalize_path_value(summary_path),
        "paper_eval_family_manifest_path": normalize_path_value(layout["family_manifest_path"]),
        "source_event_grid_path": normalize_path_value(layout["source_event_grid_path"]),
        "source_shard_plan_path": normalize_path_value(layout["source_shard_plan_path"]),
        "source_split_plan_path": normalize_path_value(layout["source_split_plan_path"]),
        "attack_event_grid_path": normalize_path_value(attack_event_grid_path),
        "attack_shard_plan_path": normalize_path_value(attack_shard_plan_path),
        "wrong_event_attestation_challenge_plan_path": normalize_path_value(
            wrong_event_attestation_challenge_plan_path
        ),
        "geometry_optional_claim_plan_path": normalize_path_value(geometry_optional_claim_plan_path),
        "event_count": len(event_grid),
        "attack_condition_count": attack_condition_count,
        "attack_event_count": attack_event_count,
        "source_shard_count": source_shard_count,
        "attack_shard_count": resolved_attack_shard_count,
        "wrong_event_challenge_parent_event_count": int(
            wrong_event_attestation_challenge_plan["positive_parent_event_count"]
        ),
        "wrong_event_challenge_available_assignment_count": int(
            wrong_event_attestation_challenge_plan["available_assignment_count"]
        ),
        "geometry_optional_claim_available_assignment_count": int(
            geometry_optional_claim_plan["available_assignment_count"]
        ),
        "severity_metadata_frozen": True,
        "severity_rule_version": ATTACK_SEVERITY_RULE_VERSION,
        "severity_axis_kind": ATTACK_SEVERITY_AXIS_KIND,
        "severity_status_counts": dict(sorted(severity_status_counts.items())),
        "severity_available_family_count": len(severity_family_level_counts),
        "severity_multi_point_family_count": sum(
            1 for count in severity_family_level_counts.values() if count > 1
        ),
    }
    write_json_atomic(summary_path, summary)
    return summary
