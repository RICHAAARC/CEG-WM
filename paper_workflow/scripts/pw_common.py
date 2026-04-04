"""
File purpose: Shared utilities for paper_workflow stage-01 orchestration.
Module type: General module
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    collect_attestation_env_summary,
    compute_file_sha256,
    ensure_directory,
    load_yaml_mapping,
    normalize_path_value,
    validate_path_within_base,
)

PAPER_WORKFLOW_ROOT_RELATIVE = "paper_workflow/families"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/default.yaml"
DEFAULT_PW_BASE_CONFIG_RELATIVE_PATH = "paper_workflow/configs/pw_base.yaml"
ACTIVE_SAMPLE_ROLE = "positive_source"
CLEAN_NEGATIVE_SAMPLE_ROLE = "clean_negative"
PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE = "planner_conditioned_control_negative"
ATTACKED_POSITIVE_SAMPLE_ROLE = "attacked_positive"
ACTIVE_SOURCE_SAMPLE_ROLES = [
    ACTIVE_SAMPLE_ROLE,
    CLEAN_NEGATIVE_SAMPLE_ROLE,
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE,
]
RESERVED_SAMPLE_ROLES = [ATTACKED_POSITIVE_SAMPLE_ROLE]
SAMPLE_ROLE_DIRECTORY_NAMES = {
    ACTIVE_SAMPLE_ROLE: "positive",
    CLEAN_NEGATIVE_SAMPLE_ROLE: "negative",
    PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE: "control_negative",
}
SOURCE_TRUTH_STAGE = "01_Paper_Full_Cuda_Parallel"


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    """
    Return a canonical JSON text representation.

    Args:
        payload: Mapping payload.

    Returns:
        Canonical JSON text.
    """
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    """
    Compute SHA256 for a canonical JSON mapping.

    Args:
        payload: Mapping payload.

    Returns:
        Lowercase SHA256 digest.
    """
    canonical_text = _canonical_json_text(payload)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def build_family_root(drive_project_root: Path, family_id: str) -> Path:
    """
    Build the paper workflow family root path.

    Args:
        drive_project_root: Drive project root.
        family_id: Family identifier.

    Returns:
        Family root path.
    """
    if not family_id.strip():
        raise TypeError("family_id must be non-empty str")

    family_root = drive_project_root / "paper_workflow" / "families" / family_id.strip()
    validate_path_within_base(drive_project_root, family_root, "paper workflow family root")
    return family_root


def resolve_family_layout_paths(family_root: Path) -> Dict[str, Path]:
    """
    Resolve canonical layout paths for one family root.

    Args:
        family_root: Family root path.

    Returns:
        Mapping of canonical directories and files.
    """
    manifests_root = family_root / "manifests"
    snapshots_root = family_root / "snapshots"
    source_shards_root = family_root / "source_shards"
    positive_shards_root = source_shards_root / "positive"
    negative_shards_root = source_shards_root / "negative"
    control_negative_shards_root = source_shards_root / "control_negative"
    return {
        "family_root": family_root,
        "manifests_root": manifests_root,
        "snapshots_root": snapshots_root,
        "source_shards_root": source_shards_root,
        "positive_shards_root": positive_shards_root,
        "negative_shards_root": negative_shards_root,
        "control_negative_shards_root": control_negative_shards_root,
        "runs_root": family_root / "runs",
        "logs_root": family_root / "logs",
        "runtime_state_root": family_root / "runtime_state",
        "exports_root": family_root / "exports",
        "family_manifest_path": manifests_root / "paper_eval_family_manifest.json",
        "source_event_grid_path": manifests_root / "source_event_grid.jsonl",
        "source_shard_plan_path": manifests_root / "source_shard_plan.json",
        "source_split_plan_path": manifests_root / "source_split_plan.json",
        "prompt_snapshot_path": snapshots_root / "prompt_snapshot.txt",
        "method_identity_snapshot_path": snapshots_root / "method_identity_snapshot.json",
        "config_snapshot_path": snapshots_root / "config_snapshot.yaml",
    }


def ensure_family_layout(family_root: Path) -> Dict[str, Path]:
    """
    Create the required family layout directories.

    Args:
        family_root: Family root path.

    Returns:
        Canonical layout path mapping.
    """
    layout = resolve_family_layout_paths(family_root)
    for key_name in [
        "family_root",
        "manifests_root",
        "snapshots_root",
        "positive_shards_root",
        "negative_shards_root",
        "control_negative_shards_root",
        "runs_root",
        "logs_root",
        "runtime_state_root",
        "exports_root",
    ]:
        ensure_directory(layout[key_name])
    return layout


def validate_source_sample_role(sample_role: str) -> str:
    """
    Validate one supported source sample role.

    Args:
        sample_role: Candidate source sample role.

    Returns:
        Normalized sample role.

    Raises:
        ValueError: If the role is not supported by PW00/PW01/PW02 source flow.
    """
    if not isinstance(sample_role, str) or not sample_role.strip():
        raise TypeError("sample_role must be non-empty str")

    normalized_role = sample_role.strip()
    if normalized_role not in ACTIVE_SOURCE_SAMPLE_ROLES:
        raise ValueError(f"unsupported sample_role: {normalized_role}")
    return normalized_role


def resolve_sample_role_directory_name(sample_role: str) -> str:
    """
    Resolve the directory token used by one source sample role.

    Args:
        sample_role: Supported source sample role.

    Returns:
        Directory token under source_shards/.
    """
    normalized_role = validate_source_sample_role(sample_role)
    return SAMPLE_ROLE_DIRECTORY_NAMES[normalized_role]


def build_source_shard_root(family_root: Path, sample_role: str, shard_index: int) -> Path:
    """
    Build one role-aware source shard root path.

    Args:
        family_root: Family root path.
        sample_role: Supported source sample role.
        shard_index: Zero-based shard index.

    Returns:
        Role-aware shard root path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise TypeError("shard_index must be non-negative int")

    role_directory_name = resolve_sample_role_directory_name(sample_role)
    shard_root = family_root / "source_shards" / role_directory_name / f"shard_{shard_index:04d}"
    validate_path_within_base(family_root, shard_root, "paper workflow source shard root")
    return shard_root


def parse_seed_list(seed_list_value: object) -> List[int]:
    """
    Parse and normalize the seed list.

    Args:
        seed_list_value: Seed list value from CLI or config.

    Returns:
        Normalized integer seed list.

    Raises:
        ValueError: If the parsed seed list is empty.
    """
    raw_values: List[object]
    if isinstance(seed_list_value, str):
        seed_text = seed_list_value.strip()
        if not seed_text:
            raise ValueError("seed_list must be non-empty")
        if seed_text.startswith("["):
            parsed = json.loads(seed_text)
            if not isinstance(parsed, list):
                raise TypeError("seed_list JSON must be list")
            raw_values = list(cast(List[object], parsed))
        else:
            raw_values = [item.strip() for item in seed_text.split(",") if item.strip()]
    elif isinstance(seed_list_value, Sequence):
        raw_values = list(cast(Sequence[object], seed_list_value))
    else:
        raise TypeError("seed_list must be str or Sequence")

    normalized: List[int] = []
    for raw_item in raw_values:
        if isinstance(raw_item, bool):
            raise TypeError("seed values must be int, bool is not allowed")
        if isinstance(raw_item, int):
            normalized.append(int(raw_item))
            continue
        if isinstance(raw_item, str) and raw_item.strip():
            normalized.append(int(raw_item.strip()))
            continue
        raise TypeError(f"invalid seed value: {raw_item!r}")

    if not normalized:
        raise ValueError("seed_list must contain at least one seed")
    return normalized


def resolve_prompt_file_path(prompt_file: str, repo_root: Path = REPO_ROOT) -> Path:
    """
    Resolve prompt file path against repository root.

    Args:
        prompt_file: Prompt file path value.
        repo_root: Repository root path.

    Returns:
        Resolved prompt file path.
    """
    if not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")

    candidate = Path(prompt_file.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def load_prompt_lines(prompt_file: str, repo_root: Path = REPO_ROOT) -> Tuple[Path, List[str]]:
    """
    Load non-empty prompt lines from prompt file.

    Args:
        prompt_file: Prompt file path value.
        repo_root: Repository root path.

    Returns:
        Tuple of resolved prompt path and prompt line list.

    Raises:
        FileNotFoundError: If prompt file does not exist.
        ValueError: If no non-empty prompt lines exist.
    """
    prompt_path = resolve_prompt_file_path(prompt_file, repo_root=repo_root)
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt file not found: {normalize_path_value(prompt_path)}")
    prompt_lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompt_lines:
        raise ValueError(f"prompt file has no non-empty lines: {normalize_path_value(prompt_path)}")
    return prompt_path, prompt_lines


def build_event_id(
    *,
    family_id: str,
    sample_role: str,
    prompt_index: int,
    prompt_sha256: str,
    seed: int,
) -> str:
    """
    Build deterministic event identifier.

    Args:
        family_id: Family identifier.
        sample_role: Sample role.
        prompt_index: Prompt index.
        prompt_sha256: Prompt SHA256 digest.
        seed: Generation seed.

    Returns:
        Stable event identifier.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not sample_role:
        raise TypeError("sample_role must be non-empty str")
    if prompt_index < 0:
        raise TypeError("prompt_index must be non-negative int")
    if not prompt_sha256:
        raise TypeError("prompt_sha256 must be non-empty str")

    digest_payload: Dict[str, Any] = {
        "family_id": family_id,
        "sample_role": sample_role,
        "prompt_index": prompt_index,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
    }
    return f"evt_{canonical_mapping_sha256(digest_payload)[:24]}"


def _build_prompt_sha256(prompt_text: str) -> str:
    """
    Compute prompt text SHA256.

    Args:
        prompt_text: Prompt text.

    Returns:
        Lowercase SHA256 digest.
    """
    if not prompt_text:
        raise TypeError("prompt_text must be non-empty str")
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def build_source_event_grid(
    *,
    family_id: str,
    prompt_lines: Sequence[str],
    seeds: Sequence[int],
    prompt_file: str,
    sample_roles: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Build deterministic multi-role source event grid.

    Args:
        family_id: Family identifier.
        prompt_lines: Prompt text sequence.
        seeds: Seed sequence.
        prompt_file: Prompt file path string.
        sample_roles: Ordered active source sample roles.

    Returns:
        Ordered event list across all requested source roles.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not prompt_file:
        raise TypeError("prompt_file must be non-empty str")
    if not prompt_lines:
        raise ValueError("prompt_lines must be non-empty sequence")
    if not seeds:
        raise ValueError("seeds must be non-empty sequence")
    if not sample_roles:
        raise ValueError("sample_roles must be non-empty sequence")

    normalized_roles: List[str] = []
    for raw_role in sample_roles:
        normalized_role = validate_source_sample_role(str(raw_role))
        if normalized_role not in normalized_roles:
            normalized_roles.append(normalized_role)

    events: List[Dict[str, Any]] = []
    event_index = 0
    for sample_role in normalized_roles:
        for prompt_index, prompt_text_value in enumerate(prompt_lines):
            prompt_text = str(prompt_text_value).strip()
            if not prompt_text:
                raise ValueError(f"prompt_lines contains empty prompt at index {prompt_index}")
            prompt_sha256 = _build_prompt_sha256(prompt_text)
            for seed_value in seeds:
                event_id = build_event_id(
                    family_id=family_id,
                    sample_role=sample_role,
                    prompt_index=prompt_index,
                    prompt_sha256=prompt_sha256,
                    seed=int(seed_value),
                )
                events.append(
                    {
                        "event_id": event_id,
                        "event_index": event_index,
                        "sample_role": sample_role,
                        "prompt_index": prompt_index,
                        "source_prompt_index": prompt_index,
                        "prompt_text": prompt_text,
                        "prompt_sha256": prompt_sha256,
                        "seed": int(seed_value),
                        "prompt_file": prompt_file,
                    }
                )
                event_index += 1
    return events


def build_positive_source_event_grid(
    *,
    family_id: str,
    prompt_lines: Sequence[str],
    seeds: Sequence[int],
    prompt_file: str,
) -> List[Dict[str, Any]]:
    """
    Build deterministic positive_source event grid.

    Args:
        family_id: Family identifier.
        prompt_lines: Prompt text sequence.
        seeds: Seed sequence.
        prompt_file: Prompt file path string.

    Returns:
        Ordered event list.
    """
    return build_source_event_grid(
        family_id=family_id,
        prompt_lines=prompt_lines,
        seeds=seeds,
        prompt_file=prompt_file,
        sample_roles=[ACTIVE_SAMPLE_ROLE],
    )


def build_source_split_plan(
    *,
    family_id: str,
    events: Sequence[Mapping[str, Any]],
    calibration_fraction: float,
) -> Dict[str, Any]:
    """
    Build deterministic calibration/evaluate split plan for source roles.

    Args:
        family_id: Family identifier.
        events: Full source event grid.
        calibration_fraction: Fraction assigned to calibration within each role.

    Returns:
        Split-plan payload with role-specific and flattened event-id lists.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(calibration_fraction, (int, float)):
        raise TypeError("calibration_fraction must be float")

    calibration_fraction_value = float(calibration_fraction)
    if not 0.0 < calibration_fraction_value < 1.0:
        raise ValueError("calibration_fraction must satisfy 0 < calibration_fraction < 1")

    events_by_role: Dict[str, List[Dict[str, Any]]] = {
        sample_role: []
        for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES
    }
    for event_node in events:
        if not isinstance(event_node, Mapping):
            raise TypeError("events must contain mappings")
        event = dict(cast(Mapping[str, Any], event_node))
        sample_role = validate_source_sample_role(str(event.get("sample_role")))
        event_index = event.get("event_index")
        event_id = event.get("event_id")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be non-negative int")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty str")
        events_by_role[sample_role].append(event)

    split_roles: Dict[str, Dict[str, Any]] = {}
    for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES:
        ordered_events = sorted(events_by_role[sample_role], key=lambda item: int(item["event_index"]))
        if not ordered_events:
            raise ValueError(f"source split requires non-empty events for role: {sample_role}")

        calibration_count = int(len(ordered_events) * calibration_fraction_value)
        if calibration_count <= 0 or calibration_count >= len(ordered_events):
            raise ValueError(
                "source split requires non-empty calibration and evaluate partitions: "
                f"sample_role={sample_role}, event_count={len(ordered_events)}, calibration_fraction={calibration_fraction_value}"
            )

        calibration_events = ordered_events[:calibration_count]
        evaluate_events = ordered_events[calibration_count:]
        split_roles[sample_role] = {
            "event_count": len(ordered_events),
            "calibration_event_count": len(calibration_events),
            "evaluate_event_count": len(evaluate_events),
            "calibration_event_ids": [str(event["event_id"]) for event in calibration_events],
            "evaluate_event_ids": [str(event["event_id"]) for event in evaluate_events],
        }

    return {
        "artifact_type": "paper_workflow_source_split_plan",
        "schema_version": "pw_stage_02_v1",
        "family_id": family_id,
        "calibration_fraction": calibration_fraction_value,
        "sample_roles_active": list(ACTIVE_SOURCE_SAMPLE_ROLES),
        "sample_roles_reserved": list(RESERVED_SAMPLE_ROLES),
        "roles": split_roles,
        "calib_pos_event_ids": list(split_roles[ACTIVE_SAMPLE_ROLE]["calibration_event_ids"]),
        "calib_neg_event_ids": list(split_roles[CLEAN_NEGATIVE_SAMPLE_ROLE]["calibration_event_ids"]),
        "calib_control_event_ids": list(
            split_roles[PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE]["calibration_event_ids"]
        ),
        "eval_pos_event_ids": list(split_roles[ACTIVE_SAMPLE_ROLE]["evaluate_event_ids"]),
        "eval_neg_event_ids": list(split_roles[CLEAN_NEGATIVE_SAMPLE_ROLE]["evaluate_event_ids"]),
        "eval_control_event_ids": list(
            split_roles[PLANNER_CONDITIONED_CONTROL_NEGATIVE_SAMPLE_ROLE]["evaluate_event_ids"]
        ),
    }


def build_source_shard_plan(
    *,
    family_id: str,
    source_shard_count: int,
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build deterministic source shard plan from event grid.

    Args:
        family_id: Family identifier.
        source_shard_count: Total source shard count.
        events: Event grid entries.

    Returns:
        Source shard plan payload.
    """
    if not family_id:
        raise TypeError("family_id must be non-empty str")
    if source_shard_count <= 0:
        raise TypeError("source_shard_count must be positive int")

    role_events: Dict[str, List[Dict[str, Any]]] = {
        sample_role: []
        for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES
    }
    for event_node in events:
        if not isinstance(event_node, Mapping):
            raise TypeError("events must contain mappings")
        event = dict(cast(Mapping[str, Any], event_node))
        sample_role = validate_source_sample_role(str(event.get("sample_role")))
        event_id = event.get("event_id")
        event_index = event.get("event_index")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty str")
        if not isinstance(event_index, int) or event_index < 0:
            raise ValueError("event_index must be non-negative int")
        role_events[sample_role].append(event)

    sample_role_plans: Dict[str, Dict[str, Any]] = {}
    for sample_role in ACTIVE_SOURCE_SAMPLE_ROLES:
        ordered_role_events = sorted(role_events[sample_role], key=lambda item: int(item["event_index"]))
        shard_rows: List[Dict[str, Any]] = [
            {
                "shard_index": shard_index,
                "sample_role": sample_role,
                "assigned_event_ids": [],
                "assigned_event_indices": [],
            }
            for shard_index in range(source_shard_count)
        ]
        for role_event_ordinal, event in enumerate(ordered_role_events):
            shard_index = role_event_ordinal % source_shard_count
            shard_rows[shard_index]["assigned_event_ids"].append(str(event["event_id"]))
            shard_rows[shard_index]["assigned_event_indices"].append(int(event["event_index"]))
        sample_role_plans[sample_role] = {
            "event_count": len(ordered_role_events),
            "shards": shard_rows,
        }

    return {
        "artifact_type": "paper_workflow_source_shard_plan",
        "schema_version": "pw_stage_01_v1",
        "family_id": family_id,
        "source_shard_count": source_shard_count,
        "sample_roles_active": list(ACTIVE_SOURCE_SAMPLE_ROLES),
        "sample_roles_reserved": list(RESERVED_SAMPLE_ROLES),
        "sample_role_plans": {
            **sample_role_plans,
            ATTACKED_POSITIVE_SAMPLE_ROLE: {
                "event_count": 0,
                "shards": [],
            },
        },
        "shards": list(sample_role_plans[ACTIVE_SAMPLE_ROLE]["shards"]),
    }


def write_jsonl(path_obj: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Write JSONL rows to file.

    Args:
        path_obj: Output JSONL path.
        rows: JSON-serializable row mappings.

    Returns:
        None.
    """
    ensure_directory(path_obj.parent)
    lines: List[str] = []
    for row in rows:
        lines.append(_canonical_json_text(row))
    path_obj.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path_obj: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL rows from file.

    Args:
        path_obj: Input JSONL path.

    Returns:
        Parsed JSON object rows.
    """
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"jsonl file not found: {normalize_path_value(path_obj)}")

    rows: List[Dict[str, Any]] = []
    for line_index, line in enumerate(path_obj.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"jsonl row must be object, line={line_index}")
        rows.append(cast(Dict[str, Any], parsed))
    return rows


def load_default_config_snapshot(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """
    Load the root default config snapshot.

    Args:
        repo_root: Repository root path.

    Returns:
        Parsed default config mapping.
    """
    return load_yaml_mapping((repo_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve())


def build_method_identity_snapshot(
    *,
    default_cfg_obj: Mapping[str, Any],
    default_cfg_path: Path,
    source_alignment_reference_files: Sequence[str],
) -> Dict[str, Any]:
    """
    Build method identity snapshot for family manifest lineage.

    Args:
        default_cfg_obj: Parsed default config object.
        default_cfg_path: Root default config path.
        source_alignment_reference_files: Source truth reference files.

    Returns:
        Method identity snapshot payload.
    """
    paper_faithfulness_node = default_cfg_obj.get("paper_faithfulness")
    paper_faithfulness_cfg: Mapping[str, Any]
    if isinstance(paper_faithfulness_node, Mapping):
        paper_faithfulness_cfg = cast(Mapping[str, Any], paper_faithfulness_node)
    else:
        paper_faithfulness_cfg = cast(Mapping[str, Any], {})
    attestation_node = default_cfg_obj.get("attestation")
    attestation_cfg: Mapping[str, Any]
    if isinstance(attestation_node, Mapping):
        attestation_cfg = cast(Mapping[str, Any], attestation_node)
    else:
        attestation_cfg = cast(Mapping[str, Any], {})

    attestation_env_summary = collect_attestation_env_summary(default_cfg_obj)
    secret_fingerprint = canonical_mapping_sha256(attestation_env_summary)

    return {
        "source_truth_stage": SOURCE_TRUTH_STAGE,
        "default_config_path": normalize_path_value(default_cfg_path),
        "default_config_sha256": compute_file_sha256(default_cfg_path),
        "policy_path": default_cfg_obj.get("policy_path"),
        "paper_faithfulness": {
            "enabled": paper_faithfulness_cfg.get("enabled"),
        },
        "attestation": {
            "enabled": attestation_cfg.get("enabled"),
            "decision_mode": attestation_cfg.get("decision_mode"),
            "use_trajectory_mix": attestation_cfg.get("use_trajectory_mix"),
        },
        "pipeline_impl_id": default_cfg_obj.get("pipeline_impl_id"),
        "model_id": default_cfg_obj.get("model_id"),
        "hf_revision": default_cfg_obj.get("hf_revision"),
        "secret_fingerprint": secret_fingerprint,
        "secret_fingerprint_basis": {
            "required_env_vars": attestation_env_summary.get("required_env_vars", []),
            "present_env_vars": attestation_env_summary.get("present_env_vars", []),
            "missing_env_vars": attestation_env_summary.get("missing_env_vars", []),
        },
        "source_alignment_reference_files": [str(item) for item in source_alignment_reference_files],
    }
