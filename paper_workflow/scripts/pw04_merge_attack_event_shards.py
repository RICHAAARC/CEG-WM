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
    ATTACKED_NEGATIVE_SAMPLE_ROLE,
    ATTACKED_POSITIVE_SAMPLE_ROLE,
    MIXED_ATTACK_SAMPLE_ROLE,
    build_family_root,
    read_jsonl,
    write_jsonl,
)
from paper_workflow.scripts.pw04_finalize_quality_metrics import run_pw04_finalize_quality_metrics
from paper_workflow.scripts.pw04_metrics_extensions import build_pw04_metrics_extensions
from paper_workflow.scripts.pw04_prepare_quality_pairs import run_pw04_prepare_quality_pairs
from paper_workflow.scripts.pw_quality_metrics import build_quality_metrics_from_pairs
from paper_workflow.scripts.pw04_paper_exports import build_pw04_paper_exports
from paper_workflow.scripts.pw04_run_quality_shard import run_pw04_quality_shard
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
ATTACK_NEGATIVE_POOL_MANIFEST_FILE_NAME = "attack_negative_pool_manifest.json"
FORMAL_ATTACK_FINAL_DECISION_METRICS_FILE_NAME = "formal_attack_final_decision_metrics.json"
FORMAL_ATTACK_ATTESTATION_METRICS_FILE_NAME = "formal_attack_attestation_metrics.json"
DERIVED_ATTACK_UNION_METRICS_FILE_NAME = "derived_attack_union_metrics.json"
FORMAL_ATTACK_NEGATIVE_METRICS_FILE_NAME = "formal_attack_negative_metrics.json"
PER_ATTACK_FAMILY_METRICS_FILE_NAME = "per_attack_family_metrics.json"
PER_ATTACK_CONDITION_METRICS_FILE_NAME = "per_attack_condition_metrics.json"
ATTACK_EVENT_TABLE_FILE_NAME = "attack_event_table.jsonl"
ATTACK_FAMILY_SUMMARY_CSV_FILE_NAME = "attack_family_summary.csv"
ATTACK_CONDITION_SUMMARY_CSV_FILE_NAME = "attack_condition_summary.csv"
CLEAN_ATTACK_OVERVIEW_FILE_NAME = "clean_attack_overview.json"
QUALITY_ROOT_DIRECTORY_NAME = "quality"
QUALITY_FINALIZE_MANIFEST_FILE_NAME = "quality_finalize_manifest.json"
QUALITY_PAIR_PLAN_FILE_NAME = "quality_pair_plan.json"
QUALITY_SHARDS_DIRECTORY_NAME = "shards"
QUALITY_SHARD_FILE_NAME_TEMPLATE = "quality_shard_{quality_shard_index:04d}.json"
QUALITY_SHARD_FILE_NAME_FIELD = "quality_shard_index"
QUALITY_SHARD_PATHS_FIELD = "quality_shard_paths"
QUALITY_SHARD_COUNT_FIELD = "quality_shard_count"
QUALITY_PAIR_PLAN_PATH_FIELD = "quality_pair_plan_path"
QUALITY_FINALIZE_MANIFEST_PATH_FIELD = "quality_finalize_manifest_path"
QUALITY_ROOT_PATH_FIELD = "quality_root"
QUALITY_ROOT_SUMMARY_FIELD = "quality_root"
QUALITY_ROOT_EXPORT_FIELD = "quality_root"
QUALITY_ROOT_LABEL = "PW04 quality root"
QUALITY_PATH_LABEL = "PW04 quality output path"
ATTACK_QUALITY_METRICS_FILE_NAME = "attack_quality_metrics.json"
CLEAN_QUALITY_METRICS_FILE_NAME = "clean_quality_metrics.json"
PW04_METRICS_DIRECTORY_NAME = "metrics"
PW04_FIGURES_DIRECTORY_NAME = "figures"
PW04_TAIL_DIRECTORY_NAME = "tail"
PW04_MODE_PREPARE = "prepare"
PW04_MODE_QUALITY_SHARD = "quality_shard"
PW04_MODE_FINALIZE = "finalize"
PW04_MODE_CHOICES = (
    PW04_MODE_PREPARE,
    PW04_MODE_QUALITY_SHARD,
    PW04_MODE_FINALIZE,
)
PREPARE_MANIFEST_FILE_NAME = "pw04_prepare_manifest.json"
PREPARED_ATTACK_EVENT_ROWS_FILE_NAME = "prepared_attack_event_rows.json"


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


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    Normalize one optional mapping node to dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Normalized dict payload.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


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


def _extract_prompt_text_from_mapping(payload: Mapping[str, Any] | None) -> str | None:
    """
    Extract prompt text from one attack-side metadata mapping when available.

    Args:
        payload: Candidate metadata mapping.

    Returns:
        Prompt text or None.
    """
    if not isinstance(payload, Mapping):
        return None
    for key_name in ["prompt_text", "prompt", "inference_prompt"]:
        prompt_value = payload.get(key_name)
        if isinstance(prompt_value, str) and prompt_value.strip():
            return prompt_value.strip()
    return None


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
    metrics_root = export_root / PW04_METRICS_DIRECTORY_NAME
    figures_root = export_root / PW04_FIGURES_DIRECTORY_NAME
    tail_root = export_root / PW04_TAIL_DIRECTORY_NAME
    quality_root = export_root / QUALITY_ROOT_DIRECTORY_NAME
    summary_path = family_root / "runtime_state" / PW04_SUMMARY_FILE_NAME
    paths = {
        "export_root": export_root,
        "manifests_root": manifests_root,
        "records_root": records_root,
        "tables_root": tables_root,
        "metrics_root": metrics_root,
        "figures_root": figures_root,
        "tail_root": tail_root,
        "quality_root": quality_root,
        "summary_path": summary_path,
        "prepare_manifest_path": manifests_root / PREPARE_MANIFEST_FILE_NAME,
        "prepared_attack_event_rows_path": manifests_root / PREPARED_ATTACK_EVENT_ROWS_FILE_NAME,
        "attack_merge_manifest_path": manifests_root / ATTACK_MERGE_MANIFEST_FILE_NAME,
        "attack_pool_manifest_path": export_root / ATTACK_POOL_MANIFEST_FILE_NAME,
        "attack_negative_pool_manifest_path": export_root / ATTACK_NEGATIVE_POOL_MANIFEST_FILE_NAME,
        "formal_attack_final_decision_metrics_path": export_root / FORMAL_ATTACK_FINAL_DECISION_METRICS_FILE_NAME,
        "formal_attack_attestation_metrics_path": export_root / FORMAL_ATTACK_ATTESTATION_METRICS_FILE_NAME,
        "derived_attack_union_metrics_path": export_root / DERIVED_ATTACK_UNION_METRICS_FILE_NAME,
        "formal_attack_negative_metrics_path": export_root / FORMAL_ATTACK_NEGATIVE_METRICS_FILE_NAME,
        "per_attack_family_metrics_path": export_root / PER_ATTACK_FAMILY_METRICS_FILE_NAME,
        "per_attack_condition_metrics_path": export_root / PER_ATTACK_CONDITION_METRICS_FILE_NAME,
        "attack_event_table_path": tables_root / ATTACK_EVENT_TABLE_FILE_NAME,
        "attack_family_summary_csv_path": tables_root / ATTACK_FAMILY_SUMMARY_CSV_FILE_NAME,
        "attack_condition_summary_csv_path": tables_root / ATTACK_CONDITION_SUMMARY_CSV_FILE_NAME,
        "clean_attack_overview_path": export_root / CLEAN_ATTACK_OVERVIEW_FILE_NAME,
        "clean_quality_metrics_path": metrics_root / CLEAN_QUALITY_METRICS_FILE_NAME,
        "attack_quality_metrics_path": metrics_root / ATTACK_QUALITY_METRICS_FILE_NAME,
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


def _resolve_pw04_mode(pw04_mode: str) -> str:
    """
    Resolve one validated PW04 execution mode.

    Args:
        pw04_mode: Candidate mode token.

    Returns:
        Normalized PW04 mode token.
    """
    if not isinstance(pw04_mode, str) or not pw04_mode.strip():
        raise TypeError("pw04_mode must be non-empty str")
    resolved_mode = pw04_mode.strip().lower()
    if resolved_mode not in PW04_MODE_CHOICES:
        raise ValueError(
            f"unsupported pw04_mode: {pw04_mode}; expected one of {', '.join(PW04_MODE_CHOICES)}"
        )
    return resolved_mode


def _resolve_manifest_bound_path(
    *,
    manifest_payload: Mapping[str, Any],
    family_root: Path,
    field_name: str,
    label: str,
) -> Path:
    """
    Resolve one family-root-constrained path from a manifest payload.

    Args:
        manifest_payload: Source manifest mapping.
        family_root: Family root path.
        field_name: Manifest field name.
        label: Human-readable label.

    Returns:
        Resolved file path.
    """
    if not isinstance(manifest_payload, Mapping):
        raise TypeError("manifest_payload must be Mapping")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("field_name must be non-empty str")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")

    path_value = manifest_payload.get(field_name)
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{field_name} missing from manifest")
    resolved_path = Path(path_value).expanduser().resolve()
    validate_path_within_base(family_root, resolved_path, label)
    return resolved_path


def _build_expected_quality_shard_paths(
    *,
    family_root: Path,
    quality_root: Path,
    quality_pair_plan: Mapping[str, Any],
) -> List[Path]:
    """
    Build the ordered expected quality shard paths from the quality plan.

    Args:
        family_root: Family root path.
        quality_root: PW04 quality root.
        quality_pair_plan: Prepared quality pair plan payload.

    Returns:
        Ordered expected shard paths.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(quality_root, Path):
        raise TypeError("quality_root must be Path")
    if not isinstance(quality_pair_plan, Mapping):
        raise TypeError("quality_pair_plan must be Mapping")

    shard_nodes = quality_pair_plan.get("shards")
    if not isinstance(shard_nodes, list):
        raise ValueError("PW04 quality pair plan missing shards")

    seen_indices: set[int] = set()
    expected_paths: List[Path] = []
    for shard_node in cast(List[Any], shard_nodes):
        if not isinstance(shard_node, Mapping):
            raise ValueError("PW04 quality pair plan shards must contain objects")
        quality_shard_index = shard_node.get(QUALITY_SHARD_FILE_NAME_FIELD)
        if not isinstance(quality_shard_index, int) or isinstance(quality_shard_index, bool) or quality_shard_index < 0:
            raise ValueError("PW04 quality pair plan shard missing quality_shard_index")
        if quality_shard_index in seen_indices:
            raise ValueError(f"duplicate quality_shard_index in PW04 quality pair plan: {quality_shard_index}")
        seen_indices.add(quality_shard_index)
        shard_path = quality_root / QUALITY_SHARDS_DIRECTORY_NAME / QUALITY_SHARD_FILE_NAME_TEMPLATE.format(
            quality_shard_index=quality_shard_index
        )
        validate_path_within_base(family_root, shard_path, QUALITY_PATH_LABEL)
        expected_paths.append(shard_path)
    return expected_paths


def _build_prepared_attack_event_rows_payload(
    *,
    family_id: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the persisted prepared attack-event rows payload.

    Args:
        family_id: Family identifier.
        attack_event_rows: Prepared materialized attack-event rows.

    Returns:
        Prepared attack-event rows payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(attack_event_rows, Sequence):
        raise TypeError("attack_event_rows must be Sequence")

    return {
        "artifact_type": "paper_workflow_pw04_prepared_attack_event_rows",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "row_count": len(attack_event_rows),
        "rows": [dict(cast(Mapping[str, Any], row)) for row in attack_event_rows],
    }


def _build_prepare_manifest_payload(
    *,
    family_id: str,
    family_root: Path,
    pw04_paths: Mapping[str, Path],
    quality_pair_plan_path: Path,
    prepared_attack_event_rows_path: Path,
    expected_quality_shard_paths: Sequence[Path],
    expected_attack_shard_count: int,
    discovered_attack_shard_count: int,
    expected_attack_event_count: int,
    discovered_attack_event_count: int,
    completed_attack_event_count: int,
    attacked_positive_event_count: int,
    attacked_negative_event_count: int,
    enable_tail_estimation: bool,
) -> Dict[str, Any]:
    """
    Build the manifest that freezes the PW04 prepare-stage outputs.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw04_paths: Canonical PW04 path mapping.
        quality_pair_plan_path: Prepared quality pair plan path.
        prepared_attack_event_rows_path: Persisted prepared attack rows path.
        expected_quality_shard_paths: Ordered expected quality shard paths.
        expected_attack_shard_count: Expected attack shard count.
        discovered_attack_shard_count: Discovered attack shard count.
        expected_attack_event_count: Expected attack event count.
        discovered_attack_event_count: Discovered attack event count.
        completed_attack_event_count: Completed attack event count.
        attacked_positive_event_count: Attacked-positive event count.
        attacked_negative_event_count: Attacked-negative event count.
        enable_tail_estimation: Frozen tail-estimation request.

    Returns:
        Prepare manifest payload.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw04_paths, Mapping):
        raise TypeError("pw04_paths must be Mapping")
    if not isinstance(quality_pair_plan_path, Path):
        raise TypeError("quality_pair_plan_path must be Path")
    if not isinstance(prepared_attack_event_rows_path, Path):
        raise TypeError("prepared_attack_event_rows_path must be Path")
    if not isinstance(expected_quality_shard_paths, Sequence):
        raise TypeError("expected_quality_shard_paths must be Sequence")
    if not isinstance(enable_tail_estimation, bool):
        raise TypeError("enable_tail_estimation must be bool")

    return {
        "artifact_type": "paper_workflow_pw04_prepare_manifest",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "stage_name": STAGE_NAME,
        "pw04_mode": PW04_MODE_PREPARE,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "status": "completed",
        "enable_tail_estimation": enable_tail_estimation,
        "attack_merge_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_merge_manifest_path"])),
        "attack_positive_pool_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_pool_manifest_path"])),
        "attack_negative_pool_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_negative_pool_manifest_path"])),
        "formal_attack_final_decision_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"])),
        "formal_attack_attestation_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_attestation_metrics_path"])),
        "derived_attack_union_metrics_path": normalize_path_value(cast(Path, pw04_paths["derived_attack_union_metrics_path"])),
        "formal_attack_negative_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_negative_metrics_path"])),
        "per_attack_family_metrics_path": normalize_path_value(cast(Path, pw04_paths["per_attack_family_metrics_path"])),
        "per_attack_condition_metrics_path": normalize_path_value(cast(Path, pw04_paths["per_attack_condition_metrics_path"])),
        "summary_path": normalize_path_value(cast(Path, pw04_paths["summary_path"])),
        "prepare_manifest_path": normalize_path_value(cast(Path, pw04_paths["prepare_manifest_path"])),
        "quality_root": normalize_path_value(cast(Path, pw04_paths["quality_root"])),
        "quality_pair_plan_path": normalize_path_value(quality_pair_plan_path),
        "prepared_attack_event_rows_path": normalize_path_value(prepared_attack_event_rows_path),
        "clean_quality_metrics_path": normalize_path_value(cast(Path, pw04_paths["clean_quality_metrics_path"])),
        "attack_quality_metrics_path": normalize_path_value(cast(Path, pw04_paths["attack_quality_metrics_path"])),
        "quality_finalize_manifest_path": normalize_path_value(cast(Path, pw04_paths["quality_root"]) / QUALITY_FINALIZE_MANIFEST_FILE_NAME),
        "expected_quality_shard_paths": [normalize_path_value(path_obj) for path_obj in expected_quality_shard_paths],
        "quality_shard_count": len(expected_quality_shard_paths),
        "expected_attack_shard_count": expected_attack_shard_count,
        "discovered_attack_shard_count": discovered_attack_shard_count,
        "expected_attack_event_count": expected_attack_event_count,
        "discovered_attack_event_count": discovered_attack_event_count,
        "completed_attack_event_count": completed_attack_event_count,
        "attacked_positive_event_count": attacked_positive_event_count,
        "attacked_negative_event_count": attacked_negative_event_count,
    }


def _load_pw04_prepare_context(
    *,
    family_id: str,
    family_root: Path,
    pw04_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    """
    Load and validate the frozen PW04 prepare-stage context.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw04_paths: Canonical PW04 path mapping.

    Returns:
        Loaded prepare context.
    """
    if not isinstance(family_id, str) or not family_id:
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw04_paths, Mapping):
        raise TypeError("pw04_paths must be Mapping")

    prepare_manifest_path = cast(Path, pw04_paths["prepare_manifest_path"])
    prepare_manifest = _load_required_json_dict(prepare_manifest_path, "PW04 prepare manifest")
    if prepare_manifest.get("family_id") != family_id:
        raise ValueError(
            "PW04 prepare manifest family_id mismatch: "
            f"expected={family_id}, actual={prepare_manifest.get('family_id')}"
        )
    if prepare_manifest.get("status") != "completed":
        raise ValueError(
            "PW04 prepare manifest must be completed before worker/finalize: "
            f"status={prepare_manifest.get('status')}"
        )

    expected_bindings = {
        "attack_merge_manifest_path": "attack_merge_manifest_path",
        "attack_positive_pool_manifest_path": "attack_pool_manifest_path",
        "attack_negative_pool_manifest_path": "attack_negative_pool_manifest_path",
        "formal_attack_final_decision_metrics_path": "formal_attack_final_decision_metrics_path",
        "formal_attack_attestation_metrics_path": "formal_attack_attestation_metrics_path",
        "derived_attack_union_metrics_path": "derived_attack_union_metrics_path",
        "formal_attack_negative_metrics_path": "formal_attack_negative_metrics_path",
        "per_attack_family_metrics_path": "per_attack_family_metrics_path",
        "per_attack_condition_metrics_path": "per_attack_condition_metrics_path",
        "clean_quality_metrics_path": "clean_quality_metrics_path",
        "attack_quality_metrics_path": "attack_quality_metrics_path",
        "summary_path": "summary_path",
        "prepare_manifest_path": "prepare_manifest_path",
        "prepared_attack_event_rows_path": "prepared_attack_event_rows_path",
    }
    resolved_paths: Dict[str, Path] = {}
    for manifest_field_name, pw04_path_key in expected_bindings.items():
        resolved_path = _resolve_manifest_bound_path(
            manifest_payload=prepare_manifest,
            family_root=family_root,
            field_name=manifest_field_name,
            label=manifest_field_name,
        )
        expected_path = cast(Path, pw04_paths[pw04_path_key]).resolve()
        if normalize_path_value(resolved_path) != normalize_path_value(expected_path):
            raise ValueError(
                f"PW04 prepare manifest {manifest_field_name} mismatch: "
                f"expected={normalize_path_value(expected_path)}, actual={normalize_path_value(resolved_path)}"
            )
        resolved_paths[manifest_field_name] = resolved_path

    quality_pair_plan_path = _resolve_manifest_bound_path(
        manifest_payload=prepare_manifest,
        family_root=family_root,
        field_name=QUALITY_PAIR_PLAN_PATH_FIELD,
        label="PW04 quality pair plan",
    )
    prepared_attack_event_rows_path = resolved_paths["prepared_attack_event_rows_path"]
    quality_pair_plan = _load_required_json_dict(quality_pair_plan_path, "PW04 quality pair plan")
    if quality_pair_plan.get("family_id") != family_id:
        raise ValueError(
            "PW04 quality pair plan family_id mismatch: "
            f"expected={family_id}, actual={quality_pair_plan.get('family_id')}"
        )
    expected_quality_shard_paths = _build_expected_quality_shard_paths(
        family_root=family_root,
        quality_root=cast(Path, pw04_paths["quality_root"]),
        quality_pair_plan=quality_pair_plan,
    )

    expected_quality_shard_paths_node = prepare_manifest.get("expected_quality_shard_paths")
    if not isinstance(expected_quality_shard_paths_node, list):
        raise ValueError("PW04 prepare manifest missing expected_quality_shard_paths")
    manifest_quality_shard_paths = [
        _resolve_manifest_bound_path(
            manifest_payload={"path": path_value},
            family_root=family_root,
            field_name="path",
            label="PW04 expected quality shard path",
        )
        for path_value in cast(List[Any], expected_quality_shard_paths_node)
    ]
    if [normalize_path_value(path_obj) for path_obj in manifest_quality_shard_paths] != [
        normalize_path_value(path_obj) for path_obj in expected_quality_shard_paths
    ]:
        raise ValueError("PW04 prepare manifest expected_quality_shard_paths mismatch with quality pair plan")

    prepared_attack_event_rows_payload = _load_required_json_dict(
        prepared_attack_event_rows_path,
        "PW04 prepared attack event rows",
    )
    if prepared_attack_event_rows_payload.get("family_id") != family_id:
        raise ValueError(
            "PW04 prepared attack event rows family_id mismatch: "
            f"expected={family_id}, actual={prepared_attack_event_rows_payload.get('family_id')}"
        )
    prepared_rows_node = prepared_attack_event_rows_payload.get("rows")
    if not isinstance(prepared_rows_node, list):
        raise ValueError("PW04 prepared attack event rows missing rows")
    materialized_attack_event_rows = [
        dict(cast(Mapping[str, Any], row))
        for row in cast(List[Any], prepared_rows_node)
        if isinstance(row, Mapping)
    ]
    if len(materialized_attack_event_rows) != len(prepared_rows_node):
        raise ValueError("PW04 prepared attack event rows must contain objects only")
    expected_row_count = prepared_attack_event_rows_payload.get("row_count")
    if isinstance(expected_row_count, int) and expected_row_count != len(materialized_attack_event_rows):
        raise ValueError(
            "PW04 prepared attack event rows row_count mismatch: "
            f"expected={expected_row_count}, actual={len(materialized_attack_event_rows)}"
        )

    return {
        "prepare_manifest_path": prepare_manifest_path,
        "prepare_manifest": prepare_manifest,
        "resolved_paths": resolved_paths,
        "quality_pair_plan_path": quality_pair_plan_path,
        "quality_pair_plan": quality_pair_plan,
        "prepared_attack_event_rows_path": prepared_attack_event_rows_path,
        "materialized_attack_event_rows": materialized_attack_event_rows,
        "expected_quality_shard_paths": expected_quality_shard_paths,
        "enable_tail_estimation": bool(prepare_manifest.get("enable_tail_estimation", False)),
    }


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
        shard_sample_role = shard_manifest.get("sample_role")
        if shard_sample_role not in {
            ATTACKED_POSITIVE_SAMPLE_ROLE,
            ATTACKED_NEGATIVE_SAMPLE_ROLE,
            MIXED_ATTACK_SAMPLE_ROLE,
        }:
            raise ValueError(
                "PW03 shard manifest sample_role mismatch: "
                f"attack_shard_index={attack_shard_index}, sample_role={shard_sample_role}"
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
            event_sample_role = event_manifest.get("sample_role")
            if event_sample_role not in {ATTACKED_POSITIVE_SAMPLE_ROLE, ATTACKED_NEGATIVE_SAMPLE_ROLE}:
                raise ValueError(
                    "PW03 event sample_role must be attacked_positive or attacked_negative before PW04: "
                    f"attack_event_id={expected_attack_event_id}, sample_role={event_sample_role}"
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
            if sample_role_value is not None and sample_role_value not in {
                ATTACKED_POSITIVE_SAMPLE_ROLE,
                ATTACKED_NEGATIVE_SAMPLE_ROLE,
            }:
                raise ValueError(
                    "PW03 staged detect record sample_role must be attacked_positive or attacked_negative: "
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
            attack_condition_base_key = event_manifest.get("attack_condition_base_key")
            attack_condition_key = event_manifest.get("attack_condition_key")
            attack_params_digest = event_manifest.get("attack_params_digest")
            if not isinstance(attack_family, str) or not attack_family:
                raise ValueError(f"PW03 event manifest missing attack_family: {attack_event_id}")
            if not isinstance(attack_config_name, str) or not attack_config_name:
                raise ValueError(f"PW03 event manifest missing attack_config_name: {attack_event_id}")
            if not isinstance(attack_condition_base_key, str) or not attack_condition_base_key:
                attack_condition_base_key = expected_attack_event.get("attack_condition_base_key")
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

            severity_metadata = _extract_mapping(event_manifest.get("severity_metadata"))
            if not severity_metadata:
                severity_metadata = _extract_mapping(detect_payload.get("paper_workflow_severity_metadata"))
            geometry_diagnostics = _extract_mapping(event_manifest.get("geometry_diagnostics"))
            if not geometry_diagnostics:
                geometry_diagnostics = _extract_mapping(detect_payload.get("paper_workflow_geometry_diagnostics"))
            geometry_optional_claim_evidence = _extract_mapping(event_manifest.get("geometry_optional_claim_evidence"))
            matrix_profile = event_manifest.get("matrix_profile", expected_attack_event.get("matrix_profile"))
            matrix_version = event_manifest.get("matrix_version", expected_attack_event.get("matrix_version"))
            matrix_attack_set_names = event_manifest.get(
                "matrix_attack_set_names",
                expected_attack_event.get("matrix_attack_set_names", []),
            )
            geometry_rescue_candidate = event_manifest.get(
                "geometry_rescue_candidate",
                expected_attack_event.get("geometry_rescue_candidate"),
            )

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
            parent_source_image_path = event_manifest.get("parent_source_image_path")
            if not isinstance(parent_source_image_path, str) or not parent_source_image_path.strip():
                detect_parent_source_path = detect_payload.get("paper_workflow_parent_source_image_path")
                if isinstance(detect_parent_source_path, str) and detect_parent_source_path.strip():
                    parent_source_image_path = detect_parent_source_path
                else:
                    parent_source_image_path = None
            parent_event_reference = _extract_mapping(event_manifest.get("parent_event_reference"))
            parent_prompt_text = _extract_prompt_text_from_mapping(parent_event_reference)
            if parent_prompt_text is None:
                parent_prompt_text = _extract_prompt_text_from_mapping(event_manifest)

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
                    "sample_role": event_sample_role,
                    "parent_event_id": parent_event_id,
                    "attack_family": attack_family,
                    "attack_config_name": attack_config_name,
                    "attack_condition_base_key": attack_condition_base_key,
                    "attack_condition_key": attack_condition_key,
                    "attack_params_digest": attack_params_digest,
                    "matrix_profile": matrix_profile,
                    "matrix_version": matrix_version,
                    "matrix_attack_set_names": copy.deepcopy(matrix_attack_set_names),
                    "geometry_rescue_candidate": geometry_rescue_candidate is True,
                    "source_finalize_manifest_digest": source_finalize_manifest_digest,
                    "threshold_artifact_paths": threshold_artifact_paths,
                    "parent_source_image_path": parent_source_image_path,
                    "parent_prompt_text": parent_prompt_text,
                    "attacked_image_path": attacked_image_path,
                    "payload_reference_sidecar_path": event_manifest.get("payload_reference_sidecar_path"),
                    "payload_decode_sidecar_path": event_manifest.get("payload_decode_sidecar_path"),
                    "content_score": content_score,
                    "content_score_source": content_score_source,
                    "event_attestation_score": attestation_score,
                    "event_attestation_score_source": attestation_score_source,
                    "lf_detect_variant": lf_detect_variant,
                    "severity_metadata": severity_metadata,
                    "severity_status": severity_metadata.get("severity_status"),
                    "severity_reason": severity_metadata.get("severity_reason"),
                    "severity_rule_version": severity_metadata.get("severity_rule_version"),
                    "severity_axis_kind": severity_metadata.get("severity_axis_kind"),
                    "severity_directionality": severity_metadata.get("severity_directionality"),
                    "severity_source_param": severity_metadata.get("severity_source_param"),
                    "severity_scalarization": severity_metadata.get("severity_scalarization"),
                    "severity_value": severity_metadata.get("severity_value"),
                    "severity_sort_value": severity_metadata.get("severity_sort_value"),
                    "severity_label": severity_metadata.get("severity_label"),
                    "severity_level_index": severity_metadata.get("severity_level_index"),
                    "geometry_optional_claim_evidence": geometry_optional_claim_evidence,
                    "geometry_diagnostics": geometry_diagnostics,
                    "sync_success": geometry_diagnostics.get("sync_success"),
                    "sync_status": geometry_diagnostics.get("sync_status"),
                    "sync_success_status": geometry_diagnostics.get("sync_success_status"),
                    "sync_success_reason": geometry_diagnostics.get("sync_success_reason"),
                    "inverse_transform_success": geometry_diagnostics.get("inverse_transform_success"),
                    "inverse_transform_success_status": geometry_diagnostics.get("inverse_transform_success_status"),
                    "inverse_transform_success_reason": geometry_diagnostics.get("inverse_transform_success_reason"),
                    "attention_anchor_available": geometry_diagnostics.get("attention_anchor_available"),
                    "attention_anchor_available_status": geometry_diagnostics.get("attention_anchor_available_status"),
                    "attention_anchor_available_reason": geometry_diagnostics.get("attention_anchor_available_reason"),
                    "sync_quality_metrics_status": geometry_diagnostics.get("sync_quality_metrics_status"),
                    "sync_quality_metrics_reason": geometry_diagnostics.get("sync_quality_metrics_reason"),
                    "geometry_failure_reason": geometry_diagnostics.get("geometry_failure_reason"),
                    "geometry_failure_reason_status": geometry_diagnostics.get("geometry_failure_reason_status"),
                    "geometry_failure_reason_reason": geometry_diagnostics.get("geometry_failure_reason_reason"),
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
    source_role: str,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build one attacked pool manifest payload.

    Args:
        family_id: Family identifier.
        source_role: Attack sample role represented by the pool.
        attack_event_rows: Collected and materialized attack event rows.

    Returns:
        Attacked pool manifest payload.
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
                "payload_reference_sidecar_path": attack_event_row.get("payload_reference_sidecar_path"),
                "payload_decode_sidecar_path": attack_event_row.get("payload_decode_sidecar_path"),
                "content_score": attack_event_row["content_score"],
                "event_attestation_score": attack_event_row["event_attestation_score"],
                "formal_final_decision_is_positive": formal_final_decision.get("is_watermarked") is True,
                "formal_event_attestation_is_positive": formal_event_attestation_decision.get("is_watermarked") is True,
                "derived_attack_union_positive": bool(formal_record.get("derived_attack_union_positive", False)),
            }
        )
    artifact_type = (
        "paper_workflow_pw04_attack_positive_pool_manifest"
        if source_role == ATTACKED_POSITIVE_SAMPLE_ROLE
        else "paper_workflow_pw04_attack_negative_pool_manifest"
    )
    return {
        "artifact_type": artifact_type,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "source_role": source_role,
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


def _build_attack_negative_metrics_export(
    *,
    family_id: str,
    content_threshold_export_path: Path,
    attestation_threshold_export_path: Path,
    attack_negative_pool_manifest_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build the attacked-negative false-accept summary export.

    Args:
        family_id: Family identifier.
        content_threshold_export_path: PW02 content threshold export path.
        attestation_threshold_export_path: PW02 attestation threshold export path.
        attack_negative_pool_manifest_path: PW04 attacked-negative pool manifest path.
        attack_event_rows: Attacked-negative materialized rows.

    Returns:
        Attacked-negative summary payload.
    """
    attack_negative_count = len(attack_event_rows)
    formal_final_false_accept_count = 0
    formal_attestation_false_accept_count = 0
    derived_attack_union_false_accept_count = 0
    for attack_event_row in attack_event_rows:
        formal_record = cast(Mapping[str, Any], attack_event_row["formal_record"])
        formal_final_decision = cast(Mapping[str, Any], formal_record.get("formal_final_decision", {}))
        formal_event_attestation_decision = cast(
            Mapping[str, Any],
            formal_record.get("formal_event_attestation_decision", {}),
        )
        if formal_final_decision.get("is_watermarked") is True:
            formal_final_false_accept_count += 1
        if formal_event_attestation_decision.get("is_watermarked") is True:
            formal_attestation_false_accept_count += 1
        if bool(formal_record.get("derived_attack_union_positive", False)):
            derived_attack_union_false_accept_count += 1

    return {
        "artifact_type": "paper_workflow_pw04_formal_attack_negative_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "source_role": ATTACKED_NEGATIVE_SAMPLE_ROLE,
        "content_threshold_export_path": normalize_path_value(content_threshold_export_path),
        "attestation_threshold_export_path": normalize_path_value(attestation_threshold_export_path),
        "attack_negative_pool_manifest_path": normalize_path_value(attack_negative_pool_manifest_path),
        "metrics": {
            "attack_negative_count": attack_negative_count,
            "formal_final_false_accept_count": formal_final_false_accept_count,
            "formal_final_attack_fpr": None if attack_negative_count <= 0 else float(formal_final_false_accept_count / attack_negative_count),
            "formal_attestation_false_accept_count": formal_attestation_false_accept_count,
            "formal_attestation_attack_fpr": None if attack_negative_count <= 0 else float(formal_attestation_false_accept_count / attack_negative_count),
            "derived_attack_union_false_accept_count": derived_attack_union_false_accept_count,
            "derived_attack_union_attack_fpr": None if attack_negative_count <= 0 else float(derived_attack_union_false_accept_count / attack_negative_count),
        },
        "notes": "This append-only export summarizes false accepts on attacked clean_negative parents and does not alter canonical positive-attack TPR tables.",
    }


def _build_grouped_attack_quality_rows(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    group_key_name: str,
) -> List[Dict[str, Any]]:
    """
    Build grouped attack image-quality rows for one categorical key.

    Args:
        pair_rows: Per-pair attack quality rows.
        group_key_name: Grouping field name.

    Returns:
        Grouped attack quality rows.
    """
    if not isinstance(group_key_name, str) or not group_key_name:
        raise TypeError("group_key_name must be non-empty str")

    grouped_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for pair_row in pair_rows:
        if not isinstance(pair_row, Mapping):
            raise TypeError("pair_rows items must be mappings")
        if pair_row.get("status") != "ok":
            continue
        group_value = pair_row.get(group_key_name)
        if not isinstance(group_value, str) or not group_value:
            raise ValueError(f"pair row missing {group_key_name}")
        grouped_rows.setdefault(group_value, []).append(pair_row)

    output_rows: List[Dict[str, Any]] = []
    for group_value in sorted(grouped_rows):
        rows = grouped_rows[group_value]
        psnr_values = [
            float(row["psnr"])
            for row in rows
            if isinstance(row.get("psnr"), (int, float)) and not isinstance(row.get("psnr"), bool)
        ]
        ssim_values = [
            float(row["ssim"])
            for row in rows
            if isinstance(row.get("ssim"), (int, float)) and not isinstance(row.get("ssim"), bool)
        ]
        lpips_values = [
            float(row["lpips"])
            for row in rows
            if isinstance(row.get("lpips"), (int, float)) and not isinstance(row.get("lpips"), bool)
        ]
        clip_values = [
            float(row["clip_text_similarity"])
            for row in rows
            if isinstance(row.get("clip_text_similarity"), (int, float)) and not isinstance(row.get("clip_text_similarity"), bool)
        ]
        grouped_row: Dict[str, Any] = {
            group_key_name: group_value,
            "quality_pair_count": len(psnr_values),
            "mean_psnr": float(sum(psnr_values) / len(psnr_values)) if psnr_values else None,
            "mean_ssim": float(sum(ssim_values) / len(ssim_values)) if ssim_values else None,
            "mean_lpips": float(sum(lpips_values) / len(lpips_values)) if lpips_values else None,
            "mean_clip_text_similarity": float(sum(clip_values) / len(clip_values)) if clip_values else None,
            "clip_sample_count": len(clip_values),
        }
        if group_key_name == "attack_condition_key":
            grouped_row["attack_family"] = rows[0].get("attack_family")
            grouped_row["attack_config_name"] = rows[0].get("attack_config_name")
        output_rows.append(grouped_row)
    return output_rows


def _build_attack_quality_metrics_export(
    *,
    family_id: str,
    output_path: Path,
    attack_event_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build one append-only attack image-quality metrics export.

    Args:
        family_id: Family identifier.
        output_path: Output artifact path.
        attack_event_rows: Materialized attack event rows.

    Returns:
        Export summary with payload and per-event lookup.
    """
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be Path")

    pair_specs: List[Dict[str, Any]] = []
    for attack_event_row in attack_event_rows:
        if not isinstance(attack_event_row, Mapping):
            raise TypeError("attack_event_rows items must be mappings")
        pair_specs.append(
            {
                "attack_event_id": attack_event_row.get("attack_event_id"),
                "parent_event_id": attack_event_row.get("parent_event_id"),
                "attack_family": attack_event_row.get("attack_family"),
                "attack_condition_key": attack_event_row.get("attack_condition_key"),
                "attack_config_name": attack_event_row.get("attack_config_name"),
                "reference_image_path": attack_event_row.get("parent_source_image_path"),
                "candidate_image_path": attack_event_row.get("attacked_image_path"),
                "prompt_text": attack_event_row.get("parent_prompt_text"),
            }
        )

    quality_summary = build_quality_metrics_from_pairs(
        pair_specs=pair_specs,
        reference_path_key="reference_image_path",
        candidate_path_key="candidate_image_path",
        pair_id_key="attack_event_id",
        text_key="prompt_text",
        extra_metadata_keys=["parent_event_id", "attack_family", "attack_condition_key", "attack_config_name"],
    )
    pair_rows = cast(List[Mapping[str, Any]], quality_summary.get("pair_rows", []))
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw04_attack_quality_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "reference_semantics": "parent_source_image_vs_attacked_image",
        "overall": {
            "status": quality_summary.get("status"),
            "availability_reason": quality_summary.get("availability_reason"),
            "expected_count": quality_summary.get("expected_count"),
            "count": quality_summary.get("count"),
            "missing_count": quality_summary.get("missing_count"),
            "error_count": quality_summary.get("error_count"),
            "mean_psnr": quality_summary.get("mean_psnr"),
            "mean_ssim": quality_summary.get("mean_ssim"),
            "lpips_status": quality_summary.get("lpips_status"),
            "lpips_reason": quality_summary.get("lpips_reason"),
            "mean_lpips": quality_summary.get("mean_lpips"),
            "mean_clip_text_similarity": quality_summary.get("mean_clip_text_similarity"),
            "clip_model_name": quality_summary.get("clip_model_name"),
            "clip_sample_count": quality_summary.get("clip_sample_count"),
            "clip_status": quality_summary.get("clip_status"),
            "clip_reason": quality_summary.get("clip_reason"),
            "quality_runtime": quality_summary.get("quality_runtime"),
            "prompt_text_expected": quality_summary.get("prompt_text_expected"),
            "prompt_text_available_count": quality_summary.get("prompt_text_available_count"),
            "prompt_text_missing_count": quality_summary.get("prompt_text_missing_count"),
            "prompt_text_coverage_status": quality_summary.get("prompt_text_coverage_status"),
            "prompt_text_coverage_reason": quality_summary.get("prompt_text_coverage_reason"),
            "quality_readiness_status": quality_summary.get("quality_readiness_status"),
            "quality_readiness_reason": quality_summary.get("quality_readiness_reason"),
            "quality_readiness_blocking": quality_summary.get("quality_readiness_blocking"),
            "quality_readiness_required_for_formal_release": quality_summary.get(
                "quality_readiness_required_for_formal_release"
            ),
        },
        "by_attack_family": _build_grouped_attack_quality_rows(
            pair_rows=pair_rows,
            group_key_name="attack_family",
        ),
        "by_attack_condition": _build_grouped_attack_quality_rows(
            pair_rows=pair_rows,
            group_key_name="attack_condition_key",
        ),
        "pair_rows": list(pair_rows),
    }
    write_json_atomic(output_path, payload)
    return {
        "path": normalize_path_value(output_path),
        "payload": payload,
        "pair_lookup": {
            str(pair_row["pair_id"]): dict(pair_row)
            for pair_row in pair_rows
            if isinstance(pair_row.get("pair_id"), str) and pair_row.get("pair_id")
        },
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
        quality_psnr_values = [
            float(row["attack_quality_psnr"])
            for row in rows
            if isinstance(row.get("attack_quality_psnr"), (int, float)) and not isinstance(row.get("attack_quality_psnr"), bool)
        ]
        quality_ssim_values = [
            float(row["attack_quality_ssim"])
            for row in rows
            if isinstance(row.get("attack_quality_ssim"), (int, float)) and not isinstance(row.get("attack_quality_ssim"), bool)
        ]
        quality_lpips_values = [
            float(row["attack_quality_lpips"])
            for row in rows
            if isinstance(row.get("attack_quality_lpips"), (int, float)) and not isinstance(row.get("attack_quality_lpips"), bool)
        ]
        quality_clip_values = [
            float(row["attack_quality_clip_text_similarity"])
            for row in rows
            if isinstance(row.get("attack_quality_clip_text_similarity"), (int, float)) and not isinstance(row.get("attack_quality_clip_text_similarity"), bool)
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
            "attack_quality_pair_count": len(quality_psnr_values),
            "attack_mean_psnr": float(sum(quality_psnr_values) / len(quality_psnr_values)) if quality_psnr_values else None,
            "attack_mean_ssim": float(sum(quality_ssim_values) / len(quality_ssim_values)) if quality_ssim_values else None,
            "attack_mean_lpips": float(sum(quality_lpips_values) / len(quality_lpips_values)) if quality_lpips_values else None,
            "attack_mean_clip_text_similarity": float(sum(quality_clip_values) / len(quality_clip_values)) if quality_clip_values else None,
            "attack_clip_sample_count": len(quality_clip_values),
        }
        if group_key_name == "attack_condition_key":
            row_payload["attack_family"] = str(rows[0]["attack_family"])
            row_payload["attack_config_name"] = str(rows[0]["attack_config_name"])
            row_payload["severity_status"] = rows[0].get("severity_status")
            row_payload["severity_reason"] = rows[0].get("severity_reason")
            row_payload["severity_rule_version"] = rows[0].get("severity_rule_version")
            row_payload["severity_axis_kind"] = rows[0].get("severity_axis_kind")
            row_payload["severity_directionality"] = rows[0].get("severity_directionality")
            row_payload["severity_source_param"] = rows[0].get("severity_source_param")
            row_payload["severity_scalarization"] = rows[0].get("severity_scalarization")
            row_payload["severity_value"] = rows[0].get("severity_value")
            row_payload["severity_sort_value"] = rows[0].get("severity_sort_value")
            row_payload["severity_label"] = rows[0].get("severity_label")
            row_payload["severity_level_index"] = rows[0].get("severity_level_index")
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
        attestation_payload = cast(Mapping[str, Any], formal_record.get("attestation", {}))
        image_evidence_payload = cast(Mapping[str, Any], attestation_payload.get("image_evidence_result", {}))
        geometry_optional_claim_evidence = _extract_mapping(attack_event_row.get("geometry_optional_claim_evidence"))
        rows.append(
            {
                "attack_event_id": attack_event_row["attack_event_id"],
                "attack_event_index": attack_event_row["attack_event_index"],
                "parent_event_id": attack_event_row["parent_event_id"],
                "attack_family": attack_event_row["attack_family"],
                "attack_config_name": attack_event_row["attack_config_name"],
                "attack_condition_base_key": attack_event_row.get("attack_condition_base_key"),
                "attack_condition_key": attack_event_row["attack_condition_key"],
                "attack_params_digest": attack_event_row["attack_params_digest"],
                "matrix_profile": attack_event_row.get("matrix_profile"),
                "matrix_version": attack_event_row.get("matrix_version"),
                "matrix_attack_set_names": copy.deepcopy(attack_event_row.get("matrix_attack_set_names", [])),
                "geometry_rescue_candidate": attack_event_row.get("geometry_rescue_candidate"),
                "severity_status": attack_event_row.get("severity_status"),
                "severity_reason": attack_event_row.get("severity_reason"),
                "severity_rule_version": attack_event_row.get("severity_rule_version"),
                "severity_axis_kind": attack_event_row.get("severity_axis_kind"),
                "severity_directionality": attack_event_row.get("severity_directionality"),
                "severity_source_param": attack_event_row.get("severity_source_param"),
                "severity_scalarization": attack_event_row.get("severity_scalarization"),
                "severity_value": attack_event_row.get("severity_value"),
                "severity_sort_value": attack_event_row.get("severity_sort_value"),
                "severity_label": attack_event_row.get("severity_label"),
                "severity_level_index": attack_event_row.get("severity_level_index"),
                "sample_role": attack_event_row.get("sample_role"),
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
                "sync_status": attack_event_row.get("sync_status"),
                "sync_success": attack_event_row.get("sync_success"),
                "sync_success_status": attack_event_row.get("sync_success_status"),
                "sync_success_reason": attack_event_row.get("sync_success_reason"),
                "inverse_transform_success": attack_event_row.get("inverse_transform_success"),
                "inverse_transform_success_status": attack_event_row.get("inverse_transform_success_status"),
                "inverse_transform_success_reason": attack_event_row.get("inverse_transform_success_reason"),
                "attention_anchor_available": attack_event_row.get("attention_anchor_available"),
                "attention_anchor_available_status": attack_event_row.get("attention_anchor_available_status"),
                "attention_anchor_available_reason": attack_event_row.get("attention_anchor_available_reason"),
                "sync_quality_metrics_status": attack_event_row.get("sync_quality_metrics_status"),
                "sync_quality_metrics_reason": attack_event_row.get("sync_quality_metrics_reason"),
                "geometry_failure_reason": attack_event_row.get("geometry_failure_reason"),
                "geometry_failure_reason_status": attack_event_row.get("geometry_failure_reason_status"),
                "geometry_failure_reason_reason": attack_event_row.get("geometry_failure_reason_reason"),
                "image_evidence_status": image_evidence_payload.get("status"),
                "geo_rescue_eligible": image_evidence_payload.get("geo_rescue_eligible"),
                "geo_rescue_applied": image_evidence_payload.get("geo_rescue_applied"),
                "geo_not_used_reason": image_evidence_payload.get("geo_not_used_reason"),
                "geometry_optional_claim_status": geometry_optional_claim_evidence.get("status"),
                "geometry_optional_claim_reason": geometry_optional_claim_evidence.get("reason"),
                "eligible_for_optional_claim": geometry_optional_claim_evidence.get("eligible_for_optional_claim"),
                "boundary_rule_version": geometry_optional_claim_evidence.get("boundary_rule_version"),
                "boundary_metric": geometry_optional_claim_evidence.get("boundary_metric"),
                "boundary_abs_margin_min": geometry_optional_claim_evidence.get("boundary_abs_margin_min"),
                "boundary_abs_margin_max": geometry_optional_claim_evidence.get("boundary_abs_margin_max"),
                "boundary_metric_value": geometry_optional_claim_evidence.get("boundary_metric_value"),
                "boundary_resolution_status": geometry_optional_claim_evidence.get("boundary_resolution_status"),
                "boundary_resolution_reason": geometry_optional_claim_evidence.get("boundary_resolution_reason"),
                "attack_quality_status": attack_event_row.get("attack_quality_status"),
                "attack_quality_psnr": attack_event_row.get("attack_quality_psnr"),
                "attack_quality_ssim": attack_event_row.get("attack_quality_ssim"),
                "attack_quality_lpips": attack_event_row.get("attack_quality_lpips"),
                "attack_quality_clip_text_similarity": attack_event_row.get("attack_quality_clip_text_similarity"),
                "attack_quality_clip_model_name": attack_event_row.get("attack_quality_clip_model_name"),
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
    attack_quality_metrics_path: Path,
    attack_negative_metrics_path: Path,
    attack_formal_metrics_payload: Mapping[str, Any],
    attack_attestation_metrics_payload: Mapping[str, Any],
    attack_derived_metrics_payload: Mapping[str, Any],
    attack_quality_metrics_payload: Mapping[str, Any],
    attack_negative_metrics_payload: Mapping[str, Any],
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
    attack_quality_overall = cast(Mapping[str, Any], attack_quality_metrics_payload.get("overall", {}))
    attack_negative_metrics = cast(Mapping[str, Any], attack_negative_metrics_payload.get("metrics", {}))
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
        "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
        "attack_negative_metrics_path": normalize_path_value(attack_negative_metrics_path),
        "clean_formal_tpr": clean_formal_metrics.get("final_decision_tpr"),
        "clean_formal_fpr": clean_formal_metrics.get("final_decision_fpr"),
        "clean_derived_union_tpr": clean_derived_metrics.get("system_tpr"),
        "attack_formal_tpr": attack_formal_metrics.get("attack_tpr"),
        "attack_formal_attestation_tpr": attack_attestation_metrics.get("attack_tpr"),
        "attack_derived_union_tpr": attack_derived_metrics.get("attack_tpr"),
        "attack_negative_formal_fpr": attack_negative_metrics.get("formal_final_attack_fpr"),
        "attack_negative_formal_attestation_fpr": attack_negative_metrics.get("formal_attestation_attack_fpr"),
        "attack_negative_derived_union_fpr": attack_negative_metrics.get("derived_attack_union_attack_fpr"),
        "attack_quality_mean_psnr": attack_quality_overall.get("mean_psnr"),
        "attack_quality_mean_ssim": attack_quality_overall.get("mean_ssim"),
        "attack_quality_mean_lpips": attack_quality_overall.get("mean_lpips"),
        "attack_quality_mean_clip_text_similarity": attack_quality_overall.get("mean_clip_text_similarity"),
        "attack_quality_clip_model_name": attack_quality_overall.get("clip_model_name"),
        "attack_quality_clip_sample_count": attack_quality_overall.get("clip_sample_count"),
        "attack_quality_clip_status": attack_quality_overall.get("clip_status"),
        "attack_quality_clip_reason": attack_quality_overall.get("clip_reason"),
    }


def _build_pw04_summary_payload(
    *,
    family_id: str,
    family_root: Path,
    summary_path: Path,
    attack_merge_manifest_path: Path,
    attack_pool_manifest_path: Path,
    attack_negative_pool_manifest_path: Path,
    formal_attack_final_decision_metrics_path: Path,
    formal_attack_attestation_metrics_path: Path,
    derived_attack_union_metrics_path: Path,
    formal_attack_negative_metrics_path: Path,
    per_attack_family_metrics_path: Path,
    per_attack_condition_metrics_path: Path,
    attack_quality_metrics_path: Path,
    attack_event_table_path: Path,
    attack_family_summary_csv_path: Path,
    attack_condition_summary_csv_path: Path,
    clean_attack_overview_path: Path,
    expected_attack_shard_count: int,
    discovered_attack_shard_count: int,
    expected_attack_event_count: int,
    discovered_attack_event_count: int,
    completed_attack_event_count: int,
    attacked_positive_event_count: int,
    attacked_negative_event_count: int,
    attack_family_count: int,
    parent_event_count: int,
    formal_record_count: int,
    enable_tail_estimation: bool,
    paper_exports_payload: Mapping[str, Any] | None = None,
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
        enable_tail_estimation: Whether optional tail estimation was enabled.
        paper_exports_payload: Append-only paper-facing export bindings.

    Returns:
        PW04 summary payload.
    """
    summary_payload = {
        "status": "completed",
        "stage_name": STAGE_NAME,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "summary_path": normalize_path_value(summary_path),
        "attack_merge_manifest_path": normalize_path_value(attack_merge_manifest_path),
        "attack_positive_pool_manifest_path": normalize_path_value(attack_pool_manifest_path),
        "attack_negative_pool_manifest_path": normalize_path_value(attack_negative_pool_manifest_path),
        "formal_attack_final_decision_metrics_path": normalize_path_value(formal_attack_final_decision_metrics_path),
        "formal_attack_attestation_metrics_path": normalize_path_value(formal_attack_attestation_metrics_path),
        "derived_attack_union_metrics_path": normalize_path_value(derived_attack_union_metrics_path),
        "formal_attack_negative_metrics_path": normalize_path_value(formal_attack_negative_metrics_path),
        "per_attack_family_metrics_path": normalize_path_value(per_attack_family_metrics_path),
        "per_attack_condition_metrics_path": normalize_path_value(per_attack_condition_metrics_path),
        "attack_quality_metrics_path": normalize_path_value(attack_quality_metrics_path),
        "attack_event_table_path": normalize_path_value(attack_event_table_path),
        "attack_family_summary_csv_path": normalize_path_value(attack_family_summary_csv_path),
        "attack_condition_summary_csv_path": normalize_path_value(attack_condition_summary_csv_path),
        "clean_attack_overview_path": normalize_path_value(clean_attack_overview_path),
        "expected_attack_shard_count": expected_attack_shard_count,
        "discovered_attack_shard_count": discovered_attack_shard_count,
        "expected_attack_event_count": expected_attack_event_count,
        "discovered_attack_event_count": discovered_attack_event_count,
        "completed_attack_event_count": completed_attack_event_count,
        "attacked_positive_event_count": attacked_positive_event_count,
        "attacked_negative_event_count": attacked_negative_event_count,
        "attack_family_count": attack_family_count,
        "parent_event_count": parent_event_count,
        "formal_record_count": formal_record_count,
        "tail_estimation_enabled": enable_tail_estimation,
    }
    if isinstance(paper_exports_payload, Mapping):
        summary_payload.update(dict(cast(Mapping[str, Any], paper_exports_payload)))
    return summary_payload


def run_pw04_merge_attack_event_shards(
    *,
    drive_project_root: Path,
    family_id: str,
    force_rerun: bool = False,
    enable_tail_estimation: bool = False,
    pw04_mode: str = PW04_MODE_PREPARE,
    quality_shard_index: int | None = None,
    quality_shard_count: int | None = None,
) -> Dict[str, Any]:
    """
    Execute the PW04 attack merge and metrics materialization stage.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        force_rerun: Whether to clear existing PW04 outputs before rerun.
        enable_tail_estimation: Whether optional tail estimation exports should be produced.
        pw04_mode: Explicit PW04 execution mode.
        quality_shard_index: Optional quality shard index for worker mode.
        quality_shard_count: Optional explicit quality shard count for prepare mode.

    Returns:
        Mode-specific PW04 execution summary.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(force_rerun, bool):
        raise TypeError("force_rerun must be bool")
    if not isinstance(enable_tail_estimation, bool):
        raise TypeError("enable_tail_estimation must be bool")
    if quality_shard_index is not None and (
        not isinstance(quality_shard_index, int)
        or isinstance(quality_shard_index, bool)
        or quality_shard_index < 0
    ):
        raise TypeError("quality_shard_index must be non-negative int when provided")
    if quality_shard_count is not None and (
        not isinstance(quality_shard_count, int)
        or isinstance(quality_shard_count, bool)
        or quality_shard_count <= 0
    ):
        raise TypeError("quality_shard_count must be positive int when provided")

    resolved_mode = _resolve_pw04_mode(pw04_mode)
    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    pw04_paths = _resolve_pw04_paths(family_root)
    if resolved_mode != PW04_MODE_QUALITY_SHARD and quality_shard_index is not None:
        raise ValueError("quality_shard_index is only valid when pw04_mode=quality_shard")
    if resolved_mode != PW04_MODE_PREPARE and quality_shard_count is not None:
        raise ValueError("quality_shard_count is only valid when pw04_mode=prepare")

    if resolved_mode == PW04_MODE_PREPARE:
        _prepare_pw04_outputs(
            family_root=family_root,
            export_root=cast(Path, pw04_paths["export_root"]),
            summary_path=cast(Path, pw04_paths["summary_path"]),
            force_rerun=force_rerun,
        )
        ensure_directory(cast(Path, pw04_paths["manifests_root"]))
        ensure_directory(cast(Path, pw04_paths["records_root"]))
        ensure_directory(cast(Path, pw04_paths["tables_root"]))
        ensure_directory(cast(Path, pw04_paths["metrics_root"]))
        ensure_directory(cast(Path, pw04_paths["figures_root"]))
        ensure_directory(cast(Path, pw04_paths["tail_root"]))
        ensure_directory(cast(Path, pw04_paths["quality_root"]))

        pw02_summary_path = family_root / "runtime_state" / PW02_SUMMARY_FILE_NAME
        finalize_manifest_path = family_root / "exports" / "pw02" / "paper_source_finalize_manifest.json"
        content_threshold_export_path = family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json"
        attestation_threshold_export_path = family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json"
        attack_shard_plan_path = family_root / "manifests" / "attack_shard_plan.json"
        attack_event_grid_path = family_root / "manifests" / "attack_event_grid.jsonl"

        pw02_summary = _load_required_json_dict(pw02_summary_path, "PW02 summary")
        _load_required_json_dict(finalize_manifest_path, "paper source finalize manifest")
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
        positive_attack_event_rows = [
            row for row in materialized_attack_event_rows if row.get("sample_role") == ATTACKED_POSITIVE_SAMPLE_ROLE
        ]
        negative_attack_event_rows = [
            row for row in materialized_attack_event_rows if row.get("sample_role") == ATTACKED_NEGATIVE_SAMPLE_ROLE
        ]

        resolved_quality_shard_count = quality_shard_count or expected_attack_shard_count

        quality_pair_plan_export = run_pw04_prepare_quality_pairs(
            family_id=family_id,
            family_root=family_root,
            pw02_summary=pw02_summary,
            attack_event_rows=positive_attack_event_rows,
            quality_root=cast(Path, pw04_paths["quality_root"]),
            planned_shard_count=resolved_quality_shard_count,
        )
        quality_pair_plan_path = Path(str(quality_pair_plan_export["path"])).expanduser().resolve()
        quality_pair_plan_payload = cast(Mapping[str, Any], quality_pair_plan_export["payload"])
        materialized_quality_shard_count = quality_pair_plan_payload.get("quality_shard_count")
        if (
            not isinstance(materialized_quality_shard_count, int)
            or isinstance(materialized_quality_shard_count, bool)
            or materialized_quality_shard_count <= 0
        ):
            raise ValueError("PW04 quality pair plan missing quality_shard_count")
        if materialized_quality_shard_count != resolved_quality_shard_count:
            raise ValueError("PW04 quality pair plan quality_shard_count mismatch with prepare contract")
        expected_quality_shard_paths = _build_expected_quality_shard_paths(
            family_root=family_root,
            quality_root=cast(Path, pw04_paths["quality_root"]),
            quality_pair_plan=quality_pair_plan_payload,
        )
        if materialized_quality_shard_count != len(expected_quality_shard_paths):
            raise ValueError(
                "PW04 quality pair plan quality_shard_count mismatch with expected quality shard paths"
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
            source_role=ATTACKED_POSITIVE_SAMPLE_ROLE,
            attack_event_rows=positive_attack_event_rows,
        )
        attack_negative_pool_manifest_payload = _build_attack_pool_manifest_payload(
            family_id=family_id,
            source_role=ATTACKED_NEGATIVE_SAMPLE_ROLE,
            attack_event_rows=negative_attack_event_rows,
        )
        formal_attack_final_decision_metrics_payload = _build_formal_attack_final_decision_metrics_export(
            family_id=family_id,
            finalize_manifest_path=finalize_manifest_path,
            content_threshold_export_path=content_threshold_export_path,
            attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
            attack_event_rows=positive_attack_event_rows,
        )
        formal_attack_attestation_metrics_payload = _build_formal_attack_attestation_metrics_export(
            family_id=family_id,
            finalize_manifest_path=finalize_manifest_path,
            attestation_threshold_export_path=attestation_threshold_export_path,
            attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
            attack_event_rows=positive_attack_event_rows,
        )
        derived_attack_union_metrics_payload = _build_derived_attack_union_metrics_export(
            family_id=family_id,
            content_threshold_export_path=content_threshold_export_path,
            attestation_threshold_export_path=attestation_threshold_export_path,
            attack_pool_manifest_path=cast(Path, pw04_paths["attack_pool_manifest_path"]),
            attack_event_rows=positive_attack_event_rows,
        )
        formal_attack_negative_metrics_payload = _build_attack_negative_metrics_export(
            family_id=family_id,
            content_threshold_export_path=content_threshold_export_path,
            attestation_threshold_export_path=attestation_threshold_export_path,
            attack_negative_pool_manifest_path=cast(Path, pw04_paths["attack_negative_pool_manifest_path"]),
            attack_event_rows=negative_attack_event_rows,
        )
        per_attack_family_metrics_payload = _build_per_attack_family_metrics_export(
            family_id=family_id,
            attack_event_rows=positive_attack_event_rows,
        )
        per_attack_condition_metrics_payload = _build_per_attack_condition_metrics_export(
            family_id=family_id,
            attack_event_rows=positive_attack_event_rows,
        )
        prepared_attack_event_rows_payload = _build_prepared_attack_event_rows_payload(
            family_id=family_id,
            attack_event_rows=materialized_attack_event_rows,
        )

        write_json_atomic(cast(Path, pw04_paths["attack_merge_manifest_path"]), attack_merge_manifest_payload)
        write_json_atomic(cast(Path, pw04_paths["attack_pool_manifest_path"]), attack_pool_manifest_payload)
        write_json_atomic(cast(Path, pw04_paths["attack_negative_pool_manifest_path"]), attack_negative_pool_manifest_payload)
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
        write_json_atomic(
            cast(Path, pw04_paths["formal_attack_negative_metrics_path"]),
            formal_attack_negative_metrics_payload,
        )
        write_json_atomic(cast(Path, pw04_paths["per_attack_family_metrics_path"]), per_attack_family_metrics_payload)
        write_json_atomic(cast(Path, pw04_paths["per_attack_condition_metrics_path"]), per_attack_condition_metrics_payload)
        write_json_atomic(
            cast(Path, pw04_paths["prepared_attack_event_rows_path"]),
            prepared_attack_event_rows_payload,
        )

        prepare_manifest_payload = _build_prepare_manifest_payload(
            family_id=family_id,
            family_root=family_root,
            pw04_paths=pw04_paths,
            quality_pair_plan_path=quality_pair_plan_path,
            prepared_attack_event_rows_path=cast(Path, pw04_paths["prepared_attack_event_rows_path"]),
            expected_quality_shard_paths=expected_quality_shard_paths,
            expected_attack_shard_count=expected_attack_shard_count,
            discovered_attack_shard_count=len(shard_rows),
            expected_attack_event_count=expected_attack_event_count,
            discovered_attack_event_count=len(materialized_attack_event_rows),
            completed_attack_event_count=len(materialized_attack_event_rows),
            attacked_positive_event_count=len(positive_attack_event_rows),
            attacked_negative_event_count=len(negative_attack_event_rows),
            enable_tail_estimation=enable_tail_estimation,
        )
        write_json_atomic(cast(Path, pw04_paths["prepare_manifest_path"]), prepare_manifest_payload)

        return {
            "artifact_type": "paper_workflow_pw04_prepare_result",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "stage_name": STAGE_NAME,
            "pw04_mode": resolved_mode,
            "family_id": family_id,
            "family_root": normalize_path_value(family_root),
            "status": "completed",
            "prepare_manifest_path": normalize_path_value(cast(Path, pw04_paths["prepare_manifest_path"])),
            "prepared_attack_event_rows_path": normalize_path_value(cast(Path, pw04_paths["prepared_attack_event_rows_path"])),
            "attack_merge_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_merge_manifest_path"])),
            "attack_positive_pool_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_pool_manifest_path"])),
            "attack_negative_pool_manifest_path": normalize_path_value(cast(Path, pw04_paths["attack_negative_pool_manifest_path"])),
            "formal_attack_final_decision_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_final_decision_metrics_path"])),
            "formal_attack_attestation_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_attestation_metrics_path"])),
            "derived_attack_union_metrics_path": normalize_path_value(cast(Path, pw04_paths["derived_attack_union_metrics_path"])),
            "formal_attack_negative_metrics_path": normalize_path_value(cast(Path, pw04_paths["formal_attack_negative_metrics_path"])),
            "per_attack_family_metrics_path": normalize_path_value(cast(Path, pw04_paths["per_attack_family_metrics_path"])),
            "per_attack_condition_metrics_path": normalize_path_value(cast(Path, pw04_paths["per_attack_condition_metrics_path"])),
            "quality_pair_plan_path": normalize_path_value(quality_pair_plan_path),
            "quality_shard_count": resolved_quality_shard_count,
            "expected_quality_shard_paths": [normalize_path_value(path_obj) for path_obj in expected_quality_shard_paths],
            "completed_attack_event_count": len(materialized_attack_event_rows),
            "attacked_positive_event_count": len(positive_attack_event_rows),
            "attacked_negative_event_count": len(negative_attack_event_rows),
            "enable_tail_estimation": enable_tail_estimation,
        }

    prepare_context = _load_pw04_prepare_context(
        family_id=family_id,
        family_root=family_root,
        pw04_paths=pw04_paths,
    )

    if resolved_mode == PW04_MODE_QUALITY_SHARD:
        if quality_shard_index is None:
            raise ValueError("quality_shard_index is required when pw04_mode=quality_shard")
        summary_path = cast(Path, pw04_paths["summary_path"])
        if summary_path.exists():
            raise RuntimeError(
                "PW04 final summary already exists; rerun quality shards requires restarting from prepare mode"
            )

        expected_quality_shard_path = (
            cast(Path, pw04_paths["quality_root"])
            / QUALITY_SHARDS_DIRECTORY_NAME
            / QUALITY_SHARD_FILE_NAME_TEMPLATE.format(quality_shard_index=quality_shard_index)
        )
        expected_quality_shard_path_set = {
            normalize_path_value(path_obj) for path_obj in cast(List[Path], prepare_context["expected_quality_shard_paths"])
        }
        if normalize_path_value(expected_quality_shard_path) not in expected_quality_shard_path_set:
            raise ValueError(
                "PW04 quality_shard mode received unexpected shard index: "
                f"quality_shard_index={quality_shard_index}"
            )
        if expected_quality_shard_path.exists():
            if not force_rerun:
                raise RuntimeError(
                    "PW04 quality shard output already exists; rerun requires force_rerun: "
                    f"quality_shard_path={normalize_path_value(expected_quality_shard_path)}"
                )
            expected_quality_shard_path.unlink()

        shard_export = run_pw04_quality_shard(
            family_id=family_id,
            quality_pair_plan_path=cast(Path, prepare_context["quality_pair_plan_path"]),
            quality_shard_index=quality_shard_index,
        )
        if normalize_path_value(Path(str(shard_export["path"])).expanduser().resolve()) != normalize_path_value(expected_quality_shard_path):
            raise ValueError("PW04 quality shard output path mismatch with frozen prepare manifest")
        return {
            "artifact_type": "paper_workflow_pw04_quality_shard_result",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "stage_name": STAGE_NAME,
            "pw04_mode": resolved_mode,
            "family_id": family_id,
            "family_root": normalize_path_value(family_root),
            "status": "completed",
            "prepare_manifest_path": normalize_path_value(cast(Path, prepare_context["prepare_manifest_path"])),
            "quality_pair_plan_path": normalize_path_value(cast(Path, prepare_context["quality_pair_plan_path"])),
            "quality_shard_index": quality_shard_index,
            "quality_shard_path": normalize_path_value(expected_quality_shard_path),
            "quality_shard_count": len(cast(List[Path], prepare_context["expected_quality_shard_paths"])),
        }

    summary_path = cast(Path, pw04_paths["summary_path"])
    if summary_path.exists() and not force_rerun:
        raise RuntimeError(
            "PW04 outputs already finalized; rerun finalize requires force_rerun: "
            f"summary_path={normalize_path_value(summary_path)}"
        )
    if enable_tail_estimation and not bool(prepare_context["enable_tail_estimation"]):
        raise ValueError(
            "PW04 finalize cannot enable tail estimation beyond the frozen prepare manifest"
        )

    expected_quality_shard_paths = cast(List[Path], prepare_context["expected_quality_shard_paths"])
    missing_quality_shard_paths = [path_obj for path_obj in expected_quality_shard_paths if not path_obj.exists()]
    if missing_quality_shard_paths:
        raise RuntimeError(
            "PW04 finalize requires all prepared quality shard outputs before reducer execution: "
            f"missing={[normalize_path_value(path_obj) for path_obj in missing_quality_shard_paths]}"
        )

    resolved_paths = cast(Dict[str, Path], prepare_context["resolved_paths"])
    materialized_attack_event_rows = [
        dict(cast(Mapping[str, Any], row))
        for row in cast(List[Mapping[str, Any]], prepare_context["materialized_attack_event_rows"])
    ]
    positive_attack_event_rows = [
        row for row in materialized_attack_event_rows if row.get("sample_role") == ATTACKED_POSITIVE_SAMPLE_ROLE
    ]
    negative_attack_event_rows = [
        row for row in materialized_attack_event_rows if row.get("sample_role") == ATTACKED_NEGATIVE_SAMPLE_ROLE
    ]

    quality_finalize_export = run_pw04_finalize_quality_metrics(
        family_id=family_id,
        quality_pair_plan_path=cast(Path, prepare_context["quality_pair_plan_path"]),
        clean_quality_metrics_path=cast(Path, pw04_paths["clean_quality_metrics_path"]),
        attack_quality_metrics_path=cast(Path, pw04_paths["attack_quality_metrics_path"]),
        quality_shard_paths=expected_quality_shard_paths,
    )
    clean_quality_metrics_payload = cast(Mapping[str, Any], quality_finalize_export["clean_quality_payload"])
    attack_quality_metrics_payload = cast(Mapping[str, Any], quality_finalize_export["attack_quality_payload"])
    quality_lookup = cast(Dict[str, Dict[str, Any]], quality_finalize_export["attack_pair_lookup"])
    attack_quality_overall = cast(Mapping[str, Any], attack_quality_metrics_payload.get("overall", {}))
    for attack_event_row in positive_attack_event_rows:
        attack_event_id = str(attack_event_row["attack_event_id"])
        quality_row = quality_lookup.get(attack_event_id, {})
        attack_event_row["attack_quality_status"] = quality_row.get("status", "unavailable")
        attack_event_row["attack_quality_psnr"] = quality_row.get("psnr")
        attack_event_row["attack_quality_ssim"] = quality_row.get("ssim")
        attack_event_row["attack_quality_lpips"] = quality_row.get("lpips")
        attack_event_row["attack_quality_clip_text_similarity"] = quality_row.get("clip_text_similarity")
        attack_event_row["attack_quality_clip_model_name"] = attack_quality_overall.get("clip_model_name")
    for attack_event_row in negative_attack_event_rows:
        attack_event_row["attack_quality_status"] = "not_applicable"
        attack_event_row["attack_quality_psnr"] = None
        attack_event_row["attack_quality_ssim"] = None
        attack_event_row["attack_quality_lpips"] = None
        attack_event_row["attack_quality_clip_text_similarity"] = None
        attack_event_row["attack_quality_clip_model_name"] = None

    formal_attack_final_decision_metrics_payload = _load_required_json_dict(
        resolved_paths["formal_attack_final_decision_metrics_path"],
        "PW04 formal attack final decision metrics",
    )
    formal_attack_attestation_metrics_payload = _load_required_json_dict(
        resolved_paths["formal_attack_attestation_metrics_path"],
        "PW04 formal attack attestation metrics",
    )
    derived_attack_union_metrics_payload = _load_required_json_dict(
        resolved_paths["derived_attack_union_metrics_path"],
        "PW04 derived attack union metrics",
    )
    formal_attack_negative_metrics_payload = _load_required_json_dict(
        resolved_paths["formal_attack_negative_metrics_path"],
        "PW04 formal attack negative metrics",
    )
    per_attack_family_metrics_payload = _build_per_attack_family_metrics_export(
        family_id=family_id,
        attack_event_rows=positive_attack_event_rows,
    )
    per_attack_condition_metrics_payload = _build_per_attack_condition_metrics_export(
        family_id=family_id,
        attack_event_rows=positive_attack_event_rows,
    )
    write_json_atomic(
        resolved_paths["per_attack_family_metrics_path"],
        per_attack_family_metrics_payload,
    )
    write_json_atomic(
        resolved_paths["per_attack_condition_metrics_path"],
        per_attack_condition_metrics_payload,
    )

    pw02_summary_path = family_root / "runtime_state" / PW02_SUMMARY_FILE_NAME
    pw02_summary = _load_required_json_dict(pw02_summary_path, "PW02 summary")
    clean_formal_metrics_path = family_root / "exports" / "pw02" / "formal_final_decision_metrics.json"
    clean_derived_metrics_path = family_root / "exports" / "pw02" / "derived_system_union_metrics.json"

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
        attack_formal_metrics_path=resolved_paths["formal_attack_final_decision_metrics_path"],
        attack_attestation_metrics_path=resolved_paths["formal_attack_attestation_metrics_path"],
        attack_derived_metrics_path=resolved_paths["derived_attack_union_metrics_path"],
        attack_quality_metrics_path=cast(Path, pw04_paths["attack_quality_metrics_path"]),
        attack_negative_metrics_path=resolved_paths["formal_attack_negative_metrics_path"],
        attack_formal_metrics_payload=formal_attack_final_decision_metrics_payload,
        attack_attestation_metrics_payload=formal_attack_attestation_metrics_payload,
        attack_derived_metrics_payload=derived_attack_union_metrics_payload,
        attack_quality_metrics_payload=attack_quality_metrics_payload,
        attack_negative_metrics_payload=formal_attack_negative_metrics_payload,
    )
    write_json_atomic(cast(Path, pw04_paths["clean_attack_overview_path"]), clean_attack_overview_payload)

    enable_tail_estimation_for_finalize = bool(prepare_context["enable_tail_estimation"])
    paper_exports_payload = build_pw04_paper_exports(
        family_id=family_id,
        family_root=family_root,
        pw02_summary=pw02_summary,
        pw04_paths=pw04_paths,
        attack_event_rows=positive_attack_event_rows,
        attack_negative_event_rows=negative_attack_event_rows,
        per_attack_family_metrics_payload=per_attack_family_metrics_payload,
        per_attack_condition_metrics_payload=per_attack_condition_metrics_payload,
        attack_quality_metrics_payload=attack_quality_metrics_payload,
        enable_tail_estimation=enable_tail_estimation_for_finalize,
    )
    pw04_metrics_extensions = build_pw04_metrics_extensions(
        family_id=family_id,
        family_root=family_root,
        export_root=cast(Path, pw04_paths["export_root"]),
        pw02_summary=pw02_summary,
        attack_event_rows=positive_attack_event_rows,
        clean_quality_metrics_payload=clean_quality_metrics_payload,
        clean_quality_metrics_path=cast(Path, pw04_paths["clean_quality_metrics_path"]),
        attack_quality_metrics_payload=attack_quality_metrics_payload,
        attack_quality_metrics_path=cast(Path, pw04_paths["attack_quality_metrics_path"]),
        main_metrics_summary_csv_path=Path(
            str(cast(Mapping[str, Any], paper_exports_payload["paper_tables_paths"])["main_metrics_summary_csv_path"])
        ).expanduser().resolve(),
        attack_family_summary_paper_csv_path=Path(
            str(cast(Mapping[str, Any], paper_exports_payload["paper_tables_paths"])["attack_family_summary_paper_csv_path"])
        ).expanduser().resolve(),
        attack_condition_summary_paper_csv_path=Path(
            str(cast(Mapping[str, Any], paper_exports_payload["paper_tables_paths"])["attack_condition_summary_paper_csv_path"])
        ).expanduser().resolve(),
        paper_metric_registry_path=Path(str(paper_exports_payload["paper_scope_registry_path"])).expanduser().resolve(),
    )
    analysis_only_artifact_paths = dict(
        cast(Mapping[str, str], pw04_metrics_extensions["analysis_only_artifact_paths"])
    )
    analysis_only_artifact_paths.update(
        {
            "pw04_quality_pair_plan": normalize_path_value(cast(Path, prepare_context["quality_pair_plan_path"])),
            "pw04_quality_finalize_manifest": str(quality_finalize_export["quality_finalize_manifest_path"]),
            "pw04_prepare_manifest": normalize_path_value(cast(Path, prepare_context["prepare_manifest_path"])),
            "pw04_prepared_attack_event_rows": normalize_path_value(cast(Path, prepare_context["prepared_attack_event_rows_path"])),
        }
    )
    analysis_only_artifact_annotations = dict(
        cast(Mapping[str, Mapping[str, Any]], pw04_metrics_extensions["analysis_only_artifact_annotations"])
    )
    analysis_only_artifact_annotations.update(
        {
            "pw04_quality_pair_plan": {"canonical": False, "analysis_only": True},
            "pw04_quality_finalize_manifest": {"canonical": False, "analysis_only": True},
            "pw04_prepare_manifest": {"canonical": False, "analysis_only": True},
            "pw04_prepared_attack_event_rows": {"canonical": False, "analysis_only": True},
        }
    )
    pw04_metrics_extensions["analysis_only_artifact_paths"] = analysis_only_artifact_paths
    pw04_metrics_extensions["analysis_only_artifact_annotations"] = analysis_only_artifact_annotations
    paper_exports_payload.update(pw04_metrics_extensions)
    paper_exports_payload.update(
        {
            "pw04_mode": resolved_mode,
            "prepare_manifest_path": normalize_path_value(cast(Path, prepare_context["prepare_manifest_path"])),
            "prepared_attack_event_rows_path": normalize_path_value(cast(Path, prepare_context["prepared_attack_event_rows_path"])),
            "clean_quality_metrics_path": str(quality_finalize_export["clean_quality_metrics_path"]),
            "quality_pair_plan_path": normalize_path_value(cast(Path, prepare_context["quality_pair_plan_path"])),
            "quality_shard_paths": [normalize_path_value(path_obj) for path_obj in expected_quality_shard_paths],
            "quality_finalize_manifest_path": str(quality_finalize_export["quality_finalize_manifest_path"]),
            "quality_shard_count": len(expected_quality_shard_paths),
        }
    )

    summary_payload = _build_pw04_summary_payload(
        family_id=family_id,
        family_root=family_root,
        summary_path=cast(Path, pw04_paths["summary_path"]),
        attack_merge_manifest_path=resolved_paths["attack_merge_manifest_path"],
        attack_pool_manifest_path=resolved_paths["attack_positive_pool_manifest_path"],
        attack_negative_pool_manifest_path=resolved_paths["attack_negative_pool_manifest_path"],
        formal_attack_final_decision_metrics_path=resolved_paths["formal_attack_final_decision_metrics_path"],
        formal_attack_attestation_metrics_path=resolved_paths["formal_attack_attestation_metrics_path"],
        derived_attack_union_metrics_path=resolved_paths["derived_attack_union_metrics_path"],
        formal_attack_negative_metrics_path=resolved_paths["formal_attack_negative_metrics_path"],
        per_attack_family_metrics_path=resolved_paths["per_attack_family_metrics_path"],
        per_attack_condition_metrics_path=resolved_paths["per_attack_condition_metrics_path"],
        attack_quality_metrics_path=cast(Path, pw04_paths["attack_quality_metrics_path"]),
        attack_event_table_path=cast(Path, pw04_paths["attack_event_table_path"]),
        attack_family_summary_csv_path=cast(Path, pw04_paths["attack_family_summary_csv_path"]),
        attack_condition_summary_csv_path=cast(Path, pw04_paths["attack_condition_summary_csv_path"]),
        clean_attack_overview_path=cast(Path, pw04_paths["clean_attack_overview_path"]),
        expected_attack_shard_count=int(cast(Mapping[str, Any], prepare_context["prepare_manifest"])["expected_attack_shard_count"]),
        discovered_attack_shard_count=int(cast(Mapping[str, Any], prepare_context["prepare_manifest"])["discovered_attack_shard_count"]),
        expected_attack_event_count=int(cast(Mapping[str, Any], prepare_context["prepare_manifest"])["expected_attack_event_count"]),
        discovered_attack_event_count=len(attack_event_table_rows),
        completed_attack_event_count=len(materialized_attack_event_rows),
        attacked_positive_event_count=len(positive_attack_event_rows),
        attacked_negative_event_count=len(negative_attack_event_rows),
        attack_family_count=len({str(row["attack_family"]) for row in materialized_attack_event_rows}),
        parent_event_count=len({str(row["parent_event_id"]) for row in materialized_attack_event_rows}),
        formal_record_count=len(materialized_attack_event_rows),
        enable_tail_estimation=enable_tail_estimation_for_finalize,
        paper_exports_payload=paper_exports_payload,
    )
    summary_payload["pw04_mode"] = resolved_mode
    write_json_atomic(summary_path, summary_payload)
    return summary_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the PW04 CLI argument parser.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Execute one explicit PW04 mode over the frozen prepare/quality_shard/finalize flow, "
            "with optional tail estimation exports at finalize time."
        )
    )
    parser.add_argument("--drive-project-root", required=True, help="Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Family identifier.")
    parser.add_argument(
        "--pw04-mode",
        required=True,
        choices=list(PW04_MODE_CHOICES),
        help="Explicit PW04 mode: prepare, quality_shard, or finalize.",
    )
    parser.add_argument(
        "--quality-shard-index",
        type=int,
        default=None,
        help="Zero-based quality shard index required for quality_shard mode.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Clear existing PW04 outputs before rerun.")
    parser.add_argument(
        "--enable-tail-estimation",
        action="store_true",
        help="Emit optional tail-estimation artifacts in addition to empirical clean FPR exports.",
    )
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
        enable_tail_estimation=bool(args.enable_tail_estimation),
        pw04_mode=str(args.pw04_mode),
        quality_shard_index=args.quality_shard_index,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())