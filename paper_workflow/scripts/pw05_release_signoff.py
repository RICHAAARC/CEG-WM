"""
File purpose: Build PW05 release package and signoff outputs from finalized paper_workflow artifacts.
Module type: General module
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from paper_workflow.scripts.pw_common import build_family_root
from scripts.notebook_runtime_common import (
    collect_file_index,
    collect_git_summary,
    collect_python_summary,
    compute_file_sha256,
    compute_mapping_sha256,
    ensure_directory,
    finalize_stage_package,
    make_stage_run_id,
    normalize_path_value,
    stage_relative_copy,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
)


STAGE_NAME = "PW05_Release_And_Signoff"
SCHEMA_VERSION = "pw_stage_05_v1"
PW04_SUMMARY_FILE_NAME = "pw04_summary.json"
PW05_SUMMARY_FILE_NAME = "pw05_summary.json"
ALLOW_FREEZE = "ALLOW_FREEZE"
BLOCK_FREEZE = "BLOCK_FREEZE"
FORMAL_SIGNOFF_PASS_STATUS = "passed"
FORMAL_SIGNOFF_BLOCK_STATUS = "blocked"


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取必需 JSON 对象文件。

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


def _resolve_path_value_under_family_root(path_value: Any, family_root: Path, label: str) -> Path:
    """
    功能：把 summary 中的路径解析并约束到 family_root 内。

    Resolve a summary-provided path and constrain it under the family root.

    Args:
        path_value: Raw path-like value.
        family_root: Family root path.
        label: Human-readable label.

    Returns:
        Resolved file path.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} must be non-empty path string")

    candidate_path = Path(path_value.strip()).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (family_root / candidate_path).resolve()
    else:
        candidate_path = candidate_path.resolve()

    validate_path_within_base(family_root, candidate_path, label)
    if not candidate_path.exists() or not candidate_path.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(candidate_path)}")
    return candidate_path


def _resolve_pw05_paths(family_root: Path) -> Dict[str, Path]:
    """
    功能：解析 PW05 的规范输出路径。

    Resolve canonical PW05 output paths.

    Args:
        family_root: Family root path.

    Returns:
        PW05 path mapping.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")

    runtime_state_root = family_root / "runtime_state"
    export_root = family_root / "exports" / "pw05"
    package_staging_root = runtime_state_root / "pw05_package_staging"
    paths = {
        "runtime_state_root": runtime_state_root,
        "export_root": export_root,
        "package_staging_root": package_staging_root,
        "summary_path": runtime_state_root / PW05_SUMMARY_FILE_NAME,
        "signoff_report_path": export_root / "signoff_report.json",
        "release_manifest_path": export_root / "release_manifest.json",
        "workflow_summary_path": export_root / "workflow_summary.json",
        "run_closure_path": export_root / "run_closure.json",
        "stage_manifest_path": export_root / "stage_manifest.json",
        "package_manifest_path": export_root / "package_manifest.json",
    }
    for path_obj in paths.values():
        validate_path_within_base(family_root, path_obj, "PW05 path")
    return paths


def _prepare_pw05_outputs(
    *,
    family_root: Path,
    export_root: Path,
    summary_path: Path,
    package_staging_root: Path,
    force_rerun: bool,
) -> None:
    """
    功能：按显式 rerun 语义准备 PW05 输出目录。

    Prepare PW05 output locations with explicit rerun semantics.

    Args:
        family_root: Family root path.
        export_root: PW05 export root.
        summary_path: PW05 summary path.
        package_staging_root: Temporary package staging root.
        force_rerun: Whether existing outputs may be cleared.

    Returns:
        None.
    """
    if export_root.exists() or summary_path.exists():
        if not force_rerun:
            raise RuntimeError(
                "PW05 outputs already exist; rerun requires force_rerun: "
                f"export_root={normalize_path_value(export_root)}"
            )
        if export_root.exists():
            shutil.rmtree(export_root)
        if summary_path.exists():
            summary_path.unlink()

    if package_staging_root.exists():
        shutil.rmtree(package_staging_root)

    ensure_directory(family_root / "runtime_state")
    ensure_directory(export_root)
    ensure_directory(package_staging_root)


def _resolve_signoff_statuses(decision: str) -> Dict[str, str]:
    """
    功能：把 freeze decision 映射为正式 signoff 结论。

    Resolve formal signoff status tokens from one decision value.

    Args:
        decision: Freeze decision token.

    Returns:
        Mapping with signoff_status, release_status, and paper_closure_status.
    """
    if not isinstance(decision, str) or not decision:
        raise TypeError("decision must be non-empty str")
    if decision == ALLOW_FREEZE:
        status_value = FORMAL_SIGNOFF_PASS_STATUS
    else:
        status_value = FORMAL_SIGNOFF_BLOCK_STATUS
    return {
        "signoff_status": status_value,
        "release_status": status_value,
        "paper_closure_status": status_value,
    }


def _build_canonical_source_paths(family_root: Path) -> Dict[str, Path]:
    """
    功能：构造 PW05 必须打包的规范源工件路径。

    Build the canonical source artifact paths that PW05 must release.

    Args:
        family_root: Family root path.

    Returns:
        Label-to-path mapping for required source artifacts.
    """
    pw04_root = family_root / "exports" / "pw04"
    pw04_metrics_root = pw04_root / "metrics"
    pw04_tables_root = pw04_root / "tables"
    return {
        "family_manifest": family_root / "manifests" / "paper_eval_family_manifest.json",
        "config_snapshot": family_root / "snapshots" / "config_snapshot.yaml",
        "pw02_finalize_manifest": family_root / "exports" / "pw02" / "paper_source_finalize_manifest.json",
        "pw02_content_threshold_export": family_root / "exports" / "pw02" / "thresholds" / "content" / "thresholds.json",
        "pw02_attestation_threshold_export": family_root / "exports" / "pw02" / "thresholds" / "attestation" / "thresholds.json",
        "pw04_summary": family_root / "runtime_state" / PW04_SUMMARY_FILE_NAME,
        "pw04_attack_merge_manifest": pw04_root / "manifests" / "attack_merge_manifest.json",
        "pw04_attack_positive_pool_manifest": pw04_root / "attack_positive_pool_manifest.json",
        "pw04_attack_negative_pool_manifest": pw04_root / "attack_negative_pool_manifest.json",
        "pw04_formal_attack_final_decision_metrics": pw04_root / "formal_attack_final_decision_metrics.json",
        "pw04_formal_attack_attestation_metrics": pw04_root / "formal_attack_attestation_metrics.json",
        "pw04_derived_attack_union_metrics": pw04_root / "derived_attack_union_metrics.json",
        "pw04_formal_attack_negative_metrics": pw04_root / "formal_attack_negative_metrics.json",
        "pw04_clean_attack_overview": pw04_root / "clean_attack_overview.json",
        "pw04_paper_metric_registry": pw04_metrics_root / "paper_metric_registry.json",
        "pw04_content_chain_metrics": pw04_metrics_root / "content_chain_metrics.json",
        "pw04_event_attestation_metrics": pw04_metrics_root / "event_attestation_metrics.json",
        "pw04_system_final_metrics": pw04_metrics_root / "system_final_metrics.json",
        "pw04_bootstrap_confidence_intervals": pw04_metrics_root / "bootstrap_confidence_intervals.json",
        "pw04_main_metrics_summary_csv": pw04_tables_root / "main_metrics_summary.csv",
        "pw04_attack_family_summary_paper_csv": pw04_tables_root / "attack_family_summary_paper.csv",
        "pw04_attack_condition_summary_paper_csv": pw04_tables_root / "attack_condition_summary_paper.csv",
        "pw04_rescue_metrics_summary_csv": pw04_tables_root / "rescue_metrics_summary.csv",
        "pw04_bootstrap_confidence_intervals_csv": pw04_tables_root / "bootstrap_confidence_intervals.csv",
    }


def _require_existing_file(path_obj: Path, label: str, family_root: Path) -> None:
    """
    功能：校验源工件文件存在且受 family_root 约束。

    Validate that one source artifact exists under the family root.

    Args:
        path_obj: Candidate file path.
        label: Human-readable label.
        family_root: Family root path.

    Returns:
        None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    validate_path_within_base(family_root, path_obj, label)
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")


def _require_summary_binding(
    *,
    pw04_summary: Mapping[str, Any],
    family_root: Path,
    field_name: str,
    expected_path: Path,
) -> Path:
    """
    功能：校验 PW04 summary 中的路径字段绑定到规范输出。

    Validate that one PW04 summary path field binds the canonical output path.

    Args:
        pw04_summary: PW04 summary mapping.
        family_root: Family root path.
        field_name: Summary field name.
        expected_path: Canonical expected path.

    Returns:
        Resolved summary path.
    """
    _require_existing_file(expected_path, field_name, family_root)
    resolved_path = _resolve_path_value_under_family_root(
        pw04_summary.get(field_name),
        family_root,
        field_name,
    )
    if normalize_path_value(resolved_path) != normalize_path_value(expected_path):
        raise ValueError(
            f"PW04 summary {field_name} must bind canonical path: "
            f"expected={normalize_path_value(expected_path)}, actual={normalize_path_value(resolved_path)}"
        )
    return resolved_path


def _collect_release_source_paths(
    *,
    family_id: str,
    family_root: Path,
    pw04_summary: Mapping[str, Any],
) -> Dict[str, Path]:
    """
    功能：收集并校验 PW05 release 所需的全部源工件。

    Collect and validate all source artifacts required for the PW05 release.

    Args:
        family_id: Family identifier.
        family_root: Family root path.
        pw04_summary: PW04 summary mapping.

    Returns:
        Fully validated label-to-path mapping.
    """
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw04_summary, Mapping):
        raise TypeError("pw04_summary must be Mapping")

    summary_family_id = pw04_summary.get("family_id")
    if isinstance(summary_family_id, str) and summary_family_id and summary_family_id != family_id:
        raise ValueError(
            "PW04 summary family_id mismatch: "
            f"expected={family_id}, actual={summary_family_id}"
        )
    if pw04_summary.get("status") != "completed":
        raise ValueError(f"PW04 summary must be completed before PW05: status={pw04_summary.get('status')}")
    if pw04_summary.get("paper_exports_completed") is not True:
        raise ValueError("PW04 summary must confirm completed paper exports before PW05")

    source_paths = _build_canonical_source_paths(family_root)
    for label, path_obj in source_paths.items():
        _require_existing_file(path_obj, label, family_root)

    fixed_bindings = {
        "attack_merge_manifest_path": "pw04_attack_merge_manifest",
        "attack_positive_pool_manifest_path": "pw04_attack_positive_pool_manifest",
        "attack_negative_pool_manifest_path": "pw04_attack_negative_pool_manifest",
        "formal_attack_final_decision_metrics_path": "pw04_formal_attack_final_decision_metrics",
        "formal_attack_attestation_metrics_path": "pw04_formal_attack_attestation_metrics",
        "derived_attack_union_metrics_path": "pw04_derived_attack_union_metrics",
        "formal_attack_negative_metrics_path": "pw04_formal_attack_negative_metrics",
        "clean_attack_overview_path": "pw04_clean_attack_overview",
        "paper_scope_registry_path": "pw04_paper_metric_registry",
        "bootstrap_confidence_intervals_path": "pw04_bootstrap_confidence_intervals",
        "bootstrap_confidence_intervals_csv_path": "pw04_bootstrap_confidence_intervals_csv",
    }
    for field_name, label in fixed_bindings.items():
        _require_summary_binding(
            pw04_summary=pw04_summary,
            family_root=family_root,
            field_name=field_name,
            expected_path=source_paths[label],
        )

    canonical_metrics_paths = pw04_summary.get("canonical_metrics_paths")
    if not isinstance(canonical_metrics_paths, Mapping):
        raise ValueError("PW04 summary missing canonical_metrics_paths")
    for metric_key, label in {
        "content_chain": "pw04_content_chain_metrics",
        "event_attestation": "pw04_event_attestation_metrics",
        "system_final": "pw04_system_final_metrics",
    }.items():
        resolved_path = _resolve_path_value_under_family_root(
            canonical_metrics_paths.get(metric_key),
            family_root,
            f"canonical_metrics_paths.{metric_key}",
        )
        if normalize_path_value(resolved_path) != normalize_path_value(source_paths[label]):
            raise ValueError(
                "PW04 canonical metric path mismatch: "
                f"key={metric_key}, expected={normalize_path_value(source_paths[label])}, "
                f"actual={normalize_path_value(resolved_path)}"
            )

    paper_tables_paths = pw04_summary.get("paper_tables_paths")
    if not isinstance(paper_tables_paths, Mapping):
        raise ValueError("PW04 summary missing paper_tables_paths")
    for table_key, label in {
        "main_metrics_summary_csv_path": "pw04_main_metrics_summary_csv",
        "attack_family_summary_paper_csv_path": "pw04_attack_family_summary_paper_csv",
        "attack_condition_summary_paper_csv_path": "pw04_attack_condition_summary_paper_csv",
        "rescue_metrics_summary_csv_path": "pw04_rescue_metrics_summary_csv",
    }.items():
        resolved_path = _resolve_path_value_under_family_root(
            paper_tables_paths.get(table_key),
            family_root,
            f"paper_tables_paths.{table_key}",
        )
        if normalize_path_value(resolved_path) != normalize_path_value(source_paths[label]):
            raise ValueError(
                "PW04 paper table path mismatch: "
                f"key={table_key}, expected={normalize_path_value(source_paths[label])}, "
                f"actual={normalize_path_value(resolved_path)}"
            )

    paper_figures_paths = pw04_summary.get("paper_figures_paths")
    if not isinstance(paper_figures_paths, Mapping) or not paper_figures_paths:
        raise ValueError("PW04 summary missing paper_figures_paths")
    for key_name, path_value in sorted(paper_figures_paths.items()):
        resolved_path = _resolve_path_value_under_family_root(
            path_value,
            family_root,
            f"paper_figures_paths.{key_name}",
        )
        source_paths[f"pw04_figure_{key_name}"] = resolved_path

    tail_estimation_paths = pw04_summary.get("tail_estimation_paths")
    if not isinstance(tail_estimation_paths, Mapping) or not tail_estimation_paths:
        raise ValueError("PW04 summary missing tail_estimation_paths")
    for key_name, path_value in sorted(tail_estimation_paths.items()):
        resolved_path = _resolve_path_value_under_family_root(
            path_value,
            family_root,
            f"tail_estimation_paths.{key_name}",
        )
        source_paths[f"pw04_tail_{key_name}"] = resolved_path

    return source_paths


def _collect_analysis_only_source_bindings(
    *,
    family_root: Path,
    pw04_summary: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    功能：收集 PW04 summary 中声明的 analysis-only 源工件绑定。

    Collect analysis-only source artifact bindings declared by the PW04 summary.

    Args:
        family_root: Family root path.
        pw04_summary: PW04 summary payload.

    Returns:
        Label-to-binding mapping containing resolved path and release flags.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(pw04_summary, Mapping):
        raise TypeError("pw04_summary must be Mapping")

    analysis_only_paths = pw04_summary.get("analysis_only_artifact_paths")
    if not isinstance(analysis_only_paths, Mapping):
        return {}
    analysis_only_annotations = pw04_summary.get("analysis_only_artifact_annotations")
    annotation_mapping = analysis_only_annotations if isinstance(analysis_only_annotations, Mapping) else {}

    bindings: Dict[str, Dict[str, Any]] = {}
    for label, path_value in analysis_only_paths.items():
        if not isinstance(label, str) or not label:
            continue
        resolved_path = _resolve_path_value_under_family_root(path_value, family_root, f"analysis_only_artifact_paths.{label}")
        annotation_node = annotation_mapping.get(label) if isinstance(annotation_mapping, Mapping) else None
        annotation_payload = annotation_node if isinstance(annotation_node, Mapping) else {}
        bindings[label] = {
            "path": resolved_path,
            "canonical": bool(annotation_payload.get("canonical", False)),
            "analysis_only": bool(annotation_payload.get("analysis_only", True)),
        }
    return bindings


def _build_release_copy_paths(family_root: Path, source_paths: Mapping[str, Path]) -> Dict[str, str]:
    """
    功能：构造 release package 内的稳定复制路径。

    Build stable copy destinations for source artifacts inside the release package.

    Args:
        family_root: Family root path.
        source_paths: Validated source artifact mapping.

    Returns:
        Label-to-relative-destination mapping.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(source_paths, Mapping):
        raise TypeError("source_paths must be Mapping")

    copy_paths: Dict[str, str] = {}
    family_root_resolved = family_root.resolve()
    for label, source_path in source_paths.items():
        if not isinstance(source_path, Path):
            raise TypeError(f"source_paths[{label}] must be Path")
        validate_path_within_base(family_root, source_path, str(label))
        relative_source_path = source_path.resolve().relative_to(family_root_resolved).as_posix()
        copy_paths[str(label)] = f"source/{relative_source_path}"
    return copy_paths


def _package_stage_outputs(
    *,
    package_root: Path,
    generated_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
    release_copy_paths: Mapping[str, str],
) -> Path:
    """
    功能：把 PW05 生成工件与源工件复制到 package staging 目录。

    Copy PW05 generated artifacts and source artifacts into the package staging root.

    Args:
        package_root: Package staging root.
        generated_paths: Generated artifact mapping.
        source_paths: Source artifact mapping.
        release_copy_paths: Package-relative destinations for source artifacts.

    Returns:
        Prepared package staging root.
    """
    if package_root.exists():
        shutil.rmtree(package_root)
    ensure_directory(package_root)

    generated_copy_map = {
        "signoff_report": "artifacts/signoff/signoff_report.json",
        "release_manifest": "artifacts/release/release_manifest.json",
        "workflow_summary": "artifacts/workflow_summary.json",
        "run_closure": "artifacts/run_closure.json",
        "stage_manifest": "artifacts/stage_manifest.json",
    }
    for label, relative_destination in generated_copy_map.items():
        stage_relative_copy(cast(Path, generated_paths[label]), package_root, relative_destination)

    for label, source_path in source_paths.items():
        relative_destination = release_copy_paths.get(str(label))
        if not isinstance(relative_destination, str) or not relative_destination:
            raise ValueError(f"release_copy_paths missing {label}")
        stage_relative_copy(source_path, package_root, relative_destination)

    return package_root


def run_pw05_release_signoff(
    *,
    drive_project_root: Path,
    family_id: str,
    stage_run_id: str | None = None,
    notebook_name: str = STAGE_NAME,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行 PW05 release 与 signoff。

    Run the PW05 release and signoff stage for one finalized paper_workflow family.

    Args:
        drive_project_root: Drive project root path.
        family_id: Family identifier.
        stage_run_id: Optional fixed stage run identifier.
        notebook_name: Notebook display name.
        force_rerun: Whether existing PW05 outputs may be replaced.

    Returns:
        PW05 summary payload.
    """
    if not isinstance(drive_project_root, Path):
        raise TypeError("drive_project_root must be Path")
    if not isinstance(family_id, str) or not family_id.strip():
        raise TypeError("family_id must be non-empty str")
    if stage_run_id is not None and (not isinstance(stage_run_id, str) or not stage_run_id.strip()):
        raise TypeError("stage_run_id must be non-empty str when provided")
    if not isinstance(notebook_name, str) or not notebook_name.strip():
        raise TypeError("notebook_name must be non-empty str")
    if not isinstance(force_rerun, bool):
        raise TypeError("force_rerun must be bool")

    normalized_drive_root = drive_project_root.expanduser().resolve()
    family_root = build_family_root(normalized_drive_root, family_id)
    pw05_paths = _resolve_pw05_paths(family_root)
    _prepare_pw05_outputs(
        family_root=family_root,
        export_root=cast(Path, pw05_paths["export_root"]),
        summary_path=cast(Path, pw05_paths["summary_path"]),
        package_staging_root=cast(Path, pw05_paths["package_staging_root"]),
        force_rerun=force_rerun,
    )

    pw04_summary_path = family_root / "runtime_state" / PW04_SUMMARY_FILE_NAME
    pw04_summary = _load_required_json_dict(pw04_summary_path, "PW04 summary")
    canonical_source_paths = _collect_release_source_paths(
        family_id=family_id,
        family_root=family_root,
        pw04_summary=pw04_summary,
    )
    analysis_only_source_bindings = _collect_analysis_only_source_bindings(
        family_root=family_root,
        pw04_summary=pw04_summary,
    )
    analysis_only_source_paths = {
        label: cast(Path, binding["path"])
        for label, binding in analysis_only_source_bindings.items()
    }
    source_paths = {**canonical_source_paths, **analysis_only_source_paths}
    source_artifact_index = collect_file_index(family_root, source_paths)
    release_copy_paths = _build_release_copy_paths(family_root, source_paths)
    analysis_only_release_annotations = {
        label: {
            "source_path": normalize_path_value(cast(Path, binding["path"])),
            "release_copy_path": release_copy_paths[label],
            "canonical": bool(binding["canonical"]),
            "analysis_only": bool(binding["analysis_only"]),
        }
        for label, binding in analysis_only_source_bindings.items()
    }

    decision = ALLOW_FREEZE
    status_payload = _resolve_signoff_statuses(decision)
    resolved_stage_run_id = stage_run_id or make_stage_run_id(STAGE_NAME)

    signoff_report_path = cast(Path, pw05_paths["signoff_report_path"])
    release_manifest_path = cast(Path, pw05_paths["release_manifest_path"])
    workflow_summary_path = cast(Path, pw05_paths["workflow_summary_path"])
    run_closure_path = cast(Path, pw05_paths["run_closure_path"])
    stage_manifest_path = cast(Path, pw05_paths["stage_manifest_path"])
    package_manifest_path = cast(Path, pw05_paths["package_manifest_path"])
    export_root = cast(Path, pw05_paths["export_root"])
    summary_path = cast(Path, pw05_paths["summary_path"])
    package_staging_root = cast(Path, pw05_paths["package_staging_root"])

    signoff_report = {
        "signoff_report_version": "v1",
        "artifact_type": "paper_workflow_pw05_signoff_report",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "source_stage_name": pw04_summary.get("stage_name", "PW04_Attack_Merge_And_Metrics"),
        "source_stage_run_id": pw04_summary.get("stage_run_id", "<absent>"),
        "source_stage_summary_path": normalize_path_value(pw04_summary_path),
        "decision": decision,
        "signoff_status": status_payload["signoff_status"],
        "release_status": status_payload["release_status"],
        "paper_closure_status": status_payload["paper_closure_status"],
        "blocking_reason_count": 0,
        "blocking_reasons": [],
        "checked_source_artifact_count": len(source_artifact_index),
        "analysis_only_artifact_count": len(analysis_only_release_annotations),
        "checked_source_artifacts": source_artifact_index,
        "pw04_summary_anchor": {
            "status": pw04_summary.get("status"),
            "paper_exports_completed": pw04_summary.get("paper_exports_completed"),
            "completed_attack_event_count": pw04_summary.get("completed_attack_event_count"),
            "paper_scope_registry_path": pw04_summary.get("paper_scope_registry_path"),
            "bootstrap_confidence_intervals_path": pw04_summary.get("bootstrap_confidence_intervals_path"),
        },
        "pw04_summary_digest": compute_mapping_sha256(pw04_summary),
        "family_manifest_sha256": compute_file_sha256(cast(Path, source_paths["family_manifest"])),
        "created_at": utc_now_iso(),
    }
    write_json_atomic(signoff_report_path, signoff_report)

    release_manifest = {
        "release_manifest_version": "v1",
        "artifact_type": "paper_workflow_pw05_release_manifest",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "source_stage_name": pw04_summary.get("stage_name", "PW04_Attack_Merge_And_Metrics"),
        "source_stage_run_id": pw04_summary.get("stage_run_id", "<absent>"),
        "source_stage_summary_path": normalize_path_value(pw04_summary_path),
        "decision": decision,
        "signoff_status": status_payload["signoff_status"],
        "release_status": status_payload["release_status"],
        "paper_closure_status": status_payload["paper_closure_status"],
        "release_scope": {
            "package_mode": "append_only_frozen_release",
            "includes_manifests": True,
            "includes_thresholds": True,
            "includes_metrics": True,
            "includes_tables": True,
            "includes_figures": True,
            "includes_tail_estimation": True,
            "includes_analysis_only_artifacts": bool(analysis_only_release_annotations),
        },
        "source_artifact_index": source_artifact_index,
        "release_copy_paths": release_copy_paths,
        "analysis_only_artifact_annotations": analysis_only_release_annotations,
        "created_at": utc_now_iso(),
    }
    write_json_atomic(release_manifest_path, release_manifest)

    workflow_summary = {
        "artifact_type": "paper_workflow_pw05_workflow_summary",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "notebook_name": notebook_name,
        "family_id": family_id,
        "decision": decision,
        "signoff_status": status_payload["signoff_status"],
        "release_status": status_payload["release_status"],
        "paper_closure_status": status_payload["paper_closure_status"],
        "source_stage_name": pw04_summary.get("stage_name", "PW04_Attack_Merge_And_Metrics"),
        "source_stage_run_id": pw04_summary.get("stage_run_id", "<absent>"),
        "source_stage_summary_path": normalize_path_value(pw04_summary_path),
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "checked_source_artifact_count": len(source_artifact_index),
        "analysis_only_artifact_count": len(analysis_only_release_annotations),
        "created_at": utc_now_iso(),
    }
    write_json_atomic(workflow_summary_path, workflow_summary)

    run_closure = {
        "artifact_type": "paper_workflow_pw05_run_closure",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "family_id": family_id,
        "decision": decision,
        "signoff_status": status_payload["signoff_status"],
        "release_status": status_payload["release_status"],
        "paper_closure_status": status_payload["paper_closure_status"],
        "status": {
            "ok": True,
            "reason": "allow_freeze",
            "details": {
                "checked_source_artifact_count": len(source_artifact_index),
                "blocking_reason_count": 0,
            },
        },
        "created_at": utc_now_iso(),
    }
    write_json_atomic(run_closure_path, run_closure)

    generated_artifact_paths = {
        "signoff_report": signoff_report_path,
        "release_manifest": release_manifest_path,
        "workflow_summary": workflow_summary_path,
        "run_closure": run_closure_path,
    }
    generated_artifact_index = collect_file_index(family_root, generated_artifact_paths)

    stage_manifest = {
        "artifact_type": "paper_workflow_pw05_stage_manifest",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "source_stage_name": pw04_summary.get("stage_name", "PW04_Attack_Merge_And_Metrics"),
        "source_stage_run_id": pw04_summary.get("stage_run_id", "<absent>"),
        "source_stage_summary_path": normalize_path_value(pw04_summary_path),
        "export_root": normalize_path_value(export_root),
        "summary_path": normalize_path_value(summary_path),
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "workflow_summary_path": normalize_path_value(workflow_summary_path),
        "run_closure_path": normalize_path_value(run_closure_path),
        "source_artifact_index": source_artifact_index,
        "generated_artifact_index": generated_artifact_index,
        "release_copy_paths": release_copy_paths,
        "analysis_only_artifact_annotations": analysis_only_release_annotations,
        "git": collect_git_summary(REPO_ROOT),
        "python": collect_python_summary(),
        "created_at": utc_now_iso(),
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    generated_paths_for_package = dict(generated_artifact_paths)
    generated_paths_for_package["stage_manifest"] = stage_manifest_path
    _package_stage_outputs(
        package_root=package_staging_root,
        generated_paths=generated_paths_for_package,
        source_paths=source_paths,
        release_copy_paths=release_copy_paths,
    )

    package_manifest = finalize_stage_package(
        stage_name=STAGE_NAME,
        stage_run_id=resolved_stage_run_id,
        package_root=package_staging_root,
        export_root=export_root,
        source_stage_run_id=(
            str(pw04_summary.get("stage_run_id"))
            if isinstance(pw04_summary.get("stage_run_id"), str) and str(pw04_summary.get("stage_run_id")).strip()
            else None
        ),
        source_stage_package_path=None,
        package_manifest_path=package_manifest_path,
    )

    final_generated_artifact_paths = {
        **generated_paths_for_package,
        "package_manifest": package_manifest_path,
        "package_index": export_root / "package_index.json",
        "package_zip": Path(str(package_manifest["package_path"])),
    }
    final_generated_artifact_index = collect_file_index(family_root, final_generated_artifact_paths)
    summary = {
        "artifact_type": "paper_workflow_pw05_summary",
        "schema_version": SCHEMA_VERSION,
        "stage_name": STAGE_NAME,
        "stage_run_id": resolved_stage_run_id,
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "source_stage_name": pw04_summary.get("stage_name", "PW04_Attack_Merge_And_Metrics"),
        "source_stage_run_id": pw04_summary.get("stage_run_id", "<absent>"),
        "pw04_summary_path": normalize_path_value(pw04_summary_path),
        "summary_path": normalize_path_value(summary_path),
        "export_root": normalize_path_value(export_root),
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "workflow_summary_path": normalize_path_value(workflow_summary_path),
        "run_closure_path": normalize_path_value(run_closure_path),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "package_index_path": normalize_path_value(export_root / "package_index.json"),
        "package_path": package_manifest["package_path"],
        "package_sha256": package_manifest["package_sha256"],
        "package_filename": package_manifest["package_filename"],
        "package_manifest_digest": compute_mapping_sha256(package_manifest),
        "decision": decision,
        "signoff_status": status_payload["signoff_status"],
        "release_status": status_payload["release_status"],
        "paper_closure_status": status_payload["paper_closure_status"],
        "release_copy_paths": release_copy_paths,
        "analysis_only_artifact_paths": {
            label: normalize_path_value(path_obj)
            for label, path_obj in analysis_only_source_paths.items()
        },
        "analysis_only_artifact_annotations": analysis_only_release_annotations,
        "source_artifact_index": source_artifact_index,
        "generated_artifact_index": final_generated_artifact_index,
        "status": "completed",
        "created_at": utc_now_iso(),
    }
    write_json_atomic(summary_path, summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    功能：构造 PW05 CLI 参数解析器。

    Build the PW05 CLI argument parser.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Build the PW05 paper_workflow release package and signoff outputs.")
    parser.add_argument("--drive-project-root", required=True, help="Drive project root path.")
    parser.add_argument("--family-id", required=True, help="Family identifier.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    parser.add_argument("--notebook-name", default=STAGE_NAME, help="Notebook display name.")
    parser.add_argument("--force-rerun", action="store_true", help="Clear existing PW05 outputs before rerun.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    功能：执行 PW05 CLI 入口。

    Execute the PW05 CLI entrypoint.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process-style exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_pw05_release_signoff(
        drive_project_root=Path(args.drive_project_root),
        family_id=str(args.family_id),
        stage_run_id=str(args.stage_run_id) if isinstance(args.stage_run_id, str) and args.stage_run_id.strip() else None,
        notebook_name=str(args.notebook_name),
        force_rerun=bool(args.force_rerun),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())