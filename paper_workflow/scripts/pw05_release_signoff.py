"""
File purpose: Build PW05 release package and signoff outputs from finalized paper_workflow artifacts.
Module type: General module
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from paper_workflow.scripts.pw_common import ACTIVE_SAMPLE_ROLE, build_family_root
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
        "formal_run_readiness_report_path": export_root / "formal_run_readiness_report.json",
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
        "pw04_attack_quality_metrics": pw04_metrics_root / "attack_quality_metrics.json",
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
        "attack_quality_metrics_path": "pw04_attack_quality_metrics",
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

    source_paths.update(
        _collect_payload_sidecar_source_paths(
            family_root=family_root,
            source_paths=source_paths,
        )
    )

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


def _extract_mapping(node: Any) -> Dict[str, Any]:
    """
    功能：安全提取映射节点。 

    Safely coerce one optional mapping-like node into a plain dict.

    Args:
        node: Candidate mapping node.

    Returns:
        Plain dict when the input is mapping-like; otherwise empty dict.
    """
    return dict(cast(Mapping[str, Any], node)) if isinstance(node, Mapping) else {}


def _coerce_non_negative_int(value: Any) -> int | None:
    """
    功能：把输入解析为非负整数。 

    Parse one optional non-negative integer value.

    Args:
        value: Candidate numeric value.

    Returns:
        Parsed non-negative integer, or None when unavailable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0.0:
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed_value = int(float(value))
        except ValueError:
            return None
        return parsed_value if parsed_value >= 0 else None
    return None


def _resolve_analysis_only_binding_path(
    analysis_only_bindings: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Path | None:
    """
    功能：解析 analysis-only 绑定中的文件路径。 

    Resolve one analysis-only artifact path from the collected binding mapping.

    Args:
        analysis_only_bindings: Analysis-only binding mapping.
        label: Stable artifact label.

    Returns:
        Resolved path when present; otherwise None.
    """
    if not isinstance(analysis_only_bindings, Mapping):
        raise TypeError("analysis_only_bindings must be Mapping")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    binding = analysis_only_bindings.get(label)
    if not isinstance(binding, Mapping):
        return None
    path_obj = binding.get("path")
    return path_obj if isinstance(path_obj, Path) else None


def _load_optional_json_dict(path_obj: Path | None, label: str) -> Dict[str, Any] | None:
    """
    功能：按存在性读取可选 JSON 对象文件。 

    Load one optional JSON object file when the path exists.

    Args:
        path_obj: Optional JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping, or None when the path is absent.
    """
    if path_obj is None:
        return None
    return _load_required_json_dict(path_obj, label)


def _build_quality_runtime_preflight() -> Dict[str, Any]:
    """
    功能：检查当前 Python 环境中的质量依赖可导入性。

    Build the quality-runtime dependency preflight payload for the current
    Python environment.

    Returns:
        Dependency preflight mapping for LPIPS and CLIP runtime imports.
    """
    preflight_payload: Dict[str, Any] = {
        "diagnostic_only": True,
        "scope": "diagnostic_only_for_pw05_env",
    }
    for module_name, ready_key, reason_key in [
        ("lpips", "lpips_dependency_ready", "lpips_dependency_reason"),
        ("open_clip", "clip_dependency_ready", "clip_dependency_reason"),
    ]:
        try:
            importlib.import_module(module_name)
            preflight_payload[ready_key] = True
            preflight_payload[reason_key] = None
        except Exception as exc:
            preflight_payload[ready_key] = False
            preflight_payload[reason_key] = f"{module_name}_import_failed:{type(exc).__name__}: {exc}"
    return preflight_payload


def _append_quality_failure_reasons(
    blocking_reasons: List[str],
    *,
    artifact_reason: Any,
    dependency_ready: Any,
    dependency_reason: Any,
    fallback_reason: str,
) -> None:
    """
    功能：按 artifact 优先、dependency 次之的顺序追加 quality 失败原因。

    Append one quality failure reason with artifact-level specificity preserved
    ahead of runtime dependency preflight diagnostics.

    Args:
        blocking_reasons: Mutable blocking-reason list.
        artifact_reason: Quality artifact reason.
        dependency_ready: Dependency readiness flag.
        dependency_reason: Dependency preflight reason.
        fallback_reason: Fallback reason when neither concrete source exists.

    Returns:
        None.
    """
    if not isinstance(blocking_reasons, list):
        raise TypeError("blocking_reasons must be list")
    if not isinstance(fallback_reason, str) or not fallback_reason.strip():
        raise TypeError("fallback_reason must be non-empty str")

    artifact_reason_text = (
        str(artifact_reason).strip()
        if isinstance(artifact_reason, str) and str(artifact_reason).strip()
        else None
    )
    dependency_reason_text = (
        str(dependency_reason).strip()
        if dependency_ready is False
        and isinstance(dependency_reason, str)
        and str(dependency_reason).strip()
        else None
    )

    if artifact_reason_text is not None:
        blocking_reasons.append(artifact_reason_text)
        if dependency_reason_text is not None:
            blocking_reasons.append(dependency_reason_text)
        return

    if dependency_reason_text is not None:
        blocking_reasons.append(dependency_reason_text)
        return

    blocking_reasons.append(fallback_reason)


def _build_quality_component_readiness(
    *,
    component_name: str,
    source_path: Path | None,
    quality_payload: Mapping[str, Any] | None,
    status_key: str,
    reason_key: str,
    count_key: str,
    expected_count_key: str,
    lpips_status_key: str,
    lpips_reason_key: str,
    clip_status_key: str,
    clip_reason_key: str,
    prompt_text_coverage_status_key: str,
    prompt_text_coverage_reason_key: str,
    dependency_preflight: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    功能：为 clean 或 attack 质量链构造统一 readiness 结论。 

    Build one normalized readiness verdict for clean or attack quality metrics.

    Args:
        component_name: Stable component name.
        source_path: Source artifact path.
        quality_payload: Quality payload mapping.
        status_key: Status field name.
        reason_key: Reason field name.
        count_key: Available pair-count field name.
        expected_count_key: Expected pair-count field name.
        lpips_status_key: LPIPS status field name.
        lpips_reason_key: LPIPS reason field name.
        clip_status_key: CLIP status field name.
        clip_reason_key: CLIP reason field name.
        prompt_text_coverage_status_key: Prompt-coverage status field name.
        prompt_text_coverage_reason_key: Prompt-coverage reason field name.

    Returns:
        Normalized readiness mapping.
    """
    payload = _extract_mapping(quality_payload)
    raw_status = payload.get(status_key)
    status_value = str(raw_status) if isinstance(raw_status, str) and raw_status else "not_available"
    reason_value = payload.get(reason_key)
    count_value = _coerce_non_negative_int(payload.get(count_key))
    expected_count_value = _coerce_non_negative_int(payload.get(expected_count_key))
    lpips_status = payload.get(lpips_status_key)
    lpips_reason = payload.get(lpips_reason_key)
    clip_status = payload.get(clip_status_key)
    clip_reason = payload.get(clip_reason_key)
    prompt_text_coverage_status = payload.get(prompt_text_coverage_status_key)
    prompt_text_coverage_reason = payload.get(prompt_text_coverage_reason_key)
    dependency_payload = _extract_mapping(dependency_preflight)
    lpips_dependency_ready = dependency_payload.get("lpips_dependency_ready")
    lpips_dependency_reason = dependency_payload.get("lpips_dependency_reason")
    clip_dependency_ready = dependency_payload.get("clip_dependency_ready")
    clip_dependency_reason = dependency_payload.get("clip_dependency_reason")

    blocking_reasons: list[str] = []
    if status_value != "ok":
        blocking_reasons.append(
            str(reason_value) if isinstance(reason_value, str) and str(reason_value).strip() else f"quality status={status_value}"
        )
    if expected_count_value is not None and count_value is not None and expected_count_value > count_value:
        blocking_reasons.append(
            f"valid image pairs available for {count_value}/{expected_count_value} expected bindings"
        )
    if lpips_status != "ok":
        _append_quality_failure_reasons(
            blocking_reasons,
            artifact_reason=lpips_reason,
            dependency_ready=lpips_dependency_ready,
            dependency_reason=lpips_dependency_reason,
            fallback_reason="LPIPS unavailable",
        )
    if clip_status != "ok":
        _append_quality_failure_reasons(
            blocking_reasons,
            artifact_reason=clip_reason,
            dependency_ready=clip_dependency_ready,
            dependency_reason=clip_dependency_reason,
            fallback_reason="CLIP unavailable",
        )
    if isinstance(prompt_text_coverage_status, str) and prompt_text_coverage_status not in {"ok", "not_configured"}:
        blocking_reasons.append(
            str(prompt_text_coverage_reason)
            if isinstance(prompt_text_coverage_reason, str) and str(prompt_text_coverage_reason).strip()
            else "prompt text coverage incomplete"
        )

    if not blocking_reasons:
        readiness_status = "ready"
        readiness_reason = None
    elif count_value is not None and count_value > 0:
        readiness_status = "partial"
        readiness_reason = "; ".join(dict.fromkeys(blocking_reasons))
    else:
        readiness_status = "not_ready"
        readiness_reason = "; ".join(dict.fromkeys(blocking_reasons))

    return {
        "component_name": component_name,
        "status": readiness_status,
        "reason": readiness_reason,
        "required_for_formal_release": True,
        "blocking": readiness_status != "ready",
        "source_path": normalize_path_value(source_path) if isinstance(source_path, Path) else None,
        "available_pair_count": count_value,
        "expected_pair_count": expected_count_value,
        "lpips_status": lpips_status,
        "clip_status": clip_status,
        "prompt_text_coverage_status": prompt_text_coverage_status,
        "runtime_preflight_diagnostic_only": bool(dependency_payload.get("diagnostic_only", True)),
        "runtime_preflight_scope": dependency_payload.get("scope"),
        "lpips_dependency_ready": lpips_dependency_ready,
        "lpips_dependency_reason": lpips_dependency_reason,
        "clip_dependency_ready": clip_dependency_ready,
        "clip_dependency_reason": clip_dependency_reason,
    }


def _build_auxiliary_summary_component_readiness(
    *,
    component_name: str,
    source_path: Path | None,
    summary_payload: Mapping[str, Any] | None,
    required_for_formal_release: bool,
) -> Dict[str, Any]:
    """
    功能：把 payload 或 wrong-event 汇总规范化为统一 readiness 结论。 

    Normalize one auxiliary summary payload into a readiness verdict.

    Args:
        component_name: Stable component name.
        source_path: Source artifact path.
        summary_payload: Summary payload mapping.
        required_for_formal_release: Whether this component blocks formal release.

    Returns:
        Normalized readiness mapping.
    """
    payload = _extract_mapping(summary_payload)
    readiness_payload = _extract_mapping(payload.get("readiness"))
    status_value = readiness_payload.get("status")
    if not isinstance(status_value, str) or not status_value:
        payload_status = payload.get("status")
        if payload_status == "ok":
            status_value = "ready"
        elif payload_status == "partial":
            status_value = "partial"
        elif payload_status == "not_applicable":
            status_value = "not_applicable"
        else:
            status_value = "not_ready"
    readiness_reason = readiness_payload.get("reason")
    if not isinstance(readiness_reason, str) or not readiness_reason.strip():
        raw_reason = payload.get("reason")
        readiness_reason = raw_reason if isinstance(raw_reason, str) and raw_reason.strip() else None
    overall_payload = _extract_mapping(payload.get("overall"))
    return {
        "component_name": component_name,
        "status": status_value,
        "reason": readiness_reason,
        "required_for_formal_release": required_for_formal_release,
        "blocking": required_for_formal_release and status_value != "ready",
        "source_path": normalize_path_value(source_path) if isinstance(source_path, Path) else None,
        "event_count": _coerce_non_negative_int(overall_payload.get("event_count")),
        "attempted_event_count": _coerce_non_negative_int(overall_payload.get("attempted_event_count")),
    }


def _build_tail_component_readiness(
    *,
    source_path: Path | None,
    diagnostics_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    功能：规范化 tail estimation 的 readiness 结论。 

    Normalize the tail-estimation readiness payload.

    Args:
        source_path: Tail diagnostics artifact path.
        diagnostics_payload: Tail diagnostics payload mapping.

    Returns:
        Normalized readiness mapping.
    """
    payload = _extract_mapping(diagnostics_payload)
    readiness_payload = _extract_mapping(payload.get("readiness"))
    readiness_status = readiness_payload.get("status")
    if not isinstance(readiness_status, str) or not readiness_status:
        readiness_status = "not_ready"
    readiness_reason = readiness_payload.get("reason")
    if not isinstance(readiness_reason, str) or not readiness_reason.strip():
        readiness_reason = None
    return {
        "component_name": "tail_estimation",
        "status": readiness_status,
        "reason": readiness_reason,
        "required_for_formal_release": False,
        "blocking": False,
        "source_path": normalize_path_value(source_path) if isinstance(source_path, Path) else None,
    }


def _build_formal_run_scaling_plan(
    *,
    family_manifest: Mapping[str, Any],
    pw04_summary: Mapping[str, Any],
    attack_quality_payload: Mapping[str, Any] | None,
    blocking_components: Sequence[str],
) -> Dict[str, Any]:
    """
    功能：为正式论文 run 生成最小扩量建议模板。 

    Build the minimal scale-up template used before a formal paper run.

    Args:
        family_manifest: Family manifest payload.
        pw04_summary: PW04 summary payload.
        attack_quality_payload: Attack quality payload mapping.
        blocking_components: Blocking component names.

    Returns:
        Machine-readable scale-up plan template.
    """
    source_parameters = _extract_mapping(family_manifest.get("source_parameters"))
    attack_parameters = _extract_mapping(family_manifest.get("attack_parameters"))
    attack_quality_overall = _extract_mapping(_extract_mapping(attack_quality_payload).get("overall"))
    quality_runtime = _extract_mapping(attack_quality_overall.get("quality_runtime"))

    source_shard_count = _coerce_non_negative_int(source_parameters.get("source_shard_count")) or 0
    attack_shard_count = _coerce_non_negative_int(attack_parameters.get("attack_shard_count")) or source_shard_count

    return {
        "status": "template",
        "plan_name": "formal_paper_run_minimal_scale_up",
        "family_id": family_manifest.get("family_id", pw04_summary.get("family_id")),
        "current_observed_attack_event_count": _coerce_non_negative_int(
            pw04_summary.get("completed_attack_event_count")
        ),
        "current_source_shard_count": source_shard_count,
        "current_attack_shard_count": attack_shard_count,
        "recommended_stage_parameters": {
            "pw00": {
                "source_shard_count": max(source_shard_count, 4),
                "attack_shard_count": max(attack_shard_count, 4),
                "freeze_family_manifest_before_parallel_run": True,
            },
            "pw01": {
                "pw01_worker_count": 2,
                "rerun_only_missing_or_failed_shards": True,
            },
            "pw03": {
                "attack_local_worker_count": 2,
                "rerun_only_missing_or_failed_shards": True,
            },
            "pw04": {
                "enable_tail_estimation": True,
                "quality_runtime_env": {
                    "PW_QUALITY_TORCH_DEVICE": str(quality_runtime.get("torch_device") or "cuda:0"),
                    "PW_QUALITY_LPIPS_BATCH_SIZE": str(quality_runtime.get("lpips_batch_size") or 2),
                    "PW_QUALITY_CLIP_BATCH_SIZE": str(quality_runtime.get("clip_batch_size") or 2),
                },
            },
        },
        "gates_before_scale_up": [str(component_name) for component_name in blocking_components],
    }


def _build_formal_run_readiness_report(
    *,
    family_root: Path,
    family_id: str,
    source_paths: Mapping[str, Path],
    analysis_only_bindings: Mapping[str, Mapping[str, Any]],
    pw04_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：汇总辅助指标链并生成正式 run readiness 报告。 

    Build the unified formal-run readiness report for auxiliary metric closure.

    Args:
        family_root: Family root path.
        family_id: Family identifier.
        source_paths: Canonical release source paths.
        analysis_only_bindings: Analysis-only source artifact bindings.
        pw04_summary: PW04 summary payload.

    Returns:
        Formal-run readiness report payload.
    """
    family_manifest = _load_required_json_dict(cast(Path, source_paths["family_manifest"]), "family manifest")
    attack_quality_payload = _load_required_json_dict(
        cast(Path, source_paths["pw04_attack_quality_metrics"]),
        "PW04 attack quality metrics",
    )
    clean_quality_summary = _load_optional_json_dict(
        _resolve_analysis_only_binding_path(analysis_only_bindings, "pw02_quality_metrics_summary_json"),
        "PW02 quality metrics summary",
    )
    payload_clean_summary = _load_optional_json_dict(
        _resolve_analysis_only_binding_path(analysis_only_bindings, "pw02_payload_clean_summary"),
        "PW02 payload clean summary",
    )
    payload_attack_summary = _load_optional_json_dict(
        _resolve_analysis_only_binding_path(analysis_only_bindings, "pw04_payload_attack_summary"),
        "PW04 payload attack summary",
    )
    wrong_event_summary = _load_optional_json_dict(
        _resolve_analysis_only_binding_path(analysis_only_bindings, "pw04_wrong_event_attestation_challenge_summary"),
        "PW04 wrong-event attestation challenge summary",
    )
    geometry_summary = _load_optional_json_dict(
        _resolve_analysis_only_binding_path(analysis_only_bindings, "pw04_conditional_rescue_metrics"),
        "PW04 conditional rescue metrics",
    )
    tail_paths = _extract_mapping(pw04_summary.get("tail_estimation_paths"))
    tail_diagnostics_path = None
    tail_diagnostics_path_value = tail_paths.get("tail_fit_diagnostics_path")
    if isinstance(tail_diagnostics_path_value, str) and tail_diagnostics_path_value.strip():
        tail_diagnostics_path = _resolve_path_value_under_family_root(
            tail_diagnostics_path_value,
            family_root,
            "tail_estimation_paths.tail_fit_diagnostics_path",
        )
    tail_diagnostics = _load_optional_json_dict(tail_diagnostics_path, "PW04 tail fit diagnostics")

    clean_quality_rows = cast(List[Mapping[str, Any]], _extract_mapping(clean_quality_summary).get("rows", []))
    clean_content_chain_row = next(
        (row for row in clean_quality_rows if row.get("scope") == "content_chain"),
        {},
    )
    attack_quality_overall = _extract_mapping(attack_quality_payload.get("overall"))
    quality_runtime_preflight = _build_quality_runtime_preflight()

    components = {
        "quality_clean": _build_quality_component_readiness(
            component_name="quality_clean",
            source_path=_resolve_analysis_only_binding_path(analysis_only_bindings, "pw02_quality_metrics_summary_json"),
            quality_payload=clean_content_chain_row,
            status_key="status",
            reason_key="reason",
            count_key="pair_count",
            expected_count_key="expected_pair_count",
            lpips_status_key="lpips_status",
            lpips_reason_key="lpips_reason",
            clip_status_key="clip_status",
            clip_reason_key="clip_reason",
            prompt_text_coverage_status_key="prompt_text_coverage_status",
            prompt_text_coverage_reason_key="prompt_text_coverage_reason",
            dependency_preflight=quality_runtime_preflight,
        ),
        "quality_attack": _build_quality_component_readiness(
            component_name="quality_attack",
            source_path=cast(Path, source_paths["pw04_attack_quality_metrics"]),
            quality_payload=attack_quality_overall,
            status_key="status",
            reason_key="availability_reason",
            count_key="count",
            expected_count_key="expected_count",
            lpips_status_key="lpips_status",
            lpips_reason_key="lpips_reason",
            clip_status_key="clip_status",
            clip_reason_key="clip_reason",
            prompt_text_coverage_status_key="prompt_text_coverage_status",
            prompt_text_coverage_reason_key="prompt_text_coverage_reason",
            dependency_preflight=quality_runtime_preflight,
        ),
        "payload_clean": _build_auxiliary_summary_component_readiness(
            component_name="payload_clean",
            source_path=_resolve_analysis_only_binding_path(analysis_only_bindings, "pw02_payload_clean_summary"),
            summary_payload=payload_clean_summary,
            required_for_formal_release=True,
        ),
        "payload_attack": _build_auxiliary_summary_component_readiness(
            component_name="payload_attack",
            source_path=_resolve_analysis_only_binding_path(analysis_only_bindings, "pw04_payload_attack_summary"),
            summary_payload=payload_attack_summary,
            required_for_formal_release=True,
        ),
        "wrong_event_attack": _build_auxiliary_summary_component_readiness(
            component_name="wrong_event_attack",
            source_path=_resolve_analysis_only_binding_path(
                analysis_only_bindings,
                "pw04_wrong_event_attestation_challenge_summary",
            ),
            summary_payload=wrong_event_summary,
            required_for_formal_release=True,
        ),
        "geometry_conditional_rescue": _build_auxiliary_summary_component_readiness(
            component_name="geometry_conditional_rescue",
            source_path=_resolve_analysis_only_binding_path(analysis_only_bindings, "pw04_conditional_rescue_metrics"),
            summary_payload=geometry_summary,
            required_for_formal_release=False,
        ),
        "tail_estimation": _build_tail_component_readiness(
            source_path=tail_diagnostics_path,
            diagnostics_payload=tail_diagnostics,
        ),
    }

    blocking_components = [
        component_name
        for component_name, component_payload in components.items()
        if component_payload.get("blocking") is True
    ]
    blocking_reasons = [
        f"{component_name}: {component_payload.get('reason') or component_payload.get('status')}"
        for component_name, component_payload in components.items()
        if component_payload.get("blocking") is True
    ]
    advisory_components = [
        component_name
        for component_name, component_payload in components.items()
        if component_payload.get("blocking") is not True and component_payload.get("status") not in {"ready", "not_applicable"}
    ]

    if blocking_components:
        overall_status = "blocked"
        decision = BLOCK_FREEZE
    else:
        overall_status = "ready"
        decision = ALLOW_FREEZE

    return {
        "artifact_type": "paper_workflow_pw05_formal_run_readiness_report",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "family_id": family_id,
        "family_root": normalize_path_value(family_root),
        "overall_status": overall_status,
        "decision": decision,
        "blocking_components": blocking_components,
        "blocking_reasons": blocking_reasons,
        "advisory_components": advisory_components,
        "quality_runtime_preflight": quality_runtime_preflight,
        "components": components,
        "recommended_run_plan": _build_formal_run_scaling_plan(
            family_manifest=family_manifest,
            pw04_summary=pw04_summary,
            attack_quality_payload=attack_quality_payload,
            blocking_components=blocking_components,
        ),
    }


def _build_payload_sidecar_source_label(prefix: str, event_index: Any, fallback_ordinal: int) -> str:
    """
    功能：为 payload sidecar 源工件生成稳定标签。

    Build one stable source-artifact label for payload sidecars.

    Args:
        prefix: Stable label prefix.
        event_index: Event index when available.
        fallback_ordinal: Fallback ordinal.

    Returns:
        Stable label token.
    """
    if not isinstance(prefix, str) or not prefix:
        raise TypeError("prefix must be non-empty str")
    if not isinstance(fallback_ordinal, int) or isinstance(fallback_ordinal, bool) or fallback_ordinal < 0:
        raise TypeError("fallback_ordinal must be non-negative int")
    if isinstance(event_index, int) and not isinstance(event_index, bool) and event_index >= 0:
        return f"{prefix}_e{event_index:06d}"
    return f"{prefix}_{fallback_ordinal:06d}"


def _collect_payload_sidecar_source_paths(
    *,
    family_root: Path,
    source_paths: Mapping[str, Path],
) -> Dict[str, Path]:
    """
    功能：从 PW02 与 PW04 pool manifest 收集 payload sidecar 源工件。

    Collect payload-sidecar source artifacts from PW02 and PW04 pool manifests.

    Args:
        family_root: Family root path.
        source_paths: Validated canonical source paths.

    Returns:
        Additional label-to-path mappings for payload sidecars.
    """
    if not isinstance(family_root, Path):
        raise TypeError("family_root must be Path")
    if not isinstance(source_paths, Mapping):
        raise TypeError("source_paths must be Mapping")

    collected_paths: Dict[str, Path] = {}

    finalize_manifest_path = source_paths.get("pw02_finalize_manifest")
    if isinstance(finalize_manifest_path, Path):
        finalize_manifest = _load_required_json_dict(finalize_manifest_path, "PW02 finalize manifest")
        source_pools = finalize_manifest.get("source_pools")
        if isinstance(source_pools, Mapping):
            positive_pool_node = source_pools.get(ACTIVE_SAMPLE_ROLE)
            if isinstance(positive_pool_node, Mapping):
                positive_pool_manifest_path_value = positive_pool_node.get("manifest_path")
                if isinstance(positive_pool_manifest_path_value, str) and positive_pool_manifest_path_value.strip():
                    positive_pool_manifest_path = _resolve_path_value_under_family_root(
                        positive_pool_manifest_path_value,
                        family_root,
                        "PW02 positive_source_pool_manifest_path",
                    )
                    positive_pool_manifest = _load_required_json_dict(
                        positive_pool_manifest_path,
                        "PW02 positive source pool manifest",
                    )
                    positive_events = positive_pool_manifest.get("events")
                    if isinstance(positive_events, list):
                        for ordinal, event_node in enumerate(positive_events):
                            if not isinstance(event_node, Mapping):
                                continue
                            event_index = event_node.get("event_index")
                            reference_path_value = event_node.get("payload_reference_sidecar_path")
                            if isinstance(reference_path_value, str) and reference_path_value.strip():
                                collected_paths[
                                    _build_payload_sidecar_source_label(
                                        "pw02_positive_source_payload_reference_sidecar",
                                        event_index,
                                        ordinal,
                                    )
                                ] = _resolve_path_value_under_family_root(
                                    reference_path_value,
                                    family_root,
                                    "PW02 payload reference sidecar",
                                )
                            decode_path_value = event_node.get("payload_decode_sidecar_path")
                            if isinstance(decode_path_value, str) and decode_path_value.strip():
                                collected_paths[
                                    _build_payload_sidecar_source_label(
                                        "pw02_positive_source_payload_decode_sidecar",
                                        event_index,
                                        ordinal,
                                    )
                                ] = _resolve_path_value_under_family_root(
                                    decode_path_value,
                                    family_root,
                                    "PW02 payload decode sidecar",
                                )

    attack_positive_pool_manifest_path = source_paths.get("pw04_attack_positive_pool_manifest")
    if isinstance(attack_positive_pool_manifest_path, Path):
        attack_positive_pool_manifest = _load_required_json_dict(
            attack_positive_pool_manifest_path,
            "PW04 attack positive pool manifest",
        )
        attack_positive_events = attack_positive_pool_manifest.get("events")
        if isinstance(attack_positive_events, list):
            for ordinal, event_node in enumerate(attack_positive_events):
                if not isinstance(event_node, Mapping):
                    continue
                attack_event_index = event_node.get("attack_event_index")
                decode_path_value = event_node.get("payload_decode_sidecar_path")
                if isinstance(decode_path_value, str) and decode_path_value.strip():
                    collected_paths[
                        _build_payload_sidecar_source_label(
                            "pw04_attacked_positive_payload_decode_sidecar",
                            attack_event_index,
                            ordinal,
                        )
                    ] = _resolve_path_value_under_family_root(
                        decode_path_value,
                        family_root,
                        "PW04 payload decode sidecar",
                    )

    return collected_paths


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
        "formal_run_readiness_report": "artifacts/readiness/formal_run_readiness_report.json",
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

    resolved_stage_run_id = stage_run_id or make_stage_run_id(STAGE_NAME)

    formal_run_readiness_report_path = cast(Path, pw05_paths["formal_run_readiness_report_path"])
    signoff_report_path = cast(Path, pw05_paths["signoff_report_path"])
    release_manifest_path = cast(Path, pw05_paths["release_manifest_path"])
    workflow_summary_path = cast(Path, pw05_paths["workflow_summary_path"])
    run_closure_path = cast(Path, pw05_paths["run_closure_path"])
    stage_manifest_path = cast(Path, pw05_paths["stage_manifest_path"])
    package_manifest_path = cast(Path, pw05_paths["package_manifest_path"])
    export_root = cast(Path, pw05_paths["export_root"])
    summary_path = cast(Path, pw05_paths["summary_path"])
    package_staging_root = cast(Path, pw05_paths["package_staging_root"])

    formal_run_readiness_report = _build_formal_run_readiness_report(
        family_root=family_root,
        family_id=family_id,
        source_paths=source_paths,
        analysis_only_bindings=analysis_only_source_bindings,
        pw04_summary=pw04_summary,
    )
    write_json_atomic(formal_run_readiness_report_path, formal_run_readiness_report)

    blocking_reasons = cast(List[str], formal_run_readiness_report.get("blocking_reasons", []))
    decision = BLOCK_FREEZE if blocking_reasons else ALLOW_FREEZE
    status_payload = _resolve_signoff_statuses(decision)

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
        "blocking_reason_count": len(blocking_reasons),
        "blocking_reasons": blocking_reasons,
        "checked_source_artifact_count": len(source_artifact_index),
        "analysis_only_artifact_count": len(analysis_only_release_annotations),
        "checked_source_artifacts": source_artifact_index,
        "formal_run_readiness_report_path": normalize_path_value(formal_run_readiness_report_path),
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
        "formal_run_readiness_report_path": normalize_path_value(formal_run_readiness_report_path),
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
        "formal_run_readiness_report_path": normalize_path_value(formal_run_readiness_report_path),
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
            "ok": decision == ALLOW_FREEZE,
            "reason": "allow_freeze" if decision == ALLOW_FREEZE else "formal_run_readiness_blocked",
            "details": {
                "checked_source_artifact_count": len(source_artifact_index),
                "blocking_reason_count": len(blocking_reasons),
            },
        },
        "created_at": utc_now_iso(),
    }
    write_json_atomic(run_closure_path, run_closure)

    generated_artifact_paths = {
        "formal_run_readiness_report": formal_run_readiness_report_path,
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
        "formal_run_readiness_report_path": normalize_path_value(formal_run_readiness_report_path),
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
        "formal_run_readiness_report_path": normalize_path_value(formal_run_readiness_report_path),
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