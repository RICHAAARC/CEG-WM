"""
文件目的：04_Release_And_Signoff 独立阶段编排入口。
Module type: General module

职责边界：
1. 只消费 stage package，不重跑 embed、detect、calibrate、evaluate 或 experiment_matrix。
2. 只做 release 审计与 freeze signoff，不恢复旧 onefile、publish、repro 或 paper_full_cuda 一体化工作流。
3. 运行时不依赖 archive 路径；如需历史逻辑，仅在当前文件内最小吸收其纯函数思想。
4. 输出独立的 stage_manifest、workflow_summary、run_closure、signoff_report、release_manifest 与 stage 04 package。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    STAGE_01_NAME,
    STAGE_02_NAME,
    STAGE_03_NAME,
    build_stage_package_filename,
    collect_file_index,
    collect_git_summary,
    collect_python_summary,
    compute_file_sha256,
    compute_mapping_sha256,
    copy_stage_manifest_snapshot,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    prepare_source_package,
    read_json_dict,
    resolve_repo_path,
    resolve_stage_roots,
    stage_relative_copy,
    summarize_manifest_fields,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
STAGE_04_NAME = "04_Release_And_Signoff"

ALLOW_FREEZE = "ALLOW_FREEZE"
BLOCK_FREEZE = "BLOCK_FREEZE"

EVALUATION_REPORT_REQUIRED_FIELDS = [
    "cfg_digest",
    "plan_digest",
    "thresholds_digest",
    "threshold_metadata_digest",
    "impl_digest",
    "fusion_rule_version",
    "attack_protocol_version",
    "attack_protocol_digest",
    "policy_path",
]

GRID_SUMMARY_REQUIRED_FIELDS = [
    "cfg_digest",
    "thresholds_digest",
    "threshold_metadata_digest",
    "attack_protocol_version",
    "attack_protocol_digest",
    "attack_coverage_digest",
    "impl_digest",
    "fusion_rule_version",
    "policy_path",
]

AGGREGATE_REPORT_REQUIRED_FIELDS = [
    "aggregate_report_version",
    "experiment_matrix_digest",
    "experiment_count",
    "success_count",
    "failure_count",
    "attack_coverage_digest",
    "policy_path",
    "anchors",
    "metrics_matrix",
]


def _parse_bool_arg(value: Any, default: bool) -> bool:
    """
    功能：解析 CLI 布尔参数。 

    Parse a CLI boolean token with a stable default.

    Args:
        value: Raw CLI token.
        default: Default boolean value when the token is absent.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the token cannot be interpreted as boolean.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError("boolean argument must be bool, str, or None")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean token: {value}")


def _build_blocking_reason(
    *,
    source: str,
    reason_code: str,
    rule: str,
    impact: str,
    fix: str,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    功能：构造结构化阻断原因。 

    Build one structured blocking reason entry for the signoff report.

    Args:
        source: Reason source domain.
        reason_code: Stable reason code.
        rule: Audited rule statement.
        impact: User-facing impact summary.
        fix: Remediation guidance.
        evidence: Supporting evidence payload.

    Returns:
        Structured blocking reason mapping.
    """
    return {
        "source": source,
        "audit_id": reason_code,
        "reason_code": reason_code,
        "rule": rule,
        "impact": impact,
        "fix": fix,
        "evidence": dict(evidence),
    }


def _append_blocking_reason(
    blocking_reasons: List[Dict[str, Any]],
    *,
    source: str,
    reason_code: str,
    rule: str,
    impact: str,
    fix: str,
    evidence: Mapping[str, Any],
) -> None:
    """
    功能：向阻断原因列表追加结构化条目。 

    Append one structured blocking reason to the mutable list.

    Args:
        blocking_reasons: Mutable blocking reason list.
        source: Reason source domain.
        reason_code: Stable reason code.
        rule: Audited rule statement.
        impact: User-facing impact summary.
        fix: Remediation guidance.
        evidence: Supporting evidence payload.

    Returns:
        None.
    """
    blocking_reasons.append(
        _build_blocking_reason(
            source=source,
            reason_code=reason_code,
            rule=rule,
            impact=impact,
            fix=fix,
            evidence=evidence,
        )
    )


def _read_required_json(path_obj: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    功能：读取必需 JSON 文件并保留错误语义。 

    Read one required JSON file and preserve a precise failure reason.

    Args:
        path_obj: JSON file path.

    Returns:
        Tuple of parsed mapping or None, and error string or None.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not path_obj.exists() or not path_obj.is_file():
        return None, f"missing_file:{path_obj.as_posix()}"
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"json_parse_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return None, f"json_root_not_dict:{path_obj.as_posix()}"
    return cast(Dict[str, Any], payload), None


def _find_existing_candidate(extracted_root: Path, relative_candidates: Sequence[str]) -> Optional[Path]:
    """
    功能：在解压包内解析首个存在的候选相对路径。 

    Resolve the first existing relative candidate under the extracted package root.

    Args:
        extracted_root: Extracted package root.
        relative_candidates: Relative candidate path list.

    Returns:
        Existing resolved path, or None when absent.
    """
    if not isinstance(extracted_root, Path):
        raise TypeError("extracted_root must be Path")
    for relative_candidate in relative_candidates:
        candidate_path = extracted_root / relative_candidate
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    return None


def _collect_required_stage_files(
    stage_key: str,
    extracted_root: Path,
    blocking_reasons: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    功能：收集并校验每个 stage package 的必需文件。 

    Collect and validate required package-relative files for one stage.

    Args:
        stage_key: Stable stage key.
        extracted_root: Extracted package root.
        blocking_reasons: Mutable blocking reason list.

    Returns:
        Mapping from logical label to resolved file path string.
    """
    if stage_key == "stage_01":
        required_map = {
            "embed_record": ["records/embed_record.json"],
            "detect_record": ["records/detect_record.json"],
            "calibration_record": ["records/calibration_record.json"],
            "evaluate_record": ["records/evaluate_record.json"],
            "thresholds_artifact": ["artifacts/thresholds/thresholds_artifact.json"],
            "threshold_metadata_artifact": ["artifacts/thresholds/threshold_metadata_artifact.json"],
            "evaluation_report": ["artifacts/evaluation_report.json"],
            "run_closure": ["artifacts/run_closure.json"],
            "workflow_summary": ["artifacts/workflow_summary.json"],
            "stage_manifest": ["artifacts/stage_manifest.json"],
            "package_manifest": ["artifacts/package_manifest.json"],
            "runtime_config_snapshot": ["runtime_metadata/runtime_config_snapshot.yaml"],
        }
    elif stage_key == "stage_02":
        required_map = {
            "calibration_record": ["records/calibration_record.json"],
            "evaluate_record": ["records/evaluate_record.json"],
            "thresholds_artifact": ["artifacts/thresholds/thresholds_artifact.json"],
            "threshold_metadata_artifact": ["artifacts/thresholds/threshold_metadata_artifact.json"],
            "evaluation_report": ["artifacts/evaluation_report.json"],
            "run_closure": ["artifacts/run_closure.json"],
            "workflow_summary": ["artifacts/workflow_summary.json"],
            "stage_manifest": ["artifacts/stage_manifest.json"],
            "lineage_stage_manifest": ["lineage/source_stage_manifest.json"],
            "lineage_package_manifest": ["lineage/source_package_manifest.json"],
        }
    elif stage_key == "stage_03":
        required_map = {
            "grid_summary": ["artifacts/grid_summary.json"],
            "grid_manifest": ["artifacts/grid_manifest.json"],
            "aggregate_report": ["artifacts/aggregate_report.json"],
            "workflow_summary": ["artifacts/workflow_summary.json"],
            "stage_manifest": ["artifacts/stage_manifest.json"],
            "lineage_stage_manifest": ["lineage/source_stage_manifest.json"],
            "lineage_package_manifest": ["lineage/source_package_manifest.json"],
            "thresholds_artifact": [
                "artifacts/thresholds/thresholds_artifact.json",
                "global_calibrate/artifacts/thresholds/thresholds_artifact.json",
            ],
            "threshold_metadata_artifact": [
                "artifacts/thresholds/threshold_metadata_artifact.json",
                "global_calibrate/artifacts/thresholds/threshold_metadata_artifact.json",
            ],
        }
    else:
        raise ValueError(f"unsupported stage_key: {stage_key}")

    resolved_files: Dict[str, str] = {}
    missing_labels: List[Dict[str, Any]] = []
    for label, relative_candidates in required_map.items():
        resolved_path = _find_existing_candidate(extracted_root, relative_candidates)
        if resolved_path is None:
            missing_labels.append({"label": label, "candidates": list(relative_candidates)})
            continue
        resolved_files[label] = resolved_path.as_posix()

    if missing_labels:
        _append_blocking_reason(
            blocking_reasons,
            source=stage_key,
            reason_code=f"{stage_key}.required_artifacts_missing",
            rule="required stage package artifacts must exist inside the extracted package",
            impact=f"{stage_key} package is incomplete for release and signoff auditing",
            fix="re-export the stage package and ensure all required records, artifacts, and lineage files are packaged",
            evidence={
                "extracted_root": extracted_root.as_posix(),
                "missing": missing_labels,
            },
        )
    return resolved_files


def _require_nonempty_string_fields(
    payload: Mapping[str, Any],
    field_names: Sequence[str],
    *,
    source: str,
    reason_code: str,
    blocking_reasons: List[Dict[str, Any]],
    rule: str,
    impact: str,
    fix: str,
) -> None:
    """
    功能：校验一组字段为非空字符串。 

    Validate that selected fields are non-empty strings.

    Args:
        payload: Mapping payload to validate.
        field_names: Required field names.
        source: Reason source domain.
        reason_code: Stable reason code.
        blocking_reasons: Mutable blocking reason list.
        rule: Audited rule statement.
        impact: User-facing impact summary.
        fix: Remediation guidance.

    Returns:
        None.
    """
    missing_fields = [
        field_name
        for field_name in field_names
        if not isinstance(payload.get(field_name), str) or not str(payload.get(field_name)).strip() or payload.get(field_name) == "<absent>"
    ]
    if missing_fields:
        _append_blocking_reason(
            blocking_reasons,
            source=source,
            reason_code=reason_code,
            rule=rule,
            impact=impact,
            fix=fix,
            evidence={"missing_fields": missing_fields},
        )


def _validate_stage_json_payloads(
    stage_key: str,
    stage_label: str,
    required_files: Mapping[str, str],
    blocking_reasons: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    功能：读取并校验 stage 关键 JSON 工件。 

    Read and validate the key JSON artifacts for one stage package.

    Args:
        stage_key: Stable stage key.
        stage_label: Human-readable stage label.
        required_files: Required file mapping.
        blocking_reasons: Mutable blocking reason list.

    Returns:
        Parsed JSON payload mapping.
    """
    parsed_payloads: Dict[str, Dict[str, Any]] = {}
    json_labels: List[str] = []
    if stage_key == "stage_01":
        json_labels = ["run_closure", "evaluation_report"]
    elif stage_key == "stage_02":
        json_labels = ["evaluation_report"]
    elif stage_key == "stage_03":
        json_labels = ["grid_summary", "aggregate_report"]

    for label in json_labels:
        raw_path = required_files.get(label)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        payload, error_text = _read_required_json(Path(raw_path))
        if error_text is not None or payload is None:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.{label}_invalid_json",
                rule="required JSON artifacts must be parseable and have dict root nodes",
                impact=f"{stage_label} {label} cannot be audited reliably",
                fix=f"regenerate {label} and ensure it is valid JSON with a dict root",
                evidence={"path": raw_path, "error": error_text},
            )
            continue
        parsed_payloads[label] = payload

    run_closure = parsed_payloads.get("run_closure")
    if stage_key == "stage_01" and isinstance(run_closure, dict):
        status_value = run_closure.get("status")
        if not isinstance(status_value, (dict, str)):
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.run_closure_missing_status",
                rule="stage 01 run_closure must expose status or equivalent status object",
                impact="stage 01 closure status cannot be verified for signoff",
                fix="regenerate stage 01 and ensure run_closure.json includes a status field or status object",
                evidence={"path": required_files.get("run_closure"), "keys": sorted(run_closure.keys())},
            )

    evaluation_report = parsed_payloads.get("evaluation_report")
    if isinstance(evaluation_report, dict):
        _require_nonempty_string_fields(
            evaluation_report,
            EVALUATION_REPORT_REQUIRED_FIELDS,
            source=stage_key,
            reason_code=f"{stage_key}.evaluation_report_missing_anchors",
            blocking_reasons=blocking_reasons,
            rule="evaluation reports must retain the frozen anchor set used for signoff",
            impact=f"{stage_label} evaluation anchors are incomplete",
            fix="regenerate evaluation_report.json with all anchor fields populated",
        )

    grid_summary = parsed_payloads.get("grid_summary")
    if isinstance(grid_summary, dict):
        _require_nonempty_string_fields(
            grid_summary,
            GRID_SUMMARY_REQUIRED_FIELDS,
            source=stage_key,
            reason_code=f"{stage_key}.grid_summary_missing_anchors",
            blocking_reasons=blocking_reasons,
            rule="grid_summary must expose experiment-matrix anchor fields for release signoff",
            impact="stage 03 grid summary is incomplete",
            fix="regenerate stage 03 outputs so grid_summary.json includes all required anchor fields",
        )

    aggregate_report = parsed_payloads.get("aggregate_report")
    if isinstance(aggregate_report, dict):
        missing_fields: List[str] = []
        for field_name in AGGREGATE_REPORT_REQUIRED_FIELDS:
            field_value = aggregate_report.get(field_name)
            if field_name in {"anchors", "metrics_matrix"}:
                if not isinstance(field_value, list) or len(field_value) == 0:
                    missing_fields.append(field_name)
            elif field_name in {"experiment_count", "success_count", "failure_count"}:
                if not isinstance(field_value, int):
                    missing_fields.append(field_name)
            elif not field_value and field_value != 0:
                missing_fields.append(field_name)
        if missing_fields:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.aggregate_report_missing_anchors",
                rule="aggregate_report must expose experiment-matrix summary anchors for release signoff",
                impact="stage 03 aggregate report is incomplete",
                fix="regenerate aggregate_report.json with the expected anchor and summary fields",
                evidence={"missing_fields": missing_fields, "path": required_files.get("aggregate_report")},
            )
    return parsed_payloads


def _prepare_stage_package_input(
    *,
    stage_key: str,
    stage_name: str,
    package_path: Optional[Path],
    required: bool,
    runtime_state_root: Path,
    blocking_reasons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：准备单个输入 stage package。 

    Prepare one stage package for downstream validation and release auditing.

    Args:
        stage_key: Stable stage key.
        stage_name: Expected stage name.
        package_path: Optional stage package path.
        required: Whether the stage is mandatory.
        runtime_state_root: Stage 04 runtime-state root.
        blocking_reasons: Mutable blocking reason list.

    Returns:
        Stage input summary mapping.
    """
    info: Dict[str, Any] = {
        "stage_key": stage_key,
        "expected_stage_name": stage_name,
        "required": required,
        "provided": package_path is not None,
        "input_path": normalize_path_value(package_path) if package_path is not None else "<absent>",
        "status": "not_provided",
    }
    if package_path is None:
        if required:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.required_package_missing",
                rule="required stage packages must be provided explicitly for release signoff",
                impact=f"{stage_name} is required but was not provided",
                fix=f"provide --{stage_key.replace('_', '-')}-package with a valid {stage_name} ZIP package",
                evidence={"input_path": "<absent>", "required": required},
            )
        return info

    try:
        source_info = prepare_source_package(package_path, runtime_state_root / stage_key)
    except Exception as exc:
        info["status"] = "prepare_failed"
        _append_blocking_reason(
            blocking_reasons,
            source=stage_key,
            reason_code=f"{stage_key}.package_prepare_failed",
            rule="stage packages must pass copy, SHA256, manifest binding, extraction, and manifest loading checks",
            impact=f"{stage_name} package could not be prepared for release signoff",
            fix="export a fresh stage package and ensure the external/internal package manifests are consistent with the ZIP",
            evidence={
                "input_path": normalize_path_value(package_path),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return info

    extracted_root = Path(str(source_info["extracted_root"]))
    stage_manifest = cast(Dict[str, Any], source_info["stage_manifest"])
    package_manifest = cast(Dict[str, Any], source_info["package_manifest"])

    info.update(
        {
            "status": "prepared",
            "source_info": source_info,
            "extracted_root": extracted_root,
            "stage_manifest": stage_manifest,
            "package_manifest": package_manifest,
            "package_sha256": str(source_info["source_package_sha256"]),
            "package_manifest_digest": str(source_info["package_manifest_digest"]),
        }
    )
    if stage_manifest.get("stage_name") != stage_name:
        _append_blocking_reason(
            blocking_reasons,
            source=stage_key,
            reason_code=f"{stage_key}.unexpected_stage_name",
            rule=f"{stage_key} package stage_name must equal {stage_name}",
            impact=f"{stage_name} lineage cannot be trusted when stage_name drifts",
            fix=f"re-export the correct {stage_name} package",
            evidence={
                "expected_stage_name": stage_name,
                "actual_stage_name": stage_manifest.get("stage_name"),
                "input_path": normalize_path_value(package_path),
            },
        )
    return info


def _persist_lineage_snapshots(
    runtime_state_root: Path,
    stage_inputs: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    功能：将成功加载的上游 manifest 快照写入 stage 04 lineage 目录。 

    Persist successful upstream manifest snapshots into the stage 04 lineage directory.

    Args:
        runtime_state_root: Stage 04 runtime-state root.
        stage_inputs: Stage input mapping.

    Returns:
        Mapping of persisted lineage paths per stage key.
    """
    lineage_root = ensure_directory(runtime_state_root / "lineage")
    persisted: Dict[str, Dict[str, str]] = {}
    for stage_key, info in stage_inputs.items():
        if info.get("status") != "prepared":
            continue
        stage_manifest = cast(Dict[str, Any], info["stage_manifest"])
        package_manifest = cast(Dict[str, Any], info["package_manifest"])
        stage_suffix = stage_key.replace("stage_", "")
        stage_manifest_path = lineage_root / f"source_stage_{stage_suffix}_manifest.json"
        package_manifest_path = lineage_root / f"source_stage_{stage_suffix}_package_manifest.json"
        copy_stage_manifest_snapshot(stage_manifest, stage_manifest_path)
        write_json_atomic(package_manifest_path, package_manifest)
        persisted[stage_key] = {
            "stage_manifest_path": stage_manifest_path.as_posix(),
            "package_manifest_path": package_manifest_path.as_posix(),
        }
    return persisted


def _validate_stage_lineage(
    stage_inputs: Mapping[str, Dict[str, Any]],
    required_files: Mapping[str, Dict[str, str]],
    blocking_reasons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：校验 stage 02 与 stage 03 相对 stage 01 的 lineage 一致性。 

    Validate stage 02 and stage 03 lineage bindings against stage 01.

    Args:
        stage_inputs: Stage input mapping.
        required_files: Required file mapping by stage key.
        blocking_reasons: Mutable blocking reason list.

    Returns:
        Lineage summary mapping.
    """
    lineage_summary: Dict[str, Any] = {}
    stage_01_info = stage_inputs.get("stage_01")
    if not isinstance(stage_01_info, dict) or stage_01_info.get("status") != "prepared":
        return lineage_summary

    stage_01_manifest = cast(Dict[str, Any], stage_01_info["stage_manifest"])
    stage_01_package_manifest = cast(Dict[str, Any], stage_01_info["package_manifest"])
    stage_01_manifest_digest = compute_mapping_sha256(stage_01_manifest)
    stage_01_package_digest = compute_mapping_sha256(stage_01_package_manifest)

    for stage_key in ("stage_02", "stage_03"):
        current_info = stage_inputs.get(stage_key)
        if not isinstance(current_info, dict) or current_info.get("status") != "prepared":
            continue
        current_manifest = cast(Dict[str, Any], current_info["stage_manifest"])
        current_files = required_files.get(stage_key, {})
        source_stage_manifest_copy_path = current_files.get("lineage_stage_manifest")
        source_package_manifest_copy_path = current_files.get("lineage_package_manifest")
        source_stage_manifest_copy, source_stage_manifest_error = (
            _read_required_json(Path(source_stage_manifest_copy_path))
            if isinstance(source_stage_manifest_copy_path, str)
            else (None, "missing_lineage_stage_manifest")
        )
        source_package_manifest_copy, source_package_manifest_error = (
            _read_required_json(Path(source_package_manifest_copy_path))
            if isinstance(source_package_manifest_copy_path, str)
            else (None, "missing_lineage_package_manifest")
        )

        required_lineage_fields = [
            "source_stage_run_id",
            "source_package_sha256",
            "source_package_manifest_digest",
            "source_stage_manifest_path",
            "source_package_manifest_path",
            "source_runtime_config_snapshot_path",
            "source_thresholds_artifact_path",
            "source_stage_manifest_copy_path",
        ]
        _require_nonempty_string_fields(
            current_manifest,
            required_lineage_fields,
            source=stage_key,
            reason_code=f"{stage_key}.lineage_fields_missing",
            blocking_reasons=blocking_reasons,
            rule="downstream stages must retain complete lineage fields for their stage 01 source package",
            impact=f"{stage_key} lineage cannot be verified",
            fix="re-export the downstream stage package with all lineage fields populated",
        )

        if source_stage_manifest_error is not None or source_stage_manifest_copy is None:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.lineage_stage_manifest_invalid",
                rule="lineage/source_stage_manifest.json must be packaged and parseable",
                impact=f"{stage_key} cannot prove its stage 01 origin",
                fix="re-export the downstream stage package with a valid lineage/source_stage_manifest.json",
                evidence={"path": source_stage_manifest_copy_path, "error": source_stage_manifest_error},
            )
        if source_package_manifest_error is not None or source_package_manifest_copy is None:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.lineage_package_manifest_invalid",
                rule="lineage/source_package_manifest.json must be packaged and parseable",
                impact=f"{stage_key} cannot prove its source package binding",
                fix="re-export the downstream stage package with a valid lineage/source_package_manifest.json",
                evidence={"path": source_package_manifest_copy_path, "error": source_package_manifest_error},
            )

        comparisons = {
            "source_stage_run_id": current_manifest.get("source_stage_run_id") == stage_01_manifest.get("stage_run_id"),
            "source_package_sha256": current_manifest.get("source_package_sha256") == stage_01_info.get("package_sha256"),
            "source_package_manifest_digest": current_manifest.get("source_package_manifest_digest") == stage_01_package_digest,
            "source_runtime_config_snapshot_path": current_manifest.get("source_runtime_config_snapshot_path") == stage_01_manifest.get("runtime_config_snapshot_path"),
            "source_thresholds_artifact_path": current_manifest.get("source_thresholds_artifact_path") == stage_01_manifest.get("thresholds_path"),
            "lineage_stage_manifest_digest": isinstance(source_stage_manifest_copy, dict) and compute_mapping_sha256(source_stage_manifest_copy) == stage_01_manifest_digest,
            "lineage_package_manifest_digest": isinstance(source_package_manifest_copy, dict) and compute_mapping_sha256(source_package_manifest_copy) == stage_01_package_digest,
        }
        failed_checks = [name for name, status_value in comparisons.items() if not status_value]
        if failed_checks:
            _append_blocking_reason(
                blocking_reasons,
                source=stage_key,
                reason_code=f"{stage_key}.lineage_mismatch",
                rule="downstream lineage fields and packaged lineage snapshots must align with the required stage 01 package",
                impact=f"{stage_key} cannot be trusted as a child of the provided stage 01 package",
                fix="re-run the downstream stage from the exact stage 01 package supplied to stage 04",
                evidence={
                    "failed_checks": failed_checks,
                    "expected_stage_run_id": stage_01_manifest.get("stage_run_id"),
                    "actual_stage_run_id": current_manifest.get("source_stage_run_id"),
                    "expected_package_sha256": stage_01_info.get("package_sha256"),
                    "actual_package_sha256": current_manifest.get("source_package_sha256"),
                    "expected_package_manifest_digest": stage_01_package_digest,
                    "actual_package_manifest_digest": current_manifest.get("source_package_manifest_digest"),
                },
            )
        lineage_summary[stage_key] = comparisons
    return lineage_summary


def _compute_signoff_decision(blocking_reasons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    功能：根据结构化阻断原因计算 freeze signoff 判定。 

    Compute the final freeze signoff decision from structured blocking reasons.

    Args:
        blocking_reasons: Structured blocking reason sequence.

    Returns:
        Decision summary mapping.
    """
    reason_list = [dict(item) for item in blocking_reasons]
    decision = ALLOW_FREEZE if len(reason_list) == 0 else BLOCK_FREEZE
    return {
        "decision": decision,
        "blocking_reasons": reason_list,
        "blocking_reason_count": len(reason_list),
    }


def _build_release_manifest(
    *,
    stage_inputs: Mapping[str, Dict[str, Any]],
    decision_summary: Mapping[str, Any],
    config_path: Path,
    stage_run_id: str,
    required_stage_02: bool,
    required_stage_03: bool,
) -> Dict[str, Any]:
    """
    功能：构建 stage 04 release_manifest。 

    Build the stage 04 release manifest summarizing consumed stage packages and the final decision.

    Args:
        stage_inputs: Stage input mapping.
        decision_summary: Final decision mapping.
        config_path: Config path used by stage 04.
        stage_run_id: Stage 04 run identifier.
        required_stage_02: Whether stage 02 is required.
        required_stage_03: Whether stage 03 is required.

    Returns:
        Release manifest mapping.
    """
    consumed_packages: Dict[str, Any] = {}
    for stage_key, stage_name in {
        "stage_01": STAGE_01_NAME,
        "stage_02": STAGE_02_NAME,
        "stage_03": STAGE_03_NAME,
    }.items():
        info = stage_inputs.get(stage_key, {})
        consumed_packages[stage_key] = {
            "stage_name": stage_name,
            "required": required_stage_02 if stage_key == "stage_02" else required_stage_03 if stage_key == "stage_03" else True,
            "provided": bool(info.get("provided", False)),
            "status": info.get("status", "not_provided"),
            "input_path": info.get("input_path", "<absent>"),
            "package_sha256": info.get("package_sha256", "<absent>"),
            "package_manifest_digest": info.get("package_manifest_digest", "<absent>"),
            "stage_run_id": info.get("stage_manifest", {}).get("stage_run_id") if isinstance(info.get("stage_manifest"), dict) else "<absent>",
        }

    return {
        "release_manifest_version": "v1",
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "config_source_path": normalize_path_value(config_path),
        "required_stage_02": required_stage_02,
        "required_stage_03": required_stage_03,
        "consumed_stage_packages": consumed_packages,
        "decision": decision_summary.get("decision"),
        "blocking_reason_count": decision_summary.get("blocking_reason_count"),
        "created_at": utc_now_iso(),
    }


def _package_stage_outputs(
    *,
    runtime_state_root: Path,
    stage_manifest_path: Path,
    workflow_summary_path: Path,
    run_closure_path: Path,
    signoff_report_path: Path,
    release_manifest_path: Path,
    runtime_config_snapshot_path: Path,
    lineage_paths: Mapping[str, Dict[str, str]],
) -> Path:
    """
    功能：构造 stage 04 package staging 目录。 

    Build the stage 04 package staging directory.

    Args:
        runtime_state_root: Stage 04 runtime-state root.
        stage_manifest_path: Stage manifest path.
        workflow_summary_path: Workflow summary path.
        run_closure_path: Run closure path.
        signoff_report_path: Signoff report path.
        release_manifest_path: Release manifest path.
        runtime_config_snapshot_path: Runtime config snapshot path.
        lineage_paths: Persisted lineage path mapping.

    Returns:
        Package staging root path.
    """
    package_root = ensure_directory(runtime_state_root / "package_staging")
    for relative_path, source_path in {
        "artifacts/stage_manifest.json": stage_manifest_path,
        "artifacts/workflow_summary.json": workflow_summary_path,
        "artifacts/run_closure.json": run_closure_path,
        "artifacts/signoff/signoff_report.json": signoff_report_path,
        "artifacts/release/release_manifest.json": release_manifest_path,
        "runtime_metadata/runtime_config_snapshot.yaml": runtime_config_snapshot_path,
    }.items():
        stage_relative_copy(source_path, package_root, relative_path)

    for stage_key, path_mapping in lineage_paths.items():
        stage_suffix = stage_key.replace("stage_", "")
        stage_manifest_copy = path_mapping.get("stage_manifest_path")
        package_manifest_copy = path_mapping.get("package_manifest_path")
        if isinstance(stage_manifest_copy, str) and stage_manifest_copy:
            stage_relative_copy(
                Path(stage_manifest_copy),
                package_root,
                f"lineage/source_stage_{stage_suffix}_manifest.json",
            )
        if isinstance(package_manifest_copy, str) and package_manifest_copy:
            stage_relative_copy(
                Path(package_manifest_copy),
                package_root,
                f"lineage/source_stage_{stage_suffix}_package_manifest.json",
            )
    return package_root


def run_stage_04(
    *,
    drive_project_root: Path,
    stage_01_package_path: Path,
    stage_02_package_path: Optional[Path],
    stage_03_package_path: Optional[Path],
    config_path: Path,
    notebook_name: str,
    stage_run_id: str,
    require_stage_02: bool,
    require_stage_03: bool,
) -> Dict[str, Any]:
    """
    功能：执行 stage 04 release 与 freeze signoff。 

    Run stage 04 release auditing and freeze signoff using already exported stage packages.

    Args:
        drive_project_root: Google Drive project root.
        stage_01_package_path: Required stage 01 package path.
        stage_02_package_path: Optional stage 02 package path.
        stage_03_package_path: Optional stage 03 package path.
        config_path: Config path for runtime snapshotting.
        notebook_name: Notebook display name.
        stage_run_id: Stage run identifier.
        require_stage_02: Whether stage 02 is mandatory.
        require_stage_03: Whether stage 03 is mandatory.

    Returns:
        Stage summary mapping.
    """
    stage_roots = resolve_stage_roots(drive_project_root, STAGE_04_NAME, stage_run_id)
    run_root = ensure_directory(stage_roots["run_root"])
    log_root = ensure_directory(stage_roots["log_root"])
    runtime_state_root = ensure_directory(stage_roots["runtime_state_root"])
    export_root = ensure_directory(stage_roots["export_root"])
    for protected_path in (run_root, log_root, runtime_state_root, export_root):
        validate_path_within_base(drive_project_root, protected_path, "stage path")

    cfg_obj = load_yaml_mapping(config_path)
    runtime_config_snapshot_path = runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    write_yaml_mapping(runtime_config_snapshot_path, cfg_obj)

    blocking_reasons: List[Dict[str, Any]] = []
    stage_inputs = {
        "stage_01": _prepare_stage_package_input(
            stage_key="stage_01",
            stage_name=STAGE_01_NAME,
            package_path=stage_01_package_path,
            required=True,
            runtime_state_root=runtime_state_root,
            blocking_reasons=blocking_reasons,
        ),
        "stage_02": _prepare_stage_package_input(
            stage_key="stage_02",
            stage_name=STAGE_02_NAME,
            package_path=stage_02_package_path,
            required=require_stage_02,
            runtime_state_root=runtime_state_root,
            blocking_reasons=blocking_reasons,
        ),
        "stage_03": _prepare_stage_package_input(
            stage_key="stage_03",
            stage_name=STAGE_03_NAME,
            package_path=stage_03_package_path,
            required=require_stage_03,
            runtime_state_root=runtime_state_root,
            blocking_reasons=blocking_reasons,
        ),
    }

    lineage_paths = _persist_lineage_snapshots(runtime_state_root, stage_inputs)
    required_files: Dict[str, Dict[str, str]] = {}
    parsed_payloads: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for stage_key, stage_label in {
        "stage_01": STAGE_01_NAME,
        "stage_02": STAGE_02_NAME,
        "stage_03": STAGE_03_NAME,
    }.items():
        stage_info = stage_inputs[stage_key]
        if stage_info.get("status") != "prepared":
            continue
        extracted_root = cast(Path, stage_info["extracted_root"])
        required_files[stage_key] = _collect_required_stage_files(stage_key, extracted_root, blocking_reasons)
        parsed_payloads[stage_key] = _validate_stage_json_payloads(
            stage_key,
            stage_label,
            required_files[stage_key],
            blocking_reasons,
        )

    lineage_summary = _validate_stage_lineage(stage_inputs, required_files, blocking_reasons)
    decision_summary = _compute_signoff_decision(blocking_reasons)

    signoff_report_path = run_root / "artifacts" / "signoff" / "signoff_report.json"
    release_manifest_path = run_root / "artifacts" / "release" / "release_manifest.json"
    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    run_closure_path = run_root / "artifacts" / "run_closure.json"
    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"

    release_manifest = _build_release_manifest(
        stage_inputs=stage_inputs,
        decision_summary=decision_summary,
        config_path=config_path,
        stage_run_id=stage_run_id,
        required_stage_02=require_stage_02,
        required_stage_03=require_stage_03,
    )
    write_json_atomic(release_manifest_path, release_manifest)

    signoff_report = {
        "signoff_report_version": "v1",
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "decision": decision_summary["decision"],
        "blocking_reason_count": decision_summary["blocking_reason_count"],
        "blocking_reasons": decision_summary["blocking_reasons"],
        "required_stage_02": require_stage_02,
        "required_stage_03": require_stage_03,
        "stage_inputs": {
            stage_key: {
                "status": info.get("status"),
                "required": info.get("required"),
                "provided": info.get("provided"),
                "input_path": info.get("input_path"),
                "stage_name": info.get("stage_manifest", {}).get("stage_name") if isinstance(info.get("stage_manifest"), dict) else "<absent>",
                "stage_run_id": info.get("stage_manifest", {}).get("stage_run_id") if isinstance(info.get("stage_manifest"), dict) else "<absent>",
                "package_sha256": info.get("package_sha256", "<absent>"),
                "package_manifest_digest": info.get("package_manifest_digest", "<absent>"),
            }
            for stage_key, info in stage_inputs.items()
        },
        "lineage_summary": lineage_summary,
        "json_anchor_summary": {
            stage_key: {
                label: summarize_manifest_fields(payload, [
                    "cfg_digest",
                    "plan_digest",
                    "thresholds_digest",
                    "threshold_metadata_digest",
                    "attack_protocol_version",
                    "attack_protocol_digest",
                    "attack_coverage_digest",
                    "impl_digest",
                    "fusion_rule_version",
                    "policy_path",
                    "aggregate_report_version",
                    "experiment_matrix_digest",
                    "experiment_count",
                ])
                for label, payload in stage_payloads.items()
            }
            for stage_key, stage_payloads in parsed_payloads.items()
        },
        "created_at": utc_now_iso(),
    }
    write_json_atomic(signoff_report_path, signoff_report)

    workflow_summary = {
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "notebook_name": notebook_name,
        "decision": decision_summary["decision"],
        "blocking_reason_count": decision_summary["blocking_reason_count"],
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "required_stage_02": require_stage_02,
        "required_stage_03": require_stage_03,
        "consumed_stage_packages": release_manifest["consumed_stage_packages"],
        "created_at": utc_now_iso(),
    }
    write_json_atomic(workflow_summary_path, workflow_summary)

    run_closure = {
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "decision": decision_summary["decision"],
        "blocking_reason_count": decision_summary["blocking_reason_count"],
        "status": {
            "ok": decision_summary["decision"] == ALLOW_FREEZE,
            "reason": "allow_freeze" if decision_summary["decision"] == ALLOW_FREEZE else "block_freeze",
            "details": {
                "blocking_reason_count": decision_summary["blocking_reason_count"],
                "required_stage_02": require_stage_02,
                "required_stage_03": require_stage_03,
            },
        },
        "created_at": utc_now_iso(),
    }
    write_json_atomic(run_closure_path, run_closure)

    stage_01_manifest = cast(Dict[str, Any], stage_inputs["stage_01"].get("stage_manifest", {}))
    stage_manifest = {
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_name": stage_01_manifest.get("stage_name"),
        "source_stage_run_id": stage_01_manifest.get("stage_run_id"),
        "config_source_path": normalize_path_value(config_path),
        "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "logs_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "exports_root": normalize_path_value(export_root),
        "records": collect_file_index(run_root, {}),
        "evaluation_report_path": normalize_path_value(signoff_report_path),
        "run_closure_path": normalize_path_value(run_closure_path),
        "workflow_summary_path": normalize_path_value(workflow_summary_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "required_stage_02": require_stage_02,
        "required_stage_03": require_stage_03,
        "consumed_stage_packages": release_manifest["consumed_stage_packages"],
        "lineage_snapshot_paths": lineage_paths,
        "git": collect_git_summary(REPO_ROOT),
        "python": collect_python_summary(),
        "created_at": utc_now_iso(),
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    package_root = _package_stage_outputs(
        runtime_state_root=runtime_state_root,
        stage_manifest_path=stage_manifest_path,
        workflow_summary_path=workflow_summary_path,
        run_closure_path=run_closure_path,
        signoff_report_path=signoff_report_path,
        release_manifest_path=release_manifest_path,
        runtime_config_snapshot_path=runtime_config_snapshot_path,
        lineage_paths=lineage_paths,
    )
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name=STAGE_04_NAME,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=cast(Optional[str], stage_01_manifest.get("stage_run_id")),
        source_stage_package_path=cast(Optional[str], stage_inputs["stage_01"].get("source_info", {}).get("source_package_path")),
        package_manifest_path=package_manifest_path,
    )

    summary = {
        "stage_name": STAGE_04_NAME,
        "stage_run_id": stage_run_id,
        "decision": decision_summary["decision"],
        "blocking_reason_count": decision_summary["blocking_reason_count"],
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "signoff_report_path": normalize_path_value(signoff_report_path),
        "release_manifest_path": normalize_path_value(release_manifest_path),
        "package_filename": build_stage_package_filename(STAGE_04_NAME, stage_run_id, cast(Optional[str], stage_01_manifest.get("stage_run_id"))),
        "package_path": package_manifest["package_path"],
        "package_sha256": package_manifest["package_sha256"],
        "status": "ok",
    }
    write_json_atomic(runtime_state_root / "stage_summary.json", summary)
    return summary


def main() -> int:
    """
    功能：stage 04 CLI 入口。 

    CLI entry for the stage 04 release and signoff workflow.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the stage-04 detached release and freeze signoff workflow.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--stage-01-package", required=True, help="Required stage-01 package ZIP path.")
    parser.add_argument("--stage-02-package", default=None, help="Optional stage-02 package ZIP path.")
    parser.add_argument("--stage-03-package", default=None, help="Optional stage-03 package ZIP path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Source config path.")
    parser.add_argument("--notebook-name", default=STAGE_04_NAME, help="Notebook display name.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    parser.add_argument("--require-stage-02", default="true", help="Whether stage 02 package is required (true/false).")
    parser.add_argument("--require-stage-03", default="true", help="Whether stage 03 package is required (true/false).")
    args = parser.parse_args()

    summary = run_stage_04(
        drive_project_root=resolve_repo_path(args.drive_project_root),
        stage_01_package_path=resolve_repo_path(args.stage_01_package),
        stage_02_package_path=resolve_repo_path(args.stage_02_package) if isinstance(args.stage_02_package, str) and args.stage_02_package.strip() else None,
        stage_03_package_path=resolve_repo_path(args.stage_03_package) if isinstance(args.stage_03_package, str) and args.stage_03_package.strip() else None,
        config_path=resolve_repo_path(args.config),
        notebook_name=str(args.notebook_name),
        stage_run_id=args.stage_run_id or make_stage_run_id(STAGE_04_NAME),
        require_stage_02=_parse_bool_arg(args.require_stage_02, True),
        require_stage_03=_parse_bool_arg(args.require_stage_03, True),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())