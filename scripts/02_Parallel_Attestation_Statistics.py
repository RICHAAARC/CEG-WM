"""
文件目的：02_Parallel_Attestation_Statistics 独立阶段编排入口。
Module type: General module

职责边界：
1. 仅消费 01 stage package，执行 event_attestation_score 的 calibrate 与 evaluate。
2. 不重跑 embed 或 detect，不覆盖 01 正式 records。
3. 输出独立 stage_manifest 与 package，并记录 source_stage_run_id lineage。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, cast

from scripts.notebook_runtime_common import (
    REPO_ROOT,
    STAGE_02_NAME,
    collect_attestation_env_summary,
    collect_cuda_summary,
    collect_file_index,
    collect_git_summary,
    collect_model_summary,
    collect_python_summary,
    collect_weight_summary,
    compute_file_sha256,
    copy_stage_manifest_snapshot,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    prepare_source_package,
    resolve_repo_path,
    resolve_stage_roots,
    run_command_with_logs,
    stage_relative_copy,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)
from scripts.workflow_acceptance_common import detect_formal_gpu_preflight


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"


def _resolve_source_lineage_paths(extracted_root: Path) -> Dict[str, Path]:
    return {
        "source_stage_manifest_path": extracted_root / "artifacts" / "stage_manifest.json",
        "source_package_manifest_path": extracted_root / "artifacts" / "package_manifest.json",
        "source_runtime_config_snapshot_path": extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml",
        "source_thresholds_artifact_path": extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
    }


def _resolve_prompt_snapshot_path(extracted_root: Path) -> str:
    prompt_root = extracted_root / "runtime_metadata" / "prompt_snapshot"
    if prompt_root.exists() and prompt_root.is_dir():
        for prompt_path in sorted(prompt_root.rglob("*")):
            if prompt_path.is_file():
                return normalize_path_value(prompt_path)
    return "<absent>"


def _load_parallel_attestation_statistics_input_contract(
    extracted_root: Path,
    source_stage_manifest: Dict[str, Any],
) -> tuple[Path, Dict[str, Any]]:
    if not isinstance(extracted_root, Path):
        raise TypeError("extracted_root must be Path")
    if not isinstance(source_stage_manifest, dict):
        raise TypeError("source_stage_manifest must be dict")

    contract_relative_path = source_stage_manifest.get(
        "parallel_attestation_statistics_input_contract_package_relative_path"
    )
    if not isinstance(contract_relative_path, str) or not contract_relative_path:
        contract_relative_path = PARALLEL_ATTESTATION_STATS_CONTRACT_RELATIVE_PATH

    contract_path = extracted_root / contract_relative_path
    if not contract_path.exists() or not contract_path.is_file():
        raise FileNotFoundError(
            "stage 02 requires parallel_attestation_statistics_input_contract in source package: "
            f"{normalize_path_value(contract_path)}"
        )

    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract_payload, dict):
        raise ValueError(f"parallel_attestation_statistics_input_contract must be JSON object: {contract_path}")
    return contract_path, contract_payload


def _stage_contract_bound_detect_records(
    extracted_root: Path,
    runtime_state_root: Path,
    contract_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(extracted_root, Path):
        raise TypeError("extracted_root must be Path")
    if not isinstance(runtime_state_root, Path):
        raise TypeError("runtime_state_root must be Path")
    if not isinstance(contract_payload, dict):
        raise TypeError("contract_payload must be dict")

    if contract_payload.get("status") != "ready":
        raise ValueError(
            "parallel_attestation_statistics_input_contract is not ready: "
            f"status={contract_payload.get('status')!r}, reason={contract_payload.get('reason')!r}"
        )

    score_name = contract_payload.get("score_name")
    if score_name != "event_attestation_score":
        raise ValueError(
            "parallel_attestation_statistics_input_contract must declare canonical event_attestation_score: "
            f"{score_name!r}"
        )

    label_summary = contract_payload.get("label_summary")
    if not isinstance(label_summary, dict) or label_summary.get("label_balanced") is not True:
        raise ValueError("parallel_attestation_statistics_input_contract must be label balanced")

    contract_records = contract_payload.get("records")
    if not isinstance(contract_records, list) or not contract_records:
        raise ValueError("parallel_attestation_statistics_input_contract must contain records")

    staged_records_root = ensure_directory(runtime_state_root / "contract_bound_detect_records")
    staged_record_paths: list[str] = []
    positive_count = 0
    negative_count = 0

    for record_index, record_entry in enumerate(contract_records, start=1):
        if not isinstance(record_entry, dict):
            raise ValueError("parallel_attestation_statistics_input_contract.records entries must be objects")

        relative_path = record_entry.get("package_relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("parallel_attestation_statistics_input_contract record missing package_relative_path")

        record_label = record_entry.get("label")
        if not isinstance(record_label, bool):
            raise ValueError("parallel_attestation_statistics_input_contract record label must be bool")

        source_record_path = extracted_root / relative_path
        if not source_record_path.exists() or not source_record_path.is_file():
            raise FileNotFoundError(
                "parallel_attestation_statistics_input_contract record missing from source package: "
                f"{normalize_path_value(source_record_path)}"
            )

        expected_sha256 = record_entry.get("sha256")
        if isinstance(expected_sha256, str) and expected_sha256:
            actual_sha256 = compute_file_sha256(source_record_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "parallel_attestation_statistics_input_contract record digest mismatch: "
                    f"{normalize_path_value(source_record_path)}"
                )

        label_tag = "positive" if record_label else "negative"
        staged_record_path = stage_relative_copy(
            source_record_path,
            staged_records_root,
            f"{record_index:03d}_{label_tag}_{source_record_path.name}",
        )
        staged_record_paths.append(normalize_path_value(staged_record_path))
        if record_label:
            positive_count += 1
        else:
            negative_count += 1

    if positive_count == 0 or negative_count == 0:
        raise ValueError("parallel_attestation_statistics_input_contract staged records are not label balanced")

    return {
        "staged_records_root": normalize_path_value(staged_records_root),
        "detect_records_glob": normalize_path_value(staged_records_root / "*.json"),
        "record_count": len(staged_record_paths),
        "positive": positive_count,
        "negative": negative_count,
        "records": staged_record_paths,
        "score_name": score_name,
    }


def _build_runtime_config(cfg_obj: Dict[str, Any], contract_detect_records_glob: str, run_root: Path) -> Dict[str, Any]:
    parallel_cfg = cfg_obj.get("parallel_attestation_statistics")
    parallel_section: Dict[str, Any] = dict(cast(Dict[str, Any], parallel_cfg)) if isinstance(parallel_cfg, dict) else {}
    config_copy = json.loads(json.dumps(cfg_obj))
    if not isinstance(contract_detect_records_glob, str) or not contract_detect_records_glob:
        raise TypeError("contract_detect_records_glob must be non-empty str")

    calibration_cfg = dict(config_copy.get("calibration")) if isinstance(config_copy.get("calibration"), dict) else {}
    calibration_cfg["score_name"] = parallel_section.get("calibration_score_name", "event_attestation_score")
    calibration_cfg["detect_records_glob"] = contract_detect_records_glob
    config_copy["calibration"] = calibration_cfg

    evaluate_cfg = dict(config_copy.get("evaluate")) if isinstance(config_copy.get("evaluate"), dict) else {}
    evaluate_cfg["score_name"] = parallel_section.get("evaluate_score_name", "event_attestation_score")
    evaluate_cfg["detect_records_glob"] = contract_detect_records_glob
    evaluate_cfg["thresholds_path"] = str((run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").resolve())
    config_copy["evaluate"] = evaluate_cfg
    return config_copy


def _run_stage(stage_name: str, config_path: Path, run_root: Path, log_root: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
        "--override",
        "run_root_reuse_allowed=true",
        "--override",
        f'run_root_reuse_reason="{STAGE_02_NAME}_{stage_name}"',
    ]
    result = run_command_with_logs(
        command=command,
        cwd=REPO_ROOT,
        stdout_log_path=log_root / f"{stage_name}_stdout.log",
        stderr_log_path=log_root / f"{stage_name}_stderr.log",
    )
    result["status"] = "ok" if result["return_code"] == 0 else "failed"
    return result


def _package_outputs(run_root: Path, runtime_state_root: Path, stage_manifest_path: Path, runtime_config_snapshot_path: Path, source_stage_manifest_copy_path: Path) -> Path:
    package_root = ensure_directory(runtime_state_root / "package_staging")
    for relative_path, source_path in {
        "records/calibration_record.json": run_root / "records" / "calibration_record.json",
        "records/evaluate_record.json": run_root / "records" / "evaluate_record.json",
        "artifacts/thresholds/thresholds_artifact.json": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "artifacts/thresholds/threshold_metadata_artifact.json": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "artifacts/evaluation_report.json": run_root / "artifacts" / "evaluation_report.json",
        "artifacts/run_closure.json": run_root / "artifacts" / "run_closure.json",
        "artifacts/workflow_summary.json": run_root / "artifacts" / "workflow_summary.json",
        "artifacts/stage_manifest.json": stage_manifest_path,
        "runtime_metadata/runtime_config_snapshot.yaml": runtime_config_snapshot_path,
        "lineage/source_stage_manifest.json": source_stage_manifest_copy_path,
        "lineage/source_package_manifest.json": runtime_state_root / "lineage" / "source_package_manifest.json",
    }.items():
        stage_relative_copy(source_path, package_root, relative_path)
    return package_root


def run_stage_02(
    *,
    drive_project_root: Path,
    config_path: Path,
    source_package_path: Path,
    notebook_name: str,
    stage_run_id: str,
) -> Dict[str, Any]:
    stage_roots = resolve_stage_roots(drive_project_root, STAGE_02_NAME, stage_run_id)
    run_root = ensure_directory(stage_roots["run_root"])
    log_root = ensure_directory(stage_roots["log_root"])
    runtime_state_root = ensure_directory(stage_roots["runtime_state_root"])
    export_root = ensure_directory(stage_roots["export_root"])
    for protected_path in (run_root, log_root, runtime_state_root, export_root):
        validate_path_within_base(drive_project_root, protected_path, "stage path")

    source_info = prepare_source_package(source_package_path, runtime_state_root)
    source_manifest = cast(Dict[str, Any], source_info["stage_manifest"])
    if source_manifest.get("stage_name") != "01_Paper_Full_Cuda":
        raise ValueError("stage 02 requires a source package produced by 01_Paper_Full_Cuda")
    extracted_root = Path(str(source_info["extracted_root"]))
    source_lineage_paths = _resolve_source_lineage_paths(extracted_root)
    missing_source_lineage = [
        label for label, path_obj in source_lineage_paths.items()
        if label != "source_package_manifest_path" and not path_obj.exists()
    ]
    if missing_source_lineage:
        raise FileNotFoundError(f"stage 02 source lineage files missing: {missing_source_lineage}")

    source_contract_path, source_contract_payload = _load_parallel_attestation_statistics_input_contract(
        extracted_root,
        source_manifest,
    )
    contract_bound_records = _stage_contract_bound_detect_records(
        extracted_root,
        runtime_state_root,
        source_contract_payload,
    )

    cfg_obj = load_yaml_mapping(config_path)
    runtime_config_snapshot_path = runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    runtime_cfg = _build_runtime_config(cfg_obj, contract_bound_records["detect_records_glob"], run_root)
    write_yaml_mapping(runtime_config_snapshot_path, runtime_cfg)

    preflight = detect_formal_gpu_preflight(runtime_config_snapshot_path)
    if not bool(preflight.get("ok", False)):
        raise RuntimeError(f"formal GPU preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

    stage_results: Dict[str, Any] = {}
    for stage_name in ("calibrate", "evaluate"):
        result = _run_stage(stage_name, runtime_config_snapshot_path, run_root, log_root)
        stage_results[stage_name] = result
        if result["return_code"] != 0:
            raise RuntimeError(f"stage 02 {stage_name} failed: return_code={result['return_code']}")

    outputs = {
        "calibration_record": run_root / "records" / "calibration_record.json",
        "evaluate_record": run_root / "records" / "evaluate_record.json",
        "thresholds_artifact": run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
        "threshold_metadata_artifact": run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
        "evaluation_report": run_root / "artifacts" / "evaluation_report.json",
        "run_closure": run_root / "artifacts" / "run_closure.json",
    }
    missing_outputs = [label for label, path_obj in outputs.items() if not path_obj.exists()]
    if missing_outputs:
        raise FileNotFoundError(f"stage 02 required outputs missing: {missing_outputs}")

    source_stage_manifest_copy_path = runtime_state_root / "lineage" / "source_stage_manifest.json"
    copy_stage_manifest_snapshot(source_manifest, source_stage_manifest_copy_path)
    source_package_manifest_copy_path = runtime_state_root / "lineage" / "source_package_manifest.json"
    write_json_atomic(source_package_manifest_copy_path, cast(Dict[str, Any], source_info["package_manifest"]))

    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    write_json_atomic(workflow_summary_path, {
        "stage_name": STAGE_02_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_path": str(source_info["source_package_path"]),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "stage_results": stage_results,
        "created_at": utc_now_iso(),
    })

    stage_manifest_path = run_root / "artifacts" / "stage_manifest.json"
    stage_manifest: Dict[str, Any] = {
        "stage_name": STAGE_02_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_name": source_manifest.get("stage_name"),
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "config_source_path": normalize_path_value(config_path),
        "runtime_config_snapshot_path": normalize_path_value(runtime_config_snapshot_path),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "logs_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "exports_root": normalize_path_value(export_root),
        "records": collect_file_index(run_root, {
            "calibration_record": outputs["calibration_record"],
            "evaluate_record": outputs["evaluate_record"],
        }),
        "thresholds_path": normalize_path_value(outputs["thresholds_artifact"]),
        "threshold_metadata_artifact_path": normalize_path_value(outputs["threshold_metadata_artifact"]),
        "evaluation_report_path": normalize_path_value(outputs["evaluation_report"]),
        "run_closure_path": normalize_path_value(outputs["run_closure"]),
        "workflow_summary_path": normalize_path_value(workflow_summary_path),
        "source_package_path": str(source_info["source_package_path"]),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "source_package_manifest_path": normalize_path_value(source_package_manifest_copy_path),
        "source_package_manifest_digest": str(source_info["package_manifest_digest"]),
        "source_parallel_attestation_statistics_input_contract_path": normalize_path_value(source_contract_path),
        "source_parallel_attestation_statistics_input_contract_status": source_contract_payload.get("status"),
        "source_parallel_attestation_statistics_input_contract_reason": source_contract_payload.get("reason"),
        "contract_bound_detect_records": contract_bound_records,
        "source_stage_manifest_path": normalize_path_value(source_stage_manifest_copy_path),
        "source_runtime_config_snapshot_path": normalize_path_value(source_lineage_paths["source_runtime_config_snapshot_path"]),
        "source_prompt_snapshot_path": _resolve_prompt_snapshot_path(extracted_root),
        "source_thresholds_artifact_path": normalize_path_value(source_lineage_paths["source_thresholds_artifact_path"]),
        "source_stage_manifest_copy_path": normalize_path_value(source_stage_manifest_copy_path),
        "notebook_name": notebook_name,
        "git": collect_git_summary(REPO_ROOT),
        "python": collect_python_summary(),
        "cuda": collect_cuda_summary(),
        "attestation_env": collect_attestation_env_summary(runtime_cfg),
        "model_summary": collect_model_summary(runtime_cfg),
        "weight_summary": collect_weight_summary(REPO_ROOT, runtime_cfg),
        "created_at": utc_now_iso(),
        "stage_results": stage_results,
    }
    write_json_atomic(stage_manifest_path, stage_manifest)

    package_root = _package_outputs(run_root, runtime_state_root, stage_manifest_path, runtime_config_snapshot_path, source_stage_manifest_copy_path)
    package_manifest_path = run_root / "artifacts" / "package_manifest.json"
    package_manifest = finalize_stage_package(
        stage_name=STAGE_02_NAME,
        stage_run_id=stage_run_id,
        package_root=package_root,
        export_root=export_root,
        source_stage_run_id=str(source_manifest.get("stage_run_id")),
        source_stage_package_path=str(source_info["source_package_path"]),
        package_manifest_path=package_manifest_path,
    )

    summary: Dict[str, Any] = {
        "stage_name": STAGE_02_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_manifest_path": normalize_path_value(source_package_manifest_copy_path),
        "source_package_manifest_digest": str(source_info["package_manifest_digest"]),
        "run_root": normalize_path_value(run_root),
        "log_root": normalize_path_value(log_root),
        "runtime_state_root": normalize_path_value(runtime_state_root),
        "export_root": normalize_path_value(export_root),
        "stage_manifest_path": normalize_path_value(stage_manifest_path),
        "package_manifest_path": normalize_path_value(package_manifest_path),
        "package_path": package_manifest["package_path"],
        "package_sha256": package_manifest["package_sha256"],
        "status": "ok",
    }
    write_json_atomic(runtime_state_root / "stage_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stage-02 detached parallel attestation statistics workflow.")
    parser.add_argument("--drive-project-root", required=True, help="Google Drive project root.")
    parser.add_argument("--source-package", required=True, help="Source stage-01 package ZIP.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.as_posix()), help="Source config path.")
    parser.add_argument("--notebook-name", default=STAGE_02_NAME, help="Notebook display name.")
    parser.add_argument("--stage-run-id", default=None, help="Optional fixed stage run identifier.")
    args = parser.parse_args()

    summary = run_stage_02(
        drive_project_root=resolve_repo_path(args.drive_project_root),
        config_path=resolve_repo_path(args.config),
        source_package_path=resolve_repo_path(args.source_package),
        notebook_name=str(args.notebook_name),
        stage_run_id=args.stage_run_id or make_stage_run_id(STAGE_02_NAME),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
