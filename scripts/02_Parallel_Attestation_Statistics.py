"""
文件目的：02_Parallel_Attestation_Statistics 独立阶段编排入口。
Module type: General module

职责边界：
1. 仅消费 01 stage package，执行 event_attestation_score 的 calibrate 与 evaluate。
2. 不重跑 embed 或 detect，不覆盖 01 正式 records。
3. 输出独立 stage_manifest 与 package，并记录 source_stage_run_id lineage。
4. 显式构建 parallel attestation statistics 输入，不再把 label-balance 补齐责任隐式下推给 calibrate/evaluate helper。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, cast

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
    ensure_attestation_env_bootstrap,
    ensure_directory,
    finalize_stage_package,
    load_yaml_mapping,
    make_stage_run_id,
    normalize_path_value,
    persist_source_package_lineage,
    prepare_source_package,
    read_required_json_dict,
    resolve_repo_path,
    resolve_source_lineage_paths,
    resolve_source_prompt_snapshot_path,
    resolve_stage_roots,
    run_command_with_logs,
    stage_relative_copy,
    utc_now_iso,
    validate_path_within_base,
    write_json_atomic,
    write_yaml_mapping,
)
from scripts.workflow_acceptance_common import detect_stage_02_preflight


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
PARALLEL_ATTESTATION_STATS_SOURCE_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_input_contract.json"
PARALLEL_ATTESTATION_STATS_BUILD_CONTRACT_RELATIVE_PATH = "artifacts/parallel_attestation_statistics_build_contract.json"
PARALLEL_ATTESTATION_STATS_STAGED_RECORDS_RELATIVE_ROOT = "artifacts/parallel_attestation_statistics_detect_records"
EVENT_ATTESTATION_SCORE_NAME = "event_attestation_score"
DIRECT_SOURCE_ONLY_MODE = "direct_source_only"
SOURCE_PLUS_DERIVED_PAIRS_MODE = "source_plus_derived_pairs"
def _resolve_detect_record_label(record_payload: Dict[str, Any]) -> bool | None:
    if not isinstance(record_payload, dict):
        raise TypeError("record_payload must be dict")
    for field_name in ["label", "ground_truth", "is_watermarked"]:
        label_value = record_payload.get(field_name)
        if isinstance(label_value, bool):
            return label_value
    return None


def _resolve_event_attestation_score(record_payload: Dict[str, Any]) -> float | None:
    if not isinstance(record_payload, dict):
        raise TypeError("record_payload must be dict")
    attestation_node = record_payload.get("attestation")
    attestation_payload = attestation_node if isinstance(attestation_node, dict) else {}
    final_decision_node = attestation_payload.get("final_event_attested_decision")
    final_decision = final_decision_node if isinstance(final_decision_node, dict) else {}
    score_name = final_decision.get("event_attestation_score_name")
    if isinstance(score_name, str) and score_name and score_name != EVENT_ATTESTATION_SCORE_NAME:
        return None
    score_value = final_decision.get("event_attestation_score")
    if isinstance(score_value, bool):
        return None
    if isinstance(score_value, (int, float)):
        return float(score_value)
    return None


def _ensure_event_attestation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")
    attestation_node = payload.get("attestation")
    attestation_payload = dict(cast(Dict[str, Any], attestation_node)) if isinstance(attestation_node, dict) else {}
    final_decision_node = attestation_payload.get("final_event_attested_decision")
    final_decision = dict(cast(Dict[str, Any], final_decision_node)) if isinstance(final_decision_node, dict) else {}
    attestation_payload["final_event_attested_decision"] = final_decision
    payload["attestation"] = attestation_payload
    return final_decision


def _apply_negative_attestation_semantics(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    final_decision = _ensure_event_attestation_payload(payload)
    final_decision["status"] = "absent"
    final_decision["is_event_attested"] = False
    final_decision["event_attestation_score"] = 0.0
    final_decision["event_attestation_score_name"] = EVENT_ATTESTATION_SCORE_NAME
    final_decision["authenticity_status"] = "statement_only"
    final_decision["image_evidence_status"] = "absent"

    attestation_node = payload.get("attestation")
    if not isinstance(attestation_node, dict):
        return

    attestation_payload = cast(Dict[str, Any], attestation_node)
    attestation_payload["status"] = "absent"
    attestation_payload["verdict"] = "absent"
    if "content_attestation_score" in attestation_payload:
        attestation_payload["content_attestation_score"] = 0.0
    if "fusion_score" in attestation_payload:
        attestation_payload["fusion_score"] = 0.0

    image_evidence_result = attestation_payload.get("image_evidence_result")
    if isinstance(image_evidence_result, dict):
        image_evidence_payload = cast(Dict[str, Any], image_evidence_result)
        image_evidence_payload["status"] = "absent"
        if "content_attestation_score" in image_evidence_payload:
            image_evidence_payload["content_attestation_score"] = 0.0
        if "fusion_score" in image_evidence_payload:
            image_evidence_payload["fusion_score"] = 0.0


def _apply_positive_attestation_semantics(payload: Dict[str, Any], score_value: float) -> None:
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")
    if not isinstance(score_value, float):
        raise TypeError("score_value must be float")

    final_decision = _ensure_event_attestation_payload(payload)
    final_decision["status"] = "attested"
    final_decision["is_event_attested"] = True
    final_decision["event_attestation_score"] = score_value
    final_decision["event_attestation_score_name"] = EVENT_ATTESTATION_SCORE_NAME


def _resolve_prompt_lines(prompt_file: str) -> List[str]:
    if not isinstance(prompt_file, str) or not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")

    prompt_path = Path(prompt_file.strip()).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (REPO_ROOT / prompt_path).resolve()
    else:
        prompt_path = prompt_path.resolve()

    if not prompt_path.exists() or not prompt_path.is_file():
        raise ValueError(f"prompt file not found: {prompt_path}")

    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"prompt file has no non-empty lines: {prompt_path}")
    return lines


def _load_parallel_attestation_statistics_source_contract(
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
        contract_relative_path = PARALLEL_ATTESTATION_STATS_SOURCE_CONTRACT_RELATIVE_PATH

    contract_path = extracted_root / contract_relative_path
    contract_payload = read_required_json_dict(contract_path, "parallel_attestation_statistics_input_contract")
    artifact_type = contract_payload.get("artifact_type")
    if artifact_type != "parallel_attestation_statistics_input_contract":
        raise ValueError(
            "stage 02 requires canonical parallel_attestation_statistics_input_contract artifact_type: "
            f"{artifact_type!r}"
        )
    contract_role = contract_payload.get("contract_role")
    if contract_role != "source_contract":
        raise ValueError(
            "stage 02 requires stage 01 source contract semantics: "
            f"contract_role={contract_role!r}"
        )
    return contract_path, contract_payload


def _resolve_parallel_attestation_statistics_build_config(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    parallel_cfg = cfg_obj.get("parallel_attestation_statistics")
    section = dict(cast(Dict[str, Any], parallel_cfg)) if isinstance(parallel_cfg, dict) else {}
    requested_build_mode = section.get("build_mode", SOURCE_PLUS_DERIVED_PAIRS_MODE)
    if requested_build_mode not in {DIRECT_SOURCE_ONLY_MODE, SOURCE_PLUS_DERIVED_PAIRS_MODE}:
        raise ValueError(
            "parallel_attestation_statistics.build_mode must be one of "
            f"{{'{DIRECT_SOURCE_ONLY_MODE}', '{SOURCE_PLUS_DERIVED_PAIRS_MODE}'}}"
        )

    target_pair_count = section.get("target_pair_count", 1)
    if not isinstance(target_pair_count, int) or target_pair_count <= 0:
        raise ValueError("parallel_attestation_statistics.target_pair_count must be positive int")

    return {
        "requested_build_mode": requested_build_mode,
        "allow_derived_input_build": section.get("allow_derived_input_build") is True,
        "derived_input_prompts_file": section.get("derived_input_prompts_file"),
        "target_pair_count": target_pair_count,
        "build_usage": section.get("build_usage", "parallel_attestation_statistics"),
        "record_derivation_kind": section.get("record_derivation_kind", "prompt_bound_label_balance"),
        "calibration_score_name": section.get("calibration_score_name", EVENT_ATTESTATION_SCORE_NAME),
        "evaluate_score_name": section.get("evaluate_score_name", EVENT_ATTESTATION_SCORE_NAME),
    }


def _validate_source_contract_for_build(contract_payload: Dict[str, Any]) -> None:
    if not isinstance(contract_payload, dict):
        raise TypeError("contract_payload must be dict")
    if contract_payload.get("score_name") != EVENT_ATTESTATION_SCORE_NAME:
        raise ValueError(
            "parallel_attestation_statistics_input_contract must declare canonical event_attestation_score: "
            f"{contract_payload.get('score_name')!r}"
        )
    if contract_payload.get("source_records_available") is not True:
        raise ValueError("parallel_attestation_statistics_input_contract must declare source records available")
    contract_records = contract_payload.get("records")
    if not isinstance(contract_records, list) or not contract_records:
        raise ValueError("parallel_attestation_statistics_input_contract must contain source records")


def _stage_direct_source_records(
    extracted_root: Path,
    staged_records_root: Path,
    contract_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(extracted_root, Path):
        raise TypeError("extracted_root must be Path")
    if not isinstance(staged_records_root, Path):
        raise TypeError("staged_records_root must be Path")
    if not isinstance(contract_payload, dict):
        raise TypeError("contract_payload must be dict")

    staged_entries: List[Dict[str, Any]] = []
    contract_records = cast(List[Dict[str, Any]], contract_payload["records"])
    for record_index, record_entry in enumerate(contract_records, start=1):
        if not isinstance(record_entry, dict):
            raise ValueError("parallel_attestation_statistics_input_contract.records entries must be objects")

        relative_path = record_entry.get("package_relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("parallel_attestation_statistics_input_contract record missing package_relative_path")

        source_record_path = extracted_root / relative_path
        source_record_payload = read_required_json_dict(source_record_path, "source detect record")
        source_label = _resolve_detect_record_label(source_record_payload)
        if not isinstance(source_label, bool):
            raise ValueError(
                "parallel_attestation_statistics_input_contract source record must resolve to bool label: "
                f"{normalize_path_value(source_record_path)}"
            )

        entry_label = record_entry.get("label")
        if isinstance(entry_label, bool) and entry_label != source_label:
            raise ValueError(
                "parallel_attestation_statistics_input_contract record label mismatch: "
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

        label_tag = "positive" if source_label else "negative"
        staged_file_name = f"{record_index:03d}_direct_{label_tag}_{source_record_path.name}"
        staged_record_path = stage_relative_copy(
            source_record_path,
            staged_records_root,
            staged_file_name,
        )
        staged_entries.append(
            {
                "record_kind": "direct",
                "label": source_label,
                "usage": "direct_source_record",
                "source_package_relative_path": relative_path,
                "source_path": normalize_path_value(source_record_path),
                "staged_path": normalize_path_value(staged_record_path),
                "package_relative_path": f"{PARALLEL_ATTESTATION_STATS_STAGED_RECORDS_RELATIVE_ROOT}/{staged_file_name}",
                "sha256": compute_file_sha256(staged_record_path),
                "score_name": EVENT_ATTESTATION_SCORE_NAME,
                "payload": source_record_payload,
                "source_score": _resolve_event_attestation_score(source_record_payload),
            }
        )
    return staged_entries


def _select_template_record(staged_entries: List[Dict[str, Any]], label: bool) -> Dict[str, Any]:
    if not isinstance(staged_entries, list) or not staged_entries:
        raise ValueError("staged_entries must be non-empty list")
    for entry in staged_entries:
        if entry.get("label") is label:
            return entry
    return staged_entries[0]


def _build_parallel_attestation_statistics_inputs(
    *,
    extracted_root: Path,
    run_root: Path,
    runtime_state_root: Path,
    source_contract_path: Path,
    source_contract_payload: Dict[str, Any],
    source_manifest: Dict[str, Any],
    source_info: Dict[str, Any],
    build_cfg: Dict[str, Any],
    stage_run_id: str,
) -> tuple[Path, Dict[str, Any]]:
    if not isinstance(extracted_root, Path):
        raise TypeError("extracted_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(runtime_state_root, Path):
        raise TypeError("runtime_state_root must be Path")
    if not isinstance(source_contract_path, Path):
        raise TypeError("source_contract_path must be Path")
    if not isinstance(source_contract_payload, dict):
        raise TypeError("source_contract_payload must be dict")
    if not isinstance(source_manifest, dict):
        raise TypeError("source_manifest must be dict")
    if not isinstance(source_info, dict):
        raise TypeError("source_info must be dict")
    if not isinstance(build_cfg, dict):
        raise TypeError("build_cfg must be dict")
    if not isinstance(stage_run_id, str) or not stage_run_id:
        raise TypeError("stage_run_id must be non-empty str")

    _validate_source_contract_for_build(source_contract_payload)

    staged_records_root = ensure_directory(runtime_state_root / "parallel_attestation_statistics_detect_records")
    direct_entries = _stage_direct_source_records(extracted_root, staged_records_root, source_contract_payload)
    positive_count = sum(1 for entry in direct_entries if entry["label"] is True)
    negative_count = sum(1 for entry in direct_entries if entry["label"] is False)
    desired_pair_count = max(build_cfg["target_pair_count"], positive_count, negative_count)
    missing_positive = max(0, desired_pair_count - positive_count)
    missing_negative = max(0, desired_pair_count - negative_count)
    needs_derived = missing_positive > 0 or missing_negative > 0
    requested_build_mode = str(build_cfg["requested_build_mode"])

    if needs_derived and requested_build_mode == DIRECT_SOURCE_ONLY_MODE:
        raise ValueError(
            "parallel_attestation_statistics direct_source_only requires label-balanced source records: "
            f"positive={positive_count}, negative={negative_count}"
        )

    resolved_build_mode = DIRECT_SOURCE_ONLY_MODE
    prompts: List[str] = []
    normalized_prompt_file = "<absent>"
    if needs_derived:
        if build_cfg.get("allow_derived_input_build") is not True:
            raise ValueError("parallel_attestation_statistics derived build is disabled by config")
        prompt_file = build_cfg.get("derived_input_prompts_file")
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise ValueError("parallel_attestation_statistics derived build requires derived_input_prompts_file")
        prompts = _resolve_prompt_lines(prompt_file)
        normalized_prompt_file = normalize_path_value(prompt_file)
        if len(prompts) < (missing_positive + missing_negative):
            raise ValueError(
                "parallel_attestation_statistics derived build requires sufficient prompt lines: "
                f"needed={missing_positive + missing_negative}, available={len(prompts)}"
            )
        resolved_build_mode = SOURCE_PLUS_DERIVED_PAIRS_MODE

    prompt_cursor = 0
    derived_entries: List[Dict[str, Any]] = []
    next_index = len(direct_entries) + 1
    for target_label, missing_count in ((True, missing_positive), (False, missing_negative)):
        for _ in range(missing_count):
            template_entry = _select_template_record(direct_entries, not target_label)
            template_payload = copy.deepcopy(cast(Dict[str, Any], template_entry["payload"]))
            prompt_value = prompts[prompt_cursor] if prompt_cursor < len(prompts) else None
            prompt_index = prompt_cursor
            prompt_cursor += 1

            template_payload["label"] = target_label
            template_payload["ground_truth"] = target_label
            template_payload["is_watermarked"] = target_label
            template_payload["ground_truth_source"] = (
                "parallel_attestation_statistics_derived_positive"
                if target_label
                else "parallel_attestation_statistics_derived_negative"
            )
            template_payload["run_id"] = str(uuid.uuid4())
            if isinstance(prompt_value, str) and prompt_value:
                template_payload["inference_prompt"] = prompt_value

            build_metadata = {
                "record_origin": "derived_record",
                "build_usage": build_cfg["build_usage"],
                "source_stage_name": source_manifest.get("stage_name"),
                "source_stage_run_id": source_manifest.get("stage_run_id"),
                "derived_from": template_entry["source_package_relative_path"],
                "derivation_kind": build_cfg["record_derivation_kind"],
                "label": target_label,
                "prompt_file": normalized_prompt_file,
                "prompt_index": prompt_index,
                "prompt_text": prompt_value,
            }
            template_payload["parallel_attestation_statistics_build"] = build_metadata

            if target_label:
                template_score = template_entry.get("source_score")
                if not isinstance(template_score, float):
                    raise ValueError(
                        "parallel_attestation_statistics positive derived build requires source event_attestation_score"
                    )
                _apply_positive_attestation_semantics(template_payload, template_score)
            else:
                _apply_negative_attestation_semantics(template_payload)

            label_tag = "positive" if target_label else "negative"
            staged_file_name = f"{next_index:03d}_derived_{label_tag}.json"
            staged_record_path = staged_records_root / staged_file_name
            write_json_atomic(staged_record_path, template_payload)
            derived_entries.append(
                {
                    "record_kind": "derived",
                    "label": target_label,
                    "usage": build_cfg["build_usage"],
                    "derived_from": template_entry["source_package_relative_path"],
                    "derivation_kind": build_cfg["record_derivation_kind"],
                    "source_package_relative_path": template_entry["source_package_relative_path"],
                    "staged_path": normalize_path_value(staged_record_path),
                    "package_relative_path": f"{PARALLEL_ATTESTATION_STATS_STAGED_RECORDS_RELATIVE_ROOT}/{staged_file_name}",
                    "sha256": compute_file_sha256(staged_record_path),
                    "score_name": EVENT_ATTESTATION_SCORE_NAME,
                    "prompt_file": normalized_prompt_file,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_value,
                }
            )
            next_index += 1

    final_entries: List[Dict[str, Any]] = []
    for entry in direct_entries:
        final_entries.append({key: value for key, value in entry.items() if key not in {"payload", "source_score"}})
    final_entries.extend(derived_entries)

    final_positive_count = sum(1 for entry in final_entries if entry["label"] is True)
    final_negative_count = sum(1 for entry in final_entries if entry["label"] is False)
    final_label_balanced = final_positive_count > 0 and final_positive_count == final_negative_count
    if not final_label_balanced:
        raise ValueError(
            "parallel_attestation_statistics explicit build must produce label-balanced detect records: "
            f"positive={final_positive_count}, negative={final_negative_count}"
        )

    build_contract_path = run_root / PARALLEL_ATTESTATION_STATS_BUILD_CONTRACT_RELATIVE_PATH
    build_contract_payload: Dict[str, Any] = {
        "artifact_type": "parallel_attestation_statistics_build_contract",
        "contract_role": "stats_input_build_contract",
        "contract_version": "v1",
        "stage_name": STAGE_02_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_name": source_manifest.get("stage_name"),
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "source_package_manifest_digest": str(source_info["package_manifest_digest"]),
        "source_source_contract_path": normalize_path_value(source_contract_path),
        "source_source_contract_package_relative_path": source_manifest.get(
            "parallel_attestation_statistics_input_contract_package_relative_path",
            PARALLEL_ATTESTATION_STATS_SOURCE_CONTRACT_RELATIVE_PATH,
        ),
        "requested_build_mode": requested_build_mode,
        "build_mode": resolved_build_mode,
        "score_name": EVENT_ATTESTATION_SCORE_NAME,
        "source_contract_summary": {
            "status": source_contract_payload.get("status"),
            "reason": source_contract_payload.get("reason"),
            "source_records_available": source_contract_payload.get("source_records_available"),
            "direct_stats_ready": source_contract_payload.get("direct_stats_ready"),
            "direct_stats_reason": source_contract_payload.get("direct_stats_reason"),
            "record_count": source_contract_payload.get("record_count"),
            "label_summary": source_contract_payload.get("label_summary"),
        },
        "build_configuration": {
            "allow_derived_input_build": build_cfg["allow_derived_input_build"],
            "derived_input_prompts_file": normalized_prompt_file,
            "target_pair_count": build_cfg["target_pair_count"],
            "build_usage": build_cfg["build_usage"],
            "record_derivation_kind": build_cfg["record_derivation_kind"],
        },
        "staged_records_root": normalize_path_value(staged_records_root),
        "detect_records_glob": normalize_path_value(staged_records_root / "*.json"),
        "direct_record_count": len(direct_entries),
        "derived_record_count": len(derived_entries),
        "final_record_count": len(final_entries),
        "final_positive_count": final_positive_count,
        "final_negative_count": final_negative_count,
        "final_label_balanced": final_label_balanced,
        "direct_only": len(derived_entries) == 0,
        "records": final_entries,
        "direct_records": [entry for entry in final_entries if entry["record_kind"] == "direct"],
        "derived_records": derived_entries,
    }
    write_json_atomic(build_contract_path, build_contract_payload)
    return build_contract_path, build_contract_payload


def _build_runtime_config(cfg_obj: Dict[str, Any], build_contract_payload: Dict[str, Any], run_root: Path) -> Dict[str, Any]:
    parallel_cfg = cfg_obj.get("parallel_attestation_statistics")
    parallel_section: Dict[str, Any] = dict(cast(Dict[str, Any], parallel_cfg)) if isinstance(parallel_cfg, dict) else {}
    config_copy = json.loads(json.dumps(cfg_obj))
    detect_records_glob = build_contract_payload.get("detect_records_glob")
    if not isinstance(detect_records_glob, str) or not detect_records_glob:
        raise TypeError("build_contract_payload.detect_records_glob must be non-empty str")

    calibration_cfg = dict(config_copy.get("calibration")) if isinstance(config_copy.get("calibration"), dict) else {}
    calibration_cfg["score_name"] = parallel_section.get("calibration_score_name", EVENT_ATTESTATION_SCORE_NAME)
    calibration_cfg["detect_records_glob"] = detect_records_glob
    config_copy["calibration"] = calibration_cfg

    evaluate_cfg = dict(config_copy.get("evaluate")) if isinstance(config_copy.get("evaluate"), dict) else {}
    evaluate_cfg["score_name"] = parallel_section.get("evaluate_score_name", EVENT_ATTESTATION_SCORE_NAME)
    evaluate_cfg["detect_records_glob"] = detect_records_glob
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


def _package_outputs(
    run_root: Path,
    runtime_state_root: Path,
    stage_manifest_path: Path,
    runtime_config_snapshot_path: Path,
    source_stage_manifest_copy_path: Path,
    build_contract_path: Path,
    build_contract_payload: Dict[str, Any],
) -> Path:
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
        "artifacts/parallel_attestation_statistics_build_contract.json": build_contract_path,
        "runtime_metadata/runtime_config_snapshot.yaml": runtime_config_snapshot_path,
        "lineage/source_stage_manifest.json": source_stage_manifest_copy_path,
        "lineage/source_package_manifest.json": runtime_state_root / "lineage" / "source_package_manifest.json",
    }.items():
        stage_relative_copy(source_path, package_root, relative_path)

    for record_entry in cast(List[Dict[str, Any]], build_contract_payload.get("records", [])):
        staged_path = record_entry.get("staged_path")
        package_relative_path = record_entry.get("package_relative_path")
        if not isinstance(staged_path, str) or not staged_path:
            raise ValueError("parallel_attestation_statistics_build_contract record missing staged_path")
        if not isinstance(package_relative_path, str) or not package_relative_path:
            raise ValueError("parallel_attestation_statistics_build_contract record missing package_relative_path")
        stage_relative_copy(Path(staged_path), package_root, package_relative_path)
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
    source_lineage_paths = resolve_source_lineage_paths(extracted_root)
    missing_source_lineage = [
        label for label, path_obj in source_lineage_paths.items()
        if label != "source_package_manifest_path" and not path_obj.exists()
    ]
    if missing_source_lineage:
        raise FileNotFoundError(f"stage 02 source lineage files missing: {missing_source_lineage}")

    source_contract_path, source_contract_payload = _load_parallel_attestation_statistics_source_contract(
        extracted_root,
        source_manifest,
    )

    cfg_obj = load_yaml_mapping(config_path)
    ensure_attestation_env_bootstrap(
        cfg_obj,
        drive_project_root,
        allow_generate=False,
        allow_missing=True,
    )
    build_cfg = _resolve_parallel_attestation_statistics_build_config(cfg_obj)
    build_contract_path, build_contract_payload = _build_parallel_attestation_statistics_inputs(
        extracted_root=extracted_root,
        run_root=run_root,
        runtime_state_root=runtime_state_root,
        source_contract_path=source_contract_path,
        source_contract_payload=source_contract_payload,
        source_manifest=source_manifest,
        source_info=source_info,
        build_cfg=build_cfg,
        stage_run_id=stage_run_id,
    )

    runtime_config_snapshot_path = runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml"
    runtime_cfg = _build_runtime_config(cfg_obj, build_contract_payload, run_root)
    write_yaml_mapping(runtime_config_snapshot_path, runtime_cfg)

    preflight = detect_stage_02_preflight(
        runtime_config_snapshot_path,
        source_package_path,
        source_contract_path,
    )
    if not bool(preflight.get("ok", False)):
        raise RuntimeError(f"stage 02 preflight failed: {json.dumps(preflight, ensure_ascii=False, sort_keys=True)}")

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

    source_lineage_snapshot_paths = persist_source_package_lineage(runtime_state_root, source_info)
    source_stage_manifest_copy_path = source_lineage_snapshot_paths["source_stage_manifest_copy_path"]
    source_package_manifest_copy_path = source_lineage_snapshot_paths["source_package_manifest_copy_path"]

    source_contract_summary = cast(Dict[str, Any], build_contract_payload["source_contract_summary"])
    build_contract_summary = {
        "requested_build_mode": build_contract_payload["requested_build_mode"],
        "build_mode": build_contract_payload["build_mode"],
        "direct_record_count": build_contract_payload["direct_record_count"],
        "derived_record_count": build_contract_payload["derived_record_count"],
        "final_record_count": build_contract_payload["final_record_count"],
        "final_positive_count": build_contract_payload["final_positive_count"],
        "final_negative_count": build_contract_payload["final_negative_count"],
        "final_label_balanced": build_contract_payload["final_label_balanced"],
        "direct_only": build_contract_payload["direct_only"],
        "detect_records_glob": build_contract_payload["detect_records_glob"],
    }

    workflow_summary_path = run_root / "artifacts" / "workflow_summary.json"
    write_json_atomic(workflow_summary_path, {
        "stage_name": STAGE_02_NAME,
        "stage_run_id": stage_run_id,
        "source_stage_run_id": source_manifest.get("stage_run_id"),
        "source_package_path": str(source_info["source_package_path"]),
        "source_package_sha256": str(source_info["source_package_sha256"]),
        "source_contract_summary": source_contract_summary,
        "build_contract_summary": build_contract_summary,
        "build_contract_path": normalize_path_value(build_contract_path),
        "build_mode": build_contract_payload["build_mode"],
        "direct_record_count": build_contract_payload["direct_record_count"],
        "derived_record_count": build_contract_payload["derived_record_count"],
        "direct_only": build_contract_payload["direct_only"],
        "detect_records_glob": build_contract_payload["detect_records_glob"],
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
        "source_parallel_attestation_statistics_input_contract_source_records_available": source_contract_payload.get("source_records_available"),
        "source_parallel_attestation_statistics_input_contract_direct_stats_ready": source_contract_payload.get("direct_stats_ready"),
        "source_parallel_attestation_statistics_input_contract_direct_stats_reason": source_contract_payload.get("direct_stats_reason"),
        "parallel_attestation_statistics_source_contract_summary": source_contract_summary,
        "parallel_attestation_statistics_build_contract_path": normalize_path_value(build_contract_path),
        "parallel_attestation_statistics_build_contract_package_relative_path": PARALLEL_ATTESTATION_STATS_BUILD_CONTRACT_RELATIVE_PATH,
        "parallel_attestation_statistics_build_contract_summary": build_contract_summary,
        "parallel_attestation_statistics_build_mode": build_contract_payload["build_mode"],
        "parallel_attestation_statistics_requested_build_mode": build_contract_payload["requested_build_mode"],
        "parallel_attestation_statistics_direct_record_count": build_contract_payload["direct_record_count"],
        "parallel_attestation_statistics_derived_record_count": build_contract_payload["derived_record_count"],
        "parallel_attestation_statistics_direct_only": build_contract_payload["direct_only"],
        "parallel_attestation_statistics_detect_records_glob": build_contract_payload["detect_records_glob"],
        "source_stage_manifest_path": normalize_path_value(source_stage_manifest_copy_path),
        "source_runtime_config_snapshot_path": normalize_path_value(source_lineage_paths["source_runtime_config_snapshot_path"]),
        "source_prompt_snapshot_path": resolve_source_prompt_snapshot_path(extracted_root),
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

    package_root = _package_outputs(
        run_root,
        runtime_state_root,
        stage_manifest_path,
        runtime_config_snapshot_path,
        source_stage_manifest_copy_path,
        build_contract_path,
        build_contract_payload,
    )
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
