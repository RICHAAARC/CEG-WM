"""
文件目的：workflow 验收摘要共用工具。
Module type: General module

职责边界：
1. 只读取现有 run_root 下的 records 与 artifacts，生成脚本级验收摘要。
2. 不改写正式 records，不改变冻结语义、阈值语义或判决出口。
3. CPU smoke 与 paper_full GPU 验收共享同一套观测逻辑，避免平行实现漂移。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, cast

from main.core.records_io import write_artifact_json_unbound


def _as_dict(value: Any) -> Dict[str, Any]:
    """
    功能：将未知对象收敛为 dict 视图。

    Normalize a dynamically loaded object to a dictionary.

    Args:
        value: Arbitrary object.

    Returns:
        Dictionary view when value is a mapping, otherwise an empty dict.
    """
    if isinstance(value, dict):
        return cast(Dict[str, Any], value)
    return {}


def load_json_dict(path: Path) -> Dict[str, Any]:
    """
    功能：加载 JSON 文件并要求根为 mapping。

    Load a JSON file and require a mapping root.

    Args:
        path: JSON file path.

    Returns:
        Parsed mapping, or an empty mapping when file is absent or invalid.
    """
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def load_runtime_config(cfg_path: Path) -> Dict[str, Any]:
    """
    功能：加载运行期配置。

    Load runtime configuration from YAML-like JSON-safe text.

    Args:
        cfg_path: Config file path.

    Returns:
        Parsed mapping, or an empty mapping when loading fails.
    """
    try:
        import yaml
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def collect_workflow_state(run_root: Path) -> Dict[str, Any]:
    """
    功能：收集 run_root 下的核心 records 与 artifacts 状态。

    Collect core workflow state from records and artifacts under one run_root.

    Args:
        run_root: Workflow run root.

    Returns:
        Mapping with loaded records, artifacts, and existence flags.
    """
    embed_record_path = run_root / "records" / "embed_record.json"
    detect_record_path = run_root / "records" / "detect_record.json"
    calibration_record_path = run_root / "records" / "calibration_record.json"
    evaluate_record_path = run_root / "records" / "evaluate_record.json"
    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    signoff_report_path = run_root / "artifacts" / "signoff" / "signoff_report.json"
    attestation_statement_path = run_root / "artifacts" / "attestation" / "attestation_statement.json"
    attestation_bundle_path = run_root / "artifacts" / "attestation" / "attestation_bundle.json"
    attestation_result_path = run_root / "artifacts" / "attestation" / "attestation_result.json"
    thresholds_artifact_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    experiment_matrix_summary_path = run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"

    embed_record = load_json_dict(embed_record_path)
    detect_record = load_json_dict(detect_record_path)
    calibration_record = load_json_dict(calibration_record_path)
    evaluate_record = load_json_dict(evaluate_record_path)
    evaluation_report = load_json_dict(evaluation_report_path)
    signoff_report = load_json_dict(signoff_report_path)

    nested_report = _as_dict(evaluate_record.get("evaluation_report"))
    if nested_report:
        evaluation_report = nested_report

    return {
        "paths": {
            "run_root": str(run_root),
            "embed_record": str(embed_record_path),
            "detect_record": str(detect_record_path),
            "calibration_record": str(calibration_record_path),
            "evaluate_record": str(evaluate_record_path),
            "evaluation_report": str(evaluation_report_path),
            "signoff_report": str(signoff_report_path),
            "thresholds_artifact": str(thresholds_artifact_path),
            "experiment_matrix_summary": str(experiment_matrix_summary_path),
        },
        "exists": {
            "embed_record": embed_record_path.exists(),
            "detect_record": detect_record_path.exists(),
            "calibration_record": calibration_record_path.exists(),
            "evaluate_record": evaluate_record_path.exists(),
            "evaluation_report": evaluation_report_path.exists(),
            "signoff_report": signoff_report_path.exists(),
            "thresholds_artifact": thresholds_artifact_path.exists(),
            "experiment_matrix_summary": experiment_matrix_summary_path.exists(),
            "attestation_statement": attestation_statement_path.exists(),
            "attestation_bundle": attestation_bundle_path.exists(),
            "attestation_result": attestation_result_path.exists(),
        },
        "embed_record": embed_record,
        "detect_record": detect_record,
        "calibration_record": calibration_record,
        "evaluate_record": evaluate_record,
        "evaluation_report": evaluation_report,
        "signoff_report": signoff_report,
    }


def _extract_content_payload(detect_record: Dict[str, Any]) -> Dict[str, Any]:
    for key_name in ("content_evidence", "content_evidence_payload"):
        payload = _as_dict(detect_record.get(key_name))
        if payload:
            return payload
    return {}


def _extract_geometry_payload(detect_record: Dict[str, Any]) -> Dict[str, Any]:
    for key_name in ("geometry_evidence", "geometry_evidence_payload"):
        payload = _as_dict(detect_record.get(key_name))
        if payload:
            return payload
    return {}


def _extract_pipeline_runtime_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    embed_record = _as_dict(state.get("embed_record"))
    detect_record = _as_dict(state.get("detect_record"))
    for record in (detect_record, embed_record):
        runtime_meta = _as_dict(record.get("pipeline_runtime_meta"))
        if runtime_meta:
            return runtime_meta
    return {}


def _compute_geo_score_semantics_ok(geometry_payload: Dict[str, Any]) -> bool:
    geo_score = geometry_payload.get("geo_score")
    status = geometry_payload.get("status")
    if status == "ok":
        return isinstance(geo_score, (int, float))
    return geo_score is None


def build_cpu_smoke_summary(run_root: Path, cfg_path: Path, workflow_exit_code: int) -> Dict[str, Any]:
    """
    功能：构建 CPU smoke 验收摘要。

    Build CPU smoke acceptance summary for lightweight closure verification.

    Args:
        run_root: Workflow run root.
        cfg_path: Runtime config path.
        workflow_exit_code: Exit code returned by workflow runner.

    Returns:
        Structured smoke acceptance summary.
    """
    cfg_obj = load_runtime_config(cfg_path)
    state = collect_workflow_state(run_root)
    detect_cfg = _as_dict(cfg_obj.get("detect"))
    content_cfg = _as_dict(detect_cfg.get("content"))
    geometry_cfg = _as_dict(detect_cfg.get("geometry"))
    attestation_cfg = _as_dict(cfg_obj.get("attestation"))
    detect_record = state["detect_record"]
    calibration_record = state["calibration_record"]
    evaluation_report = state["evaluation_report"]
    signoff_report = state["signoff_report"]
    runtime_meta = _extract_pipeline_runtime_meta(state)
    content_payload = _extract_content_payload(detect_record)
    geometry_payload = _extract_geometry_payload(detect_record)
    attestation_payload = _as_dict(detect_record.get("attestation"))
    final_decision = _as_dict(detect_record.get("final_decision"))

    readonly_guard = _as_dict(evaluation_report.get("thresholds_readonly_guard"))
    signoff_summary = _as_dict(signoff_report.get("summary"))
    content_expected = bool(content_cfg.get("enabled", False))
    geometry_expected = bool(geometry_cfg.get("enabled", False))
    attestation_expected = bool(attestation_cfg.get("enabled", False))

    workflow_runnable = bool(
        state["exists"]["embed_record"]
        and state["exists"]["detect_record"]
        and state["exists"]["calibration_record"]
        and state["exists"]["evaluate_record"]
        and state["exists"]["evaluation_report"]
    )
    records_complete = workflow_runnable
    attestation_path_selected = bool(state["exists"]["attestation_statement"] or attestation_payload)
    content_path_selected = bool(content_payload)
    geometry_path_selected = bool(geometry_payload)
    fusion_path_selected = bool(detect_record.get("fusion_result") or final_decision)
    calibrate_readonly_ok = bool(calibration_record.get("thresholds_artifact_path") and state["exists"]["thresholds_artifact"])
    evaluate_readonly_ok = bool(readonly_guard.get("unchanged") is True)
    lightweight_validation_mode = bool(runtime_meta.get("synthetic_pipeline", False) or cfg_obj.get("pipeline_impl_id") == "sd3_diffusers_shell")
    freeze_signoff_decision = signoff_summary.get("freeze_signoff_decision")
    if not isinstance(freeze_signoff_decision, str) or not freeze_signoff_decision:
        freeze_signoff_decision = signoff_report.get("freeze_signoff_decision")
    if not isinstance(freeze_signoff_decision, str) or not freeze_signoff_decision:
        freeze_signoff_decision = signoff_summary.get("FreezeSignoffDecision", "<absent>")
    smoke_verdict = bool(
        workflow_exit_code == 0
        and workflow_runnable
        and records_complete
        and (attestation_path_selected if attestation_expected else True)
        and (content_path_selected if content_expected else True)
        and (geometry_path_selected if geometry_expected else True)
        and fusion_path_selected
        and calibrate_readonly_ok
        and evaluate_readonly_ok
    )

    return {
        "profile_role": "cpu_smoke",
        "config_path": str(cfg_path),
        "workflow_exit_code": workflow_exit_code,
        "workflow_runnable": workflow_runnable,
        "records_complete": records_complete,
        "attestation_expected": attestation_expected,
        "attestation_path_selected": attestation_path_selected,
        "content_expected": content_expected,
        "content_path_selected": content_path_selected,
        "geometry_expected": geometry_expected,
        "geometry_path_selected": geometry_path_selected,
        "fusion_path_selected": fusion_path_selected,
        "calibrate_readonly_ok": calibrate_readonly_ok,
        "evaluate_readonly_ok": evaluate_readonly_ok,
        "lightweight_validation_mode": lightweight_validation_mode,
        "smoke_verdict": smoke_verdict,
        "signoff_decision": freeze_signoff_decision,
        "details": {
            "pipeline_impl_id": detect_record.get("pipeline_impl_id", state["embed_record"].get("pipeline_impl_id", "<absent>")),
            "synthetic_pipeline": bool(runtime_meta.get("synthetic_pipeline", False)),
            "runtime_device": runtime_meta.get("device", cfg_obj.get("device", "<absent>")),
            "authenticity_result": attestation_payload.get("authenticity_result"),
            "image_evidence_result": attestation_payload.get("image_evidence_result"),
            "final_event_attested_decision": attestation_payload.get("final_event_attested_decision"),
            "geo_score_semantics_ok": _compute_geo_score_semantics_ok(geometry_payload) if geometry_payload else True,
        },
        "paths": state["paths"],
    }


def build_formal_gpu_summary(
    run_root: Path,
    cfg_path: Path,
    workflow_exit_code: int,
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：构建 formal GPU 验收摘要。

    Build formal GPU acceptance summary for paper_full workflow verification.

    Args:
        run_root: Workflow run root.
        cfg_path: Runtime config path.
        workflow_exit_code: Exit code returned by workflow runner.
        preflight: GPU/attestation preflight result mapping.

    Returns:
        Structured formal acceptance summary.
    """
    cfg_obj = load_runtime_config(cfg_path)
    state = collect_workflow_state(run_root)
    detect_record = state["detect_record"]
    calibration_record = state["calibration_record"]
    evaluation_report = state["evaluation_report"]
    signoff_report = state["signoff_report"]
    runtime_meta = _extract_pipeline_runtime_meta(state)
    content_payload = _extract_content_payload(detect_record)
    geometry_payload = _extract_geometry_payload(detect_record)
    attestation_payload = _as_dict(detect_record.get("attestation"))
    signoff_summary = _as_dict(signoff_report.get("summary"))

    readonly_guard = _as_dict(evaluation_report.get("thresholds_readonly_guard"))
    pipeline_execution_ok = bool(
        workflow_exit_code == 0
        and state["exists"]["embed_record"]
        and state["exists"]["detect_record"]
        and state["exists"]["calibration_record"]
        and state["exists"]["evaluate_record"]
        and state["exists"]["evaluation_report"]
        and state["exists"]["experiment_matrix_summary"]
    )
    model_loaded = bool(
        detect_record.get("pipeline_impl_id", state["embed_record"].get("pipeline_impl_id")) == "sd3_diffusers_real"
        and not bool(runtime_meta.get("synthetic_pipeline", False))
    )
    gpu_used = bool(str(runtime_meta.get("device", cfg_obj.get("device", ""))).lower() == "cuda" and model_loaded)
    attestation_bundle_generated = bool(state["exists"]["attestation_bundle"])
    content_path_selected = bool(content_payload)
    lf_path_selected = bool(content_payload.get("lf_score") is not None or content_payload.get("lf_evidence_summary"))
    hf_path_selected = bool(content_payload.get("hf_score") is not None or content_payload.get("hf_evidence_summary"))
    geometry_path_selected = bool(geometry_payload)
    fusion_path_selected = bool(detect_record.get("fusion_result") or detect_record.get("final_decision"))
    calibrate_readonly_ok = bool(calibration_record.get("thresholds_artifact_path") and state["exists"]["thresholds_artifact"])
    evaluate_readonly_ok = bool(readonly_guard.get("unchanged") is True)
    geo_score_semantics_ok = _compute_geo_score_semantics_ok(geometry_payload) if geometry_payload else True
    freeze_signoff_decision = signoff_summary.get("freeze_signoff_decision")
    if not isinstance(freeze_signoff_decision, str) or not freeze_signoff_decision:
        freeze_signoff_decision = signoff_report.get("freeze_signoff_decision")
    if not isinstance(freeze_signoff_decision, str) or not freeze_signoff_decision:
        freeze_signoff_decision = signoff_summary.get("FreezeSignoffDecision", "<absent>")
    formal_output_expectation_ok = bool(
        pipeline_execution_ok
        and bool(preflight.get("ok", False))
        and model_loaded
        and gpu_used
        and attestation_bundle_generated
        and content_path_selected
        and lf_path_selected
        and hf_path_selected
        and geometry_path_selected
        and fusion_path_selected
        and calibrate_readonly_ok
        and evaluate_readonly_ok
        and geo_score_semantics_ok
        and freeze_signoff_decision == "ALLOW_FREEZE"
    )

    return {
        "profile_role": "paper_full_cuda_formal",
        "config_path": str(cfg_path),
        "workflow_exit_code": workflow_exit_code,
        "pipeline_execution_ok": pipeline_execution_ok,
        "formal_output_expectation_ok": formal_output_expectation_ok,
        "environment_blocked": not bool(preflight.get("ok", False)),
        "model_loaded": model_loaded,
        "gpu_used": gpu_used,
        "attestation_bundle_generated": attestation_bundle_generated,
        "content_path_selected": content_path_selected,
        "lf_path_selected": lf_path_selected,
        "hf_path_selected": hf_path_selected,
        "geometry_path_selected": geometry_path_selected,
        "fusion_path_selected": fusion_path_selected,
        "calibrate_readonly_ok": calibrate_readonly_ok,
        "evaluate_readonly_ok": evaluate_readonly_ok,
        "geo_score_semantics_ok": geo_score_semantics_ok,
        "signoff_decision": freeze_signoff_decision,
        "details": {
            "pipeline_impl_id": detect_record.get("pipeline_impl_id", state["embed_record"].get("pipeline_impl_id", "<absent>")),
            "synthetic_pipeline": bool(runtime_meta.get("synthetic_pipeline", False)),
            "runtime_device": runtime_meta.get("device", cfg_obj.get("device", "<absent>")),
            "authenticity_result": attestation_payload.get("authenticity_result"),
            "image_evidence_result": attestation_payload.get("image_evidence_result"),
            "final_event_attested_decision": attestation_payload.get("final_event_attested_decision"),
            "preflight": preflight,
        },
        "paths": state["paths"],
    }


def detect_formal_gpu_preflight(cfg_path: Path) -> Dict[str, Any]:
    """
    功能：执行 formal GPU 验收前置环境检查。

    Execute preflight checks for formal GPU verification.

    Args:
        cfg_path: Runtime config path.

    Returns:
        Preflight result with ok flag and blocking reasons.
    """
    cfg_obj = load_runtime_config(cfg_path)
    attestation_cfg = _as_dict(cfg_obj.get("attestation"))
    required_env_vars: list[str] = []
    for key_name in ("k_master_env_var", "k_prompt_env_var", "k_seed_env_var"):
        value = attestation_cfg.get(key_name)
        if isinstance(value, str) and value:
            required_env_vars.append(value)

    missing_env_vars = [item for item in required_env_vars if not os.environ.get(item)]
    nvidia_smi_path = shutil.which("nvidia-smi")

    return {
        "ok": bool(nvidia_smi_path and not missing_env_vars),
        "gpu_tool_available": bool(nvidia_smi_path),
        "nvidia_smi_path": nvidia_smi_path or "<absent>",
        "missing_attestation_env_vars": missing_env_vars,
    }


def write_workflow_summary(run_root: Path, file_name: str, summary: Dict[str, Any]) -> Path:
    """
    功能：写入脚本级 workflow 验收摘要。

    Write a script-level workflow acceptance summary under artifacts/workflow_acceptance.

    Args:
        run_root: Workflow run root.
        file_name: Summary file name.
        summary: Summary mapping.

    Returns:
        Written summary path.
    """
    if not file_name:
        raise TypeError("file_name must be non-empty str")

    artifacts_dir = run_root / "artifacts"
    summary_path = artifacts_dir / "workflow_acceptance" / file_name
    write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(summary_path),
        obj=summary,
        indent=2,
        ensure_ascii=False,
    )
    return summary_path
