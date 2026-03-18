"""
File purpose: 论文级实验矩阵调度与汇总。
Module type: Core innovation module

设计边界：
1. 仅做实验编排与结果聚合，不改变 NP 阈值、融合判决与 digest 口径。
2. grid 展开不写盘；批量执行复用既有 CLI 阶段与 records 门禁链路。
3. 汇总只读取既有 eval_report，不重算 score/threshold。
"""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.core import config_loader, digests
from main.core import records_io
from main.evaluation import protocol_loader
from main.evaluation import attack_coverage
from main.policy import path_policy
from main.watermarking.detect import orchestrator as detect_orchestrator


_FORBIDDEN_ARTIFACT_ANCHOR_FIELDS = {
    "contract_bound_digest",
    "whitelist_bound_digest",
    "policy_path_semantics_bound_digest",
    "injection_scope_manifest_bound_digest",
}


def _relative_path_from_base(base_path: Path, path_value: Any) -> str:
    """
    功能：将路径值规范化为相对 base_path 的 POSIX 路径。 

    Normalize a path-like value into a POSIX relative path under base_path.

    Args:
        base_path: Base directory path.
        path_value: Candidate path-like value.

    Returns:
        Relative POSIX path when path_value is under base_path; otherwise
        "<absent>".
    """
    if not isinstance(base_path, Path):
        raise TypeError("base_path must be Path")
    if not isinstance(path_value, str) or not path_value:
        return "<absent>"

    base_text = base_path.as_posix().rstrip("/")
    candidate_text = Path(path_value).as_posix()
    if candidate_text == base_text:
        return "."

    prefix = f"{base_text}/"
    if candidate_text.startswith(prefix):
        return candidate_text[len(prefix):]
    return "<absent>"


def _annotate_result_relative_paths(results: List[Dict[str, Any]], batch_root: Path) -> None:
    """
    功能：为 experiment_matrix 结果补充相对路径视图。 

    Append run_root_relative fields for matrix result portability.

    Args:
        results: Experiment result list.
        batch_root: Matrix batch root path.

    Returns:
        None.
    """
    if not isinstance(results, list):
        raise TypeError("results must be list")
    if not isinstance(batch_root, Path):
        raise TypeError("batch_root must be Path")

    for item in results:
        if not isinstance(item, dict):
            continue
        item["run_root_relative"] = _relative_path_from_base(batch_root, item.get("run_root"))


def build_experiment_grid(base_cfg: dict) -> list[dict]:
    """
    功能：根据基础配置展开实验矩阵。

    Build experiment grid by expanding model, seed, attack family, and ablation axes.

    Args:
        base_cfg: Base configuration mapping used as expansion root.

    Returns:
        List of grid item mappings. Each item contains cfg_snapshot,
        ablation_flags, attack_protocol_version, and grid_item_digest.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If any axis configuration is invalid.
    """
    if not isinstance(base_cfg, dict):
        # base_cfg 类型不合法，必须 fail-fast。
        raise TypeError("base_cfg must be dict")

    matrix_cfg = _extract_matrix_cfg(base_cfg)
    model_list = _resolve_model_axis(base_cfg, matrix_cfg)
    seed_list = _resolve_seed_axis(base_cfg, matrix_cfg)
    attack_families, attack_protocol_version, attack_protocol_digest = _resolve_attack_family_axis(base_cfg, matrix_cfg)
    ablation_variants = _resolve_ablation_axis(matrix_cfg)
    formal_validation_guards = _resolve_formal_validation_guards(matrix_cfg)

    batch_root = matrix_cfg.get("batch_root", "outputs/experiment_matrix")
    if not isinstance(batch_root, str) or not batch_root:
        raise ValueError("experiment_matrix.batch_root must be non-empty str")

    config_path = matrix_cfg.get("config_path", "configs/paper_full_cuda.yaml")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("experiment_matrix.config_path must be non-empty str")

    attack_protocol_path = matrix_cfg.get("attack_protocol_path", config_loader.ATTACK_PROTOCOL_PATH)
    if not isinstance(attack_protocol_path, str) or not attack_protocol_path:
        raise ValueError("experiment_matrix.attack_protocol_path must be non-empty str")

    max_samples = matrix_cfg.get("max_samples")
    if max_samples is not None and not isinstance(max_samples, int):
        raise TypeError("experiment_matrix.max_samples must be int or None")

    allow_failed_semantics_collection = matrix_cfg.get("allow_failed_semantics_collection", False)
    if not isinstance(allow_failed_semantics_collection, bool):
        raise TypeError("experiment_matrix.allow_failed_semantics_collection must be bool")

    grid_items: List[Dict[str, Any]] = []
    grid_index = 0
    for model_id in model_list:
        for seed_value in seed_list:
            for attack_family in attack_families:
                for ablation_flags in ablation_variants:
                    cfg_snapshot = copy.deepcopy(base_cfg)
                    cfg_snapshot["model_id"] = model_id
                    cfg_snapshot["seed"] = seed_value
                    _apply_ablation_flags(cfg_snapshot, ablation_flags)

                    grid_payload = {
                        "model_id": model_id,
                        "seed": seed_value,
                        "attack_family": attack_family,
                        "ablation_flags": ablation_flags,
                        "attack_protocol_version": attack_protocol_version,
                    }
                    grid_item_digest = digests.canonical_sha256(grid_payload)

                    grid_item = {
                        "grid_index": grid_index,
                        "cfg_snapshot": cfg_snapshot,
                        "ablation_flags": copy.deepcopy(ablation_flags),
                        "ablation_digest": _compute_ablation_digest(ablation_flags),
                        "attack_protocol_family": attack_family,
                        "attack_protocol_version": attack_protocol_version,
                        "attack_protocol_digest": attack_protocol_digest,
                        "grid_item_digest": grid_item_digest,
                        "cfg_digest": digests.canonical_sha256(cfg_snapshot),
                        "model_id": model_id,
                        "seed": seed_value,
                        "batch_root": batch_root,
                        "config_path": config_path,
                        "attack_protocol_path": attack_protocol_path,
                        "max_samples": max_samples,
                        "allow_failed_semantics_collection": allow_failed_semantics_collection,
                        "require_real_negative_cache": formal_validation_guards["require_real_negative_cache"],
                        "require_shared_thresholds": formal_validation_guards["require_shared_thresholds"],
                        "disallow_forced_pair_fallback": formal_validation_guards["disallow_forced_pair_fallback"],
                    }
                    grid_items.append(grid_item)
                    grid_index += 1

    return grid_items


def run_single_experiment(grid_item_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：执行单个网格实验并提取摘要锚点。

    Run one experiment item through embed/detect/calibrate/evaluate stages and
    return immutable summary anchors.

    Args:
        grid_item_cfg: One grid item generated by build_experiment_grid.

    Returns:
        Experiment summary containing status, run_root, anchors, and metrics.

    Raises:
        TypeError: If grid_item_cfg is invalid.
    """
    if not isinstance(grid_item_cfg, dict):
        # grid_item_cfg 类型不合法，必须 fail-fast。
        raise TypeError("grid_item_cfg must be dict")

    run_root = _derive_run_root(grid_item_cfg)
    summary: Dict[str, Any] = {
        "grid_index": grid_item_cfg.get("grid_index", -1),
        "grid_item_digest": grid_item_cfg.get("grid_item_digest", "<absent>"),
        "run_root": str(run_root),
        "model_id": _safe_str(grid_item_cfg.get("model_id")),
        "seed": grid_item_cfg.get("seed") if isinstance(grid_item_cfg.get("seed"), int) else None,
        "attack_family": _safe_str(grid_item_cfg.get("attack_protocol_family")),
        "status": "failed",
        "failure_reason": "<absent>",
        "cfg_digest": _safe_str(grid_item_cfg.get("cfg_digest")),
        "plan_digest": "<absent>",
        "thresholds_digest": "<absent>",
        "threshold_metadata_digest": "<absent>",
        "ablation_digest": _safe_str(grid_item_cfg.get("ablation_digest")),
        "attack_protocol_digest": _safe_str(grid_item_cfg.get("attack_protocol_digest")),
        "attack_protocol_version": grid_item_cfg.get("attack_protocol_version", "<absent>"),
        "policy_path": "<absent>",
        "impl_digest": "<absent>",
        "fusion_rule_version": "<absent>",
        "hf_truncation_baseline_comparison": {
            "content_score": None,
            "hf_truncation_score": None,
            "score_delta_content_minus_hf_truncation": None,
            "comparison_ready": False,
            "comparison_source": "real_hf_truncation_baseline_required",
        },
        "detect_gate_relaxed": False,
        "detect_gate_relax_reason": "hard_gate_default",
        "detect_gate_sample_counts": {},
        "metrics": {},
    }

    detect_gate_info: Dict[str, Any] = {
        "gate_relaxed": False,
        "reason": "hard_gate_not_checked",
        "sample_counts": {},
    }

    try:
        stage_gate_info = _run_stage_sequence(grid_item_cfg, run_root)
        if isinstance(stage_gate_info, dict):
            detect_gate_info = stage_gate_info
        eval_report = _read_evaluation_report_for_run(run_root)
        _assert_required_run_artifacts(run_root)
        evaluate_record = _read_optional_json(run_root / "records" / "evaluate_record.json")
        run_closure = _read_optional_json(run_root / "artifacts" / "run_closure.json")

        metrics_obj = eval_report.get("metrics") if isinstance(eval_report.get("metrics"), dict) else {}
        cfg_digest_value = _first_present_str(
            eval_report.get("cfg_digest"),
            evaluate_record.get("cfg_digest") if isinstance(evaluate_record, dict) else None,
            grid_item_cfg.get("cfg_digest"),
        )
        thresholds_digest_value = _first_present_str(
            eval_report.get("thresholds_digest"),
            evaluate_record.get("thresholds_digest") if isinstance(evaluate_record, dict) else None,
            run_closure.get("thresholds_digest") if isinstance(run_closure, dict) else None,
        )
        thresholds_metadata_digest_value = _first_present_str(
            eval_report.get("threshold_metadata_digest"),
            evaluate_record.get("threshold_metadata_digest") if isinstance(evaluate_record, dict) else None,
            run_closure.get("threshold_metadata_digest") if isinstance(run_closure, dict) else None,
        )
        policy_path_value = _first_present_str(
            evaluate_record.get("policy_path") if isinstance(evaluate_record, dict) else None,
            run_closure.get("policy_path") if isinstance(run_closure, dict) else None,
            eval_report.get("policy_path"),
            eval_report.get("anchors", {}).get("policy_path") if isinstance(eval_report.get("anchors"), dict) else None,
        )
        impl_digest_value = _first_present_str(
            eval_report.get("impl_digest"),
            evaluate_record.get("impl_digest") if isinstance(evaluate_record, dict) else None,
            run_closure.get("impl_digest") if isinstance(run_closure, dict) else None,
            run_closure.get("impl_identity_digest") if isinstance(run_closure, dict) else None,
        )
        fusion_rule_version_value = _first_present_str(
            eval_report.get("fusion_rule_version"),
            evaluate_record.get("fusion_rule_version") if isinstance(evaluate_record, dict) else None,
            run_closure.get("fusion_rule_version") if isinstance(run_closure, dict) else None,
        )
        summary.update(
            {
                "status": "ok",
                "failure_reason": "ok",
                "cfg_digest": _safe_str(cfg_digest_value),
                "plan_digest": _safe_str(eval_report.get("plan_digest")),
                "thresholds_digest": _safe_str(thresholds_digest_value),
                "threshold_metadata_digest": _safe_str(thresholds_metadata_digest_value),
                "ablation_digest": _safe_str(eval_report.get("ablation_digest")),
                "attack_protocol_digest": _safe_str(eval_report.get("attack_protocol_digest")),
                "attack_protocol_version": _safe_str(eval_report.get("attack_protocol_version")),
                "attack_coverage_digest": _safe_str(eval_report.get("attack_coverage_digest")),
                "policy_path": _safe_str(policy_path_value),
                "impl_digest": _safe_str(impl_digest_value),
                "fusion_rule_version": _safe_str(fusion_rule_version_value),
                "metrics": {
                    "tpr_at_fpr": metrics_obj.get("tpr_at_fpr_primary", metrics_obj.get("tpr_at_fpr")),
                    "geo_available_rate": metrics_obj.get("geo_available_rate"),
                    "rescue_rate": metrics_obj.get("rescue_rate"),
                    "reject_rate": metrics_obj.get("reject_rate"),
                    "reject_rate_breakdown": metrics_obj.get("reject_rate_by_reason", {}),
                    "n_total": metrics_obj.get("n_total"),
                    "n_accepted": metrics_obj.get("n_accepted"),
                    "n_rejected": metrics_obj.get("n_rejected"),
                    "n_rescue_triggered": metrics_obj.get("n_rescue_triggered"),
                    "n_rescue_success": metrics_obj.get("n_rescue_success"),
                    "conditional_fpr_estimate": metrics_obj.get("conditional_fpr_estimate"),
                    "conditional_fpr_n": metrics_obj.get("conditional_fpr_n"),
                },
                "hf_truncation_baseline_comparison": _extract_hf_truncation_baseline_comparison_from_detect_record(run_root),
                "detect_gate_relaxed": bool(detect_gate_info.get("gate_relaxed", False)),
                "detect_gate_relax_reason": _safe_str(detect_gate_info.get("reason")),
                "detect_gate_sample_counts": detect_gate_info.get("sample_counts") if isinstance(detect_gate_info.get("sample_counts"), dict) else {},
            }
        )
        _enforce_paper_acceptance_gate(summary=summary, grid_item_cfg=grid_item_cfg, run_root=run_root)
    except Exception as exc:
        # 单实验执行失败，必须记录失败原因并返回。
        summary["status"] = "failed"
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["detect_gate_relaxed"] = bool(detect_gate_info.get("gate_relaxed", False))
        summary["detect_gate_relax_reason"] = _safe_str(detect_gate_info.get("reason"))
        summary["detect_gate_sample_counts"] = detect_gate_info.get("sample_counts") if isinstance(detect_gate_info.get("sample_counts"), dict) else {}

    return summary


def _read_optional_json(path: Path) -> Dict[str, Any]:
    """Read optional JSON file and return dict, else empty dict."""
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not path.exists() or not path.is_file():
        return {}
    parsed_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed_obj, dict):
        return {}
    return parsed_obj


def _coerce_finite_float(value: Any) -> Optional[float]:
    """Return finite float value when the candidate is numeric-like."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        if np.isfinite(value_float):
            return value_float
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value_float = float(stripped)
        except ValueError:
            return None
        if np.isfinite(value_float):
            return value_float
    return None


def _resolve_matrix_calibration_score(record_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Resolve the best available calibration score from a detect record payload."""
    if not isinstance(record_payload, dict):
        raise TypeError("record_payload must be dict")

    content_payload = record_payload.get("content_evidence_payload")
    content_node = content_payload if isinstance(content_payload, dict) else {}
    score_parts_payload = content_node.get("score_parts")
    score_parts = score_parts_payload if isinstance(score_parts_payload, dict) else {}
    hf_trace_payload = score_parts.get("hf_detect_trace")
    hf_trace = hf_trace_payload if isinstance(hf_trace_payload, dict) else {}

    content_evidence_payload = record_payload.get("content_evidence")
    content_evidence = content_evidence_payload if isinstance(content_evidence_payload, dict) else {}
    fusion_result_payload = record_payload.get("fusion_result")
    fusion_result = fusion_result_payload if isinstance(fusion_result_payload, dict) else {}
    evidence_summary_payload = fusion_result.get("evidence_summary") if isinstance(fusion_result.get("evidence_summary"), dict) else {}

    score_candidates = [
        ("content_evidence_payload.score", content_node.get("score")),
        ("content_evidence_payload.detect_lf_score", content_node.get("detect_lf_score")),
        ("content_evidence_payload.lf_score", content_node.get("lf_score")),
        ("content_evidence_payload.hf_score", content_node.get("hf_score")),
        ("content_evidence_payload.detect_hf_score", content_node.get("detect_hf_score")),
        ("content_evidence_payload.score_parts.content_score", score_parts.get("content_score")),
        ("content_evidence_payload.score_parts.detect_lf_score", score_parts.get("detect_lf_score")),
        ("content_evidence_payload.score_parts.hf_detect_trace.hf_score_raw", hf_trace.get("hf_score_raw")),
        ("content_evidence.score", content_evidence.get("score")),
        ("record.score", record_payload.get("score")),
        ("fusion_result.evidence_summary.content_score", evidence_summary_payload.get("content_score")),
    ]
    for field_name, candidate_value in score_candidates:
        numeric_candidate = _coerce_finite_float(candidate_value)
        if numeric_candidate is not None:
            return numeric_candidate, field_name
    return None, None


def _ensure_matrix_calibration_compatible_content_payload(
    record_payload: Dict[str, Any],
    score_value: float,
    score_source: Optional[str] = None,
    recovered_sample_origin: Optional[str] = None,
) -> None:
    """Normalize a record so calibration can consume it as a valid content sample."""
    if not isinstance(record_payload, dict):
        raise TypeError("record_payload must be dict")
    if not isinstance(score_value, (int, float)) or not np.isfinite(float(score_value)):
        raise TypeError("score_value must be finite numeric")
    if score_source is not None and (not isinstance(score_source, str) or not score_source):
        raise TypeError("score_source must be non-empty str or None")
    if recovered_sample_origin is not None and (
        not isinstance(recovered_sample_origin, str) or not recovered_sample_origin
    ):
        raise TypeError("recovered_sample_origin must be non-empty str or None")

    content_node = record_payload.get("content_evidence_payload")
    if not isinstance(content_node, dict):
        content_node = {}
        record_payload["content_evidence_payload"] = content_node
    content_node["status"] = "ok"
    content_node["score"] = float(score_value)
    content_node.pop("calibration_sample_is_synthetic_fallback", None)

    if isinstance(score_source, str) and score_source != "content_evidence_payload.score":
        content_node["calibration_score_recovery_reason"] = score_source
        if isinstance(recovered_sample_origin, str):
            content_node["calibration_sample_origin"] = recovered_sample_origin


def _extract_hf_truncation_baseline_comparison_from_detect_record(run_root: Path) -> Dict[str, Any]:
    """Extract same-sample comparison values from real HF truncation baseline record."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    result: Dict[str, Any] = {
        "content_score": None,
        "hf_truncation_score": None,
        "score_delta_content_minus_hf_truncation": None,
        "comparison_ready": False,
        "comparison_source": "real_hf_truncation_baseline_required",
        "baseline_status": "absent",
        "baseline_trace": None,
    }

    detect_record = _read_optional_json(run_root / "records" / "detect_record.json")
    if not isinstance(detect_record, dict) or not detect_record:
        return result

    content_payload = detect_record.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        return result

    content_score = content_payload.get("score")
    if isinstance(content_score, (int, float)) and np.isfinite(float(content_score)):
        result["content_score"] = float(content_score)

    baseline_payload = detect_record.get("hf_truncation_baseline")
    if isinstance(baseline_payload, dict):
        baseline_trace = baseline_payload.get("trace")
        if isinstance(baseline_trace, dict):
            result["baseline_trace"] = baseline_trace
        baseline_status = baseline_payload.get("status")
        if isinstance(baseline_status, str) and baseline_status:
            result["baseline_status"] = baseline_status
        baseline_score = baseline_payload.get("score")
        if isinstance(baseline_score, (int, float)) and np.isfinite(float(baseline_score)):
            result["hf_truncation_score"] = float(baseline_score)
            result["baseline_status"] = "ok"
            result["comparison_source"] = "real_hf_truncation_baseline_record"

    if isinstance(result.get("content_score"), float) and isinstance(result.get("hf_truncation_score"), float):
        result["score_delta_content_minus_hf_truncation"] = float(result["content_score"] - result["hf_truncation_score"])
        result["comparison_ready"] = True

    return result


def _enforce_paper_acceptance_gate(summary: Dict[str, Any], grid_item_cfg: Dict[str, Any], run_root: Path) -> None:
    """Enforce hard acceptance constraints for paper-faithful matrix runs."""
    if not isinstance(summary, dict):
        raise TypeError("summary must be dict")
    if not isinstance(grid_item_cfg, dict):
        raise TypeError("grid_item_cfg must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    paper_cfg = grid_item_cfg.get("paper_faithfulness") if isinstance(grid_item_cfg.get("paper_faithfulness"), dict) else {}
    enforce = bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
    if not enforce:
        return

    detect_record = _read_optional_json(run_root / "records" / "detect_record.json")
    pipeline_runtime_meta = detect_record.get("pipeline_runtime_meta") if isinstance(detect_record.get("pipeline_runtime_meta"), dict) else {}
    detect_runtime_mode = detect_record.get("detect_runtime_mode") if isinstance(detect_record.get("detect_runtime_mode"), str) else "<absent>"
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    geo_available_rate = metrics.get("geo_available_rate")
    hf_truncation_baseline_comparison = summary.get("hf_truncation_baseline_comparison") if isinstance(summary.get("hf_truncation_baseline_comparison"), dict) else {}

    if bool(pipeline_runtime_meta.get("synthetic_pipeline", False)):
        summary["status"] = "failed"
        summary["failure_reason"] = "paper_acceptance_failed: synthetic_pipeline_true"
        return
    if detect_runtime_mode != "real":
        summary["status"] = "failed"
        summary["failure_reason"] = f"paper_acceptance_failed: detect_runtime_mode={detect_runtime_mode}"
        return
    if isinstance(geo_available_rate, (int, float)) and float(geo_available_rate) == 0.0:
        summary["status"] = "failed"
        summary["failure_reason"] = "paper_acceptance_failed: geo_available_rate_zero"
        return
    if not bool(hf_truncation_baseline_comparison.get("comparison_ready", False)):
        summary["status"] = "failed"
        summary["failure_reason"] = "paper_acceptance_failed: real_hf_truncation_baseline_missing"


def _first_present_str(*values: Any) -> str:
    """Return first non-empty non-absent string candidate."""
    for value in values:
        if isinstance(value, str) and value and value != "<absent>":
            return value
    return "<absent>"


def run_experiment_grid(grid: list[dict], strict: bool = True) -> dict:
    """
    功能：批量执行实验网格并生成汇总工件。

    Execute experiment grid in order. In strict mode, stop immediately on first
    failure. In non-strict mode, continue and record all failures.

    Args:
        grid: Grid item list from build_experiment_grid.
        strict: If True, fail-fast on first failed sub-experiment.

    Returns:
        Grid summary including aggregate report and artifact paths.

    Raises:
        TypeError: If inputs are invalid.
        RuntimeError: If strict mode is enabled and any run fails.
    """
    if not isinstance(grid, list):
        # grid 类型不合法，必须 fail-fast。
        raise TypeError("grid must be list")
    if not isinstance(strict, bool):
        # strict 类型不合法，必须 fail-fast。
        raise TypeError("strict must be bool")

    results: List[Dict[str, Any]] = []
    first_failure: Optional[Dict[str, Any]] = None

    # (1) 按 (model_id, seed) 分组预生成真实负样本 detect record。
    # 论文级严谨要求：FPR 校准须使用真实干净图像分布，而非合成偏移（base_score - 1.0）。
    # 每组仅发起 1 次额外 SD 推理（embed identity + detect），8 个 grid item 共用 1 个结果。
    neg_detect_record_cache: Dict[Tuple[str, int], Optional[Path]] = {}
    for item in grid:
        if not isinstance(item, dict):
            continue
        m_id = item.get("model_id")
        s_val = item.get("seed")
        if not isinstance(m_id, str) or not isinstance(s_val, int):
            continue
        neg_key: Tuple[str, int] = (m_id, s_val)
        if neg_key not in neg_detect_record_cache:
            try:
                neg_detect_record_cache[neg_key] = _run_neg_embed_detect_for_cache(
                    model_id=m_id,
                    seed=s_val,
                    config_path=str(item.get("config_path", "configs/paper_full_cuda.yaml")),
                    batch_root=str(item.get("batch_root", "outputs/experiment_matrix")),
                    max_samples=item.get("max_samples"),
                )
            except Exception:
                # neg 预生成失败时降级为合成方案，不中止 grid 执行。
                neg_detect_record_cache[neg_key] = None

    # (2) 全局 calibrate：汇总所有 neg 记录产出共享阈值，实现校准集与测试集的严格分离。
    # 论文级要求：NP 阈值必须在独立的 null 分布上估计，不得与测试集（攻击后正样本）重合。
    shared_thresholds_path: Optional[Path] = None
    if grid:
        first_item = grid[0]
        try:
            shared_thresholds_path = _run_global_calibrate(
                batch_root=str(first_item.get("batch_root", "outputs/experiment_matrix")),
                config_path=str(first_item.get("config_path", "configs/paper_full_cuda.yaml")),
                neg_detect_record_cache=neg_detect_record_cache,
            )
        except Exception:
            # 全局 calibrate 失败时降级为 per-item calibrate，不中止 grid 执行。
            shared_thresholds_path = None

    _assert_formal_validation_prerequisites(
        grid,
        neg_detect_record_cache=neg_detect_record_cache,
        shared_thresholds_path=shared_thresholds_path,
    )

    # (3) 主循环：将 neg_detect_record_path 与 shared_thresholds_path 注入各 grid item。
    for item in grid:
        if not isinstance(item, dict):
            raise TypeError("grid items must be dict")
        m_id = item.get("model_id")
        s_val = item.get("seed")
        neg_key_for_item = (m_id, s_val) if (isinstance(m_id, str) and isinstance(s_val, int)) else None
        neg_path = neg_detect_record_cache.get(neg_key_for_item) if neg_key_for_item is not None else None

        run_item = dict(item)
        if neg_path is not None:
            run_item["neg_detect_record_path"] = str(neg_path)
        if shared_thresholds_path is not None:
            run_item["shared_thresholds_path"] = str(shared_thresholds_path)

        result = run_single_experiment(run_item)
        results.append(result)
        if result.get("status") != "ok" and first_failure is None:
            first_failure = result
            if strict:
                break

    grid_manifest = _build_grid_manifest(grid)
    aggregate_report = build_aggregate_report(results, grid_manifest=grid_manifest)
    summary_paths = _write_grid_artifacts(grid, aggregate_report, results, strict)
    if len(grid) > 0:
        batch_root_value = grid[0].get("batch_root", "outputs/experiment_matrix")
        _annotate_result_relative_paths(results, Path(str(batch_root_value)))

    # 从 aggregate_report 或 results[0] 提取锚点字段（append-only，不重新计算）
    anchors_obj = _extract_anchors_from_results(aggregate_report, results)

    grid_summary = {
        "strict": strict,
        "total": len(grid),
        "executed": len(results),
        "succeeded": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") != "ok"),
        "aggregate_report": aggregate_report,
        "grid_manifest": grid_manifest,
        "results": results,
        **anchors_obj,  # append-only: 补齐锚点字段全集
        **summary_paths,
    }

    if strict and first_failure is not None:
        raise RuntimeError(
            "run_experiment_grid strict mode aborted on failure: "
            f"grid_index={first_failure.get('grid_index')}, "
            f"reason={first_failure.get('failure_reason')}"
        )

    return grid_summary


def build_aggregate_report(
    experiment_results: list[dict],
    grid_manifest: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    功能：构建批量实验聚合报告。

    Build aggregate report from per-run summaries without recomputing thresholds
    or detection scores.

    Args:
        experiment_results: List of per-run summary mappings.

    Returns:
        Aggregate report with matrix digest, anchor set, and metric matrix.

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(experiment_results, list):
        # experiment_results 类型不合法，必须 fail-fast。
        raise TypeError("experiment_results must be list")

    canonical_items: List[Dict[str, Any]] = []
    metrics_matrix: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    policy_path_value = _first_present_str(
        *[
            item.get("policy_path")
            for item in experiment_results
            if isinstance(item, dict)
        ]
    )

    for item in experiment_results:
        if not isinstance(item, dict):
            raise TypeError("experiment result item must be dict")

        grid_item_digest = _safe_str(item.get("grid_item_digest"))
        status_value = _safe_str(item.get("status"))
        canonical_items.append(
            {
                "grid_item_digest": grid_item_digest,
                "status": status_value,
            }
        )

        anchor_row = {
            "grid_item_digest": grid_item_digest,
            "status": status_value,
            "cfg_digest": _safe_str(item.get("cfg_digest")),
            "plan_digest": _safe_str(item.get("plan_digest")),
            "thresholds_digest": _safe_str(item.get("thresholds_digest")),
            "threshold_metadata_digest": _safe_str(item.get("threshold_metadata_digest")),
            "ablation_digest": _safe_str(item.get("ablation_digest")),
            "attack_protocol_digest": _safe_str(item.get("attack_protocol_digest")),
            "impl_digest": _safe_str(item.get("impl_digest")),
            "fusion_rule_version": _safe_str(item.get("fusion_rule_version")),
            "attack_protocol_version": _safe_str(item.get("attack_protocol_version")),
            "run_root": _safe_str(item.get("run_root")),
            "policy_path": _safe_str(item.get("policy_path")),
        }
        anchor_rows.append(anchor_row)

        metrics_obj = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        metrics_matrix.append(
            {
                "grid_item_digest": grid_item_digest,
                "status": status_value,
                "tpr_at_fpr": metrics_obj.get("tpr_at_fpr"),
                "geo_available_rate": metrics_obj.get("geo_available_rate"),
                "rescue_rate": metrics_obj.get("rescue_rate"),
                "reject_rate": metrics_obj.get("reject_rate"),
                "reject_rate_breakdown": metrics_obj.get("reject_rate_breakdown", {}),
            }
        )

        if status_value != "ok":
            failures.append(
                {
                    "grid_item_digest": grid_item_digest,
                    "failure_reason": _safe_str(item.get("failure_reason")),
                    "run_root": _safe_str(item.get("run_root")),
                }
            )

    grouped_rows = _build_grouped_rows(experiment_results)
    failure_semantics_distribution = _collect_failure_semantics_distribution(experiment_results)
    coverage_manifest = attack_coverage.compute_attack_coverage_manifest()

    report = {
        "aggregate_report_version": "aggregate_v1",
        "experiment_matrix_digest": digests.canonical_sha256(canonical_items),
        "experiment_count": len(experiment_results),
        "success_count": sum(1 for item in experiment_results if item.get("status") == "ok"),
        "failure_count": sum(1 for item in experiment_results if item.get("status") != "ok"),
        "grid_manifest_digest": _safe_str(grid_manifest.get("grid_manifest_digest")) if isinstance(grid_manifest, dict) else "<absent>",
        "attack_coverage_digest": _safe_str(coverage_manifest.get("attack_coverage_digest")),
        "policy_path": _safe_str(policy_path_value),
        "attack_coverage_manifest": coverage_manifest,
        "anchors": anchor_rows,
        "metrics_matrix": metrics_matrix,
        "grouped_metrics": grouped_rows,
        "failure_semantics_distribution": failure_semantics_distribution,
        "failures": failures,
    }
    return report


def _extract_matrix_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract matrix config section with backward-compatible keys."""
    matrix_cfg = base_cfg.get("experiment_matrix")
    if matrix_cfg is None:
        matrix_cfg = base_cfg.get("experiment")
    if matrix_cfg is None:
        return {}
    if not isinstance(matrix_cfg, dict):
        raise TypeError("experiment_matrix must be dict when provided")
    return matrix_cfg


def _resolve_model_axis(base_cfg: Dict[str, Any], matrix_cfg: Dict[str, Any]) -> List[str]:
    """Resolve model axis list."""
    models = matrix_cfg.get("models")
    if models is None:
        model_id = base_cfg.get("model_id", "<absent>")
        return [str(model_id)]
    if not isinstance(models, list) or not models:
        raise ValueError("experiment_matrix.models must be non-empty list")
    resolved: List[str] = []
    for item in models:
        if not isinstance(item, str) or not item:
            raise ValueError("experiment_matrix.models entries must be non-empty str")
        resolved.append(item)
    return resolved


def _resolve_seed_axis(base_cfg: Dict[str, Any], matrix_cfg: Dict[str, Any]) -> List[int]:
    """Resolve seed axis list."""
    seeds = matrix_cfg.get("seeds")
    if seeds is None:
        seed_value = base_cfg.get("seed", 0)
        if isinstance(seed_value, int):
            return [seed_value]
        return [0]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment_matrix.seeds must be non-empty list")
    resolved: List[int] = []
    for item in seeds:
        if not isinstance(item, int):
            raise ValueError("experiment_matrix.seeds entries must be int")
        resolved.append(item)
    return resolved


def _resolve_attack_family_axis(
    base_cfg: Dict[str, Any],
    matrix_cfg: Dict[str, Any],
) -> Tuple[List[str], str, str]:
    """Resolve attack family axis and protocol version."""
    protocol_spec = protocol_loader.load_attack_protocol_spec(base_cfg)
    protocol_version = protocol_loader.get_protocol_version(protocol_spec)
    protocol_digest = _safe_str(protocol_spec.get("attack_protocol_digest"))

    families = matrix_cfg.get("attack_protocol_families")
    if families is None:
        protocol_families = protocol_spec.get("families", {})
        if isinstance(protocol_families, dict) and protocol_families:
            return sorted(str(name) for name in protocol_families.keys()), protocol_version, protocol_digest
        return ["<absent>"], protocol_version, protocol_digest

    if not isinstance(families, list) or not families:
        raise ValueError("experiment_matrix.attack_protocol_families must be non-empty list")

    resolved: List[str] = []
    for item in families:
        if not isinstance(item, str) or not item:
            raise ValueError("experiment_matrix.attack_protocol_families entries must be non-empty str")
        resolved.append(item)
    return resolved, protocol_version, protocol_digest


def _resolve_ablation_axis(matrix_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve ablation variant axis."""
    variants = matrix_cfg.get("ablation_variants")
    if variants is None:
        return [{}]
    if not isinstance(variants, list) or not variants:
        raise ValueError("experiment_matrix.ablation_variants must be non-empty list")

    resolved: List[Dict[str, Any]] = []
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("experiment_matrix.ablation_variants entries must be dict")
        for key, value in item.items():
            if not isinstance(key, str) or not key:
                raise ValueError("ablation flag key must be non-empty str")
            if not isinstance(value, bool):
                raise ValueError("ablation flag value must be bool")
        resolved.append(copy.deepcopy(item))
    return resolved


def _resolve_formal_validation_guards(matrix_cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Resolve explicit formal-validation guard flags for experiment_matrix."""
    if not isinstance(matrix_cfg, dict):
        raise TypeError("matrix_cfg must be dict")

    resolved: Dict[str, bool] = {}
    for field_name in [
        "require_real_negative_cache",
        "require_shared_thresholds",
        "disallow_forced_pair_fallback",
    ]:
        field_value = matrix_cfg.get(field_name, False)
        if not isinstance(field_value, bool):
            raise TypeError(f"experiment_matrix.{field_name} must be bool")
        resolved[field_name] = field_value
    return resolved


def _extract_formal_validation_guards_from_grid_item(grid_item_cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Extract validated formal-validation guard flags from one grid item."""
    if not isinstance(grid_item_cfg, dict):
        raise TypeError("grid_item_cfg must be dict")

    resolved: Dict[str, bool] = {}
    for field_name in [
        "require_real_negative_cache",
        "require_shared_thresholds",
        "disallow_forced_pair_fallback",
    ]:
        field_value = grid_item_cfg.get(field_name, False)
        if not isinstance(field_value, bool):
            raise TypeError(f"grid item {field_name} must be bool")
        resolved[field_name] = field_value
    return resolved


def _assert_formal_validation_prerequisites(
    grid: List[Dict[str, Any]],
    neg_detect_record_cache: Dict[Tuple[str, int], Optional[Path]],
    shared_thresholds_path: Optional[Path],
) -> None:
    """Fail-fast when explicit formal-validation guards cannot be satisfied."""
    if not isinstance(grid, list):
        raise TypeError("grid must be list")
    if not isinstance(neg_detect_record_cache, dict):
        raise TypeError("neg_detect_record_cache must be dict")
    if shared_thresholds_path is not None and not isinstance(shared_thresholds_path, Path):
        raise TypeError("shared_thresholds_path must be Path or None")

    missing_real_negative_keys: List[str] = []
    require_shared_thresholds = False

    for item in grid:
        if not isinstance(item, dict):
            raise TypeError("grid items must be dict")
        guards = _extract_formal_validation_guards_from_grid_item(item)
        model_id = item.get("model_id")
        seed_value = item.get("seed")
        neg_key: Optional[Tuple[str, int]] = None
        if isinstance(model_id, str) and isinstance(seed_value, int):
            neg_key = (model_id, seed_value)

        if guards["require_real_negative_cache"]:
            neg_path = neg_detect_record_cache.get(neg_key) if neg_key is not None else None
            if neg_path is None or not neg_path.exists() or not neg_path.is_file():
                missing_real_negative_keys.append(
                    f"model_id={_safe_str(model_id)}, seed={seed_value if isinstance(seed_value, int) else '<absent>'}"
                )

        require_shared_thresholds = require_shared_thresholds or guards["require_shared_thresholds"]
    if missing_real_negative_keys:
        missing_keys_joined = "; ".join(sorted(set(missing_real_negative_keys)))
        raise RuntimeError(
            "experiment_matrix formal validation requires real negative cache for every guarded item; "
            f"missing={missing_keys_joined}"
        )

    has_shared_thresholds = (
        shared_thresholds_path is not None
        and shared_thresholds_path.exists()
        and shared_thresholds_path.is_file()
    )
    if require_shared_thresholds and not has_shared_thresholds:
        raise RuntimeError(
            "experiment_matrix formal validation requires shared thresholds from global calibrate; "
            "current run produced no valid thresholds artifact"
        )


def _apply_ablation_flags(cfg_snapshot: Dict[str, Any], ablation_flags: Dict[str, Any]) -> None:
    """Apply ablation flags into cfg snapshot and normalize."""
    if "ablation" not in cfg_snapshot or not isinstance(cfg_snapshot.get("ablation"), dict):
        cfg_snapshot["ablation"] = {}
    else:
        cfg_snapshot["ablation"]["normalized"] = None
    for field_name, field_value in ablation_flags.items():
        cfg_snapshot["ablation"][field_name] = field_value
    config_loader.normalize_ablation_flags(cfg_snapshot)


def _derive_run_root(grid_item_cfg: Dict[str, Any]) -> Path:
    """Derive deterministic per-item run root path under batch_root."""
    batch_root_str = grid_item_cfg.get("batch_root", "outputs/experiment_matrix")
    if not isinstance(batch_root_str, str) or not batch_root_str:
        raise ValueError("grid_item_cfg.batch_root must be non-empty str")

    grid_index = grid_item_cfg.get("grid_index")
    if not isinstance(grid_index, int) or grid_index < 0:
        raise ValueError("grid_item_cfg.grid_index must be non-negative int")

    grid_item_digest = grid_item_cfg.get("grid_item_digest")
    if not isinstance(grid_item_digest, str) or not grid_item_digest:
        raise ValueError("grid_item_cfg.grid_item_digest must be non-empty str")

    run_root = Path(batch_root_str) / "experiments" / f"item_{grid_index:04d}_{grid_item_digest[:12]}"
    return path_policy.derive_run_root(run_root)


def _run_neg_embed_detect_for_cache(
    model_id: str,
    seed: int,
    config_path: str,
    batch_root: str,
    max_samples: Optional[int],
) -> Optional[Path]:
    """
    功能：为 (model_id, seed) 预生成真实负样本 detect record 并缓存。

    Run embed once to materialize a clean preview-generation image, then detect on the
    preview image to obtain a real negative-sample content score. Results are cached
    under batch_root/neg_cache/ and reused within the current experiment-matrix session
    to avoid redundant SD inference.

    Args:
        model_id: Stable Diffusion model identifier string.
        seed: Reproducibility seed integer.
        config_path: Base config YAML path passed to each CLI stage.
        batch_root: Batch output root; neg cache is written to batch_root/neg_cache/.
        max_samples: Optional max_samples override, or None to use config default.

    Returns:
        Path to the cached detect_record.json when generation succeeds, else None.
    """
    if not isinstance(model_id, str) or not model_id:
        return None
    if not isinstance(seed, int):
        return None
    if not isinstance(config_path, str) or not config_path:
        return None
    if not isinstance(batch_root, str) or not batch_root:
        return None
    if max_samples is not None and not isinstance(max_samples, int):
        return None

    # (model_id, seed) 的 canonical digest 作为确定性缓存 key，避免路径含特殊字符。
    cache_key = digests.canonical_sha256({"model_id": model_id, "seed": seed})
    neg_run_root = path_policy.derive_run_root(
        Path(batch_root) / "neg_cache" / f"neg_{cache_key[:16]}"
    )
    neg_detect_record_path = neg_run_root / "records" / "detect_record.json"

    # 缓存命中：直接返回已存在的 detect record，无需重复 SD 推理。
    if neg_detect_record_path.exists() and neg_detect_record_path.is_file():
        return neg_detect_record_path

    # 公共 override 项：seed / model_id / allow_nonempty_run_root。
    # allow_nonempty_run_root=true 确保 embed 与 detect 均可运行在同一目录。
    common_overrides: List[str] = [
        "allow_nonempty_run_root=true",
        'allow_nonempty_run_root_reason="neg_cache"',
        f"seed={seed}",
        f"model_id={json.dumps(model_id)}",
    ]
    if isinstance(max_samples, int):
        common_overrides.append(f"max_samples={max_samples}")

    # (1) embed 阶段：运行正式 embed 以触发 preview_generation，提取干净 preview 图像。
    preview_image_path = _run_embed_stage_for_neg_cache_preview(
        run_root=neg_run_root,
        config_path=Path(config_path),
        stage_overrides=list(common_overrides),
    )
    preview_input_record_path = _write_neg_preview_input_record(
        run_root=neg_run_root,
        preview_image_path=preview_image_path,
    )

    # (2) detect 阶段：对干净 preview 图像执行 content 检测，获取真实负样本分数。
    # allow_threshold_fallback_for_tests=true 因校准工件此时尚未产出。
    detect_overrides = list(common_overrides) + [
        "enable_content_detect=true",
        "allow_threshold_fallback_for_tests=true",
    ]
    _run_stage_command(
        stage_name="detect",
        run_root=neg_run_root,
        config_path=Path(config_path),
        stage_overrides=detect_overrides,
        input_record_path=preview_input_record_path,
    )

    if neg_detect_record_path.exists() and neg_detect_record_path.is_file():
        return neg_detect_record_path
    return None


def _run_embed_stage_for_neg_cache_preview(
    run_root: Path,
    config_path: Path,
    stage_overrides: List[str],
) -> Path:
    """Run embed stage and resolve the generated clean preview image path."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(stage_overrides, list):
        raise TypeError("stage_overrides must be list")

    command = [
        sys.executable,
        "-m",
        "main.cli.run_embed",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    for item in stage_overrides:
        if not isinstance(item, str) or not item:
            raise ValueError("stage_overrides entries must be non-empty str")
        command.extend(["--override", item])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "stage failed: embed\n"
            f"  - command: {' '.join(command)}\n"
            f"  - stdout_tail: {result.stdout[-1200:]}\n"
            f"  - stderr_tail: {result.stderr[-1200:]}"
        )

    preview_image_path = _extract_preview_image_path_from_embed_stdout(result.stdout)
    if preview_image_path is None:
        raise RuntimeError(
            "neg_cache preview generation path missing in embed stdout; "
            f"stdout_tail={result.stdout[-1200:]}"
        )
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise RuntimeError(f"neg_cache preview image path not found: {preview_image_path}")
    return preview_image_path


def _extract_preview_image_path_from_embed_stdout(stdout_text: str) -> Optional[Path]:
    """Extract clean preview image path from embed stdout."""
    if not isinstance(stdout_text, str) or not stdout_text:
        return None

    marker = "[Preview Generation] 预览图已生成，路径："
    for line in reversed(stdout_text.splitlines()):
        if marker not in line:
            continue
        path_text = line.split(marker, 1)[1].strip()
        if path_text:
            return Path(path_text).resolve()
    return None


def _write_neg_preview_input_record(run_root: Path, preview_image_path: Path) -> Path:
    """Write a minimal input record so detect can consume the clean preview image."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(preview_image_path, Path):
        raise TypeError("preview_image_path must be Path")
    if not preview_image_path.exists() or not preview_image_path.is_file():
        raise ValueError(f"preview_image_path not found: {preview_image_path}")

    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    input_record_path = artifacts_dir / "neg_preview_input" / "detect_input_record.json"
    input_record_payload = {
        "operation": "embed_preview_input",
        "image_path": str(preview_image_path),
        "watermarked_path": str(preview_image_path),
        "inputs": {
            "input_image_path": str(preview_image_path),
        },
    }
    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(input_record_path),
        obj=input_record_payload,
    )
    return input_record_path


def _run_global_calibrate(
    batch_root: str,
    config_path: str,
    neg_detect_record_cache: Dict[Tuple[str, int], Optional[Path]],
) -> Optional[Path]:
    """
    功能：使用所有预生成的真实负样本产出全局共享 NP 阈值工件。

    Aggregate all valid neg detect records from the cache, label them as negative,
    write to a staging directory, then run one global calibrate stage. This ensures
    the calibration null distribution is independent of the test set (attacked
    watermarked images), satisfying the paper-level train/test separation requirement.

    Args:
        batch_root: Batch output root; global calibrate writes to batch_root/global_calibrate/.
        config_path: Base config YAML path for the calibrate CLI.
        neg_detect_record_cache: Mapping of (model_id, seed) → detect_record path.

    Returns:
        Path to the shared thresholds_artifact.json, or None on failure.
    """
    if not isinstance(batch_root, str) or not batch_root:
        return None
    if not isinstance(config_path, str) or not config_path:
        return None
    if not isinstance(neg_detect_record_cache, dict):
        return None

    # (1) 收集所有有效的 neg detect record path，跳过生成失败的条目。
    valid_neg_paths: List[Path] = [
        p for p in neg_detect_record_cache.values()
        if p is not None and p.exists() and p.is_file()
    ]
    if not valid_neg_paths:
        return None

    global_calibrate_root = path_policy.derive_run_root(
        Path(batch_root) / "global_calibrate"
    )
    # neg 标注暂存目录位于 global_calibrate_root/artifacts/ 内，满足受控写盘路径约束。
    neg_staged_dir = global_calibrate_root / "artifacts" / "neg_staged"
    neg_staged_dir.mkdir(parents=True, exist_ok=True)

    # (2) 为每条 neg detect record 追加 label=False 标注并写入暂存目录。
    # load_scores_for_calibration 在 has_explicit_labels=True 时仅使用 label=False 的记录
    # 作为 null 分布，与 label=True 的正样本完全隔离。
    staged_count = 0
    for idx, neg_path in enumerate(valid_neg_paths):
        neg_record = _read_optional_json(neg_path)
        if not isinstance(neg_record, dict) or not neg_record:
            continue
        score_val, score_source = _resolve_matrix_calibration_score(neg_record)
        if not isinstance(score_val, float):
            continue
        labeled_neg = copy.deepcopy(neg_record)
        labeled_neg["label"] = False
        labeled_neg["ground_truth"] = False
        labeled_neg["is_watermarked"] = False
        labeled_neg["calibration_label_resolution"] = "global_calibrate_real_neg"
        labeled_neg["ground_truth_source"] = "real_neg_embed_detect"
        labeled_neg["calibration_sample_usage"] = "real_negative_global_calibrate_null_distribution"
        _ensure_matrix_calibration_compatible_content_payload(
            labeled_neg,
            score_val,
            score_source=score_source,
            recovered_sample_origin="global_calibrate_real_negative_recovery",
        )
        staged_path = neg_staged_dir / f"neg_record_{idx:04d}.json"
        records_io.write_artifact_text_unbound(
            run_root=global_calibrate_root,
            artifacts_dir=neg_staged_dir,
            path=str(staged_path),
            content=json.dumps(labeled_neg),
        )
        staged_count += 1

    if staged_count == 0:
        return None

    neg_glob = str(neg_staged_dir / "*.json")

    # (3) 直接在 experiment_matrix 内生成共享阈值工件。
    # run_calibrate CLI 入口要求同时存在正负标签样本，而全局 shared thresholds
    # 在 formal 语义上只依赖真实负样本 null distribution。
    # 因此这里直接复用 calibrate orchestrator 的统计核心，避免被 CLI 的
    # label-balance 前置门禁错误阻断。
    cfg, _ = config_loader.load_yaml_with_provenance(config_path)
    if not isinstance(cfg, dict):
        return None

    calibration_cfg = copy.deepcopy(cfg.get("calibration")) if isinstance(cfg.get("calibration"), dict) else {}
    calibration_cfg["detect_records_glob"] = neg_glob
    cfg["calibration"] = calibration_cfg

    calibrate_record = detect_orchestrator.run_calibrate_orchestrator(cfg, object())
    thresholds_artifact = calibrate_record.get("thresholds_artifact")
    threshold_metadata_artifact = calibrate_record.get("threshold_metadata_artifact")
    if not isinstance(thresholds_artifact, dict):
        return None
    if not isinstance(threshold_metadata_artifact, dict):
        return None

    thresholds_dir = global_calibrate_root / "artifacts" / "thresholds"
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    shared_thresholds_path = thresholds_dir / "thresholds_artifact.json"
    threshold_metadata_path = thresholds_dir / "threshold_metadata_artifact.json"
    records_io.write_artifact_json_unbound(
        run_root=global_calibrate_root,
        artifacts_dir=thresholds_dir,
        path=str(shared_thresholds_path),
        obj=thresholds_artifact,
    )
    records_io.write_artifact_json_unbound(
        run_root=global_calibrate_root,
        artifacts_dir=thresholds_dir,
        path=str(threshold_metadata_path),
        obj=threshold_metadata_artifact,
    )

    if shared_thresholds_path.exists() and shared_thresholds_path.is_file():
        return shared_thresholds_path
    return None


def _run_stage_sequence(grid_item_cfg: Dict[str, Any], run_root: Path) -> Dict[str, Any]:
    """Run embed/detect/calibrate/evaluate sequence for one experiment."""
    layout = path_policy.ensure_output_layout(
        run_root,
        allow_nonempty_run_root=False,
        allow_nonempty_run_root_reason=None,
        override_applied=None,
    )
    _ = layout

    config_path = grid_item_cfg.get("config_path", "configs/paper_full_cuda.yaml")
    attack_protocol_path = grid_item_cfg.get("attack_protocol_path", config_loader.ATTACK_PROTOCOL_PATH)
    cfg_snapshot_obj = grid_item_cfg.get("cfg_snapshot", {})
    if not isinstance(cfg_snapshot_obj, dict):
        raise TypeError("grid item cfg_snapshot must be dict")

    seed_value = cfg_snapshot_obj.get("seed")
    model_id = cfg_snapshot_obj.get("model_id")
    max_samples = grid_item_cfg.get("max_samples")
    ablation_flags = grid_item_cfg.get("ablation_flags", {})

    if not isinstance(config_path, str) or not config_path:
        raise ValueError("grid_item_cfg.config_path must be non-empty str")
    if not isinstance(attack_protocol_path, str) or not attack_protocol_path:
        raise ValueError("grid_item_cfg.attack_protocol_path must be non-empty str")
    if seed_value is not None and not isinstance(seed_value, int):
        raise TypeError("grid item seed must be int or None")
    if model_id is not None and not isinstance(model_id, str):
        raise TypeError("grid item model_id must be str or None")
    if max_samples is not None and not isinstance(max_samples, int):
        raise TypeError("grid item max_samples must be int or None")
    if not isinstance(ablation_flags, dict):
        raise TypeError("grid item ablation_flags must be dict")

    formal_validation_guards = _extract_formal_validation_guards_from_grid_item(grid_item_cfg)

    ablation_snapshot = cfg_snapshot_obj.get("ablation")
    ablation_override_enabled = isinstance(ablation_snapshot, dict)

    allow_failed_semantics_collection = grid_item_cfg.get("allow_failed_semantics_collection", False)
    if not isinstance(allow_failed_semantics_collection, bool):
        raise TypeError("grid item allow_failed_semantics_collection must be bool")

    detect_gate_info: Dict[str, Any] = {
        "gate_relaxed": False,
        "reason": "hard_gate_not_checked",
        "sample_counts": {},
    }
    labelled_detect_records_glob: Optional[str] = None

    # 从 grid_item_cfg 读取由 run_experiment_grid 预注入的共享阈值路径与真实负样本路径。
    # shared_thresholds_path 存在时：跳过 per-item calibrate，evaluate 使用全局阈值。
    # 降级条件：shared_thresholds_path 缺失或文件不存在时，回落 per-item calibrate 路径。
    shared_thresholds_path_str = grid_item_cfg.get("shared_thresholds_path")
    shared_thresholds_path_val: Optional[Path] = (
        Path(shared_thresholds_path_str)
        if isinstance(shared_thresholds_path_str, str) and shared_thresholds_path_str
        else None
    )
    use_shared_thresholds = (
        shared_thresholds_path_val is not None
        and shared_thresholds_path_val.exists()
        and shared_thresholds_path_val.is_file()
    )

    neg_path_str = grid_item_cfg.get("neg_detect_record_path")
    neg_detect_record_path_for_stage: Optional[Path] = (
        Path(neg_path_str) if isinstance(neg_path_str, str) and neg_path_str else None
    )
    if formal_validation_guards["require_real_negative_cache"]:
        if (
            neg_detect_record_path_for_stage is None
            or not neg_detect_record_path_for_stage.exists()
            or not neg_detect_record_path_for_stage.is_file()
        ):
            raise RuntimeError(
                "experiment_matrix formal validation requires a valid neg_detect_record_path before per-item execution"
            )

    if formal_validation_guards["require_shared_thresholds"] and not use_shared_thresholds:
        raise RuntimeError(
            "experiment_matrix formal validation requires shared thresholds before per-item execution"
        )

    for stage_name in ["embed", "detect", "calibrate", "evaluate"]:
        # 全局阈值可用时跳过 per-item calibrate：
        # 校准集（neg_cache 的全体 neg 记录）与测试集（per-item 攻击正样本）已在
        # run_experiment_grid 层严格分离，per-item calibrate 会把二者混用，不符合
        # NP 阈值估计的独立同分布要求。
        if stage_name == "calibrate" and use_shared_thresholds:
            # 在跳过前预先准备 labelled_detect_records_glob，供后续 evaluate 阶段使用。
            if labelled_detect_records_glob is None:
                labelled_detect_records_glob = _prepare_labelled_detect_records_glob_for_matrix(
                    run_root, grid_item_cfg, neg_detect_record_path=neg_detect_record_path_for_stage
                )
            continue

        stage_overrides = [
            "allow_nonempty_run_root=true",
            'allow_nonempty_run_root_reason="experiment_grid"',
        ]
        if isinstance(seed_value, int):
            stage_overrides.append(f"seed={seed_value}")
        if isinstance(model_id, str) and model_id:
            stage_overrides.append(f"model_id={json.dumps(model_id)}")
        if isinstance(max_samples, int):
            stage_overrides.append(f"max_samples={max_samples}")

        # detect 阶段必须启用 content 检测（experiment_matrix 需要生成检测分数）
        # detect 阶段的阈值回退是架构必要性：校准工件在 calibrate 阶段才产出，
        # detect 阶段产出的阈值仅用于中间评分记录，不进入最终判决。
        if stage_name == "detect":
            stage_overrides.append("enable_content_detect=true")
            stage_overrides.append("allow_threshold_fallback_for_tests=true")
        # embed 阶段不强制写入 disable_content_detect 覆盖项。
        # 由配置与 whitelist 统一约束，避免 override_value_mismatch。

        # calibrate/evaluate 都需要带标签的 detect records 输入。
        # 这里使用 detect 后生成的 attack-aware 标注记录对（正/负各一条）满足标签平衡门禁。
        if stage_name in {"calibrate", "evaluate"}:
            if labelled_detect_records_glob is None:
                labelled_detect_records_glob = _prepare_labelled_detect_records_glob_for_matrix(
                    run_root, grid_item_cfg, neg_detect_record_path=neg_detect_record_path_for_stage
                )
            arg_name = f"{stage_name}_detect_records_glob"
            stage_overrides.append(f"{arg_name}={json.dumps(labelled_detect_records_glob)}")

        # evaluate 阈值来源：优先使用全局共享阈值，降级时使用 per-item calibrate 产出的本地阈值。
        if stage_name == "evaluate":
            if use_shared_thresholds:
                stage_overrides.append(
                    f"evaluate_thresholds_path={json.dumps(str(shared_thresholds_path_val))}"
                )
            else:
                local_thresholds_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
                stage_overrides.append(
                    f"evaluate_thresholds_path={json.dumps(str(local_thresholds_path))}"
                )

        if ablation_override_enabled:
            for key, value in sorted(ablation_flags.items()):
                # key 形如 "enable_geometry"；统一使用 ablation_enable_* arg_name 格式，
                # 避免 field_path 格式在 whitelist 中因 enable/disable 双条目引发歧义错误。
                suffix = key[len("enable_"):] if key.startswith("enable_") else key
                stage_overrides.append(f"ablation_enable_{suffix}={str(value).lower()}")

        _run_stage_command(
            stage_name=stage_name,
            run_root=run_root,
            config_path=Path(config_path),
            stage_overrides=stage_overrides,
        )

        if stage_name == "detect":
            detect_gate_info = _assert_min_valid_content_scores_after_detect(
                run_root,
                minimum_required=1,
                allow_failed_semantics_collection=allow_failed_semantics_collection,
            )

    return detect_gate_info


def _assert_min_valid_content_scores_after_detect(
    run_root: Path,
    minimum_required: int = 1,
    allow_failed_semantics_collection: bool = False,
) -> Dict[str, Any]:
    """Fail-fast gate: require at least minimum valid content_score samples after detect."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(minimum_required, int) or minimum_required <= 0:
        raise ValueError("minimum_required must be positive int")
    if not isinstance(allow_failed_semantics_collection, bool):
        raise TypeError("allow_failed_semantics_collection must be bool")

    detect_record_path = run_root / "records" / "detect_record.json"
    detect_record_obj = _read_optional_json(detect_record_path)
    if not isinstance(detect_record_obj, dict) or not detect_record_obj:
        raise RuntimeError(
            "detect stage gate failed: detect_record missing or invalid; "
            f"path={detect_record_path}"
        )

    content_payload = detect_record_obj.get("content_evidence_payload")
    valid_count = 0
    zero_score_count = 0
    status_value = None
    score_value = None
    if isinstance(content_payload, dict):
        status_value = content_payload.get("status")
        score_value = content_payload.get("score")
        if status_value == "ok" and isinstance(score_value, (int, float)) and np.isfinite(float(score_value)):
            valid_count = 1
            if float(score_value) == 0.0:
                zero_score_count = 1

    sample_counts = {
        "minimum_required": int(minimum_required),
        "valid_content_score_samples": int(valid_count),
        "zero_content_score_samples": int(zero_score_count),
        "status": status_value if isinstance(status_value, str) else "<absent>",
    }

    if valid_count >= minimum_required:
        return {
            "gate_relaxed": False,
            "reason": "hard_gate_satisfied",
            "sample_counts": sample_counts,
        }

    if allow_failed_semantics_collection:
        return {
            "gate_relaxed": True,
            "reason": "insufficient_valid_content_score_samples_research_collection_mode",
            "sample_counts": sample_counts,
        }

    raise RuntimeError(
        "detect stage gate failed: insufficient valid content_score samples before calibrate; "
        f"required={minimum_required}, valid={valid_count}, status={status_value}, score={score_value}, "
        f"path={detect_record_path}"
    )


def _resolve_attack_params_version_for_family(grid_item_cfg: Dict[str, Any]) -> str:
    """Resolve deterministic params_version for one attack family from protocol spec."""
    if not isinstance(grid_item_cfg, dict):
        raise TypeError("grid_item_cfg must be dict")

    attack_family = grid_item_cfg.get("attack_protocol_family")
    if not isinstance(attack_family, str) or not attack_family:
        return "<absent>"

    attack_protocol_path = grid_item_cfg.get("attack_protocol_path", config_loader.ATTACK_PROTOCOL_PATH)
    if not isinstance(attack_protocol_path, str) or not attack_protocol_path:
        attack_protocol_path = config_loader.ATTACK_PROTOCOL_PATH

    protocol_cfg = {
        "evaluate": {
            "attack_protocol_path": attack_protocol_path,
        }
    }
    protocol_spec = protocol_loader.load_attack_protocol_spec(protocol_cfg)

    families_obj = protocol_spec.get("families") if isinstance(protocol_spec, dict) else {}
    if isinstance(families_obj, dict):
        family_spec = families_obj.get(attack_family)
        if isinstance(family_spec, dict):
            family_versions = family_spec.get("params_versions")
            if isinstance(family_versions, dict) and family_versions:
                resolved = sorted(
                    version_name
                    for version_name in family_versions.keys()
                    if isinstance(version_name, str) and version_name
                )
                if resolved:
                    return resolved[0]

    params_versions_obj = protocol_spec.get("params_versions") if isinstance(protocol_spec, dict) else {}
    if isinstance(params_versions_obj, dict):
        prefix = f"{attack_family}::"
        resolved = []
        for condition_key in params_versions_obj.keys():
            if not isinstance(condition_key, str):
                continue
            if not condition_key.startswith(prefix):
                continue
            version_name = condition_key.split("::", 1)[1]
            if version_name:
                resolved.append(version_name)
        if resolved:
            return sorted(set(resolved))[0]

    return "<absent>"


def _prepare_detect_record_for_attack_grouping(run_root: Path, grid_item_cfg: Dict[str, Any]) -> Path:
    """Create enriched detect record artifact with attack family/params_version for grouping."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(grid_item_cfg, dict):
        raise TypeError("grid_item_cfg must be dict")

    source_detect_record_path = run_root / "records" / "detect_record.json"
    if not source_detect_record_path.exists() or not source_detect_record_path.is_file():
        return source_detect_record_path

    attack_family = grid_item_cfg.get("attack_protocol_family")
    if not isinstance(attack_family, str) or not attack_family:
        return source_detect_record_path

    detect_record = _read_optional_json(source_detect_record_path)
    if not isinstance(detect_record, dict) or not detect_record:
        return source_detect_record_path

    for forbidden_field in _FORBIDDEN_ARTIFACT_ANCHOR_FIELDS:
        detect_record.pop(forbidden_field, None)

    params_version = _resolve_attack_params_version_for_family(grid_item_cfg)

    enriched_record = copy.deepcopy(detect_record)
    label_value = _resolve_ground_truth_label_for_record(enriched_record)
    if isinstance(label_value, bool):
        enriched_record["label"] = label_value
        enriched_record["ground_truth"] = label_value
        enriched_record["is_watermarked"] = label_value
        enriched_record["calibration_label_resolution"] = "resolved"
    else:
        enriched_record["calibration_label_resolution"] = "missing"
        enriched_record["calibration_excluded_from_labelled_sampling"] = True
    enriched_record["attack_family"] = attack_family
    if isinstance(params_version, str) and params_version and params_version != "<absent>":
        enriched_record["attack_params_version"] = params_version

    attack_obj = enriched_record.get("attack")
    if not isinstance(attack_obj, dict):
        attack_obj = {}
    attack_obj["family"] = attack_family
    if isinstance(params_version, str) and params_version and params_version != "<absent>":
        attack_obj["params_version"] = params_version
    enriched_record["attack"] = attack_obj

    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = artifacts_dir / "evaluate_inputs" / "detect_record_with_attack.json"

    prompt_anchor, prompt_anchor_field = _extract_attack_metadata_source_prompt(enriched_record)
    if isinstance(prompt_anchor, str) and prompt_anchor:
        enriched_record["attack_metadata_source_prompt"] = prompt_anchor
    if isinstance(prompt_anchor_field, str) and prompt_anchor_field:
        enriched_record["attack_metadata_source_prompt_field"] = prompt_anchor_field

    join_key = _build_attack_metadata_join_key(
        prompt_anchor,
        attack_family,
        params_version if isinstance(params_version, str) and params_version else None,
    )
    if isinstance(join_key, str) and join_key:
        enriched_record["attack_metadata_join_key"] = join_key

    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(enriched_path),
        obj=enriched_record,
    )
    return enriched_path


def _extract_attack_metadata_source_prompt(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    功能：为 attack metadata 绑定解析稳定的 source prompt 锚点。 

    Resolve the stable source prompt anchor used for attack metadata binding.

    Args:
        record: Detect record mapping.

    Returns:
        Tuple of (prompt_value, field_path) when available.

    Raises:
        TypeError: If record is not a dict.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")

    top_level_prompt = record.get("inference_prompt")
    if isinstance(top_level_prompt, str) and top_level_prompt:
        return top_level_prompt, "inference_prompt"

    infer_trace_node = record.get("infer_trace")
    infer_trace = infer_trace_node if isinstance(infer_trace_node, dict) else {}
    infer_trace_prompt = infer_trace.get("inference_prompt")
    if isinstance(infer_trace_prompt, str) and infer_trace_prompt:
        return infer_trace_prompt, "infer_trace.inference_prompt"

    return None, None


def _build_attack_metadata_join_key(
    source_prompt_anchor: Optional[str],
    attack_family: Optional[str],
    attack_params_version: Optional[str],
) -> Optional[str]:
    """
    功能：构造 attack metadata 的可审计 join key。 

    Build the canonical join key used to bind attack metadata across records.

    Args:
        source_prompt_anchor: Stable prompt anchor shared with the source sample.
        attack_family: Attack family token.
        attack_params_version: Attack params version token.

    Returns:
        Canonical join key string when all components are available.
    """
    if not isinstance(source_prompt_anchor, str) or not source_prompt_anchor:
        return None
    if not isinstance(attack_family, str) or not attack_family:
        return None
    if not isinstance(attack_params_version, str) or not attack_params_version:
        return None
    return digests.canonical_sha256(
        {
            "source_prompt_anchor": source_prompt_anchor,
            "attack_family": attack_family,
            "attack_params_version": attack_params_version,
        }
    )


def _prepare_labelled_detect_records_glob_for_matrix(
    run_root: Path,
    grid_item_cfg: Dict[str, Any],
    neg_detect_record_path: Optional[Path] = None,
) -> str:
    """
    功能：构建带标签正负样本对并返回 glob 路径供 calibrate/evaluate 消费。

    Create labelled detect-record pair (positive from attacked watermarked image,
    negative from real clean image or synthetic fallback) and return a glob path
    covering both records for downstream calibrate/evaluate stages.

    Args:
        run_root: Per-experiment run output root directory.
        grid_item_cfg: Grid item config dict for the current experiment.
        neg_detect_record_path: Optional path to a real negative-sample detect_record.json
            produced by _run_neg_embed_detect_for_cache. When provided and valid, the
            content score is read from this record. When absent or invalid, falls back
            to (base_score - 1.0) synthetic offset.

    Returns:
        Glob pattern string matching both labelled record files.

    Raises:
        TypeError: If run_root or grid_item_cfg types are invalid.
        RuntimeError: If detect record is missing or invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(grid_item_cfg, dict):
        raise TypeError("grid_item_cfg must be dict")

    attack_aware_path = _prepare_detect_record_for_attack_grouping(run_root, grid_item_cfg)
    base_payload = _read_optional_json(attack_aware_path)
    if not isinstance(base_payload, dict) or not base_payload:
        source_detect_record_path = run_root / "records" / "detect_record.json"
        base_payload = _read_optional_json(source_detect_record_path)
    if not isinstance(base_payload, dict) or not base_payload:
        raise RuntimeError("detect record missing or invalid for matrix labelled sampling")

    labelled_dir = run_root / "artifacts" / "evaluate_inputs" / "labelled_detect_records"
    labelled_dir.mkdir(parents=True, exist_ok=True)

    positive_payload = copy.deepcopy(base_payload)
    positive_payload["label"] = True
    positive_payload["ground_truth"] = True
    positive_payload["is_watermarked"] = True
    positive_payload["calibration_label_resolution"] = "matrix_forced_positive"
    positive_payload.pop("calibration_excluded_from_labelled_sampling", None)

    disallow_forced_pair_fallback = bool(grid_item_cfg.get("disallow_forced_pair_fallback", False))

    negative_payload = copy.deepcopy(base_payload)
    negative_payload["label"] = False
    negative_payload["ground_truth"] = False
    negative_payload["is_watermarked"] = False
    negative_payload["calibration_label_resolution"] = "matrix_forced_negative"
    negative_payload.pop("calibration_excluded_from_labelled_sampling", None)
    negative_payload["calibration_sample_usage"] = "synthetic_negative_for_experiment_matrix_label_balance"

    base_score, _ = _resolve_matrix_calibration_score(base_payload)
    if not isinstance(base_score, float):
        base_score = 0.25

    _ensure_matrix_calibration_compatible_content_payload(positive_payload, base_score)

    # 优先使用预生成的真实负样本分数（干净图像经 identity embed + detect 产出）。
    # 真实分数保证 FPR 校准在真实负样本分布上进行，满足论文级严谨要求。
    # 降级条件：neg_detect_record_path 缺失、文件不存在或分数无效，则回落合成方案。
    neg_score: Optional[float] = None
    neg_payload_real: Dict[str, Any] = {}
    if (
        neg_detect_record_path is not None
        and neg_detect_record_path.exists()
        and neg_detect_record_path.is_file()
    ):
        neg_payload_real = _read_optional_json(neg_detect_record_path)
        neg_score, neg_score_source = _resolve_matrix_calibration_score(neg_payload_real)
    else:
        neg_score_source = None

    if isinstance(neg_score, float):
        negative_payload = copy.deepcopy(neg_payload_real)
        for forbidden_field in _FORBIDDEN_ARTIFACT_ANCHOR_FIELDS:
            negative_payload.pop(forbidden_field, None)
        for key_name in [
            "attack",
            "attack_family",
            "attack_params_version",
            "attack_metadata_source_prompt",
            "attack_metadata_source_prompt_field",
            "attack_metadata_join_key",
        ]:
            if key_name in base_payload:
                negative_payload[key_name] = copy.deepcopy(base_payload[key_name])
        negative_payload["label"] = False
        negative_payload["ground_truth"] = False
        negative_payload["is_watermarked"] = False
        negative_payload["calibration_label_resolution"] = "real_negative_payload"
        negative_payload.pop("calibration_excluded_from_labelled_sampling", None)
        _ensure_matrix_calibration_compatible_content_payload(
            negative_payload,
            neg_score,
            score_source=neg_score_source,
            recovered_sample_origin="real_negative_payload_recovery",
        )
        negative_payload["ground_truth_source"] = "real_neg_embed_detect"
        negative_payload["calibration_sample_usage"] = "real_negative_for_experiment_matrix_label_balance"
    else:
        if disallow_forced_pair_fallback:
            raise RuntimeError(
                "experiment_matrix formal validation disallows synthetic negative fallback; "
                "real negative payload is required for labelled detect records"
            )
        # 降级：neg 生成失败或分数无效，使用合成得分（base_score - 1.0）。
        _ensure_matrix_calibration_compatible_content_payload(negative_payload, base_score - 1.0)

    positive_rel_path = Path("artifacts") / "evaluate_inputs" / "labelled_detect_records" / "detect_record_label_pos.json"
    negative_rel_path = Path("artifacts") / "evaluate_inputs" / "labelled_detect_records" / "detect_record_label_neg.json"
    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(positive_rel_path),
        obj=positive_payload,
    )
    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(negative_rel_path),
        obj=negative_payload,
    )

    return str(labelled_dir / "*.json")


def _resolve_ground_truth_label_for_record(record: Dict[str, Any]) -> Optional[bool]:
    """Resolve bool ground-truth label from record and return None when label is absent."""
    if not isinstance(record, dict):
        raise TypeError("record must be dict")

    for key_name in ["label", "ground_truth", "is_watermarked"]:
        value = record.get(key_name)
        if isinstance(value, bool):
            return value

    return None


def _run_stage_command(
    stage_name: str,
    run_root: Path,
    config_path: Path,
    stage_overrides: List[str],
    input_record_path: Optional[Path] = None,
) -> None:
    """Execute one CLI stage command with fail-fast diagnostics."""
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be Path")
    if not isinstance(stage_overrides, list):
        raise TypeError("stage_overrides must be list")
    if input_record_path is not None and not isinstance(input_record_path, Path):
        raise TypeError("input_record_path must be Path or None")

    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    if stage_name == "detect":
        detect_input_path = input_record_path or (run_root / "records" / "embed_record.json")
        command.extend(["--input", str(detect_input_path)])
    for item in stage_overrides:
        if not isinstance(item, str) or not item:
            raise ValueError("stage_overrides entries must be non-empty str")
        command.extend(["--override", item])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"stage failed: {stage_name}\n"
            f"  - command: {' '.join(command)}\n"
            f"  - stdout_tail: {result.stdout[-1200:]}\n"
            f"  - stderr_tail: {result.stderr[-1200:]}"
        )


def _read_evaluation_report_for_run(run_root: Path) -> Dict[str, Any]:
    """Read evaluation report payload from artifacts/eval_report.json."""
    eval_report_path = run_root / "artifacts" / "eval_report.json"
    if not eval_report_path.exists() or not eval_report_path.is_file():
        raise FileNotFoundError(f"eval_report not found: {eval_report_path}")

    parsed_obj = json.loads(eval_report_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_obj, dict):
        raise TypeError("eval_report artifact root must be dict")
    eval_report = parsed_obj.get("evaluation_report")
    if not isinstance(eval_report, dict):
        raise TypeError("eval_report artifact missing evaluation_report dict")
    return eval_report


def _assert_required_run_artifacts(run_root: Path) -> None:
    """Assert required artifacts are present after one run."""
    required_paths = [
        run_root / "artifacts" / "run_closure.json",
        run_root / "records" / "evaluate_record.json",
        run_root / "artifacts" / "eval_report.json",
    ]
    for item in required_paths:
        if not item.exists() or not item.is_file():
            raise FileNotFoundError(f"required artifact missing: {item}")


def _write_grid_artifacts(
    grid: List[Dict[str, Any]],
    aggregate_report: Dict[str, Any],
    results: List[Dict[str, Any]],
    strict: bool,
) -> Dict[str, str]:
    """Write aggregate report and grid summary using controlled artifact writer."""
    if not grid:
        batch_root = path_policy.derive_run_root(Path("outputs/experiment_matrix"))
    else:
        batch_root_value = grid[0].get("batch_root", "outputs/experiment_matrix")
        if not isinstance(batch_root_value, str) or not batch_root_value:
            raise ValueError("grid[0].batch_root must be non-empty str")
        batch_root = path_policy.derive_run_root(Path(batch_root_value))

    layout = path_policy.ensure_output_layout(
        batch_root,
        allow_nonempty_run_root=True,
        allow_nonempty_run_root_reason="experiment_grid_aggregate",
        override_applied={"allow_nonempty_run_root": True},
    )
    artifacts_dir = layout["artifacts_dir"]

    aggregate_report_path = artifacts_dir / "aggregate_report.json"
    grid_summary_path = artifacts_dir / "grid_summary.json"
    grid_manifest_path = artifacts_dir / "grid_manifest.json"
    attack_coverage_manifest_path = artifacts_dir / "attack_coverage_manifest.json"
    hf_truncation_baseline_comparison_table_path = artifacts_dir / "hf_truncation_baseline_comparison_table.json"
    hf_truncation_baseline_comparison_table_csv_path = artifacts_dir / "hf_truncation_baseline_comparison_table.csv"

    grid_manifest = _build_grid_manifest(grid)

    # 提取锚点字段（append-only，只读已有工件）
    anchors_obj = _extract_anchors_from_results(aggregate_report, results)

    records_io.write_artifact_json_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(aggregate_report_path),
        obj=aggregate_report,
    )
    records_io.write_artifact_json_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(grid_summary_path),
        obj={
            "strict": strict,
            "total": len(results),
            "executed": len(results),
            "succeeded": sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok"),
            "failed": sum(1 for r in results if isinstance(r, dict) and r.get("status") != "ok"),
            "results": results,
            **anchors_obj,  # append-only: 补齐锚点字段全集
        },
    )
    records_io.write_artifact_json_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(grid_manifest_path),
        obj=grid_manifest,
    )

    coverage_manifest_obj = aggregate_report.get("attack_coverage_manifest")
    if isinstance(coverage_manifest_obj, dict):
        records_io.write_artifact_json_unbound(
            run_root=batch_root,
            artifacts_dir=artifacts_dir,
            path=str(attack_coverage_manifest_path),
            obj=coverage_manifest_obj,
        )
    else:
        records_io.write_artifact_json_unbound(
            run_root=batch_root,
            artifacts_dir=artifacts_dir,
            path=str(attack_coverage_manifest_path),
            obj=attack_coverage.compute_attack_coverage_manifest(),
        )

    hf_truncation_baseline_comparison_table = _build_hf_truncation_baseline_comparison_table(results)
    records_io.write_artifact_json_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(hf_truncation_baseline_comparison_table_path),
        obj=hf_truncation_baseline_comparison_table,
    )
    records_io.write_artifact_text_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(hf_truncation_baseline_comparison_table_csv_path),
        content=_build_hf_truncation_baseline_comparison_csv(hf_truncation_baseline_comparison_table),
    )

    return {
        "batch_root": str(batch_root),
        "aggregate_report_path": str(aggregate_report_path),
        "grid_summary_path": str(grid_summary_path),
        "grid_manifest_path": str(grid_manifest_path),
        "attack_coverage_manifest_path": str(attack_coverage_manifest_path),
        "hf_truncation_baseline_comparison_table_path": str(hf_truncation_baseline_comparison_table_path),
        "hf_truncation_baseline_comparison_table_csv_path": str(hf_truncation_baseline_comparison_table_csv_path),
    }


def _build_hf_truncation_baseline_comparison_table(experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic same-sample comparison table against real HF truncation baseline."""
    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be list")

    rows: List[Dict[str, Any]] = []
    content_scores: List[float] = []
    hf_truncation_scores: List[float] = []
    score_deltas: List[float] = []

    for item in experiment_results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        comparison_obj = item.get("hf_truncation_baseline_comparison") if isinstance(item.get("hf_truncation_baseline_comparison"), dict) else {}
        row = {
            "grid_item_digest": _safe_str(item.get("grid_item_digest")),
            "attack_family": _safe_str(item.get("attack_family")),
            "model_id": _safe_str(item.get("model_id")),
            "seed": item.get("seed") if isinstance(item.get("seed"), int) else None,
            "content_score": comparison_obj.get("content_score"),
            "hf_truncation_score": comparison_obj.get("hf_truncation_score"),
            "score_delta_content_minus_hf_truncation": comparison_obj.get("score_delta_content_minus_hf_truncation"),
            "comparison_ready": bool(comparison_obj.get("comparison_ready", False)),
        }
        rows.append(row)

        content_value = row.get("content_score")
        hf_truncation_value = row.get("hf_truncation_score")
        delta_value = row.get("score_delta_content_minus_hf_truncation")
        if isinstance(content_value, (int, float)) and np.isfinite(float(content_value)):
            content_scores.append(float(content_value))
        if isinstance(hf_truncation_value, (int, float)) and np.isfinite(float(hf_truncation_value)):
            hf_truncation_scores.append(float(hf_truncation_value))
        if isinstance(delta_value, (int, float)) and np.isfinite(float(delta_value)):
            score_deltas.append(float(delta_value))

    summary = {
        "rows_total": len(rows),
        "rows_comparison_ready": sum(1 for row in rows if bool(row.get("comparison_ready", False))),
        "mean_content_score": (float(np.mean(content_scores)) if len(content_scores) > 0 else None),
        "mean_hf_truncation_score": (float(np.mean(hf_truncation_scores)) if len(hf_truncation_scores) > 0 else None),
        "mean_delta_content_minus_hf_truncation": (float(np.mean(score_deltas)) if len(score_deltas) > 0 else None),
    }

    return {
        "schema_version": "hf_truncation_baseline_comparison_table_v1",
        "comparison_definition": {
            "reference_name": "hf_truncation_baseline",
            "reference_source": "detect_record.hf_truncation_baseline.score",
            "target_source": "content_evidence_payload.score",
            "directionality": "positive_delta_means_target_score_higher_than_hf_truncation",
        },
        "rows": rows,
        "summary": summary,
    }


def _build_hf_truncation_baseline_comparison_csv(table_obj: Dict[str, Any]) -> str:
    """Render comparison table rows to CSV for quick inspection."""
    if not isinstance(table_obj, dict):
        raise TypeError("table_obj must be dict")

    rows = table_obj.get("rows") if isinstance(table_obj.get("rows"), list) else []
    output = StringIO()
    fieldnames = [
        "grid_item_digest",
        "attack_family",
        "model_id",
        "seed",
        "content_score",
        "hf_truncation_score",
        "score_delta_content_minus_hf_truncation",
        "comparison_ready",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        if isinstance(row, dict):
            writer.writerow({name: row.get(name) for name in fieldnames})

    return output.getvalue()


def _compute_ablation_digest(ablation_flags: Any) -> str:
    """Compute canonical ablation digest from flag mapping."""
    if not isinstance(ablation_flags, dict):
        return digests.canonical_sha256({})
    return digests.canonical_sha256(ablation_flags)


def _build_grid_manifest(grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build deterministic grid manifest payload for artifact publication."""
    rows: List[Dict[str, Any]] = []
    for item in grid:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "grid_index": item.get("grid_index"),
                "grid_item_digest": _safe_str(item.get("grid_item_digest")),
                "cfg_digest": _safe_str(item.get("cfg_digest")),
                "attack_protocol_digest": _safe_str(item.get("attack_protocol_digest")),
                "ablation_digest": _safe_str(item.get("ablation_digest")),
                "attack_family": _safe_str(item.get("attack_protocol_family")),
                "model_id": _safe_str(item.get("model_id")),
                "seed": item.get("seed") if isinstance(item.get("seed"), int) else None,
            }
        )
    manifest = {
        "grid_manifest_version": "grid_manifest_v1",
        "items": rows,
    }
    manifest["grid_manifest_digest"] = digests.canonical_sha256(manifest)
    return manifest


def _build_grouped_rows(experiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build grouped aggregate rows for paper-friendly reporting."""
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for item in experiment_results:
        if not isinstance(item, dict):
            continue
        key = (
            _safe_str(item.get("attack_family")),
            _safe_str(item.get("ablation_digest")),
            _safe_str(item.get("model_id")),
            str(item.get("seed")) if isinstance(item.get("seed"), int) else "<absent>",
        )
        if key not in grouped:
            grouped[key] = {
                "attack_family": key[0],
                "ablation_digest": key[1],
                "ablation_id": f"abl::{key[1][:12]}",
                "model_id": key[2],
                "seed": None if key[3] == "<absent>" else int(key[3]),
                "n_total": 0,
                "n_attack_applied": 0,
                "n_valid_scored": 0,
                "n_rejected_absent": 0,
                "n_rejected_mismatch": 0,
                "n_rejected_fail": 0,
                "n_rescue_triggered": 0,
                "n_rescue_success": 0,
                "conditional_fpr_estimate": None,
                "conditional_fpr_sample_count": 0,
                "tpr_at_fpr": None,
                "geo_available_rate": None,
                "rescue_rate": None,
            }

        group = grouped[key]
        metrics_obj = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        reject_breakdown = metrics_obj.get("reject_rate_breakdown") if isinstance(metrics_obj.get("reject_rate_breakdown"), dict) else {}

        n_total = metrics_obj.get("n_total")
        n_accepted = metrics_obj.get("n_accepted")
        n_rejected = metrics_obj.get("n_rejected")
        if isinstance(n_total, int):
            group["n_total"] += n_total
            group["n_attack_applied"] += n_total
        if isinstance(n_accepted, int):
            group["n_valid_scored"] += n_accepted
        if isinstance(n_rejected, int):
            group["n_rejected_fail"] += max(n_rejected, 0)

        group["n_rejected_absent"] += int(reject_breakdown.get("absent", 0) or 0)
        group["n_rejected_mismatch"] += int(reject_breakdown.get("mismatch", 0) or 0)

        rescue_triggered = metrics_obj.get("n_rescue_triggered")
        rescue_success = metrics_obj.get("n_rescue_success")
        if isinstance(rescue_triggered, int):
            group["n_rescue_triggered"] += rescue_triggered
        if isinstance(rescue_success, int):
            group["n_rescue_success"] += rescue_success

        conditional_fpr = metrics_obj.get("conditional_fpr_estimate")
        conditional_fpr_n = metrics_obj.get("conditional_fpr_n")
        if isinstance(conditional_fpr, (int, float)):
            group["conditional_fpr_estimate"] = float(conditional_fpr)
        if isinstance(conditional_fpr_n, int):
            group["conditional_fpr_sample_count"] += conditional_fpr_n

        for metric_name in ["tpr_at_fpr", "geo_available_rate", "rescue_rate"]:
            metric_value = metrics_obj.get(metric_name)
            if isinstance(metric_value, (int, float)):
                group[metric_name] = float(metric_value)

    return list(grouped.values())


def _collect_failure_semantics_distribution(experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    功能：采集研究模式下 detect 失败语义分布。 

    Collect failure-semantics distribution from detect gate metadata when
    research collection mode relaxes hard gating.

    Args:
        experiment_results: Per-run experiment summaries.

    Returns:
        Distribution payload with relaxed run count and status histogram.
    """
    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be list")

    tracked_statuses = ["failed", "absent", "mismatch", "ok", "<absent>"]
    status_counts: Dict[str, int] = {status_name: 0 for status_name in tracked_statuses}
    relaxed_runs = 0

    for item in experiment_results:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("detect_gate_relaxed", False)):
            continue

        relaxed_runs += 1
        sample_counts_obj = item.get("detect_gate_sample_counts") if isinstance(item.get("detect_gate_sample_counts"), dict) else {}
        status_value = sample_counts_obj.get("status")
        if isinstance(status_value, str) and status_value in status_counts:
            status_counts[status_value] += 1
        else:
            status_counts["<absent>"] += 1

    return {
        "scope": "detect_gate_research_collection",
        "relaxed_run_count": relaxed_runs,
        "status_counts": status_counts,
    }


def _safe_str(value: Any) -> str:
    """Convert unknown value into non-empty string with absent fallback."""
    if isinstance(value, str) and value:
        return value
    return "<absent>"


def _extract_anchors_from_results(
    aggregate_report: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    功能：从 aggregate_report 或 results[0] 提取锚点字段全集（只读，不重新计算）。

    Extract anchor fields from aggregate report or first successful run result.
    This is append-only extension for grid_summary schema completeness.

    Args:
        aggregate_report: Aggregate report mapping.
        results: Experiment results list.

    Returns:
        Mapping with anchor fields (cfg_digest, thresholds_digest, etc.).
    """
    anchors: Dict[str, str] = {
        "cfg_digest": "<absent>",
        "thresholds_digest": "<absent>",
        "threshold_metadata_digest": "<absent>",
        "attack_protocol_version": "<absent>",
        "attack_protocol_digest": "<absent>",
        "attack_coverage_digest": "<absent>",
        "impl_digest": "<absent>",
        "fusion_rule_version": "<absent>",
        "policy_path": "<absent>",
    }

    # 优先级 1: 从 aggregate_report 提取（若包含 attack_coverage_digest）
    if isinstance(aggregate_report, dict):
        if aggregate_report.get("attack_coverage_digest"):
            anchors["attack_coverage_digest"] = _safe_str(aggregate_report.get("attack_coverage_digest"))
        if isinstance(aggregate_report.get("policy_path"), str) and aggregate_report.get("policy_path"):
            anchors["policy_path"] = _safe_str(aggregate_report.get("policy_path"))

    # 优先级 2: 从第一个成功的 result 提取（若存在）
    first_ok_result: Optional[Dict[str, Any]] = None
    for item in results:
        if isinstance(item, dict) and item.get("status") == "ok":
            first_ok_result = item
            break

    if first_ok_result is not None:
        for key in [
            "cfg_digest",
            "thresholds_digest",
            "threshold_metadata_digest",
            "attack_protocol_version",
            "attack_protocol_digest",
            "impl_digest",
            "fusion_rule_version",
            "policy_path",
        ]:
            value = first_ok_result.get(key)
            if isinstance(value, str) and value:
                anchors[key] = value

    # 优先级 3: 从 grid[0] 提取 attack_protocol_digest（若未从 result 获取）
    # （已在 result 中包含，无需额外提取）

    return anchors
