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


_FORBIDDEN_ARTIFACT_ANCHOR_FIELDS = {
    "contract_bound_digest",
    "whitelist_bound_digest",
    "policy_path_semantics_bound_digest",
    "injection_scope_manifest_bound_digest",
}


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
        "status": "fail",
        "failure_reason": "<absent>",
        "cfg_digest": _safe_str(grid_item_cfg.get("cfg_digest")),
        "plan_digest": "<absent>",
        "thresholds_digest": "<absent>",
        "threshold_metadata_digest": "<absent>",
        "ablation_digest": _safe_str(grid_item_cfg.get("ablation_digest")),
        "attack_protocol_digest": _safe_str(grid_item_cfg.get("attack_protocol_digest")),
        "attack_protocol_version": grid_item_cfg.get("attack_protocol_version", "<absent>"),
        "impl_digest": "<absent>",
        "fusion_rule_version": "<absent>",
        "t2smark_comparison": {
            "content_score": None,
            "t2smark_score": None,
            "score_delta_content_minus_t2smark": None,
            "comparison_ready": False,
            "comparison_source": "real_t2smark_baseline_required",
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
                "t2smark_comparison": _extract_t2smark_real_comparison_from_detect_record(run_root),
                "detect_gate_relaxed": bool(detect_gate_info.get("gate_relaxed", False)),
                "detect_gate_relax_reason": _safe_str(detect_gate_info.get("reason")),
                "detect_gate_sample_counts": detect_gate_info.get("sample_counts") if isinstance(detect_gate_info.get("sample_counts"), dict) else {},
            }
        )
        _enforce_paper_acceptance_gate(summary=summary, grid_item_cfg=grid_item_cfg, run_root=run_root)
    except Exception as exc:
        # 单实验执行失败，必须记录失败原因并返回。
        summary["status"] = "fail"
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


def _extract_t2smark_real_comparison_from_detect_record(run_root: Path) -> Dict[str, Any]:
    """Extract same-sample comparison values from real T2SMark baseline record."""
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    result: Dict[str, Any] = {
        "content_score": None,
        "t2smark_score": None,
        "score_delta_content_minus_t2smark": None,
        "comparison_ready": False,
        "comparison_source": "real_t2smark_baseline_required",
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

    baseline_payload = detect_record.get("t2smark_baseline")
    if isinstance(baseline_payload, dict):
        baseline_trace = baseline_payload.get("trace")
        if isinstance(baseline_trace, dict):
            result["baseline_trace"] = baseline_trace
        baseline_status = baseline_payload.get("status")
        if isinstance(baseline_status, str) and baseline_status:
            result["baseline_status"] = baseline_status
        baseline_score = baseline_payload.get("score")
        if isinstance(baseline_score, (int, float)) and np.isfinite(float(baseline_score)):
            result["t2smark_score"] = float(baseline_score)
            result["baseline_status"] = "ok"
            result["comparison_source"] = "real_t2smark_baseline_record"

    if isinstance(result.get("content_score"), float) and isinstance(result.get("t2smark_score"), float):
        result["score_delta_content_minus_t2smark"] = float(result["content_score"] - result["t2smark_score"])
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
    t2smark_comparison = summary.get("t2smark_comparison") if isinstance(summary.get("t2smark_comparison"), dict) else {}

    if bool(pipeline_runtime_meta.get("synthetic_pipeline", False)):
        summary["status"] = "fail"
        summary["failure_reason"] = "paper_acceptance_failed: synthetic_pipeline_true"
        return
    if detect_runtime_mode != "real":
        summary["status"] = "fail"
        summary["failure_reason"] = f"paper_acceptance_failed: detect_runtime_mode={detect_runtime_mode}"
        return
    if isinstance(geo_available_rate, (int, float)) and float(geo_available_rate) == 0.0:
        summary["status"] = "fail"
        summary["failure_reason"] = "paper_acceptance_failed: geo_available_rate_zero"
        return
    if not bool(t2smark_comparison.get("comparison_ready", False)):
        summary["status"] = "fail"
        summary["failure_reason"] = "paper_acceptance_failed: real_t2smark_baseline_missing"


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

    for item in grid:
        if not isinstance(item, dict):
            raise TypeError("grid items must be dict")
        result = run_single_experiment(item)
        results.append(result)
        if result.get("status") != "ok" and first_failure is None:
            first_failure = result
            if strict:
                break

    grid_manifest = _build_grid_manifest(grid)
    aggregate_report = build_aggregate_report(results, grid_manifest=grid_manifest)
    summary_paths = _write_grid_artifacts(grid, aggregate_report, results, strict)

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
    seed_value = grid_item_cfg.get("cfg_snapshot", {}).get("seed")
    model_id = grid_item_cfg.get("cfg_snapshot", {}).get("model_id")
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

    allow_failed_semantics_collection = grid_item_cfg.get("allow_failed_semantics_collection", False)
    if not isinstance(allow_failed_semantics_collection, bool):
        raise TypeError("grid item allow_failed_semantics_collection must be bool")

    detect_gate_info: Dict[str, Any] = {
        "gate_relaxed": False,
        "reason": "hard_gate_not_checked",
        "sample_counts": {},
    }

    for stage_name in ["embed", "detect", "calibrate", "evaluate"]:
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
        if stage_name == "detect":
            stage_overrides.append("enable_content_detect=true")
            stage_overrides.append("allow_threshold_fallback_for_tests=true")
        
        # calibrate 和 evaluate 需要 detect_records_glob 参数
        if stage_name in ["calibrate", "evaluate"]:
            detect_record_path = _prepare_detect_record_for_attack_grouping(run_root, grid_item_cfg)
            arg_name = f"{stage_name}_detect_records_glob"
            stage_overrides.append(f"{arg_name}={json.dumps(str(detect_record_path))}")
        
        # evaluate 需要额外的参数
        if stage_name == "evaluate":
            thresholds_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
            stage_overrides.append(f"evaluate_thresholds_path={json.dumps(str(thresholds_path))}")
        
        for key, value in sorted(ablation_flags.items()):
            stage_overrides.append(f"ablation.{key}={str(value).lower()}")


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

    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(enriched_path),
        obj=enriched_record,
    )
    return enriched_path


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
    t2smark_comparison_table_path = artifacts_dir / "t2smark_comparison_table.json"
    t2smark_comparison_table_csv_path = artifacts_dir / "t2smark_comparison_table.csv"

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
            "executed": len(results),
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

    t2smark_comparison_table = _build_t2smark_comparison_table(results)
    records_io.write_artifact_json_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(t2smark_comparison_table_path),
        obj=t2smark_comparison_table,
    )
    records_io.write_artifact_text_unbound(
        run_root=batch_root,
        artifacts_dir=artifacts_dir,
        path=str(t2smark_comparison_table_csv_path),
        content=_build_t2smark_comparison_csv(t2smark_comparison_table),
    )

    return {
        "batch_root": str(batch_root),
        "aggregate_report_path": str(aggregate_report_path),
        "grid_summary_path": str(grid_summary_path),
        "grid_manifest_path": str(grid_manifest_path),
        "attack_coverage_manifest_path": str(attack_coverage_manifest_path),
        "t2smark_comparison_table_path": str(t2smark_comparison_table_path),
        "t2smark_comparison_table_csv_path": str(t2smark_comparison_table_csv_path),
    }


def _build_t2smark_comparison_table(experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic same-sample comparison table against real T2SMark baseline."""
    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be list")

    rows: List[Dict[str, Any]] = []
    content_scores: List[float] = []
    t2smark_scores: List[float] = []
    score_deltas: List[float] = []

    for item in experiment_results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        comparison_obj = item.get("t2smark_comparison") if isinstance(item.get("t2smark_comparison"), dict) else {}
        row = {
            "grid_item_digest": _safe_str(item.get("grid_item_digest")),
            "attack_family": _safe_str(item.get("attack_family")),
            "model_id": _safe_str(item.get("model_id")),
            "seed": item.get("seed") if isinstance(item.get("seed"), int) else None,
            "content_score": comparison_obj.get("content_score"),
            "t2smark_score": comparison_obj.get("t2smark_score"),
            "score_delta_content_minus_t2smark": comparison_obj.get("score_delta_content_minus_t2smark"),
            "comparison_ready": bool(comparison_obj.get("comparison_ready", False)),
        }
        rows.append(row)

        content_value = row.get("content_score")
        t2smark_value = row.get("t2smark_score")
        delta_value = row.get("score_delta_content_minus_t2smark")
        if isinstance(content_value, (int, float)) and np.isfinite(float(content_value)):
            content_scores.append(float(content_value))
        if isinstance(t2smark_value, (int, float)) and np.isfinite(float(t2smark_value)):
            t2smark_scores.append(float(t2smark_value))
        if isinstance(delta_value, (int, float)) and np.isfinite(float(delta_value)):
            score_deltas.append(float(delta_value))

    summary = {
        "rows_total": len(rows),
        "rows_comparison_ready": sum(1 for row in rows if bool(row.get("comparison_ready", False))),
        "mean_content_score": (float(np.mean(content_scores)) if len(content_scores) > 0 else None),
        "mean_t2smark_score": (float(np.mean(t2smark_scores)) if len(t2smark_scores) > 0 else None),
        "mean_delta_content_minus_t2smark": (float(np.mean(score_deltas)) if len(score_deltas) > 0 else None),
    }

    return {
        "schema_version": "t2smark_comparison_table_v1",
        "comparison_definition": {
            "reference_name": "real_t2smark_baseline",
            "reference_source": "detect_record.t2smark_baseline.score",
            "target_source": "content_evidence_payload.score",
            "directionality": "positive_delta_means_target_score_higher_than_t2smark",
        },
        "rows": rows,
        "summary": summary,
    }


def _build_t2smark_comparison_csv(table_obj: Dict[str, Any]) -> str:
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
        "t2smark_score",
        "score_delta_content_minus_t2smark",
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
        ]:
            value = first_ok_result.get(key)
            if isinstance(value, str) and value:
                anchors[key] = value

    # 优先级 3: 从 grid[0] 提取 attack_protocol_digest（若未从 result 获取）
    # （已在 result 中包含，无需额外提取）

    return anchors
