"""
LF attestation 细粒度链路定位报告。
Module type: Core innovation module

审计职责：
1. 自动定位当前主样本 LF attestation trace。
2. 自动选择一个 attack-aware 且已 attested 的正样本作为对照。
3. 按 plan / trajectory / projection / posterior / bit-agreement 分层输出差异。
4. 生成可落盘的 formal comparison report，用于定位最早 divergence stage。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np


_NUMERICAL_DIFF_ATOL = 1e-12
_NUMERICAL_DIFF_RTOL = 1e-9


def _load_json(path: Path) -> Dict[str, Any] | None:
    """
    功能：加载 JSON 文件为字典。

    Load a JSON file as a dictionary.

    Args:
        path: JSON file path.

    Returns:
        Parsed dictionary, or None when unavailable.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(Dict[str, Any], payload) if isinstance(payload, dict) else None


def _find_run_root(repo_root: Path) -> Path | None:
    """
    功能：定位当前主样本所在的正式 run_root。

    Locate the formal run root that contains the main-sample attestation artifact.

    Args:
        repo_root: Repository root path.

    Returns:
        Run root path or None when unavailable.
    """
    candidates = list(repo_root.glob("outputs/*/artifacts/attestation/attestation_result.json"))
    if not candidates:
        return None
    latest_path = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return latest_path.parent.parent.parent


def _as_float_list(value: Any) -> list[float] | None:
    """
    功能：规范化数值序列为 float 列表。

    Normalize a numeric sequence into a float list.

    Args:
        value: Candidate sequence.

    Returns:
        Float list, or None when unavailable.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        result.append(float(item))
    return result


def _as_int_list(value: Any) -> list[int] | None:
    """
    功能：规范化整数序列。

    Normalize an integer sequence.

    Args:
        value: Candidate sequence.

    Returns:
        Integer list, or None when unavailable.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[int] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        result.append(int(item))
    return result


def _vector_metrics(left: Any, right: Any) -> Dict[str, Any]:
    """
    功能：计算两组向量的基础差异统计。

    Compute basic discrepancy statistics for two vectors.

    Args:
        left: Left vector-like sequence.
        right: Right vector-like sequence.

    Returns:
        Metric dictionary with status and distances.
    """
    left_values = _as_float_list(left)
    right_values = _as_float_list(right)
    if left_values is None or right_values is None:
        return {
            "status": "insufficient_observability",
            "difference_class": "numerical",
        }
    if len(left_values) != len(right_values):
        return {
            "status": "divergent",
            "difference_class": "numerical",
            "length_left": len(left_values),
            "length_right": len(right_values),
        }

    left_arr = np.asarray(left_values, dtype=np.float64)
    right_arr = np.asarray(right_values, dtype=np.float64)
    l2_distance = float(np.linalg.norm(left_arr - right_arr))
    max_abs_diff = float(np.max(np.abs(left_arr - right_arr))) if left_arr.size else 0.0
    dot = float(np.dot(left_arr, right_arr)) if left_arr.size else 0.0
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr)) if left_arr.size else 0.0
    cosine_similarity = float(dot / denom) if denom > 0.0 else None
    numerically_equal = bool(
        np.allclose(
            left_arr,
            right_arr,
            atol=_NUMERICAL_DIFF_ATOL,
            rtol=_NUMERICAL_DIFF_RTOL,
        )
    )
    return {
        "status": "ok" if numerically_equal else "divergent",
        "difference_class": "numerical",
        "length": int(left_arr.size),
        "l2_distance": l2_distance,
        "max_abs_diff": max_abs_diff,
        "cosine_similarity": cosine_similarity,
        "numerically_equal": numerically_equal,
    }


def _compare_stage_fields(
    left: Dict[str, Any],
    right: Dict[str, Any],
    field_names: Sequence[str],
) -> Dict[str, Any]:
    """
    功能：比较一组离散字段。

    Compare a group of discrete fields between two stage payloads.

    Args:
        left: Left payload.
        right: Right payload.
        field_names: Field names to compare.

    Returns:
        Structured comparison result.
    """
    compared_fields: Dict[str, Any] = {}
    differing_fields: list[str] = []
    missing_fields: list[str] = []
    for field_name in field_names:
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        compared_fields[field_name] = {
            "main": left_value,
            "reference": right_value,
            "matches": left_value == right_value,
        }
        if left_value is None or right_value is None:
            missing_fields.append(field_name)
        elif left_value != right_value:
            differing_fields.append(field_name)

    status = "divergent" if differing_fields else ("ok" if len(missing_fields) < len(field_names) else "insufficient_observability")
    return {
        "status": status,
        "difference_class": "structural",
        "compared_fields": compared_fields,
        "differing_fields": differing_fields,
        "missing_fields": missing_fields,
    }


def _resolve_record_plan_digest(record: Dict[str, Any]) -> str | None:
    """
    功能：从 attestation record 中解析 plan_digest。 

    Resolve plan_digest from an attestation-bearing record.

    Args:
        record: Detect or evaluate-input record mapping.

    Returns:
        Canonical plan_digest when available; otherwise None.
    """
    if not isinstance(record, dict):
        return None

    attestation = record.get("attestation")
    if isinstance(attestation, dict):
        statement = attestation.get("statement")
        if isinstance(statement, dict):
            plan_digest = statement.get("plan_digest")
            if isinstance(plan_digest, str) and plan_digest:
                return plan_digest

    for field_name in [
        "plan_digest_observed",
        "plan_digest_expected",
        "plan_digest",
    ]:
        plan_digest = record.get(field_name)
        if isinstance(plan_digest, str) and plan_digest:
            return plan_digest
    return None


def _resolve_trace_plan_digest(trace: Dict[str, Any], record: Dict[str, Any]) -> str | None:
    """
    功能：优先从 trace，再回退到 record 解析 plan_digest。 

    Resolve plan_digest from trace first, then fall back to record payload.

    Args:
        trace: Attestation trace payload.
        record: Record payload.

    Returns:
        Canonical plan_digest when available; otherwise None.
    """
    if isinstance(trace, dict):
        plan_digest = trace.get("plan_digest")
        if isinstance(plan_digest, str) and plan_digest:
            return plan_digest
    return _resolve_record_plan_digest(record)


def _build_plan_alignment_payload(trace: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造 plan 层比较载荷。 

    Build the comparison payload for the plan-alignment stage.

    Args:
        trace: Trace payload.
        record: Record payload.

    Returns:
        Plan-alignment comparison payload.
    """
    return {
        "basis_rank": trace.get("basis_rank"),
        "edit_timestep": trace.get("edit_timestep"),
        "variance": trace.get("variance"),
        "projection_seed": trace.get("projection_seed"),
        "plan_digest": _resolve_trace_plan_digest(trace, record),
        "lf_basis_digest": trace.get("lf_basis_digest"),
        "projection_matrix_digest": trace.get("projection_matrix_digest"),
        "trajectory_feature_spec_digest": trace.get("trajectory_feature_spec_digest"),
    }


def _build_root_cause_evidence(stage_name: str, stage_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：根据 stage 差异构造根因证据。 

    Build root-cause evidence for a divergence stage.

    Args:
        stage_name: Stage name.
        stage_result: Stage comparison result.

    Returns:
        Structured root-cause evidence mapping.
    """
    difference_class = stage_result.get("difference_class")
    if stage_name == "plan_alignment":
        return {
            "difference_class": difference_class,
            "message": "Plan-layer digests or structural anchors diverge before numeric LF evidence comparison.",
            "differing_fields": stage_result.get("differing_fields"),
            "missing_fields": stage_result.get("missing_fields"),
        }
    if stage_name == "trajectory_feature_alignment":
        return {
            "difference_class": difference_class,
            "message": "Trajectory feature vectors diverge before LF projection.",
            "metrics": stage_result,
        }
    if stage_name == "projection_alignment":
        return {
            "difference_class": difference_class,
            "message": "Projected LF coefficients diverge before posterior recovery.",
            "metrics": stage_result,
        }
    if stage_name == "posterior_alignment":
        return {
            "difference_class": difference_class,
            "message": "Posterior recovery diverges before final bit agreement aggregation.",
            "metrics": stage_result,
        }
    if stage_name == "bit_agreement_alignment":
        return {
            "difference_class": difference_class,
            "message": "Bit-level agreement diverges after posterior recovery.",
            "metrics": stage_result,
        }
    return {
        "difference_class": difference_class,
        "message": "No fully observed divergence stage was found with current artifacts.",
    }


def _find_reference_sample(
    run_root: Path,
    main_plan_digest: str | None,
) -> Dict[str, Any] | None:
    """
    功能：自动选择 attested attack-aware 正样本作为 LF 对照。

    Select an attested attack-aware positive sample as the LF reference.

    Args:
        run_root: Formal run root path.

    Returns:
        Structured selection payload, or None when unavailable.
    """
    candidates = list(run_root.glob("outputs/experiment_matrix/experiments/*/artifacts/evaluate_inputs/detect_record_with_attack.json"))
    same_plan_candidates: list[tuple[float, float, str, Path, Dict[str, Any], Path, str | None]] = []
    fallback_candidates: list[tuple[float, float, str, Path, Dict[str, Any], Path, str | None]] = []
    for record_path in candidates:
        record = _load_json(record_path)
        if not isinstance(record, dict):
            continue
        attestation = record.get("attestation")
        if not isinstance(attestation, dict):
            continue
        final_decision = attestation.get("final_event_attested_decision")
        if not isinstance(final_decision, dict) or final_decision.get("status") != "attested":
            continue
        image_evidence = attestation.get("image_evidence_result")
        if not isinstance(image_evidence, dict):
            continue
        channel_scores = image_evidence.get("channel_scores")
        lf_score = 0.0
        if isinstance(channel_scores, dict) and isinstance(channel_scores.get("lf"), (int, float)):
            lf_score = float(channel_scores.get("lf"))
        content_score = float(attestation.get("content_attestation_score") or 0.0)
        attack_name = str(record.get("attack_name") or "")
        trace_path = record_path.parent.parent / "attestation" / "lf_attestation_trace.json"
        if not trace_path.is_file():
            continue
        candidate_plan_digest = _resolve_record_plan_digest(record)
        candidate_tuple = (
            content_score,
            lf_score,
            attack_name,
            record_path,
            record,
            trace_path,
            candidate_plan_digest,
        )
        if isinstance(main_plan_digest, str) and main_plan_digest and candidate_plan_digest == main_plan_digest:
            same_plan_candidates.append(candidate_tuple)
        else:
            fallback_candidates.append(candidate_tuple)

    selected_candidates = same_plan_candidates if same_plan_candidates else fallback_candidates
    if not selected_candidates:
        return None
    selected = sorted(selected_candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)[0]
    selection_strategy = "same_plan_preferred" if same_plan_candidates else "cross_plan_fallback"
    return {
        "record_path": selected[3],
        "record": selected[4],
        "trace_path": selected[5],
        "plan_digest": selected[6],
        "selection_strategy": selection_strategy,
        "same_plan_candidate_count": len(same_plan_candidates),
        "fallback_candidate_count": len(fallback_candidates),
    }


def build_lf_attestation_chain_report(repo_root: Path) -> Dict[str, Any]:
    """
    功能：构造 LF attestation 分层对比报告。

    Build the layered LF attestation comparison report.

    Args:
        repo_root: Repository root path.

    Returns:
        Structured report dictionary.
    """
    run_root = _find_run_root(repo_root)
    if run_root is None:
        return {
            "status": "na",
            "reason": "run_root_not_found",
        }

    main_trace_path = run_root / "artifacts" / "attestation" / "lf_attestation_trace.json"
    main_record_path = run_root / "artifacts" / "detect_np" / "records" / "detect_record.json"
    main_trace = _load_json(main_trace_path)
    main_record = _load_json(main_record_path)
    main_plan_digest = _resolve_trace_plan_digest(main_trace or {}, main_record or {}) if isinstance(main_trace, dict) and isinstance(main_record, dict) else None
    reference_bundle = _find_reference_sample(run_root, main_plan_digest)
    if main_trace is None or main_record is None or reference_bundle is None:
        return {
            "status": "na",
            "reason": "required_attestation_artifacts_missing",
            "run_root": str(run_root),
        }

    reference_record_path = Path(str(reference_bundle["record_path"]))
    reference_record = cast(Dict[str, Any], reference_bundle["record"])
    reference_trace_path = Path(str(reference_bundle["trace_path"]))
    reference_trace = _load_json(reference_trace_path)
    if reference_trace is None:
        return {
            "status": "na",
            "reason": "reference_trace_missing",
            "run_root": str(run_root),
        }

    main_attestation = cast(Dict[str, Any], main_record.get("attestation") or {})
    reference_attestation = cast(Dict[str, Any], reference_record.get("attestation") or {})

    reference_plan_digest = _resolve_trace_plan_digest(reference_trace, reference_record)
    plan_alignment = _compare_stage_fields(
        _build_plan_alignment_payload(main_trace, main_record),
        _build_plan_alignment_payload(reference_trace, reference_record),
        [
            "basis_rank",
            "edit_timestep",
            "variance",
            "projection_seed",
            "plan_digest",
            "lf_basis_digest",
            "projection_matrix_digest",
            "trajectory_feature_spec_digest",
        ],
    )
    trajectory_alignment = _vector_metrics(
        main_trace.get("trajectory_feature_vector"),
        reference_trace.get("trajectory_feature_vector"),
    )
    projection_alignment = _vector_metrics(
        main_trace.get("projected_lf_coeffs"),
        reference_trace.get("projected_lf_coeffs"),
    )
    posterior_alignment = _vector_metrics(
        main_trace.get("posterior_values"),
        reference_trace.get("posterior_values"),
    )
    bit_alignment: Dict[str, Any] = {
        "status": "insufficient_observability",
    }
    main_expected = _as_int_list(main_trace.get("expected_bit_signs"))
    reference_expected = _as_int_list(reference_trace.get("expected_bit_signs"))
    main_posterior_signs = _as_int_list(main_trace.get("posterior_signs"))
    reference_posterior_signs = _as_int_list(reference_trace.get("posterior_signs"))
    main_mismatch_indices = _as_int_list(main_trace.get("mismatch_indices"))
    reference_mismatch_indices = _as_int_list(reference_trace.get("mismatch_indices"))
    if (
        main_expected is not None
        and reference_expected is not None
        and main_posterior_signs is not None
        and reference_posterior_signs is not None
        and main_mismatch_indices is not None
        and reference_mismatch_indices is not None
    ):
        overlap = sorted(set(main_mismatch_indices).intersection(reference_mismatch_indices))
        bit_alignment = {
            "status": "divergent" if main_mismatch_indices != reference_mismatch_indices else "ok",
            "difference_class": "numerical",
            "expected_bits_match": main_expected == reference_expected,
            "posterior_sign_mismatch_count": sum(
                1
                for left_value, right_value in zip(main_posterior_signs, reference_posterior_signs)
                if left_value != right_value
            ),
            "main_mismatch_indices": main_mismatch_indices,
            "reference_mismatch_indices": reference_mismatch_indices,
            "shared_mismatch_indices": overlap,
            "main_agreement_count": main_trace.get("agreement_count"),
            "reference_agreement_count": reference_trace.get("agreement_count"),
        }

    stage_order = [
        ("plan_alignment", plan_alignment),
        ("trajectory_feature_alignment", trajectory_alignment),
        ("projection_alignment", projection_alignment),
        ("posterior_alignment", posterior_alignment),
        ("bit_agreement_alignment", bit_alignment),
    ]
    divergent_stages = [
        (stage_name, stage_result)
        for stage_name, stage_result in stage_order
        if stage_result.get("status") == "divergent"
    ]
    primary_divergence_stage: Optional[str] = None
    secondary_divergence_stage: Optional[str] = None
    upstream_stages_consistent: list[str] = []
    if divergent_stages:
        primary_divergence_stage = divergent_stages[0][0]
        if len(divergent_stages) >= 2:
            secondary_divergence_stage = divergent_stages[1][0]
        primary_index = [stage_name for stage_name, _ in stage_order].index(primary_divergence_stage)
        for stage_name, stage_result in stage_order[:primary_index]:
            if stage_result.get("status") == "ok":
                upstream_stages_consistent.append(stage_name)

    if primary_divergence_stage is None:
        primary_divergence_stage = "undetermined"
    if secondary_divergence_stage is None:
        secondary_divergence_stage = "undetermined"

    stage_results = {stage_name: stage_result for stage_name, stage_result in stage_order}
    primary_root_cause_evidence = _build_root_cause_evidence(
        primary_divergence_stage,
        stage_results.get(primary_divergence_stage, {}),
    )
    secondary_root_cause_evidence = _build_root_cause_evidence(
        secondary_divergence_stage,
        stage_results.get(secondary_divergence_stage, {}),
    )

    return {
        "status": "ok",
        "run_root": str(run_root),
        "main_sample": {
            "trace_path": str(main_trace_path),
            "record_path": str(main_record_path),
            "lf_attestation_score": main_trace.get("lf_attestation_score"),
            "agreement_count": main_trace.get("agreement_count"),
            "plan_digest": main_plan_digest,
            "event_attestation_status": cast(Dict[str, Any], main_attestation.get("final_event_attested_decision") or {}).get("status"),
        },
        "reference_sample": {
            "trace_path": str(reference_trace_path),
            "record_path": str(reference_record_path),
            "attack_name": reference_record.get("attack_name"),
            "lf_attestation_score": reference_trace.get("lf_attestation_score"),
            "agreement_count": reference_trace.get("agreement_count"),
            "plan_digest": reference_plan_digest,
            "selection_strategy": reference_bundle.get("selection_strategy"),
            "plan_match_status": (
                "same_plan"
                if isinstance(main_plan_digest, str) and main_plan_digest and reference_plan_digest == main_plan_digest
                else "cross_plan"
            ),
            "event_attestation_status": cast(Dict[str, Any], reference_attestation.get("final_event_attested_decision") or {}).get("status"),
        },
        "stages": {
            "plan_alignment": plan_alignment,
            "trajectory_feature_alignment": trajectory_alignment,
            "projection_alignment": projection_alignment,
            "posterior_alignment": posterior_alignment,
            "bit_agreement_alignment": bit_alignment,
        },
        "primary_divergence_stage": primary_divergence_stage,
        "secondary_divergence_stage": secondary_divergence_stage,
        "primary_root_cause_evidence": primary_root_cause_evidence,
        "secondary_root_cause_evidence": secondary_root_cause_evidence,
        "upstream_stages_consistent": upstream_stages_consistent,
    }


def audit_lf_attestation_chain(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行 LF attestation 链路定位审计并落盘报告。

    Execute the LF attestation localization audit and persist the report.

    Args:
        repo_root: Repository root path.

    Returns:
        Structured audit result.
    """
    report = build_lf_attestation_chain_report(repo_root)
    if report.get("status") != "ok":
        return {
            "audit_id": "attestation.lf_chain_localization",
            "severity": "NON_BLOCK",
            "result": "N.A.",
            "evidence": report,
        }

    run_root = Path(str(report["run_root"]))
    report_path = run_root / "artifacts" / "attestation" / "lf_attestation_chain_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "audit_id": "attestation.lf_chain_localization",
        "severity": "NON_BLOCK",
        "result": "PASS",
        "evidence": {
            "report_path": str(report_path),
            "primary_divergence_stage": report.get("primary_divergence_stage"),
            "secondary_divergence_stage": report.get("secondary_divergence_stage"),
            "upstream_stages_consistent": report.get("upstream_stages_consistent"),
        },
    }


def main(repo_root_str: Optional[str] = None) -> int:
    """
    功能：执行 LF attestation 链路定位审计 CLI。

    Execute the LF attestation localization audit CLI.

    Args:
        repo_root_str: Optional repository root path.

    Returns:
        Exit code.
    """
    repo_root = Path(repo_root_str) if repo_root_str else Path.cwd()
    result = audit_lf_attestation_chain(repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("result") != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))