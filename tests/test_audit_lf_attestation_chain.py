"""
File purpose: LF attestation 细粒度链路定位审计测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audits import audit_lf_attestation_chain


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_attestation_record(plan_digest: str, *, status: str, content_score: float, lf_score: float, attack_name: str | None = None) -> dict:
    record = {
        "attestation": {
            "statement": {
                "plan_digest": plan_digest,
            },
            "content_attestation_score": content_score,
            "image_evidence_result": {"channel_scores": {"lf": lf_score}},
            "final_event_attested_decision": {"status": status},
        },
        "plan_digest": plan_digest,
        "plan_digest_observed": plan_digest,
        "plan_digest_expected": plan_digest,
    }
    if attack_name is not None:
        record["attack_name"] = attack_name
    return record


def _build_trace(
    *,
    lf_score: float,
    agreement_count: int,
    trajectory_feature_vector: list[float],
    projected_lf_coeffs: list[float],
    posterior_values: list[float],
    mismatch_indices: list[int],
    plan_digest: str | None = None,
    lf_basis_digest: str | None = None,
    projection_matrix_digest: str | None = None,
    trajectory_feature_spec_digest: str | None = None,
) -> dict:
    payload = {
        "lf_attestation_score": lf_score,
        "agreement_count": agreement_count,
        "basis_rank": 36,
        "edit_timestep": 0,
        "variance": 1.5,
        "projection_seed": 11,
        "trajectory_feature_vector": trajectory_feature_vector,
        "projected_lf_coeffs": projected_lf_coeffs,
        "posterior_values": posterior_values,
        "expected_bit_signs": [1, -1, 1],
        "posterior_signs": [1 if value > 0 else -1 for value in posterior_values],
        "mismatch_indices": mismatch_indices,
    }
    if plan_digest is not None:
        payload["plan_digest"] = plan_digest
    if lf_basis_digest is not None:
        payload["lf_basis_digest"] = lf_basis_digest
    if projection_matrix_digest is not None:
        payload["projection_matrix_digest"] = projection_matrix_digest
    if trajectory_feature_spec_digest is not None:
        payload["trajectory_feature_spec_digest"] = trajectory_feature_spec_digest
    return payload


def test_audit_lf_attestation_chain_writes_report_with_plan_divergence(tmp_path) -> None:
    """验证：当 plan-layer digest 不一致时，报告必须定位到 plan_alignment。"""
    run_root = tmp_path / "outputs" / "Paper_Full_Cuda"

    _write_json(
        run_root / "artifacts" / "attestation" / "attestation_result.json",
        {"final_event_attested_decision": {"status": "mismatch"}},
    )
    _write_json(
        run_root / "artifacts" / "detect_np" / "records" / "detect_record.json",
        _build_attestation_record(
            "a" * 64,
            status="mismatch",
            content_score=0.57,
            lf_score=17 / 36,
        ),
    )
    _write_json(
        run_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=17 / 36,
            agreement_count=17,
            trajectory_feature_vector=[0.1, 0.2, 0.3],
            projected_lf_coeffs=[0.5, -0.4, 0.1],
            posterior_values=[0.8, -0.7, 0.4],
            mismatch_indices=[1, 4],
            plan_digest="a" * 64,
            lf_basis_digest="b" * 64,
            projection_matrix_digest="c" * 64,
            trajectory_feature_spec_digest="d" * 64,
        ),
    )

    experiment_root = run_root / "outputs" / "experiment_matrix" / "experiments" / "item_0001"
    _write_json(
        experiment_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json",
        _build_attestation_record(
            "e" * 64,
            status="attested",
            content_score=0.71,
            lf_score=24 / 36,
            attack_name="crop::v1",
        ),
    )
    _write_json(
        experiment_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=24 / 36,
            agreement_count=24,
            trajectory_feature_vector=[0.1, 0.2, 0.31],
            projected_lf_coeffs=[0.5, -0.1, 0.8],
            posterior_values=[0.8, -0.2, 0.9],
            mismatch_indices=[2],
            plan_digest="e" * 64,
            lf_basis_digest="f" * 64,
            projection_matrix_digest="1" * 64,
            trajectory_feature_spec_digest="d" * 64,
        ),
    )

    result = audit_lf_attestation_chain.audit_lf_attestation_chain(tmp_path)

    assert result["result"] == "PASS"
    evidence = result["evidence"]
    assert evidence["primary_divergence_stage"] == "plan_alignment"
    assert evidence["secondary_divergence_stage"] == "trajectory_feature_alignment"
    report_path = Path(evidence["report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["primary_divergence_stage"] == "plan_alignment"
    assert report["secondary_divergence_stage"] == "trajectory_feature_alignment"
    assert report["primary_root_cause_evidence"]["differing_fields"] == [
        "plan_digest",
        "lf_basis_digest",
        "projection_matrix_digest",
    ]
    assert report["secondary_root_cause_evidence"]["difference_class"] == "numerical"
    assert report["reference_sample"]["selection_strategy"] == "cross_plan_fallback"


def test_audit_lf_attestation_chain_prefers_same_plan_attested_reference(tmp_path) -> None:
    """验证：存在同 plan attested 样本时，必须优先选择同 plan。"""
    run_root = tmp_path / "outputs" / "Paper_Full_Cuda"

    _write_json(
        run_root / "artifacts" / "attestation" / "attestation_result.json",
        {"final_event_attested_decision": {"status": "mismatch"}},
    )
    _write_json(
        run_root / "artifacts" / "detect_np" / "records" / "detect_record.json",
        _build_attestation_record(
            "same" * 16,
            status="mismatch",
            content_score=0.57,
            lf_score=0.45,
        ),
    )
    _write_json(
        run_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=0.45,
            agreement_count=16,
            trajectory_feature_vector=[0.1, 0.2, 0.3],
            projected_lf_coeffs=[0.2, 0.1, -0.3],
            posterior_values=[0.2, -0.3, 0.1],
            mismatch_indices=[1],
        ),
    )

    cross_plan_root = run_root / "outputs" / "experiment_matrix" / "experiments" / "item_cross"
    _write_json(
        cross_plan_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json",
        _build_attestation_record(
            "cross" * 16,
            status="attested",
            content_score=0.95,
            lf_score=0.90,
            attack_name="crop::v1",
        ),
    )
    _write_json(
        cross_plan_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=0.90,
            agreement_count=30,
            trajectory_feature_vector=[0.4, 0.5, 0.6],
            projected_lf_coeffs=[0.9, 0.8, 0.7],
            posterior_values=[0.9, -0.8, 0.7],
            mismatch_indices=[],
        ),
    )

    same_plan_root = run_root / "outputs" / "experiment_matrix" / "experiments" / "item_same"
    _write_json(
        same_plan_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json",
        _build_attestation_record(
            "same" * 16,
            status="attested",
            content_score=0.70,
            lf_score=0.68,
            attack_name="resize::v1",
        ),
    )
    _write_json(
        same_plan_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=0.68,
            agreement_count=24,
            trajectory_feature_vector=[0.15, 0.25, 0.35],
            projected_lf_coeffs=[0.25, 0.15, -0.2],
            posterior_values=[0.25, -0.2, 0.12],
            mismatch_indices=[],
        ),
    )

    report = audit_lf_attestation_chain.build_lf_attestation_chain_report(tmp_path)

    assert report["status"] == "ok"
    assert report["reference_sample"]["selection_strategy"] == "same_plan_preferred"
    assert report["reference_sample"]["plan_match_status"] == "same_plan"
    assert "item_same" in report["reference_sample"]["record_path"]


def test_audit_lf_attestation_chain_falls_back_to_cross_plan_when_needed(tmp_path) -> None:
    """验证：无同 plan attested 样本时，允许回退到跨 plan。"""
    run_root = tmp_path / "outputs" / "Paper_Full_Cuda"

    _write_json(
        run_root / "artifacts" / "attestation" / "attestation_result.json",
        {"final_event_attested_decision": {"status": "mismatch"}},
    )
    _write_json(
        run_root / "artifacts" / "detect_np" / "records" / "detect_record.json",
        _build_attestation_record(
            "main" * 16,
            status="mismatch",
            content_score=0.50,
            lf_score=0.40,
        ),
    )
    _write_json(
        run_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=0.40,
            agreement_count=15,
            trajectory_feature_vector=[0.1, 0.2, 0.3],
            projected_lf_coeffs=[0.1, 0.0, -0.1],
            posterior_values=[0.1, -0.2, 0.05],
            mismatch_indices=[0, 2],
        ),
    )

    fallback_root = run_root / "outputs" / "experiment_matrix" / "experiments" / "item_fallback"
    _write_json(
        fallback_root / "artifacts" / "evaluate_inputs" / "detect_record_with_attack.json",
        _build_attestation_record(
            "other" * 16,
            status="attested",
            content_score=0.72,
            lf_score=0.66,
            attack_name="crop::v1",
        ),
    )
    _write_json(
        fallback_root / "artifacts" / "attestation" / "lf_attestation_trace.json",
        _build_trace(
            lf_score=0.66,
            agreement_count=24,
            trajectory_feature_vector=[0.12, 0.22, 0.32],
            projected_lf_coeffs=[0.3, 0.2, -0.1],
            posterior_values=[0.3, -0.2, 0.08],
            mismatch_indices=[1],
        ),
    )

    report = audit_lf_attestation_chain.build_lf_attestation_chain_report(tmp_path)

    assert report["status"] == "ok"
    assert report["reference_sample"]["selection_strategy"] == "cross_plan_fallback"
    assert report["reference_sample"]["plan_match_status"] == "cross_plan"
    assert "item_fallback" in report["reference_sample"]["record_path"]