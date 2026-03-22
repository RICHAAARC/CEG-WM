"""
File purpose: attestation trace artifact 治理回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def _prepare_fact_sources(tmp_run_root: Path):
    from main.core.contracts import load_frozen_contracts
    from main.policy.runtime_whitelist import load_runtime_whitelist, load_policy_path_semantics
    from main.core.injection_scope import load_injection_scope_manifest

    contracts = load_frozen_contracts()
    whitelist = load_runtime_whitelist()
    semantics = load_policy_path_semantics()
    injection_scope_manifest = load_injection_scope_manifest()
    records_dir = tmp_run_root / "records"
    artifacts_dir = tmp_run_root / "artifacts"
    logs_dir = tmp_run_root / "logs"
    return contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir


def _build_lf_trace_payload() -> dict:
    return {
        "artifact_type": "lf_attestation_trace",
        "attestation_digest": "a" * 64,
        "event_binding_digest": "b" * 64,
        "trace_commit": "c" * 64,
        "lf_attestation_score": 0.5,
        "agreement_count": 18,
        "n_bits_compared": 36,
        "basis_rank": 36,
        "variance": 1.5,
        "edit_timestep": 0,
        "trajectory_feature_spec": {"feature_operator": "masked_normalized_random_projection", "edit_timestep": 0},
        "trajectory_feature_vector": [0.1, 0.2, 0.3],
        "trajectory_feature_digest": "d" * 64,
        "projected_lf_coeffs": [0.2, -0.1, 0.05],
        "projected_lf_signs": [1, -1, 1],
        "projected_lf_digest": "e" * 64,
        "expected_bit_signs": [1, -1, 1],
        "posterior_values": [0.2, -0.1, 0.05],
        "posterior_signs": [1, -1, 1],
        "posterior_margin_values": [0.2, 0.1, 0.05],
        "agreement_indices": [0, 1, 2],
        "mismatch_indices": [],
        "weakest_posterior_indices": [2, 1],
        "weakest_posterior_margins": [0.05, 0.1],
        "plan_digest": "f" * 64,
        "lf_basis_digest": "1" * 64,
        "projection_matrix_digest": "2" * 64,
        "trajectory_feature_spec_digest": "3" * 64,
        "projection_seed": 17,
        "lf_attestation_trace_digest": "4" * 64,
    }


def _build_lf_alignment_payload() -> dict:
    return {
        "artifact_type": "lf_alignment_table",
        "attestation_digest": "a" * 64,
        "event_binding_digest": "b" * 64,
        "trace_commit": "c" * 64,
        "plan_digest": "d" * 64,
        "lf_basis_digest": "e" * 64,
        "projection_matrix_digest": "f" * 64,
        "embed_closed_loop_digest": "1" * 64,
        "embed_closed_loop_step_index": 12,
        "embed_closed_loop_selection_rule": "max_lf_delta_norm",
        "n_bits_compared": 3,
        "expected_bit_signs": [1, -1, 1],
        "embed_expected_bit_signs": [1, -1, 1],
        "formal_expected_bit_signs": [1, -1, 1],
        "embed_formal_expected_signs_match": True,
        "expected_signs_mismatch_reason": None,
        "pre_injection_coeffs": [-0.4, 0.2, -0.1],
        "injected_template_coeffs": [0.3, -0.5, 0.4],
        "post_injection_coeffs": [-0.1, -0.3, 0.3],
        "detect_side_coeffs": [-0.2, 0.2, 0.1],
        "signed_pre_alignment": [-0.4, -0.2, -0.1],
        "signed_template_alignment": [0.3, 0.5, 0.4],
        "signed_post_alignment": [-0.1, 0.3, 0.3],
        "signed_detect_alignment": [-0.2, -0.2, 0.1],
        "alignment_margin_threshold": 0.15,
        "pre_agreement_count": 0,
        "post_agreement_count": 2,
        "detect_agreement_count": 1,
        "strong_negative_pre_count": 2,
        "strong_negative_post_count": 0,
        "strong_negative_detect_count": 2,
        "post_still_negative_count": 1,
        "post_crosses_target_halfspace_count": 2,
        "detect_crosses_target_halfspace_count": 0,
        "detect_reverted_after_post_positive_count": 1,
        "lf_alignment_table_digest": "2" * 64,
    }


def _build_lf_planner_risk_report_payload() -> dict:
    return {
        "artifact_type": "lf_planner_risk_report",
        "risk_report_version": "v1",
        "risk_classification": "host_baseline_dominant",
        "lf_feature_count": 8,
        "lf_decomposition_shape": [16, 8],
        "planner_rank": 4,
        "host_baseline_ratio": 1.8,
        "sign_stability": 0.9,
        "reconstruction_residual_ratio": 0.12,
        "top1_energy_ratio": 0.74,
        "topk_energy_ratio": 0.92,
        "host_baseline_dominant_flag": True,
        "basis_sample_mismatch_flag": False,
        "detect_trajectory_shift_flag": False,
        "route_basis_bridge_digest": "3" * 64,
        "plan_digest": "4" * 64,
        "basis_digest": "5" * 64,
        "embed_formal_expected_signs_match": True,
        "expected_signs_mismatch_reason": None,
        "primary_evidence": {
            "evidence_type": "lf_closed_loop_posterior_counts",
            "risk_classification_driver": "host_baseline_dominant",
        },
        "per_dimension_summary": [
            {
                "dimension_index": 0,
                "expected_bit_sign": 1,
                "signed_pre_alignment": -0.4,
                "signed_template_alignment": 0.3,
                "signed_post_alignment": -0.1,
                "signed_detect_alignment": -0.2,
                "post_positive": False,
                "detect_positive": False,
                "is_high_confidence_mismatch": True,
                "routing_tag": "lf_feature_col:3",
                "decomposition_group": "mask_routed_projection",
            }
        ],
        "high_confidence_mismatch_dimensions": [
            {
                "dimension_index": 0,
                "signed_pre_alignment": -0.4,
                "signed_post_alignment": -0.1,
                "signed_detect_alignment": -0.2,
                "detect_side_coeff": -0.2,
                "pre_strong_negative": True,
                "post_still_negative": True,
                "detect_reverted_after_post_positive": False,
            }
        ],
        "routing_pattern_summary": {
            "route_basis_bridge_digest": "3" * 64,
            "mismatch_dimension_count": 1,
            "mismatch_dimension_indices": [0],
            "mismatch_feature_cols": [3],
            "mismatch_feature_col_counts": {"3": 1},
        },
        "host_baseline_risk_summary": {
            "strong_negative_pre_count": 2,
            "post_still_negative_count": 1,
            "detect_reverted_after_post_positive_count": 0,
            "post_crosses_target_halfspace_count": 1,
            "detect_crosses_target_halfspace_count": 0,
            "dominant_signal": "host_baseline_counts",
            "dominant_count": 2,
        },
    }


def _build_lf_retain_breakdown_payload() -> dict:
    return {
        "artifact_type": "lf_retain_breakdown",
        "attestation_digest": "a" * 64,
        "event_binding_digest": "b" * 64,
        "trace_commit": "c" * 64,
        "plan_digest": "d" * 64,
        "lf_basis_digest": "e" * 64,
        "projection_matrix_digest": "f" * 64,
        "embed_closed_loop_digest": "1" * 64,
        "embed_closed_loop_step_index": 12,
        "embed_closed_loop_selection_rule": "max_lf_delta_norm",
        "edit_timestep": 12,
        "embed_edit_timestep_step_index": 12,
        "embed_terminal_step_index": 15,
        "n_bits_compared": 3,
        "expected_bit_signs": [1, -1, 1],
        "pre_injection_coeffs": [-0.4, 0.2, -0.1],
        "selected_step_post_coeffs": [-0.1, -0.3, 0.3],
        "embed_edit_timestep_coeffs": [-0.2, -0.15, 0.25],
        "embed_terminal_step_coeffs": [-0.25, -0.1, 0.2],
        "detect_exact_timestep_coeffs": [-0.2, -0.2, 0.1],
        "embed_seed": 7102221260541468996,
        "detect_seed": 4388340890186534267,
        "same_seed_as_embed_available": True,
        "same_seed_as_embed_value": 7102221260541468996,
        "detect_protocol_classification": "rerun_exact_timestep_different_seed",
        "image_conditioned_reconstruction_available": False,
        "image_conditioned_reconstruction_status": "not_implemented",
        "same_seed_control_status": "ok",
        "same_seed_control_reason": None,
        "same_seed_control_trace_digest": "3" * 64,
        "same_seed_control_trajectory_digest": "4" * 64,
        "detect_exact_timestep_coeffs_same_seed_control": [-0.25, -0.1, 0.2],
        "cross_seed_protocol_loss_count": 1,
        "same_seed_residual_loss_count": 0,
        "cross_seed_protocol_loss_ratio": 0.5,
        "same_seed_residual_loss_ratio": 0.0,
        "protocol_root_cause_classification": "cross_seed_rerun_mismatch_dominant",
        "control_protocol_segments": [
            {"segment_name": "embed_terminal_to_detect_exact_detect_seed", "lost_positive_count": 1}
        ],
        "control_protocol_summary": {
            "protocol_root_cause_classification": "cross_seed_rerun_mismatch_dominant",
        },
        "stage_summaries": {
            "pre_injection": {"positive_count": 0},
            "selected_step_post": {"positive_count": 2},
        },
        "breakdown_segments": [
            {"segment_name": "selected_step_to_edit_timestep", "lost_positive_count": 1}
        ],
        "breakdown_summary": {
            "dominant_drift_segment": "selected_step_to_edit_timestep",
        },
        "lf_retain_breakdown_digest": "2" * 64,
    }


def _build_geo_rescue_diagnostics_payload() -> dict:
    return {
        "artifact_type": "geo_rescue_diagnostics",
        "attestation_digest": "a" * 64,
        "event_binding_digest": "b" * 64,
        "trace_commit": "c" * 64,
        "decision_mode": "content_primary_geo_rescue",
        "content_attestation_score": 0.62,
        "attested_threshold": 0.65,
        "geo_rescue_band_delta_low": 0.05,
        "geo_rescue_band_lower_bound": 0.6,
        "geo_rescue_min_score": 0.3,
        "quality_score": 0.92,
        "template_match_score": 0.2,
        "geo_score": 0.2,
        "geo_score_source": "template_match_score",
        "geo_rescue_eligible": True,
        "geo_rescue_applied": False,
        "geo_not_used_reason": "geometry_score_below_rescue_min",
        "sync_status": "ok",
        "anchor_status": "ok",
        "relation_digest_binding_status": "matched",
        "uncertainty": 0.08,
        "quality_vs_template_ratio": 4.6,
        "geo_score_vs_rescue_min_ratio": 0.666667,
        "content_gap_to_attested_threshold": 0.03,
        "geo_scale_classification": "quality_pass_template_fail_source_template",
        "template_match_internal_threshold": 0.02,
        "template_match_threshold_to_rescue_min_ratio": 0.066667,
        "template_score_scale_band": "between_internal_threshold_and_rescue_gate",
        "rescue_gate_scale_classification": "template_internal_threshold_far_below_rescue_gate",
        "positive_template_match_score_summary": {"sample_count": 2, "min": 0.05, "max": 0.12, "mean": 0.085, "median": 0.085},
        "positive_quality_score_summary": {"sample_count": 2, "min": 0.9, "max": 0.92, "mean": 0.91, "median": 0.91},
        "negative_template_match_score_summary": {"sample_count": 1, "min": 0.01, "max": 0.01, "mean": 0.01, "median": 0.01},
        "negative_quality_score_summary": {"sample_count": 1, "min": 0.2, "max": 0.2, "mean": 0.2, "median": 0.2},
        "positive_template_to_gate_max_ratio": 0.4,
        "positive_template_to_internal_threshold_max_ratio": 6.0,
        "geo_repair_direction_classification": "scale_misalignment_between_template_score_and_rescue_gate",
        "scale_control_scan_source": "configured_glob",
        "scale_control_scan_glob": "outputs/**/records/detect_record.json",
        "scale_control_scanned_record_count": 3,
        "scale_control_labelled_record_count": 3,
        "geo_rescue_diagnostics_digest": "3" * 64,
    }


def test_attestation_trace_artifact_contracts_are_registered_append_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contracts_obj = yaml.safe_load((repo_root / "configs" / "frozen_contracts.yaml").read_text(encoding="utf-8"))
    schema_obj = yaml.safe_load((repo_root / "configs" / "records_schema_extensions.yaml").read_text(encoding="utf-8"))

    artifact_schema = contracts_obj.get("artifact_schema")
    assert artifact_schema.get("append_only") is True
    artifact_contracts = artifact_schema.get("artifact_contracts")
    assert "lf_attestation_trace" in artifact_contracts
    assert "hf_attestation_trace" in artifact_contracts
    assert "lf_alignment_table" in artifact_contracts
    assert "lf_retain_breakdown" in artifact_contracts
    assert "geo_rescue_diagnostics" in artifact_contracts
    assert "lf_planner_risk_report" in artifact_contracts

    lf_allowed_fields = set(artifact_contracts["lf_attestation_trace"]["allowed_top_level_fields"])
    required_lf_fields = {
        "expected_bit_signs",
        "posterior_values",
        "posterior_signs",
        "posterior_margin_values",
        "agreement_indices",
        "mismatch_indices",
        "weakest_posterior_indices",
        "weakest_posterior_margins",
        "projected_lf_coeffs",
        "projected_lf_signs",
        "trajectory_feature_vector",
        "plan_digest",
        "lf_basis_digest",
        "projection_matrix_digest",
        "trajectory_feature_spec_digest",
        "projection_seed",
    }
    assert required_lf_fields <= lf_allowed_fields

    alignment_allowed_fields = set(artifact_contracts["lf_alignment_table"]["allowed_top_level_fields"])
    assert {
        "pre_injection_coeffs",
        "injected_template_coeffs",
        "post_injection_coeffs",
        "detect_side_coeffs",
        "embed_expected_bit_signs",
        "formal_expected_bit_signs",
        "embed_formal_expected_signs_match",
        "expected_signs_mismatch_reason",
        "signed_pre_alignment",
        "signed_detect_alignment",
        "strong_negative_pre_count",
        "post_still_negative_count",
        "post_crosses_target_halfspace_count",
        "detect_reverted_after_post_positive_count",
        "lf_alignment_table_digest",
    } <= alignment_allowed_fields

    retain_allowed_fields = set(artifact_contracts["lf_retain_breakdown"]["allowed_top_level_fields"])
    assert {
        "embed_seed",
        "detect_seed",
        "same_seed_as_embed_available",
        "same_seed_as_embed_value",
        "detect_protocol_classification",
        "image_conditioned_reconstruction_available",
        "image_conditioned_reconstruction_status",
        "detect_exact_timestep_coeffs_same_seed_control",
        "cross_seed_protocol_loss_count",
        "same_seed_residual_loss_count",
        "cross_seed_protocol_loss_ratio",
        "same_seed_residual_loss_ratio",
        "protocol_root_cause_classification",
        "control_protocol_segments",
        "control_protocol_summary",
        "selected_step_post_coeffs",
        "embed_edit_timestep_coeffs",
        "embed_terminal_step_coeffs",
        "detect_exact_timestep_coeffs",
        "breakdown_segments",
        "breakdown_summary",
        "lf_retain_breakdown_digest",
    } <= retain_allowed_fields

    geo_allowed_fields = set(artifact_contracts["geo_rescue_diagnostics"]["allowed_top_level_fields"])
    assert {
        "quality_score",
        "template_match_score",
        "geo_score",
        "geo_score_source",
        "geo_rescue_eligible",
        "geo_rescue_applied",
        "geo_not_used_reason",
        "quality_vs_template_ratio",
        "geo_scale_classification",
        "template_match_internal_threshold",
        "template_match_threshold_to_rescue_min_ratio",
        "template_score_scale_band",
        "rescue_gate_scale_classification",
        "positive_template_match_score_summary",
        "positive_quality_score_summary",
        "negative_template_match_score_summary",
        "negative_quality_score_summary",
        "positive_template_to_gate_max_ratio",
        "positive_template_to_internal_threshold_max_ratio",
        "geo_repair_direction_classification",
        "scale_control_scan_source",
        "scale_control_scan_glob",
        "scale_control_scanned_record_count",
        "scale_control_labelled_record_count",
        "geo_rescue_diagnostics_digest",
    } <= geo_allowed_fields

    planner_allowed_fields = set(artifact_contracts["lf_planner_risk_report"]["allowed_top_level_fields"])
    assert {
        "risk_classification",
        "host_baseline_ratio",
        "reconstruction_residual_ratio",
        "topk_energy_ratio",
        "plan_digest",
        "basis_digest",
        "embed_formal_expected_signs_match",
        "expected_signs_mismatch_reason",
        "primary_evidence",
        "per_dimension_summary",
        "high_confidence_mismatch_dimensions",
        "routing_pattern_summary",
        "host_baseline_risk_summary",
    } <= planner_allowed_fields

    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))
    schema_fields = {
        entry.get("path")
        for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    assert "expected_bit_signs" not in registry_fields
    assert "posterior_values" not in registry_fields
    assert "expected_bit_signs" not in schema_fields
    assert "posterior_values" not in schema_fields


def test_lf_attestation_trace_artifact_rejects_unallowlisted_top_level_field(tmp_run_root) -> None:
    from main.core import records_io
    from main.core.errors import RecordsWritePolicyError

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_trace_payload()
    payload["illegal_extra_field"] = "not allowed"
    output_path = artifacts_dir / "attestation" / "lf_attestation_trace.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        with pytest.raises(RecordsWritePolicyError) as exc_info:
            records_io.write_artifact_json(str(output_path), payload)

    assert "allowlist" in str(exc_info.value).lower()


def test_lf_attestation_trace_artifact_accepts_governed_field_set(tmp_run_root) -> None:
    from main.core import records_io

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_trace_payload()
    output_path = artifacts_dir / "attestation" / "lf_attestation_trace.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        records_io.write_artifact_json(str(output_path), payload)

    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["artifact_type"] == "lf_attestation_trace"
    assert written_payload["expected_bit_signs"] == [1, -1, 1]
    assert written_payload["_artifact_audit"]["writer"] == "records_io"


def test_lf_alignment_table_artifact_accepts_governed_field_set(tmp_run_root) -> None:
    from main.core import records_io

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_alignment_payload()
    output_path = artifacts_dir / "attestation" / "lf_alignment_table.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        records_io.write_artifact_json(str(output_path), payload)

    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["artifact_type"] == "lf_alignment_table"
    assert written_payload["post_crosses_target_halfspace_count"] == 2


def test_lf_planner_risk_report_artifact_accepts_governed_field_set(tmp_run_root) -> None:
    from main.core import records_io

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_planner_risk_report_payload()
    output_path = artifacts_dir / "planner" / "lf_planner_risk_report.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        records_io.write_artifact_json(str(output_path), payload)

    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["artifact_type"] == "lf_planner_risk_report"
    assert written_payload["risk_classification"] == "host_baseline_dominant"