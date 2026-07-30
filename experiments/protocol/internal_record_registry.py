"""Executable field registry for governed internal record collections.

This module is part of the experiment-execution package.  Documentation mirrors
these names for development-time governance, but runtime validation never reads
the documentation tree.
"""

from __future__ import annotations

from types import MappingProxyType

from experiments.protocol.internal_records import (
    INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION,
    INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
)


INTERNAL_RECORD_FIELD_REGISTRY_VERSION = "ceg_wm_internal_record_fields_v1"
INTERNAL_RECORD_FIELD_NAMES = frozenset(
    {
        "analysis_unit_identity",
        "attack_config_digest",
        "branch_score_trace",
        "candidate_config_digest",
        "canonical_score",
        "case_id",
        "combined_score",
        "control_identity",
        "coverage",
        "decision_reason",
        "decision_trace",
        "detection_key_public_digest",
        "detector_trace",
        "environment_digest",
        "evidence_record_ids",
        "exact_identity_objective",
        "exclusion_reason",
        "exclusion_rule_id",
        "execution_config_digest",
        "execution_status",
        "failure_class",
        "failure_reason",
        "gap",
        "gate_id",
        "gate_status",
        "generation_seed",
        "geometry_estimation_identity",
        "geometry_failure_reason",
        "geometry_operation_identity",
        "geometry_raw_metrics",
        "geometry_reliability_identity",
        "geometry_reliability_config_digest",
        "geometry_reliable",
        "geometry_trace",
        "geometry_transform",
        "geometry_triggered",
        "hf_score",
        "identity_margin",
        "image_lineage_digest",
        "inlier_ratio",
        "input_artifact_digest",
        "input_manifest_digest",
        "key_control_trace",
        "key_margin",
        "key_role",
        "lf_score",
        "log_scale",
        "maximum_record_attempts",
        "mean_residual",
        "method_code_revision",
        "method_config_digest",
        "metric_set_digest",
        "model_revision",
        "observation_score",
        "positive_source",
        "promotion_gate_assessments",
        "promotion_stop_gate_id",
        "prompt_digest",
        "protocol_digest",
        "protocol_id",
        "protocol_version",
        "provenance_trace",
        "raw_content_score",
        "raw_detector_config_digest",
        "raw_detector_identity",
        "raw_preprocessing_identity",
        "raw_threshold_identity",
        "record_attempt_index",
        "record_collection_schema_version",
        "record_id",
        "record_schema_version",
        "record_sequence_index",
        "records",
        "rectification_status",
        "rectified_content_score",
        "rectified_detector_config_digest",
        "rectified_detector_identity",
        "rectified_preprocessing_identity",
        "rectified_threshold_identity",
        "registered_key_family_digest",
        "registered_key_public_digest",
        "registered_objective",
        "residual_rotation_degrees",
        "resource_identity_digest",
        "retry_of_record_id",
        "routing_control",
        "routing_identity",
        "routing_mask_digest",
        "routing_observation_digest",
        "routing_trace",
        "run_id",
        "second_registered_objective",
        "source_cluster_id",
        "split",
        "split_manifest_digest",
        "stop_outcome",
        "tau",
        "tau_rescue",
        "threshold_trace",
        "translation_x",
        "translation_y",
        "uniqueness",
        "unit_id",
        "watermark_decision",
    }
)
INTERNAL_RECORD_SCHEMA_BINDINGS = MappingProxyType(
    {
        "record_collection_schema_version": (
            INTERNAL_VALIDATION_RECORD_COLLECTION_SCHEMA_VERSION
        ),
        "record_schema_version": INTERNAL_VALIDATION_RECORD_SCHEMA_VERSION,
    }
)
