"""
文件目的：验证正式分数字段与 matrix scope 修复。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from main.evaluation import experiment_matrix
from main.evaluation import metrics as eval_metrics
from main.evaluation import workflow_inputs


def test_canonical_score_aliases_are_preferred_over_legacy_detect_helper() -> None:
    """
    功能：验证正式内容链分数优先读取 canonical 字段且不消费 detect_lf_score。

    Validate that canonical content-chain score fields are preferred and that
    detect_lf_score is not consumed by the formal workflow input resolver.

    Args:
        None.

    Returns:
        None.
    """
    record = {
        "content_evidence_payload": {
            "status": "ok",
            "content_chain_score": 0.81,
            "score": 0.81,
            "content_score": 0.81,
            "lf_channel_score": 0.63,
            "lf_score": 0.63,
            "lf_correlation_score": 0.27,
            "detect_lf_score": 0.27,
            "score_parts": {
                "content_chain_score": 0.81,
                "content_score": 0.81,
            },
        }
    }

    assert workflow_inputs._resolve_content_score_source(record) == (
        0.81,
        "content_evidence_payload.content_chain_score",
    )
    assert eval_metrics._extract_score_value_for_metrics(record, "content_chain_score") == (0.81, None)
    assert eval_metrics._extract_score_value_for_metrics(record, "lf_channel_score") == (0.63, None)

    diagnostic_only_record = {
        "content_evidence_payload": {
            "status": "ok",
            "detect_lf_score": 0.91,
            "lf_correlation_score": 0.91,
        }
    }
    assert workflow_inputs._resolve_content_score_source(diagnostic_only_record) == (None, None)


def test_experiment_matrix_scope_and_system_final_metrics_use_real_terminal_fields(tmp_path: Path) -> None:
    """
    功能：验证 experiment_matrix 主作用域与 system_final 统计来自真实终态字段。

    Validate that experiment_matrix accepts the new primary scope and derives
    system-level metrics from final decision and event-attestation fields.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    matrix_cfg = {
        "primary_scope": "system_final",
        "primary_summary_basis_scope": "system_final",
        "auxiliary_scopes": ["content_chain", "lf_channel"],
        "auxiliary_scope_configs": {
            "content_chain": {"metric_name": "content_chain_score"},
            "lf_channel": {
                "metric_name": "lf_channel_score",
                "analysis_metric_name": "lf_channel_score",
            },
        },
        "models": ["sd3"],
        "seeds": [0],
        "attack_protocol_families": ["rotate"],
    }
    assert experiment_matrix._resolve_matrix_primary_scope(matrix_cfg) == "system_final"
    assert experiment_matrix._resolve_matrix_auxiliary_scopes(matrix_cfg, "system_final") == ["content_chain", "lf_channel"]
    assert experiment_matrix._resolve_matrix_primary_summary_basis_scope(matrix_cfg, "system_final") == "system_final"
    auxiliary_scope_configs = experiment_matrix._resolve_matrix_auxiliary_scope_configs(
        matrix_cfg,
        ["content_chain", "lf_channel"],
    )
    assert auxiliary_scope_configs["lf_channel"]["analysis_metric_name"] == "lf_channel_score"

    grid = experiment_matrix.build_experiment_grid(
        {
            "model_id": "sd3",
            "seed": 0,
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "experiment_matrix": matrix_cfg,
        }
    )
    assert len(grid) == 1
    assert "scalar_formal_scope" not in grid[0]
    assert "scalar_formal_score_name" not in grid[0]
    assert "formal_score_name" not in grid[0]
    assert "auxiliary_analysis_metric_name" not in grid[0]
    assert grid[0]["primary_driver_mode"] == "system_final_only"
    assert experiment_matrix._extract_auxiliary_analysis_metric_name_from_grid_item(grid[0]) == "lf_channel_score"

    score_record = {
        "content_evidence_payload": {
            "status": "ok",
            "lf_channel_score": 0.72,
            "lf_score": 0.72,
        }
    }
    assert experiment_matrix._resolve_auxiliary_analysis_score(score_record, "lf_channel_score") == (
        0.72,
        "content_evidence_payload.lf_channel_score",
    )
    diagnostic_only_record = {
        "content_evidence_payload": {
            "status": "ok",
            "detect_lf_score": 0.91,
            "lf_correlation_score": 0.91,
        }
    }
    assert experiment_matrix._resolve_auxiliary_analysis_score(diagnostic_only_record, "lf_channel_score") == (
        None,
        "content_evidence_payload.lf_channel_score_missing_or_nonfinite",
    )

    records_dir = tmp_path / "artifacts" / "evaluate_inputs" / "formal_evaluate_records"
    records_dir.mkdir(parents=True)
    positive_payload = {
        "label": True,
        "final_decision": {"is_watermarked": True},
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": True},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    negative_payload = {
        "label": False,
        "final_decision": {"is_watermarked": False},
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": False},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    (records_dir / "positive.json").write_text(json.dumps(positive_payload), encoding="utf-8")
    (records_dir / "negative.json").write_text(json.dumps(negative_payload), encoding="utf-8")

    system_final_metrics = experiment_matrix._build_system_final_metrics_for_run(tmp_path)
    assert system_final_metrics["scope"] == "system_final"
    assert system_final_metrics["system_tpr"] == 1.0
    assert system_final_metrics["system_fpr"] == 0.0
    assert system_final_metrics["final_decision_tpr"] == 1.0
    assert system_final_metrics["event_attestation_tpr"] == 1.0

    aggregate_report = experiment_matrix.build_aggregate_report(
        [
            {
                "grid_item_digest": "grid01",
                "status": "ok",
                "evaluation_scope": "system_final",
                "auxiliary_scopes": ["content_chain", "lf_channel"],
                "auxiliary_scope_configs": auxiliary_scope_configs,
                "scope_manifest": experiment_matrix._build_matrix_scope_manifest(
                    primary_scope="system_final",
                    primary_summary_basis_scope="system_final",
                    auxiliary_scopes=["content_chain", "lf_channel"],
                ),
                "primary_metric_name": "system_final_metrics",
                "primary_driver_mode": "system_final_only",
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": "system_final_metrics",
                "policy_path": "content_np_geo_rescue",
                "metrics": {
                    "system_final_metrics": system_final_metrics,
                    "auxiliary_scope_metrics": {
                        "content_chain": {"metric_name": "content_chain_score", "available": True},
                        "lf_channel": {"metric_name": "lf_channel_score", "available": True},
                    },
                },
            }
        ],
        grid_manifest={"grid_manifest_digest": "grid_manifest_01"},
    )
    assert aggregate_report["primary_evaluation_scope"] == "system_final"
    assert aggregate_report["primary_metric_name"] == "system_final_metrics"
    assert aggregate_report["primary_driver_mode"] == "system_final_only"
    assert aggregate_report["primary_status_source"] == "system_final_metrics"
    assert aggregate_report["primary_summary_basis_scope"] == "system_final"
    assert aggregate_report["primary_summary_basis_metric_name"] == "system_final_metrics"
    assert aggregate_report["scope_manifest"]["primary_summary_basis_metric_name"] == "system_final_metrics"
    assert aggregate_report["scope_manifest"]["auxiliary_scopes"] == ["content_chain", "lf_channel"]
    assert aggregate_report["metrics_matrix"][0]["primary_status_source"] == "system_final_metrics"
    assert "scalar_formal_scope" not in aggregate_report
    assert "scalar_formal_score_name" not in aggregate_report
    assert "scalar_formal_scope" not in aggregate_report["scope_manifest"]
    assert "scalar_calibration_scope" not in aggregate_report["scope_manifest"]
    assert aggregate_report["system_final_metrics_presence"]["ok_rows_with_system_final_metrics"] == 1


def test_schema_and_contracts_register_new_formal_fields() -> None:
    """
    功能：验证 schema、冻结契约与 policy semantics 已登记新的正式字段。

    Validate that schema, frozen contracts, and policy semantics register the
    new canonical formal score fields.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    schema_payload = yaml.safe_load((repo_root / "configs" / "records_schema_extensions.yaml").read_text(encoding="utf-8"))
    contract_payload = yaml.safe_load((repo_root / "configs" / "frozen_contracts.yaml").read_text(encoding="utf-8"))
    policy_payload = yaml.safe_load((repo_root / "configs" / "policy_path_semantics.yaml").read_text(encoding="utf-8"))

    schema_paths = {item["path"] for item in schema_payload["fields"]}
    contract_paths = set(contract_payload["records_schema"]["field_paths_registry"])
    content_minimal = policy_payload["field_catalog"]["catalogs"]["content_minimal"]
    artifact_contracts = contract_payload["artifact_schema"]["artifact_contracts"]

    required_paths = {
        "content_evidence.content_chain_score",
        "content_evidence.lf_channel_score",
        "content_evidence.lf_correlation_score",
        "attestation.image_evidence_result.lf_channel_score",
        "attestation.final_event_attested_decision.lf_attestation_score",
    }
    assert required_paths.issubset(schema_paths)
    assert required_paths.issubset(contract_paths)
    assert "content_evidence.content_chain_score" in content_minimal
    assert "experiment_matrix_aggregate_report" in artifact_contracts
    assert "experiment_matrix_grid_summary" in artifact_contracts
    assert "primary_evaluation_scope" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_summary_basis_scope" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_summary_basis_metric_name" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_driver_mode" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_status_source" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scope_manifest" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_summary_basis_scope" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_summary_basis_metric_name" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_driver_mode" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_status_source" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "scalar_formal_scope" not in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scalar_formal_score_name" not in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scalar_formal_scope" not in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "scalar_formal_score_name" not in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]