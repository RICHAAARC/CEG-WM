"""
文件目的：验证正式分数字段与 matrix scope 修复。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from main.evaluation import experiment_matrix
from main.evaluation import metrics as eval_metrics
from main.evaluation import workflow_inputs
from main.watermarking.detect import orchestrator as detect_orchestrator


FORMAL_PRIMARY_METRIC_NAME = "formal_final_decision_metrics"
FORMAL_PRIMARY_DRIVER_MODE = "formal_final_decision_only"
DERIVED_SYSTEM_UNION_METRIC_NAME = "derived_system_union_metrics"
SYSTEM_FINAL_ALIAS_SEMANTICS = "deprecated_alias_of_derived_system_union_metrics"


def _load_stage_03_module() -> object:
    """
    功能：按文件路径加载 stage 03 脚本模块。

    Load the stage 03 script module from its filesystem path.

    Args:
        None.

    Returns:
        Loaded module object.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "03_Experiment_Matrix_Full.py"
    spec = importlib.util.spec_from_file_location("stage_03_experiment_matrix_full", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_calibration_accepts_strict_clean_negative_formal_null_scores() -> None:
    """
    功能：验证 calibration null 分布会接纳 strict clean_negative 的 formal null 0.0 分数。

    Validate that calibration null-distribution loading accepts the strict
    clean-negative formal-null score mapping.

    Args:
        None.

    Returns:
        None.
    """
    strict_clean_negative_record = {
        "label": False,
        "sample_role": "clean_negative",
        "plan_digest": "runtime-observed-plan",
        "basis_digest": "runtime-observed-basis",
        "plan_input_digest": "runtime-observed-plan-input",
        "plan_input_schema_version": "v2",
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner",
            "impl_version": "v2",
        },
        "subspace_plan": {
            "planner_input_digest": "runtime-observed-plan-input",
            "planner_impl_identity": {
                "impl_id": "subspace_planner",
                "impl_version": "v2",
            },
            "rank": 128,
        },
        "content_evidence_payload": {
            "status": "absent",
            "content_chain_score": None,
            "score": None,
            "content_score": None,
            "content_failure_reason": "detector_no_plan_expected",
        },
        "attestation": {
            "authenticity_result": {
                "status": "absent",
                "bundle_status": "absent",
                "statement_status": "absent",
            },
        },
    }
    positive_record = {
        "label": True,
        "sample_role": "positive_source",
        "content_evidence_payload": {
            "status": "ok",
            "content_chain_score": 0.81,
            "score": 0.81,
            "content_score": 0.81,
        },
    }

    assert detect_orchestrator._extract_score_for_stats(
        strict_clean_negative_record,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
    ) == 0.0

    scores, strata = detect_orchestrator.load_scores_for_calibration(
        [positive_record, strict_clean_negative_record],
        cfg={"calibration": {}},
        score_name=eval_metrics.CONTENT_CHAIN_SCORE_NAME,
    )

    assert scores == [0.0]
    assert strata["sampling_policy"]["records_with_explicit_label"] is True
    assert strata["sampling_policy"]["n_rejected_label_positive"] == 1
    assert strata["sampling_policy"]["n_selected_null"] == 1

    assert eval_metrics._extract_score_value_for_metrics(
        strict_clean_negative_record,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
    ) == (0.0, None)

    overall_metrics, _ = eval_metrics.compute_overall_metrics(
        [positive_record, strict_clean_negative_record],
        threshold_value=0.5,
        score_name=eval_metrics.CONTENT_CHAIN_SCORE_NAME,
    )
    assert overall_metrics["n_pos"] == 1
    assert overall_metrics["n_neg"] == 1
    assert overall_metrics["reject_rate_by_reason"]["status_not_ok"] == 0.0


def test_formal_final_decision_overlay_uses_np_threshold_and_strict_clean_null() -> None:
    """
    功能：验证 PW02 evaluate-side overlay 会复用 strict clean null 0.0 与 NP 阈值生成正式终态。 

    Validate that the PW02 evaluate-side overlay reuses the strict clean-null
    0.0 mapping and the bound NP threshold to build a decided terminal field.

    Args:
        None.

    Returns:
        None.
    """
    thresholds_artifact = {
        "threshold_id": "content_chain_score_np_0p01",
        "score_name": eval_metrics.CONTENT_CHAIN_SCORE_NAME,
        "target_fpr": 0.01,
        "threshold_value": float.fromhex("0x0.0000000000001p-1022"),
        "threshold_key_used": "0p01",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": 0.0,
    }
    strict_clean_negative_record = {
        "sample_role": "clean_negative",
        "content_evidence_payload": {
            "status": "absent",
            "content_chain_score": None,
            "score": None,
            "content_score": None,
            "content_failure_reason": "detector_no_plan_expected",
        },
        "attestation": {
            "authenticity_result": {
                "status": "absent",
                "bundle_status": "absent",
                "statement_status": "absent",
            },
        },
    }
    positive_record = {
        "sample_role": "positive_source",
        "content_evidence_payload": {
            "status": "ok",
            "content_chain_score": 0.81,
            "score": 0.81,
            "content_score": 0.81,
        },
    }

    negative_overlay = workflow_inputs.build_formal_final_decision_overlay(
        strict_clean_negative_record,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
        thresholds_artifact,
    )
    positive_overlay = workflow_inputs.build_formal_final_decision_overlay(
        positive_record,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
        thresholds_artifact,
    )

    assert negative_overlay == {
        "decision_origin": "formal_threshold_overlay",
        "decision_operator": "score_greater_equal_threshold_value",
        "decision_status": "decided",
        "is_watermarked": False,
        "score_name": eval_metrics.CONTENT_CHAIN_SCORE_NAME,
        "score_source": "strict_clean_negative_formal_null",
        "score_value": 0.0,
        "selected_order_stat_score": 0.0,
        "target_fpr": 0.01,
        "threshold_key_used": "0p01",
        "threshold_source": "np_canonical",
        "used_threshold_id": "content_chain_score_np_0p01",
        "used_threshold_value": float.fromhex("0x0.0000000000001p-1022"),
    }
    assert positive_overlay["decision_status"] == "decided"
    assert positive_overlay["is_watermarked"] is True
    assert positive_overlay["score_source"] == "content_evidence_payload.content_chain_score"
    assert positive_overlay["threshold_source"] == "np_canonical"


def test_formal_final_decision_metrics_prefer_overlay_and_keep_derived_metrics_on_source_fields(tmp_path: Path) -> None:
    """
    功能：验证 formal metrics 优先读取 evaluate-side overlay，而 derived metrics 仍停留在 source terminal fields。 

    Validate that formal metrics prefer the evaluate-side overlay while the
    derived system metrics continue to consume source terminal fields.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    records_dir = tmp_path / "artifacts" / "evaluate_inputs" / "formal_evaluate_records"
    records_dir.mkdir(parents=True)
    positive_payload = {
        "label": True,
        "final_decision": {
            "decision_status": "abstain",
            "is_watermarked": None,
            "threshold_source": "observation_only_pre_calibration",
        },
        "formal_final_decision": {
            "decision_status": "decided",
            "is_watermarked": True,
            "threshold_source": "np_canonical",
            "score_name": eval_metrics.CONTENT_CHAIN_SCORE_NAME,
            "score_value": 0.81,
            "decision_origin": "formal_threshold_overlay",
        },
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": True},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    negative_payload = {
        "label": False,
        "final_decision": {
            "decision_status": "error",
            "is_watermarked": None,
            "threshold_source": None,
        },
        "formal_final_decision": {
            "decision_status": "decided",
            "is_watermarked": False,
            "threshold_source": "np_canonical",
            "score_name": eval_metrics.CONTENT_CHAIN_SCORE_NAME,
            "score_value": 0.0,
            "decision_origin": "formal_threshold_overlay",
        },
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": False},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    (records_dir / "positive.json").write_text(json.dumps(positive_payload), encoding="utf-8")
    (records_dir / "negative.json").write_text(json.dumps(negative_payload), encoding="utf-8")

    formal_final_decision_metrics = experiment_matrix._build_formal_final_decision_metrics_for_run(tmp_path)
    derived_system_union_metrics = experiment_matrix._build_system_final_metrics_for_run(tmp_path)

    assert formal_final_decision_metrics["final_decision_tpr"] == 1.0
    assert formal_final_decision_metrics["final_decision_fpr"] == 0.0
    assert formal_final_decision_metrics["final_decision_status_counts"] == {"decided": 2}
    assert derived_system_union_metrics["system_tpr"] == 1.0
    assert derived_system_union_metrics["system_fpr"] == 0.0
    assert derived_system_union_metrics["final_decision_tpr"] == 0.0
    assert derived_system_union_metrics["final_decision_fpr"] == 0.0


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
        "enable_auxiliary_analysis_runtime": False,
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
    assert grid[0]["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert grid[0]["primary_driver_mode"] == FORMAL_PRIMARY_DRIVER_MODE
    assert grid[0]["enable_auxiliary_analysis_runtime"] is False
    assert experiment_matrix._extract_auxiliary_analysis_metric_name_from_grid_item(grid[0]) == "lf_channel_score"
    assert experiment_matrix._extract_auxiliary_analysis_runtime_enabled_from_grid_item(grid[0]) is False

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
        "final_decision": {"decision_status": "decided", "is_watermarked": True},
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": True},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    negative_payload = {
        "label": False,
        "final_decision": {"decision_status": "decided", "is_watermarked": False},
        "attestation": {
            "final_event_attested_decision": {"is_event_attested": False},
            "image_evidence_result": {"geo_rescue_applied": False},
        },
    }
    (records_dir / "positive.json").write_text(json.dumps(positive_payload), encoding="utf-8")
    (records_dir / "negative.json").write_text(json.dumps(negative_payload), encoding="utf-8")

    formal_final_decision_metrics = experiment_matrix._build_formal_final_decision_metrics_for_run(tmp_path)
    derived_system_union_metrics = experiment_matrix._build_system_final_metrics_for_run(tmp_path)
    assert formal_final_decision_metrics["scope"] == "formal_final_decision"
    assert formal_final_decision_metrics["final_decision_tpr"] == 1.0
    assert formal_final_decision_metrics["final_decision_fpr"] == 0.0
    assert formal_final_decision_metrics["final_decision_status_counts"] == {"decided": 2}
    assert derived_system_union_metrics["scope"] == "system_final"
    assert derived_system_union_metrics["system_tpr"] == 1.0
    assert derived_system_union_metrics["system_fpr"] == 0.0
    assert derived_system_union_metrics["final_decision_tpr"] == 1.0
    assert derived_system_union_metrics["event_attestation_tpr"] == 1.0

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
                "primary_status_source": "system_final_metrics",
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": "system_final_metrics",
                "auxiliary_analysis_runtime_executed": False,
                "policy_path": "content_np_geo_rescue",
                "metrics": {
                    FORMAL_PRIMARY_METRIC_NAME: formal_final_decision_metrics,
                    DERIVED_SYSTEM_UNION_METRIC_NAME: derived_system_union_metrics,
                    "system_final_metrics": derived_system_union_metrics,
                    "system_final_metrics_semantics": SYSTEM_FINAL_ALIAS_SEMANTICS,
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
    assert aggregate_report["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["primary_driver_mode"] == FORMAL_PRIMARY_DRIVER_MODE
    assert aggregate_report["primary_status_source"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["primary_summary_basis_scope"] == "system_final"
    assert aggregate_report["primary_summary_basis_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["auxiliary_analysis_runtime_executed"] is False
    assert aggregate_report["scope_manifest"]["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["scope_manifest"]["primary_summary_basis_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["scope_manifest"]["auxiliary_scopes"] == ["content_chain", "lf_channel"]
    assert aggregate_report["metrics_matrix"][0]["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["metrics_matrix"][0]["primary_status_source"] == FORMAL_PRIMARY_METRIC_NAME
    assert aggregate_report["metrics_matrix"][0]["auxiliary_analysis_runtime_executed"] is False
    assert aggregate_report["metrics_matrix"][0][FORMAL_PRIMARY_METRIC_NAME] == formal_final_decision_metrics
    assert aggregate_report["metrics_matrix"][0][DERIVED_SYSTEM_UNION_METRIC_NAME] == derived_system_union_metrics
    assert aggregate_report["metrics_matrix"][0]["system_final_metrics"] == derived_system_union_metrics
    assert aggregate_report["metrics_matrix"][0]["system_final_metrics_semantics"] == SYSTEM_FINAL_ALIAS_SEMANTICS
    assert "scalar_formal_scope" not in aggregate_report
    assert "scalar_formal_score_name" not in aggregate_report
    assert "scalar_formal_scope" not in aggregate_report["scope_manifest"]
    assert "scalar_calibration_scope" not in aggregate_report["scope_manifest"]
    assert aggregate_report["formal_final_decision_metrics_presence"]["ok_rows_with_formal_final_decision_metrics"] == 1
    assert aggregate_report["derived_system_union_metrics_presence"]["ok_rows_with_derived_system_union_metrics"] == 1
    assert aggregate_report["system_final_metrics_presence"]["ok_rows_with_system_final_metrics"] == 1
    assert aggregate_report["system_final_metrics_semantics"] == SYSTEM_FINAL_ALIAS_SEMANTICS


def test_run_experiment_grid_grid_summary_keeps_formal_primary_and_derived_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：验证 run_experiment_grid 的 grid_summary 不再把 system_final alias 当作 primary path。

    Validate that run_experiment_grid normalizes the grid summary so that
    formal_final_decision_metrics remains the primary path while
    system_final_metrics is only a derived alias.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    grid = [
        {
            "grid_index": 0,
            "grid_item_digest": "grid01",
            "model_id": "sd3",
            "seed": 0,
            "batch_root": (tmp_path / "batch_root").as_posix(),
            "config_path": "configs/default.yaml",
        }
    ]
    formal_final_decision_metrics = {
        "scope": "formal_final_decision",
        "n_total": 2,
        "n_positive": 1,
        "n_negative": 1,
        "final_decision_tpr": 1.0,
        "final_decision_fpr": 0.0,
    }
    derived_system_union_metrics = {
        "scope": "system_final",
        "n_total": 2,
        "n_positive": 1,
        "n_negative": 1,
        "system_tpr": 1.0,
        "system_fpr": 0.0,
        "final_decision_tpr": 1.0,
        "event_attestation_tpr": 1.0,
    }

    monkeypatch.setattr(experiment_matrix, "_run_neg_embed_detect_for_cache", lambda **_: None)
    monkeypatch.setattr(experiment_matrix, "_run_global_calibrate", lambda **_: None)
    monkeypatch.setattr(
        experiment_matrix,
        "_build_stage_03_gpu_memory_observability",
        lambda **_: {
            "gpu_memory_summary": {"status": "absent"},
            "gpu_memory_profile_breakdown": {"artifact_version": "stub"},
        },
    )
    monkeypatch.setattr(
        experiment_matrix,
        "_write_grid_artifacts",
        lambda *args, **kwargs: {
            "aggregate_report_path": "aggregate_report.json",
            "grid_summary_path": "grid_summary.json",
            "grid_manifest_path": "grid_manifest.json",
        },
    )
    monkeypatch.setattr(experiment_matrix, "_extract_anchors_from_results", lambda *_args, **_kwargs: {})

    def _fake_run_single_experiment(_grid_item: dict) -> dict:
        return {
            "grid_item_digest": "grid01",
            "status": "ok",
            "evaluation_scope": "system_final",
            "auxiliary_scopes": ["content_chain", "lf_channel"],
            "auxiliary_scope_configs": {
                "content_chain": {"metric_name": "content_chain_score"},
                "lf_channel": {
                    "metric_name": "lf_channel_score",
                    "analysis_metric_name": "lf_channel_score",
                },
            },
            "scope_manifest": experiment_matrix._build_matrix_scope_manifest(
                primary_scope="system_final",
                primary_summary_basis_scope="system_final",
                auxiliary_scopes=["content_chain", "lf_channel"],
            ),
            "primary_metric_name": "system_final_metrics",
            "primary_driver_mode": "system_final_only",
            "primary_status_source": "system_final_metrics",
            "primary_summary_basis_scope": "system_final",
            "primary_summary_basis_metric_name": "system_final_metrics",
            "auxiliary_analysis_runtime_executed": False,
            "cfg_digest": "cfg01",
            "plan_digest": "plan01",
            "thresholds_digest": "thr01",
            "threshold_metadata_digest": "meta01",
            "ablation_digest": "abl01",
            "attack_protocol_digest": "attack01",
            "impl_digest": "impl01",
            "fusion_rule_version": "v1",
            "attack_protocol_version": "attack-v1",
            "run_root": (tmp_path / "runs" / "grid01").as_posix(),
            "policy_path": "content_np_geo_rescue",
            "metrics": {
                FORMAL_PRIMARY_METRIC_NAME: formal_final_decision_metrics,
                DERIVED_SYSTEM_UNION_METRIC_NAME: derived_system_union_metrics,
                "system_final_metrics": derived_system_union_metrics,
                "system_final_metrics_semantics": SYSTEM_FINAL_ALIAS_SEMANTICS,
                "auxiliary_scope_metrics": {},
            },
            "auxiliary_analysis": {},
        }

    monkeypatch.setattr(experiment_matrix, "run_single_experiment", _fake_run_single_experiment)

    grid_summary = experiment_matrix.run_experiment_grid(grid, strict=True)

    assert grid_summary["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert grid_summary["primary_driver_mode"] == FORMAL_PRIMARY_DRIVER_MODE
    assert grid_summary["primary_status_source"] == FORMAL_PRIMARY_METRIC_NAME
    assert grid_summary["primary_summary_basis_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert grid_summary["formal_final_decision_metrics_presence"]["ok_rows_with_formal_final_decision_metrics"] == 1
    assert grid_summary["derived_system_union_metrics_presence"]["ok_rows_with_derived_system_union_metrics"] == 1
    assert grid_summary["system_final_metrics_presence"]["ok_rows_with_system_final_metrics"] == 1
    assert grid_summary["system_final_metrics_semantics"] == SYSTEM_FINAL_ALIAS_SEMANTICS
    assert grid_summary["aggregate_report"]["primary_metric_name"] == FORMAL_PRIMARY_METRIC_NAME
    assert grid_summary["aggregate_report"]["metrics_matrix"][0]["primary_status_source"] == FORMAL_PRIMARY_METRIC_NAME
    assert (
        grid_summary["aggregate_report"]["metrics_matrix"][0][DERIVED_SYSTEM_UNION_METRIC_NAME]
        == derived_system_union_metrics
    )
    assert grid_summary["aggregate_report"]["metrics_matrix"][0]["system_final_metrics"] == derived_system_union_metrics


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
    schema_descriptions = {item["path"]: item["description"] for item in schema_payload["fields"]}
    contract_paths = set(contract_payload["records_schema"]["field_paths_registry"])
    content_minimal = policy_payload["field_catalog"]["catalogs"]["content_minimal"]
    artifact_contracts = contract_payload["artifact_schema"]["artifact_contracts"]
    policy_rules = {
        item["field_path"]: item["rule"]
        for item in policy_payload["field_catalog"]["diagnostic_only_field_constraints"]
    }

    required_paths = {
        "content_evidence.content_chain_score",
        "content_evidence.lf_channel_score",
        "content_evidence.lf_correlation_score",
        "content_evidence.hf_raw_energy",
        "content_evidence.hf_content_score",
        "attestation.image_evidence_result.lf_channel_score",
        "attestation.final_event_attested_decision.lf_attestation_score",
        "experiment_matrix.auxiliary_analysis_runtime_executed",
    }
    recommended_paths = policy_payload["field_catalog"]["catalogs"]["recommended"]
    assert required_paths.issubset(schema_paths)
    assert required_paths.issubset(contract_paths)
    assert "content_evidence.content_chain_score" in content_minimal
    assert "content_evidence.hf_raw_energy" in recommended_paths
    assert "content_evidence.hf_content_score" in recommended_paths
    assert "experiment_matrix_aggregate_report" in artifact_contracts
    assert "experiment_matrix_grid_summary" in artifact_contracts
    assert "primary_evaluation_scope" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_summary_basis_scope" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_summary_basis_metric_name" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_driver_mode" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "primary_status_source" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "auxiliary_analysis_runtime_executed" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "gpu_memory_summary" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "formal_final_decision_metrics_presence" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "derived_system_union_metrics_presence" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "system_final_metrics_semantics" in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scope_manifest" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_summary_basis_scope" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_summary_basis_metric_name" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_driver_mode" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "primary_status_source" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "auxiliary_analysis_runtime_executed" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "gpu_memory_summary" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "formal_final_decision_metrics_presence" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "derived_system_union_metrics_presence" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "system_final_metrics_semantics" in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "scalar_formal_scope" not in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scalar_formal_score_name" not in artifact_contracts["experiment_matrix_aggregate_report"]["allowed_top_level_fields"]
    assert "scalar_formal_scope" not in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "scalar_formal_score_name" not in artifact_contracts["experiment_matrix_grid_summary"]["allowed_top_level_fields"]
    assert "planner-conditioned control negative path" in schema_descriptions["negative_branch_source_attestation_provenance"]
    assert "detect runtime" in schema_descriptions["negative_branch_source_attestation_provenance"]
    assert "strict clean_negative" in schema_descriptions["negative_branch_source_attestation_provenance"]
    assert "detect formal attestation extractor" not in schema_descriptions["negative_branch_source_attestation_provenance"]
    assert "authentic signed bundle verification" in schema_descriptions["negative_branch_source_attestation_provenance.statement"]
    assert "strict clean_negative" in schema_descriptions["negative_branch_source_attestation_provenance.statement"]
    assert "planner-conditioned control negative path" in policy_rules["negative_branch_source_attestation_provenance"]
    assert "detect runtime" in policy_rules["negative_branch_source_attestation_provenance"]
    assert "runtime bindings" in policy_rules["negative_branch_source_attestation_provenance"]
    assert "authentic signed bundle verification" in policy_rules["negative_branch_source_attestation_provenance.statement"]
    assert "strict clean_negative formal input" in policy_rules["negative_branch_source_attestation_provenance.statement"]
    assert "detect formal attestation extractor" not in policy_rules["negative_branch_source_attestation_provenance.statement"]


def test_formal_stage_03_skips_auxiliary_runtime_by_default(tmp_path: Path, monkeypatch) -> None:
    """
    功能：验证 formal stage 03 默认不执行 auxiliary runtime。

    Validate that formal stage 03 skips the auxiliary runtime by default.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    matrix_cfg = {
        "primary_scope": "system_final",
        "primary_summary_basis_scope": "system_final",
        "enable_auxiliary_analysis_runtime": False,
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
    grid_item = experiment_matrix.build_experiment_grid(
        {
            "model_id": "sd3",
            "seed": 0,
            "attack_protocol_path": "configs/attack_protocol.yaml",
            "experiment_matrix": matrix_cfg,
        }
    )[0]

    auxiliary_runtime_calls = []

    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", lambda *args, **kwargs: {})
    monkeypatch.setattr(experiment_matrix, "_run_stage_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_matrix,
        "_assert_min_valid_content_scores_after_detect",
        lambda *args, **kwargs: {"gate_relaxed": False, "reason": "ok", "sample_counts": {"valid": 1}},
    )
    monkeypatch.setattr(
        experiment_matrix,
        "_prepare_system_final_labelled_detect_records_glob_for_matrix",
        lambda *args, **kwargs: str(tmp_path / "labelled" / "*.json"),
    )

    def _fake_run_auxiliary_analysis_sequence(**kwargs):
        auxiliary_runtime_calls.append(kwargs)
        return {
            "driver_role": "auxiliary_only",
            "metric_name": "lf_channel_score",
            "status": "ok",
            "failure_reason": "ok",
            "shared_thresholds_used": False,
            "pair_free_evaluate_used": False,
            "auxiliary_analysis_runtime_executed": True,
        }

    monkeypatch.setattr(experiment_matrix, "_run_auxiliary_analysis_sequence", _fake_run_auxiliary_analysis_sequence)

    stage_result = experiment_matrix._run_stage_sequence(grid_item, tmp_path / "formal_stage03")

    assert auxiliary_runtime_calls == []
    assert stage_result["auxiliary_analysis"]["status"] == "skipped"
    assert stage_result["auxiliary_analysis"]["failure_reason"] == "auxiliary_analysis_not_requested"
    assert stage_result["auxiliary_analysis"]["auxiliary_analysis_runtime_executed"] is False


def test_default_stage_03_matrix_keeps_16_items() -> None:
    """
    功能：验证默认 stage 03 配置仍保持 16 个 matrix items。

    Validate that the default stage-03 configuration still expands to 16
    experiment-matrix items.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config_payload = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    grid = experiment_matrix.build_experiment_grid(config_payload)

    assert config_payload["experiment_matrix"]["primary_scope"] == "system_final"
    assert config_payload["experiment_matrix"]["primary_summary_basis_scope"] == "system_final"
    assert config_payload["experiment_matrix"]["seeds"] == [0, 1]
    assert len(grid) == 16


def test_stage_03_script_syncs_auxiliary_runtime_evidence_to_workflow_summary_and_stage_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：验证 stage 03 脚本层会把 auxiliary runtime 证据同步到 workflow_summary 与 stage_manifest。

    Validate that the stage 03 script synchronizes auxiliary runtime evidence
    into workflow_summary and stage_manifest.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_03_module = _load_stage_03_module()

    drive_project_root = tmp_path / "drive"
    drive_project_root.mkdir(parents=True)
    source_package_path = tmp_path / "source_stage01.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "default.yaml"
    config_path.write_text("experiment_matrix: {}\n", encoding="utf-8")
    stage_roots = {
        "run_root": drive_project_root / "runs" / "stage03_sync_test",
        "log_root": drive_project_root / "logs" / "stage03_sync_test",
        "runtime_state_root": drive_project_root / "runtime_state" / "stage03_sync_test",
        "export_root": drive_project_root / "exports" / "stage03_sync_test",
    }

    def _fake_prepare_source_package(_source_package_path, runtime_state_root):
        extracted_root = runtime_state_root / "source_extracted"
        (extracted_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
        (extracted_root / "runtime_metadata").mkdir(parents=True, exist_ok=True)
        stage_03_module.write_json_atomic(
            extracted_root / "artifacts" / "stage_manifest.json",
            {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
        )
        stage_03_module.write_json_atomic(
            extracted_root / "artifacts" / "package_manifest.json",
            {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
        )
        stage_03_module.write_json_atomic(
            extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
            {"thresholds_digest": "thr01"},
        )
        stage_03_module.write_json_atomic(
            extracted_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json",
            {"threshold_metadata_digest": "meta01"},
        )
        (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text("experiment_matrix: {}\n", encoding="utf-8")
        return {
            "stage_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
            "package_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
            "extracted_root": extracted_root,
            "source_package_path": source_package_path,
            "source_package_sha256": "sha256_stage01",
            "package_manifest_digest": "digest_stage01",
        }

    def _fake_run_command_with_logs(**kwargs):
        run_root = stage_roots["run_root"]
        artifacts_dir = run_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        gpu_memory_summary = {
            "status": "ok",
            "reason": "ok",
            "scanned_run_count": 2,
            "profiled_run_count": 2,
            "runs_without_profiles_count": 0,
            "experiment_item_run_count": 1,
            "neg_cache_run_count": 1,
            "entry_count": 4,
            "ok_profile_count": 4,
            "absent_profile_count": 0,
            "failed_profile_count": 0,
            "status_counts": {"ok": 4},
            "profile_role_counts": {
                "preview_generation": 1,
                "embed_watermarked_inference": 1,
                "statement_only_runtime_capture": 1,
                "detect_main_inference": 1,
            },
            "phase_label_counts": {
                "preview_generation": 1,
                "embed_watermarked_inference": 1,
                "statement_only_runtime_capture": 1,
                "detect_main_inference": 1,
            },
            "device_counts": {"cuda": 4},
            "peak_memory_allocated_bytes_max": 2048,
            "peak_memory_reserved_bytes_max": 3072,
            "peak_memory_allocated_mib_max": round(2048 / (1024.0 * 1024.0), 6),
            "peak_memory_reserved_mib_max": round(3072 / (1024.0 * 1024.0), 6),
            "peak_memory_allocated_max_entry": {"profile_role": "detect_main_inference"},
            "peak_memory_reserved_max_entry": {"profile_role": "detect_main_inference"},
        }
        stage_03_module.write_json_atomic(
            artifacts_dir / "grid_summary.json",
            {
                "primary_evaluation_scope": "system_final",
                "primary_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                "primary_driver_mode": FORMAL_PRIMARY_DRIVER_MODE,
                "primary_status_source": FORMAL_PRIMARY_METRIC_NAME,
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                "auxiliary_scopes": ["content_chain", "lf_channel"],
                "auxiliary_analysis_runtime_executed": False,
                "gpu_memory_summary": gpu_memory_summary,
                "scope_manifest": {
                    "primary_scope": "system_final",
                    "primary_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                    "primary_summary_basis_scope": "system_final",
                    "primary_summary_basis_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                    "auxiliary_scopes": ["content_chain", "lf_channel"],
                },
                "formal_final_decision_metrics_presence": {
                    "rows_with_formal_final_decision_metrics": 1,
                    "ok_rows_with_formal_final_decision_metrics": 1,
                    "rows_total": 1,
                },
                "derived_system_union_metrics_presence": {
                    "rows_with_derived_system_union_metrics": 1,
                    "ok_rows_with_derived_system_union_metrics": 1,
                    "rows_total": 1,
                },
                "system_final_metrics_presence": {
                    "rows_with_system_final_metrics": 1,
                    "ok_rows_with_system_final_metrics": 1,
                    "rows_total": 1,
                },
                "system_final_metrics_semantics": SYSTEM_FINAL_ALIAS_SEMANTICS,
            },
        )
        stage_03_module.write_json_atomic(artifacts_dir / "grid_manifest.json", {"grid_manifest_digest": "grid01"})
        stage_03_module.write_json_atomic(
            artifacts_dir / "aggregate_report.json",
            {
                "primary_evaluation_scope": "system_final",
                "primary_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                "primary_driver_mode": FORMAL_PRIMARY_DRIVER_MODE,
                "primary_status_source": FORMAL_PRIMARY_METRIC_NAME,
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                "auxiliary_scopes": ["content_chain", "lf_channel"],
                "auxiliary_analysis_runtime_executed": False,
                "gpu_memory_summary": gpu_memory_summary,
                "scope_manifest": {
                    "primary_scope": "system_final",
                    "primary_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                    "primary_summary_basis_scope": "system_final",
                    "primary_summary_basis_metric_name": FORMAL_PRIMARY_METRIC_NAME,
                    "auxiliary_scopes": ["content_chain", "lf_channel"],
                },
                "formal_final_decision_metrics_presence": {
                    "rows_with_formal_final_decision_metrics": 1,
                    "ok_rows_with_formal_final_decision_metrics": 1,
                    "rows_total": 1,
                },
                "derived_system_union_metrics_presence": {
                    "rows_with_derived_system_union_metrics": 1,
                    "ok_rows_with_derived_system_union_metrics": 1,
                    "rows_total": 1,
                },
                "system_final_metrics_presence": {
                    "rows_with_system_final_metrics": 1,
                    "ok_rows_with_system_final_metrics": 1,
                    "rows_total": 1,
                },
                "system_final_metrics_semantics": SYSTEM_FINAL_ALIAS_SEMANTICS,
            },
        )
        stage_03_module.write_json_atomic(
            artifacts_dir / "gpu_memory_profile_breakdown.json",
            {
                "artifact_version": "stage_03_gpu_memory_profile_breakdown_v1",
                "gpu_memory_summary": gpu_memory_summary,
                "runs": [{"run_kind": "experiment_item", "run_root": (run_root / "experiments" / "item_0000").as_posix(), "profile_count": 4}],
                "entries": [{"profile_role": "detect_main_inference", "phase_label": "detect_main_inference", "status": "ok"}],
            },
        )
        return {"return_code": 0}

    bootstrap_calls = []

    def _fake_ensure_attestation_env_bootstrap(
        cfg_obj: dict,
        drive_root: Path,
        *,
        allow_generate: bool,
        allow_missing: bool = False,
    ) -> dict:
        bootstrap_calls.append(
            {
                "drive_project_root": drive_root.as_posix(),
                "allow_generate": allow_generate,
                "allow_missing": allow_missing,
                "attestation_enabled": bool((cfg_obj.get("attestation") or {}).get("enabled", False)) if isinstance(cfg_obj, dict) else False,
            }
        )
        return {"status": "disabled", "required_env_vars": [], "missing_env_vars": []}

    monkeypatch.setattr(stage_03_module, "prepare_source_package", _fake_prepare_source_package)
    monkeypatch.setattr(stage_03_module, "resolve_stage_roots", lambda *args, **kwargs: stage_roots)
    monkeypatch.setattr(stage_03_module, "load_yaml_mapping", lambda path: {"experiment_matrix": {}})
    monkeypatch.setattr(stage_03_module, "detect_stage_03_preflight", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(stage_03_module, "run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(stage_03_module, "collect_git_summary", lambda path: {})
    monkeypatch.setattr(stage_03_module, "collect_python_summary", lambda: {})
    monkeypatch.setattr(stage_03_module, "collect_cuda_summary", lambda: {})
    monkeypatch.setattr(stage_03_module, "collect_attestation_env_summary", lambda cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_model_summary", lambda cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_weight_summary", lambda repo_root, cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_file_index", lambda root, mapping: {})
    monkeypatch.setattr(
        stage_03_module,
        "ensure_attestation_env_bootstrap",
        _fake_ensure_attestation_env_bootstrap,
    )
    monkeypatch.setattr(stage_03_module, "finalize_stage_package", lambda **kwargs: {"package_path": "package.zip", "package_sha256": "sha256_pkg"})

    summary = stage_03_module.run_stage_03(
        drive_project_root=drive_project_root,
        config_path=config_path,
        source_package_path=source_package_path,
        notebook_name="03_Experiment_Matrix_Full",
        stage_run_id="stage03_sync_test",
    )

    stage_manifest = json.loads((stage_roots["run_root"] / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    workflow_summary = json.loads((stage_roots["run_root"] / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert workflow_summary["auxiliary_analysis_runtime_executed"] is False
    assert stage_manifest["auxiliary_analysis_runtime_executed"] is False
    assert workflow_summary["gpu_memory_summary"]["status"] == "ok"
    assert workflow_summary["gpu_memory_summary"]["peak_memory_allocated_bytes_max"] == 2048
    assert stage_manifest["gpu_memory_summary"]["peak_memory_reserved_bytes_max"] == 3072
    assert workflow_summary["gpu_memory_profile_breakdown_path"].endswith("gpu_memory_profile_breakdown.json")
    assert stage_manifest["gpu_memory_profile_breakdown_path"].endswith("gpu_memory_profile_breakdown.json")
    assert bootstrap_calls == [
        {
            "drive_project_root": drive_project_root.as_posix(),
            "allow_generate": False,
            "allow_missing": True,
            "attestation_enabled": False,
        }
    ]