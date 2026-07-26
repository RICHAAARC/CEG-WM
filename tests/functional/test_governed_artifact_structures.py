"""验证 records 和 manifests 的最小结构。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.protocol.records import ExperimentRecord, validate_record
from paper_artifacts.artifact_manifest import build_artifact_manifest


@pytest.mark.quick
def test_experiment_record_contains_required_fields() -> None:
    """实验 record 必须包含产物重建所需的最小字段。"""
    record = ExperimentRecord(
        record_id="record_example",
        run_id="run_example",
        comparison_group_name="primary_comparison",
        comparison_protocol_digest="a" * 64,
        sample_manifest_digest="b" * 64,
        split="validation",
        method_name="example_method",
        method_role="project_method",
        method_config_digest="c" * 64,
        method_code_revision="commit_example",
        model_revision="model_revision_example",
        seed=17,
        metric_name="accuracy",
        metric_value=0.9,
        execution_status="success",
        failure_reason=None,
        exclusion_reason=None,
        metadata={},
    )
    assert validate_record(record.to_dict()) == []


@pytest.mark.quick
def test_experiment_record_rejects_unknown_method_role() -> None:
    record = ExperimentRecord(
        record_id="record_role_check",
        run_id="run_role_check",
        comparison_group_name="primary_comparison",
        comparison_protocol_digest="a" * 64,
        sample_manifest_digest="b" * 64,
        split="evaluation",
        method_name="example_method",
        method_role="unregistered_role",
        method_config_digest="c" * 64,
        method_code_revision="commit_example",
        model_revision="model_revision_example",
        seed=17,
        metric_name="accuracy",
        metric_value=0.9,
        execution_status="success",
        failure_reason=None,
        exclusion_reason=None,
        metadata={},
    )
    assert "method_role_invalid" in validate_record(record.to_dict())


@pytest.mark.quick
def test_success_record_rejects_failure_or_exclusion_reasons() -> None:
    record = ExperimentRecord(
        record_id="record_status_check",
        run_id="run_status_check",
        comparison_group_name="primary_comparison",
        comparison_protocol_digest="a" * 64,
        sample_manifest_digest="b" * 64,
        split="evaluation",
        method_name="example_method",
        method_role="project_method",
        method_config_digest="c" * 64,
        method_code_revision="commit_example",
        model_revision="model_revision_example",
        seed=17,
        metric_name="accuracy",
        metric_value=0.9,
        execution_status="success",
        failure_reason="failure_text",
        exclusion_reason="exclusion_text",
        metadata={},
    )
    violations = validate_record(record.to_dict())
    assert "successful_record_failure_reason_forbidden" in violations
    assert "successful_record_exclusion_reason_forbidden" in violations


@pytest.mark.quick
def test_failed_and_excluded_records_have_mutually_exclusive_reasons_and_no_metric() -> None:
    base = ExperimentRecord(
        record_id="record_non_success_check",
        run_id="run_non_success_check",
        comparison_group_name="primary_comparison",
        comparison_protocol_digest="a" * 64,
        sample_manifest_digest="b" * 64,
        split="evaluation",
        method_name="example_method",
        method_role="project_method",
        method_config_digest="c" * 64,
        method_code_revision="commit_example",
        model_revision="model_revision_example",
        seed=17,
        metric_name="accuracy",
        metric_value=0.9,
        execution_status="success",
        failure_reason=None,
        exclusion_reason=None,
        metadata={},
    )
    failed = replace(base, execution_status="failed", metric_value=None, failure_reason="runtime_error")
    excluded = replace(base, execution_status="excluded", metric_value=None, exclusion_reason="policy_rule")
    assert validate_record(failed.to_dict()) == []
    assert validate_record(excluded.to_dict()) == []

    failed_with_metric = replace(failed, metric_value=0.1)
    failed_with_exclusion = replace(failed, exclusion_reason="policy_rule")
    excluded_with_failure = replace(excluded, failure_reason="runtime_error")
    assert "non_success_metric_value_forbidden" in validate_record(failed_with_metric.to_dict())
    assert "failed_record_exclusion_reason_forbidden" in validate_record(failed_with_exclusion.to_dict())
    assert "excluded_record_failure_reason_forbidden" in validate_record(excluded_with_failure.to_dict())


@pytest.mark.quick
def test_artifact_manifest_records_rebuild_provenance() -> None:
    """产物 manifest 必须记录输入、输出、配置摘要和重建命令。"""
    manifest = build_artifact_manifest(
        artifact_id="table_example",
        artifact_type="table",
        input_paths=("outputs/records/example.jsonl",),
        output_paths=("outputs/tables/example.csv",),
        config={"metric_name": "accuracy"},
        code_version="uncommitted_template",
        rebuild_command="python scripts/rebuild_example_artifacts.py",
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["artifact_id"] == "table_example"
    assert manifest_dict["config_digest"]
    assert manifest_dict["rebuild_command"]
