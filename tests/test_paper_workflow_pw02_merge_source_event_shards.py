"""
File purpose: Validate PW02 source merge and global-threshold workflow closure.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import pytest

from main.cli import run_calibrate as run_calibrate_cli
from main.core import config_loader as core_config_loader
from main.evaluation import workflow_inputs as eval_workflow_inputs
import paper_workflow.scripts.pw02_merge_source_event_shards as pw02_module
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import build_source_shard_root, read_jsonl
from scripts.notebook_runtime_common import ensure_directory, write_json_atomic, write_yaml_mapping


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a PW00 fixture family for PW02 tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "pw02_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[3, 9],
        source_shard_count=2,
    )


def _materialize_completed_pw01_shards(
    summary: Dict[str, Any],
    role_shard_indices: Dict[str, Sequence[int]] | None = None,
) -> None:
    """
    Create completed PW01 shard manifests and detect records for selected source roles.

    Args:
        summary: PW00 summary payload.
        role_shard_indices: Optional mapping from sample_role to completed shard indices.
            When omitted, all planned shards are materialized for all source roles.

    Returns:
        None.
    """
    family_root = Path(str(summary["family_root"]))
    shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))
    event_lookup = {
        row["event_id"]: row
        for row in read_jsonl(Path(str(summary["source_event_grid_path"])))
    }
    sample_role_plans = cast(Dict[str, Dict[str, Any]], shard_plan["sample_role_plans"])

    if role_shard_indices is None:
        normalized_role_shard_indices: Dict[str, List[int]] = {}
        for sample_role, role_plan in sample_role_plans.items():
            normalized_role_shard_indices[sample_role] = [
                int(shard_row["shard_index"])
                for shard_row in cast(List[Dict[str, Any]], role_plan["shards"])
            ]
    else:
        normalized_role_shard_indices = {
            str(sample_role): [int(shard_index) for shard_index in shard_indices]
            for sample_role, shard_indices in role_shard_indices.items()
        }

    for sample_role, selected_shard_indices in normalized_role_shard_indices.items():
        if sample_role not in sample_role_plans:
            raise AssertionError(f"unsupported sample_role fixture request: {sample_role}")
        role_plan = sample_role_plans[sample_role]
        shard_index_filter = set(selected_shard_indices)
        for shard_row in cast(List[Dict[str, Any]], role_plan["shards"]):
            shard_index = int(shard_row["shard_index"])
            if shard_index not in shard_index_filter:
                continue
            shard_root = ensure_directory(build_source_shard_root(family_root, sample_role, shard_index))
            ensure_directory(shard_root / "records")
            events: List[Dict[str, Any]] = []
            for event_id in cast(List[str], shard_row["assigned_event_ids"]):
                event = event_lookup[event_id]
                event_index = int(event["event_index"])
                detect_record_path = shard_root / "records" / f"event_{event_index:06d}_detect_record.json"
                detect_payload: Dict[str, Any]
                if sample_role == "positive_source":
                    detect_payload = {
                        "sample_role": sample_role,
                        "content_evidence_payload": {
                            "status": "ok",
                            "score": 0.91,
                            "content_chain_score": 0.91,
                            "plan_digest": f"plan-{event_id}",
                            "basis_digest": f"basis-{event_id}",
                        },
                        "final_decision": {
                            "is_watermarked": True,
                        },
                        "attestation": {
                            "final_event_attested_decision": {
                                "status": "ok",
                                "is_event_attested": True,
                                "event_attestation_score_name": "event_attestation_score",
                                "event_attestation_score": 0.81,
                            }
                        },
                    }
                elif sample_role == "clean_negative":
                    detect_payload = {
                        "sample_role": sample_role,
                        "plan_digest": f"runtime-observed-plan-{event_id}",
                        "basis_digest": f"runtime-observed-basis-{event_id}",
                        "content_evidence_payload": {
                            "status": "absent",
                            "score": None,
                            "content_chain_score": None,
                            "content_failure_reason": "detector_no_plan_expected",
                        },
                        "final_decision": {
                            "is_watermarked": False,
                        },
                        "attestation": {
                            "final_event_attested_decision": {
                                "status": "absent",
                                "is_event_attested": False,
                                "event_attestation_score_name": "event_attestation_score",
                                "event_attestation_score": 0.0,
                            }
                        },
                    }
                else:
                    detect_payload = {
                        "sample_role": sample_role,
                        "content_evidence_payload": {
                            "status": "ok",
                            "score": 0.11,
                            "content_chain_score": 0.11,
                            "plan_digest": f"plan-{event_id}",
                            "basis_digest": f"basis-{event_id}",
                        },
                        "final_decision": {
                            "is_watermarked": False,
                        },
                        "attestation": {
                            "final_event_attested_decision": {
                                "status": "absent",
                                "is_event_attested": False,
                                "event_attestation_score_name": "event_attestation_score",
                                "event_attestation_score": 0.0,
                            }
                        },
                    }
                write_json_atomic(detect_record_path, detect_payload)
                events.append(
                    {
                        "event_id": event_id,
                        "sample_role": sample_role,
                        "event_index": event_index,
                        "detect_record_path": detect_record_path.as_posix(),
                    }
                )

            shard_manifest_path = shard_root / "shard_manifest.json"
            write_json_atomic(
                shard_manifest_path,
                {
                    "status": "completed",
                    "sample_role": sample_role,
                    "events": events,
                },
            )


def test_strict_clean_negative_runtime_plan_digest_maps_to_formal_null_content_score() -> None:
    """
    Verify runtime-observed digests do not disqualify strict clean_negative formal-null mapping.

    Args:
        None.

    Returns:
        None.
    """
    record: Dict[str, Any] = {
        "sample_role": "clean_negative",
        "plan_digest": "runtime-observed-plan",
        "basis_digest": "runtime-observed-basis",
        "content_evidence_payload": {
            "status": "absent",
            "score": None,
            "content_chain_score": None,
            "content_failure_reason": "detector_no_plan_expected",
        },
    }

    assert eval_workflow_inputs._is_strict_clean_negative_formal_null_record(record) is True
    assert eval_workflow_inputs._resolve_content_score_source(record) == (
        0.0,
        "strict_clean_negative_formal_null",
    )


def _patch_pw02_python_stage_runner(monkeypatch: Any) -> List[Dict[str, Any]]:
    """
    Patch PW02 Python stage execution to lightweight artifact-writing stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Captured stage-call summaries.
    """

    observed_calls: List[Dict[str, Any]] = []

    def fake_run_python_stage_command(
        *,
        module_name: str,
        output_dir: Path,
        config_path: Path,
        log_prefix: str,
        overrides: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        normalized_overrides = list(overrides) if overrides is not None else []
        config_payload = cast(Dict[str, Any], pw02_module.load_yaml_mapping(config_path))
        observed_calls.append(
            {
                "module_name": module_name,
                "output_dir": output_dir.as_posix(),
                "config_path": config_path.as_posix(),
                "log_prefix": log_prefix,
                "config": config_payload,
                "overrides": normalized_overrides,
            }
        )
        ensure_directory(output_dir / "logs")
        if module_name == "main.cli.run_calibrate":
            thresholds_path = output_dir / "artifacts" / "thresholds" / "thresholds_artifact.json"
            ensure_directory(thresholds_path.parent)
            write_json_atomic(
                thresholds_path,
                {
                    "status": "ok",
                    "score_name": config_payload["calibration"]["score_name"],
                    "target_fpr": 0.01,
                },
            )
            write_json_atomic(output_dir / "records" / "calibration_record.json", {"status": "ok"})
        elif module_name == "main.cli.run_evaluate":
            write_json_atomic(output_dir / "records" / "evaluate_record.json", {"status": "ok"})
            write_json_atomic(output_dir / "artifacts" / "evaluation_report.json", {"status": "ok"})
        else:
            raise ValueError(f"unsupported module_name: {module_name}")

        return {
            "return_code": 0,
            "status": "ok",
            "stdout_log_path": (output_dir / "logs" / f"{log_prefix}_stdout.log").as_posix(),
            "stderr_log_path": (output_dir / "logs" / f"{log_prefix}_stderr.log").as_posix(),
            "command": [module_name, str(config_path), str(output_dir), *normalized_overrides],
        }

    monkeypatch.setattr(pw02_module, "_run_python_stage_command", fake_run_python_stage_command)
    return observed_calls


def _load_config_loader_contract_context() -> tuple[Any, Any, Any, Any]:
    """
    Load config_loader contract context for runtime-config validation tests.

    Args:
        None.

    Returns:
        Tuple of (whitelist, semantics, contracts, interpretation).
    """
    contracts, interpretation = core_config_loader.load_frozen_contracts_interpretation()
    whitelist = core_config_loader.load_runtime_whitelist()
    semantics = core_config_loader.load_policy_path_semantics()
    return whitelist, semantics, contracts, interpretation


def _validate_pw02_runtime_config_via_config_loader(
    *,
    tmp_path: Path,
    file_name: str,
    runtime_cfg_obj: Dict[str, Any],
    override_args: List[str],
) -> Dict[str, Any]:
    """
    Validate one PW02 runtime config through the real config_loader entrypoint.

    Args:
        tmp_path: Temporary path fixture.
        file_name: Output config file name.
        runtime_cfg_obj: Runtime config payload.
        override_args: CLI override arguments used by PW02.

    Returns:
        Validated config mapping returned by config_loader.
    """
    whitelist, semantics, contracts, interpretation = _load_config_loader_contract_context()
    config_path = tmp_path / file_name
    write_yaml_mapping(config_path, runtime_cfg_obj)
    validated_cfg, _, _ = core_config_loader.load_and_validate_config(
        config_path,
        whitelist,
        semantics,
        contracts,
        interpretation,
        overrides=override_args,
    )
    return cast(Dict[str, Any], validated_cfg)


def _assert_pw02_override_applied_contract(validated_cfg: Dict[str, Any]) -> None:
    """
    Assert PW02 runtime cfg produces the exact override_applied contract block.

    Args:
        validated_cfg: Validated config mapping.

    Returns:
        None.
    """
    whitelist = core_config_loader.load_runtime_whitelist()
    override_applied = cast(Dict[str, Any], validated_cfg.get("override_applied"))
    requested_overrides = cast(List[Dict[str, Any]], override_applied.get("requested_overrides"))
    applied_fields = cast(List[Dict[str, Any]], override_applied.get("applied_fields"))

    assert validated_cfg["allow_nonempty_run_root"] is True
    assert validated_cfg["allow_nonempty_run_root_reason"] == pw02_module.PW02_RUN_ROOT_REUSE_REASON
    assert override_applied["source"] == "cli"
    assert override_applied["allowed_fields_version"] == whitelist.whitelist_version
    assert override_applied["requested_kv"] == {
        "run_root_reuse_allowed": True,
        "run_root_reuse_reason": pw02_module.PW02_RUN_ROOT_REUSE_REASON,
    }
    assert override_applied["rejected_fields"] == []
    assert len(requested_overrides) == 2
    assert len(applied_fields) == 2
    assert [item["arg_name"] for item in applied_fields] == [
        "run_root_reuse_allowed",
        "run_root_reuse_reason",
    ]
    assert [item["field_path"] for item in applied_fields] == [
        "allow_nonempty_run_root",
        "allow_nonempty_run_root_reason",
    ]
    assert [item["new_value"] for item in applied_fields] == [
        True,
        pw02_module.PW02_RUN_ROOT_REUSE_REASON,
    ]
    for requested in requested_overrides:
        assert set(requested) == {
            "arg_name",
            "field_path",
            "override_mode",
            "source",
            "raw_key",
            "raw_value",
            "value",
        }
    for applied in applied_fields:
        assert set(applied) == {
            "arg_name",
            "field_path",
            "override_mode",
            "source",
            "old_value",
            "new_value",
        }
        assert applied["source"] == "cli"
        assert applied["override_mode"] == "set"


def test_pw02_merges_dual_role_shards_and_builds_score_runs(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify PW02 merges completed PW01 shards and emits both score workflows.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_merge")
    _materialize_completed_pw01_shards(summary)
    _patch_pw02_python_stage_runner(monkeypatch)

    pw02_summary = pw02_module.run_pw02_merge_source_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw02_merge",
    )

    assert pw02_summary["status"] == "ok"
    assert pw02_summary["split_counts"] == {"calibration": 4, "evaluate": 4}
    assert set(pw02_summary["score_runs"]) == {
        pw02_module.CONTENT_SCORE_NAME,
        pw02_module.EVENT_ATTESTATION_SCORE_NAME,
    }

    content_run = cast(Dict[str, Any], pw02_summary["score_runs"][pw02_module.CONTENT_SCORE_NAME])
    attestation_run = cast(Dict[str, Any], pw02_summary["score_runs"][pw02_module.EVENT_ATTESTATION_SCORE_NAME])
    assert content_run["calibration_inputs"]["record_count"] == 4
    assert content_run["evaluate_inputs"]["record_count"] == 4
    assert attestation_run["calibration_inputs"]["record_count"] == 4
    assert attestation_run["evaluate_inputs"]["record_count"] == 4
    assert Path(str(pw02_summary["planner_conditioned_control_negative_pool_manifest_path"])).exists()
    assert pw02_summary["planner_conditioned_control_negative_cohort_status"] == "completed"
    assert pw02_summary["planner_conditioned_control_negative_role_requirement"] == "optional_diagnostic"

    content_negative_record_summary = next(
        record
        for record in content_run["evaluate_inputs"]["records"]
        if record["sample_role"] == "clean_negative"
    )
    assert content_negative_record_summary["score_source"] == "strict_clean_negative_formal_null"

    attestation_negative_record_path = Path(
        str(
            next(
                record["record_path"]
                for record in attestation_run["evaluate_inputs"]["records"]
                if record["sample_role"] == "clean_negative"
            )
        )
    )
    attestation_negative_record = json.loads(attestation_negative_record_path.read_text(encoding="utf-8"))
    assert attestation_negative_record["label"] is False
    assert attestation_negative_record["calibration_label_resolution"] == "paper_workflow_clean_negative"
    assert (
        attestation_negative_record["attestation"]["final_event_attested_decision"]["event_attestation_score"]
        == 0.0
    )

    content_negative_record_path = Path(str(content_negative_record_summary["record_path"]))
    content_negative_record = json.loads(content_negative_record_path.read_text(encoding="utf-8"))
    assert content_negative_record["paper_workflow_score_source"] == "strict_clean_negative_formal_null"
    assert str(content_negative_record["plan_digest"]).startswith("runtime-observed-plan-")
    assert str(content_negative_record["basis_digest"]).startswith("runtime-observed-basis-")

    formal_final_decision_metrics = cast(Dict[str, Any], pw02_summary["formal_final_decision_metrics"])
    derived_system_union_metrics = cast(Dict[str, Any], pw02_summary["derived_system_union_metrics"])
    assert formal_final_decision_metrics["n_total"] == 4
    assert formal_final_decision_metrics["n_positive"] == 2
    assert formal_final_decision_metrics["n_negative"] == 2
    assert derived_system_union_metrics["n_total"] == 4
    assert derived_system_union_metrics["n_positive"] == 2
    assert derived_system_union_metrics["n_negative"] == 2
    assert pw02_summary["system_final_metrics"] == derived_system_union_metrics


def test_pw02_succeeds_without_optional_control_cohort(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify PW02 formal mainline succeeds when the optional control cohort is absent.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_no_control")
    _materialize_completed_pw01_shards(
        summary,
        role_shard_indices={
            "positive_source": [0, 1],
            "clean_negative": [0, 1],
        },
    )
    _patch_pw02_python_stage_runner(monkeypatch)

    pw02_summary = pw02_module.run_pw02_merge_source_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw02_no_control",
    )

    control_negative_pool_manifest = json.loads(
        Path(str(pw02_summary["planner_conditioned_control_negative_pool_manifest_path"])).read_text(encoding="utf-8")
    )
    finalize_manifest = json.loads(
        Path(str(pw02_summary["paper_source_finalize_manifest_path"])).read_text(encoding="utf-8")
    )

    assert pw02_summary["status"] == "ok"
    assert pw02_summary["planner_conditioned_control_negative_cohort_status"] == "not_provided"
    assert pw02_summary["planner_conditioned_control_negative_role_requirement"] == "optional_diagnostic"
    assert pw02_summary["planner_conditioned_control_negative_expected_source_shard_count"] == 2
    assert pw02_summary["planner_conditioned_control_negative_discovered_source_shard_count"] == 0

    assert control_negative_pool_manifest["source_role"] == "planner_conditioned_control_negative"
    assert control_negative_pool_manifest["role_requirement"] == "optional_diagnostic"
    assert control_negative_pool_manifest["cohort_status"] == "not_provided"
    assert control_negative_pool_manifest["diagnostic_only"] is True
    assert control_negative_pool_manifest["event_count"] == 0
    assert control_negative_pool_manifest["events"] == []
    assert control_negative_pool_manifest["source_shard_manifest_paths"] == []
    assert control_negative_pool_manifest["expected_source_shard_count"] == 2
    assert control_negative_pool_manifest["discovered_source_shard_count"] == 0
    assert control_negative_pool_manifest["missing_source_shard_indices"] == [0, 1]

    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["event_count"] == 0
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["cohort_status"] == "not_provided"
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["role_requirement"] == "optional_diagnostic"
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["diagnostic_only"] is True


def test_pw02_fails_fast_when_optional_control_cohort_is_partial(tmp_path: Path) -> None:
    """
    Verify PW02 rejects a partially materialized optional control cohort.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_partial_control")
    _materialize_completed_pw01_shards(
        summary,
        role_shard_indices={
            "positive_source": [0, 1],
            "clean_negative": [0, 1],
            "planner_conditioned_control_negative": [0],
        },
    )

    with pytest.raises(RuntimeError, match="PW02 optional diagnostic cohort is partially provided"):
        pw02_module.run_pw02_merge_source_event_shards(
            drive_project_root=tmp_path / "drive",
            family_id="family_pw02_partial_control",
        )


def test_pw02_writes_top_level_exports_with_honest_system_final_metrics(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Verify PW02 writes top-level pool/finalize/threshold/evaluate exports and
    keeps system_final as an honest derived-metrics artifact.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_exports")
    _materialize_completed_pw01_shards(summary)
    _patch_pw02_python_stage_runner(monkeypatch)

    pw02_summary = pw02_module.run_pw02_merge_source_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw02_exports",
    )

    positive_pool_manifest = json.loads(
        Path(str(pw02_summary["positive_source_pool_manifest_path"])).read_text(encoding="utf-8")
    )
    clean_negative_pool_manifest = json.loads(
        Path(str(pw02_summary["clean_negative_pool_manifest_path"])).read_text(encoding="utf-8")
    )
    control_negative_pool_manifest = json.loads(
        Path(str(pw02_summary["planner_conditioned_control_negative_pool_manifest_path"])).read_text(encoding="utf-8")
    )
    finalize_manifest = json.loads(
        Path(str(pw02_summary["paper_source_finalize_manifest_path"])).read_text(encoding="utf-8")
    )
    content_threshold_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["threshold_exports"])["content"])).read_text(encoding="utf-8")
    )
    attestation_threshold_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["threshold_exports"])["attestation"])).read_text(encoding="utf-8")
    )
    content_clean_evaluate_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["clean_evaluate_exports"])["content"])).read_text(encoding="utf-8")
    )
    attestation_clean_evaluate_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["clean_evaluate_exports"])["attestation"])).read_text(encoding="utf-8")
    )
    formal_final_decision_metrics_export = json.loads(
        Path(str(pw02_summary["formal_final_decision_metrics_artifact_path"])).read_text(encoding="utf-8")
    )
    derived_system_union_metrics_export = json.loads(
        Path(str(pw02_summary["derived_system_union_metrics_artifact_path"])).read_text(encoding="utf-8")
    )

    assert positive_pool_manifest["family_id"] == "family_pw02_exports"
    assert positive_pool_manifest["source_role"] == "positive_source"
    assert positive_pool_manifest["event_count"] == 4
    assert len(positive_pool_manifest["events"]) == 4
    assert len(positive_pool_manifest["source_shard_manifest_paths"]) == 2

    assert clean_negative_pool_manifest["family_id"] == "family_pw02_exports"
    assert clean_negative_pool_manifest["source_role"] == "clean_negative"
    assert clean_negative_pool_manifest["event_count"] == 4
    assert len(clean_negative_pool_manifest["events"]) == 4
    assert len(clean_negative_pool_manifest["source_shard_manifest_paths"]) == 2

    assert control_negative_pool_manifest["family_id"] == "family_pw02_exports"
    assert control_negative_pool_manifest["source_role"] == "planner_conditioned_control_negative"
    assert control_negative_pool_manifest["role_requirement"] == "optional_diagnostic"
    assert control_negative_pool_manifest["cohort_status"] == "completed"
    assert control_negative_pool_manifest["diagnostic_only"] is True
    assert control_negative_pool_manifest["event_count"] == 4
    assert len(control_negative_pool_manifest["events"]) == 4
    assert control_negative_pool_manifest["expected_source_shard_count"] == 2
    assert control_negative_pool_manifest["discovered_source_shard_count"] == 2
    assert control_negative_pool_manifest["missing_source_shard_indices"] == []
    assert len(control_negative_pool_manifest["source_shard_manifest_paths"]) == 2

    assert finalize_manifest["source_pools"]["positive_source"]["manifest_path"] == pw02_summary["positive_source_pool_manifest_path"]
    assert finalize_manifest["source_pools"]["clean_negative"]["manifest_path"] == pw02_summary["clean_negative_pool_manifest_path"]
    assert (
        finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["manifest_path"]
        == pw02_summary["planner_conditioned_control_negative_pool_manifest_path"]
    )
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["cohort_status"] == "completed"
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["role_requirement"] == "optional_diagnostic"
    assert finalize_manifest["source_pools"]["planner_conditioned_control_negative"]["diagnostic_only"] is True
    assert finalize_manifest["threshold_exports"]["content"]["path"] == cast(Dict[str, Any], pw02_summary["threshold_exports"])["content"]
    assert finalize_manifest["clean_evaluate_exports"]["content"]["path"] == cast(Dict[str, Any], pw02_summary["clean_evaluate_exports"])["content"]

    content_run = cast(Dict[str, Any], pw02_summary["score_runs"][pw02_module.CONTENT_SCORE_NAME])
    attestation_run = cast(Dict[str, Any], pw02_summary["score_runs"][pw02_module.EVENT_ATTESTATION_SCORE_NAME])

    assert content_threshold_export["score_name"] == pw02_module.CONTENT_SCORE_NAME
    assert content_threshold_export["source_thresholds_artifact_path"] == content_run["thresholds_artifact_path"]
    assert content_threshold_export["thresholds_artifact"]["score_name"] == pw02_module.CONTENT_SCORE_NAME

    assert attestation_threshold_export["score_name"] == pw02_module.EVENT_ATTESTATION_SCORE_NAME
    assert attestation_threshold_export["source_thresholds_artifact_path"] == attestation_run["thresholds_artifact_path"]
    assert attestation_threshold_export["thresholds_artifact"]["score_name"] == pw02_module.EVENT_ATTESTATION_SCORE_NAME

    assert content_clean_evaluate_export["score_name"] == pw02_module.CONTENT_SCORE_NAME
    assert content_clean_evaluate_export["source_evaluate_run_root"] == content_run["evaluate_run_root"]
    assert content_clean_evaluate_export["source_evaluate_record_path"] == content_run["evaluate_record_path"]
    assert content_clean_evaluate_export["evaluate_input_counts"] == {
        "positive_source": 2,
        "clean_negative": 2,
    }
    assert content_clean_evaluate_export["evaluate_record"]["status"] == "ok"

    assert attestation_clean_evaluate_export["score_name"] == pw02_module.EVENT_ATTESTATION_SCORE_NAME
    assert attestation_clean_evaluate_export["source_evaluate_run_root"] == attestation_run["evaluate_run_root"]
    assert attestation_clean_evaluate_export["source_evaluate_record_path"] == attestation_run["evaluate_record_path"]
    assert attestation_clean_evaluate_export["evaluate_input_counts"] == {
        "positive_source": 2,
        "clean_negative": 2,
    }
    assert attestation_clean_evaluate_export["evaluate_record"]["status"] == "ok"

    assert formal_final_decision_metrics_export["source_kind"] == "formal_final_decision_metrics_from_content_evaluate_inputs"
    assert formal_final_decision_metrics_export["is_formal_evaluate_record"] is False
    assert formal_final_decision_metrics_export["source_score_name"] == pw02_module.CONTENT_SCORE_NAME
    assert formal_final_decision_metrics_export["source_evaluate_run_root"] == content_run["evaluate_run_root"]
    assert formal_final_decision_metrics_export["scope"] == "formal_final_decision"
    assert formal_final_decision_metrics_export["metrics"]["scope"] == "formal_final_decision"

    assert derived_system_union_metrics_export["source_kind"] == "derived_metrics_from_content_evaluate_inputs"
    assert derived_system_union_metrics_export["is_formal_evaluate_record"] is False
    assert derived_system_union_metrics_export["source_score_name"] == pw02_module.CONTENT_SCORE_NAME
    assert derived_system_union_metrics_export["source_evaluate_run_root"] == content_run["evaluate_run_root"]
    assert derived_system_union_metrics_export["scope"] == "derived_system_union"
    assert derived_system_union_metrics_export["metrics"]["scope"] == "system_final"

    assert finalize_manifest["formal_final_decision"]["mode"] == "formal_final_decision_metrics_from_content_evaluate_inputs"
    assert finalize_manifest["formal_final_decision"]["is_formal_evaluate_record"] is False
    assert (
        finalize_manifest["formal_final_decision"]["artifact_path"]
        == pw02_summary["formal_final_decision_metrics_artifact_path"]
    )
    assert finalize_manifest["derived_system_union"]["mode"] == "derived_metrics_from_content_evaluate_inputs"
    assert finalize_manifest["derived_system_union"]["is_formal_evaluate_record"] is False
    assert (
        finalize_manifest["derived_system_union"]["artifact_path"]
        == pw02_summary["derived_system_union_metrics_artifact_path"]
    )
    assert finalize_manifest["system_final"]["mode"] == "derived_metrics_from_content_evaluate_inputs"
    assert finalize_manifest["system_final"]["is_formal_evaluate_record"] is False
    assert finalize_manifest["system_final"]["artifact_path"] == pw02_summary["system_final_metrics_artifact_path"]


def test_pw02_passes_run_root_reuse_overrides_for_calibrate_and_evaluate_configs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """
    Verify PW02 forwards audited run-root reuse overrides for both calibrate and evaluate configs.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_override_contract")
    _materialize_completed_pw01_shards(summary)
    observed_calls = _patch_pw02_python_stage_runner(monkeypatch)

    pw02_module.run_pw02_merge_source_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw02_override_contract",
    )

    calibrate_call = next(call for call in observed_calls if call["module_name"] == "main.cli.run_calibrate")
    evaluate_call = next(call for call in observed_calls if call["module_name"] == "main.cli.run_evaluate")

    for call in [calibrate_call, evaluate_call]:
        runtime_cfg_obj = cast(Dict[str, Any], call["config"])
        override_args = cast(List[str], call["overrides"])
        assert runtime_cfg_obj["allow_nonempty_run_root"] is True
        assert runtime_cfg_obj["allow_nonempty_run_root_reason"] == pw02_module.PW02_RUN_ROOT_REUSE_REASON
        assert override_args == pw02_module._build_run_root_reuse_override_args(runtime_cfg_obj)

    validated_calibrate_cfg = _validate_pw02_runtime_config_via_config_loader(
        tmp_path=tmp_path,
        file_name="captured_pw02_calibrate_runtime.yaml",
        runtime_cfg_obj=cast(Dict[str, Any], calibrate_call["config"]),
        override_args=cast(List[str], calibrate_call["overrides"]),
    )
    validated_evaluate_cfg = _validate_pw02_runtime_config_via_config_loader(
        tmp_path=tmp_path,
        file_name="captured_pw02_evaluate_runtime.yaml",
        runtime_cfg_obj=cast(Dict[str, Any], evaluate_call["config"]),
        override_args=cast(List[str], evaluate_call["overrides"]),
    )

    _assert_pw02_override_applied_contract(validated_calibrate_cfg)
    _assert_pw02_override_applied_contract(validated_evaluate_cfg)


def test_pw02_calibrate_runtime_config_reaches_run_calibrate_post_validation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """
    Verify a PW02 calibrate runtime config reaches run_calibrate past config validation.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_pw02_calibrate_validation")
    _materialize_completed_pw01_shards(summary)
    observed_calls = _patch_pw02_python_stage_runner(monkeypatch)

    pw02_module.run_pw02_merge_source_event_shards(
        drive_project_root=tmp_path / "drive",
        family_id="family_pw02_calibrate_validation",
    )

    calibrate_call = next(call for call in observed_calls if call["module_name"] == "main.cli.run_calibrate")
    runtime_cfg_path = tmp_path / "pw02_calibrate_runtime_config.yaml"
    write_yaml_mapping(runtime_cfg_path, cast(Dict[str, Any], calibrate_call["config"]))

    def raise_after_config_validation(cfg: Any) -> tuple[int, int]:
        _ = cfg
        raise RuntimeError("pw02_after_config_validation_marker")

    monkeypatch.setattr(
        run_calibrate_cli,
        "_validate_detect_record_label_balance_for_calibration",
        raise_after_config_validation,
    )

    with pytest.raises(RuntimeError, match="pw02_after_config_validation_marker"):
        run_calibrate_cli.run_calibrate(
            str(tmp_path / "pw02_calibrate_real_run"),
            str(runtime_cfg_path),
            cast(List[str], calibrate_call["overrides"]),
        )