"""
File purpose: Validate PW02 source merge and global-threshold workflow closure.
Module type: General module
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import pytest
from PIL import Image

from main.cli import run_calibrate as run_calibrate_cli
from main.core import config_loader as core_config_loader
from main.evaluation import workflow_inputs as eval_workflow_inputs
import paper_workflow.scripts.pw02_merge_source_event_shards as pw02_module
import paper_workflow.scripts.pw02_metrics_extensions as pw02_metrics_extensions_module
import paper_workflow.scripts.pw_quality_metrics as pw_quality_metrics_module
from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
from paper_workflow.scripts.pw_common import build_source_shard_root, read_jsonl
from scripts.notebook_runtime_common import ensure_directory, normalize_path_value, write_json_atomic, write_yaml_mapping


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
                event_root = ensure_directory(shard_root / "events" / f"event_{event_index:06d}")
                artifacts_root = ensure_directory(event_root / "artifacts")
                source_image_path = artifacts_root / f"event_{event_index:06d}_source.png"
                preview_image_path = artifacts_root / f"event_{event_index:06d}_preview.png"
                watermarked_image_path = artifacts_root / f"event_{event_index:06d}_watermarked.png"
                source_image = Image.new("RGB", (8, 8), color=(40 + event_index, 70, 100))
                preview_image = Image.new("RGB", (8, 8), color=(42 + event_index, 70, 100))
                watermarked_image = Image.new("RGB", (8, 8), color=(45 + event_index, 70, 100))
                source_image.save(source_image_path)
                preview_image.save(preview_image_path)
                watermarked_image.save(watermarked_image_path)
                preview_generation_record_path = artifacts_root / "preview_generation_record.json"
                embed_record_path = shard_root / "records" / f"event_{event_index:06d}_embed_record.json"
                write_json_atomic(
                    preview_generation_record_path,
                    {
                        "status": "ok",
                        "persisted_artifact_path": normalize_path_value(preview_image_path),
                    },
                )
                write_json_atomic(
                    embed_record_path,
                    {
                        "watermarked_path": normalize_path_value(watermarked_image_path),
                    },
                )
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
                            "score_parts": {
                                "lf_trajectory_detect_trace": {
                                    "codeword_agreement": 0.72 + 0.06 * float(event_index),
                                    "n_bits_compared": 96,
                                    "detect_variant": "correlation_v2",
                                    "message_source": "plan_digest",
                                }
                            },
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
                        "plan_input_digest": f"runtime-observed-plan-input-{event_id}",
                        "plan_input_schema_version": "v2",
                        "plan_digest_expected": None,
                        "plan_digest_status": "absent",
                        "plan_digest_validation_status": "absent",
                        "subspace_planner_impl_identity": {
                            "impl_id": "subspace_planner",
                            "impl_version": "v2",
                            "impl_digest": f"planner-digest-{event_id}",
                        },
                        "subspace_plan": {
                            "planner_input_digest": f"runtime-observed-plan-input-{event_id}",
                            "planner_impl_identity": {
                                "impl_id": "subspace_planner",
                                "impl_version": "v2",
                            },
                            "rank": 128,
                        },
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
                            "authenticity_result": {
                                "status": "absent",
                                "bundle_status": "absent",
                                "statement_status": "absent",
                            },
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
                        "prompt_text": event.get("prompt_text"),
                        "embed_record_path": embed_record_path.as_posix(),
                        "detect_record_path": detect_record_path.as_posix(),
                        "source_image": {
                            "exists": True,
                            "path": normalize_path_value(source_image_path),
                            "package_relative_path": f"source_images/{event_index:06d}.png",
                            "missing_reason": None,
                        },
                        "plain_preview_image": {
                            "exists": True,
                            "path": normalize_path_value(preview_image_path),
                            "package_relative_path": f"plain_preview_images/{event_index:06d}.png",
                            "missing_reason": None,
                        },
                        "preview_generation_record": {
                            "exists": True,
                            "path": normalize_path_value(preview_generation_record_path),
                            "package_relative_path": f"preview_generation_records/{event_index:06d}.json",
                            "missing_reason": None,
                        },
                        "watermarked_output_image": {
                            "exists": True,
                            "path": normalize_path_value(watermarked_image_path),
                            "package_relative_path": f"watermarked_output_images/{event_index:06d}.png",
                            "missing_reason": None,
                        },
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


def test_strict_clean_negative_runtime_observed_planner_fields_map_to_formal_null_content_score() -> None:
    """
    Verify runtime-observed planner fields do not disqualify strict clean_negative formal-null mapping.

    Args:
        None.

    Returns:
        None.
    """
    record: Dict[str, Any] = {
        "sample_role": "clean_negative",
        "plan_digest": "runtime-observed-plan",
        "basis_digest": "runtime-observed-basis",
        "plan_input_digest": "runtime-observed-plan-input",
        "plan_input_schema_version": "v2",
        "plan_digest_expected": None,
        "plan_digest_status": "absent",
        "plan_digest_validation_status": "absent",
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner",
            "impl_version": "v2",
            "impl_digest": "planner-digest",
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
            "score": None,
            "content_chain_score": None,
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

    assert eval_workflow_inputs._is_strict_clean_negative_formal_null_record(record) is True
    assert eval_workflow_inputs._resolve_content_score_source(record) == (
        0.0,
        "strict_clean_negative_formal_null",
    )


def test_control_negative_statement_only_provenance_does_not_map_to_formal_null_content_score() -> None:
    """
    Verify control-negative statement-only provenance semantics stay excluded from strict clean formal-null mapping.

    Args:
        None.

    Returns:
        None.
    """
    record: Dict[str, Any] = {
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
        },
        "negative_branch_source_attestation_provenance": {
            "statement": {
                "plan_digest": "probe-plan",
            },
            "attestation_digest": "attestation-digest",
        },
        "content_evidence_payload": {
            "status": "absent",
            "score": None,
            "content_chain_score": None,
            "content_failure_reason": "detector_no_plan_expected",
        },
        "attestation": {
            "attestation_source": "negative_branch_statement_only_provenance",
            "authenticity_result": {
                "status": "statement_only",
                "bundle_status": "statement_only_provenance_no_bundle",
                "statement_status": "parsed",
            },
        },
    }

    assert eval_workflow_inputs._is_strict_clean_negative_formal_null_record(record) is False
    assert eval_workflow_inputs._resolve_content_score_source(record) == (None, None)


def _patch_pw02_python_stage_runner(monkeypatch: Any) -> List[Dict[str, Any]]:
    """
    Patch PW02 Python stage execution to lightweight artifact-writing stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Captured stage-call summaries.
    """

    def fake_clip_text_similarity(candidate_image: Any, prompt_text: str) -> float:
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError("prompt_text must be non-empty str")
        return 0.82 if prompt_text == "prompt one" else 0.74

    monkeypatch.setattr(
        pw_quality_metrics_module,
        "_compute_clip_text_similarity",
        fake_clip_text_similarity,
    )

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
            score_name = str(config_payload["calibration"]["score_name"])
            threshold_value = (
                float.fromhex("0x0.0000000000001p-1022")
                if score_name == pw02_module.CONTENT_SCORE_NAME
                else 0.5
            )
            thresholds_path = output_dir / "artifacts" / "thresholds" / "thresholds_artifact.json"
            ensure_directory(thresholds_path.parent)
            write_json_atomic(
                thresholds_path,
                {
                    "status": "ok",
                    "threshold_id": f"{score_name}_np_0p01",
                    "score_name": score_name,
                    "target_fpr": 0.01,
                    "threshold_value": threshold_value,
                    "threshold_key_used": "0p01",
                    "threshold_value_semantics": "strict_upper_bound",
                    "decision_operator": "score_greater_equal_threshold_value",
                    "selected_order_stat_score": 0.0,
                },
            )
            write_json_atomic(output_dir / "records" / "calibration_record.json", {"status": "ok"})
        elif module_name == "main.cli.run_evaluate":
            evaluate_cfg = cast(Dict[str, Any], config_payload.get("evaluate", {}))
            score_name = str(evaluate_cfg.get("score_name", config_payload["calibration"]["score_name"]))
            metrics_payload = {
                "score_name": score_name,
                "n_total": 4,
                "n_pos": 2,
                "n_neg": 2,
                "tpr_at_fpr_primary": 1.0,
                "fpr_empirical": 0.0,
            }
            breakdown_payload = {
                "confusion": {
                    "tp": 2,
                    "fp": 0,
                    "fn": 0,
                    "tn": 2,
                }
            }
            evaluation_report = {
                "status": "ok",
                "metrics": metrics_payload,
                "breakdown": breakdown_payload,
            }
            write_json_atomic(
                output_dir / "records" / "evaluate_record.json",
                {
                    "status": "ok",
                    "metrics": metrics_payload,
                    "evaluation_breakdown": breakdown_payload,
                    "evaluation_report": evaluation_report,
                },
            )
            write_json_atomic(output_dir / "artifacts" / "evaluation_report.json", evaluation_report)
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
    content_positive_record_path = Path(
        str(
            next(
                record["record_path"]
                for record in content_run["evaluate_inputs"]["records"]
                if record["sample_role"] == "positive_source"
            )
        )
    )
    content_positive_record = json.loads(content_positive_record_path.read_text(encoding="utf-8"))
    assert content_negative_record["paper_workflow_score_source"] == "strict_clean_negative_formal_null"
    assert str(content_negative_record["plan_digest"]).startswith("runtime-observed-plan-")
    assert str(content_negative_record["basis_digest"]).startswith("runtime-observed-basis-")
    assert str(content_negative_record["plan_input_digest"]).startswith("runtime-observed-plan-input-")
    assert content_negative_record["plan_input_schema_version"] == "v2"
    assert content_negative_record["subspace_planner_impl_identity"]["impl_id"] == "subspace_planner"
    assert content_negative_record["subspace_plan"]["planner_input_digest"] == content_negative_record["plan_input_digest"]
    assert content_positive_record["final_decision"] == {"is_watermarked": True}
    assert content_positive_record["formal_final_decision_source"] == "formal_threshold_overlay"
    assert content_positive_record["formal_final_decision"] == {
        "decision_origin": "formal_threshold_overlay",
        "decision_operator": "score_greater_equal_threshold_value",
        "decision_status": "decided",
        "is_watermarked": True,
        "score_name": pw02_module.CONTENT_SCORE_NAME,
        "score_source": "content_evidence_payload.content_chain_score",
        "score_value": 0.91,
        "selected_order_stat_score": 0.0,
        "target_fpr": 0.01,
        "threshold_key_used": "0p01",
        "threshold_source": "np_canonical",
        "used_threshold_id": "content_chain_score_np_0p01",
        "used_threshold_value": float.fromhex("0x0.0000000000001p-1022"),
    }
    assert content_negative_record["final_decision"] == {"is_watermarked": False}
    assert content_negative_record["formal_final_decision_source"] == "formal_threshold_overlay"
    assert content_negative_record["formal_final_decision"] == {
        "decision_origin": "formal_threshold_overlay",
        "decision_operator": "score_greater_equal_threshold_value",
        "decision_status": "decided",
        "is_watermarked": False,
        "score_name": pw02_module.CONTENT_SCORE_NAME,
        "score_source": "strict_clean_negative_formal_null",
        "score_value": 0.0,
        "selected_order_stat_score": 0.0,
        "target_fpr": 0.01,
        "threshold_key_used": "0p01",
        "threshold_source": "np_canonical",
        "used_threshold_id": "content_chain_score_np_0p01",
        "used_threshold_value": float.fromhex("0x0.0000000000001p-1022"),
    }

    formal_final_decision_metrics = cast(Dict[str, Any], pw02_summary["formal_final_decision_metrics"])
    derived_system_union_metrics = cast(Dict[str, Any], pw02_summary["derived_system_union_metrics"])
    assert formal_final_decision_metrics["n_total"] == 4
    assert formal_final_decision_metrics["n_positive"] == 2
    assert formal_final_decision_metrics["n_negative"] == 2
    assert formal_final_decision_metrics["content_chain_available_rate"] == 1.0
    assert formal_final_decision_metrics["final_decision_status_counts"] == {"decided": 4}
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
    content_clean_score_analysis = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["clean_score_analysis_exports"])["content"])).read_text(encoding="utf-8")
    )
    attestation_clean_score_analysis = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["clean_score_analysis_exports"])["attestation"])).read_text(encoding="utf-8")
    )
    content_roc_curve_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["roc_curve_paths"])["content_chain"])).read_text(encoding="utf-8")
    )
    attestation_roc_curve_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["roc_curve_paths"])["event_attestation"])).read_text(encoding="utf-8")
    )
    system_final_roc_curve_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["roc_curve_paths"])["system_final"])).read_text(encoding="utf-8")
    )
    system_final_auxiliary_roc_curve_export = json.loads(
        Path(str(cast(Dict[str, Any], pw02_summary["roc_curve_paths"])["system_final_auxiliary"])).read_text(encoding="utf-8")
    )
    system_final_auxiliary_operating_semantics = json.loads(
        Path(str(pw02_summary["system_final_auxiliary_operating_semantics_path"])).read_text(encoding="utf-8")
    )
    auc_summary = json.loads(Path(str(pw02_summary["auc_summary_path"])).read_text(encoding="utf-8"))
    eer_summary = json.loads(Path(str(pw02_summary["eer_summary_path"])).read_text(encoding="utf-8"))
    with Path(str(pw02_summary["tpr_at_target_fpr_summary_path"])).open("r", encoding="utf-8", newline="") as handle:
        tpr_rows = list(csv.DictReader(handle))
    quality_metrics_summary = json.loads(Path(str(pw02_summary["quality_metrics_summary_json_path"])).read_text(encoding="utf-8"))
    with Path(str(pw02_summary["quality_metrics_summary_csv_path"])).open("r", encoding="utf-8", newline="") as handle:
        quality_rows = list(csv.DictReader(handle))
    payload_clean_summary = json.loads(Path(str(pw02_summary["payload_clean_summary_path"])).read_text(encoding="utf-8"))
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
    assert positive_pool_manifest["expected_source_shard_count"] == 2
    assert positive_pool_manifest["discovered_source_shard_count"] == 2
    assert len(positive_pool_manifest["source_shard_manifest_paths"]) == 2

    assert clean_negative_pool_manifest["family_id"] == "family_pw02_exports"
    assert clean_negative_pool_manifest["source_role"] == "clean_negative"
    assert clean_negative_pool_manifest["event_count"] == 4
    assert len(clean_negative_pool_manifest["events"]) == 4
    assert clean_negative_pool_manifest["expected_source_shard_count"] == 2
    assert clean_negative_pool_manifest["discovered_source_shard_count"] == 2
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
    assert finalize_manifest["clean_score_analysis_exports"]["content"]["path"] == cast(Dict[str, Any], pw02_summary["clean_score_analysis_exports"])["content"]

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

    assert content_clean_score_analysis["score_name"] == pw02_module.CONTENT_SCORE_NAME
    assert content_clean_score_analysis["roc_auc"]["auc"] == pytest.approx(1.0)
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["count"] == 2
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["mean_psnr"] is not None
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["mean_ssim"] is not None
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["reference_artifact_name"] == "plain_preview_image"
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["candidate_artifact_name"] == "watermarked_output_image"
    assert content_clean_score_analysis["clean_positive_quality_metrics"]["reference_semantics"] == (
        "preview_generation_persisted_artifact_vs_watermarked_output_image"
    )
    quality_pair_rows = cast(
        List[Dict[str, Any]],
        content_clean_score_analysis["clean_positive_quality_metrics"]["pair_rows"],
    )
    assert len(quality_pair_rows) == 2
    assert all(isinstance(row.get("plain_preview_image_path"), str) and row["plain_preview_image_path"] for row in quality_pair_rows)
    assert all(
        isinstance(row.get("watermarked_output_image_path"), str) and row["watermarked_output_image_path"]
        for row in quality_pair_rows
    )
    assert {row["plain_preview_image_path"] for row in quality_pair_rows} == {
        normalize_path_value(Path(str(row["plain_preview_image_path"])).expanduser().resolve())
        for row in quality_pair_rows
    }
    assert {row["watermarked_output_image_path"] for row in quality_pair_rows} == {
        normalize_path_value(Path(str(row["watermarked_output_image_path"])).expanduser().resolve())
        for row in quality_pair_rows
    }

    assert attestation_clean_score_analysis["score_name"] == pw02_module.EVENT_ATTESTATION_SCORE_NAME
    assert attestation_clean_score_analysis["roc_auc"]["auc"] == pytest.approx(1.0)
    assert attestation_clean_score_analysis["clean_positive_quality_metrics"]["status"] == "not_applicable"

    assert set(cast(Dict[str, Any], pw02_summary["roc_curve_paths"]).keys()) == {
        "content_chain",
        "event_attestation",
        "system_final",
        "system_final_auxiliary",
    }
    assert Path(str(pw02_summary["operating_metrics_dir"])).is_dir()
    assert Path(str(pw02_summary["quality_metrics_dir"])).is_dir()
    assert Path(str(pw02_summary["payload_metrics_dir"])).is_dir()
    assert Path(str(pw02_summary["system_final_auxiliary_operating_semantics_path"])).exists()

    assert content_roc_curve_export["scope"] == "content_chain"
    assert content_roc_curve_export["status"] == "ok"
    assert content_roc_curve_export["auc"] == pytest.approx(1.0)
    assert content_roc_curve_export["source_analysis_path"] == cast(Dict[str, Any], pw02_summary["clean_score_analysis_exports"])["content"]
    assert content_roc_curve_export["roc_curve_points"]

    assert attestation_roc_curve_export["scope"] == "event_attestation"
    assert attestation_roc_curve_export["status"] == "ok"
    assert attestation_roc_curve_export["auc"] == pytest.approx(1.0)
    assert attestation_roc_curve_export["source_analysis_path"] == cast(Dict[str, Any], pw02_summary["clean_score_analysis_exports"])["attestation"]

    assert system_final_roc_curve_export["scope"] == "system_final"
    assert system_final_roc_curve_export["status"] == "not_available"
    assert "missing scalar score chain" in str(system_final_roc_curve_export["reason"])

    assert system_final_auxiliary_roc_curve_export["scope"] == "system_final_auxiliary"
    assert system_final_auxiliary_roc_curve_export["status"] == "ok"
    assert system_final_auxiliary_roc_curve_export["auc"] == pytest.approx(1.0)
    assert system_final_auxiliary_roc_curve_export["source_analysis_path"] == pw02_summary["system_final_auxiliary_operating_semantics_path"]
    assert system_final_auxiliary_operating_semantics["scope"] == "system_final_auxiliary"
    assert system_final_auxiliary_operating_semantics["canonical"] is False
    assert system_final_auxiliary_operating_semantics["analysis_only"] is True
    assert system_final_auxiliary_operating_semantics["decision_equivalence"]["status"] == "exact_match"
    assert system_final_auxiliary_operating_semantics["decision_equivalence"]["mismatch_count"] == 0
    assert system_final_auxiliary_operating_semantics["operating_metrics"]["threshold_value"] == pytest.approx(0.0)

    auc_rows_by_scope = {row["scope"]: row for row in cast(List[Dict[str, Any]], auc_summary["rows"])}
    assert set(auc_rows_by_scope.keys()) == {"content_chain", "event_attestation", "system_final", "system_final_auxiliary"}
    assert auc_rows_by_scope["content_chain"]["auc"] == pytest.approx(1.0)
    assert auc_rows_by_scope["event_attestation"]["auc"] == pytest.approx(1.0)
    assert auc_rows_by_scope["system_final"]["status"] == "not_available"
    assert auc_rows_by_scope["system_final_auxiliary"]["status"] == "ok"
    assert auc_rows_by_scope["system_final_auxiliary"]["auc"] == pytest.approx(1.0)

    eer_rows_by_scope = {row["scope"]: row for row in cast(List[Dict[str, Any]], eer_summary["rows"])}
    assert set(eer_rows_by_scope.keys()) == {"content_chain", "event_attestation", "system_final", "system_final_auxiliary"}
    assert eer_rows_by_scope["content_chain"]["status"] == "ok"
    assert eer_rows_by_scope["content_chain"]["eer"] == pytest.approx(0.0)
    assert eer_rows_by_scope["event_attestation"]["status"] == "ok"
    assert eer_rows_by_scope["system_final"]["status"] == "not_available"
    assert eer_rows_by_scope["system_final_auxiliary"]["status"] == "ok"
    assert eer_rows_by_scope["system_final_auxiliary"]["eer"] == pytest.approx(0.0)

    assert len(tpr_rows) == 16
    content_tpr_rows = [row for row in tpr_rows if row["scope"] == "content_chain"]
    system_tpr_rows = [row for row in tpr_rows if row["scope"] == "system_final"]
    auxiliary_tpr_rows = [row for row in tpr_rows if row["scope"] == "system_final_auxiliary"]
    assert {row["target_fpr"] for row in content_tpr_rows} == {"0.01", "0.001", "0.0001", "1e-05"}
    assert all(row["status"] == "ok" for row in content_tpr_rows)
    assert all(float(row["tpr"]) == pytest.approx(1.0) for row in content_tpr_rows)
    assert all(row["status"] == "not_available" for row in system_tpr_rows)
    assert all(row["status"] == "ok" for row in auxiliary_tpr_rows)
    assert all(float(row["tpr"]) == pytest.approx(1.0) for row in auxiliary_tpr_rows)

    quality_rows_by_scope = {row["scope"]: row for row in cast(List[Dict[str, Any]], quality_metrics_summary["rows"])}
    assert set(quality_rows_by_scope.keys()) == {"content_chain", "event_attestation", "system_final"}
    assert len(quality_rows) == 3
    assert quality_rows_by_scope["content_chain"]["status"] == "ok"
    assert quality_rows_by_scope["content_chain"]["pair_count"] == 2
    assert quality_rows_by_scope["content_chain"]["mean_psnr"] is not None
    assert quality_rows_by_scope["content_chain"]["mean_ssim"] is not None
    assert quality_rows_by_scope["content_chain"]["lpips_status"] in {"ok", "not_available"}
    if quality_rows_by_scope["content_chain"]["lpips_status"] == "ok":
        assert quality_rows_by_scope["content_chain"]["mean_lpips"] is not None
    else:
        assert quality_rows_by_scope["content_chain"]["lpips_reason"]
    assert quality_rows_by_scope["content_chain"]["clip_status"] == "ok"
    assert quality_rows_by_scope["content_chain"]["mean_clip_text_similarity"] is not None
    assert quality_rows_by_scope["content_chain"]["clip_model_name"] == pw_quality_metrics_module.CLIP_MODEL_NAME
    assert quality_rows_by_scope["content_chain"]["clip_sample_count"] == 2
    assert quality_rows_by_scope["event_attestation"]["status"] == "not_applicable"
    assert quality_rows_by_scope["event_attestation"]["lpips_status"] == quality_rows_by_scope["content_chain"]["lpips_status"]
    assert quality_rows_by_scope["system_final"]["status"] == "not_available"
    assert quality_rows_by_scope["system_final"]["lpips_status"] == quality_rows_by_scope["content_chain"]["lpips_status"]
    assert "quality payload is only defined" in str(quality_rows_by_scope["system_final"]["reason"])
    assert "mean_clip_text_similarity" in quality_rows[0]
    assert "clip_model_name" in quality_rows[0]
    assert "clip_sample_count" in quality_rows[0]

    assert payload_clean_summary["status"] == "ok"
    assert payload_clean_summary["reason"] is None
    assert payload_clean_summary["future_upstream_sidecar_required"] is False
    assert payload_clean_summary["overall"]["event_count"] == 2
    assert payload_clean_summary["overall"]["available_payload_event_count"] == 2
    assert payload_clean_summary["overall"]["missing_payload_event_count"] == 0
    assert payload_clean_summary["overall"]["mean_codeword_agreement"] == pytest.approx(0.87)
    assert payload_clean_summary["overall"]["min_codeword_agreement"] == pytest.approx(0.84)
    assert payload_clean_summary["overall"]["max_codeword_agreement"] == pytest.approx(0.90)
    assert payload_clean_summary["overall"]["mean_n_bits_compared"] == pytest.approx(96.0)
    assert payload_clean_summary["overall"]["mean_bit_accuracy"] == pytest.approx(0.87)
    assert payload_clean_summary["overall"]["weighted_bit_accuracy"] == pytest.approx(0.87)
    assert payload_clean_summary["overall"]["mean_bit_error_rate"] == pytest.approx(0.13)
    assert payload_clean_summary["overall"]["weighted_bit_error_rate"] == pytest.approx(0.13)
    assert payload_clean_summary["overall"]["message_success_count"] == 0
    assert payload_clean_summary["overall"]["message_success_rate"] == pytest.approx(0.0)
    assert payload_clean_summary["overall"]["payload_primary_metric_sources"] == ["codeword_agreement_and_n_bits_compared"]
    assert payload_clean_summary["overall"]["attested_event_count"] == 2
    assert payload_clean_summary["overall"]["mean_event_attestation_score"] == pytest.approx(0.81)
    assert payload_clean_summary["overall"]["lf_detect_variants"] == ["correlation_v2"]
    assert payload_clean_summary["overall"]["message_sources"] == ["plan_digest"]
    assert len(payload_clean_summary["rows"]) == 2
    assert cast(Dict[str, Any], pw02_summary["analysis_only_artifact_paths"])["pw02_system_final_auxiliary_operating_semantics"] == pw02_summary["system_final_auxiliary_operating_semantics_path"]
    assert cast(Dict[str, Any], pw02_summary["analysis_only_artifact_paths"])["pw02_system_final_auxiliary_roc_curve"] == cast(Dict[str, Any], pw02_summary["roc_curve_paths"])["system_final_auxiliary"]

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


def test_pw02_payload_clean_summary_prefers_decode_sidecar_metrics(tmp_path: Path) -> None:
    """
    Verify PW02 payload clean summary prefers decode sidecar metrics over legacy LF trace values.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    payload_decode_sidecar_path = tmp_path / "payload_decode_sidecar.json"
    prepared_record_path = tmp_path / "prepared_record.json"

    write_json_atomic(
        payload_decode_sidecar_path,
        {
            "artifact_type": "paper_workflow_payload_decode_sidecar",
            "schema_version": "pw_payload_sidecar_v1",
            "event_id": "event_000001",
            "sample_role": "positive_source",
            "reference_event_id": "event_000001",
            "message_source": "sidecar_source",
            "lf_detect_variant": "sidecar_variant",
            "n_bits_compared": 96,
            "bit_error_count": 72,
            "codeword_agreement": 0.25,
            "message_decode_success": False,
        },
    )
    write_json_atomic(
        prepared_record_path,
        {
            "sample_role": "positive_source",
            "paper_workflow_event_id": "event_000001",
            "paper_workflow_event_index": 1,
            "paper_workflow_payload_decode_sidecar_path": normalize_path_value(payload_decode_sidecar_path),
            "content_evidence_payload": {
                "score_parts": {
                    "lf_trajectory_detect_trace": {
                        "codeword_agreement": 0.91,
                        "n_bits_compared": 64,
                        "detect_variant": "trace_variant",
                        "message_source": "trace_source",
                    }
                }
            },
            "attestation": {
                "final_event_attested_decision": {
                    "event_attestation_score": 0.5,
                    "is_event_attested": True,
                }
            },
        },
    )

    payload_clean_summary = pw02_metrics_extensions_module._build_payload_clean_summary_payload(
        family_id="family_payload_sidecar_pw02",
        score_runs={
            pw02_module.CONTENT_SCORE_NAME: {
                "evaluate_inputs": {
                    "records": [
                        {
                            "record_path": normalize_path_value(prepared_record_path),
                        }
                    ]
                }
            }
        },
    )

    assert payload_clean_summary["status"] == "ok"
    assert payload_clean_summary["overall"]["mean_codeword_agreement"] == pytest.approx(0.25)
    assert payload_clean_summary["overall"]["mean_n_bits_compared"] == pytest.approx(96.0)
    assert payload_clean_summary["overall"]["payload_primary_metric_sources"] == ["codeword_agreement_and_n_bits_compared"]
    assert payload_clean_summary["overall"]["lf_detect_variants"] == ["sidecar_variant"]
    assert payload_clean_summary["overall"]["message_sources"] == ["sidecar_source"]
    assert payload_clean_summary["rows"][0]["codeword_agreement"] == pytest.approx(0.25)
    assert payload_clean_summary["rows"][0]["payload_decode_sidecar_path"] == normalize_path_value(payload_decode_sidecar_path)


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