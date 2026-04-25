"""
File purpose: Validate PW01 bound config source selection and runtime binding preservation.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from paper_workflow.scripts.pw00_build_family_manifest import run_pw00_build_family_manifest
import paper_workflow.scripts.pw01_run_source_event_shard as pw01_module
from paper_workflow.scripts.pw_common import read_jsonl
from scripts.notebook_runtime_common import (
    apply_notebook_model_snapshot_binding,
    ensure_directory,
    load_yaml_mapping,
    write_json_atomic,
    write_yaml_mapping,
)


def _build_pw00_family(tmp_path: Path, family_id: str) -> Dict[str, Any]:
    """
    Build a minimal PW00 family fixture for PW01 bound-config tests.

    Args:
        tmp_path: Pytest temporary directory.
        family_id: Fixture family identifier.

    Returns:
        PW00 summary payload.
    """
    prompt_file = tmp_path / "pw01_bound_prompts.txt"
    prompt_file.write_text("prompt one\nprompt two\n", encoding="utf-8")
    return run_pw00_build_family_manifest(
        drive_project_root=tmp_path / "drive",
        family_id=family_id,
        prompt_file=str(prompt_file),
        seed_list=[3, 9],
        source_shard_count=2,
    )


def _write_bound_config_snapshot(
    drive_project_root: Path,
    *,
    marker: str,
) -> tuple[Path, Path]:
    """
    Build a notebook-style bound config snapshot and its model snapshot root.

    Args:
        drive_project_root: Drive project root.
        marker: Stable marker stored in the config.

    Returns:
        Tuple of bound config path and model snapshot directory.
    """
    snapshot_dir = drive_project_root / "runtime_state" / f"{marker}_model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    bound_cfg = apply_notebook_model_snapshot_binding(
        load_yaml_mapping((pw01_module.REPO_ROOT / "configs" / "default.yaml").resolve()),
        env_mapping={"CEG_WM_MODEL_SNAPSHOT_PATH": snapshot_dir.as_posix()},
    )
    bound_cfg["test_config_origin"] = marker

    bound_config_path = drive_project_root / "runtime_state" / f"{marker}_bound_config.yaml"
    write_yaml_mapping(bound_config_path, bound_cfg)
    return bound_config_path, snapshot_dir


def _write_unbound_default_config(tmp_path: Path, *, marker: str) -> Path:
    """
    Build a contrasting unbound default config for lineage checks.

    Args:
        tmp_path: Pytest temporary directory.
        marker: Stable marker stored in the config.

    Returns:
        Unbound default config path.
    """
    config_path = tmp_path / f"{marker}_default_config.yaml"
    cfg_obj = load_yaml_mapping((pw01_module.REPO_ROOT / "configs" / "default.yaml").resolve())
    cfg_obj["test_config_origin"] = marker
    cfg_obj.pop("model_snapshot_path", None)
    cfg_obj.pop("model_source_binding", None)
    write_yaml_mapping(config_path, cfg_obj)
    return config_path


def _load_shard_assigned_events(summary: Dict[str, Any], shard_index: int) -> List[Dict[str, Any]]:
    """
    Load the ordered assigned events for one shard from PW00 outputs.

    Args:
        summary: PW00 summary payload.
        shard_index: Shard index.

    Returns:
        Ordered assigned event payloads.
    """
    shard_plan = json.loads(Path(str(summary["source_shard_plan_path"])).read_text(encoding="utf-8"))
    shard_assignment = pw01_module.resolve_positive_shard_assignment(
        shard_plan,
        shard_index=shard_index,
        shard_count=2,
    )
    event_lookup = {
        row["event_id"]: row
        for row in read_jsonl(Path(str(summary["source_event_grid_path"])))
    }
    return [
        cast(Dict[str, Any], event_lookup[event_id])
        for event_id in cast(List[str], shard_assignment["assigned_event_ids"])
    ]


def _patch_pw01_base_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_snapshot_path: Path,
    persistent_runtime: bool = False,
) -> Dict[str, Any]:
    """
    Patch PW01 base-runner calls with lightweight stubs and capture runtime cfgs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        expected_snapshot_path: Expected notebook-bound model snapshot path.

    Returns:
        Mutable capture mapping.
    """
    captures: Dict[str, Any] = {"preview_cfgs": []}
    if persistent_runtime:
        captures["embed_runtime_sessions"] = []
        captures["detect_runtime_sessions"] = []
        captures["embed_runtime_calls"] = []
        captures["detect_runtime_calls"] = []

    def fake_build_stage_command(stage_name: str, config_path: Path, run_root: Path) -> list[str]:
        return [stage_name, str(config_path), str(run_root)]

    def fake_prepare_source_pool_preview_artifact(
        *,
        cfg_obj: Dict[str, Any],
        prompt_run_root: Path,
        prompt_text: str,
        prompt_index: int,
        prompt_file_path: str,
    ) -> Dict[str, Any]:
        captures["preview_cfgs"].append(dict(cfg_obj))
        assert cfg_obj["model_snapshot_path"] == expected_snapshot_path.resolve().as_posix()
        assert cfg_obj["model_source_binding"]["binding_status"] == "bound"

        preview_image_path = prompt_run_root / "artifacts" / "preview" / "preview.png"
        ensure_directory(preview_image_path.parent)
        preview_image_path.write_bytes(b"preview")

        preview_record_path = prompt_run_root / "artifacts" / "preview_generation_record.json"
        preview_record: Dict[str, Any] = {
            "status": "ok",
            "prompt_text": prompt_text,
            "prompt_index": prompt_index,
            "prompt_file": prompt_file_path,
            "persisted_artifact_path": preview_image_path.as_posix(),
        }
        write_json_atomic(preview_record_path, preview_record)

        runtime_cfg: Dict[str, Any] = dict(cfg_obj)
        embed_node = runtime_cfg.get("embed")
        embed_cfg: Dict[str, Any]
        if isinstance(embed_node, dict):
            embed_cfg = cast(Dict[str, Any], embed_node.copy())
        else:
            embed_cfg = {}
        embed_cfg["input_image_path"] = preview_image_path.as_posix()
        runtime_cfg["embed"] = embed_cfg
        return {
            "runtime_cfg": runtime_cfg,
            "preview_record": preview_record,
        }

    def fake_run_stage(stage_name: str, command: list[str], run_root: Path) -> Dict[str, Any]:
        records_root = ensure_directory(run_root / "records")
        logs_root = ensure_directory(run_root / "logs")
        (logs_root / f"{stage_name}_stdout.log").write_text("ok\n", encoding="utf-8")
        (logs_root / f"{stage_name}_stderr.log").write_text("\n", encoding="utf-8")

        if stage_name == "embed":
            write_json_atomic(
                records_root / "embed_record.json",
                {
                    "record_type": "embed",
                    "content_evidence": {
                        "status": "ok",
                        "plan_digest": "plan_digest_pw01_test",
                        "basis_digest": "basis_digest_pw01_test",
                        "score_parts": {
                            "lf_metrics": {
                                "message_length": 8,
                                "ecc_sparsity": 3,
                                "plan_digest": "plan_digest_pw01_test",
                                "basis_digest": "basis_digest_pw01_test",
                                "message_source": "attestation_event_digest",
                                "parity_check_digest": "parity_check_digest_pw01_test",
                            }
                        },
                    },
                    "attestation": {
                        "event_binding_digest": "event_binding_digest_pw01_test",
                        "event_binding_mode": "trajectory_bound",
                        "lf_payload_hex": "ab",
                    },
                },
            )
        elif stage_name == "detect":
            write_json_atomic(
                records_root / "detect_record.json",
                {
                    "record_type": "detect",
                    "content_evidence_payload": {
                        "status": "ok",
                        "score": 0.75,
                        "content_chain_score": 0.75,
                        "score_parts": {
                            "lf_trajectory_detect_trace": {
                                "codeword_agreement": 1.0,
                                "n_bits_compared": 8,
                                "detect_variant": "correlation_v2",
                                "message_source": "attestation_event_digest",
                            }
                        },
                    },
                    "attestation": {
                        "_lf_attestation_trace_artifact": {
                            "mismatch_indices": [],
                            "n_bits_compared": 8,
                            "agreement_count": 8,
                        },
                        "final_event_attested_decision": {
                            "event_attestation_score_name": "event_attestation_score",
                            "event_attestation_score": 0.61,
                        }
                    },
                    "final_decision": {
                        "decision_status": "abstain",
                        "is_watermarked": None,
                    },
                },
            )
        else:
            raise ValueError(f"unsupported stage: {stage_name}")

        return {
            "return_code": 0,
            "command": command,
            "stdout_log_path": (logs_root / f"{stage_name}_stdout.log").as_posix(),
            "stderr_log_path": (logs_root / f"{stage_name}_stderr.log").as_posix(),
            "status": "ok",
        }

    def fake_resolve_source_pool_attestation_views(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        artifact_root = ensure_directory(run_root / "artifacts" / "mock_attestation" / f"event_{prompt_index:06d}")
        statement_path = artifact_root / "attestation_statement.json"
        bundle_path = artifact_root / "attestation_bundle.json"
        result_path = artifact_root / "attestation_result.json"
        write_json_atomic(statement_path, {"status": "ok"})
        write_json_atomic(bundle_path, {"status": "ok"})
        write_json_atomic(result_path, {"status": "ok"})
        return {
            "attestation_statement": {
                "exists": True,
                "path": statement_path.as_posix(),
                "package_relative_path": statement_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
            "attestation_bundle": {
                "exists": True,
                "path": bundle_path.as_posix(),
                "package_relative_path": bundle_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
            "attestation_result": {
                "exists": True,
                "path": result_path.as_posix(),
                "package_relative_path": result_path.relative_to(run_root).as_posix(),
                "missing_reason": None,
            },
        }

    def fake_resolve_source_pool_source_image_view(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        source_image_path = run_root / "artifacts" / "mock_source_images" / f"event_{prompt_index:06d}.png"
        ensure_directory(source_image_path.parent)
        source_image_path.write_bytes(b"img")
        return {
            "exists": True,
            "path": source_image_path.as_posix(),
            "package_relative_path": source_image_path.relative_to(run_root).as_posix(),
            "missing_reason": None,
        }

    def fake_resolve_source_pool_preview_generation_record_view(
        *,
        cfg_obj: Dict[str, Any],
        run_root: Path,
        prompt_run_root: Path,
        prompt_index: int,
    ) -> Dict[str, Any]:
        record_path = run_root / "artifacts" / "mock_preview_records" / f"event_{prompt_index:06d}.json"
        ensure_directory(record_path.parent)
        write_json_atomic(record_path, {"status": "ok"})
        return {
            "exists": True,
            "path": record_path.as_posix(),
            "package_relative_path": record_path.relative_to(run_root).as_posix(),
            "missing_reason": None,
        }

    def fake_normalize_direct_detect_payload(
        payload: Dict[str, Any],
        *,
        prompt_text: str,
        prompt_index: int,
        prompt_file_path: str,
        record_usage: str,
    ) -> Dict[str, Any]:
        payload = dict(payload)
        payload["label"] = True
        payload["prompt_text"] = prompt_text
        payload["prompt_index"] = prompt_index
        payload["prompt_file"] = prompt_file_path
        payload["record_usage"] = record_usage
        return payload

    def fake_build_embed_runtime_session(config_path: str, overrides: Any = None) -> Dict[str, Any]:
        session = {
            "session_kind": "embed",
            "config_path": config_path,
            "overrides": list(cast(List[str], overrides or [])),
        }
        captures["embed_runtime_sessions"].append(session)
        return session

    def fake_build_detect_runtime_session(
        config_path: str,
        overrides: Any = None,
        thresholds_path: Any = None,
    ) -> Dict[str, Any]:
        _ = thresholds_path
        session = {
            "session_kind": "detect",
            "config_path": config_path,
            "overrides": list(cast(List[str], overrides or [])),
        }
        captures["detect_runtime_sessions"].append(session)
        return session

    def fake_run_embed(
        output_dir: str,
        config_path: str,
        overrides: Any = None,
        input_image_path: Any = None,
        runtime_session: Any = None,
    ) -> None:
        _ = input_image_path
        captures["embed_runtime_calls"].append(
            {
                "output_dir": output_dir,
                "config_path": config_path,
                "overrides": list(cast(List[str], overrides or [])),
                "runtime_session_id": id(runtime_session),
            }
        )
        records_root = ensure_directory(Path(output_dir) / "records")
        write_json_atomic(
            records_root / "embed_record.json",
            {
                "record_type": "embed",
                "content_evidence": {
                    "status": "ok",
                    "plan_digest": "plan_digest_pw01_test",
                    "basis_digest": "basis_digest_pw01_test",
                    "score_parts": {
                        "lf_metrics": {
                            "message_length": 8,
                            "ecc_sparsity": 3,
                            "plan_digest": "plan_digest_pw01_test",
                            "basis_digest": "basis_digest_pw01_test",
                            "message_source": "attestation_event_digest",
                            "parity_check_digest": "parity_check_digest_pw01_test",
                        }
                    },
                },
                "attestation": {
                    "event_binding_digest": "event_binding_digest_pw01_test",
                    "event_binding_mode": "trajectory_bound",
                    "lf_payload_hex": "ab",
                },
            },
        )

    def fake_run_detect(
        output_dir: str,
        config_path: str,
        input_record_path: Any = None,
        overrides: Any = None,
        thresholds_path: Any = None,
        runtime_session: Any = None,
    ) -> None:
        _ = input_record_path
        _ = thresholds_path
        captures["detect_runtime_calls"].append(
            {
                "output_dir": output_dir,
                "config_path": config_path,
                "overrides": list(cast(List[str], overrides or [])),
                "runtime_session_id": id(runtime_session),
            }
        )
        records_root = ensure_directory(Path(output_dir) / "records")
        write_json_atomic(
            records_root / "detect_record.json",
            {
                "record_type": "detect",
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.75,
                    "content_chain_score": 0.75,
                    "score_parts": {
                        "lf_trajectory_detect_trace": {
                            "codeword_agreement": 1.0,
                            "n_bits_compared": 8,
                            "detect_variant": "correlation_v2",
                            "message_source": "attestation_event_digest",
                        }
                    },
                },
                "attestation": {
                    "_lf_attestation_trace_artifact": {
                        "mismatch_indices": [],
                        "n_bits_compared": 8,
                        "agreement_count": 8,
                    },
                    "final_event_attested_decision": {
                        "event_attestation_score_name": "event_attestation_score",
                        "event_attestation_score": 0.61,
                    }
                },
                "final_decision": {
                    "decision_status": "abstain",
                    "is_watermarked": None,
                },
            },
        )

    def fail_run_stage(stage_name: str, command: list[str], run_root: Path) -> Dict[str, Any]:
        raise AssertionError(
            f"subprocess _run_stage should not be used in persistent runtime mode: {stage_name}"
        )

    monkeypatch.setattr(pw01_module.BASE_RUNNER_MODULE, "_build_stage_command", fake_build_stage_command)
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_prepare_source_pool_preview_artifact",
        fake_prepare_source_pool_preview_artifact,
    )
    monkeypatch.setattr(pw01_module.BASE_RUNNER_MODULE, "_run_stage", fake_run_stage)
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_attestation_views",
        fake_resolve_source_pool_attestation_views,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_source_image_view",
        fake_resolve_source_pool_source_image_view,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_resolve_source_pool_preview_generation_record_view",
        fake_resolve_source_pool_preview_generation_record_view,
    )
    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_normalize_direct_detect_payload",
        fake_normalize_direct_detect_payload,
    )
    if persistent_runtime:
        monkeypatch.setattr(pw01_module.BASE_RUNNER_MODULE, "_run_stage", fail_run_stage)
        monkeypatch.setattr(pw01_module, "build_embed_runtime_session", fake_build_embed_runtime_session)
        monkeypatch.setattr(pw01_module, "build_detect_runtime_session", fake_build_detect_runtime_session)
        monkeypatch.setattr(pw01_module, "run_embed", fake_run_embed)
        monkeypatch.setattr(pw01_module, "run_detect", fake_run_detect)
    else:
        monkeypatch.setattr(pw01_module, "_build_pw01_stage_runtime_bundle", lambda **_: None)
    return captures


def _rewrite_family_default_config(summary: Dict[str, Any], default_config_path: Path) -> None:
    """
    Rewrite the family manifest default_config_path for lineage assertions.

    Args:
        summary: PW00 summary payload.
        default_config_path: Replacement default config path.

    Returns:
        None.
    """
    family_manifest_path = Path(str(summary["paper_eval_family_manifest_path"]))
    family_manifest = json.loads(family_manifest_path.read_text(encoding="utf-8"))
    family_manifest["default_config_path"] = default_config_path.resolve().as_posix()
    family_manifest_path.write_text(json.dumps(family_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def test_pw01_requires_bound_config_path_when_family_exists(tmp_path: Path) -> None:
    """
    Reject notebook-driven PW01 runs that do not provide a bound config path.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    _build_pw00_family(tmp_path, family_id="family_missing_bound_config")

    with pytest.raises(ValueError, match="bound_config_path"):
        pw01_module.run_pw01_source_event_shard(
            drive_project_root=tmp_path / "drive",
            family_id="family_missing_bound_config",
            shard_index=0,
            shard_count=2,
        )


def test_pw01_uses_bound_config_path_as_runtime_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Prefer the notebook-bound config snapshot over the family default config.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_bound_source")
    fallback_default_config = _write_unbound_default_config(tmp_path, marker="family_default")
    _rewrite_family_default_config(summary, fallback_default_config)
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="bound_source",
    )
    captures = _patch_pw01_base_runner(monkeypatch, expected_snapshot_path=snapshot_dir)

    pw01_summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_bound_source",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    shard_root = Path(str(pw01_summary["shard_root"]))
    shard_manifest = json.loads((shard_root / "shard_manifest.json").read_text(encoding="utf-8"))
    event_runtime_cfg = load_yaml_mapping(shard_root / "events" / "event_000000" / "runtime_config.yaml")

    assert shard_manifest["default_config_path"] == fallback_default_config.resolve().as_posix()
    assert shard_manifest["bound_config_path"] == bound_config_path.resolve().as_posix()
    assert event_runtime_cfg["test_config_origin"] == "bound_source"
    assert event_runtime_cfg["model_snapshot_path"] == snapshot_dir.resolve().as_posix()
    assert len(captures["preview_cfgs"]) == 2


def test_pw01_worker_plan_persists_and_worker_loads_bound_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Persist bound_config_path into worker plans and load it in worker execution.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_worker_bound_config")
    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    default_config_path = pw01_module._resolve_default_config_path(family_manifest)
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="worker_bound",
    )
    _patch_pw01_base_runner(monkeypatch, expected_snapshot_path=snapshot_dir)

    family_root = Path(str(summary["family_root"]))
    shard_root = ensure_directory(pw01_module._build_shard_root(family_root, 0))
    worker_plans = pw01_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id="family_worker_bound_config",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=1,
        shard_root=shard_root,
        default_config_path=default_config_path,
        bound_config_path=bound_config_path,
        assigned_events=_load_shard_assigned_events(summary, 0),
    )

    worker_plan_path = Path(str(worker_plans[0]["worker_plan_path"]))
    worker_plan = json.loads(worker_plan_path.read_text(encoding="utf-8"))
    assert worker_plan["bound_config_path"] == bound_config_path.resolve().as_posix()

    worker_result = pw01_module.run_pw01_source_event_shard_worker(
        drive_project_root=tmp_path / "drive",
        family_id="family_worker_bound_config",
        shard_index=0,
        pw01_worker_count=1,
        local_worker_index=0,
        worker_plan_path=worker_plan_path,
    )

    first_event = cast(Dict[str, Any], worker_result["events"][0])
    runtime_cfg = load_yaml_mapping(Path(str(first_event["runtime_config_path"])))
    assert runtime_cfg["test_config_origin"] == "worker_bound"
    assert runtime_cfg["model_snapshot_path"] == snapshot_dir.resolve().as_posix()


def test_pw01_positive_source_worker_result_includes_persistent_runtime_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Persist worker-level persistent-runtime diagnostics for positive_source execution.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    summary = _build_pw00_family(tmp_path, family_id="family_worker_runtime_positive")
    family_manifest = json.loads(Path(str(summary["paper_eval_family_manifest_path"])).read_text(encoding="utf-8"))
    default_config_path = pw01_module._resolve_default_config_path(family_manifest)
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="worker_runtime_positive",
    )
    captures = _patch_pw01_base_runner(
        monkeypatch,
        expected_snapshot_path=snapshot_dir,
        persistent_runtime=True,
    )
    perf_counter_values = iter([10.0, 12.5, 20.0, 23.0])
    monkeypatch.setattr(pw01_module.time, "perf_counter", lambda: next(perf_counter_values))

    family_root = Path(str(summary["family_root"]))
    shard_root = ensure_directory(pw01_module._build_shard_root(family_root, 0))
    worker_plans = pw01_module._prepare_local_worker_plans(
        drive_project_root=tmp_path / "drive",
        family_id="family_worker_runtime_positive",
        sample_role="positive_source",
        shard_index=0,
        shard_count=2,
        pw01_worker_count=1,
        shard_root=shard_root,
        default_config_path=default_config_path,
        bound_config_path=bound_config_path,
        assigned_events=_load_shard_assigned_events(summary, 0),
    )

    worker_result = pw01_module.run_pw01_source_event_shard_worker(
        drive_project_root=tmp_path / "drive",
        family_id="family_worker_runtime_positive",
        shard_index=0,
        pw01_worker_count=1,
        local_worker_index=0,
        worker_plan_path=Path(str(worker_plans[0]["worker_plan_path"])),
    )

    assert len(captures["embed_runtime_sessions"]) == 1
    assert len(captures["detect_runtime_sessions"]) == 1
    assert worker_result["persistent_stage_worker_enabled"] is True
    assert worker_result["embed_runtime_session_enabled"] is True
    assert worker_result["detect_runtime_session_enabled"] is True
    assert worker_result["worker_embed_runtime_init_elapsed_seconds"] == 2.5
    assert worker_result["worker_detect_runtime_init_elapsed_seconds"] == 3.0
    assert worker_result["worker_embed_event_count"] == worker_result["completed_event_count"]
    assert worker_result["worker_detect_event_count"] == worker_result["completed_event_count"]
    assert worker_result["recommended_pw01_worker_count_for_validation"] == 1


def test_pw01_positive_source_single_process_uses_persistent_stage_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reuse shard-local embed and detect runtime sessions in single-process PW01 execution.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _build_pw00_family(tmp_path, family_id="family_positive_persistent_runtime")
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(
        tmp_path / "drive",
        marker="positive_persistent_runtime",
    )
    captures = _patch_pw01_base_runner(
        monkeypatch,
        expected_snapshot_path=snapshot_dir,
        persistent_runtime=True,
    )

    summary = pw01_module.run_pw01_source_event_shard(
        drive_project_root=tmp_path / "drive",
        family_id="family_positive_persistent_runtime",
        shard_index=0,
        shard_count=2,
        bound_config_path=bound_config_path,
    )

    shard_manifest = json.loads(
        (Path(str(summary["shard_root"])) / "shard_manifest.json").read_text(encoding="utf-8")
    )
    event_count = len(cast(List[Dict[str, Any]], shard_manifest["events"]))
    first_event = cast(Dict[str, Any], shard_manifest["events"][0])

    assert len(captures["embed_runtime_sessions"]) == 1
    assert len(captures["detect_runtime_sessions"]) == 1
    assert len(captures["embed_runtime_calls"]) == event_count
    assert len(captures["detect_runtime_calls"]) == event_count
    assert {call["runtime_session_id"] for call in captures["embed_runtime_calls"]} == {
        id(captures["embed_runtime_sessions"][0])
    }
    assert {call["runtime_session_id"] for call in captures["detect_runtime_calls"]} == {
        id(captures["detect_runtime_sessions"][0])
    }
    assert first_event["stage_results"]["embed"]["execution_mode"] == "persistent_worker_runtime"
    assert first_event["stage_results"]["detect"]["execution_mode"] == "persistent_worker_runtime"
    assert Path(str(first_event["stage_results"]["embed"]["stdout_log_path"])).exists()
    assert Path(str(first_event["stage_results"]["detect"]["stdout_log_path"])).exists()


def test_run_positive_source_event_preserves_model_snapshot_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve model_snapshot_path and model_source_binding in event runtime cfg.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(tmp_path / "drive", marker="event_bound")
    bound_cfg_obj = load_yaml_mapping(bound_config_path)
    _patch_pw01_base_runner(monkeypatch, expected_snapshot_path=snapshot_dir)
    monkeypatch.delenv("CEG_WM_MODEL_SNAPSHOT_PATH", raising=False)

    event_manifest = pw01_module._run_positive_source_event(
        event={
            "event_id": "evt_bound_0000",
            "event_index": 0,
            "sample_role": "positive_source",
            "source_prompt_index": 0,
            "prompt_text": "prompt one",
            "prompt_sha256": "sha256-prompt-one",
            "seed": 3,
            "prompt_file": "prompts/paper_small.txt",
        },
        shard_root=ensure_directory(tmp_path / "drive" / "paper_workflow" / "families" / "family_event_bound" / "source_shards" / "positive" / "shard_0000"),
        default_cfg_obj=bound_cfg_obj,
        bound_config_path=bound_config_path,
    )

    runtime_cfg = load_yaml_mapping(Path(str(event_manifest["runtime_config_path"])))
    prompt_run_root = Path(str(event_manifest["runtime_config_path"])).parent / "run"
    embed_record = json.loads((prompt_run_root / "records" / "embed_record.json").read_text(encoding="utf-8"))
    detect_record = json.loads((prompt_run_root / "records" / "detect_record.json").read_text(encoding="utf-8"))
    assert runtime_cfg["model_snapshot_path"] == snapshot_dir.resolve().as_posix()
    assert runtime_cfg["model_source_binding"]["binding_status"] == "bound"
    assert runtime_cfg["test_config_origin"] == "event_bound"
    assert embed_record["record_type"] == "embed"
    assert detect_record["content_evidence_payload"]["status"] == "ok"
    assert detect_record["final_decision"]["decision_status"] == "abstain"
    assert detect_record["attestation"]["final_event_attested_decision"]["event_attestation_score"] == 0.61
    assert Path(str(event_manifest["payload_reference_sidecar_path"])).exists()
    assert Path(str(event_manifest["payload_decode_sidecar_path"])).exists()
    assert Path(str(event_manifest["payload_reference_sidecar_status_path"])).exists()
    assert Path(str(event_manifest["payload_decode_sidecar_status_path"])).exists()
    payload_reference_status = json.loads(
        Path(str(event_manifest["payload_reference_sidecar_status_path"])).read_text(encoding="utf-8")
    )
    payload_decode_status = json.loads(
        Path(str(event_manifest["payload_decode_sidecar_status_path"])).read_text(encoding="utf-8")
    )
    assert payload_reference_status["status"] == "ok"
    assert payload_decode_status["status"] == "ok"


def test_run_positive_source_event_fails_when_required_payload_reference_sidecar_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fail fast when the required PW01 payload reference sidecar cannot be built.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    bound_config_path, snapshot_dir = _write_bound_config_snapshot(tmp_path / "drive", marker="event_sidecar_failure")
    bound_cfg_obj = load_yaml_mapping(bound_config_path)
    _patch_pw01_base_runner(monkeypatch, expected_snapshot_path=snapshot_dir)

    def fake_build_payload_reference_sidecar_payload(**kwargs: Any) -> Dict[str, Any]:
        raise ValueError("embed_record missing lf_metrics required for payload reference sidecar")

    monkeypatch.setattr(
        pw01_module,
        "build_payload_reference_sidecar_payload",
        fake_build_payload_reference_sidecar_payload,
    )

    shard_root = ensure_directory(
        tmp_path / "drive" / "paper_workflow" / "families" / "family_event_sidecar_failure" / "source_shards" / "positive" / "shard_0000"
    )
    with pytest.raises(RuntimeError, match="PW01 required payload reference sidecar generation failed"):
        pw01_module._run_positive_source_event(
            event={
                "event_id": "evt_sidecar_failure_0000",
                "event_index": 0,
                "sample_role": "positive_source",
                "source_prompt_index": 0,
                "prompt_text": "prompt one",
                "prompt_sha256": "sha256-prompt-one",
                "seed": 3,
                "prompt_file": "prompts/paper_small.txt",
            },
            shard_root=shard_root,
            default_cfg_obj=bound_cfg_obj,
            bound_config_path=bound_config_path,
        )

    event_root = shard_root / "events" / "event_000000"
    payload_reference_status = json.loads(
        (event_root / "artifacts" / "payload_reference_sidecar_status.json").read_text(encoding="utf-8")
    )
    payload_decode_status = json.loads(
        (event_root / "artifacts" / "payload_decode_sidecar_status.json").read_text(encoding="utf-8")
    )
    assert payload_reference_status["status"] == "failed"
    assert payload_reference_status["required"] is True
    assert payload_reference_status["failure_reason"] == "payload_reference_sidecar_generation_failed"
    assert payload_decode_status["status"] == "failed"
    assert payload_decode_status["required"] is True
    assert payload_decode_status["failure_reason"] == "payload_reference_sidecar_generation_failed"


def test_run_positive_source_event_fails_before_preview_when_model_snapshot_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fail before preview when the runtime cfg loses a valid model snapshot path.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    preview_called = {"value": False}

    def fail_if_called(**kwargs: Any) -> Dict[str, Any]:
        preview_called["value"] = True
        raise AssertionError("preview should not run when model_snapshot_path is invalid")

    monkeypatch.setattr(
        pw01_module.BASE_RUNNER_MODULE,
        "_prepare_source_pool_preview_artifact",
        fail_if_called,
    )

    missing_snapshot_dir = tmp_path / "drive" / "runtime_state" / "missing_snapshot"
    broken_cfg = load_yaml_mapping((pw01_module.REPO_ROOT / "configs" / "default.yaml").resolve())
    broken_cfg["model_snapshot_path"] = missing_snapshot_dir.as_posix()
    broken_cfg["model_source_binding"] = {
        "binding_status": "bound",
        "binding_source": "notebook_snapshot_download",
        "model_snapshot_path": missing_snapshot_dir.as_posix(),
    }
    broken_bound_config_path = tmp_path / "drive" / "runtime_state" / "broken_bound_config.yaml"
    write_yaml_mapping(broken_bound_config_path, broken_cfg)

    with pytest.raises(RuntimeError, match="PW01 bound config missing valid model_snapshot_path before preview_precompute"):
        pw01_module._run_positive_source_event(
            event={
                "event_id": "evt_missing_snapshot_0000",
                "event_index": 0,
                "sample_role": "positive_source",
                "source_prompt_index": 0,
                "prompt_text": "prompt one",
                "prompt_sha256": "sha256-prompt-one",
                "seed": 3,
                "prompt_file": "prompts/paper_small.txt",
            },
            shard_root=ensure_directory(tmp_path / "drive" / "paper_workflow" / "families" / "family_missing_snapshot" / "source_shards" / "positive" / "shard_0000"),
            default_cfg_obj=broken_cfg,
            bound_config_path=broken_bound_config_path,
        )

    assert preview_called["value"] is False