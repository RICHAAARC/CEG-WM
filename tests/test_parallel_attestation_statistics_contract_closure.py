"""
文件目的：验证 01→02 parallel attestation statistics 的合同化收口。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import yaml

from main.cli import run_calibrate as run_calibrate_cli
from main.cli import run_evaluate as run_evaluate_cli
from main.evaluation import metrics as eval_metrics
from main.watermarking.detect import orchestrator as detect_orchestrator


def _load_script_module(relative_path: str, module_name: str) -> object:
    """
    功能：按路径加载 stage 脚本模块。

    Load a stage script module from a repository-relative path.

    Args:
        relative_path: Repository-relative script path.
        module_name: Temporary module name for importlib.

    Returns:
        Loaded module object.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    功能：写入 JSON 测试工件。

    Write a JSON payload for tests.

    Args:
        path_obj: Destination path.
        payload: JSON-serializable mapping.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_stage_01_writes_explicit_parallel_stats_contract_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 stage 01 会写出显式 parallel stats input contract，并随 package 一起导出。

    Validate that stage 01 emits an explicit parallel-stats input contract and
    exports it through stage packaging.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_01 = _load_script_module("scripts/01_Paper_Full_Cuda.py", "stage_01_contract_closure")
    drive_root = tmp_path / "drive"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_01"
    log_root = drive_root / "logs" / "stage_01"
    runtime_state_root = drive_root / "runtime_state" / "stage_01"
    export_root = drive_root / "exports" / "stage_01"

    monkeypatch.setattr(
        stage_01,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": run_root,
            "log_root": log_root,
            "runtime_state_root": runtime_state_root,
            "export_root": export_root,
        },
    )
    monkeypatch.setattr(stage_01, "load_yaml_mapping", lambda _path: {"attestation": {}})
    monkeypatch.setattr(stage_01, "detect_formal_gpu_preflight", lambda _path: {"ok": True})
    monkeypatch.setattr(
        stage_01,
        "copy_prompt_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_path": str(runtime_state_root / "runtime_metadata" / "prompt_snapshot" / "prompt.txt"),
            "source_path": "prompts/paper_small.txt",
        },
    )
    monkeypatch.setattr(stage_01, "collect_git_summary", lambda _root: {"commit": "test"})
    monkeypatch.setattr(stage_01, "collect_python_summary", lambda: {"version": "3.11"})
    monkeypatch.setattr(stage_01, "collect_cuda_summary", lambda: {"available": False})
    monkeypatch.setattr(stage_01, "collect_attestation_env_summary", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(stage_01, "collect_model_summary", lambda _cfg: {"model": "test"})
    monkeypatch.setattr(stage_01, "collect_weight_summary", lambda _root, _cfg: {"weights": []})

    def _fake_run_command_with_logs(**_kwargs: Any) -> Dict[str, Any]:
        _write_json(run_root / "records" / "embed_record.json", {"status": "ok"})
        _write_json(
            run_root / "records" / "detect_record.json",
            {
                "label": True,
                "attestation": {
                    "final_event_attested_decision": {
                        "event_attestation_score": 0.91,
                        "event_attestation_score_name": "event_attestation_score",
                    }
                },
            },
        )
        _write_json(run_root / "records" / "calibration_record.json", {"status": "ok"})
        _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
        _write_json(run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json", {"meta": True})
        _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "workflow_summary.json", {"status": "ok"})
        return {"return_code": 0}

    monkeypatch.setattr(stage_01, "run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(
        stage_01,
        "finalize_stage_package",
        lambda **_kwargs: {"package_path": str(export_root / "stage_01.zip"), "package_sha256": "sha256"},
    )

    stage_01.run_stage_01(
        drive_project_root=drive_root,
        config_path=config_path,
        notebook_name="01_Paper_Full_Cuda",
        stage_run_id="stage01_test",
    )

    contract_path = run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json"
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract_payload["artifact_type"] == "parallel_attestation_statistics_input_contract"
    assert contract_payload["score_name"] == "event_attestation_score"
    assert contract_payload["status"] == "unavailable"
    assert contract_payload["reason"] == "parallel_attestation_statistics_requires_label_balanced_detect_records"

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_input_contract_status"] == "unavailable"
    assert stage_manifest["parallel_attestation_statistics_input_contract_package_relative_path"] == (
        "artifacts/parallel_attestation_statistics_input_contract.json"
    )
    assert (runtime_state_root / "package_staging" / "artifacts" / "parallel_attestation_statistics_input_contract.json").exists()


def test_stage_02_only_consumes_contract_bound_detect_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 stage 02 只消费合同绑定的 detect records，而不是 source package glob。

    Validate that stage 02 consumes only contract-bound detect records rather
    than a broad source-package glob.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_contract_closure")
    drive_root = tmp_path / "drive"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("parallel_attestation_statistics:\n  enabled: true\n", encoding="utf-8")
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_02"
    log_root = drive_root / "logs" / "stage_02"
    runtime_state_root = drive_root / "runtime_state" / "stage_02"
    export_root = drive_root / "exports" / "stage_02"
    extracted_root = tmp_path / "source_package_extract"

    positive_record = {
        "label": True,
        "attestation": {
            "final_event_attested_decision": {
                "event_attestation_score": 0.92,
                "event_attestation_score_name": "event_attestation_score",
            }
        },
    }
    negative_record = {
        "label": False,
        "attestation": {
            "final_event_attested_decision": {
                "event_attestation_score": 0.0,
                "event_attestation_score_name": "event_attestation_score",
            }
        },
    }
    positive_path = extracted_root / "records" / "detect_positive.json"
    negative_path = extracted_root / "records" / "detect_negative.json"
    _write_json(positive_path, positive_record)
    _write_json(negative_path, negative_record)

    contract_payload = {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "status": "ready",
        "reason": "ok",
        "score_name": "event_attestation_score",
        "record_count": 2,
        "label_summary": {"positive": 1, "negative": 1, "unknown": 0, "label_balanced": True},
        "records": [
            {
                "package_relative_path": "records/detect_positive.json",
                "sha256": stage_02.compute_file_sha256(positive_path),
                "label": True,
            },
            {
                "package_relative_path": "records/detect_negative.json",
                "sha256": stage_02.compute_file_sha256(negative_path),
                "label": False,
            },
        ],
    }
    _write_json(extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json", contract_payload)
    source_stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "parallel_attestation_statistics_input_contract_package_relative_path": (
            "artifacts/parallel_attestation_statistics_input_contract.json"
        ),
    }
    _write_json(extracted_root / "artifacts" / "stage_manifest.json", source_stage_manifest)
    _write_json(extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
    (extracted_root / "runtime_metadata").mkdir(parents=True, exist_ok=True)
    (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text("calibration: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        stage_02,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": run_root,
            "log_root": log_root,
            "runtime_state_root": runtime_state_root,
            "export_root": export_root,
        },
    )
    monkeypatch.setattr(stage_02, "load_yaml_mapping", lambda _path: {"parallel_attestation_statistics": {"enabled": True}})
    monkeypatch.setattr(stage_02, "detect_formal_gpu_preflight", lambda _path: {"ok": True})
    monkeypatch.setattr(
        stage_02,
        "prepare_source_package",
        lambda _source_package_path, _runtime_state_root: {
            "stage_manifest": source_stage_manifest,
            "package_manifest": {"status": "ok"},
            "extracted_root": str(extracted_root),
            "source_package_path": str(source_package_path),
            "source_package_sha256": "source_sha256",
            "package_manifest_digest": "manifest_digest",
        },
    )
    monkeypatch.setattr(stage_02, "collect_git_summary", lambda _root: {"commit": "test"})
    monkeypatch.setattr(stage_02, "collect_python_summary", lambda: {"version": "3.11"})
    monkeypatch.setattr(stage_02, "collect_cuda_summary", lambda: {"available": False})
    monkeypatch.setattr(stage_02, "collect_attestation_env_summary", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(stage_02, "collect_model_summary", lambda _cfg: {"model": "test"})
    monkeypatch.setattr(stage_02, "collect_weight_summary", lambda _root, _cfg: {"weights": []})

    def _fake_stage_run(**kwargs: Any) -> Dict[str, Any]:
        stdout_path = kwargs["stdout_log_path"]
        stage_name = stdout_path.stem.replace("_stdout", "")
        _write_json(run_root / "records" / "calibration_record.json", {"status": "ok", "stage": stage_name})
        _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
        _write_json(run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json", {"meta": True})
        _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok", "stage": stage_name})
        _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        return {"return_code": 0}

    monkeypatch.setattr(stage_02, "run_command_with_logs", _fake_stage_run)
    monkeypatch.setattr(
        stage_02,
        "finalize_stage_package",
        lambda **_kwargs: {"package_path": str(export_root / "stage_02.zip"), "package_sha256": "sha256"},
    )

    stage_02.run_stage_02(
        drive_project_root=drive_root,
        config_path=config_path,
        source_package_path=source_package_path,
        notebook_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_test",
    )

    runtime_cfg = yaml.safe_load((runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml").read_text(encoding="utf-8"))
    detect_glob = runtime_cfg["calibration"]["detect_records_glob"]
    assert "contract_bound_detect_records" in detect_glob
    assert "*detect*.json" not in detect_glob
    staged_records_root = runtime_state_root / "contract_bound_detect_records"
    assert len(sorted(staged_records_root.glob("*.json"))) == 2


def test_stage_02_fails_fast_when_source_contract_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 stage 02 在 source contract 不可用时立即 fail-fast。

    Validate that stage 02 fails fast when the source contract is unavailable.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_contract_failure")
    drive_root = tmp_path / "drive"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("parallel_attestation_statistics:\n  enabled: true\n", encoding="utf-8")
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")
    extracted_root = tmp_path / "source_package_extract"
    _write_json(
        extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json",
        {
            "artifact_type": "parallel_attestation_statistics_input_contract",
            "status": "unavailable",
            "reason": "parallel_attestation_statistics_requires_label_balanced_detect_records",
            "score_name": "event_attestation_score",
            "label_summary": {"label_balanced": False},
            "records": [],
        },
    )
    source_stage_manifest = {
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "parallel_attestation_statistics_input_contract_package_relative_path": (
            "artifacts/parallel_attestation_statistics_input_contract.json"
        ),
    }
    _write_json(extracted_root / "artifacts" / "stage_manifest.json", source_stage_manifest)
    _write_json(extracted_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
    (extracted_root / "runtime_metadata").mkdir(parents=True, exist_ok=True)
    (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text("calibration: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        stage_02,
        "resolve_stage_roots",
        lambda *_args, **_kwargs: {
            "run_root": drive_root / "runs" / "stage_02",
            "log_root": drive_root / "logs" / "stage_02",
            "runtime_state_root": drive_root / "runtime_state" / "stage_02",
            "export_root": drive_root / "exports" / "stage_02",
        },
    )
    monkeypatch.setattr(stage_02, "load_yaml_mapping", lambda _path: {"parallel_attestation_statistics": {"enabled": True}})
    monkeypatch.setattr(
        stage_02,
        "prepare_source_package",
        lambda _source_package_path, _runtime_state_root: {
            "stage_manifest": source_stage_manifest,
            "package_manifest": {"status": "ok"},
            "extracted_root": str(extracted_root),
            "source_package_path": str(source_package_path),
            "source_package_sha256": "source_sha256",
            "package_manifest_digest": "manifest_digest",
        },
    )

    with pytest.raises(ValueError, match="parallel_attestation_statistics_input_contract is not ready"):
        stage_02.run_stage_02(
            drive_project_root=drive_root,
            config_path=config_path,
            source_package_path=source_package_path,
            notebook_name="02_Parallel_Attestation_Statistics",
            stage_run_id="stage02_failfast",
        )


def test_formal_calibrate_and_evaluate_do_not_call_synthetic_helper_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 formal calibrate/evaluate 默认不再调用 synthetic minimal ground-truth helper。

    Validate that formal calibrate/evaluate do not call the synthetic minimal
    ground-truth helper by default.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    def _fail_if_called(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("synthetic helper must not be called")

    monkeypatch.setattr(run_calibrate_cli, "ensure_minimal_ground_truth_records", _fail_if_called)
    monkeypatch.setattr(run_evaluate_cli, "ensure_minimal_ground_truth_records", _fail_if_called)

    calibration_result = run_calibrate_cli._ensure_calibration_detect_records_ready(
        {"calibration": {"detect_records_glob": "records/*.json", "allow_synthetic_minimal_ground_truth": False}},
        Path("run_root"),
    )
    evaluate_result = run_evaluate_cli._ensure_evaluate_detect_records_ready(
        {"evaluate": {"detect_records_glob": "records/*.json", "allow_synthetic_minimal_ground_truth": False}},
        Path("run_root"),
    )

    assert calibration_result["reason"] == "synthetic_minimal_ground_truth_disabled"
    assert evaluate_result["reason"] == "synthetic_minimal_ground_truth_disabled"


def test_metrics_and_detect_stats_reject_legacy_event_attestation_alias() -> None:
    """
    功能：验证 metrics 与 detect stats 读取端都拒绝 legacy event_attestation_statistics_score。

    Validate that both metrics and detect-statistics readers reject the legacy
    event_attestation_statistics_score alias.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="requires rerun"):
        eval_metrics._extract_score_value_for_metrics({}, "event_attestation_statistics_score")

    with pytest.raises(ValueError, match="requires rerun"):
        detect_orchestrator._extract_score_for_stats({}, "event_attestation_statistics_score")


def test_verify_attestation_no_longer_writes_legacy_alias_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 verify_attestation 输出的 final_event_attested_decision 不再写 legacy alias。

    Validate that verify_attestation no longer writes legacy alias fields into
    final_event_attested_decision.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import main.watermarking.provenance.attestation_statement as attestation_statement
    import main.watermarking.provenance.key_derivation as key_derivation
    import main.watermarking.content_chain.low_freq_coder as low_freq_coder
    import main.watermarking.content_chain.high_freq_embedder as high_freq_embedder

    monkeypatch.setattr(attestation_statement, "verify_statement_fields", lambda _statement: True)
    monkeypatch.setattr(
        attestation_statement,
        "statement_from_dict",
        lambda _statement: SimpleNamespace(plan_digest="plan_digest"),
    )
    monkeypatch.setattr(attestation_statement, "compute_attestation_digest", lambda _statement: "a" * 64)
    monkeypatch.setattr(
        attestation_statement,
        "verify_signed_attestation_bundle",
        lambda _bundle, _k_master: {"status": "ok"},
    )
    monkeypatch.setattr(
        key_derivation,
        "derive_attestation_keys",
        lambda *_args, **_kwargs: SimpleNamespace(k_lf="lf", k_hf="hf", event_binding_digest="b" * 64),
    )
    monkeypatch.setattr(
        low_freq_coder,
        "compute_lf_attestation_score",
        lambda **_kwargs: {"status": "ok", "lf_attestation_score": 0.9},
    )
    monkeypatch.setattr(
        high_freq_embedder,
        "compute_hf_attestation_score",
        lambda **_kwargs: {
            "status": "ok",
            "hf_attestation_score": 0.8,
            "hf_attestation_decision_score": 0.8,
            "hf_attestation_trace": {},
        },
    )

    verification = detect_orchestrator.verify_attestation(
        k_master="c" * 64,
        candidate_statement={"plan_digest": "plan_digest"},
        cfg={"attestation": {"use_trajectory_mix": False}},
        attestation_bundle={"trace_commit": "trace_commit"},
        hf_values=[0.1, 0.2, 0.3],
        lf_latent_features=[0.1, 0.2, 0.3],
        geo_score=0.4,
        attestation_decision_mode="content_primary_geo_rescue",
    )

    final_decision = verification["final_event_attested_decision"]
    assert final_decision["event_attestation_score_name"] == "event_attestation_score"
    assert "event_attestation_statistics_score" not in final_decision
    assert "event_attestation_statistics_score_name" not in final_decision
    assert "event_attestation_statistics_score_semantics" not in final_decision