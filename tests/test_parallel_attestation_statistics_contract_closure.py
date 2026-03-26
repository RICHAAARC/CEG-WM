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


def _make_detect_record(label: bool, score: float) -> Dict[str, Any]:
    """
    功能：构造最小 event_attestation detect record。

    Build a minimal detect record carrying canonical event-attestation fields.

    Args:
        label: Ground-truth label.
        score: Event attestation score.

    Returns:
        Detect record mapping.
    """
    return {
        "label": label,
        "ground_truth": label,
        "is_watermarked": label,
        "attestation": {
            "final_event_attested_decision": {
                "event_attestation_score": score,
                "event_attestation_score_name": "event_attestation_score",
                "is_event_attested": label,
            }
        },
    }


def _make_source_contract(records: list[Dict[str, Any]], *, direct_stats_ready: bool, direct_stats_reason: str) -> Dict[str, Any]:
    """
    功能：构造最小 source contract 测试载荷。

    Build a minimal stage-01 source contract payload for stage-02 tests.

    Args:
        records: Source record entries.
        direct_stats_ready: Whether the source is directly stats-ready.
        direct_stats_reason: Direct stats readiness reason.

    Returns:
        Source contract mapping.
    """
    positive = sum(1 for entry in records if entry["label"] is True)
    negative = sum(1 for entry in records if entry["label"] is False)
    return {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "source_stage",
        "status": "ok",
        "reason": "parallel_attestation_statistics_source_records_available",
        "score_name": "event_attestation_score",
        "source_records_available": True,
        "record_count": len(records),
        "label_summary": {
            "positive": positive,
            "negative": negative,
            "unknown": 0,
            "label_balanced": positive > 0 and negative > 0,
        },
        "direct_stats_ready": direct_stats_ready,
        "direct_stats_reason": direct_stats_reason,
        "records": records,
    }


def _make_stage_01_source_pool_contract(run_root: Path, prompt_count: int) -> Dict[str, Any]:
    """
    功能：构造 stage 01 direct source pool contract 测试载荷。

    Build a stage-01 direct source pool contract payload for shell tests.

    Args:
        run_root: Stage-01 run root.
        prompt_count: Number of source prompts.

    Returns:
        Source contract mapping.
    """
    records: list[Dict[str, Any]] = []
    for prompt_index in range(prompt_count):
        record_path = run_root / "artifacts" / "stage_01_source_pool_detect_records" / f"{prompt_index:03d}_detect_record.json"
        _write_json(
            record_path,
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {
                    "status": "ok",
                    "content_chain_score": 0.9 - prompt_index * 1e-3,
                },
                "attestation": {
                    "final_event_attested_decision": {
                        "event_attestation_score": 0.95 - prompt_index * 1e-3,
                        "event_attestation_score_name": "event_attestation_score",
                        "is_event_attested": True,
                    }
                },
            },
        )
        records.append(
            {
                "record_role": "direct_source_record",
                "usage": "stage_01_direct_source_pool",
                "package_relative_path": (
                    f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
                ),
                "path": str(record_path),
                "sha256": f"detect_sha_{prompt_index:03d}",
                "label": True,
                "prompt_index": prompt_index,
                "prompt_text": f"prompt {prompt_index}",
                "prompt_sha256": f"prompt_sha_{prompt_index:03d}",
                "prompt_file": "prompts/paper_small.txt",
                "score_name": "event_attestation_score",
                "score_available": True,
                "event_attestation_score_available": True,
                "threshold_score_name": "content_chain_score",
                "threshold_score_available": True,
                "content_chain_score_available": True,
            }
        )

    return {
        "artifact_type": "parallel_attestation_statistics_input_contract",
        "contract_role": "source_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_test",
        "status": "ok",
        "reason": "stage_01_direct_source_pool_ready",
        "score_name": "event_attestation_score",
        "threshold_score_name": "content_chain_score",
        "source_records_available": True,
        "record_count": prompt_count,
        "label_summary": {
            "positive": prompt_count,
            "negative": 0,
            "unknown": 0,
            "label_balanced": False,
        },
        "score_availability": {
            "content_chain_score": {
                "available_record_count": prompt_count,
                "missing_record_count": 0,
            },
            "event_attestation_score": {
                "available_record_count": prompt_count,
                "missing_record_count": 0,
            },
        },
        "direct_stats_ready": False,
        "direct_stats_reason": "parallel_attestation_statistics_requires_label_balanced_detect_records",
        "records": records,
    }


def _make_stage_01_pooled_threshold_contract(run_root: Path, prompt_count: int) -> Dict[str, Any]:
    """
    功能：构造 stage 01 pooled threshold build contract 测试载荷。

    Build a stage-01 pooled threshold build contract payload for shell tests.

    Args:
        run_root: Stage-01 run root.
        prompt_count: Number of direct source prompts.

    Returns:
        Build contract mapping.
    """
    records: list[Dict[str, Any]] = []
    direct_records: list[Dict[str, Any]] = []
    derived_records: list[Dict[str, Any]] = []
    pooled_root = run_root / "artifacts" / "stage_01_pooled_threshold_records"
    for prompt_index in range(prompt_count):
        direct_path = pooled_root / f"{prompt_index:03d}_direct_positive.json"
        _write_json(direct_path, {"label": True, "content_evidence_payload": {"status": "ok", "content_chain_score": 0.8}})
        direct_record = {
            "record_kind": "direct",
            "label": True,
            "usage": "stage_01_pooled_thresholds",
            "derived_from": None,
            "derivation_kind": None,
            "source_package_relative_path": (
                f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
            ),
            "staged_path": str(direct_path),
            "package_relative_path": f"artifacts/stage_01_pooled_threshold_records/{prompt_index:03d}_direct_positive.json",
            "sha256": f"pooled_direct_sha_{prompt_index:03d}",
            "prompt_file": "prompts/paper_small.txt",
            "prompt_index": prompt_index,
            "prompt_text": f"prompt {prompt_index}",
        }
        direct_records.append(direct_record)
        records.append(direct_record)

    for prompt_index in range(prompt_count):
        derived_path = pooled_root / f"{prompt_count + prompt_index:03d}_derived_negative.json"
        _write_json(derived_path, {"label": False, "content_evidence_payload": {"status": "ok", "content_chain_score": -0.2}})
        derived_record = {
            "record_kind": "derived",
            "label": False,
            "usage": "stage_01_pooled_thresholds",
            "derived_from": f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json",
            "derivation_kind": "prompt_bound_label_balance",
            "source_package_relative_path": (
                f"artifacts/stage_01_source_pool_detect_records/{prompt_index:03d}_detect_record.json"
            ),
            "staged_path": str(derived_path),
            "package_relative_path": (
                f"artifacts/stage_01_pooled_threshold_records/{prompt_count + prompt_index:03d}_derived_negative.json"
            ),
            "sha256": f"pooled_derived_sha_{prompt_index:03d}",
            "prompt_file": "prompts/paper_small.txt",
            "prompt_index": prompt_index,
            "prompt_text": f"prompt {prompt_index}",
        }
        derived_records.append(derived_record)
        records.append(derived_record)

    thresholds_path = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    threshold_metadata_path = run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json"
    return {
        "artifact_type": "stage_01_pooled_threshold_build_contract",
        "contract_role": "pooled_threshold_build_contract",
        "contract_version": "v1",
        "stage_name": "01_Paper_Full_Cuda",
        "stage_run_id": "stage01_test",
        "requested_build_mode": "source_plus_derived_pairs",
        "build_mode": "source_plus_derived_pairs",
        "score_name": "content_chain_score",
        "prompt_file": "prompts/paper_small.txt",
        "prompt_pool_summary": {
            "prompt_count": prompt_count,
            "prompt_indices": list(range(prompt_count)),
        },
        "staged_records_root": str(pooled_root),
        "detect_records_glob": str(pooled_root / "*.json"),
        "direct_record_count": prompt_count,
        "derived_record_count": prompt_count,
        "final_record_count": prompt_count * 2,
        "direct_summary": {"positive": prompt_count, "negative": 0},
        "derived_summary": {"positive": 0, "negative": prompt_count},
        "final_positive_count": prompt_count,
        "final_negative_count": prompt_count,
        "final_label_balanced": True,
        "build_configuration": {
            "target_pair_count": prompt_count,
            "build_usage": "stage_01_pooled_thresholds",
            "record_derivation_kind": "prompt_bound_label_balance",
        },
        "stats_input_set": {
            "score_name": "content_chain_score",
            "detect_records_glob": str(pooled_root / "*.json"),
            "direct_record_count": prompt_count,
            "derived_record_count": prompt_count,
            "final_record_count": prompt_count * 2,
            "positive_record_count": prompt_count,
            "negative_record_count": prompt_count,
            "thresholds_artifact_path": str(thresholds_path),
            "threshold_metadata_artifact_path": str(threshold_metadata_path),
        },
        "records": records,
        "direct_records": direct_records,
        "derived_records": derived_records,
        "thresholds_artifact_path": str(thresholds_path),
        "threshold_metadata_artifact_path": str(threshold_metadata_path),
    }


def _prepare_stage_02_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    stage_02: Any,
    drive_root: Path,
    run_root: Path,
    log_root: Path,
    runtime_state_root: Path,
    export_root: Path,
    extracted_root: Path,
    source_stage_manifest: Dict[str, Any],
    source_package_path: Path,
    config_payload: Dict[str, Any],
) -> None:
    """
    功能：统一设置 stage 02 测试所需 monkeypatch。

    Apply the common monkeypatch setup used by stage-02 contract tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        stage_02: Loaded stage-02 script module.
        drive_root: Temporary drive root.
        run_root: Stage run root.
        log_root: Stage log root.
        runtime_state_root: Stage runtime-state root.
        export_root: Stage export root.
        extracted_root: Prepared extracted source package root.
        source_stage_manifest: Source stage manifest payload.
        source_package_path: Placeholder ZIP path.
        config_payload: Runtime config payload returned by load_yaml_mapping.

    Returns:
        None.
    """
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
    monkeypatch.setattr(stage_02, "load_yaml_mapping", lambda _path: config_payload)
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


def test_stage_01_writes_source_contract_even_when_direct_stats_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证 stage 01 会写出 source contract，即使 direct stats 尚未 ready。

    Validate that stage 01 emits a source contract artifact even when the source
    detect output is not directly stats-ready.

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
        prompt_count = 16
        _write_json(run_root / "records" / "embed_record.json", {"status": "ok"})
        _write_json(run_root / "records" / "detect_record.json", _make_detect_record(True, 0.91))
        _write_json(run_root / "records" / "calibration_record.json", {"status": "ok"})
        _write_json(run_root / "records" / "evaluate_record.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json", {"threshold": 0.5})
        _write_json(run_root / "artifacts" / "thresholds" / "threshold_metadata_artifact.json", {"meta": True})
        _write_json(run_root / "artifacts" / "evaluation_report.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "run_closure.json", {"status": "ok"})
        _write_json(run_root / "artifacts" / "workflow_summary.json", {"status": "ok"})
        _write_json(
            run_root / "artifacts" / "parallel_attestation_statistics_input_contract.json",
            _make_stage_01_source_pool_contract(run_root, prompt_count),
        )
        _write_json(
            run_root / "artifacts" / "stage_01_pooled_threshold_build_contract.json",
            _make_stage_01_pooled_threshold_contract(run_root, prompt_count),
        )
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
    assert contract_payload["contract_role"] == "source_contract"
    assert contract_payload["status"] == "ok"
    assert contract_payload["source_records_available"] is True
    assert contract_payload["direct_stats_ready"] is False
    assert contract_payload["record_count"] == 16
    assert contract_payload["direct_stats_reason"] == "parallel_attestation_statistics_requires_label_balanced_detect_records"

    pooled_build_contract = json.loads(
        (run_root / "artifacts" / "stage_01_pooled_threshold_build_contract.json").read_text(encoding="utf-8")
    )
    assert pooled_build_contract["build_mode"] == "source_plus_derived_pairs"
    assert pooled_build_contract["direct_record_count"] == 16
    assert pooled_build_contract["derived_record_count"] == 16
    assert pooled_build_contract["final_record_count"] == 32
    assert pooled_build_contract["final_label_balanced"] is True

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_input_contract_status"] == "ok"
    assert stage_manifest["parallel_attestation_statistics_input_contract_source_records_available"] is True
    assert stage_manifest["parallel_attestation_statistics_input_contract_direct_stats_ready"] is False
    assert stage_manifest["parallel_attestation_statistics_input_contract_record_count"] == 16
    assert stage_manifest["parallel_attestation_statistics_input_contract_package_relative_path"] == (
        "artifacts/parallel_attestation_statistics_input_contract.json"
    )
    assert stage_manifest["stage_01_pooled_threshold_build_mode"] == "source_plus_derived_pairs"
    assert stage_manifest["stage_01_pooled_threshold_direct_record_count"] == 16
    assert stage_manifest["stage_01_pooled_threshold_derived_record_count"] == 16
    assert stage_manifest["stage_01_pooled_threshold_final_record_count"] == 32
    assert stage_manifest["stage_01_pooled_threshold_final_label_balanced"] is True
    assert (runtime_state_root / "package_staging" / "artifacts" / "parallel_attestation_statistics_input_contract.json").exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_source_pool_detect_records" / "015_detect_record.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_pooled_threshold_records" / "031_derived_negative.json"
    ).exists()
    assert (
        runtime_state_root / "package_staging" / "artifacts" / "stage_01_pooled_threshold_build_contract.json"
    ).exists()


def test_default_config_enables_stage_01_pool_and_stage_02_target_pair_count() -> None:
    """
    功能：验证默认配置显式启用 stage 01 prompt pool 和 stage 02 的 16 对统计目标。

    Validate that the default config enables the stage-01 prompt pool and sets
    the stage-02 target pair count to 16.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config_payload = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    assert config_payload["inference_prompt_file"] == "prompts/paper_small.txt"
    assert config_payload["stage_01_source_pool"]["enabled"] is True
    assert config_payload["stage_01_source_pool"]["use_inference_prompt_file"] is True
    assert config_payload["stage_01_source_pool"]["target_prompt_count"] == 16
    assert config_payload["stage_01_pooled_threshold_build"]["enabled"] is True
    assert config_payload["stage_01_pooled_threshold_build"]["build_mode"] == "source_plus_derived_pairs"
    assert config_payload["stage_01_pooled_threshold_build"]["target_pair_count"] == 16
    assert config_payload["parallel_attestation_statistics"]["target_pair_count"] == 16


def test_stage_02_direct_only_build_uses_source_records_and_writes_build_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证当 source 已平衡时，stage 02 直接消费 source records 并写出 direct-only build contract。

    Validate that stage 02 consumes balanced source records directly and emits a
    direct-only build contract without derived records.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_direct_only")
    drive_root = tmp_path / "drive"
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_02"
    log_root = drive_root / "logs" / "stage_02"
    runtime_state_root = drive_root / "runtime_state" / "stage_02"
    export_root = drive_root / "exports" / "stage_02"
    extracted_root = tmp_path / "source_package_extract_direct"

    positive_path = extracted_root / "records" / "detect_positive.json"
    negative_path = extracted_root / "records" / "detect_negative.json"
    _write_json(positive_path, _make_detect_record(True, 0.92))
    _write_json(negative_path, _make_detect_record(False, 0.0))

    source_contract = _make_source_contract(
        [
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
        direct_stats_ready=True,
        direct_stats_reason="ok",
    )
    _write_json(extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json", source_contract)
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

    _prepare_stage_02_monkeypatches(
        monkeypatch,
        stage_02,
        drive_root,
        run_root,
        log_root,
        runtime_state_root,
        export_root,
        extracted_root,
        source_stage_manifest,
        source_package_path,
        {
            "parallel_attestation_statistics": {
                "enabled": True,
                "build_mode": "direct_source_only",
                "allow_derived_input_build": False,
                "target_pair_count": 1,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )

    stage_02.run_stage_02(
        drive_project_root=drive_root,
        config_path=tmp_path / "config.yaml",
        source_package_path=source_package_path,
        notebook_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_direct_only",
    )

    runtime_cfg = yaml.safe_load((runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml").read_text(encoding="utf-8"))
    detect_glob = runtime_cfg["calibration"]["detect_records_glob"]
    assert "parallel_attestation_statistics_detect_records" in detect_glob
    assert "*detect*.json" not in detect_glob

    build_contract_path = run_root / "artifacts" / "parallel_attestation_statistics_build_contract.json"
    build_contract = json.loads(build_contract_path.read_text(encoding="utf-8"))
    assert build_contract["build_mode"] == "direct_source_only"
    assert build_contract["direct_record_count"] == 2
    assert build_contract["derived_record_count"] == 0
    assert build_contract["final_label_balanced"] is True

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_direct_only"] is True
    assert stage_manifest["parallel_attestation_statistics_build_mode"] == "direct_source_only"
    assert stage_manifest["parallel_attestation_statistics_derived_record_count"] == 0
    assert (runtime_state_root / "package_staging" / "artifacts" / "parallel_attestation_statistics_build_contract.json").exists()


def test_stage_02_derived_build_records_direct_and_derived_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证当 source 不平衡但允许 derived build 时，stage 02 会显式构建 derived records 并写出 build contract。

    Validate that stage 02 explicitly builds derived records when the source is
    unbalanced and the build config allows source-plus-derived construction.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_02 = _load_script_module("scripts/02_Parallel_Attestation_Statistics.py", "stage_02_derived_build")
    drive_root = tmp_path / "drive"
    source_package_path = tmp_path / "stage_01_package.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")

    run_root = drive_root / "runs" / "stage_02"
    log_root = drive_root / "logs" / "stage_02"
    runtime_state_root = drive_root / "runtime_state" / "stage_02"
    export_root = drive_root / "exports" / "stage_02"
    extracted_root = tmp_path / "source_package_extract_derived"
    prompt_file = tmp_path / "paper_small_prompts.txt"
    prompt_file.write_text("a cat on a sofa\n", encoding="utf-8")

    positive_path = extracted_root / "records" / "detect_positive.json"
    _write_json(positive_path, _make_detect_record(True, 0.92))

    source_contract = _make_source_contract(
        [
            {
                "package_relative_path": "records/detect_positive.json",
                "sha256": stage_02.compute_file_sha256(positive_path),
                "label": True,
            }
        ],
        direct_stats_ready=False,
        direct_stats_reason="parallel_attestation_statistics_requires_label_balanced_detect_records",
    )
    _write_json(extracted_root / "artifacts" / "parallel_attestation_statistics_input_contract.json", source_contract)
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

    _prepare_stage_02_monkeypatches(
        monkeypatch,
        stage_02,
        drive_root,
        run_root,
        log_root,
        runtime_state_root,
        export_root,
        extracted_root,
        source_stage_manifest,
        source_package_path,
        {
            "parallel_attestation_statistics": {
                "enabled": True,
                "build_mode": "source_plus_derived_pairs",
                "allow_derived_input_build": True,
                "derived_input_prompts_file": str(prompt_file),
                "target_pair_count": 1,
                "build_usage": "parallel_attestation_statistics",
                "record_derivation_kind": "prompt_bound_label_balance",
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )

    stage_02.run_stage_02(
        drive_project_root=drive_root,
        config_path=tmp_path / "config.yaml",
        source_package_path=source_package_path,
        notebook_name="02_Parallel_Attestation_Statistics",
        stage_run_id="stage02_derived_build",
    )

    build_contract_path = run_root / "artifacts" / "parallel_attestation_statistics_build_contract.json"
    build_contract = json.loads(build_contract_path.read_text(encoding="utf-8"))
    assert build_contract["build_mode"] == "source_plus_derived_pairs"
    assert build_contract["direct_record_count"] == 1
    assert build_contract["derived_record_count"] == 1
    assert build_contract["final_label_balanced"] is True
    assert build_contract["direct_only"] is False

    derived_record = build_contract["derived_records"][0]
    assert derived_record["derived_from"] == "records/detect_positive.json"
    assert derived_record["derivation_kind"] == "prompt_bound_label_balance"
    assert derived_record["label"] is False
    assert derived_record["usage"] == "parallel_attestation_statistics"
    assert derived_record["prompt_file"] == str(prompt_file.resolve()).replace("\\", "/")

    derived_payload_path = Path(derived_record["staged_path"])
    derived_payload = json.loads(derived_payload_path.read_text(encoding="utf-8"))
    assert derived_payload["label"] is False
    assert derived_payload["parallel_attestation_statistics_build"]["derivation_kind"] == "prompt_bound_label_balance"
    assert derived_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == 0.0

    runtime_cfg = yaml.safe_load((runtime_state_root / "runtime_metadata" / "runtime_config_snapshot.yaml").read_text(encoding="utf-8"))
    assert "parallel_attestation_statistics_detect_records" in runtime_cfg["calibration"]["detect_records_glob"]
    assert runtime_cfg["calibration"]["detect_records_glob"] == runtime_cfg["evaluate"]["detect_records_glob"]

    stage_manifest = json.loads((run_root / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["parallel_attestation_statistics_build_mode"] == "source_plus_derived_pairs"
    assert stage_manifest["parallel_attestation_statistics_derived_record_count"] == 1
    assert stage_manifest["parallel_attestation_statistics_direct_only"] is False


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