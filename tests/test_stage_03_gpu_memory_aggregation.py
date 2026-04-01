"""
文件目的：验证 stage 03 GPU 峰值聚合、摘要透传与 package 落盘合同。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest

from main.evaluation import experiment_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_03_SCRIPT_PATH = REPO_ROOT / "scripts" / "03_Experiment_Matrix_Full.py"


def _load_stage_03_module() -> ModuleType:
    """
    功能：按文件路径加载 stage 03 脚本模块。

    Load the stage-03 wrapper module from its filesystem path.

    Args:
        None.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location("test_stage_03_gpu_memory_aggregation_module", STAGE_03_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {STAGE_03_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> Path:
    """
    功能：写出测试 JSON 工件。

    Write one JSON artifact used by the GPU aggregation tests.

    Args:
        path_obj: Destination JSON path.
        payload: JSON payload mapping.

    Returns:
        Written path.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path_obj


def _cuda_profile(phase_label: str, peak_memory_allocated_bytes: int, peak_memory_reserved_bytes: int) -> Dict[str, Any]:
    """
    功能：构造最小可聚合的 CUDA memory profile。

    Build the minimal CUDA memory profile payload consumed by stage-03 GPU aggregation.

    Args:
        phase_label: Stable runtime phase label.
        peak_memory_allocated_bytes: Peak allocated bytes.
        peak_memory_reserved_bytes: Peak reserved bytes.

    Returns:
        CUDA memory profile mapping.
    """
    return {
        "status": "ok",
        "reason": "ok",
        "phase_label": phase_label,
        "device": "cuda",
        "sample_scope": "single_worker_process_local",
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "peak_memory_allocated_mib": round(peak_memory_allocated_bytes / (1024.0 * 1024.0), 6),
        "peak_memory_reserved_mib": round(peak_memory_reserved_bytes / (1024.0 * 1024.0), 6),
    }


def _write_gpu_profile_run_artifacts(
    run_root: Path,
    *,
    preview_peak_bytes: int,
    embed_peak_bytes: int,
    runtime_capture_peak_bytes: int,
    detect_peak_bytes: int,
) -> None:
    """
    功能：为一个 child run 写出可复用的 GPU profile 工件。

    Write the reusable GPU-profile artifacts emitted by one child run.

    Args:
        run_root: Child run root directory.
        preview_peak_bytes: Preview-generation allocated peak.
        embed_peak_bytes: Embed main inference allocated peak.
        runtime_capture_peak_bytes: Statement-only runtime capture peak.
        detect_peak_bytes: Detect main inference allocated peak.

    Returns:
        None.
    """
    _write_json(
        run_root / "records" / "embed_record.json",
        {
            "inference_runtime_meta": {
                "cuda_memory_profile": _cuda_profile(
                    "embed_watermarked_inference",
                    embed_peak_bytes,
                    embed_peak_bytes + 128,
                )
            },
            "content_evidence": {
                "audit": {
                    "runtime_capture_cuda_memory_profile": _cuda_profile(
                        "statement_only_runtime_capture",
                        runtime_capture_peak_bytes,
                        runtime_capture_peak_bytes + 128,
                    )
                }
            },
        },
    )
    _write_json(
        run_root / "records" / "detect_record.json",
        {
            "inference_runtime_meta": {
                "cuda_memory_profile": _cuda_profile(
                    "detect_main_inference",
                    detect_peak_bytes,
                    detect_peak_bytes + 128,
                )
            }
        },
    )
    _write_json(
        run_root / "artifacts" / "preview_generation_record.json",
        {
            "inference_runtime_meta": {
                "cuda_memory_profile": _cuda_profile(
                    "preview_generation",
                    preview_peak_bytes,
                    preview_peak_bytes + 128,
                )
            }
        },
    )


def test_run_experiment_grid_writes_stage_03_gpu_memory_summary_and_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 matrix 必须聚合 experiment item 与 neg_cache 的 GPU 峰值并落盘 breakdown。

    Verify stage-03 experiment_matrix aggregates GPU peaks from both experiment-item
    runs and neg_cache child runs, then persists the stage-level summary and breakdown.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    batch_root = tmp_path / "matrix_batch"
    runtime_config_path = tmp_path / "runtime_config_snapshot.yaml"
    runtime_config_path.write_text("experiment_matrix: {}\n", encoding="utf-8")

    def _fake_derive_run_root(path_obj: Path) -> Path:
        return Path(path_obj)

    def _fake_ensure_output_layout(
        run_root: Path,
        **_kwargs: Any,
    ) -> Dict[str, Path]:
        records_dir = run_root / "records"
        artifacts_dir = run_root / "artifacts"
        logs_dir = run_root / "logs"
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "records_dir": records_dir,
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }

    monkeypatch.setattr(experiment_matrix.path_policy, "derive_run_root", _fake_derive_run_root)
    monkeypatch.setattr(experiment_matrix.path_policy, "ensure_output_layout", _fake_ensure_output_layout)
    monkeypatch.setattr(
        experiment_matrix.attack_coverage,
        "compute_attack_coverage_manifest",
        lambda: {"attack_coverage_digest": "coverage_digest_01"},
    )
    monkeypatch.setattr(experiment_matrix, "_run_global_calibrate", lambda *args, **kwargs: None)

    def _fake_neg_cache(**_kwargs: Any) -> Path:
        neg_run_root = batch_root / "neg_cache" / "neg_0001"
        _write_gpu_profile_run_artifacts(
            neg_run_root,
            preview_peak_bytes=384,
            embed_peak_bytes=448,
            runtime_capture_peak_bytes=512,
            detect_peak_bytes=768,
        )
        return neg_run_root / "records" / "detect_record.json"

    def _fake_run_single_experiment(grid_item_cfg: Dict[str, Any]) -> Dict[str, Any]:
        run_root = experiment_matrix._derive_run_root(grid_item_cfg)
        _write_gpu_profile_run_artifacts(
            run_root,
            preview_peak_bytes=256,
            embed_peak_bytes=320,
            runtime_capture_peak_bytes=448,
            detect_peak_bytes=640,
        )
        return {
            "grid_index": grid_item_cfg.get("grid_index"),
            "grid_item_digest": grid_item_cfg.get("grid_item_digest"),
            "run_root": run_root.as_posix(),
            "model_id": grid_item_cfg.get("model_id"),
            "seed": grid_item_cfg.get("seed"),
            "attack_family": grid_item_cfg.get("attack_protocol_family"),
            "evaluation_scope": "system_final",
            "auxiliary_scopes": ["content_chain", "lf_channel"],
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
            "status": "ok",
            "failure_reason": "ok",
            "cfg_digest": "cfg_digest_01",
            "plan_digest": "plan_digest_01",
            "thresholds_digest": "thresholds_digest_01",
            "threshold_metadata_digest": "threshold_metadata_digest_01",
            "ablation_digest": "ablation_digest_01",
            "attack_protocol_digest": "attack_protocol_digest_01",
            "attack_protocol_version": "attack_protocol_version_01",
            "policy_path": "content_np_geo_rescue",
            "impl_digest": "impl_digest_01",
            "fusion_rule_version": "fusion_rule_v1",
            "hf_truncation_baseline_comparison": {
                "content_score": 0.9,
                "hf_truncation_score": 0.7,
                "score_delta_content_minus_hf_truncation": 0.2,
                "comparison_ready": True,
                "comparison_source": "real_hf_truncation_baseline_required",
            },
            "detect_gate_relaxed": False,
            "detect_gate_relax_reason": "hard_gate_satisfied",
            "detect_gate_sample_counts": {},
            "auxiliary_analysis_runtime_executed": False,
            "auxiliary_analysis": {
                "driver_role": "auxiliary_only",
                "metric_name": "lf_channel_score",
                "status": "skipped",
                "failure_reason": "auxiliary_analysis_not_requested",
                "shared_thresholds_used": False,
                "pair_free_evaluate_used": False,
                "auxiliary_analysis_runtime_executed": False,
            },
            "metrics": {
                "system_final_metrics": {
                    "scope": "system_final",
                    "n_total": 2,
                    "n_positive": 1,
                    "n_negative": 1,
                },
                "auxiliary_scope_metrics": {},
            },
        }

    monkeypatch.setattr(experiment_matrix, "_run_neg_embed_detect_for_cache", _fake_neg_cache)
    monkeypatch.setattr(experiment_matrix, "run_single_experiment", _fake_run_single_experiment)

    grid_item = experiment_matrix.build_experiment_grid(
        {
            "policy_path": "content_np_geo_rescue",
            "model_id": "sd3",
            "seed": 0,
            "experiment_matrix": {
                "models": ["sd3"],
                "seeds": [0],
                "attack_protocol_families": ["rotate"],
                "batch_root": str(batch_root),
                "config_path": str(runtime_config_path),
                "primary_scope": "system_final",
                "primary_summary_basis_scope": "system_final",
                "enable_auxiliary_analysis_runtime": False,
            },
        }
    )

    grid_summary = experiment_matrix.run_experiment_grid(grid_item, strict=True)

    aggregate_report_path = Path(str(grid_summary["aggregate_report_path"]))
    grid_summary_path = Path(str(grid_summary["grid_summary_path"]))
    breakdown_path = Path(str(grid_summary["gpu_memory_profile_breakdown_path"]))
    aggregate_report = json.loads(aggregate_report_path.read_text(encoding="utf-8"))
    persisted_grid_summary = json.loads(grid_summary_path.read_text(encoding="utf-8"))
    breakdown_payload = json.loads(breakdown_path.read_text(encoding="utf-8"))

    assert aggregate_report["gpu_memory_summary"]["status"] == "ok"
    assert aggregate_report["gpu_memory_summary"]["scanned_run_count"] == 2
    assert aggregate_report["gpu_memory_summary"]["experiment_item_run_count"] == 1
    assert aggregate_report["gpu_memory_summary"]["neg_cache_run_count"] == 1
    assert aggregate_report["gpu_memory_summary"]["peak_memory_allocated_bytes_max"] == 768
    assert persisted_grid_summary["gpu_memory_summary"]["peak_memory_reserved_bytes_max"] == 896
    assert breakdown_payload["gpu_memory_summary"]["peak_memory_allocated_bytes_max"] == 768
    assert {entry["run_kind"] for entry in breakdown_payload["entries"]} == {"experiment_item", "neg_cache"}
    assert {entry["profile_role"] for entry in breakdown_payload["entries"]} == {
        "preview_generation",
        "embed_watermarked_inference",
        "statement_only_runtime_capture",
        "detect_main_inference",
    }


def test_stage_03_script_syncs_gpu_memory_summary_and_packages_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 脚本必须同步 GPU 摘要并把 breakdown 工件打进 formal package。

    Verify the stage-03 wrapper synchronizes the GPU memory summary into its
    stage artifacts and includes the breakdown artifact inside the final package.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_03_module = _load_stage_03_module()

    drive_project_root = tmp_path / "drive"
    drive_project_root.mkdir(parents=True, exist_ok=True)
    source_package_path = tmp_path / "source_stage01.zip"
    source_package_path.write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "default.yaml"
    config_path.write_text("experiment_matrix: {}\n", encoding="utf-8")
    stage_roots = {
        "run_root": drive_project_root / "runs" / "stage03_gpu_sync_test",
        "log_root": drive_project_root / "logs" / "stage03_gpu_sync_test",
        "runtime_state_root": drive_project_root / "runtime_state" / "stage03_gpu_sync_test",
        "export_root": drive_project_root / "exports" / "stage03_gpu_sync_test",
    }

    def _fake_prepare_source_package(_source_package_path: Path, runtime_state_root: Path) -> Dict[str, Any]:
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
        (extracted_root / "runtime_metadata" / "runtime_config_snapshot.yaml").write_text(
            "experiment_matrix: {}\n",
            encoding="utf-8",
        )
        return {
            "stage_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
            "package_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01_run"},
            "extracted_root": extracted_root,
            "source_package_path": source_package_path,
            "source_package_sha256": "sha256_stage01",
            "package_manifest_digest": "digest_stage01",
        }

    def _fake_run_command_with_logs(**_kwargs: Any) -> Dict[str, Any]:
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
            "profile_role_counts": {"detect_main_inference": 1},
            "phase_label_counts": {"detect_main_inference": 1},
            "device_counts": {"cuda": 4},
            "peak_memory_allocated_bytes_max": 4096,
            "peak_memory_reserved_bytes_max": 6144,
            "peak_memory_allocated_mib_max": round(4096 / (1024.0 * 1024.0), 6),
            "peak_memory_reserved_mib_max": round(6144 / (1024.0 * 1024.0), 6),
            "peak_memory_allocated_max_entry": {"profile_role": "detect_main_inference"},
            "peak_memory_reserved_max_entry": {"profile_role": "detect_main_inference"},
        }
        stage_03_module.write_json_atomic(
            artifacts_dir / "grid_summary.json",
            {
                "primary_evaluation_scope": "system_final",
                "primary_metric_name": "system_final_metrics",
                "primary_driver_mode": "system_final_only",
                "primary_status_source": "system_final_metrics",
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": "system_final_metrics",
                "auxiliary_scopes": ["content_chain", "lf_channel"],
                "auxiliary_analysis_runtime_executed": False,
                "gpu_memory_summary": gpu_memory_summary,
                "scope_manifest": {
                    "primary_scope": "system_final",
                    "primary_metric_name": "system_final_metrics",
                    "primary_summary_basis_scope": "system_final",
                    "primary_summary_basis_metric_name": "system_final_metrics",
                    "auxiliary_scopes": ["content_chain", "lf_channel"],
                },
                "system_final_metrics_presence": {
                    "rows_with_system_final_metrics": 1,
                    "ok_rows_with_system_final_metrics": 1,
                    "rows_total": 1,
                },
            },
        )
        stage_03_module.write_json_atomic(artifacts_dir / "grid_manifest.json", {"grid_manifest_digest": "grid01"})
        stage_03_module.write_json_atomic(
            artifacts_dir / "aggregate_report.json",
            {
                "primary_evaluation_scope": "system_final",
                "primary_metric_name": "system_final_metrics",
                "primary_driver_mode": "system_final_only",
                "primary_status_source": "system_final_metrics",
                "primary_summary_basis_scope": "system_final",
                "primary_summary_basis_metric_name": "system_final_metrics",
                "auxiliary_scopes": ["content_chain", "lf_channel"],
                "auxiliary_analysis_runtime_executed": False,
                "gpu_memory_summary": gpu_memory_summary,
                "scope_manifest": {
                    "primary_scope": "system_final",
                    "primary_metric_name": "system_final_metrics",
                    "primary_summary_basis_scope": "system_final",
                    "primary_summary_basis_metric_name": "system_final_metrics",
                    "auxiliary_scopes": ["content_chain", "lf_channel"],
                },
                "system_final_metrics_presence": {
                    "rows_with_system_final_metrics": 1,
                    "ok_rows_with_system_final_metrics": 1,
                    "rows_total": 1,
                },
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

    monkeypatch.setattr(stage_03_module, "prepare_source_package", _fake_prepare_source_package)
    monkeypatch.setattr(stage_03_module, "resolve_stage_roots", lambda *args, **kwargs: stage_roots)
    monkeypatch.setattr(stage_03_module, "load_yaml_mapping", lambda _path: {"experiment_matrix": {}})
    monkeypatch.setattr(stage_03_module, "detect_stage_03_preflight", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(stage_03_module, "run_command_with_logs", _fake_run_command_with_logs)
    monkeypatch.setattr(stage_03_module, "collect_git_summary", lambda _path: {})
    monkeypatch.setattr(stage_03_module, "collect_python_summary", lambda: {})
    monkeypatch.setattr(stage_03_module, "collect_cuda_summary", lambda: {})
    monkeypatch.setattr(stage_03_module, "collect_attestation_env_summary", lambda _cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_model_summary", lambda _cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_weight_summary", lambda _repo_root, _cfg: {})
    monkeypatch.setattr(stage_03_module, "collect_file_index", lambda _root, _mapping: {})
    monkeypatch.setattr(
        stage_03_module,
        "ensure_attestation_env_bootstrap",
        lambda *_args, **_kwargs: {"status": "disabled", "required_env_vars": [], "missing_env_vars": []},
    )

    summary = stage_03_module.run_stage_03(
        drive_project_root=drive_project_root,
        config_path=config_path,
        source_package_path=source_package_path,
        notebook_name="03_Experiment_Matrix_Full",
        stage_run_id="stage03_gpu_sync_test",
    )

    workflow_summary = json.loads((stage_roots["run_root"] / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))
    stage_manifest = json.loads((stage_roots["run_root"] / "artifacts" / "stage_manifest.json").read_text(encoding="utf-8"))
    package_path = Path(str(summary["package_path"]))
    with zipfile.ZipFile(package_path, "r") as archive:
        archived_names = set(archive.namelist())

    assert workflow_summary["gpu_memory_summary"]["status"] == "ok"
    assert workflow_summary["gpu_memory_summary"]["peak_memory_allocated_bytes_max"] == 4096
    assert stage_manifest["gpu_memory_summary"]["peak_memory_reserved_bytes_max"] == 6144
    assert workflow_summary["gpu_memory_profile_breakdown_path"].endswith("gpu_memory_profile_breakdown.json")
    assert stage_manifest["gpu_memory_profile_breakdown_path"].endswith("gpu_memory_profile_breakdown.json")
    assert "artifacts/gpu_memory_profile_breakdown.json" in archived_names